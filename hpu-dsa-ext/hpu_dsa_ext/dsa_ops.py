"""HPU DSA — load + drive the two VALIDATED DSA OpBackend kernels (MVP-3).

This is the integration shim between the vLLM MLA serving path and the two captured
DSA ops that live (built) in the hpu-moe-ext repo:

    torch.ops.hpu_dsa_ext.indexer_topk(q, k, weights, cmask, index_topk) -> [Sq, kk] int32
        HW-validated lightning-indexer top-k (Jaccard 1.0, libdsa_indexer{,_lazy}.so).
        q       [H, Sq, D]  bf16   per-head index queries (rope applied UPSTREAM)
        k       [H, Sk, D]  bf16   MQA index key EXPANDED to H heads
        weights [H, Sq, 1]  bf16   q_scale*softmax_scale*n_head**-0.5 folded
        cmask   [Sq, Sk]    bool   TRUE where a key is NON-causal (key_pos > abs_pos)
        index_topk          int    clamped to Sk internally
      -> [Sq, kk] int32  top-k key indices (kk = min(index_topk, Sk); NOT -1 padded —
                         the -1 pad for short rows is a consume-side concern and is
                         covered by sparse_attn's own pad mask).

    torch.ops.hpu_dsa_ext.sparse_attn(q, k, v, topk_idx, scale, gather_width) -> [T,H,Dv]
        Built; CPU oracle 2e-7 (libdsa_sparse_attn{,_lazy}.so; coexists with the
        indexer via TORCH_LIBRARY_FRAGMENT — both schemas live in the one namespace).
        q          [T, H, 1, Dh]  bf16   per-head MLA query (nope||rope), L=1 per query
        k          [S, H, Dh]     bf16   per-head MLA key  (MQA expanded to H)
        v          [S, H, Dv]     bf16   per-head MLA value
        topk_idx   [T, K]         int32  the indexer top-k key indices (-1 padded ok)
        scale                     float  MLA softmax scale (qk_head_dim**-0.5)
        gather_width              int    MUST == K (the buffer is already the bucket)
      -> [T, H, Dv] bf16  sparse attention output.

GATE (the WHOLE point of MVP-3): everything here is reachable ONLY when the env flag
``HPU_DSA_SPARSE`` is truthy (default unset). With the flag unset, nothing in this
module is imported/called on the serving path, so the current dense-bypass serve
(``platform_patch.FULL_ATTENTION_FALLBACK=True``) is byte-for-byte unaffected.

.so SELECTION (mirrors hpu-moe-ext ``moe_wna16_hpu._load_fused_op``): the ops ship in
TWO builds per op — a LAZY-frontend .so (captures into the decode HPU-graph under
``PT_HPU_LAZY_MODE=1`` — the fast serving path glm51ar-lazy targets) and an EAGER one.
The eager .so does NOT load under lazy (its EagerOp symbols live only in the eager
plugin) and vice-versa, so we pick by ``habana_frameworks.torch.core.is_lazy()`` at
load time. The .so are built in the hpu-moe-ext repo; at serve time both repos are
mounted (hpu-moe-ext at /opt/hpu-moe-ext, hpu-dsa-ext at /opt/hpu-dsa-ext), so we
resolve the build dir against a list of candidate roots.

LOUD on misconfig (the mandate — never a silent dense fallback when sparse was asked
for): if ``HPU_DSA_SPARSE`` is on but the .so is missing or the op fails to register,
we raise. The flag was EXPLICITLY set, so a silent fallback would hide a broken build.
"""
from __future__ import annotations

import os

import torch

# ---------------------------------------------------------------------------
# The gate. OFF by default. Read once at import; the env is fixed for the
# process lifetime (set via the serve `-e HPU_DSA_SPARSE=1`). Exposed as a
# function too so tests can monkeypatch the env and re-read.
# ---------------------------------------------------------------------------
_SPARSE_ENV = "HPU_DSA_SPARSE"


def sparse_enabled() -> bool:
    """True iff the operator opted into the DSA sparse-op path (HPU_DSA_SPARSE=1).

    OFF by default. While False, the MLA forward keeps the dense bypass
    (platform_patch.FULL_ATTENTION_FALLBACK) and none of the op code below runs."""
    return os.environ.get(_SPARSE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# .so resolution. The build dir is hpu-moe-ext/tpc_kernel/opbackend/dsa/build.
# At serve time the repos mount at /opt/hpu-{moe,dsa}-ext; in the dev tree they
# live under $HOME/AI. Resolve against a candidate list; allow an explicit
# override via HPU_DSA_SO_DIR for tests / non-standard layouts.
# ---------------------------------------------------------------------------
_SO_DIR_ENV = "HPU_DSA_SO_DIR"

# Candidate roots that may contain `tpc_kernel/opbackend/dsa/build`. Ordered:
#   1. explicit override (handled separately, takes precedence),
#   2. container mount of hpu-moe-ext,
#   3. sibling of this repo's mount (…/hpu-moe-ext next to …/hpu-dsa-ext),
#   4. the dev checkout root.
_CANDIDATE_MOE_ROOTS = (
    "/opt/hpu-moe-ext",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "hpu-moe-ext")),
    "$HOME/AI/hpu-moe-ext",
)

_SO_SUBPATH = os.path.join("tpc_kernel", "opbackend", "dsa", "build")

# op-name -> (eager .so basename, lazy .so basename)
_SO_NAMES = {
    "indexer_topk": ("libdsa_indexer.so", "libdsa_indexer_lazy.so"),
    "sparse_attn": ("libdsa_sparse_attn.so", "libdsa_sparse_attn_lazy.so"),
}


def _is_lazy() -> bool:
    import habana_frameworks.torch.core as _htc
    return _htc.is_lazy()


def _candidate_build_dirs() -> list[str]:
    dirs: list[str] = []
    override = os.environ.get(_SO_DIR_ENV)
    if override:
        dirs.append(override)
    for root in _CANDIDATE_MOE_ROOTS:
        dirs.append(os.path.join(root, _SO_SUBPATH))
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for d in dirs:
        nd = os.path.normpath(d)
        if nd not in seen:
            seen.add(nd)
            out.append(nd)
    return out


def _so_path(op: str) -> str:
    """Resolve the .so for `op` (lazy vs eager picked by is_lazy()). Raises a LOUD
    RuntimeError listing every candidate dir searched if it cannot be found."""
    if op not in _SO_NAMES:
        raise KeyError(f"[hpu-dsa-ext] unknown DSA op {op!r} (have {list(_SO_NAMES)})")
    eager_name, lazy_name = _SO_NAMES[op]
    name = lazy_name if _is_lazy() else eager_name
    searched = []
    for d in _candidate_build_dirs():
        cand = os.path.join(d, name)
        searched.append(cand)
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        f"[hpu-dsa-ext] {_SPARSE_ENV}=1 but the DSA op library '{name}' (for "
        f"'{op}', mode={'lazy' if _is_lazy() else 'eager'}) was not found. "
        "Build it CPU-only inside the image:\n"
        "  docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work \\\n"
        "    -w /work/opbackend/dsa vllm-gaudi:next bash "
        f"build_dsa_{'indexer' if op == 'indexer_topk' else 'sparse_attn'}"
        f"{'_lazy' if _is_lazy() else ''}.sh\n"
        f"  or set {_SO_DIR_ENV}=<dir-with-the-.so>, or unset {_SPARSE_ENV}.\n"
        "  searched:\n    " + "\n    ".join(searched)
    )


# Load each .so exactly once, lazily, the first time its op is needed (NOT at
# import) so a CPU-only `import hpu_dsa_ext.dsa_ops` — and every non-sparse serve —
# never touches it. LOUD if the flag is on but a .so is missing / the op fails to
# register: sparse was explicitly requested, so a silent fallback would hide a
# broken build (the same contract as moe_wna16_hpu._load_fused_op).
_loaded = {"indexer_topk": False, "sparse_attn": False}


def _ensure_op(op: str):
    """Load the .so for `op` once and return the callable torch.ops handle. Both ops
    register into the SAME library namespace `hpu_dsa_ext` (the second via
    TORCH_LIBRARY_FRAGMENT), so loading both .so in one process is supported."""
    if not _loaded[op]:
        so = _so_path(op)
        torch.ops.load_library(so)
        if not (hasattr(torch.ops, "hpu_dsa_ext")
                and hasattr(torch.ops.hpu_dsa_ext, op)):
            raise RuntimeError(
                f"[hpu-dsa-ext] loaded {so}\nbut torch.ops.hpu_dsa_ext.{op} did NOT "
                "register (check the TORCH_LIBRARY / TORCH_LIBRARY_FRAGMENT def + the "
                ".so build).")
        import logging
        logging.getLogger("vllm").info(
            "[hpu-dsa-ext] loaded DSA op '%s' from %s (%s=1, mode=%s)",
            op, so, _SPARSE_ENV, "lazy" if _is_lazy() else "eager")
        _loaded[op] = True
    return getattr(torch.ops.hpu_dsa_ext, op)


def load_ops() -> None:
    """Eagerly load BOTH op libraries (used by register() so a misconfig fails at
    plugin-init, not mid-first-request). Idempotent. LOUD if HPU_DSA_SPARSE=1 and a
    .so is missing/unregisterable."""
    _ensure_op("indexer_topk")
    _ensure_op("sparse_attn")


# ===========================================================================
# Op-input builders. These translate the validated pure-torch tensor layouts
# (the executable spec in sparse_indexer_hpu.py / sparse_mla_consume.py) into the
# EXACT op-input layouts the kernels assert (see the *_shared.h contracts +
# dsa/tests/test_dsa_*_op.py, which these mirror byte-for-byte).
# ===========================================================================
def build_indexer_op_inputs(
    q_thd: torch.Tensor,        # [T, H, D]  bf16/fp8 per-head index queries (rope applied)
    k_sd: torch.Tensor,         # [S, D]     bf16 MQA index keys (single shared head)
    weights_th: torch.Tensor,   # [T, H]     bf16 folded per-head weights
    abs_pos: torch.Tensor,      # [T]        int absolute query position (causal bound)
):
    """Return (q_op[H,T,D], k_op[H,S,D], w_op[H,T,1], cmask[T,S]) for indexer_topk.

    Mirrors dsa/tests/test_dsa_indexer_op.py:151-156 exactly:
      q_op    = q.permute(1,0,2)                    -> [H,T,D]  (head-major batched Q.K^T)
      k_op    = k.unsqueeze(0).expand(H,-1,-1)      -> [H,S,D]  (MQA key expanded to H)
      w_op    = weights.permute(1,0).unsqueeze(-1)  -> [H,T,1]
      cmask   = key_pos > abs_pos                   -> [T,S] bool, TRUE => DROP (non-causal)
    q is cast to bf16 (the op consumes bf16; q_scale is already folded into weights —
    ReLU is positive-homogeneous so this is exact, sparse_indexer_hpu.py:13-17)."""
    bf16 = torch.bfloat16
    H = q_thd.shape[1]
    S = k_sd.shape[0]
    q_op = q_thd.permute(1, 0, 2).contiguous().to(bf16)                       # [H,T,D]
    k_op = k_sd.unsqueeze(0).expand(H, -1, -1).contiguous().to(bf16)          # [H,S,D]
    w_op = weights_th.permute(1, 0).unsqueeze(-1).contiguous().to(bf16)       # [H,T,1]
    key_pos = torch.arange(S, device=q_thd.device).view(1, -1)               # [1,S]
    cmask = (key_pos > abs_pos.view(-1, 1)).contiguous()                      # [T,S] bool TRUE=drop
    return q_op, k_op, w_op, cmask


# Pad the KEY axis up to a multiple of this so the op sees few distinct shapes.
# Serving grows S by 1 per decode step; without padding every step mints a NEW
# eager recipe per layer (78x per token). Pad rows are zeros and sit at
# key_pos >= S > abs_pos, so the in-recipe causal mask drops them; any pad index
# that still surfaces in a short row's top-k tail is -1'ed by the caller's causal
# post-filter (_op_score_segment). 128 == the serve block_size.
_KEY_PAD = 128


def _pad_rows(t: torch.Tensor, mult: int) -> torch.Tensor:
    """Zero-pad dim 0 of t up to the next multiple of `mult` (no-op if aligned)."""
    S = t.shape[0]
    pad = (-S) % mult
    if pad == 0:
        return t
    return torch.cat([t, t.new_zeros((pad,) + tuple(t.shape[1:]))], dim=0)


# torch.compiler.disable: these wrappers must NEVER be captured into a dynamo/
# torch.compile subgraph. The HPU compile backend caches compiled recipes
# SHAPE-AGNOSTICALLY and re-runs Synapse shape inference on a shape-changed cache
# hit -- custom OpBackend recipes have no SIF support, so a captured op crashes
# the worker with "[SHAPE AGNOSTIC] Cache hit Synapse shape inference failed!"
# (first hit: the serve's decode after prefill). Outside dynamo the op runs on the
# plain eager path (per-shape recipes), which is the HW-validated mode.
@torch.compiler.disable
def indexer_topk_op(
    q_thd: torch.Tensor,        # [T, H, D]
    k_sd: torch.Tensor,         # [S, D]
    weights_th: torch.Tensor,   # [T, H]
    abs_pos: torch.Tensor,      # [T] int
    index_topk: int,
) -> torch.Tensor:              # [T, kk] int32 top-k key indices
    """Drive the captured indexer_topk kernel. Returns the per-query top-k key indices
    [T, min(index_topk, S_padded)] int32. Causal (the cmask is applied inside the kernel
    before its top-k) over the REAL keys; emitted indices may include masked/pad keys
    ONLY in the tail of rows with fewer causal keys than kk -- the caller MUST apply the
    contract's causal post-filter (idx > abs_pos -> -1) before consuming."""
    op = _ensure_op("indexer_topk")
    k_sd = _pad_rows(k_sd, _KEY_PAD)
    # kk == min(index_topk, S_pad) must be a POWER OF TWO: the topk CGUID rejects
    # the recipe otherwise ("Invalid Node Params! Node name: n_ao", synStatus 26,
    # first seen at S_pad=1920 -> kk=1920; kk=128/2048 compile fine). When S_pad
    # is below index_topk, round it up to the next pow2 so kk = S_pad is pow2;
    # the extra rows are zero pads the causal mask drops. index_topk itself must
    # be pow2 (GLM/DeepSeek use 2048).
    if k_sd.shape[0] < index_topk:
        pow2 = 1 << (int(k_sd.shape[0]) - 1).bit_length()
        k_sd = _pad_rows(k_sd, pow2)
    return _call_indexer(op, q_thd, k_sd, weights_th, abs_pos, index_topk)


def _call_indexer(op, q_thd, k_sd, weights_th, abs_pos, index_topk):
    q_op, k_op, w_op, cmask = build_indexer_op_inputs(q_thd, k_sd, weights_th, abs_pos)
    return op(q_op, k_op, w_op, cmask, int(index_topk))


@torch.compiler.disable
def sparse_attn_op(
    q_thd: torch.Tensor,        # [T, H, Dh]  per-head MLA query (nope||rope)
    k_shd: torch.Tensor,        # [S, H, Dh]  per-head MLA key (MQA expanded)
    v_shd: torch.Tensor,        # [S, H, Dv]  per-head MLA value
    topk_idx: torch.Tensor,     # [T, K]      int32 indexer top-k (-1 padded ok)
    scale: float,
) -> torch.Tensor:              # [T, H, Dv]
    """Drive the captured sparse_attn kernel (gather-consume). The indexer's top-k
    indices ARE the gather indices (already causal); the kernel masks ONLY -1 pad slots,
    so no sentinel-scatter / causal re-derivation is needed here — that all lives inside
    the recipe. Returns [T, H, Dv] bf16, the per-head sparse attention output.

    The op asserts q is 4D [T,H,1,Dh] (L=1 per query) and gather_width == K (the buffer
    width). We add the L=1 axis and pass gather_width = topk_idx.size(1)."""
    op = _ensure_op("sparse_attn")
    bf16 = torch.bfloat16
    K = int(topk_idx.shape[1])
    q4 = q_thd.unsqueeze(2).contiguous().to(bf16)        # [T,H,1,Dh]
    # Pad the key/value S axis like the indexer's K axis (shape-churn control).
    # Gather indices only ever point at real rows (the indexer post-filter -1's
    # everything else) and -1 pads are masked inside the recipe, so zero pad rows
    # are never attended.
    k_op = _pad_rows(k_shd.contiguous().to(bf16), _KEY_PAD)   # [S_pad,H,Dh]
    v_op = _pad_rows(v_shd.contiguous().to(bf16), _KEY_PAD)   # [S_pad,H,Dv]
    ti = topk_idx.contiguous().to(torch.int32)           # [T,K] int32
    return op(q4, k_op, v_op, ti, float(scale), K)
