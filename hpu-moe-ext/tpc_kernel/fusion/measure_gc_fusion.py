"""measure_gc_fusion.py — THE decisive on-HPU measurement (RUN ON THE GAUDI BOX).

QUESTION THIS SETTLES
=====================
Does the Gaudi GC's `fuseDequantIntoMoeGemm` pass fuse the int4 dequant
(`torch.ops.hpu.convert_from_uint4`) into the MoE MME (`torch.ops.hpu.
mixture_of_experts`) — through PUBLIC `torch.ops.hpu` ops, exercised exactly the
way the stock `VllmMixtureOfExpertsOpWNA16.forward` does it — so that the bf16
weight stays SRAM-resident and only ~int4-sized bytes cross HBM? And under EXACTLY
which config (mark_step kept/dropped, HW-aligned scales on/off, GC fusion envs)?

This is the integration-architecture decider. It does NOT use raw synLaunch and
does NOT add a custom multi-node op (both internal-only, ruled out). It builds the
int4 MoE from the SAME public ops the stock op uses and reads the COMPILED GRAPH
DUMP to see whether the dequant output is an SRAM bundle (fused) or a DRAM /
persistent bf16 buffer the size of the bf16 weight (spilled).

WHY A GRAPH-DUMP READ (not a byte counter)
------------------------------------------
The authoritative, reproducible signal is the GC PostGraph dump (proven on the
standalone recipe in fusion/fused_moe_full*-PostGraph-symbol.pbtxt):
  * FUSED   : the dequant's bf16 output appears as `... | location = in SRAM |
              sramOffset = ...` inside a slicer `bundle_N` — NO full-size bf16
              buffer is ever materialized in DRAM.    (HBM weight ~ int4 size)
  * SPILLED : a bf16 tensor of size == bf16 weight (e.g. 50,331,648 B for a GLM
              w13 expert) is materialized `isPersistent | location = in DRAM`
              (often with `..._memcpy_N` aliases).     (HBM weight ~ bf16 size)
The verdict heuristic below was VALIDATED against those two reference dumps:
the FUSED dump has 0 full-bf16 DRAM weight tensors and N weight-sized bf16 tensors
in SRAM; the UNFUSED dump has the opposite. Same rule, applied to the live torch
graph here, gives the answer for the real op path.

WHAT IS SWEPT (resolves the architect disagreement)
---------------------------------------------------
  axis 1  mark_step BETWEEN the per-expert dequant and the MoE op : KEEP vs DROP
          (KEEP = byte-identical to the stock op, which calls htorch.core.mark_step()
           once after the dequant loop; DROP = let dequant + MoE land in ONE graph
           so the GC pass can even SEE both nodes to fuse them.)
  axis 2  HW-aligned scales (vllm_gaudi ConvertScaleToHwAligned) : ON vs OFF
          (the `fuseDequantIntoMoeGemm` pass gates on
           QuantizationUtils::isScaleExpBiasHWAligned — confirmed in libSynapse.so.
           STOCK vllm-gaudi int4 MoE loader does NOT align the group scales
           (hpu_compressed_tensors.py:937-945 sets weight_scale RAW), so this axis
           tells us whether shipping requires adding the alignment.)
  axis 3 (env, fixed ON for the sweep, toggled in the optional 2nd pass):
          PT_HPU_LAZY_MODE=1, FUSE_DEQUANT_TO_MOE_GEMM=1 (+GCFG_),
          ENABLE_TPC_STITCHING_TO_BGEMM=1 (+GCFG_),
          SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED=1 (+GCFG_).

Each cell prints: numeric coherence (output finite + matches an eager int4 ref),
the bf16-weight-DRAM bytes and SRAM-weight bytes read from the dump, and a verdict
FUSED / SPILLED / PARTIAL / NO-FUSE. The final matrix names the winning config or
says the pre-transpose-at-load fallback is needed, and on a clean miss dumps the
SRAM-slicer reject reason (LOG_LEVEL_SRAM_SLICE=2) that names the blocker.

RUN
---
The script sets every env itself and re-execs once under PT_HPU_LAZY_MODE=1 so a
single invocation is enough:

    cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
    python measure_gc_fusion.py                 # GLM-shaped default
    python measure_gc_fusion.py 6144 2048 128 8 # H I GS E  (override)

Exit 0  == at least one config FUSED (SRAM-resident, HBM weight ~ int4) with
           coherent output -> that config is shippable.
Exit 1  == NO config fused -> read the printed SRAM-slice reject reason; the
           pre-transpose-at-load fallback is required.

GUARDRAIL: this is the device run the human performs. It imports
habana_frameworks and uses the HPU. Do not run it in CI / the dev container.
"""
from __future__ import annotations

import glob
import os
import re
import sys


# ---------------------------------------------------------------------------
# 0. Env. We set the GC fusion knobs + lazy mode BEFORE importing torch/habana,
#    because the GC reads them at process init. We re-exec once to guarantee
#    PT_HPU_LAZY_MODE is in place before habana_frameworks loads.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DUMP_ROOT = os.path.join(_HERE, "gc_fusion_dumps")
_SENTINEL = "_MEASURE_GC_FUSION_REEXEC"

# GC pass knobs — names CONFIRMED present in /usr/lib/habanalabs/libSynapse.so.
# We set BOTH the bare and GCFG_-prefixed spellings; the GC accepts either.
_FUSION_ENV = {
    "PT_HPU_LAZY_MODE": "1",                     # graph mode (eager can't bundle)
    "FUSE_DEQUANT_TO_MOE_GEMM": "1",
    "GCFG_FUSE_DEQUANT_TO_MOE_GEMM": "1",
    "ENABLE_TPC_STITCHING_TO_BGEMM": "1",
    "GCFG_ENABLE_TPC_STITCHING_TO_BGEMM": "1",
    "SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED": "1",
    "GCFG_SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED": "1",
    # graph dump (-> <dir>/*-PostGraph-symbol.pbtxt) + slicer accept/reject log.
    "GRAPH_VISUALIZATION": "1",
    "DUMP_POST_GRAPHS": "1",
    "LOG_LEVEL_SRAM_SLICE": "2",
}


def _bootstrap_env_and_reexec() -> None:
    """Set fusion env, then re-exec once so PT_HPU_LAZY_MODE is live before any
    habana import. Idempotent via a sentinel env var."""
    if os.environ.get(_SENTINEL) == "1":
        return
    for k, v in _FUSION_ENV.items():
        os.environ.setdefault(k, v)
    os.environ[_SENTINEL] = "1"
    # Re-exec the same interpreter+args so the lazy-mode env is honored at import.
    os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_env_and_reexec()

import torch  # noqa: E402

try:
    import habana_frameworks.torch as ht  # noqa: E402,F401
    import habana_frameworks.torch.core as htcore  # noqa: E402
except Exception as ex:  # noqa: BLE001
    print(f"FATAL: habana_frameworks not importable ({ex}). Run on the HPU box.")
    sys.exit(2)

# HW-aligned scale converter — the EXACT public class the stock fp8 path uses.
try:
    from vllm_gaudi.extension.scales import ConvertScaleToHwAligned  # noqa: E402
    _HAVE_ALIGN = True
except Exception as ex:  # noqa: BLE001
    print(f"WARN: vllm_gaudi ConvertScaleToHwAligned import failed ({ex}); "
          f"aligned-scale axis will be reported as UNAVAILABLE.")
    ConvertScaleToHwAligned = None  # type: ignore[assignment]
    _HAVE_ALIGN = False


# ---------------------------------------------------------------------------
# 1. Synthetic GLM-shaped int4 MoE weights (no checkpoint).
#    Per stock loader the dequant call is:
#      convert_from_uint4(weight_packed[I, P/8]i32, weight_scale[G, P]bf16,
#                         zero_point[G, P/8]i32, out_dtype=bf16, g_idx=None)
#    For the w13 (gate+up) expert we use OUT = 2*I (== P) so the dequanted weight
#    is [H, 2I] -> the mixture_of_experts w12 slot. For w2 (down) OUT = H, in
#    slot w3, weight [I, H]. We build BOTH so the op is exercised end-to-end, but
#    the verdict keys on the w13 weight (the big one, ~bf16 50MB / ~int4 12.5MB at
#    GLM), matching the standalone recipe's reference.
# ---------------------------------------------------------------------------
def _sym_zero_packed(groups, out_dim, device):
    """GPTQ-symmetric shared zero = uint4 midpoint 8, packed 8/int32 along OUT.
    Shape [GROUPS, OUT/8] int32 — identical to the production _sym_zero and to
    verify_convert_from_uint4_args.py / profile_fusion.py."""
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)            # 0x88888888
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _pack_u4(in_dim, out_dim, seed, device):
    """Random 4-bit codes -> [in_dim, out_dim/8] int32 (8 nibbles/word along OUT)."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (in_dim, out_dim), generator=g, dtype=torch.int32)
    c = (codes.to(torch.int64) & 0xF).reshape(in_dim, out_dim // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.to(device)


def _scale(groups, out_dim, seed, device):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((groups, out_dim), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)


class ExpertW:
    """Holds one expert's packed int4 weight + (raw and aligned) group scales + zp."""

    def __init__(self, in_dim, out_dim, gs, seed, device):
        self.in_dim, self.out_dim, self.gs = in_dim, out_dim, gs
        self.groups = in_dim // gs
        self.packed = _pack_u4(in_dim, out_dim, seed, device)            # [I, O/8] i32
        self.scale_raw = _scale(self.groups, out_dim, seed + 1, device)  # [G, O] bf16
        self.zp = _sym_zero_packed(self.groups, out_dim, device)         # [G, O/8] i32
        # HW-aligned scale (pow2 / exp-bias snapped). Computed once, on device.
        if _HAVE_ALIGN:
            self.scale_aligned = ConvertScaleToHwAligned().calc(self.scale_raw.clone())
        else:
            self.scale_aligned = None

    def dequant(self, aligned: bool) -> torch.Tensor:
        scale = self.scale_aligned if aligned else self.scale_raw
        # EXACT stock call shape: convert_from_uint4(packed, scale, zp, bf16, g_idx=None)
        return torch.ops.hpu.convert_from_uint4(self.packed, scale, self.zp,
                                                torch.bfloat16, None)


# ---------------------------------------------------------------------------
# 2. The MoE forward, byte-for-byte the stock VllmMixtureOfExpertsOpWNA16 path
#    for E<=MAX_EXPERTS_PER_SLICE (moe_n_slice==1 -> a SINGLE mixture_of_experts
#    call, no inter-slice mark_step). The ONLY knob we vary is whether the single
#    htorch.core.mark_step() that the stock op places AFTER the dequant loop is
#    kept or dropped.
# ---------------------------------------------------------------------------
def moe_forward(x, topk_ids, topk_w, w13: "list[ExpertW]", w2: "list[ExpertW]",
                *, aligned: bool, keep_mark_step: bool) -> torch.Tensor:
    w12_list = [e.dequant(aligned) for e in w13]      # [H, 2I] bf16 each
    w3_list = [e.dequant(aligned) for e in w2]        # [I, H]  bf16 each
    if keep_mark_step:
        htcore.mark_step()                            # <-- stock op does exactly this
    # kwargs w12=/w3= resolve to the .fused_weights overload (the stock op's call).
    return torch.ops.hpu.mixture_of_experts(
        hidden_states=x,
        expert_routing_table=topk_ids,
        router_weights=topk_w,
        w12=w12_list,
        w3=w3_list,
        permuted_weights=True,
        activation="silu",
        experts_min=0,
        experts_max=len(w13),
    )


def eager_ref(x, topk_ids, topk_w, w13, w2, aligned: bool) -> torch.Tensor:
    """Independent dense reference: for each token, for each of its top_k experts,
    y += rw * silu(x@W13_gate) * (x@W13_up) @ W2. Uses the SAME dequant op so the
    check validates the op math, not the quant scheme. Coherence gate only (rtol
    loose: silu + bf16 accumulation order differ from the fused kernel)."""
    H = x.shape[-1]
    I = w2[0].in_dim
    y = torch.zeros_like(x)
    for t in range(x.shape[0]):
        acc = torch.zeros(H, dtype=torch.float32, device=x.device)
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k].item())
            rw = topk_w[t, k].to(torch.float32)
            w13b = w13[e].dequant(aligned).to(torch.float32)   # [H, 2I]
            w2b = w2[e].dequant(aligned).to(torch.float32)     # [I, H]
            gu = x[t].to(torch.float32) @ w13b                 # [2I]
            gate, up = gu[:I], gu[I:]
            act = torch.nn.functional.silu(gate) * up          # [I]
            acc += rw * (act @ w2b)                            # [H]
        y[t] = acc.to(y.dtype)
    return y


# ---------------------------------------------------------------------------
# 3. Read the GC PostGraph dump -> weight-bytes in DRAM (bf16-materialized =
#    spilled) vs in SRAM (bundled = fused). Heuristic VALIDATED on the proven
#    fused/unfused reference dumps.
# ---------------------------------------------------------------------------
# A tensor descriptor line in *-PostGraph-symbol.pbtxt looks like:
#   s: "NAME  |  Sizes = [..]  |  bf16  |  ... | sizeInBytes = N | isPersistent |
#       location = in DRAM | dramOffset = .. |  ..."   (or  location = in SRAM | sramOffset)
_DTYPE_RE = re.compile(r"\|\s*(bf16|float32|int32|uint4|fp8[^\s|]*)\s*\|")
_SIZE_RE = re.compile(r"sizeInBytes\s*=\s*(\d+)")
_LOC_RE = re.compile(r"location\s*=\s*in\s+(SRAM|DRAM)")


def _latest_postgraphs(dump_dir) -> "list[str]":
    pats = [os.path.join(dump_dir, "**", "*PostGraph*symbol*.pbtxt"),
            os.path.join(dump_dir, "**", "*PostGraph*.pbtxt"),
            os.path.join(dump_dir, "**", "*post_graph*.pbtxt"),
            os.path.join(dump_dir, "**", "*.pbtxt")]
    seen, out = set(), []
    for p in pats:
        for f in sorted(glob.glob(p, recursive=True), key=os.path.getmtime, reverse=True):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def analyze_dump(dump_dir, bf16_w_bytes, int4_w_bytes, mtime_after) -> dict:
    """Scan the freshest PostGraph dumps written after `mtime_after`. Returns:
      bf16_dram_max  : largest bf16 tensor materialized in DRAM that is >= 0.5x
                       the bf16 weight (the spill signal; aliased int4 views are
                       NOT bf16 so they don't count).
      bf16_sram_max  : largest bf16 tensor resident in SRAM (the bundle signal).
      n_dram_fullbf16: count of bf16 DRAM tensors ~ bf16 weight size.
      n_sram_wbf16   : count of bf16 SRAM tensors ~ a weight slice (>=0.5x int4).
      files          : dumps inspected.
    Aliased tensors (`isAliased = ...`) reuse another tensor's bytes; we still read
    them because the spill in the UNFUSED reference appears as `..._memcpy_N` ALIAS
    rows at full bf16 size. The decisive fact is the dtype==bf16 + size~bf16weight +
    location==DRAM combination, which only occurs when the dequant materialized."""
    files = [f for f in _latest_postgraphs(dump_dir) if os.path.getmtime(f) >= mtime_after - 1.0]
    res = {"bf16_dram_max": 0, "bf16_sram_max": 0, "n_dram_fullbf16": 0,
           "n_sram_wbf16": 0, "files": files, "any_bf16": 0}
    half_bf16 = bf16_w_bytes * 0.5
    half_int4 = int4_w_bytes * 0.5
    for path in files:
        try:
            with open(path, errors="ignore") as fh:
                text = fh.read()
        except Exception:  # noqa: BLE001
            continue
        for line in text.splitlines():
            if "sizeInBytes" not in line or "|" not in line:
                continue
            mdt = _DTYPE_RE.search(line)
            msz = _SIZE_RE.search(line)
            mlo = _LOC_RE.search(line)
            if not (mdt and msz):
                continue
            dtype = mdt.group(1)
            nbytes = int(msz.group(1))
            loc = mlo.group(1) if mlo else ("SRAM" if "sramOffset" in line else "DRAM")
            if dtype != "bf16":
                continue
            res["any_bf16"] += 1
            if loc == "DRAM" and nbytes >= half_bf16:
                res["bf16_dram_max"] = max(res["bf16_dram_max"], nbytes)
                res["n_dram_fullbf16"] += 1
            if loc == "SRAM" and nbytes >= half_int4:
                res["bf16_sram_max"] = max(res["bf16_sram_max"], nbytes)
                res["n_sram_wbf16"] += 1
    return res


def verdict_from_dump(d: dict, bf16_w_bytes, int4_w_bytes) -> "tuple[str, str]":
    """Map dump stats -> verdict. Rule (validated on reference dumps):
      FUSED   : NO full-bf16 weight tensor in DRAM AND >=1 weight-sized bf16 in SRAM.
      SPILLED : a full-bf16 weight tensor IS materialized in DRAM.
      NO-FUSE : graph found but neither signal (op may not have lowered / dump empty).
      PARTIAL : both signals present (some experts bundled, some spilled)."""
    spill = d["n_dram_fullbf16"] > 0
    bundle = d["n_sram_wbf16"] > 0
    mb = lambda b: b / 1048576.0
    detail = (f"bf16-in-DRAM max={mb(d['bf16_dram_max']):.1f}MB (x{d['n_dram_fullbf16']}), "
              f"bf16-in-SRAM max={mb(d['bf16_sram_max']):.1f}MB (x{d['n_sram_wbf16']}); "
              f"bf16 weight={mb(bf16_w_bytes):.1f}MB int4={mb(int4_w_bytes):.1f}MB; "
              f"bf16 tensors seen={d['any_bf16']}; dumps={len(d['files'])}")
    if spill and bundle:
        return "PARTIAL", detail
    if spill:
        return "SPILLED", detail
    if bundle:
        return "FUSED", detail
    return "NO-FUSE", detail


# ---------------------------------------------------------------------------
# 4. SRAM-slicer reject reason (for the clean-miss branch). LOG_LEVEL_SRAM_SLICE=2
#    routes the slicer/stitcher decisions to the Synapse log; we surface lines that
#    name why a node could not be stitched / sliced on the common dim (these come
#    from validateGemmCommonDim / MMESlaveBrain::validateNodeToBeStitched /
#    MmeNodeSolver::sliceNodeOnSingleCommonDim, confirmed in libSynapse.so).
# ---------------------------------------------------------------------------
def grep_slicer_reason() -> "list[str]":
    logdirs = [os.environ.get("HABANA_LOGS", ""), os.path.join(_HERE, "logs"),
               os.path.expanduser("~/.habana_logs"), "/var/log/habana_logs", _HERE]
    pats, hits = [], []
    for d in logdirs:
        if d and os.path.isdir(d):
            pats += glob.glob(os.path.join(d, "**", "*.log"), recursive=True)
            pats += glob.glob(os.path.join(d, "**", "*synapse*log*"), recursive=True)
    key = re.compile(r"stitch|common[\s_-]?dim|bundle|reject|cannot|not\s+allowed|"
                     r"unsupported|transpos|dequant|moe", re.I)
    for f in sorted(set(pats), key=os.path.getmtime, reverse=True)[:8]:
        try:
            with open(f, errors="ignore") as fh:
                for line in fh:
                    if key.search(line):
                        hits.append(f"{os.path.basename(f)}: {line.strip()[:240]}")
        except Exception:  # noqa: BLE001
            continue
    return hits[:40]


# ---------------------------------------------------------------------------
# 5. Driver: build once, sweep the matrix, print the decision.
# ---------------------------------------------------------------------------
def _shapes(argv):
    H = int(argv[1]) if len(argv) > 1 else 6144   # hidden
    I = int(argv[2]) if len(argv) > 2 else 2048   # intermediate (per gate/up)
    GS = int(argv[3]) if len(argv) > 3 else 128    # group size
    E = int(argv[4]) if len(argv) > 4 else 8       # routed experts
    return H, I, GS, E


def main() -> int:
    H, I, GS, E = _shapes(sys.argv)
    TOPK = E                                        # GLM routed: top_k == E here
    OUT13 = 2 * I                                   # gate+up fused -> w12 slot
    OUT2 = H                                        # down          -> w3  slot
    dev = "hpu"

    # weight reference sizes (the w13 expert — the big/decisive one).
    int4_w13 = H * OUT13 // 2
    bf16_w13 = H * OUT13 * 2
    mb = lambda b: b / 1048576.0

    print("=" * 80)
    print("measure_gc_fusion — does fuseDequantIntoMoeGemm fuse int4 dequant into")
    print("the MoE MME via PUBLIC torch.ops.hpu, and under which config?")
    print("=" * 80)
    print(f"shapes: H={H} I={I} GS={GS} E={E} top_k={TOPK}  (w13 OUT={OUT13}, w2 OUT={OUT2})")
    print(f"w13 weight: int4={mb(int4_w13):.1f}MB  bf16={mb(bf16_w13):.1f}MB  "
          f"-> FUSED iff weight stays SRAM (no full-bf16 DRAM tensor); "
          f"SPILLED iff a {mb(bf16_w13):.0f}MB bf16 tensor lands in DRAM.")
    print("env:", {k: os.environ.get(k) for k in
                   ("PT_HPU_LAZY_MODE", "FUSE_DEQUANT_TO_MOE_GEMM",
                    "ENABLE_TPC_STITCHING_TO_BGEMM",
                    "SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED")})
    print(f"aligned-scale axis available: {_HAVE_ALIGN}")
    if I % GS != 0 or H % GS != 0 or OUT13 % 8 != 0 or OUT2 % 8 != 0:
        print(f"FATAL: need H%GS==0,I%GS==0,OUT%8==0 (H={H} I={I} GS={GS})")
        return 2

    # --- build weights once (reused across all cells; cheap, deterministic) -----
    print("\nbuilding synthetic int4 experts ...", flush=True)
    w13 = [ExpertW(H, OUT13, GS, seed=100 + j, device=dev) for j in range(E)]
    w2 = [ExpertW(I, OUT2, GS, seed=500 + j, device=dev) for j in range(E)]

    # confirm aligned scales really are exp-bias snapped (pow2) and differ from raw.
    if _HAVE_ALIGN:
        sr, sa = w13[0].scale_raw.float(), w13[0].scale_aligned.float()
        log2 = torch.log2(sa[sa > 0])
        is_pow2 = bool(torch.allclose(log2, torch.round(log2), atol=1e-3))
        changed = float((sr != sa).float().mean().item())
        print(f"aligned-scale check: pow2/exp-bias-snapped={is_pow2} "
              f"(frac scales changed vs raw={changed:.2f})")

    x = torch.randn((1, H), dtype=torch.bfloat16, device=dev)        # single-stream T=1
    # routing: token 0 routes to all E experts (top_k==E), uniform weights.
    topk_ids = torch.arange(E, dtype=torch.int64, device=dev).reshape(1, E)
    topk_w = torch.full((1, E), 1.0 / E, dtype=torch.bfloat16, device=dev)

    cells = [(ms, al) for ms in (True, False) for al in
             ((True, False) if _HAVE_ALIGN else (False,))]
    results = []
    for keep_ms, aligned in cells:
        tag = f"mark_step={'KEEP' if keep_ms else 'DROP'} | aligned_scales={'ON' if aligned else 'OFF'}"
        print("\n" + "-" * 80)
        print(f"[CELL] {tag}")
        print("-" * 80, flush=True)

        # isolate this cell's dump in its own dir so the reader sees only its graph.
        cell_dir = os.path.join(_DUMP_ROOT, f"ms{int(keep_ms)}_al{int(aligned)}")
        os.makedirs(cell_dir, exist_ok=True)
        os.environ["GRAPH_VISUALIZATION_DIR"] = cell_dir
        os.environ["HABANA_LOGS"] = os.path.join(cell_dir, "logs")
        os.makedirs(os.environ["HABANA_LOGS"], exist_ok=True)
        t0 = _now()

        coherent = None
        err = None
        try:
            y = moe_forward(x, topk_ids, topk_w, w13, w2,
                            aligned=aligned, keep_mark_step=keep_ms)
            htcore.mark_step()
            ht.hpu.synchronize()
            yc = y.to("cpu").float()
            finite = bool(torch.isfinite(yc).all())
            # coherence vs independent eager ref (loose; gate against broken math).
            ref = eager_ref(x, topk_ids, topk_w, w13, w2, aligned).to("cpu").float()
            denom = ref.norm().item() or 1.0
            rel = (yc - ref).norm().item() / denom
            coherent = finite and rel < 0.15
            print(f"  output: shape={tuple(y.shape)} finite={finite} "
                  f"rel_err_vs_eager_ref={rel:.4f} -> coherent={coherent}")
        except Exception as ex:  # noqa: BLE001
            err = f"{type(ex).__name__}: {ex}"
            print(f"  !! op raised: {err}")

        d = analyze_dump(cell_dir, bf16_w13, int4_w13, mtime_after=t0)
        v, detail = verdict_from_dump(d, bf16_w13, int4_w13)
        if err:
            v = "ERROR"
        print(f"  dump: {detail}")
        print(f"  VERDICT: {v}")
        results.append((tag, v, coherent, detail, err, cell_dir))

    # ----------------------------- decision matrix -----------------------------
    print("\n" + "=" * 80)
    print("RESULT MATRIX")
    print("=" * 80)
    print(f"{'config':<46}{'verdict':<10}{'coherent':<10}")
    fused_ok = []
    for tag, v, coh, detail, err, cell_dir in results:
        cflag = "n/a" if coh is None else ("YES" if coh else "NO")
        print(f"{tag:<46}{v:<10}{cflag:<10}")
        if v == "FUSED" and coh:
            fused_ok.append(tag)
    print("-" * 80)

    if fused_ok:
        print(">>> FUSED WIN. Shippable config(s) (dequant SRAM-resident, HBM weight ~ int4):")
        for t in fused_ok:
            print(f"      {t}")
        print(">>> INTEGRATION: ship the int4 MoE on the public op pattern with this "
              "cell's settings. If the winning cell has aligned_scales=ON, the "
              "integration MUST apply vllm_gaudi ConvertScaleToHwAligned to the int4 "
              "group scales at load (stock vllm-gaudi does NOT — it sets weight_scale "
              "raw), enable use_hpu_aligned_scale, and set the FUSE_DEQUANT_TO_MOE_GEMM "
              "/ TPC-stitching / common-dim-slice envs. If the winner has "
              "mark_step=DROP, the integration must REMOVE the htorch.core.mark_step() "
              "the stock VllmMixtureOfExpertsOpWNA16 places between the dequant loop "
              "and the MoE op (patch ops.py) so both nodes share one graph.")
        rc = 0
    else:
        print(">>> NO config fused (no SRAM-resident dequant on any cell).")
        any_spill = any(v in ("SPILLED", "PARTIAL") for _, v, *_ in results)
        if any_spill:
            print(">>> The dequant bf16 weight round-tripped HBM in every cell. The "
                  "GC's fuseDequantIntoMoeGemm did not bundle the convert_from_uint4 "
                  "producer into the MoE MME through the public op path.")
        print(">>> SRAM-slicer reject reason(s) (LOG_LEVEL_SRAM_SLICE=2) — names the blocker:")
        reasons = grep_slicer_reason()
        if reasons:
            for r in reasons[:25]:
                print(f"      {r}")
            print(">>> If the reason names the dequant output layout / a TRANSPOSING "
                  "producer / common-dim that cannot be sliced, the blocker is the "
                  "transposing dequant feeding the MME. INTEGRATION FALLBACK: "
                  "pre-transpose-at-load — store the int4 weight already transposed so "
                  "convert_from_uint4 emits the MME's required [K, N] layout with no "
                  "transpose node, then re-run this sweep to confirm the bundle forms.")
        else:
            print("      (no slicer log lines matched; inspect the per-cell logs under")
            print(f"       {_DUMP_ROOT}/*/logs and the PostGraph dumps there.)")
            print(">>> FALLBACK regardless: pre-transpose-at-load (store int4 weight "
                  "transposed so the dequant feeds [K,N] to the MME without a transpose "
                  "node) is the next lever if no env config bundles.")
        rc = 1

    print("\nartifacts: per-cell graph dumps + slicer logs under", _DUMP_ROOT)
    print("=" * 80)
    return rc


def _now() -> float:
    import time
    return time.time()


if __name__ == "__main__":
    sys.exit(main())
