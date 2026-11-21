"""test_fused_routed_capture.py -- STANDALONE reproducer for the fused-MoE WRAPPER's
`wrap_in_hpu_graph` MULTI-BUCKET CAPTURE crash, in ISOLATION (no 12-minute GLM serve).

WHAT IT REPRODUCES (from the live Qwen-397B serve, HPU_MOE_FUSED_OP=1 under lazy,
VLLM_SKIP_WARMUP=true)
----------------------------------------------------------------------
The serve wraps the WHOLE model.forward in `wrap_in_hpu_graph` and caches ONE HPU graph
per decode bucket key `(phase, batch_size, seq_len, num_blocks)`:

  * Bucket ('decode',1,1,1) (num_blocks=1) captures + replays fine -> coherent correct
    output ("180 miles", "Paris", ...). The single-recipe op made the K WEIGHT inputs
    persistent, so the ONE-bucket capture is clean.
  * Bucket ('decode',1,1,2) (num_blocks=2, reached when the context grows past one
    block) is "not warmed-up", so it is captured ON-THE-FLY mid-generation -> FATAL:

      FATAL ERROR :: MODULE:PT_LAZY ValidateSyncInputTensors tensor_data is empty ...
      model/0/mlp/experts/hpu__input ... Strided Params Has Value: 0   (hpu_lazy_tensors.cpp:871)

    The IR at the crash: `hpu_moe_ext::fused_moe{ aten__view(x), ListConstruct x7 }`
    then `aten::view(fused_moe) -> [1,4096]`. The EMPTY tensor is the FIRST input --
    `model/0/mlp/experts/hpu__input`, i.e. **x (the experts-input activation), which
    arrives as an `aten::view`**.

DIAGNOSED MECHANISM (x is empty at the 2nd bucket's capture)
-----------------------------------------------------------
The single-recipe op consumes x as its FIRST recipe input (syn_in(0)). x arrives as an
`aten::view` of a model intermediate (apply_monolithic: `x = x.view(-1, H)`, fed by
layer-0's post-attn RMSNorm/residual). The bridge roots the recipe's x input at the
VIEW'S BASE lazy-tensor (HPUGraph.cpp get_hb_base_tensor_id_list_if_view walks
stride_params to the base). The K WEIGHTS are safe -- index_select roots them at
PERSISTENT nn.Parameters. x's base is a NON-persistent decode intermediate. One bucket
is fine; multi-bucket is the trap: at bucket-1's REPLAY every captured-graph output that
is not also an input is freed (SingleHPUGraph::replayGraph -> SetTensorDataNullOpt; plus
the cross-graph "graph1 input x; graph2 input x" hazard the bridge's own clear_inputs
comment documents). When generation grows into bucket-2 and it is captured ON-THE-FLY,
the dry-run replayV3 validates x's input and finds its view-base storage already nulled
-> ValidateSyncInputTensors "tensor_data is empty / Strided Params Has Value: 0".

WHY free_inplace=False (the __init__.py patch) does NOT cover x: HPUGraph::mark_user_outputs
gates that exemption on IsNodeSliceOrStridedInsert (hpu::slice_insert / hpu::strided_insert)
ONLY (HPUGraph.cpp:362). x's producer is a view of an add/norm, not a slice/strided-insert,
so it is never exempted.

THE FIX UNDER TEST (approach B -- PERSISTENT static-input x buffer)
------------------------------------------------------------------
An in-forward `x.contiguous().clone()` does NOT fix the 2nd bucket: a clone born inside
the captured forward is an in-graph INTERMEDIATE (an output-not-input), so the
free_inplace pass reclaims it across buckets exactly like the raw view (mark_user_outputs
only exempts recipe INPUTS or slice/strided_insert producers). The FIX is the canonical
HPU-graph static-input pattern (== vllm-gaudi's self.input_ids_hpu, allocated once in
__init__ then copy_()'d each decode step): each MoE layer pre-allocates a PERSISTENT
[max_T,H] bf16 `x_buf` at WEIGHT-LOAD time (OUTSIDE any capture); `fused_moe_routed` does
`x_buf[:T].copy_(x)` and feeds `x_buf[:T]` to the op. x_buf is a held leaf the layer owns
(like an nn.Parameter), so its base lazy-id IS in hblazy_tensors_in_ -> is_input -> NEVER
freed at replay; and the in-place copy_ is a strided_insert that free_inplace=False also
exempts. So EACH bucket's recipe roots x at the SAME live, held storage. This test
allocates x_buf in RoutedModule.__init__ (outside capture, mirroring the serve) and passes
it via x_buf=. Toggle with HPU_MOE_FUSED_X_BUFFER=0 (drop the buffer -> clone fallback ->
2nd-bucket crash); HPU_MOE_FUSED_X_MATERIALIZE=0 additionally drops the clone fallback.

This test exercises the serve's EXACT trap: x is produced INSIDE the wrapped forward as
an `aten::view` of an internal intermediate (NOT a top-level user input -- a user input
would be matched by mark_user_inputs and survive, hiding the bug), and TWO distinct
decode buckets are captured SEQUENTIALLY (a num_blocks=1 then a num_blocks=2 surrogate),
so the SECOND capture is the on-the-fly one that reproduces the FATAL.

ASSERTS
  A  NO ValidateSyncInputTensors / empty-tensor FATAL across BOTH buckets' captures +
     their replays (the bucket-2 capture is the make-or-break one).
  B  replays are numerically correct vs an INDEPENDENT per-expert convert_from_uint4
     oracle (relNorm < 1e-2) and bit-identical replay-to-replay, for BOTH buckets.
  C  each bucket is a distinct cached recipe and its 2nd+ calls are graph REPLAYS
     (CachedParams.cache_hits incrementing; TWO recipes cached -- one per bucket).

TWO multi-bucket sources are covered (both are an index_select SOURCE freed across buckets;
the op's weight/zp inputs ARE those sources, so a NON-persistent source is the whole bug):
  * x  -- the experts-input activation (aten::view of a decode intermediate). FIX: the
         persistent x_buf static-input buffer (HPU_MOE_FUSED_X_BUFFER, default ON).
  * zp -- the stacked GPTQ-sym zero-point. The serve's _fused_moe_routed built it IN-FORWARD
         via expand().contiguous() (an in-graph intermediate -> freed across buckets -> the
         LAST serve FATAL, on the zp `hpu__input`). FIX: _alloc_fused_zp materializes the
         stacked zp as a HELD LEAF at weight-load (outside capture); this test mirrors that
         with a persistent buffer (default) vs the in-forward build (negative control).
The four packed-weight/scale sources never failed: they are nn.Parameters (persistent).

NEGATIVE CONTROLS (reproduce a crash):
  * HPU_MOE_FUSED_X_BUFFER=0 -> drop the persistent x buffer -> in-forward CLONE fallback,
    reclaimed across buckets -> bucket-2 capture HITs the x multi-bucket FATAL.
    (+ HPU_MOE_FUSED_X_MATERIALIZE=0 also drops the clone -> raw aten::view x.)
  * HPU_MOE_TEST_ZP_INFORWARD=1 -> build the stacked zp IN-FORWARD (expand().contiguous(),
    the pre-fix _fused_moe_routed) -> its index_select source is freed across buckets ->
    bucket-2 capture HITs the zp multi-bucket FATAL (the LAST one this fix closes).
  * HPU_MOE_ROUTED_VIA_SINGLE_RECIPE=0 -> the legacy per-expert in-graph clone loop (the
    OLDER weight-clone reproduction).

RUN (ON THE HPU BOX ONLY -- capture is a device mechanism):
  cd $HOME/AI/hpu-moe-ext
  PT_HPU_LAZY_MODE=1 python tests/test_fused_routed_capture.py
  # shapes: T H I GS E K   (defaults = Qwen/GLM int4: 1 6144 2048 128 8 8)
  PT_HPU_LAZY_MODE=1 python tests/test_fused_routed_capture.py 1 6144 2048 128 8 8
  # reproduce the x multi-bucket crash (clone fallback freed across buckets):
  PT_HPU_LAZY_MODE=1 HPU_MOE_FUSED_X_BUFFER=0 \
      python tests/test_fused_routed_capture.py
  # reproduce it with the raw aten::view x (also drop the clone fallback):
  PT_HPU_LAZY_MODE=1 HPU_MOE_FUSED_X_BUFFER=0 HPU_MOE_FUSED_X_MATERIALIZE=0 \
      python tests/test_fused_routed_capture.py
  # reproduce the ZP multi-bucket crash (in-forward expand().contiguous() zp source):
  PT_HPU_LAZY_MODE=1 HPU_MOE_TEST_ZP_INFORWARD=1 \
      python tests/test_fused_routed_capture.py
  # reproduce the OLDER weight-clone crash (legacy per-expert loop):
  PT_HPU_LAZY_MODE=1 HPU_MOE_ROUTED_VIA_SINGLE_RECIPE=0 \
      python tests/test_fused_routed_capture.py

EXIT CODE: 0 only if A, B, C all pass for BOTH buckets. Nonzero (LOUD) otherwise.
"""
from __future__ import annotations

import os
import sys

# --- MUST be set before importing habana_frameworks: it selects the LAZY plugin at
#     import (habana_frameworks/torch/__init__.py). HPU-graph capture is lazy-only.
if "habana_frameworks" in sys.modules:
    print("FATAL: habana_frameworks was imported BEFORE this test set "
          "PT_HPU_LAZY_MODE=1; the plugin (lazy vs eager) is locked at import. "
          "Run as:  PT_HPU_LAZY_MODE=1 python tests/test_fused_routed_capture.py",
          file=sys.stderr)
    sys.exit(2)
if os.environ.get("PT_HPU_LAZY_MODE") != "1":
    print("NOTE: PT_HPU_LAZY_MODE was not '1'; forcing it to 1 (HPU-graph capture "
          "is lazy-mode-only).", file=sys.stderr)
    os.environ["PT_HPU_LAZY_MODE"] = "1"


# ----------------------------- weight builders -------------------------------
# (identical layout to tests/test_fused_moe_routed.py so the oracle matches)
def _sym_zero_packed(groups, out_dim, device):
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)             # 0x88888888 (uint4 midpoint 8)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _pack_u4(codes_2d, device):
    import torch
    IN, OUT = codes_2d.shape
    c = (codes_2d.to(torch.int64) & 0xF).reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.to(device)


def _make_expert(H, I, GS, seed, device):
    import torch
    OUT13 = 2 * I
    g = torch.Generator().manual_seed(seed)
    c13 = torch.randint(0, 16, (H, OUT13), generator=g, dtype=torch.int32)
    c2 = torch.randint(0, 16, (I, H), generator=g, dtype=torch.int32)
    w13_packed = _pack_u4(c13, device)
    w2_packed = _pack_u4(c2, device)
    s13 = (torch.rand((H // GS, OUT13), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    s2 = (torch.rand((I // GS, H), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    z13 = _sym_zero_packed(H // GS, OUT13, device)
    z2 = _sym_zero_packed(I // GS, H, device)
    return w13_packed, s13, z13, w2_packed, s2, z2


def _build_stacked(E, H, I, GS, device):
    """Stack E experts into the [E,...] tensors fused_moe_routed consumes (the same
    stacked layout the serve stashes on `layer` and passes to _fused_moe_routed)."""
    import torch
    w13p_l, s13_l, z13_l, w2p_l, s2_l, z2_l = [], [], [], [], [], []
    for e in range(E):
        w13p, s13, z13, w2p, s2, z2 = _make_expert(H, I, GS, 100 + e, device)
        w13p_l.append(w13p); s13_l.append(s13); z13_l.append(z13)
        w2p_l.append(w2p); s2_l.append(s2); z2_l.append(z2)
    return (torch.stack(w13p_l, 0), torch.stack(s13_l, 0), torch.stack(z13_l, 0),
            torch.stack(w2p_l, 0), torch.stack(s2_l, 0), torch.stack(z2_l, 0))


def _ref_routed(x, w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s,
                topk_ids, topk_weights, experts_min, I):
    """INDEPENDENT per-expert convert_from_uint4 + MoE oracle (no gather/topk/remap),
    summed over the routed LOCAL experts with per-(token,slot) router weights."""
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch.core as htorch
    E = w13p_s.shape[0]
    T, H = x.shape
    dt = s13_s.dtype
    emin = int(experts_min)
    out = torch.zeros(T, H, dtype=dt, device=x.device)
    W13 = [torch.ops.hpu.convert_from_uint4(w13p_s[e], s13_s[e], z13_s[e], dt, None) for e in range(E)]
    W2 = [torch.ops.hpu.convert_from_uint4(w2p_s[e], s2_s[e], z2_s[e], dt, None) for e in range(E)]
    htorch.mark_step()
    expert_out = []
    for e in range(E):
        gate_up = torch.matmul(x, W13[e])
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        h = F.silu(gate) * up
        expert_out.append(torch.matmul(h, W2[e]))
    tids = topk_ids.to(torch.int64).cpu()
    tws = topk_weights.to(dt).cpu()
    for t in range(T):
        for slot in range(tids.shape[1]):
            gid = int(tids[t, slot].item())
            le = gid - emin
            if 0 <= le < E:
                out[t] = out[t] + expert_out[le][t] * tws[t, slot].to(x.device)
    return out


def _relnorm(got, ref):
    g = got.float(); r = ref.float()
    diff = (g - r).norm().item()
    return diff / (r.norm().item() + 1e-9), (g - r).abs().max().item()


def _find_single_expert_so():
    """The single-expert op .so (lazy variant) — the wrapper's per-expert loop needs it,
    AND it must be loaded so the routed .so can co-load (TORCH_LIBRARY_FRAGMENT)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ob = os.path.join(root, "tpc_kernel", "opbackend", "build")
    for name in ("libmoe_fused_expert_lazy.so", "libmoe_fused_expert.so"):
        p = os.path.join(ob, name)
        if os.path.isfile(p):
            return p
    return os.path.join(ob, "libmoe_fused_expert_lazy.so")


def _diagnose_fatal(msg: str, bucket: str = "", zp_inforward: bool = False):
    """LOUD diagnosis when the capture FATAL fires."""
    print("\n" + "=" * 78, file=sys.stderr)
    print(">>> FAIL (CHECK A): the captured fused-MoE graph crashed at capture/replay"
          + (f" [bucket {bucket}]" if bucket else "") + ".", file=sys.stderr)
    print("    %s" % msg.strip().splitlines()[-1] if msg.strip() else "    (no message)",
          file=sys.stderr)
    if zp_inforward:
        print(
            "    DIAGNOSIS: this is the ZP multi-bucket FATAL (HPU_MOE_TEST_ZP_INFORWARD=1).\n"
            "    The stacked GPTQ-sym zero-point was built IN-FORWARD via\n"
            "    expand(E,...).contiguous(); that .contiguous() runs INSIDE the captured\n"
            "    graph, so the fused op's zp index_select SOURCE is an in-graph intermediate\n"
            "    (an output-not-input) the free_inplace pass reclaims across buckets. The\n"
            "    1st bucket's replay frees it -> the 2nd bucket's capture validates an EMPTY\n"
            "    zp `hpu__input` (Strided Params Has Value: 0). This is the EXACT pre-fix\n"
            "    serve bug. FIX: _alloc_fused_zp materializes the stacked zp as a HELD LEAF\n"
            "    at weight-load (OUTSIDE capture) -> a persistent recipe input surviving every\n"
            "    bucket, like the weight nn.Parameters. Drop HPU_MOE_TEST_ZP_INFORWARD to use it.",
            file=sys.stderr)
    else:
        print(
            "    DIAGNOSIS: this is the wrap_in_hpu_graph MULTI-BUCKET capture 'empty input'\n"
            "    class (ValidateSyncInputTensors / 'Strided Params Has Value: 0') on\n"
            "    model/N/mlp/experts/hpu__input == x, the experts-input activation.\n"
            "    x arrives as an `aten::view` of a NON-persistent decode intermediate; the\n"
            "    single-recipe op roots its FIRST recipe input at that view's BASE. The K\n"
            "    weights are persistent (index_select of nn.Parameters), but x's base is\n"
            "    freed at the FIRST bucket's replay (SetTensorDataNullOpt), so the SECOND\n"
            "    bucket's on-the-fly capture validates an empty x.\n"
            "    FIX: route x through a PERSISTENT static-input buffer x_buf (allocated once,\n"
            "    OUTSIDE capture; x_buf[:T].copy_(x) then feed x_buf[:T] to the op) so the op\n"
            "    roots its x input at a HELD LEAF that survives every bucket (the canonical\n"
            "    HPU-graph static-input pattern; HPU_MOE_FUSED_X_BUFFER=1, the default). An\n"
            "    in-forward clone does NOT work: it is an in-graph intermediate the\n"
            "    free_inplace pass reclaims per bucket. If THIS run set HPU_MOE_FUSED_X_BUFFER=0\n"
            "    you dropped the buffer (clone fallback) -- that is the multi-bucket reproduction.\n"
            "    (HPU_MOE_ROUTED_VIA_SINGLE_RECIPE=0 instead reproduces the OLDER per-expert\n"
            "    in-graph weight-clone crash.)",
            file=sys.stderr)
    print("=" * 78, file=sys.stderr)


def _is_capture_fatal(msg: str) -> bool:
    m = msg.lower()
    return ("validatesyncinputtensors" in m or "tensor_data is empty" in m
            or "strided params has value" in m or "neither storage attached" in m
            or ("storage" in m and "replay" in m))


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    E = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    TOP_K = E                       # T=1 -> route to all E local experts (full K gather)

    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H={H} I={I} GS={GS})")
        return 2
    if T * TOP_K > K:
        print(f"FATAL: this fast path needs T*top_k <= K (T={T} top_k={TOP_K} K={K}).")
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401  (loads the LAZY plugin)
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch as htorch
        from habana_frameworks.torch.utils.internal import is_lazy
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.",
              file=sys.stderr)
        return 2

    if not is_lazy():
        print("FATAL: is_lazy() is False after import -- HPU-graph capture is "
              "lazy-mode-only and would RAISE. PT_HPU_LAZY_MODE=%r."
              % os.environ.get("PT_HPU_LAZY_MODE"), file=sys.stderr)
        return 2

    via = os.environ.get("HPU_MOE_ROUTED_VIA_SINGLE_RECIPE", "1") != "0"
    materialize_x = os.environ.get("HPU_MOE_FUSED_X_MATERIALIZE", "1") != "0"
    print(f"=== test_fused_routed_capture (MULTI-BUCKET) ===  lazy={is_lazy()}  "
          f"T={T} H={H} I={I} GS={GS} E={E} K={K}")
    print(f"    routed path = {'SINGLE-RECIPE op' if via else 'LEGACY per-expert in-graph clone loop (UNSAFE — reproduces weight-clone FATAL)'}")
    # (the 'x feed' line is printed below, once use_x_buf is known — the persistent
    #  static-input buffer is the PRIMARY multi-bucket fix; clone/raw-view are fallbacks)

    # ---- load the single-expert op .so (the wrapper's loop needs it; it also lets
    #      the routed .so co-load via TORCH_LIBRARY_FRAGMENT) ----
    se_so = _find_single_expert_so()
    print(f"single-expert op .so: {se_so}")
    if not os.path.isfile(se_so):
        print(f"FATAL: {se_so} not found. Build it (CPU-only):\n"
              "  docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work "
              "-w /work/opbackend vllm-gaudi:next bash build_lazy.sh", file=sys.stderr)
        return 2
    try:
        torch.ops.load_library(se_so)
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: could not load {se_so} under lazy: {ex}", file=sys.stderr)
        if "EagerOp" in str(ex) or "undefined symbol" in str(ex):
            print("    DIAGNOSIS: that is the EAGER-frontend .so; under lazy you need "
                  "libmoe_fused_expert_lazy.so.", file=sys.stderr)
        return 2
    if not (hasattr(torch.ops, "hpu_moe_ext")
            and hasattr(torch.ops.hpu_moe_ext, "fused_expert")):
        print("FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after load.",
              file=sys.stderr)
        return 2

    # the wrapper is the SUBJECT under test (it self-loads the single-recipe .so)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hpu_moe_ext.fused_expert_routed import fused_moe_routed

    dev = "hpu"
    torch.manual_seed(0)
    TOL = 1e-2
    rc = 0

    # ---- stacked [E,...] int4 weights + routing (route to all E local experts) ----
    w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s = _build_stacked(E, H, I, GS, dev)
    experts_min = 0
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    g = torch.Generator().manual_seed(7)
    tw = torch.rand(T, TOP_K, generator=g, dtype=torch.float32)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    topk_weights = tw.to(dev)
    topk_ids = (torch.arange(TOP_K, dtype=torch.int64).unsqueeze(0).expand(T, -1)
                + experts_min).contiguous().to(dev)                      # [T, top_k] global ids
    htcore.mark_step()

    # trusted reference (NOT under capture) — independent per-expert oracle
    ref = _ref_routed(x, w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s,
                      topk_ids, topk_weights, experts_min, I)
    htcore.mark_step()
    refc = ref.to("cpu")
    print(f"oracle |ref| = {refc.float().norm().item():.4f}")

    # --- tiny module reproducing the serve's MULTI-BUCKET trap ---------------------
    # The serve crash is on `model/N/mlp/experts/hpu__input` == x, the experts-input
    # activation, which is a DEEP `aten::view` of a model intermediate (apply_monolithic:
    # `x = x.view(-1, H)` off layer-0's post-attn norm), NOT a top-level user input. To
    # reproduce it the module must PRODUCE x INSIDE forward as a view of an in-graph
    # intermediate -- if we passed x as a forward ARG it would be a user input matched by
    # mark_user_inputs and survive, hiding the bug. So forward takes a base activation `h`
    # and a per-bucket "block" tensor `blk` (the bucket discriminator -- it stands in for
    # the attn block_list whose length sets num_blocks in the real bucket key). x is built
    # from h reduced by blk so its base is a NON-persistent intermediate that the prior
    # bucket's replay frees; and because blk's SHAPE differs between the two captures, the
    # wrap_in_hpu_graph input-hash differs -> a SECOND distinct cached bucket whose capture
    # is the on-the-fly one that reproduces the FATAL.
    class RoutedModule(torch.nn.Module):
        def __init__(self, emin, K, H, T, E, use_x_buf, zp_inforward,
                     z13_2d=None, z2_2d=None, z13_leaf=None, z2_leaf=None):
            super().__init__()
            self.emin = emin
            self.K = K
            self.E = E
            # a fixed per-feature gain (a stand-in for a norm weight); a buffer, NOT a
            # forward arg, so it is part of the captured recipe (persistent).
            self.register_buffer("gain",
                                 (torch.randn(H) * 0.0 + 1.0).to(torch.bfloat16))
            # PERSISTENT static-input x buffer (the multi-bucket FIX under test): allocated
            # ONCE here in __init__ — OUTSIDE any capture — so it is a held leaf (a recipe
            # INPUT), exactly as the serve allocates layer._hpu_fused_x_buf at weight-load.
            # The wrapper writes x_buf[:T].copy_(x) and feeds x_buf[:T] to the op, so each
            # bucket's recipe roots x at this SAME live storage (vs an in-forward clone,
            # which is an in-graph intermediate the free_inplace pass reclaims per bucket).
            self.x_buf = (torch.zeros(max(1, T), H, dtype=torch.bfloat16)
                          if use_x_buf else None)
            # === STACKED ZERO-POINT SOURCE (the LAST multi-bucket FIX under test) ===
            # The serve's _fused_moe_routed feeds the fused op a STACKED [E,...] zp it
            # index_select's on device. Two regimes, mirroring the serve before/after the
            # fix; the op's zp INPUT is the index_select SOURCE, so whether that source is
            # a held leaf or an in-forward `.contiguous()` decides multi-bucket survival:
            #   zp_inforward=True  (NEGATIVE CONTROL — the pre-fix serve bug): keep ONLY the
            #     2D base zp as a held buffer; BUILD the stacked [E,...] zp INSIDE forward via
            #     `expand(E,...).contiguous()`. That .contiguous() runs in the captured graph
            #     -> its output is an in-graph intermediate the free_inplace pass reclaims ->
            #     the index_select SOURCE is freed at bucket-1's replay -> bucket-2's capture
            #     validates an EMPTY zp `hpu__input` (the exact serve FATAL this targets).
            #   zp_inforward=False (the FIX — _alloc_fused_zp): the caller pre-built the
            #     stacked zp as a HELD LEAF (OUTSIDE capture); register it as a buffer so it is
            #     a persistent recipe input surviving every bucket (like the weight nn.Params).
            self.zp_inforward = zp_inforward
            if zp_inforward:
                self.register_buffer("z13_2d", z13_2d)   # [groups, 2I/8] held 2D base
                self.register_buffer("z2_2d", z2_2d)     # [groups, H/8]  held 2D base
            else:
                self.register_buffer("z13_zp", z13_leaf)  # [E, groups, 2I/8] held leaf
                self.register_buffer("z2_zp", z2_leaf)    # [E, groups, H/8]  held leaf

        def forward(self, h, blk, w13p, s13, w2p, s2, tid, tw):
            # blk -> a scalar bias that does NOT change x's VALUE (mean of a constant
            # blk is the same for both bucket sizes) but DOES make blk a real consumed
            # input so the two bucket shapes hash distinctly. Building x here makes it an
            # in-graph intermediate; .view(-1, H) makes the op's first input an aten::view
            # exactly as the serve's `model/N/mlp/experts/hpu__input`.
            bias = blk.to(torch.bfloat16).mean() * 0.0          # == 0, shape-independent
            xt = (h * self.gain + bias)                          # [T,H] intermediate
            x = xt.view(-1, H)                                   # aten::view -> the op input
            if self.zp_inforward:
                # NEGATIVE CONTROL: stacked zp built IN-GRAPH (== the pre-fix
                # _fused_moe_routed). Its index_select SOURCE is this .contiguous() output,
                # an in-graph intermediate -> freed across buckets -> 2nd-bucket FATAL.
                z13 = self.z13_2d.unsqueeze(0).expand(self.E, *self.z13_2d.shape).contiguous()
                z2 = self.z2_2d.unsqueeze(0).expand(self.E, *self.z2_2d.shape).contiguous()
            else:
                z13 = self.z13_zp                                # held leaf (persistent input)
                z2 = self.z2_zp
            return fused_moe_routed(x, w13p, s13, z13, w2p, s2, z2,
                                    tid, tw, self.emin, self.K, x_buf=self.x_buf)

    # use_x_buf: the persistent static-input buffer is the PRIMARY multi-bucket fix.
    # Default ON; HPU_MOE_FUSED_X_BUFFER=0 drops it -> the in-forward clone fallback
    # (single-bucket only) and the 2nd bucket should HIT the multi-bucket FATAL.
    use_x_buf = os.environ.get("HPU_MOE_FUSED_X_BUFFER", "1") != "0"
    print(f"    x feed      = {'PERSISTENT static-input buffer (multi-bucket FIX, approach B)' if use_x_buf else ('in-forward CLONE fallback (UNSAFE — reproduces multi-bucket FATAL)' if materialize_x else 'raw aten::view (UNSAFE — reproduces multi-bucket FATAL)')}")
    # ZP SOURCE regime (the LAST multi-bucket fix). Default = the FIX (persistent stacked-zp
    # held leaf, == _alloc_fused_zp). HPU_MOE_TEST_ZP_INFORWARD=1 = the NEGATIVE CONTROL: build
    # the stacked zp IN-FORWARD via expand().contiguous() (the pre-fix _fused_moe_routed) ->
    # its index_select source is freed across buckets -> 2nd-bucket FATAL on the zp `hpu__input`.
    zp_inforward = os.environ.get("HPU_MOE_TEST_ZP_INFORWARD", "0") == "1"
    print(f"    zp source   = {'IN-FORWARD expand().contiguous() (NEGATIVE CONTROL — reproduces the zp multi-bucket FATAL)' if zp_inforward else 'PERSISTENT stacked-zp held leaf (the _alloc_fused_zp FIX — multi-bucket-safe)'}")
    # 2D base zp (one expert's slice) — the shared GPTQ-sym zero the serve broadcasts to
    # [E,...]; build the persistent stacked leaf from it OUTSIDE capture (the fix), exactly
    # as _alloc_fused_zp does at weight-load. (z13_s/z2_s are already [E,...]; take [0] as the
    # 2D base and re-broadcast so the negative control's in-forward path is byte-faithful.)
    z13_2d = z13_s[0].contiguous()
    z2_2d = z2_s[0].contiguous()
    z13_leaf = z13_2d.unsqueeze(0).expand(E, *z13_2d.shape).contiguous()   # held [E,...] leaf
    z2_leaf = z2_2d.unsqueeze(0).expand(E, *z2_2d.shape).contiguous()
    mod = RoutedModule(experts_min, K, H, T, E, use_x_buf, zp_inforward,
                       z13_2d=z13_2d, z2_2d=z2_2d,
                       z13_leaf=z13_leaf, z2_leaf=z2_leaf).to(dev)

    from habana_frameworks.torch.hpu.graphs import CachedParams
    base_hits = sum(CachedParams.cache_hits.values())

    wrapped = htorch.hpu.wrap_in_hpu_graph(mod, verbose=True, log_frequency=1)

    # the base activation h whose `* gain` view IS x (so the oracle ref above, computed on
    # x, stays the reference): gain is all-ones, bias is 0, so xt == x bit-for-bit.
    h = x.clone()
    htcore.mark_step()

    # two decode buckets: blk length = num_blocks * BLOCK (a surrogate for the attn
    # block_list). Bucket 1 = num_blocks=1, bucket 2 = num_blocks=2 -> distinct input
    # hashes -> two cached recipes; bucket 2 is captured ON-THE-FLY (the make-or-break).
    BLOCK = 4
    buckets = [(1, torch.ones(1 * BLOCK, dtype=torch.int32, device=dev)),
               (2, torch.ones(2 * BLOCK, dtype=torch.int32, device=dev))]
    N = 4   # per bucket: 1 capture + 3 replays

    # ===================================================================
    # CHECK A — BOTH buckets capture + replay coherently (NO empty-input FATAL).
    #           Bucket 2's capture is the on-the-fly one the serve crashes on.
    # ===================================================================
    print("\n--- CHECK A: capture+replay TWO sequential decode buckets (no FATAL) ---")
    per_bucket_outs = {}
    for bi, (nblk, blk) in enumerate(buckets):
        tag = f"('decode',1,1,{nblk})"
        outs = []
        try:
            for i in range(N):
                o = wrapped(h, blk, w13p_s, s13_s, w2p_s, s2_s,
                            topk_ids, topk_weights)
                htcore.mark_step()
                outs.append(o.to("cpu"))
                kind = "CAPTURE" if i == 0 else "replay "
                print(f"  bucket {tag} {kind} call#{i} ok  |out|={outs[-1].float().norm().item():.4f}")
        except Exception as ex:  # noqa: BLE001
            msg = str(ex)
            if _is_capture_fatal(msg):
                _diagnose_fatal(msg, bucket=tag, zp_inforward=zp_inforward)
                if bi == 1:
                    print("    NOTE: this is the SECOND bucket -- exactly the live-serve "
                          "multi-bucket FATAL this test targets.", file=sys.stderr)
            else:
                print(f"\n>>> FAIL (CHECK A) [bucket {tag}]: unexpected error under "
                      f"capture: {ex}", file=sys.stderr)
            return 1
        per_bucket_outs[nblk] = outs

    a_ok = True
    for nblk, outs in per_bucket_outs.items():
        rn0, ma0 = _relnorm(outs[0], refc)
        ok = rn0 <= TOL
        print(f"  bucket ('decode',1,1,{nblk}) capture: relNorm={rn0:.4e} maxAbs={ma0:.4e}"
              f"  [{'ok' if ok else 'NUMERIC FAIL'}]")
        if not ok:
            a_ok = False
    if a_ok:
        print("  >>> PASS (A): BOTH buckets captured+executed, match oracle — no "
              "empty-input FATAL (incl. the 2nd, on-the-fly bucket).")
    else:
        print(f"  >>> FAIL (A-numeric): a bucket's capture exceeded tol {TOL:.2e}.",
              file=sys.stderr)
        rc = 1

    # ===================================================================
    # CHECK B — replays bit-identical + numerically correct, for BOTH buckets.
    # ===================================================================
    print("\n--- CHECK B: replays bit-identical to capture, all match oracle (both buckets) ---")
    b_ok = True
    for nblk, outs in per_bucket_outs.items():
        for i in range(1, N):
            d = (outs[i].float() - outs[0].float()).abs().max().item()
            rni, _ = _relnorm(outs[i], refc)
            flag = "ok" if (d == 0 and rni <= TOL) else "DIFF"
            print(f"  bucket ('decode',1,1,{nblk}) replay#{i}: maxAbs(vs capture)={d:.3e}  "
                  f"relNorm(vs oracle)={rni:.3e}  [{flag}]")
            if d > 0 or rni > TOL:
                b_ok = False
    if b_ok:
        print("  >>> PASS (B): every replay (both buckets) bit-identical to its capture "
              "and matches the oracle.")
    else:
        print("  >>> FAIL (B): a replay drifted from the capture or the oracle.",
              file=sys.stderr)
        rc = 1

    # ===================================================================
    # CHECK C — TWO recipes cached (one per bucket), 2nd+ calls are REPLAYS.
    # ===================================================================
    print("\n--- CHECK C: TWO recipes cached (one per bucket), 2nd+ calls REPLAY ---")
    hits = sum(CachedParams.cache_hits.values()) - base_hits
    n_recipes = None
    try:
        fv = wrapped.forward.__closure__
        names = wrapped.forward.__code__.co_freevars
        cache_dict = None
        for nm, cl in zip(names, fv or ()):
            if nm == "cache":
                cache_dict = cl.cell_contents
        n_recipes = len(cache_dict) if cache_dict is not None else None
    except Exception:  # noqa: BLE001
        n_recipes = None
    total_calls = N * len(buckets)
    expect_hits = total_calls - len(buckets)   # each bucket: 1 capture + (N-1) replays
    print(f"  cache_hits (this module) = {hits} over {total_calls} calls "
          f"(expect >= {expect_hits}: {len(buckets)} captures + {expect_hits} replays)")
    if n_recipes is not None:
        print(f"  distinct recipes cached  = {n_recipes} (expect {len(buckets)}: one per bucket)")
    replay_ok = (hits >= expect_hits)
    recipe_ok = (n_recipes is None) or (n_recipes == len(buckets))
    if replay_ok and recipe_ok:
        print("  >>> PASS (C): each bucket cached a distinct recipe and replayed it.")
    else:
        print(f"  >>> FAIL (C): replay/cache signal wrong (hits={hits}, recipes={n_recipes}, "
              f"expect recipes={len(buckets)}). recipes==1 => the 2nd bucket did NOT capture "
              "a new recipe (no multi-bucket coverage).", file=sys.stderr)
        rc = 1

    verdict = ("LOADS + CAPTURES + REPLAYS the routed fused MoE across MULTIPLE decode "
               "buckets cleanly (no empty-input FATAL)" if rc == 0
               else "did NOT survive multi-bucket capture cleanly — see above")
    print(f"\n[done] test_fused_routed_capture {'OK' if rc == 0 else 'FAILED'} (rc={rc}); "
          f"verdict: the wrapper {verdict}.")
    if rc == 0 and use_x_buf:
        print("       (multi-bucket-safe: x routed through the PERSISTENT static-input "
              "buffer x_buf — a held leaf allocated outside capture, so it survives every "
              "bucket; re-run with HPU_MOE_FUSED_X_BUFFER=0 to see the 2nd bucket crash on "
              "the in-forward clone.)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
