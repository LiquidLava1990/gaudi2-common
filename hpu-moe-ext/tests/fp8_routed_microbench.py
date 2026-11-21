"""Microbench for the Track-A fp8-compose fused routed int4 MoE.

STANDALONE — synthetic GLM-5.1-shaped int4 weights, NO checkpoint needed. Run on
HPU by a human:

    PT_HPU_LAZY_MODE=1 python tests/fp8_routed_microbench.py

It does three things:
  (a) CORRECTNESS vs the bf16-intrinsic reference (convert_from_uint4 -> bf16 ->
      torch.ops.hpu.mixture_of_experts, i.e. the exact stock op math). Prints
      maxdiff + relative error and raises a LOUD SystemExit if the relative error
      is too large to be plausibly coherent.
  (b) LATENCY fp8 vs bf16-intrinsic at batch=1 over N iters: tok/s each + speedup.
  (c) HBM proxy: peak HPU allocator bytes per path if reachable, else latency-only.

----------------------------------------------------------------------------
CORRECTNESS THRESHOLD — WHY fp8 NEEDS A LOOSER BAR THAN 1e-2
----------------------------------------------------------------------------
The int4 routed path (HPU_MOE_ROUTED_VERIFY) is bit-EXACT vs the all-expert op
(same convert_from_uint4, same scales/zero), so it asserts md < 1e-2. This fp8
path CANNOT be that tight: the weight is quantized a SECOND time (int4 -> real ->
fp8 e4m3, only 3 mantissa bits ~ 2 decimal digits) under a single per-tensor
scale, AND the activations + the gate*up intermediate are fp8-quantized too. Each
fp8 cast injects ~2^-3 relative rounding; summed over the in-dimension (H=6144,
I=2048) the GEMM output carries a few-% relative error vs the bf16 reference. That
is EXPECTED and still coherent (fp8 inference is a known-good regime). A 1e-2 bar
would false-fail a correct kernel.

We therefore gate on RELATIVE error (Frobenius ||fp8 - ref|| / ||ref||):
    REL_TOL = 0.08   (8%)
Rationale: fp8 e4m3 per-element rel error ~6-12%; with random weights/acts the
errors partially cancel across the reduction, so the aggregate relative error of
a well-formed fp8 GEMM lands well under 8%. Anything ABOVE 8% indicates a real
bug (wrong scale convention, transpose, gather offset, dropped expert) rather than
fp8 rounding — that is what we want to catch LOUDLY. (Tighten toward ~0.05 once
the convention is hardware-confirmed; keep some headroom for the synthetic
worst-case uniform-random weights, which are harder for a per-tensor fp8 scale
than real trained weights.)
"""
import os
import time

import torch
import habana_frameworks.torch.core as htorch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk,
    unpack_quantized_values_into_int32 as up,
)
from vllm.scalar_type import scalar_types

# Import the unit under test.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpu_moe_ext.fp8_routed_moe import fp8_routed_moe, FP8_DTYPE  # noqa: E402

QT = scalar_types.uint4
DEV = "hpu"
torch.manual_seed(0)

# ---- GLM-5.1-int4-mixed-AutoRound shapes (per moe_wna16_hpu.py) ----
H = 6144            # hidden
I = 2048            # moe_intermediate (w13 out = 2*I = 4096; w2 in = I, out = H)
E = 32              # local experts per rank under EP
TOP_K = 8
GROUP = 128        # group_size
ZERO = 8           # GPTQ-symmetric uint4 midpoint
EXPERTS_MIN = 0     # single-rank bench: local window starts at 0

# bs=1 single-stream (the regime this win targets). top_k=8 distinct experts <= K.
T = 1
K = TOP_K          # static routed budget; T*TOP_K = 8 <= K guaranteed
N_ITERS = 50        # latency iters
N_WARMUP = 10

REL_TOL = 0.08      # see module docstring for the fp8 rationale


# ---- synthetic int4 expert weights, op-ready layout (mirrors moetest.py) ----
def _mk(out_, in_):
    """Random GPTQ-sym int4 weight. Returns:
      q     : [in, out] int4 codes (for the fp32 reference)
      packed: [in, out//8] int32 (op layout, what convert_from_uint4 consumes)
      scale : [groups, out] bf16 (op layout)
    """
    q = torch.randint(0, 16, (in_, out_), dtype=torch.int32)          # [in,out] codes
    gptq = pk(q, QT, packed_dim=0)                                    # [in//8, out]
    u8 = gptq.T.contiguous().view(torch.uint8)                       # [out, in//2] loaded layout
    # repack loaded uint8 -> op int32 [in, out//8] (same as _repack_weight)
    wi = u8.contiguous().view(torch.int32)                          # [out, in//8]
    vals = up(wi.transpose(0, 1).contiguous(), QT, packed_dim=0)     # [in, out]
    packed = pk(vals, QT, packed_dim=1)                            # [in, out//8]
    scale = (torch.rand(in_ // GROUP, out_, dtype=torch.bfloat16) * 0.02 + 0.005)  # [groups,out]
    return q, packed, scale


def _sym_zero(groups, out):
    return pk(torch.full((groups, out), ZERO, dtype=torch.int32), QT, packed_dim=1)


def _ref_W(q, scale):
    """FP32 reference dequant weight [in,out] = scale_bcast * (code - ZERO)."""
    sf = scale.float().repeat_interleave(GROUP, dim=0)              # [groups,out]->[in,out]
    return sf * (q.float() - float(ZERO))


def main():
    print("=" * 78)
    print("fp8-compose fused routed int4 MoE — microbench")
    print(f"  GLM shape: H={H} I={I} E={E} top_k={TOP_K} group={GROUP} "
          f"K={K} T={T}  fp8={FP8_DTYPE}")
    print("=" * 78)

    # build E experts
    q13s, p13s, s13s = [], [], []
    q2s, p2s, s2s = [], [], []
    for _e in range(E):
        q13, p13, s13 = _mk(2 * I, H)        # w13: out=2I, in=H
        q2, p2, s2 = _mk(H, I)              # w2 : out=H,  in=I
        q13s.append(q13); p13s.append(p13); s13s.append(s13)
        q2s.append(q2); p2s.append(p2); s2s.append(s2)

    w13_packed = torch.stack(p13s, 0).to(DEV)   # [E, H, (2I)//8]
    w2_packed = torch.stack(p2s, 0).to(DEV)     # [E, I, H//8]
    w13_scales = torch.stack(s13s, 0).to(DEV)   # [E, H//g, 2I]
    w2_scales = torch.stack(s2s, 0).to(DEV)     # [E, I//g, H]
    zp13 = _sym_zero(H // GROUP, 2 * I).to(DEV)  # shared symmetric zero (w13)
    zp2 = _sym_zero(I // GROUP, H).to(DEV)       # shared symmetric zero (w2)
    htorch.mark_step()

    # routing: T tokens, top_k distinct LOCAL experts, random weights renormalized
    perm = torch.randperm(E)[:TOP_K]
    topk_ids = (perm.view(1, TOP_K).repeat(T, 1) + EXPERTS_MIN).to(torch.int64).to(DEV)
    tw = torch.rand(T, TOP_K, dtype=torch.float32)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.bfloat16).to(DEV)
    x = torch.randn(T, H, dtype=torch.bfloat16).to(DEV) * 0.5
    htorch.mark_step()

    # ============================================================
    # (a) CORRECTNESS — bf16-intrinsic reference vs fp8 path
    # ============================================================
    # Reference: dequant ALL E experts to BF16 (out_dtype=bf16), exactly the stock
    # op, then torch.ops.hpu.mixture_of_experts. LOCAL ids (EXPERTS_MIN=0).
    def _bf16_ref():
        w13_deq = [torch.ops.hpu.convert_from_uint4(
            w13_packed[e], w13_scales[e], zp13, torch.bfloat16, None) for e in range(E)]
        w2_deq = [torch.ops.hpu.convert_from_uint4(
            w2_packed[e], w2_scales[e], zp2, torch.bfloat16, None) for e in range(E)]
        htorch.mark_step()
        return torch.ops.hpu.mixture_of_experts(
            hidden_states=x, expert_routing_table=topk_ids - EXPERTS_MIN,
            router_weights=tw, w12=w13_deq, w3=w2_deq,
            permuted_weights=True, activation="silu",
            experts_min=0, experts_max=E - 1)

    ref = _bf16_ref().float().cpu()
    htorch.mark_step()

    fp8_out = fp8_routed_moe(
        x, topk_ids, tw,
        w13_packed, w13_scales, w2_packed, w2_scales,
        zp13,                      # w13 shared symmetric zp
        K, activation="silu", experts_min=EXPERTS_MIN,
    ).float().cpu()
    htorch.mark_step()

    # We pass the w13-shaped zero (zp13) as the single shared `zp`, exactly like
    # production passes layer._hpu_routed_w13_zp. fp8_routed_moe recovers the constant
    # symmetric value from it and rebuilds the correctly-shaped w2 zero internally
    # (w13 out=2I and w2 out=H differ in width), so a single zp arg is sufficient.

    maxdiff = (fp8_out - ref).abs().max().item()
    rel = (fp8_out - ref).norm().item() / (ref.norm().item() + 1e-12)
    ref_norm = ref.norm().item()
    fp8_norm = fp8_out.norm().item()
    print("\n[correctness] bf16-intrinsic reference vs fp8-compose")
    print(f"  ref_norm = {ref_norm:.5f}   fp8_norm = {fp8_norm:.5f}")
    print(f"  maxdiff  = {maxdiff:.6g}")
    print(f"  REL err  = {rel:.5f}  (Frobenius ||fp8-ref||/||ref||)   tol={REL_TOL}")
    print(f"  ref[0,:6]= {[round(v, 4) for v in ref[0, :6].tolist()]}")
    print(f"  fp8[0,:6]= {[round(v, 4) for v in fp8_out[0, :6].tolist()]}")

    if not (ref_norm > 1e-6):
        raise SystemExit(
            f"\n*** FAIL: reference norm ~0 ({ref_norm:.3g}) — the bf16 ref "
            f"produced no signal; routing/weights are wrong, cannot judge fp8.")
    if rel > REL_TOL:
        raise SystemExit(
            f"\n*** FAIL: fp8 relative error {rel:.4f} > {REL_TOL} — too large to be "
            f"fp8 rounding. Suspect a BUG: scale convention (A_scale_inv/B_scale_inv "
            f"must take the FORWARD scale), a transpose (W is [in,out], trans_B=False), "
            f"a gather storage-offset (clone under lazy), or a dropped/mis-weighted "
            f"expert. maxdiff={maxdiff:.4g}.")
    print(f"  PASS: fp8 relative error {rel:.4f} <= {REL_TOL} (coherent fp8).")

    # ============================================================
    # (b) LATENCY — fp8 vs bf16-intrinsic at bs=1
    # ============================================================
    def _fp8_call():
        return fp8_routed_moe(
            x, topk_ids, tw, w13_packed, w13_scales, w2_packed, w2_scales,
            zp13, K, activation="silu", experts_min=EXPERTS_MIN)

    def _time(fn, n, warmup):
        import habana_frameworks.torch.hpu as _hpu
        for _ in range(warmup):
            fn(); htorch.mark_step()
        _hpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(); htorch.mark_step()
        _hpu.synchronize()
        return (time.perf_counter() - t0) / n

    bf16_s = _time(_bf16_ref, N_ITERS, N_WARMUP)
    fp8_s = _time(_fp8_call, N_ITERS, N_WARMUP)
    print("\n[latency] batch=1, %d iters (after %d warmup)" % (N_ITERS, N_WARMUP))
    print(f"  bf16-intrinsic (ALL {E} experts dequant): {bf16_s * 1e3:8.3f} ms/iter "
          f"-> {T / bf16_s:8.2f} tok/s")
    print(f"  fp8-compose (routed K={K}, fp8 GEMM)     : {fp8_s * 1e3:8.3f} ms/iter "
          f"-> {T / fp8_s:8.2f} tok/s")
    if fp8_s > 0:
        print(f"  SPEEDUP (bf16 / fp8)                     : {bf16_s / fp8_s:8.2f}x")
    # Fair routed-vs-routed note: the bf16 ref dequants ALL E experts (the stock
    # op's actual cost), so this speedup folds BOTH levers (routed-K gather + fp8).
    # That is the real end-to-end win at bs=1; the pure fp8-vs-bf16 GEMM ratio is a
    # subset of it.

    # ============================================================
    # (c) HBM proxy — peak HPU allocator bytes per path (best-effort)
    # ============================================================
    print("\n[hbm-proxy] peak HPU allocator bytes (proxy for weight traffic; "
          "latency above is the real signal)")
    try:
        import habana_frameworks.torch.hpu as _hpu
        def _peak(fn):
            _hpu.synchronize()
            if hasattr(_hpu, "reset_peak_memory_stats"):
                _hpu.reset_peak_memory_stats()
            fn(); htorch.mark_step(); _hpu.synchronize()
            if hasattr(_hpu, "max_memory_allocated"):
                return _hpu.max_memory_allocated()
            return _hpu.memory_stats().get("MaxInUse", 0)
        bf16_peak = _peak(_bf16_ref)
        fp8_peak = _peak(_fp8_call)
        print(f"  bf16-intrinsic peak alloc : {bf16_peak / 1e6:10.2f} MB")
        print(f"  fp8-compose    peak alloc : {fp8_peak / 1e6:10.2f} MB")
        if fp8_peak > 0:
            print(f"  ratio (bf16/fp8)          : {bf16_peak / max(fp8_peak,1):8.2f}x")
        print("  NOTE: allocator high-water, NOT literal bytes-moved. fp8 weights are "
              "half-size AND only K experts are dequant'd, so a lower peak corroborates "
              "the bandwidth win, but trust the latency speedup as the primary metric.")
    except Exception as ex:  # noqa: BLE001
        print(f"  (no reachable habana memory counter: {ex!r}) — latency-only.")

    print("\n" + "=" * 78)
    print("DONE — correctness PASSED, see latency/hbm above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
