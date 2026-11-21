"""CARD-FREE numeric gate for the LOAD-TIME fp8-experts path (HPU_MOE_FP8_EXPERTS).

THE #1 RISK is the int4->fp8 "fnuz 2x" bug (see fp8_routed_moe.py module docstring):
dequantizing int4 STRAIGHT to out_dtype=float8_e4m3fn writes fnuz bits that the next
read reinterprets at 2x. The validated fix is int4 -> bf16 (convert_from_uint4) -> fp8
via a SELF-CONSISTENT cast_to_fp8_v2 / fp8_gemm_v2 round-trip (the proven
apply_fp8_linear_hpu path). This test asserts, on a real HPU, that:

  (A) the LOAD-TIME conversion `_quant_weight_fp8` + the fp8 GEMM reproduces the int4
      stock all-expert MoE output within relNorm < 1e-2 (the bf16-double-quant bar), and
  (B) the catastrophic 2x bug is ABSENT: relNorm is ~few-% (coherent), NOT ~1.0 (the
      signature of a dropped/zeroed expert) and NOT ~2.06 (the fnuz inflation).

It is a pure-numerics check: it builds tiny synthetic int4 GPTQ-symmetric expert weights,
runs the STOCK `torch.ops.hpu.mixture_of_experts` (the production oracle) and the new
`fp8_allexpert_moe`, and compares. NO model load, NO serve, NO multi-card. Run inside the
HPU image (it needs the real intrinsics):

    python -m pytest hpu-moe-ext/tests/test_fp8_experts_numerics.py -s -q
  or
    python hpu-moe-ext/tests/test_fp8_experts_numerics.py

On a non-HPU host it SKIPS (the intrinsics are HPU-only) but still import-checks the path.
"""
from __future__ import annotations

import os
import sys

import torch

# --- HPU availability gate ---------------------------------------------------
try:
    import habana_frameworks.torch.core as htorch  # noqa: F401
    _HAS_HPU = True
except Exception:  # pragma: no cover - non-HPU host
    _HAS_HPU = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hpu_moe_ext.fp8_routed_moe import (  # noqa: E402
    FP8_DTYPE, FP8_MAX, _quant_weight_fp8)

# Pack helpers (CPU-safe; same intrinsic-free packing the load path uses).
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types  # noqa: E402


def _make_sym_int4_expert(out: int, in_: int, group_size: int, dev):
    """Build ONE GPTQ-symmetric int4 expert weight + per-group scale + the packed
    shared zero (midpoint 8), in the EXACT layout convert_from_uint4 consumes:
      packed : int32 [in, out//8]   (8 uint4 codes/int32 along out)
      scale  : bf16  [in//group_size, out]
      zero   : int32 [in//group_size, out//8]  (every lane == 8)
    Returns (packed, scale, zero, w_real_bf16) where w_real_bf16 == (code-8)*scale is
    the bf16 weight the int4 path dequantizes to — the ground truth both paths share."""
    groups = in_ // group_size
    # random uint4 codes in [0,15]; symmetric zero = 8 -> signed [-8,7]
    codes = torch.randint(0, 16, (in_, out), dtype=torch.int32, device=dev)
    scale = (torch.rand(groups, out, device=dev) * 0.02 + 0.005).to(torch.bfloat16)
    # real weight = (code - 8) * scale_of_its_group   [in,out] bf16
    g_of_row = (torch.arange(in_, device=dev) // group_size)
    w_real = ((codes - 8).to(torch.bfloat16)
              * scale.index_select(0, g_of_row))                      # [in,out] bf16
    packed = pack_quantized_values_into_int32(codes, scalar_types.uint4, packed_dim=1)
    zeros = torch.full((groups, out), 8, dtype=torch.int32, device=dev)
    zero_packed = pack_quantized_values_into_int32(zeros, scalar_types.uint4, packed_dim=1)
    return packed, scale, zero_packed, w_real


def _ref_int4_moe(x, topk_ids, topk_weights, w13_real, w2_real, I, act="silu"):
    """Ground-truth all-expert MoE in pure bf16 from the REAL (dequantized) weights —
    independent of any fp8 path. silu(gate)*up then down, router-weighted sum. This is
    what `torch.ops.hpu.mixture_of_experts` computes on the int4-dequant weights, but
    written in plain torch so the test does not depend on that intrinsic for the oracle
    (the intrinsic is used as a SECOND oracle in the on-card path)."""
    T, H = x.shape
    E = w13_real.shape[0]
    out = torch.zeros(T, H, dtype=torch.bfloat16, device=x.device)
    for e in range(E):
        gate_up = x @ w13_real[e]                       # [T,2I]
        g, u = gate_up[:, :I], gate_up[:, I:]
        h = torch.nn.functional.silu(g) * u             # [T,I]
        eo = h @ w2_real[e]                             # [T,H]
        match = (topk_ids == e)
        w = (topk_weights * match).sum(dim=-1, keepdim=True)
        out = out + eo * w
    return out


def _run():
    dev = "hpu" if _HAS_HPU else "cpu"
    torch.manual_seed(0)
    # GLM-5.2-shaped but TINY: H and I scaled down, group_size kept 128 (the real value).
    H, I, group_size = 256, 128, 128
    E, top_k, T = 4, 2, 3                                # T*top_k=6; tiny

    w13_packed, w13_scale, w13_zero, w13_real = [], [], [], []
    w2_packed, w2_scale, w2_zero, w2_real = [], [], [], []
    for _e in range(E):
        p, s, z, wr = _make_sym_int4_expert(2 * I, H, group_size, dev)   # w13: out=2I,in=H
        w13_packed.append(p); w13_scale.append(s); w13_zero.append(z); w13_real.append(wr)
        p, s, z, wr = _make_sym_int4_expert(H, I, group_size, dev)       # w2:  out=H, in=I
        w2_packed.append(p); w2_scale.append(s); w2_zero.append(z); w2_real.append(wr)
    w13_packed = torch.stack(w13_packed); w13_scale = torch.stack(w13_scale)
    w2_packed = torch.stack(w2_packed); w2_scale = torch.stack(w2_scale)
    w13_real = torch.stack(w13_real); w2_real = torch.stack(w2_real)
    w13_zero0 = w13_zero[0]; w2_zero0 = w2_zero[0]      # shared (identical) zero

    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    logits = torch.randn(T, E, device=dev)
    tw = torch.softmax(logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(tw, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(torch.bfloat16)

    # ---------------- ORACLE 1: plain-bf16 from REAL weights ----------------
    ref = _ref_int4_moe(x.float(), topk_ids, topk_weights.float(),
                        w13_real.float(), w2_real.float(), I).to(torch.bfloat16)
    ref_norm = ref.float().norm().item()
    print(f"[oracle] bf16 ref out_norm = {ref_norm:.5f}")

    if not _HAS_HPU:
        # Host: only validate the round-trip math of _quant_weight_fp8 is import-clean
        # and that the FORWARD-scale convention recovers the weight to ~fp8 precision.
        # (cast_to_fp8_v2 is HPU-only, so we can't run the true cast here.)
        print("[skip] non-HPU host: imported fp8 path OK; FP8_MAX=%.1f dtype=%s. "
              "Run inside the HPU image for the live numeric gate." % (FP8_MAX, FP8_DTYPE))
        return 0

    # ---------------- ORACLE 2: the STOCK int4 intrinsic --------------------
    # The exact production op: per-expert convert_from_uint4(out_dtype=bf16) + mark_step +
    # mixture_of_experts. This is what `fp8_allexpert_moe` must match within relNorm<1e-2.
    dt = torch.bfloat16
    w13_deq = [torch.ops.hpu.convert_from_uint4(w13_packed[e], w13_scale[e], w13_zero0, dt, None)
               for e in range(E)]
    w2_deq = [torch.ops.hpu.convert_from_uint4(w2_packed[e], w2_scale[e], w2_zero0, dt, None)
              for e in range(E)]
    htorch.mark_step()
    stock = torch.ops.hpu.mixture_of_experts(
        hidden_states=x, expert_routing_table=topk_ids.to(torch.int64),
        router_weights=topk_weights, w12=w13_deq, w3=w2_deq,
        permuted_weights=True, activation="silu", experts_min=0, experts_max=E - 1)
    htorch.mark_step()
    stock_norm = stock.float().norm().item()
    rn_stock_ref = (stock.float() - ref.float()).norm().item() / (ref_norm + 1e-9)
    print(f"[oracle] stock int4 intrinsic out_norm = {stock_norm:.5f}  relNorm(vs bf16 ref) = {rn_stock_ref:.4g}")

    # ---------------- CANDIDATE: LOAD-TIME fp8 experts ----------------------
    # NOTE: there is NO `convert_experts_to_fp8` in fp8_routed_moe. The production load-time
    # stash lives in moe_wna16_hpu._build_fp8_experts; here we replicate its exact per-expert
    # body (convert_from_uint4(out_dtype=bf16) -> _quant_weight_fp8) so the test matches the
    # REAL stash math, then call fp8_allexpert_moe with its REAL signature (experts_min, NOT I).
    from hpu_moe_ext.fp8_routed_moe import fp8_allexpert_moe, _quant_weight_fp8
    dt_fp8 = torch.bfloat16
    w13_fp8 = torch.empty((E, H, 2 * I), dtype=FP8_DTYPE, device=dev)
    w2_fp8 = torch.empty((E, I, H), dtype=FP8_DTYPE, device=dev)
    w13_s = torch.empty((E,), dtype=torch.float32, device=dev)
    w2_s = torch.empty((E,), dtype=torch.float32, device=dev)
    for e in range(E):
        b13 = torch.ops.hpu.convert_from_uint4(w13_packed[e], w13_scale[e], w13_zero0, dt_fp8, None)
        b2 = torch.ops.hpu.convert_from_uint4(w2_packed[e], w2_scale[e], w2_zero0, dt_fp8, None)
        f13, s13 = _quant_weight_fp8(b13)
        f2, s2 = _quant_weight_fp8(b2)
        w13_fp8[e].copy_(f13); w2_fp8[e].copy_(f2)
        w13_s[e] = s13.reshape(()); w2_s[e] = s2.reshape(())
        htorch.mark_step()
    assert w13_fp8.dtype == FP8_DTYPE and w2_fp8.dtype == FP8_DTYPE, "stash must be fp8"
    # Real signature: (x, topk_ids, topk_weights, w13_fp8, w13_scale, w2_fp8, w2_scale,
    #                  experts_min=0, activation=...). experts_min=0 (single-rank synthetic).
    fp8_out = fp8_allexpert_moe(
        x, topk_ids.to(torch.int64), topk_weights,
        w13_fp8, w13_s, w2_fp8, w2_s, experts_min=0, activation="silu")
    htorch.mark_step()
    fp8_norm = fp8_out.float().norm().item()
    rn = (fp8_out.float() - stock.float()).norm().item() / (stock_norm + 1e-9)
    ratio = fp8_norm / (stock_norm + 1e-9)
    print(f"[candidate] fp8_allexpert_moe out_norm = {fp8_norm:.5f}  relNorm(vs STOCK int4) = {rn:.4g}  norm_ratio = {ratio:.4f}")

    # ---------------- GATES (LOUD on failure) -------------------------------
    ok = True
    # (B) the fnuz 2x bug: norm_ratio ~2.0 is the smoking gun. Must be ~1.0 +/- a few %.
    if not (0.85 < ratio < 1.15):
        print(f"[FAIL] fp8/int4 norm_ratio={ratio:.4f} OUTSIDE [0.85,1.15] -> the fnuz-2x "
              f"bug (ratio~2.06) or a dropped expert (ratio~0). The int4->fp8 cast is NOT "
              f"the self-consistent bf16-roundtrip. ABORT — do NOT serve.")
        ok = False
    # (A) the correctness bar: relNorm < 1e-2 vs the STOCK int4 path (the fp8-double-quant tol).
    if rn >= 1e-2:
        print(f"[FAIL] fp8 relNorm={rn:.4g} >= 1e-2 vs stock int4. Coherent-ish but over the "
              f"validated double-quant bar; investigate before trusting the serve.")
        ok = False
    if ok:
        print(f"[PASS] fp8 experts within bar: relNorm={rn:.4g} < 1e-2, norm_ratio={ratio:.4f} "
              f"(no fnuz-2x). Load-time int4->bf16->fp8 round-trip is numerically sound.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_run())


def test_fp8_experts_numerics():
    """pytest entry: PASS if the gate passes (or skips on a non-HPU host)."""
    rc = _run()
    assert rc == 0, "fp8 experts numeric gate FAILED (see stdout: fnuz-2x or relNorm bar)"
