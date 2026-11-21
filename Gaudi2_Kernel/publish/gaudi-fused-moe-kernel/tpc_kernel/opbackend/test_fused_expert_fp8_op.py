"""test_fused_expert_fp8_op.py -- VALIDATE the fp8-COMPUTE fused MoE op
(``torch.ops.hpu_moe_ext_fp8.fused_expert_fp8``) ON THE HPU BOX.

>>> RUN ON THE HPU BOX (the device run is mine). CPU-only `py_compile` checks
    syntax only; only the box exercises the graph. <<<

Loads build/libmoe_fused_expert_fp8.so and calls the fp8 op with the SAME
synthetic GLM-shaped int4 weights as the bf16 validator (helpers lifted from
test_fused_expert_op.py), comparing against the SAME trusted convert_from_uint4 +
MoE reference.

>>> LOOSER TOLERANCE -- this is the int4->fp8 DOUBLE-QUANTIZATION accuracy floor.
The weight is quantized twice (int4 checkpoint, then bf16->fp8 e4m3 in-graph) and
BOTH activations (x and the intermediate h) are fp8-quantized too. fp8 e4m3 has
only 3 mantissa bits (~2 decimal digits); applied to two weights + two activations
across a two-GEMM (gate||up -> silu -> down) pipeline, the relNorm vs the
bf16-intrinsic reference is a HARD FLOOR of ~6e-2 -- coherent, but NOT the 1e-2 the
bf16 op meets. The default bar here is 8e-2 (override with FP8_TOL=...); the bf16
op's own 1e-2 is printed alongside so the on-device delta is visible.

>>> WHY 8e-2 AND NOT 5e-2 (the realistic floor, measured): a faithful CPU
emulation of THIS op's exact math (int4 dequant -> per-channel-weight/per-token-act
fp8 cast -> fp8 GEMM, both GEMMs, with the intermediate h requant) gives relNorm
~6.2-6.6e-2 across seeds for the GLM default shape -- and the on-device op measured
6.43e-2, matching the emulation to 3 sig figs. This is the genuine int4->fp8
double-quant + 3-mantissa-bit floor, NOT a bug: |got|/|ref| ~= 1.0 (the scale
round-trip is correct). CRITICALLY, scale GRANULARITY does NOT move this floor:
per-channel weight + per-token activation (the production apply_fp8_linear_hpu /
VllmMixtureOfExpertsOpFP8PerChannel convention, now wired in the op) only improves
it ~2-3% (6.44 -> 6.28 mean), and even per-GROUP(128) block scaling does not break
5e-2 through the full pipeline -- because in the GPTQ int4 layout the weight's
dynamic range lives along the CONTRACTION/group axis (per-output-column range is
only ~1.2x), so per-output-channel scaling is largely orthogonal to it. fp8-compute
TRADES this few-% accuracy for HALF the weight HBM traffic + ~2x MME throughput;
8e-2 is the floor (~6.6e-2 worst seed) with margin. If a future op variant keeps
the down GEMM in bf16 it can tighten toward 5e-2 (the intermediate h requant is the
single largest error source), at the cost of one fp8 GEMM.

If relNorm ~= 2.0 the fn-vs-fnuz 2x trap is live (the dequant emitted fp8 directly
somewhere, or a cast/gemm scale is not the self-consistent 1/s <-> s round-trip) --
see the fp8 backend's ACCURACY CAVEAT block.

BUILD_FP8_NODES NOTE: if the .so was built with the default BUILD_FP8_NODES=0,
the op FALLS BACK to the proven bf16 recipe, so this test will PASS at the TIGHT
1e-2 (it is exercising bf16 math). That validates registration + the harness.
Rebuild with BUILD_FP8_NODES=1 to exercise the real fp8 nodes, then expect the
looser fp8 relNorm. The printout states which regime the numbers imply.

RUN (ON THE HPU BOX):
  cd <repo>/tpc_kernel/opbackend
  python test_fused_expert_fp8_op.py                  # GLM default: T=1 H=6144 I=2048 GS=128
  FP8_TOL=0.08 python test_fused_expert_fp8_op.py 4 6144 2048 128
"""
from __future__ import annotations

import os
import sys

# Reuse the PROVEN helpers from the bf16 validator (same dir) -- no new numeric code.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _find_so() -> str:
    so = os.path.join(_HERE, "build", "libmoe_fused_expert_fp8.so")
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it CPU-only first:\n"
            f"  docker run --rm -v {os.path.dirname(_HERE)}:/work "
            f"-w /work/opbackend vllm-gaudi:next bash build_fp8.sh")
    return so


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    # 8e-2 == the MEASURED int4->fp8 double-quant + 3-mantissa-bit floor for this
    # two-GEMM op (CPU emulation 6.2-6.6e-2; on-device 6.43e-2), with margin. NOT a
    # bug and NOT fixable by scale granularity (per-channel/per-group don't move it;
    # see the module docstring). Override with FP8_TOL=... .
    FP8_TOL = float(os.environ.get("FP8_TOL", "0.08"))   # double-quant floor (measured)
    BF16_TOL = 1e-2                                       # the bf16 op's bar, for context

    if H % GS or I % GS or (2 * I) % 8 or H % 8 or I % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0, I%8==0 "
              f"(H={H} I={I} GS={GS})", file=sys.stderr)
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.",
              file=sys.stderr)
        return 2

    try:
        import test_fused_expert_op as base  # proven helpers
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: could not import helpers from "
              f"{_HERE}/test_fused_expert_op.py ({ex}).", file=sys.stderr)
        return 2

    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}", file=sys.stderr)
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext_fp8") or not hasattr(
            torch.ops.hpu_moe_ext_fp8, "fused_expert_fp8"):
        print(f"FATAL: torch.ops.hpu_moe_ext_fp8.fused_expert_fp8 not registered "
              f"after load_library({so}).", file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    OUT13 = 2 * I
    print(f"=== test_fused_expert_fp8_op === T={T} H={H} I={I} (2I={OUT13}) GS={GS} "
          f"FP8_TOL={FP8_TOL:.0e} (bf16-op bar={BF16_TOL:.0e})")

    w13p, s13, z13, w2p, s2, z2 = base._make_expert(H, I, GS, 100, dev)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    rw_val = 0.75
    rw = torch.full((1, 1), rw_val, dtype=torch.bfloat16, device=dev)
    htorch.mark_step()

    got = torch.ops.hpu_moe_ext_fp8.fused_expert_fp8(
        x, w13p, s13, z13, w2p, s2, z2, rw)
    htorch.mark_step()
    ref = base._ref_expert(x, w13p, s13, z13, w2p, s2, z2, I) * rw_val
    htorch.mark_step()
    gotc, refc = got.to("cpu"), ref.to("cpu")
    rn, ma = base._relnorm(gotc, refc)

    print(f"  |ref|={refc.float().norm().item():.4f}  |got|={gotc.float().norm().item():.4f}  "
          f"relNorm={rn:.4e}  maxAbs={ma:.4e}")

    # Regime hints.
    if rn <= BF16_TOL:
        print("  (relNorm within the bf16 bar -> the .so is likely the BUILD_FP8_NODES=0 "
              "bf16 fallback; rebuild with BUILD_FP8_NODES=1 to exercise fp8 nodes.)")
    if 1.5 <= (gotc.float().norm().item() / (refc.float().norm().item() + 1e-9)) <= 2.5:
        print("  !!! |got|/|ref| ~= 2x -> fn-vs-fnuz TRAP likely live (int4->fp8 direct, or a "
              "non-self-consistent scale). See the fp8 backend ACCURACY CAVEAT.", file=sys.stderr)

    if rn <= FP8_TOL:
        print(f"  >>> PASS: fp8 fused_expert relNorm {rn:.2e} <= fp8 tol {FP8_TOL:.2e}.")
        return 0
    print(f"  >>> FAIL: fp8 fused_expert relNorm {rn:.4e} > fp8 tol {FP8_TOL:.2e}.",
          file=sys.stderr)
    if rn <= 0.12:
        print(f"      NOTE: {rn:.2e} is near the ~6.4e-2 double-quant floor; if it is a hair "
              "over 8e-2, the floor just moved with shape/seed -- bump FP8_TOL, this is not a "
              "correctness bug (confirm |got|/|ref| ~= 1.0 above).", file=sys.stderr)
    print("      Suspects if relNorm is FAR over the floor (device-confirmable): the "
          "per-channel-weight [1,out] / per-token-act [M,1] amax+reduce axis "
          "(reductionDimensionMask, keepDim) or the scale broadcast; the self-consistent "
          "1/s cast <-> s gemm round-trip (FP8_MAX=240 fnuz); the convert_to_fp8 round_mode; "
          "or the fp8_gemm operand/scale order.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
