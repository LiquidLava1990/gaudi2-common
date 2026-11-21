"""test_p0_dequant.py — Phase-0 bit-exactness gate (RUN ON HPU, by a human).

Asserts the custom TPC op `hpu_moe_ext.unpack_dequant_w4` is BIT-EXACT
(maxdiff == 0) versus the oracle
    torch.ops.hpu.convert_from_uint4(weight_packed, weight_scale, zero_point,
                                     torch.bfloat16, None)
on GLM-5.1-shaped synthetic GPTQ-symmetric int4 weights.

This is the foundational correctness bar for the fused MoE: if the unpack+dequant
is not bit-identical, nothing built on top can be trusted.

  GLM-5.1 shapes (reference/model_configs/glm51_mixed_autoround_config.json):
    hidden=6144, moe_intermediate=2048, group_size=128, bits=4, sym (gptq)
    w13 expert weight: [in=6144, out=4096]   (gate||up = 2*2048)
    w2  expert weight: [in=2048, out=6144]   (down)

  Run (on the box, in the running serving image OR a dev image WITH HPU):
    export GC_KERNEL_PATH=$PWD/build/libmoe_ext_tpc_perflib.so:$GC_KERNEL_PATH
    python test_p0_dequant.py
"""
from __future__ import annotations

import sys

import torch

# The exact packer the live model uses (vllm core). Same import as moe_wna16_hpu.
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

import register  # local: loads the glue + binds the op

QT = scalar_types.uint4
GROUP_SIZE = 128
ZERO_CONST = 8  # GPTQ-symmetric uint4 bias (matches moe_wna16_hpu._sym_zero)


def make_synthetic(in_dim: int, out_dim: int, seed: int, device: str):
    """Build GPTQ-symmetric int4 packed weight + f16 scales + packed zeros,
    laid out EXACTLY as VllmMixtureOfExpertsOpWNA16Expert consumes them:
      weight_packed : int32 [in, out//8]   (uint4-packed along out)
      weight_scale  : f16   [groups, out]
      zero_point    : int32 [groups, out//8] (packed constant 8)
    """
    g = torch.Generator().manual_seed(seed)
    groups = in_dim // GROUP_SIZE
    assert in_dim % GROUP_SIZE == 0 and out_dim % 8 == 0

    # raw uint4 codes [in, out] in [0,15]
    codes = torch.randint(0, 16, (in_dim, out_dim), generator=g, dtype=torch.int32)
    # per-group f16 scales [groups, out], small positive (realistic gptq scale magnitudes)
    scales = (torch.rand((groups, out_dim), generator=g) * 0.02 + 1e-3).to(torch.float16)
    # symmetric zero = constant 8, packed [groups, out//8]
    zeros_full = torch.full((groups, out_dim), ZERO_CONST, dtype=torch.int32)

    weight_packed = pack_quantized_values_into_int32(codes, QT, packed_dim=1)      # [in, out//8]
    zero_point = pack_quantized_values_into_int32(zeros_full, QT, packed_dim=1)    # [groups, out//8]

    return (
        weight_packed.to(device),
        scales.to(device),
        zero_point.to(device),
    )


def run_case(tag: str, in_dim: int, out_dim: int, seed: int) -> bool:
    dev = "hpu"
    wp, sc, zp = make_synthetic(in_dim, out_dim, seed, dev)

    # ---- oracle ----
    ref = torch.ops.hpu.convert_from_uint4(wp, sc, zp, torch.bfloat16, None)  # [in,out] bf16

    # ---- custom kernel ----
    got = register.unpack_dequant_w4(wp, sc, zp, GROUP_SIZE, ZERO_CONST)      # [in,out] bf16

    ref_c = ref.to("cpu")
    got_c = got.to("cpu")
    if got_c.shape != ref_c.shape:
        print(f"[{tag}] SHAPE MISMATCH got={tuple(got_c.shape)} ref={tuple(ref_c.shape)}")
        return False
    if got_c.dtype != torch.bfloat16:
        print(f"[{tag}] DTYPE MISMATCH got={got_c.dtype} (want bfloat16)")
        return False

    # bit-exact: compare the raw bf16 bit patterns AND the numeric maxdiff.
    diff = (got_c.float() - ref_c.float()).abs()
    maxdiff = float(diff.max().item())
    bits_equal = torch.equal(got_c.view(torch.int16), ref_c.view(torch.int16))
    print(f"[{tag}] in={in_dim} out={out_dim} groups={in_dim//GROUP_SIZE} "
          f"maxdiff={maxdiff:.3e} bit_exact={bits_equal} "
          f"|ref|={ref_c.float().norm():.2f}")
    return bits_equal and maxdiff == 0.0


def main() -> int:
    register.load()
    cases = [
        # small, fast smoke
        ("smoke", 256, 64, 0),
        # GLM w13 (gate||up) expert weight
        ("glm_w13", 6144, 4096, 1),
        # GLM w2 (down) expert weight
        ("glm_w2", 2048, 6144, 2),
        # non-power-of-two out to stress nibble packing edges
        ("odd_out", 512, 128, 3),
    ]
    ok = True
    for tag, in_dim, out_dim, seed in cases:
        try:
            ok &= run_case(tag, in_dim, out_dim, seed)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag}] EXCEPTION: {type(ex).__name__}: {ex}")
            ok = False
    print("=" * 60)
    print("PHASE-0 RESULT:", "PASS (bit-exact)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
