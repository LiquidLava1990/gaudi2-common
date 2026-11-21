"""test_p0_dequant_general.py — Phase-0 GENERAL bit-exact dequant gate (RUN ON HPU).

Validates the recommended runnable path (tpc_kernel/dequant_intrinsic.py:dequant_w4)
which reuses the PRE-BUILT GUID via torch.ops.hpu.convert_from_uint4 — no kernel
build, no gaudi2.bc.

What it proves, PARAMETERIZED over group_size + shapes (GLM, DeepSeek-ish,
Nemotron-ish, MiniMax-ish, odd-shape stressors) and BOTH output dtypes:

  bf16 out : dequant_w4(...) is BIT-EXACT (maxdiff==0, identical int16 bits) vs the
             oracle torch.ops.hpu.convert_from_uint4(..., torch.bfloat16, None).
             (It IS that op, so this is a self-consistency + layout/zero-synthesis
             gate — if the packed-zero synthesis or scale layout were wrong it would
             diverge.) Also cross-checked against a CPU reference (code-8)*scale.

  fp8 out  : dequant_w4(..., out_dtype=fp8) returns the int4->bf16->cast path. This
             is INTENTIONALLY NOT bit-exact vs a direct int4->fp8 op (that direct op
             has the known Gaudi-2 fnuz 2x bug; see dequant_intrinsic.py). We instead
             assert the fp8 result is *coherent* (high cosine + bounded relative
             error vs the true bf16 dequant) and round-trips self-consistently
             (fp8 * recovery_scale ~= bf16 weight). This is the correct bar for the
             fp8-compute toggle.

RUN (on the box — needs an HPU; CPU container cannot run device ops):

    cd $HOME/AI/hpu-moe-ext/tpc_kernel
    #   (A) inside the running serving image (has vllm for the canonical packer):
    #       docker exec -it vllm-active bash    # then:
    #       python test_p0_dequant_general.py
    #   (B) or the dev image WITH HPU (the test falls back to a local int32 packer
    #       if vllm is absent, so it runs in the dev image too):
    #       python test_p0_dequant_general.py

Exit code 0 == PASS.
"""
from __future__ import annotations

import sys

import torch

import dequant_intrinsic as dq

# ---- canonical uint4 packer: prefer the exact one the live model uses ---------
QT = None
try:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (  # type: ignore
        pack_quantized_values_into_int32 as _vllm_pack,
    )
    from vllm.scalar_type import scalar_types  # type: ignore

    QT = scalar_types.uint4

    def pack_u4(codes_2d: torch.Tensor) -> torch.Tensor:
        """[*, out] int32 codes in [0,15] -> int32 [*, out//8] (vllm canonical)."""
        return _vllm_pack(codes_2d, QT, packed_dim=1)

    PACKER = "vllm.pack_quantized_values_into_int32"
except Exception:  # dev image without vllm: local bit-identical packer

    def pack_u4(codes_2d: torch.Tensor) -> torch.Tensor:
        """Pack 8 uint4 nibbles (LSB-first) per int32 along the last dim.
        Bit-identical to vllm pack_quantized_values_into_int32(packed_dim=1)."""
        assert codes_2d.shape[-1] % 8 == 0
        c = codes_2d.to(torch.int64) & 0xF
        rows, cols = c.shape
        c = c.reshape(rows, cols // 8, 8)
        shifts = (4 * torch.arange(8, device=c.device, dtype=torch.int64))
        word = (c << shifts).sum(dim=-1)              # [rows, cols//8] in [0, 2^32)
        word = torch.where(word >= 2**31, word - 2**32, word)
        return word.to(torch.int32)

    PACKER = "local int32 packer (vllm not importable)"


GROUP_SIZE_DEFAULT = 128


def make_synthetic(in_dim: int, out_dim: int, group_size: int, seed: int, device: str):
    """GPTQ-symmetric int4 weight in the canonical [in,out//8]+[groups,out] layout."""
    g = torch.Generator().manual_seed(seed)
    assert in_dim % group_size == 0 and out_dim % 8 == 0
    groups = in_dim // group_size
    codes = torch.randint(0, 16, (in_dim, out_dim), generator=g, dtype=torch.int32)
    scales = (torch.rand((groups, out_dim), generator=g) * 0.02 + 1e-3).to(torch.float16)
    weight_packed = pack_u4(codes).to(device)
    scales = scales.to(device)
    # packed sym zero (==8) int32 [groups, out//8] — built by the wrapper too, but we
    # also pass it explicitly here to exercise the explicit-tensor path.
    zero_point = dq._sym_zero_packed(groups, out_dim, dq.ZERO_CONST, device)
    return weight_packed, scales, zero_point, codes.to(device)


def cpu_reference(codes_2d: torch.Tensor, scales_go: torch.Tensor,
                  group_size: int) -> torch.Tensor:
    """Independent CPU oracle: (code - 8) * scale, grouped along IN. bf16."""
    codes = codes_2d.to("cpu").to(torch.float32)
    sc = scales_go.to("cpu").to(torch.float32)
    grp = (torch.arange(codes.shape[0]) // group_size).long()
    return ((codes - dq.ZERO_CONST) * sc[grp]).to(torch.bfloat16)


def run_bf16(tag, in_dim, out_dim, group_size, seed) -> bool:
    dev = "hpu"
    wp, sc, zp, codes = make_synthetic(in_dim, out_dim, group_size, seed, dev)

    ref = torch.ops.hpu.convert_from_uint4(wp, sc, zp, torch.bfloat16, None)   # oracle
    got = dq.dequant_w4(wp, sc, zp, group_size, out_dtype=torch.bfloat16)      # ours
    # also exercise zero=None synthesis path (should be identical)
    got_synth = dq.dequant_w4(wp, sc, None, group_size, out_dtype=torch.bfloat16)

    ref_c, got_c, got_s_c = ref.to("cpu"), got.to("cpu"), got_synth.to("cpu")
    cpu_ref = cpu_reference(codes, sc, group_size)

    if got_c.shape != ref_c.shape or got_c.dtype != torch.bfloat16:
        print(f"[{tag}] BAD shape/dtype got={tuple(got_c.shape)}/{got_c.dtype} "
              f"ref={tuple(ref_c.shape)}")
        return False

    bits_equal = torch.equal(got_c.view(torch.int16), ref_c.view(torch.int16))
    synth_equal = torch.equal(got_s_c.view(torch.int16), ref_c.view(torch.int16))
    maxdiff = float((got_c.float() - ref_c.float()).abs().max())
    # CPU oracle agreement (bf16 rounding of the op may differ in the last bit from a
    # pure-fp32 CPU calc; report it but gate only on the device bit-exactness).
    cpu_maxdiff = float((ref_c.float() - cpu_ref.float()).abs().max())
    print(f"[{tag}:bf16] in={in_dim} out={out_dim} gs={group_size} "
          f"groups={in_dim // group_size} bit_exact={bits_equal} zeroNone_exact={synth_equal} "
          f"maxdiff={maxdiff:.3e} cpu_maxdiff={cpu_maxdiff:.3e} |ref|={ref_c.float().norm():.1f}")
    return bits_equal and synth_equal and maxdiff == 0.0


def run_fp8(tag, in_dim, out_dim, group_size, seed) -> bool:
    dev = "hpu"
    wp, sc, zp, _ = make_synthetic(in_dim, out_dim, group_size, seed, dev)
    bf16 = dq.dequant_w4(wp, sc, zp, group_size, out_dtype=torch.bfloat16)
    fp8, scale = dq.dequant_w4_fp8(wp, sc, zp, group_size)

    bf16_c = bf16.to("cpu").float()
    # recovered = fp8 * recovery_scale  (self-consistent round-trip)
    recovered = (fp8.float() * scale.float()).to("cpu")
    num = (recovered * bf16_c).sum()
    den = recovered.norm() * bf16_c.norm() + 1e-30
    cos = float(num / den)
    rel = float((recovered - bf16_c).norm() / (bf16_c.norm() + 1e-30))
    out_dtype_ok = fp8.dtype == dq.FP8_DTYPE
    # fp8 e4m3 has ~3 mantissa bits: a per-tensor scale over a wide output loses
    # relative precision; coherent bar (not bit-exact). cos>0.99, rel<0.10.
    ok = out_dtype_ok and cos > 0.99 and rel < 0.10
    print(f"[{tag}:fp8 ] in={in_dim} out={out_dim} gs={group_size} "
          f"dtype={fp8.dtype} cos={cos:.5f} rel_err={rel:.3e} scale={float(scale):.3e} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print(f"packer: {PACKER}")
    try:
        import habana_frameworks.torch  # noqa: F401
    except Exception as ex:
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    # (tag, in, out, group_size, seed) — GENERAL: never GLM-only.
    cases = [
        # ---- smoke ----
        ("smoke",        256,   64,  128, 0),
        # ---- GLM-5.1 (H=6144, I=2048, gs=128) ----
        ("glm_w13",      6144, 4096, 128, 1),   # gate||up = 2*2048
        ("glm_w2",       2048, 6144, 128, 2),   # down
        # ---- DeepSeek-ish (H=7168, 256-expert MoE, gs=128) ----
        ("ds_w13",       7168, 4096, 128, 3),
        ("ds_w2",        2048, 7168, 128, 4),
        # ---- Nemotron-ish wide hidden, gs=128 ----
        ("nemo_w13",     8192, 6144, 128, 5),
        # ---- MiniMax-ish, gs=64 (DIFFERENT group size — generality check) ----
        ("minimax_gs64", 6144, 4096,  64, 6),
        # ---- group_size=32 stressor ----
        ("gs32",         4096, 2048,  32, 7),
        # ---- odd/edge out width (non-pow2, still %8) to stress nibble packing ----
        ("odd_out",      512,  128,  128, 8),
        ("odd_out2",     1024, 1536, 128, 9),
    ]

    ok = True
    print("=" * 78)
    print("BF16 (bit-exact vs convert_from_uint4 oracle):")
    for tag, i, o, gs, s in cases:
        try:
            ok &= run_bf16(tag, i, o, gs, s)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag}:bf16] EXCEPTION {type(ex).__name__}: {ex}")
            ok = False
    print("-" * 78)
    print("FP8 (coherent + self-consistent round-trip; NOT bit-exact by design):")
    fp8_cases = [c for c in cases if c[0] in ("smoke", "glm_w13", "glm_w2", "ds_w13", "minimax_gs64")]
    for tag, i, o, gs, s in fp8_cases:
        try:
            ok &= run_fp8(tag, i, o, gs, s)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag}:fp8 ] EXCEPTION {type(ex).__name__}: {ex}")
            ok = False
    print("=" * 78)
    print("PHASE-0 GENERAL DEQUANT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
