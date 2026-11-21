"""dequant_intrinsic.py — GENERAL fused int4 GPTQ-symmetric dequant on Gaudi 2.

RECOMMENDED RUNNABLE PATH (Path 1a: reuse the pre-built GUID — no build, no
``gaudi2.bc``).

Evidence (gathered 2026-06-08 from the 1.24 dev image, CPU-only inspection):
  * ``libtpc_kernels.so`` ALREADY contains the fully-compiled Gaudi-2 ELF for the
    int4 dequant kernel family (the ISA is embedded; ``gaudi2.bc`` is a *compile*-
    time bitcode and is NOT needed to RUN these). Registered GUIDs include:
        grouped_index_transpose_dequant_2d_with_zp_fwd_u4   (code-zp)*scale, bf16
        grouped_index_transpose_dequant_2d_fwd_u4           code*scale, bf16
        convert_from_uint4_i32                              per-channel path (fp8/bf16)
        cast_i4_to_{bf16,f32,f8,hf8,i8}                     per-element casts
  * ``torch.ops.hpu.convert_from_uint4`` is the PUBLIC op that lowers to those
    GUIDs (recovered from libhabana_pytorch_backend.so::ConvertFromInt4::AddNode,
    which string-builds "grouped_index_transpose_dequant_fwd_u4" /
    "convert_from_uint4_i32" at runtime). Authoritative schema:

        hpu::convert_from_uint4(Tensor input, Tensor scale, Tensor? zero_point,
            ScalarType out_dtype, Tensor? group_index=None, *,
            bool disable_fp8_clipping=False) -> Tensor

    so it is ALREADY general: group_size is conveyed by the scale shape
    ``[groups, out]`` (groups = in // group_size), shapes are arbitrary, and an
    fp8 ``out_dtype`` is accepted. This is *the same op* the live WNA16 MoE and the
    GLM/DeepSeek/Nemotron/MiniMax linear GPTQ paths use, and it is the bit-exact
    oracle the rest of the stack is validated against.

Therefore the most robust runnable "torch-callable grouped int4 dequant" is simply
this op, wrapped here with the canonical ``[E,in,out//8]`` int32 weight +
``[E,groups,out]`` scale + GPTQ-symmetric zero=8 layout. It needs NOTHING built and
runs today on the box.

------------------------------------------------------------------------------
fp8 OUTPUT — READ THIS (the project already paid for this lesson)
------------------------------------------------------------------------------
``convert_from_uint4(out_dtype=float8_e4m3fn)`` is an ASYMMETRIC op and is BUGGY on
Gaudi-2: the HW fp8 element is e4m3*fnuz* (bias 8) while the DTYPE handed to the op
is e4m3*fn* (bias 7); int4->fp8 WRITES fp8 bits that the next read reinterprets at a
2x factor with nothing to cancel it (root-caused in hpu_moe_ext/fp8_routed_moe.py).
So the fp8-compute toggle here does NOT dequant straight to fp8. The supported,
proven fp8 path is:  int4 --convert_from_uint4(bf16)--> bf16 weight  --then a
self-consistent cast_to_fp8_v2 round-trip--> fp8 (the factor cancels at the GEMM).
``dequant_w4(..., out_dtype=fp8)`` below returns the bf16-then-cast result and is
NOT bit-exact vs the (broken) direct fp8 op — that is intentional and correct.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# Gaudi-2 e4m3 max comes from the TRUE HW format float8_e4m3fnuz (max 240.0), even
# though the DTYPE we hand the ops is float8_e4m3fn — exactly as the live
# vllm_gaudi path does (HW maps fn->fnuz; the fnuz max prevents cast overflow).
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(torch.float8_e4m3fnuz).max)

ZERO_CONST = 8  # GPTQ-symmetric uint4 shared zero


# --- direct GUIDs, for reference / the future fused Synapse subgraph -----------
# (These are the bare kernel GUIDs the public op lowers to. They are listed so the
# fused-MoE phase can emit them via synNodeCreate / a UserCustomOpDescriptor; see
# dequant_guid_op.py. Suffix _u4 = unsigned (GPTQ codes in [0,15]); _i4 = signed.)
GUID_GROUPED_ZP = "grouped_index_transpose_dequant_2d_with_zp_fwd_u4"
GUID_GROUPED = "grouped_index_transpose_dequant_2d_fwd_u4"


def _ensure_hpu_loaded() -> None:
    # Importing habana_frameworks.torch registers the hpu:: ops + HPU backend.
    import habana_frameworks.torch  # noqa: F401


def dequant_w4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    group_size: int = 128,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    zero_const: int = ZERO_CONST,
) -> torch.Tensor:
    """GENERAL grouped int4 GPTQ-symmetric dequant, per 2D weight tensor.

    Inputs (canonical checkpoint layout, NOT GLM-specific):
      weight_packed : int32 [in, out//8]   uint4-packed along out (8 nibbles/int32)
      weight_scale  : 2D    [groups, out]  groups = in // group_size  (f16/bf16/f32)
      zero_point    : int32 [groups, out//8] packed GPTQ-sym zero (==8), OR None to
                      synthesize it from ``zero_const`` (the op needs a real tensor).
      group_size    : int   rows-per-group along IN (e.g. 128 GLM / arbitrary).

    out_dtype:
      torch.bfloat16  -> direct, BIT-EXACT vs torch.ops.hpu.convert_from_uint4
                         (this IS that op; the grouped GUID is bf16-only).
      torch.float8_*  -> int4->bf16 (proven) THEN a self-consistent cast_to_fp8_v2
                         round-trip. Returns (fp8_weight, recovery_scale). NOT a
                         direct-to-fp8 dequant (that path is the known fnuz 2x bug).

    Returns bf16 [in, out]  (or, for fp8 out_dtype, the bf16-then-cast fp8 tensor;
    use dequant_w4_fp8 if you also need the recovery scale).
    """
    _ensure_hpu_loaded()
    assert weight_packed.dim() == 2, f"weight_packed must be 2D, got {weight_packed.shape}"
    assert weight_scale.dim() == 2, (
        "grouped dequant requires a 2D [groups, out] scale "
        f"(groups = in//group_size); got {tuple(weight_scale.shape)}"
    )
    in_dim = weight_packed.shape[0]
    groups = weight_scale.shape[0]
    assert in_dim % group_size == 0, f"in={in_dim} not divisible by group_size={group_size}"
    assert groups == in_dim // group_size, (
        f"scale groups={groups} != in//gs={in_dim // group_size}; check the scale layout "
        "(op wants [groups, out], i.e. scales.transpose(out,groups) if loaded [out,groups])"
    )

    if zero_point is None:
        zero_point = _sym_zero_packed(groups, weight_scale.shape[1], zero_const,
                                      weight_packed.device)

    # The grouped GUID is BF16-ONLY (backend: "Only BFloat16 output dtype is
    # supported for grouped dequantization with group index."). Always dequant to
    # bf16 here; fp8 is a downstream self-consistent cast.
    w_bf16 = torch.ops.hpu.convert_from_uint4(
        weight_packed, weight_scale, zero_point, torch.bfloat16, None
    )
    if out_dtype == torch.bfloat16:
        return w_bf16
    if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        fp8, _scale = _bf16_to_fp8(w_bf16)
        return fp8
    # any other out_dtype: honest plain cast off the bf16 (e.g. f32 for checking)
    return w_bf16.to(out_dtype)


def dequant_w4_fp8(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    group_size: int = 128,
    *,
    zero_const: int = ZERO_CONST,
):
    """fp8-compute toggle: returns (fp8_weight [in,out], recovery_scale scalar).

    Pass the returned scale as the GEMM's ``B_scale_inv`` (the FORWARD/recovery
    scale s, where real ~= fp8 * s). This is the per-tensor, self-consistent
    round-trip the live apply_fp8_linear_hpu path uses; the fnuz factor cancels at
    fp8_gemm_v2. (Per-group fp8 weight scales would need a block-scaled GEMM.)
    """
    w_bf16 = dequant_w4(weight_packed, weight_scale, zero_point, group_size,
                        out_dtype=torch.bfloat16, zero_const=zero_const)
    return _bf16_to_fp8(w_bf16)


def _bf16_to_fp8(x: torch.Tensor):
    """Per-tensor bf16 -> fp8 e4m3 with a recovery scale, mirroring dynamic_quant
    in vllm_gaudi.extension.ops. Returns (fp8, s) with real ~= fp8 * s."""
    amax = x.abs().max().to(torch.float32)
    s = (amax / FP8_MAX).clamp(min=1e-12)               # forward/recovery scale
    fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / s, False, False, FP8_DTYPE)[0]
    return fp8, s


def _sym_zero_packed(groups: int, out_dim: int, zero_const: int,
                     device) -> torch.Tensor:
    """Build the packed GPTQ-symmetric zero tensor int32 [groups, out//8] == zero_const.

    Mirrors moe_wna16_hpu._sym_zero: torch.full(zero_const) then pack 8 nibbles
    /int32 along out. zero_const==8 packs to 0x88888888.
    """
    assert out_dim % 8 == 0, f"out={out_dim} must be a multiple of 8 to pack uint4"
    word = 0
    for nib in range(8):
        word |= (int(zero_const) & 0xF) << (4 * nib)
    # 0x88888888 overflows int32 as a Python int sign; build via int32 view.
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


__all__ = [
    "dequant_w4",
    "dequant_w4_fp8",
    "GUID_GROUPED_ZP",
    "GUID_GROUPED",
    "FP8_DTYPE",
    "FP8_MAX",
    "ZERO_CONST",
]
