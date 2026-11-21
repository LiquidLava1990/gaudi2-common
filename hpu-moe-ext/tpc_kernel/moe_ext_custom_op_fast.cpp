/*
 * moe_ext_custom_op_fast.cpp
 *
 * PT2 custom-op glue for the OPTIMIZED dequant kernels (the FAST native-unpack and
 * the LEAN U4-descriptor variants in moe_ext_unpack_dequant_w4_fast.c, wrapped by
 * the perf-lib moe_ext_perflib_fast.cpp).
 *
 * WHY THIS FILE EXISTS (the load-bearing reason — read it):
 *   The on-card A/B MUST call the NEW kernels through a REAL torch op, NOT silently
 *   fall back to torch.ops.hpu.convert_from_uint4 (which would A/B the reference
 *   against ITSELF and FALSELY pass — the exact silent-pass the build-quality bar
 *   forbids). The existing moe_ext_custom_op.cpp is hardwired to the OLD GUID
 *   `moe_ext_unpack_dequant_w4_bf16`; it cannot reach the new GUIDs. This glue binds
 *   two NEW schemas to the two NEW GUIDs so the on-card script has a genuine,
 *   distinct kernel to compare against the oracle.
 *
 * NEW FILE. Does NOT modify moe_ext_custom_op.cpp / moe_ext_perflib.cpp / build.sh
 * or any file the in-flight int4-MoE / fp8 A/B uses. Built by build_fast.sh into
 * build_fast/libmoe_ext_custom_op_fast.so.
 *
 * Registered ops (both -> bf16 [in, out]; zero folded in-kernel, NO zero_point in):
 *
 *   hpu_moe_ext::unpack_dequant_w4_lean(weight_packed, weight_scale, group_size,
 *                                       zero_const) -> Tensor
 *       GUID  moe_ext_unpack_dequant_w4_lean_bf16
 *       layout IDENTICAL to the live op (U4 descriptor load, [in, out//8] packed):
 *         weight_packed : int32 [in, out//8]   (uint4-packed along out, 8/int32)
 *         weight_scale  : f16/bf16 [groups, out]
 *       -> the PROVEN low-risk mapping; this is the PRIMARY byte-exact on-card gate.
 *
 *   hpu_moe_ext::unpack_dequant_w4_fast(weight_packed, weight_scale, group_size,
 *                                       zero_const) -> Tensor
 *       GUID  moe_ext_unpack_dequant_w4_fast_bf16
 *       native int32 unpack (kills the 8x read amplification). Same TORCH-side
 *       signature as lean (weight_packed int32 [in, out//8]); the perf-lib's FAST
 *       access pattern walks the packed words (out//8 FCD) and emits the stride-8
 *       deinterleave internally. If GC rejects the FAST access pattern at
 *       instantiate it fails LOUDLY with a GLUE_* code and the LEAN op is the
 *       drop-in (its pattern is the proven one). This is the SECONDARY gate.
 *
 * Oracle being matched (bit-exact target, validated byte-exact CARD-FREE in
 * tests/dequant_fast_cpu_validate.py):
 *   torch.ops.hpu.convert_from_uint4(weight_packed[in,out/8] int32,
 *                                    weight_scale[groups,out] f16,
 *                                    zero_point[groups,out/8] int32,
 *                                    torch.bfloat16, None)  -> bf16 [in, out]
 *
 * Build/link: see build_fast.sh (Step 4). Loaded on the box via
 * torch.ops.load_library(build_fast/libmoe_ext_custom_op_fast.so).
 */

#include "hpu_custom_op_pt2.h"

// torch/library.h (NOT torch/extension.h) — pure dispatcher registration, no
// pybind / Python.h (matches the proven moe_ext_custom_op.cpp).
#include <torch/library.h>
#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

// MUST byte-match struct MoeUnpackDequantW4FastParams in the .c kernel + the shim.
struct MoeUnpackDequantW4FastParams {
  int group_size;
  int zero_const;
};

// out is bf16 [in, out] where weight_packed : int32 [in, out/8] (8 uint4 / int32).
// SAME meta for both variants (the FAST kernel's internal transpose is hidden; the
// torch-visible output shape is [in, out]).
habana::PartialOutputMetaDataVector OutputMeta(const at::Stack& stack) {
  const at::Tensor& weight_packed = stack.at(0).toTensor();
  TORCH_CHECK(weight_packed.dim() == 2,
              "unpack_dequant_w4_fast/lean: weight_packed must be 2D [in, out/8], "
              "got dim=", weight_packed.dim());
  const int64_t in_dim = weight_packed.size(0);
  const int64_t out_dim = weight_packed.size(1) * 8;  // 8 uint4 / int32
  TORCH_CHECK(out_dim % 8 == 0,
              "unpack_dequant_w4_fast/lean: out dim must be a multiple of 8, got ",
              out_dim);

  habana::PartialOutputMetaData meta;
  meta.dtype = at::ScalarType::BFloat16;
  meta.shape = {in_dim, out_dim};
  return {meta};
}

// stack: (weight_packed, weight_scale, group_size:int, zero_const:int)
// NOTE: only 4 inputs (no zero_point) — the new kernels fold the constant 8.
std::shared_ptr<void> FillParams(const at::Stack& stack, size_t& size) {
  HPU_PARAMS_STUB(MoeUnpackDequantW4FastParams);
  const int64_t group_size = stack.at(2).toInt();
  const int64_t zero_const = stack.at(3).toInt();
  params->group_size = static_cast<int>(group_size);
  params->zero_const = static_cast<int>(zero_const);
  return params;
}

}  // namespace

// ---- torch schemas (two new ops, 4 inputs each: NO zero_point) --------------
TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "unpack_dequant_w4_fast(Tensor weight_packed, Tensor weight_scale, "
      "int group_size, int zero_const) -> Tensor");
  m.def(
      "unpack_dequant_w4_lean(Tensor weight_packed, Tensor weight_scale, "
      "int group_size, int zero_const) -> Tensor");
}

// ---- bind each schema -> its GUID + callbacks -------------------------------
TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("unpack_dequant_w4_fast", [](const at::Tensor& weight_packed,
                                      const at::Tensor& weight_scale,
                                      int64_t group_size,
                                      int64_t zero_const) -> at::Tensor {
    static const bool registered = []() {
      habana::custom_op::registerUserCustomOp(
          "hpu_moe_ext::unpack_dequant_w4_fast",
          "moe_ext_unpack_dequant_w4_fast_bf16",
          OutputMeta,
          FillParams);
      return true;
    }();
    (void)registered;
    std::vector<c10::IValue> inputs{weight_packed, weight_scale,
                                    group_size, zero_const};
    habana::custom_op::UserCustomOpDescriptor desc =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
            "hpu_moe_ext::unpack_dequant_w4_fast");
    auto out = desc.execute(inputs);
    return out.at(0);
  });

  m.impl("unpack_dequant_w4_lean", [](const at::Tensor& weight_packed,
                                      const at::Tensor& weight_scale,
                                      int64_t group_size,
                                      int64_t zero_const) -> at::Tensor {
    static const bool registered = []() {
      habana::custom_op::registerUserCustomOp(
          "hpu_moe_ext::unpack_dequant_w4_lean",
          "moe_ext_unpack_dequant_w4_lean_bf16",
          OutputMeta,
          FillParams);
      return true;
    }();
    (void)registered;
    std::vector<c10::IValue> inputs{weight_packed, weight_scale,
                                    group_size, zero_const};
    habana::custom_op::UserCustomOpDescriptor desc =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
            "hpu_moe_ext::unpack_dequant_w4_lean");
    auto out = desc.execute(inputs);
    return out.at(0);
  });
}
