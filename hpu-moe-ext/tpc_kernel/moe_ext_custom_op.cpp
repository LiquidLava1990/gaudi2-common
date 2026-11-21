/*
 * moe_ext_custom_op.cpp
 *
 * PT2 custom-op glue for the Phase-0 fused-MoE foundation kernel.
 *
 * Registers a torch operator `hpu_moe_ext::unpack_dequant_w4` that the Habana
 * graph compiler lowers to the TPC GUID `moe_ext_unpack_dequant_w4_<dtype>`
 * (defined in moe_ext_unpack_dequant_w4.c, compiled into a perf-lib .so on
 * GC_KERNEL_PATH).
 *
 * API:  habana::custom_op::UserCustomOpDescriptor{schema, guid, OutputMetaFn,
 *       FillParamsFn}  +  habana::custom_op::registerUserCustomOp(...)
 *       (declared in habana_frameworks/torch/include/hpu_custom_op_pt2.h,
 *        defined in libhabana_pytorch_backend.so).
 *
 * Build/link: see build.sh. Loaded at runtime by torch.ops.load_library(...) in
 * register.py.
 *
 * Oracle being matched (bit-exact target):
 *   torch.ops.hpu.convert_from_uint4(weight_packed[in,out/8] int32,
 *                                    weight_scale[groups,out] f16,
 *                                    zero_point[groups,out/8] int32,
 *                                    torch.bfloat16, None)  -> bf16 [in, out]
 */

#include "hpu_custom_op_pt2.h"

// Use torch/library.h (NOT torch/extension.h) so the glue does NOT pull in
// torch/python.h -> Python.h. This op is a pure dispatcher registration; it
// needs no pybind. (torch/extension.h dragged in Python headers and broke the
// headless build.)
#include <torch/library.h>
#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

// MUST byte-match struct MoeUnpackDequantW4Params in the .c kernel.
struct MoeUnpackDequantW4Params {
  int group_size;
  int zero_const;
};

// ---- output metadata -------------------------------------------------------
// out is bf16 [in, out] where:
//   weight_packed : int32 [in, out/8]  ->  in = shape[0], out = shape[1]*8
habana::PartialOutputMetaDataVector OutputMeta(const at::Stack& stack) {
  const at::Tensor& weight_packed = stack.at(0).toTensor();
  TORCH_CHECK(weight_packed.dim() == 2,
              "unpack_dequant_w4: weight_packed must be 2D [in, out/8], got dim=",
              weight_packed.dim());
  const int64_t in_dim = weight_packed.size(0);
  const int64_t out_dim = weight_packed.size(1) * 8;  // 8 uint4 / int32

  habana::PartialOutputMetaData meta;
  meta.dtype = at::ScalarType::BFloat16;
  meta.shape = {in_dim, out_dim};
  return {meta};
}

// ---- TPC user params -------------------------------------------------------
// stack: (weight_packed, weight_scale, zero_point, group_size:int, zero_const:int)
std::shared_ptr<void> FillParams(const at::Stack& stack, size_t& size) {
  HPU_PARAMS_STUB(MoeUnpackDequantW4Params);
  const int64_t group_size = stack.at(3).toInt();
  const int64_t zero_const = stack.at(4).toInt();
  params->group_size = static_cast<int>(group_size);
  params->zero_const = static_cast<int>(zero_const);
  return params;
}

}  // namespace

// ---- torch schema ----------------------------------------------------------
// zero_const >= 0  => GPTQ-symmetric shared zero (8), zero_point tensor ignored
//                     by the kernel (still passed for ABI/AWQ parity).
// zero_const  < 0  => read per-group zero_point (AWQ); reserved for a later phase.
TORCH_LIBRARY(hpu_moe_ext, m) {
  m.def(
      "unpack_dequant_w4(Tensor weight_packed, Tensor weight_scale, "
      "Tensor zero_point, int group_size, int zero_const) -> Tensor");
}

// ---- bind schema -> GUID + callbacks --------------------------------------
// The GUID suffix selects the kernel's scale/out dtype family. Phase-0 targets
// bf16 output with f16 scales; the registered GUID is the bf16 variant.
TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("unpack_dequant_w4", [](const at::Tensor& weight_packed,
                                 const at::Tensor& weight_scale,
                                 const at::Tensor& zero_point,
                                 int64_t group_size,
                                 int64_t zero_const) -> at::Tensor {
    static const bool registered = []() {
      habana::custom_op::registerUserCustomOp(
          "hpu_moe_ext::unpack_dequant_w4",
          "moe_ext_unpack_dequant_w4_bf16",
          OutputMeta,
          FillParams);
      return true;
    }();
    (void)registered;

    std::vector<c10::IValue> inputs{weight_packed, weight_scale, zero_point,
                                    group_size, zero_const};
    // getUserCustomOpDescriptor returns a const&, but execute() is non-const.
    // Copy to a local mutable descriptor (cheap: small POD + std::function).
    habana::custom_op::UserCustomOpDescriptor desc =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
            "hpu_moe_ext::unpack_dequant_w4");
    auto out = desc.execute(inputs);
    return out.at(0);
  });
}
