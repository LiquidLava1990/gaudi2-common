/*
 * dequant_guid_op.cpp  —  Path 1c (SECONDARY / for the FUSED subgraph phase).
 *
 * Binds a torch op  hpu_moe_ext::dequant_w4_guid  DIRECTLY to the PRE-BUILT TPC
 * GUID  "grouped_index_transpose_dequant_2d_with_zp_fwd_u4"  that already ships,
 * fully compiled (ELF/ISA embedded), inside /usr/lib/habanalabs/libtpc_kernels.so.
 *
 * This path needs NO gaudi2.bc (we are not compiling a kernel — the GUID's ISA is
 * resident). It exists so the later FUSED int4->(bf16|fp8)->MME MoE can emit this
 * exact dequant node into a custom Synapse subgraph (instead of the high-level
 * convert_from_uint4 op), and to confirm UserCustomOpDescriptor can target a stock
 * GUID.
 *
 * !!! HONEST RISK — why dequant_intrinsic.py (Path 1a, convert_from_uint4) is the
 *     RECOMMENDED runnable path and THIS is secondary/experimental:
 *   1. The kernel's user-params struct is NOT in the public perf_lib_layer_params.h
 *      (only ns_GroupedNonZero is). We pass NO params (size 0) below and rely on the
 *      tensor geometry; if the GUID's GetGcDefinitions REQUIRES a scalar params blob
 *      (e.g. a transpose axis / group count), instantiation will fail LOUDLY at
 *      graph-compile and you must supply the struct (recover it by disassembling
 *      tpc_kernels::GroupedIndexTransposeDequantKernel::GetGcDefinitions).
 *   2. The "grouped ... with group index" lowering in the stock backend builds a
 *      group_index tensor and asserts input+scale are 2D and out is bf16-only. A
 *      hand-emitted node must match that input layout (packed u4 [in,out//8], scale
 *      [groups,out] bf16, zp [groups,out//8]); the OutputMeta below mirrors it.
 *   So: use this ONLY after the on-HPU smoke below passes. The bf16/fp8 *foundation*
 *   ships via Path 1a regardless.
 *
 * Build: tpc_kernel/build_dequant.sh  (CPU-only docker run; host x86 g++, NOT
 *        tpc-clang — no bitcode involved).
 * Load:  dequant_guid_op.py.
 */
#include "hpu_custom_op_pt2.h"

#include <torch/library.h>
#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

// out is bf16 [in, out] where weight_packed is int32 [in, out//8] (8 u4 / int32).
habana::PartialOutputMetaDataVector OutputMeta(const at::Stack& stack) {
  const at::Tensor& weight_packed = stack.at(0).toTensor();
  TORCH_CHECK(weight_packed.dim() == 2,
              "dequant_w4_guid: weight_packed must be 2D [in, out//8], got dim=",
              weight_packed.dim());
  const int64_t in_dim = weight_packed.size(0);
  const int64_t out_dim = weight_packed.size(1) * 8;
  habana::PartialOutputMetaData meta;
  meta.dtype = at::ScalarType::BFloat16;  // grouped GUID is bf16-only
  meta.shape = {in_dim, out_dim};
  return {meta};
}

// The stock grouped dequant kernel takes its grouping from tensor geometry; we pass
// an EMPTY user-params blob. If the GUID rejects this at instantiation, recover its
// params struct (see the risk note above) and fill it here.
std::shared_ptr<void> FillParamsEmpty(const at::Stack& /*stack*/, size_t& size) {
  size = 0;
  return nullptr;
}

}  // namespace

TORCH_LIBRARY(hpu_moe_ext_guid, m) {
  // inputs mirror convert_from_uint4's grouped path operands:
  //   weight_packed int32 [in, out//8], weight_scale 2D [groups, out] (bf16),
  //   zero_point int32 [groups, out//8] (packed sym 8)
  m.def(
      "dequant_w4_guid(Tensor weight_packed, Tensor weight_scale, "
      "Tensor zero_point) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext_guid, HPU, m) {
  m.impl("dequant_w4_guid", [](const at::Tensor& weight_packed,
                               const at::Tensor& weight_scale,
                               const at::Tensor& zero_point) -> at::Tensor {
    static const bool registered = []() {
      habana::custom_op::registerUserCustomOp(
          "hpu_moe_ext_guid::dequant_w4_guid",
          // PRE-BUILT GUID resident in libtpc_kernels.so (no gaudi2.bc needed).
          "grouped_index_transpose_dequant_2d_with_zp_fwd_u4",
          OutputMeta,
          FillParamsEmpty);
      return true;
    }();
    (void)registered;

    std::vector<c10::IValue> inputs{weight_packed, weight_scale, zero_point};
    habana::custom_op::UserCustomOpDescriptor desc =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
            "hpu_moe_ext_guid::dequant_w4_guid");
    auto out = desc.execute(inputs);
    return out.at(0);
  });
}
