/*
 * moe_fused_grouped_fp8_backend.cpp -- EAGER frontend for the GROUPED (routed top-k)
 * single-recipe fused M3 swigluoai-FP8 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_grouped_fp8`. Mirrors
 * moe_fused_monolithic_fp8_backend.cpp byte-for-byte except the schema (5 tensors +
 * alpha/limit, NO router_weights, NO chunk) and the symbol names. The recipe lives in
 * the frontend-free shared header moe_fused_grouped_fp8_backend_shared.h
 * (-> emit_grouped_experts_fp8).
 *
 * Build CPU-only:  bash build_grouped_fp8.sh  -> build/libmoe_fused_grouped_fp8.so
 * RUN ON THE HPU BOX ONLY.
 */
#include "moe_fused_grouped_fp8_backend_shared.h"

#include "habana_eager/ops/eager_op.h"   // habana::eager::EagerOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <utility>
#include <vector>

namespace habana {

static at::Tensor fused_moe_grouped_fp8_smart_hpu(
    const at::Tensor& xg,
    const at::Tensor& w13_fp8, const at::Tensor& w13_scale,
    const at::Tensor& w2_fp8,  const at::Tensor& w2_scale,
    int64_t activation_mode, int64_t weight_layout,
    double alpha, double limit) {
  std::vector<std::vector<int64_t>> out_shapes{xg.sizes().vec()};  // [E,Cap,H]
  std::vector<at::IValue> inputs{
      xg,
      w13_fp8, w13_scale,
      w2_fp8,  w2_scale,
      activation_mode, weight_layout,
      alpha, limit};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_moe_grouped_fp8_smart",
      std::move(inputs), std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
  return hpu_op.call();
}

static at::Tensor fused_moe_grouped_fp8_hpu(
    const at::Tensor& xg,
    const at::Tensor& w13_fp8, const at::Tensor& w13_scale,
    const at::Tensor& w2_fp8,  const at::Tensor& w2_scale,
    double alpha, double limit) {
  return fused_moe_grouped_fp8_smart_hpu(
      xg, w13_fp8, w13_scale, w2_fp8, w2_scale,
      /*activation_mode=*/0, /*weight_layout=*/0, alpha, limit);
}

static at::Tensor fused_moe_grouped_glm_fp8_hpu(
    const at::Tensor& xg,
    const at::Tensor& w13_fp8, const at::Tensor& w13_scale,
    const at::Tensor& w2_fp8,  const at::Tensor& w2_scale,
    double limit) {
  return fused_moe_grouped_fp8_smart_hpu(
      xg, w13_fp8, w13_scale, w2_fp8, w2_scale,
      /*activation_mode=*/1, /*weight_layout=*/1,
      /*alpha unused by GLM=*/1.0, limit);
}

} // namespace habana

TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_grouped_fp8_smart(Tensor xg, "
      "Tensor w13_fp8, Tensor w13_scale, "
      "Tensor w2_fp8, Tensor w2_scale, "
      "int activation_mode, int weight_layout, "
      "float alpha, float limit) -> Tensor");
  m.def(
      "fused_moe_grouped_fp8(Tensor xg, "
      "Tensor w13_fp8, Tensor w13_scale, "
      "Tensor w2_fp8, Tensor w2_scale, "
      "float alpha, float limit) -> Tensor");
  m.def(
      "fused_moe_grouped_glm_fp8(Tensor xg, "
      "Tensor w13_fp8, Tensor w13_scale, "
      "Tensor w2_fp8, Tensor w2_scale, "
      "float limit) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl(
      "fused_moe_grouped_fp8_smart",
      habana::fused_moe_grouped_fp8_smart_hpu);
  m.impl("fused_moe_grouped_fp8", habana::fused_moe_grouped_fp8_hpu);
  m.impl(
      "fused_moe_grouped_glm_fp8",
      habana::fused_moe_grouped_glm_fp8_hpu);
}

static const auto& MoeFusedGroupedFp8SmartKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_grouped_fp8_smart",
        habana::MoeFusedGroupedFp8Smart);
