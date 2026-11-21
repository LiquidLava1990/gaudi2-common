/*
 * moe_fused_grouped_fp8_backend_lazy.cpp -- LAZY frontend for the GROUPED (routed
 * top-k) M3 swigluoai-FP8 MoE op `torch.ops.hpu_moe_ext.fused_moe_grouped_fp8`.
 *
 * Mirrors moe_fused_monolithic_fp8_backend_lazy.cpp: the eager EagerOp frontend fails
 * to LOAD under the :next lazy plugin (its EagerOpBase symbols live in the eager lib);
 * the habana_lazy::LazyOp frontend loads against the lazy plugin and emits the ONE
 * grouped recipe into the lazy IR (so the HPU-graph capture records it). The recipe body
 * (MoeFusedGroupedFp8::AddNode) + MoeFusedGroupedFp8Meta come UNCHANGED from the shared
 * header; only the schema name + the 5-tensor signature differ.
 *
 * Build CPU-only:  BUILD_LAZY=1 bash build_grouped_fp8.sh
 *   -> build/libmoe_fused_grouped_fp8_lazy.so
 */
#include "moe_fused_grouped_fp8_backend_shared.h"

#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <utility>
#include <vector>

namespace habana_lazy {
// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;`: the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified (in namespace
// habana), and MoeFusedGroupedFp8Meta is in namespace habana (shared header).
using namespace habana;

static at::Tensor fused_moe_grouped_fp8_smart_hpu(
    const at::Tensor& xg,
    const at::Tensor& w13_fp8, const at::Tensor& w13_scale,
    const at::Tensor& w2_fp8,  const at::Tensor& w2_scale,
    int64_t activation_mode, int64_t weight_layout,
    double alpha, double limit) {
  std::vector<at::IValue> inputs{
      xg,
      w13_fp8, w13_scale,
      w2_fp8,  w2_scale,
      activation_mode, weight_layout,
      alpha, limit};
  LazyOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_moe_grouped_fp8_smart", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedGroupedFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_grouped_fp8_smart, hpu_op);
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

} // namespace habana_lazy

// =============================================================================
// Registration (TORCH_LIBRARY_FRAGMENT so it co-loads with the other ops).
// =============================================================================
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
      habana_lazy::fused_moe_grouped_fp8_smart_hpu);
  m.impl("fused_moe_grouped_fp8", habana_lazy::fused_moe_grouped_fp8_hpu);
  m.impl(
      "fused_moe_grouped_glm_fp8",
      habana_lazy::fused_moe_grouped_glm_fp8_hpu);
}

static const auto& MoeFusedGroupedFp8SmartKernelRegistryLazy =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_grouped_fp8_smart",
        habana::MoeFusedGroupedFp8Smart);
