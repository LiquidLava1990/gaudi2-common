/*
 * moe_fused_monolithic_fp8_backend.cpp -- EAGER frontend for the MONOLITHIC (ALL-E
 * experts) single-recipe fused M3 swigluoai-FP8 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8`. Mirrors
 * moe_fused_monolithic_backend.cpp (the int4 eager TU) byte-for-byte except the schema
 * (4 fp8 weight inputs + alpha/limit floats instead of 6 int4 inputs) and the symbol
 * names. The recipe lives in the frontend-free shared header
 * moe_fused_monolithic_fp8_backend_shared.h (-> emit_monolithic_experts_fp8).
 *
 * Build CPU-only:  bash build_monolithic_fp8.sh  -> build/libmoe_fused_monolithic_fp8.so
 * RUN ON THE HPU BOX ONLY.
 */
#include "moe_fused_monolithic_fp8_backend_shared.h"

#include "habana_eager/ops/eager_op.h"   // habana::eager::EagerOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <utility>
#include <vector>

namespace habana {

static at::Tensor fused_moe_monolithic_fp8_hpu(
    const at::Tensor& x,
    const at::Tensor& w13_fp8, const at::Tensor& w13_scale,
    const at::Tensor& w2_fp8,  const at::Tensor& w2_scale,
    const at::Tensor& router_weights, int64_t chunk,
    double alpha, double limit) {
  std::vector<std::vector<int64_t>> out_shapes{x.sizes().vec()};  // [T,H]
  std::vector<at::IValue> inputs{
      x,
      w13_fp8, w13_scale,
      w2_fp8,  w2_scale,
      router_weights, chunk, alpha, limit};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_moe_monolithic_fp8", std::move(inputs), std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
  return hpu_op.call();
}

} // namespace habana

TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_monolithic_fp8(Tensor x, "
      "Tensor w13_fp8, Tensor w13_scale, "
      "Tensor w2_fp8, Tensor w2_scale, "
      "Tensor router_weights, int chunk, float alpha, float limit) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_monolithic_fp8", habana::fused_moe_monolithic_fp8_hpu);
}

static const auto& MoeFusedMonolithicFp8KernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_monolithic_fp8", habana::MoeFusedMonolithicFp8);
