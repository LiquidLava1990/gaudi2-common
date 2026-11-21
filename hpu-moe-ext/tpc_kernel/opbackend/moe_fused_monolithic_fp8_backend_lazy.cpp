/*
 * moe_fused_monolithic_fp8_backend_lazy.cpp -- LAZY frontend for the MONOLITHIC M3
 * swigluoai-FP8 MoE op `torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8`.
 *
 * Mirrors moe_fused_monolithic_backend_lazy.cpp (the int4 lazy TU): the eager EagerOp
 * frontend fails to LOAD under the :next lazy plugin (its EagerOpBase symbols live in
 * the eager lib); the habana_lazy::LazyOp frontend loads against the lazy plugin and
 * emits the ONE monolithic recipe into the lazy IR (so the HPU-graph capture records
 * it). The recipe body (MoeFusedMonolithicFp8::AddNode) + MoeFusedMonolithicFp8Meta
 * come UNCHANGED from the shared header; only the schema name + the 4-fp8-input
 * signature differ from the int4 TU (no int4 packed/scale/zp -- fp8 + per-out scale).
 *
 * Build CPU-only:  BUILD_LAZY=1 bash build_monolithic_fp8.sh
 *   -> build/libmoe_fused_monolithic_fp8_lazy.so
 */
#include "moe_fused_monolithic_fp8_backend_shared.h"

#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <utility>
#include <vector>

namespace habana_lazy {
// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;`: the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified (in namespace
// habana), and MoeFusedMonolithicFp8Meta is in namespace habana (shared header).
using namespace habana;

// LAZY frontend: x + the 4 stacked [E,...] fp8 weight tensors (w13_fp8/s13, w2_fp8/s2)
// + the dense [E,T,1] router tensor + chunk + the swigluoai alpha/limit scalars. NO
// int4 zp/group-scale slots. LazyOp emits into the lazy IR; the schema name routes the
// lowering to MoeFusedMonolithicFp8 via KernelRegistry().
static at::Tensor fused_moe_monolithic_fp8_hpu(
    const at::Tensor& x,
    const at::Tensor& w13_fp8, const at::Tensor& w13_scale,
    const at::Tensor& w2_fp8,  const at::Tensor& w2_scale,
    const at::Tensor& router_weights, int64_t chunk,
    double alpha, double limit) {
  std::vector<at::IValue> inputs{
      x,
      w13_fp8, w13_scale,
      w2_fp8,  w2_scale,
      router_weights, chunk, alpha, limit};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_moe_monolithic_fp8", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedMonolithicFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_monolithic_fp8, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts; TORCH_LIBRARY_FRAGMENT so it co-loads with the other ops):
//   1. schema -> torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8
//   2. TORCH_LIBRARY_IMPL(HPU) -> the LAZY frontend
//   3. KernelRegistry().add    -> binds the schema name to MoeFusedMonolithicFp8
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_monolithic_fp8(Tensor x, "
      "Tensor w13_fp8, Tensor w13_scale, "
      "Tensor w2_fp8, Tensor w2_scale, "
      "Tensor router_weights, int chunk, float alpha, float limit) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_monolithic_fp8", habana_lazy::fused_moe_monolithic_fp8_hpu);
}

static const auto& MoeFusedMonolithicFp8KernelRegistryLazy =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_monolithic_fp8", habana::MoeFusedMonolithicFp8);
