/*
 * moe_fused_monolithic_int4fp8_backend_lazy.cpp -- LAZY-frontend variant of the W4A8
 * monolithic MoE op `torch.ops.hpu_moe_ext.fused_moe_monolithic_int4fp8`. Captures into
 * the decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1). Mirrors
 * moe_fused_monolithic_backend_lazy.cpp byte-for-byte except the schema name + the
 * shared header (which calls emit_monolithic_experts_i4f8 = int4->fp8 + fp8 MME).
 *
 * Build CPU-only: bash build_monolithic_int4fp8_lazy.sh (in vllm-gaudi:next, under
 * PT_HPU_LAZY_MODE=1) -> build/libmoe_fused_monolithic_int4fp8_lazy.so. RUN ON THE HPU BOX.
 */
#include "moe_fused_monolithic_int4fp8_backend_shared.h"

#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>

#include <utility>
#include <vector>

namespace habana_lazy {

using namespace habana;

static at::Tensor fused_moe_monolithic_int4fp8_hpu(
    const at::Tensor& x,
    const at::Tensor& w13_packed, const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed,  const at::Tensor& w2_scale,  const at::Tensor& w2_zp,
    const at::Tensor& router_weights, int64_t chunk) {
  std::vector<at::IValue> inputs{
      x,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights, chunk};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_moe_monolithic_int4fp8", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedMonolithicInt4Fp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_monolithic_int4fp8, hpu_op);
}

} // namespace habana_lazy

TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_monolithic_int4fp8(Tensor x, "
      "Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
      "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weights, int chunk) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_monolithic_int4fp8", habana_lazy::fused_moe_monolithic_int4fp8_hpu);
}

static const auto& MoeFusedMonolithicInt4Fp8KernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_monolithic_int4fp8", habana::MoeFusedMonolithicInt4Fp8);
