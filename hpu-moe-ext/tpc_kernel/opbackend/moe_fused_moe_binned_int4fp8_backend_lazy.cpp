/*
 * moe_fused_moe_binned_int4fp8_backend_lazy.cpp -- LAZY-frontend variant of the W4A8
 * BINNED grouped-GEMM MoE op `torch.ops.hpu_moe_ext.fused_moe_binned_int4fp8`.
 * Captures into the decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1). Mirrors
 * moe_fused_monolithic_int4fp8_backend_lazy.cpp byte-for-byte except the schema name +
 * the shared header (emit_binned_group_i4f8) + the 2 trailing int args
 * (chunk_start, group).
 *
 * Build CPU-only: bash build_binned_int4fp8_lazy.sh (in vllm-gaudi:next, under
 * PT_HPU_LAZY_MODE=1) -> build/libmoe_fused_moe_binned_int4fp8_lazy.so. RUN ON THE HPU BOX.
 */
#include "moe_fused_moe_binned_int4fp8_backend_shared.h"

#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>

#include <utility>
#include <vector>

namespace habana_lazy {

using namespace habana;

static at::Tensor fused_moe_binned_int4fp8_hpu(
    const at::Tensor& x_perm,
    const at::Tensor& w13_packed, const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed,  const at::Tensor& w2_scale,  const at::Tensor& w2_zp,
    const at::Tensor& router_weights, int64_t chunk_start, int64_t group) {
  std::vector<at::IValue> inputs{
      x_perm,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights, chunk_start, group};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_moe_binned_int4fp8", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedMoeBinnedInt4Fp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_binned_int4fp8, hpu_op);
}

} // namespace habana_lazy

TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_binned_int4fp8(Tensor x_perm, "
      "Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
      "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weights, int chunk_start, int group) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_binned_int4fp8", habana_lazy::fused_moe_binned_int4fp8_hpu);
}

static const auto& MoeFusedMoeBinnedInt4Fp8KernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_binned_int4fp8", habana::MoeFusedMoeBinnedInt4Fp8);
