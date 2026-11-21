/*
 * moe_fused_monolithic_backend_lazy.cpp -- the LAZY-FRONTEND variant of the
 * MONOLITHIC (ALL-E experts) single-recipe fused int4 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_monolithic`. It exists so the monolithic op
 * CAPTURES into the decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1) -- the
 * durable, single-recipe serving path the qwen35-397b throughput serve uses.
 *
 * WHY A SEPARATE LAZY TU (vs the eager moe_fused_monolithic_backend.cpp): identical
 * reasoning to the batched lazy TU (moe_fused_expert_batched_backend_lazy.cpp) -- the
 * eager EagerOp frontend (a) fails to LOAD under lazy (its symbols live in the eager
 * plugin, not the lazy one loaded by PT_HPU_LAZY_MODE=1) and (b) even if it loaded,
 * bypasses the lazy IR the HPU-graph capture engine records. The LAZY frontend
 * (habana_lazy::LazyOp) fixes both: it loads against the lazy plugin and emits into
 * the lazy IR, so the ONE monolithic recipe is captured/replayed.
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering MoeFusedMonolithic (its AddNode
 * = ONE emit_monolithic_experts call) + MoeFusedMonolithicMeta, included UNCHANGED
 * from moe_fused_monolithic_backend_shared.h. ONLY the frontend differs.
 *
 * SELF-CONTAINED: this single .so carries the schema (TORCH_LIBRARY_FRAGMENT), the
 * lazy frontend binding (TORCH_LIBRARY_IMPL(HPU)) and the KernelRegistry backend
 * registration. We load the lazy .so XOR the eager .so by mode (never both).
 * TORCH_LIBRARY_FRAGMENT(hpu_moe_ext) so it co-loads with the single-expert +
 * sequential + batched lazy .so without a duplicate-library abort.
 *
 * Build CPU-only:  bash build_monolithic_lazy.sh  (inside vllm-gaudi:next, under
 * PT_HPU_LAZY_MODE=1) -> build/libmoe_fused_monolithic_lazy.so. RUN ON THE HPU BOX.
 */

// --- shared backend (the monolithic AddNode + meta + emit_monolithic_experts),
//     VERBATIM. Pulls op_backend.h / hpu_op_helper.h / stack_getter.h /
//     habana_operator.h / synapse params + sets FMT_HEADER_ONLY. ---------------
#include "moe_fused_monolithic_backend_shared.h"

// --- frontend (LAZY op) -- modeled on the bridge's own lazy custom-op template
//     (habana_kernels/lazy_custom_ops.cpp). LazyOp lives in
//     habana_kernels/lazy_kernels.h; the habana_lazy::* symbols it emits resolve at
//     RUNTIME from the loaded lazy plugin libhabana_pytorch_plugin.so. ----------
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>

#include <utility>
#include <vector>

namespace habana_lazy {

// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;`: the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified (in namespace
// habana), and MoeFusedMonolithicMeta is in namespace habana (the shared header).
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the output-meta
// fn, and dispatch through the acc-thread machinery. inputs = x + the 6 stacked
// [E,...] weight tensors + the dense [E,T,1] router tensor + the int chunk (plain
// IValues). LazyOp emits into the lazy IR, so the ONE monolithic recipe is recorded
// by the HPU-graph capture engine. The schema name routes the lowering to
// MoeFusedMonolithic via KernelRegistry().get("hpu_moe_ext::fused_moe_monolithic").
// =============================================================================
static at::Tensor fused_moe_monolithic_hpu(
    const at::Tensor& x,
    const at::Tensor& w13_packed, const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed,  const at::Tensor& w2_scale,  const at::Tensor& w2_zp,
    const at::Tensor& router_weights, int64_t chunk) {
  std::vector<at::IValue> inputs{
      x,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights, chunk};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_moe_monolithic", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedMonolithicMeta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_monolithic, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts; TORCH_LIBRARY_FRAGMENT so it co-loads with the
// single-expert + sequential + batched lazy .so):
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds torch.ops.hpu_moe_ext.fused_moe_monolithic
//   2. TORCH_LIBRARY_IMPL(HPU)        -> routes through the LAZY frontend
//   3. KernelRegistry().add           -> binds the schema name to MoeFusedMonolithic
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_monolithic(Tensor x, "
      "Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
      "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weights, int chunk) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_monolithic", habana_lazy::fused_moe_monolithic_hpu);
}

static const auto& MoeFusedMonolithicKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_monolithic", habana::MoeFusedMonolithic);
