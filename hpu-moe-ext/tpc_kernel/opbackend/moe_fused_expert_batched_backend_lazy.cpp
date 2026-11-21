/*
 * moe_fused_expert_batched_backend_lazy.cpp -- the LAZY-FRONTEND variant of the
 * BATCHED-K single-recipe fused int4 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_batched`. It exists so the batched op CAPTURES
 * into the decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1) -- the durable,
 * single-recipe serving path the qwen35-397b-sslazy8 serve uses.
 *
 * WHY A SEPARATE LAZY TU (vs the eager moe_fused_expert_batched_backend.cpp):
 * identical reasoning to the sequential lazy TU
 * (moe_fused_expert_routed_backend_lazy.cpp) -- the eager EagerOp frontend (a) fails
 * to LOAD under lazy (its symbols live in the eager plugin, not the lazy one loaded
 * by PT_HPU_LAZY_MODE=1) and (b) even if it loaded, bypasses the lazy IR the
 * HPU-graph capture engine records. The LAZY frontend (habana_lazy::LazyOp) fixes
 * both: it loads against the lazy plugin and emits into the lazy IR, so the ONE
 * batched recipe is captured/replayed.
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering MoeFusedMoeBatched (its AddNode
 * = ONE emit_batched_experts call) + MoeFusedMoeBatchedMeta, included UNCHANGED from
 * moe_fused_expert_batched_backend_shared.h. ONLY the frontend differs.
 *
 * SELF-CONTAINED: this single .so carries the schema (TORCH_LIBRARY_FRAGMENT), the
 * lazy frontend binding (TORCH_LIBRARY_IMPL(HPU)) and the KernelRegistry backend
 * registration. We load the lazy .so XOR the eager .so by mode (never both).
 * TORCH_LIBRARY_FRAGMENT(hpu_moe_ext) so it co-loads with the single-expert +
 * sequential lazy .so without a duplicate-library abort.
 *
 * Build CPU-only:  bash build_batched_lazy.sh  (inside vllm-gaudi:next, under
 * PT_HPU_LAZY_MODE=1) -> build/libmoe_fused_batched_lazy.so. RUN ON THE HPU BOX.
 */

// --- shared backend (the batched AddNode + meta + emit_batched_experts), VERBATIM.
//     Pulls op_backend.h / hpu_op_helper.h / stack_getter.h / habana_operator.h /
//     synapse params + sets FMT_HEADER_ONLY. ----------------------------------
#include "moe_fused_expert_batched_backend_shared.h"

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
// habana), and MoeFusedMoeBatchedMeta is in namespace habana (the shared header).
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the
// output-meta fn, and dispatch through the acc-thread machinery. inputs = x + the 6
// stacked [K,...] weight tensors + the stacked [K,1,1] router tensor (plain
// at::Tensor IValues). LazyOp emits into the lazy IR, so the ONE batched recipe is
// recorded by the HPU-graph capture engine. The schema name routes the lowering to
// MoeFusedMoeBatched via KernelRegistry().get("hpu_moe_ext::fused_moe_batched").
// =============================================================================
static at::Tensor fused_moe_batched_hpu(
    const at::Tensor& x,
    const at::Tensor& w13_packed, const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed,  const at::Tensor& w2_scale,  const at::Tensor& w2_zp,
    const at::Tensor& router_weights) {
  std::vector<at::IValue> inputs{
      x,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_moe_batched", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedMoeBatchedMeta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_batched, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts; TORCH_LIBRARY_FRAGMENT so it co-loads with the
// single-expert + sequential lazy .so):
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds torch.ops.hpu_moe_ext.fused_moe_batched
//   2. TORCH_LIBRARY_IMPL(HPU)        -> routes through the LAZY frontend
//   3. KernelRegistry().add           -> binds the schema name to MoeFusedMoeBatched
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_batched(Tensor x, "
      "Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
      "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_batched", habana_lazy::fused_moe_batched_hpu);
}

static const auto& MoeFusedMoeBatchedKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_batched", habana::MoeFusedMoeBatched);
