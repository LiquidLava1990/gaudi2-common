/*
 * moe_fused_expert_routed_backend.cpp -- EAGER frontend for the K-EXPERT
 * single-recipe fused int4 MoE op `torch.ops.hpu_moe_ext.fused_moe`. It registers
 * the op (schema + HPU dispatch + KernelRegistry backend) and routes the call
 * through the EAGER frontend (habana::eager::EagerOp) into the HPU JIT pipeline,
 * EXACTLY as the proven single-expert eager TU (moe_fused_expert_backend.cpp) does
 * -- only the inputs are 7 TensorLists (K experts) instead of per-tensor.
 *
 * The recipe (the ONE AddNode emitting all K experts' fused subgraphs + the
 * weighted-sum add-tree) lives in the frontend-free shared header
 * moe_fused_expert_routed_backend_shared.h, which the lazy TU
 * (moe_fused_expert_routed_backend_lazy.cpp) reuses byte-for-byte. The per-expert
 * node chain itself comes from emit_one_expert (moe_fused_expert_emit.h), factored
 * VERBATIM from the proven single-expert recipe -- so this op is K folds of a
 * numerically-correct (relNorm 4.2e-3) base. The single-expert op is UNTOUCHED.
 *
 * WHY THE EAGER FRONTEND (vs lazy): the vllm-gaudi:next image is the
 * pytorch-UPSTREAM/eager build -- the lazy plugin is NOT loaded in-process, so
 * habana_lazy::LazyOp would fail to load. EagerOp's non-inline symbols
 * (EagerOpBase::run/validate_inputs, HbEagerTensorPool::get_backend_tensor) live in
 * the LOADED libhabana_pytorch2_plugin.so and resolve at runtime -- identical to
 * the single-expert eager TU (see its header for the full rationale).
 *
 * SCHEMA REGISTRATION via TORCH_LIBRARY_FRAGMENT (NOT TORCH_LIBRARY):
 *   The single-expert .so already does TORCH_LIBRARY(hpu_moe_ext){ fused_expert }.
 *   If BOTH .so's load in one process, a 2nd TORCH_LIBRARY(hpu_moe_ext) would abort
 *   ("library hpu_moe_ext already defined"). TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)
 *   ADDS `fused_moe` to the same library without that collision -- the bridge's own
 *   idiom for spreading one library across .so's (it uses
 *   TORCH_LIBRARY_FRAGMENT(custom_op,...) the same way). This gives the required
 *   `torch.ops.hpu_moe_ext.fused_moe` name AND co-load safety with the single op.
 *   (If this .so is loaded ALONE, the FRAGMENT still creates the library fine.)
 *
 * Build CPU-only:  bash build_routed.sh   -> build/libmoe_fused_routed.so
 *   docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work \
 *     -w /work/opbackend vllm-gaudi:next bash build_routed.sh
 * RUN ON THE HPU BOX ONLY (tests/test_fused_moe_single_recipe.py).
 */

// --- shared backend (the K-expert AddNode + MoeFusedMoEMeta + emit_one_expert).
//     This pulls op_backend.h / hpu_op_helper.h / stack_getter.h /
//     habana_operator.h / synapse params + sets FMT_HEADER_ONLY. ---------------
#include "moe_fused_expert_routed_backend_shared.h"

// --- frontend (EAGER op) -- same public path the single-expert eager TU uses. ---
#include "habana_eager/ops/eager_op.h" // habana::eager::EagerOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <utility>
#include <vector>

namespace habana {

// =============================================================================
// EAGER frontend: build the EagerOp from (schema, inputs, out-shape) and call it.
// inputs = x + the 7 per-expert TensorLists (w13_packed, w13_scale, w13_zp,
// w2_packed, w2_scale, w2_zp, router_weights). Placing an at::TensorList straight
// into the std::vector<at::IValue> initializer is the bridge's OWN pattern for its
// multi-expert ops (lazy_custom_ops.cpp mixture_of_experts_fp8_lazy builds its
// inputs vector with w1/w2/w3 TensorLists exactly like this). .call() builds a
// one-node JIT graph whose schema name routes the lowering to MoeFusedMoE via
// KernelRegistry().get("hpu_moe_ext::fused_moe"), then launches it.
// =============================================================================
static at::Tensor fused_moe_hpu(
    const at::Tensor& x,
    at::TensorList w13_packed, at::TensorList w13_scale, at::TensorList w13_zp,
    at::TensorList w2_packed,  at::TensorList w2_scale,  at::TensorList w2_zp,
    at::TensorList router_weights) {
  // out shape == x shape ([M,H]); dtype == bf16 (set below).
  std::vector<std::vector<int64_t>> out_shapes{x.sizes().vec()};
  std::vector<at::IValue> inputs{
      x,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_moe", std::move(inputs), std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts; modeled on the single-expert eager TU, but with
// TORCH_LIBRARY_FRAGMENT so it co-loads with that .so):
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds torch.ops.hpu_moe_ext.fused_moe
//   2. TORCH_LIBRARY_IMPL(HPU)        -> routes through the eager frontend
//   3. KernelRegistry().add           -> binds the schema name to MoeFusedMoE
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe(Tensor x, "
      "Tensor[] w13_packed, Tensor[] w13_scale, Tensor[] w13_zp, "
      "Tensor[] w2_packed, Tensor[] w2_scale, Tensor[] w2_zp, "
      "Tensor[] router_weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe", habana::fused_moe_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& MoeFusedMoEKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe", habana::MoeFusedMoE);
