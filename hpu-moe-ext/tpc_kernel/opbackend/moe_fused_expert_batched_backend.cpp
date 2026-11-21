/*
 * moe_fused_expert_batched_backend.cpp -- EAGER frontend for the BATCHED-K
 * single-recipe fused int4 MoE op `torch.ops.hpu_moe_ext.fused_moe_batched`. It
 * registers the op (schema + HPU dispatch + KernelRegistry backend) and routes the
 * call through the EAGER frontend (habana::eager::EagerOp) into the HPU JIT
 * pipeline, EXACTLY as the proven sequential eager TU
 * (moe_fused_expert_routed_backend.cpp) does -- only the inputs are 7 STACKED [K,...]
 * tensors (one per weight class + the router) instead of 7 TensorLists.
 *
 * The recipe (the ONE AddNode emitting the 16-node batched chain) lives in the
 * frontend-free shared header moe_fused_expert_batched_backend_shared.h, which the
 * lazy TU reuses byte-for-byte. The batched node chain comes from
 * emit_batched_experts (moe_fused_expert_batched_emit.h). The sequential op +
 * single-expert op are UNTOUCHED -- this is a new, additive schema.
 *
 * SCHEMA via TORCH_LIBRARY_FRAGMENT (NOT TORCH_LIBRARY): the single-expert .so
 * already does TORCH_LIBRARY(hpu_moe_ext){...}. A 2nd TORCH_LIBRARY(hpu_moe_ext)
 * would abort ("library already defined"); the FRAGMENT ADDS `fused_moe_batched`
 * to the same library, co-load-safe with the single-expert + sequential .so.
 *
 * WHY THE EAGER FRONTEND (vs lazy): the vllm-gaudi:next image is the
 * pytorch-UPSTREAM/eager build -- the lazy plugin is NOT loaded in-process, so
 * habana_lazy::LazyOp would fail to load. EagerOp's non-inline symbols live in the
 * LOADED libhabana_pytorch2_plugin.so and resolve at runtime -- identical to the
 * sequential eager TU (see its header for the full rationale).
 *
 * Build CPU-only:  bash build_batched.sh  -> build/libmoe_fused_batched.so
 * RUN ON THE HPU BOX ONLY (tests/test_fused_moe_batched.py).
 */

// --- shared backend (the batched AddNode + meta + emit_batched_experts). Pulls
//     op_backend.h / hpu_op_helper.h / stack_getter.h / habana_operator.h /
//     synapse params + sets FMT_HEADER_ONLY. ----------------------------------
#include "moe_fused_expert_batched_backend_shared.h"

// --- frontend (EAGER op) -- same public path the sequential eager TU uses. ---
#include "habana_eager/ops/eager_op.h" // habana::eager::EagerOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <utility>
#include <vector>

namespace habana {

// =============================================================================
// EAGER frontend: build the EagerOp from (schema, inputs, out-shape) and call it.
// inputs = x + the 6 stacked [K,...] weight tensors + the stacked [K,1,1] router
// tensor (plain at::Tensor IValues, NOT TensorLists). .call() builds a one-node JIT
// graph whose schema name routes the lowering to MoeFusedMoeBatched via
// KernelRegistry().get("hpu_moe_ext::fused_moe_batched"), then launches it.
// =============================================================================
static at::Tensor fused_moe_batched_hpu(
    const at::Tensor& x,
    const at::Tensor& w13_packed, const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed,  const at::Tensor& w2_scale,  const at::Tensor& w2_zp,
    const at::Tensor& router_weights) {
  std::vector<std::vector<int64_t>> out_shapes{x.sizes().vec()};  // [M,H]
  std::vector<at::IValue> inputs{
      x,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_moe_batched", std::move(inputs), std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts; TORCH_LIBRARY_FRAGMENT so it co-loads with the
// single-expert + sequential .so):
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds torch.ops.hpu_moe_ext.fused_moe_batched
//   2. TORCH_LIBRARY_IMPL(HPU)        -> routes through the eager frontend
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
  m.impl("fused_moe_batched", habana::fused_moe_batched_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& MoeFusedMoeBatchedKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_batched", habana::MoeFusedMoeBatched);
