/*
 * moe_fused_expert_int4fp8_backend_lazy.cpp -- the LAZY-FRONTEND variant of the
 * W4A8 K-EXPERT single-recipe ROUTED MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_routed_int4fp8`. It CAPTURES into the decode
 * HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1) -- the durable single-recipe
 * serving path for the int4->fp8 routed (single-stream + MTP-verify) compute.
 *
 * Mirrors moe_fused_expert_routed_backend_lazy.cpp (the bf16 routed `fused_moe` lazy
 * TU) byte-for-byte except the schema/op name + the shared header (which calls
 * emit_one_expert_i4f8 = int4->fp8 DIRECT dequant + native fp8 MME). The backend
 * lowering MoeFusedMoEInt4Fp8 (its K-fold + the add-tree) + MoeFusedMoEInt4Fp8Meta
 * come UNCHANGED from moe_fused_expert_int4fp8_backend_shared.h; ONLY the frontend
 * differs. SELF-CONTAINED: this single .so carries the schema
 * (TORCH_LIBRARY_FRAGMENT so it co-loads with the bf16 routed / single-expert .so
 * without a duplicate-library abort), the lazy frontend binding (TORCH_LIBRARY_IMPL),
 * and the KernelRegistry backend registration. The habana_lazy::* frontend symbols
 * resolve at RUNTIME from the loaded lazy plugin (see the lazy build script).
 *
 * Build CPU-only:  bash build_routed_int4fp8_lazy.sh  (inside vllm-gaudi:next, under
 * PT_HPU_LAZY_MODE=1) -> build/libmoe_fused_routed_int4fp8_lazy.so. RUN ON THE HPU BOX.
 */

// --- shared backend (the K-expert AddNode + MoeFusedMoEInt4Fp8Meta +
//     emit_one_expert_i4f8), VERBATIM. Pulls op_backend.h / hpu_op_helper.h /
//     stack_getter.h / habana_operator.h / synapse params + sets FMT_HEADER_ONLY. ---
#include "moe_fused_expert_int4fp8_backend_shared.h"

// --- frontend (LAZY op) -- modeled on the bridge's own lazy custom-op template
//     (lazy_custom_ops.cpp mixture_of_experts_fp8_lazy). LazyOp lives in
//     habana_kernels/lazy_kernels.h; the habana_lazy::* symbols it emits resolve at
//     RUNTIME from the loaded lazy plugin libhabana_pytorch_plugin.so. ---------------
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>

#include <utility>
#include <vector>

namespace habana_lazy {

// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;`: the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified (in namespace
// habana), and MoeFusedMoEInt4Fp8Meta is in namespace habana (the shared header).
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the
// output-meta fn, and dispatch through the acc-thread machinery. inputs = x + the
// 7 per-expert TensorLists (IDENTICAL signature to the bf16 routed fused_moe). The
// schema name routes the lowering to MoeFusedMoEInt4Fp8 via
// KernelRegistry().get("hpu_moe_ext::fused_moe_routed_int4fp8") at graph-build time.
// =============================================================================
static at::Tensor fused_moe_routed_int4fp8_hpu(
    const at::Tensor& x,
    at::TensorList w13_packed, at::TensorList w13_scale, at::TensorList w13_zp,
    at::TensorList w2_packed,  at::TensorList w2_scale,  at::TensorList w2_zp,
    at::TensorList router_weights) {
  std::vector<at::IValue> inputs{
      x,
      w13_packed, w13_scale, w13_zp,
      w2_packed,  w2_scale,  w2_zp,
      router_weights};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_moe_routed_int4fp8", std::move(inputs)};
  hpu_op.SetOutputMetaFn(MoeFusedMoEInt4Fp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_moe_routed_int4fp8, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts; TORCH_LIBRARY_FRAGMENT so it co-loads with the bf16
// routed / single-expert .so):
//   1. TORCH_LIBRARY_FRAGMENT schema -> torch.ops.hpu_moe_ext.fused_moe_routed_int4fp8
//   2. TORCH_LIBRARY_IMPL(HPU)        -> routes through the LAZY frontend
//   3. KernelRegistry().add           -> binds the schema name to MoeFusedMoEInt4Fp8
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "fused_moe_routed_int4fp8(Tensor x, "
      "Tensor[] w13_packed, Tensor[] w13_scale, Tensor[] w13_zp, "
      "Tensor[] w2_packed, Tensor[] w2_scale, Tensor[] w2_zp, "
      "Tensor[] router_weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_moe_routed_int4fp8", habana_lazy::fused_moe_routed_int4fp8_hpu);
}

static const auto& MoeFusedMoEInt4Fp8KernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_moe_routed_int4fp8", habana::MoeFusedMoEInt4Fp8);
