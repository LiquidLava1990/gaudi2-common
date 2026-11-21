/*
 * moe_fused_expert_backend_lazy.cpp -- the LAZY-FRONTEND variant of the proven
 * fused int4 MoE expert OpBackend op. It exists so the op CAPTURES into the
 * decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1).
 *
 * WHY A SEPARATE LAZY TU (vs the eager moe_fused_expert_backend.cpp):
 *   The eager TU's frontend is `habana::eager::EagerOp<at::Tensor>{...}.call()`.
 *   Two problems under lazy serving:
 *     1. LOAD: EagerOp's non-inline frontend symbols (EagerOpBase::run/
 *        validate_inputs, HbEagerTensorPool::get_backend_tensor) live ONLY in the
 *        EAGER plugin libhabana_pytorch2_plugin.so. Under PT_HPU_LAZY_MODE=1 the
 *        process loads the LAZY plugin libhabana_pytorch_plugin.so instead, so the
 *        eager .so fails to load with an undefined EagerOpBase::validate_inputs.
 *     2. CAPTURE: even if it loaded, EagerOp's EagerExec path bypasses the lazy IR
 *        that the HPU-graph capture engine records -- so the op would never be
 *        captured into the decode graph.
 *   The LAZY frontend (habana_lazy::LazyOp) fixes BOTH: it loads against the lazy
 *   plugin and emits into the lazy IR, so the recipe is captured/replayed.
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering
 * `MoeFusedExpert : OpBackend` (its AddNode BuildNode chain) + the
 * `MoeFusedExpertMeta` output-meta fn are included UNCHANGED from
 * moe_fused_expert_backend_shared.h (factored out of the proven eager .cpp). The
 * SRAM-resident dequant->MME recipe is identical; ONLY the frontend differs.
 *
 * SELF-CONTAINED: this single .so carries the schema (TORCH_LIBRARY), the lazy
 * frontend binding (TORCH_LIBRARY_IMPL(HPU)) and the KernelRegistry backend
 * registration -- loading ONLY this .so under lazy is sufficient. We load the
 * lazy .so XOR the eager .so by mode (never both), so one schema def is fine.
 *
 * Build CPU-only:  bash build_lazy.sh   (inside vllm-gaudi:next, under
 * PT_HPU_LAZY_MODE=1 so the lazy plugin is the referenced one; links ONLY the
 * shipped libhabana_pytorch_backend.so; the habana_lazy::* frontend symbols --
 * flush_op, AccThread::Get, the LazyOp templates -- resolve at RUNTIME from the
 * loaded lazy plugin libhabana_pytorch_plugin.so). RUN ON THE HPU BOX ONLY.
 */

// --- shared backend (AddNode chain + MoeFusedExpertMeta), VERBATIM from the
//     proven eager TU. This pulls op_backend.h / hpu_op_helper.h / stack_getter.h
//     / habana_operator.h / synapse params + sets FMT_HEADER_ONLY. -------------
#include "moe_fused_expert_backend_shared.h"

// --- frontend (LAZY op) -- modeled VERBATIM on the bridge's own lazy custom-op
//     template, opbackend/bridge/habana_kernels/lazy_custom_ops.cpp (the
//     mixture_of_experts lazy impl, ~line 229):
//        LazyOp<at::Tensor> hpu_op{"hpu::mixture_of_experts", inputs};
//        hpu_op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);
//        RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, hpu_op);
//     LazyOp lives in habana_kernels/lazy_kernels.h (it transitively pulls
//     habana_lazy/aten_lazy_bridge.h, the lazy executor, and the acc-thread
//     headers); RUN_MAYBE_WITH_ACC_THREAD + scheduleAccTask are in
//     hpu_ops/hpu_op_helper.h (already pulled by the shared header). The
//     habana_lazy::* symbols the LazyOp templates / macro emit (flush_op,
//     AccThread::Get, create_node, empty_hpu_lazy, GetHbLazyTensor, ...) are left
//     UNDEFINED in this .so and resolve at RUNTIME from the loaded lazy plugin
//     libhabana_pytorch_plugin.so -- exactly as the eager .cpp does for
//     EagerOpBase::run against the eager plugin.
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

namespace habana_lazy {

// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;` (line 25):
// the RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified, which
// lives in namespace habana (hpu_op_helper.h); MoeFusedExpertMeta is also in
// namespace habana (the shared header). Both resolve via this using-directive,
// matching the bridge's own lazy TU exactly.
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the
// output-meta fn so shape inference works, and dispatch through the acc-thread/
// call machinery. This is the EXACT mechanism the bridge's own custom lazy ops
// use (lazy_custom_ops.cpp). LazyOp::call()/get_result() emit into the lazy IR,
// so the op is recorded by the HPU-graph capture engine (the whole point).
// The schema name "hpu_moe_ext::fused_expert" routes the lowering to
// MoeFusedExpert via KernelRegistry().get(opname) at graph-build time.
// =============================================================================
static at::Tensor fused_expert_hpu(
    const at::Tensor& x, const at::Tensor& w13_packed,
    const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed, const at::Tensor& w2_scale,
    const at::Tensor& w2_zp, const at::Tensor& router_weight) {
  LazyOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_expert",
      {x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
       router_weight}};
  // Set the output-meta fn (== the backend's MoeFusedExpertMeta: out is [M,H]
  // bf16, same as x) so LazyOp::get_result() sizes/types the lazy result tensor
  // -- verbatim with lazy_custom_ops.cpp's hpu_op.SetOutputMetaFn(...Meta).
  hpu_op.SetOutputMetaFn(MoeFusedExpertMeta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_expert, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts, all in this one .so; modeled on full_gen.cpp +
// kernels_register.cpp from tag v1.24.0-1007):
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_moe_ext.fused_expert exist
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes the call through the LAZY frontend
//                                 (habana_lazy::LazyOp) into the lazy IR pipeline
//                                 (== kernels_register.cpp's
//                                  TORCH_LIBRARY_IMPL(hpu, HPU){ m.impl(..., *_lazy) })
//   3. KernelRegistry().add    -> binds the schema name to the MoeFusedExpert
//                                 OpBackend that the lowering instantiates
// =============================================================================
TORCH_LIBRARY(hpu_moe_ext, m) {
  m.def(
      "fused_expert(Tensor x, Tensor w13_packed, Tensor w13_scale, "
      "Tensor w13_zp, Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_expert", habana_lazy::fused_expert_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
// REGISTER_HPU_BACKEND(op, class) expands to
//   add(op, [](int device_id, c10::ScalarType type){
//       return std::make_shared<class>(device_id, type); })
static const auto& MoeFusedExpertKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_expert", habana::MoeFusedExpert);

/*
 * K-EXPERT EXTENSION PATH (single-expert MVP is the goal here): identical to the
 * eager .cpp -- this op is ONE expert; the routed K-expert MoE is delivered by a
 * thin Python/ATen wrapper doing the HPU-graph-safe gather/topk/remap and calling
 * fused_expert per active (expert, w_k), accumulating. See the eager .cpp's tail
 * comment for the full fuse-the-K-launches follow-up plan.
 */
