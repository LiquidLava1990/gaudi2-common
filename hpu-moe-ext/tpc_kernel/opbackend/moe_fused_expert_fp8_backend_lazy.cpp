/*
 * moe_fused_expert_fp8_backend_lazy.cpp -- the LAZY-FRONTEND variant of the fp8
 * fused MoE expert OpBackend op. It exists so the fp8 op CAPTURES into the decode
 * HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1) -- the production serve path.
 *
 * WHY A SEPARATE LAZY TU (vs the eager moe_fused_expert_fp8_backend.cpp):
 *   The eager TU's frontend is `habana::eager::EagerOp<at::Tensor>{...}.call()`.
 *   Two problems under lazy serving (EXACTLY as the int4 op's lazy TU documents):
 *     1. LOAD: EagerOp's non-inline frontend symbols (EagerOpBase::run/
 *        validate_inputs, HbEagerTensorPool::get_backend_tensor) live ONLY in the
 *        EAGER plugin libhabana_pytorch2_plugin.so. Under PT_HPU_LAZY_MODE=1 the
 *        process loads the LAZY plugin libhabana_pytorch_plugin.so instead, so the
 *        eager .so fails to load with `undefined symbol EagerOpBase::validate_inputs`
 *        -- the exact false-null that made the live fp8 A/B never engage.
 *     2. CAPTURE: even if it loaded, EagerOp's EagerExec path bypasses the lazy IR
 *        that the HPU-graph capture engine records -- so the op would never be
 *        captured into the decode graph.
 *   The LAZY frontend (habana_lazy::LazyOp) fixes BOTH: it loads against the lazy
 *   plugin and emits into the lazy IR, so the recipe is captured/replayed.
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering `MoeFusedExpertFp8 : OpBackend`
 * (its AddNode BuildNode chain, incl. the BUILD_FP8_NODES fp8 nodes / DIRECT
 * int4->fp8 path) + the `MoeFusedExpertFp8Meta` output-meta fn + kFp8MaxFnuz are
 * included UNCHANGED from moe_fused_expert_fp8_backend_shared.h (factored out of
 * the proven eager .cpp). The SRAM-resident dequant->cast->MME recipe is identical;
 * ONLY the frontend differs. This mirrors the int4 op's eager/lazy/shared split.
 *
 * Build CPU-only:  BUILD_FP8_NODES=1 bash build_fp8_lazy.sh   (inside vllm-gaudi:next,
 * under PT_HPU_LAZY_MODE=1 so the lazy plugin is the referenced one; links ONLY the
 * shipped libhabana_pytorch_backend.so; the habana_lazy::* frontend symbols --
 * flush_op, AccThread::Get, the LazyOp templates -- resolve at RUNTIME from the
 * loaded lazy plugin libhabana_pytorch_plugin.so). RUN ON THE HPU BOX ONLY.
 */

// --- shared backend (AddNode chain + MoeFusedExpertFp8Meta + kFp8MaxFnuz),
//     VERBATIM from the proven eager TU. This pulls op_backend.h / hpu_op_helper.h
//     / stack_getter.h / habana_operator.h / synapse params + sets FMT_HEADER_ONLY
//     and BUILD_FP8_NODES (default 0; build_fp8_lazy.sh passes -DBUILD_FP8_NODES=1).
#include "moe_fused_expert_fp8_backend_shared.h"

// --- frontend (LAZY op) -- modeled VERBATIM on the int4 op's lazy TU
//     (moe_fused_expert_backend_lazy.cpp), which in turn mirrors the bridge's own
//     lazy custom-op template (opbackend/bridge/habana_kernels/lazy_custom_ops.cpp):
//        LazyOp<at::Tensor> hpu_op{"hpu::mixture_of_experts", inputs};
//        hpu_op.SetOutputMetaFn(...Meta);
//        RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, hpu_op);
//     LazyOp lives in habana_kernels/lazy_kernels.h; RUN_MAYBE_WITH_ACC_THREAD +
//     scheduleAccTask are in hpu_ops/hpu_op_helper.h (already pulled by the shared
//     header). The habana_lazy::* symbols the LazyOp templates / macro emit are
//     left UNDEFINED here and resolve at RUNTIME from the loaded lazy plugin.
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

namespace habana_lazy {

// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;` (the int4 lazy
// TU does the same): RUN_MAYBE_WITH_ACC_THREAD names scheduleAccTask unqualified
// (namespace habana, hpu_op_helper.h); MoeFusedExpertFp8Meta is also in namespace
// habana (the shared header). Both resolve via this using-directive.
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, IValue inputs), set the
// output-meta fn so shape inference works, and dispatch through the acc-thread/
// call machinery -- the EXACT mechanism the int4 lazy TU + the bridge's own lazy
// custom ops use. LazyOp::call()/get_result() emit into the lazy IR, so the op is
// recorded by the HPU-graph capture engine (the whole point).
//
// The schema name "hpu_moe_ext_fp8::fused_expert_fp8" routes the lowering to
// MoeFusedExpertFp8 via KernelRegistry().get(opname) at graph-build time.
//
// The two trailing fp8 recovery scales are OPTIONAL (HPU_MOE_FP8_DIRECT). They are
// passed as IValues so a None slot is preserved as a None c10::IValue -- IDENTICAL
// to the eager frontend's std::vector<at::IValue> construction
// (moe_fused_expert_fp8_backend.cpp:745-748). The backend StackGetter reads a None
// slot as {} (stack_getter.h:231-238) -> the proven two-hop in-graph-amax path;
// real [1] f32 scales -> the DIRECT one-dequant->fp8 path. LazyOp's ctor takes
// std::vector<at::IValue> (lazy_kernels.h:100-111), exactly like EagerOp.
// =============================================================================
static at::Tensor fused_expert_fp8_hpu(
    const at::Tensor& x, const at::Tensor& w13_packed,
    const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed, const at::Tensor& w2_scale,
    const at::Tensor& w2_zp, const at::Tensor& router_weight,
    const std::optional<at::Tensor>& w13_fp8_scale,
    const std::optional<at::Tensor>& w2_fp8_scale) {
  std::vector<at::IValue> inputs{
      x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp, router_weight,
      w13_fp8_scale.has_value() ? at::IValue(*w13_fp8_scale) : at::IValue(),
      w2_fp8_scale.has_value()  ? at::IValue(*w2_fp8_scale)  : at::IValue()};
  LazyOp<at::Tensor> hpu_op{
      "hpu_moe_ext_fp8::fused_expert_fp8",
      std::move(inputs)};
  // Set the output-meta fn (== the backend's MoeFusedExpertFp8Meta: out is [M,H]
  // bf16, same as x) so LazyOp::get_result() sizes/types the lazy result tensor --
  // verbatim with the int4 lazy TU's hpu_op.SetOutputMetaFn(...Meta).
  hpu_op.SetOutputMetaFn(MoeFusedExpertFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fused_expert_fp8, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts, all in this one .so; mirrors the eager fp8 .cpp's blocks
// + the int4 lazy TU's wiring):
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_moe_ext_fp8.fused_expert_fp8
//                                 exist (separate namespace from the int4 op so it
//                                 coexists in-process). MUST match the eager .cpp's
//                                 schema EXACTLY (incl. the two optional scales).
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes the call through the LAZY frontend
//                                 (habana_lazy::LazyOp) into the lazy IR pipeline.
//   3. KernelRegistry().add    -> binds the schema name to the MoeFusedExpertFp8
//                                 OpBackend that the lowering instantiates.
// We load the lazy .so XOR the eager .so by mode (never both -- _load_fp8_fused_op
// picks by is_lazy()), so one schema def per process is fine.
// =============================================================================
TORCH_LIBRARY(hpu_moe_ext_fp8, m) {
  // The two trailing fp8 recovery scales are OPTIONAL (default None): absent => the
  // proven two-hop int4->bf16->fp8 path; present (driver precomputes s_fp8 at load)
  // => DIRECT int4->fp8 dequant (HPU_MOE_FP8_DIRECT). Byte-identical schema to the
  // eager .cpp so existing callers are unaffected.
  m.def(
      "fused_expert_fp8(Tensor x, Tensor w13_packed, Tensor w13_scale, "
      "Tensor w13_zp, Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weight, Tensor? w13_fp8_scale=None, "
      "Tensor? w2_fp8_scale=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext_fp8, HPU, m) {
  m.impl("fused_expert_fp8", habana_lazy::fused_expert_fp8_hpu);
}

static const auto& MoeFusedExpertFp8KernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext_fp8::fused_expert_fp8", habana::MoeFusedExpertFp8);
