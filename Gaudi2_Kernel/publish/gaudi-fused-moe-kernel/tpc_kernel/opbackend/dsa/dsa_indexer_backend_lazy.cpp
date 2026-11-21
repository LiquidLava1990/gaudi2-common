/*
 * dsa_indexer_backend_lazy.cpp -- DSA MVP-1: the LAZY-frontend variant of the
 * lightning-indexer top-k OpBackend op (`torch.ops.hpu_dsa_ext::indexer_topk`).
 * It exists so the op CAPTURES into the decode HPU-graph under LAZY serving
 * (PT_HPU_LAZY_MODE=1) -- the fast serving path (zero per-op host dispatch),
 * exactly as the shipped MoE op does (moe_fused_expert_backend_lazy.cpp).
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering `DsaIndexerTopk : OpBackend`
 * (its full AddNode BuildNode chain) + `DsaIndexerTopkMeta`, included UNCHANGED
 * from dsa_indexer_backend_shared.h. The SRAM-resident batch_gemm->...->topk
 * recipe is identical; ONLY the frontend differs (LazyOp vs EagerOp).
 *
 * WHY A SEPARATE LAZY TU (vs the eager .cpp): under PT_HPU_LAZY_MODE=1 the process
 * loads the LAZY plugin (libhabana_pytorch_plugin.so), not the eager one, so (1)
 * the eager .so would fail to load (undefined EagerOpBase::validate_inputs) and
 * (2) even if it loaded, EagerOp's path bypasses the lazy IR the HPU-graph capture
 * engine records. habana_lazy::LazyOp fixes both: it loads against the lazy plugin
 * and emits into the lazy IR, so the recipe is captured/replayed.
 * (moe_fused_expert_backend_lazy.cpp:6-18.)
 *
 * SELF-CONTAINED: this single .so carries the schema (TORCH_LIBRARY), the lazy
 * frontend binding (TORCH_LIBRARY_IMPL(HPU)) and the KernelRegistry backend
 * registration -- loading ONLY this .so under lazy is sufficient. Load the lazy
 * .so XOR the eager .so by mode (never both), so one schema def is fine.
 *
 * Build CPU-only:  bash build_dsa_indexer_lazy.sh   (inside vllm-gaudi:next under
 * PT_HPU_LAZY_MODE=1 so the lazy plugin is referenced; links ONLY the shipped
 * libhabana_pytorch_backend.so; the habana_lazy::* frontend symbols resolve at
 * RUNTIME from the loaded lazy plugin). RUN ON THE HPU BOX ONLY.
 */

// --- shared backend (AddNode chain + DsaIndexerTopkMeta), VERBATIM. This pulls
//     op_backend.h / hpu_op_helper.h / stack_getter.h / habana_operator.h + the
//     synapse params headers and sets FMT_HEADER_ONLY. ------------------------
#include "dsa_indexer_backend_shared.h"

// --- frontend (LAZY op) -- modeled VERBATIM on the bridge's own lazy custom-op
//     template (lazy_custom_ops.cpp mixture_of_experts impl) and the proven MoE
//     lazy TU (moe_fused_expert_backend_lazy.cpp:43-58):
//        LazyOp<at::Tensor> hpu_op{"hpu_dsa_ext::indexer_topk", inputs};
//        hpu_op.SetOutputMetaFn(DsaIndexerTopkMeta);
//        RUN_MAYBE_WITH_ACC_THREAD(indexer_topk, hpu_op);
//     LazyOp lives in habana_kernels/lazy_kernels.h; RUN_MAYBE_WITH_ACC_THREAD +
//     scheduleAccTask are in hpu_ops/hpu_op_helper.h (pulled by the shared
//     header). The habana_lazy::* symbols the LazyOp templates / macro emit are
//     left UNDEFINED here and resolve at RUNTIME from the loaded lazy plugin
//     libhabana_pytorch_plugin.so.
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

namespace habana_lazy {

// Mirror lazy_custom_ops.cpp's file-scope `using namespace habana;` (and the MoE
// lazy TU:63-70): the RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask
// unqualified (namespace habana, hpu_op_helper.h); DsaIndexerTopkMeta is also in
// namespace habana (the shared header). Both resolve via this using-directive.
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the
// output-meta fn so shape inference works, dispatch through the acc-thread/call
// machinery. LazyOp::call()/get_result() emit into the lazy IR, so the op is
// recorded by the HPU-graph capture engine (the whole point). The schema name
// "hpu_dsa_ext::indexer_topk" routes the lowering to DsaIndexerTopk via
// KernelRegistry().get(opname). VERBATIM mechanism from
// moe_fused_expert_backend_lazy.cpp:81-95.
// =============================================================================
static at::Tensor indexer_topk_hpu(
    const at::Tensor& q,        // [H, Sq, D]  bf16
    const at::Tensor& k,        // [H, Sk, D]  bf16
    const at::Tensor& weights,  // [H, Sq, 1]  bf16
    const at::Tensor& cmask,    // [Sq, Sk]    bool
    int64_t index_topk) {
  LazyOp<at::Tensor> hpu_op{
      "hpu_dsa_ext::indexer_topk",
      {q, k, weights, cmask, index_topk}};
  // Set the output-meta fn (== the backend's DsaIndexerTopkMeta: out is [Sq, kk]
  // int32) so LazyOp::get_result() sizes/types the lazy result tensor -- verbatim
  // with lazy_custom_ops.cpp's hpu_op.SetOutputMetaFn(...Meta).
  hpu_op.SetOutputMetaFn(DsaIndexerTopkMeta);
  RUN_MAYBE_WITH_ACC_THREAD(indexer_topk, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts, one .so; modeled on moe_fused_expert_backend_lazy.cpp
// :110-127 + kernels_register.cpp from tag v1.24.0-1007):
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_dsa_ext.indexer_topk exist
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes the call through the LAZY frontend
//                                 (habana_lazy::LazyOp) into the lazy IR pipeline
//   3. KernelRegistry().add    -> binds the schema name to the DsaIndexerTopk
//                                 OpBackend that the lowering instantiates
// =============================================================================
TORCH_LIBRARY(hpu_dsa_ext, m) {
  m.def(
      "indexer_topk(Tensor q, Tensor k, Tensor weights, Tensor cmask, "
      "int index_topk) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_dsa_ext, HPU, m) {
  m.impl("indexer_topk", habana_lazy::indexer_topk_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& DsaIndexerTopkKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_dsa_ext::indexer_topk", habana::DsaIndexerTopk);

/*
 * NEXT (MVP-2): the sparse-attention gather-consume op (`hpu_dsa_ext::sparse_attn`)
 * -- a sibling OpBackend that CONSUMES the int32 indices this op returns. The MoE
 * op stays untouched. See DSA_OPBACKEND_KERNEL_DESIGN.md section 4b.
 */
