/*
 * dsa_sparse_attn_backend_lazy.cpp -- DSA MVP-2: the LAZY-frontend variant of the
 * gather-consume sparse-attention OpBackend op
 * (`torch.ops.hpu_dsa_ext::sparse_attn`). It exists so the op CAPTURES into the
 * decode HPU-graph under LAZY serving (PT_HPU_LAZY_MODE=1) -- the fast serving path
 * (zero per-op host dispatch), exactly as the shipped MoE op + the sibling indexer
 * lazy TU do.
 *
 * WHAT IS REUSED BYTE-FOR-BYTE: the backend lowering `DsaSparseAttn : OpBackend`
 * (its full AddNode gather-consume chain) + `DsaSparseAttnMeta`, included UNCHANGED
 * from dsa_sparse_attn_backend_shared.h. The SRAM-resident gather->sdpa_fwd recipe
 * is identical; ONLY the frontend differs (LazyOp vs EagerOp).
 *
 * WHY A SEPARATE LAZY TU (vs the eager .cpp): under PT_HPU_LAZY_MODE=1 the process
 * loads the LAZY plugin (libhabana_pytorch_plugin.so), not the eager one, so (1)
 * the eager .so would fail to load (undefined EagerOpBase::validate_inputs) and (2)
 * even if it loaded, EagerOp's path bypasses the lazy IR the HPU-graph capture
 * engine records. habana_lazy::LazyOp fixes both: it loads against the lazy plugin
 * and emits into the lazy IR, so the recipe is captured/replayed. (Verbatim from
 * the indexer lazy TU / moe_fused_expert_backend_lazy.cpp:6-18.)
 *
 * SELF-CONTAINED + COEXISTENCE: this single .so carries the schema
 * (TORCH_LIBRARY_FRAGMENT), the lazy frontend binding (TORCH_LIBRARY_IMPL(HPU)) and
 * the KernelRegistry registration. Loading ONLY this .so under lazy is sufficient
 * for the standalone test; the FRAGMENT (not a second TORCH_LIBRARY) also lets it
 * load ALONGSIDE the indexer lazy .so in the same serving process (MVP-3) without
 * the "single TORCH_LIBRARY per namespace" abort.
 *
 * Build CPU-only:  bash build_dsa_sparse_attn_lazy.sh   (inside vllm-gaudi:next
 * under PT_HPU_LAZY_MODE=1 so the lazy plugin is referenced; links ONLY the shipped
 * libhabana_pytorch_backend.so; the habana_lazy::* frontend symbols resolve at
 * RUNTIME from the loaded lazy plugin). RUN ON THE HPU BOX ONLY.
 */

// --- shared backend (AddNode chain + DsaSparseAttnMeta), VERBATIM. -----------
#include "dsa_sparse_attn_backend_shared.h"

// --- frontend (LAZY op) -- modeled VERBATIM on the indexer lazy TU /
//     moe_fused_expert_backend_lazy.cpp:43-58:
//        LazyOp<at::Tensor> hpu_op{"hpu_dsa_ext::sparse_attn", inputs};
//        hpu_op.SetOutputMetaFn(DsaSparseAttnMeta);
//        RUN_MAYBE_WITH_ACC_THREAD(sparse_attn, hpu_op);
//     LazyOp lives in habana_kernels/lazy_kernels.h; RUN_MAYBE_WITH_ACC_THREAD +
//     scheduleAccTask are in hpu_ops/hpu_op_helper.h (pulled by the shared header).
//     The habana_lazy::* symbols resolve at RUNTIME from the loaded lazy plugin.
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

namespace habana_lazy {

// Mirror the indexer lazy TU's file-scope `using namespace habana;`: the
// RUN_MAYBE_WITH_ACC_THREAD macro names scheduleAccTask unqualified (namespace
// habana, hpu_op_helper.h); DsaSparseAttnMeta is also in namespace habana (the
// shared header). Both resolve via this using-directive.
using namespace habana;

// =============================================================================
// LAZY frontend: build the LazyOp from (qualified-name, inputs), set the
// output-meta fn so shape inference works, dispatch through the acc-thread/call
// machinery. LazyOp::call()/get_result() emit into the lazy IR, so the op is
// recorded by the HPU-graph capture engine (the whole point). The schema name
// "hpu_dsa_ext::sparse_attn" routes the lowering to DsaSparseAttn via
// KernelRegistry().get(opname). VERBATIM mechanism from the indexer lazy TU.
// =============================================================================
static at::Tensor sparse_attn_hpu(
    const at::Tensor& q,         // [T, H, 1, Dh]  bf16
    const at::Tensor& k,         // [S, H, Dh]     bf16
    const at::Tensor& v,         // [S, H, Dv]     bf16
    const at::Tensor& topk_idx,  // [T, K]         int32 (-1 padded)
    double scale,
    int64_t gather_width) {
  LazyOp<at::Tensor> hpu_op{
      "hpu_dsa_ext::sparse_attn",
      {q, k, v, topk_idx, scale, gather_width}};
  // Set the output-meta fn (== DsaSparseAttnMeta: out is [T,H,Dv] bf16) so
  // LazyOp::get_result() sizes/types the lazy result tensor.
  hpu_op.SetOutputMetaFn(DsaSparseAttnMeta);
  RUN_MAYBE_WITH_ACC_THREAD(sparse_attn, hpu_op);
}

} // namespace habana_lazy

// =============================================================================
// Registration (3 parts, one .so; modeled on the indexer lazy TU :97-110).
// TORCH_LIBRARY_FRAGMENT (additive, coexists with the indexer .so's library).
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds hpu_dsa_ext::sparse_attn
//   2. TORCH_LIBRARY_IMPL(HPU)       -> routes through the LAZY frontend
//   3. KernelRegistry().add          -> binds the schema name to DsaSparseAttn
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_dsa_ext, m) {
  m.def(
      "sparse_attn(Tensor q, Tensor k, Tensor v, Tensor topk_idx, "
      "float scale, int gather_width) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_dsa_ext, HPU, m) {
  m.impl("sparse_attn", habana_lazy::sparse_attn_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& DsaSparseAttnKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_dsa_ext::sparse_attn", habana::DsaSparseAttn);

/*
 * NEXT (MVP-3): integrate indexer + sparse_attn into the vLLM MLA path + flip the
 * dense-bypass gate (FULL_ATTENTION_FALLBACK=False, DSA_READY=True) after the
 * on-card gates. See DSA_OPBACKEND_KERNEL_DESIGN.md section 7. The MoE + indexer
 * ops stay untouched.
 */
