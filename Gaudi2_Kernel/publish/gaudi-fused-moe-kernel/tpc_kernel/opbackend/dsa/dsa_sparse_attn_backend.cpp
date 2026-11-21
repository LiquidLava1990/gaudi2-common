/*
 * dsa_sparse_attn_backend.cpp -- DSA MVP-2: the EAGER-frontend TU for the
 * gather-consume sparse-attention OpBackend op
 * (`torch.ops.hpu_dsa_ext::sparse_attn`). This is the SECOND real DSA kernel -- a
 * GC-captured Synapse recipe that CONSUMES the MVP-1 indexer's top-k key indices
 * and produces the sparse MLA attention output (attend only the selected KV).
 *
 * The backend lowering (`DsaSparseAttn : OpBackend` + `DsaSparseAttnMeta`, incl.
 * the full gather-consume node-chain GUIDs/params and the matched oracle math)
 * lives in dsa_sparse_attn_backend_shared.h. This TU adds ONLY the EAGER frontend
 * (habana::eager::EagerOp) + the schema/registration, exactly mirroring the proven
 * dsa_indexer_backend.cpp / moe_fused_expert_backend.cpp:355-399. The lazy variant
 * (dsa_sparse_attn_backend_lazy.cpp) reuses the SAME shared backend byte-for-byte
 * and differs only in its frontend (habana_lazy::LazyOp).
 *
 * WHY OpBackend (not UserCustomOpDescriptor): the consume is a MULTI-node recipe
 * (ge -> where -> where -> concat -> gather -> permute -> sdpa_fwd -> reshape).
 * UserCustomOpDescriptor binds ONE guid; OpBackend in the KernelRegistry is how
 * the bridge ships its own multi-node ops -- the same justification proven by the
 * shipped hpu_moe_ext::fused_expert op + the sibling indexer op.
 *
 * WHY EAGER here: the vllm-gaudi:next image is the pytorch-UPSTREAM/eager build;
 * `import habana_frameworks.torch` loads the EAGER plugin
 * (libhabana_pytorch2_plugin.so), so habana::eager::EagerOp's non-inline symbols
 * resolve at runtime, while habana_lazy::LazyOp is unavailable. (Under LAZY serving
 * use dsa_sparse_attn_backend_lazy.cpp -- it captures into the decode HPU-graph.)
 *
 * Build CPU-only:  bash build_dsa_sparse_attn.sh   (inside vllm-gaudi:next; host
 * g++, links torch + the shipped libhabana_pytorch_backend(.upstream).so + the
 * cloned bridge headers, -D_GLIBCXX_USE_CXX11_ABI=1, NEVER the plugin libs).
 * RUN ON THE HPU BOX ONLY via dsa/tests/test_dsa_sparse_attn_op.py.
 */

// fmt header-only (see the shared header for the full rationale).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- shared backend (AddNode chain + DsaSparseAttnMeta). This pulls
//     op_backend.h / hpu_op_helper.h / stack_getter.h / habana_operator.h + the
//     synapse params headers and sets FMT_HEADER_ONLY. -------------------------
#include "dsa_sparse_attn_backend_shared.h"

// --- frontend (EAGER op): the public, non-codegen path the bridge's own
//     user-custom-op frontend uses in the EAGER/pt2 build
//     (habana_eager/ops/eager_op.h -> EagerOp<at::Tensor>{schema, inputs,
//     out_shapes}.call()). Identical include to dsa_indexer_backend.cpp.
#include "habana_eager/ops/eager_op.h"  // habana::eager::EagerOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

#include <algorithm>
#include <vector>

namespace habana {

// =============================================================================
// EAGER frontend: build the eager op from (schema, inputs, out-shape) and call
// it. EagerOp<...>.call() builds a one-node JIT graph whose schema name routes
// the lowering to DsaSparseAttn via KernelRegistry().get(opname), then launches
// it. VERBATIM mechanism from dsa_indexer_backend.cpp / moe_..._backend.cpp.
//
// Out shape == [T, H, Dv] bf16 (== DsaSparseAttnMeta). q is [T,H,1,Dh], v is
// [S,H,Dv]; we size the eager result from those.
// =============================================================================
static at::Tensor sparse_attn_hpu(
    const at::Tensor& q,         // [T, H, 1, Dh]  bf16
    const at::Tensor& k,         // [S, H, Dh]     bf16
    const at::Tensor& v,         // [S, H, Dv]     bf16
    const at::Tensor& topk_idx,  // [T, K]         int32 (-1 padded)
    double scale,
    int64_t gather_width) {
  TORCH_CHECK(q.dim() == 4, "[dsa_sparse_attn] q must be 4D [T,H,1,Dh], got ", q.dim());
  TORCH_CHECK(v.dim() == 3, "[dsa_sparse_attn] v must be 3D [S,H,Dv], got ", v.dim());
  const int64_t T = q.size(0);
  const int64_t H = q.size(1);
  const int64_t Dv = v.size(2);

  std::vector<std::vector<int64_t>> out_shapes{{T, H, Dv}};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_dsa_ext::sparse_attn",
      {q, k, v, topk_idx, scale, gather_width},
      std::move(out_shapes)};
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts, one .so; modeled on dsa_indexer_backend.cpp:101-117 +
// moe_fused_expert_backend.cpp:382-399). NB: the schema lives in the SAME
// `hpu_dsa_ext` library the indexer op defines -- but each op's eager/lazy .so is
// loaded INDEPENDENTLY (one at a time), and TORCH_LIBRARY(hpu_dsa_ext) may be
// declared in only ONE loaded .so per process. To allow loading the sparse_attn
// .so ALONGSIDE the indexer .so in the same process (the integration case), we
// register the sparse_attn SCHEMA via TORCH_LIBRARY_FRAGMENT (additive; multiple
// fragments of the same library coexist), not a second TORCH_LIBRARY (which would
// abort with "Only a single TORCH_LIBRARY can be used to register the namespace").
//   1. TORCH_LIBRARY_FRAGMENT schema -> adds hpu_dsa_ext::sparse_attn
//   2. TORCH_LIBRARY_IMPL(HPU)       -> routes through the eager frontend
//   3. KernelRegistry().add          -> binds the schema name to DsaSparseAttn
// =============================================================================
TORCH_LIBRARY_FRAGMENT(hpu_dsa_ext, m) {
  m.def(
      "sparse_attn(Tensor q, Tensor k, Tensor v, Tensor topk_idx, "
      "float scale, int gather_width) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_dsa_ext, HPU, m) {
  m.impl("sparse_attn", habana::sparse_attn_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
static const auto& DsaSparseAttnKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_dsa_ext::sparse_attn", habana::DsaSparseAttn);

/*
 * NEXT (MVP-3): integrate the indexer + sparse_attn ops into the vLLM MLA path
 * (oot_mla.py / sparse_mla_wire.py): land the indexer_cache_hpu + indexer_metadata
 * _hpu runner patches, set forward_oot to call indexer_topk, set the consume to
 * call sparse_attn, then flip FULL_ATTENTION_FALLBACK=False + DSA_READY=True AFTER
 * the on-card gates (indexer-fired>0 on ctx>2048; ctx<=topk sparse==dense; ctx 16K
 * sparse!=dense). See DSA_OPBACKEND_KERNEL_DESIGN.md section 7. The MoE + indexer
 * ops stay untouched.
 */
