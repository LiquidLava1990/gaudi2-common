/*
 * dsa_indexer_backend.cpp -- DSA MVP-1: the EAGER-frontend TU for the lightning-
 * indexer top-k OpBackend op (`torch.ops.hpu_dsa_ext::indexer_topk`). This is the
 * FIRST real DSA kernel -- a GC-captured Synapse recipe that replaces the slow
 * eager-torch indexer scorer in the dense bypass.
 *
 * The backend lowering (`DsaIndexerTopk : OpBackend` + `DsaIndexerTopkMeta`,
 * incl. the full node-chain GUIDs/params and the matched oracle math) lives in
 * dsa_indexer_backend_shared.h. This TU adds ONLY the EAGER frontend
 * (habana::eager::EagerOp) + the schema/registration, exactly mirroring the
 * proven moe_fused_expert_backend.cpp:355-399. The lazy variant
 * (dsa_indexer_backend_lazy.cpp) reuses the SAME shared backend byte-for-byte and
 * differs only in its frontend (habana_lazy::LazyOp).
 *
 * WHY OpBackend (not UserCustomOpDescriptor): the indexer is a MULTI-node recipe
 * (batch_gemm -> relu -> mult -> reduce_sum -> masked_fill -> topk -> cast).
 * UserCustomOpDescriptor binds ONE guid; OpBackend in the KernelRegistry is how
 * the bridge ships its own multi-node ops -- the same justification proven by the
 * shipped hpu_moe_ext::fused_expert op (moe_fused_expert_backend.cpp:20-29).
 *
 * WHY EAGER here: the vllm-gaudi:next image is the pytorch-UPSTREAM/eager build;
 * `import habana_frameworks.torch` loads the EAGER plugin
 * (libhabana_pytorch2_plugin.so), so habana::eager::EagerOp's non-inline symbols
 * (EagerOpBase::run/validate_inputs, HbEagerTensorPool::get_backend_tensor)
 * resolve at runtime, while habana_lazy::LazyOp is unavailable. (Under LAZY
 * serving use dsa_indexer_backend_lazy.cpp instead -- it captures into the decode
 * HPU-graph, the fast serving path.)  See moe_fused_expert_backend.cpp:71-83.
 *
 * Build CPU-only:  bash build_dsa_indexer.sh   (inside vllm-gaudi:next; host g++,
 * links torch + the shipped libhabana_pytorch_backend(.upstream).so + the cloned
 * bridge headers, -D_GLIBCXX_USE_CXX11_ABI=1, NEVER the plugin libs).
 * RUN ON THE HPU BOX ONLY via dsa/tests/test_dsa_indexer_op.py.
 */

// fmt header-only (see the shared header / moe_..._backend.cpp:54-62 for the full
// rationale: the fmt::v12 symbols our TU pulls are hidden in every shipped .so).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- shared backend (AddNode chain + DsaIndexerTopkMeta). This pulls
//     op_backend.h / hpu_op_helper.h / stack_getter.h / habana_operator.h +
//     the synapse params headers and sets FMT_HEADER_ONLY. -------------------
#include "dsa_indexer_backend_shared.h"

// --- frontend (EAGER op): the public, non-codegen path the bridge's own
//     user-custom-op frontend uses in the EAGER/pt2 build
//     (habana_eager/ops/eager_op.h -> EagerOp<at::Tensor>{schema, inputs,
//     out_shapes}.call()). Identical include to moe_fused_expert_backend.cpp:84.
#include "habana_eager/ops/eager_op.h"  // habana::eager::EagerOp<at::Tensor>

// --- torch (schema + HPU dispatch registration) ------------------------------
#include <torch/library.h>

#include <algorithm>
#include <vector>

namespace habana {

// =============================================================================
// EAGER frontend: build the eager op from (schema, inputs, out-shape) and call
// it. EagerOp<...>.call() builds a one-node JIT graph whose schema name routes
// the lowering to DsaIndexerTopk via KernelRegistry().get(opname), then launches
// it. VERBATIM mechanism from moe_fused_expert_backend.cpp:355-369.
//
// Out shape == [Sq, min(index_topk, Sk)] int32 (== DsaIndexerTopkMeta). q is
// [H,Sq,D], k is [H,Sk,D]; we size the eager result from those + the scalar.
// =============================================================================
static at::Tensor indexer_topk_hpu(
    const at::Tensor& q,        // [H, Sq, D]  bf16
    const at::Tensor& k,        // [H, Sk, D]  bf16
    const at::Tensor& weights,  // [H, Sq, 1]  bf16
    const at::Tensor& cmask,    // [Sq, Sk]    bool
    int64_t index_topk) {
  TORCH_CHECK(q.dim() == 3, "[dsa_indexer] q must be 3D [H,Sq,D], got ", q.dim());
  TORCH_CHECK(k.dim() == 3, "[dsa_indexer] k must be 3D [H,Sk,D], got ", k.dim());
  const int64_t Sq = q.size(1);
  const int64_t Sk = k.size(1);
  const int64_t kk = std::min<int64_t>(index_topk, Sk);

  std::vector<std::vector<int64_t>> out_shapes{{Sq, kk}};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_dsa_ext::indexer_topk",
      {q, k, weights, cmask, index_topk},
      std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::Int});  // indices are int32
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts, one .so; modeled on moe_fused_expert_backend.cpp:382-399
// + full_gen.cpp / kernels_register.cpp from tag v1.24.0-1007):
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_dsa_ext.indexer_topk exist
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes the call through the eager frontend
//                                 (habana::eager::EagerOp) into the HPU JIT pipeline
//   3. KernelRegistry().add    -> binds the schema name to the DsaIndexerTopk
//                                 OpBackend that the lowering instantiates
// =============================================================================
TORCH_LIBRARY(hpu_dsa_ext, m) {
  m.def(
      "indexer_topk(Tensor q, Tensor k, Tensor weights, Tensor cmask, "
      "int index_topk) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_dsa_ext, HPU, m) {
  m.impl("indexer_topk", habana::indexer_topk_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
// REGISTER_HPU_BACKEND(op, class) expands to
//   add(op, [](int device_id, c10::ScalarType type){
//       return std::make_shared<class>(device_id, type); })
static const auto& DsaIndexerTopkKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_dsa_ext::indexer_topk", habana::DsaIndexerTopk);

/*
 * NEXT (MVP-2): the sparse-attention gather-consume op
 * (`hpu_dsa_ext::sparse_attn`), porting sparse_mla_consume.py's
 * `sparse_mla_gather_attention` to a sibling OpBackend node-chain
 * (sentinel-scatter+cumsum-pack of these top-k indices -> gather K/V -> sdpa_fwd
 * with a pad-only additive mask). It CONSUMES the int32 indices this op returns.
 * See DSA_OPBACKEND_KERNEL_DESIGN.md section 4b. The MoE op stays untouched.
 */
