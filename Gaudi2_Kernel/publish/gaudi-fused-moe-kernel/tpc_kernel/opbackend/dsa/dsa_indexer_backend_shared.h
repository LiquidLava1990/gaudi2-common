/*
 * dsa_indexer_backend_shared.h -- the SHARED, frontend-agnostic backend lowering
 * for DSA MVP-1: the DeepSeek-Sparse-Attention *lightning indexer* top-k as a
 * GC-captured `habana::OpBackend` op (`torch.ops.hpu_dsa_ext::indexer_topk`).
 *
 * This is the FIRST real DSA kernel -- it replaces the slow eager-torch indexer
 * scorer (`hpu_dsa_ext.sparse_indexer_hpu.lightning_indexer_topk`) in the dense
 * bypass with a single captured Synapse recipe whose intermediates are
 * NON-PERSISTENT (graph-internal -> the GC bundles the per-head Q.K^T MME with
 * the ReLU / weighted-sum / top-k, SRAM-resident -- the same lever the shipped
 * `hpu_moe_ext::fused_expert` op proved on this exact box/stack).
 *
 * Modeled VERBATIM on the proven MoE OpBackend pattern
 * (../moe_fused_expert_backend_shared.h): a `struct : OpBackend` whose AddNode()
 * chains `OpBackend::BuildNode({guid, {syn_inputs}, {output_attrs}, &params,
 * size})` calls in PyTorch ROW-MAJOR order (last dim == FCD). The factoring into
 * a *_shared.h header (backend only, frontend-free) lets the eager TU
 * (dsa_indexer_backend.cpp, EagerOp) and the lazy TU
 * (dsa_indexer_backend_lazy.cpp, LazyOp) reuse this byte-for-byte.
 *
 * ============================ THE ORACLE MATH MATCHED ========================
 * From hpu-dsa-ext/hpu_dsa_ext/sparse_indexer_hpu.py
 * (`lightning_indexer_topk` :40-76 / `reference_indexer_topk` :79-92,
 *  parity-validated Jaccard 0.996, top-50 recall 1.0 vs the DeepSeek fp8 oracle):
 *
 *     logits[t,h,s] = q[t,h,:] . k[s,:]                  # per-head dot   (:66,:86)
 *     logits        = ReLU(logits)                       # :66 .clamp_(0) / :86 relu
 *     score[t,s]    = sum_h weights[t,h] * logits[t,h,s] # weighted head-sum (:67,:87)
 *     score         = masked_fill(key_pos>abs_pos, -inf) # causal s<=abs_pos  (:71,:89)
 *     idx[t]        = topk(score[t], k).indices          # top-k key indices  (:72,:91)
 *
 * `weights` already carry q_scale * softmax_scale * n_head**-0.5 (the design's
 * fp8-fold; ReLU is positive-homogeneous so q is consumed as bf16 with NO
 * q_scale/k_scale rescale -- sparse_indexer_hpu.py:13-17). NOTHING is hardcoded
 * to GLM: H/D/Sq/Sk are derived from the input tensor shapes and `index_topk` is
 * a scalar arg, so GLM-5.1 (index_n_heads=32) and DeepSeek-V3.2 (=64) both "just
 * work" -- exactly the generality the MoE op proved (FUSED_GENERALIZE_FP8_DESIGN.md).
 *
 * ============================ NODE-CHAIN (GUIDs + params) ====================
 * Inputs (PyTorch row-major; the head axis H is the leading BATCH dim so the
 * per-head Q.K^T is ONE `batch_gemm`, no head loop -- the MQA key is expanded to
 * H upstream, cheap, and keeps the geometry a clean 3D-3D bmm):
 *   q       [H, Sq, D]  bf16   index queries, per head (rope already applied UP-
 *                              STREAM in Indexer.forward -> the op is rope-style
 *                              agnostic, design 5.3)
 *   k       [H, Sk, D]  bf16   index keys, MQA single head expanded to H
 *   weights [H, Sq, 1]  bf16   per-head weight, q_scale*softmax_scale*H^-0.5 folded
 *   cmask   [Sq, Sk]   bool    causal mask, TRUE where key must be DROPPED
 *                              (key_pos > abs_pos) -- the cheap host-derived bound
 *                              (indexer_metadata_hpu.py); keeps masked_fill scalar.
 *   index_topk         int     k (2048 for GLM-5.1 / DeepSeek-V3.2; clamped to Sk)
 *
 *  1. logits = batch_gemm(q[H,Sq,D], k[H,Sk,D], {transpose_a=0, transpose_b=1})
 *                                                            -> [H, Sq, Sk] bf16
 *        guid "batch_gemm", synGEMMParams{false,true}  (matmul.cpp:105,144,200,227
 *        idiom: per-head q @ k^T). NON-PERSISTENT.
 *  2. relu = relu_fwd(logits)                                -> [H, Sq, Sk] bf16
 *        guid get_guid_with_precision("relu_fwd", bf16)  (addmm_gen.cpp:219).
 *  3. wl   = mult_fwd(relu, weights[H,Sq,1])                 -> [H, Sq, Sk] bf16
 *        guid get_guid_with_precision("mult_fwd", bf16); weights[H,Sq,1] broadcasts
 *        over the Sk (FCD) axis -- the same broadcast-mult the MoE router uses
 *        ([1,1] over [M,H], moe_fused_expert_backend.cpp:335-339).
 *  4. score = reduce_sum_multi_dim_fwd(wl, dim=0 (H), keepDim=false)
 *                                                            -> [Sq, Sk] bf16
 *        guid get_guid_with_precision("reduce_sum_multi_dim_fwd", bf16),
 *        ns_Reduction::ParamsV2{reductionDimensionMask = 1<<(rank-0-1) = 1<<2 = 4,
 *        keepDim=false}  (reduction_template.cpp:22-38: TPC-order bit (ndims-d-1)).
 *  5. masked = masked_fill_fwd(score, cmask, value.f=-inf)   -> [Sq, Sk] bf16
 *        guid get_guid_with_precision("masked_fill_fwd", bf16),
 *        ns_MaskedFill::ParamsV2{value.f = -inf}  (masked_fill_gen.cpp:60-122).
 *  6. (vals,idx64) = topk(masked, k, axis=tpc(dim=1)=0, bottomK=false)
 *                                       -> vals[Sq,k] bf16, idx[Sq,k] int64
 *        guid "topk", ns_TopkNodeV2::ParamsV4{bsw=k, axis=get_dim_in_tpc_order(1,2)
 *        =0, bottomK=false, isStable=false, isVcData=false, kType=K_TENSOR_NONE}
 *        (topk_gen.cpp:55-104; static graph -> no shape-tensor input). Largest k.
 *  7. out = cast i64 -> i32 (idx)                            -> [Sq, k] int32  RESULT
 *        OpBackend::BuildCast(idx64, [Sq,k], Long, Int)  (op_backend.h:364).
 *
 * SCOPE (MVP-1, faithful to the validated SET-match gate): returns the top-k key
 * INDICES [Sq, min(index_topk,Sk)] int32. The oracle's -1 right-pad for rows with
 * fewer than index_topk causal keys + the sentinel valid-filter (sparse_indexer_hpu
 * :73-74) are a CONSUME-side concern; for the synthetic GLM-shaped parity gate
 * (index_topk clamped to Sk, every query has >= k causal keys) the top-k index
 * SET matches the oracle exactly (order may differ on ties -- compare as a set /
 * recall, the BUILD_PLAN Gate-D contract). bf16 vs the fp32 reference is the same
 * positive-homogeneous path the scorer validated at Jaccard 0.996.
 *
 * Build CPU-only (vllm-gaudi:next, host g++):  bash build_dsa_indexer.sh
 * RUN ON THE HPU BOX ONLY via dsa/tests/test_dsa_indexer_op.py.
 */
#pragma once

// fmt (torch bundles fmtlib v12) must be header-only here: the compiled fmt::v12
// symbols our TU pulls via torch's fmt headers are NOT exported by any shipped
// torch/habana .so (hidden visibility), so they must be inlined. Guard to avoid a
// -Wmacro-redefined warning (the habana logging header also sets it). VERBATIM
// rationale from moe_fused_expert_backend_shared.h:30-32.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- bridge headers (cloned at tag v1.24.0-1007 under ../bridge) -------------
#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr, BuildNode, BuildCast
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor, get_dim_in_tpc_order
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair{pt_t, syn_t}
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

// --- torch + synapse params --------------------------------------------------
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"   // syn_type_* enums, HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp" // synGEMMParams
#include "perf_lib_layer_params.h"  // ns_TopkNodeV2::ParamsV4, ns_Reduction::ParamsV2, ns_MaskedFill::ParamsV2

#include <limits>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: the op returns the top-k key INDICES, shape [Sq, kk] int32 where
// kk = min(index_topk, Sk). Used by the frontend (size the result) AND the
// backend (final BuildNode/cast output attr). Header-inline so the single
// including TU owns the definition. Modeled on MoeFusedExpertMeta
// (moe_fused_expert_backend_shared.h:63-69).
//
// Stack order == schema: (q, k, weights, cmask, index_topk).
//   q     = stack[0]  [H, Sq, D]
//   k     = stack[1]  [H, Sk, D]
//   topk  = stack[4]  int
// =============================================================================
inline OutputMetaDataVector DsaIndexerTopkMeta(const at::Stack& stack) {
  const at::Tensor& q = stack_tensor(stack, 0);  // [H, Sq, D]
  const at::Tensor& k = stack_tensor(stack, 1);  // [H, Sk, D]
  TORCH_CHECK(q.dim() == 3, "[dsa_indexer] q must be 3D [H,Sq,D], got ", q.dim());
  TORCH_CHECK(k.dim() == 3, "[dsa_indexer] k must be 3D [H,Sk,D], got ", k.dim());
  const int64_t Sq = q.size(1);
  const int64_t Sk = k.size(1);
  const int64_t topk = stack.at(4).toInt();
  const int64_t kk = std::min(topk, Sk);

  OutputMetaDataVector meta(1);
  meta[0].shape = {Sq, kk};          // top-k key indices per query
  meta[0].dtype = c10::ScalarType::Int;  // int32 (consume-side contract)
  return meta;
}

// =============================================================================
// The OpBackend: emits the indexer node-chain (see the file header). Each
// non-final BuildNode output is a graph-internal (non-persistent)
// synapse_helpers::tensor; its .get() synTensor feeds the next node -- that is
// the SRAM-resident edge (the proven MoE lever). Modeled VERBATIM on
// MoeFusedExpert (moe_fused_expert_backend_shared.h:78-87).
// =============================================================================
struct DsaIndexerTopk : OpBackend {
  DsaIndexerTopk(int device_id, c10::ScalarType scalar_type)
      // guid "batch_gemm" is a harmless placeholder (real guids set per-BuildNode
      // below); res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(DsaIndexerTopkMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void DsaIndexerTopk::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "DsaIndexerTopk::AddNode");

  // ---- typed inputs (order == schema) ----
  auto q       = sg.getNextInput<TensorsPair>();  // bf16  [H, Sq, D]
  auto k       = sg.getNextInput<TensorsPair>();  // bf16  [H, Sk, D]
  auto weights = sg.getNextInput<TensorsPair>();  // bf16  [H, Sq, 1]
  auto cmask   = sg.getNextInput<TensorsPair>();  // bool  [Sq, Sk]  TRUE => drop
  const int64_t topk = sg.getNextInput<int64_t>();  // index_topk (long == int64_t)

  // ---- derive shapes (PyTorch row-major; last dim == FCD) ----
  const auto q_sizes = q.pt_t.sizes().vec();
  const auto k_sizes = k.pt_t.sizes().vec();
  TORCH_CHECK(q.pt_t.dim() == 3, "[dsa_indexer] q must be 3D [H,Sq,D]");
  TORCH_CHECK(k.pt_t.dim() == 3, "[dsa_indexer] k must be 3D [H,Sk,D]");
  const int64_t H  = q_sizes[0];
  const int64_t Sq = q_sizes[1];
  const int64_t D  = q_sizes[2];
  const int64_t Hk = k_sizes[0];
  const int64_t Sk = k_sizes[1];
  const int64_t Dk = k_sizes[2];
  TORCH_CHECK(Hk == H, "[dsa_indexer] k head dim ", Hk, " != q head dim ", H,
              " (expand the MQA indexer key to H upstream)");
  TORCH_CHECK(Dk == D, "[dsa_indexer] k head_dim ", Dk, " != q head_dim ", D);
  TORCH_CHECK(weights.pt_t.dim() == 3 && weights.pt_t.size(0) == H &&
                  weights.pt_t.size(1) == Sq && weights.pt_t.size(2) == 1,
              "[dsa_indexer] weights must be [H,Sq,1]");
  TORCH_CHECK(cmask.pt_t.dim() == 2 && cmask.pt_t.size(0) == Sq &&
                  cmask.pt_t.size(1) == Sk,
              "[dsa_indexer] cmask must be [Sq,Sk] bool");
  const int64_t kk = std::min(topk, Sk);
  TORCH_CHECK(kk >= 1, "[dsa_indexer] index_topk clamps to ", kk, " (<1); Sk=", Sk);

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // ===========================================================================
  // NODE 1 -- per-head Q.K^T: logits[H,Sq,Sk] = batch_gemm(q, k, b^T).
  // bmm contract (bmm_gen.cpp:42-60): self{b,n,m} @ mat2{b,m,p} -> {b,n,p}; with
  // synGEMMParams{transpose_a=false, transpose_b=true} mat2 is transposed on its
  // last two dims so k[H,Sk,D] -> k^T[H,D,Sk] and out = q @ k^T = [H,Sq,Sk]. This
  // is exactly the bridge Matmul's batch_gemm idiom (matmul.cpp:105/144/200/227).
  // NodeOutputAttr.sizes is a NON-OWNING at::IntArrayRef -> every shape MUST be a
  // NAMED vector that outlives the BuildNode call (the bridge's own discipline,
  // moe_fused_expert_backend_shared.h:196).
  // ===========================================================================
  synGEMMParams qkt_params{/*transpose_a=*/false, /*transpose_b=*/true};
  std::vector<int64_t> logits_shape{H, Sq, Sk};
  auto logits = OpBackend::BuildNode(
      this, graph,
      {"batch_gemm",
       {q.syn_t, k.syn_t},
       {{logits_shape, bf16, std::nullopt}},  // NON-PERSISTENT
       &qkt_params, sizeof(qkt_params)});

  // ===========================================================================
  // NODE 2 -- ReLU: relu[H,Sq,Sk] = relu_fwd(logits).  (== oracle :66 clamp_(0).)
  // ===========================================================================
  auto relu = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("relu_fwd"sv, bf16),
       {logits[0].get()},
       {{logits_shape, bf16, std::nullopt}}});

  // ===========================================================================
  // NODE 3 -- per-head weight: wl[H,Sq,Sk] = relu * weights[H,Sq,1] (broadcast
  // over the Sk/FCD axis). The [H,Sq,1] -> mult_fwd broadcast is the PROVEN form
  // (the MoE router mult broadcasts [1,1] over [M,H], moe_..._shared.h:289-295).
  // This is the per-head scale of the weighted head-sum (oracle :67/:87).
  // ===========================================================================
  auto wl = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {relu[0].get(), weights.syn_t},
       {{logits_shape, bf16, std::nullopt}}});

  // ===========================================================================
  // NODE 4 -- sum over heads: score[Sq,Sk] = reduce_sum(wl, dim=0(H), keepDim=0).
  // guid reduce_sum_multi_dim_fwd; ns_Reduction::ParamsV2{reductionDimensionMask,
  // keepDim}. The mask bit for PyTorch dim d of a rank-N tensor is (N-d-1) (TPC
  // order, reduction_template.cpp:33). Reducing dim 0 of rank 3 -> bit (3-0-1)=2
  // -> mask = 1<<2 = 4. keepDim=false collapses H -> [Sq,Sk]. (oracle :67/:87.)
  // ===========================================================================
  ns_Reduction::ParamsV2 red_params{};
  red_params.reductionDimensionMask = 1u << static_cast<unsigned>(3 - 0 - 1);  // dim 0 of rank 3
  red_params.keepDim = false;
  std::vector<int64_t> score_shape{Sq, Sk};
  auto score = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, bf16),
       {wl[0].get()},
       {{score_shape, bf16, std::nullopt}},
       &red_params, sizeof(red_params)});

  // ===========================================================================
  // NODE 5 -- causal mask: masked[Sq,Sk] = masked_fill(score, cmask, -inf).
  // guid masked_fill_fwd; ns_MaskedFill::ParamsV2{value.f = -inf} (the scalar
  // path, masked_fill_gen.cpp:86-89 sets params->value.f for a float self).
  // cmask is bool [Sq,Sk], TRUE where the key is non-causal (key_pos>abs_pos) ->
  // filled with -inf so top-k never selects it (oracle :71/:89). Inputs {self,
  // mask}; the value is carried in params (NOT a 3rd syn input).
  // ===========================================================================
  ns_MaskedFill::ParamsV2 mf_params{};
  mf_params.value.f = -std::numeric_limits<float>::infinity();
  auto masked = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("masked_fill_fwd"sv, bf16),
       {score[0].get(), cmask.syn_t},
       {{score_shape, bf16, std::nullopt}},
       &mf_params, sizeof(mf_params)});

  // ===========================================================================
  // NODE 6 -- top-k over keys: (vals,idx64) = topk(masked, k=kk, axis, largest).
  // guid "topk"; ns_TopkNodeV2::ParamsV4{bsw=kk, axis=get_dim_in_tpc_order(1, 2),
  // bottomK=false (largest), isStable=false, isVcData=false, kType=K_TENSOR_NONE}.
  // Static (non-dynamic) graph -> NO shape-tensor input (topk_gen.cpp:91-93 only
  // sets kType=K_TENSOR_SHAPE for is_dynamic_graph()). Two outputs: values (bf16,
  // same dtype as input) and indices (int64). We keep ONLY the indices.
  // (oracle :72/:91 score.topk(kk).indices.)  axis for the last dim (1) of a rank-2
  // tensor is (2-1-1)=0 -- the innermost/FCD axis in TPC order.
  // ===========================================================================
  ns_TopkNodeV2::ParamsV4 topk_params{};
  topk_params.bsw = static_cast<unsigned int>(kk);
  topk_params.axis = get_dim_in_tpc_order(/*dim=*/1, /*max_dims=*/2);  // last dim -> 0
  topk_params.bottomK = false;        // largest
  topk_params.isStable = false;
  topk_params.isVcData = false;
  topk_params.kType = K_TENSOR_NONE;  // static k
  std::vector<int64_t> topk_shape{Sq, kk};
  auto topk_out = OpBackend::BuildNode(
      this, graph,
      {"topk",
       {masked[0].get()},
       {{topk_shape, bf16, std::nullopt},               // values (discarded)
        {topk_shape, c10::ScalarType::Long, std::nullopt}},  // indices int64
       &topk_params, sizeof(topk_params)});

  // ===========================================================================
  // NODE 7 -- cast indices int64 -> int32 (the consume-side contract dtype).
  // BuildCast(op, graph, syn_in, sizes, from, to, final_result_index=0) ties the
  // cast output to syn_out(0) (op_backend.h:364-371). Returns a SINGLE tensor.
  // [Sq,kk] int32 == DsaIndexerTopkMeta's declared result.
  // ===========================================================================
  auto out_i32 = OpBackend::BuildCast(
      this, graph, topk_out[1].get(), topk_shape,
      c10::ScalarType::Long, c10::ScalarType::Int, /*final_result_index=*/0);

  syn_out(0) = std::move(out_i32);
}

} // namespace habana
