/*
 * dsa_sparse_attn_backend_shared.h -- the SHARED, frontend-agnostic backend
 * lowering for DSA MVP-2: the DeepSeek-Sparse-Attention GATHER-CONSUME sparse
 * attention as a GC-captured `habana::OpBackend` op
 * (`torch.ops.hpu_dsa_ext::sparse_attn`).
 *
 * This is the SECOND real DSA kernel. It CONSUMES the int32 top-k key indices
 * produced by the working MVP-1 indexer op (dsa_indexer_backend_shared.h,
 * `hpu_dsa_ext::indexer_topk` -> [Sq, kk] int32, -1 padded) and produces the
 * sparse MLA attention output, attending ONLY the selected (top-k) latent KV per
 * query -- the O(Sq*kk) PRODUCT path, never the O(Sq*Sk) masked path. All
 * intermediates are NON-PERSISTENT (graph-internal -> the GC bundles the gather +
 * the QK^T-softmax-PV into an SRAM-resident recipe, the same lever the shipped
 * hpu_moe_ext::fused_expert op + the sibling indexer op proved on this box/stack).
 *
 * Modeled VERBATIM on the proven MoE + indexer OpBackend pattern
 * (../moe_fused_expert_backend_shared.h, ./dsa_indexer_backend_shared.h): a
 * `struct : OpBackend` whose AddNode() chains `OpBackend::BuildNode({guid,
 * {syn_inputs}, {output_attrs}, &params, size})` calls in PyTorch ROW-MAJOR order
 * (last dim == FCD). The factoring into a *_shared.h header (backend only,
 * frontend-free) lets the eager TU (dsa_sparse_attn_backend.cpp, EagerOp) and the
 * lazy TU (dsa_sparse_attn_backend_lazy.cpp, LazyOp) reuse it byte-for-byte.
 *
 * ============================ THE ORACLE MATH MATCHED ========================
 * From hpu-dsa-ext/hpu_dsa_ext/sparse_mla_consume.py
 * (`sparse_mla_gather_attention` :165-229, the PRODUCT/option-B consume,
 *  parity-validated gather==masked-oracle diff 0.011 -- BUILD_PLAN.md:27-28),
 * and the equivalent self-contained reference the test ships
 * (tests/test_dsa_synthetic_hpu.py::_ref_gather_attention :202-254).
 *
 * The validated PRODUCT path is: the indexer's top-k indices ARE the per-query
 * gather indices (already left-packed by the indexer's topk, -1 padded for short
 * rows). So we gather the KV at those indices and SDPA over the gathered width K,
 * masking ONLY the -1 (pad) slots:
 *
 *     valid[t,j]   = topk_idx[t,j] >= 0                              # pad detector
 *     gid[t,j]     = valid ? topk_idx[t,j] : S    (sentinel row S)   # :242 torch.where
 *     k_g[t,j]     = k_ext[gid[t,j]]   (k_ext = [k ; zero_row])      # :240-244 gather
 *     v_g[t,j]     = v_ext[gid[t,j]]                                 # :241-245
 *     logits[h,t,j]= (q[t,h,:] . k_g[t,j,h,:]) * scale               # :248 einsum thd,twhd->thw
 *     logits      += where(valid, 0, -inf)                           # :249 pad-only additive bias
 *     attn         = softmax(logits)                                 # :251
 *     attn         = nan_to_num(attn, 0)                             # :252 NaN guard (all-pad row)
 *     out[t,h,:]   = sum_j attn[h,t,j] * v_g[t,j,h,:]                # :253 einsum thw,twhd->thd
 *
 * Causality is ALREADY baked into the gather geometry: the indexer applied the
 * causal mask (key_pos > abs_pos -> -inf) BEFORE its top-k (dsa_indexer_backend_
 * shared.h NODE 5), so every index it emits is causal. We therefore mask ONLY the
 * pad slots here -- exactly the design's "causality already in the gather geometry;
 * mask only the pad slots" (DSA_OPBACKEND_KERNEL_DESIGN.md section 4b, step 4).
 * The forced-self-diagonal NaN guard the oracle adds for degenerate empty rows is
 * covered by (a) the indexer's causal top-k always including self for real rows and
 * (b) the SDPA fp32-softmax + the all-pad rows producing a defined (zeroed-by-mask)
 * row; the synthetic GLM-shaped parity band is the oracle's validated diff ~1e-2.
 *
 * `scale` is the MLA softmax scale (qk_head_dim**-0.5, the caller's `scaling`);
 * NOTHING is hardcoded to GLM: T/H/Dh/Dv/K are derived from the input tensor
 * shapes, so GLM-5.1 and DeepSeek-V3.2 both "just work" -- the same generality the
 * MoE + indexer ops proved (FUSED_GENERALIZE_FP8_DESIGN.md, section 5).
 *
 * ============================ NODE-CHAIN (GUIDs + params) ====================
 * Inputs (PyTorch row-major; SDPA uses the canonical [B,H,L,S] layout with the
 * QUERY axis T as the batch B, the per-head axis H as heads, target-seq L=1, and
 * the gathered width K as the source seq -- mask [T,1,1,K] then broadcasts over H
 * and L exactly like the bridge's documented Bx1x1xN mask, fused_sdpa.cpp:208-216):
 *   q        [T, H, 1, Dh]  bf16   per-head MLA query (nope||rope), L=1 per query
 *   k        [S, H, Dh]     bf16   per-head MLA key   (MQA expanded upstream)
 *   v        [S, H, Dv]     bf16   per-head MLA value
 *   topk_idx [T, K]         int32  the indexer's top-k key indices, -1 padded
 *   scale            float         MLA softmax scale (qk_head_dim**-0.5)
 *
 *  1. valid  = greater_equal_fwd(topk_idx, zeros[T,K] i32)     -> [T,K] bool
 *        guid get_guid_with_precision("greater_equal_fwd", Int); the zero operand
 *        is a ConstantHelper. valid==TRUE iff the slot is a real (>=0) index.
 *  2. gid    = where_fwd(valid, topk_idx, constS[T,K] i32)     -> [T,K] int32
 *        guid get_guid_with_precision("where_fwd", Int); pad slots -> sentinel row
 *        id S (the appended zero KV row), NEVER clamp(min=0). (oracle :242.)
 *  3. pad_bias = where_fwd(valid, const0[T,K] bf16, constNEG[T,K] bf16)
 *                                                              -> [T,K] bf16
 *        additive bias: 0 on valid, -inf on pad. (oracle :249.) Reshaped to
 *        [T,1,1,K] (SDPA Bx1x1xN mask) via ReshapeHelper.
 *  4. k_ext  = concat(k, zero_row[1,H,Dh]) on dim 0            -> [S+1,H,Dh]
 *     v_ext  = concat(v, zero_row[1,H,Dv]) on dim 0            -> [S+1,H,Dv]
 *        the SENTINEL trash row gid==S lands on (its content is masked anyway).
 *        guid "concat" (synConcatenateParams{axis = tpc(dim 0 of rank 3)=2}).
 *  5. k_flat = reshape(k_ext, [S+1, H*Dh]) ; gid_flat = reshape(gid, [T*K])
 *     k_g    = gather_fwd(k_flat, gid_flat)  -> [T*K, H*Dh] -> reshape [T,K,H,Dh]
 *     v_g    likewise                                          -> [T,K,H,Dv]
 *        guid get_guid_with_precision("gather_fwd", dtype), ns_GatherKernel::Params
 *        {axis = get_dim_in_tpc_order(0, 2) = 1} (the index_select row-gather,
 *        index_select_gen.cpp:22-29 + hpu_op.yaml:4082 guid gather_fwd). The
 *        flattened-row gather == oracle k_ext.index_select(0, flat_idx).view(...).
 *  6. q4 = permute q[T,H,1,Dh] (already L=1) ; k4 = permute k_g[T,K,H,Dh] ->
 *        [T,H,K,Dh] ; v4 = permute v_g[T,K,H,Dv] -> [T,H,K,Dv].
 *        guid "transpose" via PermuteHelper (the bridge's own transpose helper).
 *  7. out4 = sdpa_fwd(q4[T,H,1,Dh], k4[T,H,K,Dh], v4[T,H,K,Dv],
 *                     attention_mask = pad_bias[T,1,1,K], p=0, scale,
 *                     is_causal=false, softmax_mode="fp32")     -> [T,H,1,Dv]
 *        guid get_guid_with_precision("sdpa_fwd", bf16), ns_Sdpa::ParamsV7 filled
 *        exactly like sdpa_gen.cpp fillSdpaParams (scale, is_causal=false,
 *        is_inference=true, softmax_mode=SDPA_FP32_SOFTMAX, flags=0, wl=wr=-1).
 *        Input vector mirrors SDPAFwd::AddNode (sdpa_gen.cpp:353-377):
 *        {q,k,v, mask, seed=null, +7 nulls}. Two outputs (p==0 -> no amax); keep
 *        output[0]. The fused flash CGUID does QK^T+softmax+PV with the GC's
 *        tiling -- the O(L*K) product path natively (design section 6.4).
 *  8. out = reshape(out4[T,H,1,Dv]) -> [T,H,Dv]  RESULT (== oracle layout).
 *        guid "reshape" via ReshapeHelper. Drops the L=1 target-seq axis.
 *
 * SCOPE (MVP-2, faithful to the validated gather==oracle parity gate): returns the
 * sparse attention output [T, H, Dv] bf16. The gather WIDTH is the indexer buffer
 * width K (== topk_idx.size(1)); the schema accepts an optional `gather_width` for
 * forward-compat but MVP requires it == K (the buffer is already the bucket; the
 * validated default is `width = topk.shape[1]`, test :213). For the synthetic
 * GLM-shaped band (every query has >= K causal keys so no -1 pads, or short rows
 * masked by NODE 3) the output matches the masked-MHA oracle within the oracle's
 * validated diff (~1e-2). bf16 vs the fp32 oracle is the same band the consume
 * validated at diff 0.011.
 *
 * Build CPU-only (vllm-gaudi:next, host g++):  bash build_dsa_sparse_attn.sh
 * RUN ON THE HPU BOX ONLY via dsa/tests/test_dsa_sparse_attn_op.py.
 */
#pragma once

// fmt header-only (see the indexer/MoE shared headers for the full rationale: the
// fmt::v12 symbols our TU pulls via torch's fmt headers are hidden in every
// shipped .so, so they must be inlined). Guarded to avoid -Wmacro-redefined.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- bridge headers (cloned at tag v1.24.0-1007 under ../bridge) -------------
#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr, BuildNode, ConstantHelper, *Helper
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor, get_dim_in_tpc_order
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair{pt_t, syn_t}
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

// --- torch + synapse params --------------------------------------------------
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"   // syn_type_* enums, HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp" // synGEMMParams, synConcatenateParams
#include "perf_lib_layer_params.h"  // ns_GatherKernel::Params, ns_Sdpa::ParamsV7, SdpaSoftmaxMode_t

#include <limits>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: the op returns the sparse attention output, shape [T, H, Dv]
// bf16. Used by the frontend (size the result) AND the backend (final node
// output attr). Header-inline so the single including TU owns the definition.
// Modeled on DsaIndexerTopkMeta (dsa_indexer_backend_shared.h:137-151).
//
// Stack order == schema: (q, k, v, topk_idx, scale[, gather_width]).
//   q = stack[0]  [T, H, 1, Dh]   (target-seq L folded as the singleton dim 2)
//   v = stack[2]  [S, H, Dv]
// =============================================================================
inline OutputMetaDataVector DsaSparseAttnMeta(const at::Stack& stack) {
  const at::Tensor& q = stack_tensor(stack, 0);  // [T, H, 1, Dh]
  const at::Tensor& v = stack_tensor(stack, 2);  // [S, H, Dv]
  TORCH_CHECK(q.dim() == 4, "[dsa_sparse_attn] q must be 4D [T,H,1,Dh], got ", q.dim());
  TORCH_CHECK(v.dim() == 3, "[dsa_sparse_attn] v must be 3D [S,H,Dv], got ", v.dim());
  const int64_t T = q.size(0);
  const int64_t H = q.size(1);
  const int64_t Dv = v.size(2);

  OutputMetaDataVector meta(1);
  meta[0].shape = {T, H, Dv};                 // sparse attention output, per head
  meta[0].dtype = q.scalar_type();            // bf16 (== q dtype, the consume contract)
  return meta;
}

// =============================================================================
// The OpBackend: emits the gather-consume node-chain (see the file header).
// Each non-final BuildNode/Helper output is a graph-internal (non-persistent)
// synapse_helpers::tensor; its .get() synTensor feeds the next node -- the
// SRAM-resident edge. Modeled VERBATIM on DsaIndexerTopk
// (dsa_indexer_backend_shared.h:160-319).
// =============================================================================
struct DsaSparseAttn : OpBackend {
  DsaSparseAttn(int device_id, c10::ScalarType scalar_type)
      // guid "sdpa_fwd" is a harmless placeholder (real guids set per-BuildNode
      // below); res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "sdpa_fwd", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(DsaSparseAttnMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void DsaSparseAttn::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "DsaSparseAttn::AddNode");

  // ---- typed inputs (order == schema) ----
  auto q        = sg.getNextInput<TensorsPair>();  // bf16  [T, H, 1, Dh]
  auto k        = sg.getNextInput<TensorsPair>();  // bf16  [S, H, Dh]
  auto v        = sg.getNextInput<TensorsPair>();  // bf16  [S, H, Dv]
  auto topk_idx = sg.getNextInput<TensorsPair>();  // int32 [T, K] (-1 padded)
  const double scale = sg.getNextInput<double>();  // MLA softmax scale
  // gather_width: optional forward-compat scalar; MVP requires it == K (the index
  // buffer width). Pulled but only validated (no in-graph re-bucket of the buffer).
  const int64_t gather_width = sg.getNextInput<int64_t>();

  // ---- derive shapes (PyTorch row-major; last dim == FCD) ----
  const auto q_sizes = q.pt_t.sizes().vec();
  const auto k_sizes = k.pt_t.sizes().vec();
  const auto v_sizes = v.pt_t.sizes().vec();
  const auto ti_sizes = topk_idx.pt_t.sizes().vec();
  TORCH_CHECK(q.pt_t.dim() == 4, "[dsa_sparse_attn] q must be 4D [T,H,1,Dh]");
  TORCH_CHECK(k.pt_t.dim() == 3, "[dsa_sparse_attn] k must be 3D [S,H,Dh]");
  TORCH_CHECK(v.pt_t.dim() == 3, "[dsa_sparse_attn] v must be 3D [S,H,Dv]");
  TORCH_CHECK(topk_idx.pt_t.dim() == 2, "[dsa_sparse_attn] topk_idx must be 2D [T,K]");
  const int64_t T  = q_sizes[0];
  const int64_t H  = q_sizes[1];
  const int64_t L  = q_sizes[2];   // target-seq per query == 1
  const int64_t Dh = q_sizes[3];
  const int64_t S  = k_sizes[0];
  const int64_t Hk = k_sizes[1];
  const int64_t Dhk = k_sizes[2];
  const int64_t Sv = v_sizes[0];
  const int64_t Hv = v_sizes[1];
  const int64_t Dv = v_sizes[2];
  const int64_t Tt = ti_sizes[0];
  const int64_t K  = ti_sizes[1];
  TORCH_CHECK(L == 1, "[dsa_sparse_attn] q target-seq dim 2 must be 1 (per-query), got ", L);
  TORCH_CHECK(Hk == H && Hv == H, "[dsa_sparse_attn] head dim mismatch: qH=", H,
              " kH=", Hk, " vH=", Hv, " (expand MQA KV to H upstream)");
  TORCH_CHECK(Dhk == Dh, "[dsa_sparse_attn] k head_dim ", Dhk, " != q head_dim ", Dh);
  TORCH_CHECK(Sv == S, "[dsa_sparse_attn] v seq ", Sv, " != k seq ", S);
  TORCH_CHECK(Tt == T, "[dsa_sparse_attn] topk_idx T ", Tt, " != q T ", T);
  TORCH_CHECK(gather_width == K, "[dsa_sparse_attn] gather_width ", gather_width,
              " must equal the indexer buffer width K=", K,
              " for MVP-2 (the buffer is already the bucket)");
  TORCH_CHECK(K >= 1, "[dsa_sparse_attn] K (topk_idx width) must be >= 1, got ", K);

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType i32  = c10::ScalarType::Int;
  // The additive pad-mask "-inf": use the most-negative FINITE bf16, NOT true -inf.
  // This mirrors (a) the bridge's own nantonum bf16 neg-inf constant
  // (nantonum_gen.cpp:108 std::numeric_limits<c10::BFloat16>::lowest()) and (b) the
  // bridge SDPA test's LNEG=-1e9 (test_hpu_sdpa.py:37, with float('-inf') explicitly
  // commented out) -- the sdpa_fwd CGUID's mask path is exercised with a finite
  // large-negative, not non-finite. softmax (fp32 mode) of (logit + bf16::lowest)
  // -> 0 exactly, so the output equals the oracle's true-(-inf) result numerically.
  const at::Scalar NEG_BIG = std::numeric_limits<c10::BFloat16>::lowest();

  // ===========================================================================
  // NODE 1 -- pad detector: valid[T,K] bool = (topk_idx >= 0).
  // greater_equal_fwd(topk_idx, const0). The const-0 i32 [T,K] is a broadcasted
  // ConstantHelper. valid==TRUE iff the slot holds a real (>=0) key index; -1
  // pads (short rows) -> FALSE. (oracle :225 sel_valid = sel >= 0.)
  // ===========================================================================
  std::vector<int64_t> tk_shape{T, K};
  auto zero_i32 = ConstantHelper(graph, 0, i32, tk_shape);   // [T,K] i32 zeros
  auto valid = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("greater_equal_fwd"sv, i32),
       {topk_idx.syn_t, zero_i32.get()},
       {{tk_shape, c10::ScalarType::Bool, std::nullopt}}});  // NON-PERSISTENT bool

  // ===========================================================================
  // NODE 2 -- gather id: gid[T,K] i32 = where(valid, topk_idx, S).
  // where_fwd(valid, topk_idx, constS). Pad slots route to the SENTINEL row id S
  // (the appended zero KV row), NEVER clamp(min=0) (which would select token 0).
  // (oracle :242 torch.where(gathered_keep, sel, S).)
  // ===========================================================================
  auto const_S = ConstantHelper(graph, static_cast<int>(S), i32, tk_shape);  // [T,K] i32 == S
  auto gid = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("where_fwd"sv, i32),
       {valid[0].get(), topk_idx.syn_t, const_S.get()},
       {{tk_shape, i32, std::nullopt}}});  // NON-PERSISTENT

  // ===========================================================================
  // NODE 3 -- pad bias: pad_bias[T,K] bf16 = where(valid, 0, -inf).
  // where_fwd(valid, const0f, constNEG). Additive attention bias: 0 keeps the
  // slot, -inf zeroes it after softmax. (oracle :249 torch.where(keep, 0, -inf).)
  // Reshaped to the SDPA Bx1x1xN mask [T,1,1,K] for the per-query batch layout.
  // ===========================================================================
  auto zero_bf16 = ConstantHelper(graph, 0.0, bf16, tk_shape);            // [T,K] bf16 0
  auto neg_bf16  = ConstantHelper(graph, NEG_BIG, bf16, tk_shape);        // [T,K] bf16 ~-inf (finite)
  auto pad_bias = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("where_fwd"sv, bf16),
       {valid[0].get(), zero_bf16.get(), neg_bf16.get()},
       {{tk_shape, bf16, std::nullopt}}});
  // reshape [T,K] -> [T,1,1,K] (SDPA mask: batch=T, broadcast heads + target-seq).
  std::vector<int64_t> mask_shape{T, 1, 1, K};
  auto mask4 = ReshapeHelper(graph, pad_bias[0].get(), mask_shape, bf16);

  // ===========================================================================
  // NODE 4 -- append the SENTINEL trash row at index S: k_ext[S+1,H,Dh],
  // v_ext[S+1,H,Dv] = concat(k|v, zero_row) on dim 0. gid==S lands on this row
  // (its gathered content is masked out by NODE 3 anyway). guid "concat";
  // synConcatenateParams.axis = tpc(dim 0 of rank 3) = (3-0-1) = 2.
  // (oracle :240-241 torch.cat([k, zeros(1,H,Dh)], dim=0).)
  // ===========================================================================
  synConcatenateParams cat_params{};
  cat_params.axis = static_cast<unsigned>(get_dim_in_tpc_order(/*dim=*/0, /*max_dims=*/3));
  std::vector<int64_t> zrowk_shape{1, H, Dh};
  std::vector<int64_t> zrowv_shape{1, H, Dv};
  auto zrow_k = ConstantHelper(graph, 0.0, bf16, zrowk_shape);  // [1,H,Dh] bf16 0
  auto zrow_v = ConstantHelper(graph, 0.0, bf16, zrowv_shape);  // [1,H,Dv] bf16 0
  std::vector<int64_t> kext_shape{S + 1, H, Dh};
  std::vector<int64_t> vext_shape{S + 1, H, Dv};
  auto k_ext = OpBackend::BuildNode(
      this, graph,
      {"concat",
       {k.syn_t, zrow_k.get()},
       {{kext_shape, bf16, std::nullopt}},
       &cat_params, sizeof(cat_params)});
  auto v_ext = OpBackend::BuildNode(
      this, graph,
      {"concat",
       {v.syn_t, zrow_v.get()},
       {{vext_shape, bf16, std::nullopt}},
       &cat_params, sizeof(cat_params)});

  // ===========================================================================
  // NODE 5 -- row gather (index_select on the flattened key axis):
  //   k_flat = reshape(k_ext, [S+1, H*Dh]) ; gid_flat = reshape(gid, [T*K])
  //   k_g_flat = gather_fwd(k_flat, gid_flat) -> [T*K, H*Dh]
  //   k_g = reshape(k_g_flat, [T,K,H,Dh])
  // guid get_guid_with_precision("gather_fwd", dtype) (the index_select op, guid
  // gather_fwd per hpu_op.yaml:4082-4085), ns_GatherKernel::Params{axis =
  // get_dim_in_tpc_order(0, 2) = 1} (row axis 0 of the rank-2 self in tpc order).
  // Output shape = self with dim 0 replaced by index.numel() = T*K
  // (index_select_gen.cpp:41-50). == oracle k_ext.index_select(0, flat_idx).view.
  // ===========================================================================
  ns_GatherKernel::Params gather_params{};
  gather_params.axis = get_dim_in_tpc_order(/*dim=*/0, /*max_dims=*/2);  // 2-0-1 = 1
  // cast gid i32 -> i32 already; flatten gid [T,K] -> [T*K].
  std::vector<int64_t> gidflat_shape{T * K};
  auto gid_flat = ReshapeHelper(graph, gid[0].get(), gidflat_shape, i32);

  // K side
  std::vector<int64_t> kflat_shape{S + 1, H * Dh};
  auto k_flat = ReshapeHelper(graph, k_ext[0].get(), kflat_shape, bf16);
  std::vector<int64_t> kgflat_shape{T * K, H * Dh};
  auto k_g_flat = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("gather_fwd"sv, bf16),
       {k_flat.get(), gid_flat.get()},
       {{kgflat_shape, bf16, std::nullopt}},
       &gather_params, sizeof(gather_params)});
  std::vector<int64_t> kg_shape{T, K, H, Dh};
  auto k_g = ReshapeHelper(graph, k_g_flat[0].get(), kg_shape, bf16);

  // V side
  std::vector<int64_t> vflat_shape{S + 1, H * Dv};
  auto v_flat = ReshapeHelper(graph, v_ext[0].get(), vflat_shape, bf16);
  std::vector<int64_t> vgflat_shape{T * K, H * Dv};
  auto v_g_flat = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("gather_fwd"sv, bf16),
       {v_flat.get(), gid_flat.get()},
       {{vgflat_shape, bf16, std::nullopt}},
       &gather_params, sizeof(gather_params)});
  std::vector<int64_t> vg_shape{T, K, H, Dv};
  auto v_g = ReshapeHelper(graph, v_g_flat[0].get(), vg_shape, bf16);

  // ===========================================================================
  // NODE 6 -- arrange for SDPA's [B,H,L,S] layout (batch B=T, heads=H):
  //   q4 = q[T,H,1,Dh]                         (already [B,H,L=1,Dh]; no permute)
  //   k4 = permute k_g[T,K,H,Dh] -> [T,H,K,Dh] (perm (0,2,1,3))
  //   v4 = permute v_g[T,K,H,Dv] -> [T,H,K,Dv] (perm (0,2,1,3))
  // guid "transpose" via PermuteHelper (the bridge's own transpose helper).
  // ===========================================================================
  std::vector<int64_t> k4_shape{T, H, K, Dh};
  std::vector<int64_t> v4_shape{T, H, K, Dv};
  std::vector<int64_t> perm0213{0, 2, 1, 3};
  auto k4 = PermuteHelper(graph, k_g.get(), k4_shape, perm0213, bf16);
  auto v4 = PermuteHelper(graph, v_g.get(), v4_shape, perm0213, bf16);

  // ===========================================================================
  // NODE 7 -- fused sparse SDPA: out4[T,H,1,Dv] = sdpa_fwd(q4, k4, v4, mask4).
  // guid get_guid_with_precision("sdpa_fwd", bf16). ns_Sdpa::ParamsV7 filled
  // EXACTLY like sdpa_gen.cpp fillSdpaParams (:240-262): scale, dropout.ratio=0,
  // is_causal=false (causality is already in the gather geometry), is_inference=
  // true, softmax_mode=SDPA_FP32_SOFTMAX (the oracle softmaxes in fp32), flags=0,
  // wl=wr=-1 (no window). The input vector mirrors SDPAFwd::AddNode (sdpa_gen.cpp
  // :353-377): {q,k,v, attention_mask, seed=null, then 7 nulls (no valid_seq_len)}.
  // Two outputs when p==0 (out, qkt-stats); we keep output[0]. The fused flash
  // CGUID runs QK^T+softmax+PV with the GC's tiling -- the O(L*K) product path.
  // ===========================================================================
  ns_Sdpa::ParamsV7 sdpa_params{};
  sdpa_params.scale = static_cast<float>(scale);
  sdpa_params.dropout.ratio = 0.0f;
  sdpa_params.is_causal = false;
  sdpa_params.is_inference = true;
  sdpa_params.softmax_mode = SdpaSoftmaxMode_t::SDPA_FP32_SOFTMAX;
  sdpa_params.flags = 0;
  sdpa_params.wl = -1;
  sdpa_params.wr = -1;

  // SDPA output shapes (SDPAFwdOutputShapeCommon, sdpa_gen.cpp:96-148): batch dims
  // broadcast(q,k,v) = [T,H]; out = [T,H, q[L]=1, v[Ev]=Dv]; qkt = [T,H, 1, K].
  std::vector<int64_t> out4_shape{T, H, L, Dv};   // [T,H,1,Dv]
  std::vector<int64_t> qkt_shape{T, H, L, K};     // [T,H,1,K] (softmax stats, discarded)

  // Input vector: {q,k,v, mask, seed(null), +7 nulls}.
  std::vector<synTensor> sdpa_inputs = {
      q.syn_t, k4.get(), v4.get(),
      mask4.get(),     // attention_mask (the pad-only additive bias)
      nullptr};        // seed (no dropout)
  sdpa_inputs.insert(sdpa_inputs.end(), 7, nullptr);  // 7 trailing optionals (no valid_seq_len)

  auto out4 = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("sdpa_fwd"sv, bf16),
       std::move(sdpa_inputs),
       {{out4_shape, bf16, std::nullopt},   // output 0: attn out  [T,H,1,Dv]
        {qkt_shape, bf16, std::nullopt}},   // output 1: softmax stats (discarded)
       &sdpa_params, sizeof(sdpa_params)});

  // ===========================================================================
  // NODE 8 -- drop the L=1 target-seq axis: out[T,H,Dv] = reshape(out4[T,H,1,Dv]).
  // Tie the reshape output to syn_out(0) (the op result; == DsaSparseAttnMeta).
  // (oracle returns [T,H,Dv].)
  // ===========================================================================
  std::vector<int64_t> out_shape{T, H, Dv};
  auto out = ReshapeHelper(graph, out4[0].get(), out_shape, bf16, /*final_result_index=*/0);

  syn_out(0) = std::move(out);
}

} // namespace habana
