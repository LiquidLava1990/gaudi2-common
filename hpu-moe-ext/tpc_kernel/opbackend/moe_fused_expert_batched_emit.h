/*
 * moe_fused_expert_batched_emit.h -- the BATCHED-K per-recipe emitter for the
 * single-recipe fused int4 MoE op. Where moe_fused_expert_emit.h::emit_one_expert
 * emits the proven per-expert chain ONCE PER EXPERT (the SEQUENTIAL op, K folds ->
 * ~K*14 nodes), this header emits the WHOLE K-expert MoE as ONE batched chain whose
 * node count is INDEPENDENT of K (16 nodes for any K): a grouped dequant over the
 * stacked [K,...] weights + a single batch_gemm over the K-expert batch dim + a
 * batched SiLU + a 2nd grouped dequant + a 2nd batch_gemm + the router-weighted
 * reduce-sum over K. This is the SINGLE-STREAM launch-count lever (Gaudi-2 decode
 * is host/launch-bound, see memory gaudi-single-stream-speed): K*14 -> 16.
 *
 * WHY A 2ND EMIT HEADER (not editing emit_one_expert): emit_one_expert is the
 * PROVEN, numerically-validated (relNorm 4.2e-3 on device) per-expert chain that
 * the SEQUENTIAL op (moe_fused_expert_routed_backend*.cpp) ships and that stays the
 * validated FALLBACK. This header is purely ADDITIVE: it builds the SAME math
 * (dequant W13 -> gemm -> gate/up slice -> silu -> dequant W2 -> gemm -> * router
 * -> sum_k) with the SAME guids (reinterpret_cast / dequantize_4_bit_bf16 / gemm
 * (now batch_gemm) / slice / sigmoid_fwd / mult_fwd / reduce_sum_multi_dim_fwd) and
 * the SAME PyTorch-row-major geometry, but ONE op per batched stage. Both stay in
 * lockstep -- if the proven math changes, change it in BOTH.
 *
 * ============================ THE BATCHED NODE-CHAIN (GUIDs + params) ===========
 * Inputs (all PyTorch ROW-MAJOR, last dim == Synapse FCD). x is shared across the K
 * experts; every weight class is a SINGLE STACKED [K, ...] tensor (NOT a TensorList
 * -- the wrapper hands a contiguous index_select output, so the held-leaf source
 * stays one persistent base, multi-bucket-capture-safe):
 *   x        bf16  [M, H]
 *   w13p     int32 [K, H, 2I/8]      w13s  bf16 [K, H/GS, 2I]      w13z int32 [K, H/GS, 2I/8]
 *   w2p      int32 [K, I, H/8]       w2s   bf16 [K, I/GS, H]       w2z  int32 [K, I/GS, H/8]
 *   rw       bf16  [K, 1, 1]         (per-expert scalar router weight; the T==1 fold)
 *
 *  1. w13_u4  = reinterpret_cast(w13p[K,H,2I/8] -> uint4 [K,H,2I])
 *  2. w13z_u4 = reinterpret_cast(w13z[K,H/GS,2I/8] -> uint4 [K,H/GS,2I])
 *  3. W13     = dequantize_4_bit_bf16(w13_u4, w13s[K,H/GS,2I], w13z_u4) -> [K,H,2I] bf16
 *        NON-GROUPED dequant SIF (convert_from_int4_gen.cpp ValidateShapeNonGrouped):
 *        input.dim() in [1,4] so a 3D [K,...] input is ACCEPTED; the scale
 *        [K,H/GS,2I] passes is_equal_except_one_and_divisible vs the unpacked
 *        [K,H,2I] (dims 0,2 equal; dim 1 H%(H/GS)==0) so the scale broadcasts over
 *        the GROUP axis EXACTLY as the 2D per-expert path -- and the leading K
 *        (expert) axis is shared 1:1 between input and scale, so each expert's own
 *        scale/zp row is applied to its own weight. NO cross-expert mixing. NON-
 *        PERSISTENT (the GC bundles the dequant into the batch_gemm's MMEs, SRAM-
 *        resident, per the proven single-expert fuseDequantIntoMoeGemm lever).
 *  4. x_b    = broadcast(x[M,H] -> [K,M,H])  (BroadcastHelper; expert-shared activ.)
 *  5. gu     = batch_gemm(x_b[K,M,H], W13[K,H,2I], {false,false}) -> [K,M,2I] bf16
 *        ONE bmm over the K-expert batch dim (matmul.cpp batch_gemm idiom); contract
 *        H. NON-PERSISTENT.
 *  6. gate   = slice(gu, dim 2, [0,I))   -> [K,M,I]
 *  7. up     = slice(gu, dim 2, [I,2I))  -> [K,M,I]   (synSliceParamsV2, TPC-rev axes)
 *  8. sgte   = sigmoid_fwd(gate)         -> [K,M,I]
 *  9. silu   = mult_fwd(gate, sgte)      -> [K,M,I]
 * 10. h      = mult_fwd(silu, up)        -> [K,M,I]
 * 11. w2_u4  = reinterpret_cast(w2p[K,I,H/8] -> uint4 [K,I,H])
 * 12. w2z_u4 = reinterpret_cast(w2z[K,I/GS,H/8] -> uint4 [K,I/GS,H])
 * 13. W2     = dequantize_4_bit_bf16(w2_u4, w2s[K,I/GS,H], w2z_u4) -> [K,I,H] bf16  NON-PERSISTENT
 * 14. y      = batch_gemm(h[K,M,I], W2[K,I,H], {false,false}) -> [K,M,H] bf16  NON-PERSISTENT
 * 15. yw     = mult_fwd(y, rw[K,1,1])   -> [K,M,H]  (per-expert router-weight; [K,1,1]
 *        broadcasts over the M,H trailing axes -- the proven router broadcast form,
 *        just batched over K. Each expert k is scaled by its OWN rw[k] before the sum,
 *        so the per-expert weight is exact -- no cross-expert leak.)
 * 16. out    = reduce_sum_multi_dim_fwd(yw, dim 0 (K), keepDim=false) -> [M,H]  RESULT
 *        sum over the K experts == the sequential op's add-tree, but ONE reduce node.
 *        mask bit for PyTorch dim 0 of rank 3 = 1<<(3-0-1)=4 (reduction_template.cpp).
 *
 * NET LAUNCH COUNT: 16 nodes for ANY K (vs the sequential op's ~K*14 + (K-1) adds:
 * K=8 -> 121, K=10 -> 149). The K experts' dequant tiles co-reside in ONE recipe
 * (the SAME top-risk as the sequential single-recipe op); both dequants are NON-
 * PERSISTENT so the GC's SRAM slicer bundles each batch_gemm's dequant input -- the
 * on-device weight-HBM/iter counter is the spill verdict (test CHECK 3).
 *
 * Frontend-free (pulls ONLY op_backend.h / hpu_op_helper.h + synapse params), so
 * either the eager or the lazy batched TU can include it.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr / NodeOutputAttr, BroadcastHelper
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision
#include "backend/synapse_helpers/graph.h"

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"     // syn_type_* enums, HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp"   // synGEMMParams + synSliceParamsV2
#include "perf_lib_layer_params.h"    // ns_CastKernel::ParamsV3, ns_Reduction::ParamsV2

#include <algorithm>   // std::fill_n
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// emit_batched_experts -- add the BATCHED K-expert fused-int4-MoE node chain (the
// 16 nodes above) and RETURN its [M,H] bf16 output (the final reduce_sum result
// vector; element [0].get() is the out synTensor, tied to syn_out(0)).
//
// Inputs are raw synTensors + their PyTorch shapes (a raw synTensor carries no
// sizes), exactly as emit_one_expert takes them -- but each weight-class tensor is
// the STACKED [K, ...] tensor (one syn input), so the K experts are processed in
// ONE pass per stage. The caller (the batched OpBackend AddNode) sources them from
// the schema's stacked Tensor inputs.
//
//   x        bf16  [M, H]                 + M, H
//   K, I, OUT13(=2I)                       (loop-invariant dims)
//   w13p     int32 [K, H, 2I/8]   + w13p_shape
//   w13s     bf16  [K, H/GS, 2I]
//   w13z     int32 [K, H/GS, 2I/8] + w13z_shape
//   w2p      int32 [K, I, H/8]     + w2p_shape
//   w2s      bf16  [K, I/GS, H]
//   w2z      int32 [K, I/GS, H/8]  + w2z_shape
//   rw       bf16  [K, 1, 1]
//
// The final reduce_sum's output is ALWAYS the op result (final_result_index=0):
// there is no degenerate K==1 special case -- the reduce over a K==1 batch axis is
// well-defined and still collapses [1,M,H] -> [M,H].
// =============================================================================
inline std::vector<sh::tensor> emit_batched_experts(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t M, int64_t H, int64_t K, int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // --- reinterpret_cast int32-packed -> uint4 (VERBATIM the per-expert form, only
  //     the rank is now 3): unpacked PyTorch shape = packed shape with LAST dim *8;
  //     out-attr carries the uint4 view. The leading K axis is untouched. ---
  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;  // packed int32 last dim (cols/8) -> uint4 last dim (cols)
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast",
         {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // --- dequantize_4_bit_bf16(weight_u4[K,IN,OUT], scale[K,IN/GS,OUT], zp_u4) -> bf16.
  //     SAME non-grouped guid + zero-init ns_CastKernel::ParamsV3 as the per-expert
  //     path; the 3D [K,...] input is accepted by ValidateShapeNonGrouped (dim<=4)
  //     and the scale broadcasts over the GROUP axis (dim 1) while the K axis is
  //     shared 1:1 -- per-expert scale/zp exact (see this file's header). ---
  auto dequant = [&](synTensor w_packed, const std::vector<int64_t>& w_shape,
                     synTensor scale,
                     synTensor zp_packed, const std::vector<int64_t>& zp_shape) {
    auto w_u4  = reinterpret_u4(w_packed, w_shape);
    auto zp_u4 = reinterpret_u4(zp_packed, zp_shape);
    std::vector<synTensor> in{w_u4[0].get(), scale, zp_u4[0].get()};
    std::vector<int64_t> out_shape = w_shape;
    out_shape.back() *= 8;
    ns_CastKernel::ParamsV3 params{};
    return OpBackend::BuildNode(
        op, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, bf16),
         std::move(in),
         {{out_shape, bf16, std::nullopt}},  // NON-PERSISTENT intermediate
         &params, sizeof(params)});
  };

  // ---- dequant #1: W13_bf16 [K, H, 2I] (PyTorch order; graph-internal SRAM target) ----
  auto W13 = dequant(w13p, w13p_shape, w13s, w13z, w13z_shape);

  // ---- broadcast the shared activation x[M,H] -> x_b[K,M,H] (expert batch dim) ----
  // The K-expert batch_gemm needs both operands to carry the K batch axis; x is
  // shared, so we replicate it across K. To make the rank increase UNAMBIGUOUS (the
  // GC `broadcast` node expands size-1 dims; it should NOT have to infer a prepended
  // dim), we FIRST reshape x[M,H] -> x3[1,M,H] (explicit leading unit dim) THEN
  // broadcast x3[1,M,H] -> x_b[K,M,H] (a pure size-1 -> K expansion). Both helpers are
  // the bridge's STATIC forms (BuildReshape op_backend.h:396, BuildBroadcast :404;
  // the protected member helpers are not callable from this free function), each
  // RETURNING a single sh::tensor. The broadcast is view-style (the GC folds it; it
  // does NOT materialize K copies of x in HBM). NAMED shape vectors (the
  // NodeOutputAttr / helper sizes are non-owning IntArrayRef -- must outlive the
  // call).
  std::vector<int64_t> x3_shape{1, M, H};
  auto x3 = OpBackend::BuildReshape(op, graph, x, x3_shape, bf16);
  std::vector<int64_t> xb_shape{K, M, H};
  auto x_b = OpBackend::BuildBroadcast(op, graph, x3.get(), xb_shape, bf16);

  // ---- batch_gemm #1: gu[K,M,2I] = bmm(x_b[K,M,H], W13[K,H,2I]) (contract H) ----
  // synGEMMParams{false,false}; per the bmm contract self{K,M,H} @ mat2{K,H,2I} ->
  // {K,M,2I}. ONE node over the K-expert batch (matmul.cpp:221-229 batch_gemm idiom).
  synGEMMParams gemm_params{false, false};
  std::vector<int64_t> gu_shape{K, M, OUT13};
  auto gu = OpBackend::BuildNode(
      op, graph,
      {"batch_gemm",
       {x_b.get(), W13[0].get()},
       {{gu_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- slice split along PyTorch feature dim 2 (2I) of the rank-3 gu[K,M,2I]:
  //      gate = gu[:, :, :I], up = gu[:, :, I:]. synSliceParamsV2 filled per-dim with
  //      TPC-reversed axes (axes[i] = rank-i-1), the sliced dim carrying [begin,end)
  //      step 1 and every other dim its full range -- the SAME idiom as the 2D
  //      per-expert slice (moe_fused_expert_emit.h), generalized to rank 3. ----
  const int64_t gu_rank = 3;                 // gu is PyTorch [K, M, 2I]
  const int64_t slice_dim = 2;               // PyTorch feature axis (2I)
  const int64_t gu_dims[gu_rank] = {K, M, OUT13};
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};                    // zero-init
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    for (int64_t i = 0; i < gu_rank; ++i) {
      sp.axes[i] = static_cast<unsigned>(gu_rank - i - 1);  // TPC-reversed axis
      if (i == slice_dim) {
        sp.starts[i] = static_cast<TSize>(begin);
        sp.ends[i] = static_cast<TSize>(end);
        sp.steps[i] = static_cast<TSize>(1);
      } else {
        sp.starts[i] = static_cast<TSize>(0);
        sp.ends[i] = static_cast<TSize>(gu_dims[i]);
        sp.steps[i] = static_cast<TSize>(1);
      }
    }
    std::vector<int64_t> out_shape{K, M, end - begin};  // PyTorch [K, M, I]
    return OpBackend::BuildNode(
        op, graph,
        {"slice",
         {in},
         {{out_shape, bf16, std::nullopt}},
         &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- SiLU(gate)=gate*sigmoid(gate); h = SiLU(gate) * up (all [K,M,I]) ----
  std::vector<int64_t> im_shape{K, M, I};
  auto sgte = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()},
       {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()},
       {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()},
       {{im_shape, bf16, std::nullopt}}});

  // ---- dequant #2: W2_bf16 [K, I, H] (PyTorch order; graph-internal SRAM target) ----
  auto W2 = dequant(w2p, w2p_shape, w2s, w2z, w2z_shape);

  // ---- batch_gemm #2: y[K,M,H] = bmm(h[K,M,I], W2[K,I,H]) (contract I) ----
  std::vector<int64_t> y_shape{K, M, H};
  auto y = OpBackend::BuildNode(
      op, graph,
      {"batch_gemm",
       {h[0].get(), W2[0].get()},
       {{y_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- yw = y * router_weight rw[K,1,1] (broadcast over M,H); per-expert weight ----
  // rw[K,1,1] * y[K,M,H]: the K axis is shared 1:1 (each expert scaled by its own
  // rw[k]); the trailing [1,1] broadcasts over [M,H] -- the proven router broadcast,
  // batched. NON-PERSISTENT (the reduce consumes it).
  auto yw = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw},
       {{y_shape, bf16, std::nullopt}}});

  // ---- out = reduce_sum over the K (expert) axis (dim 0 of rank 3) -> [M,H]. FINAL ----
  // ns_Reduction::ParamsV2{reductionDimensionMask = 1<<(rank-dim-1) = 1<<(3-0-1)=4,
  // keepDim=false}; the TPC-order bit for PyTorch dim 0 of rank 3 (reduction_template
  // .cpp:33). keepDim=false collapses K -> [M,H] (== x.sizes() == the op result).
  ns_Reduction::ParamsV2 red_params{};
  red_params.reductionDimensionMask = 1u << static_cast<unsigned>(gu_rank - 0 - 1);  // dim 0 of rank 3 -> bit 2 -> 4
  red_params.keepDim = false;
  std::vector<int64_t> out_shape{M, H};
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, bf16),
       {yw[0].get()},
       {{out_shape, bf16, /*final_result_index=*/0}},
       &red_params, sizeof(red_params)});
}

} // namespace habana
