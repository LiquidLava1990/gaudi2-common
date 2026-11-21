/*
 * moe_fused_monolithic_emit.h -- the MONOLITHIC (ALL-E experts) per-recipe emitter
 * for the THROUGHPUT/concurrent regime. It emits the whole E-expert int4 MoE as ONE
 * captured recipe, processing E in CHUNKS of `CHUNK` experts so that each chunk's
 * per-expert dequant stays NON-PERSISTENT (SRAM-resident -> bundled into its
 * batch_gemm MME) -- the fusion lever -- while iterating over all E experts. This is
 * the path that beats the STOCK monolithic op `VllmMixtureOfExpertsOpWNA16`, which
 * dequantizes all E experts to bf16 -> HBM -> a separate `mixture_of_experts` gemm
 * (the ~99us/token bf16->HBM round-trip floor at concurrency).
 *
 * ===========================================================================
 * RELATIONSHIP TO THE PROVEN BATCHED EMITTER (moe_fused_expert_batched_emit.h)
 * ===========================================================================
 * The PER-CHUNK math is BYTE-FOR-BYTE the proven batched chain
 * (emit_batched_experts): grouped dequant over the stacked [C,...] weights -> a
 * batch_gemm over the C-expert batch -> rank-3 gate/up slice -> batched SiLU -> 2nd
 * grouped dequant -> 2nd batch_gemm -> the router-weighted reduce-sum over C. The
 * THREE differences, all justified below:
 *
 *   1. ROUTER WEIGHT IS PER-(token, expert), NOT a [C,1,1] scalar. The batched/
 *      sequential ops fold a [K,1,1] per-expert SCALAR (valid only at T==1, where
 *      each gathered expert is hit by exactly one token). The MONOLITHIC op runs at
 *      T>1 (the throughput regime), where expert e is hit by a DIFFERENT weight for
 *      each of the T tokens (and 0 for tokens that did not route to e). So the
 *      router weight is a DENSE [C, T, 1] tensor (this chunk's slice of the full
 *      [E, T, 1] = the transpose of the [T, E] dense routing weight matrix the
 *      Python side builds from topk_ids/topk_weights). [C,T,1] broadcasts over the
 *      trailing H exactly as [C,1,1] did over [M,H] -- same mult_fwd_bf16 node, just
 *      a [C,T,1] operand. yw[c,t,:] = y[c,t,:] * W[c,t]; the reduce over c then sums
 *      each token's routed-expert contributions == the stock mixture_of_experts
 *      semantics (out[t] = sum_e expert_e(x[t]) * W[t,e]). NON-ROUTED (t,e) carry
 *      W=0 so they contribute exactly 0 -- the dense form is numerically identical
 *      to gathering, it just lets the GC compute every expert (no host-dependent
 *      gather) which is what makes a SINGLE static recipe possible for ALL E.
 *
 *   2. E IS CHUNKED. E (=64 for Qwen3.5-397B at TP8: 512/8) experts' dequant cannot
 *      all be SRAM-resident at once (each W13 [H,2I] bf16 ~16MB + W2 [I,H] ~8MB =>
 *      ~24MB/expert; 64*24MB = 1.5GB >> 48MB Gaudi-2 SRAM). We emit one batched
 *      sub-chain PER CHUNK of `CHUNK` experts (default 8 -> the proven K=8 batched
 *      tile) and chain the chunk outputs with add_fwd_bf16. Each chunk's two
 *      dequants are NON-PERSISTENT, so within a chunk the GC bundles each expert's
 *      dequant into its batch_gemm MME, SRAM-resident -- the SAME per-expert SRAM
 *      win the single/batched op proved, now BOUNDED to CHUNK experts co-resident at
 *      a time. CHUNK is the SRAM-pressure knob: smaller CHUNK == fewer dequant tiles
 *      co-resident == less spill risk (at the cost of more, smaller batch_gemms). E
 *      need NOT be divisible by CHUNK: the last chunk is the remainder (its batched
 *      sub-chain uses C = E - chunk_start, a smaller batch dim -- a distinct recipe
 *      node, fine since the bridge recipe-caches per shape and E/CHUNK are static).
 *
 *   3. THE WEIGHTS ARE THE FULL STACKED [E,...] TENSORS (the op slices each chunk
 *      ON-DEVICE). No Python gather: the monolithic op is fed the layer's persistent
 *      stacked [E,...] nn.Parameters (w13_qweight/scales/zp etc.) DIRECTLY. Each
 *      chunk's [C,...] sub-tensor is a `slice` node along PyTorch dim 0 (the SAME
 *      synSliceParamsV2 idiom the gate/up split uses, just on the leading expert
 *      axis) -- so the chunk weights are graph-internal views of the persistent
 *      bases, no freed python intermediate (multi-bucket-capture-safe: the held-leaf
 *      weights ARE the recipe inputs). The router weight [E,T,1] is likewise sliced
 *      [C,T,1] per chunk.
 *
 * NET: ONE recipe for the whole MoE layer, all E experts computed with per-expert
 * SRAM-resident dequant (no bf16->HBM round-trip), at T>1. The bridge recipe-caches
 * one recipe per (E, CHUNK, T, H, I) -- all static for a given serve/bucket.
 *
 * ============================ THE PER-CHUNK NODE-CHAIN ==========================
 * For a chunk of C experts (the stacked [C,...] slices of the [E,...] inputs +
 * the x[T,H] shared activation + the chunk's router weights rw_c[C,T,1]):
 *   reinterpret_cast(w13c) -> uint4 [C,H,2I]; dequant -> W13c[C,H,2I] bf16 (NON-PERS)
 *   x3 = reshape(x[T,H] -> [1,T,H]); x_b = broadcast -> [C,T,H]
 *   gu = batch_gemm(x_b[C,T,H], W13c[C,H,2I]) -> [C,T,2I]
 *   gate=slice(gu,dim2,[0,I)); up=slice(gu,dim2,[I,2I))
 *   silu = gate*sigmoid(gate); h = silu*up                          -> [C,T,I]
 *   reinterpret_cast(w2c) -> uint4 [C,I,H]; dequant -> W2c[C,I,H] bf16 (NON-PERS)
 *   y = batch_gemm(h[C,T,I], W2c[C,I,H]) -> [C,T,H]
 *   yw = y * rw_c[C,T,1]  (per-(token,expert) weight; broadcasts over H)
 *   chunk_out = reduce_sum(yw, dim 0 (C)) -> [T,H]
 * The chunk_out's are then add-chained by the caller; the FINAL add ties to syn_out(0).
 *
 * Frontend-free (pulls ONLY op_backend.h / hpu_op_helper.h + synapse params), so
 * either the eager or the lazy monolithic TU can include it. Reuses the SAME guids /
 * geometry as moe_fused_expert_batched_emit.h -- keep in lockstep with the proven
 * batched chain if its math ever changes.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr / NodeOutputAttr, BuildReshape/BuildBroadcast
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
// emit_chunk_slice -- slice a [E, ...] stacked tensor along PyTorch dim 0 (the
// expert axis) to [C, ...] for the experts [chunk_start, chunk_start+C). Same
// synSliceParamsV2 + TPC-reversed-axes idiom as the gate/up split, on the leading
// dim. `full_shape` is the PyTorch [E, d1, d2] shape of `in`; returns the [C,d1,d2]
// slice synTensor. out_dtype must match `in`'s element type (int32 for packed/zp,
// bf16 for scale/rw). The output is graph-internal (a view of the persistent base).
// =============================================================================
inline std::vector<sh::tensor> emit_chunk_slice(
    OpBackend* op, sh::graph& graph, synTensor in,
    const std::vector<int64_t>& full_shape,
    int64_t chunk_start, int64_t C, at::ScalarType out_dtype) {
  const int64_t rank = static_cast<int64_t>(full_shape.size());
  synSliceParamsV2 sp{};
  std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
  std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
  std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
  std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
  for (int64_t i = 0; i < rank; ++i) {
    sp.axes[i] = static_cast<unsigned>(rank - i - 1);  // TPC-reversed axis
    if (i == 0) {  // slice the leading EXPERT dim 0
      sp.starts[i] = static_cast<TSize>(chunk_start);
      sp.ends[i] = static_cast<TSize>(chunk_start + C);
      sp.steps[i] = static_cast<TSize>(1);
    } else {
      sp.starts[i] = static_cast<TSize>(0);
      sp.ends[i] = static_cast<TSize>(full_shape[i]);
      sp.steps[i] = static_cast<TSize>(1);
    }
  }
  std::vector<int64_t> out_shape = full_shape;
  out_shape[0] = C;
  return OpBackend::BuildNode(
      op, graph,
      {"slice",
       {in},
       {{out_shape, out_dtype, std::nullopt}},
       &sp, sizeof(sp)});
}

// =============================================================================
// emit_monolithic_chunk -- the proven batched chain for ONE chunk of C experts,
// returning chunk_out[T,H] bf16 (the reduce_sum over the chunk's expert axis). The
// inputs are the FULL stacked [E,...] synTensors (+ their PyTorch shapes) and the
// [chunk_start, C) window; this helper slices each [E,...] -> [C,...] on-device then
// runs the batched math. The router weight rw is the FULL [E, T, 1] tensor; this
// chunk uses rw[chunk_start:chunk_start+C] -> [C, T, 1] (broadcasts over H, per
// (token,expert)).
//
//   x        bf16  [T, H]               + T, H
//   E, chunk_start, C, I, OUT13(=2I)     (loop-invariant + the chunk window)
//   w13p     int32 [E, H, 2I/8]   + w13p_shape (full stacked)
//   w13s     bf16  [E, H/GS, 2I]   + w13s_shape
//   w13z     int32 [E, H/GS, 2I/8] + w13z_shape
//   w2p      int32 [E, I, H/8]     + w2p_shape
//   w2s      bf16  [E, I/GS, H]     + w2s_shape
//   w2z      int32 [E, I/GS, H/8]  + w2z_shape
//   rw       bf16  [E, T, 1]       + rw_shape
//
// chunk_out is normally NON-PERSISTENT (the caller's add-chain consumes it). When
// `is_final` is true (the single-chunk E<=CHUNK case: no add-tree) this chunk's
// reduce_sum IS the op result -> final_result_index=0 == syn_out(0).
// =============================================================================
inline std::vector<sh::tensor> emit_monolithic_chunk(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t T, int64_t H,
    int64_t E, int64_t chunk_start, int64_t C, int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s, const std::vector<int64_t>& w13s_shape,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,  const std::vector<int64_t>& w2s_shape,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw,   const std::vector<int64_t>& rw_shape,
    bool is_final = false) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // ---- slice each FULL [E,...] stacked input to this chunk's [C,...] (dim 0) ----
  // Packed weights + zp are int32; scales + rw are bf16. The slice output is a
  // graph-internal view of the persistent base (multi-bucket-safe -- the base IS
  // the recipe input; the slice is not a freed python intermediate).
  auto w13p_c = emit_chunk_slice(op, graph, w13p, w13p_shape, chunk_start, C, at::kInt);
  auto w13s_c = emit_chunk_slice(op, graph, w13s, w13s_shape, chunk_start, C, bf16);
  auto w13z_c = emit_chunk_slice(op, graph, w13z, w13z_shape, chunk_start, C, at::kInt);
  auto w2p_c  = emit_chunk_slice(op, graph, w2p,  w2p_shape,  chunk_start, C, at::kInt);
  auto w2s_c  = emit_chunk_slice(op, graph, w2s,  w2s_shape,  chunk_start, C, bf16);
  auto w2z_c  = emit_chunk_slice(op, graph, w2z,  w2z_shape,  chunk_start, C, at::kInt);
  auto rw_c   = emit_chunk_slice(op, graph, rw,   rw_shape,   chunk_start, C, bf16);

  // chunk-local PyTorch shapes (leading dim now C) for the dequant out-attrs.
  std::vector<int64_t> w13p_cs{C, H, OUT13 / 8};
  std::vector<int64_t> w13z_cs{C, w13z_shape[1], OUT13 / 8};
  std::vector<int64_t> w2p_cs{C, I, H / 8};
  std::vector<int64_t> w2z_cs{C, w2z_shape[1], H / 8};

  // --- reinterpret_cast int32-packed -> uint4 (VERBATIM the batched form): unpacked
  //     PyTorch shape = packed shape with LAST dim *8; out-attr carries the uint4 view.
  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast",
         {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // --- dequantize_4_bit_bf16(weight_u4[C,IN,OUT], scale[C,IN/GS,OUT], zp_u4) -> bf16.
  //     SAME non-grouped guid + zero-init ns_CastKernel::ParamsV3 as the batched path;
  //     3D [C,...] input accepted by ValidateShapeNonGrouped; scale broadcasts over
  //     the GROUP axis (dim 1) while the C axis is shared 1:1 -> per-expert scale/zp
  //     exact, NON-PERSISTENT (the GC bundles it into the batch_gemm MME, SRAM). ---
  auto dequant = [&](synTensor w_packed, const std::vector<int64_t>& w_cshape,
                     synTensor scale,
                     synTensor zp_packed, const std::vector<int64_t>& zp_cshape) {
    auto w_u4  = reinterpret_u4(w_packed, w_cshape);
    auto zp_u4 = reinterpret_u4(zp_packed, zp_cshape);
    std::vector<synTensor> in{w_u4[0].get(), scale, zp_u4[0].get()};
    std::vector<int64_t> out_shape = w_cshape;
    out_shape.back() *= 8;
    ns_CastKernel::ParamsV3 params{};
    return OpBackend::BuildNode(
        op, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, bf16),
         std::move(in),
         {{out_shape, bf16, std::nullopt}},  // NON-PERSISTENT intermediate
         &params, sizeof(params)});
  };

  // ---- dequant #1: W13_bf16 [C, H, 2I] (PyTorch order; graph-internal SRAM target) ----
  auto W13 = dequant(w13p_c[0].get(), w13p_cs, w13s_c[0].get(), w13z_c[0].get(), w13z_cs);

  // ---- broadcast the shared activation x[T,H] -> x_b[C,T,H] (expert batch dim) ----
  // x3 = reshape x[T,H] -> [1,T,H] (explicit leading unit dim); x_b = broadcast
  // [1,T,H] -> [C,T,H] (pure size-1 -> C expansion; the GC folds it, no HBM copy).
  // NAMED shape vectors (the helper sizes are non-owning IntArrayRef -- must outlive).
  std::vector<int64_t> x3_shape{1, T, H};
  auto x3 = OpBackend::BuildReshape(op, graph, x, x3_shape, bf16);
  std::vector<int64_t> xb_shape{C, T, H};
  auto x_b = OpBackend::BuildBroadcast(op, graph, x3.get(), xb_shape, bf16);

  // ---- batch_gemm #1: gu[C,T,2I] = bmm(x_b[C,T,H], W13[C,H,2I]) (contract H) ----
  synGEMMParams gemm_params{false, false};
  std::vector<int64_t> gu_shape{C, T, OUT13};
  auto gu = OpBackend::BuildNode(
      op, graph,
      {"batch_gemm",
       {x_b.get(), W13[0].get()},
       {{gu_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- slice split along PyTorch feature dim 2 (2I) of the rank-3 gu[C,T,2I]:
  //      gate=gu[:,:,:I], up=gu[:,:,I:]. synSliceParamsV2, TPC-reversed axes. ----
  const int64_t gu_rank = 3;
  const int64_t slice_dim = 2;
  const int64_t gu_dims[gu_rank] = {C, T, OUT13};
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    for (int64_t i = 0; i < gu_rank; ++i) {
      sp.axes[i] = static_cast<unsigned>(gu_rank - i - 1);
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
    std::vector<int64_t> out_shape{C, T, end - begin};
    return OpBackend::BuildNode(
        op, graph,
        {"slice",
         {in},
         {{out_shape, bf16, std::nullopt}},
         &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- SiLU(gate)=gate*sigmoid(gate); h = SiLU(gate) * up (all [C,T,I]) ----
  std::vector<int64_t> im_shape{C, T, I};
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

  // ---- dequant #2: W2_bf16 [C, I, H] (PyTorch order; graph-internal SRAM target) ----
  auto W2 = dequant(w2p_c[0].get(), w2p_cs, w2s_c[0].get(), w2z_c[0].get(), w2z_cs);

  // ---- batch_gemm #2: y[C,T,H] = bmm(h[C,T,I], W2[C,I,H]) (contract I) ----
  std::vector<int64_t> y_shape{C, T, H};
  auto y = OpBackend::BuildNode(
      op, graph,
      {"batch_gemm",
       {h[0].get(), W2[0].get()},
       {{y_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- yw = y * router_weight rw_c[C,T,1] (broadcast over H); per-(token,expert) ----
  // rw_c[C,T,1] * y[C,T,H]: the C axis is shared 1:1 (each expert), the T axis is
  // shared 1:1 (each token), the trailing [1] broadcasts over H. So yw[c,t,:] =
  // y[c,t,:] * W[chunk_start+c, t] -- token t's weight for expert chunk_start+c (0 if
  // t did not route there). NON-PERSISTENT (the reduce consumes it).
  auto yw = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw_c[0].get()},
       {{y_shape, bf16, std::nullopt}}});

  // ---- chunk_out = reduce_sum over the C (expert) axis (dim 0 of rank 3) -> [T,H] ----
  // Sums this chunk's expert contributions per token. ns_Reduction::ParamsV2{
  // reductionDimensionMask = 1<<(rank-dim-1) = 1<<(3-0-1)=4, keepDim=false}. NON-
  // PERSISTENT when there is an add-chain over chunks (the default); when is_final
  // (single chunk, no add-tree) this reduce IS the op result (final_result_index=0).
  ns_Reduction::ParamsV2 red_params{};
  red_params.reductionDimensionMask = 1u << static_cast<unsigned>(gu_rank - 0 - 1);
  red_params.keepDim = false;
  std::vector<int64_t> out_shape{T, H};
  const std::optional<int> final_idx =
      is_final ? std::optional<int>(0) : std::nullopt;
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, bf16),
       {yw[0].get()},
       {{out_shape, bf16, final_idx}},
       &red_params, sizeof(red_params)});
}

// =============================================================================
// emit_monolithic_experts -- the FULL E-expert monolithic chain: chunk E into
// ceil(E/CHUNK) windows of <=CHUNK experts, emit emit_monolithic_chunk per window,
// and add-chain the chunk outputs -> the single [T,H] result tied to syn_out(0).
//
//   chunk-add tree (left fold of add_fwd_bf16, == the sequential op's add-tree):
//     acc = chunk_0 ; acc = add_fwd_bf16(acc, chunk_j) for j=1..nchunks-1 ;
//     the LAST add (or the lone chunk's reduce) ties to syn_out(0).
//
// Inputs are the FULL stacked [E,...] synTensors (+ shapes), x[T,H], the full router
// weight rw[E,T,1], and the CHUNK size. Returns the final [T,H] result vector
// (element [0].get() == syn_out(0)).
// =============================================================================
inline std::vector<sh::tensor> emit_monolithic_experts(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t T, int64_t H,
    int64_t E, int64_t CHUNK, int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s, const std::vector<int64_t>& w13s_shape,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,  const std::vector<int64_t>& w2s_shape,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw,   const std::vector<int64_t>& rw_shape) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  std::vector<int64_t> th_shape{T, H};  // NAMED (non-owning IntArrayRef) -- outlive adds.

  const int64_t nchunks = (E + CHUNK - 1) / CHUNK;  // ceil(E/CHUNK)

  auto emit_chunk = [&](int64_t chunk_start, int64_t C, bool is_final) {
    return emit_monolithic_chunk(
        op, graph, x, T, H, E, chunk_start, C, I, OUT13,
        w13p, w13p_shape, w13s, w13s_shape, w13z, w13z_shape,
        w2p, w2p_shape, w2s, w2s_shape, w2z, w2z_shape,
        rw, rw_shape, is_final);
  };

  if (nchunks == 1) {
    // single chunk (E <= CHUNK): the chunk's own reduce_sum IS the op result
    // (is_final=true ties it to syn_out(0) -- no add-tree, no duplicate work).
    return emit_chunk(0, E, /*is_final=*/true);
  }

  // nchunks >= 2: add-chain the chunk reduce outputs.
  const int64_t C0 = std::min<int64_t>(CHUNK, E);
  auto acc = emit_chunk(0, C0, /*is_final=*/false);   // chunk_0 (non-persistent reduce)
  for (int64_t j = 1; j < nchunks; ++j) {
    const int64_t cs = j * CHUNK;
    const int64_t C = std::min<int64_t>(CHUNK, E - cs);
    auto cj = emit_chunk(cs, C, /*is_final=*/false);  // chunk_j (non-persistent reduce)
    const bool last = (j == nchunks - 1);
    const std::optional<int> final_idx =
        last ? std::optional<int>(0) : std::nullopt;   // last add == op result
    acc = OpBackend::BuildNode(
        op, graph,
        {get_guid_with_precision("add_fwd"sv, bf16),
         {acc[0].get(), cj[0].get()},
         {{th_shape, bf16, final_idx}}});
  }
  return acc;
}

} // namespace habana
