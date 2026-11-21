/*
 * moe_fused_monolithic_fp8_emit.h -- the MONOLITHIC (ALL-E experts) per-recipe emitter
 * for the M3 swigluoai-fp8 THROUGHPUT path. Adapted from moe_fused_monolithic_emit.h
 * (the proven GLM int4 monolithic emitter) with THREE deltas:
 *
 *   DELTA 1 (DTYPE): the experts are RESIDENT fp8 (float8_e4m3fn), NOT int4. There is
 *     NO dequantize_4_bit, NO reinterpret_cast-to-u4, NO bf16 weight intermediate. The
 *     chunk slice of the [E,H,2I]/[E,I,H] fp8 weight feeds the native fp8_gemm DIRECTLY
 *     as B, with the PRECOMPUTED per-output-channel FORWARD scale (s13[E,2I]/s2[E,H],
 *     sliced to [C,2I]/[C,H]) as scaleB. The activation x (and the swigluoai output h)
 *     is dynamic-quantized to fp8 in-graph (per-token amax cast == dynamic_quant). The
 *     MME runs fp8 x fp8 -> bf16 at 2x rate -- the whole point. (Mirrors
 *     moe_fused_expert_fp8_backend_shared.h's make_fp8 + fp8_gemm, lifted to the
 *     batched/chunked [C,...] geometry.)
 *
 *   DELTA 2 (ACTIVATION): swiglu -> swigluoai. The GLM emitter applies
 *     silu(gate)*up = (gate*sigmoid(gate)) * up. M3's swigluoai is
 *       gate = gate.clamp(max=limit)              # UPPER bound only
 *       up   = up.clamp(min=-limit, max=limit)    # symmetric
 *       glu  = gate * sigmoid(gate * alpha)       # alpha INSIDE sigmoid
 *       h    = (up + 1.0) * glu                   # the +1, then (up+1)*glu
 *     alpha=1.702 limit=7.0 (M3 config swiglu_alpha/swiglu_limit). The gate/up split is
 *     CONTIGUOUS (gate=first half [0,I), up=second [I,2I)) -- identical to the GLM split.
 *     See emit_swigluoai_tail below for the exact node chain (clamp/sigmoid/mul/add).
 *
 *   DELTA 3 (NO ZERO-POINT, NO GROUP SCALE): fp8 weights have a single per-OUTPUT-CHANNEL
 *     recovery scale (not a grouped int4 [H/GS,2I] scale + a packed zero-point). The op
 *     takes 4 stacked weight inputs (w13_fp8, s13, w2_fp8, s2) instead of 6.
 *
 * EVERYTHING ELSE IS THE PROVEN MONOLITHIC STRUCTURE: E chunked into CHUNK-expert
 * batched sub-chains (each chunk's fp8 weight slice + its gemms SRAM-resident,
 * NON-PERSISTENT), the dense per-(token,expert) router weight rw[E,T,1] sliced to
 * [C,T,1] and multiplied after the down-gemm, the reduce_sum over the C axis, and the
 * add-chain across chunks tied to syn_out(0). For M3 at TP8: E=16 local experts, the
 * whole fp8 set (~0.91 GB) is HBM-resident (vs GLM-753B-int4's monolithic OOM).
 *
 * ============================ THE PER-CHUNK NODE-CHAIN ==========================
 * For a chunk of C experts (the stacked [C,...] slices + x[T,H] + rw_c[C,T,1]):
 *   w13c = slice(w13_fp8[E,H,2I], dim0, [cs,cs+C)) -> [C,H,2I] fp8   (SRAM-resident)
 *   s13c = slice(s13[E,2I],       dim0, [cs,cs+C)) -> [C,2I]  f32
 *   x_fp8, x_s = dynamic_quant(x[T,H])  -> [T,H] fp8, [T,1] f32  (per-token; ONCE/chunk)
 *   x_fp8_b = broadcast(reshape(x_fp8 -> [1,T,H]) -> [C,T,H])    (expert batch dim)
 *   x_s_b   = broadcast(reshape(x_s   -> [1,T,1]) -> [C,T,1])
 *   gu = fp8_batch_gemm(x_fp8_b[C,T,H], w13c[C,H,2I], sA=x_s_b[C,T,1], sB=s13c[C,1,2I])
 *        -> [C,T,2I] bf16                                         (fp8 MME, 2x rate)
 *   gate=slice(gu,dim2,[0,I)); up=slice(gu,dim2,[I,2I))          -> [C,T,I]
 *   h = swigluoai(gate, up, alpha, limit)                        -> [C,T,I] bf16
 *   h_fp8, h_s = dynamic_quant(h)  -> [C,T,I] fp8, [C,T,1] f32   (per-(expert,token))
 *   w2c  = slice(w2_fp8[E,I,H], dim0,[cs,cs+C)) -> [C,I,H] fp8 ; s2c = slice(s2)->[C,H]
 *   y = fp8_batch_gemm(h_fp8[C,T,I], w2c[C,I,H], sA=h_s[C,T,1], sB=s2c[C,1,H])
 *        -> [C,T,H] bf16
 *   yw = y * rw_c[C,T,1]   (per-(token,expert) weight; broadcasts over H)
 *   chunk_out = reduce_sum(yw, dim 0 (C)) -> [T,H]
 * The chunk_out's are add-chained by the caller; the FINAL add ties to syn_out(0).
 *
 * Frontend-free. Reuses the SAME guids/geometry as moe_fused_monolithic_emit.h for the
 * slice/broadcast/swigluoai/reduce skeleton, and the fp8 cast/gemm guids + params from
 * moe_fused_expert_fp8_backend_shared.h. Keep in lockstep with both if their math changes.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"
#include "hpu_ops/hpu_op_helper.h"
#include "backend/synapse_helpers/graph.h"

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"
#include "synapse_common_types.hpp"     // synGEMMParams + synSliceParamsV2
#include "perf_lib_layer_params.h"      // ns_Reduction::ParamsV2, ns_ConvertToFp8::ParamsV2, ns_Fp8Gemm::ParamsV2

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Gaudi-2 e4m3 FNUZ max (== moe_fused_expert_fp8_backend_shared.h kFp8MaxFnuz). The
// activation dynamic-quant denominator so the cast stays inside the real e4m3 range;
// the cast(1/s)<->gemm(s) pair cancels the fn<->fnuz factor (see that file's PROOF).
static constexpr float kMonoFp8MaxFnuz = 240.0f;

// =============================================================================
// emit_chunk_slice_fp8 -- slice a [E, ...] stacked tensor along PyTorch dim 0 (the
// expert axis) to [C, ...]. IDENTICAL idiom to the int4 emitter's emit_chunk_slice,
// generalized over out_dtype (fp8 for the weights, f32 for the scales, bf16 for rw).
// =============================================================================
inline std::vector<sh::tensor> emit_chunk_slice_fp8(
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
    if (i == 0) {
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
      {"slice", {in}, {{out_shape, out_dtype, std::nullopt}}, &sp, sizeof(sp)});
}

// =============================================================================
// emit_dynamic_quant -- per-token (last-axis) amax fp8 cast == vllm_gaudi dynamic_quant
// / moe_fused_expert_fp8 make_fp8(keep_dim=0), lifted to a rank-R tensor whose LAST dim
// is the feature axis to reduce. Returns {data_fp8 (same shape), fwd_scale (shape with
// last dim==1)}. The fwd_scale is the FORWARD recovery scale s = amax_eps/240 fed to
// fp8_gemm's scaleA; data_fp8 = convert_to_fp8(data, 240/amax_eps).
//
//   data       bf16  [..., F]          (rank 2 [T,H] or rank 3 [C,T,F])
//   shape      the PyTorch shape of data
// reduce over the LAST axis (the feature/contraction dim); keepDim so scale is [...,1].
// =============================================================================
inline std::vector<sh::tensor> emit_dynamic_quant(
    OpBackend* op, sh::graph& graph, synTensor data,
    const std::vector<int64_t>& shape) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32 = c10::ScalarType::Float;
  const int64_t rank = static_cast<int64_t>(shape.size());
  const int64_t reduce_dim = rank - 1;                 // last axis == feature

  std::vector<int64_t> scale_shape = shape;
  scale_shape[reduce_dim] = 1;                          // [...,1]

  // 1. |data|  (bf16)
  auto adata = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("abs_fwd"sv, bf16), {data}, {{shape, bf16, std::nullopt}}});

  // 2. amax bf16 = max over the LAST axis (keepDim). reversed-bit mask: dim d -> 1<<(rank-d-1).
  ns_Reduction::ParamsV2 rparams{};
  rparams.reductionDimensionMask =
      static_cast<unsigned>(1u << static_cast<unsigned>(rank - reduce_dim - 1));  // == 1<<0 == 1
  rparams.keepDim = true;
  auto amax_bf16 = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, bf16),
       {adata[0].get()}, {{scale_shape, bf16, std::nullopt}}, &rparams, sizeof(rparams)});

  // 3. amax -> f32 (scale arithmetic + the scale tensors are f32, like the live .float()).
  auto amax_f32 = OpBackend::BuildCast(op, graph, amax_bf16[0].get(), scale_shape, bf16, f32);

  // 4. amax_eps = amax + 1e-8  (dynamic_quant epsilon; 0/0-NaN guard).
  auto eps = OpBackend::BuildConstantTensor(op, graph, at::Scalar(1e-8), f32);
  auto amax_eps = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("add_fwd"sv, f32),
       {amax_f32.get(), eps.get()}, {{scale_shape, f32, std::nullopt}}});

  // 5. FORWARD s = amax_eps/240 (recovers fp8->real at the gemm); COMPRESS 1/s = 240/amax_eps.
  auto c240 = OpBackend::BuildConstantTensor(op, graph, at::Scalar(kMonoFp8MaxFnuz), f32);
  auto fwd_scale = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("div_fwd"sv, f32),
       {amax_eps[0].get(), c240.get()}, {{scale_shape, f32, std::nullopt}}});
  auto inv_scale = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("div_fwd"sv, f32),
       {c240.get(), amax_eps[0].get()}, {{scale_shape, f32, std::nullopt}}});

  // 6. data_fp8 = convert_to_fp8(data, 1/s). guid precision is the INPUT dtype (bf16);
  //    the fp8 out rides the output attr. ROUND_HALF_NE (GetCastParams non-stochastic).
  const at::ScalarType fp8 = c10::ScalarType::Float8_e4m3fn;
  ns_ConvertToFp8::ParamsV2 cparams{};
  cparams.round_mode = CAST_ROUND_HALF_NE;
  auto data_fp8 = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("convert_to_fp8"sv, bf16),
       {data, inv_scale[0].get()}, {{shape, fp8, std::nullopt}}, &cparams, sizeof(cparams)});

  std::vector<sh::tensor> out;
  out.reserve(2);
  out.emplace_back(std::move(data_fp8[0]));   // [0] = fp8 data
  out.emplace_back(std::move(fwd_scale[0]));  // [1] = FORWARD scale s
  return out;
}

// =============================================================================
// emit_swigluoai_tail -- the EXACT M3 swigluoai gate on a rank-3 [C,T,I] gate/up pair.
// gate/up are the CONTIGUOUS halves of gu[C,T,2I]. Returns h[C,T,I] bf16.
//   gate = gate.clamp(max=limit)             # UPPER bound only      (clamp_fwd max)
//   up   = up.clamp(min=-limit, max=limit)   # symmetric             (clamp_fwd min+max)
//   glu  = gate * sigmoid(gate * alpha)      # alpha INSIDE sigmoid  (mul const, sigmoid, mul)
//   h    = (up + 1.0) * glu                  # the +1, then (up+1)*glu (add const, mul)
// alpha/limit are M3 config swiglu_alpha=1.702 / swiglu_limit=7.0 (passed by the AddNode).
// Uses clamp_fwd (perf_lib has ns_Clamp / "clamp_fwd"; lowerBound/upperBound params) for
// the clamps; constant tensors for alpha and 1.0; sigmoid_fwd + mult_fwd from the int4 tail.
// =============================================================================
inline std::vector<sh::tensor> emit_swigluoai_tail(
    OpBackend* op, sh::graph& graph,
    synTensor gate, synTensor up,
    const std::vector<int64_t>& im_shape,   // [C,T,I]
    float alpha, float limit) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // gate.clamp(max=limit): clamp_fwd with upperBound=limit, lowerBound=-inf (the GLU gate
  // is only upper-bounded). ns_ClampKernel::Params carries {lowerBound, upperBound} f32.
  ns_ClampKernel::Params gate_clamp{};
  gate_clamp.lowerBound.f =-3.0e38f;                 // effectively -inf (no lower clamp on gate)
  gate_clamp.upperBound.f =limit;                    // = 7.0
  auto gate_c = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("clamp_fwd"sv, bf16),
       {gate}, {{im_shape, bf16, std::nullopt}}, &gate_clamp, sizeof(gate_clamp)});

  // up.clamp(min=-limit, max=limit): symmetric.
  ns_ClampKernel::Params up_clamp{};
  up_clamp.lowerBound.f =-limit;                     // = -7.0
  up_clamp.upperBound.f =limit;                      // = 7.0
  auto up_c = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("clamp_fwd"sv, bf16),
       {up}, {{im_shape, bf16, std::nullopt}}, &up_clamp, sizeof(up_clamp)});

  // ga = gate_c * alpha  (alpha INSIDE the sigmoid). alpha is a size-1 const broadcast.
  auto c_alpha = OpBackend::BuildConstantTensor(op, graph, at::Scalar(alpha), bf16);
  auto ga = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate_c[0].get(), c_alpha.get()}, {{im_shape, bf16, std::nullopt}}});

  // sig = sigmoid(ga)
  auto sig = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {ga[0].get()}, {{im_shape, bf16, std::nullopt}}});

  // glu = gate_c * sig
  auto glu = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate_c[0].get(), sig[0].get()}, {{im_shape, bf16, std::nullopt}}});

  // up1 = up_c + 1.0
  auto c_one = OpBackend::BuildConstantTensor(op, graph, at::Scalar(1.0), bf16);
  auto up1 = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("add_fwd"sv, bf16),
       {up_c[0].get(), c_one.get()}, {{im_shape, bf16, std::nullopt}}});

  // h = up1 * glu
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {up1[0].get(), glu[0].get()}, {{im_shape, bf16, std::nullopt}}});
}

// =============================================================================
// emit_fp8_batch_gemm -- batched fp8 x fp8 -> bf16 over the C-expert batch.
// A_fp8 [C,T,K] fp8, sA [C,T,1] f32 (per-token forward scale broadcast over K_out);
// B_fp8 [C,K,N] fp8, sB [C,1,N] f32 (per-OUTPUT-channel forward scale broadcast over T);
// out  [C,T,N] bf16 = (A.B) * sA * sB. Uses the fp8_gemm complex-guid (== Fp8Gemm::AddNode
// / fp8_gemm_gen.cpp) which broadcasts each scale over its complementary axis. The
// transpose flags are false/false (A is [C,T,K], B is [C,K,N] -- contract K, the
// [in,out] weight orientation the route-b stacks carry).
//
// NB: the live fp8_gemm_v2 in m3_expert_runtime_patch is 2D per-expert; here we run the
// BATCHED variant over the C axis. The fp8_gemm complex-guid accepts the leading batch
// dim the same way batch_gemm does (the int4 emitter's batch_gemm proves the [C,...]
// batch lowering); scales carry the matching leading C. (If a future bridge build only
// exposes the 2D fp8_gemm, the AddNode falls back to a per-c loop -- see the .h note.)
// =============================================================================
inline std::vector<sh::tensor> emit_fp8_batch_gemm(
    OpBackend* op, sh::graph& graph,
    synTensor A_fp8, synTensor sA, synTensor B_fp8, synTensor sB,
    const std::vector<int64_t>& out_shape,   // [C,T,N]
    bool is_result,
    bool transpose_b = false) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  ns_Fp8Gemm::ParamsV2 gparams{};
  gparams.transpose_a = false;
  gparams.transpose_b = transpose_b;
  std::vector<synTensor> gin{A_fp8, B_fp8, sA, sB,
                             nullptr /*bias*/, nullptr /*D (no accumulate)*/,
                             nullptr /*FUSE_CONVERT_MME (GC fills)*/};
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("fp8_gemm"sv, bf16),
       std::move(gin),
       {{out_shape, bf16, is_result ? std::optional<int>(0) : std::nullopt}},
       &gparams, sizeof(gparams)});
}

// =============================================================================
// emit_monolithic_chunk_fp8 -- the fp8 swigluoai chain for ONE chunk of C experts.
//   x        bf16  [T, H]               + T, H
//   E, chunk_start, C, I, OUT13(=2I)     (loop-invariant + the chunk window)
//   alpha, limit                         (swigluoai params: 1.702, 7.0)
//   w13_fp8  fp8   [E, H, 2I]    + shape (FULL stacked; route-b _w13_fp8_stack)
//   s13      f32   [E, 2I]       + shape (per-output-channel forward scale)
//   w2_fp8   fp8   [E, I, H]     + shape
//   s2       f32   [E, H]        + shape
//   rw       bf16  [E, T, 1]     + shape (dense per-(token,expert) router weight)
// Returns chunk_out[T,H] bf16. is_final ties the reduce to syn_out(0) (single-chunk case).
// =============================================================================
inline std::vector<sh::tensor> emit_monolithic_chunk_fp8(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t T, int64_t H,
    int64_t E, int64_t chunk_start, int64_t C, int64_t I, int64_t OUT13,
    float alpha, float limit,
    synTensor w13_fp8, const std::vector<int64_t>& w13_shape,
    synTensor s13,     const std::vector<int64_t>& s13_shape,
    synTensor w2_fp8,  const std::vector<int64_t>& w2_shape,
    synTensor s2,      const std::vector<int64_t>& s2_shape,
    synTensor rw,      const std::vector<int64_t>& rw_shape,
    bool is_final = false) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32 = c10::ScalarType::Float;
  const at::ScalarType fp8 = c10::ScalarType::Float8_e4m3fn;

  // ---- slice each FULL [E,...] stacked input to this chunk's [C,...] (dim 0) ----
  auto w13_c = emit_chunk_slice_fp8(op, graph, w13_fp8, w13_shape, chunk_start, C, fp8);  // [C,H,2I]
  auto s13_c = emit_chunk_slice_fp8(op, graph, s13,     s13_shape, chunk_start, C, f32);  // [C,2I]
  auto w2_c  = emit_chunk_slice_fp8(op, graph, w2_fp8,  w2_shape,  chunk_start, C, fp8);  // [C,I,H]
  auto s2_c  = emit_chunk_slice_fp8(op, graph, s2,      s2_shape,  chunk_start, C, f32);  // [C,H]
  auto rw_c  = emit_chunk_slice_fp8(op, graph, rw,      rw_shape,  chunk_start, C, bf16); // [C,T,1]

  // ---- dynamic-quant x[T,H] -> x_fp8[T,H] + x_s[T,1] (per-token), broadcast over C ----
  std::vector<int64_t> x_shape{T, H};
  auto xq = emit_dynamic_quant(op, graph, x, x_shape);     // {x_fp8[T,H], x_s[T,1]}
  // x_fp8 [T,H] -> [1,T,H] -> [C,T,H]
  std::vector<int64_t> x3_shape{1, T, H};
  auto x3 = OpBackend::BuildReshape(op, graph, xq[0].get(), x3_shape, fp8);
  std::vector<int64_t> xb_shape{C, T, H};
  auto x_b = OpBackend::BuildBroadcast(op, graph, x3.get(), xb_shape, fp8);
  // x_s [T,1] -> [1,T,1] -> [C,T,1]
  std::vector<int64_t> xs3_shape{1, T, 1};
  auto xs3 = OpBackend::BuildReshape(op, graph, xq[1].get(), xs3_shape, f32);
  std::vector<int64_t> xsb_shape{C, T, 1};
  auto x_s_b = OpBackend::BuildBroadcast(op, graph, xs3.get(), xsb_shape, f32);

  // s13_c [C,2I] -> [C,1,2I] per-output-channel scaleB (broadcast over T).
  std::vector<int64_t> s13b_shape{C, 1, OUT13};
  auto s13_b = OpBackend::BuildReshape(op, graph, s13_c[0].get(), s13b_shape, f32);

  // ---- fp8 batch_gemm #1: gu[C,T,2I] = bmm(x_fp8_b, w13c) * x_s_b * s13_b ----
  std::vector<int64_t> gu_shape{C, T, OUT13};
  auto gu = emit_fp8_batch_gemm(op, graph, x_b.get(), x_s_b.get(), w13_c[0].get(), s13_b.get(),
                                gu_shape, /*is_result=*/false);

  // ---- slice gate=gu[:,:,:I], up=gu[:,:,I:] (rank-3 feature dim 2) ----
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
        op, graph, {"slice", {in}, {{out_shape, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- swigluoai: h[C,T,I] = (up.clamp(-l,l)+1) * (gate.clamp(max=l) * sigmoid(alpha*gate.clamp)) ----
  std::vector<int64_t> im_shape{C, T, I};
  auto h = emit_swigluoai_tail(op, graph, gate[0].get(), up[0].get(), im_shape, alpha, limit);

  // ---- dynamic-quant h[C,T,I] -> h_fp8[C,T,I] + h_s[C,T,1] (per-(expert,token)) ----
  auto hq = emit_dynamic_quant(op, graph, h[0].get(), im_shape);   // {h_fp8[C,T,I], h_s[C,T,1]}

  // s2_c [C,H] -> [C,1,H] per-output-channel scaleB (broadcast over T).
  std::vector<int64_t> s2b_shape{C, 1, H};
  auto s2_b = OpBackend::BuildReshape(op, graph, s2_c[0].get(), s2b_shape, f32);

  // ---- fp8 batch_gemm #2: y[C,T,H] = bmm(h_fp8, w2c) * h_s * s2_b ----
  std::vector<int64_t> y_shape{C, T, H};
  auto y = emit_fp8_batch_gemm(op, graph, hq[0].get(), hq[1].get(), w2_c[0].get(), s2_b.get(),
                               y_shape, /*is_result=*/false);

  // ---- yw = y * rw_c[C,T,1] (per-(token,expert); broadcast over H) ----
  auto yw = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw_c[0].get()}, {{y_shape, bf16, std::nullopt}}});

  // ---- chunk_out = reduce_sum over C (dim 0) -> [T,H] ----
  ns_Reduction::ParamsV2 red_params{};
  red_params.reductionDimensionMask = 1u << static_cast<unsigned>(gu_rank - 0 - 1);  // 1<<2 == 4
  red_params.keepDim = false;
  std::vector<int64_t> out_shape{T, H};
  const std::optional<int> final_idx = is_final ? std::optional<int>(0) : std::nullopt;
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, bf16),
       {yw[0].get()}, {{out_shape, bf16, final_idx}}, &red_params, sizeof(red_params)});
}

// =============================================================================
// emit_monolithic_experts_fp8 -- the FULL E-expert fp8 monolithic chain. chunk E into
// ceil(E/CHUNK) windows, emit emit_monolithic_chunk_fp8 per window, add-chain the chunk
// outputs -> the single [T,H] result tied to syn_out(0). (Structurally IDENTICAL to the
// int4 emit_monolithic_experts; only the per-chunk helper + the input set differ.)
// =============================================================================
inline std::vector<sh::tensor> emit_monolithic_experts_fp8(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t T, int64_t H,
    int64_t E, int64_t CHUNK, int64_t I, int64_t OUT13,
    float alpha, float limit,
    synTensor w13_fp8, const std::vector<int64_t>& w13_shape,
    synTensor s13,     const std::vector<int64_t>& s13_shape,
    synTensor w2_fp8,  const std::vector<int64_t>& w2_shape,
    synTensor s2,      const std::vector<int64_t>& s2_shape,
    synTensor rw,      const std::vector<int64_t>& rw_shape) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  std::vector<int64_t> th_shape{T, H};

  const int64_t nchunks = (E + CHUNK - 1) / CHUNK;

  auto emit_chunk = [&](int64_t chunk_start, int64_t C, bool is_final) {
    return emit_monolithic_chunk_fp8(
        op, graph, x, T, H, E, chunk_start, C, I, OUT13, alpha, limit,
        w13_fp8, w13_shape, s13, s13_shape, w2_fp8, w2_shape, s2, s2_shape,
        rw, rw_shape, is_final);
  };

  if (nchunks == 1) {
    return emit_chunk(0, E, /*is_final=*/true);
  }

  const int64_t C0 = std::min<int64_t>(CHUNK, E);
  auto acc = emit_chunk(0, C0, /*is_final=*/false);
  for (int64_t j = 1; j < nchunks; ++j) {
    const int64_t cs = j * CHUNK;
    const int64_t C = std::min<int64_t>(CHUNK, E - cs);
    auto cj = emit_chunk(cs, C, /*is_final=*/false);
    const bool last = (j == nchunks - 1);
    const std::optional<int> final_idx = last ? std::optional<int>(0) : std::nullopt;
    acc = OpBackend::BuildNode(
        op, graph,
        {get_guid_with_precision("add_fwd"sv, bf16),
         {acc[0].get(), cj[0].get()}, {{th_shape, bf16, final_idx}}});
  }
  return acc;
}

} // namespace habana
