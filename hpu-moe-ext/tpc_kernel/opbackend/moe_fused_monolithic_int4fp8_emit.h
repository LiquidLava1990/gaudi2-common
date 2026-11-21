/*
 * moe_fused_monolithic_int4fp8_emit.h -- W4A8 MONOLITHIC (ALL-E) MoE emitter:
 * int4-resident experts DEQUANTIZED IN-KERNEL TO FP8, run through the NATIVE fp8 MME
 * (fp8 x fp8 -> bf16, 2x bf16 rate), SRAM-resident. The int4 weights stay HBM-resident
 * (GLM-753B can't hold ~90GB/card of fp8); the dequant -> fp8 cast happens per-chunk in
 * SRAM (non-persistent), so we capture BOTH the int4 storage win AND the 2x fp8 MME.
 *
 * VALIDATED EXTERNALLY: Intel DPQ (arXiv 2505.14638) does int4->fp8->fp8-matmul on
 * Gaudi-2/3 for up to 3x over W4A16 at +0.28 ppl. The "fnuz/fn trap" is a myth on Gaudi
 * (AMD-only load-time scale fold); direct int4->fp8 is fine. Gaudi-2 e4m3 saturates at
 * +/-240 -> the per-output-channel weight scale + per-token activation scale both
 * calibrate to 240 (kMonoFp8MaxFnuz), exactly the fp8 monolithic op's scheme.
 *
 * This is moe_fused_monolithic_emit.h (the int4->bf16 chain) with the bf16 batch_gemms
 * replaced by the fp8 path of moe_fused_monolithic_fp8_emit.h:
 *   int4 -> dequant -> W_bf16 (SRAM)               [from the int4 emit]
 *   W_bf16 -> per-OUTPUT-channel fp8 quant -> W_fp8 + s_w   [NEW: reduce over the
 *                                                            CONTRACTION axis (dim 1)]
 *   x/h    -> per-TOKEN fp8 quant -> X_fp8 + s_x            [emit_dynamic_quant, reduce last]
 *   gu/y   = fp8_batch_gemm(X_fp8, W_fp8, sA=s_x, sB=s_w) -> bf16   [native fp8 MME 2x]
 * Standard SiLU (GLM glm_moe, NOT M3 swigluoai). Same chunking / router-weight / reduce
 * / add-chain as the int4 emitter; one recipe per (E,CHUNK,T,H,I).
 *
 * Frontend-free (pulls only op_backend.h / hpu_op_helper.h + synapse params).
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
#include "synapse_common_types.hpp"      // synGEMMParams + synSliceParamsV2
#include "perf_lib_layer_params.h"       // ns_CastKernel::ParamsV3, ns_Reduction::ParamsV2,
                                         // ns_ConvertToFp8::ParamsV2, ns_Fp8Gemm::ParamsV2

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Gaudi-2 e4m3 max usable magnitude (== fp8 emit's kMonoFp8MaxFnuz). The dynamic-quant
// denominator so the cast stays inside the real e4m3 range; the cast(1/s)<->gemm(s) pair
// cancels the fn<->fnuz factor (see moe_fused_expert_fp8_backend.cpp PROOF).
static constexpr float kI4F8MaxFnuz = 240.0f;

// =============================================================================
// emit_chunk_slice_i4f8 -- slice a [E,...] stacked tensor along PyTorch dim 0 (expert
// axis) to [C,...]. VERBATIM the int4/fp8 emitters' emit_chunk_slice idiom.
// =============================================================================
inline std::vector<sh::tensor> emit_chunk_slice_i4f8(
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
    sp.axes[i] = static_cast<unsigned>(rank - i - 1);     // TPC-reversed axis
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
// emit_fp8_quant -- amax fp8 cast over a CHOSEN reduce axis (generalizes the fp8
// emit's emit_dynamic_quant, which was hard-wired to the last axis).
//   reduce_dim = rank-1  -> per-TOKEN activation scale  (x[T,H] / h[C,T,I])
//   reduce_dim = 1       -> per-OUTPUT-channel weight scale (W[C,K,N], reduce K=dim1,
//                           keep N) -- the per-channel scale the fp8 gemm's sB wants.
// Returns {data_fp8 (same shape), fwd_scale (shape with reduce_dim==1)}. fwd_scale s =
// amax_eps/240 (fed to fp8_gemm scaleA/B); data_fp8 = convert_to_fp8(data, 240/amax_eps).
// =============================================================================
inline std::vector<sh::tensor> emit_fp8_quant(
    OpBackend* op, sh::graph& graph, synTensor data,
    const std::vector<int64_t>& shape, int64_t reduce_dim) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32 = c10::ScalarType::Float;
  const at::ScalarType fp8 = c10::ScalarType::Float8_e4m3fn;
  const int64_t rank = static_cast<int64_t>(shape.size());

  std::vector<int64_t> scale_shape = shape;
  scale_shape[reduce_dim] = 1;

  // 1. |data|
  auto adata = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("abs_fwd"sv, bf16), {data}, {{shape, bf16, std::nullopt}}});

  // 2. amax = max over reduce_dim (keepDim). reversed-bit mask: dim d -> 1<<(rank-d-1).
  ns_Reduction::ParamsV2 rparams{};
  rparams.reductionDimensionMask =
      static_cast<unsigned>(1u << static_cast<unsigned>(rank - reduce_dim - 1));
  rparams.keepDim = true;
  auto amax_bf16 = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, bf16),
       {adata[0].get()}, {{scale_shape, bf16, std::nullopt}}, &rparams, sizeof(rparams)});

  // 3. amax -> f32
  auto amax_f32 = OpBackend::BuildCast(op, graph, amax_bf16[0].get(), scale_shape, bf16, f32);

  // 4. amax_eps = amax + 1e-8
  auto eps = OpBackend::BuildConstantTensor(op, graph, at::Scalar(1e-8), f32);
  auto amax_eps = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("add_fwd"sv, f32),
       {amax_f32.get(), eps.get()}, {{scale_shape, f32, std::nullopt}}});

  // 5. fwd s = amax_eps/240 ; inv 1/s = 240/amax_eps
  auto c240 = OpBackend::BuildConstantTensor(op, graph, at::Scalar(kI4F8MaxFnuz), f32);
  auto fwd_scale = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("div_fwd"sv, f32),
       {amax_eps[0].get(), c240.get()}, {{scale_shape, f32, std::nullopt}}});
  auto inv_scale = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("div_fwd"sv, f32),
       {c240.get(), amax_eps[0].get()}, {{scale_shape, f32, std::nullopt}}});

  // 6. data_fp8 = convert_to_fp8(data, 1/s). guid precision = INPUT dtype (bf16); fp8 out
  //    rides the output attr. ROUND_HALF_NE.
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
// emit_fp8_batch_gemm_i4f8 -- batched fp8 x fp8 -> bf16 (== the fp8 emit's
// emit_fp8_batch_gemm). A_fp8[C,T,K], sA[C,T,1]; B_fp8[C,K,N], sB[C,1,N]; out[C,T,N] bf16.
// =============================================================================
inline std::vector<sh::tensor> emit_fp8_batch_gemm_i4f8(
    OpBackend* op, sh::graph& graph,
    synTensor A_fp8, synTensor sA, synTensor B_fp8, synTensor sB,
    const std::vector<int64_t>& out_shape, bool is_result) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  ns_Fp8Gemm::ParamsV2 gparams{};
  gparams.transpose_a = false;
  gparams.transpose_b = false;
  std::vector<synTensor> gin{A_fp8, B_fp8, sA, sB,
                             nullptr /*bias*/, nullptr /*D*/, nullptr /*FUSE_CONVERT_MME*/};
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("fp8_gemm"sv, bf16),
       std::move(gin),
       {{out_shape, bf16, is_result ? std::optional<int>(0) : std::nullopt}},
       &gparams, sizeof(gparams)});
}

// =============================================================================
// emit_monolithic_chunk_i4f8 -- the W4A8 chain for ONE chunk of C experts. Slices the
// FULL int4 [E,...] stacks to [C,...], dequants int4->bf16 (SRAM), per-channel-fp8-quants
// the weights, per-token-fp8-quants x/h, runs the two fp8 batch-gemms with standard SiLU,
// applies the per-(token,expert) router weight and reduces over the chunk's expert axis.
// Returns chunk_out[T,H] bf16. Signature mirrors emit_monolithic_chunk (int4).
// =============================================================================
inline std::vector<sh::tensor> emit_monolithic_chunk_i4f8(
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
  auto w13p_c = emit_chunk_slice_i4f8(op, graph, w13p, w13p_shape, chunk_start, C, at::kInt);
  auto w13s_c = emit_chunk_slice_i4f8(op, graph, w13s, w13s_shape, chunk_start, C, bf16);
  auto w13z_c = emit_chunk_slice_i4f8(op, graph, w13z, w13z_shape, chunk_start, C, at::kInt);
  auto w2p_c  = emit_chunk_slice_i4f8(op, graph, w2p,  w2p_shape,  chunk_start, C, at::kInt);
  auto w2s_c  = emit_chunk_slice_i4f8(op, graph, w2s,  w2s_shape,  chunk_start, C, bf16);
  auto w2z_c  = emit_chunk_slice_i4f8(op, graph, w2z,  w2z_shape,  chunk_start, C, at::kInt);
  auto rw_c   = emit_chunk_slice_i4f8(op, graph, rw,   rw_shape,   chunk_start, C, bf16);

  std::vector<int64_t> w13p_cs{C, H, OUT13 / 8};
  std::vector<int64_t> w13z_cs{C, w13z_shape[1], OUT13 / 8};
  std::vector<int64_t> w2p_cs{C, I, H / 8};
  std::vector<int64_t> w2z_cs{C, w2z_shape[1], H / 8};

  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast",
         {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // int4 -> bf16 dequant (SRAM, non-persistent) -- VERBATIM the int4 emitter's dequant.
  auto dequant_bf16 = [&](synTensor w_packed, const std::vector<int64_t>& w_cshape,
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
         {{out_shape, bf16, std::nullopt}},
         &params, sizeof(params)});
  };

  // int4 -> fp8 DIRECTLY (DPQ scheme: w_fp8 = (w4-z)*group_scale, cast to e4m3 in one op;
  // the group scale is FOLDED so sB=1 at the gemm). NO per-step amax, NO separate bf16->fp8
  // convert -> the dequant cost is ~the bf16 path while the MME runs fp8 2x. The 16 int4
  // levels*scale land in e4m3 (DPQ +0.28ppl). Output dtype on the out-attr selects the f8 guid.
  const at::ScalarType fp8e = c10::ScalarType::Float8_e4m3fn;
  const at::ScalarType f32e = c10::ScalarType::Float;
  auto dequant_fp8 = [&](synTensor w_packed, const std::vector<int64_t>& w_cshape,
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
        {get_guid_with_precision("dequantize_4_bit"sv, fp8e),
         std::move(in),
         {{out_shape, fp8e, std::nullopt}},
         &params, sizeof(params)});
  };
  // sB=1 (the group scale is folded into the fp8 weight). One [C,1,N] f32 const per gemm.
  auto build_ones = [&](const std::vector<int64_t>& shp) {
    auto one = OpBackend::BuildConstantTensor(op, graph, at::Scalar(1.0), f32e);
    std::vector<int64_t> s111{1, 1, 1};   // scalar -> [1,1,1] (1 elem, valid reshape)
    auto r = OpBackend::BuildReshape(op, graph, one.get(), s111, f32e);
    return OpBackend::BuildBroadcast(op, graph, r.get(), shp, f32e);  // [1,1,1] -> [C,1,N]
  };

  // ============ W13: int4 -> fp8 DIRECTLY (sB=1) ============
  auto W13_fp8 = dequant_fp8(w13p_c[0].get(), w13p_cs, w13s_c[0].get(), w13z_c[0].get(), w13z_cs);
  std::vector<int64_t> s13_shape{C, 1, OUT13};
  auto s13_one = build_ones(s13_shape);

  // ============ activation x[T,H] -> per-token fp8 -> broadcast to [C,T,H] ============
  std::vector<int64_t> x2_shape{T, H};
  auto xq = emit_fp8_quant(op, graph, x, x2_shape, /*reduce_dim=*/1);  // xq[0]=x_fp8[T,H], xq[1]=x_s[T,1]
  const at::ScalarType fp8 = c10::ScalarType::Float8_e4m3fn;
  const at::ScalarType f32 = c10::ScalarType::Float;
  std::vector<int64_t> x3_shape{1, T, H};
  auto x_fp8_3 = OpBackend::BuildReshape(op, graph, xq[0].get(), x3_shape, fp8);
  std::vector<int64_t> xb_shape{C, T, H};
  auto x_fp8_b = OpBackend::BuildBroadcast(op, graph, x_fp8_3.get(), xb_shape, fp8);
  std::vector<int64_t> xs3_shape{1, T, 1};
  auto x_s_3 = OpBackend::BuildReshape(op, graph, xq[1].get(), xs3_shape, f32);
  std::vector<int64_t> xsb_shape{C, T, 1};
  auto x_s_b = OpBackend::BuildBroadcast(op, graph, x_s_3.get(), xsb_shape, f32);

  // ============ gemm #1: gu[C,T,2I] = fp8_bmm(x_fp8_b, W13_fp8) ============
  std::vector<int64_t> gu_shape{C, T, OUT13};
  auto gu = emit_fp8_batch_gemm_i4f8(op, graph,
      x_fp8_b.get(), x_s_b.get(), W13_fp8[0].get(), s13_one.get(), gu_shape, /*is_result=*/false);

  // ---- slice gate/up along feature dim 2 ----
  const int64_t gu_rank = 3;
  const int64_t gu_dims[3] = {C, T, OUT13};
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    for (int64_t i = 0; i < gu_rank; ++i) {
      sp.axes[i] = static_cast<unsigned>(gu_rank - i - 1);
      if (i == 2) { sp.starts[i] = static_cast<TSize>(begin); sp.ends[i] = static_cast<TSize>(end); }
      else { sp.starts[i] = static_cast<TSize>(0); sp.ends[i] = static_cast<TSize>(gu_dims[i]); }
      sp.steps[i] = static_cast<TSize>(1);
    }
    std::vector<int64_t> out_shape{C, T, end - begin};
    return OpBackend::BuildNode(
        op, graph, {"slice", {in}, {{out_shape, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- standard SiLU: silu=gate*sigmoid(gate); h=silu*up  (GLM glm_moe) ----
  std::vector<int64_t> im_shape{C, T, I};
  auto sgte = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()}, {{im_shape, bf16, std::nullopt}}});

  // ============ h[C,T,I] -> per-token fp8 (reduce last=I) ============
  auto hq = emit_fp8_quant(op, graph, h[0].get(), im_shape, /*reduce_dim=*/2);  // hq[0]=h_fp8, hq[1]=h_s[C,T,1]

  // ============ W2: int4 -> fp8 DIRECTLY (sB=1) ============
  auto W2_fp8 = dequant_fp8(w2p_c[0].get(), w2p_cs, w2s_c[0].get(), w2z_c[0].get(), w2z_cs);
  std::vector<int64_t> s2_shape{C, 1, H};
  auto s2_one = build_ones(s2_shape);

  // ============ gemm #2: y[C,T,H] = fp8_bmm(h_fp8, W2_fp8) ============
  std::vector<int64_t> y_shape{C, T, H};
  auto y = emit_fp8_batch_gemm_i4f8(op, graph,
      hq[0].get(), hq[1].get(), W2_fp8[0].get(), s2_one.get(), y_shape, /*is_result=*/false);

  // ---- yw = y * router_weight rw_c[C,T,1] (broadcast over H) ----
  auto yw = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw_c[0].get()}, {{y_shape, bf16, std::nullopt}}});

  // ---- chunk_out = reduce_sum over C (dim 0) -> [T,H] ----
  ns_Reduction::ParamsV2 red_params{};
  red_params.reductionDimensionMask = 1u << static_cast<unsigned>(gu_rank - 0 - 1);
  red_params.keepDim = false;
  std::vector<int64_t> out_shape{T, H};
  const std::optional<int> final_idx = is_final ? std::optional<int>(0) : std::nullopt;
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("reduce_sum_multi_dim_fwd"sv, bf16),
       {yw[0].get()}, {{out_shape, bf16, final_idx}}, &red_params, sizeof(red_params)});
}

// =============================================================================
// emit_monolithic_experts_i4f8 -- chunk E into ceil(E/CHUNK) windows, emit the W4A8
// chunk per window, add-chain -> the single [T,H] result tied to syn_out(0). VERBATIM
// the int4 emitter's emit_monolithic_experts add-tree, calling emit_monolithic_chunk_i4f8.
// =============================================================================
inline std::vector<sh::tensor> emit_monolithic_experts_i4f8(
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
  std::vector<int64_t> th_shape{T, H};

  const int64_t nchunks = (E + CHUNK - 1) / CHUNK;
  auto emit_chunk = [&](int64_t chunk_start, int64_t C, bool is_final) {
    return emit_monolithic_chunk_i4f8(
        op, graph, x, T, H, E, chunk_start, C, I, OUT13,
        w13p, w13p_shape, w13s, w13s_shape, w13z, w13z_shape,
        w2p, w2p_shape, w2s, w2s_shape, w2z, w2z_shape, rw, rw_shape, is_final);
  };

  if (nchunks == 1) return emit_chunk(0, E, /*is_final=*/true);

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
