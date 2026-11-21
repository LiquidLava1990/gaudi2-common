/*
 * moe_fused_expert_int4fp8_emit.h -- W4A8 ROUTED per-expert emitter: the int4
 * ROUTED chain (moe_fused_expert_emit.h::emit_one_expert) with the bf16
 * batch_gemms replaced by the DIRECT int4->fp8 + native fp8 MME path proven in
 * moe_fused_monolithic_int4fp8_emit.h (the W4A8 monolithic op that ships at
 * ~1.33x on GLM concurrency). This is the SINGLE-STREAM + MTP-verify sibling of
 * that monolithic op: it processes ONE routed expert at a time (2D per-expert
 * gemms [M,H]/[H,2I], NOT the all-expert 3D batch), so the K-expert routed op
 * (moe_fused_expert_int4fp8_backend_shared.h) calls it K times and add-folds the
 * weighted results -- EXACTLY as the bf16 routed op calls emit_one_expert.
 *
 * THE TRANSFORM (verbatim the monolithic's DIRECT-fp8 scheme; see that header's
 * long rationale + the DPQ proof, NOT repeated here):
 *   int4 -> dequant -> fp8 DIRECTLY (dequantize_4_bit with Float8_e4m3fn out;
 *           the int4 group scale is FOLDED into the fp8 weight so the gemm's
 *           weight scale sB == 1, built by build_ones). NO per-step weight amax,
 *           NO bf16->fp8 convert -> the dequant cost stays ~the bf16 path while
 *           the MME runs fp8 (2x bf16). This is the version that FLIPPED the
 *           monolithic from slower to faster (W4A8_BUILD.md FINAL RESULT).
 *   x / h -> per-TOKEN fp8 quant (emit_fp8_quant, reduce the feature axis ->
 *           [M,1] scale) -- the EXACT proven helper from the monolithic header,
 *           reused by include (no copy -> no drift).
 *   gu / y = fp8_gemm(X_fp8, W_fp8, sA=[M,1] per-token, sB=[1,N]==1 folded) ->
 *           bf16 (emit_fp8_batch_gemm_i4f8; the fp8_gemm complex-guid is
 *           dimension-agnostic -- it takes the SAME 2D operands the proven
 *           single-expert fp8 op feeds it, sA[M,1]/sB[1,N]).
 * Standard SiLU (GLM glm_moe) + the final `* router_weight` broadcast are kept
 * BYTE-FOR-BYTE from emit_one_expert (the [1,1] T==1 fold / [T,1] T>1 per-token
 * column, both broadcast over [M,H]).
 *
 * WHY REUSE THE MONOLITHIC HEADER (vs re-copying the fp8 helpers): emit_fp8_quant
 * and emit_fp8_batch_gemm_i4f8 are the PROVEN, on-device-validated helpers; their
 * bodies are dimension-agnostic (emit_fp8_quant takes an explicit shape+reduce_dim;
 * emit_fp8_batch_gemm_i4f8 just feeds the operands to the fp8_gemm guid). Reusing by
 * include keeps them byte-identical to the shipped monolithic op (a future fix lands
 * in BOTH), and the unused monolithic chunk emitters are inline-and-unreferenced here
 * (not emitted). The only NEW code is emit_one_expert_i4f8 (a UNIQUE symbol -- no ODR
 * clash with the bf16 emit_one_expert when both .so co-load) and its three LOCAL
 * lambdas (reinterpret_u4 / dequant_fp8 / build_ones), which are function-local (no
 * external linkage).
 *
 * Frontend-free (pulls only op_backend.h / hpu_op_helper.h + synapse params, via
 * the monolithic emit header) so either the eager or the lazy K-expert TU can
 * include it.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr / NodeOutputAttr
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision
#include "backend/synapse_helpers/graph.h"

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"     // syn_type_* enums, HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp"   // synGEMMParams + synSliceParamsV2
#include "perf_lib_layer_params.h"    // ns_CastKernel::ParamsV3, ns_Fp8Gemm::ParamsV2

// The PROVEN fp8 helpers, reused byte-for-byte (emit_fp8_quant per-token cast +
// emit_fp8_batch_gemm_i4f8 the fp8_gemm complex-guid + kI4F8MaxFnuz). Pulls only the
// same frontend-free headers this TU already needs.
#include "moe_fused_monolithic_int4fp8_emit.h"

#include <algorithm>   // std::fill_n
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// emit_one_expert_i4f8 -- the W4A8 (int4->fp8 + native fp8 MME) per-expert chain
// for ONE routed expert; returns its [M,H] bf16 output (router weight folded by
// the final mult). The K-expert routed-int4fp8 op calls this K times and add-folds
// the y_k's. Signature MIRRORS emit_one_expert (the bf16 routed emit) EXACTLY so
// the routed backend's k-fold is unchanged -- only the op-internal compute is fp8.
//
// Inputs (all PyTorch row-major; same per-expert shapes as the bf16 routed op):
//   x        bf16  [M, H]            (shared across all K experts)
//   M,H,I,OUT13                      (loop-invariant dims; OUT13 == 2I)
//   w13p     int32 [H, 2I/8]   + w13p_shape   (this expert's packed gate+up weight)
//   w13s     bf16  [H/GS, 2I]                 (this expert's gate+up int4 group scales)
//   w13z     int32 [H/GS, 2I/8] + w13z_shape  (this expert's gate+up zero-points)
//   w2p      int32 [I, H/8]     + w2p_shape    (this expert's packed down weight)
//   w2s      bf16  [I/GS, H]                   (this expert's down int4 group scales)
//   w2z      int32 [I/GS, H/8]  + w2z_shape    (this expert's down zero-points)
//   rw       bf16  [1, 1] (T==1) or [T, 1] (T>1)   (this expert's router weight)
//
// is_final_result == true ONLY for the K==1 degenerate case (the final `* rw` node
// IS the op result). Otherwise the y_k is graph-internal (NON-PERSISTENT) and the
// caller chains the add-tree onto it -- so the GC bundles each expert's two
// int4->fp8 dequants into its fp8 MMEs, preserving the SRAM-residency win PER
// expert exactly as the bf16 routed op does.
// =============================================================================
inline std::vector<sh::tensor> emit_one_expert_i4f8(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t M, int64_t H, int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw,
    bool is_final_result = false) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType fp8e = c10::ScalarType::Float8_e4m3fn;
  const at::ScalarType f32e = c10::ScalarType::Float;

  // --- reinterpret_cast int32-packed -> uint4 (VERBATIM the bf16 routed emit):
  //     unpacked PyTorch shape == packed shape with the LAST dim *8. ---
  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;  // packed int32 last dim (cols/8) -> uint4 last dim (cols)
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast",
         {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // --- int4 -> fp8 DIRECTLY (the monolithic's proven dequant_fp8, 2D per-expert):
  //     dequantize_4_bit with a Float8_e4m3fn out_dtype emits fp8 straight from int4
  //     (the 16 int4 levels * group_scale land in e4m3, DPQ +0.28ppl). The group
  //     scale is FOLDED into the stored fp8 value, so the gemm's weight scale sB==1
  //     (build_ones). NO per-step amax, NO separate bf16->fp8 convert -> the dequant
  //     cost is ~the bf16 path while the MME runs fp8 2x. NON-PERSISTENT out (SRAM,
  //     bundled into the following fp8 MME). ns_CastKernel::ParamsV3 default (NO
  //     CAST_CLIP -- byte-for-byte the monolithic dequant_fp8 that shipped). ---
  auto dequant_fp8 = [&](synTensor w_packed, const std::vector<int64_t>& w_shape,
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
        {get_guid_with_precision("dequantize_4_bit"sv, fp8e),
         std::move(in),
         {{out_shape, fp8e, std::nullopt}},   // int4 -> fp8 in ONE node (NON-PERSISTENT)
         &params, sizeof(params)});
  };

  // --- sB == 1 (the int4 group scale is folded into the fp8 weight). One [1,N] f32
  //     const per gemm. Scalar const -> reshape to [1,1] (exactly 1 elem, valid) ->
  //     broadcast to [1,N] (the 2D analog of the monolithic's [1,1,1]->[C,1,N]; the
  //     reshape target must hold exactly 1 element, NOT [1,N] -- the bug the
  //     monolithic header documents). [1,N] sB is the proven per-channel weight-scale
  //     shape the single-expert fp8 op feeds fp8_gemm. ---
  auto build_ones = [&](const std::vector<int64_t>& shp) {
    auto one = OpBackend::BuildConstantTensor(op, graph, at::Scalar(1.0), f32e);
    std::vector<int64_t> s11{1, 1};   // scalar -> [1,1] (1 elem, valid reshape)
    auto r = OpBackend::BuildReshape(op, graph, one.get(), s11, f32e);
    return OpBackend::BuildBroadcast(op, graph, r.get(), shp, f32e);  // [1,1] -> [1,N]
  };

  // ============ W13: int4 -> fp8 DIRECTLY (sB=1) -> [H, 2I] fp8 ============
  auto W13_fp8 = dequant_fp8(w13p, w13p_shape, w13s, w13z, w13z_shape);
  std::vector<int64_t> s13_shape{1, OUT13};
  auto s13_one = build_ones(s13_shape);

  // ============ activation x[M,H] -> per-token fp8 (reduce feature dim 1) ============
  std::vector<int64_t> x_shape{M, H};
  auto xq = emit_fp8_quant(op, graph, x, x_shape, /*reduce_dim=*/1);  // {x_fp8[M,H], x_s[M,1]}

  // ============ gemm #1: gu[M, 2I] = fp8_gemm(x_fp8, W13_fp8) (contract H) ============
  std::vector<int64_t> gu_shape{M, OUT13};
  auto gu = emit_fp8_batch_gemm_i4f8(op, graph,
      xq[0].get(), xq[1].get(), W13_fp8[0].get(), s13_one.get(), gu_shape, /*is_result=*/false);

  // ---- slice gate/up along PyTorch feature dim 1 (2I): gate=gu[:, :I], up=gu[:, I:] ----
  const int64_t gu_rank = 2;
  const int64_t slice_dim = 1;
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    const int64_t gu_dims[gu_rank] = {M, OUT13};
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
    std::vector<int64_t> out_shape{M, end - begin};  // PyTorch [M, I]
    return OpBackend::BuildNode(
        op, graph,
        {"slice", {in}, {{out_shape, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- SiLU(gate)=gate*sigmoid(gate); h = SiLU(gate) * up  (BYTE-FOR-BYTE bf16 routed) ----
  std::vector<int64_t> im_shape{M, I};
  auto sgte = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()}, {{im_shape, bf16, std::nullopt}}});

  // ============ h[M,I] -> per-token fp8 (reduce feature dim 1) ============
  auto hq = emit_fp8_quant(op, graph, h[0].get(), im_shape, /*reduce_dim=*/1);  // {h_fp8[M,I], h_s[M,1]}

  // ============ W2: int4 -> fp8 DIRECTLY (sB=1) -> [I, H] fp8 ============
  auto W2_fp8 = dequant_fp8(w2p, w2p_shape, w2s, w2z, w2z_shape);
  std::vector<int64_t> s2_shape{1, H};
  auto s2_one = build_ones(s2_shape);

  // ============ gemm #2: y[M, H] = fp8_gemm(h_fp8, W2_fp8) (contract I) ============
  std::vector<int64_t> mh_shape{M, H};
  auto y = emit_fp8_batch_gemm_i4f8(op, graph,
      hq[0].get(), hq[1].get(), W2_fp8[0].get(), s2_one.get(), mh_shape, /*is_result=*/false);

  // ---- y_k = y * router_weight ([1,1] T==1 fold / [T,1] T>1 per-token, broadcast
  //      over [M,H]) -- BYTE-FOR-BYTE the bf16 routed emit's final node. ----
  const std::optional<int> final_idx =
      is_final_result ? std::optional<int>(0) : std::nullopt;
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw},
       {{mh_shape, bf16, final_idx}}});
}

} // namespace habana
