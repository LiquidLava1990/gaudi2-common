/*
 * moe_fused_grouped_fp8_emit.h -- the GROUPED (routed top-k) per-recipe emitter for
 * the M3 swigluoai-fp8 CONCURRENCY path. A strict SUBSET of the PROVEN monolithic-fp8
 * emit (moe_fused_monolithic_fp8_emit.h, the MMEL_M3_FUSED +27% win): it REUSES that
 * file's emit_dynamic_quant / emit_fp8_batch_gemm / emit_swigluoai_tail VERBATIM (this
 * header #includes it) and only re-orchestrates them for per-expert GATHERED tiles.
 *
 * THE DELTA vs the monolithic emit (both removals -> strictly fewer nodes):
 *   - x is a PER-EXPERT tile xg[E, Cap, H] (the Python wrapper gathered token-expert
 *     bins) -- NOT the shared [T,H] broadcast to [C,T,H]. So there is NO broadcast node:
 *     dynamic_quant runs on xg[E,Cap,H] directly and the E axis IS the batch.
 *   - the output is the raw per-expert yg[E, Cap, H] -- NO dense router-weight multiply
 *     and NO reduce_sum over E. The wrapper applies the per-(expert,slot) router weight
 *     and scatter-adds back to [T,H] (the routing the CPU oracle proves byte-exact).
 *
 * Net: ONE batched chain over the E experts (Cap << T tokens each):
 *   xg_fp8,xg_s = dynamic_quant(xg[E,Cap,H])                    (per (expert,token))
 *   gu[E,Cap,2I] = fp8_batch_gemm(xg_fp8, w13_fp8[E,H,2I], xg_s, s13[E,1,2I])  (fp8 MME)
 *   gate=gu[:,:,:I]; up=gu[:,:,I:]
 *   h[E,Cap,I]   = swigluoai(gate, up, alpha, limit)
 *   h_fp8,h_s    = dynamic_quant(h[E,Cap,I])
 *   y[E,Cap,H]   = fp8_batch_gemm(h_fp8, w2_fp8[E,I,H], h_s, s2[E,1,H])        (RESULT)
 * All intermediates NON-PERSISTENT -> the GC bundles dequant+swigluoai+the two MMEs
 * SRAM-resident, exactly as the monolithic op proved. M3 TP8: E=16, Cap tunable.
 *
 * The weights are the layer's PERSISTENT route-b fp8 stacks (op._w13_fp8_stack [E,H,2I]
 * + op._s13_stack [E,2I]; op._w2_fp8_stack [E,I,H] + op._s2_stack [E,H]); fed DIRECTLY,
 * NO per-expert slice (the whole E batch is processed -- Cap is small so SRAM holds it).
 * Keep in lockstep with moe_fused_monolithic_fp8_emit.h if its fp8/swigluoai math moves.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "moe_fused_monolithic_fp8_emit.h"   // emit_dynamic_quant / emit_fp8_batch_gemm /
                                             // emit_swigluoai_tail (REUSED VERBATIM)

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// emit_grouped_experts_fp8 -- the full E-expert grouped fp8 swigluoai chain over the
// gathered tiles xg[E,Cap,H]. Returns yg[E,Cap,H] bf16, the LAST gemm tied to
// syn_out(0) (final_result_index=0). One batched recipe; no chunk loop (Cap<<T keeps
// the [E,Cap,...] intermediates SRAM-resident -- the monolithic op chunked only because
// its broadcast x made [CHUNK,T,...] large; here there is no broadcast).
// =============================================================================
inline std::vector<sh::tensor> emit_grouped_experts_fp8(
    OpBackend* op, sh::graph& graph,
    synTensor xg, int64_t E, int64_t Cap, int64_t H, int64_t I, int64_t OUT13,
    float alpha, float limit,
    synTensor w13_fp8, const std::vector<int64_t>& /*w13_shape*/,
    synTensor s13,     const std::vector<int64_t>& /*s13_shape*/,
    synTensor w2_fp8,  const std::vector<int64_t>& /*w2_shape*/,
    synTensor s2,      const std::vector<int64_t>& /*s2_shape*/) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32 = c10::ScalarType::Float;

  // ---- dynamic-quant xg[E,Cap,H] -> xg_fp8[E,Cap,H] + xg_s[E,Cap,1] (per (expert,token)) ----
  std::vector<int64_t> xg_shape{E, Cap, H};
  auto xq = emit_dynamic_quant(op, graph, xg, xg_shape);      // {xg_fp8, xg_s}

  // ---- s13[E,2I] -> [E,1,2I] per-output-channel scaleB (broadcast over Cap) ----
  std::vector<int64_t> s13b_shape{E, 1, OUT13};
  auto s13_b = OpBackend::BuildReshape(op, graph, s13, s13b_shape, f32);

  // ---- fp8 batch_gemm #1: gu[E,Cap,2I] = bmm(xg_fp8, w13_fp8) * xg_s * s13_b ----
  std::vector<int64_t> gu_shape{E, Cap, OUT13};
  auto gu = emit_fp8_batch_gemm(op, graph, xq[0].get(), xq[1].get(), w13_fp8, s13_b.get(),
                                gu_shape, /*is_result=*/false);

  // ---- slice gate=gu[:,:,:I], up=gu[:,:,I:] (rank-3 feature dim 2) ----
  const int64_t r = 3;
  const int64_t dims[3] = {E, Cap, OUT13};
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    for (int64_t i = 0; i < r; ++i) {
      sp.axes[i] = static_cast<unsigned>(r - i - 1);   // TPC-reversed axis
      if (i == 2) {
        sp.starts[i] = static_cast<TSize>(begin);
        sp.ends[i] = static_cast<TSize>(end);
        sp.steps[i] = static_cast<TSize>(1);
      } else {
        sp.starts[i] = static_cast<TSize>(0);
        sp.ends[i] = static_cast<TSize>(dims[i]);
        sp.steps[i] = static_cast<TSize>(1);
      }
    }
    std::vector<int64_t> os{E, Cap, end - begin};
    return OpBackend::BuildNode(
        op, graph, {"slice", {in}, {{os, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- swigluoai: h[E,Cap,I] = (up.clamp(-l,l)+1) * (gate.clamp(max=l)*sigmoid(alpha*gate.clamp)) ----
  std::vector<int64_t> im_shape{E, Cap, I};
  auto h = emit_swigluoai_tail(op, graph, gate[0].get(), up[0].get(), im_shape, alpha, limit);

  // ---- dynamic-quant h[E,Cap,I] -> h_fp8[E,Cap,I] + h_s[E,Cap,1] (per (expert,token)) ----
  auto hq = emit_dynamic_quant(op, graph, h[0].get(), im_shape);

  // ---- s2[E,H] -> [E,1,H] per-output-channel scaleB (broadcast over Cap) ----
  std::vector<int64_t> s2b_shape{E, 1, H};
  auto s2_b = OpBackend::BuildReshape(op, graph, s2, s2b_shape, f32);

  // ---- fp8 batch_gemm #2: y[E,Cap,H] = bmm(h_fp8, w2_fp8) * h_s * s2_b  (the RESULT) ----
  std::vector<int64_t> y_shape{E, Cap, H};
  return emit_fp8_batch_gemm(op, graph, hq[0].get(), hq[1].get(), w2_fp8, s2_b.get(),
                             y_shape, /*is_result=*/true);
}

// =============================================================================
// emit_grouped_experts_glm_fp8 -- GLM-5.3 exact-clamp specialization of the
// existing routed/grouped FP8 engine.
//
// The routing geometry is unchanged: Python supplies xg[E,Cap,H], containing
// only the routed token rows for each local expert.  The two differences are
// model-contract details, not a second routing implementation:
//   * GLM stores the checkpoint weights as [E,2I,H] / [E,H,I], so both FP8
//     GEMMs consume B with transpose_b=true (no persistent transposed copy).
//   * GLM's activation is exactly
//       silu(clamp(gate,max=L)) * clamp(up,min=-L,max=L)
//     with L=10.  There is no M3 alpha and no GPT-SwiGLU (up + 1) term.
//
// Everything else reuses the champion grouped path: dynamic activation FP8,
// per-output-channel checkpoint scales, one batched recipe, and raw
// [E,Cap,H] output for the wrapper's weighted scatter-add.
// =============================================================================
inline std::vector<sh::tensor> emit_grouped_experts_glm_fp8(
    OpBackend* op, sh::graph& graph,
    synTensor xg, int64_t E, int64_t Cap, int64_t H, int64_t I, int64_t OUT13,
    float limit,
    synTensor w13_fp8, const std::vector<int64_t>& /*w13_shape*/,
    synTensor s13,     const std::vector<int64_t>& /*s13_shape*/,
    synTensor w2_fp8,  const std::vector<int64_t>& /*w2_shape*/,
    synTensor s2,      const std::vector<int64_t>& /*s2_shape*/) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32 = c10::ScalarType::Float;

  // xg[E,Cap,H] -> dynamic FP8, one scale per (expert,token) row.
  std::vector<int64_t> xg_shape{E, Cap, H};
  auto xq = emit_dynamic_quant(op, graph, xg, xg_shape);

  // GLM w13 is [E,2I,H].  transpose_b=true yields [E,Cap,2I]
  // directly and avoids a persistent [E,H,2I] transpose.
  std::vector<int64_t> s13b_shape{E, 1, OUT13};
  auto s13_b = OpBackend::BuildReshape(op, graph, s13, s13b_shape, f32);
  std::vector<int64_t> gu_shape{E, Cap, OUT13};
  auto gu = emit_fp8_batch_gemm(
      op, graph, xq[0].get(), xq[1].get(), w13_fp8, s13_b.get(),
      gu_shape, /*is_result=*/false, /*transpose_b=*/true);

  // Contiguous checkpoint split: gate first, up second.
  const int64_t r = 3;
  const int64_t dims[3] = {E, Cap, OUT13};
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    for (int64_t i = 0; i < r; ++i) {
      sp.axes[i] = static_cast<unsigned>(r - i - 1);
      if (i == 2) {
        sp.starts[i] = static_cast<TSize>(begin);
        sp.ends[i] = static_cast<TSize>(end);
      } else {
        sp.starts[i] = static_cast<TSize>(0);
        sp.ends[i] = static_cast<TSize>(dims[i]);
      }
    }
    std::vector<int64_t> os{E, Cap, end - begin};
    return OpBackend::BuildNode(
        op, graph, {"slice", {in}, {{os, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up = slice(gu[0].get(), I, OUT13);

  std::vector<int64_t> im_shape{E, Cap, I};
  ns_ClampKernel::Params gate_clamp{};
  gate_clamp.lowerBound.f = -3.0e38f;
  gate_clamp.upperBound.f = limit;
  auto gate_c = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("clamp_fwd"sv, bf16),
       {gate[0].get()}, {{im_shape, bf16, std::nullopt}},
       &gate_clamp, sizeof(gate_clamp)});

  ns_ClampKernel::Params up_clamp{};
  up_clamp.lowerBound.f = -limit;
  up_clamp.upperBound.f = limit;
  auto up_c = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("clamp_fwd"sv, bf16),
       {up[0].get()}, {{im_shape, bf16, std::nullopt}},
       &up_clamp, sizeof(up_clamp)});

  auto sig = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate_c[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto silu_gate = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate_c[0].get(), sig[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu_gate[0].get(), up_c[0].get()}, {{im_shape, bf16, std::nullopt}}});

  auto hq = emit_dynamic_quant(op, graph, h[0].get(), im_shape);

  // GLM w2 is [E,H,I].  transpose_b=true yields [E,Cap,H].
  std::vector<int64_t> s2b_shape{E, 1, H};
  auto s2_b = OpBackend::BuildReshape(op, graph, s2, s2b_shape, f32);
  std::vector<int64_t> y_shape{E, Cap, H};
  return emit_fp8_batch_gemm(
      op, graph, hq[0].get(), hq[1].get(), w2_fp8, s2_b.get(),
      y_shape, /*is_result=*/true, /*transpose_b=*/true);
}

// =============================================================================
// One canonical smart-kernel emitter.  Model adapters select a static recipe
// descriptor; they do not own a second kernel implementation.
//
// activation_mode: 0 = GPT-SwiGLU/M3, 1 = exact clamped-SiLU/GLM
// weight_layout:   0 = [E,H,2I]/[E,I,H], 1 = [E,2I,H]/[E,H,I]
//
// These scalar descriptors are compile-time constants when AddNode lowers a
// bucket, so each captured recipe contains only the requested path.  The op
// name, routing contract, FP8 quantization, GEMMs, and output ABI stay shared.
// =============================================================================
inline std::vector<sh::tensor> emit_grouped_experts_fp8_smart(
    OpBackend* op, sh::graph& graph,
    synTensor xg, int64_t E, int64_t Cap, int64_t H, int64_t I, int64_t OUT13,
    int64_t activation_mode, int64_t weight_layout, float alpha, float limit,
    synTensor w13_fp8, const std::vector<int64_t>& w13_shape,
    synTensor s13,     const std::vector<int64_t>& s13_shape,
    synTensor w2_fp8,  const std::vector<int64_t>& w2_shape,
    synTensor s2,      const std::vector<int64_t>& s2_shape) {
  if (activation_mode == 0 && weight_layout == 0) {
    return emit_grouped_experts_fp8(
        op, graph, xg, E, Cap, H, I, OUT13, alpha, limit,
        w13_fp8, w13_shape, s13, s13_shape,
        w2_fp8, w2_shape, s2, s2_shape);
  }
  TORCH_CHECK(
      activation_mode == 1 && weight_layout == 1,
      "[fused_moe_grouped_fp8_smart] unsupported descriptor activation_mode=",
      activation_mode, " weight_layout=", weight_layout);
  return emit_grouped_experts_glm_fp8(
      op, graph, xg, E, Cap, H, I, OUT13, limit,
      w13_fp8, w13_shape, s13, s13_shape,
      w2_fp8, w2_shape, s2, s2_shape);
}

} // namespace habana
