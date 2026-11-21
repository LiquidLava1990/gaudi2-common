/*
 * moe_fused_moe_binned_int4fp8_emit.h -- W4A8 BINNED (capacity-permuted grouped-GEMM)
 * MoE emitter (WAVE-2). The SPARSE-compute sibling of the DENSE monolithic op
 * (moe_fused_monolithic_int4fp8_emit.h): instead of computing EVERY local expert over
 * EVERY token (dense E*T, the dense [E,T] router zeroing ~97% of the products at the
 * live EP TP=8 config), it processes a permuted, capacity-binned activation
 * x_perm[E,C,H] where each expert e owns ONLY its C routed-token rows (E*C work,
 * C ~ per-expert load << T). The Python wrapper (hpu_moe_ext/binned_moe.py) builds
 * x_perm by an on-device static-capacity scatter and un-bins the output.
 *
 *   WHY THIS IS A COMPUTE+WORKING-SET LEVER, NOT A READ LEVER (honest, grounded):
 *   the union of distinct LOCAL experts hit at c128 is ~E (all 32) -> every expert is
 *   STILL dequanted once/step (read is already amortized once-per-expert-per-STEP --
 *   BATCHED_MOE_FIX.md). What binning cuts is the per-expert TOKEN count (T -> C) ->
 *   the GEMM MACs (E*T -> E*C) and the bf16 activation transient (gu/y scale with C
 *   not T) -> ~T/C x fewer GEMM (oracle: ~4-10x @c128 if routing balanced) AND it
 *   un-OOMs the prefill warmup the dense monolithic hits (moe_wna16_hpu.py:285-300).
 *   Defeats the "K->E=32 by bs16" clamp on the T-axis (per-expert load), not the
 *   E-axis union (=32, conceded).
 *
 * THIS OP PROCESSES ONE EXPERT GROUP [G,C,H] -> [G,C,H] (router-weighted, NO reduce):
 * the wrapper calls it once per CHUNK group (slicing E on-device via chunk_start to
 * dodge the lazy storage-offset bug -- BATCHED_MOE_CO_READY.md) and torch.cat's the
 * G-sized outputs into [E,C,H]. (A single-recipe internal-concat variant is a perf
 * refinement that saves ~E/CHUNK-1 launches/layer; deferred -- the concat node is not
 * card-free-validatable and this box bans unvalidated low-level nodes.)
 *
 * Reuses the PROVEN int4->fp8 helpers from moe_fused_monolithic_int4fp8_emit.h
 * (emit_chunk_slice_i4f8 / emit_fp8_quant / emit_fp8_batch_gemm_i4f8) VERBATIM. The
 * int4fp8 + monolithic + routed ops are UNTOUCHED -- this is a NEW additive schema.
 * Standard SiLU (GLM glm_moe). Lazy serving path. RUN ON THE HPU BOX for the device
 * byte-exact gate; this file is only COMPILE-validated card-free.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "moe_fused_monolithic_int4fp8_emit.h"   // emit_chunk_slice_i4f8, emit_fp8_quant,
                                                 // emit_fp8_batch_gemm_i4f8, kI4F8MaxFnuz

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// emit_binned_group_i4f8 -- the W4A8 chain for ONE GROUP of G experts over their
// capacity-C permuted activation x_perm[E,C,H] (sliced [G,C,H] at chunk_start).
//   x_perm  : bf16  [E, C, H]   (the wrapper's scatter output; sliced on-device)
//   w13/w2  : the FULL stacked [E,...] int4 weights/scales/zp (sliced on-device)
//   rw      : bf16  [E, C, 1]   per-(expert,slot) router weight (sliced on-device)
//   returns : bf16  [G, C, H]   = router-weighted expert output, NO reduce over G.
// This is emit_monolithic_chunk_i4f8 with TWO deltas: (1) x is sliced from the
// per-expert x_perm (NOT broadcast over a shared [T,H]); (2) the result is yw[G,C,H]
// returned directly (NOT reduce_sum'd over the expert axis). Everything else --
// int4->fp8 direct dequant (sB=1 folded), per-token fp8 activation quant, the two fp8
// batch-gemms, standard SiLU -- is byte-identical to the proven int4fp8 chunk.
// =============================================================================
inline std::vector<sh::tensor> emit_binned_group_i4f8(
    OpBackend* op, sh::graph& graph,
    synTensor x_perm, const std::vector<int64_t>& x_perm_shape,   // [E,C,H]
    int64_t E, int64_t chunk_start, int64_t G, int64_t C, int64_t H,
    int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s, const std::vector<int64_t>& w13s_shape,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,  const std::vector<int64_t>& w2s_shape,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw,   const std::vector<int64_t>& rw_shape) {        // [E,C,1]
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType fp8e = c10::ScalarType::Float8_e4m3fn;
  const at::ScalarType f32e = c10::ScalarType::Float;

  // ---- slice the FULL [E,...] inputs to this group's [G,...] on dim 0 (expert axis) ----
  auto w13p_c = emit_chunk_slice_i4f8(op, graph, w13p, w13p_shape, chunk_start, G, at::kInt);
  auto w13s_c = emit_chunk_slice_i4f8(op, graph, w13s, w13s_shape, chunk_start, G, bf16);
  auto w13z_c = emit_chunk_slice_i4f8(op, graph, w13z, w13z_shape, chunk_start, G, at::kInt);
  auto w2p_c  = emit_chunk_slice_i4f8(op, graph, w2p,  w2p_shape,  chunk_start, G, at::kInt);
  auto w2s_c  = emit_chunk_slice_i4f8(op, graph, w2s,  w2s_shape,  chunk_start, G, bf16);
  auto w2z_c  = emit_chunk_slice_i4f8(op, graph, w2z,  w2z_shape,  chunk_start, G, at::kInt);
  auto rw_c   = emit_chunk_slice_i4f8(op, graph, rw,   rw_shape,   chunk_start, G, bf16);
  // DELTA #1: x_perm is per-expert [E,C,H] -> slice [G,C,H] (NO broadcast over tokens).
  auto x_c    = emit_chunk_slice_i4f8(op, graph, x_perm, x_perm_shape, chunk_start, G, bf16);

  std::vector<int64_t> w13p_cs{G, H, OUT13 / 8};
  std::vector<int64_t> w13z_cs{G, w13z_shape[1], OUT13 / 8};
  std::vector<int64_t> w2p_cs{G, I, H / 8};
  std::vector<int64_t> w2z_cs{G, w2z_shape[1], H / 8};

  // int4 packed int32 -> uint4 reinterpret (FCD x8). VERBATIM the int4fp8 chunk.
  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast",
         {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };
  // int4 -> fp8 DIRECTLY (DPQ: w_fp8 = (w4-z)*group_scale, e4m3; group scale folded -> sB=1).
  // VERBATIM the int4fp8 chunk's dequant_fp8 lambda.
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
  auto build_ones = [&](const std::vector<int64_t>& shp) {
    auto one = OpBackend::BuildConstantTensor(op, graph, at::Scalar(1.0), f32e);
    std::vector<int64_t> s111{1, 1, 1};
    auto r = OpBackend::BuildReshape(op, graph, one.get(), s111, f32e);
    return OpBackend::BuildBroadcast(op, graph, r.get(), shp, f32e);
  };

  // ============ W13: int4 -> fp8 (sB=1) ============
  auto W13_fp8 = dequant_fp8(w13p_c[0].get(), w13p_cs, w13s_c[0].get(), w13z_c[0].get(), w13z_cs);
  std::vector<int64_t> s13_shape{G, 1, OUT13};
  auto s13_one = build_ones(s13_shape);

  // ============ x_perm group [G,C,H] -> per-(expert,slot)-token fp8 (reduce over H=dim2) ====
  // DELTA #1 (cont): x_c is ALREADY [G,C,H] (one row per routed token-slot), so the
  // per-token activation scale reduces the LAST axis (H) -> [G,C,1]. No broadcast node.
  std::vector<int64_t> xg_shape{G, C, H};
  auto xq = emit_fp8_quant(op, graph, x_c[0].get(), xg_shape, /*reduce_dim=*/2);  // x_fp8[G,C,H], x_s[G,C,1]

  // ============ gemm #1: gu[G,C,2I] = fp8_bmm(x_fp8[G,C,H], W13_fp8[G,H,2I]) ============
  std::vector<int64_t> gu_shape{G, C, OUT13};
  auto gu = emit_fp8_batch_gemm_i4f8(op, graph,
      xq[0].get(), xq[1].get(), W13_fp8[0].get(), s13_one.get(), gu_shape, /*is_result=*/false);

  // ---- slice gate/up along feature dim 2 (2I) ----
  const int64_t gu_rank = 3;
  const int64_t gu_dims[3] = {G, C, OUT13};
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
    std::vector<int64_t> out_shape{G, C, end - begin};
    return OpBackend::BuildNode(
        op, graph, {"slice", {in}, {{out_shape, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- standard SiLU: silu=gate*sigmoid(gate); h=silu*up ----
  std::vector<int64_t> im_shape{G, C, I};
  auto sgte = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()}, {{im_shape, bf16, std::nullopt}}});

  // ============ h[G,C,I] -> per-token fp8 (reduce last=I) ============
  auto hq = emit_fp8_quant(op, graph, h[0].get(), im_shape, /*reduce_dim=*/2);

  // ============ W2: int4 -> fp8 (sB=1) ============
  auto W2_fp8 = dequant_fp8(w2p_c[0].get(), w2p_cs, w2s_c[0].get(), w2z_c[0].get(), w2z_cs);
  std::vector<int64_t> s2_shape{G, 1, H};
  auto s2_one = build_ones(s2_shape);

  // ============ gemm #2: y[G,C,H] = fp8_bmm(h_fp8[G,C,I], W2_fp8[G,I,H]) ============
  std::vector<int64_t> y_shape{G, C, H};
  auto y = emit_fp8_batch_gemm_i4f8(op, graph,
      hq[0].get(), hq[1].get(), W2_fp8[0].get(), s2_one.get(), y_shape, /*is_result=*/false);

  // ---- DELTA #2: yw = y * router_weight rw_c[G,C,1] (broadcast over H) -- THIS IS THE
  //      OUTPUT (NO reduce over the expert axis; each (expert,slot) is a distinct token). ----
  return OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw_c[0].get()}, {{y_shape, bf16, std::optional<int>(0)}}});
}

} // namespace habana
