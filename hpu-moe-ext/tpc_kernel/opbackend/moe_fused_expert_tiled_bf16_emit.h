/*
 * moe_fused_expert_tiled_bf16_emit.h -- BYTE-EXACT CONTRACTION-TILED bf16 fold
 * (GATE-A discriminator).
 *
 * GOAL: prove whether ELIMINATING the int4->bf16 weight-materialization round-trip
 * moves the bs=1 single-stream step. The shipped routed ops dequant each expert's
 * FULL weight ([H,2I] for w13 ~= 25 MB bf16, [I,H] for w2) as ONE "NON-PERSISTENT/
 * SRAM" tensor that does NOT fit Gaudi-2's ~48 MB SRAM (x K experts co-resident in
 * the one K-fold recipe), so the GC SPILLS it to HBM -> the dequanted weight is
 * written to HBM and re-read by the MME (~4.5 B/elem of weight traffic vs the
 * 0.5 B/elem int4-read floor). That spill is the suspected cause of the recorded
 * "Routed W4A8 int4->fp8 marginal SS (mem-bound)" verdict.
 *
 * THE TRANSFORM (this header): slice the CONTRACTION dim (H for w13, I for w2) into
 * Kt-row tiles small enough that the dequanted bf16 tile [Kt, N] fits SRAM, dequant
 * each tile int4->bf16 in the SAME graph, run a per-tile bf16-MME gemm with an FP32
 * output, FP32-ACCUMULATE the partials across tiles, and cast to bf16 ONCE at the
 * end. GC then bundles each small tile's dequant into its MME and keeps it
 * SRAM-resident -> the only HBM weight traffic is the int4 read once.
 *
 * BYTE-EXACTNESS (the lossless bar): the per-element dequant is the IDENTICAL stock
 * dequantize_4_bit_bf16(w_u4, scale, zp_u4) (NOT the lean kernel), and the GEMM
 * accumulation stays in FP32 across tiles with a SINGLE final bf16 round -- exactly
 * like the stock single-gemm MME (fp32 accumulator, bf16 output). The only
 * difference vs stock `emit_one_expert` is the fp32 partial-sum GROUPING (within-tile
 * then across-tile vs one full-K reduction), a non-associative fp32 reordering that
 * perturbs the result by ~1 fp32-ULP -> relNorm ~1e-6, IDENTICAL greedy tokens. So
 * this is byte-exact-grade for the greedy decode (NOT bit-0 logits; the gate's bar is
 * the identical greedy token stream, robust to fp32-accum-order ULPs).
 *
 * TILE SIZE: env HPU_MOE_ROUTED_TILE_BF16=Kt (rows of the contraction dim per tile).
 *   Kt<=0 or Kt>=dim  -> ONE tile (== the stock single gemm, fp32-out + cast; an A==A
 *                        sanity path). Kt is rounded DOWN to a multiple of the int4
 *                        group_size GS (so each tile's scale groups align); min GS.
 *   Recommended Kt for GLM (H=6144=48*GS128, GS=128): 1024 or 2048 (bf16 tile
 *   [2048,4096]*2B = 16 MB; [1024,4096]*2B = 8 MB). Sweep down if the GC still spills.
 *
 * ACCUM PRECISION: the per-tile gemm outputs FP32 (bf16 in -> fp32 out, the MME's
 *   natural no-downcast mode) so the cross-tile add is lossless. If on-card graph
 *   codegen ever REJECTS an fp32-output "gemm" (the one property NOT verifiable
 *   card-free -- it surfaces at the first on-card compile, LOUD), set
 *   HPU_MOE_TILE_ACC_BF16=1 to output bf16 partials + fp32-add (a few bf16 ULPs of
 *   loss across the 3-6 tiles; relNorm ~1e-3 -- still likely greedy-identical, but
 *   NOT the lossless bar). The CPU harness reports both.
 *
 * Frontend-free (pulls ONLY op_backend.h / hpu_op_helper.h + synapse params, exactly
 * like moe_fused_expert_emit.h). PROVEN code is UNTOUCHED: this is a NEW free function
 * (emit_one_expert_tiled_bf16) with a UNIQUE symbol -- the K-expert tiled TU includes
 * it; the stock emit_one_expert / the single-expert op / the int4fp8 op are unchanged.
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
#include "perf_lib_layer_params.h"    // ns_CastKernel::ParamsV3

#include <algorithm>   // std::fill_n, std::min
#include <cstdint>
#include <cstdlib>     // std::getenv
#include <string>      // std::stoll
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Parse HPU_MOE_ROUTED_TILE_BF16 once. <=0 -> 0 (== one-tile / stock). Function-local
// static across TUs is fine (same value); mirrors emit_one_expert's MOE_DEQUANT_KERNEL.
inline int64_t _tiled_bf16_kt_env() {
  static const int64_t kKt = [] {
    const char* e = std::getenv("HPU_MOE_ROUTED_TILE_BF16");
    if (e == nullptr || *e == '\0') return (int64_t)0;
    char* end = nullptr;
    long long v = std::strtoll(e, &end, 10);
    return (v > 0) ? (int64_t)v : (int64_t)0;
  }();
  return kKt;
}

// Output FP32 partials (lossless cross-tile add) unless HPU_MOE_TILE_ACC_BF16=1.
inline bool _tiled_bf16_acc_bf16_env() {
  static const bool kBf16 = [] {
    const char* e = std::getenv("HPU_MOE_TILE_ACC_BF16");
    return e != nullptr && e[0] == '1';
  }();
  return kBf16;
}

// =============================================================================
// emit_one_expert_tiled_bf16 -- the byte-exact contraction-tiled bf16 per-expert
// chain for ONE routed expert; returns its [M,H] bf16 output (router weight folded
// by the final mult). Signature MIRRORS emit_one_expert EXACTLY + the two scale
// shapes (w13s_shape/w2s_shape) the tiling needs to align scale groups. The K-expert
// tiled backend calls this K times and add-folds the y_k's, identical to the stock
// routed backend.
//
//   x        bf16  [M, H]            (shared across all K experts)
//   M,H,I,OUT13                      (loop-invariant dims; OUT13 == 2I)
//   w13p     int32 [H, 2I/8]   + w13p_shape
//   w13s     bf16  [H/GS, 2I]   + w13s_shape   (NEW vs stock: needed to slice groups)
//   w13z     int32 [H/GS, 2I/8] + w13z_shape
//   w2p      int32 [I, H/8]     + w2p_shape
//   w2s      bf16  [I/GS, H]    + w2s_shape    (NEW)
//   w2z      int32 [I/GS, H/8]  + w2z_shape
//   rw       bf16  [1, 1] (T==1) or [T, 1] (T>1)
// =============================================================================
inline std::vector<sh::tensor> emit_one_expert_tiled_bf16(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t M, int64_t H, int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s, const std::vector<int64_t>& w13s_shape,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,  const std::vector<int64_t>& w2s_shape,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw,
    bool is_final_result = false) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::ScalarType f32 = c10::ScalarType::Float;
  const at::ScalarType acc_dt = _tiled_bf16_acc_bf16_env() ? bf16 : f32;

  // ---- int4 group_size per projection (must divide cleanly; LOUD otherwise) ----
  TORCH_CHECK(w13s_shape.size() == 2 && w2s_shape.size() == 2,
              "[tiled_bf16] scale shapes must be 2D");
  const int64_t GS13 = H / w13s_shape[0];   // [H/GS, 2I] -> GS = H / rows
  const int64_t GS2  = I / w2s_shape[0];     // [I/GS, H]  -> GS = I / rows
  TORCH_CHECK(GS13 > 0 && H % GS13 == 0,
              "[tiled_bf16] H=", H, " not divisible by GS13=", GS13);
  TORCH_CHECK(GS2 > 0 && I % GS2 == 0,
              "[tiled_bf16] I=", I, " not divisible by GS2=", GS2);

  // ---- effective tile size per projection: round the env Kt DOWN to a multiple of
  //      GS; clamp to [GS, dim]. Kt<=0 (env unset) -> the full dim (one tile). ----
  auto eff_kt = [](int64_t kt_env, int64_t dim, int64_t gs) -> int64_t {
    if (kt_env <= 0 || kt_env >= dim) return dim;          // one tile (== stock)
    int64_t kt = (kt_env / gs) * gs;                        // multiple of GS
    if (kt < gs) kt = gs;
    if (kt > dim) kt = dim;
    return kt;
  };
  const int64_t kt_env = _tiled_bf16_kt_env();
  const int64_t KT13 = eff_kt(kt_env, H, GS13);   // tile the H contraction of gemm#1
  const int64_t KT2  = eff_kt(kt_env, I, GS2);     // tile the I contraction of gemm#2

  // ---- generic slice of dim d of a rank-r synTensor to [begin,end) (TPC-reversed
  //      axis idiom, VERBATIM from emit_one_expert's gate/up slice). ----
  auto slice_dim = [&](synTensor in, const std::vector<int64_t>& in_shape,
                       int64_t d, int64_t begin, int64_t end,
                       at::ScalarType dt) {
    const int64_t r = static_cast<int64_t>(in_shape.size());
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    for (int64_t i = 0; i < r; ++i) {
      sp.axes[i] = static_cast<unsigned>(r - i - 1);   // TPC-reversed
      if (i == d) {
        sp.starts[i] = static_cast<TSize>(begin);
        sp.ends[i] = static_cast<TSize>(end);
      } else {
        sp.starts[i] = static_cast<TSize>(0);
        sp.ends[i] = static_cast<TSize>(in_shape[i]);
      }
      sp.steps[i] = static_cast<TSize>(1);
    }
    std::vector<int64_t> out_shape = in_shape;
    out_shape[d] = end - begin;
    return OpBackend::BuildNode(
        op, graph, {"slice", {in}, {{out_shape, dt, std::nullopt}}, &sp, sizeof(sp)});
  };

  // ---- reinterpret int32-packed -> uint4 (VERBATIM emit_one_expert): last dim *8 ----
  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast", {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // ---- STOCK dequant of a [kt, N/8]-packed tile -> [kt, N] bf16 (NON-PERSISTENT,
  //      bundled into the following per-tile MME). BYTE-FOR-BYTE the stock path. ----
  auto dequant_tile = [&](synTensor w_packed_tile, const std::vector<int64_t>& wt_shape,
                          synTensor scale_tile,
                          synTensor zp_packed_tile, const std::vector<int64_t>& zpt_shape) {
    auto w_u4  = reinterpret_u4(w_packed_tile, wt_shape);
    auto zp_u4 = reinterpret_u4(zp_packed_tile, zpt_shape);
    std::vector<synTensor> in{w_u4[0].get(), scale_tile, zp_u4[0].get()};
    std::vector<int64_t> out_shape = wt_shape;
    out_shape.back() *= 8;
    ns_CastKernel::ParamsV3 params{};
    return OpBackend::BuildNode(
        op, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, bf16),
         std::move(in), {{out_shape, bf16, std::nullopt}}, &params, sizeof(params)});
  };

  // ---- TILED gemm: out[M,N] (bf16) = sum_tiles( A[:,tile] @ dequant(W[tile,:]) ),
  //      accumulated in acc_dt (fp32 by default), final cast to bf16.
  //   a, a_shape       : activation [M, K] (contraction = dim 1)
  //   wp,wp_shape      : packed weight [K, N/8]  (contraction = dim 0)
  //   ws               : scale [K/GS, N]   (contraction-group = dim 0)
  //   wz,wz_shape      : zp [K/GS, N/8]    (contraction-group = dim 0)
  //   Kdim,N,GS,KT     : contraction size, output cols, group size, tile rows. ----
  auto tiled_gemm = [&](synTensor a, const std::vector<int64_t>& a_shape,
                        synTensor wp, const std::vector<int64_t>& wp_shape,
                        synTensor ws, const std::vector<int64_t>& ws_shape,
                        synTensor wz, const std::vector<int64_t>& wz_shape,
                        int64_t Kdim, int64_t N, int64_t GS, int64_t KT)
      -> std::vector<sh::tensor> {
    synGEMMParams gp{false, false};
    std::vector<int64_t> out_shape{M, N};
    std::vector<sh::tensor> acc;   // [M,N] in acc_dt
    for (int64_t k0 = 0; k0 < Kdim; k0 += KT) {
      const int64_t kt = std::min<int64_t>(KT, Kdim - k0);
      const int64_t g0 = k0 / GS;             // scale/zp group-row start
      const int64_t g1 = (k0 + kt) / GS;       // group-row end (exclusive)
      // slice the activation along its contraction (dim 1): [M, kt]
      auto a_t = slice_dim(a, a_shape, /*d=*/1, k0, k0 + kt, bf16);
      // slice the packed weight along its contraction (dim 0): [kt, N/8]
      auto wp_t = slice_dim(wp, wp_shape, /*d=*/0, k0, k0 + kt, at::kInt);
      std::vector<int64_t> wp_t_shape{kt, wp_shape[1]};
      // slice the scale (dim 0, by group): [kt/GS, N]
      auto ws_t = slice_dim(ws, ws_shape, /*d=*/0, g0, g1, bf16);
      // slice the zp (dim 0, by group): [kt/GS, N/8]
      auto wz_t = slice_dim(wz, wz_shape, /*d=*/0, g0, g1, at::kInt);
      std::vector<int64_t> wz_t_shape{g1 - g0, wz_shape[1]};
      // dequant the tile -> [kt, N] bf16 (SRAM-resident, bundled into the MME)
      auto w_t = dequant_tile(wp_t[0].get(), wp_t_shape, ws_t[0].get(),
                              wz_t[0].get(), wz_t_shape);
      // per-tile gemm: [M,kt] @ [kt,N] -> [M,N] in acc_dt (fp32 = no downcast)
      auto part = OpBackend::BuildNode(
          op, graph,
          {"gemm", {a_t[0].get(), w_t[0].get()},
           {{out_shape, acc_dt, std::nullopt}}, &gp, sizeof(gp)});
      if (acc.empty()) {
        acc = std::move(part);                 // tile 0
      } else {
        acc = OpBackend::BuildNode(
            op, graph,
            {get_guid_with_precision("add_fwd"sv, acc_dt),
             {acc[0].get(), part[0].get()}, {{out_shape, acc_dt, std::nullopt}}});
      }
    }
    return acc;   // [M,N] in acc_dt
  };

  // ============ gemm #1: gu[M, 2I] = x[M,H] @ W13[H,2I] (contract H, tiled) ============
  auto gu_acc = tiled_gemm(x, std::vector<int64_t>{M, H},
                           w13p, w13p_shape, w13s, w13s_shape, w13z, w13z_shape,
                           /*Kdim=*/H, /*N=*/OUT13, GS13, KT13);
  std::vector<int64_t> gu_shape{M, OUT13};
  // final single bf16 round (no-op if acc already bf16)
  sh::tensor gu = (acc_dt == bf16)
      ? std::move(gu_acc[0])
      : OpBackend::BuildCast(op, graph, gu_acc[0].get(), gu_shape, f32, bf16);

  // ---- gate/up split + SiLU(gate)*up (BYTE-FOR-BYTE emit_one_expert) ----
  const int64_t gu_rank = 2;
  auto split = [&](int64_t begin, int64_t end) {
    return slice_dim(gu.get(), gu_shape, /*d=*/1, begin, end, bf16);
  };
  (void)gu_rank;
  auto gate = split(0, I);
  auto up   = split(I, OUT13);
  std::vector<int64_t> im_shape{M, I};
  auto sgte = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph, {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()}, {{im_shape, bf16, std::nullopt}}});

  // ============ gemm #2: y[M, H] = h[M,I] @ W2[I,H] (contract I, tiled) ============
  auto y_acc = tiled_gemm(h[0].get(), im_shape,
                          w2p, w2p_shape, w2s, w2s_shape, w2z, w2z_shape,
                          /*Kdim=*/I, /*N=*/H, GS2, KT2);
  std::vector<int64_t> mh_shape{M, H};
  sh::tensor y = (acc_dt == bf16)
      ? std::move(y_acc[0])
      : OpBackend::BuildCast(op, graph, y_acc[0].get(), mh_shape, f32, bf16);

  // ---- y_k = y * router_weight (BYTE-FOR-BYTE emit_one_expert's final node) ----
  const std::optional<int> final_idx =
      is_final_result ? std::optional<int>(0) : std::nullopt;
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y.get(), rw}, {{mh_shape, bf16, final_idx}}});
}

} // namespace habana
