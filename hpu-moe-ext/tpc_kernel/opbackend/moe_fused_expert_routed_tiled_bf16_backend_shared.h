/*
 * moe_fused_expert_routed_tiled_bf16_backend_shared.h -- the SHARED, frontend-agnostic
 * backend lowering for the BYTE-EXACT CONTRACTION-TILED bf16 K-EXPERT single-recipe
 * ROUTED MoE op `torch.ops.hpu_moe_ext.fused_moe_routed_tiled_bf16` (GATE-A).
 *
 * This is moe_fused_expert_routed_backend_shared.h (the bf16 routed op `fused_moe`)
 * with the ONLY change being the per-expert emitter: it calls
 * emit_one_expert_tiled_bf16 (contraction-tiled int4->bf16 dequant + per-tile bf16 MME
 * + fp32-accumulate across tiles + ONE final bf16 round, the GATE-A transform in
 * moe_fused_expert_tiled_bf16_emit.h) instead of emit_one_expert (int4->bf16 of the
 * FULL [H,2I] weight in one shot). SAME schema, SAME K-TensorList input layout, SAME
 * left-fold add-tree over the K experts, SAME [1,1]/[T,1] router-weight handling.
 *
 * THE DISCRIMINATOR (why this op exists): the stock routed op dequants each expert's
 * FULL weight as ONE non-persistent tensor (~25 MB fp8 / 50 MB bf16) that CANNOT fit
 * Gaudi-2's ~48 MB SRAM (x K co-resident in the one K-fold recipe) -> GC spills it to
 * HBM (the weight is written + re-read, ~4.5 B/elem vs the 0.5 B/elem int4 floor). The
 * tiled emit slices the contraction dim into Kt-row tiles whose dequanted bf16 tile
 * fits SRAM, so the GC bundles each tile's dequant into its MME and keeps it
 * SRAM-resident -> the only HBM weight traffic is the int4 read once. If eliminating
 * that materialization round-trip moves the bs=1 step, the fold is worth building; if
 * flat (with the workspace proof) the device-serial wall is real. BYTE-EXACT: the
 * per-element dequant is the IDENTICAL stock dequantize_4_bit_bf16, the GEMM stays
 * fp32 across tiles with a single final bf16 round -> relNorm ~1e-6, identical greedy
 * tokens (the lossless bar). See moe_fused_expert_tiled_bf16_emit.h's header.
 *
 * PERF-LIB CONTRACT: this op introduces NO new custom TPC guid -- it uses ONLY stock
 * guids (gemm / dequantize_4_bit / add_fwd / slice / reinterpret_cast / sigmoid_fwd /
 * mult_fwd / cast), each of which already carries its own shape inference in the
 * shipped perf-lib. The OpBackend-level output shape is provided by
 * MoeFusedMoETiledBf16Meta via SetOutputMetaFn (== the stock routed op). So the
 * "GetShapeInference mandatory" contract is honoured by construction (no custom guid
 * to register one for).
 *
 * Gated OFF by default (Python fused_expert_routed routes here only under
 * HPU_MOE_ROUTED_TILE_BF16=<Kt>, a positive int). The bf16 routed op + the int4fp8 /
 * monolithic / single-expert ops are UNTOUCHED -- this is a NEW additive schema with a
 * DISTINCT op name + a DISTINCT backend struct (MoeFusedMoETiledBf16) + a DISTINCT
 * emitter symbol (emit_one_expert_tiled_bf16), so it co-loads with the other .so's
 * with NO ODR clash.
 *
 * Distinct backend symbols (MoeFusedMoETiledBf16 / MoeFusedMoETiledBf16Meta) avoid an
 * ODR clash with the bf16 / int4fp8 routed shared headers if more than one is ever
 * pulled into one TU. The full GEOMETRY / capture / TensorList rationale lives in the
 * bf16 routed shared header + the tiled emit's header -- read those first; it is not
 * repeated here.
 *
 * THE ONE DELTA vs the int4fp8 backend's K-fold: emit_one_expert_tiled_bf16 ALSO needs
 * the two SCALE SHAPES (w13_scale[k].sizes() / w2_scale[k].sizes()) to derive the int4
 * group size and align each tile's scale-group slice. So emit_k passes those two extra
 * shape vectors (the int4fp8 / stock emitters take only the scale syn_t). Everything
 * else -- the K homogeneity checks, the dim derivation, the left-fold add-tree -- is
 * byte-for-byte the proven routed backend.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr / NodeOutputAttr
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair, std::vector<TensorsPair>
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

#include "moe_fused_expert_tiled_bf16_emit.h"   // emit_one_expert_tiled_bf16 (the tiled chain)

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: out is [M, H] bf16 (== x). M is read from x (stack pos 0), correct
// for ANY token count (T==1 decode or T=depth+1 MTP verify). Distinct symbol from
// the bf16 / int4fp8 routed Meta (avoids ODR clash if more than one co-loads).
// =============================================================================
inline OutputMetaDataVector MoeFusedMoETiledBf16Meta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [M, H]
  meta[0].dtype = x.scalar_type();      // bf16 (the fp32 accumulate is internal; out stays bf16)
  return meta;
}

// =============================================================================
// The contraction-tiled bf16 K-expert OpBackend: one AddNode emits all K experts'
// tiled fused subgraphs (via emit_one_expert_tiled_bf16) + the weighted-sum add-tree
// -> ONE captured recipe. Mirrors MoeFusedMoE / MoeFusedMoEInt4Fp8 byte-for-byte
// except the emitter call (which additionally takes the two scale shapes).
// =============================================================================
struct MoeFusedMoETiledBf16 : OpBackend {
  MoeFusedMoETiledBf16(int device_id, c10::ScalarType scalar_type)
      // guid "gemm" is a harmless placeholder (real guids are per-BuildNode);
      // res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMoETiledBf16Meta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMoETiledBf16::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMoETiledBf16::AddNode");

  // ---- typed inputs (order == schema): x, then 6 weight-class lists + the
  //      router-weight list. Each list is K per-expert tensors. IDENTICAL to the
  //      bf16 / int4fp8 routed op (the Python feed in fused_expert_routed is
  //      unchanged -- the tiled op has the SAME 8-arg schema). ----
  auto x          = sg.getNextInput<TensorsPair>();              // bf16 [M, H]
  auto w13_packed = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [H, 2I/8]
  auto w13_scale  = sg.getNextInput<std::vector<TensorsPair>>(); // K x bf16  [H/GS, 2I]
  auto w13_zp     = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [I, H/8]
  auto w2_scale   = sg.getNextInput<std::vector<TensorsPair>>(); // K x bf16  [I/GS, H]
  auto w2_zp      = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [I/GS, H/8]
  auto router_w   = sg.getNextInput<std::vector<TensorsPair>>(); // K x bf16  [1, 1] (T==1) / [T, 1]

  // ---- K + homogeneity checks (LOUD on any mismatch; the routed caller always
  //      passes equal-length lists -- a mismatch is a wiring bug, fail fast) ----
  const int64_t K = static_cast<int64_t>(w13_packed.size());
  TORCH_CHECK(K >= 1, "[fused_moe_routed_tiled_bf16] need >=1 expert, got K=", K);
  TORCH_CHECK(
      static_cast<int64_t>(w13_scale.size()) == K &&
      static_cast<int64_t>(w13_zp.size()) == K &&
      static_cast<int64_t>(w2_packed.size()) == K &&
      static_cast<int64_t>(w2_scale.size()) == K &&
      static_cast<int64_t>(w2_zp.size()) == K &&
      static_cast<int64_t>(router_w.size()) == K,
      "[fused_moe_routed_tiled_bf16] all 7 expert lists must have the same length K=", K,
      " (w13_scale=", w13_scale.size(), " w13_zp=", w13_zp.size(),
      " w2_packed=", w2_packed.size(), " w2_scale=", w2_scale.size(),
      " w2_zp=", w2_zp.size(), " router_weights=", router_w.size(), ")");

  // ---- derive the loop-invariant dims ONCE (per-expert shapes are homogeneous):
  //      M,H from x; OUT13(=2I),I from w13_scale[0] [H/GS, 2I]. ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe_routed_tiled_bf16] x must be 2D [M,H]");
  const int64_t M = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto s13_sizes = w13_scale[0].pt_t.sizes().vec();   // [H/GS, 2I]
  TORCH_CHECK(w13_scale[0].pt_t.dim() == 2,
              "[fused_moe_routed_tiled_bf16] w13_scale[k] must be 2D [H/GS, 2I]");
  const int64_t OUT13 = s13_sizes[1];                       // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_routed_tiled_bf16] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  // ===========================================================================
  // K-FOLD: emit each expert's TILED bf16 subgraph (emit_one_expert_tiled_bf16) ->
  // y_k[M,H] (router weight folded), then a left-fold add tree -> the single [M,H]
  // result.
  //   K==1: emit with is_final_result=true (its final mult IS the op result).
  //   K>=2: every emit is non-final (y_k graph-internal); chain add_fwd_bf16, last
  //         add ties to syn_out(0).
  //
  // THE ONE DELTA vs the stock/int4fp8 emit_k: emit_one_expert_tiled_bf16 needs the
  // two SCALE SHAPES (w13_scale[k].sizes() / w2_scale[k].sizes()) to derive the int4
  // group size and align each tile's scale-group slice, so they are passed here.
  // ===========================================================================
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  std::vector<int64_t> mh_shape{M, H};   // NAMED (non-owning IntArrayRef) -- must
                                         // outlive every add BuildNode call.

  auto emit_k = [&](int64_t k, bool is_final) {
    return emit_one_expert_tiled_bf16(
        this, graph,
        x.syn_t, M, H, I, OUT13,
        w13_packed[k].syn_t, w13_packed[k].pt_t.sizes().vec(),
        w13_scale[k].syn_t,  w13_scale[k].pt_t.sizes().vec(),   // + scale SHAPE (tiling)
        w13_zp[k].syn_t,     w13_zp[k].pt_t.sizes().vec(),
        w2_packed[k].syn_t,  w2_packed[k].pt_t.sizes().vec(),
        w2_scale[k].syn_t,   w2_scale[k].pt_t.sizes().vec(),    // + scale SHAPE (tiling)
        w2_zp[k].syn_t,      w2_zp[k].pt_t.sizes().vec(),
        router_w[k].syn_t,
        is_final);
  };

  if (K == 1) {
    auto y0 = emit_k(0, /*is_final=*/true);
    syn_out(0) = std::move(y0[0]);
    return;
  }

  // K >= 2: accumulate y_0 + y_1 + ... + y_{K-1} via a left-fold add chain.
  auto acc = emit_k(0, /*is_final=*/false);          // y_0 (non-persistent)
  for (int64_t k = 1; k < K; ++k) {
    auto yk = emit_k(k, /*is_final=*/false);          // y_k (non-persistent)
    const bool last = (k == K - 1);
    const std::optional<int> final_idx =
        last ? std::optional<int>(0) : std::nullopt;  // last add == op result
    acc = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("add_fwd"sv, bf16),
         {acc[0].get(), yk[0].get()},
         {{mh_shape, bf16, final_idx}}});
  }
  syn_out(0) = std::move(acc[0]);
}

} // namespace habana
