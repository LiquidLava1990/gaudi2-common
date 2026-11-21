/*
 * moe_fused_expert_int4fp8_backend_shared.h -- the SHARED, frontend-agnostic
 * backend lowering for the W4A8 K-EXPERT single-recipe ROUTED MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_routed_int4fp8`.
 *
 * This is moe_fused_expert_routed_backend_shared.h (the bf16 routed op `fused_moe`)
 * with the ONLY change being the per-expert emitter: it calls emit_one_expert_i4f8
 * (int4->fp8 DIRECT dequant + native fp8 MME, the W4A8 transform proven in the
 * monolithic op) instead of emit_one_expert (int4->bf16 + bf16 MME). SAME schema,
 * SAME K-TensorList input layout, SAME left-fold add-tree over the K experts, SAME
 * [1,1]/[T,1] router-weight handling. It is the SINGLE-STREAM + MTP-verify fp8 path:
 * the routed gather feeds <=K experts at T==1 (decode) or T=depth+1 (MTP verify),
 * and now each runs fp8 (2x MME) instead of bf16.
 *
 * Validated externally by DPQ (arXiv 2505.14638, Intel, Gaudi-2): int4->fp8->fp8-MME
 * = up to 3x over W4A16. Gated OFF by default (Python fused_expert_routed routes here
 * only under HPU_MOE_ROUTED_INT4FP8=1). The bf16 routed op + the monolithic / single-
 * expert ops are UNTOUCHED -- this is a NEW additive schema with a DISTINCT op name +
 * a DISTINCT backend struct (MoeFusedMoEInt4Fp8) + a DISTINCT emitter symbol
 * (emit_one_expert_i4f8), so it co-loads with the bf16 .so with NO ODR clash.
 *
 * Distinct backend symbols (MoeFusedMoEInt4Fp8 / MoeFusedMoEInt4Fp8Meta) avoid an ODR
 * clash with the bf16 routed shared header if both are ever pulled into one TU. The
 * full GEOMETRY / capture / TensorList rationale lives in the bf16 routed shared
 * header + emit_one_expert_i4f8's header -- read those first; it is not repeated here.
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

#include "moe_fused_expert_int4fp8_emit.h"   // emit_one_expert_i4f8 (the W4A8 per-expert chain)

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
// the bf16 routed MoeFusedMoEMeta (avoids ODR clash if both headers co-load).
// =============================================================================
inline OutputMetaDataVector MoeFusedMoEInt4Fp8Meta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [M, H]
  meta[0].dtype = x.scalar_type();      // bf16 (the fp8 is internal; out stays bf16)
  return meta;
}

// =============================================================================
// The W4A8 K-expert OpBackend: one AddNode emits all K experts' int4->fp8 fused
// subgraphs (via emit_one_expert_i4f8) + the weighted-sum add-tree -> ONE captured
// recipe. Mirrors MoeFusedMoE byte-for-byte except the emitter call.
// =============================================================================
struct MoeFusedMoEInt4Fp8 : OpBackend {
  MoeFusedMoEInt4Fp8(int device_id, c10::ScalarType scalar_type)
      // guid "gemm" is a harmless placeholder (real guids are per-BuildNode);
      // res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMoEInt4Fp8Meta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMoEInt4Fp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMoEInt4Fp8::AddNode");

  // ---- typed inputs (order == schema): x, then 6 weight-class lists + the
  //      router-weight list. Each list is K per-expert tensors. IDENTICAL to the
  //      bf16 routed op (the Python feed in fused_expert_routed is unchanged). ----
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
  TORCH_CHECK(K >= 1, "[fused_moe_routed_int4fp8] need >=1 expert, got K=", K);
  TORCH_CHECK(
      static_cast<int64_t>(w13_scale.size()) == K &&
      static_cast<int64_t>(w13_zp.size()) == K &&
      static_cast<int64_t>(w2_packed.size()) == K &&
      static_cast<int64_t>(w2_scale.size()) == K &&
      static_cast<int64_t>(w2_zp.size()) == K &&
      static_cast<int64_t>(router_w.size()) == K,
      "[fused_moe_routed_int4fp8] all 7 expert lists must have the same length K=", K,
      " (w13_scale=", w13_scale.size(), " w13_zp=", w13_zp.size(),
      " w2_packed=", w2_packed.size(), " w2_scale=", w2_scale.size(),
      " w2_zp=", w2_zp.size(), " router_weights=", router_w.size(), ")");

  // ---- derive the loop-invariant dims ONCE (per-expert shapes are homogeneous):
  //      M,H from x; OUT13(=2I),I from w13_scale[0] [H/GS, 2I]. ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe_routed_int4fp8] x must be 2D [M,H]");
  const int64_t M = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto s13_sizes = w13_scale[0].pt_t.sizes().vec();   // [H/GS, 2I]
  TORCH_CHECK(w13_scale[0].pt_t.dim() == 2,
              "[fused_moe_routed_int4fp8] w13_scale[k] must be 2D [H/GS, 2I]");
  const int64_t OUT13 = s13_sizes[1];                       // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_routed_int4fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  // ===========================================================================
  // K-FOLD: emit each expert's W4A8 subgraph (emit_one_expert_i4f8) -> y_k[M,H]
  // (router weight folded), then a left-fold add tree -> the single [M,H] result.
  //   K==1: emit with is_final_result=true (its final mult IS the op result).
  //   K>=2: every emit is non-final (y_k graph-internal); chain add_fwd_bf16, last
  //         add ties to syn_out(0).
  // ===========================================================================
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  std::vector<int64_t> mh_shape{M, H};   // NAMED (non-owning IntArrayRef) -- must
                                         // outlive every add BuildNode call.

  auto emit_k = [&](int64_t k, bool is_final) {
    return emit_one_expert_i4f8(
        this, graph,
        x.syn_t, M, H, I, OUT13,
        w13_packed[k].syn_t, w13_packed[k].pt_t.sizes().vec(),
        w13_scale[k].syn_t,
        w13_zp[k].syn_t,     w13_zp[k].pt_t.sizes().vec(),
        w2_packed[k].syn_t,  w2_packed[k].pt_t.sizes().vec(),
        w2_scale[k].syn_t,
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
