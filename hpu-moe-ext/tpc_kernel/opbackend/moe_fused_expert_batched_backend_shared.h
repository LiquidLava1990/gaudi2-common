/*
 * moe_fused_expert_batched_backend_shared.h -- the SHARED, frontend-agnostic
 * backend lowering for the BATCHED-K single-recipe fused int4 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_batched`. It is the SIBLING of the SEQUENTIAL
 * single-recipe op (moe_fused_expert_routed_backend_shared.h ::
 * MoeFusedMoE::AddNode), built for SINGLE-STREAM speed: instead of looping
 * emit_one_expert K times (-> ~K*14 node launches + a (K-1)-add tree), it calls
 * emit_batched_experts ONCE to emit a chain of 16 nodes INDEPENDENT of K (a grouped
 * dequant over the stacked [K,...] weights + a batch_gemm over the K-expert batch +
 * a batched SiLU + a 2nd grouped dequant + a 2nd batch_gemm + the router-weighted
 * reduce-sum over K). Gaudi-2 decode is host/launch-bound (memory
 * gaudi-single-stream-speed), so cutting K*14 -> 16 launches/token is the lever.
 *
 * THE SEQUENTIAL OP STAYS THE VALIDATED FALLBACK. This op is a NEW schema +
 * registration; it does not touch fused_moe / fused_expert. It is gated OFF by
 * default (the Python wrapper routes to it only under HPU_MOE_FUSED_BATCHED=1).
 *
 * ============================ STACKED-K INPUT LAYOUT ===========================
 * Unlike the SEQUENTIAL op (which takes each weight class as a Tensor[] TensorList
 * of K per-expert tensors, then emit_one_expert consumes element k), this op takes
 * each weight class as ONE STACKED [K, ...] Tensor (the wrapper's index_select
 * output, fed DIRECTLY -- no .unbind). That is the natural batched feed AND keeps the
 * held-leaf source contiguous: the wrapper's x_buf (held-leaf static-input) + the
 * _alloc_fused_zp stacked zero-points (held leaves) flow in as single persistent
 * bases, so multi-bucket HPU-graph capture stays coherent (the SAME invariant the
 * sequential op's x_buf + zp inputs preserve -- NOT regressed here).
 *
 *   Schema (added to the hpu_moe_ext library via TORCH_LIBRARY_FRAGMENT so it
 *   co-loads with the single-expert + sequential .so without a duplicate-library
 *   abort):
 *     fused_moe_batched(Tensor x,
 *               Tensor w13_packed, Tensor w13_scale, Tensor w13_zp,
 *               Tensor w2_packed,  Tensor w2_scale,  Tensor w2_zp,
 *               Tensor router_weights) -> Tensor
 *   x is [M,H] bf16 (M==T tokens; single-stream T=1 priority). Each weight tensor
 *   is STACKED over K experts (PyTorch row-major, the proven per-expert op layout
 *   with a leading K axis):
 *     w13_packed[K, H, 2I/8] i32  w13_scale[K, H/GS, 2I] bf16  w13_zp[K, H/GS, 2I/8] i32
 *     w2_packed [K, I, H/8]  i32  w2_scale [K, I/GS, H]  bf16  w2_zp [K, I/GS, H/8]  i32
 *     router_weights[K, 1, 1] bf16  (each expert's scalar router weight, the T==1
 *                                    fold; folded by emit_batched_experts' mult+sum.
 *                                    For T>1 the caller passes 1 and post-scales, as
 *                                    fused_expert_routed.py does today.)
 *   Output [M,H] bf16 = sum_k( fused_expert(x, W_k) * router_weights[k] ).
 *
 * ============================ K DERIVATION + CHECKS ============================
 * K is the leading dim-0 of every weight tensor (a compile-time-per-recipe constant
 * == the gather width); the bridge recipe-caches one recipe per (K, shapes). M,H
 * come from x; OUT13(=2I),I come from w13_scale [K, H/GS, 2I]. All shapes are
 * cross-checked LOUDLY (a mismatch is a wiring bug -> fail fast).
 *
 * Frontend-free (does NOT pull eager_op.h or lazy_kernels.h), so the eager TU
 * (moe_fused_expert_batched_backend.cpp) and the lazy TU
 * (moe_fused_expert_batched_backend_lazy.cpp) each include it + add their own
 * frontend + the TORCH_LIBRARY_FRAGMENT/_IMPL + KernelRegistry registration --
 * mirroring the sequential shared-header split.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr / NodeOutputAttr, BroadcastHelper
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

#include "moe_fused_expert_batched_emit.h"   // emit_batched_experts (the 16-node batched chain)

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: out is [M, H] bf16 (== x). M is read from x (stack pos 0), so it is
// correct for ANY token count (the routed caller's T). Header-inline so the single
// including TU owns the definition. (Same shape/dtype contract as the sequential op
// -- MoeFusedMoEMeta -- but a distinct symbol to avoid an ODR clash if both shared
// headers are ever pulled into one TU.)
// =============================================================================
inline OutputMetaDataVector MoeFusedMoeBatchedMeta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [M, H]
  meta[0].dtype = x.scalar_type();      // bf16
  return meta;
}

// =============================================================================
// The BATCHED-K OpBackend: ONE AddNode emits the 16-node batched chain (via
// emit_batched_experts) -> ONE captured recipe whose node count is independent of K.
// =============================================================================
struct MoeFusedMoeBatched : OpBackend {
  MoeFusedMoeBatched(int device_id, c10::ScalarType scalar_type)
      // guid "batch_gemm" is a harmless placeholder (real guids are per-BuildNode);
      // res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMoeBatchedMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMoeBatched::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMoeBatched::AddNode");

  // ---- typed inputs (order == schema): x + 6 stacked [K,...] weight tensors + the
  //      stacked [K,1,1] router-weight tensor. Each is ONE TensorsPair (a single
  //      stacked tensor, NOT a list) -- so syn_in is one index per weight class. ----
  auto x          = sg.getNextInput<TensorsPair>();  // bf16  [M, H]
  auto w13_packed = sg.getNextInput<TensorsPair>();  // int32 [K, H, 2I/8]
  auto w13_scale  = sg.getNextInput<TensorsPair>();  // bf16  [K, H/GS, 2I]
  auto w13_zp     = sg.getNextInput<TensorsPair>();  // int32 [K, H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<TensorsPair>();  // int32 [K, I, H/8]
  auto w2_scale   = sg.getNextInput<TensorsPair>();  // bf16  [K, I/GS, H]
  auto w2_zp      = sg.getNextInput<TensorsPair>();  // int32 [K, I/GS, H/8]
  auto router_w   = sg.getNextInput<TensorsPair>();  // bf16  [K, 1, 1]

  // ---- derive dims + LOUD cross-checks (PyTorch row-major; last dim == FCD) ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe_batched] x must be 2D [M,H], got dim ",
              x.pt_t.dim());
  const int64_t M = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto w13p_sizes = w13_packed.pt_t.sizes().vec();   // [K, H, 2I/8]
  TORCH_CHECK(w13_packed.pt_t.dim() == 3,
              "[fused_moe_batched] w13_packed must be 3D [K,H,2I/8], got dim ",
              w13_packed.pt_t.dim());
  const int64_t K = w13p_sizes[0];
  TORCH_CHECK(K >= 1, "[fused_moe_batched] need >=1 expert, got K=", K);
  TORCH_CHECK(w13p_sizes[1] == H,
              "[fused_moe_batched] w13_packed dim1 (", w13p_sizes[1],
              ") must == x H (", H, ")");

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();     // [K, H/GS, 2I]
  TORCH_CHECK(w13_scale.pt_t.dim() == 3,
              "[fused_moe_batched] w13_scale must be 3D [K,H/GS,2I], got dim ",
              w13_scale.pt_t.dim());
  TORCH_CHECK(s13_sizes[0] == K,
              "[fused_moe_batched] w13_scale dim0 (", s13_sizes[0],
              ") must == K (", K, ")");
  const int64_t OUT13 = s13_sizes[2];                      // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_batched] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  // The remaining stacked tensors must all share the leading K (a mismatch is a
  // caller wiring bug; fail fast rather than emit a mis-batched recipe).
  TORCH_CHECK(
      w13_zp.pt_t.dim() == 3 && w13_zp.pt_t.size(0) == K &&
      w2_packed.pt_t.dim() == 3 && w2_packed.pt_t.size(0) == K &&
      w2_scale.pt_t.dim() == 3 && w2_scale.pt_t.size(0) == K &&
      w2_zp.pt_t.dim() == 3 && w2_zp.pt_t.size(0) == K &&
      router_w.pt_t.dim() == 3 && router_w.pt_t.size(0) == K,
      "[fused_moe_batched] every stacked weight/router tensor must be 3D with "
      "leading dim K=", K, " (w13_zp=", w13_zp.pt_t.sizes(),
      " w2_packed=", w2_packed.pt_t.sizes(), " w2_scale=", w2_scale.pt_t.sizes(),
      " w2_zp=", w2_zp.pt_t.sizes(), " router_weights=", router_w.pt_t.sizes(), ")");
  TORCH_CHECK(
      router_w.pt_t.size(1) == 1 && router_w.pt_t.size(2) == 1,
      "[fused_moe_batched] router_weights must be [K,1,1] (the T==1 scalar fold), "
      "got ", router_w.pt_t.sizes());

  // ===========================================================================
  // ONE batched chain (16 nodes, independent of K) -> the single [M,H] result.
  // emit_batched_experts ties its final reduce_sum to syn_out(0).
  // ===========================================================================
  auto out = emit_batched_experts(
      this, graph,
      x.syn_t, M, H, K, I, OUT13,
      w13_packed.syn_t, w13_packed.pt_t.sizes().vec(),
      w13_scale.syn_t,
      w13_zp.syn_t,     w13_zp.pt_t.sizes().vec(),
      w2_packed.syn_t,  w2_packed.pt_t.sizes().vec(),
      w2_scale.syn_t,
      w2_zp.syn_t,      w2_zp.pt_t.sizes().vec(),
      router_w.syn_t);

  syn_out(0) = std::move(out[0]);
}

} // namespace habana
