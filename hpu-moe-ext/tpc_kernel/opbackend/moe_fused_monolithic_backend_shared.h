/*
 * moe_fused_monolithic_backend_shared.h -- the SHARED, frontend-agnostic backend
 * lowering for the MONOLITHIC (ALL-E experts) fused int4 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_monolithic`. This is the THROUGHPUT/concurrent-
 * regime sibling of the routed/batched single-recipe ops: it processes ALL E local
 * experts (no Python gather) with per-expert SRAM-resident dequant, applying a DENSE
 * per-(token,expert) router-weight matrix, so it replaces the STOCK monolithic op
 * `VllmMixtureOfExpertsOpWNA16` (which dequantizes all E experts to bf16 -> HBM -> a
 * separate `mixture_of_experts` gemm -- the ~99us/token bf16->HBM round-trip floor
 * that dominates at concurrency). It emits the WHOLE E-expert MoE as ONE captured
 * recipe via emit_monolithic_experts, chunking E into CHUNK-expert batched
 * sub-chains so each chunk's dequant stays NON-PERSISTENT (SRAM-resident) -- the
 * fusion lever -- while iterating over all E.
 *
 * THE STOCK MONOLITHIC OP STAYS THE FALLBACK. This op is a NEW schema +
 * registration; it does not touch fused_moe / fused_expert / fused_moe_batched. It is
 * gated OFF by default (the Python apply_monolithic routes to it only under
 * HPU_MOE_FUSED_MONOLITHIC=1).
 *
 * ============================ STACKED-E INPUT LAYOUT ===========================
 * Each weight class is the FULL STACKED [E, ...] tensor (the layer's persistent
 * nn.Parameter, fed DIRECTLY -- NO Python index_select/gather). The op slices each
 * chunk's [C,...] sub-tensor ON-DEVICE (a `slice` node on PyTorch dim 0). So the held-
 * leaf weights ARE the recipe inputs (multi-bucket-capture-safe: no freed python
 * intermediate). x is the [T,H] shared activation (the held-leaf x_buf slice). The
 * router weight is a DENSE [E, T, 1] tensor == the transpose of the [T, E] weight
 * matrix the Python side builds from topk_ids/topk_weights (W[t,e] = sum of router
 * weights of every top_k slot of token t that routed to LOCAL expert e, 0 otherwise).
 *
 *   Schema (added to the hpu_moe_ext library via TORCH_LIBRARY_FRAGMENT so it
 *   co-loads with the single-expert + routed + batched .so without a duplicate-
 *   library abort):
 *     fused_moe_monolithic(Tensor x,
 *               Tensor w13_packed, Tensor w13_scale, Tensor w13_zp,
 *               Tensor w2_packed,  Tensor w2_scale,  Tensor w2_zp,
 *               Tensor router_weights, int chunk) -> Tensor
 *   x is [T,H] bf16. Each weight tensor is STACKED over E experts (PyTorch row-major,
 *   the proven per-expert op layout with a leading E axis):
 *     w13_packed[E, H, 2I/8] i32  w13_scale[E, H/GS, 2I] bf16  w13_zp[E, H/GS, 2I/8] i32
 *     w2_packed [E, I, H/8]  i32  w2_scale [E, I/GS, H]  bf16  w2_zp [E, I/GS, H/8]  i32
 *     router_weights[E, T, 1] bf16  (dense per-(token,expert) weight; NON-routed
 *                                    (t,e) == 0 so they contribute exactly 0)
 *     chunk  int   (the SRAM-pressure knob: experts processed per batched sub-chain;
 *                   clamped to [1, E]. E need NOT be divisible by chunk -- the last
 *                   chunk is the remainder.)
 *   Output [T,H] bf16 = sum_e( fused_expert_e(x) * router_weights[e] ) ==
 *   the stock mixture_of_experts semantics (out[t] = sum_e expert_e(x[t]) * W[t,e]).
 *
 * ============================ DIM DERIVATION + CHECKS ==========================
 * E is the leading dim-0 of every weight tensor; T,H come from x; OUT13(=2I),I come
 * from w13_scale [E, H/GS, 2I]. T MUST match router_weights dim 1. All shapes are
 * cross-checked LOUDLY (a mismatch is a wiring bug -> fail fast). The bridge recipe-
 * caches one recipe per (E, chunk, T, H, I) -- all static for a given serve/bucket.
 *
 * Frontend-free (does NOT pull eager_op.h or lazy_kernels.h), so the eager TU
 * (moe_fused_monolithic_backend.cpp) and the lazy TU
 * (moe_fused_monolithic_backend_lazy.cpp) each include it + add their own frontend +
 * the TORCH_LIBRARY_FRAGMENT/_IMPL + KernelRegistry registration -- mirroring the
 * batched/sequential shared-header split.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr / NodeOutputAttr, BuildReshape/BuildBroadcast
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND
#include "backend/synapse_helpers/graph.h"

#include "moe_fused_monolithic_emit.h"   // emit_monolithic_experts (the chunked E-expert chain)

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: out is [T, H] bf16 (== x). T is read from x (stack pos 0), so it is
// correct for ANY token count (the concurrent caller's T). Header-inline so the
// single including TU owns the definition. (Same shape/dtype contract as the other
// fused ops -- but a distinct symbol to avoid an ODR clash if multiple shared headers
// are pulled into one TU.)
// =============================================================================
inline OutputMetaDataVector MoeFusedMonolithicMeta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [T, H]
  meta[0].dtype = x.scalar_type();      // bf16
  return meta;
}

// =============================================================================
// The MONOLITHIC OpBackend: ONE AddNode emits the chunked E-expert chain (via
// emit_monolithic_experts) -> ONE captured recipe that computes ALL E experts with
// per-expert SRAM-resident dequant, applying the dense per-(token,expert) routing.
// =============================================================================
struct MoeFusedMonolithic : OpBackend {
  MoeFusedMonolithic(int device_id, c10::ScalarType scalar_type)
      // guid "batch_gemm" is a harmless placeholder (real guids are per-BuildNode);
      // res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMonolithicMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMonolithic::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMonolithic::AddNode");

  // ---- typed inputs (order == schema): x + 6 stacked [E,...] weight tensors + the
  //      dense [E,T,1] router-weight tensor + the int chunk size. Each weight is ONE
  //      TensorsPair (a single stacked tensor) -- one syn_in index per class. ----
  auto x          = sg.getNextInput<TensorsPair>();  // bf16  [T, H]
  auto w13_packed = sg.getNextInput<TensorsPair>();  // int32 [E, H, 2I/8]
  auto w13_scale  = sg.getNextInput<TensorsPair>();  // bf16  [E, H/GS, 2I]
  auto w13_zp     = sg.getNextInput<TensorsPair>();  // int32 [E, H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<TensorsPair>();  // int32 [E, I, H/8]
  auto w2_scale   = sg.getNextInput<TensorsPair>();  // bf16  [E, I/GS, H]
  auto w2_zp      = sg.getNextInput<TensorsPair>();  // int32 [E, I/GS, H/8]
  auto router_w   = sg.getNextInput<TensorsPair>();  // bf16  [E, T, 1]
  auto chunk_in   = sg.getNextInput<long>();         // int   chunk size

  // ---- derive dims + LOUD cross-checks (PyTorch row-major; last dim == FCD) ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe_monolithic] x must be 2D [T,H], got dim ",
              x.pt_t.dim());
  const int64_t T = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto w13p_sizes = w13_packed.pt_t.sizes().vec();   // [E, H, 2I/8]
  TORCH_CHECK(w13_packed.pt_t.dim() == 3,
              "[fused_moe_monolithic] w13_packed must be 3D [E,H,2I/8], got dim ",
              w13_packed.pt_t.dim());
  const int64_t E = w13p_sizes[0];
  TORCH_CHECK(E >= 1, "[fused_moe_monolithic] need >=1 expert, got E=", E);
  TORCH_CHECK(w13p_sizes[1] == H,
              "[fused_moe_monolithic] w13_packed dim1 (", w13p_sizes[1],
              ") must == x H (", H, ")");

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();     // [E, H/GS, 2I]
  TORCH_CHECK(w13_scale.pt_t.dim() == 3,
              "[fused_moe_monolithic] w13_scale must be 3D [E,H/GS,2I], got dim ",
              w13_scale.pt_t.dim());
  TORCH_CHECK(s13_sizes[0] == E,
              "[fused_moe_monolithic] w13_scale dim0 (", s13_sizes[0],
              ") must == E (", E, ")");
  const int64_t OUT13 = s13_sizes[2];                      // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_monolithic] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  // The remaining stacked tensors must all share the leading E (a mismatch is a
  // caller wiring bug; fail fast). The router weight is [E, T, 1] (dense per-
  // (token,expert)); its dim 1 MUST == T (the x token count).
  TORCH_CHECK(
      w13_zp.pt_t.dim() == 3 && w13_zp.pt_t.size(0) == E &&
      w2_packed.pt_t.dim() == 3 && w2_packed.pt_t.size(0) == E &&
      w2_scale.pt_t.dim() == 3 && w2_scale.pt_t.size(0) == E &&
      w2_zp.pt_t.dim() == 3 && w2_zp.pt_t.size(0) == E,
      "[fused_moe_monolithic] every stacked weight tensor must be 3D with leading "
      "dim E=", E, " (w13_zp=", w13_zp.pt_t.sizes(),
      " w2_packed=", w2_packed.pt_t.sizes(), " w2_scale=", w2_scale.pt_t.sizes(),
      " w2_zp=", w2_zp.pt_t.sizes(), ")");
  const auto rw_sizes = router_w.pt_t.sizes().vec();        // [E, T, 1]
  TORCH_CHECK(
      router_w.pt_t.dim() == 3 && rw_sizes[0] == E && rw_sizes[1] == T &&
      rw_sizes[2] == 1,
      "[fused_moe_monolithic] router_weights must be [E,T,1] (dense per-(token,"
      "expert)), got ", router_w.pt_t.sizes(), " (E=", E, " T=", T, ")");

  // clamp chunk to [1, E] (a 0/negative or oversized chunk is a caller slip; a CHUNK
  // > E degenerates to one chunk of E, which is correct).
  int64_t CHUNK = static_cast<int64_t>(chunk_in);
  if (CHUNK < 1) CHUNK = 1;
  if (CHUNK > E) CHUNK = E;

  // ===========================================================================
  // ONE chunked chain -> the single [T,H] result. emit_monolithic_experts ties its
  // final chunk-add (or the lone chunk's reduce) to syn_out(0).
  // ===========================================================================
  auto out = emit_monolithic_experts(
      this, graph,
      x.syn_t, T, H, E, CHUNK, I, OUT13,
      w13_packed.syn_t, w13_packed.pt_t.sizes().vec(),
      w13_scale.syn_t,  w13_scale.pt_t.sizes().vec(),
      w13_zp.syn_t,     w13_zp.pt_t.sizes().vec(),
      w2_packed.syn_t,  w2_packed.pt_t.sizes().vec(),
      w2_scale.syn_t,   w2_scale.pt_t.sizes().vec(),
      w2_zp.syn_t,      w2_zp.pt_t.sizes().vec(),
      router_w.syn_t,   router_w.pt_t.sizes().vec());

  syn_out(0) = std::move(out[0]);
}

} // namespace habana
