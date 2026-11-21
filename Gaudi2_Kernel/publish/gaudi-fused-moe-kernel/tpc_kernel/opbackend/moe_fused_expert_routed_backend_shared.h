/*
 * moe_fused_expert_routed_backend_shared.h -- the SHARED, frontend-agnostic
 * backend lowering for the K-EXPERT single-recipe fused int4 MoE op. It emits ALL
 * K routed experts' fused subgraphs into ONE OpBackend::AddNode -> ONE GC-captured
 * recipe -> ONE launch, collapsing the K separate op launches the Python wrapper
 * (hpu_moe_ext/fused_expert_routed.py) currently issues (one recipe EACH).
 *
 * THE WIN: at single-stream T=1 the per-expert recipe is already the fusion + SRAM
 * win (proven by the single-expert op). What dominates the host/launch overhead is
 * the K=8 separate graph dispatches/replays + K host-side accumulates per token.
 * Fusing the K subgraphs + the weighted sum into one AddNode means the bridge
 * recipe-caches ONE recipe for the whole MoE layer and the box does ONE launch per
 * token -- far less host/launch overhead, the per-token latency lever.
 *
 * RELATIONSHIP TO THE PROVEN SINGLE-EXPERT OP:
 *   - The per-expert node chain is emitted by `emit_one_expert(...)`
 *     (moe_fused_expert_emit.h), factored BYTE-FOR-BYTE out of the proven
 *     single-expert AddNode. This op calls it K times. The single-expert op is
 *     UNTOUCHED (it keeps its own inline copy of that chain).
 *   - Each expert's two int4 dequants stay NON-PERSISTENT, so the GC's
 *     fuseDequantIntoMoeGemm + SRAM slicer bundle EACH expert into its MMEs exactly
 *     as in the single-expert recipe -- the SRAM win is preserved PER expert inside
 *     the one fused recipe. (TOP RISK: K experts' dequant tiles co-resident in one
 *     recipe -- does the GC still bundle each, or spill to HBM? The on-device
 *     HBM-bytes/iter check confirms; see the test's CHECK 3.)
 *
 * STACKED-K INPUT LAYOUT -- K SEPARATE INPUT TENSORS PER WEIGHT CLASS (TensorList):
 *   We pass each weight class as a `Tensor[]` list of K per-expert tensors (NOT one
 *   stacked [K,...] tensor). This is the bridge's OWN idiom for a multi-expert op
 *   (hpu_ops/.../mixture_of_experts takes `at::TensorList w1, w2, w3`), and it is
 *   the cleanest possible mapping here: `StackGetter::getNextInput<
 *   std::vector<TensorsPair>>()` returns K TensorsPair whose `.syn_t` are ALREADY
 *   wired to consecutive `op->syn_in(...)` (the StackGetter increments the syn
 *   index once per list element -- stack_getter.h:134-156). So expert k's tensors
 *   are just `w13p[k].syn_t` etc. -- clean 2D `[H,2I/8]` / `[H/GS,2I]` / ... with
 *   NO dim-0 slice, NO squeeze, NO new geometry: `emit_one_expert` consumes them
 *   VERBATIM as the single-expert op consumes its per-tensor inputs. The Python
 *   side does the SAME gather as today (index_select -> [K,...]) then `.unbind(0)`
 *   to a list (cheap, HPU-graph-safe). Per-expert PyTorch shapes are homogeneous,
 *   so M/H/I/OUT13 are derived ONCE (from x and w13_scale[0]) and reused per k.
 *
 *   Schema (added to the existing `hpu_moe_ext` library via TORCH_LIBRARY_FRAGMENT
 *   so it co-loads with the single-expert .so without a duplicate-library abort):
 *     fused_moe(Tensor x,
 *               Tensor[] w13_packed, Tensor[] w13_scale, Tensor[] w13_zp,
 *               Tensor[] w2_packed,  Tensor[] w2_scale,  Tensor[] w2_zp,
 *               Tensor[] router_weights) -> Tensor
 *   x is [M,H] bf16 (M==T tokens; single-stream T=1 priority). Each list has K
 *   elements; element k is expert k's tensor in the PROVEN per-expert op layout:
 *     w13_packed[k][H, 2I/8] i32  w13_scale[k][H/GS, 2I] bf16  w13_zp[k][H/GS, 2I/8] i32
 *     w2_packed [k][I, H/8]  i32  w2_scale [k][I/GS, H]  bf16  w2_zp [k][I/GS, H/8]  i32
 *     router_weights[k][1,1] bf16  (this expert's scalar router weight, folded by
 *                                   emit_one_expert's final mult -- the T=1 fold;
 *                                   for T>1 the caller passes 1 and post-scales,
 *                                   exactly as fused_expert_routed.py does today).
 *   Output [M,H] bf16 = sum_k( fused_expert(x, W_k) * router_weights[k] ).
 *
 * ADDNODE ACCUMULATION -- a LEFT-FOLD add tree of add_fwd_bf16:
 *   acc = y_0 ;  acc = add_fwd_bf16(acc, y_k)  for k = 1..K-1 ;  syn_out(0)=acc.
 *   All acc_* are graph-internal (NON-PERSISTENT) except the final, which ties to
 *   syn_out(0) (final_result_index=0). For K==1 there is no add: emit_one_expert is
 *   called with is_final_result=true so its `* router_weight` node is the result.
 *   (A left fold, not a balanced tree: the K subgraphs are independent so the GC
 *   schedules them in parallel regardless; the add chain is K-1 cheap elementwise
 *   adds on [M,H] -- negligible vs the 2K MMEs. A balanced tree would not change
 *   the captured recipe's critical path materially.)
 *
 * This header carries ONLY the backend half (the recipe + its output-meta fn). It
 * is frontend-free (does NOT pull eager_op.h or lazy_kernels.h), so the eager TU
 * (moe_fused_expert_routed_backend.cpp) and the lazy TU
 * (moe_fused_expert_routed_backend_lazy.cpp) each include it and add their own
 * frontend + the TORCH_LIBRARY_FRAGMENT/_IMPL + KernelRegistry registration --
 * exactly mirroring the single-expert shared-header split.
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

#include "moe_fused_expert_emit.h"   // emit_one_expert (the per-expert chain)

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// Output meta: out is [M, H] bf16 (== x). M is read from x (stack pos 0), so it
// is correct for ANY token count (the routed caller's T), not just the single
// op's assumption. Header-inline so the single including TU owns the definition.
// =============================================================================
inline OutputMetaDataVector MoeFusedMoEMeta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [M, H]
  meta[0].dtype = x.scalar_type();      // bf16
  return meta;
}

// =============================================================================
// The K-expert OpBackend: one AddNode emits all K experts' fused subgraphs (via
// emit_one_expert) + the weighted-sum add-tree -> ONE captured recipe.
// =============================================================================
struct MoeFusedMoE : OpBackend {
  MoeFusedMoE(int device_id, c10::ScalarType scalar_type)
      // guid "gemm" is a harmless placeholder (real guids are per-BuildNode);
      // res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMoEMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMoE::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMoE::AddNode");

  // ---- typed inputs (order == schema): x, then 6 weight-class lists + the
  //      router-weight list. Each list is K per-expert tensors; getNextInput<
  //      std::vector<TensorsPair>>() returns K TensorsPair whose .syn_t are the
  //      consecutive syn_in(...) for that list's elements (StackGetter handles the
  //      syn-index increment per element). ----
  auto x          = sg.getNextInput<TensorsPair>();              // bf16 [M, H]
  auto w13_packed = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [H, 2I/8]
  auto w13_scale  = sg.getNextInput<std::vector<TensorsPair>>(); // K x bf16  [H/GS, 2I]
  auto w13_zp     = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [I, H/8]
  auto w2_scale   = sg.getNextInput<std::vector<TensorsPair>>(); // K x bf16  [I/GS, H]
  auto w2_zp      = sg.getNextInput<std::vector<TensorsPair>>(); // K x int32 [I/GS, H/8]
  auto router_w   = sg.getNextInput<std::vector<TensorsPair>>(); // K x bf16  [1, 1]

  // ---- K + homogeneity checks (LOUD on any mismatch; the routed caller always
  //      passes equal-length lists -- a mismatch is a wiring bug, fail fast) ----
  const int64_t K = static_cast<int64_t>(w13_packed.size());
  TORCH_CHECK(K >= 1, "[fused_moe] need >=1 expert, got K=", K);
  TORCH_CHECK(
      static_cast<int64_t>(w13_scale.size()) == K &&
      static_cast<int64_t>(w13_zp.size()) == K &&
      static_cast<int64_t>(w2_packed.size()) == K &&
      static_cast<int64_t>(w2_scale.size()) == K &&
      static_cast<int64_t>(w2_zp.size()) == K &&
      static_cast<int64_t>(router_w.size()) == K,
      "[fused_moe] all 7 expert lists must have the same length K=", K,
      " (w13_scale=", w13_scale.size(), " w13_zp=", w13_zp.size(),
      " w2_packed=", w2_packed.size(), " w2_scale=", w2_scale.size(),
      " w2_zp=", w2_zp.size(), " router_weights=", router_w.size(), ")");

  // ---- derive the loop-invariant dims ONCE (per-expert shapes are homogeneous):
  //      M,H from x; OUT13(=2I),I from w13_scale[0] [H/GS, 2I]. ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe] x must be 2D [M,H]");
  const int64_t M = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto s13_sizes = w13_scale[0].pt_t.sizes().vec();   // [H/GS, 2I]
  TORCH_CHECK(w13_scale[0].pt_t.dim() == 2,
              "[fused_moe] w13_scale[k] must be 2D [H/GS, 2I]");
  const int64_t OUT13 = s13_sizes[1];                       // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  // ===========================================================================
  // K-FOLD: emit each expert's proven subgraph -> y_k[M,H] (router weight folded),
  // then a left-fold add tree -> the single [M,H] result.
  //
  //   K==1: emit_one_expert(..., is_final_result=true) -- its final mult IS the
  //         op result (no add needed).
  //   K>=2: every emit_one_expert is non-final (y_k graph-internal); chain
  //         add_fwd_bf16, and the LAST add ties to syn_out(0).
  // ===========================================================================
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  std::vector<int64_t> mh_shape{M, H};   // NAMED (non-owning IntArrayRef) -- must
                                         // outlive every add BuildNode call.

  // Per-expert packed PyTorch shapes (passed to emit_one_expert; a raw synTensor
  // has no sizes). Homogeneous across k, but each tensor's own shape is used so a
  // caller layout slip is caught by the dequant SIF, not silently mis-shaped.
  auto emit_k = [&](int64_t k, bool is_final) {
    return emit_one_expert(
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
