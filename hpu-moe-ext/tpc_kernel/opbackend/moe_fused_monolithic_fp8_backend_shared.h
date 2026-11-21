/*
 * moe_fused_monolithic_fp8_backend_shared.h -- the SHARED, frontend-agnostic backend
 * lowering for the MONOLITHIC (ALL-E experts) fused M3 swigluoai-FP8 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8`. Adapted from
 * moe_fused_monolithic_backend_shared.h (the proven GLM int4 monolithic backend) with
 * the input set + dim derivation changed for RESIDENT fp8 experts (no int4 packing, no
 * zero-point, no group scale) and a swigluoai (alpha,limit) gate.
 *
 * ============================ STACKED-E INPUT LAYOUT ===========================
 * The weights are the layer's PERSISTENT route-b fp8 stacks (built by
 * m3_fp8_static_patch._build_fp8_stacks): op._w13_fp8_stack [E,H,2I] fp8 +
 * op._s13_stack [E,2I] f32, op._w2_fp8_stack [E,I,H] fp8 + op._s2_stack [E,H] f32. They
 * are fed DIRECTLY (no Python gather); the op slices each chunk's [C,...] on-device.
 * x is the [T,H] shared activation. router_weights is a DENSE [E,T,1] bf16 tensor ==
 * the transpose of the [T,E] dense routing matrix the Python wrapper builds from
 * topk_ids/topk_weights (W[t,e] = sum of router weights of every top_k slot of token t
 * that routed to LOCAL expert e, 0 otherwise) -- IDENTICAL semantics to the int4 op.
 *
 *   Schema (TORCH_LIBRARY_FRAGMENT so it co-loads with the int4/fp8 .so's):
 *     fused_moe_monolithic_fp8(Tensor x,
 *               Tensor w13_fp8, Tensor w13_scale,
 *               Tensor w2_fp8,  Tensor w2_scale,
 *               Tensor router_weights, int chunk,
 *               float alpha, float limit) -> Tensor
 *   x is [T,H] bf16. Each weight tensor is STACKED over E experts (route-b orientation):
 *     w13_fp8[E, H, 2I] float8_e4m3fn   w13_scale[E, 2I] f32  (per-output-channel)
 *     w2_fp8 [E, I, H]  float8_e4m3fn   w2_scale [E, H]  f32  (per-output-channel)
 *     router_weights[E, T, 1] bf16  (dense per-(token,expert); NON-routed (t,e)==0)
 *     chunk  int    (SRAM-pressure knob; clamped to [1,E])
 *     alpha  float  (swigluoai sigmoid gain, 1.702)
 *     limit  float  (swigluoai clamp limit, 7.0)
 *   Output [T,H] bf16 = sum_e( swigluoai_expert_e(x) * router_weights[e] ).
 *
 * ============================ DIM DERIVATION + CHECKS ==========================
 * E is dim-0 of every weight tensor; T,H from x; OUT13(=2I) from w13_scale [E,2I] (or
 * w13_fp8 [E,H,2I] dim 2). I = 2I/2. T MUST match router_weights dim 1. All cross-
 * checked LOUDLY. (No GS/zero-point checks -- fp8 has no grouping or zero-point.)
 *
 * Frontend-free; the eager TU (moe_fused_monolithic_fp8_backend.cpp) and lazy TU
 * (moe_fused_monolithic_fp8_backend_lazy.cpp) each include it + add their frontend +
 * the TORCH_LIBRARY_FRAGMENT/_IMPL + KernelRegistry registration.
 */
#pragma once

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/stack_getter.h"
#include "backend/habana_operator.h"
#include "backend/synapse_helpers/graph.h"

#include "moe_fused_monolithic_fp8_emit.h"   // emit_monolithic_experts_fp8

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Out meta: out is [T,H] bf16 (== x). Distinct symbol from the int4 op to avoid an ODR
// clash if both shared headers are pulled into one TU.
inline OutputMetaDataVector MoeFusedMonolithicFp8Meta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [T, H]
  meta[0].dtype = x.scalar_type();      // bf16
  return meta;
}

struct MoeFusedMonolithicFp8 : OpBackend {
  MoeFusedMonolithicFp8(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMonolithicFp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMonolithicFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMonolithicFp8::AddNode");

  // ---- typed inputs (order == schema): x + 4 stacked fp8 weight/scale tensors + the
  //      dense [E,T,1] router + int chunk + float alpha/limit. ----
  auto x         = sg.getNextInput<TensorsPair>();  // bf16  [T, H]
  auto w13_fp8   = sg.getNextInput<TensorsPair>();  // fp8   [E, H, 2I]
  auto w13_scale = sg.getNextInput<TensorsPair>();  // f32   [E, 2I]
  auto w2_fp8    = sg.getNextInput<TensorsPair>();  // fp8   [E, I, H]
  auto w2_scale  = sg.getNextInput<TensorsPair>();  // f32   [E, H]
  auto router_w  = sg.getNextInput<TensorsPair>();  // bf16  [E, T, 1]
  auto chunk_in  = sg.getNextInput<long>();         // int   chunk size
  auto alpha_in  = sg.getNextInput<double>();        // float swigluoai alpha (1.702)
  auto limit_in  = sg.getNextInput<double>();        // float swigluoai limit (7.0)

  // ---- derive dims + LOUD cross-checks ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe_monolithic_fp8] x must be 2D [T,H], got dim ",
              x.pt_t.dim());
  const int64_t T = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto w13_sizes = w13_fp8.pt_t.sizes().vec();   // [E, H, 2I]
  TORCH_CHECK(w13_fp8.pt_t.dim() == 3,
              "[fused_moe_monolithic_fp8] w13_fp8 must be 3D [E,H,2I], got dim ",
              w13_fp8.pt_t.dim());
  const int64_t E = w13_sizes[0];
  TORCH_CHECK(E >= 1, "[fused_moe_monolithic_fp8] need >=1 expert, got E=", E);
  TORCH_CHECK(w13_sizes[1] == H,
              "[fused_moe_monolithic_fp8] w13_fp8 dim1 (", w13_sizes[1],
              ") must == x H (", H, ")");
  const int64_t OUT13 = w13_sizes[2];                  // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_monolithic_fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  const auto s13_sizes = w13_scale.pt_t.sizes().vec(); // [E, 2I]
  TORCH_CHECK(
      w13_scale.pt_t.dim() == 2 && s13_sizes[0] == E && s13_sizes[1] == OUT13,
      "[fused_moe_monolithic_fp8] w13_scale must be [E,2I]=[", E, ",", OUT13,
      "] (per-output-channel), got ", w13_scale.pt_t.sizes());

  const auto w2_sizes = w2_fp8.pt_t.sizes().vec();     // [E, I, H]
  TORCH_CHECK(
      w2_fp8.pt_t.dim() == 3 && w2_sizes[0] == E && w2_sizes[1] == I && w2_sizes[2] == H,
      "[fused_moe_monolithic_fp8] w2_fp8 must be [E,I,H]=[", E, ",", I, ",", H,
      "], got ", w2_fp8.pt_t.sizes());
  const auto s2_sizes = w2_scale.pt_t.sizes().vec();   // [E, H]
  TORCH_CHECK(
      w2_scale.pt_t.dim() == 2 && s2_sizes[0] == E && s2_sizes[1] == H,
      "[fused_moe_monolithic_fp8] w2_scale must be [E,H]=[", E, ",", H,
      "] (per-output-channel), got ", w2_scale.pt_t.sizes());

  const auto rw_sizes = router_w.pt_t.sizes().vec();   // [E, T, 1]
  TORCH_CHECK(
      router_w.pt_t.dim() == 3 && rw_sizes[0] == E && rw_sizes[1] == T && rw_sizes[2] == 1,
      "[fused_moe_monolithic_fp8] router_weights must be [E,T,1], got ",
      router_w.pt_t.sizes(), " (E=", E, " T=", T, ")");

  // fp8 dtype sanity (a wrong-dtype weight is a wiring bug -> fail fast).
  TORCH_CHECK(
      w13_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn &&
      w2_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "[fused_moe_monolithic_fp8] w13_fp8/w2_fp8 must be float8_e4m3fn, got ",
      w13_fp8.pt_t.scalar_type(), "/", w2_fp8.pt_t.scalar_type());

  int64_t CHUNK = static_cast<int64_t>(chunk_in);
  if (CHUNK < 1) CHUNK = 1;
  if (CHUNK > E) CHUNK = E;

  const float alpha = static_cast<float>(alpha_in);
  const float limit = static_cast<float>(limit_in);

  // ===========================================================================
  // ONE chunked chain -> the single [T,H] result.
  // ===========================================================================
  auto out = emit_monolithic_experts_fp8(
      this, graph,
      x.syn_t, T, H, E, CHUNK, I, OUT13, alpha, limit,
      w13_fp8.syn_t,   w13_fp8.pt_t.sizes().vec(),
      w13_scale.syn_t, w13_scale.pt_t.sizes().vec(),
      w2_fp8.syn_t,    w2_fp8.pt_t.sizes().vec(),
      w2_scale.syn_t,  w2_scale.pt_t.sizes().vec(),
      router_w.syn_t,  router_w.pt_t.sizes().vec());

  syn_out(0) = std::move(out[0]);
}

} // namespace habana
