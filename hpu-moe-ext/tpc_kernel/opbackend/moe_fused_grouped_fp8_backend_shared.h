/*
 * moe_fused_grouped_fp8_backend_shared.h -- the SHARED, frontend-agnostic backend
 * lowering for the GROUPED (routed top-k) fused M3 swigluoai-FP8 MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_grouped_fp8`. The CONCURRENCY-regime sibling of the
 * monolithic fp8 op: it consumes the Python wrapper's PER-EXPERT gathered token tiles
 * xg[E,Cap,H] and returns the raw per-expert yg[E,Cap,H] (the wrapper weights + scatters
 * back to [T,H]). Modeled byte-for-byte on moe_fused_monolithic_fp8_backend_shared.h.
 *
 * ============================ INPUT LAYOUT =====================================
 *   Schema (TORCH_LIBRARY_FRAGMENT so it co-loads with the monolithic/route .so's):
 *     fused_moe_grouped_fp8(Tensor xg,
 *               Tensor w13_fp8, Tensor w13_scale,
 *               Tensor w2_fp8,  Tensor w2_scale,
 *               float alpha, float limit) -> Tensor
 *   xg is [E,Cap,H] bf16 (E per-expert capacity bins of Cap gathered tokens). The
 *   weights are the layer's PERSISTENT route-b fp8 stacks (m3_fp8_static_patch.
 *   _build_fp8_stacks), STACKED over E in route-b orientation, fed DIRECTLY:
 *     w13_fp8[E, H, 2I] float8_e4m3fn   w13_scale[E, 2I] f32  (per-output-channel)
 *     w2_fp8 [E, I, H]  float8_e4m3fn   w2_scale [E, H]  f32  (per-output-channel)
 *     alpha  float  (swigluoai sigmoid gain, 1.702)
 *     limit  float  (swigluoai clamp limit, 7.0)
 *   Output [E,Cap,H] bf16 = swigluoai_expert_e(xg[e]) per bin (UNWEIGHTED, UNSCATTERED).
 *
 * ============================ DIM DERIVATION + CHECKS ==========================
 * E,Cap,H from xg; OUT13(=2I) from w13_fp8 [E,H,2I] dim 2 (cross-checked vs w13_scale
 * [E,2I]); I = 2I/2; w2_fp8 must be [E,I,H], the scales per-output-channel. All LOUD.
 *
 * Frontend-free; the eager TU (moe_fused_grouped_fp8_backend.cpp) and lazy TU
 * (moe_fused_grouped_fp8_backend_lazy.cpp) each include it + add their frontend + the
 * TORCH_LIBRARY_FRAGMENT/_IMPL + KernelRegistry registration.
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

#include "moe_fused_grouped_fp8_emit.h"   // emit_grouped_experts_fp8

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Out meta: out is [E,Cap,H] bf16 (== xg). Distinct symbol to avoid an ODR clash if
// multiple shared headers are pulled into one TU.
inline OutputMetaDataVector MoeFusedGroupedFp8Meta(const at::Stack& stack) {
  const at::Tensor& xg = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = xg.sizes().vec();      // [E, Cap, H]
  meta[0].dtype = xg.scalar_type();      // bf16
  return meta;
}

struct MoeFusedGroupedFp8 : OpBackend {
  MoeFusedGroupedFp8(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedGroupedFp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

// Same routed/grouped engine with GLM-5.3's checkpoint orientation and exact
// clamped-SwiGLU contract.  It is a separate op name so the existing M3
// swigluoai ABI and recipes remain byte-for-byte compatible.
struct MoeFusedGroupedGlmFp8 : OpBackend {
  MoeFusedGroupedGlmFp8(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedGroupedFp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

// Canonical backend used by every model adapter.  The two integer descriptor
// inputs are static per captured recipe, allowing one op ABI to select the
// exact activation and checkpoint orientation without a model-specific kernel.
struct MoeFusedGroupedFp8Smart : OpBackend {
  MoeFusedGroupedFp8Smart(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedGroupedFp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedGroupedFp8Smart::AddNode(
    sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedGroupedFp8Smart::AddNode");

  auto xg              = sg.getNextInput<TensorsPair>();
  auto w13_fp8         = sg.getNextInput<TensorsPair>();
  auto w13_scale       = sg.getNextInput<TensorsPair>();
  auto w2_fp8          = sg.getNextInput<TensorsPair>();
  auto w2_scale        = sg.getNextInput<TensorsPair>();
  auto activation_mode = sg.getNextInput<int64_t>();
  auto weight_layout   = sg.getNextInput<int64_t>();
  auto alpha_in        = sg.getNextInput<double>();
  auto limit_in        = sg.getNextInput<double>();

  TORCH_CHECK(
      xg.pt_t.dim() == 3,
      "[fused_moe_grouped_fp8_smart] xg must be [E,Cap,H], got ",
      xg.pt_t.sizes());
  const auto xg_sizes = xg.pt_t.sizes().vec();
  const int64_t E = xg_sizes[0];
  const int64_t Cap = xg_sizes[1];
  const int64_t H = xg_sizes[2];
  TORCH_CHECK(
      E >= 1 && Cap >= 1,
      "[fused_moe_grouped_fp8_smart] need E>=1,Cap>=1, got E=", E,
      " Cap=", Cap);
  TORCH_CHECK(
      (activation_mode == 0 && weight_layout == 0) ||
          (activation_mode == 1 && weight_layout == 1),
      "[fused_moe_grouped_fp8_smart] unsupported descriptor activation_mode=",
      activation_mode, " weight_layout=", weight_layout);

  const auto w13_sizes = w13_fp8.pt_t.sizes().vec();
  TORCH_CHECK(
      w13_fp8.pt_t.dim() == 3 && w13_sizes[0] == E,
      "[fused_moe_grouped_fp8_smart] w13_fp8 must be rank-3 with E=", E,
      ", got ", w13_fp8.pt_t.sizes());
  const int64_t OUT13 =
      weight_layout == 0 ? w13_sizes[2] : w13_sizes[1];
  TORCH_CHECK(
      (weight_layout == 0 && w13_sizes[1] == H) ||
          (weight_layout == 1 && w13_sizes[2] == H),
      "[fused_moe_grouped_fp8_smart] w13 layout mismatch for H=", H,
      " layout=", weight_layout, " shape=", w13_fp8.pt_t.sizes());
  TORCH_CHECK(
      OUT13 % 2 == 0,
      "[fused_moe_grouped_fp8_smart] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();
  TORCH_CHECK(
      w13_scale.pt_t.dim() == 2 && s13_sizes[0] == E &&
          s13_sizes[1] == OUT13,
      "[fused_moe_grouped_fp8_smart] w13_scale must be [E,2I]=[", E,
      ",", OUT13, "], got ", w13_scale.pt_t.sizes());

  const auto w2_sizes = w2_fp8.pt_t.sizes().vec();
  TORCH_CHECK(
      w2_fp8.pt_t.dim() == 3 && w2_sizes[0] == E &&
          ((weight_layout == 0 && w2_sizes[1] == I && w2_sizes[2] == H) ||
           (weight_layout == 1 && w2_sizes[1] == H && w2_sizes[2] == I)),
      "[fused_moe_grouped_fp8_smart] w2 layout mismatch; expected layout=",
      weight_layout, " E=", E, " I=", I, " H=", H,
      " got ", w2_fp8.pt_t.sizes());
  const auto s2_sizes = w2_scale.pt_t.sizes().vec();
  TORCH_CHECK(
      w2_scale.pt_t.dim() == 2 && s2_sizes[0] == E && s2_sizes[1] == H,
      "[fused_moe_grouped_fp8_smart] w2_scale must be [E,H]=[", E,
      ",", H, "], got ", w2_scale.pt_t.sizes());

  TORCH_CHECK(
      w13_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn &&
          w2_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "[fused_moe_grouped_fp8_smart] weights must be float8_e4m3fn, got ",
      w13_fp8.pt_t.scalar_type(), "/", w2_fp8.pt_t.scalar_type());

  const float alpha = static_cast<float>(alpha_in);
  const float limit = static_cast<float>(limit_in);
  TORCH_CHECK(
      limit > 0.0f,
      "[fused_moe_grouped_fp8_smart] limit must be >0, got ", limit);

  auto out = emit_grouped_experts_fp8_smart(
      this, graph,
      xg.syn_t, E, Cap, H, I, OUT13,
      activation_mode, weight_layout, alpha, limit,
      w13_fp8.syn_t,   w13_fp8.pt_t.sizes().vec(),
      w13_scale.syn_t, w13_scale.pt_t.sizes().vec(),
      w2_fp8.syn_t,    w2_fp8.pt_t.sizes().vec(),
      w2_scale.syn_t,  w2_scale.pt_t.sizes().vec());
  syn_out(0) = std::move(out[0]);
}

inline void MoeFusedGroupedFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedGroupedFp8::AddNode");

  // ---- typed inputs (order == schema): xg + 4 stacked fp8 weight/scale + alpha/limit ----
  auto xg        = sg.getNextInput<TensorsPair>();  // bf16  [E, Cap, H]
  auto w13_fp8   = sg.getNextInput<TensorsPair>();  // fp8   [E, H, 2I]
  auto w13_scale = sg.getNextInput<TensorsPair>();  // f32   [E, 2I]
  auto w2_fp8    = sg.getNextInput<TensorsPair>();  // fp8   [E, I, H]
  auto w2_scale  = sg.getNextInput<TensorsPair>();  // f32   [E, H]
  auto alpha_in  = sg.getNextInput<double>();        // float swigluoai alpha (1.702)
  auto limit_in  = sg.getNextInput<double>();        // float swigluoai limit (7.0)

  // ---- derive dims + LOUD cross-checks ----
  const auto xg_sizes = xg.pt_t.sizes().vec();
  TORCH_CHECK(xg.pt_t.dim() == 3,
              "[fused_moe_grouped_fp8] xg must be 3D [E,Cap,H], got dim ", xg.pt_t.dim());
  const int64_t E = xg_sizes[0];
  const int64_t Cap = xg_sizes[1];
  const int64_t H = xg_sizes[2];
  TORCH_CHECK(E >= 1 && Cap >= 1, "[fused_moe_grouped_fp8] need E>=1,Cap>=1, got E=", E,
              " Cap=", Cap);

  const auto w13_sizes = w13_fp8.pt_t.sizes().vec();   // [E, H, 2I]
  TORCH_CHECK(w13_fp8.pt_t.dim() == 3 && w13_sizes[0] == E && w13_sizes[1] == H,
              "[fused_moe_grouped_fp8] w13_fp8 must be [E,H,2I]=[", E, ",", H,
              ",2I], got ", w13_fp8.pt_t.sizes());
  const int64_t OUT13 = w13_sizes[2];                  // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_grouped_fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  const auto s13_sizes = w13_scale.pt_t.sizes().vec(); // [E, 2I]
  TORCH_CHECK(
      w13_scale.pt_t.dim() == 2 && s13_sizes[0] == E && s13_sizes[1] == OUT13,
      "[fused_moe_grouped_fp8] w13_scale must be [E,2I]=[", E, ",", OUT13,
      "] (per-output-channel), got ", w13_scale.pt_t.sizes());

  const auto w2_sizes = w2_fp8.pt_t.sizes().vec();     // [E, I, H]
  TORCH_CHECK(
      w2_fp8.pt_t.dim() == 3 && w2_sizes[0] == E && w2_sizes[1] == I && w2_sizes[2] == H,
      "[fused_moe_grouped_fp8] w2_fp8 must be [E,I,H]=[", E, ",", I, ",", H,
      "], got ", w2_fp8.pt_t.sizes());
  const auto s2_sizes = w2_scale.pt_t.sizes().vec();   // [E, H]
  TORCH_CHECK(
      w2_scale.pt_t.dim() == 2 && s2_sizes[0] == E && s2_sizes[1] == H,
      "[fused_moe_grouped_fp8] w2_scale must be [E,H]=[", E, ",", H,
      "] (per-output-channel), got ", w2_scale.pt_t.sizes());

  // fp8 dtype sanity (a wrong-dtype weight is a wiring bug -> fail fast).
  TORCH_CHECK(
      w13_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn &&
      w2_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "[fused_moe_grouped_fp8] w13_fp8/w2_fp8 must be float8_e4m3fn, got ",
      w13_fp8.pt_t.scalar_type(), "/", w2_fp8.pt_t.scalar_type());

  const float alpha = static_cast<float>(alpha_in);
  const float limit = static_cast<float>(limit_in);

  // ===========================================================================
  // ONE batched grouped chain -> the single [E,Cap,H] result.
  // ===========================================================================
  auto out = emit_grouped_experts_fp8(
      this, graph,
      xg.syn_t, E, Cap, H, I, OUT13, alpha, limit,
      w13_fp8.syn_t,   w13_fp8.pt_t.sizes().vec(),
      w13_scale.syn_t, w13_scale.pt_t.sizes().vec(),
      w2_fp8.syn_t,    w2_fp8.pt_t.sizes().vec(),
      w2_scale.syn_t,  w2_scale.pt_t.sizes().vec());

  syn_out(0) = std::move(out[0]);
}

inline void MoeFusedGroupedGlmFp8::AddNode(
    sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedGroupedGlmFp8::AddNode");

  // xg [E,Cap,H], checkpoint-native weights [E,2I,H]/[E,H,I], scales,
  // and the model's clamp limit (10.0).
  auto xg        = sg.getNextInput<TensorsPair>();
  auto w13_fp8   = sg.getNextInput<TensorsPair>();
  auto w13_scale = sg.getNextInput<TensorsPair>();
  auto w2_fp8    = sg.getNextInput<TensorsPair>();
  auto w2_scale  = sg.getNextInput<TensorsPair>();
  auto limit_in  = sg.getNextInput<double>();

  const auto xg_sizes = xg.pt_t.sizes().vec();
  TORCH_CHECK(
      xg.pt_t.dim() == 3,
      "[fused_moe_grouped_glm_fp8] xg must be [E,Cap,H], got ",
      xg.pt_t.sizes());
  const int64_t E = xg_sizes[0];
  const int64_t Cap = xg_sizes[1];
  const int64_t H = xg_sizes[2];
  TORCH_CHECK(
      E >= 1 && Cap >= 1,
      "[fused_moe_grouped_glm_fp8] need E>=1,Cap>=1, got E=", E,
      " Cap=", Cap);

  // GLM checkpoint-native orientation.  Both GEMMs use transpose_b=true.
  const auto w13_sizes = w13_fp8.pt_t.sizes().vec();
  TORCH_CHECK(
      w13_fp8.pt_t.dim() == 3 && w13_sizes[0] == E && w13_sizes[2] == H,
      "[fused_moe_grouped_glm_fp8] w13_fp8 must be [E,2I,H]=[", E,
      ",2I,", H, "], got ", w13_fp8.pt_t.sizes());
  const int64_t OUT13 = w13_sizes[1];
  TORCH_CHECK(
      OUT13 % 2 == 0,
      "[fused_moe_grouped_glm_fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();
  TORCH_CHECK(
      w13_scale.pt_t.dim() == 2 && s13_sizes[0] == E &&
          s13_sizes[1] == OUT13,
      "[fused_moe_grouped_glm_fp8] w13_scale must be [E,2I]=[", E,
      ",", OUT13, "], got ", w13_scale.pt_t.sizes());

  const auto w2_sizes = w2_fp8.pt_t.sizes().vec();
  TORCH_CHECK(
      w2_fp8.pt_t.dim() == 3 && w2_sizes[0] == E &&
          w2_sizes[1] == H && w2_sizes[2] == I,
      "[fused_moe_grouped_glm_fp8] w2_fp8 must be [E,H,I]=[", E,
      ",", H, ",", I, "], got ", w2_fp8.pt_t.sizes());
  const auto s2_sizes = w2_scale.pt_t.sizes().vec();
  TORCH_CHECK(
      w2_scale.pt_t.dim() == 2 && s2_sizes[0] == E && s2_sizes[1] == H,
      "[fused_moe_grouped_glm_fp8] w2_scale must be [E,H]=[", E,
      ",", H, "], got ", w2_scale.pt_t.sizes());

  TORCH_CHECK(
      w13_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn &&
          w2_fp8.pt_t.scalar_type() == c10::ScalarType::Float8_e4m3fn,
      "[fused_moe_grouped_glm_fp8] weights must be float8_e4m3fn, got ",
      w13_fp8.pt_t.scalar_type(), "/", w2_fp8.pt_t.scalar_type());

  const float limit = static_cast<float>(limit_in);
  TORCH_CHECK(
      limit > 0.0f,
      "[fused_moe_grouped_glm_fp8] limit must be >0, got ", limit);

  auto out = emit_grouped_experts_glm_fp8(
      this, graph,
      xg.syn_t, E, Cap, H, I, OUT13, limit,
      w13_fp8.syn_t,   w13_fp8.pt_t.sizes().vec(),
      w13_scale.syn_t, w13_scale.pt_t.sizes().vec(),
      w2_fp8.syn_t,    w2_fp8.pt_t.sizes().vec(),
      w2_scale.syn_t,  w2_scale.pt_t.sizes().vec());

  syn_out(0) = std::move(out[0]);
}

} // namespace habana
