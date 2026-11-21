/*
 * moe_fused_monolithic_int4fp8_backend_shared.h -- SHARED frontend-agnostic backend
 * lowering for the W4A8 monolithic MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_monolithic_int4fp8`.
 *
 * IDENTICAL schema + input layout + dim checks to moe_fused_monolithic_backend_shared.h
 * (the bf16 monolithic op) -- the ONLY difference is the chunk chain calls
 * emit_monolithic_experts_i4f8 (int4->fp8 dequant + native fp8 MME, SRAM-resident)
 * instead of emit_monolithic_experts (int4->bf16 + bf16 MME). Same 9 inputs (x + 6
 * stacked int4 weight tensors + dense [E,T,1] router + int chunk) -> [T,H] bf16.
 *
 * Validated by DPQ (arXiv 2505.14638, Intel, Gaudi-2): int4->fp8->fp8-matmul = up to 3x
 * over W4A16. Gated OFF by default (Python apply_monolithic routes here only under
 * HPU_MOE_FUSED_MONO_INT4FP8=1 + HPU_MOE_FUSED_MONOLITHIC=1). The bf16 monolithic op +
 * the routed/batched ops are UNTOUCHED -- this is a NEW additive schema.
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

#include "moe_fused_monolithic_int4fp8_emit.h"   // emit_monolithic_experts_i4f8

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Output meta: out is [T,H] bf16 (== x). Distinct symbol (avoids ODR clash with the
// bf16 monolithic shared header if both are pulled into one TU).
inline OutputMetaDataVector MoeFusedMonolithicInt4Fp8Meta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [T, H]
  meta[0].dtype = x.scalar_type();      // bf16
  return meta;
}

struct MoeFusedMonolithicInt4Fp8 : OpBackend {
  MoeFusedMonolithicInt4Fp8(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMonolithicInt4Fp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMonolithicInt4Fp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMonolithicInt4Fp8::AddNode");

  auto x          = sg.getNextInput<TensorsPair>();  // bf16  [T, H]
  auto w13_packed = sg.getNextInput<TensorsPair>();  // int32 [E, H, 2I/8]
  auto w13_scale  = sg.getNextInput<TensorsPair>();  // bf16  [E, H/GS, 2I]
  auto w13_zp     = sg.getNextInput<TensorsPair>();  // int32 [E, H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<TensorsPair>();  // int32 [E, I, H/8]
  auto w2_scale   = sg.getNextInput<TensorsPair>();  // bf16  [E, I/GS, H]
  auto w2_zp      = sg.getNextInput<TensorsPair>();  // int32 [E, I/GS, H/8]
  auto router_w   = sg.getNextInput<TensorsPair>();  // bf16  [E, T, 1]
  auto chunk_in   = sg.getNextInput<long>();         // int   chunk size

  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_moe_monolithic_int4fp8] x must be 2D [T,H], got dim ",
              x.pt_t.dim());
  const int64_t T = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto w13p_sizes = w13_packed.pt_t.sizes().vec();
  TORCH_CHECK(w13_packed.pt_t.dim() == 3,
              "[fused_moe_monolithic_int4fp8] w13_packed must be 3D [E,H,2I/8], got dim ",
              w13_packed.pt_t.dim());
  const int64_t E = w13p_sizes[0];
  TORCH_CHECK(E >= 1, "[fused_moe_monolithic_int4fp8] need >=1 expert, got E=", E);
  TORCH_CHECK(w13p_sizes[1] == H,
              "[fused_moe_monolithic_int4fp8] w13_packed dim1 (", w13p_sizes[1],
              ") must == x H (", H, ")");

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();
  TORCH_CHECK(w13_scale.pt_t.dim() == 3,
              "[fused_moe_monolithic_int4fp8] w13_scale must be 3D [E,H/GS,2I], got dim ",
              w13_scale.pt_t.dim());
  TORCH_CHECK(s13_sizes[0] == E,
              "[fused_moe_monolithic_int4fp8] w13_scale dim0 (", s13_sizes[0],
              ") must == E (", E, ")");
  const int64_t OUT13 = s13_sizes[2];
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_monolithic_int4fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  TORCH_CHECK(
      w13_zp.pt_t.dim() == 3 && w13_zp.pt_t.size(0) == E &&
      w2_packed.pt_t.dim() == 3 && w2_packed.pt_t.size(0) == E &&
      w2_scale.pt_t.dim() == 3 && w2_scale.pt_t.size(0) == E &&
      w2_zp.pt_t.dim() == 3 && w2_zp.pt_t.size(0) == E,
      "[fused_moe_monolithic_int4fp8] every stacked weight tensor must be 3D with leading "
      "dim E=", E);
  const auto rw_sizes = router_w.pt_t.sizes().vec();
  TORCH_CHECK(
      router_w.pt_t.dim() == 3 && rw_sizes[0] == E && rw_sizes[1] == T && rw_sizes[2] == 1,
      "[fused_moe_monolithic_int4fp8] router_weights must be [E,T,1], got ",
      router_w.pt_t.sizes());

  int64_t CHUNK = static_cast<int64_t>(chunk_in);
  if (CHUNK < 1) CHUNK = 1;
  if (CHUNK > E) CHUNK = E;

  auto out = emit_monolithic_experts_i4f8(
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
