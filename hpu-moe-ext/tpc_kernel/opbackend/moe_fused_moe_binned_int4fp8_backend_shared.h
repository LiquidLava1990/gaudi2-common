/*
 * moe_fused_moe_binned_int4fp8_backend_shared.h -- SHARED frontend-agnostic backend
 * lowering for the W4A8 BINNED grouped-GEMM MoE op
 * `torch.ops.hpu_moe_ext.fused_moe_binned_int4fp8`.
 *
 * Processes ONE expert GROUP [G,C,H] -> [G,C,H] from a capacity-permuted activation
 * x_perm[E,C,H] (the wrapper slices E into CHUNK groups via chunk_start/group, calls
 * this op per group, torch.cat's the outputs). Reuses the PROVEN int4->fp8 dequant +
 * fp8 MME chain (emit_binned_group_i4f8 -> moe_fused_monolithic_int4fp8_emit.h
 * helpers). NEW additive schema; the monolithic/int4fp8/routed ops are UNTOUCHED.
 *
 * Inputs (9 tensors + 2 ints):
 *   x_perm    bf16  [E, C, H]        (permuted: expert e owns C routed-token rows)
 *   w13_packed int32 [E, H, 2I/8]    (FULL stacked; sliced on-device at chunk_start)
 *   w13_scale  bf16  [E, H/GS, 2I]
 *   w13_zp     int32 [E, H/GS, 2I/8]
 *   w2_packed  int32 [E, I, H/8]
 *   w2_scale   bf16  [E, I/GS, H]
 *   w2_zp      int32 [E, I/GS, H/8]
 *   router_w   bf16  [E, C, 1]       (per-(expert,slot) router weight)
 *   chunk_start int                  (first expert of this group)
 *   group       int                  (G = experts in this group, <= E - chunk_start)
 * Output: bf16 [G, C, H] = router-weighted expert output for experts
 *   [chunk_start, chunk_start+G).  NO reduce (each (expert,slot) is a distinct token).
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

#include "moe_fused_moe_binned_int4fp8_emit.h"   // emit_binned_group_i4f8

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// Output meta: out is [G, C, H] bf16. G + C are read from the `group` int arg + x_perm.
inline OutputMetaDataVector MoeFusedMoeBinnedInt4Fp8Meta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);          // x_perm [E,C,H]
  const auto xs = x.sizes().vec();
  TORCH_CHECK(xs.size() == 3,
              "[fused_moe_binned_int4fp8] x_perm must be 3D [E,C,H], got dim ", xs.size());
  const int64_t C = xs[1];
  const int64_t H = xs[2];
  const int64_t group = stack.at(9).toInt();             // G
  OutputMetaDataVector meta(1);
  meta[0].shape = std::vector<int64_t>{group, C, H};
  meta[0].dtype = x.scalar_type();                       // bf16
  return meta;
}

struct MoeFusedMoeBinnedInt4Fp8 : OpBackend {
  MoeFusedMoeBinnedInt4Fp8(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "batch_gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedMoeBinnedInt4Fp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedMoeBinnedInt4Fp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter sg(this, stack, "MoeFusedMoeBinnedInt4Fp8::AddNode");

  auto x_perm     = sg.getNextInput<TensorsPair>();  // bf16  [E, C, H]
  auto w13_packed = sg.getNextInput<TensorsPair>();  // int32 [E, H, 2I/8]
  auto w13_scale  = sg.getNextInput<TensorsPair>();  // bf16  [E, H/GS, 2I]
  auto w13_zp     = sg.getNextInput<TensorsPair>();  // int32 [E, H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<TensorsPair>();  // int32 [E, I, H/8]
  auto w2_scale   = sg.getNextInput<TensorsPair>();  // bf16  [E, I/GS, H]
  auto w2_zp      = sg.getNextInput<TensorsPair>();  // int32 [E, I/GS, H/8]
  auto router_w   = sg.getNextInput<TensorsPair>();  // bf16  [E, C, 1]
  auto chunk_start_in = sg.getNextInput<long>();
  auto group_in       = sg.getNextInput<long>();

  const auto xs = x_perm.pt_t.sizes().vec();
  TORCH_CHECK(x_perm.pt_t.dim() == 3,
              "[fused_moe_binned_int4fp8] x_perm must be 3D [E,C,H], got dim ", x_perm.pt_t.dim());
  const int64_t E = xs[0];
  const int64_t C = xs[1];
  const int64_t H = xs[2];

  const auto w13p_sizes = w13_packed.pt_t.sizes().vec();
  TORCH_CHECK(w13_packed.pt_t.dim() == 3 && w13p_sizes[0] == E,
              "[fused_moe_binned_int4fp8] w13_packed must be [E,H,2I/8] with E=", E);
  TORCH_CHECK(w13p_sizes[1] == H,
              "[fused_moe_binned_int4fp8] w13_packed dim1 (", w13p_sizes[1],
              ") must == x H (", H, ")");

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();
  TORCH_CHECK(w13_scale.pt_t.dim() == 3 && s13_sizes[0] == E,
              "[fused_moe_binned_int4fp8] w13_scale must be [E,H/GS,2I] with E=", E);
  const int64_t OUT13 = s13_sizes[2];
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_moe_binned_int4fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  TORCH_CHECK(
      w13_zp.pt_t.dim() == 3 && w13_zp.pt_t.size(0) == E &&
      w2_packed.pt_t.dim() == 3 && w2_packed.pt_t.size(0) == E &&
      w2_scale.pt_t.dim() == 3 && w2_scale.pt_t.size(0) == E &&
      w2_zp.pt_t.dim() == 3 && w2_zp.pt_t.size(0) == E,
      "[fused_moe_binned_int4fp8] every stacked weight tensor must be 3D with leading dim E=", E);

  const auto rw_sizes = router_w.pt_t.sizes().vec();
  TORCH_CHECK(
      router_w.pt_t.dim() == 3 && rw_sizes[0] == E && rw_sizes[1] == C && rw_sizes[2] == 1,
      "[fused_moe_binned_int4fp8] router_weights must be [E,C,1], got ", router_w.pt_t.sizes());

  int64_t chunk_start = static_cast<int64_t>(chunk_start_in);
  int64_t G = static_cast<int64_t>(group_in);
  TORCH_CHECK(chunk_start >= 0 && G >= 1 && (chunk_start + G) <= E,
              "[fused_moe_binned_int4fp8] need 0<=chunk_start and chunk_start+group<=E (E=", E,
              ", chunk_start=", chunk_start, ", group=", G, ")");

  auto out = emit_binned_group_i4f8(
      this, graph,
      x_perm.syn_t, xs,
      E, chunk_start, G, C, H, I, OUT13,
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
