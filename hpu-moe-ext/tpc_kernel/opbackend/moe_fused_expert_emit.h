/*
 * moe_fused_expert_emit.h -- the REUSABLE per-expert subgraph emitter, factored
 * VERBATIM out of the proven single-expert AddNode chain
 * (moe_fused_expert_backend_shared.h :: MoeFusedExpert::AddNode) so the new
 * K-EXPERT fused op (moe_fused_expert_routed_backend.cpp) emits the EXACT SAME,
 * numerically-correct (relNorm 4.2e-3 on device) node chain for each expert.
 *
 * WHY A 2ND HEADER (not editing moe_fused_expert_backend_shared.h's AddNode in
 * place): the single-expert .cpp/header is the PROVEN, committed foundation --
 * per the build contract it must NOT be edited. This header is purely ADDITIVE:
 * it declares one free function `emit_one_expert(...)` carrying the per-expert
 * body BYTE-FOR-BYTE (same guids, same PyTorch-row-major geometry, same
 * NON-PERSISTENT dequant->MME edges, same synSliceParamsV2 gate/up split, same
 * sigmoid+mult SiLU, same final `* router_weight`). The single-expert shared
 * header is left UNTOUCHED (its inline AddNode still works exactly as before);
 * only the NEW K-expert TUs include THIS header and call `emit_one_expert` K
 * times. Both the single op and the helper stay in lockstep -- if the proven
 * recipe ever changes, change it in BOTH.
 *
 * GROUND TRUTH: every node below is copied unchanged from
 * moe_fused_expert_backend_shared.h. The long GEOMETRY rationale (the node-6 SIF
 * NULL-deref fix, PyTorch-order everywhere, the named-vector lifetime rule for the
 * non-owning IntArrayRef sizes, the uint4 reinterpret out-attr, the [1,1]->[M,H]
 * router broadcast) lives in that file's comments and is NOT repeated here -- read
 * it first.
 *
 * This header is frontend-free (pulls ONLY op_backend.h / hpu_op_helper.h +
 * synapse params, exactly like the single-expert shared header) so either the
 * eager or the lazy K-expert TU can include it.
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

#include <algorithm>   // std::fill_n
#include <cstdint>
#include <cstdlib>     // std::getenv (MOE_DEQUANT_KERNEL fold toggle)
#include <string_view> // std::string_view (env compare)
#include <vector>

namespace habana {

namespace sh = synapse_helpers;

// =============================================================================
// emit_one_expert -- add the PROVEN per-expert fused-int4-MoE node chain for ONE
// expert and RETURN its [M,H] bf16 output (the BuildNode result vector; element
// [0].get() is the y_k synTensor). The router weight is ALREADY folded by the
// final mult. VERBATIM the single-expert AddNode body; only the inputs are passed
// as raw synTensors + their PyTorch shapes, so the caller can source them from
// the K-expert op's k-th TensorList element without this helper knowing which.
//
// Inputs (all PyTorch row-major; same per-expert shapes as the single-expert op):
//   x        bf16  [M, H]            (shared across all K experts; one synTensor)
//   M,H,I,OUT13                      (loop-invariant dims; OUT13 == 2I)
//   w13p     int32 [H, 2I/8]   + w13p_shape   (this expert's packed gate+up weight)
//   w13s     bf16  [H/GS, 2I]                 (this expert's gate+up scales)
//   w13z     int32 [H/GS, 2I/8] + w13z_shape  (this expert's gate+up zero-points)
//   w2p      int32 [I, H/8]     + w2p_shape    (this expert's packed down weight)
//   w2s      bf16  [I/GS, H]                   (this expert's down scales)
//   w2z      int32 [I/GS, H/8]  + w2z_shape    (this expert's down zero-points)
//   rw       bf16  [1, 1]                       (this expert's router weight)
//
// Shapes are passed EXPLICITLY because a raw synTensor carries no PyTorch sizes;
// the caller derives them once from the at::Tensor metadata (homogeneous across k).
//
// The returned y_k node is normally graph-internal (NON-PERSISTENT): the K-expert
// caller chains an add-tree onto the K y_k's and ties only the FINAL accumulator
// to syn_out(0). The `is_final_result` flag exists ONLY for the K==1 degenerate
// case (no add-tree): when true, this expert's final `* router_weight` node IS the
// op result (final_result_index=0). All intermediates are non-persistent, so the
// GC bundles EACH expert's two dequants into its MMEs exactly as in the
// single-expert recipe -- the SRAM-residency win is preserved PER expert.
// =============================================================================
inline std::vector<sh::tensor> emit_one_expert(
    OpBackend* op, sh::graph& graph,
    synTensor x, int64_t M, int64_t H, int64_t I, int64_t OUT13,
    synTensor w13p, const std::vector<int64_t>& w13p_shape,
    synTensor w13s,
    synTensor w13z, const std::vector<int64_t>& w13z_shape,
    synTensor w2p,  const std::vector<int64_t>& w2p_shape,
    synTensor w2s,
    synTensor w2z,  const std::vector<int64_t>& w2z_shape,
    synTensor rw,
    bool is_final_result = false) {
  using namespace std::literals;
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // --- reinterpret_cast int32-packed -> uint4 (VERBATIM): unpacked PyTorch shape
  //     = packed shape with LAST dim *8; out-attr carries the uint4 view. ---
  auto reinterpret_u4 = [&](synTensor packed, std::vector<int64_t> shape) {
    shape.back() *= 8;  // packed int32 last dim (cols/8) -> uint4 last dim (cols)
    return OpBackend::BuildNode(
        op, graph,
        {"reinterpret_cast",
         {packed},
         {{shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // --- dequant lambda: int4-packed -> bf16, PyTorch-order out, NON-PERSISTENT
  //     intermediate (so the GC bundles it INTO the following MME -> 0 MB workspace).
  //
  // Two paths, selected at runtime by env MOE_DEQUANT_KERNEL:
  //   default / "stock": the proven stock complex-guid dequantize_4_bit_bf16
  //                      (weight_u4, scale, zp_u4) -- relNorm 4.2e-3 on device.
  //   "lean"           : OUR custom TPC kernel moe_ext_unpack_dequant_w4_lean_bf16
  //                      (weight_u4, scale) -- the symmetric zero=8 is FOLDED in the
  //                      kernel (NO zp tensor/load), G128 scale hoisted once/group,
  //                      U4-DESCRIPTOR load + CONTIGUOUS store (byte-exact vs the
  //                      reference; moe_ext_unpack_dequant_w4_fast.c::main_lean).
  //                      Requires GC_KERNEL_PATH to include libmoe_ext_perflib_fast.so.
  //                      This is the "#3 fold": the unpack is bundled into the MME on
  //                      the SAME non-persistent edge -> 0 MB workspace, and it drops
  //                      stock's zp load. zp_packed/zp_shape are IGNORED for lean.
  static const bool kUseLeanDequant = [] {
    const char* e = std::getenv("MOE_DEQUANT_KERNEL");
    return e != nullptr && std::string_view(e) == "lean"sv;
  }();
  auto dequant = [&](synTensor w_packed, const std::vector<int64_t>& w_shape,
                     synTensor scale,
                     synTensor zp_packed, const std::vector<int64_t>& zp_shape) {
    std::vector<int64_t> out_shape = w_shape;
    out_shape.back() *= 8;
    if (kUseLeanDequant) {
      // OUR dequant guid feeds UINT8 (not uint4): the kernel byte-loads + uses the
      // hardware u4->u8 deinterleave (v_convert_u4_to_u8_all_b). int32-packed
      // [.., OUT/8] -> uint8 [.., OUT/2] (1 int32 = 4 bytes). 2 inputs (weight_u8,
      // scale); zero=8 folded in-kernel. group_size=128 (GLM G128).
      std::vector<int64_t> u8_shape = w_shape;
      u8_shape.back() *= 4;  // packed int32 last dim (cols/8) -> bytes (cols/2)
      auto w_u8 = OpBackend::BuildNode(
          op, graph,
          {"reinterpret_cast",
           {w_packed},
           {{u8_shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint8}}});
      struct LeanParams { int32_t group_size; int32_t zero_const; } lean_params{128, 8};
      std::vector<synTensor> in{w_u8[0].get(), scale};
      return OpBackend::BuildNode(
          op, graph,
          {"moe_ext_unpack_dequant_w4_lean_bf16",
           std::move(in),
           {{out_shape, bf16, std::nullopt}},  // NON-PERSISTENT intermediate
           &lean_params, sizeof(lean_params)});
    }
    // STOCK (default, proven): dequantize_4_bit_bf16(weight_u4, scale, zp_u4).
    auto w_u4 = reinterpret_u4(w_packed, w_shape);
    auto zp_u4 = reinterpret_u4(zp_packed, zp_shape);
    std::vector<synTensor> in{w_u4[0].get(), scale, zp_u4[0].get()};
    ns_CastKernel::ParamsV3 params{};
    return OpBackend::BuildNode(
        op, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, bf16),
         std::move(in),
         {{out_shape, bf16, std::nullopt}},  // NON-PERSISTENT intermediate
         &params, sizeof(params)});
  };

  // ---- dequant #1: W13_bf16 [H, 2I] (PyTorch order; graph-internal SRAM target) ----
  auto W13 = dequant(w13p, w13p_shape, w13s, w13z, w13z_shape);

  // ---- MME gemm #1: gu[M, 2I] = gemm(x[M,H], W13[H,2I]) (contract H) ----
  synGEMMParams gemm_params{false, false};
  std::vector<int64_t> gu_shape{M, OUT13};
  auto gu = OpBackend::BuildNode(
      op, graph,
      {"gemm",
       {x, W13[0].get()},
       {{gu_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- slice split along PyTorch feature dim 1 (2I): gate=gu[:, :I], up=gu[:, I:] ----
  const int64_t gu_rank = 2;                 // gu is PyTorch [M, 2I]
  const int64_t slice_dim = 1;               // PyTorch feature axis (2I)
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};                    // zero-init
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    const int64_t gu_dims[gu_rank] = {M, OUT13};
    for (int64_t i = 0; i < gu_rank; ++i) {
      sp.axes[i] = static_cast<unsigned>(gu_rank - i - 1);  // TPC-reversed axis
      if (i == slice_dim) {
        sp.starts[i] = static_cast<TSize>(begin);
        sp.ends[i] = static_cast<TSize>(end);
        sp.steps[i] = static_cast<TSize>(1);
      } else {
        sp.starts[i] = static_cast<TSize>(0);
        sp.ends[i] = static_cast<TSize>(gu_dims[i]);
        sp.steps[i] = static_cast<TSize>(1);
      }
    }
    std::vector<int64_t> out_shape{M, end - begin};  // PyTorch [M, I]
    return OpBackend::BuildNode(
        op, graph,
        {"slice",
         {in},
         {{out_shape, bf16, std::nullopt}},
         &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- SiLU(gate)=gate*sigmoid(gate); h = SiLU(gate) * up ----
  std::vector<int64_t> im_shape{M, I};
  auto sgte = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()},
       {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()},
       {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()},
       {{im_shape, bf16, std::nullopt}}});

  // ---- dequant #2: W2_bf16 [I, H] (PyTorch order; graph-internal SRAM target) ----
  auto W2 = dequant(w2p, w2p_shape, w2s, w2z, w2z_shape);

  // ---- MME gemm #2: y[M, H] = gemm(h[M,I], W2[I,H]) (contract I) ----
  std::vector<int64_t> mh_shape{M, H};
  auto y = OpBackend::BuildNode(
      op, graph,
      {"gemm",
       {h[0].get(), W2[0].get()},
       {{mh_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- y_k = y * router_weight ([1,1] broadcast over [M,H]) ----
  // Normally NON-PERSISTENT (the add-tree consumes it). Only for K==1 does the
  // caller pass is_final_result=true, making this the op result (final_idx 0).
  const std::optional<int> final_idx =
      is_final_result ? std::optional<int>(0) : std::nullopt;
  return OpBackend::BuildNode(
      op, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), rw},
       {{mh_shape, bf16, final_idx}}});
}

} // namespace habana
