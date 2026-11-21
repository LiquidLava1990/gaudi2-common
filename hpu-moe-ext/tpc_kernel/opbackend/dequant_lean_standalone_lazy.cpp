/*
 * dequant_lean_standalone_lazy.cpp  -- minimal standalone OpBackend op to test the
 * CORRECTED LEAN int4->bf16 dequant kernel (moe_ext_unpack_dequant_w4_lean_bf16) on
 * device, fed the way the kernel requires: a uint4 [OUT, IN] tensor via reinterpret_
 * cast (the generic UserCustomOp path CANNOT reinterpret int32->uint4, which is why
 * the previous moe_ext_custom_op_fast.cpp standalone produced garbage -- it handed the
 * u4-descriptor kernel a raw int32 tensor + a stride-8 store that the TPC does not honor).
 *
 * This op chains EXACTLY the proven fold edge:
 *     reinterpret_cast(int32-packed [OUT/8, IN]) -> uint4 [OUT, IN]
 *     moe_ext_unpack_dequant_w4_lean_bf16(weight_u4, scale) -> bf16 [OUT, IN]
 * so a PASS here proves the same kernel+geometry the #3 fold uses.
 *
 * Registers torch.ops.hpu_moe_ext.dequant_w4_lean_sa(weight_packed_i32, weight_scale,
 *   group_size, zero_const) -> Tensor   (PyTorch row-major: weight_packed [IN, OUT/8]
 *   int32, weight_scale [groups, OUT] bf16  -> out [IN, OUT] bf16). NO zero_point input
 *   (the symmetric zero=8 is folded in the kernel).
 *
 * Requires GC_KERNEL_PATH to include libmoe_ext_perflib_fast.so (the perf-lib that
 * registers the LEAN guid). Build: opbackend/build_dequant_lean_lazy.sh (mirrors
 * build_routed_lazy.sh byte-for-byte; same backend-only link).
 */

// Force fmt's header-only mode BEFORE any bridge header pulls in fmt/{core,format}.h
// (the bridge's hl_logger uses fmt). Without this, fmt::v12::assert_fail /
// detail::is_printable are referenced as EXTERNAL symbols (we link no fmt lib) ->
// "undefined symbol: fmt::v12::assert_fail" at load. The routed/emit TUs guard the
// same way (moe_fused_expert_emit.h defines FMT_HEADER_ONLY); mirror it here.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "hpu_ops/op_backend.h"        // OpBackend, NodeAttr / NodeOutputAttr
#include "hpu_ops/hpu_op_helper.h"
#include "backend/synapse_helpers/graph.h"
#include "backend/habana_operator.h"   // KernelRegistry(), REGISTER_HPU_BACKEND
#include "habana_kernels/lazy_kernels.h"  // habana_lazy::LazyOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace habana {

namespace sh = synapse_helpers;
using namespace std::literals;

// Output meta: weight_packed PyTorch [IN, OUT/8] int32 -> out [IN, OUT] bf16.
static OutputMetaDataVector DequantLeanSaMeta(const at::Stack& stack);

// Backend: reinterpret_cast(int32 -> uint4, last dim *8) -> LEAN dequant guid.
struct DequantLeanSa : OpBackend {
  DequantLeanSa(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "memcpy", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(DequantLeanSaMeta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void DequantLeanSa::AddNode(sh::graph& graph, const at::Stack& stack) {
  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
  const at::Tensor& w_packed_t = stack.at(0).toTensor();   // [IN, OUT/8] int32
  const int64_t IN = w_packed_t.size(0);
  const int64_t OUT = w_packed_t.size(1) * 8;
  const int64_t group_size = stack.at(2).toInt();
  const int64_t zero_const = stack.at(3).toInt();

  synTensor w_packed = syn_in(0);   // int32 packed [IN, OUT/8]
  synTensor scale    = syn_in(1);   // bf16 [groups, OUT]

  // reinterpret_cast int32-packed [IN, OUT/8] -> uint8 [IN, OUT/2] (PyTorch order;
  // last dim *4 since 1 int32 = 4 bytes). The kernel loads these bytes and uses the
  // HARDWARE u4->u8 deinterleave (v_convert_u4_to_u8_all_b) -- the i32 / u4-descriptor
  // load does NOT auto-unpack nibbles (proven on-card: constant output).
  std::vector<int64_t> u8_shape{IN, OUT / 2};
  auto w_u8 = OpBackend::BuildNode(
      this, graph,
      {"reinterpret_cast",
       {w_packed},
       {{u8_shape, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint8}}});

  // dequant guid: 2 inputs (weight_u8, scale); zero=8 folded in-kernel.
  struct LeanParams { int32_t group_size; int32_t zero_const; } params{
      static_cast<int32_t>(group_size), static_cast<int32_t>(zero_const)};
  std::vector<int64_t> out_shape{IN, OUT};
  auto out = OpBackend::BuildNode(
      this, graph,
      {"moe_ext_unpack_dequant_w4_lean_bf16",
       {w_u8[0].get(), scale},
       {{out_shape, bf16, 0}},   // index 0 -> syn_out(0): the op result (PERSISTENT)
       &params, sizeof(params)});
  syn_out(0) = std::move(out[0]);
}

static OutputMetaDataVector DequantLeanSaMeta(const at::Stack& stack) {
  const at::Tensor& w = stack.at(0).toTensor();   // [IN, OUT/8] int32
  OutputMetaDataVector meta(1);
  meta[0].shape = {w.size(0), w.size(1) * 8};     // [IN, OUT] bf16
  meta[0].dtype = at::kBFloat16;
  return meta;
}

}  // namespace habana

namespace habana_lazy {
using namespace habana;

static at::Tensor dequant_w4_lean_sa_hpu(const at::Tensor& weight_packed,
                                         const at::Tensor& weight_scale,
                                         int64_t group_size, int64_t zero_const) {
  std::vector<at::IValue> inputs{weight_packed, weight_scale, group_size, zero_const};
  LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::dequant_w4_lean_sa", std::move(inputs)};
  hpu_op.SetOutputMetaFn(habana::DequantLeanSaMeta);
  RUN_MAYBE_WITH_ACC_THREAD(dequant_w4_lean_sa, hpu_op);
}

}  // namespace habana_lazy

TORCH_LIBRARY_FRAGMENT(hpu_moe_ext, m) {
  m.def(
      "dequant_w4_lean_sa(Tensor weight_packed, Tensor weight_scale, "
      "int group_size, int zero_const) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("dequant_w4_lean_sa", habana_lazy::dequant_w4_lean_sa_hpu);
}

static const auto& DequantLeanSaKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::dequant_w4_lean_sa", habana::DequantLeanSa);
