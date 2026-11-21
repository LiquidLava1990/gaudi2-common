/*
 * moe_fused_expert_backend.cpp -- APPROACH 3: the PROVEN fused int4 MoE Synapse
 * graph as a REAL, GC-captured `torch.ops` op, built against the version-exact
 * open-source Habana bridge (gaudi-pytorch-bridge tag v1.24.0-1007 ==
 * habana-torch-plugin 1.24.0.1007, torch 2.10, _GLIBCXX_USE_CXX11_ABI=1).
 *
 * WHY THIS (vs approach B = manual synGraphCompile/synLaunch in
 * ../fusion/fused_moe_op.cpp): here the per-expert recipe is emitted *inside*
 * the bridge's own lowering, as a `habana::OpBackend` subclass whose AddNode()
 * chains BuildNode() calls. The bridge -> Graph Compiler then OWNS the recipe:
 *   - intermediates between BuildNode() calls carry NO persistent section, so
 *     the two dequant outputs (W13_bf16, W2_bf16) are graph-internal -> the GC's
 *     fuseDequantIntoMoeGemm + SRAM slicer bundle them into the MME (the
 *     SRAM-resident win), exactly as the standalone validator proved.
 *   - the op is captured into the HPU graph like any built-in op (no side-band
 *     synLaunch, no manual stream/workspace bookkeeping, recipe-cached by the
 *     bridge), so it is durable across HPU-graph capture / lazy replay -- the
 *     guaranteed-fast serving path.
 *
 * WHY NOT habana::custom_op::UserCustomOpDescriptor (the "registerUserCustomOp"
 * public path): verified from include/habanalabs/hpu_custom_op_pt2.h in this
 * exact tag -- UserCustomOpDescriptor binds ONE `guid_` (a single TPC kernel)
 * with one OutputMetaFn + one FillParamsFn. It cannot express our multi-node
 * recipe (2x reinterpret_cast pairs, 2x dequantize_4_bit_bf16, 2x MME `gemm`,
 * 2x slice, sigmoid+mult SiLU, router mult) nor the non-persistent
 * dequant->MME bundling. So we register a full OpBackend in the KernelRegistry,
 * which IS how the bridge ships its own multi-node ops (modeled verbatim on
 * hpu_ops/backend/mixture_of_experts_gen.cpp + convert_from_int4_gen.cpp +
 * full_gen.cpp from this tag).
 *
 * GRAPH PORTED (single expert; verbatim topology + FCD geometry + node guids +
 * the proven fixes from ../fusion/fused_expert_graph.h):
 *   int32 w13_packed --reinterpret_cast--> uint4 W13
 *   int32 w13_zp     --reinterpret_cast--> uint4 ZP13
 *     dequantize_4_bit_bf16(W13_u4, w13_scale_bf16, ZP13_u4) -> W13_bf16 [OUT13,H] (NON-PERSISTENT)
 *   gemm(x[H,M], W13_bf16[OUT13,H]) -> gu[OUT13,M]
 *   slice(gu, 0:I)=gate ; slice(gu, I:2I)=up  (guid "slice" + synSliceParamsV2;
 *     the bridge's idiomatic forward-slice -- eager BuildNode rejected slice_axis)
 *   sigmoid_fwd_bf16(gate)=sg ; mult_fwd_bf16(gate,sg)=silu ; mult_fwd_bf16(silu,up)=h
 *   int32 w2_packed --reinterpret_cast--> uint4 W2 ; int32 w2_zp --reinterpret--> uint4 ZP2
 *     dequantize_4_bit_bf16(W2_u4, w2_scale_bf16, ZP2_u4) -> W2_bf16 [H,I] (NON-PERSISTENT)
 *   gemm(h[I,M], W2_bf16[H,I]) -> y[H,M]
 *   mult_fwd_bf16(y, router_weight[1,1]) -> out[H,M]  (== PyTorch [M,H])
 *
 * GEOMETRY: Synapse sizes[0]==FCD==PyTorch innermost/last dim. A PyTorch
 * *contiguous* tensor binds DIRECTLY (no transpose/repack). GLM shapes
 * H=6144,I=2048,GS=128 are parameterized (derived from the input tensors).
 *
 * Build CPU-only:  bash build.sh   (inside vllm-gaudi:next; links torch + the
 * shipped libhabana_pytorch_backend.so + the cloned bridge headers, host x86
 * g++, -D_GLIBCXX_USE_CXX11_ABI=1). RUN ON THE HPU BOX ONLY via the test.
 */

// fmt (torch bundles fmtlib v12) must be header-only here: the two compiled
// fmt::v12 symbols (assert_fail, is_printable) our TU pulls via torch's fmt
// headers are NOT exported by any shipped torch/habana .so (hidden visibility),
// so they must be inlined. The habana logging header also sets this; guard to
// avoid a -Wmacro-redefined warning. (Verified: dropping this -> 2 undefined
// fmt symbols -> the .so fails to load on the box.)
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

// --- bridge headers (cloned at tag v1.24.0-1007 under ./bridge) --------------
#include "hpu_ops/op_backend.h"      // OpBackend, NodeAttr{guid,inputs,output_attrs,params,param_size}
#include "hpu_ops/hpu_op_helper.h"   // get_guid_with_precision, stack_tensor, PARAMS_STUB
#include "hpu_ops/stack_getter.h"    // StackGetter, TensorsPair{pt_t, syn_t}
#include "backend/habana_operator.h" // KernelRegistry(), REGISTER_HPU_BACKEND, HabanaOperator
#include "backend/synapse_helpers/graph.h"

// --- frontend (eager op) -- the public, non-codegen path the bridge's own
//     user-custom-op frontend uses in the EAGER (pt2) build:
//     habana_eager/user_custom_op_frontend.cpp ->
//       habana::eager::EagerOp<...>{schema, inputs, out_shapes}.call().
//     The vllm-gaudi:next image is the pytorch-UPSTREAM/eager build: the lazy
//     plugin (libhabana_pytorch_plugin.so) is NOT loaded, so habana_lazy::LazyOp
//     and its helpers (mul_out_hpu_lazy, ...) are absent in-process and CANNOT
//     resolve at load. EagerOp is the in-process frontend here -- its non-inline
//     symbols (EagerOpBase::run/validate_inputs/m_mutex,
//     HbEagerTensorPool::get_backend_tensor) all live in the LOADED
//     libhabana_pytorch2_plugin.so and resolve at runtime (verified via nm -DC),
//     so we do NOT need to link that plugin (which would re-trigger the
//     duplicate hpu::habana_d2d_memcpy_other registration).
#include "habana_eager/ops/eager_op.h" // habana::eager::EagerOp<at::Tensor>

// --- torch + synapse params --------------------------------------------------
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"     // syn_type_* enums (DATA_TENSOR etc.), HABANA_DIM_MAX, TSize
#include "synapse_common_types.hpp"   // synGEMMParams + synSliceParamsV2 (the bridge's forward-slice param struct)
#include "perf_lib_layer_params.h"    // ns_CastKernel::ParamsV3 (the dequant params blob)

#include <algorithm>   // std::fill_n (synSliceParamsV2 array init)
#include <string>
#include <vector>

namespace sh = synapse_helpers;

namespace habana {

// =============================================================================
// Output meta: out has the SAME shape+dtype as x ([M,H] bf16). Used by both the
// frontend (to size the lazy result) and the backend (BuildNode output attr).
// =============================================================================
static OutputMetaDataVector MoeFusedExpertMeta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();      // [M, H]
  meta[0].dtype = x.scalar_type();      // bf16
  return meta;
}

// =============================================================================
// The OpBackend: ports ../fusion/fused_expert_graph.h::buildExpertRecipe() as a
// chain of OpBackend::BuildNode() calls. Each non-final BuildNode output is a
// graph-internal (non-persistent) synapse_helpers::tensor; its .get() synTensor
// feeds the next node -- that is precisely the SRAM-resident dequant->MME edge.
// =============================================================================
struct MoeFusedExpert : OpBackend {
  MoeFusedExpert(int device_id, c10::ScalarType scalar_type)
      // guid "gemm" is a harmless placeholder (real guids are set per-BuildNode
      // below); res_ids={0} marks output 0 as the op result; not out/inplace.
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedExpertMeta);
  }

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void MoeFusedExpert::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "MoeFusedExpert::AddNode");

  // ---- typed inputs (order == schema) ----
  auto x          = sg.getNextInput<TensorsPair>();  // bf16  [M, H]
  auto w13_packed = sg.getNextInput<TensorsPair>();  // int32 [H, 2I/8]
  auto w13_scale  = sg.getNextInput<TensorsPair>();  // bf16  [H/GS, 2I]
  auto w13_zp     = sg.getNextInput<TensorsPair>();  // int32 [H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<TensorsPair>();  // int32 [I, H/8]
  auto w2_scale   = sg.getNextInput<TensorsPair>();  // bf16  [I/GS, H]
  auto w2_zp      = sg.getNextInput<TensorsPair>();  // int32 [I/GS, H/8]
  auto router_w   = sg.getNextInput<TensorsPair>();  // bf16  [1, 1]

  // ---- derive shapes (PyTorch row-major; last dim == FCD) ----
  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_expert] x must be 2D [M,H]");
  const int64_t M = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();  // [H/GS, 2I]
  TORCH_CHECK(w13_scale.pt_t.dim() == 2,
              "[fused_expert] w13_scale must be 2D [H/GS, 2I]");
  const int64_t OUT13 = s13_sizes[1];                   // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_expert] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;

  // ===========================================================================
  // GEOMETRY -- CRITICAL FIX (this is what made node 6's SIF NULL-deref).
  //
  // OpBackend::BuildNode's NodeOutputAttr.sizes is a PyTorch ROW-MAJOR shape:
  // BuildNode -> GetProxyTensor(dtype,sizes) does set_sizes_contiguous(sizes)
  // (op_backend.cpp:75,888), i.e. it builds a torch tensor in PyTorch order and
  // the bridge reverses to Synapse FCD internally. The complex-GUID dequant SIF
  // runs in this SAME PyTorch-order world (its meta = GetUnpackedShape(input),
  // ValidateShapeGrouped checks input.size(0)%scale.size(0)==0 &&
  // unpacked[1]==scale.size(1); convert_from_int4_gen.cpp:161-198,200-278).
  //
  // The earlier port reused the SYNAPSE FCD-ORDER shapes from
  // ../fusion/fused_expert_graph.h (sizes[0]==FCD, transposed) -- e.g. it told
  // the dequant its W13 output was {OUT13,H}=[2I,H]. But the PyTorch-order
  // dequant output of w13_packed[H,2I/8] is [H,2I]. That transpose desynced the
  // SIF's dim arithmetic (unpacked[1] no longer == scale.size(1)) -> the cguid
  // SIF dereferenced a mis-indexed dim/ptr -> GLUE_SIF_NULL_PTR on node 6.
  //
  // FIX: build EVERY node in PyTorch row-major order, exactly mirroring the
  // trusted reference _ref_expert() (test_fused_expert_op.py:83-90) and the
  // bridge's own ConvertFromInt4 (PyTorch-order in/out) + Matmul (PyTorch-order
  // gemm self[M,K]@other[K,N]->[M,N]):
  //   W13 = convert_from_uint4(w13_packed[H,2I/8],...) -> [H, 2I]
  //   gu  = x[M,H] @ W13[H,2I]                          -> [M, 2I]
  //   gate=gu[:, :I]  up=gu[:, I:]  (slice PyTorch dim 1) -> [M, I]
  //   h   = silu(gate) * up                            -> [M, I]
  //   W2  = convert_from_uint4(w2_packed[I,H/8],...)   -> [I, H]
  //   y   = h[M,I] @ W2[I,H]                            -> [M, H]
  //   out = y * router_w[1,1]                           -> [M, H]  (== x.sizes())
  // ===========================================================================

  // reinterpret_cast int32-packed -> uint4. VERBATIM convert_from_int4_gen.cpp
  // (ReinterpretCast, :219-228): unpacked PyTorch shape = packed shape with the
  // LAST dim *8; the NodeOutputAttr.syn_data_type carries the uint4 view.
  auto reinterpret_u4 = [&](const TensorsPair& t) {
    auto unpacked = t.pt_t.sizes().vec();
    unpacked.back() *= 8;  // packed int32 last dim (cols/8) -> uint4 last dim (cols)
    return OpBackend::BuildNode(
        this, graph,
        {"reinterpret_cast",
         {t.syn_t},
         {{unpacked, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };

  // dequantize_4_bit_bf16(weight_u4, scale, zp_u4) -> bf16, PyTorch-order out.
  // VERBATIM convert_from_int4_gen.cpp non-grouped path (:230-275): inputs are
  // {unpacked_weight, scale, unpacked_zp} (zp reinterpreted to uint4 because it
  // is packed like the weight -- :237-245); out_shape = input PyTorch shape with
  // the LAST dim *8 (:248-249); params = a zero-init ns_CastKernel::ParamsV3
  // (mode stays CAST_NO_CLIP for bf16 out). The cguid's grouped/non-grouped mode
  // is inferred by its SIF from scale.size(0) vs input.size(0); both reference
  // and we pass NO group_index (the SIF does not need it -- proven by
  // test_hpu_convert_from_int4.py:211 fn(input,scale,zp,bf16) with no group_index).
  auto dequant = [&](const TensorsPair& w_packed, const TensorsPair& scale,
                     const TensorsPair& zp_packed) {
    auto w_u4  = reinterpret_u4(w_packed);
    auto zp_u4 = reinterpret_u4(zp_packed);
    std::vector<synTensor> in{w_u4[0].get(), scale.syn_t, zp_u4[0].get()};
    std::vector<int64_t> out_shape = w_packed.pt_t.sizes().vec();
    out_shape.back() *= 8;              // PyTorch-order dequant output (last dim *8)
    ns_CastKernel::ParamsV3 params{};   // zero-init (no FP8 clipping for bf16 out)
    return OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, bf16),
         std::move(in),
         {{out_shape, bf16, std::nullopt}},  // NON-PERSISTENT intermediate
         &params, sizeof(params)});
  };

  // ---- dequant #1: W13_bf16 [H, 2I] (PyTorch order; graph-internal SRAM target) ----
  auto W13 = dequant(w13_packed, w13_scale, w13_zp);

  // ---- MME gemm #1: gu[M, 2I] = gemm(x[M,H], W13[H,2I]) (contract H) ----
  // PyTorch-order gemm, exactly like the bridge Matmul (matmul.cpp:48,62,67-70):
  // synGEMMParams{false,false}; self=x[M,H], other=W13[H,2I]; out=[M,2I]. This is
  // x @ W13 (== _ref_expert: gate_up = torch.matmul(x, W13)).
  // NodeOutputAttr.sizes is a non-owning at::IntArrayRef, so every shape passed
  // to BuildNode MUST be a NAMED vector that outlives the call (a temporary
  // rvalue would dangle -> garbage geometry) -- the bridge's own pattern.
  synGEMMParams gemm_params{false, false};
  std::vector<int64_t> gu_shape{M, OUT13};
  auto gu = OpBackend::BuildNode(
      this, graph,
      {"gemm",
       {x.syn_t, W13[0].get()},
       {{gu_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- slice split along PyTorch feature dim 1 (2I): gate=gu[:, :I], up=gu[:, I:] ----
  // (== _ref_expert: gate=gate_up[:, :I]; up=gate_up[:, I:].) The EAGER
  // OpBackend::BuildNode path REJECTS the slice_axis guid + synSliceAxisParams
  // (V1); use the bridge's IDIOMATIC "slice" guid + synSliceParamsV2 (full
  // per-dim axes/starts/ends/steps, HABANA_DIM_MAX=25), filled exactly like
  // SliceBackward::AddNode (slice_backward_gen.cpp:110-130):
  //   - EVERY PyTorch dim i gets an entry; axes[i] = rank - i - 1 (TPC-reversed,
  //     == get_dim_in_tpc_order, hpu_op_helper.h:187).
  //   - the SLICED PyTorch dim (here dim 1 == the 2I feature axis) carries
  //     [begin,end) step 1; every other dim is the full range [0,size).
  // gu is PyTorch [M, 2I]; we slice dim 1. out_shape is a NAMED local (non-owning
  // IntArrayRef -- a temporary rvalue would dangle).
  const int64_t gu_rank = 2;                 // gu is PyTorch [M, 2I]
  const int64_t slice_dim = 1;               // PyTorch feature axis (2I)
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};                    // zero-init (value-init -> all 0)
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    const int64_t gu_dims[gu_rank] = {M, OUT13};
    for (int64_t i = 0; i < gu_rank; ++i) {
      sp.axes[i] = static_cast<unsigned>(gu_rank - i - 1);  // TPC-reversed axis
      if (i == slice_dim) {                                 // slice dim (2I)
        sp.starts[i] = static_cast<TSize>(begin);
        sp.ends[i] = static_cast<TSize>(end);
        sp.steps[i] = static_cast<TSize>(1);
      } else {                                              // full range
        sp.starts[i] = static_cast<TSize>(0);
        sp.ends[i] = static_cast<TSize>(gu_dims[i]);
        sp.steps[i] = static_cast<TSize>(1);
      }
    }
    std::vector<int64_t> out_shape{M, end - begin};  // PyTorch [M, I]
    return OpBackend::BuildNode(
        this, graph,
        {"slice",
         {in},
         {{out_shape, bf16, std::nullopt}},
         &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  // ---- SiLU(gate) = gate * sigmoid(gate); h = SiLU(gate) * up ----
  // sigmoid+mult decomposition (== ../fusion default; not USE_SILU_CGUID).
  // All three outputs are PyTorch [M, I]; one NAMED shared shape vector,
  // lifetime-stable across all three BuildNode calls (non-owning IntArrayRef).
  std::vector<int64_t> im_shape{M, I};
  auto sgte = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()},
       {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()},
       {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()},
       {{im_shape, bf16, std::nullopt}}});

  // ---- dequant #2: W2_bf16 [I, H] (PyTorch order; graph-internal SRAM target) ----
  auto W2 = dequant(w2_packed, w2_scale, w2_zp);

  // ---- MME gemm #2: y[M, H] = gemm(h[M,I], W2[I,H]) (contract I) ----
  // PyTorch-order gemm: self=h[M,I], other=W2[I,H], out=[M,H] (== _ref_expert:
  // torch.matmul(h, W2)). y and out are both PyTorch [M, H]; one NAMED shared
  // shape vector (non-owning IntArrayRef), function-scoped so it outlives the
  // syn_out move. [M,H] also == MoeFusedExpertMeta's declared result shape.
  std::vector<int64_t> mh_shape{M, H};
  auto y = OpBackend::BuildNode(
      this, graph,
      {"gemm",
       {h[0].get(), W2[0].get()},
       {{mh_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});

  // ---- out = y * router_weight ([1,1] broadcast over [M,H]); FINAL result ----
  // final_result_index=0 ties this node's output to the op's syn_out(0). The
  // [1,1] rw -> mult_fwd_bf16 broadcast is the PROVEN form (fused_moe_full.cpp
  // ran correctly with the raw [1,1] tensor, no explicit broadcast guid).
  auto out = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), router_w.syn_t},
       {{mh_shape, bf16, 0}}});

  syn_out(0) = std::move(out[0]);
}

// =============================================================================
// Frontend: build the eager op from (schema, inputs, out-shape) and call it.
// This is the EXACT mechanism the bridge's own user-custom-op frontend uses in
// the EAGER/pt2 build (habana_eager/user_custom_op_frontend.cpp ->
// habana::eager::EagerOp<...>{schema, inputs, out_shapes}.call()): .call()
// builds a one-node JIT graph whose schema name routes the lowering to
// MoeFusedExpert via KernelRegistry().get(opname) (see
// backend/kernel/hpu_habana_launch_op_pt.cpp), then launches it. This replaces
// the lazy LazyOp path, which is unavailable in the eager image (its plugin and
// helpers, e.g. mul_out_hpu_lazy, are not loaded in-process).
// =============================================================================
static at::Tensor fused_expert_hpu(
    const at::Tensor& x, const at::Tensor& w13_packed,
    const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed, const at::Tensor& w2_scale,
    const at::Tensor& w2_zp, const at::Tensor& router_weight) {
  // out shape == x shape ([M,H]); dtype == bf16 (set below).
  std::vector<std::vector<int64_t>> out_shapes{x.sizes().vec()};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_moe_ext::fused_expert",
      {x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
       router_weight},
      std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
  return hpu_op.call();
}

} // namespace habana

// =============================================================================
// Registration (3 parts, all in this one .so; modeled on full_gen.cpp +
// kernels_register.cpp from tag v1.24.0-1007):
//   1. TORCH_LIBRARY schema    -> makes torch.ops.hpu_moe_ext.fused_expert exist
//   2. TORCH_LIBRARY_IMPL(HPU) -> routes the call through the eager frontend
//                                 (habana::eager::EagerOp) into the HPU JIT pipeline
//   3. KernelRegistry().add    -> binds the schema name to the MoeFusedExpert
//                                 OpBackend that the lowering instantiates
// =============================================================================
TORCH_LIBRARY(hpu_moe_ext, m) {
  m.def(
      "fused_expert(Tensor x, Tensor w13_packed, Tensor w13_scale, "
      "Tensor w13_zp, Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) {
  m.impl("fused_expert", habana::fused_expert_hpu);
}

// Static initializer: register the backend factory in the bridge's KernelRegistry.
// REGISTER_HPU_BACKEND(op, class) expands to
//   add(op, [](int device_id, c10::ScalarType type){
//       return std::make_shared<class>(device_id, type); })
static const auto& MoeFusedExpertKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext::fused_expert", habana::MoeFusedExpert);

/*
 * K-EXPERT EXTENSION PATH (single-expert MVP is the goal here):
 *   This op is ONE expert. The routed K-expert MoE is delivered the same way as
 *   approach B (../fusion/fused_moe_op.py): a thin Python/ATen wrapper that does
 *   the PROVEN HPU-graph-safe gather/topk/remap, calls fused_expert per active
 *   (expert, w_k) -- the router_weight [1,1] is already folded in by the graph's
 *   final mult -- and accumulates. The per-expert recipe (this file) is the
 *   fusion + SRAM win; the K launches dominate routing math regardless.
 *   To fuse the K launches into ONE captured recipe later, add a sibling
 *   OpBackend `fused_moe` here that takes stacked [E,...] weights + topk_ids/
 *   topk_weights and emits the gather + K expert-subgraphs + weighted sum as one
 *   AddNode chain (BuildConstantTensor for literals, syn_in(i) per expert slice)
 *   -- a larger graph-construction effort, left as a follow-up.
 */
