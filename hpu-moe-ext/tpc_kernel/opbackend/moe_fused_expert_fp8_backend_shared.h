/*
 * moe_fused_expert_fp8_backend_shared.h -- the SHARED, frontend-agnostic backend
 * lowering for the fp8-COMPUTE fused MoE expert op, factored VERBATIM out of the
 * proven eager TU (moe_fused_expert_fp8_backend.cpp) so the lazy TU
 * (moe_fused_expert_fp8_backend_lazy.cpp) can reuse the EXACT same
 * `MoeFusedExpertFp8 : OpBackend` AddNode chain + `MoeFusedExpertFp8Meta`
 * byte-for-byte. Mirrors the int4 op's moe_fused_expert_backend_shared.h split.
 *
 * This header carries ONLY the backend half (the GC-captured Synapse recipe + its
 * output-meta fn + the kFp8MaxFnuz constant). It is intentionally frontend-free:
 * it does NOT pull eager_op.h or lazy_kernels.h, so either frontend TU can include
 * it and add its own (EagerOp vs LazyOp) frontend + the TORCH_LIBRARY/_IMPL +
 * KernelRegistry registration. The backend recipe is identical across both builds.
 *
 * BUILD_FP8_NODES: honored here exactly as in the eager .cpp (default 0 -> compiles
 * + falls back to the proven bf16 recipe so the .so loads and the schema exists; 1
 * -> emits the real fp8 nodes). Each frontend build passes -DBUILD_FP8_NODES=1 via
 * build_fp8.sh / build_fp8_lazy.sh.
 *
 * GROUND TRUTH: this AddNode chain + MoeFusedExpertFp8Meta + kFp8MaxFnuz are copied
 * UNCHANGED from moe_fused_expert_fp8_backend.cpp. Keep in lockstep with that file
 * if the recipe ever changes. See that file's header for the full ACCURACY CAVEAT
 * (the int4->fp8 double-quant ~6e-2 floor + the fn-vs-fnuz 2x trap + the DIRECT
 * int4->fp8 proof) and the per-node bridge provenance.
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

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "perf_lib_layer_params.h"   // ns_CastKernel::ParamsV3, ns_ConvertToFp8::ParamsV2, ns_Fp8Gemm::ParamsV2

#include <algorithm>
#include <string>
#include <vector>

// Flip to 1 once the two fp8 param structs + the in-graph amax wiring are
// confirmed on device. 0 => compiles + falls back to the proven bf16 recipe so
// the .so loads and the schema exists for harness/registration validation.
#ifndef BUILD_FP8_NODES
#define BUILD_FP8_NODES 0
#endif

namespace sh = synapse_helpers;

namespace habana {

// Gaudi-2 e4m3 FNUZ max (NOT the fn 448): the scale denominator so the cast does
// not overflow the real HW e4m3 range. Matches vllm_gaudi ops.py:30 +
// hpu_moe_ext/fp8_routed_moe.py:FP8_MAX.
static constexpr float kFp8MaxFnuz = 240.0f;

inline OutputMetaDataVector MoeFusedExpertFp8Meta(const at::Stack& stack) {
  const at::Tensor& x = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = x.sizes().vec();   // [M, H]
  meta[0].dtype = x.scalar_type();   // bf16 (the fp8 is internal; out stays bf16)
  return meta;
}

struct MoeFusedExpertFp8 : OpBackend {
  MoeFusedExpertFp8(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "gemm", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(MoeFusedExpertFp8Meta);
  }
  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

inline void MoeFusedExpertFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  StackGetter sg(this, stack, "MoeFusedExpertFp8::AddNode");

  auto x          = sg.getNextInput<TensorsPair>();  // bf16  [M, H]
  auto w13_packed = sg.getNextInput<TensorsPair>();  // int32 [H, 2I/8]
  auto w13_scale  = sg.getNextInput<TensorsPair>();  // bf16  [H/GS, 2I]
  auto w13_zp     = sg.getNextInput<TensorsPair>();  // int32 [H/GS, 2I/8]
  auto w2_packed  = sg.getNextInput<TensorsPair>();  // int32 [I, H/8]
  auto w2_scale   = sg.getNextInput<TensorsPair>();  // bf16  [I/GS, H]
  auto w2_zp      = sg.getNextInput<TensorsPair>();  // int32 [I/GS, H/8]
  auto router_w   = sg.getNextInput<TensorsPair>();  // bf16  [1, 1]
  // >>> DIRECT int4->fp8 (HPU_MOE_FP8_DIRECT) -- two OPTIONAL trailing inputs: the
  // per-expert FORWARD fp8 recovery scales s_fp8 = (|W_real|.max + eps)/240, PRECOMPUTED
  // ONCE at weight-load (the weights are static) by the driver. Each is a [1] f32 scalar
  // (per-tensor, == fp8_routed_moe._quant_weight_fp8). When PRESENT, the op folds 1/s_fp8
  // into the int4 group scale and emits ONE dequantize_4_bit straight to fp8 (no bf16
  // weight, no in-graph weight amax); the SAME s_fp8 is fp8_gemm's scaleB so the fnuz 2x
  // cancels (cast-write <-> gemm-read agree on the hf8 grid -- see the DIRECT-DEQUANT
  // PROOF below). When ABSENT (None, the default 8-arg call), the op falls back to the
  // proven in-graph-amax two-hop path. StackGetter<optional<TensorsPair>> returns {} for
  // a None slot (stack_getter.h:231-238); the eager frontend passes c10::nullopt as a
  // None IValue. NB: these are the FORWARD scale s (real ~= fp8*s), NOT 1/s.
  // (consumed unconditionally to advance the StackGetter past the 2 optional slots;
  //  only USED under BUILD_FP8_NODES -- [[maybe_unused]] silences the fallback build.)
  [[maybe_unused]] auto w13_fp8_scale = sg.getNextInput<std::optional<TensorsPair>>();  // f32 [1] or None
  [[maybe_unused]] auto w2_fp8_scale  = sg.getNextInput<std::optional<TensorsPair>>();  // f32 [1] or None

  const auto x_sizes = x.pt_t.sizes().vec();
  TORCH_CHECK(x.pt_t.dim() == 2, "[fused_expert_fp8] x must be 2D [M,H]");
  const int64_t M = x_sizes[0];
  const int64_t H = x_sizes[1];

  const auto s13_sizes = w13_scale.pt_t.sizes().vec();  // [H/GS, 2I]
  TORCH_CHECK(w13_scale.pt_t.dim() == 2,
              "[fused_expert_fp8] w13_scale must be 2D [H/GS, 2I]");
  const int64_t OUT13 = s13_sizes[1];                   // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_expert_fp8] 2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;

  const at::ScalarType bf16 = c10::ScalarType::BFloat16;
#if BUILD_FP8_NODES
  const at::ScalarType fp8 = c10::ScalarType::Float8_e4m3fn;
  const at::ScalarType f32_t = c10::ScalarType::Float;
  // DIRECT int4->fp8 is taken iff BOTH precomputed per-expert fp8 scales are present.
  const bool direct_fp8 = w13_fp8_scale.has_value() && w2_fp8_scale.has_value();
#endif

  // ---- int4 -> bf16 dequant (VERBATIM from the bf16 op; the proven out_dtype for the
  //      TWO-HOP path -- see ACCURACY CAVEAT. The DIRECT path uses dequant_fp8 below) --
  auto reinterpret_u4 = [&](const TensorsPair& t) {
    auto unpacked = t.pt_t.sizes().vec();
    unpacked.back() *= 8;
    return OpBackend::BuildNode(
        this, graph,
        {"reinterpret_cast",
         {t.syn_t},
         {{unpacked, at::kInt, std::nullopt, DATA_TENSOR, syn_type_uint4}}});
  };
  auto dequant_bf16 = [&](const TensorsPair& w_packed, const TensorsPair& scale,
                          const TensorsPair& zp_packed) {
    auto w_u4  = reinterpret_u4(w_packed);
    auto zp_u4 = reinterpret_u4(zp_packed);
    std::vector<synTensor> in{w_u4[0].get(), scale.syn_t, zp_u4[0].get()};
    std::vector<int64_t> out_shape = w_packed.pt_t.sizes().vec();
    out_shape.back() *= 8;
    ns_CastKernel::ParamsV3 params{};
    return OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, bf16),
         std::move(in),
         {{out_shape, bf16, std::nullopt}},
         &params, sizeof(params)});
  };

#if BUILD_FP8_NODES
  // ===========================================================================
  // DIRECT int4 -> fp8 dequant (HPU_MOE_FP8_DIRECT). ONE dequantize_4_bit emits
  // fp8 straight from int4 -- NO bf16 intermediate, NO in-graph weight amax/cast.
  //
  // ---- WHY THIS IS SAFE (the fn-vs-fnuz 2x is HANDLED, not avoided) ----------
  // The historical "int4->fp8 == 2x bug" was NOT about the bf16 detour per se; it
  // was that the OLD direct attempt wrote fp8 bits with NO matching FORWARD scale at
  // the gemm to recover them, so the hf8-grid read had nothing to cancel the
  // fn(bias7)-vs-fnuz(bias8) factor (see this file's ACCURACY CAVEAT + fp8_routed_moe).
  //
  // The cancellation is a property of the WRITE<->READ scale PAIR, not of bf16:
  //   * dequantize_4_bit computes the NUMERIC value V = (code - zp) * scale_in and
  //     stores it onto the SAME hardware hf8 (e4m3) grid that convert_to_fp8 writes
  //     (cast_sequence.cpp:168 maps Float8_e4m3fn -> CastType::hf8 for BOTH ops;
  //     CAST_CLIP matches convert_from_int4_gen.cpp:261-267 for fp8 out).
  //   * the int4 dequant computes W_real = (code - zp) * w_scale_int4. If we instead
  //     pass scale_in = w_scale_int4 * (1/s_fp8), it stores V = W_real / s_fp8 onto
  //     the hf8 grid -- BIT-IDENTICAL to convert_to_fp8(W_real, 1/s_fp8). Then
  //     fp8_gemm(..., scaleB = s_fp8) reads the hf8 value (~= W_real/s_fp8) and
  //     multiplies by s_fp8 -> recovers W_real. The fn-vs-fnuz factor cancels EXACTLY
  //     as in the two-hop cast<->gemm round-trip; |out|/|ref| ~= 1.0 (NOT 2x).
  //   PROVEN card-free: with s_fp8 folded + recovered, the stored fp8 bits are
  //   bit-identical to the two-hop path and y_direct == y_twohop (relNorm 0.0); the
  //   2x appears ONLY when the s_fp8 recovery is omitted.
  //
  // s_fp8 = (|W_real|.max + eps)/240 is PRECOMPUTED ONCE at weight-load by the driver
  // (the weights are static -> one reduce_max per expert at load, NOT per step) and
  // handed in as w13_fp8_scale / w2_fp8_scale [1] f32. So the DIRECT in-graph weight
  // cost is ONE f32->bf16 cast of s_fp8 + ONE div_fwd (fold) + ONE dequant->fp8 PER
  // WEIGHT -- vs the two-hop's dequant_bf16 + abs + reduce_max + cast + add + 2x div +
  // convert_to_fp8. (3 nodes vs 8 per weight; the bf16 weight tensor is gone too.)
  //
  // FOLD = scale_in[H/GS,2I] = w_scale_int4 / s_fp8 (ONE div_fwd; the [1] s_fp8 scalar
  // broadcasts over the whole group-scale tensor). The dequant's own grouped/non-grouped
  // inference is UNCHANGED (it keys on scale rows vs input rows), and scale_in keeps
  // w_scale_int4's [H/GS,2I] shape + bf16 dtype, so the dequant SIF sees the identical
  // geometry/type it sees today -- only the values are pre-divided.
  // ===========================================================================
  auto dequant_fp8 = [&](const TensorsPair& w_packed, const TensorsPair& scale,
                         const TensorsPair& zp_packed,
                         const TensorsPair& s_fp8_fwd) {
    auto w_u4  = reinterpret_u4(w_packed);
    auto zp_u4 = reinterpret_u4(zp_packed);
    // FOLD: scale_in = w_scale_int4 / s_fp8  (ONE div_fwd; the [1] s_fp8 scalar
    // broadcasts over the [H/GS, 2I] (or [I/GS, H]) group-scale). div_fwd is the
    // bridge's element divide (optimizer_adam_gen.cpp:144). The output is BF16 --
    // the SAME scale dtype the proven dequantize_4_bit ingests today, so the dequant
    // SIF sees the identical scale type/shape, only the VALUES are pre-divided. The
    // [1] s_fp8 (FORWARD scale, f32 from the driver) is cast to bf16 first so div_fwd
    // is homogeneous bf16. scale_int4/s_fp8 makes the dequant write W_real/s_fp8 onto
    // the hf8 grid (BIT-IDENTICAL to convert_to_fp8(W_real,1/s_fp8)) -- see the
    // DIRECT-DEQUANT PROOF in the header; fp8_gemm recovers with the f32 s_fp8.
    std::vector<int64_t> one{1};
    auto s_bf16 = OpBackend::BuildCast(
        this, graph, s_fp8_fwd.syn_t, one, f32_t, bf16);   // s_fp8 f32 -> bf16
    auto scale_sizes = scale.pt_t.sizes().vec();           // [H/GS, 2I]
    auto scale_in = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("div_fwd"sv, bf16),
         {scale.syn_t, s_bf16.get()},
         {{scale_sizes, bf16, std::nullopt}}});            // NON-PERSISTENT folded scale
    std::vector<synTensor> in{w_u4[0].get(), scale_in[0].get(), zp_u4[0].get()};
    std::vector<int64_t> out_shape = w_packed.pt_t.sizes().vec();
    out_shape.back() *= 8;
    // CAST_CLIP for fp8 out -- VERBATIM convert_from_int4_gen.cpp:261-267 (clamps to
    // the real e4m3 range so a high-magnitude tile cannot overflow the hf8 grid).
    ns_CastKernel::ParamsV3 params{};
    params.mode = CAST_CLIP;
    return OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("dequantize_4_bit"sv, fp8),
         std::move(in),
         {{out_shape, fp8, std::nullopt}},             // ONE node: int4 -> fp8 directly
         &params, sizeof(params)});
  };
#endif

#if BUILD_FP8_NODES
  // W13 weight-prep: DIRECT (one dequant->fp8) when scales given, else two-hop bf16.
  std::vector<sh::tensor> W13;  // bf16 [H,2I] (two-hop) -- unused on the direct path
  if (!direct_fp8) {
    W13 = dequant_bf16(w13_packed, w13_scale, w13_zp);  // [H, 2I] bf16
  }
#else
  auto W13 = dequant_bf16(w13_packed, w13_scale, w13_zp);  // [H, 2I] bf16
#endif

#if BUILD_FP8_NODES
  // ===========================================================================
  // fp8 PATH. Each of the four cast subgraphs (W13, W2, x, h) is:
  //   amax  = reduce: a [1] f32 = max(|data|)        (abs_fwd -> reduce_max_fwd)
  //   inv_s = amax_to_inv_scale(amax) = 240.0/amax   (the COMPRESS scale 1/s)
  //   s     = amax/240.0                             (the FORWARD/recover scale)
  //   fp8   = convert_to_fp8(data, inv_s)            (round_mode CAST_ROUND_HALF_NE)
  // then fp8_gemm(A_fp8, B_fp8, scaleA=s_act, scaleB=s_weight, out=bf16).
  //
  // The scalar-scale plumbing (amax -> 1/s and -> s as [1] f32 tensors) is the
  // ONE remaining device-confirmable wiring detail (BuildConstantTensor for 240,
  // a reciprocal node, a mult). It is intentionally factored into the helpers
  // below; their bodies are the spec lines the box owner fills/validates. With
  // BUILD_FP8_NODES==1 but the helpers stubbed, the op aborts LOUDLY at build so
  // there is never a silent wrong-scale path.
  // ===========================================================================
  // make_fp8(data, shape) -> {data_fp8, fwd_scale} (a 2-element vector; the
  // tensors are MOVE-returned because sh::tensor is non-default-constructible).
  // fp8_gemm(...) -> {out} (1-element).
  //
  // ============================ FILLED (the spec, realized) =================
  // The per-operand cast is the SELF-CONSISTENT round-trip the live Gaudi-2 fp8
  // linear uses (apply_fp8_linear_hpu / dynamic_quant, mirrored 1:1 in
  // hpu_moe_ext/fp8_routed_moe.py:_quant_weight_fp8/_quant_act_fp8 and
  // tpc_kernel/dequant_intrinsic.py:_bf16_to_fp8):
  //     s     = (|data|.max + 1e-8) / FP8_MAX     # FORWARD/recovery scale (real ~= fp8*s)
  //     inv_s = 1 / s                             # COMPRESS scale (real -> fp8 grid)
  //     data_fp8 = convert_to_fp8(data, inv_s)    # cast packs with 1/s
  //     ...                                       # gemm later recovers with s
  // FP8_MAX = kFp8MaxFnuz = 240 (Gaudi-2 e4m3 FNUZ max), NOT the fn 448 -- the
  // cast/gemm DTYPE is float8_e4m3fn but the HW maps fn->fnuz, so 240 keeps the
  // cast inside the real e4m3 range. The cast (1/s) and the gemm (s) agree on the
  // fnuz bits, so the 2x fn-vs-fnuz factor CANCELS (see this file's ACCURACY
  // CAVEAT). NOTHING here ever hands the int4 dequant an fp8 out_dtype.
  //
  // Node-by-node provenance (every guid/param matched to a bridge call site):
  //   abs_fwd                : all_gen.cpp:86 (guid "abs_fwd", in-dtype precision;
  //                             abs_fwd_bf16 is the registered element kernel --
  //                             confirmed in libtpc_kernels.so. NB optimizer_adamw_gen
  //                             :42 writes bare "abs" but that goes via a cguid alias;
  //                             the explicit "abs_fwd" is the directly-registered form.)
  //   reduce_max_multi_dim_fwd: optimizer_adamw_gen.cpp:53 +
  //                             ns_Reduction::ParamsV2{reductionDimensionMask=0,
  //                             keepDim=false} (reduction_template.cpp:26-37; mask=0
  //                             reduces ALL dims -> scalar [1]).
  //   div_fwd                : optimizer_adamw_gen.cpp:62 (amax_div = const/amax).
  //   convert_to_fp8         : fp8_ops.cpp:153-167 + GetCastParams:126
  //                             (ns_ConvertToFp8::ParamsV2{round_mode=CAST_ROUND_HALF_NE}).
  //
  // PRECISION: abs+reduce run in the data dtype (bf16) like optimizer_adamw; the
  // amax is then cast to f32 so the +1e-8 / div / the scale TENSORS handed to the
  // cast and the gemm are f32 -- exactly the live path, whose x_scale/w_scale are
  // .float() (fp8_routed_moe.py:_quant_*_fp8 return s.float(); the GEMM scale
  // operands are f32). The +1e-8 (added in f32) mirrors dynamic_quant's epsilon
  // and makes an all-zero tile give s>0 (no 0/0 NaN) -- a LOUD-failure guard.
  const at::ScalarType f32 = c10::ScalarType::Float;
  // make_fp8(data[2D], shape, keep_dim) -> {data_fp8, fwd_scale}.
  //
  // SCALE GRANULARITY (the accuracy lever; see this file's ACCURACY CAVEAT and the
  // DESIGN note). `keep_dim` selects the production-standard granularity:
  //   keep_dim == 1 : PER-OUTPUT-CHANNEL weight scale -> scale [1, out]. Reduce the
  //                   contraction/row axis (dim 0) only; each output COLUMN gets its
  //                   own recovery scale. This is exactly what apply_fp8_linear_hpu /
  //                   VllmMixtureOfExpertsOpFP8PerChannel pass as the weight scale
  //                   (hpu_compressed_tensors.py:301 transposes weight_scale to
  //                   [1,out] for trans_B=false; ops.py:1292 d_scale_w12).
  //   keep_dim == 0 : PER-TOKEN activation scale -> scale [M, 1]. Reduce the feature
  //                   axis (dim 1) only; each token ROW gets its own recovery scale.
  //                   This is EXACTLY dynamic_quant() default (ops.py:936, the proven
  //                   live activation cast: scale = |x|.max(dim=-1)/FP8_MAX).
  // Both the convert_to_fp8 cast and fp8_gemm broadcast a non-scalar scale along the
  // complementary axis: the per-token [M,1] cast is the proven dynamic_quant path,
  // and the per-channel [1,out] weight scale into scaleB is the proven
  // apply_fp8_linear_hpu/compressed-tensors path. (The earlier per-TENSOR scalar
  // wasted fp8's 3-mantissa range on the high-magnitude tile; per-channel/per-token
  // is the production convention and is strictly >= a single scalar.)
  //
  // reductionDimensionMask is REVERSED-bit (reduction_template.cpp:33: bit for dim d
  // == 1 << (ndims-d-1)). data is 2D, so to KEEP dim `keep_dim` we REDUCE the other
  // dim with keepDim=true (so the scale keeps a size-1 axis for clean broadcast).
  auto make_fp8 = [&](synTensor data, const std::vector<int64_t>& shape,
                      int keep_dim) -> std::vector<sh::tensor> {
    TORCH_CHECK(shape.size() == 2, "[fused_expert_fp8] make_fp8 expects 2D data");
    const int reduce_dim = (keep_dim == 1) ? 0 : 1;        // reduce the OTHER axis
    std::vector<int64_t> scale_shape = shape;
    scale_shape[reduce_dim] = 1;                           // [1,out] or [M,1]

    // 1. |data| (bf16, same shape). "abs_fwd" -> abs_fwd_bf16 (registered element
    //    kernel; all_gen.cpp:86 uses this exact form).
    auto adata = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("abs_fwd"sv, bf16),
         {data},
         {{shape, bf16, std::nullopt}}});

    // 2. amax bf16 = max over the reduced axis only (keepDim=true -> size-1 axis).
    //    mask bit = 1 << (2 - reduce_dim - 1).
    ns_Reduction::ParamsV2 rparams{};
    rparams.reductionDimensionMask =
        static_cast<unsigned>(1u << (2 - reduce_dim - 1));
    rparams.keepDim = true;
    auto amax_bf16 = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("reduce_max_multi_dim_fwd"sv, bf16),
         {adata[0].get()},
         {{scale_shape, bf16, std::nullopt}},
         &rparams, sizeof(rparams)});

    // 3. amax -> f32 (so the scale arithmetic + the scale tensors are f32, exactly
    //    like the live path's .float() scales).
    auto amax_f32 = OpBackend::BuildCast(
        this, graph, amax_bf16[0].get(), scale_shape, bf16, f32);

    // 4. amax_eps f32 = amax + 1e-8 (dynamic_quant epsilon; 0/0-NaN guard).
    //    eps/240 are size-1 constants; add_fwd/div_fwd broadcast them over the
    //    [1,out] or [M,1] vector.
    std::vector<int64_t> scalar_shape{1};
    auto eps = OpBackend::BuildConstantTensor(this, graph, at::Scalar(1e-8), f32);
    auto amax_eps = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("add_fwd"sv, f32),
         {amax_f32.get(), eps.get()},
         {{scale_shape, f32, std::nullopt}}});

    // 5. scale tensors. const240 f32 once; FORWARD s = amax_eps/240 (recovers fp8
    //    -> real at the gemm); COMPRESS inv_s = 240/amax_eps (packs real -> fp8 at
    //    the cast). div_fwd order matches amax_div (const/amax) at adamw:62.
    auto c240 = OpBackend::BuildConstantTensor(
        this, graph, at::Scalar(kFp8MaxFnuz), f32);
    auto fwd_scale = OpBackend::BuildNode(   // s = amax_eps / 240   (shape == scale_shape)
        this, graph,
        {get_guid_with_precision("div_fwd"sv, f32),
         {amax_eps[0].get(), c240.get()},
         {{scale_shape, f32, std::nullopt}}});
    auto inv_scale = OpBackend::BuildNode(   // 1/s = 240 / amax_eps (shape == scale_shape)
        this, graph,
        {get_guid_with_precision("div_fwd"sv, f32),
         {c240.get(), amax_eps[0].get()},
         {{scale_shape, f32, std::nullopt}}});

    // 6. data_fp8 = convert_to_fp8(data, inv_s). guid precision is the INPUT dtype
    //    (bf16), NOT the fp8 out (fp8_ops.cpp:153 keys on src_type); the fp8 dtype
    //    rides the output attr. ROUND_HALF_NE == GetCastParams non-stochastic. The
    //    [1,out]/[M,1] inv_scale broadcasts over the complementary axis (per-token
    //    [M,1] is the proven dynamic_quant cast).
    ns_ConvertToFp8::ParamsV2 cparams{};
    cparams.round_mode = CAST_ROUND_HALF_NE;
    auto data_fp8 = OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("convert_to_fp8"sv, bf16),
         {data, inv_scale[0].get()},
         {{shape, fp8, std::nullopt}},   // NON-PERSISTENT fp8 intermediate
         &cparams, sizeof(cparams)});

    std::vector<sh::tensor> out;
    out.reserve(2);
    out.emplace_back(std::move(data_fp8[0]));   // [0] = fp8 data
    out.emplace_back(std::move(fwd_scale[0]));  // [1] = FORWARD scale s (for fp8_gemm)
    return out;
  };
  // fp8_gemm(A_fp8, B_fp8, sA, sB) -> out[out_shape] bf16. Matches the PROVEN live
  // fp8_gemm_v2 lowering (Fp8GemmV2::AddNode, fp8_gemm_gen.cpp:286-376), which is
  // what apply_fp8_linear_hpu / fp8_routed_moe call:
  //   guid    : get_guid_with_precision("fp8_gemm", outType=bf16)   (:304)
  //   inputs  : {A, B, scaleA, scaleB, bias=nullptr, D=nullptr, FUSE_CONVERT_MME=nullptr}
  //             -- the non-accumulate V2 input vector (:306-341): A,B then the two
  //             scale tensors, then bias(null), then the D slot(null, accumulate=
  //             false), then the trailing FUSE_CONVERT_MME(null) the GC fills.
  //   params  : ns_Fp8Gemm::ParamsV2{transpose_a=false, transpose_b=false} (:343-345).
  //             is_hw_aligned/scale_method_hash_id are set ONLY when both scales are
  //             None-or-H2D (:347-362); our scales are plain in-graph DEVICE tensors,
  //             so that branch is skipped and those fields stay zero -- which is the
  //             correct/identical lowering for a plain device-tensor scale.
  //   scaleA/scaleB carry the FORWARD scales s_act/s_weight: out = (A.B) * sA * sB
  //   = (A_fp8*sA).(B_fp8*sB), recovering the real product (the *_scale_inv NAMES
  //   are misleading; the live sites pass the forward s -- fp8_routed_moe.py:62-94).
  //
  //   >>> sA/sB are now NON-SCALAR (per-token/per-channel) recovery VECTORS:
  //       sA = per-token  [M, 1]  (recovers A row m)   -- the dynamic_quant convention
  //       sB = per-channel[1, out](recovers B col o)   -- the apply_fp8_linear_hpu /
  //            compressed-tensors convention (weight_scale transposed to [1,out] for
  //            trans_B=false, hpu_compressed_tensors.py:301). The fp8_gemm complex-
  //            guid broadcasts each over its complementary axis; this is the SAME
  //            device-tensor scale slot the live fp8_gemm_v2 fills (HandleScaleTensor,
  //            fp8_ops.cpp:63-84, pushes a numel>1 scale tensor straight through).
  //            NOTE: this op builds the fp8_gemm GUID directly (like Fp8Gemm::AddNode,
  //            fp8_gemm_gen.cpp:185-224), so the scale tensor is fed as-is at input
  //            slots 2/3 -- no Fp8GemmV2 reshape is needed (the vectors already carry
  //            the correct [M,1]/[1,out] shape).
  auto fp8_gemm = [&](synTensor A_fp8, synTensor B_fp8, synTensor sA, synTensor sB,
                      const std::vector<int64_t>& out_shape,
                      bool is_result) -> std::vector<sh::tensor> {
    ns_Fp8Gemm::ParamsV2 gparams{};
    gparams.transpose_a = false;
    gparams.transpose_b = false;
    std::vector<synTensor> gin{A_fp8, B_fp8, sA, sB,
                               nullptr /*bias*/, nullptr /*D (no accumulate)*/,
                               nullptr /*FUSE_CONVERT_MME (GC fills)*/};
    return OpBackend::BuildNode(
        this, graph,
        {get_guid_with_precision("fp8_gemm"sv, bf16),
         std::move(gin),
         {{out_shape, bf16, is_result ? std::optional<int>(0) : std::nullopt}},
         &gparams, sizeof(gparams)});
  };

  std::vector<int64_t> w13_shape{H, OUT13};
  std::vector<int64_t> x_shape{M, H};
  // x is [M, H] -> PER-TOKEN activation scale [M,1] (keep_dim=0, reduce H). The
  // activation cast STAYS in-graph (dynamic per step) on BOTH paths -- the GOAL
  // only folds the static WEIGHT scale at load.
  auto x_fp8 = make_fp8(x.syn_t, x_shape, /*keep_dim=*/0);  // {fp8, [M,1]}

  // W13 weight operand for the gemm: {fp8 weight, FORWARD scaleB}.
  //   DIRECT: ONE dequant->fp8 with folded scale; scaleB = the precomputed [1] s_fp8.
  //   TWO-HOP: dequant->bf16 then make_fp8 (per-channel [1,2I] amax+cast in-graph).
  synTensor W13_fp8_data, W13_sB;
  std::vector<sh::tensor> W13_fp8_direct;   // direct fp8 weight  (kept alive for the gemm)
  std::vector<sh::tensor> W13_fp8;          // two-hop {fp8, [1,2I]} (kept alive)
  if (direct_fp8) {
    W13_fp8_direct = dequant_fp8(w13_packed, w13_scale, w13_zp,
                                 *w13_fp8_scale);         // [H,2I] fp8, ONE node
    W13_fp8_data = W13_fp8_direct[0].get();
    W13_sB       = w13_fp8_scale->syn_t;                  // [1] f32 FORWARD recovery scale
  } else {
    // W13 is [in=H, out=2I] -> PER-CHANNEL weight scale [1,2I] (keep_dim=1, reduce H).
    W13_fp8      = make_fp8(W13[0].get(), w13_shape, /*keep_dim=*/1);  // {fp8, [1,2I]}
    W13_fp8_data = W13_fp8[0].get();
    W13_sB       = W13_fp8[1].get();
  }

  std::vector<int64_t> gu_shape{M, OUT13};
  auto gu = fp8_gemm(x_fp8[0].get(), W13_fp8_data, x_fp8[1].get(),
                     W13_sB, gu_shape, /*is_result=*/false);
#else
  // ---- BUILD_FP8_NODES==0 fallback: the PROVEN bf16 gemm (so the .so loads and
  //      the schema exists; the test then runs the bf16-accurate path). ----
  synGEMMParams gemm_params{false, false};
  std::vector<int64_t> gu_shape{M, OUT13};
  auto gu = OpBackend::BuildNode(
      this, graph,
      {"gemm",
       {x.syn_t, W13[0].get()},
       {{gu_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});
#endif

  // ---- slice / silu (IDENTICAL to the bf16 op) ----
  const int64_t gu_rank = 2;
  const int64_t slice_dim = 1;
  auto slice = [&](synTensor in, int64_t begin, int64_t end) {
    synSliceParamsV2 sp{};
    std::fill_n(sp.axes, HABANA_DIM_MAX, 0u);
    std::fill_n(sp.starts, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.ends, HABANA_DIM_MAX, static_cast<TSize>(0));
    std::fill_n(sp.steps, HABANA_DIM_MAX, static_cast<TSize>(1));
    const int64_t gu_dims[gu_rank] = {M, OUT13};
    for (int64_t i = 0; i < gu_rank; ++i) {
      sp.axes[i] = static_cast<unsigned>(gu_rank - i - 1);
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
    std::vector<int64_t> out_shape{M, end - begin};
    return OpBackend::BuildNode(
        this, graph, {"slice", {in}, {{out_shape, bf16, std::nullopt}}, &sp, sizeof(sp)});
  };
  auto gate = slice(gu[0].get(), 0, I);
  auto up   = slice(gu[0].get(), I, OUT13);

  std::vector<int64_t> im_shape{M, I};
  auto sgte = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("sigmoid_fwd"sv, bf16),
       {gate[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto silu = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {gate[0].get(), sgte[0].get()}, {{im_shape, bf16, std::nullopt}}});
  auto h = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {silu[0].get(), up[0].get()}, {{im_shape, bf16, std::nullopt}}});

  std::vector<int64_t> mh_shape{M, H};
#if BUILD_FP8_NODES
  // W2 weight-prep mirrors W13: DIRECT (one dequant->fp8) when scales given, else two-hop.
  std::vector<int64_t> w2_shape{I, H};
  // h is [M, I] -> PER-TOKEN activation scale [M,1] (in-graph on BOTH paths; the
  // intermediate h is dynamic and is the largest single error source either way).
  auto h_fp8 = make_fp8(h[0].get(), im_shape, /*keep_dim=*/0);   // {fp8, [M,1]}

  synTensor W2_fp8_data, W2_sB;
  std::vector<sh::tensor> W2;              // two-hop bf16 [I,H] (outer scope, like W13)
  std::vector<sh::tensor> W2_fp8_direct;   // direct fp8 weight (kept alive for the gemm)
  std::vector<sh::tensor> W2_fp8;          // two-hop {fp8, [1,H]} (kept alive)
  if (direct_fp8) {
    W2_fp8_direct = dequant_fp8(w2_packed, w2_scale, w2_zp,
                                *w2_fp8_scale);         // [I,H] fp8, ONE node
    W2_fp8_data = W2_fp8_direct[0].get();
    W2_sB       = w2_fp8_scale->syn_t;                  // [1] f32 FORWARD recovery scale
  } else {
    // W2 is [in=I, out=H] -> PER-CHANNEL weight scale [1,H] (keep_dim=1, reduce I).
    W2          = dequant_bf16(w2_packed, w2_scale, w2_zp);   // [I, H] bf16
    W2_fp8      = make_fp8(W2[0].get(), w2_shape, /*keep_dim=*/1);   // {fp8, [1,H]}
    W2_fp8_data = W2_fp8[0].get();
    W2_sB       = W2_fp8[1].get();
  }
  auto y = fp8_gemm(h_fp8[0].get(), W2_fp8_data, h_fp8[1].get(),
                    W2_sB, mh_shape, /*is_result=*/false);
#else
  auto W2 = dequant_bf16(w2_packed, w2_scale, w2_zp);   // [I, H] bf16
  auto y = OpBackend::BuildNode(
      this, graph,
      {"gemm",
       {h[0].get(), W2[0].get()},
       {{mh_shape, bf16, std::nullopt}},
       &gemm_params, sizeof(gemm_params)});
#endif

  // ---- out = y * router_weight (IDENTICAL to the bf16 op; FINAL result) ----
  auto out = OpBackend::BuildNode(
      this, graph,
      {get_guid_with_precision("mult_fwd"sv, bf16),
       {y[0].get(), router_w.syn_t},
       {{mh_shape, bf16, 0}}});

  syn_out(0) = std::move(out[0]);
}

} // namespace habana
