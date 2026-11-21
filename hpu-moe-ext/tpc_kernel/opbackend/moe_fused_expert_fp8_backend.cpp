/*
 * moe_fused_expert_fp8_backend.cpp -- fp8-COMPUTE variant of the PROVEN fused
 * int4 MoE OpBackend (moe_fused_expert_backend.cpp). Registers
 *   torch.ops.hpu_moe_ext_fp8.fused_expert_fp8(x, w13_packed, w13_scale, w13_zp,
 *                                          w2_packed, w2_scale, w2_zp, router_weight)
 * (separate library namespace so it can coexist in-process with the bf16 op's
 * `hpu_moe_ext`; same schema/inputs/output as the bf16 op) -- but the in-graph node
 * sequence dequants int4 -> bf16 -> **fp8** and runs the two expert GEMMs on the
 * Gaudi-2 MME in **fp8** (native, ~2x the bf16 MME rate).
 *
 * >>> SCAFFOLD STATUS <<<
 * This is a PRECISE, COMPILE-READY scaffold built against the same version-exact
 * bridge (tag v1.24.0-1007) as the bf16 op, reusing the bridge's OWN fp8 node
 * constructions verbatim:
 *   - convert_to_fp8  (hpu_ops/backend/fp8_ops.cpp:153 CastToFp8::AddNode)
 *   - fp8_gemm        (hpu_ops/backend/fp8_gemm_gen.cpp:206 Fp8Gemm::AddNode)
 * The node topology, geometry, and the int4 dequant are IDENTICAL to the proven
 * bf16 op; the ONLY changes are (a) a per-tensor bf16->fp8 cast after each dequant
 * and on each GEMM's activation input, and (b) `gemm` -> `fp8_gemm` with FORWARD
 * recovery scales. It is gated behind a build flag so it does NOT perturb the
 * proven bf16 .so. It MUST pass an on-device accuracy gate at a LOOSER tolerance
 * (int4->fp8 is double quantization; see the ACCURACY CAVEAT block) before use.
 *
 * Build CPU-only (separate .so so the bf16 op is untouched):
 *   docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work \
 *     -w /work/opbackend vllm-gaudi:next bash build_fp8.sh
 *   -> build/libmoe_fused_expert_fp8.so
 * (build_fp8.sh == build.sh with SRC=this file, OUT=libmoe_fused_expert_fp8.so,
 *  and -DFUSED_FP8=1. The bridge link/symbol logic is identical.)
 *
 * RUN ON THE HPU BOX ONLY (test_fused_expert_fp8_op.py -- looser tol).
 *
 * =============================================================================
 * ACCURACY CAVEAT -- int4 -> fp8 IS DOUBLE QUANTIZATION + the fn-vs-fnuz 2x TRAP
 * =============================================================================
 * 1. DOUBLE QUANT: the weight is quantized to int4 (the checkpoint) and then the
 *    dequantized real weight is re-quantized to fp8 e4m3 (3 mantissa bits). fp8
 *    e4m3 has ~2 decimal digits; stacked on the int4 error AND applied to TWO
 *    weights + TWO activations across the gate||up->silu->down pipeline, the relNorm
 *    vs the bf16-intrinsic reference is a HARD FLOOR of ~6.4e-2 -- coherent, but NOT
 *    within the 1e-2 the bf16 op passes. MEASURED: on-device 6.43e-2, matching a
 *    faithful CPU emulation of this exact op (6.2-6.6e-2 across seeds, GLM shape).
 *    The on-box gate uses a LOOSER tol of 8e-2 (floor + margin; see test).
 *
 *    >>> SCALE GRANULARITY IS NOT THE FIX (investigated + measured). The op now uses
 *    the production PER-CHANNEL weight + PER-TOKEN activation scales (was per-tensor
 *    scalar). That is the CORRECT convention (apply_fp8_linear_hpu /
 *    VllmMixtureOfExpertsOpFP8PerChannel) and never regresses, but it only improves
 *    the floor ~2-3% (6.44 -> 6.28e-2 mean). Per-GROUP(128) block scaling does NOT
 *    break 5e-2 through the full pipeline either: in the GPTQ int4 layout the
 *    weight's dynamic range lives along the CONTRACTION/group axis (per-output-column
 *    range is only ~1.2x), so per-output-channel scaling is largely orthogonal to it,
 *    and the binding constraint is fp8's 3 mantissa bits + the intermediate-h requant,
 *    not the scale. fp8-compute TRADES this few-% for half the weight HBM traffic +
 *    ~2x MME. (Mirrors the sibling hpu_moe_ext/fp8_routed_moe.py double-quant note.)
 *
 * 2. fn-vs-fnuz 2x TRAP (this project's history; the root cause of the prior
 *    "fp8_norm ~= 2.06 * ref_norm" failure): the Gaudi-2 HW fp8 element is
 *    float8_e4m3FNUZ (exponent bias 8, max 240); the dtype name handed to the ops
 *    is float8_e4m3FN (bias 7, max 448). For the SAME stored e4m3 bits the two
 *    interpretations differ by exactly 2^(7-8) = 2x. Therefore:
 *      - The TWO-HOP path's dequant stays bf16 out (VERBATIM the bf16 op's
 *        `dequantize_4_bit_bf16`) and the ONLY fp8 cast is a SELF-CONSISTENT
 *        round-trip: convert_to_fp8 packs with 1/s, fp8_gemm recovers with s. Both
 *        agree on the fnuz bits, so the 2x CANCELS -- the proven apply_fp8_linear_hpu
 *        / dynamic_quant path.
 *      - FP8_MAX for the scale is the Gaudi-2 fnuz max 240.0 (NOT 448), even though
 *        the cast/gemm DTYPE is float8_e4m3fn -- the HW maps fn->fnuz and 240 keeps
 *        the cast inside the real e4m3 range (matches vllm_gaudi ops.py:30 +
 *        fp8_routed_moe.py:FP8_MAX).
 *
 *    >>> DIRECT int4->fp8 (HPU_MOE_FP8_DIRECT) -- WHY HANDING THE DEQUANT AN fp8
 *        OUT_DTYPE IS SAFE *HERE* (and was the "2x bug" before). The old prohibition
 *        ("NEVER hand the int4 dequant an fp8 out_dtype") was an over-broad
 *        statement of the real rule. The 2x bug was NOT caused by skipping bf16; it
 *        was caused by writing fp8 bits with NO matching FORWARD scale to recover
 *        them at the gemm. The fn-vs-fnuz cancellation is a property of the
 *        WRITE<->READ scale PAIR, not the bf16 detour:
 *          - dequantize_4_bit computes V = (code - zp) * scale_in and stores V onto
 *            the SAME hardware hf8 (e4m3) grid convert_to_fp8 writes (BOTH map
 *            Float8_e4m3fn -> CastType::hf8, cast_sequence.cpp:168; CAST_CLIP matches
 *            convert_from_int4_gen.cpp:261-267). There is no separate "fn write" --
 *            the stored object is the hf8 grid point either way.
 *          - The DIRECT path FOLDS the precomputed forward scale into the dequant:
 *            scale_in = scale_int4 * (1/s_fp8), so V = W_real/s_fp8 -- BIT-IDENTICAL
 *            to convert_to_fp8(W_real, 1/s_fp8). Then fp8_gemm(scaleB = s_fp8) reads
 *            the hf8 value (~= W_real/s_fp8) and multiplies by s_fp8 -> recovers
 *            W_real. The 2x cancels EXACTLY as in the two-hop round-trip;
 *            |out|/|ref| ~= 1.0. (Card-free proof: stored fp8 bits bit-identical to
 *            the two-hop, y_direct == y_twohop @ relNorm 0.0; the 2x reappears ONLY
 *            when s_fp8 recovery is dropped.)
 *          - s_fp8 = (|W_real|.max + eps)/240 is PRECOMPUTED ONCE AT WEIGHT-LOAD by
 *            the driver (the weights are static) and passed as the two OPTIONAL
 *            trailing inputs w13_fp8_scale / w2_fp8_scale [1] f32. So the DIRECT path
 *            has NO bf16 weight, NO in-graph weight amax/cast -- just reciprocal +
 *            mult (fold) + ONE dequant->fp8. When the scales are ABSENT (None, the
 *            default 8-arg call), the op uses the TWO-HOP in-graph-amax path above.
 *
 * TWO-HOP in-graph scale (when no precomputed s_fp8): s = |W|.max/240 is COMPUTED
 * from the dequantized bf16 weight (a reduce_max node) so it is a real recovery
 * scale; an SRAM-resident captured recipe cannot take a host-side scalar per replay.
 * The DIRECT path removes that reduce_max by precomputing s_fp8 at load.
 *
 * =============================================================================
 * NODE SEQUENCE (deltas from the bf16 op are marked >>>; everything else is the
 * IDENTICAL proven topology, PyTorch row-major, geometry derived from inputs):
 * =============================================================================
 *   SCALE GRANULARITY: PER-CHANNEL weight + PER-TOKEN activation (the production
 *   apply_fp8_linear_hpu / VllmMixtureOfExpertsOpFP8PerChannel convention), NOT a
 *   single per-tensor scalar. A weight scale is reduce_max over the CONTRACTION axis
 *   only -> [1,out] (one recovery scale per output column); an activation scale is
 *   reduce_max over the FEATURE axis only -> [M,1] (one per token row, == the proven
 *   dynamic_quant). See make_fp8 + the ACCURACY CAVEAT for why this is correct but
 *   does NOT break the ~6e-2 int4->fp8 double-quant floor.
 *
 *   x[M,H] bf16 ----------------------------------------- (input)
 *   (1a) W13_bf16 = dequantize_4_bit_bf16(reint_u4(w13_packed), w13_scale,
 *                       reint_u4(w13_zp)) -> [H, 2I] bf16   (UNCHANGED; NON-PERSISTENT)
 *   >>> (1b) s_w13 = reduce_max(|W13_bf16|, dim=0)/240  -> [1,2I] f32 PER-CHANNEL fwd scale
 *   >>> (1c) W13_fp8 = convert_to_fp8(W13_bf16, 1/s_w13) -> [H, 2I] float8_e4m3fn
 *   >>> (Xa) s_x = reduce_max(|x|, dim=-1)/240          -> [M,1] f32 PER-TOKEN act scale
 *   >>> (Xb) x_fp8 = convert_to_fp8(x, 1/s_x)          -> [M, H] float8_e4m3fn
 *   >>> (2)  gu = fp8_gemm(x_fp8[M,H], W13_fp8[H,2I], scaleA=s_x[M,1], scaleB=s_w13[1,2I],
 *                          out_dtype=bf16) -> [M, 2I] bf16   (was `gemm`)
 *   (3)  gate = gu[:, :I] ; up = gu[:, I:]               (UNCHANGED slice)
 *   (4)  silu = gate*sigmoid(gate) ; h = silu*up -> [M, I] bf16  (UNCHANGED)
 *   (5a) W2_bf16 = dequantize_4_bit_bf16(reint_u4(w2_packed), w2_scale,
 *                       reint_u4(w2_zp)) -> [I, H] bf16    (UNCHANGED; NON-PERSISTENT)
 *   >>> (5b) s_w2 = reduce_max(|W2_bf16|, dim=0)/240     -> [1,H] f32 PER-CHANNEL
 *   >>> (5c) W2_fp8 = convert_to_fp8(W2_bf16, 1/s_w2)   -> [I, H] float8_e4m3fn
 *   >>> (Ha) s_h = reduce_max(|h|, dim=-1)/240          -> [M,1] f32 PER-TOKEN
 *   >>> (Hb) h_fp8 = convert_to_fp8(h, 1/s_h)           -> [M, I] float8_e4m3fn
 *   >>> (6)  y = fp8_gemm(h_fp8[M,I], W2_fp8[I,H], scaleA=s_h[M,1], scaleB=s_w2[1,H],
 *                         out_dtype=bf16) -> [M, H] bf16    (was `gemm`)
 *   (7)  out = y * router_weight[1,1] -> [M, H] bf16     (UNCHANGED; FINAL result)
 *
 * The two W*_fp8 (and x_fp8/h_fp8) are graph-internal NON-PERSISTENT tensors, so
 * the GC still bundles dequant->cast->MME -- the SRAM-residency win is preserved
 * (the fp8 weight is HALF the bf16 bytes, so the win is, if anything, larger).
 *
 * EXACT BRIDGE NODE CONSTRUCTIONS (so this is precise, not hand-waved):
 *  - convert_to_fp8 : guid get_guid_with_precision("convert_to_fp8", BFloat16);
 *      inputs {data_syn, scale_syn}; out {shape, Float8_e4m3fn, idx};
 *      params ns_ConvertToFp8::ParamsV2{ round_mode = CAST_ROUND_HALF_NE }.
 *      (fp8_ops.cpp:153 + GetCastParams:126. scale here = the INVERSE 1/s.)
 *  - reduce_max amax : guid get_guid_with_precision("reduce_max_fwd", Float)
 *      on |data| (abs_fwd) -> a [1] f32, then mult by const 1/240 (the FORWARD
 *      scale s). Equivalently use the cast's built-in amax output (CastToFp8 with
 *      a non-empty amax out gives the per-tensor amax for free -- fp8_ops.cpp:162),
 *      then 1/s = 240/amax. Either is one captured subgraph.
 *  - fp8_gemm : guid get_guid_with_precision("fp8_gemm", BFloat16);
 *      inputs {A_fp8, B_fp8, scaleA, scaleB, bias=nullptr, (no D), nullptr};
 *      out {outShape, BFloat16, 0}; params ns_Fp8Gemm::ParamsV2{transpose_a=false,
 *      transpose_b=false, ...}.  (fp8_gemm_gen.cpp:206-223. scaleA/scaleB = the
 *      FORWARD recovery scales s_x/s_w -- real ~= fp8*s.)
 *
 * WHY THIS FILE IS A SCAFFOLD (not yet a built .so): the bf16 op took a
 * multi-iteration on-device bring-up (the geometry/SIF fixes documented in the
 * bf16 file's header). The fp8 path adds 8 nodes whose param structs
 * (ns_ConvertToFp8::ParamsV2, ns_Fp8Gemm::ParamsV2) and the in-graph amax/reduce
 * wiring need the SAME on-device validation loop, which is the box owner's run.
 * Everything the build needs is here and in build_fp8.sh; flip BUILD_FP8_NODES
 * to 1 (and supply the two param structs from the bridge's perf_lib headers) to
 * emit the fp8 nodes. With BUILD_FP8_NODES==0 the op compiles and FALLS BACK to
 * the proven bf16 recipe (so the .so always loads and the schema always exists),
 * letting the test harness + registration be validated before the fp8 nodes are
 * turned on. The fp8 AddNode body below is written out in full as the spec.
 */

// === REFACTOR (int4-op-style eager/lazy/shared split) ========================
// The backend lowering (MoeFusedExpertFp8 : OpBackend + its AddNode chain incl.
// the BUILD_FP8_NODES fp8 nodes / DIRECT int4->fp8 path), the MoeFusedExpertFp8Meta
// output-meta fn, and the kFp8MaxFnuz constant USED to live inline here. They are
// now factored VERBATIM into moe_fused_expert_fp8_backend_shared.h so the LAZY
// frontend TU (moe_fused_expert_fp8_backend_lazy.cpp) can reuse the EXACT same
// recipe byte-for-byte -- mirroring the int4 op's eager/lazy/shared split. This
// eager TU keeps ONLY the EAGER frontend + the registration. The shared header
// pulls FMT_HEADER_ONLY, the bridge backend headers, the synapse params, and
// honors BUILD_FP8_NODES (default 0 -> bf16 fallback; build_fp8.sh passes
// -DBUILD_FP8_NODES=1). The recipe is identical across the eager + lazy .so.
#include "moe_fused_expert_fp8_backend_shared.h"

// --- frontend (eager op) -- the public, non-codegen path the bridge's own
//     user-custom-op frontend uses in the EAGER (pt2) build. EagerOp's non-inline
//     frontend symbols (EagerOpBase::run/validate_inputs, HbEagerTensorPool::
//     get_backend_tensor) live in the loaded EAGER plugin libhabana_pytorch2_plugin
//     .so and resolve at runtime -- so this eager .so loads ONLY under the eager
//     build (NOT under PT_HPU_LAZY_MODE=1; the lazy serve uses the _lazy .so).
#include "habana_eager/ops/eager_op.h" // habana::eager::EagerOp<at::Tensor>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

namespace habana {

static at::Tensor fused_expert_fp8_hpu(
    const at::Tensor& x, const at::Tensor& w13_packed,
    const at::Tensor& w13_scale, const at::Tensor& w13_zp,
    const at::Tensor& w2_packed, const at::Tensor& w2_scale,
    const at::Tensor& w2_zp, const at::Tensor& router_weight,
    const std::optional<at::Tensor>& w13_fp8_scale,
    const std::optional<at::Tensor>& w2_fp8_scale) {
  std::vector<std::vector<int64_t>> out_shapes{x.sizes().vec()};
  // Separate library namespace `hpu_moe_ext_fp8` (NOT `hpu_moe_ext`): if this
  // fp8 .so and the bf16 .so are loaded in the SAME process, two
  // TORCH_LIBRARY(hpu_moe_ext,...) blocks would collide (duplicate library
  // registration abort). The opname here MUST equal the registered qualified
  // name so the eager frontend routes to MoeFusedExpertFp8 via KernelRegistry.
  //
  // The two trailing fp8 scales are OPTIONAL. An absent (nullopt) optional builds a
  // None c10::IValue, which the backend StackGetter reads as {} (stack_getter.h:231-238)
  // -> the proven two-hop in-graph-amax path. When the driver precomputes them at load
  // (HPU_MOE_FP8_DIRECT), they are real [1] f32 tensors -> the DIRECT one-dequant->fp8
  // path. Built as IValues explicitly so a None slot is preserved through EagerOp's
  // m_inputs (std::vector<at::IValue>; eager_op.h:117).
  std::vector<at::IValue> inputs{
      x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp, router_weight,
      w13_fp8_scale.has_value() ? at::IValue(*w13_fp8_scale) : at::IValue(),
      w2_fp8_scale.has_value()  ? at::IValue(*w2_fp8_scale)  : at::IValue()};
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu_moe_ext_fp8::fused_expert_fp8",
      std::move(inputs),
      std::move(out_shapes)};
  hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
  return hpu_op.call();
}

} // namespace habana

TORCH_LIBRARY(hpu_moe_ext_fp8, m) {
  // The two trailing fp8 recovery scales are OPTIONAL (default None): absent => the
  // proven two-hop int4->bf16->fp8 path; present (driver precomputes s_fp8 at load)
  // => DIRECT int4->fp8 dequant (HPU_MOE_FP8_DIRECT). Backward-compatible: existing
  // 8-arg callers (default-off path) are byte-identical.
  m.def(
      "fused_expert_fp8(Tensor x, Tensor w13_packed, Tensor w13_scale, "
      "Tensor w13_zp, Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, "
      "Tensor router_weight, Tensor? w13_fp8_scale=None, "
      "Tensor? w2_fp8_scale=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext_fp8, HPU, m) {
  m.impl("fused_expert_fp8", habana::fused_expert_fp8_hpu);
}

static const auto& MoeFusedExpertFp8KernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu_moe_ext_fp8::fused_expert_fp8", habana::MoeFusedExpertFp8);
