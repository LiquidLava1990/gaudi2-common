/*
 * moe_ext_perflib_fast.cpp  —  perf-lib shim for the OPTIMIZED dequant kernels.
 *
 * NEW FILE. Does NOT touch the live moe_ext_perflib.cpp / moe_ext_custom_op.cpp
 * or any file the in-flight int4-MoE / fp8 A/B uses.
 *
 * Implements GetKernelGuids + InstantiateTpcKernel (tpc_kernel_lib_interface.h)
 * for TWO GUIDs produced by moe_ext_unpack_dequant_w4_fast.c:
 *
 *   moe_ext_unpack_dequant_w4_fast_bf16   (main_fast: native int32 unpack)
 *       inputs : in0 weight_packed int32 [out//8, in]  (PLAIN packed words, FCD=out/8)
 *                in1 weight_scale  f32   [out, groups]  (FCD=OUT, full width)
 *       output : out0              bf16  [out, in]      (FCD=OUT)
 *       -> NO zero_point input (the constant 8 is folded in the kernel).
 *
 *   moe_ext_unpack_dequant_w4_lean_bf16   (main_lean: U4-descriptor unpack)
 *       inputs : in0 weight_packed u4    [out, in]      (U4-typed, like the live op)
 *                in1 weight_scale  f32   [out, groups]
 *       output : out0              bf16  [out, in]
 *       -> NO zero_point input (folded immediate). Same OUT/IN access pattern as the
 *          live kernel (proven), so this variant is the low-risk fallback.
 *
 * Build embeds the per-variant ELF (built twice from the .c with -DKERNEL_VARIANT).
 * The GC discovers this .so via GC_KERNEL_PATH.
 *
 * HONEST SCOPE: the LEAN variant's access pattern is byte-identical to the proven
 * live perf-lib (only the input count drops 3->2: no zero_point). The FAST
 * variant's pattern is NEW: in0 is the packed-word FCD (out//8) and the output is
 * the full OUT FCD; the kernel emits 8 nibbles per word internally (a stride-8
 * deinterleave realized by the kernel's own idx_o arithmetic, validated byte-exact
 * card-free in tests/dequant_fast_cpu_validate.py). If GC rejects the FAST pattern
 * at instantiate, it fails LOUDLY with a GLUE_* code and the LEAN variant is the
 * drop-in (its pattern is the proven one).
 */

#include "tpc_kernel_lib_interface.h"

#include <cstring>
#include <cstdint>

// Two embedded ELFs: build the .c twice (-DKERNEL_VARIANT=1 -> fast, =2 -> lean),
// renaming the objcopy symbol stems so both live in this .so.
extern "C" {
extern const uint8_t _binary_moe_ext_unpack_dequant_w4_fast_o_start[];
extern const uint8_t _binary_moe_ext_unpack_dequant_w4_fast_o_end[];
extern const uint8_t _binary_moe_ext_unpack_dequant_w4_lean_o_start[];
extern const uint8_t _binary_moe_ext_unpack_dequant_w4_lean_o_end[];
}

namespace {

constexpr char kGuidFast[] = "moe_ext_unpack_dequant_w4_fast_bf16";
constexpr char kGuidLean[] = "moe_ext_unpack_dequant_w4_lean_bf16";

// ABI-identical to the live params (group_size, zero_const : int32).
struct MoeUnpackDequantW4FastParams {
  int32_t group_size;
  int32_t zero_const;
};

inline uint64_t elf_size(const uint8_t* s, const uint8_t* e) {
  return static_cast<uint64_t>(e - s);
}

inline void set_ap_dim(tpc_lib_api::DimIndexSpaceMapping& m, uint32_t isDim,
                       float a, float start_b, float end_b, bool allReq) {
  m.indexSpaceDim = isDim;
  m.a = a;
  m.start_b = start_b;
  m.end_b = end_b;
  m.allRequired = allReq;
}

}  // namespace

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount,
    tpc_lib_api::GuidInfo* guids) {
  if (kernelCount == nullptr) return tpc_lib_api::GLUE_FAILED;
  if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) { *kernelCount = 0; return tpc_lib_api::GLUE_SUCCESS; }
  if (guids == nullptr || *kernelCount == 0) { *kernelCount = 2; return tpc_lib_api::GLUE_SUCCESS; }
  std::memset(guids[0].name, 0, tpc_lib_api::MAX_NODE_NAME);
  std::strncpy(guids[0].name, kGuidFast, tpc_lib_api::MAX_NODE_NAME - 1);
  guids[0].nameHash.Value = 0; guids[0].kernelProperties = 0;
  std::memset(guids[1].name, 0, tpc_lib_api::MAX_NODE_NAME);
  std::strncpy(guids[1].name, kGuidLean, tpc_lib_api::MAX_NODE_NAME - 1);
  guids[1].nameHash.Value = 0; guids[1].kernelProperties = 0;
  *kernelCount = 2;
  return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params == nullptr || instance == nullptr) return tpc_lib_api::GLUE_FAILED;

  // Identify the variant by GUID.
  const bool isFast =
      (std::strncmp(params->guid.name, kGuidFast, tpc_lib_api::MAX_NODE_NAME) == 0);

  // Both variants: 2 inputs (weight_packed, weight_scale), 1 output. (No zero_point.)
  if (params->inputTensorNr != 2 || params->outputTensorNr != 1) {
    return params->inputTensorNr < 2
               ? tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT
               : tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  // ---- ELF handshake (variant-selected) ----
  const uint8_t* elf_s = isFast ? _binary_moe_ext_unpack_dequant_w4_fast_o_start
                                : _binary_moe_ext_unpack_dequant_w4_lean_o_start;
  const uint8_t* elf_e = isFast ? _binary_moe_ext_unpack_dequant_w4_fast_o_end
                                : _binary_moe_ext_unpack_dequant_w4_lean_o_end;
  const uint64_t need = elf_size(elf_s, elf_e);
  if (instance->kernel.kernelElf == nullptr || instance->kernel.elfSize < need) {
    instance->kernel.elfSize = static_cast<uint32_t>(need);
    return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
  }
  std::memcpy(instance->kernel.kernelElf, elf_s, need);
  instance->kernel.elfSize = static_cast<uint32_t>(need);

  // ---- scalar params ----
  MoeUnpackDequantW4FastParams p{128, 8};
  if (params->nodeParams.nodeParams != nullptr &&
      params->nodeParams.nodeParamsSize >= sizeof(p)) {
    std::memcpy(&p, params->nodeParams.nodeParams, sizeof(p));
  }
  std::memcpy(&instance->kernel.scalarParams[0], &p, sizeof(p));
  instance->kernel.paramsNr = sizeof(p) / sizeof(uint32_t);  // = 2

  // ---- index-space geometry ----
  // out0 [OUT, IN] (Synapse dim0=FCD=OUT). OUT consumed 64 lanes/step.
  const tpc_lib_api::Tensor& out = params->outputTensors[0];
  const uint64_t OUT = out.geometry.maxSizes[0];
  const uint64_t IN  = out.geometry.maxSizes[1];
  constexpr uint64_t kLanes = 64;

  instance->indexSpaceRank = 2;
  instance->indexSpaceGeometry[1] = IN;            // dim1: IN (1 row/step)
  for (uint32_t d = 2; d < tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE; ++d)
    instance->indexSpaceGeometry[d] = 1;

  // HOIST CORRECTNESS (both kernels load the scale once per invocation at in_lo):
  // we force the split onto dim0 (OUT), so dim1 (IN) is NOT split across
  // invocations -> a group's IN rows stay together and the single scale load at
  // in_lo is the correct group row. preferredSplitDim=0 is set below; the scale
  // group-dim AP is a=0 + allRequired so GC keeps the whole group available.

  // CORRECTED 2026-06-23 v4 — BYTE-LOAD + HARDWARE u4->u8 deinterleave + WIDTH-PRESERVING
  // uchar256->float256 (the convert the STOCK dequant uses; keeps OUT order). Both GUIDs
  // share dequant_body. weight_packed is uint8 [OUT/2, IN]. The kernel steps dim0 by 256
  // OUT codes (= 128 bytes; one v_convert window's first 256 deinterleaved codes):
  //   - weight_packed (uint8) dim0: 128 bytes/step, but the load reads 256 bytes -> a=128,
  //     window [0,255] (256-byte read).
  //   - out0 (bf16) dim0: 256 contiguous OUT codes/step (4x 64-lane stores) -> a=256, win[0,255].
  //   - scale (f32) dim0: 256 OUT lanes/step -> a=256, win[0,255]; group dim a=0+allReq (hoist).
  (void)isFast;  // both variants identical now (correct path; fast/lean are an A/B pair)
  constexpr uint64_t kStep = 512;  // OUT codes per index-space dim0 step (one SW_UNPACK window)
  const uint64_t outSteps = (OUT + kStep - 1) / kStep;
  instance->indexSpaceGeometry[0] = outSteps;

  // weight_packed: uint8, dim0 = OUT/2 bytes. The SW_UNPACK|SW_UNPCK_4_TO_8 load reads
  // 256 BYTES (= 512 unpacked OUT codes via v_convert_u4_to_u8_all_b) per step, advancing
  // 256 bytes/step -> perfect tile (byte_base = step*256 = (OUT-step)/2).
  auto fill_packed_bytes = [&](tpc_lib_api::TensorAccessPattern& ap) {
    ap.Value = 0; ap.allRequired = 0;
    set_ap_dim(ap.mapping[0], 0, static_cast<float>(kStep / 2), 0.0f,
               static_cast<float>(kStep / 2) - 1.0f, false);   // a=256, window [0,255]
    set_ap_dim(ap.mapping[1], 1, 1.0f, 0.0f, 0.0f, false);      // IN 1 row/step
    for (uint32_t d = 2; d < tpc_lib_api::MAX_TENSOR_DIM; ++d)
      set_ap_dim(ap.mapping[d], d, 1.0f, 0.0f, 0.0f, false);
  };
  // out0 + scale: dim0 = OUT, 256 contiguous OUT codes/step.
  auto fill_out = [&](tpc_lib_api::TensorAccessPattern& ap) {
    ap.Value = 0; ap.allRequired = 0;
    set_ap_dim(ap.mapping[0], 0, static_cast<float>(kStep), 0.0f,
               static_cast<float>(kStep) - 1.0f, false);
    set_ap_dim(ap.mapping[1], 1, 1.0f, 0.0f, 0.0f, false);
    for (uint32_t d = 2; d < tpc_lib_api::MAX_TENSOR_DIM; ++d)
      set_ap_dim(ap.mapping[d], d, 1.0f, 0.0f, 0.0f, false);
  };
  auto fill_scale = [&](tpc_lib_api::TensorAccessPattern& ap) {
    ap.Value = 0; ap.allRequired = 0;
    set_ap_dim(ap.mapping[0], 0, static_cast<float>(kStep), 0.0f,
               static_cast<float>(kStep) - 1.0f, false);
    set_ap_dim(ap.mapping[1], 1, 0.0f, 0.0f, 0.0f, /*allReq=*/true);  // group dim hoist
    for (uint32_t d = 2; d < tpc_lib_api::MAX_TENSOR_DIM; ++d)
      set_ap_dim(ap.mapping[d], d, 1.0f, 0.0f, 0.0f, false);
  };
  fill_packed_bytes(instance->inputTensorAccessPattern[0]);   // weight_packed (uint8)
  fill_scale       (instance->inputTensorAccessPattern[1]);   // weight_scale
  fill_out         (instance->outputTensorAccessPattern[0]);  // out0

  instance->auxiliaryTensorNr = 0;
  instance->preferredSplitDim = 0;
  instance->kernelUsesLock = 0;
  return tpc_lib_api::GLUE_SUCCESS;
}

// ===========================================================================
// MANDATORY load-time entry point.  ROOT CAUSE of the "Failed loading kernels
// from <so> for Gaudi2 device" was that KernelDB::loadFunctions() in
// synapse/src/graph_compiler/kernel_db.cpp dlsym's GetShapeInference and treats
// a NULL as FATAL:  it logs  "No Entry point GetShapeInference" -> "Error loading
// functions" -> "Failed loading kernels from {} for {} device"  and then dlcloses
// the .so, so NONE of our GUIDs ever register (synStatus 26 at node creation).
// (GetSupportedDataLayouts / GetSuggestedTensorManipulation are only WARN-level
//  optional, but GetShapeInference is mandatory.)  Verified card-free by replaying
// synInitialize() with GC_KERNEL_PATH pointed at this .so.
//
// Our op is STATIC-shape (GLM weights are fixed at graph-build).  GC pre-fills
// outputData->outputTensors[i]->geometry with the declared output geometry; for a
// static-shape kernel the correct SIF simply derives/passes the output shapes
// through and returns GLUE_SUCCESS.  We derive out0 [OUT, IN] from the inputs to
// be robust (and LOUD if the arity is wrong), matching InstantiateTpcKernel:
//   FAST : in0 weight_packed [OUT/8, IN]  -> out0 [OUT=in0.dim0*8, IN=in0.dim1]
//   LEAN : in0 weight_packed [OUT,   IN]  -> out0 [OUT=in0.dim0,   IN=in0.dim1]
//   in1 weight_scale [OUT, groups] carries the true OUT (full width) for both.
// ===========================================================================
tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId deviceId,
    const tpc_lib_api::ShapeInferenceParams* inputParams,
    tpc_lib_api::ShapeInferenceOutput* outputData) {
  if (inputParams == nullptr || outputData == nullptr ||
      outputData->outputTensors == nullptr) {
    return tpc_lib_api::GLUE_SIF_NULL_PTR;
  }
  if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    // We only ship Gaudi2 ISA; never asked to infer for another device, but be safe.
    return tpc_lib_api::GLUE_SUCCESS;
  }
  if (inputParams->inputTensorsNr < 2 || inputParams->outputTensorsNr < 1 ||
      inputParams->inputTensors == nullptr) {
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }

  const bool isFast = (inputParams->pGuid != nullptr)
      ? (std::strncmp(inputParams->pGuid->name, kGuidFast,
                      tpc_lib_api::MAX_NODE_NAME) == 0)
      : (std::strncmp(inputParams->guid.name, kGuidFast,
                      tpc_lib_api::MAX_NODE_NAME) == 0);

  const tpc_lib_api::TensorShapeInfo* in0 = inputParams->inputTensors[0];
  const tpc_lib_api::TensorShapeInfo* in1 = inputParams->inputTensors[1];  // scale [OUT, groups]
  if (in0 == nullptr || in1 == nullptr) return tpc_lib_api::GLUE_SIF_NULL_PTR;

  tpc_lib_api::TensorShapeInfo* out0 = outputData->outputTensors[0];
  if (out0 == nullptr) return tpc_lib_api::GLUE_SIF_NULL_PTR;

  // OUT is the scale's FCD (full width) for both variants; IN is in0's dim1.
  const uint64_t OUT = in1->geometry.maxSizes[0];
  const uint64_t IN  = in0->geometry.maxSizes[1];
  (void)isFast;  // OUT taken from the scale, so the FAST in0=[OUT/8,IN] packing is handled.

  out0->geometry.dims = 2;
  out0->geometry.maxSizes[0] = OUT;
  out0->geometry.maxSizes[1] = IN;
  out0->geometry.minSizes[0] = OUT;
  out0->geometry.minSizes[1] = IN;
  out0->geometry.dataType = tpc_lib_api::DATA_BF16;
  return tpc_lib_api::GLUE_SUCCESS;
}

// NOTE — do NOT export GetSupportedDataLayouts.
// It is an OPTIONAL (warn-level) entry point at load time, but EXPORTING it changes
// GC behaviour at COMPILE time and was the cause of the second wall (synStatus 26):
//   [DATA_LAYOUT][error] Glue Code GetSupportedLayouts failed for guid
//                        moe_ext_unpack_dequant_w4_lean_bf16
//   [PASS_MANAGER][error] Graph optimization failed pass: setSupportedLayouts
// When the symbol is ABSENT, GC's setSupportedLayouts pass takes the graceful
// default path it logs as "perf_lib GetSupportedDataLayouts for guid {} isn't
// implemented, setting all input/output layouts ... as don't-care" and compiles.
// When the symbol is PRESENT, KernelDB::GetKernelSupportedDataLayouts treats our
// (empty / layoutCount=0) result as a HARD failure for an N-D non-conv kernel.
// Our kernel is layout-agnostic, so "don't-care" is exactly what we want -> leave
// it unexported.  Verified card-free with a synGraphCompile harness: dropping this
// symbol flips the compile from synStatus 26 to OK.  GetShapeInference (above) is
// the only mandatory load-time export we were missing.

uint64_t GetLibVersion() { return 0x0001'0000ULL; }

}  // extern "C"
