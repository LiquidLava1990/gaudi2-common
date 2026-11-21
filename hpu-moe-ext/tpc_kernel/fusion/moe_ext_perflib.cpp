/*
 * moe_ext_perflib.cpp  —  P0 deliverable: a real TPC perf-lib shim.
 *
 * Implements the two mandatory perf-lib entry points from
 *   /usr/include/habanalabs/tpc_kernel_lib_interface.h  (namespace tpc_lib_api):
 *     GetKernelGuids        (KERNEL_GUIDS_ENTRY_POINT_NAME)
 *     InstantiateTpcKernel  (KERNEL_INSTANTIATION_ENTRY_POINT_NAME)
 * plus the optional GetLibVersion, and exposes ONE custom GUID:
 *     moe_ext_unpack_dequant_w4_bf16
 * whose Gaudi-2 ISA is the ELF produced by tpc-clang from
 *   ../moe_ext_unpack_dequant_w4.c  (verified to compile rc=0, see ../build.sh).
 *
 * The compiled .o is embedded into THIS .so at build time via llvm-objcopy
 * (-I binary), exporting the symbols
 *     _binary_moe_ext_unpack_dequant_w4_o_start / _end
 * which we hand to GC verbatim as the kernel ELF in InstantiateTpcKernel.
 *
 * The GC discovers this .so through  GC_KERNEL_PATH=.../libmoe_ext_perflib.so .
 * On the first HPU op that lowers GUID "moe_ext_unpack_dequant_w4_bf16", GC calls
 * GetKernelGuids (to learn the GUID is ours) then InstantiateTpcKernel (to fetch
 * the ELF + index-space geometry + per-tensor access patterns).
 *
 * SCOPE / HONEST NOTE
 * -------------------
 * This is the "ship our OWN compiled int4-dequant kernel" foundation. It is a
 * fully-formed, COMPILING perf-lib that GC can load. It is the harder, riskier of
 * the two P0 paths because InstantiateTpcKernel must describe the kernel's
 * index-space + access pattern EXACTLY as the .c kernel walks memory, and that is
 * validated blind (no HPU here). The kernel body (moe_ext_unpack_dequant_w4.c)
 * iterates dim0=OUT in 64-f32-lane (256B) steps and dim1=IN one row per step;
 * the access patterns below mirror that. If GC rejects a pattern at instantiate
 * time it fails LOUDLY with a GLUE_* code (see test output) and the geometry here
 * is the thing to adjust.
 *
 * The SAFER, primary runnable P0 path (and the one the P1 fused graph reuses) binds
 * the RESIDENT, already-cost-modeled GUID
 *   grouped_index_transpose_dequant_2d_with_zp_fwd_u4
 * (see ../dequant_guid_op.cpp + this dir's profile/test). Both ship; this shim
 * proves the build+register+discover pipeline for a user-authored ELF end to end.
 *
 * Build: ./build_perflib.sh   (CPU-only docker run; host g++, NO tpc-clang here —
 *        the TPC .o is built by ../build.sh / build_perflib.sh drives both).
 */

#include "tpc_kernel_lib_interface.h"

#include <cstring>
#include <cstdint>

// ---- embedded TPC ELF (llvm-objcopy -I binary) -----------------------------
// These symbols are emitted by objcopy from moe_ext_unpack_dequant_w4.o. The
// build script renames the input object so the symbol stem is stable.
extern "C" {
extern const uint8_t _binary_moe_ext_unpack_dequant_w4_o_start[];
extern const uint8_t _binary_moe_ext_unpack_dequant_w4_o_end[];
}

namespace {

constexpr char kGuid[] = "moe_ext_unpack_dequant_w4_bf16";

// MUST byte-match struct MoeUnpackDequantW4Params in the .c kernel AND the
// FillParams in moe_ext_custom_op.cpp (group_size, zero_const : int32).
struct MoeUnpackDequantW4Params {
  int32_t group_size;
  int32_t zero_const;
};

inline uint64_t elf_size() {
  return static_cast<uint64_t>(_binary_moe_ext_unpack_dequant_w4_o_end -
                               _binary_moe_ext_unpack_dequant_w4_o_start);
}

// Identity-ish access-pattern helper: tensor dim d advances 1:1 with index-space
// dim d, covering [0, size_d). a=1, start_b=0, end_b=1 means element e of dim d
// maps to index-space coordinate e (one element per step on that index dim).
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

// ---- GetKernelGuids --------------------------------------------------------
// Report our single GUID for Gaudi2. Other devices: report 0 (we only ship g2).
tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId deviceId,
    uint32_t* kernelCount,
    tpc_lib_api::GuidInfo* guids) {
  if (kernelCount == nullptr) {
    return tpc_lib_api::GLUE_FAILED;
  }
  if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    *kernelCount = 0;
    return tpc_lib_api::GLUE_SUCCESS;
  }
  // Probe call: GC passes *kernelCount==0 to ask how many we have.
  if (guids == nullptr || *kernelCount == 0) {
    *kernelCount = 1;
    return tpc_lib_api::GLUE_SUCCESS;
  }
  std::memset(guids[0].name, 0, tpc_lib_api::MAX_NODE_NAME);
  std::strncpy(guids[0].name, kGuid, tpc_lib_api::MAX_NODE_NAME - 1);
  guids[0].nameHash.Value = 0;          // optional accelerator hash; 0 = unset
  guids[0].kernelProperties = 0;        // no dynamic shape / no 64b sizes
  *kernelCount = 1;
  return tpc_lib_api::GLUE_SUCCESS;
}

// ---- InstantiateTpcKernel --------------------------------------------------
// Describe to GC how to integrate one node of our GUID into a recipe:
//   * copy the embedded ELF into the GC-provided buffer (size-handshake first),
//   * declare the index-space geometry,
//   * declare per-tensor access patterns,
//   * forward the scalar params.
//
// Tensor order (matches moe_ext_custom_op.cpp / the .c main signature):
//   in0 weight_packed : u4  [out, in]   FCD=OUT  (logical; packed int32 on disk)
//   in1 weight_scale  : f32 [out, groups]
//   in2 zero_point    : f32 [out, groups]
//   out0              : bf16[out, in]   FCD=OUT
//
// Synapse FCD == dim0 == "fastest changing" == OUT here. The kernel processes
// OUT in 64-lane chunks (one float64 vector = 64 f32 = 256B) and IN one per step.
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    const tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params == nullptr || instance == nullptr) {
    return tpc_lib_api::GLUE_FAILED;
  }
  if (params->inputTensorNr != 3 || params->outputTensorNr != 1) {
    return params->inputTensorNr < 3
               ? tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT
               : tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  // ---- 1. ELF handshake --------------------------------------------------
  const uint64_t need = elf_size();
  if (instance->kernel.kernelElf == nullptr ||
      instance->kernel.elfSize < need) {
    // GC calls once with a too-small / null buffer to learn the size, then again
    // with a buffer of (at least) elfSize bytes. Report the needed size.
    instance->kernel.elfSize = static_cast<uint32_t>(need);
    return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
  }
  std::memcpy(instance->kernel.kernelElf,
              _binary_moe_ext_unpack_dequant_w4_o_start, need);
  instance->kernel.elfSize = static_cast<uint32_t>(need);

  // ---- 2. scalar params (group_size, zero_const) -------------------------
  // GC forwarded the UserParams blob from synNodeCreate / the PT2 FillParams.
  // Pack them into DeviceKernel.scalarParams for the TPC program prologue.
  MoeUnpackDequantW4Params p{128, 8};
  if (params->nodeParams.nodeParams != nullptr &&
      params->nodeParams.nodeParamsSize >= sizeof(MoeUnpackDequantW4Params)) {
    std::memcpy(&p, params->nodeParams.nodeParams,
                sizeof(MoeUnpackDequantW4Params));
  }
  std::memcpy(&instance->kernel.scalarParams[0], &p, sizeof(p));
  instance->kernel.paramsNr = sizeof(p) / sizeof(uint32_t);  // = 2

  // ---- 3. index-space geometry -------------------------------------------
  // out0 geometry: maxSizes[0]=OUT, maxSizes[1]=IN  (Synapse: dim0 is FCD).
  const tpc_lib_api::Tensor& out = params->outputTensors[0];
  const uint64_t OUT = out.geometry.maxSizes[0];
  const uint64_t IN  = out.geometry.maxSizes[1];

  // OUT is consumed 64 lanes / index step (float64 vector). Round UP so a ragged
  // tail still gets an invocation (the kernel guards with size[] anyway).
  constexpr uint64_t kLanes = 64;
  const uint64_t outSteps = (OUT + kLanes - 1) / kLanes;

  instance->indexSpaceRank = 2;
  instance->indexSpaceGeometry[0] = outSteps;   // dim0: OUT / 64
  instance->indexSpaceGeometry[1] = IN;         // dim1: IN (1 row / step)
  for (uint32_t d = 2; d < tpc_lib_api::MAX_INDEX_SPACE_DIM_SIZE; ++d) {
    instance->indexSpaceGeometry[d] = 1;
  }

  // ---- 4. access patterns -------------------------------------------------
  // GC pre-allocates the AP arrays (inputTensorAccessPattern[inputTensorNr],
  // outputTensorAccessPattern[outputTensorNr]); we fill them.
  //
  // weight_packed (in0) and out0: dim0(OUT) strides 64 elems per index step,
  // dim1(IN) strides 1 row per index step.
  auto fill_w_or_out = [&](tpc_lib_api::TensorAccessPattern& ap) {
    ap.Value = 0;
    ap.allRequired = 0;
    // dim0 OUT: a=64 (64 elements per index-space step on dim0).
    set_ap_dim(ap.mapping[0], /*isDim=*/0, /*a=*/static_cast<float>(kLanes),
               /*start_b=*/0.0f, /*end_b=*/static_cast<float>(kLanes) - 1.0f,
               /*allReq=*/false);
    // dim1 IN: a=1 (one row per step).
    set_ap_dim(ap.mapping[1], /*isDim=*/1, 1.0f, 0.0f, 0.0f, false);
    for (uint32_t d = 2; d < tpc_lib_api::MAX_TENSOR_DIM; ++d) {
      set_ap_dim(ap.mapping[d], d, 1.0f, 0.0f, 0.0f, false);
    }
  };

  // scale/zero (in1/in2): [OUT, groups]. dim0 OUT strides 64/step like the weight;
  // dim1 is the GROUP row = IN/group_size. We let the kernel address the group via
  // the same dim1 index but with a=1/group_size so step `in` selects group in/gs.
  // (allRequired on the group dim is the conservative correctness choice — it tells
  //  GC every group row may be touched across the IN sweep.)
  auto fill_group = [&](tpc_lib_api::TensorAccessPattern& ap) {
    ap.Value = 0;
    ap.allRequired = 0;
    set_ap_dim(ap.mapping[0], 0, static_cast<float>(kLanes), 0.0f,
               static_cast<float>(kLanes) - 1.0f, false);
    // group dim: mark allRequired so GC does not slice it away (the scale/zero
    // group selection is data-dependent on `in`); a=0 broadcast within a group.
    set_ap_dim(ap.mapping[1], 1, 0.0f, 0.0f, 0.0f, /*allReq=*/true);
    for (uint32_t d = 2; d < tpc_lib_api::MAX_TENSOR_DIM; ++d) {
      set_ap_dim(ap.mapping[d], d, 1.0f, 0.0f, 0.0f, false);
    }
  };

  fill_w_or_out(instance->inputTensorAccessPattern[0]);   // weight_packed
  fill_group(instance->inputTensorAccessPattern[1]);      // weight_scale
  fill_group(instance->inputTensorAccessPattern[2]);      // zero_point
  fill_w_or_out(instance->outputTensorAccessPattern[0]);  // out0

  // ---- 5. misc ------------------------------------------------------------
  instance->auxiliaryTensorNr = 0;
  instance->preferredSplitDim = 0;   // let GC pick the split dim
  instance->kernelUsesLock = 0;

  return tpc_lib_api::GLUE_SUCCESS;
}

// ---- GetLibVersion (optional) ----------------------------------------------
uint64_t GetLibVersion() { return 0x0001'0000ULL; }  // 1.0

}  // extern "C"
