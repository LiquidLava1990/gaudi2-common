/*
 * fused_moe_recipe.cpp  —  P1 deliverable: the make-or-break FUSED graph.
 *
 *  (Synapse geometry below is [FCD=sizes[0], outer]; FCD == PyTorch innermost/last dim.)
 *
 *   int4 packed weight (HBM, persistent, int32 storage)   x activations (HBM, persistent)
 *            |                                                       |
 *            v                                                       |
 *   [ reinterpret_cast ]  int32 [OUT/8, IN]  ->  uint4 [OUT, IN]     |
 *            |   (storage-only view; FCD expands 8x, no compute)     |
 *            v                                                       |
 *   [ dequantize_4_bit_bf16 ]  (uint4 W, bf16 scale, i32 packed zp)  |
 *            |                                                       |
 *            v                                                       v
 *      bf16 W tile  --->  (NON-PERSISTENT intermediate; NO section assigned) ---.
 *      [OUT, IN]                                            x = [IN, M] ------->|
 *                                          [ MME gemm node ]  guid = "gemm"  <--'
 *                                                   |   out = x @ W (contract IN)
 *                                                   v
 *                                   out = [OUT, M]  (HBM, persistent)
 *
 * WHY THIS IS THE EXPERIMENT
 * --------------------------
 * The bf16 dequanted weight is a graph-internal tensor with NO persistent section.
 * Whether GC keeps it in SRAM (fused, near-max BW: only ~int4 bytes touch HBM) or
 * spills it to an HBM workspace (fusion failed: ~bf16 bytes touch HBM) is decided
 * by GC's pipeline-management / SRAM-slicer passes. From libSynapse.so we verified
 * the exact machinery that CAN do this:
 *    - pass  fuseDequantIntoMoeGemm  (src/graph_compiler/fuse_dequant_into_moe_gemm.cpp)
 *      "Fuse dequantization nodes to mme gemm ... for MOE"
 *    - SRAMSlicingManager::addTpcProducer / MantaRayBundlizer::
 *      generateMultiMmeWithTpcProducersBundles / setTpcBundleSlicingChunk
 *    - allocateTensorInSram + SlicingStrategy::setInputIsInSRAM
 *    - GCFG_SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED  (contraction-dim tiling)
 * This program BUILDS that graph explicitly and COMPILES it; the GC then either
 * bundles+slices it into SRAM or doesn't. We make the verdict observable two ways:
 *    (A) the GC post-graph + SRAM-slicer logs/visualization (compile-time), and
 *    (B) on device: synProfiler HBM byte counters around synLaunch (run-time).
 *
 * THE DEQUANT NODE — REAL GRAPH GUID (decoded on-device 2026-06-08)
 * ----------------------------------------------------------------
 * The OLD code used guid "grouped_index_transpose_dequant_2d_with_zp_fwd_u4" and got
 * synNodeCreate synStatus 26 ("TPC kernel with guid ... doesn't exist"). That string
 * is a low-level KERNEL ELF name in libtpc_kernels.so (KERNELINFO; note its
 * _o_start/_o_size/_o_end glue-symbol decorations) — it is NOT a graph-node GUID that
 * synNodeCreate accepts.
 *
 * The graph-node GUID that torch.ops.hpu.convert_from_uint4 actually emits is the
 * COMPLEX GUID "dequantize_4_bit" (output-dtype suffix appended -> "dequantize_4_bit_bf16"),
 * handled by the ComplexGuid lib (COMPLEXGUID_LIB 1.24.0), PRECEDED by a "reinterpret_cast"
 * node on the int32-packed input. Decoded from libhabana_pytorch_backend.so:
 *   habana::ConvertFromInt4::AddNode  (hpu_ops/backend/convert_from_int4_gen.cpp)
 *   - reads inputs via StackGetter: input, scale, zero_point?, out_dtype, group_index?,
 *     disable_fp8_clipping(kw).
 *   - AddNode's lambda emits a "reinterpret_cast" node (OpBackend::BuildNode +
 *     GetUnpackedShape) that reinterprets the int32 packed tensor to the UNPACKED int4
 *     geometry, then validates shapes (ValidateShapeGrouped / ValidateShapeNonGrouped)
 *     and builds the dequant node whose guid = get_guid_with_precision("dequantize_4_bit",
 *     out_dtype) = "dequantize_4_bit_bf16".
 * At the Synapse C-API level (this file) PyTorch's int32->uint4 reinterpret is replaced
 * by declaring the packed weight DIRECTLY as syn_type_uint4 and inserting an explicit
 * "reinterpret_cast" node int32[IN,OUT/8] -> uint4[IN,OUT] (mirrors the on-device graph
 * 1:1 so the fuseDequantIntoMoeGemm pattern matcher sees the identical node sequence).
 *
 * TENSOR CONTRACT (verified against PRODUCTION callers — all 3 agree):
 *   hpu_moe_ext/moe_wna16_hpu.py:578   convert_from_uint4(w13p[k], s13[k], w13z, dt, None)
 *   reference/extension_ops.py:1449    MoeWNA16Matmul.get_dequant_weight
 *   tpc_kernel/fusion/profile_fusion.py:329 / test_p1_fusion.py:58
 * with the ATen schema:
 *   hpu::convert_from_uint4(Tensor input, Tensor scale, Tensor? zero_point,
 *                           ScalarType out_dtype, Tensor? group_index=None, *,
 *                           bool disable_fp8_clipping=False) -> Tensor
 *   input  (packed weight) : int32  [IN, OUT/8]   (8 u4 nibbles per int32 along OUT)
 *   scale                  : bf16   [GROUPS, OUT]  (GROUPS = IN/GS)
 *   zero_point             : int32  [GROUPS, OUT/8] PACKED symmetric zero (uint4 mid=8)
 *   out_dtype              : bf16
 *   group_index            : None   (uniform groups; not the desc_act path)
 *   OUTPUT (dequant W)     : bf16   [IN, OUT]
 * then  out = x[M,IN] @ W[IN,OUT] = [M,OUT]   (plain matmul; W is operand B).
 *
 * SYNAPSE GEOMETRY MAPPING: Synapse sizes[0] is the FCD (innermost/CONTIGUOUS dim) ==
 * PyTorch's LAST dim, so each PyTorch [a, b] (b innermost) becomes Synapse [b, a]:
 *   input [IN,OUT/8] -> [OUT/8, IN] | scale [GROUPS,OUT] -> [OUT, GROUPS]
 *   zp [GROUPS,OUT/8] -> [OUT/8, GROUPS] | W [IN,OUT] -> [OUT, IN] | x [M,IN] -> [IN, M]
 *   out [M,OUT] -> [OUT, M]. This is why the make2d() calls below look "transposed"
 *   vs the PyTorch shapes — they are the SAME tensors in FCD order. Decoded from
 *   habana::GetUnpackedShape: reinterpret expands the INNERMOST dim (== FCD) 8x.
 *
 * !!! WHY THE None-ZERO-POINT PROBE FAILED (GLUE_SIF_NULL_PTR) + THE FIX
 * The probe convert_from_uint4(inp[256,64]i32, scale[2,512]bf16, None, bf16, None) hit
 * GLUE_SIF_NULL_PTR in shape-inference. The ATen schema marks zero_point Tensor? (optional),
 * but for the GROUPED dequant (GROUPS>1, here GROUPS=2) the complex-guid SIF requires a
 * real zero_point tensor; with None it receives a null tensor handle and crashes in SIF.
 * PRODUCTION NEVER passes None — it always passes the packed symmetric zero (w13z from
 * _sym_zero, shape [GROUPS, OUT/8]). FIX: ALWAYS provide the packed zp tensor (we build a
 * persistent zp input here and the verify script passes the same). The backend validates
 * "The zero_point tensor shape must match the packed input shape, unpacked output shape,
 * or be equal in all dimensions except one which divides the output dimension" — the packed
 * [GROUPS, OUT/8] shape satisfies the first clause (rows==GROUPS divide IN-rows; cols==OUT/8
 * divides OUT). Its dtype (int32) must match the input dtype (the int32 packed weight).
 *
 * No custom perf-lib / GC_KERNEL_PATH / gaudi2.bc is needed: reinterpret_cast is a core
 * Synapse graph node and dequantize_4_bit_bf16 is the stock ComplexGuid op. The failing
 * shim build/libmoe_ext_perflib.so ([KERNEL_DB] Failed loading kernels) is DROPPED.
 *
 * TILING / SRAM BUDGET
 * --------------------
 * GLM w13 bf16 = [IN=6144, OUT=4096] * 2B = 50.3 MB > Gaudi2 SRAM (48 MB usable).
 * So a *full* expert weight cannot live in SRAM at once — it MUST be tiled along
 * the contraction dim `IN`. We do NOT hand-tile here: we build the full (untiled)
 * producer->consumer edge and let GC's GEMM_COMMON_DIM slicer choose the chunk
 * (setTpcBundleSlicingChunk). That is the supported path and it is exactly what we
 * are testing — that GC slices the bundle so each bf16 chunk fits SRAM and never
 * lands in HBM. The chosen chunk + SRAM budget are printed by the SRAM_SLICE log.
 *
 * SHAPES ARE GENERAL. Pass on argv:  M in out group_size  (defaults = GLM w13).
 *
 * Build: ./build_recipe.sh  (CPU-only docker; host g++ links libSynapse.so).
 * Run  : ON THE HPU BOX ONLY — ./build/fused_moe_recipe [M in out gs]
 *        (it acquires a device + launches; the human runs it, never this agent).
 *        Compile-only dry run (no device) is possible with FUSED_COMPILE_ONLY=1.
 *
 * If the on-device compile complains the dequant node REQUIRES a params struct, build
 * with -DDEQUANT_CAST_PARAMS=1 to pass a zero-initialized ns_CastKernel::ParamsV3
 * (round_mode=HALF_NE, seed=0, mode=NO_CLIP_NO_SAT). The default (no params) mirrors what
 * the PyTorch backend's BuildNode passes (it sets NodeAttr flags, not a CastKernel blob;
 * the ComplexGuid expander supplies the defaults).
 */

#include "synapse_api.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"  // synGEMMParams

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
#define SYN_CHECK(expr)                                                        \
  do {                                                                         \
    synStatus _s = (expr);                                                     \
    if (_s != synSuccess) {                                                    \
      std::fprintf(stderr, "[FATAL] %s -> synStatus=%d (line %d)\n", #expr,    \
                   (int)_s, __LINE__);                                         \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

namespace {

// REAL graph-node guids (decoded on-device; see file header).
//   reinterpret_cast       : core Synapse node (graph_compiler/habana_nodes/
//                            reinterpret_cast_node.cpp; NodeFactory type name
//                            "reinterpret_cast"). int32[IN,OUT/8] -> uint4[IN,OUT].
//   dequantize_4_bit_bf16  : ComplexGuid op = "dequantize_4_bit" + bf16 output suffix
//                            (habana::get_guid_with_precision). The exact guid string
//                            torch.ops.hpu.convert_from_uint4 logs on device.
constexpr char kReinterpretGuid[] = "reinterpret_cast";
constexpr char kDequantGuid[]     = "dequantize_4_bit_bf16";
constexpr char kGemmGuid[]        = "gemm";

// GEMM orientation. out = x @ W. In Synapse FCD geometry the dequant emits
// W_bf16 = [OUT, IN] (FCD=OUT) and x = [IN, M] (FCD=IN). Synapse MME gemm for 2D
// operands stored [FCD=dim0, dim1], A=x, B=W:
//   transpose_a=false : A treated as [K, M]  -> K = A.dim0 = IN, M = A.dim1
//   transpose_b=false : B treated as [N, K]  -> N = B.dim0 = OUT, K = B.dim1 = IN
//   contraction K = IN matches (A.dim0 == B.dim1); output = [N=OUT, M].
// => out[OUT,M] = sum_IN W[OUT,IN] * x[IN,M], the same values as production
//    torch.matmul(x[M,IN], W[IN,OUT]) = [M,OUT], in Synapse's [OUT,M] FCD layout.
// (No transpose needed: the packed-dim-is-FCD convention makes W land as [OUT,IN],
// the natural MME weight operand.) If on-device numerics come out transposed, flip
// these flags — test_p1_fusion.py localizes which one.
constexpr bool kTransposeA = false;
constexpr bool kTransposeB = false;

// Create a tensor handle, set 2D geometry {d0(FCD), d1}, device dtype, and either
// assign a fresh persistent section (persistent=true -> bound at launch) or leave
// section-less (persistent=false -> graph-internal; GC may place it in SRAM).
synTensor make2d(synGraphHandle g, const char* name, uint64_t d0, uint64_t d1,
                 synDataType dtype, bool persistent,
                 std::vector<synSectionHandle>* sections) {
  synTensor t = nullptr;
  SYN_CHECK(synTensorHandleCreate(&t, g, DATA_TENSOR, name));

  synTensorGeometry geo;
  std::memset(&geo, 0, sizeof(geo));
  geo.dims = 2;
  geo.sizes[0] = d0;  // FCD
  geo.sizes[1] = d1;
  SYN_CHECK(synTensorSetGeometry(t, &geo, synGeometrySizes));
  SYN_CHECK(synTensorSetDeviceDataType(t, dtype));

  if (persistent) {
    synSectionHandle sec = nullptr;
    SYN_CHECK(synSectionCreate(&sec, 0, g));
    SYN_CHECK(synSectionSetPersistent(sec, true));
    SYN_CHECK(synTensorAssignToSection(t, sec, 0));
    sections->push_back(sec);
  }
  // persistent==false: NO section => internal tensor; GC owns placement (SRAM or
  // HBM workspace). This is the residency we are testing.
  return t;
}

size_t bytesOfElt(synDataType d) {
  switch (d) {
    case syn_type_int32:
    case syn_type_uint32:
    case syn_type_single: return 4;
    case syn_type_bf16:
    case syn_type_fp16:
    case syn_type_int16:  return 2;
    // 4-bit types are sub-byte; callers that need the U4 physical footprint must
    // compute IN*OUT/2 directly (see wPacked_b), not via this per-element helper.
    case syn_type_int4:
    case syn_type_uint4:  return 1;  // not byte-addressable; returns 1 as a guard
    default:              return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  // ---- shapes (GENERAL; default GLM w13 = gate||up) ----------------------
  uint64_t M  = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 32;     // tokens
  uint64_t IN = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 6144;   // hidden (contraction)
  uint64_t OUT= (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 4096;   // 2*moe_int
  uint64_t GS = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 128;    // group size
  const bool compileOnly = std::getenv("FUSED_COMPILE_ONLY") != nullptr;

  if (IN % GS != 0 || OUT % 8 != 0) {
    std::fprintf(stderr, "[FATAL] need IN%%GS==0 and OUT%%8==0 (IN=%llu GS=%llu OUT=%llu)\n",
                 (unsigned long long)IN, (unsigned long long)GS, (unsigned long long)OUT);
    return 2;
  }
  const uint64_t GROUPS = IN / GS;
  const uint64_t OUT_PK = OUT / 8;  // int32-packed u4 along OUT

  std::printf("=== fused_moe_recipe ===\n");
  std::printf("  M=%llu  IN=%llu  OUT=%llu  GS=%llu  GROUPS=%llu  OUT_PK=%llu\n",
              (unsigned long long)M, (unsigned long long)IN, (unsigned long long)OUT,
              (unsigned long long)GS, (unsigned long long)GROUPS, (unsigned long long)OUT_PK);
  const double bf16_w_mb = (double)IN * OUT * 2 / (1024.0 * 1024.0);
  const double int4_w_mb = (double)IN * OUT / 2 / (1024.0 * 1024.0);
  std::printf("  W bf16 = %.1f MB (must NOT round-trip HBM)   W int4 = %.1f MB (the HBM target)\n",
              bf16_w_mb, int4_w_mb);
  std::printf("  dequant guid = \"%s\"  (preceded by \"%s\")\n", kDequantGuid, kReinterpretGuid);
  std::printf("  compileOnly=%d\n", (int)compileOnly);

  SYN_CHECK(synInitialize());

  // ---- device (skip acquire in compile-only mode) ------------------------
  synDeviceId dev = 0;
  bool haveDev = false;
  if (!compileOnly) {
    synStatus s = synDeviceAcquire(&dev, nullptr);
    if (s != synSuccess) {
      std::fprintf(stderr,
                   "[WARN] synDeviceAcquire failed (synStatus=%d). Falling back to "
                   "COMPILE-ONLY (graph build + compile; no launch).\n", (int)s);
    } else {
      haveDev = true;
      uint64_t maxRmw = 0, sramSz = 0;
      synDeviceAttribute a1 = DEVICE_ATTRIBUTE_MAX_RMW_SIZE;
      synDeviceAttribute a2 = DEVICE_ATTRIBUTE_SRAM_SIZE;
      synDeviceGetAttribute(&maxRmw, &a1, 1, dev);
      synDeviceGetAttribute(&sramSz, &a2, 1, dev);
      std::printf("  device acquired. SRAM=%.1f MB  MAX_RMW=%.1f MB\n",
                  sramSz / 1048576.0, maxRmw / 1048576.0);
    }
  }

  // ---- graph -------------------------------------------------------------
  synGraphHandle graph = nullptr;
  SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));

  std::vector<synSectionHandle> sections;

  // Inputs (persistent, HBM). NOTE Synapse geometry convention: sizes[0] is the FCD
  // (Fastest-Changing / innermost / CONTIGUOUS dim) == PyTorch's LAST dim. The packed
  // int32 holds 8 u4 nibbles along OUT, and that packed dim is the contiguous one
  // (production weight is [in, out/8] with out/8 innermost). So the packed dim is
  // sizes[0]. Decoded from habana::GetUnpackedShape (convert_from_int4_gen.cpp): it
  // does `shl 0x3` (×8) on the LAST/innermost shape element — i.e. the FCD expands 8×
  // on reinterpret. Hence (Synapse [FCD, outer]):
  //   weight_packed_i32 : int32 [OUT_PK, IN]      (FCD=OUT_PK; 8 u4/int32 along OUT)
  //   weight_scale      : bf16  [OUT,   GROUPS]   (FCD=OUT; per-group scale)
  //   zero_point        : int32 [OUT_PK, GROUPS]  PACKED symmetric zero (NEVER None)
  //   x                 : bf16  [IN,    M]        (FCD=IN; contraction along IN)
  // These map 1:1 to the PyTorch contract input[in,out/8]/scale[groups,out]/
  // zp[groups,out/8]/x[m,in] once the FCD<->last-dim flip is applied.
  //
  // DEFAULT path mirrors the on-device graph 1:1: a PERSISTENT int32 packed weight
  // [OUT_PK, IN] feeding a reinterpret_cast -> uint4. Build with -DDEQUANT_DIRECT_U4 to
  // skip the reinterpret_cast and declare the persistent weight DIRECTLY as uint4
  // [OUT, IN] (Synapse owns the nibble packing; identical IN*OUT/2 HBM bytes). Use the
  // direct path if the on-device compile rejects the standalone reinterpret_cast node.
#ifdef DEQUANT_DIRECT_U4
  synTensor wU4        = make2d(graph, "weight_packed", OUT, IN,     syn_type_uint4, true, &sections);
#else
  synTensor wPackedI32 = make2d(graph, "weight_packed", OUT_PK, IN,  syn_type_int32, true, &sections);
  // Intermediate A (NON-persistent): the int32 packed weight REINTERPRETED to its
  // unpacked uint4 geometry. Storage-identical to weight_packed (same IN*OUT/2 bytes),
  // FCD expands 8× (OUT_PK -> OUT). Synapse [OUT, IN] uint4.
  synTensor wU4        = make2d(graph, "weight_u4_view", OUT, IN,    syn_type_uint4, false, nullptr);
#endif
  synTensor wScale     = make2d(graph, "weight_scale",  OUT,    GROUPS, syn_type_bf16,  true, &sections);
#ifdef DEQUANT_DIRECT_U4
  synTensor wZpU4      = make2d(graph, "zero_point",    OUT,    GROUPS, syn_type_uint4, true, &sections);
#else
  synTensor wZpI32     = make2d(graph, "zero_point",    OUT_PK, GROUPS, syn_type_int32, true, &sections);
  // dequantize_4_bit requires zp dtype == weight dtype (uint4) or scale dtype (bf16);
  // int32 is rejected (COMPLEXGUID error 400). Reinterpret the packed int32 zp -> uint4
  // [OUT, GROUPS], mirroring the weight view (this is what convert_from_uint4 does
  // internally with its int32 zp input).
  synTensor wZpU4      = make2d(graph, "zp_u4_view",    OUT,    GROUPS, syn_type_uint4, false, nullptr);
#endif
  synTensor x          = make2d(graph, "x",             IN,     M,      syn_type_bf16,  true, &sections);

  // Intermediate B (NON-persistent => GC placement; THE tensor under test):
  //   W_bf16 : bf16 [OUT, IN]   (dequant output, FCD=OUT; MME operand B). Logically the
  //   convert_from_uint4 result [in,out] (out innermost == FCD).
  synTensor wBf16 = make2d(graph, "W_bf16_intermediate", OUT, IN, syn_type_bf16, false, nullptr);

  // Output (persistent, HBM):  out = x @ W => Synapse [OUT, M] (FCD=OUT).
  synTensor out = make2d(graph, "out", OUT, M, syn_type_bf16, true, &sections);

  // ---- node 1: reinterpret_cast  int32 [OUT/8, IN] -> uint4 [OUT, IN] ----
  // Storage-only bitcast (no arithmetic): the SIF derives the output geometry from the
  // 8x element-count ratio (int32 = 8 u4), expanding the FCD (sizes[0]: OUT/8 -> OUT).
  // reinterpretCastShapeInferenceFunction needs no user params
  // (ReinterpretCastNode::getShapeInferenceFunctionUserParamsSize == 0), so
  // pUserParams=nullptr,size=0.  (Skipped entirely under -DDEQUANT_DIRECT_U4: wU4 is
  // then the persistent uint4 weight directly.)
#ifndef DEQUANT_DIRECT_U4
  {
    synTensor in[1] = {wPackedI32};
    synTensor o[1]  = {wU4};
    SYN_CHECK(synNodeCreate(graph, in, o, 1, 1,
                            /*pUserParams=*/nullptr, /*paramsSize=*/0,
                            kReinterpretGuid, "reinterpret_w4", nullptr, nullptr));
  }
#endif

  // ---- node 1b: reinterpret_cast  int32 [OUT/8, GROUPS] -> uint4 [OUT, GROUPS] ----
  // Packed symmetric zero, same nibble layout as the weight; gives the dequant complex
  // guid a zp whose dtype (uint4) == weight dtype. Skipped under -DDEQUANT_DIRECT_U4.
#ifndef DEQUANT_DIRECT_U4
  {
    synTensor in[1] = {wZpI32};
    synTensor o[1]  = {wZpU4};
    SYN_CHECK(synNodeCreate(graph, in, o, 1, 1,
                            /*pUserParams=*/nullptr, /*paramsSize=*/0,
                            kReinterpretGuid, "reinterpret_zp", nullptr, nullptr));
  }
#endif

  // ---- node 2: dequantize_4_bit_bf16  (3 in -> 1 out) --------------------
  // Inputs MUST match the convert_from_uint4 contract (see header), here in Synapse
  // FCD geometry (sizes[0]=FCD; == the PyTorch contract with last-dim<->FCD flipped):
  //   in[0] uint4  [OUT, IN]       (reinterpreted weight; PyTorch [in,out])
  //   in[1] bf16   [OUT, GROUPS]   (per-group scale;       PyTorch [groups,out])
  //   in[2] int32  [OUT/8, GROUPS] PACKED symmetric zero;  PyTorch [groups,out/8]  (NEVER None)
  //   out   bf16   [OUT, IN]       (PyTorch [in,out])
  // The ComplexGuid expander supplies the dequant kernel + (on device) the implicit
  // reinterpret it would otherwise add; we already provide the uint4 view so it dequants
  // directly. Default: no params struct (mirrors the backend BuildNode). If the on-device
  // compile demands one, rebuild with -DDEQUANT_CAST_PARAMS=1 (see below).
  {
    synTensor in[3] = {wU4, wScale, wZpU4};
    synTensor o[1]  = {wBf16};
#ifdef DEQUANT_CAST_PARAMS
    // ns_CastKernel::ParamsV3 {CastF32RoundMode_t round_mode; unsigned seed; CastSatMode_t mode;}
    // zero-init => round_mode=CAST_ROUND_HALF_NE(0), seed=0, mode=CAST_NO_CLIP_NO_SAT(0).
    struct { int32_t round_mode; uint32_t seed; int32_t mode; } castParams;
    std::memset(&castParams, 0, sizeof(castParams));
    SYN_CHECK(synNodeCreate(graph, in, o, 3, 1,
                            &castParams, sizeof(castParams),
                            kDequantGuid, "dequant_w4", nullptr, nullptr));
#else
    SYN_CHECK(synNodeCreate(graph, in, o, 3, 1,
                            /*pUserParams=*/nullptr, /*paramsSize=*/0,
                            kDequantGuid, "dequant_w4", nullptr, nullptr));
#endif
  }

  // ---- node 3: MME gemm  (out[OUT,M] = W[OUT,IN] contract x[IN,M]) --------
  // A=x[IN,M], B=W_bf16[OUT,IN], no transpose => contract over IN, out=[OUT,M].
  {
    synGEMMParams gemm(kTransposeA, kTransposeB);
    synTensor in[2]  = {x, wBf16};
    synTensor o[1]   = {out};
    SYN_CHECK(synNodeCreate(graph, in, o, 2, 1,
                            &gemm, sizeof(gemm),
                            kGemmGuid, "moe_gemm", nullptr, nullptr));
  }

  // ---- compile -----------------------------------------------------------
  // Set GRAPH_VISUALIZATION + post-graph dump here so the human can read the
  // residency verdict from the compile log even before launching. (These are
  // also settable via env; setting them programmatically guarantees they're on.)
  synConfigurationSet("GRAPH_VISUALIZATION", "1");
  synConfigurationSet("DUMP_POST_GRAPHS", "1");

  std::printf("\n--- compiling fused graph (watch SRAM_SLICE / fuseDequantIntoMoeGemm logs) ---\n");
  synRecipeHandle recipe = nullptr;
  synStatus cs = synGraphCompile(&recipe, graph, "fused_moe_recipe", nullptr);
  if (cs != synSuccess) {
    std::fprintf(stderr,
        "[FATAL] synGraphCompile failed (synStatus=%d).\n"
        "  Most likely causes:\n"
        "   - the dequant complex-guid needs a CastKernel params struct: rebuild with\n"
        "     -DDEQUANT_CAST_PARAMS=1 (passes a zero-init ns_CastKernel::ParamsV3), OR\n"
        "   - the gemm operand orientation/contraction mismatch (flip kTransposeA/B), OR\n"
        "   - a dtype mismatch on an MME operand / the dequant scale/zp dtype.\n"
        "  Re-run with: HABANA_LOGS=./logs LOG_LEVEL_ALL=2 LOG_LEVEL_GC=1 for the pass trace.\n",
        (int)cs);
    return 2;
  }
  std::printf("[OK] graph compiled. recipe=%p\n", (void*)recipe);

  uint64_t wsSize = 0;
  SYN_CHECK(synWorkspaceGetSize(&wsSize, recipe));
  // THE COMPILE-TIME VERDICT (heuristic): if the GC kept W_bf16 in SRAM and sliced
  // the bundle, the workspace should be SMALL (just the sliced chunks / scratch),
  // NOT ~= the full bf16 weight. A workspace >= bf16_w_mb strongly implies the
  // intermediate was spilled to an HBM workspace tensor (fusion FAILED).
  const double ws_mb = wsSize / (1024.0 * 1024.0);
  std::printf("\n=== COMPILE-TIME RESIDENCY HINT ===\n");
  std::printf("  workspace size      = %.2f MB\n", ws_mb);
  std::printf("  W bf16 size         = %.2f MB\n", bf16_w_mb);
  std::printf("  W int4 size         = %.2f MB\n", int4_w_mb);
  if (ws_mb >= bf16_w_mb * 0.9) {
    std::printf("  >>> HINT: workspace >= ~bf16 weight  => W_bf16 likely SPILLED to HBM "
                "(fusion FAILED). Inspect post-graph dump + SRAM_SLICE log.\n");
  } else {
    std::printf("  >>> HINT: workspace << bf16 weight  => consistent with SRAM-resident "
                "sliced bundle (fusion likely WORKING). Confirm via on-device HBM profile.\n");
  }
  std::printf("  (Authoritative verdict = on-device synProfiler HBM bytes, below / profile_fusion.py.)\n");

  if (compileOnly || !haveDev) {
    std::printf("\n[compile-only] graph built + compiled OK; skipping launch (no device / FUSED_COMPILE_ONLY).\n");
    synRecipeDestroy(recipe);
    synGraphDestroy(graph);
    for (auto s : sections) synSectionDestroy(s);
    synDestroy();
    return 0;
  }

  // ---- allocate device buffers for persistent tensors --------------------
  auto devAlloc = [&](uint64_t bytes) -> uint64_t {
    uint64_t p = 0;
    SYN_CHECK(synDeviceMalloc(dev, bytes, 0, 0, &p));
    return p;
  };
  // weight_packed is int32 [OUT_PK, IN] -> OUT_PK*IN*4 bytes == IN*OUT/2 bytes (4 bits/elem).
  const uint64_t wPacked_b = OUT_PK * IN     * bytesOfElt(syn_type_int32);  // == IN*OUT/2
  const uint64_t wScale_b  = GROUPS * OUT    * bytesOfElt(syn_type_bf16);
  const uint64_t wZp_b     = GROUPS * OUT_PK * bytesOfElt(syn_type_int32);
  const uint64_t x_b       = IN     * M      * bytesOfElt(syn_type_bf16);
  const uint64_t out_b     = OUT    * M      * bytesOfElt(syn_type_bf16);

  uint64_t d_wPacked = devAlloc(wPacked_b);
  uint64_t d_wScale  = devAlloc(wScale_b);
  uint64_t d_wZp     = devAlloc(wZp_b);
  uint64_t d_x       = devAlloc(x_b);
  uint64_t d_out     = devAlloc(out_b);
  uint64_t d_ws      = wsSize ? devAlloc(wsSize) : 0;

  // Initialize inputs to simple constants (host -> device). Correctness numerics
  // are validated by test_p1_fusion.py against convert_from_uint4+matmul; here we
  // only need a valid launch to MEASURE HBM bytes, so zero-fill x/scale is fine
  // except we set scale!=0 and a valid packed-zero so the kernel does real work.
  synStreamHandle stream = nullptr;
  SYN_CHECK(synStreamCreateGeneric(&stream, dev, 0));

  auto hostFillAndCopy = [&](uint64_t dptr, uint64_t bytes, int fillByte) {
    void* h = nullptr;
    SYN_CHECK(synHostMalloc(dev, bytes, 0, &h));
    std::memset(h, fillByte, bytes);
    SYN_CHECK(synMemCopyAsync(stream, (uint64_t)h, bytes, dptr, HOST_TO_DRAM));
    SYN_CHECK(synStreamSynchronize(stream));
    SYN_CHECK(synHostFree(dev, h, 0));
  };
  // packed zero = 0x88888888 (sym zero 8). scale bytes set to a small bf16 pattern.
  hostFillAndCopy(d_wPacked, wPacked_b, 0x12);  // arbitrary nonzero codes
  hostFillAndCopy(d_wScale,  wScale_b,  0x3c);  // bf16 ~ small positive (0x3c00=1.0 hi byte)
  hostFillAndCopy(d_wZp,     wZp_b,     0x88);  // 0x88888888 -> sym zero 8
  hostFillAndCopy(d_x,       x_b,       0x3c);

  // ---- launch tensor info (persistent only; intermediate is internal) ----
  std::vector<synLaunchTensorInfo> lt;
  auto addLT = [&](const char* name, uint64_t addr) {
    synLaunchTensorInfo i;
    std::memset(&i, 0, sizeof(i));
    i.tensorName = name;
    i.pTensorAddress = addr;
    i.tensorType = DATA_TENSOR;
    lt.push_back(i);
  };
  addLT("weight_packed", d_wPacked);
  addLT("weight_scale",  d_wScale);
  addLT("zero_point",    d_wZp);
  addLT("x",             d_x);
  addLT("out",           d_out);

  // ---- profile HBM bytes around the launch -------------------------------
  std::printf("\n--- launching + profiling (synProfiler device trace) ---\n");
  uint32_t needBytes = 0;
  if (synProfilerQueryRequiredMemory(dev, &needBytes) == synSuccess && needBytes) {
    std::printf("  (profiler internal buffer requirement: %u bytes)\n", needBytes);
  }
  synProfilerStart(synTraceDevice, dev);

  // warm + timed launches
  for (int it = 0; it < 5; ++it) {
    SYN_CHECK(synLaunch(stream, lt.data(), (uint32_t)lt.size(), d_ws, recipe,
                        SYN_FLAGS_TENSOR_NAME));
  }
  SYN_CHECK(synStreamSynchronize(stream));

  synProfilerStop(synTraceDevice, dev);

  // Write the trace to disk (TEF JSON) for offline HBM-byte extraction; the
  // companion profile_fusion.py parses engine BW counters from it. Passing a null
  // buffer with format=TEF makes Synapse dump the trace file to ./ (HABANA_PROF
  // location), and returns the total size.
  size_t traceSize = 0, traceEntries = 0;
  synProfilerGetTrace(synTraceDevice, dev, synTraceFormatTEF, nullptr,
                      &traceSize, &traceEntries);
  std::printf("  profiler trace: %zu entries, %zu bytes (TEF dumped to the profiler dir).\n",
              traceEntries, traceSize);
  std::printf("  -> read HBM read/write bytes from the trace (MME+TPC DRAM counters) and compare:\n");
  std::printf("       ~int4 (%.1f MB/iter) = SRAM-resident fusion WORKS;\n", int4_w_mb);
  std::printf("       ~bf16 (%.1f MB/iter) = GC spilled W_bf16 to HBM, fusion FAILED.\n", bf16_w_mb);

  // ---- cleanup -----------------------------------------------------------
  synStreamSynchronize(stream);
  if (d_ws) synDeviceFree(dev, d_ws, 0);
  synDeviceFree(dev, d_out, 0);
  synDeviceFree(dev, d_x, 0);
  synDeviceFree(dev, d_wZp, 0);
  synDeviceFree(dev, d_wScale, 0);
  synDeviceFree(dev, d_wPacked, 0);
  synRecipeDestroy(recipe);
  synGraphDestroy(graph);
  for (auto s : sections) synSectionDestroy(s);
  synDestroy();
  std::printf("\n[done] fused_moe_recipe complete.\n");
  return 0;
}
