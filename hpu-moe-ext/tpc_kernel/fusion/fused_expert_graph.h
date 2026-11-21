/*
 * fused_expert_graph.h -- the PROVEN per-expert fused int4 MoE Synapse graph,
 * factored out of fused_moe_full.cpp so BOTH the standalone validator
 * (fused_moe_full.cpp, the numeric oracle) AND the torch custom op
 * (fused_moe_op.cpp, the vLLM bridge) build the IDENTICAL recipe.
 *
 * This header is the single source of truth for the graph topology + tensor
 * geometry. It is a VERBATIM lift of buildExpert() from fused_moe_full.cpp
 * (committed 6ac14bc), which COMPILES, RUNS, and is NUMERICALLY CORRECT
 * (relNorm ~0.5% vs CPU ref at GLM shapes) and SRAM-resident (workspace=0). The
 * only change is making it a reusable inline function + exposing the stable
 * tensor-name list for synLaunch binding.
 *
 * GRAPH (one expert):
 *   int32 packed -> reinterpret_cast -> uint4
 *     -> dequantize_4_bit_bf16 (W13_bf16, NON-PERSISTENT)  -> MME gemm #1 (gu)
 *     -> slice gate/up -> sigmoid+mult (SiLU) -> mult (h)
 *   int32 packed -> reinterpret_cast -> uint4
 *     -> dequantize_4_bit_bf16 (W2_bf16, NON-PERSISTENT)   -> MME gemm #2 (y)
 *     -> mult by router_weight -> out
 *
 * GEOMETRY (Synapse: sizes[0] == FCD == PyTorch innermost/last dim). With
 * H=hidden, I=moe_intermediate, M=tokens, OUT13=2*I, GS=group size:
 *
 *   PyTorch contiguous tensor          Synapse make2d(d0=FCD, d1)          dtype
 *   x            [M, H]                {H,        M}                        bf16
 *   w13_packed   [H, 2I/8]            {2I/8,     H}                         int32
 *   w13_scale    [H/GS, 2I]          {2I,       H/GS}                       bf16
 *   w13_zp       [H/GS, 2I/8]        {2I/8,     H/GS}                       int32
 *   w2_packed    [I, H/8]            {H/8,      I}                          int32
 *   w2_scale     [I/GS, H]           {H,        I/GS}                       bf16
 *   w2_zp        [I/GS, H/8]         {H/8,      I/GS}                       int32
 *   router_weight scalar [1,1] (or full [M,H] with RW_FULL)               bf16
 *   out          [M, H]              {H,        M}                         bf16
 *
 * >>> KEY for the torch op: a PyTorch *contiguous* tensor's data_ptr() is
 * row-major with the LAST dim contiguous == the Synapse FCD. So the per-expert
 * torch tensors produced by the vLLM repack (w13_qweight[e] = int32 [H, 2I/8],
 * w13_scales[e] = bf16 [H/GS, 2I], ...) bind DIRECTLY to these graph tensors
 * with NO transpose and NO repack -- their byte layout already equals the
 * geometry above. (This is why approach B is a thin wrapper, not a rewrite.)
 *
 * Build flags honored (parity with build_full.sh):
 *   -DUSE_SILU_CGUID=1      single silu_fwd_bf16 ComplexGuid node (else sigmoid+mult)
 *   -DDEQUANT_CAST_PARAMS=1 pass a zero-init params blob to the dequant node
 *   -DRW_FULL=1             full [M,H] router_weight operand (no [1,1] broadcast)
 */
#ifndef HPU_MOE_EXT_FUSED_EXPERT_GRAPH_H
#define HPU_MOE_EXT_FUSED_EXPERT_GRAPH_H

#include "synapse_api.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"  // synGEMMParams

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace hpu_moe_ext {

// --------------------------------------------------------------------------
#ifndef FEG_SYN_CHECK
#define FEG_SYN_CHECK(expr)                                                     \
  do {                                                                         \
    synStatus _s = (expr);                                                     \
    if (_s != synSuccess) {                                                    \
      std::fprintf(stderr, "[FATAL][fused_expert_graph] %s -> synStatus=%d "   \
                           "(line %d)\n",                                      \
                   #expr, (int)_s, __LINE__);                                  \
      std::abort();                                                            \
    }                                                                         \
  } while (0)
#endif

// REAL graph-node guids (decoded on-device / from libSynapse + libtpc_kernels).
inline const char* kReinterpretGuid() { return "reinterpret_cast"; }      // core Synapse node
inline const char* kDequantGuid()     { return "dequantize_4_bit_bf16"; } // ComplexGuid (proven)
inline const char* kGemmGuid()        { return "gemm"; }                  // MME (proven)
inline const char* kSliceAxisGuid()   { return "slice_axis"; }            // TYPE_SLICE_AXIS
inline const char* kSigmoidGuid()     { return "sigmoid_fwd_bf16"; }      // base TPC kernel
inline const char* kMultGuid()        { return "mult_fwd_bf16"; }         // base TPC kernel
#ifdef USE_SILU_CGUID
inline const char* kSiluGuid()        { return "silu_fwd_bf16"; }         // ComplexGuid (UNCONFIRMED)
#endif

// The stable persistent tensor names, in the order the launcher binds them.
// (out is the single output.)  Used to build the synLaunchTensorInfo list.
struct ExpertTensorNames {
  static constexpr const char* kX          = "x";
  static constexpr const char* kW13Packed  = "w13_packed";
  static constexpr const char* kW13Scale   = "w13_scale";
  static constexpr const char* kW13Zp      = "w13_zp";
  static constexpr const char* kW2Packed   = "w2_packed";
  static constexpr const char* kW2Scale    = "w2_scale";
  static constexpr const char* kW2Zp       = "w2_zp";
  static constexpr const char* kRouterW    = "router_weight";
  static constexpr const char* kOut        = "out";
};

// One compiled per-expert graph + its persistent-section bookkeeping + workspace.
struct ExpertRecipe {
  synGraphHandle  graph  = nullptr;
  synRecipeHandle recipe = nullptr;
  std::vector<synSectionHandle> sections;
  uint64_t wsSize = 0;
  // cache key (so a static map can dedup recipes per unique shape)
  uint64_t M = 0, H = 0, I = 0, GS = 0;
  bool unfuse = false;
};

// --- tensor factory: 2D, FCD=d0 (verbatim from fused_moe_full.cpp) ------------
inline synTensor feg_make2d(synGraphHandle g, const char* name, uint64_t d0,
                            uint64_t d1, synDataType dtype, bool persistent,
                            std::vector<synSectionHandle>* sections) {
  synTensor t = nullptr;
  FEG_SYN_CHECK(synTensorHandleCreate(&t, g, DATA_TENSOR, name));
  synTensorGeometry geo;
  std::memset(&geo, 0, sizeof(geo));
  geo.dims = 2;
  geo.sizes[0] = d0;  // FCD
  geo.sizes[1] = d1;
  FEG_SYN_CHECK(synTensorSetGeometry(t, &geo, synGeometrySizes));
  FEG_SYN_CHECK(synTensorSetDeviceDataType(t, dtype));
  if (persistent) {
    synSectionHandle sec = nullptr;
    FEG_SYN_CHECK(synSectionCreate(&sec, 0, g));
    FEG_SYN_CHECK(synSectionSetPersistent(sec, true));
    FEG_SYN_CHECK(synTensorAssignToSection(t, sec, 0));
    sections->push_back(sec);
  }
  return t;
}

// GEMM orientation (proven): out = W . x, no transpose.
//   MME #1: A=x[H,M], B=W13[OUT13,H] -> gu[OUT13,M]   (K=H)
//   MME #2: A=h[I,M], B=W2 [H,   I] -> y [H,   M]     (K=I)

// Build the full per-expert graph into `er`, then COMPILE it. recipeName must be
// unique per distinct recipe (the GC uses it to name the recipe + dumps). On any
// failure this aborts LOUDLY with the exact diagnostic (same text as the proven
// validator). When unfuse==true the two dequant outputs are made PERSISTENT
// (defeats SRAM bundling -> bf16 HBM round-trip; for A/B only).
inline void buildExpertRecipe(ExpertRecipe& er, uint64_t M, uint64_t H,
                              uint64_t I, uint64_t GS, bool unfuse,
                              const char* recipeName) {
  const uint64_t OUT13   = 2 * I;
  const uint64_t OUT13PK = OUT13 / 8;
  const uint64_t G13     = H / GS;   // w13 contracts over H
  const uint64_t H_PK    = H / 8;
  const uint64_t G2      = I / GS;   // w2 contracts over I

  er.M = M; er.H = H; er.I = I; er.GS = GS; er.unfuse = unfuse;

  FEG_SYN_CHECK(synGraphCreate(&er.graph, synDeviceGaudi2));
  synGraphHandle g = er.graph;
  auto* S = &er.sections;

  // ---------------- persistent inputs (HBM) ----------------
  synTensor x          = feg_make2d(g, ExpertTensorNames::kX,         H,       M,    syn_type_bf16,  true, S);
  synTensor w13_packed = feg_make2d(g, ExpertTensorNames::kW13Packed, OUT13PK, H,    syn_type_int32, true, S);
  synTensor w13_scale  = feg_make2d(g, ExpertTensorNames::kW13Scale,  OUT13,   G13,  syn_type_bf16,  true, S);
  synTensor w13_zp_i32 = feg_make2d(g, ExpertTensorNames::kW13Zp,     OUT13PK, G13,  syn_type_int32, true, S);
  synTensor w2_packed  = feg_make2d(g, ExpertTensorNames::kW2Packed,  H_PK,    I,    syn_type_int32, true, S);
  synTensor w2_scale   = feg_make2d(g, ExpertTensorNames::kW2Scale,   H,       G2,   syn_type_bf16,  true, S);
  synTensor w2_zp_i32  = feg_make2d(g, ExpertTensorNames::kW2Zp,      H_PK,    G2,   syn_type_int32, true, S);
#ifdef RW_FULL
  synTensor rw         = feg_make2d(g, ExpertTensorNames::kRouterW,   H,       M,    syn_type_bf16,  true, S);
#else
  synTensor rw         = feg_make2d(g, ExpertTensorNames::kRouterW,   1,       1,    syn_type_bf16,  true, S);
#endif
  synTensor out        = feg_make2d(g, ExpertTensorNames::kOut,       H,       M,    syn_type_bf16,  true, S);

  // ---------------- intermediates (non-persistent) ----------------
  synTensor w13_u4    = feg_make2d(g, "w13_u4_view", OUT13, H,    syn_type_uint4, false, nullptr);
  synTensor w13_zp_u4 = feg_make2d(g, "w13_zp_u4",   OUT13, G13,  syn_type_uint4, false, nullptr);
  synTensor w2_u4     = feg_make2d(g, "w2_u4_view",  H,     I,    syn_type_uint4, false, nullptr);
  synTensor w2_zp_u4  = feg_make2d(g, "w2_zp_u4",    H,     G2,   syn_type_uint4, false, nullptr);

  // dequant outputs -- THE residency under test. persistent iff unfuse.
  synTensor W13_bf16 = feg_make2d(g, "W13_bf16", OUT13, H, syn_type_bf16, unfuse, S);
  synTensor W2_bf16  = feg_make2d(g, "W2_bf16",  H,     I, syn_type_bf16, unfuse, S);

  synTensor gu   = feg_make2d(g, "gu",        OUT13, M, syn_type_bf16, false, nullptr);
  synTensor gate = feg_make2d(g, "gate",      I,     M, syn_type_bf16, false, nullptr);
  synTensor up   = feg_make2d(g, "up",        I,     M, syn_type_bf16, false, nullptr);
#ifndef USE_SILU_CGUID
  synTensor sg   = feg_make2d(g, "sigmoid_g", I,     M, syn_type_bf16, false, nullptr);
#endif
  synTensor silu = feg_make2d(g, "silu_g",    I,     M, syn_type_bf16, false, nullptr);
  synTensor h    = feg_make2d(g, "h",         I,     M, syn_type_bf16, false, nullptr);
  synTensor y    = feg_make2d(g, "y",         H,     M, syn_type_bf16, false, nullptr);

  // ---- node: reinterpret_cast int32 packed -> uint4
  auto reinterp = [&](synTensor in, synTensor o, const char* nm) {
    synTensor in1[1] = {in};
    synTensor o1[1]  = {o};
    FEG_SYN_CHECK(synNodeCreate(g, in1, o1, 1, 1, nullptr, 0, kReinterpretGuid(), nm, nullptr, nullptr));
  };
  reinterp(w13_packed, w13_u4,    "reint_w13");
  reinterp(w13_zp_i32, w13_zp_u4, "reint_w13_zp");
  reinterp(w2_packed,  w2_u4,     "reint_w2");
  reinterp(w2_zp_i32,  w2_zp_u4,  "reint_w2_zp");

  // ---- node: dequantize_4_bit_bf16 (u4 W, bf16 scale, u4 zp) -> bf16 W
  auto dequant = [&](synTensor wu4, synTensor sc, synTensor zpu4, synTensor o, const char* nm) {
    synTensor in3[3] = {wu4, sc, zpu4};
    synTensor o1[1]  = {o};
#ifdef DEQUANT_CAST_PARAMS
    struct { int32_t round_mode; uint32_t seed; int32_t mode; } cp;
    std::memset(&cp, 0, sizeof(cp));
    FEG_SYN_CHECK(synNodeCreate(g, in3, o1, 3, 1, &cp, sizeof(cp), kDequantGuid(), nm, nullptr, nullptr));
#else
    FEG_SYN_CHECK(synNodeCreate(g, in3, o1, 3, 1, nullptr, 0, kDequantGuid(), nm, nullptr, nullptr));
#endif
  };
  dequant(w13_u4, w13_scale, w13_zp_u4, W13_bf16, "dequant_w13");
  dequant(w2_u4,  w2_scale,  w2_zp_u4,  W2_bf16,  "dequant_w2");

  // ---- MME gemm #1  gu[OUT13,M] = W13[OUT13,H] . x[H,M]  (contract H)
  {
    synGEMMParams gp(false, false);
    synTensor in2[2] = {x, W13_bf16};
    synTensor o1[1]  = {gu};
    FEG_SYN_CHECK(synNodeCreate(g, in2, o1, 2, 1, &gp, sizeof(gp), kGemmGuid(), "gemm_w13", nullptr, nullptr));
  }

  // ---- slice_axis split gu along axis 0 (FCD=OUT13): gate[0:I], up[I:2I]
  {
    synSliceAxisParams sp; std::memset(&sp, 0, sizeof(sp));
    sp.axis = 0; sp.begin = 0; sp.end = (unsigned)I;
    synTensor in1[1] = {gu}; synTensor o1[1] = {gate};
    FEG_SYN_CHECK(synNodeCreate(g, in1, o1, 1, 1, &sp, sizeof(sp), kSliceAxisGuid(), "slice_gate", nullptr, nullptr));
  }
  {
    synSliceAxisParams sp; std::memset(&sp, 0, sizeof(sp));
    sp.axis = 0; sp.begin = (unsigned)I; sp.end = (unsigned)OUT13;
    synTensor in1[1] = {gu}; synTensor o1[1] = {up};
    FEG_SYN_CHECK(synNodeCreate(g, in1, o1, 1, 1, &sp, sizeof(sp), kSliceAxisGuid(), "slice_up", nullptr, nullptr));
  }

  // ---- silu(gate) = gate * sigmoid(gate) ; h = silu(gate) * up
#ifdef USE_SILU_CGUID
  {
    struct { float alpha; } sparams; sparams.alpha = 1.0f;
    synTensor in1[1] = {gate}; synTensor o1[1] = {silu};
    FEG_SYN_CHECK(synNodeCreate(g, in1, o1, 1, 1, &sparams, sizeof(sparams), kSiluGuid(), "silu", nullptr, nullptr));
  }
#else
  {
    synTensor in1[1] = {gate}; synTensor o1[1] = {sg};
    FEG_SYN_CHECK(synNodeCreate(g, in1, o1, 1, 1, nullptr, 0, kSigmoidGuid(), "sigmoid_gate", nullptr, nullptr));
  }
  {
    synTensor in2[2] = {gate, sg}; synTensor o1[1] = {silu};
    FEG_SYN_CHECK(synNodeCreate(g, in2, o1, 2, 1, nullptr, 0, kMultGuid(), "silu_mul", nullptr, nullptr));
  }
#endif
  {
    synTensor in2[2] = {silu, up}; synTensor o1[1] = {h};
    FEG_SYN_CHECK(synNodeCreate(g, in2, o1, 2, 1, nullptr, 0, kMultGuid(), "h_mul", nullptr, nullptr));
  }

  // ---- MME gemm #2  y[H,M] = W2[H,I] . h[I,M]  (contract I)
  {
    synGEMMParams gp(false, false);
    synTensor in2[2] = {h, W2_bf16}; synTensor o1[1] = {y};
    FEG_SYN_CHECK(synNodeCreate(g, in2, o1, 2, 1, &gp, sizeof(gp), kGemmGuid(), "gemm_w2", nullptr, nullptr));
  }

  // ---- out = y * router_weight  (broadcast [1,1] over [H,M], or full [H,M])
  {
    synTensor in2[2] = {y, rw}; synTensor o1[1] = {out};
    FEG_SYN_CHECK(synNodeCreate(g, in2, o1, 2, 1, nullptr, 0, kMultGuid(), "router_scale", nullptr, nullptr));
  }

  // ---- compile ----
  synStatus cs = synGraphCompile(&er.recipe, er.graph, recipeName, nullptr);
  if (cs != synSuccess) {
    std::fprintf(stderr,
        "[FATAL][fused_expert_graph] synGraphCompile(%s) failed (synStatus=%d).\n"
        "  Likely (device-confirmable): dequant complex-guid wants a params blob\n"
        "  (-DDEQUANT_CAST_PARAMS=1); slice_axis param mismatch; a guid not registered;\n"
        "  the [1,1] router-weight broadcast rejected on mult_fwd_bf16 (-DRW_FULL=1); or\n"
        "  gemm orientation. Re-run the standalone fused_moe_full with HABANA_LOGS=./logs\n"
        "  LOG_LEVEL_ALL=2 LOG_LEVEL_GC=1 for the pass trace.\n",
        recipeName, (int)cs);
    std::abort();
  }
  FEG_SYN_CHECK(synWorkspaceGetSize(&er.wsSize, er.recipe));
}

inline void destroyExpertRecipe(ExpertRecipe& er) {
  if (er.recipe) synRecipeDestroy(er.recipe);
  if (er.graph)  synGraphDestroy(er.graph);
  for (auto s : er.sections) synSectionDestroy(s);
  er.sections.clear();
  er.recipe = nullptr; er.graph = nullptr;
}

}  // namespace hpu_moe_ext

#endif  // HPU_MOE_EXT_FUSED_EXPERT_GRAPH_H
