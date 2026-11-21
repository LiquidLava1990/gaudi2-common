/*
 * fused_moe_full.cpp -- the FULL per-expert fused int4 MoE compute (one expert).
 *
 * Extends the PROVEN single-expert fused recipe (fused_moe_recipe.cpp, which
 * builds  int32 packed -> reinterpret_cast -> uint4 -> dequantize_4_bit_bf16
 * (NON-PERSISTENT) -> MME "gemm", and which the GC was CONFIRMED to bundle the
 * dequant into the MME with 0 MB workspace / SRAM-resident, no bf16 HBM
 * round-trip) into the complete expert forward:
 *
 *     gu  = W13 . x                      (MME #1; W13 dequanted from int4)
 *     gate, up = split(gu)               (slice_axis along OUT13)
 *     h   = silu(gate) * up              (sigmoid_fwd_bf16 + mult_fwd_bf16)
 *     y   = W2 . h                       (MME #2; W2 dequanted from int4)
 *     out = y * router_weight            (mult_fwd_broadcast / mult_fwd_bf16)
 *
 * This is the production per-expert math (verbatim from
 * hpu_moe_ext/fused_routed_moe.py:210-214 / fp8_routed_moe.py:_silu_mul):
 *     gate_up = x @ W13 ; gate = gate_up[:, :I] ; up = gate_up[:, I:]
 *     h = F.silu(gate) * up ; expert_out = h @ W2 ; out = expert_out * w_k
 *
 * GEOMETRY (Synapse: sizes[0] == FCD == PyTorch innermost/last dim).
 * --------------------------------------------------------------------------
 * Each PyTorch [a, b] (b innermost) is Synapse {b, a}. With H=hidden,
 * I=moe_intermediate, M=tokens, OUT13=2*I, GS=group size:
 *
 *   x            PyTorch [M, H]          -> Synapse [H,        M]        (FCD=H)
 *   --- w13 (gate||up) packed int4 ; dequant out W13_bf16 [OUT13, H] ---
 *   w13_packed   PyTorch [H, OUT13/8]    -> Synapse [OUT13/8,  H]        (FCD=OUT13/8; int32)
 *   w13_scale    PyTorch [GROUPS13,OUT13]-> Synapse [OUT13,    GROUPS13] (FCD=OUT13; bf16)
 *   w13_zp       PyTorch [GROUPS13,OUT13/8]->Synapse[OUT13/8,  GROUPS13] (FCD=OUT13/8; int32 packed sym-zero)
 *   W13_bf16     (dequant out)           -> Synapse [OUT13,    H]        (FCD=OUT13; NON-PERSISTENT)
 *   gu = W13.x   contract H              -> Synapse [OUT13,    M]        (FCD=OUT13)
 *   gate = gu[0:I]    (slice_axis ax0)   -> Synapse [I,        M]
 *   up   = gu[I:OUT13](slice_axis ax0)   -> Synapse [I,        M]
 *   sg = sigmoid(gate)                   -> Synapse [I,        M]
 *   h  = gate * up_via_silu              -> Synapse [I,        M]
 *   --- w2 (down) packed int4 ; dequant out W2_bf16 [H, I] ---
 *   w2_packed    PyTorch [I, H/8]        -> Synapse [H/8,      I]        (FCD=H/8; int32)
 *   w2_scale     PyTorch [GROUPS2, H]    -> Synapse [H,        GROUPS2]  (FCD=H; bf16)
 *   w2_zp        PyTorch [GROUPS2, H/8]  -> Synapse [H/8,      GROUPS2]  (FCD=H/8; int32)
 *   W2_bf16      (dequant out)           -> Synapse [H,        I]        (FCD=H; NON-PERSISTENT)
 *   y = W2.h     contract I              -> Synapse [H,        M]        (FCD=H)
 *   out = y * rw                         -> Synapse [H,        M]        (persistent)
 *
 * GROUPS13 = H/GS  (w13 contracts over H), GROUPS2 = I/GS  (w2 contracts over I).
 *
 * WHY THE DEQUANTS ARE NON-PERSISTENT (the whole point)
 * --------------------------------------------------------------------------
 * W13_bf16 and W2_bf16 carry NO section -> graph-internal. The GC's
 * fuseDequantIntoMoeGemm + SRAM slicer bundle each dequant TPC node into its
 * consuming MME and slice the contraction dim so the bf16 weight stays SRAM-
 * resident and never round-trips HBM (confirmed for the single-expert recipe:
 * fusedTPCNode bundle op nodes + split_logical/concat_logical in the PostGraph
 * dump, 0 MB workspace). The UNFUSED A/B variant (UNFUSE=1 / make_unfused=true)
 * forces W13_bf16/W2_bf16 PERSISTENT, which defeats the bundling and pays the
 * full bf16 HBM round-trip -> the latency delta is the measured fusion win.
 *
 * SILU DECOMPOSITION
 * --------------------------------------------------------------------------
 * libtpc_kernels.so has NO standalone silu_fwd_bf16 graph kernel (only
 * sigmoid_fwd_bf16 + mult_fwd_bf16 are registered base elementwise kernels; a
 * `silu`/`silu_fwd` ComplexGuid exists in libComplexGuid.so but its bf16
 * lowering on this build is UNCONFIRMED). So SiLU is composed exactly as
 * production does it: silu(g) = g * sigmoid(g), then * up:
 *     sg = sigmoid_fwd_bf16(gate)        # [I, M]
 *     gs = mult_fwd_bf16(gate, sg)       # silu(gate)
 *     h  = mult_fwd_bf16(gs, up)         # silu(gate) * up
 * (Build with -DUSE_SILU_CGUID=1 to instead emit a single "silu_fwd_bf16"
 * ComplexGuid node for gate, if the device confirms it lowers; default uses the
 * proven sigmoid+mult base kernels.)
 *
 * ROUTER WEIGHT
 * --------------------------------------------------------------------------
 * router_weight is a scalar provided as a [1,1] bf16 persistent tensor. We scale
 * y by it with a broadcast mult. mult_fwd_broadcast_fcd/non_fcd kernels exist;
 * the simplest portable form passes the [1,1] tensor and lets the broadcast
 * variant of mult handle it. If the device rejects the [1,1] broadcast on the
 * stock "mult_fwd_bf16" guid (it may demand an explicit broadcast guid or a
 * full-size operand), the fallback is to bake rw into the w2 scale on the host
 * (mathematically identical) -- noted in the return summary; the default emits a
 * mult node with the [1,1] rw tensor.
 *
 * NUMERIC VALIDATION (default; the correctness gate)
 * --------------------------------------------------------------------------
 * In non-compile-only mode the program fills deterministic int4 codes + scales +
 * sym-zero + x + rw on the host, launches the graph, copies `out` back, and
 * compares against a CPU reference computed in-process with the SAME dequant
 * rule ((code-8)*scale per group), the SAME split, SAME silu(gate)*up, SAME
 * @W2, SAME *rw. Prints maxdiff + rel; exits nonzero (LOUD) if rel > 1e-2.
 *
 * LATENCY A/B
 * --------------------------------------------------------------------------
 * With LATENCY_AB=1 the program builds BOTH the fused and the unfused recipe,
 * times N=200 synLaunch iters (after warmup) of each with synEventElapsedTime,
 * and prints ms/iter, tok/s-equiv, and the speedup. (Or run the binary twice:
 * once default = fused, once UNFUSE=1 = unfused, and compare the printed
 * ms/iter -- both paths print their own timing block.)
 *
 * Shapes are GENERAL. argv:  M H I GS   (defaults = GLM-5.1 w13/w2 expert).
 *   M  tokens                (default 32)
 *   H  hidden                (default 4096)   [GLM-5.1 hidden_size]
 *   I  moe_intermediate      (default 1408)   [GLM-5.1 moe_intermediate_size]
 *   GS group size            (default 128)
 * (OUT13 = 2*I.)  Require H%GS==0, I%GS==0, (2*I)%8==0, H%8==0.
 *
 * Build CPU-only (host g++ links libSynapse.so):  ./build_full.sh
 * Run ON THE HPU BOX ONLY:
 *   FUSED_COMPILE_ONLY=1 ./build/fused_moe_full 32 4096 1408 128   # compile sanity
 *   ./build/fused_moe_full 32 4096 1408 128                        # numeric validation (fused)
 *   UNFUSE=1 ./build/fused_moe_full 32 4096 1408 128               # numeric validation (unfused)
 *   LATENCY_AB=1 ./build/fused_moe_full 32 4096 1408 128           # A/B latency, both variants
 */

#include "synapse_api.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"  // synGEMMParams

// SHARED graph builder (single source of truth). The torch bridge op
// (fused_moe_op.cpp) builds the IDENTICAL recipe from this header. This file
// keeps the deterministic host inputs, the CPU reference, the numeric-validation
// gate, and the A/B latency harness, but delegates the graph topology to the
// header so the validator proves exactly what the op runs.
#include "fused_expert_graph.h"

#include <cmath>
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

// NOTE: the graph topology (guids, gemm orientation, make2d, buildExpert) now lives
// in fused_expert_graph.h (single source of truth, shared with the torch op). This
// file keeps ONLY the deterministic host inputs, the CPU reference, the numeric gate,
// and the A/B latency harness below.

// ---- bf16 <-> float host helpers (round-to-nearest-even on store) ------------
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round to nearest even
  uint32_t lsb = (u >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  u += rounding_bias;
  return (uint16_t)(u >> 16);
}

// pack 8 uint4 codes (each 0..15) little-endian into one int32 word
inline int32_t pack8(const uint8_t* codes) {
  uint32_t w = 0;
  for (int n = 0; n < 8; ++n) w |= (uint32_t)(codes[n] & 0xF) << (4 * n);
  return (int32_t)w;
}

// --- thin adapters over the SHARED graph builder (fused_expert_graph.h) --------
// ExpertGraph is the validator's local alias for the header's ExpertRecipe so the
// rest of main() (which reads .recipe / .wsSize) is unchanged. buildExpert wraps
// buildExpertRecipe and adds the validator's banner + workspace print; the dump
// configs are enabled here (the op does NOT dump). This guarantees the validator
// proves the EXACT recipe the torch op builds.
using ExpertGraph = hpu_moe_ext::ExpertRecipe;

void buildExpert(ExpertGraph& eg, uint64_t M, uint64_t H, uint64_t I, uint64_t GS,
                 bool unfuse, const char* recipeName) {
  synConfigurationSet("GRAPH_VISUALIZATION", "1");
  synConfigurationSet("DUMP_POST_GRAPHS", "1");
  std::printf("\n--- compiling %s graph (%s) ---\n", recipeName,
              unfuse ? "UNFUSED: dequant outputs PERSISTENT -> bf16 HBM round-trip"
                     : "FUSED: dequant outputs NON-PERSISTENT -> GC bundles into MME (SRAM)");
  hpu_moe_ext::buildExpertRecipe(eg, M, H, I, GS, unfuse, recipeName);
  std::printf("[OK] %s compiled. recipe=%p  workspace=%.2f MB\n",
              recipeName, (void*)eg.recipe, eg.wsSize / 1048576.0);
}

void destroyExpert(ExpertGraph& eg) { hpu_moe_ext::destroyExpertRecipe(eg); }

// --------------------------------------------------------------------------
// Host-side deterministic inputs + CPU reference.
// --------------------------------------------------------------------------
struct HostData {
  // packed weights (int32) and unpacked codes (for reference), scales (bf16 bits),
  // x (bf16 bits), router weight.
  std::vector<int32_t> w13_packed, w13_zp, w2_packed, w2_zp;
  std::vector<uint16_t> w13_scale, w2_scale, x, rw;     // bf16 bit patterns
  // reference unpacked codes [row][col] in PyTorch orientation:
  //   w13 codes: [H][OUT13] ; w2 codes: [I][H]
  std::vector<uint8_t> w13_codes, w2_codes;
  std::vector<uint16_t> out_ref;                        // [M][H] bf16 bits (row-major M outer)
};

// deterministic pseudo-random 0..15
inline uint8_t code_at(uint64_t a, uint64_t b, uint32_t salt) {
  uint64_t z = a * 1315423911u + b * 2654435761u + salt * 40503u;
  z ^= z >> 13; z *= 0x5bd1e995u; z ^= z >> 15;
  return (uint8_t)(z & 0xF);
}

// ---------------------------------------------------------------------------
// LAYOUT CONVENTION (the one rule that makes host<->device byte copies correct):
// every device tensor declared make2d(g, name, d0, d1, ...) has d0 == FCD ==
// the CONTIGUOUS/innermost dim. So a logical element (fcd, outer) lives at the
// linear (raw-byte) index  outer*d0 + fcd.  We store + read EVERY uploaded /
// downloaded host buffer in exactly that order so synMemCopyAsync (a raw byte
// copy) lands each value at the position the device's geometry expects. The
// macro FIDX(fcd, outer, d0) is that index, used identically in makeHost,
// cpuReference, and the readback compare. Logical names below match the graph:
//   w13_packed [OUT13PK, H]  : (pk, in)
//   w13_scale  [OUT13, G13]  : (out, group)     group of an `in` = in/GS
//   w2_packed  [H_PK, I]     : (pk, in)
//   w2_scale   [H, G2]       : (out, group)
//   x          [H, M]        : (hidden, token)
//   out        [H, M]        : (hidden, token)
// ---------------------------------------------------------------------------
#define FIDX(fcd, outer, d0) ((uint64_t)(outer) * (uint64_t)(d0) + (uint64_t)(fcd))

void makeHost(HostData& hd, uint64_t M, uint64_t H, uint64_t I, uint64_t GS) {
  const uint64_t OUT13   = 2 * I;
  const uint64_t G13     = H / GS;        // groups along contraction H for w13
  const uint64_t G2      = I / GS;        // groups along contraction I for w2
  const uint64_t OUT13PK = OUT13 / 8;
  const uint64_t H_PK    = H / 8;

  // ---- w13 codes: logical (out, in) ; kept in a private [in][out] array purely for
  // the reference (NOT uploaded). codes_w13[in*OUT13 + out].
  hd.w13_codes.assign(H * OUT13, 0);
  for (uint64_t in = 0; in < H; ++in)
    for (uint64_t o = 0; o < OUT13; ++o)
      hd.w13_codes[in * OUT13 + o] = code_at(o, in, 13);
  // w13_packed device [OUT13PK, H] (FCD=OUT13PK): element (pk, in) packs the 8 out-
  // nibbles out=pk*8..pk*8+7 at fixed in. Stored FCD-linear at FIDX(pk, in, OUT13PK).
  hd.w13_packed.assign(OUT13PK * H, 0);
  for (uint64_t pk = 0; pk < OUT13PK; ++pk)
    for (uint64_t in = 0; in < H; ++in) {
      uint8_t c8[8];
      for (int n = 0; n < 8; ++n) c8[n] = hd.w13_codes[in * OUT13 + (pk * 8 + n)];
      hd.w13_packed[FIDX(pk, in, OUT13PK)] = pack8(c8);
    }
  // w13_scale device [OUT13, G13] (FCD=OUT13): element (out, group). FCD-linear.
  hd.w13_scale.assign(OUT13 * G13, 0);
  for (uint64_t o = 0; o < OUT13; ++o)
    for (uint64_t gp = 0; gp < G13; ++gp)
      hd.w13_scale[FIDX(o, gp, OUT13)] = f32_to_bf16(0.01f + 0.003f * (float)((o + gp) % 7));
  // w13_zp device [OUT13PK, G13]: packed symmetric zero (code 8) = 0x88888888 (uniform).
  hd.w13_zp.assign(OUT13PK * G13, (int32_t)0x88888888u);

  // ---- w2 codes: logical (out=hidden, in=intermediate). codes_w2[in*H + out].
  hd.w2_codes.assign(I * H, 0);
  for (uint64_t in = 0; in < I; ++in)
    for (uint64_t o = 0; o < H; ++o)
      hd.w2_codes[in * H + o] = code_at(o, in, 2);
  // w2_packed device [H_PK, I] (FCD=H_PK): element (pk, in) packs out=pk*8..pk*8+7.
  hd.w2_packed.assign(H_PK * I, 0);
  for (uint64_t pk = 0; pk < H_PK; ++pk)
    for (uint64_t in = 0; in < I; ++in) {
      uint8_t c8[8];
      for (int n = 0; n < 8; ++n) c8[n] = hd.w2_codes[in * H + (pk * 8 + n)];
      hd.w2_packed[FIDX(pk, in, H_PK)] = pack8(c8);
    }
  // w2_scale device [H, G2] (FCD=H): element (out, group). FCD-linear.
  hd.w2_scale.assign(H * G2, 0);
  for (uint64_t o = 0; o < H; ++o)
    for (uint64_t gp = 0; gp < G2; ++gp)
      hd.w2_scale[FIDX(o, gp, H)] = f32_to_bf16(0.008f + 0.002f * (float)((o + gp) % 5));
  hd.w2_zp.assign(H_PK * G2, (int32_t)0x88888888u);

  // ---- x device [H, M] (FCD=H): element (hidden, token). FCD-linear.
  hd.x.assign(H * M, 0);
  for (uint64_t hh = 0; hh < H; ++hh)
    for (uint64_t m = 0; m < M; ++m)
      hd.x[FIDX(hh, m, H)] = f32_to_bf16(0.05f * std::sin(0.01f * (float)(hh + 1) + 0.3f * (float)m));
  // ---- router weight. Same scalar value either way; sized [1] (broadcast) or
  // [H*M] (full operand) to match the graph's rw tensor under -DRW_FULL.
#ifdef RW_FULL
  hd.rw.assign(H * M, f32_to_bf16(0.75f));
#else
  hd.rw.assign(1, f32_to_bf16(0.75f));
#endif
}

// CPU reference in float; dequant rule = (code - 8) * scale_group (symmetric zp=8).
// Reads the SAME host buffers the device gets, using the SAME FIDX layout, so a
// match proves the device graph computes this exact expression. out_ref is stored
// FCD-linear [H, M] (FIDX(hh, m, H)) to compare 1:1 with the read-back `out`.
//   gu[out,m]  = sum_in W13f[out,in] * x[in,m] ,  W13f = (code13[out,in]-8)*s13[out, in/GS]
//   gate=gu[0:I], up=gu[I:2I] ; h[i,m] = silu(gate[i,m]) * up[i,m]
//   y[hh,m]    = sum_i  W2f[hh,i]  * h[i,m]  ,  W2f  = (code2[hh,i]-8)*s2[hh, i/GS]
//   out[hh,m]  = y[hh,m] * rw
void cpuReference(HostData& hd, uint64_t M, uint64_t H, uint64_t I, uint64_t GS) {
  const uint64_t OUT13 = 2 * I;
  const float rw = bf16_to_f32(hd.rw[0]);
  hd.out_ref.assign(M * H, 0);

  auto xf = [&](uint64_t hh, uint64_t m) { return bf16_to_f32(hd.x[FIDX(hh, m, H)]); };
  auto s13 = [&](uint64_t o, uint64_t in) {
    return bf16_to_f32(hd.w13_scale[FIDX(o, in / GS, OUT13)]);
  };
  auto c13 = [&](uint64_t o, uint64_t in) { return hd.w13_codes[in * OUT13 + o]; };
  auto s2 = [&](uint64_t o, uint64_t in) {
    return bf16_to_f32(hd.w2_scale[FIDX(o, in / GS, H)]);
  };
  auto c2 = [&](uint64_t o, uint64_t in) { return hd.w2_codes[in * H + o]; };

  std::vector<float> h(I * M);  // [I, M] FCD-linear
  for (uint64_t m = 0; m < M; ++m) {
    std::vector<float> gu(OUT13, 0.0f);
    for (uint64_t o = 0; o < OUT13; ++o) {
      float acc = 0.0f;
      for (uint64_t in = 0; in < H; ++in)
        acc += ((float)c13(o, in) - 8.0f) * s13(o, in) * xf(in, m);
      gu[o] = acc;
    }
    for (uint64_t i = 0; i < I; ++i) {
      float gate = gu[i], upv = gu[I + i];
      float sil = gate * (1.0f / (1.0f + std::exp(-gate)));
      h[FIDX(i, m, I)] = sil * upv;
    }
  }
  for (uint64_t m = 0; m < M; ++m)
    for (uint64_t hh = 0; hh < H; ++hh) {
      float acc = 0.0f;
      for (uint64_t i = 0; i < I; ++i)
        acc += ((float)c2(hh, i) - 8.0f) * s2(hh, i) * h[FIDX(i, m, I)];
      hd.out_ref[FIDX(hh, m, H)] = f32_to_bf16(acc * rw);
    }
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t M  = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 32;
  uint64_t H  = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 4096;
  uint64_t I  = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 1408;
  uint64_t GS = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 128;

  const bool compileOnly = std::getenv("FUSED_COMPILE_ONLY") != nullptr;
  const bool unfuseEnv   = std::getenv("UNFUSE") != nullptr;
  const bool latencyAB   = std::getenv("LATENCY_AB") != nullptr;

  const uint64_t OUT13 = 2 * I;
  if (H % GS != 0 || I % GS != 0 || OUT13 % 8 != 0 || H % 8 != 0) {
    std::fprintf(stderr,
                 "[FATAL] need H%%GS==0, I%%GS==0, (2I)%%8==0, H%%8==0 "
                 "(H=%llu I=%llu GS=%llu)\n",
                 (unsigned long long)H, (unsigned long long)I, (unsigned long long)GS);
    return 2;
  }

  std::printf("=== fused_moe_full ===\n");
  std::printf("  M=%llu  H=%llu  I=%llu (OUT13=%llu)  GS=%llu\n",
              (unsigned long long)M, (unsigned long long)H, (unsigned long long)I,
              (unsigned long long)OUT13, (unsigned long long)GS);
  const double w13_bf16_mb = (double)H * OUT13 * 2 / 1048576.0;
  const double w13_int4_mb = (double)H * OUT13 / 2 / 1048576.0;
  const double w2_bf16_mb  = (double)I * H * 2 / 1048576.0;
  const double w2_int4_mb  = (double)I * H / 2 / 1048576.0;
  std::printf("  w13: bf16=%.1f MB int4=%.1f MB | w2: bf16=%.1f MB int4=%.1f MB\n",
              w13_bf16_mb, w13_int4_mb, w2_bf16_mb, w2_int4_mb);
  std::printf("  total bf16 W = %.1f MB (the round-trip the fusion deletes); "
              "total int4 W = %.1f MB (HBM target)\n",
              w13_bf16_mb + w2_bf16_mb, w13_int4_mb + w2_int4_mb);
  std::printf("  mode: compileOnly=%d  UNFUSE=%d  LATENCY_AB=%d\n",
              (int)compileOnly, (int)unfuseEnv, (int)latencyAB);

  SYN_CHECK(synInitialize());

  // ---- device ----
  synDeviceId dev = 0;
  bool haveDev = false;
  if (!compileOnly) {
    synStatus s = synDeviceAcquire(&dev, nullptr);
    if (s != synSuccess) {
      std::fprintf(stderr, "[WARN] synDeviceAcquire failed (synStatus=%d) -> compile-only.\n", (int)s);
    } else {
      haveDev = true;
      uint64_t sramSz = 0;
      synDeviceAttribute a2 = DEVICE_ATTRIBUTE_SRAM_SIZE;
      synDeviceGetAttribute(&sramSz, &a2, 1, dev);
      std::printf("  device acquired. SRAM=%.1f MB\n", sramSz / 1048576.0);
    }
  }

  // =====================================================================
  // COMPILE-ONLY path: build fused (and, for parity, unfused) + report.
  // =====================================================================
  if (compileOnly || !haveDev) {
    ExpertGraph fused;
    buildExpert(fused, M, H, I, GS, /*unfuse=*/false, "fused_moe_full");
    std::printf("\n=== COMPILE-TIME RESIDENCY HINT (fused) ===\n");
    std::printf("  fused workspace = %.2f MB ; total bf16 W = %.2f MB\n",
                fused.wsSize / 1048576.0, w13_bf16_mb + w2_bf16_mb);
    if (fused.wsSize / 1048576.0 >= (w13_bf16_mb + w2_bf16_mb) * 0.9)
      std::printf("  >>> HINT: workspace ~>= bf16 W => dequant likely SPILLED (fusion may have FAILED).\n");
    else
      std::printf("  >>> HINT: workspace << bf16 W => consistent with SRAM-resident bundles (fusion likely WORKING).\n");
    destroyExpert(fused);

    ExpertGraph unf;
    buildExpert(unf, M, H, I, GS, /*unfuse=*/true, "fused_moe_full_unfused");
    std::printf("  unfused workspace = %.2f MB (expect >= bf16 W; dequant forced PERSISTENT)\n",
                unf.wsSize / 1048576.0);
    destroyExpert(unf);

    std::printf("\n[compile-only] both graphs built + compiled OK; no device launch.\n");
    synDestroy();
    return 0;
  }

  // =====================================================================
  // ON-DEVICE: build the requested variant(s), upload inputs, launch.
  // =====================================================================
  HostData hd;
  makeHost(hd, M, H, I, GS);
  cpuReference(hd, M, H, I, GS);

  synStreamHandle stream = nullptr;
  SYN_CHECK(synStreamCreateGeneric(&stream, dev, 0));

  // device buffer sizes (bytes)
  const uint64_t OUT13PK = OUT13 / 8, G13 = H / GS, H_PK = H / 8, G2 = I / GS;
  const uint64_t b_x        = H * M * 2;
  const uint64_t b_w13p     = OUT13PK * H * 4;      // == H*OUT13/2
  const uint64_t b_w13s     = OUT13 * G13 * 2;
  const uint64_t b_w13z     = OUT13PK * G13 * 4;
  const uint64_t b_w2p      = H_PK * I * 4;         // == I*H/2
  const uint64_t b_w2s      = H * G2 * 2;
  const uint64_t b_w2z      = H_PK * G2 * 4;
#ifdef RW_FULL
  const uint64_t b_rw       = H * M * 2;   // full [H,M] router-weight operand
#else
  const uint64_t b_rw       = 1 * 1 * 2;   // scalar [1,1], broadcast in the mult
#endif
  const uint64_t b_out      = H * M * 2;

  auto devAlloc = [&](uint64_t bytes) -> uint64_t {
    uint64_t p = 0; SYN_CHECK(synDeviceMalloc(dev, bytes, 0, 0, &p)); return p;
  };
  uint64_t d_x    = devAlloc(b_x);
  uint64_t d_w13p = devAlloc(b_w13p);
  uint64_t d_w13s = devAlloc(b_w13s);
  uint64_t d_w13z = devAlloc(b_w13z);
  uint64_t d_w2p  = devAlloc(b_w2p);
  uint64_t d_w2s  = devAlloc(b_w2s);
  uint64_t d_w2z  = devAlloc(b_w2z);
  uint64_t d_rw   = devAlloc(b_rw);
  uint64_t d_out  = devAlloc(b_out);

  auto upload = [&](uint64_t dptr, const void* src, uint64_t bytes) {
    void* h = nullptr;
    SYN_CHECK(synHostMalloc(dev, bytes, 0, &h));
    std::memcpy(h, src, bytes);
    SYN_CHECK(synMemCopyAsync(stream, (uint64_t)h, bytes, dptr, HOST_TO_DRAM));
    SYN_CHECK(synStreamSynchronize(stream));
    SYN_CHECK(synHostFree(dev, h, 0));
  };
  upload(d_x,    hd.x.data(),          b_x);
  upload(d_w13p, hd.w13_packed.data(), b_w13p);
  upload(d_w13s, hd.w13_scale.data(),  b_w13s);
  upload(d_w13z, hd.w13_zp.data(),     b_w13z);
  upload(d_w2p,  hd.w2_packed.data(),  b_w2p);
  upload(d_w2s,  hd.w2_scale.data(),   b_w2s);
  upload(d_w2z,  hd.w2_zp.data(),      b_w2z);
  upload(d_rw,   hd.rw.data(),         b_rw);

  auto makeLT = [&](std::vector<synLaunchTensorInfo>& lt) {
    auto add = [&](const char* nm, uint64_t a) {
      synLaunchTensorInfo i; std::memset(&i, 0, sizeof(i));
      i.tensorName = nm; i.pTensorAddress = a; i.tensorType = DATA_TENSOR;
      lt.push_back(i);
    };
    add("x", d_x);
    add("w13_packed", d_w13p); add("w13_scale", d_w13s); add("w13_zp", d_w13z);
    add("w2_packed",  d_w2p);  add("w2_scale",  d_w2s);  add("w2_zp",  d_w2z);
    add("router_weight", d_rw);
    add("out", d_out);
  };

  // ---- numeric validation against one variant (default fused, or UNFUSE) ----
  auto runValidate = [&](bool unfuse) -> int {
    ExpertGraph eg;
    buildExpert(eg, M, H, I, GS, unfuse,
                unfuse ? "fused_moe_full_unfused" : "fused_moe_full");
    uint64_t d_ws = eg.wsSize ? devAlloc(eg.wsSize) : 0;

    std::vector<synLaunchTensorInfo> lt; makeLT(lt);
    // zero out
    upload(d_out, std::vector<uint16_t>(H * M, 0).data(), b_out);
    SYN_CHECK(synLaunch(stream, lt.data(), (uint32_t)lt.size(), d_ws, eg.recipe, SYN_FLAGS_TENSOR_NAME));
    SYN_CHECK(synStreamSynchronize(stream));

    // read back out [H, M] (FCD=H). Device stores (hh,m) at FIDX(hh,m,H); out_ref
    // uses the identical FCD-linear layout, so compare element-for-element.
    std::vector<uint16_t> got(H * M, 0);
    void* hbuf = nullptr;
    SYN_CHECK(synHostMalloc(dev, b_out, 0, &hbuf));
    SYN_CHECK(synMemCopyAsync(stream, d_out, b_out, (uint64_t)hbuf, DRAM_TO_HOST));
    SYN_CHECK(synStreamSynchronize(stream));
    std::memcpy(got.data(), hbuf, b_out);
    SYN_CHECK(synHostFree(dev, hbuf, 0));

    double maxAbs = 0.0, maxRel = 0.0, refNorm = 0.0, gotNorm = 0.0, diffNorm = 0.0;
    uint64_t argm = 0;
    for (uint64_t hh = 0; hh < H; ++hh)
      for (uint64_t m = 0; m < M; ++m) {
        uint64_t idx = FIDX(hh, m, H);
        float g = bf16_to_f32(got[idx]);
        float r = bf16_to_f32(hd.out_ref[idx]);
        double a = std::fabs((double)g - (double)r);
        double rel = a / (std::fabs((double)r) + 1e-6);
        if (a > maxAbs) { maxAbs = a; argm = idx; }
        if (rel > maxRel) maxRel = rel;
        refNorm += (double)r * r; gotNorm += (double)g * g; diffNorm += a * a;
      }
    refNorm = std::sqrt(refNorm); gotNorm = std::sqrt(gotNorm); diffNorm = std::sqrt(diffNorm);
    // NORM-RELATIVE error is the correct bf16 gate. Per-element maxRel explodes where the
    // reference element is ~0 (a 2e-5 ref with a 2e-3 abs diff = maxRel 80, a pure artifact),
    // so gate on ||got-ref|| / ||ref|| instead. maxRel/maxAbs stay printed for info.
    const double relNorm = diffNorm / (refNorm + 1e-9);
    std::printf("\n=== NUMERIC VALIDATION (%s) ===\n", unfuse ? "UNFUSED" : "FUSED");
    std::printf("  |ref|=%.4f  |got|=%.4f  relNorm=%.4e  maxAbs=%.4e  maxRel=%.4e(elt,info)  (worst idx=%llu)\n",
                refNorm, gotNorm, relNorm, maxAbs, maxRel, (unsigned long long)argm);
    const double TOL = 1e-2;
    int rc = 0;
    if (!(relNorm <= TOL)) {
      std::fprintf(stderr,
          "  >>> FAIL: relNorm %.4e > tol %.2e. The fused graph does NOT match the CPU\n"
          "      reference. Most likely: gemm orientation (kTransposeA/B), the slice_axis\n"
          "      split halves (gate vs up swapped), the silu decomposition, or the dequant\n"
          "      group/scale indexing. Inspect the worst index above.\n", relNorm, TOL);
      rc = 1;
    } else {
      std::printf("  >>> PASS: fused graph matches CPU reference (relNorm %.2e within bf16 tol).\n", relNorm);
    }
    if (d_ws) synDeviceFree(dev, d_ws, 0);
    destroyExpert(eg);
    return rc;
  };

  // ---- latency timing of one already-built recipe ----
  auto timeRecipe = [&](ExpertGraph& eg, uint64_t d_ws, const char* tag) -> double {
    std::vector<synLaunchTensorInfo> lt; makeLT(lt);
    synEventHandle e0 = nullptr, e1 = nullptr;
    SYN_CHECK(synEventCreate(&e0, dev, /*EVENT_COLLECT_TIME=*/1));  // flags=0 -> synEventElapsedTime synStatus=1
    SYN_CHECK(synEventCreate(&e1, dev, /*EVENT_COLLECT_TIME=*/1));
    const int WARM = 20, N = 200;
    for (int i = 0; i < WARM; ++i)
      SYN_CHECK(synLaunch(stream, lt.data(), (uint32_t)lt.size(), d_ws, eg.recipe, SYN_FLAGS_TENSOR_NAME));
    SYN_CHECK(synStreamSynchronize(stream));
    SYN_CHECK(synEventRecord(e0, stream));
    for (int i = 0; i < N; ++i)
      SYN_CHECK(synLaunch(stream, lt.data(), (uint32_t)lt.size(), d_ws, eg.recipe, SYN_FLAGS_TENSOR_NAME));
    SYN_CHECK(synEventRecord(e1, stream));
    SYN_CHECK(synStreamSynchronize(stream));
    uint64_t ns = 0;
    SYN_CHECK(synEventElapsedTime(&ns, e0, e1));
    synEventDestroy(e0); synEventDestroy(e1);
    double ms_iter = (double)ns / 1e6 / (double)N;
    double toks = (double)M / (ms_iter / 1000.0);   // tokens/sec for this 1-expert layer call
    std::printf("  [%s] %.4f ms/iter  (%.1f tok/s-equiv for M=%llu over 1 expert; N=%d)\n",
                tag, ms_iter, toks, (unsigned long long)M, N);
    return ms_iter;
  };

  int rc = 0;

  if (latencyAB) {
    // Build both, validate fused (correctness), then A/B time both.
    rc |= runValidate(/*unfuse=*/false);

    ExpertGraph fused; buildExpert(fused, M, H, I, GS, false, "fused_moe_full");
    uint64_t ws_f = fused.wsSize ? devAlloc(fused.wsSize) : 0;
    ExpertGraph unf;   buildExpert(unf,   M, H, I, GS, true,  "fused_moe_full_unfused");
    uint64_t ws_u = unf.wsSize ? devAlloc(unf.wsSize) : 0;

    std::printf("\n=== LATENCY A/B (fused SRAM-resident vs unfused bf16-HBM round-trip) ===\n");
    std::printf("  fused workspace=%.2f MB | unfused workspace=%.2f MB\n",
                fused.wsSize / 1048576.0, unf.wsSize / 1048576.0);
    double t_f = timeRecipe(fused, ws_f, "FUSED  ");
    double t_u = timeRecipe(unf,   ws_u, "UNFUSED");
    std::printf("\n  >>> speedup (unfused/fused) = %.2fx  (fused saves the %.1f MB bf16 HBM round-trip)\n",
                t_u / t_f, w13_bf16_mb + w2_bf16_mb);
    std::printf("      (fused should approach the int4-bandwidth limit; unfused pays bf16 HBM traffic.)\n");

    if (ws_f) synDeviceFree(dev, ws_f, 0);
    if (ws_u) synDeviceFree(dev, ws_u, 0);
    destroyExpert(fused); destroyExpert(unf);
  } else {
    // single-variant numeric validation (default fused; UNFUSE=1 for the other)
    rc |= runValidate(unfuseEnv);
  }

  // cleanup
  synDeviceFree(dev, d_out, 0);
  synDeviceFree(dev, d_rw, 0);
  synDeviceFree(dev, d_w2z, 0);
  synDeviceFree(dev, d_w2s, 0);
  synDeviceFree(dev, d_w2p, 0);
  synDeviceFree(dev, d_w13z, 0);
  synDeviceFree(dev, d_w13s, 0);
  synDeviceFree(dev, d_w13p, 0);
  synDeviceFree(dev, d_x, 0);
  synStreamDestroy(stream);
  synDestroy();

  std::printf("\n[done] fused_moe_full %s (rc=%d).\n",
              rc == 0 ? "OK" : "FAILED", rc);
  return rc;
}
