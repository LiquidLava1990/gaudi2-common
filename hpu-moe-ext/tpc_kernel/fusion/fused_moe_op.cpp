/*
 * fused_moe_op.cpp -- the vLLM BRIDGE: wrap the PROVEN fused int4 MoE Synapse
 * graph (fused_expert_graph.h) as callable torch ops.
 *
 * WRAP APPROACH: B (torch.library op that drives the Synapse graph API directly).
 * -----------------------------------------------------------------------------
 * Approach A (habana::custom_op::UserCustomOpDescriptor / HabanaCustomOpDescriptor)
 * was RULED OUT by reading hpu_custom_op_pt2.h + hpu_custom_op.h: both descriptors
 * bind exactly ONE TPC GUID (a single `std::string guid_` / one `NodeDesc{tpc_guid}`),
 * with one OutputMeta + one FillParams callback. They CANNOT express our multi-node
 * recipe -- two MME `gemm` nodes (MME, not a TPC GUID at all), the dequant complex-
 * guids, slice_axis, and the NON-PERSISTENT dequant outputs the GC must bundle into
 * the MME. The fusion ONLY exists inside an explicitly-built Synapse graph (op-level
 * auto-fusion is dead: convert_from_uint4 is opaque). So the torch op must BUILD and
 * synLaunch that graph itself. That is approach B, implemented here.
 *
 * INTEROP FACTS (verified from libSynapse.so exports + habana_frameworks headers):
 *   - libSynapse.so exports synLaunch / synDeviceGetCount / synStreamCreateGeneric /
 *     synDeviceMalloc / synGraphCompile / synTensorHandleCreate as public T symbols,
 *     and the habana torch backend dynamically loads the SAME symbols -> there is ONE
 *     Synapse singleton + device context shared between torch and this op. So after
 *     `import habana_frameworks.torch` has acquired the device, we DO NOT acquire it
 *     again (a 2nd synDeviceAcquire fails); we use device id 0 (the single acquired
 *     Gaudi2) and the caller's compute stream.
 *   - A torch HPU stream wraps a synStreamHandle (InternalStreamHandle*); Python
 *     exposes it as `torch.hpu.current_stream().hpu_stream` (an int). We take that
 *     int64 as an op arg and synLaunch on it -> ordering vs torch ops is automatic,
 *     no cross-stream event juggling. (stream_handle==0 -> the op lazily creates +
 *     caches its OWN stream as a fallback; correct but unordered vs torch, so the
 *     Python wrapper mark_step()s around it.)
 *   - A PyTorch *contiguous* tensor's data_ptr() is the device VA, row-major with the
 *     LAST dim contiguous == the Synapse FCD. The per-expert repacked tensors
 *     (w13_qweight[e] int32 [H,2I/8], w13_scales[e] bf16 [H/GS,2I], ...) therefore
 *     bind DIRECTLY to the graph's persistent inputs with NO transpose/repack.
 *
 * OPS REGISTERED (library: hpu_moe_ext):
 *   fused_expert(x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
 *                router_weight, stream_handle) -> Tensor
 *       The BRIDGE PROOF. ONE expert. Builds/caches the recipe per (M,H,I,GS),
 *       binds the torch tensors' data_ptr()s, synLaunches, returns out [M,H] bf16.
 *       router_weight is a [1,1] bf16 HPU tensor (the scalar w_k for this expert);
 *       the K-expert accumulate is done by the caller (Python fused_moe).
 *
 * The K-expert routed `fused_moe(...)` op is implemented in Python
 * (fused_moe_op.py) on top of fused_expert, reusing the PROVEN _routed_moe
 * gather/scatter/topk/accumulate (HPU-graph-safe). See that file + the scaffold
 * note at the bottom of this file for the all-C++ variant and what it needs.
 *
 * Build CPU-only:  bash build_op.sh   (inside vllm-gaudi:next; links libSynapse +
 * the torch/habana backend, host x86 g++). Run ON THE HPU BOX ONLY via the test.
 */
#include "fused_expert_graph.h"

#include "synapse_api.h"
#include "synapse_common_types.h"

#include <torch/library.h>
#include <ATen/ATen.h>          // at::empty + tensor factory functions
#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace hpu_moe_ext {
namespace {

// ---- recipe cache: one compiled ExpertRecipe per unique (M,H,I,GS) -----------
// Keyed by the four shape dims; the graph topology is identical across experts,
// so all experts of a layer (same M,H,I,GS) share ONE recipe + ONE workspace.
using ShapeKey = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;

struct CachedRecipe {
  ExpertRecipe er;
  uint64_t d_ws = 0;          // device workspace VA, 0 if wsSize==0
  at::Tensor ws_tensor;       // torch-owned workspace backing d_ws: raw synDeviceMalloc
                              // fails under torch device ownership -> use torch's pool
};

std::mutex g_cache_mtx;
std::map<ShapeKey, CachedRecipe> g_cache;

// device id of the single Gaudi2 the habana backend already acquired.
// We DO NOT acquire (torch holds it); just confirm one exists. Cached.
synDeviceId resolveDeviceId() {
  uint32_t count = 0;
  synStatus s = synDeviceGetCount(&count);
  TORCH_CHECK(s == synSuccess,
              "[fused_moe_op] synDeviceGetCount failed (synStatus=", (int)s,
              "); is the HPU acquired? (import habana_frameworks.torch first)");
  TORCH_CHECK(count >= 1,
              "[fused_moe_op] no HPU device acquired (count=", count,
              "). Run on the box after importing habana_frameworks.torch.");
  return (synDeviceId)0;  // single-device serving: the acquired Gaudi2 is id 0
}

// Fallback stream (used only when the caller passes stream_handle==0). Created
// once on first need, on device 0, and reused. NOT torch-ordered.
synStreamHandle g_fallback_stream = nullptr;
std::mutex g_stream_mtx;
synStreamHandle fallbackStream(synDeviceId dev) {
  std::lock_guard<std::mutex> lk(g_stream_mtx);
  if (!g_fallback_stream) {
    FEG_SYN_CHECK(synStreamCreateGeneric(&g_fallback_stream, dev, 0));
  }
  return g_fallback_stream;
}

// get-or-build the cached recipe (+workspace) for these dims.
CachedRecipe& getRecipe(synDeviceId dev, uint64_t M, uint64_t H, uint64_t I,
                        uint64_t GS) {
  ShapeKey key{M, H, I, GS};
  std::lock_guard<std::mutex> lk(g_cache_mtx);
  auto it = g_cache.find(key);
  if (it != g_cache.end()) return it->second;

  CachedRecipe cr;
  // recipe name must be unique per shape (the GC names the recipe by it).
  std::string nm = "hpu_moe_ext_fused_expert_" + std::to_string(M) + "_" +
                   std::to_string(H) + "_" + std::to_string(I) + "_" +
                   std::to_string(GS);
  buildExpertRecipe(cr.er, M, H, I, GS, /*unfuse=*/false, nm.c_str());
  (void)dev;  // device is torch-owned; we no longer raw-malloc on it
  if (cr.er.wsSize) {
    // Workspace MUST come from torch's allocator: torch owns the HBM pool, so a raw
    // synDeviceMalloc here fails (synStatus 26). Allocate a uint8 HPU tensor, keep it
    // alive in the cache, and pass its VA as the synLaunch workspace.
    cr.ws_tensor = at::empty({(int64_t)cr.er.wsSize},
                             at::TensorOptions().dtype(at::kByte).device(at::kHPU));
    cr.d_ws = (uint64_t)cr.ws_tensor.data_ptr();
  }
  auto res = g_cache.emplace(key, std::move(cr));
  return res.first->second;
}

// shape/dtype/contiguity validation helper for an input tensor.
void checkInput(const at::Tensor& t, const char* nm, c10::ScalarType dt,
                int64_t d0, int64_t d1) {
  TORCH_CHECK(t.is_hpu(), "[fused_expert] ", nm, " must be on HPU, got ",
              t.device());
  TORCH_CHECK(t.is_contiguous(), "[fused_expert] ", nm,
              " must be contiguous (data_ptr binds to a row-major Synapse FCD "
              "tensor; clone()/contiguous() it first).");
  TORCH_CHECK(t.scalar_type() == dt, "[fused_expert] ", nm, " dtype mismatch: expected ",
              dt, " got ", t.scalar_type());
  TORCH_CHECK(t.dim() == 2, "[fused_expert] ", nm, " must be 2D, got dim=", t.dim());
  TORCH_CHECK(t.size(0) == d0 && t.size(1) == d1, "[fused_expert] ", nm,
              " shape mismatch: expected [", d0, ",", d1, "] got [", t.size(0),
              ",", t.size(1), "]");
}

// =============================================================================
// fused_expert: ONE expert. The bridge proof.
// =============================================================================
// x            bf16  [M, H]
// w13_packed   int32 [H, 2I/8]
// w13_scale    bf16  [H/GS, 2I]
// w13_zp       int32 [H/GS, 2I/8]
// w2_packed    int32 [I, H/8]
// w2_scale     bf16  [I/GS, H]
// w2_zp        int32 [I/GS, H/8]
// router_weight bf16 [1, 1]      (the scalar w_k for this expert)
// stream_handle int64            (torch.hpu.current_stream().hpu_stream; 0 -> fallback)
// -> out       bf16  [M, H]
at::Tensor fused_expert(const at::Tensor& x, const at::Tensor& w13_packed,
                        const at::Tensor& w13_scale, const at::Tensor& w13_zp,
                        const at::Tensor& w2_packed, const at::Tensor& w2_scale,
                        const at::Tensor& w2_zp, const at::Tensor& router_weight,
                        int64_t stream_handle) {
  // ---- derive shapes from x + w13 ----
  TORCH_CHECK(x.is_hpu() && x.dim() == 2 && x.scalar_type() == c10::ScalarType::BFloat16,
              "[fused_expert] x must be HPU bf16 [M,H]");
  const int64_t M = x.size(0);
  const int64_t H = x.size(1);
  TORCH_CHECK(w13_scale.dim() == 2, "[fused_expert] w13_scale must be 2D [H/GS, 2I]");
  const int64_t OUT13 = w13_scale.size(1);             // 2I
  TORCH_CHECK(OUT13 % 2 == 0, "[fused_expert] w13_scale.size(1)=2I must be even, got ", OUT13);
  const int64_t I = OUT13 / 2;
  TORCH_CHECK(w13_scale.size(0) > 0 && H % w13_scale.size(0) == 0,
              "[fused_expert] H must be divisible by #w13 groups");
  const int64_t GS = H / w13_scale.size(0);            // group size (from w13: groups = H/GS)

  TORCH_CHECK(H % GS == 0 && I % GS == 0 && OUT13 % 8 == 0 && H % 8 == 0,
              "[fused_expert] need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H=", H,
              " I=", I, " GS=", GS, ")");

  const int64_t G13 = H / GS, G2 = I / GS;
  // ---- validate every input against the proven geometry ----
  checkInput(x,          "x",          c10::ScalarType::BFloat16, M, H);
  checkInput(w13_packed, "w13_packed", c10::ScalarType::Int,      H, OUT13 / 8);
  checkInput(w13_scale,  "w13_scale",  c10::ScalarType::BFloat16, G13, OUT13);
  checkInput(w13_zp,     "w13_zp",     c10::ScalarType::Int,      G13, OUT13 / 8);
  checkInput(w2_packed,  "w2_packed",  c10::ScalarType::Int,      I, H / 8);
  checkInput(w2_scale,   "w2_scale",   c10::ScalarType::BFloat16, G2, H);
  checkInput(w2_zp,      "w2_zp",      c10::ScalarType::Int,      G2, H / 8);
  TORCH_CHECK(router_weight.is_hpu() && router_weight.scalar_type() == c10::ScalarType::BFloat16,
              "[fused_expert] router_weight must be HPU bf16");
  TORCH_CHECK(router_weight.numel() == 1,
              "[fused_expert] router_weight must be a single element [1,1] (this build "
              "uses the proven [1,1]->[H,M] broadcast); got numel=", router_weight.numel());
  TORCH_CHECK(router_weight.is_contiguous(), "[fused_expert] router_weight must be contiguous");

  synDeviceId dev = resolveDeviceId();
  CachedRecipe& cr = getRecipe(dev, (uint64_t)M, (uint64_t)H, (uint64_t)I, (uint64_t)GS);

  // ---- output tensor (torch-owned HBM; its data_ptr binds to `out`) ----
  at::Tensor out = at::empty({M, H}, x.options());  // bf16 HPU, contiguous [M,H]

  // ---- bind torch device addresses to the recipe's persistent tensors ----
  std::vector<synLaunchTensorInfo> lt;
  lt.reserve(9);
  auto add = [&](const char* nm, const at::Tensor& t) {
    synLaunchTensorInfo i; std::memset(&i, 0, sizeof(i));
    i.tensorName = nm;
    i.pTensorAddress = (uint64_t)t.data_ptr();
    i.tensorType = DATA_TENSOR;
    lt.push_back(i);
  };
  add(ExpertTensorNames::kX,         x);
  add(ExpertTensorNames::kW13Packed, w13_packed);
  add(ExpertTensorNames::kW13Scale,  w13_scale);
  add(ExpertTensorNames::kW13Zp,     w13_zp);
  add(ExpertTensorNames::kW2Packed,  w2_packed);
  add(ExpertTensorNames::kW2Scale,   w2_scale);
  add(ExpertTensorNames::kW2Zp,      w2_zp);
  add(ExpertTensorNames::kRouterW,   router_weight);
  add(ExpertTensorNames::kOut,       out);

  // ---- stream: caller's torch compute stream (ordered), or our fallback ----
  synStreamHandle stream =
      stream_handle ? reinterpret_cast<synStreamHandle>(stream_handle)
                    : fallbackStream(dev);

  FEG_SYN_CHECK(synLaunch(stream, lt.data(), (uint32_t)lt.size(), cr.d_ws,
                          cr.er.recipe, SYN_FLAGS_TENSOR_NAME));
  // No synStreamSynchronize here: the launch is enqueued on the caller's stream;
  // torch's own stream semantics (and the Python wrapper's mark_step/.cpu()) order
  // any subsequent consumption. (If stream_handle==0 / fallback, the Python wrapper
  // must synchronize before reading `out` -- it does.)
  return out;
}

}  // namespace
}  // namespace hpu_moe_ext

// ---- torch registration ------------------------------------------------------
TORCH_LIBRARY(hpu_moe_ext_fused, m) {
  m.def(
      "fused_expert(Tensor x, Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
      "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, Tensor router_weight, "
      "int stream_handle) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu_moe_ext_fused, HPU, m) {
  m.impl("fused_expert", TORCH_FN(hpu_moe_ext::fused_expert));
}

/*
 * SCAFFOLD -- all-C++ fused_moe(x, w13_packed[E,...], ..., topk_ids[T,K],
 *             topk_weights[T,K], experts_min) -> [T,H]:
 *
 * The K-expert routed op is delivered in PYTHON (fused_moe_op.py) for now, because
 * the gather/route is the PROVEN, HPU-graph-safe _routed_moe logic (present/scatter/
 * topk -> active, index_select, remap, per-(token,slot) weight) and re-deriving it in
 * C++/ATen here would duplicate ~40 lines of subtle lazy-replay-safe tensor ops with
 * no benefit at T=1,K<=8 (the priority case): the cost is K launches of fused_expert,
 * which dominate the routing math regardless of language.
 *
 * To promote it to a single C++ op later (only worth it to fuse the K launches into
 * ONE recipe, or to drop the per-expert Python overhead at high concurrency):
 *   1. Add  at::Tensor fused_moe(x, w13_packed, w13_scale, w13_zp, w2_packed,
 *            w2_scale, w2_zp, topk_ids, topk_weights, int64 experts_min,
 *            int64 stream_handle)  here.
 *   2. Reproduce _routed_moe's gather with ATen (index_select / scatter / topk are
 *      all ATen-on-HPU) to get `active[K]`, per-expert packed/scale slices (CLONE each
 *      slice: a k>0 view carries a nonzero storage_offset that convert_from_uint4/the
 *      graph reads wrong -- see moe_wna16_hpu.py:560-595), and the per-(token,slot)
 *      router weight w_k[K] (a [1,1] bf16 device tensor per expert).
 *   3. Loop k: fused_expert(...) and accumulate out += expert_out (the router_weight
 *      [1,1] is already folded in by the graph's router_scale mult). Materialize the
 *      index_select results at a graph boundary BEFORE binding their data_ptr()s
 *      (the same mark_step the reference op requires) -- in C++ that means a
 *      synStreamSynchronize after the gather, before the first synLaunch.
 *   4. Build a TRUE multi-expert recipe (gather + K fused experts + weighted sum in
 *      ONE Synapse graph) to delete the K separate launches -- the real perf win, but
 *      a larger graph-construction effort (left for a follow-up).
 */
