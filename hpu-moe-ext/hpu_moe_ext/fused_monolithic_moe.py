"""MONOLITHIC (ALL-E experts) fused int4 MoE for the THROUGHPUT/concurrent regime.

This is the THROUGHPUT-regime sibling of the routed wrapper (fused_expert_routed.py).
Where the routed path gathers the <=K locally-routed experts (the single-stream T==1
win, gated tokens*top_k<=K), the MONOLITHIC path processes ALL E LOCAL experts via the
single-recipe op

    torch.ops.hpu_moe_ext.fused_moe_monolithic(
        x[T,H] bf16,
        w13_packed[E,H,2I//8] int32, w13_scale[E,H//GS,2I] bf16, w13_zp[E,H//GS,2I//8] int32,
        w2_packed[E,I,H//8]  int32, w2_scale[E,I//GS,H]  bf16, w2_zp[E,I//GS,H//8]  int32,
        router_weights[E,T,1] bf16, int chunk) -> [T,H] bf16

(moe_fused_monolithic_backend*.cpp). That op chunks E into `chunk`-expert batched
sub-chains so each chunk's int4 dequant stays SRAM-resident (NON-PERSISTENT -> bundled
into its batch_gemm MME -- the fusion lever), avoiding the STOCK monolithic op's bf16
-> HBM round-trip (the ~99us/token floor at concurrency). It is fed the layer's
PERSISTENT stacked [E,...] nn.Parameters DIRECTLY (no gather), so the held-leaf weights
ARE the recipe inputs (multi-bucket-capture-safe). The ONLY per-call work here is
building the DENSE [E,T,1] router-weight matrix from topk_ids/topk_weights.

=== WHY THIS BEATS STOCK AT CONCURRENCY (the diagnosis) ===
Single-stream (T==1) is host/collective-OVERHEAD-bound (~90% of the token is TP=8
all-reduces + host-sync), so fusion can't help -- that is the routed path's regime, and
it is a GATHER (tokens*top_k<=K). At THROUGHPUT (max-num-seqs>1, T>1) the MoE is
BANDWIDTH-bound and the stock VllmMixtureOfExpertsOpWNA16 dequantizes ALL E experts to
bf16 -> HBM -> a separate mixture_of_experts gemm: ~24MB/expert * E experts streamed
through HBM twice (write the dequant, read it for the gemm). The fused monolithic op
keeps each expert's dequant SRAM-resident and feeds it straight into the gemm -- no
bf16->HBM round-trip. That is where MoE fusion's bandwidth win actually manifests.

=== THE ROUTING (match the stock mixture_of_experts semantics EXACTLY) ===
The stock op computes out[t] = sum over top_k slots s of expert(topk_ids[t,s])(x[t]) *
topk_weights[t,s], for experts in the LOCAL window [0, E) (apply_monolithic already
remaps GLOBAL ids -> LOCAL and zeroes non-local slots). We build the EQUIVALENT dense
[T,E] weight matrix W where

    W[t,e] = sum over slots s with local_id(t,s)==e of safe_w[t,s]

(safe_w zeroes non-local slots), then pass its transpose [E,T,1] to the op. The op
computes sum_e( expert_e(x) * W[:,e] ); a non-routed (t,e) carries W=0 so it contributes
exactly 0 -- numerically identical to the stock gather-and-weight, just dense so the
op can compute every expert in ONE static recipe. The dense scatter is a vectorized
one_hot(local_id) * safe_w sum over slots (NO host sync, NO dynamic shape -- it is the
SAME class of vectorized routing the routed path uses for its per-(token,k) weights,
generalized to all E instead of the gathered K).

=== MULTI-BUCKET CAPTURE SAFETY ===
The op's inputs are: x (the held-leaf x_buf slice, the persistent static-input buffer
_alloc_fused_x_buf builds at weight-load), the layer's stacked weight nn.Parameters
(persistent), the stacked zero-points (the held leaves _alloc_fused_zp builds), and the
[E,T,1] router matrix. The router matrix is built IN-FORWARD (it depends on this step's
routing), so like the routed path's per-(token,k) weights it is an in-graph intermediate
-- but it is a fresh-every-step VALUE the op CONSUMES, not a persistent INPUT the op's
index_select SOURCES from, so the free_inplace pass reclaiming it across buckets is
fine: each bucket's capture rebuilds it. (The capture hazard the held-leaf machinery
fixes was specifically about a SOURCE tensor of an index_select being freed; the
monolithic op does NO index_select on a python-built tensor -- it slices the persistent
stacked weights ON-DEVICE -- so the only in-forward tensor is W, a consumed value.) x
still flows through the held-leaf x_buf so the op's FIRST input is rooted at persistent
storage across buckets.

GPTQ-symmetric only (the stacked zp is the broadcast shared midpoint; AWQ per-expert
zeros also work since the op takes a full [E,...] zp, but the plugin only stacks the zp
for the GPTQ-sym case -- AWQ falls back, identical to the routed path's gate). silu only
(the op's graph hard-codes silu(gate)*up). Lazy + eager. RUN ON THE HPU BOX.
"""
from __future__ import annotations

import os

import torch
import habana_frameworks.torch.core as htorch

# CHUNK: experts processed per batched sub-chain inside the op (the SRAM-pressure knob).
# E (=64 for Qwen3.5-397B at TP8) experts' dequant can't ALL be SRAM-resident at once
# (~24MB/expert * 64 = 1.5GB >> 48MB Gaudi-2 SRAM), so the op chunks E into `chunk`-
# expert batched sub-chains; each chunk's dequant is NON-PERSISTENT (SRAM-resident,
# bundled into its batch_gemm). Smaller chunk -> fewer dequant tiles co-resident ->
# less spill risk, at the cost of more (smaller) batch_gemms. Default 8 (the proven K=8
# batched tile). HPU_MOE_FUSED_MONOLITHIC_CHUNK overrides; clamped to [1,E] in the op.
_CHUNK = int(os.environ.get("HPU_MOE_FUSED_MONOLITHIC_CHUNK", "8") or "8")

_mono_so_loaded = [False]

# W4A8: HPU_MOE_FUSED_MONO_INT4FP8=1 routes the monolithic path to the int4->fp8 op
# (dequant int4->fp8 in-SRAM + native fp8 MME, 2x bf16; DPQ arXiv 2505.14638 = up to 3x
# over W4A16 on Gaudi-2). IDENTICAL op signature -> the rest of fused_moe_monolithic is
# unchanged; only the bound op + the .so differ. lazy-only (the serving path).
_INT4FP8 = os.environ.get("HPU_MOE_FUSED_MONO_INT4FP8") == "1"
_MONO_OP = "fused_moe_monolithic_int4fp8" if _INT4FP8 else "fused_moe_monolithic"

# === GATE-B per-layer MoE fence COUNTER (HPU_MOE_FENCE_COUNT=1; default OFF, strict no-op) ===
# Measurement instrument ONLY -- changes NO value and emits NO op; when off every line below is
# behind `if _FENCE_COUNT`, so the serve is byte- and perf-identical to today. It counts, for the
# CONCURRENCY monolithic MoE path (the c128 champion path), per a sliding window:
#   * op-calls   = how many times the monolithic op ran (== MoE layers processed; per model
#                  forward this is num_moe_layers, so op-calls / num_moe_layers == #forwards).
#   * fences     = how many times THIS op fired its OWN htorch.mark_step (the legacy route/gather
#                  boundary). With the champion default HPU_MOE_FUSED_MONO_MARKSTEP=0 this stays
#                  0 (the boundary was already removed -> the GATE-B "fence removed" state); set
#                  HPU_MOE_FUSED_MONO_MARKSTEP=1 to restore it and watch fences/call go 0 -> 1.
# This is the A/B knob for "does the per-layer MoE index-build fence matter at concurrency".
# IMPORTANT (documented in the LOUD log): this counts ONLY the MoE op's own mark_step. Per the
# leakmap (__init__.py:256-295 + memory glm52-kernel-fragmentation-wall) the DOMINANT per-layer
# fences are the TP=8 in-graph all-reduce collectives (2/layer) + the DecoderLayer forward_hook +
# the per-layer ExecuteCachedGraph replay -- none of which this MoE mark_step is, and none of
# which a mark_step removal eliminates. Use HPU_LAYER_FUSE=1 to A/B removing those too.
_FENCE_COUNT = os.environ.get("HPU_MOE_FENCE_COUNT", "0") == "1"
_FENCE_EVERY = max(1, int(os.environ.get("HPU_MOE_FENCE_COUNT_EVERY", "400") or "400"))
_fence_calls = [0]   # monolithic-op invocations (the per-MoE-layer denominator)
_fence_fired = [0]   # monolithic mark_step fences actually emitted (0 when MONO_MARKSTEP=0)


def _load_monolithic_op():
    """Load the MONOLITHIC op .so and return torch.ops.hpu_moe_ext.<_MONO_OP> (the bf16
    op, or the int4->fp8 W4A8 op under HPU_MOE_FUSED_MONO_INT4FP8=1), or None if
    unavailable (caller falls back to the stock op). Idempotent. Co-loads via
    TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, _MONO_OP):
        return getattr(torch.ops.hpu_moe_ext, _MONO_OP)
    if _mono_so_loaded[0]:
        return None
    _mono_so_loaded[0] = True
    if _INT4FP8:
        name = "libmoe_fused_monolithic_int4fp8_lazy.so"   # lazy-only (the serving path)
    else:
        name = ("libmoe_fused_monolithic_lazy.so" if htorch.is_lazy()
                else "libmoe_fused_monolithic.so")
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: MONOLITHIC %s op missing (%s); the .so is absent -- falling "
            "back to the stock VllmMixtureOfExpertsOpWNA16.", _MONO_OP, so)
        return None
    torch.ops.load_library(so)
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, _MONO_OP):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: loaded MONOLITHIC %s op from %s (%s)", _MONO_OP, so,
            "W4A8 int4->fp8 in-SRAM dequant + native fp8 MME (2x; DPQ-validated)"
            if _INT4FP8 else "all-E SRAM-resident bf16 dequant; throughput fusion")
        return getattr(torch.ops.hpu_moe_ext, _MONO_OP)
    return None


def fused_moe_monolithic(x, w13_packed, w13_scale, w13_zp,
                         w2_packed, w2_scale, w2_zp,
                         topk_ids, topk_weights, experts_min, chunk=None,
                         x_buf=None):
    """ALL-E monolithic int4 MoE via the chunked single-recipe op.

    Args (stacked per-expert tensors, PyTorch row-major, on HPU):
      x            : [T, H] bf16 hidden states (T tokens; the THROUGHPUT regime, T>1)
      w13_packed   : [E, H, 2I//8] int32 stacked packed gate+up weights (layer nn.Parameter)
      w13_scale    : [E, H//GS, 2I] bf16 op-layout scales
      w13_zp       : [E, H//GS, 2I//8] int32 packed zero-points (stacked; held leaf)
      w2_packed    : [E, I, H//8]  int32 stacked packed down weights
      w2_scale     : [E, I//GS, H] bf16 op-layout scales
      w2_zp        : [E, I//GS, H//8] int32 packed zero-points (stacked; held leaf)
      topk_ids     : [T, top_k] int -- GLOBAL expert ids (this rank's window @ experts_min)
      topk_weights : [T, top_k] -- router weights (already softmax/renormalized)
      experts_min  : this rank's first GLOBAL expert id (ep_rank * E)
      chunk        : experts/batched-sub-chain (default _CHUNK); clamped to [1,E] in the op
      x_buf        : OPTIONAL persistent [max_T, H] bf16 static-input buffer for x (the
                     multi-bucket capture fix -- the held-leaf x_buf the layer allocates
                     at weight-load). When supplied, x is written into it in place and
                     x_buf[:T] feeds the op, so the op's x input is rooted at storage that
                     survives every decode bucket. None -> x passed directly.

    Returns: [T, H] bf16 = sum_e( expert_e(x) * W[:,e] ), matching the stock
    mixture_of_experts semantics for the LOCAL expert window [0, E).
    """
    op = _load_monolithic_op()
    if op is None:
        return None  # .so absent -> caller falls back to stock
    if _FENCE_COUNT:
        _fence_calls[0] += 1   # GATE-B counter: one monolithic-op invocation (== one MoE layer)
    E = w13_packed.shape[0]
    emin = int(experts_min)
    dev = x.device
    dt = torch.bfloat16                # the op's hardcoded scale/zp(out)/router dtype
    T = x.shape[0]
    ch = _CHUNK if chunk is None else int(chunk)

    # ---- DENSE per-(token,expert) router-weight matrix W[T,E] -> [E,T,1] ----
    # global expert id -> local [0, E); a slot whose id is out of the local window is
    # NON-LOCAL (other rank) and must contribute 0 (its weight zeroed). This mirrors
    # apply_monolithic's remap (and _routed_moe's valid/safe_w) EXACTLY -- the stock
    # path zeroes non-local slots too, so the dense W matches its semantics.
    local = topk_ids.to(torch.int64) - emin            # [T, top_k] local ids (may be <0 / >=E)
    valid = (local >= 0) & (local < E)                  # routed to a LOCAL expert?
    safe_id = torch.where(valid, local, torch.zeros_like(local))   # [T, top_k] in [0,E) (masked)
    safe_w = torch.where(valid, topk_weights.to(torch.float32),
                         torch.zeros_like(topk_weights, dtype=torch.float32))  # [T,top_k] f32 (0 non-local)
    # Dense W[t,e] = sum over slots s with safe_id[t,s]==e of safe_w[t,s] (0 for non-local
    # slots). Built as a VECTORIZED one_hot * weight sum (NOT scatter_add) -- the EXACT form
    # the proven routed path uses for its per-(token,k) weights (fused_expert_routed.py: wK =
    # (onehot * safe_w).sum), generalized to all E columns: no scatter op (avoids the HPU bf16-
    # scatter dtype quirk + the in-place-scatter re-zero hazard), no host sync, no dynamic
    # shape -- it re-materializes cleanly every decode step (lazy-HPU-graph-replay safe).
    # Computed in float32 (the proven routed scatter is f32 too) for accurate accumulation,
    # then cast to the op's bf16 router contract. onehot[T,top_k,E] is tiny (T*top_k*E).
    erange = torch.arange(E, device=dev)
    onehot = (safe_id.unsqueeze(-1) == erange).to(torch.float32)   # [T, top_k, E]
    W = (onehot * safe_w.unsqueeze(-1)).sum(dim=1)                 # [T, E] f32
    # transpose to the op's [E, T, 1] (each expert's per-token weight column), cast bf16.
    # .contiguous() so the op's per-chunk dim-0 slice reads a real stride; unsqueeze the
    # trailing broadcast dim that multiplies over H.
    rw = W.transpose(0, 1).contiguous().to(dt).unsqueeze(-1)       # [E, T, 1] bf16

    # ---- x through the held-leaf static-input buffer (multi-bucket capture fix) ----
    # Same pattern as the routed wrapper: write x into the persistent x_buf and feed
    # x_buf[:T], so the op's FIRST recipe input is rooted at storage the layer holds
    # (survives every decode bucket's replay). Only under lazy capture; eager passes x
    # directly. If x_buf is too small for T (e.g. a big concurrency bucket beyond the
    # buffer's rows), pass x directly for that step (correct, just not buffer-rooted).
    if x_buf is not None and htorch.is_lazy() and x_buf.shape[0] >= T:
        if x_buf.dtype != x.dtype or x_buf.device != x.device:
            raise RuntimeError(
                "hpu-moe-ext fused_moe_monolithic x-buffer mismatch: buf is "
                f"{x_buf.dtype}/{x_buf.device} but x is {x.dtype}/{x.device}.")
        x_buf[:T].copy_(x)
        x_op = x_buf[:T]
    else:
        x_op = x

    # === STATIC-SHAPE RECIPE FUSION (Angle C) ===
    # Historically a mark_step() sat HERE, copied from the ROUTED path's MANDATORY
    # boundary. But the two are NOT the same. The routed path's mark_step is mandatory
    # because it precedes a DATA-DEPENDENT `index_select(0, active)` whose indices
    # (`active = topk(present)`) are values the HPU-graph capture cannot read until they
    # are materialized -- an un-materialized dynamic index makes the gather read garbage.
    #
    # The MONOLITHIC op has NO such gather: it consumes the FULL persistent stacked
    # [E,...] weights DIRECTLY (slicing each chunk on-device) and a DENSE router tensor
    # `rw[E,T,1]`. The ONLY thing built in-Python here is `rw`, whose entire producer
    # chain -- `topk_ids - emin` (sub), `>=/<` (compare), `where` (select), `==` against a
    # static `arange(E)` (one_hot), `*`, `.sum(dim=1)`, `.transpose().contiguous()`,
    # `.to(bf16).unsqueeze(-1)` -- is STATIC-SHAPE value-only arithmetic on [T,top_k]/[T,
    # top_k,E] tensors (E, T, top_k all fixed per decode bucket). There is no
    # data-dependent INDEX anywhere. So the bridge can capture the rw build + the op into
    # ONE recipe; the mark_step only FRAGMENTS the layer into 2 recipes (2 synLaunch + 2
    # host-blocked synchronizeEvent) for no correctness reason. Dropping it lets the rw
    # arithmetic fuse INTO the op's recipe -> the whole MoE layer is ONE recipe ->
    # ~1 sync instead of the per-layer ~7. This is the structural per-layer fusion lever.
    #
    # Byte-exactness is preserved: removing a mark_step changes ONLY recipe SEGMENTATION,
    # not the op graph or any value. rw is the identical tensor either way (the bridge
    # lowers the same arithmetic nodes); the op consumes the identical inputs. The x_buf
    # copy_ is a strided_insert (free_inplace-exempt) whose write-then-read ordering the
    # bridge preserves within a single recipe just as across a mark_step. Greedy
    # byte-exact gate (glm52 France/arithmetic) must confirm on-device.
    #
    # HPU_MOE_FUSED_MONO_MARKSTEP=1 restores the legacy boundary (bisect / fallback if a
    # capture edge needs rw pre-materialized).
    if os.environ.get("HPU_MOE_FUSED_MONO_MARKSTEP", "0") == "1":
        htorch.mark_step()
        if _FENCE_COUNT:
            _fence_fired[0] += 1   # GATE-B counter: this op fired its legacy route/gather fence

    # Pin scales to the op's bf16 dequant contract; weights/zp stay int32. The stacked
    # weights/scales/zp are the layer's persistent tensors, passed DIRECTLY (no gather);
    # the op slices each chunk on-device.
    out = op(
        x_op,
        w13_packed, w13_scale.to(dt), w13_zp,
        w2_packed, w2_scale.to(dt), w2_zp,
        rw, int(ch))                                    # [T, H] bf16
    if _FENCE_COUNT and (_fence_calls[0] % _FENCE_EVERY == 0):
        import logging
        logging.getLogger("vllm").warning(
            "[moe-fence-count] monolithic MoE op-calls=%d  mono-fences=%d  fences/call=%.3f  "
            "(HPU_MOE_FUSED_MONO_MARKSTEP=%s). op-calls/num_moe_layers == #forwards. This counts "
            "ONLY the MoE op's OWN mark_step (the GATE-B route/gather fence) -- already 0 at the "
            "champion default markstep=0. The DOMINANT per-layer fences are the TP=8 all-reduce "
            "collectives (2/layer) + DecoderLayer forward_hook + ExecuteCachedGraph replay (A/B "
            "those with HPU_LAYER_FUSE=1); a mark_step removal does NOT eliminate an in-graph "
            "hcclAllReduce, so the per-layer-fence lever is collective/replay-bound, not MoE-index.",
            _fence_calls[0], _fence_fired[0],
            _fence_fired[0] / max(1, _fence_calls[0]),
            os.environ.get("HPU_MOE_FUSED_MONO_MARKSTEP", "0"))
    return out
