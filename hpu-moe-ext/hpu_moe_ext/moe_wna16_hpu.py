"""HPU WNA16 FusedMoE method for GPTQ/AWQ checkpoints.

Reuses the proven, format-agnostic Gaudi INT4 MoE op
``vllm_gaudi.extension.ops.VllmMixtureOfExpertsOpWNA16`` (the same
``torch.ops.hpu.convert_from_uint4`` intrinsic the linear GPTQ/AWQ/CT paths use).

Strategy: subclass upstream ``MoeWNA16Method`` so weight LOADING (the
``w13_qweight`` / ``w2_qweight`` / ``w*_scales`` / ``w*_qzeros`` params, the
gptq+awq weight loaders, group-size handling) is inherited unchanged. We only
override ``process_weights_after_loading`` to re-pack the loaded uint8 weights
into the int32 layout the HPU op wants and bind ``layer.moe_op``, plus
``apply``/``apply_monolithic`` to dispatch through that op (copied from the CT
HPU method ``HPUCompressedTensorsWNA16MoEMethod``).
"""
from __future__ import annotations

from typing import Any, Optional, Union

import torch

from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

import os
import habana_frameworks.torch.core as htorch
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpWNA16
from vllm_gaudi.ops.hpu_gptq import GPTQHPULinearMethod

# ENGAGE: LOUD silent-fallback logs + HPU_ENGINE_STRICT hard-fail + the resolved-state
# ENGAGE-MANIFEST. Default-OFF for strict (only logs); manifest default-ON (pure output).
# Imported here so the MoE forward can mark every fast-path-vs-fallback decision.
from .engage_manifest import (
    engage_fallback, emit_manifest, strict_enabled, manifest_enabled)

_EP_RANK_MISSING = object()


def _resolve_ep_rank(layer: torch.nn.Module, default=_EP_RANK_MISSING) -> int:
    """Resolve expert-parallel rank across legacy and modular vLLM MoE APIs.

    Legacy ``FusedMoE`` exposes ``layer.ep_rank`` directly.  Modular
    ``RoutedExperts`` instead exposes it through ``layer.moe_config.ep_rank``
    (which may itself forward ``moe_parallel_config.ep_rank``).  Cache a
    private normalized value and best-effort legacy alias so the rest of the
    HPU implementation remains API-version agnostic.
    """
    rank = getattr(layer, "ep_rank", None)
    if rank is None:
        rank = getattr(layer, "_hpu_ep_rank", None)
    if rank is None:
        moe_config = getattr(layer, "moe_config", None)
        rank = getattr(moe_config, "ep_rank", None)
        if rank is None:
            parallel_config = getattr(moe_config, "moe_parallel_config", None)
            rank = getattr(parallel_config, "ep_rank", None)
    if rank is None:
        if default is not _EP_RANK_MISSING:
            return int(default)
        raise AttributeError(
            "hpu-moe-ext: MoE layer exposes neither ep_rank nor "
            "moe_config.ep_rank")

    rank = int(rank)
    layer._hpu_ep_rank = rank
    if getattr(layer, "ep_rank", None) is None:
        try:
            layer.ep_rank = rank
        except (AttributeError, TypeError):
            pass
    return rank

# Diagnostic probe: log MoE routing and output norms on first N calls.
# Controlled by HPU_MOE_PROBE (number of calls to log, default 0 = off).
_PROBE_N = int(os.environ.get("HPU_MOE_PROBE", "0") or "0")
_probe_count = [0]  # mutable counter (list to allow mutation from nested scope)

if _PROBE_N > 0:
    import logging as _logging
    _probe_log = _logging.getLogger("vllm")

    @torch.compiler.disable
    def _probe_log_call(count, ep_rank, experts_min, experts_max,
                        topk_ids, topk_weights, out):
        """Log routing + output info. @torch.compiler.disable prevents graph-break issues."""
        try:
            ids_flat = topk_ids.detach().cpu().flatten().tolist()[:32]
            local_ids = [i for i in ids_flat if experts_min <= i <= experts_max]
            w_sum = float(topk_weights.detach().cpu().float().sum().item())
            o_norm = float(out.detach().cpu().float().norm().item())
            o_max = float(out.detach().cpu().float().abs().max().item())
            _probe_log.warning(
                "[moe-probe #%d] ep_rank=%d range=[%d..%d] "
                "local_routed=%d/%d ids=%s w_sum=%.4f "
                "out_norm=%.4f out_max=%.4f",
                count, ep_rank, experts_min, experts_max,
                len(local_ids), len(ids_flat), str(ids_flat[:8]),
                w_sum, o_norm, o_max,
            )
        except Exception as ex:
            _probe_log.warning("[moe-probe] log error: %s", ex)

# === PER-DECODE-STEP ROUTING-FREEZE PROBE (lazy-capture diagnosis) =============
# Controlled by HPU_ROUTED_STEP_PROBE=<N> (default 0 = OFF = fully inert; no host
# sync, no behavior change). When >0, the FIRST MoE layer's _routed_moe logs a
# one-line fingerprint on EACH of the first N decode calls of that layer:
#   x_fp   = sum(|x|)        -- the per-step experts-INPUT activation. CHANGES every step.
#   act_fp = sum(active ids) -- the K locally-routed experts CHOSEN this step.
#   out_fp = ||routed_out||  -- the MoE output this step.
# DIAGNOSIS (grep the docker log for "[routed-step]"):
#   * x_fp changes every step BUT act_fp (and/or out_fp) FREEZES to a constant after the
#     first ~1-2 steps  =>  CAPTURE-FREEZE confirmed: under wrap_in_hpu_graph the routing
#     intermediates (active/remap/...) were folded to the capture step's values and are NOT
#     recomputed on replay, so every later token reuses step-0's experts -> coherent first
#     tokens then greedy repetition. This is a NUMERICAL/capture bug in _routed_moe, NOT a
#     decode-bucket boundary (block_size=128 -> num_blocks=1 for tokens 0..127; there is no
#     bucket switch at the ~8-token onset).
#   * x_fp, act_fp, out_fp ALL keep changing per step  =>  routing is live; the degeneration
#     is downstream (sampler/residual/attention), NOT this MoE path.
# Gated to layer index 0 only (one line/step) so the log stays readable across 75 layers.
# Uses @torch.compiler.disable + .item() (host sync) — diagnosis ONLY; never the hot path.
_ROUTED_STEP_PROBE = int(os.environ.get("HPU_ROUTED_STEP_PROBE", "0") or "0")
_routed_step_count = {}  # per-layer-id call counter

if _ROUTED_STEP_PROBE > 0:
    import logging as _logging
    _routed_step_log = _logging.getLogger("vllm")

    @torch.compiler.disable
    def _routed_step_probe(layer, x, active, routed_out):
        try:
            lid = id(layer)
            # only the first MoE layer, to keep it to one line per decode step
            if _routed_step_count and lid not in _routed_step_count and len(_routed_step_count) >= 1:
                return
            n = _routed_step_count.get(lid, 0)
            if n >= _ROUTED_STEP_PROBE:
                return
            _routed_step_count[lid] = n + 1
            x_fp = float(x.detach().float().abs().sum().item())
            act_fp = int(active.detach().to(torch.int64).sum().item())
            act_lst = active.detach().cpu().flatten().tolist()
            out_fp = float(routed_out.detach().float().norm().item())
            _routed_step_log.warning(
                "[routed-step lid=%d call=%d] rows=%d x_fp=%.5f act_sum=%d active=%s out_norm=%.5f",
                lid % 100000, n, int(x.shape[0]), x_fp, act_fp, str(act_lst[:8]), out_fp)
        except Exception as ex:
            _routed_step_log.warning("[routed-step] log error: %s", ex)

# 4-bit / 8-bit unsigned scalar types (carry .bias = midpoint used as the
# symmetric zero-point), matching WNA16_SUPPORTED_TYPES_MAP in vLLM core.
_WNA16_TYPES = {4: scalar_types.uint4, 8: scalar_types.uint8}

# 2d ROUTED-GATHER (single-stream win): the stock VllmMixtureOfExpertsOpWNA16 dequantizes
# ALL local_num_experts (e.g. 128) per token via convert_from_uint4, writing bf16 to HBM —
# a ~99ms/token bandwidth floor at batch=1. When HPU_MOE_ROUTED_K>0 we instead dequant only
# the <=K locally-routed experts. Used ONLY when tokens*top_k <= K (then #distinct local
# experts <= K is guaranteed, so no host sync / no dynamic shape / no recompiles); otherwise
# the stock all-expert op runs (correctness on prefill/large batch). HPU_MOE_ROUTED_VERIFY=1
# runs BOTH paths and asserts they match (bit-exact validation). GPTQ-symmetric only (shared
# zero-point); AWQ per-expert zp falls back to the all-expert op.
_ROUTED_K = int(os.environ.get("HPU_MOE_ROUTED_K", "0") or "0")
_ROUTED_VERIFY = os.environ.get("HPU_MOE_ROUTED_VERIFY") == "1"

# === bs>1 ROUTED/FUSED gather — the MTP-VERIFY lever ==========================
# The gate above admits the routed/fused gather ONLY when tokens*top_k <= HPU_MOE_ROUTED_K
# (=8). That makes plain decode (T=1, 1*8=8<=8) route through the fast int4-fused kernel,
# but the MTP VERIFY forward (a [bs=depth+1] batch, e.g. 4*8=32 > 8) FALLS to the all-expert
# path (~194ms/step, dequant ALL n_routed_experts=256) -> MTP is a NET LOSS. Yet at bs=4 the
# UNION of distinct LOCAL experts the batch routes to is tiny (~3.8 mean, p95~7, max~11 of
# 256), so the gather is ~2-2.7x cheaper than all-expert -- the verify SHOULD route. The
# T*top_k<=K gate is the SAFETY proxy "union <= K is guaranteed", but at T>1 it conflates
# the (large) worst-case T*top_k with the (small) actual union and needlessly bans the win.
#
# HPU_MOE_ROUTED_BSGT1=1 widens the gate to admit T>1 too. K is then sized PER-T from a
# STATIC bucket (recipe-cache-stable, no host sync -- a fixed K per decode bucket means ONE
# routed recipe per bucket, NOT a per-step dynamic shape). T>1 reaches the EXISTING
# offset-safe per-expert LOOP in fused_expert_routed.py (the [1,1]-router single-recipe fast
# path is structurally T==1 and is NOT taken at T>1), which applies the [T,1] router weight
# OUTSIDE the op and clones each per-expert slice to storage-offset 0 -- already correct for
# any token count (the C++ fused_expert op reads M from x: M-agnostic).
#
# DEFAULT OFF -> the gate is the unchanged T*top_k<=_ROUTED_K, BYTE-IDENTICAL to today
# (plain bs=1 decode only). This flag ONLY ADDS the T>1 admission; it changes NO bs=1 path.
_ROUTED_BSGT1 = os.environ.get("HPU_MOE_ROUTED_BSGT1") == "1"
# Static per-T K bucket "T:K,T:K,...". Sized to COVER the bs<=T routed union:
#   * CONSERVATIVE / correctness-only default (this map empty): K = min(E, T*top_k). Since
#     union <= T*top_k ALWAYS, topk(present,K) can NEVER truncate -> ZERO dropped experts,
#     bit-faithful to the all-expert oracle. No win at large T (K==T*top_k == the all-pick
#     count) but proves the path end-to-end. This is the safe A/B baseline.
#   * THE WIN (set e.g. "1:8,2:8,4:16"): a bucket BELOW T*top_k (16 < 4*8=32 at bs=4) bets
#     the union <= K (measured max~11 << 16, comfortable margin). Fewer experts gathered ->
#     the verify routes ~2x cheaper. If the union EVER exceeds K, the dropped expert is
#     cleanly EXCLUDED (the sentinel-K remap in fused_expert_routed.py), NOT wrong-folded --
#     a small BOUNDED relNorm bump the HPU_MOE_ROUTED_VERIFY / HPU_MOE_FUSED_VERIFY oracle
#     catches. So the bucket degrades SAFELY; validate it (verify on) before trusting it.
# K is ALWAYS clamped to [#distinct-coverable .. E]: K=min(E, max(top_k, bucket_or_T*top_k))
# so it can never under-run a single token's top_k (>= top_k distinct guaranteed at T>=1)
# nor exceed the local expert count. NOT used unless HPU_MOE_ROUTED_BSGT1=1.
def _parse_kbucket(s):
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            t_str, k_str = part.split(":")
            out[int(t_str)] = int(k_str)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger("vllm").warning(
                "hpu-moe-ext: ignoring malformed HPU_MOE_ROUTED_KBUCKET entry %r "
                "(want 'T:K'); using the conservative K=min(E,T*top_k) default.", part)
    return out
_ROUTED_KBUCKET = _parse_kbucket(os.environ.get("HPU_MOE_ROUTED_KBUCKET", ""))

def _routed_bsgt1_K(T, top_k, E):
    """STATIC gather width K for a T-token routed/fused forward under HPU_MOE_ROUTED_BSGT1.
    Bucket override if given for this T, else the conservative T*top_k; ALWAYS clamped to
    [top_k .. E] (>= one token's picks; <= local expert count). Pure int math on host
    SCALARS already known before the op (T from x.shape[0], top_k/E layer constants) -- no
    device read, no per-step sync; the same K recurs for every step in a decode bucket so
    the routed recipe is cache-stable."""
    base = _ROUTED_KBUCKET.get(int(T), int(T) * int(top_k))
    return max(int(top_k), min(int(E), int(base)))

# HPU_ROUTED_DEBUG=1 -> dump the routing internals of _routed_moe for the FIRST call only
# (active / remap / first-row remapped+safe_w / routed-vs-ref norms) so a single request reveals
# exactly where the routed path diverges from the all-expert op. Off (0) by default = zero overhead.
_ROUTED_DEBUG = os.environ.get("HPU_ROUTED_DEBUG") == "1"
_routed_dbg_count = [0]

# === FUSED int4-MoE OpBackend op (item 2/4) ====================================
# HPU_MOE_FUSED_OP=1 -> in the routed fast path replace the per-expert
# convert_from_uint4 list -> mark_step -> mixture_of_experts tail (the bandwidth-
# bound bf16-to-HBM round trip) with the GC-captured, SRAM-resident
# torch.ops.hpu_moe_ext.fused_expert op, driven per-expert by the K-expert
# wrapper hpu_moe_ext.fused_expert_routed.fused_moe_routed(...). Same eligibility
# gate as the stock routed path (GPTQ-symmetric int4 only, tokens*top_k <= K), so
# the fused op only ever sees the <=K distinct locally-routed experts. The stock
# mixture_of_experts path stays the fallback when the flag is off.
# HPU_MOE_FUSED_VERIFY=1 -> run BOTH (fused op vs the stock mixture_of_experts op)
# every call and assert relNorm < 1e-2, LOUD on mismatch (mirrors the existing
# HPU_MOE_ROUTED_VERIFY harness; stays coherent by returning the stock result).
_FUSED_OP = os.environ.get("HPU_MOE_FUSED_OP") == "1"
_FUSED_VERIFY = os.environ.get("HPU_MOE_FUSED_VERIFY") == "1"
# EP-FILTER (HPU_MOE_EP_A2A=1, DP path): the all-gather brings dp*N tokens to every rank, but ~34% have NO
# local expert (they contribute exactly 0 through the tid-remap). Run the op on only a STATIC capacity of
# tokens (HPU_MOE_EP_CAP fraction, default 0.80) selected by topk of the has-local-expert score -> the
# dropped tokens are the score-0 ones (contribute 0 anyway), so this is BYTE-EXACT as long as CAP >= the
# per-rank local-expert fraction, while cutting ~(1-CAP) of the expert compute. STATIC shape (topk to a fixed
# C) -> lazy-mode-safe (unlike a dynamic boolean filter, which crashed). Default off -> byte-identical.
_EP_FILTER = os.environ.get("HPU_MOE_EP_A2A") == "1"
_EP_CAP = float(os.environ.get("HPU_MOE_EP_CAP") or "0.80")
_EP_FILTER_LOGGED = [False]
# ---- FORENSIC CORRUPTION PROBE (default OFF; pure no-op when HPU_MOE_POISON unset) ----
# HPU_MOE_POISON=1 scales the routed-fused MoE output by 1.15 at the EXACT bs=1 decode
# return site that the b1timed serve takes (apply_monolithic -> _fused_moe_routed ->
# torch.ops.hpu_moe_ext.fused_expert). If our MoE op actually executes on device, this
# visibly GARBLES greedy output; if greedy output stays coherent, our op did NOT run
# (stock shadow / stale-recipe replay) -> override CONFIRMED. LOUD install banner so the
# probe is unmistakably armed. MUST be run with a WIPED recipe cache (the poison is
# import-time/in-graph; a warm ,false, recipe would replay the un-poisoned graph).
_MOE_POISON = os.environ.get("HPU_MOE_POISON") == "1"
if _MOE_POISON:
    try:
        import logging as _pl
        _pl.getLogger("vllm").warning(
            "[HPU_MOE_POISON] ARMED: routed-fused MoE output will be scaled x1.15 at the "
            "bs=1 decode return (moe_wna16_hpu apply_monolithic). Coherent greedy output "
            "==> our MoE op did NOT execute (override/stale-recipe). Greedy garble ==> our "
            "op runs. WIPE /recipe_cache/<key> before serving or a warm recipe masks this.")
    except Exception:  # noqa: BLE001
        pass
# MULTI-BUCKET CAPTURE FIX — PERSISTENT STATIC-INPUT x BUFFER (default ON).
# The single-recipe fused op consumes x (the experts-input activation) as its FIRST
# recipe input, but x arrives as an `aten::view` of a NON-persistent decode intermediate
# (apply_monolithic: `x = x.view(-1, H)`). The bridge roots the recipe's x input at that
# view's BASE lazy-tensor, which the FIRST decode bucket's replay frees -> the SECOND
# bucket's on-the-fly capture validates an EMPTY x (ValidateSyncInputTensors / "Strided
# Params Has Value: 0" on model/N/mlp/experts/hpu__input). The earlier in-forward
# `x.contiguous().clone()` did NOT fix it: a clone born INSIDE the captured forward is an
# in-graph intermediate (an output-not-input), so free_inplace reclaims it across buckets.
# FIX (the canonical HPU-graph static-input pattern, == vllm-gaudi's self.input_ids_hpu
# allocated once in __init__ then copy_()'d each step): give each MoE layer a PERSISTENT,
# pre-allocated [max_T, H] bf16 buffer created ONCE at weight-load time (process_weights_
# after_loading -> layer._hpu_fused_x_buf), OUTSIDE any capture. Each captured decode step
# the wrapper does `buf[:T].copy_(x)` and passes `buf[:T]` to the op. The buffer is a held
# leaf the layer owns (like an nn.Parameter / the weights), so its base lazy-tensor IS in
# the recipe's input set (hblazy_tensors_in_) -> HPUGraph::mark_user_outputs flags it
# is_input -> it is NEVER freed at replay; and the in-place copy_ is a strided_insert,
# which free_inplace=False (the __init__.py patch) ALSO exempts. So EACH bucket's recipe
# roots x at the SAME live, held storage. HPU_MOE_FUSED_X_BUFFER=0 -> fall back to the
# in-forward clone (the OLD path; reproduces the multi-bucket FATAL — bisect only).
_FUSED_X_BUFFER = os.environ.get("HPU_MOE_FUSED_X_BUFFER", "1") != "0"
# Rows of the persistent x buffer. Plain decode feeds a single [1,H] token (T==1). With
# HPU_MOE_ROUTED_BSGT1=1 the captured single-recipe path ALSO serves the MTP-VERIFY batch
# (T = padded_batch_size * (depth+1)), so the buffer must cover that T or the verify step
# falls back to the in-forward clone (NOT multi-bucket-safe under capture -> the 2nd-bucket
# FATAL). SINGLE-STREAM MTP (the prod target, max-num-seqs=1): T = depth+1 (4 at depth-3) <=
# the default 8 -- FITS. CONCURRENCY (max-num-seqs=S, depth d): T up to S*(d+1); raise
# HPU_MOE_FUSED_X_BUF_T to >= that (e.g. 8 seqs * depth-3 = 32) so the verify stays buffered.
# A small headroom is cheap (the buffer is [T,H] bf16, a few hundred KB). T beyond capacity
# falls back to the per-call clone for that step (correct numerics; only the multi-bucket
# capture guarantee is lost -- it would re-crash under capture, so size it to cover the
# verify T when HPU_MOE_ROUTED_BSGT1=1).
_FUSED_X_BUF_T = int(os.environ.get("HPU_MOE_FUSED_X_BUF_T", "8") or "8")

# === FUSED MONOLITHIC int4-MoE op (the THROUGHPUT/concurrent regime) ===========
# HPU_MOE_FUSED_MONOLITHIC=1 -> in apply_monolithic's ALL-EXPERT path (the path taken
# at concurrency, when tokens*top_k > HPU_MOE_ROUTED_K so the routed gather is
# ineligible) replace the STOCK VllmMixtureOfExpertsOpWNA16 (which dequantizes ALL E
# experts to bf16 -> HBM -> a separate mixture_of_experts gemm, the ~99us/token bf16->
# HBM round-trip floor) with the single-recipe torch.ops.hpu_moe_ext.fused_moe_monolithic
# op (driven by hpu_moe_ext.fused_monolithic_moe.fused_moe_monolithic). That op processes
# ALL E LOCAL experts with per-expert SRAM-resident dequant (chunked over E so each
# chunk's dequant stays NON-PERSISTENT -> bundled into its batch_gemm MME -- the fusion
# lever), applying a DENSE [E,T,1] per-(token,expert) router matrix that reproduces the
# stock mixture_of_experts semantics exactly. This is the regime where MoE fusion's
# bandwidth win actually manifests (single-stream is overhead-bound -> the routed path;
# throughput is bandwidth-bound -> THIS). GPTQ-symmetric int4 only (None stacked-zp ->
# AWQ -> falls back to stock); silu only. The stock op stays the fallback when the flag
# is off / the op is ineligible / the .so is missing.
# HPU_MOE_FUSED_MONOLITHIC_VERIFY=1 -> A/B vs stock (relNorm < 1e-2) -- EAGER-ONLY debug:
# the .item() the relNorm needs breaks lazy HPU-graph capture, so this must NOT be set in
# a captured lazy serve (it would split/blow the recipe). LOUD + stay-coherent (return
# stock) on mismatch, mirroring the routed/fused verify harnesses.
_FUSED_MONOLITHIC = os.environ.get("HPU_MOE_FUSED_MONOLITHIC") == "1"
_FUSED_MONOLITHIC_VERIFY = os.environ.get("HPU_MOE_FUSED_MONOLITHIC_VERIFY") == "1"
# HPU_MOE_FUSED_MONO_MAX_T = N (default 0 = unlimited) -> the monolithic op engages ONLY for
# batches whose flat token count T (= x.shape[0]) is <= N; any LARGER T falls straight through
# to the bounded stock all-expert mixture_of_experts op. WHY THIS EXISTS (the c64/c128 HBM fit):
# the monolithic computes ALL E local experts over EVERY token (a dense [E,T,1] router zeroes the
# non-routed ones), so its transient per-chunk bf16 intermediates -- gu[chunk,T,2I], y[chunk,T,H]
# -- scale LINEARLY with T. At DECODE concurrency T = the running batch (<= max_num_seqs, e.g.
# 64/128) -> a few MB/chunk. At PREFILL T = the prompt-chunk length (up to max_num_batched_tokens,
# 2k-16k) -> gu[8,16384,4096] bf16 = 1 GiB/chunk, and the WARMUP graph-capture of the (default-
# unbounded, ~64) prompt buckets exhausts HBM -> "Prompt warmup processing ... PT_DEVMEM Allocation
# failed" (the observed c64/c128 OOM). Prefill is ALSO where the monolithic is maximally wasteful
# (E*T expert-token products vs the stock op's ~T*top_k routed compute), so diverting T>N to the
# stock op is BOTH the footprint fix AND a prefill speedup -- and it is numerically the monolithic's
# OWN validation reference (the stock all-expert op it was checked against at relNorm<1e-2), so
# greedy output stays coherent. Set N = the decode batch cap (== max_num_seqs) so every decode
# bucket keeps the SRAM-resident fp8-MME win while every (larger-T) prefill bucket takes the
# bounded stock op. DEFAULT 0 -> unlimited -> BYTE-IDENTICAL to the historical behavior.
_FUSED_MONO_MAX_T = int(os.environ.get("HPU_MOE_FUSED_MONO_MAX_T", "0") or "0")

# === REAP EXPERT-PRUNE PERF PROBE — HPU_MOE_REAP_PROBE (default-OFF byte-exact no-op) =============
# Answers ONE question cheaply: does pruning the MoE 256->192 experts (REAP, within-EP-window) drop
# the CONCURRENCY step time? At concurrency the champion runs the MONOLITHIC op (apply_monolithic ->
# _fused_monolithic), which processes ALL E LOCAL experts -- a DENSE [E,T,1] router zeroes the non-
# routed ones, but the captured recipe's expert-GEMM work scales with E, NOT top_k. So fewer resident
# experts == proportionally fewer expert-GEMMs == a POTENTIAL MoE speedup (UNLIKE the bs=1 routed-K
# gather, where K=top_k regardless of E -> pruning the TOTAL is moot). This PROBE measures that
# perf WITHOUT the multi-hour 192-checkpoint regeneration: it slices each rank's stacked expert
# weights to the kept count at runtime so the op builds a SMALLER (keep-expert) recipe. The OUTPUT
# IS QUALITY-GARBAGE (the kept rows are NOT the REAP-selected survivors and are not rescaled) -- this
# is PERF-ONLY (read HPU_STEP_TIMER step-ms 192-vs-256); a real win then justifies the physical
# prune_experts.py transform (correct + quality-gated). It is FAITHFUL to the shipped checkpoint's
# COST because the mask is within-EP-window-BALANCED (validated below): the real 192 checkpoint
# reshards to EXACTLY keep_per_window experts on EVERY rank, so a contiguous [:keep] slice reproduces
# the real per-rank expert COUNT (and the monolithic op's cost is count-driven, not which-experts-
# driven). DEFAULT (unset) -> the `if _REAP_PROBE` block never runs -> byte-identical.
_REAP_PROBE = os.environ.get("HPU_MOE_REAP_PROBE") == "1"
_REAP_PROBE_MASK = os.environ.get("HPU_MOE_REAP_PROBE_MASK", "").strip()
_reap_probe_logged = [False]


def _reap_parse_mask(raw: "dict") -> "dict":
    """PURE (no torch / no op / card-free) validate+summarize a within-window REAP mask. Returns
    {src,dst,ep_window,nwin,drop_per_window,keep_per_window}. LOUD (ValueError) UNLESS the drop is
    WITHIN-EP-WINDOW-BALANCED and UNIFORM across layers -- the invariant the contiguous-slice perf
    probe relies on. (If a future mask is NOT within-window-balanced, the real 192-reshard would NOT
    be a uniform per-rank [:keep], so [:keep] would mis-predict the per-rank load -> raise so the
    operator switches to a per-id index_select probe instead of measuring a wrong cost.)"""
    if not isinstance(raw, dict):
        raise ValueError("REAP mask must be a JSON object")
    meta = raw.get("meta", {}) or {}
    drop = raw.get("drop", {}) or {}
    src = int(meta.get("n_routed_experts_src", 0))
    dst = int(meta.get("n_routed_experts_dst", 0))
    ep_window = int(meta.get("ep_window", 0))
    if not (0 < dst < src):
        raise ValueError(f"REAP mask meta needs 0 < dst({dst}) < src({src})")
    if ep_window <= 0 or src % ep_window != 0:
        raise ValueError(f"REAP mask ep_window({ep_window}) must be >0 and divide src({src})")
    if not isinstance(drop, dict) or not drop:
        raise ValueError("REAP mask has empty/invalid 'drop' (nothing to prune)")
    nwin = src // ep_window
    drop_per_window = None
    for L, ids in drop.items():
        if not isinstance(ids, (list, tuple)):
            raise ValueError(f"REAP mask layer {L}: 'drop' value must be a list of ids")
        hist = [0] * nwin
        for e in ids:
            ei = int(e)
            if not (0 <= ei < src):
                raise ValueError(f"REAP mask layer {L}: drop id {ei} out of [0,{src})")
            hist[ei // ep_window] += 1
        d0 = hist[0]
        if any(h != d0 for h in hist):
            raise ValueError(
                f"REAP mask layer {L}: drop NOT within-EP-window-balanced (per-window {hist}); the "
                "contiguous-slice perf probe requires every window to drop the same count")
        if drop_per_window is None:
            drop_per_window = d0
        elif d0 != drop_per_window:
            raise ValueError(
                f"REAP mask layer {L}: drop_per_window {d0} != {drop_per_window} (non-uniform across "
                "layers); the perf probe assumes one uniform per-rank expert count")
    keep_per_window = ep_window - int(drop_per_window)
    if keep_per_window <= 0:
        raise ValueError(f"REAP mask keep_per_window {keep_per_window} <= 0 (would delete the MoE)")
    if keep_per_window * nwin != dst:
        raise ValueError(
            f"REAP mask inconsistent: keep_per_window({keep_per_window})*nwin({nwin}) != dst({dst})")
    return {"src": src, "dst": dst, "ep_window": ep_window, "nwin": nwin,
            "drop_per_window": int(drop_per_window), "keep_per_window": keep_per_window}


def _reap_load_info():
    """Read+validate the mask at HPU_MOE_REAP_PROBE_MASK. LOUD RuntimeError on ANY problem -- a
    silent no-op here would make the probe measure 256 experts and FALSELY report 'REAP dead'."""
    if not _REAP_PROBE:
        return None
    import json
    if not _REAP_PROBE_MASK:
        raise RuntimeError("[hpu-moe-ext] HPU_MOE_REAP_PROBE=1 but HPU_MOE_REAP_PROBE_MASK is unset "
                           "(point it at the mounted glm52_prune192.json).")
    if not os.path.isfile(_REAP_PROBE_MASK):
        raise RuntimeError(f"[hpu-moe-ext] HPU_MOE_REAP_PROBE_MASK={_REAP_PROBE_MASK!r} is not a "
                           "readable file (mount the REAP mask into the container).")
    try:
        with open(_REAP_PROBE_MASK) as _fh:
            raw = json.load(_fh)
        info = _reap_parse_mask(raw)
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(
            f"[hpu-moe-ext] HPU_MOE_REAP_PROBE: bad mask {_REAP_PROBE_MASK!r}: {ex}") from ex
    return info


_REAP_INFO = _reap_load_info() if _REAP_PROBE else None


def _reap_probe_keep(E: int):
    """Kept LOCAL expert count for a rank window of E experts under the perf probe, or None to leave
    the layer at full E. This rank's window == one EP window, so keep = keep_per_window when E
    matches the mask's ep_window; if a different TP makes E != ep_window, fall back to the dst/src
    keep ratio. Returns None when keep >= E (nothing to slice)."""
    info = _REAP_INFO
    if info is None:
        return None
    if int(E) == int(info["ep_window"]):
        keep = int(info["keep_per_window"])
    else:
        keep = int(round(int(E) * info["dst"] / info["src"]))
    keep = max(1, min(keep, int(E)))
    return keep if keep < int(E) else None


# === PP (disable_tensor_cache=False) PREFILL FIX — HPU_MOE_MONO_E_CHUNK / _T_CHUNK / ROUTED_MAX_T ===
# Under PIPELINE PARALLELISM on lazy mode the model is wrapped with disable_tensor_cache=False
# (FORCED: with True the PP capture hangs on the IntermediateTensors hand-off -- _maybe_wrap_
# in_hpu_graph, hpu_model_runner.py:867 `_dtc = not (pp>1)`). With the cache NOT disabled the
# capture RETAINS the forward's transients (the free_inplace reclaim that lets disable_tensor_
# cache=True reuse per-forward workspace is gone), so the FIRST large-T PREFILL forward fails
# TWO ways depending on which MoE path it takes:
#   (A) all-expert STOCK op (VllmMixtureOfExpertsOpWNA16): dequants ALL E (=128 @ GLM-5.2 TP2/EP2)
#       int4 experts -> bf16, each w13 [H,2I]=48MB / w2 [I,H]=24MB. Those transients can't be
#       reclaimed (cache on) and the fragmented free pool has no 48MB-contiguous hole ->
#       "PT_DEVMEM Allocation failed for size 48MB" (ops.py mixture_of_experts dequant). Shrinking
#       MAX_EXPERTS_PER_SLICE does NOT help: nothing is freed between slices, so peak stays ~E.
#   (B) routed/fused single-recipe gather (HPU_MOE_ROUTED_BSGT1=1): for a large prefill T the
#       static gather width K = _routed_bsgt1_K(T) clamps T*top_k UP TO E (= all local experts),
#       so the SEQUENTIAL single-recipe op is asked to emit E expert subgraphs over T (up to 2-4k)
#       tokens in ONE recipe -> the host-side Synapse graph compile never returns (0% device util,
#       hang). That op was designed for the T==1 / small-K decode + MTP-verify regime, NOT prefill.
#   (C) our MONOLITHIC op (the all-expert prefill replacement for (A)): emit_monolithic_experts
#       add-chains ALL E experts' dequant+gemm+silu nodes into ONE captured recipe (the `chunk`
#       arg is only SRAM tiling, NOT recipe segmentation -- moe_fused_monolithic_emit.h; "caches
#       one recipe per (E, chunk, T, H, I)"). So the recipe NODE COUNT scales with E, NOT T. At
#       E_local=128 (TP2/EP2 -> 256/2; 4x the E=32 validated on TP8) that all-128-expert recipe
#       HANGS the host-side Synapse compiler exactly like (B) -- and a T=128 prefill hangs the
#       same as T=2048, proving it is E-scaled (the node count), not T-scaled (the tensor sizes).
#       So TOKEN-chunking alone does NOT fix it: each sub-batch still feeds the full [E=128,...]
#       weights -> the same all-128 recipe -> the same hang.
#
# THE FIX (all default 0 = OFF):
#   * HPU_MOE_ROUTED_MAX_T=N  -> explicit cap for the routed/fused gather at T<=N (the
#     decode/verify ceiling, e.g. == max_num_seqs). A larger-T prefill SKIPS the gather
#     (avoids the (B) hang) and falls through to monolithic/stock all-expert dispatch.
#     The cap originally engaged only for PP>1, but replicated TP1 GPTQ serves need the same
#     protection when HPU_MOE_ROUTED_BSGT1 admits batched decode: large prefills must not build
#     an E-wide sequential routed recipe. Unset/0 remains byte-identical for every existing serve.
#   * HPU_MOE_MONO_E_CHUNK=ec -> THE (C) HANG FIX. EXPERT-AXIS chunking: slice the stacked weights
#     to experts [g,g+ec) and call the op once per group with experts_min=emin+g, so each call
#     builds a BOUNDED ec-expert recipe (ec=32 == the proven TP8 size). The op's global->local
#     remap zeroes out-of-group experts -> a group's partial = sum over ITS experts; SUMMING the
#     groups == the full MoE (EXACT). Cached by (E=ec,chunk,T,H,I): all groups same shape ->
#     compiled ONCE. OFFSET-SAFE with plain VIEWS (no fragmenting clone): the op slices each
#     weight with synSliceParamsV2 FIRST (emit_chunk_slice), which HONOURS a view's storage-offset
#     (fused_expert_routed.py:709-711), and the reinterpret runs on the slice's offset-0 output.
#   * HPU_MOE_MONO_T_CHUNK=tc -> DEFENSIVE (the (A)-class transient bound): the op's per-sub-chain
#     gu[chunk,T,2I]/y[chunk,T,H] scale with T; split the TOKEN axis into <=tc-row sub-batches so
#     each call's largest transient stays small (~<=12MB @ tc<=128 -- fits a fragmented pool) and
#     reclaimable. tc==_FUSED_X_BUF_T routes every sub-batch through the SAME persistent x_buf.
# Both chunkings are EXACT (expert groups SUM -- non-group experts carry router-weight 0; token
# sub-batches CONCAT -- each token's MoE output is independent) -> byte-identical to one call modulo
# recipe segmentation. NOTE: the op's gu/y compute workspace is INSIDE the C++ op (not a Python
# tensor), so it cannot be a Python-side held leaf without a .so rebuild; bounding it via the
# expert+token chunk + reusing the persistent x_buf is the card-free equivalent.
_MONO_E_CHUNK = int(os.environ.get("HPU_MOE_MONO_E_CHUNK", "0") or "0")
_MONO_T_CHUNK = int(os.environ.get("HPU_MOE_MONO_T_CHUNK", "0") or "0")
_ROUTED_MAX_T = int(os.environ.get("HPU_MOE_ROUTED_MAX_T", "0") or "0")

# Cached pipeline_parallel_size>1 probe (== the serve runs disable_tensor_cache=False). The
# distributed groups are up by the time any MoE forward runs; a pre-init / single-process failure
# returns False UNCACHED (retry next call), so a TP-only serve is byte-identical (the PP-only
# fixes above never engage). One resolve per worker process once the PP group exists.
_pp_gt1 = [None]


def _is_pp_gt1() -> bool:
    if _pp_gt1[0] is not None:
        return _pp_gt1[0]
    try:
        from vllm.distributed.parallel_state import get_pp_group
        val = bool(get_pp_group().world_size > 1)
    except Exception:  # noqa: BLE001
        return False  # groups not yet initialized -> don't cache; retry on the next forward
    _pp_gt1[0] = val
    return val


def _dp_out_reshape(t, input_shape, dp):
    """Reshape a fast-path (fused / monolithic / routed / fp8) MoE output back to the caller's
    shape. Under DP-attention the token axis was all-gathered in apply_monolithic (dispatch_tensor),
    so the output carries the GATHERED leading dim and MUST keep it -- the runner's combine
    (reduce_scatter over the DP group) is what splits the tokens home; collapsing it back to the
    pre-gather local input_shape here is the size/stride mismatch that crashed the DP+monolithic
    compose ("view size is not compatible ... Use .reshape()"). Only the trailing feature dims come
    from input_shape. Non-DP restores the exact input shape. This mirrors the stock all-expert
    return at the tail of apply_monolithic; the fast-path returns simply never got the DP branch.
    .reshape (not .view) so a non-contiguous fused/cat output is copied instead of raising."""
    return t.reshape(t.size(0), *input_shape[1:]) if dp else t.reshape(*input_shape)


# One-shot engagement log for the PP prefill expert/token-chunk path (per worker process).
_logged_mono_chunk = [False]

# === LOAD-TIME fp8 ALL-EXPERT experts (the single-stream / MTP-decode bandwidth win) ===
# HPU_MOE_FP8_EXPERTS=1 -> at weight-load convert each LOCAL int4 expert int4->bf16
# (convert_from_uint4, the ONLY proven out_dtype) then bf16->fp8 via the self-consistent
# cast_to_fp8_v2 round-trip (NEVER int4->fp8 directly -- that hits the Gaudi-2 fn-vs-fnuz
# 2x bug; see fp8_routed_moe.py), and STASH the resident fp8 weights + per-tensor FORWARD
# scales on the layer (layer._fp8_w13/_fp8_w2/_fp8_w13_s/_fp8_w2_s). In apply_monolithic the
# all-expert path then runs fp8_allexpert_moe against those stored fp8 weights with ZERO
# per-step dequant. This HALVES the target-forward all-expert weight bandwidth (fp8 = 1 B/elem
# vs bf16 2 B/elem materialized) -- the confirmed #1 single-stream device cost, and the MTP
# bs>=2 decode regime (tokens*top_k > HPU_MOE_ROUTED_K disables the routed gather, so the
# all-expert path is what runs). DEFAULT OFF => byte-identical to the stock int4 serve.
#
# HBM: fp8 experts ~90.6 GB/card (GLM-5.2: H=6144, I=2048, E=32/card, 75 MoE layers) vs the
# int4 ~46.7 GB/card. They CANNOT both stay resident (137 GB > 96 GB/card), so when the flag
# is on we FREE the stacked int4 expert weights after the fp8 build (HPU_MOE_FP8_FREE_INT4=1,
# default ON under the flag). Net resident expert mem ~90.6 GB/card -> TIGHT (run small ctx /
# few seqs). HPU_MOE_FP8_FREE_INT4=0 keeps int4 too (debug/verify only -> will OOM at scale).
#
# GPTQ-symmetric int4 only (shared zp; AWQ per-expert zp -> ineligible, stays on the stock
# path). silu/gelu (the activation fp8_allexpert_moe supports). The stock op stays the
# fallback when the flag is off / the layer is ineligible.
# HPU_MOE_FP8_EXPERTS_VERIFY=1 -> at LOAD, per layer, A/B the stored-fp8 all-expert output
# vs the stock int4 all-expert op on a synthetic routed batch and assert relNorm < the fp8
# tol (LOUD on mismatch -- catches the 2x fnuz bug / a transpose / a scale-slot error). This
# uses .item() (host sync) so it is a LOAD-TIME gate only; never set it for the hot decode path.
_FP8_EXPERTS = os.environ.get("HPU_MOE_FP8_EXPERTS") == "1"
# HPU_MOE_FP8_FUSED=1 (requires HPU_MOE_FP8_EXPERTS=1) -> route the fp8 expert path to the
# GC-CAPTURED, SRAM-resident FUSED fp8 op (torch.ops.hpu_moe_ext_fp8.fused_expert_fp8,
# moe_fused_expert_fp8_backend.cpp built BUILD_FP8_NODES=1) INSTEAD of the resident-fp8
# torch.ops.hpu.mixture_of_experts path / the per-expert fp8_allexpert_moe Python loop.
# The fused op dequants int4->bf16->fp8 + runs both native fp8 MME GEMMs + silu IN ONE
# recipe per expert (the proper single-recipe fp8 kernel -- the per-expert convert/cast/
# gemm/mark_step Python overhead that contaminated the loop's timing is GONE). It is the
# fp8 analog of the int4 fused op (HPU_MOE_FUSED_OP), wired identically (fp8_routed_moe.
# fp8_fused_moe drives it). Because it consumes the LIVE INT4 weights (dequanting in
# graph), this mode KEEPS int4 resident and does NOT build/free the separate resident-fp8
# tensors -- so _build_fp8_experts is a no-op here and the existing resident-fp8 branch is
# never taken. DEFAULT OFF: with only HPU_MOE_FP8_EXPERTS=1 (this unset) behavior is
# EXACTLY the prior resident-fp8 path -- byte-identical, this flag changes nothing.
_FP8_FUSED = os.environ.get("HPU_MOE_FP8_FUSED") == "1"
# HPU_MOE_FP8_DIRECT=1 (requires HPU_MOE_FP8_FUSED=1) -> the fused fp8 op dequants int4
# DIRECTLY to fp8 in ONE node (no bf16 intermediate, no per-step weight amax/cast). It
# needs the per-expert FORWARD fp8 recovery scale s_fp8 = (|W_real|.max + eps)/240
# PRECOMPUTED ONCE at weight-load (the weights are static) -- built by
# _alloc_fused_fp8_scale and passed as the op's two OPTIONAL trailing inputs
# (w13_fp8_scale / w2_fp8_scale). DEFAULT OFF: when unset the fused fp8 op runs its proven
# in-graph-amax two-hop path (the scales are passed as None) -- byte-identical to before.
# This folds the static weight scale; the activation cast stays in-graph either way.
_FP8_DIRECT = os.environ.get("HPU_MOE_FP8_DIRECT") == "1"
_FP8_FREE_INT4 = os.environ.get("HPU_MOE_FP8_FREE_INT4", "1") != "0"
_FP8_EXPERTS_VERIFY = os.environ.get("HPU_MOE_FP8_EXPERTS_VERIFY") == "1"
# Frobenius-relative tol for the load-time fp8 all-expert verify. int4->fp8 double-quant +
# per-token activation fp8 carries a few-% relative error vs the int4 reference (EXPECTED,
# coherent); a value ABOVE this means a real bug (the 2x fnuz inflation, a transpose, a wrong
# scale slot). 0.08 mirrors fp8_routed_microbench.REL_TOL (uniform-random synthetic worst case
# is harder for a per-tensor fp8 scale than real trained weights). The "goal" relNorm<~1e-2
# is the int4-path bar; fp8's second quantization makes 1e-2 a false-fail (see that microbench).
_FP8_EXPERTS_REL_TOL = float(os.environ.get("HPU_MOE_FP8_EXPERTS_REL_TOL", "0.08") or "0.08")

# One-shot guard for the alloc INFO log (per worker process).
_alloc_logged = [False]
# One-shot guard for the fp8-experts load-time INFO log (per worker process).
_fp8_experts_logged = [False]


def _alloc_fused_x_buf(layer) -> None:
    """Allocate the PERSISTENT static-input x buffer for the fused-MoE single-recipe op,
    ONCE per MoE layer, at weight-load time (OUTSIDE any HPU-graph capture). Sized
    [_FUSED_X_BUF_T, H] bf16 on the layer's device. Stored as layer._hpu_fused_x_buf.

    This is the multi-bucket capture fix: a leaf the layer holds (allocated before any
    capture) is a recipe INPUT, never freed across decode buckets — unlike an in-forward
    clone, which is an in-graph intermediate the free_inplace pass reclaims. PER-LAYER
    (not one global buffer): all MoE layers live in ONE captured recipe, so a shared
    buffer would be a write-after-read hazard between layers. No-op unless the fused op
    is enabled (HPU_MOE_FUSED_OP=1) and the buffer feature is on (HPU_MOE_FUSED_X_BUFFER).
    H is the activation hidden size == the last dim of w2_scales ([E, I/GS, H]).

    Allocated for the routed-fused op (HPU_MOE_FUSED_OP) AND the monolithic-fused op
    (HPU_MOE_FUSED_MONOLITHIC) -- both consume x as their first recipe input and need it
    rooted at held-leaf storage to survive multi-bucket capture."""
    if not ((_FUSED_OP or _FUSED_MONOLITHIC or _FP8_FUSED) and _FUSED_X_BUFFER):
        return
    try:
        import habana_frameworks.torch as _htorch
        if not _htorch.is_lazy():
            return  # the captured single-recipe path (and thus the buffer) is lazy-only
    except Exception:  # noqa: BLE001
        return
    if getattr(layer, "_hpu_fused_x_buf", None) is not None:
        return
    H = int(layer.w2_scales.shape[-1])
    dev = layer.w2_scales.device
    layer._hpu_fused_x_buf = torch.zeros(
        max(1, _FUSED_X_BUF_T), H, dtype=torch.bfloat16, device=dev)
    # One-shot INFO so the ALLOC is verifiable in the serve logs (the load-time half of
    # the multi-bucket fix). If this line is absent for a fused serve, the buffer was
    # never created -> the wrapper falls back to the freed clone/view. Per worker.
    if not _alloc_logged[0]:
        _alloc_logged[0] = True
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: _alloc_fused_x_buf allocated persistent static-input x buffer "
            "[%d, %d] bf16 on %s (HPU_MOE_FUSED_X_BUFFER on) -- per-MoE-layer, at "
            "weight-load, outside capture.", max(1, _FUSED_X_BUF_T), H, dev)


# One-shot guard for the stacked-zp alloc INFO log (per worker process).
_alloc_zp_logged = [False]
# One-shot guard for the fp8-direct scale alloc INFO log (per worker process).
_alloc_fp8_scale_logged = [False]


def _alloc_fused_fp8_scale(layer) -> None:
    """Precompute the PERSISTENT per-expert FORWARD fp8 recovery scales for the DIRECT
    int4->fp8 fused op (HPU_MOE_FP8_DIRECT=1), ONCE per MoE layer, at weight-load time.
    Stored as held leaves layer._hpu_fused_w13_fp8_scale / _hpu_fused_w2_fp8_scale, each
    [E, 1] float32 (one per-tensor scale per LOCAL expert).

    s_fp8[e] = (|W_real[e]|.max + 1e-8) / FP8_MAX   (FP8_MAX = 240, the Gaudi-2 fnuz max).
    This is EXACTLY fp8_routed_moe._quant_weight_fp8's recovery scale (per-tensor), but the
    expensive part -- the int4->bf16 dequant + the abs().max() over the whole [in,out]
    weight -- runs ONCE here at load (the weights are STATIC), not per decode step. The
    fused op then folds 1/s_fp8 into the int4 group scale and emits ONE dequantize_4_bit
    straight to fp8, and passes this same s_fp8 as fp8_gemm's scaleB so the fn-vs-fnuz 2x
    cancels (the WRITE<->READ scale pair agrees on the hf8 grid; see the .cpp DIRECT-DEQUANT
    PROOF). No bf16 weight, no per-step weight amax/cast.

    Built from the STACKED int4 weights (layer.w13_qweight [E,in,out//8], w13_scales
    [E,groups,out]) + the shared GPTQ-sym zero, via the trusted torch.ops.hpu.
    convert_from_uint4(out_dtype=bf16) -- the SAME oracle the int4 path is validated
    against. GPTQ-symmetric only (AWQ per-expert zp -> the fused-fp8 dispatch is already
    ineligible; no-op here). No-op unless HPU_MOE_FP8_DIRECT=1 (and the fused fp8 op is on).

    Default OFF: when HPU_MOE_FP8_DIRECT is unset these leaves are absent, the driver
    passes None for the two optional scale inputs, and the op runs its proven two-hop path."""
    if not (_FP8_DIRECT and _FP8_FUSED):
        return
    w13z = getattr(layer, "_hpu_routed_w13_zp", None)
    w2z = getattr(layer, "_hpu_routed_w2_zp", None)
    if w13z is None or w2z is None:
        return  # AWQ per-expert zp -> fused-fp8 ineligible -> no fp8 scale needed
    E = int(layer._hpu_local_experts)
    existing = getattr(layer, "_hpu_fused_w13_fp8_scale", None)
    if existing is not None and existing.shape[0] == E:
        return
    from .fp8_routed_moe import FP8_MAX
    dev = layer.w13_qweight.data.device
    w13p, w2p = layer.w13_qweight.data, layer.w2_qweight.data       # [E,in,out//8]
    s13, s2 = layer.w13_scales.data, layer.w2_scales.data           # [E,groups,out]
    eps = 1e-8
    sc13 = torch.empty(E, 1, dtype=torch.float32, device=dev)
    sc2 = torch.empty(E, 1, dtype=torch.float32, device=dev)
    import habana_frameworks.torch as _ht   # top-level pkg: has .hpu.synchronize (htorch=.core)
    _lazy = htorch.is_lazy()

    def _co(t):
        # Per-expert slice of the stacked [E,...] param carries a NONZERO storage_offset for
        # e>0. .contiguous() does NOT reset storage_offset when strides are already contiguous
        # (a clean sub-block) -> convert_from_uint4 would read the WRONG expert under lazy.
        # .clone() forces a fresh offset-0 buffer (exactly the safety _build_fp8_experts uses);
        # eager tolerates the offset so skip the copy there.
        return t.clone() if _lazy else t.contiguous()
    for e in range(E):
        # int4 -> bf16 via the TRUSTED convert_from_uint4 (out_dtype=bf16; the only proven
        # out_dtype -- NEVER fp8 here). Per-tensor forward scale = max|W_real| / FP8_MAX.
        w13_bf16 = torch.ops.hpu.convert_from_uint4(
            _co(w13p[e]), _co(s13[e]), w13z, torch.bfloat16, None)
        w2_bf16 = torch.ops.hpu.convert_from_uint4(
            _co(w2p[e]), _co(s2[e]), w2z, torch.bfloat16, None)
        sc13[e, 0] = (w13_bf16.float().abs().max() + eps) / FP8_MAX
        sc2[e, 0] = (w2_bf16.float().abs().max() + eps) / FP8_MAX
        if _lazy:
            htorch.mark_step()   # free each transient bf16 weight before the next expert
            # FIX 2026-06-21: mark_step alone does NOT flush in process_weights_after_loading
            # -> the per-expert convert graphs ACCUMULATE across all E experts until the
            # oversized-graph reject at load (same failure _build_fp8_experts hit). Force a
            # real device flush so each expert's transient bf16 executes+frees. Load-time only.
            _ht.hpu.synchronize()
    layer._hpu_fused_w13_fp8_scale = sc13       # [E,1] f32 FORWARD recovery scales
    layer._hpu_fused_w2_fp8_scale = sc2
    if _lazy:
        _ht.hpu.synchronize()                   # materialize before any decode capture
    if not _alloc_fp8_scale_logged[0]:
        _alloc_fp8_scale_logged[0] = True
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: _alloc_fused_fp8_scale precomputed per-expert DIRECT int4->fp8 "
            "recovery scales w13%s / w2%s (f32) on %s at weight-load (HPU_MOE_FP8_DIRECT=1) "
            "-- the fused fp8 op now dequants int4 STRAIGHT to fp8 in one node (no bf16 "
            "intermediate, no per-step weight amax/cast).",
            tuple(sc13.shape), tuple(sc2.shape), dev)
# One-shot guard for the in-forward stacked-zp FALLBACK warning (should NEVER fire in a
# correct serve; one-shot so a regression is surfaced once, not flooded per decode token).
_warned_zp_inforward = [False]


def _alloc_fused_zp(layer) -> None:
    """Build the PERSISTENT stacked GPTQ-symmetric zero-point tensors for the fused-MoE
    single-recipe op, ONCE per MoE layer, at weight-load time (OUTSIDE any HPU-graph
    capture). Stored as layer._hpu_fused_w13_zp / _hpu_fused_w2_zp (held leaves).

    *** THE LAST MULTI-BUCKET CAPTURE FIX ***. The fused single-recipe op takes 6 stacked
    [E,...] weight inputs that the wrapper index_select(0, active)'s on device:
    w13_packed, w13_scale, w2_packed, w2_scale (all nn.Parameters: w13_qweight/w2_qweight/
    w13_scales/w2_scales) AND the two zero-points. Four of the six are nn.Parameters, so
    the bridge roots them as PERSISTENT recipe inputs (is_input) that survive every decode
    bucket's replay. The two zero-points were the exception: _fused_moe_routed built them
    LAZILY, INSIDE the first decode forward, via `w13z.unsqueeze(0).expand(E,...).contiguous()`.
    That `.contiguous()` runs INSIDE the captured graph, so its output is an IN-GRAPH
    INTERMEDIATE (an output-not-input) the free_inplace pass reclaims across buckets --
    EXACTLY the class the in-forward x-clone fell into. The 1st decode bucket's replay frees
    the stacked-zp storage; the 2nd bucket's on-the-fly capture then validates an EMPTY zp:
        ValidateSyncInputTensors tensor_data is empty ... model/0/mlp/experts/hpu__input ...
        Strided Params Has Value: 0
    -- the failing `hpu__input` is the index_select SOURCE of the zp (high lazy-id, NOT an
    nn.Parameter), while the weight/scale index_select sources (low lazy-id nn.Parameters)
    stay live. (The packed weights show as aten::view[E,...] of a persistent base in the IR;
    the dead leaves are the only two index_select sources with no persistent root.)

    FIX (identical in spirit to _alloc_fused_x_buf): MATERIALIZE the stacked zp HERE, at
    weight-load, OUTSIDE any capture, as a held leaf the layer owns. A leaf allocated before
    capture is in the recipe's input set (hblazy_tensors_in_) -> is_input -> NEVER freed at
    replay -- just like x_buf and the weight nn.Parameters. Then the wrapper's
    index_select(0, active) gathers from live, persistent storage every bucket (the gather
    OUTPUT is alive already; only its SOURCE was being freed). No gather-on-device needed:
    relocating the gather would not help, because the empty tensor is the gather SOURCE, and
    a non-persistent `.contiguous()` source is empty across buckets whether the gather runs on
    host or inside the op -- the cure is making the SOURCE persistent, which this does.

    GPTQ-symmetric only (layer._hpu_routed_w13_zp is None for AWQ per-expert zeros -> the
    fused path is ineligible and falls back, so no stacked zp is needed). No-op unless a
    fused op + the static-input-buffer feature are enabled (same gate as _alloc_fused_x_buf:
    the multi-bucket hazard only exists on the captured lazy single-recipe path). Built for
    BOTH the routed-fused op (HPU_MOE_FUSED_OP, which index_selects the stacked zp) AND the
    monolithic-fused op (HPU_MOE_FUSED_MONOLITHIC, which slices the stacked zp on-device);
    either way the stacked zp must be a persistent held leaf surviving every decode bucket."""
    if not ((_FUSED_OP or _FUSED_MONOLITHIC or _FP8_FUSED) and _FUSED_X_BUFFER):
        return
    try:
        import habana_frameworks.torch as _htorch
        if not _htorch.is_lazy():
            return  # the captured single-recipe path (and thus the hazard) is lazy-only
    except Exception:  # noqa: BLE001
        return
    w13z = getattr(layer, "_hpu_routed_w13_zp", None)
    w2z = getattr(layer, "_hpu_routed_w2_zp", None)
    if w13z is None or w2z is None:
        return  # AWQ per-expert zp -> fused path ineligible -> no stacked zp required
    E = int(layer._hpu_local_experts)
    existing = getattr(layer, "_hpu_fused_w13_zp", None)
    if existing is not None and existing.shape[0] == E:
        return
    # Broadcast the shared [groups, out//8] sym zero to a CONTIGUOUS stacked [E, groups,
    # out//8] held leaf (every expert identical for GPTQ-sym). Built here at load time so
    # the .contiguous() runs OUTSIDE capture -> the result is a persistent recipe input,
    # not an in-graph intermediate. .contiguous() (not a stride-0 expand view) so the
    # wrapper's index_select reads a real per-expert stride.
    layer._hpu_fused_w13_zp = w13z.unsqueeze(0).expand(E, *w13z.shape).contiguous()
    layer._hpu_fused_w2_zp = w2z.unsqueeze(0).expand(E, *w2z.shape).contiguous()
    if not _alloc_zp_logged[0]:
        _alloc_zp_logged[0] = True
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: _alloc_fused_zp materialized persistent stacked GPTQ-sym "
            "zero-points w13_zp%s / w2_zp%s on %s at weight-load (outside capture) -- the "
            "LAST multi-bucket-capture fix: the fused op's zp index_select source is now a "
            "held leaf (recipe input), surviving every decode bucket like the weight "
            "nn.Parameters.",
            tuple(layer._hpu_fused_w13_zp.shape),
            tuple(layer._hpu_fused_w2_zp.shape), layer._hpu_fused_w13_zp.device)


def _build_fp8_experts(layer, has_zp: bool) -> None:
    """LOAD-TIME int4 -> fp8 ALL-EXPERT conversion (HPU_MOE_FP8_EXPERTS=1).

    For each LOCAL expert, dequant its int4 weight to bf16 with convert_from_uint4 (the ONLY
    proven out_dtype -- exactly what _routed_moe / the stock op use), then cast bf16 -> fp8
    with the SELF-CONSISTENT cast_to_fp8_v2 round-trip from fp8_routed_moe (the same proven
    apply_fp8_linear_hpu / dynamic_quant path), and STASH the resident fp8 weight + a single
    per-tensor FORWARD recovery scale per weight tensor. Never int4 -> fp8 directly (that hits
    the Gaudi-2 fn-vs-fnuz 2x bug). Stored on the layer as held leaves (nn.Parameters):
        layer._fp8_w13   [E, H, 2I] float8_e4m3fn   layer._fp8_w13_s  [E] float32 (forward scale)
        layer._fp8_w2    [E, I, H ] float8_e4m3fn   layer._fp8_w2_s   [E] float32
    apply_monolithic's all-expert path then calls fp8_allexpert_moe against these (zero per-step
    dequant). GPTQ-symmetric int4 only (AWQ per-expert zp -> ineligible, no-op; stock path runs).

    Done PER EXPERT with a mark_step between experts so the transient bf16 weight (~38 MB) is
    freed before the next -- the resident footprint grows only by the half-size fp8 weight.
    When _FP8_FREE_INT4 (default ON under the flag) the stacked int4 expert weights + the
    moe_op's per-expert references are dropped AFTER the build (fp8 90.6 GB + int4 46.7 GB =
    137 GB > 96 GB/card would OOM; fp8 alone ~90.6 GB/card fits tight). Built from the SAME
    stacked tensors the routed path uses (layer.w13_qweight [E,H,2I/8] int32, layer.w13_scales
    [E,H/GS,2I] bf16), so it composes with the repack cache / eager / stacked-layout paths
    identically. No-op unless HPU_MOE_FP8_EXPERTS=1."""
    if not _FP8_EXPERTS:
        return
    if _FP8_FUSED:
        # FUSED fp8 op mode (HPU_MOE_FP8_FUSED=1): the fp8 expert path is the GC-captured
        # fused fp8 op, which DEQUANTS THE INT4 WEIGHTS IN-GRAPH (it takes int4-packed inputs,
        # like the int4 fused op). So there is NOTHING to convert at load and we MUST keep the
        # int4 weights live (do NOT build/free the resident-fp8 tensors). Leaving _fp8_w13
        # unset keeps the existing resident-fp8 branch (apply_monolithic ~L1576) untaken; the
        # fused-fp8 dispatch (added alongside the int4 fused dispatch) handles this layer. The
        # stacked GPTQ-sym zp the fused op index_selects is built by _alloc_fused_zp (its gate
        # now includes _FP8_FUSED). No-op here. (AWQ -> the fused-fp8 dispatch's silu/zp gate
        # falls back to the stock int4 op, identical to the int4 fused path.)
        return
    if has_zp:
        # AWQ per-expert zero-point: this fp8 all-expert path is GPTQ-symmetric only (a single
        # shared zp). Leave the layer on the stock int4 path (apply_monolithic checks _fp8_w13).
        return
    if getattr(layer, "_fp8_w13", None) is not None:
        return
    from .fp8_routed_moe import _quant_weight_fp8, FP8_DTYPE
    w13z = layer._hpu_routed_w13_zp
    w2z = layer._hpu_routed_w2_zp
    if w13z is None or w2z is None:
        return  # not GPTQ-sym (defensive; mirrors has_zp gate)
    E = int(layer._hpu_local_experts)
    # The build reads the stacked [E,...] int4 weights off layer.w13_qweight/w2_qweight. Under
    # the perexpert cache layout those are (E,1,1) placeholders (int4 lives only in moe_op) ->
    # LOUD rather than silently converting a placeholder. (The cache path also guards this.)
    if (layer.w13_qweight.dim() != 3 or layer.w13_qweight.shape[0] != E
            or layer.w13_qweight.shape[1] <= 1):
        raise RuntimeError(
            "hpu-moe-ext: HPU_MOE_FP8_EXPERTS=1 but layer.w13_qweight is not the stacked "
            f"[E,in,out//8] int4 weight (got {tuple(layer.w13_qweight.shape)}). Use the "
            "default stacked repack layout (HPU_MOE_REPACK_LAYOUT=stacked).")
    dev = layer.w13_scales.data.device
    dt = layer.w13_scales.dtype          # bf16 (the proven convert_from_uint4 out_dtype)
    H = int(layer.w2_scales.shape[-1])      # down out == hidden
    I2 = int(layer.w13_scales.shape[-1])     # w13 out == 2*I (gate||up)
    I = int(layer.w2_qweight.shape[1])       # w2_qweight is [E, I, H//8] -> in == I == moe_intermediate

    w13_fp8 = torch.empty((E, H, I2), dtype=FP8_DTYPE, device=dev)
    w2_fp8 = torch.empty((E, I, H), dtype=FP8_DTYPE, device=dev)
    w13_s = torch.empty((E,), dtype=torch.float32, device=dev)
    w2_s = torch.empty((E,), dtype=torch.float32, device=dev)

    import habana_frameworks.torch as _htorch
    _lazy = _htorch.is_lazy()

    def _co(t):
        # Same lazy storage-offset safety the routed path uses: a per-expert slice of the
        # stacked [E,...] param carries a nonzero storage_offset for e>0; clone under lazy so
        # convert_from_uint4 reads offset 0 (eager tolerates the offset -> skip the copy).
        return t.clone() if _lazy else t

    for e in range(E):
        # (1) int4 -> bf16 (the ONLY proven convert_from_uint4 out_dtype). Per-expert; the
        # bf16 weight is transient and freed by the per-expert mark_step below.
        w13_bf16 = torch.ops.hpu.convert_from_uint4(
            _co(layer.w13_qweight.data[e]), _co(layer.w13_scales.data[e]), w13z, dt, None)  # [H,2I]
        w2_bf16 = torch.ops.hpu.convert_from_uint4(
            _co(layer.w2_qweight.data[e]), _co(layer.w2_scales.data[e]), w2z, dt, None)      # [I,H]
        # (2) bf16 -> fp8, per-tensor FORWARD scale (the self-consistent cast_to_fp8_v2
        # round-trip == dynamic_quant(single_scale=True); real_w ~= w_fp8 * s).
        f13, s13 = _quant_weight_fp8(w13_bf16)
        f2, s2 = _quant_weight_fp8(w2_bf16)
        w13_fp8[e].copy_(f13)
        w2_fp8[e].copy_(f2)
        w13_s[e] = s13.reshape(())
        w2_s[e] = s2.reshape(())
        htorch.mark_step()   # free this expert's transient bf16 before the next
        # FIX 2026-06-19: mark_step alone does NOT flush in the process_weights_after_loading
        # context -> the per-expert convert/cast/copy_ graphs ACCUMULATE across all E experts
        # (the slice ops for e=28..31 were still live at the crash) until ValidateSyncInputTensors
        # (hpu_lazy_tensors.cpp:871) rejects the oversized graph at load. Force an actual device
        # flush so each expert's transient bf16 executes + frees before the next. Load-time only.
        if _lazy:
            _htorch.hpu.synchronize()

    layer._fp8_w13 = torch.nn.Parameter(w13_fp8, requires_grad=False)
    layer._fp8_w2 = torch.nn.Parameter(w2_fp8, requires_grad=False)
    layer._fp8_w13_s = torch.nn.Parameter(w13_s, requires_grad=False)
    layer._fp8_w2_s = torch.nn.Parameter(w2_s, requires_grad=False)
    htorch.mark_step()

    # D2: per-expert weight/scale VIEW tuples for the STOCK batched torch.ops.hpu.mixture_of_experts
    # intrinsic (the apply_monolithic fp8 fast path). Built ONCE here -- never per forward (rebuilding
    # E tuples every step is exactly the host overhead D2 removes). Mirrors ops.py
    # VllmMixtureOfExpertsOpFP8PerChannel._cache_weight_lists. The intrinsic wants a Python tuple of
    # per-expert 2-D weight tensors (w12=[in,out] gate||up = _fp8_w13[e] [H,2I]; w3=[in,out] down =
    # _fp8_w2[e] [I,H]) + a tuple of per-expert FORWARD recovery scales as [1] (d_scale_w12/d_scale_w3
    # -> fp8_gemm_v2 B_scale_inv; real_w ~= w_fp8 * s).
    _build_fp8_expert_views(layer)

    # LOAD-TIME VERIFY (HPU_MOE_FP8_EXPERTS_VERIFY=1): A/B the stored-fp8 all-expert output vs
    # the stock int4 all-expert op on a synthetic routed batch; LOUD if relNorm exceeds the fp8
    # tol (the 2x fnuz bug / a transpose / a wrong scale slot would blow this). .item() host
    # sync -> LOAD ONLY (never the hot path). Run BEFORE freeing int4 (the int4 op is the oracle).
    if _FP8_EXPERTS_VERIFY:
        _verify_fp8_experts(layer)

    # FREE the int4 source (fp8 + int4 both resident = 137 GB > 96 GB/card -> OOM). Drop the
    # stacked int4 params AND the moe_op per-expert references (views of them under the stacked
    # layout), then synchronize so the pool reclaims the blocks. Keep the int4 scales (tiny) for
    # the symmetric-zp rebuild only if something still needs them -- the fp8 path does not, so
    # drop them too. The all-expert path now reads ONLY _fp8_*.
    if _FP8_FREE_INT4:
        _free_int4_experts(layer)

    if not _fp8_experts_logged[0]:
        _fp8_experts_logged[0] = True
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: HPU_MOE_FP8_EXPERTS on -- converted %d local int4 experts -> resident "
            "fp8 (w13 %s / w2 %s, per-tensor forward scales) on %s; free_int4=%s, verify=%s. "
            "apply_monolithic all-expert path now runs fp8_allexpert_moe (zero per-step dequant).",
            E, tuple(layer._fp8_w13.shape), tuple(layer._fp8_w2.shape), dev,
            _FP8_FREE_INT4, _FP8_EXPERTS_VERIFY)


def _build_fp8_expert_views(layer) -> None:
    """Cache the per-expert fp8 weight/scale VIEW tuples the stock batched
    torch.ops.hpu.mixture_of_experts intrinsic consumes (the D2 fast path in
    apply_monolithic). Built ONCE at load -- NEVER per forward (rebuilding E tuples every
    step is exactly the host overhead D2 removes). Mirrors ops.py
    VllmMixtureOfExpertsOpFP8PerChannel._cache_weight_lists.

    LOUD guard: the stock int4 oracle (VllmMixtureOfExpertsOpWNA16, our numeric reference)
    runs the intrinsic in a SINGLE call only when num_experts <= MAX_EXPERTS_PER_SLICE; above
    that it splits into moe_n_slice sub-windows, each its own experts_min/max + accumulate
    (ops.py:1471/1501). The D2 fp8 path emits exactly ONE call, so it is only equivalent under
    one slice. For the GLM serve (256 experts / TP8 EP = 32 local, MAX_EXPERTS_PER_SLICE=64)
    E=32<=64 holds; assert it so a future >MAX_EXPERTS_PER_SLICE-local config fails LOUD at load
    instead of silently zeroing the experts beyond the first slice (the experts_min!=0 zeroing
    bug the moe_op LOCAL-window build defends against)."""
    import habana_frameworks.torch as _htorch
    from vllm_gaudi.extension.ops import MAX_EXPERTS_PER_SLICE
    E = int(layer._fp8_w13.shape[0])
    _max_per_slice = MAX_EXPERTS_PER_SLICE if MAX_EXPERTS_PER_SLICE > 0 else E
    if E > _max_per_slice:
        raise RuntimeError(
            "hpu-moe-ext D2: %d local fp8 experts > MAX_EXPERTS_PER_SLICE=%d -- the stock "
            "single-call mixture_of_experts fp8 path would zero experts beyond the first slice "
            "(the EP experts_min!=0 zeroing bug; the int4 moe_op splits into moe_n_slice here). "
            "Multi-slice fp8 is not implemented; raise MAX_EXPERTS_PER_SLICE or add slicing."
            % (E, _max_per_slice))
    # Under LAZY ONLY: the intrinsic captures the weight tuple into the decode HPUGraph and
    # SynapseAI rejects storage-shared slices (e>0 carries a nonzero storage_offset into the
    # stacked [E,...] param) -- the SAME constraint the int4 build handles with _co/_cl. In
    # eager/compile a slice with offset!=0 is fine; cloning there would DOUBLE the resident fp8
    # (~90 GB) and OOM, so gate the clone on is_lazy().
    _cl = (lambda t: t.clone()) if _htorch.is_lazy() else (lambda t: t)
    layer._fp8_w13_views = tuple(_cl(layer._fp8_w13.data[e]) for e in range(E))
    layer._fp8_w2_views = tuple(_cl(layer._fp8_w2.data[e]) for e in range(E))
    layer._fp8_w13_s_views = tuple(_cl(layer._fp8_w13_s.data[e].reshape(1)) for e in range(E))
    layer._fp8_w2_s_views = tuple(_cl(layer._fp8_w2_s.data[e].reshape(1)) for e in range(E))
    htorch.mark_step()


def _free_int4_experts(layer) -> None:
    """Drop the stacked int4 expert weights + the moe_op's per-expert references after the
    fp8 build, and synchronize so the Habana pool reclaims the blocks. The fp8 all-expert path
    reads only layer._fp8_*; the int4 stacked weights/scales are dead weight once fp8 exists,
    and fp8 (90.6 GB) + int4 (46.7 GB) = 137 GB cannot co-reside on a 96 GB card. Replace the
    big params with 1-element int32 placeholders (same shape signal create_weights uses) so any
    incidental attribute access does not crash; the storage is freed."""
    ph = lambda: torch.nn.Parameter(torch.zeros((1, 1, 1), dtype=torch.int32,
                                                device=layer.w13_scales.data.device),
                                    requires_grad=False)
    moe_op = getattr(layer, "moe_op", None)
    if moe_op is not None:
        # Release the per-expert weight references (views of the stacked params under the
        # stacked layout, fresh tensors under perexpert). Clearing them drops the only other
        # holders so the storage frees. The monolithic op is NOT used on the fp8 path.
        for lst in (getattr(moe_op, "w13_list", []), getattr(moe_op, "w2_list", [])):
            for w in lst:
                try:
                    w.weight = None
                    w.weight_packed = None
                except Exception:  # noqa: BLE001
                    pass
    for nm in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales"):
        if getattr(layer, nm, None) is not None:
            setattr(layer, nm, ph())
    # CRITICAL (BUG #1 fix): the routed-gather gate in apply_monolithic
    # (`_ROUTED_K>0 and tokens*top_k<=_ROUTED_K and _hpu_routed_w13_zp is not None`) and the
    # _routed_moe / _fused_moe_routed bodies key SOLELY on _hpu_routed_w13_zp being non-None.
    # We just replaced w13_qweight/w2_qweight with (1,1,1) placeholders, so any bs=1 forward
    # (plain decode AND every MTP draft forward: 1*top_k<=K) would still ENTER the routed path
    # and index_select(0, active in [0,K)) a dim-0-size-1 placeholder -> out-of-bounds crash
    # (eager) / silent garbage (lazy capture). Once int4 is freed, the fp8 all-expert path is
    # the ONLY valid path for THIS layer, so null the routed/fused zp handles to force the gate
    # to fall through to the fp8 branch for ALL batch sizes. fp8_allexpert_moe does NOT use these
    # handles, so this is safe. (If int4 is kept -- HPU_MOE_FP8_FREE_INT4=0 -- this is not called
    # and the int4 routed path stays available, which is correct since the weights are live.)
    for nm in ("_hpu_routed_w13_zp", "_hpu_routed_w2_zp",
               "_hpu_fused_w13_zp", "_hpu_fused_w2_zp"):
        if getattr(layer, nm, None) is not None:
            setattr(layer, nm, None)
    import gc as _gc
    import habana_frameworks.torch as _ht
    _gc.collect()
    if _ht.is_lazy():
        htorch.mark_step()
    _ht.hpu.synchronize()


@torch.compiler.disable
def _verify_fp8_experts(layer) -> None:
    """LOAD-TIME adversarial numeric gate for HPU_MOE_FP8_EXPERTS (the #1 risk = the fnuz 2x
    bug). Build a synthetic single-token routed batch over this rank's LOCAL experts, run BOTH
    the stored-fp8 all-expert path (fp8_allexpert_moe) and the stock int4 all-expert op
    (layer.moe_op with the LOCAL-window remap), and assert Frobenius relNorm < _FP8_EXPERTS_REL_TOL.
    LOUD (RuntimeError) on mismatch -- a 2x output would land relNorm ~1.0, far above the few-%
    fp8-rounding floor, so this catches the asymmetric-write bug, a transpose, or a wrong scale
    slot at LOAD, before serving. .item() host sync -> never call on the decode path."""
    import logging
    log = logging.getLogger("vllm")
    from .fp8_routed_moe import fp8_allexpert_moe
    E = int(layer._hpu_local_experts)
    emin = int(layer._hpu_experts_min)
    dev = layer._fp8_w13.data.device
    H = int(layer._fp8_w2.shape[-1])         # down out == hidden (int4 scales may be freed already)
    K = min(E, max(1, int(getattr(layer, "top_k", 8))))
    # One synthetic token routing to K distinct LOCAL experts (global ids = emin + [0..K-1]).
    topk_ids = (emin + torch.arange(K, dtype=torch.int64, device=dev)).view(1, K)
    tw = torch.full((1, K), 1.0 / K, dtype=torch.bfloat16, device=dev)
    x = (torch.randn(1, H, dtype=torch.bfloat16, device=dev) * 0.5)
    htorch.mark_step()
    fp8_out = fp8_allexpert_moe(
        x, topk_ids, tw,
        layer._fp8_w13.data, layer._fp8_w13_s.data,
        layer._fp8_w2.data, layer._fp8_w2_s.data,
        experts_min=emin, activation="silu")
    # Stock int4 oracle: moe_op was built with the LOCAL window [0,E-1] (the EP fix), so feed
    # LOCAL ids; non-local slots -> out-of-window with weight 0 (here all K are local).
    rtid = topk_ids - emin
    rtw = tw
    ref = layer.moe_op(x, rtid, rtw, permuted_weights=False, activation="silu")
    htorch.mark_step()
    diff = (fp8_out.float() - ref.float()).norm().item()
    rn = diff / (ref.float().norm().item() + 1e-9)
    md = (fp8_out.float() - ref.float()).abs().max().item()
    if not (ref.float().norm().item() > 1e-6):
        raise RuntimeError(
            "[fp8-experts-verify] stock int4 ref norm ~0 -- routing/weights wrong, cannot "
            "judge fp8 (LOAD aborted to avoid serving an unverified fp8 expert path).")
    if rn >= _FP8_EXPERTS_REL_TOL:
        raise RuntimeError(
            "[fp8-experts-verify] FAIL relNorm=%.4g (tol %.3g) maxAbs=%.4g on rank=%d -- too "
            "large to be fp8 rounding. The #1 suspect is the Gaudi-2 fn-vs-fnuz 2x bug "
            "(relNorm ~1.0 => convert_from_uint4 was handed an fp8 out_dtype, or a scale slot "
            "is inverted); also check the w13/w2 transpose (weights are [in,out], trans_B=False) "
            "and the per-tensor forward scale (gemm *_scale_inv must take s, cast uses 1/s). "
            "LOAD aborted -- NOT serving an incorrect expert path."
            % (rn, _FP8_EXPERTS_REL_TOL, md, _resolve_ep_rank(layer, -1)))
    log.info("[fp8-experts-verify] PASS relNorm=%.5g (tol %.3g) maxAbs=%.4g rank=%d",
             rn, _FP8_EXPERTS_REL_TOL, md, _resolve_ep_rank(layer, -1))
# Relative to this file: hpu_moe_ext/moe_wna16_hpu.py -> ../tpc_kernel/opbackend/build/...
# The op ships in TWO builds: the LAZY-frontend .so (captures into the decode HPU-graph
# under PT_HPU_LAZY_MODE=1 -> the fast serving path) and the EAGER-frontend .so. The
# eager .so does NOT load under lazy (its EagerOp symbols live only in the eager plugin)
# and vice-versa, so pick the right one by mode at load time.
_FUSED_SO_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "tpc_kernel", "opbackend", "build"))


def _fused_so_path():
    import habana_frameworks.torch.core as _htc
    name = "libmoe_fused_expert_lazy.so" if _htc.is_lazy() else "libmoe_fused_expert.so"
    return os.path.join(_FUSED_SO_DIR, name)
# Load the .so exactly once, lazily, the first time the fused path is taken (NOT at
# import) so a CPU-only `import hpu_moe_ext` — and every non-fused serve — never
# touches it. LOUD if the flag is on but the .so is missing or the op fails to
# register: the fused path was explicitly requested, so a silent fallback would hide
# a broken build. Returns the callable torch.ops.hpu_moe_ext.fused_expert handle.
_fused_op_loaded = [False]


def _load_fused_op():
    """Load libmoe_fused_expert.so once and return torch.ops.hpu_moe_ext.fused_expert.
    LOUD (RuntimeError) if HPU_MOE_FUSED_OP is on but the .so is absent / the op does
    not register. Idempotent: re-loading the same library is a torch no-op, but we
    short-circuit after the first success so the registration check runs only once."""
    if not _fused_op_loaded[0]:
        so = _fused_so_path()
        if not os.path.isfile(so):
            raise RuntimeError(
                "hpu-moe-ext: HPU_MOE_FUSED_OP=1 but the fused-expert op library is "
                f"missing:\n  {so}\nBuild it (CPU-only, inside the image) with "
                "`bash tpc_kernel/opbackend/build.sh` (eager) / `build_lazy.sh` (lazy), "
                "or unset HPU_MOE_FUSED_OP.")
        torch.ops.load_library(so)
        if not (hasattr(torch.ops, "hpu_moe_ext")
                and hasattr(torch.ops.hpu_moe_ext, "fused_expert")):
            raise RuntimeError(
                "hpu-moe-ext: loaded "
                f"{so}\nbut torch.ops.hpu_moe_ext.fused_expert did NOT register "
                "(check TORCH_LIBRARY(hpu_moe_ext) + the .so build).")
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded fused-expert op from %s "
            "(HPU_MOE_FUSED_OP=1, verify=%s)", so, _FUSED_VERIFY)
        _fused_op_loaded[0] = True
    return torch.ops.hpu_moe_ext.fused_expert


def _emit_moe_engage_manifest(layer, T) -> None:
    """Print the resolved-state ENGAGE-MANIFEST for the MoE path, ONCE per worker (==
    per TP rank). Resolves the SAME predicates apply_monolithic uses at THIS shape T,
    against THIS layer's resolved load-time state — so the banner is ground truth, not
    env intent. Pure output (default-ON via HPU_ENGAGE_MANIFEST); no device touch.

    The fields read the layer attributes set at weight-load (_fp8_w13, _hpu_routed_w13_zp,
    _hpu_fused_x_buf, _hpu_local_experts) + the module flags, so they report what the
    forward will ACTUALLY do, not what the env asked for. Called from apply_monolithic's
    first forward (T is the live token count of that first call)."""
    if not manifest_enabled():
        return
    # DEFENSIVE: the manifest is a DIAGNOSTIC and must NEVER break a live forward. Any
    # surprise in the layer attributes -> swallow + one-line warn, never propagate. (The
    # actual dispatch below is unaffected; only this banner is skipped.)
    try:
        rank = _resolve_ep_rank(layer, 0)
        E = int(getattr(layer, "_hpu_local_experts", 0) or 0)
        gptq_sym = getattr(layer, "_hpu_routed_w13_zp", None) is not None
        fp8_layer = _FP8_EXPERTS and getattr(layer, "_fp8_w13", None) is not None
        # --- resolve the MoE route the SAME way apply_monolithic does ---
        if _FP8_FUSED and _FP8_EXPERTS:
            if gptq_sym:
                moe = "FUSED-fp8-op" + ("(int4 live)" if _FP8_DIRECT else "")
            else:
                moe = "stock-int4 (AWQ -> fp8-fused ineligible)"
        elif fp8_layer:
            moe = "fp8-resident-allexpert(K clamps E=%d)" % E
        elif _ROUTED_K > 0 and gptq_sym:
            if _FUSED_OP:
                moe = "ROUTED-FUSED-op(K=%d)" % _ROUTED_K
            else:
                moe = "ROUTED-gather(K=%d)" % _ROUTED_K
        elif _FUSED_MONOLITHIC and gptq_sym:
            if _FUSED_MONO_MAX_T > 0:
                # T-capped: decode (T<=N) takes the monolithic; prefill (T>N) -> stock op.
                moe = ("FUSED-MONOLITHIC-op(T<=%d; T>%d->stock-int4-all-expert)"
                       % (_FUSED_MONO_MAX_T, _FUSED_MONO_MAX_T))
            else:
                moe = "FUSED-MONOLITHIC-op"
        else:
            moe = "stock-int4-all-expert"
        # --- fp8 mode ---
        if _FP8_FUSED and _FP8_EXPERTS:
            fp8 = "FUSED-DIRECT" if _FP8_DIRECT else "FUSED-2hop"
        elif fp8_layer:
            fp8 = "resident-allexpert"
        else:
            fp8 = "OFF"
        # --- bs>1 admission (the KBUCKET / MTP-verify lever) ---
        if _ROUTED_BSGT1:
            if _ROUTED_KBUCKET:
                bsgt1 = "admitted(KBUCKET %s)" % ",".join(
                    "%d:%d" % (t, k) for t, k in sorted(_ROUTED_KBUCKET.items()))
            else:
                bsgt1 = "admitted(conservative K=min(E,T*top_k))"
        else:
            bsgt1 = "all-expert(T>1 -> K clamps E=%d)" % E
        # --- x-buffer residency (the multi-bucket capture fix) ---
        xbuf = getattr(layer, "_hpu_fused_x_buf", None)
        if (_FUSED_OP or _FUSED_MONOLITHIC or _FP8_FUSED):
            if xbuf is not None:
                xb = "resident%s" % (tuple(xbuf.shape),)
            else:
                xb = "in-forward-clone (NOT multi-bucket-safe)"
        else:
            xb = "n/a (no fused op)"
        emit_manifest(rank, {
            "MoE": moe,
            "fp8": fp8,
            "bs>1": bsgt1,
            "x_buf": xb,
            "STRICT": "ON" if strict_enabled() else "off",
        }, once_key="moe")
    except Exception as _ex:  # noqa: BLE001 — a diagnostic must never break the forward
        import logging
        logging.getLogger("vllm").warning(
            "[ENGAGE-MANIFEST] MoE manifest emit skipped (non-fatal): %s: %s",
            type(_ex).__name__, _ex)


# The fused-expert path is the per-K-expert replacement for the routed fast path, so
# it is ONLY reachable inside the `_ROUTED_K > 0` gate (that gate is what GUARANTEES
# #distinct local experts <= K — the static-shape / no-host-sync invariant the fused
# op relies on). HPU_MOE_FUSED_OP=1 without HPU_MOE_ROUTED_K>0 would be a silent no-op
# (the routed branch never runs), so warn LOUDLY once at import.
if _FUSED_OP and _ROUTED_K <= 0:
    # The user asked for the fused routed op but it can NEVER run (no routed gate) -> an
    # UNINTENDED no-op. LOUD always; HARD-FAIL under HPU_ENGINE_STRICT (engage_fallback
    # with intended=True) so a fused A/B with a forgotten HPU_MOE_ROUTED_K crashes instead
    # of silently measuring the stock all-expert path.
    engage_fallback(
        "moe_wna16.import:fused-op-needs-routed-k",
        fast="HPU_MOE_FUSED_OP routed fused op",
        reason="HPU_MOE_ROUTED_K is unset/0 so the routed-gather gate never admits the "
               "fused path (it lives INSIDE that gate)",
        slow="stock all-expert mixture_of_experts",
        intended=True, strict_env="HPU_MOE_FUSED_OP (set HPU_MOE_ROUTED_K=<top_k>)")

# HPU_MOE_FP8_FUSED=1 is a NO-OP unless HPU_MOE_FP8_EXPERTS=1 is ALSO set: the entire
# fused-fp8 dispatch (apply_monolithic ~L1950, _emit_moe_engage_manifest ~L909/926,
# process_weights_after_loading ~L581, _alloc_fused_zp's gate) is `_FP8_FUSED and
# _FP8_EXPERTS` / guarded by `if not _FP8_EXPERTS: return`. With only HPU_MOE_FP8_FUSED=1
# the fp8 op's loader (_load_fp8_fused_op) is NEVER CALLED, NO "loaded FUSED fp8 op"
# banner prints, NO engage_fallback fires from inside the (never-entered) dispatch, and
# the layer SILENTLY runs the stock int4 all-expert path -- the EXACT false-null that made
# a HPU_MOE_FP8_FUSED=1 single-stream A/B measure the int4 fallback (flat ~68.8ms) as if it
# were the fused fp8 op. (HPU_MOE_FP8_EXPERTS is the master fp8 enable; with HPU_MOE_FP8_FUSED
# it KEEPS int4 resident -- the op dequants in-graph -- and does not build resident-fp8, but
# the flag is still the gate.) LOUD always; HARD-FAIL under HPU_ENGINE_STRICT (intended=True)
# so a fp8-fused run with a forgotten HPU_MOE_FP8_EXPERTS crashes instead of silently
# benchmarking int4. This MIRRORS the HPU_MOE_FUSED_OP-needs-ROUTED_K guard directly above.
if _FP8_FUSED and not _FP8_EXPERTS:
    engage_fallback(
        "moe_wna16.import:fp8-fused-needs-fp8-experts",
        fast="HPU_MOE_FP8_FUSED fused fp8 op (torch.ops.hpu_moe_ext_fp8.fused_expert_fp8)",
        reason="HPU_MOE_FP8_EXPERTS is unset so the fused-fp8 dispatch gate "
               "(`_FP8_FUSED and _FP8_EXPERTS`) is never entered -- the op's .so is "
               "never loaded and no banner/fallback ever fires",
        slow="stock int4 all-expert mixture_of_experts (the per-step-dequant path the "
             "fp8 op was supposed to ELIMINATE -> a FALSE NULL in any fp8 A/B)",
        intended=True,
        strict_env="HPU_MOE_FP8_FUSED (also set HPU_MOE_FP8_EXPERTS=1)")

# HPU_MOE_FP8_DIRECT=1 (fold the static weight scale -> one-node int4->fp8 dequant) is a
# NO-OP unless HPU_MOE_FP8_FUSED=1 (its scale is consumed ONLY by the fused fp8 op, and the
# _alloc_fused_fp8_scale gate is `_FP8_DIRECT and _FP8_FUSED`). With DIRECT on but FUSED off
# the op never runs, so the precomputed s_fp8 is dead. LOUD + strict (intended) so a DIRECT
# A/B with a forgotten HPU_MOE_FP8_FUSED cannot silently measure something else.
if _FP8_DIRECT and not _FP8_FUSED:
    engage_fallback(
        "moe_wna16.import:fp8-direct-needs-fp8-fused",
        fast="HPU_MOE_FP8_DIRECT one-node int4->fp8 weight dequant",
        reason="HPU_MOE_FP8_FUSED is unset so the fused fp8 op (the ONLY consumer of the "
               "precomputed per-expert fp8 scale) never runs",
        slow="no fp8 op at all (DIRECT scale is dead) -> the requested fast path cannot engage",
        intended=True,
        strict_env="HPU_MOE_FP8_DIRECT (also set HPU_MOE_FP8_FUSED=1 and HPU_MOE_FP8_EXPERTS=1)")

# Set True by the MLA-absorption monkeypatch (hpu-dsa-ext/deq_weight_move) DURING load only, so
# GPTQDequantToBF16.apply flushes (defrags) around the absorption's weight extraction — which OOMs
# on the fragmented :next pool. MUST stay False during inference (a mark_step inside the decode
# apply would split the HPU graph at every 8-bit linear -> terrible latency).
_DEQ_ABSORB = [False]
_DP_GATHER_LOGGED = [False]  # one-shot banner for the DP-attention token all-gather


def _hpu_to_cpu_chunked(t, target_bytes=1_048_576):
    """Move an HPU tensor to CPU in BYTE-sized row-chunks (~1MB each, regardless of the row
    width) so each staging allocation fits a fragmented device pool block. A whole-tensor (or
    even 128-row) .to("cpu") needs a contiguous 6-12MB staging block that the capped+fragmented
    lazy/:next pool can't serve near the process_weights peak (PT_DEVMEM fail despite free HBM).
    mark_step after each chunk reclaims that chunk's staging before the next. No-op for
    non-HPU / tiny tensors."""
    if not (hasattr(t, "device") and getattr(t.device, "type", "") == "hpu"):
        return t.detach().to("cpu")
    row_bytes = max(1, (t.numel() // t.shape[0]) * t.element_size()) if t.dim() >= 1 and t.shape[0] > 0 else t.element_size()
    chunk_rows = max(1, target_bytes // row_bytes)
    if t.dim() == 0 or t.shape[0] <= chunk_rows:
        out = t.detach().to("cpu")
        htorch.mark_step()
        return out
    out = torch.empty(tuple(t.shape), dtype=t.dtype, device="cpu")
    for _i in range(0, t.shape[0], chunk_rows):
        out[_i:_i + chunk_rows] = t[_i:_i + chunk_rows].detach().to("cpu")
        htorch.mark_step()
    return out


def _hpu_safe_select_experts(layer, hidden_states: torch.Tensor,
                             router_logits: torch.Tensor,
                             input_ids: Optional[torch.Tensor] = None):
    """Resolve routing semantics without invoking CUDA-only fused router ops.

    Modular vLLM can collapse ``n_group=topk_group=1`` into a non-grouped
    ``FusedTopKBiasRouter``.  ``layer.use_grouped_topk`` is therefore not a
    sufficient description of the routing algorithm: GLM-style sigmoid,
    correction-bias, normalization and routed scaling still apply.  Read the
    live router capabilities instead of dispatching on model names.
    """
    import torch.nn.functional as F

    router = getattr(layer, "router", None)
    if getattr(layer, "use_grouped_topk", False):
        selector = getattr(router, "select_experts", None)
        if selector is None:
            selector = getattr(router, "_select_experts", None)
        if not callable(selector):
            raise AttributeError(
                "hpu-moe-ext: grouped router has no expert selector")
        return selector(hidden_states=hidden_states,
                        router_logits=router_logits)

    def resolved(name, default=None):
        value = getattr(router, name, None) if router is not None else None
        if value is None:
            value = getattr(layer, name, default)
        return getattr(value, "value", value)

    scoring_func = str(resolved("scoring_func", "softmax")).lower()
    logits_fp32 = router_logits.to(torch.float32)
    if scoring_func == "softmax":
        scores = F.softmax(logits_fp32, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits_fp32)
    elif scoring_func == "sqrtsoftplus":
        scores = torch.sqrt(F.softplus(logits_fp32))
    else:
        raise ValueError(
            f"hpu-moe-ext: unsupported HPU routing score {scoring_func!r}")

    correction_bias = resolved("e_score_correction_bias")
    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.to(
            device=scores.device, dtype=scores.dtype).reshape(1, -1)
    else:
        scores_for_choice = scores

    top_k = int(resolved("top_k", getattr(layer, "top_k", 1)))
    hash_indices_table = getattr(router, "_hash_indices_table", None)
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError(
                "hpu-moe-ext: hash router requires input_ids for selection")
        topk_ids = hash_indices_table[input_ids].reshape(scores.shape[0], -1)
        topk_ids = topk_ids[:, :top_k]
    else:
        # Match the pre-modular HPU path's deterministic torch.topk ordering.
        topk_ids = torch.topk(scores_for_choice, top_k, dim=-1,
                              sorted=True).indices

    # Correction bias chooses experts but never becomes a mixture weight.
    topk_weights = scores.gather(1, topk_ids.to(torch.int64))
    if bool(resolved("renormalize", True)):
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True)
    routed_scaling_factor = float(resolved("routed_scaling_factor", 1.0))
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    num_shared = int(resolved("num_fused_shared_experts", 0))
    if num_shared > 0:
        global_experts = int(resolved("global_num_experts", scores.shape[-1]))
        shared_ids = torch.arange(
            global_experts, global_experts + num_shared,
            dtype=topk_ids.dtype, device=topk_ids.device,
        ).expand(topk_ids.shape[0], num_shared)
        shared_weight = float(resolved("shared_expert_weight", 1.0))
        shared_weights = torch.full(
            (topk_weights.shape[0], num_shared), shared_weight,
            dtype=topk_weights.dtype, device=topk_weights.device)
        topk_ids = torch.cat((topk_ids, shared_ids), dim=-1)
        topk_weights = torch.cat((topk_weights, shared_weights), dim=-1)

    return topk_weights.to(hidden_states.dtype), topk_ids


class HPUWNA16MoEConfig:
    """Minimal config carrying just what MoeWNA16Method.create_weights /
    get_fused_moe_quant_config read. Built directly from a checkpoint's HF
    quantization_config so we BYPASS MoeWNA16Config.__init__'s AWQ
    device-capability gate (which rejects HPU, capability == -1)."""

    def __init__(self, linear_quant_method: str, weight_bits: int,
                 group_size: int, has_zp: bool,
                 modules_to_not_convert: Optional[list[str]],
                 full_config: dict[str, Any]):
        self.linear_quant_method = linear_quant_method
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // weight_bits
        self.lm_head_quantized = False
        self.use_marlin = False
        self.modules_to_not_convert = modules_to_not_convert or []
        self.full_config = full_config

    @classmethod
    def get_name(cls) -> str:
        # The FusedMoE weight loader branches on this; the loaded layout matches
        # upstream moe_wna16, so identify as such.
        return "moe_wna16"

    @classmethod
    def from_hf(cls, qcfg: dict[str, Any]) -> "HPUWNA16MoEConfig":
        method = qcfg.get("quant_method", "").lower()
        if method in ("auto-round", "auto_round"):
            # auto_round:auto_gptq packing == symmetric int4 gptq.
            method = "gptq"
        bits = qcfg.get("bits")
        group_size = qcfg.get("group_size")
        if method == "gptq":
            has_zp = not qcfg.get("sym", True)
            mtnc = qcfg.get("modules_to_not_convert") or []
        elif method in ("awq", "awq_marlin"):
            has_zp = bool(qcfg.get("zero_point", True))
            mtnc = qcfg.get("modules_to_not_convert") or []
        else:
            raise ValueError(f"hpu-moe-ext: unsupported MoE quant_method {method!r}")
        return cls(method, bits, group_size, has_zp, mtnc, qcfg)


class MoeWNA16HPUMethod(MoeWNA16Method):
    """GPTQ/AWQ W4A16 FusedMoE on Gaudi via VllmMixtureOfExpertsOpWNA16."""

    # FusedMoE dispatch on HPU calls apply_monolithic() when this is True.
    @property
    def is_monolithic(self) -> bool:
        return True

    def _quant_type(self):
        return _WNA16_TYPES[self.quant_config.weight_bits]

    # --- OFFLINE PRE-REPACK — create_weights override (LOAD path only) ---------
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        """When HPU_MOE_REPACK_LOAD is set, allocate the per-expert checkpoint
        params as tiny int32 PLACEHOLDERS and neutralize their weight_loader, so:
          * the ~52GB uint8 fused `w13_qweight`/`w2_qweight` originals are NEVER
            the load target (under lazy they'd be pinned by the repack and never
            freed -> no KV room). With no big original, there is nothing to trap.
          * vLLM's per-expert checkpoint loader does NOT try to fill the
            (now wrong-shaped) params and does NOT error on the shape change.
        Everything real is loaded from the cache in _process_weights_from_cache.

        When the env is UNSET we delegate verbatim to the upstream create_weights
        (inherited via MoeWNA16Method) -> the eager :1.24 path is byte-identical."""
        from . import repack_cache
        # PER-LAYER fallback: when LOAD is on but THIS layer has no cache file (e.g. the
        # DeepSeek-MTP layer 78 added after the 0..77 cache was minted), build the FULL
        # uint8 params so process_weights can repack it live. Mark it so the matching
        # process path repacks instead of cache-loads. (No cache dir at all -> same.)
        _load_dir = repack_cache.load_dir()
        if not _load_dir or not repack_cache.cache_exists(layer, _load_dir):
            if _load_dir:
                layer._hpu_repack_uncached = True  # tell process_weights to eager-repack
            return super().create_weights(
                layer, num_experts, hidden_size,
                intermediate_size_per_partition, params_dtype,
                **extra_weight_attrs)

        # LOAD path. Build the upstream params (gives us layer.quant_config,
        # group_size, the wrapped weight_loader, and correctly-SHAPED scale/zero/
        # g_idx params), THEN shrink the two big qweight params to placeholders and
        # disarm the loaders on every per-expert checkpoint param. We still keep
        # the scale/qzero params allocated at full shape (they're tiny vs qweight)
        # but disarm their loaders too — the real values come from the cache.
        super().create_weights(
            layer, num_experts, hidden_size,
            intermediate_size_per_partition, params_dtype,
            **extra_weight_attrs)

        def _noop_loader(param, loaded_weight, weight_name, shard_id,
                         expert_id, return_success: bool = False, **kw):
            # Swallow every per-expert checkpoint tensor for these params: the
            # real (already-repacked) data is loaded from the cache later. Report
            # success so the model loader marks the checkpoint tensor consumed and
            # does NOT raise "missing weight" / "unexpected weight".
            return True if return_success else None

        # Replace the 52GB uint8 fused weights with 1-element int32 placeholders.
        # int32 (vs the original uint8) signals downstream that these are NOT the
        # load target; _process_weights_from_cache overwrites them with the real
        # cached int32 tensors. requires_grad=False, same as upstream.
        for nm in ("w13_qweight", "w2_qweight"):
            ph = torch.nn.Parameter(
                torch.zeros((num_experts, 1, 1), dtype=torch.int32),
                requires_grad=False)
            ph.weight_loader = _noop_loader
            setattr(ph, "_hpu_repack_placeholder", True)
            layer.register_parameter(nm, ph)

        # Disarm the loaders on the remaining per-expert checkpoint params so the
        # loader fills nothing into them (cache supplies the real values). Keep the
        # params (correct shapes) so any code that inspects them still works.
        for nm in ("w13_scales", "w2_scales", "w13_qzeros", "w2_qzeros"):
            p = getattr(layer, nm, None)
            if p is not None:
                p.weight_loader = _noop_loader

    # --- repack one expert-stack of uint8-packed int4 -> int32-packed -------
    def _repack_weight(self, qweight_u8: torch.Tensor) -> torch.Tensor:
        """Upstream loads experts as uint8 [E, out, in//2]; reinterpret to int32 [out, in//8],
        transpose so input is the packed dim, unpack, re-pack along output -> int32 [E, in, out//8]
        (the layout the HPU mixture_of_experts op consumes). Fresh contiguous output (lazy-safe;
        the in-place variant aliased storage and crashed the MoE op under lazy). The freed original
        is unpinned per-layer by a synchronize in process_weights so the pool reuses it."""
        qt = self._quant_type()
        outs = []
        for e in range(qweight_u8.shape[0]):
            w_i32 = qweight_u8[e].contiguous().view(torch.int32)   # [out, in//8]
            vals = unpack_quantized_values_into_int32(
                w_i32.transpose(0, 1).contiguous(), qt, packed_dim=0)  # [in, out]
            outs.append(pack_quantized_values_into_int32(vals, qt, packed_dim=1))  # [in, out//8]
            if htorch.is_lazy():
                htorch.mark_step()
        result = torch.stack(outs, dim=0)
        htorch.mark_step()
        return result

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Guard against being called twice on the same layer: the second call
        # would re-pack the already-int32 weights as if they were uint8 -> garbage.
        if getattr(layer, "_hpu_moe_ext_done", False):
            return
        # OFFLINE PRE-REPACK — LOAD path: weights are already int32 op-ready on disk.
        # Skip the uint8 repack entirely (the whole point: nothing to free/pin).
        # EXCEPT a layer flagged _hpu_repack_uncached (no cache file for it, e.g. the
        # MTP layer 78): it loaded its FULL uint8 weights in create_weights, so fall
        # through to the live eager repack below (and SAVE it if SAVE is on).
        from . import repack_cache
        if repack_cache.load_dir() and not getattr(layer, "_hpu_repack_uncached", False):
            return self._process_weights_from_cache(layer)
        has_zp = self.quant_config.has_zp

        w13_packed = self._repack_weight(layer.w13_qweight.data)   # [E, in, out//8]
        w2_packed = self._repack_weight(layer.w2_qweight.data)
        layer.w13_qweight = torch.nn.Parameter(w13_packed, requires_grad=False)
        layer.w2_qweight = torch.nn.Parameter(w2_packed, requires_grad=False)
        # Unpin this layer's just-freed uint8 originals so the Habana pool REUSES their blocks for
        # the NEXT layer instead of trapping them (+0.6GB/layer -> +42GB -> ~94GB pool, no KV room).
        # The freed block stays pinned by the repack's read stream-event until a synchronize
        # processes it; mark_step only flushes the graph, it does NOT unpin. Done PER LAYER (while
        # HBM still has room — a single post-load synchronize OOMs once the pool is full). Lazy only.
        if htorch.is_lazy():
            import gc as _gc
            import habana_frameworks.torch as _ht
            _gc.collect()
            _ht.hpu.synchronize()

        # scales loaded [E, out, num_groups] -> op wants [E, num_groups, out]
        w13_scale = layer.w13_scales.data.transpose(1, 2).contiguous()
        w2_scale = layer.w2_scales.data.transpose(1, 2).contiguous()
        layer.w13_scales = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_scales = torch.nn.Parameter(w2_scale, requires_grad=False)

        num_experts = layer.local_num_experts
        ep_shift = _resolve_ep_rank(layer) * num_experts
        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        # GLM-5.1 / EP FIX: the HPU mixture_of_experts intrinsic returns ~0 for the routed
        # experts when experts_min != 0 (non-zero GLOBAL window under expert-parallel) — the
        # routed contribution silently vanishes -> only the shared expert survives -> token
        # salad. The working path (and _routed_moe, which the 397B uses via HPU_MOE_ROUTED_K,
        # masking this bug) uses a LOCAL window [0, num_experts-1] with ids remapped
        # global->local. Build the op with the LOCAL window; apply_monolithic does the remap.
        layer.moe_op = VllmMixtureOfExpertsOpWNA16(num_experts, 0, num_experts - 1)

        dev = layer.w13_qweight.data.device
        if not has_zp:
            # GPTQ symmetric: constant zero-point = type midpoint (bias), shared
            # across experts. Shape [num_groups, out] packed -> [num_groups, out//8].
            w13_zp = self._sym_zero(w13_scale.shape[1], w13_scale.shape[2], dev)
            w2_zp = self._sym_zero(w2_scale.shape[1], w2_scale.shape[2], dev)

        # 2d routed-gather: stash the stacked-tensor handles + shared sym zero so the
        # fast path can index_select K experts. GPTQ-sym only (has_zp -> AWQ -> None -> fallback).
        layer._hpu_routed_w13_zp = w13_zp if not has_zp else None
        layer._hpu_routed_w2_zp = w2_zp if not has_zp else None
        layer._hpu_experts_min = experts_min
        layer._hpu_local_experts = num_experts
        # PERSISTENT static-input x buffer (multi-bucket capture fix). Allocated HERE,
        # at weight-load (outside any HPU-graph capture), so it is a held leaf the layer
        # owns -> a recipe INPUT survived across every decode bucket (see _alloc_fused_x_buf).
        _alloc_fused_x_buf(layer)
        # PERSISTENT stacked zero-points (the LAST multi-bucket capture fix). Built HERE,
        # at weight-load (outside capture), so the fused op's zp index_select source is a
        # held leaf that survives every decode bucket -- not an in-forward .contiguous()
        # intermediate the free_inplace pass reclaims. See _alloc_fused_zp.
        _alloc_fused_zp(layer)
        # PERSISTENT per-expert DIRECT int4->fp8 recovery scales (HPU_MOE_FP8_DIRECT=1).
        # Precomputed HERE at weight-load (the weights are static) so the fused fp8 op can
        # dequant int4 STRAIGHT to fp8 in one node -- no bf16 intermediate, no per-step
        # weight amax/cast. No-op unless HPU_MOE_FP8_DIRECT=1. See _alloc_fused_fp8_scale.
        _alloc_fused_fp8_scale(layer)

        # .clone() (fresh storage_offset=0) is REQUIRED only for lazy HPUGraph capture
        # (SynapseAI rejects storage-shared slices). In eager/compile it merely DOUBLES
        # expert-weight memory at load -> OOMs the larger GLM-5.1 744B load. Gate it:
        # clone in lazy, pass the slice directly (offset!=0 is fine without graph capture).
        import habana_frameworks.torch as _htorch
        _clone_slices = _htorch.is_lazy()
        def _cl(t):
            return t.clone() if _clone_slices else t

        for e in range(num_experts):
            # .clone() is REQUIRED for HPU lazy-graph (HPUGraph) capture.
            # layer.w13_qweight.data[e] is a slice of the stacked [E, ...] Parameter:
            # for e > 0 the storage_offset is non-zero (= e * stride[0] elements into
            # the parent buffer). The HPU SynapseAI graph validator rejects any tensor
            # whose storage is shared with a larger parent:
            #   "Neither storage attached to input tensor, not its view.
            #    Strided Params Has Value: 0"
            # IMPORTANT: .contiguous() on an already-contiguous slice returns the SAME
            # tensor (same data_ptr, same storage_offset) — it does NOT create a new
            # independent storage. Only .clone() allocates a fresh tensor with
            # storage_offset=0, satisfying the graph capture constraint.
            # This happens once at weight-load time so the copy cost is negligible.
            layer.moe_op.w13_list[e].set_weight_packed(_cl(layer.w13_qweight.data[e]))
            layer.moe_op.w2_list[e].set_weight_packed(_cl(layer.w2_qweight.data[e]))
            layer.moe_op.w13_list[e].set_weight_scale(_cl(layer.w13_scales.data[e]))
            layer.moe_op.w2_list[e].set_weight_scale(_cl(layer.w2_scales.data[e]))
            if has_zp:
                # AWQ asymmetric: _awq_zero internally calls pack_quantized_values_into_int32
                # which returns a freshly-allocated tensor (offset=0), so no clone needed
                # on its output. But the input slice data[e] has nonzero offset; pass
                # .clone() so intermediate ops inside _awq_zero also see a clean tensor.
                layer.moe_op.w13_list[e].set_zero_point(
                    self._awq_zero(_cl(layer.w13_qzeros.data[e])))
                layer.moe_op.w2_list[e].set_zero_point(
                    self._awq_zero(_cl(layer.w2_qzeros.data[e])))
            else:
                # _sym_zero returns a freshly-allocated tensor (torch.full -> pack):
                # storage_offset=0, shared across all experts — safe as-is.
                layer.moe_op.w13_list[e].set_zero_point(w13_zp)
                layer.moe_op.w2_list[e].set_zero_point(w2_zp)

        htorch.mark_step()
        layer._hpu_moe_ext_done = True

        # AUTO-MINT a just-eager-repacked UNCACHED layer (e.g. MTP layer 78) INTO the
        # LOAD dir, so the next run finds it on the fast path. Best-effort: a failure
        # here must NOT break serving (the layer is already live in HBM). Only when a
        # LOAD dir is set, the layer was the uncached fallback, and SAVE is not already
        # handling it below.
        from . import repack_cache as _rc_mint
        _load_dir_mint = _rc_mint.load_dir()
        if (_load_dir_mint and getattr(layer, "_hpu_repack_uncached", False)
                and not _rc_mint.save_dir()):
            try:
                _mp = _rc_mint.save_layer(layer, _load_dir_mint, has_zp)
                import logging
                logging.getLogger("vllm").info(
                    "[hpu-moe-repack] auto-minted uncached rank=%d layer=%s -> %s",
                    _rc_mint._rank(layer), _rc_mint._layer_idx(layer), _mp)
            except Exception as _ex:  # noqa: BLE001
                import logging
                logging.getLogger("vllm").warning(
                    "[hpu-moe-repack] auto-mint failed (non-fatal) rank=%d layer=%s: %s",
                    _rc_mint._rank(layer), _rc_mint._layer_idx(layer), _ex)

        # OFFLINE PRE-REPACK — SAVE path (env HPU_MOE_REPACK_SAVE=<dir>, default off).
        # After the existing repack + scale-transpose + moe_op build, persist THIS
        # rank's op-ready int32 weights/scales(/AWQ zeros) for THIS layer so a later
        # LAZY run can load them directly and skip _repack_weight entirely. Reuses the
        # real EP/TP-sharded, fused tensors (no offline layout re-derivation). Best run
        # ONCE under eager (PT_HPU_LAZY_MODE=0) to mint the cache. Loud on failure: a
        # silent miss here would mean the lazy load later finds nothing.
        from . import repack_cache
        _save = repack_cache.save_dir()
        if _save:
            try:
                _p = repack_cache.save_layer(layer, _save, has_zp)
                import logging
                logging.getLogger("vllm").info(
                    "[hpu-moe-repack] saved rank=%d layer=%s -> %s",
                    repack_cache._rank(layer), repack_cache._layer_idx(layer), _p)
            except Exception as _ex:  # noqa: BLE001
                import logging
                logging.getLogger("vllm").error(
                    "[hpu-moe-repack] SAVE FAILED rank=%d layer=%s: %s",
                    repack_cache._rank(layer), repack_cache._layer_idx(layer), _ex)
                raise

        # LOAD-TIME int4 -> fp8 ALL-EXPERT conversion (HPU_MOE_FP8_EXPERTS=1; default off =>
        # no-op => byte-identical to the stock int4 serve). DONE LAST so the SAVE/auto-mint
        # blocks above read intact int4 weights first; _build_fp8_experts then converts +
        # stashes resident fp8 and (default) FREES the int4 source. GPTQ-sym only (no-op for AWQ).
        _build_fp8_experts(layer, has_zp)

    def _process_weights_from_cache(self, layer: torch.nn.Module) -> None:
        """OFFLINE PRE-REPACK — LOAD path (env HPU_MOE_REPACK_LOAD=<dir>).

        Skip _repack_weight (and the uint8 originals it would free) entirely:
        load THIS rank+layer's cached op-ready int32 weights/scales(/AWQ zeros)
        straight from disk into the moe_op. Under create_weights' int32-placeholder
        allocation (see HPURepackLoadMixin.create_weights) there is no 52GB uint8
        fused param to pin/untrap — nothing to trap means KV fits.

        Preserves EVERY coherence fix: the moe_op is still built with the LOCAL
        window [0, num_experts-1] (the EP global->local remap fix), the routed-gather
        stashes (_hpu_routed_*, _hpu_experts_min, _hpu_local_experts) are set
        identically, and the GPTQ-symmetric shared zero is rebuilt by _sym_zero
        (NOT cached) so it matches the eager path bit-for-bit."""
        if getattr(layer, "_hpu_moe_ext_done", False):
            return
        from . import repack_cache
        has_zp = self.quant_config.has_zp
        import os as _os
        cached = repack_cache.load_layer_to_device(layer, repack_cache.load_dir())  # CPU tensors

        num_experts = layer.local_num_experts
        ep_shift = _resolve_ep_rank(layer) * num_experts
        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        # SAME LOCAL-window op as the eager path (GLM-5.1 / EP global->local fix).
        layer.moe_op = VllmMixtureOfExpertsOpWNA16(num_experts, 0, num_experts - 1)

        dev = layer.w13_qweight.data.device   # the (E,1,1) int32 placeholder lives on HPU
        layer.w13_scales = torch.nn.Parameter(cached["w13_scales"].to(dev), requires_grad=False)
        layer.w2_scales = torch.nn.Parameter(cached["w2_scales"].to(dev), requires_grad=False)

        if not has_zp:
            # GPTQ symmetric: shared constant zero, rebuilt deterministically (NOT cached).
            w13_zp = self._sym_zero(layer.w13_scales.shape[1], layer.w13_scales.shape[2], dev)
            w2_zp = self._sym_zero(layer.w2_scales.shape[1], layer.w2_scales.shape[2], dev)
        layer._hpu_routed_w13_zp = w13_zp if not has_zp else None
        layer._hpu_routed_w2_zp = w2_zp if not has_zp else None
        layer._hpu_experts_min = experts_min
        layer._hpu_local_experts = num_experts
        # PERSISTENT static-input x buffer (multi-bucket capture fix) — CACHE load path.
        # Same as the eager path: allocated at load (outside capture) as a held leaf, so
        # the fused op's x input survives every decode bucket. See _alloc_fused_x_buf.
        _alloc_fused_x_buf(layer)
        # PERSISTENT stacked zero-points (the LAST multi-bucket capture fix) — CACHE path.
        # Same as the eager path: materialized at load (outside capture) so the fused op's
        # zp index_select source is a held leaf surviving every decode bucket. The cache
        # path sets _hpu_routed_w13_zp/_hpu_routed_w2_zp + _hpu_local_experts just above, so
        # the prerequisites are in place. See _alloc_fused_zp.
        _alloc_fused_zp(layer)
        # NOTE: _alloc_fused_fp8_scale (HPU_MOE_FP8_DIRECT) is DEFERRED to AFTER the stacked
        # int4 weights are assigned to layer.w13_qweight/w2_qweight below. At THIS point
        # layer.w13_qweight is still the (E,1,1) int32 PLACEHOLDER (see line ~1230); calling
        # the DIRECT-scale precompute here dequants the placeholder -> convert_from_uint4
        # "scale shape [groups,out] vs output shape [1,8]" RuntimeError (the real int4 lives
        # in `cached[...]`, copied into w13_qweight only in the stacked block). The eager
        # _process_weights path can alloc inline because there w13_qweight already holds the
        # repacked [E,...] weights; the cache path must wait. See the deferred call after the
        # stacked-layout assignment.

        import habana_frameworks.torch as _htorch
        _lazy = _htorch.is_lazy()
        # GLM's MoE is ~45GB/card, so the eager-style 2x (stacked [E,...] + per-expert clones)
        # is ~90GB and OOMs under lazy. Keep exactly ONE on-device copy and pick which forward
        # path it serves (default 'stacked' = single-stream routed-K):
        #   stacked   -> [E,...] on-device for _routed_moe.index_select; op weights are VIEWS of
        #                it (the monolithic op is unused at bs=1, so HPUGraph never validates them).
        #   perexpert -> op's per-expert fresh tensors are the only copy (monolithic/concurrency);
        #                layer.w13_qweight stays the (E,1,1) placeholder (routed path unused at bs>1).
        _layout = _os.environ.get("HPU_MOE_REPACK_LAYOUT", "stacked")

        if _layout == "stacked":
            layer.w13_qweight = torch.nn.Parameter(cached["w13_qweight"].to(dev), requires_grad=False)
            layer.w2_qweight = torch.nn.Parameter(cached["w2_qweight"].to(dev), requires_grad=False)
            htorch.mark_step()
            for e in range(num_experts):
                layer.moe_op.w13_list[e].set_weight_packed(layer.w13_qweight.data[e])  # VIEW (1x)
                layer.moe_op.w2_list[e].set_weight_packed(layer.w2_qweight.data[e])
                layer.moe_op.w13_list[e].set_weight_scale(layer.w13_scales.data[e])
                layer.moe_op.w2_list[e].set_weight_scale(layer.w2_scales.data[e])
                if has_zp:
                    layer.moe_op.w13_list[e].set_zero_point(cached["w13_qzeros"][e].to(dev))
                    layer.moe_op.w2_list[e].set_zero_point(cached["w2_qzeros"][e].to(dev))
                else:
                    layer.moe_op.w13_list[e].set_zero_point(w13_zp)
                    layer.moe_op.w2_list[e].set_zero_point(w2_zp)
        else:  # perexpert
            def _cl(t):
                return t.clone() if _lazy else t
            for e in range(num_experts):
                layer.moe_op.w13_list[e].set_weight_packed(cached["w13_qweight"][e].contiguous().to(dev))
                layer.moe_op.w2_list[e].set_weight_packed(cached["w2_qweight"][e].contiguous().to(dev))
                layer.moe_op.w13_list[e].set_weight_scale(_cl(layer.w13_scales.data[e]))
                layer.moe_op.w2_list[e].set_weight_scale(_cl(layer.w2_scales.data[e]))
                if has_zp:
                    layer.moe_op.w13_list[e].set_zero_point(cached["w13_qzeros"][e].contiguous().to(dev))
                    layer.moe_op.w2_list[e].set_zero_point(cached["w2_qzeros"][e].contiguous().to(dev))
                else:
                    layer.moe_op.w13_list[e].set_zero_point(w13_zp)
                    layer.moe_op.w2_list[e].set_zero_point(w2_zp)
                if _lazy:
                    htorch.mark_step()

        htorch.mark_step()
        layer._hpu_moe_ext_done = True
        import logging
        logging.getLogger("vllm").info(
            "[hpu-moe-repack] LOADED rank=%d layer=%s from cache (no repack)",
            repack_cache._rank(layer), repack_cache._layer_idx(layer))

        # PERSISTENT per-expert DIRECT int4->fp8 recovery scales (HPU_MOE_FP8_DIRECT=1) —
        # CACHE path. DEFERRED to HERE (was prematurely called at line ~1255 BEFORE the stacked
        # int4 weights were assigned -> dequanted the (E,1,1) placeholder -> convert_from_uint4
        # shape RuntimeError). Now layer.w13_qweight holds the real cache-loaded [E,...] int4
        # weights (stacked block above), exactly the precondition _alloc_fused_fp8_scale +
        # _build_fp8_experts need. Requires the STACKED layout (perexpert keeps the placeholder
        # on w13_qweight); LOUD rather than dequant a placeholder again. No-op unless
        # HPU_MOE_FP8_DIRECT=1 (+ HPU_MOE_FP8_FUSED=1); harmless on the int4-only serve.
        if _FP8_DIRECT and _FP8_FUSED and _layout != "stacked":
            raise RuntimeError(
                "hpu-moe-ext: HPU_MOE_FP8_DIRECT=1 requires HPU_MOE_REPACK_LAYOUT=stacked "
                "(the DIRECT int4->fp8 recovery scale dequants the real [E,...] int4 weights "
                f"on layer.w13_qweight); got layout={_layout!r}. Unset HPU_MOE_REPACK_LAYOUT "
                "or set it to 'stacked'.")
        _alloc_fused_fp8_scale(layer)

        # LOAD-TIME int4 -> fp8 ALL-EXPERT conversion (HPU_MOE_FP8_EXPERTS=1; default off).
        # Requires the STACKED layout (layer.w13_qweight holds the real [E,...] int4 weights);
        # _build_fp8_experts asserts this and is LOUD on the perexpert layout (where the int4
        # lives only in moe_op). The glm52 serve uses the default stacked layout. No-op for AWQ.
        if _FP8_EXPERTS and _layout != "stacked":
            raise RuntimeError(
                "hpu-moe-ext: HPU_MOE_FP8_EXPERTS=1 requires HPU_MOE_REPACK_LAYOUT=stacked "
                "(the int4 [E,...] weights must be on layer.w13_qweight for the int4->fp8 "
                f"build); got layout={_layout!r}. Unset HPU_MOE_REPACK_LAYOUT or set it to "
                "'stacked'.")
        _build_fp8_experts(layer, has_zp)

    def _sym_zero(self, num_groups: int, out: int, device) -> torch.Tensor:
        # GPTQ stores symmetric qzeros with the autogptq "-1" convention; the
        # upstream loader's convert_gptq_int4_qzeros adds +1 (moe_wna16_upstream.py:430)
        # to recover the true zero. The op dequant is (code - zero_point) * scale, so
        # zero_point must equal the uint4 bias = 8 — exactly what the working
        # compressed-tensors path uses (hpu_compressed_tensors.py:887 quant_type.bias).
        # Verified on the real Qwen3-30B-A3B-GPTQ checkpoint: stored qzeros code == 7
        # uniformly, so true zero = 7 + 1 = 8 (NOT 9).
        qt = self._quant_type()
        bits = self.quant_config.weight_bits
        zero_val = 1 << (bits - 1)                # 4-bit -> 8 (= uint4 bias; matches CT path)
        zeros = torch.full((num_groups, out), zero_val, dtype=torch.int32, device=device)
        return pack_quantized_values_into_int32(zeros, qt, packed_dim=1)  # [groups, out//8]

    def _awq_zero(self, qzeros_u8: torch.Tensor) -> torch.Tensor:
        """AWQ qzeros uint8 [out//2, num_groups] -> int32 [num_groups, out//8]
        to match scale [num_groups, out]. (Validated separately for GLM-AWQ.)"""
        qt = self._quant_type()
        lo = (qzeros_u8 & 0x0F).to(torch.int32)
        hi = (qzeros_u8 >> 4).to(torch.int32)
        vals = torch.stack((lo, hi), dim=1).reshape(-1, qzeros_u8.shape[1])  # [out, groups]
        vals = vals.transpose(0, 1).contiguous()                            # [groups, out]
        return pack_quantized_values_into_int32(vals, qt, packed_dim=1)

    def _routed_moe(self, layer, x, topk_ids, topk_weights, perm, act, K):
        """2d static-K routed-gather. Dequant only the (<=K) locally-routed experts.
        Bit-exact vs the all-expert op: same convert_from_uint4 / scales / shared zero;
        experts not routed contribute 0 in both paths. Caller guarantees #distinct local
        experts <= K (tokens*top_k <= K), so NO host sync and NO dynamic shape. Returns
        the output tensor, or None if ineligible (AWQ per-expert zp)."""
        w13z = layer._hpu_routed_w13_zp
        w2z = layer._hpu_routed_w2_zp
        if w13z is None:
            return None
        E = layer._hpu_local_experts
        emin = layer._hpu_experts_min
        dev = x.device
        local = topk_ids.to(torch.int64) - emin            # global expert id -> local [0,E)
        flat = local.reshape(-1)
        valid = (flat >= 0) & (flat < E)                    # routed to a LOCAL expert?
        flat_safe = torch.where(valid, flat, torch.zeros_like(flat))   # for gather; where()-masked below
        # Mark routed-LOCAL experts. Non-local (other-rank) entries MUST scatter to a DUMMY
        # bin E, not index 0: scatter_ is last-write-wins, so collapsing the ~7/10 non-local
        # entries onto index 0 clobbers a genuinely-routed local expert 0 -> dropped expert ->
        # wrong output. (This was the maxdiff=1 verify failure.)
        scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
        # FUNCTIONAL scatter (NOT in-place scatter_): under lazy HPU-graph replay an in-place
        # scatter_ on a captured "fresh" zeros buffer can fail to re-zero each step -> stale
        # routing accumulates across decode steps -> wrong experts -> degeneration. The
        # out-of-place .scatter materializes a new tensor every step (graph-replay-safe).
        present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
            0, scatter_idx, valid.to(torch.float32))[:E]
        active = torch.topk(present, K).indices.to(torch.int64)    # [K] local ids (routed first)
        # MATERIALIZE the routing (present/scatter/topk -> `active`) BEFORE the data-dependent
        # gather. HPU-graph capture mishandles index_select with UN-materialized dynamic indices
        # (the gather reads stale/garbage `active` -> routed-verify md=100-200 on real weights).
        # A 2026-06-09 edit removed this boundary on the theory that it FREEZES `active` to the
        # capture step at replay -- that theory is CONTRADICTED by the validated-coherent
        # Qwen-397B fused serve, which runs the IDENTICAL routing -> mark_step -> index_select
        # shape (fused_expert_routed.py "MANDATORY mark_step"; `active` crosses that boundary)
        # and stayed live/coherent over long generations. Keep BOTH this boundary and the
        # post-dequant one below (mirrors the fused wrapper + the reference-op contract).
        htorch.mark_step()
        # gather K experts' packed weights + scales, dequant ONLY those K
        w13p = layer.w13_qweight.data.index_select(0, active)      # [K, in, out//8]
        w2p = layer.w2_qweight.data.index_select(0, active)
        s13 = layer.w13_scales.data.index_select(0, active)        # [K, groups, out]
        s2 = layer.w2_scales.data.index_select(0, active)
        dt = s13.dtype
        # ROOT-CAUSE FIX (HPU_MOE_ROUTED_VERIFY md=100..200): under LAZY HPUGraph the
        # SynapseAI backend mishandles a tensor whose storage is SHARED with a larger
        # parent (nonzero storage_offset). w13p/s13 are fresh index_select buffers, but
        # the per-slice views w13p[k]/s13[k] for k>0 carry a nonzero storage_offset
        # (= k * stride[0] into that buffer). Feeding them straight into
        # convert_from_uint4 makes the kernel read from the WRONG offset (effectively
        # expert active[0]'s bytes) -> every token routed to a k>0 expert dequants the
        # wrong weights -> huge maxdiff exactly when an expert is computed (md=0 when
        # nothing is routed locally). This is the SAME invariant the monolithic op
        # honors by .clone()-ing every per-expert slice at load (see _repack_weight /
        # _process_weights_from_cache: ".contiguous() on a contiguous slice is a no-op;
        # only .clone() gives a fresh storage_offset=0 tensor"). Mirror it here: clone
        # each per-expert slice under lazy so convert_from_uint4 sees offset=0, exactly
        # like get_dequant_weight does. Eager (no graph capture) tolerates the offset,
        # so skip the K extra copies there.
        _need_clone = htorch.is_lazy()
        def _co(t):
            return t.clone() if _need_clone else t
        w13_deq = [torch.ops.hpu.convert_from_uint4(_co(w13p[k]), _co(s13[k]), w13z, dt, None) for k in range(K)]
        w2_deq = [torch.ops.hpu.convert_from_uint4(_co(w2p[k]), _co(s2[k]), w2z, dt, None) for k in range(K)]
        # ROOT-CAUSE FIX (HPU_MOE_ROUTED_VERIFY md=100..200 under LAZY): mark_step() here is
        # MANDATORY in BOTH eager AND lazy — it is the exact boundary the coherent reference op
        # VllmMixtureOfExpertsOpWNA16.forward inserts (ops.py: builds the per-expert
        # convert_from_uint4 dequant list, then `htorch.core.mark_step()`, then calls
        # torch.ops.hpu.mixture_of_experts). The mixture_of_experts intrinsic requires its
        # w12/w3 weight lists to be MATERIALIZED at a graph boundary; feeding it
        # freshly-produced convert_from_uint4 outputs from WITHIN the same un-split lazy graph
        # yields wrong results (the dequant tensors are not realized when the MoE kernel reads
        # them) -> md=100..200 -> decode degenerates ("Paris, capital, capital...").
        # The previous `if not htorch.is_lazy()` guard skipped this in lazy on the (false)
        # assumption it would fragment the graph into 94 pieces/token. The reference op proves
        # otherwise: it ALWAYS mark_steps here, including under lazy, and stays coherent — so
        # _routed_moe must mirror it. (This is the true fix; the earlier per-slice .clone() of
        # w13p[k]/s13[k] for a phantom storage-offset bug did NOT and could NOT help, since the
        # per-expert dequant args were already verified bit-identical to the monolithic op.)
        htorch.mark_step()
        # remap routed global ids -> compact [0,K-1]; non-routed/non-local -> K (outside the
        # op's [0,K-1] window => skipped, exactly as out-of-window ids are in the stock op).
        # FUNCTIONAL scatter (NOT in-place) — same lazy HPU-graph-replay safety as `present` above.
        remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(   # default compact id 0 (in-window)
            0, active, torch.arange(K, dtype=torch.int64, device=dev))
        # Skipped (non-local / non-routed) slots -> compact id 0 BUT with weight ZEROED, so they
        # contribute exactly 0. (Relying on the op to skip an out-of-[0,K-1] sentinel id=K was the
        # bug: all-non-local tokens did NOT zero out -> maxdiff=1. Zeroing the weight is robust to
        # the op's out-of-window semantics and bit-exact: the all-128 ref skips them == 0.)
        valid2d = valid.view_as(topk_ids)
        remapped = torch.where(valid, remap[flat_safe], torch.zeros_like(flat)).view_as(topk_ids)
        safe_w = torch.where(valid2d, topk_weights, torch.zeros_like(topk_weights))
        routed_out = torch.ops.hpu.mixture_of_experts(
            hidden_states=x, expert_routing_table=remapped, router_weights=safe_w,
            w12=w13_deq, w3=w2_deq, permuted_weights=perm, activation=act,
            experts_min=0, experts_max=K - 1)
        if _ROUTED_DEBUG and _routed_dbg_count[0] < 1:
            _routed_dbg_count[0] += 1
            self._routed_debug_dump(layer, x, topk_ids, topk_weights, perm, act,
                                    active, remap, remapped, safe_w, valid, routed_out)
        if _ROUTED_STEP_PROBE > 0:
            _routed_step_probe(layer, x, active, routed_out)
        return routed_out

    def _fused_moe_routed(self, layer, x, topk_ids, topk_weights, act, K):
        """FUSED-OP routed path (HPU_MOE_FUSED_OP=1). Same eligibility + routing as
        _routed_moe, but the dequant+GEMM tail is the GC-captured, SRAM-resident
        torch.ops.hpu_moe_ext.fused_expert op (driven per-expert by the K-expert
        wrapper hpu_moe_ext.fused_expert_routed.fused_moe_routed). Returns the output
        tensor, or None if ineligible (AWQ per-expert zp) — caller then falls back.

        === STORED-LAYOUT -> OP-CONTRACT MAPPING (the whole point of this method) ===
        fused_moe_routed takes the SAME stacked [E,...] tensors the stock routed path
        already stashes on `layer`, in exactly the op's PyTorch-row-major layout —
        because both the fused op's internal dequantize_4_bit_bf16 and the stock path's
        torch.ops.hpu.convert_from_uint4 are the SAME intrinsic (proven byte-identical
        in tpc_kernel/opbackend/test_fused_expert_op.py). The packed weights + scales
        therefore pass straight through with NO transpose/reshape:

          plugin tensor (stacked; the wrapper index_selects K, op uses each slice)
          ----------------------------------------------------  ->  op per-expert contract
          layer.w13_qweight  [E, H,    2I/8] int32 (_repack_weight)  w13_packed [H,    2I/8]
          layer.w13_scales   [E, H/GS, 2I  ] bf16  (.transpose(1,2)) w13_scale  [H/GS, 2I  ]
          layer.w2_qweight   [E, I,    H/8 ] int32 (_repack_weight)  w2_packed  [I,    H/8 ]
          layer.w2_scales    [E, I/GS, H   ] bf16  (.transpose(1,2)) w2_scale   [I/GS, H   ]

        WHY the weights/scales match by construction (see process_weights_after_loading):
          * _repack_weight returns [E, in, out//8]: w13 loads uint8 [E, out=2I, in=H//2]
            -> [E, H, 2I/8]; w2 loads [E, out=H, in=I//2] -> [E, I, H/8]. Exactly the
            packed shapes the op derives (w13 out=2I, w2 out=H).
          * scales load [E, out, num_groups], .transpose(1,2) -> [E, groups, out] ==
            op's [H/GS, 2I] / [I/GS, H].

        === ZERO-POINT: the ONE shape adaptation needed ===
        fused_moe_routed expects a STACKED per-expert zp w13_zp[E, H/GS, 2I/8] /
        w2_zp[E, I/GS, H/8] (it does w13_zp.index_select(0, active) alongside the
        weights, so the contract works for AWQ per-expert zeros too — see its docstring).
        But the plugin's GPTQ-symmetric zero is a SINGLE SHARED 2D tensor
        [H/GS, 2I/8] / [I/GS, H/8] (_sym_zero -> packed midpoint 8 == 0x88888888),
        stashed as layer._hpu_routed_w13_zp / _hpu_routed_w2_zp (None for AWQ). Passing
        that 2D tensor unchanged would make the wrapper's index_select(0, active) index
        the GROUPS dim, not experts -> wrong slice. So we BROADCAST the shared zero to
        [E, groups, out//8] (every expert identical for GPTQ-sym) and cache it on the
        layer (built once; the zp is tiny — ~3 MB for GLM-5.1 — and identical across all
        decode steps, so no per-token allocation). .contiguous() so index_select reads a
        real per-expert stride, not a stride-0 broadcast dim. AWQ (w13z is None) is
        rejected up front and falls back, identical to _routed_moe's gate.

        E (num_local_experts) and I (moe_intermediate) are derived INSIDE
        fused_moe_routed from the stacked tensors, so the call passes only experts_min +
        K — matching the documented entrypoint signature. The wrapper owns all
        gather/route/remap/clone HPU-graph-safety (it mirrors _routed_moe verbatim) and
        calls the per-expert op K times (router_weight folded into the op's final mult)."""
        w13z = layer._hpu_routed_w13_zp
        w2z = layer._hpu_routed_w2_zp
        if w13z is None:
            # AWQ per-expert zp: the fused op is GPTQ-sym only. LEGITIMATE ineligibility
            # (the op genuinely cannot run on AWQ weights) -> intended=False, never strict-
            # raises; the caller drops to the stock _routed_moe (correct, not a worse path).
            engage_fallback(
                "moe_wna16._fused_moe_routed:awq-zp",
                fast="HPU_MOE_FUSED_OP routed fused op",
                reason="AWQ per-expert zero-point (the fused op is GPTQ-symmetric only)",
                slow="stock _routed_moe (convert_from_uint4 + mixture_of_experts)",
                intended=False, strict_env="HPU_MOE_FUSED_OP")
            return None  # AWQ per-expert zp -> ineligible (same as _routed_moe)
        if act != "silu":
            engage_fallback(
                "moe_wna16._fused_moe_routed:non-silu",
                fast="HPU_MOE_FUSED_OP routed fused op",
                reason="activation=%r (the fused-expert graph hard-codes silu(gate)*up)" % act,
                slow="stock _routed_moe",
                intended=False, strict_env="HPU_MOE_FUSED_OP")
            return None  # the fused-expert graph hard-codes silu(gate)*up
        _load_fused_op()  # LOUD if the requested .so is missing / fails to register
        from .fused_expert_routed import fused_moe_routed
        emin = layer._hpu_experts_min
        E = layer._hpu_local_experts
        # PERSISTENT stacked GPTQ-sym zero-points: the held leaves _alloc_fused_zp built at
        # WEIGHT-LOAD time (outside capture). Reading the pre-built leaf is the multi-bucket
        # capture fix: the fused op's zp index_select source is then a recipe INPUT that
        # survives every decode bucket -- NOT an in-forward `.contiguous()` intermediate the
        # free_inplace pass reclaims (the 2nd-bucket FATAL: zp `hpu__input` validated empty).
        # FALLBACK (lazy build) only if the load-time alloc is somehow absent (a regression,
        # or _alloc_fused_zp was gated off): build it here so correctness holds, but warn --
        # this in-forward `.contiguous()` is NOT multi-bucket-safe (it re-crashes on the 2nd
        # decode bucket exactly as before the fix). In a correct serve this branch never runs.
        w13z_stacked = getattr(layer, "_hpu_fused_w13_zp", None)
        w2z_stacked = getattr(layer, "_hpu_fused_w2_zp", None)
        if (w13z_stacked is None or w13z_stacked.shape[0] != E
                or w2z_stacked is None):
            # The persistent stacked-zp leaf (_alloc_fused_zp) is absent at decode -> we are
            # on the in-forward .contiguous() path, which is the KNOWN multi-bucket FATAL
            # (2nd decode bucket validates an empty zp). This is a REGRESSION the user did
            # NOT intend (they enabled the fused op AND its load-time zp materialization) ->
            # intended=True so HPU_ENGINE_STRICT hard-fails instead of letting the serve
            # re-crash cryptically at the 2nd bucket. (Off-strict: build it + continue, as
            # before -- the one-shot dedup keeps it to one line.)
            engage_fallback(
                "moe_wna16._fused_moe_routed:zp-in-forward",
                fast="persistent stacked-zp leaf (_alloc_fused_zp at weight-load)",
                reason="_hpu_fused_w13_zp absent/mis-shaped at decode (load-time alloc did "
                       "not run)",
                slow="in-forward .contiguous() zp build (NOT multi-bucket-capture-safe; "
                     "re-crashes at the 2nd decode bucket)",
                intended=True, strict_env="HPU_MOE_FUSED_OP")
            w13z_stacked = w13z.unsqueeze(0).expand(E, *w13z.shape).contiguous()
            w2z_stacked = w2z.unsqueeze(0).expand(E, *w2z.shape).contiguous()
            layer._hpu_fused_w13_zp = w13z_stacked
            layer._hpu_fused_w2_zp = w2z_stacked
        return fused_moe_routed(
            x,
            layer.w13_qweight.data, layer.w13_scales.data, w13z_stacked,
            layer.w2_qweight.data, layer.w2_scales.data, w2z_stacked,
            topk_ids, topk_weights, emin, K,
            # PERSISTENT static-input x buffer (multi-bucket capture fix): a held leaf
            # allocated at load time (outside capture), so the fused op roots its x input
            # at storage that survives every decode bucket's replay. None -> the wrapper
            # falls back to its in-forward clone (HPU_MOE_FUSED_X_BUFFER=0, or non-lazy).
            x_buf=getattr(layer, "_hpu_fused_x_buf", None))

    def _fp8_fused_moe_routed(self, layer, x, topk_ids, topk_weights, act):
        """FUSED fp8 OP path (HPU_MOE_FP8_EXPERTS=1 + HPU_MOE_FP8_FUSED=1). The fp8-compute
        analog of _fused_moe_routed: same eligibility + the SAME stored stacked int4 weights/
        scales/zp, but the dequant+GEMM tail is the GC-captured, SRAM-resident fp8 op
        torch.ops.hpu_moe_ext_fp8.fused_expert_fp8 (int4->bf16->fp8 dequant + native fp8 MME
        + silu, ONE recipe per expert), driven by fp8_routed_moe.fp8_fused_moe. Returns the
        output tensor, or None if ineligible (AWQ per-expert zp / non-silu / .so missing) --
        caller then falls back to the stock int4 op (int4 is LIVE in this mode).

        Routes through fp8_fused_moe with the routed-K budget when tokens*top_k <= HPU_MOE_
        ROUTED_K (the single-stream gather win); otherwise K=None -> ALL E local experts (the
        non-routed / MTP bs>=2 regime that fp8_allexpert_moe served). Either way EVERY expert
        is computed by the fused fp8 op -- never the per-expert fp8 Python loop.

        STORED-LAYOUT -> OP-CONTRACT MAPPING is IDENTICAL to _fused_moe_routed (the fp8 op
        takes the SAME int4-packed schema as the int4 op): the packed weights + transposed
        scales pass straight through (both the fp8 op's in-graph dequantize_4_bit and the int4
        path's convert_from_uint4 are the same intrinsic), and the GPTQ-sym shared zp is
        broadcast to the STACKED [E,...] held leaf the op index_selects per expert."""
        w13z = getattr(layer, "_hpu_routed_w13_zp", None)
        w2z = getattr(layer, "_hpu_routed_w2_zp", None)
        if w13z is None or w2z is None:
            # AWQ per-expert zp: LEGITIMATE ineligibility (the fused fp8 op is GPTQ-sym
            # only) -> intended=False; caller drops to the stock int4 op (int4 is LIVE in
            # this mode, so that is correct -- never silently to a worse fp8 path).
            engage_fallback(
                "moe_wna16._fp8_fused_moe_routed:awq-zp",
                fast="HPU_MOE_FP8_FUSED fused fp8 op",
                reason="AWQ per-expert zero-point (the fused fp8 op is GPTQ-symmetric only)",
                slow="stock int4 op (int4 weights are live in fp8-fused mode)",
                intended=False, strict_env="HPU_MOE_FP8_FUSED")
            return None  # AWQ per-expert zp -> ineligible (same as _fused_moe_routed)
        if act != "silu":
            engage_fallback(
                "moe_wna16._fp8_fused_moe_routed:non-silu",
                fast="HPU_MOE_FP8_FUSED fused fp8 op",
                reason="activation=%r (the fused fp8 graph hard-codes silu(gate)*up)" % act,
                slow="stock int4 op (int4 weights are live in fp8-fused mode)",
                intended=False, strict_env="HPU_MOE_FP8_FUSED")
            return None  # the fused fp8 graph hard-codes silu(gate)*up
        from .fp8_routed_moe import fp8_fused_moe
        emin = layer._hpu_experts_min
        E = layer._hpu_local_experts
        # PERSISTENT stacked GPTQ-sym zero-points (held leaves _alloc_fused_zp built at
        # WEIGHT-LOAD time, outside capture -- its gate now includes _FP8_FUSED). The fused
        # fp8 op index_selects the stacked zp alongside the weights, so it must be a persistent
        # recipe input surviving every decode bucket (same held-leaf invariant the int4 routed
        # path relies on). FALLBACK (in-forward build) only if the load-time alloc is absent (a
        # regression / non-lazy / x-buffer feature off) -- correct, but the in-graph
        # .contiguous() is NOT multi-bucket-capture-safe (warn once, mirroring _fused_moe_routed).
        w13z_stacked = getattr(layer, "_hpu_fused_w13_zp", None)
        w2z_stacked = getattr(layer, "_hpu_fused_w2_zp", None)
        if (w13z_stacked is None or w13z_stacked.shape[0] != E
                or w2z_stacked is None):
            # Persistent stacked-zp leaf absent at decode -> the in-forward .contiguous()
            # multi-bucket-FATAL path. REGRESSION (the user enabled fp8-fused AND its
            # load-time zp alloc) -> intended=True so HPU_ENGINE_STRICT hard-fails.
            engage_fallback(
                "moe_wna16._fp8_fused_moe_routed:zp-in-forward",
                fast="persistent stacked-zp leaf (_alloc_fused_zp at weight-load)",
                reason="_hpu_fused_w13_zp absent/mis-shaped at decode (load-time alloc did "
                       "not run; its gate includes _FP8_FUSED)",
                slow="in-forward .contiguous() zp build (NOT multi-bucket-capture-safe)",
                intended=True, strict_env="HPU_MOE_FP8_FUSED")
            w13z_stacked = w13z.unsqueeze(0).expand(E, *w13z.shape).contiguous()
            w2z_stacked = w2z.unsqueeze(0).expand(E, *w2z.shape).contiguous()
            layer._hpu_fused_w13_zp = w13z_stacked
            layer._hpu_fused_w2_zp = w2z_stacked
        # Routed-K budget: gather the <=K locally-routed experts when provably safe
        # (tokens*top_k <= K); else K=None -> all E local experts. Mirrors the int4 gate.
        K = _ROUTED_K if (_ROUTED_K > 0 and (x.shape[0] * layer.top_k) <= _ROUTED_K) else None
        # DIRECT int4->fp8 (HPU_MOE_FP8_DIRECT): pass the per-expert FORWARD fp8 scales
        # precomputed at load by _alloc_fused_fp8_scale (held leaves [E,1] f32). When absent
        # (flag off / not built) -> None -> the op runs its proven two-hop in-graph-amax path.
        w13_fp8_s = getattr(layer, "_hpu_fused_w13_fp8_scale", None) if _FP8_DIRECT else None
        w2_fp8_s = getattr(layer, "_hpu_fused_w2_fp8_scale", None) if _FP8_DIRECT else None
        return fp8_fused_moe(
            x, topk_ids, topk_weights,
            layer.w13_qweight.data, layer.w13_scales.data,
            layer.w2_qweight.data, layer.w2_scales.data,
            w13z_stacked, w2z_stacked,
            experts_min=emin, activation=act, K=K,
            w13_fp8_scale=w13_fp8_s, w2_fp8_scale=w2_fp8_s)

    def _fused_monolithic(self, layer, x, topk_ids, topk_weights, act):
        """FUSED MONOLITHIC path (HPU_MOE_FUSED_MONOLITHIC=1) -- the THROUGHPUT regime.
        Replace the STOCK VllmMixtureOfExpertsOpWNA16 (all-E dequant -> HBM -> a separate
        mixture_of_experts gemm) with the single-recipe torch.ops.hpu_moe_ext.
        fused_moe_monolithic op (driven by fused_monolithic_moe.fused_moe_monolithic),
        which processes ALL E LOCAL experts with per-expert SRAM-resident dequant (chunked
        over E) and a DENSE [E,T,1] per-(token,expert) router matrix. Returns the output
        tensor, or None if ineligible (AWQ per-expert zp / non-silu / .so missing) -- caller
        then falls back to the stock op.

        SAME stored-layout -> op-contract mapping as _fused_moe_routed (the packed weights
        + transposed scales pass straight through; both the op's dequantize_4_bit_bf16 and
        the stock convert_from_uint4 are the same intrinsic). The op takes the FULL stacked
        [E,...] tensors DIRECTLY (no Python gather -- it slices each chunk on-device), so the
        held-leaf weights/scales/zp ARE the recipe inputs. GPTQ-symmetric only (the stacked
        zp _alloc_fused_zp built; None for AWQ -> ineligible, same gate as the routed path).

        topk_ids/topk_weights are the GLOBAL ids + weights (as apply_monolithic computed
        them, BEFORE the global->local remap); the wrapper does the remap + the dense
        [E,T,1] routing internally (matching the stock mixture_of_experts LOCAL-window
        semantics exactly)."""
        w13z = getattr(layer, "_hpu_routed_w13_zp", None)
        if w13z is None:
            # AWQ per-expert zp: LEGITIMATE ineligibility (the monolithic fused op is
            # GPTQ-sym only) -> intended=False; caller drops to the stock all-expert op.
            engage_fallback(
                "moe_wna16._fused_monolithic:awq-zp",
                fast="HPU_MOE_FUSED_MONOLITHIC fused monolithic op",
                reason="AWQ per-expert zero-point (the monolithic fused op is GPTQ-sym only)",
                slow="stock all-expert mixture_of_experts",
                intended=False, strict_env="HPU_MOE_FUSED_MONOLITHIC")
            return None  # AWQ per-expert zp -> ineligible (same as the routed path)
        if act != "silu":
            engage_fallback(
                "moe_wna16._fused_monolithic:non-silu",
                fast="HPU_MOE_FUSED_MONOLITHIC fused monolithic op",
                reason="activation=%r (the fused-monolithic graph hard-codes silu(gate)*up)" % act,
                slow="stock all-expert mixture_of_experts",
                intended=False, strict_env="HPU_MOE_FUSED_MONOLITHIC")
            return None  # the fused-monolithic graph hard-codes silu(gate)*up
        from .fused_monolithic_moe import fused_moe_monolithic
        emin = layer._hpu_experts_min
        E = layer._hpu_local_experts
        # PERSISTENT stacked GPTQ-sym zero-points: the held leaves _alloc_fused_zp built at
        # WEIGHT-LOAD time (outside capture). The op slices each chunk's [C,...] zp ON-DEVICE
        # from these stacked tensors, so they must be persistent recipe inputs surviving every
        # decode bucket (same held-leaf invariant the routed path relies on). FALLBACK (lazy
        # build) only if the load-time alloc is somehow absent (a regression / gated off);
        # in a correct serve this branch never runs.
        w13z_stacked = getattr(layer, "_hpu_fused_w13_zp", None)
        w2z_stacked = getattr(layer, "_hpu_fused_w2_zp", None)
        if (w13z_stacked is None or w13z_stacked.shape[0] != E
                or w2z_stacked is None):
            w2z = layer._hpu_routed_w2_zp
            w13z_stacked = w13z.unsqueeze(0).expand(E, *w13z.shape).contiguous()
            w2z_stacked = w2z.unsqueeze(0).expand(E, *w2z.shape).contiguous()
            layer._hpu_fused_w13_zp = w13z_stacked
            layer._hpu_fused_w2_zp = w2z_stacked
        # PERSISTENT static-input x buffer (multi-bucket capture fix): the held leaf
        # allocated at load time, so the op roots its x input at storage that survives
        # every decode bucket. None -> x passed directly (single-bucket / non-lazy).
        x_buf = getattr(layer, "_hpu_fused_x_buf", None)
        _Tmono = x.shape[0]
        # === PP PREFILL CHUNK (pp>1 ONLY; the disable_tensor_cache=False prefill fix) ===
        # The op emits ALL E local experts' dequant+gemm+silu nodes into ONE captured recipe
        # (emit_monolithic_experts add-chains ceil(E/CHUNK) sub-chains -> one AddNode -> one
        # graph; the `chunk` arg is only SRAM tiling, NOT recipe segmentation -- moe_fused_
        # monolithic_emit.h). So the recipe NODE COUNT scales with E, NOT T. At E_local=128
        # (TP=2/EP=2 -> 256/2; 4x the E=32 the op was validated at on TP=8) that all-128-expert
        # recipe is what the host-side Synapse compiler HANGS on (0% device util, never returns
        # -- NOT an OOM). T is irrelevant to the node count (a T=128 prefill hangs the same), so
        # TOKEN-chunking alone is USELESS here: every token sub-batch still passes the full
        # [E=128,...] weights -> the same all-128 recipe -> the same hang.
        #
        # FIX = EXPERT-AXIS chunking (primary) x token-axis chunking (defensive):
        #   * EXPERT (HPU_MOE_MONO_E_CHUNK=ec): slice the stacked weights/scales/zp to the
        #     experts [g, g+ec) and call the op once per group with experts_min = emin+g, so each
        #     call builds a BOUNDED ec-expert recipe (ec=32 == the proven TP-8 size). The op's
        #     global->local remap (fused_moe_monolithic: local = ids - experts_min, valid in
        #     [0,ec)) zeroes the out-of-group experts, so a group's partial = the sum over ITS
        #     experts and SUMMING the groups == the full MoE -- EXACT. The recipe is cached by
        #     (E=ec, chunk, T, H, I): every group has the SAME shape -> compiled ONCE, replayed
        #     per group. OFFSET-SAFE with plain VIEWS (no fragmenting clone): the op's FIRST node
        #     on each weight is emit_chunk_slice == synSliceParamsV2, which HONOURS the view's
        #     storage-offset (fused_expert_routed.py:709-711: "the slice node honours the offset
        #     where a raw reinterpret does not"); the reinterpret runs on the slice's offset-0
        #     output.
        #   * TOKEN (HPU_MOE_MONO_T_CHUNK=tc): the op's per-sub-chain transient gu[chunk,T,2I]/
        #     y[chunk,T,H] scales with T; split the token axis into <=tc rows so each call's
        #     largest transient stays small (~<=12MB at tc<=128) and reclaimable. tc==_FUSED_X_BUF_T
        #     routes every sub-batch through the SAME persistent x_buf (the held leaf).
        # A mark_step after each group flushes its recipe + reclaims its transient before the next.
        # Token-chunk OUTER (concat tokens), expert-group INNER (SUM partials). Gate on
        # _is_pp_gt1() -> TP serves BYTE-IDENTICAL; ec=0 AND tc=0 -> the single un-chunked call.
        _pp = _is_pp_gt1()
        # TOKEN-chunk ALSO engages under DP-attention (dp_size>1), not just PP. apply_monolithic
        # all-gathers x across the DP group (x -> [N*dp, H]) BEFORE this call, so _Tmono is the
        # DP-inflated token count and the op's per-sub-chain transient gu[chunk,T,2I] scales with it;
        # at c64 the un-capped transient OOMs ("worker death -> gloo cascade"). The old _pp-only gate
        # left HPU_MOE_MONO_T_CHUNK a silent NO-OP under DP2xTP4 (PP world_size==1), which is exactly
        # why the DP+W4A8 compose crashed. Expert-chunk stays PP-only: DP does NOT change the E-scaled
        # recipe node count (the compile-hang the E gate fixes), only the token/transient size.
        _dp_gt1 = getattr(getattr(layer, "moe_config", None), "dp_size", 1) > 1
        _ec = _MONO_E_CHUNK if (_pp and _MONO_E_CHUNK > 0 and E > _MONO_E_CHUNK) else 0
        _tc = _MONO_T_CHUNK if ((_pp or _dp_gt1) and _MONO_T_CHUNK > 0 and _Tmono > _MONO_T_CHUNK) else 0
        if _ec > 0 or _tc > 0:
            _e_step = _ec if _ec > 0 else E
            _t_step = _tc if _tc > 0 else _Tmono
            if not _logged_mono_chunk[0]:
                _logged_mono_chunk[0] = True
                import logging
                logging.getLogger("vllm").info(
                    "hpu-moe-ext: monolithic PP/DP CHUNK engaged (pp=%s dp=%s: E=%d in groups of %d, "
                    "T=%d in sub-batches of %d, x_buf rows=%s) -- bounds the captured recipe in the "
                    "expert axis (node count -> compile, the E-scaled PP hang fix) and/or the token "
                    "axis (transient size, the DP-inflated-T OOM fix).",
                    bool(_pp), bool(_dp_gt1), int(E), int(_e_step), int(_Tmono), int(_t_step),
                    None if x_buf is None else int(x_buf.shape[0]))
            _out_chunks = []
            for _s in range(0, _Tmono, _t_step):
                _se = min(_s + _t_step, _Tmono)
                _xc = x[_s:_se]
                _tic = topk_ids[_s:_se]
                _twc = topk_weights[_s:_se]
                _acc = None
                for _g in range(0, E, _e_step):
                    _ge = min(_g + _e_step, E)
                    _part = fused_moe_monolithic(
                        _xc,
                        layer.w13_qweight.data[_g:_ge], layer.w13_scales.data[_g:_ge],
                        w13z_stacked[_g:_ge],
                        layer.w2_qweight.data[_g:_ge], layer.w2_scales.data[_g:_ge],
                        w2z_stacked[_g:_ge],
                        _tic, _twc, emin + _g,          # group's global->local base = emin+g
                        x_buf=x_buf)
                    if _part is None:
                        return None  # .so absent -> caller falls back to stock (same as un-chunked)
                    _acc = _part if _acc is None else _acc + _part   # SUM expert-group partials
                    htorch.mark_step()  # flush this group's recipe; reclaim its transient
                _out_chunks.append(_acc)
            return _out_chunks[0] if len(_out_chunks) == 1 else torch.cat(_out_chunks, dim=0)
        # === REAP PERF PROBE (HPU_MOE_REAP_PROBE=1; default-OFF -> skipped -> byte-exact) ===
        # Slice each rank's stacked experts to the REAP-kept count so the monolithic op builds a
        # SMALLER (keep-expert) recipe -> measure the concurrency step-ms 192-vs-256 WITHOUT
        # regenerating the checkpoint. Contiguous [:keep] view (offset 0, dim-0 expert axis): the
        # op's on-device emit_chunk_slice honours it; ZERO per-forward copy; the view is rooted at
        # the persistent nn.Parameter storage so the held-leaf/multi-bucket-capture invariant holds
        # and the keep-shaped recipe is stable across decode buckets. QUALITY-GARBAGE output by
        # design (kept rows are NOT the REAP survivors; ~drop_per_window/E of the routing weight is
        # zeroed) -- PERF-ONLY. NON-PP path only (the champion); the PP-chunked branch above stays
        # byte-exact. One-shot LOUD banner so a live serve confirms the probe is engaged (a silent
        # no-op would measure 256 experts -> a FALSE 'REAP dead' verdict).
        if _REAP_PROBE:
            _keep = _reap_probe_keep(int(E))
            if _keep is not None:
                if not _reap_probe_logged[0]:
                    _reap_probe_logged[0] = True
                    import logging
                    logging.getLogger("vllm").warning(
                        "[hpu-moe-ext] HPU_MOE_REAP_PROBE ACTIVE -- monolithic op sliced to %d of %d "
                        "LOCAL experts/rank (REAP %d->%d within-EP-window). PERF-ONLY: output is "
                        "QUALITY-GARBAGE (kept rows are NOT the REAP survivors); read step-ms only. "
                        "Run glm52/quant/prune_experts.py for the correct quality-gated transform "
                        "if this wins.",
                        int(_keep), int(E),
                        int(_REAP_INFO["src"]) if _REAP_INFO else 0,
                        int(_REAP_INFO["dst"]) if _REAP_INFO else 0)
                return fused_moe_monolithic(
                    x,
                    layer.w13_qweight.data[:_keep], layer.w13_scales.data[:_keep],
                    w13z_stacked[:_keep],
                    layer.w2_qweight.data[:_keep], layer.w2_scales.data[:_keep],
                    w2z_stacked[:_keep],
                    topk_ids, topk_weights, emin,
                    x_buf=x_buf)
        return fused_moe_monolithic(
            x,
            layer.w13_qweight.data, layer.w13_scales.data, w13z_stacked,
            layer.w2_qweight.data, layer.w2_scales.data, w2z_stacked,
            topk_ids, topk_weights, emin,
            x_buf=x_buf)

    @torch.compiler.disable
    def _routed_debug_dump(self, layer, x, topk_ids, topk_weights, perm, act,
                           active, remap, remapped, safe_w, valid, routed_out):
        """HPU_ROUTED_DEBUG=1, first call only: print the routing internals + routed-vs-ref
        norms so ONE request shows exactly where _routed_moe diverges from the all-expert op.
        @torch.compiler.disable so the .cpu()/print don't break the graph. Best-effort."""
        import logging
        log = logging.getLogger("vllm")
        try:
            emin = getattr(layer, "_hpu_experts_min", 0)
            nloc = getattr(layer, "_hpu_local_experts", None)
            tid = topk_ids.to(torch.int64)
            tw_ = topk_weights.to(x.dtype)
            _rtid, _rtw = tid, tw_
            if nloc is not None:
                _iw = (tid >= emin) & (tid < emin + nloc)
                _rtid = torch.where(_iw, tid - emin, torch.zeros_like(tid))
                _rtw = torch.where(_iw, tw_, torch.zeros_like(tw_))
            ref = layer.moe_op(x, _rtid, _rtw, permuted_weights=perm, activation=act)
            md = float((routed_out.float() - ref.float()).abs().max().item())
            log.warning(
                "[routed-debug] x0=%d emin=%s nloc=%s\n"
                "  active      = %s\n"
                "  remap(active)=%s\n"
                "  topk_ids[0] = %s\n"
                "  valid[0]    = %s\n"
                "  remapped[0] = %s\n"
                "  safe_w[0]   = %s\n"
                "  routed_norm=%.5f ref_norm=%.5f MAXDIFF=%.5g",
                x.shape[0], emin, nloc,
                active.detach().cpu().tolist(),
                remap.detach().cpu().index_select(0, active.detach().cpu()).tolist(),
                tid.detach().cpu()[0].tolist(),
                valid.view_as(topk_ids).detach().cpu()[0].tolist(),
                remapped.detach().cpu()[0].tolist(),
                [round(v, 4) for v in safe_w.float().detach().cpu()[0].tolist()],
                float(routed_out.float().norm().item()),
                float(ref.float().norm().item()), md,
            )
        except Exception as ex:  # noqa: BLE001
            log.warning("[routed-debug] dump error: %s", ex)

    def apply_monolithic(self, layer, x: torch.Tensor,
                         router_logits: torch.Tensor,
                         **kwargs) -> Union[torch.Tensor, tuple]:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        # ENGAGE-MANIFEST (default-ON, one-shot per rank): the TRUE resolved MoE fast-path
        # state at this shape, resolved from THIS layer's load-time attributes + the flags
        # (the same predicates the dispatch below uses). Pure output; no behavior change.
        _emit_moe_engage_manifest(layer, int(x.shape[0]))
        topk_weights, topk_ids = _hpu_safe_select_experts(
            layer, x, router_logits, input_ids=kwargs.get("input_ids"))
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        act = getattr(layer.activation, "value", layer.activation)  # normalize Enum -> str
        perm = bool(getattr(layer, "_hpu_moe_unquant", False))  # unquant op stores [out,in]
        tid = topk_ids.to(torch.int64)
        tw_ = topk_weights.to(x.dtype)
        # ---- DP-ATTENTION + EP: all-gather (dispatch) tokens across the DP group -----------
        # The int4 WNA16 op has NO dispatch_fn, so under dp_size>1 the tokens must be gathered
        # HERE (HpuCommunicator.dispatch() is a no-op that defers to the method). The runner then
        # combine()=reduce_scatter(DP) + _maybe_reduce_final_output=all_reduce(TP) to reassemble.
        # Without this the op returns local [N,H] but reduce_scatter expects [N*dp,H] -> "!!!!".
        # Mirrors the stock HPUUnquantizedFusedMoEMethod.apply_monolithic; the global->local expert
        # remap below is correct on the gathered global ids.
        _dp = getattr(getattr(layer, "moe_config", None), "dp_size", 1) > 1
        if _dp:
            from vllm_gaudi.v1.worker.hpu_dp_utils import (get_hpu_dp_metadata,
                                                           dispatch_tensor)
            _md = get_hpu_dp_metadata()
            _sp = bool(getattr(layer, "is_sequence_parallel", False))
            x   = dispatch_tensor(x, None, _sp)
            tid = dispatch_tensor(tid, _md.topk_ids_across_dp     if _md is not None else None, _sp)
            tw_ = dispatch_tensor(tw_, _md.topk_weights_across_dp if _md is not None else None, _sp)
            if not _DP_GATHER_LOGGED[0]:
                _DP_GATHER_LOGGED[0] = True
                import logging
                logging.getLogger("vllm").warning(
                    "[hpu-moe-ext] MoeWNA16 DP-DISPATCH engaged: gathered x/tid/tw across "
                    "dp_group -> x=%s (combine=reduce_scatter(DP)+all_reduce(TP))", tuple(x.shape))
        # ------------------------------------------------------------------------------------
        # 2d: routed-gather only when it is provably safe (distinct local experts <= K,
        # i.e. tokens*top_k <= K) and eligible (GPTQ-sym). Else fall through to all-expert op.
        # When this layer's int4 experts were converted+freed for fp8 (HPU_MOE_FP8_EXPERTS=1 +
        # default HPU_MOE_FP8_FREE_INT4), the routed/fused path's int4 weights are gone. Skip the
        # routed gate entirely so it can never index_select a freed (1,1,1) placeholder (BUG #1
        # belt-and-suspenders -- _free_int4_experts already nulls the routed zp, but this guards a
        # future free-without-null regression). With fp8 resident the all-expert fp8 branch below
        # is the only correct path for ALL batch sizes.
        _fp8_layer = _FP8_EXPERTS and getattr(layer, "_fp8_w13", None) is not None
        # FUSED fp8 OP path (HPU_MOE_FP8_EXPERTS=1 + HPU_MOE_FP8_FUSED=1): the proper
        # single-recipe fp8 kernel. Replaces BOTH the resident-fp8 mixture_of_experts branch
        # below AND the per-expert fp8_allexpert_moe/fp8_routed_moe Python loop with the
        # GC-captured fused fp8 op (torch.ops.hpu_moe_ext_fp8.fused_expert_fp8), which dequants
        # int4->bf16->fp8 + runs both native fp8 MME GEMMs + silu IN ONE recipe per expert. In
        # this mode int4 is KEPT resident and the resident-fp8 tensors are NOT built (so
        # _fp8_layer is False here -- the op consumes the LIVE int4 weights). Mirrors the int4
        # _FUSED_OP dispatch EXACTLY (same routed-K gather when eligible, all-expert otherwise).
        # _fp8_fused_moe_routed returns None when ineligible (AWQ / non-silu / .so missing) ->
        # fall through to the stock int4 path below (int4 is live, so that is correct -- never
        # silently to a worse fp8 path). Gated on top of HPU_MOE_FP8_EXPERTS so default
        # (HPU_MOE_FP8_FUSED unset) is byte-identical.
        if _FP8_FUSED and _FP8_EXPERTS:
            fp8_fused = self._fp8_fused_moe_routed(layer, x, tid, tw_, act)
            if fp8_fused is not None:
                return _dp_out_reshape(fp8_fused, input_shape, _dp)
            # ineligible (AWQ / non-silu / .so missing) -> stock int4 path below (int4 live).
        # ROUTED/FUSED gather eligibility + the per-call STATIC gather width _rk.
        #  * bs=1 / DEFAULT (HPU_MOE_ROUTED_BSGT1 off): admit ONLY tokens*top_k <= _ROUTED_K
        #    and pass K=_ROUTED_K -> BYTE-IDENTICAL to the historical gate (the bs=1 win;
        #    the MTP-verify bs=4 batch (32 > 8) still falls to the all-expert path below).
        #  * HPU_MOE_ROUTED_BSGT1=1 (the MTP-VERIFY lever): ALSO admit T>1, with K sized from
        #    the static per-T bucket (_routed_bsgt1_K: bucket override or the conservative
        #    T*top_k, clamped [top_k..E]). The T==1 K is UNCHANGED from _ROUTED_K when it is
        #    >= top_k (it always is: _ROUTED_K=8==top_k in every serve env), so flipping the
        #    flag on does NOT perturb the bs=1 path -- it only opens the T>1 admission. Under
        #    lazy the captured T>1 verify routes through the CAPTURE-SAFE SEQUENTIAL single-
        #    recipe op (fused_expert_routed.py: slices the K experts ON-DEVICE -> no freed in-
        #    graph clone; per-expert [T,1] router weight broadcast per-token), NOT the in-graph-
        #    clone loop (which is the historical bs>1 FATAL / relNorm~1e11). Eager T>1 keeps the
        #    loop (uncaptured -> no freed-intermediate class).
        _T = x.shape[0]
        # PREFILL DIVERT (explicit HPU_MOE_ROUTED_MAX_T>0): the routed/fused single-recipe
        # gather balloons its static width K to E (= all local experts) for a large-T prefill
        # (_routed_bsgt1_K clamps T*top_k to E), and emitting E expert subgraphs over a 2-4k-token
        # prefill in ONE recipe HANGS the host-side Synapse compiler under PP's disable_tensor_
        # cache=False capture (0% device util, never returns). Cap the routed admission to
        # T<=_ROUTED_MAX_T (the decode/verify regime the gather was built for) so a larger-T
        # prefill SKIPS the gather and falls to monolithic/stock all-expert dispatch. This is an
        # explicit opt-in; default _ROUTED_MAX_T=0 stays unlimited and byte-identical.
        _routed_tcap = (_ROUTED_MAX_T > 0 and _T > _ROUTED_MAX_T)
        if _ROUTED_BSGT1:
            _routed_ok = (_ROUTED_K > 0) and not _routed_tcap
            _rk = _routed_bsgt1_K(_T, layer.top_k, layer._hpu_local_experts) if _routed_ok else 0
        else:
            _routed_ok = (_ROUTED_K > 0 and (_T * layer.top_k) <= _ROUTED_K) and not _routed_tcap
            _rk = _ROUTED_K
        if (not _fp8_layer) and _routed_ok \
                and getattr(layer, "_hpu_routed_w13_zp", None) is not None:
            # FUSED-OP path (HPU_MOE_FUSED_OP=1): replace the convert_from_uint4-list ->
            # mark_step -> mixture_of_experts tail with the GC-captured fused_expert op
            # (driven per-expert by fused_moe_routed). SAME eligibility gate as the stock
            # routed path above + GPTQ-sym int4 (None zp -> AWQ -> falls through). When the
            # op is ineligible for this layer (AWQ/non-silu) _fused_moe_routed returns None
            # and we drop to the stock _routed_moe below — never silently to a worse path.
            if _FUSED_OP:
                fused = self._fused_moe_routed(layer, x, tid, tw_, act, _rk)
                if _MOE_POISON and fused is not None: fused = fused * 1.15  # FORENSIC PROBE (HPU_MOE_POISON=1): garble routed-fused MoE output; coherent==our-op-didn't-run
                if fused is not None:
                    if _FUSED_VERIFY:
                        # ref = the STOCK routed path (convert_from_uint4 + the proven
                        # mixture_of_experts intrinsic) — bit-exact-equivalent to the
                        # all-expert op by _routed_moe's own HPU_MOE_ROUTED_VERIFY proof,
                        # so it is the right oracle for the fused op. relNorm < 1e-2 is the
                        # bf16 norm-relative tol the fused-op recipe was validated against
                        # (test_fused_expert_op.py); LOUD + stay-coherent (return stock) on
                        # mismatch, mirroring the routed-verify harness.
                        ref = self._routed_moe(layer, x, tid, tw_, perm, act, _rk)
                        import logging
                        if ref is None:
                            logging.getLogger("vllm").warning(
                                "[fused-verify] stock ref unavailable (ineligible); "
                                "skipping verify, using fused result")
                            return _dp_out_reshape(fused, input_shape, _dp)
                        diff = (fused.float() - ref.float()).norm().item()
                        rn = diff / (ref.float().norm().item() + 1e-9)
                        if rn >= 1e-2:
                            ma = (fused.float() - ref.float()).abs().max().item()
                            logging.getLogger("vllm").warning(
                                "[fused-verify] MISMATCH relNorm=%.4g maxAbs=%.4g x0=%d "
                                "-- falling back to stock mixture_of_experts result"
                                % (rn, ma, x.shape[0]))
                            return _dp_out_reshape(ref, input_shape, _dp)   # stay coherent; no crash
                        logging.getLogger("vllm").info(
                            "[fused-verify] relNorm=%.6g x0=%d" % (rn, x.shape[0]))
                    return _dp_out_reshape(fused, input_shape, _dp)
                # fall through to stock routed path (AWQ / non-silu ineligible)
            routed = self._routed_moe(layer, x, tid, tw_, perm, act, _rk)
            if routed is not None:
                if _ROUTED_VERIFY:
                    # ref via the all-expert op needs LOCAL ids: layer.moe_op was built with a
                    # LOCAL window [0,nloc-1] (the EP global->local fix), so passing GLOBAL ids
                    # makes ref ~0 routed and FALSELY rejects the (correct) _routed_moe. Mirror
                    # the apply_monolithic remap here so ref is the true all-expert result.
                    _emin = getattr(layer, "_hpu_experts_min", 0)
                    _nloc = getattr(layer, "_hpu_local_experts", None)
                    _rtid, _rtw = tid, tw_
                    if _nloc is not None:
                        _iw = (tid >= _emin) & (tid < _emin + _nloc)
                        _rtid = torch.where(_iw, tid - _emin, torch.full_like(tid, _nloc))
                        _rtw = torch.where(_iw, tw_, torch.zeros_like(tw_))
                    ref = layer.moe_op(x, _rtid, _rtw, permuted_weights=perm, activation=act)
                    md = (routed.float() - ref.float()).abs().max().item()
                    import logging
                    if md >= 1e-2:
                        logging.getLogger("vllm").warning(
                            "[routed-verify] MISMATCH md=%.4g x0=%d" % (md, x.shape[0]))
                        return _dp_out_reshape(ref, input_shape, _dp)   # stay coherent; no crash
                    logging.getLogger("vllm").info("[routed-verify] maxdiff=%.6g" % md)
                return _dp_out_reshape(routed, input_shape, _dp)
        # fp8 ALL-EXPERT path (HPU_MOE_FP8_EXPERTS=1) -- the single-stream / MTP-verify bandwidth
        # win. Reached whenever the routed-K gather was NOT taken: tokens*top_k > HPU_MOE_ROUTED_K
        # (e.g. MTP bs>=2 decode: 4*8=32 > K=8) OR HPU_MOE_ROUTED_K unset. The stock
        # VllmMixtureOfExpertsOpWNA16 below would dequant ALL E local int4 experts -> bf16 -> HBM
        # EVERY forward (the confirmed #1 single-stream device cost).
        # D2: run the STOCK BATCHED torch.ops.hpu.mixture_of_experts intrinsic ONCE against the
        # RESIDENT fp8 experts built at load (zero per-step dequant; native fp8 MME ~2x; works at
        # ALL batch sizes -- this replaces the per-expert + per-expert mark_step Python loop in
        # fp8_allexpert_moe that was SLOWER than int4 at bs>=1). Mirrors ops.py
        # VllmMixtureOfExpertsOpFP8PerChannel.forward (dynamic_quant activation + per-expert
        # FORWARD weight scales). To stay byte-faithful to the int4 numeric oracle
        # (VllmMixtureOfExpertsOpWNA16, ops.py:1491) NO extra chunk_size/total_experts kwargs are
        # passed (that op passes none; the single-slice guard lives in _build_fp8_expert_views).
        # EP CONVENTION (identical to the int4 fallback below + the moe_op built LOCAL [0,E-1] at
        # ~L853): the intrinsic ZEROES routed experts when experts_min!=0, so feed LOCAL ids with
        # experts_min=0/experts_max=E-1 + the SAME global->local remap -- NOT global ids + emin.
        # Weights are [in,out] per expert (= convert_from_uint4 orientation) -> permuted_weights
        # =False (the int4 op uses False too; ops.py's FP8PerChannel uses True only because ITS
        # weights are the transposed [out,in] vLLM-standard layout -- ours are NOT). Only when this
        # layer has the stashed fp8 (GPTQ-sym + flag on); else fall through to the stock int4 op
        # (byte-identical when off).
        if _FP8_EXPERTS and getattr(layer, "_fp8_w13", None) is not None and act in ("silu", "gelu"):
            from vllm_gaudi.extension.ops import dynamic_quant
            emin = getattr(layer, "_hpu_experts_min", 0)
            E = int(layer._fp8_w13.shape[0])
            # View tuples are built at load (_build_fp8_expert_views). LOUD if missing rather than
            # silently rebuilding-every-forward or dropping to a slow/wrong path -- the cache is
            # always set by _build_fp8_experts when _fp8_w13 exists, so this only fires on a real
            # load regression.
            if getattr(layer, "_fp8_w13_views", None) is None:
                raise RuntimeError(
                    "hpu-moe-ext D2: fp8 serve path reached but the per-expert fp8 view cache is "
                    "missing (layer._fp8_w13_views is None). _build_fp8_experts must populate it "
                    "at load; refusing to silently fall through to a slow/wrong path.")
            # global->local id remap (mirrors the int4 fallback ~L1574-1575 EXACTLY): in-window ids
            # -> [0,E); non-local / pad slots parked at compact id E (out-of-window) with router
            # weight 0 so they contribute exactly 0. Pads MUST go to E, not 0 -- the op collapses
            # same-id slots, so a zero-weight pad at 0 would DROP a genuinely-routed local expert 0.
            _in_win = (tid >= emin) & (tid < emin + E)
            _ltid = torch.where(_in_win, tid - emin, torch.full_like(tid, E))
            _ltw = torch.where(_in_win, tw_, torch.zeros_like(tw_))
            x_fp8, x_scale = dynamic_quant(x)
            out = torch.ops.hpu.mixture_of_experts(
                hidden_states=x_fp8,
                expert_routing_table=_ltid.to(torch.int64),
                router_weights=_ltw.to(x.dtype),
                w12=layer._fp8_w13_views,
                w3=layer._fp8_w2_views,
                d_scale_hidden_states=x_scale,
                d_scale_w12=layer._fp8_w13_s_views,
                d_scale_w3=layer._fp8_w2_s_views,
                permuted_weights=False,
                activation=act,
                experts_min=0,
                experts_max=E - 1)
            return _dp_out_reshape(out, input_shape, _dp)
        # FUSED MONOLITHIC path (HPU_MOE_FUSED_MONOLITHIC=1) -- the THROUGHPUT regime.
        # When the routed gather is NOT taken (concurrency: tokens*top_k > HPU_MOE_ROUTED_K,
        # or HPU_MOE_ROUTED_K unset), the stock VllmMixtureOfExpertsOpWNA16 below dequantizes
        # ALL E experts to bf16 -> HBM -> a separate mixture_of_experts gemm (the ~99us/token
        # bf16->HBM round-trip floor). Replace it with the SRAM-resident fused monolithic op.
        # Passes the GLOBAL tid/tw_ (the wrapper does its own global->local remap + the dense
        # [E,T,1] routing internally). _fused_monolithic returns None when ineligible (AWQ /
        # non-silu / .so missing) -> we fall straight through to the stock op below (never
        # silently to a worse path). EAGER-ONLY verify (the relNorm .item() breaks lazy capture).
        # T-CAP (HPU_MOE_FUSED_MONO_MAX_T=N, default 0=unlimited): when N>0 the monolithic only
        # takes T<=N (the decode batch); LARGER T (prefill) drops to the bounded stock op below --
        # this is the c64/c128 HBM-fit fix (the monolithic's gu/y intermediates scale with T, so
        # the large-T prefill warmup buckets OOM). See _FUSED_MONO_MAX_T at module top.
        if _FUSED_MONOLITHIC and (_FUSED_MONO_MAX_T <= 0 or _T <= _FUSED_MONO_MAX_T):
            mono = self._fused_monolithic(layer, x, tid, tw_, act)
            if mono is not None:
                if _FUSED_MONOLITHIC_VERIFY:
                    # ref = the STOCK all-expert op with the LOCAL-window remap (the op was
                    # built with window [0,nloc-1], so it needs LOCAL ids -- mirror the remap
                    # below). relNorm < 1e-2 is the bf16 norm-relative tol the fused recipe was
                    # validated against; LOUD + stay-coherent (return stock) on mismatch.
                    _emin = getattr(layer, "_hpu_experts_min", 0)
                    _nloc = getattr(layer, "_hpu_local_experts", None)
                    _rtid, _rtw = tid, tw_
                    if _nloc is not None:
                        _iw = (tid >= _emin) & (tid < _emin + _nloc)
                        _rtid = torch.where(_iw, tid - _emin, torch.full_like(tid, _nloc))
                        _rtw = torch.where(_iw, tw_, torch.zeros_like(tw_))
                    ref = layer.moe_op(x, _rtid, _rtw, permuted_weights=perm, activation=act)
                    diff = (mono.float() - ref.float()).norm().item()
                    rn = diff / (ref.float().norm().item() + 1e-9)
                    import logging
                    if rn >= 1e-2:
                        ma = (mono.float() - ref.float()).abs().max().item()
                        logging.getLogger("vllm").warning(
                            "[fused-monolithic-verify] MISMATCH relNorm=%.4g maxAbs=%.4g "
                            "x0=%d -- falling back to stock mixture_of_experts result"
                            % (rn, ma, x.shape[0]))
                        return _dp_out_reshape(ref, input_shape, _dp)   # stay coherent; no crash
                    logging.getLogger("vllm").info(
                        "[fused-monolithic-verify] relNorm=%.6g x0=%d" % (rn, x.shape[0]))
                return _dp_out_reshape(mono, input_shape, _dp)
            # fall through to the stock all-expert op (AWQ / non-silu / .so missing)
        # GLM-5.1 / EP FIX (pairs with the LOCAL-window op above): remap GLOBAL expert ids ->
        # LOCAL [0, num_local-1] by subtracting this rank's expert offset; non-local experts ->
        # compact id 0 with weight ZEROED so they contribute exactly 0 (mirrors _routed_moe;
        # relying on out-of-window id skipping is unreliable per that path's note). Without this,
        # global ids never match the op's local weight list -> routed output 0 -> salad.
        # EP-FILTER (HPU_MOE_EP_A2A=1): STATIC-capacity compute-win on the DP path. Select C=CAP*N tokens by
        # topk of the has-local-expert score and run the op on only those. The DROPPED tokens are the score-0
        # ones (no local expert -> contribute 0 through the remap anyway), so this is BYTE-EXACT as long as
        # CAP >= the per-rank local-expert fraction (~0.66 for GLM E256/k8/ep8), while cutting ~(1-CAP) of the
        # expert compute. topk->fixed C keeps the shape STATIC == lazy-safe (a dynamic boolean filter crashed:
        # "size of tensor a(256) vs b(0)"). index_select/index_copy avoid boolean indexing entirely.
        if _EP_FILTER and _dp and getattr(layer, "_hpu_local_experts", None) is not None:
            _emin = getattr(layer, "_hpu_experts_min", 0)
            _nloc = layer._hpu_local_experts
            _N = int(tid.shape[0])
            _C = max(1, min(_N, int(_N * _EP_CAP)))                                       # STATIC per bucket
            _score = ((tid >= _emin) & (tid < _emin + _nloc)).any(1).to(torch.float32)    # 1 iff local expert
            _idx = torch.topk(_score, _C, sorted=False).indices                           # STATIC [_C] indices
            _tf = torch.index_select(tid, 0, _idx)
            _wf = torch.index_select(tw_, 0, _idx)
            _iw = (_tf >= _emin) & (_tf < _emin + _nloc)
            _tf = torch.where(_iw, _tf - _emin, torch.full_like(_tf, _nloc))
            _wf = torch.where(_iw, _wf, torch.zeros_like(_wf))
            _of = layer.moe_op(torch.index_select(x, 0, _idx), _tf, _wf, permuted_weights=perm, activation=act)
            out = torch.zeros(_N, _of.size(-1), dtype=_of.dtype, device=_of.device)
            out.index_copy_(0, _idx, _of)
            if not _EP_FILTER_LOGGED[0]:
                _EP_FILTER_LOGGED[0] = True
                import logging
                logging.getLogger("vllm").warning(
                    "[hpu-moe-ext] EP-FILTER engaged (static topk, CAP=%.2f): op on %d/%d gathered tokens "
                    "(byte-exact iff CAP >= per-rank local-expert fraction).", _EP_CAP, _C, _N)
            return out.view(out.size(0), *input_shape[1:])
        emin = getattr(layer, "_hpu_experts_min", 0)
        nloc = getattr(layer, "_hpu_local_experts", None)
        if nloc is not None:
            # Applies to ALL ranks incl. ep_rank 0 (emin=0): the op does NOT cleanly skip
            # out-of-[0,nloc) global ids, so non-local experts must have their WEIGHT zeroed
            # (tid->compact 0). The tid-emin subtraction is a harmless no-op when emin=0.
            in_win = (tid >= emin) & (tid < emin + nloc)
            # Pad/non-local slots MUST NOT use compact id 0: the op collapses same-id
            # slots, so zero-weight pads at id 0 DROP a genuinely-routed local expert 0
            # (proven by moe_op_cpu_judge: only id-0-routed steps mismatch, relNorm
            # ~0.7-0.96, both eager AND lazy). Park pads at nloc (out-of-window, w=0).
            tid = torch.where(in_win, tid - emin, torch.full_like(tid, nloc))
            tw_ = torch.where(in_win, tw_, torch.zeros_like(tw_))
        out = layer.moe_op(x, tid, tw_, permuted_weights=perm, activation=act)
        if _PROBE_N > 0:
            _probe_count[0] += 1
            if _probe_count[0] <= _PROBE_N:
                ep_rank = _resolve_ep_rank(layer, -1)
                emin = getattr(layer, "_hpu_experts_min", 0)
                eloc = getattr(layer, "_hpu_local_experts", 0)
                _probe_log_call(_probe_count[0], ep_rank, emin, emin + eloc - 1,
                                tid, tw_, out)
        # DP: return the GATHERED [N*dp, H] shape so reduce_scatter(DP) splits tokens home
        # correctly; non-DP keeps the exact input shape (byte-identical to stock).
        return out.view(out.size(0), *input_shape[1:]) if _dp else out.view(*input_shape)


class GPTQDequantToBF16LinearMethod(GPTQHPULinearMethod):
    """8-bit (or any non-4-bit) AutoRound/GPTQ linear that this HPU build has
    no working kernel for. Subclass GPTQHPULinearMethod so create_weights and
    the proven int->code unpackers are inherited; OVERRIDE
    process_weights_after_loading to dequantize the packed int32 weight to a
    dense bf16 weight, and OVERRIDE apply to a plain bf16 matmul. Avoids both
    the uint4-only convert_from_uint4 op AND the bits=4-hardcoded pack_tensor.

    Symmetric (sym=True) gptq: w = (code - zero) * scale, with the autogptq
    stored-minus-one convention -> the inherited unpack_zeros adds +1.
    """

    def __init__(self, group_size: int, bits: int = 8):
        # Build a tiny per-layer GPTQ config so pack_factor / weight_bits are
        # correct for the inherited unpackers (they read self.quant_config).
        from vllm_gaudi.ops.hpu_gptq import GPTQHPUConfig
        cfg = GPTQHPUConfig(weight_bits=bits, group_size=group_size,
                            desc_act=False, lm_head_quantized=False)
        super().__init__(cfg)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_hpu_deq_done", False):
            return
        # Reclaim prior layers' pending frees before this layer's unpack stages HPU->CPU (lazy
        # only — mark_step is a no-op in eager). Helps the tighter :next device pool serve the
        # inherited unpack's .cpu() staging during load (which succeeds at gpu-mem 0.98).
        htorch.mark_step()
        bits = self.quant_config.weight_bits
        self.wf = torch.tensor(list(range(0, 32, bits)),
                               dtype=torch.int32).unsqueeze(0)
        # ENTIRE dequant runs ON HPU (no .cpu() anywhere): under lazy the loaded strided TP-shard
        # params cannot be moved to CPU ("Strided Params Has Value: 0 / tensor_data is empty"), but
        # on-HPU access works (the MoE path proves it). The unpack methods are overridden below to
        # the stock math minus the .cpu(); the dequant arithmetic is device-agnostic. Per-layer
        # intermediates are small and freed by the mark_step below.
        codes = self.unpack_weight_from_cuda_old_format(layer).to(torch.float32)   # HPU [in,out]
        zeros = self.unpack_zeros_from_cuda_old_format(layer).to(torch.float32)    # HPU [groups,out]
        scales = layer.scales.data.to(torch.float32)                # [groups,out] HPU
        gs = (self.quant_config.group_size
              if self.quant_config.group_size != -1 else codes.shape[0])
        g_idx = (torch.arange(codes.shape[0], device=codes.device) // gs).long()  # desc_act=False; on-device
        # ROW-PARALLEL FIX (verified against a live rank-4 o_proj dump, cos 0.9999987,
        # maxdiff 1.2e-4; reproduced exactly for all 8 ranks, cos 1.0 maxdiff 0):
        # scales/qzeros are created with scale_and_zero_size = input_size//group_size
        # (FULL groups, declared output_dim-only via ChannelQuantScaleParameter), so they
        # are NOT input-sharded. But `codes` IS the per-rank input shard for a
        # RowParallelLinear (o_proj). A local g_idx then indexes rank-0's groups on EVERY
        # rank -> wrong scales/zeros -> the input-sharded weight collapses (~0.4x,
        # decorrelated) -> the o_proj all-reduce sums corrupt partials -> token salad.
        # (Output-sharded ColumnParallel linears are fine: full-group scales, group count
        # == local.) Offset g_idx into THIS rank's global groups when input-sharded.
        #
        # The offset is exactly the rank's contiguous input-shard index, i.e. shard r owns
        # global input rows [r*in_per : (r+1)*in_per] and global groups [r*local_groups : ...].
        # Derive that index from the LAYER's own partition metadata rather than re-calling
        # get_tensor_model_parallel_rank(): RowParallelLinear stores layer.tp_rank (the rank
        # it actually used to shard its input/codes) at construction, which is robust under
        # --enable-expert-parallel and any TP/EP group context changes between build and here.
        local_groups = codes.shape[0] // gs
        if scales.shape[0] > local_groups:
            # layer.tp_rank is the rank RowParallelLinear used to shard this layer's
            # input/codes (set in its __init__); prefer it. Fall back to the global TP
            # rank only if the attribute is absent.
            shard_idx = getattr(layer, "tp_rank", None)
            if shard_idx is None:
                from vllm.distributed import get_tensor_model_parallel_rank
                shard_idx = get_tensor_model_parallel_rank()
            g_idx = g_idx + int(shard_idx) * local_groups
        z = zeros[g_idx]                                            # [in,out]
        s = scales[g_idx]                                          # [in,out]
        w = (codes - z) * s                                        # [in,out]
        weight = w.t().contiguous().to(torch.bfloat16).to("hpu")    # [out,in]
        for n in ("qweight", "qzeros", "scales", "g_idx"):
            if hasattr(layer, n):
                delattr(layer, n)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        # Free the just-replaced quant weights + this layer's intermediates NOW (lazy: pending-free
        # until flushed). With the MoE-repack reclaim above, this keeps the device pool low+
        # defragmented through load so the next layer's unpack staging fits on the tighter :next
        # image. The model is only ~52GB; the earlier OOMs were load fragmentation, not capacity.
        htorch.mark_step()
        layer._hpu_deq_done = True

    def apply(self, layer, x, bias=None):
        # Plain bf16 matmul. Do NOT use @torch.compiler.disable here.
        #
        # @torch.compiler.disable creates a graph break: dynamo stops tracing at this
        # call and the HPU hpu_backend receives the break output as an opaque tensor.
        # The HPU compiler then fails to handle the subsequent ops (e.g. view/split
        # after kv_b_proj in forward_mha) with:
        #   AttributeError: 'NoneType' object has no attribute 'file'
        # In the cases where it doesn't crash, it produces WRONG outputs (token salad)
        # because the HPU graph generated around the break may not use the eager result
        # correctly. Removing the decorator lets dynamo/hpu_backend trace through this
        # function normally as a plain bf16 F.linear, which compiles and executes
        # correctly.
        #
        # The "521 recompile" warnings seen without this decorator are NORMAL: dynamo
        # compiles one graph variant per (batch_size, seq_len) shape during warmup,
        # and with 75 layers × multiple shapes ≈ 521 total graph variants. This is
        # warmup overhead only, NOT per-inference recompiles.
        import torch.nn.functional as F
        if _DEQ_ABSORB[0]:                       # load-time MLA absorption only (never inference)
            htorch.mark_step()
            out = F.linear(x, layer.weight, bias)
            htorch.mark_step()
            return out
        return F.linear(x, layer.weight, bias)

    # --- HPU-native unpackers (override the inherited .cpu()-based ones) ------------------
    # Identical math to vllm_gaudi.ops.hpu_gptq, but read qweight/qzeros ON their device (HPU)
    # instead of .cpu(). Under lazy the loaded strided TP-shard params can't be moved to CPU
    # ("Strided Params Has Value: 0"); on-HPU access works. Eager (:1.24) is unaffected (same
    # integer result, just computed on-device). self.wf is set in process_weights before these run.
    def unpack_weight_from_cuda_old_format(self, layer):
        qweight = layer.qweight                       # [in//pack, out] int32 (on HPU under lazy)
        bits = self.quant_config.weight_bits
        wf = self.wf.to(qweight.device)
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2 ** bits) - 1)
        return weight.reshape((weight.shape[0] * weight.shape[1], weight.shape[2]))

    def unpack_zeros_from_cuda_old_format(self, layer):
        bits = self.quant_config.weight_bits
        wf = self.wf.to(layer.qzeros.device)
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(layer.qzeros, 2).expand(-1, -1, 32 // bits),
            wf.unsqueeze(0),
        ).to(torch.int16 if bits == 8 else torch.int8)
        zeros = zeros + 1
        zeros = torch.bitwise_and(zeros, (2 ** bits) - 1).to(layer.scales.dtype)
        return zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
