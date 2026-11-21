"""K-EXPERT routed MoE on top of the WORKING single-expert fused op (item 1 of 4).

This layers the routed gather/topk/remap of the proven ``_routed_moe``
(``moe_wna16_hpu.py``) on top of the GC-captured single-expert op

    torch.ops.hpu_moe_ext.fused_expert(
        x[M,H] bf16,
        w13_packed[H, 2I//8] int32, w13_scale[H//GS, 2I] bf16, w13_zp[H//GS, 2I//8] int32,
        w2_packed[I, H//8]  int32, w2_scale[I//GS, H]  bf16, w2_zp[I//GS, H//8]  int32,
        router_weight[1,1] bf16) -> [M, H] bf16

(``tpc_kernel/opbackend/moe_fused_expert_backend.cpp``, loaded from
``tpc_kernel/opbackend/build/libmoe_fused_expert.so``). That op does the FULL
fused per-expert int4 MoE (dequant W13 -> MME -> SiLU -> dequant W2 -> MME ->
``* router_weight``), SRAM-resident, numerically correct (relNorm 4.2e-3 on
device). **It FOLDS the router weight** via its final ``mult_fwd_bf16(y,
router_weight[1,1])`` broadcasting the ``[1,1]`` scalar over the whole ``[M,H]``
output.

All tensors are PyTorch ROW-MAJOR (the op consumes them directly, last dim ==
Synapse FCD), per expert:
    w13_packed [H, 2I//8]   w13_scale [H//GS, 2I]   w13_zp [H//GS, 2I//8]
    w2_packed  [I, H//8]    w2_scale  [I//GS, H]    w2_zp  [I//GS, H//8]

=== ROUTING ===
Reuses the EXACT gather/route/remap of ``MoeWNA16HPUMethod._routed_moe`` (the
397B's single-stream win): present/scatter/topk over a LOCAL-expert window to get
the ``<=K`` routed LOCAL expert ids (``active``), with the global->local remap,
the mandatory ``mark_step()`` BEFORE the data-dependent gather, the functional
(out-of-place) scatters for lazy-HPU-graph-replay safety, and the per-slice
``.clone()`` under lazy for the storage-offset hazard. The DIFFERENCE from
``_routed_moe`` is the tail: instead of one ``torch.ops.hpu.mixture_of_experts``
intrinsic over a dequant list, we call the fused single-expert op once per active
expert and accumulate.

=== PER-(token, slot) ROUTER WEIGHT — the crux ===
The fused op multiplies the WHOLE ``[M,H]`` expert output by ONE ``[1,1]`` scalar.
- Single-stream **T == 1** (the priority; ``K <= 8``): each active expert is
  routed to by at most one (token, slot), so its router weight collapses to a
  SINGLE scalar ``w_k`` shape ``[1,1]``. We pass that ``[1,1]`` straight to the
  op so it FOLDS the weight exactly as on the single-expert validation path — no
  post-scaling, the op's recipe does the multiply in bf16 on device.
- **T > 1** (prefill / batch): a single ``[1,1]`` scalar cannot carry a different
  weight per token, so we pass ``router_weight = 1`` to the op (un-weighted expert
  output) and apply the per-token ``[T,1]`` weight ``w_k`` OUTSIDE, accumulating
  ``expert_out * w_k``. Numerically identical (the fold is just ``y * w``); it only
  moves the multiply out of the captured recipe for the >1 case the op can't fold.

Either way the math is: ``out = sum_over_active_e( fused_expert(x, W_e) * w_{e} )``
where ``w_{e}`` for token t is the sum of router weights of every top_k slot of t
that routed to expert e (0 if none) — identical to ``_routed_moe`` /
``fused_routed_moe``'s ``w_k``.

Lazy container (PT_HPU_LAZY_MODE=1), Habana 1.24. This is a free function; it does
NOT touch the production ``MoeWNA16HPUMethod``. RUN ON THE HPU BOX (the .so must be
loaded + the op exercised on device).
"""
from __future__ import annotations

import os

import torch
import habana_frameworks.torch.core as htorch

# ENGAGE: LOUD silent-fallback log + HPU_ENGINE_STRICT hard-fail. CPU-safe import.
from .engage_manifest import engage_fallback

# ---------------------------------------------------------------------------
# CAPTURE-SAFE ROUTING (HPU-graph wrap_in_hpu_graph) — see fused_moe_routed below.
#
# Under LAZY serving the whole model.forward is recorded by wrap_in_hpu_graph into
# ONE recipe and replay()'d every decode token. The per-expert eager loop below
# materializes each expert's weight/zp with an in-graph `.contiguous().clone()`
# (the offset-0 fix). Under capture those K clones are FRESH intermediates the
# capturing allocator owns; with vllm-gaudi's lazy default disable_tensor_cache=True,
# mark_user_outputs(..., free_inplace=True) FREES such in-place/intermediate buffers
# after capture, so at replayV3 the fused-op's per-expert `hpu__input` points at
# freed storage -> the serve's FATAL:
#   PT_LAZY ValidateSyncInputTensors tensor_data is empty ...
#   model/3/mlp/experts/hpu__input ... Strided Params Has Value: 0
# (layer 3 == the first int4 MoE layer). The single-EXPERT op captures fine because
# it is one bare call on PERSISTENT external tensors with no in-graph clone; the NEW
# variable is this wrapper's python-side per-expert clone inside the captured region.
#
# FIX (option c — the diagnosis-supported one): under lazy, do NOT clone per-expert
# in python. Route the WHOLE K-expert MoE through the SINGLE-RECIPE C++ op
# `torch.ops.hpu_moe_ext.fused_moe` (TensorList; moe_fused_expert_routed_backend*.cpp,
# libmoe_fused_routed_lazy.so) — it emits all K experts' fused int4 subgraphs + the
# weighted sum as ONE captured recipe whose inputs are the persistent stacked weights
# (the K slices happen ON-DEVICE inside the recipe via the op's own slice nodes, so
# there is NO freed python intermediate to validate at replay). Its lazy-capture test
# (tests/test_fused_moe_single_recipe.py CHECK 2) already PASSES. We feed it the K
# experts as `.unbind(0)` views of the index_select outputs (the exact feed the
# backend header prescribes: "index_select -> [K,...] then .unbind(0) to lists").
#
# The single-recipe op FOLDS a per-expert [1,1] router weight (the T==1 form), so it
# is used under lazy ONLY at T==1 (the single-stream decode priority the whole fused
# path targets; the routed gate already guarantees T*top_k <= K). For T>1 a [1,1]
# scalar cannot carry a per-token weight, so we keep the per-expert loop (un-weighted
# op + outside [T,1] weighting) — but T>1 is prefill, which is NOT wrapped in the
# decode HPU-graph, so the in-graph-clone hazard does not apply there. EAGER stays the
# per-expert loop in all cases (no capture -> no freed-intermediate class; the
# offset-0 clone is the proven real-weight correctness path, relNorm 4e-3).
#
# Set HPU_MOE_ROUTED_VIA_SINGLE_RECIPE=0 to force the legacy per-expert loop even
# under lazy (debug/bisect only — it reproduces the capture FATAL).
#
# SECOND CAPTURE HAZARD — MULTI-BUCKET (the single-recipe op fixed the WEIGHTS, not x):
# the single-recipe route makes the K weight inputs persistent (rooted at nn.Parameters
# via index_select), so the ONE-bucket capture is coherent. But the op's FIRST input,
# x (the experts-input activation), arrives as an `aten::view` of a NON-persistent decode
# intermediate (apply_monolithic: `x = x.view(-1, H)` off layer-0's post-attn norm). The
# bridge roots the recipe's x input at that view's BASE lazy-tensor. One decode bucket is
# fine, but when generation grows into a SECOND decode bucket ('decode',1,1,2) the prior
# bucket's replay has already freed x's base (SingleHPUGraph::replayGraph frees every
# output-that-is-not-an-input; + the cross-graph input hazard clear_inputs documents), so
# the new bucket's on-the-fly capture validates an EMPTY x ->
#   ValidateSyncInputTensors tensor_data is empty ... model/0/mlp/experts/hpu__input ...
#   Strided Params Has Value: 0.
# free_inplace=False does NOT cover x (HPUGraph::mark_user_outputs exempts only
# slice_insert/strided_insert nodes; x's producer is a view of an add/norm).
#
# WHAT DID NOT WORK: materializing x as an in-forward `x.contiguous().clone()` does NOT
# fix the 2nd bucket. A clone created in THIS wrapper runs INSIDE the captured forward, so
# it is an in-graph INTERMEDIATE (an output-not-input), NOT a graph input. mark_user_outputs
# exempts a tensor from freeing only when its lazy-id is in the recipe INPUT set
# (hblazy_tensors_in_: is_input, HPUGraph.cpp:351) or it is a slice/strided_insert producer
# (free_inplace=False, :362). A fresh clone is neither -> the free_inplace pass reclaims it
# across buckets exactly like the raw view -> the 2nd bucket STILL validates empty x.
#
# FIX (B — the canonical HPU-graph STATIC-INPUT pattern, == vllm-gaudi's self.input_ids_hpu:
# a buffer allocated ONCE in __init__, copy_()'d each decode step, a slice fed to the wrapped
# graph): each MoE layer pre-allocates a PERSISTENT [max_T, H] bf16 `x_buf` at WEIGHT-LOAD
# time (moe_wna16_hpu._alloc_fused_x_buf, OUTSIDE any capture); the wrapper writes
# `x_buf[:T].copy_(x)` and feeds `x_buf[:T]` to the op. x_buf is a held leaf the layer owns
# (like an nn.Parameter / the weights), so its base lazy-id IS in hblazy_tensors_in_ ->
# is_input -> NEVER freed at replay; AND the in-place copy_ is a strided_insert that
# free_inplace=False also exempts. So EACH bucket's recipe roots x at the SAME live, held
# storage. Gated HPU_MOE_FUSED_X_BUFFER (moe_wna16_hpu, default ON); x_buf=None falls back to
# the in-forward clone (single-bucket only, bisect). See the long comment at the op call site.
# ---------------------------------------------------------------------------
_VIA_SINGLE_RECIPE = os.environ.get("HPU_MOE_ROUTED_VIA_SINGLE_RECIPE", "1") != "0"

# HPU_MOE_FUSED_BATCHED=1 -> route the captured lazy/T==1 fast path to the BATCHED-K
# single-recipe op `torch.ops.hpu_moe_ext.fused_moe_batched` (one chain of 16 nodes
# INDEPENDENT of K: grouped dequant over the stacked [K,...] weights + a batch_gemm
# over the K-expert batch + batched SiLU + 2nd grouped dequant + 2nd batch_gemm + the
# router-weighted reduce-sum over K) INSTEAD of the SEQUENTIAL single-recipe op (K
# folds of emit_one_expert -> ~K*14 node launches). Gaudi-2 decode is host/launch-
# bound (memory gaudi-single-stream-speed), so cutting K*14 -> 16 launches/token is
# the single-stream latency lever. DEFAULT OFF: the SEQUENTIAL op is the validated
# fallback; flip this on only after the batched op is validated on device
# (tests/test_fused_moe_batched.py + the qwen35-397b-sslazy8 serve tok/s vs the
# sequential op + stock). It engages ONLY under the SAME captured regime the
# sequential fast path uses (lazy + T==1, K guaranteed by the routed gate); eager /
# T>1 keep the per-expert loop. The batched op is fed the index_select outputs
# DIRECTLY as stacked [K,...] tensors (NO .unbind) + a [K,1,1] router tensor, so the
# held-leaf x_buf + the persistent stacked zero-points (_alloc_fused_zp) flow in as
# single persistent bases -- multi-bucket capture stays coherent (NOT regressed).
_VIA_BATCHED = os.environ.get("HPU_MOE_FUSED_BATCHED", "0") != "0"

# HPU_MOE_ROUTED_SEQ_OFFSET_CLONE=0 -> feed the SEQUENTIAL single-recipe op the RAW
# .unbind(0) views of the index_select outputs (the pre-fix path). DEFAULT ON (=1): clone
# every per-expert slice to a FRESH offset-0 contiguous leaf (_co) before the op call.
#
# WHY THIS IS THE bs>1 FIX (root-caused to the bridge, not the kernel math):
#   Each .unbind(0) element w13p[k] for k>0 is a VIEW with storage_offset = k*stride[0] into
#   the index_select buffer. The op's LAZY frontend (moe_fused_expert_routed_backend_lazy.cpp:
#   67-80) builds the LazyOp from those views but NEVER calls hpu_op.viewUpdateInputs() -- the
#   one bridge call that consumes a view's stride_params to emit an on-device slice/strided_view
#   node carrying the offset (lazy_kernels.h viewUpdateInputs -> habana_lazy/view_utils.cpp
#   HandleViews, where params.offset is injected). Without it, each view's IR binds to the
#   SHARED index_select storage BASE (HbLazyTensor::GetIrValue -> IrReconnectAsInputNode), the
#   per-view offset is silently DROPPED, and the op's FIRST node -- reinterpret_cast int32->uint4
#   (moe_fused_expert_emit.h L103-110), the most offset-sensitive consumer in the codebase --
#   reads from offset 0 == compact-expert-0's bytes for EVERY k. So a real expert at k>0 dequants
#   GARBAGE -> the serve's maxAbs~200 / relNorm~1e11 at bs>1 (a no-local-expert rank trivially
#   returns 0, masking it on some ranks). The bridge's OWN mixture_of_experts op calls
#   op.viewUpdateInputs() in exactly the slot this custom op left empty; the custom op was modeled
#   on the fp8 template, which is fed pre-materialized contiguous weights and so never needed it.
#   (Under EAGER the bridge DOES encode the offset via the duplicate-section path, so eager was
#   already correct -- but the SERVE is lazy, where the offset is dropped.)
#
#   The _co clone (.contiguous().clone()) makes each input a FRESH offset-0 tensor with EMPTY
#   stride_params, so the missing viewUpdateInputs() is irrelevant and the IR binds to the correct
#   offset-0 storage -- on BOTH paths. It is strictly more robust than adding viewUpdateInputs()
#   to the lazy TU (which would fix only lazy + still emit an extra per-expert slice node under
#   capture); it removes the entire wrong-offset read class. This is the SAME fix the eager loop
#   below (_co at L605-606/615-616, relNorm~4e-3) and the load-time precompute helpers
#   (_alloc_fused_fp8_scale / _build_fp8_experts: "clone under lazy so convert_from_uint4 reads
#   the WRONG expert under lazy; .contiguous() does NOT reset storage_offset for a clean sub-block")
#   already apply -- now brought to the captured SEQUENTIAL op so the bs>1 MTP-verify (T=depth+1)
#   and tree-verify (T=K) batches are numerically correct. The K(<=~16) small clones are negligible
#   at single-stream / the verify batch. Set =0 ONLY to bisect / A-B the 1e11 reproduction.
_SEQ_OFFSET_CLONE = os.environ.get("HPU_MOE_ROUTED_SEQ_OFFSET_CLONE", "1") != "0"

# One-shot ENGAGEMENT signal (per worker): emit the FIRST time the SEQUENTIAL single-recipe
# op runs with the bs>1 offset-0 _co clone ACTIVE (and once for the raw-feed bisect path).
# This is the POSITIVE proof the fix engaged -- in the serve/test logs it confirms the bs>1
# captured path is feeding offset-0 clones (not the raw .unbind views that read expert-0's
# bytes -> relNorm~1e11). Pairs with the on-card A/B (SEQ_OFFSET_CLONE=1 PASS vs =0 FAIL).
_logged_seq_offset_clone = [False]

# One-shot guard: LOUDLY warn (once) if the captured single-recipe regime (lazy T==1)
# runs WITHOUT the persistent static-input x buffer — i.e. it took the in-forward clone
# (or raw view) fallback, which is the KNOWN multi-bucket-FATAL path. In a correct serve
# the buffer is allocated at weight-load (moe_wna16_hpu._alloc_fused_x_buf) for every MoE
# layer, so this fallback must NOT fire; if it does (a missing/None buffer regression) the
# serve would otherwise re-crash cryptically at the 2nd decode bucket. Surface it instead.
_warned_no_x_buf = [False]

# One-shot INFO: emit the FIRST time the persistent static-input x buffer is actually
# fed to the single-recipe op (the buffer path at line ~307 is taken). This is the
# POSITIVE engagement signal: if this line is in the serve logs, x is being routed
# through the held-leaf buffer (multi-bucket-safe) rather than the raw view / clone
# fallback. Pairs with _warned_no_x_buf (the NEGATIVE signal). Per worker process.
_logged_x_buf_engaged = [False]

# HPU_MOE_FUSED_X_MATERIALIZE=0 -> drop even the in-forward CLONE FALLBACK (used only when
# the persistent x_buf is absent — HPU_MOE_FUSED_X_BUFFER=0 — or T exceeds the buffer's
# rows), feeding the single-recipe op the raw `aten::view` x. That reproduces the MULTI-
# BUCKET capture FATAL (x's view-base is freed by the 1st bucket's replay -> the 2nd
# bucket's capture validates an empty x). With the persistent buffer ON (the default) this
# flag is irrelevant — x_buf handles x. Default ON (keep the clone fallback). The PRIMARY
# fix is the persistent x_buf (HPU_MOE_FUSED_X_BUFFER); this is the legacy fallback toggle.
_MATERIALIZE_X = os.environ.get("HPU_MOE_FUSED_X_MATERIALIZE", "1") != "0"

# W4A8: HPU_MOE_ROUTED_INT4FP8=1 routes the SEQUENTIAL single-recipe routed path to the
# int4->fp8 op `torch.ops.hpu_moe_ext.fused_moe_routed_int4fp8` (dequant int4->fp8
# in-SRAM + native fp8 MME, 2x bf16; DPQ arXiv 2505.14638 = up to 3x over W4A16 on
# Gaudi-2 -- the SAME DIRECT-fp8 transform the monolithic op ships at ~1.33x). The op
# signature is IDENTICAL to the bf16 routed `fused_moe` (x + 7 per-expert TensorLists +
# the [1,1]/[T,1] router fold), so the entire routing/gather/feed in fused_moe_routed
# below is UNCHANGED -- only the bound op + the .so differ. It accelerates the
# single-stream decode (T==1) and the MTP-verify batch (T=depth+1) routed gathers. The
# .so is lazy-only (the serving path); built by build_routed_int4fp8_lazy.sh. DEFAULT
# OFF -> the bf16 routed op is byte-identical/unchanged.
_ROUTED_INT4FP8 = os.environ.get("HPU_MOE_ROUTED_INT4FP8") == "1"

# GATE-A: HPU_MOE_ROUTED_TILE_BF16=<Kt> (a POSITIVE int) routes the SEQUENTIAL
# single-recipe routed path to the BYTE-EXACT CONTRACTION-TILED bf16 fold op
# `torch.ops.hpu_moe_ext.fused_moe_routed_tiled_bf16` (slice the contraction dim into
# Kt-row tiles whose dequanted bf16 tile fits SRAM, dequant int4->bf16 per tile in the
# SAME graph, per-tile bf16 MME with an fp32 accumulator carried across tiles, ONE final
# bf16 round). This is the discriminator for whether ELIMINATING the int4-weight-
# materialization HBM round-trip (the ~25 MB/expert spill the stock op forces) moves the
# bs=1 step. The op signature is IDENTICAL to the bf16 routed `fused_moe` (x + 7
# per-expert TensorLists + the [1,1]/[T,1] router fold), so the entire routing/gather/
# feed in fused_moe_routed below is UNCHANGED -- only the bound op + the .so differ. The
# tile size Kt is read from THIS SAME env INSIDE the C++ emit (NOT a schema arg); the
# Python side detects only that the flag is SET (Kt>0) to select the op + .so. relNorm
# ~1e-6 vs stock -> identical greedy tokens (the lossless bar). The .so is lazy-only (the
# serving path); built by build_routed_tiled_bf16_lazy.sh. DEFAULT OFF -> the bf16 routed
# op is byte-identical/unchanged (strict no-op unless the flag is a positive int).


def _parse_tile_bf16_kt():
    """Parse HPU_MOE_ROUTED_TILE_BF16 -> the tile size Kt as an int (0 == unset/invalid
    == op NOT engaged). The C++ emit re-parses the same env for the actual tiling (and
    rounds Kt down to a multiple of the int4 group size); the Python side uses only the
    >0 truth to bind the tiled op + .so. A non-positive / non-numeric value is a strict
    no-op (the stock / int4fp8 op stays selected)."""
    e = os.environ.get("HPU_MOE_ROUTED_TILE_BF16")
    if e is None or e == "":
        return 0
    try:
        v = int(e)
    except (TypeError, ValueError):
        return 0
    return v if v > 0 else 0


_ROUTED_TILE_BF16_KT = _parse_tile_bf16_kt()
_ROUTED_TILE_BF16 = _ROUTED_TILE_BF16_KT > 0

# Mutual exclusion: the tiled-bf16 and int4fp8 flags each bind a DIFFERENT single-recipe
# op (a DIFFERENT .so); enabling BOTH is a misconfiguration that would silently pick one
# and mismeasure the other. Fail LOUD at import (this module is imported lazily, only when
# the routed/fused path is actually exercised, so the raise surfaces exactly the A/B that
# is mis-set -- it never spuriously breaks the stock path's import).
if _ROUTED_TILE_BF16 and _ROUTED_INT4FP8:
    raise RuntimeError(
        "hpu-moe-ext: HPU_MOE_ROUTED_TILE_BF16 and HPU_MOE_ROUTED_INT4FP8 are MUTUALLY "
        "EXCLUSIVE routed-op selectors (each binds a different single-recipe op + .so). "
        "Set exactly ONE: HPU_MOE_ROUTED_TILE_BF16=<Kt> for the byte-exact "
        "contraction-tiled bf16 fold (GATE-A), or HPU_MOE_ROUTED_INT4FP8=1 for the "
        "int4->fp8 op. Got TILE_BF16=%r and INT4FP8=1." % os.environ.get(
            "HPU_MOE_ROUTED_TILE_BF16"))

if _ROUTED_TILE_BF16:
    _ROUTED_OP = "fused_moe_routed_tiled_bf16"
elif _ROUTED_INT4FP8:
    _ROUTED_OP = "fused_moe_routed_int4fp8"
else:
    _ROUTED_OP = "fused_moe"

_routed_so_loaded = [False]


def _load_single_recipe_op():
    """Load the K-expert single-recipe routed op .so and return
    torch.ops.hpu_moe_ext.<_ROUTED_OP> (the bf16 `fused_moe`, or the W4A8
    `fused_moe_routed_int4fp8` op under HPU_MOE_ROUTED_INT4FP8=1), or None if
    unavailable (caller then falls back to the per-expert loop). Idempotent. The .so
    co-loads with the single-expert .so via TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, _ROUTED_OP):
        return getattr(torch.ops.hpu_moe_ext, _ROUTED_OP)
    if _routed_so_loaded[0]:
        return None
    _routed_so_loaded[0] = True
    if _ROUTED_TILE_BF16:
        name = "libmoe_fused_routed_tiled_bf16_lazy.so"   # lazy-only (the GATE-A serving path)
    elif _ROUTED_INT4FP8:
        name = "libmoe_fused_routed_int4fp8_lazy.so"   # lazy-only (the serving path)
    else:
        name = ("libmoe_fused_routed_lazy.so" if htorch.is_lazy()
                else "libmoe_fused_routed.so")
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        import logging
        build_hint = ("build_routed_tiled_bf16_lazy.sh" if _ROUTED_TILE_BF16
                      else "build_routed_int4fp8_lazy.sh" if _ROUTED_INT4FP8
                      else "build_routed%s.sh" % ("_lazy" if htorch.is_lazy() else ""))
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: single-recipe %s op missing (%s); routed fused "
            "path falls back to the per-expert loop (NOT capture-safe under lazy). "
            "Build it: bash tpc_kernel/opbackend/%s", _ROUTED_OP, so, build_hint)
        # Reached only inside the captured-lazy fast path -> the per-expert in-graph-clone
        # loop is the KNOWN multi-bucket FATAL. UNINTENDED (the user enabled the fused op)
        # -> intended=True so HPU_ENGINE_STRICT hard-fails rather than serving a path that
        # re-crashes / silently slow-paths. For the int4fp8 op, the strict gate is the
        # explicit HPU_MOE_ROUTED_INT4FP8 flag (a missing .so must fail LOUD, never
        # silently serve the bf16/per-expert path the user did not ask for).
        engage_fallback(
            "fused_expert_routed._load_single_recipe_op:so-missing",
            fast="single-recipe %s op (capture-safe routed path)" % _ROUTED_OP,
            reason="library not found at %s" % so,
            slow="per-expert in-graph-clone loop (NOT capture-safe under lazy; "
                 "multi-bucket FATAL)",
            intended=True,
            strict_env=("HPU_MOE_ROUTED_TILE_BF16" if _ROUTED_TILE_BF16
                        else "HPU_MOE_ROUTED_INT4FP8" if _ROUTED_INT4FP8
                        else "HPU_MOE_FUSED_OP"))
        return None
    torch.ops.load_library(so)
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, _ROUTED_OP):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded single-recipe %s op from %s (capture-safe routed "
            "path%s)", _ROUTED_OP, so,
            ("; byte-exact contraction-tiled bf16 fold, Kt=%d (GATE-A: SRAM-resident "
             "dequant, fp32 accumulate)" % _ROUTED_TILE_BF16_KT) if _ROUTED_TILE_BF16
            else "; W4A8 int4->fp8 in-SRAM dequant + native fp8 MME (2x; DPQ-validated)"
            if _ROUTED_INT4FP8 else "")
        return getattr(torch.ops.hpu_moe_ext, _ROUTED_OP)
    engage_fallback(
        "fused_expert_routed._load_single_recipe_op:no-register",
        fast="single-recipe %s op" % _ROUTED_OP,
        reason="loaded %s but torch.ops.hpu_moe_ext.%s did not register" % (so, _ROUTED_OP),
        slow="per-expert in-graph-clone loop (NOT capture-safe under lazy)",
        intended=True,
        strict_env=("HPU_MOE_ROUTED_TILE_BF16" if _ROUTED_TILE_BF16
                    else "HPU_MOE_ROUTED_INT4FP8" if _ROUTED_INT4FP8
                    else "HPU_MOE_FUSED_OP"))
    return None


_batched_so_loaded = [False]


def _load_batched_op():
    """Load the BATCHED-K single-recipe op .so (lazy variant under lazy, eager else)
    and return torch.ops.hpu_moe_ext.fused_moe_batched, or None if unavailable (caller
    then falls back to the SEQUENTIAL single-recipe op). Idempotent. The .so co-loads
    with the single-expert + sequential .so via TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_batched"):
        return torch.ops.hpu_moe_ext.fused_moe_batched
    if _batched_so_loaded[0]:
        return None
    _batched_so_loaded[0] = True
    name = ("libmoe_fused_batched_lazy.so" if htorch.is_lazy()
            else "libmoe_fused_batched.so")
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: BATCHED-K fused_moe_batched op missing (%s); HPU_MOE_FUSED_"
            "BATCHED=1 requested but the .so is absent -- falling back to the SEQUENTIAL "
            "single-recipe op. Build it: bash tpc_kernel/opbackend/build_batched%s.sh",
            so, "_lazy" if htorch.is_lazy() else "")
        # The SEQUENTIAL op is correctness-equivalent, but the user asked for the BATCHED
        # speed lever (K*14 -> 16 launches/token) and would silently measure the slower
        # sequential path -> intended=True so HPU_ENGINE_STRICT hard-fails.
        engage_fallback(
            "fused_expert_routed._load_batched_op:so-missing",
            fast="HPU_MOE_FUSED_BATCHED batched-K fused op (single-stream speed)",
            reason="library not found at %s" % so,
            slow="SEQUENTIAL single-recipe op (correct, but K*14 launches/token)",
            intended=True, strict_env="HPU_MOE_FUSED_BATCHED")
        return None
    torch.ops.load_library(so)
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_batched"):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded BATCHED-K fused_moe_batched op from %s "
            "(K*14 -> 16 launches/token; single-stream speed path)", so)
        return torch.ops.hpu_moe_ext.fused_moe_batched
    engage_fallback(
        "fused_expert_routed._load_batched_op:no-register",
        fast="HPU_MOE_FUSED_BATCHED batched-K fused op",
        reason="loaded %s but torch.ops.hpu_moe_ext.fused_moe_batched did not register"
               % so,
        slow="SEQUENTIAL single-recipe op",
        intended=True, strict_env="HPU_MOE_FUSED_BATCHED")
    return None


def fused_moe_routed(x, w13_packed, w13_scale, w13_zp,
                     w2_packed, w2_scale, w2_zp,
                     topk_ids, topk_weights, experts_min, K, x_buf=None):
    """K-expert routed int4 MoE via the fused single-expert op.

    Args (stacked per-expert tensors, PyTorch row-major, on HPU):
      x            : [T, H] bf16 hidden states (T tokens; single-stream T=1, K<=8 priority)
      w13_packed   : [E, H, 2I//8] int32 stacked packed gate+up weights
      w13_scale    : [E, H//GS, 2I] bf16 op-layout scales
      w13_zp       : [E, H//GS, 2I//8] int32 packed zero-points (per-expert slice)
      w2_packed    : [E, I, H//8]  int32 stacked packed down weights
      w2_scale     : [E, I//GS, H] bf16 op-layout scales
      w2_zp        : [E, I//GS, H//8] int32 packed zero-points (per-expert slice)
      topk_ids     : [T, top_k] int — GLOBAL expert ids (this rank's window starts at experts_min)
      topk_weights : [T, top_k] — router weights (already softmax/renormalized)
      experts_min  : this rank's first GLOBAL expert id (ep_rank * E)
      K            : STATIC gather width. Caller sizes it to COVER the routed union of
                     distinct local experts. bs=1 / conservative T>1: K=min(E,T*top_k)
                     guarantees coverage (union<=T*top_k). The bs>1 VERIFY K-bucket may
                     set K below T*top_k (e.g. 16 at bs=4) betting union<=K; if the union
                     ever exceeds K, topk(present,K) truncates and the dropped expert is
                     cleanly EXCLUDED (sentinel-K remap below), NOT wrong-folded -- a
                     small bounded error the verify oracle catches, never corruption.
      x_buf        : OPTIONAL persistent [max_T, H] bf16 static-input buffer for x (the
                     MULTI-BUCKET capture fix). When supplied AND the captured single-recipe
                     path is taken (lazy, T==1), x is written into it in place
                     (`x_buf[:T].copy_(x)`) and the op's x input is `x_buf[:T]` — a slice of
                     a held leaf the layer allocated OUTSIDE capture, so the op roots its x
                     input at storage that survives every decode bucket's replay (an
                     in-forward clone does NOT: it is an in-graph intermediate the
                     free_inplace pass reclaims). None -> the in-forward clone fallback
                     (single-bucket only). See the long comment at the materialization site.

    Returns: [T, H] bf16 routed output (sum over the <=K locally-routed experts,
    each weighted by its per-(token,slot) router weight).

    NOTE the zero-point layout: this signature takes a STACKED ``w13_zp[E,...]`` /
    ``w2_zp[E,...]`` so it works for BOTH GPTQ-symmetric (the shared constant
    zero, broadcast into [E,...] by the caller) AND AWQ per-expert zeros. The
    fused op wants a per-expert [H//GS, 2I//8] / [I//GS, H//8] zp slice, which we
    gather alongside the weights.
    """
    E = w13_packed.shape[0]
    emin = int(experts_min)
    dev = x.device
    # The fused op's dequant nodes are built with the bf16 `dequantize_4_bit` guid
    # (moe_fused_expert_backend_shared.h: get_guid_with_precision("dequantize_4_bit",
    # bf16)) and its router-mult is mult_fwd_bf16, so EVERY scale/router input MUST
    # be bf16 — independent of what dtype w13_scale arrives as (a model loaded in
    # float16 would otherwise feed fp16-byte scales into a bf16-typed dequant ->
    # reinterpreted garbage). Pin the op's scale/router dtype to bf16 here.
    dt = torch.bfloat16            # the op's hardcoded scale/zp(out)/router dtype
    T = x.shape[0]
    H = x.shape[-1]

    # ---- ROUTING: verbatim mirror of _routed_moe (see that method for the deep
    #      rationale on each lazy-HPU-graph hazard) ----
    # global expert id -> local [0, E)
    local = topk_ids.to(torch.int64) - emin
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)                       # routed to a LOCAL expert?
    flat_safe = torch.where(valid, flat, torch.zeros_like(flat))   # masked-safe gather index

    # Mark routed-LOCAL experts; non-local entries scatter to a DUMMY bin E (NOT
    # index 0, which would clobber a genuinely-routed local expert 0). FUNCTIONAL
    # scatter (not scatter_): out-of-place re-materializes every decode step ->
    # graph-replay-safe.
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
        0, scatter_idx, valid.to(torch.float32))[:E]
    active = torch.topk(present, K).indices.to(torch.int64)        # [K] local ids (routed first)

    # *** MULTI-BUCKET CAPTURE FIX (single-recipe op only): route x through a
    # PERSISTENT, pre-allocated static-input buffer ***. The single-recipe op consumes x
    # as its FIRST recipe input (syn_in(0)). In the serve x arrives as an `aten::view` of
    # a NON-persistent decode intermediate (apply_monolithic: `x = x.view(-1, H)` off
    # layer-0's post-attn norm); the bridge roots the recipe's x input at that VIEW'S BASE
    # lazy-tensor (HPUGraph get_hb_base_tensor_id_list_if_view walks stride_params to the
    # base). The K weights are safe — index_select roots them at PERSISTENT nn.Parameters —
    # but x's base is freed at the FIRST decode bucket's replay (SingleHPUGraph::replayGraph
    # -> SetTensorDataNullOpt for every output-that-is-not-an-input; + the cross-graph
    # "graph1 input x; graph2 input x" hazard clear_inputs documents). So the SECOND decode
    # bucket ('decode',1,1,2), captured ON-THE-FLY mid-generation, validates an empty x ->
    # the serve's FATAL: ValidateSyncInputTensors tensor_data is empty ...
    # model/0/mlp/experts/hpu__input ... Strided Params Has Value: 0.
    #
    # WHY A PLAIN `x.contiguous().clone()` HERE DID NOT FIX IT (the failed prior attempt):
    # a clone created INSIDE this wrapper runs INSIDE the captured forward, so it is an
    # in-graph INTERMEDIATE — an OUTPUT of the recipe, not a graph input. HPUGraph::
    # mark_user_outputs only exempts a tensor from freeing when its lazy-id is in the
    # recipe's INPUT set (hblazy_tensors_in_: is_input, HPUGraph.cpp:351) OR it is a
    # slice/strided_insert producer (free_inplace=False, HPUGraph.cpp:362). A fresh clone
    # born in-forward is NEITHER (its producer is an aten::clone of a view-of-add/norm), so
    # with disable_tensor_cache=True the free_inplace pass reclaims it across buckets just
    # like the raw view — the 2nd bucket still validates empty x.
    #
    # FIX (B — the canonical HPU-graph STATIC-INPUT pattern, identical to vllm-gaudi's
    # self.input_ids_hpu: a buffer allocated ONCE in __init__, then copy_()'d each decode
    # step and a slice fed to the wrapped graph): the caller (_fused_moe_routed) hands us a
    # PERSISTENT `x_buf` [max_T, H] bf16 allocated at WEIGHT-LOAD time, OUTSIDE any capture.
    # We write x INTO it in place (`x_buf[:T].copy_(x)`) and feed `x_buf[:T]` to the op.
    # Because x_buf pre-exists the recipe and the layer holds it (a leaf, like an
    # nn.Parameter / the weights), its base lazy-id IS in hblazy_tensors_in_ -> is_input ->
    # NEVER freed at replay; AND the in-place copy_ is a strided_insert, which
    # free_inplace=False (the __init__.py patch, default on) ALSO exempts. So EACH bucket's
    # recipe roots x at the SAME live, held storage — the clone could not achieve this
    # because it was reborn (and reclaimed) inside every capture. The copy_ into a
    # persistent buffer is the standard HPU-graph static-decode-input write; it is
    # capture-safe (recorded once, replayed each step like input_ids_hpu's copy_/scatter_).
    # x_buf is sized [_FUSED_X_BUF_T, H] (T==1 steady state); if T ever exceeds its rows we
    # fall back to the in-forward clone for that step (correct, just not multi-bucket-proof).
    # HPU_MOE_FUSED_X_BUFFER=0 (caller) passes x_buf=None -> the in-forward clone path below
    # (reproduces the multi-bucket FATAL: bisect only). HPU_MOE_FUSED_X_MATERIALIZE=0 drops
    # the clone fallback. The lazy single-recipe path uses x_op; eager keeps x as-is.
    #
    # bs>1 (MTP-VERIFY) note: the captured single-recipe path now also serves T>1 (the verify
    # batch), so _x_capture is T-AGNOSTIC under lazy. The serve wraps the TARGET model in
    # wrap_in_hpu_graph(disable_tensor_cache=True) (hpu_model_runner.py ~L4765) and warms the
    # EXPANDED spec-decode decode buckets (bucketing/common.py generate_spec_decode_buckets),
    # so the bs=(depth+1) verify forward is REPLAYED from a captured graph too -> x's view-base
    # is freed across buckets EXACTLY as at T==1. So the persistent x_buf is needed at T>1 as
    # well; x_buf is sized _FUSED_X_BUF_T (default 8 >= a depth-3 verify's T=4), so T=4 lands
    # in the buffer. (The OLD "T>1 is prefill, uncaptured" assumption did NOT hold for the MTP
    # verify, which is a captured DECODE batch -- that mis-assumption is why the T>1 in-graph-
    # clone loop FATAL'd / produced relNorm~1e11 when bs>1 routing was attempted.)
    _x_capture = (_VIA_SINGLE_RECIPE and htorch.is_lazy())
    if _x_capture and x_buf is not None and x_buf.shape[0] >= T:
        # PERSISTENT static-input buffer: in-place write + slice. x_buf is the held leaf
        # (recipe input, survives every bucket); x_buf[:T] is a view whose base is x_buf.
        # copy_ casts dtype implicitly; the op pins bf16 and apply_monolithic feeds bf16 x,
        # so this is a same-dtype/same-device write — but guard against a silent surprise.
        if x_buf.dtype != x.dtype or x_buf.device != x.device:
            raise RuntimeError(
                "hpu-moe-ext fused_moe x-buffer mismatch: buf is "
                f"{x_buf.dtype}/{x_buf.device} but x is {x.dtype}/{x.device}; "
                "the persistent static-input buffer must match x exactly "
                "(see _alloc_fused_x_buf).")
        x_buf[:T].copy_(x)
        x_op = x_buf[:T]
        if not _logged_x_buf_engaged[0]:
            _logged_x_buf_engaged[0] = True
            import logging
            logging.getLogger("vllm").info(
                "hpu-moe-ext: fused_moe ENGAGED persistent static-input x buffer "
                "(x_buf shape=%s, T=%d, K=%d) -> x routed through the held-leaf buffer "
                "(multi-bucket-capture-safe). This is the positive engagement signal.",
                tuple(x_buf.shape), int(T), int(K))
    elif _x_capture:
        # FALLBACK in the captured regime WITHOUT the persistent buffer (x_buf None, or T
        # over its capacity): this is the KNOWN multi-bucket-FATAL path — an in-forward
        # clone is an in-graph intermediate the free_inplace pass reclaims, so it does NOT
        # survive a 2nd decode bucket. In a correct serve the buffer is always allocated,
        # so reaching here is a regression; warn LOUDLY (once) so it is diagnosable rather
        # than re-crashing cryptically at the 2nd bucket. _MATERIALIZE_X=0 drops the clone.
        if not _warned_no_x_buf[0]:
            _warned_no_x_buf[0] = True
            import logging
            logging.getLogger("vllm").warning(
                "hpu-moe-ext: fused_moe captured (lazy) WITHOUT the persistent "
                "static-input x buffer (x_buf=%s, T=%d) -> falling back to the in-forward "
                "%s path, which is NOT multi-bucket-capture-safe (it re-crashes with "
                "ValidateSyncInputTensors 'tensor_data is empty' on the 2nd decode "
                "bucket). Ensure _alloc_fused_x_buf ran at weight-load and "
                "HPU_MOE_FUSED_X_BUFFER!=0.",
                None if x_buf is None else tuple(x_buf.shape), int(T),
                "clone" if _MATERIALIZE_X else "raw-view")
        # Distinguish a DELIBERATE disable (HPU_MOE_FUSED_X_BUFFER=0 -> the bisect toggle,
        # a LEGITIMATE choice -> intended=False, never strict-raises) from a REGRESSION
        # (the buffer feature is ON but the leaf is absent/too-small -> intended=True ->
        # HPU_ENGINE_STRICT hard-fails, because this captured path re-crashes at the 2nd
        # decode bucket and a benchmark must not silently ride it).
        _xbuf_disabled = os.environ.get("HPU_MOE_FUSED_X_BUFFER", "1") == "0"
        engage_fallback(
            "fused_expert_routed.fused_moe:x-buf-absent-under-capture",
            fast="persistent static-input x buffer (multi-bucket-capture-safe)",
            reason=("HPU_MOE_FUSED_X_BUFFER=0 (buffer deliberately disabled)"
                    if _xbuf_disabled else
                    "x_buf absent/too-small under capture (T=%d) despite the buffer "
                    "feature being on" % int(T)),
            slow="in-forward %s (NOT multi-bucket-capture-safe; 2nd-bucket FATAL)"
                 % ("clone" if _MATERIALIZE_X else "raw-view"),
            intended=(not _xbuf_disabled), strict_env="HPU_MOE_FUSED_X_BUFFER")
        x_op = x.contiguous().clone() if _MATERIALIZE_X else x
    else:
        x_op = x

    # *** MANDATORY mark_step ***: MATERIALIZE the routing (present/scatter/topk ->
    # `active`) BEFORE the data-dependent index_select. HPU-graph capture mishandles
    # index_select with un-materialized dynamic indices (reads stale/garbage `active`).
    # This is the ONE mark_step _routed_moe keeps before the gather; mirror it. It ALSO
    # flushes the x_op step above (the `x_buf[:T].copy_(x)` write into the persistent
    # static-input buffer, or the clone fallback) at this same boundary, keeping the
    # weight index_select + the op in one segment.
    htorch.mark_step()

    # gather the K routed experts' packed weights + scales + zero-points
    w13p = w13_packed.index_select(0, active)              # [K, H, 2I//8]
    w2p = w2_packed.index_select(0, active)                # [K, I, H//8]
    s13 = w13_scale.index_select(0, active)                # [K, H//GS, 2I]
    s2 = w2_scale.index_select(0, active)                  # [K, I//GS, H]
    z13 = w13_zp.index_select(0, active)                   # [K, H//GS, 2I//8]
    z2 = w2_zp.index_select(0, active)                     # [K, I//GS, H//8]

    # Per-slice views w13p[k]/s13[k]/... for k>0 carry a NONZERO storage_offset
    # (= k * stride[0] into the index_select buffer). The fused op's FIRST node on
    # every weight/zp input is a `reinterpret_cast` int32 -> uint4 (FCD x8) feeding a
    # NON-PERSISTENT, SRAM-resident dequant (moe_fused_expert_backend_shared.h) —
    # the most stride/offset-sensitive consumer in this codebase. A nonzero-offset
    # input makes that reinterpret read from the WRONG base (effectively expert
    # active[0]'s — or out-of-buffer — bytes), so a real expert dequants GARBAGE
    # (the serve's maxAbs~200 / relNorm~1e11) while a no-local-expert rank trivially
    # returns 0. This is the SAME invariant the monolithic op honours by .clone()-ing
    # every per-expert slice at load. Materialize a FRESH, CONTIGUOUS, offset-0 copy
    # of each per-expert slice in BOTH eager AND lazy: the K(<=8) small copies are
    # negligible at single-stream and remove the entire wrong-offset read class (the
    # earlier lazy-only clone left the offset live wherever a non-lazy/compile or a
    # capture edge fed the op an offset view). Force the op's exact dtypes too
    # (weights/zp int32, scales the bf16 the dequantize_4_bit guid is hardcoded to),
    # so a fp16-origin scale param can never be reinterpreted by the bf16 dequant.
    def _co(t):
        return t.contiguous().clone()
    def _cs(t):  # scales -> the op's bf16 dequant contract, fresh + contiguous
        return t.to(dt).contiguous().clone()

    # remap routed global->local id -> compact [0, K-1] so slot_k below indexes the
    # gathered (compact) experts. FUNCTIONAL scatter (lazy-replay safe). DEFAULT = the
    # SENTINEL compact id K (OUT of the [0,K-1] window the loop / one-hot iterate), NOT
    # 0. WHY THE SENTINEL (the bs>1 K-bucket safety, default-byte-identical):
    #   * When K >= #distinct-local-experts (the bs=1 production guarantee, and the
    #     conservative T>1 default K=min(E,T*top_k) below), EVERY routed-local id is in
    #     `active`, so `remap[id]` is its REAL compact id and the default K is NEVER read
    #     for a valid slot -> this is BYTE-IDENTICAL to the old `0` default (the default
    #     was only ever hit by non-local/non-routed slots, which `safe_w` already zeroes).
    #   * When the bs>1 K-BUCKET deliberately sizes K BELOW T*top_k (e.g. K=16 at bs=4,
    #     T*top_k=32, betting union<=16) and the union ever EXCEEDS K, `topk(present,K)`
    #     truncates: a routed-LOCAL expert is dropped from `active` and keeps the default.
    #     With the OLD `0` default that dropped expert's slot (valid=True, safe_w != 0)
    #     would fold its router weight onto compact-expert-0's output -> a SILENT
    #     WRONG-EXPERT FOLD. With the sentinel K its slot_k=K never matches any k in
    #     range(K) (loop) nor any column of arange(K) (one-hot) -> it is cleanly DROPPED
    #     (contributes 0), exactly as the stock op drops out-of-[0,K-1] ids. That turns a
    #     corrupting failure into a small, BOUNDED relNorm bump the HPU_MOE_ROUTED_VERIFY /
    #     HPU_MOE_FUSED_VERIFY oracle catches -- the K-bucket degrades SAFELY, never wrong.
    # (This mirrors _routed_moe's own "out-of-window ids are skipped" convention; we keep
    # the weight-zeroing belt on top via safe_w so a non-local slot is doubly safe.)
    remap = torch.full((E,), K, dtype=torch.int64, device=dev).scatter(
        0, active, torch.arange(K, dtype=torch.int64, device=dev))    # local id -> compact k (else K)
    # per-(token, slot) routing into compact-k space; non-local/non-routed/dropped -> the
    # sentinel K (excluded by the [0,K-1] iterate below). safe_w zeroes non-local slots too.
    safe_w = torch.where(valid.view_as(topk_weights), topk_weights,
                         torch.zeros_like(topk_weights)).to(dt)        # [T, top_k]
    slot_k = torch.where(valid, remap[flat_safe],
                         torch.full_like(flat, K)).view_as(topk_ids)   # [T, top_k] -> compact k (else K)

    # === CAPTURE-SAFE FAST PATH (lazy, ANY T): ONE single-recipe op call ===
    # Under wrap_in_hpu_graph the per-expert loop below clones each expert IN-GRAPH;
    # those clones are freed at capture-end (free_inplace, disable_tensor_cache=True) ->
    # the serve's ValidateSyncInputTensors FATAL on the MoE input at replay. The single-
    # recipe op avoids that entirely: it takes the K experts (as TensorLists for the
    # SEQUENTIAL op, or as STACKED [K,...] tensors for the BATCHED op) and slices/dequants/
    # sums ON-DEVICE inside ONE recipe whose inputs are the persistent stacked weights -- no
    # python intermediate to free.
    #
    # bs>1 (MTP-VERIFY) — WHY THIS NOW SERVES T>1 (the whole point of this fix):
    # the MTP verify is a CAPTURED decode batch (T = depth+1), so the in-graph-clone LOOP
    # below is FATAL there (freed intermediate at replay == the relNorm~1e11 / crash seen
    # when bs>1 routing was attempted with HPU_MOE_ROUTED_K=32). The SEQUENTIAL op slices on
    # device -> capture-safe at any T. The ONE thing that differs at T>1 is the router fold:
    # the op's per-expert final node is mult_fwd_bf16(y[M,H], rw) and the routed backend does
    # NOT constrain rw's shape (moe_fused_expert_routed_backend_shared.h: only checks the
    # LIST length == K; emit_one_expert just feeds rw to the broadcast mult). So we hand each
    # expert a [T,1] per-TOKEN weight column; Synapse mult_fwd broadcasts [T,1] over [T,H] ->
    # the correct per-(token,expert) weighting, folded ON-DEVICE inside the captured recipe.
    # At T==1 a [1,1] is the same fold as before -> BYTE-IDENTICAL to the historical path.
    # (The BATCHED op stays T==1: it HARD-checks router_weights==[K,1,1] and reduce-sums over
    # K with a per-expert scalar, so it cannot carry per-token weights -- forced off at T>1.)
    if _VIA_SINGLE_RECIPE and htorch.is_lazy():
        # per-(compact-k, token) router weight matrix wKT[K, T]: wKT[k, t] = sum of router
        # weights of every (slot) of token t that routed to compact expert k (0 where none).
        # Vectorized over K and T so NO python/host sync; one_hot[T,top_k,K] * safe_w then
        # sum over the top_k slot axis -> [T, K] -> transpose [K, T]. At T==1 the per-token
        # axis is length 1 so wKT[:, 0] == the old [K] per-expert fold (byte-identical feed).
        krange = torch.arange(K, device=dev)
        onehot = (slot_k.unsqueeze(-1) == krange).to(dt)               # [T, top_k, K]
        wKT = (onehot * safe_w.unsqueeze(-1)).sum(dim=1).transpose(0, 1).contiguous()  # [K, T]
        s13b = s13.to(dt); s2b = s2.to(dt)                            # op's bf16 dequant contract

        # --- BATCHED-K op (HPU_MOE_FUSED_BATCHED=1): the SINGLE-STREAM-SPEED path. T==1 ONLY
        # (it HARD-checks router_weights==[K,1,1] and reduce-sums a per-expert SCALAR over K,
        # so it structurally cannot carry the [K,T] per-token fold the verify needs). At T>1
        # we skip it and use the SEQUENTIAL op (which broadcasts [T,1] per expert). Feed the
        # index_select outputs DIRECTLY as STACKED [K,...] tensors (NO .unbind) + a [K,1,1]
        # router tensor. x_op is the SAME multi-bucket-safe x; the stacked weights/zp are the
        # same persistent index_select outputs -> capture stays coherent. If the batched .so
        # is absent it returns None -> fall through to the SEQUENTIAL op (graceful). ---
        if _VIA_BATCHED and T == 1:
            batched_op = _load_batched_op()
            if batched_op is not None:
                rw_stacked = wKT.view(K, 1, 1).to(dt)                 # [K,1,1] bf16 (T==1)
                out = batched_op(
                    x_op,
                    w13p, s13b, z13,
                    w2p, s2b, z2,
                    rw_stacked)                                        # [1, H] bf16
                return out

        # --- SEQUENTIAL single-recipe op (the validated default / fallback; ALSO the bs>1
        # verify path): K experts as TensorLists -- the index_select outputs, then ONE
        # offset-0 clone PER SLICE (the bs>1 _co fix). Pin scales to the op's bf16 dequant
        # contract; weights/zp stay int32. x_op is the MULTI-BUCKET-SAFE x. The router-weight
        # list is K x [T,1] (per expert, per-token column) -> the op's per-expert
        # mult_fwd_bf16(y[T,H], rw[T,1]) broadcasts per-token. At T==1 each [T,1] is [1,1] ->
        # the historical scalar fold, byte-identical. ---
        single_op = _load_single_recipe_op()
        if single_op is not None:
            # *** THE bs>1 OFFSET-0 FIX (HPU_MOE_ROUTED_BSGT1 regime; gated _SEQ_OFFSET_CLONE) ***
            # Each .unbind(0) element w13p[k] (k>0) is a VIEW with storage_offset = k*stride[0]
            # into the index_select buffer. The bridge carries that offset into the syn_in tensor
            # verbatim (create_tensor.cpp get_syn_offset = storage_offset * itemsize -> .set_offset),
            # but the op's FIRST node on every weight/zp input is a `reinterpret_cast` int32->uint4
            # (FCD x8) feeding a NON-persistent SRAM dequant (moe_fused_expert_emit.h L103-110) --
            # the most offset-sensitive consumer in the codebase. UNDER LAZY that reinterpret reads
            # from the WRONG base (effectively the index_select buffer's start == compact-expert-0's
            # bytes), so a real expert at k>0 dequants GARBAGE (the serve's maxAbs~200 / relNorm~1e11
            # at bs>1; a no-local-expert rank trivially returns 0, masking it on some ranks). This is
            # the EXACT class _alloc_fused_fp8_scale / _build_fp8_experts handle at LOAD time with the
            # same `t.clone() if lazy else t` helper ("convert_from_uint4 would read the WRONG expert
            # under lazy; .contiguous() does NOT reset storage_offset for a clean sub-block -> .clone()
            # forces a fresh offset-0 buffer"). The eager per-expert LOOP below already _co-clones every
            # slice (the proven relNorm~4e-3 path); this brings the SAME offset-0 guarantee to the
            # captured SEQUENTIAL op so bs>1 (the MTP-verify T=depth+1 and the tree-verify T=K batches)
            # is numerically correct, not 1e11.
            #
            # WHY .clone() AND NOT .contiguous() (load-time _co's hard lesson, L417-423): an unbind
            # slice is ALREADY a contiguous sub-block (clean strides), so .contiguous() is a no-op that
            # KEEPS the nonzero storage_offset -> the bug persists. Only .clone() (or
            # .contiguous().clone()) forces a brand-new offset-0 allocation. _co below is
            # .contiguous().clone() (offset-0 even if a future caller hands a strided slice).
            #
            # CAPTURE NOTE (multi-bucket; the on-card gate must confirm): a clone created INSIDE the
            # captured forward is an in-graph INTERMEDIATE (an output-not-input), which vllm-gaudi's
            # lazy free_inplace=True + disable_tensor_cache=True can reclaim ACROSS decode buckets --
            # the same class that forced the persistent x_buf (_alloc_fused_x_buf) and stacked-zp
            # (_alloc_fused_zp) leaves for x/zp. The .unbind views were capture-safe precisely because
            # they root at the persistent index_select sources (nn.Parameters); cloning trades that
            # persistence for offset-0. For the SINGLE-BUCKET MTP/tree verify (one decode bucket, T<=
            # _FUSED_X_BUF_T) this is correct as-is. If a multi-bucket verify ever re-hits the
            # 'ValidateSyncInputTensors tensor_data is empty' FATAL on a weight `hpu__input`, the
            # capture-safe follow-on is the C++ on-device slice (feed the STACKED [K,...] tensors +
            # synSliceParamsV2 per expert, as the BATCHED op already does -- one offset-0 persistent
            # input, the slice node honours the offset where a raw reinterpret does not) OR persistent
            # per-slot gathered-weight leaves. HPU_MOE_ROUTED_SEQ_OFFSET_CLONE=0 reverts to the raw
            # .unbind feed (bisect / capture-A-B only; reproduces the bs>1 1e11 under lazy).
            if _SEQ_OFFSET_CLONE:
                w13l = [_co(t) for t in w13p.unbind(0)]
                s13l = [_cs(t) for t in s13b.unbind(0)]
                z13l = [_co(t) for t in z13.unbind(0)]
                w2l = [_co(t) for t in w2p.unbind(0)]
                s2l = [_cs(t) for t in s2b.unbind(0)]
                z2l = [_co(t) for t in z2.unbind(0)]
            else:
                # raw offset-view feed (bisect): reproduces the bs>1 relNorm~1e11 under lazy.
                w13l = list(w13p.unbind(0)); s13l = list(s13b.unbind(0)); z13l = list(z13.unbind(0))
                w2l = list(w2p.unbind(0)); s2l = list(s2b.unbind(0)); z2l = list(z2.unbind(0))
            if not _logged_seq_offset_clone[0]:
                _logged_seq_offset_clone[0] = True
                import logging
                logging.getLogger("vllm").info(
                    "hpu-moe-ext: fused_moe SEQUENTIAL op engaged with offset-0 clone %s "
                    "(K=%d, T=%d) -- the bs>1 _co fix (HPU_MOE_ROUTED_SEQ_OFFSET_CLONE=%s). "
                    "%s This is the positive engagement signal for the bs>1 routed path.",
                    "ON" if _SEQ_OFFSET_CLONE else "OFF (raw .unbind, bisect)",
                    int(K), int(T), "1" if _SEQ_OFFSET_CLONE else "0",
                    "Per-expert weight/zp slices cloned to storage-offset 0."
                    if _SEQ_OFFSET_CLONE else
                    "RAW offset-view feed -> expect relNorm~1e11 under lazy at k>0.")
            # K x [T,1] bf16: expert k's per-token weight column, each a FRESH offset-0 contiguous
            # leaf (a nonzero storage_offset on the rw input would mis-read like the weight slices,
            # so use the SAME offset-0 _co clone, not a bare .contiguous() that keeps the offset).
            # The op's per-expert final node is mult_fwd_bf16(y[T,H], rw):
            #   * T==1: rw is [1,1] -> the PROVEN scalar broadcast (byte-identical to before).
            #   * T>1 : rw is [T,1] -> Synapse mult_fwd broadcasts the trailing dim [T,1] over
            #     [T,H] = per-token weighting. This trailing-only broadcast is a strict subset
            #     of the proven [1,1] (broadcast-both-dims) form, so it is expected to hold --
            #     *** the ONE on-card assumption to confirm: validate the bs>1 relNorm < 0.08
            #     (test_fused_moe_routed.py 4 6144 2048 128 32 16 8). If a Gaudi mult_fwd ever
            #     rejects/mis-broadcasts [T,1]x[T,H], the fallback is to pre-broadcast in Python
            #     (wKT[k].view(T,1).expand(T,H).contiguous()) -- a fresh [T,H] leaf, still
            #     capture-safe -- but [T,1] is preferred (no H-wide materialization). ***
            rws = [_co(w.reshape(T, 1)) for w in wKT.unbind(0)]           # K x [T,1] bf16
            out = single_op(
                x_op,
                w13l, s13l, z13l,
                w2l, s2l, z2l,
                rws)                                                    # [T, H] bf16
            return out

    out = torch.zeros(T, H, dtype=dt, device=dev)

    # === EXPERT COMPUTE: one fused-op call per gathered (compact) expert k ===
    # REACHED ONLY when the single-recipe op did NOT serve this call: i.e. EAGER (no
    # capture -> the in-graph .clone() is harmless; the offset-0 clone is the proven
    # real-weight correctness path, relNorm 4e-3), OR the single-recipe .so is absent
    # (degraded fallback). Under LAZY the captured path above ALWAYS returns first (T==1
    # AND T>1 now), so this in-graph-clone loop NEVER runs under decode capture in a
    # correct serve -- which is the whole reason the bs>1 verify is safe (the loop here is
    # the historical bs>1 FATAL; the sequential op above replaces it). The loop still
    # carries per-token weights correctly at any T (T==1 folds the [1,1] scalar in the op;
    # T>1 runs the op un-weighted and applies the [T,1] weight OUTSIDE), so eager bs>1 and
    # the .so-missing fallback stay numerically correct.
    for k in range(K):
        # per-(token, slot) router weight for THIS compact expert k: sum the router
        # weights of every top_k slot that routed to k (0 where none) -> [T, 1].
        # Identical to _routed_moe / fused_routed_moe's w_k.
        w_k = torch.where(slot_k == k, safe_w, torch.zeros_like(safe_w)).sum(
            dim=-1, keepdim=True).to(dt)                              # [T, 1]

        if T == 1:
            # T==1 PRIORITY: the whole [1,H] output scales by the SINGLE scalar
            # w_k[0,0]; pass it as the op's [1,1] router_weight so the op FOLDS it
            # (the validated single-expert path). Reshape (not .item(): keep it on
            # device / in-graph, no host sync). _co() -> storage_offset=0 [1,1].
            rw = _cs(w_k.view(1, 1))                                   # [1,1] bf16
            expert_out = torch.ops.hpu_moe_ext.fused_expert(
                x, _co(w13p[k]), _cs(s13[k]), _co(z13[k]),
                _co(w2p[k]), _cs(s2[k]), _co(z2[k]), rw)               # [1,H] (already weighted)
            out = out + expert_out
        else:
            # T>1: a [1,1] scalar can't carry a per-token weight. Run the op
            # UN-WEIGHTED (router_weight = 1) and apply the [T,1] w_k outside.
            # Numerically identical (the fold is just y*w), only moves the multiply
            # out of the captured recipe for the case the op can't fold.
            one = torch.ones((1, 1), dtype=dt, device=dev)
            expert_out = torch.ops.hpu_moe_ext.fused_expert(
                x, _co(w13p[k]), _cs(s13[k]), _co(z13[k]),
                _co(w2p[k]), _cs(s2[k]), _co(z2[k]), one)             # [T,H] un-weighted
            out = out + expert_out * w_k

    return out


# =============================================================================
# PERF FOLLOW-ON (TODO — scaffold only, NOT built here)
# =============================================================================
#
# `fused_moe_routed` above issues K separate `torch.ops.hpu_moe_ext.fused_expert`
# launches (one captured recipe EACH) plus the host-side accumulate. At T=1 the
# per-expert recipe is already the fusion + SRAM win (dequant->MME bundled,
# weights SRAM-resident; see moe_fused_expert_backend.cpp's docstring), but the K
# launches + K weighted-adds are K graph dispatches/replays per token. The
# follow-on collapses those K launches into ONE captured recipe.
#
# DELIVERABLE (a SIBLING OpBackend op `fused_moe`, NOT this Python wrapper):
#   torch.ops.hpu_moe_ext.fused_moe(
#       x[T,H] bf16,
#       w13_packed[K,H,2I//8] int32, w13_scale[K,H//GS,2I] bf16, w13_zp[K,H//GS,2I//8] int32,
#       w2_packed[K,I,H//8]  int32, w2_scale[K,I//GS,H]  bf16, w2_zp[K,I//GS,H//8]  int32,
#       router_weights[T,K] bf16) -> [T,H] bf16
#   == sum_k( fused_expert(x, W_k) * router_weights[:, k:k+1] ), as ONE recipe.
#   (The Python side still does the gather/topk/remap to produce the [K,...] stacked
#    slices + the per-(token,k) `router_weights` matrix — that routing is cheap and
#    HPU-graph-safe as _routed_moe proves; only the K expert subgraphs + weighted sum
#    move into the op.)
#
# HOW IT EXTENDS moe_fused_expert_backend.cpp (`MoeFusedExpert::AddNode`):
#   1. NEW OpBackend `MoeFusedMoE` (register a 2nd schema `hpu_moe_ext::fused_moe`
#      via a 2nd TORCH_LIBRARY_IMPL + KernelRegistry().REGISTER_HPU_BACKEND, same
#      3-part pattern as `fused_expert`). Its OutputMetaFn returns [T,H] bf16 (==
#      MoeFusedExpertMeta but T may be > the x.size(0) the single op assumes — read
#      it from x). The eager frontend `fused_moe_hpu` mirrors `fused_expert_hpu`.
#   2. In AddNode, FACTOR the current per-expert body of `MoeFusedExpert::AddNode`
#      (reinterpret_u4 -> dequant W13 -> gemm gu -> slice gate/up -> sigmoid/mult
#      SiLU -> dequant W2 -> gemm y) into a lambda `build_expert(k) -> y_k[T,H]`
#      that slices the k-th expert from the stacked inputs. The per-k weight inputs
#      come from a `slice` node on the [K,...] packed/scale/zp tensors along PyTorch
#      dim 0 (same synSliceParamsV2 idiom already used for the gate/up split, just
#      a different axis), each reinterpret_cast'd to uint4 as today. The two dequant
#      outputs stay NON-PERSISTENT per k so the GC bundles each into its MME exactly
#      as now (the SRAM win is preserved PER expert).
#   3. WEIGHTED SUM as an AddNode chain (the "one AddNode chain" fusing the K
#      launches): per k, slice `router_weights[:, k:k+1]` -> [T,1], `mult_fwd_bf16`
#      it onto y_k -> yw_k[T,H] (the same broadcast the single op's final node does,
#      but [T,1] over [T,H] instead of [1,1]). Then chain `add_fwd_bf16`:
#         acc_0 = yw_0 ;  acc_k = add_fwd_bf16(acc_{k-1}, yw_k)  for k in 1..K-1
#      All acc_* are graph-internal (non-persistent); only the final acc_{K-1} ties
#      to syn_out(0) (final_result_index=0). This is the single captured recipe that
#      replaces the K Python launches + K host-side `out = out + ...` adds.
#   4. K is a COMPILE-TIME-per-recipe constant (== the stacked dim-0 of the inputs),
#      so the bridge recipe-caches one recipe per (K, shapes) — fine: the routed
#      caller always uses the same static K (the gather width). No dynamic control
#      flow inside the graph (the routing/masking is done in Python before the op,
#      and zeroed router_weights make a non-routed expert contribute 0, exactly as
#      the single-op path folds a 0 weight today).
#
# WHY DEFER: this is a larger graph-construction effort (slice-along-dim0 of the
# stacked weights for K experts, the add-chain, a 2nd full registration triple),
# and the per-expert recipe in moe_fused_expert_backend.cpp is the correctness +
# SRAM-residency MVP this file builds on. The K-launch -> one-recipe fusion is a
# pure-latency optimization layered on a proven-correct base — exactly the order
# the backend's own closing comment ("add a sibling OpBackend `fused_moe` ... emits
# the gather + K expert-subgraphs + weighted sum as one AddNode chain") prescribes.
