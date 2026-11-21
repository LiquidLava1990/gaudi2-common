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
# ---------------------------------------------------------------------------
_VIA_SINGLE_RECIPE = os.environ.get("HPU_MOE_ROUTED_VIA_SINGLE_RECIPE", "1") != "0"

_routed_so_loaded = [False]


def _load_single_recipe_op():
    """Load the K-expert single-recipe op .so (lazy variant under lazy, eager else)
    and return torch.ops.hpu_moe_ext.fused_moe, or None if unavailable (caller then
    falls back to the per-expert loop). Idempotent. The .so co-loads with the
    single-expert .so via TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe"):
        return torch.ops.hpu_moe_ext.fused_moe
    if _routed_so_loaded[0]:
        return None
    _routed_so_loaded[0] = True
    name = ("libmoe_fused_routed_lazy.so" if htorch.is_lazy()
            else "libmoe_fused_routed.so")
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: single-recipe fused_moe op missing (%s); routed fused "
            "path falls back to the per-expert loop (NOT capture-safe under lazy). "
            "Build it: bash tpc_kernel/opbackend/build_routed%s.sh", so,
            "_lazy" if htorch.is_lazy() else "")
        return None
    torch.ops.load_library(so)
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe"):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded single-recipe fused_moe op from %s "
            "(capture-safe routed path)", so)
        return torch.ops.hpu_moe_ext.fused_moe
    return None


def fused_moe_routed(x, w13_packed, w13_scale, w13_zp,
                     w2_packed, w2_scale, w2_zp,
                     topk_ids, topk_weights, experts_min, K):
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
      K            : static gather width (>= #distinct local experts; caller guarantees T*top_k <= K)

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

    # *** MANDATORY mark_step ***: MATERIALIZE the routing (present/scatter/topk ->
    # `active`) BEFORE the data-dependent index_select. HPU-graph capture mishandles
    # index_select with un-materialized dynamic indices (reads stale/garbage `active`).
    # This is the ONE mark_step _routed_moe keeps before the gather; mirror it.
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
    # gathered (compact) experts. FUNCTIONAL scatter (lazy-replay safe).
    remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
        0, active, torch.arange(K, dtype=torch.int64, device=dev))    # local id -> compact k
    # per-(token, slot) routing into compact-k space, non-local/non-routed -> 0 (weight
    # zeroed below so it contributes nothing). safe_w zeroes non-local slots.
    safe_w = torch.where(valid.view_as(topk_weights), topk_weights,
                         torch.zeros_like(topk_weights)).to(dt)        # [T, top_k]
    slot_k = torch.where(valid, remap[flat_safe],
                         torch.zeros_like(flat)).view_as(topk_ids)     # [T, top_k] -> compact k

    # === CAPTURE-SAFE FAST PATH (lazy + T==1): ONE single-recipe op call ===
    # Under wrap_in_hpu_graph the per-expert loop below clones each expert IN-GRAPH;
    # those clones are freed at capture-end (free_inplace) -> the serve's
    # ValidateSyncInputTensors FATAL on the MoE input at replay. The single-recipe op
    # avoids that entirely: it takes the K experts as TensorLists (`.unbind(0)` views
    # of the persistent index_select outputs) and slices/dequants/sums ON-DEVICE
    # inside ONE recipe whose inputs are the persistent stacked weights — no python
    # intermediate to free. It folds a per-expert [1,1] router weight (== T==1).
    if _VIA_SINGLE_RECIPE and T == 1 and htorch.is_lazy():
        single_op = _load_single_recipe_op()
        if single_op is not None:
            # per-compact-k [1,1] router weight: w_k[k] = sum of router weights of
            # every (token,slot) that routed to compact expert k (0 where none).
            # Vectorized over K so no python sync; one_hot[T,top_k,K] * safe_w summed.
            krange = torch.arange(K, device=dev)
            onehot = (slot_k.unsqueeze(-1) == krange).to(dt)            # [T, top_k, K]
            wK = (onehot * safe_w.unsqueeze(-1)).sum(dim=(0, 1))        # [K] (T==1)
            rws = list(wK.view(K, 1, 1).to(dt).unbind(0))              # K x [1,1] bf16
            # K experts as TensorLists — unbind the index_select outputs (the op slices
            # on-device; the backend header prescribes exactly this feed). Pin scales
            # to the op's bf16 dequant contract; weights/zp stay int32.
            s13b = s13.to(dt); s2b = s2.to(dt)
            out = single_op(
                x,
                list(w13p.unbind(0)), list(s13b.unbind(0)), list(z13.unbind(0)),
                list(w2p.unbind(0)), list(s2b.unbind(0)), list(z2.unbind(0)),
                rws)                                                    # [1, H] bf16
            return out

    out = torch.zeros(T, H, dtype=dt, device=dev)

    # === EXPERT COMPUTE: one fused-op call per gathered (compact) expert k ===
    # (eager, or T>1 prefill, or single-recipe op unavailable — NOT under decode
    # capture in the supported config, so the in-graph clone is safe here.)
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
