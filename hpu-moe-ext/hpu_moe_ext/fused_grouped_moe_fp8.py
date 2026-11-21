"""GROUPED (routed top-k) fused M3 swigluoai-FP8 MoE for the CONCURRENCY regime.

THE WASTE THIS CUTS (the live M3 champion, minimax-m3-fp8-static):
  The champion serves the routed experts through `_fp8_route_b_allexpert`
  (m3_expert_runtime_patch.py): a Python loop over ALL E=16 LOCAL experts that
  dequants each fp8 weight -> bf16 and runs a bf16 matmul over ALL T tokens, then
  masks by the dense [T,E] router. At top_k=4 that is 4x more token-expert GEMMs
  than the routing needs, AND the MME never sees fp8 (bf16 matmul) -- the "8x
  excess GEMM" (4x all-16-vs-routed-4  x  2x bf16-vs-fp8). M3 is COMPUTE-bound at
  concurrency (bs=256 ~44% device util), so this excess is the dominant cost.

WHAT THIS DOES (the routed/grouped cut, ONE captured compute recipe):
  Permute the T tokens into per-expert capacity bins, run ONLY the routed
  token-expert pairs through the native fp8 MME, then scatter the weighted results
  back. Total expert GEMM work = E*Cap (Cap << T) instead of E*T -- the routing
  cut -- and the two matmuls run fp8 x fp8 -> bf16 on the MME (2x) instead of the
  route-b bf16 path.

  Result is BYTE-EQUIVALENT to `_fp8_route_b_allexpert` when no expert bin
  overflows (Cap >= max tokens-per-expert): the routed-4 contributions are exactly
  the all-16 contributions with the 12 non-routed experts contributing 0
  (the dense router weight is 0 there). See tests/test_grouped_fp8_cpu.py.

THE SPLIT (capture + risk discipline):
  * THE GEMM .so  (torch.ops.hpu_moe_ext.fused_moe_grouped_fp8) is a strict SUBSET
    of the PROVEN monolithic-fp8 emit (moe_fused_monolithic_fp8_emit.h, the
    MMEL_M3_FUSED +27% win): per-expert tiles xg[E,Cap,H] (no broadcast), output
    yg[E,Cap,H] (no router-mul, no reduce_sum). Same dynamic_quant / fp8_batch_gemm
    / swigluoai_tail nodes. Low novelty -> high on-card-correctness confidence.
  * THE PERMUTE / GATHER / SCATTER (the novel routing) live HERE, in torch, fully
    covered card-free by the CPU oracle. The plan build is STATIC-SHAPE (a T*E
    functional scatter -- no boolean-masked dynamic shapes), so it captures clean.

ENGAGE: hpu_m3_ext.m3_moe_fuse_patch (M3_FP8_STATIC=1 AND M3_MOE_FUSE=1) swaps the
  route-b `_m3_fp8_apply` to call grouped_moe_fp8(...). Strict no-op when unset.

M3 DIMS (TP8 EP): H=6144, I=3072, 2I=6144, E_local=16, top_k=4, alpha=1.702,
  limit=7.0. The whole fp8 expert set (~0.91 GB/card) is HBM-resident. RUN ON BOX.
"""
from __future__ import annotations

import os
import sys

import torch

try:
    import habana_frameworks.torch.core as htorch
except Exception:  # noqa: BLE001 -- card-free import (CPU oracle, selftest)
    htorch = None

_ALPHA_DEFAULT = 1.702
_LIMIT_DEFAULT = 7.0
_GLM_LIMIT_DEFAULT = 10.0

# Cap = per-expert capacity (static bin width). The grouped GEMM does E*Cap token-
# expert products. At c128/top_k=4/E=16 the mean count is 128*4/16=32; Cap must
# exceed the WORST expert's count to stay byte-exact (no drop). Default 2x the
# uniform mean + slack; tune per the on-card load histogram. M3_GROUPED_CAP overrides.
_DEFAULT_CAP = int(os.environ.get("M3_GROUPED_CAP", "0") or "0")  # 0 => caller/auto
_DEFAULT_GLM_CAP = int(os.environ.get("HPU_GLM53_GROUPED_CAP", "0") or "0")

_so_loaded = [False]

_ACTIVATION_SWIGLU_OAI = 0
_ACTIVATION_GLM_CLAMPED_SILU = 1
_WEIGHT_LAYOUT_E_H_2I = 0
_WEIGHT_LAYOUT_E_2I_H = 1


def _call_smart_m3(xg, w13_fp8, w13_scale, w2_fp8, w2_scale,
                   alpha, limit):
    """Compatibility adapter into the one canonical smart grouped op."""
    return torch.ops.hpu_moe_ext.fused_moe_grouped_fp8_smart(
        xg, w13_fp8, w13_scale, w2_fp8, w2_scale,
        _ACTIVATION_SWIGLU_OAI, _WEIGHT_LAYOUT_E_H_2I,
        float(alpha), float(limit),
    )


def _call_smart_glm(xg, w13_fp8, w13_scale, w2_fp8, w2_scale,
                    limit):
    """GLM descriptor adapter; no GLM-specific backend/kernel is selected."""
    return torch.ops.hpu_moe_ext.fused_moe_grouped_fp8_smart(
        xg, w13_fp8, w13_scale, w2_fp8, w2_scale,
        _ACTIVATION_GLM_CLAMPED_SILU, _WEIGHT_LAYOUT_E_2I_H,
        1.0, float(limit),
    )


def _load_grouped_fp8_op():
    """Load the grouped fp8 GEMM .so (lazy variant under lazy, eager else). Idempotent.
    Returns torch.ops.hpu_moe_ext.fused_moe_grouped_fp8 or None (caller falls back to
    the route-b loop). Co-loads via TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_fp8_smart"):
        return _call_smart_m3
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_fp8"):
        return torch.ops.hpu_moe_ext.fused_moe_grouped_fp8
    if _so_loaded[0]:
        return None
    _so_loaded[0] = True
    is_lazy = bool(htorch is not None and htorch.is_lazy())
    name = ("libmoe_fused_grouped_fp8_lazy.so" if is_lazy
            else "libmoe_fused_grouped_fp8.so")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("HPU_MOE_GROUPED_FP8_SO", ""),
        os.path.join(here, name),
        os.path.join(here, "..", "tpc_kernel", "opbackend", "build", name),
    ]
    so = next(
        (os.path.normpath(path) for path in candidates if path and os.path.isfile(path)),
        os.path.normpath(candidates[-1]),
    )
    # LOUD (stderr): which .so + the serve mode, so a lazy/eager mismatch is visible in the serve log.
    print(f"m3-grouped-load: is_lazy={is_lazy} -> {name} (exists={os.path.isfile(so)}) path={so}",
          file=sys.stderr, flush=True)
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: GROUPED fp8 op missing (%s); caller falls back to the "
            "route-b all-expert loop. Build it: bash tpc_kernel/opbackend/"
            "build_grouped_fp8.sh%s", so, " (BUILD_LAZY=1 for the lazy .so)")
        return None
    try:
        torch.ops.load_library(so)
    except Exception as e:  # noqa: BLE001
        # .so present but fails to dlopen (ABI / missing backend lib / not on the HPU image).
        # LOUD, then degrade to the route-b fallback rather than crash the serve (the grouped
        # path is by DESIGN optional -- m3_moe_fuse_patch delegates to route-b on None).
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: GROUPED fp8 op present (%s) but failed to load (%r); falling back to the "
            "route-b all-expert loop.", so, e)
        return None
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_fp8_smart"):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded canonical SMART grouped fp8 op from %s "
            "(model activation/layout selected by static descriptor)", so)
        return _call_smart_m3
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_fp8"):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded GROUPED fp8 op from %s (routed top-k -> native fp8 "
            "MME (2x); the concurrency-regime routed-cut over route-b all-16)", so)
        return torch.ops.hpu_moe_ext.fused_moe_grouped_fp8
    return None


def _load_grouped_glm_fp8_op():
    """Load the GLM exact-clamp specialization registered by the same grouped
    kernel library.  Loading the M3 symbol first keeps one dlopen/namespace and
    preserves the existing ABI for every other model."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_fp8_smart"):
        return _call_smart_glm
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_glm_fp8"):
        return torch.ops.hpu_moe_ext.fused_moe_grouped_glm_fp8
    _load_grouped_fp8_op()
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_grouped_fp8_smart"):
        return _call_smart_glm
    op = getattr(
        getattr(torch.ops, "hpu_moe_ext", None),
        "fused_moe_grouped_glm_fp8",
        None,
    )
    if op is None:
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: grouped .so lacks fused_moe_grouped_glm_fp8; "
            "GLM caller must use its exact dense rollback path.")
    return op


def _auto_cap(T: int, top_k: int, E: int) -> int:
    """A safe default capacity if the caller passed none: ceil(T*top_k/E) doubled
    (covers a 2x load imbalance) + a small floor; clamped to <= T (a single expert
    can never get more than T tokens). The byte-exact path wants Cap >= the worst
    expert's count; this is a heuristic -- the oracle/serve set it explicitly."""
    import math
    mean = math.ceil((T * top_k) / max(E, 1))
    return int(min(T, max(8, 2 * mean)))


def _auto_glm_cap(T: int, top_k: int, global_experts: int) -> int:
    """Capacity for GLOBAL routing over an EP-local expert window.

    The old grouped heuristic divided by ``E_local`` and therefore
    over-provisioned GLM by the EP world size (4x on TP4/EP4), erasing most of
    the routed-compute win.  GLM routes top-k over all 288 experts, so the
    expected local per-expert load is T*top_k/288.  Double that mean, retain a
    four-row floor for the small buckets, and clamp to T.

    This is a performance default, not a proof that an arbitrary learned
    routing distribution cannot overflow.  The GLM wrapper exposes a strict
    overflow guard for calibration/coherence gates; production promotion must
    use a capacity whose live histogram passes that guard."""
    import math
    mean = math.ceil((T * top_k) / max(global_experts, 1))
    return int(min(T, max(4, 2 * mean)))


def build_group_plan(topk_ids, topk_weights, E, experts_min, cap, device=None,
                     dtype=None, return_overflow=False):
    """STATIC-SHAPE counting-sort permute. Returns (gather_idx[E,cap] long,
    wg[E,cap] f32) where, for each LOCAL expert e and slot c<count(e):
        gather_idx[e,c] = a token id t routed to e ; wg[e,c] = route[t,e]
    and pad/overflow slots carry gather_idx=0, wg=0 (so they gather a finite row and
    scatter-add 0). `route` is the EXACT dense router matrix
    `_fp8_route_b_allexpert` builds (scatter_add of the masked top_k weights).

    CAPTURE-CLEAN: every op is static-shape (the scatter is over the full T*E pair
    set -- NO boolean-masked / data-dependent-size tensor). Overflow tokens (rank >=
    cap for a hot expert) route to a DUMP column (cap) that is sliced off -> dropped.
    Mirrors the FUNCTIONAL-scatter / graph-replay-safe idiom of fused_routed_moe.py."""
    dev = device if device is not None else topk_ids.device
    wdt = dtype if dtype is not None else (
        topk_weights.dtype if topk_weights.is_floating_point() else torch.float32)
    T, k = topk_ids.shape
    cap = int(cap)

    local = topk_ids.to(torch.int64) - int(experts_min)          # [T,k] local ids
    in_win = (local >= 0) & (local < E)                          # routed to a LOCAL expert?
    safe_local = torch.where(in_win, local, torch.zeros_like(local))
    safe_w = torch.where(in_win, topk_weights.to(torch.float32),
                         torch.zeros_like(topk_weights, dtype=torch.float32))

    # route[T,E] = dense per-(token,expert) weight (EXACT _fp8_route_b_allexpert build).
    route = torch.zeros(T, E, dtype=torch.float32, device=dev).scatter_add_(
        1, safe_local, safe_w)                                   # [T,E]
    # present[T,E] = did any NON-ZERO top-k slot route token t to expert e?
    #
    # Gaudi HPU graphs pad prompt/decode buckets with zero router weights while
    # leaving the padded expert ids at zero.  Counting ``in_win`` alone makes
    # those dummy rows consume expert-0 capacity and can evict a later real
    # expert-0 route.  Router probabilities are non-negative, so a non-zero
    # accumulated presence mask is the exact live-route contract while still
    # preserving arbitrarily small positive weights.
    slot_present = in_win & (safe_w != 0)
    present = torch.zeros(T, E, dtype=torch.float32, device=dev).scatter_add_(
        1, safe_local, slot_present.to(torch.float32)) > 0      # [T,E] bool

    # Rank within expert (0-based slot), via cumsum down the T axis.  Do NOT
    # run this prefix scan as int64 on Gaudi: the lazy int64 cumsum/reduction
    # path has produced dimensionally impossible ranks (for example load 52 at
    # T=16).  FP32 represents every supported token bucket exactly and maps to
    # the proven HPU prefix kernel; convert the final indices to int64 only for
    # scatter addressing.
    rank_f = present.to(torch.float32).cumsum(dim=0) - 1.0      # [T,E]
    in_cap = present & (rank_f < float(cap))                    # within capacity
    dump_f = torch.full_like(rank_f, float(cap))                # sliced-off column
    slot = torch.where(in_cap, rank_f, dump_f).to(torch.int64) # absent/overflow -> cap

    e_idx = torch.arange(E, device=dev).view(1, E).expand(T, E)   # [T,E]
    t_idx = torch.arange(T, device=dev).view(T, 1).expand(T, E)   # [T,E]
    dest = (e_idx * (cap + 1) + slot).reshape(-1)                # [T*E] flat dest in [0,E*(cap+1))

    # FUNCTIONAL scatter into a flat [E*(cap+1)] buffer (out-of-place -> replay-safe).
    # Valid columns (slot<cap, present) have a UNIQUE writer (rank is unique within e);
    # the dump column (cap) collides but is sliced away.
    gi_flat = torch.zeros(E * (cap + 1), dtype=torch.int64, device=dev).scatter(
        0, dest, t_idx.reshape(-1))
    routev = torch.where(present, route, torch.zeros_like(route)).reshape(-1)   # [T*E]
    gw_flat = torch.zeros(E * (cap + 1), dtype=torch.float32, device=dev).scatter(
        0, dest, routev)

    gather_idx = gi_flat.view(E, cap + 1)[:, :cap].contiguous()   # [E,cap] long
    wg = gw_flat.view(E, cap + 1)[:, :cap].contiguous().to(wdt)   # [E,cap]
    if return_overflow:
        # Audit the plan that will actually be consumed, rather than deriving
        # overflow a second time from ``rank``.  This catches both a genuinely
        # hot expert and any backend-specific cumsum/scatter mismatch.  Routed
        # softmax weights are strictly positive, while every padding slot is
        # exactly zero, so non-zero grouped weights are the kept-route census.
        expected_routes = present.to(torch.float32).sum()
        kept_routes = (wg != 0).to(torch.float32).sum()
        overflow = expected_routes - kept_routes
        max_load = present.to(torch.float32).sum(dim=0).max()
        return gather_idx, wg, overflow, max_load
    return gather_idx, wg


def grouped_moe_fp8(x, w13_fp8, w13_scale, w2_fp8, w2_scale,
                    topk_ids, topk_weights, experts_min, top_k,
                    alpha=None, limit=None, cap=None, gemm_fn=None,
                    materialize=None):
    """ROUTED/grouped M3 swigluoai-fp8 MoE. Byte-equivalent to
    `_fp8_route_b_allexpert` (within fp8 grid noise) when no bin overflows.

    Args (the layer's persistent route-b fp8 stacks; PyTorch row-major, on HPU):
      x          : [T, H] bf16 hidden states (concurrency regime, T>1)
      w13_fp8    : [E, H, 2I] float8_e4m3fn  (op._w13_fp8_stack)
      w13_scale  : [E, 2I]    f32 per-output-channel forward scale (op._s13_stack)
      w2_fp8     : [E, I, H]  float8_e4m3fn  (op._w2_fp8_stack)
      w2_scale   : [E, H]     f32 per-output-channel forward scale (op._s2_stack)
      topk_ids   : [T, top_k] int -- GLOBAL expert ids (this rank's window @ experts_min)
      topk_weights : [T, top_k] -- router weights (already sigmoid-scored/renormalized)
      experts_min  : this rank's first GLOBAL expert id == layer._hpu_experts_min
      top_k      : routing fan-out (M3: 4); only used for the auto-cap heuristic
      alpha/limit: swigluoai params (default 1.702 / 7.0)
      cap        : per-expert bin width (static). None -> _auto_cap. The byte-exact
                   path needs cap >= the worst expert's token count.
      gemm_fn    : OPTIONAL callable(xg, w13_fp8, w13_scale, w2_fp8, w2_scale, alpha,
                   limit) -> yg[E,cap,H]. None -> the .so. The CPU oracle injects a
                   reference fp8-MME emulation here (so the wrapper's permute/gather/
                   scatter is exercised card-free).
      materialize: insert the routing-materialization mark_step before the dynamic
                   gather (the fused_routed_moe hazard: index_select with an
                   un-realized dynamic index reads stale bytes under lazy capture).
                   None -> auto (on under lazy). The COMPUTE recipe (gather->gemm->
                   scatter) is the single captured region after this one fence; this
                   does NOT add a fence vs route-b (which also materializes routing).

    Returns: [T, H] bf16 routed output, or None if the .so is absent and no gemm_fn
    was provided (caller falls back to the route-b loop)."""
    E = int(w13_fp8.shape[0])
    H = int(x.shape[-1])
    T = int(x.shape[0])
    emin = int(experts_min)
    a = _ALPHA_DEFAULT if alpha is None else float(alpha)
    lim = _LIMIT_DEFAULT if limit is None else float(limit)
    if cap is None:
        # cap AUTO-SCALES with T (ceil(T*top_k/E)*2, clamped to T) -> drop-safe for M3's near-uniform
        # router at ANY T (decode AND prefill). M3_GROUPED_CAP, if set, is an optional FLOOR (a minimum
        # bin width), NOT a fixed cap -> it can never UNDER-provision a large-T call (the prefill-drop
        # footgun the fixed default had). The oracle passes an explicit cap for the byte-exact test.
        cap = _auto_cap(T, int(top_k), E)
        if _DEFAULT_CAP > 0:
            cap = max(cap, _DEFAULT_CAP)
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"grouped_moe_fp8: cap must be >=1, got {cap}")

    if gemm_fn is None:
        gemm_fn = _load_grouped_fp8_op()
        if gemm_fn is None:
            return None  # .so absent -> caller falls back to the per-expert route-b loop

    # ---- static-shape counting-sort permute plan ----
    gather_idx, wg = build_group_plan(
        topk_ids, topk_weights, E, emin, cap, device=x.device, dtype=x.dtype)

    # ---- materialize the routing BEFORE the dynamic gather (lazy capture hazard) ----
    do_ms = (htorch is not None and htorch.is_lazy()) if materialize is None else bool(materialize)
    if do_ms and htorch is not None:
        htorch.mark_step()

    # ---- gather per-expert token tiles xg[E,cap,H] (dynamic index, STATIC shape) ----
    flat_idx = gather_idx.reshape(-1)                                  # [E*cap]
    xg = x.index_select(0, flat_idx).view(E, cap, H)                  # [E,cap,H] bf16

    # ---- the GROUPED fp8 GEMM (.so subset of the proven monolithic emit) ----
    yg = gemm_fn(xg, w13_fp8, w13_scale, w2_fp8, w2_scale, float(a), float(lim))  # [E,cap,H]
    if yg is None:
        return None

    # ---- weight by the per-(expert,slot) router weight, scatter-add back to [T,H] ----
    yw = (yg * wg.view(E, cap, 1).to(yg.dtype)).reshape(E * cap, H)   # [E*cap,H]
    out = torch.zeros(T, H, dtype=yg.dtype, device=x.device).index_add(
        0, flat_idx, yw)                                              # [T,H]
    return out


def grouped_moe_glm_fp8(
        x, w13_fp8, w13_scale, w2_fp8, w2_scale,
        topk_ids, topk_weights, experts_min, top_k, global_experts,
        limit=None, cap=None, gemm_fn=None, materialize=None,
        verify_no_overflow=None):
    """Exact GLM-5.3 routed/clamped FP8 MoE using the existing grouped engine.

    Unlike :func:`grouped_moe_fp8`, this specialization consumes checkpoint-
    native GLM weights without a persistent transpose:
      w13_fp8 [E,2I,H], w13_scale [E,2I]
      w2_fp8  [E,H,I],  w2_scale  [E,H]
    The backend executes both FP8 GEMMs with ``transpose_b=true`` and applies
    exactly ``silu(clamp(gate,max=L))*clamp(up,-L,L)``.

    ``verify_no_overflow`` is intentionally strict and synchronizing.  Enable
    it for capacity calibration and coherence tests; any dropped route raises
    instead of silently degrading the model.  It defaults from
    HPU_GLM53_GROUPED_VERIFY_OVERFLOW.
    """
    E = int(w13_fp8.shape[0])
    H = int(x.shape[-1])
    T = int(x.shape[0])
    emin = int(experts_min)
    lim = _GLM_LIMIT_DEFAULT if limit is None else float(limit)
    if cap is None:
        cap = _auto_glm_cap(T, int(top_k), int(global_experts))
        if _DEFAULT_GLM_CAP > 0:
            cap = max(cap, _DEFAULT_GLM_CAP)
    cap = int(min(T, cap))
    if cap < 1:
        raise ValueError(f"grouped_moe_glm_fp8: cap must be >=1, got {cap}")

    if gemm_fn is None:
        gemm_fn = _load_grouped_glm_fp8_op()
        if gemm_fn is None:
            return None

    if verify_no_overflow is None:
        verify_no_overflow = os.environ.get(
            "HPU_GLM53_GROUPED_VERIFY_OVERFLOW", "0") == "1"

    gather_idx, wg = build_group_plan(
        topk_ids,
        topk_weights,
        E,
        emin,
        cap,
        device=x.device,
        dtype=x.dtype,
        return_overflow=False,
    )

    do_ms = ((htorch is not None and htorch.is_lazy())
             if materialize is None else bool(materialize))
    if verify_no_overflow:
        # In lazy mode an HPU scalar read without an explicit step can observe
        # an unmaterialized/stale scalar.  That produced the impossible prior
        # diagnostic ``max_load < cap`` with non-zero overflow.  Materialize
        # the complete plan first, then inspect both audit scalars.
        if do_ms and htorch is not None:
            htorch.mark_step()
        # mark_step submits the lazy graph but does not guarantee completion.
        # The diagnostic copies below must observe completed route-plan buffers;
        # otherwise stale HPU storage can look like impossible expert loads or
        # an all-nonzero capacity matrix even though the live plan is correct.
        if x.device.type == "hpu" and hasattr(torch, "hpu"):
            torch.hpu.synchronize()
        # Audit with CPU truth, not HPU scalar reductions.  This is deliberately
        # diagnostic-only: copy the input route table and the *materialized plan
        # actually consumed by the kernel*, then prove its kept-route census.
        # It catches true capacity pressure as well as a bad HPU prefix/scatter.
        ids_cpu = topk_ids.detach().cpu().to(torch.int64)
        wg_cpu = wg.detach().cpu()
        weights_cpu = topk_weights.detach().cpu().to(torch.float32)
        valid_cpu = (
            (ids_cpu >= emin)
            & (ids_cpu < emin + E)
            & (weights_cpu != 0)
        )
        local_slots_cpu = ids_cpu - emin
        safe_local_cpu = torch.where(
            valid_cpu, local_slots_cpu, torch.zeros_like(local_slots_cpu))
        # Count unique (token,expert) pairs, matching scatter_add routing when a
        # synthetic test repeats an expert in multiple top-k slots.
        present_cpu = torch.zeros(
            ids_cpu.shape[0], E, dtype=torch.int32).scatter_add_(
                1, safe_local_cpu, valid_cpu.to(torch.int32)) > 0
        counts_cpu = present_cpu.to(torch.int64).sum(dim=0)
        if counts_cpu.numel():
            observed_max = int(counts_cpu.max().item())
            expected_routes = int(counts_cpu.sum().item())
            expected_kept = int(counts_cpu.clamp(max=cap).sum().item())
        else:
            observed_max = 0
            expected_routes = 0
            expected_kept = 0
        actual_kept = int((wg_cpu != 0).sum().item())
        n_overflow = expected_routes - expected_kept
        if actual_kept != expected_kept:
            id_min = int(ids_cpu.min().item()) if ids_cpu.numel() else 0
            id_max = int(ids_cpu.max().item()) if ids_cpu.numel() else 0
            raise RuntimeError(
                "grouped_moe_glm_fp8 route-plan mismatch: "
                f"T={T} E_local={E} expert_window=[{emin},{emin + E}) "
                f"cap={cap} max_load={observed_max} expected_kept={expected_kept} "
                f"actual_kept={actual_kept} id_range=[{id_min},{id_max}]")
        if n_overflow:
            raise RuntimeError(
                "grouped_moe_glm_fp8 capacity overflow: "
                f"T={T} E_local={E} cap={cap} max_load={observed_max} "
                f"dropped_routes={n_overflow}; raise HPU_GLM53_GROUPED_CAP")
        import logging
        logging.getLogger("vllm").warning(
            "GLM53_GROUPED_CAPACITY_OK T=%d E_local=%d expert_window=[%d,%d) "
            "cap=%d max_load=%d local_routes=%d id_range=[%d,%d]",
            T, E, emin, emin + E, cap, observed_max, expected_routes,
            int(ids_cpu.min().item()), int(ids_cpu.max().item()))

    if do_ms and htorch is not None and not verify_no_overflow:
        htorch.mark_step()

    flat_idx = gather_idx.reshape(-1)
    xg = x.index_select(0, flat_idx).view(E, cap, H)
    yg = gemm_fn(
        xg,
        w13_fp8,
        w13_scale.reshape(E, -1),
        w2_fp8,
        w2_scale.reshape(E, -1),
        float(lim),
    )
    if yg is None:
        return None

    yw = (yg * wg.view(E, cap, 1).to(yg.dtype)).reshape(E * cap, H)
    return torch.zeros(T, H, dtype=yg.dtype, device=x.device).index_add(
        0, flat_idx, yw)
