"""binned_moe.py -- WAVE-2 capacity-binned ("permuted grouped-GEMM") MoE wrapper.

The SPARSE-compute sibling of fused_monolithic_moe.fused_moe_monolithic. The dense
monolithic computes EVERY local expert over EVERY token (dense E*T; at the live EP
TP=8 config the [E,T] router zeroes ~97% of the products), then reduces over E. This
wrapper instead:

  1. BINS tokens by routed expert into a STATIC-capacity permuted activation
     x_perm[E,C,H] -- each local expert e owns ONLY its <=C routed-token rows. The bin
     is an on-device, static-shape, no-host-sync scatter: a cumsum over the one_hot of
     each pick's expert gives its slot WITHIN that expert; overflow (slot>=C) or
     non-local picks route to a SENTINEL bin (dropped, never wrong-folded -- the
     KBUCKET safe-degrade contract).
  2. runs the W4A8 int4->fp8 grouped GEMM over E*C rows (not E*T) via the new
     torch.ops.hpu_moe_ext.fused_moe_binned_int4fp8 op, one CHUNK-expert group per
     call, torch.cat -> y_perm[E,C,H] (router weight folded INSIDE the op).
  3. UN-BINS: each pick gathers its (expert,slot) output; sum over the token's top_k.

WHY THIS WINS (honest -- it is a COMPUTE + working-set lever, NOT a read lever):
  - union of distinct local experts @c128 ~= E (all 32) -> all still dequanted once
    per STEP (read already amortized; BATCHED_MOE_FIX.md). Binning does NOT cut reads.
  - it cuts the per-expert TOKEN count T -> C ~= per-expert load (~4 @c128) -> GEMM
    MACs E*T -> E*C and the bf16 transient (gu/y scale with C not T). Oracle
    (tests/binned_moe_oracle.py): ~4-10x GEMM cut @c128 IF routing balanced, x W4A8's
    2x MME. Also un-OOMs the prefill warmup the dense monolithic hits
    (moe_wna16_hpu.py:285-300) since C << T_prefill.
  - defeats the "K->E=32 by bs16" clamp on the T-axis (tokens-per-expert), not the
    E-axis union (=32, conceded).

BYTE-EXACTNESS: bit-faithful to the dense monolithic when C >= the batch's max
per-expert local load (oracle V1 relNorm 0.0). Undersized C drops only the overflow
picks (the lowest-traffic tail) to the sentinel -> contribute EXACTLY 0, never
wrong-fold (oracle V3/V4). HPU_MOE_BINNED_CAP sets C; HPU_MOE_BINNED_DEBUG=1 logs a
one-shot per-rank max-load + overflow count so the operator CALIBRATES C against live
routing (no real routing dump exists yet -- EXPERT_LOCALITY.md).

GPTQ-symmetric int4 only (stacked zp); silu only; lazy serving path. STRICT no-op
unless the wiring flag (HPU_MOE_BINNED=1) is set -- see binned_moe_patch.py. The .so is
COMPILE-validated card-free; the on-device byte-exact gate is the on-card A/B
(HPU_MOE_BINNED_VERIFY=1 vs the dense monolithic, relNorm<1e-2).

GLM-5.2-int4 shapes: H=6144, I=2048, E_local=32, top_k=8, GS=128, GPTQ-sym zp=8.
"""
from __future__ import annotations

import os

import torch

# C (capacity): max routed tokens per local expert per step. Sized to the per-rank
# load (~T*top_k*E_local/E_global). Default 32 (~lossless if routing is balanced/mild;
# oracle: 100%/98% of c128 batches lossless at C=32 for balanced/mild). Larger C ->
# safer (fewer drops) but less GEMM cut (T/C). HPU_MOE_BINNED_CAP overrides.
_CAP = int(os.environ.get("HPU_MOE_BINNED_CAP", "32") or "32")
# CHUNK: experts per grouped-GEMM op call (dequant SRAM-residency knob, == the
# monolithic's chunk). Default 8 (the proven batched tile). E/CHUNK op calls/layer.
_CHUNK = int(os.environ.get("HPU_MOE_BINNED_CHUNK", "8") or "8")
_DEBUG = os.environ.get("HPU_MOE_BINNED_DEBUG") == "1"
# Drop the two materializing mark_steps (scatter|op and op|unbin) to let the GC fuse
# the whole path into fewer recipes. Default KEEP them (correctness-first; they mirror
# _routed_moe's mandatory pre-gather boundary). Fusion is a perf refinement -- validate
# byte-exact before trusting HPU_MOE_BINNED_FUSE=1.
_FUSE = os.environ.get("HPU_MOE_BINNED_FUSE") == "1"

_so_loaded = [False]
_dbg_emitted = [False]


def _load_binned_op():
    """Load libmoe_fused_moe_binned_int4fp8_lazy.so -> the grouped-GEMM op, or None."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_binned_int4fp8"):
        return torch.ops.hpu_moe_ext.fused_moe_binned_int4fp8
    if _so_loaded[0]:
        return None
    _so_loaded[0] = True
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build",
        "libmoe_fused_moe_binned_int4fp8_lazy.so"))
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: BINNED op .so missing (%s) -- caller falls back to the dense "
            "monolithic / stock op.", so)
        return None
    torch.ops.load_library(so)
    op = getattr(torch.ops.hpu_moe_ext, "fused_moe_binned_int4fp8", None)
    if op is not None:
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: loaded BINNED W4A8 grouped-GEMM op from %s (capacity-permuted "
            "E*C compute; %s)", so, "int4->fp8 in-SRAM dequant + native fp8 MME")
    return op


# =============================================================================
# bin_scatter / unbin -- PURE TORCH (no habana import) so the indexing math is
# CPU-testable card-free against tests/binned_moe_oracle.py. Static shape, no host
# sync; functional scatters re-materialize each decode step (HPU-graph-replay safe).
# =============================================================================
def bin_scatter(x, topk_ids, topk_weights, emin, E, C):
    """Return (x_perm[E,C,H] bf16, w_perm[E,C,1] dt, bin_idx[P] int64, n_overflow[]).

    x[T,H]; topk_ids[T,K] GLOBAL ids; topk_weights[T,K]. Each pick (token t, slot s) ->
    local expert e=ids-emin; its slot within e is the running count (cumsum); pick maps
    to bin e*C+slot, or the SENTINEL bin E*C if non-local or slot>=C (dropped)."""
    dev = x.device
    H = x.shape[-1]
    T, K = topk_ids.shape
    P = T * K
    dt = topk_weights.dtype

    local = topk_ids.to(torch.int64) - int(emin)          # [T,K]
    flat_e = local.reshape(-1)                             # [P]
    flat_v = (flat_e >= 0) & (flat_e < E)                  # local?
    flat_e_safe = torch.where(flat_v, flat_e, torch.zeros_like(flat_e))
    flat_w = topk_weights.reshape(-1)                      # [P]
    tok_of_pick = torch.arange(T, device=dev).repeat_interleave(K)  # [P] pick -> token

    # slot within expert via inclusive cumsum of the (valid) one-hot, take own column.
    erange = torch.arange(E, device=dev)
    onehot = ((flat_e_safe.unsqueeze(1) == erange.unsqueeze(0)) &
              flat_v.unsqueeze(1)).to(torch.int32)         # [P,E]
    csum = onehot.cumsum(0)                                # inclusive
    slot = ((csum - 1) * onehot).sum(dim=1)               # [P] 0-based slot per expert
    overflow = (~flat_v) | (slot >= C)                     # routes to sentinel
    sentinel = E * C
    bin_idx = torch.where(overflow, torch.full_like(flat_e_safe, sentinel),
                          flat_e_safe * C + slot)          # [P]
    # CAPACITY overflow (the only LOSSY drops; non-local is correct to drop).
    n_overflow = (flat_v & (slot >= C)).sum()

    # functional scatter into a fresh [E*C+1, H] (sentinel row = overflow sink, dropped).
    base_x = x.new_zeros(sentinel + 1, H)
    x_perm = base_x.index_add(0, bin_idx, x.index_select(0, tok_of_pick))
    base_w = topk_weights.new_zeros(sentinel + 1)
    w_pick = torch.where(overflow, torch.zeros_like(flat_w), flat_w)
    w_perm = base_w.index_add(0, bin_idx, w_pick)

    x_perm = x_perm[:sentinel].reshape(E, C, H)
    w_perm = w_perm[:sentinel].reshape(E, C, 1).to(dt)
    return x_perm, w_perm, bin_idx, n_overflow


def unbin(y_perm, bin_idx, T, K):
    """y_perm[E,C,H] (router-weighted) -> out[T,H]: gather each pick's (e,slot) row
    (sentinel -> 0) and sum over the token's top_k picks."""
    E, C, H = y_perm.shape
    y_flat = torch.cat([y_perm.reshape(E * C, H),
                        y_perm.new_zeros(1, H)], dim=0)     # sentinel row 0
    picked = y_flat.index_select(0, bin_idx)               # [P,H]
    return picked.reshape(T, K, H).sum(dim=1)              # [T,H]


def reference_group_gemm(x_perm, W13_deq, W2_deq, w_perm, chunk_start, G,
                         intermediate_size, activation="silu"):
    """CPU/verify reference for ONE expert group (float, no .so): mirrors the op's
    math (gu=x@W13 -> silu(gate)*up -> y=h@W2 -> *router_w). W13_deq[E,H,2I],
    W2_deq[E,I,H] are ALREADY-dequanted (the int4->fp8 dequant is the op's job and is
    shared/cancels in the dense-vs-binned comparison)."""
    import torch.nn.functional as F
    I = intermediate_size
    xg = x_perm[chunk_start:chunk_start + G]                # [G,C,H]
    W13 = W13_deq[chunk_start:chunk_start + G]              # [G,H,2I]
    W2 = W2_deq[chunk_start:chunk_start + G]                # [G,I,H]
    rw = w_perm[chunk_start:chunk_start + G]                # [G,C,1]
    gu = torch.bmm(xg, W13)                                 # [G,C,2I]
    gate, up = gu[:, :, :I], gu[:, :, I:]
    h = F.silu(gate) * up                                   # [G,C,I]
    y = torch.bmm(h, W2)                                    # [G,C,H]
    return y * rw


def fused_moe_binned(x, topk_ids, topk_weights,
                     w13_packed, w13_scale, w13_zp,
                     w2_packed, w2_scale, w2_zp,
                     experts_min, num_local_experts,
                     C=None, chunk=None, group_gemm_fn=None):
    """Capacity-binned W4A8 MoE. Returns [T,H] bf16, or None if ineligible/op missing.

    w*_zp are the STACKED per-expert [E,...] zero-points (the held leaves the monolithic
    path builds via _alloc_fused_zp). group_gemm_fn overrides the .so op (verify/CPU)."""
    if w13_zp is None:
        return None                                        # AWQ per-expert zp -> ineligible
    E = int(num_local_experts)
    Ccap = int(_CAP if C is None else C)
    ch = int(_CHUNK if chunk is None else chunk)
    T, K = topk_ids.shape
    # A single expert holds at most T tokens (each token contributes <=1 pick per
    # expert), so C>T is pure waste; clamp. At T<=C the bin is dense-equivalent
    # (C=T -> no overflow, no GEMM cut) -> binning gracefully degrades to the dense
    # work at low concurrency and only pays off when T>C (the high-conc target).
    Ccap = max(1, min(Ccap, T))

    op = group_gemm_fn if group_gemm_fn is not None else _load_binned_op()
    if op is None:
        return None

    x_perm, w_perm, bin_idx, n_overflow = bin_scatter(
        x, topk_ids, topk_weights, experts_min, E, Ccap)

    if not _FUSE:
        _mark_step()                                       # materialize scatter before op + gather

    if _DEBUG and not _dbg_emitted[0]:
        _emit_overflow_debug(topk_ids, experts_min, E, Ccap, n_overflow, T, K)

    # grouped GEMM, one CHUNK-expert group per call, torch.cat -> y_perm[E,C,H].
    ys = []
    g = 0
    while g < E:
        G = min(ch, E - g)
        ys.append(op(x_perm, w13_packed, w13_scale.to(torch.bfloat16), w13_zp,
                     w2_packed, w2_scale.to(torch.bfloat16), w2_zp,
                     w_perm.to(torch.bfloat16), g, G))
        g += G
    y_perm = ys[0] if len(ys) == 1 else torch.cat(ys, dim=0)   # [E,C,H]

    if not _FUSE:
        _mark_step()                                       # materialize y_perm before the unbin gather

    return unbin(y_perm, bin_idx, T, K)


def _mark_step():
    try:
        import habana_frameworks.torch.core as htorch
        htorch.mark_step()
    except Exception:  # noqa: BLE001  (CPU/test path: no habana)
        pass


def _emit_overflow_debug(topk_ids, emin, E, C, n_overflow, T, K):
    """One-shot LOUD: per-rank max local load + capacity overflow at this C, so the
    operator calibrates HPU_MOE_BINNED_CAP. Does ONE .item() (debug only, not hot path)."""
    _dbg_emitted[0] = True
    try:
        import logging
        local = (topk_ids.to(torch.int64) - int(emin)).reshape(-1)
        valid = (local >= 0) & (local < E)
        loads = torch.bincount(torch.where(valid, local, torch.full_like(local, E)),
                               minlength=E + 1)[:E]
        maxload = int(loads.max().item())
        n_of = int(n_overflow.item())
        total_local = int(loads.sum().item())
        logging.getLogger("vllm").warning(
            "[HPU_MOE_BINNED] one-shot @T=%d: cap C=%d, per-rank max local load=%d, "
            "mean=%.1f, total_local_picks=%d, CAPACITY_OVERFLOW(dropped)=%d %s. "
            "GEMM cut ~T/C=%.1fx. If overflow>0: raise HPU_MOE_BINNED_CAP to >=max_load "
            "(=%d here) for byte-exact; current drop is safe-degraded (sentinel, 0-fold).",
            T, C, maxload, total_local / max(E, 1), total_local, n_of,
            "LOSSLESS" if n_of == 0 else "LOSSY-but-safe", T / max(C, 1), maxload)
    except Exception as _ex:  # noqa: BLE001
        try:
            import logging
            logging.getLogger("vllm").warning(
                "[HPU_MOE_BINNED] debug emit failed (non-fatal): %r", _ex)
        except Exception:  # noqa: BLE001
            pass
