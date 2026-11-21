"""HPU DeepSeek Sparse Attention — the sparse MLA CONSUME side (option B: GATHER).

Given the indexer's `topk_indices_buffer` ([T, index_topk], int32, **-1 padded**), restrict MLA
attention to ONLY the selected latent-KV tokens. Two paths, both numerically equivalent and both
unit-testable off-hardware against a dense-but-masked oracle:

  * ``sparse_mla_gather_attention`` (PRODUCT, scales to 200K): sentinel-column scatter builds the
    ≤index_topk gathered key/value set per query, then a single dense SDPA runs over the gathered
    width K = min(index_topk, S). Cost O(L·K) memory + compute, NOT O(L·S). This is the function
    ``HPUMLAAttention.forward_impl`` (oot_mla.py) calls for prefill (``forward_mha``) and decode
    (``forward_mqa``).
  * ``sparse_mla_masked_attention`` (≤32K ORACLE, parity only): builds a full [L, S] additive
    bias (-inf on non-selected / non-causal keys) and runs dense SDPA over all S keys. O(L·S) —
    used only as the correctness oracle in the CPU parity test; never on the 200K serving path.

WHY A SELF-BUILT BIAS (causality regardless of is_causal):
  ``forward_mha`` hardcodes ``is_causal=True`` (hpu_attn.py:293) and routes through
  ``ModuleFusedSDPA``, which DROPS ``is_causal`` whenever an ``attn_mask`` is present
  (vllm_gaudi/extension/utils.py:416-419: ``if is_causal and attn_mask is not None: is_causal =
  False``). So the moment we hand it a mask, the kernel applies ONLY our mask. We therefore encode
  causality (s ≤ abs_pos(t)) AND sparsity (selected-only) AND padding-row masking entirely inside
  the additive bias / the gather geometry — correct whether the caller passed is_causal True or
  False, INC or non-INC. We never rely on the kernel's own causal triangle.

LOCKED CONTRACT POINTS (verifier-resolved — see BUILD_PLAN.md):
  5. -1 padding consumed via SENTINEL-COLUMN scatter (append a dummy key column / row, scatter the
     selected entries there, slice it off). NEVER ``idx.clamp(min=0)`` — that silently selects
     token 0 for every empty slot. (Confirmed correct: sentinel-scatter yields exactly the valid
     count; clamp(min=0) over-selects.)
  6. NaN guard: a query whose every causal key is non-selected → an all-(-inf) softmax row → NaN.
     We (a) always force-include the query's own diagonal (self) key so ≥1 valid key exists, and
     (b) post-clamp any residual NaN→0 as a belt-and-suspenders guard.
  7. Consume = GATHER (option B). Masked-MHA additive -inf bias is the ≤32K oracle only.
  10. Dims are config-driven; nothing about index_topk / heads / head_dim is hardcoded here — they
     are derived from the buffer / tensor shapes the caller passes in.

This module is pure torch (no metadata/cache/HPU-op dependency) so it runs on CPU for parity. The
caller in oot_mla.py supplies the already-projected per-head MLA q/k/v (post W_UK/W_UV up-proj, the
same tensors ``forward_mha`` builds) plus the indexer topk indices and each query's absolute
position; this module owns ONLY the "restrict to selected tokens + attend" step.
"""
from __future__ import annotations

import torch

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------------------------
# Internal: turn the indexer's [T, index_topk] (-1 padded) selection into a boolean keep-mask
# [T, S] over the key axis, with causality and forced self-inclusion folded in. Built via a
# SENTINEL COLUMN (width S+1, dump -1 into column S, slice it off) — never clamp(min=0).
# ---------------------------------------------------------------------------------------------
def _build_keep_mask(
    topk_indices: torch.Tensor,   # [T, K_idx] int, -1 padded (the indexer output, possibly wider
                                  #            than the live key count S)
    abs_query_pos: torch.Tensor,  # [T] int, absolute position of each query along the key axis
    num_keys: int,                # S = number of live keys this query block attends over
) -> torch.Tensor:
    """Return a boolean [T, S] mask: True = key is selected-and-causal-and-attendable.

    Sentinel-column scatter: scatter the (clamped-to-valid) selected indices' True bits into a
    width ``S+1`` buffer where the extra column ``S`` is the trash sink for -1 pads; slice it off.
    Then AND with the causal bound and force the diagonal (self) on so no row is fully empty.
    """
    T = topk_indices.shape[0]
    device = topk_indices.device
    idx = topk_indices.long()

    # SENTINEL COLUMN: route every -1 pad (and any stray out-of-range index >= S, which can occur
    # because the indexer buffer width index_topk may exceed the live S during short prefills) to
    # the trash column `num_keys`. Scatter True there, then drop the trash column.
    valid_entry = (idx >= 0) & (idx < num_keys)          # [T, K_idx]
    sink = torch.where(valid_entry, idx, torch.full_like(idx, num_keys))  # -1/oob -> S
    keep = torch.zeros((T, num_keys + 1), dtype=torch.bool, device=device)
    keep.scatter_(1, sink, torch.ones_like(sink, dtype=torch.bool))
    keep = keep[:, :num_keys]                            # slice off the sentinel column -> [T, S]

    # Causal bound: a key at position s is attendable only if s <= abs_pos(t) (self included).
    key_pos = torch.arange(num_keys, device=device).view(1, -1)   # [1, S]
    causal = key_pos <= abs_query_pos.view(-1, 1)        # [T, S]
    keep &= causal

    # NaN guard (a): force the query's own diagonal key ON. abs_pos(t) indexes the self key within
    # this block's key axis (caller guarantees self is in-range, i.e. 0 <= abs_pos(t) < S). The
    # indexer's causal top-k already includes self, but a degenerate/empty row could drop it; this
    # makes "≥1 valid key" unconditional so the softmax row is never all -inf.
    self_col = abs_query_pos.clamp(0, num_keys - 1).view(-1, 1)   # [T, 1]
    keep.scatter_(1, self_col, torch.ones_like(self_col, dtype=torch.bool))
    return keep


# ---------------------------------------------------------------------------------------------
# Internal: compact each row's selected keys to the left so we can GATHER a fixed-width K slab
# (K = min(index_topk, S)) of key/value rows per query — the O(L·K) product path. Returns the
# per-row gather indices into [0, S) (sentinel rows point at a dummy slot) plus a validity mask.
# ---------------------------------------------------------------------------------------------
def _compact_gather_indices(
    keep_mask: torch.Tensor,   # [T, S] bool from _build_keep_mask
    width: int,                # K = gather width (== min(index_topk, S))
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (gather_idx [T, K] long in [0, S], valid [T, K] bool).

    For each query we want up to ``width`` selected key positions packed to the left. We do this
    without a data-dependent loop (graph-friendly): rank selected columns by cumulative count and
    scatter their column id into the packed slot. Unfilled slots get the SENTINEL row id ``S`` (a
    dummy key/value row the caller appends), so a later slice / mask zeroes them — NEVER index 0.
    """
    T, S = keep_mask.shape
    device = keep_mask.device
    keep_i = keep_mask.to(torch.int32)
    # rank[t, s] = number of selected keys at columns < s (0-based slot this column lands in).
    rank = torch.cumsum(keep_i, dim=1) - keep_i          # [T, S], exclusive prefix sum
    # Only selected columns with rank < width actually land in the packed slab.
    lands = keep_mask & (rank < width)                   # [T, S]
    col_id = torch.arange(S, device=device).view(1, -1).expand(T, S)
    # Build [T, width] packed gather indices initialised to the SENTINEL row id S.
    gather_idx = torch.full((T, width), S, dtype=torch.long, device=device)
    slot = torch.where(lands, rank.long(), torch.full_like(rank.long(), width))  # land-or-trash
    # scatter selected column ids into their packed slot; trash slot == width is sliced off.
    gather_idx_ext = torch.full((T, width + 1), S, dtype=torch.long, device=device)
    gather_idx_ext.scatter_(1, slot, col_id.long())
    gather_idx = gather_idx_ext[:, :width]               # [T, K], unfilled -> sentinel row S
    valid = gather_idx < S                               # [T, K] bool: real key vs sentinel pad
    return gather_idx, valid


# =============================================================================================
# PUBLIC API
# =============================================================================================
def sparse_mla_masked_attention(
    q: torch.Tensor,              # [T, H, Dh]  per-head MLA query (nope+rope concatenated)
    k: torch.Tensor,              # [S, H, Dh]  per-head MLA key   (broadcast MQA already expanded)
    v: torch.Tensor,              # [S, H, Dv]  per-head MLA value
    topk_indices: torch.Tensor,   # [T, K_idx]  int32, -1 padded (indexer output)
    abs_query_pos: torch.Tensor,  # [T]         int, absolute key-axis position of each query
    scale: float,
) -> torch.Tensor:                # [T, H, Dv]
    """≤32K ORACLE consume: dense SDPA over ALL S keys with an additive -inf bias on every
    non-selected / non-causal key (sentinel-scatter built). O(L·S) memory — parity oracle only.

    Returned layout matches the gather path so the parity test can diff them directly. Causality
    lives entirely in the bias (does not depend on any kernel is_causal flag). NaN guard applied.
    """
    T, H, _ = q.shape
    S = k.shape[0]
    keep = _build_keep_mask(topk_indices, abs_query_pos, S)   # [T, S] bool, self-incl, causal
    bias = torch.where(
        keep, torch.zeros((), dtype=q.dtype, device=q.device),
        torch.full((), NEG_INF, dtype=q.dtype, device=q.device),
    )  # [T, S] additive bias

    qf = q.transpose(0, 1).to(torch.float32)          # [H, T, Dh]
    kf = k.transpose(0, 1).to(torch.float32)          # [H, S, Dh]
    vf = v.transpose(0, 1).to(torch.float32)          # [H, S, Dv]
    logits = torch.matmul(qf, kf.transpose(-1, -2)) * scale          # [H, T, S]
    logits = logits + bias.unsqueeze(0).to(torch.float32)            # broadcast bias over heads
    probs = torch.softmax(logits, dim=-1)
    # NaN guard (b): any residual all-(-inf) row -> NaN -> 0 (self-inclusion should prevent this).
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, vf)                     # [H, T, Dv]
    return out.transpose(0, 1).to(q.dtype)            # [T, H, Dv]


def sparse_mla_gather_attention(
    q: torch.Tensor,              # [T, H, Dh]
    k: torch.Tensor,              # [S, H, Dh]
    v: torch.Tensor,              # [S, H, Dv]
    topk_indices: torch.Tensor,   # [T, K_idx]  int32, -1 padded (indexer output)
    abs_query_pos: torch.Tensor,  # [T]         int
    scale: float,
    gather_width: int | None = None,   # K; default min(index_topk, S). Lets the caller bucket K.
) -> torch.Tensor:                # [T, H, Dv]
    """PRODUCT consume (option B, scales to 200K): GATHER the ≤K selected latent-KV rows per query
    then dense SDPA over just those K. O(L·K) memory + compute.

    Implementation:
      1. keep-mask via sentinel-scatter (causal + self-incl), 2. compact to [T, K] gather indices
      with a SENTINEL row id S for empty slots, 3. append one dummy key/value row at index S,
      4. gather -> [T, K, H, Dh/Dv], 5. batched per-query SDPA, 6. mask sentinel slots to -inf in
      the logits (NEVER let them contribute; their gathered content is the dummy row but we mask
      anyway), 7. NaN->0 guard.
    """
    T, H, Dh = q.shape
    S = k.shape[0]
    Dv = v.shape[-1]

    keep = _build_keep_mask(topk_indices, abs_query_pos, S)        # [T, S]
    # The keep-mask is the SINGLE SOURCE OF TRUTH for which keys are attended (index-topk selected,
    # causal, plus the forced self key). The gather slab MUST be wide enough to hold every kept key
    # of the busiest row, else we'd silently drop a selected key (the gather would disagree with the
    # masked-MHA oracle). So the default width = max kept-count, NOT index_topk — because forcing
    # self-inclusion (NaN guard) can push a row to index_topk + 1 keys when self wasn't already in
    # the top-k. The caller MAY override `gather_width` to bucket K for the graph compiler, but it
    # must then be >= the max kept-count or attention is lossy; we clamp up to keep correctness.
    if gather_width is None:
        max_keep = int(keep.sum(dim=1).max().item()) if keep.numel() else 1
        width = max_keep
    else:
        # STATIC-SHAPE FIX (2026-06-19): caller passes index_topk+1, which is >= max_keep by
        # construction (top-k keeps <= index_topk keys, +1 for the forced-self NaN-guard key) ->
        # SKIP the max_keep.item() D2H sync. Result: STATIC gather width (the per-sparse-layer
        # gather recipe is now capturable -> de-fragments) AND no per-layer host sync (~78/fwd).
        # Byte-exact: a wider-than-needed gather only adds -inf-masked sentinel columns (greedy-gated).
        width = gather_width
    width = max(1, min(width, S))             # never wider than the live key count; never 0

    gather_idx, valid = _compact_gather_indices(keep, width)       # [T, K] long, [T, K] bool

    # Append ONE sentinel key/value row at index S (the trash sink); gather indices == S land here.
    k_ext = torch.cat([k, torch.zeros((1, H, Dh), dtype=k.dtype, device=k.device)], dim=0)  # [S+1,H,Dh]
    v_ext = torch.cat([v, torch.zeros((1, H, Dv), dtype=v.dtype, device=v.device)], dim=0)  # [S+1,H,Dv]

    # Gather the selected rows: [T, K, H, D]. index_select on a flattened key axis, then reshape.
    flat_idx = gather_idx.reshape(-1)                              # [T*K]
    k_g = k_ext.index_select(0, flat_idx).view(T, width, H, Dh)    # [T, K, H, Dh]
    v_g = v_ext.index_select(0, flat_idx).view(T, width, H, Dv)    # [T, K, H, Dv]

    # Per-query SDPA over the K gathered keys. Move heads to the front: [H, T, K, D].
    qf = q.permute(1, 0, 2).unsqueeze(2).to(torch.float32)         # [H, T, 1, Dh]
    kf = k_g.permute(2, 0, 1, 3).to(torch.float32)                 # [H, T, K, Dh]
    vf = v_g.permute(2, 0, 1, 3).to(torch.float32)                 # [H, T, K, Dv]
    logits = torch.matmul(qf, kf.transpose(-1, -2)).squeeze(2) * scale   # [H, T, K]

    # Mask sentinel (pad) slots so they never contribute, regardless of their gathered content.
    pad_bias = torch.where(
        valid, torch.zeros((), dtype=torch.float32, device=q.device),
        torch.full((), NEG_INF, dtype=torch.float32, device=q.device),
    )  # [T, K]
    logits = logits + pad_bias.unsqueeze(0)                        # [H, T, K]
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)                       # NaN guard (b)
    out = torch.matmul(probs.unsqueeze(2), vf).squeeze(2)         # [H, T, Dv]
    return out.permute(1, 0, 2).to(q.dtype)                        # [T, H, Dv]


def build_sparse_attn_bias(
    topk_indices: torch.Tensor,   # [T, K_idx] int32, -1 padded
    abs_query_pos: torch.Tensor,  # [T] int
    num_keys: int,                # S
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> torch.Tensor:                # [T, S] additive bias (0 keep / -inf drop)
    """Expose the additive [T, S] selected+causal bias for callers (e.g. the oot_mla.py ≤32K path
    or a decode block_bias build) that want to hand FusedSDPA / flat_pa_mla a mask directly rather
    than gathering. Sentinel-scatter built; NaN-safe via forced self-inclusion. Pass this as the
    ModuleFusedSDPA ``attn_mask`` and the kernel will ignore its own is_causal (utils.py:416-419)."""
    keep = _build_keep_mask(
        topk_indices.to(device) if device is not None else topk_indices,
        abs_query_pos.to(device) if device is not None else abs_query_pos,
        num_keys,
    )
    dev = device if device is not None else keep.device
    return torch.where(
        keep,
        torch.zeros((), dtype=dtype, device=dev),
        torch.full((), NEG_INF, dtype=dtype, device=dev),
    )
