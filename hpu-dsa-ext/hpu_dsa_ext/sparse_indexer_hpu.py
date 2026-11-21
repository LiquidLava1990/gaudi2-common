"""HPU DeepSeek Sparse Attention — the numerically-validated lightning-indexer CORE
plus the `SparseAttnIndexer.forward_oot` op that drives it on real HPU metadata/cache.

The op is registered as `forward_oot` (HPU is out-of-tree; CustomOp dispatch routes there,
NOT to forward_hpu) — see register() at the bottom. The pure-torch scorer
(`lightning_indexer_topk` / `reference_indexer_topk`) has no metadata/cache dependency so it
is unit-testable off-hardware against the HF GlmMoeDsaIndexer / DeepSeek reference
(parity gate in tests/parity_indexer.py). The forward_oot orchestration around it
(K-cache insert, HPUMLAMetadata-derived causal bounds, past-key gather, M-tiled scoring,
prefill+decode) is the productizable ≤32-64K gather-consume path; ~200K needs a TPC kernel.

Math (resolved + validated: Jaccard 0.996, top-50 recall 1.0 vs DeepSeek fp8 oracle):
  `weights` already carry q_scale * softmax_scale * n_head**-0.5, so q is used as bf16 with NO
  q_scale and NO k_scale multiply (mathematically exact because ReLU is positive-homogeneous and
  q_scale>0):
      score[t,s] = sum_h weights[t,h] * ReLU( q_bf16[t,h,:] . k_bf16[s,:] ),  causal s<=abs_pos(t)
      topk[t]    = topk(score[t], index_topk)        (-1 padded for short/empty rows)

Scope: correct at any length; productizable to ~32-64K in pure torch. ~200K needs a fused TPC
kernel (prefill TTFT wall). Memory bounding uses small M tiles + `del` (mark_step is a no-op in
PT_HPU_LAZY_MODE=0). Do NOT materialize [M,H,S] monolithically beyond M<=128, and do NOT head-loop
(measured worse on HPU). See BUILD_PLAN.md.

CONTRACT (locked, verifier-resolved):
  forward_oot(self, hidden_states, q_quant, k, weights)  — 4 args.
    q_quant IS q_fp8 [T, n_head, head_dim] (q_scale folded into weights — NOT passed).
    k       = [T, head_dim]  current-step indexer keys (MQA, single shared head).
    weights = [T, n_head]    already carry q_scale * softmax_scale * n_head**-0.5.
  Writes self.topk_indices_buffer[:T, :topk_tokens] (int32, -1 padded). Returns that buffer.
  Honors self.skip_k_cache_insert (shared layer reuses the prior layer's cache+buffer).
"""
import torch

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Validated pure-torch scorer (DO NOT rewrite — parity-gated off-hardware).
# ---------------------------------------------------------------------------
def lightning_indexer_topk(
    q: torch.Tensor,        # [T, H, D]  fp8 or bf16 (q_scale folded into weights — do NOT rescale)
    k: torch.Tensor,        # [S, D]     bf16 indexer keys (MQA, single head shared across H)
    weights: torch.Tensor,  # [T, H]     bf16, already folded with q_scale*softmax_scale*H**-0.5
    topk_tokens: int,       # index_topk (2048 for GLM-5.1 / DeepSeek-V3.2)
    abs_query_pos: torch.Tensor,  # [T] int, absolute position of each query (for causal bound vs k)
    tile_m: int = 128,      # rows per tile; <=128 bounds the [m,H,S] transient (~3.4GB @200K)
) -> torch.Tensor:
    """Causal top-k token selection. Returns int32 [T, topk_tokens], -1 padded.

    A query at absolute position p may select any key index s with s <= p (self included). Rows with
    fewer than topk_tokens causal keys get -1 in the unused slots (consume side must treat -1 as
    'not selected' via sentinel-scatter, never clamp(min=0)).
    """
    T, H, D = q.shape
    S = k.shape[0]
    kk = min(topk_tokens, S)
    out = torch.full((T, topk_tokens), -1, dtype=torch.int32, device=q.device)
    kt = k.to(torch.bfloat16)
    key_pos = torch.arange(S, device=q.device).view(1, -1)  # [1, S]

    for m0 in range(0, T, tile_m):
        m1 = min(m0 + tile_m, T)
        qm = q[m0:m1].to(torch.bfloat16)          # [m, H, D]
        wm = weights[m0:m1].to(torch.bfloat16)    # [m, H]
        # per-head dot products -> ReLU -> weighted head-sum -> [m, S]
        logits = torch.einsum("mhd,sd->mhs", qm, kt).clamp_(min=0.0)  # ReLU (in-place)
        score = torch.einsum("mhs,mh->ms", logits, wm)               # [m, S]
        del logits
        # causal mask: invalidate keys beyond this query's absolute position
        ap = abs_query_pos[m0:m1].view(-1, 1)     # [m, 1]
        score.masked_fill_(key_pos > ap, NEG_INF)
        idx = score.topk(kk, dim=-1).indices.to(torch.int32)         # [m, kk]
        valid = torch.gather(score, 1, idx.long()) > NEG_INF         # drop -inf (short rows)
        out[m0:m1, :kk] = torch.where(valid, idx, torch.full_like(idx, -1))
        del score, idx, valid
    return out


def reference_indexer_topk(q, k, weights, topk_tokens, abs_query_pos):
    """Naive full-precision reference (oracle) — no tiling, fp32 — for the parity test."""
    T, H, D = q.shape
    S = k.shape[0]
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    wf = weights.to(torch.float32)
    logits = torch.relu(torch.einsum("thd,sd->ths", qf, kf))   # [T, H, S]
    score = torch.einsum("ths,th->ts", logits, wf)             # [T, S]
    key_pos = torch.arange(S).view(1, -1)
    score = score.masked_fill(key_pos > abs_query_pos.view(-1, 1), NEG_INF)
    kk = min(topk_tokens, S)
    idx = score.topk(kk, dim=-1).indices
    return idx  # [T, kk]


# ---------------------------------------------------------------------------
# forward_oot — drive the scorer on real HPU metadata + the bf16 indexer K-cache.
#
# Plumbing lives in two sibling modules (imported lazily so this module stays
# CPU-importable / parity-testable). The REAL surfaces (verified against the
# landed siblings) are:
#
#   indexer_cache_hpu  (BUILD_PLAN step 2, blocker #3) — owns the self-allocated bf16,
#     head_dim, FLAT (num_blocks*block_size, head_dim) indexer K-cache. Surface:
#       class IndexerKCache(bound_cache, block_size)
#         .insert(k[T,(1,)D], slot_mapping[T]) -> cache      # index_copy_ current-step k
#         .gather_blocks(block_list)           -> [nblk*bs, D]  # paged prefix read (block order)
#         .gather_rows(flat_slots)             -> [N, D]        # arbitrary flat-slot read
#       apply()  — installs the runner get_kv_cache_spec + bind_kv_cache patches.
#     `bound_cache` is the SparseAttnIndexer's DeepseekV32IndexerCache.kv_cache tensor.
#
#   indexer_metadata_hpu  (BUILD_PLAN step 2, blockers #2/#8) — turns the HPUMLAMetadata
#     namedtuple into causal bounds (DeepseekV32IndexerMetadata does NOT exist on HPU).
#     Surface:
#       build_indexer_metadata(attn_metadata) -> IndexerBatchMeta | None
#         .is_prompt ; .active -> IndexerPrefillMeta | IndexerDecodeMeta
#       IndexerPrefillMeta: num_tokens, batch_size, query_lens[B], context_lens[B],
#         seq_lens[B], abs_pos[T], req_id_per_token[T], cu_query_lens[B+1],
#         slot_mapping[T], block_list|None, block_size
#       IndexerDecodeMeta:  num_tokens, batch_size, next_n, seq_lens[B], abs_pos[T],
#         req_id_per_token[T], slot_mapping[T], block_list, block_groups, block_usage,
#         block_size
#       per_request_block_ids(block_list, block_groups, batch_size) -> list[Tensor]
#
# CONSUME-CONTRACT for the buffer this writes (matches sparse_mla_consume + the
# masked-MHA ≤32K oracle): topk_indices are ABSOLUTE token positions within each
# request's own key axis [0, seq_len_r), and abs_query_pos[t] is the query's absolute
# position on that same axis. forward_oot therefore gathers each request's keys in
# ABSOLUTE order (prefix blocks in block order, then the current step), so a scorer
# index == the absolute token position with no remap. (No-prefix-caching is required
# for GLM-5.1 — LAUNCH_FLAGS — so prefill context_lens==0 and keys == current step.)
#
# To keep this file independently testable, forward_oot ALSO accepts a hand-built plan
# via self._dsa_plan_override (set only by the CPU unit test); in the real run that is
# absent and the sibling modules are used.
# ---------------------------------------------------------------------------
def _resolve_plumbing():
    """Import the sibling plumbing lazily. Returns (cache_mod, meta_mod) or (None, None)
    if not yet available (so this module still imports for the CPU scorer parity test)."""
    try:
        from . import indexer_cache_hpu as cache_mod
        from . import indexer_metadata_hpu as meta_mod
    except Exception:  # pragma: no cover - siblings authored in parallel
        try:
            import indexer_cache_hpu as cache_mod  # type: ignore
            import indexer_metadata_hpu as meta_mod  # type: ignore
        except Exception:
            return None, None
    return cache_mod, meta_mod


def _gather_request_keys(kc, meta_active, r, is_decode):
    """Gather request r's indexer keys in ABSOLUTE order [0, seq_len_r), bf16 [S_r, D].

    Prefill: prefix blocks (block order == absolute order) trimmed to context_len_r,
             then this request's just-inserted current-step keys (its slot_mapping
             slice). With no-prefix-caching context_len_r==0 -> keys == current step.
    Decode : the request's full paged history via per_request_block_ids -> gather_blocks,
             trimmed to seq_len_r (the just-written current key is the last slot).

    Returns a contiguous [S_r, head_dim] bf16 tensor whose row index == absolute pos.
    """
    from . import indexer_metadata_hpu as meta_mod

    if is_decode:
        seq_len = int(meta_active.seq_lens[r].item())
        blocks = meta_mod.per_request_block_ids(
            meta_active.block_list, meta_active.block_groups, meta_active.batch_size
        )[r]
        keys = kc.gather_blocks(blocks)            # [nblk*bs, D], block (==absolute) order
        keys = keys[:seq_len]                      # trim padded tail of last block
        return keys.view(keys.shape[0], -1).to(torch.bfloat16)

    # prefill
    ctx = int(meta_active.context_lens[r].item())
    qlen = int(meta_active.query_lens[r].item())
    seq_len = ctx + qlen
    tok_start = int(meta_active.cu_query_lens[r].item())
    tok_end = int(meta_active.cu_query_lens[r + 1].item())
    # current-step keys for this request: its own slot_mapping slice (absolute order).
    cur = kc.gather_rows(meta_active.slot_mapping[tok_start:tok_end])   # [qlen, D]
    if ctx > 0 and meta_active.block_list is not None:
        # prefix-cached context: gather this request's prefix blocks. NOTE this requires
        # per-request block splitting; with no-prefix-caching (required) ctx==0 so this
        # branch is unused. We gather the whole prefix block_list and trim to ctx as a
        # best-effort single-request fallback.
        past = kc.gather_blocks(meta_active.block_list)[:ctx]           # [ctx, D]
        keys = torch.cat([past, cur], dim=0)                           # [seq_len, D]
    else:
        keys = cur
    return keys.view(keys.shape[0], -1).to(torch.bfloat16)


def _score_segment_into_buffer(
    self,
    q_quant: torch.Tensor,     # [T, H, D] fp8
    weights: torch.Tensor,     # [T, H]
    keys: torch.Tensor,        # [S, D] bf16 (past+current for this segment)
    abs_query_pos: torch.Tensor,  # [tok] long
    token_start: int,
    token_end: int,
) -> None:
    """Score one request's query rows against its key block and write the int32
    top-k indices (-1 padded) into self.topk_indices_buffer[token_start:token_end].

    The scorer already M<=128 tiles internally and `del`s the [m,H,S] transient. The
    returned indices are positions WITHIN `keys` for this request — and because
    `_gather_request_keys` assembles `keys` in ABSOLUTE order [0, seq_len_r), that
    local index == the absolute token position, which is exactly what the masked-MHA
    ≤32K oracle and the gather-consume path (sparse_mla_consume) both expect.
    """
    topk = self.topk_tokens
    idx = lightning_indexer_topk(
        q_quant[token_start:token_end],
        keys,
        weights[token_start:token_end],
        topk,
        abs_query_pos,
    )  # [tok, topk] int32, -1 padded
    self.topk_indices_buffer[token_start:token_end, :topk] = idx
    del idx


def forward_oot(self, hidden_states, q_quant, k, weights):  # noqa: ANN001
    """HPU sparse-attention indexer. Fills self.topk_indices_buffer with causal top-k
    latent-KV token indices per query and returns it. Prefill + decode.

    Args (locked contract):
      hidden_states : [T, hidden]  (unused here; kept for signature parity — k/weights
                      already projected by the in-image Indexer.forward).
      q_quant       : [T, n_head, head_dim] fp8 (q_scale folded into `weights`).
      k             : [T, head_dim] bf16 current-step indexer keys (MQA single head).
      weights       : [T, n_head]   bf16, already * q_scale*softmax_scale*n_head**-0.5.
    """
    num_tokens = q_quant.shape[0]
    topk = self.topk_tokens

    # Reset the active rows to -1 (sentinel for 'no selection'; consume side scatters,
    # never clamp(min=0)). Mirrors upstream sparse_attn_indexer line 175.
    self.topk_indices_buffer[:num_tokens] = -1

    # --- CPU unit-test override path (no HPU metadata / cache) -------------------
    # The test sets self._dsa_plan_override (list of segment dicts) and provides the
    # already-gathered keys per segment, so the full prefill/decode orchestration is
    # exercised on CPU without the sibling plumbing.
    plan_override = getattr(self, "_dsa_plan_override", None)
    if plan_override is not None:
        for seg in plan_override:
            _score_segment_into_buffer(
                self,
                q_quant,
                weights,
                seg["keys"],
                seg["abs_query_pos"],
                seg["token_start"],
                seg["token_end"],
            )
        return self.topk_indices_buffer

    # --- Real HPU path ----------------------------------------------------------
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    # Dummy/profiling run: attn_metadata is None (or lacks our derivation inputs).
    # Leave the buffer at -1 and return; the consume side treats -1 as no-selection.
    if attn_metadata is None:
        return self.topk_indices_buffer

    cache_mod, meta_mod = _resolve_plumbing()
    if cache_mod is None or meta_mod is None:  # pragma: no cover
        raise RuntimeError(
            "[hpu-dsa-ext] forward_oot needs indexer_cache_hpu + indexer_metadata_hpu "
            "(BUILD_PLAN step 2). They are absent — refusing to silently mis-select. "
            "The validated scorer is hpu_dsa_ext.sparse_indexer_hpu.lightning_indexer_topk."
        )

    # 1) Derive causal bounds + cache maps from the HPU attention namedtuple. None on a
    #    profiling/dummy run -> leave the -1 buffer (consume treats -1 as no-selection).
    meta = meta_mod.build_indexer_metadata(attn_metadata)
    if meta is None:
        return self.topk_indices_buffer
    active = meta.active
    is_decode = not meta.is_prompt
    if active.num_tokens == 0:
        return self.topk_indices_buffer

    # 2) Open our self-allocated flat bf16 indexer K-cache and insert current-step k
    #    (unless this is a shared layer reusing the prior layer's cache+buffer).
    #    self.k_cache is the DeepseekV32IndexerCache; .kv_cache is the bound flat tensor.
    #    Pass index_head_dim=self.head_dim (128) so IndexerKCache narrows the page-uniform
    #    over-wide backing tensor (allocated at the MLA head_size) to the real 128-wide cache.
    kc = cache_mod.IndexerKCache(
        self.k_cache.kv_cache,
        block_size=active.block_size,
        index_head_dim=getattr(self, "head_dim", None),
    )
    if not self.skip_k_cache_insert:
        kc.insert(k, active.slot_mapping)

    # 3) Per request: gather its keys in ABSOLUTE order so the scorer's local index ==
    #    absolute token position (the consume contract), score (M<=128-tiled + del inside
    #    the scorer), write the int32 top-k (-1 padded) into the buffer rows for that req.
    for r in range(active.batch_size):
        if is_decode:
            # decode: row r holds this request's next_n query tokens contiguously.
            tok_start = r * active.next_n
            tok_end = tok_start + active.next_n
        else:
            tok_start = int(active.cu_query_lens[r].item())
            tok_end = int(active.cu_query_lens[r + 1].item())
        keys = _gather_request_keys(kc, active, r, is_decode)   # [S_r, head_dim] bf16
        _score_segment_into_buffer(
            self,
            q_quant,
            weights,
            keys,
            active.abs_pos[tok_start:tok_end].to(torch.long),
            tok_start,
            tok_end,
        )
        del keys

    return self.topk_indices_buffer


# ---- registration (call from hpu_dsa_ext.register, then flip platform_patch.DSA_READY) ----
# IMPORTANT: HPU dispatch -> forward_oot (NOT forward_hpu). Pattern:
# vllm_gaudi/ops/hpu_grouped_topk_router.py:112 (we monkeypatch the method onto the
# upstream class rather than subclassing, to leave the model's Indexer wiring untouched).
def register():
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    SparseAttnIndexer.forward_oot = forward_oot
