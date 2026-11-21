"""HPU DSA — indexer/consume metadata derivation from the HPU attention namedtuple.

The DeepSeek Sparse Attention indexer (the lightning scorer) and the gather-consume
need, per forward step:
  * per-query ABSOLUTE position (for the causal key bound  s <= abs_pos, self-inclusive),
  * the number of causal keys each query may see,
  * the is_prompt (prefill vs decode) split,
  * the slot_mapping (write current-step indexer-K) and the per-request paged block layout
    (gather past indexer-K for scoring + gather selected latent-KV rows in consume).

On HPU NONE of the DeepSeek CUDA metadata exists. `get_forward_context().attn_metadata`
is a SINGLE `TrimmedAttentionMetadata` namedtuple (vllm_gaudi `trim_attn_metadata`,
hpu_model_runner.py:872-902) — NOT a dict, NOT a `DeepseekV32IndexerMetadata`. This module
reconstructs everything the op/consume need from the fields that DO survive trimming:
  prefill: is_prompt, slot_mapping[B,Sp], seq_lens_tensor(=query_lens), context_lens_tensor,
           query_start_loc/query_start_loc_p, block_list (prefix-cache blocks), block_size.
  decode:  is_prompt, slot_mapping[B*next_n,1], block_list/block_groups/block_usage,
           block_size  (seq_lens_tensor is None in the standard non-mamba decode path, so the
           per-request context length is reconstructed from block_groups+block_usage).

Grounding (all read, none guessed):
  * prefill abs-pos: token_positions = range(context_len, context_len+query_len)
    (hpu_model_runner.py:2419) ; make_prefill_metadata sets seq_lens_tensor=query_lens,
    context_lens_tensor=context_lens (2692-2694). DeepSeek causal end is
    start_pos+1+offset = context_len+j+1, self-inclusive (mla_indexer.py:760-765).
  * decode block layout: block_groups[j]=request index of gathered block j (-1 pad),
    block_usage[j]=#valid KV slots in block j (=block_size except each request's last block
    = slot%block_size+1), block_list[j]=physical block id  (get_habana_paged_attn_buffers,
    hpu_model_runner.py:2149-2194). Per-request context_len = sum_j[group==r] block_usage[j].
  * flat paged cache: shape (num_blocks*block_size, num_kv_heads, head_size),
    slot = block_id*block_size + offset, write=index_copy_(0,slot,k),
    per-block read=index_select(0, block_list).unflatten(0,(-1,block_size))
    (hpu_paged_attn.py:74 ; HPUMLAAttentionBackend.get_kv_cache_shape hpu_attn.py:96-102).

Everything here is plain torch (CPU-testable). No HPU import, no model load.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Fields the indexer/consume read off attn_metadata. Anything NOT in the trimmed
# namedtuple raises AttributeError on plain getattr, so we ALWAYS use _get() (getattr
# with default None) — the parent dataclass (HPUMLAMetadata) carries extra fields the
# trimmed tuple drops, and vice-versa, so neither shape may be assumed.
_PREFILL_FIELDS = (
    "is_prompt",
    "slot_mapping",
    "seq_lens_tensor",
    "context_lens_tensor",
    "query_start_loc",
    "query_start_loc_p",
    "block_list",
    "block_size",
)
_DECODE_FIELDS = (
    "is_prompt",
    "slot_mapping",
    "block_list",
    "block_groups",
    "block_usage",
    "seq_lens_tensor",
    "block_size",
)


def _get(meta: object, name: str, default=None):
    """Defensive field access that works for both the parent HPUMLAMetadata dataclass
    and the trimmed namedtuple (which only carries a subset of fields)."""
    if meta is None:
        return default
    val = getattr(meta, name, default)
    return val if val is not None else default


@dataclass
class IndexerPrefillMeta:
    """Per-query causal bounds + cache maps for the PREFILL (multi-query) path.

    All tensors are flat over the *valid* prefill tokens (padding dropped), in the
    same token order the model emitted q/k (request-major, then position within request).
    """

    num_tokens: int                      # total valid prefill query tokens (sum of query_lens)
    batch_size: int                      # number of prefill requests (excludes padding rows)
    query_lens: torch.Tensor             # [B] int32, new-query length per request
    context_lens: torch.Tensor           # [B] int32, cached/prefix length per request (start_pos)
    seq_lens: torch.Tensor               # [B] int32, context_lens + query_lens (total causal extent)
    # per valid query token (length == num_tokens):
    abs_pos: torch.Tensor                # [T] int32, absolute position = context_len + j
    num_causal_keys: torch.Tensor        # [T] int32, abs_pos + 1 (self-inclusive)
    req_id_per_token: torch.Tensor       # [T] int32, request index each query token belongs to
    cu_query_lens: torch.Tensor          # [B+1] int32, exclusive prefix-sum of query_lens
    # cache maps:
    slot_mapping: torch.Tensor           # [T] int64, flat KV-cache slot per current query token
    block_list: torch.Tensor | None      # [Nblk] int64 prefix-cache physical blocks (None if no prefix)
    block_size: int


@dataclass
class IndexerDecodeMeta:
    """Per-query causal bounds + cache maps for the DECODE (single/few-query) path.

    Decode batches carry one (or next_n) query token(s) per request. The current key
    has already been written into the paged cache (slot_mapping) BEFORE the indexer runs,
    so each request's context_len already counts its own current token (self-inclusive).
    """

    num_tokens: int                      # total decode query tokens (batch_size * next_n)
    batch_size: int                      # number of decode requests (excludes padding rows)
    next_n: int                          # query tokens per request (1 normally, >1 for spec-decode)
    seq_lens: torch.Tensor               # [B] int32, total context length per request (incl. current)
    # per query token (length == num_tokens, row-major [B, next_n] flattened):
    abs_pos: torch.Tensor                # [T] int32, absolute position of each query token
    num_causal_keys: torch.Tensor        # [T] int32, abs_pos + 1
    req_id_per_token: torch.Tensor       # [T] int32, request index each query token belongs to
    # paged layout (the gathered-block view flat_pa_mla consumes):
    slot_mapping: torch.Tensor           # [T] int64, flat KV-cache slot of each current query token
    block_list: torch.Tensor             # [Nblk] int64 physical block id per gathered block
    block_groups: torch.Tensor           # [Nblk] int64 request index per gathered block (-1 = pad)
    block_usage: torch.Tensor            # [Nblk] int   valid KV slots in each gathered block
    block_size: int


@dataclass
class IndexerBatchMeta:
    """One coherent view per forward step. Exactly one of prefill/decode is populated
    (HPU never mixes prefill+decode in a single step — they are separate forward passes,
    set_forward_context is called once per pass). `is_prompt` selects which."""

    is_prompt: bool
    prefill: IndexerPrefillMeta | None
    decode: IndexerDecodeMeta | None

    @property
    def active(self) -> "IndexerPrefillMeta | IndexerDecodeMeta":
        return self.prefill if self.is_prompt else self.decode


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------
def _build_prefill(meta: object) -> IndexerPrefillMeta:
    slot_mapping = _get(meta, "slot_mapping")
    seq_lens_tensor = _get(meta, "seq_lens_tensor")     # == query_lens in prefill
    context_lens = _get(meta, "context_lens_tensor")
    block_size = int(_get(meta, "block_size", 0) or 0)
    block_list = _get(meta, "block_list")

    if slot_mapping is None or seq_lens_tensor is None:
        raise RuntimeError(
            "[hpu-dsa-ext] prefill indexer metadata needs slot_mapping + seq_lens_tensor "
            "(query_lens); got slot_mapping=%s seq_lens_tensor=%s. The HPU prefill builder "
            "(make_prefill_metadata) sets both — upstream drift?"
            % (type(slot_mapping), type(seq_lens_tensor))
        )

    # BUILD ON CPU. This metadata is host CONTROL data (valid-request mask, cu boundaries,
    # per-token req-id/abs-pos) built from DATA-DEPENDENT-shape ops: boolean index
    # (query_lens[valid_req], sm_flat[sm_flat>=0]) and repeat_interleave-by-query_lens. On HPU
    # EAGER those route through Synapse's shape-agnostic recipe cache, which fails to re-infer
    # on a shape-changed cache hit -> "[SHAPE AGNOSTIC] Cache hit Synapse shape inference
    # failed!" (crashed the >2048 sparse prefill; the <=2048 dense short-circuit never reached
    # here). Computing on CPU dodges Synapse for all of it; the small result tensors move back
    # to the original device once at the end, preserving the exact device contract.
    device = slot_mapping.device
    cpu = torch.device("cpu")
    slot_mapping = slot_mapping.to(cpu)
    query_lens = seq_lens_tensor.to(cpu).to(torch.int32)
    # context_lens is the prefix/start_pos; absent => pure (uncached) prompt => zeros.
    if context_lens is None:
        context_lens = torch.zeros_like(query_lens)
    else:
        context_lens = context_lens.to(cpu).to(torch.int32)

    # Padding rows have query_len == 0 (query_lens.extend([0]*pad), hpu_model_runner.py:2457).
    # Keep only real requests so per-token tensors cover exactly the valid prefill tokens.
    valid_req = query_lens > 0
    if not bool(torch.all(valid_req)):
        query_lens = query_lens[valid_req]
        context_lens = context_lens[: valid_req.shape[0]][valid_req]
    batch_size = int(query_lens.shape[0])
    seq_lens = context_lens + query_lens

    # exclusive prefix-sum of query_lens -> token slice boundaries
    cu = torch.zeros(batch_size + 1, dtype=torch.int32)
    if batch_size > 0:
        cu[1:] = torch.cumsum(query_lens, dim=0).to(torch.int32)
    num_tokens = int(cu[-1].item()) if batch_size > 0 else 0

    # per-token abs_pos = context_len[r] + (within-request offset); req id per token.
    # Built with repeat_interleave + a per-request arange so it is vectorized and
    # independent of how slot_mapping was padded.
    if num_tokens > 0:
        req_id_per_token = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32), query_lens.to(torch.int64)
        )
        ctx_per_token = torch.repeat_interleave(context_lens, query_lens.to(torch.int64))
        # within-request offset j: global token index minus the request's start
        starts_per_token = torch.repeat_interleave(cu[:-1], query_lens.to(torch.int64))
        within = torch.arange(num_tokens, dtype=torch.int32) - starts_per_token
        abs_pos = ctx_per_token + within
    else:
        req_id_per_token = torch.zeros(0, dtype=torch.int32)
        abs_pos = torch.zeros(0, dtype=torch.int32)
    num_causal_keys = abs_pos + 1  # self-inclusive (DeepSeek start_pos+1+offset)

    # slot_mapping arrives as [B, padded_seq] (or already flat). Flatten and take the first
    # `num_tokens` real slots in token order. The runner lays valid tokens first within each
    # request row, padding (-1) after; but the canonical consumer flattens then slices to the
    # valid token count — mirror that. We additionally drop any negative (pad) slot defensively.
    sm_flat = slot_mapping.reshape(-1)
    sm_valid = sm_flat[sm_flat >= 0]
    if sm_valid.shape[0] >= num_tokens:
        slot_out = sm_valid[:num_tokens].to(torch.int64)
    else:
        # Fallback: trust the raw flatten head (shouldn't happen, but never silently truncate).
        slot_out = sm_flat[:num_tokens].to(torch.int64)

    # Move the (small) control tensors back to the model's device for the indexer/consume ops.
    return IndexerPrefillMeta(
        num_tokens=num_tokens,
        batch_size=batch_size,
        query_lens=query_lens.to(device),
        context_lens=context_lens.to(device),
        seq_lens=seq_lens.to(device),
        abs_pos=abs_pos.to(device),
        num_causal_keys=num_causal_keys.to(device),
        req_id_per_token=req_id_per_token.to(device),
        cu_query_lens=cu.to(device),
        slot_mapping=slot_out.to(device),
        block_list=(block_list.to(torch.int64).to(device) if block_list is not None else None),
        block_size=block_size,
    )


def _build_decode(meta: object) -> IndexerDecodeMeta:
    slot_mapping = _get(meta, "slot_mapping")
    block_list = _get(meta, "block_list")
    block_groups = _get(meta, "block_groups")
    block_usage = _get(meta, "block_usage")
    seq_lens_tensor = _get(meta, "seq_lens_tensor")   # None in the standard non-mamba decode path
    block_size = int(_get(meta, "block_size", 0) or 0)

    if block_list is None or block_groups is None or block_usage is None:
        raise RuntimeError(
            "[hpu-dsa-ext] decode indexer metadata needs block_list/block_groups/block_usage "
            "(the paged decode layout); got %s/%s/%s. make_decode_metadata sets all three — "
            "upstream drift?" % (type(block_list), type(block_groups), type(block_usage))
        )
    # BUILD ON CPU (same rationale as _build_prefill: block_groups>=0 boolean index,
    # index_add_, repeat_interleave are data-dependent-shape ops that break HPU-eager Synapse
    # shape inference at >2048). Results move back to the model device at the end.
    device = block_list.device
    cpu = torch.device("cpu")
    block_list = block_list.to(cpu).to(torch.int64)
    block_groups = block_groups.to(cpu).to(torch.int64)
    # block_usage is model_config.dtype (float) — round to int slot counts.
    block_usage_i = torch.round(block_usage.to(cpu).to(torch.float32)).to(torch.int32)
    if slot_mapping is not None:
        slot_mapping = slot_mapping.to(cpu)
    if seq_lens_tensor is not None:
        seq_lens_tensor = seq_lens_tensor.to(cpu)

    # Per-request context length = sum of block_usage over that request's gathered blocks.
    # block_groups[j] == -1 marks padding blocks (excluded). Request indices are 0..B-1 dense.
    real = block_groups >= 0
    if real.any():
        batch_size = int(block_groups[real].max().item()) + 1
    else:
        batch_size = 0
    ctx_lens = torch.zeros(batch_size, dtype=torch.int32)
    if batch_size > 0:
        ctx_lens.index_add_(
            0,
            block_groups.clamp_min(0)[real],
            block_usage_i[real],
        )

    # next_n (query tokens per request). slot_mapping is [B*next_n, 1] (or [B, next_n]); the
    # *decode* batch has exactly batch_size requests, so next_n = valid_slots / batch_size.
    sm_flat = slot_mapping.reshape(-1) if slot_mapping is not None else torch.zeros(0)
    # Decode padding rows use dummy slots (>=0) so we cannot filter by sign; instead derive
    # next_n from the request count. The runner pads rows AFTER the real ones.
    if batch_size > 0 and sm_flat.numel() >= batch_size:
        next_n = max(1, int(sm_flat.numel() // batch_size))
        # but cap next_n so we never read past real rows: real tokens == batch_size*next_n only
        # when the runner did not over-pad; guard by using the schedule when seq_lens given.
    else:
        next_n = 1
    # Spec-decode normally uses next_n>1 only when slot_mapping was viewed [B*next_n,1].
    # If seq_lens_tensor is present (mamba-style) it is authoritative for next_n inference,
    # but for MLA decode it is None — fall back to 1 unless slot_mapping is an exact multiple
    # larger than batch_size AND padding would not explain it. Be conservative: default 1.
    if seq_lens_tensor is None:
        next_n = 1
    num_tokens = batch_size * next_n

    # seq_lens per request: prefer the reconstructed ctx_lens (counts the just-written current
    # key, so it is the full self-inclusive extent). If seq_lens_tensor is provided, trust it.
    if seq_lens_tensor is not None:
        seq_lens = seq_lens_tensor.to(torch.int32).reshape(-1)[:batch_size]
    else:
        seq_lens = ctx_lens

    if num_tokens > 0:
        req_id_per_token = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32),
            torch.full((batch_size,), next_n, dtype=torch.int64),
        )
        seq_per_token = torch.repeat_interleave(
            seq_lens, torch.full((batch_size,), next_n, dtype=torch.int64)
        )
        # within-request query offset for next_n>1: the j-th of next_n tokens sits at
        # (seq_len - next_n + j). For next_n==1 this is seq_len-1.
        within = (
            torch.arange(num_tokens, dtype=torch.int32)
            - req_id_per_token * next_n
        )
        abs_pos = seq_per_token - next_n + within
    else:
        req_id_per_token = torch.zeros(0, dtype=torch.int32)
        abs_pos = torch.zeros(0, dtype=torch.int32)
        seq_per_token = torch.zeros(0, dtype=torch.int32)
    num_causal_keys = abs_pos + 1

    slot_out = sm_flat[:num_tokens].to(torch.int64) if num_tokens > 0 else sm_flat.to(torch.int64)

    # Move the (small) control tensors back to the model's device for the indexer/consume ops.
    return IndexerDecodeMeta(
        num_tokens=num_tokens,
        batch_size=batch_size,
        next_n=next_n,
        seq_lens=seq_lens.to(device),
        abs_pos=abs_pos.to(device),
        num_causal_keys=num_causal_keys.to(device),
        req_id_per_token=req_id_per_token.to(device),
        slot_mapping=slot_out.to(device),
        block_list=block_list.to(device),
        block_groups=block_groups.to(device),
        block_usage=block_usage_i.to(device),
        block_size=block_size,
    )


def build_indexer_metadata(attn_metadata: object) -> IndexerBatchMeta | None:
    """Top-level entry: turn the HPU `TrimmedAttentionMetadata`/`HPUMLAMetadata` namedtuple
    into a clean IndexerBatchMeta the indexer op + gather-consume import.

    Returns None when attn_metadata is None (profiling / dummy run) so callers can early-out
    with the correct -1 (no-selection) buffer instead of crashing.
    """
    if attn_metadata is None:
        return None
    # is_prompt is present on both prefill and decode trimmed tuples; default True is a
    # safe-fail (prefill path is the more conservative full-causal derivation).
    is_prompt = bool(_get(attn_metadata, "is_prompt", True))
    if is_prompt:
        return IndexerBatchMeta(is_prompt=True, prefill=_build_prefill(attn_metadata), decode=None)
    return IndexerBatchMeta(is_prompt=False, prefill=None, decode=_build_decode(attn_metadata))


def per_request_block_ids(
    block_list: torch.Tensor,
    block_groups: torch.Tensor,
    batch_size: int,
) -> list[torch.Tensor]:
    """Decode helper: split the gathered flat block_list into one [n_blocks_r] physical-block
    tensor per request, in gathered order (the order flat_pa_mla walks). Padding blocks
    (group == -1) are dropped. Used by the gather-consume to assemble each request's past
    latent-KV / indexer-K rows independent of contiguous-PA assumptions."""
    out: list[torch.Tensor] = []
    for r in range(batch_size):
        out.append(block_list[block_groups == r])
    return out


# --------------------------------------------------------------------------------------
# CPU self-test (no HPU, no vllm import) — run:  python -m hpu_dsa_ext.indexer_metadata_hpu
# --------------------------------------------------------------------------------------
def _selftest() -> None:
    import collections

    Trim = collections.namedtuple(
        "TrimmedAttentionMetadata",
        "is_prompt slot_mapping seq_lens_tensor context_lens_tensor "
        "query_start_loc query_start_loc_p block_list block_groups block_usage block_size",
    )

    # ---- PREFILL: 2 requests, query_lens=[3,2], context_lens=[4,0], padded to batch 3 / seq 4
    bsz_pad, seq_pad, blk = 3, 4, 16
    qlens = torch.tensor([3, 2, 0], dtype=torch.int32)        # row 2 is padding
    clens = torch.tensor([4, 0, 0], dtype=torch.int32)
    # slot_mapping[b, s] flat slots; valid first, -1 pad. Use easily-checked slot ids.
    sm = torch.full((bsz_pad, seq_pad), -1, dtype=torch.int64)
    sm[0, :3] = torch.tensor([100, 101, 102])
    sm[1, :2] = torch.tensor([200, 201])
    p = Trim(True, sm, qlens, clens, None, None, None, None, None, blk)
    m = build_indexer_metadata(p)
    pf = m.prefill
    assert m.is_prompt and pf.num_tokens == 5 and pf.batch_size == 2, (pf.num_tokens, pf.batch_size)
    assert pf.abs_pos.tolist() == [4, 5, 6, 0, 1], pf.abs_pos.tolist()
    assert pf.num_causal_keys.tolist() == [5, 6, 7, 1, 2], pf.num_causal_keys.tolist()
    assert pf.req_id_per_token.tolist() == [0, 0, 0, 1, 1], pf.req_id_per_token.tolist()
    assert pf.slot_mapping.tolist() == [100, 101, 102, 200, 201], pf.slot_mapping.tolist()
    assert pf.seq_lens.tolist() == [7, 2], pf.seq_lens.tolist()

    # ---- DECODE: 2 requests, seq_lens 33 and 17 (block_size 16) -> blocks [16,16,1] and [16,1]
    # Request 0: 3 blocks usage [16,16,1]=33 ; Request 1: 2 blocks usage [16,1]=17.
    block_list = torch.tensor([10, 11, 12, 20, 21, 0, 0], dtype=torch.int64)   # padded with PAD id
    block_groups = torch.tensor([0, 0, 0, 1, 1, -1, -1], dtype=torch.int64)
    block_usage = torch.tensor([16, 16, 1, 16, 1, 1, 1], dtype=torch.float32)  # float per runner
    sm_d = torch.tensor([[512], [848]], dtype=torch.int64)                     # one slot per req
    d = Trim(False, sm_d, None, None, None, None, block_list, block_groups, block_usage, blk)
    md = build_indexer_metadata(d)
    dec = md.decode
    assert not md.is_prompt and dec.batch_size == 2 and dec.next_n == 1, (dec.batch_size, dec.next_n)
    assert dec.seq_lens.tolist() == [33, 17], dec.seq_lens.tolist()
    assert dec.abs_pos.tolist() == [32, 16], dec.abs_pos.tolist()
    assert dec.num_causal_keys.tolist() == [33, 17], dec.num_causal_keys.tolist()
    assert dec.req_id_per_token.tolist() == [0, 1], dec.req_id_per_token.tolist()
    assert dec.slot_mapping.tolist() == [512, 848], dec.slot_mapping.tolist()
    pr = per_request_block_ids(dec.block_list, dec.block_groups, dec.batch_size)
    assert pr[0].tolist() == [10, 11, 12] and pr[1].tolist() == [20, 21], [t.tolist() for t in pr]

    # ---- None (profiling) -> None
    assert build_indexer_metadata(None) is None

    # ---- defensive getattr: a tuple missing a decode field must raise a clear RuntimeError
    Partial = collections.namedtuple("TrimmedAttentionMetadata2", "is_prompt slot_mapping block_size")
    try:
        build_indexer_metadata(Partial(False, sm_d, blk))
        raise AssertionError("expected RuntimeError for missing decode block fields")
    except RuntimeError as e:
        assert "block_list" in str(e)

    print("indexer_metadata_hpu self-test: PASS")


if __name__ == "__main__":
    _selftest()
