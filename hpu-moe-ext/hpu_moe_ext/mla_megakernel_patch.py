"""MLA-DECODE MEGAKERNEL — extend SRAM-residency to the attention half of the GLM-5.2
decoder layer (single-stream lever; the MoE half already won 2.6x via the monolithic op).

The GLM-5.2 single-stream decode critical path has TWO poles: the MoE expert-FFN (already
SRAM-resident via fused_monolithic_moe) and the MLA decode attention — absorb -> rope ->
paged gather -> sdpa -> V-up — which round-trips ~11.6ms of gather/attn through HBM as a
chain of small ops (bmm, index_select, the block-online flat_pa_mla softmax, bmm). This
patch replaces that whole decode-attention chain with ONE fused OpBackend megakernel,

    torch.ops.hpu_mla_ext.mla_decode_paged(
        q_nope[B,N,P], q_pe[B,N,R], flat_cache[NB*bs,L+R], block_table[B,MB] int32,
        W_UK_T[N,P,L], W_UV[N,L,V], q_cos[B,1,R], q_sin[B,1,R], c_cos[B,S,R], c_sin[B,S,R],
        pad_mask[B,1,1,S], float scale, int rope_interleave, int block_size) -> [B, N*V]

which does the FULL forward_mqa replacement IN-KERNEL: absorb (bmm q_nope @ W_UK_T) ->
in-graph RoPE -> NODE-0 paged gather (flat_cache + block_table) -> single dense SDPA over
the gathered latent (value = latent[:, :L]) -> V-up (bmm @ W_UV). It is oracle-validated
byte-exact (mmel/mla_decode_oracle.py: flat_pa==dense 8e-16, absorb-identity 1e-15, bf16
chain ~2.3e-3; on-card test_mla_decode_op relNorm < 2e-2) and the paged variant's gather is
the byte-exact image of vllm_gaudi.extension.ops.flat_pa_mla's keys_fetch_func gather.

WHERE IT WIRES (the seam): vllm_gaudi.attention.oot_mla.HPUMLAAttention.forward_impl, the
decode branch. Stock decode does: q.split -> q_nope,q_pe; absorb decode_ql_nope =
bmm(q_nope, W_UK_T); WRITE current token latent+rope to kv_cache[0] via
self.impl.latent_cache_k(...); output = self.impl.forward_mqa(decode_ql_nope, q_pe, ...);
output = self._v_up_proj(output). The megakernel REPLACES the absorb + forward_mqa + V-up
(it takes raw q_nope and does absorb+gather+sdpa+V-up itself). The latent_cache_k WRITE at
the top MUST still run first (the kernel's NODE-0 gather reads that cache) — so this patch
keeps the cache write byte-identical and only swaps the decode OUTPUT computation.

ROPE CONTRACT — OPTION 1a (byte-exact first): the stock serve PRE-ropes q_pe and the cached
pe BEFORE the cache write (proven: flat_pa_mla applies NO rope and dots query directly
against the cached key, so both q_pe and the cached pe are already in the roped space). The
megakernel ropes in-kernel. To avoid double-rope WITHOUT touching the cache format, we feed
the already-roped q_pe + the roped cache and pass IDENTITY rope tables (q_cos=ones,
q_sin=zeros, c_cos=ones, c_sin=zeros), so the in-kernel rope is out = x*1 + rotate(x)*0 = x.
This isolates the absorb->gather->sdpa->V-up residency win, byte-exact. (A later option 1b
could fold the real rope in-kernel and drop the upstream rope — NOT done here.)

ENGAGEMENT — DECODE B>=1 (B==1 validated, B>1 NEW), env-gated, default OFF:
  * HPU_MLA_MEGAKERNEL unset  -> install() returns immediately, touches NOTHING; the stock
    HPUMLAAttention.forward_impl runs byte-for-byte. (BYTE-EXACT contract, same as
    layer_prune_patch.)
  * HPU_MLA_MEGAKERNEL=1 -> patch HPUMLAAttention.forward_impl. The megakernel engages on any
    decode step -- B==1 (single-stream, the validated relNorm=0 path) AND B>1 (concurrency, NEW).
    PREFILL is untouched. At B==1 block_table[0] is the whole padded block_list; at B>1 each query's
    blocks (block_groups==b, SCATTERED by physical id under contiguous-PA) are COMPACTED into their
    own [B,MB_max] row via a device-side cumsum scatter (capture-safe) — see
    _decode_block_table_and_mask. So the served result is either the megakernel or, if a precondition
    is unmet / VERIFY trips, byte-exact stock. (First B>1 bring-up MUST run with VERIFY=1.)
  * HPU_MLA_MEGAKERNEL_VERIFY=1 -> per decode step ALSO run the stock forward_mqa+V-up and
    assert relNorm(mega, stock) < 1e-2; LOUD on mismatch and RETURN STOCK (stay coherent);
    else return the megakernel. Mirrors the [fused-monolithic-verify] harness in
    moe_wna16_hpu.py. NOTE: the relNorm .item() forces a per-step host sync that BREAKS lazy
    HPU-graph capture and slows the serve — this is the CORRECTNESS GATE only; measure perf
    with VERIFY unset.

THE flat -> block_table/pad_mask MAPPING (the single biggest correctness risk — get the
paged gather layout wrong and it gathers garbage):
  The serve carries a FLAT decode layout (get_habana_paged_attn_buffers, hpu_model_runner.py
  :2261-2306): block_list[g] (physical block ids, pad=_PAD_BLOCK_ID=num_blocks), block_groups
  [g] (the decode query each block belongs to, pad=-1), block_usage[g] (valid slots in block
  g: block_size for full blocks, last_block_usage=slot%bs+1 for the tail, pad=1). flat_pa_mla
  (extension/ops.py:157-206) gathers key = keys_fetch_func(kv_cache[0].unflatten(0,(-1,bs)),
  block_list) and applies the additive block_bias (attn_bias) per gathered block.
  The kernel instead wants a per-query block_table[B,MB] (oracle paged_block_gather_table:
  flat_cache.reshape(NB,bs*LR); gather rows by block_table.flatten()) + an additive pad_mask
  [B,1,1,S], S=MB*bs. At B==1 every real block belongs to query 0, so block_table[0] = the
  whole padded block_list (pads land on the in-range _PAD_BLOCK_ID=num_blocks pad block, which
  the cache allocates +1 for and which the mask fully suppresses), and the per-row valid mask
  is exactly (block_groups==0) & (within-block pos < block_usage) — FULLY DEVICE-SIDE, no
  host sync, capture-safe. Attention is permutation-invariant over keys (keys/values gathered
  together + mask gathered with them), so the physical-block-id order of contiguous-PA does
  not change the softmax result. flat_cache = kv_cache[0] directly ([NB_total*bs, L+R], the
  SAME tensor latent_cache_k wrote to). VERIFIED against capture_mla_island.py:611-647 (the
  on-card paged op-call) and hpu_attn.py forward_mqa:309-336.
  AT B>1 (CONCURRENCY): with use_contiguous_pa (the serve's mode -- VERIFIED hpu_model_runner.py
  :2274-2306), get_habana_paged_attn_buffers re-indexes the flat arrays BY PHYSICAL BLOCK ID
  (padding_fn = gather_list(tensor, indices, pad), indices[physical_id]=flat_pos), so a query's
  blocks are NOT contiguous -- block_groups[g] just names the owner of physical block g (or -1).
  So we cannot slice contiguous runs; we GROUP by block_groups==b and COMPACT each query's blocks
  into its own row of a rectangular block_table[B,MB_max] (MB_max = the bucketed flat length, the
  only capture-safe upper bound) via a per-query running column index (exclusive cumsum of the
  per-query one-hot, then scatter). Each row holds EXACTLY that query's blocks (structural cross-
  sequence isolation); short rows are pad_block_id-padded and the mask suppresses them. Same
  permutation-invariance => compaction is exact. (Width MB_max=G means more masked SDPA positions
  than a tight per-query bucket would; the win at B>1 is HOST-LAUNCH consolidation -- one fused op
  vs the ~11-op chain -- which the host-orchestration-bound decode is gated on. A tighter per-query
  MB_max bucket is the follow-up perf lever; correctness is independent of it.)

GUARDRAILS (DSA build quality bar — fail LOUD, never silently corrupt):
  * a CONTRACT violation once engaged (kv_cache[0] not 2D [NB*bs, L+R], L+R mismatch,
    block_size mismatch, NB not whole) -> RuntimeError (LOUD). A wrong gather would silently
    corrupt every token.
  * an ENGAGE precondition not met (op .so missing, prefill, no paged block metadata, no KV cache)
    -> quiet by-design fallback to the EXACT stock path (correct, never a wrong answer).
  * a B>1 SCATTER violation (a query's row would miss its own block or hold another's) -> LOUD
    RuntimeError under strict/VERIFY (the structural net); the per-step VERIFY relNorm gate is the
    runtime backstop -> LOUD + return stock.
  * VERIFY mismatch -> LOUD warning + return stock (coherent).

BYTE-EXACTNESS: install() reassigns nothing when HPU_MLA_MEGAKERNEL is unset. When set, the
patched forward_impl is a faithful copy of the stock decode (the cache write + stock absorb +
forward_mqa + V-up are byte-identical) with the engaged B==1 branch swapping ONLY the decode
output computation. Prefill path is the stock forward_mha, untouched.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm")

_ENV = "HPU_MLA_MEGAKERNEL"
_ENV_VERIFY = "HPU_MLA_MEGAKERNEL_VERIFY"

# bf16 norm-relative tol the megakernel was validated against (test_mla_decode_op uses 2e-2
# vs an fp64 oracle; here we compare bf16-vs-bf16 against the live stock path so 1e-2 is the
# tight bar, matching the [fused-monolithic-verify] tol).
_REL_TOL = 1e-2

_VERIFY = os.environ.get(_ENV_VERIFY) == "1"

# the op .so is loaded once, lazily, on the first engaged decode (so a no-op install never
# touches torch.ops / never dlopen's anything).
_so_loaded = [False]

# module state: once-per-worker engage banner + verify-under-lazy warning.
_state: dict = {"engaged_logged": False, "lazy_verify_warned": False}

# identity-rope tensor cache, keyed by (B, S, R, device-str, dtype) -> (q_cos,q_sin,c_cos,c_sin).
_rope_cache: dict = {}


# ==================================================================================================
# .so load — mirrors fused_monolithic_moe._load_monolithic_op (lazy .so on the serving path).
# ==================================================================================================
def _load_mla_paged_op():
    """Load libmla_decode_lazy.so (lazy serve) / libmla_decode.so (eager) once and return
    torch.ops.hpu_mla_ext.mla_decode_paged, or None if the .so is absent (caller falls back to
    the stock MLA decode). Idempotent."""
    if hasattr(torch.ops, "hpu_mla_ext") and hasattr(torch.ops.hpu_mla_ext, "mla_decode_paged"):
        return torch.ops.hpu_mla_ext.mla_decode_paged
    if _so_loaded[0]:
        return None
    _so_loaded[0] = True
    try:
        import habana_frameworks.torch.core as htorch
        is_lazy = htorch.is_lazy()
    except Exception:  # noqa: BLE001 -- off-card / import failure: no megakernel.
        is_lazy = False
    name = "libmla_decode_lazy.so" if is_lazy else "libmla_decode.so"
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "mla", "build", name))
    if not os.path.isfile(so):
        logger.warning(
            "[MLA-MEGAKERNEL] %s set but the op .so is missing (%s) -- the stock MLA decode "
            "(forward_mqa) runs unchanged.", _ENV, so)
        return None
    torch.ops.load_library(so)
    if hasattr(torch.ops, "hpu_mla_ext") and hasattr(torch.ops.hpu_mla_ext, "mla_decode_paged"):
        logger.warning(
            "[MLA-MEGAKERNEL] loaded fused MLA-decode op from %s (lazy=%s) -- SRAM-resident "
            "absorb->rope->paged-gather->sdpa->V-up; engages at decode B==1.", so, is_lazy)
        return torch.ops.hpu_mla_ext.mla_decode_paged
    logger.warning("[MLA-MEGAKERNEL] load_library(%s) did not register mla_decode_paged.", so)
    return None


# ==================================================================================================
# PURE input builders (card-free testable on CPU tensors; no self, no hpu, no .item()).
# ==================================================================================================
def _identity_rope_tensors(B: int, S: int, R: int, device, dtype):
    """Option-1a IDENTITY rope tables so the in-kernel rope is a no-op (out = x*1 + rot(x)*0).
       q_cos/q_sin: [B,1,R] (broadcast over heads); c_cos/c_sin: [B,S,R]. Cached per key."""
    key = (int(B), int(S), int(R), str(device), dtype)
    hit = _rope_cache.get(key)
    if hit is not None:
        return hit
    q_cos = torch.ones((B, 1, R), device=device, dtype=dtype)
    q_sin = torch.zeros((B, 1, R), device=device, dtype=dtype)
    c_cos = torch.ones((B, S, R), device=device, dtype=dtype)
    c_sin = torch.zeros((B, S, R), device=device, dtype=dtype)
    _rope_cache[key] = (q_cos, q_sin, c_cos, c_sin)
    return _rope_cache[key]


def _decode_block_table_and_mask(block_list, block_groups, block_usage, B, block_size, neg_big,
                                 pad_block_id=0, strict=False):
    """Build the kernel's per-query block_table[B,MB] int32 + additive pad_mask[B,1,1,S] from
    the serve's FLAT decode layout (block_list/block_groups/block_usage, all 1-D length MB_flat).

    DEVICE-SIDE, capture-safe (NO .item()/.tolist()/.cpu()/data-dependent shapes -- every shape is
    a Python int fixed by the decode BUCKET), works on BOTH B==1 (single-stream) and B>1 (concurrency).

      * B==1 (UNCHANGED -- the validated relNorm=0 path): query 0 owns every real block, so
        block_table[0] is the WHOLE padded block_list (pads kept IN PLACE, masked off) and the per-row
        valid mask is (block_groups==0) & (within-block pos < block_usage). Byte-identical to before.
      * B>1 (NEW): the serve's contiguous-PA flat arrays are indexed BY PHYSICAL BLOCK ID (VERIFIED:
        get_habana_paged_attn_buffers' gather_list re-index, hpu_model_runner.py:2274-2306), so a query's
        blocks are SCATTERED across the flat array, NOT contiguous. We group by block_groups==b and
        COMPACT each query's blocks to the front of its OWN row [B, MB_max] (MB_max = the bucketed flat
        length, a safe upper bound: a query owns at most ALL blocks) using a per-query running column
        index = exclusive cumsum of the per-query one-hot. Short rows are padded with pad_block_id (an
        IN-RANGE block, mask-suppressed). Each row b therefore contains EXACTLY query b's blocks --
        STRUCTURAL cross-sequence isolation: a query never gathers another query's keys. The additive
        mask is, per table slot (b,c) and within-block pos p: 0 iff that slot holds a real owned block
        AND p < its usage, else neg_big. Permutation-invariance over keys (keys+mask gathered together)
        makes the per-query block ORDER irrelevant, so compaction is exact.

    Returns (block_table[B,MB] int32, pad_mask[B,1,1,S] dtype=neg_big.dtype, S=MB*block_size) or None if
    the flat layout is absent. Raises RuntimeError (LOUD) on a contract violation; with strict=True
    (VERIFY/self-test only -- it SYNCS) also LOUD-asserts the B>1 scatter placed every owned block in
    its own row exactly once."""
    if block_list is None or block_groups is None or block_usage is None:
        return None
    if block_list.dim() != 1 or block_groups.dim() != 1 or block_usage.dim() != 1:
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] expected 1-D flat block_list/groups/usage, got "
            f"{tuple(block_list.shape)}/{tuple(block_groups.shape)}/{tuple(block_usage.shape)}.")
    MB_flat = block_list.shape[0]
    if block_groups.shape[0] != MB_flat or block_usage.shape[0] != MB_flat:
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] flat block layout length mismatch: list={MB_flat} "
            f"groups={block_groups.shape[0]} usage={block_usage.shape[0]}.")
    Bi = int(B)
    if Bi < 1:
        raise RuntimeError(f"[MLA-MEGAKERNEL] invalid decode batch B={Bi}.")
    bs = int(block_size)
    dev = block_list.device

    # ---- B==1: EXACT validated path (byte-identical; DO NOT ALTER) ----
    if Bi == 1:
        MB = MB_flat
        S = MB * bs
        # block_table: the whole padded block_list as query 0's blocks (pads land on the in-range
        # _PAD_BLOCK_ID pad block; the mask below fully suppresses them).
        block_table = block_list.reshape(1, MB).to(torch.int32)
        # per-(block, within-block-pos) validity:  owned by q0  AND  pos < usage.
        pos = torch.arange(bs, device=dev, dtype=torch.int32).reshape(1, bs)      # [1, bs]
        usage = block_usage.reshape(MB, 1).to(torch.int32)                        # [MB,1]
        owned = (block_groups.reshape(MB, 1) == 0)                                # [MB,1] bool
        valid = owned & (pos < usage)                                            # [MB, bs] bool
        zero = torch.zeros((), device=dev, dtype=neg_big.dtype)
        pad_mask = torch.where(valid, zero, neg_big).reshape(1, 1, 1, S)         # [1,1,1,S]
        return block_table, pad_mask

    # ---- B>1: per-query compaction (device-side, capture-safe) ----
    G = MB_flat
    MB_max = MB_flat                              # safe upper bound: a query owns <= all G blocks.
    # TIGHT-MB CAP (the B>1 quadratic-slowdown fix): MB_flat is the batch-SUM of blocks, so compacting
    # to [B, MB_flat] makes every query's SDPA run over ALL batch keys = O(B * total_KV) (measured 3-7x
    # regression @c32-c128). A query owns at most ceil(max_model_len/block_size) blocks, which is <<
    # MB_flat at concurrency. HPU_MLA_MEGAKERNEL_MB_CAP=<int> sets MB_max to that tight, capture-safe
    # Python int (the CPU gate ALREADY validates the tight-MB path byte-exact at MB=2/3/4). CONTRACT:
    # the cap MUST be >= the true max per-query block count (= ceil(max_model_len/block_size)); a query
    # owning more would collide via the col-clamp below. Left at MB_flat (0/unset) = the always-safe
    # loose bound. e.g. max_model_len=1024, block_size=128 -> cap=9 (8 blocks + margin).
    _mb_cap = int(os.environ.get("HPU_MLA_MEGAKERNEL_MB_CAP", "0") or "0")
    if _mb_cap > 0 and _mb_cap < MB_flat:
        MB_max = _mb_cap
    S = MB_max * bs
    b_ar = torch.arange(Bi, device=dev, dtype=torch.long).reshape(1, Bi)         # [1,B]
    grp = block_groups.reshape(G, 1).to(torch.long)                             # [G,1] owner (>=0) or -1
    onehot = (grp == b_ar).to(torch.int32)                                      # [G,B] (pad rows all 0)
    # per-query running column index = EXCLUSIVE prefix sum of the one-hot down the flat axis.
    excl = torch.cumsum(onehot, dim=0) - onehot                                 # [G,B]
    col = (excl * onehot).sum(dim=1).to(torch.long)                             # [G] col within own query
    col = torch.clamp(col, max=MB_max - 1)                                      # defensive no-op (col<G=MB_max)
    owner = grp.reshape(G)                                                       # [G] long, -1 for pad
    is_real = (owner >= 0) & (owner < Bi)                                        # block has a query in THIS batch
    scratch = Bi * MB_max                                                        # junk slot for pad/foreign rows
    dest = torch.where(is_real, owner.clamp(min=0, max=Bi - 1) * MB_max + col,
                       torch.full_like(col, scratch))                            # [G] long, always in [0,scratch]
    # scatter physical block ids -> [B, MB_max]; empty/short-row slots = pad_block_id (in-range, masked).
    fill = int(pad_block_id)
    flat_tab = torch.full((Bi * MB_max + 1,), fill, device=dev, dtype=torch.int32)
    flat_tab.scatter_(0, dest, block_list.reshape(G).to(torch.int32))
    block_table = flat_tab[:Bi * MB_max].reshape(Bi, MB_max)                     # [B,MB_max] int32
    # scatter usages the SAME way; empty/short-row slots keep usage 0 -> fully masked.
    flat_use = torch.zeros((Bi * MB_max + 1,), device=dev, dtype=torch.int32)
    flat_use.scatter_(0, dest, block_usage.reshape(G).to(torch.int32))
    usage_table = flat_use[:Bi * MB_max].reshape(Bi, MB_max)                     # [B,MB_max] int32
    pos = torch.arange(bs, device=dev, dtype=torch.int32).reshape(1, 1, bs)      # [1,1,bs]
    valid = pos < usage_table.reshape(Bi, MB_max, 1)                             # [B,MB_max,bs] bool
    zero = torch.zeros((), device=dev, dtype=neg_big.dtype)
    pad_mask = torch.where(valid, zero, neg_big).reshape(Bi, 1, 1, S)            # [B,1,1,S]

    if strict:
        # LOUD structural guard (SYNCS -- only under VERIFY/self-test): every owned block must land in
        # its OWN row exactly once (no drop, no collision, no cross-query bleed). A real block always
        # has usage>=1, an empty/pad slot has usage 0, so populated-per-row == owned-per-query iff the
        # scatter is correct. A failure here would be CATASTROPHIC cross-sequence attention.
        cnt_groups = onehot.sum(dim=0).to(torch.long)                           # [B] owned blocks per query
        cnt_table = (usage_table > 0).sum(dim=1).to(torch.long)                 # [B] populated slots per row
        if not bool((cnt_groups == cnt_table).all().item()):
            raise RuntimeError(
                f"[MLA-MEGAKERNEL] B>1 block scatter MISMATCH: owned-per-query from block_groups="
                f"{cnt_groups.tolist()} != populated-per-row in block_table={cnt_table.tolist()} -- a "
                f"query's row would miss its own block or hold another query's (CATASTROPHIC cross-"
                f"sequence attention). Refusing to gather.")
    return block_table, pad_mask


# ==================================================================================================
# the engaged decode forward (B==1).  Returns [B, N*V] or None (precondition -> stock fallback);
# raises RuntimeError on a contract violation.
# ==================================================================================================
def _run_megakernel(self, q_nope, q_pe, kv_cache, attn_metadata):
    op = _load_mla_paged_op()
    if op is None:
        return None
    B = q_nope.shape[0]
    if kv_cache is None or len(kv_cache) < 2 or kv_cache[0] is None:
        return None  # no KV cache (e.g. warmup dummy) -> stock.

    N = self.num_heads
    P = self.qk_nope_head_dim
    R = self.qk_rope_head_dim
    L = self.kv_lora_rank
    V = self.v_head_dim
    block_size = int(attn_metadata.block_size)

    # ---- flat_cache = kv_cache[0] : [NB_total*block_size, L+R] (CONTRACT-checked, LOUD) ----
    flat_cache = kv_cache[0]
    if flat_cache.dim() != 2:
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] kv_cache[0] must be 2-D [NB*bs, L+R]; got {tuple(flat_cache.shape)}.")
    if flat_cache.shape[1] != L + R:
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] kv_cache[0] last dim {flat_cache.shape[1]} != kv_lora+qk_rope "
            f"{L + R} -- latent cache layout mismatch (gather would read garbage).")
    if flat_cache.shape[0] % block_size != 0:
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] kv_cache[0] rows {flat_cache.shape[0]} not a multiple of "
            f"block_size {block_size}.")

    # ---- block_table[B,MB] + pad_mask[B,1,1,S] from the flat decode layout (device-side) ----
    # in-range pad block for short-row padding at B>1 == the serve's _PAD_BLOCK_ID (= num_blocks =
    # NB_total-1); the cache allocates this extra block (_PAD_SLOT_ID = num_blocks*bs) and the mask
    # fully suppresses it. Shape-derived -> a Python int, capture-safe. (Unused on the B==1 path.)
    pad_block = flat_cache.shape[0] // block_size - 1
    neg_big = torch.tensor(torch.finfo(torch.bfloat16).min, device=flat_cache.device,
                           dtype=torch.bfloat16)
    bt_pm = _decode_block_table_and_mask(
        getattr(attn_metadata, "block_list", None),
        getattr(attn_metadata, "block_groups", None),
        getattr(attn_metadata, "block_usage", None),
        B, block_size, neg_big, pad_block_id=pad_block, strict=_VERIFY)
    if bt_pm is None:
        return None  # no paged block metadata (not a normal paged decode) -> stock.
    block_table, pad_mask = bt_pm
    MB = block_table.shape[1]
    S = MB * block_size

    # ---- identity rope (Option 1a) + bf16 kernel inputs ----
    q_cos, q_sin, c_cos, c_sin = _identity_rope_tensors(B, S, R, flat_cache.device, torch.bfloat16)
    q_nope_k = q_nope.contiguous().to(torch.bfloat16)         # [B,N,P]
    q_pe_k = q_pe.contiguous().to(torch.bfloat16)             # [B,N,R] (already roped in serve)
    flat_cache_k = flat_cache if flat_cache.dtype == torch.bfloat16 else flat_cache.to(torch.bfloat16)
    W_UK_T = self.W_UK_T.contiguous().to(torch.bfloat16)      # [N,P,L]
    W_UV = self.W_UV.contiguous().to(torch.bfloat16)          # [N,L,V]
    scale = float(self.scale)
    interleave = 1                                            # GLM = PAIRWISE (no-op under 1a)

    if q_nope_k.shape != (B, N, P) or q_pe_k.shape != (B, N, R):
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] q split shape mismatch: q_nope {tuple(q_nope_k.shape)} "
            f"(want [{B},{N},{P}]), q_pe {tuple(q_pe_k.shape)} (want [{B},{N},{R}]).")

    out = op(q_nope_k, q_pe_k, flat_cache_k, block_table, W_UK_T, W_UV,
             q_cos, q_sin, c_cos, c_sin, pad_mask, scale, interleave, block_size)
    if list(out.shape) != [B, N * V]:
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] op returned {list(out.shape)}, expected [{B}, {N * V}].")

    if not _state["engaged_logged"]:
        rank = -1
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("[MLA-MEGAKERNEL] engaged rank=%d (decode B=%d, S=%d, MB=%d, verify=%s)",
                       rank, B, S, MB, _VERIFY)
        _state["engaged_logged"] = True
    return out


# ==================================================================================================
# INSTALL — monkeypatch HPUMLAAttention.forward_impl. No-op (stock) when HPU_MLA_MEGAKERNEL unset.
# ==================================================================================================
def install() -> None:
    if os.environ.get(_ENV) != "1":
        return  # BYTE-EXACT: env unset -> touch nothing; stock forward_impl runs unchanged.

    try:
        from vllm_gaudi.attention.oot_mla import HPUMLAAttention
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(
            f"[MLA-MEGAKERNEL] {_ENV}=1 but vllm_gaudi.attention.oot_mla.HPUMLAAttention is "
            f"unavailable: {ex!r}")

    if getattr(HPUMLAAttention.forward_impl, "_hpu_mla_megakernel", False):
        return  # already patched (idempotent)

    _orig_forward_impl = HPUMLAAttention.forward_impl

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                     output=None, output_scale=None, output_block_scale=None):
        is_prefill = attn_metadata.is_prompt

        # ---- decode: split q (q_pe is ALREADY roped upstream — Option 1a) ----
        if not is_prefill:
            q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---- WRITE current-token latent+rope to kv_cache[0] (BYTE-IDENTICAL to stock; the
        #      kernel's NODE-0 gather reads this) ----
        slot_mapping = attn_metadata.slot_mapping.flatten() \
            if attn_metadata.slot_mapping is not None else None
        latent_vec_k = torch.concat(
            (k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
        latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
        if kv_cache is not None and len(kv_cache) >= 2:
            self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)

        # ---- PREFILL: stock forward_mha, untouched ----
        if is_prefill:
            return self.impl.forward_mha(q, latent_vec_k, kv_cache, attn_metadata)

        # ---- DECODE output ----
        mega = _run_megakernel(self, q_nope, q_pe, kv_cache, attn_metadata)
        if mega is not None and not _VERIFY:
            return mega

        # stock decode (reference for VERIFY, or the fallback when not engaged) — byte-identical
        # to the upstream decode branch (absorb -> forward_mqa -> V-up).
        decode_ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
        stock = self._v_up_proj(self.impl.forward_mqa(decode_ql_nope, q_pe, kv_cache, attn_metadata))
        if mega is None:
            return stock

        # ---- VERIFY: A/B (relNorm < tol); LOUD + return stock on mismatch (stay coherent) ----
        if not _state["lazy_verify_warned"]:
            try:
                import habana_frameworks.torch.core as _ht
                if _ht.is_lazy():
                    logger.warning(
                        "[MLA-MEGAKERNEL-VERIFY] relNorm .item() forces a per-step host sync that "
                        "BREAKS lazy HPU-graph capture and slows the serve -- CORRECTNESS GATE only; "
                        "measure perf with %s unset.", _ENV_VERIFY)
            except Exception:  # noqa: BLE001
                pass
            _state["lazy_verify_warned"] = True
        diff = (mega.float() - stock.float()).norm().item()
        rn = diff / (stock.float().norm().item() + 1e-9)
        if rn >= _REL_TOL:
            ma = (mega.float() - stock.float()).abs().max().item()
            logger.warning(
                "[MLA-MEGAKERNEL-VERIFY] MISMATCH relNorm=%.4g maxAbs=%.4g B=%d -- returning "
                "stock forward_mqa result (stay coherent). Suspect the paged gather layout "
                "(block_table/pad_mask from the flat block_list/groups/usage).", rn, ma, q_nope.shape[0])
            return stock
        logger.info("[MLA-MEGAKERNEL-VERIFY] relNorm=%.6g B=%d", rn, q_nope.shape[0])
        return mega

    forward_impl._hpu_mla_megakernel = True
    forward_impl._hpu_orig_forward_impl = _orig_forward_impl
    HPUMLAAttention.forward_impl = forward_impl
    logger.warning(
        "[MLA-MEGAKERNEL] %s=1 ACTIVE -- patched HPUMLAAttention.forward_impl (decode B>=1 "
        "SRAM-resident megakernel; B==1 validated, B>1 NEW -- run VERIFY=1 first; verify=%s). "
        "ON-CARD COHERENCE GATE REQUIRED.", _ENV, _VERIFY)


# ==================================================================================================
# Card-free self-test:  python -m hpu_moe_ext.mla_megakernel_patch --selftest
#   Exercises the PURE input builders (identity rope + the flat->block_table/pad_mask mapping)
#   on CPU tensors — no habana / no hpu / no .so.
# ==================================================================================================
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  ok:   {msg}")

    dev = torch.device("cpu")
    R = 64
    bs = 4   # tiny block_size for the test

    # ---- identity rope: shapes + values (ones / zeros) ----
    B, S = 1, 12
    q_cos, q_sin, c_cos, c_sin = _identity_rope_tensors(B, S, R, dev, torch.float32)
    check(tuple(q_cos.shape) == (B, 1, R) and tuple(q_sin.shape) == (B, 1, R),
          f"q_cos/q_sin shape [B,1,R] (got {tuple(q_cos.shape)})")
    check(tuple(c_cos.shape) == (B, S, R) and tuple(c_sin.shape) == (B, S, R),
          f"c_cos/c_sin shape [B,S,R] (got {tuple(c_cos.shape)})")
    check(bool((q_cos == 1).all()) and bool((c_cos == 1).all()), "cos == ones (identity)")
    check(bool((q_sin == 0).all()) and bool((c_sin == 0).all()), "sin == zeros (identity)")
    # cache returns the SAME objects on a repeat key
    again = _identity_rope_tensors(B, S, R, dev, torch.float32)
    check(again[0] is q_cos and again[2] is c_cos, "identity-rope cache hit returns same tensors")

    # ---- flat -> block_table/pad_mask (B==1).  Build a realistic contiguous-PA flat layout:
    #      query 0 owns 3 real blocks (full, full, tail-usage=2) + 1 pad block. ----
    neg = torch.tensor(torch.finfo(torch.bfloat16).min, device=dev, dtype=torch.bfloat16)
    PAD_BLOCK = 99
    block_list = torch.tensor([5, 6, 7, PAD_BLOCK], dtype=torch.long)     # 3 real + 1 pad
    block_groups = torch.tensor([0, 0, 0, -1], dtype=torch.long)          # q0,q0,q0,pad
    block_usage = torch.tensor([bs, bs, 2, 1], dtype=torch.bfloat16)      # full,full,tail=2,pad
    res = _decode_block_table_and_mask(block_list, block_groups, block_usage, 1, bs, neg)
    check(res is not None, "B==1 returns a (block_table, pad_mask)")
    block_table, pad_mask = res
    MB = block_list.shape[0]
    S2 = MB * bs
    check(tuple(block_table.shape) == (1, MB) and block_table.dtype == torch.int32,
          f"block_table [1,MB] int32 (got {tuple(block_table.shape)} {block_table.dtype})")
    check(bool((block_table[0] == block_list.to(torch.int32)).all()),
          "block_table[0] == the whole padded block_list (pads kept, masked off)")
    check(tuple(pad_mask.shape) == (1, 1, 1, S2),
          f"pad_mask [1,1,1,S=MB*bs] (got {tuple(pad_mask.shape)})")
    # expected validity: block0 all bs valid, block1 all bs valid, block2 first 2 valid,
    # block3 (pad, group=-1) entirely masked.
    flat = pad_mask.reshape(MB, bs)
    expect_valid = torch.zeros(MB, bs, dtype=torch.bool)
    expect_valid[0, :bs] = True
    expect_valid[1, :bs] = True
    expect_valid[2, :2] = True
    got_valid = (flat == 0)
    got_masked = (flat == neg.to(flat.dtype))
    check(bool((got_valid == expect_valid).all()), "pad_mask keeps exactly the valid slots (0)")
    check(bool((got_masked == ~expect_valid).all()), "pad_mask masks exactly the rest (NEG_BIG)")
    check(int(got_valid.sum()) == bs + bs + 2, f"valid slot count == {bs+bs+2} (got {int(got_valid.sum())})")

    # ============================================================================================
    # B>1 (CONCURRENCY): the contiguous-PA flat arrays are indexed BY PHYSICAL BLOCK ID, so a query's
    # blocks are SCATTERED. Verify the cumsum-scatter COMPACTS each query's blocks into its OWN row,
    # masks per-query usage exactly, gives STRUCTURAL cross-query isolation, and that queries with
    # DIFFERENT block counts + tails + pads all land correctly. (per_phys[pid] = (owner, usage); owner
    # = -1 marks a free/pad physical slot -> block_list=PAD_BLOCK, group=-1, usage=1.)
    # ============================================================================================
    def _build_flat(per_phys, pad_block):
        bl, bg, bu, owned = [], [], [], {}
        for pid, (owner, usage) in enumerate(per_phys):
            if owner < 0:
                bl.append(pad_block); bg.append(-1); bu.append(1)
            else:
                bl.append(pid); bg.append(owner); bu.append(usage)
                owned.setdefault(owner, []).append((pid, usage))   # ASCENDING physical-id == flat order
        return (torch.tensor(bl, dtype=torch.long), torch.tensor(bg, dtype=torch.long),
                torch.tensor(bu, dtype=torch.bfloat16), owned, len(per_phys))

    def _check_layout(tag, Bn, per_phys, pad_block):
        bl, bg, bu, owned, G = _build_flat(per_phys, pad_block)
        # strict=True exercises the LOUD structural guard on CPU (must NOT false-trip on valid data).
        res = _decode_block_table_and_mask(bl, bg, bu, Bn, bs, neg, pad_block_id=pad_block, strict=True)
        check(res is not None, f"{tag}: returns (block_table, pad_mask)")
        bt, pm = res
        check(tuple(bt.shape) == (Bn, G) and bt.dtype == torch.int32,
              f"{tag}: block_table [B,MB_max=G] int32 (got {tuple(bt.shape)} {bt.dtype})")
        check(tuple(pm.shape) == (Bn, 1, 1, G * bs), f"{tag}: pad_mask [B,1,1,S=G*bs] (got {tuple(pm.shape)})")
        all_pids = {pid for lst in owned.values() for (pid, _u) in lst}
        mask = (pm.reshape(Bn, G, bs) == 0)                        # [B,G,bs] True where valid (additive 0)
        attended = []
        for b in range(Bn):
            mine = owned.get(b, [])
            k = len(mine)
            row = bt[b].tolist()
            check(row[:k] == [pid for pid, _u in mine],
                  f"{tag} q{b}: row front == own blocks {[pid for pid,_u in mine]} (got {row[:k]})")
            check(all(x == pad_block for x in row[k:]), f"{tag} q{b}: row tail all PAD_BLOCK({pad_block})")
            # STRUCTURAL isolation: the row holds NONE of any OTHER query's physical blocks.
            others = all_pids - {pid for pid, _u in mine}
            check(not (set(row) & others), f"{tag} q{b}: row contains NO other query's block (isolation)")
            # per-(col,pos) mask == own per-block usage exactly; pad/empty cols fully masked.
            exp = torch.zeros(G, bs, dtype=torch.bool)
            for c, (_pid, usage) in enumerate(mine):
                exp[c, :int(usage)] = True
            check(bool((mask[b] == exp).all()), f"{tag} q{b}: per-query usage mask EXACT (valid={int(exp.sum())})")
            # every VALID column gathers ONLY this query's blocks (no cross-seq key ever exposed).
            cols = {bt[b, c].item() for c in range(G) if bool(mask[b, c].any())}
            check(cols <= {pid for pid, _u in mine}, f"{tag} q{b}: valid cols gather ONLY own blocks (got {cols})")
            attended.append(cols)
        return bt, mask, owned, attended

    # B=2: q0 owns 2 blocks (full + tail=3), q1 owns 3 (full,full,tail=2); SCATTERED among free slots.
    #   phys: 0:free 1:free 2:q0 3:q1 4:q1 5:q0(tail3) 6:free 7:q1(tail2) 8:free 9:free
    per2 = [(-1, 0), (-1, 0), (0, bs), (1, bs), (1, bs), (0, 3), (-1, 0), (1, 2), (-1, 0), (-1, 0)]
    _bt2, m2, _ow2, att2 = _check_layout("B=2", 2, per2, PAD_BLOCK)
    check(int(m2[0].sum()) == bs + 3 and int(m2[1].sum()) == bs + bs + 2,
          f"B=2 valid counts q0={bs+3} q1={bs+bs+2} (got {int(m2[0].sum())},{int(m2[1].sum())})")
    check(att2[0] == {2, 5} and att2[1] == {3, 4, 7},
          f"B=2 isolation: q0 attends {{2,5}}={att2[0]}, q1 attends {{3,4,7}}={att2[1]}")
    check(not (att2[0] & att2[1]), "B=2 q0's mask NEVER exposes q1's blocks (disjoint attend sets)")

    # B=3: DIFFERENT counts (2,1,4) + tails + scattered + free slots.
    #   phys: 0:free 1:q0 2:q1 3:free 4:q0(tail2) 5:q2 6:q2 7:q2 8:free 9:q2(tail1) 10:free 11:free
    per3 = [(-1, 0), (0, bs), (1, bs), (-1, 0), (0, 2), (2, bs), (2, bs), (2, bs),
            (-1, 0), (2, 1), (-1, 0), (-1, 0)]
    _bt3, m3, _ow3, att3 = _check_layout("B=3", 3, per3, PAD_BLOCK)
    check(int(m3[0].sum()) == bs + 2 and int(m3[1].sum()) == bs and int(m3[2].sum()) == 3 * bs + 1,
          f"B=3 valid counts (got {int(m3[0].sum())},{int(m3[1].sum())},{int(m3[2].sum())})")
    check(att3[0] == {1, 4} and att3[1] == {2} and att3[2] == {5, 6, 7, 9}, f"B=3 attend sets {att3}")
    check(not (att3[0] & att3[1]) and not (att3[0] & att3[2]) and not (att3[1] & att3[2]),
          "B=3 attended blocks pairwise DISJOINT across all 3 queries (cross-seq isolation)")

    # ---- None metadata -> None ----
    check(_decode_block_table_and_mask(None, None, None, 1, bs, neg) is None,
          "missing flat metadata returns None")
    # ---- length mismatch -> LOUD RuntimeError (B==1 AND B>1) ----
    try:
        _decode_block_table_and_mask(block_list, block_groups[:2], block_usage, 1, bs, neg)
        check(False, "B==1 length-mismatched flat layout should raise")
    except RuntimeError:
        check(True, "B==1 length-mismatched flat layout raises RuntimeError (LOUD)")
    try:
        bl2, bg2, bu2, _o, _g = _build_flat(per2, PAD_BLOCK)
        _decode_block_table_and_mask(bl2, bg2[:5], bu2, 2, bs, neg, pad_block_id=PAD_BLOCK)
        check(False, "B=2 length-mismatched flat layout should raise")
    except RuntimeError:
        check(True, "B=2 length-mismatched flat layout raises RuntimeError (LOUD)")

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
