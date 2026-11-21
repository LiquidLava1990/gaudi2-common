"""HPU DSA MVP-3 — the GATED, op-backed indexer that actually FIRES on HPU.

Two coordinated monkeypatches, BOTH installed ONLY when ``HPU_DSA_SPARSE=1``
(default unset → this module is never imported on the serving path):

  1. ``Indexer.forward`` (vllm_core_deepseek_v2.Indexer) → an HPU-SAFE rewrite that
     runs the trained-weight projections + partial rope (the clean torch the in-image
     forward already does) but SKIPS the CUDA/triton ``per_token_group_quant_fp8`` step
     (the call that crashes on HPU: ``triton.next_power_of_2`` is absent when
     HAS_TRITON=False). q is consumed as BF16 with q_scale folded as 1.0 into
     ``weights`` — mathematically EXACT because ReLU is positive-homogeneous and
     q_scale>0 (the design's fp8-fold, sparse_indexer_hpu.py:13-17; HF's bf16 path uses
     q_scale=1 identically, modeling_glm_moe_dsa.py:204-216). It then calls the same
     ``self.indexer_op(hidden_states, q_bf16, k, weights)`` the original does, which on
     HPU dispatches to the op-backed ``forward_oot`` below.

  2. ``SparseAttnIndexer.forward_oot`` → builds the indexer-op inputs per request and
     calls the captured ``torch.ops.hpu_dsa_ext.indexer_topk`` kernel (the HW-validated
     .so), writing the int32 top-k indices into ``self.topk_indices_buffer``. This is
     the OP variant of the pure-torch ``sparse_indexer_hpu.forward_oot``: same metadata
     derivation, same per-request ABSOLUTE-order key gather, same K-cache insert — only
     the inner scorer is the GC-captured kernel instead of ``lightning_indexer_topk``.

WHY REPLACE ``Indexer.forward`` AT ALL (vs just forward_oot): the crash is INSIDE the
in-image ``Indexer.forward`` (the fp8 quant), strictly BEFORE ``forward_oot`` is
reached. Today the whole indexer is disabled (``hpu_mla_skip_indexer`` forces
``skip_topk=True``) precisely to dodge that crash. To let the indexer FIRE (so there is
a real ``topk_indices_buffer`` to feed ``sparse_attn``), MVP-3 must provide an HPU-safe
``Indexer.forward``. We keep the change minimal and behind the gate; with the gate off,
``hpu_mla_skip_indexer`` still forces skip_topk=True and the indexer never runs.

GATE INTERACTION (the OFF-by-default contract):
  * ``HPU_DSA_SPARSE`` UNSET  → this module's ``register()`` is a no-op; the indexer is
    skipped (skip_topk=True) and the consume is dense (FULL_ATTENTION_FALLBACK=True).
  * ``HPU_DSA_SPARSE=1``      → register() (a) flips ``platform_patch.FULL_ATTENTION_
    FALLBACK=False``, (b) UN-skips the indexer (skip_topk=False on every wrapper) so
    ``mla.py:169`` fires our safe forward, (c) installs the op-backed forward_oot, and
    (d) eagerly loads both .so (LOUD if missing). The model still needs HPU_DSA_READY=1
    to lift the platform raise (glm51ar-lazy.env already sets it).

Scope (MVP-3): prefill (no-prefix, the clean path) is the validated branch; decode
re-uses the same per-request gather. ctx ≤ index_topk: sparse == dense (every causal key
is selected), so the threshold below short-circuits to the dense path (documented).
"""
from __future__ import annotations

import os

import torch

# NOTE: ``dsa_ops`` and ``platform_patch`` are imported LAZILY inside the functions
# that need them (register / scoring), NOT at module top-level, so importing this
# module never pulls vllm_gaudi (platform_patch's dependency). This keeps the
# gate-off import path light and decoupled (the rest of the overlay uses the same
# lazy-import discipline).


# ---------------------------------------------------------------------------
# Threshold: the context length at/below which sparse == dense.
#
# The indexer keeps min(index_topk, S) keys. When S (the causal key count for a
# query) ≤ index_topk, EVERY causal key is selected, so sparse attention is
# IDENTICAL to dense causal MLA. There is then nothing to gain (and a tiny bf16
# round-trip to lose) from the sparse path, so for ctx ≤ index_topk we let the
# dense forward run. The indexer only "fires" (selects a strict subset) when a
# query has MORE than index_topk causal keys, i.e. abs_pos(t) + 1 > index_topk.
# We gate on the batch's max sequence length so a request whose longest query
# exceeds the threshold takes the sparse path. (Validation gate (b) asserts
# sparse==dense at/below the threshold; gate (c) asserts they DIFFER above it.)
# ---------------------------------------------------------------------------
def indexer_fires(max_seq_len: int, index_topk: int) -> bool:
    """True iff at least one query can have > index_topk causal keys (so the top-k is
    a strict subset and sparse != dense). Equivalently max_seq_len > index_topk."""
    return int(max_seq_len) > int(index_topk)


# ===========================================================================
# (1) HPU-safe Indexer.forward — projections + rope, NO fp8 quant.
# ===========================================================================
_ORIG_INDEXER_FORWARD = [None]   # set by register(); the in-image Indexer.forward


# torch.compiler.disable ON THE WHOLE INDEXER (not just the op call): dynamo
# otherwise compiles the resume-fragments AROUND the op (the post-filter, the
# buffer write, the projections), shape-agnostic-caches them, and a shape-changed
# cache hit dies in Synapse shape inference ("[SHAPE AGNOSTIC] Cache hit Synapse
# shape inference failed!"). The indexer must be a single eager island.
@torch.compiler.disable
def _hpu_indexer_forward(self, hidden_states: torch.Tensor, qr: torch.Tensor,
                         positions, rotary_emb) -> torch.Tensor:
    """HPU-safe replacement for ``Indexer.forward`` (vllm_core_deepseek_v2.py:678-743).

    Identical math to the original EXCEPT the fp8 quant (steps the original does at
    :728-741 are replaced by the q_scale=1 bf16 fold). Reuses the trained
    ``wq_b`` / ``wk_weights_proj`` / ``k_norm`` and the partial-rope, then calls
    ``self.indexer_op(hidden_states, q_bf16, k, weights)`` exactly as the original
    (:743), so the downstream dispatch (→ forward_oot) is unchanged.
    """
    # q projection: [T, n_head*head_dim] -> [T, n_head, head_dim]
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)

    # fused wk + weights_proj: one GEMM, split into k[head_dim] and weights[n_head].
    # Slice the LAST dim + flatten: under dynamo graph-capture hidden_states arrives
    # 3D [1, T, H] (fake-tensor batch dim), where a dim-1 slice would cut the TOKEN
    # axis -> k_norm(128) sees (1, 128, 160) and warmup crashes (the same 3D trap
    # documented in hpu_mla_skip_indexer.py for the in-image Indexer.forward).
    kw, _ = self.wk_weights_proj(hidden_states)
    kw = kw.reshape(-1, kw.shape[-1])              # [*, T, F] -> [T, F]
    k = kw[:, : self.head_dim]
    weights = kw[:, self.head_dim :]
    k = self.k_norm(k)

    # partial rope on the rope_dim slice of q (per head) and k (single MQA head).
    # rotary_emb signature matches the in-image call (:715): (positions, q_pe, k_pe).
    q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
    # rope may add leading dims under compile; reshape back to token-flattened shapes.
    q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
    k_pe = k_pe.reshape(-1, 1, self.rope_dim)
    q = torch.cat([q_pe, q_nope], dim=-1)                      # [T, n_head, head_dim]
    k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)          # [T, head_dim]  (MQA)

    # === the ONLY change vs the original: NO fp8 quant. ===
    # The original quantizes q to fp8 and folds q_scale into weights. On HPU we keep q
    # bf16 (q_scale == 1) and fold ONLY softmax_scale * n_head**-0.5 into weights —
    # exact because ReLU is positive-homogeneous (q_scale would cancel). HF's bf16 path
    # is identical (modeling_glm_moe_dsa.py:204-216).
    q_bf16 = q.contiguous().to(torch.bfloat16)
    weights = (weights.unsqueeze(-1) * self.softmax_scale * self.n_head ** -0.5).squeeze(-1)

    # Same terminal call as the original (:743) -> HPU dispatch -> forward_oot below.
    return self.indexer_op(hidden_states, q_bf16, k, weights)


# ===========================================================================
# (2) op-backed SparseAttnIndexer.forward_oot — calls the indexer_topk kernel.
#
# This mirrors sparse_indexer_hpu.forward_oot 1:1 (metadata derivation, the K-cache
# insert, per-request ABSOLUTE-order key gather, the reset-to-(-1) buffer) and swaps
# ONLY the inner scorer: lightning_indexer_topk(...) -> dsa_ops.indexer_topk_op(...).
# ===========================================================================
@torch.compiler.disable
def _op_score_segment(self, q_bf16, weights, keys, abs_pos, tok_start, tok_end):
    """Score one request's query rows with the indexer_topk KERNEL and write the int32
    top-k indices into self.topk_indices_buffer[tok_start:tok_end]. The op returns
    [tok, kk] (kk = min(topk, S_padded)); we write it into the (already -1-reset)
    buffer rows so the [kk, topk) tail stays -1.

    CAUSAL POST-FILTER (the contract's "-1 right-pad lives OUTSIDE the op",
    dsa_indexer_backend_shared.h:80-83): for a row with FEWER causal keys than kk,
    the in-recipe topk fills the tail with indices of -inf-masked (non-causal / pad)
    keys. The consume side masks ONLY -1 -- so without this filter those tail
    indices would be GATHERED and attended (future-token leakage on every prefill
    row t < kk). Replace every idx > abs_pos[t] with -1. This is exactly what the
    locked pure-torch scorer does internally."""
    from . import dsa_ops
    topk = self.topk_tokens
    idx = dsa_ops.indexer_topk_op(
        q_bf16[tok_start:tok_end], keys, weights[tok_start:tok_end], abs_pos, topk
    )  # [tok, kk] int32
    idx = torch.where(idx <= abs_pos.view(-1, 1).to(idx.dtype),
                      idx, torch.full_like(idx, -1))
    kk = idx.shape[1]
    self.topk_indices_buffer[tok_start:tok_end, :kk] = idx


# Gate (a) observability: a process-wide counter of op-backed indexer invocations.
# ctx>index_topk prompt => this MUST be > 0 (proves the indexer fired, not dense).
# Read via hpu_indexer_op_forward.indexer_fired_count() (or HPU_DSA_FIRED_LOG=1 logs it).
INDEXER_FIRED_COUNT = [0]
_FIRED_LOG = os.environ.get("HPU_DSA_FIRED_LOG", "0").strip().lower() in ("1", "true", "yes", "on")


def indexer_fired_count() -> int:
    """How many times the op-backed indexer forward_oot scored a real (non-dummy) batch
    this process. Gate (a): > 0 on a ctx>index_topk prompt proves the indexer fired."""
    return INDEXER_FIRED_COUNT[0]


@torch.compiler.disable
def _op_forward_oot(self, hidden_states, q_quant, k, weights):  # noqa: ANN001
    """OP variant of sparse_indexer_hpu.forward_oot. ``q_quant`` is BF16 here (the
    HPU-safe Indexer.forward folds q_scale=1), shape [T, n_head, head_dim]."""
    from . import sparse_indexer_hpu as base

    num_tokens = q_quant.shape[0]
    self.topk_indices_buffer[:num_tokens] = -1   # reset active rows to the -1 sentinel

    # CPU unit-test override path (no HPU metadata / cache): same hook the torch
    # forward_oot honors, so the op path is exercisable on a (device) synthetic harness.
    plan_override = getattr(self, "_dsa_plan_override", None)
    if plan_override is not None:
        for seg in plan_override:
            _op_score_segment(self, q_quant, weights, seg["keys"],
                              seg["abs_query_pos"], seg["token_start"], seg["token_end"])
        return self.topk_indices_buffer

    from vllm.forward_context import get_forward_context
    attn_metadata = get_forward_context().attn_metadata
    if attn_metadata is None:                      # profiling / dummy run -> leave -1
        return self.topk_indices_buffer

    cache_mod, meta_mod = base._resolve_plumbing()
    if cache_mod is None or meta_mod is None:      # pragma: no cover
        raise RuntimeError(
            "[hpu-dsa-ext] op forward_oot needs indexer_cache_hpu + indexer_metadata_hpu "
            "(landed runner patches). They are absent — refusing to silently mis-select.")

    meta = meta_mod.build_indexer_metadata(attn_metadata)
    if meta is None:
        return self.topk_indices_buffer
    active = meta.active
    is_decode = not meta.is_prompt
    if active.num_tokens == 0:
        return self.topk_indices_buffer

    kc = cache_mod.IndexerKCache(
        self.k_cache.kv_cache, block_size=active.block_size,
        index_head_dim=getattr(self, "head_dim", None))
    if not self.skip_k_cache_insert:
        kc.insert(k, active.slot_mapping)

    for r in range(active.batch_size):
        if is_decode:
            tok_start = r * active.next_n
            tok_end = tok_start + active.next_n
        else:
            tok_start = int(active.cu_query_lens[r].item())
            tok_end = int(active.cu_query_lens[r + 1].item())
        keys = base._gather_request_keys(kc, active, r, is_decode)   # [S_r, head_dim] bf16
        _op_score_segment(self, q_quant, weights, keys,
                          active.abs_pos[tok_start:tok_end].to(torch.long),
                          tok_start, tok_end)
        del keys

    # Gate (a): mark that the op-backed indexer scored a real batch this step.
    INDEXER_FIRED_COUNT[0] += 1
    if _FIRED_LOG:
        import logging
        logging.getLogger("vllm").info(
            "[hpu-dsa-ext] indexer FIRED (op) #%d: %s batch=%d tokens=%d topk=%d",
            INDEXER_FIRED_COUNT[0], "decode" if is_decode else "prefill",
            active.batch_size, active.num_tokens, self.topk_tokens)
    return self.topk_indices_buffer


# ===========================================================================
# install / uninstall
# ===========================================================================
def register() -> None:
    """Install the gated op-backed indexer. NO-OP unless HPU_DSA_SPARSE=1.

    Side effects when the flag is on:
      * platform_patch.FULL_ATTENTION_FALLBACK = False  (re-enable the sparse consume)
      * Indexer.forward          -> _hpu_indexer_forward (HPU-safe, no fp8)
      * SparseAttnIndexer.forward_oot -> _op_forward_oot (indexer_topk kernel)
      * un-skip the indexer on every HPUMultiHeadLatentAttentionWrapper (skip_topk=False)
      * eagerly load both DSA .so (LOUD if missing)
    Idempotent. Raises loudly on upstream drift (missing classes) — never silent."""
    from . import dsa_ops
    if not dsa_ops.sparse_enabled():
        return  # gate off -> dense bypass unchanged; do NOT import vllm_gaudi/core here

    # (a) re-enable the sparse consume side. hpu_mla_skip_indexer sets this True
    #     UNCONDITIONALLY at its own register(); we flip it back ONLY under the gate.
    from . import platform_patch
    platform_patch.FULL_ATTENTION_FALLBACK = False

    # (b) swap Indexer.forward for the HPU-safe (no-fp8) version.
    from vllm.model_executor.models.deepseek_v2 import Indexer
    if not getattr(Indexer.forward, "_hpu_dsa_op_forward", False):
        _ORIG_INDEXER_FORWARD[0] = Indexer.forward
        _hpu_indexer_forward._hpu_dsa_op_forward = True
        Indexer.forward = _hpu_indexer_forward

    # (c) install the op-backed forward_oot on the SparseAttnIndexer.
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    _op_forward_oot._hpu_dsa_op_forward = True
    SparseAttnIndexer.forward_oot = _op_forward_oot

    # (d) un-skip the indexer so mla.py:169 fires our safe forward. The skip is set in
    #     two places: the wrapper instances' self.skip_topk (set True by
    #     hpu_mla_skip_indexer's patched __init__) and any already-built instances. We
    #     re-patch __init__ to NOT force-skip under the gate, and flip live instances.
    _unskip_indexer()

    # (e) eagerly load both .so now so a misconfig fails at plugin-init, LOUD.
    dsa_ops.load_ops()

    import logging
    logging.getLogger("vllm").warning(
        "[hpu-dsa-ext] HPU_DSA_SPARSE=1 -> DSA SPARSE OP PATH ACTIVE (indexer_topk + "
        "sparse_attn kernels). FULL_ATTENTION_FALLBACK=False, indexer un-skipped. This "
        "is the gated sparse path; ctx<=index_topk still runs dense (sparse==dense).")


def _indexshare_skip_topk(wrapper, caller_skip_topk: bool) -> bool:
    """IndexShare cadence (GLM-5.2): decide whether THIS layer REUSES the prior full
    layer's top-k buffer, from the model config + the layer id parsed from
    ``wrapper.prefix`` — OVERRIDING the caller's ``skip_topk``.

    WHY override: upstream deepseek_v2.py computes the cadence from
    ``index_topk_pattern`` (NULL on GLM-5.2) and falls back to ``max(layer_id-1,0) %
    freq`` — a baked-in offset=2. GLM-5.2 ships an explicit ``indexer_types`` array with
    ``index_skip_topk_offset=3``, so the upstream value is WRONG (shared layers would
    reuse the wrong full layer). We read the authoritative cadence from
    ``glm_moe_dsa_wire.DsaDims`` (unit-tested against the HF __post_init__ generator).
    For GLM-5.1 (freq=1) the two agree (every layer full), so this is a no-op there.

    Fails SOFT (LOUD warning, keep the caller's value) if config/prefix are unavailable
    so a future upstream wiring change degrades to prior behaviour instead of crashing
    the serve; non-DSA configs are returned untouched.
    """
    import logging
    import re

    log = logging.getLogger("vllm")
    try:
        from vllm.config import get_current_vllm_config
        from . import glm_moe_dsa_wire as wire

        hf_config = get_current_vllm_config().model_config.hf_config
        if not wire.config_is_sparse(hf_config):
            return bool(caller_skip_topk)
        prefix = getattr(wrapper, "prefix", "") or ""
        m = re.search(r"layers\.(\d+)\b", prefix)
        if m is None:
            log.warning("[hpu-dsa-ext] IndexShare: no layer id in prefix %r; keeping "
                        "caller skip_topk=%s.", prefix, caller_skip_topk)
            return bool(caller_skip_topk)
        dims = wire.read_dsa_dims(hf_config)
        layer_id = int(m.group(1))
        skip = dims.layer_skips_topk(layer_id)
        if skip != bool(caller_skip_topk):
            log.warning("[hpu-dsa-ext] IndexShare override: layer %d skip_topk %s -> %s "
                        "(config offset=%d freq=%d, indexer_types=%s).",
                        layer_id, bool(caller_skip_topk), skip,
                        dims.index_skip_topk_offset, dims.index_topk_freq,
                        "explicit" if dims.indexer_types is not None else "derived")
        return skip
    except Exception as e:  # noqa: BLE001
        log.warning("[hpu-dsa-ext] IndexShare cadence override failed (%s: %s); keeping "
                    "caller skip_topk=%s.", type(e).__name__, e, caller_skip_topk)
        return bool(caller_skip_topk)


def _unskip_indexer() -> None:
    """Override hpu_mla_skip_indexer's force-skip so the indexer FIRES under the gate.

    hpu_mla_skip_indexer.register() patched ``HPUMultiHeadLatentAttentionWrapper.__init__``
    to set ``self.skip_topk = True`` unconditionally. Under HPU_DSA_SPARSE we re-patch
    __init__ to set skip_topk from the AUTHORITATIVE IndexShare cadence
    (``_indexshare_skip_topk`` -> ``glm_moe_dsa_wire.DsaDims``), NOT the caller's value:
    upstream computes the cadence with the wrong field/offset for GLM-5.2. Full layers
    -> False (run their own indexer); shared layers -> True (reuse the prior full
    layer's top-k buffer). So the indexer fires on exactly the layers GLM-5.2 was
    trained with — 21 of 78, not every layer (GLM-5.1 freq=1 -> all full, unchanged)."""
    from vllm_gaudi.attention.oot_mla import HPUMultiHeadLatentAttentionWrapper as W

    if not getattr(W.__init__, "_hpu_dsa_unskip", False):
        # The current __init__ is hpu_mla_skip_indexer's _patched_init (forces True).
        # Its closure holds the ORIGINAL via _orig_init; we cannot reach that closure
        # cleanly, so instead wrap the CURRENT __init__ and set the correct cadence after.
        _cur_init = W.__init__

        def _gated_init(self, *args, skip_topk: bool = False, **kwargs):
            _cur_init(self, *args, skip_topk=skip_topk, **kwargs)
            # IndexShare (GLM-5.2): set skip_topk from the CONFIG cadence
            # (indexer_types / offset=3), overriding both the prior force-True AND the
            # caller's value (upstream computes it with the wrong offset=2 / null field).
            self.skip_topk = _indexshare_skip_topk(self, skip_topk)

        _gated_init._hpu_dsa_unskip = True
        W.__init__ = _gated_init


def threshold_help(index_topk: int) -> str:
    """Human note on the sparse-vs-dense threshold for a given config."""
    return (
        f"DSA sparse fires only when a query has > index_topk (={index_topk}) causal "
        f"keys; at ctx <= {index_topk} every causal key is selected so sparse == dense "
        f"(the MLA forward runs the dense path). Gate (c) must use ctx > {index_topk} "
        "(e.g. 16K) to exercise a strict-subset selection.")
