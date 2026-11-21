"""HPU DSA — wire the gather-consume into HPUMLAAttention.forward_impl (the LAST code piece).

Monkeypatches `vllm_gaudi.attention.oot_mla.HPUMLAAttention.forward_impl` so that, for SPARSE
(DSA) layers, MLA attention is restricted to the indexer-selected top-k tokens via
`sparse_mla_consume.sparse_mla_gather_attention` — instead of dense MLA. Non-sparse layers and
the profiling/dummy run fall through to the ORIGINAL dense forward_impl unchanged.

NO SILENT DENSE (the PR #1301 trap): a sparse layer that hits an unsupported case RAISES loudly
rather than quietly running dense. Launch GLM-5.1 with `--no-enable-prefix-caching` (required).

MLA internals used (verified, oot_mla.py:106-218):
  q:[T,H,qk_head]=nope+rope · k_c_normed:[T,kv_lora] · k_pe:[T,1,qk_rope]
  W_UK_T:[H,qk_nope,kv_lora]  W_UV:[H,kv_lora,v_head]   per-head materialization:
    k_nope[s,h]=k_c[s]·W_UK_T[h]^T ; k=[k_nope ‖ k_pe(shared)] ; v[s,h]=k_c[s]·W_UV[h]
  latent cache kv_cache[0] is flat (num_blocks*block_size, 1, kv_lora+qk_rope); current step is
  written via impl.latent_cache_k(slot_mapping) BEFORE scoring (so decode history is readable).

STATUS: authored against the real internals; PREFILL (no-prefix, S==prompt) is the clean path and
matches the CPU gather==masked oracle. DECODE gathers each request's paged latent history. BOTH
are pending the on-HPU validation loop (synthetic-on-card test, then the 744B one-shot + a
sparse!=dense gate). The patch only fires when platform_patch.DSA_READY is True (gated by
HPU_DSA_READY=1), so it CANNOT affect serving until that gate is set post-validation.

Scope: <=32-64K (materializes full per-head k/v over the live key set). 200K needs the
gather-then-materialize TPC path (out of scope here).
"""
from __future__ import annotations

import os

import torch

from . import platform_patch
from .indexer_metadata_hpu import build_indexer_metadata, per_request_block_ids
from .sparse_mla_consume import sparse_mla_gather_attention

# MVP-3 op path (gated, OFF by default). Imported lazily so a CPU-only import of this
# module never pulls the .so loader; the symbols are only touched under HPU_DSA_SPARSE.
try:  # pragma: no cover - import-time presence only
    from . import dsa_ops as _dsa_ops
    from . import hpu_indexer_op_forward as _dsa_idx
except Exception:  # the op modules are optional siblings; absence == op path unavailable
    _dsa_ops = None
    _dsa_idx = None

_ORIG_FORWARD_IMPL = None  # set by register(); the original dense HPUMLAAttention.forward_impl

# Diagnostic: log dense-MLA q-input + attention-output norms on the first N forward_impl
# calls (gated by HPU_ATTN_PROBE, default 0 = off). Pairs with HPU_MOE_PROBE to localise
# whether the residual is already garbage at the ATTENTION output (q/out NaN or norm
# blow-up) BEFORE the MoE runs. @torch.compiler.disable keeps it out of the HPU graph.
_ATTN_PROBE_N = int(os.environ.get("HPU_ATTN_PROBE", "0") or "0")
_attn_probe_count = [0]

if _ATTN_PROBE_N > 0:
    import logging as _logging
    _attn_log = _logging.getLogger("vllm")

    @torch.compiler.disable
    def _attn_probe(layer_name, is_prompt, q, out):
        try:
            qf = q.detach().float()
            of = out.detach().float()
            _attn_log.warning(
                "[attn-probe #%d] layer=%s prompt=%s q_norm=%.4f q_absmax=%.4f "
                "q_nan=%d out_norm=%.4f out_absmax=%.4f out_nan=%d out_inf=%d",
                _attn_probe_count[0], layer_name, bool(is_prompt),
                float(qf.norm()), float(qf.abs().max()),
                int(torch.isnan(qf).any()),
                float(of.norm()), float(of.abs().max()),
                int(torch.isnan(of).any()), int(torch.isinf(of).any()),
            )
        except Exception as ex:  # noqa: BLE001
            _attn_log.warning("[attn-probe] log error: %s", ex)


def _topk_buffer(self) -> torch.Tensor:
    """The indexer's [T, index_topk] int32 (-1 padded) selection buffer for this layer."""
    idx = getattr(self, "indexer", None)
    buf = getattr(idx, "topk_indices_buffer", None) if idx is not None else None
    if buf is None:
        buf = getattr(self, "topk_indices_buffer", None)
    if buf is None:
        raise RuntimeError(
            "[hpu-dsa-ext] sparse layer has no topk_indices_buffer — the indexer did not run "
            "before attention (check the MLA wrapper's indexer call at mla.py:168)."
        )
    return buf


def _materialize_kv_per_head(k_c_all, k_pe_all, W_UK_T, W_UV, qk_rope):
    """latent (k_c:[S,kv_lora], k_pe:[S,qk_rope]) -> per-head k:[S,H,qk_head], v:[S,H,v_head]."""
    S = k_c_all.shape[0]
    H = W_UK_T.shape[0]
    kc = k_c_all.to(W_UV.dtype)
    k_nope = torch.einsum("sl,hnl->shn", kc, W_UK_T)                          # [S,H,qk_nope]
    k_pe_b = k_pe_all.to(kc.dtype).reshape(S, 1, qk_rope).expand(S, H, qk_rope)
    k = torch.cat([k_nope, k_pe_b], dim=-1)                                   # [S,H,qk_head]
    v = torch.einsum("sl,hlv->shv", kc, W_UV)                                 # [S,H,v_head]
    return k, v


def _gather_request_latent(kv_cache0, block_ids, block_size, ctx_len, kv_lora, qk_rope):
    """Read one request's history latent rows (length ctx_len) from the flat paged cache, in
    causal order, and split into (k_c:[ctx,kv_lora], k_pe:[ctx,qk_rope])."""
    rows = (block_ids.view(-1, 1) * block_size
            + torch.arange(block_size, device=block_ids.device).view(1, -1)).reshape(-1)
    rows = rows[:ctx_len]                                       # trim trailing slots of last block
    latent = kv_cache0.index_select(0, rows).reshape(ctx_len, kv_lora + qk_rope)
    return latent[:, :kv_lora], latent[:, kv_lora:]


def _op_consume_forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                             output, output_scale, output_block_scale):
    """MVP-3 SPARSE OP consume (gated HPU_DSA_SPARSE=1). For sparse layers, restrict MLA
    attention to the indexer-selected top-k via the captured ``sparse_attn`` kernel.

    Identical structure to the pure-torch sparse branch below (same latent-cache write,
    same per-head materialization, same metadata), but the inner gather+SDPA is the
    GC-captured ``torch.ops.hpu_dsa_ext.sparse_attn`` op instead of the torch
    ``sparse_mla_gather_attention``. The indexer's top-k indices (filled by the op-backed
    forward_oot) are passed straight through — the kernel masks only -1 pads + relies on
    the indexer's causal selection (no sentinel/causal re-derivation here).

    THRESHOLD: when the batch's max sequence length ≤ index_topk every causal key is
    selected, so sparse == dense; we run the genuine dense forward (documented short-cut).
    Falls back to dense for the profiling/dummy run and non-sparse layers, exactly like
    the torch path."""
    # Non-sparse layer / profiling run -> dense (verbatim), same as the torch branch.
    if not getattr(self, "use_sparse", False) or attn_metadata is None:
        return _ORIG_FORWARD_IMPL(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                                  output, output_scale, output_block_scale)

    # The sparse decision + body run as ONE eager island (torch.compiler.disable):
    # dynamo must never compile fragments of the sparse path — shape-agnostic recipe
    # caching cannot shape-infer them (or the custom op) on a shape-changed cache hit
    # ("[SHAPE AGNOSTIC] Cache hit Synapse shape inference failed!"). Returns None
    # when the batch is below the sparse threshold -> genuine dense forward (which
    # stays COMPILED — only the sparse island is eager).
    out = _op_consume_sparse_or_none(self, q, k_c_normed, k_pe, kv_cache, attn_metadata)
    if out is None:
        return _ORIG_FORWARD_IMPL(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                                  output, output_scale, output_block_scale)
    return out


@torch.compiler.disable
def _op_consume_sparse_or_none(self, q, k_c_normed, k_pe, kv_cache, attn_metadata):
    """The sparse-op consume body (eager island). None => caller runs dense."""
    qk_rope, kv_lora = self.qk_rope_head_dim, self.kv_lora_rank
    H, Dv = self.num_heads, self.v_head_dim
    index_topk = int(getattr(self, "topk_tokens", 0) or 0)

    # THRESHOLD short-cut (computed BEFORE the cache write to avoid a double index_copy):
    # ctx <= index_topk => every causal key is selected => sparse == dense. Run the genuine
    # dense forward (which does its OWN latent-cache write). Determine the batch's max
    # causal extent from the cheap CPU-derived metadata.
    meta = build_indexer_metadata(attn_metadata)
    active = meta.active if meta is not None else None
    max_seq = int(active.seq_lens.max().item()) if (active is not None and active.seq_lens.numel()) else 0
    if index_topk <= 0 or not _dsa_idx.indexer_fires(max_seq, index_topk):
        return None

    # Sparse path: write current-step latent to the MLA cache exactly as the dense path
    # does (so the decode history read below sees this step's key).
    slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
    latent_vec_k = torch.concat(
        (k_c_normed, k_pe.view(*k_c_normed.shape[:-1], qk_rope)), dim=-1
    ).view(-1, qk_rope + kv_lora)
    if kv_cache is not None and len(kv_cache) >= 2:
        self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)

    topk_buf = _topk_buffer(self)
    T = q.shape[0]
    out = torch.empty((T, H * Dv), dtype=q.dtype, device=q.device)

    if attn_metadata.is_prompt:
        am = meta.prefill
        if am.block_list is not None:
            raise NotImplementedError(
                "[hpu-dsa-ext] sparse-op prefill with prefix-cache blocks unsupported — "
                "launch with --no-enable-prefix-caching (context_lens must be 0).")
        kpe_flat = k_pe.reshape(-1, qk_rope)
        cu = am.cu_query_lens
        for r in range(am.batch_size):
            s0, s1 = int(cu[r]), int(cu[r + 1])
            if int(am.context_lens[r]) != 0:
                raise NotImplementedError(
                    "[hpu-dsa-ext] non-zero prefill context_len (prefix cache) unsupported")
            k_r, v_r = _materialize_kv_per_head(
                k_c_normed[s0:s1], kpe_flat[s0:s1], self.W_UK_T, self.W_UV, qk_rope)
            sp = _dsa_ops.sparse_attn_op(q[s0:s1], k_r, v_r, topk_buf[s0:s1], float(self.scale))
            out[s0:s1] = sp.reshape(s1 - s0, H * Dv)
        return out

    # decode: per request, gather its paged latent history, materialize per-head, consume.
    dm = meta.decode
    if kv_cache is None or len(kv_cache) < 1:
        raise RuntimeError("[hpu-dsa-ext] sparse-op decode requires the latent kv_cache")
    blk_per_req = per_request_block_ids(dm.block_list, dm.block_groups, dm.batch_size)
    for r in range(dm.batch_size):
        ctx = int(dm.seq_lens[r])
        k_c_r, k_pe_r = _gather_request_latent(
            kv_cache[0], blk_per_req[r], dm.block_size, ctx, kv_lora, qk_rope)
        k_r, v_r = _materialize_kv_per_head(k_c_r, k_pe_r, self.W_UK_T, self.W_UV, qk_rope)
        sp = _dsa_ops.sparse_attn_op(q[r:r + 1], k_r, v_r, topk_buf[r:r + 1], float(self.scale))
        out[r] = sp.reshape(H * Dv)
    return out


def _sparse_forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                         output=None, output_scale=None, output_block_scale=None):
    # MVP-3 SPARSE OP PATH (gated HPU_DSA_SPARSE=1, OFF by default). When the gate is
    # on, hpu_indexer_op_forward.register() has set FULL_ATTENTION_FALLBACK=False and
    # un-skipped the indexer, so the op-backed indexer has filled topk_indices_buffer;
    # route the consume through the captured sparse_attn kernel. When the gate is OFF,
    # this entire block is skipped and the dense FULL_ATTENTION_FALLBACK below runs
    # unchanged — the current serve is byte-for-byte unaffected.
    if _dsa_ops is not None and _dsa_idx is not None and _dsa_ops.sparse_enabled():
        return _op_consume_forward_impl(self, q, k_c_normed, k_pe, kv_cache,
                                        attn_metadata, output, output_scale,
                                        output_block_scale)

    # FULL-ATTENTION mode (the coherent HPU path today): the indexer is skipped
    # (hpu_mla_skip_indexer forces skip_topk=True), so the topk buffer is all -1 and
    # there is nothing valid to gather. Run the GENUINE dense MLA forward over all
    # causal tokens instead of restricting to an empty selection (which the consume
    # NaN guard would collapse to self-only attention -> incoherent output). This
    # keeps the full DSA gather-consume code below intact for a future sparse path;
    # it is simply bypassed while platform_patch.FULL_ATTENTION_FALLBACK is True.
    if platform_patch.FULL_ATTENTION_FALLBACK:
        out = _ORIG_FORWARD_IMPL(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                                 output, output_scale, output_block_scale)
        if _ATTN_PROBE_N > 0 and _attn_probe_count[0] < _ATTN_PROBE_N:
            _attn_probe_count[0] += 1
            _attn_probe(getattr(self, "layer_name", "?"),
                        getattr(attn_metadata, "is_prompt", None), q, out)
        return out

    # Non-sparse layer / profiling run -> original dense path (verbatim).
    if not getattr(self, "use_sparse", False) or attn_metadata is None:
        return _ORIG_FORWARD_IMPL(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                                  output, output_scale, output_block_scale)

    qk_rope, kv_lora = self.qk_rope_head_dim, self.kv_lora_rank
    H, Dv = self.num_heads, self.v_head_dim
    # Write current-step latent to the MLA cache exactly as the dense path does.
    slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
    latent_vec_k = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], qk_rope)), dim=-1).view(-1, qk_rope + kv_lora)
    if kv_cache is not None and len(kv_cache) >= 2:
        self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)

    meta = build_indexer_metadata(attn_metadata)
    topk = _topk_buffer(self)
    T = q.shape[0]
    out = torch.empty((T, H * Dv), dtype=q.dtype, device=q.device)

    if attn_metadata.is_prompt:
        am = meta.prefill
        if am.block_list is not None:
            raise NotImplementedError(
                "[hpu-dsa-ext] sparse prefill with prefix-cache blocks not yet supported — "
                "launch with --no-enable-prefix-caching (context_lens must be 0)."
            )
        kpe_flat = k_pe.reshape(-1, qk_rope)
        cu = am.cu_query_lens
        for r in range(am.batch_size):
            s0, s1 = int(cu[r]), int(cu[r + 1])
            if int(am.context_lens[r]) != 0:
                raise NotImplementedError("[hpu-dsa-ext] non-zero prefill context_len (prefix cache) unsupported")
            k_r, v_r = _materialize_kv_per_head(k_c_normed[s0:s1], kpe_flat[s0:s1], self.W_UK_T, self.W_UV, qk_rope)
            sp = sparse_mla_gather_attention(q[s0:s1], k_r, v_r, topk[s0:s1],
                                             am.abs_pos[s0:s1].to(torch.int64), float(self.scale),
                                             gather_width=index_topk + 1)
            out[s0:s1] = sp.reshape(s1 - s0, H * Dv)
        return out

    # decode: per request, gather its paged latent history (current step already written), then
    # materialize + gather-consume. (on-HPU validation pending.)
    dm = meta.decode
    if kv_cache is None or len(kv_cache) < 1:
        raise RuntimeError("[hpu-dsa-ext] sparse decode requires the latent kv_cache")
    blk_per_req = per_request_block_ids(dm.block_list, dm.block_groups, dm.batch_size)
    for r in range(dm.batch_size):
        ctx = int(dm.seq_lens[r])
        k_c_r, k_pe_r = _gather_request_latent(kv_cache[0], blk_per_req[r], dm.block_size, ctx, kv_lora, qk_rope)
        k_r, v_r = _materialize_kv_per_head(k_c_r, k_pe_r, self.W_UK_T, self.W_UV, qk_rope)
        # one query row (next_n==1) at absolute position ctx-1; topk row r selects within [0,ctx).
        q_r = q[r:r + 1]
        abs_r = torch.tensor([ctx - 1], dtype=torch.int64, device=q.device)
        sp = sparse_mla_gather_attention(q_r, k_r, v_r, topk[r:r + 1], abs_r, float(self.scale), gather_width=index_topk + 1)
        out[r] = sp.reshape(H * Dv)
    return out


def register() -> None:
    """Install the forward_impl patch. Idempotent. The patch is INERT for non-sparse layers and
    only engages sparse layers (which only exist once a DSA model loads); serving is unaffected
    until platform_patch.DSA_READY routes a sparse model here."""
    global _ORIG_FORWARD_IMPL
    from vllm_gaudi.attention.oot_mla import HPUMLAAttention

    if getattr(HPUMLAAttention.forward_impl, "_hpu_dsa_wired", False):
        return
    _ORIG_FORWARD_IMPL = HPUMLAAttention.forward_impl
    _sparse_forward_impl._hpu_dsa_wired = True
    HPUMLAAttention.forward_impl = _sparse_forward_impl
    # expose for the gate / tests
    platform_patch.CONSUME_WIRED = True
