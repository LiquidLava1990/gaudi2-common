"""Route the GLM/DeepSeek MLA-latent KV store->int4 and read->dequant, for the real
capacity win (the fake-quant in oot_mla proved the TEXT quality; this STORES the codes).

THE MLA SEAM (vllm_gaudi/attention/backends/hpu_attn.py HPUMLAImpl, + oot_mla.py)
--------------------------------------------------------------------------------
  * The MLA cache is ONE combined latent vector [T, kv_lora_rank + rope]. HPUMLAImpl
    owns ``self.latent_cache_k`` (VLLMKVCache / VLLMFP8KVCache).
  * STORE (oot_mla.forward_impl ~190): ``self.impl.latent_cache_k(latent_vec_k,
    kv_cache[0], slot_mapping)``. We replace latent_cache_k with a _TQLatentCache that,
    given the 3-plane kv_cache tuple (int4-latent, rope, scale), quantizes the latent to
    int4 (WHT-rotate + per-group uniform, byte-identical to the on-card-validated
    oot_mla._mla_fake_quant_latent) and index_copy_s the 3 planes -- AND stashes the
    plane refs on a shared controller for the read. (oot_mla passes the full kv_cache
    tuple to the store when HPU_MLA_KV_INT4STORE is set -- one guarded edit there.)
  * READ (hpu_attn.py forward_mha ~255 prefix + forward_mqa ~333 decode): both call
    ``self.latent_cache_k.fetch_from_cache(kv_cache[0].unflatten(0,(-1,block_size)),
    block_list)``. Our fetch_from_cache gathers the 3 STASHED planes over the block_list
    and returns the dequant'd bf16 latent [nblk, block_size, 1, kv_lora+rope] -- the same
    shape the stock fetch returns, so MLA attention proceeds unchanged.

GUARDED: install happens only when HPU_MLA_KV_INT4STORE is truthy. Unset -> this module
never patches anything and every MLA serve stays exactly fp8_inc.
"""
from __future__ import annotations

import os

import torch

from . import kvq_ops

_MLA_V_BITS = 4
_MLA_GROUP = 64


def int4_store_enabled() -> bool:
    return os.environ.get("HPU_MLA_KV_INT4STORE", "0").strip().lower() in ("1", "true", "yes", "on")


class _TQLatentController:
    """Per-MLA-layer coordinator. Holds config + the per-forward stash of the 3 planes
    (int4 latent, bf16 rope, f32 scale) so the read (fetch_from_cache) can find them."""

    def __init__(self, kv_lora_rank: int, rope_dim: int, use_contiguous_pa: bool):
        self.kv_lora_rank = int(kv_lora_rank)
        self.rope_dim = int(rope_dim)
        self.use_contiguous_pa = bool(use_contiguous_pa)
        self.packed = None   # int32 [Nslots, 1, kv_lora/8]
        self.rope = None     # bf16  [Nslots, 1, rope_dim]
        self.scale = None    # f32   [Nslots, 1, kv_lora/group]

    @torch.compiler.disable
    def store(self, latent, kv_cache, slot_mapping):
        """latent [T, kv_lora+rope]; kv_cache = (packed_plane, rope_plane, scale_plane[, ...])."""
        self.packed, self.rope, self.scale = kv_cache[0], kv_cache[1], kv_cache[2]
        if latent is None or slot_mapping is None:
            return
        L, R = self.kv_lora_rank, self.rope_dim
        lat2d = latent.reshape(-1, L + R)
        packed, scale, rope = kvq_ops.mla_latent_quant_op(lat2d, L, _MLA_V_BITS, _MLA_GROUP)
        slots = slot_mapping.reshape(-1).to(torch.long)
        T = slots.shape[0]
        self.packed.index_copy_(0, slots, packed.reshape(T, 1, packed.shape[-1]))
        self.rope.index_copy_(0, slots, rope.reshape(T, 1, R).to(self.rope.dtype))
        self.scale.index_copy_(0, slots, scale.reshape(T, 1, scale.shape[-1]))

    @torch.compiler.disable
    def fetch(self, cache_unflat, blocks):
        if self.packed is None:
            raise RuntimeError("[hpu-kvq-ext] MLA fetch before store -- latent planes not captured.")
        block_size = int(cache_unflat.shape[1])
        L, R = self.kv_lora_rank, self.rope_dim

        def gather(plane):
            u = plane.unflatten(0, (-1, block_size))            # [num_blocks, bs, 1, W]
            return u[:blocks.size(0)] if self.use_contiguous_pa else u.index_select(0, blocks)

        pk = gather(self.packed)
        rp = gather(self.rope)
        sc = gather(self.scale)
        nblk, bs = pk.shape[0], pk.shape[1]
        latent = kvq_ops.mla_latent_dequant_op(
            pk.reshape(-1, pk.shape[-1]), sc.reshape(-1, sc.shape[-1]),
            rp.reshape(-1, R), L, _MLA_V_BITS, _MLA_GROUP)                # [nblk*bs, L+R] bf16
        return latent.reshape(nblk, bs, 1, L + R)


class _TQLatentCache(torch.nn.Module):
    """Replaces HPUMLAImpl.latent_cache_k. STORE = int4-quantize + write 3 planes (called
    with the full kv_cache tuple via the guarded oot_mla edit). READ = dequant the stash."""

    def __init__(self, ctrl: _TQLatentController):
        super().__init__()
        self.ctrl = ctrl

    # store entry point used by the guarded oot_mla edit (full kv_cache tuple)
    def store_latent_int4(self, latent, kv_cache, slot_mapping):
        self.ctrl.store(latent, kv_cache, slot_mapping)

    # stock-signature forward kept as a safety fallback (should not be hit when int4 store is on)
    def forward(self, input, cache, slot_mapping, **kwargs):
        raise RuntimeError("[hpu-kvq-ext] MLA int4 latent cache called via the stock 1-plane store "
                           "path; oot_mla must route through store_latent_int4 (guarded edit missing?).")

    def fetch_from_cache(self, cache, blocks, **kwargs):
        return self.ctrl.fetch(cache, blocks)


def register() -> None:
    """Patch HPUMLAImpl.__init__ so an MLA serve with HPU_MLA_KV_INT4STORE installs the int4
    latent cache in place of VLLMKVCache/VLLMFP8KVCache. Idempotent, guarded, no-op when unset."""
    if not int4_store_enabled():
        return
    from vllm_gaudi.attention.backends import hpu_attn as A

    impl = A.HPUMLAImpl
    if getattr(impl.__init__, "_kvq_mla_patched", False):
        return
    _orig_init = impl.__init__

    def __init__(self, *args, **kwargs):  # noqa: ANN001
        _orig_init(self, *args, **kwargs)
        rope_dim = int(self.qk_rope_head_dim)
        ctrl = _TQLatentController(
            kv_lora_rank=int(self.kv_lora_rank),
            rope_dim=rope_dim,
            use_contiguous_pa=getattr(getattr(self, "latent_cache_k", None), "use_contiguous_pa", False),
        )
        self._kvq_mla_ctrl = ctrl
        self.latent_cache_k = _TQLatentCache(ctrl)

    __init__._kvq_mla_patched = True
    __init__._kvq_mla_orig = _orig_init
    impl.__init__ = __init__
