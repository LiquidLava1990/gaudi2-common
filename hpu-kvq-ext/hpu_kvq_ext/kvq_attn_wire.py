"""Route GQA KV store -> kv_rotate_quant and read -> kv_dequant_unrotate for the
TurboQuant dtypes, by swapping HPUAttentionImpl's ``k_cache`` / ``v_cache`` modules.

THE SEAM (vllm_gaudi/attention/backends/hpu_attn.py, as of 2026-07-01)
----------------------------------------------------------------------
  * __init__ (line ~416-419): ``self.k_cache`` / ``self.v_cache`` are VLLMKVCache
    (or VLLMFP8KVCache). We replace them with a coordinated (_TQKCache, _TQVCache)
    pair sharing one controller when the serve dtype is turboquant_*.
  * STORE (line ~553-564): ``key_cache = self.k_cache(key, key_cache, slot_mapping,
    scales=k_scales, ...)`` then ``value_cache = self.v_cache(value, value_cache,
    scales=v_scales, ...)``. The two calls are separate but the combined WRITE op
    needs K AND V together, so _TQKCache stashes K on the controller and _TQVCache
    (called immediately after, same slot_mapping) runs kv_rotate_quant once and
    index_copy_s the 4 planes.
  * READ (extension/ops.py flat_pa ~224-226, decode-only): ``keys_fetch_func`` /
    ``values_fetch_func`` are ``self.k_cache.fetch_from_cache`` /
    ``self.v_cache.fetch_from_cache`` (bound at hpu_attn.py:683-684). Each recomputes
    the combined kv_dequant_unrotate over the block_list (from the controller's 4
    planes, stashed at the same-forward store) and returns K (resp. V). out_fp8=0 ->
    bf16 SDPA (first-draft coherence path).

PREFILL uses key/value DIRECTLY (prompt_attention), never the cache fetch, so the
fetch path is decode-only and a store always precedes it in the same forward. MLA
uses a different impl (HPUMLAImpl) which we never touch -> MLA stays fp8_inc.
"""
from __future__ import annotations

import torch

from . import kvq_ops
from .platform_patch_kvq import KVQ_DTYPES, v_bits_for_dtype

class _TQController:
    """Per-layer coordinator shared by the K/V cache modules. Holds config + the
    per-forward stash of the 4 cache planes and the current-step K."""

    def __init__(self, v_bits: int, num_kv_heads: int, head_size: int, use_contiguous_pa: bool):
        self.v_bits = int(v_bits)
        self.H = int(num_kv_heads)
        self.D = int(head_size)
        self.use_contiguous_pa = bool(use_contiguous_pa)
        # per-forward stash (planes are stable across steps but re-captured each store)
        self.k = None            # current step's rotated-source K [T,H,D] (pre-quant)
        self.key_cache = None     # fp8 [Nslots,H,D]
        self.value_cache = None   # int32 [Nslots,H,lanes]
        self.key_scales = None    # f32 [Nslots,H,1]
        self.value_scales = None  # f32 [Nslots,H,1]

    # -- combined WRITE: called from _TQVCache.forward (K already stashed) ----
    @torch.compiler.disable
    def write(self, value, slot_mapping):
        key = self.k
        if key is None or value is None:
            return
        D, H = self.D, self.H
        assert key.shape[-1] == D, f"[hpu-kvq-ext] K head_dim {key.shape[-1]} != {D}"
        k2d = key.reshape(-1, D)
        v2d = value.reshape(-1, D)
        k_fp8, k_scale, v_packed, v_scale = kvq_ops.kv_rotate_quant_op(k2d, v2d, self.v_bits)
        slots = slot_mapping.reshape(-1).to(torch.long)
        T = slots.shape[0]
        lanes = v_packed.shape[-1]
        self.key_cache.index_copy_(0, slots, k_fp8.reshape(T, H, D))
        self.key_scales.index_copy_(0, slots, k_scale.reshape(T, H, 1))
        self.value_cache.index_copy_(0, slots, v_packed.reshape(T, H, lanes))
        self.value_scales.index_copy_(0, slots, v_scale.reshape(T, H, 1))
        self.k = None  # consumed

    # -- combined READ: called from either fetch_from_cache ------------------
    @torch.compiler.disable
    def dequant_blocks(self, cache_unflat, blocks):
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError(
                "[hpu-kvq-ext] fetch before store -- the TurboQuant KV planes were not "
                "captured this forward (kv-sharing / unexpected path?).")
        block_size = int(cache_unflat.shape[1])

        def gather(plane):
            u = plane.unflatten(0, (-1, block_size))   # [num_blocks, bs, H, W]
            if self.use_contiguous_pa:
                return u[:blocks.size(0)]
            return u.index_select(0, blocks)           # [nblk, bs, H, W]

        kf = gather(self.key_cache)
        ks = gather(self.key_scales)
        vp = gather(self.value_cache)
        vs = gather(self.value_scales)
        nblk, bs, H, D = kf.shape
        k, v = kvq_ops.kv_dequant_unrotate_op(
            kf.reshape(-1, D), ks.reshape(-1, 1),
            vp.reshape(-1, vp.shape[-1]), vs.reshape(-1, 1), self.v_bits, out_fp8=0)
        return (k.reshape(nblk, bs, H, D), v.reshape(nblk, bs, H, D))


class _TQKCache(torch.nn.Module):
    """K side. STORE: stash K (deferred; the write happens in the V module). READ:
    combined dequant, return K."""

    def __init__(self, ctrl: _TQController):
        super().__init__()
        self.ctrl = ctrl

    def forward(self, input, cache, slot_mapping, scales=None, block_size=None,
                is_prompt=False, **kwargs):
        if input is not None:
            self.ctrl.k = input
            self.ctrl.key_cache = cache
            self.ctrl.key_scales = scales
        return cache

    def fetch_from_cache(self, cache, blocks, **kwargs):
        k, _v = self.ctrl.dequant_blocks(cache, blocks)
        return k


class _TQVCache(torch.nn.Module):
    """V side. STORE: run the combined kv_rotate_quant write into all 4 planes. READ:
    combined dequant, return V."""

    def __init__(self, ctrl: _TQController):
        super().__init__()
        self.ctrl = ctrl

    def forward(self, input, cache, slot_mapping, scales=None, block_size=None,
                is_prompt=False, **kwargs):
        if input is not None:
            self.ctrl.value_cache = cache
            self.ctrl.value_scales = scales
            self.ctrl.write(input, slot_mapping)
        return cache

    def fetch_from_cache(self, cache, blocks, **kwargs):
        _k, v = self.ctrl.dequant_blocks(cache, blocks)
        return v


def register() -> None:
    """Monkeypatch HPUAttentionImpl.__init__ so a turboquant serve installs the
    TurboQuant K/V cache modules (a coordinated pair sharing one controller) in place
    of VLLMKVCache. Idempotent. Never touches the MLA impl (MLA stays fp8_inc)."""
    from vllm_gaudi.attention.backends import hpu_attn as A

    impl = A.HPUAttentionImpl
    if getattr(impl.__init__, "_kvq_patched", False):
        return
    _orig_init = impl.__init__

    def __init__(self, *args, **kwargs):  # noqa: ANN001
        _orig_init(self, *args, **kwargs)
        dt = getattr(self, "kv_cache_dtype", None)
        # Some HPUAttentionImpl instances report kv_cache_dtype='auto' even on a turboquant
        # serve (per-layer dtype not threaded), yet patch_get_kv_cache_spec converted THEIR
        # cache to int4 (it keys on spec TYPE = FullAttentionSpec, not the impl dtype). Those
        # layers then have an int4 cache but a stock bf16 write -> index_copy dtype crash. Key
        # the install on the SERVE dtype so the wire matches the spec conversion exactly.
        if dt not in KVQ_DTYPES:
            try:
                from vllm.config import get_current_vllm_config
                sdt = get_current_vllm_config().cache_config.cache_dtype
                if sdt in KVQ_DTYPES:
                    dt = sdt
            except Exception:  # noqa: BLE001
                pass
        if dt in KVQ_DTYPES:
            ctrl = _TQController(
                v_bits=v_bits_for_dtype(dt),
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                use_contiguous_pa=getattr(self, "use_contiguous_pa", False),
            )
            self._kvq_ctrl = ctrl
            self.k_cache = _TQKCache(ctrl)
            self.v_cache = _TQVCache(ctrl)

    __init__._kvq_patched = True
    __init__._kvq_orig = _orig_init
    impl.__init__ = __init__
