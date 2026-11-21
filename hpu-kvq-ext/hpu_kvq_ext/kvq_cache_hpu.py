"""Size + allocate the asymmetric TurboQuant k8v4 KV cache on HPU.

WHY THIS FILE EXISTS
--------------------
The stock HPU fp8 KV path is elegant because fp8 is 1-byte, SAME shape as bf16 --
K and V drop into two symmetric ``(num_blocks*block_size, num_kv_heads, head_size)``
planes with the dtype swapped. TurboQuant k8v4 is NOT symmetric:

    key_cache    fp8_e4m3fn  (N, H, D)         rotated K, fp8   (1 byte/elem, == fp8 K)
    value_cache  int32       (N, H, D//8)      rotated V, int4 packed 8-per-int32
    key_scales   float32     (N, H, 1)         per-(token,head) K scale
    value_scales float32     (N, H, 1)         per-(token,head) V scale

``FullAttentionSpec`` assumes K and V are symmetric (one dtype, one head_size) for
``page_size_bytes``, and the runner's ``initialize_kv_cache`` allocates two symmetric
planes. If we let it run with a uint8 marker dtype it allocates 2*N*H*D bytes for a
budget sized at ~1.5*N*H*D (K fp8 + V int4 + scales) -> ~1.33x OVER the budget ->
init OOM. So we (1) report the TRUE asymmetric page_size via a spec subclass and
(2) INTERCEPT the non-hybrid allocation to build the 4 correct planes.

Three install hooks (all idempotent, out-of-tree monkeypatch, LOUD on the unhandled):
  1. register_dtype_strings()   -- stop hpu_model_runner.py:1099 KeyError.
  2. patch_get_kv_cache_spec()  -- swap FullAttentionSpec -> TurboQuantKVSpec for the
                                   turboquant decoder layers (never MLA).
  3. patch_allocation()         -- wrap initialize_kv_cache; for the turboquant
                                   (non-hybrid) case allocate the 4 asymmetric planes.

Everything is shape-driven (H, D from the layer). GQA-only; MLA/hybrid raise LOUD.
"""
from __future__ import annotations

import dataclasses
import os

import torch

from .platform_patch_kvq import KVQ_DTYPES, v_bits_for_dtype

# fp8 K plane dtype (matches the compiled kv_rotate_quant RESULT 0 = float8_e4m3fn).
_K_DTYPE = torch.float8_e4m3fn
# int32 packed-V plane + f32 scale planes (match RESULT 2 / RESULT 1,3).
_V_DTYPE = torch.int32
_SCALE_DTYPE = torch.float32

# Cache of the built spec subclass (built lazily against the installed vLLM).
_TQ_SPEC_CLS = None


def _turboquant_dtype_str(runner) -> str | None:
    """The serve KV dtype string IFF it is a turboquant dtype, else None."""
    dt = getattr(runner.vllm_config.cache_config, "cache_dtype", None)
    return dt if dt in KVQ_DTYPES else None


def _mla_int4_enabled() -> bool:
    return os.environ.get("HPU_MLA_KV_INT4STORE", "0").strip().lower() in ("1", "true", "yes", "on")


def _mla_shrink_enabled() -> bool:
    """Stage 2 capacity win: report the smaller int4 page_size so vLLM computes MORE num_blocks."""
    return os.environ.get("HPU_MLA_KV_SHRINK", "0").strip().lower() in ("1", "true", "yes", "on")


# MLA int4 latent store planes: packed int4 latent + f32 per-group scale + bf16 rope tail.
_MLA_PACK_DTYPE = torch.int32
_MLA_SCALE_DTYPE = torch.float32
_MLA_ROPE_DTYPE = torch.bfloat16
_MLA_V_BITS = 4
_MLA_GROUP = 64
_MLA_ROPE_DIM = 64   # standard MLA qk_rope_head_dim (GLM-5.x, DeepSeek); asserted in the alloc.


def _mla_slot_bytes(head_size: int) -> int:
    """Bytes per (token, head) of the int4 3-plane MLA layout. MUST equal what the alloc allocates
    (int4 latent lanes + bf16 rope + f32 group-scale) or num_blocks vs planes mismatch -> OOM."""
    kv_lora = int(head_size) - _MLA_ROPE_DIM
    lanes = kv_lora // (32 // _MLA_V_BITS)          # int4 -> kv_lora//8 int32 lanes
    n_groups = kv_lora // _MLA_GROUP
    return lanes * 4 + _MLA_ROPE_DIM * 2 + n_groups * 4     # int32 + bf16 + f32


def _tq_mla_real_page_size(self) -> int:
    return int(self.block_size) * int(self.num_kv_heads) * _mla_slot_bytes(self.head_size)


def patch_mla_shrink() -> None:
    """Monkeypatch MLAAttentionSpec.real_page_size_bytes so the MLA latent page reflects the int4
    3-plane layout -> vLLM computes MORE num_blocks (the capacity win). Keeps the spec TYPE as
    MLAAttentionSpec so the KV-manager's EXACT-type checks (type(spec) in (...MLAAttentionSpec...))
    all still pass -- no subclass trap. Gated by HPU_MLA_KV_SHRINK; idempotent; only touches a serve
    that sets the flag."""
    if not _mla_shrink_enabled():
        return
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    if getattr(MLAAttentionSpec, "_kvq_shrunk", False):
        return
    MLAAttentionSpec.real_page_size_bytes = property(_tq_mla_real_page_size)
    MLAAttentionSpec._kvq_shrunk = True


def _eff_v_bits(v_bits: int) -> int:
    """HPU_KVQ_VFP8=1 forces fp8-V (v_bits->8) CONSISTENTLY across spec.page_size + allocation + wire,
    so the spec, the allocated plane dtype, and the quant all agree (fixes the int32-plane vs fp8-source
    index_copy crash). The wire reads it via v_bits_for_dtype; the spec/alloc read it here."""
    if os.environ.get("HPU_KVQ_VFP8", "0").strip().lower() in ("1", "true", "yes", "on"):
        return 8
    return v_bits


def _v_lanes(head_dim: int, v_bits: int) -> int:
    """Last-dim size of the V plane. int4/int2 pack into int32 lanes (8 or 16 codes/lane);
    v_bits==8 is the fp8-V (k8v8) path -> NO packing, one fp8 per element -> lanes == head_dim."""
    if v_bits >= 8:
        return head_dim
    per_lane = 8 if v_bits == 4 else 16
    if head_dim % per_lane != 0:
        raise RuntimeError(
            f"[hpu-kvq-ext] head_dim {head_dim} not a multiple of {per_lane} "
            f"({v_bits}-bit packing) -- unsupported for TurboQuant.")
    return head_dim // per_lane


# TurboQuantKVSpec MUST be MODULE-LEVEL: vLLM pickles the kv_cache_config to each worker
# proc, and a function-local class -> "Can't pickle local object". Defined at module scope
# so pickle resolves it by qualified name. The FullAttentionSpec import is safe here because
# register() imports this module only AFTER vLLM core is loaded. frozen matches the parent so
# the dataclass decorator won't raise ("cannot inherit non-frozen from frozen" and vice-versa).
from vllm.v1.kv_cache_interface import FullAttentionSpec as _FullAttentionSpec

_TQ_FROZEN = bool(getattr(_FullAttentionSpec, "__dataclass_params__", None)
                  and _FullAttentionSpec.__dataclass_params__.frozen)


@dataclasses.dataclass(frozen=_TQ_FROZEN)
class TurboQuantKVSpec(_FullAttentionSpec):
    # v_bits has a DEFAULT so it sorts after the parent's fields (no dataclass
    # "non-default after default" error regardless of the parent's field set).
    v_bits: int = 4

    @property
    def page_size_bytes(self) -> int:
        D = self.head_size
        k_bytes = D * 1                          # fp8 K
        vb = _eff_v_bits(self.v_bits)            # HPU_KVQ_VFP8 forces 8 (spec.v_bits may reset to default)
        if vb >= 8:
            v_bytes = D * 1                       # fp8 V (k8v8), no packing
        else:
            per_lane = 8 if vb == 4 else 16
            v_bytes = (D // per_lane) * 4         # int32 packed-V lanes
        s_bytes = 4 + 4                          # f32 k_scale + f32 v_scale (per token-head)
        return self.block_size * self.num_kv_heads * (k_bytes + v_bytes + s_bytes)


def _get_spec_cls():
    """Return the module-level (picklable) TurboQuantKVSpec."""
    return TurboQuantKVSpec


def _to_turboquant_spec(full_spec, v_bits: int):
    """Rebuild a plain FullAttentionSpec instance as a TurboQuantKVSpec, copying EVERY
    parent field generically (robust to unknown/added FullAttentionSpec fields)."""
    cls = _get_spec_cls()
    fields = {f.name: getattr(full_spec, f.name) for f in dataclasses.fields(full_spec)}
    return cls(**fields, v_bits=v_bits)


# ---------------------------------------------------------------------------
# 1) dtype-string registration -- stop the STR_DTYPE_TO_TORCH_DTYPE KeyError.
# ---------------------------------------------------------------------------
def register_dtype_strings() -> None:
    """hpu_model_runner.py:1099 does ``self.kv_cache_dtype =
    STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]``. The turboquant strings are
    not in that dict -> KeyError at runner init. Register them -> torch.uint8 (a 1-byte
    MARKER; the real per-plane dtypes are set by patch_allocation, and the spec's
    page_size is overridden, so this default is never used for sizing). Idempotent."""
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    for dt in KVQ_DTYPES:
        STR_DTYPE_TO_TORCH_DTYPE.setdefault(dt, torch.uint8)
    # The HPU runner ALSO does the REVERSE lookup HPU_TORCH_DTYPE_TO_STR_DTYPE[kv_cache_dtype]
    # (hpu_model_runner ~:1135) to get kv_cache_dtype_str. uint8 (our marker) isn't there ->
    # KeyError at runner init. Map it back to the primary turboquant string. Guarded: the dict
    # only exists on HPU runners (present in the stock + M3-patched runner); harmless elsewhere.
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPU_TORCH_DTYPE_TO_STR_DTYPE
        HPU_TORCH_DTYPE_TO_STR_DTYPE.setdefault(torch.uint8, KVQ_DTYPES[0])
    except Exception:  # noqa: BLE001 -- dict/module absent on non-HPU or renamed; not fatal here
        pass


# ---------------------------------------------------------------------------
# 2) get_kv_cache_spec -- swap FullAttentionSpec -> TurboQuantKVSpec (never MLA).
# ---------------------------------------------------------------------------
def patch_get_kv_cache_spec() -> None:
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    runner_cls = _find_runner_class(hmr)
    if getattr(runner_cls.get_kv_cache_spec, "_kvq_patched", False):
        return
    _orig = runner_cls.get_kv_cache_spec

    def get_kv_cache_spec(self):  # noqa: ANN001
        spec = _orig(self)
        dt = _turboquant_dtype_str(self)
        if dt is None or getattr(self.model_config, "use_mla", False):
            return spec  # not turboquant, or MLA -> leave every spec as-is (fp8 stays)
        v_bits = v_bits_for_dtype(dt)
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        for name, s in list(spec.items()):
            # EXACT type: never touch MLAAttentionSpec (a FullAttentionSpec subclass).
            if type(s) is FullAttentionSpec:
                _v_lanes(s.head_size, v_bits)  # validate head_dim packability (LOUD)
                spec[name] = _to_turboquant_spec(s, v_bits)
        return spec

    get_kv_cache_spec._kvq_patched = True
    get_kv_cache_spec._kvq_orig = _orig
    runner_cls.get_kv_cache_spec = get_kv_cache_spec


# ---------------------------------------------------------------------------
# 3) initialize_kv_cache -- allocate the 4 asymmetric planes for turboquant layers.
# ---------------------------------------------------------------------------
def patch_allocation() -> None:
    """Wrap HPUModelRunner.initialize_kv_cache. When any group's spec is a
    TurboQuantKVSpec we take over the NON-HYBRID allocation (the only supported case)
    so the turboquant layers get the 4 asymmetric planes instead of the stock two
    symmetric ones (which would over-allocate ~1.33x -> init OOM). Everything else
    delegates to the original. Idempotent; stores _orig. LOUD on any layout it cannot
    handle (hybrid/mamba, shared-KV, GDN-compact, KV-transfer)."""
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    runner_cls = _find_runner_class(hmr, method="initialize_kv_cache")
    if getattr(runner_cls.initialize_kv_cache, "_kvq_patched", False):
        return
    _orig = runner_cls.initialize_kv_cache

    def initialize_kv_cache(self, kv_cache_config):  # noqa: ANN001
        # MLA int4 latent store (GLM/DeepSeek): the spec stays stock MLAAttentionSpec (Stage 1 keeps
        # fp8-sized num_blocks -> no capacity change yet, proves store/read coherence); we take over the
        # alloc to build the 3 leaner planes (int4 latent, bf16 rope, f32 scale) for the MLA layers.
        if _mla_int4_enabled() and getattr(self.model_config, "use_mla", False):
            return _initialize_kv_cache_mla_int4(self, hmr, kv_cache_config)
        spec_cls = _get_spec_cls()
        has_tq = any(isinstance(g.kv_cache_spec, spec_cls)
                     for g in kv_cache_config.kv_cache_groups)
        if not has_tq:
            return _orig(self, kv_cache_config)
        # --- turboquant present: only the pure non-hybrid GQA path is supported. ---
        if getattr(self, "num_mamba_like_layers", 0) > 0:
            raise RuntimeError(
                "[hpu-kvq-ext] TurboQuant KV + hybrid/mamba model is unsupported "
                "(first-draft wire). Serve this model with --kv-cache-dtype fp8_inc.")
        if getattr(self, "shared_kv_cache_layers", None):
            raise RuntimeError("[hpu-kvq-ext] TurboQuant KV + cross-layer KV sharing "
                               "unsupported (first-draft wire).")
        _has_xfer = getattr(hmr, "has_kv_transfer_group", None)
        if callable(_has_xfer) and _has_xfer():
            raise RuntimeError("[hpu-kvq-ext] TurboQuant KV + KV-transfer/disagg "
                               "unsupported (first-draft wire).")
        return _initialize_kv_cache_turboquant(self, hmr, kv_cache_config, spec_cls)

    initialize_kv_cache._kvq_patched = True
    initialize_kv_cache._kvq_orig = _orig
    runner_cls.initialize_kv_cache = initialize_kv_cache


def _initialize_kv_cache_turboquant(self, hmr, kv_cache_config, spec_cls):
    """Faithful re-implementation of HPUModelRunner.initialize_kv_cache's NON-HYBRID
    branch (mirrors vllm_gaudi hpu_model_runner.py:7041-7148 as of 2026-07-01), with
    the TurboQuant branch added. Re-sync this against the source if the runner drifts."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    kv_caches: dict[str, tuple] = {}
    num_blocks = 0
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_spec = None
            for group in kv_cache_config.kv_cache_groups:
                if layer_name in group.layer_names:
                    kv_cache_spec = group.kv_cache_spec
                    break
            assert kv_cache_spec is not None, f"No spec found for {layer_name}"
            assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
            num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
            assert num_blocks >= kv_cache_config.num_blocks

            if isinstance(kv_cache_spec, spec_cls):
                # ---- TurboQuant k8v4: 4 asymmetric planes (fp8 K, int4 V, 2 scales). ----
                shape = self.attn_backend.get_kv_cache_shape(
                    num_blocks + 1, kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                n_slots, H, D = int(shape[0]), int(shape[1]), int(shape[2])
                _vb = _eff_v_bits(kv_cache_spec.v_bits)  # HPU_KVQ_VFP8 forces 8 (spec.v_bits may reset to default)
                lanes = _v_lanes(D, _vb)
                dev = self.device
                key_cache = torch.zeros((n_slots, H, D), dtype=_K_DTYPE, device=dev)
                _vdt = _K_DTYPE if _vb >= 8 else _V_DTYPE  # fp8 V plane for k8v8 / HPU_KVQ_VFP8
                value_cache = torch.zeros((n_slots, H, lanes), dtype=_vdt, device=dev)
                key_scales = torch.zeros((n_slots, H, 1), dtype=_SCALE_DTYPE, device=dev)
                value_scales = torch.zeros((n_slots, H, 1), dtype=_SCALE_DTYPE, device=dev)
                kv_caches[layer_name] = (key_cache, value_cache, key_scales, value_scales)
            elif isinstance(kv_cache_spec, FullAttentionSpec):
                # ---- stock symmetric branch (verbatim from the runner). ----
                kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                    num_blocks + 1, kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                v_cache_shape = None if self.model_config.use_mla else kv_cache_shape
                dtype = kv_cache_spec.dtype
                key_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=self.device)
                if v_cache_shape is not None:
                    value_cache = torch.zeros(v_cache_shape, dtype=dtype, device=self.device)
                else:
                    value_cache = None
                kv_caches[layer_name] = (key_cache, value_cache, None, None)
            elif isinstance(kv_cache_spec, MambaSpec):
                raise RuntimeError("[hpu-kvq-ext] TurboQuant KV + Mamba layer unsupported "
                                   "(first-draft wire).")
            else:
                raise ValueError(
                    f"[hpu-kvq-ext] Unknown KV cache spec for layer {layer_name}: "
                    f"{type(kv_cache_spec)}")

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            if layer_name in getattr(self, "runner_only_attn_layers", set()):
                continue
            layer_names.add(layer_name)
    assert layer_names == set(kv_caches.keys()), "Some layers are not correctly initialized"

    hmr.bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context,
                      self.kv_caches)

    if getattr(self, "enable_bucketing", False):
        self.bucketing_manager.num_hpu_blocks = num_blocks

    self._PAD_BLOCK_ID = num_blocks
    self._PAD_SLOT_ID = num_blocks * self.attn_block_size
    self._MAMBA_PAD_BLOCK_ID = num_blocks
    self._dummy_num_blocks = num_blocks

    if getattr(self, "_compact_gdn_group_ids", None):
        raise RuntimeError("[hpu-kvq-ext] TurboQuant KV + GDN-compact unsupported "
                           "(first-draft wire).")


def _initialize_kv_cache_mla_int4(self, hmr, kv_cache_config):
    """MLA int4 latent store: allocate 3 leaner planes (int4 latent, bf16 rope, f32 scale) per MLA
    layer in place of the stock single fp8 latent plane. Stage 1 keeps the stock (fp8) num_blocks
    (page_size unchanged) so this is capacity-neutral but proves store/read coherence; the planes
    ARE smaller so a later spec.page_size override yields the capacity win. Mirrors the GQA alloc's
    num_blocks/bind/pad exactly."""
    hf = self.model_config.hf_text_config
    kv_lora = int(hf.kv_lora_rank)
    rope_dim = int(hf.qk_rope_head_dim)
    # The Stage-2 shrink page_size (_mla_slot_bytes) assumes the standard MLA rope=64. If a model
    # differs, the reported page_size would mismatch the allocated planes -> OOM: fail LOUD instead.
    if _mla_shrink_enabled():
        assert rope_dim == _MLA_ROPE_DIM, (
            f"[hpu-kvq-ext] HPU_MLA_KV_SHRINK assumes qk_rope_head_dim={_MLA_ROPE_DIM}, got {rope_dim}; "
            "update _MLA_ROPE_DIM (and _mla_slot_bytes) before enabling shrink for this model.")
    lanes = kv_lora // (32 // _MLA_V_BITS)     # int4 -> 8 codes/int32 -> kv_lora//8
    n_groups = kv_lora // _MLA_GROUP
    dev = self.device

    kv_caches: dict[str, tuple] = {}
    num_blocks = 0
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_spec = None
            for group in kv_cache_config.kv_cache_groups:
                if layer_name in group.layer_names:
                    kv_cache_spec = group.kv_cache_spec
                    break
            assert kv_cache_spec is not None, f"No spec found for {layer_name}"
            assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
            num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
            assert num_blocks >= kv_cache_config.num_blocks
            block_size = kv_cache_spec.block_size
            n_slots = (num_blocks + 1) * block_size
            packed = torch.zeros((n_slots, 1, lanes), dtype=_MLA_PACK_DTYPE, device=dev)
            rope = torch.zeros((n_slots, 1, rope_dim), dtype=_MLA_ROPE_DTYPE, device=dev)
            scale = torch.zeros((n_slots, 1, n_groups), dtype=_MLA_SCALE_DTYPE, device=dev)
            # 4-tuple to match the stock MLA binding (key, value, k_scales, v_scales): forward_mqa's
            # split_kv_cache unpacks 4 (then uses only [0] for MLA). We map [0]=packed int4 latent,
            # [1]=rope, [2]=group-scale, [3]=None -- our store/fetch use [0,1,2]; [3] is unused.
            kv_caches[layer_name] = (packed, rope, scale, None)

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        for layer_name in group.layer_names:
            if layer_name in getattr(self, "runner_only_attn_layers", set()):
                continue
            layer_names.add(layer_name)
    assert layer_names == set(kv_caches.keys()), "Some MLA layers are not correctly initialized"

    hmr.bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context,
                      self.kv_caches)

    if getattr(self, "enable_bucketing", False):
        self.bucketing_manager.num_hpu_blocks = num_blocks

    self._PAD_BLOCK_ID = num_blocks
    self._PAD_SLOT_ID = num_blocks * self.attn_block_size
    self._MAMBA_PAD_BLOCK_ID = num_blocks
    self._dummy_num_blocks = num_blocks


# ---------------------------------------------------------------------------
def _find_runner_class(hmr, method: str = "get_kv_cache_spec"):
    """The HPU model-runner class name varies by vllm-gaudi version; locate whichever
    defines `method`."""
    for name in ("HPUModelRunner", "HpuModelRunner", "HPUModelRunnerBase", "HpuModelRunnerBase"):
        cls = getattr(hmr, name, None)
        if cls is not None and method in cls.__dict__:
            return cls
    import inspect
    for _n, cls in inspect.getmembers(hmr, inspect.isclass):
        if cls.__module__ == hmr.__name__ and method in cls.__dict__:
            return cls
    raise RuntimeError(
        f"[hpu-kvq-ext] could not locate the HPU model-runner class defining {method} "
        "in vllm_gaudi.v1.worker.hpu_model_runner")


# ---------------------------------------------------------------------------
def register_spec_manager() -> None:
    """vLLM's single_type_kv_cache_manager.spec_manager_map is keyed by EXACT spec type
    (``spec_manager_map[type(kv_cache_spec)]``, NOT isinstance), so our FullAttentionSpec
    subclass misses -> KeyError at KVCacheCoordinator init. Map TurboQuantKVSpec to the
    SAME manager as FullAttentionSpec: the manager only does block bookkeeping (allocate/
    free/prefix), which is identical regardless of the per-plane byte layout. Idempotent."""
    from vllm.v1.core import single_type_kv_cache_manager as m
    mgr = m.spec_manager_map.get(_FullAttentionSpec)
    if mgr is None:
        raise RuntimeError("[hpu-kvq-ext] FullAttentionSpec not in spec_manager_map "
                           "-- vLLM layout changed; cannot map TurboQuantKVSpec.")
    m.spec_manager_map.setdefault(TurboQuantKVSpec, mgr)


def apply() -> None:
    """Install all runner + core patches. Call once from register() BEFORE the runner
    builds the KV cache. Idempotent."""
    register_dtype_strings()
    register_spec_manager()
    patch_get_kv_cache_spec()
    patch_allocation()
