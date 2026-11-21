"""HPU indexer K-cache: allocation hook + insert/gather helpers for DSA.

WHY THIS FILE EXISTS
--------------------
The lightning indexer (``DeepseekV32IndexerCache``, one per sparse MLA layer) needs a
paged K-cache to (a) store the current step's indexer key and (b) re-read past keys when
scoring causal top-k. On CUDA this is a uint8 fp8-packed cache (head_dim+packed-scale=132,
see vllm_core_deepseek_v2.py:654) fed by DeepGEMM kernels. On HPU we score in bf16
(sparse_indexer_hpu.lightning_indexer_topk), so we want a plain bf16, head_dim=128 cache.

THE PROBLEM (BUILD_PLAN.md blocker #3)
-------------------------------------
The HPU model runner's ``get_kv_cache_spec`` (hpu_model_runner.py:1440) only recognises
``Attention``/``MLAAttention``/``MambaBase``/``FusedMoE`` by isinstance and silently DROPS
every other ``AttentionLayerBase`` — including ``DeepseekV32IndexerCache``. So the indexer
cache never gets a spec, never gets memory, and ``k_cache.kv_cache`` stays ``torch.tensor([])``
→ crash on first indexer call. (The GPU runner instead iterates ALL ``AttentionLayerBase``
and calls ``attn_module.get_kv_cache_spec`` — that is the behaviour we restore here, but with
our own bf16/head_dim=128 spec instead of the layer's uint8/132 default.)

A SECOND HPU-ONLY SNAG: ``bind_kv_cache`` (vllm core) raises ``NotImplementedError`` when two
layers map to the same layer index unless the platform is cuda_alike/cpu/xpu. The indexer
cache ``...layers.N.self_attn.indexer.k_cache`` and the MLA attn ``...layers.N.self_attn.attn``
both extract index N, so once the indexer is a real KV layer they collide. On GPU this is the
benign ``pass`` branch; on HPU it raises. We wrap ``bind_kv_cache`` to extend that tolerant
branch to HPU (the per-layer ``forward_context[name].kv_cache = tensor`` binding below it is
unaffected; only the ``runner_kv_caches`` ordering list is impacted, and that list is not
indexed by-layer on HPU).

WHAT WE GET FOR FREE
--------------------
``MLAAttentionSpec`` is a *subclass* of ``FullAttentionSpec``, and the HPU MLA backend's
``get_kv_cache_shape`` returns the FLAT 2D ``(num_blocks*block_size, head_size)`` layout. So by
handing the runner an ``MLAAttentionSpec(head_size=128, dtype=bf16)`` for the indexer, the
existing ``initialize_kv_cache`` FullAttentionSpec branch allocates exactly a flat
``((num_blocks+1)*block_size, 128)`` bf16 key tensor and binds it as the 4-tuple
``(key_cache, None, None, None)`` (v cache is None for MLA). That flat slot-indexed tensor is
the ``(num_blocks*block_size, 1, 128)`` cache the contract asks for (the head dim of 1 is added
back as a view in :class:`IndexerKCache`).

EXACT RUNNER EDIT THIS REPLACES (documented for reviewers; we monkeypatch instead of editing)
---------------------------------------------------------------------------------------------
In ``HpuModelRunner.get_kv_cache_spec`` (hpu_model_runner.py ~line 1453), inside the
``for layer_name, attn_module in forward_ctx.items()`` loop, ADD before ``return``::

    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    ...
        elif isinstance(attn_module, DeepseekV32IndexerCache):
            if layer_name in kv_cache_spec:
                continue
            kv_cache_spec[layer_name] = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=indexer_head_dim,        # 128 (config.index_head_dim), NOT 132
                dtype=torch.bfloat16,              # bf16, NOT the uint8 fp8-packed default
                cache_dtype_str=None,
            )

and change ``bind_kv_cache``'s ``raise NotImplementedError`` (vllm core
v1/worker/utils.py) so HPU joins the ``is_cuda_alike()/is_cpu()/is_xpu()`` ``pass`` branch.
``apply()`` installs both via monkeypatch so no in-image source edit is required.

This module owns ONLY the cache (spec + tensor + insert/gather). The scorer is
``sparse_indexer_hpu.lightning_indexer_topk``; the op wiring is ``sparse_indexer_hpu.forward_oot``
(another file); the consume side is ``sparse_mla.py``. Dims are config-driven — nothing here
hardcodes 128 except as the documented default; callers pass ``index_head_dim`` from hf_config.
"""

from __future__ import annotations

import torch

# bf16 indexer cache: head dim is config.index_head_dim (GLM/DeepSeek both 128 for the
# indexer K). Kept as a module attribute so a future model with a different indexer head
# dim is handled purely by the per-layer spec the patch reads from each cache module.
INDEXER_CACHE_DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# 1) KV-cache SPEC hook — make the HPU runner reserve + allocate the cache.
# ---------------------------------------------------------------------------
def make_indexer_spec(
    block_size: int,
    head_dim: int,
    *,
    alloc_head_size: int | None = None,
    cache_dtype_str: str | None = None,
    compress_ratio: int = 1,
    model_version: str | None = None,
):
    """Return the bf16 MLAAttentionSpec for one indexer cache.

    MLAAttentionSpec ⊂ FullAttentionSpec, so the runner's existing FullAttentionSpec
    allocation branch produces a flat ``((num_blocks+1)*block_size, head_size)`` bf16 tensor
    (v cache None) and binds it as ``(key_cache, None, None, None)``. We deliberately do NOT
    reuse ``DeepseekV32IndexerCache.get_kv_cache_spec`` because that returns head_size=132,
    dtype=uint8 (the fp8-packed CUDA layout) which the HPU bf16 scorer cannot read.

    WHY ``alloc_head_size`` / ``cache_dtype_str`` MIRROR THE MLA ATTENTION SPEC
    --------------------------------------------------------------------------
    The model's MLA attention layers emit ``MLAAttentionSpec(head_size=576,
    cache_dtype_str=cache_config.cache_dtype)`` (mla_attention.py:964 → "auto" by
    default).  If the indexer spec differs in ANY of ``cache_dtype_str`` /
    ``compress_ratio`` / ``model_version``, ``MLAAttentionSpec.merge`` raises
    (kv_cache_interface.py:371), so ``is_kv_cache_spec_uniform`` returns False and
    grouping falls into the ``UniformTypeKVCacheSpecs`` path.  That path sizes each
    layer's tensor by its OWN page_size (indexer=32768·N) but the HPU non-hybrid
    init (hpu_model_runner.py:6215) divides by ``group.kv_cache_spec.page_size_bytes``
    — the UniformType wrapper's *sum* of all per-layer pages — so
    ``32768·N % sum`` ≠ 0 → the AssertionError this fix removes.

    By mirroring the MLA spec's ``cache_dtype_str`` (+ compress_ratio/model_version)
    AND ``head_size`` (``alloc_head_size``), every spec in the dict is byte-identical
    in the merge-relevant fields → ``is_kv_cache_spec_uniform`` is True → the uniform
    branch makes ONE group whose merged spec has page=147456, every tensor sized
    147456·N, and the assert ``147456·N % 147456 == 0`` always holds.

    The indexer only stores ``head_dim`` (128) real columns; with ``alloc_head_size``
    set to the wider MLA head_size (576) the allocated key tensor is over-wide and
    :class:`IndexerKCache` narrows it to ``[:, :head_dim]`` at runtime (a non-contig
    view; verified working with ``index_copy_``/``index_select`` on Gaudi 2).
    NOTE: the allocated width is driven by the *merged group spec* (the MLA layer's
    head_size, since it sorts first), so this is over-wide regardless of
    ``alloc_head_size``; mirroring head_size here only makes the spec set robust to
    dict ordering and to any future routing through ``UniformTypeKVCacheSpecs``.
    """
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=alloc_head_size if alloc_head_size is not None else head_dim,
        dtype=INDEXER_CACHE_DTYPE,
        cache_dtype_str=cache_dtype_str,
        compress_ratio=compress_ratio,
        model_version=model_version,
    )


def _indexer_head_dim(cache_module, default: int = 128) -> int:
    """Indexer K head dim (config.index_head_dim). The DeepseekV32IndexerCache stores
    head_dim = index_head_dim + index_head_dim//quant_block_size*4 (the packed-scale
    width). Recover the true index_head_dim: it is the largest power-of-two-ish value such
    that head_dim - hd == hd//128*4. In practice quant_block_size=128 and index_head_dim=128
    → stored 132. We read the canonical value if the module exposes it, else strip the
    4-byte-per-128 scale tail, else fall back to ``default``."""
    hd = getattr(cache_module, "index_head_dim", None)
    if isinstance(hd, int) and hd > 0:
        return hd
    stored = getattr(cache_module, "head_dim", None)
    if isinstance(stored, int) and stored > 0:
        # invert head_dim = d + d//128*4  (quant_block_size=128, 4 bytes fp32 scale)
        for d in (stored, stored - stored // 132 * 4):
            if d + (d // 128) * 4 == stored:
                return d
        return stored  # unknown packing — use as-is (bf16 cache of that width still works)
    return default


def patch_get_kv_cache_spec() -> None:
    """Wrap ``HPUModelRunner.get_kv_cache_spec`` to add a spec for every
    ``DeepseekV32IndexerCache`` in the static forward context. Idempotent."""
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    runner_cls = _find_runner_class(hmr)
    if getattr(runner_cls.get_kv_cache_spec, "_dsa_indexer_patched", False):
        return
    _orig = runner_cls.get_kv_cache_spec

    def get_kv_cache_spec(self):  # noqa: ANN001
        spec = _orig(self)
        try:
            from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
            from vllm.v1.kv_cache_interface import MLAAttentionSpec
        except Exception:  # noqa: BLE001 — model class not importable ⇒ nothing to add
            return spec
        block_size = self.vllm_config.cache_config.block_size

        # The model's MLA attention layers are already in ``spec`` (the runner built
        # them). Mirror their merge-relevant fields (cache_dtype_str / compress_ratio
        # / model_version) and head_size onto every indexer spec so the whole dict is
        # uniform under ``MLAAttentionSpec.merge`` — see make_indexer_spec docstring
        # for why a mismatch here triggers the line-6215 page_size AssertionError.
        mla_ref = next(
            (
                s
                for s in spec.values()
                if isinstance(s, MLAAttentionSpec) and s.block_size == block_size
            ),
            None,
        )
        if mla_ref is not None:
            alloc_hs = mla_ref.head_size
            cds = mla_ref.cache_dtype_str
            compress_ratio = mla_ref.compress_ratio
            model_version = mla_ref.model_version
        else:
            # No MLA attention layer found (unexpected for DSA, but stay safe): fall
            # back to the indexer's own dims/defaults — uniform among indexers at least.
            alloc_hs = None
            cds = None
            compress_ratio = 1
            model_version = None

        fwd_ctx = self.vllm_config.compilation_config.static_forward_context
        for layer_name, module in fwd_ctx.items():
            if layer_name in spec:
                continue
            if isinstance(module, DeepseekV32IndexerCache):
                hd = _indexer_head_dim(module)
                spec[layer_name] = make_indexer_spec(
                    block_size,
                    hd,
                    alloc_head_size=alloc_hs,
                    cache_dtype_str=cds,
                    compress_ratio=compress_ratio,
                    model_version=model_version,
                )
        return spec

    get_kv_cache_spec._dsa_indexer_patched = True
    get_kv_cache_spec._dsa_orig = _orig
    runner_cls.get_kv_cache_spec = get_kv_cache_spec


def _find_runner_class(hmr):
    """The HPU model-runner class is named HpuModelRunner / HPUModelRunner depending on
    the vllm-gaudi version; locate whichever defines get_kv_cache_spec."""
    for name in ("HPUModelRunner", "HpuModelRunner", "HPUModelRunnerBase", "HpuModelRunnerBase"):
        cls = getattr(hmr, name, None)
        if cls is not None and "get_kv_cache_spec" in cls.__dict__:
            return cls
    # Fallback: any class in the module that defines get_kv_cache_spec.
    import inspect

    for _n, cls in inspect.getmembers(hmr, inspect.isclass):
        if cls.__module__ == hmr.__name__ and "get_kv_cache_spec" in cls.__dict__:
            return cls
    raise RuntimeError(
        "[hpu-dsa-ext] could not locate the HPU model-runner class defining "
        "get_kv_cache_spec in vllm_gaudi.v1.worker.hpu_model_runner"
    )


# ---------------------------------------------------------------------------
# 2) bind_kv_cache hook — tolerate the indexer/attn same-index collision on HPU.
# ---------------------------------------------------------------------------
def patch_bind_kv_cache() -> None:
    """Patch the ``bind_kv_cache`` symbol the HPU runner imported so the multi-layer-per-index
    branch does the cuda_alike-style ``pass`` on HPU instead of raising NotImplementedError.

    The indexer cache and the MLA attn of the same decoder layer share a layer index; once the
    indexer is a real KV layer they land in the same ``index2name`` bucket. The only effect of
    the tolerant branch is the *order/identity* of entries appended to ``runner_kv_caches`` (a
    flat list the HPU runner does not address by layer — it binds caches via
    ``forward_context[name].kv_cache = tensor``, which the same function does afterwards and
    which is unaffected). So tolerating the collision is safe on HPU. Idempotent.
    """
    from vllm_gaudi.v1.worker import hpu_model_runner as hmr

    if getattr(hmr, "_dsa_bind_kv_cache_patched", False):
        return
    _orig_bind = hmr.bind_kv_cache

    def bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):  # noqa: ANN001
        try:
            return _orig_bind(kv_caches, forward_context, runner_kv_caches, num_attn_module)
        except NotImplementedError:
            # Reproduce the original semantics but with the collision tolerated (the
            # cuda_alike/cpu/xpu ``pass`` branch), then bind to the forward context.
            return _bind_kv_cache_tolerant(
                kv_caches, forward_context, runner_kv_caches, num_attn_module
            )

    bind_kv_cache._dsa_orig = _orig_bind
    hmr.bind_kv_cache = bind_kv_cache
    hmr._dsa_bind_kv_cache_patched = True

    # hpu_worker.determine_available_memory (hpu_worker.py:388) ALSO calls bind_kv_cache, via its
    # OWN module-level import (hpu_worker.py:32 `from vllm.v1.worker.utils import bind_kv_cache`),
    # a separate binding the hmr patch above does NOT touch -> it still raised NotImplementedError
    # on the DSA indexer/attn same-layer-index collision during memory profiling. Rebind that copy
    # to the same tolerant wrapper (same underlying stock fn, so _orig_bind is identical).
    try:
        from vllm_gaudi.v1.worker import hpu_worker as _hw
        _hw.bind_kv_cache = bind_kv_cache
    except Exception:  # noqa: BLE001
        pass


def _bind_kv_cache_tolerant(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):
    """HPU-tolerant re-implementation of vllm core ``bind_kv_cache``: identical except the
    ``len(layer_names) > 1`` case does ``pass`` (append all) rather than raising. Kept in lock
    step with the upstream body; ``_compat.verify`` guards the imported symbols against drift.

    Reached only from the ``except NotImplementedError`` fallback, where the original may have
    already partially populated ``runner_kv_caches`` before raising (it appends per layer-index,
    and raises at the first colliding index *after* appending earlier ones). So we clear it and
    rebuild from scratch rather than asserting it is empty."""
    from collections import defaultdict

    from vllm.model_executor.models.utils import extract_layer_index

    runner_kv_caches.clear()
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])
    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


# ---------------------------------------------------------------------------
# 3) Runtime insert / gather over the bound cache (mirrors VLLMKVCache).
# ---------------------------------------------------------------------------
def as_key_tensor(bound_cache, index_head_dim: int | None = None) -> torch.Tensor:
    """Normalise whatever ``DeepseekV32IndexerCache.kv_cache`` was bound to into the flat
    2D key tensor ``[num_blocks*block_size, index_head_dim]``.

    After our spec hook the runner binds the MLA 4-tuple ``(key_cache, None, None, None)``
    (key_cache is flat 2D). We also accept a bare tensor or a list/tuple whose first element
    is the key tensor, for robustness against runner-layout changes.

    OVER-WIDE BACKING TENSOR → NARROW TO ``index_head_dim``
    ------------------------------------------------------
    To pass ``initialize_kv_cache`` the indexer spec is made page-uniform with the MLA
    attention spec (see make_indexer_spec), so the runner allocates the indexer key
    tensor at the MLA head_size (e.g. 576) rather than the indexer's real ``index_head_dim``
    (128).  Only the first ``index_head_dim`` columns are real indexer keys; the rest are
    never written or read.  We return a ``narrow(1, 0, index_head_dim)`` view so every
    downstream op (insert/index_copy_, gather_blocks/unflatten+index_select, gather_rows)
    operates on exactly the 128-wide key cache.  The narrow is a non-contiguous view but
    ``index_copy_`` / ``index_select`` / ``unflatten`` over it are verified on Gaudi 2.
    When ``index_head_dim`` is None or already equals the backing width, this is a no-op
    (keeps the pre-uniform behaviour and any direct 128-wide allocation working)."""
    c = bound_cache
    if isinstance(c, (tuple, list)):
        c = c[0]
    if c is None or (isinstance(c, torch.Tensor) and c.numel() == 0):
        raise RuntimeError(
            "[hpu-dsa-ext] indexer K-cache is empty (tensor([])) — the get_kv_cache_spec "
            "patch did not run or did not include this indexer layer. Call "
            "indexer_cache_hpu.apply() during register()."
        )
    # Flatten any leading singleton head dim, e.g. (N,1,D) -> (N,D); keep (N,D) as is.
    if c.dim() == 3 and c.shape[1] == 1:
        c = c.squeeze(1)
    # Narrow an over-allocated (page-uniform) key tensor to the real indexer head dim.
    if index_head_dim is not None and 0 < index_head_dim < c.shape[-1]:
        c = c.narrow(c.dim() - 1, 0, index_head_dim)
    return c


class IndexerKCache:
    """Thin runtime view over one sparse-MLA layer's bound indexer K-cache.

    Construct per forward call from the module's bound cache::

        kc = IndexerKCache(layer.indexer.k_cache.kv_cache, block_size=attn_md.block_size,
                           index_head_dim=indexer_op.head_dim)   # 128
        kc.insert(k_bf16, slot_mapping)                  # write current step's keys
        past = kc.gather_blocks(block_list)              # [n_blk*block_size, D] prefix keys
        rows = kc.gather_rows(flat_slot_indices)         # [N, D] arbitrary selected keys

    All ops use the FLAT slot convention ``slot = block_id * block_size + offset`` — identical
    to the MLA latent cache, so the SAME ``attn_metadata.slot_mapping`` / ``block_list`` drive
    both. ``k`` is stored bf16 (no quant); the scorer reads it directly.

    ``index_head_dim`` is the real indexer key width (config.index_head_dim, 128). The bound
    tensor may be wider (allocated at the MLA head_size for page-uniform kv-cache init); when
    so, :func:`as_key_tensor` narrows it to ``index_head_dim`` and ``self.head_dim`` is 128.
    Pass it from ``SparseAttnIndexer.head_dim``; omit (None) to use the bound width as-is.
    """

    __slots__ = ("key", "block_size", "head_dim")

    def __init__(self, bound_cache, block_size: int, index_head_dim: int | None = None):
        self.key = as_key_tensor(bound_cache, index_head_dim)  # [nblk*block_size, head_dim]
        self.block_size = int(block_size)
        self.head_dim = self.key.shape[-1]

    # --- write -------------------------------------------------------------
    def insert(self, k: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
        """Write current-step indexer keys into the cache at flat slots.

        k: ``[T, head_dim]`` or ``[T, 1, head_dim]`` bf16 indexer keys (MQA, single head).
        slot_mapping: ``[T]`` int flat slot ids (``attn_metadata.slot_mapping`` flattened).
        Mirrors ``VLLMKVCache.forward`` (``cache.index_copy_(0, slot_mapping, input)``).
        Returns the cache tensor (for op-graph dependency, like the MLA path)."""
        if k.dim() == 3:
            k = k.reshape(-1, k.shape[-1])
        if k.dtype != self.key.dtype:
            k = k.to(self.key.dtype)
        slots = slot_mapping.reshape(-1).to(torch.long)
        # Truncate to the real token count (decode/spec-decode may pad slot_mapping/k apart);
        # callers pass matching lengths, but guard against an over-long k.
        n = slots.shape[0]
        if k.shape[0] != n:
            k = k[:n]
        self.key.index_copy_(0, slots, k)
        return self.key

    # --- read: paged prefix (blocks) --------------------------------------
    def gather_blocks(self, block_list: torch.Tensor) -> torch.Tensor:
        """Gather all keys in ``block_list`` (``attn_metadata.block_list``) → flat
        ``[len(block_list)*block_size, head_dim]`` in block_list order.

        Mirrors ``latent_cache_k.fetch_from_cache(cache.unflatten(0,(-1,block_size)),
        block_list)`` then flattening. The caller masks the unused tail of the last partial
        block via the causal bound (the scorer's ``key_pos > abs_query_pos`` fill)."""
        if block_list is None or block_list.numel() == 0:
            return self.key.new_zeros((0, self.head_dim))
        blocks = block_list.reshape(-1).to(torch.long)
        unflat = self.key.unflatten(0, (-1, self.block_size))     # [num_blocks, bs, D]
        sel = unflat.index_select(0, blocks)                       # [nblk, bs, D]
        return sel.reshape(-1, self.head_dim)                      # [nblk*bs, D]

    # --- read: arbitrary flat rows (selected keys for consume) -------------
    def gather_rows(self, flat_slots: torch.Tensor) -> torch.Tensor:
        """Gather keys at arbitrary flat slot indices → ``[N, head_dim]``.

        Used by the gather-consume path (BUILD_PLAN.md consume option B): the top-k indices,
        mapped from per-sequence local positions to flat cache slots, select the ≤index_topk
        latent-KV rows per query. NOTE: ``-1`` padding from short top-k rows must be resolved
        by the CONSUMER via sentinel-column scatter BEFORE calling here — never pass ``-1``
        (it would wrap to the last slot). This method intentionally does not clamp."""
        idx = flat_slots.reshape(-1).to(torch.long)
        return self.key.index_select(0, idx)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def apply() -> None:
    """Install both runner patches. Call once from ``hpu_dsa_ext.register()`` BEFORE the
    runner builds the KV cache (i.e. at plugin load). Idempotent."""
    patch_get_kv_cache_spec()
    patch_bind_kv_cache()
