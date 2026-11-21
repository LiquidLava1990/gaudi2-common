"""Offline pre-repack cache for GLM-5.1 MoE int4 experts on Gaudi.

WHY
---
``MoeWNA16HPUMethod.process_weights_after_loading`` loads each expert stack as
upstream uint8 ``[E, out, in//2]`` and re-packs it to the int32 layout the HPU
``mixture_of_experts`` op consumes (``[E, in, out//8]``) via ``_repack_weight``.
Under **lazy** mode the freed uint8 originals (~52GB across 94 layers) stay
pinned by the repack's read stream-event and can only be unpinned by a host
``synchronize`` while HBM still has room — but on the tighter ``:next`` pool the
pinned originals leave no room for the KV cache.

This module mints a per-rank, per-layer cache of the ALREADY-repacked int32
op-ready tensors during ONE eager run (reusing the real EP/TP sharding + fuse
math, so we never re-derive it offline), then loads it directly into the op on
a lazy run — so there is no uint8 original to allocate and no repack to free.

The eager ``:1.24`` path is UNAFFECTED unless an env var is set:
  * ``HPU_MOE_REPACK_SAVE=<dir>``  -> mint the cache (save after repack)
  * ``HPU_MOE_REPACK_LOAD=<dir>``  -> load from the cache (skip repack)

Both are off by default; setting neither leaves today's behavior identical.

CACHE CONTENTS (one file per rank+layer)
----------------------------------------
``<dir>/rank{RANK}_layer{LAYER}.safetensors`` with keys::

    w13_qweight   int32  [E, in, out13//8]   op-ready (post _repack_weight)
    w2_qweight    int32  [E, in, out2//8]    op-ready
    w13_scales    <dt>   [E, num_groups, out13]   post transpose(1,2)
    w2_scales     <dt>   [E, num_groups, out2]    post transpose(1,2)
    w13_qzeros    uint8  [E, ...]  (AWQ has_zp only; absent for GPTQ-sym)
    w2_qzeros     uint8  [E, ...]  (AWQ has_zp only; absent for GPTQ-sym)

GPTQ-symmetric (GLM-5.1) has NO per-expert qzeros: the shared symmetric zero
is rebuilt deterministically on load via ``_sym_zero`` (no need to cache it).

RANK
----
Under ``--enable-expert-parallel`` with TP=8, every group's rank is the global
process rank, so ``ep_rank == tp_rank == torch.distributed.get_rank()``. We key
the cache on ``layer.ep_rank`` (the rank that actually sharded these experts),
falling back to ``torch.distributed.get_rank()`` / ``$RANK``.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

# safetensors ships in the image (verified 0.7.0).
from safetensors.torch import save_file as _st_save
from safetensors.torch import load_file as _st_load


def save_dir() -> Optional[str]:
    d = os.environ.get("HPU_MOE_REPACK_SAVE")
    return d or None


def load_dir() -> Optional[str]:
    d = os.environ.get("HPU_MOE_REPACK_LOAD")
    return d or None


def _rank(layer) -> int:
    """Global rank that owns this layer's expert shard.

    Prefer ``layer.ep_rank`` (the rank RoutedExperts actually used to shard the
    experts). Under TP=8 EP this equals the global process rank, so a SAVE run
    and a LOAD run agree even if one uses get_rank() as the fallback."""
    r = getattr(layer, "ep_rank", None)
    if r is not None:
        return int(r)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    return int(os.environ.get("RANK", "0") or "0")


def _layer_idx(layer) -> int:
    """Stable per-layer index. ``layer.layer_id`` parses the module prefix
    (e.g. 'model.layers.7.mlp.experts') via extract_layer_index."""
    lid = getattr(layer, "layer_id", None)
    if lid is not None:
        return int(lid)
    # Last resort: a process-local monotonic counter (only if layer_id absent).
    _layer_idx._n = getattr(_layer_idx, "_n", -1) + 1  # type: ignore[attr-defined]
    return _layer_idx._n  # type: ignore[attr-defined]


def cache_path(directory: str, layer) -> str:
    return os.path.join(directory,
                        f"rank{_rank(layer)}_layer{_layer_idx(layer)}.safetensors")


def cache_exists(layer, directory: Optional[str]) -> bool:
    """True iff this rank+layer's cache file is present on disk.

    Lets the MoE method decide PER LAYER whether to take the fast cache-load path
    or fall back to a live eager repack — needed for layers the cache was minted
    WITHOUT (e.g. the DeepSeek-MTP layer 78, added later than the main 0..77 cache)."""
    if not directory:
        return False
    return os.path.exists(cache_path(directory, layer))


def _to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
    """Materialize an HPU/strided tensor as a fresh contiguous CPU tensor.

    Only ever called on the SAVE path, which runs in EAGER mode (PT_HPU_LAZY_MODE=0):
    there a whole-tensor ``.to('cpu')`` on the (already fresh, offset=0) stacked
    op weights is valid — the 'Strided Params Has Value: 0' lazy restriction does
    NOT apply in eager. ``.contiguous()`` guarantees safetensors' no-shared-storage
    requirement."""
    return t.detach().to("cpu").contiguous()


def save_layer(layer, directory: str, has_zp: bool) -> str:
    """Persist THIS rank's processed (op-ready) MoE tensors for THIS layer.

    Must be called AFTER process_weights_after_loading has repacked w13/w2 to
    int32 and transposed the scales — i.e. with the exact tensors that were fed
    to set_weight_packed / set_weight_scale. We read them back off the moe_op so
    what we cache is byte-for-byte what the op will consume."""
    os.makedirs(directory, exist_ok=True)
    tensors = {
        # int32 op-ready packed weights (post _repack_weight)
        "w13_qweight": _to_cpu_contig(layer.w13_qweight.data),
        "w2_qweight": _to_cpu_contig(layer.w2_qweight.data),
        # scales already transposed to [E, num_groups, out]
        "w13_scales": _to_cpu_contig(layer.w13_scales.data),
        "w2_scales": _to_cpu_contig(layer.w2_scales.data),
    }
    if has_zp:
        # AWQ asymmetric: the op consumes per-expert int32 zero-points built by
        # _awq_zero from the loaded uint8 qzeros. Cache the PROCESSED (op-ready)
        # zero-points straight off the moe_op so load needs no re-derivation.
        z13 = [layer.moe_op.w13_list[e].zero_point for e in range(layer.local_num_experts)]
        z2 = [layer.moe_op.w2_list[e].zero_point for e in range(layer.local_num_experts)]
        tensors["w13_qzeros"] = _to_cpu_contig(torch.stack(z13, dim=0))
        tensors["w2_qzeros"] = _to_cpu_contig(torch.stack(z2, dim=0))
    path = cache_path(directory, layer)
    # Metadata is informational only; safetensors requires str->str.
    meta = {
        "rank": str(_rank(layer)),
        "layer": str(_layer_idx(layer)),
        "has_zp": "1" if has_zp else "0",
        "local_num_experts": str(int(layer.local_num_experts)),
        "ep_shift": str(int(layer.ep_rank) * int(layer.local_num_experts)),
        "format": "hpu_moe_repack_v1",
    }
    _st_save(tensors, path, metadata=meta)
    return path


def load_layer_to_device(layer, directory: str) -> dict:
    """Read this rank+layer's cached op-ready tensors from disk to HPU.

    Returns a dict of FRESH device tensors (loaded straight from disk, NOT
    in-place views of any parameter) so they are lazy-safe: each has
    storage_offset 0 and unshared storage, exactly what HPUGraph capture
    requires. Raises FileNotFoundError loudly if the cache is missing (we must
    NOT silently fall back to the repack — that defeats the whole point)."""
    path = cache_path(directory, layer)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"hpu-moe-ext repack cache miss: {path!r} "
            f"(rank={_rank(layer)} layer={_layer_idx(layer)}). "
            f"Mint it first with HPU_MOE_REPACK_SAVE pointing at the same dir "
            f"under an EAGER run.")
    # safetensors returns CPU tensors; KEEP them on CPU. The caller moves each
    # expert's weight CPU->HPU individually so the stacked [E,...] NEVER lands
    # on-device — otherwise under lazy the op clones each per-expert slice for the
    # HPUGraph validator and HBM holds BOTH the [E,...] (~45GB) and the clones
    # (~45GB) = 2x -> the load_model synchronize OOMs with no KV room.
    cpu = _st_load(path)
    return {k: v.contiguous() for k, v in cpu.items()}
