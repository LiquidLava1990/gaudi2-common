"""Plugin-side fix for GLM-5.1 / DeepSeek-V3.2 MLA attention projections when the
checkpoint stores them as GPTQ/AutoRound-packed int8, but vLLM builds a fused
``fused_qkv_a_proj`` (MergedColumnParallelLinear) in place of the individual
``q_a_proj`` + ``kv_a_proj_with_mqa`` modules.

Background
----------
GLM-5.1 has q_lora_rank=2048 (not None). ``DeepseekV2MLAAttention.__init__``
(deepseek_v2.py:928) therefore takes the ``if self.q_lora_rank is not None`` branch
and builds:

    self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
        ..., quant_config=quant_config, prefix=f"{prefix}.fused_qkv_a_proj")

``DeepSeekV2FusedQkvAProjLinear`` is a ``MergedColumnParallelLinear`` whose
``quant_config`` dispatch prefix is ``...self_attn.fused_qkv_a_proj``.

``AutoRoundHPUConfig.get_quant_method`` is then called with that prefix.
``_checkpoint_is_quantized("...fused_qkv_a_proj")`` looks up
``_norm_prefix("...fused_qkv_a_proj")`` = ``"layers.N.self_attn.fused_qkv_a_proj"``
in the set of prefixes that actually own a ``.qweight`` in the checkpoint shard
headers.  That prefix does NOT exist — the checkpoint stores individual projections
(``q_a_proj.qweight`` and ``kv_a_proj_with_mqa.qweight``), never a fused one.
So ``get_quant_method`` returns ``UnquantizedLinearMethod``.  The model is built
with a plain bf16 ``fused_qkv_a_proj.weight`` parameter (no qweight/qzeros/scales).

At load time, ``load_weights`` processes checkpoint key
``...self_attn.kv_a_proj_with_mqa.qweight`` (and ``q_a_proj.qweight``):

  1. ``_try_load_fp8_indexer_wk``: returns False (not an indexer.wk key).
  2. ``stacked_params_mapping`` loop hits entry
     ``("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1)`` and rewrites name to
     ``...fused_qkv_a_proj.qweight``.  The guard ``param_name == "fused_qkv_a_proj"
     AND name_mapped NOT in params_dict`` fires (model has .weight, not .qweight) ->
     ``continue``.
  3. Falls through to ``else`` branch: ``params_dict[name]`` where
     ``name == "...kv_a_proj_with_mqa.qweight"`` -> **KeyError**.

All five projections share the same failure mode:
  * ``q_a_proj.qweight`` / ``q_a_proj.qzeros`` / ``q_a_proj.scales``  (shard 0)
  * ``kv_a_proj_with_mqa.qweight`` / ``.qzeros`` / ``.scales``          (shard 1)
  * ``q_b_proj``, ``kv_b_proj``, ``o_proj`` (standalone, build correctly as
    GPTQDequantToBF16LinearMethod — these work fine once the fused path is handled)

Fix
---
Monkeypatch the module-level ``_try_load_fp8_indexer_wk`` in
``vllm.model_executor.models.deepseek_v2`` with a wrapper that intercepts the
GPTQ-packed ``{q_a_proj,kv_a_proj_with_mqa}.{qweight,qzeros,scales}`` tensors
BEFORE the stacked_params_mapping loop sees them:

  * Buffers qweight/qzeros/scales for each projection individually (same pattern
    as the wk-indexer patch already uses for ``indexer.wk``).
  * Once all three packed tensors for a projection are buffered, dequantizes to
    dense bf16 via ``GPTQDequantToBF16LinearMethod`` (proven path, identical to
    standalone linears).
  * Loads the resulting bf16 shard into ``fused_qkv_a_proj.weight`` via its
    ``weight_loader(param, bf16_weight, shard_id)`` (shard_id 0 for q_a_proj,
    1 for kv_a_proj_with_mqa) — exactly what the stacked_params_mapping loader
    would do if the fused param had been built as a GPTQ layer.

The bits (8) and group_size (128) are read from the live ``AutoRoundHPUConfig``
when available, with hardcoded (8, 128) as the fallback (matching the GLM-5.1
extra_config).

Risk containment: the wrapper is a no-op for every tensor that is not one of the
two targeted projection .qweight/.qzeros/.scales names; the working FP8 path,
the standalone q_b_proj/kv_b_proj/o_proj, the indexer wk, and all non-DSA models
are byte-for-byte unaffected.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vllm")

_INSTALLED = False

# Bits/group_size for the q_a_proj and kv_a_proj_with_mqa projections.
# GLM-5.1 AutoRound export: extra_config.{q_a_proj,kv_a_proj_with_mqa}.bits = 8,
# top-level group_size = 128.  These defaults are used when the live config is
# not available; they are always overridden from the config in _install().
_MLA_A_PROJ_BITS = 8
_MLA_A_PROJ_GROUP_SIZE = 128

# The two checkpoint sub-names that are fused into fused_qkv_a_proj, and their
# shard indices in MergedColumnParallelLinear (0 = q_a_proj, 1 = kv_a_proj_with_mqa).
_FUSED_SHARDS: dict[str, int] = {
    "q_a_proj": 0,
    "kv_a_proj_with_mqa": 1,
}


def _gptq_dequant_to_bf16(qweight, qzeros, scales, group_size, bits):
    """Dequantize a GPTQ/AutoRound-packed tensor to a dense BF16 [out, in].

    Mirrors indexer_wk_patch._gptq_dequant_wk_to_bf16; kept here as a standalone
    copy so this module has no circular import dependency on indexer_wk_patch.
    """
    from .moe_wna16_hpu import GPTQDequantToBF16LinearMethod

    method = GPTQDequantToBF16LinearMethod(group_size=group_size, bits=bits)
    method.wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    class _FakeLayer:
        pass

    fl = _FakeLayer()
    fl.qweight = qweight
    fl.qzeros = qzeros
    fl.scales = scales

    codes = method.unpack_weight_from_cuda_old_format(fl).to(torch.float32)
    zeros = method.unpack_zeros_from_cuda_old_format(fl).to(torch.float32)
    scales_f = scales.detach().to("cpu").to(torch.float32)
    gs = group_size if group_size != -1 else codes.shape[0]
    g_idx = (torch.arange(codes.shape[0]) // gs).long()
    w = (codes - zeros[g_idx]) * scales_f[g_idx]
    return w.t().contiguous().to(torch.bfloat16)  # [out, in] CPU bf16


def install(bits: int = _MLA_A_PROJ_BITS,
            group_size: int = _MLA_A_PROJ_GROUP_SIZE):
    """Replace deepseek_v2._try_load_fp8_indexer_wk with an extended wrapper
    that also handles GPTQ-packed q_a_proj and kv_a_proj_with_mqa."""
    global _INSTALLED, _MLA_A_PROJ_BITS, _MLA_A_PROJ_GROUP_SIZE
    if _INSTALLED:
        return

    _MLA_A_PROJ_BITS = bits
    _MLA_A_PROJ_GROUP_SIZE = group_size

    try:
        import vllm.model_executor.models.deepseek_v2 as _dsv2
    except Exception:  # noqa: BLE001
        return  # deepseek_v2 not importable

    if not hasattr(_dsv2, "_try_load_fp8_indexer_wk"):
        logger.warning(
            "hpu-moe-ext: deepseek_v2._try_load_fp8_indexer_wk not found; "
            "MLA-proj patch NOT installed (upstream drift)."
        )
        return

    _orig = _dsv2._try_load_fp8_indexer_wk

    def _try_load_gptq_mla_a_proj(
        name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
    ):
        # Determine which MLA-A projection this tensor belongs to, if any.
        matched_shard_name = None
        for shard_name in _FUSED_SHARDS:
            # Match e.g. ".self_attn.q_a_proj.qweight" but NOT ".indexer.wq_b.qweight"
            if (f".self_attn.{shard_name}." in name
                    and name.rsplit(".", 1)[-1] in ("qweight", "qzeros", "scales")):
                matched_shard_name = shard_name
                break

        if matched_shard_name is None:
            # Not one of the two target projections -> delegate entirely.
            return _orig(name, tensor, buf, params_dict, loaded_params,
                         pp_missing_layer_names)

        suffix = name.rsplit(".", 1)[-1]   # qweight | qzeros | scales

        # Pipeline-parallel: layer lives on another rank -> swallow & skip.
        if any(name.startswith(m) for m in pp_missing_layer_names):
            return True

        # Buffer the three packed tensors under a per-layer, per-projection key.
        # Use a namespace that cannot collide with the FP8-wk path's dict entries.
        layer_prefix = name.split(f".{matched_shard_name}.")[0]  # ...self_attn
        buf_key = f"{layer_prefix}::{matched_shard_name}"
        entry = buf.setdefault(buf_key, {})
        entry[suffix] = tensor

        if not all(k in entry for k in ("qweight", "qzeros", "scales")):
            return True  # still waiting for the other two tensors

        qweight = entry["qweight"]
        qzeros = entry["qzeros"]
        scales = entry["scales"]
        del buf[buf_key]

        # Dequantize to dense bf16 on CPU.
        weight_bf16 = _gptq_dequant_to_bf16(
            qweight, qzeros, scales, _MLA_A_PROJ_GROUP_SIZE, _MLA_A_PROJ_BITS
        )  # [out, in] cpu bf16

        # Target the fused param: ...self_attn.fused_qkv_a_proj.weight
        fused_name = f"{layer_prefix}.fused_qkv_a_proj.weight"
        if fused_name not in params_dict:
            # Fallback: model may have been built WITHOUT fused_qkv_a_proj
            # (q_lora_rank is None, or an older vllm build).  In that case
            # the individual param should exist with a plain .weight name.
            plain_name = f"{layer_prefix}.{matched_shard_name}.weight"
            if plain_name not in params_dict:
                raise KeyError(
                    f"hpu-moe-ext MLA-proj patch: neither {fused_name!r} nor "
                    f"{plain_name!r} found in params_dict while loading GPTQ "
                    f"{matched_shard_name} from {name!r}"
                )
            # Plain (unfused) path — just load the bf16 weight directly.
            param = params_dict[plain_name]
            weight_bf16 = weight_bf16.to(param.data.device, dtype=param.data.dtype)
            param.weight_loader(param, weight_bf16)
            loaded_params.add(plain_name)
            return True

        # Fused path: call the MergedColumnParallelLinear weight_loader with the
        # correct shard index (0=q_a_proj, 1=kv_a_proj_with_mqa).
        shard_id = _FUSED_SHARDS[matched_shard_name]
        param = params_dict[fused_name]
        weight_bf16 = weight_bf16.to(param.data.device, dtype=param.data.dtype)
        param.weight_loader(param, weight_bf16, shard_id)
        loaded_params.add(fused_name)
        return True

    _dsv2._try_load_fp8_indexer_wk = _try_load_gptq_mla_a_proj
    _INSTALLED = True
    logger.info(
        "hpu-moe-ext: patched deepseek_v2._try_load_fp8_indexer_wk for "
        "GPTQ/AutoRound MLA q_a_proj+kv_a_proj_with_mqa -> fused_qkv_a_proj "
        "(dequant int8->bf16, load as fused shards 0+1)."
    )
