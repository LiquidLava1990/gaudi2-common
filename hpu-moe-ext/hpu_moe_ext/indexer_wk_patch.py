"""Plugin-side fix for the GLM-5.1 / DeepSeek-V3.2 DSA lightning-indexer when the
checkpoint stores ``indexer.wk`` as GPTQ-packed int (AutoRound), not FP8.

Background
----------
``Indexer.__init__`` (deepseek_v2.py:636) FUSES wk + weights_proj into a single
``MergedColumnParallelLinear`` named ``wk_weights_proj`` built with
``quant_config=None`` -> the model only owns a plain BF16
``wk_weights_proj.weight`` ([head_dim + n_head, hidden] = [160, 6144] for GLM-5.1).
There is NO ``wk_weights_proj.qweight`` param.

``load_weights`` (deepseek_v2.py:1405) calls module-level
``_try_load_fp8_indexer_wk`` first, then falls through to ``stacked_params_mapping``
which contains ``("wk_weights_proj", "wk", 0)``. With an FP8 checkpoint, wk is
buffered+dequantized into shard 0 by ``_try_load_fp8_indexer_wk``. With an
AutoRound checkpoint, wk arrives as ``indexer.wk.{qweight,qzeros,scales}``;
``_try_load_fp8_indexer_wk`` returns False for those (not fp8 .weight, no
weight_scale_inv) -> the stacked mapping rewrites ``...wk.qweight`` to
``...wk_weights_proj.qweight`` -> ``params_dict[...]`` KeyError (deepseek_v2.py:1447).

Fix
---
Monkeypatch the module-level ``_try_load_fp8_indexer_wk`` in
``vllm.model_executor.models.deepseek_v2`` with a wrapper that:
  * delegates EVERYTHING that is not an isolated ``indexer.wk.{qweight,qzeros,
    scales}`` to the original (FP8 path + all non-wk tensors are untouched);
  * for the GPTQ case, buffers the three tensors in the SAME ``buf`` dict the
    caller already threads through (``_pending_wk_fp8``), then dequantizes to
    BF16 via the proven ``GPTQDequantToBF16LinearMethod`` and writes the result
    into shard 0 of ``wk_weights_proj.weight`` -- exactly mirroring the FP8 path's
    ``param.weight_loader(param, weight_bf16, 0)`` call.

weights_proj (shard 1) is plain BF16 and already loads via stacked_params_mapping
-- left untouched. wq_b is a standalone ``ReplicatedLinear(quant_config=...)`` and
loads through the normal ``AutoRoundHPUConfig`` -> ``GPTQDequantToBF16LinearMethod``
dispatch -- no special handling.

Risk containment: the wrapper is a no-op for every checkpoint that is not an
AutoRound DSA model (returns the original's result for non-wk-GPTQ names), so the
working Qwen path and the FP8 DeepSeek path are byte-for-byte unaffected.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vllm")

_INSTALLED = False

# Indexer wk/wq_b are stored 8-bit in the GLM-5.1 AutoRound export (extra_config:
# indexer.wk -> bits:8, indexer.wq_b -> bits:8, indexer.weights_proj -> bits:16).
# group_size comes from the top-level quantization_config (128). We read these from
# the live AutoRoundHPUConfig when available and fall back to (8, 128).
_INDEXER_WK_BITS = 8


def _gptq_dequant_wk_to_bf16(qweight, qzeros, scales, group_size, bits):
    """Dequantize a GPTQ/AutoRound-packed wk tensor to a dense BF16 [out, in].

    Reuses the PROVEN GPTQDequantToBF16LinearMethod's inherited CPU unpackers
    (``unpack_weight_from_cuda_old_format`` / ``unpack_zeros_from_cuda_old_format``,
    hpu_gptq.py) so the bit layout, the autogptq stored-minus-one (+1) zero
    convention, and the per-group ``(code - zero) * scale`` math are IDENTICAL to the
    standalone linear path (moe_wna16_hpu.py:314). We replicate only the final
    transpose+cast here (instead of calling ``process_weights_after_loading``) so we
    stay on CPU -- the method's own routine hard-codes ``.to("hpu")`` and
    ``delattr``s, which would force a device round-trip and a device-acquire at this
    exact call. The returned CPU bf16 [out_features, in_features] (= [head_dim,
    hidden] for wk) is handed to the loader, which copies it onto the param device --
    exactly mirroring the FP8 path.
    """
    from .moe_wna16_hpu import GPTQDequantToBF16LinearMethod

    method = GPTQDequantToBF16LinearMethod(group_size=group_size, bits=bits)
    # The unpackers read self.wf (set in the method's PWAL); set it up-front here.
    method.wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    class _FakeLayer:
        pass

    fl = _FakeLayer()
    fl.qweight = qweight
    fl.qzeros = qzeros
    fl.scales = scales

    codes = method.unpack_weight_from_cuda_old_format(fl).to(torch.float32)  # [in,out]
    zeros = method.unpack_zeros_from_cuda_old_format(fl).to(torch.float32)   # [groups,out] (+1 applied)
    scales_f = scales.detach().to("cpu").to(torch.float32)                   # [groups,out]
    gs = group_size if group_size != -1 else codes.shape[0]
    g_idx = (torch.arange(codes.shape[0]) // gs).long()                      # desc_act=False
    w = (codes - zeros[g_idx]) * scales_f[g_idx]                            # [in,out]
    return w.t().contiguous().to(torch.bfloat16)                            # [out,in] CPU bf16


def install():
    """Replace deepseek_v2._try_load_fp8_indexer_wk with the GPTQ-aware wrapper."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import vllm.model_executor.models.deepseek_v2 as _dsv2
    except Exception:  # noqa: BLE001
        return  # deepseek_v2 not importable -> nothing to patch (not a DSA build)

    if not hasattr(_dsv2, "_try_load_fp8_indexer_wk"):
        # Upstream renamed/removed the helper -> fail loud rather than silently
        # leave the indexer broken.
        logger.warning(
            "hpu-moe-ext: deepseek_v2._try_load_fp8_indexer_wk not found; "
            "GPTQ DSA-indexer patch NOT installed (upstream drift)."
        )
        return

    _orig = _dsv2._try_load_fp8_indexer_wk

    def _try_load_gptq_or_fp8_indexer_wk(
        name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
    ):
        # Only intercept an ISOLATED indexer.wk tensor; never the fused param itself.
        if "indexer.wk." not in name or "wk_weights" in name:
            return _orig(
                name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
            )

        suffix = name.rsplit(".", 1)[-1]
        if suffix not in ("qweight", "qzeros", "scales"):
            # FP8 wk (.weight / weight_scale_inv) or anything else -> original path.
            return _orig(
                name, tensor, buf, params_dict, loaded_params, pp_missing_layer_names
            )

        # ---- GPTQ/AutoRound-packed wk branch ----
        layer_prefix = name.rsplit(".wk.", 1)[0]  # ...self_attn.indexer
        fused_name = f"{layer_prefix}.wk_weights_proj.weight"

        # Pipeline-parallel: this layer lives on another rank -> swallow & skip.
        if any(name.startswith(m) for m in pp_missing_layer_names):
            return True

        # Buffer the three packed tensors in the caller's dict under a distinct key
        # so we never collide with the FP8 path's {"weight","scale"} entries.
        key = layer_prefix + "::gptq_wk"
        entry = buf.setdefault(key, {})
        entry[suffix] = tensor
        if not all(k in entry for k in ("qweight", "qzeros", "scales")):
            return True  # still waiting for the other tensors

        qweight = entry["qweight"]
        qzeros = entry["qzeros"]
        scales = entry["scales"]
        del buf[key]

        # indexer.wk bit width is PER-LAYER and must be read from the shard shapes,
        # NOT hardcoded: GLM-5.1's AutoRound export lists indexer.wk -> bits:8 in
        # extra_config for the MAIN layers (0..77), but the MTP layer (78) has NO
        # extra_config entry and falls back to the default 4-bit (qweight [768,128]
        # vs the 8-bit layers' [1536,128]). Hardcoding 8 there unpacks with the wrong
        # pack factor -> "size of tensor a (128) must match b (64)" at dequant.
        # Derive bits from the GPTQ packing geometry (immune to a missing/lying
        # extra_config): scales is [groups, out], qzeros is [groups, out//(32//bits)],
        # so 32//bits = out_cols/qzero_cols => bits = 32 * qzeros.shape[1] / scales.shape[1].
        group_size = 128
        try:
            pack = int(scales.shape[1]) // int(qzeros.shape[1])  # values packed per int32
            bits = 32 // pack if pack in (2, 4, 8) else _INDEXER_WK_BITS
        except Exception:  # noqa: BLE001
            bits = _INDEXER_WK_BITS

        weight_bf16 = _gptq_dequant_wk_to_bf16(
            qweight, qzeros, scales, group_size, bits
        )  # [out=head_dim, in=hidden] bf16, on CPU

        if fused_name not in params_dict:
            # Should never happen (the fused BF16 param is always built), but fail
            # loud rather than KeyError deep in the loader.
            raise KeyError(
                f"hpu-moe-ext indexer patch: {fused_name} missing from params_dict "
                f"(while loading GPTQ wk from {name})"
            )

        param = params_dict[fused_name]
        weight_bf16 = weight_bf16.to(param.data.device, dtype=param.data.dtype)
        # shard_id=0 == wk slot (rows 0:head_dim) of the fused [head_dim+n_head, in].
        param.weight_loader(param, weight_bf16, 0)
        loaded_params.add(fused_name)
        return True

    _dsv2._try_load_fp8_indexer_wk = _try_load_gptq_or_fp8_indexer_wk

    # ALSO patch the DeepSeek-MTP draft module: deepseek_mtp.py does
    # `from .deepseek_v2 import _try_load_fp8_indexer_wk`, binding the ORIGINAL fn into its
    # OWN namespace at import time. Its load_weights calls that local name, so patching only
    # deepseek_v2 leaves the MTP (layer-78) indexer wk on the UNPATCHED path ->
    # KeyError 'model.layers.78.mtp_block.self_attn.indexer.wk_weights_proj.qweight'. Rebind
    # the re-exported reference too (no-op if the MTP module isn't present / didn't import it).
    try:
        import vllm.model_executor.models.deepseek_mtp as _dsmtp
        if hasattr(_dsmtp, "_try_load_fp8_indexer_wk"):
            _dsmtp._try_load_fp8_indexer_wk = _try_load_gptq_or_fp8_indexer_wk
    except Exception:  # noqa: BLE001
        pass  # MTP module absent -> nothing to rebind

    _INSTALLED = True
    logger.info(
        "hpu-moe-ext: patched deepseek_v2._try_load_fp8_indexer_wk (+ deepseek_mtp re-export) "
        "for GPTQ/AutoRound DSA-indexer wk (dequant->bf16->wk_weights_proj shard 0)."
    )
