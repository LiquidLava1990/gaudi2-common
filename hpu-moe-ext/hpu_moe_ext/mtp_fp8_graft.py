"""Graft the PRE-QUANTIZED fp8 MTP layer (78) onto the int4 serving checkpoint — COMPLIANT (no bf16).

WHY: the int4 export's layer 78 is DEAD; the only real native-MTP weights are zai-org's bf16 layer-78,
which tests/mtp_fp8_quant.py has already quantized OFFLINE to fp8 (glm51-fp8-l78, e4m3 + per-output-channel
weight_scale; norms/gate kept fp16). This graft loads ONLY that fp8 (never bf16 in the serve process) and
serves it via vllm-gaudi's HPU fp8 path (Fp8LinearMethod / HPUFp8MoEMethod -> apply_fp8_linear_hpu /
fp8_gemm_v2). CPU oracle: fp8 == bf16 == 73.3% accept (zero loss). Same provenance as the int4 main model
(both quantized from a bf16 original; nothing bf16 is stored or served).

WHAT (gate: HPU_MTP_FP8_GRAFT=<dir with l78_fp8.safetensors + index>, default off):
  1. Quant-method override: for every layer-78 module, AutoRound get_quant_method returns the vllm fp8
     method (Fp8Config, is_checkpoint_fp8_serialized=True -> loads fp8 directly, no re-quant) instead of int4;
     norms/gate (no weight_scale) fall back to unquantized fp16.
  2. Weight substitution: DeepSeekMTP.load_weights' iterable is wrapped — int4 layer-78 entries dropped,
     the fp8 tensors (+ <name>.weight_scale) from the graft dir injected under the same base names; vllm's
     Fp8 weight loaders route '<name>' -> weight and '<name>.weight_scale' -> the scale param.
mtp_share_head_patch (embed/head tie) still applies. LOUD: raises if the graft dir/index is missing.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("vllm")
_ENV = "HPU_MTP_FP8_GRAFT"


def _graft_dir() -> str | None:
    d = os.environ.get(_ENV, "").strip()
    return d or None


def _spec_layer_prefixes() -> list[str]:
    ckpt = os.environ.get("HPU_MOE_EXT_CKPT", "").strip()
    cfg_file = os.path.join(ckpt, "config.json")
    if not ckpt or not os.path.exists(cfg_file):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV} set but HPU_MOE_EXT_CKPT/config.json unreadable")
    c = json.load(open(cfg_file))
    n = int(c.get("num_hidden_layers", 0)); k = int(c.get("num_nextn_predict_layers", 0))
    if not (n and k):
        raise RuntimeError("[hpu-moe-ext] checkpoint config has no nextn layers")
    return [f"model.layers.{n + i}." for i in range(k)]


def _fp8_config():
    """A per-channel, dynamic-activation fp8 config for the already-fp8-serialized graft weights."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    return Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                     ignored_layers=None, weight_block_size=None)


def install() -> None:
    d = _graft_dir()
    if not d:
        return
    idx_file = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx_file):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}={d} but {idx_file} is missing")
    wm = json.load(open(idx_file))["weight_map"]
    prefixes = _spec_layer_prefixes()
    # names with a sibling .weight_scale are fp8-quantized; the rest (norms/gate) stay fp16/unquantized.
    fp8_quant_weights = {n for n in wm if (n + ".weight_scale") in wm}

    # (1) quant-method override for spec-layer modules -> vllm fp8 (loads the serialized fp8 directly).
    from .auto_round_hpu import AutoRoundHPUConfig
    if not getattr(AutoRoundHPUConfig.get_quant_method, "_mtp_fp8", False):
        _orig_gqm = AutoRoundHPUConfig.get_quant_method
        _fp8cfg = _fp8_config()

        def _gqm(self, layer, prefix):
            if any(prefix.startswith(p) for p in prefixes):
                from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
                from ._compat import is_fused_moe_layer
                # FusedMoE + fp8 Linears use vllm's fp8 method (HPU-patched); norms/router gate (kept fp16)
                # have no weight_scale -> unquantized.
                if is_fused_moe_layer(layer):
                    return _fp8cfg.get_quant_method(layer, prefix)
                if isinstance(layer, LinearBase):
                    m = _fp8cfg.get_quant_method(layer, prefix)
                    return m if m is not None else UnquantizedLinearMethod()
                return None
            return _orig_gqm(self, layer, prefix)

        _gqm._mtp_fp8 = True
        AutoRoundHPUConfig.get_quant_method = _gqm

    # (2) weight substitution in the MTP loader.
    from vllm.model_executor.models import deepseek_mtp as _mtp
    if not getattr(_mtp.DeepSeekMTP.load_weights, "_mtp_fp8", False):
        _orig_lw = _mtp.DeepSeekMTP.load_weights

        def _lw(self, weights):
            graft_names = [n for n in wm if any(n.startswith(p) for p in prefixes)]
            if not graft_names:
                raise RuntimeError(f"[hpu-moe-ext] {_ENV}: graft index has NO spec-layer weights")
            from safetensors import safe_open
            _handles: dict[str, object] = {}

            def _read(name):
                fn = wm[name]
                if fn not in _handles:
                    _handles[fn] = safe_open(os.path.join(d, fn), framework="pt", device="cpu")
                return _handles[fn].get_tensor(name)

            def _gen():
                dropped = 0
                for name, w in weights:
                    if any(name.startswith(p) for p in prefixes):
                        dropped += 1
                        continue            # int4 entries for the grafted layer
                    yield name, w
                logger.warning("[hpu-moe-ext] MTP FP8 GRAFT: dropped %d int4 layer-78 entries; injecting "
                               "%d fp8 tensors (%d quantized) from %s",
                               dropped, len(graft_names), len(fp8_quant_weights), d)
                for name in graft_names:
                    # the fp8 dir stores the scale as "<...>.weight.weight_scale" (sibling-of-tensor); vllm's
                    # fp8 module registers the param as "<...>.weight_scale" (sibling of .weight). rename it.
                    out = name
                    if name.endswith(".weight.weight_scale"):
                        out = name[: -len(".weight.weight_scale")] + ".weight_scale"
                    yield out, _read(name)

            return _orig_lw(self, _gen())

        _lw._mtp_fp8 = True
        _mtp.DeepSeekMTP.load_weights = _lw

    logger.warning("[hpu-moe-ext] MTP FP8 GRAFT armed: spec layer loads fp8 from %s "
                   "(vllm fp8 method + weight substitution; NO bf16 served)", d)
