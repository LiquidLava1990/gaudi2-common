"""Graft the ORIGINAL bf16 MTP layer (78) onto the int4 serving checkpoint.

WHY: the AutoRound int4-mixed export's layer 78 is DEAD (fp64 oracle + 7-variant
wiring sweep: constant junk under every wiring; bf16 weights under the SAME
wiring give 62% draft-target agreement — tests/glm_cpu_mtp_bf16.py). The MTP
lever needs real weights; zai-org/GLM-5.1 ships layer 78 bf16 in 4 shards.

WHAT (gate: HPU_MTP_BF16_GRAFT=<dir with the orig shards + index>, default off):
  1. Quant config bypass: AutoRound get_quant_method returns None (unquantized)
     for every module whose prefix lies in the spec layer -> vllm builds PLAIN
     bf16 Linear/FusedMoE modules for the whole MTP layer.
  2. Weight substitution: DeepSeekMTP.load_weights' iterable is wrapped — the
     int4 checkpoint's layer-78 entries (qweight/scales/...) are DROPPED and the
     bf16 tensors from the graft dir are INJECTED under the same names. The
     existing spec-layer name rewriting handles the rest.
Draft-only memory cost ~2.5GB/card at TP=8. mtp_share_head_patch (embed/head
tie) still applies. LOUD: raises if the graft dir/index is missing or empty.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("vllm")
_ENV = "HPU_MTP_BF16_GRAFT"


def _graft_dir() -> str | None:
    d = os.environ.get(_ENV, "").strip()
    return d or None


def _spec_layer_prefixes() -> list[str]:
    ckpt = os.environ.get("HPU_MOE_EXT_CKPT", "").strip()
    cfg_file = os.path.join(ckpt, "config.json")
    if not ckpt or not os.path.exists(cfg_file):
        raise RuntimeError(
            f"[hpu-moe-ext] {_ENV} set but HPU_MOE_EXT_CKPT/config.json not "
            "readable - cannot derive the spec-layer prefixes")
    with open(cfg_file) as f:
        c = json.load(f)
    n = int(c.get("num_hidden_layers", 0)); k = int(c.get("num_nextn_predict_layers", 0))
    if not (n and k):
        raise RuntimeError("[hpu-moe-ext] checkpoint config has no nextn layers")
    return [f"model.layers.{n + i}." for i in range(k)]


def install() -> None:
    d = _graft_dir()
    if not d:
        return
    idx_file = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx_file):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}={d} but {idx_file} is missing")

    # (1) quant bypass for spec-layer prefixes.
    from .auto_round_hpu import AutoRoundHPUConfig
    if not getattr(AutoRoundHPUConfig.get_quant_method, "_mtp_graft", False):
        _orig_gqm = AutoRoundHPUConfig.get_quant_method

        _prefixes = _spec_layer_prefixes()

        def _gqm(self, layer, prefix):
            if any(prefix.startswith(p) for p in _prefixes):
                # Linear layers REQUIRE an explicit unquantized method ("All
                # linear layers should support quant method"); FusedMoE and
                # embeddings treat None as unquantized.
                from vllm.model_executor.layers.linear import (
                    LinearBase, UnquantizedLinearMethod)
                if isinstance(layer, LinearBase):
                    return UnquantizedLinearMethod()
                return None
            return _orig_gqm(self, layer, prefix)

        _gqm._mtp_graft = True
        AutoRoundHPUConfig.get_quant_method = _gqm

    # (2) weight substitution in the MTP loader.
    from vllm.model_executor.models import deepseek_mtp as _mtp
    if not getattr(_mtp.DeepSeekMTP.load_weights, "_mtp_graft", False):
        _orig_lw = _mtp.DeepSeekMTP.load_weights

        def _lw(self, weights):
            prefixes = _spec_layer_prefixes()
            with open(idx_file) as f:
                wm = json.load(f)["weight_map"]
            graft_names = [n for n in wm
                           if any(n.startswith(p) for p in prefixes)]
            if not graft_names:
                raise RuntimeError(
                    f"[hpu-moe-ext] {_ENV}: graft index has NO spec-layer "
                    f"weights for prefixes {prefixes}")

            from safetensors import safe_open
            _handles: dict[str, object] = {}

            def _read(name):
                fn = wm[name]
                if fn not in _handles:
                    _handles[fn] = safe_open(os.path.join(d, fn),
                                             framework="pt", device="cpu")
                return _handles[fn].get_tensor(name)

            def _gen():
                dropped = 0
                for name, w in weights:
                    if any(name.startswith(p) for p in prefixes):
                        dropped += 1
                        continue            # int4 entries for the grafted layer
                    yield name, w
                logger.warning(
                    "[hpu-moe-ext] MTP BF16 GRAFT: dropped %d int4 layer-78 "
                    "entries; injecting %d bf16 tensors from %s",
                    dropped, len(graft_names), d)
                for name in graft_names:
                    yield name, _read(name)

            return _orig_lw(self, _gen())

        _lw._mtp_graft = True
        _mtp.DeepSeekMTP.load_weights = _lw

    logger.warning("[hpu-moe-ext] MTP BF16 GRAFT armed: spec layer loads bf16 "
                   "from %s (quant bypass + weight substitution)", d)


