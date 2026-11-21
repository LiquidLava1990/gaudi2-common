"""Dump GLM-5.1 decoder-layer in/out tensors (HPU) for CPU-reference comparison.

FAST batched mode: dump a SPREAD of layers in ONE on-card run (HPU_DUMP_LAYERS="0,3,4,10,40,77"),
each to /opt/hpu-dsa-ext/layer<N>_dump.pt. A CPU script then checks EACH layer independently
(its own dumped input -> reference forward -> compare output), localizing ALL remaining per-layer
bugs in a single run — the reference comparison iterates with no box restarts.

Per layer we save: positions, hidden_in, residual_in, hidden_out, residual_out, and the live
o_proj weight (attention check). Gated HPU_LAYER_DUMP=1.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")
# Comma list of layer indices to dump (default just layer 0). HPU_DUMP_LAYER kept for back-compat.
_LAYERS = os.environ.get("HPU_DUMP_LAYERS") or os.environ.get("HPU_DUMP_LAYER", "0")
_TARGETS = set(int(s) for s in _LAYERS.replace(" ", "").split(",") if s != "")
_DONE = set()


def install() -> None:
    if os.environ.get("HPU_LAYER_DUMP", "0") != "1":
        return
    import torch
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    except Exception as ex:  # noqa: BLE001
        log.warning("[layer-dump] cannot import DeepseekV2DecoderLayer: %s", ex)
        return
    if getattr(DeepseekV2DecoderLayer.forward, "_layer_dump", False):
        return
    _orig = DeepseekV2DecoderLayer.forward
    _runner = {}

    def _is_real():
        mr = _runner.get("mr")
        wm = getattr(mr, "warmup_mode", None) if mr is not None else None
        return (wm is False) if wm is not None else True

    @torch.compiler.disable
    def _save(layer, lidx, positions, hs, res, out):
        try:
            o0 = out[0] if isinstance(out, (tuple, list)) else out
            o1 = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
            d = {
                "layer": lidx,
                "positions": positions.detach().to("cpu", torch.float32) if hasattr(positions, "detach") else None,
                "hidden_in": hs.detach().to("cpu", torch.float32) if hs is not None else None,
                "residual_in": res.detach().to("cpu", torch.float32) if res is not None else None,
                "hidden_out": o0.detach().to("cpu", torch.float32) if o0 is not None else None,
                "residual_out": o1.detach().to("cpu", torch.float32) if o1 is not None else None,
            }
            try:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    for nm, mod in attn.named_modules():
                        if nm.endswith("o_proj") and hasattr(mod, "weight"):
                            d["o_proj_weight"] = mod.weight.detach().to("cpu", torch.float32)
                            d["o_proj_type"] = type(mod).__name__
                            break
            except Exception:  # noqa: BLE001
                pass
            torch.save(d, "/opt/hpu-dsa-ext/layer%d_dump.pt" % lidx)
            log.warning("[layer-dump] saved layer-%d: hidden_in %s residual_in %s hidden_out %s",
                        lidx, list(d["hidden_in"].shape) if d["hidden_in"] is not None else None,
                        list(d["residual_in"].shape) if d["residual_in"] is not None else None,
                        list(d["hidden_out"].shape) if d["hidden_out"] is not None else None)
        except Exception as ex:  # noqa: BLE001
            log.warning("[layer-dump] save error layer %d: %s", lidx, ex)

    def forward(self, *args, **kwargs):  # noqa: ANN001
        out = _orig(self, *args, **kwargs)
        lidx = getattr(self, "layer_idx", -1)
        if lidx in _TARGETS and lidx not in _DONE and _is_real():
            _DONE.add(lidx)
            positions = args[0] if len(args) > 0 else kwargs.get("positions")
            hs = args[1] if len(args) > 1 else kwargs.get("hidden_states")
            res = args[2] if len(args) > 2 else kwargs.get("residual")
            _save(self, lidx, positions, hs, res, out)
        return out

    forward._layer_dump = True
    DeepseekV2DecoderLayer.forward = forward

    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
        if not getattr(HPUModelRunner.__init__, "_layer_dump_hooked", False):
            _oi = HPUModelRunner.__init__
            def __init__(self, *a, **k):
                _oi(self, *a, **k); _runner["mr"] = self
            __init__._layer_dump_hooked = True
            HPUModelRunner.__init__ = __init__
    except Exception:  # noqa: BLE001
        pass

    log.warning("[layer-dump] installed (HPU_LAYER_DUMP=1, target layers=%s)", sorted(_TARGETS))
