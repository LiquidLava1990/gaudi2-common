"""Make the DeepSeek-MTP ``eh_proj`` quantization-aware for AutoRound checkpoints.

WHY
---
vLLM core's ``DeepSeekMultiTokenPredictorLayer.__init__`` (deepseek_mtp.py) hardcodes
the MTP embed+hidden mixing projection as a *plain* ``nn.Linear``::

    self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

That is correct for the original DeepSeek-V3 MTP head, whose ``eh_proj`` ships in bf16.
But GLM-5.1's AutoRound int4-mixed export quantized layer 78's ``eh_proj`` to GPTQ-packed
int4 (on disk: ``model.layers.78.eh_proj.{qweight[1536,6144],qzeros[96,768],scales[96,6144]}``,
group_size=128, NOT in the fp16 extra_config exceptions). A plain ``nn.Linear`` only owns a
``.weight`` param, so ``DeepSeekMTP.load_weights`` KeyErrors at::

    KeyError: 'model.layers.78.eh_proj.qweight'

(deepseek_mtp.py load_weights: ``param = params_dict[name]`` with name ``...eh_proj.qweight``).

FIX
---
Wrap ``DeepSeekMultiTokenPredictorLayer.__init__`` so that, AFTER the stock init runs, if
the checkpoint actually stores this layer's ``eh_proj`` quantized (decided by the active
quant_config's ``get_quant_method`` on a probe ``ReplicatedLinear`` — i.e. AutoRound's
on-disk ``.qweight`` ground truth), we REPLACE ``self.eh_proj`` with a quant-aware
``ReplicatedLinear(hidden*2, hidden, bias=False, quant_config=quant_config, prefix=
"<layer>.eh_proj", return_bias=False)``. That builds ``.qweight/.qzeros/.scales`` params and
routes through the SAME AutoRound GPTQDequantToBF16 path the main model's int4 linears use,
so the stock ``load_weights`` (which does ``params_dict[name].weight_loader(...)`` for the
``eh_proj.*`` packed shards) finds the params and dequants to bf16 at load.

``return_bias=False`` keeps the call site working: the MTP forward calls ``self.eh_proj(x)``
and uses the tensor directly (a default ``ReplicatedLinear`` returns ``(out, bias)``).

SCOPE / SAFETY
--------------
* No-op unless the model is actually a DeepSeek-MTP draft (the class exists) AND this layer's
  eh_proj is stored quantized. bf16 MTP checkpoints (DeepSeek-V3) are untouched.
* The prefix passed to the layer's __init__ is ``model.layers.<N>`` (the MTP block prefix), so
  ``<prefix>.eh_proj`` == the on-disk quantized prefix — AutoRound's quant-method lookup matches.
* Idempotent: guarded by a sentinel attribute on the patched __init__.
"""

from __future__ import annotations

import logging

_log = logging.getLogger("vllm")


def install() -> None:
    try:
        import vllm.model_executor.models.deepseek_mtp as mtp_mod
    except Exception as e:  # noqa: BLE001
        # MTP module absent in this vllm build -> nothing to patch (non-fatal).
        _log.info("hpu-moe-ext: deepseek_mtp not importable (%s); eh_proj patch skipped", e)
        return

    cls = getattr(mtp_mod, "DeepSeekMultiTokenPredictorLayer", None)
    if cls is None:
        _log.info("hpu-moe-ext: DeepSeekMultiTokenPredictorLayer absent; eh_proj patch skipped")
        return

    orig_init = cls.__init__
    if getattr(orig_init, "_hpu_eh_proj_patched", False):
        return  # idempotent

    from vllm.config import VllmConfig  # noqa: F401  (type only, keep import local)

    def __init__(self, vllm_config, prefix: str) -> None:  # noqa: ANN001
        orig_init(self, vllm_config, prefix)

        quant_config = getattr(vllm_config, "quant_config", None)
        if quant_config is None:
            return  # unquantized model -> stock nn.Linear is correct

        config = self.config
        eh_prefix = f"{prefix}.eh_proj"

        # Decide whether this eh_proj is stored quantized using the SAME ground truth
        # the rest of the load uses: ask the active quant_config for the method it would
        # assign to an eh_proj LinearBase at this prefix. AutoRoundHPUConfig keys on the
        # on-disk .qweight set, so a quantized eh_proj -> a real (dequant) method; a bf16
        # eh_proj (or a non-AutoRound config) -> UnquantizedLinearMethod, and we leave the
        # stock nn.Linear in place.
        try:
            from vllm.model_executor.layers.linear import (
                ReplicatedLinear,
                UnquantizedLinearMethod,
            )

            probe = ReplicatedLinear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=eh_prefix,
                return_bias=False,
            )
            method = getattr(probe, "quant_method", None)
            is_quantized = method is not None and not isinstance(
                method, UnquantizedLinearMethod
            )
            if is_quantized:
                # The probe IS a fully-formed quantized ReplicatedLinear; adopt it.
                self.eh_proj = probe
                _log.info(
                    "hpu-moe-ext: MTP eh_proj at %s built as quantized ReplicatedLinear "
                    "(%s) — AutoRound int4 checkpoint", eh_prefix, type(method).__name__
                )
            # else: leave the stock bf16 nn.Linear (probe is discarded).
        except Exception as e:  # noqa: BLE001
            # Never break model construction; fall back to the stock nn.Linear.
            _log.warning(
                "hpu-moe-ext: MTP eh_proj quant probe failed at %s (%s); "
                "keeping stock nn.Linear", eh_prefix, e
            )

    __init__._hpu_eh_proj_patched = True  # type: ignore[attr-defined]
    cls.__init__ = __init__
    _log.info("hpu-moe-ext: patched DeepSeekMultiTokenPredictorLayer.__init__ "
              "(quant-aware eh_proj for AutoRound MTP)")
