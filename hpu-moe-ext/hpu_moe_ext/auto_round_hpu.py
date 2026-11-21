"""AutoRound (INT4 packed as auto_gptq) quantization config for HPU.

Registers a first-class "auto-round"/"auto_round" QuantizationConfig so vLLM's
named-method resolution (vllm/.../quantization/__init__.py update wins over
the static INCConfig mapping) lands on us. Subclasses GPTQHPUConfig to inherit
the PROVEN 4-bit linear method (GPTQHPULinearMethod) and adds per-layer
extra_config dispatch:
    bits>=16 -> bf16 (Unquantized*)   bits==4 -> proven path   bits==8 -> dequant-bf16
Routed experts (4-bit, never listed in extra_config) -> proven MoeWNA16HPUMethod.
Out-of-quant-scope linears (vision tower, lm_head, embed) -> bf16 via
block_name_to_quantize gate.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import torch

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm_gaudi.ops.hpu_gptq import GPTQHPUConfig, GPTQHPULinearMethod

from ._compat import (is_fused_moe_layer,
                      unquantized_fused_moe_method_cls)
from .moe_wna16_hpu import MoeWNA16HPUMethod, HPUWNA16MoEConfig


# --- extra_config -> per-prefix bit width -------------------------------------
# AutoRound keys are EITHER exact HF module names OR regexes. Verified on both
# real configs: regex keys are EXACTLY those containing '*' (GLM has 4:
# '.*layers\\.0\\.mlp.*' etc); every other key is an exact module name. The
# escaped dots ('\\.') make '.*layers\\.0\\.mlp.*' match layer 0 but NOT 10.
# vLLM and the AutoRound checkpoint disagree on wrapper-segment ORDER/PRESENCE:
# checkpoint keys are HF-style "model.language_model.layers.N..."; vLLM's runtime
# prefix is "language_model.model.layers.N..." (top prefix="", then
# maybe_prefix("","language_model")="language_model", then
# maybe_prefix("language_model","model")="language_model.model"; see vllm
# qwen3_5.py:469-470,801). The decoder block path is identical only from
# ".layers." onward. Normalise by dropping the wrapper tokens both sides add so
# HF keys match runtime prefixes. (240 AutoRound keys verified normalized-unique.)
_WRAP_SEGMENTS = {"model", "language_model"}


def _norm_prefix(p: str) -> str:
    """Strip wrapper segments vLLM/HF disagree on, so a checkpoint module name
    and the vLLM runtime prefix become comparable."""
    return ".".join(s for s in p.split(".") if s not in _WRAP_SEGMENTS)


def _compile_extra_config(extra_config: dict) -> tuple[dict, list]:
    exact: dict[str, int] = {}
    regex: list[tuple[re.Pattern, int]] = []
    for key, spec in (extra_config or {}).items():
        if not isinstance(spec, dict) or "bits" not in spec:
            continue
        bits = int(spec["bits"])
        if "*" in key:                       # AutoRound regex form
            regex.append((re.compile(key), bits))
        else:                                # exact module name
            exact[key] = bits
    return exact, regex


def _bits_for(prefix: str, exact: dict, regex: list, default_bits: int) -> int:
    """Effective weight-bits for a layer prefix under AutoRound extra_config.
    Exact match wins; then suffix-equal alias; then first regex hit; else default.
    Matching is on wrapper-normalised names because the vLLM runtime prefix
    ("language_model.model.layers...") and the checkpoint key
    ("model.language_model.layers...") differ only in the wrapper segments."""
    npfx = _norm_prefix(prefix)
    nexact = {_norm_prefix(k): b for k, b in exact.items()}
    if npfx in nexact:
        return nexact[npfx]
    for k, b in nexact.items():
        if npfx.endswith("." + k) or k.endswith("." + npfx):
            return b
    for pat, b in regex:
        # regex keys are authored against HF names; try both raw and normalised.
        if pat.search(prefix) or pat.search(npfx):
            return b
    return default_bits


def _is_experts(layer) -> bool:
    return is_fused_moe_layer(layer)


def _moe_hf(hf: dict) -> dict:
    """AutoRound (auto_round:auto_gptq) == symmetric int4 gptq packing. The MoE
    config builder (HPUWNA16MoEConfig.from_hf) only knows gptq/awq, so present
    auto-round as gptq for the routed-expert path."""
    h = dict(hf)
    if h.get("quant_method", "").lower() in ("auto-round", "auto_round"):
        h["quant_method"] = "gptq"
    return h


@register_quantization_config("auto-round")
@register_quantization_config("auto_round")
class AutoRoundHPUConfig(GPTQHPUConfig):
    """AutoRound GPTQ-packed W4 / (mixed W8/W16) on Gaudi."""

    def __init__(self, weight_bits, group_size, desc_act, sym, extra_config,
                 quant_blocks, full_config):
        super().__init__(weight_bits=weight_bits, group_size=group_size,
                         desc_act=bool(desc_act), lm_head_quantized=False)
        self.sym = sym
        self._hf_quant_cfg = full_config
        self._exact, self._regex = _compile_extra_config(extra_config)
        # Architecture-agnostic ground truth: the set of module prefixes that are
        # ACTUALLY stored quantized (own a `.qweight`), read from the checkpoint's
        # safetensors index. AutoRound's extra_config can disagree with the stored
        # bytes in BOTH directions: the Qwen-397B export mislabels int4
        # shared_expert.*_proj as bits:16, while GLM-mixed genuinely keeps some
        # in-scope linears at fp16. Trusting the stored format (not the label or a
        # leaf-name heuristic) is correct for both. Needs HPU_MOE_EXT_CKPT set to
        # the model dir (serve.sh does this); falls back to a leaf heuristic if not.
        self._quantized_prefixes = self._load_quantized_prefixes()
        # block_name_to_quantize: str or list[str]; layers OUTSIDE these blocks
        # (vision tower, embed, lm_head) are never quantized -> bf16.
        if quant_blocks is None:
            self._quant_blocks: list[str] = []
        elif isinstance(quant_blocks, str):
            self._quant_blocks = [quant_blocks]
        else:
            self._quant_blocks = list(quant_blocks)

    @classmethod
    def get_name(cls) -> str:
        return "auto-round"

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []          # config lives in config.json::quantization_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoRoundHPUConfig":
        return cls(
            weight_bits=int(config.get("bits", 4)),
            group_size=int(config.get("group_size", 128)),
            # NOTE: desc_act is present-but-null in the real configs -> coerce.
            desc_act=bool(config.get("desc_act") or False),
            sym=bool(config.get("sym", True)),
            extra_config=config.get("extra_config") or {},
            quant_blocks=config.get("block_name_to_quantize"),
            full_config=dict(config),
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant,
                                     hf_config=None) -> Optional[str]:
        method = (hf_quant_cfg or {}).get("quant_method", "").lower()
        if user_quant in ("auto-round", "auto_round"):
            return "auto-round"
        if user_quant in (None, "auto") and method in ("auto-round", "auto_round"):
            return "auto-round"
        return None

    # --- single dispatch authority -------------------------------------------
    def _in_quant_scope(self, prefix: str) -> bool:
        if not self._quant_blocks:
            return True
        npfx = _norm_prefix(prefix)
        for b in self._quant_blocks:
            nb = _norm_prefix(b)
            if npfx.startswith(nb) or ("." + nb + ".") in ("." + npfx + "."):
                return True
        return False

    # Leaf names of LinearBase modules that AutoRound genuinely leaves in
    # bf16 inside the quant scope. Verified against the real 397B checkpoint
    # (tests/ar_dispatch_audit.py): the ONLY in-scope linears stored as plain
    # ``.weight`` are the MoE router gate and the shared-expert gate; every
    # other in-scope linear (attention proj, linear_attn in/out proj, AND
    # ``shared_expert.{gate,up,down}_proj``) is stored as int4 ``.qweight``.
    _FP_LINEAR_LEAVES = ("gate", "shared_expert_gate")

    def _load_quantized_prefixes(self):
        """Set of wrapper-normalised module prefixes that own a `.qweight` in the
        checkpoint (= actually stored quantized).

        GROUND TRUTH = the actual safetensors SHARD HEADERS, NOT the index's
        weight_map. The Qwen3.5-397B AutoRound export ships a BROKEN
        model.safetensors.index.json: its weight_map lists 180 phantom
        ``mlp.shared_expert.{gate,up,down}_proj.{qweight,scales,qzeros}`` entries
        (all 60 layers) that DO NOT EXIST in any shard — the shards actually store
        those modules as plain bf16 ``.weight`` (the extra_config bits:16 label was
        correct after all). Trusting the index (.qweight keys) wrongly routes every
        shared_expert to GPTQDequantToBF16LinearMethod; vLLM's loader then finds no
        real qweight/scales/qzeros to fill, leaving create_weights' torch.empty()
        params UNINITIALISED -> dequant of garbage (empty bf16 scales hold NaN bit
        patterns) -> NaN shared_expert output added to the residual at all 60 MoE
        layers -> logit collapse to a single token, with ZERO load warnings (the
        phantom keys are never iterated, and the real .weight tensors are simply
        the unused ones). Reading the shard headers directly is immune to a lying
        index. Returns None if unavailable (-> leaf-heuristic fallback)."""
        import os
        ckpt = os.environ.get("HPU_MOE_EXT_CKPT")
        if not ckpt:
            return None
        import glob
        import json as _json
        import struct
        shards = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
        if not shards:
            return None
        prefixes: set[str] = set()
        try:
            for path in shards:
                with open(path, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    hdr = _json.loads(fh.read(n))
                for k in hdr:
                    if k != "__metadata__" and k.endswith(".qweight"):
                        prefixes.add(_norm_prefix(k[:-len(".qweight")]))
        except Exception:  # noqa: BLE001
            # Fall back to the (possibly-wrong) index rather than crashing.
            idxp = os.path.join(ckpt, "model.safetensors.index.json")
            try:
                with open(idxp) as f:
                    wm = _json.load(f).get("weight_map", {})
            except Exception:  # noqa: BLE001
                return None
            return {_norm_prefix(k[:-len(".qweight")])
                    for k in wm if k.endswith(".qweight")}
        return prefixes

    # vLLM's DeepseekV2Attention (deepseek_v2.py:928) fuses q_a_proj and
    # kv_a_proj_with_mqa into a single MergedColumnParallelLinear named
    # "fused_qkv_a_proj" when q_lora_rank is not None (GLM-5.1: q_lora_rank=2048).
    # The checkpoint stores these as SEPARATE quantized tensors with their original
    # names (q_a_proj.qweight, kv_a_proj_with_mqa.qweight), so
    # "fused_qkv_a_proj" never appears in _quantized_prefixes. Without this mapping,
    # _checkpoint_is_quantized returns False for the fused module -> it gets
    # UnquantizedLinearMethod -> no .qweight param built -> load_weights' stacked_
    # params_mapping skips the fusion fallback and tries params_dict[
    # "...kv_a_proj_with_mqa.qweight"] -> KeyError.
    # Map each vLLM-fused leaf name to the checkpoint shard names that represent it.
    _FUSED_LEAF_TO_SHARDS: dict[str, tuple[str, ...]] = {
        "fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
        # GLM shared-expert / dense MLP (DeepseekV2MLP) fuses gate+up into
        # gate_up_proj; checkpoint stores gate_proj/up_proj separately. (A model
        # whose gate_up_proj is fused-IN-checkpoint is handled by direct-check-first.)
        "gate_up_proj": ("gate_proj", "up_proj"),
    }

    def _checkpoint_is_quantized(self, prefix: str) -> bool:
        """True iff this LinearBase is ACTUALLY stored quantized (has `.qweight`),
        regardless of what extra_config says. AutoRound's metadata disagrees with
        the stored bytes both ways: Qwen-397B mislabels int4 shared_expert.*_proj
        as bits:16 (trusting it drops the int4 weights -> garbage shared-expert
        output added to the residual at every MoE layer -> word salad), while
        GLM-mixed genuinely keeps some in-scope linears fp16. Keying on the real
        stored format is correct for both. Falls back to a leaf-name heuristic
        only if the index is unavailable.

        Special case: vLLM fuses q_a_proj + kv_a_proj_with_mqa into a single
        fused_qkv_a_proj module. The checkpoint has NO fused_qkv_a_proj.qweight;
        it has the individual shard qweights. Treat the fused module as quantized
        iff at least one of its component shards is quantized in the checkpoint."""
        leaf = prefix.rsplit(".", 1)[-1]
        if self._quantized_prefixes is not None:
            # Direct: this exact module is stored quantized (covers a gate_up_proj
            # that is fused IN the checkpoint, e.g. some Qwen builds) -> True.
            if _norm_prefix(prefix) in self._quantized_prefixes:
                return True
            # Fused-in-MODEL but stored as separate shards (GLM fused_qkv_a_proj,
            # gate_up_proj): quantized iff any component shard is quantized.
            shards = self._FUSED_LEAF_TO_SHARDS.get(leaf)
            if shards is not None:
                parent = prefix[: -(len(leaf))]
                return any(self._checkpoint_is_quantized(parent + s) for s in shards)
            return False
        if leaf in self._FP_LINEAR_LEAVES:
            return False
        return self._in_quant_scope(prefix)

    def _bits_for_prefix(self, prefix: str) -> int:
        # router gate is never quantized -> force bf16 (same guard as the
        # existing __init__._skip_linear last=='gate' rule).
        leaf = prefix.rsplit(".", 1)[-1]
        if leaf == "gate":
            return 16
        # If the fused module is fused IN the checkpoint (directly quantized), use
        # its own bits. Only recurse to shards when it is NOT directly present
        # (GLM fuses in-model; checkpoint has separate gate_proj/up_proj shards).
        direct = (self._quantized_prefixes is not None
                  and _norm_prefix(prefix) in self._quantized_prefixes)
        if not direct:
            shards = self._FUSED_LEAF_TO_SHARDS.get(leaf)
            if shards is not None:
                # resolve bits from component shards; max (least-aggressive) keeps
                # the GPTQ pack-factor consistent (GLM-5.1 shards are both bits:8).
                parent = prefix[: -(len(leaf))]
                return max(
                    _bits_for(parent + s, self._exact, self._regex, self.weight_bits)
                    for s in shards
                )
        return _bits_for(prefix, self._exact, self._regex, self.weight_bits)

    def get_quant_method(self, layer, prefix):
        # 1) MoE routed experts -> 4-bit proven path (or bf16 if skipped).
        if _is_experts(layer):
            bits = self._bits_for_prefix(prefix)
            if bits >= 16:
                return unquantized_fused_moe_method_cls()(layer.moe_config)
            return MoeWNA16HPUMethod(
                HPUWNA16MoEConfig.from_hf(_moe_hf(self._hf_quant_cfg)),
                layer.moe_config)
        # 2) Linear layers.
        if isinstance(layer, LinearBase):
            if not self._in_quant_scope(prefix):     # vision/lm_head/embed
                return UnquantizedLinearMethod()
            # STORED FORMAT IS GROUND TRUTH (extra_config disagrees with the bytes
            # BOTH ways): Qwen-397B mislabels int4 shared_expert.*_proj as bits:16,
            # AND some default-bits:4 layers are genuinely fp16 (the first 3 layers'
            # attention are stored as plain .weight). So a linear with NO .qweight
            # is bf16; one WITH a .qweight is dequant-to-bf16 at its real bit width.
            if not self._checkpoint_is_quantized(prefix):
                return UnquantizedLinearMethod()
            bits = self._bits_for_prefix(prefix)
            if bits >= 16:
                bits = self.weight_bits   # mislabeled int4 -> true 4-bit
            # int4/int8 LINEARS -> dequant to bf16 at load. The HPU int4-linear
            # convert_from_uint4 path compiles pathologically slowly at 397B scale
            # (~4h warmup; engine-death on lazy compile). Attention/shared_experts
            # are a small memory fraction, and int4 on Gaudi is a memory-fit lever
            # (weights dequant to bf16 at compute anyway), so bf16-at-load costs
            # little memory, compiles fast (bf16 matmul), and is numerically exact.
            # Only the bulky MoE experts stay int4 on the proven fast WNA16 op.
            from .moe_wna16_hpu import GPTQDequantToBF16LinearMethod
            return GPTQDequantToBF16LinearMethod(self.group_size, bits=bits)
        # 3) lm_head etc. -> base (bf16 unless lm_head_quantized).
        return super().get_quant_method(layer, prefix)
