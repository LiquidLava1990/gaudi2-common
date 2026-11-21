#!/usr/bin/env python3
"""Audit AutoRoundHPUConfig dispatch decisions vs the real checkpoint storage.

For every quantized-or-not module in layer 0 (+ vision + lm_head + embed), ask the
plugin's get_quant_method what method it picks, and compare to what the checkpoint
ACTUALLY stores (.qweight present => must be a quant method; only .weight => must
be unquantized). Any mismatch is a load-time-or-garbage bug.
"""
import json, struct, os, sys, types, re
sys.path.insert(0, "/ext")

# Build the real quant config dict
MD = os.environ.get("MODEL_DIR", "/m")
cfg = json.load(open(f"{MD}/config.json"))
qc = cfg["quantization_config"]

# What the checkpoint stores, per module prefix (layer 0 + globals)
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]

def stored_kind(prefix):
    """Return 'quant' if <prefix>.qweight exists, 'fp' if <prefix>.weight exists, else None."""
    # search both index and file headers
    has_q = (prefix + ".qweight") in idx
    has_w = (prefix + ".weight") in idx
    return ("quant" if has_q else ("fp" if has_w else None)), has_q, has_w

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod
import hpu_moe_ext.auto_round_hpu as arh

conf = arh.AutoRoundHPUConfig.from_config(qc)
print("quant_blocks:", conf._quant_blocks)
print("weight_bits:", conf.weight_bits, "exact EC keys:", len(conf._exact), "regex:", len(conf._regex))

# Fake layer objects
class FakeLinear(LinearBase):
    def __init__(self): pass
class FakeMoE:
    pass

def method_name(layer, prefix):
    try:
        m = conf.get_quant_method(layer, prefix)
    except Exception as e:
        return f"EXC:{type(e).__name__}:{e}"
    return type(m).__name__ if m is not None else "None"

# Linear prefixes from layer 0 + vision + globals
lin_prefixes = [
 "model.language_model.layers.0.linear_attn.in_proj_qkvz",  # vLLM merged name
 "model.language_model.layers.0.linear_attn.in_proj_ba",
 "model.language_model.layers.0.linear_attn.in_proj_qkv",   # ckpt name
 "model.language_model.layers.0.linear_attn.in_proj_z",
 "model.language_model.layers.0.linear_attn.in_proj_b",
 "model.language_model.layers.0.linear_attn.in_proj_a",
 "model.language_model.layers.0.linear_attn.out_proj",
 "model.language_model.layers.0.linear_attn.conv1d",
 "model.language_model.layers.0.mlp.gate",
 "model.language_model.layers.0.mlp.shared_expert.gate_proj",
 "model.language_model.layers.0.mlp.shared_expert.up_proj",
 "model.language_model.layers.0.mlp.shared_expert.down_proj",
 "model.language_model.layers.0.mlp.shared_expert_gate",
 "model.visual.blocks.0.attn.qkv",
 "model.visual.blocks.0.mlp.gate_proj",
 "lm_head",
 "model.language_model.embed_tokens",
]
print("\n--- LINEAR dispatch (l=LinearBase) ---")
fl = object.__new__(FakeLinear)
for p in lin_prefixes:
    kind, hq, hw = stored_kind(p)
    name = method_name(fl, p)
    quantish = name in ("GPTQHPULinearMethod","GPTQDequantToBF16LinearMethod")
    flag = ""
    if kind == "quant" and not quantish: flag = "  <<< MISMATCH: stored quant but method unquant"
    if kind == "fp" and quantish: flag = "  <<< MISMATCH: stored fp but method quant"
    print(f"  bits={conf._bits_for_prefix(p):>2} scope={conf._in_quant_scope(p)} stored={kind} -> {name}{flag}  [{p}]")

print("\n--- MoE experts dispatch ---")
me = FakeMoE(); me.moe_config=None
# patch _is_experts to treat FakeMoE as experts
orig = arh._is_experts
arh._is_experts = lambda layer: isinstance(layer, FakeMoE) or orig(layer)
try:
    for p in ["model.language_model.layers.0.mlp.experts"]:
        try:
            m = conf.get_quant_method(me, p)
            print(f"  {p} -> {type(m).__name__} (bits={conf._bits_for_prefix(p)})")
        except Exception as e:
            print(f"  {p} -> EXC {type(e).__name__}: {e}")
finally:
    arh._is_experts = orig
print("\nDONE")
