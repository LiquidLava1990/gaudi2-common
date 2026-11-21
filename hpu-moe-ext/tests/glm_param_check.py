"""
Check what parameters are built for fused_qkv_a_proj when dispatched to
GPTQDequantToBF16LinearMethod(bits=8). The critical question is:
does mla_proj_patch look for .weight but create_weights builds .qweight?
"""
import sys, os, json
sys.path.insert(0, '/opt/hpu-moe-ext')
os.environ['HPU_MOE_EXT_CKPT'] = '/models/GLM-5.1-int4-mixed-AutoRound'

from hpu_moe_ext.auto_round_hpu import AutoRoundHPUConfig
from hpu_moe_ext.moe_wna16_hpu import GPTQDequantToBF16LinearMethod

with open('/models/GLM-5.1-int4-mixed-AutoRound/config.json') as f:
    qcfg = json.load(f)['quantization_config']
cfg = AutoRoundHPUConfig.from_config(qcfg)

print("=== fused_qkv_a_proj param analysis ===")

# What method does it get?
from vllm.model_executor.layers.linear import LinearBase
class FakeLinear(LinearBase):
    def __init__(self): pass
method = cfg.get_quant_method(FakeLinear(), 'language_model.model.layers.0.self_attn.fused_qkv_a_proj')
print(f"fused_qkv_a_proj method: {type(method).__name__} bits={getattr(getattr(method,'quant_config',None),'weight_bits',None)}")

# Simulate create_weights to see what params are registered
# GPTQDequantToBF16LinearMethod inherits from GPTQHPULinearMethod.create_weights
# which registers: qweight, g_idx, qzeros, scales
import torch
class LayerSim(torch.nn.Module):
    def __init__(self): super().__init__()

layer = LayerSim()

# The fused_qkv_a_proj has:
# input_size = 6144 (hidden_size)
# output_size = [q_lora_rank, kv_lora_rank + qk_rope_head_dim] = [2048, 576]
# total output = 2624
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter, GroupQuantScaleParameter,
    PackedColumnParameter, PackedvLLMParameter, RowvLLMParameter)

input_size = 6144
output_partition_sizes = [2048, 576]  # q_a and kv_a outputs
group_size = 128
bits = 8
pack_factor = 32 // bits  # = 4

try:
    method.create_weights(
        layer=layer,
        input_size_per_partition=input_size,
        output_partition_sizes=output_partition_sizes,
        input_size=input_size,
        output_size=sum(output_partition_sizes),
        params_dtype=torch.bfloat16,
        weight_loader=lambda p, t, *a: None,
    )
    print(f"\nParameters created by create_weights:")
    for name, param in layer.named_parameters():
        print(f"  {name}: shape={list(param.shape)}, dtype={param.dtype}, type={type(param).__name__}")
except Exception as e:
    print(f"create_weights error: {e}")
    import traceback; traceback.print_exc()

print()
print("=== Key question: does params_dict have '.weight' or '.qweight'? ===")
print("After create_weights: layer has 'qweight' (int32), not 'weight' (bf16)")
print("mla_proj_patch looks for: fused_qkv_a_proj.weight")
print("At load time (before process_weights_after_loading):")
print("  params_dict has: fused_qkv_a_proj.qweight, .qzeros, .scales, .g_idx")
print("  params_dict does NOT have: fused_qkv_a_proj.weight")
print()
print("So mla_proj_patch.fused_name check FAILS -> falls to plain_name")
print("plain_name = '...q_a_proj.weight' -> also not in params_dict -> KeyError!")
print()
print("BUT WAIT: if mla_proj_patch raises KeyError, the serve would crash, not produce garbage.")
print("Let's check: what ACTUALLY happens if the fused_qkv_a_proj.qweight exists?")
print()

# Alternative hypothesis: mla_proj_patch was designed for the case where
# fused_qkv_a_proj gets UnquantizedLinearMethod (not quantized).
# In that case: create_weights builds .weight (bf16 Parameter).
# The mla_proj_patch then finds .weight and loads the dequanted bf16 into it.
#
# But we showed _checkpoint_is_quantized returns True for fused_qkv_a_proj.
# So it gets GPTQDequantToBF16LinearMethod -> .qweight NOT .weight.
# -> mla_proj_patch KeyError.
#
# RESOLUTION: One of these must be true:
# A) The KeyError IS raised but swallowed somewhere (serve logs it but continues with
#    uninitialised qweight -> garbage dequant -> garbage output)
# B) _checkpoint_is_quantized actually returns False for fused_qkv_a_proj
#    (because of how the shard check works)
# C) Something else prevents the KeyError

# Let's carefully retrace _checkpoint_is_quantized for fused_qkv_a_proj
print("=== Retrace _checkpoint_is_quantized('...fused_qkv_a_proj') ===")
prefix = 'language_model.model.layers.0.self_attn.fused_qkv_a_proj'
leaf = 'fused_qkv_a_proj'
print(f"  leaf = {leaf!r}")
print(f"  _norm_prefix(prefix) = {cfg._norm_prefix(prefix) if hasattr(cfg,'_norm_prefix') else '(method not accessible directly)'}")

from hpu_moe_ext.auto_round_hpu import _norm_prefix
normed = _norm_prefix(prefix)
print(f"  normed = {normed!r}")
print(f"  normed in _quantized_prefixes: {normed in cfg._quantized_prefixes}")

# check _FUSED_LEAF_TO_SHARDS
shards = cfg._FUSED_LEAF_TO_SHARDS.get(leaf)
print(f"  _FUSED_LEAF_TO_SHARDS[{leaf!r}] = {shards}")
if shards:
    parent = prefix[:-(len(leaf))]
    for s in shards:
        sub = cfg._checkpoint_is_quantized(parent + s)
        print(f"    shard {parent+s} -> is_quantized={sub}")
    result = any(cfg._checkpoint_is_quantized(parent + s) for s in shards)
    print(f"  -> any shard quantized: {result} -> fused_qkv_a_proj is_quantized = {result}")

print()
print("So if fused_qkv_a_proj is considered quantized:")
print("  get_quant_method returns GPTQDequantToBF16LinearMethod(bits=8)")
print("  create_weights builds .qweight (not .weight)")
print("  mla_proj_patch cannot find .weight -> KeyError at load")
print()
print("FIX NEEDED: mla_proj_patch must look for .qweight (not .weight)")
print("and load the PACKED (int32) qweight into it, rather than dequanting first.")
print("OR: fused_qkv_a_proj should get UnquantizedLinearMethod and have .weight,")
print("which means _checkpoint_is_quantized should return False for the fused name")
print("so that mla_proj_patch can load bf16 into .weight, relying on the patch to dequant.")
