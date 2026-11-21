"""
Audit get_quant_method dispatch for GLM-5.1 key layers.
"""
import sys, os, json
sys.path.insert(0, '/opt/hpu-moe-ext')
os.environ['HPU_MOE_EXT_CKPT'] = '/models/GLM-5.1-int4-mixed-AutoRound'

from hpu_moe_ext.auto_round_hpu import AutoRoundHPUConfig
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

with open('/models/GLM-5.1-int4-mixed-AutoRound/config.json') as f:
    full_cfg = json.load(f)
qcfg = full_cfg['quantization_config']
cfg = AutoRoundHPUConfig.from_config(qcfg)

# Create a minimal LinearBase subclass instance
class FakeLinear(LinearBase):
    def __init__(self):
        pass

fake_layer = FakeLinear()

method_cases = [
    ('language_model.model.layers.0.self_attn.q_a_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.0.self_attn.fused_qkv_a_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.0.self_attn.o_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.0.self_attn.kv_b_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.0.mlp.gate_proj', 'UnquantizedLinearMethod', None),
    ('language_model.model.layers.3.mlp.gate', 'UnquantizedLinearMethod', None),
    ('language_model.model.layers.3.mlp.shared_experts.gate_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.3.mlp.shared_experts.gate_up_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.3.mlp.shared_experts.down_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.0.self_attn.indexer.wq_b', 'GPTQDequantToBF16LinearMethod', 8),
    # Layer 77 (last layer) same as 0
    ('language_model.model.layers.77.self_attn.q_a_proj', 'GPTQDequantToBF16LinearMethod', 8),
    ('language_model.model.layers.77.self_attn.fused_qkv_a_proj', 'GPTQDequantToBF16LinearMethod', 8),
]

print("=== get_quant_method dispatch ===")
all_pass = True
for prefix, exp_cls, exp_bits in method_cases:
    method = cfg.get_quant_method(fake_layer, prefix)
    cls_name = type(method).__name__
    bits_attr = getattr(getattr(method, 'quant_config', None), 'weight_bits', None)

    cls_ok = cls_name == exp_cls
    bits_ok = (exp_bits is None and bits_attr is None) or (bits_attr == exp_bits)
    ok = cls_ok and bits_ok
    if not ok:
        all_pass = False
    status = 'OK' if ok else 'FAIL'
    print(f"  [{status}] {prefix}")
    if not ok:
        print(f"       expected: cls={exp_cls} bits={exp_bits}")
        print(f"       actual:   cls={cls_name} bits={bits_attr}")

print()

# Extra check: for GPTQDequantToBF16LinearMethod bits=4, would that be dispatched anywhere?
# The mislabeled bits>=16 case re-assigns to self.weight_bits (4)
# but for GLM there shouldn't be any non-MoE 4-bit linear with a qweight
print("=== Checking for accidental 4-bit linear dispatch (non-MoE) ===")
# Sample some later layers
suspicious = []
for layer_n in [0, 1, 2, 3, 50, 77]:
    for proj in ['q_a_proj', 'kv_a_proj_with_mqa', 'q_b_proj', 'kv_b_proj', 'o_proj']:
        prefix = f'language_model.model.layers.{layer_n}.self_attn.{proj}'
        bits = cfg._bits_for_prefix(prefix)
        is_q = cfg._checkpoint_is_quantized(prefix)
        if is_q and bits == 4:
            suspicious.append((prefix, bits))

if suspicious:
    print(f"  WARNING: These quantized linears got bits=4 (unexpected for GLM attn):")
    for p, b in suspicious:
        print(f"    {p} -> bits={b}")
else:
    print("  No accidental 4-bit linear dispatch found for attn projections")

# Check MLP shared experts
for layer_n in [3, 77]:
    for proj in ['gate_proj', 'up_proj', 'down_proj', 'gate_up_proj']:
        prefix = f'language_model.model.layers.{layer_n}.mlp.shared_experts.{proj}'
        bits = cfg._bits_for_prefix(prefix)
        is_q = cfg._checkpoint_is_quantized(prefix)
        if is_q and bits not in (8, 16):
            suspicious.append((prefix, bits))

print()
if all_pass:
    print("ALL get_quant_method checks PASSED")
else:
    print("SOME CHECKS FAILED")
