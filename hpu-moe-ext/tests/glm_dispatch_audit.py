"""
Audit the AutoRoundHPUConfig dispatch for GLM-5.1:
- Which layers get which quant method (bits, class)?
- Are the 8-bit linear layers dispatched to GPTQDequantToBF16LinearMethod(bits=8)?
- Does fused_qkv_a_proj get the right bits (8, not 4)?
- Does the gate (router) correctly get UnquantizedLinearMethod?
- Check gate_up_proj fusion dispatch.
- Check the _bits_for_prefix for all known GLM-specific problematic prefixes.

This does NOT load the model - just exercises the config dispatch logic.
"""
import sys, os, json, re
sys.path.insert(0, '/opt/hpu-moe-ext')
os.environ['HPU_MOE_EXT_CKPT'] = '/models/GLM-5.1-int4-mixed-AutoRound'

# Import after setting env
from hpu_moe_ext.auto_round_hpu import AutoRoundHPUConfig

# Load config
with open('/models/GLM-5.1-int4-mixed-AutoRound/config.json') as f:
    full_cfg = json.load(f)
qcfg = full_cfg['quantization_config']

cfg = AutoRoundHPUConfig.from_config(qcfg)
print(f"AutoRoundHPUConfig: bits={cfg.weight_bits}, group_size={cfg.group_size}, sym={cfg.sym}")
print(f"_quant_blocks: {cfg._quant_blocks}")
print(f"_quantized_prefixes loaded: {len(cfg._quantized_prefixes) if cfg._quantized_prefixes is not None else 'None'}")
if cfg._quantized_prefixes:
    sample = list(sorted(cfg._quantized_prefixes))[:10]
    print(f"  Sample quantized prefixes (normalized): {sample}")
print()

# Test _bits_for_prefix for all problematic cases
print("=== _bits_for_prefix results ===")
test_cases = [
    # (prefix, expected_bits, note)
    ('language_model.model.layers.0.self_attn.q_a_proj', 8, '8-bit attn proj'),
    ('language_model.model.layers.0.self_attn.kv_a_proj_with_mqa', 8, '8-bit attn proj'),
    ('language_model.model.layers.0.self_attn.q_b_proj', 8, '8-bit attn proj'),
    ('language_model.model.layers.0.self_attn.kv_b_proj', 8, '8-bit attn proj'),
    ('language_model.model.layers.0.self_attn.o_proj', 8, '8-bit attn proj'),
    ('language_model.model.layers.0.self_attn.fused_qkv_a_proj', 8, 'FUSED: should resolve to 8 via shards'),
    ('language_model.model.layers.0.mlp.gate_proj', 16, 'fp16 dense MLP layer 0'),
    ('language_model.model.layers.3.mlp.shared_experts.gate_proj', 8, '8-bit shared expert'),
    ('language_model.model.layers.3.mlp.shared_experts.down_proj', 8, '8-bit shared expert'),
    ('language_model.model.layers.3.mlp.shared_experts.gate_up_proj', 8, 'fused gate_up: should resolve to 8'),
    ('language_model.model.layers.3.mlp.gate', 4, 'router gate - but _bits_for_prefix forces 16 for leaf==gate'),
    ('language_model.model.layers.77.self_attn.q_a_proj', 8, 'last layer 8-bit attn'),
    ('language_model.model.layers.77.self_attn.fused_qkv_a_proj', 8, 'last layer fused 8-bit'),
]

all_pass = True
for prefix, expected, note in test_cases:
    actual = cfg._bits_for_prefix(prefix)
    status = 'OK' if actual == expected else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f"  [{status}] {prefix}")
    print(f"       expected bits={expected}, got bits={actual}  ({note})")

print()

# Check _checkpoint_is_quantized for key layers
print("=== _checkpoint_is_quantized results ===")
quant_cases = [
    ('language_model.model.layers.0.self_attn.q_a_proj', True),
    ('language_model.model.layers.0.self_attn.fused_qkv_a_proj', True),  # should be True via shards
    ('language_model.model.layers.0.mlp.gate_proj', False),  # fp16, NOT stored as qweight
    ('language_model.model.layers.3.mlp.shared_experts.gate_proj', True),
    ('language_model.model.layers.3.mlp.shared_experts.gate_up_proj', True),  # fused -> check shards
    ('language_model.model.layers.3.mlp.gate', False),  # router gate: fp16
    ('language_model.model.layers.0.self_attn.indexer.wk', True),  # 8-bit
    ('language_model.model.layers.0.self_attn.indexer.weights_proj', False),  # 16-bit
    ('language_model.model.layers.0.self_attn.indexer.wq_b', True),  # 8-bit
]

for prefix, expected in quant_cases:
    actual = cfg._checkpoint_is_quantized(prefix)
    status = 'OK' if actual == expected else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f"  [{status}] _checkpoint_is_quantized({prefix}) = {actual} (expected {expected})")

print()
print("=== get_quant_method dispatch for key LinearBase layers ===")
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

# We need to check what class get_quant_method returns for each key prefix.
# Build a mock layer
class MockLinear:
    pass

linear_mock = MockLinear()
linear_mock.__class__.__bases__ = (LinearBase,)

# Monkeypatch isinstance to treat MockLinear as LinearBase
import vllm.model_executor.layers.linear as _lin_mod
_orig_isinstance = __builtins__.__dict__.get('isinstance', None)

method_cases = [
    ('language_model.model.layers.0.self_attn.q_a_proj', 'GPTQDequantToBF16LinearMethod bits=8'),
    ('language_model.model.layers.0.self_attn.fused_qkv_a_proj', 'GPTQDequantToBF16LinearMethod bits=8'),
    ('language_model.model.layers.0.self_attn.o_proj', 'GPTQDequantToBF16LinearMethod bits=8'),
    ('language_model.model.layers.0.mlp.gate_proj', 'UnquantizedLinearMethod'),  # fp16 dense
    ('language_model.model.layers.3.mlp.gate', 'UnquantizedLinearMethod'),   # router
    ('language_model.model.layers.3.mlp.shared_experts.gate_proj', 'GPTQDequantToBF16LinearMethod bits=8'),
    ('language_model.model.layers.3.mlp.shared_experts.gate_up_proj', 'GPTQDequantToBF16LinearMethod bits=8'),
]

class FakeLinear(LinearBase):
    def __init__(self):
        pass  # skip super().__init__

fake_layer = FakeLinear()

for prefix, expected_desc in method_cases:
    method = cfg.get_quant_method(fake_layer, prefix)
    cls_name = type(method).__name__
    bits_attr = getattr(getattr(method, 'quant_config', None), 'weight_bits', '?')
    actual_desc = f"{cls_name}" + (f" bits={bits_attr}" if bits_attr != '?' else "")
    status = 'OK' if expected_desc in actual_desc or actual_desc in expected_desc else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f"  [{status}] {prefix}")
    print(f"       expected: {expected_desc}")
    print(f"       actual:   {actual_desc}")

print()
if all_pass:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED - see FAIL lines above")
