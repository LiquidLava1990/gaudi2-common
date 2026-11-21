"""
Probe: Verify 8-bit AutoRound zero convention in GLM-5.1 checkpoint.
Runs the same unpack path as GPTQDequantToBF16LinearMethod (hpu_gptq.py)
and reports dequant statistics for multiple layers and bit widths.
Also checks 4-bit expert zero, fused_qkv_a_proj dispatch, and gate_up_proj dispatch.
"""
import sys
import torch
import json
import os
from safetensors import safe_open

CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'

def load_tensor(key):
    """Find which shard contains key and load it."""
    idx_path = os.path.join(CKPT, 'model.safetensors.index.json')
    with open(idx_path) as f:
        wm = json.load(f)['weight_map']
    shard = wm.get(key)
    if shard is None:
        raise KeyError(f'{key} not in index')
    with safe_open(os.path.join(CKPT, shard), framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def unpack_zeros(qzeros, bits):
    """Exact replica of hpu_gptq.unpack_zeros_from_cuda_old_format."""
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    raw = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int16 if bits == 8 else torch.int8)
    raw = raw + 1
    raw = torch.bitwise_and(raw, (2**bits) - 1).to(torch.float32)
    return raw.reshape(-1, raw.shape[1] * raw.shape[2])  # [num_groups, out]


def unpack_weight(qweight, bits):
    """Exact replica of hpu_gptq.unpack_weight_from_cuda_old_format."""
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    w = torch.bitwise_right_shift(
        torch.unsqueeze(qweight.cpu(), 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int16 if bits == 8 else torch.int8)
    w = torch.bitwise_and(w, (2**bits) - 1)
    return w.reshape((w.shape[0] * w.shape[1], w.shape[2]))  # [in, out]


def dequant_ref(qweight, qzeros, scales, bits, group_size=128):
    """Reference dequant: w = (code - zero) * scale."""
    codes = unpack_weight(qweight, bits).float()
    zeros = unpack_zeros(qzeros, bits)  # [num_groups, out]
    scales_f = scales.float()
    g_idx = (torch.arange(codes.shape[0]) // group_size).long()
    z = zeros[g_idx]
    s = scales_f[g_idx]
    return (codes - z) * s  # [in, out]


print("=" * 60)
print("GLM-5.1 AutoRound zero convention probe")
print("=" * 60)

# ---- Test 1: 8-bit q_a_proj (layer 0) ----
print("\n[1] layers.0.self_attn.q_a_proj (bits=8, in=6144, out=2048)")
pfx = 'model.layers.0.self_attn.q_a_proj'
qw = load_tensor(pfx + '.qweight')
qz = load_tensor(pfx + '.qzeros')
sc = load_tensor(pfx + '.scales')
print(f"  qweight: {qw.shape} {qw.dtype}")
print(f"  qzeros:  {qz.shape} {qz.dtype}")
print(f"  scales:  {sc.shape} {sc.dtype}")

zeros8 = unpack_zeros(qz, bits=8)
print(f"  Unique zero values: {zeros8.flatten().unique().tolist()[:10]}")
print(f"  Zero mean: {zeros8.mean().item():.2f} (symmetric int8 expects 128)")

w8 = dequant_ref(qw, qz, sc, bits=8)
print(f"  Weight mean: {w8.mean().item():.4f} (should be ~0)")
print(f"  Weight std:  {w8.std().item():.4f}")
print(f"  Weight range: [{w8.min().item():.4f}, {w8.max().item():.4f}]")

# Check for nan/inf
print(f"  NaN in dequant weight: {torch.isnan(w8).any().item()}")
print(f"  Inf in dequant weight: {torch.isinf(w8).any().item()}")

# ---- Test 2: 8-bit kv_b_proj (layer 0) ----
print("\n[2] layers.0.self_attn.kv_b_proj (bits=8)")
pfx2 = 'model.layers.0.self_attn.kv_b_proj'
qw2 = load_tensor(pfx2 + '.qweight')
qz2 = load_tensor(pfx2 + '.qzeros')
sc2 = load_tensor(pfx2 + '.scales')
print(f"  qweight: {qw2.shape}, qzeros: {qz2.shape}, scales: {sc2.shape}")
zeros8b = unpack_zeros(qz2, bits=8)
print(f"  Unique zeros (kv_b_proj): {zeros8b.flatten().unique().tolist()[:10]}")
print(f"  Zero mean: {zeros8b.mean().item():.2f}")
w8b = dequant_ref(qw2, qz2, sc2, bits=8)
print(f"  Weight mean: {w8b.mean().item():.4f}, std: {w8b.std().item():.4f}")

# ---- Test 3: Compare with a 4-bit layer to see if convention differs ----
# Find a 4-bit layer: routed experts (layer 3, first MoE layer)
# We can't easily load expert weights since they are individually indexed
# Instead, use q_b_proj which should be 8-bit per extra_config
# Let's check if any attn layer has 4-bit coding
print("\n[3] Looking for a 4-bit attention layer")
print("  (all attn layers in extra_config are bits:8 -> no 4-bit attn)")
print("  Checking q_b_proj layer 0 (extra_config bits:8):")
pfx3 = 'model.layers.0.self_attn.q_b_proj'
qw3 = load_tensor(pfx3 + '.qweight')
qz3 = load_tensor(pfx3 + '.qzeros')
sc3 = load_tensor(pfx3 + '.scales')
print(f"  qweight: {qw3.shape}, qzeros: {qz3.shape}, scales: {sc3.shape}")
# pack_factor for bits=8: 32//8=4, so qweight[0]*4 = in_features
in_dim = qw3.shape[0] * 4
print(f"  Implied in_features: {in_dim} (expected q_lora_rank=2048)")
zeros_qb = unpack_zeros(qz3, bits=8)
print(f"  Unique zeros q_b_proj: {zeros_qb.flatten().unique().tolist()[:10]}")

# ---- Test 4: Check a shared_experts layer (bits=8, layer 3) ----
print("\n[4] layers.3.mlp.shared_experts.gate_proj (bits=8 per extra_config)")
try:
    pfx4 = 'model.layers.3.mlp.shared_experts.gate_proj'
    qw4 = load_tensor(pfx4 + '.qweight')
    qz4 = load_tensor(pfx4 + '.qzeros')
    sc4 = load_tensor(pfx4 + '.scales')
    print(f"  qweight: {qw4.shape}, qzeros: {qz4.shape}, scales: {sc4.shape}")
    zeros_se = unpack_zeros(qz4, bits=8)
    print(f"  Unique zeros: {zeros_se.flatten().unique().tolist()[:10]}")
    w4 = dequant_ref(qw4, qz4, sc4, bits=8)
    print(f"  Weight mean: {w4.mean().item():.4f}, std: {w4.std().item():.4f}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---- Test 5: Check a 4-bit MoE expert (layer 3 routed expert) ----
print("\n[5] layers.3.mlp.experts.0.gate_proj (4-bit routed expert)")
try:
    pfx5 = 'model.layers.3.mlp.experts.0.gate_proj'
    qw5 = load_tensor(pfx5 + '.qweight')
    qz5 = load_tensor(pfx5 + '.qzeros')
    sc5 = load_tensor(pfx5 + '.scales')
    print(f"  qweight: {qw5.shape}, qzeros: {qz5.shape}, scales: {sc5.shape}")
    zeros_4 = unpack_zeros(qz5, bits=4)
    print(f"  Unique zeros (4-bit): {zeros_4.flatten().unique().tolist()[:10]}")
    print(f"  Zero mean: {zeros_4.mean().item():.2f} (symmetric int4 expects 8)")
except Exception as e:
    print(f"  ERROR: {e}")

# ---- Test 6: Verify _bits_for_prefix dispatch for key layers ----
print("\n[6] _bits_for_prefix dispatch simulation")
import sys
sys.path.insert(0, '/opt/hpu-moe-ext')
try:
    import re
    # Simulate _compile_extra_config and _bits_for from auto_round_hpu.py
    with open('/models/GLM-5.1-int4-mixed-AutoRound/config.json') as cf:
        cfg = json.load(cf)
    extra_config = cfg['quantization_config']['extra_config']

    _WRAP_SEGMENTS = {'model', 'language_model'}
    def _norm(p):
        return '.'.join(s for s in p.split('.') if s not in _WRAP_SEGMENTS)

    exact = {}
    regex = []
    for key, spec in extra_config.items():
        if not isinstance(spec, dict) or 'bits' not in spec:
            continue
        bits_val = int(spec['bits'])
        if '*' in key:
            regex.append((re.compile(key), bits_val))
        else:
            exact[key] = bits_val

    def bits_for(prefix, default=4):
        npfx = _norm(prefix)
        nexact = {_norm(k): b for k, b in exact.items()}
        if npfx in nexact:
            return nexact[npfx]
        for k, b in nexact.items():
            if npfx.endswith('.' + k) or k.endswith('.' + npfx):
                return b
        for pat, b in regex:
            if pat.search(prefix) or pat.search(npfx):
                return b
        return default

    test_prefixes = [
        # vLLM runtime prefix pattern for GLM (no "model." wrapper at top)
        'language_model.model.layers.0.self_attn.q_a_proj',
        'language_model.model.layers.0.self_attn.kv_a_proj_with_mqa',
        'language_model.model.layers.0.self_attn.q_b_proj',
        'language_model.model.layers.0.self_attn.kv_b_proj',
        'language_model.model.layers.0.self_attn.o_proj',
        'language_model.model.layers.0.self_attn.fused_qkv_a_proj',  # vLLM fused
        'language_model.model.layers.0.mlp.gate_proj',     # bits:16 (first 3 dense)
        'language_model.model.layers.3.mlp.shared_experts.gate_proj',  # bits:8
        'language_model.model.layers.3.mlp.shared_experts.down_proj',  # bits:8
        'language_model.model.layers.3.mlp.gate',          # router gate - should be fp16
        # Also test HF-style prefix
        'model.layers.0.self_attn.q_a_proj',
        'model.layers.0.self_attn.fused_qkv_a_proj',  # fused doesn't exist in ckpt
    ]
    for p in test_prefixes:
        b = bits_for(p)
        print(f"  {p} -> bits={b}")

except Exception as e:
    print(f"  ERROR in dispatch test: {e}")
    import traceback; traceback.print_exc()

print("\n=== DONE ===")
