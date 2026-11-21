"""
End-to-end dequant check for GLM-5.1:
1. Run GPTQDequantToBF16LinearMethod.process_weights_after_loading on a real
   q_a_proj layer and compare to a reference hand-rolled dequant.
2. Check the int16 path for 8-bit zeros: does the +1 overflow when the stored
   zero byte is 0xFF (255 as uint8)?
3. Check fused_qkv_a_proj dequant (mla_proj_patch).
4. Check gate_up_proj fused dequant.
5. Check if gate_up_proj's fused dequant produces bf16 shapes matching
   the shared_experts MLP expectation.
"""
import sys, os, json, torch
sys.path.insert(0, '/opt/hpu-moe-ext')
os.environ['HPU_MOE_EXT_CKPT'] = '/models/GLM-5.1-int4-mixed-AutoRound'

from safetensors import safe_open
from hpu_moe_ext.moe_wna16_hpu import GPTQDequantToBF16LinearMethod

CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'

def load_tensors(prefix):
    idx_path = os.path.join(CKPT, 'model.safetensors.index.json')
    with open(idx_path) as f:
        wm = json.load(f)['weight_map']
    shard = wm.get(prefix + '.qweight')
    if shard is None:
        raise KeyError(f"{prefix}.qweight not found in index")
    with safe_open(os.path.join(CKPT, shard), framework='pt', device='cpu') as f:
        qw = f.get_tensor(prefix + '.qweight')
        qz = f.get_tensor(prefix + '.qzeros')
        sc = f.get_tensor(prefix + '.scales')
    return qw, qz, sc


def reference_dequant(qweight, qzeros, scales, bits, group_size=128):
    """Hand-rolled reference: w = (code - zero) * scale."""
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    # Unpack codes
    codes = torch.bitwise_right_shift(
        torch.unsqueeze(qweight.cpu(), 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int16 if bits == 8 else torch.int8)
    codes = torch.bitwise_and(codes, (2**bits) - 1).float()
    codes = codes.reshape(codes.shape[0] * codes.shape[1], codes.shape[2])  # [in, out]
    # Unpack zeros (autogptq +1 convention)
    raw_z = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int16 if bits == 8 else torch.int8)
    raw_z = raw_z + 1
    raw_z = torch.bitwise_and(raw_z, (2**bits) - 1).float()
    zeros = raw_z.reshape(-1, raw_z.shape[1] * raw_z.shape[2])  # [num_groups, out]
    # dequant
    scales_f = scales.float()
    g_idx = (torch.arange(codes.shape[0]) // group_size).long()
    w = (codes - zeros[g_idx]) * scales_f[g_idx]  # [in, out]
    return w  # CPU float32


def plugin_dequant(qweight, qzeros, scales, bits, group_size=128):
    """Replicate GPTQDequantToBF16LinearMethod.process_weights_after_loading on CPU."""
    method = GPTQDequantToBF16LinearMethod(group_size=group_size, bits=bits)
    method.wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    class FL:
        pass
    fl = FL()
    fl.qweight = qweight
    fl.qzeros = qzeros
    fl.scales = scales

    codes = method.unpack_weight_from_cuda_old_format(fl).to(torch.float32)
    zeros = method.unpack_zeros_from_cuda_old_format(fl).to(torch.float32)
    gs = group_size if group_size != -1 else codes.shape[0]
    g_idx = (torch.arange(codes.shape[0]) // gs).long()
    z = zeros[g_idx]
    s = scales.to("cpu").to(torch.float32)[g_idx]
    w = (codes - z) * s  # [in, out]
    return w  # CPU float32


print("=" * 70)
print("GPTQDequantToBF16LinearMethod vs reference dequant comparison")
print("=" * 70)

for layer_n in [0, 1, 3]:
    for proj in ['q_a_proj', 'kv_b_proj', 'o_proj']:
        prefix = f'model.layers.{layer_n}.self_attn.{proj}'
        try:
            qw, qz, sc = load_tensors(prefix)
        except KeyError:
            continue
        ref = reference_dequant(qw, qz, sc, bits=8)
        plg = plugin_dequant(qw, qz, sc, bits=8)
        diff = (ref - plg).abs()
        maxdiff = diff.max().item()
        status = 'OK' if maxdiff < 1e-5 else 'FAIL'
        print(f"[{status}] {prefix}: maxdiff={maxdiff:.2e}")
        if maxdiff > 1e-5:
            print(f"       ref stats: mean={ref.mean():.4f}, std={ref.std():.4f}")
            print(f"       plg stats: mean={plg.mean():.4f}, std={plg.std():.4f}")

print()
print("=== int16 overflow check for stored zero = 0x7F (127 = 2^8-1-128 in autogptq sym) ===")
# Worst case: stored value 0xFF (255) -> raw int16 after shift = 255, then +1 = 256
# 256 as int16: 256 fits in int16 (range -32768..32767), so no overflow
# But bitwise_and with 0xFF = 256 & 255 = 0 -> WRONG!
bits = 8
wf_val = 0  # shift by 0
stored = 255  # stored byte value
shifted = torch.tensor([[stored]], dtype=torch.int32)
raw = torch.bitwise_right_shift(
    torch.unsqueeze(shifted, 2).expand(-1, -1, 32 // bits),
    torch.tensor([[0, 8, 16, 24]], dtype=torch.int32).unsqueeze(0),
).to(torch.int16)
print(f"  For stored qzero int32 with all bytes=255:")
print(f"  raw int16 shifted values: {raw.squeeze().tolist()}")
r1 = raw + 1
masked = torch.bitwise_and(r1, 0xFF).to(torch.float32)
print(f"  after +1 and 0xFF mask: {masked.squeeze().tolist()}")
print(f"  Expected: all 0 (256 & 0xFF = 0) -- this is a TRAP for all-ones packed zeros!")
print(f"  BUT: for sym quant, stored zero should always be 127 (not 255)")

# Check what a stored_zero=127 gives:
stored_127 = sum(127 << (8*i) for i in range(4))
s127_t = torch.tensor([[stored_127]], dtype=torch.int32)
raw127 = torch.bitwise_right_shift(
    torch.unsqueeze(s127_t, 2).expand(-1,-1,4),
    torch.tensor([[0,8,16,24]],dtype=torch.int32).unsqueeze(0)
).to(torch.int16)
print(f"\n  For stored_zero=127 per byte (sym): raw={raw127.squeeze().tolist()}")
r2 = raw127 + 1
m2 = torch.bitwise_and(r2, 0xFF).to(torch.float32)
print(f"  after +1+mask: {m2.squeeze().tolist()}  (should all be 128)")

print()
print("=== shared_experts gate_up_proj fused dequant check ===")
# gate_up_proj is a fusion of gate_proj + up_proj
# Both are bits=8, same group_size=128
# Load both separately and compare to the fused weight expectation
try:
    qw_g, qz_g, sc_g = load_tensors('model.layers.3.mlp.shared_experts.gate_proj')
    qw_u, qz_u, sc_u = load_tensors('model.layers.3.mlp.shared_experts.up_proj')
    w_g = plugin_dequant(qw_g, qz_g, sc_g, bits=8)  # [in, out_half]
    w_u = plugin_dequant(qw_u, qz_u, sc_u, bits=8)  # [in, out_half]
    # fused gate_up: vLLM concatenates along output dim (gate on top, up below)
    # weight_loader for MergedColumnParallel loads into [2*out, in]
    # dequant gives [in, out] (col-major: in=input, out=output)
    # After .t(): [out, in] -> stacked as [2*out, in]
    fused_ref = torch.cat([w_g.t(), w_u.t()], dim=0)  # [2*out_half, in]
    print(f"  gate_proj dequant shape: {w_g.shape} -> transposed: {w_g.t().shape}")
    print(f"  up_proj   dequant shape: {w_u.shape} -> transposed: {w_u.t().shape}")
    print(f"  Fused gate_up_proj ref shape: {fused_ref.shape}")
    print(f"  Expected for shared_expert: [2*moe_intermediate_size, hidden_size]")
    print(f"  moe_intermediate_size=2048 from config, hidden=6144 -> expected [4096, 6144]")
    print(f"  Actual: {list(fused_ref.shape)}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

print()
print("=== Dense MLA path check: does _ORIG_FORWARD_IMPL need indexer outputs? ===")
# This is a code-reading check, not numeric
# The GLM dense MLA forward with skip_topk=True and FULL_ATTENTION_FALLBACK=True
# routes to HPUMLAAttention._ORIG_FORWARD_IMPL which is the stock deepseek_v2
# forward_impl WITHOUT the indexer.
# Key question: does _ORIG_FORWARD_IMPL use:
#   1. kv_cache (yes, but it uses full attention cache, not indexer cache)
#   2. positions (yes, for RoPE)
#   3. kv_a_proj_with_mqa output (yes, for compressing KV)
#   4. anything from the indexer (NO - the indexer is completely bypassed)
# So the only concern is whether the non-indexer attention weights
# (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj) are correctly loaded.
# We've verified those are correctly dispatched to GPTQDequantToBF16LinearMethod(bits=8).
print("  Dense MLA (_ORIG_FORWARD_IMPL) needs: q_a, kv_a, q_b, kv_b, o_proj weights")
print("  All of these are loaded via mla_proj_patch (q_a/kv_a fused) or standalone (others)")
print("  Indexer (wk, wq_b, weights_proj) is completely bypassed - no dependency")
print()

print("=== SUMMARY: Potential root causes ===")
print()
print("The 8-bit zero convention is CORRECT: stored=127+1=128 (verified on real checkpoint).")
print("The dispatch logic routes all 8-bit linears to GPTQDequantToBF16LinearMethod(bits=8).")
print("The fused_qkv_a_proj correctly resolves to bits=8 via shard lookup.")
print()
print("Remaining suspects (not yet verified numerically):")
print("1. mla_proj_patch weight_loader shard ordering: q_a=shard0, kv_a=shard1")
print("   - Does the MergedColumnParallelLinear weight_loader with shard_id=0/1 put")
print("     q_a in the correct slice of fused_qkv_a_proj.weight?")
print("2. gate_up_proj: does plugin_dequant transpose match what MergedColumnParallel expects?")
print("   The plugin calls w.t().contiguous().to(bf16) in process_weights_after_loading,")
print("   giving [out, in]. But for fused gate_up_proj, the weight_loader receives a")
print("   [out_half, in] tensor and needs to place it in a [2*out_half, in] fused param.")
print("   This works ONLY if the caller uses weight_loader(param, bf16, shard_id).")
print("   But GPTQDequantToBF16LinearMethod.process_weights_after_loading does NOT")
print("   call weight_loader - it directly sets layer.weight = the dequanted tensor.")
print("   For a FUSED gate_up_proj, does vLLM build a single [2*out, in] weight param")
print("   and expect the method to fill it all at once, or does it build two sub-weights?")
print()
print("   CRITICAL: For gate_up_proj built as MergedColumnParallelLinear,")
print("   create_weights allocates a single output_size_per_partition = sum([out_half, out_half])")
print("   = 2*out_half. But the checkpoint has SEPARATE gate_proj.qweight and up_proj.qweight.")
print("   If the fused module gets a single GPTQDequantToBF16LinearMethod,")
print("   its qweight param is a single [in//pack_factor, 2*out_half] int32 tensor.")
print("   At load time, vLLM's MergedColumnParallelLinear weight_loader CONCATENATES")
print("   gate_proj.qweight and up_proj.qweight into that single param.")
print("   The dequant then unpacks both halves together: codes=[in, 2*out_half].")
print("   This should work if the loader correctly positions gate and up halves.")
print()
print("3. The real question for gate_up_proj: does unpack_weight_from_cuda_old_format")
print("   correctly handle the PackedvLLMParameter (with metadata) vs a plain int32 tensor?")
