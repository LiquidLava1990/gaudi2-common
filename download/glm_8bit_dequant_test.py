"""
CPU numerics test: compare plugin's 8-bit dequant vs canonical auto_gptq sym dequant
for a real GLM-5.1 8-bit attention weight (layer 3 o_proj).

THE KEY QUESTIONS:
1. Does unpack_weight produce correct codes (0..255)?
2. Does unpack_zeros produce correct zero (should be 128 for sym-8bit)?
3. Is dequant w = (codes - zero) * scales correct vs reference?
4. Does the int16 intermediate in unpack cause sign-extension issues (8-bit codes 0-255 vs signed int16)?
"""
import sys
import torch
from fractions import Fraction
from safetensors import safe_open

MODEL_DIR = "/models/GLM-5.1-int4-mixed-AutoRound"
SHARD = MODEL_DIR + "/model-00001-of-00075.safetensors"
PREFIX = "model.layers.3.self_attn.o_proj"

# ---- Load raw tensors from checkpoint ----
print("=== Loading raw checkpoint tensors ===")
with safe_open(SHARD, framework="pt", device="cpu") as f:
    qweight_raw = f.get_tensor(PREFIX + ".qweight")
    qzeros_raw = f.get_tensor(PREFIX + ".qzeros")
    scales_raw = f.get_tensor(PREFIX + ".scales")

print(f"qweight shape: {qweight_raw.shape}  dtype: {qweight_raw.dtype}")
print(f"qzeros  shape: {qzeros_raw.shape}  dtype: {qzeros_raw.dtype}")
print(f"scales  shape: {scales_raw.shape}  dtype: {scales_raw.dtype}")
print(f"qzeros raw values [row0, first 4 int32 words]: {qzeros_raw[0,:4].tolist()}")
print(f"scales  sample [row0, first 4]: {scales_raw[0,:4].tolist()}")
print()

# ---- Reproduce the plugin's unpack logic verbatim ----
# This is GPTQDequantToBF16LinearMethod.process_weights_after_loading
# which calls the inherited GPTQHPULinearMethod methods.

bits = 8
pack_factor = Fraction(32, bits)  # = 4
wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
print(f"bits={bits}, pack_factor={pack_factor}, wf={wf}")
print()

# --- Plugin unpack_weight_from_cuda_old_format ---
def plugin_unpack_weight(qweight, bits):
    """Exact copy of GPTQHPULinearMethod.unpack_weight_from_cuda_old_format"""
    wf_local = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf_local.unsqueeze(-1),
    ).to(torch.int16 if bits == 8 else torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape((weight.shape[0] * weight.shape[1], weight.shape[2]))
    return weight

# --- Plugin unpack_zeros_from_cuda_old_format ---
def plugin_unpack_zeros(qzeros, scales, bits):
    """Exact copy of GPTQHPULinearMethod.unpack_zeros_from_cuda_old_format"""
    wf_local = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
        wf_local.unsqueeze(0),
    ).to(torch.int16 if bits == 8 else torch.int8)
    zeros = zeros + 1
    zeros = torch.bitwise_and(zeros, (2**bits) - 1).to(scales.dtype)
    zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
    return zeros

# --- Reference canonical auto_gptq sym 8-bit dequant ---
def ref_unpack_weight_8bit(qweight):
    """Canonical 8-bit unpack: each int32 holds 4 uint8 values packed LSB-first.
    Codes should be in range [0, 255]."""
    # Each int32 at position i holds 4 codes:
    # code[4*i+0] = (qweight[i] >> 0) & 0xFF
    # code[4*i+1] = (qweight[i] >> 8) & 0xFF
    # code[4*i+2] = (qweight[i] >> 16) & 0xFF
    # code[4*i+3] = (qweight[i] >> 24) & 0xFF
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
    # qweight: [in//4, out]
    # expand: [in//4, 4, out]
    expanded = torch.unsqueeze(qweight, 1).expand(-1, 4, -1)
    codes = (torch.bitwise_right_shift(expanded, shifts.view(1,-1,1).to(torch.int32)) & 0xFF).to(torch.int32)
    # reshape: [in, out]
    codes = codes.reshape(qweight.shape[0] * 4, qweight.shape[1])
    return codes

def ref_unpack_zeros_8bit_sym(qzeros, scales):
    """Canonical sym 8-bit zero-point from GPTQ checkpoint.

    AutoRound (packing_format=auto_round:auto_gptq) uses the autogptq convention:
    stored_zero = true_zero - 1  (i.e. +1 is applied on load).
    For symmetric 8-bit: true_zero should be 128 (midpoint of uint8).
    So stored value should be 127 (= 128 - 1).

    Let's unpack and check.
    """
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
    expanded = torch.unsqueeze(qzeros, 2).expand(-1, -1, 4)
    raw_codes = (torch.bitwise_right_shift(expanded, shifts.view(1,1,-1).to(torch.int32)) & 0xFF).to(torch.int32)
    raw_codes = raw_codes.reshape(qzeros.shape[0], qzeros.shape[1] * 4)
    # Apply +1 convention (autogptq)
    true_zeros = (raw_codes + 1) & 0xFF
    return true_zeros.to(scales.dtype)

# ---- Run plugin unpackers ----
print("=== Plugin unpack_weight ===")
plugin_codes = plugin_unpack_weight(qweight_raw, bits)
print(f"Output shape: {plugin_codes.shape}  dtype: {plugin_codes.dtype}")
print(f"Min: {plugin_codes.min().item():.0f}  Max: {plugin_codes.max().item():.0f}  Mean: {plugin_codes.float().mean().item():.2f}")
print(f"Sample first 8 codes [row 0]: {plugin_codes[0, :8].tolist()}")
print(f"Sample first 8 codes [row 1]: {plugin_codes[1, :8].tolist()}")
print()

print("=== Reference unpack_weight (int32 no int16 cast) ===")
ref_codes = ref_unpack_weight_8bit(qweight_raw)
print(f"Output shape: {ref_codes.shape}  dtype: {ref_codes.dtype}")
print(f"Min: {ref_codes.min().item():.0f}  Max: {ref_codes.max().item():.0f}  Mean: {ref_codes.float().mean().item():.2f}")
print(f"Sample first 8 codes [row 0]: {ref_codes[0, :8].tolist()}")
print(f"Sample first 8 codes [row 1]: {ref_codes[1, :8].tolist()}")
print()

# CRITICAL: check if int16 cast causes negative values for codes >= 128
print("=== CRITICAL: int16 cast sign extension check ===")
# int16 can hold -32768..32767; but after masking with 0xFF codes are 0..255, all positive in int16 too
# However: torch.int16 has range [-128..127]... wait no, int16 is -32768..32767
# The issue may be: .to(torch.int16) before bitwise_and -- does the right-shift into int16 work?
# Let's check: after right-shift and cast to int16, before masking
wf_test = torch.tensor(list(range(0, 32, 8)), dtype=torch.int32).unsqueeze(0)
pre_mask = torch.bitwise_right_shift(
    torch.unsqueeze(qweight_raw[:4], 1).expand(-1, 4, -1),
    wf_test.unsqueeze(-1),
).to(torch.int16)
print(f"After right-shift + int16 cast (before mask), shape: {pre_mask.shape}")
print(f"  Min: {pre_mask.min().item()}  Max: {pre_mask.max().item()}")
print(f"  Negative count: {(pre_mask < 0).sum().item()} out of {pre_mask.numel()}")

# After mask
post_mask_int16 = torch.bitwise_and(pre_mask, (2**8) - 1)
print(f"After mask with 0xFF (int16), Min: {post_mask_int16.min().item()}  Max: {post_mask_int16.max().item()}")
print(f"  Negative count after mask: {(post_mask_int16 < 0).sum().item()}")
print()

# Compare plugin vs reference codes
codes_match = torch.allclose(plugin_codes.float(), ref_codes.float())
maxdiff_codes = (plugin_codes.float() - ref_codes.float()).abs().max().item()
print(f"=== Weight codes match: {codes_match}, maxdiff: {maxdiff_codes} ===")
print()

# ---- Run plugin unpack_zeros ----
print("=== Plugin unpack_zeros ===")
plugin_zeros = plugin_unpack_zeros(qzeros_raw, scales_raw, bits)
print(f"Output shape: {plugin_zeros.shape}  dtype: {plugin_zeros.dtype}")
print(f"Min: {plugin_zeros.min().item():.1f}  Max: {plugin_zeros.max().item():.1f}  Mean: {plugin_zeros.float().mean().item():.2f}")
print(f"Unique zero values (should be ~128 for sym-8bit): {torch.unique(plugin_zeros.float()).tolist()[:10]}")
print(f"Sample [row0, first 8]: {plugin_zeros[0, :8].tolist()}")
print()

print("=== Reference unpack_zeros (canonical auto_gptq sym 8-bit) ===")
ref_zeros = ref_unpack_zeros_8bit_sym(qzeros_raw, scales_raw)
print(f"Output shape: {ref_zeros.shape}  dtype: {ref_zeros.dtype}")
print(f"Min: {ref_zeros.min().item():.1f}  Max: {ref_zeros.max().item():.1f}  Mean: {ref_zeros.float().mean().item():.2f}")
print(f"Unique zero values: {torch.unique(ref_zeros.float()).tolist()[:10]}")
print(f"Sample [row0, first 8]: {ref_zeros[0, :8].tolist()}")
print()

# CRITICAL: check raw stored zeros -- what value is actually in qzeros?
print("=== RAW stored qzeros analysis ===")
raw_z_shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
raw_z_expanded = torch.unsqueeze(qzeros_raw, 2).expand(-1, -1, 4)
raw_z_codes = (torch.bitwise_right_shift(raw_z_expanded, raw_z_shifts.view(1,1,-1).to(torch.int32)) & 0xFF).to(torch.int32)
raw_z_codes = raw_z_codes.reshape(qzeros_raw.shape[0], qzeros_raw.shape[1] * 4)
print(f"Raw zero codes shape: {raw_z_codes.shape}")
print(f"Unique raw stored zero values: {torch.unique(raw_z_codes).tolist()}")
print(f"Sample raw stored zeros [row0, first 8]: {raw_z_codes[0, :8].tolist()}")
print()

# CRITICAL: int16 for zeros unpack - same sign extension risk
print("=== CRITICAL: zeros int16 cast sign extension check ===")
wf_z = torch.tensor(list(range(0, 32, 8)), dtype=torch.int32).unsqueeze(0)
zeros_pre = torch.bitwise_right_shift(
    torch.unsqueeze(qzeros_raw[:4], 2).expand(-1, -1, 4),
    wf_z.unsqueeze(0),
).to(torch.int16)
print(f"Zeros after right-shift + int16 cast, Min: {zeros_pre.min().item()}  Max: {zeros_pre.max().item()}")
zeros_post = torch.bitwise_and(zeros_pre, (2**8) - 1) + 1
print(f"After mask+1 (plugin path), unique values: {torch.unique(zeros_post.float()).tolist()[:10]}")
print()

# Now compare zeros
zeros_match = torch.allclose(plugin_zeros.float(), ref_zeros.float())
maxdiff_zeros = (plugin_zeros.float() - ref_zeros.float()).abs().max().item()
print(f"=== Zero-point match: {zeros_match}, maxdiff: {maxdiff_zeros} ===")
print()

# ---- Full dequant comparison ----
print("=== Full dequant: w = (codes - zeros) * scales ===")
group_size = 128

# Plugin dequant
p_codes = plugin_codes.float()
p_zeros = plugin_zeros.float()  # shape [num_groups, out]
p_scales = scales_raw.float()   # shape [num_groups, out]
# p_codes shape [in, out]; map rows to groups
g_idx = (torch.arange(p_codes.shape[0]) // group_size).long()
p_z = p_zeros[g_idx]   # [in, out]
p_s = p_scales[g_idx]  # [in, out]
plugin_w = (p_codes - p_z) * p_s

# Reference dequant
r_codes = ref_codes.float()
r_zeros = ref_zeros.float()
r_scales = scales_raw.float()
r_z = r_zeros[g_idx]
r_s = r_scales[g_idx]
ref_w = (r_codes - r_z) * r_s

maxdiff_w = (plugin_w - ref_w).abs().max().item()
print(f"Plugin weight Min: {plugin_w.min().item():.4f}  Max: {plugin_w.max().item():.4f}")
print(f"Ref    weight Min: {ref_w.min().item():.4f}  Max: {ref_w.max().item():.4f}")
print(f"=== Dequant maxdiff plugin vs ref: {maxdiff_w} ===")
print()

# ---- Expected zero for sym-8bit ----
print("=== Zero-point correctness analysis ===")
print(f"Sym 8-bit: expected zero_point = 128 (midpoint of uint8 0..255)")
print(f"autogptq stored convention: stored_zero = true_zero - 1 = 127")
print(f"Plugin adds +1: stored_zero + 1 = 128 -> CORRECT IF stored=127")
print(f"  Actual plugin zero mean: {plugin_zeros.float().mean().item():.2f}  (should be 128.0)")
print(f"  Actual ref zero mean: {ref_zeros.float().mean().item():.2f}  (should be 128.0)")
print()

# ---- Verify dequant is physically reasonable ----
# For sym-8bit with zero=128, codes are 0-255 centered at 128
# (codes-128) gives -128..+127 range, then scaled
print("=== Dequant physical reasonability check ===")
# Check scales
print(f"Scale sample [row0, first 4]: {scales_raw[0, :4].tolist()}")
print(f"Scale magnitude (abs mean): {scales_raw.float().abs().mean().item():.6f}")
# With a typical LLM scale ~1e-3 to 1e-2 and code range 0-255 centered at 128:
# weight range ~ +/- 128 * scale ~ +/- 0.1 to 1.3 -- plausible
print(f"Plugin weight range: [{plugin_w.min().item():.4f}, {plugin_w.max().item():.4f}]")
print(f"Plugin weight non-zero fraction: {(plugin_w != 0).float().mean().item():.4f}")
print()

# ---- The critical bug hunt: what does autogptq ACTUALLY do for 8-bit sym? ----
# AutoRound with sym=True, 8-bit uses packing_format=auto_round:auto_gptq
# In auto_gptq convention for sym quant: zero_point = 0 is stored because
# for sym quant, auto_gptq does NOT use the standard "+1" offset trick.
# Let me check: is the stored zero actually 0x00 or 0x7F?
print("=== AutoRound vs GPTQ zero convention analysis ===")
print(f"Raw stored qzero bytes unique values: {torch.unique(raw_z_codes).tolist()}")
print()
# If auto_round sym packs zero_point as actual INT8 centered (i.e. 0 for zero-centered),
# then stored_zero = 0, and after +1 -> plugin zero = 1 -> WRONG (should be 128)
# If auto_round packs like autogptq sym4 (stored = midpoint - 1 = 127), plugin zero = 128 -> CORRECT

# Let's also check what value makes dequant produce near-zero mean weight (sanity)
print("=== Checking correct zero by examining weight statistics ===")
for test_zero in [0, 1, 127, 128, 129, 255]:
    test_w = (r_codes.float() - test_zero) * r_scales[g_idx]
    print(f"  zero={test_zero}: weight mean={test_w.mean().item():.6f}, abs_mean={test_w.float().abs().mean().item():.6f}")
print()

print("=== SUMMARY ===")
print(f"Codes: plugin vs ref maxdiff = {maxdiff_codes}")
print(f"Zeros: plugin vs ref maxdiff = {maxdiff_zeros}")
print(f"Dequant: plugin vs ref maxdiff = {maxdiff_w}")
print(f"Plugin zero value (mean): {plugin_zeros.float().mean().item():.2f}")
print(f"Ref zero value (mean): {ref_zeros.float().mean().item():.2f}")
if abs(plugin_zeros.float().mean().item() - 128.0) > 1:
    print("BUG: zero-point is NOT 128 -- dequant offset is wrong!")
else:
    print("Zero-point looks correct (128)")
if maxdiff_codes > 0.5:
    print("BUG: weight code unpacking is wrong!")
else:
    print("Weight code unpack: matches reference")
