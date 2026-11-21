"""
CRITICAL: verify int16 truncation in 8-bit zero unpack.
For 8-bit GPTQ, each int32 packs 4 zero bytes.
0x7F7F7F7F >> 8 = 0x007F7F7F as int32 = 8355711, which when cast to int16
truncates to the LOW 16 bits: 0x7F7F = 32639.
32639 + 1 = 32640 = 0x7F80. 0x7F80 & 0xFF = 0x80 = 128. Still correct!

But let's verify rigorously for all 4 shift positions.
"""
import torch

print("=== int16 truncation in 8-bit unpack_zeros ===")
packed_127 = sum(127 << (8*i) for i in range(4))
print(f"packed_127 = {hex(packed_127)} = {packed_127}")

# Simulate the exact unpack_zeros_from_cuda_old_format code
bits = 8
wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
qzeros_1d = torch.tensor([[packed_127]], dtype=torch.int32)  # [1 group, 1 packed col]

# Step by step
expanded = torch.unsqueeze(qzeros_1d, 2).expand(-1, -1, 32 // bits)
shifted = torch.bitwise_right_shift(expanded, wf.unsqueeze(0))
print(f"shifted int32 values: {shifted.squeeze().tolist()}")
shifted_i16 = shifted.to(torch.int16)
print(f"shifted int16 values: {shifted_i16.squeeze().tolist()}")
plus1 = shifted_i16 + 1
print(f"after +1:             {plus1.squeeze().tolist()}")
masked = torch.bitwise_and(plus1, (2**bits) - 1).to(torch.float32)
print(f"after &0xFF:          {masked.squeeze().tolist()}")
print(f"Expected: all 128")
print()

# Verify per-shift by hand
print("--- Per-shift manual check ---")
val = packed_127
for s in [0, 8, 16, 24]:
    v32 = torch.tensor(val, dtype=torch.int32)
    shifted_val = (v32 >> s).item()
    i16 = torch.tensor(shifted_val, dtype=torch.int32).to(torch.int16).item()
    p1 = i16 + 1
    msk = p1 & 0xFF
    print(f"  shift={s}: int32={hex(shifted_val)}={shifted_val}, int16={i16}={hex(i16 & 0xFFFF)}, +1={p1}, &0xFF={msk}")

print()
print("=== Now check the WEIGHT unpack (int16 for 8-bit codes) ===")
# weight unpack uses the same int16 path for bits==8
# For a weight code e.g. 200 (> 128, upper half of range)
# packed: 0xC8C8C8C8
code_200 = sum(200 << (8*i) for i in range(4))
qweight_1d = torch.tensor([[code_200]], dtype=torch.int32)
expanded_w = torch.unsqueeze(qweight_1d, 1).expand(-1, 32 // bits, -1)
shifted_w = torch.bitwise_right_shift(expanded_w, wf.unsqueeze(-1))
print(f"Code 200 packed: {hex(code_200)}")
print(f"shifted int32: {shifted_w.squeeze().tolist()}")
shifted_w_i16 = shifted_w.to(torch.int16)
print(f"shifted int16: {shifted_w_i16.squeeze().tolist()}")
masked_w = torch.bitwise_and(shifted_w_i16, (2**bits) - 1).to(torch.float32)
print(f"after &0xFF:   {masked_w.squeeze().tolist()}")
print(f"Expected: all 200")
print()

# For code 200: 0x200200... >> 8 = 0x00C8C8... as int32 = 0xC8C8 in int16 = -14136 (signed)
# But &0xFF: -14136 & 0xFF = 200? Let's check
print("--- Code 200 per-shift manual check ---")
for s in [0, 8, 16, 24]:
    v32 = torch.tensor(code_200, dtype=torch.int32)
    shifted_val = (v32 >> s).item()
    i16 = torch.tensor(shifted_val, dtype=torch.int32).to(torch.int16).item()
    msk = i16 & 0xFF  # note: python int & 0xFF
    print(f"  shift={s}: shifted_i32={hex(shifted_val & 0xFFFFFFFF)}, int16={i16}={hex(i16 & 0xFFFF)}, &0xFF={msk}")

print()
print("=== VERDICT for 8-bit unpack_zeros ===")
# Based on the above: the &0xFF mask correctly extracts the lowest byte regardless
# of int16 sign, because Python (and PyTorch) bitwise_and with int16 uses 2's complement
# So even negative int16 values give correct results after &0xFF
# CONFIRMED: the 8-bit zero unpack is numerically correct.
all_128 = masked.flatten().tolist()
all_200 = masked_w.flatten().tolist()
zero_ok = all(abs(v - 128) < 1e-5 for v in all_128)
code_ok = all(abs(v - 200) < 1e-5 for v in all_200)
print(f"  Zeros all=128: {zero_ok}")
print(f"  Codes all=200: {code_ok}")

if zero_ok and code_ok:
    print("  CONFIRMED: 8-bit int16 unpack is numerically correct.")
else:
    print("  FAIL: 8-bit unpack has errors!")
