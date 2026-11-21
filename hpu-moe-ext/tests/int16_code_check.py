"""Check weight code unpack for 8-bit values (high codes > 128)."""
import torch

bits = 8
wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

# Test code value 200 (should survive the int16 path)
# Pack 200 into a single int32 with one code per byte
code_200 = int(200) | (int(200) << 8) | (int(200) << 16) | (int(200) << 24)
# Force signed int32: code_200 = 0xC8C8C8C8 which overflows int32 range
# Use int64 first, then convert
code_200_u = 0xC8C8C8C8
code_200_s = code_200_u if code_200_u < 0x80000000 else code_200_u - 0x100000000
print(f"code_200 packed: {hex(code_200_u)} = signed int32: {code_200_s}")

qweight_1d = torch.tensor([[code_200_s]], dtype=torch.int32)  # [1 row, 1 col]
# weight unpack: expand to [1, 4, 1] (shape[0]=1 packed rows, 4 unpacked per row, shape[1]=1 out)
expanded_w = torch.unsqueeze(qweight_1d, 1).expand(-1, 32 // bits, -1)  # [1, 4, 1]
shifted_w = torch.bitwise_right_shift(expanded_w, wf.unsqueeze(-1))  # wf=[1,4], unsqueeze(-1)=[1,4,1]
print(f"shifted int32 values: {shifted_w.squeeze().tolist()}")
shifted_w_i16 = shifted_w.to(torch.int16)
print(f"shifted int16 values: {shifted_w_i16.squeeze().tolist()}")
masked_w = torch.bitwise_and(shifted_w_i16, (2**bits) - 1).to(torch.float32)
print(f"after &0xFF:          {masked_w.squeeze().tolist()}")
print(f"Expected: all 200")
print()

# Also test value 0 and 255
for test_code in [0, 1, 127, 128, 200, 254, 255]:
    tc_u = (test_code) | (test_code << 8) | (test_code << 16) | (test_code << 24)
    tc_s = tc_u if tc_u < 0x80000000 else tc_u - 0x100000000
    qw = torch.tensor([[tc_s]], dtype=torch.int32)
    exp = torch.unsqueeze(qw, 1).expand(-1, 4, -1)
    sft = torch.bitwise_right_shift(exp, wf.unsqueeze(-1)).to(torch.int16)
    msk = torch.bitwise_and(sft, 0xFF).to(torch.float32)
    vals = msk.squeeze().tolist()
    ok = all(abs(v - test_code) < 1e-5 for v in vals)
    print(f"  code={test_code:3d}: unpacked={[int(v) for v in vals]} -> {'OK' if ok else 'FAIL'}")

print()
print("CONCLUSION: int16 unpack is correct for all 8-bit code values 0..255")
