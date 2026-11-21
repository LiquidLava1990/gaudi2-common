import json, torch
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

MP = "/m"
idx = json.load(open(f"{MP}/model.safetensors.index.json"))["weight_map"]
qt = scalar_types.uint4

def load(name):
    f = idx[name]
    with safe_open(f"{MP}/{f}", framework="pt") as h:
        return h.get_tensor(name)

import sys
# pick one routed expert proj (layer 0 expert 0)
PROJ = sys.argv[1] if len(sys.argv) > 1 else "gate_proj"
pre = f"model.language_model.layers.0.mlp.experts.0.{PROJ}"
qw = load(pre + ".qweight")   # gptq int32 [in//8, out]
sc = load(pre + ".scales")    # [groups, out]
qz = load(pre + ".qzeros")    # [groups, out//8] int32
print("qweight", qw.shape, qw.dtype, "scales", sc.shape, sc.dtype, "qzeros", qz.shape, qz.dtype)

GS = 128
IN = qw.shape[0] * 8
OUT = qw.shape[1]
print("IN", IN, "OUT", OUT, "groups expected", IN // GS)

# ---------- TRUSTED dequant: autogptq LSB-first nibble unpack ----------
# gptq qweight int32 [in//8, out], each int32 packs 8 codes along the IN dim, LSB first
def trusted_unpack_codes(qw_i32):
    inp8, out = qw_i32.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = (qw_i32.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF   # [in//8, 8, out]
    codes = codes.reshape(inp8 * 8, out)                          # [in, out]
    return codes.to(torch.int64)

codes_t = trusted_unpack_codes(qw.to(torch.int32))   # [in, out]
g_idx = (torch.arange(IN) // GS)
s_t = sc.to(torch.float32)[g_idx]                    # [in, out]
# trusted symmetric zero = 8 (autoround sym, qzeros code 7 + 1)
w_trusted = (codes_t.to(torch.float32) - 8.0) * s_t   # [in, out] dequant weight, F.linear orient x@W

# verify qzeros all code 7
qz_codes = trusted_unpack_codes(qz.to(torch.int32).transpose(0,1)).transpose(0,1) if False else None
# unpack qzeros along out dim
zc = (qz.to(torch.int32).unsqueeze(2) >> torch.arange(0,32,4,dtype=torch.int32).view(1,1,8)) & 0xF
zc = zc.reshape(qz.shape[0], -1)
print("qzeros unique codes:", torch.unique(zc).tolist())

# ---------- PLUGIN path emulation ----------
# 1. loader: loaded_weight = qw.T.contiguous().view(uint8)  -> [out, in//2] uint8
w8 = qw.T.contiguous().view(torch.uint8)             # [out, in//2]  (this becomes w13_qweight[e])
print("plugin loaded uint8", w8.shape)

# 2. _repack_weight per expert:
w_i32 = w8.contiguous().view(torch.int32)            # [out, in//8]
vals = unpack_quantized_values_into_int32(
    w_i32.transpose(0,1).contiguous(), qt, packed_dim=0)   # [in, out]
packed = pack_quantized_values_into_int32(vals, qt, packed_dim=1)  # [in, out//8]
print("plugin repacked", packed.shape)

# 3. scales: loaded as [out, groups] (loader does loaded_weight.T for scales -> [out, groups])
# In create_weights w13_scales is [E, out, groups]. plugin transposes (1,2) -> [groups, out]
sc_loaded = sc.T.contiguous()    # [out, groups]  (what loader stores into w13_scales[e])
sc_plugin = sc_loaded.transpose(0,1).contiguous()   # [groups, out]  fed to set_weight_scale

# 4. zero point: _sym_zero -> [groups, out//8] all packed value 8
zero_val = 8
zeros = torch.full((sc_plugin.shape[0], sc_plugin.shape[1]), zero_val, dtype=torch.int32)
zp_packed = pack_quantized_values_into_int32(zeros, qt, packed_dim=1)  # [groups, out//8]

# 5. emulate convert_from_uint4(weight_packed[in,out//8], weight_scale[groups,out], zero_point[groups,out//8], g_idx=None)
# The HPU op: unpack weight_packed along packed dim (out), unpack zero_point along out,
# expand scale/zero by group index over the IN dim, dequant = (code - zero)*scale.
def emulate_convert_from_uint4(weight_packed, weight_scale, zero_point, group_size):
    # weight_packed [in, out//8] int32 -> codes [in, out]
    codes = unpack_quantized_values_into_int32(weight_packed, qt, packed_dim=1)  # [in, out]
    # zero_point [groups, out//8] -> [groups, out]
    zcodes = unpack_quantized_values_into_int32(zero_point, qt, packed_dim=1)    # [groups, out]
    inp = codes.shape[0]
    gidx = (torch.arange(inp) // group_size)
    z = zcodes[gidx].to(torch.float32)            # [in, out]
    s = weight_scale[gidx].to(torch.float32)      # [in, out]
    return (codes.to(torch.float32) - z) * s      # [in, out]

w_plugin = emulate_convert_from_uint4(packed, sc_plugin, zp_packed, GS)

print("w_trusted", w_trusted.shape, "w_plugin", w_plugin.shape)
diff = (w_plugin - w_trusted).abs()
print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())
print("num mismatched elems:", (diff > 1e-6).sum().item(), "of", diff.numel())
# find a nonzero region to prove it's not trivially all-zero matching
nz = (w_trusted.abs() > 1e-4).nonzero()
if len(nz):
    r, c = nz[0].tolist()
    print(f"nonzero at [{r},{c}] trusted={w_trusted[r,c].item()} plugin={w_plugin[r,c].item()}")
print("trusted absmax:", w_trusted.abs().max().item(), "codes range:", codes_t.min().item(), codes_t.max().item())
