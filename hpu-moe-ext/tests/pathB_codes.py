#!/usr/bin/env python3
"""Pinpoint: do the CODES that the plugin path feeds convert_from_uint4 match
the trusted autogptq codes elementwise? Isolates repack ordering bug."""
import json, struct, os
import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

QT = scalar_types.uint4
MD = os.environ["MODEL_DIR"]
IDX = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]

def load(key):
    path = f"{MD}/{IDX[key]}"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n
        m = hdr[key]; s, e = m["data_offsets"]
        f.seek(base + s); raw = f.read(e - s)
    dt = {"I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}[m["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).clone().view(*m["shape"])

base = os.environ.get("PROJ", "model.language_model.layers.30.mlp.experts.100.gate_proj")
qw = load(base + ".qweight")   # [in//8, out] int32
IN = qw.shape[0]*8; OUT = qw.shape[1]
print("base", base, "IN", IN, "OUT", OUT)

# trusted codes [in, out]
sh = torch.arange(8, dtype=torch.int64)*4
codes_t = ((qw.to(torch.int64).unsqueeze(-1) >> sh) & 0xF).reshape(IN, OUT)

# plugin: loader -> uint8 [out,in//2]; repack -> int32 [in,out//8]; unpack codes [in,out]
w8 = qw.T.contiguous().view(torch.uint8)              # [out, in//2]
w_i32 = w8.contiguous().view(torch.int32)             # [out, in//8]
vals = unpack_quantized_values_into_int32(w_i32.transpose(0,1).contiguous(), QT, packed_dim=0)  # [in,out]
packed = pack_quantized_values_into_int32(vals, QT, packed_dim=1)   # [in,out//8]
codes_p = unpack_quantized_values_into_int32(packed, QT, packed_dim=1)  # [in,out]

print("codes_t", codes_t.shape, "codes_p", codes_p.shape)
mism = (codes_t != codes_p)
print("codes mismatch frac:", mism.float().mean().item())

# Also test the intermediate 'vals' directly against trusted (vals is [in,out] BEFORE repack)
print("vals vs trusted mismatch frac:", (vals != codes_t).float().mean().item())

# show a small corner
print("trusted[0,:8]:", codes_t[0,:8].tolist())
print("vals   [0,:8]:", vals[0,:8].tolist())
print("plugin [0,:8]:", codes_p[0,:8].tolist())
print("trusted[:8,0]:", codes_t[:8,0].tolist())
print("vals   [:8,0]:", vals[:8,0].tolist())
