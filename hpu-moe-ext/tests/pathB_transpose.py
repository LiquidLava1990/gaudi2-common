#!/usr/bin/env python3
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
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8+n
        m = hdr[key]; s,e = m["data_offsets"]; f.seek(base+s); raw=f.read(e-s)
    dt = {"I32":torch.int32,"F16":torch.float16}[m["dtype"]]
    return torch.frombuffer(bytearray(raw),dtype=dt).clone().view(*m["shape"])

base = os.environ.get("PROJ","model.language_model.layers.30.mlp.experts.100.gate_proj")
qw = load(base+".qweight"); IN=qw.shape[0]*8; OUT=qw.shape[1]
sh = torch.arange(8,dtype=torch.int64)*4
codes_t = ((qw.to(torch.int64).unsqueeze(-1)>>sh)&0xF).reshape(IN,OUT)   # [in,out]

w8 = qw.T.contiguous().view(torch.uint8)
w_i32 = w8.contiguous().view(torch.int32)             # [out, in//8]
vals = unpack_quantized_values_into_int32(w_i32.transpose(0,1).contiguous(), QT, packed_dim=0)  # claims [in,out]
print("vals shape", vals.shape, "codes_t shape", codes_t.shape)
print("vals == codes_t      mism:", (vals!=codes_t).float().mean().item())
print("vals == codes_t.T    mism:", (vals!=codes_t.t()).float().mean().item() if vals.shape==codes_t.t().shape else "shape-mismatch")

# CORRECT repack: codes are [in,out]; we want packed [in, out//8] for the op.
# Trusted codes are [in,out]. Just pack along out:
packed_correct = pack_quantized_values_into_int32(codes_t.to(torch.int32), QT, packed_dim=1)  # [in,out//8]
codes_back = unpack_quantized_values_into_int32(packed_correct, QT, packed_dim=1)
print("correct roundtrip mism:", (codes_back!=codes_t).float().mean().item())

# What does the op actually want? Check: convert_from_uint4 unpacks packed along packed dim.
# weight_packed [in, out//8] -> codes [in,out] -> dequant [in,out]. matmul x[tok,in]@W[in,out].
# So we need packed[in,out//8] whose unpack == codes_t. packed_correct does that.
