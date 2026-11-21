#!/usr/bin/env python3
"""Trace plugin repack vs PROVEN trusted (movedim) codes, step by step."""
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
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n)); base=8+n
        m=hdr[key]; s,e=m["data_offsets"]; f.seek(base+s); raw=f.read(e-s)
    dt={"I32":torch.int32,"F16":torch.float16}[m["dtype"]]
    return torch.frombuffer(bytearray(raw),dtype=dt).clone().view(*m["shape"])

def trusted_lsb(qweight):  # PROVEN: [in//8,out]->[in,out]
    p=qweight.to(torch.int64); sh=torch.arange(8,dtype=torch.int64)*4
    moved=p.movedim(0,-1)
    out=(moved.unsqueeze(-1)>>sh)&0xF
    return out.reshape(*moved.shape[:-1],-1).movedim(-1,0).to(torch.int32)

base=os.environ.get("PROJ","model.language_model.layers.30.mlp.experts.100.gate_proj")
qw=load(base+".qweight"); IN=qw.shape[0]*8; OUT=qw.shape[1]
print("base",base,"IN",IN,"OUT",OUT)
codes_t = trusted_lsb(qw)   # [in,out] PROVEN

# direct CT unpack of checkpoint
codes_direct = unpack_quantized_values_into_int32(qw.contiguous(), QT, packed_dim=0)
print("direct CT unpack vs trusted mism:", (codes_direct!=codes_t).float().mean().item())

# ---- PLUGIN path exactly as in moe_wna16_hpu._repack_weight ----
# loader builds w13_qweight[e] = qw.T.contiguous().view(uint8)  -> [out, in//2]
qweight_u8 = qw.T.contiguous().view(torch.uint8)      # this is layer.w13_qweight.data[e]
w_i32 = qweight_u8.contiguous().view(torch.int32)     # [out, in//8]
vals = unpack_quantized_values_into_int32(w_i32.transpose(0,1).contiguous(), QT, packed_dim=0)  # [in,out]
packed = pack_quantized_values_into_int32(vals, QT, packed_dim=1)   # [in,out//8]
codes_p = unpack_quantized_values_into_int32(packed, QT, packed_dim=1)
print("plugin vals vs trusted mism:", (vals!=codes_t).float().mean().item())
print("plugin final codes vs trusted mism:", (codes_p!=codes_t).float().mean().item())

# Is w_i32 (=[out,in//8]) == qw.T ?  byte roundtrip should give qw.T
print("w_i32 == qw.T :", (w_i32 == qw.t()).all().item())
print("w_i32.transpose(0,1) == qw :", (w_i32.transpose(0,1)==qw).all().item())

# So vals = CT_unpack(qw, packed_dim=0). That should == codes_t (direct). Why differ?
vals2 = unpack_quantized_values_into_int32(qw.contiguous(), QT, packed_dim=0)
print("vals2(direct on qw) vs vals(via transpose) mism:", (vals2!=vals).float().mean().item())
print("trusted[0,:8]:",codes_t[0,:8].tolist())
print("vals   [0,:8]:",vals[0,:8].tolist())
print("direct [0,:8]:",codes_direct[0,:8].tolist())
