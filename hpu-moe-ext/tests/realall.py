import json, glob, torch
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk, unpack_quantized_values_into_int32 as upk)
from vllm.scalar_type import scalar_types
qt=scalar_types.uint4
M='/mnt/fastmodels/Qwen3.5-35B-A3B-GPTQ-Int4'
wm=json.load(open(glob.glob(M+'/*.index.json')[0]))['weight_map']
def load(n):
    with safe_open(M+'/'+wm[n],'pt') as s: return s.get_tensor(n)
def upk_d0(w):
    n8,m=w.shape; o=torch.empty(n8*8,m,dtype=torch.int32)
    for k in range(8): o[k::8]=(w>>(4*k))&0xF
    return o
def upk_d1(w):
    n,m8=w.shape; o=torch.empty(n,m8*8,dtype=torch.int32)
    for k in range(8): o[:,k::8]=(w>>(4*k))&0xF
    return o
for proj in ['gate_proj','up_proj','down_proj']:
    p=f'model.language_model.layers.0.mlp.experts.0.{proj}'
    qw=load(p+'.qweight').to(torch.int32); sc=load(p+'.scales').float(); qz=load(p+'.qzeros').to(torch.int32)
    IN=qw.shape[0]*8; OUT=qw.shape[1]; G=IN//sc.shape[0]
    q=upk_d0(qw); z=upk_d1(qz)
    zf=(z+1).float().repeat_interleave(G,0); scf=sc.repeat_interleave(G,0)
    Wref=scf*(q.float()-zf)
    # my path
    u8=qw.T.contiguous().view(torch.uint8); wi=u8.contiguous().view(torch.int32)
    vals=upk(wi.transpose(0,1).contiguous(),qt,packed_dim=0)
    Wmine=scf*(vals.float()-9.0)
    print(f'{proj}: IN{IN} OUT{OUT} G{G} z+1uniq{torch.unique(z+1).tolist()[:3]} '
          f'unpack_ok={torch.equal(vals,q)} Wdiff_z9={(Wmine-Wref).abs().max().item():.5f}')
