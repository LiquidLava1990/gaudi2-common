import json, glob, torch
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk, unpack_quantized_values_into_int32 as upk)
from vllm.scalar_type import scalar_types
qt = scalar_types.uint4
M='/mnt/fastmodels/Qwen3.5-35B-A3B-GPTQ-Int4'
wm=json.load(open(glob.glob(M+'/*.index.json')[0]))['weight_map']
pre='model.language_model.layers.0.mlp.experts.0.gate_proj'
def load(name):
    f=wm[name]; 
    with safe_open(M+'/'+f,'pt') as s: return s.get_tensor(name)
qw=load(pre+'.qweight'); sc=load(pre+'.scales'); qz=load(pre+'.qzeros'); gi=load(pre+'.g_idx')
print('qweight',tuple(qw.shape),qw.dtype,'scales',tuple(sc.shape),sc.dtype,'qzeros',tuple(qz.shape),qz.dtype)
IN = qw.shape[0]*8; OUT = qw.shape[1]; G = IN//sc.shape[0]
print('IN',IN,'OUT',OUT,'group',G,'g_idx uniq', torch.unique(gi)[:5].tolist(), 'len', gi.shape)

# reference autogptq unpack: q[i,o] = (qweight[i//8,o] >> 4*(i%8)) & 0xF
def unpack_gptq(w):  # int32 [n//8, m] -> [n, m] values
    n8,m=w.shape; out=torch.empty(n8*8,m,dtype=torch.int32)
    for k in range(8): out[k::8]=(w>>(4*k))&0xF
    return out
q_ref = unpack_gptq(qw.to(torch.int32))            # [IN, OUT]
z_ref = unpack_gptq(qz.to(torch.int32))            # [IN//G ...]? qzeros [IN//G, OUT//8]
# qzeros: [num_groups, OUT//8] int32 -> unpack along dim1 (out) -> [num_groups, OUT]
def unpack_gptq_dim1(w):
    n,m8=w.shape; out=torch.empty(n,m8*8,dtype=torch.int32)
    for k in range(8): out[:,k::8]=(w>>(4*k))&0xF
    return out
z = unpack_gptq_dim1(qz.to(torch.int32))           # [groups, OUT]
print('qzeros unpacked uniq values:', torch.unique(z).tolist())
print('  => real gptq zero (z+1) uniq:', torch.unique(z+1).tolist())
print('my assumed zero: 8')

# reference W: scales[i//G,o]*(q - (z+1))
zf = (z+1).float().repeat_interleave(G,0)          # [IN,OUT]
scf = sc.float().repeat_interleave(G,0)            # scales [groups,OUT]->[IN,OUT]
W_ref = scf*(q_ref.float()-zf)

# MY path: emulate loader uint8, repack, dequant with zero=8
u8 = qw.to(torch.int32).T.contiguous().view(torch.uint8)   # [OUT, IN//2]
wi = u8.contiguous().view(torch.int32)                     # [OUT, IN//8]
vals = upk(wi.transpose(0,1).contiguous(), qt, packed_dim=0)  # [IN, OUT]
print('my unpack == gptq ref q:', torch.equal(vals, q_ref))
W_mine_zero8 = scf*(vals.float()-8.0)
W_mine_realz = scf*(vals.float()-zf)
print('W diff (mine zero=8 vs ref):', (W_mine_zero8-W_ref).abs().max().item())
print('W diff (mine real-z vs ref):', (W_mine_realz-W_ref).abs().max().item())
