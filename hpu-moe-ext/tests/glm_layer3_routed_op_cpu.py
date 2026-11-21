"""Reproduce MoeWNA16HPUMethod._repack_weight + the convert_from_uint4 dequant logic
on CPU for the routed int4 experts, and compare to the reference dequant. Also reproduce
how upstream MoeWNA16Method loads w13_qweight as uint8 [E,out,in//2], to feed _repack_weight
exactly as the live model does. PURE CPU."""
import json, torch, sys
sys.path.insert(0,'/opt/hpu-moe-ext')
import torch.nn.functional as F
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}',framework='pt',device='cpu') as f: return f.get_tensor(n)

# reference int4 dequant (the proven path) -> [in,out] fp64
def ref_dq(p,gs=128,bits=4):
    qw=get(p+'.qweight');qz=get(p+'.qzeros');sc=get(p+'.scales')
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int32)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT);c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])  # [in,out]
    rz=torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int32)
    rz=torch.bitwise_and(rz+1,(2**bits)-1).to(DT);z=rz.reshape(-1,rz.shape[1]*rz.shape[2])
    scf=sc.to(DT);g=(torch.arange(c.shape[0])//gs).long();return (c-z[g])*scf[g]

# ------ Replicate upstream MoeWNA16 weight LOADING into uint8 [out, in//2] for one expert ------
# Upstream stores per-expert w13_qweight as uint8 [E, out, in//pack] where the GPTQ int32
# qweight [in//8, out] is reinterpreted. We must reproduce EXACTLY what layer.w13_qweight.data[e]
# holds at process_weights_after_loading time. The moe_wna16 loader for gptq does:
#   loaded int32 qweight [in//8, out] -> it stores as uint8 by viewing int32 as 4 bytes.
# _repack_weight expects: qweight_u8[e] of shape [out, in//2] (uint8), .view(int32)->[out,in//8].
# So the stored layout is the int32 qweight TRANSPOSED to [out, in//8] then byte-viewed [out,in//2].
def make_uint8_from_ckpt(p):
    qw=get(p+'.qweight')          # int32 [in//8, out]  (gptq packed, 8 vals/int32 along in)
    w_oi = qw.transpose(0,1).contiguous()      # [out, in//8] int32
    u8 = w_oi.view(torch.uint8)                 # [out, in//2] uint8
    return u8

QT=scalar_types.uint4
def repack_one(u8):
    # mirror MoeWNA16HPUMethod._repack_weight for a single expert
    w_i32=u8.contiguous().view(torch.int32)              # [out, in//8]
    vals=unpack_quantized_values_into_int32(w_i32.transpose(0,1).contiguous(), QT, packed_dim=0)  # [in,out]
    return pack_quantized_values_into_int32(vals, QT, packed_dim=1)   # [in, out//8]

# convert_from_uint4 reference (the HPU intrinsic) replicated on CPU:
# dequant w = (code - zero) * scale, code unpacked from int32 packed along out dim,
# zero = uint4 bias = 8 (sym). scales [groups,out].
def cpu_convert_from_uint4(packed_int32_in_outdiv8, scales_go, zero_val=8, gs=128, bits=4):
    # packed [in, out//8] int32; unpack along dim=1 -> [in,out]
    codes = unpack_quantized_values_into_int32(packed_int32_in_outdiv8, QT, packed_dim=1).to(DT)  # [in,out]
    g=(torch.arange(codes.shape[0])//gs).long()
    return (codes - zero_val) * scales_go[g]

def cs(a,b):
    a=a.reshape(-1).to(DT);b=b.reshape(-1).to(DT);return (a@b/(a.norm()*b.norm()+1e-30)).item()

eid=104
MP=f'model.layers.{L}.mlp'
for proj,is13 in [('gate_proj',True),('up_proj',True),('down_proj',False)]:
    p=f'{MP}.experts.{eid}.{proj}'
    ref=ref_dq(p)                                  # [in,out] fp64 (true)
    u8=make_uint8_from_ckpt(p)
    packed=repack_one(u8)                          # [in, out//8] int32
    sc=get(p+'.scales').to(DT)                      # [groups,out]
    deq=cpu_convert_from_uint4(packed, sc)          # [in,out]
    md=(ref-deq).abs().max().item()
    print(f'expert{eid}.{proj:9s} ref{tuple(ref.shape)} deq{tuple(deq.shape)} maxdiff={md:.3e} cos={cs(ref,deq):.6f} |ref|={ref.norm():.2f} |deq|={deq.norm():.2f}')
