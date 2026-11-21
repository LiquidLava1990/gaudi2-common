import sys, torch, traceback
import habana_frameworks.torch.core as htorch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk, unpack_quantized_values_into_int32 as up)
from vllm.scalar_type import scalar_types
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpWNA16
qt=scalar_types.uint4; dev='hpu'; torch.manual_seed(0)
HID,I,G=128,64,128
def mk(out_,in_):
    q=torch.randint(0,16,(in_,out_),dtype=torch.int32)
    u8=pk(q,qt,packed_dim=0).T.contiguous().view(torch.uint8)
    s=(torch.rand(out_,in_//G,dtype=torch.bfloat16)+0.1)
    return u8,s
def repack(u8):
    wi=u8.contiguous().view(torch.int32); v=up(wi.transpose(0,1).contiguous(),qt,packed_dim=0)
    return pk(v,qt,packed_dim=1)
def sz(g,o): return pk(torch.full((g,o),8,dtype=torch.int32),qt,packed_dim=1)
try:
    u13,s13=mk(2*I,HID); u2,s2=mk(HID,I)
    op=VllmMixtureOfExpertsOpWNA16(1,0,0)
    op.w13_list[0].set_weight_packed(repack(u13).to(dev)); op.w2_list[0].set_weight_packed(repack(u2).to(dev))
    op.w13_list[0].set_weight_scale(s13.transpose(0,1).contiguous().to(dev)); op.w2_list[0].set_weight_scale(s2.transpose(0,1).contiguous().to(dev))
    op.w13_list[0].set_zero_point(sz(HID//G,2*I).to(dev)); op.w2_list[0].set_zero_point(sz(I//G,HID).to(dev))
    # reference from the op's OWN dequant weights
    W13=op.w13_list[0].get_dequant_weight().cpu().float()  # [HID,2I]
    W2=op.w2_list[0].get_dequant_weight().cpu().float()    # [I,HID]
    htorch.mark_step()
    print("W13",tuple(W13.shape),"W2",tuple(W2.shape),flush=True)
    x=torch.randn(2,HID,dtype=torch.bfloat16)
    out=op(x.to(dev),torch.zeros(2,1,dtype=torch.int64).to(dev),torch.ones(2,1,dtype=torch.bfloat16).to(dev),
           permuted_weights=False,activation='silu').cpu().float()
    htorch.mark_step()
    xf=x.float()
    g=xf@W13[:,:I]; u=xf@W13[:,I:]
    for nm,h in [('gate_first',F.silu(g)*u),('up_first',F.silu(u)*g)]:
        r=h@W2
        print(f"{nm}: maxdiff {(out-r).abs().max().item():.4f} match {torch.allclose(out,r,atol=0.05)}",flush=True)
    print("op_out[0,:3]",out[0,:3].tolist(),flush=True)
except Exception:
    traceback.print_exc()
