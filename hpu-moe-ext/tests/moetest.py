import torch, habana_frameworks.torch.core as htorch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk, unpack_quantized_values_into_int32 as up)
from vllm.scalar_type import scalar_types
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpWNA16
qt = scalar_types.uint4; dev='hpu'; torch.manual_seed(0)
E, HID, I, G = 2, 128, 64, 128   # experts, hidden(in), intermediate, group
T = 3                            # tokens

# random int4 expert weights + scales (per the loaded-uint8 layout)
def mk(out_, in_):
    q = torch.randint(0,16,(in_,out_),dtype=torch.int32)             # [in,out] values
    gptq = pk(q, qt, packed_dim=0); u8 = gptq.T.contiguous().view(torch.uint8)  # [out,in//2]
    scale = (torch.rand(out_, in_//G, dtype=torch.bfloat16)+0.1)     # [out, groups] (loaded layout)
    return q, u8, scale
# w13: out=2I, in=HID ; w2: out=HID, in=I
data = []
for e in range(E):
    q13,u13,s13 = mk(2*I, HID); q2,u2,s2 = mk(HID, I)
    data.append((q13,u13,s13,q2,u2,s2))

def repack(u8):
    wi = u8.contiguous().view(torch.int32)
    v = up(wi.transpose(0,1).contiguous(), qt, packed_dim=0)
    return pk(v, qt, packed_dim=1)
def symzero(groups,out): 
    return pk(torch.full((groups,out),8,dtype=torch.int32), qt, packed_dim=1)
def deqW(q, scale):  # reference dequant [in,out]
    sf = scale.T.float().repeat_interleave(G, dim=0)   # scale loaded [out,groups]->[groups,out]->[in,out]
    return sf*(q.float()-8.0)

op = VllmMixtureOfExpertsOpWNA16(E,0,E-1)
for e in range(E):
    q13,u13,s13,q2,u2,s2 = data[e]
    op.w13_list[e].set_weight_packed(repack(u13).to(dev))
    op.w2_list[e].set_weight_packed(repack(u2).to(dev))
    op.w13_list[e].set_weight_scale(s13.transpose(0,1).contiguous().to(dev))  # [groups,out]
    op.w2_list[e].set_weight_scale(s2.transpose(0,1).contiguous().to(dev))
    op.w13_list[e].set_zero_point(symzero(HID//G,2*I).to(dev))
    op.w2_list[e].set_zero_point(symzero(I//G,HID).to(dev))

x = torch.randn(T,HID,dtype=torch.bfloat16)
topk_ids = torch.zeros(T,1,dtype=torch.int64)        # all -> expert 0
topk_w = torch.ones(T,1,dtype=torch.bfloat16)
out = op(x.to(dev), topk_ids.to(dev), topk_w.to(dev), permuted_weights=False, activation='silu').cpu().float()
htorch.mark_step()

# reference: expert 0, gate=w13[:,:I]? need to find split. deqW gives [in,out]=[HID,2I]
W13 = deqW(data[0][0], data[0][2])   # [HID, 2I]
W2  = deqW(data[0][3], data[0][5])   # [I, HID]
xf = x.float()
def ref_with(splitA):
    g = xf @ W13[:, :I]; u = xf @ W13[:, I:]       # [T,I]
    if splitA=='gate_first': h = F.silu(g)*u
    else: h = F.silu(u)*g
    return h @ W2.T*0 + h @ W2  # W2 is [I,HID]? out = h[T,I] @ W2[I,HID]? W2 deq is [in=I,out=HID]
# W2 deq [I, HID] => out = h @ W2 -> [T,HID]
g = xf @ W13[:, :I]; u = xf @ W13[:, I:]
for nm,h in [('gate_first',F.silu(g)*u),('up_first',F.silu(u)*g)]:
    r = h @ W2
    d = (out-r).abs().max().item()
    print(f'{nm}: maxdiff {d:.4f} match {torch.allclose(out,r,atol=0.1)}')
print('out sample', out[0,:4].tolist())
