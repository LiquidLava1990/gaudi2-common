import torch, habana_frameworks.torch as ht  # noqa
from hpu_dsa_ext import sparse_mla_consume as C
torch.manual_seed(0)
T,H,Dh,Dv,S = 4,2,8,8,6
qc=torch.randn(T,H,Dh); kc=torch.randn(S,H,Dh); vc=torch.randn(S,H,Dv)
tc=torch.tensor([[0,2,-1],[1,3,4],[0,-1,-1],[2,5,1]],dtype=torch.int32)
ac=torch.tensor([2,4,0,5],dtype=torch.int64)
o_cpu=C.sparse_mla_gather_attention(qc,kc,vc,tc,ac,0.5)
d='hpu'
o_hpu=C.sparse_mla_gather_attention(qc.to(d),kc.to(d),vc.to(d),tc.to(d),ac.to(d),0.5).cpu()
print("OUT diff:", (o_cpu-o_hpu).abs().max().item())
km_c=C._build_keep_mask(tc,ac,S); km_h=C._build_keep_mask(tc.to(d),ac.to(d),S).cpu()
print("keep_mask diff:", (km_c.int()-km_h.int()).abs().max().item())
gi_c,va_c=C._compact_gather_indices(km_c,3); gi_h,va_h=C._compact_gather_indices(km_c.to(d),3)
print("gather_idx diff:", (gi_c-gi_h.cpu()).abs().max().item(), "valid diff:", (va_c.int()-va_h.cpu().int()).abs().max().item())
x=torch.tensor([[1.0,2.0,float('-inf')]])
print("CPU softmax(-inf):", torch.softmax(x,-1).tolist())
print("HPU softmax(-inf):", torch.softmax(x.to(d),-1).cpu().tolist())
print("HPU nan_to_num(nan):", torch.nan_to_num(torch.tensor([float('nan'),1.0]).to(d)).cpu().tolist())
