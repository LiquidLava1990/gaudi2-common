import torch, habana_frameworks.torch as ht  # noqa
from hpu_dsa_ext.sparse_indexer_hpu import lightning_indexer_topk as IDX
from hpu_dsa_ext.sparse_mla_consume import sparse_mla_gather_attention as GA
torch.manual_seed(0)
T,H,D,S,K,Dv = 48,4,16,48,8,12
qi=torch.randn(T,H,D); ki=torch.randn(S,D); wi=torch.randn(T,H); ap=torch.arange(T).int()
qc=torch.randn(T,H,D); kc=torch.randn(S,H,D); vc=torch.randn(S,H,Dv); sc=D**-0.5
th=IDX(qi.to('hpu').bfloat16(),ki.to('hpu').bfloat16(),wi.to('hpu').bfloat16(),K,ap.to('hpu'))
tc=IDX(qi,ki,wi,K,ap)
out_h=GA(qc.to('hpu').bfloat16(),kc.to('hpu').bfloat16(),vc.to('hpu').bfloat16(),th,ap.to('hpu'),sc).cpu().float()
out_c_same=GA(qc.bfloat16().float(),kc.bfloat16().float(),vc.bfloat16().float(),th.cpu(),ap,sc)  # CPU, SAME topk, bf16-cast
out_c_own =GA(qc,kc,vc,tc,ap,sc)                                                                 # CPU, its OWN topk (test d)
print("SAME-topk HPU-vs-CPU (true consume precision):", round((out_h-out_c_same).abs().max().item(),5))
print("DIFF-topk HPU-vs-CPU (churn, what test d measured):", round((out_h-out_c_own).abs().max().item(),5))
tot=sm=0
for t in range(T):
  a=set(th.cpu()[t][th.cpu()[t]>=0].tolist()); b=set(tc[t][tc[t]>=0].tolist()); tot+=len(b); sm+=len(a&b)
print("topk recall HPU-vs-CPU:", round(sm/max(1,tot),4))
