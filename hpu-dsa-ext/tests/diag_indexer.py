import torch, habana_frameworks.torch as ht  # noqa
from hpu_dsa_ext.sparse_indexer_hpu import lightning_indexer_topk
torch.manual_seed(1)
T,H,D,S,topk=8,4,16,12,5
q=torch.randn(T,H,D); k=torch.randn(S,D); w=torch.randn(T,H)
ap=torch.clamp(torch.arange(T)+4,max=S-1).to(torch.int64)
ic=lightning_indexer_topk(q,k,w,topk,ap)
ih=lightning_indexer_topk(q.to('hpu'),k.to('hpu'),w.to('hpu'),topk,ap.to('hpu')).cpu()
tot=same=0
for t in range(T):
  sc=set(ic[t][ic[t]>=0].tolist()); sh=set(ih[t][ih[t]>=0].tolist())
  tot+=len(sc|sh); same+=len(sc&sh)
print("indexer HPU-vs-CPU overall jaccard:", round(same/max(1,tot),4))
# also: does the HPU bf16 scorer match a CPU-bf16 scorer (isolating bf16 vs the device)?
icb=lightning_indexer_topk(q.bfloat16().float(),k.bfloat16().float(),w.bfloat16().float(),topk,ap)
tb=sb=0
for t in range(T):
  sc=set(icb[t][icb[t]>=0].tolist()); sh=set(ih[t][ih[t]>=0].tolist())
  tb+=len(sc|sh); sb+=len(sc&sh)
print("HPU vs CPU-bf16-cast jaccard:", round(sb/max(1,tb),4))
