"""Decisive pre/post-all-reduce test. Build the FULL all-rank MoE (all 8 selected experts
per token + shared) and compare hidden_out to:
  H1 (pre-reduce, rank3): 1*shared + 2.5*routed_localRank3
  H2 (post-reduce):       8*shared + 2.5*routed_ALLranks   (shared replicated x8, summed)
  H3 (post-reduce, shared reduced separately once): 1*shared + 2.5*routed_ALLranks
Whichever matches tells us where in the pipeline hidden_out sits and what's missing."""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size'];rsf=cfg['routed_scaling_factor']
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}',framework='pt',device='cpu') as f: return f.get_tensor(n)
def dq(p,gs=128,bits=4):
    qw=get(p+'.qweight');qz=get(p+'.qzeros');sc=get(p+'.scales')
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int32)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT);c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])
    rz=torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int32)
    rz=torch.bitwise_and(rz+1,(2**bits)-1).to(DT);z=rz.reshape(-1,rz.shape[1]*rz.shape[2])
    scf=sc.to(DT);g=(torch.arange(c.shape[0])//gs).long();return (c-z[g])*scf[g]
def cs(a,b):
    a=a.reshape(-1).to(DT);b=b.reshape(-1).to(DT);return (a@b/(a.norm()*b.norm()+1e-30)).item()
ref=torch.load('/opt/hpu-dsa-ext/layer3_moe_ref.pt',map_location='cpu',weights_only=False)
normed2=ref['normed2'].to(DT);topk_ids=ref['topk_ids'];topk_w=ref['topk_w'].to(DT)
hidden_out=ref['hidden_out'].to(DT);shared=ref['shared'].to(DT)
routed_all=ref['routed'].to(DT)  # all 8 experts/token (all ranks)
S=normed2.shape[0]
# routed restricted to rank-3 local experts [96,128)
MP=f'model.layers.{L}.mlp'
emin,emax=96,128;ec={}
def eo(e,x):
    if e not in ec: ec[e]=(dq(f'{MP}.experts.{e}.gate_proj'),dq(f'{MP}.experts.{e}.up_proj'),dq(f'{MP}.experts.{e}.down_proj'))
    g,u,dn=ec[e];return (F.silu(x@g)*(x@u))@dn
routed_r3=torch.zeros(S,H,dtype=DT)
for i in range(S):
    for k in range(topk_ids.shape[1]):
        e=int(topk_ids[i,k])
        if emin<=e<emax: routed_r3[i]+=topk_w[i,k]*eo(e,normed2[i])
cands={
 'H1 pre-reduce rank3: 1*shared+2.5*routed_r3':  shared+rsf*routed_r3,
 'H1b pre-reduce: 1*shared only':                shared,
 'H2 post-reduce: 8*shared+2.5*routed_all':      8*shared+rsf*routed_all,
 'H3 post-reduce: 1*shared+2.5*routed_all':      shared+rsf*routed_all,
}
for nm,t in cands.items():
    print(f'  {nm:46s} cos={[round(cs(t[i],hidden_out[i]),5) for i in range(S)]} ratio={[round((hidden_out[i].norm()/(t[i].norm()+1e-9)).item(),3) for i in range(S)]}')
