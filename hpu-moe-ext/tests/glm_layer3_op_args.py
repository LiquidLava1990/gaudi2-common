"""Sanity: does the HPU mixture_of_experts op's expected expert-id windowing match
what apply_monolithic passes? The op receives GLOBAL topk_ids (0-255) + experts_min/max.
Reproduce on CPU the op's per-token expert selection to confirm token->local-weight mapping,
and check the routed output is NONZERO for the tokens that route to local expert 104."""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; rsf=cfg['routed_scaling_factor']
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

ref=torch.load('/opt/hpu-dsa-ext/layer3_moe_ref.pt',map_location='cpu',weights_only=False)
normed2=ref['normed2'].to(DT); topk_ids=ref['topk_ids']; topk_w=ref['topk_w'].to(DT)
S=normed2.shape[0]
emin,emax=96,128  # rank 3 local window [experts_min, experts_max]=(96,127)
# Simulate the op: for each token, for each of its top-8 (global id, weight), if
# emin<=id<emax use local weight[id-emin], accumulate weight*expert(x). This is the
# op's contract. Verify nonzero for pos with local expert.
MP=f'model.layers.{L}.mlp'
ecache={}
def eo(eid,x):
    if eid not in ecache:
        ecache[eid]=(dq(f'{MP}.experts.{eid}.gate_proj'),dq(f'{MP}.experts.{eid}.up_proj'),dq(f'{MP}.experts.{eid}.down_proj'))
    g,u,d=ecache[eid]
    return (F.silu(x@g)*(x@u))@d
op_out=torch.zeros(S,H,dtype=DT)
for i in range(S):
    for k in range(topk_ids.shape[1]):
        gid=int(topk_ids[i,k])
        if emin<=gid<emax:
            op_out[i]+=topk_w[i,k]*eo(gid,normed2[i])
print('=== simulated HPU-op routed output (rank-3 window [96,127]) ===')
for i in range(S):
    locals_=[int(topk_ids[i,k]) for k in range(topk_ids.shape[1]) if emin<=int(topk_ids[i,k])<emax]
    print(f'  pos{i}: local experts={locals_}  |op_out|={op_out[i].norm().item():9.3f}  |2.5*op|={ (rsf*op_out[i]).norm().item():9.3f}')
print()
print('CONCLUSION: pos2/3/4 should yield NONZERO routed (expert 104 local). The LIVE')
print('hidden_out shows ~0 routed -> the on-device op is returning zero despite correct')
print('weights + correct global topk_ids + correct [96,127] window.')
