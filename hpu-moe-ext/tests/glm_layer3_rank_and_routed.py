"""Identify the dump's TP/EP rank (via o_proj input-shard match), and decompose
hidden_out vs (shared, routed-of-local-experts) to see if hidden_out is pre- or
post-EP-combine, and whether the routed op is dropped."""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; eps=cfg['rms_norm_eps']; rsf=cfg['routed_scaling_factor']
TP=8
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}',framework='pt',device='cpu') as f: return f.get_tensor(n)
def dq(p,gs=128,bits=8):
    qw=get(p+'.qweight');qz=get(p+'.qzeros');sc=get(p+'.scales')
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int32)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT);c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])
    rz=torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int32)
    rz=torch.bitwise_and(rz+1,(2**bits)-1).to(DT);z=rz.reshape(-1,rz.shape[1]*rz.shape[2])
    scf=sc.to(DT);g=(torch.arange(c.shape[0])//gs).long();return (c-z[g])*scf[g]
def cs(a,b):
    a=a.reshape(-1).to(DT);b=b.reshape(-1).to(DT);return (a@b/(a.norm()*b.norm()+1e-30)).item()

d=torch.load('/opt/hpu-dsa-ext/layer3_dump.pt',map_location='cpu',weights_only=False)
ref=torch.load('/opt/hpu-dsa-ext/layer3_moe_ref.pt',map_location='cpu',weights_only=False)
o_rank=d['o_proj_weight'].to(DT)   # [6144,2048]  this rank's o_proj (out,in_shard)
# full o_proj dequant [16384,6144] -> nn.Linear [6144,16384]; rank r owns input cols [r*2048:(r+1)*2048]
w_o_full=dq(f'model.layers.{L}.self_attn.o_proj').t()  # [6144,16384]
in_per=16384//TP
best=-1;bc=-2
for r in range(TP):
    sl=w_o_full[:, r*in_per:(r+1)*in_per]
    c=cs(o_rank, sl)
    if c>bc: bc=c; best=r
    print(f'  o_proj rank {r}: cos={c:.6f}')
print(f'=> dump is TP/EP rank {best} (cos {bc:.6f})')
print()

normed2=ref['normed2'].to(DT); topk_ids=ref['topk_ids']; topk_w=ref['topk_w'].to(DT)
hidden_out=ref['hidden_out'].to(DT)
shared=ref['shared'].to(DT)
S=normed2.shape[0]
# experts local to this rank under EP (ep_rank == tp rank typically when ep_size==world)
ep_rank=best; n_local=256//TP
emin=ep_rank*n_local; emax=emin+n_local
MP=f'model.layers.{L}.mlp'
# routed restricted to LOCAL experts only (what this rank's op would compute, pre-combine)
routed_local=torch.zeros(S,H,dtype=DT)
need_local=set()
for i in range(S):
    for k in range(topk_ids.shape[1]):
        e=int(topk_ids[i,k])
        if emin<=e<emax: need_local.add(e)
cache={e:(dq(f'{MP}.experts.{e}.gate_proj',bits=4),dq(f'{MP}.experts.{e}.up_proj',bits=4),dq(f'{MP}.experts.{e}.down_proj',bits=4)) for e in need_local}
for i in range(S):
    acc=torch.zeros(H,dtype=DT)
    for k in range(topk_ids.shape[1]):
        e=int(topk_ids[i,k])
        if emin<=e<emax:
            g,u,dn=cache[e]; acc=acc+topk_w[i,k]*((F.silu(normed2[i]@g)*(normed2[i]@u))@dn)
    routed_local[i]=acc
print(f'=== rank {ep_rank} owns experts [{emin},{emax}); local routed present this rank ===')
print('  local experts hit:', sorted(need_local))
print('  |routed_local*2.5|', [round((rsf*routed_local[i]).norm().item(),3) for i in range(S)])
print()
# Hypotheses for hidden_out:
cands={
 'shared only':                              shared,
 'shared + 2.5*routed_LOCAL (pre-EP-combine)': shared + rsf*routed_local,
 '2.5*routed_LOCAL only':                    rsf*routed_local,
}
print('=== hidden_out decomposition ===')
for nm,t in cands.items():
    print(f'  {nm:42s} cos={[round(cs(t[i],hidden_out[i]),5) for i in range(S)]} ratio={[round((hidden_out[i].norm()/(t[i].norm()+1e-9)).item(),3) for i in range(S)]}')
