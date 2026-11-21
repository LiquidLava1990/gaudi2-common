"""Final decomposition: prove hidden_out == single shared expert with ZERO routed,
even for the local selected expert (104) on rank 3. Also test residual-direction:
does the MISSING routed term explain the salad? (the full-model residual would be
wrong-direction once routed is dropped across all MoE layers)."""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; eps=cfg['rms_norm_eps']; rsf=cfg['routed_scaling_factor']
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

ref=torch.load('/opt/hpu-dsa-ext/layer3_moe_ref.pt',map_location='cpu',weights_only=False)
normed2=ref['normed2'].to(DT); topk_ids=ref['topk_ids']; topk_w=ref['topk_w'].to(DT)
hidden_out=ref['hidden_out'].to(DT); shared=ref['shared'].to(DT); routed=ref['routed'].to(DT)
S=normed2.shape[0]
# per-pos residual: subtract single shared from hidden_out -> what's left? (should be ~0 if routed dropped)
resid = hidden_out - shared
print('=== hidden_out - shared (the leftover; should be ~0 if routed fully dropped) ===')
for i in range(S):
    print(f'  pos{i}: |hidden_out-shared|={resid[i].norm().item():9.4f}  |2.5*routed_full|={ (rsf*routed[i]).norm().item():9.3f}  cos(leftover,2.5routed)={cs(resid[i], rsf*routed[i]):.4f}')
print()
# is shared single or 8x? compare hidden_out to k*shared for k in {1,8}
for k in [1.0,8.0]:
    print(f'  cos(hidden_out, {k}*shared)=', [round(cs(hidden_out[i], k*shared[i]),5) for i in range(S)],
          'ratio|hidden|/|{k}shared|=', [round((hidden_out[i].norm()/(k*shared[i].norm())).item(),3) for i in range(S)])
