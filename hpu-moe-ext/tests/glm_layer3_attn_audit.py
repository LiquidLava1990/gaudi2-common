"""Audit whether the layer-3 ATTENTION (and hence normed2 / router input) is correct.
The MoE-input normed2 derives from residual_out (authoritative HPU) so the router input
IS correct regardless of my attn ref. But check: does the layer-3 attention ref match HPU
as well as layer 0 did (cos 0.99998)? A 0.987 attn cos would mean an UPSTREAM (attention)
error is corrupting the residual that feeds ALL subsequent layers -> flat router -> salad.
Also: is the router output really flat, or does normed2 give a peaked distribution?"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; eps=cfg['rms_norm_eps']
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
normed2=ref['normed2'].to(DT)
gate_w=get(f'model.layers.{L}.mlp.gate.weight').to(DT)
ebias=get(f'model.layers.{L}.mlp.gate.e_score_correction_bias').to(DT)
logits=normed2@gate_w.t()
scores=torch.sigmoid(logits)
biased=scores+ebias[None]
S=normed2.shape[0]
print('=== router score peakiness (biased, used for selection) ===')
for i in range(S):
    tv,ti=torch.topk(biased[i],12)
    print(f'  pos{i}: top12 biased={[round(x,4) for x in tv.tolist()]}')
    print(f'         sigmoid(min..max)={scores[i].min().item():.4f}..{scores[i].max().item():.4f} bias(min..max)={ebias.min().item():.3f}..{ebias.max().item():.3f}')
print()
print('  -> if top biased scores are all ~equal (flat), routing is near-random ⇒ the MoE-input')
print('     direction (normed2) is wrong ⇒ UPSTREAM (attention/residual) corruption, not MoE op.')
print()
print('=== logits raw stats (pre-sigmoid) ===')
for i in range(S):
    print(f'  pos{i}: logit min/mean/max = {logits[i].min().item():.3f}/{logits[i].mean().item():.3f}/{logits[i].max().item():.3f}  std={logits[i].std().item():.3f}')
