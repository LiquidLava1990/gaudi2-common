"""Quantify router selection sensitivity: perturb normed2 by ~bf16-level and ~10% noise,
recompute top-8, measure overlap with the unperturbed picks. If overlap is low, the
'rank-3 routed=0' finding is a FALSE LEAD (live router picks different experts than my
fp64 ref), and the real bug is upstream (the ~10%/layer attention error feeding a
hypersensitive bias-dominated router)."""
import json, torch
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}',framework='pt',device='cpu') as f: return f.get_tensor(n)
ref=torch.load('/opt/hpu-dsa-ext/layer3_moe_ref.pt',map_location='cpu',weights_only=False)
normed2=ref['normed2'].to(DT)
gate_w=get(f'model.layers.{L}.mlp.gate.weight').to(DT)
ebias=get(f'model.layers.{L}.mlp.gate.e_score_correction_bias').to(DT)
def pick(x):
    s=torch.sigmoid(x@gate_w.t()); b=s+ebias[None]
    return torch.topk(b,8,dim=-1).indices
base=pick(normed2)
S=normed2.shape[0]
torch.manual_seed(0)
for lvl,nm in [(1e-3,'0.1% (bf16-ish)'),(1e-2,'1%'),(5e-2,'5%'),(1e-1,'10% (attn err)')]:
    ov=[]
    for t in range(20):
        pert=normed2*(1+lvl*torch.randn_like(normed2))
        p=pick(pert)
        for i in range(S):
            a=set(base[i].tolist()); bb=set(p[i].tolist()); ov.append(len(a&bb)/8.0)
    import statistics
    print(f'  perturb {nm:18s}: mean top-8 overlap with ref = {statistics.mean(ov)*100:.1f}%')
print()
print('If even 0.1% perturbation drops overlap well below 100%, the live (bf16, slightly')
print('different attn) model routes to DIFFERENT experts than this fp64 ref -> a single-rank')
print('"routed=0" dump CANNOT prove the routed op is broken. The bias~34 saturation is the')
print('mechanism: sigmoid scores (0..1) are dwarfed by bias spread, so selection is unstable.')
