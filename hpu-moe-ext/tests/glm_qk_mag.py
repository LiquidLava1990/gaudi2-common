"""Compare q/k vector magnitudes at layer 0 vs layer 3 to see if attention saturation
(huge logits) is a real bug. For MLA, qk_head=256 dims; logit = q.k. If |q|,|k| ~ O(100),
q.k ~ O(100*100*sqrt-ish) can be ~600 after /16 scale. Check if this is normal (layer 0
attn was cos 0.99998 so its saturation, if any, was CORRECT)."""
import json, torch
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'
cfg=json.load(open(f'{CKPT}/config.json'))
n_heads=cfg['num_attention_heads'];kv_lora=cfg['kv_lora_rank']
qk_nope=cfg['qk_nope_head_dim'];qk_rope=cfg['qk_rope_head_dim'];qk_head=cfg['qk_head_dim']
v_head=cfg['v_head_dim'];eps=cfg['rms_norm_eps'];rope_theta=cfg['rope_parameters']['rope_theta']
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
def rms(x,w): return (w.to(DT)+1.0)*(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps))

for L,dumpf in [(0,'/opt/hpu-dsa-ext/layer0_dump.pt'),(3,'/opt/hpu-dsa-ext/layer3_dump.pt')]:
    d=torch.load(dumpf,map_location='cpu',weights_only=False)
    hin=d['hidden_in'].to(DT)
    rin=d.get('residual_in')
    if rin is not None: rin=rin[0].to(DT); R1=hin+rin
    else: R1=hin  # layer0 residual_in may be None
    if R1.dim()==3: R1=R1[0]
    S=5
    P=f'model.layers.{L}.self_attn'
    w_q_a=dq(f'{P}.q_a_proj');w_q_b=dq(f'{P}.q_b_proj')
    w_kv_a=dq(f'{P}.kv_a_proj_with_mqa');w_kv_b=dq(f'{P}.kv_b_proj')
    ln_in=get(f'model.layers.{L}.input_layernorm.weight')
    q_a_ln=get(f'{P}.q_a_layernorm.weight');kv_a_ln=get(f'{P}.kv_a_layernorm.weight')
    normed1=rms(R1[:S][None],ln_in)
    q=(rms(normed1@w_q_a,q_a_ln)@w_q_b).view(1,S,n_heads,qk_head)
    comp=normed1@w_kv_a;kc,kpe=comp.split([kv_lora,qk_rope],-1);kc=rms(kc,kv_a_ln)
    kv=(kc@w_kv_b).view(1,S,n_heads,qk_nope+v_head);kno,vst=kv.split([qk_nope,v_head],-1)
    qn=q[...,:qk_nope]
    print(f'--- layer {L} ---')
    print(f'  |normed1|/pos:', [round(normed1[0,i].norm().item(),2) for i in range(S)])
    print(f'  |q full|/head(pos4,h0..3):', [round(q[0,4,h,:].norm().item(),1) for h in range(4)])
    print(f'  |q_nope|/head(pos4,h0):', round(qn[0,4,0,:].norm().item(),1), ' |k_nope|/head:', round(kno[0,4,0,:].norm().item(),1))
    print(f'  q_nope.k_nope (pos4,h0 self):', round((qn[0,4,0,:]@kno[0,4,0,:]).item(),1), ' (pre-scale; /16 = scaled logit)')
    print(f'  |v|/head(pos4,h0):', round(vst[0,4,0,:].norm().item(),2))
