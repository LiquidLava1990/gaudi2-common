"""Is the layer-3 attention ref-vs-HPU gap (cos 0.987) a real bug or accumulated bf16?
Decompose: (a) attention on the EXACT HPU normed-input vs HPU attn output, isolating the
attention op alone (no residual contamination). We have R1=hin+rin (authoritative running
residual into layer 3), normed1=input_layernorm(R1). Compare ref attn(normed1) to HPU attn
(=residual_out-R1) per position. Layer 0 gave 0.99998; if layer 3 gives 0.987, attention
degrades with depth -> an upstream per-layer bug accumulates.

Also test: is the gap explained by bf16 rounding of the input? Compare |ref-hpu| structure."""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L=3
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; n_heads=cfg['num_attention_heads']; kv_lora=cfg['kv_lora_rank']
qk_nope=cfg['qk_nope_head_dim'];qk_rope=cfg['qk_rope_head_dim'];qk_head=cfg['qk_head_dim']
v_head=cfg['v_head_dim'];eps=cfg['rms_norm_eps'];rope_theta=cfg['rope_parameters']['rope_theta']
scaling=qk_head**-0.5
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
def rms(x,w): return (w.to(DT)+1.0)*(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps))
def rh(x): return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),-1)
def freqs(pos,dim):
    inv=1.0/(rope_theta**(torch.arange(0,dim,2,dtype=DT)/dim));return torch.einsum('s,d->sd',pos,inv)
def rope(x,f):
    e=torch.cat((f,f),-1);return x*e.cos()[None,None]+rh(x)*e.sin()[None,None]

d=torch.load('/opt/hpu-dsa-ext/layer3_dump.pt',map_location='cpu',weights_only=False)
positions=d['positions'][0].to(DT); hidden_in=d['hidden_in'].to(DT)
residual_in=d['residual_in'][0].to(DT); residual_out=d['residual_out'][0].to(DT)
R1=hidden_in+residual_in; hpu_attn=residual_out-R1
S=5
P=f'model.layers.{L}.self_attn'
w_q_a=dq(f'{P}.q_a_proj');w_q_b=dq(f'{P}.q_b_proj')
w_kv_a=dq(f'{P}.kv_a_proj_with_mqa');w_kv_b=dq(f'{P}.kv_b_proj');w_o=dq(f'{P}.o_proj')
ln_in=get(f'model.layers.{L}.input_layernorm.weight')
q_a_ln=get(f'{P}.q_a_layernorm.weight');kv_a_ln=get(f'{P}.kv_a_layernorm.weight')
normed1=rms(R1[:S][None],ln_in)
f=freqs(positions[:S],qk_rope)
q=(rms(normed1@w_q_a,q_a_ln)@w_q_b).view(1,S,n_heads,qk_head).transpose(1,2)
qn,qp=q.split([qk_nope,qk_rope],-1)
comp=normed1@w_kv_a;kc,kpe=comp.split([kv_lora,qk_rope],-1);kc=rms(kc,kv_a_ln)
kv=(kc@w_kv_b).view(1,S,n_heads,qk_nope+v_head);kno,vst=kv.split([qk_nope,v_head],-1)
kno=kno.transpose(1,2);vst=vst.transpose(1,2)
qp=rope(qp,f);kpe2=rope(kpe.view(1,1,S,qk_rope),f).expand(-1,n_heads,-1,-1)
qs=torch.cat([qn,qp],-1);ks=torch.cat([kno,kpe2],-1)
mask=torch.full((S,S),float('-inf'),dtype=DT).triu(1)[None,None]
aw=F.softmax((qs@ks.transpose(2,3))*scaling+mask,-1)
vmix=(aw@vst).transpose(1,2).reshape(S,n_heads*v_head);attn_o=vmix@w_o
print('=== layer-3 attention ref vs HPU (cos / ratio / abs-diff) ===')
for i in range(S):
    diff=(attn_o[i]-hpu_attn[i]).norm().item()
    print(f'  pos{i}: cos={cs(attn_o[i],hpu_attn[i]):.5f} |ref|={attn_o[i].norm():.3f} |hpu|={hpu_attn[i].norm():.3f} |diff|={diff:.3f} reldiff={diff/hpu_attn[i].norm():.4f}')
print()
print('softmax peakiness (max attn weight per query, head 0):', [round(aw[0,0,i,:].max().item(),3) for i in range(S)])
print('NOTE: |hidden_in|>>|attn|: hidden_in norm', [round(hidden_in[i].norm().item(),1) for i in range(S)],
      '  attn norm', [round(hpu_attn[i].norm().item(),1) for i in range(S)])
print('=> attn is ~2%% of the residual; even a perfect attn barely moves the router. The flat,')
print('   bias~34-dominated router means routing is essentially FIXED by e_score_correction_bias,')
print('   nearly independent of the hidden state.')
