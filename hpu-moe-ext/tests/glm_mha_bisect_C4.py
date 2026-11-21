"""
PART (C4): pin down the exact all-reduce/combine behavior of the HPU o_proj.
We have the 8 CORRECT per-rank partials (from checkpoint full attention). Test:
  - HPU == single rank r partial  (for each r)
  - HPU == sum over a SUBSET of ranks (greedy / all 2^8? -> test cumulative best-subset)
  - HPU == scaled single rank
  - which rank's process produced layer0_dump (residual_out)
Also report cos of HPU to each partial and to full, and the best linear fit restricted to
NONNEGATIVE integer subset sums.
"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'
MHA='/opt/hpu-dsa-ext/mha_dump.pt'; L0='/opt/hpu-dsa-ext/layer0_dump.pt'
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; n_heads=cfg['num_attention_heads']; kv_lora=cfg['kv_lora_rank']
qk_nope=cfg['qk_nope_head_dim']; qk_rope=cfg['qk_rope_head_dim']; qk_head=cfg['qk_head_dim']
v_head=cfg['v_head_dim']; eps=cfg['rms_norm_eps']; rope_theta=cfg['rope_parameters']['rope_theta']
scaling=qk_head**-0.5; TP=8; hpr=n_heads//TP; oin=hpr*v_head
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}',framework='pt',device='cpu') as f: return f.get_tensor(n)
def dq(p,gs=128,bits=8):
    qw=get(p+'.qweight');qz=get(p+'.qzeros');sc=get(p+'.scales')
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int16)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT); c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])
    rz=torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int16)
    rz=torch.bitwise_and(rz+1,(2**bits)-1).to(DT); z=rz.reshape(-1,rz.shape[1]*rz.shape[2])
    scf=sc.to(DT); g=(torch.arange(c.shape[0])//gs).long(); return (c-z[g])*scf[g]
def cs(a,b):
    a=a.reshape(-1);b=b.reshape(-1);return (a@b/(a.norm()*b.norm()+1e-30)).item()
def rms(x,w): return (w.to(DT)+1.0)*(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps))
def rh(x): return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),-1)
def br(pos,dim):
    inv=1.0/(rope_theta**(torch.arange(0,dim,2,dtype=DT)/dim));f=torch.einsum('s,d->sd',pos,inv)
    e=torch.cat((f,f),-1);return e.cos()[None],e.sin()[None]
def ar(x,c,s,u): return x*c.unsqueeze(u)+rh(x)*s.unsqueeze(u)
ld=torch.load(L0,map_location='cpu',weights_only=False)
hidden_in=ld['hidden_in'][0].to(DT); positions=ld['positions'][0].to(DT)
hpu_attn=(ld['residual_out'][0]-ld['hidden_in'][0]).to(DT); S=5
w_q_a=dq('model.layers.0.self_attn.q_a_proj'); w_q_b=dq('model.layers.0.self_attn.q_b_proj')
w_kv_a=dq('model.layers.0.self_attn.kv_a_proj_with_mqa'); w_kv_b=dq('model.layers.0.self_attn.kv_b_proj')
w_o=dq('model.layers.0.self_attn.o_proj')
ln_in=get('model.layers.0.input_layernorm.weight')
q_a_ln=get('model.layers.0.self_attn.q_a_layernorm.weight'); kv_a_ln=get('model.layers.0.self_attn.kv_a_layernorm.weight')
hs=rms(hidden_in[:S][None],ln_in); cos,sin=br(positions[:S],qk_rope)
q=(rms(hs@w_q_a,q_a_ln)@w_q_b).view(1,S,n_heads,qk_head).transpose(1,2)
qn,qp=q.split([qk_nope,qk_rope],-1); qp=ar(qp,cos,sin,1); qs=torch.cat([qn,qp],-1)
comp=hs@w_kv_a; kc,kpe=comp.split([kv_lora,qk_rope],-1); kc=rms(kc,kv_a_ln)
kv=(kc@w_kv_b).view(1,S,n_heads,qk_nope+v_head); kno,vst=kv.split([qk_nope,v_head],-1)
kno=kno.transpose(1,2); vst=vst.transpose(1,2)
kpe=ar(kpe.view(1,1,S,qk_rope),cos,sin,1).expand(-1,n_heads,-1,-1); ks=torch.cat([kno,kpe],-1)
mask=torch.full((S,S),float('-inf'),dtype=DT).triu(1)[None,None]
aw=F.softmax((qs@ks.transpose(2,3))*scaling+mask,-1)
vmix=(aw@vst).transpose(1,2).reshape(S,n_heads*v_head)
full_o=vmix@w_o
parts=[vmix[:,r*oin:(r+1)*oin]@w_o[r*oin:(r+1)*oin,:] for r in range(TP)]  # [5,6144] each
P0=torch.stack([parts[r][0] for r in range(TP)],0)  # [8,6144]

print('HPU pos0 norm=%.3f  full pos0 norm=%.3f'%(hpu_attn[0].norm().item(), full_o[0].norm().item()))
print()
print('cos(HPU_pos0, single rank partial) and best scale s=<HPU,part>/|part|^2 and residual:')
for r in range(TP):
    p=parts[r][0]; s=(hpu_attn[0]@p/(p@p)).item(); res=((hpu_attn[0]-s*p).norm()/hpu_attn[0].norm()).item()
    print('  rank %d: cos=%.4f scale=%.3f resid=%.4f |part|=%.3f'%(r,cs(hpu_attn[0],p),s,res,p.norm().item()))

# best CONTIGUOUS-prefix / arbitrary subset sum by greedy correlation
import itertools
print()
print('Best subset-sum of correct partials matching HPU_pos0 (search all 255 nonempty subsets):')
best=(-2,None)
for k in range(1,TP+1):
    for sub in itertools.combinations(range(TP),k):
        v=sum(parts[r][0] for r in sub)
        c=cs(v,hpu_attn[0])
        if c>best[0]: best=(c,sub,v.norm().item())
print('  best subset=%s cos=%.4f |subset|=%.3f (HPU=%.3f)'%(best[1],best[0],best[2],hpu_attn[0].norm().item()))

# Which rank's PROCESS produced residual_out? If reduce dropped, residual_out=hidden_in+own_partial.
# Test hidden-in+partial_r interpretation already covered: HPU_attn=partial_r. None matched well.
# So the all-reduce is NOT simply dropped. Characterize spectrum:
print()
print('energy of HPU in each correct-partial direction (normalized dot^2):')
for r in range(TP):
    p=parts[r][0]; e=(hpu_attn[0]@p/(hpu_attn[0].norm()*p.norm()))**2
    print('  rank %d: %.4f'%(r,e.item()))

# Last: is HPU ~ full_o * c with c~0.41 plus orthogonal noise? decompose:
c=(hpu_attn[0]@full_o[0]/(full_o[0]@full_o[0])).item()
resid=(hpu_attn[0]-c*full_o[0])
print()
print('HPU = %.4f*full_o + noise; noise/|HPU|=%.4f (so only %.1f%% of HPU is the correct direction)'
      %(c,(resid.norm()/hpu_attn[0].norm()).item(), 100*(c*full_o[0]).norm().item()/hpu_attn[0].norm()))
