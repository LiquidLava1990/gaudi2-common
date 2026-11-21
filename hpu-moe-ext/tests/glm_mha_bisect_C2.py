"""
PART (C2): identify which head-shard the dumped rank actually holds, and reconcile o_proj.
The (C) cross-check failed (cos 0.001) -> dumped fwd_mha_out is NOT checkpoint heads 16-23.
Find which of the 8 contiguous 8-head blocks (or none) the dumped value-mixture matches,
by comparing the dumped q AND latent against the full reference per-head q/value tensors.
"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'
MHA='/opt/hpu-dsa-ext/mha_dump.pt'; L0='/opt/hpu-dsa-ext/layer0_dump.pt'
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; n_heads=cfg['num_attention_heads']; kv_lora=cfg['kv_lora_rank']
qk_nope=cfg['qk_nope_head_dim']; qk_rope=cfg['qk_rope_head_dim']; qk_head=cfg['qk_head_dim']
v_head=cfg['v_head_dim']; eps=cfg['rms_norm_eps']; rope_theta=cfg['rope_parameters']['rope_theta']
scaling=qk_head**-0.5; TP=8; hpr=n_heads//TP
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

md=torch.load(MHA,map_location='cpu',weights_only=False)
q_dump=md['q'].to(DT)            # [128,8,256]
latent_dump=md['latent_vec_k'].to(DT)  # [128,576]
fwd=md['fwd_mha_out'].to(DT)
ld=torch.load(L0,map_location='cpu',weights_only=False)
hidden_in=ld['hidden_in'][0].to(DT); positions=ld['positions'][0].to(DT)
S=5

# full reference q (all 64 heads, post rope) and latent (replicated, single)
w_q_a=dq('model.layers.0.self_attn.q_a_proj'); w_q_b=dq('model.layers.0.self_attn.q_b_proj')
w_kv_a=dq('model.layers.0.self_attn.kv_a_proj_with_mqa')
ln_in=get('model.layers.0.input_layernorm.weight')
q_a_ln=get('model.layers.0.self_attn.q_a_layernorm.weight')
kv_a_ln=get('model.layers.0.self_attn.kv_a_layernorm.weight')
hs=rms(hidden_in[:S][None],ln_in)
cos,sin=br(positions[:S],qk_rope)
q_resid=rms(hs@w_q_a,q_a_ln)
qfull=(q_resid@w_q_b).view(1,S,n_heads,qk_head)             # [1,5,64,256]
qn,qp=qfull.split([qk_nope,qk_rope],-1)
qp=ar(qp.transpose(1,2),cos,sin,1).transpose(1,2)            # rope on pe
qfull=torch.cat([qn,qp],-1)[0]                               # [5,64,256]
comp=hs@w_kv_a
kc,kpe=comp.split([kv_lora,qk_rope],-1)
kc=rms(kc,kv_a_ln)                                           # [1,5,512]
kpe_r=ar(kpe.view(1,1,S,qk_rope),cos,sin,1)[0,0]            # [5,64]
latent_ref=torch.cat([kc[0],kpe_r],-1)                      # [5,576]

# dumped q for this rank: [5,8,256]; match to which contiguous 8-head block of qfull?
qd=q_dump[:S]      # [5,8,256]
print('latent match: cos(dumped latent, ref latent) =', round(cs(latent_dump[:S], latent_ref),6),
      ' |dump|=%.3f |ref|=%.3f'%(latent_dump[:S].norm().item(), latent_ref.norm().item()))
print()
print('Which contiguous 8-head block does the dumped q match?')
best=-2; bestblk=-1
for blk in range(TP):
    ref_blk=qfull[:,blk*hpr:(blk+1)*hpr,:]    # [5,8,256]
    c=cs(qd,ref_blk)
    print('  block %d (heads %2d-%2d): cos=%.5f'%(blk,blk*hpr,(blk+1)*hpr-1,c))
    if c>best: best=c; bestblk=blk

# Also test: maybe heads are STRIDED (head h on rank r = r + TP*j) i.e. interleaved sharding
print()
print('Strided/interleaved sharding test (rank r holds heads r, r+8, r+16, ...):')
for r in range(TP):
    ref_str=qfull[:, r::TP, :]   # [5,8,256]
    c=cs(qd,ref_str)
    print('  rank %d (heads %s): cos=%.5f'%(r, list(range(r,n_heads,TP)), c))

print()
print('BEST contiguous block =',bestblk,'cos=%.5f'%best)
