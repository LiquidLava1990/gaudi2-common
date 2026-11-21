"""
PART (C3): characterize the o_proj/all-reduce corruption precisely.
Findings so far: HPU implied attn_out has cos=0.11 to FULL correct o_proj, mag = full/2.42.
That LOW cos (not ~1.0) rules out a pure magnitude bug. Test specific corruptions:
  H1: HPU = sum over ranks of (rank_vmix @ WRONG_weight_slice)  -> incoherent sum (head-perm/shard mismatch)
  H2: HPU = full_o but with a partial (subset of ranks) all-reduce
  H3: HPU = full_o computed with o_proj weight TRANSPOSED or mis-dequanted
  H4: decompose HPU into the 8 per-rank partial subspaces - how much energy in each.
We can only reconstruct THIS rank's (heads48-55) partial exactly. But we can compute the
FULL correct partials for all 8 ranks from the checkpoint (full attention), and test whether
the HPU output is the sum of a PERMUTED assignment of value-mixtures to weight-slices.
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
hpu_attn=(ld['residual_out'][0]-ld['hidden_in'][0]).to(DT)
S=5

# full reference attention -> 64-head value mixture
w_q_a=dq('model.layers.0.self_attn.q_a_proj'); w_q_b=dq('model.layers.0.self_attn.q_b_proj')
w_kv_a=dq('model.layers.0.self_attn.kv_a_proj_with_mqa'); w_kv_b=dq('model.layers.0.self_attn.kv_b_proj')
w_o=dq('model.layers.0.self_attn.o_proj')      # [16384,6144]
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
vmix=(aw@vst).transpose(1,2).reshape(S,n_heads*v_head)   # [5,16384]
full_o=vmix@w_o                                          # [5,6144]  CORRECT all-reduced
print('full_o norm/pos:',[round(full_o[i].norm().item(),3) for i in range(S)])
print('HPU norm/pos:   ',[round(hpu_attn[i].norm().item(),3) for i in range(S)])
print('cos(full_o,HPU)/pos:',[round(cs(full_o[i],hpu_attn[i]),5) for i in range(S)])
print()

# per-rank correct partials
parts=[vmix[:,r*oin:(r+1)*oin] @ w_o[r*oin:(r+1)*oin,:] for r in range(TP)]  # each [5,6144]
print('--- per-rank CORRECT partial norms (pos0) and cos to HPU(pos0) ---')
for r in range(TP):
    print('  rank %d: |part|=%.3f  cos(part,HPU_pos0)=%.4f'%(r, parts[r][0].norm().item(), cs(parts[r][0],hpu_attn[0])))
print('sum of all 8 partials == full_o? maxdiff=%.2e'%(sum(parts)-full_o).abs().max().item())

# H4: project HPU onto the span of the 8 correct partials (least squares) - residual energy
P=torch.stack([parts[r][0] for r in range(TP)],1)   # [6144,8]
coef,_,_,_=torch.linalg.lstsq(P, hpu_attn[0])
recon=P@coef
print()
print('H4 lstsq: HPU_pos0 onto 8 correct-partial subspace: coefs=',[round(c.item(),3) for c in coef])
print('   recon cos=%.4f  residual frac=%.4f'%(cs(recon,hpu_attn[0]), ((hpu_attn[0]-recon).norm()/hpu_attn[0].norm()).item()))

# H3: o_proj weight transposed / wrong orientation? Test full_o with w_o.T won't conform (16384!=6144). skip.
# Test: maybe device applies o_proj but the ATTN_OUT is only THIS rank everywhere (no gather) i.e.
# every rank multiplies the SAME (rank6) value-mixture by ITS OWN weight slice then sums:
md=torch.load(MHA,map_location='cpu',weights_only=False)
fwd=md['fwd_mha_out'][:S].to(DT)   # rank6 value-mixture [5,2048]
# sum_r fwd @ w_o[r-slice]  == fwd @ (sum of all slices)  == fwd @ w_o.view(8,2048,6144).sum(0)
wo_blocks=w_o.view(TP,oin,H)
H_sameval = fwd @ wo_blocks.sum(0)    # [5,6144]
print()
print('H_sameval (all ranks use rank6 vmix, sum weight-slices): cos to HPU/pos=',
      [round(cs(H_sameval[i],hpu_attn[i]),4) for i in range(S)],
      ' |/|HPU|=', [round((H_sameval[i].norm()/hpu_attn[i].norm()).item(),3) for i in range(S)])

# H5: NO all-reduce, output = just rank6 partial (what one rank returns). already low. confirm:
print('H5 rank6-only partial: cos to HPU/pos=',[round(cs(parts[6][i],hpu_attn[i]),4) for i in range(S)],
      ' |/|HPU|=',[round((parts[6][i].norm()/hpu_attn[i].norm()).item(),3) for i in range(S)])

# H6: maybe HPU divides full_o by something AND adds noise. Check direction-only: cos already 0.11 -> direction wrong.
# H7: o_proj applied to v-mixture but with heads in WRONG order on each rank (intra-rank head permutation):
# Test: full_o with each rank's 8 heads internally reversed (a common transpose/reshape bug).
vmix_h=vmix.view(S,n_heads,v_head)
def permuted_full(perm_within_rank):
    vp=vmix_h.clone().view(S,TP,hpr,v_head)
    vp=vp[:,:,perm_within_rank,:].reshape(S,n_heads*v_head)
    return vp@w_o
rev=list(range(hpr))[::-1]
fo_rev=permuted_full(rev)
print()
print('H7 heads reversed within each rank: cos to HPU/pos=',[round(cs(fo_rev[i],hpu_attn[i]),4) for i in range(S)])
# H8: value head dim (256) split into 2x128 swapped? (qk_rope/nope confusion in v) - test halves swap
def vhalf_swap():
    vp=vmix.view(S,n_heads,v_head)
    vp=torch.cat([vp[...,v_head//2:],vp[...,:v_head//2]],-1).reshape(S,n_heads*v_head)
    return vp@w_o
fo_vs=vhalf_swap()
print('H8 v-head halves swapped: cos to HPU/pos=',[round(cs(fo_vs[i],hpu_attn[i]),4) for i in range(S)])
