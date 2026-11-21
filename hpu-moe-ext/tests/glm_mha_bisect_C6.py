"""
PART (C6): re-examine layer0_dump semantics. HPU (residual_out-hidden_in) is orthogonal (89%)
to ANY correct o_proj rearrangement. Test what residual_out-hidden_in ACTUALLY is:
  - the correct full layer attention output? (already: cos 0.11)
  - the correct full layer output incl MLP?
  - just MLP? (dense layer-0 MLP is bf16, computed from post-attn-residual)
  - dominated by the padding tokens (positions 5..127)?
Compute the full correct layer-0 (attn + dense MLP) on CPU and compare every interpretation.
Also: examine padding-token positions (5..127) - are they nonzero/garbage on HPU?
"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; L0='/opt/hpu-dsa-ext/layer0_dump.pt'
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; n_heads=cfg['num_attention_heads']; kv_lora=cfg['kv_lora_rank']
qk_nope=cfg['qk_nope_head_dim']; qk_rope=cfg['qk_rope_head_dim']; qk_head=cfg['qk_head_dim']
v_head=cfg['v_head_dim']; eps=cfg['rms_norm_eps']; rope_theta=cfg['rope_parameters']['rope_theta']
scaling=qk_head**-0.5
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
hidden_in_full=ld['hidden_in'][0].to(DT)        # [128,6144]
positions=ld['positions'][0].to(DT)
residual_out_full=ld['residual_out'][0].to(DT)  # [128,6144]
hidden_out_full=ld['hidden_out'].to(DT)         # [128,6144]
print('positions[:10]=',positions[:10].tolist())
print('hidden_in norm/pos[:8]:',[round(hidden_in_full[i].norm().item(),2) for i in range(8)])
print('residual_out norm/pos[:8]:',[round(residual_out_full[i].norm().item(),2) for i in range(8)])
print('hidden_out  norm/pos[:8]:',[round(hidden_out_full[i].norm().item(),2) for i in range(8)])
print('residual_out norm pos[120:128]:',[round(residual_out_full[i].norm().item(),2) for i in range(120,128)])
print('(residual_out-hidden_in) norm/pos[:8]:',[round((residual_out_full-hidden_in_full)[i].norm().item(),2) for i in range(8)])
print()
S=5
hidden_in=hidden_in_full[:S][None]
hpu_attn=(residual_out_full-hidden_in_full)[:S]
# full correct attn
w_q_a=dq('model.layers.0.self_attn.q_a_proj'); w_q_b=dq('model.layers.0.self_attn.q_b_proj')
w_kv_a=dq('model.layers.0.self_attn.kv_a_proj_with_mqa'); w_kv_b=dq('model.layers.0.self_attn.kv_b_proj')
w_o=dq('model.layers.0.self_attn.o_proj')
ln_in=get('model.layers.0.input_layernorm.weight'); ln_post=get('model.layers.0.post_attention_layernorm.weight')
q_a_ln=get('model.layers.0.self_attn.q_a_layernorm.weight'); kv_a_ln=get('model.layers.0.self_attn.kv_a_layernorm.weight')
gate_w=get('model.layers.0.mlp.gate_proj.weight').to(DT); up_w=get('model.layers.0.mlp.up_proj.weight').to(DT); down_w=get('model.layers.0.mlp.down_proj.weight').to(DT)
hs=rms(hidden_in,ln_in); cos,sin=br(positions[:S],qk_rope)
q=(rms(hs@w_q_a,q_a_ln)@w_q_b).view(1,S,n_heads,qk_head).transpose(1,2)
qn,qp=q.split([qk_nope,qk_rope],-1); qp=ar(qp,cos,sin,1); qs=torch.cat([qn,qp],-1)
comp=hs@w_kv_a; kc,kpe=comp.split([kv_lora,qk_rope],-1); kc=rms(kc,kv_a_ln)
kv=(kc@w_kv_b).view(1,S,n_heads,qk_nope+v_head); kno,vst=kv.split([qk_nope,v_head],-1)
kno=kno.transpose(1,2); vst=vst.transpose(1,2)
kpe=ar(kpe.view(1,1,S,qk_rope),cos,sin,1).expand(-1,n_heads,-1,-1); ks=torch.cat([kno,kpe],-1)
mask=torch.full((S,S),float('-inf'),dtype=DT).triu(1)[None,None]
aw=F.softmax((qs@ks.transpose(2,3))*scaling+mask,-1)
vmix=(aw@vst).transpose(1,2).reshape(S,n_heads*v_head); attn_o=vmix@w_o   # [5,6144]
resid_after_attn=hidden_in[0]+attn_o
mn=rms(resid_after_attn[None],ln_post)[0]
mlp_out=(F.silu(mn@gate_w.t())*(mn@up_w.t()))@down_w.t()
full_layer=resid_after_attn+mlp_out

print('INTERPRETATION TESTS for (residual_out - hidden_in) at pos0..4:')
def line(name,c):
    print('  %-40s cos/pos=%s'%(name,[round(cs(c[i],hpu_attn[i]),4) for i in range(S)]))
line('correct attn_o (o_proj out)',attn_o)
line('correct mlp_out',mlp_out)
line('correct attn_o+mlp_out (full delta)',attn_o+mlp_out)
print()
print('also: residual_out vs correct (hidden_in+attn_o):')
print('  cos/pos=',[round(cs(residual_out_full[i],(hidden_in[0]+attn_o)[i]),4) for i in range(S)])
print('hidden_out (mlp delta) vs correct mlp_out:')
print('  cos/pos=',[round(cs(hidden_out_full[i],mlp_out[i]),4) for i in range(S)],
      ' |hpu|/|ref|=',[round((hidden_out_full[i].norm()/mlp_out[i].norm()).item(),3) for i in range(S)])
print()
print('FULL layer out: HPU(residual_out+hidden_out) vs correct full_layer:')
hpu_full=residual_out_full[:S]+hidden_out_full[:S]
print('  cos/pos=',[round(cs(hpu_full[i],full_layer[i]),4) for i in range(S)],
      ' norms HPU=',[round(hpu_full[i].norm().item(),2) for i in range(S)],' ref=',[round(full_layer[i].norm().item(),2) for i in range(S)])
