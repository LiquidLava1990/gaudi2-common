"""Verify the dump's hidden_in matches the checkpoint embedding for the prompt tokens.
If hidden_in is already corrupt, the layer-0 forward bug is upstream (embedding)."""
import json, torch
import torch.nn.functional as F
torch.set_grad_enabled(False); DT=torch.float32
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; DUMP='/opt/hpu-dsa-ext/layer0_dump.pt'
from safetensors import safe_open
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
emb_shard=idx['model.embed_tokens.weight']
with safe_open(f'{CKPT}/{emb_shard}','pt','cpu') as f:
    emb=f.get_tensor('model.embed_tokens.weight')
print('embed dtype',emb.dtype,'shape',emb.shape)
emb=emb.to(DT)
d=torch.load(DUMP,map_location='cpu',weights_only=False)
hi=d['hidden_in'][0,:5].to(DT)   # [5,6144]
print('hidden_in norm/pos:',[round(x,4) for x in hi.norm(dim=-1).tolist()])
print('embed row norm (typical):', round(emb.norm(dim=-1).mean().item(),4),'+-',round(emb.norm(dim=-1).std().item(),4))
# tokenize the prompt
try:
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(CKPT)
    for p in ['The capital of France is','The capital of France is ']:
        ids=tok(p,return_tensors='pt').input_ids[0]
        print(f'prompt {p!r} -> ids {ids.tolist()}')
except Exception as e:
    print('tok err',e); ids=None
# For each hidden_in position, find nearest embedding row
for i in range(5):
    v=hi[i]
    # cosine to all embed rows is expensive (155k x 6144); do it in chunks
    sims=torch.empty(emb.shape[0])
    vn=v/v.norm()
    bs=20000
    for s in range(0,emb.shape[0],bs):
        e=emb[s:s+bs]; en=e/(e.norm(dim=-1,keepdim=True)+1e-9)
        sims[s:s+bs]=en@vn
    top=sims.topk(3)
    print(f'pos{i}: hidden_in norm={v.norm():.4f}  best-match embed rows={top.indices.tolist()} cos={[round(x,4) for x in top.values.tolist()]}')
    # exact: is hidden_in == embed[best]?
    b=top.indices[0].item()
    diff=(v-emb[b]).abs().max().item()
    print(f'       embed[{b}] norm={emb[b].norm():.4f}  maxabsdiff(hidden_in,embed)={diff:.4e}  ratio_norm={ (v.norm()/emb[b].norm()).item():.4f}')

print()
print('='*90); print('TP=8 PARTIAL-REDUCTION TEST: does HPU = only-some-shards of o_proj?'); print('='*90)
import json as _j
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}','pt','cpu') as f: return f.get_tensor(n)
cfg=_j.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size'];n_heads=64;kv_lora=512;qk_nope=192;qk_rope=64;qk_head=256;v_head=256;eps=cfg['rms_norm_eps']
rope_theta=cfg['rope_parameters']['rope_theta'];scaling=qk_head**-0.5;S=5;B=1
def dq(p,gs=128,bits=8):
    qw=get(p+'.qweight');qz=get(p+'.qzeros');sc=get(p+'.scales')
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int16)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT);c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])
    z=torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int16)
    z=z+1;z=torch.bitwise_and(z,(2**bits)-1).to(DT);z=z.reshape(-1,z.shape[1]*z.shape[2])
    g=(torch.arange(c.shape[0])//gs).long();return (c-z[g])*sc.to(DT)[g]
def rms(x,w):
    v=x.to(DT).pow(2).mean(-1,keepdim=True);return (w.to(DT)+1.0)*(x.to(DT)*torch.rsqrt(v+eps))
def rh(x): return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),-1)
def cs(a,b):
    a=a.reshape(-1).double();b=b.reshape(-1).double();return (a@b/(a.norm()*b.norm())).item()
P='model.layers.0';
w_q_a=dq(f'{P}.self_attn.q_a_proj');w_q_b=dq(f'{P}.self_attn.q_b_proj')
w_kv_a=dq(f'{P}.self_attn.kv_a_proj_with_mqa');w_kv_b=dq(f'{P}.self_attn.kv_b_proj');w_o=dq(f'{P}.self_attn.o_proj')
ln_in=get(f'{P}.input_layernorm.weight');q_a_ln=get(f'{P}.self_attn.q_a_layernorm.weight');kv_a_ln=get(f'{P}.self_attn.kv_a_layernorm.weight')
ro=d['residual_out'][0,:5].to(DT); ha=ro-hi
hs=rms(hi[None],ln_in)
inv=1.0/(rope_theta**(torch.arange(0,qk_rope,2).float()/qk_rope));fr=torch.einsum('s,d->sd',torch.tensor([0,1,2,3,4.]),inv)
emb_=torch.cat((fr,fr),-1);cosN=emb_.cos()[None];sinN=emb_.sin()[None]
qres=rms((hs@w_q_a),q_a_ln);q=(qres@w_q_b).view(B,S,n_heads,qk_head).transpose(1,2)
qn,qp=torch.split(q,[qk_nope,qk_rope],-1)
comp=hs@w_kv_a;kc,kp=torch.split(comp,[kv_lora,qk_rope],-1);kc=rms(kc,kv_a_ln)
kve=(kc@w_kv_b).view(B,S,n_heads,qk_nope+v_head);kn,vs=torch.split(kve,[qk_nope,v_head],-1);kn=kn.transpose(1,2);vs=vs.transpose(1,2)
kp=kp.view(B,1,S,qk_rope)
def neox(x,c,s,u): cc=c.unsqueeze(u);ss=s.unsqueeze(u);return x*cc+rh(x)*ss
qp=neox(qp,cosN,sinN,1);kp=neox(kp,cosN,sinN,1).expand(-1,n_heads,-1,-1)
am=torch.full((S,S),float('-inf')).triu(1)[None,None]
aw=F.softmax((torch.cat([qn,qp],-1)@torch.cat([kn,kp],-1).transpose(2,3))*scaling+am,-1,dtype=DT)
ao=(aw@vs).transpose(1,2).reshape(B,S,n_heads*v_head)[0]  # [5,16384]
# partial-shard o_proj: sum over first k shards only (each shard=8 heads=2048 dims)
woh=w_o.view(n_heads,v_head,H)  # [64,256,6144]
for k in [1,2,4,8]:
    mask=torch.zeros(n_heads,1,1);mask[:k*8]=1
    out=(ao.view(S,n_heads,v_head).unsqueeze(-1)*(woh*mask).unsqueeze(0)).sum((1,2))  # [5,6144]
    cv=[round(cs(out[i],ha[i]),4) for i in range(S)]
    print('o_proj first %d/8 shards: norm/pos=%s cos=%s'%(k,[round(x,2) for x in out.norm(dim=-1).tolist()],cv))
print('HPU implied norm/pos=%s'%[round(x,2) for x in ha.norm(dim=-1).tolist()])
print()
# Pure-noise baseline: cos of random 6144-vec to ha
torch.manual_seed(1);rnd=torch.randn(5,6144)
print('random-vector baseline cos to HPU:',[round(cs(rnd[i],ha[i]),4) for i in range(5)],'(|cos|~%.3f = noise floor)'%(1/ (6144**0.5)))
