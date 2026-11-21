"""
PART (C): o_proj / all-reduce bisection.
(A) proved the per-rank pre-o_proj value mixture is CORRECT (cos~1.0).
Now: the FULL post-o_proj all-reduced attention output (residual_out - hidden_in)
has cos~0.11 and ~1/sqrt(8) magnitude vs the CPU reference (ref pos0 o_proj norm 42.96).

This script determines whether the HPU result equals:
  - the FULL all-reduced o_proj over all 8 ranks  (correct), OR
  - a SINGLE rank's o_proj partial (full/sqrt8 or full/8) -> missing/wrong all-reduce.

We reconstruct rank 2's EXACT o_proj partial from the dumped fwd_mha_out (this rank's
8-head value mixture) times the rank-2 column slice of the dequanted o_proj weight, and
also compute the FULL 64-head reference o_proj from the checkpoint.

Run:
  docker exec -e CUDA_VISIBLE_DEVICES='' -e HABANA_VISIBLE_DEVICES='' \
              -e HABANA_VISIBLE_MODULES='' vllm-active \
              python3 /opt/hpu-moe-ext/tests/glm_mha_bisect_C.py
"""
import json
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)
DT = torch.float64

CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
MHA  = '/opt/hpu-dsa-ext/mha_dump.pt'
L0   = '/opt/hpu-dsa-ext/layer0_dump.pt'

cfg = json.load(open(f'{CKPT}/config.json'))
H        = cfg['hidden_size']            # 6144
n_heads  = cfg['num_attention_heads']    # 64
q_lora   = cfg['q_lora_rank']            # 2048
kv_lora  = cfg['kv_lora_rank']           # 512
qk_nope  = cfg['qk_nope_head_dim']       # 192
qk_rope  = cfg['qk_rope_head_dim']       # 64
qk_head  = cfg['qk_head_dim']            # 256
v_head   = cfg['v_head_dim']             # 256
eps      = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']
scaling  = qk_head ** -0.5

TP = 8
RANK = 6   # CORRECTED: dump is the rank holding contiguous heads 48-55 (verified cos=1.0 in C2)
heads_per_rank = n_heads // TP           # 8
o_in_per_rank = heads_per_rank * v_head  # 2048

idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(name):
    with safe_open(f'{CKPT}/{idx[name]}', framework='pt', device='cpu') as f:
        return f.get_tensor(name)

# 8-bit sym dequant -> [in, out]
def dequant8(prefix, group_size=128, bits=8):
    qw = get(prefix + '.qweight'); qz = get(prefix + '.qzeros'); sc = get(prefix + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    codes = torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1), wf.unsqueeze(-1)).to(torch.int16)
    codes = torch.bitwise_and(codes, (2**bits)-1).to(DT)
    codes = codes.reshape(codes.shape[0]*codes.shape[1], codes.shape[2])
    raw_z = torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits), wf.unsqueeze(0)).to(torch.int16)
    raw_z = (raw_z + 1); raw_z = torch.bitwise_and(raw_z, (2**bits)-1).to(DT)
    zeros = raw_z.reshape(-1, raw_z.shape[1]*raw_z.shape[2])
    scf = sc.to(DT); g = (torch.arange(codes.shape[0])//group_size).long()
    return (codes - zeros[g]) * scf[g]

def cossim(a,b):
    a=a.reshape(-1); b=b.reshape(-1)
    return (a@b/(a.norm()*b.norm()+1e-30)).item()

# ---------- load dumps ----------
md = torch.load(MHA, map_location='cpu', weights_only=False)
fwd_mha_out = md['fwd_mha_out'].to(DT)          # [128,2048]  rank2's 8-head value mixture
ld = torch.load(L0, map_location='cpu', weights_only=False)
hidden_in    = ld['hidden_in'][0].to(DT)        # [128,6144]
residual_out = ld['residual_out'][0].to(DT)     # [128,6144]
hpu_attn = residual_out - hidden_in             # [128,6144]  HPU's post-o_proj all-reduced attn out

# ---------- o_proj weight ----------
P = 'model.layers.0.self_attn.o_proj'
w_o = dequant8(P)                                # [16384, 6144]  ([in,out])
print('o_proj dequant shape (in,out):', tuple(w_o.shape))
assert w_o.shape == (n_heads*v_head, H), w_o.shape

# rank-2's slice of o_proj INPUT dims = [RANK*2048 : (RANK+1)*2048]
w_o_rank = w_o[RANK*o_in_per_rank:(RANK+1)*o_in_per_rank, :]   # [2048, 6144]
print('rank-%d o_proj slice shape:' % RANK, tuple(w_o_rank.shape))

S = 5
# rank-2's o_proj PARTIAL contribution for the 5 real tokens:
rank_partial = fwd_mha_out[:S] @ w_o_rank        # [5,6144]

# ---------- full 64-head reference o_proj (from checkpoint) ----------
# Reproduce the full attention to get the 64-head value mixture, then full o_proj.
def rmsnorm(x,w,plus_one=True):
    wf=w.to(DT)+(1.0 if plus_one else 0.0)
    return wf*(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps))
def rotate_half(x):
    x1=x[...,:x.shape[-1]//2]; x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),-1)
def build_rope(pos,dim):
    inv=1.0/(rope_theta**(torch.arange(0,dim,2,dtype=DT)/dim))
    f=torch.einsum('s,d->sd',pos,inv); emb=torch.cat((f,f),-1)
    return emb.cos()[None], emb.sin()[None]
def apply_rope(x,cos,sin,ud):
    return x*cos.unsqueeze(ud)+rotate_half(x)*sin.unsqueeze(ud)

w_q_a = dequant8('model.layers.0.self_attn.q_a_proj')
w_q_b = dequant8('model.layers.0.self_attn.q_b_proj')
w_kv_a= dequant8('model.layers.0.self_attn.kv_a_proj_with_mqa')
w_kv_b= dequant8('model.layers.0.self_attn.kv_b_proj')
ln_in = get('model.layers.0.input_layernorm.weight')
q_a_ln= get('model.layers.0.self_attn.q_a_layernorm.weight')
kv_a_ln=get('model.layers.0.self_attn.kv_a_layernorm.weight')

positions = ld['positions'][0,:S].to(DT)
hs = rmsnorm(hidden_in[:S][None], ln_in)         # [1,5,6144]
cos,sin = build_rope(positions, qk_rope)
B=1
q_resid = rmsnorm(hs @ w_q_a, q_a_ln)
q = (q_resid @ w_q_b).view(B,S,n_heads,qk_head).transpose(1,2)
q_nope,q_pe = q.split([qk_nope,qk_rope],-1)
q_pe = apply_rope(q_pe,cos,sin,1)
qs = torch.cat([q_nope,q_pe],-1)
comp = hs @ w_kv_a
k_comp,k_pe = comp.split([kv_lora,qk_rope],-1)
k_comp = rmsnorm(k_comp,kv_a_ln)
kv = (k_comp @ w_kv_b).view(B,S,n_heads,qk_nope+v_head)
k_nope,vstate = kv.split([qk_nope,v_head],-1)
k_nope=k_nope.transpose(1,2); vstate=vstate.transpose(1,2)
k_pe=apply_rope(k_pe.view(B,1,S,qk_rope),cos,sin,1).expand(-1,n_heads,-1,-1)
ks=torch.cat([k_nope,k_pe],-1)
mask=torch.full((S,S),float('-inf'),dtype=DT).triu(1)[None,None]
aw=F.softmax((qs@ks.transpose(2,3))*scaling+mask,-1)
ao=(aw@vstate).transpose(1,2).reshape(B,S,n_heads*v_head)   # [1,5,16384]
full_o = (ao[0] @ w_o)                                       # [5,6144]  FULL all-reduced o_proj ref

# cross-check: full reference attn_out[:,rank2 8 heads] should match dumped fwd_mha_out
ref_rank2_vmix = ao[0].view(S, n_heads, v_head)[:, RANK*heads_per_rank:(RANK+1)*heads_per_rank, :].reshape(S,-1)
print('cross-check ref-rank2 value-mixture vs dumped fwd_mha_out cos:',
      round(cossim(ref_rank2_vmix, fwd_mha_out[:S]),6),
      '|ref|=%.3f |dump|=%.3f'%(ref_rank2_vmix.norm().item(), fwd_mha_out[:S].norm().item()))

# ---------- compare ----------
print()
print('='*88)
print('PART (C): HPU implied attn_out  vs  {full o_proj, single-rank partial}')
print('='*88)
print(f'{"pos":>3s} {"|HPU|":>9s} {"|full_o|":>9s} {"|rank2|":>9s}  {"cos(HPU,full)":>13s} {"cos(HPU,rank2)":>14s}  {"|full|/|HPU|":>12s}')
for i in range(S):
    h=hpu_attn[i]; fo=full_o[i]; rp=rank_partial[i]
    print(f'{i:3d} {h.norm().item():9.3f} {fo.norm().item():9.3f} {rp.norm().item():9.3f}  '
          f'{cossim(h,fo):13.5f} {cossim(h,rp):14.5f}  {(fo.norm()/h.norm()).item():12.4f}')

print()
print('sum of 8 identical-magnitude random partials ~ sqrt(8)*single; full=sum of 8 partials.')
print('full/HPU ratio ~ sqrt(8)=%.4f would indicate HPU = ONE rank (missing all-reduce, partials add incoherently)' % (TP**0.5))
print('full/HPU ratio ~ 8=%.1f would indicate HPU = full/8 (averaged instead of summed)' % TP)

# Is HPU ~ exactly rank2 partial?  (the dumped rank). If the all-reduce is dropped, every
# rank returns ITS OWN partial; the one we can reconstruct exactly is rank 2.
print()
print('DIRECT: cos(HPU_attn, rank2_partial) and magnitude ratio:')
for i in range(S):
    h=hpu_attn[i]; rp=rank_partial[i]
    print('  pos %d: cos=%.5f  |rank2|/|HPU|=%.4f' % (i, cossim(h,rp), (rp.norm()/h.norm()).item()))

# Decompose: does HPU lie in the span of a single rank partial, or is it the full sum?
# Also report cos(full_o, rank2_partial) to know how correlated a single rank is to the full.
print()
print('reference correlations:')
for i in range(S):
    print('  pos %d: cos(full_o, rank2_partial)=%.5f  |rank2|/|full|=%.4f'
          % (i, cossim(full_o[i], rank_partial[i]), (rank_partial[i].norm()/full_o[i].norm()).item()))
