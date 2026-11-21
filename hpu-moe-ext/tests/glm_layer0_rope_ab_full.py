"""Full layer-0 attention under NeoX vs INTERLEAVED rope, compared to HPU output.
Decides whether the HPU dump was produced with the wrong (interleaved) rope."""
import json, torch
import torch.nn.functional as F
torch.set_grad_enabled(False); DT=torch.float32
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'; DUMP='/opt/hpu-dsa-ext/layer0_dump.pt'
cfg=json.load(open(f'{CKPT}/config.json'))
H=cfg['hidden_size']; n_heads=cfg['num_attention_heads']; kv_lora=cfg['kv_lora_rank']
qk_nope=cfg['qk_nope_head_dim']; qk_rope=cfg['qk_rope_head_dim']; qk_head=cfg['qk_head_dim']
v_head=cfg['v_head_dim']; eps=cfg['rms_norm_eps']; rope_theta=cfg['rope_parameters']['rope_theta']
scaling=qk_head**-0.5
from safetensors import safe_open
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}','pt','cpu') as f: return f.get_tensor(n)
def dequant8(p,gs=128,bits=8):
    qw=get(p+'.qweight');qz=get(p+'.qzeros');sc=get(p+'.scales')
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int16)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT); c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])
    z=torch.bitwise_right_shift(torch.unsqueeze(qz,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int16)
    z=z+1; z=torch.bitwise_and(z,(2**bits)-1).to(DT); z=z.reshape(-1,z.shape[1]*z.shape[2])
    scf=sc.to(DT); g=(torch.arange(c.shape[0])//gs).long()
    return (c-z[g])*scf[g]
def rmsnorm(x,w):
    wf=w.to(DT)+1.0; v=x.to(DT).pow(2).mean(-1,keepdim=True)
    return wf*(x.to(DT)*torch.rsqrt(v+eps))
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]),dim=-1)
def cossim(a,b):
    a=a.reshape(-1).double();b=b.reshape(-1).double();return (a@b/(a.norm()*b.norm())).item()

d=torch.load(DUMP,map_location='cpu',weights_only=False)
pos=d['positions'][0,:5].to(DT); hidden_in=d['hidden_in'][:,:5,:].to(DT)
residual_out=d['residual_out'][0,:5].to(DT)
hpu_attn=residual_out-hidden_in[0]  # [5,6144]
S=5;B=1
P='model.layers.0'
w_q_a=dequant8(f'{P}.self_attn.q_a_proj');w_q_b=dequant8(f'{P}.self_attn.q_b_proj')
w_kv_a=dequant8(f'{P}.self_attn.kv_a_proj_with_mqa');w_kv_b=dequant8(f'{P}.self_attn.kv_b_proj')
w_o=dequant8(f'{P}.self_attn.o_proj')
ln_in=get(f'{P}.input_layernorm.weight');q_a_ln=get(f'{P}.self_attn.q_a_layernorm.weight')
kv_a_ln=get(f'{P}.self_attn.kv_a_layernorm.weight')
hs=rmsnorm(hidden_in,ln_in)

# cos/sin for NeoX (cat) and the building blocks
inv=1.0/(rope_theta**(torch.arange(0,qk_rope,2).float()/qk_rope))
freqs=torch.einsum('s,d->sd',pos,inv)        # [5,32]
emb=torch.cat((freqs,freqs),-1)              # [5,64]
cosN=emb.cos(); sinN=emb.sin()

q_resid=rmsnorm(w_q_a.new_tensor(hs[0]@w_q_a)[None] if False else (hs@w_q_a),q_a_ln)
q=(q_resid@w_q_b).view(B,S,n_heads,qk_head).transpose(1,2)  # [B,H,S,256]
q_nope,q_pe=torch.split(q,[qk_nope,qk_rope],dim=-1)
compressed=hs@w_kv_a
k_comp,k_pe=torch.split(compressed,[kv_lora,qk_rope],dim=-1)
k_comp=rmsnorm(k_comp,kv_a_ln)
kv_exp=(k_comp@w_kv_b).view(B,S,n_heads,qk_nope+v_head)
k_nope,value_states=torch.split(kv_exp,[qk_nope,v_head],dim=-1)
k_nope=k_nope.transpose(1,2); value_states=value_states.transpose(1,2)
k_pe=k_pe.view(B,1,S,qk_rope)
attn_mask=torch.full((S,S),float('-inf')).triu(1)[None,None]

def neox(x,cos,sin,ud):
    c=cos.unsqueeze(ud);s=sin.unsqueeze(ud)
    return x*c+rotate_half(x)*s
def interleaved(x,cos,sin,ud):
    # GPT-J interleaved: cos/sin use first-half freqs, pairs adjacent
    dd=x.shape[-1]; x_=x.float().reshape(*x.shape[:-1],dd//2,2)
    x1=x_[...,0];x2=x_[...,1]
    c=cos[...,:dd//2].unsqueeze(ud); s=sin[...,:dd//2].unsqueeze(ud)
    o1=x1*c-x2*s; o2=x1*s+x2*c
    return torch.stack([o1,o2],-1).reshape(*x.shape)

def run(rope):
    cosX=cosN[None]; sinX=sinN[None]      # [1,5,64]
    qpe=rope(q_pe,cosX,sinX,1)
    qs=torch.cat([q_nope,qpe],-1)
    kpe=rope(k_pe,cosX,sinX,1).expand(-1,n_heads,-1,-1)
    ks=torch.cat([k_nope,kpe],-1)
    aw=(qs@ks.transpose(2,3))*scaling+attn_mask
    aw=F.softmax(aw,dim=-1,dtype=DT); ao=aw@value_states
    aot=ao.transpose(1,2).reshape(B,S,n_heads*v_head)
    return (aot@w_o)[0], aw.max(-1).values.mean().item()

for name,fn in [('NeoX (correct)',neox),('Interleaved (bug)',interleaved)]:
    yo,peak=run(fn)
    cosv=[round(cossim(yo[i],hpu_attn[i]),4) for i in range(S)]
    rat=[round((yo[i].norm()/hpu_attn[i].norm()).item(),3) for i in range(S)]
    print(f'{name:20s} |attn|/pos={[round(x,2) for x in yo.norm(dim=-1).tolist()]}')
    print(f'{"":20s} cos_vs_HPU/pos={cosv}  ratio={rat}  softmax_peak={peak:.3f}')
print(f'{"HPU implied":20s} |attn|/pos={[round(x,2) for x in hpu_attn.norm(dim=-1).tolist()]}')

print()
print('='*90)
print('STRUCTURAL HYPOTHESIS SWEEP vs HPU implied attn_out (norm ~20/pos)')
print('='*90)
cosX=cosN[None]; sinX=sinN[None]
qpe=neox(q_pe,cosX,sinX,1); qs=torch.cat([q_nope,qpe],-1)
kpe=neox(k_pe,cosX,sinX,1).expand(-1,n_heads,-1,-1); ks=torch.cat([k_nope,kpe],-1)
aw=F.softmax((qs@ks.transpose(2,3))*scaling+attn_mask,dim=-1,dtype=DT)
ao=aw@value_states
aot=ao.transpose(1,2).reshape(B,S,n_heads*v_head)[0]   # [5,16384] pre-oproj

def C(name,t):
    cv=[round(cossim(t[i],hpu_attn[i]),4) for i in range(S)]
    print(f'{name:42s} |/pos={[round(x,2) for x in t.norm(dim=-1).tolist()]}  cos={cv}')

# H1: standard full attn + o_proj  (already shown, cos 0.11)
C('full-attn + o_proj (reference)', aot@w_o)
# H2: o_proj transposed
# H3: self-only attention (each token attends to itself only)
ao_self=value_states  # weights = identity -> attn_out = v at each position
aot_self=ao_self.transpose(1,2).reshape(B,S,n_heads*v_head)[0]
C('self-only-attn + o_proj', aot_self@w_o)
# H4: uniform causal attention
uni=torch.tril(torch.ones(S,S)); uni=uni/uni.sum(-1,keepdim=True)
ao_u=(uni[None,None]@value_states); aot_u=ao_u.transpose(1,2).reshape(B,S,n_heads*v_head)[0]
C('uniform-causal-attn + o_proj', aot_u@w_o)
# H5: attention WITHOUT rope at all (nope-only, pe zeroed)
qz=torch.cat([q_nope, torch.zeros_like(q_pe)],-1); kz=torch.cat([k_nope, torch.zeros_like(k_pe.expand(-1,n_heads,-1,-1))],-1)
awz=F.softmax((qz@kz.transpose(2,3))*scaling+attn_mask,dim=-1,dtype=DT)
aoz=(awz@value_states).transpose(1,2).reshape(B,S,n_heads*v_head)[0]
C('no-rope(pe=0)-attn + o_proj', aoz@w_o)
# H6: NEGATED rope sin (sign error in rotate_half)
def neox_negsin(x,c,s,ud): cc=c.unsqueeze(ud);ss=s.unsqueeze(ud);return x*cc - rotate_half(x)*ss
qpe2=neox_negsin(q_pe,cosX,sinX,1);kpe2=neox_negsin(k_pe,cosX,sinX,1).expand(-1,n_heads,-1,-1)
aw2=F.softmax((torch.cat([q_nope,qpe2],-1)@torch.cat([k_nope,kpe2],-1).transpose(2,3))*scaling+attn_mask,-1,dtype=DT)
ao2=(aw2@value_states).transpose(1,2).reshape(B,S,n_heads*v_head)[0]
C('neg-sin-rope + o_proj', ao2@w_o)
# H7: scale too small by sqrt(head) i.e. used 1/qk_head instead of 1/sqrt -> near uniform
aw3=F.softmax((qs@ks.transpose(2,3))*(1.0/qk_head)+attn_mask,-1,dtype=DT)
ao3=(aw3@value_states).transpose(1,2).reshape(B,S,n_heads*v_head)[0]
C('scale=1/qk_head(too small) + o_proj', ao3@w_o)
print('HPU implied: |/pos=%s'%[round(x,2) for x in hpu_attn.norm(dim=-1).tolist()])

print()
print('='*90)
print('POSITION-0 DETERMINISTIC DECOMPOSITION (attn weights are forced = identity at pos0)')
print('='*90)
hpu_attn0 = hpu_attn[0]   # [6144]  HPU implied attn output at pos 0
# At pos 0, attn_out = value_states[:,:,0,:] flattened over heads -> [16384]; then o_proj
v0 = value_states[0,:,0,:].reshape(-1)   # [64*256=16384]  per-head value at pos0
print('v0 (per-head value, pos0) norm=%.3f'%v0.norm())
ref_o0 = v0 @ w_o   # [6144]
print('ref o_proj(v0) norm=%.3f  cos_vs_HPU=%.4f'%(ref_o0.norm(), cossim(ref_o0, hpu_attn0)))
print('HPU attn0 norm=%.3f'%hpu_attn0.norm())

# Is HPU's pos0 = o_proj applied to a DIFFERENT head layout? Test reversed nope/v split in kv_b
# kv_b output per head is [nope(192)+v(256)]=448. What if HPU splits [v, nope] (wrong order)?
kv_exp_full=(rmsnorm(torch.split(hs@w_kv_a,[kv_lora,qk_rope],dim=-1)[0],kv_a_ln)@w_kv_b).view(B,S,n_heads,qk_nope+v_head)
# WRONG order: value first 256? No—dim is 448=192+256. If split [256,192] value would be first 256 dims.
v_wrong = kv_exp_full[0,:,:,:v_head][0]  # take first 256 dims as 'value' at pos0  -> wrong
# but that overlaps nope. Just test: value = LAST 256 (correct) vs FIRST 256 (wrong)
v_last = kv_exp_full[0,0,:,qk_nope:].reshape(-1)   # correct (192:448)
v_first= kv_exp_full[0,0,:,:v_head].reshape(-1)     # wrong (0:256)
print('value=last256 (correct): o_proj cos_vs_HPU=%.4f norm=%.3f'%(cossim(v_last@w_o,hpu_attn0),(v_last@w_o).norm()))
print('value=first256 (wrong) : o_proj cos_vs_HPU=%.4f norm=%.3f'%(cossim(v_first@w_o,hpu_attn0),(v_first@w_o).norm()))

# Maybe HPU residual_out scheme differs: maybe residual_out = hidden_in + attn AND it's already
# half because of fp16 routed_scaling? deepseek scales hidden by 1/routed_scaling_factor when fp16.
# Check config routed_scaling_factor:
rsf=cfg.get('routed_scaling_factor'); print('routed_scaling_factor=',rsf)
# If HPU is bf16, that fp16 branch does NOT fire. So no scaling expected.

# Decompose hpu_attn0 onto o_proj column space: solve least squares x@w_o = hpu_attn0 for x[16384]
# (find what attn_out the HPU o_proj WOULD need). Compare its norm to v0.
# w_o is [16384,6144]; we want x (1x16384) s.t. x@w_o ~ hpu_attn0. Use pinverse on w_o.T (6144x16384)
xhat = torch.linalg.lstsq(w_o.t().double(), hpu_attn0.double().unsqueeze(1)).solution.squeeze()  # [16384]
print('implied HPU pre-oproj attn_out norm=%.3f (vs ref v0 norm=%.3f)'%(xhat.norm(), v0.norm()))
print('cos(implied HPU pre-oproj, ref v0)=%.4f'%cossim(xhat.float(), v0))
# reshape per head and see if it matches v0 per-head up to permutation/scale
xh=xhat.float().view(n_heads,v_head); v0h=v0.view(n_heads,v_head)
percos=[round(cossim(xh[h],v0h[h]),3) for h in range(8)]
print('per-head cos(implied vs ref value) first 8 heads:',percos)
print('implied per-head norms first8:',[round(x,2) for x in xh.norm(dim=-1)[:8].tolist()])
print('ref     per-head norms first8:',[round(x,2) for x in v0h.norm(dim=-1)[:8].tolist()])

print()
print('='*90)
print('FREEZE / COLLAPSE SIGNATURE CHECK on HPU residual_out')
print('='*90)
ro=residual_out  # [5,6144] = HPU residual after attention
print('HPU residual_out per-pos norm:', [round(x,4) for x in ro.norm(dim=-1).tolist()])
print('pairwise cos between HPU residual_out positions:')
for i in range(S):
    print('  pos%d:'%i, [round(cossim(ro[i],ro[j]),4) for j in range(S)])
print()
hi=hidden_in[0]
print('HPU hidden_in (embed) per-pos norm:', [round(x,4) for x in hi.norm(dim=-1).tolist()])
print('cos(residual_out, hidden_in) per-pos (is residual ~ hidden_in?):',
      [round(cossim(ro[i],hi[i]),4) for i in range(S)])
ha=ro-hi  # implied attn
print('implied attn (ro-hi) per-pos norm:', [round(x,4) for x in ha.norm(dim=-1).tolist()])
print('pairwise cos between implied-attn positions (constant => frozen):')
for i in range(S):
    print('  pos%d:'%i, [round(cossim(ha[i],ha[j]),4) for j in range(S)])
# Also: hidden_out (mlp delta) collapse check
ho=d['hidden_out'][:5].to(DT)
print()
print('HPU hidden_out(mlp) per-pos norm:', [round(x,3) for x in ho.norm(dim=-1).tolist()])
print('pairwise cos between hidden_out(mlp) positions:')
for i in range(S):
    print('  pos%d:'%i, [round(cossim(ho[i],ho[j]),4) for j in range(S)])

print()
print('='*90)
print('CONTRAST: reference attention is content-aware (distinct per pos); HPU is collapsed')
print('='*90)
# reference full-attn o_proj output, per-position pairwise cos
ref_attn=(aot@w_o)  # wait aot is [5,16384]; need full per-pos. recompute properly:
cosX=cosN[None]; sinX=sinN[None]
qpe=neox(q_pe,cosX,sinX,1); qsr=torch.cat([q_nope,qpe],-1)
kper=neox(k_pe,cosX,sinX,1).expand(-1,n_heads,-1,-1); ksr=torch.cat([k_nope,kper],-1)
awr=F.softmax((qsr@ksr.transpose(2,3))*scaling+attn_mask,-1,dtype=DT)
aor=(awr@value_states).transpose(1,2).reshape(B,S,n_heads*v_head)[0]
refo=aor@w_o  # [5,6144]
print('REFERENCE attn_out pairwise cos between positions (should be DISTINCT/lower):')
for i in range(S):
    print('  pos%d:'%i,[round(cossim(refo[i],refo[j]),4) for j in range(S)])
print('REFERENCE attn softmax weights head0 (content-aware):')
for i in range(S): print('   pos',i,'->',[round(x,3) for x in awr[0,0,i,:].tolist()])
print()
# What does the HPU collapse most resemble? mean-of-values (uniform over all V incl future = bidirectional bug)?
mv_all=value_states.mean(dim=2)  # [B,H,256] mean over all positions
mv_flat=mv_all[0].reshape(-1)    # [16384]
print('mean-V-over-ALL-positions -> o_proj norm=%.3f, cos to HPU pos0=%.4f pos2=%.4f'%(
    (mv_flat@w_o).norm(), cossim(mv_flat@w_o, ha[0]), cossim(mv_flat@w_o, ha[2])))

print()
print('='*90)
print('IS IT o_proj OR value? Reference attn_pre_oproj per pos vs HPU output')
print('='*90)
# reference pre-o_proj attention output (the value-mixture) per position
ref_pre = aor   # [5,16384]  -- this is what o_proj is applied to in reference
print('ref pre-oproj norm/pos:', [round(x,3) for x in ref_pre.norm(dim=-1).tolist()])
print('HPU final attn norm/pos:', [round(x,3) for x in ha.norm(dim=-1).tolist()])
print('cos(ref_oproj,   HPU_final) per pos:', [round(cossim(refo[i], ha[i]),4) for i in range(S)])

# Direct: does HPU output = ref_pre projected by SOME consistent linear map? 
# Fit M (6144x6144 too big). Instead: per position, HPU = ref_pre @ Wo would need cos high if Wo=w_o.
# Already cos(refo,HPU)=0.11. Try: maybe HPU o_proj input has heads in TP-sharded WRONG order.
# TP=8: 64 heads -> 8 shards of 8 heads. o_proj is RowParallel: input split across TP by head-groups,
# partial sums all-reduced. A head-permutation in o_proj input would scramble.
# Test: o_proj with value heads REVERSED within each shard or shards reversed.
v0h = value_states[0,:,0,:]  # [64,256] pos0
# try reversing head order entirely
v0_rev = v0h.flip(0).reshape(-1)
print('o_proj(value pos0, heads reversed): cos to HPU pos0 =', round(cossim(v0_rev@w_o, ha[0]),4))
# try shard-internal layout: o_proj input expects [head0_dim0..255, head1..] our layout; that's standard.

# KEY: compare reference VALUE per head to what HPU would need. Check kv_b_proj value half vs the W_UV path.
# Maybe HPU uses W_UV (the absorbed value projection) differently. In prefill HPU uses kv_b_proj directly (correct).
# Let's just report the magnitude ratio cleanly.
print()
print('SUMMARY ratios (HPU/ref): norm pos0 = %.3f (HPU %.2f / ref-oproj %.2f)'%(ha[0].norm()/refo[0].norm(), ha[0].norm(), refo[0].norm()))
print('  ref pre-oproj norm pos0 = %.2f  (HPU matches PRE-oproj norm? %.3f)'%(ref_pre[0].norm(), ha[0].norm()/ref_pre[0].norm()))

print()
print('='*90); print('TP=8 o_proj HEAD-SHARD PERMUTATION TEST'); print('='*90)
# o_proj is RowParallelLinear: input dim = num_heads*v_head = 64*256=16384, split across TP=8.
# Each rank gets 8 heads (heads [r*8:(r+1)*8]) -> 2048 input dims, computes partial o_proj, all-reduced.
# w_o dequant is [16384, 6144] = [in, out]. Correct mapping: input head h occupies dims [h*256:(h+1)*256].
# A bug: if the o_proj WEIGHT rows were loaded in a different head order than the attention output heads.
v0h = value_states[0,:,0,:]   # [64,256] reference value at pos0 (saturated -> attn_out=value[0])
ref0 = v0h.reshape(-1) @ w_o
print('baseline cos(ref o_proj(v0), HPU0)=%.4f'%cossim(ref0, ha[0]))
import itertools
# Test: heads grouped into 8 shards of 8; permute SHARD order
TP=8; hps=8
best=None
v0r=v0h  # [64,256]
def perm_shards(order):
    blocks=[v0r[s*hps:(s+1)*hps] for s in range(TP)]
    return torch.cat([blocks[o] for o in order],0).reshape(-1)
# try reverse shard order and a few rolls
for desc,order in [('reverse',list(range(TP))[::-1]),
                   ('roll1',[(i+1)%TP for i in range(TP)]),
                   ('pairswap',[1,0,3,2,5,4,7,6])]:
    c=cossim(perm_shards(order)@w_o, ha[0]); print('shard perm %-9s cos=%.4f'%(desc,c))
# Test: within-shard head reversal
blocks=[v0r[s*hps:(s+1)*hps].flip(0) for s in range(TP)]
c=cossim(torch.cat(blocks,0).reshape(-1)@w_o, ha[0]); print('within-shard head flip   cos=%.4f'%c)
# Test: full head reverse
c=cossim(v0r.flip(0).reshape(-1)@w_o, ha[0]); print('full head reverse        cos=%.4f'%c)
# Test: o_proj weight itself reorganized -> reshape w_o as [64,256,6144], permute heads
woh=w_o.view(64,256,6144)
for desc,order in [('w reverse',list(range(64))[::-1])]:
    wperm=woh[order].reshape(16384,6144)
    c=cossim(v0r.reshape(-1)@wperm, ha[0]); print('w_o head perm %-9s cos=%.4f'%(desc,c))

print()
print('='*90); print('NORM-OFFSET HYPOTHESIS: HPU ran WITHOUT (1+weight) on some/all norms?'); print('='*90)
def rms_no1(x,w):
    v=x.to(DT).pow(2).mean(-1,keepdim=True); return w.to(DT)*(x.to(DT)*torch.rsqrt(v+eps))
def run_variant(in1, qa1, kva1, label):
    hs_=rmsnorm(hidden_in,ln_in) if in1 else rms_no1(hidden_in,ln_in)
    qres=(rmsnorm if qa1 else rms_no1)((hs_@w_q_a),q_a_ln)
    q_=(qres@w_q_b).view(B,S,n_heads,qk_head).transpose(1,2)
    qn,qp=torch.split(q_,[qk_nope,qk_rope],-1)
    comp=hs_@w_kv_a; kc,kp=torch.split(comp,[kv_lora,qk_rope],-1)
    kc=(rmsnorm if kva1 else rms_no1)(kc,kv_a_ln)
    kve=(kc@w_kv_b).view(B,S,n_heads,qk_nope+v_head)
    kn,vs=torch.split(kve,[qk_nope,v_head],-1); kn=kn.transpose(1,2); vs=vs.transpose(1,2)
    kp=kp.view(B,1,S,qk_rope)
    qp=neox(qp,cosN[None],sinN[None],1); kp=neox(kp,cosN[None],sinN[None],1).expand(-1,n_heads,-1,-1)
    qs2=torch.cat([qn,qp],-1); ks2=torch.cat([kn,kp],-1)
    aw=F.softmax((qs2@ks2.transpose(2,3))*scaling+attn_mask,-1,dtype=DT)
    ao=(aw@vs).transpose(1,2).reshape(B,S,n_heads*v_head)[0]
    out=ao@w_o
    cv=[round(cossim(out[i],ha[i]),4) for i in range(S)]
    print('%-40s norm/pos=%s cos=%s peak=%.3f'%(label,[round(x,2) for x in out.norm(dim=-1).tolist()],cv,aw.max(-1).values.mean()))
run_variant(True,True,True,  'ALL +1 (reference, correct)')
run_variant(True,False,False,'q_a & kv_a NO +1 (the 26x bug)')
run_variant(True,True,False, 'kv_a NO +1 only (the 167x latent bug)')
run_variant(False,False,False,'ALL NO +1 (raw weight norms)')
print('HPU implied: norm/pos=%s'%[round(x,2) for x in ha.norm(dim=-1).tolist()])
