"""
CPU reference forward for GLM-5.1 layer 0, compared against the HPU dump.
Localizes a wrong-direction forward bug by per-intermediate cosine similarity.

Pure CPU. No HPU. Run via:
  docker exec -e CUDA_VISIBLE_DEVICES='' -e HABANA_VISIBLE_DEVICES='' \
              -e HABANA_VISIBLE_MODULES='' vllm-active \
              python3 /opt/hpu-moe-ext/tests/glm_layer0_ref_compare.py
"""
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
DUMP = '/opt/hpu-dsa-ext/layer0_dump.pt'

torch.set_grad_enabled(False)
DT = torch.float32

# ----------------------- config -----------------------
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']               # 6144
n_heads = cfg['num_attention_heads'] # 64
q_lora = cfg['q_lora_rank']          # 2048
kv_lora = cfg['kv_lora_rank']        # 512
qk_nope = cfg['qk_nope_head_dim']    # 192
qk_rope = cfg['qk_rope_head_dim']    # 64
qk_head = cfg['qk_head_dim']         # 256
v_head = cfg['v_head_dim']           # 256
eps = cfg['rms_norm_eps']            # 1e-5
rope_theta = cfg['rope_parameters']['rope_theta']  # 1e6
scaling = qk_head ** -0.5            # 256^-0.5

idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']


def get(name):
    shard = idx[name]
    with safe_open(f'{CKPT}/{shard}', framework='pt', device='cpu') as f:
        return f.get_tensor(name)


# proven 8-bit sym dequant (group_size=128, autogptq +1 zero) -> returns [in, out]
def dequant8(prefix, group_size=128, bits=8):
    qw = get(prefix + '.qweight')
    qz = get(prefix + '.qzeros')
    sc = get(prefix + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    codes = torch.bitwise_right_shift(
        torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16)
    codes = torch.bitwise_and(codes, (2 ** bits) - 1).to(DT)
    codes = codes.reshape(codes.shape[0] * codes.shape[1], codes.shape[2])  # [in, out]
    raw_z = torch.bitwise_right_shift(
        torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
    ).to(torch.int16)
    raw_z = raw_z + 1
    raw_z = torch.bitwise_and(raw_z, (2 ** bits) - 1).to(DT)
    zeros = raw_z.reshape(-1, raw_z.shape[1] * raw_z.shape[2])  # [groups, out]
    scf = sc.to(DT)
    g = (torch.arange(codes.shape[0]) // group_size).long()
    w = (codes - zeros[g]) * scf[g]   # [in, out]
    return w


# Linear with weight [in, out] (dequant orientation): y = x @ w
def lin_dq(x, w):
    return x @ w


def rmsnorm(x, w, plus_one):
    wf = w.to(DT)
    if plus_one:
        wf = wf + 1.0
    var = x.to(DT).pow(2).mean(-1, keepdim=True)
    xn = x.to(DT) * torch.rsqrt(var + eps)
    return wf * xn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, unsqueeze_dim):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def build_rope(positions, dim=qk_rope_head_dim if False else qk_rope):
    # NeoX: inv_freq over dim, emb = cat(freqs, freqs)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = positions.to(DT)  # [B, S]
    freqs = torch.einsum('s,d->sd', pos[0], inv_freq)  # [S, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)            # [S, dim]
    cos = emb.cos()[None]   # [1, S, dim]
    sin = emb.sin()[None]
    return cos, sin


def cossim(a, b):
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def relerr(a, b):
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


# ----------------------- load dump -----------------------
d = torch.load(DUMP, map_location='cpu', weights_only=False)
positions = d['positions'][:, :5].to(DT)          # [1, 5]
hidden_in = d['hidden_in'][:, :5, :].to(DT)       # [1, 5, 6144]
hidden_out_hpu = d['hidden_out'][:5].to(DT)       # [5, 6144]
residual_out_hpu = d['residual_out'][0, :5].to(DT)  # [5, 6144]

S = 5
B = 1

# ----------------------- load layer-0 weights -----------------------
P = 'model.layers.0'
w_q_a = dequant8(f'{P}.self_attn.q_a_proj')               # [6144, 2048]
w_q_b = dequant8(f'{P}.self_attn.q_b_proj')               # [2048, 64*256]
w_kv_a = dequant8(f'{P}.self_attn.kv_a_proj_with_mqa')    # [6144, 512+64]
w_kv_b = dequant8(f'{P}.self_attn.kv_b_proj')             # [512, 64*(192+256)]
w_o = dequant8(f'{P}.self_attn.o_proj')                   # [64*256, 6144]

ln_in = get(f'{P}.input_layernorm.weight')
ln_post = get(f'{P}.post_attention_layernorm.weight')
q_a_ln = get(f'{P}.self_attn.q_a_layernorm.weight')
kv_a_ln = get(f'{P}.self_attn.kv_a_layernorm.weight')

gate_w = get(f'{P}.mlp.gate_proj.weight').to(DT)   # [inter, hidden] (HF nn.Linear: [out, in])
up_w = get(f'{P}.mlp.up_proj.weight').to(DT)       # [inter, hidden]
down_w = get(f'{P}.mlp.down_proj.weight').to(DT)   # [hidden, inter]

print('dequant shapes:',
      'q_a', tuple(w_q_a.shape), 'q_b', tuple(w_q_b.shape),
      'kv_a', tuple(w_kv_a.shape), 'kv_b', tuple(w_kv_b.shape), 'o', tuple(w_o.shape))
print('mlp shapes: gate', tuple(gate_w.shape), 'up', tuple(up_w.shape), 'down', tuple(down_w.shape))

# ----------------------- forward -----------------------
cos, sin = build_rope(positions)   # [1, S, 64]

results = []  # (name, tensor_or_None_for_no_ref)

residual = hidden_in   # [1, S, 6144]

# 1. input_layernorm (+1)
hs = rmsnorm(hidden_in, ln_in, plus_one=True)
results.append(('post_input_layernorm', hs))

# ===== attention =====
# query path
q_resid = rmsnorm(lin_dq(hs, w_q_a), q_a_ln, plus_one=True)  # [B,S,2048]
q = lin_dq(q_resid, w_q_b)                                   # [B,S,64*256]
q = q.view(B, S, n_heads, qk_head).transpose(1, 2)          # [B,H,S,256]
q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)
q_pe = apply_rope(q_pe, cos, sin, unsqueeze_dim=1)
query_states = torch.cat([q_nope, q_pe], dim=-1)            # [B,H,S,256]

# kv path
compressed = lin_dq(hs, w_kv_a)                            # [B,S,512+64]
k_comp, k_pe = torch.split(compressed, [kv_lora, qk_rope], dim=-1)
k_comp = rmsnorm(k_comp, kv_a_ln, plus_one=True)          # [B,S,512]
kv_exp = lin_dq(k_comp, w_kv_b)                            # [B,S,64*(192+256)]
kv_exp = kv_exp.view(B, S, n_heads, qk_nope + v_head)
k_nope, value_states = torch.split(kv_exp, [qk_nope, v_head], dim=-1)
k_nope = k_nope.transpose(1, 2)                            # [B,H,S,192]
value_states = value_states.transpose(1, 2)               # [B,H,S,256]
k_pe = k_pe.view(B, 1, S, qk_rope)
k_pe = apply_rope(k_pe, cos, sin, unsqueeze_dim=1)
k_pe = k_pe.expand(-1, n_heads, -1, -1)
key_states = torch.cat([k_nope, k_pe], dim=-1)            # [B,H,S,256]

# causal SDPA (no indexer / no DSA mask for dense full attention; topk>=S)
attn_mask = torch.full((S, S), float('-inf')).triu(1)[None, None]  # causal
attn_weights = (query_states @ key_states.transpose(2, 3)) * scaling + attn_mask
attn_weights = F.softmax(attn_weights, dim=-1, dtype=DT)
attn_out = attn_weights @ value_states                    # [B,H,S,256]
attn_out_t = attn_out.transpose(1, 2).reshape(B, S, n_heads * v_head)  # [B,S,64*256]
results.append(('attn_pre_o_proj', attn_out_t))

attn_o = lin_dq(attn_out_t, w_o)                          # [B,S,6144]
results.append(('attn_post_o_proj', attn_o))

# residual add
hs2 = residual + attn_o                                   # [B,S,6144]
results.append(('post_attn_residual', hs2))

# ===== MLP (dense) =====
residual2 = hs2
mn = rmsnorm(hs2, ln_post, plus_one=True)
# HF nn.Linear weight is [out, in]; y = x @ w.t()
gate = mn @ gate_w.t()
up = mn @ up_w.t()
mlp_inter = F.silu(gate) * up
mlp_out = mlp_inter @ down_w.t()                          # [B,S,6144]
results.append(('mlp_output', mlp_out))

final = residual2 + mlp_out                               # [B,S,6144]
results.append(('final_layer_output', final))

# ----------------------- compare -----------------------
# residual_out_hpu is what the layer feeds forward as residual (post-attn residual in vLLM fused-add scheme).
# hidden_out_hpu is the layer output (post-MLP). Compare both interpretations.

print()
print('=' * 96)
print(f'{"intermediate":28s} {"cos(vs HPU final out)":>22s} {"relerr":>12s}  {"norm_cpu":>10s} {"norm_ref":>10s}')
print('-' * 96)
ref_final = hidden_out_hpu  # [5,6144]
for name, t in results:
    tt = t[0] if t.dim() == 3 else t  # -> [S, dim]
    if tt.shape == ref_final.shape:
        cs = cossim(tt, ref_final)
        re = relerr(tt, ref_final)
        print(f'{name:28s} {cs:22.6f} {re:12.4f}  {tt.norm().item():10.3f} {ref_final.norm().item():10.3f}')
    else:
        print(f'{name:28s} {"(shape "+str(tuple(tt.shape))+")":>22s}')

print()
print('=' * 96)
print('CORRECT vLLM fused-residual interpretation:')
print('  HPU residual_out = attn_out + hidden_in   (the running residual after attention)')
print('  HPU hidden_out   = mlp_out (the MLP delta, NOT full output)')
print('  full layer output = hidden_out + residual_out')
print('=' * 96)
cpu_resid_after_attn = hs2[0]           # attn_out + hidden_in   -> compare to HPU residual_out
cpu_mlp = mlp_out[0]                     # mlp delta              -> compare to HPU hidden_out
cpu_full = final[0]                      # full output
hpu_full = hidden_out_hpu + residual_out_hpu

def line(label, a, b):
    print(f'{label:46s} cos={cossim(a,b): .6f}  relerr={relerr(a,b):8.4f}  |cpu|={a.norm().item():9.3f} |hpu|={b.norm().item():9.3f}')

print('--- ATTENTION path (isolated) ---')
line('CPU(attn_out+hidden_in) vs HPU residual_out', cpu_resid_after_attn, residual_out_hpu)
line('CPU attn_out(post o_proj) vs HPU(resid-hin)', attn_o[0], residual_out_hpu - hidden_in[0])
print('--- MLP path (isolated) ---')
line('CPU mlp_out vs HPU hidden_out', cpu_mlp, hidden_out_hpu)
print('--- FULL layer output ---')
line('CPU full vs HPU full(hidden_out+residual_out)', cpu_full, hpu_full)
print()
print('norms per-pos:')
print('  hidden_in (embed)     :', [round(x,4) for x in hidden_in[0].norm(dim=-1).tolist()])
print('  HPU residual_out      :', [round(x,3) for x in residual_out_hpu.norm(dim=-1).tolist()])
print('  CPU attn_out+hidden_in:', [round(x,3) for x in cpu_resid_after_attn.norm(dim=-1).tolist()])
print('  HPU hidden_out(mlp)   :', [round(x,3) for x in hidden_out_hpu.norm(dim=-1).tolist()])
print('  CPU mlp_out           :', [round(x,3) for x in cpu_mlp.norm(dim=-1).tolist()])
print('  CPU post_input_LN     :', [round(x,3) for x in hs[0].norm(dim=-1).tolist()])

# ===================== ATTENTION INTROSPECTION =====================
print()
print('=' * 96)
print('ATTENTION INTERNALS (CPU reference)')
print('=' * 96)
print('q_resid  norm/pos:', [round(x,3) for x in q_resid[0].norm(dim=-1).tolist()])
print('query_states |q|/head(pos0):', [round(x,2) for x in query_states[0,:4,0,:].norm(dim=-1).tolist()], '... (first 4 heads)')
print('q_nope absmax', round(q_nope.abs().max().item(),3), 'q_pe absmax', round(q_pe.abs().max().item(),3))
print('k_compressed norm/pos:', [round(x,3) for x in k_comp[0].norm(dim=-1).tolist()])
print('k_nope absmax', round(k_nope.abs().max().item(),3), 'k_pe absmax', round(k_pe.abs().max().item(),3))
print('value_states absmax', round(value_states.abs().max().item(),3), '|v|/head(pos0,h0):', round(value_states[0,0,0,:].norm().item(),3))
print('scaling =', scaling)
# attn weights row sums and a peek
print('attn_weights row0 head0 (pos i attends to j<=i):')
for i in range(S):
    print('   pos', i, '->', [round(x,4) for x in attn_weights[0,0,i,:].tolist()])
print('attn_out absmax', round(attn_out.abs().max().item(),3), '|attn_out|/head(pos0):', [round(x,3) for x in attn_out[0,:4,0,:].norm(dim=-1).tolist()])
print('attn_out_t (reshaped) norm/pos:', [round(x,3) for x in attn_out_t[0].norm(dim=-1).tolist()])
print('o_proj weight |w| fro:', round(w_o.norm().item(),3), 'mean', round(w_o.mean().item(),5), 'std', round(w_o.std().item(),5))
print('attn_o (post o_proj) norm/pos:', [round(x,3) for x in attn_o[0].norm(dim=-1).tolist()])

# Diagnostic: what does HPU attn_out look like (= residual_out - hidden_in)?
hpu_attn = residual_out_hpu - hidden_in[0]
print()
print('HPU implied attn_out (residual_out - hidden_in) norm/pos:', [round(x,3) for x in hpu_attn.norm(dim=-1).tolist()])
print('cos(CPU attn_o, HPU attn_out) per-pos:')
for i in range(S):
    a=attn_o[0,i].double(); b=hpu_attn[i].double()
    print('   pos', i, 'cos=', round((a@b/(a.norm()*b.norm())).item(),4), 'ratio|cpu|/|hpu|=', round((a.norm()/b.norm()).item(),3))

# ===================== ROPE / SPLIT DIAGNOSTICS =====================
print()
print('=' * 96)
print('ROPE / SPLIT DIAGNOSTICS')
print('=' * 96)
# q before rope: view per-head, look at per-channel magnitude across the 256 dims
q_raw = lin_dq(q_resid, w_q_b).view(B, S, n_heads, qk_head)   # [B,S,H,256]
# per-channel RMS over (B,S,H)
ch_rms = q_raw.reshape(-1, qk_head).pow(2).mean(0).sqrt()     # [256]
print('q per-channel RMS: dims[0:192] (nope) mean=%.3f max=%.3f | dims[192:256] (rope) mean=%.3f max=%.3f'
      % (ch_rms[:qk_nope].mean(), ch_rms[:qk_nope].max(), ch_rms[qk_nope:].mean(), ch_rms[qk_nope:].max()))
print('  full q per-channel RMS first 8:', [round(x,3) for x in ch_rms[:8].tolist()])
print('  full q per-channel RMS dims 188..200:', [round(x,3) for x in ch_rms[188:200].tolist()])
print('  full q per-channel RMS last 8:', [round(x,3) for x in ch_rms[-8:].tolist()])

# kv_b output per-head: 448 = 192 nope + 256 v
kv_raw = lin_dq(k_comp, w_kv_b).view(B, S, n_heads, qk_nope+v_head)
kch = kv_raw.reshape(-1, qk_nope+v_head).pow(2).mean(0).sqrt()
print('kv_b per-channel RMS: nope[0:192] mean=%.3f max=%.3f | v[192:448] mean=%.3f max=%.3f'
      % (kch[:qk_nope].mean(), kch[:qk_nope].max(), kch[qk_nope:].mean(), kch[qk_nope:].max()))

# k_pe (the rope channel from kv_a) BEFORE rope
compressed_dbg = lin_dq(hs, w_kv_a)
kpe_raw = compressed_dbg[..., kv_lora:]   # [B,S,64]
print('k_pe raw (pre-rope) per-channel RMS mean=%.3f max=%.3f' % (kpe_raw.reshape(-1,qk_rope).pow(2).mean(0).sqrt().mean(), kpe_raw.abs().max()))

# QK logit breakdown: nope-vs-pe contribution
qn = q_nope          # [B,H,S,192]
qp = q_pe            # [B,H,S,64] (post rope)
kn = k_nope          # [B,H,S,192]
kp = k_pe            # [B,H,S,64] (post rope)
logit_nope = torch.einsum('bhsd,bhtd->bhst', qn, kn) * scaling
logit_pe   = torch.einsum('bhsd,bhtd->bhst', qp, kp) * scaling
print('logit contribution scale: |nope part| absmax=%.2f  |pe part| absmax=%.2f' % (logit_nope.abs().max(), logit_pe.abs().max()))
print('  full logit absmax=%.2f (diag should be biggest)' % (logit_nope+logit_pe).abs().max())
print('  sample full logit head0 pos4 row:', [round(x,2) for x in (logit_nope+logit_pe)[0,0,4,:].tolist()])

# ===================== O_PROJ + SATURATION CHECK =====================
print()
print('='*96); print('O_PROJ + SATURATION CHECK'); print('='*96)
# The pre-o_proj magnitude matches HPU. The blow-up is at o_proj (CPU 2.6x).
# Hypothesis A: o_proj weight orientation. dequant gives [in, out]=[16384,6144]; y=x@w. correct?
# x is attn_out_t [B,S,16384]; w_o [16384,6144]. y=[B,S,6144]. Sanity: w_o.t() instead?
y1 = attn_out_t @ w_o            # [B,S,6144]
y2 = attn_out_t @ w_o.t() if w_o.shape[0]==w_o.shape[1] else None
print('o_proj y=x@w  norm/pos:', [round(x,3) for x in y1[0].norm(dim=-1).tolist()])
hpu_attn = residual_out_hpu - hidden_in[0]
print('HPU implied attn_out norm/pos:', [round(x,3) for x in hpu_attn.norm(dim=-1).tolist()])
print('cos(y=x@w, HPU)/pos:', [round((y1[0,i].double()@hpu_attn[i].double()/(y1[0,i].norm()*hpu_attn[i].norm())).item(),4) for i in range(S)])

# Hypothesis B: attention is saturated (one-hot). Maybe HPU softmax not saturated because
# HPU uses a different (correct) scale OR the q/k assembled differently.
# Test: NON-saturated reference — what if pe and nope are concatenated in opposite ORDER for o-proj head layout?
# More important: check whether o_proj expects head order [h0_v(256), h1_v(256)...] (our layout) — yes attn_out.transpose(1,2).reshape.

# Hypothesis C: the value/attn magnitude. attn_out_t matches HPU pre-oproj. So o_proj weight is the only diff.
# Compare o_proj column norms to see if dequant produced a 2.6x-too-large weight.
print('w_o stats: fro=%.3f  per-output-col RMS mean=%.4f' % (w_o.norm(), w_o.pow(2).mean(0).sqrt().mean()))
# A random projection x[16384]@w[16384,6144]: output RMS ~ |x|_rms * |w_col|_rms * sqrt(16384)
x_rms = attn_out_t[0].pow(2).mean().sqrt()
wcol_rms = w_o.pow(2).mean(0).sqrt().mean()
print('predicted o_proj out RMS ~ x_rms(%.4f)*wcol_rms(%.5f)*sqrt(16384)=%.4f -> norm/pos=%.2f'
      % (x_rms, wcol_rms, x_rms*wcol_rms*(16384**0.5), x_rms*wcol_rms*(16384**0.5)*(6144**0.5)))

# DECISIVE: is the attention saturated on HPU too? We cannot see HPU softmax, but we CAN test:
# if we DON'T saturate (e.g. correct scale), does o_proj output magnitude drop to ~20?
# Re-run attention with attn fully UNIFORM (avg over causal) to bound the value contribution:
uni = torch.tril(torch.ones(S,S)); uni = uni/uni.sum(-1,keepdim=True)
attn_uni = (uni[None,None] @ value_states)
attn_uni_t = attn_uni.transpose(1,2).reshape(B,S,n_heads*v_head)
yu = attn_uni_t @ w_o
print('UNIFORM-attn o_proj out norm/pos:', [round(x,3) for x in yu[0].norm(dim=-1).tolist()])

# ===================== ROPE STYLE TEST (NeoX vs interleaved) =====================
print()
print('='*96); print('ROPE STYLE A/B TEST  (does HPU match NeoX or interleaved?)'); print('='*96)
hpu_attn = residual_out_hpu - hidden_in[0]   # [S,6144]

def rope_interleaved(x, cos, sin, ud):
    # GPT-J/interleaved: pairs (x0,x1),(x2,x3)... rotated. cos/sin built from cat(freqs,freqs) won't match;
    # interleaved needs cos/sin of shape [.,dim] where each freq repeats adjacent. Build proper interleaved.
    # x: [...,D]; reshape to [...,D/2,2]
    d = x.shape[-1]
    x_ = x.float().reshape(*x.shape[:-1], d//2, 2)
    x1 = x_[..., 0]; x2 = x_[..., 1]
    # cos/sin here are [1,S,D] from cat(freqs,freqs); take first half = the per-pair freq
    c = cos[..., :d//2].unsqueeze(ud)  # [1,1,S,D/2] after unsqueeze
    s = sin[..., :d//2].unsqueeze(ud)
    o1 = x1*c - x2*s
    o2 = x1*s + x2*c
    out = torch.stack([o1,o2], dim=-1).reshape(*x.shape)
    return out

def run_attn(rope_fn):
    qpe = rope_fn(q_nope.new_tensor(q[..., qk_nope:]) if False else q[..., qk_nope:].clone(), cos, sin, 1) if rope_fn is not apply_rope else apply_rope(q[...,qk_nope:].clone(), cos, sin, 1)
    qn2 = q[..., :qk_nope]
    qs = torch.cat([qn2, qpe], dim=-1)
    kpe2 = k_pe_raw_t.clone()
    if rope_fn is apply_rope:
        kpe_a = apply_rope(kpe2, cos, sin, 1)
    else:
        kpe_a = rope_fn(kpe2, cos, sin, 1)
    kpe_a = kpe_a.expand(-1, n_heads, -1, -1)
    ks = torch.cat([k_nope, kpe_a], dim=-1)
    aw = (qs @ ks.transpose(2,3))*scaling + attn_mask
    aw = F.softmax(aw, dim=-1, dtype=DT)
    ao = aw @ value_states
    aot = ao.transpose(1,2).reshape(B,S,n_heads*v_head)
    yo = aot @ w_o
    return yo[0], aw

# rebuild q/k base (pre-rope) tensors
q = lin_dq(q_resid, w_q_b).view(B,S,n_heads,qk_head).transpose(1,2)
k_pe_raw_t = compressed[..., kv_lora:].view(B,1,S,qk_rope)

for label, fn in [('NeoX rotate_half', apply_rope), ('Interleaved (GPT-J)', rope_interleaved)]:
    yo, aw = run_attn(fn)
    coss = [round((yo[i].double()@hpu_attn[i].double()/(yo[i].norm()*hpu_attn[i].norm())).item(),4) for i in range(S)]
    sat = aw.max(dim=-1).values.mean().item()  # ~1.0 = saturated
    print(f'{label:22s} attn_out norm/pos={[round(x,2) for x in yo.norm(dim=-1).tolist()]}  cos_vs_HPU={coss}  softmax_peakiness={sat:.3f}')
print('HPU implied attn_out norm/pos:', [round(x,2) for x in hpu_attn.norm(dim=-1).tolist()])

# ===================== ROOT-CAUSE: is pe-dominance real? compare to a CLEAN HF instantiation =====================
print()
print('='*96); print('GROUND-TRUTH HF GlmMoeDsaAttention forward (eager, fp32, +1 norms)'); print('='*96)
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
hfcfg = AutoConfig.from_pretrained(CKPT)
hfcfg._attn_implementation = 'eager'
attn = M.GlmMoeDsaAttention(hfcfg, layer_idx=0).to(DT).eval()
# load dequanted weights (dequant is [in,out]; nn.Linear.weight is [out,in] -> transpose)
sd = {}
sd['q_a_proj.weight']  = w_q_a.t().contiguous()
sd['q_b_proj.weight']  = w_q_b.t().contiguous()
sd['kv_a_proj_with_mqa.weight'] = w_kv_a.t().contiguous()
sd['kv_b_proj.weight'] = w_kv_b.t().contiguous()
sd['o_proj.weight']    = w_o.t().contiguous()
sd['q_a_layernorm.weight']  = (q_a_ln.to(DT)+1.0)
sd['kv_a_layernorm.weight'] = (kv_a_ln.to(DT)+1.0)
missing, unexpected = attn.load_state_dict(sd, strict=False)
# zero indexer (unused for dense path; but skip_topk may be False -> indexer runs). Force full attention.
print('missing(non-indexer):', [m for m in missing if 'indexer' not in m])
# Build rotary via HF and run with a causal mask, bypassing indexer by monkeypatching topk to full
rot = M.GlmMoeDsaRotaryEmbedding(hfcfg)
posids = positions.long()
cos_hf, sin_hf = rot(hidden_in, posids)
# causal mask 4D [B,1,S,S]
cm = torch.full((B,1,S,S), 0.0); cm = cm.masked_fill(torch.triu(torch.ones(S,S,dtype=torch.bool),1)[None,None], float('-inf'))
# Force indexer to select ALL tokens (skip DSA) so it equals full causal attention:
attn.indexer.index_topk = 10**9
# also need indexer weights; they are random-init -> but with index_topk>=S it selects everything regardless of scores
attn.config._attn_implementation = 'eager'
with torch.no_grad():
    out, _, _ = attn(hs.to(DT), (cos_hf, sin_hf), cm, past_key_values=None, prev_topk_indices=None)
hf_attn_out = out[0]   # [S,6144]
print('HF attn_out norm/pos:', [round(x,3) for x in hf_attn_out.norm(dim=-1).tolist()])
print('HPU implied attn_out norm/pos:', [round(x,3) for x in hpu_attn.norm(dim=-1).tolist()])
print('cos(HF attn, HPU)/pos:', [round((hf_attn_out[i].double()@hpu_attn[i].double()/(hf_attn_out[i].norm()*hpu_attn[i].norm())).item(),4) for i in range(S)])
print('cos(HF attn, my-CPU attn_o)/pos:', [round((hf_attn_out[i].double()@attn_o[0,i].double()/(hf_attn_out[i].norm()*attn_o[0,i].norm())).item(),4) for i in range(S)])

# ===================== vLLM ROPE OP TEST (the actual HPU rope) =====================
print()
print('='*96); print('vLLM get_rope op (is_neox_style=True) vs HF NeoX rotate_half'); print('='*96)
from vllm.model_executor.layers.rotary_embedding import get_rope as vget_rope
rope_theta_i = int(rope_theta)
# Build the rope exactly as deepseek_v2 does, then patched to neox=True
rope_params = {'rope_type':'default','rope_theta':rope_theta}
try:
    vrope = vget_rope(qk_rope, max_position=cfg['max_position_embeddings'],
                      rope_parameters=rope_params, is_neox_style=True)
except TypeError:
    # older signature
    vrope = vget_rope(qk_rope, rotary_dim=qk_rope, max_position=cfg['max_position_embeddings'],
                      base=rope_theta_i, is_neox_style=True)
posids = positions[0].long()  # [5]
# vLLM rope expects q [num_tokens, num_heads*head] or [tokens,heads,dim]; here it's applied to the rope slice
# Replicate wrapper call: q[..., nope:] shape [tokens, heads, 64]; k_pe [tokens,1,64]
qfull = lin_dq(q_resid, w_q_b).view(S, n_heads, qk_head)   # [5,64,256]
qrope_slice = qfull[..., qk_nope:].clone()                  # [5,64,64]
kpe_slice = compressed[0][..., kv_lora:].view(S,1,qk_rope).clone()  # [5,1,64]
qv = qrope_slice.clone(); kv_ = kpe_slice.clone()
q_out, k_out = vrope(posids, qv, kv_)
# Reference NeoX on same:
cos2, sin2 = build_rope(positions)   # [1,S,64]
qref = apply_rope(qrope_slice.unsqueeze(0).transpose(1,2), cos2, sin2, 1).transpose(1,2)[0]  # mimic [S,H,64]
# easier: apply per-token
def neox_tok(x):  # x [S, H, 64]
    c = cos2[0].unsqueeze(1); s = sin2[0].unsqueeze(1)  # [S,1,64]
    return x*c + rotate_half(x)*s
qref2 = neox_tok(qrope_slice)
kref2 = neox_tok(kpe_slice)
print('vLLM-rope q vs HF-neox q: max abs diff =', (q_out-qref2).abs().max().item(), 'cos=', cossim(q_out, qref2))
print('vLLM-rope k vs HF-neox k: max abs diff =', (k_out-kref2).abs().max().item(), 'cos=', cossim(k_out, kref2))
print('vLLM rope is_neox_style attr:', getattr(vrope,'is_neox_style',None))
