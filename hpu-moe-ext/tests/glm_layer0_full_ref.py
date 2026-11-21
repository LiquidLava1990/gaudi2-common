"""
Full layer-0 reference recompute (pure CPU fp64) from hidden_in, with the CORRECT
o_proj dequant, compared against the HPU dump (residual_out / hidden_out).

o_proj weight in the dump matches the CORRECT per-rank dequant for r=4 (cos 0.9999987),
AND this dump is physically rank 4 (kv_b match), so the current fix
(offset = get_tensor_model_parallel_rank() = 4) produced the RIGHT o_proj weight.
=> The prior "OVERSHOT to 110" was a misdiagnosis. Now hunt the real divergence.

Tests:
  - full all-64-head attention -> o_proj -> all-reduce (the device sums all 8 ranks)
  - NeoX rotate-half rope  vs  GLM interleaved rope (rope_interleave=True in config)
  - compares attn_o (residual_out-hidden_in) and full layer (residual_out+hidden_out)
"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
L0 = '/opt/hpu-dsa-ext/layer0_dump.pt'
MHA = '/opt/hpu-dsa-ext/mha_dump.pt'
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']
scaling = qk_head ** -0.5
TP = 8; hpr = n_heads // TP; oin = hpr * v_head
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']

def get(n):
    with safe_open(f'{CKPT}/{idx[n]}', framework='pt', device='cpu') as f:
        return f.get_tensor(n)

def dq(p, gs=128, bits=8):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    c = torch.bitwise_right_shift(torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int16)
    c = torch.bitwise_and(c, (2 ** bits) - 1).to(DT); c = c.reshape(c.shape[0] * c.shape[1], c.shape[2])
    rz = torch.bitwise_right_shift(torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int16)
    rz = torch.bitwise_and(rz + 1, (2 ** bits) - 1).to(DT); z = rz.reshape(-1, rz.shape[1] * rz.shape[2])
    scf = sc.to(DT); g = (torch.arange(c.shape[0]) // gs).long(); return (c - z[g]) * scf[g]

def cs(a, b):
    a = a.reshape(-1); b = b.reshape(-1); return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

def rms(x, w):
    return (w.to(DT) + 1.0) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

# --- two rope conventions ---
def rh_neox(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)

def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim))
    return torch.einsum('s,d->sd', pos, inv)  # [S, dim/2]

def rope_neox(x, f):
    # x [1, h, S, dim]; f [S, dim/2]; NeoX: cos/sin duplicated [f,f]
    e = torch.cat((f, f), -1)             # [S, dim]
    c = e.cos()[None, None]; s = e.sin()[None, None]  # [1,1,S,dim]
    return x * c + rh_neox(x) * s

def rope_interleave(x, f):
    # GLM interleaved: pair (x0,x1),(x2,x3)... rotate each pair by its freq.
    # x [..., S, dim] with rope applied on last dim. f: [S, dim/2]
    xr = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    c = f.cos(); s = f.sin()
    # broadcast c,s [S,dim/2] onto xr [...,S,dim/2,2]
    while c.dim() < xr.dim() - 1:
        c = c.unsqueeze(0); s = s.unsqueeze(0)
    x0 = xr[..., 0]; x1 = xr[..., 1]
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return torch.stack((o0, o1), -1).reshape_as(x)

ld = torch.load(L0, map_location='cpu', weights_only=False)
hidden_in = ld['hidden_in'][0].to(DT)        # [128,6144]
positions = ld['positions'][0].to(DT)
residual_out = ld['residual_out'][0].to(DT)  # [128,6144]
hidden_out = ld['hidden_out'].to(DT)         # [128,6144]
hpu_attn = residual_out - hidden_in          # [128,6144]
S = 5

w_q_a = dq('model.layers.0.self_attn.q_a_proj'); w_q_b = dq('model.layers.0.self_attn.q_b_proj')
w_kv_a = dq('model.layers.0.self_attn.kv_a_proj_with_mqa'); w_kv_b = dq('model.layers.0.self_attn.kv_b_proj')
w_o = dq('model.layers.0.self_attn.o_proj')  # [16384,6144]
ln_in = get('model.layers.0.input_layernorm.weight')
ln_post = get('model.layers.0.post_attention_layernorm.weight')
q_a_ln = get('model.layers.0.self_attn.q_a_layernorm.weight')
kv_a_ln = get('model.layers.0.self_attn.kv_a_layernorm.weight')
gate_w = get('model.layers.0.mlp.gate_proj.weight').to(DT)
up_w = get('model.layers.0.mlp.up_proj.weight').to(DT)
down_w = get('model.layers.0.mlp.down_proj.weight').to(DT)

def attn_for(rope_mode):
    hs = rms(hidden_in[:S][None], ln_in)
    f = freqs(positions[:S], qk_rope)  # [S, qk_rope/2]
    q = (rms(hs @ w_q_a, q_a_ln) @ w_q_b).view(1, S, n_heads, qk_head).transpose(1, 2)  # [1,h,S,256]
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = hs @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
    kv = (kc @ w_kv_b).view(1, S, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1)
    kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)  # [1,h,S,*]
    if rope_mode == 'neox':
        qp = rope_neox(qp, f)  # qp [1,h,S,64]; f [S,32]
        kpe2 = rope_neox(kpe.view(1, 1, S, qk_rope), f)
    else:
        qp = rope_interleave(qp, f)
        kpe2 = rope_interleave(kpe.view(1, 1, S, qk_rope), f)
    qs = torch.cat([qn, qp], -1)
    kpe2 = kpe2.expand(-1, n_heads, -1, -1)
    ks = torch.cat([kno, kpe2], -1)
    mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)[None, None]
    aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
    vmix = (aw @ vst).transpose(1, 2).reshape(S, n_heads * v_head)  # [S,16384]
    attn_o = vmix @ w_o  # [S,6144]
    return attn_o, vmix

print('positions[:8]=', positions[:8].tolist())
print('hpu_attn norm/pos[:5]:', [round(hpu_attn[i].norm().item(), 3) for i in range(S)])
print()
for mode in ['neox', 'interleave']:
    attn_o, vmix = attn_for(mode)
    print('=== rope=%s ===' % mode)
    print('  ref attn_o norm/pos:', [round(attn_o[i].norm().item(), 3) for i in range(S)])
    print('  cos(attn_o, hpu_attn)/pos:', [round(cs(attn_o[i], hpu_attn[i]), 5) for i in range(S)])
    print('  |hpu|/|ref| pos:', [round((hpu_attn[i].norm() / attn_o[i].norm()).item(), 4) for i in range(S)])

# Use whichever rope matches better for the full-layer check
best = None; bestcos = -2
for mode in ['neox', 'interleave']:
    attn_o, vmix = attn_for(mode)
    c0 = cs(attn_o[0], hpu_attn[0])
    if c0 > bestcos: bestcos = c0; best = mode
print()
print('best rope =', best, 'cos pos0 =', round(bestcos, 5))
attn_o, vmix = attn_for(best)

# full layer
resid_after = hidden_in[:S] + attn_o
mn = rms(resid_after[None], ln_post)[0]
mlp_out = (F.silu(mn @ gate_w.t()) * (mn @ up_w.t())) @ down_w.t()
full_ref = resid_after + mlp_out

print()
print('=== residual_out (HPU) vs (hidden_in + ref attn_o) ===')
print('  cos/pos:', [round(cs(residual_out[i], resid_after[i]), 5) for i in range(S)])
print('  |hpu|/|ref| pos:', [round((residual_out[i].norm() / resid_after[i].norm()).item(), 4) for i in range(S)])
print()
print('=== hidden_out (HPU) vs ref mlp_out ===')
print('  cos/pos:', [round(cs(hidden_out[i], mlp_out[i]), 5) for i in range(S)])
print('  |hpu|/|ref| pos:', [round((hidden_out[i].norm() / mlp_out[i].norm()).item(), 4) for i in range(S)])
print()
print('=== FULL layer: HPU(residual_out+hidden_out) vs ref(resid_after+mlp_out) ===')
hpu_full = residual_out[:S] + hidden_out[:S]
print('  cos/pos:', [round(cs(hpu_full[i], full_ref[i]), 5) for i in range(S)])
print('  |hpu|/|ref| pos:', [round((hpu_full[i].norm() / full_ref[i].norm()).item(), 4) for i in range(S)])
