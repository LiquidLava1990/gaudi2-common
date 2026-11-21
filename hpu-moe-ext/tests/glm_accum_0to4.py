"""
INDEPENDENT multi-layer accumulation test. Start from L0's dumped input (= embedding,
verified cos=1.0). Run a FULLY independent fp64 reference forward through layers
0,1,2,3 (dense MLP for 0/1/2, MoE for 3), computing our OWN residual stream (NOT
borrowing any HPU residual). Then compare our (hidden,residual) AFTER layer 3 to the
DUMPED L4 input (= dumped L3 output). cos<1 here = a real per-layer bug that the
single-layer check masked by borrowing R2_hpu. cos=1 => live agrees with reference,
the corruption is intrinsic (quantization / something upstream of L0).
"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']; scaling = qk_head ** -0.5
n_exp = cfg['n_routed_experts']; topk = cfg['num_experts_per_tok']
rsf = cfg['routed_scaling_factor']; norm_topk = cfg['norm_topk_prob']; fkd = cfg['first_k_dense_replace']
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
S = 5
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}', framework='pt', device='cpu') as f: return f.get_tensor(n)
def dq(p, gs=128, bits=8):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    c = torch.bitwise_right_shift(torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int32)
    c = torch.bitwise_and(c, (2 ** bits) - 1).to(DT); c = c.reshape(c.shape[0] * c.shape[1], c.shape[2])
    rz = torch.bitwise_right_shift(torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int32)
    rz = torch.bitwise_and(rz + 1, (2 ** bits) - 1).to(DT); z = rz.reshape(-1, rz.shape[1] * rz.shape[2])
    scf = sc.to(DT); g = (torch.arange(c.shape[0]) // gs).long(); return (c - z[g]) * scf[g]
def cs(a, b):
    a = a.reshape(-1).to(DT); b = b.reshape(-1).to(DT); return (a @ b / (a.norm() * b.norm() + 1e-30)).item()
def rms(x, w):
    return (w.to(DT) + 1.0) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim)); return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]; return x * c + rh(x) * s

d0 = torch.load('/opt/hpu-dsa-ext/layer0_dump.pt', map_location='cpu', weights_only=False)
positions = d0['positions'][0].to(DT)
hidden = d0['hidden_in']; hidden = (hidden[0] if hidden.dim()==3 else hidden).to(DT)[:S]  # embedding
residual = torch.zeros_like(hidden)
f = freqs(positions[:S], qk_rope)

def attn(L, normed1):
    P = f'model.layers.{L}.self_attn'
    w_q_a = dq(f'{P}.q_a_proj'); w_q_b = dq(f'{P}.q_b_proj')
    w_kv_a = dq(f'{P}.kv_a_proj_with_mqa'); w_kv_b = dq(f'{P}.kv_b_proj'); w_o = dq(f'{P}.o_proj')
    q_a_ln = get(f'{P}.q_a_layernorm.weight'); kv_a_ln = get(f'{P}.kv_a_layernorm.weight')
    q = (rms(normed1 @ w_q_a, q_a_ln) @ w_q_b).view(1, S, n_heads, qk_head).transpose(1, 2)
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = normed1 @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
    kv = (kc @ w_kv_b).view(1, S, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
    qp = rope_neox(qp, f); kpe2 = rope_neox(kpe.view(1,1,S,qk_rope), f).expand(-1,n_heads,-1,-1)
    qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
    mask = torch.full((S,S), float('-inf'), dtype=DT).triu(1)[None,None]
    aw = F.softmax((qs @ ks.transpose(2,3)) * scaling + mask, -1)
    vmix = (aw @ vst).transpose(1,2).reshape(S, n_heads*v_head)
    return vmix @ w_o

def mlp(L, normed2):
    MP = f'model.layers.{L}.mlp'
    if L < fkd:
        gate_w = get(f'{MP}.gate_proj.weight').to(DT); up_w = get(f'{MP}.up_proj.weight').to(DT); down_w = get(f'{MP}.down_proj.weight').to(DT)
        return (F.silu(normed2 @ gate_w.t()) * (normed2 @ up_w.t())) @ down_w.t()
    gate_w = get(f'{MP}.gate.weight').to(DT); ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)
    scores = torch.sigmoid(normed2 @ gate_w.t()); biased = scores + ebias[None]
    topk_ids = torch.topk(biased, topk, -1).indices; topk_w = scores.gather(1, topk_ids)
    if norm_topk: topk_w = topk_w / topk_w.sum(-1, keepdim=True)
    routed = torch.zeros(S, H, dtype=DT)
    cache = {}
    for eid in set(int(e) for e in topk_ids.reshape(-1).tolist()):
        cache[eid] = (dq(f'{MP}.experts.{eid}.gate_proj', bits=4), dq(f'{MP}.experts.{eid}.up_proj', bits=4), dq(f'{MP}.experts.{eid}.down_proj', bits=4))
    for i in range(S):
        acc = torch.zeros(H, dtype=DT)
        for k in range(topk):
            eid = int(topk_ids[i,k]); w = topk_w[i,k]; g,u,dn = cache[eid]
            acc = acc + w * ((F.silu(normed2[i]@g)*(normed2[i]@u))@dn)
        routed[i] = acc
    sg = dq(f'{MP}.shared_experts.gate_proj', bits=8); su = dq(f'{MP}.shared_experts.up_proj', bits=8); sd = dq(f'{MP}.shared_experts.down_proj', bits=8)
    shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd
    return shared + rsf * routed

# DeepseekV2DecoderLayer fused-residual forward:
#   normed1, residual = input_layernorm(hidden, residual)   [if residual is None: residual=hidden, normed1=norm(hidden)]
#   hidden = attn(normed1)
#   normed2, residual = post_attention_layernorm(hidden, residual)
#   hidden = mlp(normed2)
#   return hidden, residual
def layer(L, hidden, residual, first):
    ln_in = get(f'model.layers.{L}.input_layernorm.weight')
    ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
    if first:
        r1 = hidden
    else:
        r1 = hidden + residual
    normed1 = rms(r1[None], ln_in)[0]
    a = attn(L, normed1[None])
    r2 = r1 + a
    normed2 = rms(r2[None], ln_post)[0]
    m = mlp(L, normed2)
    return m, r2

dumps = {}
for L in [3,4,10,40,77]:
    dd = torch.load(f'/opt/hpu-dsa-ext/layer{L}_dump.pt', map_location='cpu', weights_only=False)
    fix = lambda t: None if t is None else (t[0] if t.dim()==3 else t).to(DT)
    dumps[L] = (fix(dd['hidden_in']), fix(dd['residual_in']))

print('Independent forward from embedding. Compare our (hidden,residual) entering layer N to dump.')
for L in range(0, 4):
    hidden, residual = layer(L, hidden, residual, first=(L==0))
    nxt = L + 1
    if nxt in dumps:
        dh, dr = dumps[nxt]
        print(f'after L{L} -> entering L{nxt}:')
        print(f'   cos(hidden, dump.hin)={[round(cs(hidden[i], dh[i]),5) for i in range(S)]}')
        print(f'   |hidden|/|dump.hin| ={[round((hidden[i].norm()/dh[i].norm()).item(),4) for i in range(S)]}')
        print(f'   cos(resid,  dump.rin)={[round(cs(residual[i], dr[i]),5) for i in range(S)]}')
        print(f'   |resid|/|dump.rin|  ={[round((residual[i].norm()/dr[i].norm()).item(),4) for i in range(S)]}')
