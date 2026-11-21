"""
FULL 78-layer independent fp64 forward from the embedding, A/B on rope convention
(neox rotate-half  vs  GLM interleaved), then model.norm + lm_head, reporting the
rank/logit of ' Paris' at the last position. Decides whether the live NeoX rope is
the salad bug. 5 real positions only.

Run mode via argv: 'neox' or 'interleave' (default runs both).
"""
import json, sys, torch, time
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']; scaling = qk_head ** -0.5
topk = cfg['num_experts_per_tok']; rsf = cfg['routed_scaling_factor']
norm_topk = cfg['norm_topk_prob']; fkd = cfg['first_k_dense_replace']; nL = cfg['num_hidden_layers']
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
S = 5
prompt_ids = [785, 6722, 315, 9621, 374]
paris = 12089

# open all shards once, cache handles
_handles = {}
def H_(fn):
    if fn not in _handles:
        _handles[fn] = safe_open(f'{CKPT}/{fn}', framework='pt', device='cpu')
    return _handles[fn]
def get(n):
    return H_(idx[n]).get_tensor(n)
def dq(p, gs=128, bits=8):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    c = torch.bitwise_right_shift(torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int32)
    c = torch.bitwise_and(c, (2 ** bits) - 1).to(DT); c = c.reshape(c.shape[0] * c.shape[1], c.shape[2])
    rz = torch.bitwise_right_shift(torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int32)
    rz = torch.bitwise_and(rz + 1, (2 ** bits) - 1).to(DT); z = rz.reshape(-1, rz.shape[1] * rz.shape[2])
    scf = sc.to(DT); g = (torch.arange(c.shape[0]) // gs).long(); return (c - z[g]) * scf[g]
def rms(x, w):
    return (w.to(DT) + 1.0) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim)); return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]; return x * c + rh(x) * s
def rope_inter(x, f):
    xr = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    c = f.cos(); s = f.sin()
    while c.dim() < xr.dim() - 1:
        c = c.unsqueeze(0); s = s.unsqueeze(0)
    x0 = xr[..., 0]; x1 = xr[..., 1]
    return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), -1).reshape_as(x)

d0 = torch.load('/opt/hpu-dsa-ext/layer0_dump.pt', map_location='cpu', weights_only=False)
positions = d0['positions'][0].to(DT)
embed_in = d0['hidden_in']; embed_in = (embed_in[0] if embed_in.dim()==3 else embed_in).to(DT)[:S]
f_all = freqs(positions[:S], qk_rope)

def attn(L, normed1, ropefn):
    P = f'model.layers.{L}.self_attn'
    w_q_a = dq(f'{P}.q_a_proj'); w_q_b = dq(f'{P}.q_b_proj')
    w_kv_a = dq(f'{P}.kv_a_proj_with_mqa'); w_kv_b = dq(f'{P}.kv_b_proj'); w_o = dq(f'{P}.o_proj')
    q_a_ln = get(f'{P}.q_a_layernorm.weight'); kv_a_ln = get(f'{P}.kv_a_layernorm.weight')
    q = (rms(normed1 @ w_q_a, q_a_ln) @ w_q_b).view(1, S, n_heads, qk_head).transpose(1, 2)
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = normed1 @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
    kv = (kc @ w_kv_b).view(1, S, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
    qp = ropefn(qp, f_all); kpe2 = ropefn(kpe.view(1,1,S,qk_rope), f_all).expand(-1,n_heads,-1,-1)
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

def run(ropefn, label):
    t0 = time.time()
    hidden = embed_in.clone(); residual = torch.zeros_like(hidden)
    for L in range(nL):
        ln_in = get(f'model.layers.{L}.input_layernorm.weight')
        ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
        r1 = hidden if L == 0 else hidden + residual
        normed1 = rms(r1[None], ln_in)[0]
        a = attn(L, normed1[None], ropefn)
        r2 = r1 + a
        normed2 = rms(r2[None], ln_post)[0]
        hidden = mlp(L, normed2); residual = r2
    final = hidden + residual
    norm_w = get('model.norm.weight').to(DT); lm = get('lm_head.weight').to(DT)
    normed = norm_w * (final * torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + eps))
    logits = normed @ lm.t()
    print(f'\n========== ROPE = {label}  ({time.time()-t0:.0f}s) ==========')
    print('final hidden norms:', [round(final[i].norm().item(),1) for i in range(S)])
    for i in range(S):
        top = torch.topk(logits[i], 6); toks = [tok.decode([int(t)]) for t in top.indices]
        probs = torch.softmax(logits[i].float(), -1)
        print(f'  pos{i} after {tok.decode([prompt_ids[i]])!r}: {[(toks[j], round(probs[int(top.indices[j])].item(),3)) for j in range(6)]}')
    lg4 = logits[4]; rank = (lg4 > lg4[paris]).sum().item()
    print(f'  >>> Paris rank at pos4 = {rank} / {lg4.numel()} ; Paris logit={lg4[paris].item():.2f} ; argmax={tok.decode([int(lg4.argmax())])!r}')

mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
if mode in ('neox', 'both'): run(rope_neox, 'neox (rotate-half)')
if mode in ('interleave', 'both'): run(rope_inter, 'interleaved (GLM)')
