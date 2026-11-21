"""Teacher-forced fp64 CPU oracle for the shared serve drift.

ONE full-78-layer fp64 forward over (prompt + the served drifting completion),
greedy-argmax at each completion position vs the token the serve produced:
  - CPU argmax matches the drift -> int4 checkpoint quality floor (serve is RIGHT)
  - CPU argmax stays coherent     -> serving path bug to hunt
Reuses glm_full78_norm_ab.py math (PLUS1=False == coherent live config).
"""
import json, sys, time
import torch
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

PROMPT = "List the planets of the solar system in order:"
SERVED = " 1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6. Saturn 7. Uranus 8. Neptune 9. 2. Earth 4. Jupiter 5. 6. 8. 3. 7. 5. 6"
prompt_ids = tok(PROMPT)["input_ids"]
served_ids = tok(SERVED, add_special_tokens=False)["input_ids"]
ids = prompt_ids + served_ids
S = len(ids)
print(f"S={S} prompt={len(prompt_ids)} served={len(served_ids)}", flush=True)

_h = {}
def H_(fn):
    if fn not in _h: _h[fn] = safe_open(f'{CKPT}/{fn}', framework='pt', device='cpu')
    return _h[fn]
def get(n): return H_(idx[n]).get_tensor(n)
def dq(p, gs=128, bits=8):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    c = torch.bitwise_right_shift(torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int32)
    c = torch.bitwise_and(c, (2 ** bits) - 1).to(DT); c = c.reshape(c.shape[0] * c.shape[1], c.shape[2])
    rz = torch.bitwise_right_shift(torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int32)
    rz = torch.bitwise_and(rz + 1, (2 ** bits) - 1).to(DT); z = rz.reshape(-1, rz.shape[1] * rz.shape[2])
    scf = sc.to(DT); g = (torch.arange(c.shape[0]) // gs).long(); return (c - z[g]) * scf[g]
def rms(x, w):
    return w.to(DT) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim)); return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]; return x * c + rh(x) * s

emb = get('model.embed_tokens.weight')
embed_in = emb[torch.tensor(ids)].to(DT)
embed_in = torch.nan_to_num(embed_in)  # mirror HPU_SANITIZE_EMBED
positions = torch.arange(S, dtype=DT)
f_all = freqs(positions, qk_rope)

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
    qp = rope_neox(qp, f_all); kpe2 = rope_neox(kpe.view(1, 1, S, qk_rope), f_all).expand(-1, n_heads, -1, -1)
    qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
    mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)[None, None]
    aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
    vmix = (aw @ vst).transpose(1, 2).reshape(S, n_heads * v_head)
    return vmix @ w_o

def mlp(L, normed2):
    MP = f'model.layers.{L}.mlp'
    if L < fkd:
        gate_w = get(f'{MP}.gate_proj.weight').to(DT); up_w = get(f'{MP}.up_proj.weight').to(DT); down_w = get(f'{MP}.down_proj.weight').to(DT)
        return (F.silu(normed2 @ gate_w.t()) * (normed2 @ up_w.t())) @ down_w.t()
    gate_w = get(f'{MP}.gate.weight').to(DT); ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)
    scores = torch.sigmoid(normed2 @ gate_w.t()); biased = scores + ebias[None]
    topk_ids = torch.topk(biased, topk, -1).indices; topk_w = scores.gather(1, topk_ids)
    if norm_topk: topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
    routed = torch.zeros(S, H, dtype=DT); cache = {}
    for eid in set(int(e) for e in topk_ids.reshape(-1).tolist()):
        cache[eid] = (dq(f'{MP}.experts.{eid}.gate_proj', bits=4), dq(f'{MP}.experts.{eid}.up_proj', bits=4), dq(f'{MP}.experts.{eid}.down_proj', bits=4))
    for i in range(S):
        acc = torch.zeros(H, dtype=DT)
        for k in range(topk):
            eid = int(topk_ids[i, k]); w = topk_w[i, k]; g, u, dn = cache[eid]
            acc = acc + w * ((F.silu(normed2[i] @ g) * (normed2[i] @ u)) @ dn)
        routed[i] = acc
    sg = dq(f'{MP}.shared_experts.gate_proj', bits=8); su = dq(f'{MP}.shared_experts.up_proj', bits=8); sd = dq(f'{MP}.shared_experts.down_proj', bits=8)
    shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd
    return shared + rsf * routed

t0 = time.time()
hidden = embed_in.clone(); residual = torch.zeros_like(hidden)
for L in range(nL):
    ln_in = get(f'model.layers.{L}.input_layernorm.weight')
    ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
    r1 = hidden if L == 0 else hidden + residual
    normed1 = rms(r1[None], ln_in)[0]
    a = attn(L, normed1[None]); r2 = r1 + a
    normed2 = rms(r2[None], ln_post)[0]
    hidden = mlp(L, normed2); residual = r2
    print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)
final = hidden + residual
norm_w = get('model.norm.weight').to(DT); lm = get('lm_head.weight').to(DT)
normed = norm_w * (final * torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + eps))
logits = normed @ lm.t()

print(f"\n===== teacher-forced argmax vs served ({time.time()-t0:.0f}s) =====")
match = 0; total = 0
for i in range(len(prompt_ids) - 1, S - 1):
    am = int(logits[i].argmax()); nxt = ids[i + 1]; total += 1; match += int(am == nxt)
    mark = "==" if am == nxt else "!="
    print(f"pos {i:3d} ctx..{tok.decode([ids[i]])!r}: cpu {tok.decode([am])!r} {mark} served {tok.decode([nxt])!r}", flush=True)
print(f"\nmatched {match}/{total} — high match = quant floor (serve right); coherent-CPU mismatch at drift = serve bug")
