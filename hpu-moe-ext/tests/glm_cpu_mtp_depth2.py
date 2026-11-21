"""MTP depth-2 acceptance oracle — quantifies the num_speculative_tokens=2 gain.

Extends glm_cpu_mtp_bf16.py: in addition to the depth-1 draft (MTP head on the
main-model hidden + true next-token embedding), it simulates the proposer's
AUTOREGRESSIVE 2nd step (hpu_eagle.py:86-135): re-run the SAME layer-78 MTP block
on its OWN output hidden + the depth-1 drafted token's embedding, predicting the
+3 token. This is the real recurrence the HPU proposer runs for depth>1.

Reports:
  a1  = depth-1 per-token accept rate  (== the validated ~62-75%)
  a2  = depth-2 conditional accept rate (draft2 == target | draft1 accepted)
  E[len] = 1 + a1 + a1*a2   (expected accepted tokens / forward step)
and the implied single-stream multiplier vs depth-1 (1+a1).

Run: docker run --rm -v /mnt/fastmodels:/models:ro -v $HOME/AI/hpu-moe-ext:/w \
       --entrypoint python3 vllm-gaudi:next /w/tests/glm_cpu_mtp_depth2.py
"""
import json, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
ORIG = '/models/glm51-orig-l78'
CACHE = '/w/tests/.mtp_hidden_cache.pt'
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']; scaling = qk_head ** -0.5
topk = cfg['num_experts_per_tok']; rsf = cfg['routed_scaling_factor']
norm_topk = cfg['norm_topk_prob']; nL = cfg['num_hidden_layers']
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
oidx = json.load(open(f'{ORIG}/model.safetensors.index.json'))['weight_map']
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)

PROMPT = "List the planets of the solar system in order:"
SERVED = " 1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6. 7. 8. 2. 9. 6. 5. 3. 4. "
prompt_ids = tok(PROMPT)["input_ids"]
served_ids = tok(SERVED, add_special_tokens=False)["input_ids"]
ids = prompt_ids + served_ids
S = len(ids)

_h = {}
def H_(path, fn):
    key = (path, fn)
    if key not in _h: _h[key] = safe_open(f'{path}/{fn}', framework='pt', device='cpu')
    return _h[key]
def get(n): return H_(CKPT, idx[n]).get_tensor(n)
def g78(n): return H_(ORIG, oidx[n]).get_tensor(n)
def w78(n): return g78(n).to(DT).t()
def rms(x, w): return w.to(DT) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim)); return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]; return x * c + rh(x) * s

d = torch.load(CACHE)
final, logits_main = d['final'], d['logits_main']
emb = get('model.embed_tokens.weight')
lm = get('lm_head.weight').to(DT)
L = nL; P = f'model.layers.{L}'; A = f'{P}.self_attn'; MP = f'{P}.mlp'

# preload MTP layer-78 weights once
W = {n: w78(f'{A}.{n}.weight') for n in
     ['q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj']}
NRM = {n: g78(f'{P}.{n}.weight') if 'layernorm' in n or n in ('enorm','hnorm') else None for n in []}
gate_w = g78(f'{MP}.gate.weight').to(DT); ebias = g78(f'{MP}.gate.e_score_correction_bias').to(DT)
shared_w = {n: w78(f'{MP}.shared_experts.{n}.weight') for n in ['gate_proj', 'up_proj', 'down_proj']}
_ecache = {}
def expert(eid):
    if eid not in _ecache:
        _ecache[eid] = (w78(f'{MP}.experts.{eid}.gate_proj.weight'),
                        w78(f'{MP}.experts.{eid}.up_proj.weight'),
                        w78(f'{MP}.experts.{eid}.down_proj.weight'))
    return _ecache[eid]

def mtp_block(en_in, hn_in, pos_ids):
    """One pass of the layer-78 MTP block. en_in=[T,H] token-embeds, hn_in=[T,H] prev hidden,
    pos_ids=[T] absolute positions (for rope). Returns (h78=[T,H], keys=[1,nh,T,hd], vals=[1,nh,T,vd])."""
    T = en_in.shape[0]
    en = rms(en_in, g78(f'{P}.enorm.weight')); hn = rms(hn_in, g78(f'{P}.hnorm.weight'))
    x = torch.cat([en, hn], -1) @ w78(f'{P}.eh_proj.weight')
    n1 = rms(x[None], g78(f'{P}.input_layernorm.weight'))[0]
    f_rope = freqs(pos_ids.to(DT), qk_rope)
    q = (rms(n1 @ W['q_a_proj'], g78(f'{A}.q_a_layernorm.weight')) @ W['q_b_proj']).view(1, T, n_heads, qk_head).transpose(1, 2)
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = n1 @ W['kv_a_proj_with_mqa']; kc, kpe = comp.split([kv_lora, qk_rope], -1)
    kc = rms(kc, g78(f'{A}.kv_a_layernorm.weight'))
    kv = (kc @ W['kv_b_proj']).view(1, T, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
    qp = rope_neox(qp, f_rope); kpe2 = rope_neox(kpe.view(1, 1, T, qk_rope), f_rope).expand(-1, n_heads, -1, -1)
    qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
    return x, n1, qs, ks, vst

def moe(n2):
    T = n2.shape[0]
    scores = torch.sigmoid(n2 @ gate_w.t()); biased = scores + ebias[None]
    tk_ids = torch.topk(biased, topk, -1).indices; tk_w = scores.gather(1, tk_ids)
    if norm_topk: tk_w = tk_w / (tk_w.sum(-1, keepdim=True) + 1e-20)
    routed = torch.zeros(T, H, dtype=DT)
    for i in range(T):
        acc = torch.zeros(H, dtype=DT)
        for k in range(topk):
            eid = int(tk_ids[i, k]); w = tk_w[i, k]; g, u, dn = expert(eid)
            acc = acc + w * ((F.silu(n2[i] @ g) * (n2[i] @ u)) @ dn)
        routed[i] = acc
    shared = (F.silu(n2 @ shared_w['gate_proj']) * (n2 @ shared_w['up_proj'])) @ shared_w['down_proj']
    return shared + rsf * routed

# ---- DEPTH 1: MTP head over all positions (h_t main-hidden + embed(true tok_{t+1})) ----
t0 = time.time()
emb_shift = torch.nan_to_num(emb[torch.tensor(ids[1:])].to(DT))   # embed(tok_{i+1})
h_prev = final[:-1]; Sq = S - 1
pos1 = torch.arange(1, S)                                          # absolute positions
x1, n1_1, qs1, ks1, vst1 = mtp_block(emb_shift, h_prev, pos1)
mask = torch.full((Sq, Sq), float('-inf'), dtype=DT).triu(1)[None, None]
aw1 = F.softmax((qs1 @ ks1.transpose(2, 3)) * scaling + mask, -1)
a1_attn = (aw1 @ vst1).transpose(1, 2).reshape(Sq, n_heads * v_head) @ W['o_proj']
r2_1 = x1 + a1_attn
n2_1 = rms(r2_1[None], g78(f'{P}.post_attention_layernorm.weight'))[0]
h78_1 = r2_1 + moe(n2_1)
logits_mtp1 = rms(h78_1, g78(f'{P}.shared_head.norm.weight')) @ lm.t()
draft1 = logits_mtp1.argmax(-1)                                   # predicts tok_{t+2}
print(f"depth-1 computed ({time.time()-t0:.0f}s)", flush=True)

# ---- DEPTH 2: autoregressive — MTP head on (h78_1, embed(draft1)) predicts tok_{t+3} ----
# mirrors hpu_eagle.py:86-135 (mtp: hidden_states <- own output, input_ids <- prev draft).
# per-position: query at t+1 attends over depth-1 keys[0..t] (+ its own new key) -> draft2.
emb_d1 = torch.nan_to_num(emb[draft1].to(DT))                     # embed(drafted tok)
pos2 = torch.arange(2, S + 1)                                     # drafted token sits at +1
x2, n1_2, qs2, ks2, vst2 = mtp_block(emb_d1, h78_1, pos2)
# attention: query t (depth-2) sees depth-1 keys 0..t AND its own new key at t.
draft2 = torch.zeros(Sq, dtype=torch.long)
for t in range(Sq):
    K = torch.cat([ks1[:, :, :t + 1], ks2[:, :, t:t + 1]], 2)     # [1,nh, t+2, hd]
    V = torch.cat([vst1[:, :, :t + 1], vst2[:, :, t:t + 1]], 2)
    q = qs2[:, :, t:t + 1]
    aw = F.softmax((q @ K.transpose(2, 3)) * scaling, -1)
    at = (aw @ V).transpose(1, 2).reshape(1, n_heads * v_head) @ W['o_proj']
    r = x2[t:t + 1] + at
    n2 = rms(r[None], g78(f'{P}.post_attention_layernorm.weight'))[0]
    h2 = r + moe(n2)
    draft2[t] = (rms(h2, g78(f'{P}.shared_head.norm.weight')) @ lm.t())[0].argmax()
print(f"depth-2 computed ({time.time()-t0:.0f}s)", flush=True)

# ---- score acceptance against the main model's greedy tokens ----
start = len(prompt_ids) - 1
n1 = n2 = a1 = a2 = 0
for t in range(start, S - 3):
    tgt2 = int(logits_main[t + 1].argmax())      # main greedy tok_{t+2}
    tgt3 = int(logits_main[t + 2].argmax())      # main greedy tok_{t+3}
    ok1 = int(draft1[t]) == tgt2
    n1 += 1; a1 += int(ok1)
    if ok1:                                      # depth-2 only counts if depth-1 accepted
        n2 += 1; a2 += int(int(draft2[t]) == tgt3)
A1 = a1 / max(n1, 1); A2 = a2 / max(n2, 1)
E1 = 1 + A1; E2 = 1 + A1 + A1 * A2
print(f"\n=== MTP acceptance (vs main greedy, n={n1}) ===")
print(f"  a1 (depth-1)           = {a1}/{n1} = {A1*100:.0f}%")
print(f"  a2 (depth-2 | d1 acc)  = {a2}/{n2} = {A2*100:.0f}%")
print(f"  E[len] depth-1 (1+a1)        = {E1:.2f} tok/step")
print(f"  E[len] depth-2 (1+a1+a1*a2)  = {E2:.2f} tok/step")
print(f"  num_spec=2 vs =1 multiplier  = {E2/E1:.3f}x")
print(f"  => if base=14 tok/s: d1~{14*E1:.0f}, d2~{14*E2:.0f} tok/s (pre-overhead)")
