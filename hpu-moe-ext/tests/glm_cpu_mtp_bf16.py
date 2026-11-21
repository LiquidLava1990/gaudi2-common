"""MTP layer-78 oracle with the ORIGINAL bf16 weights (the graft validation).

Loads layer 78 from /models/glm51-orig-l78 (4 shards of zai-org/GLM-5.1, plain
bf16 .weight tensors), everything else (cached main-model hiddens, lm_head,
embed) from the int4 serving checkpoint — exactly what the serve graft will do.
Healthy draft-vs-target agreement here => the graft is GO.

Run: docker run --rm -v /mnt/fastmodels:/models:ro -v $HOME/AI/hpu-moe-ext:/w \
       --entrypoint python3 vllm-gaudi:next /w/tests/glm_cpu_mtp_bf16.py
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
def g78(n): return H_(ORIG, oidx[n]).get_tensor(n)          # original bf16 layer-78
def w78(n): return g78(n).to(DT).t()                        # [out,in] -> [in,out] for x @ w
def rms(x, w):
    return w.to(DT) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim)); return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]; return x * c + rh(x) * s

d = torch.load(CACHE)
final, logits_main = d['final'], d['logits_main']
print(f"cached hiddens {tuple(final.shape)}", flush=True)
emb = get('model.embed_tokens.weight')
lm = get('lm_head.weight').to(DT)
L = nL  # 78
P = f'model.layers.{L}'

t0 = time.time()
emb_shift = torch.nan_to_num(emb[torch.tensor(ids[1:])].to(DT))
h_prev = final[:-1]; Sq = S - 1
en = rms(emb_shift, g78(f'{P}.enorm.weight'))
hn = rms(h_prev, g78(f'{P}.hnorm.weight'))
x = torch.cat([en, hn], dim=-1) @ w78(f'{P}.eh_proj.weight')
print(f"frontend: enorm={en.norm():.2f} hnorm={hn.norm():.2f} eh_proj={x.norm():.2f}", flush=True)

f_mtp = freqs(torch.arange(1, S, dtype=DT), qk_rope)
n1 = rms(x[None], g78(f'{P}.input_layernorm.weight'))[0]
A = f'{P}.self_attn'
q = (rms(n1 @ w78(f'{A}.q_a_proj.weight'), g78(f'{A}.q_a_layernorm.weight')) @ w78(f'{A}.q_b_proj.weight')).view(1, Sq, n_heads, qk_head).transpose(1, 2)
qn, qp = q.split([qk_nope, qk_rope], -1)
comp = n1 @ w78(f'{A}.kv_a_proj_with_mqa.weight'); kc, kpe = comp.split([kv_lora, qk_rope], -1)
kc = rms(kc, g78(f'{A}.kv_a_layernorm.weight'))
kv = (kc @ w78(f'{A}.kv_b_proj.weight')).view(1, Sq, n_heads, qk_nope + v_head)
kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
qp = rope_neox(qp, f_mtp); kpe2 = rope_neox(kpe.view(1, 1, Sq, qk_rope), f_mtp).expand(-1, n_heads, -1, -1)
qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
mask = torch.full((Sq, Sq), float('-inf'), dtype=DT).triu(1)[None, None]
aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
a = (aw @ vst).transpose(1, 2).reshape(Sq, n_heads * v_head) @ w78(f'{A}.o_proj.weight')
print(f"attn_out={a.norm():.2f} ({time.time()-t0:.0f}s)", flush=True)
r2 = x + a
n2 = rms(r2[None], g78(f'{P}.post_attention_layernorm.weight'))[0]

MP = f'{P}.mlp'
gate_w = g78(f'{MP}.gate.weight').to(DT); ebias = g78(f'{MP}.gate.e_score_correction_bias').to(DT)
scores = torch.sigmoid(n2 @ gate_w.t()); biased = scores + ebias[None]
topk_ids = torch.topk(biased, topk, -1).indices; topk_w = scores.gather(1, topk_ids)
if norm_topk: topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
routed = torch.zeros(Sq, H, dtype=DT); cache = {}
for eid in set(int(e) for e in topk_ids.reshape(-1).tolist()):
    cache[eid] = (w78(f'{MP}.experts.{eid}.gate_proj.weight'),
                  w78(f'{MP}.experts.{eid}.up_proj.weight'),
                  w78(f'{MP}.experts.{eid}.down_proj.weight'))
for i in range(Sq):
    acc = torch.zeros(H, dtype=DT)
    for k in range(topk):
        eid = int(topk_ids[i, k]); w = topk_w[i, k]; g, u, dn = cache[eid]
        acc = acc + w * ((F.silu(n2[i] @ g) * (n2[i] @ u)) @ dn)
    routed[i] = acc
shared = (F.silu(n2 @ w78(f'{MP}.shared_experts.gate_proj.weight')) * (n2 @ w78(f'{MP}.shared_experts.up_proj.weight'))) @ w78(f'{MP}.shared_experts.down_proj.weight')
m = shared + rsf * routed
print(f"mlp_out={m.norm():.2f}", flush=True)
h78 = r2 + m
logits_mtp = rms(h78, g78(f'{P}.shared_head.norm.weight')) @ lm.t()

agree = 0; total = 0
start = len(prompt_ids) - 1
for t in range(start, S - 2):
    d_tok = int(logits_mtp[t].argmax()); t_next = int(logits_main[t + 1].argmax())
    total += 1; agree += int(d_tok == t_next)
    mark = "==" if d_tok == t_next else "!="
    print(f"pos {t:3d}: draft {tok.decode([d_tok])!r} {mark} target {tok.decode([t_next])!r}", flush=True)
print(f"\nBF16-78 DRAFT-TARGET AGREEMENT {agree}/{total} = {agree/max(total,1)*100:.0f}% — healthy => GRAFT IS GO", flush=True)
