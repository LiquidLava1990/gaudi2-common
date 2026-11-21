"""MTP wiring-variant explorer (CPU fp64, reuses cached main-model hiddens).

The straight DeepSeek-MTP wiring of GLM-5.1's layer 78 produces constant junk
('odesk'/'lop') in BOTH the serve and the independent fp64 oracle. Upstream
vLLM's wiring is proven on CUDA — but those deployments run bf16/NVFP4
checkpoints, not this AutoRound int4 export. Two live hypotheses:
  (a) the wiring for THIS export differs (cat order / norm assignment / etc.)
  (b) the int4 export simply broke the MTP layer (calibration never saw it)

This script runs the heavy 78-layer fp64 forward ONCE (cached to disk), then
sweeps cheap wiring variants of the MTP layer and prints draft-vs-target
agreement for each. ANY variant with decent agreement = wiring bug (adopt it);
ALL variants junk = the checkpoint's MTP layer is broken (close the lever).

Run:
  docker run --rm -v /mnt/fastmodels:/models:ro -v $HOME/AI/hpu-moe-ext:/w \
    --entrypoint python3 vllm-gaudi:next /w/tests/glm_cpu_mtp_variants.py
"""
import json, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
CACHE = '/w/tests/.mtp_hidden_cache.pt'
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
SERVED = " 1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6. 7. 8. 2. 9. 6. 5. 3. 4. "
prompt_ids = tok(PROMPT)["input_ids"]
served_ids = tok(SERVED, add_special_tokens=False)["input_ids"]
ids = prompt_ids + served_ids
S = len(ids)

_h = {}
def H_(fn):
    if fn not in _h: _h[fn] = safe_open(f'{CKPT}/{fn}', framework='pt', device='cpu')
    return _h[fn]
def get(n): return H_(idx[n]).get_tensor(n)
def dq(p, gs=128):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    bits = int(32 * qw.shape[0] // (sc.shape[0] * gs))
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

def attn(L, normed1, f_all, Sq):
    P = f'model.layers.{L}.self_attn'
    w_q_a = dq(f'{P}.q_a_proj'); w_q_b = dq(f'{P}.q_b_proj')
    w_kv_a = dq(f'{P}.kv_a_proj_with_mqa'); w_kv_b = dq(f'{P}.kv_b_proj'); w_o = dq(f'{P}.o_proj')
    q_a_ln = get(f'{P}.q_a_layernorm.weight'); kv_a_ln = get(f'{P}.kv_a_layernorm.weight')
    q = (rms(normed1 @ w_q_a, q_a_ln) @ w_q_b).view(1, Sq, n_heads, qk_head).transpose(1, 2)
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = normed1 @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
    kv = (kc @ w_kv_b).view(1, Sq, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
    qp = rope_neox(qp, f_all); kpe2 = rope_neox(kpe.view(1, 1, Sq, qk_rope), f_all).expand(-1, n_heads, -1, -1)
    qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
    mask = torch.full((Sq, Sq), float('-inf'), dtype=DT).triu(1)[None, None]
    aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
    vmix = (aw @ vst).transpose(1, 2).reshape(Sq, n_heads * v_head)
    return vmix @ w_o

# Cross-variant expert cache ONLY for the MTP layer (L78) — caching every
# layer's fp64 dequants OOMs the host (~300MB/expert x ~30/layer x 75 layers).
_expert_cache = {}
def mlp(L, normed2, Sq):
    MP = f'model.layers.{L}.mlp'
    gate_w = get(f'{MP}.gate.weight').to(DT); ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)
    scores = torch.sigmoid(normed2 @ gate_w.t()); biased = scores + ebias[None]
    topk_ids = torch.topk(biased, topk, -1).indices; topk_w = scores.gather(1, topk_ids)
    if norm_topk: topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
    routed = torch.zeros(Sq, H, dtype=DT)
    cache = _expert_cache if L == nL else {}
    for eid in set(int(e) for e in topk_ids.reshape(-1).tolist()):
        if (L, eid) not in cache:
            cache[(L, eid)] = (dq(f'{MP}.experts.{eid}.gate_proj'), dq(f'{MP}.experts.{eid}.up_proj'), dq(f'{MP}.experts.{eid}.down_proj'))
    for i in range(Sq):
        acc = torch.zeros(H, dtype=DT)
        for k in range(topk):
            eid = int(topk_ids[i, k]); w = topk_w[i, k]; g, u, dn = cache[(L, eid)]
            acc = acc + w * ((F.silu(normed2[i] @ g) * (normed2[i] @ u)) @ dn)
        routed[i] = acc
    sg = dq(f'{MP}.shared_experts.gate_proj'); su = dq(f'{MP}.shared_experts.up_proj'); sd = dq(f'{MP}.shared_experts.down_proj')
    shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd
    return shared + rsf * routed

# ---- main-model hiddens: compute once, cache to disk ----
if os.path.exists(CACHE):
    d = torch.load(CACHE)
    final, logits_main = d['final'], d['logits_main']
    print(f"loaded cached hiddens {tuple(final.shape)}", flush=True)
else:
    t0 = time.time()
    embed_in = torch.nan_to_num(emb[torch.tensor(ids)].to(DT))
    f_main = freqs(torch.arange(S, dtype=DT), qk_rope)
    hidden = embed_in.clone(); residual = torch.zeros_like(hidden)
    for L in range(nL):
        ln_in = get(f'model.layers.{L}.input_layernorm.weight')
        ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
        r1 = hidden if L == 0 else hidden + residual
        normed1 = rms(r1[None], ln_in)[0]
        if L < fkd:
            MP = f'model.layers.{L}.mlp'
            a = attn(L, normed1[None], f_main, S); r2 = r1 + a
            normed2 = rms(r2[None], ln_post)[0]
            gate_w = get(f'{MP}.gate_proj.weight').to(DT); up_w = get(f'{MP}.up_proj.weight').to(DT); down_w = get(f'{MP}.down_proj.weight').to(DT)
            hidden = (F.silu(normed2 @ gate_w.t()) * (normed2 @ up_w.t())) @ down_w.t()
        else:
            a = attn(L, normed1[None], f_main, S); r2 = r1 + a
            normed2 = rms(r2[None], ln_post)[0]
            hidden = mlp(L, normed2, S)
        residual = r2
        if L % 10 == 0: print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)
    final = hidden + residual
    norm_w = get('model.norm.weight').to(DT); lm = get('lm_head.weight').to(DT)
    normed = norm_w * (final * torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + eps))
    logits_main = normed @ lm.t()
    torch.save({'final': final, 'logits_main': logits_main}, CACHE)
    print(f"cached hiddens ({time.time()-t0:.0f}s)", flush=True)

L78 = nL
lm = get('lm_head.weight').to(DT)
enorm_w = get(f'model.layers.{L78}.enorm.weight'); hnorm_w = get(f'model.layers.{L78}.hnorm.weight')
w_eh = dq(f'model.layers.{L78}.eh_proj')
ln_in = get(f'model.layers.{L78}.input_layernorm.weight')
ln_post = get(f'model.layers.{L78}.post_attention_layernorm.weight')
sh_norm = get(f'model.layers.{L78}.shared_head.norm.weight')
norm_main = get('model.norm.weight')
emb_shift = torch.nan_to_num(emb[torch.tensor(ids[1:])].to(DT))
h_prev = final[:-1]
Sq = S - 1

def run_variant(name, cat_order, e_on_embed, use_input_ln, pos_start, head_norm):
    en_w, hn_w = (enorm_w, hnorm_w) if e_on_embed else (hnorm_w, enorm_w)
    en = rms(emb_shift, en_w); hn = rms(h_prev, hn_w)
    fused = torch.cat([en, hn] if cat_order == 'eh' else [hn, en], dim=-1)
    x = fused @ w_eh
    f_mtp = freqs(torch.arange(pos_start, pos_start + Sq, dtype=DT), qk_rope)
    n1 = rms(x[None], ln_in)[0] if use_input_ln else x
    a = attn(L78, n1[None], f_mtp, Sq)
    r2 = x + a
    n2 = rms(r2[None], ln_post)[0]
    m = mlp(L78, n2, Sq)
    h78 = r2 + m
    logits = rms(h78, head_norm) @ lm.t()
    agree = 0; total = 0; sample = []
    start = len(prompt_ids) - 1
    for t in range(start, S - 2):
        d_tok = int(logits[t].argmax()); t_next = int(logits_main[t + 1].argmax())
        total += 1; agree += int(d_tok == t_next)
        if len(sample) < 6: sample.append((tok.decode([d_tok]), tok.decode([t_next])))
    print(f"[{name}] agreement {agree}/{total} ({agree/max(total,1)*100:.0f}%)  "
          f"draft-vs-target sample: {sample}", flush=True)
    return agree / max(total, 1)

print("\n===== variant sweep =====", flush=True)
run_variant("baseline DeepSeek (eh, e-on-embed, input_ln, pos1, shared_head.norm)",
            'eh', True, True, 1, sh_norm)
run_variant("cat swapped (he)", 'he', True, True, 1, sh_norm)
run_variant("norms swapped (hnorm on embed)", 'eh', False, True, 1, sh_norm)
run_variant("cat+norms swapped", 'he', False, True, 1, sh_norm)
run_variant("no input_layernorm", 'eh', True, False, 1, sh_norm)
run_variant("positions from 0", 'eh', True, True, 0, sh_norm)
run_variant("head uses model.norm", 'eh', True, True, 1, norm_main)
