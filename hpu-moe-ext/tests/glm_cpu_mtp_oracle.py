"""CPU fp64 oracle for the GLM-5.1 MTP (layer-78) draft head.

Answers, checkpoint-side, the two questions the 0%-acceptance hunt needs:
  1. WHAT SHOULD THE DRAFT PREDICT? For each position, compute the MTP layer's
     greedy prediction (token t+2 from hidden[t] + embed(token[t+1])) and the
     MAIN model's greedy next token. Their agreement rate ~= the acceptance the
     serve should see. High agreement + serve at 0% = serve-side draft bug.
  2. REFERENCE STAGE NORMS at REAL inputs (eh_proj/attn/mlp/final) to compare
     against the serve-side probe values — the diverging stage is the bug.

Method: exactly glm_cpu_teacherforce.py (full 78-layer fp64 teacher-forced
forward over prompt+served tokens, plain w*x RMSNorm, NeoX rope, int4/int8
GPTQ-style dequant) + one extra MTP step. The checkpoint stores the MTP layer
under model.layers.78.* with NORMAL submodule names (self_attn/mlp/...), so the
same attn()/mlp() code runs it; the MTP-specific parts are enorm/hnorm/eh_proj
(int4) fusion in front and shared_head.norm -> lm_head behind.

Run (CPU only, no HPU needed — side container while a serve holds the cards):
  docker run --rm -v /mnt/fastmodels:/models:ro -v $HOME/AI/hpu-moe-ext:/w \
    vllm-gaudi:next python3 /w/tests/glm_cpu_mtp_oracle.py
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
SERVED = " 1. Mercury 2. Venus 3. Earth 4. Mars 5. Jupiter 6. 7. 8. 2. 9. 6. 5. 3. 4. "
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
def dq(p, gs=128, bits=None):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    if bits is None:
        # AUTO-DETECT from shapes: in_dim = groups*gs = sc.shape[0]*gs and the packed
        # qweight has in_dim/(32/bits) rows -> bits = 32*rows/in_dim. The mixed
        # AutoRound export uses int8 attn projs on layers 0-77 but int4 EVERYWHERE
        # on the MTP layer 78, so per-call constants are wrong there.
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
embed_in = emb[torch.tensor(ids)].to(DT)
embed_in = torch.nan_to_num(embed_in)  # mirror HPU_SANITIZE_EMBED

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

def mlp(L, normed2, Sq):
    MP = f'model.layers.{L}.mlp'
    if L < fkd:
        gate_w = get(f'{MP}.gate_proj.weight').to(DT); up_w = get(f'{MP}.up_proj.weight').to(DT); down_w = get(f'{MP}.down_proj.weight').to(DT)
        return (F.silu(normed2 @ gate_w.t()) * (normed2 @ up_w.t())) @ down_w.t()
    gate_w = get(f'{MP}.gate.weight').to(DT); ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)
    scores = torch.sigmoid(normed2 @ gate_w.t()); biased = scores + ebias[None]
    topk_ids = torch.topk(biased, topk, -1).indices; topk_w = scores.gather(1, topk_ids)
    if norm_topk: topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
    routed = torch.zeros(Sq, H, dtype=DT); cache = {}
    for eid in set(int(e) for e in topk_ids.reshape(-1).tolist()):
        cache[eid] = (dq(f"{MP}.experts.{eid}.gate_proj"), dq(f'{MP}.experts.{eid}.up_proj'), dq(f'{MP}.experts.{eid}.down_proj'))
    for i in range(Sq):
        acc = torch.zeros(H, dtype=DT)
        for k in range(topk):
            eid = int(topk_ids[i, k]); w = topk_w[i, k]; g, u, dn = cache[eid]
            acc = acc + w * ((F.silu(normed2[i] @ g) * (normed2[i] @ u)) @ dn)
        routed[i] = acc
    sg = dq(f'{MP}.shared_experts.gate_proj'); su = dq(f'{MP}.shared_experts.up_proj'); sd = dq(f'{MP}.shared_experts.down_proj')
    shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd
    return shared + rsf * routed

# ---- main model teacher-forced forward (verbatim glm_cpu_teacherforce.py) ----
t0 = time.time()
f_main = freqs(torch.arange(S, dtype=DT), qk_rope)
hidden = embed_in.clone(); residual = torch.zeros_like(hidden)
for L in range(nL):
    ln_in = get(f'model.layers.{L}.input_layernorm.weight')
    ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
    r1 = hidden if L == 0 else hidden + residual
    normed1 = rms(r1[None], ln_in)[0]
    a = attn(L, normed1[None], f_main, S); r2 = r1 + a
    normed2 = rms(r2[None], ln_post)[0]
    hidden = mlp(L, normed2, S); residual = r2
    print(f"layer {L} done {time.time()-t0:.0f}s", flush=True)
final = hidden + residual                      # pre-final-norm hiddens == vllm 'hidden_states'
norm_w = get('model.norm.weight').to(DT); lm = get('lm_head.weight').to(DT)
normed = norm_w * (final * torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + eps))
logits_main = normed @ lm.t()

# ---- MTP layer (78): draft prediction for t+2 from hidden[t] + embed(ids[t+1]) ----
L78 = nL  # 78
print(f"\n===== MTP layer {L78} fp64 ({time.time()-t0:.0f}s) =====", flush=True)
emb_shift = emb[torch.tensor(ids[1:])].to(DT)            # embed(token[t+1]), rows 0..S-2
emb_shift = torch.nan_to_num(emb_shift)
h_prev = final[:-1]                                       # hidden[t], rows 0..S-2
enorm_w = get(f'model.layers.{L78}.enorm.weight'); hnorm_w = get(f'model.layers.{L78}.hnorm.weight')
en = rms(emb_shift, enorm_w); hn = rms(h_prev, hnorm_w)
w_eh = dq(f'model.layers.{L78}.eh_proj')          # [2H, H] int4 AutoRound
x = torch.cat([en, hn], dim=-1) @ w_eh                    # [S-1, H]
print(f"stage norms: emb={emb_shift.norm():.2f} enorm={en.norm():.2f} hnorm={hn.norm():.2f} eh_proj={x.norm():.2f} (rows={S-1})", flush=True)

Sq = S - 1
f_mtp = freqs(torch.arange(1, S, dtype=DT), qk_rope)      # original positions 1..S-1
ln_in = get(f'model.layers.{L78}.input_layernorm.weight')
ln_post = get(f'model.layers.{L78}.post_attention_layernorm.weight')
n1 = rms(x[None], ln_in)[0]
a = attn(L78, n1[None], f_mtp, Sq)
print(f"stage norms: attn_out={a.norm():.2f}", flush=True)
r2 = x + a
n2 = rms(r2[None], ln_post)[0]
m = mlp(L78, n2, Sq)
print(f"stage norms: mlp_in(normed)={n2.norm():.2f} mlp_out={m.norm():.2f}", flush=True)
h78 = r2 + m
sh_norm = get(f'model.layers.{L78}.shared_head.norm.weight')
logits_mtp = rms(h78, sh_norm) @ lm.t()
print(f"stage norms: block_out={h78.norm():.2f}", flush=True)

print(f"\n===== draft vs target greedy ({time.time()-t0:.0f}s) =====", flush=True)
agree = 0; total = 0
start = len(prompt_ids) - 1
for t in range(start, S - 2):
    # serve compares: draft(token[t+1], hidden[t]) vs target's own greedy at t+1.
    draft_tok = int(logits_mtp[t].argmax())               # row t = (hidden[t], ids[t+1])
    tgt_next = int(logits_main[t + 1].argmax())           # target greedy after consuming ids[t+1]
    total += 1; agree += int(draft_tok == tgt_next)
    mark = "==" if draft_tok == tgt_next else "!="
    print(f"pos {t:3d} in={tok.decode([ids[t+1]])!r}: draft {tok.decode([draft_tok])!r} {mark} target {tok.decode([tgt_next])!r}", flush=True)
print(f"\nDRAFT-TARGET AGREEMENT {agree}/{total} = {agree/max(total,1)*100:.0f}% — this is the acceptance the serve SHOULD see. 0% on serve with healthy agreement here = serve-side draft bug; stage norms above are the per-stage references.", flush=True)
