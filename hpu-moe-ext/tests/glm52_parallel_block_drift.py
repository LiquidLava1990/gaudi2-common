"""
CARD-FREE offline drift harness for the PARALLEL-TRANSFORMER-BLOCK reformulation of GLM-5.2.

WHY  --------------------------------------------------------------------------------------------
GLM-5.2's TP=8 decode pays TWO per-layer all-reduces (o_proj + MoE down_proj) = 2*78 = 156/token.
The PARALLEL block (GPT-J / Falcon form) computes attn AND moe BOTH from the SAME input-norm'd
hidden, so their TP-partials sum into ONE fused all-reduce:  h = s + all_reduce(attn_p + moe_p).
That is EXACT for the all-reduce (linearity: Σ(a+m)=Σa+Σm). The ONLY quality cost is the MATH change:
   SERIAL  : moe reads  norm_post(s + attn)
   PARALLEL: moe reads  norm_post(s)            <-- attn no longer feeds the MoE input
So the drift is driven entirely by how much `attn` perturbs the residual stream s, i.e. ||attn||/||s||
and cos(norm_post(s), norm_post(s+attn)).

WHAT THIS DOES  ---------------------------------------------------------------------------------
A full 78-layer fp64 CPU forward on the REAL GLM-5.2 int4 weights (experts int4 AutoRound, attn /
shared / dense bf16), single (non-TP) tensor so attn_p+moe_p == the post-all-reduce attn+moe. We run:
  (A) a SERIAL reference forward that records the per-layer drift driver  ||attn||/||s|| and
      cos(norm_post(s), norm_post(s+attn))  -- the inherent, config-independent profile.
  (B) end-to-end SERIAL-vs-PARALLEL final-logit / top-1 agreement, for FULL parallel and for
      SELECTIVE subsets (parallelize the K safest-drift layers, sweep K).

The serial baseline must predict ' Paris' for "The capital of France is" (coherence gate) or the
whole pipeline (dequant / rope / norm) is wrong and every downstream number is meaningless.

Run:  $HOME/.venvs/m3calib/bin/python glm52_parallel_block_drift.py [quick|full]
"""
import json, sys, time, math, os
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)
torch.set_num_threads(torch.get_num_threads())  # use all cores
DT = torch.float32 if os.environ.get('PB_FP32') else torch.float64
ROPE_STYLE = os.environ.get('PB_ROPE', 'gptj')   # gptj(GLM-5.2 interleave, CORRECT) | neox | none

CKPT = '/mnt/fastmodels/GLM-5.2-int4-experts-AutoRound'
cfg = json.load(open(f'{CKPT}/config.json'))
H        = cfg['hidden_size']            # 6144
n_heads  = cfg['num_attention_heads']    # 64
kv_lora  = cfg['kv_lora_rank']           # 512
qk_nope  = cfg['qk_nope_head_dim']       # 192
qk_rope  = cfg['qk_rope_head_dim']       # 64
qk_head  = cfg['qk_head_dim']            # 256
v_head   = cfg['v_head_dim']             # 256
q_lora   = cfg['q_lora_rank']            # 2048
eps      = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']
scaling  = qk_head ** -0.5
topk     = cfg['num_experts_per_tok']    # 8
rsf      = cfg['routed_scaling_factor']  # 2.5
norm_topk= cfg['norm_topk_prob']
fkd      = cfg['first_k_dense_replace']  # 3 (dense MLP layers 0..2)
nL       = cfg['num_hidden_layers']      # 78

idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
_h = {}
def H_(fn):
    if fn not in _h:
        _h[fn] = safe_open(f'{CKPT}/{fn}', framework='pt', device='cpu')
    return _h[fn]
def get(n):
    return H_(idx[n]).get_tensor(n)
def lin(x, name):
    """bf16 Linear stored [out,in]:  y = x @ W.T ."""
    return x @ get(name).to(DT).t()

# ---- int4 AutoRound dequant (experts only); GPTQ packing 8x int4 per int32 along dim0 -> [in,out]
_dqc = {}
def dq(p, gs=128, bits=4):
    key = (p, bits)
    c = _dqc.get(key)
    if c is not None:
        return c
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    cc = torch.bitwise_right_shift(torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int32)
    cc = torch.bitwise_and(cc, (2 ** bits) - 1).to(DT); cc = cc.reshape(cc.shape[0] * cc.shape[1], cc.shape[2])
    rz = torch.bitwise_right_shift(torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int32)
    rz = torch.bitwise_and(rz + 1, (2 ** bits) - 1).to(DT); z = rz.reshape(-1, rz.shape[1] * rz.shape[2])
    scf = sc.to(DT); g = (torch.arange(cc.shape[0]) // gs).long()
    w = (cc - z[g]) * scf[g]
    if len(_dqc) < _DQ_CACHE_MAX:
        _dqc[key] = w
    return w
_DQ_CACHE_MAX = 0   # set >0 to cache dequantized experts across forwards (memory heavy)

# ---- norms / rope ------------------------------------------------------------------------------
def rms(x, w):                 # per-layer RMSNorm uses (1+weight)  (GLM HPU_GLM_NORM_OFFSET=1)
    return (w.to(DT) + 1.0) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rms_plain(x, w):           # model.norm uses plain weight
    return w.to(DT) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim))
    return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]
    return x * c + rh(x) * s
def rope_gptj(x, f):                    # interleaved pairs (2i,2i+1) — vLLM is_neox_style=False
    c = f.cos()[None, None]; s = f.sin()[None, None]
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    o1 = x1 * c - x2 * s; o2 = x2 * c + x1 * s
    return torch.stack((o1, o2), -1).flatten(-2)
def rope(x, f):
    if ROPE_STYLE == 'gptj': return rope_gptj(x, f)
    if ROPE_STYLE == 'none': return x
    return rope_neox(x, f)

# ---- prompt (few-shot factual, in-distribution; the reasoning model wants a pattern not raw text)
# "The capital of France is Paris. The capital of Japan is" -> expect " Tokyo" at the last position.
prompt_ids = [785, 6722, 315, 9621, 374, 12089, 13, 576, 6722, 315, 6323, 374]
PARIS = 26113                                 # " Tokyo" (last-position expected next token)
S = len(prompt_ids)
positions = torch.arange(S, dtype=DT)
f_all = freqs(positions, qk_rope)
embed_in = get('model.embed_tokens.weight')[prompt_ids].to(DT)   # [S,H]

# ---- per-layer ops (full / non-TP) -------------------------------------------------------------
def attn(L, normed1):
    """normed1: [S,H] (already input_layernorm'd). returns attn output [S,H] (post o_proj)."""
    P = f'model.layers.{L}.self_attn'
    x = normed1
    q = lin(x, f'{P}.q_a_proj.weight')
    q = rms(q, get(f'{P}.q_a_layernorm.weight'))
    q = lin(q, f'{P}.q_b_proj.weight').view(S, n_heads, qk_head).transpose(0, 1)[None]  # [1,n,S,qk_head]
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = lin(x, f'{P}.kv_a_proj_with_mqa.weight')
    kc, kpe = comp.split([kv_lora, qk_rope], -1)
    kc = rms(kc, get(f'{P}.kv_a_layernorm.weight'))
    kv = lin(kc, f'{P}.kv_b_proj.weight').view(S, n_heads, qk_nope + v_head).transpose(0, 1)[None]
    kno, vst = kv.split([qk_nope, v_head], -1)
    qp = rope(qp, f_all)
    kpe2 = rope(kpe.view(1, 1, S, qk_rope), f_all).expand(-1, n_heads, -1, -1)
    qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
    mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)[None, None]
    aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
    vmix = (aw @ vst).transpose(1, 2).reshape(S, n_heads * v_head)
    return lin(vmix, f'{P}.o_proj.weight')

def mlp(L, normed2):
    """normed2: [S,H] (already post_attention_layernorm'd). returns FFN output [S,H]."""
    MP = f'model.layers.{L}.mlp'
    if L < fkd:                                   # dense MLP (layers 0..2)
        return (F.silu(lin(normed2, f'{MP}.gate_proj.weight')) * lin(normed2, f'{MP}.up_proj.weight')) \
               @ get(f'{MP}.down_proj.weight').to(DT).t()
    gate_w = get(f'{MP}.gate.weight').to(DT); ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)
    scores = torch.sigmoid(normed2 @ gate_w.t())
    biased = scores + ebias[None]
    topk_ids = torch.topk(biased, topk, -1).indices
    topk_w = scores.gather(1, topk_ids)
    if norm_topk:
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
    routed = torch.zeros(S, H, dtype=DT)
    cache = {}
    for eid in set(int(e) for e in topk_ids.reshape(-1).tolist()):
        cache[eid] = (dq(f'{MP}.experts.{eid}.gate_proj'),
                      dq(f'{MP}.experts.{eid}.up_proj'),
                      dq(f'{MP}.experts.{eid}.down_proj'))
    for i in range(S):
        acc = torch.zeros(H, dtype=DT)
        for k in range(topk):
            eid = int(topk_ids[i, k]); w = topk_w[i, k]; g, u, dn = cache[eid]
            acc = acc + w * ((F.silu(normed2[i] @ g) * (normed2[i] @ u)) @ dn)
        routed[i] = acc
    sg = get(f'{MP}.shared_experts.gate_proj.weight').to(DT).t()
    su = get(f'{MP}.shared_experts.up_proj.weight').to(DT).t()
    sd = get(f'{MP}.shared_experts.down_proj.weight').to(DT).t()
    shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd
    return shared + rsf * routed

# ---- full forward; `parallel` is a set of layer indices to run in PARALLEL-block form ----------
def forward(parallel=frozenset(), record=None):
    """Returns logits [S,V]. If `record` is a list, append per-layer drift dicts (serial stream)."""
    hidden = embed_in.clone(); residual = torch.zeros_like(hidden)
    for L in range(nL):
        ln_in   = get(f'model.layers.{L}.input_layernorm.weight')
        ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
        s = hidden if L == 0 else hidden + residual            # residual stream entering layer
        normed1 = rms(s, ln_in)
        a = attn(L, normed1)                                   # attn output (== post all-reduce)
        spa = s + a                                            # s + attn
        if record is not None:
            np_s   = rms(s,   ln_post)
            np_spa = rms(spa, ln_post)
            cs  = F.cosine_similarity(np_s, np_spa, dim=-1)    # per token
            an  = a.norm(dim=-1); sn = s.norm(dim=-1).clamp_min(1e-12)
            record.append(dict(L=L, cos=cs.mean().item(), cos_min=cs.min().item(),
                               a_over_s=(an / sn).mean().item()))
        if L in parallel:
            normed2 = rms(s,   ln_post)                        # PARALLEL: moe reads norm_post(s)
        else:
            normed2 = rms(spa, ln_post)                        # SERIAL  : moe reads norm_post(s+attn)
        m = mlp(L, normed2)
        # residual stream after layer == s + a + m  (identical bookkeeping for both forms)
        hidden = a + m; residual = s
    final = hidden + residual
    normed = rms_plain(final, get('model.norm.weight'))
    return normed @ get('lm_head.weight').to(DT).t()

def summarize(logits, ref=None, tag=''):
    top1 = logits.argmax(-1)
    lg4 = logits[S - 1]; rank = int((lg4 > lg4[PARIS]).sum())
    out = dict(tag=tag, top1=[int(t) for t in top1], paris_rank=rank,
               argmax_last=int(lg4.argmax()))
    if ref is not None:
        rtop1 = ref.argmax(-1)
        agree = float((top1 == rtop1).float().mean())
        # logit-space agreement on the predictive last position
        c = float(F.cosine_similarity(logits[S - 1][None].float(), ref[S - 1][None].float())[0])
        pj = F.softmax(logits[S - 1].float(), -1); pr = F.softmax(ref[S - 1].float(), -1)
        kl = float((pr * (pr.clamp_min(1e-12).log() - pj.clamp_min(1e-12).log())).sum())
        out.update(top1_agree=agree, last_cos=c, last_kl=kl)
    return out

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    t0 = time.time()
    rec = []
    serial = forward(parallel=frozenset(), record=rec)
    sser = summarize(serial, tag='SERIAL')
    print(f'\n[{time.time()-t0:.0f}s] SERIAL baseline: Tokyo_rank={sser["paris_rank"]} '
          f'argmax_last={sser["argmax_last"]!r}  top1={sser["top1"]}')
    print('  (coherence gate: Tokyo_rank should be 0 for the few-shot prompt)')

    print('\n=== PER-LAYER DRIFT DRIVER (serial residual stream) ===')
    print('  L   cos(np(s),np(s+a))  cos_min   ||attn||/||s||')
    for r in rec:
        print(f'  {r["L"]:2d}     {r["cos"]:.5f}          {r["cos_min"]:.5f}    {r["a_over_s"]:.4f}')

    import statistics as st
    coss = [r['cos'] for r in rec]; aos = [r['a_over_s'] for r in rec]
    print(f'\n  cos:   mean={st.mean(coss):.4f} min={min(coss):.4f} median={st.median(coss):.4f}')
    print(f'  a/s:   mean={st.mean(aos):.4f} max={max(aos):.4f} median={st.median(aos):.4f}')

    torch.save(dict(rec=rec, serial_logits=serial), '$HOME/AI/hpu-moe-ext/tests/_glm52_pb_serial.pt')

    if mode == 'quick':
        sys.exit(0)

    # SELECTIVE: order layers by safety (highest cos first), parallelize K safest, sweep K.
    # cos ordering puts layer 0 (cos<0) LAST, so K<=77 excludes it; K=78 adds it (the decisive jump).
    order = [r['L'] for r in sorted(rec, key=lambda r: -r['cos'])]
    print('\n=== END-TO-END SERIAL vs PARALLEL SWEEP (K safest-by-cos layers parallel) ===')
    print('  K   layers                   #allreduce(156->)  top1_agree  Tokyo_rank  last_cos   last_kl')
    configs = [60, 77, 78]   # 77 = all-but-layer-0 ; 78 = all (adds the cos<0 layer 0)
    for K in configs:
        pset = frozenset(order[:K])
        lg = forward(parallel=pset)
        s = summarize(lg, ref=serial, tag=f'K{K}')
        ar = 156 - K
        label = ('ALL-but-L0' if K == 77 else ('ALL-78(+L0)' if K == 78 else f'{K} safest'))
        print(f'  {K:2d}  {label:22s}     {ar:3d}            {s["top1_agree"]:.3f}       '
              f'{s["paris_rank"]:5d}    {s["last_cos"]:.4f}    {s["last_kl"]:.4f}')
    print(f'\n(serial baseline paris_rank={sser["paris_rank"]}; top1_agree=1.0 means byte-for-byte '
          f'next-token match across all {S} positions)')
    print(f'TOTAL {time.time()-t0:.0f}s')
