"""
COMPREHENSIVE single-pass per-layer bug hunt for GLM-5.1.
Runs the fp64 CPU reference forward for layers 0,3,4,10,40,77 independently,
each from its OWN dumped input (ground-truth), and compares to the dumped output.

Fused-residual decode (verified for L0/L3):
  R1 = hidden_in + residual_in          (residual_in=0 for L0)
  normed1 = input_layernorm(R1)
  attn(normed1) -> attn_o  (full 64-head, full o_proj, = post all-reduce)
  R2 = R1 + attn_o = residual_out (authoritative)
  normed2 = post_attention_layernorm(R2)
  mlp/moe(normed2) -> hidden_out (the delta)

Reports per layer: cos(attn), cos(R2 residual), cos(mlp/moe), cos(full),
plus attention-logit saturation diagnostics (q/k norms, max logit).
"""
import json, torch, sys
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']
scaling = qk_head ** -0.5
n_exp = cfg['n_routed_experts']; topk = cfg['num_experts_per_tok']
rsf = cfg['routed_scaling_factor']; norm_topk = cfg['norm_topk_prob']
fkd = cfg['first_k_dense_replace']
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
S = 5  # real positions

def get(n):
    with safe_open(f'{CKPT}/{idx[n]}', framework='pt', device='cpu') as f:
        return f.get_tensor(n)

def dq(p, gs=128, bits=8):
    qw = get(p + '.qweight'); qz = get(p + '.qzeros'); sc = get(p + '.scales')
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    c = torch.bitwise_right_shift(torch.unsqueeze(qw, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)).to(torch.int32)
    c = torch.bitwise_and(c, (2 ** bits) - 1).to(DT); c = c.reshape(c.shape[0] * c.shape[1], c.shape[2])
    rz = torch.bitwise_right_shift(torch.unsqueeze(qz, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)).to(torch.int32)
    rz = torch.bitwise_and(rz + 1, (2 ** bits) - 1).to(DT); z = rz.reshape(-1, rz.shape[1] * rz.shape[2])
    scf = sc.to(DT); g = (torch.arange(c.shape[0]) // gs).long(); return (c - z[g]) * scf[g]

def cs(a, b):
    a = a.reshape(-1).to(DT); b = b.reshape(-1).to(DT)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

def rms(x, w):  # GlmMoeDsaRMSNorm: (1+weight)
    return (w.to(DT) + 1.0) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

def rh(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)

def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim))
    return torch.einsum('s,d->sd', pos, inv)

def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]
    return x * c + rh(x) * s

def run_layer(L):
    d = torch.load(f'/opt/hpu-dsa-ext/layer{L}_dump.pt', map_location='cpu', weights_only=False)
    positions = d['positions'][0].to(DT)
    hin = d['hidden_in']
    hin = (hin[0] if hin.dim() == 3 else hin).to(DT)        # [128,6144]
    rin = d['residual_in']
    if rin is None:
        rin = torch.zeros_like(hin)
    else:
        rin = (rin[0] if rin.dim() == 3 else rin).to(DT)
    hout = d['hidden_out']
    hout = (hout[0] if hout.dim() == 3 else hout).to(DT)    # [128,6144] delta
    rout = d['residual_out']
    rout = (rout[0] if rout.dim() == 3 else rout).to(DT)    # [128,6144]

    R1 = hin + rin
    hpu_attn = rout - R1                                     # implied attn_o (post all-reduce)

    P = f'model.layers.{L}.self_attn'
    w_q_a = dq(f'{P}.q_a_proj'); w_q_b = dq(f'{P}.q_b_proj')
    w_kv_a = dq(f'{P}.kv_a_proj_with_mqa'); w_kv_b = dq(f'{P}.kv_b_proj')
    w_o = dq(f'{P}.o_proj')
    ln_in = get(f'model.layers.{L}.input_layernorm.weight')
    ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
    q_a_ln = get(f'{P}.q_a_layernorm.weight'); kv_a_ln = get(f'{P}.kv_a_layernorm.weight')

    normed1 = rms(R1[:S][None], ln_in)
    f = freqs(positions[:S], qk_rope)
    q = (rms(normed1 @ w_q_a, q_a_ln) @ w_q_b).view(1, S, n_heads, qk_head).transpose(1, 2)
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = normed1 @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
    kv = (kc @ w_kv_b).view(1, S, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
    qp = rope_neox(qp, f)
    kpe2 = rope_neox(kpe.view(1, 1, S, qk_rope), f).expand(-1, n_heads, -1, -1)
    qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
    logits = (qs @ ks.transpose(2, 3)) * scaling
    mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)[None, None]
    aw = F.softmax(logits + mask, -1)
    vmix = (aw @ vst).transpose(1, 2).reshape(S, n_heads * v_head)
    attn_o = vmix @ w_o

    # saturation diag
    qnorm = qs.reshape(-1, S, qk_head).norm(dim=-1).mean().item()
    knorm = ks.reshape(-1, S, qk_head).norm(dim=-1).mean().item()
    logmax = logits.max().item()

    R2_ref = R1[:S] + attn_o
    R2_hpu = rout[:S]
    cos_attn = [cs(attn_o[i], hpu_attn[i]) for i in range(S)]
    cos_r2 = [cs(R2_ref[i], R2_hpu[i]) for i in range(S)]

    # ---- MLP / MoE on HPU-authoritative R2 ----
    normed2 = rms(R2_hpu[None], ln_post)[0]
    MP = f'model.layers.{L}.mlp'
    if L < fkd:  # dense
        gate_w = get(f'{MP}.gate_proj.weight').to(DT)
        up_w = get(f'{MP}.up_proj.weight').to(DT)
        down_w = get(f'{MP}.down_proj.weight').to(DT)
        mlp_out = (F.silu(normed2 @ gate_w.t()) * (normed2 @ up_w.t())) @ down_w.t()
        moe_full = mlp_out
        router_info = None
    else:
        gate_w = get(f'{MP}.gate.weight').to(DT)
        ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)
        rl = normed2 @ gate_w.t()
        scores = torch.sigmoid(rl)
        biased = scores + ebias[None]
        topk_ids = torch.topk(biased, topk, dim=-1).indices
        topk_w = scores.gather(1, topk_ids)
        if norm_topk:
            topk_w = topk_w / topk_w.sum(-1, keepdim=True)
        need = set(int(e) for e in topk_ids.reshape(-1).tolist())
        ecache = {}
        for eid in need:
            ecache[eid] = (dq(f'{MP}.experts.{eid}.gate_proj', bits=4),
                           dq(f'{MP}.experts.{eid}.up_proj', bits=4),
                           dq(f'{MP}.experts.{eid}.down_proj', bits=4))
        routed = torch.zeros(S, H, dtype=DT)
        for i in range(S):
            acc = torch.zeros(H, dtype=DT)
            for k in range(topk):
                eid = int(topk_ids[i, k]); w = topk_w[i, k]
                g, u, dn = ecache[eid]
                eo = (F.silu(normed2[i] @ g) * (normed2[i] @ u)) @ dn
                acc = acc + w * eo
            routed[i] = acc
        sg = dq(f'{MP}.shared_experts.gate_proj', bits=8)
        su = dq(f'{MP}.shared_experts.up_proj', bits=8)
        sd = dq(f'{MP}.shared_experts.down_proj', bits=8)
        shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd
        moe_full = shared + rsf * routed
        router_info = (topk_ids, topk_w, routed, shared)

    cos_mlp = [cs(moe_full[i], hout[i]) for i in range(S)]
    full_ref = R2_ref + moe_full
    hpu_full = R2_hpu + hout[:S]
    cos_full = [cs(full_ref[i], hpu_full[i]) for i in range(S)]
    ratio_mlp = [(hout[i].norm() / (moe_full[i].norm() + 1e-9)).item() for i in range(S)]
    ratio_attn = [(hpu_attn[i].norm() / (attn_o[i].norm() + 1e-9)).item() for i in range(S)]

    return dict(L=L, qnorm=qnorm, knorm=knorm, logmax=logmax,
                cos_attn=cos_attn, cos_r2=cos_r2, cos_mlp=cos_mlp, cos_full=cos_full,
                ratio_mlp=ratio_mlp, ratio_attn=ratio_attn,
                attn_o=attn_o, hpu_attn=hpu_attn, moe_full=moe_full, hout=hout,
                R2_hpu=R2_hpu, router_info=router_info)

def m(x):  # min over positions, rounded
    return round(min(x), 5)
def a(x):
    return [round(v, 5) for v in x]

print(f'{"L":>3} {"cos_attn(min)":>13} {"cos_R2(min)":>11} {"cos_mlp(min)":>12} {"cos_full(min)":>13} {"qN":>7} {"kN":>7} {"logmax":>9} {"attn|h/r|":>10} {"mlp|h/r|":>9}')
results = {}
for L in [0, 3, 4, 10, 40, 77]:
    r = run_layer(L)
    results[L] = r
    print(f'{L:>3} {m(r["cos_attn"]):>13} {m(r["cos_r2"]):>11} {m(r["cos_mlp"]):>12} {m(r["cos_full"]):>13} '
          f'{r["qnorm"]:>7.2f} {r["knorm"]:>7.2f} {r["logmax"]:>9.1f} '
          f'{round(sum(r["ratio_attn"])/S,3):>10} {round(sum(r["ratio_mlp"])/S,3):>9}')

print()
print('=== full per-position detail ===')
for L in [0, 3, 4, 10, 40, 77]:
    r = results[L]
    print(f'L{L}: cos_attn={a(r["cos_attn"])}')
    print(f'    cos_mlp ={a(r["cos_mlp"])}')
    print(f'    cos_full={a(r["cos_full"])}  ratio_attn={[round(x,3) for x in r["ratio_attn"]]} ratio_mlp={[round(x,3) for x in r["ratio_mlp"]]}')

torch.save({L: {k: v for k, v in results[L].items()
                if torch.is_tensor(v)} for L in results}, '/opt/hpu-dsa-ext/alllayer_ref.pt')
print('\nsaved /opt/hpu-dsa-ext/alllayer_ref.pt')
