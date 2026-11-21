"""
Attention saturation root-cause. logmax ~620-778 is abnormal (normal ~ a few × 1).
For each layer: decompose the qk logit into nope vs rope contributions, check q/k
per-component norms, and compare to layer 0. Determine if saturation is REAL (a scale
bug) or just a legitimately-peaked attention (and whether bf16 would argmax-flip).

Also: check if the attn cos dip (0.987) is driven by the saturated positions and
whether it's a real per-head error or just fp64-vs-bf16 softmax-edge noise.
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
rope_theta = cfg['rope_parameters']['rope_theta']
scaling = qk_head ** -0.5
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
S = 5
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
def rms(x, w):
    return (w.to(DT) + 1.0) * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
def rh(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
def freqs(pos, dim):
    inv = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=DT) / dim))
    return torch.einsum('s,d->sd', pos, inv)
def rope_neox(x, f):
    e = torch.cat((f, f), -1); c = e.cos()[None, None]; s = e.sin()[None, None]
    return x * c + rh(x) * s

for L in [0, 3, 40]:
    d = torch.load(f'/opt/hpu-dsa-ext/layer{L}_dump.pt', map_location='cpu', weights_only=False)
    positions = d['positions'][0].to(DT)
    hin = d['hidden_in']; hin = (hin[0] if hin.dim()==3 else hin).to(DT)
    rin = d['residual_in']
    rin = torch.zeros_like(hin) if rin is None else (rin[0] if rin.dim()==3 else rin).to(DT)
    R1 = hin + rin
    P = f'model.layers.{L}.self_attn'
    w_q_a = dq(f'{P}.q_a_proj'); w_q_b = dq(f'{P}.q_b_proj')
    w_kv_a = dq(f'{P}.kv_a_proj_with_mqa'); w_kv_b = dq(f'{P}.kv_b_proj')
    ln_in = get(f'model.layers.{L}.input_layernorm.weight')
    q_a_ln = get(f'{P}.q_a_layernorm.weight'); kv_a_ln = get(f'{P}.kv_a_layernorm.weight')
    normed1 = rms(R1[:S][None], ln_in)
    f = freqs(positions[:S], qk_rope)
    q = (rms(normed1 @ w_q_a, q_a_ln) @ w_q_b).view(1, S, n_heads, qk_head).transpose(1, 2)
    qn, qp = q.split([qk_nope, qk_rope], -1)
    comp = normed1 @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
    kv = (kc @ w_kv_b).view(1, S, n_heads, qk_nope + v_head)
    kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
    qpr = rope_neox(qp, f); kper = rope_neox(kpe.view(1,1,S,qk_rope), f).expand(-1,n_heads,-1,-1)
    # decompose logits: nope part + rope part
    log_nope = (qn @ kno.transpose(2,3)) * scaling
    log_rope = (qpr @ kper.transpose(2,3)) * scaling
    log_tot = log_nope + log_rope
    # per-component q/k norms (averaged over heads)
    print(f'=== L{L} ===')
    print(f'  |qn|/head={qn.norm(dim=-1).mean():.2f}  |qp|/head={qp.norm(dim=-1).mean():.2f}  '
          f'|kno|/head={kno.norm(dim=-1).mean():.2f}  |kpe|/head={kper.norm(dim=-1).mean():.2f}')
    print(f'  log_nope: max={log_nope.max():.1f} mean|={log_nope.abs().mean():.2f}   '
          f'log_rope: max={log_rope.max():.1f} mean|={log_rope.abs().mean():.2f}')
    print(f'  log_total: max={log_tot.max():.1f}  scaling=1/sqrt(256)={scaling:.5f}')
    # is the max on the diagonal (self-attn, position i attends to itself) - expected for peaked
    mask = torch.full((S,S), float('-inf'), dtype=DT).triu(1)[None,None]
    aw = F.softmax(log_tot + mask, -1)
    # entropy of attention per query (low = peaked)
    ent = -(aw * (aw+1e-30).log()).sum(-1).mean().item()
    print(f'  attn entropy (avg over heads,queries)={ent:.3f} (0=fully peaked, log(S)={torch.log(torch.tensor(float(S))):.3f}=uniform)')
    # bf16 vs fp64 argmax flip test: how many (head,query) argmax positions flip in bf16?
    aw_bf = F.softmax((log_tot+mask).to(torch.bfloat16).float(), -1)
    flips = (aw.argmax(-1) != aw_bf.argmax(-1).double()).float().mean().item()
    print(f'  bf16 argmax-flip fraction = {flips:.3f}')
    print()
