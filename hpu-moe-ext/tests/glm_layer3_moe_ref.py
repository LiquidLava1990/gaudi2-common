"""
Layer-3 (first MoE layer) full CPU fp64 reference, bisecting the MoE.

Fused-residual decode (DeepseekV2DecoderLayer.forward, verified for layer 0):
  in:  hidden_in (=prev MLP delta), residual_in (=running residual)
  input_layernorm(hidden_in, residual_in) -> (normed1, R1) with R1 = hidden_in + residual_in
  attn(normed1) -> attn_out (post o_proj + all-reduce)
  post_attention_layernorm(attn_out, R1) -> (normed2, R2) with R2 = attn_out + R1
  mlp(normed2) -> hidden_out  (the MoE delta, NOT incl. residual)
  layer returns (hidden_out, residual_out=R2)
So:
  residual_out (HPU) = attn_out + hidden_in + residual_in   -> gives attn_out
  hidden_out  (HPU)  = MoE(normed2)                          -> the thing we bisect

MoE (GLM-5.1): router sigmoid + e_score_correction_bias, n_group=1/topk_group=1 (grouping
no-op), top-8 of 256 on BIASED scores, weights = UNBIASED sigmoid gathered, norm_topk_prob,
then routed output *= routed_scaling_factor(2.5) (applied ONCE in the runner; the FusedMoE
router gets factor=1.0 because apply_routed_scale_to_output=True). shared expert (8-bit) added
WITHOUT the 2.5 scale. final = shared + 2.5*routed, then TP/EP all-reduce.
"""
import json, torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
DUMP = '/opt/hpu-dsa-ext/layer3_dump.pt'
L = 3
cfg = json.load(open(f'{CKPT}/config.json'))
H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
rope_theta = cfg['rope_parameters']['rope_theta']
scaling = qk_head ** -0.5
n_exp = cfg['n_routed_experts']; topk = cfg['num_experts_per_tok']
n_group = cfg['n_group']; topk_group = cfg['topk_group']
rsf = cfg['routed_scaling_factor']; norm_topk = cfg['norm_topk_prob']
moe_inter = cfg['moe_intermediate_size']
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']

def get(n):
    with safe_open(f'{CKPT}/{idx[n]}', framework='pt', device='cpu') as f:
        return f.get_tensor(n)

# generic gptq-sym dequant (autogptq +1 zero). bits=4 or 8. returns [in,out] fp64
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

# ---------------- load dump ----------------
d = torch.load(DUMP, map_location='cpu', weights_only=False)
positions = d['positions'][0].to(DT)
hidden_in = d['hidden_in'].to(DT)              # [128,6144]
residual_in = d['residual_in'][0].to(DT)       # [128,6144]
hidden_out = d['hidden_out'].to(DT)            # [128,6144] (MoE delta)
residual_out = d['residual_out'][0].to(DT)     # [128,6144]
o_proj_w_rank = d['o_proj_weight'].to(DT)      # [6144,2048] this rank only
S = 5

# implied HPU attn_out (post o_proj + all-reduce) = residual_out - (hidden_in + residual_in)
R1 = hidden_in + residual_in                   # running residual after input
hpu_attn = residual_out - R1                    # [128,6144]
print('positions[:8]=', positions[:8].tolist())
print('|hidden_in|', [round(hidden_in[i].norm().item(),2) for i in range(S)])
print('|residual_in|', [round(residual_in[i].norm().item(),2) for i in range(S)])
print('|R1=hin+rin|', [round(R1[i].norm().item(),2) for i in range(S)])
print('|hpu_attn=rout-R1|', [round(hpu_attn[i].norm().item(),2) for i in range(S)])
print('|hpu hidden_out(MoE)|', [round(hidden_out[i].norm().item(),2) for i in range(S)])
print()

# ---------------- STEP 1: attention reference (full, all heads) ----------------
P = f'model.layers.{L}.self_attn'
w_q_a = dq(f'{P}.q_a_proj'); w_q_b = dq(f'{P}.q_b_proj')
w_kv_a = dq(f'{P}.kv_a_proj_with_mqa'); w_kv_b = dq(f'{P}.kv_b_proj')
w_o = dq(f'{P}.o_proj')                          # [16384,6144] full
ln_in = get(f'model.layers.{L}.input_layernorm.weight')
q_a_ln = get(f'{P}.q_a_layernorm.weight'); kv_a_ln = get(f'{P}.kv_a_layernorm.weight')

normed1 = rms(R1[:S][None], ln_in)               # input_layernorm on R1
f = freqs(positions[:S], qk_rope)
q = (rms(normed1 @ w_q_a, q_a_ln) @ w_q_b).view(1, S, n_heads, qk_head).transpose(1, 2)
qn, qp = q.split([qk_nope, qk_rope], -1)
comp = normed1 @ w_kv_a; kc, kpe = comp.split([kv_lora, qk_rope], -1); kc = rms(kc, kv_a_ln)
kv = (kc @ w_kv_b).view(1, S, n_heads, qk_nope + v_head)
kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
qp = rope_neox(qp, f)
kpe2 = rope_neox(kpe.view(1, 1, S, qk_rope), f).expand(-1, n_heads, -1, -1)
qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)[None, None]
aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
vmix = (aw @ vst).transpose(1, 2).reshape(S, n_heads * v_head)   # [S,16384]
attn_o = vmix @ w_o                              # [S,6144] full o_proj
print('=== STEP 1: ATTENTION (full ref vs HPU implied attn) ===')
print('  |ref attn_o|', [round(attn_o[i].norm().item(),2) for i in range(S)])
print('  cos(ref_attn, hpu_attn)/pos:', [round(cs(attn_o[i], hpu_attn[i]),5) for i in range(S)])
print('  |hpu|/|ref|:', [round((hpu_attn[i].norm()/attn_o[i].norm()).item(),4) for i in range(S)])
# we use HPU's attn (authoritative) for the residual that MoE sees, so MoE bisection
# is not contaminated by any residual attn error.
R2_hpu = residual_out                            # = attn_out + R1 (authoritative)
R2_ref = (R1[:S] + attn_o)                       # ref-attn residual
print('  cos(R2_ref, R2_hpu)/pos:', [round(cs(R2_ref[i], R2_hpu[i]),5) for i in range(S)])
print()

# ---------------- STEP 2: MoE reference ----------------
# normed2 = post_attention_layernorm(attn_out, R1) -> RMSNorm(R2).  Use HPU R2 (authoritative).
ln_post = get(f'model.layers.{L}.post_attention_layernorm.weight')
normed2 = rms(R2_hpu[:S][None], ln_post)[0]       # [S,6144]  MoE input

MP = f'model.layers.{L}.mlp'
gate_w = get(f'{MP}.gate.weight').to(DT)          # [256,6144]
ebias = get(f'{MP}.gate.e_score_correction_bias').to(DT)  # [256]

# router: sigmoid(logits) + bias -> grouped top-8 (group no-op) -> unbiased weights -> norm
logits = normed2 @ gate_w.t()                     # [S,256]
scores = torch.sigmoid(logits)
biased = scores + ebias[None]
# n_group=1/topk_group=1: whole group selected, equals plain top-k on biased
topk_ids = torch.topk(biased, topk, dim=-1).indices   # [S,8]
topk_w = scores.gather(1, topk_ids)               # UNBIASED weights
if norm_topk:
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)
print('=== STEP 2/3a: ROUTER ===')
for i in range(S):
    print(f'  pos{i} ids={topk_ids[i].tolist()} w={[round(x,4) for x in topk_w[i].tolist()]} sum={topk_w[i].sum().item():.4f}')
print()

# routed experts (int4) dequant + compute
def expert_out(x_row, eid):
    g = dq(f'{MP}.experts.{eid}.gate_proj', bits=4)   # [6144,2048]
    u = dq(f'{MP}.experts.{eid}.up_proj', bits=4)
    dn = dq(f'{MP}.experts.{eid}.down_proj', bits=4)  # [2048,6144]
    return (F.silu(x_row @ g) * (x_row @ u)) @ dn      # [6144]

routed = torch.zeros(S, H, dtype=DT)
import itertools
need = set(int(e) for e in topk_ids[:S].reshape(-1).tolist())
ecache = {}
for eid in need:
    ecache[eid] = (dq(f'{MP}.experts.{eid}.gate_proj', bits=4),
                   dq(f'{MP}.experts.{eid}.up_proj', bits=4),
                   dq(f'{MP}.experts.{eid}.down_proj', bits=4))
for i in range(S):
    acc = torch.zeros(H, dtype=DT)
    for k in range(topk):
        eid = int(topk_ids[i, k]); w = topk_w[i, k]
        g, u, dn = ecache[eid]
        eo = (F.silu(normed2[i] @ g) * (normed2[i] @ u)) @ dn
        acc = acc + w * eo
    routed[i] = acc
routed_scaled = routed * rsf                       # *2.5 (runner)

# shared expert (8-bit)
sg = dq(f'{MP}.shared_experts.gate_proj', bits=8)   # [6144,2048]
su = dq(f'{MP}.shared_experts.up_proj', bits=8)
sd = dq(f'{MP}.shared_experts.down_proj', bits=8)   # [2048,6144]
shared = (F.silu(normed2 @ sg) * (normed2 @ su)) @ sd   # [S,6144]

moe_full = shared + routed_scaled                  # final MoE output (delta)

print('=== MoE component norms (ref) ===')
print('  |routed (unscaled)|', [round(routed[i].norm().item(),2) for i in range(S)])
print('  |routed*2.5|', [round(routed_scaled[i].norm().item(),2) for i in range(S)])
print('  |shared|', [round(shared[i].norm().item(),2) for i in range(S)])
print('  |moe_full=shared+2.5routed|', [round(moe_full[i].norm().item(),2) for i in range(S)])
print('  |HPU hidden_out|', [round(hidden_out[i].norm().item(),2) for i in range(S)])
print()
print('=== STEP 2: FULL MoE ref vs HPU hidden_out ===')
print('  cos(moe_full, hidden_out)/pos:', [round(cs(moe_full[i], hidden_out[i]),5) for i in range(S)])
print('  |hpu|/|ref|:', [round((hidden_out[i].norm()/moe_full[i].norm()).item(),4) for i in range(S)])
print()
print('=== BISECT: which combine matches HPU hidden_out best? ===')
cands = {
  'shared + 2.5*routed (correct)': shared + 2.5*routed,
  'shared + routed (no 2.5)':       shared + routed,
  'shared + 6.25*routed (double)':  shared + 6.25*routed,
  '2.5*routed only (no shared)':    2.5*routed,
  'shared only (no routed)':        shared,
  '2.5*(shared+routed)':            2.5*(shared+routed),
}
for nm, t in cands.items():
    cc = [round(cs(t[i], hidden_out[i]),5) for i in range(S)]
    rr = [round((hidden_out[i].norm()/(t[i].norm()+1e-9)).item(),3) for i in range(S)]
    print(f'  {nm:34s} cos={cc} ratio={rr}')
print()
# Save intermediates for the routed/shared isolation script (step 3b/3c via live op)
torch.save({'normed2': normed2.to(torch.float32),
            'topk_ids': topk_ids, 'topk_w': topk_w.to(torch.float32),
            'routed': routed.to(torch.float32), 'shared': shared.to(torch.float32),
            'moe_full': moe_full.to(torch.float32),
            'hidden_out': hidden_out[:S].to(torch.float32)},
           '/opt/hpu-dsa-ext/layer3_moe_ref.pt')
print('saved /opt/hpu-dsa-ext/layer3_moe_ref.pt')
