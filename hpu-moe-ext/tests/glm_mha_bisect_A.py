"""
PART (A): PRE-o_proj / value-mixture op check (THE KEY TEST).
Pure CPU. Reproduce forward_mha for this TP rank's 8 heads from the dumped
q / latent_vec_k / kv_b_weight / scale, and compare to the dumped fwd_mha_out.

Run:
  docker exec -e CUDA_VISIBLE_DEVICES='' -e HABANA_VISIBLE_DEVICES='' \
              -e HABANA_VISIBLE_MODULES='' vllm-active \
              python3 /opt/hpu-moe-ext/tests/glm_mha_bisect_A.py
"""
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
DT = torch.float64  # high precision for the reference

MHA = '/opt/hpu-dsa-ext/mha_dump.pt'

d = torch.load(MHA, map_location='cpu', weights_only=False)
q_all          = d['q'].to(DT)              # [128, 8, 256]
latent_all     = d['latent_vec_k'].to(DT)   # [128, 576]
fwd_mha_out    = d['fwd_mha_out'].to(DT)     # [128, 2048]
kv_b_weight    = d['kv_b_weight'].to(DT)     # [3584, 512]  (nn.Linear orient [out,in])
scale          = float(d['scale'])           # 0.0625 = 256**-0.5
nh   = d['num_heads']     # 8
nope = d['qk_nope']       # 192
rope = d['qk_rope']       # 64
vh   = d['v_head']        # 256
klora= d['kv_lora']       # 512
qkh  = d['qk_head']       # 256

print('dims: nh=%d nope=%d rope=%d vh=%d klora=%d qkh=%d scale=%.6f' % (nh,nope,rope,vh,klora,qkh,scale))
print('shapes: q', tuple(q_all.shape), 'latent', tuple(latent_all.shape),
      'fwd_mha_out', tuple(fwd_mha_out.shape), 'kv_b_weight', tuple(kv_b_weight.shape))

# Only the first 5 positions are real tokens ("The capital of France is").
S = 5
q      = q_all[:S]          # [5, 8, 256]
latent = latent_all[:S]     # [5, 576]

# ---- reference forward_mha (the exact recipe forward_mha implements) ----
# split latent -> k_c_normed[512], k_pe[64]
k_c_normed, k_pe = latent.split([klora, rope], dim=-1)   # [5,512],[5,64]
k_pe = k_pe.view(S, 1, rope)                              # [5,1,64]

# kv = k_c_normed @ kv_b_weight.T  -> [5, 3584] -> view [5,8,448]
kv = k_c_normed @ kv_b_weight.T                           # [5, 8*(192+256)] = [5,3584]
kv = kv.view(S, nh, nope + vh)                            # [5,8,448]
k_nope, v = kv.split([nope, vh], dim=-1)                  # [5,8,192],[5,8,256]

# k = cat(k_nope, k_pe.expand) -> [5,8,256]
k = torch.cat((k_nope, k_pe.expand((S, nh, rope))), dim=-1)   # [5,8,256]

# causal SDPA per head:  q [5,8,256], k [5,8,256], v [5,8,256]
# transpose to [8,5,256]
qh = q.transpose(0, 1)     # [8,5,256]
kh = k.transpose(0, 1)     # [8,5,256]
vh_ = v.transpose(0, 1)    # [8,5,256]

attn_mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)   # causal
logits = (qh @ kh.transpose(1, 2)) * scale + attn_mask           # [8,5,5]
probs = F.softmax(logits, dim=-1)                                 # [8,5,5]
out = probs @ vh_                                                 # [8,5,256]
ref = out.transpose(0, 1).reshape(S, nh * vh)                    # [5,2048]

# ---- compare to dumped fwd_mha_out[:5] ----
hpu = fwd_mha_out[:S]    # [5,2048]

def cossim(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

print()
print('=' * 80)
print('PART (A): reference forward_mha  vs  dumped fwd_mha_out  (per position)')
print('=' * 80)
print(f'{"pos":>3s} {"cos":>12s} {"|ref|":>12s} {"|hpu|":>12s} {"ratio ref/hpu":>14s}')
for i in range(S):
    a = ref[i]; b = hpu[i]
    cs = cossim(a, b)
    print(f'{i:3d} {cs:12.6f} {a.norm().item():12.4f} {b.norm().item():12.4f} {(a.norm()/b.norm()).item():14.4f}')

print()
print('overall cos (all 5 pos flattened):', round(cossim(ref, hpu), 6))
print('overall |ref|=%.4f |hpu|=%.4f ratio=%.4f' % (ref.norm().item(), hpu.norm().item(), (ref.norm()/hpu.norm()).item()))

# per-head cos at pos0 (deterministic: attends only to itself -> out = v[pos0])
print()
print('PER-HEAD cos at pos0 (out should equal value[pos0,head]):')
ref0 = out[:, 0, :]   # [8,256]
hpu0 = hpu[0].view(nh, vh)
for h in range(nh):
    a = ref0[h]; b = hpu0[h]
    print('  head %d: cos=%.6f |ref|=%.4f |hpu|=%.4f' % (h, cossim(a,b), a.norm().item(), b.norm().item()))

# Also: pos0 ref value mixture should be exactly v[:,0,:] (softmax over single token = 1)
vpos0 = vh_[:, 0, :]   # [8,256]
print('sanity: pos0 ref-out == value[pos0]?  maxdiff=%.3e' % (out[:,0,:]-vpos0).abs().max().item())
