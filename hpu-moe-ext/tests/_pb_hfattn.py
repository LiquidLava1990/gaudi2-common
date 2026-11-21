import torch, glm52_parallel_block_drift as M
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding
torch.set_grad_enabled(False); DT = torch.float64
cfg = AutoConfig.from_pretrained(M.CKPT, trust_remote_code=True)
cfg._attn_implementation = 'eager'
L = 3  # indexer_type 'shared' -> indexer=None
attn = GlmMoeDsaAttention(cfg, L).double().eval()
P = f'model.layers.{L}.self_attn'
sd = {}
for n in ['q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj','o_proj']:
    sd[f'{n}.weight'] = M.get(f'{P}.{n}.weight').double()
# norms: checkpoint stores (true-1); HF RMSNorm is plain weight -> load (cw+1)
for n in ['q_a_layernorm','kv_a_layernorm']:
    sd[f'{n}.weight'] = M.get(f'{P}.{n}.weight').double() + 1.0
missing, unexpected = attn.load_state_dict(sd, strict=False)
print('unexpected:', unexpected, ' missing(non-indexer):', [m for m in missing if 'indexer' not in m])

S = 5
torch.manual_seed(1); x = torch.randn(1, S, M.H, dtype=DT) * 0.5   # a normed-hidden stand-in
rot = GlmMoeDsaRotaryEmbedding(cfg).double()
pos = torch.arange(S)[None]
cos, sin = rot(x, pos)
prev_topk = torch.arange(S)[None, None, :].expand(1, S, S).contiguous()   # all tokens selected -> full attn
cmask = torch.full((1, 1, S, S), 0.0, dtype=DT)
cmask = cmask.masked_fill(torch.ones(S, S, dtype=torch.bool).triu(1)[None, None], torch.finfo(DT).min)
out_hf = attn(x, (cos, sin), cmask, position_ids=pos, prev_topk_indices=prev_topk)[0][0]   # [S,H]

# my attn expects [S,H] normed input
out_me = M.attn(L, x[0])
def cos_(a, b):
    a = a.flatten(); b = b.flatten(); return float((a@b)/(a.norm()*b.norm()))
print('HF attn vs my attn:  cos=%.6f  nrat=%.4f  maxdiff=%.3e' %
      (cos_(out_hf, out_me), out_me.norm().item()/out_hf.norm().item(), (out_hf-out_me).abs().max().item()))
