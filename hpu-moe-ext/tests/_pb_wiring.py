"""Validate harness WIRING: run first K layers via real HF GlmMoeDsaDecoderLayer (forcing full
attention, no indexer) on the REAL embed, compare hidden to the harness forward's first-K output."""
import torch, glm52_parallel_block_drift as M
from transformers import AutoConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as Mod
torch.set_grad_enabled(False); DT = torch.float64
cfg = AutoConfig.from_pretrained(M.CKPT, trust_remote_code=True); cfg._attn_implementation = 'eager'
S = M.S; KMAX = 3
rot = Mod.GlmMoeDsaRotaryEmbedding(cfg).double(); pos = torch.arange(S)[None]
cos, sin = rot(torch.zeros(1, S, 8, dtype=DT), pos)
cm = torch.zeros(1, 1, S, S, dtype=DT).masked_fill(torch.ones(S, S, dtype=torch.bool).triu(1)[None, None], torch.finfo(DT).min)
prev_topk = torch.arange(S)[None, None, :].expand(1, S, S).contiguous()   # all tokens -> full attn

def load_layer(L):
    lyr = Mod.GlmMoeDsaDecoderLayer(cfg, L).double().eval()
    P = f'model.layers.{L}'
    sd = {}
    for n in ['q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj','o_proj']:
        sd[f'self_attn.{n}.weight'] = M.get(f'{P}.self_attn.{n}.weight').double()
    for n in ['q_a_layernorm','kv_a_layernorm']:
        sd[f'self_attn.{n}.weight'] = M.get(f'{P}.self_attn.{n}.weight').double() + 1.0
    sd['input_layernorm.weight'] = M.get(f'{P}.input_layernorm.weight').double() + 1.0
    sd['post_attention_layernorm.weight'] = M.get(f'{P}.post_attention_layernorm.weight').double() + 1.0
    if L < M.fkd:
        for n in ['gate_proj','up_proj','down_proj']:
            sd[f'mlp.{n}.weight'] = M.get(f'{P}.mlp.{n}.weight').double()
    lyr.load_state_dict(sd, strict=False)
    lyr.self_attn.indexer = None   # force full attention (prev_topk = all tokens)
    return lyr

# HF reference forward (first KMAX layers)
h = M.embed_in[None].clone()   # [1,S,H]
for L in range(KMAX):
    lyr = load_layer(L)
    h = lyr(h, position_embeddings=(cos, sin), attention_mask=cm, position_ids=pos,
            prev_topk_indices=prev_topk)[0]

# harness forward (first KMAX layers), capture hidden after KMAX
hidden = M.embed_in.clone(); residual = torch.zeros_like(hidden)
for L in range(KMAX):
    s = hidden if L == 0 else hidden + residual
    a = M.attn(L, M.rms(s, M.get(f'model.layers.{L}.input_layernorm.weight')))
    spa = s + a
    m = M.mlp(L, M.rms(spa, M.get(f'model.layers.{L}.post_attention_layernorm.weight')))
    hidden = a + m; residual = s
mine = hidden + residual   # residual stream after layer KMAX-1

def cs(a, b): a = a.flatten(); b = b.flatten(); return float((a@b)/(a.norm()*b.norm()))
print(f'WIRING first {KMAX} layers: cos(HF, harness)=%.6f  nrat=%.4f  maxdiff=%.3e'
      % (cs(h[0], mine), mine.norm().item()/h[0].norm().item(), (h[0]-mine).abs().max().item()))
print('  ROPE_STYLE=', M.ROPE_STYLE)
