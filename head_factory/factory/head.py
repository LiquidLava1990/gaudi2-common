"""Universal MTP draft head — config-driven, trains for ANY target model.

Depth-1 next-next-token draft: consumes (prev_hidden h_t, embed(token_{t+1})) and predicts
token_{t+2}. Frozen embed + lm_head (the draft must speak the target's vocab geometry);
TRAINED = enorm/hnorm/eh_proj + one lean decoder block (attn + gated MLP) + shared_head norm.
The lean block is a from-scratch transformer layer (NOT the target's MoE) — cheap to run/train.

Per-model knobs live in a HeadConfig (dims, norm type, rope, activation, mlp-dim). GLM: plain RMS
+ silu; M3: Gemma RMS + swigluoai. Everything else identical => one factory, many heads."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    hidden: int
    vocab: int
    n_heads: int
    head_dim: int
    mlp_dim: int = 8192
    norm: str = "rms"          # "rms" (GLM/Llama: x*w) | "gemma" (M3: x*(1+w))
    act: str = "silu"          # "silu" | "swigluoai"
    rope_theta: float = 1.0e6
    rope_dim: int = 0          # 0 => head_dim (full rope)
    rope_neox: bool = True
    eps: float = 1.0e-5
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    n_blocks: int = 2          # stacked draft blocks (capacity)
    use_attn: bool = False     # OFF: serve-correct Medusa-style MLP head (depth-1 decode has no seq context,
                               #      so the draft must not rely on sequence self-attention). ON only for
                               #      experiments where train==serve sequence context is guaranteed.
    n_aux: int = 0             # 0 = legacy single-final-hidden; 3 = EAGLE-3 low/mid/high aux-residual fusion
    fc_norm: bool = True       # per-aux RMSNorm before fc (stabilizes the un-normed residual-stream aux)


def _norm(x, w, cfg: HeadConfig):
    d = x.float()
    d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + cfg.eps)
    scale = (1.0 + w.float()) if cfg.norm == "gemma" else w.float()
    return (d * scale).to(x.dtype)


def _act(gate_up, cfg: HeadConfig):
    half = gate_up.shape[-1] // 2
    g, u = gate_up[..., :half], gate_up[..., half:]
    if cfg.act == "swigluoai":
        g = g.clamp(max=cfg.swiglu_limit)
        u = u.clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        return (u + 1.0) * (g * torch.sigmoid(g * cfg.swiglu_alpha))
    return F.silu(g) * u


def _rope(positions, dim, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    ang = positions.float()[:, None] * inv[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    return cos.to(dtype), sin.to(dtype)


def _rotate_neox(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class LeanBlock(nn.Module):
    """A draft block. MLP-only by default (post_norm -> gated MLP -> +res): serve-correct for depth-1 decode
    where there is no sequence context. With cfg.use_attn, additionally does input_norm -> MHA(rope) -> +res
    first (experiment-only; NOT serve-correct under per-token decode)."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        H, nh, hd = cfg.hidden, cfg.n_heads, cfg.head_dim
        ni = torch.ones(H) if cfg.norm == "rms" else torch.zeros(H)  # scale=1 at init (rms: w, gemma: 1+w)
        self.post_ln = nn.Parameter(ni.clone())
        self.gate_up = nn.Linear(H, 2 * cfg.mlp_dim, bias=False)
        self.down = nn.Linear(cfg.mlp_dim, H, bias=False)
        if cfg.use_attn:
            self.input_ln = nn.Parameter(ni.clone())
            self.q = nn.Linear(H, nh * hd, bias=False)
            self.k = nn.Linear(H, nh * hd, bias=False)
            self.v = nn.Linear(H, nh * hd, bias=False)
            self.o = nn.Linear(nh * hd, H, bias=False)

    def _attn(self, x, positions):
        cfg = self.cfg; T = x.shape[0]; nh, hd = cfg.n_heads, cfg.head_dim
        q = self.q(x).view(T, nh, hd); k = self.k(x).view(T, nh, hd); v = self.v(x).view(T, nh, hd)
        rd = cfg.rope_dim or hd
        cos, sin = _rope(positions, rd, cfg.rope_theta, x.device, x.dtype)
        cos = cos.repeat(1, 2)[:, None, :]; sin = sin.repeat(1, 2)[:, None, :]
        qr, kr = q[..., :rd], k[..., :rd]
        q = torch.cat([qr * cos + _rotate_neox(qr) * sin, q[..., rd:]], -1)
        k = torch.cat([kr * cos + _rotate_neox(kr) * sin, k[..., rd:]], -1)
        q = q.transpose(0, 1); k = k.transpose(0, 1); v = v.transpose(0, 1)  # [nh,T,hd]
        out = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
        return self.o(out.transpose(0, 1).reshape(T, nh * hd))

    def forward(self, x, positions):
        if self.cfg.use_attn:
            x = x + self._attn(_norm(x, self.input_ln, self.cfg), positions)
        x = x + self.down(_act(self.gate_up(_norm(x, self.post_ln, self.cfg)), self.cfg))
        return x


class DraftHead(nn.Module):
    """Full depth-1 draft head. embed_w + lm_head_w are FROZEN (passed in)."""

    def __init__(self, cfg: HeadConfig, embed_w: torch.Tensor, lm_head_w: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        ni = torch.ones(cfg.hidden) if cfg.norm == "rms" else torch.zeros(cfg.hidden)  # scale=1 at init
        self.enorm = nn.Parameter(ni.clone())
        self.hnorm = nn.Parameter(ni.clone())
        self.eh_proj = nn.Linear(cfg.hidden * 2, cfg.hidden, bias=False)
        if cfg.n_aux:
            self.fc = nn.Linear(cfg.hidden * cfg.n_aux, cfg.hidden, bias=False)   # EAGLE-3 fusion 3H -> H
            self.fc_norm = nn.ParameterList([nn.Parameter(ni.clone()) for _ in range(cfg.n_aux)]) \
                if cfg.fc_norm else None
        self.blocks = nn.ModuleList([LeanBlock(cfg) for _ in range(cfg.n_blocks)])
        self.final_norm = nn.Parameter(ni.clone())
        self.register_buffer("embed_w", embed_w, persistent=False)     # [vocab, H] frozen
        self.register_buffer("lm_head_w", lm_head_w, persistent=False)  # [vocab, H] frozen

    def _fuse_aux(self, aux):                       # aux: [T, n_aux*H] -> [T, H] (EAGLE-3 low/mid/high fusion)
        if getattr(self, "fc_norm", None) is not None:
            ch = aux.chunk(self.cfg.n_aux, dim=-1)
            aux = torch.cat([_norm(c, w, self.cfg) for c, w in zip(ch, self.fc_norm)], dim=-1)
        return self.fc(aux)

    def forward_hidden_embeds(self, prev_hidden, inputs_embeds, positions):
        """Core hidden path from PRE-COMPUTED token embeddings (the vllm serve path: inputs_embeds comes
        from the shared TARGET embed_tokens, not our frozen embed_w). Returns final hidden [T,H].
        EAGLE-3: if prev_hidden is the wide [T, n_aux*H] aux stack, fuse it (fc 3H->H) first; width-detected
        so the SAME module trains+serves with either legacy [T,H] or fused [T,3H] input."""
        if self.cfg.n_aux and prev_hidden.shape[-1] != self.cfg.hidden:
            prev_hidden = self._fuse_aux(prev_hidden)
        e = torch.where(positions.unsqueeze(-1) == 0, torch.zeros_like(inputs_embeds), inputs_embeds)
        x = self.eh_proj(torch.cat([_norm(e, self.enorm, self.cfg),
                                    _norm(prev_hidden, self.hnorm, self.cfg)], dim=-1))
        for blk in self.blocks:
            x = blk(x, positions)
        return _norm(x, self.final_norm, self.cfg)

    def forward_hidden(self, prev_hidden, tok_next, positions):
        """Everything UP TO (not incl.) the lm_head -> final hidden [T,H]. Training/eval path: embeds
        tok_next with our frozen embed_w, then runs the shared core."""
        e = F.embedding(tok_next, self.embed_w)
        return self.forward_hidden_embeds(prev_hidden, e, positions)

    def compute_logits(self, hidden):
        """final hidden [T,H] -> logits [T,vocab] (the EAGLE proposer calls this after forward)."""
        return F.linear(hidden, self.lm_head_w)

    def forward(self, prev_hidden, tok_next, positions):
        """prev_hidden [T,H], tok_next [T] (token id at t+1), positions [T] -> logits [T,vocab]."""
        return self.compute_logits(self.forward_hidden(prev_hidden, tok_next, positions))

    def trainable_params(self):
        return [p for n, p in self.named_parameters() if "embed_w" not in n and "lm_head_w" not in n]
