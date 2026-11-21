#!/usr/bin/env python3
"""Abliterate GLM-5.2 by projecting a refusal direction out of the attention o_proj weights.

Edits ONLY the non-expert graft (glm52_fp8_other.safetensors) -> keeps the int4 experts + W4A8 +
int4-KV perf stack 100% intact (o_proj is orthogonal to the MoE). For each of the 78 layers:

    W   = o_proj.weight (fp8 [6144, 16384]) dequantised by its per-tensor weight_scale
    W'  = (I - r rᵀ) W      # remove the r-component from everything o_proj writes to the residual
    weight' = (W' / scale) -> fp8 e4m3fn     # requantise, keep the original per-tensor scale

r is a UNIT vector in the 6144-d residual space (the refusal direction). Supply it via --direction
(a .pt/.safetensors holding a [6144] or [n_layers,6144] float tensor). Providing it per-layer
([78,6144]) abliterates each layer with its own direction; a single [6144] is applied to all.

Dequant uses float32 (NOT bfloat16) to sidestep the Gaudi fp8->bf16 top-bin bug; this runs on CPU.
Coherence + refusal behaviour MUST be gated on TEXT after applying (a too-strong r degrades quality).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0


def _load_direction(path: str, n_layers: int, hidden: int) -> torch.Tensor:
    if path.endswith(".safetensors"):
        with safe_open(path, framework="pt") as f:
            r = f.get_tensor(f.keys().__iter__().__next__())
    else:
        r = torch.load(path, map_location="cpu")
        if isinstance(r, dict):
            r = next(iter(r.values()))
    r = r.float()
    if r.dim() == 1:
        r = r.unsqueeze(0).expand(n_layers, -1)
    assert r.shape[-1] == hidden, f"direction last dim {r.shape[-1]} != hidden {hidden}"
    # unit-normalise each layer's direction (projection needs ||r||=1)
    return r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graft", default="/mnt/fastmodels/glm52-fp8-other/glm52_fp8_other.safetensors")
    ap.add_argument("--direction", required=True, help="[6144] or [78,6144] unit refusal direction")
    ap.add_argument("--out", default="/mnt/fastmodels/glm52-fp8-other-abliterated/glm52_fp8_other.safetensors")
    ap.add_argument("--scale", type=float, default=1.0, help="strength multiplier on r (1.0 = full ablation)")
    ap.add_argument("--dry-run", action="store_true", help="report per-layer perturbation, write nothing")
    args = ap.parse_args()

    n_layers, hidden = 78, 6144
    r_all = _load_direction(args.direction, n_layers, hidden) * args.scale

    tensors: dict[str, torch.Tensor] = {}
    edited = 0
    max_rel = 0.0
    with safe_open(args.graft, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith("self_attn.o_proj.weight") and t.dtype == _FP8:
                li = int(k.split(".layers.")[1].split(".")[0])
                scale = f.get_tensor(k + ".weight_scale").float()
                W = t.float() * scale                                   # [6144, 16384]
                r = r_all[li]                                           # [6144]
                Wp = W - torch.outer(r, r @ W)                         # (I - r rᵀ) W
                rel = (Wp - W).norm() / W.norm().clamp_min(1e-9)
                max_rel = max(max_rel, float(rel))
                if not args.dry_run:
                    q = (Wp / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
                    tensors[k] = q
                edited += 1
            elif not args.dry_run:
                tensors[k] = t
    print(f"[abliterate] o_proj layers edited: {edited}/{n_layers}  max per-layer relΔ: {max_rel:.4%}")
    if args.dry_run:
        print("[abliterate] dry-run: nothing written."); return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata={"format": "pt", "abliterated": "o_proj"})
    src_idx = os.path.join(os.path.dirname(args.graft), "model.safetensors.index.json")
    if os.path.exists(src_idx):
        idx = json.load(open(src_idx))
        json.dump(idx, open(os.path.join(os.path.dirname(args.out), "model.safetensors.index.json"), "w"))
    print(f"[abliterate] wrote {args.out}  ({edited} o_proj projected, relΔ≤{max_rel:.2%})")


if __name__ == "__main__":
    main()
