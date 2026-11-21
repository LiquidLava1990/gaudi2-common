#!/usr/bin/env python3
"""Path B end-to-end CPU harness for MoeWNA16HPUMethod.

Goal: reproduce, in plain CPU torch (no HPU), the EXACT pipeline the plugin
runs for the routed experts:
  1. upstream gptq weight loader transform (checkpoint int32 -> uint8 params)
  2. plugin _repack_weight (uint8 -> int32 packed for the HPU op)
  3. plugin scale transpose + _sym_zero
  4. EMULATE VllmMixtureOfExpertsOpWNA16: get_dequant_weight == convert_from_uint4
     (q - zero)*scale with g_idx, then the mixture_of_experts matmul/activation
  5. forward a random token routed to a couple experts
  6. compare to an INDEPENDENT trusted per-expert dequant + matmul
  7. check finite + sane magnitude; sanity-check softmax/topk/renormalize routing.

Run with MODEL_DIR pointing at the AutoRound 397B (broken) and at the
Qwen3-30B-A3B-GPTQ (proven) to contrast.
"""
import json, struct, os, sys
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

QT = scalar_types.uint4
MD = os.environ["MODEL_DIR"]
IDX = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]


def load(key):
    path = f"{MD}/{IDX[key]}"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n)); base = 8 + n
        m = hdr[key]; s, e = m["data_offsets"]
        f.seek(base + s); raw = f.read(e - s)
    dt = {"I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16,
          "F32": torch.float32}[m["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).clone().view(*m["shape"])


# ---------------- trusted independent dequant (autogptq LSB-first) ----------
def trusted_dequant(qweight, qzeros, scales, group_size):
    """qweight int32 [in//8,out], qzeros int32 [groups,out//8], scales [groups,out].
    Returns dense W [in,out] = (code - (zcode+1)) * scale  (sym=False/True both;
    autoround sym stores zcode==7 -> true zero 8)."""
    IN = qweight.shape[0] * 8
    OUT = qweight.shape[1]
    sh = torch.arange(8, dtype=torch.int64) * 4
    p = qweight.to(torch.int64)
    moved = p.movedim(0, -1)                                             # [out, in//8]
    codes = ((moved.unsqueeze(-1) >> sh) & 0xF).reshape(*moved.shape[:-1], -1).movedim(-1, 0)  # [in,out]
    pz = qzeros.to(torch.int64)
    zc = ((pz.unsqueeze(-1) >> sh) & 0xF).reshape(qzeros.shape[0], -1)   # [groups,out]
    zc = zc + 1                                                          # autogptq
    g = (torch.arange(IN) // group_size).long()
    z = zc[g].float(); s = scales[g].float()
    W = (codes.float() - z) * s                                         # [in,out]
    return W, codes, zc


# ---------------- emulate the convert_from_uint4 HPU op ---------------------
def emulate_convert_from_uint4(weight_packed, weight_scale, zero_point, group_size, g_idx=None):
    """Mirror torch.ops.hpu.convert_from_uint4(weight_packed, scale, zero_point, dtype, g_idx).
    weight_packed int32 [in, out//8]; weight_scale [groups,out]; zero_point int32 [groups,out//8].
    Returns dequant W [in,out]. With g_idx=None, group index = arange(in)//group_size."""
    codes = unpack_quantized_values_into_int32(weight_packed, QT, packed_dim=1)   # [in,out]
    zc = unpack_quantized_values_into_int32(zero_point, QT, packed_dim=1)         # [groups,out]
    IN = codes.shape[0]
    if g_idx is None:
        g = (torch.arange(IN) // group_size).long()
    else:
        g = g_idx.long()
    z = zc[g].float(); s = weight_scale[g].float()
    return (codes.float() - z) * s                                                # [in,out]


# ---------------- plugin transforms (copied logic) --------------------------
def loader_gptq_weight(qw_i32):
    # moe_wna16 gptq loader: loaded_weight.T.contiguous().view(uint8)
    return qw_i32.T.contiguous().view(torch.uint8)        # [out, in//2]

def loader_gptq_scale(sc):
    return sc.T.contiguous()                              # [out, groups]

def plugin_repack(qweight_u8):
    # _repack_weight for a single expert (E loop done by caller)
    w_i32 = qweight_u8.contiguous().view(torch.int32)     # [out, in//8]
    vals = unpack_quantized_values_into_int32(
        w_i32.transpose(0, 1).contiguous(), QT, packed_dim=0)   # [in,out]
    return pack_quantized_values_into_int32(vals, QT, packed_dim=1)  # [in,out//8]

def plugin_sym_zero(num_groups, out, bits=4):
    zero_val = 1 << (bits - 1)   # 8
    zeros = torch.full((num_groups, out), zero_val, dtype=torch.int32)
    return pack_quantized_values_into_int32(zeros, QT, packed_dim=1)  # [groups,out//8]


# =================== run one proj end-to-end ================================
def run_proj(base, group_size, tag):
    qw = load(base + ".qweight")    # [in//8, out]
    qz = load(base + ".qzeros")     # [groups, out//8]
    sc = load(base + ".scales").float()  # [groups, out]
    IN = qw.shape[0] * 8; OUT = qw.shape[1]; groups = sc.shape[0]
    print(f"\n--- {tag}: {base}")
    print(f"    qweight{list(qw.shape)} IN={IN} OUT={OUT} groups={groups} (IN//gs={IN//group_size})")

    # trusted
    Wt, codes_t, zc_t = trusted_dequant(qw, qz, sc, group_size)
    zc_unique = torch.unique(zc_t).tolist()
    print(f"    trusted zeros unique (+1): {zc_unique[:6]}  W range [{Wt.min():.4f},{Wt.max():.4f}] finite={torch.isfinite(Wt).all().item()}")

    # plugin path
    w8 = loader_gptq_weight(qw)               # [out, in//2]
    packed = plugin_repack(w8)                # [in, out//8]
    sc_loaded = loader_gptq_scale(sc)         # [out, groups]
    sc_plugin = sc_loaded.transpose(0, 1).contiguous()  # [groups, out]
    zp = plugin_sym_zero(groups, OUT)         # [groups, out//8] value 8

    Wp = emulate_convert_from_uint4(packed, sc_plugin, zp, group_size)  # [in,out]
    diff = (Wp - Wt).abs()
    print(f"    plugin W range [{Wp.min():.4f},{Wp.max():.4f}] finite={torch.isfinite(Wp).all().item()}")
    print(f"    WEIGHT max abs diff vs trusted: {diff.max().item():.3e}  mean {diff.mean().item():.3e}")
    return Wt, Wp, IN, OUT


# =================== full expert forward (gate/up/down) =====================
def run_expert_forward(layer_idx, expert_id, hidden, moe_inter, group_size, model_kind):
    """Build w13 = [gate;up] fused (out=2*moe_inter), w2 = down (out=hidden).
    Run SwiGLU expert: y = (silu(x@gate.T?) ... ) using vLLM convention.
    The mixture_of_experts op computes, per token routed to expert e:
        h = act(x @ w13_dequant) ; here w13 dequant is [in=hidden, out=2*inter]
        gate, up = split ; a = silu(gate) * up
        out = a @ w2_dequant  ; w2 dequant [in=inter, out=hidden]
    We replicate with both trusted and plugin dequant weights and compare."""
    if model_kind == "397b":
        pre = f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}"
    else:
        pre = f"model.layers.{layer_idx}.mlp.experts.{expert_id}"
    Wg_t, Wg_p, IN, OUT = run_proj(pre + ".gate_proj", group_size, f"L{layer_idx}E{expert_id} gate")
    Wu_t, Wu_p, _, _ = run_proj(pre + ".up_proj", group_size, f"L{layer_idx}E{expert_id} up")
    Wd_t, Wd_p, IN2, OUT2 = run_proj(pre + ".down_proj", group_size, f"L{layer_idx}E{expert_id} down")

    torch.manual_seed(0)
    x = torch.randn(4, hidden)   # O(1) hidden state magnitude

    def fwd(Wg, Wu, Wd):
        # W*_dequant is [in, out]; vLLM linear y = x @ W (in->out)  (W already x-orient)
        g = x @ Wg               # [tok, inter]
        u = x @ Wu               # [tok, inter]
        a = F.silu(g) * u        # [tok, inter]
        return a @ Wd            # [tok, hidden]

    y_t = fwd(Wg_t, Wu_t, Wd_t)
    y_p = fwd(Wg_p, Wu_p, Wd_p)
    print(f"\n=== expert forward L{layer_idx}E{expert_id} ===")
    print(f"    trusted out range [{y_t.min():.4f},{y_t.max():.4f}] finite={torch.isfinite(y_t).all().item()}")
    print(f"    plugin  out range [{y_p.min():.4f},{y_p.max():.4f}] finite={torch.isfinite(y_p).all().item()}")
    print(f"    max abs diff trusted vs plugin: {(y_t-y_p).abs().max().item():.3e}")


# =================== routing sanity =========================================
def routing_sanity(num_experts, top_k):
    print(f"\n=== routing sanity (E={num_experts}, top_k={top_k}) ===")
    torch.manual_seed(1)
    for scale in [1.0, 1e3, 1e-6, 0.0]:
        logits = torch.randn(4, num_experts) * scale
        if scale == 0.0:
            logits = torch.zeros(4, num_experts)
        tw = F.softmax(logits, dim=1, dtype=torch.float32)
        topk_w, topk_id = torch.topk(tw, top_k, dim=-1)
        denom = topk_w.sum(dim=-1, keepdim=True)
        renorm = topk_w / denom
        print(f"    logit_scale={scale}: topk_w finite={torch.isfinite(topk_w).all().item()} "
              f"denom min={denom.min().item():.3e} renorm finite={torch.isfinite(renorm).all().item()}")


if __name__ == "__main__":
    kind = os.environ.get("MODEL_KIND", "397b")
    if kind == "397b":
        run_expert_forward(0, 0, 4096, 1024, 128, kind)
        run_expert_forward(0, 5, 4096, 1024, 128, kind)
        run_expert_forward(30, 100, 4096, 1024, 128, kind)
        routing_sanity(512, 10)
    else:
        run_expert_forward(0, 0, 2048, 768, 128, kind)
        routing_sanity(128, 8)
    print("\nDONE")
