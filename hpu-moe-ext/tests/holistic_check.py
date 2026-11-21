#!/usr/bin/env python3
"""Holistic CPU check: emulate the EXACT GPTQDequantToBF16LinearMethod pipeline
(inherited hpu_gptq unpackers + the plugin's process_weights_after_loading math)
for a real Qwen3.5 full-attention q_proj AND a linear_attn in_proj_qkv, and
compare layer.weight ELEMENTWISE to an independent trusted dequant.

Also probe scale-group ordering and qzeros uniformity.
"""
import json, struct, os, torch

MD = os.environ.get("MODEL_DIR", "/m")
IDX = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
GROUP = 128
BITS = 4

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

# ---- trusted independent autogptq dequant (LSB-first nibble) ----
def trusted_dequant(qweight, qzeros, scales):
    IN = qweight.shape[0]*8; OUT = qweight.shape[1]
    p = qweight.to(torch.int64)
    sh = torch.arange(8, dtype=torch.int64)*4
    moved = p.movedim(0,-1)                            # [out, in//8]
    codes = ((moved.unsqueeze(-1) >> sh) & 0xF).reshape(*moved.shape[:-1], -1).movedim(-1,0)  # [in,out]
    pz = qzeros.to(torch.int64)
    zc = ((pz.unsqueeze(-1) >> sh) & 0xF).reshape(pz.shape[0], -1)  # [groups,out]
    zc = zc + 1                                        # autogptq true zero
    g = (torch.arange(IN)//GROUP).long()
    z = zc[g].float(); s = scales[g].float()
    W = (codes.float() - z) * s                        # [in,out]
    return W, codes.to(torch.int32), zc.to(torch.int32)

# ---- emulate inherited hpu_gptq unpackers exactly ----
def old_unpack_weight(qweight):
    wf = torch.tensor(list(range(0,32,BITS)),dtype=torch.int32).unsqueeze(0)
    w = torch.bitwise_right_shift(qweight.unsqueeze(1), wf.unsqueeze(-1)).to(torch.int8)
    w = torch.bitwise_and(w, (2**BITS)-1)
    return w.reshape((w.shape[0]*w.shape[1], w.shape[2])).to(torch.int32)   # [in,out]

def old_unpack_zeros(qzeros, scales_dtype):
    wf = torch.tensor(list(range(0,32,BITS)),dtype=torch.int32).unsqueeze(0)
    z = torch.bitwise_right_shift(torch.unsqueeze(qzeros,2).expand(-1,-1,32//BITS), wf.unsqueeze(0)).to(torch.int8)
    z = z + 1
    z = torch.bitwise_and(z,(2**BITS)-1)
    z = z.reshape(-1, z.shape[1]*z.shape[2])
    return z   # [groups,out]

def plugin_dequant(qweight, qzeros, scales):
    codes = old_unpack_weight(qweight).to(torch.float32)   # [in,out]
    zeros = old_unpack_zeros(qzeros, scales.dtype).to(torch.float32)  # [groups,out]
    sc = scales.to(torch.float32)
    gs = GROUP
    g_idx = (torch.arange(codes.shape[0])//gs).long()
    z = zeros[g_idx]; s = sc[g_idx]
    w = (codes - z)*s          # [in,out]
    weight = w.t().contiguous()  # [out,in]  (what apply F.linear uses)
    return weight, codes.to(torch.int32), zeros.to(torch.int32)

def check(tag, base):
    print(f"\n===== {tag}: {base} =====")
    qw = load(base+".qweight"); qz = load(base+".qzeros"); sc = load(base+".scales").float()
    IN = qw.shape[0]*8; OUT = qw.shape[1]; groups = sc.shape[0]
    print(f"qweight{list(qw.shape)} in={IN} out={OUT} groups={groups} scales{list(sc.shape)} qzeros{list(qz.shape)}")
    print(f"  IN//GROUP = {IN//GROUP} ; matches groups? {IN//GROUP==groups}")
    refW, refc, refz = trusted_dequant(qw, qz, sc)
    plW, plc, plz = plugin_dequant(qw, qz, sc)
    # plW is [out,in]; refW is [in,out]
    print(f"  zeros unique (trusted, +1): {torch.unique(refz).tolist()[:8]}")
    print(f"  codes mismatch frac: {(plc!=refc).float().mean().item():.4%}")
    print(f"  zeros mismatch frac: {(plz!=refz).float().mean().item():.4%}")
    diff = (plW - refW.t()).abs()
    print(f"  WEIGHT [out,in] max abs diff vs trusted.t(): {diff.max().item():.6e}  mean {diff.mean().item():.3e}")
    print(f"  ref W range [{refW.min():.4f},{refW.max():.4f}]")

# full-attention q/k/v/o
check("FULL-ATTN q_proj (layer3)", "model.language_model.layers.3.self_attn.q_proj")
check("FULL-ATTN o_proj (layer3)", "model.language_model.layers.3.self_attn.o_proj")
# linear-attn projections
check("LINEAR-ATTN in_proj_qkv (layer0)", "model.language_model.layers.0.linear_attn.in_proj_qkv")
check("LINEAR-ATTN out_proj (layer0)", "model.language_model.layers.0.linear_attn.out_proj")
print("\nDONE")
