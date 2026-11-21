#!/usr/bin/env python3
"""Non-circular numeric diagnostic for an AutoRound (auto_gptq packing) checkpoint.

Loads ONE real expert's gate_proj qweight/qzeros/scales from a *sharded* AutoRound
checkpoint, then compares two independent dequantizations:
  [A] plugin path : unpack via vLLM unpack_quantized_values_into_int32 (what
      MoeWNA16HPUMethod._repack_weight assumes) -> the autogptq/compressed-tensors order.
  [REF] trusted   : autogptq LSB-first nibble unpack (value j in bits [4j,4j+4)).
If [A] != [REF] -> AutoRound packs qweight in a DIFFERENT nibble order than the
plugin assumes -> repack scrambles weights -> garbage. Then test zero=8 vs 9.

Usage (in the gaudi container, model mounted at /m):
  MODEL_DIR=/m python3 autoround_diag.py
"""
import json, struct, os, torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

MD = os.environ.get("MODEL_DIR", "/m")
IDX = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
GROUP = 128
QT = scalar_types.uint4

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

# locate expert 0 gate_proj
base = None
for k in IDX:
    if ".mlp.experts.0.gate_proj.qweight" in k:
        base = k[:-len(".qweight")]; break
assert base, "no expert gate_proj found"
print("expert base:", base)
qweight = load(base + ".qweight")        # int32 [in//8, out] (autogptq)
scales  = load(base + ".scales").float() # [groups, out]
qzeros  = load(base + ".qzeros")          # int32 [groups, out//8]
IN = qweight.shape[0] * 8; OUT = qweight.shape[1]
groups = scales.shape[0]
print(f"shapes: qweight {list(qweight.shape)} -> in={IN} out={OUT}, groups={groups}")

# --- [REF] trusted autogptq LSB-first unpack along the packed (input) dim ---
def gptq_unpack_lsb(packed_i32, axis):
    p = packed_i32.to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    moved = p.movedim(axis, -1)
    out = (moved.unsqueeze(-1) >> shifts) & 0xF
    out = out.reshape(*moved.shape[:-1], -1).movedim(-1, axis)
    return out.to(torch.int32)

ref_codes = gptq_unpack_lsb(qweight, axis=0)          # [in, out]
# --- [A] plugin/vLLM unpack: transpose so packed dim is 0 then unpack ---
a_codes = unpack_quantized_values_into_int32(
    qweight.transpose(0, 1).contiguous(), QT, packed_dim=0)  # expects [out_packed?, ...]
# vLLM unpack on the transposed gives [in, out]; align shape
print("[A] vLLM-unpacked codes shape:", list(a_codes.shape),
      " [REF] shape:", list(ref_codes.shape))
try:
    diff = (a_codes.flatten().sort().values[:OUT*IN] !=
            ref_codes.flatten().sort().values[:OUT*IN])
except Exception:
    pass
# direct elementwise if shapes match
if list(a_codes.shape) == list(ref_codes.shape):
    mm = (a_codes != ref_codes).float().mean().item()
    print(f"[A vs REF] elementwise nibble mismatch fraction = {mm:.4%}")
else:
    # compare value histograms as a weaker signal
    ha = torch.bincount(a_codes.flatten(), minlength=16)
    hr = torch.bincount(ref_codes.flatten(), minlength=16)
    print("[A vs REF] shape differ; code histograms equal:", bool((ha == hr).all()))

# --- [B] dequant with zero=8 vs zero=9, against trusted ref dequant ---
g_idx = (torch.arange(IN) // GROUP).long()
zsc = scales[g_idx]                                  # [in,out]
ref_w = (ref_codes - 8).float() * zsc                # trusted: zero=8
for z in (8, 9):
    w = (ref_codes - z).float() * zsc
    md = (w - ref_w).abs().max().item()
    print(f"  zero={z}: maxdiff vs trusted(zero=8) = {md:.5f}")
print("sample ref_codes[0,:8]:", ref_codes[0, :8].tolist())
print("done")
