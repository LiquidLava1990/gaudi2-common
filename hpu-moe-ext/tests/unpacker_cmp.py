#!/usr/bin/env python3
"""Which unpacker is correct for AutoRound's auto_gptq packing?
Compares, on ONE real AutoRound attention tensor (q_proj) AND one expert tensor:
  TRUSTED  : autogptq LSB-first nibble unpack (value j in bits [4j,4j+4) along packed dim)
  CT       : vLLM unpack_quantized_values_into_int32 (used by the PROVEN expert _repack path)
  OLD      : replicate hpu_gptq.unpack_weight_from_cuda_old_format (used by my dequant-linear path)
If OLD != TRUSTED, the dequant-linear path is using the wrong nibble order -> word salad.
Run in gaudi container with model mounted at /m.
"""
import json, struct, os, torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

MD = os.environ.get("MODEL_DIR", "/m")
IDX = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
QT = scalar_types.uint4

def load(key):
    p = f"{MD}/{IDX[key]}"
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); b = 8+n
        m = h[key]; s, e = m["data_offsets"]; f.seek(b+s); raw = f.read(e-s)
    dt = {"I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}[m["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).clone().view(*m["shape"])

def trusted_lsb(qweight):  # [in//8, out] -> codes [in, out]
    p = qweight.to(torch.int64)
    sh = torch.arange(8, dtype=torch.int64) * 4
    moved = p.movedim(0, -1)                       # [out, in//8]
    out = (moved.unsqueeze(-1) >> sh) & 0xF        # [out, in//8, 8]
    return out.reshape(*moved.shape[:-1], -1).movedim(-1, 0).to(torch.int32)  # [in, out]

def ct_unpack(qweight):    # what _repack_weight does: transpose then CT-unpack packed_dim=0
    return unpack_quantized_values_into_int32(qweight.contiguous(), QT, packed_dim=0)  # [in, out]

def old_unpack(qweight):   # replicate hpu_gptq.unpack_weight_from_cuda_old_format (bits=4)
    bits = 4
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)  # [1,8]
    # cuda_old_format: weight = bitwise_right_shift(qweight.unsqueeze(1), wf.unsqueeze(-1)) then &0xF
    w = torch.bitwise_right_shift(qweight.unsqueeze(1), wf.unsqueeze(-1)).to(torch.int16)
    w = torch.bitwise_and(w, (2**bits)-1)
    # shape [in//8, 8, out] -> [in, out]
    return w.reshape(-1, w.shape[-1]).to(torch.int32)

for tag, key in [("ATTN q_proj", None), ("EXPERT gate_proj", None)]:
    if "ATTN" in tag:
        key = next(k for k in IDX if "self_attn.q_proj.qweight" in k)
    else:
        key = next(k for k in IDX if ".experts.0.gate_proj.qweight" in k)
    qw = load(key)
    print(f"\n=== {tag} ({key.split('.')[-2]}) qweight {list(qw.shape)} ===")
    tr = trusted_lsb(qw)
    try:
        ct = ct_unpack(qw)
        print("  CT  shape", list(ct.shape), "== trusted:",
              bool(list(ct.shape) == list(tr.shape) and (ct == tr).all()))
    except Exception as e:
        print("  CT  error:", e)
    try:
        old = old_unpack(qw)
        print("  OLD shape", list(old.shape), "== trusted:",
              bool(list(old.shape) == list(tr.shape) and (old == tr).all()),
              "| mismatch frac:", round((old != tr).float().mean().item(), 4) if list(old.shape)==list(tr.shape) else "shape-diff")
    except Exception as e:
        print("  OLD error:", e)
    print("  trusted codes[0,:8]:", tr[0, :8].tolist())
