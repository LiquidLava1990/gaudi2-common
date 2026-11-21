#!/usr/bin/env python3
"""Non-circular numeric proof for the AutoRound 397B checkpoint (CPU).

We do NOT trust the plugin. We build an INDEPENDENT autogptq dequant from raw
bytes (LSB-first nibble unpack, true_zero = stored_nibble + 1), then check that
the vLLM unpack helpers the plugin relies on agree with it, and that the
plugin's repack/scale-transpose preserves the dequant matrix.

Covers a routed expert AND a linear_attn quantized projection (the linear path
that goes through AutoRoundHPUConfig -> GPTQHPULinearMethod).
"""
import json, struct, os, torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32, pack_quantized_values_into_int32)
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

def gptq_unpack_lsb(packed_i32, axis):
    """Independent autogptq unpack: nibble j lives in bits [4j,4j+4) along axis."""
    p = packed_i32.to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    moved = p.movedim(axis, -1)
    out = (moved.unsqueeze(-1) >> shifts) & 0xF
    out = out.reshape(*moved.shape[:-1], -1).movedim(-1, axis)
    return out.to(torch.int32)

def indep_dequant(qweight, qzeros, scales):
    """Independent reference: returns dequant W of shape [in, out]."""
    IN = qweight.shape[0] * 8; OUT = qweight.shape[1]
    codes = gptq_unpack_lsb(qweight, axis=0)            # [in, out]
    # qzeros [groups, out//8] -> unpack along axis=1 -> [groups, out]; +1 true zero
    zcodes = gptq_unpack_lsb(qzeros, axis=1) + 1        # [groups, out]
    g = (torch.arange(IN) // GROUP).long()
    z = zcodes[g].float()                               # [in,out]
    s = scales[g].float()                               # [in,out]
    return (codes.float() - z) * s, codes, zcodes

def plugin_repack_dequant(qweight, scales):
    """Reproduce MoeWNA16HPUMethod._repack_weight + apply sym zero=8 + scale.
    The HPU op consumes int32-packed [in, out//8] (dequant [in,out]).
    Loader hands experts as uint8 [out, in//2] (=int32 [in//8,out].T viewed u8).
    We start from the ORIGINAL int32 [in//8,out] and emulate the loader+repack."""
    IN = qweight.shape[0] * 8; OUT = qweight.shape[1]
    # loader: loaded_weight.T.contiguous().view(uint8) -> [out, in//2] uint8
    u8 = qweight.t().contiguous().view(torch.uint8)     # [out, in//2]
    # _repack_weight: view back int32 [out,in//8], transpose->[in//8,out], unpack->[in,out]
    w_i32 = u8.contiguous().view(torch.int32)           # [out, in//8]
    vals = unpack_quantized_values_into_int32(
        w_i32.transpose(0, 1).contiguous(), QT, packed_dim=0)   # [in, out]
    packed = pack_quantized_values_into_int32(vals, QT, packed_dim=1)  # [in, out//8]
    # now UNPACK again to recover the codes the op would see (op does code-zero)
    codes_op = unpack_quantized_values_into_int32(packed, QT, packed_dim=1)  # [in,out]
    # scales: loader gives [out, groups]?? No: scales stored [groups,out]; loader .T -> [out,groups]
    # then process_weights transposes (1,2) back to [groups,out]. Net = stored [groups,out].
    s = scales
    g = (torch.arange(IN) // GROUP).long()
    sg = s[g].float()
    w = (codes_op.float() - 8.0) * sg                   # sym zero=8
    return w, codes_op

def check(name, base, packed_axis_in=True):
    print(f"\n===== {name}: {base} =====")
    qweight = load(base + ".qweight")
    scales  = load(base + ".scales").float()
    qzeros  = load(base + ".qzeros")
    IN = qweight.shape[0]*8; OUT = qweight.shape[1]; groups = scales.shape[0]
    print(f"qweight {list(qweight.shape)} in={IN} out={OUT} groups={groups} "
          f"scales {list(scales.shape)} qzeros {list(qzeros.shape)}")
    ref_w, ref_codes, zcodes = indep_dequant(qweight, qzeros, scales)
    print(f"qzeros nibble unique (pre+1): {torch.unique(gptq_unpack_lsb(qzeros,1)).tolist()[:8]}")
    print(f"ref dequant W range [{ref_w.min():.4f},{ref_w.max():.4f}] mean {ref_w.mean():.5f}")
    # plugin path reproduction
    plug_w, plug_codes = plugin_repack_dequant(qweight, scales)
    code_mm = (plug_codes != ref_codes).float().mean().item()
    print(f"[plugin codes vs indep codes] mismatch = {code_mm:.4%}")
    # if zeros are all 7 (true 8) the sym=8 dequant should match indep
    w_mm = (plug_w - ref_w).abs().max().item()
    print(f"[plugin W(zero=8) vs indep W] max abs diff = {w_mm:.6f}")
    return code_mm, w_mm

# routed expert gate_proj (MoE path)
exp = None
for k in IDX:
    if ".mlp.experts.0.gate_proj.qweight" in k:
        exp = k[:-len(".qweight")]; break
check("ROUTED EXPERT gate_proj (MoE path)", exp)

# routed expert down_proj
dn = exp.replace("gate_proj","down_proj")
check("ROUTED EXPERT down_proj (MoE path)", dn)

# linear_attn quantized projection (LINEAR path via AutoRoundHPUConfig)
la = None
for k in IDX:
    if "layers.0.linear_attn.in_proj_qkv.qweight" in k:
        la = k[:-len(".qweight")]; break
if la:
    check("LINEAR_ATTN in_proj_qkv (linear path)", la)
print("\nDONE")
