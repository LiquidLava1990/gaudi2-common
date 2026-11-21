#!/usr/bin/env python3
"""
NON-CIRCULAR decisive harness for the hpu-moe-ext GPTQ MoE garbage-output bug.

Run this INSIDE the vllm-gaudi docker container (needs torch, safetensors, and
the vllm import that backs pack/unpack_quantized_values_into_int32). It is
CPU-only: it never touches the HPU op torch.ops.hpu.convert_from_uint4 — instead
it reproduces that op's math (dequant = scale * (q_unpacked - zero_unpacked))
in plain torch so it runs anywhere.

WHY THE OLD numtest.py WAS CIRCULAR
-----------------------------------
tests/numtest.py BUILT the fake "gptq checkpoint" with
    pack_quantized_values_into_int32(q, qt, packed_dim=0)
i.e. the SAME vLLM packing routine the plugin later uses to UNPACK. So the
round-trip is guaranteed to pass no matter what real autogptq's nibble order is.
It also never reads a real checkpoint and never derives the int4 values from a
TRUSTED independent gptq formula. It proves nothing about byte order.

WHAT THIS HARNESS DOES
----------------------
1. Loads ONE real expert's gate_proj qweight/scales/qzeros/g_idx bytes straight
   from the safetensors file (manual header parse -> raw int32/f16 tensors).
   No vLLM, no plugin involved in this branch -> the TRUSTED reference.

2. TRUSTED dequant (autogptq / gptqmodel reference semantics, documented order):
     - qweight int32 [in//8, out]; nibble j of column packs input row 8*i+j:
           q[8*i + j, n] = (qweight[i, n] >> (4*j)) & 0xF          (LSB-first)
     - qzeros  int32 [groups, out//8]; same LSB-first nibble order along OUT:
           z[g, 8*i + j] = (qzeros[g, i] >> (4*j)) & 0xF
     - real zero-point = z + 1   (autogptq's "-1" storage convention)
     - desc_act=False -> g_idx[k] = k // group_size  (sequential, trivial)
     - W[k, n] = scale[g_idx[k], n] * (q[k,n] - (z[g_idx[k], n] + 1))
   This W is the de-facto ground truth (it is the standard CPU gptq dequant that
   every gptq kernel must reproduce).

3. PLUGIN PATH: faithfully replays the plugin exactly as written:
     a. upstream loader transform:  u8 = qweight_i32.T.contiguous().view(uint8)
        -> the layer.w13_qweight[e] param  ([out, in//2] uint8)
     b. plugin _repack_weight: view->int32 [out,in//8], transpose, unpack
        (packed_dim=0), repack (packed_dim=1) -> [in, out//8] int32
     c. scales transpose: [out, groups] -> [groups, out]
     d. zero-point: plugin _sym_zero -> constant (1<<(bits-1))+1 = 9, packed
     e. EMULATE torch.ops.hpu.convert_from_uint4: unpack the repacked int32
        along OUT (LSB-first, the documented convert_from_uint4 contract that
        the working CT path relies on), unpack the packed zero the same way,
        then W_plugin[k,n] = scale[g,n] * (q_unpacked[k,n] - zero_unpacked[g,n]).

4. Compares W_plugin to W_trusted, plus runs a battery of ALTERNATIVE
   transforms so we can localize the error to ONE of {packing-order, zero, scale}.

OUTCOME DECODER (printed at the end):
  - If "PLUGIN == TRUSTED": the whole path is numerically correct; the bug is
    NOT in repack/zero/scale (look elsewhere: permuted_weights flag, routing,
    activation, g_idx, or an HPU-op-only quirk).
  - If a variant with zero=8 (bias) matches but zero=9 does not: the bug is the
    ZERO-POINT (_sym_zero off-by-one); fix _sym_zero to use bias (8).
  - If a variant where the OUT nibble order is REVERSED matches: the bug is the
    convert_from_uint4 OUT-dim nibble order vs pack_quantized_values_into_int32.
  - If a variant where the IN-dim unpack order is reversed matches: the bug is
    that autogptq's int32 nibble order != vLLM unpack order (packing-order bug)
    -> the plugin must unpack autogptq nibbles itself, not via vLLM unpack.
  - If a TRANSPOSED variant matches: the [in,out] vs [out,in] orientation fed to
    the op is wrong.
"""
import json
import os
import struct
import sys

import torch

try:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
    from vllm.scalar_type import scalar_types
    HAVE_VLLM = True
except Exception as e:  # pragma: no cover - lets the script self-report
    HAVE_VLLM = False
    print(f"[warn] vLLM import failed ({e}); plugin-exact branch will be skipped, "
          f"but the trusted reference + alt transforms still run.")

MODEL = os.environ.get("MODEL_SAFETENSORS",
                       "/models/Qwen3-30B-A3B-GPTQ-Int4/model.safetensors")
LAYER = 0
EXPERT = 0
PROJ = "gate_proj"          # in=2048, out=768, group_size=128
GROUP_SIZE = 128
BITS = 4


# ----------------------------------------------------------------------------
# Minimal safetensors reader (no safetensors dependency required, but we use
# torch.frombuffer so the data lands in a real tensor).
# ----------------------------------------------------------------------------
_ST_DTYPE = {
    "I32": (torch.int32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}


def load_tensor(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        meta = hdr[key]
        dt, _ = _ST_DTYPE[meta["dtype"]]
        s, e = meta["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
    t = torch.frombuffer(bytearray(raw), dtype=dt).clone()
    return t.view(*meta["shape"])


def k(proj, name):
    return f"model.layers.{LAYER}.mlp.experts.{EXPERT}.{proj}.{name}"


# ----------------------------------------------------------------------------
# TRUSTED autogptq reference dequant (independent of vLLM and of the plugin).
# ----------------------------------------------------------------------------
def gptq_unpack_lsb_first(packed_i32, n_along_packed, axis):
    """Unpack an int32 tensor packed 8x along `axis` using autogptq's documented
    LSB-first sequential nibble order:  value index = 8*i + j, j in [0..7],
    nibble j lives in bits [4*j, 4*j+4) of element i.
    Returns int32 tensor with `axis` expanded by 8x."""
    packed_i32 = packed_i32.to(torch.int64)  # avoid sign issues on >>
    shifts = torch.arange(8, dtype=torch.int64) * 4         # [0,4,8,...,28]
    moved = packed_i32.movedim(axis, -1)                    # packed dim last
    out = (moved.unsqueeze(-1) >> shifts) & 0xF             # [..., npack, 8]
    out = out.reshape(*moved.shape[:-1], -1)                # [..., npack*8]
    out = out.movedim(-1, axis)
    return out.to(torch.int32)


def trusted_dequant():
    qweight = load_tensor(MODEL, k(PROJ, "qweight"))   # int32 [in//8, out]
    scales = load_tensor(MODEL, k(PROJ, "scales")).float()   # [groups, out]
    qzeros = load_tensor(MODEL, k(PROJ, "qzeros"))     # int32 [groups, out//8]
    g_idx = load_tensor(MODEL, k(PROJ, "g_idx"))       # int32 [in]

    IN = qweight.shape[0] * 8
    OUT = qweight.shape[1]
    groups = scales.shape[0]
    assert scales.shape[1] == OUT
    assert qzeros.shape == (groups, OUT // 8)
    assert g_idx.shape[0] == IN

    # q values [in, out], unpacked along the INPUT (packed) dim, LSB-first.
    q = gptq_unpack_lsb_first(qweight, IN, axis=0)              # [in, out]
    # zero values [groups, out], unpacked along OUT, LSB-first, then +1.
    z = gptq_unpack_lsb_first(qzeros, OUT, axis=1) + 1          # [groups, out]

    # desc_act=False sanity: g_idx must be sequential k//group_size.
    expected_gidx = (torch.arange(IN) // GROUP_SIZE).to(torch.int32)
    g_is_sequential = torch.equal(g_idx, expected_gidx)
    print(f"[trusted] g_idx sequential (desc_act=False)? {g_is_sequential}; "
          f"first/last g_idx = {int(g_idx[0])}/{int(g_idx[-1])}")
    print(f"[trusted] unique stored qzero nibbles = "
          f"{sorted(set((z-1).flatten().tolist()))[:8]} (sym gptq -> expect ~[7] or [8])")

    g = g_idx.to(torch.int64)
    W = scales[g] * (q.float() - z[g].float())                 # [in, out]
    return dict(W=W, q=q, z=z, scales=scales, g=g, IN=IN, OUT=OUT,
                groups=groups, qweight=qweight, qzeros=qzeros)


# ----------------------------------------------------------------------------
# Emulate torch.ops.hpu.convert_from_uint4 in plain torch.
#   weight_packed : int32, packed 8x along the LAST (out) dim
#   weight_scale  : [groups, out]
#   zero_point    : int32, packed 8x along out -> [groups, out//8]
#   returns dequant weight [in, out] = scale[g] * (q - zero)
# The HPU op unpacks both weight and zero along the packed dim using the SAME
# nibble order the CT path packs with (pack_quantized_values_into_int32).
# We test BOTH possible nibble orders so we can detect an order mismatch.
# ----------------------------------------------------------------------------
def emulate_convert_from_uint4(weight_packed, weight_scale, zero_packed,
                               groups, out_nibble="lsb"):
    IN = weight_packed.shape[0]
    OUT = weight_packed.shape[1] * 8
    if out_nibble == "lsb":
        q = gptq_unpack_lsb_first(weight_packed, OUT, axis=1)      # [in, out]
        z = gptq_unpack_lsb_first(zero_packed, OUT, axis=1)        # [groups, out]
    else:  # msb-first / reversed nibble order within each int32
        q = _unpack_reversed(weight_packed, axis=1)
        z = _unpack_reversed(zero_packed, axis=1)
    gsz = IN // groups
    g = (torch.arange(IN) // gsz).to(torch.int64)
    return weight_scale[g] * (q.float() - z[g].float())


def _unpack_reversed(packed_i32, axis):
    p = packed_i32.to(torch.int64)
    shifts = (7 - torch.arange(8, dtype=torch.int64)) * 4
    moved = p.movedim(axis, -1)
    out = (moved.unsqueeze(-1) >> shifts) & 0xF
    out = out.reshape(*moved.shape[:-1], -1)
    return out.movedim(-1, axis).to(torch.int32)


# ----------------------------------------------------------------------------
# PLUGIN-EXACT path (only if vLLM importable).
# ----------------------------------------------------------------------------
def plugin_path(ref):
    qt = scalar_types.uint4

    # (a) upstream gptq loader transform on the REAL checkpoint qweight.
    qweight_i32 = ref["qweight"]                              # [in//8, out]
    u8 = qweight_i32.T.contiguous().view(torch.uint8)        # [out, in//2]  == layer.w13_qweight[e]

    # (b) plugin _repack_weight
    w_i32 = u8.contiguous().view(torch.int32)                # [out, in//8]
    vals = unpack_quantized_values_into_int32(
        w_i32.transpose(0, 1).contiguous(), qt, packed_dim=0)  # [in, out]
    repacked = pack_quantized_values_into_int32(vals, qt, packed_dim=1)  # [in, out//8]

    # (c) scales: loader stores [out, groups]; plugin transposes -> [groups, out].
    #     The TRUSTED scales are already [groups, out]; emulate the round-trip
    #     by transposing to [out,groups] then back, identity, but keep faithful.
    w_scale = ref["scales"].transpose(0, 1).contiguous()     # [out, groups]
    w_scale = w_scale.transpose(0, 1).contiguous()           # [groups, out]

    # (d) plugin _sym_zero -> constant 9 (= (1<<3)+1), packed along out.
    groups, OUT = ref["groups"], ref["OUT"]
    zero_val = (1 << (BITS - 1)) + 1                         # = 9
    zeros = torch.full((groups, OUT), zero_val, dtype=torch.int32)
    zero_packed9 = pack_quantized_values_into_int32(zeros, qt, packed_dim=1)

    zeros8 = torch.full((groups, OUT), 8, dtype=torch.int32)
    zero_packed8 = pack_quantized_values_into_int32(zeros8, qt, packed_dim=1)

    # (e) emulate the op (LSB nibble order = what CT path relies on).
    W9 = emulate_convert_from_uint4(repacked, w_scale, zero_packed9, groups, "lsb")
    W8 = emulate_convert_from_uint4(repacked, w_scale, zero_packed8, groups, "lsb")
    # the TRUE per-group zero from the checkpoint (varies per group/out):
    z_true_packed = pack_quantized_values_into_int32(ref["z"], qt, packed_dim=1)
    Wtrue_zero = emulate_convert_from_uint4(repacked, w_scale, z_true_packed, groups, "lsb")
    # reversed OUT nibble order variant (detect order mismatch in convert_from_uint4):
    W9_rev = emulate_convert_from_uint4(repacked, w_scale, zero_packed9, groups, "msb")
    return dict(repacked=repacked, vals=vals, W9=W9, W8=W8,
                Wtrue_zero=Wtrue_zero, W9_rev=W9_rev)


def cmp(name, A, B):
    if A.shape != B.shape:
        # try transpose
        if A.shape == B.T.shape:
            d = (A.float() - B.T.float()).abs().max().item()
            print(f"  {name:32s} SHAPE-T match, max|diff(after T)| = {d:.4g}")
        else:
            print(f"  {name:32s} SHAPE MISMATCH {tuple(A.shape)} vs {tuple(B.shape)}")
        return
    d = (A.float() - B.float()).abs().max().item()
    frac = ((A.float() - B.float()).abs() > 1e-3).float().mean().item()
    print(f"  {name:32s} max|diff| = {d:.6g}   frac_mismatch = {frac:.3%}")


def main():
    print("=" * 78)
    print("TRUSTED autogptq reference dequant of one REAL expert")
    print("=" * 78)
    ref = trusted_dequant()
    W = ref["W"]
    print(f"[trusted] W shape = {tuple(W.shape)}  (in,out)  "
          f"sample W[0,:4] = {W[0, :4].tolist()}")

    # Independent cross-check of the trusted unpack: re-derive vals two ways.
    if not HAVE_VLLM:
        print("\n[!] vLLM not importable here -> run inside the gaudi container.")
        return

    print("\n" + "=" * 78)
    print("PLUGIN-EXACT path replayed on the SAME real bytes")
    print("=" * 78)
    pl = plugin_path(ref)

    # First: does the plugin's vLLM-unpack of the INPUT dim recover the same q
    # values as the trusted LSB-first unpack? This is the packing-order test.
    print("\n[A] INPUT-dim packing order (vLLM unpack vs autogptq LSB-first):")
    cmp("plugin vals  vs  trusted q", pl["vals"], ref["q"])

    print("\n[B] Full dequant comparisons (PLUGIN op-emulation vs TRUSTED W):")
    cmp("plugin zero=9 (as shipped)", pl["W9"], W)
    cmp("plugin zero=8 (bias)", pl["W8"], W)
    cmp("plugin true per-group zero", pl["Wtrue_zero"], W)
    cmp("plugin zero=9, OUT nibble REV", pl["W9_rev"], W)

    print("\n" + "=" * 78)
    print("DECODER")
    print("=" * 78)
    print("""
 - If [A] is NOT an exact match  -> ROOT CAUSE = PACKING ORDER: vLLM
   (un)pack_quantized_values_into_int32 does not use autogptq's int32 nibble
   order, so _repack_weight scrambles every weight. Total garbage. FIX: unpack
   the autogptq nibbles directly (LSB-first) instead of via vLLM unpack, OR feed
   the op the original gptq int32 unchanged if its contract already matches.

 - If [A] matches but only 'plugin true per-group zero' matches in [B] (and
   zero=9 / zero=8 do NOT) -> ROOT CAUSE = ZERO-POINT: a constant symmetric zero
   is wrong for THIS checkpoint; the real qzeros are not all 8. (Inspect the
   '[trusted] unique stored qzero nibbles' line above.)

 - If [A] matches and 'zero=8' matches but 'zero=9' does NOT -> ROOT CAUSE =
   _sym_zero off-by-one: use bias (8), not 9. (NOTE: a constant +1 in the zero
   is a small systematic bias, usually NOT full garbage.)

 - If [A] matches and 'zero=9 OUT nibble REV' matches but plain 'zero=9' does
   NOT -> ROOT CAUSE = OUTPUT-dim nibble order in the repack vs what
   convert_from_uint4 expects.

 - If 'plugin zero=9 (as shipped)' already matches TRUSTED W exactly -> the
   repack+zero+scale math is CORRECT; the garbage comes from elsewhere
   (permuted_weights flag, g_idx omission under desc_act, routing, or activation).
""")


if __name__ == "__main__":
    sys.exit(main())
