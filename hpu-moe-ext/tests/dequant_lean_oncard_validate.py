#!/usr/bin/env python3
"""ON-CARD correctness gate for the CORRECTED LEAN dequant (U4-descriptor + contiguous
store), fed via reinterpret_cast->uint4 the way the kernel requires (the OpBackend op
dequant_w4_lean_sa = reinterpret_cast + moe_ext_unpack_dequant_w4_lean_bf16).

This REPLACES the broken UserCustomOp standalone (which handed the u4-descriptor kernel
a raw int32 tensor -> the stride-8 store landed garbage, relNorm ~2.5). A PASS here
proves the SAME kernel+edge the #3 fold uses.

Run (box, cards free):
  export GC_KERNEL_PATH=$PWD/tpc_kernel/build_fast/libmoe_ext_perflib_fast.so:$GC_KERNEL_PATH
  PT_HPU_LAZY_MODE=1 MOE_LEAN_SA_SO=$PWD/tpc_kernel/opbackend/build/libdequant_lean_sa_lazy.so \
    python tests/dequant_lean_oncard_validate.py
"""
import os, sys, time

# The LEAN guid is only registered if libmoe_ext_perflib_fast.so is on GC_KERNEL_PATH
# (the GC's KERNEL_DB scans it). Set it HERE, before importing torch/habana below, so
# it is in place whenever the GC inits -- otherwise synNodeCreate reports the guid
# "doesn't exist". GC_KERNEL_PATH is read live at synInitialize, so a pre-import set
# is the robust placement.
_PERFLIB = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "tpc_kernel", "build_fast",
    "libmoe_ext_perflib_fast.so"))
if not os.path.exists(_PERFLIB):
    sys.exit(f"!!! perf-lib missing: {_PERFLIB}  (build: bash tpc_kernel/build_fast.sh)")
_cur = os.environ.get("GC_KERNEL_PATH", "")
_parts = _cur.split(":") if _cur else []
if _PERFLIB not in _parts:
    os.environ["GC_KERNEL_PATH"] = ":".join([_PERFLIB] + _parts)
print(f"[setup] GC_KERNEL_PATH={os.environ['GC_KERNEL_PATH']}", flush=True)

import torch
import habana_frameworks.torch.core as htorch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk)
from vllm.scalar_type import scalar_types

QT = scalar_types.uint4
DEV, DT = "hpu", torch.bfloat16
H, I, TWO_I, GS, ZERO = 6144, 2048, 4096, 128, 8

_SO = os.environ.get("MOE_LEAN_SA_SO", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "tpc_kernel", "opbackend", "build",
    "libdequant_lean_sa_lazy.so")))

def die(m): print("\n!!! FATAL:", m, file=sys.stderr, flush=True); sys.exit(2)

def load_op():
    if not os.path.exists(_SO): die(f"standalone LEAN .so not built: {_SO}")
    torch.ops.load_library(_SO)
    op = getattr(torch.ops, "hpu_moe_ext", None)
    if op is None or not hasattr(op, "dequant_w4_lean_sa"):
        die("hpu_moe_ext.dequant_w4_lean_sa not registered")
    print("  [setup] loaded op hpu_moe_ext.dequant_w4_lean_sa", flush=True)
    return op.dequant_w4_lean_sa

def mk(out_, in_, seed):
    torch.manual_seed(seed)
    codes = torch.randint(0, 16, (in_, out_), dtype=torch.int32)
    packed = pk(codes, QT, packed_dim=1)                 # [in, out//8] int32
    scale = (torch.rand(in_ // GS, out_, dtype=DT) * 0.02 + 0.01)
    return packed.to(DEV), scale.to(DEV)

def symzp(groups, out_):
    return pk(torch.full((groups, out_), ZERO, dtype=torch.int32), QT, packed_dim=1).to(DEV)

def ref(packed, scale, zp):
    return torch.ops.hpu.convert_from_uint4(packed, scale, zp, torch.bfloat16, None)

def relnorm(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30)).item()

def value_dump(op):
    """LOOP-BREAKER: a TINY fully-deterministic case. Dump the DEVICE output, the
    reference, and AUTO-DIAGNOSE the divergence pattern (transpose? nibble-permute?
    scale/zero off?). The CPU numpy model has mispredicted the device 3x, so this is
    ground truth -- the orchestrator runs it on-card and reads the pattern.

    Case: IN=4, OUT=16 (so packed is [4, 2] int32 = 2 words/row), ONE group (GS>=IN).
    codes[i,o] = (o + 1) deterministic (1..16 along OUT, same every row) so every
    output element is hand-predictable: out[i,o] = (codes[i,o]-8)*scale[0,o].
    scale[0,o] = 1.0 for all o (bf16-exact) so out[i,o] = codes[i,o]-8 EXACTLY.
    Then ref[i,o] should be the column ramp {-7,-6,...,+8} on every row.
    """
    # TRUSTWORTHY permute probe: OUT=512 = EXACTLY one kernel step (SW_UNPACK 256-byte window
    # -> v_convert_u4_to_u8_all_b -> 512 codes). Encode the column in the SCALE: codes ALL = 9
    # (code-8 = +1), scale[o] = o -> out[i,o] = o = the SOURCE col index. bf16 is integer-exact
    # to 256, so cols 0..255 give an UNAMBIGUOUS map (the deinterleave is periodic). IN=2 so
    # row1!=row0 catches the IN-iteration bug (row1 all-zero == IN not stored).
    IN_, OUT_ = 2, 512
    print(f"\n[value-dump] TRUSTWORTHY permute probe: IN={IN_} OUT={OUT_} (= one kernel 512-code "
          f"step). codes=9 (code-8=+1), scale[o]=o -> out[i,o]=o; cols 0..255 bf16-exact -> "
          f"DEVICE[d]=TRUE source col landed at device col d (unambiguous).", flush=True)
    # TWO independent probes in ONE run:
    #  CODE probe : codes[i,o] = (o % 15) + 1  (unique within each 15-col block), scale = 1.0.
    #               -> out[i,o] = code-8. Tests the NIBBLE-DEINTERLEAVE / code path. Decode the
    #               permute from cols 0..14 (unique values -7..+7) -> device col d's code reveals
    #               the SOURCE col it came from (mod 15). This is what catches the lane scramble.
    #  SCALE probe: codes=9 (code-8=+1), scale[o]=o -> out[i,o]=o. Tests the SCALE/store-index
    #               path (each device col should hold scale at its OWN col).
    oo1 = torch.arange(OUT_)
    codes = ((oo1 % 15) + 1).to(torch.int32).unsqueeze(0).repeat(IN_, 1)   # CODE probe
    packed = pk(codes, QT, packed_dim=1)
    scale = torch.ones(1, OUT_, dtype=DT)                      # scale = 1.0
    zp = symzp(1, OUT_)
    packed_d, scale_d = packed.to(DEV), scale.to(DEV)
    r = ref(packed_d, scale_d, zp); htorch.mark_step(); r = r.detach().to("cpu").float()
    try:
        o = op(packed_d, scale_d, GS, ZERO); htorch.mark_step(); o = o.detach().to("cpu").float()
    except Exception as e:
        print(f"  op raised: {type(e).__name__}: {str(e)[:200]}"); return
    exp = (codes.float() - ZERO)                              # = [(o%15)+1-8 per col]

    def slc(t):
        if t.dim() != 2: return f"(non-2D {tuple(t.shape)}): first32={t.flatten()[:32].tolist()}"
        rows = []
        for r_ in t:
            head = " ".join(f"{v:5.0f}" for v in r_[:24].tolist())
            tail = " ".join(f"{v:5.0f}" for v in r_[-8:].tolist())
            rows.append(f"[{head} ... {tail}]")
        return "\n      " + "\n      ".join(rows)
    print(f"  shapes: device={tuple(o.shape)} ref={tuple(r.shape)} expected={tuple(exp.shape)}")
    print(f"  EXPECTED row (= col index 0,1,2,...) first24..last8={slc(exp)}")
    print(f"  REF      (convert_from_uint4)        first24..last8={slc(r)}")
    print(f"  DEVICE   out                         first24..last8={slc(o)}")
    # ---- DECODE the deinterleave from the CODE probe: DEVICE[d] = code_at_source - 8, so
    #      code = round(DEVICE[d]) + 8, and the SOURCE col (mod 15) = code - 1. Within cols
    #      0..14 the source is unique -> exact map; the pattern is periodic mod 15. ----
    if o.dim() == 2:
        row0 = o[0].tolist()
        dev_srcmod = [(int(round(v)) + ZERO - 1) for v in row0]   # source col MOD 15 (-1 if blank/0)
        print("  --- DEINTERLEAVE: device col d -> source col MOD 15 (=round(DEVICE[d])+8-1), first 64 ---")
        for base in range(0, 64, 32):
            print("    " + " ".join(f"{base+j}:{dev_srcmod[base+j]}" for j in range(32)))
        # how many device cols are 0/blank (the data-loss signature)?
        nblank0 = sum(1 for v in row0 if abs(v) < 0.5)
        nblank1 = sum(1 for v in o[1].tolist() if abs(v) < 0.5) if o.shape[0] > 1 else -1
        print(f"  --- blanks: row0 has {nblank0}/{OUT_} zeros ; row1 has {nblank1}/{OUT_} zeros ---")
        print(f"  --- row0[:16] codes = {[int(round(v))+ZERO for v in row0[:16]]} (expect 1..16 if identity) ---")
        if o.shape[0] > 1:
            print(f"  --- row1[:16]       = {[round(v,1) for v in o[1][:16].tolist()]} (expect == row0 if IN ok) ---")
    # ---- AUTO-DIAGNOSE ----
    def eq(a, b):
        try: return a.shape == b.shape and bool(torch.allclose(a.float(), b.float(), atol=0.51))
        except Exception: return False
    print("  --- divergence diagnosis (device vs candidate layouts) ---")
    print(f"    device == ref            (CORRECT)        : {eq(o, r)}")
    print(f"    device == ref.T          (TRANSPOSE)      : {eq(o, r.t()) if r.dim()==2 else 'n/a'}")
    print(f"    device == expected                        : {eq(o, exp)}")
    print(f"    device == expected.T     (TRANSPOSE)      : {eq(o, exp.t()) if exp.dim()==2 else 'n/a'}")
    if o.dim()==2 and o.numel()==exp.numel():
        of, ef = o.flatten(), exp.flatten()
        # is device a flat-reshape (row-major) of expected reshaped to device.shape?
        print(f"    device == expected.reshape(device.shape) : {eq(o, exp.flatten().reshape(o.shape))}")
        print(f"    device sorted == expected sorted (PERMUTE only): "
              f"{eq(torch.sort(of).values, torch.sort(ef).values)}")
    print(f"  relNorm(device,ref)={relnorm(o, r):.3e}")
    print("[value-dump] END  --- paste this whole block back ---\n", flush=True)


def main():
    op = load_op()
    if os.environ.get("VALUE_DUMP", "1") == "1":
        value_dump(op)
    ok = True
    for name, out_, in_ in [("w13", TWO_I, H), ("w2", H, I)]:
        packed, scale = mk(out_, in_, seed=(hash(name) & 0xFFFF))
        zp = symzp(in_ // GS, out_)
        r = ref(packed, scale, zp); htorch.mark_step(); _ = r.detach().to("cpu")
        try:
            o = op(packed, scale, GS, ZERO); htorch.mark_step(); o = o.detach().to("cpu")
        except Exception as e:
            print(f"  [{name}] LEAN raised: {type(e).__name__}: {str(e)[:300]}"); ok = False; continue
        rn = relnorm(o, r.detach().to("cpu"))
        be = torch.equal(o.view(torch.int16), r.detach().to("cpu").view(torch.int16))
        nd = int((o.view(torch.int16) != r.detach().to("cpu").view(torch.int16)).sum().item())
        print(f"  [{name}] LEAN relNorm={rn:.3e}  bf16-bits-identical={be}  ({nd} bits differ)")
        if rn > 1e-2: print(f"     !!! relNorm {rn:.3e} > 1e-2 -- FAIL"); ok = False
    # adversarial
    packed, scale = mk(256, 512, seed=7); zp = symzp(512 // GS, 256)
    r = ref(packed, scale, zp); htorch.mark_step(); rb = r.detach().to("cpu").view(torch.int16)
    clean = op(packed, scale, GS, ZERO); htorch.mark_step(); cb = clean.detach().to("cpu").view(torch.int16)
    bad = packed.clone(); bad[0, 0] = bad[0, 0] ^ (0xF << 12)
    corr = op(bad, scale, GS, ZERO); htorch.mark_step(); xb = corr.detach().to("cpu").view(torch.int16)
    clean_eq = torch.equal(cb, rb); corrupt_diff = not torch.equal(xb, rb)
    print(f"  [adversarial] clean==ref: {clean_eq} ; corrupt!=ref: {corrupt_diff}")
    if not (clean_eq and corrupt_diff): ok = False
    print("=" * 60); print("RESULT:", "PASS" if ok else "FAIL"); print("=" * 60)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
