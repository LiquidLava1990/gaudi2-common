#!/usr/bin/env python3
"""ON-CARD validation + A/B micro-benchmark of the OPTIMIZED dequant kernels.

Single-card, NO vLLM, NO docker, NO model. Cards must be FREE. The ORCHESTRATOR
runs this; the build agent never does.

It proves, on the device, that the new int4->bf16 dequant kernels
(moe_ext_unpack_dequant_w4_{lean,fast}_bf16) compute the SAME bf16 as the live
reference torch.ops.hpu.convert_from_uint4, and times them against it.

  LEAN  (variant 2) = U4-descriptor load, folded zero, hoisted scale. Its access
                      pattern is IDENTICAL to the proven live kernel -> this is the
                      PRIMARY byte-exact gate (must pass bit-for-bit).
  FAST  (variant 1) = native int32 unpack (kills the 8x read amplification). Its AP
                      is NEW (stride-8 deinterleave); if GC rejects it at instantiate
                      the script reports it LOUDLY and continues with LEAN as the
                      shippable result (per the kernel source's honesty note).

=== NO SILENT-PASS (the load-bearing design rule) ===
The new kernels are invoked through a REAL torch op
(torch.ops.hpu_moe_ext.unpack_dequant_w4_{lean,fast}) registered by the PT2 glue
build_fast/libmoe_ext_custom_op_fast.so. If that op is NOT loadable/callable this
script RAISES — it NEVER falls back to convert_from_uint4 (which would A/B the
reference against itself and falsely pass). A genuine distinct kernel, or a loud
failure. That is the whole point of the separate glue.

GATES (LOUD; non-zero exit on any fail):
  1. LEAN bf16 == convert_from_uint4 bf16 BYTE-EXACT (bits-identical) on GLM w13/w2.
     [PRIMARY — load-bearing correctness]
  2. relNorm(LEAN, ref) <= 1e-2 (the int4 path bar). [redundant w/ 1, kept as a
     softer cross-check + the number to print]
  3. FAST attempted: if it instantiates, same byte-exact + relNorm gates as LEAN;
     if GC rejects its AP, report LOUDLY and mark FAST=SKIPPED (NOT a hard fail —
     LEAN is the shippable variant; FAST is the upside).
  4. A self-check that the harness can DETECT a mismatch: a deliberately-corrupted
     packed weight MUST make the new-kernel output differ from the reference (proves
     gate 1 is not vacuously passing). [adversarial]
  5. micro-benchmark: time convert_from_uint4 vs LEAN (and FAST if live) over
     (experts x layers) of GLM w13+w2 with a real device sync; print ms/iter +
     speedup. (Timing is informational; correctness gates are 1-4.)

USAGE (on the box, cards free):
  export GC_KERNEL_PATH=$PWD/../tpc_kernel/build_fast/libmoe_ext_perflib_fast.so:$GC_KERNEL_PATH
  PT_HPU_LAZY_MODE=1 python tests/dequant_fast_oncard_validate.py --experts 8 --layers 75
(or just run tpc_kernel/oncard_dequant_ab.sh which sets all of this up.)

Exit 0 = all hard gates (1,2,4 + FAST if it instantiated) pass.
"""
import argparse
import os
import sys
import time

import torch
import habana_frameworks.torch.core as htorch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk)
from vllm.scalar_type import scalar_types

QT = scalar_types.uint4
DEV, DT = "hpu", torch.bfloat16
H, I, TWO_I, GS, ZERO = 6144, 2048, 4096, 128, 8

# Resolve the PT2 glue .so (relative to this file: ../tpc_kernel/build_fast/...).
_HERE = os.path.dirname(os.path.abspath(__file__))
_GLUE_SO = os.path.abspath(
    os.path.join(_HERE, "..", "tpc_kernel", "build_fast",
                 "libmoe_ext_custom_op_fast.so"))
_PERFLIB_SO = os.path.abspath(
    os.path.join(_HERE, "..", "tpc_kernel", "build_fast",
                 "libmoe_ext_perflib_fast.so"))


def _die(msg, code=2):
    print("\n!!! FATAL:", msg, file=sys.stderr, flush=True)
    sys.exit(code)


def load_new_ops():
    """Load the PT2 glue and return the two real ops. LOUD failure if anything is
    missing — NEVER returns a convert_from_uint4 stand-in."""
    if not os.path.exists(_PERFLIB_SO):
        _die(f"perf-lib not built: {_PERFLIB_SO}\n"
             f"  build it CPU-only:  bash tpc_kernel/build_fast.sh  (in the dev image)")
    # Ensure the GC can resolve the new GUIDs.
    cur = os.environ.get("GC_KERNEL_PATH", "")
    parts = cur.split(":") if cur else []
    if _PERFLIB_SO not in parts:
        os.environ["GC_KERNEL_PATH"] = ":".join([_PERFLIB_SO] + parts)
        print(f"  [setup] prepended perf-lib to GC_KERNEL_PATH", flush=True)
    if not os.path.exists(_GLUE_SO):
        _die(f"PT2 glue not built: {_GLUE_SO}\n"
             f"  build it in the dev image:  bash tpc_kernel/build_fast.sh\n"
             f"  (Step 4 needs torch importable -> run inside pytorch-installer.)")
    try:
        torch.ops.load_library(_GLUE_SO)
    except Exception as e:  # noqa: BLE001
        _die(f"torch.ops.load_library({_GLUE_SO}) failed: {e}")
    op = getattr(torch.ops, "hpu_moe_ext", None)
    if op is None or not hasattr(op, "unpack_dequant_w4_lean") \
            or not hasattr(op, "unpack_dequant_w4_fast"):
        _die("glue loaded but hpu_moe_ext.unpack_dequant_w4_{lean,fast} not "
             "registered — the new ops are NOT callable. Refusing to fall back to "
             "convert_from_uint4 (that would be a silent self-A/B).")
    print(f"  [setup] loaded new ops: hpu_moe_ext.unpack_dequant_w4_lean / _fast",
          flush=True)
    return op.unpack_dequant_w4_lean, op.unpack_dequant_w4_fast


def mk_expert(out_, in_, seed):
    torch.manual_seed(seed)
    codes = torch.randint(0, 16, (in_, out_), dtype=torch.int32)
    packed = pk(codes, QT, packed_dim=1)                    # [in, out//8] int32
    scale = (torch.rand(in_ // GS, out_, dtype=DT) * 0.02 + 0.01)
    return packed.to(DEV), scale.to(DEV)


def sym_zero(groups, out_):
    return pk(torch.full((groups, out_), ZERO, dtype=torch.int32), QT,
              packed_dim=1).to(DEV)


def ref_convert(packed, scale, zp):
    return torch.ops.hpu.convert_from_uint4(packed, scale, zp, torch.bfloat16, None)


def relnorm(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return (torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-30)).item()


def try_new(op, packed, scale, label):
    """Call a new op; return (tensor or None, err). LOUD on the actual GC error so a
    rejected access pattern is visible (not swallowed)."""
    try:
        out = op(packed, scale, GS, ZERO)
        htorch.mark_step()
        _ = out.detach().to("cpu")     # force realize -> surfaces GC instantiate err
        return out, None
    except Exception as e:  # noqa: BLE001
        print(f"     [{label}] new op raised on instantiate/exec: "
              f"{type(e).__name__}: {str(e)[:300]}", flush=True)
        return None, e


def correctness(op_lean, op_fast):
    """Gates 1-4. Returns (lean_ok, fast_status) where fast_status in
    {'PASS','FAIL','SKIPPED'}."""
    print("\n[correctness] LEAN (primary) + FAST (secondary) vs convert_from_uint4",
          flush=True)
    lean_ok = True
    fast_status = "PASS"   # downgraded to SKIPPED if GC rejects, FAIL if wrong bits

    for name, out_, in_ in [("w13", TWO_I, H), ("w2", H, I)]:
        packed, scale = mk_expert(out_, in_, seed=(hash(name) & 0xFFFF))
        zp = sym_zero(in_ // GS, out_)
        ref = ref_convert(packed, scale, zp)
        htorch.mark_step()
        _ = ref.detach().to("cpu")

        # ---- LEAN (must be byte-exact) ----
        lean, _ = try_new(op_lean, packed, scale, f"{name}/lean")
        if lean is None:
            print(f"  [{name}] !!! LEAN failed to run — this IS a hard fail "
                  f"(its AP is the proven live pattern).")
            lean_ok = False
        else:
            rn = relnorm(lean, ref)
            bits_eq = torch.equal(lean.view(torch.int16), ref.view(torch.int16))
            n_diff = int((lean.view(torch.int16) != ref.view(torch.int16))
                         .sum().item())
            print(f"  [{name}] LEAN relNorm={rn:.3e}  bf16-bits-identical={bits_eq}"
                  f"  ({n_diff} bits differ)")
            if not bits_eq:
                print(f"     !!! LEAN NOT byte-exact ({n_diff} differ) — HARD FAIL")
                lean_ok = False
            if rn > 1e-2:
                print(f"     !!! LEAN relNorm {rn:.3e} > 1e-2 int4 bar — HARD FAIL")
                lean_ok = False

        # ---- FAST (byte-exact if it instantiates; LOUD-skip if GC rejects AP) ----
        if fast_status != "SKIPPED":
            fast, ferr = try_new(op_fast, packed, scale, f"{name}/fast")
            if fast is None:
                print(f"  [{name}] FAST did not instantiate (GC rejected the "
                      f"native-unpack AP) -> FAST=SKIPPED; LEAN is the shippable "
                      f"variant. (See the kernel source's stride-8 AP note.)")
                fast_status = "SKIPPED"
            else:
                rn = relnorm(fast, ref)
                bits_eq = torch.equal(fast.view(torch.int16), ref.view(torch.int16))
                n_diff = int((fast.view(torch.int16) != ref.view(torch.int16))
                             .sum().item())
                print(f"  [{name}] FAST relNorm={rn:.3e}  "
                      f"bf16-bits-identical={bits_eq}  ({n_diff} bits differ)")
                if not bits_eq or rn > 1e-2:
                    print(f"     !!! FAST wrong on-card ({n_diff} bits differ, "
                          f"relNorm {rn:.3e}) — the stride-8 deinterleave AP needs "
                          f"adjustment. HARD FAIL for FAST (LEAN still shippable).")
                    fast_status = "FAIL"

    # ---- gate 4: adversarial — a corrupted weight MUST change the LEAN output ----
    print("\n[adversarial] corrupt one packed word -> LEAN output MUST differ from "
          "the reference (proves the byte-exact gate isn't vacuous)", flush=True)
    packed, scale = mk_expert(256, 512, seed=7)
    zp = sym_zero(512 // GS, 256)
    ref = ref_convert(packed, scale, zp)
    htorch.mark_step()
    clean, _ = try_new(op_lean, packed, scale, "adv/clean")
    bad = packed.clone()
    bad[0, 0] = bad[0, 0] ^ (0xF << 12)        # flip nibble 3 of word (0,0)
    corrupt, _ = try_new(op_lean, bad, scale, "adv/corrupt")
    adv_ok = False
    if clean is not None and corrupt is not None:
        ref_bits = ref.view(torch.int16)
        clean_eq = torch.equal(clean.view(torch.int16), ref_bits)
        corrupt_diff = not torch.equal(corrupt.view(torch.int16), ref_bits)
        print(f"  clean==ref: {clean_eq} ; corrupt!=ref: {corrupt_diff}")
        adv_ok = clean_eq and corrupt_diff
        if not adv_ok:
            print("     !!! adversarial check FAILED — the gate may be vacuous "
                  "(clean must match ref AND corrupt must differ). HARD FAIL.")
    else:
        print("     !!! adversarial check could not run (LEAN op unavailable).")
    return lean_ok, fast_status, adv_ok


def bench(fn, iters, warmup, label):
    for _ in range(warmup):
        o = fn()
        htorch.mark_step()
    _ = o.detach().to("cpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn()
        htorch.mark_step()
    _ = o.detach().to("cpu")
    dt = (time.perf_counter() - t0) / iters
    print(f"  {label:46s} {dt * 1e3:9.3f} ms/iter", flush=True)
    return dt


def microbench(op_lean, op_fast, fast_status, experts, layers, iters, warmup):
    print(f"\n[microbench] {experts} experts x {layers} layers of GLM w13+w2",
          flush=True)
    ex = [(mk_expert(TWO_I, H, e), mk_expert(H, I, e + 1000))
          for e in range(experts)]
    z13 = sym_zero(H // GS, TWO_I)
    z2 = sym_zero(I // GS, H)

    def body(kernel, use_zp):
        acc = torch.zeros((), dtype=torch.float32, device=DEV)
        for _ in range(layers):
            for (p13, s13), (p2, s2) in ex:
                if use_zp:
                    W13 = kernel(p13, s13, z13)
                    W2 = kernel(p2, s2, z2)
                else:
                    W13 = kernel(p13, s13)
                    W2 = kernel(p2, s2)
                acc = acc + W13[0, 0].float() + W2[0, 0].float()
        return acc

    t_ref = bench(lambda: body(lambda p, s, z: ref_convert(p, s, z), True),
                  iters, warmup, "convert_from_uint4 (reference)")
    t_lean = bench(lambda: body(lambda p, s: op_lean(p, s, GS, ZERO), False),
                   iters, warmup, "new kernel (LEAN)")
    print(f"\n  LEAN speedup = {t_ref / t_lean:.3f}x  "
          f"(LEAN {t_lean * 1e3:.2f} ms vs ref {t_ref * 1e3:.2f} ms)")
    if fast_status == "PASS":
        t_fast = bench(lambda: body(lambda p, s: op_fast(p, s, GS, ZERO), False),
                       iters, warmup, "new kernel (FAST)")
        print(f"  FAST speedup = {t_ref / t_fast:.3f}x  "
              f"(FAST {t_fast * 1e3:.2f} ms vs ref {t_ref * 1e3:.2f} ms)")
    else:
        print(f"  FAST timing skipped (status={fast_status})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-bench", action="store_true",
                    help="correctness gates only (skip timing)")
    args = ap.parse_args()

    print("=" * 74)
    print(f"ON-CARD dequant A/B | GLM H={H} I={I} GS={GS} | lazy={htorch.is_lazy()}")
    print(f"GC_KERNEL_PATH={os.environ.get('GC_KERNEL_PATH', '(unset!)')}")
    print("=" * 74)

    op_lean, op_fast = load_new_ops()
    lean_ok, fast_status, adv_ok = correctness(op_lean, op_fast)

    if not args.no_bench:
        microbench(op_lean, op_fast, fast_status, args.experts, args.layers,
                   args.iters, args.warmup)

    print("=" * 74)
    print(f"  LEAN byte-exact   : {'PASS' if lean_ok else 'FAIL'}")
    print(f"  FAST native-unpack: {fast_status}")
    print(f"  adversarial gate  : {'PASS' if adv_ok else 'FAIL'}")
    # Hard pass = LEAN byte-exact AND adversarial AND FAST not actively wrong
    # (FAST=SKIPPED is acceptable; FAST=FAIL is not).
    hard_ok = lean_ok and adv_ok and (fast_status in ("PASS", "SKIPPED"))
    print("RESULT:", "PASS" if hard_ok else "FAIL (see !!! lines)")
    print("=" * 74)
    return 0 if hard_ok else 1


if __name__ == "__main__":
    sys.exit(main())
