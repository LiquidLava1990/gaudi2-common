#!/usr/bin/env python3
"""LAUNCH-vs-COMPUTE probe: is the int4 dequant cost per-op LAUNCH overhead or genuine
TPC compute? Decides between lever (1) fewer/bigger kernels and lever (2) faster unpack.

Compares, at GLM dims, the SAME total dequant work done as:
  (P) PER-EXPERT 2D dequant : N separate convert_from_uint4 calls (the prod shape) x layers.
  (B) BATCHED 3D dequant    : ONE convert_from_uint4 on a stacked [N, in, out] tensor x layers
                              (1 call instead of N -> N x fewer launches, same elements unpacked).
  (Bf) FULL BATCH           : ONE call on [N*layers, ...] -> 1 launch for ALL the work.

If (B)/(Bf) << (P): LAUNCH-bound -> lever = fuse/batch the kernels.
If (B)/(Bf) ~= (P): COMPUTE-bound -> the TPC unpack itself is the floor -> lever = faster
                    unpack kernel or fp8-resident (skip unpack).

Also times a same-#elements bf16->bf16 copy (no unpack) as the "unpack-free" control:
  (C) the elementwise streaming floor for that many bf16 elements (no int4 unpack math).
(B)-(C) isolates the int4-unpack ARITHMETIC from the elementwise/streaming cost.

RUN: PT_HPU_LAZY_MODE=1 python tests/launch_vs_compute_probe.py --experts 8 --layers 75
"""
import argparse, os, sys, time
import torch
import habana_frameworks.torch.core as htorch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk)
from vllm.scalar_type import scalar_types

QT = scalar_types.uint4
DEV = "hpu"; DT = torch.bfloat16
H, I, TWO_I, GS, ZERO = 6144, 2048, 4096, 128, 8


def mk(out_, in_, n):
    """n experts' packed int4 [n,in,out//8] + scale [n,in//GS,out] + shared zp."""
    codes = torch.randint(0, 16, (n, in_, out_), dtype=torch.int32)
    packed = torch.stack([pk(codes[i], QT, packed_dim=1) for i in range(n)], 0)  # [n,in,out//8]
    scale = (torch.rand(n, in_ // GS, out_, dtype=DT) * 0.02 + 0.01)
    zp = pk(torch.full((in_ // GS, out_), ZERO, dtype=torch.int32), QT, packed_dim=1)
    return packed.to(DEV), scale.to(DEV), zp.to(DEV)


def deq2d(p, s, z):
    return torch.ops.hpu.convert_from_uint4(p, s, z, torch.bfloat16, None)


def bench(fn, iters, warmup, label):
    for _ in range(warmup):
        o = fn(); htorch.mark_step()
    _ = o.detach().to("cpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn(); htorch.mark_step()
    _ = o.detach().to("cpu")
    dt = (time.perf_counter() - t0) / iters
    print(f"  {label:46s} {dt*1e3:9.3f} ms/iter", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    N, L = args.experts, args.layers

    # per-expert weights for w13 [H,2I] (the bigger of the two; representative)
    p, s, z = mk(TWO_I, H, N)          # [N, H, 2I//8]
    p_list = [p[i].contiguous() for i in range(N)]
    s_list = [s[i].contiguous() for i in range(N)]
    # full-batch stacked over N*L (replicate the per-token, per-layer reads)
    htorch.mark_step()

    nelem = N * H * TWO_I
    print("=" * 78, flush=True)
    print(f"LAUNCH-vs-COMPUTE | w13 dequant [H={H},2I={TWO_I}] | experts={N} layers={L} "
          f"lazy={htorch.is_lazy()}", flush=True)
    print(f"  elements unpacked/iter = {nelem*L/1e9:.3f} G int4 codes -> bf16", flush=True)
    print("=" * 78, flush=True)

    # (P) per-expert 2D: N calls x L layers
    def fP():
        acc = torch.zeros((), dtype=torch.float32, device=DEV)
        for _ in range(L):
            for i in range(N):
                acc = acc + deq2d(p_list[i], s_list[i], z).float().sum()
        return acc

    # batched needs a zp that matches the 3D batched shape: replicate the shared zp to [N,...]
    z3d = z.unsqueeze(0).expand(N, -1, -1).contiguous()   # [N, H//GS, 2I//8]
    htorch.mark_step()
    # (B) batched 3D: 1 call (on [N,...]) x L layers  -> N x fewer launches
    def fB():
        acc = torch.zeros((), dtype=torch.float32, device=DEV)
        for _ in range(L):
            acc = acc + deq2d(p, s, z3d).float().sum()
        return acc

    # (C) unpack-free control: a bf16 elementwise (x2) over the SAME #elements x L
    ctrl = torch.randn(N, H, TWO_I, dtype=DT, device=DEV)
    htorch.mark_step()
    def fC():
        acc = torch.zeros((), dtype=torch.float32, device=DEV)
        for _ in range(L):
            acc = acc + (ctrl * 2.0).float().sum()
        return acc

    tP = bench(fP, args.iters, args.warmup, "(P) per-expert 2D dequant (N calls x L)")
    tB = bench(fB, args.iters, args.warmup, "(B) batched 3D dequant (1 call x L)")
    tC = bench(fC, args.iters, args.warmup, "(C) unpack-free bf16 elementwise control")

    print("\n" + "=" * 78, flush=True)
    print("VERDICT", flush=True)
    print("=" * 78, flush=True)
    print(f"  (P) per-expert (N={N} calls/layer) = {tP*1e3:.2f} ms", flush=True)
    print(f"  (B) batched (1 call/layer)        = {tB*1e3:.2f} ms   "
          f"-> batching is {tP/tB:.2f}x faster (kills {N-1} launches/layer)", flush=True)
    print(f"  (C) unpack-free control           = {tC*1e3:.2f} ms   "
          f"(same #elems, no int4 unpack math)", flush=True)
    print("", flush=True)
    print(f"  batched dequant (B) vs control (C): {tB/tC:.2f}x  -> the int4-UNPACK ARITHMETIC "
          f"adds {(tB-tC)*1e3:+.2f} ms over a plain bf16 elementwise", flush=True)
    print("", flush=True)
    if tP > 1.5 * tB:
        print("  >>> Per-op LAUNCH overhead is significant: batching the dequant cuts it a lot.", flush=True)
        print("      Lever (1) fewer/bigger kernels HELPS the dequant in isolation.", flush=True)
    else:
        print("  >>> Launch overhead is NOT the dequant's main cost (batched ~= per-expert).", flush=True)
    if tB > 1.5 * tC:
        print("  >>> The int4-UNPACK ARITHMETIC dominates over plain streaming -> a faster", flush=True)
        print("      unpack kernel / fp8-resident (no unpack) is the real compute lever.", flush=True)
    else:
        print("  >>> The unpack arithmetic is ~free vs streaming -> the cost is elementwise/BW,", flush=True)
        print("      not the int4 bit-unpack -> fp8-resident saves little on this op.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
