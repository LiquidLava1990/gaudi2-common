#!/usr/bin/env python3
"""Standalone HPU microbench for the graph-compiler-fusion routed-MoE spike.

RUN THIS LATER ON THE HPU (inside the lazy container), AFTER EACH FIX. It is fully
self-contained: synthetic GLM-5.1-shaped int4 expert weights (CORRECT H=6144,
I=2048, E=32, top_k=8, group_size=128, GPTQ-sym zero=8), packed/scaled exactly the
way moe_wna16_hpu loads them, so the packing & layout are bit-faithful to production.

It reports three things:
  (a) CORRECTNESS — maxdiff(fused, intrinsic-reference) < 1e-2, else LOUD SystemExit.
  (b) LATENCY     — fused vs intrinsic over N bs=1 iters: tok/s + speedup, plus an
                    explicit `[diagnosis]` line if fused >= intrinsic (means the graph
                    compiler did NOT fuse convert_from_uint4 with the matmul -> the
                    spike captured no win; the HBM round-trip is still there).
  (c) HBM proxy   — if an HBM-bandwidth / device-memory counter is reachable, sample
                    it across both legs so a fused win shows up as less HBM traffic.

NOTE: This is a single-process, single-card microbench (no vLLM, no model load). It
exercises ONLY the routed-MoE math, which is the batch=1 bottleneck. It does NOT touch
the live server. It does NOT call docker. It allocates ~K experts of weights, tiny.

Usage:
  python tests/fused_routed_microbench.py [--iters 200] [--K 8] [--tokens 1] [--warmup 30]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import habana_frameworks.torch.core as htorch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk,
    unpack_quantized_values_into_int32 as upk)
from vllm.scalar_type import scalar_types

# Import the spike + its reference from the package under test.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpu_moe_ext.fused_routed_moe import fused_routed_moe, reference_routed_moe

QT = scalar_types.uint4
DEV = "hpu"

# === GLM-5.1-int4 CORRECT shapes ===
H = 6144            # hidden
I = 2048            # moe_intermediate
E = 32              # local experts on this rank
TOPK = 8            # top_k
GS = 128            # group_size
ZERO = 8            # GPTQ-sym shared zero-point (uint4 bias)
DT = torch.bfloat16


def _mk_expert(out_, in_):
    """One expert's quantized weight in the LOADED layout:
      packed int32 [in, out//8]   (op layout after repack)
      scale (op layout) [in//GS, out] bf16
    Built from random int4 codes via the SAME pack helper production uses, so the
    convert_from_uint4 round-trip is bit-faithful. zero is the shared constant 8.
    """
    codes = torch.randint(0, 16, (in_, out_), dtype=torch.int32)         # [in, out]
    packed = pk(codes, QT, packed_dim=1)                                 # [in, out//8] int32
    scale = (torch.rand(in_ // GS, out_, dtype=DT) * 0.02 + 0.01)        # [groups, out]
    return codes, packed, scale


def _sym_zero(groups, out):
    return pk(torch.full((groups, out), ZERO, dtype=torch.int32), QT, packed_dim=1)  # [groups, out//8]


def build_synthetic():
    """Stacked op-ready tensors for E experts, on HPU. Returns the same handles the
    production layer stashes (w13_packed/w2_packed/scales/zp + routing meta)."""
    torch.manual_seed(0)
    w13p_list, w2p_list, s13_list, s2_list = [], [], [], []
    for _ in range(E):
        _, w13p, s13 = _mk_expert(2 * I, H)     # w13: in=H, out=2I -> packed [H, 2I//8], scale [H//GS, 2I]
        _, w2p, s2 = _mk_expert(H, I)           # w2 : in=I, out=H  -> packed [I, H//8],  scale [I//GS, H]
        w13p_list.append(w13p)
        w2p_list.append(w2p)
        s13_list.append(s13)
        s2_list.append(s2)
    w13_packed = torch.stack(w13p_list, 0).to(DEV)      # [E, H, 2I//8]
    w2_packed = torch.stack(w2p_list, 0).to(DEV)        # [E, I, H//8]
    w13_scales = torch.stack(s13_list, 0).to(DEV)       # [E, H//GS, 2I]
    w2_scales = torch.stack(s2_list, 0).to(DEV)         # [E, I//GS, H]
    w13_zp = _sym_zero(H // GS, 2 * I).to(DEV)          # [H//GS, 2I//8]
    w2_zp = _sym_zero(I // GS, H).to(DEV)               # [I//GS, H//8]
    htorch.mark_step()
    return w13_packed, w2_packed, w13_scales, w2_scales, w13_zp, w2_zp


def make_routing(tokens, K):
    """Random GLOBAL top_k routing (this rank's window is [0, E)). Keep distinct local
    experts <= K (the eligibility invariant tokens*top_k <= K is satisfied for tokens=1,
    K>=TOPK). experts_min=0 for the single-rank microbench."""
    torch.manual_seed(1)
    x = torch.randn(tokens, H, dtype=DT, device=DEV)
    topk_ids = torch.randint(0, E, (tokens, TOPK), dtype=torch.int64, device=DEV)
    w = torch.rand(tokens, TOPK, dtype=torch.float32, device=DEV)
    topk_weights = (w / w.sum(dim=-1, keepdim=True)).to(DT)
    htorch.mark_step()
    return x, topk_ids, topk_weights


def _read_hbm_proxy():
    """Best-effort HBM/device-memory proxy. Returns (label, value) or None. Tries
    habana_frameworks memory stats first (always present), then any BW counter. The
    intent: a fused win writes far less bf16 weight to HBM, so `MaxInUse`/peak-alloc
    and any traffic counter should be lower for the fused leg."""
    try:
        import habana_frameworks.torch.hpu as hpu
        stats = hpu.memory_stats()
        for key in ("MaxInUse", "InUse", "MaxAllocated", "Allocated"):
            if key in stats:
                return (f"hpu.memory_stats[{key}]", int(stats[key]))
    except Exception:
        pass
    return None


def _reset_hbm_proxy():
    try:
        import habana_frameworks.torch.hpu as hpu
        if hasattr(hpu, "reset_peak_memory_stats"):
            hpu.reset_peak_memory_stats()
    except Exception:
        pass


def run_correctness(weights, routing, K):
    w13_packed, w2_packed, w13_scales, w2_scales, w13_zp, w2_zp = weights
    x, topk_ids, topk_weights = routing
    common = dict(experts_min=0, num_local_experts=E, K=K, intermediate_size=I, activation="silu")

    fused = fused_routed_moe(x, topk_ids, topk_weights,
                             w13_packed, w2_packed, w13_scales, w2_scales,
                             w13_zp, w2_zp, **common)
    htorch.mark_step()
    ref = reference_routed_moe(x, topk_ids, topk_weights,
                               w13_packed, w2_packed, w13_scales, w2_scales,
                               w13_zp, w2_zp, **common)
    htorch.mark_step()
    if fused is None or ref is None:
        print("[FAIL] eligibility: fused or reference returned None (expected GPTQ-sym path)", flush=True)
        raise SystemExit(2)
    f_cpu = fused.detach().to("cpu").float()
    r_cpu = ref.detach().to("cpu").float()
    maxdiff = (f_cpu - r_cpu).abs().max().item()
    rel = maxdiff / max(1e-6, r_cpu.abs().max().item())
    print(f"[correctness] fused vs intrinsic-reference  maxdiff={maxdiff:.4e}  "
          f"rel={rel:.4e}  ref_absmax={r_cpu.abs().max().item():.4f}", flush=True)
    if not (maxdiff < 1e-2):
        print("=" * 72, flush=True)
        print(f"[FAIL] CORRECTNESS: maxdiff {maxdiff:.4e} >= 1e-2 — the fused spike does "
              f"NOT match the proven convert_from_uint4 + mixture_of_experts reference. "
              f"The spike is WRONG, do not trust any latency numbers below.", flush=True)
        print(f"  fused[0,:6]={f_cpu[0,:6].tolist()}", flush=True)
        print(f"  ref  [0,:6]={r_cpu[0,:6].tolist()}", flush=True)
        print("=" * 72, flush=True)
        raise SystemExit(1)
    print("[correctness] PASS (maxdiff < 1e-2)", flush=True)
    return fused


def _bench_leg(fn, iters, warmup):
    """Latency of a callable that returns an HPU tensor. Warmup compiles the graph;
    then time `iters` calls and force a device sync each iter (mark_step + .item on a
    1-elem reduction) so we measure real wall time, not just lazy-graph enqueue."""
    for _ in range(warmup):
        o = fn()
        htorch.mark_step()
    # final sync before timing
    _ = o.detach().to("cpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn()
        htorch.mark_step()
    _ = o.detach().to("cpu")     # one sync at the end of the timed batch
    dt = time.perf_counter() - t0
    return dt / iters


def run_latency(weights, routing, K, iters, warmup, tokens):
    w13_packed, w2_packed, w13_scales, w2_scales, w13_zp, w2_zp = weights
    x, topk_ids, topk_weights = routing
    common = dict(experts_min=0, num_local_experts=E, K=K, intermediate_size=I, activation="silu")

    fused_fn = lambda: fused_routed_moe(x, topk_ids, topk_weights,
                                        w13_packed, w2_packed, w13_scales, w2_scales,
                                        w13_zp, w2_zp, **common)
    intr_fn = lambda: reference_routed_moe(x, topk_ids, topk_weights,
                                           w13_packed, w2_packed, w13_scales, w2_scales,
                                           w13_zp, w2_zp, **common)

    _reset_hbm_proxy()
    hbm_before = _read_hbm_proxy()
    intr_s = _bench_leg(intr_fn, iters, warmup)
    hbm_intr = _read_hbm_proxy()

    _reset_hbm_proxy()
    fused_s = _bench_leg(fused_fn, iters, warmup)
    hbm_fused = _read_hbm_proxy()

    intr_toks = tokens / intr_s
    fused_toks = tokens / fused_s
    speedup = intr_s / fused_s if fused_s > 0 else 0.0

    print("", flush=True)
    print(f"[latency] K={K} tokens={tokens} iters={iters} (bs=1 single-stream)", flush=True)
    print(f"  intrinsic (mark_step + mixture_of_experts):  "
          f"{intr_s*1e3:8.3f} ms/iter   {intr_toks:8.2f} tok/s", flush=True)
    print(f"  fused     (dequant+matmul, no mark_step)  :  "
          f"{fused_s*1e3:8.3f} ms/iter   {fused_toks:8.2f} tok/s", flush=True)
    print(f"  speedup (intrinsic / fused) = {speedup:.3f}x", flush=True)

    if fused_s >= intr_s:
        print("", flush=True)
        print("[diagnosis] FUSED >= INTRINSIC: the spike captured NO win.", flush=True)
        print("[diagnosis]   The graph compiler did NOT fuse convert_from_uint4 into the matmul,", flush=True)
        print("[diagnosis]   OR it fused but still spills the bf16 tile to HBM, so the per-token", flush=True)
        print("[diagnosis]   HBM round-trip the spike was meant to delete is still present.", flush=True)
        print("[diagnosis]   Next: inspect the GC graph (GRAPH_VISUALIZATION=1 / hltv dump) to see", flush=True)
        print("[diagnosis]   whether convert_from_uint4 and the GEMM landed in ONE fused TPC/MME", flush=True)
        print("[diagnosis]   cluster or remained two ops with a bf16 [in,out] tensor materialized", flush=True)
        print("[diagnosis]   between them (= the HBM write). Until fused < intrinsic, do not ship.", flush=True)
    else:
        print(f"[result] fused is {speedup:.2f}x faster -> the GC fusion is capturing a win.", flush=True)

    if hbm_intr and hbm_fused:
        print("", flush=True)
        print(f"[hbm-proxy] {hbm_intr[0]}  intrinsic_peak={hbm_intr[1]}  fused_peak={hbm_fused[1]}  "
              f"delta={hbm_intr[1]-hbm_fused[1]} (positive => fused touched less device memory)",
              flush=True)
    else:
        print("[hbm-proxy] no reachable HBM/device-memory counter; skipped.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--K", type=int, default=TOPK)     # >= distinct local experts; default == top_k
    ap.add_argument("--tokens", type=int, default=1)   # single-stream
    args = ap.parse_args()

    if args.tokens * TOPK > args.K:
        print(f"[warn] tokens*top_k ({args.tokens*TOPK}) > K ({args.K}); the routed fast path "
              f"assumes distinct local experts <= K. Raising K to {args.tokens*TOPK}.", flush=True)
        args.K = args.tokens * TOPK

    print(f"[setup] GLM-5.1-int4 synthetic: H={H} I={I} E={E} top_k={TOPK} gs={GS} "
          f"zero={ZERO} K={args.K} lazy={htorch.is_lazy()}", flush=True)
    weights = build_synthetic()
    routing = make_routing(args.tokens, args.K)

    run_correctness(weights, routing, args.K)
    run_latency(weights, routing, args.K, args.iters, args.warmup, args.tokens)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
