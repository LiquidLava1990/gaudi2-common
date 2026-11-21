#!/usr/bin/env python3
"""READ-vs-UNPACK decisive microbench for the GLM int4-MoE dequant on Gaudi2.

THE QUESTION (single-stream 40+ campaign): the two `fused_kernel`s = ~51 ms of the
bs=1 forward. Is that 51 ms HBM-READ-BANDWIDTH-bound (reading the routed experts'
int4 weights — fundamental, can't cut) or TPC-UNPACK-COMPUTE-bound (the int4->bf16
bit-unpack — cuttable by a faster kernel / fp8-resident)?

This isolates the two by timing, at GLM-5.1/5.2 dims (H=6144, I=2048, GS=128), a
PER-LAYER MoE weight footprint and scaling it to the 75 MoE layers. Three regimes
on the SAME packed int4 weights (synthetic, packed bit-faithfully via the prod
helper), each timed with a real device sync:

  (R) PURE HBM READ      : read packed int4 + a cheap reduction (sum) -> 1 scalar.
                           No unpack, no GEMM. == the HBM read floor for those bytes.
  (D) READ + DEQUANT     : torch.ops.hpu.convert_from_uint4(int4)->bf16 + mark_step.
                           == read + the TPC int4->bf16 unpack, bf16 MATERIALIZED to HBM
                           (the stock _routed_moe shape: dequant then a mark_step boundary).
  (Dx) READ + DEQUANT, no writeback proxy : dequant -> immediately reduce to 1 scalar
                           in the SAME graph (no mark_step between) so the bf16 need not
                           be fully realized to HBM -> isolates unpack WITHOUT the bf16 write.
  (G) FUSED DEQUANT+MME  : convert_from_uint4 -> matmul(x, W) in one graph (the real
                           expert GEMM compute on the dequantized weight).

INTERPRETATION:
  * If (D) ~= (R)  -> the unpack is ~free; the cost is the HBM read => READ-BOUND.
  * If (D) >> (R)  -> the unpack/elementwise dominates => UNPACK/COMPUTE-BOUND.
  * (Dx) vs (D)    -> how much of (D) is the bf16 HBM WRITEBACK vs the unpack itself.
  * (G) vs (D)     -> the MME compute increment (should be small at bs=1).

We bench BOTH expert counts that the prod path uses:
  --experts 8   : the routed-K8 single-stream DECODE forward (1 expert/card avg, but
                  we time the per-card serial work as N experts on ONE card here).
  --experts 32  : the all-LOCAL-expert MTP-VERIFY forward (EP=8 -> 32 local/card).

All work is sized as (N experts) x (75 MoE layers) of GLM w13+w2 to be directly
comparable to the measured ~51 ms two-fused-kernel figure.

RUN ON THE HPU BOX (cards must be free):
  PT_HPU_LAZY_MODE=1 python tests/read_vs_unpack_microbench.py --experts 8 --layers 75
  PT_HPU_LAZY_MODE=1 python tests/read_vs_unpack_microbench.py --experts 32 --layers 75

Single-process, single-card, NO vLLM, NO model load, NO docker. Allocates N experts'
int4 weights (~19 MB/expert) -> e.g. 32 experts ~620 MB, fits 96 GB easily.
"""
import argparse
import os
import sys
import time

import torch
import habana_frameworks.torch.core as htorch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pk,
)
from vllm.scalar_type import scalar_types

QT = scalar_types.uint4
DEV = "hpu"
DT = torch.bfloat16

# === GLM-5.1/5.2-int4 shapes ===
H = 6144            # hidden
I = 2048            # moe_intermediate
TWO_I = 2 * I       # 4096 gate+up
GS = 128            # group size
ZERO = 8            # GPTQ-sym shared zero

HBM_BW = 2.45e12    # bytes/s (project's cited Gaudi2 figure) -- for the floor printout


def _mk_expert(out_, in_):
    """One expert weight in the LOADED op layout: packed int32 [in, out//8], scale
    bf16 [in//GS, out]. Built from random int4 codes via the prod pack helper so the
    convert_from_uint4 round-trip is bit-faithful."""
    codes = torch.randint(0, 16, (in_, out_), dtype=torch.int32)
    packed = pk(codes, QT, packed_dim=1)              # [in, out//8] int32
    scale = (torch.rand(in_ // GS, out_, dtype=DT) * 0.02 + 0.01)
    return packed, scale


def _sym_zero(groups, out):
    return pk(torch.full((groups, out), ZERO, dtype=torch.int32), QT, packed_dim=1)


def build(experts):
    """Build `experts` distinct GLM experts' (w13, w2) packed int4 + scales on HPU.
    Returns lists so the bench reads DISTINCT bytes per expert (no cache reuse)."""
    torch.manual_seed(0)
    w13p, w13s, w2p, w2s = [], [], [], []
    for _ in range(experts):
        p13, s13 = _mk_expert(TWO_I, H)   # w13: in=H,out=2I -> [H,2I//8], scale [H//GS,2I]
        p2, s2 = _mk_expert(H, I)         # w2 : in=I,out=H  -> [I,H//8],  scale [I//GS,H]
        w13p.append(p13.to(DEV)); w13s.append(s13.to(DEV))
        w2p.append(p2.to(DEV));   w2s.append(s2.to(DEV))
    z13 = _sym_zero(H // GS, TWO_I).to(DEV)   # [H//GS, 2I//8]
    z2 = _sym_zero(I // GS, H).to(DEV)        # [I//GS, H//8]
    x = torch.randn(1, H, dtype=DT, device=DEV)   # bs=1 activation
    # contiguous stacks for the clean read-bandwidth ceiling
    big13 = torch.stack(w13p, 0).contiguous()  # [experts, H, 2I//8]
    big2 = torch.stack(w2p, 0).contiguous()    # [experts, I, H//8]
    htorch.mark_step()
    return dict(w13p=w13p, w13s=w13s, w2p=w2p, w2s=w2s, z13=z13, z2=z2, x=x,
                big13=big13, big2=big2)


def _u4(packed, out_cols):
    """reinterpret int32-packed -> uint4 view (the bridge idiom) for convert_from_uint4.
    convert_from_uint4 takes the PACKED int32 directly (it reinterprets internally),
    so we just pass the packed tensor; out_cols documents the unpacked width."""
    return packed


def dequant(packed, scale, zp):
    """int4 -> bf16 via the public op (== the stock get_dequant_weight)."""
    return torch.ops.hpu.convert_from_uint4(packed, scale, zp, torch.bfloat16, None)


# ---------------------------------------------------------------------------
# Regimes. Each `body(w)` does ONE full-model-worth of MoE weight work for the
# given expert count x layer count, returning a tiny tensor to sync on.
# We replicate the per-expert weights across `layers` by RE-READING the same
# expert tensors `layers` times (the bytes are re-read from HBM each pass, so the
# HBM traffic == experts*layers*expert_bytes, matching the real per-token read).
# ---------------------------------------------------------------------------
def body_read(w, experts, layers):
    """(R) Pure HBM read: sum the raw int4 packed bytes (as int32) -> 1 scalar.
    Touches every weight byte, no unpack, no GEMM."""
    acc = torch.zeros((), dtype=torch.float32, device=DEV)
    for _ in range(layers):
        for e in range(experts):
            acc = acc + w["w13p"][e].float().sum() + w["w2p"][e].float().sum()
    return acc


def body_read_contig(w, experts, layers):
    """(Rc) Pure HBM read, CONTIGUOUS: read the pre-stacked [experts, ...] int32 blocks
    with ONE reduction each, x layers. Removes per-expert launch overhead -> the true
    HBM read-bandwidth ceiling for these bytes. To DEFEAT caching/CSE of a loop-invariant
    read (which would let it read from SRAM after pass 1), we add the loop index into the
    tensor each pass so every pass reads + touches distinct values (still one big op)."""
    acc = torch.zeros((), dtype=torch.float32, device=DEV)
    big13 = w["big13"]   # [experts, H, 2I//8] int32 contiguous
    big2 = w["big2"]     # [experts, I, H//8]  int32 contiguous
    for k in range(layers):
        acc = acc + (big13 + k).to(torch.float32).sum() + (big2 + k).to(torch.float32).sum()
    return acc


def body_dequant_writeback(w, experts, layers):
    """(D) read + int4->bf16 unpack, bf16 MATERIALIZED (mark_step per expert, the
    stock _routed_moe shape). Returns a sum of tiny slices to sync."""
    accs = []
    for _ in range(layers):
        for e in range(experts):
            W13 = dequant(w["w13p"][e], w["w13s"][e], w["z13"])   # [H, 2I] bf16
            W2 = dequant(w["w2p"][e], w["w2s"][e], w["z2"])       # [I, H]  bf16
            htorch.mark_step()   # MATERIALIZE (the stock mandatory boundary)
            accs.append(W13[0, 0].float() + W2[0, 0].float())
    return torch.stack(accs).sum()


def body_dequant_nowrite(w, experts, layers):
    """(Dx) read + unpack but reduce-in-graph (NO mark_step between dequant and the
    reduce) so the GC need not realize the full bf16 to HBM -> isolates the unpack
    elementwise WITHOUT the bf16 writeback cost."""
    acc = torch.zeros((), dtype=torch.float32, device=DEV)
    for _ in range(layers):
        for e in range(experts):
            W13 = dequant(w["w13p"][e], w["w13s"][e], w["z13"])
            W2 = dequant(w["w2p"][e], w["w2s"][e], w["z2"])
            acc = acc + W13.float().sum() + W2.float().sum()
    return acc


def body_fused_gemm(w, experts, layers):
    """(G) read + unpack + the real expert GEMM: gu = x@W13 ; y = h@W2 (in one graph
    so the GC can fuse dequant->MME, SRAM-resident). bs=1 single-stream."""
    acc = torch.zeros((), dtype=torch.float32, device=DEV)
    x = w["x"]
    for _ in range(layers):
        for e in range(experts):
            W13 = dequant(w["w13p"][e], w["w13s"][e], w["z13"])   # [H, 2I]
            gu = torch.matmul(x, W13)                              # [1, 2I]
            gate = gu[:, :I]; up = gu[:, I:]
            h = torch.nn.functional.silu(gate) * up               # [1, I]
            W2 = dequant(w["w2p"][e], w["w2s"][e], w["z2"])       # [I, H]
            y = torch.matmul(h, W2)                               # [1, H]
            acc = acc + y.sum()
    return acc


def bench(body, w, experts, layers, iters, warmup, label):
    fn = lambda: body(w, experts, layers)
    for _ in range(warmup):
        o = fn(); htorch.mark_step()
    _ = o.detach().to("cpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn(); htorch.mark_step()
    _ = o.detach().to("cpu")
    dt = (time.perf_counter() - t0) / iters
    print(f"  {label:42s}  {dt*1e3:9.3f} ms/iter", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    exp_int4 = (H * TWO_I + I * H) * 0.5
    scale_b = ((H // GS) * TWO_I + (I // GS) * H) * 2
    total_int4 = args.experts * args.layers * (exp_int4 + scale_b)
    total_bf16 = args.experts * args.layers * (H * TWO_I + I * H) * 2

    print("=" * 78, flush=True)
    print(f"READ-vs-UNPACK microbench | GLM dims H={H} I={I} GS={GS} | "
          f"experts={args.experts} layers={args.layers} lazy={htorch.is_lazy()}", flush=True)
    print(f"  per-expert int4+scale = {(exp_int4+scale_b)/1e6:.2f} MB", flush=True)
    print(f"  TOTAL int4 read/iter  = {total_int4/1e9:.4f} GB  -> theory READ floor "
          f"{total_int4/HBM_BW*1e3:.3f} ms @ {HBM_BW/1e12} TB/s", flush=True)
    print(f"  TOTAL bf16 if written = {total_bf16/1e9:.4f} GB  -> theory WRITE "
          f"{total_bf16/HBM_BW*1e3:.3f} ms", flush=True)
    print("=" * 78, flush=True)

    w = build(args.experts)

    print("\n[regimes] (lower = faster; compare to ~51 ms two-fused-kernel target)", flush=True)
    # (Rb) TRUE HBM read ceiling: one giant bf16 block of EXACTLY total_int4 bytes, summed
    # NATIVELY in bf16 (no dtype-convert compute) once per iter (no loop, no CSE) -> the real
    # device read bandwidth. bf16 .sum() is a near-pure streaming read (tree-reduce, ~1 op/elem).
    # cap the giant block at ~8 GB to avoid OOM; loop it to reach total_int4 bytes so the
    # measured BW is for the same byte volume.
    cap_bytes = min(int(total_int4), 8 * 1024**3)
    nelem = max(1, cap_bytes // 2)         # 2 bytes/bf16 elem
    npass = max(1, round(total_int4 / cap_bytes))
    giant = torch.randn(nelem, dtype=DT, device=DEV)
    htorch.mark_step()
    def body_giant():
        a = torch.zeros((), dtype=torch.float32, device=DEV)
        for p in range(npass):
            a = a + (giant + p).sum()      # +p defeats CSE across passes; bf16 streaming read
        return a
    tRb = bench(lambda *a: body_giant(), w, args.experts, args.layers, args.iters, args.warmup,
                f"(Rb) giant bf16 read x{npass} (true BW ceiling)")
    tRc = bench(body_read_contig, w, args.experts, args.layers, args.iters, args.warmup,
                "(Rc) contiguous stacked read (per-pass)")
    tR = bench(body_read, w, args.experts, args.layers, args.iters, args.warmup,
               "(R) pure HBM read, per-expert (w/ launch ovh)")
    tDx = bench(body_dequant_nowrite, w, args.experts, args.layers, args.iters, args.warmup,
                "(Dx) read+dequant, reduce-in-graph (no WB)")
    tD = bench(body_dequant_writeback, w, args.experts, args.layers, args.iters, args.warmup,
               "(D) read+dequant, bf16 materialized (WB)")
    tG = bench(body_fused_gemm, w, args.experts, args.layers, args.iters, args.warmup,
               "(G) read+dequant+MME GEMM (fused)")

    print("\n" + "=" * 78, flush=True)
    print("VERDICT", flush=True)
    print("=" * 78, flush=True)
    print(f"  read floor (theory @2.45TB/s) : {total_int4/HBM_BW*1e3:7.3f} ms", flush=True)
    print(f"  (Rb) TRUE giant-block read    : {tRb*1e3:7.3f} ms   "
          f"(achieved BW {total_int4/tRb/1e12:.3f} TB/s = {100*total_int4/tRb/HBM_BW:.0f}% of {HBM_BW/1e12})", flush=True)
    tRc_eff = tRb  # use the trustworthy giant-block read as the ceiling
    print(f"  (Rc) contiguous stacked read  : {tRc*1e3:7.3f} ms   (may CSE; (Rb) is the trusted ceiling)", flush=True)
    print(f"  (R)  per-expert read+launch   : {tR*1e3:7.3f} ms   "
          f"(= {(tR-tRc)*1e3:+.1f} ms launch/granularity overhead over Rc)", flush=True)
    print(f"  (Dx) +unpack (no WB)          : {tDx*1e3:7.3f} ms", flush=True)
    print(f"  (D)  +unpack +bf16 WB         : {tD*1e3:7.3f} ms", flush=True)
    print(f"  (G)  +MME (fused)             : {tG*1e3:7.3f} ms", flush=True)
    print("", flush=True)
    # Use the per-expert read (R) as the read baseline for the dequant regimes, since
    # (Dx)/(D)/(G) are ALSO per-expert (same launch granularity) -> apples-to-apples.
    print(f"  --- read-vs-unpack (per-expert granularity, R as baseline) ---", flush=True)
    unpack_add = tDx - tR
    print(f"  (Dx) - (R) = pure unpack-compute add = {unpack_add*1e3:+.2f} ms "
          f"({100*unpack_add/tDx:.0f}% of Dx)", flush=True)
    print(f"  (D)  - (Dx) = bf16 WRITEBACK add     = {(tD-tDx)*1e3:+.2f} ms "
          f"(the stock mark_step HBM-write penalty)", flush=True)
    print(f"  (G)  - (Dx) = MME-fused increment    = {(tG-tDx)*1e3:+.2f} ms", flush=True)
    print("", flush=True)
    print(f"  --- read-vs-unpack (CLEAN, Rb bandwidth ceiling as the read floor) ---", flush=True)
    tRc = tRc_eff
    print(f"  read ceiling (Rb) = {tRc*1e3:.2f} ms ; full fused-expert (G) = {tG*1e3:.2f} ms", flush=True)
    print(f"  unpack+overhead over the read ceiling = {(tG-tRc)*1e3:+.2f} ms = {100*(tG-tRc)/tG:.0f}% of (G)", flush=True)
    if tRc > 0 and tG > 1.8 * tRc:
        print("  >>> UNPACK/OVERHEAD-BOUND: the read ceiling is a small fraction of the work.", flush=True)
        print("      The cost is the TPC int4->bf16 unpack + per-op launch, NOT HBM read.", flush=True)
        print("      LEVERS: (1) fewer/bigger kernels (batched-K, one recipe); (2) fp8-resident", flush=True)
        print("      (skip unpack); (3) amortize via accept-len. fp8 does NOT cut HBM read here", flush=True)
        print("      (same int4 bytes) but cuts unpack + 2x MME.", flush=True)
    elif tRc > 0:
        print("  >>> READ-BANDWIDTH-BOUND: HBM read is the dominant floor; only amortization helps.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
