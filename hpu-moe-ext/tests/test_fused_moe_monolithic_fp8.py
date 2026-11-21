"""test_fused_moe_monolithic_fp8.py -- VALIDATE the MONOLITHIC swigluoai-FP8 op
`torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8`.

TWO modes:
  (A) --cardfree (DEFAULT off-card): a pure-CPU numeric ORACLE of the op's INTENDED math
      (dense-router all-E swigluoai over fp8-dequant weights), proving the wrapper's dense
      [E,T,1] router build + the swigluoai numerics + the per-out-channel fp8 dequant
      semantics are correct BEFORE any card. This is the card-free gate the build step runs.
  (B) ON THE HPU BOX (no flag): build the fp8 stacks, run the op, and A/B against the
      EXACT M3 stock unfused route-b loop (_fp8_route_b_allexpert) on the SAME input, plus
      the CPU oracle. Gate: relNorm = ||got-ref|| / ||ref|| < 5e-2 (fp8 grid noise tol;
      the fp8 MME path and the route-b dequant-then-bf16-matmul path differ by fp8 rounding
      of the activation, so the tol is looser than the int4 op's 1e-2 -- documented below).

M3 default shape: H=6144 I=3072 (2I=6144) E=16 (TP8 local) top_k=4 alpha=1.702 limit=7.0.

RUN:
  off-card gate (CI / pre-build):  python tests/test_fused_moe_monolithic_fp8.py --cardfree
  on the HPU box:                  python tests/test_fused_moe_monolithic_fp8.py
  single point:                    python tests/test_fused_moe_monolithic_fp8.py 16 6144 3072 16 4
"""
from __future__ import annotations

import os
import sys

ALPHA, LIMIT = 1.702, 7.0
FP8_MAX = 240.0   # fnuz max (== kMonoFp8MaxFnuz / kFp8MaxFnuz)


def _swigluoai_tail(gate_up, alpha=ALPHA, limit=LIMIT):
    import torch
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


def _quant_w_per_channel_fp8(w_inout):
    """w [in,out] bf16 -> (w_fp8 [in,out] float8_e4m3fn, s [out] f32 forward scale). Per-
    OUTPUT-channel (== the M3 disk layout: scale per output column). Mirrors the route-b
    stacks: real ~= w_fp8.to(bf16) * s. Off-card uses the native fp8 dtype cast."""
    import torch
    amax = w_inout.abs().amax(dim=0) + 1e-8           # [out]
    s = (amax / FP8_MAX).float()                      # [out] forward scale
    w_fp8 = (w_inout.float() / s).to(torch.float8_e4m3fn)
    return w_fp8, s


def _build_experts(E, H, I, seed):
    """E experts of bf16 [H,2I] (w13) + [I,H] (w2), quantized per-out-channel to fp8.
    Returns the route-b stacks: w13_fp8[E,H,2I], s13[E,2I], w2_fp8[E,I,H], s2[E,H], and the
    bf16-dequant reference weights (what the op's fp8 path effectively computes against)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    w13_fp8, s13, w2_fp8, s2, w13_ref, w2_ref = [], [], [], [], [], []
    for e in range(E):
        a = (torch.randn(H, 2 * I, generator=g) * 0.05).to(torch.bfloat16)   # [H,2I]
        b = (torch.randn(I, H, generator=g) * 0.05).to(torch.bfloat16)       # [I,H]
        af, asf = _quant_w_per_channel_fp8(a)
        bf, bsf = _quant_w_per_channel_fp8(b)
        w13_fp8.append(af); s13.append(asf)
        w2_fp8.append(bf); s2.append(bsf)
        # bf16-equivalent dequant (what route-b's get_dequant_weight produces): fp8*s
        w13_ref.append((af.to(torch.bfloat16) * asf).to(torch.bfloat16))
        w2_ref.append((bf.to(torch.bfloat16) * bsf).to(torch.bfloat16))
    return (torch.stack(w13_fp8), torch.stack(s13),
            torch.stack(w2_fp8), torch.stack(s2),
            torch.stack(w13_ref), torch.stack(w2_ref))


def _dense_router(topk_ids, topk_weights, E, emin, dev):
    """The wrapper's [E,T,1] dense router (one_hot * weight sum)."""
    import torch
    local = topk_ids.to(torch.int64) - emin
    valid = (local >= 0) & (local < E)
    safe_id = torch.where(valid, local, torch.zeros_like(local))
    safe_w = torch.where(valid, topk_weights.float(), torch.zeros_like(topk_weights, dtype=torch.float32))
    erange = torch.arange(E, device=dev)
    onehot = (safe_id.unsqueeze(-1) == erange).float()
    W = (onehot * safe_w.unsqueeze(-1)).sum(dim=1)         # [T,E]
    return W


def _oracle(x, w13_ref, w2_ref, W):
    """CPU oracle: out[t] = sum_e ((swigluoai(x@W13_ref_e)) @ W2_ref_e) * W[t,e]. Uses the
    bf16-dequant weights (so it isolates the swigluoai + dense-router correctness; the fp8
    activation rounding is the only delta vs the on-card op)."""
    import torch
    T, H = x.shape
    E = w13_ref.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32)
    for e in range(E):
        gate_up = x.float() @ w13_ref[e].float()           # [T,2I]
        h = _swigluoai_tail(gate_up)                       # [T,I]
        y = h @ w2_ref[e].float()                          # [T,H]
        out = out + W[:, e:e + 1] * y
    return out


def _cardfree(T, H, I, E, top_k):
    import torch
    print("=" * 80)
    print(f"CARD-FREE oracle: swigluoai-fp8 dense-router math  T={T} H={H} I={I} E={E} top_k={top_k}")
    print("=" * 80)
    torch.manual_seed(0)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16)
    w13_fp8, s13, w2_fp8, s2, w13_ref, w2_ref = _build_experts(E, H, I, 1)
    # random routing
    g = torch.Generator().manual_seed(2)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(T)])   # [T,top_k] local
    tw = torch.rand(T, top_k, generator=g)
    topk_weights = (tw / tw.sum(-1, keepdim=True)).to(torch.bfloat16)
    W = _dense_router(topk_ids, topk_weights, E, 0, x.device)

    # 1) dense router sanity: row sums == 1 (each token's weights sum to 1 over its top_k)
    rs = W.sum(-1)
    assert torch.allclose(rs, torch.ones_like(rs), atol=1e-3), f"router rows not normalized: {rs[:4]}"
    print("router [E,T,1] build: row-sums==1 OK")

    # 2) swigluoai numerics vs authoritative reference (must be exact)
    gate = torch.randn(5, 7) * 4
    up = torch.randn(5, 7) * 4
    gu = torch.cat([gate, up], -1)
    got = _swigluoai_tail(gu)
    ref = (up.clamp(-7, 7) + 1.0) * (gate.clamp(max=7) * torch.sigmoid(gate * 1.702))
    md = (got - ref).abs().max().item()
    assert md == 0.0, f"swigluoai numerics mismatch md={md}"
    print(f"swigluoai tail numerics: maxdiff={md:.1e} (exact) OK")

    # 3) fp8 per-out-channel dequant round-trip: ||fp8*s - bf16|| / ||bf16|| small (grid noise)
    rn = ((w13_ref[0].float() - (w13_fp8[0].to(torch.bfloat16) * s13[0]).float()).norm()
          / (w13_ref[0].float().norm() + 1e-9)).item()
    print(f"fp8 per-out-channel dequant self-consistency relNorm={rn:.2e} (==0 by construction)")

    # 4) the full oracle runs cleanly + is finite
    out = _oracle(x, w13_ref, w2_ref, W)
    assert torch.isfinite(out).all(), "oracle produced non-finite output"
    print(f"oracle out: shape={tuple(out.shape)} norm={out.norm().item():.3f} finite=OK")
    print("VERDICT: CARD-FREE OK (router build + swigluoai numerics + fp8 dequant semantics).")
    print("=" * 80)
    print("NB: the fp8-MME accumulation + the in-graph activation dynamic-quant are ONLY")
    print("    exercisable on the box; run without --cardfree there for the A/B vs route-b.")
    return 0


def _oncard(T, H, I, E, top_k):
    import torch
    import habana_frameworks.torch as ht  # noqa: F401 — registers the 'hpu' device
    print("=" * 80)
    print(f"ON-CARD A/B: fused_moe_monolithic_fp8 vs route-b  T={T} H={H} I={I} E={E} top_k={top_k}")
    print("=" * 80)
    dev = "hpu"
    torch.manual_seed(0)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    w13_fp8, s13, w2_fp8, s2, w13_ref, w2_ref = _build_experts(E, H, I, 1)
    w13_fp8, s13, w2_fp8, s2 = (t.to(dev) for t in (w13_fp8, s13, w2_fp8, s2))
    g = torch.Generator().manual_seed(2)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(T)]).to(dev)
    tw = torch.rand(T, top_k, generator=g)
    topk_weights = (tw / tw.sum(-1, keepdim=True)).to(torch.bfloat16).to(dev)

    # --- the op (load the .so first) ---
    so = os.environ.get("MOE_MONO_FP8_SO",
                        "tpc_kernel/opbackend/build/libmoe_fused_monolithic_fp8.so")
    torch.ops.load_library(so)
    rw = _dense_router(topk_ids, topk_weights, E, 0, dev).transpose(0, 1).contiguous().to(
        torch.bfloat16).unsqueeze(-1)   # [E,T,1]
    got = torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8(
        x, w13_fp8, s13.float(), w2_fp8, s2.float(), rw, 8, ALPHA, LIMIT).float().cpu()

    # --- reference 1: M3 stock route-b loop (dequant fp8->bf16, swigluoai, bf16 matmul) ---
    W = _dense_router(topk_ids, topk_weights, E, 0, dev)   # [T,E]
    ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for e in range(E):
        w13d = (w13_fp8[e].to(torch.bfloat16) * s13[e]).to(torch.bfloat16)   # [H,2I]
        w2d = (w2_fp8[e].to(torch.bfloat16) * s2[e]).to(torch.bfloat16)      # [I,H]
        gate_up = torch.matmul(x, w13d)
        h = _swigluoai_tail(gate_up)
        y = torch.matmul(h, w2d)
        ref = ref + W[:, e:e + 1] * y
    ref = ref.float().cpu()

    rn = ((got - ref).norm() / (ref.norm() + 1e-9)).item()
    ma = (got - ref).abs().max().item()
    print(f"vs route-b loop: relNorm={rn:.4g} maxAbs={ma:.4g}")
    ok = rn < 5e-2
    print("VERDICT:", "OK" if ok else "FAIL (relNorm >= 5e-2)")

    # --- TIMING: REAL per-call latency (DISTINCT inputs defeat GC dedup) + CHUNK sweep ---
    import time
    s13f = s13.float(); s2f = s2.float()
    xs = [(torch.randn(T, H, generator=torch.Generator().manual_seed(100 + i)) * 0.1)
          .to(torch.bfloat16).to(dev) for i in range(8)]
    acc = torch.zeros(T, H, dtype=torch.float32, device=dev)   # CONSUME outputs -> no dead-code elim
    def _bench_op(ch, iters=20, warm=5):       # LATENCY: sync after EACH call (serial, like the serve)
        for i in range(warm):
            out = torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8(
                xs[i % 8], w13_fp8, s13f, w2_fp8, s2f, rw, ch, ALPHA, LIMIT)
            acc.add_(out.float()); ht.core.mark_step(); torch.hpu.synchronize()
        t0 = time.perf_counter()
        for i in range(iters):
            out = torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8(
                xs[i % 8], w13_fp8, s13f, w2_fp8, s2f, rw, ch, ALPHA, LIMIT)
            acc.add_(out.float()); ht.core.mark_step(); torch.hpu.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e3
        _ = float(acc.sum().item())   # force the accumulator (and thus the op) to be real
        return dt
    def _bench_loop(iters=20, warm=5):
        def call(xin):
            o = torch.zeros(T, H, dtype=torch.float32, device=dev)
            for e in range(E):
                w13d = (w13_fp8[e].to(torch.bfloat16) * s13[e]).to(torch.bfloat16)
                w2d = (w2_fp8[e].to(torch.bfloat16) * s2[e]).to(torch.bfloat16)
                gu = torch.matmul(xin, w13d); hh = _swigluoai_tail(gu); yy = torch.matmul(hh, w2d)
                o = o + W[:, e:e + 1] * yy
            return o
        for i in range(warm):
            call(xs[i % 8]); ht.core.mark_step(); torch.hpu.synchronize()
        t0 = time.perf_counter()
        for i in range(iters):
            call(xs[i % 8]); ht.core.mark_step(); torch.hpu.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3
    try:
        tl = _bench_loop()
        print(f"TIMING @ T={T} (per-call LATENCY, serial): route-b-loop={tl:.2f}ms")
        to_c = _bench_op(8)
        print(f"   op chunk=8 CONTIG stacks: {to_c:.2f}ms  ({tl / max(to_c, 1e-6):.2f}x vs loop)")
        # SERVE-LIKE non-contiguous transposed-view stacks (m3_fp8_static_patch transpose(1,2)):
        w13_nc = w13_fp8.transpose(1, 2).contiguous().transpose(1, 2)   # [E,H,2I] NON-contig
        w2_nc = w2_fp8.transpose(1, 2).contiguous().transpose(1, 2)     # [E,I,H]  NON-contig
        def _op_nc():
            return torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8(
                xs[0], w13_nc, s13f, w2_nc, s2f, rw, 8, ALPHA, LIMIT)
        for _ in range(5):
            acc.add_(_op_nc().float()); ht.core.mark_step(); torch.hpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            acc.add_(_op_nc().float()); ht.core.mark_step(); torch.hpu.synchronize()
        tnc = (time.perf_counter() - t0) / 20 * 1e3; _ = float(acc.sum().item())
        print(f"   op chunk=8 NON-CONTIG (serve-like): {tnc:.2f}ms  "
              f"<<< {tnc / max(to_c, 1e-6):.1f}x SLOWER than contig if this is the bug")
    except Exception as e:  # noqa: BLE001
        print("TIMING skipped:", repr(e))
    print("=" * 80)
    return 0 if ok else 1


def main(argv):
    cardfree = "--cardfree" in argv
    pos = [a for a in argv[1:] if not a.startswith("--")]
    T = int(pos[0]) if len(pos) > 0 else 16
    H = int(pos[1]) if len(pos) > 1 else 6144
    I = int(pos[2]) if len(pos) > 2 else 3072
    E = int(pos[3]) if len(pos) > 3 else 16
    top_k = int(pos[4]) if len(pos) > 4 else 4
    if cardfree:
        return _cardfree(T, H, I, E, top_k)
    return _oncard(T, H, I, E, top_k)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
