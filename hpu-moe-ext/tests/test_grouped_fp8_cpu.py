#!/usr/bin/env python3
"""CARD-FREE CPU oracle for the GROUPED routed-fp8 M3 MoE island
(hpu_moe_ext/fused_grouped_moe_fp8.py + the grouped .so).

Proves the ALGORITHM card-free (the .so build proves COMPILATION; on-card A/B is the
coordinator's deferred gate). Two comparisons + a drop probe:

  (A) GROUPING IS BYTE-EXACT.  grouped(wrapper) vs all-16 dense, BOTH using the SAME
      fp8-MME emulation per expert. The only difference is the permute/gather/scatter
      -> relNorm must be ~0 (machine eps). This is the load-bearing "routed-4  subset
      of all-16, non-routed contribute 0" proof.

  (B) FP8 PATH MATCHES THE CHAMPION.  grouped(fp8 emulation) vs the EXACT champion
      `_fp8_route_b_allexpert` math (fp8 weight -> bf16 dequant, bf16 matmul, dense
      [T,E] router) -> relNorm < 5e-2 (the fp8 grid noise vs bf16 dequant). This is
      the serve-equivalence bound.

  (C) DROP PROBE (informational).  cap < worst-expert count -> overflow tokens drop;
      report the relNorm so the serve cap can be chosen against the load histogram.

The fp8-MME emulation mirrors moe_fused_monolithic_fp8_emit.h verbatim: per-token
dynamic_quant (amax_eps/240, ROUND_HALF_NE via torch's e4m3 cast), fp8 x fp8 -> bf16,
the EXACT swigluoai tail (m3_expert_runtime_patch._swigluoai_tail), then the same for
the down GEMM. Run under an OpenBLAS venv (e.g. ~/.venvs/hf): python tests/test_grouped_fp8_cpu.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hpu_moe_ext.fused_grouped_moe_fp8 import build_group_plan, grouped_moe_fp8  # noqa: E402

FP8 = torch.float8_e4m3fn
F32 = torch.float32
BF16 = torch.bfloat16
ALPHA, LIMIT = 1.702, 7.0
FP8_MAX = 240.0  # kMonoFp8MaxFnuz (the emit's dynamic-quant denominator)


# ----------------------------------------------------------------------------------
# EXACT swigluoai tail (m3_expert_runtime_patch._swigluoai_tail, lines 217-223).
# ----------------------------------------------------------------------------------
def swigluoai_tail(gate_up, alpha=ALPHA, limit=LIMIT):
    gate, up = gate_up.chunk(2, dim=-1)          # gate=FIRST, up=SECOND (contiguous)
    gate = gate.clamp(max=limit)                 # UPPER bound only
    up = up.clamp(min=-limit, max=limit)         # symmetric
    glu = gate * torch.sigmoid(gate * alpha)     # alpha INSIDE sigmoid
    return (up + 1.0) * glu                       # (up+1)*glu


# ----------------------------------------------------------------------------------
# fp8-MME emulation == emit_dynamic_quant + emit_fp8_batch_gemm (per expert, one GEMM).
#   a_bf16 [..., K] ; w_fp8 [K, N] ; w_scale [N]  ->  [..., N] bf16
# w_rec = w_fp8.f32 * w_scale (per-OUT-channel, broadcast over K=in).
# a_rec = round_fp8(a * 240/(amax_eps)).f32 * (amax_eps/240) (per-token, last-axis amax).
# ----------------------------------------------------------------------------------
def _fp8_mme(a_bf16, w_fp8, w_scale):
    a = a_bf16.to(F32)
    amax = a.abs().amax(dim=-1, keepdim=True)                 # [...,1]
    amax_eps = amax + 1e-8
    a_fp8 = (a * (FP8_MAX / amax_eps)).to(FP8)                # convert_to_fp8(1/s)
    a_rec = a_fp8.to(F32) * (amax_eps / FP8_MAX)             # gemm recovers with s
    w_rec = w_fp8.to(F32) * w_scale.to(F32).view(1, -1)      # [K,N] per-out-channel
    return (a_rec @ w_rec).to(BF16)


def expert_fp8(x_e, w13_fp8_e, s13_e, w2_fp8_e, s2_e, alpha=ALPHA, limit=LIMIT):
    """ONE expert, fp8 MME both GEMMs (the .so's per-expert chain). x_e [.,H] ->[.,H]."""
    gu = _fp8_mme(x_e, w13_fp8_e, s13_e)                     # [.,2I] bf16
    h = swigluoai_tail(gu, alpha, limit)                     # [.,I]
    return _fp8_mme(h, w2_fp8_e, s2_e)                        # [.,H] bf16


# ----------------------------------------------------------------------------------
# The grouped GEMM the wrapper calls (the .so stand-in). xg[E,cap,H] -> yg[E,cap,H].
# ----------------------------------------------------------------------------------
def grouped_gemm_cpu(xg, w13_fp8, w13_scale, w2_fp8, w2_scale, alpha, limit):
    E, cap, H = xg.shape
    yg = torch.empty(E, cap, H, dtype=BF16)
    for e in range(E):
        yg[e] = expert_fp8(xg[e], w13_fp8[e], w13_scale[e], w2_fp8[e], w2_scale[e],
                           alpha, limit)
    return yg


# ----------------------------------------------------------------------------------
# PER-(token,expert) PRODUCT tensors, fp8 emulation. Isolates the GROUPING (routing +
# gather + per-expert GEMM + weight) from the benign bf16 sum-ORDER noise of the final
# reduce -- so (A) tests EXACTLY the claim "routed-4 subset of all-16, non-routed 0".
#   P[t,e,:] = route[t,e] * expert_fp8(x[t], e)
# ----------------------------------------------------------------------------------
def allexpert_products(x, w13_fp8, w13_scale, w2_fp8, w2_scale, route):
    T, H = x.shape
    E = w13_fp8.shape[0]
    P = torch.zeros(T, E, H, dtype=BF16)
    for e in range(E):
        y_e = expert_fp8(x, w13_fp8[e], w13_scale[e], w2_fp8[e], w2_scale[e])  # [T,H]
        P[:, e, :] = route[:, e:e + 1].to(BF16) * y_e
    return P


def grouped_products(x, w13_fp8, w13_scale, w2_fp8, w2_scale, ids, weights, emin, E, cap):
    """The SAME products, reached via the wrapper's plan + grouped GEMM, scattered to
    [T,E,H]. Each (t,e) has at most ONE bin entry -> no accumulation -> a bit-exact
    grouping must reproduce allexpert_products()."""
    T, H = x.shape
    gi, wg = build_group_plan(ids, weights, E, emin, cap)         # [E,cap]
    xg = x.index_select(0, gi.reshape(-1)).view(E, cap, H)        # [E,cap,H]
    yg = grouped_gemm_cpu(xg, w13_fp8, w13_scale, w2_fp8, w2_scale, ALPHA, LIMIT)
    yw = (yg * wg.view(E, cap, 1).to(BF16)).view(E * cap, H)      # [E*cap,H] weighted
    e_of = torch.arange(E).view(E, 1).expand(E, cap).reshape(-1)  # [E*cap] expert per slot
    dest = gi.reshape(-1) * E + e_of                              # [E*cap] -> (t,e) flat
    P = torch.zeros(T * E, H, dtype=BF16).index_add(0, dest, yw)
    return P.view(T, E, H)


def allexpert_fp8_emu(x, w13_fp8, w13_scale, w2_fp8, w2_scale, route):
    """Dense all-16 end-to-end (fp8 emulation), for the drop-probe baseline."""
    return allexpert_products(x, w13_fp8, w13_scale, w2_fp8, w2_scale, route).sum(dim=1)


# ----------------------------------------------------------------------------------
# The EXACT champion math: _fp8_route_b_allexpert (fp8->bf16 dequant, bf16 matmul).
# ----------------------------------------------------------------------------------
def route_b_allexpert_bf16(x, w13_fp8, w13_scale, w2_fp8, w2_scale, route):
    T, H = x.shape
    E = w13_fp8.shape[0]
    out = torch.zeros(T, H, dtype=BF16)
    for e in range(E):
        w13_deq = (w13_fp8[e].to(F32) * w13_scale[e].view(1, -1)).to(BF16)   # [H,2I]
        w2_deq = (w2_fp8[e].to(F32) * w2_scale[e].view(1, -1)).to(BF16)      # [I,H]
        gate_up = x.to(BF16) @ w13_deq                                       # [T,2I]
        h = swigluoai_tail(gate_up, ALPHA, LIMIT)                            # [T,I]
        y_e = h @ w2_deq                                                     # [T,H]
        out = out + route[:, e:e + 1].to(BF16) * y_e
    return out


def build_route(topk_ids, topk_weights, E, emin):
    """The dense [T,E] router matrix _fp8_route_b_allexpert builds."""
    local = topk_ids.to(torch.int64) - emin
    in_win = (local >= 0) & (local < E)
    safe_local = torch.where(in_win, local, torch.zeros_like(local))
    safe_w = torch.where(in_win, topk_weights.to(F32), torch.zeros_like(topk_weights, dtype=F32))
    return torch.zeros(topk_ids.shape[0], E, dtype=F32).scatter_add_(1, safe_local, safe_w)


def relnorm(a, b):
    a = a.to(F32); b = b.to(F32)
    return (torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12)).item()


def rand_fp8(*shape):
    """A random e4m3 tensor in a realistic weight range (round bf16 -> fp8)."""
    return (torch.randn(*shape) * 0.1).to(FP8)


def make_case(T=128, E=16, H=512, I=256, top_k=4, emin=0, cross_rank_frac=0.15, seed=0):
    """Synthetic M3-shaped case. Small H/I for a fast CPU GEMM; routing matches the
    real serve (top_k of this rank's E-window, with a few GLOBAL ids out-of-window to
    exercise the in_win mask). Returns the stacks + routing + the dense route."""
    g = torch.Generator().manual_seed(seed)
    twoI = 2 * I
    x = (torch.randn(T, H, generator=g) * 1.0).to(BF16)
    w13_fp8 = (torch.randn(E, H, twoI, generator=g) * 0.1).to(FP8)
    w2_fp8 = (torch.randn(E, I, H, generator=g) * 0.1).to(FP8)
    w13_scale = (torch.rand(E, twoI, generator=g) * 0.02 + 0.005).to(F32)
    w2_scale = (torch.rand(E, H, generator=g) * 0.02 + 0.005).to(F32)

    # routing: for each token, top_k DISTINCT experts. A fraction land OUT of this
    # rank's [emin, emin+E) window (global id >= emin+E) to exercise the mask.
    ids = torch.empty(T, top_k, dtype=torch.int64)
    for t in range(T):
        perm = torch.randperm(E, generator=g)[:top_k] + emin
        ids[t] = perm
    # push some entries cross-rank (out of window) by adding E to a random subset
    cross = (torch.rand(T, top_k, generator=g) < cross_rank_frac)
    ids = torch.where(cross, ids + E, ids)
    logits = torch.randn(T, top_k, generator=g)
    weights = torch.softmax(logits, dim=-1).to(BF16)

    route = build_route(ids, weights, E, emin)
    return dict(x=x, w13_fp8=w13_fp8, w13_scale=w13_scale, w2_fp8=w2_fp8,
                w2_scale=w2_scale, ids=ids, weights=weights, route=route,
                E=E, emin=emin, top_k=top_k, I=I, H=H, T=T)


def worst_count(route):
    return int((route != 0).sum(dim=0).max().item())


def main():
    print("=" * 90)
    print("GROUPED routed-fp8 M3 MoE -- CARD-FREE CPU oracle")
    print("=" * 90)
    torch.manual_seed(0)
    ok = True

    for seed in (0, 1, 2):
        c = make_case(seed=seed)
        route = c["route"]
        wc = worst_count(route)
        cap = wc                      # tight, NO drop -> byte-exact path
        print(f"\n--- case seed={seed}  T={c['T']} E={c['E']} top_k={c['top_k']} "
              f"H={c['H']} I={c['I']} | worst expert count={wc} -> cap={cap} ---")

        # ---- (A) grouping byte-exact, at the per-(token,expert) PRODUCT level ----
        P_grp = grouped_products(
            c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"], c["w2_scale"],
            c["ids"], c["weights"], c["emin"], c["E"], cap)
        P_all = allexpert_products(
            c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"], c["w2_scale"], route)
        rn_a = relnorm(P_grp, P_all)
        pass_a = rn_a < 1e-5
        print(f"  (A) grouped vs all-16 products [exact] relNorm = {rn_a:.3e}   "
              f"{'PASS' if pass_a else 'FAIL'} (<1e-5 -> routing+GEMM byte-exact)")
        ok &= pass_a

        # ---- (B) fp8 path vs the champion route-b bf16 (END-TO-END wrapper) ----
        out_grp = grouped_moe_fp8(
            c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"], c["w2_scale"],
            c["ids"], c["weights"], c["emin"], c["top_k"],
            alpha=ALPHA, limit=LIMIT, cap=cap, gemm_fn=grouped_gemm_cpu,
            materialize=False)
        out_rb = route_b_allexpert_bf16(
            c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"], c["w2_scale"], route)
        rn_b = relnorm(out_grp, out_rb)
        pass_b = rn_b < 5e-2
        print(f"  (B) grouped fp8 vs route-b bf16        relNorm = {rn_b:.3e}   "
              f"{'PASS' if pass_b else 'FAIL'} (<5e-2 -> fp8 grid noise)")
        ok &= pass_b

        # ---- (C) drop probe: cap below worst count ----
        cap_drop = max(1, wc // 2)
        out_drop = grouped_moe_fp8(
            c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"], c["w2_scale"],
            c["ids"], c["weights"], c["emin"], c["top_k"],
            alpha=ALPHA, limit=LIMIT, cap=cap_drop, gemm_fn=grouped_gemm_cpu,
            materialize=False)
        out_all_fp8 = allexpert_fp8_emu(
            c["x"], c["w13_fp8"], c["w13_scale"], c["w2_fp8"], c["w2_scale"], route)
        rn_c = relnorm(out_drop, out_all_fp8)
        print(f"  (C) drop probe cap={cap_drop} (<{wc})       relNorm = {rn_c:.3e}   "
              f"(informational: overflow tokens dropped)")

        # ---- plan static-shape + capture-clean checks ----
        gi, wg = build_group_plan(c["ids"], c["weights"], c["E"], c["emin"], cap)
        assert tuple(gi.shape) == (c["E"], cap) and tuple(wg.shape) == (c["E"], cap), \
            f"plan shape not static: {tuple(gi.shape)} {tuple(wg.shape)}"
        # weight conservation: every routed (token,expert) weight is represented exactly
        # once in the plan (no drop at cap>=worst) -> sum(wg) == sum(route).
        cons = abs(wg.to(F32).sum().item() - route.sum().item())
        assert cons < 1e-3, f"weight not conserved (drop?): |sum(wg)-sum(route)|={cons:.3e}"
        print(f"  plan shape static [E,cap]=[{c['E']},{cap}]; weight conserved "
              f"|Δ|={cons:.2e} (no drop). OK")

    print("\n" + "=" * 90)
    print(f"ORACLE {'PASS' if ok else 'FAIL'} "
          f"({'grouping byte-exact (A) AND fp8<5e-2 (B) on all cases' if ok else 'see FAIL above'})")
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
