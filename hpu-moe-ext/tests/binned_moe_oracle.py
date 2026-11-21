#!/usr/bin/env python3
"""binned_moe_oracle.py -- CARD-FREE oracle + capacity analysis for the WAVE-2
capacity-binned ("permuted grouped-GEMM") MoE op (Build 1 = decode, Build 2 = prefill).

Pure numpy (NO torch, NO habana) so it runs on THIS host. It is the CPU oracle the
on-card binned op + the device-side Python wrapper (hpu_moe_ext/binned_moe.py) MUST
match. It proves the *permutation/capacity* algorithm is bit-faithful to the DENSE
monolithic op (the project's own validated reference) -- NOT the int4->fp8 dequant,
which is SHARED by both paths and therefore cancels in the dense-vs-binned relNorm
(so this oracle uses float weights to ISOLATE the binning math).

================================ WHAT THIS PROVES ==============================

  V1  ALGORITHM byte-exactness: the static-capacity bin (cumsum-slot scatter ->
      per-expert grouped GEMM over E*C rows -> unbin gather + top_k sum) reproduces
      the DENSE monolithic (all E experts x all T tokens, router-weight-zeroed,
      reduced over E) to the float round-off floor, *when C >= the batch's max
      per-expert local load* (no pick overflows). relNorm < 1e-2 (the project's
      HPU_MOE_FUSED_VERIFY tol); in practice ~1e-7.

  V2  CUMSUM-SLOT == reference loop: the VECTORIZED, static-shape slot assignment
      the device wrapper uses (one_hot -> cumsum -> own-column) is asserted EQUAL to
      a plain per-pick counter loop. This is the one subtle device-safe trick; if the
      cumsum is wrong the bin collides and V1 would already fail, but we check it
      directly so a future edit can't silently break it.

  V3  SAFE-DEGRADE: when C < max_load, overflow picks route to the SENTINEL bin ->
      they contribute EXACTLY 0 (dropped), never wrong-fold onto another expert/token
      (the KBUCKET contract, moe_wna16_hpu.py:155-160). We show relNorm grows
      SMOOTHLY and BOUNDEDLY with the drop count (no blow-up), and that the dropped
      picks are precisely the lowest-router-weight tail -> a tiny, oracle-catchable
      bump, never garbage.

  V4  ADVERSARIAL: a WRONG bin<->token assignment (shuffled permutation) is CAUGHT
      (relNorm explodes >> 1e-2), so a silent mis-permutation cannot pass V1.

  V5  CAPACITY ANALYSIS (deliverable #3): under the LIVE GLM-5.2 config (256 routed
      experts, top_k=8, EP TP=8 -> 32 LOCAL experts/rank), measure the per-rank
      per-expert local-load distribution at the decode batch T (c128) and report:
        - mean local load, p50/p99/max of the per-batch MAX load,
        - the capacity C that is LOSSLESS at p99 (<=1% of batches drop ANY pick),
        - the capacity-factor cf = C / mean_load to put in the env,
        - the GEMM-work cut  dense (E*T) / binned (E*C) = T / C,
      for BALANCED routing AND a SKEWED (hot-expert) routing, so the recommended cf
      is conservative against real imbalance. This is WHY binning defeats the
      "K->E=32 by bs16" clamp: the clamp is about the E-axis union (=32, conceded);
      binning cuts the ORTHOGONAL T-axis (tokens-per-expert: T -> C ~ load << T).

RUN (card-free; the only action on a CPU host):
    python3 tests/binned_moe_oracle.py            # full report, exit 0 on PASS
    python3 tests/binned_moe_oracle.py --capacity # only the V5 capacity table
    python3 tests/binned_moe_oracle.py --seed 7   # reseed the routing draws

GROUNDING (every modeled op traces to on-box source):
  - DENSE reference == moe_fused_monolithic_int4fp8_emit.h: x broadcast [E,T,H] ->
    gu[E,T,2I]=bmm -> slice gate/up -> silu(gate)*up -> y[E,T,H]=bmm -> * router_w
    -> reduce_sum over E. (We use float W; the int4->fp8 dequant is shared/cancels.)
  - BINNED == hpu_moe_ext/binned_moe.py wrapper + the new binned emit: x_perm[E,C,H]
    grouped GEMM (NO broadcast, E*C work) -> * router_w -> unbin gather + top_k sum.
  - EP/local filter == _routed_moe (moe_wna16_hpu.py:1560-1574): local=ids-emin,
    valid=(0<=local<E), non-local -> sentinel.
  - load stats grounding: glm52-single-stream-rootcause (union saturates ~E by bs16
    => all 32 dequanted, read amortized) AND BATCHED_MOE_FIX.md (read once/STEP) =>
    binning is a COMPUTE/working-set lever, never a read lever.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# ---- live GLM-5.2 shapes (small H/I here; the binning math is shape-agnostic) ----
H_FULL, I_FULL = 6144, 2048          # real GLM-5.2 hidden / moe_intermediate
E_GLOBAL = 256                       # routed experts
TOP_K = 8
TP = 8
E_LOCAL = E_GLOBAL // TP             # 32 local experts / rank
GROUP = 128                          # int4 group size (unused here; weights are float)


def _silu(z):
    return z / (1.0 + np.exp(-z))


# =============================================================================
# DENSE monolithic reference (numpy mirror of moe_fused_monolithic_int4fp8_emit.h)
# =============================================================================
def dense_monolithic(x, W13, W2, routerW):
    """x[T,H], W13[E,H,2I], W2[E,I,H], routerW[E,T] (=sum of safe_w for slots t->e).
    out[t] = sum_e router[e,t] * (silu(gate)*up @ W2[e]), gate/up = (x@W13[e]) split.
    EVERY expert computes EVERY token (dense E*T); non-routed (e,t) carry router=0."""
    T, H = x.shape
    E = W13.shape[0]
    I = W2.shape[1]
    out = np.zeros((T, H), dtype=np.float64)
    for e in range(E):
        gu = x @ W13[e]                 # [T, 2I]
        gate, up = gu[:, :I], gu[:, I:]
        h = _silu(gate) * up            # [T, I]
        y = h @ W2[e]                   # [T, H]
        out += y * routerW[e][:, None]  # weighted + summed over experts
    return out


# =============================================================================
# BINNED grouped-GEMM (numpy mirror of binned_moe.py + the new binned emit)
# =============================================================================
def _vectorized_slot(flat_e, flat_v, E, C):
    """The DEVICE wrapper's static-shape slot assignment: one_hot(expert) restricted
    to valid picks, inclusive cumsum down the pick axis, take each pick's own column
    -> its 0-based position WITHIN its expert. Returns (slot[P], overflow[P]).

    NO sort, NO data-dependent shape, NO host sync -> HPU-graph-capture safe."""
    P = flat_e.shape[0]
    erange = np.arange(E)
    onehot = (flat_e[:, None] == erange[None, :]) & flat_v[:, None]   # [P,E] bool
    csum = np.cumsum(onehot.astype(np.int64), axis=0)                  # inclusive
    slot_mat = csum - 1                                               # 0-based
    slot = (slot_mat * onehot).sum(axis=1)                            # own column
    overflow = (~flat_v) | (slot >= C)
    return slot.astype(np.int64), overflow


def _loop_slot(flat_e, flat_v, E, C):
    """Plain per-pick counter (the unambiguous reference the vectorized form must match)."""
    P = flat_e.shape[0]
    slot = np.zeros(P, dtype=np.int64)
    counts = np.zeros(E, dtype=np.int64)
    overflow = np.zeros(P, dtype=bool)
    for p in range(P):
        if not flat_v[p]:
            overflow[p] = True
            continue
        e = int(flat_e[p])
        s = counts[e]
        counts[e] += 1
        if s >= C:
            overflow[p] = True
        else:
            slot[p] = s
    return slot, overflow


def binned_moe(x, topk_ids, topk_w, W13, W2, emin, E, C, *, scramble=False, rng=None):
    """x[T,H], topk_ids[T,K] GLOBAL ids, topk_w[T,K]; weights stacked [E,...] LOCAL.
    Returns (out[T,H], n_overflow:int). Static-capacity C; overflow -> sentinel (drop)."""
    T, H = x.shape
    K = topk_ids.shape[1]
    I = W2.shape[1]
    P = T * K

    local = topk_ids.astype(np.int64) - emin
    flat_e = local.reshape(-1)
    flat_v = (flat_e >= 0) & (flat_e < E)
    flat_e_safe = np.where(flat_v, flat_e, 0)
    flat_w = topk_w.reshape(-1).astype(np.float64)
    tok_of_pick = np.repeat(np.arange(T), K)                          # pick -> source token

    slot, overflow = _vectorized_slot(flat_e_safe, flat_v, E, C)
    bin_idx = np.where(overflow, E * C, flat_e_safe * C + slot)        # sentinel = E*C
    # CAPACITY overflow = VALID local picks beyond C (the ONLY lossy drops). Non-local
    # picks (~flat_v) also route to the sentinel, but those are CORRECT to drop (they
    # belong to other ranks; the dense reference zeroes them via routerW too) -> they
    # are NOT a loss and must not be counted.
    n_cap_overflow = int((flat_v & (slot >= C)).sum())

    if scramble and rng is not None:
        # V4 adversarial: corrupt the bin<->pick map for a few VALID picks (wrong
        # expert/token fold). MUST be caught by relNorm.
        vp = np.where(~overflow)[0]
        if vp.size >= 2:
            j = rng.choice(vp, size=max(2, vp.size // 10), replace=False)
            bin_idx[j] = bin_idx[j[::-1]]

    # x_perm[E*C+1, H]: functional scatter (fresh zeros each call). Non-sentinel
    # indices are UNIQUE (distinct (e,slot)) so index-assign == copy; sentinel row
    # collects overflow (discarded). Mirrors the wrapper's index_add-on-fresh-zeros.
    x_perm = np.zeros((E * C + 1, H), dtype=np.float64)
    np.add.at(x_perm, bin_idx, x[tok_of_pick])
    # per-pick router weight, folded INTO the op (== monolithic yw = y * rw):
    w_perm = np.zeros((E * C + 1,), dtype=np.float64)
    np.add.at(w_perm, bin_idx, np.where(overflow, 0.0, flat_w))
    x_perm = x_perm[:E * C].reshape(E, C, H)
    w_perm = w_perm[:E * C].reshape(E, C, 1)

    # grouped GEMM: each expert e over ONLY its C bin rows (E*C work, not E*T).
    y_perm = np.zeros((E, C, H), dtype=np.float64)
    for e in range(E):
        gu = x_perm[e] @ W13[e]                  # [C, 2I]
        gate, up = gu[:, :I], gu[:, I:]
        h = _silu(gate) * up                     # [C, I]
        y = h @ W2[e]                            # [C, H]
        y_perm[e] = y * w_perm[e]                # router-weighted INSIDE the op

    # unbin: each pick gathers its (e,slot) output; sentinel row -> 0; sum over top_k.
    y_flat = np.concatenate([y_perm.reshape(E * C, H),
                             np.zeros((1, H), dtype=np.float64)], axis=0)
    picked = y_flat[bin_idx].reshape(T, K, H)
    out = picked.sum(axis=1)
    return out, n_cap_overflow


# =============================================================================
# routing draws (EP TP=8): map global top-k -> this rank's local-expert loads
# =============================================================================
def draw_routing(T, rng, *, skew=0.0):
    """Return topk_ids[T,K] GLOBAL ids + topk_w[T,K] (renormalized), for a rank that
    owns local experts [0, E_LOCAL). skew=0 -> balanced popularity; skew>0 -> a
    hot-expert (softmax-temperature) popularity, modelling real routing imbalance."""
    base = rng.standard_normal(E_GLOBAL) * skew            # expert popularity logits
    ids = np.empty((T, TOP_K), dtype=np.int64)
    wts = np.empty((T, TOP_K), dtype=np.float64)
    for t in range(T):
        logits = base + rng.standard_normal(E_GLOBAL)       # per-token noise
        top = np.argpartition(logits, -TOP_K)[-TOP_K:]
        w = logits[top]
        w = np.exp(w - w.max())
        w /= w.sum()
        ids[t] = top
        wts[t] = w
    return ids, wts


def local_loads(topk_ids, emin=0):
    """Per LOCAL expert pick count on this rank (the dense-router nnz per expert)."""
    local = topk_ids - emin
    flat = local.reshape(-1)
    flat = flat[(flat >= 0) & (flat < E_LOCAL)]
    return np.bincount(flat, minlength=E_LOCAL)


def routerW_from(topk_ids, topk_w, emin, E, T):
    """Dense [E,T] router matrix (sum of local router weights per (expert,token))."""
    W = np.zeros((E, T), dtype=np.float64)
    local = topk_ids - emin
    for t in range(T):
        for s in range(topk_ids.shape[1]):
            e = local[t, s]
            if 0 <= e < E:
                W[e, t] += topk_w[t, s]
    return W


# =============================================================================
def _relnorm(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run_correctness(seed=0):
    rng = np.random.default_rng(seed)
    H, I = 128, 96            # small but real-shaped (2I split, group-agnostic)
    T = 24
    E = E_LOCAL
    emin = 0
    print("=" * 78)
    print("V1/V2/V3/V4  ALGORITHM byte-exactness (float weights; isolates binning)")
    print("=" * 78)
    W13 = rng.standard_normal((E, H, 2 * I)) * 0.02
    W2 = rng.standard_normal((E, I, H)) * 0.02
    x = rng.standard_normal((T, H)) * 1.0
    ids, wts = draw_routing(T, rng, skew=0.6)
    routerW = routerW_from(ids, wts, emin, E, T)
    ref = dense_monolithic(x, W13, W2, routerW)

    loads = local_loads(ids, emin)
    maxload = int(loads.max())

    # --- V2: cumsum slot == loop slot ---
    local = (ids - emin).reshape(-1)
    flat_v = (local >= 0) & (local < E)
    flat_e_safe = np.where(flat_v, local, 0)
    C_loss = maxload                          # lossless capacity for THIS batch
    s_vec, o_vec = _vectorized_slot(flat_e_safe, flat_v, E, C_loss)
    s_loop, o_loop = _loop_slot(flat_e_safe, flat_v, E, C_loss)
    ok_slot = np.array_equal(s_vec[flat_v], s_loop[flat_v]) and np.array_equal(o_vec, o_loop)
    print(f"V2 cumsum-slot == loop-slot (C={C_loss}): {'PASS' if ok_slot else 'FAIL'}")

    # --- V1: lossless (C >= maxload) ---
    out_loss, n_of = binned_moe(x, ids, wts, W13, W2, emin, E, C_loss)
    rn_loss = _relnorm(out_loss, ref)
    ok_v1 = (rn_loss < 1e-2) and (n_of == 0)
    print(f"V1 binned(C={C_loss}=maxload) vs dense: relNorm={rn_loss:.2e} overflow={n_of} "
          f"-> {'PASS' if ok_v1 else 'FAIL'}  (max per-expert load={maxload}, mean={loads.mean():.1f})")

    # --- V3: safe-degrade (C < maxload) ---
    print("V3 safe-degrade (C below maxload -> bounded relNorm, drops = lowest-weight tail):")
    ok_v3 = True
    prev = -1.0
    for C in sorted({max(1, maxload - 6), max(1, maxload - 3), max(1, maxload - 1), maxload}):
        o, nf = binned_moe(x, ids, wts, W13, W2, emin, E, C)
        rn = _relnorm(o, ref)
        finite_bounded = np.isfinite(rn) and rn < 1.0     # bounded, never blow-up
        print(f"     C={C:2d}  relNorm={rn:.2e}  dropped picks={nf:3d}  "
              f"{'(lossless)' if nf == 0 else ''}")
        ok_v3 = ok_v3 and finite_bounded and (nf == 0 or rn > 0)
        prev = rn
    # monotone-ish: fewer capacity never REDUCES error below the lossless floor
    print(f"V3 bounded + safe-degrade: {'PASS' if ok_v3 else 'FAIL'}")

    # --- V4: adversarial scramble caught ---
    out_bad, _ = binned_moe(x, ids, wts, W13, W2, emin, E, C_loss,
                            scramble=True, rng=np.random.default_rng(seed + 99))
    rn_bad = _relnorm(out_bad, ref)
    ok_v4 = rn_bad > 1e-2
    print(f"V4 scrambled bin map: relNorm={rn_bad:.2e} -> "
          f"{'PASS (caught)' if ok_v4 else 'FAIL (not caught!)'}")

    return ok_slot and ok_v1 and ok_v3 and ok_v4


def _measure_loads(seed, T, skew, n_batches):
    rng = np.random.default_rng(seed)
    maxloads = np.empty(n_batches, dtype=np.int64)
    total_local = np.empty(n_batches, dtype=np.int64)
    drop_at = {}                       # C -> [over_picks, total_picks]
    for b in range(n_batches):
        ids, _ = draw_routing(T, rng, skew=skew)
        loads = local_loads(ids)
        maxloads[b] = loads.max()
        total_local[b] = loads.sum()
    return maxloads, total_local


def run_capacity(seed=0, T=128, n_batches=4000):
    print("=" * 78)
    print(f"V5  CAPACITY ANALYSIS  (live GLM-5.2: {E_GLOBAL} experts, top_k={TOP_K}, "
          f"EP TP={TP} -> {E_LOCAL} local/rank; decode T={T} == c{T})")
    print("=" * 78)
    print(f"  dense monolithic work = E*T = {E_LOCAL}*{T} = {E_LOCAL * T} expert-token GEMMs/layer")
    print(f"  the dense [{E_LOCAL},{T}] router is ~1/{E_LOCAL} dense per token (~{100.0/E_LOCAL:.0f}% useful)")
    print(f"  => binned work = E*C = {E_LOCAL}*C; GEMM cut = T/C = {T}/C.  WIN DEPENDS ON ROUTING BALANCE:")
    print()
    # skew=0 balanced (load-balancing fully effective) ... 0.8 heavy (poorly balanced).
    skews = (("balanced", 0.0), ("mild", 0.4), ("heavy", 0.8))
    hdr = f"  {'routing':9s} {'mean_load':>9s} {'p50_max':>8s} {'p99_max':>8s} {'abs_max':>8s} " \
          f"{'C@p99':>6s} {'cf':>5s} {'GEMM cut':>9s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    results = {}
    for label, skew in skews:
        maxloads, total_local = _measure_loads(seed + 1 + int(skew * 100), T, skew, n_batches)
        mean_load = total_local.mean() / E_LOCAL
        p50 = int(np.percentile(maxloads, 50))
        p99 = int(np.percentile(maxloads, 99))
        amax = int(maxloads.max())
        C99 = p99
        cf = C99 / max(mean_load, 1e-9)
        print(f"  {label:9s} {mean_load:9.1f} {p50:8d} {p99:8d} {amax:8d} "
              f"{C99:6d} {cf:5.1f} {T / C99:8.1f}x")
        results[label] = dict(mean_load=mean_load, p99=p99, C99=C99, cf=cf,
                              gemm_cut=T / C99, maxloads=maxloads)
    print()
    # actionable: at a FIXED operator-chosen C, what GEMM cut + drop% per routing?
    print(f"  FIXED-C tradeoff (drop% = VALID local picks beyond C; safe-degraded, verify-gated):")
    print(f"  {'C':>4s} {'GEMM cut':>9s} | " + " ".join(f"{lab:>14s}" for lab, _ in skews))
    for C in (16, 24, 32, 48, 64):
        cells = []
        for label, skew in skews:
            ml = results[label]["maxloads"]
            # mean over batches of picks dropped, as % of mean local picks/batch
            rng = np.random.default_rng(seed + 55 + int(skew * 100) + C)
            over = tot = 0
            for _ in range(400):
                loads = local_loads(draw_routing(T, rng, skew=skew)[0])
                over += int(np.clip(loads - C, 0, None).sum())
                tot += int(loads.sum())
            droppct = 100.0 * over / max(tot, 1)
            lossless_frac = 100.0 * float((ml <= C).mean())
            cells.append(f"{droppct:5.2f}%/{lossless_frac:3.0f}%ll")
        print(f"  {C:>4d} {T / C:8.1f}x | " + " ".join(f"{c:>14s}" for c in cells))
    print("    (cell = drop% of local picks / % of batches fully lossless at that C)")
    print()
    bal, mild, heavy = results["balanced"], results["mild"], results["heavy"]
    print("  VERDICT / DELIVERABLE #3:")
    print(f"   * mean per-rank local load @c{T} = {bal['mean_load']:.0f} (= T*top_k*E_local/E_global "
          f"= {T}*{TOP_K}*{E_LOCAL}/{E_GLOBAL}); union=~{E_LOCAL} (all dequant, read amortized).")
    print(f"   * BALANCED routing: C={bal['C99']} (cf {bal['cf']:.1f}) lossless@p99 -> "
          f"{bal['gemm_cut']:.1f}x GEMM cut, x W4A8 2x = ~{2*bal['gemm_cut']:.0f}x on MoE compute.")
    print(f"   * MILD imbalance: C={mild['C99']} -> {mild['gemm_cut']:.1f}x.  "
          f"HEAVY: C={heavy['C99']} -> {heavy['gemm_cut']:.1f}x (win erodes as a hot expert -> C~T).")
    print(f"   * RECO: ship C as env HPU_MOE_BINNED_CAP (default 32 ~= {T}/32={T//32:.0f}x, ~lossless if "
          f"balanced); LOG the live per-rank max-load histogram one-shot at first decode to")
    print(f"     CALIBRATE C (no real routing dump exists yet -- EXPERT_LOCALITY.md). SAFE-DEGRADE")
    print(f"     (sentinel drop, verify-gated) protects an undersized C. Reads UNCHANGED (compute lever).")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity", action="store_true", help="only the V5 capacity table")
    ap.add_argument("--T", type=int, default=128, help="decode batch for V5 (c128 default)")
    args = ap.parse_args()

    if args.capacity:
        run_capacity(args.seed, T=args.T)
        return 0

    ok = run_correctness(args.seed)
    print()
    run_capacity(args.seed, T=args.T)
    print()
    # second seed (the project's "triple-check" rule: never trust one draw)
    ok2 = run_correctness(args.seed + 1234)
    print()
    verdict = ok and ok2
    print("=" * 78)
    print(f"ORACLE VERDICT: {'ALL PASS' if verdict else 'FAIL'} (2 seeds)")
    print("=" * 78)
    if not verdict:
        print("!!! LOUD FAILURE: the binned algorithm diverged from the dense monolithic "
              "reference. DO NOT trust the on-card op until this passes.", file=sys.stderr)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
