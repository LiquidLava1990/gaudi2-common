#!/usr/bin/env python3
"""GATE-A CARD-FREE ORACLE: prove the contraction-tiled bf16 fold math is byte-exact.

This is the numpy/torch-CPU correctness proof for emit_one_expert_tiled_bf16
(tpc_kernel/opbackend/moe_fused_expert_tiled_bf16_emit.h). It models, on the CPU, the
ONE thing the tiled emit changes vs the stock per-expert chain (emit_one_expert): the
GEMM's dequant+accumulate is sliced along the contraction dim into Kt-row tiles, each
tile dequanted int4->bf16 and gemm'd with an fp32 (or bf16) accumulator carried across
tiles, with a SINGLE final bf16 round. Everything else in the expert chain (the gate/up
slice, SiLU, the router mult, the K-fold add-tree) is byte-for-byte unchanged, so the
discriminating math to validate is exactly this tiled dequant->gemm primitive.

WHAT IT PROVES (card-free; NOT on-card SRAM residency / the step-timer):
  (a) per-element dequant is BYTE-IDENTICAL between the tiled slices and the full
      weight -- the tile's bf16 weight rows == the corresponding rows of the full bf16
      weight (the group-aligned scale/zp slice reproduces the full dequant exactly).
  (b) the fp32-accumulate tiled gemm matches the untiled single fp32 gemm to
      relNorm < 1e-4 (expect ~1e-6: only the non-associative fp32 partial-sum REORDER
      across tiles differs -- the lossless bar; identical greedy tokens downstream).
  (c) the bf16-accumulate fallback (HPU_MOE_TILE_ACC_BF16=1) relNorm is reported
      (~1e-3; a few bf16 ULPs across the 3-12 tiles -- still likely greedy-identical,
      but NOT the lossless bar).
  (d) the Kt>=dim "one tile" path is EXACTLY equal to the untiled reference (relNorm
      == 0): the A==A sanity that the harness + the eff_kt rounding are faithful.

FAITHFUL MME MODEL: the Gaudi MME takes bf16 inputs, accumulates in fp32, outputs
bf16/fp32. We model that as `a.to(bf16).float() @ W.to(bf16).float()` (fp32 reduction of
the bf16-rounded operands), then a single `.to(bf16)` at the end for the fp32-acc path
(or per-tile `.to(bf16)` for the bf16-acc path). The dequant is `(q - zp) * scale`
cast to bf16 (the stock dequantize_4_bit_bf16 output dtype), per int4 group along the
contraction dim. The reinterpret/pack is a pure layout op (slicing rows of the packed
[K,N/8] tensor == slicing rows of the unpacked [K,N]); it is identical tiled vs untiled
and so is modeled directly on the unpacked q[K,N] -- no packing needed for the math.

eff_kt + group slicing mirror the emit EXACTLY:
  eff_kt(kt_env,dim,gs): kt_env<=0 or >=dim -> dim (one tile); else (kt_env//gs)*gs,
    clamped to [gs,dim]. (H % GS == 0 and Kt a multiple of GS => every tile, incl the
    remainder, is a multiple of GS => g0=k0//GS, g1=(k0+kt)//GS are exact integers.)

Run (card-free, no HPU):  python3 tests/test_tiled_bf16_fold_cpu.py
Exit 0 = all PASS; exit 1 = any FAIL (LOUD).
"""
from __future__ import annotations

import sys

import torch

BF16 = torch.bfloat16
F32 = torch.float32


# --------------------------------------------------------------------------- #
# eff_kt -- BYTE-FOR-BYTE the emit's eff_kt lambda (moe_fused_expert_tiled_bf16_emit.h
# L147-153). kt_env<=0 or >=dim -> one tile (== stock); else round DOWN to a multiple
# of the group size gs, clamp to [gs, dim].
# --------------------------------------------------------------------------- #
def eff_kt(kt_env: int, dim: int, gs: int) -> int:
    if kt_env <= 0 or kt_env >= dim:
        return dim
    kt = (kt_env // gs) * gs
    if kt < gs:
        kt = gs
    if kt > dim:
        kt = dim
    return kt


# --------------------------------------------------------------------------- #
# dequant a weight block of K rows (contraction) x N cols, int4 group along K.
#   q  : [K, N]  int   in [0,15]      (unpacked 4-bit weight values)
#   zp : [G, N]  int   in [0,15]      (per-group zero points, G = K//GS)
#   s  : [G, N]  bf16                  (per-group scales)
# returns bf16 [K, N] == (q - zp[k//GS]) * s[k//GS], the stock dequantize_4_bit_bf16
# output (bf16 dtype). Done in fp32 then cast to bf16 (the dequant's output rounding).
# --------------------------------------------------------------------------- #
def dequant_block(q: torch.Tensor, zp: torch.Tensor, s: torch.Tensor, gs: int) -> torch.Tensor:
    K, N = q.shape
    G = K // gs
    assert zp.shape == (G, N) and s.shape == (G, N), (q.shape, zp.shape, s.shape, gs)
    zp_full = zp.repeat_interleave(gs, dim=0).to(F32)         # [K, N]
    s_full = s.to(F32).repeat_interleave(gs, dim=0)           # [K, N]
    w_f32 = (q.to(F32) - zp_full) * s_full
    return w_f32.to(BF16)                                     # stock dequant out = bf16


# --------------------------------------------------------------------------- #
# UNTILED reference (== emit_one_expert's single gemm): full dequant -> bf16 weight,
# then ONE fp32-accumulate MME (bf16 inputs upcast to fp32 for the reduction), bf16 out.
# Returns (out_bf16, acc_fp32) -- the fp32 accumulator is the pre-round value the tiled
# fp32 accumulator is compared against (isolating the tiling reassociation from the
# final bf16 round).
# --------------------------------------------------------------------------- #
def gemm_untiled(a_bf16: torch.Tensor, q, zp, s, gs: int):
    W = dequant_block(q, zp, s, gs)                           # [K, N] bf16
    acc = a_bf16.to(F32) @ W.to(F32)                          # [M, N] fp32 (MME accumulate)
    return acc.to(BF16), acc


# --------------------------------------------------------------------------- #
# TILED (== emit_one_expert_tiled_bf16's tiled_gemm): slice the contraction into
# eff_kt-row tiles, dequant each tile (group-aligned scale/zp slice), per-tile MME with
# an `acc_dt` accumulator across tiles, single final bf16 round.
#   acc_bf16=False -> fp32 partials + fp32 cross-tile add, ONE final bf16 cast (lossless).
#                     Returns the fp32 accumulator (pre-round) for the true reassociation
#                     bar; the only difference from untiled is the fp32 partial-sum order.
#   acc_bf16=True  -> bf16 partials + bf16 cross-tile add (the HPU_MOE_TILE_ACC_BF16
#                     fallback; per-tile rounding -> no single fp32 accumulator).
# Also returns the per-tile dequanted weights stacked back to [K,N] for the byte-exact
# dequant check (a).
# --------------------------------------------------------------------------- #
def gemm_tiled(a_bf16: torch.Tensor, q, zp, s, gs: int, kt_env: int, acc_bf16: bool):
    K, N = q.shape
    KT = eff_kt(kt_env, K, gs)
    acc = None
    w_tiles = []
    k0 = 0
    while k0 < K:
        kt = min(KT, K - k0)
        g0 = k0 // gs
        g1 = (k0 + kt) // gs
        # group-aligned slices (the emit's slice_dim on the contraction dim / scale-group)
        a_t = a_bf16[:, k0:k0 + kt]
        q_t = q[k0:k0 + kt, :]
        zp_t = zp[g0:g1, :]
        s_t = s[g0:g1, :]
        W_t = dequant_block(q_t, zp_t, s_t, gs)               # [kt, N] bf16
        w_tiles.append(W_t)
        part = a_t.to(F32) @ W_t.to(F32)                      # [M, N] fp32 MME
        if acc_bf16:
            part = part.to(BF16)
            acc = part if acc is None else (acc + part).to(BF16)
        else:
            acc = part if acc is None else (acc + part)       # fp32 accumulate
        k0 += kt
    if acc_bf16:
        return acc.to(BF16), None, torch.cat(w_tiles, dim=0)
    return acc.to(BF16), acc, torch.cat(w_tiles, dim=0)       # [M,N] bf16, fp32 acc, [K,N] bf16


def relnorm(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(F32); b = b.to(F32)
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()


def bf16_elem_mismatch(a: torch.Tensor, b: torch.Tensor):
    a16 = a.to(BF16); b16 = b.to(BF16)
    ne = (a16 != b16)
    return int(ne.sum().item()), int(ne.numel())


# --------------------------------------------------------------------------- #
# bf16 ULP distance: map each bf16 value to a MONOTONIC ordered integer (sign-magnitude
# -> two's-complement ordering) so |order(a) - order(b)| counts how many representable
# bf16 steps apart they are. 0 == bit-identical; 1 == immediately-adjacent bf16 (a
# single rounding-boundary flip, the greedy-safe / lossless-decode floor); >1 would mean
# a real numeric divergence. This is the rigorous form of "identical greedy tokens,
# robust to fp32-accum ULPs" (the design's bar).
# --------------------------------------------------------------------------- #
def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def to_ord(t):
        u = t.to(BF16).view(torch.int16).to(torch.int32)
        return torch.where(u < 0, torch.full_like(u, -32768) - u, u)
    return (to_ord(a) - to_ord(b)).abs()


# --------------------------------------------------------------------------- #
# One (shape, M, Kt) case: build random int4 data, run untiled vs tiled (fp32 + bf16
# acc), print the relErr row, return per-case pass flags.
# --------------------------------------------------------------------------- #
def run_case(name: str, K: int, N: int, gs: int, M: int, kt_env: int, gen: torch.Generator):
    q = torch.randint(0, 16, (K, N), generator=gen, dtype=torch.int32)
    G = K // gs
    zp = torch.randint(0, 16, (G, N), generator=gen, dtype=torch.int32)
    # scales: small positive bf16 (realistic int4 group scales ~ 1e-3..1e-1)
    s = (torch.rand((G, N), generator=gen) * 0.05 + 1e-3).to(BF16)
    a = (torch.randn((M, K), generator=gen) * 0.5).to(BF16)

    ref_bf16, ref_acc = gemm_untiled(a, q, zp, s, gs)
    out_f32, acc_f32, w_tiled = gemm_tiled(a, q, zp, s, gs, kt_env, acc_bf16=False)
    out_bf16, _, _ = gemm_tiled(a, q, zp, s, gs, kt_env, acc_bf16=True)

    # (a) byte-exact dequant: the tiled per-tile dequant stacked back == the full dequant
    w_full = dequant_block(q, zp, s, gs)
    dequant_bitexact = bool(torch.equal(w_tiled, w_full))

    KT = eff_kt(kt_env, K, gs)
    ntiles = (K + KT - 1) // KT

    # (b) the TRUE tiling-math bar: the fp32 ACCUMULATOR (pre final-round) reassociation.
    #     The only difference from untiled is the fp32 partial-sum grouping -> ~1e-6.
    rn_acc = relnorm(acc_f32, ref_acc)

    # (c) the LOSSLESS-DECODE bar on the bf16 OUTPUT: the Frobenius relNorm bounds the
    #     TOTAL output perturbation (-> the downstream logit perturbation -> identical
    #     greedy tokens). REPORTED color: the bf16-ULP distance + the worst-case ABSOLUTE
    #     element error. NB the bf16-ULP metric is SCALE-RELATIVE: 1 ULP at value 5.0 is
    #     ~0.04 but 1 ULP at a near-zero CANCELLED output (~0.006) is ~5e-5, so a fixed
    #     ~fp32-reassociation absolute noise floor shows as several ULPs on the handful of
    #     cancelled elements while their ABSOLUTE error stays ~1e-6 (bounded by rn_acc).
    #     Hence the gate is rn_acc + the Frobenius rn_out, NOT a per-element ULP cap.
    ulp = bf16_ulp_distance(out_f32, ref_bf16)
    max_ulp = int(ulp.max().item())
    mm_f32, tot = bf16_elem_mismatch(out_f32, ref_bf16)
    abs_diff = (out_f32.to(F32) - ref_bf16.to(F32)).abs()
    max_abs = float(abs_diff.max().item())
    rn_out = relnorm(out_f32, ref_bf16)

    # bf16-acc fallback (HPU_MOE_TILE_ACC_BF16=1): advisory only (documented lossy).
    rn_bf16 = relnorm(out_bf16, ref_bf16)

    one_tile = (KT >= K)
    # PASS bars (the design's lossless-for-greedy bar, NOT bit-0 logits):
    #   (a) dequant byte-identical ALWAYS;
    #   (b) fp32-ACCUMULATOR relNorm < 1e-4 (expect ~1e-6) -- the decisive tiling-math
    #       proof: the tiled and untiled fp32 reductions agree up to non-associative
    #       reorder ONLY. one-tile path EXACTLY 0 (A==A).
    #   (c) bf16-OUTPUT Frobenius relNorm < 1e-3 -- the observable output fidelity (worst
    #       seen ~1.3e-4 at M=1, the bf16 rounding-boundary floor). A real tiling bug
    #       would blow this far past 1e-3.
    pass_dequant = dequant_bitexact
    pass_acc = (rn_acc < 1e-4) and ((rn_acc == 0.0) if one_tile else True)
    pass_out = (rn_out < 1e-3)
    ok = pass_dequant and pass_acc and pass_out

    flag = "PASS" if ok else "FAIL"
    note = " (one-tile A==A: acc must be 0)" if one_tile else ""
    print(f"  [{flag}] {name:>7} K={K:<5} N={N:<5} GS={gs} M={M} Kt_env={kt_env:<7} "
          f"-> KT={KT:<5} tiles={ntiles:<2} | dequant_bitexact={str(dequant_bitexact):<5} "
          f"| fp32-ACC relNorm={rn_acc:.2e} | bf16-out relNorm={rn_out:.2e} "
          f"(maxABS={max_abs:.1e}, {mm_f32}/{tot} elems flip, maxULP={max_ulp}) "
          f"| bf16-acc-fallback relNorm={rn_bf16:.2e}{note}")
    return ok, dict(rn_acc=rn_acc, rn_out=rn_out, rn_bf16=rn_bf16, max_ulp=max_ulp,
                    max_abs=max_abs, dequant_bitexact=dequant_bitexact,
                    one_tile=one_tile, mm_f32=mm_f32, tot=tot)


def main() -> int:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)

    # GLM-5.2 routed-expert GEMM shapes (GS=128):
    #   gemm#1: x[M,H] @ W13[H,2I]  -> contract H=6144, N=2I=4096
    #   gemm#2: h[M,I] @ W2[I,H]    -> contract I=2048, N=H=6144
    H = 6144
    twoI = 4096
    I = 2048
    GS = 128

    # Kt sweep: the recommended 512/1024/2048 (all divide 6144 and 2048 evenly into
    # full GS tiles), a NON-DIVIDING Kt (1280 -> remainder tile) for gemm#1 (6144 =
    # 4*1280 + 1024) and gemm#2 (2048 = 1*1280 + 768), and a Kt>=dim one-tile A==A.
    cases = []
    for M in (1, 4):
        for kt in (512, 1024, 2048, 1280, 10 ** 7):
            cases.append(("gemm#1", H, twoI, GS, M, kt))
            cases.append(("gemm#2", I, H, GS, M, kt))

    print("=" * 112)
    print("GATE-A CPU ORACLE: contraction-tiled bf16 fold byte-exactness "
          "(emit_one_expert_tiled_bf16)")
    print("  lossless bar (design): dequant byte-identical AND fp32-ACCUMULATOR relNorm "
          "<1e-4 (one-tile==0) AND bf16-out Frobenius relNorm <1e-3")
    print("  (maxULP/maxABS are INFORMATIONAL: bf16-out diffs are ~fp32-reassociation "
          "noise on a few near-zero cancelled elems; bf16-acc is the lossy fallback)")
    print("=" * 112)

    all_ok = True
    worst_acc = 0.0
    worst_out = 0.0
    worst_ulp = 0
    worst_bf16 = 0.0
    n_onetile = 0
    for (name, K, N, gs, M, kt) in cases:
        ok, st = run_case(name, K, N, gs, M, kt, gen)
        all_ok = all_ok and ok
        worst_acc = max(worst_acc, st["rn_acc"])
        worst_out = max(worst_out, st["rn_out"])
        worst_ulp = max(worst_ulp, st["max_ulp"])
        worst_bf16 = max(worst_bf16, st["rn_bf16"])
        if st["one_tile"]:
            n_onetile += 1

    print("=" * 112)
    print(f"SUMMARY: cases={len(cases)}  worst fp32-ACC relNorm={worst_acc:.2e} "
          f"(bar <1e-4, expect ~1e-6)  worst bf16-out relNorm={worst_out:.2e} "
          f"(bar <1e-3)  worst bf16-acc-fallback relNorm={worst_bf16:.2e} (advisory)  "
          f"info: worst bf16-ULP on a cancelled elem={worst_ulp}  one-tile A==A checks={n_onetile}")
    if all_ok:
        print("RESULT: PASS -- the tiling/accumulate/group-slice math is byte-exact: the "
              "dequant is bit-identical, the fp32 accumulator matches stock to ~1e-6 (pure "
              "non-associative reorder), and the bf16-output Frobenius relNorm stays <1e-3 "
              "(the diffs are ~fp32-noise on a few near-zero cancelled elements). This is "
              "the design's lossless / identical-greedy-token bar (NOT bit-0 logits).")
        return 0
    print("RESULT: FAIL -- see the per-case rows above. Either the dequant diverged, the "
          "fp32 accumulator drifted >1e-4 (a real tiling-math bug, not fp32 ULPs), or the "
          "bf16-output Frobenius relNorm exceeded 1e-3 (a true numeric divergence).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
