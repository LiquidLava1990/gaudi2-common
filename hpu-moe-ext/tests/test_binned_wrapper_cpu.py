#!/usr/bin/env python3
"""test_binned_wrapper_cpu.py -- CARD-FREE cross-check of the SHIPPING torch wrapper
(hpu_moe_ext/binned_moe.py: bin_scatter + reference_group_gemm + unbin) against the
INDEPENDENT numpy dense-monolithic reference (tests/binned_moe_oracle.py). This proves
the actual torch indexing the device path uses matches the validated algorithm -- the
'independent adversarial checker' the box's triple-check rule requires (numpy model vs
torch implementation, two code paths). Runs on torch-CPU; no .so, no HPU.

  python3 tests/test_binned_wrapper_cpu.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))               # tests/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # pkg root

import binned_moe_oracle as ora                                              # numpy reference
from hpu_moe_ext.binned_moe import bin_scatter, unbin, reference_group_gemm


def _relnorm(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run(seed=0):
    rng = np.random.default_rng(seed)
    H, I = 128, 96
    T = 24
    E = ora.E_LOCAL
    emin = 0
    chunk = 8

    W13 = rng.standard_normal((E, H, 2 * I)) * 0.02
    W2 = rng.standard_normal((E, I, H)) * 0.02
    x = rng.standard_normal((T, H))
    ids, wts = ora.draw_routing(T, rng, skew=0.6)
    routerW = ora.routerW_from(ids, wts, emin, E, T)
    ref = ora.dense_monolithic(x, W13, W2, routerW)                          # numpy dense
    maxload = int(ora.local_loads(ids, emin).max())

    # ---- torch wrapper path (the SHIPPING bin_scatter/unbin) ----
    xt = torch.tensor(x, dtype=torch.float64)
    idt = torch.tensor(ids, dtype=torch.int64)
    wt = torch.tensor(wts, dtype=torch.float64)
    W13t = torch.tensor(W13, dtype=torch.float64)
    W2t = torch.tensor(W2, dtype=torch.float64)

    results = []
    for C in (max(1, maxload - 3), maxload, maxload + 4):
        x_perm, w_perm, bin_idx, n_of = bin_scatter(xt, idt, wt, emin, E, C)
        # grouped GEMM via the float reference (the .so op's math; int4->fp8 cancels)
        ys = []
        g = 0
        while g < E:
            G = min(chunk, E - g)
            ys.append(reference_group_gemm(x_perm, W13t, W2t, w_perm, g, G, I))
            g += G
        y_perm = torch.cat(ys, dim=0)
        out = unbin(y_perm, bin_idx, T, idt.shape[1]).numpy()
        rn = _relnorm(out, ref)
        results.append((C, rn, int(n_of.item())))

    print(f"seed={seed}  T={T} E={E} maxload={maxload}")
    ok = True
    for C, rn, nof in results:
        lossless = (C >= maxload)
        tag = "lossless" if nof == 0 else f"{nof} dropped (safe)"
        passed = (rn < 1e-6) if lossless else (rn < 1.0 and np.isfinite(rn))
        ok = ok and passed
        print(f"  C={C:3d}  relNorm={rn:.2e}  overflow={nof:3d}  [{tag}]  "
              f"{'PASS' if passed else 'FAIL'}")
    # adversarial: a shuffled bin_idx must blow up
    x_perm, w_perm, bin_idx, _ = bin_scatter(xt, idt, wt, emin, E, maxload)
    ys = []
    g = 0
    while g < E:
        G = min(chunk, E - g)
        ys.append(reference_group_gemm(x_perm, W13t, W2t, w_perm, g, G, I))
        g += G
    y_perm = torch.cat(ys, dim=0)
    perm = torch.randperm(bin_idx.numel())
    bad = unbin(y_perm, bin_idx[perm], T, idt.shape[1]).numpy()
    rn_bad = _relnorm(bad, ref)
    caught = rn_bad > 1e-2
    ok = ok and caught
    print(f"  adversarial shuffled unbin: relNorm={rn_bad:.2e}  "
          f"{'PASS (caught)' if caught else 'FAIL'}")
    return ok


def main():
    ok = run(0) and run(1234)
    print("=" * 60)
    print(f"WRAPPER CPU CROSS-CHECK: {'ALL PASS' if ok else 'FAIL'}")
    if not ok:
        print("!!! LOUD: the torch wrapper diverged from the numpy oracle.", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
