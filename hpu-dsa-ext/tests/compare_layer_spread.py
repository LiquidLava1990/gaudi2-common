#!/usr/bin/env python3
"""Walk the layer-spread decode dumps depth-first: for each step, for each dumped
layer (ascending), diff hidden_in/hidden_out between two runs. The first layer whose
OUTPUT diverges while its INPUT still matches localizes the lazy-mode bug to that
layer's stage class (dense MLP 0-2 vs MoE >=3 vs attention).

Usage: python3 compare_layer_spread.py <dir_a> <dir_b> [rank] [cos_threshold]
"""
import glob
import os
import re
import sys
import torch


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() == 0 or a.numel() != b.numel():
        return float("nan")
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else (1.0 if a.norm() == b.norm() else 0.0)


def main():
    da, db = sys.argv[1], sys.argv[2]
    rank = sys.argv[3] if len(sys.argv) > 3 else "0"
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.999
    layers = sorted({int(m.group(1)) for f in glob.glob(f"{da}/layer*.rank{rank}.step0.pt")
                     if (m := re.match(r".*layer(\d+)\.rank", f))})
    print(f"layers dumped: {layers}")
    first = None
    for step in range(64):
        row = []
        found = False
        for L in layers:
            fa = f"{da}/layer{L}.rank{rank}.step{step}.pt"
            fb = f"{db}/layer{L}.rank{rank}.step{step}.pt"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            found = True
            ta = torch.load(fa, weights_only=False)
            tb = torch.load(fb, weights_only=False)
            cin = cos(ta["hidden_in"], tb["hidden_in"])
            cout = cos(ta["hidden_out"], tb["hidden_out"])
            crin = cos(ta["residual_in"], tb["residual_in"]) if torch.is_tensor(ta.get("residual_in")) else float("nan")
            crout = cos(ta["residual_out"], tb["residual_out"]) if torch.is_tensor(ta.get("residual_out")) else float("nan")
            row.append(f"L{L}: in={cin:.5f}/{crin:.5f} out={cout:.5f}/{crout:.5f}")
            bad_out = (cout < thr) or (crout == crout and crout < thr)
            ok_in = (cin >= thr) and (crin != crin or crin >= thr)
            if first is None and bad_out and ok_in:
                first = (step, L, cout, crout)
        if not found:
            break
        print(f"step {step:2d}  " + "   ".join(row))
        if first and step > first[0] + 1:
            print("(truncated after divergence)")
            break
    print()
    if first:
        s, L, co, cr = first
        print(f"VERDICT: first in-clean/out-divergent layer = L{L} at step {s} "
              f"(hidden_out cos={co:.5f}, residual_out cos={cr:.5f})")
        print("L0-2 = dense MLP; L>=3 = MoE (K=1 monolithic dequant-all path under test).")
    else:
        print("VERDICT: no layer showed clean-in/divergent-out beyond threshold; "
              "divergence may enter via residual accumulation across MANY layers "
              "(every layer slightly off) — compare per-layer DELTA growth instead.")


if __name__ == "__main__":
    main()
