#!/usr/bin/env python3
"""Diff the per-decode-step boundary dumps from decode_step_dump.py across two runs
(lazy=1 uncaptured vs lazy=0) and name the first diverging stage.

Usage: python3 compare_decode_dumps.py <dir_lazy> <dir_eager> [rank]

Reads <kind>.rank<r>.step<k>.pt for kind in (layer0, mla0, mqa0). For every step and
tensor key present in both runs prints max-abs-diff + cosine; flags the FIRST
(step, kind, key) whose cosine drops below 0.999 — with the rule of thumb:
  * an INPUT key diverging first  -> the divergence began UPSTREAM (earlier stage/step)
  * an OUTPUT key diverging while the same step's inputs match -> THAT stage is the
    broken op under lazy.
Stops the detailed log after divergence + 2 steps (the rest is cascade noise).
"""
import sys
import torch

KINDS = {
    "layer0": ["positions", "hidden_in", "residual_in", "hidden_out", "residual_out"],
    "mla0": ["q", "k_c_normed", "k_pe", "attn_out"],
    "mqa0": ["ql_nope", "q_pe", "out"],
}
ORDER = ["layer0:positions", "layer0:hidden_in", "layer0:residual_in",
         "mla0:q", "mla0:k_c_normed", "mla0:k_pe",
         "mqa0:ql_nope", "mqa0:q_pe", "mqa0:out",
         "mla0:attn_out", "layer0:hidden_out", "layer0:residual_out"]


def cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() == 0 or a.numel() != b.numel():
        return float("nan")
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else (1.0 if a.norm() == b.norm() else 0.0)


def main():
    da, db, rank = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "0")
    first_div = None
    for step in range(64):
        loaded = {}
        any_found = False
        for kind in KINDS:
            try:
                ta = torch.load(f"{da}/{kind}.rank{rank}.step{step}.pt", weights_only=False)
                tb = torch.load(f"{db}/{kind}.rank{rank}.step{step}.pt", weights_only=False)
                loaded[kind] = (ta, tb)
                any_found = True
            except FileNotFoundError:
                continue
        if not any_found:
            break
        print(f"--- step {step} ---")
        for label in ORDER:
            kind, key = label.split(":")
            if kind not in loaded:
                continue
            ta, tb = loaded[kind]
            va, vb = ta.get(key), tb.get(key)
            if not (torch.is_tensor(va) and torch.is_tensor(vb)):
                continue
            if va.shape != vb.shape:
                print(f"  {label:24s} SHAPE MISMATCH {tuple(va.shape)} vs {tuple(vb.shape)}")
                continue
            c = cos(va, vb)
            mad = float((va.float() - vb.float()).abs().max())
            mark = ""
            if c < 0.999 and first_div is None:
                first_div = (step, label, c, mad)
                mark = "   <<< FIRST DIVERGENCE"
            print(f"  {label:24s} cos={c:.6f} maxabs={mad:.6g}{mark}")
        if first_div and step > first_div[0] + 2:
            print("(cascade truncated)")
            break
    print()
    if first_div:
        s, label, c, mad = first_div
        print(f"VERDICT: first divergence at step {s}, stage {label} (cos={c:.6f}, maxabs={mad:.4g})")
        print("If this is an OUTPUT whose same-step inputs match -> that stage's op chain is")
        print("the lazy-mode bug. If it's an input -> walk upstream (prior stage/step).")
    else:
        print("VERDICT: no divergence beyond bf16 noise found in dumped steps/tensors.")


if __name__ == "__main__":
    main()
