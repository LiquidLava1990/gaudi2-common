#!/usr/bin/env python3
"""Judge the stock MoE op's on-device output against a CPU fp32 GPTQ reference,
per mode, from the moeop<L> dumps (decode_step_dump.py section 5).

For each dump dir given: load moeop<L>.rank0.step<k>.pt (x, tid_local, tw, out),
dequant the routed experts' checkpoint weights on CPU (int4 GPTQ sym: W = (q-8)*scale,
group 128, +1 qzeros convention — element-wise verified in the original bring-up),
compute ref = sum_k tw_k * down(silu(gate(x)) * up(x)), and report
relNorm(out_device, ref) + maxabs. The MODE whose op output disagrees with its OWN
inputs' fp32 reference (beyond bf16-GEMM noise ~1e-2) is the numerically-broken one.

Usage: moe_op_cpu_judge.py <model_dir> <layer> <emin> <dump_dir> [<dump_dir2> ...]
       (emin = rank0's first global expert id; 0 for rank0 of EP8.)
"""
import json
import os
import sys

import torch
from safetensors import safe_open

GS = 128  # group_size from quantization_config


def load_tensor(model_dir, index, name):
    shard = index["weight_map"][name]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def dequant_gptq(qweight, scales, qzeros=None):
    """int4 GPTQ sym -> fp32 [in, out]. W[i,o] = (nib(i,o) - 8) * scales[i//GS, o]."""
    iw8, out = qweight.shape
    q = qweight.view(torch.int32)
    nibs = torch.empty(iw8 * 8, out, dtype=torch.float32)
    for j in range(8):
        nibs[j::8] = ((q >> (4 * j)) & 0xF).float()
    scale = scales.float().repeat_interleave(GS, dim=0)[: iw8 * 8]
    return (nibs - 8.0) * scale


def expert_forward(model_dir, index, layer, gid, x32):
    pre = f"model.layers.{layer}.mlp.experts.{gid}."
    wg = dequant_gptq(load_tensor(model_dir, index, pre + "gate_proj.qweight"),
                      load_tensor(model_dir, index, pre + "gate_proj.scales"))
    wu = dequant_gptq(load_tensor(model_dir, index, pre + "up_proj.qweight"),
                      load_tensor(model_dir, index, pre + "up_proj.scales"))
    wd = dequant_gptq(load_tensor(model_dir, index, pre + "down_proj.qweight"),
                      load_tensor(model_dir, index, pre + "down_proj.scales"))
    g = x32 @ wg
    u = x32 @ wu
    h = torch.nn.functional.silu(g) * u
    return h @ wd


def judge(model_dir, index, layer, emin, dump_dir, rank="0"):
    print(f"== {dump_dir} ==")
    for step in range(3):
        p = f"{dump_dir}/moeop{layer}.rank{rank}.step{step}.pt"
        if not os.path.exists(p):
            continue
        d = torch.load(p, weights_only=False)
        x32 = d["x"].float().view(1, -1)
        tid = d["tid_local"].view(-1).tolist()
        tw = d["tw"].float().view(-1)
        out_dev = d["out"].float().view(1, -1)
        ref = torch.zeros_like(x32)
        used = []
        skipped = []
        for k, lid in enumerate(tid):
            w = float(tw[k])
            if w == 0.0:
                continue
            if lid < 0 or lid > 4095:  # padding slot (e.g. bf16 -1.0 viewed as int) with stale weight
                skipped.append((lid, round(w, 4)))
                continue
            used.append((lid, round(w, 4)))
            e = expert_forward(model_dir, index, layer, int(lid) + emin, x32)
            print(f"   dbg k={k} lid={lid} gid={int(lid)+emin} w={w:.4f} e_norm={float(e.norm()):.6f}")
            ref += w * e
        if skipped:
            print(f" step {step}: SKIPPED pad slots (lid,w)={skipped}")
        diff = out_dev - ref
        rel = float(diff.norm() / (ref.norm() + 1e-9))
        mad = float(diff.abs().max())
        print(f" step {step} rank {rank}: x_norm={float(x32.norm()):.4f} experts(local,w)={used}")
        print(f"          out_norm={float(out_dev.norm()):.4f} ref_norm={float(ref.norm()):.4f} "
              f"relNorm={rel:.6g} maxabs={mad:.6g}")


def main():
    model_dir, layer, emin = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    for dd in sys.argv[4:]:
        for rank in range(8):
            judge(model_dir, index, layer, emin + rank * 32, dd, rank=str(rank))


if __name__ == "__main__":
    main()
