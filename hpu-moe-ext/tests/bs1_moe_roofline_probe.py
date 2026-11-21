#!/usr/bin/env python3
"""bs1_moe_roofline_probe.py -- THE DECISIVE single-stream MoE roofline probe (RUN ON THE HPU BOX).

THE ONE QUESTION (GLM-5.2 40+ single-stream campaign):
  In the bs=1 decode forward the two `fused_kernel`s = ~51 ms (35.3 + 16.0 ms in traces).
  They move only ~1.7 GB/card yet take ~51 ms => ~33 GB/s => ~1.4% of Gaudi-2's ~2.45 TB/s
  peak HBM. Is that ~51 ms:
      (i)  COMPUTE / BANDWIDTH-bound  -> the device is genuinely busy reading+unpacking+MME-ing;
                                          a "Build-A" mega-kernel CANNOT beat the byte/flop floor.
      (ii) LATENCY / DISPATCH-bound   -> the device is mostly IDLE between tiny serial ops;
                                          a Build-A that collapses launches / overlaps WINS big.
  The ACHIEVED HBM BANDWIDTH at bs=1 is the number that decides the entire strategy, and it
  has never been measured. This probe measures it on the REAL production fused op.

WHAT THIS PROBE DOES (and does NOT do):
  * Runs the ACTUAL production fused path -- torch.ops.hpu_moe_ext.fused_expert (the per-expert
    int4->[SRAM dequant]->MME op the routed wrapper issues K times at T=1) and, when present,
    torch.ops.hpu_moe_ext.fused_moe_batched (the K-stacked single-recipe op). Same op, same
    GLM-5.2 shapes (H=6144, I=2048, 256 experts / 8 routed + 1 shared, GS=128) the serve runs.
  * Synthesizes int4 weights at those shapes (NO 414 GB load, NO vLLM serve, NO model). One
    card. K=8 routed experts (the bs=1 decode width) x replicated to 75 MoE layers' worth of
    per-token weight reads -- directly comparable to the measured ~51 ms two-fused-kernel figure.
  * Measures THREE things at bs=1:
      (a) DEVICE TIME per MoE layer  -- from torch.profiler HPU self_device_time_total (the
          time the AIP/TPC/MME engines were actually BUSY, excludes host gaps). device_busy/
          wall_time = the device DUTY CYCLE: low duty => idle => LATENCY-bound.
      (b) ACHIEVED HBM GB/s          -- from the Synapse device profiler (HABANA_PROFILE=1)
          synprof CSV's HBM/DRAM-traffic column (the authoritative per-node byte counter, the
          SAME mechanism profile_fusion.py asserts on). bytes_read / device_time = achieved BW.
          Fallback if no CSV column: known-bytes / measured-device-time (still decisive: ~33 vs
          ~2000 GB/s) PLUS device-time-vs-theoretical-floor.
      (c) MME util proxy             -- bs=1 GEMM FLOPs / device_time vs Gaudi-2 BF16 peak.
  * SWEEPS bs in {1, 8, 32, 128} to show the ROOFLINE CLIMB: at bs=1 the weight bytes dominate
    (1 token amortized over the whole expert weight); as bs grows the SAME weight read is
    amortized over more tokens so achieved BW (and MME util) should climb toward peak. This
    directly VISUALIZES why concurrency already won (~90 tok/s @ c50) and bs=1 didn't.
  * BYTE-EXACT-ish correctness gate: every measured op is first checked vs an INDEPENDENT eager
    convert_from_uint4 + MoE-math reference (relNorm < 1e-2, the bf16 norm tol). A probe that
    measures a WRONG kernel sends the project the wrong way -> LOUD nonzero-exit on mismatch,
    NO silent fallback. (This gates a big kernel build; per the DSA build-quality bar.)

DECISION TABLE (printed at the end; thresholds chosen from the physics):
  device_duty = device_busy_time / wall_time ;  achieved_BW (GB/s) ;  per-layer device ms
  ---------------------------------------------------------------------------------------------
  duty < ~0.5  (device idle > half the wall)                      => LATENCY/DISPATCH-bound
        => Build-A (collapse K launches, overlap host) WINS. The 51 ms is host/sync gaps.
  duty >= ~0.5 AND achieved_BW < ~30% of 2.45 TB/s               => still under-fed at bs=1:
        small-op granularity starves HBM => Build-A (one big read) HELPS reach the roofline.
  duty >= ~0.5 AND achieved_BW >= ~60% of 2.45 TB/s              => BANDWIDTH-bound
        => Build-A canNOT beat the byte floor; only fp8/amortization (accept-len, concurrency)
           cut HBM => Build-A NOT worth it for single-stream.
  (The bs sweep cross-checks: if BW rises steeply 1->128, bs=1 is under-fed (latency/granularity),
   confirming the lever is overlap/amortization, not a faster arithmetic kernel.)

------------------------------------------------------------------------------------------------
RUN ON THE HPU BOX (cards FREE; the device run is the main session's, serially):

  cd $HOME/AI/hpu-moe-ext

  # PRIMARY: production fused_expert op (LAZY = the serve's capture path), device-time + sweep,
  # with the Synapse profiler ON so a synprof CSV with an HBM column is emitted:
  HABANA_PROFILE=1 PT_HPU_LAZY_MODE=1 HPU_MOE_FUSED_OP=1 HPU_MOE_ROUTED_K=8 \
    python tests/bs1_moe_roofline_probe.py --op fused_expert --layers 75 --bs 1,8,32,128

  # CROSS-CHECK with the BATCHED single-recipe op (K*14 -> 16 nodes), same profiler:
  HABANA_PROFILE=1 PT_HPU_LAZY_MODE=1 HPU_MOE_FUSED_OP=1 HPU_MOE_ROUTED_K=8 \
    MOE_BATCHED_SO=tpc_kernel/opbackend/build/libmoe_fused_batched_lazy.so \
    python tests/bs1_moe_roofline_probe.py --op fused_moe_batched --layers 75 --bs 1,8,32,128

  # bs=1 ONLY, fast (skip sweep), still gets device-time + achieved BW:
  HABANA_PROFILE=1 PT_HPU_LAZY_MODE=1 python tests/bs1_moe_roofline_probe.py --bs 1

The profiler artifacts (synprof CSV / kineto trace) land under tests/bs1_prof/. If HABANA_PROFILE
is unset the probe still runs and reports device-time-from-kineto + bytes/time achieved-BW (it
just won't have the per-node DRAM CSV column; it says so explicitly rather than guessing).

Single process, single card, NO vLLM, NO model load, NO docker run BY THIS SCRIPT (it is launched
INSIDE the dev container by the caller, like every other tests/*.py here). ~K*19 MB of int4
weights -> trivially fits 96 GB.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# ===========================================================================
# GLM-5.2-int4 shapes (verified against
# /mnt/fastmodels/GLM-5.2-int4-experts-AutoRound/config.json):
#   hidden_size=6144  moe_intermediate_size=2048  n_routed_experts=256
#   n_shared_experts=1  num_experts_per_tok=8  group_size=128
# ===========================================================================
H = 6144            # hidden_size
I = 2048            # moe_intermediate_size
TWO_I = 2 * I       # 4096 gate+up (w13 out dim)
GS = 128            # group_size
ZERO = 8            # GPTQ-symmetric shared zero (uint4 midpoint)
TOP_K = 8           # num_experts_per_tok (the bs=1 routed width)

HBM_BW_PEAK = 2.45e12       # bytes/s, Gaudi-2 HBM2e peak (project's cited figure)
# Gaudi-2 (HL-225) BF16 MME peak ~432 TFLOP/s (vendor spec; used only for the MME-util proxy).
MME_BF16_PEAK = 432e12


# ---------------------------------------------------------------------------
# Proven packing helpers (bit-identical to tests/test_fused_moe_batched.py and
# tests/test_fused_expert_general.py -> the op's validated packer). Lifted so this
# probe introduces ZERO new numeric assumptions; only measurement code is new.
# ---------------------------------------------------------------------------
def _pack_u4(codes_2d, device):
    """codes [IN,OUT] (0..15) -> packed int32 [IN, OUT//8] (8 nibbles/word, little-endian)."""
    import torch
    IN, OUT = codes_2d.shape
    c = (codes_2d.to(torch.int64) & 0xF).reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.to(device)


def _sym_zero_packed(groups, out_dim, device):
    """uint4 midpoint 8 packed 8/int32 along OUT -> [groups, out//8] int32 (0x88888888)."""
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _make_expert(seed, device):
    """One GLM expert's packed int4 weights + bf16 scales + shared sym zero (op layout):
      w13_packed[H, 2I//8] int32, w13_scale[H//GS, 2I] bf16, w13_zp[H//GS, 2I//8] int32,
      w2_packed [I, H//8]  int32, w2_scale [I//GS, H]  bf16, w2_zp [I//GS, H//8]  int32."""
    import torch
    g = torch.Generator().manual_seed(seed)
    c13 = torch.randint(0, 16, (H, TWO_I), generator=g, dtype=torch.int32)
    c2 = torch.randint(0, 16, (I, H), generator=g, dtype=torch.int32)
    w13p = _pack_u4(c13, device)
    w2p = _pack_u4(c2, device)
    s13 = (torch.rand((H // GS, TWO_I), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    s2 = (torch.rand((I // GS, H), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    z13 = _sym_zero_packed(H // GS, TWO_I, device)
    z2 = _sym_zero_packed(I // GS, H, device)
    return w13p, s13, z13, w2p, s2, z2


def _relnorm(got, ref):
    g = got.float(); r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    ma = (g - r).abs().max().item()
    return rn, ma


def _ref_expert(x, w13p, s13, z13, w2p, s2, z2):
    """Independent eager reference: TRUSTED convert_from_uint4 + MoE math -> [M,H] (UNWEIGHTED)."""
    import torch
    import torch.nn.functional as F
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H,2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)       # [I,H]
    gate_up = torch.matmul(x, W13)             # [M,2I]
    gate = gate_up[:, :I]; up = gate_up[:, I:]
    h = F.silu(gate) * up                      # [M,I]
    return torch.matmul(h, W2)                 # [M,H]


# ---------------------------------------------------------------------------
# Byte accounting (per the GLM shapes). These are the bytes the op MUST read
# from HBM per (expert, layer): the int4 weights + bf16 scales (the activation
# + output are tiny at bs=1). The achieved-BW denominator.
# ---------------------------------------------------------------------------
def _expert_weight_bytes():
    """int4 packed weight bytes + bf16 scale bytes for ONE expert (w13 + w2)."""
    int4 = (H * TWO_I + I * H) * 0.5                      # 4 bits/code
    scale = ((H // GS) * TWO_I + (I // GS) * H) * 2       # bf16 scales
    return int4, scale


def _expert_flops(bs):
    """bs-token MoE expert FLOPs (2 GEMMs): x@W13 [bs,H]x[H,2I] + h@W2 [bs,I]x[I,H].
    2*M*N*K each."""
    return 2.0 * bs * H * TWO_I + 2.0 * bs * I * H


# ---------------------------------------------------------------------------
# Profiler glue. Two independent device measurements:
#   (1) DEVICE-BUSY TIME  : torch.profiler HPU activity -> key_averages()
#       self_device_time_total (sum over the fused-op events). This is the time
#       the device engines were BUSY (excludes host launch gaps).
#   (2) HBM BYTES         : the Synapse device profiler (HABANA_PROFILE=1) synprof
#       CSV HBM/DRAM column, parsed by the SAME logic profile_fusion.py uses.
# ---------------------------------------------------------------------------
def _kineto_device_us(prof) -> "tuple[float, float, str]":
    """Return (total_self_device_us, total_self_cpu_us, detail) from a finished
    torch.profiler. self_device_time_total is the HPU-busy time; self_cpu is host.
    Robust across torch versions (device vs cuda attr name)."""
    try:
        ka = prof.key_averages()
    except Exception as ex:  # noqa: BLE001
        return 0.0, 0.0, f"key_averages() failed: {type(ex).__name__}: {ex}"
    dev_us = 0.0
    cpu_us = 0.0
    for e in ka:
        # torch renamed cuda_time -> device_time across versions; try both, prefer device.
        for attr in ("self_device_time_total", "self_hpu_time_total",
                     "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                dev_us += float(v)
                break
        v = getattr(e, "self_cpu_time_total", None)
        if v:
            cpu_us += float(v)
    return dev_us, cpu_us, f"summed {len(ka)} key_averages rows"


def _find_hbm_from_synprof(prof_dir) -> "tuple[float|None, str]":
    """Reuse profile_fusion.py's robust synprof-CSV / TEF HBM parser if importable;
    else inline a minimal CSV scan. Returns (MB, detail) or (None, reason)."""
    # Prefer the committed, battle-tested parser.
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "dsa_e2e", "build_stage", "fusion"))
        import profile_fusion as PF  # type: ignore
        mb, src = PF._parse_synprof_hbm(prof_dir)
        if mb is not None:
            return mb, f"profile_fusion._parse_synprof_hbm: {src}"
        # try TEF traces too
        tefs = sorted(glob.glob(os.path.join(prof_dir, "**", "*.json"), recursive=True),
                      key=os.path.getmtime, reverse=True)
        for t in tefs:
            mb, src = PF._parse_tef_hbm(t)
            if mb is not None:
                return mb, f"profile_fusion._parse_tef_hbm: {src}"
        return None, f"profile_fusion parser found no HBM column under {prof_dir}"
    except Exception as ex:  # noqa: BLE001
        return None, (f"profile_fusion import/parse unavailable ({type(ex).__name__}: {ex}); "
                      f"no inline HBM CSV found under {prof_dir}")


# ---------------------------------------------------------------------------
# Op runners. Each `run_layers(w, bs, layers)` does ONE full-model-worth of MoE
# weight work: for each of `layers` layers, the K-expert routed compute for `bs`
# tokens. We RE-READ the per-expert weights each layer so HBM traffic ==
# K*layers*expert_bytes (matching the real per-token, per-layer weight read).
# ---------------------------------------------------------------------------
def _build(K, bs, device):
    """K distinct GLM experts (stacked + listed) + a bs-token activation."""
    import torch
    import habana_frameworks.torch.core as htorch
    experts = [_make_expert(100 + k, device) for k in range(K)]
    w13p = torch.stack([e[0] for e in experts], 0)   # [K,H,2I//8]
    s13 = torch.stack([e[1] for e in experts], 0)    # [K,H//GS,2I]
    z13 = torch.stack([e[2] for e in experts], 0)    # [K,H//GS,2I//8]
    w2p = torch.stack([e[3] for e in experts], 0)    # [K,I,H//8]
    s2 = torch.stack([e[4] for e in experts], 0)     # [K,I//GS,H]
    z2 = torch.stack([e[5] for e in experts], 0)     # [K,I//GS,H//8]
    x = (torch.randn(bs, H) * 0.1).to(torch.bfloat16).to(device)
    # per-expert offset-0 contiguous slices (the op's stride-sensitivity invariant; see
    # fused_expert_routed.py _co/_cs). Materialize ONCE here, outside the timed loop.
    def _co(t):
        return t.contiguous().clone()
    def _cs(t):
        return t.to(torch.bfloat16).contiguous().clone()
    seq = []
    for k in range(K):
        seq.append((
            _co(w13p[k]), _cs(s13[k]), _co(z13[k]),
            _co(w2p[k]), _cs(s2[k]), _co(z2[k])))
    rw1 = torch.ones((bs, 1), dtype=torch.bfloat16, device=device)   # unweighted (we gate vs unweighted ref)
    rw_stacked = (torch.ones(K, 1, 1, dtype=torch.bfloat16, device=device) / K)  # batched op: [K,1,1]
    htorch.mark_step()
    return dict(seq=seq, w13p=w13p, s13=s13, z13=z13, w2p=w2p, s2=s2, z2=z2,
                x=x, rw1=rw1, rw_stacked=rw_stacked, K=K, bs=bs)


def _run_fused_expert(w, layers):
    """Production per-expert op, K calls x layers (the routed wrapper's T=1 shape).
    Unweighted (rw=1) so it sums to the unweighted K-expert reference exactly."""
    import torch
    acc = None
    x = w["x"]; rw = w["rw1"]
    for _ in range(layers):
        for (w13p, s13, z13, w2p, s2, z2) in w["seq"]:
            y = torch.ops.hpu_moe_ext.fused_expert(x, w13p, s13, z13, w2p, s2, z2, rw)  # [bs,H]
            acc = y if acc is None else acc + y
    return acc


def _run_fused_moe_batched(w, batched_op, layers):
    """BATCHED single-recipe op (K*14 -> 16 nodes), stacked feed x layers. The op
    reduce-sums over K with the [K,1,1] router weights (here 1/K) -> we compare to the
    MEAN of the K unweighted experts (CHECK builds the matching reference)."""
    acc = None
    for _ in range(layers):
        out = batched_op(w["x"], w["w13p"], w["s13"], w["z13"],
                         w["w2p"], w["s2"], w["z2"], w["rw_stacked"])   # [bs,H]
        acc = out if acc is None else acc + out
    return acc


def _correctness_gate(torch, htorch, w, op, batched_op, TOL):
    """LOUD byte-exact-ish check of the MEASURED op vs an independent eager reference.
    Returns 0 on PASS, 1 on FAIL. NO silent fallback -- a wrong kernel must abort."""
    x = w["x"]; K = w["K"]
    if op == "fused_expert":
        got = None
        for (w13p, s13, z13, w2p, s2, z2) in w["seq"]:
            y = torch.ops.hpu_moe_ext.fused_expert(x, w13p, s13, z13, w2p, s2, z2, w["rw1"])
            got = y if got is None else got + y
        htorch.mark_step()
        # reference: sum of K unweighted experts via the trusted convert_from_uint4 path
        ref = None
        for (w13p, s13, z13, w2p, s2, z2) in w["seq"]:
            e = _ref_expert(x, w13p, s13, z13, w2p, s2, z2)
            ref = e if ref is None else ref + e
        htorch.mark_step()
    else:  # fused_moe_batched: reduce-sum over K weighted by 1/K -> MEAN of K experts
        got = batched_op(w["x"], w["w13p"], w["s13"], w["z13"],
                         w["w2p"], w["s2"], w["z2"], w["rw_stacked"])
        htorch.mark_step()
        ref = None
        for (w13p, s13, z13, w2p, s2, z2) in w["seq"]:
            e = _ref_expert(x, w13p, s13, z13, w2p, s2, z2)
            ref = e if ref is None else ref + e
        ref = ref / K
        htorch.mark_step()
    gc, rc = got.to("cpu"), ref.to("cpu")
    rn, ma = _relnorm(gc, rc)
    print(f"  CORRECTNESS [{op}]: |ref|={rc.float().norm().item():.4f} "
          f"|got|={gc.float().norm().item():.4f} relNorm={rn:.4e} maxAbs={ma:.4e}", flush=True)
    if tuple(gc.shape) != (w["bs"], H):
        print(f"  >>> FAIL: output shape {tuple(gc.shape)} != ({w['bs']},{H}).", file=sys.stderr)
        return 1
    if rn > TOL:
        print(f"  >>> FAIL: relNorm {rn:.4e} > tol {TOL:.0e} -- the MEASURED op does NOT match "
              f"the eager reference. ABORTING (a wrong kernel would mis-direct the project).",
              file=sys.stderr)
        return 1
    print(f"  >>> PASS: measured op matches the independent eager reference (relNorm {rn:.2e}).",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# Timing: wall-clock (host) + device-busy (profiler) for one (bs, layers) point.
# ---------------------------------------------------------------------------
def _time_point(torch, htorch, runner, iters, warmup):
    """Returns (wall_ms_per_iter, device_us_per_iter, cpu_us_per_iter, prof_detail).
    Wall = perf_counter around iters synced launches. Device = profiler self_device
    time over a SEPARATE short profiled window (profiling adds overhead, so wall and
    device are measured in different windows; both are per-iter)."""
    # warmup + wall timing (no profiler overhead)
    for _ in range(warmup):
        o = runner(); htorch.mark_step()
    _ = o.detach().to("cpu")
    t0 = time.perf_counter()
    for _ in range(iters):
        o = runner(); htorch.mark_step()
    _ = o.detach().to("cpu")
    wall_ms = (time.perf_counter() - t0) / iters * 1e3

    # device-busy timing via torch.profiler HPU activity (short window)
    dev_us = cpu_us = 0.0
    detail = "profiler unavailable"
    try:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU]
        if hasattr(ProfilerActivity, "HPU"):
            acts.append(ProfilerActivity.HPU)
        n_prof = max(3, min(iters, 5))
        with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
            for _ in range(n_prof):
                o = runner(); htorch.mark_step()
            _ = o.detach().to("cpu")
        d_us, c_us, detail = _kineto_device_us(prof)
        dev_us = d_us / n_prof
        cpu_us = c_us / n_prof
    except Exception as ex:  # noqa: BLE001
        detail = f"torch.profiler failed: {type(ex).__name__}: {ex}"
    return wall_ms, dev_us, cpu_us, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["fused_expert", "fused_moe_batched"],
                    default="fused_expert",
                    help="which production fused op to measure (default: fused_expert)")
    ap.add_argument("--bs", default="1,8,32,128",
                    help="comma list of batch sizes for the roofline sweep (default 1,8,32,128)")
    ap.add_argument("--layers", type=int, default=75,
                    help="MoE layers to replicate per iter (GLM-5.2 ~ 75); scales total bytes")
    ap.add_argument("--routed-k", type=int, default=int(os.environ.get("HPU_MOE_ROUTED_K", "8") or "8"),
                    help="K routed experts/token (GLM-5.2 num_experts_per_tok=8)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()
    K = args.routed_k
    bs_list = [int(b) for b in str(args.bs).split(",") if b.strip()]

    # ---- env / framework ----
    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run INSIDE the dev container "
              f"on the HPU box.", file=sys.stderr)
        return 2

    dev = "hpu"
    prof_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bs1_prof")
    os.makedirs(prof_dir, exist_ok=True)

    # ---- load the production op .so (lazy variant under lazy mode) ----
    here = os.path.dirname(os.path.abspath(__file__))
    build = os.path.join(here, "..", "tpc_kernel", "opbackend", "build")
    lazy = htorch.is_lazy()

    def _load(soname, opname):
        if hasattr(torch.ops, "hpu_moe_ext") and hasattr(torch.ops.hpu_moe_ext, opname):
            return getattr(torch.ops.hpu_moe_ext, opname)
        so = os.path.abspath(os.path.join(build, soname))
        if not os.path.isfile(so):
            return None, so
        torch.ops.load_library(so)
        return getattr(torch.ops.hpu_moe_ext, opname, None), so

    batched_op = None
    if args.op == "fused_expert":
        # fused_expert lives in the routed .so (TORCH_LIBRARY_FRAGMENT co-registers it).
        soname = "libmoe_fused_routed_lazy.so" if lazy else "libmoe_fused_routed.so"
        res = _load(soname, "fused_expert")
        op_handle = res[0] if isinstance(res, tuple) else res
        so_path = res[1] if isinstance(res, tuple) else soname
        if op_handle is None:
            # fall back to the single-expert-only .so
            soname2 = "libmoe_fused_expert_lazy.so" if lazy else "libmoe_fused_expert.so"
            res = _load(soname2, "fused_expert")
            op_handle = res[0] if isinstance(res, tuple) else res
            so_path = res[1] if isinstance(res, tuple) else soname2
        if op_handle is None:
            print(f"FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after loading "
                  f"{so_path}. Build it (CPU-only in the dev image):\n"
                  f"  docker run --rm -v {os.path.abspath(os.path.join(here, '..', 'tpc_kernel'))}:/work "
                  f"-w /work/opbackend vllm-gaudi:next bash build{'_lazy' if lazy else ''}.sh",
                  file=sys.stderr)
            return 2
    else:  # fused_moe_batched
        env_so = os.environ.get("MOE_BATCHED_SO")
        soname = (os.path.abspath(env_so) if env_so
                  else ("libmoe_fused_batched_lazy.so" if lazy else "libmoe_fused_batched.so"))
        res = _load(soname if env_so else os.path.basename(soname), "fused_moe_batched")
        batched_op = res[0] if isinstance(res, tuple) else res
        so_path = res[1] if isinstance(res, tuple) else soname
        if batched_op is None:
            print(f"FATAL: torch.ops.hpu_moe_ext.fused_moe_batched not registered after loading "
                  f"{so_path}. Build it (CPU-only):\n"
                  f"  docker run --rm -v {os.path.abspath(os.path.join(here, '..', 'tpc_kernel'))}:/work "
                  f"-w /work/opbackend vllm-gaudi:next bash build_batched{'_lazy' if lazy else ''}.sh",
                  file=sys.stderr)
            return 2

    int4_b, scale_b = _expert_weight_bytes()
    per_expert_b = int4_b + scale_b
    profiling = (os.environ.get("HABANA_PROFILE") or "0") in ("1", "true", "True")

    print("=" * 90, flush=True)
    print(f"bs=1 MoE ROOFLINE PROBE | op={args.op} | GLM-5.2 H={H} I={I} GS={GS} "
          f"K={K} layers={args.layers} | lazy={lazy} | HABANA_PROFILE={'ON' if profiling else 'OFF'}",
          flush=True)
    print(f"  loaded so: {so_path}", flush=True)
    print(f"  per-expert weight bytes (int4 {int4_b/1e6:.2f}MB + scale {scale_b/1e6:.2f}MB) "
          f"= {per_expert_b/1e6:.2f} MB", flush=True)
    print(f"  per-iter weight read = K*layers*per_expert = {K}*{args.layers}*{per_expert_b/1e6:.2f}MB "
          f"= {K*args.layers*per_expert_b/1e9:.4f} GB", flush=True)
    print(f"  HBM peak {HBM_BW_PEAK/1e12} TB/s | MME bf16 peak {MME_BF16_PEAK/1e12:.0f} TFLOP/s",
          flush=True)
    print("=" * 90, flush=True)

    # ---- byte volume per iter is INDEPENDENT of bs (same K experts x layers read) ----
    weight_bytes_iter = K * args.layers * per_expert_b

    rows = []   # (bs, wall_ms, dev_ms, cpu_ms, duty, achieved_GBs, mme_util)
    rc = 0
    for bs in bs_list:
        print(f"\n----- bs={bs} -----", flush=True)
        w = _build(K, bs, dev)

        # CORRECTNESS GATE (once per bs; LOUD on fail) -------------------------------
        gate_rc = _correctness_gate(torch, htorch, w, args.op, batched_op, args.tol)
        rc |= gate_rc
        if gate_rc:
            print(f"  (skipping timing for bs={bs}: correctness gate FAILED)", flush=True)
            continue

        # runner closure for the timed loop
        if args.op == "fused_expert":
            runner = lambda: _run_fused_expert(w, args.layers)
        else:
            runner = lambda: _run_fused_moe_batched(w, batched_op, args.layers)

        wall_ms, dev_us, cpu_us, pdetail = _time_point(
            torch, htorch, runner, args.iters, args.warmup)
        dev_ms = dev_us / 1e3
        cpu_ms = cpu_us / 1e3
        # device duty cycle: how much of the wall the device engines were BUSY.
        duty = (dev_ms / wall_ms) if wall_ms > 0 else 0.0
        # achieved BW: bytes that MUST move / device-busy time (the honest device-side BW).
        # If device time is unavailable, fall back to wall (clearly labelled).
        bw_denom_s = (dev_ms / 1e3) if dev_ms > 0 else (wall_ms / 1e3)
        bw_basis = "device" if dev_ms > 0 else "wall"
        achieved_GBs = (weight_bytes_iter / bw_denom_s) / 1e9 if bw_denom_s > 0 else 0.0
        # MME util proxy: total GEMM FLOPs / device time vs peak.
        flops_iter = K * args.layers * _expert_flops(bs)
        mme_tflops = (flops_iter / bw_denom_s) / 1e12 if bw_denom_s > 0 else 0.0
        mme_util = mme_tflops / (MME_BF16_PEAK / 1e12)

        per_layer_dev_ms = dev_ms / args.layers
        print(f"  wall      = {wall_ms:9.3f} ms/iter   (host loop, synced)", flush=True)
        print(f"  device    = {dev_ms:9.3f} ms/iter   (HPU self_device_time; {pdetail})", flush=True)
        print(f"  host(cpu) = {cpu_ms:9.3f} ms/iter   (self_cpu_time)", flush=True)
        print(f"  device DUTY CYCLE = {duty*100:5.1f}%   "
              f"({'device mostly IDLE -> latency/dispatch' if duty < 0.5 else 'device mostly BUSY'})",
              flush=True)
        print(f"  per-MoE-layer device time = {per_layer_dev_ms:.3f} ms "
              f"(x{args.layers} layers; compare to the ~51 ms two-fused-kernel trace)", flush=True)
        print(f"  achieved HBM BW = {achieved_GBs:8.1f} GB/s  "
              f"= {100*achieved_GBs*1e9/HBM_BW_PEAK:4.1f}% of {HBM_BW_PEAK/1e12} TB/s  "
              f"(basis: {bw_basis} time)", flush=True)
        print(f"  MME util (proxy) = {mme_util*100:4.1f}% of {MME_BF16_PEAK/1e12:.0f} TFLOP/s "
              f"({mme_tflops:.1f} TFLOP/s achieved)", flush=True)
        rows.append((bs, wall_ms, dev_ms, cpu_ms, duty, achieved_GBs, mme_util))

    # ===================================================================
    # ROOFLINE SWEEP TABLE + DECISION
    # ===================================================================
    print("\n" + "=" * 90, flush=True)
    print(f"ROOFLINE SWEEP ({args.op}, K={K}, layers={args.layers}) -- the bs->roofline climb",
          flush=True)
    print("=" * 90, flush=True)
    print(f"  {'bs':>4} | {'wall ms':>9} | {'dev ms':>9} | {'duty%':>6} | "
          f"{'HBM GB/s':>9} | {'%peakBW':>8} | {'MMEutil%':>8}", flush=True)
    print("  " + "-" * 78, flush=True)
    for (bs, wall_ms, dev_ms, cpu_ms, duty, gbs, mme) in rows:
        print(f"  {bs:>4} | {wall_ms:9.2f} | {dev_ms:9.2f} | {duty*100:6.1f} | "
              f"{gbs:9.1f} | {100*gbs*1e9/HBM_BW_PEAK:8.1f} | {mme*100:8.1f}", flush=True)

    # ---- the authoritative HBM-byte cross-check from the Synapse profiler CSV ----
    print("\n[HBM-bytes cross-check from the Synapse device profiler]", flush=True)
    if profiling:
        mb, src = _find_hbm_from_synprof(prof_dir)
        if mb is not None:
            print(f"  synprof measured HBM traffic = {mb:.2f} MB total  ({src})", flush=True)
            print(f"  (compare to the per-iter weight read {weight_bytes_iter/1e6:.1f} MB; "
                  f"a synprof BW = measured_MB / measured_device_time is the ground-truth "
                  f"achieved bandwidth -- use the per-node column to attribute weight vs act.)",
                  flush=True)
        else:
            print(f"  no per-node HBM/DRAM column found ({src}).", flush=True)
            print(f"  -> the bytes/device-time achieved-BW above is the decisive number; the "
                  f"profiler CSV would only refine the byte attribution. To force a CSV with an "
                  f"HBM column on the box, run with HABANA_PROFILE=1 and post-process the .hltv "
                  f"via hl-prof-config / habana-perf-tool (see profile_fusion.py).", flush=True)
    else:
        print("  HABANA_PROFILE unset -> no synprof CSV. The bytes/device-time achieved-BW above "
              "is computed from KNOWN bytes and measured DEVICE time (decisive: ~33 vs ~2000 "
              "GB/s). Re-run with HABANA_PROFILE=1 to also get the per-node DRAM byte column.",
              flush=True)

    # ---- DECISION ----
    print("\n" + "=" * 90, flush=True)
    print("DECISION (bs=1 is the single-stream question)", flush=True)
    print("=" * 90, flush=True)
    bs1 = next((r for r in rows if r[0] == 1), None)
    if bs1 is None:
        print("  bs=1 not measured (gate failed or not requested) -> rerun with --bs 1.",
              flush=True)
    else:
        _, wall1, dev1, cpu1, duty1, gbs1, mme1 = bs1
        pct1 = 100 * gbs1 * 1e9 / HBM_BW_PEAK
        print(f"  bs=1: device {dev1:.2f} ms/iter ({dev1/args.layers:.3f} ms/layer), "
              f"duty {duty1*100:.1f}%, achieved HBM {gbs1:.0f} GB/s ({pct1:.1f}% of peak), "
              f"MME {mme1*100:.1f}%", flush=True)
        # roofline climb signal
        climb = ""
        if len(rows) >= 2:
            gbs_max = max(r[5] for r in rows)
            climb = (f"  roofline: HBM BW climbs {gbs1:.0f} -> {gbs_max:.0f} GB/s across the bs "
                     f"sweep ({gbs_max/max(gbs1,1e-9):.1f}x). ")
        if duty1 < 0.5:
            verdict = ("LATENCY/DISPATCH-BOUND -- the device is IDLE most of the wall at bs=1. "
                       "The ~51 ms is host launch/sync gaps, NOT device work. "
                       ">>> Build-A (collapse the K launches into one recipe + overlap host) IS "
                       "WORTH IT: it directly removes the idle gaps. Pair with the async-overlap "
                       "port (the already-chosen lever).")
        elif pct1 < 30.0:
            verdict = ("UNDER-FED / GRANULARITY-BOUND -- device is busy but only "
                       f"{pct1:.0f}% of HBM peak: the small per-expert ops can't saturate HBM at "
                       "bs=1. >>> Build-A (one big batched read feeding the MME) IS WORTH IT to "
                       "reach the roofline; fp8 also helps (fewer bytes + 2x MME).")
        elif pct1 >= 60.0:
            verdict = ("BANDWIDTH-BOUND -- device is busy AND HBM is ~saturated at bs=1: the "
                       f"{gbs1:.0f} GB/s is near the {HBM_BW_PEAK/1e12} TB/s floor for these "
                       "bytes. >>> Build-A canNOT beat the byte floor for single-stream; the "
                       "only single-stream wins are FEWER BYTES (fp8-resident) or AMORTIZATION "
                       "(higher accept-len / concurrency). Build-A NOT worth it for tok/s here.")
        else:
            verdict = (f"MIXED ({pct1:.0f}% of peak BW, duty {duty1*100:.0f}%) -- partially fed. "
                       "Build-A likely gives a MODERATE win (closes the granularity gap) but is "
                       "not a silver bullet; weigh it against the cheaper async-overlap + fp8 "
                       "levers. Inspect the bs sweep slope + the synprof per-node column.")
        print(climb + "\n  >>> " + verdict, flush=True)

    print(f"\n[done] bs1_moe_roofline_probe {'OK' if rc == 0 else 'FAILED (correctness gate)'} "
          f"(rc={rc})", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
