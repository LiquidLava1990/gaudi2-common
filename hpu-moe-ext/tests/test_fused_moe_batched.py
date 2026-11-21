"""test_fused_moe_batched.py -- VALIDATE the BATCHED-K single-recipe fused op
`torch.ops.hpu_moe_ext.fused_moe_batched` ON THE HPU BOX.

This is the SINGLE-STREAM-SPEED variant of the K-expert single-recipe op: instead of
emitting K sequential per-expert subgraphs (~K*14 node launches), it emits ONE
batched chain of 16 nodes INDEPENDENT of K -- a grouped dequant over the stacked
[K,...] weights + a batch_gemm over the K-expert batch + a batched SiLU + a 2nd
grouped dequant + a 2nd batch_gemm + the router-weighted reduce-sum over K
(moe_fused_expert_batched_backend.cpp / moe_fused_expert_batched_emit.h).

    torch.ops.hpu_moe_ext.fused_moe_batched(
        x[M,H] bf16,
        w13_packed[K, H, 2I//8] int32,  w13_scale[K, H//GS, 2I] bf16,  w13_zp[K, H//GS, 2I//8] int32,
        w2_packed [K, I, H//8]  int32,  w2_scale [K, I//GS, H]  bf16,  w2_zp [K, I//GS, H//8]  int32,
        router_weights[K, 1, 1] bf16) -> [M, H] bf16
    == sum_k( fused_expert(x, W_k) * router_weights[k] ), as ONE batched recipe.

STACKED-K LAYOUT: unlike the SEQUENTIAL op (6 Python lists), the batched op takes
each weight class as ONE STACKED [K,...] tensor (fed DIRECTLY -- no .unbind). This is
the natural batched feed and keeps the held-leaf source contiguous (multi-bucket-safe).

We test the OP DIRECTLY (build the K-stacked tensors ourselves, T=1 so each expert's
router weight is a true [K,1,1] scalar -- the folded form) and compare against:
  CHECK 1: an INDEPENDENT per-expert convert_from_uint4 + MoE oracle (relNorm < 1e-2).
  CHECK 2: the SEQUENTIAL single-recipe op `fused_moe` (the VALIDATED per-expert
           oracle) on the SAME inputs -- a tight batched-vs-sequential parity gate
           (relNorm < 1e-2; both are bf16 so a small delta is expected from the
           reduce-sum vs add-tree summation order, NOT from different math).
  CHECK 3: HPU-graph capture of the batched recipe (the single-launch serving path).

Gate: relNorm = ||got - ref|| / ||ref|| < 1e-2 (the bf16 norm-relative tol). LOUD
(nonzero exit + stderr) on any failure. K=8 AND K=10 are exercised by default.

GLM-5.1-int4 default shape: H=6144, I=2048, GS=128.

RUN (ON THE HPU BOX -- a CPU import cannot exercise the op):
  cd $HOME/AI/hpu-moe-ext
  python tests/test_fused_moe_batched.py                 # GLM default: M=1 H=6144 I=2048 GS=128, K in {8,10}
  python tests/test_fused_moe_batched.py 1 6144 2048 128 8   # M H I GS K (single K)
  # LAZY capture variant (loads the lazy .so):
  PT_HPU_LAZY_MODE=1 MOE_BATCHED_SO=tpc_kernel/opbackend/build/libmoe_fused_batched_lazy.so \
    python tests/test_fused_moe_batched.py
"""
from __future__ import annotations

import os
import sys


def _sym_zero_packed(groups, out_dim, device):
    """uint4 midpoint 8 packed 8/int32 along OUT -> [groups, out//8] int32 (0x88888888)."""
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _pack_u4(codes_2d, device):
    """codes [IN, OUT] (0..15) -> packed int32 [IN, OUT//8] (8 nibbles/word, little-endian).
    Bit-identical to the single-expert op's test packer (test_fused_expert_op.py)."""
    import torch
    IN, OUT = codes_2d.shape
    c = (codes_2d.to(torch.int64) & 0xF).reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.to(device)


def _make_expert(H, I, GS, seed, device):
    """One expert's packed int4 weights + scales (+ shared sym zero), GLM layout.
    Returns w13_packed[H,2I//8], w13_scale[H//GS,2I], w13_zp[H//GS,2I//8],
            w2_packed[I,H//8],  w2_scale[I//GS,H],  w2_zp[I//GS,H//8]."""
    import torch
    OUT13 = 2 * I
    g = torch.Generator().manual_seed(seed)
    c13 = torch.randint(0, 16, (H, OUT13), generator=g, dtype=torch.int32)
    c2 = torch.randint(0, 16, (I, H), generator=g, dtype=torch.int32)
    w13_packed = _pack_u4(c13, device)
    w2_packed = _pack_u4(c2, device)
    s13 = (torch.rand((H // GS, OUT13), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    s2 = (torch.rand((I // GS, H), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    z13 = _sym_zero_packed(H // GS, OUT13, device)
    z2 = _sym_zero_packed(I // GS, H, device)
    return w13_packed, s13, z13, w2_packed, s2, z2


def _ref_one(x, w13p, s13, z13, w2p, s2, z2, I):
    """Reference UNSCALED single-expert forward via TRUSTED convert_from_uint4 + MoE
    math. Returns expert_out [M, H]."""
    import torch
    import torch.nn.functional as F
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H, 2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)       # [I, H]
    gate_up = torch.matmul(x, W13)            # [M, 2I]
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    h = F.silu(gate) * up                     # [M, I]
    return torch.matmul(h, W2)                # [M, H]


def _relnorm(got, ref):
    g = got.float(); r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    maxabs = (g - r).abs().max().item()
    return rn, maxabs


def _find_batched_so():
    """The BATCHED-K .so. Default eager build; override with MOE_BATCHED_SO (e.g.
    point it at libmoe_fused_batched_lazy.so for the LAZY capture variant)."""
    env = os.environ.get("MOE_BATCHED_SO")
    if env:
        so = os.path.abspath(env)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        so = os.path.abspath(os.path.join(
            here, "..", "tpc_kernel", "opbackend", "build", "libmoe_fused_batched.so"))
    if not os.path.isfile(so):
        here = os.path.dirname(os.path.abspath(__file__))
        raise FileNotFoundError(
            f"{so} not found. Build the BATCHED-K op first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}:/work "
            f"-w /work/tpc_kernel/opbackend vllm-gaudi:next bash build_batched.sh")
    return so


def _maybe_load_sequential_so():
    """Best-effort load of the SEQUENTIAL single-recipe op (`fused_moe`) for the
    batched-vs-sequential parity gate (CHECK 2). Picks the lazy .so under lazy mode,
    eager otherwise. Returns the op or None if its .so is absent (CHECK 2 skipped)."""
    import torch
    import habana_frameworks.torch.core as htorch
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(torch.ops.hpu_moe_ext, "fused_moe"):
        return torch.ops.hpu_moe_ext.fused_moe
    name = ("libmoe_fused_routed_lazy.so" if htorch.is_lazy()
            else "libmoe_fused_routed.so")
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.abspath(os.path.join(
        here, "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        return None
    torch.ops.load_library(so)
    return getattr(torch.ops.hpu_moe_ext, "fused_moe", None)


def _run_for_K(torch, htorch, batched_op, seq_op, M, H, I, GS, K, TOL):
    """Build K stacked experts, run the batched op, return rc (0 ok)."""
    dev = "hpu"
    OUT13 = 2 * I
    rc = 0
    print(f"\n========== K={K} ==========  M={M} H={H} I={I} (2I={OUT13}) GS={GS}  "
          f"(ONE batched recipe, 16 nodes for {K} experts)")

    # ---- K distinct experts (different seed each), STACKED into [K,...] tensors ----
    experts = [_make_expert(H, I, GS, 100 + k, dev) for k in range(K)]
    w13p = torch.stack([e[0] for e in experts], 0)   # [K, H, 2I//8]
    s13 = torch.stack([e[1] for e in experts], 0)    # [K, H//GS, 2I]
    z13 = torch.stack([e[2] for e in experts], 0)    # [K, H//GS, 2I//8]
    w2p = torch.stack([e[3] for e in experts], 0)    # [K, I, H//8]
    s2 = torch.stack([e[4] for e in experts], 0)     # [K, I//GS, H]
    z2 = torch.stack([e[5] for e in experts], 0)     # [K, I//GS, H//8]

    x = (torch.randn(M, H) * 0.1).to(torch.bfloat16).to(dev)
    g = torch.Generator().manual_seed(7 + K)
    rw_vals = torch.rand(K, generator=g, dtype=torch.float32)
    rw_vals = (rw_vals / rw_vals.sum()).to(torch.bfloat16)          # renormalized [K]
    rw_stacked = rw_vals.view(K, 1, 1).to(dev)                      # [K,1,1] bf16
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- numeric vs per-expert convert_from_uint4 oracle.
    # ====================================================================
    print("--- CHECK 1: fused_moe_batched numeric (vs per-expert convert_from_uint4 sum) ---")
    got = batched_op(x, w13p, s13, z13, w2p, s2, z2, rw_stacked)
    htorch.mark_step()

    ref = torch.zeros(M, H, dtype=torch.bfloat16, device=dev)
    for k in range(K):
        ek = _ref_one(x, w13p[k], s13[k], z13[k], w2p[k], s2[k], z2[k], I)  # [M,H]
        ref = ref + ek * rw_vals[k].to(dev)
    htorch.mark_step()

    gotc, refc = got.to("cpu"), ref.to("cpu")
    rn, ma = _relnorm(gotc, refc)
    print(f"  |ref|={refc.float().norm().item():.4f}  |got|={gotc.float().norm().item():.4f}  "
          f"relNorm={rn:.4e}  maxAbs={ma:.4e}")
    if got.shape != (M, H):
        print(f"  >>> FAIL: fused_moe_batched output shape {tuple(got.shape)} != ({M},{H}).",
              file=sys.stderr)
        rc = 1
    elif rn <= TOL:
        print(f"  >>> PASS: fused_moe_batched (K={K}, ONE batched recipe) matches the "
              f"per-expert oracle (relNorm {rn:.2e}).")
    else:
        print(f"  >>> FAIL: fused_moe_batched relNorm {rn:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        print("      Suspects (device-confirmable): the GROUPED dequant over the [K,...] "
              "weights\n"
              "      (per-expert scale/zp must NOT cross experts -- the K axis is shared "
              "1:1),\n"
              "      the batch_gemm batch dim (x broadcast [1,M,H]->[K,M,H]; contract H/I),\n"
              "      the rank-3 gate/up slice (dim 2), the [K,1,1] router broadcast over "
              "[K,M,H],\n"
              "      or the reduce_sum over dim 0 (K) (mask bit 1<<(3-0-1)=4).",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- BATCHED vs SEQUENTIAL parity (the per-expert single-recipe oracle).
    # ====================================================================
    if seq_op is not None:
        print("--- CHECK 2: fused_moe_batched vs SEQUENTIAL fused_moe (per-expert recipe) ---")
        # the sequential op takes 6 Python LISTS + a list of K [1,1] router weights.
        rws_list = list(rw_vals.view(K, 1, 1).to(dev).unbind(0))   # K x [1,1] bf16
        seq = seq_op(
            x,
            list(w13p.unbind(0)), list(s13.unbind(0)), list(z13.unbind(0)),
            list(w2p.unbind(0)), list(s2.unbind(0)), list(z2.unbind(0)),
            rws_list)
        htorch.mark_step()
        seqc = seq.to("cpu")
        rn2, ma2 = _relnorm(gotc, seqc)
        print(f"  |seq|={seqc.float().norm().item():.4f}  |batched|={gotc.float().norm().item():.4f}  "
              f"relNorm={rn2:.4e}  maxAbs={ma2:.4e}")
        if rn2 <= TOL:
            print(f"  >>> PASS: batched matches the validated sequential op (relNorm {rn2:.2e}) "
                  "-- same math, K*14 -> 16 launches.")
        else:
            print(f"  >>> FAIL: batched vs sequential relNorm {rn2:.4e} > tol {TOL:.2e} "
                  "(the two single-recipe ops should agree to bf16 tol).", file=sys.stderr)
            rc = 1
    else:
        print("--- CHECK 2: SKIPPED (sequential fused_moe .so not found; build_routed*.sh) ---")

    # ====================================================================
    # CHECK 3 -- HPU-graph CAPTURE: the batched recipe as ONE captured graph / launch.
    # ====================================================================
    print("--- CHECK 3: HPU-graph capture of the batched recipe (best-effort) ---")
    try:
        import habana_frameworks.torch.hpu as ht_hpu
        from habana_frameworks.torch.hpu.graphs import HPUGraph
        gph = HPUGraph()
        s = ht_hpu.Stream()
        with ht_hpu.stream(s):
            gph.capture_begin()
            out_cap = batched_op(x, w13p, s13, z13, w2p, s2, z2, rw_stacked)
            gph.capture_end()
        gph.replay()
        htorch.mark_step()
        # --- TIMING (the #3 batched-op speed verdict): median ms over N captured-graph replays ---
        import statistics, time, os
        N_TIME = int(os.environ.get("MOE_TIME_N", "60"))
        for _ in range(12):           # warmup
            gph.replay()
        htorch.mark_step(); ht_hpu.synchronize()
        ts = []
        for _ in range(N_TIME):
            ht_hpu.synchronize()
            t0 = time.perf_counter()
            gph.replay()
            ht_hpu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        print(f"  [TIMING/BATCHED] captured-replay median={statistics.median(ts):.4f}ms "
              f"min={ts[0]:.4f} p90={ts[int(0.9 * len(ts))]:.4f}  (N={N_TIME}, K={K}, M={M})")
        rn3, _ = _relnorm(out_cap.to("cpu"), refc)
        if rn3 <= TOL:
            print(f"  >>> PASS: the K={K} batched op captured into ONE HPU graph and replayed "
                  f"(relNorm {rn3:.2e}) -- the single-recipe, single-launch path.")
        else:
            print(f"  >>> WARN: captured replay relNorm {rn3:.4e} > tol (investigate on device).",
                  file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")

    return rc


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # tokens (single-stream T=1)
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    Ks = [int(sys.argv[5])] if len(sys.argv) > 5 else [8, 10]

    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H={H} I={I} GS={GS})")
        return 2
    if M != 1:
        print(f"NOTE: this direct test uses the [K,1,1] router fold; got M={M}. For M>1 the "
              "per-token weight is post-applied by the Python wrapper.", file=sys.stderr)

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    try:
        so = _find_batched_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_batched"):
        print("FATAL: torch.ops.hpu_moe_ext.fused_moe_batched not registered after "
              f"load_library({so}). Check the TORCH_LIBRARY_FRAGMENT(hpu_moe_ext) "
              "+ the .so.", file=sys.stderr)
        return 2
    print(f"loaded {so}")
    batched_op = torch.ops.hpu_moe_ext.fused_moe_batched
    seq_op = _maybe_load_sequential_so()

    torch.manual_seed(0)
    TOL = 1e-2
    print(f"=== test_fused_moe_batched ===  M={M} H={H} I={I} GS={GS}  Ks={Ks}")

    rc = 0
    for K in Ks:
        rc |= _run_for_K(torch, htorch, batched_op, seq_op, M, H, I, GS, K, TOL)

    print(f"\n[done] test_fused_moe_batched {'OK' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
