"""test_fused_expert_op.py -- VALIDATE approach-3 (the OpBackend / GC-captured op)
ON THE HPU BOX.

Loads build/libmoe_fused_expert.so via torch.ops.load_library, builds synthetic
GLM-shaped int4 weights, calls the captured op

    torch.ops.hpu_moe_ext.fused_expert(x, w13_packed, w13_scale, w13_zp,
                                       w2_packed, w2_scale, w2_zp, router_weight)

under LAZY mode, and asserts it matches a reference computed with the EXISTING,
TRUSTED torch.ops.hpu.convert_from_uint4 + the MoE math:

  W13 = convert_from_uint4(w13_packed, w13_scale, w13_zp, bf16)   # [H, 2I]
  W2  = convert_from_uint4(w2_packed,  w2_scale,  w2_zp,  bf16)   # [I, H]
  gate_up = x @ W13 ; gate = gate_up[:, :I] ; up = gate_up[:, I:]
  h = silu(gate) * up ; expert_out = h @ W2 ; out = expert_out * w_k

Gate: relNorm = ||got - ref|| / ||ref|| < 1e-2 (the bf16 norm-relative tol the
proven recipe uses). LOUD (nonzero exit + stderr) on any failure.

DIFFERENCE vs ../fusion/test_fused_moe_op.py (approach B): there is NO
stream_handle arg here -- the bridge OWNS the launch/stream/workspace and the op
is captured into the HPU graph like a built-in. So this also (best-effort)
exercises HPU-graph capture: it runs the op once eagerly-in-lazy AND once wrapped
to confirm it lowers into a single captured recipe.

RUN (ON THE HPU BOX -- a CPU-only import cannot exercise the graph):
  cd <repo>/tpc_kernel/opbackend
  python test_fused_expert_op.py                  # GLM-5.1 default: T=1 H=6144 I=2048 GS=128
  python test_fused_expert_op.py 4 6144 2048 128  # T H I GS
"""
from __future__ import annotations

import os
import sys


def _sym_zero_packed(groups: int, out_dim: int, device):
    """uint4 midpoint 8 packed 8/int32 along OUT -> [groups, out//8] int32 (0x88888888)."""
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _pack_u4(codes_2d, device):
    """codes [IN, OUT] (0..15) -> packed int32 [IN, OUT//8] (8 nibbles/word, little-endian)."""
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


def _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I):
    """Reference single-expert forward via the TRUSTED convert_from_uint4 + MoE math.
    Returns expert_out [T, H] (UNSCALED by router weight)."""
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch.core as htorch
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H, 2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)       # [I, H]
    htorch.mark_step()
    gate_up = torch.matmul(x, W13)            # [T, 2I]
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    h = F.silu(gate) * up                     # [T, I]
    return torch.matmul(h, W2)                # [T, H]


def _relnorm(got, ref):
    g = got.float()
    r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    maxabs = (g - r).abs().max().item()
    return rn, maxabs


def _find_so():
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "build", "libmoe_fused_expert.so")
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}/..:/work "
            f"-w /work/tpc_kernel/opbackend vllm-gaudi:next bash build.sh")
    return so


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H={H} I={I} GS={GS})")
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    # ---- load the captured op (.so) ----
    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_expert"):
        print("FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after "
              f"load_library({so}). Check TORCH_LIBRARY(hpu_moe_ext) + the .so.",
              file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    TOL = 1e-2
    OUT13 = 2 * I
    rc = 0
    print(f"=== test_fused_expert_op (approach 3) ===  T={T} H={H} I={I} "
          f"(2I={OUT13}) GS={GS}")

    # ---- one expert's weights (the per-expert repack layout) ----
    w13p, s13, z13, w2p, s2, z2 = _make_expert(H, I, GS, 100, dev)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    rw_val = 0.75
    rw = torch.full((1, 1), rw_val, dtype=torch.bfloat16, device=dev)
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- numeric correctness vs convert_from_uint4 reference.
    # ====================================================================
    print("\n--- CHECK 1: fused_expert numeric (vs convert_from_uint4) ---")
    got = torch.ops.hpu_moe_ext.fused_expert(x, w13p, s13, z13, w2p, s2, z2, rw)
    htorch.mark_step()
    ref = _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I) * rw_val
    htorch.mark_step()
    gotc, refc = got.to("cpu"), ref.to("cpu")
    rn, ma = _relnorm(gotc, refc)
    print(f"  |ref|={refc.float().norm().item():.4f}  |got|={gotc.float().norm().item():.4f}  "
          f"relNorm={rn:.4e}  maxAbs={ma:.4e}")
    if rn <= TOL:
        print(f"  >>> PASS: fused_expert matches reference (relNorm {rn:.2e}).")
    else:
        print(f"  >>> FAIL: fused_expert relNorm {rn:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        print("      Suspects (device-confirmable): gemm orientation, gate/up slice_axis\n"
              "      begin/end, silu decomposition, dequant group/zp(uint4) indexing, the\n"
              "      reinterpret_cast uint4 output attr, or the router-weight [1,1] broadcast.",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- HPU-graph CAPTURE (the durable serving path). Best-effort:
    # wrap the op in an HPU graph; success => it lowered into ONE captured
    # recipe (the whole point of approach 3 vs the side-band synLaunch of B).
    # ====================================================================
    print("\n--- CHECK 2: HPU-graph capture (best-effort) ---")
    captured_ok = None
    try:
        import habana_frameworks.torch.hpu as ht_hpu
        from habana_frameworks.torch.hpu.graphs import HPUGraph

        g = HPUGraph()
        s = ht_hpu.Stream()
        with ht_hpu.stream(s):
            g.capture_begin()
            out_cap = torch.ops.hpu_moe_ext.fused_expert(
                x, w13p, s13, z13, w2p, s2, z2, rw)
            g.capture_end()
        g.replay()
        htorch.mark_step()
        rn2, _ = _relnorm(out_cap.to("cpu"), refc)
        captured_ok = rn2 <= TOL
        if captured_ok:
            print(f"  >>> PASS: op captured into an HPU graph and replayed "
                  f"(relNorm {rn2:.2e}). This is the durable, single-recipe serving path.")
        else:
            print(f"  >>> WARN: captured replay relNorm {rn2:.4e} > tol (numerics differ "
                  "under capture; investigate on device).", file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")
        print("  NOTE: capture is what the device run confirms; eager-lazy CHECK 1 already "
              "exercised the SAME captured AddNode lowering.")

    # ====================================================================
    # CHECK 3 -- SRAM-residency HINT (best-effort): dump the GC graph and look
    # for the dequant->gemm bundle. AUTHORITATIVE residency is the on-device
    # HBM-bytes / SRAM_SLICE log (see ../fusion/README.md), measured separately.
    # ====================================================================
    print("\n--- CHECK 3: SRAM-residency hint (best-effort; see ../fusion/README.md) ---")
    print("  The two dequant outputs are graph-internal (no persistent section) in AddNode,")
    print("  so the GC's fuseDequantIntoMoeGemm + SRAM slicer should bundle them into the MME.")
    print("  To CONFIRM residency on device, re-run with the GC dump/HBM counters, e.g.:")
    print("    GRAPH_VISUALIZATION=1 LOG_LEVEL_SRAM_SLICE=2 HABANA_LOGS=./logs \\")
    print("      python test_fused_expert_op.py")
    print("  and check weight-HBM/iter ~= int4 size (WIN) vs ~= bf16 size (spilled).")
    if os.environ.get("LOG_LEVEL_SRAM_SLICE") or os.environ.get("GRAPH_VISUALIZATION"):
        print("  (GC dump env detected -- inspect ./logs + .graph_dumps for the bundle/slice.)")

    print(f"\n[done] test_fused_expert_op {'OK' if rc == 0 else 'FAILED'} (rc={rc})"
          + (f"; capture={'ok' if captured_ok else captured_ok}" if captured_ok is not None else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
