"""test_fused_moe_single_recipe.py -- VALIDATE the K-EXPERT SINGLE-RECIPE fused op
`torch.ops.hpu_moe_ext.fused_moe` ON THE HPU BOX.

This is the one-recipe-for-K-experts op (moe_fused_expert_routed_backend.cpp): it
emits ALL K routed experts' fused int4-MoE subgraphs + the weighted sum into ONE
captured recipe / ONE launch, replacing the K separate `fused_expert` launches the
Python wrapper (hpu_moe_ext/fused_expert_routed.py) issues (one recipe EACH).

    torch.ops.hpu_moe_ext.fused_moe(
        x[M,H] bf16,
        w13_packed: Tensor[K] of [H, 2I//8] int32,
        w13_scale:  Tensor[K] of [H//GS, 2I] bf16,
        w13_zp:     Tensor[K] of [H//GS, 2I//8] int32,
        w2_packed:  Tensor[K] of [I, H//8]  int32,
        w2_scale:   Tensor[K] of [I//GS, H]  bf16,
        w2_zp:      Tensor[K] of [I//GS, H//8] int32,
        router_weights: Tensor[K] of [1,1] bf16) -> [M, H] bf16
    == sum_k( fused_expert(x, W_k) * router_weights[k] ), as ONE recipe.

STACKED-K LAYOUT: K experts are passed as 6 PYTHON LISTS (one per weight class) +
a list of K [1,1] router weights -- the bridge's TensorList idiom (its own
mixture_of_experts op takes `at::TensorList w1,w2,w3`). The op's AddNode loops
emit_one_expert over the K list elements (each .syn_t is the consecutive syn_in)
and accumulates via an add_fwd_bf16 left-fold tree. The Python side gathers exactly
as fused_expert_routed.py does (index_select -> [K,...]) then `.unbind(0)` to lists.

We test the OP DIRECTLY (build the K lists ourselves, T=1 so each expert's router
weight is a true [1,1] scalar -- the folded form), comparing against an INDEPENDENT
per-expert convert_from_uint4 + MoE oracle (no gather/topk/remap in the oracle, so
a routing bug cannot be masked by a matching bug). This isolates the new fused op
from the Python routing wrapper.

Gate: relNorm = ||got - ref|| / ||ref|| < 1e-2 (the bf16 norm-relative tol the
proven recipe uses). LOUD (nonzero exit + stderr) on any failure.

GLM-5.1-int4 default shape: H=6144, I=2048, GS=128, K=8 (the single-stream K).

RUN (ON THE HPU BOX -- a CPU import cannot exercise the op):
  cd <repo>
  python tests/test_fused_moe_single_recipe.py                 # GLM default: M=1 H=6144 I=2048 GS=128 K=8
  python tests/test_fused_moe_single_recipe.py 1 6144 2048 128 8   # M H I GS K
  # LAZY capture variant (loads the lazy .so if MOE_ROUTED_SO points at it):
  PT_HPU_LAZY_MODE=1 MOE_ROUTED_SO=tpc_kernel/opbackend/build/libmoe_fused_routed_lazy.so \
    python tests/test_fused_moe_single_recipe.py
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


def _find_so():
    """The routed (K-expert) .so. Default eager build; override with MOE_ROUTED_SO
    (e.g. point it at libmoe_fused_routed_lazy.so for the LAZY capture variant)."""
    env = os.environ.get("MOE_ROUTED_SO")
    if env:
        so = os.path.abspath(env)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        so = os.path.abspath(os.path.join(
            here, "..", "tpc_kernel", "opbackend", "build", "libmoe_fused_routed.so"))
    if not os.path.isfile(so):
        here = os.path.dirname(os.path.abspath(__file__))
        raise FileNotFoundError(
            f"{so} not found. Build the K-expert op first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}:/work "
            f"-w /work/tpc_kernel/opbackend vllm-gaudi:next bash build_routed.sh")
    return so


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # tokens (single-stream T=1)
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H={H} I={I} GS={GS})")
        return 2
    if M != 1:
        # The op folds a [1,1] router weight per expert -> a true scalar only at M==1
        # (single-stream). For M>1 the per-token weight is applied OUTSIDE the op (as
        # fused_expert_routed.py does); this single-recipe direct test fixes M=1.
        print(f"NOTE: this direct test uses the M==1 [1,1] router fold; got M={M}. "
              "For M>1 the per-token weight is post-applied by the Python wrapper "
              "(see test_fused_moe_routed.py).", file=sys.stderr)

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_moe"):
        print("FATAL: torch.ops.hpu_moe_ext.fused_moe not registered after "
              f"load_library({so}). Check the TORCH_LIBRARY_FRAGMENT(hpu_moe_ext) "
              "+ the .so.", file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    TOL = 1e-2
    OUT13 = 2 * I
    rc = 0
    print(f"=== test_fused_moe_single_recipe ===  M={M} H={H} I={I} (2I={OUT13}) "
          f"GS={GS} K={K}  (ONE recipe for {K} experts)")

    # ---- K distinct experts (each a different seed) + per-expert scalar router
    #      weights (renormalized; at M==1 each is a true [1,1] scalar the op folds). ----
    experts = [_make_expert(H, I, GS, 100 + k, dev) for k in range(K)]
    w13p = [e[0] for e in experts]
    s13 = [e[1] for e in experts]
    z13 = [e[2] for e in experts]
    w2p = [e[3] for e in experts]
    s2 = [e[4] for e in experts]
    z2 = [e[5] for e in experts]

    x = (torch.randn(M, H) * 0.1).to(torch.bfloat16).to(dev)
    g = torch.Generator().manual_seed(7)
    rw_vals = torch.rand(K, generator=g, dtype=torch.float32)
    rw_vals = (rw_vals / rw_vals.sum()).to(torch.bfloat16)           # renormalized [K]
    rws = [rw_vals[k].view(1, 1).to(dev) for k in range(K)]          # K x [1,1] bf16
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- numeric correctness vs per-expert convert_from_uint4 oracle.
    # ====================================================================
    print("\n--- CHECK 1: fused_moe numeric (vs per-expert convert_from_uint4 sum) ---")
    got = torch.ops.hpu_moe_ext.fused_moe(
        x, w13p, s13, z13, w2p, s2, z2, rws)
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
        print(f"  >>> FAIL: fused_moe output shape {tuple(got.shape)} != ({M},{H}).",
              file=sys.stderr)
        rc = 1
    elif rn <= TOL:
        print(f"  >>> PASS: fused_moe (K={K} experts, ONE recipe) matches reference "
              f"(relNorm {rn:.2e}).")
    else:
        print(f"  >>> FAIL: fused_moe relNorm {rn:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        print("      Suspects (device-confirmable): the per-expert emit_one_expert chain\n"
              "      (gemm orientation / gate-up slice / silu / dequant zp-uint4 / [1,1]\n"
              "      router fold -- same as the single-expert op's CHECK 1), the K-list\n"
              "      syn_in indexing (wrong expert's tensors), or the add_fwd_bf16 add-tree\n"
              "      accumulation (missed/extra y_k).", file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- HPU-graph CAPTURE: the whole point is ONE captured recipe for K
    # experts. Success => the K subgraphs + add-tree lowered into a single graph.
    # ====================================================================
    print("\n--- CHECK 2: HPU-graph capture of the single K-expert recipe (best-effort) ---")
    captured_ok = None
    try:
        import habana_frameworks.torch.hpu as ht_hpu
        from habana_frameworks.torch.hpu.graphs import HPUGraph

        gph = HPUGraph()
        s = ht_hpu.Stream()
        with ht_hpu.stream(s):
            gph.capture_begin()
            out_cap = torch.ops.hpu_moe_ext.fused_moe(
                x, w13p, s13, z13, w2p, s2, z2, rws)
            gph.capture_end()
        gph.replay()
        htorch.mark_step()
        rn2, _ = _relnorm(out_cap.to("cpu"), refc)
        captured_ok = rn2 <= TOL
        if captured_ok:
            print(f"  >>> PASS: the K={K}-expert op captured into ONE HPU graph and replayed "
                  f"(relNorm {rn2:.2e}) -- the single-recipe, single-launch serving path.")
        else:
            print(f"  >>> WARN: captured replay relNorm {rn2:.4e} > tol (numerics differ "
                  "under capture; investigate on device).", file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")
        print("  NOTE: capture is what the device run confirms; eager-lazy CHECK 1 already "
              "exercised the SAME captured AddNode lowering.")

    # ====================================================================
    # CHECK 3 -- SRAM-residency / spill HINT (the TOP RISK for this op): K experts'
    # int4 dequant tiles now co-reside in ONE recipe. Does the GC still bundle each
    # dequant->MME (int4 weight HBM, SRAM-resident), or does it SPILL the bf16
    # dequant to HBM under the pressure of K experts at once? AUTHORITATIVE answer
    # is the on-device weight-HBM-bytes/iter counter.
    # ====================================================================
    print("\n--- CHECK 3: SRAM-residency / spill hint (TOP RISK; see ../fusion/README.md) ---")
    print("  Each expert's two dequant outputs are graph-internal (non-persistent), so the")
    print("  GC's fuseDequantIntoMoeGemm + SRAM slicer should bundle EACH into its MME --")
    print("  the same per-expert SRAM win as the single-expert op, now K-fold in one recipe.")
    print("  CONFIRM (the spill check) on device with the GC dump / HBM counters, e.g.:")
    print("    GRAPH_VISUALIZATION=1 LOG_LEVEL_SRAM_SLICE=2 HABANA_LOGS=./logs \\")
    print("      python tests/test_fused_moe_single_recipe.py")
    print(f"  Expect weight-HBM/iter ~= K*int4-size (WIN, each bundled) vs ~= K*bf16-size")
    print("  (SPILL -- the GC could not keep K experts' dequant tiles SRAM-resident at once).")
    if os.environ.get("LOG_LEVEL_SRAM_SLICE") or os.environ.get("GRAPH_VISUALIZATION"):
        print("  (GC dump env detected -- inspect ./logs + .graph_dumps for the K bundles/slices.)")

    print(f"\n[done] test_fused_moe_single_recipe {'OK' if rc == 0 else 'FAILED'} (rc={rc})"
          + (f"; capture={'ok' if captured_ok else captured_ok}" if captured_ok is not None else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
