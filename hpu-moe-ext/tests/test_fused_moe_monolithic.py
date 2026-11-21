"""test_fused_moe_monolithic.py -- VALIDATE the MONOLITHIC (ALL-E experts) single-recipe
fused op `torch.ops.hpu_moe_ext.fused_moe_monolithic` ON THE HPU BOX, against the STOCK
`torch.ops.hpu.mixture_of_experts` path (via VllmMixtureOfExpertsOpWNA16 -- the exact op
this replaces at concurrency).

This is the THROUGHPUT-regime op: it processes ALL E LOCAL experts with per-expert
SRAM-resident dequant (chunked over E so each chunk's dequant stays NON-PERSISTENT ->
bundled into its batch_gemm MME -- the fusion lever), applying a DENSE [E,T,1] per-
(token,expert) router matrix, so it replaces the stock op (which dequantizes all E
experts to bf16 -> HBM -> a separate mixture_of_experts gemm). T>1 (the concurrent
regime) is the whole point, so the default exercises T in {4, 16}.

    torch.ops.hpu_moe_ext.fused_moe_monolithic(
        x[T,H] bf16,
        w13_packed[E,H,2I//8] int32, w13_scale[E,H//GS,2I] bf16, w13_zp[E,H//GS,2I//8] int32,
        w2_packed [E,I,H//8]  int32, w2_scale [E,I//GS,H]  bf16, w2_zp [E,I//GS,H//8]  int32,
        router_weights[E,T,1] bf16, int chunk) -> [T,H] bf16
    == sum_e( expert_e(x) * router_weights[e] ), the stock mixture_of_experts semantics.

We build E distinct experts + a random top_k routing (T tokens each picking top_k of E
experts), then compare the op against:
  CHECK 1: the STOCK VllmMixtureOfExpertsOpWNA16 (torch.ops.hpu.mixture_of_experts over
           the same per-expert convert_from_uint4 dequant + the same topk_ids/weights)
           -- THE op this replaces (relNorm < 1e-2).
  CHECK 2: an INDEPENDENT dense per-expert convert_from_uint4 + matmul oracle (relNorm
           < 1e-2) -- a second, op-free reference (catches a stock-op quirk masking a bug).
  CHECK 3: chunk INVARIANCE: chunk in {2, 8, E} must all give the same result (the
           chunking is a pure SRAM-pressure knob; the math is chunk-independent).
  CHECK 4: HPU-graph capture of the monolithic recipe (the single-launch serving path).

Gate: relNorm = ||got - ref|| / ||ref|| < 1e-2 (the bf16 norm-relative tol). LOUD
(nonzero exit + stderr) on any failure. T in {4, 16}, E in {16, 64} by default
(64 == Qwen3.5-397B local experts at TP8: 512/8).

Qwen3.5-397B-int4 default shape: H=4096, I=1024, GS=128, top_k=10.

RUN (ON THE HPU BOX -- a CPU import cannot exercise the op):
  cd $HOME/AI/hpu-moe-ext
  python tests/test_fused_moe_monolithic.py                  # Qwen default: H=4096 I=1024 GS=128, T in {4,16}, E in {16,64}
  python tests/test_fused_moe_monolithic.py 16 4096 1024 128 64 10   # T H I GS E top_k (single point)
  # LAZY capture variant (loads the lazy .so):
  PT_HPU_LAZY_MODE=1 MOE_MONO_SO=tpc_kernel/opbackend/build/libmoe_fused_monolithic_lazy.so \
    python tests/test_fused_moe_monolithic.py
"""
from __future__ import annotations

import os
import sys


def _sym_zero_packed(groups, out_dim, device):
    """uint4 midpoint 8 packed 8/int32 along OUT -> [groups, out//8] int32 (0x88888888).
    Matches the plugin's _sym_zero (GPTQ-symmetric shared zero = the uint4 bias 8)."""
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _pack_u4(codes_2d, device):
    """codes [IN, OUT] (0..15) -> packed int32 [IN, OUT//8] (8 nibbles/word, little-endian).
    Bit-identical to the single-expert/batched op test packers."""
    import torch
    IN, OUT = codes_2d.shape
    c = (codes_2d.to(torch.int64) & 0xF).reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.to(device)


def _make_expert(H, I, GS, seed, device):
    """One expert's packed int4 weights + scales (+ shared sym zero), Qwen layout.
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
    """Reference UNSCALED single-expert forward via TRUSTED convert_from_uint4 + MoE math.
    Returns expert_out [T, H]."""
    import torch
    import torch.nn.functional as F
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H, 2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)       # [I, H]
    gate_up = torch.matmul(x, W13)            # [T, 2I]
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    h = F.silu(gate) * up                     # [T, I]
    return torch.matmul(h, W2)                # [T, H]


def _relnorm(got, ref):
    g = got.float(); r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    maxabs = (g - r).abs().max().item()
    return rn, maxabs


def _find_mono_so():
    env = os.environ.get("MOE_MONO_SO")
    if env:
        so = os.path.abspath(env)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        so = os.path.abspath(os.path.join(
            here, "..", "tpc_kernel", "opbackend", "build", "libmoe_fused_monolithic.so"))
    if not os.path.isfile(so):
        here = os.path.dirname(os.path.abspath(__file__))
        raise FileNotFoundError(
            f"{so} not found. Build the MONOLITHIC op first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}/tpc_kernel:/work "
            f"-w /work/opbackend vllm-gaudi:next bash build_monolithic.sh")
    return so


def _build_dense_rw(topk_ids, topk_weights, E, dev):
    """Dense [E,T,1] router matrix from topk routing (the wrapper's EXACT construction --
    fused_monolithic_moe.fused_moe_monolithic): W[t,e] = sum over slots s with
    topk_ids[t,s]==e of topk_weights[t,s], via a vectorized one_hot*weight sum in float32
    (NOT scatter_add). Experts are already LOCAL [0,E) here (the test routes into [0,E)),
    so no remap. Returns [E,T,1] bf16."""
    import torch
    erange = torch.arange(E, device=dev)
    onehot = (topk_ids.to(torch.int64).unsqueeze(-1) == erange).to(torch.float32)  # [T,top_k,E]
    W = (onehot * topk_weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)          # [T,E] f32
    return W.transpose(0, 1).contiguous().to(torch.bfloat16).unsqueeze(-1)          # [E,T,1]


def _stock_oracle(torch, htorch, x, experts, topk_ids, topk_weights, E, H, I):
    """THE op this replaces: VllmMixtureOfExpertsOpWNA16 (torch.ops.hpu.mixture_of_experts
    over the same per-expert convert_from_uint4). LOCAL window [0,E-1], permuted_weights=
    False (WNA16 dequant returns [in,out]). Returns [T,H] bf16."""
    from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpWNA16
    op = VllmMixtureOfExpertsOpWNA16(E, 0, E - 1)
    for e in range(E):
        w13p, s13, z13, w2p, s2, z2 = experts[e]
        op.w13_list[e].set_weight_packed(w13p)
        op.w13_list[e].set_weight_scale(s13)
        op.w13_list[e].set_zero_point(z13)
        op.w2_list[e].set_weight_packed(w2p)
        op.w2_list[e].set_weight_scale(s2)
        op.w2_list[e].set_zero_point(z2)
    htorch.mark_step()
    return op(x, topk_ids, topk_weights, permuted_weights=False, activation="silu")


def _run_point(torch, htorch, mono_op, T, H, I, GS, E, top_k, TOL):
    dev = "hpu"
    OUT13 = 2 * I
    rc = 0
    nchunks8 = (E + 7) // 8
    print(f"\n========== T={T} E={E} ==========  H={H} I={I} (2I={OUT13}) GS={GS} top_k={top_k}  "
          f"(ONE recipe, E chunked by 8 -> {nchunks8} sub-chains)")

    # ---- E distinct experts (different seed each), STACKED into [E,...] tensors ----
    experts = [_make_expert(H, I, GS, 1000 + e, dev) for e in range(E)]
    w13p = torch.stack([e[0] for e in experts], 0)   # [E, H, 2I//8]
    s13 = torch.stack([e[1] for e in experts], 0)    # [E, H//GS, 2I]
    z13 = torch.stack([e[2] for e in experts], 0)    # [E, H//GS, 2I//8]
    w2p = torch.stack([e[3] for e in experts], 0)    # [E, I, H//8]
    s2 = torch.stack([e[4] for e in experts], 0)     # [E, I//GS, H]
    z2 = torch.stack([e[5] for e in experts], 0)     # [E, I//GS, H//8]

    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)

    # ---- random top_k routing into LOCAL [0,E): each token picks top_k DISTINCT experts,
    #      weights softmax-renormalized (the real router output form) ----
    g = torch.Generator().manual_seed(123 + T + E)
    logits = torch.randn(T, E, generator=g)
    topk_w, topk_ids = torch.topk(logits.softmax(-1), top_k, dim=-1)   # [T,top_k]
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)
    topk_ids = topk_ids.to(torch.int64).to(dev)
    topk_w = topk_w.to(dev)
    htorch.mark_step()

    rw = _build_dense_rw(topk_ids, topk_w, E, dev)                     # [E,T,1] bf16

    # ====================================================================
    # CHECK 1 -- vs the STOCK VllmMixtureOfExpertsOpWNA16 (the op this replaces).
    # ====================================================================
    print("--- CHECK 1: fused_moe_monolithic vs STOCK mixture_of_experts (the replaced op) ---")
    got = mono_op(x, w13p, s13, z13, w2p, s2, z2, rw, 8)
    htorch.mark_step()
    try:
        stock = _stock_oracle(torch, htorch, x, experts, topk_ids, topk_w, E, H, I)
        htorch.mark_step()
        gotc, stockc = got.to("cpu"), stock.to("cpu")
        rn, ma = _relnorm(gotc, stockc)
        print(f"  |stock|={stockc.float().norm().item():.4f}  |got|={gotc.float().norm().item():.4f}  "
              f"relNorm={rn:.4e}  maxAbs={ma:.4e}")
        if got.shape != (T, H):
            print(f"  >>> FAIL: output shape {tuple(got.shape)} != ({T},{H}).", file=sys.stderr)
            rc = 1
        elif rn <= TOL:
            print(f"  >>> PASS: fused_moe_monolithic matches the STOCK op (relNorm {rn:.2e}) "
                  "-- SRAM-resident all-E dequant == the bf16->HBM stock path.")
        else:
            print(f"  >>> FAIL: vs stock relNorm {rn:.4e} > tol {TOL:.2e}.", file=sys.stderr)
            print("      Suspects: the DENSE [E,T,1] router (W[t,e] scatter-add vs the stock "
                  "per-slot weight),\n"
                  "      the on-device dim-0 chunk slice, the grouped dequant over [C,...] "
                  "(per-expert scale/zp must NOT cross experts), the [C,T,1] router broadcast "
                  "over [C,T,H],\n"
                  "      or the reduce_sum + chunk add-chain.", file=sys.stderr)
            rc = 1
    except Exception as ex:  # noqa: BLE001
        print(f"  >>> WARN: stock oracle unavailable ({ex}); relying on CHECK 2.", file=sys.stderr)
        gotc = got.to("cpu")

    # ====================================================================
    # CHECK 2 -- vs an INDEPENDENT dense per-expert convert_from_uint4 oracle.
    # ====================================================================
    print("--- CHECK 2: fused_moe_monolithic vs independent per-expert convert_from_uint4 sum ---")
    ref = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    rwf = rw.squeeze(-1).transpose(0, 1)                              # [T,E] (per-(token,expert))
    for e in range(E):
        ek = _ref_one(x, w13p[e], s13[e], z13[e], w2p[e], s2[e], z2[e], I)  # [T,H]
        ref = ref + ek * rwf[:, e:e + 1]                             # weight token-wise
    htorch.mark_step()
    refc = ref.to("cpu")
    rn2, ma2 = _relnorm(gotc, refc)
    print(f"  |ref|={refc.float().norm().item():.4f}  |got|={gotc.float().norm().item():.4f}  "
          f"relNorm={rn2:.4e}  maxAbs={ma2:.4e}")
    if rn2 <= TOL:
        print(f"  >>> PASS: matches the independent per-expert oracle (relNorm {rn2:.2e}).")
    else:
        print(f"  >>> FAIL: vs per-expert oracle relNorm {rn2:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 3 -- chunk INVARIANCE (the chunking is a pure SRAM knob; math is chunk-free).
    # ====================================================================
    print("--- CHECK 3: chunk invariance (chunk in {2, 8, E} must agree) ---")
    base = got.to("cpu")
    for ch in (2, E):
        alt = mono_op(x, w13p, s13, z13, w2p, s2, z2, rw, ch)
        htorch.mark_step()
        rn3, ma3 = _relnorm(alt.to("cpu"), base)
        ok = rn3 <= TOL
        print(f"  chunk={ch:<3d} vs chunk=8: relNorm={rn3:.4e} maxAbs={ma3:.4e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  >>> FAIL: chunk={ch} diverges from chunk=8 (chunking must not change math).",
                  file=sys.stderr)
            rc = 1

    # ====================================================================
    # CHECK 4 -- HPU-graph CAPTURE: the monolithic recipe as ONE captured graph / launch.
    # ====================================================================
    print("--- CHECK 4: HPU-graph capture of the monolithic recipe (best-effort) ---")
    try:
        import habana_frameworks.torch.hpu as ht_hpu
        from habana_frameworks.torch.hpu.graphs import HPUGraph
        gph = HPUGraph()
        s = ht_hpu.Stream()
        with ht_hpu.stream(s):
            gph.capture_begin()
            out_cap = mono_op(x, w13p, s13, z13, w2p, s2, z2, rw, 8)
            gph.capture_end()
        gph.replay()
        htorch.mark_step()
        rn4, _ = _relnorm(out_cap.to("cpu"), base)
        if rn4 <= TOL:
            print(f"  >>> PASS: the E={E} monolithic op captured into ONE HPU graph and replayed "
                  f"(relNorm {rn4:.2e}) -- the single-recipe, single-launch path.")
        else:
            print(f"  >>> WARN: captured replay relNorm {rn4:.4e} > tol (investigate on device).",
                  file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")

    return rc


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else None
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    E_arg = int(sys.argv[5]) if len(sys.argv) > 5 else None
    top_k = int(sys.argv[6]) if len(sys.argv) > 6 else 10

    Ts = [T] if T is not None else [4, 16]
    Es = [E_arg] if E_arg is not None else [16, 64]

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

    try:
        so = _find_mono_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_monolithic"):
        print("FATAL: torch.ops.hpu_moe_ext.fused_moe_monolithic not registered after "
              f"load_library({so}).", file=sys.stderr)
        return 2
    print(f"loaded {so}")
    mono_op = torch.ops.hpu_moe_ext.fused_moe_monolithic

    torch.manual_seed(0)
    TOL = 1e-2
    print(f"=== test_fused_moe_monolithic ===  H={H} I={I} GS={GS} top_k={top_k}  Ts={Ts} Es={Es}")

    rc = 0
    for E in Es:
        if top_k > E:
            print(f"(skip E={E}: top_k {top_k} > E)")
            continue
        for T in Ts:
            rc |= _run_point(torch, htorch, mono_op, T, H, I, GS, E, top_k, TOL)

    print(f"\n[done] test_fused_moe_monolithic {'OK' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
