"""test_fused_moe_op.py -- VALIDATE the fused-MoE bridge ops ON THE HPU BOX.

Loads build/libfused_moe_op.so (via fused_moe_op.load), builds synthetic GLM-shaped
int4 weights + routing, and asserts BOTH ops match a reference computed with the
EXISTING, TRUSTED torch.ops.hpu.convert_from_uint4 + the MoE math:

  W13 = convert_from_uint4(w13_packed, w13_scale, w13_zp, bf16)   # [H, 2I]
  W2  = convert_from_uint4(w2_packed,  w2_scale,  w2_zp,  bf16)   # [I, H]
  gate_up = x @ W13 ; gate = gate_up[:, :I] ; up = gate_up[:, I:]
  h = silu(gate) * up ; expert_out = h @ W2 ; out = sum_k expert_out_k * w_k

Two checks:
  (1) fused_expert  -- ONE expert (the BRIDGE PROOF), rw = a scalar w_k.
  (2) fused_moe     -- K routed experts, the full routed sum.

Gate: relNorm = ||got - ref|| / ||ref|| < 1e-2 (the bf16 norm-relative tol the proven
recipe uses). LOUD (nonzero exit + stderr) on any failure.

RUN (ON THE HPU BOX -- a CPU-only import cannot exercise the graph):
  cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
  python test_fused_moe_op.py                 # GLM single-stream default: T=1 H=4096 I=1408 GS=128 E=8 K=8
  python test_fused_moe_op.py 1 6144 2048 128 16 8   # T H I GS E K  (e.g. GLM-5.1 H=6144 I=2048)
"""
from __future__ import annotations

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
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H, 2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)       # [I, H]
    import habana_frameworks.torch.core as htorch
    htorch.mark_step()
    gate_up = torch.matmul(x, W13)            # [T, 2I]
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    h = F.silu(gate) * up                     # [T, I]
    return torch.matmul(h, W2)                # [T, H]


def _relnorm(got, ref):
    import torch
    g = got.float()
    r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    maxabs = (g - r).abs().max().item()
    return rn, maxabs


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 1408
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    E = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 8

    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H={H} I={I} GS={GS})")
        return 2
    if T * K < T:  # nonsense guard
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    import fused_moe_op
    try:
        fused_moe_op.load()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2

    dev = "hpu"
    torch.manual_seed(0)
    TOL = 1e-2
    OUT13 = 2 * I
    rc = 0

    print(f"=== test_fused_moe_op ===  T={T} H={H} I={I} (2I={OUT13}) GS={GS} E={E} K={K}")

    # ---- build E experts' weights, stacked [E, ...] (the vLLM repack layout) ----
    w13p_l, s13_l, z13_l, w2p_l, s2_l, z2_l = [], [], [], [], [], []
    for e in range(E):
        a, b, c, d, f, h = _make_expert(H, I, GS, 100 + e, dev)
        w13p_l.append(a); s13_l.append(b); z13_l.append(c)
        w2p_l.append(d); s2_l.append(f); z2_l.append(h)
    w13_packed = torch.stack(w13p_l)        # [E, H, 2I//8]
    w13_scale = torch.stack(s13_l)          # [E, H//GS, 2I]
    w13_zp = z13_l[0]                        # shared sym zero [H//GS, 2I//8]
    w2_packed = torch.stack(w2p_l)          # [E, I, H//8]
    w2_scale = torch.stack(s2_l)            # [E, I//GS, H]
    w2_zp = z2_l[0]                         # shared sym zero [I//GS, H//8]

    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    import habana_frameworks.torch.core as htorch
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- fused_expert (the BRIDGE PROOF): one expert, scalar rw.
    # ====================================================================
    print("\n--- CHECK 1: fused_expert (single expert) ---")
    e0 = 0
    rw_val = 0.75
    rw = torch.full((1, 1), rw_val, dtype=torch.bfloat16, device=dev)
    sh = fused_moe_op._current_stream_handle()
    # contiguous per-expert slices (clone -> fresh storage_offset=0, lazy-safe)
    w13p0 = w13_packed[e0].clone().contiguous()
    s130 = w13_scale[e0].clone().contiguous()
    w2p0 = w2_packed[e0].clone().contiguous()
    s20 = w2_scale[e0].clone().contiguous()
    htorch.mark_step()
    got1 = torch.ops.hpu_moe_ext.fused_expert(
        x, w13p0, s130, w13_zp, w2p0, s20, w2_zp, rw, sh)
    htorch.mark_step()
    ref1 = _ref_expert(x, w13p0, s130, w13_zp, w2p0, s20, w2_zp, I) * rw_val
    htorch.mark_step()
    got1c, ref1c = got1.to("cpu"), ref1.to("cpu")
    rn1, ma1 = _relnorm(got1c, ref1c)
    print(f"  |ref|={ref1c.float().norm().item():.4f}  |got|={got1c.float().norm().item():.4f}  "
          f"relNorm={rn1:.4e}  maxAbs={ma1:.4e}")
    if rn1 <= TOL:
        print(f"  >>> PASS: fused_expert matches convert_from_uint4 reference (relNorm {rn1:.2e}).")
    else:
        print(f"  >>> FAIL: fused_expert relNorm {rn1:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        print("      Suspects: tensor-addr binding (data_ptr->graph FCD), gemm orientation,\n"
              "      gate/up slice, silu decomposition, dequant group/zp indexing, or rw broadcast.",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- fused_moe (K routed experts): the full routed sum.
    # ====================================================================
    print("\n--- CHECK 2: fused_moe (K routed experts) ---")
    experts_min = 0
    top_k = min(K, E)
    # route T tokens to top_k DISTINCT experts each (T*top_k <= K guaranteed for T==1).
    # Build deterministic topk_ids / weights. For T==1 pick the first top_k experts.
    g = torch.Generator().manual_seed(7)
    topk_ids_l, topk_w_l = [], []
    for t in range(T):
        perm = torch.randperm(E, generator=g)[:top_k]
        topk_ids_l.append(perm)
        w = torch.rand(top_k, generator=g) + 0.1
        w = w / w.sum()
        topk_w_l.append(w)
    topk_ids = torch.stack(topk_ids_l).to(torch.int64).to(dev)         # [T, top_k] GLOBAL ids
    topk_weights = torch.stack(topk_w_l).to(torch.bfloat16).to(dev)    # [T, top_k]
    htorch.mark_step()

    got2 = torch.ops.hpu_moe_ext.fused_moe(
        x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
        topk_ids, topk_weights, experts_min, K)
    htorch.mark_step()

    # reference: sum over the routed (expert, weight) pairs via convert_from_uint4 math.
    ref2 = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    for t in range(T):
        for s in range(top_k):
            e = int(topk_ids[t, s].item()) - experts_min
            if e < 0 or e >= E:
                continue
            wk = topk_weights[t, s].float().item()
            eo = _ref_expert(
                x[t:t + 1],
                w13_packed[e].clone().contiguous(), w13_scale[e].clone().contiguous(), w13_zp,
                w2_packed[e].clone().contiguous(), w2_scale[e].clone().contiguous(), w2_zp, I)
            htorch.mark_step()
            ref2[t:t + 1] = ref2[t:t + 1] + eo.to(torch.bfloat16) * wk
    htorch.mark_step()
    got2c, ref2c = got2.to("cpu"), ref2.to("cpu")
    rn2, ma2 = _relnorm(got2c, ref2c)
    print(f"  |ref|={ref2c.float().norm().item():.4f}  |got|={got2c.float().norm().item():.4f}  "
          f"relNorm={rn2:.4e}  maxAbs={ma2:.4e}")
    if rn2 <= TOL:
        print(f"  >>> PASS: fused_moe matches the routed convert_from_uint4 reference (relNorm {rn2:.2e}).")
    else:
        print(f"  >>> FAIL: fused_moe relNorm {rn2:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        print("      Suspects: K-expert accumulate, routing gather (active/remap/slot_k),\n"
              "      per-(token,slot) router weight, or the per-expert graph (see CHECK 1).",
              file=sys.stderr)
        rc = 1

    print(f"\n[done] test_fused_moe_op {'OK' if rc == 0 else 'FAILED'} (rc={rc}).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
