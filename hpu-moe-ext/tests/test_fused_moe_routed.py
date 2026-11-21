"""test_fused_moe_routed.py -- VALIDATE the K-EXPERT routed MoE
(`hpu_moe_ext.fused_expert_routed.fused_moe_routed`) ON THE HPU BOX.

Builds synthetic GLM-shaped int4 weights for E experts (stacked, op layout),
loads the WORKING single-expert op (build/libmoe_fused_expert.so), runs

    fused_moe_routed(x, w13_packed, w13_scale, w13_zp,
                     w2_packed, w2_scale, w2_zp,
                     topk_ids, topk_weights, experts_min, K)

under LAZY mode, and asserts it matches a reference computed with the EXISTING,
TRUSTED torch.ops.hpu.convert_from_uint4 PER EXPERT + the MoE math, summed over
the routed experts:

  for each routed expert e (global id in topk_ids):
    W13 = convert_from_uint4(w13_packed[e_local], w13_scale[e_local], w13_zp[e_local], bf16)  # [H, 2I]
    W2  = convert_from_uint4(w2_packed[e_local],  w2_scale[e_local],  w2_zp[e_local],  bf16)  # [I, H]
    gate_up = x @ W13 ; gate = gate_up[:, :I] ; up = gate_up[:, I:]
    h = silu(gate) * up ; expert_out = h @ W2                                                 # [T, H]
    out += expert_out * router_weight(token, e)

Gate: relNorm = ||got - ref|| / ||ref|| < 1e-2. LOUD (nonzero exit + stderr) on
any failure.

GLM-5.1-int4 default shape: H=6144, I=2048, GS=128, E=8, top_k=8 (so T=1 routes
to all 8 local experts -> exercises the full K=8 single-stream gather/accumulate).

RUN (ON THE HPU BOX -- a CPU import cannot exercise the op):
  cd $HOME/AI/hpu-moe-ext
  python tests/test_fused_moe_routed.py                  # GLM default: T=1 H=6144 I=2048 GS=128 E=8 K=8
  python tests/test_fused_moe_routed.py 1 6144 2048 128 8 8       # T H I GS E K (top_k defaults to E)
  # bs>1 MTP-VERIFY path (the int4-fused bs>1 fix): T H I GS E K TOP_K, TOP_K<E -> distinct routing.
  # Production GLM EP shape (E=32 local, top_k=8); the K=16 bucket covers the bs=4 union (~3.8,max~12):
  python tests/test_fused_moe_routed.py 4 6144 2048 128 32 16 8   # bs=4 verify, K=16 bucket
  python tests/test_fused_moe_routed.py 4 6144 2048 128 32 32 8   # bs=4 verify, K=32 conservative (no-drop)
  # NOTE: this harness drives fused_moe_routed directly; run it under PT_HPU_LAZY_MODE=1 to
  # exercise the CAPTURE-SAFE sequential-op path that the serve's captured bs>1 verify uses,
  # and PT_HPU_LAZY_MODE=0 (eager) to exercise the per-expert loop path. Both must PASS.
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


def _build_stacked(E, H, I, GS, device):
    """Stack E experts into the [E,...] op-ready tensors fused_moe_routed consumes."""
    import torch
    w13p_l, s13_l, z13_l, w2p_l, s2_l, z2_l = [], [], [], [], [], []
    for e in range(E):
        w13p, s13, z13, w2p, s2, z2 = _make_expert(H, I, GS, 100 + e, device)
        w13p_l.append(w13p); s13_l.append(s13); z13_l.append(z13)
        w2p_l.append(w2p); s2_l.append(s2); z2_l.append(z2)
    return (torch.stack(w13p_l, 0), torch.stack(s13_l, 0), torch.stack(z13_l, 0),
            torch.stack(w2p_l, 0), torch.stack(s2_l, 0), torch.stack(z2_l, 0))


def _ref_routed(x, w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s,
                topk_ids, topk_weights, experts_min, I):
    """Reference K-expert routed forward via TRUSTED convert_from_uint4 PER EXPERT
    + MoE math, summed over the routed (LOCAL) experts with per-(token,slot) router
    weights. Returns out [T, H]. Pure dequant-list math -- no gather/topk/remap, so
    it is an INDEPENDENT oracle for fused_moe_routed (which DOES route)."""
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch.core as htorch
    E = w13p_s.shape[0]
    T, H = x.shape
    dt = s13_s.dtype
    emin = int(experts_min)
    out = torch.zeros(T, H, dtype=dt, device=x.device)
    # Dequant every LOCAL expert once (E is small in the test), then sum each
    # token's routed contributions. This avoids re-implementing the routed gather
    # (so a routing bug in fused_moe_routed cannot be masked by a matching bug here).
    W13 = [torch.ops.hpu.convert_from_uint4(w13p_s[e], s13_s[e], z13_s[e], dt, None) for e in range(E)]  # [H,2I]
    W2 = [torch.ops.hpu.convert_from_uint4(w2p_s[e], s2_s[e], z2_s[e], dt, None) for e in range(E)]      # [I,H]
    htorch.mark_step()
    expert_out = []
    for e in range(E):
        gate_up = torch.matmul(x, W13[e])         # [T, 2I]
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        h = F.silu(gate) * up                     # [T, I]
        expert_out.append(torch.matmul(h, W2[e]))  # [T, H]
    # accumulate per (token, top_k slot): only LOCAL experts contribute
    tids = topk_ids.to(torch.int64).cpu()
    tws = topk_weights.to(dt).cpu()
    for t in range(T):
        for slot in range(tids.shape[1]):
            gid = int(tids[t, slot].item())
            le = gid - emin
            if 0 <= le < E:
                out[t] = out[t] + expert_out[le][t] * tws[t, slot].to(x.device)
    return out


def _relnorm(got, ref):
    g = got.float(); r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    maxabs = (g - r).abs().max().item()
    return rn, maxabs


def _find_so():
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "..", "tpc_kernel", "opbackend", "build", "libmoe_fused_expert.so")
    so = os.path.abspath(so)
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build the single-expert op first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}:/work "
            f"-w /work/tpc_kernel/opbackend vllm-gaudi:next bash build.sh")
    return so


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    E = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    K = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    # OPTIONAL 7th arg TOP_K (default E -> the legacy "route to all E local" case). Pass it
    # to decouple top_k from E and exercise the bs>1 MTP-VERIFY path with DISTINCT-per-token
    # routing (the production GLM EP shape: E=32 local, top_k=8). When TOP_K < E each token
    # picks TOP_K DISTINCT random local experts, so at T>1 the batch UNION (<=min(T*top_k,E))
    # is what the K gather must cover -- the exact bs>1 scenario this fix targets.
    TOP_K = int(sys.argv[7]) if len(sys.argv) > 7 else E
    DISTINCT = TOP_K < E            # per-token distinct routing (the real bs>1 union test)

    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0, I%GS==0, (2I)%8==0, H%8==0 (H={H} I={I} GS={GS})")
        return 2
    if TOP_K > E:
        print(f"FATAL: top_k ({TOP_K}) must be <= E ({E}).")
        return 2
    # K must cover the batch union of distinct LOCAL experts. Worst case union <= min(T*top_k, E).
    # The K-bucket deliberately sizes K below T*top_k (betting union<=K); with DISTINCT random
    # routing over E this test draws an ACTUAL union and checks it fits K, so a too-small K is
    # caught here (the sentinel-K remap would otherwise SAFELY drop the overflow -> a relNorm
    # bump, but we want the test to validate the no-drop regime the bucket is sized for).
    if not DISTINCT and T * TOP_K > K:
        print(f"FATAL: non-distinct routing needs T*top_k <= K (T={T} top_k={TOP_K} K={K}); "
              "raise K or lower T so #distinct local experts <= K.")
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    # ---- load the single-expert op (.so) BEFORE importing the wrapper ----
    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}")
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_expert"):
        print("FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after "
              f"load_library({so}).", file=sys.stderr)
        return 2
    print(f"loaded {so}")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hpu_moe_ext.fused_expert_routed import fused_moe_routed

    dev = "hpu"
    torch.manual_seed(0)
    TOL = 1e-2
    print(f"=== test_fused_moe_routed ===  T={T} H={H} I={I} (2I={2*I}) GS={GS} "
          f"E={E} top_k={TOP_K} K={K} routing={'DISTINCT(bs>1)' if DISTINCT else 'all-E'} "
          f"lazy={htorch.is_lazy()}")

    # ---- stacked E experts (op-ready layout) ----
    w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s = _build_stacked(E, H, I, GS, dev)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)

    # ---- routing: random renormalized router weights; experts_min!=0 exercises the
    #      global->local remap. TWO modes:
    #        * legacy (TOP_K==E): every token routes to ALL E local experts (arange).
    #        * DISTINCT (TOP_K<E): each token picks TOP_K DISTINCT random local experts -> the
    #          bs>1 MTP-VERIFY shape. The batch UNION of distinct ids is what K must cover.
    experts_min = 0
    g = torch.Generator().manual_seed(7)
    tw = torch.rand((T, TOP_K), generator=g, dtype=torch.float32)
    tw = (tw / tw.sum(dim=-1, keepdim=True)).to(torch.bfloat16)             # renormalized
    topk_weights = tw.to(dev)
    if DISTINCT:
        # per-token: TOP_K distinct local ids in [0,E), random. Global = +experts_min.
        rows = [torch.randperm(E, generator=g)[:TOP_K] for _ in range(T)]
        local_ids = torch.stack(rows, 0).to(torch.int64)                   # [T, top_k] in [0,E)
        topk_ids = (local_ids + experts_min).contiguous().to(dev)
        union = int(torch.unique(local_ids).numel())
        print(f"  [bs>1] DISTINCT routing: batch union of distinct LOCAL experts = {union} "
              f"(K={K}); {'COVERED (no drop)' if union <= K else 'EXCEEDS K -> sentinel-drop'}")
        if union > K:
            print(f"  NOTE: union {union} > K {K}: the sentinel-K remap will SAFELY DROP the "
                  f"overflow experts (bounded relNorm bump, NOT corruption). To validate the "
                  f"no-drop regime, raise K >= {union}.", file=sys.stderr)
    else:
        topk_ids = (torch.arange(TOP_K, dtype=torch.int64).unsqueeze(0).expand(T, -1)
                    + experts_min).contiguous().to(dev)                    # [T, top_k] global ids
    htorch.mark_step()

    print("\n--- CHECK: fused_moe_routed numeric (vs per-expert convert_from_uint4) ---")
    got = fused_moe_routed(x, w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s,
                           topk_ids, topk_weights, experts_min, K)
    htorch.mark_step()
    ref = _ref_routed(x, w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s,
                      topk_ids, topk_weights, experts_min, I)
    htorch.mark_step()
    gotc, refc = got.to("cpu"), ref.to("cpu")
    rn, ma = _relnorm(gotc, refc)
    print(f"  |ref|={refc.float().norm().item():.4f}  |got|={gotc.float().norm().item():.4f}  "
          f"relNorm={rn:.4e}  maxAbs={ma:.4e}")
    rc = 0
    if rn <= TOL:
        print(f"  >>> PASS: fused_moe_routed matches reference (relNorm {rn:.2e}).")
    else:
        print(f"  >>> FAIL: fused_moe_routed relNorm {rn:.4e} > tol {TOL:.2e}.", file=sys.stderr)
        print("      Suspects (device-confirmable): the routed gather (active/remap/topk),\n"
              "      the per-(token,slot) router-weight w_k (the [1,1] fold at T=1 vs the\n"
              "      [T,1] post-scale at T>1), per-expert slice .clone() (storage offset),\n"
              "      or the underlying fused_expert op (see test_fused_expert_op.py CHECK 1).",
              file=sys.stderr)
        rc = 1

    print(f"\n[done] test_fused_moe_routed {'OK' if rc == 0 else 'FAILED'} (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
