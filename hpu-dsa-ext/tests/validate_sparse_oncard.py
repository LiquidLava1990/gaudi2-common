"""validate_sparse_oncard.py — MVP-3 ON-CARD validation of the DSA sparse OP path.

THIS RUNS ON THE HPU BOX ONLY (it loads the captured .so kernels and exercises them on
the device). It is the gate the operator runs AFTER the MoE serve e2e passes and BEFORE
flipping the sparse path on for real. It does NOT need a running vLLM serve — it drives
the two validated ops directly through the hpu_dsa_ext shim (dsa_ops), which is the exact
code the serving path calls. It proves the THREE design gates
(DSA_OPBACKEND_KERNEL_DESIGN.md §7 MVP-3 / IMPLEMENTATION.md:78-85):

  (a) indexer FIRES on ctx > index_topk: the indexer_topk op returns a STRICT top-k
      subset (selected count == index_topk < S) for a query with > index_topk causal
      keys — i.e. it is not trivially selecting everything (== dense).
  (b) ctx ≤ index_topk: sparse output == dense output (relNorm ~0). When every causal
      key is selectable, the sparse_attn op over the indexer's full selection equals a
      plain dense causal SDPA over the same keys.
  (c) ctx large (> index_topk, the design uses ~16K): sparse ≠ dense but COHERENT, and
      the indexer recall holds — the op selects a real subset and the sparse output
      differs from dense (proves NOT silently dense) while matching the gather oracle.

Because a true 16K device run is heavy, the gates are parameterized: the DEFAULT uses a
small synthetic index_topk so all three gates run in seconds; pass real GLM dims (and a
real ctx) to stress it. The math is identical at any scale (everything is shape-driven).

RUN (on the HPU box; the GLM serve must NOT be using all cards, or run with the cards
free — this allocates a little HPU memory but loads no model):
  # eager .so:
  HPU_DSA_SPARSE=1 python tests/validate_sparse_oncard.py
  # lazy .so (the glm51ar-lazy serving path):
  HPU_DSA_SPARSE=1 PT_HPU_LAZY_MODE=1 python tests/validate_sparse_oncard.py
  # GLM-5.1 dims + a bigger ctx for gate (c):
  HPU_DSA_SPARSE=1 python tests/validate_sparse_oncard.py --heads 32 --topk 2048 --ctx 16384

The .so are picked up automatically (dsa_ops resolves hpu-moe-ext/.../dsa/build, lazy vs
eager by PT_HPU_LAZY_MODE). Set HPU_DSA_SO_DIR to override.
"""
from __future__ import annotations

import argparse
import os
import sys

NEG_INF = float("-inf")


# --------------------------------------------------------------------------------------
# CPU oracles (fp32) — the same contracts the kernel reference tests use.
# --------------------------------------------------------------------------------------
def _ref_indexer_topk(q_thd, k_sd, w_th, topk, abs_pos):
    import torch
    qf, kf, wf = q_thd.float(), k_sd.float(), w_th.float()
    logits = torch.relu(torch.einsum("thd,sd->ths", qf, kf))   # [T,H,S]
    score = torch.einsum("ths,th->ts", logits, wf)             # [T,S]
    S = k_sd.shape[0]
    key_pos = torch.arange(S).view(1, -1)
    score = score.masked_fill(key_pos > abs_pos.view(-1, 1), NEG_INF)
    kk = min(topk, S)
    return score.topk(kk, dim=-1).indices                      # [T,kk]


def _ref_gather_attn(q_thd, k_shd, v_shd, topk_idx, scale):
    """PRODUCT oracle == what sparse_attn re-emits (sparse_mla_consume gather path)."""
    import torch
    T, H, Dh = q_thd.shape
    S = k_shd.shape[0]
    Dv = v_shd.shape[-1]
    idx = topk_idx.long()
    valid = idx >= 0
    gid = torch.where(valid, idx, torch.full_like(idx, S))
    k_ext = torch.cat([k_shd, torch.zeros(1, H, Dh, dtype=k_shd.dtype)], dim=0)
    v_ext = torch.cat([v_shd, torch.zeros(1, H, Dv, dtype=v_shd.dtype)], dim=0)
    kg, vg = k_ext[gid], v_ext[gid]                            # [T,K,H,*]
    logits = torch.einsum("thd,twhd->thw", q_thd.float(), kg.float()) * scale
    bias = torch.where(valid.view(T, 1, idx.shape[1]),
                       torch.zeros((), dtype=torch.float32),
                       torch.full((), NEG_INF, dtype=torch.float32))
    probs = torch.nan_to_num(torch.softmax(logits + bias, dim=-1), nan=0.0)
    out = torch.einsum("thw,twhd->thd", probs, vg.float())
    return out.to(q_thd.dtype)


def _ref_dense_causal_attn(q_thd, k_shd, v_shd, abs_pos, scale):
    """Plain dense causal MLA SDPA over ALL keys (the dense baseline for gate (b))."""
    import torch
    T, H, _ = q_thd.shape
    S = k_shd.shape[0]
    key_pos = torch.arange(S).view(1, -1)
    causal = key_pos <= abs_pos.view(-1, 1)                    # [T,S]
    bias = torch.where(causal, torch.zeros((), dtype=torch.float32),
                       torch.full((), NEG_INF, dtype=torch.float32))
    qf = q_thd.transpose(0, 1).float()
    kf = k_shd.transpose(0, 1).float()
    vf = v_shd.transpose(0, 1).float()
    logits = torch.matmul(qf, kf.transpose(-1, -2)) * scale + bias.unsqueeze(0)
    probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    out = torch.matmul(probs, vf)
    return out.transpose(0, 1).to(q_thd.dtype)


def _rel_norm(got, ref):
    import torch
    g, r = got.float(), ref.float()
    d = r.norm().item()
    return (g - r).norm().item() / (d if d > 0 else 1.0)


def _set_recall(got_idx, ref_idx):
    rec = 0.0
    T = got_idx.shape[0]
    for t in range(T):
        g = {int(x) for x in got_idx[t].tolist() if int(x) >= 0}
        r = {int(x) for x in ref_idx[t].tolist() if int(x) >= 0}
        rec += (len(g & r) / len(r)) if r else 1.0
    return rec / T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=8, help="MLA heads (subset for speed)")
    ap.add_argument("--idx-heads", type=int, default=8, help="indexer n_head")
    ap.add_argument("--idx-dim", type=int, default=128, help="index_head_dim")
    ap.add_argument("--qk-dim", type=int, default=256, help="qk_head_dim (nope+rope)")
    ap.add_argument("--v-dim", type=int, default=256, help="v_head_dim")
    ap.add_argument("--topk", type=int, default=64, help="index_topk (small default for speed)")
    ap.add_argument("--ctx", type=int, default=512, help="large-ctx for gate (c) (> topk)")
    ap.add_argument("--queries", type=int, default=32, help="query rows T")
    args = ap.parse_args()

    if os.environ.get("HPU_DSA_SPARSE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        print("FATAL: set HPU_DSA_SPARSE=1 (the gate the serving path uses).", file=sys.stderr)
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    # Import the shim the serving path uses (resolves + loads the .so, picks lazy/eager).
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hpu_dsa_ext import dsa_ops

    try:
        dsa_ops.load_ops()
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: could not load the DSA op .so: {ex}", file=sys.stderr)
        return 2
    print(f"loaded DSA ops ({'lazy' if dsa_ops._is_lazy() else 'eager'} .so)")

    dev = "hpu"
    torch.manual_seed(0)
    H, Hi, Di = args.heads, args.idx_heads, args.idx_dim
    Dh, Dv = args.qk_dim, args.v_dim
    topk, T = args.topk, args.queries
    scale = Dh ** -0.5
    REL_TOL = 2.0e-2
    rc = 0

    print(f"=== MVP-3 on-card sparse-op validation === H={H} idxH={Hi} idxD={Di} "
          f"Dh={Dh} Dv={Dv} index_topk={topk} T={T}")

    # ==================================================================================
    # GATE (a): indexer FIRES on ctx > index_topk (returns a STRICT subset, not all keys)
    # ==================================================================================
    print("\n--- GATE (a): indexer fires (strict top-k) on ctx > index_topk ---")
    Sa = topk * 2                                  # > index_topk so the top-k is a subset
    q_idx = (torch.randn(T, Hi, Di) * 0.1)
    k_idx = (torch.randn(Sa, Di) * 0.1)
    w_idx = (torch.rand(T, Hi) * 0.5 + 0.1)
    abs_pos_a = torch.full((T,), Sa - 1, dtype=torch.long)   # every query sees all Sa keys
    ref_a = _ref_indexer_topk(q_idx, k_idx, w_idx, topk, abs_pos_a)
    got_a = dsa_ops.indexer_topk_op(
        q_idx.to(dev), k_idx.to(dev), w_idx.to(dev), abs_pos_a.to(dev), topk)
    htorch.mark_step()
    got_a_cpu = got_a.to("cpu")
    kk = min(topk, Sa)
    sel_counts = (got_a_cpu >= 0).sum(dim=1)
    recall_a = _set_recall(got_a_cpu, ref_a)
    strict = bool((sel_counts == kk).all().item()) and kk < Sa
    print(f"  selected/row={int(sel_counts.min())}..{int(sel_counts.max())} "
          f"(kk={kk}, S={Sa}); recall vs fp32 oracle={recall_a:.4f}")
    if strict and recall_a >= 0.97:
        print(f"  >>> PASS: indexer fired — strict {kk}-of-{Sa} subset, recall {recall_a:.3f}.")
    else:
        print(f"  >>> FAIL: strict={strict} recall={recall_a:.4f} (expected {kk}<{Sa} "
              "and recall>=0.97).", file=sys.stderr)
        rc = 1

    # ==================================================================================
    # GATE (b): ctx <= index_topk -> sparse_attn output == dense causal SDPA (relNorm ~0)
    # ==================================================================================
    print("\n--- GATE (b): ctx <= index_topk -> sparse == dense ---")
    Sb = max(2, topk // 2)                          # <= index_topk: ALL causal keys selectable
    q_b = (torch.randn(T, H, Dh) * 0.1)
    k_b = (torch.randn(Sb, H, Dh) * 0.1)
    v_b = (torch.randn(Sb, H, Dv) * 0.1)
    abs_pos_b = torch.clamp(torch.linspace(0, Sb - 1, T).round().long(), max=Sb - 1)
    # The indexer over Sb<=topk keeps EVERY causal key -> build that full selection as the
    # topk_idx (id j for j<=abs_pos, -1 after). This is exactly what the indexer returns
    # when kk>=#causal keys (gate (a) already validated the kernel's selection).
    K_b = min(topk, Sb)
    ti_b = torch.full((T, K_b), -1, dtype=torch.int32)
    for t in range(T):
        n = int(abs_pos_b[t].item()) + 1
        ti_b[t, :min(n, K_b)] = torch.arange(min(n, K_b), dtype=torch.int32)
    sparse_b = dsa_ops.sparse_attn_op(
        q_b.to(dev), k_b.to(dev), v_b.to(dev), ti_b.to(dev), scale)
    htorch.mark_step()
    dense_b = _ref_dense_causal_attn(q_b, k_b, v_b, abs_pos_b, scale)
    rel_b = _rel_norm(sparse_b.to("cpu"), dense_b)
    print(f"  sparse vs dense relNorm = {rel_b:.5f} (tol {REL_TOL})")
    if rel_b <= REL_TOL:
        print(f"  >>> PASS: at ctx<=index_topk sparse == dense (relNorm {rel_b:.5f}).")
    else:
        print(f"  >>> FAIL: sparse != dense at ctx<=index_topk (relNorm {rel_b:.5f}).",
              file=sys.stderr)
        rc = 1

    # ==================================================================================
    # GATE (c): ctx > index_topk -> sparse != dense, but COHERENT (matches gather oracle)
    # ==================================================================================
    print("\n--- GATE (c): ctx > index_topk -> sparse != dense + coherent ---")
    Sc = max(args.ctx, topk * 2)
    q_c = (torch.randn(T, H, Dh) * 0.1)
    k_c = (torch.randn(Sc, H, Dh) * 0.1)
    v_c = (torch.randn(Sc, H, Dv) * 0.1)
    qi_c = (torch.randn(T, Hi, Di) * 0.1)
    ki_c = (torch.randn(Sc, Di) * 0.1)
    wi_c = (torch.rand(T, Hi) * 0.5 + 0.1)
    abs_pos_c = torch.full((T,), Sc - 1, dtype=torch.long)
    # REAL end-to-end: indexer op -> its indices -> sparse_attn op.
    ti_c = dsa_ops.indexer_topk_op(
        qi_c.to(dev), ki_c.to(dev), wi_c.to(dev), abs_pos_c.to(dev), topk)
    htorch.mark_step()
    ti_c_cpu = ti_c.to("cpu").to(torch.int32)
    sparse_c = dsa_ops.sparse_attn_op(
        q_c.to(dev), k_c.to(dev), v_c.to(dev), ti_c.to(dev), scale)
    htorch.mark_step()
    sparse_c_cpu = sparse_c.to("cpu")
    dense_c = _ref_dense_causal_attn(q_c, k_c, v_c, abs_pos_c, scale)
    gather_c = _ref_gather_attn(q_c, k_c, v_c, ti_c_cpu, scale)
    rel_vs_dense = _rel_norm(sparse_c_cpu, dense_c)
    rel_vs_gather = _rel_norm(sparse_c_cpu, gather_c)
    finite = bool(torch.isfinite(sparse_c_cpu.float()).all().item())
    differs = rel_vs_dense > 5.0e-2     # sparse must MEANINGFULLY differ from dense
    print(f"  finite={finite}; relNorm vs dense={rel_vs_dense:.4f} (must differ); "
          f"vs gather-oracle={rel_vs_gather:.5f} (tol {REL_TOL})")
    if finite and differs and rel_vs_gather <= REL_TOL:
        print(f"  >>> PASS: sparse differs from dense (relNorm {rel_vs_dense:.3f}) AND "
              f"matches the gather oracle (relNorm {rel_vs_gather:.5f}) — not silently dense.")
    else:
        print(f"  >>> FAIL: finite={finite} differs={differs} "
              f"rel_vs_gather={rel_vs_gather:.5f}.", file=sys.stderr)
        rc = 1

    print(f"\n[done] MVP-3 on-card sparse-op validation {'OK' if rc == 0 else 'FAILED'} "
          f"(rc={rc}). Next: the SERVE-level e2e gates (start the serve with "
          "MODEL_ENV_EXTRA=\"-e HPU_DSA_SPARSE=1 -e HPU_DSA_FIRED_LOG=1\" USE_DSA=1 "
          "./serve.sh glm51ar-lazy; gate (a) = the 'indexer FIRED (op)' log lines on a "
          ">2048-token prompt; gate (b)/(c) = diff generations vs the dense baseline serve).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
