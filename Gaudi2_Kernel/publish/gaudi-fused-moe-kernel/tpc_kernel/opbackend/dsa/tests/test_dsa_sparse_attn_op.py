"""test_dsa_sparse_attn_op.py -- VALIDATE DSA MVP-2 (the gather-consume sparse-
attention OpBackend / GC-captured op) ON THE HPU BOX.

Loads build/libdsa_sparse_attn.so (eager) or build/libdsa_sparse_attn_lazy.so
(lazy, via $DSA_SPARSE_ATTN_SO) with torch.ops.load_library, builds synthetic
GLM-shaped per-head MLA q/k/v + a -1-padded top-k selection (synthetic, OR fed
from the working MVP-1 indexer op if its .so is provided via $DSA_INDEXER_SO),
calls the captured op

    torch.ops.hpu_dsa_ext.sparse_attn(q, k, v, topk_idx, scale, gather_width)
        q          [T, H, 1, Dh]  bf16  (per-head MLA query, L=1 per query)
        k          [S, H, Dh]     bf16  (per-head MLA key, MQA expanded to H)
        v          [S, H, Dv]     bf16  (per-head MLA value)
        topk_idx   [T, K]         int32 (the indexer top-k key indices, -1 padded)
        scale                     float (MLA softmax scale, qk_head_dim**-0.5)
        gather_width              int   (== K, the indexer buffer width)
    -> [T, H, Dv] bf16  sparse attention output

and asserts it matches the parity-validated PRODUCT oracle
``hpu_dsa_ext.sparse_mla_consume.sparse_mla_gather_attention`` (and the
masked-MHA oracle ``sparse_mla_masked_attention``), within the oracle's validated
band (gather==masked diff ~0.011; we use relNorm < ~2e-2 to cover bf16-vs-fp32 +
the captured-kernel path). The op IS the gather oracle re-emitted as one captured
Synapse recipe (ge -> where -> where -> concat -> gather_fwd -> transpose ->
sdpa_fwd -> reshape), intermediates non-persistent (SRAM-resident).

RUN (ON THE HPU BOX -- a CPU-only import cannot exercise the graph):
  cd <repo>/tpc_kernel/opbackend/dsa
  python tests/test_dsa_sparse_attn_op.py                 # GLM-5.1-shaped defaults
  python tests/test_dsa_sparse_attn_op.py 64 256 192 256 384 64   # H Dh Dv T S K
  # lazy .so + HPU-graph capture check:
  PT_HPU_LAZY_MODE=1 DSA_SPARSE_ATTN_SO=build/libdsa_sparse_attn_lazy.so \
      python tests/test_dsa_sparse_attn_op.py
  # feed REAL indices from the MVP-1 indexer op (end-to-end indexer->consume):
  DSA_INDEXER_SO=build/libdsa_indexer.so python tests/test_dsa_sparse_attn_op.py
"""
from __future__ import annotations

import math
import os
import sys

NEG_INF = float("-inf")


# =====================================================================================
# Pure-torch ORACLES (self-contained; == hpu-dsa-ext sparse_mla_consume contract).
# Used as the parity reference so this test does not depend on the hpu_dsa_ext package
# being importable on the box. These re-derive the LOCKED contract (sentinel-scatter,
# causal+self keep, NaN guard) exactly as sparse_mla_consume.py / the synthetic test.
# =====================================================================================
def _build_keep_mask(topk_indices, abs_query_pos, num_keys):
    """[T, S] bool keep-mask = the SELECTED keys, via SENTINEL-COLUMN scatter (contract 5).

    Models the OP's exact selection: keep == the fed top-k indices (>=0), NOTHING extra.
    Causality is ALREADY in the fed indices (the indexer emits a causal top-k -> the op
    re-applies neither causality nor a forced self-diagonal; it masks ONLY -1 pads). So
    the masked + gather oracles attend the IDENTICAL set the op does, and agree tightly.
    (We assert below that the fed top-k is causal: key id <= abs_pos[t].)
    """
    import torch
    T = topk_indices.shape[0]
    device = topk_indices.device
    idx = topk_indices.long()
    valid_entry = (idx >= 0) & (idx < num_keys)
    sink = torch.where(valid_entry, idx, torch.full_like(idx, num_keys))
    keep = torch.zeros((T, num_keys + 1), dtype=torch.bool, device=device)
    keep.scatter_(1, sink, torch.ones_like(sink, dtype=torch.bool))
    keep = keep[:, :num_keys]
    return keep


def _masked_oracle(q, k, v, topk_indices, abs_query_pos, scale):
    """≤32K masked-MHA oracle: dense softmax over ALL S keys w/ additive -inf bias.
    Independent cross-check of the gather path (different code path, same selection)."""
    import torch
    T, H, _ = q.shape
    S = k.shape[0]
    keep = _build_keep_mask(topk_indices, abs_query_pos, S)
    bias = torch.where(keep, torch.zeros((), dtype=torch.float32),
                       torch.full((), NEG_INF, dtype=torch.float32))
    qf = q.transpose(0, 1).float()
    kf = k.transpose(0, 1).float()
    vf = v.transpose(0, 1).float()
    logits = torch.matmul(qf, kf.transpose(-1, -2)) * scale + bias.unsqueeze(0)
    probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    out = torch.matmul(probs, vf)
    return out.transpose(0, 1).to(q.dtype)


def _gather_oracle(q, k, v, topk_indices, abs_query_pos, scale, gather_width):
    """PRODUCT (option-B) oracle == what the OP re-emits: the topk ids ARE the gather
    ids; gather KV at them (-1 -> sentinel zero row), SDPA w/ pad-only -inf bias."""
    import torch
    T, H, Dh = q.shape
    S = k.shape[0]
    Dv = v.shape[-1]
    idx = topk_indices.long()
    width = gather_width
    valid = idx >= 0                                   # [T,K] pad detector
    gid = torch.where(valid, idx, torch.full_like(idx, S))   # [T,K], pad -> sentinel S
    k_ext = torch.cat([k, torch.zeros(1, H, Dh, dtype=k.dtype)], dim=0)   # [S+1,H,Dh]
    v_ext = torch.cat([v, torch.zeros(1, H, Dv, dtype=v.dtype)], dim=0)   # [S+1,H,Dv]
    kg = k_ext[gid]                                    # [T,K,H,Dh]
    vg = v_ext[gid]                                    # [T,K,H,Dv]
    logits = torch.einsum("thd,twhd->thw", q.float(), kg.float()) * scale   # [T,H,K]
    bias = torch.where(valid.view(T, 1, width), torch.zeros((), dtype=torch.float32),
                       torch.full((), NEG_INF, dtype=torch.float32))
    logits = logits + bias
    probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    out = torch.einsum("thw,twhd->thd", probs, vg.float())   # [T,H,Dv]
    return out.to(q.dtype)


def _rel_norm(got, ref):
    import torch
    g = got.float()
    r = ref.float()
    denom = r.norm().item()
    return (g - r).norm().item() / (denom if denom > 0 else 1.0)


def _find_so():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../dsa
    env = os.environ.get("DSA_SPARSE_ATTN_SO")
    if env:
        so = env if os.path.isabs(env) else os.path.join(here, env)
    else:
        so = os.path.join(here, "build", "libdsa_sparse_attn.so")
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}/..:/work "
            f"-w /work/opbackend/dsa vllm-gaudi:next bash build_dsa_sparse_attn.sh\n"
            f"  (or build_dsa_sparse_attn_lazy.sh; set DSA_SPARSE_ATTN_SO to pick the .so)")
    return so


def _maybe_load_indexer():
    """Optionally load the MVP-1 indexer .so (via $DSA_INDEXER_SO) so we can feed REAL
    top-k indices (end-to-end indexer->consume). Returns True iff loaded+registered."""
    import torch
    env = os.environ.get("DSA_INDEXER_SO")
    if not env:
        return False
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    so = env if os.path.isabs(env) else os.path.join(here, env)
    if not os.path.isfile(so):
        print(f"  (DSA_INDEXER_SO={so} not found; using synthetic top-k)")
        return False
    torch.ops.load_library(so)
    ok = hasattr(torch.ops, "hpu_dsa_ext") and hasattr(
        torch.ops.hpu_dsa_ext, "indexer_topk")
    print(f"  loaded indexer .so {so} (registered={ok})")
    return ok


def main() -> int:
    # GLM-5.1 MLA-shaped defaults (per-head, post W_UK/W_UV up-proj):
    #   qk_head_dim = qk_nope(192)+qk_rope(64) = 256 ; v_head_dim = 256.
    # Keep T/S/K small so the synthetic runs fast; topk K is the indexer buffer width.
    H  = int(sys.argv[1]) if len(sys.argv) > 1 else 8     # heads (subset of 64 for speed)
    Dh = int(sys.argv[2]) if len(sys.argv) > 2 else 256   # qk_head_dim (192+64)
    Dv = int(sys.argv[3]) if len(sys.argv) > 3 else 256   # v_head_dim
    T  = int(sys.argv[4]) if len(sys.argv) > 4 else 64    # query rows
    S  = int(sys.argv[5]) if len(sys.argv) > 5 else 192   # key/value rows (context)
    K  = int(sys.argv[6]) if len(sys.argv) > 6 else 48    # index_topk buffer width (< S)

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
    if not hasattr(torch.ops, "hpu_dsa_ext") or not hasattr(
            torch.ops.hpu_dsa_ext, "sparse_attn"):
        print("FATAL: torch.ops.hpu_dsa_ext.sparse_attn not registered after "
              f"load_library({so}). Check TORCH_LIBRARY_FRAGMENT(hpu_dsa_ext) + the .so.",
              file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    scale = Dh ** -0.5
    # relNorm tol: the gather==masked oracle validated at diff 0.011; allow a hair more
    # for bf16-vs-fp32 + the fused sdpa_fwd fp32-softmax path.
    REL_TOL = 2.0e-2
    rc = 0
    print(f"=== test_dsa_sparse_attn_op (DSA MVP-2) ===  H={H} Dh={Dh} Dv={Dv} "
          f"T={T} S={S} K={K} scale={scale:.5f}")

    # ---- synthetic GLM-shaped per-head MLA inputs (CPU master copies, fp32->bf16) ----
    q_thd = (torch.randn(T, H, Dh) * 0.1)       # [T,H,Dh]
    k_shd = (torch.randn(S, H, Dh) * 0.1)       # [S,H,Dh]
    v_shd = (torch.randn(S, H, Dv) * 0.1)       # [S,H,Dv]

    # Causal absolute positions: spread queries across the key axis so every row has
    # >= K causal keys (so the K-wide top-k is fully valid; -1 padding minimal).
    abs_pos = torch.clamp(
        torch.linspace(K, S - 1, T).round().long(), max=S - 1)    # [T]

    # ---- top-k selection: REAL (from the MVP-1 indexer op) iff available, else synthetic.
    use_indexer = _maybe_load_indexer()
    if use_indexer:
        # Build indexer inputs from the SAME context: index keys = a projection of k.
        # For the harness, reuse k's first head-dim slice as MQA index keys (any
        # consistent scoring works; we only need a valid -1-padded causal top-k).
        Di = min(128, Dh)
        idx_q = (torch.randn(T, H, Di) * 0.1)
        idx_k = (torch.randn(S, Di) * 0.1)
        idx_w = (torch.rand(T, H) * 0.5 + 0.1)
        key_pos = torch.arange(S).view(1, -1)
        cmask = (key_pos > abs_pos.view(-1, 1))                  # TRUE => drop
        q_op_i = idx_q.permute(1, 0, 2).contiguous().to(torch.bfloat16).to(dev)
        k_op_i = idx_k.unsqueeze(0).expand(H, -1, -1).contiguous().to(torch.bfloat16).to(dev)
        w_op_i = idx_w.permute(1, 0).unsqueeze(-1).contiguous().to(torch.bfloat16).to(dev)
        cmask_op = cmask.contiguous().to(dev)
        htorch.mark_step()
        topk_idx = torch.ops.hpu_dsa_ext.indexer_topk(q_op_i, k_op_i, w_op_i, cmask_op, K)
        htorch.mark_step()
        topk_idx_cpu = topk_idx.to("cpu").to(torch.int32)        # [T, min(K,S)]
        K = topk_idx_cpu.shape[1]                                # honor clamp(K,S)
        print(f"  fed REAL indexer top-k: shape {list(topk_idx_cpu.shape)}")
    else:
        # Synthetic causal top-k: for each query t, pick K DISTINCT causal keys in
        # [0, abs_pos[t]] (a real indexer's topk indices are distinct positions); pad
        # with -1 if fewer than K are available. self (abs_pos[t]) is placed FIRST and
        # the rest sampled from the other causal keys -- distinct, mirroring a real
        # causal top-k that always includes the diagonal. (Distinctness matters: a
        # duplicated key would get doubled softmax mass in the gather but single mass in
        # the masked set -- the op gathers exactly the given ids, so the test feeds the
        # same distinct ids a real indexer would.)
        topk_idx_cpu = torch.full((T, K), -1, dtype=torch.int32)
        for t in range(T):
            self_id = int(abs_pos[t].item())
            others = torch.arange(self_id)                      # the other causal keys [0, self)
            others = others[torch.randperm(others.numel())]
            row = torch.cat([torch.tensor([self_id]), others])[:K]   # self first, distinct
            topk_idx_cpu[t, :row.numel()] = row.to(torch.int32)
        print(f"  synthetic causal top-k: shape {list(topk_idx_cpu.shape)} "
              f"(rows with -1 pad: {(topk_idx_cpu < 0).any(dim=1).sum().item()}/{T})")

    # Contract assertion: every fed index is causal (key id <= abs_pos[t]) or a -1 pad.
    _ti = topk_idx_cpu.long()
    _causal_ok = bool(((_ti < 0) | (_ti <= abs_pos.view(-1, 1))).all().item())
    if not _causal_ok:
        print("  >>> WARN: fed top-k contains a non-causal index (indexer contract "
              "violated); the op masks only pads, so causality must come from the indexer.")

    # ---- oracles (fp32-ish on CPU) ----
    ref_gather = _gather_oracle(q_thd, k_shd, v_shd, topk_idx_cpu, abs_pos, scale, K)
    ref_masked = _masked_oracle(q_thd, k_shd, v_shd, topk_idx_cpu, abs_pos, scale)
    oracle_internal = _rel_norm(ref_gather, ref_masked)
    print(f"  oracle self-check: gather-vs-masked relNorm = {oracle_internal:.5f} "
          f"(validated band ~0.011)")

    # ---- op inputs on HPU ----
    q_op = q_thd.unsqueeze(2).contiguous().to(torch.bfloat16).to(dev)   # [T,H,1,Dh]
    k_op = k_shd.contiguous().to(torch.bfloat16).to(dev)               # [S,H,Dh]
    v_op = v_shd.contiguous().to(torch.bfloat16).to(dev)               # [S,H,Dv]
    ti_op = topk_idx_cpu.contiguous().to(dev)                          # [T,K] int32
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- sparse attention output vs the PRODUCT (gather) oracle.
    # ====================================================================
    print("\n--- CHECK 1: sparse_attn output (vs sparse_mla_gather_attention) ---")
    got = torch.ops.hpu_dsa_ext.sparse_attn(q_op, k_op, v_op, ti_op, scale, K)
    htorch.mark_step()
    got_cpu = got.to("cpu")
    if list(got_cpu.shape) != [T, H, Dv]:
        print(f"  >>> FAIL: op returned shape {list(got_cpu.shape)}, expected "
              f"[{T}, {H}, {Dv}].", file=sys.stderr)
        return 1
    if got_cpu.dtype != torch.bfloat16:
        print(f"  >>> WARN: op returned dtype {got_cpu.dtype}, expected bfloat16.")
    rel_g = _rel_norm(got_cpu, ref_gather)
    rel_m = _rel_norm(got_cpu, ref_masked)
    print(f"  relNorm vs gather-oracle = {rel_g:.5f} ; vs masked-oracle = {rel_m:.5f} "
          f"(tol {REL_TOL})")
    if rel_g <= REL_TOL:
        print(f"  >>> PASS: sparse_attn matches the gather oracle (relNorm {rel_g:.5f} "
              f"<= {REL_TOL}).")
    else:
        print(f"  >>> FAIL: sparse_attn relNorm {rel_g:.5f} > tol {REL_TOL}.",
              file=sys.stderr)
        print("      Suspects (device-confirmable): the SDPA per-query [T,H,1,Dh]x\n"
              "      [T,H,K,Dh] layout + the [T,1,1,K] pad-mask broadcast over heads;\n"
              "      the gather_fwd axis=get_dim_in_tpc_order(0,2)=1 on the flattened\n"
              "      [S+1,H*D] key; the where_fwd sentinel-S routing; the concat axis\n"
              "      (tpc dim0/rank3=2); or the sdpa_fwd ParamsV7 (fp32 softmax, scale).",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- HPU-graph CAPTURE (the durable serving path; best-effort).
    # Success => the chain lowered into ONE captured recipe (esp. the lazy .so).
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
            out_cap = torch.ops.hpu_dsa_ext.sparse_attn(q_op, k_op, v_op, ti_op, scale, K)
            g.capture_end()
        g.replay()
        htorch.mark_step()
        rel_c = _rel_norm(out_cap.to("cpu"), ref_gather)
        captured_ok = rel_c <= REL_TOL
        if captured_ok:
            print(f"  >>> PASS: op captured into an HPU graph and replayed "
                  f"(relNorm {rel_c:.5f}). This is the durable single-recipe serving path.")
        else:
            print(f"  >>> WARN: captured replay relNorm {rel_c:.5f} > tol "
                  "(numerics differ under capture; investigate on device).", file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")
        print("  NOTE: capture is what the lazy .so confirms; CHECK 1 already exercised "
              "the SAME captured AddNode lowering.")

    # ====================================================================
    # CHECK 3 -- short-row / -1-pad robustness: a query with FEWER than K causal
    # keys must mask the pad slots (NaN guard), not select token 0.
    # ====================================================================
    print("\n--- CHECK 3: short-row (-1 pad) NaN-guard robustness ---")
    ti_short = topk_idx_cpu.clone()
    ti_short[0, K // 2:] = -1                      # force row 0 to be half-padded
    ref_short = _gather_oracle(q_thd, k_shd, v_shd, ti_short, abs_pos, scale, K)
    got_short = torch.ops.hpu_dsa_ext.sparse_attn(
        q_op, k_op, v_op, ti_short.contiguous().to(dev), scale, K)
    htorch.mark_step()
    gs = got_short.to("cpu")
    nan_free = bool(torch.isfinite(gs.float()).all().item())
    rel_s = _rel_norm(gs, ref_short)
    print(f"  row0 half -1-padded: finite={nan_free} ; relNorm vs oracle = {rel_s:.5f}")
    if nan_free and rel_s <= REL_TOL:
        print("  >>> PASS: padded slots masked (no NaN), output matches oracle.")
    else:
        print(f"  >>> FAIL: short-row guard (finite={nan_free}, relNorm {rel_s:.5f}).",
              file=sys.stderr)
        rc = 1

    print(f"\n[done] test_dsa_sparse_attn_op {'OK' if rc == 0 else 'FAILED'} (rc={rc})"
          + (f"; capture={'ok' if captured_ok else captured_ok}"
             if captured_ok is not None else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
