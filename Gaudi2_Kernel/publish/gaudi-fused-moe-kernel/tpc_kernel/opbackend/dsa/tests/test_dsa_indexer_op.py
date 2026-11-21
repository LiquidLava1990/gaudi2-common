"""test_dsa_indexer_op.py -- VALIDATE DSA MVP-1 (the lightning-indexer top-k
OpBackend / GC-captured op) ON THE HPU BOX.

Loads build/libdsa_indexer.so (eager) or build/libdsa_indexer_lazy.so (lazy, via
$DSA_INDEXER_SO) with torch.ops.load_library, builds synthetic GLM-shaped indexer
inputs, calls the captured op

    torch.ops.hpu_dsa_ext.indexer_topk(q, k, weights, cmask, index_topk)
        q       [H, Sq, D]  bf16   (per-head index queries, rope already applied)
        k       [H, Sk, D]  bf16   (MQA index key expanded to H)
        weights [H, Sq, 1]  bf16   (q_scale*softmax_scale*H^-0.5 folded)
        cmask   [Sq, Sk]   bool    (TRUE where key is non-causal -> dropped)
        index_topk         int     (clamped to Sk internally)
    -> [Sq, kk] int32  top-k key indices per query   (kk = min(index_topk, Sk))

and asserts its top-k index SET matches a pure-torch fp32 ORACLE modeled on
hpu-dsa-ext/hpu_dsa_ext/sparse_indexer_hpu.py::reference_indexer_topk (the
parity-validated spec: score = sum_h w*ReLU(q.k), causal mask, topk). Because
top-k order may differ on ties / bf16-vs-fp32, we compare as a SET per query row
(mean Jaccard + mean recall over the kk selected keys) -- the BUILD_PLAN Gate-D
contract (the scorer itself validated at Jaccard 0.996, top-50 recall 1.0).

This is the FIRST real DSA kernel: it replaces the slow eager-torch indexer
scorer with one captured Synapse recipe (batch_gemm -> relu -> mult -> reduce_sum
-> masked_fill -> topk -> cast), intermediates non-persistent (SRAM-resident).

RUN (ON THE HPU BOX -- a CPU-only import cannot exercise the graph):
  cd <repo>/tpc_kernel/opbackend/dsa
  python tests/test_dsa_indexer_op.py                 # GLM-5.1 default H=32 D=128 Sq=256 Sk=384 topk=2048
  python tests/test_dsa_indexer_op.py 64 128 256 384 2048   # H D Sq Sk topk (DeepSeek-V3.2 H=64)
  # lazy .so + HPU-graph capture check:
  PT_HPU_LAZY_MODE=1 DSA_INDEXER_SO=build/libdsa_indexer_lazy.so python tests/test_dsa_indexer_op.py
"""
from __future__ import annotations

import os
import sys

NEG_INF = float("-inf")


def _reference_indexer_topk(q_thd, k_sd, w_th, topk, abs_pos):
    """fp32 ORACLE (== sparse_indexer_hpu.reference_indexer_topk :79-92).
       q_thd [Sq,H,D], k_sd [Sk,D], w_th [Sq,H], abs_pos [Sq] -> idx [Sq, kk]."""
    import torch
    Sq, H, D = q_thd.shape
    Sk = k_sd.shape[0]
    qf = q_thd.to(torch.float32)
    kf = k_sd.to(torch.float32)
    wf = w_th.to(torch.float32)
    logits = torch.relu(torch.einsum("thd,sd->ths", qf, kf))   # [Sq,H,Sk]
    score = torch.einsum("ths,th->ts", logits, wf)             # [Sq,Sk]
    key_pos = torch.arange(Sk).view(1, -1)
    score = score.masked_fill(key_pos > abs_pos.view(-1, 1), NEG_INF)
    kk = min(topk, Sk)
    return score.topk(kk, dim=-1).indices                      # [Sq, kk]


def _set_metrics(got_idx, ref_idx):
    """Per-row top-k SET agreement: mean Jaccard + mean recall over the rows."""
    Sq = got_idx.shape[0]
    jac, rec = 0.0, 0.0
    for t in range(Sq):
        g = set(int(x) for x in got_idx[t].tolist() if int(x) >= 0)
        r = set(int(x) for x in ref_idx[t].tolist() if int(x) >= 0)
        if not r:
            jac += 1.0
            rec += 1.0
            continue
        inter = len(g & r)
        union = len(g | r)
        jac += inter / union if union else 1.0
        rec += inter / len(r)
    return jac / Sq, rec / Sq


def _find_so():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../dsa
    env = os.environ.get("DSA_INDEXER_SO")
    if env:
        so = env if os.path.isabs(env) else os.path.join(here, env)
    else:
        so = os.path.join(here, "build", "libdsa_indexer.so")
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}/..:/work "
            f"-w /work/opbackend/dsa vllm-gaudi:next bash build_dsa_indexer.sh\n"
            f"  (or build_dsa_indexer_lazy.sh; set DSA_INDEXER_SO to pick the .so)")
    return so


def main() -> int:
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 32     # GLM-5.1 index_n_heads (DeepSeek=64)
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 128    # index_head_dim
    Sq = int(sys.argv[3]) if len(sys.argv) > 3 else 256   # query rows (a few hundred)
    Sk = int(sys.argv[4]) if len(sys.argv) > 4 else 384   # key rows
    topk = int(sys.argv[5]) if len(sys.argv) > 5 else 2048  # index_topk (clamps to Sk)

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
            torch.ops.hpu_dsa_ext, "indexer_topk"):
        print("FATAL: torch.ops.hpu_dsa_ext.indexer_topk not registered after "
              f"load_library({so}). Check TORCH_LIBRARY(hpu_dsa_ext) + the .so.",
              file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    kk = min(topk, Sk)
    # Set-agreement tol: the validated scorer hit Jaccard 0.996 / recall 1.0; we
    # require a high bar but allow a hair of bf16-vs-fp32 tie reshuffling.
    JAC_TOL = 0.97
    REC_TOL = 0.98
    rc = 0
    print(f"=== test_dsa_indexer_op (DSA MVP-1) ===  H={H} D={D} Sq={Sq} Sk={Sk} "
          f"index_topk={topk} (kk={kk})")

    # ---- synthetic GLM-shaped inputs (CPU master copies, fp32 -> bf16) ----
    # Per-head queries [Sq,H,D] and a single MQA key [Sk,D] (the indexer is MQA --
    # one shared key head). weights [Sq,H] are the folded per-head weights.
    q_thd = (torch.randn(Sq, H, D) * 0.1)         # [Sq,H,D]
    k_sd = (torch.randn(Sk, D) * 0.1)             # [Sk,D]  (MQA single head)
    w_th = (torch.rand(Sq, H) * 0.5 + 0.1)        # [Sq,H]  positive folded weights

    # Causal absolute positions: query t sees keys [0, abs_pos[t]]. Spread the
    # queries across the key axis so most rows have >= kk causal keys (so the
    # top-k SET is well-defined and -1 padding is not exercised here).
    abs_pos = torch.clamp(
        torch.linspace(kk, Sk - 1, Sq).round().long(), max=Sk - 1)  # [Sq]
    key_pos = torch.arange(Sk).view(1, -1)
    cmask = (key_pos > abs_pos.view(-1, 1))       # [Sq,Sk] bool, TRUE => drop (non-causal)

    # ---- oracle (fp32, CPU) ----
    ref_idx = _reference_indexer_topk(q_thd, k_sd, w_th, topk, abs_pos)  # [Sq,kk]

    # ---- op inputs on HPU (op layout: head axis leading for the batched Q.K^T) ----
    q_op = q_thd.permute(1, 0, 2).contiguous().to(torch.bfloat16).to(dev)   # [H,Sq,D]
    k_op = k_sd.unsqueeze(0).expand(H, -1, -1).contiguous().to(torch.bfloat16).to(dev)  # [H,Sk,D]
    w_op = w_th.permute(1, 0).unsqueeze(-1).contiguous().to(torch.bfloat16).to(dev)     # [H,Sq,1]
    cmask_op = cmask.contiguous().to(dev)          # [Sq,Sk] bool
    htorch.mark_step()

    # ====================================================================
    # CHECK 1 -- top-k index SET vs the fp32 oracle.
    # ====================================================================
    print("\n--- CHECK 1: indexer_topk index set (vs reference_indexer_topk) ---")
    got = torch.ops.hpu_dsa_ext.indexer_topk(q_op, k_op, w_op, cmask_op, topk)
    htorch.mark_step()
    got_idx = got.to("cpu")
    if list(got_idx.shape) != [Sq, kk]:
        print(f"  >>> FAIL: op returned shape {list(got_idx.shape)}, expected "
              f"[{Sq}, {kk}].", file=sys.stderr)
        return 1
    if got_idx.dtype != torch.int32:
        print(f"  >>> FAIL: op returned dtype {got_idx.dtype}, expected int32.",
              file=sys.stderr)
        rc = 1
    jac, rec = _set_metrics(got_idx, ref_idx)
    print(f"  mean Jaccard={jac:.4f}  mean recall={rec:.4f}  (kk={kk} per row, Sq={Sq} rows)")
    if jac >= JAC_TOL and rec >= REC_TOL:
        print(f"  >>> PASS: indexer_topk set matches oracle (Jaccard {jac:.3f} "
              f">= {JAC_TOL}, recall {rec:.3f} >= {REC_TOL}).")
    else:
        print(f"  >>> FAIL: indexer_topk Jaccard {jac:.4f} / recall {rec:.4f} below "
              f"tol ({JAC_TOL}/{REC_TOL}).", file=sys.stderr)
        print("      Suspects (device-confirmable): batch_gemm transpose_b (q@k^T)\n"
              "      orientation, the [H,Sq,1] weight broadcast, reduce_sum dim-mask\n"
              "      (1<<(rank-d-1)=4 for dim0/rank3), masked_fill -inf value.f, the\n"
              "      topk axis=get_dim_in_tpc_order(1,2)=0, or the i64->i32 cast.",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- HPU-graph CAPTURE (the durable serving path). Best-effort:
    # wrap the op in an HPU graph; success => it lowered into ONE captured
    # recipe (esp. relevant for the lazy .so). Mirrors test_fused_expert_op.py.
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
            out_cap = torch.ops.hpu_dsa_ext.indexer_topk(q_op, k_op, w_op, cmask_op, topk)
            g.capture_end()
        g.replay()
        htorch.mark_step()
        jac2, rec2 = _set_metrics(out_cap.to("cpu"), ref_idx)
        captured_ok = jac2 >= JAC_TOL and rec2 >= REC_TOL
        if captured_ok:
            print(f"  >>> PASS: op captured into an HPU graph and replayed "
                  f"(Jaccard {jac2:.3f}). This is the durable single-recipe serving path.")
        else:
            print(f"  >>> WARN: captured replay Jaccard {jac2:.4f} below tol "
                  "(numerics differ under capture; investigate on device).", file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped capture check: {ex})")
        print("  NOTE: capture is what the lazy .so confirms; CHECK 1 already exercised "
              "the SAME captured AddNode lowering.")

    # ====================================================================
    # CHECK 3 -- topk-at-large-Sk perf HINT (the design's #1 measure-on-device
    # item). Re-run with a large Sk to surface the topk CGUID cost at scale.
    # ====================================================================
    print("\n--- CHECK 3: topk-at-large-Sk hint (design risk #2; best-effort) ---")
    print("  The top-k over a long key axis is the #1 'needs measurement' item")
    print("  (DSA_OPBACKEND_KERNEL_DESIGN.md risk 2). To probe cost at scale, e.g.:")
    print(f"    python tests/test_dsa_indexer_op.py {H} {D} 128 65536 2048")
    print("  and time the call; if topk dominates, fall back to sort_fwd+slice or")
    print("  a 2-stage block-topk (M<=128 tiling already bounds the score width).")

    print(f"\n[done] test_dsa_indexer_op {'OK' if rc == 0 else 'FAILED'} (rc={rc})"
          + (f"; capture={'ok' if captured_ok else captured_ok}"
             if captured_ok is not None else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
