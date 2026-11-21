"""CPU unit test for SparseAttnIndexer.forward_oot orchestration (no HPU, no plumbing).

Exercises the real forward_oot code path via the _dsa_plan_override hook:
  - buffer reset to -1
  - multi-segment prefill (batched, with prior context => non-trivial abs positions)
  - decode (single query row per request against full history)
  - sentinel -1 padding for short rows (S < topk_tokens)
  - parity of selected key SET against reference_indexer_topk (the fp32 oracle)

Run:  python3 $HOME/AI/hpu-dsa-ext/tests/test_forward_oot_cpu.py
"""
import sys
import types

import torch

sys.path.insert(0, "$HOME/AI/hpu-dsa-ext")
from hpu_dsa_ext import sparse_indexer_hpu as M  # noqa: E402


class FakeIndexer:
    """Stand-in for SparseAttnIndexer carrying just what forward_oot touches."""
    def __init__(self, topk_tokens, max_tokens):
        self.topk_tokens = topk_tokens
        self.skip_k_cache_insert = False
        self.topk_indices_buffer = torch.full(
            (max_tokens, topk_tokens), 123456, dtype=torch.int32
        )  # poison so we can prove the reset to -1 happened


def _sorted_set(idx_row):
    v = idx_row[idx_row >= 0].tolist()
    return sorted(v)


def _ref_causal_set(q_seg, keys, w_seg, topk, abs_pos, row):
    """Oracle selected SET for one row, with -inf short-row fills removed (apples-to-
    apples vs lightning_indexer_topk, which sentinel-drops non-causal picks)."""
    qf = q_seg[row:row + 1].to(torch.float32)
    kf = keys.to(torch.float32)
    wf = w_seg[row:row + 1].to(torch.float32)
    logits = torch.relu(torch.einsum("thd,sd->ths", qf, kf))
    score = torch.einsum("ths,th->ts", logits, wf)[0]          # [S]
    S = keys.shape[0]
    key_pos = torch.arange(S)
    score = score.masked_fill(key_pos > int(abs_pos[row]), float("-inf"))
    kk = min(topk, S)
    idx = score.topk(kk).indices
    valid = score[idx] > float("-inf")
    return sorted(idx[valid].tolist())


def make_qkw(T, H, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(T, D, generator=g, dtype=torch.float32).to(torch.bfloat16)
    w = torch.rand(T, H, generator=g, dtype=torch.float32).to(torch.bfloat16) + 0.1
    return q, k, w


def test_prefill_multi_segment():
    H, D, topk = 8, 128, 6
    # two requests in one prefill batch; req0 has 5 prior-context keys, req1 has 0.
    # segment keys = past_ctx ++ current; abs_query_pos counts from ctx start.
    seg_specs = [
        dict(ctx=5, qlen=10, seed=1),
        dict(ctx=0, qlen=7, seed=2),
    ]
    T = sum(s["qlen"] for s in seg_specs)
    bigq = torch.empty(T, H, D, dtype=torch.bfloat16)
    bigw = torch.empty(T, H, dtype=torch.bfloat16)
    idxr = FakeIndexer(topk, max_tokens=T + 3)

    plan = []
    seg_tensors = []
    tok = 0
    for s in seg_specs:
        ctx, qlen = s["ctx"], s["qlen"]
        S = ctx + qlen
        qk_q, keys, qk_w = make_qkw(S, H, D, seed=s["seed"])  # keys: [S, D]
        # current-step queries are the LAST qlen rows; their abs pos = ctx..ctx+qlen-1
        q_seg = qk_q[ctx:ctx + qlen]
        w_seg = qk_w[ctx:ctx + qlen]
        abs_pos = torch.arange(ctx, ctx + qlen, dtype=torch.long)
        bigq[tok:tok + qlen] = q_seg
        bigw[tok:tok + qlen] = w_seg
        plan.append(dict(keys=keys, abs_query_pos=abs_pos,
                         token_start=tok, token_end=tok + qlen))
        seg_tensors.append((q_seg, keys, w_seg, abs_pos))
        tok += qlen

    idxr._dsa_plan_override = plan
    out = M.forward_oot(idxr, None, bigq, None, bigw)

    # buffer reset proof: rows beyond T stay poisoned, active rows are valid indices/-1
    assert torch.equal(out[T:], torch.full((3, topk), 123456, dtype=torch.int32)), \
        "rows beyond num_tokens must be untouched (reset only active rows)"

    tok = 0
    for si, s in enumerate(seg_specs):
        qlen = s["qlen"]
        q_seg, keys, w_seg, abs_pos = seg_tensors[si]
        for r in range(qlen):
            got = _sorted_set(out[tok + r])
            exp = _ref_causal_set(q_seg, keys, w_seg, topk, abs_pos, r)
            assert got == exp, (
                f"seg{si} row{r}: selected set mismatch\n got={got}\n exp={exp}"
            )
            # causal: no selected key index may exceed this row's abs position
            ap = s["ctx"] + r
            assert all(i <= ap for i in got), f"causal violation row {r}: {got} > {ap}"
        tok += qlen
    print(f"[ok] prefill multi-segment: {T} rows, 2 segments, topk={topk} — set parity + causal")


def test_short_rows_padding():
    H, D, topk = 4, 128, 32  # topk >> available keys => -1 padding required
    qlen, ctx = 5, 0
    S = qlen
    q, k, w = make_qkw(S, H, D, seed=7)
    idxr = FakeIndexer(topk, max_tokens=S)
    abs_pos = torch.arange(qlen, dtype=torch.long)
    idxr._dsa_plan_override = [
        dict(keys=k, abs_query_pos=abs_pos, token_start=0, token_end=qlen)
    ]
    out = M.forward_oot(idxr, None, q, None, w)
    for r in range(qlen):
        n_causal = r + 1  # keys 0..r
        sel = out[r][out[r] >= 0]
        assert sel.numel() == n_causal, (
            f"row {r}: expected {n_causal} valid keys, got {sel.numel()} ({out[r].tolist()})"
        )
        # remaining slots are exactly -1 (sentinel), never clamped-to-0
        pad = out[r][n_causal:]
        assert torch.all(pad == -1), f"row {r} pad must be -1, got {pad.tolist()}"
    print(f"[ok] short-row sentinel: topk={topk} >> S={S}, padding is -1 (never token 0)")


def test_decode_single_row():
    H, D, topk = 8, 128, 4
    hist = 20  # full history length; one decode query at abs pos = hist-1 (last token)
    q, k, w = make_qkw(hist, H, D, seed=11)
    idxr = FakeIndexer(topk, max_tokens=1)
    q_dec = q[hist - 1:hist]          # [1, H, D]
    w_dec = w[hist - 1:hist]          # [1, H]
    abs_pos = torch.tensor([hist - 1], dtype=torch.long)
    idxr._dsa_plan_override = [
        dict(keys=k, abs_query_pos=abs_pos, token_start=0, token_end=1)
    ]
    out = M.forward_oot(idxr, None, q_dec, None, w_dec)
    got = _sorted_set(out[0])
    ref = _ref_causal_set(q_dec, k, w_dec, topk, abs_pos, 0)
    assert got == ref, f"decode mismatch: got={got} exp={ref}"
    assert len(got) == topk and all(i <= hist - 1 for i in got)
    print(f"[ok] decode single-row: history={hist}, topk={topk} — set parity + causal")


def test_buffer_returned_is_same_object():
    H, D, topk = 2, 128, 3
    q, k, w = make_qkw(4, H, D, seed=3)
    idxr = FakeIndexer(topk, max_tokens=4)
    idxr._dsa_plan_override = [
        dict(keys=k, abs_query_pos=torch.arange(4), token_start=0, token_end=4)
    ]
    out = M.forward_oot(idxr, None, q, None, w)
    assert out is idxr.topk_indices_buffer, "forward_oot must return the buffer it mutates"
    print("[ok] returns the mutated buffer object (mutates_args contract)")


if __name__ == "__main__":
    test_prefill_multi_segment()
    test_short_rows_padding()
    test_decode_single_row()
    test_buffer_returned_is_same_object()
    print("\nALL CPU forward_oot orchestration tests PASSED")
