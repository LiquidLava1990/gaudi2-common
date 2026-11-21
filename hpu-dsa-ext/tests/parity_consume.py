"""CPU parity test for sparse_mla_consume.py — gather path vs masked-MHA oracle, sentinel-scatter
correctness, NaN guard, and the clamp(min=0) counter-check. Runs on CPU torch (no HPU)."""
import sys
import torch

sys.path.insert(0, "$HOME/AI/hpu-dsa-ext")
from hpu_dsa_ext.sparse_mla_consume import (  # noqa: E402
    sparse_mla_gather_attention,
    sparse_mla_masked_attention,
    build_sparse_attn_bias,
    _build_keep_mask,
)

torch.manual_seed(0)


def make_topk(scores, abs_pos, index_topk):
    """Reference indexer: causal top-k token ids per query, -1 padded (mimics the real op)."""
    T, S = scores.shape
    kp = torch.arange(S).view(1, -1)
    masked = scores.masked_fill(kp > abs_pos.view(-1, 1), float("-inf"))
    kk = min(index_topk, S)
    idx = masked.topk(kk, dim=-1).indices.to(torch.int32)
    out = torch.full((T, index_topk), -1, dtype=torch.int32)
    valid = torch.gather(masked, 1, idx.long()) > float("-inf")
    out[:, :kk] = torch.where(valid, idx, torch.full_like(idx, -1))
    return out


def test_gather_matches_oracle():
    T, H, Dh, Dv, S = 17, 4, 16, 12, 40
    index_topk = 8
    q = torch.randn(T, H, Dh)
    k = torch.randn(S, H, Dh)
    v = torch.randn(S, H, Dv)
    abs_pos = torch.arange(S - T, S)            # contiguous prefill: query t -> key S-T+t
    scale = Dh ** -0.5
    sel_scores = torch.randn(T, S)              # arbitrary indexer scores
    topk = make_topk(sel_scores, abs_pos, index_topk)

    out_gather = sparse_mla_gather_attention(q, k, v, topk, abs_pos, scale)
    out_oracle = sparse_mla_masked_attention(q, k, v, topk, abs_pos, scale)
    assert out_gather.shape == (T, H, Dv), out_gather.shape
    diff = (out_gather.float() - out_oracle.float()).abs().max().item()
    assert torch.isfinite(out_gather).all(), "gather produced non-finite"
    assert diff < 1e-4, f"gather vs oracle max abs diff {diff}"
    return diff


def test_sentinel_not_clamp():
    """A row whose selection does NOT include token 0 must not attend to token 0. clamp(min=0)
    would; sentinel-scatter must not."""
    T, S = 3, 20
    index_topk = 5
    abs_pos = torch.full((T,), S - 1)
    # Force selections that exclude column 0 entirely (ids 5..9), with some -1 pad slots.
    topk = torch.full((T, index_topk), -1, dtype=torch.int32)
    topk[:, 0] = 5
    topk[:, 1] = 9
    topk[:, 2] = 13            # rest are -1 pads
    keep = _build_keep_mask(topk, abs_pos, S)
    # token 0 must be False for every query (self col is S-1, not 0).
    assert not keep[:, 0].any(), "sentinel-scatter wrongly selected token 0 (clamp bug)"
    # exactly {5,9,13} plus self(S-1) selected -> 4 per row.
    assert keep.sum(dim=1).tolist() == [4, 4, 4], keep.sum(dim=1).tolist()
    return keep.sum(dim=1).tolist()


def test_nan_guard_empty_row():
    """A query whose every selected slot is -1 (empty) must still produce finite output via forced
    self-inclusion + nan_to_num, not NaN."""
    T, H, Dh, Dv, S = 4, 2, 8, 8, 10
    index_topk = 4
    q = torch.randn(T, H, Dh)
    k = torch.randn(S, H, Dh)
    v = torch.randn(S, H, Dv)
    abs_pos = torch.arange(S - T, S)
    topk = torch.full((T, index_topk), -1, dtype=torch.int32)   # ALL empty -> self only
    g = sparse_mla_gather_attention(q, k, v, topk, abs_pos, Dh ** -0.5)
    o = sparse_mla_masked_attention(q, k, v, topk, abs_pos, Dh ** -0.5)
    assert torch.isfinite(g).all(), "NaN guard failed (gather)"
    assert torch.isfinite(o).all(), "NaN guard failed (oracle)"
    # With only self selected, each query attends to exactly its own key -> out == v[self].
    self_v = v[abs_pos]                                          # [T, H, Dv]
    assert (g.float() - self_v.float()).abs().max() < 1e-4, "self-only attn != v[self]"
    return float((g.float() - self_v.float()).abs().max())


def test_bias_helper_shape():
    T, S, index_topk = 5, 12, 4
    abs_pos = torch.arange(S - T, S)
    topk = make_topk(torch.randn(T, S), abs_pos, index_topk)
    bias = build_sparse_attn_bias(topk, abs_pos, S, dtype=torch.float32)
    assert bias.shape == (T, S)
    # selected entries are 0, dropped are -inf; every row has >=1 finite (self).
    assert torch.isfinite(bias).any(dim=1).all(), "a row had no attendable key"
    # causal: no finite entry beyond abs_pos.
    kp = torch.arange(S).view(1, -1)
    assert not torch.isfinite(bias.masked_fill(kp <= abs_pos.view(-1, 1), 0.0)).any() or True
    fin_beyond = torch.isfinite(bias) & (kp > abs_pos.view(-1, 1))
    assert not fin_beyond.any(), "bias attends beyond causal bound"
    return bias.shape


def test_gather_width_bucketing():
    """gather_width override (bucketing K) must still match the oracle when width >= max selected."""
    T, H, Dh, Dv, S = 9, 3, 8, 8, 30
    index_topk = 6
    q = torch.randn(T, H, Dh); k = torch.randn(S, H, Dh); v = torch.randn(S, H, Dv)
    abs_pos = torch.arange(S - T, S)
    topk = make_topk(torch.randn(T, S), abs_pos, index_topk)
    out_w = sparse_mla_gather_attention(q, k, v, topk, abs_pos, Dh ** -0.5, gather_width=16)
    out_o = sparse_mla_masked_attention(q, k, v, topk, abs_pos, Dh ** -0.5)
    d = (out_w.float() - out_o.float()).abs().max().item()
    assert d < 1e-4, f"bucketed gather vs oracle {d}"
    return d


if __name__ == "__main__":
    r1 = test_gather_matches_oracle()
    r2 = test_sentinel_not_clamp()
    r3 = test_nan_guard_empty_row()
    r4 = test_bias_helper_shape()
    r5 = test_gather_width_bucketing()
    print("PASS gather==oracle max_abs_diff:", r1)
    print("PASS sentinel!=clamp selected_counts:", r2)
    print("PASS nan_guard self-only max_abs_diff:", r3)
    print("PASS bias_helper shape:", r4)
    print("PASS gather_width bucketing max_abs_diff:", r5)
    print("ALL CONSUME PARITY TESTS PASSED")
