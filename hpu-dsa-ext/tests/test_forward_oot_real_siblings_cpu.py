"""CPU cross-boundary integration test: forward_oot REAL path through the REAL
indexer_cache_hpu + indexer_metadata_hpu siblings (NO _dsa_plan_override hook).

This is the compose proof for the op<->cache<->metadata boundary: it fakes
`vllm.forward_context.get_forward_context().attn_metadata` with a TrimmedAttentionMetadata
namedtuple and a CPU-backed flat bf16 cache, then asserts forward_oot's top-k selections
(absolute-position indices) match the fp32 reference_indexer_topk oracle for both prefill
(single request, no prefix cache => ctx=0) and decode (paged history reconstructed from
block_list/block_groups/block_usage). It also confirms the topk buffer feeds the consume
side (sparse_mla_consume) on the SAME absolute-position key axis.

NO HPU device, NO model load. Run:
  python tests/test_forward_oot_real_siblings_cpu.py
"""
import collections
import os
import sys
import types

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---- Fake vllm.forward_context BEFORE importing the op (it imports lazily, but be safe) --
_fake_ctx = types.ModuleType("vllm.forward_context")


class _Ctx:
    attn_metadata = None


_fake_ctx.get_forward_context = lambda: _Ctx()
if "vllm" not in sys.modules:
    _vllm = types.ModuleType("vllm")
    _vllm.__path__ = []  # mark as package
    sys.modules["vllm"] = _vllm
sys.modules["vllm.forward_context"] = _fake_ctx

import hpu_dsa_ext.sparse_indexer_hpu as OP  # noqa: E402
from hpu_dsa_ext.sparse_indexer_hpu import reference_indexer_topk  # noqa: E402

Trim = collections.namedtuple(
    "TrimmedAttentionMetadata",
    "is_prompt slot_mapping seq_lens_tensor context_lens_tensor query_start_loc "
    "query_start_loc_p block_list block_groups block_usage block_size",
)


class _FakeIdxCache:
    """Stand-in for DeepseekV32IndexerCache: just holds the bound flat bf16 tensor."""

    def __init__(self, num_slots, D):
        self.kv_cache = torch.zeros(num_slots, D, dtype=torch.bfloat16)
        self.head_dim = D
        self.prefix = "model.layers.0.self_attn.indexer.k_cache"


class _FakeIndexer:
    """Stand-in for SparseAttnIndexer carrying exactly what forward_oot reads."""

    def __init__(self, topk, D, num_slots, max_tokens):
        self.topk_tokens = topk
        self.head_dim = D
        self.skip_k_cache_insert = False
        self.k_cache = _FakeIdxCache(num_slots, D)
        self.topk_indices_buffer = torch.full((max_tokens, topk), -1, dtype=torch.int32)


def _causal_set(row, lim):
    return sorted(x for x in row.tolist() if 0 <= x <= lim)


def test_prefill_single_request_real_siblings():
    torch.manual_seed(0)
    H, D, topk = 4, 16, 6
    qlen, blk, num_blocks = 20, 16, 4
    sm = torch.arange(qlen, dtype=torch.int64).view(1, -1)  # current keys -> slots 0..qlen-1
    qlens = torch.tensor([qlen], dtype=torch.int32)
    clens = torch.tensor([0], dtype=torch.int32)  # no prefix cache
    _Ctx.attn_metadata = Trim(True, sm, qlens, clens, None, None, None, None, None, blk)

    idxr = _FakeIndexer(topk, D, num_slots=num_blocks * blk, max_tokens=qlen + 2)
    q = torch.randn(qlen, H, D).to(torch.bfloat16)
    k = torch.randn(qlen, D).to(torch.bfloat16)  # current-step indexer keys (MQA)
    w = (torch.rand(qlen, H) + 0.1).to(torch.bfloat16)

    out = OP.forward_oot(idxr, None, q, k, w)

    abs_pos = torch.arange(qlen, dtype=torch.int32)
    ref = reference_indexer_topk(q, k, w, topk, abs_pos)
    for t in range(qlen):
        got = _causal_set(out[t], t)
        exp = sorted(x for x in ref[t].tolist() if x <= t)
        assert got == exp, f"prefill row {t}: got {got} exp {exp}"
        assert all(i <= t for i in got), f"prefill row {t} causal violation: {got}"
    assert bool((out[qlen:] == -1).all().item()), "rows beyond num_tokens must stay -1"
    print(f"[ok] prefill single-req REAL siblings: {qlen} rows, topk={topk} — set parity + causal")


def test_decode_single_request_real_siblings():
    torch.manual_seed(1)
    H, D, topk = 4, 16, 5
    hist, blk = 18, 16
    idxr = _FakeIndexer(topk, D, num_slots=4 * blk, max_tokens=1)
    kfull = torch.randn(hist, D).to(torch.bfloat16)
    idxr.k_cache.kv_cache[:hist] = kfull  # prior steps already inserted
    idxr.skip_k_cache_insert = True       # current key already present

    block_list = torch.tensor([0, 1], dtype=torch.int64)
    block_groups = torch.tensor([0, 0], dtype=torch.int64)
    block_usage = torch.tensor([16, 2], dtype=torch.float32)  # 16 + 2 = 18
    sm_d = torch.tensor([[hist - 1]], dtype=torch.int64)
    _Ctx.attn_metadata = Trim(
        False, sm_d, None, None, None, None, block_list, block_groups, block_usage, blk
    )

    q_dec = torch.randn(1, H, D).to(torch.bfloat16)
    w_dec = (torch.rand(1, H) + 0.1).to(torch.bfloat16)
    out = OP.forward_oot(idxr, None, q_dec, None, w_dec)

    abs_dec = torch.tensor([hist - 1], dtype=torch.int32)
    ref = reference_indexer_topk(q_dec, kfull, w_dec, topk, abs_dec)
    got = sorted(x for x in out[0].tolist() if x >= 0)
    exp = sorted(ref[0].tolist())
    assert got == exp, f"decode set mismatch: got {got} exp {exp}"
    assert len(got) == topk and all(i <= hist - 1 for i in got)
    print(f"[ok] decode single-req REAL siblings: history={hist}, topk={topk} — set parity + causal")


def test_topk_feeds_consume_same_axis():
    """The buffer forward_oot writes (absolute positions) is consumed by sparse_mla_consume
    on the SAME [0, S) axis with abs_query_pos = arange(S). Proves op->consume composes."""
    from hpu_dsa_ext.sparse_mla_consume import (
        sparse_mla_gather_attention,
        sparse_mla_masked_attention,
    )

    torch.manual_seed(2)
    H, D, topk = 4, 16, 6
    qlen, blk = 16, 16
    sm = torch.arange(qlen, dtype=torch.int64).view(1, -1)
    qlens = torch.tensor([qlen], dtype=torch.int32)
    clens = torch.tensor([0], dtype=torch.int32)
    _Ctx.attn_metadata = Trim(True, sm, qlens, clens, None, None, None, None, None, blk)

    idxr = _FakeIndexer(topk, D, num_slots=4 * blk, max_tokens=qlen)
    q = torch.randn(qlen, H, D).to(torch.bfloat16)
    k = torch.randn(qlen, D).to(torch.bfloat16)
    w = (torch.rand(qlen, H) + 0.1).to(torch.bfloat16)
    topk_buf = OP.forward_oot(idxr, None, q, k, w)  # [qlen, topk] absolute positions

    # consume over per-head latent kv on the SAME axis (abs_query_pos = arange).
    Dh, Dv = 16, 12
    qh = torch.randn(qlen, H, Dh)
    kh = torch.randn(qlen, H, Dh)
    vh = torch.randn(qlen, H, Dv)
    abs_pos = torch.arange(qlen, dtype=torch.int32)
    scale = Dh ** -0.5
    out_g = sparse_mla_gather_attention(qh, kh, vh, topk_buf[:qlen], abs_pos, scale)
    out_m = sparse_mla_masked_attention(qh, kh, vh, topk_buf[:qlen], abs_pos, scale)
    assert torch.isfinite(out_g).all() and torch.isfinite(out_m).all()
    diff = (out_g.float() - out_m.float()).abs().max().item()
    assert diff < 1e-3, f"gather vs masked on real topk buffer: {diff}"
    print(f"[ok] op->consume same-axis: gather==masked on real topk buffer (diff {diff:.2e})")


if __name__ == "__main__":
    test_prefill_single_request_real_siblings()
    test_decode_single_request_real_siblings()
    test_topk_feeds_consume_same_axis()
    print("\nALL forward_oot REAL-SIBLING integration tests PASSED")
