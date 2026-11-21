"""FAST synthetic end-to-end harness for the HPU DeepSeek Sparse Attention (DSA) path.

This is THE fast-iteration artifact: it exercises the WHOLE sparse path
(indexer scorer -> -1-padded top-k -> sparse consume) on SYNTHETIC tensors with a
TINY config in SECONDS, with NO model load and (by default) NO HPU device use, so it
can run anywhere and gate every code change before the 744B integration test.

What it proves
--------------
(a) CPU indexer parity: ``lightning_indexer_topk`` (tiled, bf16, the validated scorer
    we SHIP) vs ``reference_indexer_topk`` (naive fp32 oracle) -- Jaccard + top-k token
    recall, asserted >= 0.98 (contract gate (a)).
(b) Consume correctness: the gather-consume (option B, O(L*topk)) output equals the
    masked-MHA dense output (the <=32K additive -inf oracle) on a tiny seq -- they must
    be ~equal. Driven against the real ``hpu_dsa_ext.sparse_mla_consume`` if that sibling
    module is present, otherwise against a self-contained reference baked in here (so the
    harness is runnable during parallel authoring, before the consume file lands).
(c) NaN-guard + sentinel(-1) semantics: an all-empty (every-slot--1) query must produce
    FINITE output (forced self / nan->0), and a selection that excludes token 0 must NOT
    attend to token 0 (the clamp(min=0) trap).
(d) HPU-device variant (double-gated by ``torch.hpu.is_available()`` AND env
    ``HPU_DSA_TEST_DEVICE=1``): runs scorer + consume end-to-end on-device on synthetic
    metadata to prove integration without the 744B. Stays OFF by default so the fast CPU
    suite never contends with the Gaudi cards that are serving the flagship.
(e) Bonus CPU oracle vs HF ``transformers`` ``GlmMoeDsaIndexer`` (skipped if not
    importable): replicates HF's projection+RoPE pipeline, feeds the IDENTICAL projected
    (q, k, weights) to our scorer, and asserts the selected top-k index sets agree -- i.e.
    our scorer reproduces HF's selection given identical inputs.

Run
---
  standalone (no pytest needed; the container has torch but not pytest):
      python tests/test_dsa_synthetic_hpu.py
  via pytest (if installed):
      python -m pytest tests/test_dsa_synthetic_hpu.py -q
  on the HPU device (only when you OWN the cards):
      HPU_DSA_TEST_DEVICE=1 python tests/test_dsa_synthetic_hpu.py

Contract anchors: BUILD_PLAN.md (the math + MUST-FIX list) and the locked contract.
"""
from __future__ import annotations

import os
import sys

import torch

# Make ``hpu_dsa_ext`` importable when run from anywhere (standalone or pytest).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The validated scorer we SHIP -- USE it, never rewrite it (locked contract point 3).
from hpu_dsa_ext.sparse_indexer_hpu import (  # noqa: E402
    lightning_indexer_topk,
    reference_indexer_topk,
)

NEG_INF = float("-inf")
torch.manual_seed(0)

# Whether pytest is the driver. When it is, _skip() raises a real skip (clean report);
# in the standalone runner it returns a "SKIP ..." string the runner recognises.
_UNDER_PYTEST = "pytest" in sys.modules


def _skip(msg: str):
    if _UNDER_PYTEST:
        import pytest

        pytest.skip(msg)
    return f"SKIP ({msg})"


# Tests deliberately RETURN their metric (the standalone runner prints it). Silence the
# pytest>=8 PytestReturnNotNoneWarning for this module so pytest stays green.
if _UNDER_PYTEST:
    import pytest as _pytest

    pytestmark = _pytest.mark.filterwarnings("ignore::pytest.PytestReturnNotNoneWarning")


# =====================================================================================
# Tiny synthetic config (mirrors GlmMoeDsa dims, shrunk so the whole path runs in ms).
# Real GLM-5.1: index_topk=2048, index_n_heads=32, index_head_dim=128, qk_rope=64,
# first_k_dense_replace=3. DeepSeek-V3.2: index_n_heads=64. Nothing here is hardcoded
# into the SHIPPED code -- these are test-only knobs read by the synthetic builders.
# =====================================================================================
class TinyDSAConfig:
    def __init__(
        self,
        n_layers: int = 2,
        index_topk: int = 8,
        index_n_heads: int = 4,
        index_head_dim: int = 16,
        qk_rope_head_dim: int = 8,
        seq_len: int = 64,
        v_head_dim: int = 12,
        first_k_dense_replace: int = 3,
    ):
        self.n_layers = n_layers
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.seq_len = seq_len
        self.v_head_dim = v_head_dim
        self.first_k_dense_replace = first_k_dense_replace


TINY = TinyDSAConfig()


# =====================================================================================
# Synthetic tensor + metadata factory.
# =====================================================================================
def make_indexer_inputs(cfg: TinyDSAConfig, device="cpu", dtype=torch.float32, seed=0):
    """Produce the post-projection (q, k, weights, abs_pos) the Indexer hands the scorer.

    Mirrors ``Indexer.forward`` (vllm_core_deepseek_v2.py:678-743): q is
    [T, n_head, head_dim] (here returned bf16-castable; q_scale folded so it's used as bf16
    directly), k is [S, head_dim] (MQA single key), weights is [T, n_head] already carrying
    softmax_scale * n_head**-0.5 (q_scale==1 in synthetic). abs_pos is a contiguous prefill:
    a single sequence so query t sits at absolute position t (keys == queries here).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = S = cfg.seq_len
    H, D = cfg.index_n_heads, cfg.index_head_dim
    q = torch.randn(T, H, D, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(S, D, generator=g).to(device=device, dtype=dtype)
    # weights folded with softmax_scale * n_head**-0.5 (positive -> ReLU-order-preserving);
    # randn->abs keeps them positive like the trained weights_proj * positive scales path.
    softmax_scale = D ** -0.5
    raw_w = torch.randn(T, H, generator=g).abs()
    weights = (raw_w * softmax_scale * H ** -0.5).to(device=device, dtype=dtype)
    abs_pos = torch.arange(T, device=device, dtype=torch.int32)  # single-seq prefill
    return q, k, weights, abs_pos


def make_consume_inputs(cfg: TinyDSAConfig, device="cpu", dtype=torch.float32, seed=1):
    """Per-head latent KV the consume side attends over (q[T,H,Dh], k[S,H,Dh], v[S,H,Dv]).

    The consume operates AFTER MLA up-projection so keys/values are per-head; abs_pos is the
    same contiguous single-seq prefill mapping (query t -> latent key t).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = S = cfg.seq_len
    H, Dh, Dv = cfg.index_n_heads, cfg.index_head_dim, cfg.v_head_dim
    q = torch.randn(T, H, Dh, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(S, H, Dh, generator=g).to(device=device, dtype=dtype)
    v = torch.randn(S, H, Dv, generator=g).to(device=device, dtype=dtype)
    abs_pos = torch.arange(T, device=device, dtype=torch.int32)
    return q, k, v, abs_pos


# =====================================================================================
# Self-contained REFERENCE consume (used iff hpu_dsa_ext.sparse_mla_consume is absent).
# These deliberately re-derive the contract semantics (sentinel-scatter, NaN guard) so the
# harness can run during parallel authoring AND so its expectations are independent of the
# module under test. When the sibling lands, we prefer it (real integration).
# =====================================================================================
def _ref_build_keep_mask(topk: torch.Tensor, abs_pos: torch.Tensor, S: int) -> torch.Tensor:
    """[T, S] bool keep-mask from -1-padded topk via SENTINEL-COLUMN scatter (contract 5).

    Append a dummy key column S, scatter selected ids (mapping -1 -> the sentinel col so it
    is harmlessly absorbed), slice the sentinel off -> NEVER touches token 0. Then force
    self-inclusion (>=1 valid key, contract 6).
    """
    T = topk.shape[0]
    keep = torch.zeros(T, S + 1, dtype=torch.bool, device=topk.device)
    idx = topk.long().clone()
    idx[idx < 0] = S  # sentinel column for -1 pads (NOT clamp(min=0))
    keep.scatter_(1, idx, True)
    keep = keep[:, :S]  # drop sentinel
    # causal safety: never keep a key beyond this query's absolute position
    key_pos = torch.arange(S, device=topk.device).view(1, -1)
    keep &= key_pos <= abs_pos.view(-1, 1).long()
    # NaN guard: guarantee >=1 valid key (self) for every query
    self_col = abs_pos.long().clamp(max=S - 1).view(-1, 1)
    keep.scatter_(1, self_col, True)
    return keep


def _ref_masked_attention(q, k, v, topk, abs_pos, scale):
    """Masked-MHA oracle: additive -inf bias on non-selected keys, dense softmax (contract 7).

    q [T,H,Dh], k [S,H,Dh], v [S,H,Dv] -> [T,H,Dv]. This is the <=32K correctness oracle.
    """
    T, H, _ = q.shape
    S = k.shape[0]
    keep = _ref_build_keep_mask(topk, abs_pos, S)  # [T, S]
    # scores [T, H, S]
    scores = torch.einsum("thd,shd->ths", q.float(), k.float()) * scale
    bias = torch.where(keep, 0.0, NEG_INF).view(T, 1, S)
    scores = scores + bias
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # belt-and-suspenders NaN guard
    out = torch.einsum("ths,shd->thd", attn, v.float())
    return out.to(q.dtype)


def _ref_gather_attention(q, k, v, topk, abs_pos, scale, gather_width=None):
    """Gather-consume (option B): gather the <=topk selected latent rows per query, attend.

    Uses the sentinel-column trick to map -1 pads to an appended all-zero key/value row that
    is then masked out, so padded slots contribute nothing and token 0 is never spuriously
    selected. O(T * topk) instead of O(T * S).
    """
    T, H, Dh = q.shape
    S = k.shape[0]
    Dv = v.shape[-1]
    topk = topk.long()
    width = gather_width if gather_width is not None else topk.shape[1]
    if width < topk.shape[1]:
        topk = topk[:, :width]
    elif width > topk.shape[1]:
        pad = torch.full((T, width - topk.shape[1]), -1, dtype=topk.dtype, device=topk.device)
        topk = torch.cat([topk, pad], dim=1)

    # causal-validate + self-inclusion on the *gathered* representation
    key_pos = torch.arange(S, device=q.device).view(1, -1)
    keep_full = _ref_build_keep_mask(topk, abs_pos, S)  # [T, S] (already self-incl + causal)
    # Rebuild a per-slot id list that respects keep_full: take topk ids that survive, else -1.
    sel = topk.clone()
    sel_valid = (sel >= 0) & (sel < S)
    # mask out ids the keep-mask rejected (beyond-causal); they become -1 (sentinel)
    gathered_keep = torch.zeros_like(sel, dtype=torch.bool)
    in_range = sel.clamp(min=0, max=S - 1)
    gathered_keep = sel_valid & torch.gather(keep_full, 1, in_range)
    # ensure each row has self present in its gather slots (use first slot if missing)
    self_id = abs_pos.long().clamp(max=S - 1)
    has_self = (sel == self_id.view(-1, 1)).any(dim=1)
    if (~has_self).any():
        sel = sel.clone()
        gathered_keep = gathered_keep.clone()
        sel[~has_self, 0] = self_id[~has_self]
        gathered_keep[~has_self, 0] = True

    # SENTINEL gather: append a zero row at index S for the -1/invalid slots.
    k_ext = torch.cat([k, torch.zeros(1, H, Dh, dtype=k.dtype, device=k.device)], dim=0)
    v_ext = torch.cat([v, torch.zeros(1, H, Dv, dtype=v.dtype, device=v.device)], dim=0)
    gather_id = torch.where(gathered_keep, sel, torch.full_like(sel, S))  # [T, width]
    # gather per-head: [T, width, H, D]
    kg = k_ext[gather_id]  # [T, width, H, Dh]
    vg = v_ext[gather_id]  # [T, width, H, Dv]

    # scores [T, H, width]
    scores = torch.einsum("thd,twhd->thw", q.float(), kg.float()) * scale
    bias = torch.where(gathered_keep.view(T, 1, width), 0.0, NEG_INF)
    scores = scores + bias
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.einsum("thw,twhd->thd", attn, vg.float())
    return out.to(q.dtype)


# Prefer the real consume module (true integration); fall back to the reference above.
try:
    from hpu_dsa_ext.sparse_mla_consume import (  # noqa: E402
        sparse_mla_gather_attention as _gather_attention,
        sparse_mla_masked_attention as _masked_attention,
    )
    try:
        from hpu_dsa_ext.sparse_mla_consume import _build_keep_mask as _keep_mask  # noqa: E402
    except Exception:  # noqa: BLE001
        _keep_mask = _ref_build_keep_mask
    CONSUME_SRC = "hpu_dsa_ext.sparse_mla_consume"
except Exception:  # noqa: BLE001
    _gather_attention = _ref_gather_attention
    _masked_attention = _ref_masked_attention
    _keep_mask = _ref_build_keep_mask
    CONSUME_SRC = "builtin-reference (sparse_mla_consume not yet present)"


# =====================================================================================
# Metrics
# =====================================================================================
def _topk_set_metrics(a: torch.Tensor, b: torch.Tensor, abs_pos: torch.Tensor | None = None):
    """Per-row Jaccard + recall between two [T, k] index sets, ignoring -1 pads.

    Returns (mean_jaccard, mean_recall) with recall = |a ∩ b| / |b| (b = the oracle).

    IMPORTANT: both ``a`` and ``b`` are restricted to the per-row causal bound
    (key id <= abs_pos[t]) when ``abs_pos`` is given. The naive oracles
    (``reference_indexer_topk`` and HF ``GlmMoeDsaIndexer``) return raw
    ``topk().indices``, which for SHORT rows (fewer causal keys than k) include
    arbitrary indices pointing at the -inf-masked future positions. Those are NOT
    genuine selections -- ``lightning_indexer_topk`` correctly emits -1 there -- so
    comparing against them un-filtered spuriously deflates the score. Filtering both
    to the causal set is the mathematically faithful comparison.
    """
    T = a.shape[0]
    jac, rec = [], []
    for t in range(T):
        sa = {int(x) for x in a[t].tolist() if int(x) >= 0}
        sb = {int(x) for x in b[t].tolist() if int(x) >= 0}
        if abs_pos is not None:
            lim = int(abs_pos[t])
            sa = {x for x in sa if x <= lim}
            sb = {x for x in sb if x <= lim}
        if not sb:
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        jac.append(inter / union if union else 1.0)
        rec.append(inter / len(sb))
    mj = sum(jac) / len(jac) if jac else 1.0
    mr = sum(rec) / len(rec) if rec else 1.0
    return mj, mr


# =====================================================================================
# (a) CPU indexer parity: shipped scorer vs naive fp32 oracle.
# =====================================================================================
def test_indexer_parity_cpu():
    cfg = TINY
    q, k, weights, abs_pos = make_indexer_inputs(cfg)
    got = lightning_indexer_topk(q, k, weights, cfg.index_topk, abs_pos, tile_m=8)
    ref = reference_indexer_topk(q, k, weights, cfg.index_topk, abs_pos)
    assert got.shape == (cfg.seq_len, cfg.index_topk), got.shape
    assert got.dtype == torch.int32, got.dtype
    mj, mr = _topk_set_metrics(got, ref, abs_pos)
    # contract gate (a): Jaccard + top-k recall >= 0.98
    assert mj >= 0.98, f"indexer Jaccard {mj:.4f} < 0.98"
    assert mr >= 0.98, f"indexer top-k recall {mr:.4f} < 0.98"
    return mj, mr


def test_indexer_short_rows_padding_cpu():
    """Rows with fewer causal keys than index_topk MUST be -1 padded (consume relies on it)."""
    cfg = TINY
    q, k, weights, abs_pos = make_indexer_inputs(cfg)
    got = lightning_indexer_topk(q, k, weights, cfg.index_topk, abs_pos, tile_m=8)
    # query at abs pos p has p+1 causal keys; rows with p+1 < index_topk must show -1 pads.
    for t in range(cfg.index_topk - 1):  # these rows are short
        n_valid = int((got[t] >= 0).sum())
        expected = min(t + 1, cfg.index_topk)
        assert n_valid == expected, f"row {t}: {n_valid} valid != expected {expected}"
        # no selected id may exceed the causal bound
        assert int(got[t].max()) <= t, f"row {t} selected a future key {int(got[t].max())}"
    return True


# =====================================================================================
# (b) Consume correctness: gather == masked-MHA dense (end-to-end through the scorer's topk).
# =====================================================================================
def _end_to_end_topk(cfg, device="cpu"):
    """Run the REAL scorer to get the -1-padded topk that the consume will receive."""
    q_i, k_i, w_i, abs_pos = make_indexer_inputs(cfg, device=device)
    topk = lightning_indexer_topk(q_i, k_i, w_i, cfg.index_topk, abs_pos, tile_m=8)
    return topk, abs_pos


def test_consume_gather_matches_masked_cpu():
    cfg = TINY
    topk, abs_pos = _end_to_end_topk(cfg)
    q, k, v, _ = make_consume_inputs(cfg)
    scale = cfg.index_head_dim ** -0.5
    out_g = _gather_attention(q, k, v, topk, abs_pos, scale)
    out_m = _masked_attention(q, k, v, topk, abs_pos, scale)
    assert out_g.shape == (cfg.seq_len, cfg.index_n_heads, cfg.v_head_dim), out_g.shape
    assert torch.isfinite(out_g).all(), "gather consume produced non-finite"
    assert torch.isfinite(out_m).all(), "masked consume produced non-finite"
    diff = (out_g.float() - out_m.float()).abs().max().item()
    assert diff < 1e-3, f"gather vs masked-MHA max abs diff {diff} (consume={CONSUME_SRC})"
    return diff


# =====================================================================================
# (c) NaN guard + sentinel(-1) semantics.
# =====================================================================================
def test_nan_guard_all_empty_row_cpu():
    """A query whose every selected slot is -1 must still yield FINITE output (forced self)."""
    cfg = TINY
    q, k, v, abs_pos = make_consume_inputs(cfg)
    topk = torch.full((cfg.seq_len, cfg.index_topk), -1, dtype=torch.int32)  # ALL empty
    scale = cfg.index_head_dim ** -0.5
    out_g = _gather_attention(q, k, v, topk, abs_pos, scale)
    out_m = _masked_attention(q, k, v, topk, abs_pos, scale)
    assert torch.isfinite(out_g).all(), "NaN guard failed (gather)"
    assert torch.isfinite(out_m).all(), "NaN guard failed (masked)"
    # with only self selected, each query attends to exactly its own key -> out == v[self]
    self_v = v[abs_pos.long()]  # [T, H, Dv]
    dg = (out_g.float() - self_v.float()).abs().max().item()
    dm = (out_m.float() - self_v.float()).abs().max().item()
    assert dg < 1e-3, f"self-only gather != v[self] ({dg})"
    assert dm < 1e-3, f"self-only masked != v[self] ({dm})"
    return dg, dm


def test_sentinel_not_clamp_cpu():
    """Sentinel-scatter must NOT mark token 0 selected when 0 is not in the topk (clamp trap)."""
    cfg = TINY
    S = cfg.seq_len
    abs_pos = torch.full((S,), S - 1, dtype=torch.int32)  # all queries see full context
    topk = torch.full((S, cfg.index_topk), -1, dtype=torch.int32)
    # select ids {5, 9, 13} only; the rest are -1 pads. Token 0 is deliberately excluded.
    topk[:, 0] = 5
    topk[:, 1] = 9
    topk[:, 2] = 13
    keep = _keep_mask(topk, abs_pos, S)
    assert not bool(keep[:, 0].any()), "sentinel-scatter wrongly selected token 0 (clamp bug)"
    # {5,9,13} plus forced self (S-1) -> 4 keys per row.
    counts = keep.sum(dim=1)
    assert (counts == 4).all(), f"expected 4 kept/row, got {counts.tolist()[:5]}"
    return counts[0].item()


# =====================================================================================
# (d) HPU-device end-to-end variant -- double-gated (is_available AND env opt-in) so the
#     fast CPU suite never contends with the serving Gaudi cards.
# =====================================================================================
def _hpu_enabled():
    if os.environ.get("HPU_DSA_TEST_DEVICE", "0") not in ("1", "true", "True"):
        return False
    try:
        return bool(torch.hpu.is_available())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False


def test_hpu_device_end_to_end():
    """forward path (scorer -> topk -> consume) on the HPU device on synthetic metadata.

    Proves the integration runs on-device (bf16) and agrees with a CPU run -- WITHOUT the
    744B. Skipped unless HPU_DSA_TEST_DEVICE=1 (so it never touches the serving cards by
    default). Returns a status string for the standalone runner.
    """
    if not _hpu_enabled():
        return _skip("HPU_DSA_TEST_DEVICE unset or no HPU")
    cfg = TinyDSAConfig(seq_len=48, index_topk=8, index_n_heads=4, index_head_dim=16, v_head_dim=12)
    dev = "hpu"
    # --- CPU reference run (fp32) ---
    q_i, k_i, w_i, abs_pos = make_indexer_inputs(cfg, device="cpu", dtype=torch.float32)
    topk_cpu = lightning_indexer_topk(q_i, k_i, w_i, cfg.index_topk, abs_pos, tile_m=16)
    qc, kc, vc, _ = make_consume_inputs(cfg, device="cpu", dtype=torch.float32)
    scale = cfg.index_head_dim ** -0.5
    out_cpu = _gather_attention(qc, kc, vc, topk_cpu, abs_pos, scale)

    # --- HPU run (bf16, the serving dtype) ---
    q_h = q_i.to(device=dev, dtype=torch.bfloat16)
    k_h = k_i.to(device=dev, dtype=torch.bfloat16)
    w_h = w_i.to(device=dev, dtype=torch.bfloat16)
    abs_h = abs_pos.to(dev)
    topk_hpu = lightning_indexer_topk(q_h, k_h, w_h, cfg.index_topk, abs_h, tile_m=16)
    assert topk_hpu.device.type == "hpu", topk_hpu.device
    qh = qc.to(device=dev, dtype=torch.bfloat16)
    kh = kc.to(device=dev, dtype=torch.bfloat16)
    vh = vc.to(device=dev, dtype=torch.bfloat16)
    out_hpu = _gather_attention(qh, kh, vh, topk_hpu, abs_h, scale).to("cpu").float()

    assert torch.isfinite(out_hpu).all(), "HPU consume produced non-finite"
    # top-k SET agreement CPU<->HPU (bf16 vs fp32 ties allowed): recall must stay high.
    mj, mr = _topk_set_metrics(topk_hpu.to("cpu"), topk_cpu, abs_pos)
    assert mr >= 0.95, f"HPU vs CPU top-k recall {mr:.4f} < 0.95"
    # attention output agreement (bf16 tolerance).
    # Consume PRECISION must hold TOP-K FIXED across devices: comparing HPU output (HPU top-k)
    # against CPU output (a DIFFERENT but equally-valid CPU top-k) conflates benign bf16-vs-fp32
    # indexer boundary churn with consume error — and on a tiny config one swapped token swings the
    # output by ~O(1) (verified: same-topk diff 0.004 vs diff-topk 1.4). Indexer agreement is gated
    # SEPARATELY by the recall assertion above; here we isolate true consume precision.
    out_cpu_same = _gather_attention(qc, kc, vc, topk_hpu.to("cpu"), abs_pos, scale).float()
    diff = (out_hpu - out_cpu_same).abs().max().item()
    assert diff < 5e-2, f"HPU vs CPU consume out (same top-k) max abs diff {diff}"
    return f"OK recall={mr:.4f} consume_out_diff={diff:.4e}"


# =====================================================================================
# (e) Bonus CPU oracle vs HF transformers GlmMoeDsaIndexer (skipped if not importable).
# =====================================================================================
def _hf_indexer_available():
    try:
        import transformers  # noqa: F401
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: F401
            GlmMoeDsaIndexer,
        )
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (  # noqa: F401
            GlmMoeDsaConfig,
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def test_indexer_vs_hf_oracle_cpu():
    """Our scorer must reproduce HF GlmMoeDsaIndexer's top-k selection on IDENTICAL inputs.

    Strategy: build a tiny real ``GlmMoeDsaConfig`` + ``GlmMoeDsaIndexer``, run its forward
    (which projects + RoPEs internally and returns top-k), then replicate the SAME
    projection+RoPE here to recover the post-RoPE (q, k, weights) and feed them to
    ``lightning_indexer_topk``. Folding HF's softmax_scale into the (positive) weights is
    ReLU-order-preserving, so the selected index sets must agree (modulo ties).
    """
    if not _hf_indexer_available():
        return _skip("transformers GlmMoeDsaIndexer not importable")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaIndexer,
        apply_rotary_pos_emb,
    )
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    torch.manual_seed(7)
    B, S = 1, 48
    H, D, rope = 4, 16, 8
    topk_k = 8
    cfg = GlmMoeDsaConfig(
        hidden_size=64,
        q_lora_rank=32,
        index_n_heads=H,
        index_head_dim=D,
        qk_rope_head_dim=rope,
        index_topk=topk_k,
        num_hidden_layers=4,
        indexer_types=["full", "full", "full", "full"],
    )
    idx = GlmMoeDsaIndexer(cfg, layer_idx=0).eval().float()

    hidden = torch.randn(B, S, cfg.hidden_size)
    q_resid = torch.randn(B, S, cfg.q_lora_rank)
    # cos/sin for the indexer's NeoX RoPE over the rope dims; positions 0..S-1.
    pos = torch.arange(S).float()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope, 2).float() / rope))
    ang = torch.outer(pos, inv_freq)  # [S, rope/2]
    emb = torch.cat([ang, ang], dim=-1)  # [S, rope]
    cos = emb.cos().unsqueeze(0)  # [B, S, rope]
    sin = emb.sin().unsqueeze(0)
    # causal additive mask [B, 1, S, S] -> HF adds it to index_scores (broadcast [B,S,S]).
    causal = torch.full((S, S), NEG_INF).triu(1).view(1, 1, S, S)

    with torch.no_grad():
        hf_topk = idx(hidden, q_resid, (cos, sin), attention_mask=causal[:, 0])  # [B,S,topk]

    # --- replicate HF projection+RoPE to recover (q,k,weights) for OUR scorer ---
    with torch.no_grad():
        q = idx.wq_b(q_resid).view(B, S, H, D)
        q_pe, q_nope = torch.split(q, [rope, D - rope], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)[0]  # [S, H, D]
        k = idx.k_norm(idx.wk(hidden))  # [B, S, D]
        k_pe, k_nope = torch.split(k, [rope, D - rope], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)[0]  # [S, D]
        # HF: weights = weights_proj(x) * n_heads**-0.5 ; score folds softmax_scale at einsum.
        # Our scorer folds BOTH into weights (ReLU-order-preserving, positive scale).
        w = idx.weights_proj(hidden).float()[0] * (H ** -0.5) * idx.softmax_scale  # [S, H]

    abs_pos = torch.arange(S, dtype=torch.int32)
    our_topk = lightning_indexer_topk(q, k, w, topk_k, abs_pos, tile_m=16)
    mj, mr = _topk_set_metrics(our_topk, hf_topk[0], abs_pos)
    assert mr >= 0.98, f"our-scorer vs HF GlmMoeDsaIndexer recall {mr:.4f} < 0.98"
    assert mj >= 0.95, f"our-scorer vs HF GlmMoeDsaIndexer Jaccard {mj:.4f} < 0.95"
    return f"OK jaccard={mj:.4f} recall={mr:.4f}"


# =====================================================================================
# Standalone runner (pytest not installed in the container image).
# =====================================================================================
_TESTS = [
    ("a  indexer_parity_cpu", test_indexer_parity_cpu),
    ("a  indexer_short_rows_padding_cpu", test_indexer_short_rows_padding_cpu),
    ("b  consume_gather==masked_cpu", test_consume_gather_matches_masked_cpu),
    ("c  nan_guard_all_empty_row_cpu", test_nan_guard_all_empty_row_cpu),
    ("c  sentinel_not_clamp_cpu", test_sentinel_not_clamp_cpu),
    ("d  hpu_device_end_to_end", test_hpu_device_end_to_end),
    ("e  indexer_vs_hf_oracle_cpu", test_indexer_vs_hf_oracle_cpu),
]


def main() -> int:
    print(f"[dsa-synthetic] consume backend = {CONSUME_SRC}")
    failed = 0
    for name, fn in _TESTS:
        try:
            res = fn()
            tag = "PASS"
            if isinstance(res, str) and res.startswith("SKIP"):
                tag = "SKIP"
            print(f"  {tag} {name}: {res}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    if failed:
        print(f"[dsa-synthetic] {failed} test(s) FAILED")
        return 1
    print("[dsa-synthetic] ALL SYNTHETIC DSA TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
