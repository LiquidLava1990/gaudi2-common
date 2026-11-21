from types import SimpleNamespace

import torch

from hpu_moe_ext.moe_wna16_hpu import _hpu_safe_select_experts


def test_sigmoid_bias_normalization_and_scaling_survive_degenerate_grouping():
    logits = torch.tensor([[2.0, 1.0, -1.0, 0.0]])
    hidden = torch.zeros((1, 8))
    router = SimpleNamespace(
        scoring_func="sigmoid",
        top_k=2,
        e_score_correction_bias=torch.tensor([0.0, 0.0, 10.0, 0.0]),
        renormalize=True,
        routed_scaling_factor=2.5,
        num_fused_shared_experts=0,
    )
    layer = SimpleNamespace(
        router=router,
        use_grouped_topk=False,
        top_k=2,
    )

    weights, ids = _hpu_safe_select_experts(layer, hidden, logits)

    assert ids.tolist() == [[2, 0]]
    unbiased = torch.sigmoid(logits).gather(1, ids)
    expected = unbiased / unbiased.sum(dim=-1, keepdim=True) * 2.5
    torch.testing.assert_close(weights, expected)


def test_plain_softmax_fallback_preserves_legacy_hpu_semantics():
    logits = torch.tensor([[0.5, 3.0, 1.0]])
    hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        router=SimpleNamespace(top_k=2, scoring_func="softmax"),
        use_grouped_topk=False,
        top_k=2,
        renormalize=True,
    )

    weights, ids = _hpu_safe_select_experts(layer, hidden, logits)

    assert ids.tolist() == [[1, 2]]
    expected = torch.softmax(logits.float(), dim=-1).gather(1, ids)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(weights.float(), expected, rtol=0.004, atol=0.004)
    assert weights.dtype == torch.bfloat16


def test_fused_shared_expert_slots_are_retained():
    logits = torch.tensor([[4.0, 1.0, 0.0]])
    hidden = torch.zeros((1, 4))
    router = SimpleNamespace(
        scoring_func="softmax",
        top_k=1,
        renormalize=True,
        num_fused_shared_experts=2,
        global_num_experts=3,
        shared_expert_weight=0.25,
    )
    layer = SimpleNamespace(router=router, use_grouped_topk=False, top_k=1)

    weights, ids = _hpu_safe_select_experts(layer, hidden, logits)

    assert ids.tolist() == [[0, 3, 4]]
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.25, 0.25]]))


def test_grouped_router_keeps_its_existing_hpu_selector():
    expected_weights = torch.tensor([[0.7, 0.3]])
    expected_ids = torch.tensor([[4, 1]])

    def select_experts(**kwargs):
        assert kwargs["hidden_states"].shape == (1, 4)
        assert kwargs["router_logits"].shape == (1, 6)
        return expected_weights, expected_ids

    layer = SimpleNamespace(
        router=SimpleNamespace(select_experts=select_experts),
        use_grouped_topk=True,
    )
    weights, ids = _hpu_safe_select_experts(
        layer, torch.zeros((1, 4)), torch.zeros((1, 6)))

    assert weights is expected_weights
    assert ids is expected_ids


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} HPU router semantic tests")
