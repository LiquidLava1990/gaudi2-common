"""EXPERT-PRUNE MASK — reversible, card-free QUALITY GATE for WAVE-2 calibrated expert pruning.

Companion to prune_experts.py (the physical .pt transform). Where the transform PHYSICALLY drops the
low-saliency experts (the dequant + HBM win, shipped once a ratio passes the gate), THIS patch MASKS
the same experts OUT OF ROUTING in-place at load — so you can measure the QUALITY of any prune ratio
against the BASE checkpoint with NO 310 GB regeneration. The masked model's OUTPUT distribution is
IDENTICAL to the physically-pruned model (same survivors selected, same norm_topk renorm over them);
only the dead experts stay resident (no dequant/HBM win — that is the transform's job). So:

    perplexity / 32k-needle / coherence measured under HPU_PRUNE_EXPERTS  ==  the shipped prune's
    quality  ->  sweep ratios cheaply here, pick the one that passes, THEN run prune_experts.py once.

MECHANISM (GLM-5.2 is topk_method="noaux_tc", scoring_func="sigmoid"): FusedMoE.select_experts picks
the top-k of  scores_for_choice = sigmoid(router_logits) + e_score_correction_bias  (the bias is the
aux-loss-free balance term). Setting  e_score_correction_bias[pruned] = -1e30  guarantees a pruned
expert can NEVER enter the top-k (its choice score is -1e30 << any live expert), so it is never
routed and contributes 0 — EXACTLY the physical drop. The ACTUAL routing weight is sigmoid(logits) of
the SELECTED experts, renormalized over them (norm_topk_prob), so survivors renorm to sum-1 just as in
the pruned checkpoint. e_score_correction_bias is the SAME [n_routed] tensor the layer hands to
FusedMoE (by reference), and pruned ids are GLOBAL (the gate is replicated, not EP-sharded), so one
in-place edit per pruned layer excludes them on every rank. We ALSO mask the gate's returned logit
columns to -1e30 (belt-and-suspenders + the only needed path for a model with NO correction bias,
where select_experts tops sigmoid(logits) directly).

ENV (default UNSET -> pure no-op, byte-for-byte the stock routing):
  HPU_PRUNE_EXPERTS=<path to droppable-set.json from reap_saliency.py>
      { "drop": { "<layer>": [global expert ids...], ... }, ... }   (the "keep" half is ignored here)

LOUD validation (DSA build quality bar): a malformed set, an out-of-range expert id, or a pruned id on
a non-MoE / dense / out-of-range layer raises RuntimeError at install — never a silently wrong route.

BYTE-EXACTNESS: when HPU_PRUNE_EXPERTS is unset, install() returns immediately and NOTHING is patched.
When set, the ONLY change is the per-layer one-shot bias/logit mask on the listed experts; all other
values are identical. Mirrors layer_prune_patch.py's monkeypatch + first-construction-resolve contract.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("vllm")

_ENV = "HPU_PRUNE_EXPERTS"
_NEG = -1.0e30          # finite "-inf" for the choice score (avoids any -inf*0 NaN in vectorized paths)


# ==================================================================================================
# PURE parsing/validation core (card-free testable; no torch / no vllm)
# ==================================================================================================
def parse_drop_set(raw: dict) -> dict[int, set[int]]:
    """{layer:int -> set(global expert ids)} from a droppable-set's 'drop' map. LOUD on garbage."""
    if not isinstance(raw, dict) or "drop" not in raw:
        raise ValueError("droppable-set JSON must be an object with a 'drop' key")
    out: dict[int, set[int]] = {}
    for k, ids in raw["drop"].items():
        try:
            L = int(k)
        except Exception as ex:  # noqa: BLE001
            raise ValueError(f"bad layer key {k!r} in 'drop' ({ex})")
        if not isinstance(ids, (list, tuple)):
            raise ValueError(f"layer {L}: 'drop' value must be a list of ids, got {type(ids).__name__}")
        s = set()
        for e in ids:
            ei = int(e)
            if ei < 0:
                raise ValueError(f"layer {L}: negative expert id {ei}")
            s.add(ei)
        if s:
            out[L] = s
    if not out:
        raise ValueError("droppable-set has no experts to drop (empty 'drop')")
    return out


def validate_against_config(drop: dict[int, set[int]], num_hidden_layers: int,
                            first_k_dense_replace: int, n_routed_experts: int) -> None:
    """Raise RuntimeError (LOUD) if any pruned id/layer is invalid for THIS model:
      * layer in the dense band [0, first_k_dense_replace) — no routed experts there.
      * layer index > num_hidden_layers — beyond the MTP/nextn layer.
      * a pruned expert id >= n_routed_experts.
      * pruning EVERY expert of a layer (would delete the MoE)."""
    problems = []
    for L, ids in sorted(drop.items()):
        if L < int(first_k_dense_replace):
            problems.append(f"layer {L} is in the dense band [0,{first_k_dense_replace}) (no experts)")
        if L > int(num_hidden_layers):
            problems.append(f"layer {L} > num_hidden_layers={num_hidden_layers} (beyond MTP)")
        oob = sorted(e for e in ids if e >= int(n_routed_experts))
        if oob:
            problems.append(f"layer {L}: expert id(s) {oob[:5]} >= n_routed_experts={n_routed_experts}")
        if len(ids) >= int(n_routed_experts):
            problems.append(f"layer {L}: pruning all {len(ids)} experts would delete the MoE")
    if problems:
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}: invalid droppable-set:\n  - " + "\n  - ".join(problems))


def _layer_idx_from_prefix(prefix: str) -> int | None:
    """'model.layers.{L}.mlp' -> L (the MoE module prefix). None if it doesn't match."""
    import re
    m = re.search(r"\.layers\.(\d+)\.mlp$", prefix)
    return int(m.group(1)) if m else None


# ==================================================================================================
# INSTALL — wrap DeepseekV2MoE.__init__ to apply the one-shot mask once weights are loaded.
# ==================================================================================================
def install() -> None:
    spec = os.environ.get(_ENV, "").strip()
    if not spec:
        return  # BYTE-EXACT: env unset -> touch nothing; stock routing runs unchanged.

    if not os.path.isfile(spec):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}={spec!r} is not a readable file "
                           f"(want the droppable-set JSON from reap_saliency.py --out).")
    try:
        raw = json.load(open(spec))
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}: failed to parse {spec!r}: {ex}") from ex
    drop = parse_drop_set(raw)

    try:
        from vllm.model_executor.models import deepseek_v2 as _dv2
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"[hpu-moe-ext] {_ENV} set but deepseek_v2 unavailable: {ex!r}")

    MoE = _dv2.DeepseekV2MoE
    if getattr(MoE.__init__, "_hpu_expert_prune", False):
        return

    import torch  # noqa: F401
    _orig_init = MoE.__init__
    _state = {"validated": False, "n_masked_layers": 0, "logged": False}

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix")
        if prefix is None and args:
            # prefix is the last positional in DeepseekV2MoE.__init__ (config, parallel_config,
            # quant_config, prefix); fall back to scanning args for the 'model.layers.N.mlp' string.
            for a in args:
                if isinstance(a, str) and ".mlp" in a:
                    prefix = a
                    break
        L = _layer_idx_from_prefix(prefix) if isinstance(prefix, str) else None

        # One-time validation against the live config (now that a layer is built).
        if not _state["validated"]:
            cfg = getattr(self, "config", None) or getattr(args[0] if args else None, "hf_config", None)
            # self.n_routed_experts is set by the stock __init__; prefer it.
            n_routed = int(getattr(self, "n_routed_experts", 0)) or 256
            num_hidden = int(getattr(cfg, "num_hidden_layers", 78)) if cfg else 78
            fkd = int(getattr(cfg, "first_k_dense_replace", 3)) if cfg else 3
            validate_against_config(drop, num_hidden, fkd, n_routed)
            _state["validated"] = True

        if L is None or L not in drop:
            return  # this layer is not pruned

        pruned = sorted(drop[L])
        # Stash for the forward-time logit mask (the no-bias fallback path).
        self._hpu_pruned_experts = torch.tensor(pruned, dtype=torch.long)

        # PRIMARY (noaux_tc): one-shot, post-construction in-place bias edit. The bias param is the
        # SAME tensor handed to FusedMoE (by reference), so this excludes the pruned experts from the
        # top-k choice on every rank. Done HERE (after _orig_init built the param); the weight loader
        # overwrites bias data LATER, so we also (re)apply on first forward via the gate wrap below.
        bias = getattr(getattr(self, "gate", None), "e_score_correction_bias", None)
        if bias is not None:
            with torch.no_grad():
                bias.data[self._hpu_pruned_experts.to(bias.device)] = _NEG

        # Wrap the gate forward so that (a) AFTER weight-load the bias mask is re-asserted once, and
        # (b) the returned router logits have the pruned columns set to _NEG (covers a no-bias model
        # and makes the pruned sigmoid weight exactly 0). Idempotent + cheap (a scatter on [.,256]).
        gate = self.gate
        if not getattr(gate, "_hpu_expert_prune_wrapped", False):
            _gate_fwd = gate.forward
            _applied = {"bias": False}
            _pruned_idx = self._hpu_pruned_experts

            def _wrapped(x, *a, **k):
                out = _gate_fwd(x, *a, **k)
                logits = out[0] if isinstance(out, tuple) else out
                idx = _pruned_idx.to(logits.device)
                if not _applied["bias"]:
                    b = getattr(gate, "e_score_correction_bias", None)
                    if b is not None:
                        with torch.no_grad():
                            b.data[idx.to(b.device)] = _NEG
                    _applied["bias"] = True
                logits = logits.index_fill(-1, idx, _NEG)
                return (logits,) + tuple(out[1:]) if isinstance(out, tuple) else logits

            gate.forward = _wrapped
            gate._hpu_expert_prune_wrapped = True

        _state["n_masked_layers"] += 1
        if not _state["logged"]:
            logger.warning(
                "[hpu-moe-ext] %s ACTIVE — masking pruned experts out of routing (bias+logit -> %.0e) "
                "for QUALITY-GATE measurement on the BASE ckpt; this is FAITHFUL to the physical prune "
                "(same survivors+renorm) but does NOT free HBM/dequant (run prune_experts.py for that). "
                "First layer %s drops %d experts. ON-CARD PERPLEXITY+COHERENCE GATE REQUIRED.",
                _ENV, _NEG, L, len(pruned))
            _state["logged"] = True

    __init__._hpu_expert_prune = True
    MoE.__init__ = __init__
    logger.info("[hpu-moe-ext] patched DeepseekV2MoE.__init__ (%s expert-mask quality gate; "
                "%d layers in set)", _ENV, len(drop))


# ==================================================================================================
# Card-free self-test:  python -m hpu_moe_ext.expert_prune_patch --selftest
# ==================================================================================================
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok:   " if cond else "  FAIL: ") + msg)
        if not cond:
            ok = False

    # parse
    d = parse_drop_set({"drop": {"5": [0, 1, 2], "6": [10, 20]}})
    check(d == {5: {0, 1, 2}, 6: {10, 20}}, f"parse drop map (got {d})")
    d2 = parse_drop_set({"drop": {"5": [3], "7": []}})    # empty layer list pruned out
    check(d2 == {5: {3}}, "empty per-layer list dropped from map")
    for bad in ({}, {"drop": {}}, {"drop": {"5": [-1]}}, {"drop": {"x": [1]}}, {"drop": {"5": 3}}):
        try:
            parse_drop_set(bad); check(False, f"bad set {bad} should raise")
        except ValueError:
            check(True, f"bad set {bad} raises (LOUD)")

    # validate against GLM-5.2 config (78 layers, dense band 3, 256 experts)
    validate_against_config({5: {0, 255}}, 78, 3, 256)   # ok
    check(True, "valid set (layer 5, ids 0..255) passes")
    for bad, why in (({2: {0}}, "dense band layer 2"),
                     ({79: {0}}, "layer 79 > 78"),
                     ({5: {256}}, "id 256 >= n_routed 256"),
                     ({5: set(range(256))}, "prune all experts")):
        try:
            validate_against_config(bad, 78, 3, 256); check(False, f"{why} should raise")
        except RuntimeError:
            check(True, f"{why} raises (LOUD)")
    # MTP layer 78 is allowed (== num_hidden_layers)
    validate_against_config({78: {0, 1}}, 78, 3, 256)
    check(True, "MTP layer 78 prunable (== num_hidden_layers)")

    # prefix parse
    check(_layer_idx_from_prefix("model.layers.5.mlp") == 5, "prefix -> layer 5")
    check(_layer_idx_from_prefix("model.layers.78.mlp") == 78, "prefix -> layer 78")
    check(_layer_idx_from_prefix("model.layers.5.self_attn") is None, "non-mlp prefix -> None")
    check(_layer_idx_from_prefix("model.embed_tokens") is None, "embed prefix -> None")

    # byte-exact: env unset -> install() returns without importing torch/vllm (no raise)
    saved = os.environ.pop(_ENV, None)
    try:
        install()  # should be a clean no-op
        check(True, "install() is a no-op when HPU_PRUNE_EXPERTS unset (byte-exact)")
    except Exception as ex:  # noqa: BLE001
        check(False, f"install() no-op raised {ex!r}")
    finally:
        if saved is not None:
            os.environ[_ENV] = saved

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
