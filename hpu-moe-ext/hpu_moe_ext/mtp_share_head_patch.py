"""Tie the MTP draft's embed_tokens + shared_head.head to the TARGET model when
the checkpoint does not ship its own copies (GLM-5.1 style).

THE BUG (cost: spec-decode acceptance EXACTLY 0%): DeepSeekMTP.load_weights
skips every weight outside the spec layer (deepseek_mtp.py:277-278
`get_spec_layer_idx_from_weight_name(...) is None -> continue`), so
`lm_head.weight` and `model.embed_tokens.weight` never reach the draft.
DeepSeek-V3 checkpoints ship their OWN copies INSIDE the MTP layer
(`model.layers.61.embed_tokens.weight`, `...shared_head.head.weight`) so
upstream never notices — but GLM-5.1's layer 78 ships ONLY `shared_head.norm`
(the design shares the target's embedding + head). Result on GLM: the draft
ran with a RANDOM-INIT embedding table and a RANDOM-INIT output head — its
argmax was the constant junk token 70684 ('odesk') regardless of input
(probe-verified 2026-06-10), hence 0% acceptance.

Upstream's _maybe_share_embeddings/_maybe_share_lm_head only handle
EAGLE-style drafts (`has_own_lm_head` attr / a top-level `lm_head` module);
DeepSeekMTP keeps its head inside the per-layer SharedHead, so both no-op.

THE FIX: wrap EagleProposer.load_model; after the draft loads, decide from
the CHECKPOINT INDEX (general, no model-name special-casing):
  * `model.layers.{idx}.embed_tokens.*` absent  -> draft.model.embed_tokens
       = target.model.embed_tokens (module tie; identical TP sharding)
  * `model.layers.{idx}.shared_head.head.*` absent -> every MTP layer's
       shared_head.head = target.lm_head (module tie)
Checkpoints that ship their own copies (DeepSeek-V3) are left untouched.
LOUD: logs exactly what was tied and why; raises if the target lacks the
modules we must tie to (never leave a silently-random draft head again).
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("vllm")


def _ckpt_has_prefix(model_path: str, prefix: str) -> bool:
    """True if any checkpoint weight name starts with prefix (safetensors index,
    falling back to a single-file model.safetensors header scan)."""
    idx_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_file):
        with open(idx_file) as f:
            wm = json.load(f)["weight_map"]
        return any(k.startswith(prefix) for k in wm)
    # Single-file fallback: read the safetensors header only.
    st = os.path.join(model_path, "model.safetensors")
    if os.path.exists(st):
        import struct
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return any(k.startswith(prefix) for k in hdr)
    return False


def install() -> None:
    try:
        from vllm.v1.spec_decode.eagle import EagleProposer
    except Exception:  # noqa: BLE001 — no spec-decode support in this build
        return
    if getattr(EagleProposer.load_model, "_hpu_mtp_share_patch", False):
        return
    _orig = EagleProposer.load_model

    def _patched(self, target_model, *args, **kwargs):
        out = _orig(self, target_model, *args, **kwargs)
        try:
            _maybe_tie_mtp_shared(self, target_model)
        except Exception:
            logger.exception(
                "[hpu-moe-ext] MTP shared-weight tie FAILED — the draft may be "
                "running with random embed/head (acceptance will be ~0%).")
            raise
        return out

    _patched._hpu_mtp_share_patch = True
    EagleProposer.load_model = _patched
    logger.info("hpu-moe-ext: patched EagleProposer.load_model "
                "(tie MTP embed/shared_head.head to target when ckpt omits them)")


def _maybe_tie_mtp_shared(proposer, target_model) -> None:
    draft = getattr(proposer, "model", None)
    inner = getattr(draft, "model", None)
    layers = getattr(inner, "layers", None)
    # Only DeepSeekMTP-shaped drafts (an inner predictor with SharedHead layers).
    if layers is None or not hasattr(inner, "embed_tokens"):
        return
    mtp_layers = [l for l in layers.values() if hasattr(l, "shared_head")] \
        if hasattr(layers, "values") else []
    if not mtp_layers:
        return

    cfg = getattr(proposer, "vllm_config", None)
    model_path = cfg.model_config.model if cfg is not None else None
    if model_path is None:
        return
    spec_idx = int(getattr(draft.config, "num_hidden_layers", 0))

    tgt_inner = getattr(target_model, "model", target_model)

    if not _ckpt_has_prefix(model_path, f"model.layers.{spec_idx}.embed_tokens."):
        if not hasattr(tgt_inner, "embed_tokens"):
            raise RuntimeError(
                "[hpu-moe-ext] MTP ckpt ships no draft embed_tokens and the "
                "target has no embed_tokens to share — draft would be random.")
        inner.embed_tokens = tgt_inner.embed_tokens
        logger.warning(
            "[hpu-moe-ext] MTP TIE: checkpoint ships no "
            "model.layers.%d.embed_tokens — sharing the TARGET's embed_tokens "
            "with the draft.", spec_idx)

    if not _ckpt_has_prefix(model_path, f"model.layers.{spec_idx}.shared_head.head."):
        tgt_head = getattr(target_model, "lm_head", None)
        if tgt_head is None:
            raise RuntimeError(
                "[hpu-moe-ext] MTP ckpt ships no shared_head.head and the "
                "target has no lm_head to share — draft head would be random.")
        for l in mtp_layers:
            l.shared_head.head = tgt_head
        logger.warning(
            "[hpu-moe-ext] MTP TIE: checkpoint ships no "
            "model.layers.%d.shared_head.head — sharing the TARGET's lm_head "
            "with %d MTP layer(s).", spec_idx, len(mtp_layers))
