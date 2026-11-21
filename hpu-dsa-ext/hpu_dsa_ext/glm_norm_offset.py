"""GLM-5.1 (1+weight) RMSNorm fix.

ROOT CAUSE (2026-06-07): GLM-5.1 (`GlmMoeDsaForCausalLM`) is mapped by vLLM's registry
to `("deepseek_v2", "GlmMoeDsaForCausalLM")` — a bare `class GlmMoeDsaForCausalLM(
DeepseekV2ForCausalLM): pass`. So it runs on the DeepSeek-V2 path, whose `RMSNorm`
applies PLAIN `weight`. But GLM-5.1's RMSNorm weights are stored as (1 + weight) OFFSETS,
clustered near 0 (kv_a_layernorm rms≈0.006, q_a≈0.17, input_layernorm≈0.04) rather than
near 1. Applied as plain `weight` they scale activations by ~0.006-0.17 instead of
~1.006-1.17 — ~26x too weak everywhere, and ~167x too weak on the kv_a latent, which
collapses K-content + V to ~zero ⇒ DEAD attention ⇒ deterministic token salad
("andiandi…", "iecieciec"). Confirmed via mha_diag: q_norm≈100 but attn out_norm≈0.1,
V rms≈0.0002, k_c_normed rms≈0.0009.

(A working GLM under standard RMSNorm would have norm weights ~1; transformers'
Glm4MoeRMSNorm/GlmRMSNorm init to torch.ones. Near-0 weights ⇒ (1+weight) parameterization.)

FIX: after the model loads, add 1.0 to every RMSNorm weight so the DeepSeek path's plain
`weight * x_normed` becomes the intended `(1 + weight) * x_normed`. Idempotent; gated by
HPU_GLM_NORM_OFFSET=1. Greedy-safe even for the final norm (a uniform logit scale does not
change argmax).
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")


def _apply_offset_after_load(model, *, final_norm_suffixes):
    """Add 1.0 to every RMSNorm weight on `model` except the final-norm modules
    (whose checkpoint weights already cluster near 1). Idempotent per-module.

    `final_norm_suffixes`: tuple of module-name SUFFIXES to EXCLUDE (the
    already-applied final norm). For the main model that's "model.norm"/"norm";
    for the DeepSeek-MTP draft it is the per-MTP-layer "shared_head.norm" (verified
    rms≈1.30, mean≈1.29 — same near-1 family as model.norm, NOT a (-1) offset)."""
    import torch
    from vllm.model_executor.layers.layernorm import RMSNorm

    n = 0
    sample_before = sample_after = None
    with torch.no_grad():
        for name, m in model.named_modules():
            if name == "norm" or any(
                name == s or name.endswith("." + s) for s in final_norm_suffixes
            ):
                continue
            if isinstance(m, RMSNorm) and not getattr(m, "_glm_offset_done", False):
                w = getattr(m, "weight", None)
                if w is None:
                    continue
                if n == 0:
                    sample_before = float(w.data.float().pow(2).mean().sqrt())
                w.data.add_(1.0)
                if n == 0:
                    sample_after = float(w.data.float().pow(2).mean().sqrt())
                m._glm_offset_done = True
                n += 1
    return n, sample_before, sample_after


def _wrap_load_weights(cls, label, final_norm_suffixes):
    """Wrap cls.load_weights so the (1+weight) RMSNorm restore runs after load."""
    if cls is None or not hasattr(cls, "load_weights"):
        log.warning("[glm-norm-offset] %s.load_weights absent; skip", label)
        return
    if getattr(cls.load_weights, "_glm_norm_offset", False):
        return
    _orig = cls.load_weights

    def load_weights(self, *args, **kwargs):  # noqa: ANN001
        ret = _orig(self, *args, **kwargs)
        n, sb, sa = _apply_offset_after_load(
            self, final_norm_suffixes=final_norm_suffixes
        )
        log.warning(
            "[glm-norm-offset] (%s) applied (1+weight) to %d RMSNorm modules "
            "(first norm rms %.4f -> %.4f)", label, n, sb or -1, sa or -1)
        return ret

    load_weights._glm_norm_offset = True
    cls.load_weights = load_weights
    log.warning("[glm-norm-offset] installed on %s.load_weights "
                "(HPU_GLM_NORM_OFFSET=1)", label)


def install() -> None:
    # The MAIN-model (1+weight) offset is gated on HPU_GLM_NORM_OFFSET (kept OFF in the
    # proven-coherent config — the 4 real fixes don't include it). The MTP-DRAFT offset is
    # gated SEPARATELY on HPU_MTP_NORM_OFFSET so the draft can be experimented with WITHOUT
    # perturbing the (coherent) main model's norms.
    _main = os.environ.get("HPU_GLM_NORM_OFFSET", "0") == "1"
    _mtp = os.environ.get("HPU_MTP_NORM_OFFSET", "0") == "1"
    if not (_main or _mtp):
        return
    try:
        from vllm.model_executor.models import deepseek_v2 as dsv2  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-norm-offset] import failed: %s", ex)
        return

    if _main:
        # MAIN model: exclude the final model.norm (clusters near 1).
        _wrap_load_weights(
            getattr(dsv2, "DeepseekV2ForCausalLM", None),
            "DeepseekV2ForCausalLM",
            final_norm_suffixes=("model.norm",),
        )

    if _mtp:
        # DeepSeek-MTP DRAFT (nextn / GLM-5.1 MTP head): the draft is a SEPARATE model
        # instance (DeepSeekMTP), so the main wrapper never touches its norms. If the GLM
        # MTP norms follow the same (1+weight) convention, leaving them ~26x too weak gives
        # a dead draft -> garbage draft tokens -> 0% spec-decode acceptance. Exclude the MTP
        # "final" norm shared_head.norm (rms≈1.30 mean≈1.29 — already-applied family).
        try:
            from vllm.model_executor.models import deepseek_mtp as dsmtp
            _wrap_load_weights(
                getattr(dsmtp, "DeepSeekMTP", None),
                "DeepSeekMTP",
                final_norm_suffixes=("shared_head.norm",),
            )
        except Exception as ex:  # noqa: BLE001
            log.info("[glm-norm-offset] deepseek_mtp not importable (%s); MTP norm "
                     "offset skipped", ex)
