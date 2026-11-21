"""HPU NaN-sanitize for the token-embedding output.

Diagnosis (2026-06-06, real-request layer probe): GLM-5.1 generation was
deterministic garbage. The per-layer probe showed the model's *embedding
output* already carries a NaN at step 1 (`EMBED embed_out norm=nan nan=1`),
before ANY decoder layer runs. The embed_tokens WEIGHT is clean (verified
nan=0, inf=0), so the NaN is produced in the embedding *forward* on HPU
(the TP=8 vocab-parallel masked-gather + all-reduce path) — a single spurious
NaN element. That one NaN poisons the residual stream; a downstream NaN-guard
then freezes activations to a constant (every layer showed the identical
norm=17.1114), so the model emits token-salad.

Fix: nan_to_num the embedding output (and, defensively, the final norm input is
left to the model). A NaN in a real token's embedding -> 0 is far better than a
NaN that poisons every subsequent layer; a NaN in a padding slot -> 0 is exactly
correct. Plugin-side, no /opt edits. Gated by HPU_SANITIZE_EMBED=1.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("HPU_SANITIZE_EMBED", "0") != "1":
        return
    import torch
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
    except Exception as ex:  # noqa: BLE001
        log.warning("[nan-sanitize] cannot import DeepseekV2Model: %s", ex)
        return
    if not hasattr(DeepseekV2Model, "embed_input_ids"):
        log.warning("[nan-sanitize] DeepseekV2Model.embed_input_ids absent; skip")
        return
    if getattr(DeepseekV2Model.embed_input_ids, "_nan_sanitized", False):
        return
    _orig = DeepseekV2Model.embed_input_ids

    def embed_input_ids(self, *args, **kwargs):  # noqa: ANN001
        out = _orig(self, *args, **kwargs)
        try:
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:  # noqa: BLE001
            return out

    embed_input_ids._nan_sanitized = True
    DeepseekV2Model.embed_input_ids = embed_input_ids
    _INSTALLED = True
    log.warning("[nan-sanitize] DeepseekV2Model.embed_input_ids NaN-sanitize installed "
                "(nan_to_num on embedding output; HPU_SANITIZE_EMBED=1)")
