"""Test the lazy-capture buffer-reuse drift hypothesis for GLM-5.1.

GLM lazy decode produces correct-then-degrading output (a repetition cycle) with
positions confirmed LIVE and the rope fixed — i.e. a subtle per-step NUMERICAL
drift inside the captured graph, not a frozen tensor. The HPU-graph capture is
created with `wrap_in_hpu_graph(..., disable_tensor_cache=True)`, which lets the
capture REUSE/ALIAS intermediate tensor storage across the graph to save HBM.
Aliasing an intermediate that should be fresh each replay is a textbook cause of
exactly this kind of accumulating per-step drift.

This module forces `disable_tensor_cache=False` (cache ON -> no cross-step buffer
aliasing) so we can A/B whether the drift is the buffer reuse. Costs more HBM
(intermediates retained), so it may need a smaller max-model-len. Gated by
HPU_GLM_NO_TENSOR_CACHE=1; a no-op otherwise.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")


def install() -> None:
    if os.environ.get("HPU_GLM_NO_TENSOR_CACHE", "0") != "1":
        return
    try:
        import habana_frameworks.torch.hpu as _h
    except Exception as e:  # noqa: BLE001
        log.warning("[glm-notc] cannot import habana_frameworks.torch.hpu: %s", e)
        return
    if getattr(getattr(_h, "wrap_in_hpu_graph", None), "_glm_notc", False):
        return
    _orig = _h.wrap_in_hpu_graph

    def _wrap(module, *a, **k):  # noqa: ANN001
        # Force the cache ON regardless of the caller's request.
        k["disable_tensor_cache"] = False
        return _orig(module, *a, **k)

    _wrap._glm_notc = True
    _h.wrap_in_hpu_graph = _wrap
    # Patch the name already bound into the model runner (it does a bare import).
    try:
        import vllm_gaudi.v1.worker.hpu_model_runner as _hr
        if hasattr(_hr, "wrap_in_hpu_graph"):
            _hr.wrap_in_hpu_graph = _wrap
    except Exception:
        pass
    log.warning("[glm-notc] forced wrap_in_hpu_graph(disable_tensor_cache=False) "
                "-- A/B test the lazy-capture buffer-reuse numerical drift")
