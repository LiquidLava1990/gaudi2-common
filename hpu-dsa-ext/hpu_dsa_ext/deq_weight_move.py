"""Post-load HPU move for GPTQDequantToBF16's deferred (CPU) bf16 weights (lazy/:next path).

Under lazy mode the 8-bit dequant (hpu-moe-ext GPTQDequantToBF16LinearMethod) keeps its bf16
weight on CPU during process_weights and marks the layer `_hpu_deq_pending_move=True`, so the
bf16 does NOT accumulate on HPU and OOM the next layer's unpack .cpu() staging (PT_DEVMEM 6MB
fail despite free HBM — the inflation peak). This patch moves all such weights onto HPU AFTER
HPUModelRunner.load_model completes: by then the quantized source weights are freed, so the
quant copy and the bf16 copy never coexist on HPU, and the move has room.

Wired from hpu_dsa_ext.register() so it installs at startup (before load_model runs).
"""
from __future__ import annotations

import logging

log = logging.getLogger("vllm")


def _move(model) -> None:
    import torch
    import habana_frameworks.torch.core as htorch
    n = 0
    for mod in model.modules():
        if getattr(mod, "_hpu_deq_pending_move", False):
            dw = getattr(mod, "_deferred_weight", None)   # CPU bf16 stashed as a plain attr
            if dw is not None:
                # Move one weight at a time + flush, so each ~MB allocation reuses freed space
                # and the pool stays defragmented (the runner's all-at-once bulk move OOMs here).
                mod.weight = torch.nn.Parameter(dw.to("hpu"), requires_grad=False)
                try:
                    del mod._deferred_weight
                except Exception:  # noqa: BLE001
                    pass
                n += 1
                htorch.mark_step()
            mod._hpu_deq_pending_move = False
    htorch.mark_step()
    if n:
        log.warning("[deq-move] moved %d deferred bf16 weights to HPU post-load", n)


def _patch_mla_marksteps() -> None:
    """Add per-layer mark_step around the MLA weight-absorption (W_UK_T/W_UV from kv_b_proj),
    which under lazy churns HBM with no flush and fragments the pool -> PT_DEVMEM 12MB fail
    during load (despite ~42GB free). Flushing per MLA layer reclaims each layer's absorption
    intermediates before the next. No-op in eager."""
    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        import habana_frameworks.torch.core as htorch
    except Exception as ex:  # noqa: BLE001
        log.warning("[deq-move] MLA markstep patch: import failed: %s", ex)
        return
    if getattr(MLAAttention.process_weights_after_loading, "_mla_markstep", False):
        return
    _orig = MLAAttention.process_weights_after_loading
    try:
        from hpu_moe_ext import moe_wna16_hpu as _mw   # to toggle its absorption-flush flag
    except Exception:  # noqa: BLE001
        _mw = None

    def process_weights_after_loading(self, *a, **k):  # noqa: ANN001
        htorch.mark_step()
        if _mw is not None:
            _mw._DEQ_ABSORB[0] = True   # make GPTQDequantToBF16.apply flush around extraction
        try:
            r = _orig(self, *a, **k)
        finally:
            if _mw is not None:
                _mw._DEQ_ABSORB[0] = False
        htorch.mark_step()
        return r

    process_weights_after_loading._mla_markstep = True
    MLAAttention.process_weights_after_loading = process_weights_after_loading
    log.warning("[deq-move] installed per-layer mark_step around MLA absorption")


def _patch_get_and_maybe_dequant() -> None:
    """Short-circuit get_and_maybe_dequant_weights for GPTQDequantToBF16 layers. The MLA absorption
    extracts each projection's weight via the generic O(N^3) identity-matrix trick
    (eye=torch.eye(in); quant_method.apply(eye).T) — which under lazy both OOMs (the big eye +
    intermediates fragment the pool) and hits a strided-.T lazy error. But our layer.weight is
    ALREADY the dequanted bf16 [out,in], which is exactly what that trick returns — so just return
    it. Patches mla_attention's bound reference (it imports the name)."""
    try:
        from vllm.model_executor.layers.attention import mla_attention
    except Exception as ex:  # noqa: BLE001
        log.warning("[deq-move] get_and_maybe_dequant patch: import failed: %s", ex)
        return
    _orig = getattr(mla_attention, "get_and_maybe_dequant_weights", None)
    if _orig is None or getattr(_orig, "_deq_short", False):
        return

    def get_and_maybe_dequant_weights(layer, *a, **k):  # noqa: ANN001
        qm = getattr(layer, "quant_method", None)
        w = getattr(layer, "weight", None)
        if (qm is not None and type(qm).__name__ == "GPTQDequantToBF16LinearMethod"
                and w is not None and not hasattr(layer, "weight_scale")):
            out_dtype = a[0] if a else k.get("out_dtype", w.dtype)
            return w.to(out_dtype)            # already [out, in] dequanted bf16
        return _orig(layer, *a, **k)

    get_and_maybe_dequant_weights._deq_short = True
    mla_attention.get_and_maybe_dequant_weights = get_and_maybe_dequant_weights
    log.warning("[deq-move] short-circuited get_and_maybe_dequant_weights for GPTQDequantToBF16")


def install() -> None:
    _patch_mla_marksteps()
    _patch_get_and_maybe_dequant()
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except Exception as ex:  # noqa: BLE001
        log.warning("[deq-move] cannot import HPUModelRunner: %s", ex)
        return
    if getattr(HPUModelRunner.load_model, "_deq_move", False):
        return
    _orig = HPUModelRunner.load_model

    def load_model(self, *a, **k):  # noqa: ANN001
        r = _orig(self, *a, **k)
        try:
            m = getattr(self, "model", None)
            if m is None and hasattr(self, "get_model"):
                m = self.get_model()
            if m is not None:
                _move(m)
        except Exception as ex:  # noqa: BLE001
            log.warning("[deq-move] post-load move failed: %s", ex)
        # RECLAIM the freed MoE-repack originals (the uint8 weights replaced by the int32 repack):
        # under lazy on :next the device pool keeps them at the load high-water mark (~+42GB), so
        # vLLM measures the model at ~94GB and leaves no room for KV cache. Flush + release the
        # cached pool memory so the freed blocks return and the KV sizing sees the real ~52GB.
        # NOTE: no pool reclaim here — the in-place MoE repack (moe_wna16_hpu._repack_weight) keeps
        # the device pool at the real ~52GB, so there is nothing trapped to release. (A post-load
        # hpu.synchronize() to unpin freed chunks OOMs anyway once HBM is full, and Habana has no
        # empty_cache.) Left as a no-op hook in case future deferred weights need moving.
        return r

    load_model._deq_move = True
    HPUModelRunner.load_model = load_model
    log.warning("[deq-move] installed post-load HPU move for deferred bf16 weights")
