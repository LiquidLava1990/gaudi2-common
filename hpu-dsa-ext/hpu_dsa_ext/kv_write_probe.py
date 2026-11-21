"""HOST-SIDE KV-cache write probe — decide WRITE-side vs READ-side for the GLM lazy
decode drift, with ZERO in-graph ops (the instrument the in-graph ring-buffer probe
could not be: its index_copy_ slot index was baked at capture, so replays always wrote
the capture slot and the drain read zeros — the probe was eaten by the bug class it
hunts).

MECHANISM: wrap HPUModelRunner._execute_model_generic. AFTER the original returns (the
step's captured graph has fully executed; the .item() below is a natural post-step sync,
never inside a capture), checksum the layer-0 KV cache row(s) at the slot(s) the HOST
believes this step wrote (attn_metadata.slot_mapping). Also re-checksum the FIRST
recorded slot on every later step.

VERDICT (grep "[kv-write-probe]"):
  * cur_sum ~ 0.0 while pos advances        -> the write MISSED its slot on replay
                                               (slot index frozen) => WRITE-side bug.
  * first_slot_sum CHANGES on later steps   -> later writes are landing ON the first
                                               step's slot => WRITE-side bug (frozen
                                               write index, the specific form).
  * cur_sum > 0 every step AND first_slot_sum stable after its own step
                                            -> writes land where the host thinks;
                                               the bug is READ-side (block_list /
                                               block_usage / attention) or elsewhere.

Gated: HPU_KV_WRITE_PROBE=<N> decode steps to log (default 0 = OFF = fully inert).
Diagnosis only — the per-step .item() host syncs serialize the pipeline; never leave
this on in a production serve.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("vllm")

_PROBE_N = int(os.environ.get("HPU_KV_WRITE_PROBE", "0") or "0")


def install() -> None:
    if _PROBE_N <= 0:
        return
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except Exception as ex:  # noqa: BLE001
        log.warning("[kv-write-probe] cannot import HPUModelRunner; NOT installed: %s", ex)
        return
    if getattr(HPUModelRunner._execute_model_generic, "_kv_write_probe", False):
        return

    _orig = HPUModelRunner._execute_model_generic
    state = {"n": 0, "first_slot": None, "first_sum_at_write": None}

    def _layer0_latent_cache(self):
        kc = self.kv_caches[0] if getattr(self, "kv_caches", None) else None
        if kc is None:
            return None
        if isinstance(kc, (tuple, list)):
            return kc[0]
        return kc

    def _execute_model_generic(self, token_ids, position_ids, attn_metadata,
                               logits_indices, kv_caches, lora_logits_mask,
                               lora_mask, warmup_mode=False, **kw):
        out = _orig(self, token_ids, position_ids, attn_metadata, logits_indices,
                    kv_caches, lora_logits_mask, lora_mask, warmup_mode, **kw)
        try:
            if (not warmup_mode and state["n"] < _PROBE_N
                    and not getattr(attn_metadata, "is_prompt", True)
                    and getattr(attn_metadata, "slot_mapping", None) is not None):
                import torch  # noqa: F401
                cache = _layer0_latent_cache(self)
                if cache is not None and cache.dim() >= 2:
                    n = state["n"]
                    state["n"] = n + 1
                    slots_dev = attn_metadata.slot_mapping.flatten().to(torch.int64)
                    # post-step host syncs (OUTSIDE any captured graph)
                    slots = slots_dev.cpu().tolist()
                    cur = float(cache.index_select(0, slots_dev)
                                .float().abs().sum().item())
                    if state["first_slot"] is None:
                        state["first_slot"] = int(slots[0])
                    fs = state["first_slot"]
                    fsum = float(cache[fs].float().abs().sum().item())
                    if state["first_sum_at_write"] is None:
                        state["first_sum_at_write"] = fsum
                    pos = int(position_ids.flatten()[0].item()) \
                        if position_ids is not None else -1
                    log.warning(
                        "[kv-write-probe] step=%d pos=%d slots=%s cur_sum=%.4f "
                        "first_slot=%d first_slot_sum=%.4f (at_write=%.4f)",
                        n, pos, slots[:4], cur, fs, fsum,
                        state["first_sum_at_write"])
        except Exception as ex:  # noqa: BLE001
            log.warning("[kv-write-probe] probe error (step skipped): %s", ex)
        return out

    import torch  # noqa: F401  (bound for the closure above)
    _execute_model_generic._kv_write_probe = True
    HPUModelRunner._execute_model_generic = _execute_model_generic
    log.warning("[kv-write-probe] installed host-side post-step KV write probe "
                "(N=%d decode steps; layer-0 latent cache). HPU_KV_WRITE_PROBE", _PROBE_N)
