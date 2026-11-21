"""A.2 RESIDUAL ADAPTER serve wiring — ADD a trained adapter's logits to the NATIVE GLM-5.2 MTP head's
draft logits, in parallel, gated. The adapter LIFTS per-depth accept above the measured native floor
(per-position [0.44,0.13,0.04] -> accept-len ~1.50). Card-free-trained on the per-step target_argmax
(head_factory/factory/residual_adapter.py + ttt_train.py --adapter).

GATE: HPU_MTP_ADAPTER=<path to head_native_adapter.pt>   (OFF/unset -> byte-IDENTICAL to native).
      HPU_MTP_ADAPTER_ALPHA=<float, default 1.0>          (the serve residual gate; 0.0 == native exactly,
                                                           so even ON it can be made a no-op for an A/B).
      HPU_MTP_ADAPTER_CFG=<path to config.train.d1.host.json>  (the head geometry; default = sibling of the
                                                           ckpt or the d1 config under /opt/hf/glm52).

WHERE IT TAPS (verified against vllm-gaudi-patch/v1/spec_decode/hpu_eagle.py HpuEagleProposer.propose):
  propose() drafts K tokens. At EVERY draft depth it does `logits = self.model.compute_logits(h)` then
  `argmax` (depth-1 L66/86, loop L160/161). We:
    (a) wrap `propose` to, at entry, set up the adapter's self-feed state from the SAME seed the native
        head starts from -- seed_hidden = target_hidden_states[last_token_indices] (== the dumped `hidden`),
        seed_token = target_token_ids[last_token_indices], positions = target_positions[last_token_indices];
    (b) for the DURATION of that propose call, monkeypatch self.model.compute_logits so it returns
        native_logits + alpha*adapter_residual. The caller's argmax then picks from the SUM (== the served
        draft). The adapter advances its OWN state each call: next hidden = the adapter body's output this
        depth; next token = the SUMMED argmax (the committed draft, == what the native loop feeds back too);
        next pos = +1. This mirrors the offline per-step unroll (ttt_train.verify_unroll_loss) EXACTLY, so
        the trained adapter behaves at serve as it was trained.
  Restored after propose returns (no global mutation persists). NEVER raises into the serve (on any error it
  falls back to native compute_logits for the rest of that call).

IDENTITY / never-regress: alpha INIT/knob 0.0 -> residual == 0 -> served draft == native byte-exact. The
  adapter ships ONLY if the on-card A/B (HPU_MTP_ADAPTER on vs HPU_MTP_ADAPTER_ALPHA=0) shows accept-len
  beating measured native; worst case alpha=0 == native (the floor is mechanically guaranteed).

fp8: the adapter's trainable linears are fp8-rounded at load (eval_head._fp8_round_head proves fp8==fp32
  accept) -> no bf16 in the serve process, consistent with the fp8 MTP graft.
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("vllm")
_ENV = "HPU_MTP_ADAPTER"
_STATE = {"adapter": None, "alpha": 1.0, "loaded": False}


def _fp8_round(w):
    import torch
    amax = w.detach().abs().amax(0, keepdim=True).clamp_min(1e-8)
    s = amax / 448.0
    return ((w / s).to(torch.float8_e4m3fn).to(w.dtype)) * s


def _load_adapter():
    """Build the ResidualAdapter from head_factory and load the trained ckpt (frozen embed/lm_head from the
    target). fp8-round the trainable linears. LOUD on any failure (ON-without-weights must be loud)."""
    import json
    import torch
    import torch.nn as nn
    ckpt_path = os.environ.get(_ENV, "").strip()
    if not ckpt_path:
        return None
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}={ckpt_path} is not a file (adapter ckpt missing).")
    # head_factory must be importable (mounted at /opt/hf in the serve image; runbook mounts it).
    for cand in ("/opt/hf/factory", os.path.join(os.path.dirname(ckpt_path) or ".", "..", "factory")):
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from residual_adapter import ResidualAdapter
        from head import HeadConfig
        from train import _load_frozen_embed_lmhead
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}: head_factory not importable ({ex!r}). Mount it at "
                           "/opt/hf (the runbook does -v $HF:/opt/hf).")
    cfg_path = os.environ.get("HPU_MTP_ADAPTER_CFG", "").strip()
    if not cfg_path:
        for c in (os.path.join(os.path.dirname(ckpt_path), "config.train.d1.host.json"),
                  "/opt/hf/glm52/config.train.d1.host.json"):
            if os.path.isfile(c):
                cfg_path = c
                break
    if not cfg_path or not os.path.isfile(cfg_path):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}: HPU_MTP_ADAPTER_CFG not set / not found (the head "
                           "geometry config). Point it at config.train.d1.host.json.")
    cfg_d = json.load(open(cfg_path))
    cfg = HeadConfig(**{k: v for k, v in cfg_d.items() if k in HeadConfig.__dataclass_fields__})
    embed, lm = _load_frozen_embed_lmhead(cfg_d)
    adapter = ResidualAdapter(cfg, embed, lm)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["state"] if "state" in ck else ck
    miss, unexp = adapter.load_state_dict(st, strict=False)
    body_missing = [m for m in miss if m.startswith("body.") and "embed_w" not in m and "lm_head_w" not in m]
    if body_missing:
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}: adapter ckpt is MISSING body params {body_missing[:6]} "
                           f"for the geometry in {cfg_path} (config/ckpt mismatch). Refusing a partial "
                           "adapter (would corrupt the residual).")
    # fp8-round the trainable linears (served precision; norms stay fp16). Frozen embed/lm_head untouched.
    n = 0
    with torch.no_grad():
        for _name, m in adapter.named_modules():
            if isinstance(m, nn.Linear) and m.weight is not None:
                m.weight.copy_(_fp8_round(m.weight)); n += 1
    adapter.eval()
    # device: match the proposer's model device at install time (set in install()).
    logger.warning("[hpu-moe-ext] MTP RESIDUAL ADAPTER loaded from %s (cfg %s; %d linears fp8-rounded; "
                   "trained alpha=%.3f).", ckpt_path, cfg_path, n,
                   float(adapter.alpha.detach()) if hasattr(adapter, "alpha") else 1.0)
    return adapter


def install() -> None:
    """Wrap HpuEagleProposer.propose to inject the adapter residual (gated). OFF -> stock (no wrap)."""
    if not os.environ.get(_ENV, "").strip():
        return
    _STATE["alpha"] = float(os.environ.get("HPU_MTP_ADAPTER_ALPHA", "1.0"))
    try:
        from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer
    except Exception as ex:  # noqa: BLE001
        logger.warning("[hpu-moe-ext] %s set but HpuEagleProposer not importable (%r) -- adapter OFF.",
                       _ENV, ex)
        return
    if getattr(HpuEagleProposer.propose, "_mtp_adapter", False):
        return
    import torch
    _oprop = HpuEagleProposer.propose

    def _lazy_adapter(self):
        if not _STATE["loaded"]:
            _STATE["loaded"] = True
            try:
                ad = _load_adapter()
                if ad is not None:
                    dev = next(self.model.parameters()).device
                    ad = ad.to(dev)
                _STATE["adapter"] = ad
            except Exception:
                logger.exception("[hpu-moe-ext] %s: adapter load FAILED -- serving NATIVE (no residual).",
                                 _ENV)
                _STATE["adapter"] = None
        return _STATE["adapter"]

    def _propose(self, target_token_ids, target_positions, target_hidden_states, last_token_indices,
                 common_attn_metadata, block_table_cpu_tensor, model_runner, *a, **k):
        adapter = _lazy_adapter(self)
        if adapter is None or _STATE["alpha"] == 0.0 or getattr(self, "method", None) != "mtp":
            return _oprop(self, target_token_ids, target_positions, target_hidden_states,
                          last_token_indices, common_attn_metadata, block_table_cpu_tensor,
                          model_runner, *a, **k)
        # --- set up the adapter self-feed seed from the SAME inputs the native head drafts from ---
        try:
            th = target_hidden_states.view(-1, target_hidden_states.shape[-1])[last_token_indices]  # [B,H]
            tt = target_token_ids.view(-1)[last_token_indices]                                       # [B]
            tp = target_positions.view(-1)[last_token_indices]                                       # [B]
            st = {"hidden": th, "token": tt, "pos": tp.long(), "depth": 0}
        except Exception:
            logger.exception("[hpu-moe-ext] %s: adapter seed setup failed -- NATIVE for this step.", _ENV)
            return _oprop(self, target_token_ids, target_positions, target_hidden_states,
                          last_token_indices, common_attn_metadata, block_table_cpu_tensor,
                          model_runner, *a, **k)
        alpha = _STATE["alpha"]
        _ocl = self.model.compute_logits

        def _cl(hidden, *ca, **ck):
            native = _ocl(hidden, *ca, **ck)
            try:
                with torch.no_grad():
                    ah = adapter.forward_hidden(st["hidden"], st["token"], st["pos"])   # [B,H] adapter hidden
                    ares = adapter.compute_logits(ah)                                   # alpha_trained*lm(ah)
                    # serve gate alpha (env) scales the trained residual; alpha=0 -> native exactly.
                    summed = native + (alpha * ares if alpha != 1.0 else ares)
                    # advance adapter state for the NEXT depth: own hidden + the COMMITTED draft (sum argmax)
                    st["hidden"] = ah
                    st["token"] = summed.argmax(dim=-1)
                    st["pos"] = st["pos"] + 1
                    st["depth"] += 1
                    return summed
            except Exception:
                logger.exception("[hpu-moe-ext] %s: adapter residual failed at depth %d -- NATIVE rest of "
                                 "this step.", _ENV, st["depth"])
                return native

        self.model.compute_logits = _cl
        try:
            return _oprop(self, target_token_ids, target_positions, target_hidden_states,
                          last_token_indices, common_attn_metadata, block_table_cpu_tensor,
                          model_runner, *a, **k)
        finally:
            self.model.compute_logits = _ocl   # restore (no persistent global mutation)

    _propose._mtp_adapter = True
    HpuEagleProposer.propose = _propose
    logger.warning("[hpu-moe-ext] MTP RESIDUAL ADAPTER armed (%s; serve alpha=%.3f). Drafts = argmax(native "
                   "+ alpha*adapter). alpha=0 == native byte-exact.", _ENV, _STATE["alpha"])
