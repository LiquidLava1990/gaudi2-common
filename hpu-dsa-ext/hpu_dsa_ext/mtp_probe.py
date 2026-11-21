"""MTP draft-acceptance probe (gate HPU_MTP_PROBE=1, default OFF = inert).

Wraps HpuEagleProposer.propose to log, for each call, the target's sampled
token ids going IN (the draft's input token) and the draft token ids coming
OUT (argmax of the MTP head), plus tensor norms. Acceptance is exactly 0%,
so the decisive signal is the RELATIONSHIP between draft output and the
target stream:

  * draft ids CONSTANT / vocab-junk        -> dead head (lm_head/shared_head
    loading or quant bug on layer 78)
  * draft id == its own INPUT token        -> the head is an identity-ish
    copy (input threading swallowed the block)
  * draft ids plausible-but-always-wrong   -> input misalignment (off-by-one
    in hidden/token pairing) or a numerically-broken mtp_block forward

Host-side readback AFTER the propose call (post-step, outside any capture).
Logs at WARNING so it lands in the default serve log. One line per call,
first 8 lanes.
"""
from __future__ import annotations

import os

import torch


def install() -> None:
    if os.environ.get("HPU_MTP_PROBE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer
    except Exception:  # noqa: BLE001  (non-spec serve: nothing to probe)
        return

    if getattr(HpuEagleProposer.propose, "_mtp_probe", False):
        return
    _orig = HpuEagleProposer.propose

    import logging
    log = logging.getLogger("vllm")

    def _wrap_compute_logits(model):
        """One-time wrap of the draft's compute_logits: log hidden-in norm and
        logits top-5 (ids + values) — separates 'block outputs constant hidden'
        from 'head maps varied hidden to constant logits'."""
        if getattr(model.compute_logits, "_mtp_probe", False):
            return
        _cl = model.compute_logits

        def _probed_cl(hidden_states, *a, **kw):
            logits = _cl(hidden_states, *a, **kw)
            try:
                hn = hidden_states.float().norm(dim=-1).reshape(-1)[:4]
                top = logits.float().topk(5, dim=-1)
                log.warning("[mtp-probe] compute_logits: hid_norms=%s top5_ids=%s top5_vals=%s",
                            [round(float(x), 2) for x in hn.to("cpu")],
                            top.indices.reshape(-1, 5)[:2].to("cpu").tolist(),
                            [[round(v, 2) for v in row] for row in
                             top.values.reshape(-1, 5)[:2].to("cpu").tolist()])
            except Exception as ex:  # noqa: BLE001
                log.warning("[mtp-probe] compute_logits readback failed: %s", ex)
            return logits

        _probed_cl._mtp_probe = True
        model.compute_logits = _probed_cl

    def _hook_block(model):
        """One-time forward hooks on the MTP layer's submodules: output norms
        pinpoint WHICH stage explodes (eh_proj fusion vs attention vs MoE)."""
        if getattr(model, "_mtp_block_hooked", False):
            return
        model._mtp_block_hooked = True
        try:
            inner = model
            for _ in range(4):          # descend wrapper .model chain to the predictor
                if hasattr(inner, "layers"):
                    break
                inner = inner.model
            layer = next(iter(inner.layers.values()))
            targets = {
                "enorm": layer.enorm, "hnorm": layer.hnorm,
                "eh_proj": layer.eh_proj,
                "attn": layer.mtp_block.self_attn,
                "mlp": layer.mtp_block.mlp,
            }
        except Exception as ex:  # noqa: BLE001
            log.warning("[mtp-probe] block-hook resolve failed: %s", ex)
            return

        def _mk(nm):
            def _h(_m, _i, out):
                try:
                    t = out[0] if isinstance(out, tuple) else out
                    if isinstance(t, torch.Tensor):
                        log.warning("[mtp-probe] %s out_norm=%.2f shape=%s", nm,
                                    float(t.float().norm().item()), tuple(t.shape))
                except Exception:  # noqa: BLE001
                    pass
            return _h

        for nm, mod in targets.items():
            try:
                mod.register_forward_hook(_mk(nm))
            except Exception as ex:  # noqa: BLE001
                log.warning("[mtp-probe] hook %s failed: %s", nm, ex)
        log.warning("[mtp-probe] block hooks installed on %s", list(targets))

    def _manual_frontend(model, token_ids, hidden):
        """Re-run the MTP fusion front-end EAGERLY (module hooks are useless —
        the draft forward is HPU-graph captured and replays skip python). Logs
        embed/enorm/hnorm/eh_proj output norms on the real inputs; an exploded
        eh_proj here convicts the quant-aware eh_proj dequant."""
        try:
            inner = model
            for _ in range(4):
                if hasattr(inner, "layers"):
                    break
                inner = inner.model
            layer = next(iter(inner.layers.values()))
            tid = token_ids.reshape(-1)[:2].to(torch.long)
            emb = inner.embed_tokens(tid)
            h = hidden.reshape(-1, hidden.shape[-1])[:2]
            en = layer.enorm(emb)
            hn = layer.hnorm(h)
            fused = torch.cat([en, hn], dim=-1)
            eh = layer.eh_proj(fused)
            eh_t = eh[0] if isinstance(eh, tuple) else eh
            log.warning(
                "[mtp-probe] FRONTEND emb=%.2f enorm=%.2f hnorm=%.2f eh_proj=%.2f",
                float(emb.float().norm().item()), float(en.float().norm().item()),
                float(hn.float().norm().item()), float(eh_t.float().norm().item()))
            # Stage 2: the block's MoE directly on the (normed) fused input —
            # bypasses attention. All ranks reach this point in lockstep, so the
            # MoE's TP/EP collectives match. Huge output here convicts the
            # live-repack (layer-78-uncached) MoE path; sane output points at
            # the draft's attention/KV instead.
            blk = layer.mtp_block
            x = blk.post_attention_layernorm(eh_t)[0] \
                if isinstance(blk.post_attention_layernorm(eh_t), tuple) \
                else blk.post_attention_layernorm(eh_t)
            moe_out = blk.mlp(x if x.dim() == 2 else x.unsqueeze(0))
            moe_t = moe_out[0] if isinstance(moe_out, tuple) else moe_out
            log.warning("[mtp-probe] MOE-DIRECT in=%.2f out=%.2f",
                        float(x.float().norm().item()),
                        float(moe_t.float().norm().item()))
        except Exception as ex:  # noqa: BLE001
            log.warning("[mtp-probe] manual frontend failed: %r", ex)

    def _probed(self, target_token_ids, target_positions, target_hidden_states,
                last_token_indices, common_attn_metadata, block_table_cpu_tensor,
                model_runner):
        _wrap_compute_logits(self.model)
        _manual_frontend(self.model, target_token_ids, target_hidden_states)
        out = _orig(self, target_token_ids, target_positions, target_hidden_states,
                    last_token_indices, common_attn_metadata, block_table_cpu_tensor,
                    model_runner)
        try:
            tin = target_token_ids.reshape(-1).to("cpu")[:8].tolist()
            pos = target_positions.reshape(-1).to("cpu")[:8].tolist()
            hs = target_hidden_states
            hn = float(hs.float().norm().item()) if hs is not None else -1.0
            dout = out.reshape(-1).to("cpu")[:8].tolist()
            log.warning("[mtp-probe] in_tok=%s pos=%s hid_norm=%.3f -> draft=%s",
                        tin, pos, hn, dout)
        except Exception as ex:  # noqa: BLE001 — never break serving from a probe
            log.warning("[mtp-probe] readback failed: %s", ex)
        return out

    _probed._mtp_probe = True
    HpuEagleProposer.propose = _probed
    log.warning("[mtp-probe] INSTALLED (HPU_MTP_PROBE=1): logging draft in/out token ids")
