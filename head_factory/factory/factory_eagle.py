"""vllm-gaudi EAGLE draft-model wrapper for the from-scratch factory head (M3 / K2.7 / any model with no
native MTP layer to graft). Serves head_factory/<model>/head_v*.pt via the eagle spec-decode path:
HpuEagleProposer calls self.model(input_ids, positions, hidden_states) -> hidden, then
self.model.compute_logits(hidden) -> draft logits. Our DraftHead splits exactly into those (forward_hidden /
compute_logits). For method="eagle" (single hidden, no aux fusion) combine_hidden_states is NOT called, so a
single-layer LeanBlock head fits directly.

Registration + serve (after creating a draft-ckpt dir with config.json{architectures:[FactoryEagleForCausalLM],
+ our head HeadConfig fields + frozen_src} and the head weights as safetensors):
  from hpu_moe_ext... import factory_eagle; factory_eagle.register()   # ModelRegistry.register_model
  vllm serve <target> --speculative-config '{"method":"eagle","model":"<draft-ckpt-dir>","num_speculative_tokens":1}'
The embed/lm_head are the TARGET's (shared) — tied by mtp_share_head_patch at load, same as the native MTP path;
load_weights here loads ONLY the trainable head params (enorm/hnorm/eh_proj/block/final_norm).

STATUS: structure complete + maps 1:1 to the verified interface; needs serve-iteration to validate on HPU
(config plumbing + the embed/lm_head tie are the serve-test surface), exactly like the fp8 graft bring-up.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from head import DraftHead, HeadConfig  # noqa: E402  (the proven head; reused, not reimplemented)


class _FactoryInner(nn.Module):
    """Mirrors DeepSeekMultiTokenPredictor: holds the draft head + the (shared) embed_tokens slot + the
    logits_processor. The proposer wires self.embed_tokens -> the TARGET's VocabParallelEmbedding."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        dummy = torch.zeros(1, cfg.hidden, dtype=torch.bfloat16)  # embed_w/lm_head_w UNUSED at serve
        self.head = DraftHead(cfg, dummy, dummy)
        self.embed_tokens = None                      # proposer sets -> TARGET VocabParallelEmbedding
        self.logits_processor = LogitsProcessor(cfg.vocab)

    def combine_hidden_states(self, hidden_states):
        """EAGLE-3 fusion hook (method='eagle3'). The HpuEagleProposer calls
        self.model.model.combine_hidden_states(aux_3H) BEFORE forward (hpu_eagle.py:36) to reduce the
        target's 3-layer aux stack [T, n_aux*H] -> [T, H], then asserts shape[-1]==hidden_size. This is
        EXACTLY our head's _fuse_aux (per-chunk RMSNorm + fc 3H->H), so the trained fc/fc_norm run at
        serve => train==serve for the aux head. With n_aux=0 (legacy) the head has no fc -> pass through
        (the proposer wouldn't call this anyway, since method='eagle3' implies an aux head)."""
        if getattr(self.head.cfg, "n_aux", 0) and hidden_states.shape[-1] != self.head.cfg.hidden:
            return self.head._fuse_aux(hidden_states)
        return hidden_states


class FactoryEagleForCausalLM(nn.Module):
    """vllm draft-model adapter for the factory head, shaped like a native MTP model (NO has_own_lm_head /
    has_own_embed_tokens) so the spec-decode proposer shares the TARGET's embed_tokens -> self.model.embed_tokens
    and lm_head -> self.lm_head. forward()/compute_logits() read exactly those shared modules; the trained head
    params live in self.model.head. For method='eagle' the proposer unpacks (last_hidden, hidden) so forward
    returns a 2-tuple (the 2nd is only used at depth>=2; identical to the 1st for our depth-1 head).

    EAGLE-3 (method='eagle3'): the proposer pre-fuses the target's 3-layer aux stack via
    self.model.model.combine_hidden_states (see _FactoryInner) -> the trained fc/fc_norm run at serve,
    matching the aux-trained head (train==serve). supports_eagle3 advertises the capability."""

    supports_eagle3 = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        hf = vllm_config.model_config.hf_config
        # transformers' LlamaConfig drops our custom 'head_config' nested dict (and even hidden_size on the
        # draft config object is unreliable through the spec-decode load path), so read the HeadConfig
        # straight from the draft-ckpt config.json. LOUD failure if it can't be resolved (no silent fallback).
        import json as _json, os as _os
        hcd = getattr(hf, "head_config", None)
        if isinstance(hcd, str):
            hcd = _json.loads(hcd)
        if not hcd:
            cand = []
            p = getattr(hf, "_name_or_path", None)
            if p:
                cand.append(p)
            sc = getattr(vllm_config, "speculative_config", None)
            dmc = getattr(sc, "draft_model_config", None) if sc else None
            if dmc is not None:
                cand.append(getattr(dmc, "model", None))
            for c in cand:
                cj = _os.path.join(c, "config.json") if c and not c.endswith(".json") else c
                if cj and _os.path.isfile(cj):
                    raw = _json.load(open(cj))
                    hcd = raw.get("head_config") or {k: raw[k] for k in HeadConfig.__dataclass_fields__ if k in raw}
                    if hcd:
                        break
        hcd = dict(hcd or {})
        hcd.setdefault("hidden", getattr(hf, "hidden_size", None))
        hcd.setdefault("vocab", getattr(hf, "vocab_size", None))
        cfg_kw = {k: v for k, v in hcd.items() if k in HeadConfig.__dataclass_fields__ and v is not None}
        missing = [f for f in ("hidden", "vocab", "n_heads", "head_dim") if f not in cfg_kw]
        if missing:
            raise RuntimeError(
                f"[factory-eagle] could not resolve HeadConfig {missing} from draft ckpt; "
                f"hf._name_or_path={getattr(hf, '_name_or_path', None)!r} head_config_attr={getattr(hf, 'head_config', None)!r}")
        cfg = HeadConfig(**cfg_kw)
        self.config = hf
        self.model = _FactoryInner(cfg)
        self.lm_head = None  # proposer sets -> TARGET ParallelLMHead (shared); compute_logits reads it

    def get_input_embeddings(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None, **kw):
        # proposer: self.model(input_ids, positions, hidden_states=last_target_hidden, inputs_embeds=None, ...)
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        h = self.model.head.forward_hidden_embeds(hidden_states, inputs_embeds, positions)
        return h, h  # (last_hidden, hidden); method='eagle' unpacks both, depth-1 uses only the first

    def compute_logits(self, hidden, *a, **k):
        # shared TARGET lm_head via the LogitsProcessor (TP all-gather), exactly like native MTP compute_logits
        return self.model.logits_processor(self.lm_head, hidden)

    def combine_hidden_states(self, hidden_states):
        # EAGLE-3 aux fusion. The proposer may resolve `_inner` to THIS class (an extra wrapper
        # level: self.model.model == FactoryEagleForCausalLM) instead of _FactoryInner; delegate so
        # the trained fc/fc_norm (3H->H) run regardless of which level the proposer lands on.
        return self.model.combine_hidden_states(hidden_states)

    def load_weights(self, weights):
        """Load ONLY the trainable head params into self.model.head (embed/lm_head are shared from target).
        MUST return FULLY-QUALIFIED param names (e.g. 'model.head.eh_proj.weight') — vllm's
        track_weights_loading flags any model.named_parameters() name not in the returned set as
        'not initialized from checkpoint'. Safetensors keys are 'head.<k>'; the live param is 'model.head.<k>'.
        fp8-stored heads ship a per-tensor '<w>_scale' alongside each fp8 weight -> dequant at load (one-time;
        the served param is the model's activation dtype, like every other weight on this bf16-activation stack)."""
        params = dict(self.named_parameters())
        weights = list(weights)
        scales = {n[: -len("_scale")]: w for n, w in weights if n.endswith("_scale")}
        loaded = set()
        for name, w in weights:
            if name.endswith("_scale"):
                continue
            full = name if name in params else ("model." + name)  # 'head.<k>' -> 'model.head.<k>'
            if full not in params or "embed_w" in full or "lm_head_w" in full:
                continue
            if name in scales:  # fp8 e4m3 -> real values
                w = w.to(torch.float32) * scales[name].to(torch.float32)
            with torch.no_grad():
                params[full].copy_(w.to(params[full].dtype))
            loaded.add(full)
        return loaded


def export_draft_ckpt(head_pt: str, out_dir: str) -> None:
    """head_v*.pt (cfg + trainable state) -> a vllm-loadable EAGLE draft ckpt (config.json + safetensors).
    The embed/lm_head are NOT exported (tied to the target at serve-load)."""
    import json
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    d = torch.load(head_pt, map_location="cpu", weights_only=False)
    cfg_d = dict(d["cfg"]); state = d["state"]
    head_fields = {k: v for k, v in cfg_d.items() if k in HeadConfig.__dataclass_fields__}
    # model_type="llama" so transformers AutoConfig loads it (factory_eagle is unknown to HF); vllm dispatches
    # on architectures=[FactoryEagleForCausalLM]. The HeadConfig rides in a nested 'head_config' dict (top-level
    # extras get dropped by LlamaConfig, so __init__ reads head_config).
    hf = {"architectures": ["FactoryEagleForCausalLM"], "model_type": "llama",
          "hidden_size": cfg_d["hidden"], "vocab_size": cfg_d["vocab"],
          "num_hidden_layers": 1, "num_attention_heads": cfg_d.get("n_heads", 32),
          "tie_word_embeddings": True, "head_config": head_fields}
    # EAGLE-3 hints (read by the eagle proposer / some config paths): how many aux residuals the head
    # fuses (n_aux) so combine_hidden_states sizing + the aux-layer count line up. Harmless for n_aux=0.
    if int(cfg_d.get("n_aux", 0) or 0) > 0:
        hf["num_aux_hidden_states"] = int(cfg_d["n_aux"])
        hf["use_aux_hidden_state"] = True
    json.dump(hf, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    save_file({f"head.{k}": v.contiguous() for k, v in state.items()
               if "embed_w" not in k and "lm_head_w" not in k},
              os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    print(f"[factory-eagle] exported EAGLE draft ckpt -> {out_dir}")


def quantize_head_fp8(ckpt_dir: str) -> None:
    """Post-process an exported eagle_ckpt to fp8 e4m3 (bf16-ban compliance; matches the GLM fp8-graft
    precedent). Per-tensor-scaled fp8 for the 2-D Linear weights (eh_proj / blocks.*.gate_up / *.down);
    1-D norms stay bf16 (tiny + scale-sensitive). load_weights() dequants <w>+<w>_scale at serve."""
    import json
    from safetensors.torch import load_file, save_file
    p = os.path.join(ckpt_dir, "model.safetensors")
    sd = load_file(p)
    FP8_MAX = 448.0  # e4m3 max magnitude
    out = {}
    nq = 0
    for k, w in sd.items():
        if k.endswith("_scale"):
            continue  # already fp8-quantized; re-derive below
        if w.dim() == 2 and w.numel() > 4096:  # a Linear weight -> fp8
            amax = w.abs().max().to(torch.float32).clamp(min=1e-8)
            scale = (amax / FP8_MAX)
            wq = (w.to(torch.float32) / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            out[k] = wq
            out[k + "_scale"] = scale.reshape(1)
            nq += 1
        else:
            out[k] = w.to(torch.bfloat16) if w.dtype == torch.float32 else w  # norms: bf16, never fp32
    save_file(out, p, metadata={"format": "pt"})
    print(f"[factory-eagle] fp8-quantized {nq} Linear weights (e4m3 per-tensor) + bf16 norms -> {p}")


def register() -> None:
    try:
        from vllm import ModelRegistry
        # vllm's eagle method prepends "Eagle" to the draft arch -> "EagleFactoryEagleForCausalLM";
        # method=eagle3 prepends "Eagle3" -> "Eagle3FactoryEagleForCausalLM". Register the plain +
        # both prefixed names so SpeculativeConfig validation finds it for eagle AND eagle3.
        for name in ("FactoryEagleForCausalLM", "EagleFactoryEagleForCausalLM",
                     "Eagle3FactoryEagleForCausalLM"):
            ModelRegistry.register_model(name, "factory_eagle:FactoryEagleForCausalLM")
    except Exception as ex:  # noqa: BLE001
        print(f"[factory-eagle] register failed: {ex!r}", file=sys.stderr)
