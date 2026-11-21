"""Real-request hidden-state probe for GLM-5.1 / DeepSeek-V3.2 on HPU.

PROBLEM BEING DIAGNOSED
-----------------------
"The capital of France is" -> "andiandiandi..." in greedy decode.
Every candidate fix so far (bit-exact weights, compile-boundary, FusedSDPA bypass,
full-attention fallback via FULL_ATTENTION_FALLBACK=True) has NOT fixed it.

THIS PROBE (v2 - enforce-eager compatible)
------------------------------------------
Gates on REAL requests only.  The v1 probe used a skip-counter (_SKIP_N=200)
calibrated for torch.compile mode where warmup runs ~200+ model.forward passes.
Under --enforce-eager the runner calls model.forward at most a handful of times
during warmup (or zero, since warmup_graphs is skipped for enforce_eager).  So
the v1 counter never exceeded _SKIP_N=200 and the probe never fired.

GATE STRATEGY (v2)
------------------
Two independent mechanisms, either of which can enable logging:

  MECHANISM A -- slot_mapping size gate (primary, works for BOTH modes):
    A real "capital of France" prompt has <=64 tokens padded to nearest bucket
    (e.g. 64 or 128).  Warmup dummy batches use max_model_len x batch_size tokens
    or at least fill a large bucket.  So ANY forward whose slot_mapping numel is
    small (<=HPU_LAYER_PROBE_MAX_TOKENS, default 64) is a real request, regardless
    of warmup mode or step count.

    Under --enforce-eager: warmup_graphs is skipped entirely -> no large dummy
    batches -> the first model.forward call IS the real request.  The slot_mapping
    gate fires immediately, correctly.

    Under torch.compile: warmup_graphs runs hundreds of buckets, each with large
    slot_mappings (always > 64 tokens even for smallest bucket which is typically
    128 or 256).  The gate stays silent during warmup and fires only on the real
    request.  This is safe because the smallest vllm-gaudi prefill bucket is
    configured by VLLM_PROMPT_QUERY_BUCKET_MIN (default 128).

  MECHANISM B -- step counter (belt-and-suspenders for compile mode):
    After _SKIP_N model.forward calls (default 0 under eager, 200 under compile),
    also enable logging.  Set HPU_LAYER_PROBE_SKIP to override.

  Final gate: _real_logged[0] < _PROBE_N caps total logged requests.

WHAT IS LOGGED
--------------
  - embed_tokens output: norm, nan, dtype
  - Per-layer [0,1,2,4,8,16,32,64,77]:
      * input to input_layernorm (hidden + residual norm)
      * mlp/moe output norm (delta)
      * final residual norm after layer
      * combined (h+r) norm
      * nan/inf count for each
  - HPUMLAImpl.forward_mha: Q norm, latent_vec_k norm, output norm
  - HPUMLAImpl.forward_mqa: q_nope norm, q_pe norm, output norm
  - compute_logits: hidden_in norm, top-1 token IDs (first 8 positions)
  - Rank 0 only (TP rank check)

USAGE
-----
  # Enforce-eager (recommended for debugging) -- slot gate fires immediately:
  HPU_LAYER_PROBE=3 python -m vllm.entrypoints.openai.api_server --enforce-eager ...

  # torch.compile mode (set skip to warmup count, e.g. 300):
  HPU_LAYER_PROBE=3 HPU_LAYER_PROBE_SKIP=300 python -m vllm.entrypoints.openai.api_server ...

  # Tighten the real-request gate (for very short prompts):
  HPU_LAYER_PROBE=3 HPU_LAYER_PROBE_MAX_TOKENS=64 ...

Search for "[layer-probe]" in vllm logs.

HOW TO READ THE OUTPUT
----------------------
  [layer-probe] REAL-GATE FIRE: slot_mapping=12 step=1 (real request #1)
  [layer-probe] STEP 1 EMBED embed_out shape=[12, 6144] norm=142.3 ...
  [layer-probe] L00 hidden_in(h+r) norm=142.3 nan=0 ...
  [layer-probe] L00 mlp_out(delta) norm=5.2   nan=0 ...
  [layer-probe] L00 residual_out   norm=144.0 nan=0 ...
  [layer-probe] L00 combined_out   norm=144.0 nan=0 ...
  ...
  [layer-probe] L07 mlp_out(delta) norm=0.00  nan=0  <- ZERO: collapse here
  ...
  [layer-probe] LM_HEAD top1_tokens (first 8): [1234, 5678, ...]

A norm that collapses to 0 or explodes to NaN/Inf at a specific layer reveals
the collapse point.  Expected healthy norms for a 6144-hidden model: 10-500+.
Zero or NaN at a specific layer is the diagnostic signal.

IMPORTANT NOTE on layer indexing for GLM-5.1:
  hidden_size=6144, num_layers=78, first_k_dense_replace=3
  Layers 0,1,2 = dense MLP; layer 3+ = MoE (256 experts, top-8)
  kv_lora_rank=512, qk_nope=192, qk_rope=64, v_head=256, q_lora=2048
  num_attention_heads=64, num_nextn_predict_layers=1 (last layer)
"""
from __future__ import annotations

import logging
import os
import torch

log = logging.getLogger("vllm")

# --------------------------------------------------------------------------- #
# Configuration (env vars)
# --------------------------------------------------------------------------- #
_PROBE_N = int(os.environ.get("HPU_LAYER_PROBE", "0") or "0")

# HPU_LAYER_PROBE_SKIP: step-counter threshold.
# Default 0 under --enforce-eager (no warmup_graphs -> step counter stays low).
# Set to ~300 if using torch.compile mode to skip warmup passes.
_SKIP_N  = int(os.environ.get("HPU_LAYER_PROBE_SKIP", "0") or "0")

# HPU_LAYER_PROBE_MAX_TOKENS: primary real-request gate.
# Real prompts for short queries are <=64 tokens.
# Warmup smallest bucket is typically 128+ tokens even on smallest config.
# Set lower (e.g. 32) for very short prompts, higher if your prompt is longer.
_MAX_TOKENS_REAL = int(os.environ.get("HPU_LAYER_PROBE_MAX_TOKENS", "64") or "64")

# Layers to probe.  Covers: dense-MLP layers (0,1,2), first MoE (3 -> using 4),
# then powers-of-2 through the 78-layer GLM-5.1 model, plus the last layer.
_PROBE_LAYERS = {0, 1, 2, 4, 8, 16, 32, 64, 77}

# --------------------------------------------------------------------------- #
# Shared state (mutable via list to allow closure mutation)
# --------------------------------------------------------------------------- #
_model_step   = [0]   # total model.forward calls (warmup + real)
_real_logged  = [0]   # how many real-request logs we've emitted
_installed    = [False]


# --------------------------------------------------------------------------- #
# Logging helpers (all @torch.compiler.disable -- must not enter HPU graph)
# --------------------------------------------------------------------------- #

@torch.compiler.disable
def _stat(t: torch.Tensor, name: str) -> str:
    """Return a compact stats string for a tensor, always on CPU float32."""
    try:
        f = t.detach().to(dtype=torch.float32, device="cpu")
        nan_count = int(f.isnan().sum().item())
        inf_count = int(f.isinf().sum().item())
        return (
            f"{name} shape={list(t.shape)} dtype={t.dtype} "
            f"norm={f.norm().item():.4f} "
            f"absmax={f.abs().max().item():.4f} "
            f"mean={f.mean().item():.6f} "
            f"nan={nan_count} "
            f"inf={inf_count}"
        )
    except Exception as ex:
        return f"{name}: stat-error({ex})"


@torch.compiler.disable
def _log_tensor(tag: str, t: torch.Tensor, name: str) -> None:
    log.warning("[layer-probe] %s %s", tag, _stat(t, name))


@torch.compiler.disable
def _log_real_gate_banner(step: int, real_n: int, sm_numel: int) -> None:
    log.warning(
        "[layer-probe] ===== REAL-GATE FIRE: slot_mapping=%d step=%d "
        "(real request #%d) skip=%d probe_n=%d max_tokens=%d =====",
        sm_numel, step, real_n, _SKIP_N, _PROBE_N, _MAX_TOKENS_REAL,
    )


@torch.compiler.disable
def _is_real_request_slot_gate(attn_metadata) -> tuple[bool, int]:
    """Return (is_real, slot_mapping_numel).

    Primary gate: slot_mapping.numel() <= _MAX_TOKENS_REAL.
    This is the ONLY gate needed under --enforce-eager.

    Returns (False, 0) if metadata unavailable.
    """
    if attn_metadata is None:
        return False, 0
    sm = getattr(attn_metadata, "slot_mapping", None)
    if sm is None:
        return False, 0
    n = sm.numel()
    if n == 0 or n > _MAX_TOKENS_REAL:
        return False, n
    return True, n


@torch.compiler.disable
def _get_attn_metadata_from_context():
    """Extract attn_metadata from ForwardContext, handling dict-keyed variants."""
    try:
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        if fctx is None:
            return None
        am = fctx.attn_metadata
        if isinstance(am, dict):
            am = next(iter(am.values()), None)
        return am
    except Exception:
        return None


@torch.compiler.disable
def _should_log_this_forward() -> tuple[bool, int]:
    """Gate check: should we log this forward pass?

    Returns (should_log, slot_mapping_numel).
    Fires if:
      - _real_logged[0] < _PROBE_N  (haven't hit our cap yet)
      - AND either:
      (A) slot_mapping gate: numel <= _MAX_TOKENS_REAL  (primary, works for eager)
      (B) step counter gate: _model_step[0] > _SKIP_N  (secondary, compile mode)
    """
    if _real_logged[0] >= _PROBE_N:
        return False, 0
    am = _get_attn_metadata_from_context()
    slot_ok, n = _is_real_request_slot_gate(am)
    step_ok = _model_step[0] > _SKIP_N
    return (slot_ok or step_ok), n


# --------------------------------------------------------------------------- #
# TP rank check (log rank 0 only to avoid 8x flood)
# --------------------------------------------------------------------------- #

@torch.compiler.disable
def _is_rank_zero() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return True  # conservative: log if rank cannot be determined


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #

def install() -> None:
    """Register all forward hooks. Idempotent."""
    if _installed[0]:
        return
    if _PROBE_N <= 0:
        return

    # ------------------------------------------------------------------ #
    # 1. Model-level step counter hook on DeepseekV2Model.forward
    #    Increments _model_step[0] on every call (warmup and real).
    #    Under enforce-eager this counter reaches 1-5; under compile it
    #    reaches 200+.  The step-counter gate (_SKIP_N) is secondary;
    #    slot_mapping size is the primary gate.
    # ------------------------------------------------------------------ #
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
        _orig_model_fwd = DeepseekV2Model.forward

        @torch.compiler.disable
        def _model_fwd_counter(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
            _model_step[0] += 1
            return _orig_model_fwd(self, input_ids, positions, intermediate_tensors, inputs_embeds)

        DeepseekV2Model.forward = _model_fwd_counter
        log.warning("[layer-probe] DeepseekV2Model.forward step-counter hook installed")
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook DeepseekV2Model.forward: %s", ex)

    # ------------------------------------------------------------------ #
    # 2. embed_tokens probe -- hook on DeepseekV2Model.embed_input_ids
    #    Logs the embedding output norm+nan for the probed request.
    #    Also fires the REAL-GATE FIRE banner here so it appears before
    #    any layer logs.
    # ------------------------------------------------------------------ #
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
        _orig_embed = DeepseekV2Model.embed_input_ids

        @torch.compiler.disable
        def _embed_probe(self, input_ids):
            out = _orig_embed(self, input_ids)
            if not _is_rank_zero():
                return out
            should, n = _should_log_this_forward()
            if should:
                sm_n = n if n > 0 else input_ids.numel()
                _log_real_gate_banner(_model_step[0], _real_logged[0] + 1, sm_n)
                _log_tensor(f"STEP {_model_step[0]} EMBED", out, "embed_out")
            return out

        DeepseekV2Model.embed_input_ids = _embed_probe
        log.warning("[layer-probe] DeepseekV2Model.embed_input_ids probe installed")
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook embed_input_ids: %s", ex)

    # ------------------------------------------------------------------ #
    # 3. Per-layer decoder hook on DeepseekV2DecoderLayer.forward
    #
    #    Signature: forward(positions, hidden_states, residual, llama_4_scaling=None)
    #
    #    At each layer entry:
    #      hidden_states = RMSNorm(prev_h_delta + prev_residual)  [normed component]
    #      residual      = prev_h_delta + prev_residual            [unnormed accumulator]
    #    So full_input = hidden_states + residual gives the unnormed state.
    #    (At layer 0: residual=None, residual := clone(hidden_states) inside.)
    #
    #    Returns: (mlp_out_delta, residual_out)
    #      residual_out = old_residual + attn_out  (after post_attention_layernorm)
    #      mlp_out_delta = mlp/moe output
    #    Next layer: input_layernorm fuses residual_out + mlp_out_delta.
    # ------------------------------------------------------------------ #
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
        _orig_layer_fwd = DeepseekV2DecoderLayer.forward

        @torch.compiler.disable
        def _layer_probe(self_layer, positions, hidden_states, residual, *extra, **kwargs):
            layer_idx = getattr(self_layer, "layer_idx", -1)
            should, _n = _should_log_this_forward()
            should = should and (layer_idx in _PROBE_LAYERS) and _is_rank_zero()

            if should:
                tag = f"L{layer_idx:02d}"
                if residual is not None:
                    full_in = hidden_states.float() + residual.float()
                    _log_tensor(f"{tag} hidden_in(h+r)", full_in, "hidden_in")
                else:
                    _log_tensor(f"{tag} hidden_in(normed,no-resid)", hidden_states, "hidden_states")

            result = _orig_layer_fwd(self_layer, positions, hidden_states, residual, *extra, **kwargs)

            if should:
                tag = f"L{layer_idx:02d}"
                h_out, r_out = result if isinstance(result, tuple) else (result, None)
                _log_tensor(f"{tag} mlp_out(delta)", h_out, "mlp_out")
                if r_out is not None:
                    _log_tensor(f"{tag} residual_out", r_out, "residual_out")
                    combined = h_out.float() + r_out.float()
                    _log_tensor(f"{tag} combined_out(h+r)", combined, "combined")
                if layer_idx == max(_PROBE_LAYERS):
                    _real_logged[0] += 1
                    log.warning(
                        "[layer-probe] ===== REQUEST #%d COMPLETE (step=%d) =====",
                        _real_logged[0], _model_step[0],
                    )

            return result

        DeepseekV2DecoderLayer.forward = _layer_probe
        log.warning(
            "[layer-probe] DeepseekV2DecoderLayer.forward probe installed (layers %s)",
            sorted(_PROBE_LAYERS),
        )
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook DeepseekV2DecoderLayer.forward: %s", ex)

    # ------------------------------------------------------------------ #
    # 4. Attention inner probe on HPUMLAImpl.forward_mha (prefill path)
    #    Logs: q norm, latent_vec_k norm, output norm.
    #    Gate uses attn_metadata from the argument (not ForwardContext).
    # ------------------------------------------------------------------ #
    try:
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl
        _orig_fmha = HPUMLAImpl.forward_mha

        @torch.compiler.disable
        def _fmha_probe(self_impl, q, latent_vec_k, k_cache, attn_metadata, *extra, **kwargs):
            slot_ok, _n = _is_real_request_slot_gate(attn_metadata)
            step_ok = _model_step[0] > _SKIP_N
            should = (slot_ok or step_ok) and (_real_logged[0] < _PROBE_N) and _is_rank_zero()

            if should:
                impl_tag = f"fmha[{id(self_impl) & 0xFFFF:04x}]"
                _log_tensor(f"{impl_tag} Q", q, "q")
                _log_tensor(f"{impl_tag} latent_vec_k", latent_vec_k, "latent_vec_k")

            out = _orig_fmha(self_impl, q, latent_vec_k, k_cache, attn_metadata, *extra, **kwargs)

            if should:
                _log_tensor(f"{impl_tag} OUT", out, "attn_out")

            return out

        HPUMLAImpl.forward_mha = _fmha_probe
        log.warning("[layer-probe] HPUMLAImpl.forward_mha probe installed")
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook HPUMLAImpl.forward_mha: %s", ex)

    # ------------------------------------------------------------------ #
    # 5. HPUMLAImpl.forward_mqa (decode path)
    #    Logs: q_nope norm, q_pe norm, output norm.
    # ------------------------------------------------------------------ #
    try:
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl
        _orig_fmqa = HPUMLAImpl.forward_mqa

        @torch.compiler.disable
        def _fmqa_probe(self_impl, q_nope, q_pe, k_cache, attn_metadata, *extra, **kwargs):
            slot_ok, _n = _is_real_request_slot_gate(attn_metadata)
            step_ok = _model_step[0] > _SKIP_N
            should = (slot_ok or step_ok) and (_real_logged[0] < _PROBE_N) and _is_rank_zero()

            if should:
                impl_tag = f"fmqa[{id(self_impl) & 0xFFFF:04x}]"
                _log_tensor(f"{impl_tag} q_nope", q_nope, "q_nope")
                _log_tensor(f"{impl_tag} q_pe", q_pe, "q_pe")

            out = _orig_fmqa(self_impl, q_nope, q_pe, k_cache, attn_metadata, *extra, **kwargs)

            if should:
                _log_tensor(f"{impl_tag} OUT", out, "attn_out")

            return out

        HPUMLAImpl.forward_mqa = _fmqa_probe
        log.warning("[layer-probe] HPUMLAImpl.forward_mqa probe installed")
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook HPUMLAImpl.forward_mqa: %s", ex)

    # ------------------------------------------------------------------ #
    # 6. HPUMLAAttention.forward_impl (oot_mla wire level)
    # ------------------------------------------------------------------ #
    try:
        from vllm_gaudi.attention.oot_mla import HPUMLAAttention as _HPUMLAAttn
        _orig_wimpl = _HPUMLAAttn.forward_impl

        @torch.compiler.disable
        def _wire_probe(self_attn, q, k_c_normed, k_pe, kv_cache, attn_metadata, *extra, **kwargs):
            slot_ok, _n = _is_real_request_slot_gate(attn_metadata)
            step_ok = _model_step[0] > _SKIP_N
            should = (slot_ok or step_ok) and (_real_logged[0] < _PROBE_N) and _is_rank_zero()
            layer_nm = getattr(self_attn, "layer_name", f"id{id(self_attn) & 0xFFFF:04x}")

            if should:
                _log_tensor(f"wire[{layer_nm}] Q_in", q, "q")
                _log_tensor(f"wire[{layer_nm}] kc_normed", k_c_normed, "k_c_normed")
                if k_pe is not None:
                    _log_tensor(f"wire[{layer_nm}] k_pe", k_pe, "k_pe")

            out = _orig_wimpl(self_attn, q, k_c_normed, k_pe, kv_cache, attn_metadata, *extra, **kwargs)

            if should:
                _log_tensor(f"wire[{layer_nm}] OUT", out, "attn_out")

            return out

        _HPUMLAAttn.forward_impl = _wire_probe
        log.warning("[layer-probe] HPUMLAAttention.forward_impl wire probe installed")
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook HPUMLAAttention.forward_impl: %s", ex)

    # ------------------------------------------------------------------ #
    # 7. LM-head logit probe (GlmMoeDsaForCausalLM.compute_logits)
    #    Logs: hidden_in norm, full logit stats, top-1 token IDs.
    # ------------------------------------------------------------------ #
    try:
        from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
        _orig_logits = GlmMoeDsaForCausalLM.compute_logits

        @torch.compiler.disable
        def _logit_probe(self_model, hidden_states, **kwargs):
            if not _is_rank_zero():
                return _orig_logits(self_model, hidden_states, **kwargs)
            should, _n = _should_log_this_forward()
            if should:
                _log_tensor("LM_HEAD hidden_in", hidden_states, "hidden_states")
            out = _orig_logits(self_model, hidden_states, **kwargs)
            if should and out is not None:
                _log_tensor("LM_HEAD logits", out, "logits")
                try:
                    top1 = out.detach().to(dtype=torch.float32, device="cpu").argmax(dim=-1)
                    log.warning(
                        "[layer-probe] LM_HEAD top1_tokens (first 8): %s",
                        top1.flatten()[:8].tolist(),
                    )
                except Exception:
                    pass
            return out

        GlmMoeDsaForCausalLM.compute_logits = _logit_probe
        log.warning("[layer-probe] GlmMoeDsaForCausalLM.compute_logits probe installed")
    except Exception as ex:
        log.warning("[layer-probe] WARN: cannot hook GlmMoeDsaForCausalLM: %s", ex)

    _installed[0] = True
    log.warning(
        "[layer-probe] ALL probes installed. "
        "Will log real requests #1..%d. "
        "Gate A (primary): slot_mapping.numel <= %d. "
        "Gate B (secondary): step > %d. "
        "Probe layers: %s. "
        "Set HPU_LAYER_PROBE=0 to disable.",
        _PROBE_N, _MAX_TOKENS_REAL, _SKIP_N, sorted(_PROBE_LAYERS),
    )
