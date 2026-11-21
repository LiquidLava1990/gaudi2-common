"""Real-request activation probe for GLM-5.1 / GlmMoeDsa on Gaudi 2.

PURPOSE
-------
The existing HPU_ATTN_PROBE and HPU_MOE_PROBE fire on the FIRST N forward_impl
calls, which are HPU graph warmup (dummy) passes — their data is warmup-contaminated
and must not be trusted as real-request behavior.

This probe gates on the real request by checking whether the current forward is a
warmup pass and skipping it if so. It uses TWO complementary gating mechanisms:

  1. MODEL RUNNER FLAG (primary): HPUModelRunner sets self.warmup_mode = True before
     every _execute_dummy_scenario call and self.warmup_mode = False after
     (hpu_model_runner.py:3883 / 3972). The probe reads this flag via a reference to
     the HPUModelRunner instance injected at install time. When warmup_mode is True
     the probe skips logging.

  2. LATENT-NORM SANITY GUARD (secondary, belt-and-suspenders): warmup/dummy passes
     feed token_ids = list(range(total_tokens)) with all-padding slots (slot_mapping
     maps to dummy slots). The latent KV written to the cache during warmup is
     model-weight-derived (embed_tokens(range(N))), not random — it can be nonzero.
     However the warmup passes also populate k_c_normed from dummy embeddings. To
     guard against accidental re-fire on an edge case (e.g. DP dummy pass after real
     warmup), the probe also checks that it fires at most PROBE_MAX_REAL times after
     warmup mode has exited.

  3. FORWARD COUNTER (tertiary): a global call counter incremented on every
     forward_impl call. Warmup fires the full bucket matrix (all (batch, seq,
     num_blocks) combos), which on a 128-layer model means hundreds of calls. The
     real first request will be the first call AFTER warmup_mode transitions to False.
     The probe fires on the FIRST PROBE_MAX_REAL such calls only.

WHAT IT LOGS
------------
On the first real prefill and the first real decode, it logs per-layer:
  - layer index and phase (prefill/decode)
  - q_norm, q_absmax, q_nan: query quality indicator (collapse -> norm near 0)
  - attn_out_norm, attn_out_absmax, attn_out_nan: attention output health
  - k_c_normed_norm: latent KV input norm (signals embedding collapse)

Additionally, on the first call it logs the full embedding output norm by hooking
DeepseekV2Model.forward before the first layer.

INSTALL
-------
Set HPU_REAL_PROBE=N (N > 0) before starting vllm. The probe installs itself
via install(model_runner) called from the plugin registration hook. Alternatively
call install() from a monkey-patch point that has access to the model runner.

If you cannot inject the model runner reference (e.g. for quick testing), set
HPU_REAL_PROBE_NO_WARMUP_GATE=1 to skip the warmup gate and use the counter alone
(the probe will still skip the first WARMUP_SKIP_COUNT calls as a heuristic).

EXAMPLE
-------
  HPU_REAL_PROBE=2 python -m vllm.entrypoints.openai.api_server ...

This will log the first 2 real forward_impl calls (one prefill + one decode) with
full per-layer activation stats.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    pass

log = logging.getLogger("vllm")

# --- Configuration -----------------------------------------------------------

# Number of real (post-warmup) forward_impl calls to log. 0 = disabled.
_PROBE_N: int = int(os.environ.get("HPU_REAL_PROBE", "0") or "0")

# When HPU_REAL_PROBE_NO_WARMUP_GATE=1, skip the model-runner warmup_mode check
# and instead skip the first WARMUP_SKIP_HEURISTIC calls (heuristic).
_NO_WARMUP_GATE: bool = os.environ.get("HPU_REAL_PROBE_NO_WARMUP_GATE", "0") == "1"

# Heuristic warmup-skip count used only when _NO_WARMUP_GATE=True.
# GLM-5.1 warmup: ~(num_prompt_buckets + num_decode_buckets) * num_layers calls.
# Setting 5000 covers a full warmup matrix before the first real request.
_WARMUP_SKIP_HEURISTIC: int = int(os.environ.get("HPU_REAL_PROBE_WARMUP_SKIP", "5000") or "5000")

# --- State -------------------------------------------------------------------

_lock = threading.Lock()
_total_calls = 0          # total forward_impl calls seen
_real_calls_logged = 0    # real calls actually logged
_model_runner_ref = None  # injected HPUModelRunner instance (or None)
_installed = False


# --- Core probe logic --------------------------------------------------------

@torch.compiler.disable
def _is_real_call() -> bool:
    """Return True iff the current forward_impl call is a real (post-warmup) request."""
    global _total_calls
    with _lock:
        _total_calls += 1
        cur = _total_calls

    if _NO_WARMUP_GATE:
        # Heuristic: skip the first _WARMUP_SKIP_HEURISTIC calls.
        return cur > _WARMUP_SKIP_HEURISTIC

    # Primary gate: check model runner warmup_mode flag.
    mr = _model_runner_ref
    if mr is not None:
        try:
            if getattr(mr, "warmup_mode", False):
                return False
        except Exception:
            pass
        return True  # model runner says NOT warmup -> real request

    # No model runner injected: fall back to heuristic.
    return cur > _WARMUP_SKIP_HEURISTIC


@torch.compiler.disable
def _log_real_forward(
    layer_name: str,
    is_prompt: bool,
    q: torch.Tensor,
    k_c_normed: torch.Tensor,
    attn_out: torch.Tensor,
    call_idx: int,
) -> None:
    """Log activation stats for one real forward_impl call. All .cpu() transfers
    are inside this @torch.compiler.disable scope so they don't affect the HPU graph."""
    try:
        qf = q.detach().float()
        kf = k_c_normed.detach().float()
        of = attn_out.detach().float()

        q_norm = float(qf.norm())
        q_absmax = float(qf.abs().max()) if qf.numel() > 0 else 0.0
        q_nan = bool(torch.isnan(qf).any())
        q_inf = bool(torch.isinf(qf).any())

        kc_norm = float(kf.norm())
        kc_absmax = float(kf.abs().max()) if kf.numel() > 0 else 0.0
        kc_nan = bool(torch.isnan(kf).any())

        o_norm = float(of.norm())
        o_absmax = float(of.abs().max()) if of.numel() > 0 else 0.0
        o_nan = bool(torch.isnan(of).any())
        o_inf = bool(torch.isinf(of).any())

        log.warning(
            "[real-probe #%d] layer=%s phase=%s "
            "q: shape=%s norm=%.4f absmax=%.4f nan=%s inf=%s | "
            "kc_normed: norm=%.4f absmax=%.4f nan=%s | "
            "attn_out: shape=%s norm=%.4f absmax=%.4f nan=%s inf=%s",
            call_idx,
            layer_name,
            "prefill" if is_prompt else "decode",
            list(q.shape), q_norm, q_absmax, q_nan, q_inf,
            kc_norm, kc_absmax, kc_nan,
            list(attn_out.shape), o_norm, o_absmax, o_nan, o_inf,
        )
    except Exception as ex:
        log.warning("[real-probe] log error in _log_real_forward: %s", ex)


# --- Embedding probe (fires once on first real prefill) ----------------------

_embed_probe_done = False
_embed_probe_lock = threading.Lock()


@torch.compiler.disable
def _log_embed(hidden_states: torch.Tensor) -> None:
    try:
        hf = hidden_states.detach().float()
        log.warning(
            "[real-probe] embed_tokens output: shape=%s norm=%.4f absmax=%.4f "
            "nan=%s inf=%s mean=%.4f std=%.4f",
            list(hf.shape),
            float(hf.norm()),
            float(hf.abs().max()) if hf.numel() > 0 else 0.0,
            bool(torch.isnan(hf).any()),
            bool(torch.isinf(hf).any()),
            float(hf.mean()),
            float(hf.std()),
        )
    except Exception as ex:
        log.warning("[real-probe] log error in _log_embed: %s", ex)


# --- Main patch --------------------------------------------------------------

def install(model_runner=None) -> bool:
    """Install the real-request probe.

    Parameters
    ----------
    model_runner : optional
        The HPUModelRunner instance. When provided, the probe uses its
        warmup_mode flag as the primary real-request gate (most reliable).
        If None, the heuristic skip counter is used instead.

    Returns
    -------
    bool : True if installed, False if already installed or probe disabled.
    """
    global _installed, _model_runner_ref

    if _PROBE_N <= 0:
        return False

    with _lock:
        if _installed:
            return False
        _installed = True
        if model_runner is not None:
            _model_runner_ref = model_runner

    _do_install()
    return True


def _do_install() -> None:
    """Internal: apply the monkeypatches. Called once under _lock."""

    # --- 1. Patch HPUMLAAttention.forward_impl (the _ORIG_FORWARD_IMPL target) ---
    try:
        from vllm_gaudi.attention.oot_mla import HPUMLAAttention
    except ImportError:
        log.warning("[real-probe] cannot import HPUMLAAttention — probe not installed")
        return

    # Grab the current forward_impl (which may already be sparse_mla_wire's
    # _sparse_forward_impl). We wrap THAT so we get called regardless of whether
    # FULL_ATTENTION_FALLBACK is True or False.
    _current_fwd = HPUMLAAttention.forward_impl

    def _probed_forward_impl(
        self,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        global _real_calls_logged

        # Run the real forward first (probe is read-only).
        result = _current_fwd(
            self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )

        # Gate: only log on real (post-warmup) calls.
        if not _is_real_call():
            return result

        with _lock:
            if _real_calls_logged >= _PROBE_N:
                return result
            call_idx = _real_calls_logged
            _real_calls_logged += 1

        is_prompt = getattr(attn_metadata, "is_prompt", None)
        layer_name = getattr(self, "layer_name", "?")

        _log_real_forward(
            layer_name=layer_name,
            is_prompt=bool(is_prompt),
            q=q,
            k_c_normed=k_c_normed,
            attn_out=result,
            call_idx=call_idx,
        )

        return result

    _probed_forward_impl._real_probe_wired = True
    HPUMLAAttention.forward_impl = _probed_forward_impl
    log.warning(
        "[real-probe] installed on HPUMLAAttention.forward_impl "
        "(PROBE_N=%d, warmup_gate=%s, warmup_skip_heuristic=%d)",
        _PROBE_N,
        "model_runner.warmup_mode" if _model_runner_ref is not None else "heuristic(%d)" % _WARMUP_SKIP_HEURISTIC,
        _WARMUP_SKIP_HEURISTIC,
    )

    # --- 2. Patch embed_tokens output for embedding-level collapse detection ---
    _install_embed_probe()

    # --- 3. Patch DeepseekV2DecoderLayer for per-layer hidden state norms ----
    _install_decoder_layer_probe()


def _install_embed_probe() -> None:
    """Hook embed_input_ids (the first thing DeepseekV2Model.forward calls) to
    log the very first real embedding output.

    DeepseekV2Model.forward (deepseek_v2.py:1285) calls self.embed_input_ids(input_ids)
    on the first PP rank. We hook that method so we see the raw embedding output
    before any layer runs — the key signal for embedding-level collapse (fp16
    embed_tokens producing near-zero or NaN embeddings).

    Signature: embed_input_ids(self, input_ids) -> hidden_states (fp16/bf16 tensor).
    """
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
    except ImportError:
        log.warning("[real-probe] cannot import DeepseekV2Model for embed probe")
        return

    _orig_embed = DeepseekV2Model.embed_input_ids

    def _probed_embed(self, input_ids, **kwargs):
        global _embed_probe_done

        result = _orig_embed(self, input_ids, **kwargs)

        # Only log once, only on a real (post-warmup) call.
        # Use the warmup_mode flag directly (don't use _is_real_call() here
        # because that increments _total_calls, and embed fires before forward_impl).
        is_warmup = getattr(_model_runner_ref, "warmup_mode", None)
        is_real = (is_warmup is False) if (is_warmup is not None) else True

        if is_real and not _embed_probe_done:
            with _embed_probe_lock:
                if not _embed_probe_done:
                    _embed_probe_done = True
                    _log_embed(result)

        return result

    DeepseekV2Model.embed_input_ids = _probed_embed
    log.warning("[real-probe] embed probe installed on DeepseekV2Model.embed_input_ids")


def _install_decoder_layer_probe() -> None:
    """Hook DeepseekV2DecoderLayer.forward to log per-layer hidden state norms
    on the first real call. Logs layers 0-4 (embedding through first MoE) and
    the last layer to see where norms collapse."""
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    except ImportError:
        log.warning("[real-probe] cannot import DeepseekV2DecoderLayer for layer probe")
        return

    _orig_layer_fwd = DeepseekV2DecoderLayer.forward
    _layer_probe_call_count = [0]   # mutable box for closure
    _LAYER_PROBE_MAX = 1            # log for first 1 real model pass only

    @torch.compiler.disable
    def _log_layer(layer_idx, phase, hidden_states, residual, out):
        try:
            hf = hidden_states.detach().float() if hidden_states is not None else None
            rf = residual.detach().float() if residual is not None else None
            of_raw = out[0] if isinstance(out, (tuple, list)) else out
            of = of_raw.detach().float() if of_raw is not None else None

            parts = [
                f"[real-probe] layer={layer_idx} phase={phase}",
            ]
            if hf is not None:
                parts.append(
                    f"  IN: norm={float(hf.norm()):.4f} absmax={float(hf.abs().max()):.4f} "
                    f"nan={bool(torch.isnan(hf).any())} inf={bool(torch.isinf(hf).any())}"
                )
            if rf is not None:
                parts.append(
                    f"  RESIDUAL_IN: norm={float(rf.norm()):.4f} absmax={float(rf.abs().max()):.4f} "
                    f"nan={bool(torch.isnan(rf).any())}"
                )
            if of is not None:
                parts.append(
                    f"  OUT: norm={float(of.norm()):.4f} absmax={float(of.abs().max()):.4f} "
                    f"nan={bool(torch.isnan(of).any())} inf={bool(torch.isinf(of).any())}"
                )
            log.warning("\n".join(parts))
        except Exception as ex:
            log.warning("[real-probe] layer log error layer=%d: %s", layer_idx, ex)

    def _probed_layer_fwd(self, *args, **kwargs):
        # Variadic pass-through: DeepseekV2DecoderLayer.forward's signature varies by
        # vllm version (positions, hidden_states, residual, [+ extra positional in
        # 0.22.x]); accept anything and forward verbatim so the probe can NEVER break
        # the model's forward. (Previous fixed 4-arg sig crashed warmup: "takes 4
        # positional arguments but 5 were given".)
        layer_idx = getattr(self, "layer_idx", -1)

        # Only probe on first real pass and only for diagnostic layers.
        should_probe = (
            _layer_probe_call_count[0] < _LAYER_PROBE_MAX
            and layer_idx in (0, 1, 2, 3, 4, 5, 76, 77)  # first 6 + last 2 of GLM-5.1
            and not getattr(_model_runner_ref, "warmup_mode", True)  # not warmup
        )

        result = _orig_layer_fwd(self, *args, **kwargs)

        if should_probe:
            # positional convention: (positions, hidden_states, residual, ...)
            hidden_states = args[1] if len(args) > 1 else kwargs.get("hidden_states")
            residual = args[2] if len(args) > 2 else kwargs.get("residual")
            _log_layer(layer_idx, "layer", hidden_states, residual, result)

        # Advance pass counter at the last layer.
        if layer_idx == 77:
            _layer_probe_call_count[0] += 1

        return result

    DeepseekV2DecoderLayer.forward = _probed_layer_fwd
    log.warning("[real-probe] decoder layer probe installed on DeepseekV2DecoderLayer.forward")


# --- Convenience: auto-install from environment at import time ---------------
# This fires when the module is imported (e.g. via hpu_dsa_ext register()).
# The model_runner injection happens later via inject_model_runner().

def inject_model_runner(model_runner) -> None:
    """Inject the HPUModelRunner reference after the model runner is initialized.

    Call this from the model runner's __init__ or warmup_model hook so the probe
    can use the authoritative warmup_mode flag. Idempotent.

    Usage (from a monkey-patch on HPUModelRunner.__init__):
        from hpu_dsa_ext.real_request_probe import inject_model_runner
        inject_model_runner(self)
    """
    global _model_runner_ref
    with _lock:
        _model_runner_ref = model_runner
    if _PROBE_N > 0:
        log.warning(
            "[real-probe] model_runner injected — warmup gate is now "
            "model_runner.warmup_mode (authoritative)"
        )


def register() -> None:
    """Entry-point compatible with hpu_dsa_ext._SPARSE_COMPONENTS pattern.

    Installs the probe if HPU_REAL_PROBE > 0. Also installs a hook on
    HPUModelRunner.__init__ to auto-inject the model runner reference so
    the warmup gate works without manual injection.
    """
    if _PROBE_N <= 0:
        return

    # Install the forward_impl probe immediately (model runner not yet available).
    install(model_runner=None)

    # Install a hook on HPUModelRunner.__init__ to auto-inject the runner ref.
    _install_model_runner_hook()


def _install_model_runner_hook() -> None:
    """Wrap HPUModelRunner.__init__ to auto-inject the model runner reference."""
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except ImportError:
        log.warning(
            "[real-probe] cannot import HPUModelRunner — model_runner injection "
            "will not happen automatically. Call inject_model_runner(runner) manually."
        )
        return

    if getattr(HPUModelRunner.__init__, "_real_probe_hooked", False):
        return

    _orig_init = HPUModelRunner.__init__

    def _hooked_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        inject_model_runner(self)

    _hooked_init._real_probe_hooked = True
    HPUModelRunner.__init__ = _hooked_init
    log.warning("[real-probe] HPUModelRunner.__init__ hooked for auto model_runner injection")
