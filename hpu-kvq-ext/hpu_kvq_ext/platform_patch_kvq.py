"""Extend the vllm_gaudi platform KV-dtype gate so the TurboQuant sub-fp8 KV
dtypes ('turboquant_k8v4', 'turboquant_k8v2') become *acceptable* on HPU --
WITHOUT touching /opt or the installed plugins.

The stock gate (vllm_gaudi/platform.py, HpuPlatform.is_kv_cache_dtype_supported)
returns ``kv_cache_dtype == "fp8_inc"`` -- fp8_inc is the ONLY accepted HPU KV
dtype. This monkeypatches that classmethod IN-PROCESS to ALSO accept the
TurboQuant dtypes, gated behind KVQ_READY (default False). It is the KV-dtype
twin of hpu-dsa-ext's platform_patch.apply().

SAFETY (the PR #1301 / DSA_READY trap -- do NOT repeat it):
  * KVQ_READY defaults False. Until it is True, this patch is a NO-OP: the gate
    still returns ``== "fp8_inc"`` for everything, so fp8_inc remains the ONLY
    accepted dtype and every current serve is unchanged.
  * KVQ_READY flips True ONLY when HPU_KVQ_READY is truthy AND the two compiled
    ops (kv_rotate_quant + kv_dequant_unrotate) are loaded AND the cache-spec /
    attention wiring that actually stores+reads the TurboQuant KV is installed.
    Accepting the dtype WITHOUT that wiring would let vLLM allocate a KV pool in a
    dtype nothing knows how to write/read -> garbage. So the gate stays closed
    until the write/read path is proven, exactly as DSA_READY gates the sparse path.
  * MLA GUARD: TurboQuant is a GQA/MHA (raw per-head K,V) technique. On an MLA
    latent it is measured-dead (~4-5x fp8 error, GLM_MLA_TURBOQUANT_PLAN.md), so
    this gate NEVER accepts a turboquant dtype for an MLA model -- MLA stays
    fp8_inc, always. This is defense-in-depth (a turboquant dtype on an MLA serve
    falls back to the stock 'fp8_inc only' refusal, loud, never silent garbage).

Out-of-tree: apply() saves the ORIGINAL classmethod, installs the wrapper, is
idempotent. A CPU-only import never imports vllm_gaudi (the import is inside apply()).
"""
from __future__ import annotations

import os

# The TurboQuant KV dtypes this package introduces. k8v4 == K fp8 / V int4 (the
# safe knee, default sub-fp8 mode); k8v2 == K fp8 / V int2 (the "lightest" mode,
# quality-risky per quant-kv-turboquant.md -- built + measured, NEVER the default).
KVQ_DTYPES = ("turboquant_k8v4", "turboquant_k8v2", "turboquant_k8v8")

# Set True ONLY when the compiled ops are loaded AND the TurboQuant KV write/read
# path is wired + validated on-card. Until then this patch is a NO-OP. Mirrors
# platform_patch.DSA_READY.
KVQ_READY = False

# Exposed for the gate / tests (echoes whether the store+read wiring is installed).
# The dtype must NOT be accepted unless this is True (else vLLM allocates an
# unwritable pool). register() flips this True after the wiring is installed.
WIRE_INSTALLED = False

_READY_ENV = "HPU_KVQ_READY"

# The original (or previously-chained) unbound is_kv_cache_dtype_supported captured
# at apply() time, so we chain through anything already installed.
_ORIG = None


def ready() -> bool:
    """True iff the operator opted into the (validated) live TurboQuant-KV flip via env."""
    return os.environ.get(_READY_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def _gate_open() -> bool:
    """The dtype is accepted ONLY when the module gate is open AND the env opt-in is
    set AND the store/read wiring is installed -- all three. Defense in depth so a
    stray --kv-cache-dtype turboquant_k8v4 on a serve WITHOUT the kernels loaded
    still falls back to the stock 'fp8_inc only' refusal (loud, never silent garbage)."""
    return KVQ_READY and ready() and WIRE_INSTALLED


def _is_mla(model_config) -> bool:
    """True for an MLA-latent model (GLM-5.x, DeepSeek-family, Kimi-K2-family).
    TurboQuant is dead on MLA -> such a model must stay fp8_inc. Robust: prefer the
    ModelConfig.use_mla flag, else sniff the hf text config for a kv_lora_rank."""
    if getattr(model_config, "use_mla", False):
        return True
    hf = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    lora = getattr(hf, "kv_lora_rank", None)
    return isinstance(lora, int) and lora > 0


def _patched(cls, kv_cache_dtype, model_config) -> bool:
    """is_kv_cache_dtype_supported override: accept the TurboQuant dtypes IFF the gate
    is open AND the model is NOT MLA; otherwise defer to the captured original (the
    stock 'fp8_inc only')."""
    if kv_cache_dtype in KVQ_DTYPES:
        if _is_mla(model_config):
            # MLA stays fp8_inc (measured ~4-5x fp8 error for int4-latent). Never turboquant.
            return False
        if _gate_open():
            return True
        # Gate closed: refuse explicitly (do NOT silently accept an unwired dtype).
        return False
    return _ORIG(cls, kv_cache_dtype, model_config)


def apply() -> None:
    """Install the gate extension. Captures the CURRENT is_kv_cache_dtype_supported as
    the chain-through original, then binds our override. Idempotent. Out-of-tree
    (monkeypatch only; no /opt edit). Mirrors platform_patch.apply()."""
    global _ORIG
    from vllm_gaudi.platform import HpuPlatform

    cur = HpuPlatform.is_kv_cache_dtype_supported
    if getattr(cur, "_hpu_kvq_wired", False):
        return
    _ORIG = cur.__func__ if hasattr(cur, "__func__") else cur
    _patched._hpu_kvq_wired = True
    HpuPlatform.is_kv_cache_dtype_supported = classmethod(_patched)


def maybe_flip_ready() -> bool:
    """Flip KVQ_READY=True IFF HPU_KVQ_READY is truthy AND the write/read wiring is
    installed. Called LAST by register(), only after the compiled ops are loaded and
    the cache-spec/attention seam is bound. Until then (or without the env) the
    TurboQuant dtypes stay REFUSED and fp8_inc remains the only accepted dtype.
    Returns the resulting KVQ_READY value. Mirrors platform_patch_msa.maybe_flip_ready()."""
    global KVQ_READY
    if ready() and WIRE_INSTALLED:
        KVQ_READY = True
    return KVQ_READY


def v_bits_for_dtype(kv_cache_dtype: str) -> int:
    """Map an accepted TurboQuant dtype string to its V quant width (for the ops/runner)."""
    base = {"turboquant_k8v4": 4, "turboquant_k8v2": 2, "turboquant_k8v8": 8}[kv_cache_dtype]
    # HPU_KVQ_VFP8=1: force fp8-V (near-lossless -> full-coherence) on the k8v4 dtype, which my plugin
    # cleanly owns -> avoids the upstream-registry entanglement that broke a real turboquant_k8v8 dtype.
    # ~fp8 capacity (V no longer 2x lean) but restores long-gen coherence; the int4-V path stays default.
    if os.environ.get("HPU_KVQ_VFP8", "0").strip().lower() in ("1", "true", "yes", "on"):
        return 8
    return base
