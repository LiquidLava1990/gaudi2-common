"""GLM-5.1 / DeepSeek-V3.2-V4 wiring onto the HPU DeepSeek Sparse Attention path.

This is the *orchestration glue* for the DSA family. It does three things and only
three things — it does NOT re-implement the indexer, the cache, or the consume side
(those live in the sibling modules and are imported by ``hpu_dsa_ext.register``):

  1. CONFIG-DRIVEN ARCH BINDING (``register_arch``):
       ``GlmMoeDsaForCausalLM`` / ``DeepseekV3ForCausalLM`` / ``DeepseekV2ForCausalLM``
       are ALREADY in vLLM core's registry and ALREADY config-drive their sparse
       behaviour via ``is_v32 = hasattr(config, "index_topk")`` (the in-tree class
       at ``vllm.model_executor.models.deepseek_v2`` —
       ``class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass``). We therefore do
       NOT subclass or shadow them (that would break MoE param extraction +
       AutoWeightsLoader). ``register_arch`` is an *idempotent, defensive*
       re-binding of those arch names to the canonical in-tree class so the names
       survive a future vllm prune; it is a no-op when the registry already resolves
       them. The actual sparse-vs-dense decision stays 100% config-driven inside the
       in-tree class + the in-tree ``MLAModules(is_sparse=is_v32)`` plumbing.

  2. CONFIG VALIDATION (``assert_sparse_config``, ``read_dsa_dims``):
       a fail-loud gate that confirms an hf_config will actually fire the sparse
       path on HPU, and a single helper that reads EVERY DSA dim from the config so
       the rest of the overlay hardcodes nothing (GLM-5.1: index_n_heads=32;
       DeepSeek-V3.2/V4: 64). Honors ``skip_topk`` semantics (index_topk_freq /
       index_topk_pattern) and ``first_k_dense_replace`` (dense layers 0..k-1).

  3. ACTIVATION GATING (``ready()``, ``maybe_flip_ready``):
       the DSA path stays BLOCKED (platform_patch.DSA_READY=False -> explicit
       NotImplementedError, never dense) unless the operator opts in with
       ``HPU_DSA_READY=1``. ``hpu_dsa_ext.register`` calls ``maybe_flip_ready`` LAST,
       after every sibling component (indexer op + K-cache + consume) is registered.

Launch env for GLM-5.1 is documented at the bottom (``LAUNCH_ENV`` / ``launch_help``).

Contract anchors (LOCKED — verifier-resolved):
  * platform.py:206 sets ``compilation_config.custom_ops = ["all"]`` -> the sibling
    ``SparseAttnIndexer.forward_oot`` is auto-enabled and dispatched (HPU is OOT).
  * arch classes are config-driven: ``is_v32 = hasattr(config, "index_topk")``
    (vllm_core_deepseek_v2.py:999 / :1231).
  * dims live on hf_config: index_topk, index_n_heads, index_head_dim,
    qk_rope_head_dim, first_k_dense_replace, indexer_rope_interleave,
    index_topk_freq / index_topk_pattern (vllm_core_deepseek_v2.py:621-624,
    1027-1033, 1145).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Canonical in-tree module that defines the DSA-capable, config-driven arch classes.
# (vLLM core ships these; we re-point the registry names at them defensively.)
_DEEPSEEK_V2_MODULE = "vllm.model_executor.models.deepseek_v2"

# Arch names that share the single DeepseekV2 MLA+Indexer structure (cross-model
# reuse is config-only, verified). Order is irrelevant; all map to the same module.
DSA_ARCHES: tuple[str, ...] = (
    "GlmMoeDsaForCausalLM",      # GLM-5.1
    "DeepseekV3ForCausalLM",     # DeepSeek-V3 / V3.2 / V4 (V3.2+ carry index_topk)
    "DeepseekV2ForCausalLM",     # base (also used directly by some V3.2 configs)
)

# Env switch that gates the live flip. DSA stays BLOCKED (NotImplementedError, never
# dense) until BOTH the sibling stack is registered AND this is truthy.
_READY_ENV = "HPU_DSA_READY"


# --------------------------------------------------------------------------------
# 2) Config validation / dims  (hardcode NOTHING — every value comes from hf_config)
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class DsaDims:
    """Every DSA dimension the overlay needs, read straight from hf_config.

    Used by the indexer op (n_head/head_dim/topk), the K-cache allocator
    (head_dim, NOT the fp8-packed +4/128 default), and the consume side (topk).
    """
    index_topk: int            # GLM/DeepSeek-V3.2 = 2048   (top-k tokens kept)
    index_n_heads: int         # GLM = 32, DeepSeek = 64    (indexer MQA q heads)
    index_head_dim: int        # 128                        (indexer head dim)
    qk_rope_head_dim: int      # 64                         (partial-rope dim on keys)
    first_k_dense_replace: int # 3                          (layers 0..k-1 dense MLP)
    indexer_rope_interleave: bool  # rope interleave flag (is_neox_style = not this)
    index_topk_freq: int           # IndexShare cadence period (shared-layer reuse)
    index_skip_topk_offset: int    # IndexShare cadence offset (GLM-5.2=3; HF default 2)
    num_hidden_layers: int         # total layers (the cadence is defined over all of them)
    indexer_types: tuple[str, ...] | None       # explicit per-layer full/shared (GLM-5.2 SHIPS this)
    index_topk_pattern: tuple[str, ...] | None   # legacy explicit per-layer _/S pattern

    # ---- MLP-dense band (INDEPENDENT of the indexer cadence) ----------------
    def layer_is_dense(self, layer_id: int) -> bool:
        """Layers [0, first_k_dense_replace) use a DENSE MLP (vs MoE).

        NOTE: this is an *MLP* property ONLY. It does NOT decide whether the
        lightning indexer runs — that is the IndexShare cadence
        (``layer_runs_indexer`` / ``indexer_types``), which in GLM-5.2 is defined
        over ALL layers. (GLM-5.2's offset=3 happens to align the seed-full band
        with the 3 dense layers, but the two concepts are still distinct.)
        """
        return layer_id < self.first_k_dense_replace

    # ---- IndexShare cadence (the F/S schedule) ------------------------------
    @staticmethod
    def _is_shared_token(tok: str) -> bool:
        """A per-layer cadence token meaning 'reuse the prior full layer's top-k'.

        Accepts HF's ``full``/``shared`` (GLM-5.2 ``indexer_types``) AND the legacy
        ``_``/``S`` (``index_topk_pattern``). Case-insensitive. Raises LOUD on an
        unrecognized token rather than silently treating it as full.
        """
        t = str(tok).strip().lower()
        if t in ("shared", "s"):
            return True
        if t in ("full", "f", "_", ""):
            return False
        raise ValueError(
            f"[hpu-dsa-ext] unrecognized indexer cadence token {tok!r} "
            "(expected full/shared, F/S, or _/S)."
        )

    def _full_by_freq(self, layer_id: int) -> bool:
        """HF generator (configuration_glm_moe_dsa.py.__post_init__): a layer runs
        its OWN indexer ('full') iff ``max(i - offset + 1, 0) % freq == 0``.

        GLM-5.2 (offset=3, freq=4) -> full at 0,1,2,6,10,...,76 (the 'F F F | S S S F'
        cadence). The leading ``max(.,0)`` clamps the first ``offset-1`` layers to
        argument 0 -> all 'full' (the index-seeding band, per the DSA paper). With the
        HF default offset=2 this reduces to ``max(i-1,0) % freq`` — i.e. exactly the
        prior vllm_core_deepseek_v2.py:1031 formula (so non-5.2 configs are unchanged).
        """
        freq = max(self.index_topk_freq, 1)
        off = self.index_skip_topk_offset
        return (max(layer_id - off + 1, 0) % freq) == 0

    def layer_runs_indexer(self, layer_id: int) -> bool:
        """True iff layer ``layer_id`` computes its OWN top-k (a 'full' layer).

        Authoritative order: explicit ``indexer_types`` (what GLM-5.2 SHIPS and was
        TRAINED with) > legacy ``index_topk_pattern`` > the freq/offset generator.
        Defined over ALL layers (not gated on the MLP-dense band).
        """
        if self.indexer_types is not None:
            if not (0 <= layer_id < len(self.indexer_types)):
                raise IndexError(
                    f"[hpu-dsa-ext] layer_id {layer_id} out of range for indexer_types "
                    f"(len {len(self.indexer_types)})."
                )
            return not self._is_shared_token(self.indexer_types[layer_id])
        if self.index_topk_pattern is not None:
            if 0 <= layer_id < len(self.index_topk_pattern):
                return not self._is_shared_token(self.index_topk_pattern[layer_id])
            return True  # past an explicit pattern -> treat as full (own indexer)
        return self._full_by_freq(layer_id)

    def layer_skips_topk(self, layer_id: int) -> bool:
        """True iff layer ``layer_id`` REUSES the prior full layer's top-k buffer
        (a 'shared' layer; the indexer op is NOT called for it).

        Exactly ``not layer_runs_indexer`` — the IndexShare contract. For GLM-5.1
        (freq=1) every layer is full so this is always False (prior behavior); for
        GLM-5.2 (freq=4/offset=3) it is True on ~57 of 78 layers.
        """
        return not self.layer_runs_indexer(layer_id)

    def indexer_schedule(self) -> tuple[str, ...]:
        """The resolved per-layer cadence as ``full``/``shared`` (len num_hidden_layers)."""
        return tuple(
            "full" if self.layer_runs_indexer(i) else "shared"
            for i in range(self.num_hidden_layers)
        )

    def full_indexer_layers(self) -> tuple[int, ...]:
        """Layer ids that run their own indexer (the 'full' layers)."""
        return tuple(i for i in range(self.num_hidden_layers)
                     if self.layer_runs_indexer(i))

    def reuse_source(self, layer_id: int) -> int:
        """The layer whose top-k buffer ``layer_id`` consumes: itself if 'full', else
        the NEAREST PRECEDING 'full' layer (the IndexShare reuse edge).

        Fails LOUD if a 'shared' layer has no preceding 'full' layer — that would mean
        reusing an unwritten buffer (silent garbage). The seed-band normally makes
        layer 0 'full', so this never trips on a well-formed cadence.
        """
        if self.layer_runs_indexer(layer_id):
            return layer_id
        for j in range(layer_id - 1, -1, -1):
            if self.layer_runs_indexer(j):
                return j
        raise ValueError(
            f"[hpu-dsa-ext] shared layer {layer_id} has no preceding full indexer layer "
            "to reuse — IndexShare would consume an unwritten top-k buffer. Check "
            "indexer_types / index_topk_freq / index_skip_topk_offset."
        )

    def assert_indexshare_wellformed(self) -> None:
        """Fail-loud self-check of the whole cadence (call once at build / in tests).

        Guarantees: (1) every shared layer has a preceding full layer to reuse, and
        (2) the seeding layer 0 is full. Raises ValueError on violation.
        """
        if self.num_hidden_layers <= 0:
            return
        if not self.layer_runs_indexer(0):
            raise ValueError(
                "[hpu-dsa-ext] IndexShare cadence has layer 0 as 'shared' — there is no "
                "buffer to seed it. The seed-band (max(i-offset+1,0)) should force layer 0 "
                "full; check index_skip_topk_offset / indexer_types."
            )
        for i in range(self.num_hidden_layers):
            self.reuse_source(i)  # raises if a shared layer lacks a preceding full


def config_is_sparse(hf_config) -> bool:
    """The single source of truth vLLM itself uses: a DSA config has ``index_topk``.

    Matches ``is_v32 = hasattr(config, "index_topk")`` (vllm_core_deepseek_v2.py:999).
    """
    return hasattr(hf_config, "index_topk") and getattr(hf_config, "index_topk", None)


def read_dsa_dims(hf_config) -> DsaDims:
    """Read all DSA dims from hf_config (raises if the config is not actually DSA).

    Hardcodes nothing: GLM-5.1 yields index_n_heads=32; DeepSeek-V3.2/V4 yields 64.
    """
    if not config_is_sparse(hf_config):
        raise ValueError(
            "[hpu-dsa-ext] read_dsa_dims called on a non-DSA config "
            "(no truthy `index_topk`); this model does not use sparse attention."
        )
    missing = [
        a for a in ("index_topk", "index_n_heads", "index_head_dim",
                    "qk_rope_head_dim", "first_k_dense_replace")
        if not hasattr(hf_config, a)
    ]
    if missing:
        raise ValueError(
            "[hpu-dsa-ext] DSA config is missing required field(s): "
            + ", ".join(missing)
            + ". Refusing to build a sparse layer with assumed dims."
        )
    pattern = getattr(hf_config, "index_topk_pattern", None)
    itypes = getattr(hf_config, "indexer_types", None)
    return DsaDims(
        index_topk=int(hf_config.index_topk),
        index_n_heads=int(hf_config.index_n_heads),
        index_head_dim=int(hf_config.index_head_dim),
        qk_rope_head_dim=int(hf_config.qk_rope_head_dim),
        first_k_dense_replace=int(hf_config.first_k_dense_replace),
        indexer_rope_interleave=bool(getattr(hf_config, "indexer_rope_interleave", False)),
        index_topk_freq=int(getattr(hf_config, "index_topk_freq", 1)),
        # HF default offset is 2 (-> the legacy max(i-1,0)%freq formula); GLM-5.2 ships 3.
        index_skip_topk_offset=int(getattr(hf_config, "index_skip_topk_offset", 2)),
        num_hidden_layers=int(getattr(hf_config, "num_hidden_layers", 0)),
        indexer_types=tuple(itypes) if itypes is not None else None,
        index_topk_pattern=tuple(pattern) if pattern is not None else None,
    )


def assert_sparse_config(hf_config) -> DsaDims:
    """Fail-loud gate: confirm this config WILL fire the HPU sparse path as designed.

    Returns the parsed dims on success. Intended to be called once at model-build
    time (and by tests) so a misconfigured DSA checkpoint fails immediately with a
    clear message instead of silently degrading. Does NOT touch HPU.
    """
    dims = read_dsa_dims(hf_config)
    if dims.index_topk <= 0:
        raise ValueError(f"[hpu-dsa-ext] index_topk must be > 0, got {dims.index_topk}")
    if dims.index_n_heads <= 0 or dims.index_head_dim <= 0:
        raise ValueError(
            "[hpu-dsa-ext] index_n_heads / index_head_dim must be > 0, got "
            f"{dims.index_n_heads} / {dims.index_head_dim}"
        )
    if dims.first_k_dense_replace < 0:
        raise ValueError(
            f"[hpu-dsa-ext] first_k_dense_replace must be >= 0, got "
            f"{dims.first_k_dense_replace}"
        )
    return dims


# --------------------------------------------------------------------------------
# 1) Config-driven arch binding (idempotent / defensive — no subclassing)
# --------------------------------------------------------------------------------
def register_arch() -> list[str]:
    """Defensively (re)bind the DSA arch names to the canonical in-tree class.

    The in-tree ``DeepseekV2ForCausalLM`` subclasses already config-drive sparse via
    ``is_v32``; we re-point the registry names at them so a future vllm that prunes
    a name still resolves on HPU. This is a NO-OP overlay (same class) and never
    shadows the working MoE/weight-loading path. Returns the list of names bound.

    We import the in-tree classes to validate they exist + still subclass the base
    (loud on drift), but register by ``"module:Class"`` string so vLLM lazy-loads as
    usual. The actual config gate (``index_topk``) is enforced by ``config_is_sparse``
    / ``read_dsa_dims`` (unit-tested), not by source introspection here.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    # Loud drift-guard (structural, version-robust): the in-tree module must define
    # each DSA arch class, and each must subclass DeepseekV2ForCausalLM (the shared
    # MLA+Indexer structure that config-drives sparse via is_v32 / index_topk).
    import importlib
    mod = importlib.import_module(_DEEPSEEK_V2_MODULE)
    base = getattr(mod, "DeepseekV2ForCausalLM", None)
    if base is None:
        raise RuntimeError(
            f"[hpu-dsa-ext] {_DEEPSEEK_V2_MODULE}.DeepseekV2ForCausalLM is gone; "
            "the DSA base class moved/renamed upstream — review glm_moe_dsa_wire."
        )
    for arch in DSA_ARCHES:
        cls = getattr(mod, arch, None)
        if cls is None:
            raise RuntimeError(
                f"[hpu-dsa-ext] {_DEEPSEEK_V2_MODULE} no longer defines {arch}; "
                "the DSA arch classes moved/renamed upstream — review glm_moe_dsa_wire."
            )
        if not (isinstance(cls, type) and issubclass(cls, base)):
            raise RuntimeError(
                f"[hpu-dsa-ext] {arch} no longer subclasses DeepseekV2ForCausalLM; "
                "the shared DSA structure changed upstream — review glm_moe_dsa_wire."
            )

    bound: list[str] = []
    for arch in DSA_ARCHES:
        ModelRegistry.register_model(arch, f"{_DEEPSEEK_V2_MODULE}:{arch}")
        bound.append(arch)
    return bound


# --------------------------------------------------------------------------------
# 3) Activation gating
# --------------------------------------------------------------------------------
def ready() -> bool:
    """True iff the operator opted into the (validated) live DSA flip via env."""
    return os.environ.get(_READY_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def maybe_flip_ready() -> bool:
    """Flip ``platform_patch.DSA_READY=True`` IFF ``HPU_DSA_READY`` is truthy.

    Called LAST by ``hpu_dsa_ext.register`` — only after the indexer op, the K-cache
    hook, and the consume side are all registered. Until then (or without the env)
    the platform patch keeps refusing sparse with a clear message (NEVER dense).
    Returns the resulting DSA_READY value.
    """
    from . import platform_patch
    if ready():
        platform_patch.DSA_READY = True
    return platform_patch.DSA_READY


# --------------------------------------------------------------------------------
# Launch documentation — GLM-5.1 on 8x Gaudi (TP8 + EP, AWQ/GPTQ, no prefix cache)
# --------------------------------------------------------------------------------
# Env that MUST be exported, and the serve flags. Kept as data so a launcher script
# (or a test) can import and assert on them rather than copy-pasting strings.
LAUNCH_ENV: dict[str, str] = {
    # PT_HPU_LAZY_MODE=0 is REQUIRED: the indexer/consume memory bounding uses
    # M<=128 tiles + explicit `del` (mark_step is a no-op in eager mode); the whole
    # DSA design is validated under eager. Do NOT set lazy mode.
    "PT_HPU_LAZY_MODE": "0",
    # Opt into the live sparse flip (else DSA stays BLOCKED, never dense).
    "HPU_DSA_READY": "1",
    # Indexer K-cache is bf16/head_dim=128/flat and is allocated by the overlay's
    # runner hook (NOT the fp8-packed default). Contiguous-PA disables prefix cache
    # anyway; we ALSO forbid prefix caching explicitly below — the indexer K-cache /
    # topk buffer derivation assumes no cross-request prefix reuse.
}

# vLLM serve flags (argv). EP + TP8 across the 8 cards; sparse is config-driven so
# no extra "enable sparse" flag exists. Prefix caching MUST be off.
LAUNCH_FLAGS: tuple[str, ...] = (
    "--tensor-parallel-size", "8",
    "--enable-expert-parallel",
    "--no-enable-prefix-caching",   # indexer K-cache + topk buffers assume no reuse
    "--block-size", "128",
    "--max-model-len", "32768",     # phase-1 target; 200K needs the TPC kernel
    # Quantization: awq_hpu (gated on hpu-moe-ext/tests/awq_numeric_proof.py) or gptq
    # (AutoRound checkpoint). Pass --quantization awq_hpu OR gptq as appropriate:
    #   "--quantization", "awq_hpu",
)


def launch_help() -> str:
    """Human-readable launch recipe for GLM-5.1 sparse-as-designed on 8x Gaudi 2."""
    env = " ".join(f"{k}={v}" for k, v in LAUNCH_ENV.items())
    flags = " ".join(LAUNCH_FLAGS)
    return (
        "GLM-5.1 (GlmMoeDsa) — sparse DSA on 8x Gaudi 2:\n"
        f"  {env} \\\n"
        f"  vllm serve <GLM-5.1-checkpoint> {flags} \\\n"
        "    --quantization awq_hpu   # or: gptq (AutoRound fallback checkpoint)\n"
        "\n"
        "Notes:\n"
        "  * Sparse is CONFIG-DRIVEN (config has index_topk) — no sparse flag needed.\n"
        "  * DSA stays BLOCKED (NotImplementedError, never dense) unless HPU_DSA_READY=1\n"
        "    AND the full sibling stack (indexer op + K-cache + consume) is registered.\n"
        "  * Prefix caching MUST be off (indexer K-cache / topk buffers assume no reuse).\n"
        "  * 32K phase-1 target (gather-consume). 200K needs a TPC kernel (out of scope).\n"
        "  * Cross-model: DeepSeek-V3.2/V4 reuse this same path; only the config dims\n"
        "    differ (index_n_heads 64 vs GLM 32) — read_dsa_dims handles both.\n"
    )


# Convenience: a single call the orchestrator can make to bind arches. Safe/idempotent.
def register() -> dict:
    """Bind the DSA arch names (config-driven, defensive). Returns a small summary.

    Does NOT flip DSA_READY (that is ``maybe_flip_ready``, called last by the
    package ``register``). Does NOT touch HPU. Idempotent.
    """
    bound = register_arch()
    return {"bound_arches": bound, "dsa_ready_env": _READY_ENV, "ready": ready()}
