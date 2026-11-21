"""GLM-5.1 RoPE fixes — TWO independent gated patches in one module:

  (A) CONVENTION  (HPU_GLM_ROPE_NEOX=1): force is_neox_style=True for the
      deepseek_v2 rope construction. GLM-5.1 uses NeoX rotate-half RoPE; the
      DeepSeek MLA path hardcodes is_neox_style=False (GPT-J interleaved). This is
      the SEPARATE, proven-coherent EAGER fix. UNCHANGED below.

  (B) LAZY-CAPTURE LIVE COS/SIN  (HPU_GLM_ROPE_INLINE, default ON whenever (A) is
      on AND lazy): rewrite HPURotaryEmbedding.forward_oot to derive cos/sin INLINE
      from its own `positions` ARGUMENT and apply the rotation in the SAME call,
      bypassing the prepare_cos_sin -> register_buffer("cos"/"sin") -> self.cos/
      self.sin handoff. Fixes the decode-graph "position frozen at step 0" loop
      ("Paris. The capital of France is Paris...", "The answer is: The answer is:")
      that survives VLLM_COS_SIN_RECOMPUTE under lazy HPU-graph capture.

ROOT CAUSE OF (B) — established by reading the capture machinery, not a HW probe
(a .item() probe would itself break capture and mask the bug):

  * Lazy serve wraps HpuModelAdapter in
    `htorch.hpu.wrap_in_hpu_graph(disable_tensor_cache=True)`
    (hpu_model_runner.py:837 via _maybe_wrap_in_hpu_graph).
  * HpuModelAdapter.forward (hpu_model_runner.py:780-781) calls
    `self._rotary_prepare_cos_sin(kwargs['positions'], recompute_cos_sin=...)`
    BEFORE `self.model(...)` (L802), then `_reset_rotary_cos_sin()` delattrs
    cos/sin AFTER (L804). There is NO htcore.mark_step() between the pre-call and
    the model body, so prepare_cos_sin's `cos_sin_cache.index_select(0, positions)`
    is captured INSIDE the same decode HPU graph (the cos/sin python buffers are
    in-graph intermediates, NOT a cross-graph python attribute — so the literal
    "register_buffer handle is frozen" story is not the mechanism).
  * The EFFECTIVE freeze is one level deeper and is what (B) removes: under
    `disable_tensor_cache=True` the graph is replayed (graphs.py
    wrapped_hpugraph_forward, cache-hit branch L759-764). The gathered cos/sin rows
    advance on replay ONLY if the gather's index tensor (`positions`) is carried
    live — i.e. either spliced by `replayV3` when it is in
    `get_user_input_match_indices()`, or `copy_to`'d into the capture-time storage
    the captured gather reads from. The current path SPLITS the gather index (the
    ADAPTER's `kwargs['positions']`, consumed by the pre-call prepare_cos_sin) from
    the rope-apply consumer (forward_oot reads self.cos/self.sin, and with
    recompute_cos_sin=False does NOT re-gather from its OWN deeper `positions`
    argument). That split + the register_buffer/delattr indirection is exactly the
    shape most exposed to the matcher dropping the gather's liveness, leaving cos/
    sin pinned to the FIRST captured decode position -> every decode step applies
    the same rope angle -> attention cannot tell positions apart -> looping. Eager
    has no capture, so eager stays correct (and is left on the original path).

WHY (B) IS ROBUST: forward_oot recomputes `cos,sin = cos_sin_cache.index_select(0,
positions)` from ITS OWN `positions` argument and consumes them immediately in the
same function. The gather index and its consumer become ONE fused data-dependency
chain rooted in the very tensor the attention op already receives and depends on —
no buffer, no pre-call, no delattr, no adapter/model positions split. The rotation
math is replicated BIT-FOR-BIT from the original HPURotaryEmbedding.forward_oot /
prepare_cos_sin (same flatten, index_select, chunk, neox cat / gptj repeat_interleave
expansion, same apply_rotary_pos_emb(p, cos, sin, None, 0, mode) BLOCKWISE/PAIRWISE
kernel call, same head_size==rotary_dim fast path and the rotary_dim slice slow
path, same offset handling), so numerics are identical to the stock op — (B) only
changes WHERE/WHEN cos/sin are produced, not the values.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")


# ----------------------------------------------------------------------------
# (A) Convention fix — force is_neox_style=True in the deepseek_v2 rope builder.
# ----------------------------------------------------------------------------
def _install_neox_flip() -> None:
    try:
        import vllm.model_executor.models.deepseek_v2 as dsv2
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-rope-neox] cannot import deepseek_v2: %s", ex)
        return
    if getattr(getattr(dsv2, "get_rope", None), "_glm_rope_neox", False):
        return
    _orig = dsv2.get_rope

    def get_rope(*args, **kwargs):  # noqa: ANN001
        # GLM-5.1 needs NeoX (rotate-half). The deepseek path passes is_neox_style=False
        # (interleaved); flip it. Only flip explicit False -> True; leave True/unset.
        if kwargs.get("is_neox_style") is False:
            kwargs["is_neox_style"] = True
        return _orig(*args, **kwargs)

    get_rope._glm_rope_neox = True
    dsv2.get_rope = get_rope
    log.warning("[glm-rope-neox] forced is_neox_style=True for deepseek_v2.get_rope "
                "(GLM-5.1 uses NeoX rotate-half RoPE; config rope_interleave is a dead field)")


# ----------------------------------------------------------------------------
# (B) Lazy-capture live cos/sin — inline forward_oot that gathers cos/sin from its
#     OWN positions argument (graph op driven by the replayed input; cannot freeze).
# ----------------------------------------------------------------------------
def _install_inline_forward_oot() -> None:
    try:
        from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-rope-neox] cannot import HPURotaryEmbedding; inline cos/sin "
                    "fix NOT installed: %s", ex)
        return

    if getattr(HPURotaryEmbedding.forward_oot, "_glm_rope_inline", False):
        return

    import torch

    def forward_oot(
        self,
        positions: "torch.Tensor",
        query: "torch.Tensor",
        key: "torch.Tensor",
        offsets=None,
    ):  # noqa: ANN001
        # Bit-for-bit equivalent of the stock HPURotaryEmbedding.prepare_cos_sin +
        # forward_oot (vllm_gaudi/ops/hpu_rotary_embedding.py:17-85), FUSED so cos/sin
        # are derived from THIS call's `positions` argument and consumed immediately —
        # no register_buffer("cos"/"sin"), no adapter pre-call handoff, no delattr.
        # That makes the position->gather->rope chain a single live graph dependency
        # rooted in the replayed `positions` input, so it cannot freeze at the
        # capture-time (step-0) decode position under disable_tensor_cache replay.
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)

        # ---- prepare_cos_sin (inline, offsets handled exactly as the original) ----
        pos = positions
        if offsets is not None:
            pos = pos + offsets.view(pos.shape[0], -1)
        pos = pos.flatten()
        num_tokens = pos.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, pos).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # GPT-NeoX kernel (BLOCKWISE): expand via concatenation.
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            # GPT-J kernel (PAIRWISE): expand via repeat_interleave.
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])

        rope_mode: "RotaryPosEmbeddingMode"
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style \
            else RotaryPosEmbeddingMode.PAIRWISE

        # ---- forward_oot rotation (inline, identical to the stock op) ----
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        if self.head_size == self.rotary_dim:
            # Avoid unnecessary slicing and concatenation (GLM-5.1 takes this path:
            # qk_rope_head_dim == rotary_dim == head_size == 64).
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)

        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    forward_oot._glm_rope_inline = True
    HPURotaryEmbedding.forward_oot = forward_oot
    log.warning("[glm-rope-neox] installed INLINE forward_oot on HPURotaryEmbedding "
                "(live per-step cos/sin from positions arg; fixes lazy-capture decode "
                "rope freeze / looping). HPU_GLM_ROPE_INLINE")


def _make_inline_forward_oot(cache_attr: str):
    """Factory: build an inline forward_oot that gathers cos/sin from ``self.<cache_attr>``
    using THIS call's ``positions`` argument (the live, replayed input) — identical math to
    the stock op, only fused so the gather index and its consumer are ONE live dependency.
    ``cache_attr`` is the per-class cache field (``cos_sin_cache`` for all the scaling
    classes; ``long_short_cos_sin_cache`` for Phi3-LongRoPE)."""
    import torch

    def forward_oot(self, positions, query, key, offsets=None):  # noqa: ANN001
        from habana_frameworks.torch.hpex.kernels import (RotaryPosEmbeddingMode, apply_rotary_pos_emb)
        cache = getattr(self, cache_attr)
        pos = positions
        if offsets is not None:
            pos = pos + offsets.view(pos.shape[0], -1)
        pos = pos.flatten()
        num_tokens = pos.shape[0]
        cos_sin = cache.index_select(0, pos).view(num_tokens, 1, -1)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        else:
            sin = torch.repeat_interleave(sin, 2, dim=-1, output_size=cos_sin.shape[-1])
            cos = torch.repeat_interleave(cos, 2, dim=-1, output_size=cos_sin.shape[-1])
        rope_mode = RotaryPosEmbeddingMode.BLOCKWISE if self.is_neox_style \
            else RotaryPosEmbeddingMode.PAIRWISE
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)
        if self.head_size == self.rotary_dim:
            query = apply_rotary_pos_emb(query, cos, sin, None, 0, rope_mode)
            key = apply_rotary_pos_emb(key, cos, sin, None, 0, rope_mode)
            return query.reshape(query_shape), key.reshape(key_shape)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    forward_oot._glm_rope_inline = True
    return forward_oot


def _install_inline_forward_oot_subclasses() -> None:
    """DEFENSIVE BROADEN (gated HPU_GLM_ROPE_INLINE_ALL, default ON whenever the base inline
    fix is on): apply the SAME live-cos/sin inline forward_oot to every HPU RotaryEmbedding
    SUBCLASS that uses the ``cos_sin_cache.index_select(0, positions)`` pattern, not just the
    base class.

    WHY: the base-class inline fix only overrides ``HPURotaryEmbedding.forward_oot``. A model
    whose MLA rope resolves to a SCALING subclass (HPUDeepseekScalingRotaryEmbedding,
    YaRN/NTK/Linear/Llama3, Phi3-LongRoPE) keeps the subclass's OWN forward_oot, which still
    routes through prepare_cos_sin -> register_buffer("cos"/"sin") -> self.cos/self.sin — the
    EXACT split-index handoff that freezes the gather at the step-0 decode position under lazy
    HPU-graph replay (disable_tensor_cache=True). GLM-5.1 as-configured uses the BASE class
    (rope_type="default"), so this is belt-and-suspenders for it; but it closes the freeze for
    any DeepSeek-family MLA model that DOES take a scaling subclass (e.g. yarn-scaled V3.2),
    and guards against an upstream change routing GLM onto a subclass. Math is bit-identical to
    the stock op (same flatten/index_select/chunk/neox-vs-gptj expansion/apply_rotary_pos_emb,
    same head_size==rotary_dim fast path), so eager numerics are untouched (eager never installs
    this; it's lazy-only like the base fix)."""
    try:
        from vllm_gaudi.ops import hpu_rotary_embedding as rope_mod
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-rope-neox] cannot import hpu_rotary_embedding for subclass broaden: %s", ex)
        return

    # (class_name, cache_field). Every scaling subclass shares cos_sin_cache; Phi3 uses
    # long_short_cos_sin_cache. We patch by introspection so a missing class is skipped, not fatal.
    targets = [
        ("HPULinearScalingRotaryEmbedding", "cos_sin_cache"),
        ("HPUDynamicNTKScalingRotaryEmbedding", "cos_sin_cache"),
        ("HPUYaRNScalingRotaryEmbedding", "cos_sin_cache"),
        ("HPUDeepseekScalingRotaryEmbedding", "cos_sin_cache"),
        ("HPULlama3RotaryEmbedding", "cos_sin_cache"),
        ("HPUPhi3LongRoPEScaledRotaryEmbedding", "long_short_cos_sin_cache"),
    ]
    patched = []
    for cls_name, cache_attr in targets:
        cls = getattr(rope_mod, cls_name, None)
        if cls is None:
            continue
        if getattr(cls.forward_oot, "_glm_rope_inline", False):
            continue
        cls.forward_oot = _make_inline_forward_oot(cache_attr)
        patched.append(cls_name)
    if patched:
        log.warning("[glm-rope-neox] installed INLINE forward_oot on rope SUBCLASSES %s "
                    "(live per-step cos/sin; closes the capture-freeze for scaling-rope MLA "
                    "models). HPU_GLM_ROPE_INLINE_ALL", patched)


def install() -> None:
    if os.environ.get("HPU_GLM_ROPE_NEOX", "0") != "1":
        return

    # (A) convention flip — always when HPU_GLM_ROPE_NEOX=1 (proven eager + lazy).
    _install_neox_flip()

    # (B) inline live cos/sin — needed ONLY under lazy HPU-graph capture (eager has no
    #     capture and stays on the stock path, preserving the proven-coherent eager
    #     numerics). Default ON when lazy; HPU_GLM_ROPE_INLINE overrides (1/0) for A/B.
    try:
        import habana_frameworks.torch as htorch
        _is_lazy = bool(htorch.utils.internal.is_lazy())
    except Exception:  # noqa: BLE001
        _is_lazy = False
    _inline_env = os.environ.get("HPU_GLM_ROPE_INLINE")
    if _inline_env is not None:
        _inline = _inline_env.strip().lower() in ("1", "true", "yes", "on")
    else:
        _inline = _is_lazy  # default: on under lazy, off under eager
    if _inline:
        _install_inline_forward_oot()
        # Defensive broaden to scaling subclasses (default ON with the base inline fix). A
        # DeepSeek-family MLA model whose rope resolves to a scaling subclass would otherwise
        # keep the frozen self.cos/self.sin path; this closes that. GLM uses the base class,
        # so for GLM it is a no-op safety net. HPU_GLM_ROPE_INLINE_ALL=0 disables it.
        _all_env = os.environ.get("HPU_GLM_ROPE_INLINE_ALL")
        _all = True if _all_env is None else _all_env.strip().lower() in ("1", "true", "yes", "on")
        if _all:
            _install_inline_forward_oot_subclasses()
    else:
        log.warning("[glm-rope-neox] inline forward_oot NOT installed "
                    "(lazy=%s, HPU_GLM_ROPE_INLINE=%s) — using stock prepare_cos_sin path",
                    _is_lazy, _inline_env)
