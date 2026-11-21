"""LAYER-PRUNE — shorten the device-serial dependency chain for GLM-5.2 single-stream.

The confirmed GLM-5.2 single-stream wall is the DEVICE-SERIAL DECODE CHAIN: every token
walks all 78 decoder layers in strict sequence (layer N+1 depends on layer N's residual),
and at batch=1 the host can never overlap them, so the per-token latency is ~Σ(per-layer
serial cost). Executing FEWER layers per token shortens that chain ~linearly. This is the
DSA-sparsity philosophy (skip redundant *compute*) extended from the attention KV to the
layer stack / the FFN.

(NOTE: an older "layer-skip is dead" verdict was confounded by the OLD fence-count root
cause — removing a layer didn't remove enough mark_step fences to matter. That root cause
is gone; against the *device-serial-chain* wall, removing a whole layer removes a whole
serial segment, which is exactly the lever.)

TWO env-gated, mutually-independent options (BOTH default OFF -> this module is a pure no-op
and the stock vLLM forwards run UNTOUCHED, byte-for-byte):

  1. HPU_SKIP_LAYERS=<comma/range list>   (e.g. "30-37" or "30-37,40,42-45")
       WHOLE-LAYER skip. In DeepseekV2Model.forward's decoder loop, the listed global layer
       indices are passed over entirely (the residual stream flows straight through:
       `if idx in skip: continue`). Attention + FFN + both norms of that layer are all
       bypassed. Biggest step-reduction per skipped layer, biggest quality risk.

  2. HPU_SKIP_MOE_LAYERS=<comma/range list>   (safer)
       MoE-ONLY skip. In DeepseekV2DecoderLayer.forward the expert-FFN (self.mlp, a
       DeepseekV2MoE) is replaced by IDENTITY-ON-RESIDUAL: attention, the input/post-attn
       RMSNorms, and the residual add all still run (so the layer still ATTENDS); only the
       expert FFN's contribution to the residual stream is zeroed. Smaller step-reduction
       per layer (attention stays), but far gentler on coherence.

PRECEDENCE when both envs are set: a whole-layer skip is a SUPERSET of a MoE skip, so if the same
layer index appears in both lists the whole-layer `continue` wins (that layer's MoE-skip forward
never runs). Intersecting the two lists is therefore redundant; use them as disjoint sets (or just
one at a time) during a sweep so the two levers stay separable.

WHY THE PATCH TARGETS ARE REACHED (and why a monkeypatch of .forward is honored):
  vllm_gaudi/platform.py forces compilation_config.mode = CompilationMode.NONE on HPU, so
  vLLM's @support_torch_compile wrapper (on DeepseekV2Model) sets do_not_compile=True and its
  __call__ does exactly `return self.forward(*args, **kwargs)` — a normal attribute lookup.
  GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM) -> self.model is DeepseekV2Model, whose
  .forward owns the decoder loop, and each layer is a DeepseekV2DecoderLayer whose .forward
  owns the `hidden_states = self.mlp(hidden_states)` line. Patching those two class methods
  is therefore the correct, image-rebuild-proof insertion point (it edits NO file under
  /opt/vllm — same contract as every other graft in this plugin).

GUARDRAILS (fail LOUD — DSA build quality bar; never silently corrupt the residual):
  * dense layers [0, first_k_dense_replace)  — refused (they're the seed dense MLP band;
        whole-skip removes attention you can't afford, MoE-skip is a no-op since their mlp is
        a DeepseekV2MLP not a MoE).
  * EAGLE-3 aux-capture layers (2, L//2, L-3) — refused (a draft head taps their hidden
        state; skipping them silently corrupts speculative drafting).
  * any index >= num_hidden_layers (the MTP layer-78 etc.) — refused.
  * an unparseable / empty / out-of-range token — RuntimeError at register, not a wrong answer.

BYTE-EXACTNESS contract:
  install() returns immediately when BOTH envs are unset -> the stock methods are never
  reassigned -> the served graph is identical to stock. When set, the patched whole-layer
  loop is a line-for-line copy of the upstream loop with ONE added `if idx in skip: continue`,
  and the MoE-skip returns torch.zeros_like(<mlp input>) for the FFN delta (the residual add
  in the next layer then adds 0, i.e. the FFN computed nothing) — no other value changes.

The largest K (count of skipped layers) at which the on-card COHERENCE GATE still holds is
the usable prune; K_coherent x (single-layer step fraction) is the single-stream gain.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm")

_ENV_WHOLE = "HPU_SKIP_LAYERS"
_ENV_MOE = "HPU_SKIP_MOE_LAYERS"


# ==================================================================================================
# PURE parsing core (card-free testable; no torch / no vllm)
# ==================================================================================================
def parse_layer_list(spec: str) -> set[int]:
    """Parse a comma/range list of non-negative ints into a set.

    Accepts e.g. "30-37", "30-37,40,42-45", " 5 , 7 ". Whitespace tolerant.
    Raises ValueError (LOUD) on any malformed token, a reversed range, or a negative — we
    never want a typo to silently skip the wrong (or no) layers.
    """
    out: set[int] = set()
    if spec is None:
        return out
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, _, hi_s = tok.partition("-")
            lo_s, hi_s = lo_s.strip(), hi_s.strip()
            if not (lo_s.isdigit() and hi_s.isdigit()):
                raise ValueError(f"bad range token {tok!r} in {spec!r} (want '<int>-<int>')")
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"reversed range {tok!r} in {spec!r} (hi < lo)")
            out.update(range(lo, hi + 1))
        else:
            if not tok.isdigit():
                raise ValueError(f"bad token {tok!r} in {spec!r} (want a non-negative int)")
            out.add(int(tok))
    return out


def forbidden_layers(num_hidden_layers: int, first_k_dense_replace: int) -> set[int]:
    """Indices that must NEVER be skipped:
      * layer 0 ALWAYS — it is the residual SEED (residual starts None; layer 0 sets it). Skipping
        it leaves residual=None for the next layer's aux capture (`hidden_states + residual`) and
        breaks the stream. This holds even if first_k_dense_replace==0 (general DeepseekV2 safety).
      * the dense band [0, first_k_dense_replace) — the seed dense-MLP layers.
      * the EAGLE-3 aux taps (2, L//2, L-3) — mirrors
        DeepseekV2ForCausalLM.get_eagle3_aux_hidden_state_layers() (a draft head reads them).
    """
    forbidden: set[int] = {0}
    forbidden.update(range(max(0, int(first_k_dense_replace))))
    L = int(num_hidden_layers)
    if L >= 3:
        forbidden.update({2, L // 2, L - 3})
    return forbidden


def validate_skip(skip: set[int], num_hidden_layers: int, first_k_dense_replace: int,
                  which: str) -> set[int]:
    """Return the validated skip set or raise RuntimeError (LOUD) describing exactly what's
    wrong. `which` is a human label for the error message ("HPU_SKIP_LAYERS" / ...)."""
    L = int(num_hidden_layers)
    oob = {i for i in skip if i < 0 or i >= L}
    if oob:
        raise RuntimeError(
            f"[hpu-moe-ext] {which}: layer index(es) {sorted(oob)} out of range "
            f"[0, {L}) (num_hidden_layers={L}; the MTP layer {L} and beyond are not skippable here).")
    forb = forbidden_layers(L, first_k_dense_replace) & skip
    if forb:
        raise RuntimeError(
            f"[hpu-moe-ext] {which}: refusing to skip protected layer(s) {sorted(forb)} "
            f"— dense band [0, {first_k_dense_replace}) and EAGLE-3 aux taps "
            f"(2, {L // 2}, {L - 3}) must run. Choose a middle block, e.g. 30-37.")
    return skip


# ==================================================================================================
# INSTALL — monkeypatch DeepseekV2Model.forward (whole-layer) and
#           DeepseekV2DecoderLayer.forward (MoE-only). No-op (stock) when both envs unset.
# ==================================================================================================
def install() -> None:
    whole_spec = os.environ.get(_ENV_WHOLE, "").strip()
    moe_spec = os.environ.get(_ENV_MOE, "").strip()
    if not whole_spec and not moe_spec:
        return  # BYTE-EXACT: neither env set -> touch nothing; stock forwards run unchanged.

    try:
        from vllm.model_executor.models import deepseek_v2 as _dv2
    except Exception as ex:  # noqa: BLE001
        # This plugin is GLM/DeepSeek-family only; if that module is absent there is nothing
        # to prune. Fail loud — the operator asked for a prune and we can't honor it.
        raise RuntimeError(
            f"[hpu-moe-ext] {_ENV_WHOLE}/{_ENV_MOE} set but "
            f"vllm.model_executor.models.deepseek_v2 is unavailable: {ex!r}")

    # Parse the raw specs once (validation against the real config dims happens lazily on the
    # first forward, when hf_config is bound — see _resolve_* below).
    try:
        whole_raw = parse_layer_list(whole_spec)
        moe_raw = parse_layer_list(moe_spec)
    except ValueError as ex:
        raise RuntimeError(f"[hpu-moe-ext] layer-prune spec parse error: {ex}") from ex

    if whole_raw:
        _install_whole_layer_skip(_dv2, whole_raw)
    if moe_raw:
        _install_moe_only_skip(_dv2, moe_raw)


def _install_whole_layer_skip(_dv2, whole_raw: set[int]) -> None:
    Model = _dv2.DeepseekV2Model
    if getattr(Model.forward, "_hpu_layer_prune", False):
        return
    from itertools import islice
    from vllm.distributed import get_pp_group
    from vllm.sequence import IntermediateTensors

    _orig_forward = Model.forward
    # Resolved once on first forward against the live hf_config (loud-validated then cached).
    _state: dict = {"skip": None}

    def _resolve(self) -> set[int]:
        if _state["skip"] is None:
            cfg = self.config
            _state["skip"] = validate_skip(
                set(whole_raw),
                int(cfg.num_hidden_layers),
                int(getattr(cfg, "first_k_dense_replace", 0)),
                _ENV_WHOLE)
            logger.warning(
                "[hpu-moe-ext] %s ACTIVE — whole-layer skip set=%s (%d of %d layers); "
                "residual passes straight through. ON-CARD COHERENCE GATE REQUIRED.",
                _ENV_WHOLE, sorted(_state["skip"]), len(_state["skip"]),
                int(cfg.num_hidden_layers))
        return _state["skip"]

    # This is a LINE-FOR-LINE copy of upstream DeepseekV2Model.forward (the @support_torch_compile
    # do_not_compile path calls self.forward directly on HPU) with ONE added guard:
    #     if idx in skip: continue
    # Everything else — embed, llama_4_scaling, aux_hidden_state capture, the final norm, the
    # PP intermediate-tensors return — is identical to stock.
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        skip = _resolve(self)
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to DeepseekV2Model.forward")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        if llama_4_scaling_config is not None:
            llama_4_scaling = _dv2._get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            if idx in skip:
                continue  # LAYER-PRUNE: residual stream flows through untouched.
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    forward._hpu_layer_prune = True
    forward._hpu_orig_forward = _orig_forward
    Model.forward = forward
    logger.info("[hpu-moe-ext] patched DeepseekV2Model.forward (HPU_SKIP_LAYERS whole-layer skip)")


def _install_moe_only_skip(_dv2, moe_raw: set[int]) -> None:
    Layer = _dv2.DeepseekV2DecoderLayer
    if getattr(Layer.forward, "_hpu_moe_prune", False):
        return
    import torch
    DeepseekV2MLP = _dv2.DeepseekV2MLP
    DeepseekV2MoE = _dv2.DeepseekV2MoE

    _orig_forward = Layer.forward
    # Resolve+validate the skip set ONCE, at the FIRST DecoderLayer construction. vllm_config is
    # passed directly into __init__ (and set_current_vllm_config is active during model build), so
    # hf_config is reliably available HERE — unlike at forward time, where get_current_vllm_config()
    # would raise (its contextmanager has already exited). Stamp the validated set onto a module
    # holder so every layer's forward reads a plain set (no per-step config lookup).
    _state: dict = {"skip": None, "logged": False}

    def _resolve_from_vllm_config(vllm_config) -> set[int]:
        hf = vllm_config.model_config.hf_config
        s = validate_skip(
            set(moe_raw),
            int(hf.num_hidden_layers),
            int(getattr(hf, "first_k_dense_replace", 0)),
            _ENV_MOE)
        return s

    # Wrap __init__ ONLY if not already wrapped, and capture _orig_init INSIDE the guard so a
    # second call can never stack a wrapped __init__ onto itself (the register() _REGISTERED guard +
    # the forward guard above already prevent re-entry in practice; this makes the two wraps
    # impossible to desync even under a future refactor / partial-install).
    if not getattr(Layer.__init__, "_hpu_moe_prune", False):
        _orig_init = Layer.__init__

        def __init__(self, vllm_config, prefix, *args, **kwargs):
            _orig_init(self, vllm_config, prefix, *args, **kwargs)
            if _state["skip"] is None:
                _state["skip"] = _resolve_from_vllm_config(vllm_config)

        __init__._hpu_moe_prune = True
        Layer.__init__ = __init__

    # LINE-FOR-LINE copy of upstream DeepseekV2DecoderLayer.forward with ONE change: when this
    # layer's index is in the MoE-skip set AND its mlp is a real MoE, the FFN call is replaced by
    # an identity-on-residual (the FFN delta becomes zeros, so the next layer's fused add-norm
    # `x + residual` adds nothing). Attention, both RMSNorms, and the residual are unchanged.
    def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
        skip = _state["skip"] or set()
        if residual is None:
            residual = hidden_states  # PR#46651: drop redundant clone (input_layernorm returns a new tensor; original is not mutated)
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs = {"positions": positions, "hidden_states": hidden_states}
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (not isinstance(self.self_attn, _dv2.DeepseekAttention)
                and hidden_states.dtype == torch.float16):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # ---- the ONLY divergence from stock ------------------------------------------------
        if self.layer_idx in skip and isinstance(self.mlp, DeepseekV2MoE):
            # MoE-ONLY skip: zero the expert-FFN contribution to the residual stream. Keep the
            # post-attn `residual` intact -> attention + norms + residual all still apply.
            if not _state["logged"]:
                logger.warning(
                    "[hpu-moe-ext] %s ACTIVE — MoE-only skip set=%s; attention+norms+residual "
                    "kept, expert-FFN zeroed on those layers. ON-CARD COHERENCE GATE REQUIRED.",
                    _ENV_MOE, sorted(skip))
                _state["logged"] = True
            hidden_states = torch.zeros_like(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        # ------------------------------------------------------------------------------------

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual

    forward._hpu_moe_prune = True
    forward._hpu_orig_forward = _orig_forward
    Layer.forward = forward
    logger.info("[hpu-moe-ext] patched DeepseekV2DecoderLayer.forward "
                "(HPU_SKIP_MOE_LAYERS MoE-only skip)")


# ==================================================================================================
# Card-free self-test:  python -m hpu_moe_ext.layer_prune_patch --selftest
# ==================================================================================================
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  ok:   {msg}")

    # parse
    check(parse_layer_list("30-37") == set(range(30, 38)), "range 30-37 -> 30..37")
    check(parse_layer_list("30-37,40,42-45") == set(range(30, 38)) | {40} | set(range(42, 46)),
          "mixed list")
    check(parse_layer_list(" 5 , 7 ") == {5, 7}, "whitespace tolerant")
    check(parse_layer_list("") == set() and parse_layer_list(None) == set(), "empty -> empty set")
    for bad in ("a", "3-", "-3", "5-2", "3.5", "1,,2x"):
        try:
            parse_layer_list(bad)
            check(False, f"bad spec {bad!r} should raise")
        except ValueError:
            check(True, f"bad spec {bad!r} raises")

    # forbidden set for GLM-5.2 (L=78, k=3): dense {0,1,2} + aux {2,39,75}
    fb = forbidden_layers(78, 3)
    check(fb == {0, 1, 2, 39, 75}, f"forbidden(78,3) == {{0,1,2,39,75}} (got {sorted(fb)})")
    # general-patch safety: layer 0 (residual seed) is ALWAYS protected, even when k==0
    fb0 = forbidden_layers(60, 0)
    check(0 in fb0, f"forbidden(60,0) protects layer 0 (got {sorted(fb0)})")
    check(fb0 == {0, 2, 30, 57}, f"forbidden(60,0) == {{0,2,30,57}} (got {sorted(fb0)})")

    # validate: a clean middle block passes; protected/oob raise
    check(validate_skip(set(range(30, 38)), 78, 3, "T") == set(range(30, 38)),
          "30-37 validates clean")
    for bad_set, why in (({2}, "aux 2"), ({39}, "aux 39"), ({75}, "aux 75"),
                         ({0}, "dense 0"), ({78}, "oob 78"), ({100}, "oob 100")):
        try:
            validate_skip(bad_set, 78, 3, "T")
            check(False, f"skip {bad_set} ({why}) should raise")
        except RuntimeError:
            check(True, f"skip {bad_set} ({why}) raises")

    # the documented sweep lists must be coherence-eligible (no protected layer): contiguous
    # middle blocks that stay BELOW the EAGLE midpoint 39 and above the dense band {0,1,2}.
    for spec in ("30-37", "23-38", "15-38"):
        s = parse_layer_list(spec)
        check(not (forbidden_layers(78, 3) & s),
              f"sweep list {spec!r} avoids protected layers {{0,1,2,39,75}}")
    # K counts the sweep uses (8 / 16 / 24)
    check(len(parse_layer_list("30-37")) == 8, "K=8  for 30-37")
    check(len(parse_layer_list("23-38")) == 16, "K=16 for 23-38")
    check(len(parse_layer_list("15-38")) == 24, "K=24 for 15-38")

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
