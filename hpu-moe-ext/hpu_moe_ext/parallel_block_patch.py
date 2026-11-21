"""PARALLEL-TRANSFORMER-BLOCK — fuse GLM-5.2's two per-layer TP all-reduces into one (156 -> 78).

THE WALL: GLM-5.2 TP=8 decode pays TWO tensor-parallel all-reduces PER LAYER —
  #1 o_proj (RowParallelLinear.reduce_results) at the end of attention, and
  #2 the MoE down_proj reduce inside the FusedMoE —
i.e. 2 * 78 = 156 all-reduces per generated token, and they are SERIAL within the layer
(o_proj reduce must finish before post_attention_layernorm before the MoE).

THE REFORMULATION (GPT-J / Falcon "parallel block"): GLM's DecoderLayer ALREADY has two
SEPARATE norms (input_layernorm, post_attention_layernorm) — exactly Falcon's parallel form.
The SERIAL layer is:
    s        = input_layernorm(x, r)            # residual stream; hidden = norm_in(s)
    attn     = self_attn(norm_in(s))            # o_proj ALL-REDUCE #1
    post     = post_attention_layernorm(attn,s) # reads s+attn  -> norm_post(s+attn)
    moe      = mlp(norm_post(s+attn))           # down_proj ALL-REDUCE #2
    out_res  = s + attn + moe
The PARALLEL layer computes attn AND moe BOTH from the input-norm'd residual s, so their
TP-PARTIALS are summable into ONE all-reduce (all-reduce is LINEAR: Σ(a+m)=Σa+Σm):
    attn_p   = self_attn(norm_in(s))            # o_proj reduce DISABLED -> per-rank partial
    moe_p    = mlp(norm_post(s))                # MoE reduce  DISABLED -> per-rank partial   (reads s, NOT s+attn)
    out_res  = s + all_reduce(attn_p + moe_p)   # ONE all-reduce  (== s + attn + moe)
This is EXACT for the all-reduce. The ONLY quality cost is the MATH change: the MoE now reads
norm_post(s) instead of norm_post(s+attn). The offline card-free drift harness
(tests/glm52_parallel_block_drift.py) measures that cost per layer (driver = ||attn||/||s||).
It also BREAKS the intra-layer attn->MoE serial dependency (attn and MoE can co-issue).

The DSA indexer reads attention's OWN input (input_layernorm(s)), unaffected by this change.

TWO env knobs (BOTH default OFF -> pure no-op, stock forwards run byte-for-byte):
  HPU_PARALLEL_BLOCK=1                 -> ALL layers [0, num_hidden_layers) run parallel-block.
  HPU_PARALLEL_BLOCK_LAYERS=<list>     -> SELECTIVE: only the listed global layer indices
                                         (e.g. "1-77" or "3-38,40-77"); the rest stay SERIAL,
                                         byte-for-byte. Takes precedence over =1 if both set.
Each parallelized layer turns its 2 all-reduces into 1 -> total = 156 - (#parallel layers).

CORRECTNESS / BYTE-EXACTNESS contract (DSA build quality bar — fail LOUD, never silently corrupt):
  * Neither env set -> install() returns immediately; nothing is patched.
  * Non-selected layers run the ORIGINAL DeceoderLayer.forward unchanged (we call the captured
    _orig_forward), so they are byte-identical to stock including the fp16-overflow path.
  * Selected layers: o_proj.reduce_results and the MoE/dense down_proj reduce are set False at
    construction; the parallel forward sums the partials and does exactly ONE
    tensor_model_parallel_all_reduce. At tp_world==1 (e.g. the card-free harness) the partials
    are already full and the reduce is a no-op -> the parallel forward then computes EXACTLY
    s + attn + mlp(norm_post(s)), matching the offline reference.
  * fp16 activations are REFUSED on a selected layer (the routed_scaling_factor overflow shuffle
    is not reproduced here); GLM-5.2 serves bf16. Loud NotImplementedError.
  * is_sequence_parallel MoE (all-gather, not all-reduce) is REFUSED (the fusion identity does not
    hold for the SP pattern). Loud RuntimeError at resolve.

ON-CARD VALIDATION IS STILL REQUIRED before trusting throughput/quality (this module is the wiring;
the offline harness predicts the quality, the serve must confirm the coherence gate + the latency).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm")

_ENV_ALL = "HPU_PARALLEL_BLOCK"
_ENV_SEL = "HPU_PARALLEL_BLOCK_LAYERS"


# ==================================================================================================
# PURE parsing/validation core (card-free testable; no torch / no vllm) — reuse the layer_prune parser.
# ==================================================================================================
from .layer_prune_patch import parse_layer_list  # noqa: E402  (range/comma list -> set[int], LOUD on junk)


def resolve_selected(all_flag: str, sel_spec: str, num_hidden_layers: int) -> set[int]:
    """Return the validated set of layer indices to run in parallel-block form.

    sel_spec (HPU_PARALLEL_BLOCK_LAYERS) wins if non-empty; else all_flag truthy -> all layers.
    Raises ValueError/RuntimeError (LOUD) on a malformed or out-of-range spec.
    """
    L = int(num_hidden_layers)
    sel_spec = (sel_spec or "").strip()
    all_flag = (all_flag or "").strip().lower()
    if sel_spec:
        s = parse_layer_list(sel_spec)
    elif all_flag in ("1", "true", "yes", "on", "all"):
        s = set(range(L))
    else:
        return set()
    oob = {i for i in s if i < 0 or i >= L}
    if oob:
        raise RuntimeError(
            f"[hpu-moe-ext] {_ENV_SEL}/{_ENV_ALL}: layer index(es) {sorted(oob)} out of range "
            f"[0, {L}); the MTP layer {L} and beyond are not parallelizable here.")
    return s


def _install_final_ar_gate(runner_cls) -> bool:
    """Patch ``runner_cls._maybe_reduce_final_output`` ONCE so it honours a per-instance
    ``_hpu_pb_skip_final_ar`` flag. When that flag is set on a runner instance, its MoE forward
    returns the per-rank PARTIAL (shared+routed summed, routed-scaled, padding trimmed) WITHOUT
    the cross-rank all-reduce; every unflagged runner is byte-for-byte unchanged (the only added
    work is one attribute lookup). Returns True iff the class exposes ``_maybe_reduce_final_output``
    (the deployed modular MoERunner), False otherwise (e.g. the FusedMoE wrapper, or a fake).

    WHY an instance flag and not a config field (deployed vllm-gaudi:next, verified in-image):
    there is NO ``moe_config.skip_final_all_reduce`` -- the string does not occur anywhere in
    vllm core OR the gaudi plugin. The MoE cross-rank all-reduce lives ONLY in
    ``MoERunner._maybe_reduce_final_output`` (fused_moe/runner/moe_runner.py), gated on
    ``not is_sequence_parallel and (tp_size>1 or ep_size>1) and not _fused_output_is_reduced``
    (all read-only @property fields, so they cannot be flipped on the instance either). Re-installing
    the method to read a per-instance flag is graph-stable: method identity stays constant (no
    dynamo/Synapse recompile churn) and the flag is set once, before the first forward/capture."""
    meth = getattr(runner_cls, "_maybe_reduce_final_output", None)
    if meth is None or not callable(meth):
        return False
    if not getattr(meth, "_hpu_pb_gated", False):
        _orig = meth

        def _maybe_reduce_final_output(self, states, trunc_size, __orig=_orig):
            if getattr(self, "_hpu_pb_skip_final_ar", False):
                # PARALLEL-BLOCK: keep the MoE output a per-rank PARTIAL. Trim the padded dims
                # exactly as upstream, but SKIP the tensor_model_parallel_all_reduce -- the
                # decoder's parallel-block forward sums attn_partial + moe_partial and issues the
                # ONE combined all-reduce (all-reduce is linear: Sum(a+m) == Sum(a)+Sum(m)).
                return states[..., :trunc_size]
            return __orig(self, states, trunc_size)

        _maybe_reduce_final_output._hpu_pb_gated = True
        runner_cls._maybe_reduce_final_output = _maybe_reduce_final_output
        logger.info(
            "[hpu-moe-ext] parallel-block: installed the _hpu_pb_skip_final_ar gate on "
            "%s._maybe_reduce_final_output (this build has NO skip_final_all_reduce field; the "
            "MoE all-reduce is gated on is_sequence_parallel / tp,ep sizes / _fused_output_is_reduced).",
            getattr(runner_cls, "__name__", runner_cls))
    return True


def _disable_moe_all_reduce(experts) -> bool:
    """Disable the routed-MoE cross-rank FINAL all-reduce so the layer's MoE output stays a
    per-rank PARTIAL (the parallel-block forward sums the attn + MoE partials and issues ONE
    combined all-reduce). Returns True if a reduce mechanism was disabled, False if none was
    found (the caller raises LOUD on False -- never run a half-fused / double-reduced layer).

    DEPLOYED MECHANISM (vllm-gaudi:next, verified by reading the image):
      * ``mlp.experts`` is a FusedMoE that holds the MoERunner as ``experts.runner``
        (fused_moe/layer.py:423  `self.runner = MoERunner(...)`); FusedMoE.forward just calls
        ``self.runner.forward(...)``.
      * The ONLY MoE all-reduce is ``MoERunner._maybe_reduce_final_output`` (moe_runner.py:396).
        There is NO ``moe_config.skip_final_all_reduce`` and NO ``FusedMoE.reduce_results`` on
        this generation (both ABSENT -> the old hasattr/getattr paths could never fire, which is
        exactly the "no skip_final_all_reduce and no legacy reduce_results" failure this fixes).
      * The HPU W4A8 monolithic op (apply_monolithic) returns a per-rank partial and never
        self-reduces (0 all_reduce calls in hpu_fused_moe.py); the shared expert is built with
        reduce_results=False (deepseek_v2.py:322); and DeepseekV2MoE.forward adds no reduce of
        its own (non-SP path just returns experts(...)). So the runner's final all-reduce is the
        SOLE MoE reduce and gating it is complete & correct.
    We gate ``_maybe_reduce_final_output`` on a per-instance ``_hpu_pb_skip_final_ar`` flag (set
    here via object.__setattr__ -- the runner is an nn.Module, PluggableLayer) and HARDEN-check
    that the kernel does not self-reduce (else the caller would DOUBLE-REDUCE the MoE term).
    Legacy/forward-compat fields (moe_config.skip_final_all_reduce, FusedMoE.reduce_results) are
    still honoured best-effort for other builds and the card-free selftest. Idempotent."""
    if experts is None:
        return False
    disabled = False
    for holder in (experts, getattr(experts, "runner", None)):
        if holder is None:
            continue
        # PRIMARY (deployed modular runner): gate this runner class's _maybe_reduce_final_output.
        if _install_final_ar_gate(type(holder)):
            # HARDEN (DSA quality bar -- never silently corrupt): the parallel-block identity
            # REQUIRES a per-rank partial; if the kernel SELF-reduces (output already full),
            # skipping the final AR here would make the caller DOUBLE-REDUCE the MoE term. Fail
            # LOUD when we can read that state. (getattr(...,False) also swallows the property's
            # AttributeError on backends whose quant method has no `moe_kernel` -> treated as
            # not-self-reducing, which matches the monolithic path's output_is_reduced()==False.)
            try:
                self_reduces = bool(getattr(holder, "_fused_output_is_reduced", False))
            except Exception:  # noqa: BLE001  (defensive: any property side-effect -> can't confirm)
                self_reduces = False
            if self_reduces:
                raise RuntimeError(
                    "[hpu-moe-ext] parallel-block: this MoE kernel SELF-REDUCES its output "
                    "(_fused_output_is_reduced=True); the parallel-block fusion needs a per-rank "
                    "PARTIAL and would DOUBLE-REDUCE the MoE term. Refusing (disable "
                    "HPU_PARALLEL_BLOCK for this quant/backend).")
            object.__setattr__(holder, "_hpu_pb_skip_final_ar", True)
            logger.warning(
                "[hpu-moe-ext] parallel-block: MoE FINAL all-reduce DISABLED on runner %s "
                "(_hpu_pb_skip_final_ar=True) -> MoE now returns a per-rank PARTIAL; the combined "
                "attn+MoE all-reduce is issued by the decoder's parallel-block forward.",
                getattr(holder, "layer_name", type(holder).__name__))
            disabled = True
        # FORWARD-COMPAT / other builds + the card-free selftest fakes: honour a
        # moe_config.skip_final_all_reduce field or a legacy FusedMoE.reduce_results flag if
        # present. Both are ABSENT on the deployed build -> harmless no-ops there.
        mc = getattr(holder, "moe_config", None)
        if mc is not None and hasattr(mc, "skip_final_all_reduce"):
            try:
                mc.skip_final_all_reduce = True
                disabled = True
            except Exception:  # noqa: BLE001  (read-only property on some builds -> best effort)
                pass
        if hasattr(holder, "reduce_results"):
            try:
                holder.reduce_results = False
                disabled = True
            except Exception:  # noqa: BLE001
                pass
    return disabled


def _reshape_moe_in(moe_in, attn_p):
    """Shape the ONE-arg post_attention_layernorm output (norm_post(s)) to what self.mlp expects.

    HPU shape quirk (root cause of the on-card `too many values to unpack` crash):
    ``HPURMSNorm.forward_oot`` (vllm_gaudi/ops/hpu_layernorm.py) reshapes the fused-kernel result
    back to the input shape ONLY on the 2-arg (residual) path -- ``return x.reshape(orig_shape),
    residual`` -- while the 1-arg path just does ``return x``. Habana's ``FusedRMSNorm`` "requires
    3D tensors as inputs" (its comment), so the 1-arg ``post_attention_layernorm(residual)`` comes
    back 3D. The stock serial layer only ever feeds self.mlp the 2-arg-norm output, and
    ``DeepseekV2MoE.forward`` requires 2D (``num_tokens, hidden_dim = hidden_states.shape``) -> a
    3D moe_in unpacks to 3 values and raises.

    Reshape norm_post(s) to ``attn_p.shape`` -- the EXACT shape the stock 2-arg path reshapes its
    normed output to (the attention-output shape == 2D ``[num_tokens, hidden]``). This makes
    self.mlp receive 2D (byte-identical shape to stock) and makes moe_p line up with attn_p for the
    ``attn_p + moe_p`` add and the returned residual. It is a pure view (identical elements/order;
    numel is guaranteed equal -- both attn_p and residual are num_tokens*hidden) so there is NO
    math change; only the rank that the HPU 1-arg kernel failed to restore. Also unwraps a
    (normed, residual) tuple, in case a norm variant returns the 2-arg form."""
    if isinstance(moe_in, tuple):
        moe_in = moe_in[0]
    return moe_in.reshape(attn_p.shape)


# ==================================================================================================
# INSTALL — monkeypatch DeepseekV2DecoderLayer.__init__ (disable reduce on selected) + .forward
#           (parallel branch on selected, ORIGINAL forward on the rest). No-op when both envs unset.
# ==================================================================================================
def install() -> None:
    all_flag = os.environ.get(_ENV_ALL, "").strip()
    sel_spec = os.environ.get(_ENV_SEL, "").strip()
    if not all_flag and not sel_spec:
        return  # BYTE-EXACT: neither env set -> touch nothing.

    try:
        from vllm.model_executor.models import deepseek_v2 as _dv2
        from vllm.distributed import (tensor_model_parallel_all_reduce,
                                       get_tensor_model_parallel_world_size)
        import torch
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(
            f"[hpu-moe-ext] {_ENV_ALL}/{_ENV_SEL} set but vllm deepseek_v2/distributed "
            f"unavailable: {ex!r}")

    Layer = _dv2.DeepseekV2DecoderLayer
    DeepseekV2MoE = _dv2.DeepseekV2MoE
    DeepseekV2MLP = _dv2.DeepseekV2MLP
    DeepseekAttention = getattr(_dv2, "DeepseekAttention", ())

    if getattr(Layer.forward, "_hpu_parallel_block", False):
        return  # idempotent

    _state: dict = {"sel": None}          # resolved layer set (lazy, against live hf_config)
    _tp = {"world": None}

    def _resolve(vllm_config) -> set[int]:
        if _state["sel"] is None:
            hf = vllm_config.model_config.hf_config
            sel = resolve_selected(all_flag, sel_spec, int(hf.num_hidden_layers))
            # is_sequence_parallel MoE uses all-gather, not all-reduce -> fusion identity invalid.
            if getattr(vllm_config.parallel_config, "use_sequence_parallel_moe", False) and sel:
                raise RuntimeError(
                    "[hpu-moe-ext] HPU_PARALLEL_BLOCK incompatible with use_sequence_parallel_moe "
                    "(the MoE uses all-gather, not all-reduce; the parallel-block fusion identity "
                    "does not hold). Disable SP-MoE or the parallel block.")
            _state["sel"] = sel
            _tp["world"] = get_tensor_model_parallel_world_size()
            if sel:
                logger.warning(
                    "[hpu-moe-ext] %s ACTIVE — parallel-block on %d/%d layers %s; "
                    "all-reduces 156 -> %d (2 per serial layer -> 1 per parallel layer). "
                    "MoE reads norm_post(s) (NOT norm_post(s+attn)). ON-CARD COHERENCE GATE REQUIRED. "
                    "(offline drift harness flags layer 0 as high-risk: cos(np(s),np(s+a))<0.)",
                    _ENV_ALL, len(sel), int(hf.num_hidden_layers),
                    (sorted(sel) if len(sel) <= 12 else f"{min(sel)}..{max(sel)} ({len(sel)})"),
                    2 * int(hf.num_hidden_layers) - len(sel))
        return _state["sel"]

    def _disable_reduce_on_selected(self) -> None:
        """Disable this selected layer's per-rank reduces so both attention and MoE return
        UNREDUCED partials for the single combined all-reduce: o_proj.reduce_results=False on the
        attention RowParallelLinear, and the MoE runner's _hpu_pb_skip_final_ar=True (deployed
        modular build -- gates MoERunner._maybe_reduce_final_output) / experts.reduce_results=False
        (legacy) via _disable_moe_all_reduce; dense layers set down_proj.reduce_results=False.
        Called once per layer after a forward first sees it as selected."""
        if getattr(self, "_hpu_pb_reduce_off", False):
            return
        # --- o_proj (attention output RowParallelLinear): set on every alias we can reach ---
        attn = getattr(self, "self_attn", None)
        op = getattr(attn, "o_proj", None)
        for holder in (attn, getattr(attn, "mla_attn", None)):
            o = getattr(holder, "o_proj", None) if holder is not None else None
            if o is not None and hasattr(o, "reduce_results"):
                o.reduce_results = False
                op = o
        if op is None or not hasattr(op, "reduce_results"):
            raise RuntimeError(
                f"[hpu-moe-ext] parallel-block: could not locate o_proj.reduce_results on layer "
                f"{getattr(self,'layer_idx','?')} (attn type {type(attn).__name__}); refusing to "
                f"run a half-fused (incorrect) all-reduce.")
        # --- MoE final all-reduce (modular MoERunner) / dense down_proj reduce ---
        mlp = getattr(self, "mlp", None)
        if isinstance(mlp, DeepseekV2MoE):
            # Deployed build has NO FusedMoE.reduce_results AND NO moe_config.skip_final_all_reduce.
            # The MoE all-reduce is MoERunner._maybe_reduce_final_output (moe_runner.py), gated on
            # is_sequence_parallel / (tp>1 or ep>1) / _fused_output_is_reduced. _disable_moe_all_reduce
            # re-installs that method to honour a per-instance _hpu_pb_skip_final_ar flag (set here) so
            # the runner returns the per-rank partial; the HPU W4A8 monolithic op returns a per-rank
            # partial and never self-reduces, so this runner gate is the ONLY MoE all-reduce.
            if not _disable_moe_all_reduce(getattr(mlp, "experts", None)):
                raise RuntimeError(
                    f"[hpu-moe-ext] parallel-block: could not disable the MoE cross-rank all-reduce "
                    f"on layer {getattr(self,'layer_idx','?')} (experts type "
                    f"{type(getattr(mlp,'experts',None)).__name__}: no moe_config."
                    f"skip_final_all_reduce and no legacy reduce_results); refusing to run a "
                    f"half-fused (double-reduced / incorrect) all-reduce.")
        elif isinstance(mlp, DeepseekV2MLP):
            dp = getattr(mlp, "down_proj", None)
            if dp is None or not hasattr(dp, "reduce_results"):
                raise RuntimeError(
                    f"[hpu-moe-ext] parallel-block: dense MLP down_proj has no reduce_results on "
                    f"layer {getattr(self,'layer_idx','?')}.")
            dp.reduce_results = False
        else:
            raise RuntimeError(
                f"[hpu-moe-ext] parallel-block: unexpected mlp type {type(mlp).__name__} on layer "
                f"{getattr(self,'layer_idx','?')}.")
        self._hpu_pb_reduce_off = True

    _orig_forward = Layer.forward

    def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
        # Resolve the selected set lazily (config is reachable via the model build context the first
        # time; we stash it on the class once). Fall back: resolve from the layer's own attribute set.
        sel = _state["sel"]
        if sel is None:
            sel = _resolve_lazy(self)
        if self.layer_idx not in sel:
            # NON-SELECTED -> byte-exact stock forward.
            return _orig_forward(self, positions, hidden_states, residual, llama_4_scaling)

        # ---- PARALLEL-BLOCK FORM (selected layer) ------------------------------------------------
        _disable_reduce_on_selected(self)
        if hidden_states.dtype == torch.float16:
            raise NotImplementedError(
                "[hpu-moe-ext] HPU_PARALLEL_BLOCK supports bf16 activations only "
                "(the fp16 routed_scaling overflow path is not reproduced in the parallel form).")
        # input_layernorm: residual <- s (running residual stream); hidden <- norm_in(s)
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_kwargs = {"positions": positions, "hidden_states": hidden_states}
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        attn_p = self.self_attn(**attn_kwargs)             # per-rank partial (o_proj reduce OFF)

        # PARALLEL: MoE reads norm_post(s), NOT norm_post(s+attn). One-arg RMSNorm == plain norm of s.
        # HPU's 1-arg RMSNorm returns the fused kernel's RAW (3D) output -- it only reshapes back to
        # the 2D input shape on the 2-arg path -- so reshape norm_post(s) to attn_p's 2D
        # [num_tokens, hidden] shape (what stock feeds self.mlp; what DeepseekV2MoE.forward requires
        # and what lines up with attn_p for the add). See _reshape_moe_in.
        moe_in = _reshape_moe_in(self.post_attention_layernorm(residual), attn_p)
        moe_p = self.mlp(moe_in)                            # per-rank partial (down reduce OFF)

        fused = attn_p + moe_p
        reduced = tensor_model_parallel_all_reduce(fused) if (_tp["world"] or 1) > 1 else fused
        # next layer: input_layernorm(reduced, residual=s) -> norm(reduced + s) = norm(s + attn + moe)
        return reduced, residual

    def _resolve_lazy(self):
        # Recover vllm_config from the model build context if the __init__ hook didn't run; else
        # resolve from num layers inferred off the config attached to the layer's submodules.
        cfg = getattr(self, "config", None)
        nl = None
        if cfg is not None:
            nl = getattr(cfg, "num_hidden_layers", None)
        if nl is None:
            from vllm.config import get_current_vllm_config
            vc = get_current_vllm_config()
            return _resolve(vc)
        sel = resolve_selected(all_flag, sel_spec, int(nl))
        _state["sel"] = sel
        from vllm.distributed import get_tensor_model_parallel_world_size as _w
        _tp["world"] = _w()
        return sel

    # __init__ hook: resolve against vllm_config (reliable) the first time any layer is built.
    if not getattr(Layer.__init__, "_hpu_parallel_block", False):
        _orig_init = Layer.__init__

        def __init__(self, vllm_config, prefix, *args, **kwargs):
            _orig_init(self, vllm_config, prefix, *args, **kwargs)
            _resolve(vllm_config)

        __init__._hpu_parallel_block = True
        Layer.__init__ = __init__

    forward._hpu_parallel_block = True
    forward._hpu_orig_forward = _orig_forward
    Layer.forward = forward
    logger.info("[hpu-moe-ext] patched DeepseekV2DecoderLayer (__init__/forward) for "
                "HPU_PARALLEL_BLOCK (all=%r sel=%r)", all_flag, sel_spec)


# ==================================================================================================
# Card-free self-test:  python -m hpu_moe_ext.parallel_block_patch --selftest
# ==================================================================================================
def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False; print(f"  FAIL: {msg}")
        else:
            print(f"  ok:   {msg}")

    check(resolve_selected("", "", 78) == set(), "both empty -> no-op set")
    check(resolve_selected("1", "", 78) == set(range(78)), "=1 -> all 78")
    check(resolve_selected("0", "", 78) == set(), "=0 -> empty")
    check(resolve_selected("true", "", 78) == set(range(78)), "=true -> all")
    check(resolve_selected("1", "1-77", 78) == set(range(1, 78)), "LAYERS wins over =1")
    check(resolve_selected("", "3-38,40-77", 78) == (set(range(3, 39)) | set(range(40, 78))),
          "selective union")
    for bad in ("a", "3-", "5-2"):
        try:
            resolve_selected("", bad, 78); check(False, f"bad {bad!r} should raise")
        except (ValueError, RuntimeError):
            check(True, f"bad {bad!r} raises")
    try:
        resolve_selected("", "78", 78); check(False, "oob 78 should raise")
    except RuntimeError:
        check(True, "oob 78 raises")

    # --- _disable_moe_all_reduce: layout-robust MoE-reduce disable (card-free fakes) ---
    class _Cfg:  # stand-in FusedMoEConfig (has the skip_final_all_reduce dataclass field)
        def __init__(self): self.skip_final_all_reduce = False
    class _Runner:  # newer vLLM: FusedMoE(...) returns the MoERunner -> .moe_config directly
        def __init__(self): self.moe_config = _Cfg()
    r = _Runner()
    check(_disable_moe_all_reduce(r) is True and r.moe_config.skip_final_all_reduce is True,
          "runner.moe_config.skip_final_all_reduce set True")
    class _FusedMoE:  # older modular vLLM: FusedMoE module holding the runner as .runner
        def __init__(self): self.runner = _Runner()
    fm = _FusedMoE()
    check(_disable_moe_all_reduce(fm) is True and fm.runner.moe_config.skip_final_all_reduce is True,
          "experts.runner.moe_config.skip_final_all_reduce set True")
    class _Legacy:  # pre-modular vLLM: FusedMoE.reduce_results attribute
        def __init__(self): self.reduce_results = True
    lg = _Legacy()
    check(_disable_moe_all_reduce(lg) is True and lg.reduce_results is False,
          "legacy experts.reduce_results set False")
    check(_disable_moe_all_reduce(None) is False, "None experts -> False (no reduce found)")
    check(_disable_moe_all_reduce(object()) is False, "bare object -> False (no reduce found)")

    # --- DEPLOYED modular runner gate (the REAL vllm-gaudi:next mechanism) ------------------------
    class _Slice:  # records states[..., :trunc] so we can prove the all-reduce was SKIPPED
        def __getitem__(self, key): return ("TRIMMED_NO_AR", key)
    class _RealRunner:  # mimics MoERunner: _maybe_reduce_final_output + _fused_output_is_reduced
        _reduced = False
        layer_name = "layers.3.mlp.experts"
        @property
        def _fused_output_is_reduced(self): return self._reduced
        def _maybe_reduce_final_output(self, states, trunc_size):
            return ("ALL_REDUCED", trunc_size)  # sentinel: the ORIGINAL path all-reduces
    rr = _RealRunner()
    check(_disable_moe_all_reduce(rr) is True and getattr(rr, "_hpu_pb_skip_final_ar", None) is True,
          "real runner: _hpu_pb_skip_final_ar set True (primary gate fired)")
    check(rr._maybe_reduce_final_output(_Slice(), 5) == ("TRIMMED_NO_AR", (Ellipsis, slice(None, 5))),
          "flagged runner SKIPS all-reduce (returns padding-trim only)")
    fresh = _RealRunner()  # gated class, but never disabled -> no flag -> original AR preserved
    check(fresh._maybe_reduce_final_output(_Slice(), 7) == ("ALL_REDUCED", 7),
          "unflagged runner keeps original all-reduce (byte-exact for non-selected layers)")
    class _RealFused:  # FusedMoE-style wrapper: the RUNNER (held as .runner) must be flagged
        def __init__(self): self.runner = _RealRunner()
    fused = _RealFused()
    check(_disable_moe_all_reduce(fused) is True
          and getattr(fused.runner, "_hpu_pb_skip_final_ar", None) is True,
          "FusedMoE wrapper: experts.runner flagged (not the wrapper)")
    class _RedRunner(_RealRunner):  # SELF-reducing kernel -> must LOUD-refuse (else double-reduce)
        _reduced = True
    try:
        _disable_moe_all_reduce(_RedRunner()); check(False, "self-reducing kernel must raise LOUD")
    except RuntimeError:
        check(True, "self-reducing kernel raises LOUD (no double-reduce)")

    # --- SHAPE FIX: HPU 1-arg RMSNorm returns 3D -> _reshape_moe_in must give self.mlp attn_p's 2D --
    class _FakeT:  # minimal shape/reshape stand-in (numel-checked); no torch/numpy needed
        def __init__(self, shape): self.shape = tuple(shape)
        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            src = 1
            for d in self.shape: src *= d
            dst = 1
            for d in shape: dst *= d
            if dst != src:
                raise ValueError(f"reshape numel {self.shape}->{tuple(shape)}")
            return _FakeT(shape)
    attn2d = _FakeT((7, 128))  # attn output: 2D [num_tokens, hidden] (stock invariant: mlp needs 2D)
    check(_reshape_moe_in(_FakeT((1, 7, 128)), attn2d).shape == (7, 128),
          "1-arg-norm 3D moe_in reshaped to attn_p 2D shape (feeds self.mlp)")
    check(_reshape_moe_in((_FakeT((7, 128)), _FakeT((1, 7, 128))), attn2d).shape == (7, 128),
          "tuple moe_in -> unwrap[0] then reshape to attn_p 2D shape")
    check(_reshape_moe_in(_FakeT((7, 128)), attn2d).shape == (7, 128),
          "already-2D moe_in stays attn_p 2D shape (byte-exact no-op reshape)")
    try:
        _reshape_moe_in(_FakeT((1, 5, 128)), attn2d); check(False, "numel mismatch must raise")
    except ValueError:
        check(True, "moe_in/attn_p numel mismatch raises LOUD (never silently mis-shape)")

    # all-reduce accounting: K parallel layers -> 156 - K all-reduces
    for K, spec in ((77, "1-77"), (75, "3-77"), (78, "1")):
        s = resolve_selected(("1" if spec == "1" else ""), (spec if spec != "1" else ""), 78)
        print(f"    K={len(s):2d} ({spec!r}) -> all-reduces 156 -> {156 - len(s)}")
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
