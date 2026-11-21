"""hpu-moe-ext — out-of-tree vLLM plugin.

register() is the ``vllm.general_plugins`` entry point. It runs in-process at
engine init, AFTER the stock vllm-gaudi plugin (entry-point name "04." sorts
later), and:

  1. verifies the upstream symbols it depends on still exist (loud failure if not);
  2. routes GPTQ/AWQ MoE *experts* on HPU to MoeWNA16HPUMethod (reusing the
     proven VllmMixtureOfExpertsOpWNA16 op) — the missing dispatch branch;
  3. honors the GPTQ "dynamic" / AWQ "modules_to_not_convert" skip rules on the
     linear path (leave excluded bf16 layers unquantized) and on experts;
  4. enables auto-detect: a gptq/awq checkpoint selects the HPU kernel with no
     explicit --quantization flag.

It edits NO files under /opt/vllm-gaudi; an image rebuild or version bump cannot
clobber it, and _compat.verify() turns upstream drift into a build-time error.
"""
import re

_REGISTERED = False


def _skip_linear(prefix: str, hf_cfg: dict) -> bool:
    """True => leave this linear in bf16. Honors AWQ modules_to_not_convert
    (substring) + GPTQModel dynamic negative rules ('-:<regex>')."""
    # MoE router gates are kept in bf16 by GPTQModel/llm-compressor by default
    # (never listed in dynamic/modules_to_not_convert because they're never
    # quantized). Without skipping them here they'd be sent to the gptq/awq
    # linear method, which has no qweight to load -> garbage router -> garbage
    # routing. Match the router gate ('...mlp.gate') but NOT 'gate_proj'/'gate_up'.
    last = prefix.rsplit(".", 1)[-1]
    if last == "gate":
        return True
    for m in (hf_cfg.get("modules_to_not_convert") or []):
        if m and m in prefix:
            return True
    for pat in (hf_cfg.get("dynamic") or {}):
        if pat.startswith("-:"):
            if re.match(pat[2:], prefix):
                return True
        else:
            pat2 = pat[2:] if pat.startswith("+:") else pat
            if re.match(pat2, prefix):
                return False
    return False


def _install_config_patches():
    from vllm.model_executor.layers.linear import (LinearBase,
                                                   UnquantizedLinearMethod)
    import vllm.model_executor.layers.quantization.moe_wna16 as moe_wna16_mod
    from vllm_gaudi.ops.hpu_gptq import GPTQHPUConfig
    from vllm_gaudi.ops.hpu_awq import AWQHPUConfig

    from ._compat import (is_fused_moe_layer,
                          unquantized_fused_moe_method_cls)
    from .moe_wna16_hpu import MoeWNA16HPUMethod, HPUWNA16MoEConfig

    # Any code path that constructs the upstream (CUDA) expert method now gets
    # the HPU version instead.
    moe_wna16_mod.MoeWNA16Method = MoeWNA16HPUMethod

    def _is_experts(layer) -> bool:
        return is_fused_moe_layer(layer)

    def _make_patched(orig_from_config, orig_get_quant_method, method_name):

        def from_config(cls, config):
            inst = orig_from_config.__func__(cls, config) \
                if hasattr(orig_from_config, "__func__") \
                else orig_from_config(config)
            try:
                inst._hf_quant_cfg = dict(config)
            except Exception:  # noqa: BLE001
                inst._hf_quant_cfg = config
            return inst

        def get_quant_method(self, layer, prefix):
            hf = getattr(self, "_hf_quant_cfg", {}) or {}
            # Experts: route to the HPU WNA16 MoE method (or skip -> unquantized).
            if _is_experts(layer):
                cfg = HPUWNA16MoEConfig.from_hf(hf)
                if _skip_linear(prefix, hf):
                    return unquantized_fused_moe_method_cls()(layer.moe_config)
                return MoeWNA16HPUMethod(cfg, layer.moe_config)
            # Linear excluded by dynamic/modules_to_not_convert -> bf16.
            if isinstance(layer, LinearBase) and _skip_linear(prefix, hf):
                return UnquantizedLinearMethod()
            return orig_get_quant_method(self, layer, prefix)

        return classmethod(from_config), get_quant_method

    for Cfg, name in ((GPTQHPUConfig, "gptq_hpu"), (AWQHPUConfig, "awq_hpu")):
        new_from_config, new_gqm = _make_patched(
            Cfg.from_config, Cfg.get_quant_method, name)
        Cfg.from_config = new_from_config
        Cfg.get_quant_method = new_gqm

        # Auto-detect: pick the HPU kernel straight from the checkpoint's
        # quant_method, so no explicit --quantization flag is required.
        def _override(cls, hf_quant_cfg, user_quant, hf_config=None, _name=name):
            method = (hf_quant_cfg or {}).get("quant_method", "").lower()
            want = {"gptq_hpu": ("gptq", "auto_gptq"), "awq_hpu": ("awq",)}[_name]
            if user_quant == _name or (user_quant in (None, "auto") and method in want):
                return _name
            return None
        Cfg.override_quantization_method = classmethod(_override)


def _install_hpu_graph_patch():
    """Patch wrap_in_hpu_graph to pass free_inplace=False.

    When disable_tensor_cache=True (the vllm-gaudi default under lazy mode),
    HPUGraph uses a dry_run path:
        capture_begin(dry_run=True) -> orig_fwd -> capture_end
        mark_user_outputs(tlist, free_inplace=True)   <- FREES in-place buffers
        replayV3(...)   <- CRASHES: "Neither storage attached to input tensor"

    The crash triggers because the KV-state write-back buffer (core_attn_out,
    allocated as torch.zeros inside the forward and then written via
    hpu__slice_insert) is treated as an in-place tensor by HPUGraph.
    mark_user_outputs with free_inplace=True frees its storage.  replayV3 then
    fails to access it: "Strided Params Has Value: 0".

    Passing free_inplace=False keeps all in-place tensor storages alive after
    mark_user_outputs; dry_run replayV3 succeeds; subsequent per-inference
    replay (copy_to + replayV3) also succeeds.  Memory cost: the freed buffers
    are NOT reclaimed, so each graph instance holds slightly more HBM (the
    core_attn_out / ssm_state working tensors per layer × num_graph_buckets).
    At TP=8 the per-card graph budget is ~10 GB; the extra retention is on the
    order of hundreds of MB and does not OOM (verified: TP=4 OOMs even without
    this patch; TP=8 has sufficient headroom).

    Controlled by HPU_GRAPH_FREE_INPLACE env var (default "0" → patch active;
    set to "1" to restore the upstream free_inplace=True behavior).
    """
    import os
    if os.environ.get("HPU_GRAPH_FREE_INPLACE", "0") == "1":
        return  # opt-out: use upstream default
    try:
        import functools
        import habana_frameworks.torch as htorch
        _orig_wrap = htorch.hpu.wrap_in_hpu_graph

        @functools.wraps(_orig_wrap)
        def _patched_wrap(module, *args, free_inplace=False, **kwargs):
            # Force free_inplace=False so mark_user_outputs does NOT free the
            # storage of in-place state buffers (linear_attn / SSM state).
            return _orig_wrap(module, *args, free_inplace=free_inplace, **kwargs)

        htorch.hpu.wrap_in_hpu_graph = _patched_wrap
        try:
            import logging
            logging.getLogger("vllm").info(
                "hpu-moe-ext: patched wrap_in_hpu_graph free_inplace=False "
                "(fixes hpu__slice_insert 'Neither storage attached' at replayV3)")
        except Exception:  # noqa: BLE001
            pass
    except Exception as _ex:  # noqa: BLE001
        try:
            import logging
            logging.getLogger("vllm").warning(
                "hpu-moe-ext: wrap_in_hpu_graph patch failed (non-fatal): %s", _ex)
        except Exception:  # noqa: BLE001
            pass


def _install_communicator_patch():
    """Phase 2b (opt-in via env HPU_MARKSTEP_LAZY_ONLY=1): gate the unconditional
    htorch.core.mark_step() before dist.all_reduce in HpuCommunicator on is_lazy().

    That mark_step (hpu_communicator.py:42) is a FIXME workaround the upstream code
    documents as needed ONLY when PT_HPU_ENABLE_LAZY_COLLECTIVES=true (lazy-mode
    HPUGraph TP). On this torch.compile base (PT_HPU_LAZY_MODE=0) it is a pure host
    sync that serializes ~120 ungraphed all-reduces/token. combine() at
    hpu_communicator.py:81 already guards its mark_step the same way; this mirrors it
    onto all_reduce. Default OFF (env unset) => upstream behavior unchanged. Under
    lazy mode the guard re-enables the mark_step, so this is safe either way.
    """
    import os
    _ms_lazy = os.environ.get("HPU_MARKSTEP_LAZY_ONLY") == "1"
    # HPU_AR_NO_MARKSTEP=1: DROP the per-all_reduce mark_step ENTIRELY and rely on
    # PT_HPU_ENABLE_LAZY_COLLECTIVES to capture the collective IN-GRAPH. Upstream does a
    # mark_step before EVERY all_reduce = ~156 graph-fragmenting host-device syncs/token
    # (2 RowParallel reductions × 78 layers) — the prime suspect for the 0%-util sync-bound
    # decode. Removing it is also the decisive sync-vs-fabric test: faster => sync-bound
    # (software win); same => the all_reduce wall-time is the fabric (baseboard).
    _ms_none = os.environ.get("HPU_AR_NO_MARKSTEP") == "1"
    if not (_ms_lazy or _ms_none or os.environ.get("HPU_TIME_AR") == "1"):
        return
    import inspect
    import torch.distributed as dist
    import habana_frameworks.torch as htorch
    try:
        from vllm_gaudi.distributed.device_communicators.hpu_communicator import \
            HpuCommunicator
    except Exception:  # noqa: BLE001
        return  # communicator not present (e.g. TP=1) -> nothing to patch
    try:
        src = inspect.getsource(HpuCommunicator.all_reduce)
    except Exception:  # noqa: BLE001
        src = ""
    # Only patch the exact upstream shape (unconditional mark_step). If upstream
    # already gated it on is_lazy(), or the method changed, leave it ALONE.
    if "is_lazy()" in src or "mark_step" not in src:
        return

    if os.environ.get("HPU_TIME_AR") == "1":
        # INSTRUMENTATION: time each all_reduce wall-clock (synchronize-bounded) to read the
        # FABRIC latency directly. ~15-30us = healthy intra-node; ~1-3ms = degraded scale-up
        # fabric stalling the collective (the baseboard wall). Logs running avg.
        import time, logging as _lg
        _acc = [0.0, 0]
        _log = _lg.getLogger("vllm")
        def all_reduce(self, input_):
            htorch.hpu.synchronize(); _t = time.perf_counter()
            dist.all_reduce(input_, group=self.device_group)
            htorch.hpu.synchronize(); _acc[0] += time.perf_counter() - _t; _acc[1] += 1
            if _acc[1] % 312 == 0:   # ~2 tokens worth (156 AR/token)
                _log.warning("[AR-TIME] avg all_reduce = %.3f ms over %d calls",
                             _acc[0] / _acc[1] * 1000.0, _acc[1])
            return input_
    elif _ms_none:
        def all_reduce(self, input_):
            dist.all_reduce(input_, group=self.device_group)   # NO mark_step — in-graph collective
            return input_
    else:
        def all_reduce(self, input_):
            if htorch.utils.internal.is_lazy():
                htorch.core.mark_step()
            dist.all_reduce(input_, group=self.device_group)
            return input_

    HpuCommunicator.all_reduce = all_reduce
    import logging
    logging.getLogger("vllm").warning(
        "hpu-moe-ext: HpuCommunicator.all_reduce patched (no_markstep=%s, lazy_only=%s)",
        _ms_none, _ms_lazy)
    try:
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext Phase2b: HpuCommunicator.all_reduce mark_step gated on is_lazy()")
    except Exception:  # noqa: BLE001
        pass


def _install_layer_fuse_defaults():
    """HPU_LAYER_FUSE=1 — ONE switch for WHOLE-LAYER static-shape recipe fusion (default OFF).

    The GLM-5.2 decode layer is segmented into separate Synapse recipes by a handful of
    REMOVABLE boundaries in OUR editable code. Each is individually gated; this consolidator
    flips them ALL ON together (so the layer captures toward ~1 recipe instead of being split
    by per-layer mark_steps) via a SINGLE env var, and ONLY when the operator has not already
    set the individual knob (so an explicit per-knob choice in the env always wins / bisects).

    The boundaries it removes (each measured + validated independently — see the cited code):
      1. LAYER-BOUNDARY mark_step — the forward_hook the runner registers on EVERY
         DecoderLayer (hpu_model_runner.py:503-504 forward_hook -> htorch.core.mark_step()),
         installed with interval VLLM_CONFIG_HIDDEN_LAYERS (default 1 == every layer). Setting
         it to the layer count makes the hook fire only once at the model tail (the interval
         gate counter[0] % n; n=78 -> only the 78th match registers the hook). We default it to
         a large value (10000) so it never fires per-layer regardless of the model's depth.
      2. PER-ALL_REDUCE mark_step — the two RowParallel reductions per MoE layer
         (o_proj + experts) each prefix dist.all_reduce with htorch.core.mark_step()
         (hpu_communicator.py FIXME, mirrored/gated in _install_communicator_patch:226-235).
         HPU_AR_NO_MARKSTEP=1 drops it and relies on PT_HPU_ENABLE_LAZY_COLLECTIVES to capture
         the collective IN-GRAPH (the env already sets that flag).
      3. MoE ROUTED-GATHER mark_step — the routed-K path's MANDATORY pre-index_select boundary
         (fused_expert_routed.py:436). It exists for the data-dependent gather, so it is NOT
         droppable on the routed path. Whole-layer fusion routes the MoE through the MONOLITHIC
         single-recipe op instead (HPU_MOE_FUSED_MONOLITHIC=1, HPU_MOE_ROUTED_K=0), whose ONLY
         in-Python work is the STATIC-shape dense [E,T,1] router build -> NO mandatory mark_step
         (HPU_MOE_FUSED_MONO_MARKSTEP=0; fused_monolithic_moe.py:202-232).

    BYTE-EXACTNESS: every flag here changes recipe SEGMENTATION ONLY, not the op graph or any
    value (removing a mark_step re-segments; the monolithic op is bit-equivalent to the routed
    gather — non-routed experts carry router-weight 0, proven by HPU_MOE_FUSED_VERIFY /
    HPU_MOE_ROUTED_VERIFY). Greedy France/arithmetic on-card is the gate. Default (env unset)
    leaves EVERY individual knob at its upstream default => byte-for-byte unchanged.

    BISECT: set HPU_LAYER_FUSE=1 AND override any single knob to its OFF value to isolate which
    boundary a divergence (if any) needs restored — e.g. HPU_MOE_FUSED_MONO_MARKSTEP=1 (restore
    the MoE boundary), or VLLM_CONFIG_HIDDEN_LAYERS=1 (restore the per-layer hook), or
    HPU_AR_NO_MARKSTEP=0 (restore the all-reduce boundary).

    NOTE (honest expectation, from the leakmap): at single-stream the decode is already
    ~1.6 recipes/layer (MoE = one fused op, MLA is captured inline, RMSNorms inline), and the
    ~633 synchronizeEvent fences/step are dominated by the TP=8 in-graph all-reduce collectives
    (2/layer x 78) + the irreducible per-layer ExecuteCachedGraph replay — neither of which a
    mark_step removal eliminates (an in-graph hcclAllReduce still fences across the 8 cards).
    This is why the prior scattered-flag fuse run was ~nil. This switch makes the lever a single
    measurable A/B; the win is bounded by the collective floor, not the mark_step count."""
    import os
    if os.environ.get("HPU_LAYER_FUSE", "0") != "1":
        return
    # setdefault: an explicit operator choice for any knob ALWAYS wins (bisect-friendly).
    _defaults = {
        "VLLM_CONFIG_HIDDEN_LAYERS": "10000",   # #1: per-layer forward_hook mark_step -> tail-only
        "HPU_AR_NO_MARKSTEP": "1",              # #2: drop per-all_reduce mark_step (in-graph collective)
        "HPU_MOE_FUSED_MONOLITHIC": "1",       # #3: route MoE through the single-recipe monolithic op
        "HPU_MOE_FUSED_MONO_MARKSTEP": "0",    #     ...with its only (legacy) boundary removed
        "HPU_MOE_ROUTED_K": "0",               #     ...disable the routed-K gather (its mandatory mark_step)
    }
    _set = {}
    for k, v in _defaults.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
            _set[k] = v
    try:
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: HPU_LAYER_FUSE=1 -> whole-layer recipe fusion defaults applied %s "
            "(any pre-set knob left untouched; PT_HPU_ENABLE_LAZY_COLLECTIVES must be true for "
            "the in-graph all_reduce). Greedy byte-exact is the on-card gate.", _set)
    except Exception:  # noqa: BLE001
        pass


def _install_spec_decode_kv_patch():
    """MTP / speculative-decode on HPU: vllm's bind_kv_cache raises NotImplementedError
    when a layer_index maps to >1 kv-cache name — which the spec-decode DRAFT layer
    triggers (its attention layer collides in extract_layer_index with a target layer).
    cuda/cpu/xpu explicitly `pass` here (the multi-name case binds correctly into both
    runner_kv_caches and forward_context); HPU was simply never whitelisted. Treat HPU
    the same so MTP can profile + bind. hpu_worker does `from ...utils import bind_kv_cache`
    (a bare name), so patch BOTH the source module and the already-imported reference."""
    try:
        import vllm.v1.worker.utils as _u
        from vllm.v1.worker.utils import extract_layer_index
        from collections import defaultdict

        def _bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):
            assert len(runner_kv_caches) == 0
            index2name = defaultdict(list)
            for layer_name in kv_caches:
                index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
            for layer_index in sorted(index2name.keys()):
                # multi-name (spec-decode draft / encoder-decoder) is benign on every
                # runner incl. HPU — cuda/cpu already `pass`; just append in index order.
                for layer_name in index2name[layer_index]:
                    runner_kv_caches.append(kv_caches[layer_name])
            for layer_name, kv_cache in kv_caches.items():
                forward_context[layer_name].kv_cache = kv_cache

        _u.bind_kv_cache = _bind_kv_cache
        try:
            import vllm_gaudi.v1.worker.hpu_worker as _hw
            if hasattr(_hw, "bind_kv_cache"):
                _hw.bind_kv_cache = _bind_kv_cache
        except Exception:
            pass  # not yet imported -> its later `from ...import` picks up the patched _u
        print("hpu-moe-ext: patched bind_kv_cache (MTP/spec-decode multi-name layer_index on HPU)")
    except Exception as e:  # never block startup on this
        print(f"hpu-moe-ext: bind_kv_cache spec-decode patch skipped ({e!r})")


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    from . import _compat
    _compat.verify()
    # HPU_LAYER_FUSE=1 (default OFF): apply the whole-layer recipe-fusion env defaults FIRST,
    # before the communicator patch reads HPU_AR_NO_MARKSTEP and before the runner reads
    # VLLM_CONFIG_HIDDEN_LAYERS / the MoE method reads HPU_MOE_*. setdefault, so any explicit
    # per-knob env always wins. No-op (byte-identical) when unset.
    _install_layer_fuse_defaults()
    _install_hpu_graph_patch()
    _install_config_patches()
    _install_communicator_patch()
    _install_spec_decode_kv_patch()
    from . import decode_profiler_patch  # NO-OP unless HPU_PROFILE_DECODE=1 — single-step budget split
    decode_profiler_patch.install()
    decode_profiler_patch.install_step_timer()  # NO-OP unless HPU_STEP_TIMER=1 — forward-cadence timing
    decode_profiler_patch.install_engine_step_split()  # NO-OP unless HPU_ENGINE_STEP_SPLIT=1 — attribute the inter-step gap (EngineCore loop)
    from . import decode_dump_patch  # NO-OP unless HPU_MTP_DUMP=1 — MTP head training-data dump
    decode_dump_patch.install()
    from . import native_selffeed_dump  # NO-OP unless HPU_MTP_SELFFEED_DUMP=1 — native MTP-head self-feed
    native_selffeed_dump.install()      # drift capture (HEAD-TTT OPTION A; taps propose_eagle_decode)
    from . import auto_round_hpu  # noqa: F401  triggers @register_quantization_config
    # DSA lightning-indexer: AutoRound stores indexer.wk as GPTQ-packed int, but
    # vllm only special-cases FP8 wk -> dequant+fuse it into wk_weights_proj.weight.
    # No-op for non-AutoRound/non-DSA models (delegates to the original fn).
    from . import indexer_wk_patch
    indexer_wk_patch.install()
    # MLA A-projections (q_a_proj + kv_a_proj_with_mqa): AutoRound stores these as
    # GPTQ int8, but vLLM FUSES them into a single fused_qkv_a_proj
    # (MergedColumnParallelLinear) when q_lora_rank is not None (GLM-5.1:
    # q_lora_rank=2048). get_quant_method is called with prefix
    # "...self_attn.fused_qkv_a_proj", which owns no .qweight in the checkpoint
    # (only the individual q_a_proj/kv_a_proj_with_mqa shards do) -> WITHOUT a fix it
    # dispatches UnquantizedLinearMethod -> the model builds fused_qkv_a_proj.weight
    # (bf16) only -> load_weights' stacked_params_mapping rewrites the shard
    # ...q_a_proj.qweight -> ...fused_qkv_a_proj.qweight, finds it absent
    # (name_mapped not in params_dict), `continue`s, then falls to the else branch and
    # KeyErrors on params_dict["...kv_a_proj_with_mqa.qweight"].
    #
    # FIX (config-side, in auto_round_hpu.py): AutoRoundHPUConfig._checkpoint_is_quantized
    # and _bits_for_prefix special-case the fused leaf via _FUSED_LEAF_TO_SHARDS:
    # "fused_qkv_a_proj" is treated as quantized iff a component shard
    # (q_a_proj / kv_a_proj_with_mqa) is quantized, at the max of the shards' bits (8).
    # get_quant_method then returns GPTQDequantToBF16LinearMethod(bits=8) and
    # create_weights builds REAL fused qweight[1536,2624]/qzeros[48,656]/scales[48,2624]
    # params (verified by CPU repro). The stock stacked_params_mapping entries
    # ("fused_qkv_a_proj","q_a_proj",0) / (...,"kv_a_proj_with_mqa",1) then find those
    # params and load each shard via the new-style PackedvLLMParameter weight_loader;
    # process_weights_after_loading dequants int8->bf16 — the SAME proven path as the
    # standalone q_b_proj/kv_b_proj/o_proj. No loader interception needed.
    #
    # NOTE: the older mla_proj_patch.py (a _try_load_fp8_indexer_wk-style loader patch
    # that intercepts the shards and writes into fused_qkv_a_proj.weight) is the
    # ALTERNATIVE to the config-side fix and is MUTUALLY EXCLUSIVE with it: once
    # _FUSED_LEAF_TO_SHARDS makes the fused layer build .qweight (not .weight), that
    # patch hits its KeyError fallback ("neither fused_qkv_a_proj.weight nor
    # q_a_proj.weight found"). It is intentionally NOT installed. Do not re-enable it
    # while the config-side fix is present.
    # The HPU platform (vllm_gaudi/platform.py) keeps its OWN supported_quantization
    # allowlist; vllm/platforms/interface.py raises "<q> quantization is currently
    # not supported in <device>" for anything not in it — BEFORE our registered
    # config is consulted. register_quantization_config does not touch it, so add
    # our methods here.
    try:
        from vllm.platforms import current_platform
        for _q in ("auto-round", "auto_round"):
            if _q not in current_platform.supported_quantization:
                current_platform.supported_quantization.append(_q)
    except Exception:  # noqa: BLE001
        pass
    # Fix Bug #1: Sampler.gather_logprobs / gather_specific_token_logprobs call
    # mark_unbacked inside the sampler which the HPU runner compiles via
    # torch.compile (regional compilation, line ~4590 of hpu_model_runner.py).
    # dynamo raises "Attempt to trace forbidden callable mark_unbacked" on every
    # request after warmup, crashing all workers. Wrap the calls in a
    # @torch.compiler.disable helper — the same pattern used in hpu_model_runner
    # for _mark_unbacked_dim0(attn_metadata.block_list).
    from . import sampler_patch
    sampler_patch.install()
    # DeepSeek-MTP (nextn) speculative head: GLM-5.1's AutoRound export quantized the MTP
    # layer-78 eh_proj to int4, but vllm core builds it as a plain bf16 nn.Linear ->
    # KeyError 'model.layers.78.eh_proj.qweight' at draft load. Rebuild eh_proj as a
    # quant-aware ReplicatedLinear when the checkpoint stores it quantized. No-op for
    # non-MTP models and for bf16 MTP heads (DeepSeek-V3).
    from . import mtp_eh_proj_patch
    mtp_eh_proj_patch.install()
    # MTP draft embed/head tie: GLM-5.1's checkpoint ships NO embed_tokens /
    # shared_head.head inside the MTP layer (shares the target's by design), but
    # DeepSeekMTP.load_weights skips all non-spec-layer weights -> the draft ran
    # with a RANDOM embedding + RANDOM head (constant junk draft, acceptance 0%).
    # Ties to the target's modules when (and only when) the ckpt omits them.
    from . import mtp_share_head_patch
    mtp_share_head_patch.install()
    # A.2 RESIDUAL ADAPTER (gate HPU_MTP_ADAPTER=<adapter.pt>; OFF -> stock, no proposer wrap): ADD a
    # card-free-trained residual adapter's logits to the NATIVE MTP head's drafts (argmax(native+alpha*
    # adapter)) to LIFT per-depth accept above the measured native floor (~1.50). HPU_MTP_ADAPTER_ALPHA=0
    # == native byte-exact (the never-regress floor + the A/B knob). Wraps HpuEagleProposer.propose.
    from . import mtp_residual_adapter
    mtp_residual_adapter.install()
    # MTP bf16 layer-78 graft (gate HPU_MTP_BF16_GRAFT=<dir>): the int4 export's
    # MTP layer is dead weights (fp64 oracle, PROJECT_FINDINGS §8); graft the
    # original bf16 layer (quant bypass + load-time weight substitution).
    from . import mtp_bf16_graft
    mtp_bf16_graft.install()
    # MTP fp8 layer-78 graft (gate HPU_MTP_FP8_GRAFT=<dir>): COMPLIANT path — loads the
    # OFFLINE-prequantized fp8 layer-78 (no bf16 served) via vllm-gaudi's HPU fp8 method.
    from . import mtp_fp8_graft
    mtp_fp8_graft.install()
    # GLM-5.2 whole-model fp8-OTHER graft (gate HPU_FP8_OTHER_GRAFT=<dir>; no-op when unset): routes the
    # non-expert big Linears (MLA/shared/dense) to vllm fp8 while routed experts stay int4 (AutoRound)
    # and the DSA indexer / router gate / norms stay bf16. Completes the int4+fp8 3-tier precision scheme.
    from . import glm52_fp8_other_graft
    glm52_fp8_other_graft.install()
    # LAYER-PRUNE (gate HPU_SKIP_LAYERS=<list> and/or HPU_SKIP_MOE_LAYERS=<list>; BOTH default off ->
    # this is a pure no-op and the stock DeepseekV2Model/DecoderLayer forwards run byte-for-byte
    # unchanged). Shortens the device-serial decode chain for GLM-5.2 single-stream by skipping whole
    # layers (HPU_SKIP_LAYERS) or just the expert-FFN of a layer (HPU_SKIP_MOE_LAYERS, attention kept).
    # Loud-validated against hf_config on first forward (refuses dense band + EAGLE-3 aux taps). The
    # largest K that survives the on-card coherence gate is the usable prune.
    from . import layer_prune_patch
    layer_prune_patch.install()
    # EXPERT-PRUNE MASK (gate HPU_PRUNE_EXPERTS=<droppable-set.json>; default OFF -> pure no-op, stock
    # routing byte-for-byte). Reversible card-free QUALITY GATE for WAVE-2 calibrated expert pruning:
    # masks the low-saliency experts (reap_saliency.py) OUT OF ROUTING via the noaux_tc bias + a logit
    # mask, so perplexity/needle/coherence of any prune ratio can be measured on the BASE checkpoint
    # (no 310G regen). FAITHFUL to the physical prune (prune_experts.py) -- same survivors+renorm -- but
    # does NOT free HBM/dequant (that is the .pt transform's job). Loud-validated against hf_config.
    from . import expert_prune_patch
    expert_prune_patch.install()
    # PARALLEL-TRANSFORMER-BLOCK (gate HPU_PARALLEL_BLOCK=1 or HPU_PARALLEL_BLOCK_LAYERS=<list>;
    # default OFF -> pure no-op, stock DeepseekV2DecoderLayer.forward byte-for-byte). Fuses the TWO
    # per-layer TP all-reduces (o_proj + MoE down_proj) into ONE by computing attn AND MoE both from
    # the input-norm'd residual s (Falcon/GPT-J parallel form): h = s + all_reduce(attn_p + moe_p).
    # 156 -> 156-K all-reduces (K = #parallel layers). EXACT for the all-reduce; the only quality cost
    # is MoE reads norm_post(s) not norm_post(s+attn) (offline drift harness
    # tests/glm52_parallel_block_drift.py: safe layers 1-77 cos>0.999, layer 0 high-risk cos<0).
    from . import parallel_block_patch
    parallel_block_patch.install()
    # MLA-DECODE MEGAKERNEL (gate HPU_MLA_MEGAKERNEL=1; default off -> pure no-op, stock
    # HPUMLAAttention.forward_impl runs byte-for-byte). Extends SRAM-residency to the attention
    # half of the GLM-5.2 decode layer: replaces the absorb->rope->paged-gather->sdpa->V-up chain
    # with the fused torch.ops.hpu_mla_ext.mla_decode_paged op at decode B==1 (the single-stream
    # critical-path lever; the MoE half already won 2.6x via fused_monolithic_moe). Option-1a
    # identity-rope = byte-exact; HPU_MLA_MEGAKERNEL_VERIFY=1 A/Bs vs stock (relNorm<1e-2, LOUD).
    from . import mla_megakernel_patch
    mla_megakernel_patch.install()
    # BINNED W4A8 MoE (WAVE-2; gate HPU_MOE_BINNED=1, default OFF -> pure no-op, the dense
    # W4A8 monolithic runs byte-for-byte). Routes the W4A8 MoE through a capacity-permuted
    # grouped-GEMM (binned_moe.fused_moe_binned): each local expert computes ONLY its <=C
    # routed-token rows (sparse E*C compute vs the dense monolithic's E*T) -> ~T/C fewer GEMM
    # MACs + smaller bf16 transient (un-OOMs prefill), stacking with W4A8's 2x MME. Reads
    # UNCHANGED (union ~= E, dequant once/STEP). Byte-exact when C >= max per-expert load;
    # undersized C safe-degrades (sentinel drop, never wrong-fold). CPU-oracle PASS
    # (tests/binned_moe_oracle.py + test_binned_wrapper_cpu.py); on-card gate = HPU_MOE_BINNED_VERIFY=1.
    from . import binned_moe_patch
    binned_moe_patch.install()
    # EAGLE draft for from-scratch factory heads (M3/K2.7 — no native MTP). Gate HPU_EAGLE_HEAD=<head_factory dir>:
    # registers FactoryEagleForCausalLM so --speculative-config method=eagle can load an exported head draft-ckpt.
    import os as _os
    import logging as _lg
    _eh = _os.environ.get("HPU_EAGLE_HEAD", "").strip()
    if _eh:
        try:
            import sys as _sys
            _sys.path.insert(0, _eh)               # head_factory/factory on sys.path
            import factory_eagle as _fe
            _fe.register()
            _lg.getLogger("vllm").warning("[hpu-moe-ext] FactoryEagleForCausalLM registered (HPU_EAGLE_HEAD=%s)", _eh)
        except Exception as _ex:  # noqa: BLE001
            _lg.getLogger("vllm").warning("[hpu-moe-ext] factory_eagle register failed: %r", _ex)
    _REGISTERED = True
    try:
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext registered: GPTQ/AWQ FusedMoE -> VllmMixtureOfExpertsOpWNA16 on HPU")
    except Exception:  # noqa: BLE001
        pass
