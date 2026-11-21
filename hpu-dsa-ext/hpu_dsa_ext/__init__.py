"""hpu-dsa-ext: run DeepSeek Sparse Attention models (GlmMoeDsa, DeepSeek-V3.2)
on HPU AS DESIGNED — real lightning-indexer top-k + sparse attention, never a
dense fallback. Loads via the vllm.general_plugins entry-point '05.hpu_dsa_ext'.

STATUS: orchestrated. `register()` applies the platform patch, then registers the
full sparse path (indexer forward_oot op, indexer K-cache hook, gather-consume MLA,
config-driven arch binding), then flips platform_patch.DSA_READY ONLY behind the
HPU_DSA_READY=1 env guard — and only if every component registered cleanly. Until
then DSA stays correctly BLOCKED (explicit NotImplementedError, never dense). See
BUILD_PLAN.md / IMPLEMENTATION.md.

Sparse path components (registered in dependency order by `register()`):
  1. sparse_indexer_hpu.py  — SparseAttnIndexer.forward_oot: real lightning-indexer
       top-k over the indexer K-cache, causal, top-k=index_topk -> topk_indices_buffer
       (HPU is out-of-tree; dispatch routes to forward_oot, not forward_hpu).
  2. indexer_cache_hpu.py   — allocate the bf16/head_dim=128/flat indexer K-cache
       (the HPU runner drops DeepseekV32IndexerCache) + insert/gather hook.
  3. sparse_mla_consume.py  — HPUMLAAttention gather-consume of topk_indices_buffer
       (sentinel-scatter, NaN guard) in forward_mha (prefill) / forward_mqa (decode).
  4. glm_moe_dsa_wire.py    — bind GlmMoeDsa/DeepseekV3 onto the (config-driven)
       sparse path + dims validation + the HPU_DSA_READY gate + launch env.
"""

from . import _compat


# Sibling components of the sparse path, in dependency order. Each entry:
#   (module, callable_attr_or_None, human_label)
# A module with an install entry-point (register()/apply()) has it invoked; a
# module that is a pure library (callable=None) is merely IMPORTED to verify it
# is present + importable. Order matters: indexer op + K-cache hook + consume
# library must all exist before the arch wiring flips DSA_READY.
#
# NOTE on entry-point names (verified against each module's public surface):
#   * sparse_indexer_hpu.register()  -> monkeypatches SparseAttnIndexer.forward_oot
#   * indexer_cache_hpu.apply()      -> installs the get_kv_cache_spec + bind_kv_cache
#       runner patches (the module exposes apply(), NOT register()).
#   * sparse_mla_consume             -> a PURE TORCH library (gather/masked consume +
#       sentinel scatter + NaN guard). It has no install hook: restricting the MLA
#       forward to the selected tokens is a forward-path monkeypatch of the installed
#       vllm_gaudi MLA impl (oot_mla.HPUMLAAttention.forward_impl / forward_mha /
#       forward_mqa), which must introspect the on-HPU class and is wired by the main
#       HPU process. Here we only import it (presence check); see consume_wire below.
#   * glm_moe_dsa_wire.register()    -> binds the DSA arch names + dims validation.
_SPARSE_COMPONENTS = (
    ("sparse_indexer_hpu", "register", "indexer forward_oot (lightning-indexer top-k)"),
    ("indexer_cache_hpu", "apply", "indexer bf16 K-cache alloc + insert/gather hook"),
    ("sparse_mla_consume", None, "gather-consume sparse MLA library (forward-wire on HPU)"),
    ("sparse_mla_wire", "register", "wire gather-consume into HPUMLAAttention.forward_impl"),
    ("glm_moe_dsa_wire", "register", "GlmMoeDsa/DeepseekV3 arch binding (config-driven)"),
)


def _load_component(modname, attr, label, *, strict):
    """Import `hpu_dsa_ext.<modname>` and run its `<attr>()` if present.

    strict=True  (operator opted into the live flip via HPU_DSA_READY): a missing
                 or failing component is a HARD error — we will NOT flip DSA_READY
                 on a broken/incomplete sparse stack (that is the dense-fallback
                 trap). Raise loudly.
    strict=False (default, skeleton load): tolerate a not-yet-authored sibling so
                 the overlay still loads and stays correctly BLOCKED. Returns the
                 import error string (or None on success) for the boot log.
    """
    import importlib

    try:
        mod = importlib.import_module(f"{__name__}.{modname}")
        if attr is not None:
            getattr(mod, attr)()
        return None
    except Exception as e:  # noqa: BLE001
        if strict:
            raise RuntimeError(
                f"[hpu-dsa-ext] HPU_DSA_READY=1 but sparse component "
                f"'{modname}' ({label}) failed to register: {type(e).__name__}: {e}. "
                "Refusing to flip DSA_READY on an incomplete/broken sparse stack."
            ) from e
        return f"{modname} ({label}): {type(e).__name__}: {e}"


def register():
    # 1) Loud drift-guard FIRST — refuse to load on upstream API drift.
    _compat.verify()

    # 2) Patch the platform to allow sparse+MLA routing. While
    #    platform_patch.DSA_READY is False this still REFUSES sparse with a clear
    #    message (never serves dense) — safe to load now.
    from . import platform_patch
    platform_patch.apply()

    # 2b) Force skip_topk=True on HPUMultiHeadLatentAttentionWrapper.
    #     UNCONDITIONAL — must run before HPU_DSA_READY is checked because the
    #     model crashes at dynamo graph-capture warmup (Indexer.forward calls
    #     per_token_group_quant_fp8 which hits triton.next_power_of_2, absent on
    #     HPU where HAS_TRITON=False) regardless of the DSA gate.
    from . import hpu_mla_skip_indexer
    hpu_mla_skip_indexer.register()

    # 2c) Prevent the HPU torch.compile backend from crashing on
    #     HPUMLAImpl.forward_mha (aten.slice on 3D view -> AttributeError:
    #     'NoneType' object has no attribute 'file').  UNCONDITIONAL — the crash
    #     occurs for any MLA model at warmup graph-capture, independent of DSA.
    from . import hpu_mla_compile_fix
    hpu_mla_compile_fix.register()

    # 2c2) TREE-VERIFY fix (gated HPU_FSDPA_BIAS_AUTHORITATIVE=1, default OFF): make an
    #      explicit prefill attn_bias the SOLE authority over FusedSDPA masking by dropping
    #      the kernel's own is_causal when a bias is present. The tree-verify Pass A runs a
    #      bs=Tsum prefill over the tree nodes WITH committed context + the [Tsum,ctx+Tsum]
    #      ancestor bias; the in-image _fsdpa_prompt_attention only drops is_causal under INC
    #      (USING_INC), so without QUANT_CONFIG the kernel double-applies its causal triangle
    #      over the non-square geometry ON TOP of the ancestor bias -> wrong context -> garbage.
    #      Default-OFF == byte-identical (a standard no-prefix prefill carries NO explicit bias).
    from . import tree_verify_bias_authoritative
    tree_verify_bias_authoritative.register()

    # 2d) NaN-sanitize the embedding output (gated HPU_SANITIZE_EMBED=1). The HPU
    #     vocab-parallel embedding forward emits a spurious NaN (the embed weight
    #     is verified clean) that poisons the residual stream -> a downstream
    #     NaN-guard freezes activations -> token-salad. nan_to_num breaks the chain.
    from . import nan_sanitize_embed
    nan_sanitize_embed.install()

    # 2e) DIAGNOSTIC (gated HPU_MHA_DIAG=N): log dense-MLA prefill forward_mha norms to
    #     localize the near-zero attention-output bug (q~100 but out~0.1). Read-only.
    from . import mha_diag
    mha_diag.install()

    # 2f) GLM-5.1 RMSNorm restore (gated HPU_GLM_NORM_OFFSET=1): the AutoRound export stored
    #     GlmMoeDsaRMSNorm weights offset by -1 (decoder norms ~0 vs the standard ~1). Add 1.0
    #     at load so attention isn't dead. Confirmed: V rms 0.0002->0.047. ROOT-CAUSE fix #1.
    from . import glm_norm_offset
    glm_norm_offset.install()

    # 2g) GLM-5.1 RoPE convention (gated HPU_GLM_ROPE_NEOX=1): GLM-5.1 uses NeoX rotate-half
    #     RoPE but the DeepSeek MLA path hardcodes is_neox_style=False (interleaved) -> scrambled
    #     positions -> alive-attention-but-incoherent. Force is_neox_style=True. ROOT-CAUSE fix #2.
    from . import glm_rope_neox
    glm_rope_neox.install()

    # 2g2) GLM-5.1 MLA decode KV-cache write->read fix for lazy HPU-graph capture
    #      (gated HPU_GLM_MLA_DECODE_CACHE_FIX; default ON under lazy when ROPE_NEOX=1).
    #      The residual lazy-decode drift after rope+routed-MoE are excluded: the decode
    #      cache index_copy_ WRITE (utils.py:75, indexed by per-step slot_mapping) and the
    #      contiguous-PA prefix-slice READ (ops.py:165) have NO data-flow edge, so under
    #      disable_tensor_cache=True the slot_mapping index can lose liveness on replay
    #      (same class as the rope cos/sin freeze) -> current token's self-KV mis-placed ->
    #      correct-then-degrading repetition. Fix threads the live slot_mapping + the
    #      returned cache handle as one dependency and clones the written value. Bit-
    #      identical numerics; eager never installs it. ROOT-CAUSE fix #3 (capture).
    from . import glm_mla_decode_cache_fix
    glm_mla_decode_cache_fix.install()

    # 2g2) DIAGNOSTIC (gated HPU_KV_WRITE_PROBE=N): HOST-side post-step KV-write probe —
    #     decides WRITE-side vs READ-side for the lazy decode drift with ZERO in-graph ops
    #     (the in-graph ring-buffer probe was eaten by the capture: index baked at capture
    #     -> replays wrote the capture slot -> drain read zeros). Inert unless gated on.
    from . import kv_write_probe
    kv_write_probe.install()

    # 2h) DIAGNOSTIC (gated HPU_LAYER_DUMP=1): dump layer-0 in/out tensors for a CPU
    #     reference comparison vs transformers GlmMoeDsaDecoderLayer. Read-only.
    from . import glm_layer_dump
    glm_layer_dump.install()

    # 2h2) DIAGNOSTIC (gated HPU_GLM_TRACE=1): LOUD, capture-SAFE per-step/per-layer
    #     decode tracer for the GLM-5.1 LAZY-capture accumulating-error bug (coherent
    #     ~8-10 tokens then greedy repetition). Logs one WARNING line per decode step:
    #     step idx, position, decode bucket bs, argmax token id + logit + gap-to-2nd,
    #     final-hidden L2, and ~3 sampled layers' attn-out + MoE-out L2. Reads ZERO
    #     device scalars mid-forward (no .item()/.cpu()/synchronize inside the captured
    #     region) -> never masks the lazy bug. Device ring-buffer written by forward
    #     hooks (copy_), dumped at the post-step sync that already exists
    #     (AsyncHPUModelRunnerOutput.get_output). No-op unless HPU_GLM_TRACE=1.
    from . import glm_lazy_trace
    glm_lazy_trace.install()

    # 2i) Post-load HPU move for the 8-bit dequant's deferred-on-CPU bf16 weights (lazy/:next).
    #     Under lazy the GPTQDequantToBF16 keeps bf16 on CPU during process_weights so it doesn't
    #     accumulate on HPU and OOM the next layer's unpack staging; this moves them onto HPU after
    #     load_model (quant freed -> room). UNCONDITIONAL (no-op if no layers were deferred).
    from . import deq_weight_move
    deq_weight_move.install()

    # 3) Register the full sparse path: indexer forward_oot op, indexer K-cache
    #    hook, gather-consume MLA, and the config-driven arch binding. The arch
    #    classes (GlmMoeDsaForCausalLM, DeepseekV3ForCausalLM) are already
    #    config-driven in vLLM core (is_v32 = hasattr(config, "index_topk")); the
    #    wiring re-binds them defensively and validates dims.
    from . import glm_moe_dsa_wire

    strict = glm_moe_dsa_wire.ready()  # HPU_DSA_READY=1 -> every component MUST load
    deferred = []
    for modname, attr, label in _SPARSE_COMPONENTS:
        err = _load_component(modname, attr, label, strict=strict)
        if err is not None:
            deferred.append(err)

    # 3b) MVP-3 SPARSE OP PATH (gated HPU_DSA_SPARSE=1, OFF by default). Installed AFTER
    #     the sparse components (so the indexer-cache/metadata runner patches are landed)
    #     AND after hpu_mla_skip_indexer (so this module can flip FULL_ATTENTION_FALLBACK
    #     back to False + un-skip the indexer under the gate). When the flag is UNSET,
    #     hpu_indexer_op_forward.register() is a NO-OP — the dense bypass is unchanged.
    #     It loads the validated .so (LOUD if missing) and swaps the indexer scorer +
    #     consume for the captured indexer_topk / sparse_attn kernels. Errors here are a
    #     HARD failure ONLY when the operator asked for the op path (HPU_DSA_SPARSE=1) —
    #     a broken/missing kernel must never silently fall back to dense.
    from . import dsa_ops
    if dsa_ops.sparse_enabled():
        from . import hpu_indexer_op_forward
        try:
            hpu_indexer_op_forward.register()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "[hpu-dsa-ext] HPU_DSA_SPARSE=1 but the sparse OP path failed to "
                f"install: {type(e).__name__}: {e}. Refusing to silently serve dense "
                "in place of the requested sparse path."
            ) from e

    # 4) Flip platform_patch.DSA_READY=True ONLY behind HPU_DSA_READY=1, and only
    #    after every component above registered (strict=True guarantees that). Until
    #    then DSA stays BLOCKED with a clear message — never silently dense.
    dsa_ready = glm_moe_dsa_wire.maybe_flip_ready()

    # 4b) MTP draft-acceptance probe (gated HPU_MTP_PROBE=1, default off): wraps
    #     HpuEagleProposer.propose to log draft in/out token ids — the decisive
    #     signal for the 0%-acceptance bug class (dead head vs identity vs misalign).
    from . import mtp_probe
    mtp_probe.install()

    # 5) Real-request diagnostic probe (gated by HPU_LAYER_PROBE env var, default off).
    #    Installed AFTER the full sparse path so it wraps the final wired forward methods.
    import os as _os
    if int(_os.environ.get("HPU_LAYER_PROBE", "0") or "0") > 0:
        from . import hidden_state_probe
        hidden_state_probe.install()

    # 6) Real-request activation probe (gated by HPU_REAL_PROBE=N, default 0=off).
    #    Unlike HPU_ATTN_PROBE/HPU_MOE_PROBE (which fire on warmup dummy passes), this
    #    probe gates on the model runner's warmup_mode flag and only logs REAL requests.
    #    It logs: embed output norm, per-layer (0-5, last-2) hidden state norms,
    #    and per-call q_norm / k_c_normed_norm / attn_out_norm for each forward_impl.
    #    Installed LAST so it wraps whatever forward_impl is active (sparse or dense).
    if int(_os.environ.get("HPU_REAL_PROBE", "0") or "0") > 0:
        from . import real_request_probe
        real_request_probe.register()

    # 6b) Capture-freeze probe (gated HPU_CAPTURE_PROBE=N, default 0=off). Confirms ON HW
    #     whether the lazy decode loop is a graph-capture position/rope FREEZE (the cos/sin
    #     class bug) vs a downstream numerical drift — via a deferred, one-step-late device
    #     readback that NEVER inserts a host sync into the captured decode graph (a sync would
    #     mask the freeze; see capture_freeze_probe.py + glm_rope_neox (B)). Installed LAST so
    #     it wraps the final wired rope forward_oot + forward_impl.
    if int(_os.environ.get("HPU_CAPTURE_PROBE", "0") or "0") > 0:
        from . import capture_freeze_probe
        capture_freeze_probe.register()

    # 6c) Per-DECODE-STEP boundary-tensor dump (gated HPU_DECODE_DUMP=N) for the
    #     LAZY-vs-EAGER value diff (PROJECT_FINDINGS §6c: lazy-mode execution numerics).
    #     Installed LAST so it wraps the FINAL installed forward_impl/forward_mqa.
    if int(_os.environ.get("HPU_DECODE_DUMP", "0") or "0") > 0:
        from . import decode_step_dump
        decode_step_dump.install()

    if not dsa_ready and deferred:
        # Skeleton load with siblings still being authored — surface what is pending
        # but do NOT fail (the platform patch keeps DSA correctly BLOCKED).
        import logging

        logging.getLogger(__name__).warning(
            "[hpu-dsa-ext] loaded with DSA BLOCKED (DSA_READY=False). Pending sparse "
            "components: %s. Set HPU_DSA_READY=1 only once all are registered.",
            "; ".join(deferred),
        )
