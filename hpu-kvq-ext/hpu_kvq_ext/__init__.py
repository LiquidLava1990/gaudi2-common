"""hpu-kvq-ext: TurboQuant sub-fp8 KV cache (k8v4 = fp8-K + int4-V, Walsh-Hadamard,
calibration-free) for vllm-gaudi. GQA/MHA only -- MLA-latent models stay fp8_inc
(int4-latent is measured-dead, ~4-5x fp8 error). Loads via the vllm.general_plugins
entry-point '06.hpu_kvq_ext' (after stock vllm-gaudi, hpu-moe-ext '04.', hpu-dsa-ext '05.').

STATUS: gated. ``register()`` ALWAYS installs the platform gate extension (inert --
still refuses the turboquant dtypes until wired+ready), and ONLY when HPU_KVQ_READY=1
does it load the two compiled ops (kv_rotate_quant + kv_dequant_unrotate), install the
cache-spec/allocation + attention store/read wiring, and flip platform_patch_kvq.KVQ_READY.
Until then every serve is byte-for-byte the stock fp8_inc path. Mirrors hpu_dsa_ext.register
/ the DSA_READY discipline (never serve a KV dtype nothing can write/read).

Components (installed in dependency order when HPU_KVQ_READY=1):
  1. kvq_ops.load_ops()          -- load libkv_rotate_quant{,_lazy}.so + libkv_dequant_unrotate{,_lazy}.so
  2. kvq_cache_hpu.apply()       -- STR_DTYPE registration + get_kv_cache_spec (FullAttentionSpec
                                    -> TurboQuantKVSpec, never MLA) + initialize_kv_cache (4
                                    asymmetric planes: fp8 K, int4 V, 2 f32 scales)
  3. kvq_attn_wire.register()    -- HPUAttentionImpl store->kv_rotate_quant / read->kv_dequant_unrotate
"""


def register():
    from . import platform_patch_kvq as pp

    # 1) Platform gate extension. Inert: while KVQ_READY/WIRE_INSTALLED are False (or
    #    HPU_KVQ_READY unset) it still returns the stock 'fp8_inc only', and it NEVER
    #    accepts a turboquant dtype for an MLA model. Safe to install unconditionally.
    pp.apply()

    # 1b) MLA int4 latent store (GLM/DeepSeek). Independent of the GQA turboquant gate: it runs UNDER
    #     a normal --kv-cache-dtype fp8_inc MLA serve (reroutes the latent alloc + store/read to int4),
    #     so it activates on HPU_MLA_KV_INT4STORE alone. Default off -> byte-identical fp8_inc MLA.
    from . import kvq_mla_wire
    if kvq_mla_wire.int4_store_enabled():
        from . import kvq_cache_hpu
        kvq_cache_hpu.apply()          # patch_allocation's MLA branch keys on HPU_MLA_KV_INT4STORE
        kvq_cache_hpu.patch_mla_shrink()   # Stage 2 (HPU_MLA_KV_SHRINK): smaller page_size -> more blocks
        kvq_mla_wire.register()        # swaps HPUMLAImpl.latent_cache_k -> _TQLatentCache
        import logging
        logging.getLogger("vllm").info(
            "[hpu-kvq-ext] MLA int4 latent store ACTIVE (HPU_MLA_KV_INT4STORE=1; fp8_inc serve)")

    # 2) Default-inert: without the explicit HPU_KVQ_READY=1 opt-in we install NOTHING
    #    else (no .so load, no runner/attention patches). The serve is unchanged.
    if not pp.ready():
        return

    # 3) Operator opted in (HPU_KVQ_READY=1): install the full write/read path. LOUD on
    #    any failure -- the flag was explicitly set, so a silent fallback would hide a
    #    broken build (never serve a KV dtype nothing can write/read).
    from . import kvq_ops
    kvq_ops.load_ops()

    from . import kvq_cache_hpu
    kvq_cache_hpu.apply()

    from . import kvq_attn_wire
    kvq_attn_wire.register()

    # 4) Mark the write/read wiring installed, then flip KVQ_READY (ready() is already
    #    True). Only now does the platform gate ACCEPT --kv-cache-dtype turboquant_k8v4
    #    (for a non-MLA model).
    pp.WIRE_INSTALLED = True
    pp.maybe_flip_ready()

    import logging
    logging.getLogger("vllm").info(
        "[hpu-kvq-ext] TurboQuant KV wired + READY (KVQ_READY=%s). Accepted dtypes: %s "
        "(GQA/MHA only; MLA stays fp8_inc).", pp.KVQ_READY, ", ".join(pp.KVQ_DTYPES))
