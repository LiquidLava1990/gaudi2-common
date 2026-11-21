"""Loud drift-guard. Fail at registration (boot/build time) if any borrowed
upstream symbol moved — never silently produce a wrong (dense) answer.

Every symbol this overlay monkeypatches or subclasses is enumerated here. On a
vllm / vllm-gaudi bump, a moved symbol becomes a RuntimeError at startup, the
mandate's requirement (loud on drift)."""

# (module_path, attribute) pairs this overlay depends on.
_REQUIRED = [
    ("vllm_gaudi.platform", "HpuPlatform"),
    ("vllm.model_executor.layers.sparse_attn_indexer", "SparseAttnIndexer"),
    ("vllm_gaudi.attention.oot_mla", "HPUMLAAttention"),
    ("vllm_gaudi.attention.oot_mla", "HPUMultiHeadLatentAttentionWrapper"),
    ("vllm_gaudi.attention.backends.hpu_attn", "HPUMLAAttentionBackend"),
]


def verify():
    import importlib

    missing = []
    for mod, attr in _REQUIRED:
        try:
            m = importlib.import_module(mod)
            if not hasattr(m, attr):
                missing.append(f"{mod}.{attr} (attribute missing)")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod} (import failed: {e})")

    # The platform must STILL hard-raise on sparse — that is the thing we patch.
    # If upstream already handles sparse, our patch is stale and must be reviewed.
    try:
        import inspect
        from vllm_gaudi.platform import HpuPlatform

        src = inspect.getsource(HpuPlatform.get_attn_backend_cls.__func__)
        if "use_sparse" not in src:
            missing.append(
                "HpuPlatform.get_attn_backend_cls lost its use_sparse branch "
                "(upstream may now handle sparse — review platform_patch)"
            )
    except Exception as e:  # noqa: BLE001
        missing.append(f"get_attn_backend_cls introspection failed: {e}")

    # MVP-3: the op path monkeypatches Indexer.forward (vllm core deepseek_v2). Guard
    # that symbol ONLY when the op path is requested (HPU_DSA_SPARSE=1) so the default
    # serve does not depend on it. Loud on drift, never silent.
    import os
    if os.environ.get("HPU_DSA_SPARSE", "0").strip().lower() in ("1", "true", "yes", "on"):
        try:
            m = importlib.import_module("vllm.model_executor.models.deepseek_v2")
            if not hasattr(m, "Indexer"):
                missing.append(
                    "vllm.model_executor.models.deepseek_v2.Indexer (gone — the DSA "
                    "op path patches Indexer.forward; review hpu_indexer_op_forward)")
        except Exception as e:  # noqa: BLE001
            missing.append(f"deepseek_v2.Indexer introspection failed: {e}")

    if missing:
        raise RuntimeError(
            "[hpu-dsa-ext] upstream drift detected; refusing to load to avoid a "
            "silently-wrong (dense) DSA path. Missing/changed:\n  - "
            + "\n  - ".join(missing)
        )
