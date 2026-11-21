"""Upstream-symbol guard.

This plugin reuses several upstream vLLM / vllm-gaudi internals by name. If a
future engine version renames or moves any of them, we want a LOUD, descriptive
failure at plugin registration (a build/gate failure) rather than a silent
wrong-answer or a confusing late crash. verify() is called first thing in
register().
"""

# (module path, attribute) pairs this plugin depends on unconditionally.
_REQUIRED = [
    ("vllm.model_executor.layers.quantization.moe_wna16", "MoeWNA16Method"),
    ("vllm.model_executor.layers.quantization.moe_wna16", "MoeWNA16Config"),
    ("vllm.model_executor.layers.quantization.utils.quant_utils",
     "pack_quantized_values_into_int32"),
    ("vllm.model_executor.layers.quantization.utils.quant_utils",
     "unpack_quantized_values_into_int32"),
    ("vllm_gaudi.extension.ops", "VllmMixtureOfExpertsOpWNA16"),
    ("vllm_gaudi.ops.hpu_gptq", "GPTQHPUConfig"),
    ("vllm_gaudi.ops.hpu_awq", "AWQHPUConfig"),
    ("vllm.model_executor.layers.quantization", "register_quantization_config"),
    ("vllm_gaudi.ops.hpu_gptq", "GPTQHPULinearMethod"),
]

# vLLM exposes two materially different fused-MoE APIs.  Older releases pass
# the monolithic FusedMoE layer to QuantizationConfig.get_quant_method; newer
# modular releases expose a FusedMoEFactory and pass its RoutedExperts weight
# container instead.  Treat these as live API capabilities, not model/version
# names, so one plugin can serve either stack.
_FUSED_MOE_API_VARIANTS = (
    (
        "legacy",
        (("vllm.model_executor.layers.fused_moe.layer", "FusedMoE"),),
    ),
    (
        "modular",
        (
            ("vllm.model_executor.layers.fused_moe", "FusedMoEFactory"),
            ("vllm.model_executor.layers.fused_moe", "RoutedExperts"),
        ),
    ),
)

_FUSED_MOE_LAYER_TYPES = (
    ("vllm.model_executor.layers.fused_moe.layer", "FusedMoE"),
    ("vllm.model_executor.layers.fused_moe", "RoutedExperts"),
    ("vllm.model_executor.layers.fused_moe.routed_experts", "RoutedExperts"),
    ("vllm.model_executor.layers.fused_moe", "SharedFusedMoE"),
)

_UNQUANTIZED_FUSED_MOE_METHODS = (
    (
        "vllm.model_executor.layers.fused_moe.layer",
        "UnquantizedFusedMoEMethod",
    ),
    (
        "vllm.model_executor.layers.fused_moe",
        "UnquantizedFusedMoEMethod",
    ),
    (
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
        "UnquantizedFusedMoEMethod",
    ),
)


def _load_symbol(mod_path: str, attr: str):
    import importlib

    try:
        module = importlib.import_module(mod_path)
    except Exception:  # noqa: BLE001
        return None
    return getattr(module, attr, None)


def fused_moe_api_variant() -> str | None:
    """Return the live fused-MoE API family, if one is complete."""

    for name, requirements in _FUSED_MOE_API_VARIANTS:
        if all(_load_symbol(mod_path, attr) is not None
               for mod_path, attr in requirements):
            return name
    return None


def fused_moe_layer_types() -> tuple[type, ...]:
    """Resolve the concrete expert containers accepted by quant dispatch."""

    resolved: list[type] = []
    for mod_path, attr in _FUSED_MOE_LAYER_TYPES:
        candidate = _load_symbol(mod_path, attr)
        if isinstance(candidate, type) and candidate not in resolved:
            resolved.append(candidate)
    if not resolved:
        raise RuntimeError(
            "hpu-moe-ext: fused-MoE API is present but exposes no concrete "
            "FusedMoE/RoutedExperts layer type")
    return tuple(resolved)


def is_fused_moe_layer(layer) -> bool:
    """Capability-driven routed-expert test shared by every quant path."""

    if isinstance(layer, fused_moe_layer_types()):
        return True
    # Preserve support for third-party wrappers without matching a model name.
    return (
        type(layer).__name__ in {"FusedMoE", "RoutedExperts", "SharedFusedMoE"}
        and hasattr(layer, "moe_config")
    )


def unquantized_fused_moe_method_cls():
    """Resolve the legacy or modular unquantized expert method class."""

    for mod_path, attr in _UNQUANTIZED_FUSED_MOE_METHODS:
        candidate = _load_symbol(mod_path, attr)
        if isinstance(candidate, type):
            return candidate
    raise RuntimeError(
        "hpu-moe-ext: no UnquantizedFusedMoEMethod implementation is exported")


def verify() -> None:
    import importlib
    missing = []
    for mod_path, attr in _REQUIRED:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod_path} (import failed: {e})")
            continue
        if not hasattr(mod, attr):
            missing.append(f"{mod_path}.{attr}")
    if fused_moe_api_variant() is None:
        variants = []
        for name, requirements in _FUSED_MOE_API_VARIANTS:
            symbols = ", ".join(f"{module}.{attr}" for module, attr in requirements)
            variants.append(f"{name}: {symbols}")
        missing.append("one complete fused-MoE API variant (" + "; ".join(variants) + ")")
    try:
        fused_moe_layer_types()
    except RuntimeError as exc:
        missing.append(str(exc))
    try:
        unquantized_fused_moe_method_cls()
    except RuntimeError as exc:
        missing.append(str(exc))
    if missing:
        raise RuntimeError(
            "hpu-moe-ext: required upstream symbols are missing — the engine "
            "version likely changed. Update/retire the plugin. Missing:\n  "
            + "\n  ".join(missing))
