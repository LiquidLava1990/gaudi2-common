"""Whole-model fp8-OTHER graft for GLM-5.2 — the 3-tier mixed-precision LOAD dispatch.

Serves the GLM-5.2 scheme (user directive 2026-06-16): int4 routed experts + fp8 non-expert big
Linears + fp16 small/sensitive, all PRE-QUANTIZED on disk (no numeric conversion at load). This is the
PROVEN mtp_fp8_graft.py mechanism (live at GLM-5.1 ITER-4) WIDENED from the single MTP layer to the
whole main model.

WHAT (gate: HPU_FP8_OTHER_GRAFT=<dir with glm52_fp8_other.safetensors + index>, default off):
  1. Quant-method override: for each main-model (layers 0..num_hidden_layers-1) NON-expert,
     NON-router-gate, NON-indexer LinearBase, AutoRoundHPUConfig.get_quant_method returns vllm's fp8
     method (Fp8Config, is_checkpoint_fp8_serialized=True -> loads the serialized fp8 directly, NO
     re-quant). Routed experts (FusedMoE / .qweight) fall through to the ORIGINAL int4 method; the
     router gate, the DSA indexer, and norms fall through to their original (bf16/special) handling.
  2. Weight substitution: the main model's load_weights iterable is wrapped — the int4 checkpoint's
     bf16 copies of the fp8'd Linears are DROPPED, and the fp8 tensors (+ <name>.weight_scale) from the
     graft dir are injected under the same base names; fused modules (fused_qkv_a_proj, gate_up_proj)
     merge their fp8 constituents via vllm's MergedColumnParallel loader (proven at ITER-4).

Composes with mtp_fp8_graft (layer 78) — the two route disjoint layer ranges and chain get_quant_method.
LOUD: raises if the graft dir / index is missing. Card-free `--selftest` checks the routing + index
parsing (no torch, no vllm).
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("vllm")
_ENV = "HPU_FP8_OTHER_GRAFT"


# ==================================================================================================
# PURE routing core (card-free testable; no torch/vllm)
# ==================================================================================================
def fp8_weight_names_from_index(weight_map: dict) -> set[str]:
    """The set of '<module>.weight' names that were fp8'd (each has a sibling '<...>.weight_scale').

    Stage ② stores the fp8 weight under '<name>.weight' and the scale under '<name>.weight.weight_scale'
    (sibling-of-tensor). So a weight name N is fp8 iff (N + '.weight_scale') is in the map."""
    return {n for n in weight_map
            if n.endswith(".weight") and (n + ".weight_scale") in weight_map}


def route_is_fp8(prefix: str, num_hidden_layers: int) -> bool:
    """True iff the Linear at `prefix` should load via the vllm fp8 method (a main-model non-expert
    big Linear). False -> fall through to the original handling (int4 experts / bf16 gate-indexer-norm /
    the separate MTP graft for layer >= num_hidden_layers).

    Routing by EXCLUSION (mirrors quantize_glm52_fp8_other._selects_fp8):
      * routed expert  ('.mlp.experts.<d>.')         -> int4 (passthrough)
      * router gate    (endswith '.mlp.gate')        -> bf16 (passthrough)  [NOT '.mlp.gate_up_proj']
      * DSA indexer    ('.indexer')                  -> special/bf16 (passthrough)
      * MTP layer      (layer id >= num_hidden_layers)-> separate MTP graft (passthrough)
      * everything else LinearBase                   -> FP8  (incl. fused_qkv_a_proj, gate_up_proj,
                                                              q_b_proj, kv_b_proj, o_proj, down_proj,
                                                              shared_experts.*)
    """
    import re
    m = re.match(r"^model\.layers\.(\d+)\.", prefix)
    if m is not None and int(m.group(1)) >= num_hidden_layers:
        return False                                  # MTP layer -> mtp_fp8_graft, not here
    if re.search(r"\.mlp\.experts\.\d+\.", prefix) or prefix.endswith(".mlp.experts"):
        return False                                  # routed expert -> int4
    if prefix.endswith(".mlp.gate"):
        return False                                  # router gate -> bf16 (NOT gate_up_proj)
    if ".indexer" in prefix:
        return False                                  # DSA lightning indexer -> special/bf16
    return True


# ==================================================================================================
# install (gated; needs vllm)
# ==================================================================================================
def _graft_dir() -> str | None:
    d = os.environ.get(_ENV, "").strip()
    return d or None


def _num_hidden_layers() -> int:
    ckpt = os.environ.get("HPU_MOE_EXT_CKPT", "").strip()
    cfg_file = os.path.join(ckpt, "config.json")
    if not ckpt or not os.path.exists(cfg_file):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV} set but HPU_MOE_EXT_CKPT/config.json unreadable")
    c = json.load(open(cfg_file))
    n = int(c.get("text_config", c).get("num_hidden_layers", 0))
    if not n:
        raise RuntimeError("[hpu-moe-ext] checkpoint config has no num_hidden_layers")
    return n


def _fp8_config():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    return Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                     ignored_layers=None, weight_block_size=None)


def install() -> None:
    d = _graft_dir()
    if not d:
        return
    idx_file = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(idx_file):
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}={d} but {idx_file} is missing")
    wm = json.load(open(idx_file))["weight_map"]
    fp8_weight_names = fp8_weight_names_from_index(wm)
    if not fp8_weight_names:
        raise RuntimeError(f"[hpu-moe-ext] {_ENV}={d} index has NO fp8 weights (name+'.weight_scale')")
    n_hidden = _num_hidden_layers()

    # (1) quant-method override — CHAINS the current get_quant_method (composes w/ mtp_fp8_graft).
    from .auto_round_hpu import AutoRoundHPUConfig
    if not getattr(AutoRoundHPUConfig.get_quant_method, "_fp8_other", False):
        _orig_gqm = AutoRoundHPUConfig.get_quant_method
        _fp8cfg = _fp8_config()

        def _gqm(self, layer, prefix):
            from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
            if isinstance(layer, LinearBase) and route_is_fp8(prefix, n_hidden):
                m = _fp8cfg.get_quant_method(layer, prefix)
                return m if m is not None else UnquantizedLinearMethod()
            return _orig_gqm(self, layer, prefix)      # experts(int4) / gate-indexer-norm(bf16) / MTP

        _gqm._fp8_other = True
        AutoRoundHPUConfig.get_quant_method = _gqm

    # (2) weight substitution in the MAIN model loader (GlmMoeDsa -> DeepseekV2ForCausalLM).
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM as _M
    if not getattr(_M.load_weights, "_fp8_other", False):
        _orig_lw = _M.load_weights
        graft_names = [n for n in wm]   # all entries (fp8 weights + their scales)

        def _lw(self, weights):
            from safetensors import safe_open
            _handles: dict[str, object] = {}

            def _read(name):
                fn = wm[name]
                if fn not in _handles:
                    _handles[fn] = safe_open(os.path.join(d, fn), framework="pt", device="cpu")
                return _handles[fn].get_tensor(name)

            def _gen():
                dropped = 0
                for name, w in weights:
                    if name in fp8_weight_names:
                        dropped += 1
                        continue                       # drop the int4-ckpt bf16 copy; fp8 injected below
                    yield name, w
                logger.warning("[hpu-moe-ext] FP8-OTHER GRAFT: dropped %d int4-ckpt bf16 Linears; "
                               "injecting %d fp8 tensors (%d quantized) from %s",
                               dropped, len(graft_names), len(fp8_weight_names), d)
                for name in graft_names:
                    out = name
                    if name.endswith(".weight.weight_scale"):
                        out = name[: -len(".weight.weight_scale")] + ".weight_scale"
                    yield out, _read(name)

            return _orig_lw(self, _gen())

        _lw._fp8_other = True
        _M.load_weights = _lw

    logger.warning("[hpu-moe-ext] FP8-OTHER GRAFT armed: %d non-expert Linears load fp8 from %s "
                   "(experts stay int4, gate/indexer/norms bf16; NO conversion at load)",
                   len(fp8_weight_names), d)


# ==================================================================================================
# card-free self-test
# ==================================================================================================
def _selftest() -> int:
    nh = 78
    # routing: what MUST be fp8 vs passthrough on GLM-5.2's module names
    fp8_cases = [
        "model.layers.5.self_attn.fused_qkv_a_proj",   # fused q_a + kv_a -> fp8
        "model.layers.5.self_attn.q_b_proj",
        "model.layers.5.self_attn.kv_b_proj",
        "model.layers.5.self_attn.o_proj",
        "model.layers.5.mlp.gate_up_proj",             # fused dense/shared gate+up -> fp8
        "model.layers.0.mlp.down_proj",                # dense MLP -> fp8
        "model.layers.7.mlp.shared_experts.gate_up_proj",
        "model.layers.7.mlp.shared_experts.down_proj",
    ]
    passthrough_cases = [
        "model.layers.7.mlp.experts.0.gate_proj",      # routed expert -> int4
        "model.layers.7.mlp.experts.255.down_proj",
        "model.layers.7.mlp.gate",                     # router gate -> bf16
        "model.layers.5.self_attn.indexer.wk",         # DSA indexer -> special/bf16
        "model.layers.5.self_attn.indexer.weights_proj",
        "model.layers.78.self_attn.o_proj",            # MTP layer -> separate graft
        "model.layers.78.mlp.gate_up_proj",
    ]
    for p in fp8_cases:
        assert route_is_fp8(p, nh), f"expected FP8 routing for {p}"
    for p in passthrough_cases:
        assert not route_is_fp8(p, nh), f"expected PASSTHROUGH for {p}"

    # index parsing: a tiny weight_map with fp8 (weight + scale) + a kept fp16 (weight only)
    wm = {
        "model.layers.0.self_attn.o_proj.weight": "s.safetensors",
        "model.layers.0.self_attn.o_proj.weight.weight_scale": "s.safetensors",
        "model.layers.0.self_attn.indexer.wk.weight": "s.safetensors",   # kept fp16 (no scale)
    }
    got = fp8_weight_names_from_index(wm)
    assert got == {"model.layers.0.self_attn.o_proj.weight"}, got

    print("glm52_fp8_other_graft self-test: routing + index parsing OK "
          f"({len(fp8_cases)} fp8 / {len(passthrough_cases)} passthrough cases)")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("usage: python3 glm52_fp8_other_graft.py --selftest")
