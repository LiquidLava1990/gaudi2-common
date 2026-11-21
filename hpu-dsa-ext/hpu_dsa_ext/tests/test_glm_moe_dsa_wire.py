"""CPU unit tests for glm_moe_dsa_wire (no torch / no HPU / no real vllm).

Covers the pure-Python contract surface:
  * config_is_sparse / read_dsa_dims / assert_sparse_config (GLM=32, DeepSeek=64 dims)
  * DsaDims.layer_is_dense / layer_skips_topk (first_k_dense_replace + skip_topk)
  * ready()/maybe_flip_ready() env gating (HPU_DSA_READY)
  * register_arch() binding logic (against a stub ModelRegistry + stub in-tree module)
  * launch docs invariants (TP8, EP, no prefix cache, PT_HPU_LAZY_MODE=0)

Run: python3 hpu_dsa_ext/tests/test_glm_moe_dsa_wire.py
"""
import os
import sys
import types

# Import the module under test directly by file path (avoids importing the package
# __init__, which pulls in vllm). This keeps the test torch/vllm-free.
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_WIRE = os.path.normpath(os.path.join(_HERE, "..", "glm_moe_dsa_wire.py"))
_spec = importlib.util.spec_from_file_location("glm_moe_dsa_wire", _WIRE)
w = importlib.util.module_from_spec(_spec)
sys.modules["glm_moe_dsa_wire"] = w  # py3.14 dataclasses resolves annotations via sys.modules
_spec.loader.exec_module(w)


class Cfg:
    """Minimal stand-in for an hf_config (attribute bag)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def glm_config():
    # GLM-5.1: 32 indexer heads, topk 2048, dense layers 0..2
    return Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
               qk_rope_head_dim=64, first_k_dense_replace=3,
               indexer_rope_interleave=True)


def deepseek_config():
    # DeepSeek-V3.2: 64 indexer heads (the cross-model reuse check)
    return Cfg(index_topk=2048, index_n_heads=64, index_head_dim=128,
               qk_rope_head_dim=64, first_k_dense_replace=3,
               indexer_rope_interleave=False)


def t_config_is_sparse():
    assert w.config_is_sparse(glm_config())
    assert w.config_is_sparse(deepseek_config())
    assert not w.config_is_sparse(Cfg())                       # plain dense
    assert not w.config_is_sparse(Cfg(index_topk=0))           # present but falsy
    assert not w.config_is_sparse(Cfg(index_topk=None))


def t_read_dims_glm_vs_deepseek():
    g = w.read_dsa_dims(glm_config())
    assert (g.index_topk, g.index_n_heads, g.index_head_dim) == (2048, 32, 128)
    assert g.first_k_dense_replace == 3 and g.indexer_rope_interleave is True
    d = w.read_dsa_dims(deepseek_config())
    assert d.index_n_heads == 64 and d.indexer_rope_interleave is False
    # nothing hardcoded: heads differ purely from config
    assert g.index_n_heads != d.index_n_heads


def t_read_dims_rejects_non_dsa_and_partial():
    try:
        w.read_dsa_dims(Cfg())
        assert False, "expected ValueError on non-DSA config"
    except ValueError:
        pass
    # has index_topk but missing required dims -> loud
    try:
        w.read_dsa_dims(Cfg(index_topk=2048))
        assert False, "expected ValueError on partial DSA config"
    except ValueError as e:
        assert "missing required field" in str(e)


def t_assert_sparse_config():
    dims = w.assert_sparse_config(glm_config())
    assert dims.index_topk == 2048
    for bad in (Cfg(index_topk=2048, index_n_heads=0, index_head_dim=128,
                    qk_rope_head_dim=64, first_k_dense_replace=3),
                Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
                    qk_rope_head_dim=64, first_k_dense_replace=-1)):
        try:
            w.assert_sparse_config(bad)
            assert False, "expected ValueError on invalid dims"
        except ValueError:
            pass


def t_layer_dense_and_skip():
    d = w.read_dsa_dims(glm_config())            # first_k_dense_replace=3, freq=1
    assert d.layer_is_dense(0) and d.layer_is_dense(2)
    assert not d.layer_is_dense(3)
    # freq=1 -> no sparse layer skips topk
    assert not d.layer_skips_topk(5)
    assert not d.layer_skips_topk(1)             # dense -> not "skip", it's dense
    # freq cadence: every-other sparse layer reuses prior buffer
    d2 = w.read_dsa_dims(Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
                             qk_rope_head_dim=64, first_k_dense_replace=3,
                             index_topk_freq=2))
    # layer (id-1) % 2 != 0 -> skip. id=4 -> 3%2=1 -> skip; id=5 -> 4%2=0 -> compute
    assert d2.layer_skips_topk(4) and not d2.layer_skips_topk(5)
    # explicit pattern wins
    pat = ["_"] * 6
    pat[4] = "S"
    d3 = w.read_dsa_dims(Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
                             qk_rope_head_dim=64, first_k_dense_replace=3,
                             index_topk_pattern=pat))
    assert d3.layer_skips_topk(4) and not d3.layer_skips_topk(5)


# ---- GLM-5.2 IndexShare cadence (the net-new 5.2 behaviour) -----------------
def _hf_indexer_types(num_layers, freq, offset):
    """The HF configuration_glm_moe_dsa.__post_init__ generator (ground truth):
    full iff max(i - offset + 1, 0) % freq == 0."""
    return ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
            for i in range(num_layers)]


def glm52_config(explicit=False):
    """GLM-5.2: 78 layers, IndexShare freq=4 / offset=3 (vs GLM-5.1 freq=1).

    explicit=True attaches the shipped ``indexer_types`` array (what 5.2 ships);
    explicit=False exercises the freq/offset generator fallback.
    """
    kw = dict(index_topk=2048, index_n_heads=32, index_head_dim=128,
              qk_rope_head_dim=64, first_k_dense_replace=3,
              indexer_rope_interleave=True, index_topk_freq=4,
              index_skip_topk_offset=3, num_hidden_layers=78,
              index_topk_pattern=None)
    if explicit:
        kw["indexer_types"] = _hf_indexer_types(78, 4, 3)
    return Cfg(**kw)


def t_glm52_indexshare_cadence_from_formula():
    d = w.read_dsa_dims(glm52_config(explicit=False))
    assert d.indexer_schedule() == tuple(_hf_indexer_types(78, 4, 3)), \
        "5.2 freq/offset cadence != HF __post_init__ generator"
    # 'F F F | S S S F' — full at 0,1,2 then every 4th starting at 6, through 76
    full = d.full_indexer_layers()
    assert full == (0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54,
                    58, 62, 66, 70, 74), full
    assert len(full) == 21 and (78 - len(full)) == 57          # 21 full / 57 shared
    assert d.indexer_schedule()[:7] == ("full", "full", "full",
                                        "shared", "shared", "shared", "full")
    assert all(d.layer_runs_indexer(i) for i in (0, 1, 2))     # seed band == dense band


def t_glm52_offset_is_honored():
    # The OLD upstream formula baked in offset=2 (max(i-1,0)%freq); 5.2 needs offset=3.
    d3 = w.read_dsa_dims(glm52_config(explicit=False))                 # offset 3
    d2 = w.read_dsa_dims(Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
                             qk_rope_head_dim=64, first_k_dense_replace=3,
                             index_topk_freq=4, index_skip_topk_offset=2,
                             num_hidden_layers=78))
    # offset=3 -> full at 6 (not 5); offset=2 -> full at 5 (not 6). They MUST differ.
    assert d3.layer_runs_indexer(6) and not d3.layer_runs_indexer(5)
    assert d2.layer_runs_indexer(5) and not d2.layer_runs_indexer(6)
    assert d3.indexer_schedule() != d2.indexer_schedule()


def t_glm52_reuse_source_mapping():
    d = w.read_dsa_dims(glm52_config(explicit=False))
    assert d.reuse_source(2) == 2 and d.reuse_source(6) == 6     # full -> self
    assert d.reuse_source(3) == 2 and d.reuse_source(5) == 2     # shared -> prior full
    assert d.reuse_source(7) == 6 and d.reuse_source(9) == 6
    assert d.reuse_source(10) == 10 and d.reuse_source(11) == 10
    d.assert_indexshare_wellformed()        # no shared layer reuses an unwritten buffer


def t_glm52_explicit_indexer_types_wins_and_agrees():
    # explicit indexer_types honored verbatim AND equal to the freq/offset fallback
    d = w.read_dsa_dims(glm52_config(explicit=True))
    assert d.indexer_schedule() == tuple(_hf_indexer_types(78, 4, 3))
    d_formula = w.read_dsa_dims(glm52_config(explicit=False))
    assert d.indexer_schedule() == d_formula.indexer_schedule()
    # an explicit array OVERRIDES the formula (force layer 7 'full' -> honored)
    arr = list(_hf_indexer_types(78, 4, 3))
    arr[7] = "full"
    d_ov = w.read_dsa_dims(Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
                               qk_rope_head_dim=64, first_k_dense_replace=3,
                               index_topk_freq=4, index_skip_topk_offset=3,
                               num_hidden_layers=78, indexer_types=arr))
    assert d_ov.layer_runs_indexer(7) and not d_formula.layer_runs_indexer(7)


def t_glm52_illformed_cadence_is_loud():
    # a 'shared' layer 0 (no seed) must fail loud, never reuse an unwritten buffer
    arr = _hf_indexer_types(78, 4, 3)
    arr[0] = "shared"
    d = w.read_dsa_dims(Cfg(index_topk=2048, index_n_heads=32, index_head_dim=128,
                            qk_rope_head_dim=64, first_k_dense_replace=3,
                            index_topk_freq=4, index_skip_topk_offset=3,
                            num_hidden_layers=78, indexer_types=arr))
    try:
        d.assert_indexshare_wellformed()
        assert False, "expected ValueError on a shared layer 0 (no seed)"
    except ValueError:
        pass


def t_shared_token_parsing():
    f = w.DsaDims._is_shared_token
    assert f("shared") and f("S") and f("s")
    assert not f("full") and not f("F") and not f("_") and not f("")
    try:
        f("bogus")
        assert False, "expected ValueError on unknown cadence token"
    except ValueError:
        pass


def t_ready_env_gating():
    saved = os.environ.get("HPU_DSA_READY")
    try:
        for v, exp in (("0", False), ("", False), ("1", True), ("true", True),
                       ("YES", True), ("on", True), ("nope", False)):
            if v == "":
                os.environ.pop("HPU_DSA_READY", None)
            else:
                os.environ["HPU_DSA_READY"] = v
            assert w.ready() is exp, (v, exp)
    finally:
        if saved is None:
            os.environ.pop("HPU_DSA_READY", None)
        else:
            os.environ["HPU_DSA_READY"] = saved


def t_maybe_flip_ready():
    # stub the package-relative `platform_patch` import
    pkg = types.ModuleType("glm_moe_dsa_wire_pkg")
    # maybe_flip_ready does `from . import platform_patch`; emulate by injecting a
    # fake parent package + submodule into sys.modules and rebinding __package__.
    parent = "hpu_dsa_ext_stub"
    sys.modules[parent] = types.ModuleType(parent)
    pp = types.ModuleType(parent + ".platform_patch")
    pp.DSA_READY = False
    sys.modules[parent + ".platform_patch"] = pp
    saved_pkg = w.__package__
    saved_env = os.environ.get("HPU_DSA_READY")
    try:
        w.__package__ = parent
        os.environ.pop("HPU_DSA_READY", None)
        assert w.maybe_flip_ready() is False and pp.DSA_READY is False
        os.environ["HPU_DSA_READY"] = "1"
        assert w.maybe_flip_ready() is True and pp.DSA_READY is True
    finally:
        w.__package__ = saved_pkg
        if saved_env is None:
            os.environ.pop("HPU_DSA_READY", None)
        else:
            os.environ["HPU_DSA_READY"] = saved_env
        sys.modules.pop(parent + ".platform_patch", None)
        sys.modules.pop(parent, None)


def t_register_arch_binds_all():
    # Stub vllm.model_executor.models.registry.ModelRegistry + the in-tree module.
    calls = []

    class FakeRegistry:
        @staticmethod
        def register_model(name, target):
            calls.append((name, target))

    reg_mod = types.ModuleType("vllm.model_executor.models.registry")
    reg_mod.ModelRegistry = FakeRegistry

    # Fake in-tree deepseek_v2 module: arch classes subclassing the base (the
    # structural drift-guard register_arch enforces).
    ds_mod = types.ModuleType(w._DEEPSEEK_V2_MODULE)

    class DeepseekV2ForCausalLM:  # noqa: D401
        pass

    class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
        pass

    class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
        pass

    for cls in (DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM):
        cls.__module__ = w._DEEPSEEK_V2_MODULE
        setattr(ds_mod, cls.__name__, cls)

    saved = {}
    for m in ("vllm", "vllm.model_executor", "vllm.model_executor.models",
              "vllm.model_executor.models.registry", w._DEEPSEEK_V2_MODULE):
        saved[m] = sys.modules.get(m)
    try:
        # minimal package chain so importlib.import_module finds our stubs
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.model_executor"] = types.ModuleType("vllm.model_executor")
        sys.modules["vllm.model_executor.models"] = types.ModuleType("vllm.model_executor.models")
        sys.modules["vllm.model_executor.models.registry"] = reg_mod
        sys.modules[w._DEEPSEEK_V2_MODULE] = ds_mod
        bound = w.register_arch()
        assert set(bound) == set(w.DSA_ARCHES), bound
        names = {n for n, _ in calls}
        assert names == set(w.DSA_ARCHES), names
        # every binding points back at the canonical in-tree module
        for _, target in calls:
            assert target.startswith(w._DEEPSEEK_V2_MODULE + ":")
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v


def t_register_arch_drift_guard():
    # An arch that is NOT a subclass of the base must raise (loud on drift).
    calls = []

    class FakeRegistry:
        @staticmethod
        def register_model(name, target):
            calls.append((name, target))

    reg_mod = types.ModuleType("vllm.model_executor.models.registry")
    reg_mod.ModelRegistry = FakeRegistry
    ds_mod = types.ModuleType(w._DEEPSEEK_V2_MODULE)

    class DeepseekV2ForCausalLM:
        pass

    class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
        pass

    class GlmMoeDsaForCausalLM:  # BROKEN: not a subclass
        pass

    for cls in (DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM):
        cls.__module__ = w._DEEPSEEK_V2_MODULE
        setattr(ds_mod, cls.__name__, cls)

    saved = {}
    for m in ("vllm", "vllm.model_executor", "vllm.model_executor.models",
              "vllm.model_executor.models.registry", w._DEEPSEEK_V2_MODULE):
        saved[m] = sys.modules.get(m)
    try:
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.model_executor"] = types.ModuleType("vllm.model_executor")
        sys.modules["vllm.model_executor.models"] = types.ModuleType("vllm.model_executor.models")
        sys.modules["vllm.model_executor.models.registry"] = reg_mod
        sys.modules[w._DEEPSEEK_V2_MODULE] = ds_mod
        try:
            w.register_arch()
            assert False, "expected RuntimeError on non-subclass arch"
        except RuntimeError as e:
            assert "no longer subclasses" in str(e)
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v


def t_launch_docs_invariants():
    assert w.LAUNCH_ENV["PT_HPU_LAZY_MODE"] == "0"
    assert w.LAUNCH_ENV["HPU_DSA_READY"] == "1"
    flags = w.LAUNCH_FLAGS
    assert "--tensor-parallel-size" in flags and "8" in flags
    assert "--enable-expert-parallel" in flags
    assert "--no-enable-prefix-caching" in flags
    txt = w.launch_help()
    for needle in ("GLM-5.1", "CONFIG-DRIVEN", "Prefix caching MUST be off",
                   "DeepSeek-V3.2/V4", "awq_hpu", "gptq"):
        assert needle in txt, needle


def t_register_returns_summary():
    # `register` calls register_arch; reuse the stubbing from t_register_arch_binds_all
    # by monkeypatching register_arch to avoid re-stubbing vllm here.
    saved = w.register_arch
    try:
        w.register_arch = lambda: list(w.DSA_ARCHES)
        out = w.register()
        assert out["bound_arches"] == list(w.DSA_ARCHES)
        assert out["dsa_ready_env"] == "HPU_DSA_READY"
        assert "ready" in out
    finally:
        w.register_arch = saved


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("t_") and callable(v)]
    passed = 0
    for fn in tests:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(tests)} passed")


if __name__ == "__main__":
    main()
