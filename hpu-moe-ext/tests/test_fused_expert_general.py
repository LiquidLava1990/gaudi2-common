"""test_fused_expert_general.py -- GENERALIZE the OpBackend fused int4 MoE op
(``torch.ops.hpu_moe_ext.fused_expert``) ACROSS MoE MODEL SHAPE/GROUP PROFILES.

>>> RUN ON THE HPU BOX (the device-only run is mine). A CPU-only `py_compile`
    checks syntax; only the box exercises the graph. <<<

WHAT THIS PROVES
----------------
The committed validator ``../tpc_kernel/opbackend/test_fused_expert_op.py`` runs
ONE GLM profile (H=6144, I=2048, GS=128). The op's AddNode, however, derives
EVERY shape (M, H, I=2I/2, GS implicitly via scale.size(0)) from the input tensor
geometry -- it hardcodes NOTHING (see moe_fused_expert_backend.cpp:147-191, and
the dequant cguid infers grouped/non-grouped from scale.size(0) vs input.size(0)).
So a NEW MoE model should "just work" once its per-expert int4 weights are packed
in the proven PyTorch-row-major layout. This test exercises that claim on the
real shape/group matrix of the MoE families we serve / might serve:

  PROFILE              H      I     2I     GS    why it is in the matrix
  -------------------  -----  ----  -----  ----  -----------------------------------
  GLM-5.1              6144   2048  4096   128   the proven baseline (sanity anchor)
  DeepSeek-V3-ish      7168   2048  4096   128   wider hidden, MLA family, same GS
  Nemotron-ish         8192   2048  4096   128   widest hidden tested
  MiniMax-ish          6144   2048  4096    64   finer group size (GS=64)
  odd-width + GS=32    5120   1536  3072    32   non-power-of-2-friendly I + GS=32

Each profile asserts relNorm = ||got - ref|| / ||ref|| < 1e-2 vs the TRUSTED
``torch.ops.hpu.convert_from_uint4`` + MoE math reference (the exact oracle the
single-profile validator uses). LOUD (nonzero exit + stderr) on any failure;
a per-profile PASS/FAIL table + an overall verdict are printed.

The synthetic int4 weights, the packer, the sym-zero, and the reference are the
SAME proven helpers as the single-profile test (lifted verbatim so this adds NO
new numeric assumptions -- only new shapes).

CONSTRAINTS the op requires of each profile (asserted up front, skip-with-FAIL if
violated): H%GS==0, I%GS==0, (2I)%8==0, H%8==0, I%8==0 (the int4 pack is 8
nibbles/int32 along the OUT dim; the dequant groups along the IN dim by GS).

RUN (ON THE HPU BOX):
  cd $HOME/AI/hpu-moe-ext/tests
  python test_fused_expert_general.py            # all profiles, T=1 (single-stream)
  python test_fused_expert_general.py 4          # all profiles, T=4 (batch)
  python test_fused_expert_general.py 1 glm      # only the GLM profile (substring match)

The .so is located the same way the single-profile test does (built CPU-only by
  docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work \
    -w /work/opbackend vllm-gaudi:next bash build.sh
producing ../tpc_kernel/opbackend/build/libmoe_fused_expert.so).
"""
from __future__ import annotations

import os
import sys

# Locate the committed single-profile test so we reuse its PROVEN helpers
# (_pack_u4, _sym_zero_packed, _make_expert, _ref_expert, _relnorm) verbatim --
# this test introduces ZERO new numeric code, only new shape profiles.
_HERE = os.path.dirname(os.path.abspath(__file__))
_OPBACKEND = os.path.join(_HERE, "..", "tpc_kernel", "opbackend")
sys.path.insert(0, _OPBACKEND)


# ---- the model shape/group matrix -------------------------------------------
# (name, H, I, GS). 2I is the gate||up width (== w13 out); the op derives I=2I/2.
PROFILES = [
    ("GLM-5.1",          6144, 2048, 128),   # proven baseline anchor
    ("DeepSeek-V3-ish",  7168, 2048, 128),   # wider hidden, MLA family
    ("Nemotron-ish",     8192, 2048, 128),   # widest hidden
    ("MiniMax-ish",      6144, 2048,  64),   # finer group size GS=64
    ("odd-width+GS32",   5120, 1536,  32),   # odd-ish I=1536, GS=32
]


def _shape_ok(H: int, I: int, GS: int) -> tuple[bool, str]:
    """The op's hard layout invariants (int4 pack is 8/int32 along OUT, groups
    along IN by GS). Returns (ok, reason)."""
    if H % GS:
        return False, f"H({H}) % GS({GS}) != 0"
    if I % GS:
        return False, f"I({I}) % GS({GS}) != 0"
    if (2 * I) % 8:
        return False, f"(2I)({2 * I}) % 8 != 0"
    if H % 8:
        return False, f"H({H}) % 8 != 0"
    if I % 8:
        return False, f"I({I}) % 8 != 0"
    return True, ""


def _find_so() -> str:
    so = os.path.join(_OPBACKEND, "build", "libmoe_fused_expert.so")
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it CPU-only first:\n"
            f"  docker run --rm -v {os.path.join(_HERE, '..', 'tpc_kernel')}:/work "
            f"-w /work/opbackend vllm-gaudi:next bash build.sh")
    return so


def _run_profile(torch, htorch, helpers, name, H, I, GS, T, TOL) -> tuple[bool, str]:
    """Run ONE profile; return (passed, one-line summary). Uses the proven
    helpers from the single-profile test (lifted, not reimplemented)."""
    _make_expert = helpers["_make_expert"]
    _ref_expert = helpers["_ref_expert"]
    _relnorm = helpers["_relnorm"]

    ok, why = _shape_ok(H, I, GS)
    if not ok:
        return False, f"INVALID SHAPE for the op layout: {why}"

    dev = "hpu"
    OUT13 = 2 * I
    # Distinct seed per profile so we are not accidentally reusing one weight set.
    seed = 100 + abs(hash(name)) % 9000
    w13p, s13, z13, w2p, s2, z2 = _make_expert(H, I, GS, seed, dev)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    rw_val = 0.75
    rw = torch.full((1, 1), rw_val, dtype=torch.bfloat16, device=dev)
    htorch.mark_step()

    got = torch.ops.hpu_moe_ext.fused_expert(x, w13p, s13, z13, w2p, s2, z2, rw)
    htorch.mark_step()
    ref = _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I) * rw_val
    htorch.mark_step()
    gotc, refc = got.to("cpu"), ref.to("cpu")
    rn, ma = _relnorm(gotc, refc)
    passed = rn <= TOL
    summary = (f"H={H:<5} I={I:<5} (2I={OUT13:<5}) GS={GS:<4} T={T} | "
               f"|ref|={refc.float().norm().item():9.3f} |got|={gotc.float().norm().item():9.3f} "
               f"relNorm={rn:.3e} maxAbs={ma:.3e} -> {'PASS' if passed else 'FAIL'}")
    return passed, summary


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    only = sys.argv[2].lower() if len(sys.argv) > 2 else None  # substring filter

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.",
              file=sys.stderr)
        return 2

    # Import the proven helpers from the committed single-profile validator.
    try:
        import test_fused_expert_op as base  # from _OPBACKEND on sys.path
        helpers = {
            "_make_expert": base._make_expert,
            "_ref_expert": base._ref_expert,
            "_relnorm": base._relnorm,
        }
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: could not import proven helpers from "
              f"{_OPBACKEND}/test_fused_expert_op.py ({ex}).", file=sys.stderr)
        return 2

    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}", file=sys.stderr)
        return 2
    torch.ops.load_library(so)
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_expert"):
        print(f"FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after "
              f"load_library({so}).", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    TOL = 1e-2
    print(f"loaded {so}")
    print(f"=== test_fused_expert_general === T={T} TOL={TOL:.0e} "
          f"profiles={'all' if not only else only}")
    print("-" * 100)

    profiles = [p for p in PROFILES if (only is None or only in p[0].lower())]
    if not profiles:
        print(f"FATAL: no profile matches filter {only!r}; "
              f"have: {[p[0] for p in PROFILES]}", file=sys.stderr)
        return 2

    results = []
    for name, H, I, GS in profiles:
        try:
            passed, summary = _run_profile(torch, htorch, helpers, name, H, I, GS, T, TOL)
        except Exception as ex:  # noqa: BLE001
            passed, summary = False, f"EXCEPTION: {ex}"
        results.append((name, passed))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<18} {summary}")
        if not passed:
            print(f"        ^ device-confirmable suspects: gemm orientation, gate/up "
                  f"slice begin/end, dequant group inference (scale.size(0) vs in), the "
                  f"uint4 reinterpret_cast out attr, or the [1,1] router broadcast.",
                  file=sys.stderr)

    print("-" * 100)
    n_pass = sum(1 for _, p in results if p)
    n_tot = len(results)
    allok = n_pass == n_tot
    print(f"[verdict] {n_pass}/{n_tot} profiles PASS "
          f"({'GENERALIZES across the matrix' if allok else 'SHAPE-GENERALIZATION GAP -- see FAILs above'})")
    if not allok:
        failed = [name for name, p in results if not p]
        print(f"          FAILED: {failed}", file=sys.stderr)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
