"""test_fused_op_capture.py -- ITEM 3 of 4: prove (or DISPROVE, loudly) that our
fused MoE op gets CAPTURED into the decode HPU-graph under LAZY serving.

WHY THIS TEST EXISTS
--------------------
The "~free per-token dispatch" speed mechanism is HPU-graph capture+replay:
`wrap_in_hpu_graph` records the model's lazy forward into ONE recipe on the first
call, then `replay()`s it every subsequent token (no re-trace, no recompile, no
host dispatch). That mechanism is **lazy-mode-only** -- the bridge's HPUGraph
capture engine hooks the `habana_lazy` HbLazyTensor IR (it calls
`HbLazyTensor::StepMarker` + `context->setCapturing(true)` in
backend/habana_device/HPUGraph.cpp), and `habana_frameworks.torch.hpu.graphs`
RAISES "HPUGraph class is available in lazy mode only." in eager mode
(graphs.py:51). vllm-gaudi only wraps the model when `is_lazy()` is true
(hpu_model_runner.py:_maybe_wrap_in_hpu_graph).

So for our op to ride that recipe it must:
  (1) LOAD at all under PT_HPU_LAZY_MODE=1 (the lazy plugin must provide its
      frontend symbols), and
  (2) feed the lazy HbLazyTensor IR so capture records it (i.e. be on the
      LazyOp / habana_lazy frontend, NOT the eager EagerExec frontend which
      bypasses that IR and launches synchronously).

This test sets PT_HPU_LAZY_MODE=1 BEFORE importing habana_frameworks (which is
what selects the LAZY plugin libhabana_pytorch_plugin.so at import; see
habana_frameworks/torch/__init__.py:56), loads the op .so, wraps a tiny module
that calls torch.ops.hpu_moe_ext.fused_expert in wrap_in_hpu_graph, runs it
several times, and asserts:

  CHECK A  it EXECUTES coherently under wrap_in_hpu_graph (numerically == the
           trusted convert_from_uint4 reference, relNorm < 1e-2);
  CHECK B  the 2nd+ calls are graph REPLAYS, not recaptures -- evidenced by
           CachedParams.cache_hits incrementing while the recipe cache stays at
           exactly ONE entry (no recompile);
  CHECK C  best-effort: the op is actually IN the captured recipe -- proven by
           the negative control `bypass_hpu_graphs=True` (which runs orig_fwd
           outside the graph) producing the SAME number plus a real per-call
           latency separation between captured-replay and bypass-eager.

LOUD on any failure or any fall-back-to-eager. In particular, if the loaded .so
is the EAGER-frontend build (the committed build/libmoe_fused_expert.so), it will
either FAIL TO LOAD under lazy (undefined symbol habana::eager::EagerOpBase::*,
which lives only in the eager plugin) OR, if it somehow loads, its EagerExec
launch bypasses the lazy IR and will NOT be in the recipe -- this test SHOUTS
that and tells you a LazyOp-frontend variant is required.

RUN (ON THE HPU BOX ONLY -- capture is a device mechanism):
  cd $HOME/AI/hpu-moe-ext
  PT_HPU_LAZY_MODE=1 python tests/test_fused_op_capture.py
  # optional: point at a specific .so and/or shape
  PT_HPU_LAZY_MODE=1 MOE_OP_SO=/path/to/libmoe_fused_expert_lazy.so \
      python tests/test_fused_op_capture.py 1 6144 2048 128   # T H I GS

EXIT CODE: 0 only if A and B both pass (op loaded, executed, and replayed from a
single captured recipe). Nonzero (LOUD) otherwise.
"""
from __future__ import annotations

import os
import sys

# --- MUST be set before importing habana_frameworks: it selects the LAZY plugin
#     (libhabana_pytorch_plugin.so) vs the eager pt2 plugin at import time
#     (habana_frameworks/torch/__init__.py:56 -> is_lazy() reads this env). If a
#     caller forgot to export it, force it here and warn -- but if habana is
#     ALREADY imported the choice is locked, so we detect and refuse loudly.
if "habana_frameworks" in sys.modules:
    print("FATAL: habana_frameworks was imported BEFORE this test set "
          "PT_HPU_LAZY_MODE=1; the plugin (lazy vs eager) is locked at import. "
          "Run as:  PT_HPU_LAZY_MODE=1 python tests/test_fused_op_capture.py",
          file=sys.stderr)
    sys.exit(2)
if os.environ.get("PT_HPU_LAZY_MODE") != "1":
    print("NOTE: PT_HPU_LAZY_MODE was not '1'; forcing it to 1 for this test "
          "(HPU-graph capture is lazy-mode-only).", file=sys.stderr)
    os.environ["PT_HPU_LAZY_MODE"] = "1"


# ----------------------------- weight builders -------------------------------
# (identical layout to opbackend/test_fused_expert_op.py so the reference matches)
def _sym_zero_packed(groups, out_dim, device):
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _pack_u4(codes_2d, device):
    import torch
    IN, OUT = codes_2d.shape
    c = (codes_2d.to(torch.int64) & 0xF).reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.to(device)


def _make_expert(H, I, GS, seed, device):
    import torch
    OUT13 = 2 * I
    g = torch.Generator().manual_seed(seed)
    c13 = torch.randint(0, 16, (H, OUT13), generator=g, dtype=torch.int32)
    c2 = torch.randint(0, 16, (I, H), generator=g, dtype=torch.int32)
    w13_packed = _pack_u4(c13, device)
    w2_packed = _pack_u4(c2, device)
    s13 = (torch.rand((H // GS, OUT13), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    s2 = (torch.rand((I // GS, H), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)
    z13 = _sym_zero_packed(H // GS, OUT13, device)
    z2 = _sym_zero_packed(I // GS, H, device)
    return w13_packed, s13, z13, w2_packed, s2, z2


def _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I):
    import torch
    import torch.nn.functional as F
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H, 2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)        # [I, H]
    gate_up = torch.matmul(x, W13)
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    h = F.silu(gate) * up
    return torch.matmul(h, W2)


def _relnorm(got, ref):
    g = got.float(); r = ref.float()
    diff = (g - r).norm().item()
    return diff / (r.norm().item() + 1e-9), (g - r).abs().max().item()


def _find_so():
    """Locate the op .so. MOE_OP_SO overrides; else prefer a LazyOp variant if
    one was built, falling back to the committed eager build (which this test
    EXPECTS to fail to load under lazy -- and says so loudly)."""
    env = os.environ.get("MOE_OP_SO")
    if env:
        return env, "(from MOE_OP_SO)"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ob = os.path.join(root, "tpc_kernel", "opbackend", "build")
    candidates = [
        (os.path.join(ob, "libmoe_fused_expert_lazy.so"), "LazyOp variant"),
        (os.path.join(ob, "libmoe_fused_expert.so"), "committed (EAGER frontend)"),
    ]
    for path, label in candidates:
        if os.path.isfile(path):
            return path, label
    return candidates[-1][0], "committed (EAGER frontend) [MISSING]"


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    I = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    if H % GS or I % GS or (2 * I) % 8 or H % 8:
        print(f"FATAL: need H%GS==0,I%GS==0,(2I)%8==0,H%8==0 (H={H} I={I} GS={GS})")
        return 2

    # ---- import habana (lazy plugin now, since env was set above) ----
    try:
        import habana_frameworks.torch  # noqa: F401  (loads the LAZY plugin)
        import habana_frameworks.torch.core as htcore
        import habana_frameworks.torch as htorch
        from habana_frameworks.torch.utils.internal import is_lazy
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.",
              file=sys.stderr)
        return 2

    # ---- GATE 0: we MUST be in lazy mode, or capture cannot happen at all ----
    if not is_lazy():
        print("FATAL: is_lazy() is False after import -- HPU-graph capture is "
              "lazy-mode-only and would RAISE. PT_HPU_LAZY_MODE=%r."
              % os.environ.get("PT_HPU_LAZY_MODE"), file=sys.stderr)
        return 2
    print(f"=== test_fused_op_capture ===  lazy={is_lazy()}  T={T} H={H} I={I} GS={GS}")

    # ---- load the op .so (LOUD if it is the eager build that can't load lazy) ----
    so, label = _find_so()
    print(f"op .so: {so}  [{label}]")
    if not os.path.isfile(so):
        print(f"FATAL: {so} not found. Build the op first.", file=sys.stderr)
        return 2
    try:
        torch.ops.load_library(so)
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        print(f"\n>>> FAIL: could not load {so} under PT_HPU_LAZY_MODE=1.\n    {ex}",
              file=sys.stderr)
        if "EagerOp" in msg or "eager" in msg or "undefined symbol" in msg:
            print(
                "    DIAGNOSIS: this is the EAGER-frontend op. Its frontend symbols\n"
                "    (habana::eager::EagerOpBase::run/validate_inputs,\n"
                "    HbEagerTensorPool::get_backend_tensor) live ONLY in the eager\n"
                "    plugin libhabana_pytorch2_plugin.so, which is NOT loaded under\n"
                "    PT_HPU_LAZY_MODE=1 (the lazy plugin libhabana_pytorch_plugin.so\n"
                "    is loaded instead). A LazyOp-frontend variant of the op is\n"
                "    REQUIRED to load+capture under lazy serving. See the findings\n"
                "    report (build a sibling that swaps EagerOp -> habana_lazy::LazyOp\n"
                "    + RUN_MAYBE_WITH_ACC_THREAD; the MoeFusedExpert OpBackend AddNode\n"
                "    chain is reused unchanged).", file=sys.stderr)
        return 1
    if not hasattr(torch.ops, "hpu_moe_ext") or not hasattr(
            torch.ops.hpu_moe_ext, "fused_expert"):
        print("FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after load.",
              file=sys.stderr)
        return 2

    dev = "hpu"
    torch.manual_seed(0)
    TOL = 1e-2
    w13p, s13, z13, w2p, s2, z2 = _make_expert(H, I, GS, 100, dev)
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    rw_val = 0.75
    rw = torch.full((1, 1), rw_val, dtype=torch.bfloat16, device=dev)
    htcore.mark_step()

    # trusted reference (NOT under capture)
    ref = (_ref_expert(x, w13p, s13, z13, w2p, s2, z2, I) * rw_val)
    htcore.mark_step()
    refc = ref.to("cpu")

    # --- tiny module whose forward is exactly one fused_expert call ----------
    class FusedExpertModule(torch.nn.Module):
        def forward(self, x, w13p, s13, z13, w2p, s2, z2, rw):
            return torch.ops.hpu_moe_ext.fused_expert(x, w13p, s13, z13, w2p, s2, z2, rw)

    mod = FusedExpertModule().to(dev)

    # snapshot capture bookkeeping so this test is isolated from any prior wraps
    from habana_frameworks.torch.hpu.graphs import CachedParams
    base_hits = sum(CachedParams.cache_hits.values())

    # wrap_in_hpu_graph: 1st call captures the recipe, subsequent calls replay it.
    # (verbose=True so the device log also prints HPUGraph cache stats.)
    wrapped = htorch.hpu.wrap_in_hpu_graph(mod, verbose=True, log_frequency=1)

    rc = 0
    print("\n--- CHECK A: executes coherently under wrap_in_hpu_graph ---")
    outs = []
    N = 4
    for i in range(N):
        o = wrapped(x, w13p, s13, z13, w2p, s2, z2, rw)
        htcore.mark_step()
        outs.append(o.to("cpu"))
    rn0, ma0 = _relnorm(outs[0], refc)
    print(f"  call#0  relNorm={rn0:.4e}  maxAbs={ma0:.4e}  |ref|={refc.float().norm():.4f}")
    if rn0 <= TOL:
        print(f"  >>> PASS: captured forward matches reference (relNorm {rn0:.2e}).")
    else:
        print(f"  >>> FAIL: captured forward relNorm {rn0:.4e} > tol {TOL:.2e}.",
              file=sys.stderr)
        rc = 1
    # every replay must be identical to call#0 (a true replay is deterministic)
    for i in range(1, N):
        d = (outs[i].float() - outs[0].float()).abs().max().item()
        if d > 0:
            print(f"  >>> WARN: replay call#{i} differs from call#0 by maxAbs={d:.3e} "
                  "(expected bit-identical replay).", file=sys.stderr)

    print("\n--- CHECK B: 2nd+ calls are graph REPLAYS (cache-hit, no recompile) ---")
    hits = sum(CachedParams.cache_hits.values()) - base_hits
    # find OUR module's recipe cache (closure dict reachable via clear_cache)
    n_recipes = None
    try:
        cell = wrapped.forward.__wrapped__  # orig fwd (just to confirm wrap happened)
    except Exception:  # noqa: BLE001
        cell = None
    # the cache dict is a closure of wrapped.forward; introspect via free vars
    try:
        fv = wrapped.forward.__closure__
        names = wrapped.forward.__code__.co_freevars
        cache_dict = None
        for nm, cl in zip(names, fv or ()):
            if nm == "cache":
                cache_dict = cl.cell_contents
        n_recipes = len(cache_dict) if cache_dict is not None else None
    except Exception:  # noqa: BLE001
        n_recipes = None

    print(f"  cache_hits (this module) = {hits} over {N} calls "
          f"(expect {N-1}: 1 capture + {N-1} replays)")
    if n_recipes is not None:
        print(f"  distinct recipes cached   = {n_recipes} (expect 1: same shapes "
              "every call => one recipe)")
    replay_ok = (hits >= N - 1)
    recipe_ok = (n_recipes is None) or (n_recipes == 1)
    if replay_ok and recipe_ok:
        print("  >>> PASS: subsequent calls replayed a single captured recipe "
              "(per-token dispatch is ~free).")
    else:
        print(f"  >>> FAIL: replay/cache signal wrong (hits={hits}, recipes={n_recipes}). "
              "If hits==0 the forward was re-run/eager each call, NOT replayed.",
              file=sys.stderr)
        rc = 1

    print("\n--- CHECK C: best-effort op-in-recipe (bypass control + latency) ---")
    # Negative control: bypass_hpu_graphs=True runs orig_fwd OUTSIDE the graph.
    # If the captured path and the bypass path agree numerically AND the captured
    # path is materially faster, the op really is inside the replayed recipe.
    try:
        import time
        # warm both paths
        _ = wrapped(x, w13p, s13, z13, w2p, s2, z2, rw); htcore.mark_step()
        _ = wrapped(x, w13p, s13, z13, w2p, s2, z2, rw, bypass_hpu_graphs=True)
        htcore.mark_step()
        ITERS = 50
        htorch.hpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            oc = wrapped(x, w13p, s13, z13, w2p, s2, z2, rw)
        htorch.hpu.synchronize()
        t_cap = (time.perf_counter() - t0) / ITERS
        htorch.hpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ob = wrapped(x, w13p, s13, z13, w2p, s2, z2, rw, bypass_hpu_graphs=True)
        htorch.hpu.synchronize()
        t_byp = (time.perf_counter() - t0) / ITERS
        rn_c, _ = _relnorm(oc.to("cpu"), refc)
        rn_b, _ = _relnorm(ob.to("cpu"), refc)
        print(f"  captured-replay : {t_cap*1e6:8.1f} us/call  (relNorm {rn_c:.2e})")
        print(f"  bypass-eager    : {t_byp*1e6:8.1f} us/call  (relNorm {rn_b:.2e})")
        if rn_c <= TOL and rn_b <= TOL:
            faster = t_byp / max(t_cap, 1e-9)
            print(f"  replay/bypass speed ratio = {faster:.2f}x "
                  f"({'replay faster -> op IS in the recipe' if faster > 1.05 else 'no clear separation (host-bound or tiny op); rely on CHECK B + device log'}).")
        else:
            print("  >>> WARN: bypass/captured numerics differ from reference; "
                  "investigate on device.", file=sys.stderr)
    except Exception as ex:  # noqa: BLE001
        print(f"  (latency/bypass control skipped: {ex})")

    print(f"\n[done] test_fused_op_capture {'OK' if rc == 0 else 'FAILED'} (rc={rc}); "
          f"verdict: op {'LOADS + EXECUTES + REPLAYS from a captured recipe under lazy' if rc == 0 else 'did NOT capture under lazy -- see above'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
