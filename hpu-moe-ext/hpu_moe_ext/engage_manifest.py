"""ENGAGE — the silent-fallback killer (LOUD fallbacks + a STRICT hard-fail mode +
a one-block ENGAGEMENT MANIFEST that prints the TRUE resolved fast-path state).

────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS (the box owner's directive, 2026-06-22)
────────────────────────────────────────────────────────────────────────────
We have been repeatedly burned by SILENT FALLBACKS measured as if they were the
fast path -> false conclusions, hours of circling:
  * the fused int4 MoE op silently dequants all-32 (not routed-8),
  * KBUCKET silently no-ops,
  * the bs>1 MTP verify silently falls to all-expert,
  * the engine chain silently falls to stock at 0 WARMED,
  * the fp8 ops silently fail to load.

This module makes EVERY fast-path-vs-fallback decision OBSERVABLE and, on demand,
FATAL:

  1. ``engage_fallback(...)`` — at every fallback site, emit a ONE-SHOT (per
     process, per site) LOUD ``[ENGAGE-FALLBACK]`` WARNING naming the fast path
     NOT taken, the reason, and the slow path that runs instead. So a slow path
     can never again masquerade as the fast path in the logs.

  2. ``HPU_ENGINE_STRICT=1`` (default OFF) — converts an UNINTENDED fallback into
     a HARD RAISE (``EngineStrictError``) that kills the worker. "Unintended" ==
     the user asked for the fast path (the corresponding flag is set) but the code
     resolved to the fallback. A LEGITIMATE delegation (prefill->stock, a
     genuinely-ineligible-at-this-shape op, a flag deliberately off) is ALLOWLISTED
     and never raises — so strict does not false-fail. The caller passes
     ``intended=`` (did the user ask for this fast path?) to distinguish the two.

  3. ``HPU_ENGAGE_MANIFEST=1`` (default ON) — prints, ONCE per worker process (==
     per TP rank), a single ``[ENGAGE-MANIFEST rankR]`` block giving the TRUE
     RESOLVED state of every fast path (MoE route, fp8 mode, bs>1 admission,
     engine-chain, x-buffer residency, DSA), resolving the SAME predicates the
     forward uses — NOT the env intent. So a human/test SEES engagement at a glance.

────────────────────────────────────────────────────────────────────────────
DEFAULT-OFF / BYTE-IDENTICAL DISCIPLINE (non-negotiable)
────────────────────────────────────────────────────────────────────────────
  * STRICT defaults OFF. With ``HPU_ENGINE_STRICT`` unset, ``engage_fallback``
    NEVER raises — it only logs. So enabling this module changes NO control flow;
    the only behavioral delta vs before is the added log lines.
  * The MANIFEST defaults ON but is pure output (a logger call) — it reads state,
    never mutates it. ``HPU_ENGAGE_MANIFEST=0`` silences it.
  * NO ACCELERATOR AT IMPORT. No torch / habana import at module load; the env
    reads + the registry + the one-shot guards are card-free, so this file
    ``py_compile``s and self-tests on a CPU-only box. (The manifest's predicate
    resolution takes plain values the caller already computed — it does not touch
    the device.)
  * LOUD by construction. A fallback that fires with strict-on and intended-on is
    an exception with the exact env/reason to fix it.

Run the card-free self-test:
    python hpu-moe-ext/hpu_moe_ext/engage_manifest.py
Exit 0 = the env gates, the one-shot dedup, the intended-vs-legitimate strict
split, and the manifest line builder are all correct (card-free).
"""
from __future__ import annotations

import os
from typing import Optional


# ----------------------------------------------------------------------------- #
#  ENV GATES (read once at import; the only env this module owns).               #
#                                                                               #
#   HPU_ENGINE_STRICT=1     -> an UNINTENDED fallback HARD-RAISES (default OFF). #
#   HPU_ENGAGE_MANIFEST=0   -> silence the resolved-state banner   (default ON). #
# ----------------------------------------------------------------------------- #
def _truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


_STRICT = _truthy(os.environ.get("HPU_ENGINE_STRICT"))
# manifest defaults ON: only an explicit 0/false/no/off silences it.
_MANIFEST_ON = (os.environ.get("HPU_ENGAGE_MANIFEST", "1").strip().lower()
                not in ("0", "false", "no", "off"))


def strict_enabled() -> bool:
    """True iff HPU_ENGINE_STRICT is set (an unintended fast-path fallback hard-fails)."""
    return _STRICT


def manifest_enabled() -> bool:
    """True iff the resolved-state ENGAGE-MANIFEST banner should print (default ON)."""
    return _MANIFEST_ON


class EngineStrictError(RuntimeError):
    """HPU_ENGINE_STRICT=1 and a fast path the user explicitly requested resolved to
    its fallback instead. Raised to KILL the worker LOUD rather than silently run the
    slow path and let a benchmark measure a lie. The message names the env to fix."""


# ----------------------------------------------------------------------------- #
#  Logger — reuse vLLM's logger in-serve; a stderr shim on a bare CPU box.       #
#  No hard dependency at import (mirrors the other modules' contract).           #
# ----------------------------------------------------------------------------- #
def _get_logger():
    try:
        import logging
        return logging.getLogger("vllm")
    except Exception:  # noqa: BLE001 — never let logging setup break the serve
        import sys

        class _Stderr:
            def _emit(self, lvl, msg, *a):
                if a:
                    try:
                        msg = msg % a
                    except Exception:  # noqa: BLE001
                        msg = msg + " " + " ".join(str(x) for x in a)
                sys.stderr.write(f"[engage][{lvl}] {msg}\n")

            def info(self, m, *a):
                self._emit("INFO", m, *a)

            def warning(self, m, *a):
                self._emit("WARNING", m, *a)

            def error(self, m, *a):
                self._emit("ERROR", m, *a)

        return _Stderr()


# ----------------------------------------------------------------------------- #
#  ONE-SHOT FALLBACK REGISTRY (per process, per site).                           #
#                                                                               #
#  Keyed on the call SITE string so a per-token hot path logs ONCE, not per      #
#  decode step (the no-flood rule). The registry is a module-global set, so it   #
#  is per worker PROCESS == per TP rank (each rank is its own process).          #
# ----------------------------------------------------------------------------- #
_FIRED: set = set()


def _reset_for_test() -> None:
    """Clear the one-shot registry (card-free self-test only)."""
    _FIRED.clear()


def engage_fallback(
    site: str,
    fast: str,
    reason: str,
    slow: str,
    *,
    intended: bool,
    strict_env: Optional[str] = None,
) -> None:
    """Record that a FAST path was NOT taken and the SLOW path runs instead.

    Behavior:
      * Always (one-shot per (process, ``site``)): emit a LOUD
        ``[ENGAGE-FALLBACK]`` WARNING ``FAST PATH <fast> NOT taken (<reason>) ->
        running <slow>``. So the slow path is visible in the logs, never silent.
      * If ``HPU_ENGINE_STRICT`` is set AND ``intended`` is True (the user asked
        for this fast path via its flag, so the fallback is UNINTENDED) -> RAISE
        ``EngineStrictError`` (kills the worker LOUD). ``strict_env`` names the env
        the operator set that the strict raise is enforcing (for the message).
      * If ``intended`` is False (a LEGITIMATE delegation: prefill->stock, a
        genuinely-ineligible op at this shape, a flag deliberately off) -> NEVER
        raise, even under strict. This is the allowlist that keeps strict from
        false-failing.

    Args:
      site     : a STABLE, unique key for this fallback location (the one-shot key).
                 Convention: "<module>.<func>:<what>" e.g.
                 "moe_wna16.apply_monolithic:routed-gather".
      fast     : the fast path that was skipped (human label).
      reason   : WHY it was skipped (the diagnosable cause).
      slow     : the slow/fallback path that runs instead.
      intended : did the user EXPLICITLY ask for ``fast`` (its flag set)? True =>
                 the fallback is unintended (strict raises); False => legitimate
                 delegation (strict allows). THE load-bearing arg.
      strict_env : the env name whose value the strict raise enforces (message only).

    DEFAULT-OFF: with HPU_ENGINE_STRICT unset this only logs (one-shot) and returns
    — no control-flow change vs before this call existed.
    """
    log = _get_logger()
    first = site not in _FIRED
    if first:
        _FIRED.add(site)
        # The LOUD one-shot. INTENDED fallbacks are the dangerous class (the user
        # wanted the fast path and is silently on the slow one) -> mark them.
        log.warning(
            "[ENGAGE-FALLBACK] FAST PATH %s NOT taken (%s) -> running %s [site=%s "
            "intended=%s]%s",
            fast, reason, slow, site, intended,
            "" if not (intended and _STRICT)
            else "  -- HPU_ENGINE_STRICT=1: RAISING (this fallback was requested but "
                 "did not engage).")
    if intended and _STRICT:
        ev = strict_env or "the corresponding HPU_* flag"
        raise EngineStrictError(
            f"[ENGAGE-STRICT] {fast} was REQUESTED ({ev} set) but the code resolved to "
            f"the fallback '{slow}' (reason: {reason}; site={site}). HPU_ENGINE_STRICT=1 "
            f"converts this UNINTENDED fast-path fallback into a hard failure so a "
            f"benchmark cannot silently measure the slow path. Fix the cause, or unset "
            f"HPU_ENGINE_STRICT to allow the (logged) fallback, or unset {ev} if the "
            f"fast path is genuinely not wanted here.")


# ----------------------------------------------------------------------------- #
#  THE ENGAGEMENT MANIFEST — one resolved-state block per worker (== per rank).  #
#                                                                               #
#  Prints the TRUE RUNTIME state of every fast path resolved from the SAME       #
#  predicates the forward uses (passed in by the caller, which already computed  #
#  them at weight-load). NOT the env intent — the resolved reality.              #
# ----------------------------------------------------------------------------- #
_MANIFEST_FIRED: set = set()


def emit_manifest(rank: int, fields: dict, *, once_key: str = "global") -> None:
    """Emit the ``[ENGAGE-MANIFEST rankR]`` one-block banner of resolved fast-path
    state, ONCE per (process, ``once_key``).

    ``fields`` is an ORDERED dict {label: resolved_value}; the caller resolves each
    value the SAME way the forward does (so the banner is ground truth, not intent).
    Example fields for the MoE path:
        MoE      = "ROUTED-FUSED(K=8)" | "all-expert(K clamps E=32)" | "stock-int4"
        fp8      = "FUSED-DIRECT" | "FUSED" | "resident" | "OFF"
        bs>1     = "admitted(KBUCKET 1:8,2:8,4:16)" | "all-expert(>K)"
        x_buf    = "resident[8,6144]" | "in-forward-clone" | "n/a"
        chain    = "STATIC-tail" | "LAYER-7island" | "MoE-island" | "OFF"
        DSA      = "sparse" | "dense" | "n/a"
        STRICT   = "ON" | "off"

    DEFAULT-ON but pure output: it reads ``fields`` (already-resolved values) and
    logs — it never touches the device or mutates state. ``HPU_ENGAGE_MANIFEST=0``
    silences it entirely. ``once_key`` lets two distinct manifests (e.g. the MoE
    layer-load manifest and the engine-chain warmup manifest) each fire once."""
    if not _MANIFEST_ON:
        return
    key = (once_key, int(rank))
    if key in _MANIFEST_FIRED:
        return
    _MANIFEST_FIRED.add(key)
    body = " | ".join(f"{k}={v}" for k, v in fields.items())
    _get_logger().warning(
        "[ENGAGE-MANIFEST rank%d] %s%s", int(rank), body,
        "" if not _STRICT else "  (HPU_ENGINE_STRICT=1: an unintended fast-path "
        "fallback will HARD-FAIL)")


# ============================================================================= #
#  CARD-FREE SELF-TEST (no accelerator touched).                                 #
# ============================================================================= #
def _t1_gates():
    # env predicate parsing for both gates.
    saved_s = os.environ.get("HPU_ENGINE_STRICT")
    saved_m = os.environ.get("HPU_ENGAGE_MANIFEST")
    try:
        # We test the PURE parser (_truthy) + the documented manifest default, not the
        # import-time snapshot (which is frozen at module load).
        for v, want in (("1", True), ("on", True), ("true", True), ("YES", True),
                        ("0", False), ("", False), ("off", False), (None, False)):
            assert _truthy(v) is want, f"_truthy({v!r}) != {want}"
        # manifest default-ON semantics: only 0/false/no/off silence it.
        for v, want in (("1", True), ("on", True), ("", True), ("anything", True),
                        ("0", False), ("false", False), ("no", False), ("off", False)):
            got = (v.strip().lower() not in ("0", "false", "no", "off"))
            assert got is want, f"manifest gate {v!r} != {want}"
    finally:
        if saved_s is None:
            os.environ.pop("HPU_ENGINE_STRICT", None)
        else:
            os.environ["HPU_ENGINE_STRICT"] = saved_s
        if saved_m is None:
            os.environ.pop("HPU_ENGAGE_MANIFEST", None)
        else:
            os.environ["HPU_ENGAGE_MANIFEST"] = saved_m
    return "T1 strict + manifest env gate parsing correct (manifest default-ON)"


def _t2_oneshot_dedup():
    _reset_for_test()
    import io
    import logging
    log = logging.getLogger("vllm")
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    log.addHandler(h)
    lvl = log.level
    log.setLevel(logging.WARNING)
    try:
        # fire the SAME site 5x; must log exactly once.
        for _ in range(5):
            engage_fallback("t2.site:x", "FAST", "because", "SLOW", intended=False)
        n = buf.getvalue().count("[ENGAGE-FALLBACK]")
        assert n == 1, f"one-shot site logged {n} times, want 1"
        # a DIFFERENT site logs again.
        engage_fallback("t2.site:y", "FAST2", "reason2", "SLOW2", intended=False)
        n2 = buf.getvalue().count("[ENGAGE-FALLBACK]")
        assert n2 == 2, f"distinct site should add a line, got {n2}"
    finally:
        log.removeHandler(h)
        log.setLevel(lvl)
    return "T2 one-shot fallback log dedups per site (per process), distinct sites re-log"


def _t3_strict_intended_raises_legit_allows():
    # With strict ON: intended=True raises; intended=False (legitimate) does NOT.
    saved = os.environ.get("HPU_ENGINE_STRICT")
    global _STRICT
    prev = _STRICT
    try:
        _STRICT = True  # simulate HPU_ENGINE_STRICT=1 (import snapshot is frozen)
        _reset_for_test()
        # legitimate delegation: never raises even under strict.
        engage_fallback("t3.legit", "FUSED", "AWQ per-expert zp (ineligible here)",
                        "stock int4", intended=False, strict_env="HPU_MOE_FUSED_OP")
        # unintended (user asked, didn't engage): MUST raise under strict.
        raised = False
        try:
            engage_fallback("t3.intended", "FUSED", "stacked zp absent at decode",
                            "stock int4", intended=True, strict_env="HPU_MOE_FUSED_OP")
        except EngineStrictError as e:
            raised = "HPU_MOE_FUSED_OP" in str(e) and "REQUESTED" in str(e)
        assert raised, "strict + intended must raise EngineStrictError naming the env"
        # with strict OFF: intended fallback only logs, never raises.
        _STRICT = False
        _reset_for_test()
        engage_fallback("t3.offnoraise", "FUSED", "x", "stock", intended=True,
                        strict_env="HPU_MOE_FUSED_OP")  # must NOT raise
    finally:
        _STRICT = prev
        if saved is None:
            os.environ.pop("HPU_ENGINE_STRICT", None)
        else:
            os.environ["HPU_ENGINE_STRICT"] = saved
    return ("T3 strict raises ONLY the unintended (user-requested) fallback; legitimate "
            "delegations + strict-off never raise (the allowlist that prevents false-fail)")


def _t4_manifest_oneshot():
    import io
    import logging
    global _MANIFEST_ON
    prev = _MANIFEST_ON
    _MANIFEST_FIRED.clear()
    log = logging.getLogger("vllm")
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    log.addHandler(h)
    lvl = log.level
    log.setLevel(logging.WARNING)
    try:
        _MANIFEST_ON = True
        flds = {"MoE": "ROUTED-FUSED(K=8)", "fp8": "OFF", "bs>1": "all-expert(>K)",
                "chain": "OFF", "x_buf": "resident[8,6144]", "DSA": "sparse"}
        emit_manifest(0, flds)
        emit_manifest(0, flds)  # second call same rank+key: must NOT re-emit.
        s = buf.getvalue()
        assert s.count("[ENGAGE-MANIFEST rank0]") == 1, "manifest must fire once per rank"
        assert "MoE=ROUTED-FUSED(K=8)" in s and "DSA=sparse" in s, "fields must render"
        # a different rank emits its own line.
        emit_manifest(3, flds)
        assert "[ENGAGE-MANIFEST rank3]" in buf.getvalue(), "rank3 should emit"
        # OFF silences entirely.
        _MANIFEST_ON = False
        _MANIFEST_FIRED.clear()
        before = buf.getvalue()
        emit_manifest(7, flds)
        assert buf.getvalue() == before, "HPU_ENGAGE_MANIFEST=0 must silence the banner"
    finally:
        _MANIFEST_ON = prev
        log.removeHandler(h)
        log.setLevel(lvl)
    return "T4 manifest emits one block per rank, renders resolved fields, OFF silences"


def _selftest() -> int:
    tests = [_t1_gates, _t2_oneshot_dedup, _t3_strict_intended_raises_legit_allows,
             _t4_manifest_oneshot]
    ok = 0
    for t in tests:
        try:
            print(f"  PASS  {t()}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\nengage_manifest self-test: {ok}/{len(tests)} passed")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
