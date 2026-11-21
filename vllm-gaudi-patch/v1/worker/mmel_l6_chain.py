"""MMEL L6 — G1: the chained-execution decoder-layer body (live-fed).

This module supplies the REAL body of ``run_chained_decoder_layer`` that the L6
seam (``mmel_l6_exec.py``) detects-and-routes-to once present, replacing the
``MMELChainNotWiredError`` stub. It is the G1 deliverable of
``mmel/L6_BUILD_PLAN.md`` (the live-decode driver for the captured 1-fence
fused-execution-decoder layer).

────────────────────────────────────────────────────────────────────────────
WHAT G1 IS (and is NOT), grounded in the build plan + the on-card evidence
────────────────────────────────────────────────────────────────────────────
The captured *islands* (``mmel/cap_{norm,qproj,kvproj,mla_paged,oproj,routed}``)
proved, in the standalone C harness (``route2_chain_multi.c``, RUN_CHAIN STEP-4),
that a FULL decoder layer ``norm → q/kv-proj → MLA-paged → o-proj → norm → MoE``
chains on ONE Synapse stream behind **ONE host fence**, byte-exact vs an N-fence
serial control (relNorm 0.0, cmp+sha256). But those islands were replayed from
**FIXED captured ``.bin`` inputs**. The GAP G1 closes: drive that same op chain
from **LIVE decode tensors** (per-step hidden_states / positions / paged KV) so
the bridge captures it as the same collapsed recipe inside a real serve step.

G1 takes the *torch-level fence-collapse* path the L6 design specifies
(``MODERN_MOE_EXECUTION_LAYER_DESIGN.md`` §1.L6 / §L6 lines 271-286): compose the
proven device ops (the int4 routed MoE op ``torch.ops.hpu_moe_ext.fused_moe``,
the MLA-decode op ``torch.ops.hpu_mla_ext.mla_decode``, the bridge RMSNorm) into
ONE ``forward`` **with no per-op host sync inside it**, so the bridge's
``wrap_in_hpu_graph`` (which runs AFTER our ``.forward`` swap, runner ``:4756``)
captures the whole layer as a single (or few) recipe instead of the ~135
fragmenting submissions/forward the stock path issues
(``glm52-single-stream-rootcause``). The raw-CS / ctypes ``synLaunch`` byte-handoff
of the serialized blobs (the C harness's mechanism) is the DEEPER structural
lever (it can drive the *exact* captured recipe), but it cannot bridge the
q/kv-projection BRANCH edge by a raw byte-handoff (the MLA OUT is [1,16384] but
the layer I/O is [1,6144]); it is reserved for B5/G3. G1 is the bring-up-correct,
on-card-gateable first draft of the live-fed chain.

THE HONEST FRAME (the build plan §0/§6 — do NOT overclaim): an on-card top_k
discriminator proved the bs=1 floor is **structural, not dequant-compute** — so
this fence/structure collapse is the *only* surviving single-stream lever, but
fence-collapse ALONE is ~10% (68.8→61.5 ms). The larger single-stream payoff is
gated on B5 (collective-overlap + async layer-pipeline), built separately. G1's
unambiguous value is correctness/control/generality + being the substrate the
accept-multiplier (tree-verify) rides. **This is a FIRST DRAFT to be on-card
gated (B3.3 recipe-count, B4.1 byte-exact A/B, B4.2 tok/s), NOT a declared win.**

────────────────────────────────────────────────────────────────────────────
DESIGN DISCIPLINE (the project's standing rules — held here)
────────────────────────────────────────────────────────────────────────────
  * LOUD failures. Every "cannot do this" path RAISES with the exact reason; the
    seam's coherent-A/B wrapper is what keeps a live generation safe, not a silent
    fallback here. A wrong bind must crash a bring-up gate, not corrupt output.
  * STATIC SHAPE (G4 / W4.1). The whole forward must replay ONE recipe per warmup
    bucket: static gather widths (``index_topk+1``, never ``max_keep.item()``),
    key-pad to 128 multiples, pow2 ``kk``, CPU-built routing metadata moved to
    device once — the four DSA serve-bug fixes verbatim. Any residual
    data-dependent step is fenced into an EAGER ISLAND (``@torch.compiler.disable``)
    that is correct and captures the rest (the named G4 fallback). This is what
    keeps us off the DSA mixed-shape ``synStatus 26`` / ``[SHAPE AGNOSTIC]`` crash.
  * BYTE-EXACT INTENT. I/O is identical to ``DeepseekV2DecoderLayer.forward``;
    ``run_chained_decoder_layer`` is gated by the seam's ``MmelGreedyABHarness``.
  * GATED, DEFAULT-OFF. The chain only runs behind ``MODERN_MOE_EXEC=1`` (read in
    the seam). Importing THIS module flips ``chain_is_wired()`` true (it deletes
    the stub marker + exports ``MMEL_CHAIN_WIRED=1``), so the seam will *install*
    the swap when the flag is on — but with the flag OFF the serve is byte-for-byte
    stock and nothing here ever runs.
  * NO ACCELERATOR AT IMPORT. torch / habana are imported lazily inside the
    functions that need them, so this file ``py_compile``s and self-tests on a
    CPU-only box (the m3calib venv). The dataclasses + the static-shape arithmetic
    + the routing-metadata builder are card-free unit-tested at the bottom.
  * REUSE, DON'T REBUILD. The per-config contract comes from the L5 ``ExecConfig``
    (``exec_config.py``); the device ops come from the already-built ``.so``s
    (``libmoe_fused_routed{,_lazy}.so``, ``libmla_decode{,_lazy}.so``); the
    dense/MoE-per-layer oracle is ``exec_cfg.layer_is_dense``. We add NO new kernel.

Run the card-free self-test:
    $HOME/.venvs/m3calib/bin/python \
        $HOME/AI/vllm-gaudi-patch/v1/worker/mmel_l6_chain.py
Exit 0 = the static-shape arithmetic, the routing-metadata builder, the op-symbol
gate logic, and the wired-detection contract are all correct (card-free).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


# ----------------------------------------------------------------------------- #
#  Logger — reuse the seam's logger factory so in-serve we get vLLM's logger and #
#  on a bare CPU box we get a stderr shim. No hard dependency at import.          #
# ----------------------------------------------------------------------------- #
def _get_logger():
    # Prefer vLLM's logger when in-serve.
    try:
        from vllm.logger import init_logger  # type: ignore

        return init_logger("mmel.l6_chain")
    except Exception:  # noqa: BLE001
        class _Stderr:
            @staticmethod
            def _emit(level, msg, *a):
                if a:
                    try:
                        msg = msg % a
                    except Exception:  # noqa: BLE001
                        msg = msg + " " + " ".join(str(x) for x in a)
                sys.stderr.write(f"[mmel.l6_chain][{level}] {msg}\n")

            def info(self, m, *a):
                self._emit("INFO", m, *a)

            def warning(self, m, *a):
                self._emit("WARNING", m, *a)

            def error(self, m, *a):
                self._emit("ERROR", m, *a)

            def debug(self, m, *a):
                pass

        return _Stderr()


logger = _get_logger()


# ----------------------------------------------------------------------------- #
#  ENGAGE bridge — the shared LOUD-fallback / HPU_ENGINE_STRICT / ENGAGE-MANIFEST #
#  helper lives in the hpu-moe-ext package (engage_manifest.py). It is a tiny,    #
#  dependency-free, CPU-importable module, but it is NOT in the vllm_gaudi worker #
#  package — so we resolve it the SAME dual-path way the op .so's are resolved:   #
#  the installed module first, then the in-container /opt mount, then the host    #
#  tree (the CPU self-test box). If it cannot be found we degrade to a LOCAL inert #
#  shim (log-only, never strict) so the chain still runs — the manifest/strict are #
#  diagnostics, never a hard dependency of the serve. Resolved ONCE, cached.      #
# ----------------------------------------------------------------------------- #
_ENGAGE_MOD = [None]   # cache: the resolved module (or the inert shim)
_ENGAGE_SEARCH = (
    "/opt/hpu-moe-ext/hpu_moe_ext/engage_manifest.py",       # in-container mount
    "$HOME/AI/hpu-moe-ext/hpu_moe_ext/engage_manifest.py",  # host self-test box
)


def _engage():
    """Return the shared engage_manifest module (LOUD fallbacks + strict + manifest),
    resolved once. Tries the installed package, then the in-container /opt path, then the
    host tree; if none resolve, returns a LOCAL inert shim (log-only) so the chain never
    hard-depends on it. The env override ENGAGE_MANIFEST_PY pins an exact path."""
    if _ENGAGE_MOD[0] is not None:
        return _ENGAGE_MOD[0]
    mod = None
    try:  # installed package (the normal in-serve case once hpu_moe_ext is on the path)
        from hpu_moe_ext import engage_manifest as mod  # type: ignore
    except Exception:  # noqa: BLE001 — fall to the path-based resolution
        mod = None
    if mod is None:
        import importlib.util
        cands = []
        _ovr = os.environ.get("ENGAGE_MANIFEST_PY")
        if _ovr:
            cands.append(_ovr)
        cands.extend(_ENGAGE_SEARCH)
        for cand in cands:
            if cand and os.path.isfile(cand):
                try:
                    spec = importlib.util.spec_from_file_location(
                        "hpu_moe_ext_engage_manifest", cand)
                    m = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(m)  # type: ignore[union-attr]
                    mod = m
                    break
                except Exception:  # noqa: BLE001 — try the next candidate
                    continue
    if mod is None:
        # Inert shim: log-only, never strict (the shared module was not found; we must
        # NOT fabricate strict behavior off a missing dep). One-shot per site like the real
        # one, so a degraded box still gets the LOUD fallback line.
        class _Shim:
            _fired = set()

            @staticmethod
            def engage_fallback(site, fast, reason, slow, *, intended, strict_env=None):
                if site in _Shim._fired:
                    return
                _Shim._fired.add(site)
                logger.warning(
                    "[ENGAGE-FALLBACK] FAST PATH %s NOT taken (%s) -> running %s "
                    "[site=%s intended=%s] (engage_manifest module not found; strict "
                    "DISABLED)", fast, reason, slow, site, intended)

            @staticmethod
            def emit_manifest(rank, fields, *, once_key="global"):
                logger.warning("[ENGAGE-MANIFEST rank%d] %s", int(rank),
                               " | ".join(f"{k}={v}" for k, v in fields.items()))

            @staticmethod
            def strict_enabled():
                return False

            @staticmethod
            def manifest_enabled():
                return os.environ.get("HPU_ENGAGE_MANIFEST", "1").strip().lower() \
                    not in ("0", "false", "no", "off")
        mod = _Shim()
        logger.warning(
            "[mmel.l6_chain] engage_manifest module not found (searched %s); using an "
            "inert log-only shim. HPU_ENGINE_STRICT is DISABLED for the engine-chain "
            "fallbacks (set ENGAGE_MANIFEST_PY to the engage_manifest.py path to enable "
            "strict here).", list(_ENGAGE_SEARCH))
    _ENGAGE_MOD[0] = mod
    return mod

# ---- FORENSIC CORRUPTION PROBE (default OFF; pure no-op when MMEL_POISON unset) ----
# MMEL_POISON=1 perturbs the per-layer residual (+1% then *1.03) at the chained-decoder
# return — the EXACT residual that threads through all 78 layers when MODERN_MOE_EXEC=1
# has swapped the decoder .forward onto OUR chained-exec path. If our L6 engine chain
# actually executes on device, this visibly GARBLES greedy output; coherent greedy output
# ==> the chain did NOT run (stock decoder forward / not engaged) -> override CONFIRMED.
# NB: in the glm52-b1timed serve MODERN_MOE_EXEC is UNSET, so the chain (and thus this
# poison) is INERT there; arm it under the glm52-l6 key (MODERN_MOE_EXEC=1) WITH a WIPED
# recipe cache to test the engine path.
_MMEL_POISON = os.environ.get("MMEL_POISON") == "1"
if _MMEL_POISON:
    try:
        logger.warning(
            "[MMEL_POISON] ARMED: per-layer residual will be perturbed at "
            "run_chained_decoder_layer return. Requires MODERN_MOE_EXEC=1 (else the chain "
            "is not installed and this is inert). Coherent greedy ==> chain did NOT run; "
            "greedy garble ==> our chained-exec runs. WIPE the recipe cache before serving.")
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------- #
#  Errors — LOUD, specific. A mis-wire must surface at a bring-up gate, never as  #
#  a silent degrade. (The seam's MMELChainNotWiredError is for the STUB; once     #
#  THIS module is imported the chain IS wired, so we use our own error types.)    #
# ----------------------------------------------------------------------------- #
class MMELChainError(RuntimeError):
    """A chained-exec precondition failed (bad layer tree, missing op, bad shape)."""


class MMELOpMissingError(MMELChainError):
    """A required device op (.so) is not registered. Names the env to fix it."""


class MMELShapeError(MMELChainError):
    """A live tensor violated the static-shape contract the recipe was captured at."""


# ----------------------------------------------------------------------------- #
#  The proven device ops (already built; map-only, never rebuilt here).          #
#  These are the .so artifacts the island captures replay (RUN_CHAIN/cap_*).      #
# ----------------------------------------------------------------------------- #
# Default eager .so; the lazy/serving variant is the *_lazy.so (the live serve is
# lazy/torch.compile — the same selection the capture scripts make via
# htorch.is_lazy()). The env overrides let the orchestrator pin the exact .so the
# serve image loaded so the captured recipe and the live op are byte-identical.
#
# IN-CONTAINER PATHS (the fix, 2026-06-22): the serve runs INSIDE the vllm-gaudi
# container, which bind-mounts the host hpu-moe-ext tree at /opt/hpu-moe-ext
# (serve.sh:68 / containers.sh:56: `-v $HERE/hpu-moe-ext:/opt/hpu-moe-ext`). So the
# HOST build dir ($HOME/AI/hpu-moe-ext/...) does NOT exist in-container — the
# default MUST be the in-container /opt path. We keep the host path ONLY as a
# trailing fallback so the CPU-only self-test box (which sees the host tree, not
# /opt) still resolves a real .so for the path-existence checks. The env override
# (MOE_ROUTED_SO / MLA_DECODE_SO, set in models/glm52-l6.env to the /opt path) wins
# over both, so the orchestrator can always pin the exact serve-image .so.
#   in-container build dirs (the mount target):
_MOE_EXT_BUILD_INCONTAINER = "/opt/hpu-moe-ext/tpc_kernel/opbackend/build"
_MLA_EXT_BUILD_INCONTAINER = "/opt/hpu-moe-ext/tpc_kernel/opbackend/mla/build"
#   host build dirs (CPU self-test box fallback; NOT present in-container):
_MOE_EXT_BUILD_HOST = "$HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/build"
_MLA_EXT_BUILD_HOST = "$HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/mla/build"

MOE_ROUTED_SO_ENV = "MOE_ROUTED_SO"     # libmoe_fused_routed{,_lazy}.so
MLA_DECODE_SO_ENV = "MLA_DECODE_SO"     # libmla_decode{,_lazy}.so


def _first_existing_or_first(cands) -> str:
    """Return the first path in ``cands`` that exists on this filesystem; if NONE
    exist, return the FIRST candidate (the in-container default) so the loader's own
    existence check produces a LOUD, in-container-correct error message — never a
    silent host-path miss."""
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return cands[0]


def _routed_so_path(is_lazy: bool) -> str:
    p = os.environ.get(MOE_ROUTED_SO_ENV)
    if p:
        return p
    name = "libmoe_fused_routed_lazy.so" if is_lazy else "libmoe_fused_routed.so"
    # in-container /opt first (the live serve), host path as a self-test fallback.
    return _first_existing_or_first([
        os.path.join(_MOE_EXT_BUILD_INCONTAINER, name),
        os.path.join(_MOE_EXT_BUILD_HOST, name),
    ])


def _mla_so_path(is_lazy: bool) -> str:
    p = os.environ.get(MLA_DECODE_SO_ENV)
    if p:
        return p
    name = "libmla_decode_lazy.so" if is_lazy else "libmla_decode.so"
    # in-container /opt first (the live serve), host path as a self-test fallback.
    return _first_existing_or_first([
        os.path.join(_MLA_EXT_BUILD_INCONTAINER, name),
        os.path.join(_MLA_EXT_BUILD_HOST, name),
    ])


# Cache the "ops loaded" state per process so we only load_library once.
_OPS_LOADED = {"moe": False, "mla": False}


def moe_op_swap_enabled() -> bool:
    """True iff the operator opted into the routed int4 op-swap (MMEL_CHAIN_MOE_OP=1).

    This is the SINGLE gate that decides whether G1 directly drives the routed int4
    MoE op (and therefore HARD-requires its .so). In baseline mode (the flag unset,
    the B3.3 fence-collapse measurement) the chain routes the MoE through the stock
    submodule and NEVER calls torch.ops.hpu_moe_ext.fused_moe — so the routed .so is
    OPTIONAL and a missing/unbound op must NOT raise (it would block the count gate).
    """
    return os.environ.get("MMEL_CHAIN_MOE_OP", "").strip().lower() in (
        "1", "true", "yes", "on")


def _moe_island_replay_enabled() -> bool:
    """True iff MMEL_CHAIN_MOE_OP=2 (the device-resident island-replay path).

    Distinct from moe_op_swap_enabled() (==1, the unbound torch-op swap): =2 routes
    the routed-MoE block through the FIRST REAL csengine wire-in — replaying the
    captured cap_routed recipe on the bridge stream behind one synStreamSynchronize.
    Read directly (not via the island module, so the gate works even off-card)."""
    return os.environ.get("MMEL_CHAIN_MOE_OP", "").strip().lower() in (
        "2", "replay", "island")


def _chain_layer_enabled() -> bool:
    """True iff MMEL_CHAIN_LAYER=1 (the S2/S3 FULL-chained-decoder-layer device path).

    DISTINCT from MMEL_CHAIN_MOE_OP (S1, one MoE island): MMEL_CHAIN_LAYER=1 routes
    the WHOLE decoder layer (norm->q/kv->MLA->o_proj->norm->MoE = 7 captured islands)
    through ONE bridge-stream chain behind ONE synStreamSynchronize, live-fed on the
    linear spine + staged on the pre-attention branch (mmel_chain_layer, RUN_CHAIN
    STEP-4). Read directly (not via the chain module) so the gate works off-card.

    Mutually exclusive with MMEL_CHAIN_MOE_OP=2: if BOTH are set the LAYER chain
    wins (it subsumes the MoE island as its tail) — we assert that LOUD at the call
    site rather than silently picking one."""
    return os.environ.get("MMEL_CHAIN_LAYER", "").strip().lower() in (
        "1", "true", "yes", "on", "layer")


def _chain_static_enabled() -> bool:
    """True iff MMEL_CHAIN_STATIC=1 — the CORRECT static post-attention tail path
    (ENGINE_BUILD_GAP_AUDIT §4.3/§4.4). DISTINCT from MMEL_CHAIN_LAYER (the unsound
    all-7 that device-replays the frozen MLA): MMEL_CHAIN_STATIC keeps input_norm +
    self_attn (MLA + DSA, paged read, per-step KV write) + the fused-add
    post_attention_layernorm on the STOCK bucketed path, and drives ONLY the static
    tail device-side (the MoE block — the dominant per-layer dispatch share — via the
    proven S1 device-resident island replay, behind ONE synStreamSynchronize, live-
    fed). This is the byte-exact-able shape the audit says to land FIRST: every
    device island is genuinely fixed-shape + independent of context length, and the
    full layer (stock norm/attn/post-norm + device tail) is byte-comparable to stock
    layer.forward end-to-end (the thing the all-7 staged chain could never do).

    SCOPE NOTE (honest, grounded in the served model): o_proj lives INSIDE self_attn
    (deepseek_v2.py:570 / oot_mla.py:260 return the POST-o_proj output), so at the
    decoder-layer seam the reachable static tail is the MoE block (o_proj already ran
    correctly inside stock attention). The full [o_proj, post_norm, MoE]
    SubChainBReplay + its card-free BYTE-EXACT-vs-STOCK o_proj proof (mmel_chain_layer
    E14) is the standalone mechanism/correctness substrate that backs this design."""
    return os.environ.get("MMEL_CHAIN_STATIC", "").strip().lower() in (
        "1", "true", "yes", "on", "static")


def _chain_live_enabled() -> bool:
    """True iff MMEL_CHAIN_LIVE=1 — the LIVE-FED MLA attention chain (the correctness
    fix, mmel_chain_layer / LIVE_MLA_DECOMP_DESIGN). It replaces ``input_layernorm +
    self_attn`` (steps 1+2 of the op-for-op forward) with the device-resident
    ``input_norm -> fused_qkv_a -> q_a_norm -> q_b -> kv_a_norm -> MLA(live-KV) ->
    o_proj`` chain on the covered decode bucket, EVERY activation edge LIVE (the staged
    pre-attention branch of MMEL_CHAIN_LAYER is the bug this removes). It is the
    ATTENTION driver; the post-attention tail (post_norm + MoE) flows through the
    existing G1/static path unchanged.

    Mutually exclusive with the layer drivers MMEL_CHAIN_LAYER (the unsound all-7) and
    MMEL_CHAIN_STATIC: if a conflicting driver is also set the live-attn path simply
    does NOT engage (it defers, logging once) — never a raise on the live forward. Read
    directly so the gate works off-card."""
    return os.environ.get("MMEL_CHAIN_LIVE", "").strip().lower() in (
        "1", "true", "yes", "on", "live")


def owned_all_layers_enabled() -> bool:
    """True iff MMEL_OWNED_ALL_LAYERS=1 — the #4b ALL-LAYER device-resident forward.

    This is the Build-#4b graduation of the Phase-1 single-layer island. It is a
    STRICT scope-widener layered ON TOP of MMEL_CHAIN_STATIC=1 (the correct static-
    tail path): it removes the MMEL_BANZ_LAYER 1-layer pin so EVERY MoE layer's tail
    is driven device-resident in the SAME fence-free chained forward — but it drives
    that tail through the LIVE-WEIGHT-BOUND, LIVE-ROUTING int4 LazyOp
    (``layer.mlp`` == ``fused_moe_routed`` -> ``torch.ops.hpu_moe_ext.fused_moe`` when
    the serve image has HPU_MOE_FUSED_OP=1), NOT the frozen ``cap_routed`` island
    replay. See the long rationale in ``_run_moe_block`` (the §B.2 / OWNED_FORWARD_PLAN
    edit). Default-OFF: unset => the Phase-1 1-layer island gate (MMEL_BANZ_LAYER) is
    UNCHANGED, byte-for-byte.

    WHY THE LazyOp AND NOT THE ISLAND (the one fact that rewrites the build):
      * The island uploads a DUPLICATE, FROZEN copy of one layer's int4 expert weights
        to fresh resident HBM (~150 MB/layer) AND freezes the per-token K-of-256
        routing (capture_routed_island.py: K=8 identical experts bound once). On 78
        layers that is ~11.4 GB of duplicate HBM (OOM) AND every token routed to the
        SAME frozen 8 experts on every layer (garbage). It is correct ONLY as a 1-layer
        MECHANISM proof, never as the all-layer driver (OWNED_FORWARD_PLAN §A fact 1+2).
      * The LazyOp (``layer.mlp``) binds the model's ALREADY-RESIDENT int4 expert
        tensors in place via ``index_select(0, active)`` of ``layer.w13_qweight.data``
        etc. (fused_expert_routed.fused_moe_routed: zero re-upload -> ZERO extra HBM),
        routes LIVE per token (the real K-of-256 top-k, on-device, no .item()), and is
        captured inside the bridge's own graph. So all-layer device-resident is, by
        construction, NOT bigger than the stock forward's footprint and IS per-layer-
        correct (OWNED_FORWARD_PLAN §B.2/§B.3).

    This flag is INERT unless MMEL_CHAIN_STATIC=1 also engages (it only changes which
    MoE layers the static tail covers); the forward asserts that pairing LOUD."""
    return os.environ.get("MMEL_OWNED_ALL_LAYERS", "").strip().lower() in (
        "1", "true", "yes", "on", "all")


def _banz_layer_pins_this_layer_off(layer_idx: int) -> bool:
    """Phase-1 1-layer gate: True iff MMEL_BANZ_LAYER is set to a layer index that is
    NOT this layer (so this layer's MoE must DELEGATE to the stock dynamic MoE).

    This is the literal wall-#4 (OOM) gate the island forced: the frozen ``cap_routed``
    island replays ONE captured recipe (frozen routing + its OWN duplicated resident
    weights), so engaging it on every MoE layer OOMs (~150 MB/layer) and is wrong
    (frozen routing). MMEL_BANZ_LAYER=<idx> pins the device island to EXACTLY one layer;
    every other MoE layer returns ``layer.mlp(...)`` (byte-exact stock).

    #4b (MMEL_OWNED_ALL_LAYERS=1) does NOT use this pin — it widens the device tail to
    ALL MoE layers via the LazyOp (no island, no per-layer weight upload), so the OOM
    that forced the pin does not exist on that path. Card-free; pure env+int logic.

    Returns False (do NOT pin off) when MMEL_BANZ_LAYER is unset or non-integer — i.e.
    no single-layer pin in effect (the all-layers-via-island fallback, only reachable
    when MMEL_OWNED_ALL_LAYERS is OFF; left as-is for back-compat / debug)."""
    raw = os.environ.get("MMEL_BANZ_LAYER", "").strip()
    if not raw.lstrip("-").isdigit():
        return False
    return int(raw) != int(layer_idx)


# ----------------------------------------------------------------------------- #
#  MMEL_BACKEND — the owned-forward MMEL backend selector (roadmap closure).     #
#  Selects the MmelBackend that run_chained_decoder_layer drives on the covered  #
#  M=1 decode bucket. DEFAULT-OFF: unset => the existing G1/static spine runs,    #
#  byte-identical stock (the ARM_OFF baseline, ONCARD_RUNBOOK.md:94).            #
# ----------------------------------------------------------------------------- #
_OWNED_ENGINE_DIR = os.environ.get(
    "MMEL_OWNED_ENGINE_DIR", "$HOME/AI/mmel/owned_engine")


def _owned_backend_name() -> str:
    """The selected owned-forward MmelBackend (env MMEL_BACKEND). LOUD on unknown.

      ""/unset/off/none/stock -> OFF (the existing chain runs; byte-identical stock).
      "hpu"                    -> construct HpuBackend + drive the device-resident
                                  per-layer chain (ONCARD_RUNBOOK §3; hpu_backend.py).
      "cpu"                    -> the card-free CpuBackend (oracle only; it cannot bind
                                  real device tensors -> refused in a live forward).
    Read directly so the gate works off-card; never silently pick a backend."""
    v = os.environ.get("MMEL_BACKEND", "").strip().lower()
    if v in ("", "off", "none", "stock"):
        return ""
    if v in ("hpu", "cpu"):
        return v
    raise MMELChainError(
        f"[mmel.l6_chain] MMEL_BACKEND={v!r} is unknown; valid = 'hpu' (on-card "
        f"device-resident chain), 'cpu' (card-free oracle only), or unset (OFF, stock).")


def _owned_backend_enabled() -> bool:
    """True iff the owned-forward HpuBackend wire owns the layer (MMEL_BACKEND=hpu)."""
    return _owned_backend_name() == "hpu"


def _owned_one_sync() -> bool:
    """True (DEFAULT) => ONE synStreamSynchronize per forward (the 417->1 collapse:
    enqueue every layer NO-fence, sync once at the forward boundary). MMEL_OWNED_PERLAYER_SYNC=1
    reverts to one fence per layer -- the CORRECT-but-slow safe first bring-up (it orders
    our stream after torch's ops per layer; reuse_bridge_device_and_stream note, :1972)."""
    return os.environ.get("MMEL_OWNED_PERLAYER_SYNC", "").strip().lower() not in (
        "1", "true", "yes", "on")


def _run_decoder_layer_via_chain_replay(layer, hidden_states):
    """Lazy bridge to mmel_chain_layer.run_decoder_layer_via_chain_replay (on-card).

    Imported lazily so importing mmel_l6_chain on a CPU box that lacks the chain
    module never trips; the chain module imports torch/habana/libSynapse only inside
    its on-card methods. A missing chain module under MMEL_CHAIN_LAYER=1 is a LOUD
    MMELChainError (the operator asked for the full-layer chain and must be told it
    isn't present), never a silent fall-through to stock."""
    try:
        from vllm_gaudi.v1.worker.mmel_chain_layer import (  # type: ignore
            run_decoder_layer_via_chain_replay)
    except Exception:  # noqa: BLE001 — try the path-based sibling import next
        run_decoder_layer_via_chain_replay = _import_chain_layer_entry()
    if run_decoder_layer_via_chain_replay is None:
        raise MMELChainError(
            "[mmel.l6_chain] MMEL_CHAIN_LAYER=1 (device-resident full-layer chain) "
            "requested but mmel_chain_layer is not importable. Ensure "
            "mmel_chain_layer.py ships in vllm_gaudi.v1.worker, or unset "
            "MMEL_CHAIN_LAYER to use the op-for-op G1 forward.")
    return run_decoder_layer_via_chain_replay(layer, hidden_states)


def _import_chain_layer_entry():
    """Path-based fallback import of the chain-layer entry (sibling file)."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "mmel_chain_layer.py")
    if not os.path.isfile(cand):
        return None
    try:
        import sys as _sys
        modname = "vllm_gaudi.v1.worker.mmel_chain_layer"
        spec = importlib.util.spec_from_file_location(modname, cand)
        mod = importlib.util.module_from_spec(spec)
        # Register BEFORE exec so @dataclass's machinery resolves cls.__module__ in
        # sys.modules (py3.14; same fix as the island entry).
        _sys.modules[modname] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.run_decoder_layer_via_chain_replay
    except Exception:  # noqa: BLE001
        return None


def _chain_layer_covers_bucket(hidden_states) -> bool:
    """Whether the live forward geometry == the M=1 decode bucket the 7-island chain
    was captured at (so the seam engages the chain) vs a forward it does NOT cover —
    PREFILL ([128,6144]), any non-M=1 decode bucket, the warmup buckets — which is
    delegated to the G1 op-for-op spine below.

    THE 2026-06-22 0-WARMED FIX: previously the seam routed EVERY forward (incl. the
    prompt/prefill) to the chain; the chain's M=1 static-shape guard correctly REFUSED
    the [128,6144] prefill but that raise KILLED the worker on the FIRST forward, so no
    M=1 decode step ever warmed/logged (0 WARMED) and the serve fell to stock (false
    flat). Now the chain drives ONLY its captured bucket; everything else flows through
    G1 (byte-exact stock, handles any M). Lazy bridge to
    mmel_chain_layer.chain_layer_covers; card-free; never raises (undecidable => False
    => delegate, never crash). A genuinely MISMATCHED M=1 capture still raises LOUD at
    the chain's own warmup/assert_live_hidden — this predicate only routes, it does not
    weaken that gate."""
    try:
        from vllm_gaudi.v1.worker.mmel_chain_layer import (  # type: ignore
            chain_layer_covers)
    except Exception:  # noqa: BLE001 — path-based sibling import next
        chain_layer_covers = _import_chain_layer_coverage()
    if chain_layer_covers is None:
        # The chain module isn't importable. Do NOT claim coverage here — the LOUD
        # "module not importable" error belongs to _run_decoder_layer_via_chain_replay
        # (raised when the chain IS asked to drive a covered bucket), not to a silent
        # prefill. Returning False here means: if the module is missing AND the gate is
        # on, the operator still gets the LOUD raise the instant a decode bucket would
        # have engaged the chain (we never reach a "covered" decision without it).
        return False
    try:
        return bool(chain_layer_covers(hidden_states))
    except Exception:  # noqa: BLE001 — the predicate is contractually non-raising
        return False


def _import_chain_layer_coverage():
    """Path-based fallback import of the chain-layer coverage predicate (sibling)."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "mmel_chain_layer.py")
    if not os.path.isfile(cand):
        return None
    try:
        import sys as _sys
        modname = "vllm_gaudi.v1.worker.mmel_chain_layer"
        if modname in _sys.modules:
            return getattr(_sys.modules[modname], "chain_layer_covers", None)
        spec = importlib.util.spec_from_file_location(modname, cand)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[modname] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "chain_layer_covers", None)
    except Exception:  # noqa: BLE001
        return None


# One-shot LOUD breadcrumb (per worker) so the A/B can SEE that the chain engaged on
# the decode bucket while prefill/other buckets used G1 — not a silent stock pass.
_CHAIN_LAYER_DELEGATE_LOGGED = False
_CHAIN_LAYER_ENGAGE_LOGGED = False
# Same for the CORRECT static-tail path (MMEL_CHAIN_STATIC=1).
_CHAIN_STATIC_DELEGATE_LOGGED = False
_CHAIN_STATIC_ENGAGE_LOGGED = False
# One-shot proof that _run_moe_block was ENTERED with MMEL_CHAIN_STATIC=1 (the static
# block reached at all), emitted at WARNING BEFORE the coverage decision. This makes a
# zero-breadcrumb run unambiguous: this line PRESENT + no ENGAGE/DELEGATE => coverage
# predicate itself never returned (an exception path); this line ABSENT => _run_moe_block
# was never reached with the static flag (a wiring/flag-propagation bug). The 2026-06-26
# "0 breadcrumbs" report could not distinguish these because the engage/delegate lines
# were INFO (filtered) and there was no WARNING-level entry proof.
_CHAIN_STATIC_ENTERED_LOGGED = False
# Same for the LIVE-FED MLA attention chain (MMEL_CHAIN_LIVE=1).
_CHAIN_LIVE_ENGAGE_LOGGED = False
_CHAIN_LIVE_DELEGATE_LOGGED = False
_CHAIN_LIVE_CONFLICT_LOGGED = False


def _subchain_b_covers_bucket(hidden_states) -> bool:
    """Whether the live POST-attention hidden geometry == the M=1 decode bucket the
    static tail engages at ([1,6144] = 12288 B). Lazy bridge to
    mmel_chain_layer.chain_layer_covers (the MoE tail's input is the post-norm
    output, same byte size as the layer hidden). Card-free; never raises."""
    # WIRE (MMEL_BACKEND=hpu): the owned-forward wire captures its OWN cap_tail<idx>
    # recipe, so it must NOT gate on the Artifact-A 7-island plan (which may be absent
    # -> captured_layer_input_nbytes returns None -> coverage False -> the owned path
    # never engages -> never captures: the chicken-and-egg that made the first bring-up
    # silently DELEGATE). Engage directly on the M=1 decode bucket ([1,6144] bf16 =
    # 12288 B): pure shape arithmetic, never a device touch, never raises. Prefill/
    # non-M=1 (nbytes != 12288) still delegate to the G1 spine (byte-exact stock).
    if _owned_backend_enabled():
        try:
            numel = 1
            for d in tuple(hidden_states.shape):
                numel *= int(d)
            return (numel * 2) == 12288
        except Exception:  # noqa: BLE001 — contractually non-raising
            return False
    try:
        from vllm_gaudi.v1.worker.mmel_chain_layer import (  # type: ignore
            chain_layer_covers)
    except Exception:  # noqa: BLE001 — path-based sibling import next
        chain_layer_covers = _import_chain_layer_coverage()
    if chain_layer_covers is None:
        return False
    try:
        return bool(chain_layer_covers(hidden_states))
    except Exception:  # noqa: BLE001 — the predicate is contractually non-raising
        return False


def _captured_plan_undecidable() -> bool:
    """True iff the captured-bucket coverage plan cannot be resolved at all (captures
    missing/malformed) -> the chain can NEVER engage on ANY bucket (the 0-WARMED
    false-flat). DISTINCT from 'this forward is not the M=1 bucket' (a normal,
    decidable delegation). Card-free; never raises. Used by the STRICT-mode guard to
    raise ONLY on the genuinely-can-never-engage regression, not on a routine prefill
    delegate. Resolved via mmel_chain_layer.captured_layer_input_nbytes (returns None
    when the plan is undecidable)."""
    try:
        from vllm_gaudi.v1.worker.mmel_chain_layer import (  # type: ignore
            captured_layer_input_nbytes)
    except Exception:  # noqa: BLE001 — path-based sibling import next
        captured_layer_input_nbytes = None
        try:
            import importlib.util
            here = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(here, "mmel_chain_layer.py")
            if os.path.isfile(cand):
                import sys as _sys
                modname = "vllm_gaudi.v1.worker.mmel_chain_layer"
                if modname in _sys.modules:
                    captured_layer_input_nbytes = getattr(
                        _sys.modules[modname], "captured_layer_input_nbytes", None)
                else:
                    spec = importlib.util.spec_from_file_location(modname, cand)
                    m = importlib.util.module_from_spec(spec)
                    _sys.modules[modname] = m
                    spec.loader.exec_module(m)  # type: ignore[union-attr]
                    captured_layer_input_nbytes = getattr(
                        m, "captured_layer_input_nbytes", None)
        except Exception:  # noqa: BLE001
            captured_layer_input_nbytes = None
    if captured_layer_input_nbytes is None:
        # cannot even resolve the predicate function -> treat as undecidable.
        return True
    try:
        return captured_layer_input_nbytes() is None
    except Exception:  # noqa: BLE001 — contractually non-raising; treat error as undecidable
        return True


# ----------------------------------------------------------------------------- #
#  LIVE-FED MLA ATTENTION CHAIN (MMEL_CHAIN_LIVE=1) — seam bridges (card-free).   #
#                                                                               #
#  These lazy-bridge to mmel_chain_layer's live symbols (live_attn_covers,        #
#  build_live_ctx_for_layer, run_attn_chain_via_live_replay) the SAME way the      #
#  layer/static bridges do, so importing mmel_l6_chain on a CPU box never trips    #
#  and the chain module's torch/libSynapse touch stays inside its on-card methods. #
# ----------------------------------------------------------------------------- #
def _chain_layer_symbol(name: str):
    """Lazy-import a named symbol from mmel_chain_layer (package import, then path
    fallback). Returns None if unavailable. Card-free; never raises."""
    try:
        from vllm_gaudi.v1.worker import mmel_chain_layer as _m  # type: ignore
        return getattr(_m, name, None)
    except Exception:  # noqa: BLE001 — path-based sibling import next
        pass
    try:
        import importlib.util
        import sys as _sys
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "mmel_chain_layer.py")
        if not os.path.isfile(cand):
            return None
        modname = "vllm_gaudi.v1.worker.mmel_chain_layer"
        if modname in _sys.modules:
            return getattr(_sys.modules[modname], name, None)
        spec = importlib.util.spec_from_file_location(modname, cand)
        m = importlib.util.module_from_spec(spec)
        _sys.modules[modname] = m   # register before exec (py3.14 frozen-dataclass fix)
        spec.loader.exec_module(m)  # type: ignore[union-attr]
        return getattr(m, name, None)
    except Exception:  # noqa: BLE001
        return None


def _live_attn_covers_bucket(hidden_states) -> bool:
    """Whether the live hidden geometry == the captured live-attn [M,H] decode bucket
    (so the seam routes input_norm+self_attn through the live chain) vs a forward it does
    NOT cover (prefill / non-M / warmup -> stock). Lazy bridge to
    mmel_chain_layer.live_attn_covers; card-free; never raises (undecidable => False =>
    stock)."""
    fn = _chain_layer_symbol("live_attn_covers")
    if fn is None:
        return False
    try:
        return bool(fn(hidden_states))
    except Exception:  # noqa: BLE001 — the predicate is contractually non-raising
        return False


def _find_mla_attn_module(attn):
    """Find the submodule that owns ``.kv_cache`` (the HPUMLAAttention wrapper nested in
    the decoder ``self_attn``; oot_mla.forward reads ``self.kv_cache``). Cheap attr
    candidates first, then a shallow module walk. Returns None if not found. Never
    raises."""
    if attn is None:
        return None
    try:
        for nm in ("mla_attn", "attn", "mla", "impl"):
            c = getattr(attn, nm, None)
            if c is not None and getattr(c, "kv_cache", None) is not None:
                return c
        if getattr(attn, "kv_cache", None) is not None:
            return attn
        modules = getattr(attn, "modules", None)
        if callable(modules):
            for m in modules():
                if getattr(m, "kv_cache", None) is not None:
                    return m
    except Exception:  # noqa: BLE001
        return None
    return None


def _resolve_live_attn_sources(layer):
    """Resolve ``(attn_metadata, kv_cache)`` the live MLA chain binds, from the vLLM
    forward context + the layer's attention submodule (oot_mla: ``attn_metadata =
    get_forward_context().attn_metadata`` — possibly a per-layer dict keyed by
    ``layer_name``; ``kv_cache = <mla module>.kv_cache``). Best-effort; returns
    ``(None, None)`` on any miss; never raises (the seam then delegates to stock)."""
    attn = getattr(layer, "self_attn", None)
    mla_mod = _find_mla_attn_module(attn)
    kv_cache = getattr(mla_mod, "kv_cache", None) if mla_mod is not None else None
    attn_metadata = None
    try:
        from vllm.forward_context import get_forward_context  # type: ignore
        fc = get_forward_context()
        attn_metadata = getattr(fc, "attn_metadata", None)
        if isinstance(attn_metadata, dict):
            ln = (getattr(mla_mod, "layer_name", None)
                  or getattr(attn, "layer_name", None))
            attn_metadata = attn_metadata.get(ln) if ln is not None else None
    except Exception:  # noqa: BLE001 — no forward context (off-serve) => delegate
        attn_metadata = None
    return attn_metadata, kv_cache


def _build_live_attn_ctx(layer, positions):
    """Assemble the MLA island's ``live_ctx`` (latent_kv/q_cos/q_sin/pad_mask, keyed by
    recipe tensor name) from the live kv_cache[0] + attn_metadata. Lazy bridge to
    mmel_chain_layer.build_live_ctx_for_layer. Card-free; never raises; returns None on
    any miss (the seam delegates to stock)."""
    import sys as _ds
    _g = globals(); _g['_LCD'] = _g.get('_LCD', 0) + 1
    _dbg = _g['_LCD'] <= 6
    fn = _chain_layer_symbol("build_live_ctx_for_layer")
    if fn is None:
        if _dbg:
            print("[LCTX] FAIL symbol build_live_ctx_for_layer is None", file=_ds.stderr, flush=True)
        return None
    try:
        attn_metadata, kv_cache = _resolve_live_attn_sources(layer)
        if attn_metadata is None or kv_cache is None:
            if _dbg:
                print("[LCTX] FAIL sources am=%s kv=%s" % (
                    attn_metadata is not None, kv_cache is not None), file=_ds.stderr, flush=True)
            return None
        if _dbg:
            try:
                _k0 = kv_cache[0]
                print("[KVC] kv_cache[0] nbytes=%s shape=%s dtype=%s (recipe flat_cache wants 222068736=NB1506)" % (
                    _k0.numel() * _k0.element_size(), tuple(_k0.shape), _k0.dtype), file=_ds.stderr, flush=True)
            except Exception as _ke:  # noqa: BLE001
                print("[KVC] err %s" % _ke, file=_ds.stderr, flush=True)
        _r = fn(layer, attn_metadata, kv_cache, positions)
        if _r is not None:
            # latent-write side-channel: the slot_mapping + kv_cache[0] the chain's run()
            # uses to write this token's latent KV BEFORE the read-only MLA op (popped there
            # before the have==need check; the cache-write stock does via reshape_and_cache_mla).
            try:
                _r["__slot__"] = getattr(attn_metadata, "slot_mapping", None)
                _r["__kv0__"] = (kv_cache[0]
                                 if (kv_cache is not None and len(kv_cache) >= 1) else None)
                # ROUTE B side-channel: the stock MLA module + its inputs, so run() can call
                # the byte-exact HPUMLAAttention instead of the drifting custom mla_decode kernel.
                _r["__layer__"] = layer
                _r["__pos__"] = positions
                _r["__am__"] = attn_metadata
                _r["__kv__"] = kv_cache
            except Exception:  # noqa: BLE001
                pass
        if _dbg:
            print("[LCTX] build_live_ctx returned None=%s" % (_r is None), file=_ds.stderr, flush=True)
        return _r
    except Exception as _e:  # noqa: BLE001 — contractually non-raising
        if _dbg:
            print("[LCTX] EXC %s: %s" % (type(_e).__name__, _e), file=_ds.stderr, flush=True)
        return None


def _run_attn_chain_via_live_replay(layer, hidden_states, live_ctx):
    """Lazy bridge to mmel_chain_layer.run_attn_chain_via_live_replay (the on-card
    device replay of the live-fed MLA attention chain). Raises if the chain module is
    absent — the seam's ``_try_live_attn`` wraps this and delegates to stock on ANY
    failure, so the live forward never raises out of the seam."""
    fn = _chain_layer_symbol("run_attn_chain_via_live_replay")
    if fn is None:
        raise MMELChainError(
            "[mmel.l6_chain] MMEL_CHAIN_LIVE=1 (live-fed MLA attention chain) requested "
            "but mmel_chain_layer.run_attn_chain_via_live_replay is not importable. "
            "Ensure mmel_chain_layer.py ships in vllm_gaudi.v1.worker, or unset "
            "MMEL_CHAIN_LIVE.")
    return fn(layer, hidden_states, live_ctx)


def _oproj_all_reduce_partial(attn_out):
    """Complete the RowParallel o_proj of the live attention chain: all-reduce the
    chain's PER-RANK PARTIAL across the tensor-parallel group — the same reduction stock
    ``RowParallelLinear`` (``reduce_results=True``) applies inside ``o_proj``
    (_src/vllm_attn/mla.py:181). The chain's captured o_proj island is the bare per-rank
    GEMM (a collective cannot live inside the single-rank raw-synLaunch chain,
    LIVE_MLA_DECOMP_DESIGN §7 risk-4), so the cross-rank sum is done HERE, on the torch
    side, right after the chain returns and BEFORE the output is used as the residual
    addend.

    TP=1 (or any single-rank / CPU self-test path) => NO-OP (the partial IS the full
    output, exactly as stock RowParallelLinear no-ops the reduce at world_size 1).
    Raises only if the vllm.distributed collective is genuinely unavailable in a TP>1
    serve — ``_try_live_attn`` catches that and DELEGATES to stock (which all-reduces),
    so a TP>1 forward is NEVER served a silently-unreduced partial."""
    from vllm.distributed import (  # lazy — serve/card only; never imported off-serve
        get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce)
    if get_tensor_model_parallel_world_size() <= 1:
        return attn_out
    return tensor_model_parallel_all_reduce(attn_out)


def _try_live_attn(layer, positions, hidden_states, residual):
    """Drive THIS layer's input_norm + self_attn via the LIVE-FED MLA chain
    (MMEL_CHAIN_LIVE=1) when covered. Returns ``(attn_out, residual_out)`` on success
    (the o_proj output = the pre-post-norm attention addend, plus the threaded residual,
    ready for the existing post-attention tail), or None to DELEGATE to stock.

    NEVER RAISES — any failure (conflicting driver, not the captured bucket, live_ctx
    unresolvable, off-card, missing recipe, replay error) returns None so the stock
    steps 1+2 run byte-identically. This is the 'undecidable => stock' rule.

    Residual contract (byte-exact to stock deepseek_v2.py:1176-1180): the chain's
    input_norm island is a PLAIN RMSNorm, so it receives the POST-fused-add hidden
    (``residual + hidden`` for layers>0; ``hidden`` for layer 0) as its norm input, and
    the new residual is threaded forward unchanged into post_attention_layernorm."""
    global _CHAIN_LIVE_ENGAGE_LOGGED, _CHAIN_LIVE_DELEGATE_LOGGED
    global _CHAIN_LIVE_CONFLICT_LOGGED
    # mutual exclusion: a conflicting layer driver wins; live-attn DEFERS (never raise).
    if _chain_layer_enabled() or _chain_static_enabled() or _moe_island_replay_enabled():
        if not _CHAIN_LIVE_CONFLICT_LOGGED:
            _CHAIN_LIVE_CONFLICT_LOGGED = True
            logger.warning(
                "[mmel.l6_chain][LIVE-ATTN DEFER] MMEL_CHAIN_LIVE=1 set alongside a "
                "conflicting layer driver (MMEL_CHAIN_LAYER / MMEL_CHAIN_STATIC / "
                "MMEL_CHAIN_MOE_OP=2); the live-fed attention chain DEFERS to it (live-attn "
                "is the ATTENTION driver only). Unset the other to engage the correctness "
                "chain.")
        return None
    try:
        if not _live_attn_covers_bucket(hidden_states):
            if not _CHAIN_LIVE_DELEGATE_LOGGED:
                _CHAIN_LIVE_DELEGATE_LOGGED = True
                try:
                    _shp = tuple(hidden_states.shape)
                except Exception:  # noqa: BLE001
                    _shp = "?"
                logger.info(
                    "[mmel.l6_chain][LIVE-ATTN DELEGATE] MMEL_CHAIN_LIVE=1 but live hidden "
                    "%s is NOT the captured live-attn decode bucket (prefill / non-M / "
                    "warmup) -> stock input_norm+self_attn; the live chain engages on its "
                    "captured [M,H] bucket (set MMEL_CHAIN_M / MMEL_CAP_DIR to match).",
                    _shp)
            return None
        live_ctx = _build_live_attn_ctx(layer, positions)
        if live_ctx is None:
            if not _CHAIN_LIVE_DELEGATE_LOGGED:
                _CHAIN_LIVE_DELEGATE_LOGGED = True
                logger.info(
                    "[mmel.l6_chain][LIVE-ATTN DELEGATE] MMEL_CHAIN_LIVE=1, bucket covered, "
                    "but the live MLA context (latent_kv<-kv_cache[0]; block_table<-block_list; "
                    "pad_mask<-attn_bias; q_cos/q_sin<-rotary_emb.cos_sin_cache@positions) "
                    "could not be assembled this forward -> stock attention (undecidable => "
                    "stock; a source — kv_cache / attn_metadata field / rotary table / "
                    "positions — was absent or the rope style was indeterminate).")
            return None
        # reproduce the stock fused-add residual to get the rmsnorm INPUT the chain norms.
        if residual is None:
            residual_out = hidden_states.clone()
            norm_input = hidden_states
        else:
            # The residual threaded from the prior layer may carry a different RANK than
            # this layer's hidden (e.g. (32,1,H) vs (32,H)). A naive add BROADCASTS
            # ((32,1,H)+(32,H) -> (32,32,H) = 32x = the 12.5MB STATIC-SHAPE blowup).
            # Reshape the residual to the hidden's shape (identical nbytes => a free view)
            # so the fused-add is ELEMENT-WISE, exactly as stock input_layernorm does it.
            _res = (residual.reshape(hidden_states.shape)
                    if tuple(residual.shape) != tuple(hidden_states.shape) else residual)
            residual_out = _res + hidden_states
            norm_input = residual_out
        import sys as _ds5
        _gg = globals(); _gg['_NI'] = _gg.get('_NI', 0) + 1
        if _gg['_NI'] <= 6:
            print("[NI] hidden=%s residual=%s norm_input=%s" % (
                tuple(hidden_states.shape),
                (tuple(residual.shape) if residual is not None else None),
                tuple(norm_input.shape)), file=_ds5.stderr, flush=True)
        attn_out = _run_attn_chain_via_live_replay(layer, norm_input, live_ctx)
        # o_proj is RowParallel => the chain's o_proj island emits the PER-RANK PARTIAL.
        # Complete the TP=8 reduction (what stock o_proj's RowParallelLinear does) BEFORE
        # attn_out is used as the residual addend. TP=1 / CPU self-test path => no-op.
        attn_out = _oproj_all_reduce_partial(attn_out)
        if not _CHAIN_LIVE_ENGAGE_LOGGED:
            _CHAIN_LIVE_ENGAGE_LOGGED = True
            logger.info(
                "[mmel.l6_chain][LIVE-ATTN ENGAGE] MMEL_CHAIN_LIVE=1: input_norm + "
                "self_attn driven by the device-resident LIVE-FED MLA chain (input_norm-> "
                "fused_qkv_a->q_a_norm->q_b->kv_a_norm->MLA(live-KV)->o_proj, every "
                "activation edge LIVE; o_proj per-rank partial all-reduced across TP; "
                "OUT = the pre-post-norm attention addend). post_norm + MoE run the "
                "stock/G1 tail.")
        return attn_out, residual_out
    except Exception as e:  # noqa: BLE001 — NEVER raise on the live forward: stock fallback.
        import sys as _ds4
        _gg = globals(); _gg['_TLE'] = _gg.get('_TLE', 0) + 1
        if _gg['_TLE'] <= 6:
            import traceback as _tb
            print("[TLA-EXC] %s: %s" % (type(e).__name__, e), file=_ds4.stderr, flush=True)
            _tb.print_exc(file=_ds4.stderr)
        if not _CHAIN_LIVE_DELEGATE_LOGGED:
            _CHAIN_LIVE_DELEGATE_LOGGED = True
            logger.warning(
                "[mmel.l6_chain][LIVE-ATTN DELEGATE] MMEL_CHAIN_LIVE=1 live attention "
                "replay failed (%s: %s) -> stock input_norm+self_attn this forward "
                "(coherent fallback; the live chain is the correctness path under "
                "validation). Capture the per-rank islands / set MMEL_CAP_DIR to engage.",
                type(e).__name__, e)
        return None


def _run_routed_moe_via_island_replay(layer, hidden_states):
    """Lazy bridge to mmel_moe_island.run_routed_moe_via_island_replay (on-card).

    Imported lazily (inside the call) so importing mmel_l6_chain on a CPU box that
    lacks the island module never trips; the island module itself imports torch/
    habana/libSynapse only inside its on-card methods. A missing island module under
    MMEL_CHAIN_MOE_OP=2 is a LOUD MMELChainError (the operator asked for the replay
    path and must be told it isn't present), never a silent fall-through to stock."""
    try:
        from vllm_gaudi.v1.worker.mmel_moe_island import (  # type: ignore
            run_routed_moe_via_island_replay)
    except Exception:  # noqa: BLE001 — try the path-based sibling import next
        run_routed_moe_via_island_replay = _import_island_entry()
    if run_routed_moe_via_island_replay is None:
        raise MMELChainError(
            "[mmel.l6_chain] MMEL_CHAIN_MOE_OP=2 (device-resident island replay) "
            "requested but mmel_moe_island is not importable. Ensure "
            "mmel_moe_island.py ships in vllm_gaudi.v1.worker, or unset "
            "MMEL_CHAIN_MOE_OP to use the stock MoE inside the fence-free forward.")
    return run_routed_moe_via_island_replay(layer, hidden_states)


def _import_island_entry():
    """Path-based fallback import of the island entry (sibling file)."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "mmel_moe_island.py")
    if not os.path.isfile(cand):
        return None
    try:
        import sys as _sys
        modname = "vllm_gaudi.v1.worker.mmel_moe_island"
        spec = importlib.util.spec_from_file_location(modname, cand)
        mod = importlib.util.module_from_spec(spec)
        # Register BEFORE exec so @dataclass's _is_type can resolve cls.__module__ in
        # sys.modules (py3.14 looks it up there; a path-import with a dotted name that
        # isn't registered makes the frozen dataclass machinery deref None).
        _sys.modules[modname] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.run_routed_moe_via_island_replay
    except Exception:  # noqa: BLE001
        return None


def _try_load_op(torch, so: str, ns: str, op: str) -> bool:
    """Best-effort load of ``so`` and confirm ``torch.ops.<ns>.<op>`` is registered.

    Returns True iff the op is registered AFTER the attempt. Never raises on a
    missing file or a load error — the caller decides (per the op-swap gate) whether
    an absence is fatal. LOUD-warns so a real serve still shows what happened."""
    if hasattr(torch.ops, ns) and hasattr(getattr(torch.ops, ns), op):
        return True
    if not os.path.isfile(so):
        logger.warning(
            "[mmel.l6_chain] op .so not found at %r (torch.ops.%s.%s not pre-loaded "
            "by the serve image either).", so, ns, op)
        return False
    try:
        torch.ops.load_library(so)
    except Exception as e:  # noqa: BLE001 — a load failure is non-fatal in baseline
        logger.warning("[mmel.l6_chain] torch.ops.load_library(%r) failed: %s", so, e)
        return False
    ok = hasattr(torch.ops, ns) and hasattr(getattr(torch.ops, ns), op)
    if not ok:
        logger.warning(
            "[mmel.l6_chain] loaded %r but torch.ops.%s.%s is still not registered "
            "(ABI / image drift).", so, ns, op)
    return ok


def _ensure_ops_loaded(need_mla: bool) -> None:
    """Make the proven op .so's available in torch.ops (idempotent), with the
    routed int4 op REQUIRED only when the op-swap is on (MMEL_CHAIN_MOE_OP=1).

    Baseline mode (MMEL_CHAIN_MOE_OP unset — the B3.3 fence-collapse measurement):
    G1's forward drives BOTH the attention spine and the MoE block through the
    layer's OWN stock submodules (which the serve image already backs with the
    proven device ops). G1 calls NEITHER torch.ops.hpu_moe_ext.fused_moe NOR
    torch.ops.hpu_mla_ext.mla_decode directly. So here we only BEST-EFFORT pre-load
    them (handy if a later path wants them) and a missing .so must NOT raise — a
    hard-require would block the recipe-count gate on a path that never uses the op.

    Op-swap mode (MMEL_CHAIN_MOE_OP=1): the routed int4 op IS called directly
    (_moe_routed_via_our_op), so its .so MUST be present + registered — LOUD-raise if
    not, naming the env to fix it. (That bind itself is still gated behind B4.1.)

    The serve image normally already registers these ops; this is belt-and-braces.
    Lazy/eager .so is chosen to match the live pipeline (htorch.is_lazy())."""
    import torch  # lazy — card only

    try:
        import habana_frameworks.torch as htorch  # type: ignore

        is_lazy = bool(htorch.is_lazy())
    except Exception:  # noqa: BLE001 — if we cannot tell, default to the lazy/serve .so
        is_lazy = True

    require_routed = moe_op_swap_enabled()

    if not _OPS_LOADED["moe"]:
        so = _routed_so_path(is_lazy)
        ok = _try_load_op(torch, so, "hpu_moe_ext", "fused_moe")
        if not ok and require_routed:
            # ONLY the op-swap path actually calls the routed op directly — so this is
            # the one place a missing/unregistered routed .so is fatal.
            raise MMELOpMissingError(
                f"[mmel.l6_chain] routed int4 MoE op .so not found/registered at "
                f"{so!r} but MMEL_CHAIN_MOE_OP=1 (the op-swap) requires it. Set "
                f"{MOE_ROUTED_SO_ENV} to the serve image's "
                "libmoe_fused_routed(_lazy).so (the in-container /opt path), or unset "
                "MMEL_CHAIN_MOE_OP to run the stock MoE inside the fence-free forward."
            )
        # In baseline mode we mark MoE 'resolved' regardless (the stock submodule is
        # what runs; ok just records whether the op happens to be available too).
        _OPS_LOADED["moe"] = True

    # The MLA-decode op is driven by the stock self_attn submodule (which the serve
    # image already backs with it), NOT by G1 directly — so it is best-effort here in
    # both modes: pre-load if available, never raise on absence (the stock spine has
    # its own op). need_mla just scopes the (cheap) attempt to the MLA families.
    if need_mla and not _OPS_LOADED["mla"]:
        so = _mla_so_path(is_lazy)
        _try_load_op(torch, so, "hpu_mla_ext", "mla_decode")
        _OPS_LOADED["mla"] = True


# ----------------------------------------------------------------------------- #
#  STATIC-SHAPE contract (G4 / W4.1) — the bucket-stable geometry.               #
#                                                                               #
#  These are pure arithmetic helpers (card-free testable). They convert the      #
#  content-dependent shapes that force per-step recompiles into bucket-stable    #
#  ones, exactly the four DSA serve-bug fixes (PROJECT_FINDINGS §7).            #
# ----------------------------------------------------------------------------- #
def next_pow2(n: int) -> int:
    """Smallest power-of-two >= n (>=1). Used for the DSA ``kk`` key-tile so the
    gather/topk recipe geometry is pow2-stable (the captured DSA fix)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def pad_to_multiple(n: int, multiple: int) -> int:
    """Round n UP to a multiple of `multiple` (>=1). Used to key-pad the gathered
    context width to 128-multiples so the sdpa/gather recipe is shape-stable."""
    if multiple <= 1:
        return n
    return ((n + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class StaticShape:
    """The bucket-stable shape the chained recipe is captured/replayed at.

    Built from the L5 ExecConfig + the serve's bucket (batch / context). Every
    field here is a STATIC integer for a given bucket — never derived from a live
    ``.item()`` — so the whole forward replays ONE recipe per bucket (no
    ``[SHAPE AGNOSTIC]`` re-fire, no ``synStatus 26``).

    Fields:
      batch          : the decode bucket batch (bs; 1 for single-stream).
      hidden_size    : H (the layer I/O width, e.g. 6144).
      gather_width   : the STATIC paged-context width (key-padded to 128 multiples).
      index_topk_pad : static DSA kept-token width (``index_topk + 1``; 0 if no DSA).
      kk             : pow2 key-tile for the DSA gather (0 if no DSA).
      n_routed       : routed expert count (the int4 op's E).
      top_k          : experts per token (the routed op's K).
    """

    batch: int
    hidden_size: int
    gather_width: int
    index_topk_pad: int
    kk: int
    n_routed: int
    top_k: int

    @classmethod
    def from_exec_cfg(cls, exec_cfg: Any, *, batch: int, context_len: int,
                      block_size: int = 128) -> "StaticShape":
        """Derive the bucket-stable shape. ``context_len`` is the serve's bucketed
        max context for this step (already a bucket constant, not a live value);
        ``block_size`` is the paged-KV block (128 on this stack)."""
        if batch < 1:
            raise MMELShapeError(f"[mmel.l6_chain] batch must be >=1, got {batch}.")
        if context_len < 1:
            raise MMELShapeError(
                f"[mmel.l6_chain] context_len must be >=1, got {context_len}.")
        H = int(exec_cfg.hidden_size)
        # key-pad the gathered context to a 128-multiple (the captured key-pad fix).
        gather_width = pad_to_multiple(int(context_len), int(block_size))
        if getattr(exec_cfg, "has_dsa", False) and exec_cfg.dsa is not None:
            index_topk = int(exec_cfg.dsa.index_topk)
            # static kept-token width = index_topk + 1 (never max_keep.item()).
            index_topk_pad = index_topk + 1
            kk = next_pow2(index_topk_pad)
        else:
            index_topk_pad = 0
            kk = 0
        return cls(
            batch=int(batch),
            hidden_size=H,
            gather_width=int(gather_width),
            index_topk_pad=int(index_topk_pad),
            kk=int(kk),
            n_routed=int(exec_cfg.n_routed_experts),
            top_k=int(exec_cfg.top_k),
        )

    def assert_live_tensor(self, name: str, got_shape, *, dim: int, expect: int) -> None:
        """LOUD guard: a live tensor's ``dim`` must equal the static ``expect`` the
        recipe was captured at. A drift here is the mixed-shape crash class — we
        refuse to launch a recipe against a tensor whose geometry would re-fire it."""
        if int(got_shape[dim]) != int(expect):
            raise MMELShapeError(
                f"[mmel.l6_chain] STATIC-SHAPE violation: live tensor {name!r} has "
                f"dim[{dim}]={int(got_shape[dim])} but the captured recipe expects "
                f"{int(expect)} (bucket {self}). The forward would re-fire / hit "
                "synStatus 26. Re-capture this bucket or eager-island the step."
            )


# ----------------------------------------------------------------------------- #
#  ROUTING METADATA (static, on-device-once) — the int4 routed op's contract.    #
#                                                                               #
#  The proven routed op is torch.ops.hpu_moe_ext.fused_moe(x, w13p, s13, z13,    #
#  w2p, s2, z2, rws): a SINGLE-RECIPE K-expert op where the router weights `rws` #
#  already fold the per-token top-k selection to per-expert scalars (the         #
#  capture's [1,1] scalar fold). G1 must produce `rws` WITHOUT a host sync —     #
#  the top-k stays on-device (the no-.item() rule). This builder is the pure     #
#  arithmetic of that fold; the device tensor ops are applied in the forward.    #
# ----------------------------------------------------------------------------- #
def route_fold_is_static(n_routed: int, top_k: int) -> bool:
    """The single-recipe routed op is only shape-stable when the routed-expert set
    is a STATIC K (not a content-dependent variable expert count). At bs=1 the
    captured op uses K experts with scalar router weights — static by construction.
    Returns whether (n_routed, top_k) admits the static single-recipe fold."""
    return 1 <= top_k <= n_routed


# ----------------------------------------------------------------------------- #
#  THE CHAINED DECODER FORWARD (G1 core) — the real body of the seam's stub.     #
#                                                                               #
#  Signature + return are IDENTICAL to DeepseekV2DecoderLayer.forward (the seam  #
#  asserts this). NO host sync inside (no .item()); dense-vs-MoE per layer is    #
#  the L5 oracle. The op composition is grounded in the captured island ops.     #
# ----------------------------------------------------------------------------- #
def run_chained_decoder_layer(
    layer: Any,
    positions: "Any",
    hidden_states: "Any",
    residual: "Optional[Any]",
    *,
    exec_cfg: Any,
    llama_4_scaling: "Optional[Any]" = None,
):
    """Drive ONE decoder layer via the device-resident chained execution.

    Contract honored (so the swap stays byte-exact-capable and fence-collapsing):
      * I/O identical to stock: ``(positions, hidden_states, residual)`` in,
        ``(hidden_states, residual)`` out; ``llama_4_scaling`` threaded unchanged.
      * NO host sync inside (no ``.item()`` / ``.cpu()``): the routing top-k stays
        on-device; the whole body is one fence-free op chain the bridge captures.
      * dense-vs-MoE = ``exec_cfg.layer_is_dense(layer.layer_idx)`` (the L5 oracle).
      * static shape per bucket (the ``StaticShape`` guards); a residual
        data-dependent step is delegated to the layer's own stock submodule as an
        EAGER ISLAND (correct; captures the rest) — never a host sync.

    BRING-UP STANCE (first draft, on-card-gated): G1 composes the proven device
    ops into one fence-free forward by REUSING the layer's own stock submodules
    for the projection/attention spine (which already call the device ops) and
    routing OUR chained int4 MoE op for the expert block, all WITHOUT inserting a
    per-op ``mark_step``. The byte-exact A/B harness (seam) is the arbiter; the
    recipe-count gate (B3.3) measures whether the bridge collapsed the fences. A
    bind/shape error RAISES (LOUD) so a bring-up gate catches it — the seam's
    coherent wrapper keeps the live generation safe.
    """
    import torch  # lazy — card only

    layer_idx = int(getattr(layer, "layer_idx", -1))
    if layer_idx < 0:
        raise MMELChainError(
            "[mmel.l6_chain] layer has no layer_idx; cannot decide dense/MoE or "
            "validate the static-shape bucket. Is this a DeepseekV2DecoderLayer?"
        )

    # ── MMEL_BACKEND=hpu — the OWNED-FORWARD live-serve wire (roadmap closure) ────
    # On the covered M=1 decode bucket, drive the WHOLE layer through the on-card
    # HpuBackend chain (capture-once / bind-live / chain / one-sync). Returns
    # (hidden, residual) exactly like stock. On a NON-covered bucket (prefill / non-M=1
    # / warmup) it returns None and we FALL THROUGH to the G1/static spine below
    # (byte-exact stock) — never a silent stock-shadow on the covered bucket, never a
    # raise on prefill. Default-off: MMEL_BACKEND unset => this branch is not taken.
    if _owned_backend_enabled():
        _owned = _run_owned_forward_via_hpu_backend(
            layer, positions, hidden_states, residual,
            exec_cfg=exec_cfg, llama_4_scaling=llama_4_scaling)
        if _owned is not None:
            return _owned
        # not covered: delegate to the existing spine below (byte-exact stock).

    # ── THE CORRECT STATIC TAIL (MMEL_CHAIN_STATIC=1) — the audit's recommended ──
    # FIRST landing. Run the stock-faithful op-for-op forward below (input_norm +
    # self_attn[MLA+DSA] + fused-add post_attention_layernorm = STOCK, byte-exact,
    # context-correct) and drive ONLY the static tail (the MoE block — the dominant
    # per-layer dispatch share) device-side via the proven S1 island replay, on the
    # M=1 decode bucket. UNLIKE the all-7 MMEL_CHAIN_LAYER path, NOTHING
    # context-dynamic is device-replayed: attention stays on the stock bucketed path,
    # so the full layer is byte-comparable to stock layer.forward end-to-end. This is
    # a MODE flag for the G1 forward below: it engages the device MoE tail (the same
    # mechanism MMEL_CHAIN_MOE_OP=2 uses) on covered buckets, with LOUD breadcrumbs.
    # Mutually exclusive with MMEL_CHAIN_LAYER (the unsound all-7 it supersedes).
    if _chain_static_enabled():
        if _chain_layer_enabled():
            raise MMELChainError(
                "[mmel.l6_chain] BOTH MMEL_CHAIN_STATIC=1 and MMEL_CHAIN_LAYER=1 are "
                "set; they are mutually-exclusive layer drivers. MMEL_CHAIN_STATIC is "
                "the CORRECT static-tail path (attention on stock, byte-exact "
                "end-to-end); MMEL_CHAIN_LAYER is the unsound all-7 that device-"
                "replays the frozen MLA (ENGINE_BUILD_GAP_AUDIT). Pick ONE — prefer "
                "MMEL_CHAIN_STATIC.")
        # the static-tail engagement is realized inside _run_moe_block (below): on a
        # covered M=1 decode bucket it routes the MoE through the device island replay;
        # prefill/non-M=1 run the stock MoE submodule (delegate, byte-exact). The G1
        # op-for-op spine below is what keeps norm/attn/post-norm correct on stock.
        # (No early return: we fall through to the stock-faithful forward.)
        pass
    elif owned_all_layers_enabled():
        # #4b GATING INVARIANT: the all-layer device-resident forward is a scope-widener
        # for the static tail; it is meaningless without the static-tail spine. Refuse to
        # silently no-op — a benchmark must never measure a flag that did nothing.
        raise MMELChainError(
            "[mmel.l6_chain] MMEL_OWNED_ALL_LAYERS=1 (the #4b all-layer device-resident "
            "forward) requires MMEL_CHAIN_STATIC=1 (the static-tail spine it widens). "
            "MMEL_OWNED_ALL_LAYERS only changes WHICH MoE layers the static tail covers "
            "(all of them, via the live-routing LazyOp); it does not stand alone. Set "
            "MMEL_CHAIN_STATIC=1 as well, or unset MMEL_OWNED_ALL_LAYERS.")

    # ── S2/S3: the FULL device-resident chained LAYER (MMEL_CHAIN_LAYER=1) ──────
    # When the operator opts into the full-layer chain, the WHOLE decoder layer
    # (norm->q/kv->MLA->o_proj->norm->MoE = 7 captured islands) is replayed on ONE
    # bridge stream behind ONE synStreamSynchronize (live-fed spine, staged branch
    # — RUN_CHAIN STEP-4, now in-serve), REPLACING the op-for-op G1 spine below.
    # This is the S2/S3 increment: 1 fence/layer in the LIVE step vs the per-layer
    # recipe-boundary syncs. It returns (hidden_states, residual) exactly like stock.
    #
    # IMPORTANT residual contract: the captured chain reproduces the layer's
    # fused-add-RMSNorm internally (the norm islands carry the residual add), so the
    # chained OUTPUT is the post-layer hidden. To keep the (hidden, residual) tuple
    # contract that threads through all 78 layers, we follow the SAME stock rule the
    # G1 path uses below: on layer 0 the residual is a clone of the pre-norm hidden;
    # otherwise the incoming residual is threaded. (The staged-branch chain is gated
    # byte-exact vs a SERIAL CONTROL, not vs live stock end-to-end — see
    # mmel_chain_layer's HONEST FRAME; the live-stock A/B is the S3-FULL gate.)
    if _chain_layer_enabled():
        # one-shot LOUD breadcrumbs (per worker) — declared up front so the gate below
        # is unambiguous about scope (the chain ENGAGES on the decode bucket vs DELEGATES
        # prefill/non-M=1 to G1 — never a silent stock pass).
        global _CHAIN_LAYER_ENGAGE_LOGGED, _CHAIN_LAYER_DELEGATE_LOGGED
        if _moe_island_replay_enabled():
            raise MMELChainError(
                "[mmel.l6_chain] BOTH MMEL_CHAIN_LAYER=1 and MMEL_CHAIN_MOE_OP=2 are "
                "set; they are mutually-exclusive layer drivers (the full-layer chain "
                "subsumes the MoE island as its tail). Pick ONE: MMEL_CHAIN_LAYER=1 "
                "for the 7-island layer chain (S2/S3), or MMEL_CHAIN_MOE_OP=2 for the "
                "single MoE island (S1).")
        # ── THE CAPTURED-BUCKET ROUTING GATE (the 2026-06-22 0-WARMED fix) ───────
        # The 7 islands are captured at the M=1 single-stream DECODE bucket
        # ([1,6144] bf16 = 12288 B). The seam must engage the chain ONLY on that
        # bucket; PREFILL ([128,6144]) + any non-M=1 decode/warmup bucket flow
        # through the G1 op-for-op spine below (byte-exact stock, handles every M).
        # BEFORE this gate, the seam routed prefill into the chain too — the chain's
        # M=1 shape guard correctly REFUSED it, but that raise killed the worker on
        # the FIRST (prompt) forward, so no decode step ever warmed => 0 WARMED, the
        # serve fell to stock, the A/B logged a false flat. This is NOT a silent
        # stock-fallback: a MISMATCHED M=1 capture still raises LOUD inside the chain
        # (warmup/assert_live_hidden unchanged); prefill is simply out of the M=1
        # chain's scope, and we log LOUD which forwards the chain drove vs delegated.
        if _chain_layer_covers_bucket(hidden_states):
            if not _CHAIN_LAYER_ENGAGE_LOGGED:
                _CHAIN_LAYER_ENGAGE_LOGGED = True
                logger.info(
                    "[mmel.l6_chain][CHAIN-ENGAGE] MMEL_CHAIN_LAYER=1: live hidden %s "
                    "matches the captured M=1 decode bucket -> driving the WHOLE layer "
                    "via the device-resident 7-island chain (warmup on first hit logs "
                    "'WARMED 7-island decode-layer chain'). Prefill/non-M=1 buckets are "
                    "delegated to the G1 spine (NOT a silent stock fall-through).",
                    tuple(hidden_states.shape))
                # ENGAGE-MANIFEST: the chain ENGAGED on its covered bucket -> the positive
                # signal a human/test greps to confirm the engine drove the layer (vs the
                # 0-WARMED false-flat where it never engaged). once_key separates this from
                # the MoE manifest.
                _engage().emit_manifest(
                    int(getattr(layer, "layer_idx", 0) or 0),
                    {"engine-chain": "LAYER-7island(ENGAGED M=1 decode)",
                     "STRICT": "ON" if _engage().strict_enabled() else "off"},
                    once_key="engine-chain")
            # static-shape guard on the live hidden (the chain replays at the captured
            # bucket; a drift would re-fire / hit synStatus 26).
            sshape = StaticShape.from_exec_cfg(
                exec_cfg,
                batch=int(hidden_states.shape[0]) if hidden_states.dim() >= 1 else 1,
                context_len=_resolve_context_len(positions, hidden_states))
            sshape.assert_live_tensor(
                "hidden_states", hidden_states.shape, dim=-1, expect=sshape.hidden_size)
            # the residual the tuple-contract threads forward (stock rule, byte-exact).
            if residual is None:
                residual_out = hidden_states.clone()
            else:
                residual_out = residual
            hidden_out = _run_decoder_layer_via_chain_replay(layer, hidden_states)
            if _MMEL_POISON and residual_out is not None:
                residual_out = residual_out * 1.03 + 0.01  # FORENSIC PROBE
            return hidden_out, residual_out
        # NOT the captured bucket (prefill / non-M=1): delegate to the G1 spine below.
        # LOUD one-shot breadcrumb so the A/B can see this is a deliberate route, not a
        # silent stock pass.
        if not _CHAIN_LAYER_DELEGATE_LOGGED:
            _CHAIN_LAYER_DELEGATE_LOGGED = True
            try:
                _shp = tuple(hidden_states.shape)
            except Exception:  # noqa: BLE001
                _shp = "?"
            logger.info(
                "[mmel.l6_chain][CHAIN-DELEGATE] MMEL_CHAIN_LAYER=1 but live hidden %s "
                "is NOT the captured M=1 decode bucket (prefill / non-M=1 / warmup) -> "
                "this forward runs the G1 op-for-op spine (byte-exact stock); the "
                "device-resident chain engages on the M=1 decode bucket it was captured "
                "at. (Set MMEL_CAP_DIR to a matching re-capture to widen the bucket.)",
                _shp)
            # STRICT false-null guard (same as the static tail): raise ONLY when the
            # capture plan is UNDECIDABLE (the chain can never engage on ANY bucket);
            # a routine prefill/non-M=1 delegate (decidable plan) is intended=False.
            _plan_undecidable = _captured_plan_undecidable()
            _engage().emit_manifest(
                int(getattr(layer, "layer_idx", 0) or 0),
                {"engine-chain": ("LAYER-7island(REQUESTED but plan UNDECIDABLE -> "
                                  "delegating ALL forwards to G1)" if _plan_undecidable
                                  else "LAYER-7island(delegating non-M=1; engages on M=1)"),
                 "STRICT": "ON" if _engage().strict_enabled() else "off"},
                once_key="engine-chain")
            if _plan_undecidable:
                _engage().engage_fallback(
                    "mmel_l6_chain.run_chained_decoder_layer:layer-plan-undecidable",
                    fast="MMEL_CHAIN_LAYER device-resident 7-island chain",
                    reason="the captured 7-island plan is UNDECIDABLE (captures "
                           "missing/malformed) -> the chain can NEVER engage (0-WARMED)",
                    slow="G1 op-for-op spine for EVERY forward",
                    intended=True, strict_env="MMEL_CHAIN_LAYER")
        # fall through to the G1 path below (no return).

    # Load the proven device ops (idempotent). The MLA op is only needed for the
    # MLA-latent attention families; GQA families (Qwen3) don't use it.
    need_mla = bool(getattr(exec_cfg, "attn_mode", "") == _ATTN_MLA_LATENT)
    _ensure_ops_loaded(need_mla=need_mla)

    # The static-shape bucket for this step (LOUD if the live geometry would
    # re-fire the recipe). batch/context come from the live tensor + forward ctx;
    # at bs=1 single-stream the batch is hidden_states.shape[0].
    bsz = int(hidden_states.shape[0]) if hidden_states.dim() >= 1 else 1
    ctx_len = _resolve_context_len(positions, hidden_states)
    sshape = StaticShape.from_exec_cfg(
        exec_cfg, batch=bsz, context_len=ctx_len)
    sshape.assert_live_tensor(
        "hidden_states", hidden_states.shape, dim=-1, expect=sshape.hidden_size)

    is_dense = bool(exec_cfg.layer_is_dense(layer_idx))

    # ── The fence-free layer chain ───────────────────────────────────────────
    # input_layernorm → self_attn (MLA-paged, the device op) → post_attn_norm →
    # MLP/MoE. We drive the projection + attention SPINE through the layer's own
    # stock submodules (they already call the proven device ops: q/kv/o-proj
    # GEMMs + the MLA-decode op + DSA indexer) so the BRANCH wiring (q/kv split,
    # paged gather by block table) is handled correctly — and we route the EXPERT
    # block through OUR chained int4 op for the fence-collapse, all with no
    # intervening host sync. This is the bring-up-correct composition; B5 replaces
    # the spine with the raw-CS recipe handoff for the deeper structural collapse.
    #
    # NOTE: the exact stock submodule names (input_layernorm / self_attn /
    # post_attention_layernorm / mlp) are asserted present below so a model-tree
    # drift fails LOUD rather than mis-binding.
    _assert_layer_submodules(layer)

    # The stock forward (deepseek_v2.py:1168-1216) does, in order, exactly:
    #   input_layernorm(residual-add) → self_attn → [fp16 overflow scale] →
    #   post_attention_layernorm(residual-add) → mlp → [fp16 overflow scale].
    # We reproduce it OP-FOR-OP with NO per-op mark_step so the bridge captures the
    # whole layer as one (few) recipe(s). Each branch below mirrors the stock source.

    # 1) input RMSNorm (fused, residual-add) — stock submodule (device rms_norm op).
    #    CRITICAL (byte-exact to stock deepseek_v2.py:1176-1180): on the first layer
    #    (residual is None) the residual MUST be a *clone* of the pre-norm
    #    hidden_states, NOT an alias. The single-arg RMSNorm returns a fresh tensor
    #    and does NOT preserve the input as the residual (layernorm.py:88-94 returns
    #    only `out`), so the caller owns the residual; and every later layer's fused
    #    add-RMSNorm is `fused_add_rms_norm.maybe_inplace` (layernorm.py:96), which
    #    MAY mutate the residual storage in place. Aliasing here (`residual =
    #    hidden_states`) lets a later in-place op / a reused embeddings buffer corrupt
    #    the residual stream that is threaded through all 78 layers and fused into the
    #    final `self.norm(hidden_states, residual)` (model forward) — producing a
    #    garbage final hidden state and EMPTY generations. `.clone()` is the stock
    #    contract and removes the hazard.
    #
    # ── MMEL_CHAIN_LIVE=1: device-resident LIVE-FED MLA attention chain (steps 1+2) ──
    # When enabled + the live hidden is the captured decode bucket, _try_live_attn drives
    # input_norm->fused_qkv_a->q_a_norm->q_b->kv_a_norm->MLA(live-KV)->o_proj device-side
    # (EVERY activation edge LIVE = the correctness fix for the STAGED MMEL_CHAIN_LAYER
    # branch) and returns (attn_out, residual_out), so the post-attention tail (steps 3-6)
    # runs unchanged. NEVER raises: on any miss (flag off, not the captured bucket,
    # live_ctx unresolvable, off-card, missing recipe, replay error) it returns None and
    # the stock input_layernorm + self_attn below run BYTE-IDENTICALLY. Default-off (flag
    # unset) => _live_attn_done stays False => the stock path is untouched.
    _live_attn_done = False
    import sys as _dsys
    _dg = globals()
    try:
        _dlead = 1
        for _x in hidden_states.shape[:-1]:
            _dlead *= int(_x)
        _dlast = int(hidden_states.shape[-1])
    except Exception:  # noqa: BLE001
        _dlead = -1; _dlast = -1
    if _dlast == 6144 and _dlead == 32 and _dg.get('_D32', 0) < 6:
        _dg['_D32'] = _dg.get('_D32', 0) + 1
        try:
            _dc = _live_attn_covers_bucket(hidden_states)
            _nb = hidden_states.numel() * hidden_states.element_size()
        except Exception as _e:  # noqa: BLE001
            _dc = 'err:%s' % _e; _nb = '?'
        print("[D32] shape=%s lead=%s nbytes=%s want=%s covers=%s live=%s" % (
            tuple(hidden_states.shape), _dlead, _nb, 32 * 6144 * 2, _dc,
            _chain_live_enabled()), file=_dsys.stderr, flush=True)
    if _chain_live_enabled():
        _live_r = _try_live_attn(layer, positions, hidden_states, residual)
        if _live_r is not None:
            hidden_states, residual = _live_r
            _live_attn_done = True

    if not _live_attn_done:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)

        # 2) self-attention (MLA-paged device op + DSA indexer on full layers). The
        #    stock self_attn reads the paged KV metadata from the forward context and
        #    runs the proven mla_decode op; we do NOT host-sync around it. Byte-exact to
        #    stock's attn_kwargs (see _call_self_attn): a non-MHA layer (MLA /
        #    DeepseekV2Attention) ALWAYS receives llama_4_scaling — even when it is None,
        #    since that arg is required-positional on those forwards — while an MHA layer
        #    omits it.
        hidden_states = _call_self_attn(layer, positions, hidden_states, llama_4_scaling)

    # 3) fp16 overflow scaling (stock deepseek_v2.py:1190-1201). BYTE-EXACT to stock:
    #    the guard is `not isinstance(self.self_attn, DeepseekAttention) and
    #    hidden_states.dtype == torch.float16` — keyed on the attention CLASS (every
    #    non-DeepseekAttention attn, i.e. MLA / DeepseekV2Attention), NOT on `use_mha`
    #    and NOT on an `rsf != 1.0` test (which stock does not have; with rsf==1.0 the
    #    scale is a harmless identity but the *branch* must still match so the captured
    #    recipe is identical). bf16 serve (GLM-5.2) has dtype != float16, so this whole
    #    block is a no-op on the live path — byte-identical to stock. Stock mutates in
    #    place (`*=`); we mirror that exactly. `_is_deepseek_attention` keys on the
    #    class NAME to avoid importing the vLLM model classes at module load.
    rsf = float(getattr(layer, "routed_scaling_factor", 1.0) or 1.0)
    if (not _is_deepseek_attention(layer.self_attn)) and \
            hidden_states.dtype == torch.float16:
        # Fix FP16 overflow: scale both hidden_states and residual before the next
        # rmsnorm (rmsnorm is scale-invariant). Residual is scaled only on layer 0
        # (it is shared across all layers).
        hidden_states *= 1.0 / rsf
        if layer_idx == 0:
            residual *= 1.0 / rsf

    # 4) post-attention RMSNorm (fused, residual-add).
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)

    # 5) MLP (dense prefix `DeepseekV2MLP`) or MoE (`DeepseekV2MoE`). In stock BOTH
    #    are the SAME attribute `self.mlp` and are called identically as
    #    `self.mlp(hidden_states)` (deepseek_v2.py:1205). We mirror that: the dense
    #    path calls the stock submodule; the MoE path routes through _run_moe_block
    #    (which, in the default/baseline op-off mode, ALSO just calls layer.mlp —
    #    byte-exact — and only diverges under the gated MMEL_CHAIN_MOE_OP=1 swap).
    #    The dense/MoE split here is the L5 oracle purely to pick the post-MLP scale
    #    target; the call itself is the stock submodule in both arms.
    if is_dense:
        hidden_states = layer.mlp(hidden_states)
    else:
        hidden_states = _run_moe_block(
            layer, hidden_states, exec_cfg=exec_cfg, sshape=sshape, torch=torch)

    # 6) post-MLP fp16 overflow scale (stock deepseek_v2.py:1209-1213). BYTE-EXACT:
    #    applied ONLY to the DENSE MLP (`isinstance(self.mlp, DeepseekV2MLP)`), never
    #    to MoE — the MoE submodule does its own scaling internally. We key on the L5
    #    `is_dense` oracle (== the layer's mlp being the dense DeepseekV2MLP) and the
    #    fp16 dtype, matching stock. No-op on the bf16 serve path.
    if is_dense and hidden_states.dtype == torch.float16:
        hidden_states *= 1.0 / rsf

    if _MMEL_POISON and residual is not None: residual = residual * 1.03 + 0.01  # FORENSIC PROBE (MMEL_POISON=1): perturb residual; coherent==chain-didn't-run
    return hidden_states, residual


def _is_deepseek_attention(self_attn: Any) -> bool:
    """True iff ``self_attn`` is the stock ``DeepseekAttention`` (the MHA-on-deepseek
    attention), keyed on the class name to avoid importing the vLLM model classes at
    module load (this file must ``py_compile`` + self-test on a CPU-only box).

    Stock keys the post-attention fp16 overflow-scale on
    ``not isinstance(self.self_attn, DeepseekAttention)`` (deepseek_v2.py:1191): the
    scale fires for every attention EXCEPT ``DeepseekAttention``. We reproduce that
    predicate by name. NOTE: ``DeepseekAttention`` is NOT the same as
    ``DeepseekV2Attention`` — the latter still gets the scale, so we match the exact
    class name, not a substring."""
    try:
        for kls in type(self_attn).__mro__:
            if kls.__name__ == "DeepseekAttention":
                return True
    except Exception:  # noqa: BLE001 — a duck-typed test stand-in has no __mro__
        return False
    return False


def _call_self_attn(layer: Any, positions, hidden_states, llama_4_scaling):
    """Call the stock self_attn faithfully — byte-exact to stock's attn_kwargs.

    Stock (deepseek_v2.py:1182-1188) builds::

        attn_kwargs = {"positions": positions, "hidden_states": hidden_states}
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling   # set EVEN IF None
        hidden_states = self.self_attn(**attn_kwargs)

    i.e. for a non-MHA layer (MLA / DeepseekV2Attention) ``llama_4_scaling`` is
    ALWAYS passed, even when it is ``None`` — because those attentions declare
    ``llama_4_scaling`` as a *required positional* arg with NO default
    (DeepseekV2MLAAttention.forward:1075-1081). GLM-5.2 sets no ``llama_4_scaling``
    config, so it is ``None`` at every step; the previous "pass only when not None"
    guard would call the MLA forward MISSING a required arg (a latent TypeError) and
    is NOT what stock does. Only an MHA layer (``use_mha is True``) omits it."""
    use_mha = bool(getattr(layer, "use_mha", False))
    attn_kwargs = {"positions": positions, "hidden_states": hidden_states}
    if not use_mha:
        attn_kwargs["llama_4_scaling"] = llama_4_scaling
    return layer.self_attn(**attn_kwargs)


def _run_moe_block(layer: Any, hidden_states, *, exec_cfg: Any,
                   sshape: "StaticShape", torch) -> Any:
    """Run the MoE block via OUR chained int4 routed op, static-shape, no host sync.

    First-draft composition: the MoE submodule's gate + shared-expert path is reused
    from the stock layer (those are already device ops); the ROUTED expert block is
    where OUR chained single-recipe op replaces the stock per-expert loop — that is
    the fence-collapse for the dominant per-layer cost (the routed MoE island, ~0.52
    ms, RUN_CHAIN). The static single-recipe fold requires a static K (asserted).

    BRING-UP: for the very first on-card gate we route the whole MoE through the
    stock submodule (correct + a clean A/B baseline) and assert the routed op is
    available; the op-swap is enabled by ``MMEL_CHAIN_MOE_OP=1`` so B3.3 can measure
    the recipe-count delta WITH and WITHOUT the swap on-card without a rebuild. This
    keeps the first A/B byte-exact-trivially-true while the recipe-count gate proves
    the collapse, per the plan's 'gate before claim' rule."""
    if not route_fold_is_static(sshape.n_routed, sshape.top_k):
        raise MMELShapeError(
            f"[mmel.l6_chain] routed MoE fold is not static for "
            f"n_routed={sshape.n_routed} top_k={sshape.top_k}; the single-recipe op "
            "needs a static K. Eager-island this layer's MoE or fix the config."
        )
    # If the operator opted into the op-swap (MMEL_CHAIN_MOE_OP=1), route the routed
    # block through OUR int4 op (its .so was hard-required + loaded in
    # _ensure_ops_loaded); else use the stock MoE submodule (byte-exact baseline)
    # while still capturing fence-free. The gate is the SAME predicate
    # _ensure_ops_loaded uses to decide whether the routed .so is required, so a
    # baseline run never needs the op present.
    # MMEL_CHAIN_MOE_OP=2 — the FIRST REAL device-resident wire-in: replay the
    # captured cap_routed int4-MoE recipe on the bridge's OWN Synapse stream behind
    # ONE synStreamSynchronize, live-fed (NOT a Python re-issue of the stock
    # submodule). This is the literal csengine 1-completion primitive driven from
    # live decode (mmel_moe_island.run_routed_moe_via_island_replay). Gated, on-card
    # byte-exact-A/B'd vs layer.mlp before it is trusted; a bind/shape error RAISES
    # LOUD and the seam's coherent wrapper keeps the live generation safe.
    if _moe_island_replay_enabled():
        # ENGAGE-MANIFEST (one-shot): the S1 MoE-island device replay is driving the MoE
        # block -- the positive signal the engine engaged (vs a silent stock pass).
        _engage().emit_manifest(
            int(getattr(layer, "layer_idx", 0) or 0),
            {"engine-chain": "MoE-island(S1 device replay, MMEL_CHAIN_MOE_OP=2)",
             "STRICT": "ON" if _engage().strict_enabled() else "off"},
            once_key="engine-chain")
        return _run_routed_moe_via_island_replay(layer, hidden_states)

    # ── THE CORRECT STATIC-TAIL ENGAGEMENT (MMEL_CHAIN_STATIC=1) ────────────────
    # Drive the MoE block (the dominant per-layer dispatch share) device-side via the
    # SAME proven S1 island replay, but ONLY on the covered M=1 decode bucket; prefill
    # and non-M=1 run the stock MoE submodule (delegate, byte-exact). Attention + both
    # norms already ran on the stock path above (context-correct), so this is the
    # audit's static tail with NOTHING context-dynamic device-replayed. LOUD one-shot
    # breadcrumbs so the A/B sees engage-vs-delegate (never a silent stock pass).
    if _chain_static_enabled():
        global _CHAIN_STATIC_ENGAGE_LOGGED, _CHAIN_STATIC_DELEGATE_LOGGED
        global _CHAIN_STATIC_ENTERED_LOGGED
        # ENTRY PROOF (WARNING, one-shot): _run_moe_block reached with MMEL_CHAIN_STATIC=1.
        # Emitted BEFORE the coverage decision so a 0-engage run is diagnosable (see the
        # _CHAIN_STATIC_ENTERED_LOGGED comment at its declaration).
        if not _CHAIN_STATIC_ENTERED_LOGGED:
            _CHAIN_STATIC_ENTERED_LOGGED = True
            try:
                _eshp = tuple(hidden_states.shape)
            except Exception:  # noqa: BLE001
                _eshp = "?"
            logger.warning(
                "[mmel.l6_chain][STATIC-ENTER] _run_moe_block reached with "
                "MMEL_CHAIN_STATIC=1 (owned_all_layers=%s, banz_layer=%r) on layer_idx=%s, "
                "hidden=%s -> resolving coverage next (expect [1,6144]=12288 B at M=1 "
                "decode). If you see this but NO [STATIC-ENGAGE]/[STATIC-DELEGATE], the "
                "coverage predicate raised; if you see NEITHER this NOR them, the static "
                "block was never reached.",
                owned_all_layers_enabled(),
                os.environ.get("MMEL_BANZ_LAYER", "<unset>"),
                getattr(layer, "layer_idx", "?"), _eshp)
        if _subchain_b_covers_bucket(hidden_states):
            # ── #4b: ALL-LAYER DEVICE-RESIDENT TAIL via the LIVE-ROUTING LazyOp ──────
            # (MMEL_OWNED_ALL_LAYERS=1) — the graduation of the Phase-1 1-layer island.
            # On EVERY covered MoE layer, drive the MoE tail through the LIVE-WEIGHT-BOUND,
            # LIVE-ROUTING int4 LazyOp instead of the frozen island. The LazyOp IS the
            # stock submodule call ``layer.mlp(hidden_states)``: in the serve image
            # (HPU_MOE_FUSED_OP=1, PT_HPU_LAZY_MODE=1, M=1) that dispatch lands in
            # MoeWNA16HPUMethod.apply_monolithic -> _fused_moe_routed ->
            # fused_expert_routed.fused_moe_routed, which (1) routes LIVE on-device (the
            # real K-of-256 top-k softmax+topk, NO .item()), (2) index_selects the K
            # ACTIVE experts from the model's ALREADY-RESIDENT stacked int4 weights
            # (layer.w13_qweight.data etc. — ZERO re-upload, ZERO extra HBM), and (3)
            # emits ``torch.ops.hpu_moe_ext.fused_moe`` as ONE LazyOp the bridge captures.
            # Because it is a LazyOp (NOT a raw ctypes synLaunch), it needs NO eager
            # capture-bypass for the MoE tier and collapses the per-layer recipe boundary
            # inside the bridge's own graph (OWNED_FORWARD_PLAN §B.2). This kills BOTH
            # Phase-1 cheats: no duplicate frozen weights (-> all-78-layers FITS, §B.3) and
            # no frozen K=8 routing (-> per-layer-correct, §A fact 2). The island
            # (_run_routed_moe_via_island_replay) is KEPT below as the eager-bypass
            # mechanism proof; it is NOT the all-layer driver.
            #
            # NOTE this is the SAME `layer.mlp(...)` call the MMEL_BANZ_LAYER non-pinned
            # layers already delegate to (below); #4b simply makes it the path on ALL
            # covered MoE layers under one device-resident fence-free chained forward,
            # AND logs a distinct ENGAGE breadcrumb so the A/B can prove it engaged (vs a
            # silent stock fall-through). The fence-collapse is the chained forward's
            # NO-per-op-mark_step composition + (under MMEL_WHOLEFWD) the single
            # per-tier synStreamSynchronize, NOT a different MoE op.
            if owned_all_layers_enabled():
                if not _CHAIN_STATIC_ENGAGE_LOGGED:
                    _CHAIN_STATIC_ENGAGE_LOGGED = True
                    try:
                        _shp = tuple(hidden_states.shape)
                    except Exception:  # noqa: BLE001
                        _shp = "?"
                    # WARNING level (NOT info): the [mmel.l6][ENGAGE]/[mmel.banz][ENGAGE]
                    # install banners that the operator CONFIRMED visible are WARNING; the
                    # serve runs at vLLM's default level which suppressed the prior INFO
                    # breadcrumbs (the 2026-06-26 "0 breadcrumbs" false-null — the path WAS
                    # reached, the INFO line was filtered). Emit the engage proof at WARNING
                    # so it is unmissable at the same level the working banners use.
                    logger.warning(
                        "[mmel.l6_chain][STATIC-ENGAGE][OWNED-ALL-LAYERS] "
                        "MMEL_OWNED_ALL_LAYERS=1: post-norm hidden %s matches the M=1 "
                        "decode bucket -> driving the MoE tail on EVERY MoE layer via the "
                        "LIVE-ROUTING int4 LazyOp (layer.mlp -> fused_moe_routed -> "
                        "torch.ops.hpu_moe_ext.fused_moe; live K-of-256 top-k + "
                        "index_select of the RESIDENT int4 experts -> ZERO re-upload, "
                        "per-layer-correct). The frozen cap_routed island is NOT used "
                        "(no 11.4GB OOM, no frozen-8 routing). Attention + both norms "
                        "stayed on the stock bucketed path.", _shp)
                    _engage().emit_manifest(
                        int(getattr(layer, "layer_idx", 0) or 0),
                        {"engine-chain": ("OWNED-ALL-LAYERS(live-routing LazyOp on every "
                                          "MoE layer; resident weights, no island)"),
                         "STRICT": "ON" if _engage().strict_enabled() else "off"},
                        once_key="engine-chain")
                return layer.mlp(hidden_states)

            # vllm-banz Phase-1 scoping (MMEL_BANZ_LAYER): the cap_routed island replays
            # ONE captured recipe (frozen routing + its OWN resident weights). Engaging it
            # on every MoE layer uploads ~100MB of expert-weight copies PER layer -> HBM
            # OOM, and is wrong (frozen routing != this layer's live experts). When
            # MMEL_BANZ_LAYER is set, ONLY that layer drives the device island; every other
            # MoE layer delegates to the stock dynamic MoE op (byte-exact). Phase 2 (#4b,
            # MMEL_OWNED_ALL_LAYERS=1, handled above) makes the device path
            # per-layer-correct via the live-weight LazyOp instead of the island.
            if _banz_layer_pins_this_layer_off(int(getattr(layer, "layer_idx", -1))):
                return layer.mlp(hidden_states)
            if not _CHAIN_STATIC_ENGAGE_LOGGED:
                _CHAIN_STATIC_ENGAGE_LOGGED = True
                try:
                    _shp = tuple(hidden_states.shape)
                except Exception:  # noqa: BLE001
                    _shp = "?"
                # WARNING level (the serve filters INFO — see the OWNED-ALL-LAYERS note).
                logger.warning(
                    "[mmel.l6_chain][STATIC-ENGAGE] MMEL_CHAIN_STATIC=1: post-norm "
                    "hidden %s matches the M=1 decode bucket -> driving the static "
                    "MoE tail device-side via the island replay (attention + both "
                    "norms stayed on the stock bucketed path; nothing context-dynamic "
                    "is device-replayed). Prefill/non-M=1 delegate to the stock MoE "
                    "(NOT a silent stock fall-through).", _shp)
                # ENGAGE-MANIFEST: the static MoE tail ENGAGED on its covered bucket --
                # the positive signal that the device island drove the MoE (vs 0-WARMED).
                _engage().emit_manifest(
                    int(getattr(layer, "layer_idx", 0) or 0),
                    {"engine-chain": "STATIC-tail(ENGAGED M=1 decode, attn on stock)",
                     "STRICT": "ON" if _engage().strict_enabled() else "off"},
                    once_key="engine-chain")
            return _run_routed_moe_via_island_replay(layer, hidden_states)
        if not _CHAIN_STATIC_DELEGATE_LOGGED:
            _CHAIN_STATIC_DELEGATE_LOGGED = True
            try:
                _shp = tuple(hidden_states.shape)
            except Exception:  # noqa: BLE001
                _shp = "?"
            # WARNING level (the serve filters INFO — see the OWNED-ALL-LAYERS note). A
            # DELEGATE here under MMEL_OWNED_ALL_LAYERS=1 means coverage FAILED (the live
            # hidden is NOT [1,6144]=12288 B) -> the owned arm never ran. That is the
            # concrete signal to grep if the ENGAGE line is absent: DELEGATE present =>
            # reached but not covered (re-check the decode bucket); NEITHER present =>
            # _run_moe_block not reached with MMEL_CHAIN_STATIC=1 (a real wiring bug).
            logger.warning(
                "[mmel.l6_chain][STATIC-DELEGATE] MMEL_CHAIN_STATIC=1 but post-norm "
                "hidden %s is NOT the M=1 decode bucket (prefill / non-M=1) -> the "
                "stock MoE submodule runs (byte-exact); the device MoE tail engages "
                "on the M=1 decode bucket it was captured at.", _shp)
            # STRICT-mode false-null guard: distinguish a LEGITIMATE prefill/non-M=1
            # delegate (the coverage plan resolved, this forward simply isn't the M=1
            # bucket -> intended=False, never raises) from a REGRESSION where the capture
            # plan is UNDECIDABLE (the chain can NEVER engage -> the 0-WARMED false-flat
            # the memory flags -> intended=True -> HPU_ENGINE_STRICT hard-fails). The
            # decidable-but-not-this-bucket case is the normal, byte-exact delegation.
            _plan_undecidable = _captured_plan_undecidable()
            _engage().emit_manifest(
                int(getattr(layer, "layer_idx", 0) or 0),
                {"engine-chain": ("STATIC-tail(REQUESTED but plan UNDECIDABLE -> "
                                  "delegating ALL forwards to stock)" if _plan_undecidable
                                  else "STATIC-tail(delegating non-M=1; engages on M=1)"),
                 "STRICT": "ON" if _engage().strict_enabled() else "off"},
                once_key="engine-chain")
            if _plan_undecidable:
                _engage().engage_fallback(
                    "mmel_l6_chain._run_moe_block:static-plan-undecidable",
                    fast="MMEL_CHAIN_STATIC device MoE tail (island replay)",
                    reason="the captured sub-chain plan is UNDECIDABLE (captures "
                           "missing/malformed) -> the chain can NEVER engage on any "
                           "bucket (the 0-WARMED false-flat)",
                    slow="stock MoE submodule for EVERY forward",
                    intended=True, strict_env="MMEL_CHAIN_STATIC")
        return layer.mlp(hidden_states)

    if not moe_op_swap_enabled():
        # Stock MoE submodule, but invoked inside our fence-free forward (no per-op
        # mark_step) so the bridge still collapses the layer's recipes. The A/B is
        # trivially byte-exact here; B3.3 measures the count, B4.2 the tok/s.
        return layer.mlp(hidden_states)

    # Op-swap path: drive the routed experts through OUR single-recipe int4 op.
    # The op signature (proven, capture_routed_island.py:708):
    #   torch.ops.hpu_moe_ext.fused_moe(x, w13_packed, s13, z13, w2_packed, s2, z2, rws)
    # where rws are the folded per-expert router scalars (top-k, on-device, no
    # .item()). We pull the packed int4 weights + scales/zeros from the stock MoE
    # submodule's experts (already on device, resident) and the shared-expert /
    # gate path from the stock submodule. The exact attribute plumbing is bound in
    # _moe_routed_via_our_op (kept isolated so a model-tree drift fails LOUD there).
    return _moe_routed_via_our_op(
        layer, hidden_states, exec_cfg=exec_cfg, sshape=sshape, torch=torch)


def _moe_routed_via_our_op(layer: Any, hidden_states, *, exec_cfg: Any,
                           sshape: "StaticShape", torch) -> Any:
    """Bind the stock MoE submodule's int4 expert weights to OUR chained routed op.

    Isolated so the (image-specific) attribute plumbing is the ONLY thing that
    drifts on a vLLM/quant bump — and it fails LOUD here rather than mis-binding.
    This is the deepest first-draft surface; it is GATED behind MMEL_CHAIN_MOE_OP
    and on-card byte-exact-A/B'd before it is ever trusted. Until the on-card B4.1
    confirms the bind, the default path (_run_moe_block, op off) is the shipped one.
    """
    raise MMELChainError(
        "[mmel.l6_chain] MMEL_CHAIN_MOE_OP=1 routed-op swap requested but the "
        "int4 expert-weight plumbing for THIS serve image has not been bound + "
        "on-card byte-exact-validated yet (B4.1). Run with MMEL_CHAIN_MOE_OP unset "
        "to use the stock MoE inside the fence-free forward (the B3.3 count gate "
        "does not require the op-swap). Binding lands after the first A/B gate."
    )


# ----------------------------------------------------------------------------- #
#  Live-tensor helpers (LOUD, no host sync).                                      #
# ----------------------------------------------------------------------------- #
_ATTN_MLA_LATENT = "MLA_LATENT"  # mirrors exec_config.ATTN_MLA_LATENT (no import cycle)


def _resolve_context_len(positions, hidden_states) -> int:
    """Best-effort STATIC context length for the bucket WITHOUT a host sync.

    We must NOT call ``.item()`` (the no-host-sync rule). The serve's bucketing
    already fixes the context per bucket; at bring-up we use the static tensor
    extent as the bucket proxy (positions is [num_tokens]; hidden_states is
    [num_tokens, H]). The StaticShape key-pads this to a 128-multiple, so as long
    as the serve's bucket grid is 128-aligned (it is — block_size 128) this is the
    bucket constant, not a live-value read. If a future path needs the true context
    it must come from the forward context's bucketed metadata, never a .item()."""
    try:
        n = int(hidden_states.shape[0])
    except Exception:  # noqa: BLE001
        n = 1
    return max(1, n)


def _assert_layer_submodules(layer: Any) -> None:
    """LOUD guard that the stock decoder submodules we compose are present with the
    expected names. A model-tree drift (vLLM bump) fails here, not as a mis-bind."""
    missing = [nm for nm in (
        "input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        if not hasattr(layer, nm)]
    if missing:
        raise MMELChainError(
            f"[mmel.l6_chain] decoder layer is missing expected submodule(s) "
            f"{missing}; the stock DeepseekV2DecoderLayer composition drifted. "
            "Re-ground mmel_l6_chain against the new deepseek_v2.py before routing."
        )


# ============================================================================= #
#  RAW-CS RECIPE REPLAY SCAFFOLD (the B5/G3 DEEPER structural lever — STAGED).   #
#                                                                               #
#  This is NOT used by the default G1 forward (which composes torch ops and lets #
#  the bridge capture). It is the substrate for B5/G3: replay the EXACT captured #
#  1-fence recipe (cap_*/island.recipe) from LIVE decode tensors via ctypes      #
#  synLaunch on the bridge's OWN device + stream — chaining N islands behind ONE #
#  host fence inside the live step, which is what goes PAST the ~61.5 ms         #
#  fence-only floor (overlap the collective + pipeline the issue, COLLECTIVE_SEQ #
#  / W7). It is built card-free here (the ABI surface + the bind contract) and   #
#  EXERCISED on-card by the orchestrator's B5 step — never by this build agent.  #
#                                                                               #
#  The mechanism is the one the int4 fused-MoE op already uses in-serve (proven, #
#  hpu-moe-ext fused_moe_op.cpp): inside a live habana process you CANNOT acquire #
#  a 2nd device — you reuse device 0 (synDeviceGetCount confirms it) and torch's  #
#  current compute stream (int(htorch.hpu.current_stream().hpu_stream)), and a    #
#  CONTIGUOUS HPU tensor's .data_ptr() IS the device VA synLaunch binds.          #
# ============================================================================= #
# The synLaunchTensorInfo struct, byte-exact vs syn_min.h (the harness ABI).
# Declared lazily (ctypes is stdlib, but we only build the CDLL surface on-card).
_LIBSYNAPSE_CANDIDATES = (
    "/usr/lib/habanalabs/libSynapse.so",
    "libSynapse.so",
)
SYN_FLAGS_TENSOR_NAME = 0x1
_DATA_TENSOR = 0
_HABANA_DIM_MAX = 25


def _make_ctypes_syn_surface():
    """Build the minimal ctypes surface for recipe replay (on-card only).

    Mirrors syn_min.h field-for-field (the C harness ABI). Returns
    ``(syn, LaunchTensorInfo_cls)`` or raises MMELChainError if libSynapse is not
    loadable (a CPU box) — callers must only invoke this on-card."""
    import ctypes

    lib = None
    for cand in _LIBSYNAPSE_CANDIDATES:
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    if lib is None:
        raise MMELChainError(
            "[mmel.l6_chain] libSynapse.so not loadable; the raw-CS replay scaffold "
            "runs ON-CARD only (this is the B5/G3 path, not the default forward)."
        )

    class _TSize(ctypes.Structure):
        pass

    class LaunchTensorInfo(ctypes.Structure):
        # byte-exact vs syn_min.h synLaunchTensorInfo.
        _fields_ = [
            ("tensorName", ctypes.c_char_p),
            ("pTensorAddress", ctypes.c_uint64),
            ("tensorType", ctypes.c_int),
            ("tensorSize", ctypes.c_uint64 * _HABANA_DIM_MAX),
            ("tensorId", ctypes.c_uint64),
        ]

    # signatures (subset; the replay path).
    lib.synDeviceGetCount.restype = ctypes.c_int
    lib.synDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.synRecipeDeSerialize.restype = ctypes.c_int
    lib.synRecipeDeSerialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.synWorkspaceGetSize.restype = ctypes.c_int
    lib.synWorkspaceGetSize.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
    lib.synLaunch.restype = ctypes.c_int
    lib.synLaunch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64,
        ctypes.c_void_p, ctypes.c_uint32,
    ]
    lib.synStreamSynchronize.restype = ctypes.c_int
    lib.synStreamSynchronize.argtypes = [ctypes.c_void_p]
    # synStreamCreateGeneric(synStreamHandle* out, synDeviceId dev, uint32_t flags)
    # — syn_min.h:144, the C harness's proven call (route2_real_island.c:652).
    # Returns a REAL synStreamHandle (opaque pointer), unlike torch's .hpu_stream
    # which is a stream INDEX synLaunch rejects.
    lib.synStreamCreateGeneric.restype = ctypes.c_int
    lib.synStreamCreateGeneric.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint32,
    ]
    return lib, LaunchTensorInfo


# vllm-banz: a REAL Synapse stream handle (synStreamCreateGeneric, a c_void_p),
# lazily created + cached. Used instead of torch's current_stream().hpu_stream,
# which is a stream INDEX (0, 4, ...) that synLaunch rejects under the eager
# capture-bypass. None until the first on-card replay creates it.
_BANZ_STREAM = None


def reuse_bridge_device_and_stream():
    """Return ``(device_id, stream_handle_int)`` of the bridge's ALREADY-acquired
    device + torch's current HPU compute stream — the only correct context for an
    in-serve synLaunch (you must NOT acquire a second device). On-card only.

    Proven mechanism (hpu-moe-ext fused_moe_op.cpp / fused_moe_op.py): the single
    acquired Gaudi-2 is device id 0 (synDeviceGetCount confirms count>=1); the
    stream is ``int(habana_frameworks.torch.hpu.current_stream().hpu_stream)`` so
    our launches order against torch's ops automatically."""
    import ctypes

    syn, _ = _make_ctypes_syn_surface()
    cnt = ctypes.c_uint32(0)
    st = syn.synDeviceGetCount(ctypes.byref(cnt))
    if st != 0 or cnt.value < 1:
        raise MMELChainError(
            f"[mmel.l6_chain] synDeviceGetCount -> status={st} count={cnt.value}; "
            "the HPU is not acquired (import habana_frameworks.torch first / run "
            "inside the live serve)."
        )
    device_id = 0  # single-device serve: the acquired Gaudi-2 is id 0.
    # Create + cache a REAL Synapse stream (the proven C-harness path,
    # route2_real_island.c:652). torch's current_stream().hpu_stream is a stream
    # INDEX (0, 4, ...), NOT the opaque synStreamHandle synLaunch needs — under the
    # eager capture-bypass that index is what got bound and synLaunch rejected it
    # (status=1, every other arg valid). synStreamCreateGeneric returns the real
    # handle; the device-wide mark_step+synchronize around each replay keeps our
    # stream ordered after torch's ops (correctness holds; cost is lost pipelining).
    global _BANZ_STREAM
    if _BANZ_STREAM is None:
        h = ctypes.c_void_p()
        st = syn.synStreamCreateGeneric(
            ctypes.byref(h), ctypes.c_uint32(device_id), ctypes.c_uint32(0))
        if st != 0 or not h.value:
            raise MMELChainError(
                f"[mmel.l6_chain] synStreamCreateGeneric(dev={device_id}) -> "
                f"status={st} handle={h.value}; cannot create the replay stream.")
        _BANZ_STREAM = h
    return device_id, int(_BANZ_STREAM.value)


@dataclass
class RawCsIslandReplay:
    """Staged B5/G3 replay of ONE captured island recipe from LIVE tensors.

    Holds a deserialized recipe + its workspace, and replays it by binding the
    recipe's named launch tensors to LIVE contiguous HPU tensors' ``.data_ptr()``
    on the bridge's stream — NO host fence between chained islands (the
    fence-collapse the C harness proved, now driven live). This is the substrate
    the orchestrator wires + gates at B5; G1's default forward does NOT call it.

    Built so the on-card B5 step can: (1) deserialize cap_*/island.recipe once at
    warmup, (2) per step bind {recipe_tensor_name -> live tensor data_ptr}, (3)
    synLaunch all islands on the bridge stream, (4) ONE synStreamSynchronize.
    The byte-exact A/B harness (seam) is the arbiter; static-shape is enforced by
    the same StaticShape guards as the torch path."""

    recipe_path: str
    _recipe: Any = None          # ctypes c_void_p (the deserialized handle)
    _ws_size: int = 0
    _syn: Any = None
    _lti_cls: Any = None

    def deserialize(self) -> "RawCsIslandReplay":
        """Deserialize the recipe + query its workspace size (on-card, once)."""
        import ctypes
        import os as _os

        if not _os.path.isfile(self.recipe_path):
            raise MMELChainError(
                f"[mmel.l6_chain] recipe not found: {self.recipe_path!r} "
                "(re-capture this island on-card per B1.1)."
            )
        self._syn, self._lti_cls = _make_ctypes_syn_surface()
        h = ctypes.c_void_p()
        st = self._syn.synRecipeDeSerialize(ctypes.byref(h), self.recipe_path.encode())
        if st != 0:
            raise MMELChainError(
                f"[mmel.l6_chain] synRecipeDeSerialize({self.recipe_path}) -> "
                f"status={st}. The recipe's TPC kernel glob must be present "
                "(run in-container, the op .so loaded)."
            )
        self._recipe = h
        ws = ctypes.c_uint64(0)
        st = self._syn.synWorkspaceGetSize(ctypes.byref(ws), h)
        if st != 0:
            raise MMELChainError(
                f"[mmel.l6_chain] synWorkspaceGetSize -> status={st}.")
        self._ws_size = int(ws.value)
        return self

    def build_launch_array(self, name_to_live_tensor: dict):
        """Build the synLaunchTensorInfo[] binding recipe tensor names to LIVE
        contiguous HPU tensors. Each tensor MUST be contiguous (its data_ptr is the
        row-major device VA the recipe expects); LOUD if not. Returns a ctypes array
        kept alive by the caller across the synLaunch."""
        if self._lti_cls is None:
            raise MMELChainError("[mmel.l6_chain] call deserialize() before build.")
        # CRITICAL (mem_handle bug): a torch HPU tensor's .data_ptr() is a mem_HANDLE
        # (id<<offset|offset), NOT the device VA. The bridge resolves handle->VA and patches
        # synLaunchTensorInfo.pTensorAddress at its own launch; a RAW synLaunch that binds the
        # handle => device writes/reads a BOGUS address => islands compute on garbage (real
        # magnitude, wrong values). `_data_ptr` is the SAME resolver the bridge uses.
        try:
            from habana_frameworks.torch.utils.experimental import _data_ptr as _real_dp
        except Exception:  # noqa: BLE001
            _real_dp = None
        n = len(name_to_live_tensor)
        arr = (self._lti_cls * n)()
        # keep the name bytes alive (c_char_p must reference a live buffer).
        self._name_keepalive = []
        for i, (nm, t) in enumerate(name_to_live_tensor.items()):
            if not t.is_contiguous():
                raise MMELChainError(
                    f"[mmel.l6_chain] live tensor for recipe input {nm!r} is not "
                    "contiguous; .data_ptr() would mis-bind. .contiguous() it."
                )
            nb = nm.encode()
            self._name_keepalive.append(nb)
            arr[i].tensorName = nb
            # PRE-BOUND handle (the owned wire's _LiveHandle): it ALREADY carries the TRUE
            # device VA (resolved via THIS SAME experimental _data_ptr at _live_buffer, the
            # proven island mechanism). Use it DIRECTLY -- re-resolving _real_dp() on a
            # non-torch handle returns 0 (the synLaunch status=30 null-VA bug). A real torch
            # tensor (the island path) still goes through _real_dp (a raw .data_ptr() is a
            # mem-handle, not the device VA).
            if getattr(t, "_mmel_prebound", False):
                arr[i].pTensorAddress = int(t.data_ptr())
            else:
                arr[i].pTensorAddress = int(
                    _real_dp(t) if _real_dp is not None else t.data_ptr())
            arr[i].tensorType = _DATA_TENSOR
            arr[i].tensorId = 0
        return arr

    def launch(self, stream_handle: int, launch_arr, ws_ptr: int = 0) -> None:
        """Enqueue this island's recipe on ``stream_handle`` (NO sync — the chain
        caller does the single final synStreamSynchronize). ws_ptr is a device
        workspace buffer of at least ``self._ws_size`` bytes (0 if size 0)."""
        import ctypes

        if self._recipe is None:
            raise MMELChainError("[mmel.l6_chain] call deserialize() before launch.")
        st = self._syn.synLaunch(
            ctypes.c_void_p(stream_handle), launch_arr, len(launch_arr),
            ctypes.c_uint64(ws_ptr), self._recipe, SYN_FLAGS_TENSOR_NAME)
        if st != 0:
            # vllm-banz bisect: synLaunch status=1 (synInvalidArgument) does not say
            # WHICH arg is bad. Dump every bound arg so the traceback pins it: a null
            # (0x0) tensor/ws address = an uncommitted VA; ws_ptr=0 with ws_size>0 =
            # the workspace wasn't passed; stream=0x0 = the bridge stream is wrong.
            try:
                _t = [
                    (bytes(launch_arr[i].tensorName).decode("ascii", "replace")
                     if launch_arr[i].tensorName else "<null>",
                     int(launch_arr[i].pTensorAddress))
                    for i in range(len(launch_arr))
                ]
            except Exception as _e:  # noqa: BLE001
                _t = [(f"<dump-failed:{_e!r}>", 0)]
            _nulls = [nm for nm, a in _t if a == 0]
            raise MMELChainError(
                f"[mmel.l6_chain] synLaunch({self.recipe_path}) -> status={st}. "
                f"DIAG stream=0x{stream_handle:x} ws_ptr=0x{ws_ptr:x} "
                f"ws_size={self._ws_size} ntensors={len(launch_arr)} "
                f"null_addr={_nulls} tensors=["
                + ", ".join(f"{nm}@0x{a:x}" for nm, a in _t) + "]")


# ============================================================================= #
#  THE OWNED-FORWARD HpuBackend WIRE (MMEL_BACKEND=hpu).                         #
#                                                                               #
#  Reuses, unchanged, the CARD-FREE-VALIDATED orchestration:                     #
#    * MoEDecodeSpec.from_exec_config  (owned_engine.py:135)  — the spec.        #
#    * HpuBackend(MmelBackend)         (hpu_backend.py:157)   — capture/commit/  #
#      enqueue/synchronize on the PROVEN synLaunch surface (RawCsIslandReplay,   #
#      reuse_bridge_device_and_stream above). H1..H6 + real_recipe_test PASS.    #
#  Adds ONLY the live bind: map each committed recipe's external launch-tensor    #
#  (bind-plan, hpu_backend.py:476) to a LIVE resident VA, enqueue all N layers,   #
#  ONE synStreamSynchronize per forward. Card-validated-only lines are banner-    #
#  marked; a wrong bind RAISES (LOUD) so a bring-up gate catches it (never        #
#  corrupts output — the seam's coherent A/B wrapper keeps the live serve safe).  #
#  The card-free correctness arbiter is owned_wire_oracle.py (W1..W5) + G-byte.   #
# ============================================================================= #
_OWNED_CHAIN = None            # per-worker singleton _OwnedHpuChain (built at warmup)
_OWNED_CHAIN_KEY = None        # (id(exec_cfg), family) — rebuild if the model changed


def _owned_import_engine():
    """Lazily import owned_engine + hpu_backend (card-only surface). LOUD if absent."""
    # hpu_backend._import_chain (hpu_backend.py:239) does a BARE `import mmel_l6_chain`
    # to reuse RawCsIslandReplay + reuse_bridge_device_and_stream + the ctypes synLaunch
    # surface. In-serve THIS module is loaded as `vllm_gaudi.v1.worker.mmel_l6_chain`, so
    # the bare name is not importable -> ModuleNotFoundError at hbe.capture/commit. Alias
    # the already-loaded module under the bare name so the reuse import resolves (idempotent;
    # setdefault never clobbers a real standalone mmel_l6_chain on a card-free box).
    sys.modules.setdefault("mmel_l6_chain", sys.modules[__name__])
    if _OWNED_ENGINE_DIR not in sys.path:
        sys.path.insert(0, _OWNED_ENGINE_DIR)
    try:
        import hpu_backend as _hpu          # noqa: E402 — on-card only
        import owned_engine as _eng         # noqa: E402
    except Exception as e:  # noqa: BLE001
        raise MMELChainError(
            f"[mmel.l6_chain] MMEL_BACKEND=hpu but the owned engine is not importable "
            f"from {_OWNED_ENGINE_DIR!r} ({type(e).__name__}: {e}). Mount "
            f"$HOME/AI/mmel/owned_engine into the container (ONCARD_RUNBOOK §2) "
            f"or unset MMEL_BACKEND.") from e
    return _eng, _hpu


def _owned_resolve_va(t) -> int:
    """The TRUE device VA of a live HPU tensor (a torch .data_ptr() is a mem-HANDLE,
    not the VA — build_launch_array note, :2044). Must be .contiguous() first."""
    try:
        from habana_frameworks.torch.utils.experimental import _data_ptr as _dp
    except Exception:  # noqa: BLE001
        _dp = None
    return int(_dp(t) if _dp is not None else t.data_ptr())


def _run_owned_forward_via_hpu_backend(layer, positions, hidden_states, residual, *,
                                       exec_cfg, llama_4_scaling):
    """Entry point for MMEL_BACKEND=hpu. Returns (hidden, residual) on the covered M=1
    decode bucket, or None to DELEGATE (prefill / non-M=1 / warmup -> the G1 spine runs
    byte-exact). Builds the singleton chain at warmup; drives it per layer at decode."""
    global _OWNED_CHAIN, _OWNED_CHAIN_KEY
    # coverage: identical predicate to the static tail (mmel_l6_chain.py:1644) so the
    # engage/delegate split is consistent (prefill/non-M=1 delegate, never raise).
    if not _subchain_b_covers_bucket(hidden_states):
        return None
    # Stage-A one-layer gate: MMEL_BANZ_LAYER=<idx> pins the owned wire to EXACTLY that
    # layer (every other MoE layer DELEGATES to the byte-exact stock spine). This is the
    # safe first G-byte gate (one layer owned; unset => all covered layers). Reuses the
    # existing pin helper (mmel_l6_chain.py:436).
    if _banz_layer_pins_this_layer_off(int(getattr(layer, "layer_idx", -1))):
        return None
    key = (id(exec_cfg), getattr(exec_cfg, "family", "?"))
    if _OWNED_CHAIN is None or _OWNED_CHAIN_KEY != key:
        _OWNED_CHAIN = _OwnedHpuChain(exec_cfg)          # WARMUP build (once per model)
        _OWNED_CHAIN_KEY = key
    return _OWNED_CHAIN.run_layer(layer, positions, hidden_states, residual,
                                  llama_4_scaling=llama_4_scaling)


# ----------------------------------------------------------------------------- #
#  TAG -> RECIPE-POSITIONAL-NAME BRIDGE (§5.B — the make-or-break; PURE/card-free).#
#                                                                               #
#  Captured GC recipes name their persistent launch tensors POSITIONALLY ("0".. #
#  "57", cap_routed/manifest.txt), but HpuBackend.enqueue binds them by NAME     #
#  (build_launch_array({name: handle})). So the engine tags recorded in          #
#  _record_layer_structure (exp_c13, exp_s13, ...) must be translated to the      #
#  recipe's real positional names before enqueue. We derive the map from the      #
#  recipe's manifest.txt by BYTE SIZE + role + ORDER — the SAME byte-size+role    #
#  disambiguation mmel_chain_layer.mla_live_ctx_roles uses for the MLA island.    #
# ----------------------------------------------------------------------------- #
def _parse_recipe_manifest(path):
    """Parse a captured recipe's ``manifest.txt`` -> ``(header_dims, rows)``. PURE,
    card-free. ``header_dims`` = every ``k=v`` int on the leading ``# ...`` lines
    (H/I/GS/top_k/n_routed/K/...). ``rows`` = ``[(name, nbytes, role, note)]`` for each
    tensor line — ``name`` IS the recipe's own launch-tensor name; ``role`` in
    {IN_FILE, OUT, LIVE_KV, ...}. Never raises on a well-formed manifest."""
    import re
    header, rows = {}, []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                for k, v in re.findall(r"([A-Za-z_]\w*)=(-?\d+)", line):
                    header.setdefault(k, int(v))
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            try:
                nbytes = int(parts[1])
            except ValueError:
                continue
            rows.append((parts[0], nbytes, parts[2], parts[3] if len(parts) > 3 else ""))
    return header, rows


def _manifest_tag_positions(path):
    """{engine_tag: [recipe_positional_name, ...]} from a captured recipe's manifest.txt.

    Classifies each launch tensor by BYTE SIZE (derived from the header dims H/I/GS) +
    role + manifest ORDER. The six int4 expert classes have DISTINCT byte sizes for GLM
    dims, so exp_c13/exp_s13/exp_z13/exp_c2/exp_s2/exp_z2 are unambiguous (this is the
    C14-asserted core). Same-size spine inputs (attn_out/residual/hidden, all H*2 bf16)
    are split by manifest ORDER into layer_in0/layer_in1/layer_out. Router/shared classes
    (Wgate=E*H*2, sh_gu/sh_d) are labelled when the size is resolvable. PURE; card-free.
    LOUD only if the header lacks H/I/GS (cannot size the int4 classes)."""
    header, rows = _parse_recipe_manifest(path)
    H, I, GS = header.get("H"), header.get("I"), header.get("GS")
    if not (H and I and GS):
        raise MMELChainError(
            f"[mmel.l6_chain] recipe manifest {path!r} header lacks H/I/GS (got "
            f"{header}); cannot size the int4 expert classes for the tag->name bridge. "
            f"Re-capture the recipe with the dims header (capture_routed_island.py:22).")
    twoI8, H8, HG, IG = (2 * I) // 8, H // 8, H // GS, I // GS
    E = header.get("n_routed") or header.get("n_routed_experts")
    size_tag = {
        H * twoI8 * 4:    "exp_c13",   # w13 packed [H, 2I//8]   int32
        HG * (2 * I) * 2: "exp_s13",   # w13 scale  [H//GS, 2I]  bf16
        HG * twoI8 * 4:   "exp_z13",   # w13 zp     [H//GS, 2I//8] int32
        I * H8 * 4:       "exp_c2",    # w2 packed  [I, H//8]    int32
        IG * H * 2:       "exp_s2",    # w2 scale   [I//GS, H]   bf16
        IG * H8 * 4:      "exp_z2",    # w2 zp      [I//GS, H//8] int32
    }
    if E:
        # router weight [E, H] bf16. setdefault so the int4 EXPERT classes always win a
        # size tie (in the shrunk cap_routed capture E=8 makes E*H*2 == exp_z13's H//GS*
        # 2I//8*4 = 98304; the real tail has E=256 -> 3145728, no tie).
        size_tag.setdefault(E * H * 2, "Wgate")
    # shared expert (n_shared=1): bf16 DeepseekV2MLP, Ish == moe_inter == I. Distinct
    # sizes from every int4/router class above, so unambiguous when present.
    size_tag.setdefault(2 * I * H * 2, "sh_gu")   # [2*Ish, H] bf16
    size_tag.setdefault(H * I * 2, "sh_d")        # [H, Ish]  bf16
    out = {}
    spine_in = 0
    for name, nbytes, role, _note in rows:
        if role == "OUT":
            out.setdefault("layer_out", []).append(name)
            continue
        tag = size_tag.get(nbytes)
        if tag is None and nbytes == H * 2:
            # a bf16 [1,H] spine input (attn_out / residual / hidden), by manifest ORDER.
            tag = f"layer_in{spine_in}"
            spine_in += 1
        if tag is not None:
            out.setdefault(tag, []).append(name)
    return out


def _tensor_width(t):
    """Element byte-width of a recipe launch tensor (int32=4, bf16=2), from nbytes/nelem
    (dtype-agnostic; needs no synapse dtype enum). PURE; card-free."""
    return int(t["nbytes"]) // max(int(t.get("nelem", 1)), 1)


def _is_moe_tail_signature(tensors, min_packed=8 * 1024 * 1024):
    """SELECT-BY-SIGNATURE predicate (PURE; card-free). True iff a recipe's launch-tensor
    list carries the stacked int4 MoE-TAIL signature: the SINGLE LARGEST launch tensor is a
    LARGE int32 (width 4 = the stacked packed int4 expert weight exp_c13/exp_c2), with >=2
    such large int32 tensors. Distinguishes the MoE tail from: a DENSE tail (largest is a
    bf16 GEMM weight, width 2), and attention/embed/head (no large int32). Sharding-
    agnostic — the packed weight stays the largest by far. This is the discriminator that
    replaces door-A 'newest-after' (which fails: the M=1 tail is a cache HIT, not new)."""
    if not tensors:
        return False
    big4 = [t for t in tensors
            if int(t["nbytes"]) >= min_packed and _tensor_width(t) == 4]
    largest = max(tensors, key=lambda t: int(t["nbytes"]))
    return (_tensor_width(largest) == 4 and int(largest["nbytes"]) >= min_packed
            and len(big4) >= 2)


def _pick_m1_decode_tail(cand_tensor_lists):
    """Pick the M=1 DECODE tail among int4-tail candidates (the SAME MoE tail at DIFFERENT
    shape buckets): the STRUCTURALLY LEANEST recipe — FEWEST launch tensors, tiebreak
    smallest total nbytes. The decode tail (18 tensors) has none of the [S,6144] activation
    intermediates the prompt/seq buckets (34/35) carry. LOUD (None) ONLY if the top-2 tie
    EXACTLY on (n_tensors, total_bytes). PURE; card-free; no size heuristics."""
    if not cand_tensor_lists:
        return None, "no candidates"
    scored = sorted((len(tl), sum(int(t["nbytes"]) for t in tl), i)
                    for i, tl in enumerate(cand_tensor_lists))
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        return None, f"top-2 tie on (n_tensors={scored[0][0]}, total_bytes={scored[0][1]})"
    return scored[0][2], f"fewest tensors ({scored[0][0]}), total {scored[0][1]} B"


def _pair_tags_to_recipe(lbuckets, rbuckets, degen_order):
    """PURE (card-free) pairing of live engine tags -> recipe launch-tensor names, per
    (nbytes,width,is_input) bucket. Within a bucket: sort live tags by ``degen_order``
    priority, sort recipe names by GC-id order (already so in rbuckets), zip. Leftover live
    tags (more live than recipe tensors) are BAKED — the recipe CONST-FOLDED them (a
    replicated bf16 weight GC baked; a sharded weight baked from rank-0), so they are OMITTED
    (skipped, not bound). Leftover recipe tensors (more recipe than live) fall through to
    their captured value (static control/routing metadata we have no live tensor for).
    Returns ({tag: [recipe_name]}, ordered_degenerate_list). This is how the wire tolerates
    GC baking: only the live externals that MATCH a recipe launch tensor get bound."""
    out, ordered = {}, []
    for key, tags in lbuckets.items():
        rnames = rbuckets.get(key, [])
        tags_sorted = sorted(
            tags, key=lambda t: (degen_order.index(t) if t in degen_order else 99, str(t)))
        k = min(len(tags_sorted), len(rnames))
        for i in range(k):
            out[tags_sorted[i]] = [rnames[i]]
        # tags_sorted[k:] => BAKED (no recipe launch tensor); rnames[k:] => fall through.
        if len(rnames) > 1 and k >= 1:
            ordered.append((key, list(zip(tags_sorted[:k], rnames[:k]))))
    return out, ordered


class _OwnedHpuChain:
    """Per-worker holder of the owned-forward chain: the spec, the HpuBackend, the
    per-layer committed recipes, and the per-forward sync bookkeeping. Constructed at
    warmup; driven per layer at decode. LOUD on every precondition."""

    def __init__(self, exec_cfg):
        eng, hpu = _owned_import_engine()
        self._eng, self._hpu = eng, hpu
        # (1) spec from the REAL ExecConfig (full dims; owned_engine.py:135).
        self.spec = eng.MoEDecodeSpec.from_exec_config(exec_cfg)
        self.n_layers = int(exec_cfg.num_hidden_layers)
        self.nsteps = int(os.environ.get("MMEL_OWNED_NSTEPS", "8"))   # nominal (KV is live)
        # (2) the on-card HpuBackend (route-2 default; SOB gated). recipe_map None =>
        #     the <MMEL_CAP_DIR>/cap_<name>/island.recipe convention (hpu_backend.py:444).
        rmap = self._recipe_map()
        self.hbe = hpu.HpuBackend(recipe_map=rmap)       # MmelBackend, H1..H6 PASS
        if not self.hbe.on_card:
            raise MMELChainError(
                "[mmel.l6_chain] MMEL_BACKEND=hpu but HpuBackend reports off-card "
                "(no libSynapse / /dev/accel). Run inside the serve container "
                "(ONCARD_RUNBOOK §0); the wire is CARD-ONLY (the card-free proof is "
                "owned_wire_oracle.py + hpu_backend.py --validate).")
        self.one_sync = _owned_one_sync()
        self._recipes = {}          # layer_idx -> committed HpuRecipe (captured on 1st hit)
        self._live_by_tag = {}      # layer_idx -> {tag: live-torch-tensor} for rebind
        self._tag_to_name = {}      # layer_idx -> {engine_tag: [recipe_positional_name]}
        self._captured_tail = set() # layer indices whose tail recipe was captured this run
        self._degen_warned = False  # one-shot LOUD for order-resolved degenerate buckets
        self._baked_warned = False  # one-shot LOUD for const-folded (baked) skipped tags
        self._recipe_tensors = {}   # layer_idx -> [{name,is_input,nbytes,...}] (full set)
        self._resident_by_name = {} # layer_idx -> {recipe_name: resident zero Buffer}
        self._launchset_logged = False  # one-shot LOUD for the full launch-set breakdown
        self._last_idx = None       # forward-boundary detection
        self._engaged_logged = False

    def _recipe_map(self):
        """Explicit MMEL_OWNED_RECIPE_MAP (JSON) if set, else None (convention path)."""
        raw = os.environ.get("MMEL_OWNED_RECIPE_MAP", "").strip()
        if not raw:
            return None
        import json
        m = json.loads(raw)
        if not isinstance(m, dict):
            raise MMELChainError(
                "[mmel.l6_chain] MMEL_OWNED_RECIPE_MAP must be a JSON object "
                "{recipe_name: path}; got %r." % (type(m).__name__,))
        return m

    def _recipe_name(self, idx):
        """The per-layer recipe name. Stage A (static tail, MMEL_OWNED_ATTN=stock) drives
        the [post_norm -> MoE] tail => ``tail<idx>``; the full-layer attention stage
        (MMEL_OWNED_ATTN=owned) drives ``layer<idx>``. HpuBackend resolves the path via
        recipe_map or the ``cap_<name>/island.recipe`` convention (hpu_backend.py:444)."""
        return (f"tail{idx}" if self._attn_is_stock() else f"layer{idx}")

    # ---- warmup: capture + commit ONE per-layer recipe (kept SEPARATE) -------------
    def _ensure_recipe(self, layer, live):
        """Capture+commit this layer's per-layer GC recipe on first hit, using the FULL
        live map (hidden/residual/attn_out shapes come from the live decode tensors, not a
        weights-only re-collect). Stage A captures the [post_norm -> MoE] tail
        (_maybe_capture_tail); the recipe is then deserialized + the bind-plan computed
        (hpu_backend.py:458). Kept SEPARATE per layer (chain-capture, NOT a mega-recipe;
        the mega-recipe compile-blowup is ONCARD_RUNBOOK §6)."""
        idx = int(layer.layer_idx)
        if idx in self._recipes:
            return self._recipes[idx]
        name = self._recipe_name(idx)
        # Stage-A: capture the static tail on-card (once) if its recipe is not on disk.
        self._maybe_capture_tail(layer, idx, live)

        # build() records the op STRUCTURE with the engine tags; HpuBackend IGNORES the
        # fns (deserializes the real GC recipe) and uses the tags for the bind-plan
        # (hpu_backend.py:432-441). Structure mirrors owned_engine _layer_build.
        def build(ctx):
            self._record_layer_structure(ctx, idx, live)

        rec = self.hbe.commit(self.hbe.capture(name, build))
        # Read the recipe's COMPLETE launch-tensor set ONCE (synRecipeDeSerialize readback).
        # synLaunch REQUIRES every persistent launch tensor bound (a partial set => status=30
        # / failed section validation), exactly as the proven island binds ALL of its
        # tensors (static-from-.bin + outputs + live). We do NOT have .bin, so:
        #   * live-OVERRIDE the externals we MATCH by (nbytes,width,is_input) -> live VA
        #     (the 6 SHARDED int4 experts + attn_out/residual + the output `hidden`);
        #   * RESIDENT zero buffer for EVERY other recipe tensor (outputs/scratch: values
        #     don't matter; baked/control persistent inputs: we lack the .bin -> zeros ->
        #     G-byte adjudicates whether any control value actually matters).
        recipe_tensors = self._read_recipe_tensors(
            self.hbe._resolve_recipe_path(name))
        self._recipe_tensors[idx] = recipe_tensors
        # the tag -> recipe-positional-name bridge (§5.B): map engine tags -> recipe names
        # (matched by live-tensor (nbytes,width,is_input) -> recipe launch tensor).
        self._tag_to_name[idx] = self._build_tag_to_recipe_name(recipe_tensors, live)
        matched = {n for names in self._tag_to_name[idx].values()
                   if len(names) == 1 for n in names}
        import numpy as _np
        resident = {}
        for t in recipe_tensors:
            nm = t["name"]
            if nm in matched:
                continue                         # live-overridden below (per step)
            va = self.hbe._torch_scratch_zeros(int(t["nbytes"]))   # resident, stable VA
            resident[nm] = self.hbe._register(va, (int(t["nbytes"]),), _np.uint8, nm)
        self._resident_by_name[idx] = resident
        self._recipes[idx] = rec
        self._live_by_tag[idx] = live
        self._log_launchset(idx, recipe_tensors, matched, resident)
        return rec

    def _log_launchset(self, idx, recipe_tensors, matched, resident):
        """One-shot LOUD: the FULL launch-set breakdown (live-override vs resident-
        persistent), so the operator can confirm every recipe tensor is provided and see
        which control/output tensors fell back to zero buffers."""
        if self._launchset_logged:
            return
        self._launchset_logged = True
        res_sizes = sorted(int(self._req_nbytes(t)) for t in recipe_tensors
                           if t["name"] in resident and int(t.get("is_input", 1)))
        logger.warning(
            "[mmel.l6_chain][OWNED-HPU][LAUNCHSET] layer%d: recipe needs %d launch tensors "
            "= %d LIVE-override (matched: sharded int4 experts + attn_out/residual + "
            "output) + %d RESIDENT (outputs/scratch zero'd; %d persistent-INPUT control "
            "tensors zero'd, sizes=%s -> G-byte adjudicates if any value matters).",
            idx, len(recipe_tensors), len(matched), len(resident),
            len(res_sizes), res_sizes)

    @staticmethod
    def _req_nbytes(t):
        return int(t["nbytes"])

    # engine tags that can be metadata-DEGENERATE (same nbytes,width,is_input — the recipe
    # reader gives NO ndim, ndim=0 unreliable) and are separable only by graph ORDER, paired
    # against the recipe's same-key launch tensors in GC-id order. Two families collide at
    # the TP-sharded sizes: the [1,H] bf16 spine (attn_out=post_norm in0, residual=in1,
    # w_post=gamma, all 12288 B is_input=1), and two weight pairs (exp_s2==sh_gu at 6291456
    # bf16; Wgate==sh_d at 3145728 bf16). The weight-pair ORDER here is a GUESS validated by
    # the G-byte gate (a wrong guess GARBLES -> flip the pair; a wrong bind crashes a gate,
    # never corrupts a live gen). The wire LOUD-lists every order-resolved pairing.
    _DEGEN_ORDER = ("attn_out", "residual", "w_post", "exp_s2", "sh_gu",
                    "Wgate", "sh_d", "hidden")
    # the static-tail RECIPE OUTPUT tags (is_input=0 in the recipe); everything else is an
    # input (is_input=1). Lets the live<->recipe match key on is_input (separates the
    # 12288 B OUT `hidden` from the 12288 B inputs attn_out/residual/w_post).
    _OUTPUT_TAGS = ("hidden",)

    def _cap_module(self):
        """Load capture_routed_island.py ONCE (its proven recipe-cache + recipe-tensor
        readback helpers). LOUD if absent (Stage-A needs it on-card)."""
        m = getattr(self, "_cap_mod", None)
        if m is not None:
            return m
        import importlib.util
        cap_py = os.path.join(self.hbe._cap_dir, "capture_routed_island.py")
        if not os.path.isfile(cap_py):
            raise MMELChainError(
                f"[mmel.l6_chain] Stage-A needs capture_routed_island.py (recipe-cache "
                f"select + recipe-tensor readback) at {cap_py!r}; mount the mmel tree or "
                f"set MMEL_CAP_DIR.")
        spec = importlib.util.spec_from_file_location("mmel_capture_routed", cap_py)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)  # type: ignore[union-attr]
        self._cap_mod = m
        return m

    def _read_recipe_tensors(self, path):
        """The recipe's launch-tensor list [{name,id,is_input,dtype,ndim,nelem,nbytes}]
        via capture_routed_island._read_recipe_tensors (synRecipeDeSerialize readback)."""
        reader = self._cap_module().__dict__.get("_read_recipe_tensors")
        if reader is None:
            raise MMELChainError(
                "[mmel.l6_chain] capture_routed_island._read_recipe_tensors absent; "
                "cannot read the recipe's launch tensors for the tag->name bridge.")
        return reader(path)

    def _live_is_input(self, tag):
        """0 iff this engine tag is the recipe's OUTPUT launch tensor (the static-tail
        `hidden`), else 1 (an input). Lets the live<->recipe match key on is_input."""
        return 0 if tag in self._OUTPUT_TAGS else 1

    def _build_tag_to_recipe_name(self, tensors, live):
        """{engine_tag: [recipe_positional_name]} by matching each LIVE tensor to a recipe
        launch tensor by (nbytes, element_width, is_input) — SHARDING-CORRECT (uses the
        actual runtime sizes, never a card-free computed/unsharded size). NO ndim: the
        recipe reader reports ndim=0 unreliably, so ndim is NOT in the key. (nbytes,width,
        is_input) resolves most classes; the residual metadata-degenerate buckets — the
        [1,H] bf16 spine (attn_out/residual/w_post) and the TP-sharded weight collisions
        (exp_s2==sh_gu at 6291456 bf16; Wgate==sh_d at 3145728 bf16) — are paired by graph
        ORDER via _DEGEN_ORDER (LOUD-listed; G-byte-validated). ``tensors`` is the recipe's
        pre-read launch-tensor list. Only the MATCHED live externals are returned; every
        unmatched recipe tensor is bound to a resident buffer by _ensure_recipe (the FULL
        launch set)."""
        import torch  # noqa: F401 — card only
        # recipe buckets by (nbytes, width, is_input), preserving GC-id order (read order).
        rbuckets = {}
        for t in sorted(tensors, key=lambda x: int(x.get("id", 0))):
            nb = int(t["nbytes"]); ne = max(int(t.get("nelem", 1)), 1)
            key = (nb, nb // ne, int(t.get("is_input", 1)))
            rbuckets.setdefault(key, []).append(t["name"])
        # live tags bucketed by the SAME key (metadata only; no host copy of weights).
        lbuckets = {}
        for tag, val in live.items():
            if str(tag).startswith("__") or not hasattr(val, "element_size"):
                continue
            try:
                vt = val if val.is_contiguous() else val.contiguous()
                nb = int(vt.numel()) * int(vt.element_size())
                key = (nb, int(vt.element_size()), self._live_is_input(tag))
            except Exception:  # noqa: BLE001 — unresolvable live tensor: skip (LOUD later)
                continue
            lbuckets.setdefault(key, []).append(tag)
        # Pair per bucket (pure); leftover live tags with NO recipe launch tensor are BAKED
        # (GC const-folded them) -> OMITTED from `out` -> _rebind SKIPS them (partial bind).
        out, ordered = _pair_tags_to_recipe(lbuckets, rbuckets, self._DEGEN_ORDER)
        if ordered and not self._degen_warned:
            self._degen_warned = True
            logger.warning(
                "[mmel.l6_chain][OWNED-HPU][BIND] %d size-degenerate launch-tensor "
                "bucket(s) resolved by ORDER (recipe GC-id <-> _DEGEN_ORDER; recipe ndim "
                "unavailable): %s. If G-byte GARBLES, an order guess is wrong -> flip the "
                "offending pair in _DEGEN_ORDER.", len(ordered), ordered)
        return out

    def _record_layer_structure(self, ctx, idx, live):
        """Record the whole-layer op structure (tags only; the math is the GC recipe).
        Mirrors owned_engine.OwnedDecodeEngine._layer_build (owned_engine.py:701) in the
        vLLM (hidden, residual) representation. Buffers are register-only handles at the
        live VAs (NO host copy) so the bind-plan sees the true externals."""
        be, sp = self.hbe, self.spec
        B = live["hidden"].shape[0]
        # register-only Buffers at the live VAs (no synDeviceMalloc, no H2D). SKIP the
        # non-tensor sentinels (e.g. "__attn_stock__") — they route structure, not bind.
        buf = {tag: self._live_buffer(t, tag)
               for tag, t in live.items() if not str(tag).startswith("__")}
        H = sp.hidden
        # register-only SYMBOLIC workspace (NO synDeviceMalloc: bridge-owns-HBM => a raw
        # 48KB alloc returns status=26). These are recipe-INTERNAL edges (produced by one
        # recorded op, consumed by the next) -> _launch_tensors classifies them as workspace
        # (excluded from the bind-plan); the real numeric workspace is inside the GC recipe.
        b_normed = be.alloc_symbolic((B, H), self._eng.F64, tag=f"normed{idx}")
        b_attn = be.alloc_symbolic((B, H), self._eng.F64, tag=f"attn{idx}")
        b_normed2 = be.alloc_symbolic((B, H), self._eng.F64, tag=f"normed2{idx}")
        nop = lambda *a, **k: None
        # ── ATTENTION ────────────────────────────────────────────────────────────────
        if live.get("__attn_stock__"):
            # DELEGATE mode (MMEL_OWNED_ATTN=stock, the recommended first gate): input_norm
            # + self_attn ran on the stock path (run_layer._stock_attn, byte-exact); the
            # owned recipe is the STATIC TAIL [post_norm(fused-add) -> MoE], fed the LIVE
            # attention output `attn_out`. This is the audit's byte-exact static tail
            # (mmel_l6_chain.py:360-380) via the owned HpuBackend with RESIDENT-VA experts.
            ctx.record(f"post_norm{idx}", nop,
                       ins=(buf["attn_out"], buf["residual"], buf["w_post"]),
                       outs=(b_normed2, buf["residual"]))
        else:
            # OWNED attention STAGE (MMEL_OWNED_ATTN=owned): the raw-decomposed MLA. The
            # single `attn` op here is a PLACEHOLDER — a real paged-MLA capture also
            # references q_a/kv_a norm gammas + W_UK_T/W_UV (kv_b absorbed) + rope/pad_mask/
            # block_table; extend this record to the captured cap_mla_paged tensor set when
            # the attention recipe is brought up (§6.C).
            ctx.record(f"in_norm{idx}", nop,
                       ins=(buf["hidden"], buf["residual"], buf["w_in"]),
                       outs=(b_normed, buf["residual"]))
            attn_ins = [b_normed, buf["Wq"], buf["Wo"], buf["kv"],
                        buf["slot"], buf["ctxlen"]]
            if sp.attn_mode == self._eng.ATTN_MLA:
                attn_ins += [buf["Wkva"], buf["kvb"]]
            else:
                attn_ins += [buf["Wk"], buf["Wv"]]
            ctx.record(f"attn{idx}", nop, ins=tuple(attn_ins), outs=(b_attn, buf["kv"]))
            ctx.record(f"post_norm{idx}", nop,
                       ins=(b_attn, buf["residual"], buf["w_post"]),
                       outs=(b_normed2, buf["residual"]))
        # ── MoE / DENSE TAIL ─────────────────────────────────────────────────────────
        # The int4 expert feed MUST be distinct VAs (offset-0-clone gate,
        # hpu_backend.py:407): assert before recording so a bs>1 alias is caught.
        if sp.layer_is_dense(idx):
            # GLM dense = DeepseekV2MLP: MERGED gate_up + down (2 tensors, not 3).
            ctx.record(f"dense{idx}", nop,
                       ins=(b_normed2, buf["Wgu"], buf["Wd"]),
                       outs=buf["hidden"])
        else:
            # packed + scale + ZERO-POINT per int4 class (GLM auto-round int4 binds a
            # stacked zp tensor even though sym; cap_routed/manifest.txt pos 17-24 & 41-48).
            moe_feed = [buf["exp_c13"], buf["exp_s13"], buf["exp_z13"],
                        buf["exp_c2"], buf["exp_s2"], buf["exp_z2"]]
            be.assert_distinct_vas(moe_feed, where=f"moe{idx}")
            moe_ins = [b_normed2, buf["Wgate"]] + moe_feed
            if sp.n_shared > 0:
                # GLM shared expert is a bf16 DeepseekV2MLP (merged gate_up + down),
                # NOT int4 — summed with the routed output.
                moe_ins += [buf["sh_gu"], buf["sh_d"]]
            ctx.record(f"moe{idx}", nop, ins=tuple(moe_ins), outs=buf["hidden"])

    def _live_buffer(self, t, tag):
        """Register a LIVE resident HPU tensor as a device Buffer (VA-only; NO host copy).
        Contiguity is REQUIRED (its data_ptr is the row-major VA the recipe expects,
        :2058). Uses HpuBackend._register — the same register-only primitive to_device
        uses for the dev_va path (hpu_backend.py:312/334)."""
        import numpy as _np
        tc = t.contiguous() if not t.is_contiguous() else t
        va = _owned_resolve_va(tc)
        if not va:
            raise MMELChainError(
                f"[mmel.l6_chain] live tensor {tag!r} resolved to a NULL VA "
                f"(uncommitted lazy op?). mark_step before binding (:2755).")
        # shape/dtype are for the registry records only; we NEVER host-copy the weight.
        return self.hbe._register(int(va), (int(tc.numel()),), _np.uint8, tag)

    # ---- decode: bind live + enqueue + (one) sync --------------------------------
    def run_layer(self, layer, positions, hidden_states, residual, *, llama_4_scaling):
        import torch  # lazy — card only
        idx = int(layer.layer_idx)
        # forward-boundary detection (idx wraps to 0 => a new forward began).
        if self._last_idx is None or idx <= self._last_idx:
            self.hbe._launch_keepalive.clear()   # drop last forward's launch arrays
        # STATIC-TAIL wiring (§6.B): when attention is DELEGATED (MMEL_OWNED_ATTN=stock),
        # run the stock input_norm + self_attn HERE (byte-exact) to get the live attention
        # output the owned [post_norm -> MoE] recipe binds as `attn_out`, with the residual
        # threaded exactly as stock. When attention is OWNED, keep the (hidden, residual)
        # contract (layer-0 residual is a CLONE, not an alias; mmel_l6_chain.py:1462).
        attn_out = None
        if self._attn_is_stock():
            attn_out, residual = self._stock_attn(layer, positions, hidden_states,
                                                  residual, llama_4_scaling)
        elif residual is None:
            residual = hidden_states.clone()
        # re-collect the per-step LIVE tensors (hidden/residual/KV move every step) and
        # LOUD-check they are the SAME shapes the recipe was captured at.
        live = self._collect_live_tensors(layer, hidden_states=hidden_states,
                                          residual=residual, positions=positions)
        if attn_out is not None:
            live["attn_out"] = attn_out          # the static-tail post_norm addend
        # MATERIALISE the live activations BEFORE VA resolution (the island's commit-then-
        # bind, mmel_moe_island.py:759-782): attn_out/residual/hidden are un-flushed lazy
        # outputs whose .data_ptr() has no backed device VA until a graph boundary commits
        # them. One mark_step here flushes the producer graphs so _live_buffer resolves REAL
        # VAs (the resident weights are already settled). Both _ensure_recipe (first-hit
        # capture) and _rebind resolve within THIS call, so one flush covers them.
        try:
            import habana_frameworks.torch.core as _htcore  # lazy — card only
            _htcore.mark_step()
        except Exception:  # noqa: BLE001 — off-card / eager: no-op (VAs resolve directly)
            pass
        rec = self._ensure_recipe(layer, live)
        self._rebind(rec, idx, live)             # patch the bind-plan VAs for this step
        if not self._engaged_logged:
            self._engaged_logged = True
            logger.warning(
                "[mmel.l6_chain][OWNED-HPU][ENGAGE] MMEL_BACKEND=hpu: layer%d live "
                "hidden %s -> device-resident whole-layer recipe chain (route-2 "
                "synLaunch NO-fence + %s synStreamSynchronize). N-layer chain, resident "
                "weights (no re-upload). Prefill/non-M=1 delegate to the G1 spine.",
                idx, tuple(hidden_states.shape),
                "ONE/forward" if self.one_sync else "per-layer")
            _engage().emit_manifest(idx, {
                "engine-chain": "OWNED-HPU(whole-layer recipe chain; %s sync)" % (
                    "one/forward" if self.one_sync else "per-layer"),
                "STRICT": "ON" if _engage().strict_enabled() else "off"},
                once_key="engine-chain")
        self.hbe.enqueue(rec)                     # synLaunch NO fence (hpu_backend.py:518)
        if not self.one_sync or idx == self.n_layers - 1:
            self.hbe.synchronize()                # THE one host completion / forward
        self._last_idx = idx
        # the recipe wrote hidden_states/residual storage in place; return the live pair.
        out_h = self._live_by_tag[idx]["hidden"]
        out_r = self._live_by_tag[idx]["residual"]
        if _MMEL_POISON and out_r is not None:
            out_r = out_r * 1.03 + 0.01           # FORENSIC PROBE (G-poison)
        return out_h, out_r

    def _rebind(self, rec, idx, live):
        """Update the per-step mutable VAs on the recipe's bind-plan Buffers (hidden /
        residual / attn_out / KV move every step; weights are static). Re-registers the
        live VAs UNDER THE RECIPE'S POSITIONAL NAMES (the §5.B tag->name bridge) so
        HpuBackend.enqueue's build_launch_array binds the fresh addresses to the right
        recipe tensors. The engine-tag order is captured ONCE (bind_plan is overwritten
        each step with positional names, so we can't re-read tags from it)."""
        self._live_by_tag[idx] = live
        tags = getattr(rec, "_engine_bind_tags", None)
        if tags is None:
            tags = tuple(t for t, _ in rec.bind_plan)   # first step: still engine tags
            rec._engine_bind_tags = tags
        name_map = self._tag_to_name.get(idx) or {}
        plan, baked, live_names = [], [], set()
        for tag in tags:
            if tag not in live:
                raise MMELChainError(
                    f"[mmel.l6_chain] rebind: engine tag {tag!r} lost its live binding "
                    f"at decode (layer{idx}).")
            recipe_name = self._recipe_tensor_name(name_map, tag, idx)
            if recipe_name is None:
                # BAKED: GC const-folded this weight into the recipe (no launch tensor) ->
                # SKIP (no launch tensor to bind); it keeps its captured (const-section)
                # value. (Replicated: correct on all ranks; sharded: rank-0 -> _log_baked
                # SHARDED-RISK.)
                baked.append(tag)
                continue
            plan.append((recipe_name, self._live_buffer(live[tag], recipe_name)))
            live_names.add(recipe_name)
        # FULL LAUNCH SET: add the RESIDENT (stable-VA) buffers for EVERY recipe launch
        # tensor we did NOT live-override (outputs/scratch + baked/control persistent inputs).
        # synLaunch REQUIRES the complete persistent set (a partial set => status=30); the
        # resident buffers were alloc'd once at _ensure_recipe. Mirrors the island binding
        # ALL of its recipe tensors (static + outputs + live).
        for nm, buf in (self._resident_by_name.get(idx) or {}).items():
            plan.append((nm, buf))
        # bind-plan Buffers now carry the recipe's POSITIONAL names (enqueue keys by name).
        rec.bind_plan = tuple(plan)
        # COUNT guard: the launch set MUST equal the recipe's full tensor count, else
        # synLaunch rejects (status=30 / failed section validation). LOUD if under/over.
        expected = len(self._recipe_tensors.get(idx) or ())
        if expected and len(plan) != expected:
            raise MMELChainError(
                f"[mmel.l6_chain] layer{idx} launch set has {len(plan)} tensors but the "
                f"recipe needs {expected} ({len(live_names)} live-override + "
                f"{len(self._resident_by_name.get(idx) or {})} resident). synLaunch would "
                f"reject (status=30). A live tag matched a recipe tensor that is also in "
                f"the resident set (double-bind) or a recipe tensor is unaccounted for.")
        self._log_baked(idx, baked)

    # weight tags that are TP-REPLICATED (router logits + RMSNorm gammas): baking them
    # (const-folding into the recipe) is CORRECT on every rank (all ranks hold the same
    # values). Everything else that bakes (shared-expert gate_up/down, the int4 experts,
    # the attn projections) is SHARDED -> baking rank-0's values is WRONG on ranks>0.
    _REPLICATED_TAGS = frozenset({"Wgate", "w_post", "w_in", "ctxlen", "slot"})

    def _bake_class(self, tag):
        return "replicated" if tag in self._REPLICATED_TAGS else "sharded"

    def _log_baked(self, idx, baked):
        """One-shot LOUD [OWNED-HPU][BAKED] listing every engine tag const-folded into the
        recipe (skipped, not live-bound), classified replicated (bake correct) vs sharded
        (bake uses rank-0 values -> WRONG on ranks>0 -> per-rank capture needed)."""
        if not baked or self._baked_warned:
            return
        self._baked_warned = True
        classed = [(t, self._bake_class(t)) for t in baked]
        sharded = [t for t, c in classed if c == "sharded"]
        logger.warning(
            "[mmel.l6_chain][OWNED-HPU][BAKED] layer%d: %d launch tensor(s) CONST-FOLDED "
            "into the recipe (kept at captured values, NOT live-bound) -> %s. Replicated "
            "(router/norm) baking is CORRECT on all ranks; SHARDED baking uses rank-0 "
            "values.", idx, len(classed), classed)
        if sharded:
            logger.warning(
                "[mmel.l6_chain][OWNED-HPU][BAKED][SHARDED-RISK] %s are SHARDED but baked "
                "from rank-0 -> G-byte will likely GARBLE on TP ranks>0. If so, capture "
                "PER-RANK (each rank writes its own cap_tail<idx>). NOT implemented yet -- "
                "G-byte adjudicates whether it is actually needed.", sharded)

    # static-tail engine tag -> the manifest bridge's byte-size/ordinal key. The three
    # [1,H]-sized spine/norm tensors (attn_out, residual, w_post) are size-degenerate (all
    # H*2 bf16), so the bridge keys them by manifest ORDER (layer_in0..2). This alias
    # assumes the capture order attn_out < residual < w_post (the read order of
    # post_attention_layernorm(attn_out, residual, gamma)); a WRONG order fails the G-byte
    # gate LOUD (never corrupts a live generation — the seam's coherent A/B is the guard).
    _TAG_ALIAS = {"attn_out": "layer_in0", "residual": "layer_in1",
                  "w_post": "layer_in2", "hidden": "layer_out"}

    def _recipe_tensor_name(self, name_map, tag, idx):
        """Translate an engine tag -> the recipe's real positional tensor name via the
        tag->name bridge. Identity when there is no bridge (the recipe already uses semantic
        names). Returns None when the tag has NO matching recipe launch tensor -> it was
        CONST-FOLDED (baked) by GC -> _rebind SKIPS it (partial binding; the baked value is
        kept). LOUD only if a tag maps to MANY recipe tensors (a per-expert-sliced FROZEN
        recipe like cap_routed is NOT adoptable by a single stacked resident VA)."""
        if not name_map:
            return tag                       # no bridge => recipe uses semantic names
        names = name_map.get(tag)
        if names is None:
            names = name_map.get(self._TAG_ALIAS.get(tag, tag))
        if not names:
            return None                      # BAKED: no matching recipe launch tensor
        if len(names) != 1:
            raise MMELChainError(
                f"[mmel.l6_chain] layer{idx}: engine tag {tag!r} maps to {len(names)} "
                f"recipe tensors {names} -- a per-expert-sliced FROZEN recipe (e.g. "
                f"cap_routed) is NOT adoptable by the single STACKED resident VA. "
                f"Re-capture the tail from the LIVE-ROUTING STACKED op so each expert "
                f"class is ONE launch tensor (owned_all_layers §B.2).")
        return names[0]

    def _collect_live_tensors(self, layer, *, hidden_states=None, residual=None,
                              positions=None):
        """Map engine TAGS -> the layer's LIVE resident tensors. CARD-VALIDATED-ONLY: the
        exact attribute plumbing (input_layernorm.weight / self_attn.{q,kv,o} / experts.*)
        is image-specific and fails LOUD here on drift (never mis-binds). Weight tags are
        resolved once (warmup); hidden/residual/KV/positions are the per-step mutables."""
        m = {}
        # per-step mutables (only present at decode; None at warmup capture).
        if hidden_states is not None:
            m["hidden"] = hidden_states
        if residual is not None:
            m["residual"] = residual
        # --- norms ---
        m["w_in"] = self._req(layer, "input_layernorm", "weight")
        m["w_post"] = self._req(layer, "post_attention_layernorm", "weight")
        # --- attention weights + paged KV + slot/ctxlen (card-only plumbing) ---
        m.update(self._attn_live(layer, positions))
        # --- MoE / dense weights (resident int4; adopted by VA, ZERO re-upload) ---
        m.update(self._moe_live(layer))
        return m

    def _req(self, obj, *path):
        cur = obj
        for p in path:
            cur = getattr(cur, p, None)
            if cur is None:
                raise MMELChainError(
                    f"[mmel.l6_chain] required submodule/attr {'/'.join(path)!r} missing "
                    f"on {type(obj).__name__}; model-tree drift -> refusing to mis-bind.")
        return cur

    # ---- resident-weight resolvers (LOUD on drift; NO host copy) -------------------
    def _w(self, obj, *path):
        """Resolve a resident weight PARAMETER's ``.data`` tensor (the true device VA the
        recipe binds). LOUD via _req on any model-tree drift."""
        p = self._req(obj, *path)
        return getattr(p, "data", p)

    def _fused_zp(self, experts, attr, src_attr):
        """The op-ready STACKED [E,...] int4 zero-point (matches the cap_routed manifest
        layout). It is built LAZILY the first time the fused MoE op runs
        (moe_wna16_hpu.py:749-750) from the shared sym-zero ``src_attr`` (_hpu_routed_*_zp,
        set at process_weights_after_loading, :1461). At capture time a warmup MoE forward
        has already run, so ``attr`` exists; if not we build+CACHE it once on the module so
        its VA is STABLE across steps (a fresh broadcast every step would move the VA and
        break the static recipe)."""
        t = getattr(experts, attr, None)
        if t is not None:
            return getattr(t, "data", t)
        src = getattr(experts, src_attr, None)
        if src is None:
            raise MMELChainError(
                f"[mmel.l6_chain] MoE zero-point {attr!r} (and source {src_attr!r}) "
                f"absent on {type(experts).__name__}. Run one warmup MoE forward before "
                f"the owned capture so the fused int4 op builds the stacked zp "
                f"(moe_wna16_hpu.py:749), or the model is not GPTQ-sym int4.")
        E = int(getattr(experts, "_hpu_local_experts", None)
                or self.spec.n_routed_experts)
        s = getattr(src, "data", src)
        stacked = s.unsqueeze(0).expand(E, *s.shape).contiguous()
        setattr(experts, attr, stacked)      # cache for a STABLE VA (mirror :749)
        return stacked

    def _attn_is_stock(self):
        """True (DEFAULT) => attention is DELEGATED to the stock/C12 path and the owned
        recipe is the STATIC TAIL [post_norm -> MoE] (MMEL_OWNED_ATTN=stock, the
        recommended first gate). False => the OWNED MLA attention stage."""
        return os.environ.get("MMEL_OWNED_ATTN", "stock").strip().lower() in (
            "", "stock", "delegate", "off")

    def _attn_live(self, layer, positions):
        """Attention externals mapped to the engine tags.

        DEFAULT (MMEL_OWNED_ATTN=stock): DELEGATE — return the sentinel
        ``{"__attn_stock__": True}``. input_norm + self_attn run on the stock path
        (run_layer._stock_attn, byte-exact); the owned recipe is the STATIC TAIL and binds
        the live stock-attention output as ``attn_out``.

        MMEL_OWNED_ATTN=owned: the attention STAGE — resolve the resident MLA projection
        weights (adopted by VA, no re-upload) + the LIVE paged-KV/control tensors, REUSING
        the proven C12 resolvers (_resolve_live_attn_sources / _build_live_attn_ctx). LOUD
        on any unresolved source (a wrong KV bind must crash a gate, not corrupt output)."""
        mode = os.environ.get("MMEL_OWNED_ATTN", "stock").strip().lower()
        if mode in ("", "stock", "delegate", "off"):
            return {"__attn_stock__": True}
        if mode not in ("owned", "hpu", "on"):
            raise MMELChainError(
                f"[mmel.l6_chain] MMEL_OWNED_ATTN={mode!r} unknown; use 'stock' (delegate,"
                f" the static-tail first gate) or 'owned' (the MLA attention stage).")
        attn = self._req(layer, "self_attn")
        m = {}
        # resident projection weights (adopted by VA). GLM MLA q_lora=2048 -> fused_qkv_a
        # + q_b_proj; the no-q_lora branch (q_proj + kv_a_proj_with_mqa) is kept for
        # portability (deepseek_v2.py:926-969).
        if getattr(attn, "q_lora_rank", None) is not None or hasattr(
                attn, "fused_qkv_a_proj"):
            m["Wkva"] = self._w(attn, "fused_qkv_a_proj", "weight")   # a-proj (q_a‖kv_a‖rope)
            m["Wq"] = self._w(attn, "q_b_proj", "weight")             # q up-proj
        else:
            m["Wq"] = self._w(attn, "q_proj", "weight")
            m["Wkva"] = self._w(attn, "kv_a_proj_with_mqa", "weight")
        m["kvb"] = self._w(attn, "kv_b_proj", "weight")   # NB: absorbed into W_UK_T/W_UV
        m["Wo"] = self._w(attn, "o_proj", "weight")
        # LIVE paged-KV + control tensors — REUSE the C12 seam verbatim.
        attn_metadata, kv_cache = _resolve_live_attn_sources(layer)
        if attn_metadata is None or kv_cache is None or len(kv_cache) < 1:
            raise MMELChainError(
                "[mmel.l6_chain] MMEL_OWNED_ATTN=owned: could not resolve "
                "(attn_metadata, kv_cache) from self_attn + the forward context "
                "(_resolve_live_attn_sources, :758). Delegate attention "
                "(MMEL_OWNED_ATTN=stock) or bring up the MLA recipe first.")
        m["kv"] = kv_cache[0]
        m["slot"] = getattr(attn_metadata, "slot_mapping", None)
        if m["slot"] is None:
            raise MMELChainError(
                "[mmel.l6_chain] MMEL_OWNED_ATTN=owned: attn_metadata has no "
                "slot_mapping (KV write target); the MLA cache-write cannot bind.")
        m["ctxlen"] = positions            # static per bucket; no host .item() (:1878)
        # the paged-MLA op's live control set (block_table / pad_mask / q_cos / q_sin),
        # keyed by the cap_mla_paged recipe's own names — extra keys are harmless to the
        # bind_plan check (one-directional) and are consumed once the attention STAGE
        # records the real MLA op (§6.C).
        live_ctx = _build_live_attn_ctx(layer, positions)
        if live_ctx is not None:
            for k, v in live_ctx.items():
                if not str(k).startswith("__") and k not in m:
                    m[k] = v
        return m

    def _moe_live(self, layer):
        """MoE/dense externals from the RESIDENT int4 experts, adopted by VA (no re-upload
        -> the OOM the frozen island forced does not exist here; owned_all_layers §B.2,
        mmel_l6_chain.py:415). Live routing is IN-GRAPH (computed from Wgate), so the
        K-of-256 selection is a recipe intermediate, NOT an external tensor -> per-layer-
        correct (unlike the frozen cap_routed island). LOUD on any drift.

        Attribute names are the ACTIVE fused-op serve path (moe_wna16_hpu.py:639-640,
        1420-1439): experts.w13_qweight/w2_qweight (int32 stacked [E,...]),
        w13_scales/w2_scales (bf16 op-layout), _hpu_fused_w13_zp/_hpu_fused_w2_zp (int32
        stacked zp). Dense (idx<first_k_dense) is a plain DeepseekV2MLP: MERGED gate_up +
        down (deepseek_v2.py:216-233). Shared expert (n_shared=1) is a bf16 DeepseekV2MLP
        (deepseek_v2.py:316), summed with the routed output — NOT int4."""
        mlp = self._req(layer, "mlp")
        idx = int(layer.layer_idx)
        m = {}
        if self.spec.layer_is_dense(idx):
            m["Wgu"] = self._w(mlp, "gate_up_proj", "weight")      # [2*Idense, H] bf16
            m["Wd"] = self._w(mlp, "down_proj", "weight")          # [H, Idense]  bf16
            return m
        m["Wgate"] = self._w(mlp, "gate", "weight")                # [E, H] bf16 router
        experts = self._req(mlp, "experts")                        # FusedMoE
        m["exp_c13"] = self._w(experts, "w13_qweight")             # [E, H, 2I//8] int32
        m["exp_s13"] = self._w(experts, "w13_scales")              # [E, H//GS, 2I] bf16
        m["exp_c2"] = self._w(experts, "w2_qweight")               # [E, I, H//8] int32
        m["exp_s2"] = self._w(experts, "w2_scales")                # [E, I//GS, H] bf16
        m["exp_z13"] = self._fused_zp(experts, "_hpu_fused_w13_zp", "_hpu_routed_w13_zp")
        m["exp_z2"] = self._fused_zp(experts, "_hpu_fused_w2_zp", "_hpu_routed_w2_zp")
        if self.spec.n_shared > 0:
            sh = self._req(mlp, "shared_experts")
            m["sh_gu"] = self._w(sh, "gate_up_proj", "weight")     # [2*Ish, H] bf16
            m["sh_d"] = self._w(sh, "down_proj", "weight")         # [H, Ish]  bf16
        return m

    def _stock_attn(self, layer, positions, hidden_states, residual, llama_4_scaling):
        """Run the stock input_norm + self_attn to produce ``(attn_out, residual)`` for the
        static-tail owned recipe [post_norm -> MoE]. BYTE-EXACT BY CONSTRUCTION: it calls
        the IDENTICAL submodules the stock DeepseekV2DecoderLayer.forward calls, with the
        identical residual contract (deepseek_v2.py:1176-1200). Chosen over the C12
        _try_live_attn device replay precisely because that is byte-exact only when its
        captures match (extra risk); the whole point of the static-tail baseline is a
        trivially-byte-exact attention. LOUD if self_attn is unavailable."""
        import torch  # lazy — card only
        in_norm = self._req(layer, "input_layernorm")
        attn = self._req(layer, "self_attn")
        if residual is None:
            residual = hidden_states.clone()          # layer-0: CLONE (deepseek_v2.py:1177)
            normed = in_norm(hidden_states)
        else:
            normed, residual = in_norm(hidden_states, residual)   # fused-add
        attn_kwargs = {"positions": positions, "hidden_states": normed}
        if not getattr(layer, "use_mha", False):
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        attn_out = attn(**attn_kwargs)
        if attn_out is None:
            raise MMELChainError(
                f"[mmel.l6_chain] stock self_attn returned None for layer "
                f"{getattr(layer, 'layer_idx', '?')}; cannot feed the static tail.")
        # FP16-overflow parity (no-op for bf16 GLM; deepseek_v2.py:1191-1200). MLA path
        # (not use_mha) scales attn_out, and the residual once on layer 0.
        if attn_out.dtype == torch.float16 and not getattr(layer, "use_mha", False):
            rsf = float(getattr(layer, "routed_scaling_factor", 1.0)) or 1.0
            attn_out = attn_out * (1.0 / rsf)
            if int(layer.layer_idx) == 0:
                residual = residual * (1.0 / rsf)
        return attn_out, residual

    def _maybe_capture_tail(self, layer, idx, live):
        """Stage-A build_tail (§6.4, CARD-ONLY to RUN): if this layer's tail recipe is not
        on disk, capture it ONCE at warmup by tracing the REAL layer tail
        (post_attention_layernorm fused-add + layer.mlp live-routing fused op) under the
        bridge's recipe-cache serialize route (the SAME route the per-op islands used,
        capture_routed_island.py door A), then select the blob -> cap_<name>/island.recipe
        + write its manifest.txt. Replays thereafter. NOT the nop build — the nop only
        supplies the STRUCTURE/tags for the bind-plan; THIS produces the numeric GC recipe.

        Only for the static tail (MMEL_OWNED_ATTN=stock); the owned MLA stage reuses the
        existing cap_qproj/cap_kvproj/cap_mla_paged/cap_oproj recipes (§6.C)."""
        if not self._attn_is_stock() or idx in self._captured_tail:
            return
        name = self._recipe_name(idx)
        cap_dir = os.path.join(self.hbe._cap_dir, f"cap_{name}")
        recipe_out = os.path.join(cap_dir, "island.recipe")
        # already on disk (a prior run / offline capture / another rank)? -> replay.
        if os.path.isfile(recipe_out):
            self._captured_tail.add(idx)
            return
        # RANK-SAFETY: only the driver rank captures+writes; the other 7 ranks WAIT for the
        # shared recipe file, then replay it binding THEIR OWN resident VAs (a VA-agnostic
        # launch-tensor recipe is rank-portable — proven by the per-op islands running on
        # all 8 ranks from one shared cap_*/island.recipe). Avoids the 8-way write race +
        # duplicate selection. Undetectable rank => every rank captures but writes via an
        # ATOMIC os.replace(tmp,final), so concurrent writers never tear the file.
        rank = self._driver_rank()
        if rank is not None and rank > 0:
            self._wait_for_recipe(recipe_out, idx)
            self._captured_tail.add(idx)
            return
        os.makedirs(cap_dir, exist_ok=True)
        self._cap_module()                     # validate the capture helpers are importable
        rcache = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "")
        if not rcache:
            raise MMELChainError(
                f"[mmel.l6_chain] Stage-A auto-capture requires PT_HPU_RECIPE_CACHE_CONFIG "
                f"set to a serializing cache at SERVE LAUNCH (door A is a PRE-import env; it "
                f"cannot be set at decode). Launch the serve with "
                f"PT_HPU_RECIPE_CACHE_CONFIG=<dir>/_rcache,false,<MB> (capture_routed_"
                f"island.py:45), OR pre-capture cap_{name}/island.recipe offline. The wire "
                f"then deserializes+replays it (no per-decode capture).")
        rdir = rcache.split(",")[0]
        # _trace_tail guarantees the tail recipe is COMPILED + in the cache; it is a CACHE
        # HIT (layers 3..idx-1 ran the structurally-identical M=1 tail via stock decode in
        # this same step), so door-A "newest-after-t0" finds NOTHING new. We instead
        # SELECT-BY-SIGNATURE over ALL cache blobs (below).
        self._trace_tail(layer, live)
        blob, tensors, big4 = self._select_tail_blob(rdir)
        # ── #1 ON-CARD RISK DIAGNOSTIC (adopt-vs-fold): does the selected recipe carry the
        #    E-stacked int4 expert weights as EXTERNAL launch tensors (=> ADOPTABLE by live
        #    VA, zero-copy) — or are they absent (=> const-folded/baked => not adoptable)? We
        #    FOUND the blob by its external packed experts, so EXTERNAL=yes here; the 0-match
        #    path in _select_tail_blob is the EXTERNAL=no (folded) LOUD.
        n_infile = sum(1 for t in tensors if int(t.get("is_input", 1)))
        logger.warning(
            "[mmel.l6_chain][OWNED-HPU][CAPTURE] tail%d experts EXTERNAL=yes (n_infile=%d, "
            "expert_tensors=%s)", idx,
            n_infile, [(t["name"], int(t["nbytes"])) for t in big4])
        # atomic publish (tmp -> os.replace) so a concurrent/undetected-rank writer never
        # tears the file another rank is deserializing.
        import shutil
        tmp = recipe_out + f".tmp.{os.getpid()}"
        shutil.copyfile(blob, tmp)
        os.replace(tmp, recipe_out)
        self._write_tail_manifest(cap_dir, tensors)
        self._captured_tail.add(idx)
        logger.warning(
            "[mmel.l6_chain][OWNED-HPU][CAPTURE] Stage-A tail recipe for layer%d SELECTED "
            "by signature -> %s (%d launch tensors; + manifest). Replaying thereafter.",
            idx, recipe_out, len(tensors))

    # a launch tensor is a stacked int4 PACKED expert weight iff it is a LARGE int32
    # (4-byte) tensor; scales are bf16 (2-byte) and dense GEMM weights are bf16, so the
    # single LARGEST launch tensor being int32 uniquely marks the stacked int4 MoE tail
    # (sharding-agnostic: the packed weight stays the largest by far, ~hundreds of MB).
    _TAIL_MIN_PACKED = 8 * 1024 * 1024      # 8 MB floor (sharded packed ~200MB; scales ~MB)

    def _select_tail_blob(self, rcache_dir):
        """SELECT-BY-SIGNATURE (replaces door-A newest-after, which fails because the M=1
        tail recipe is structurally identical across MoE layers and was already compiled —
        a CACHE HIT — by an earlier layer's stock decode, so no NEW blob appears). Scan ALL
        recipe blobs; a tail candidate is a blob whose SINGLE LARGEST launch tensor is a
        large int32 (width 4) — the stacked int4 PACKED expert weight (exp_c13/exp_c2) —
        with >=2 such large int32 tensors. Dense tails' largest is bf16; attention/embed/
        head have no large int32. Multiple candidates = the SAME MoE tail at different SHAPE
        BUCKETS (prompt-tails at various seqlens + the M=1 decode tail); disambiguate to the
        M=1 DECODE tail via _pick_m1_decode_tail (superset of the live M=1 multiset +
        smallest total). LOUD on 0 (=> experts CONST-FOLDED => NOT adoptable) or an
        indistinguishable tie."""
        import glob
        blobs = [p for p in glob.glob(os.path.join(rcache_dir, "**", "*"), recursive=True)
                 if os.path.isfile(p) and not os.path.basename(p).lower().endswith(
                     (".lock", ".log", ".txt", ".json", ".csv"))]
        cands, summary = [], []
        for p in blobs:
            try:
                tensors = self._read_recipe_tensors(p)
            except Exception:  # noqa: BLE001 — not a readable recipe blob: skip
                continue
            if not tensors:
                continue
            largest = max(tensors, key=lambda t: int(t["nbytes"]))
            summary.append((os.path.basename(p), len(tensors),
                            int(largest["nbytes"]), _tensor_width(largest)))
            if _is_moe_tail_signature(tensors, self._TAIL_MIN_PACKED):
                big4 = [t for t in tensors if int(t["nbytes"]) >= self._TAIL_MIN_PACKED
                        and _tensor_width(t) == 4]
                cands.append((p, tensors, big4))
        if not cands:
            raise MMELChainError(
                f"[mmel.l6_chain][OWNED-HPU][CAPTURE] tail select-by-signature: 0 matches "
                f"=> the stacked int4 expert weights are NOT external launch tensors "
                f"(CONST-FOLDED/baked) => EXTERNAL=no => NOT adoptable by live VA (+HBM) AND "
                f"a shared recipe would bake rank-0 weights (WRONG on other ranks) => "
                f"capture PER-RANK. scanned {len(summary)} blobs in {rcache_dir!r} "
                f"[(name,n_tensors,largest_nbytes,largest_width)]={summary}")
        if len(cands) == 1:
            return cands[0]
        # >1: same MoE tail at different shape buckets -> pick the M=1 DECODE tail = the
        # structurally LEANEST recipe (fewest launch tensors; prompt/seq buckets carry extra
        # [S,6144] activation intermediates).
        idx_pick, reason = _pick_m1_decode_tail([c[1] for c in cands])
        if idx_pick is None:
            detail = [(os.path.basename(p), len(tensors),
                       sorted((int(t["nbytes"]), _tensor_width(t), int(t.get("is_input", 1)))
                              for t in tensors))
                      for (p, tensors, _b) in cands]
            raise MMELChainError(
                f"[mmel.l6_chain][OWNED-HPU][CAPTURE] tail select: {len(cands)} int4-tail "
                f"candidates (same MoE tail at different shape buckets) but cannot pick the "
                f"M=1 decode tail ({reason}). candidate "
                f"(name,n_tensors,[(nbytes,width,is_input)])={detail}")
        logger.warning(
            "[mmel.l6_chain][OWNED-HPU][CAPTURE] disambiguated %d int4-tail shape buckets "
            "-> M=1 decode tail %s (%s)", len(cands),
            os.path.basename(cands[idx_pick][0]), reason)
        return cands[idx_pick]

    def _driver_rank(self):
        """The TP rank if resolvable (vllm.distributed), else None (=> undetectable; the
        caller then uses atomic-replace so concurrent writers don't tear the file)."""
        try:
            from vllm.distributed import get_tensor_model_parallel_rank  # type: ignore
            return int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            return None

    def _wait_for_recipe(self, recipe_out, idx, timeout=300.0):
        """Non-driver ranks BOUNDED-poll for the driver's shared tail recipe, then proceed
        to replay it. LOUD timeout (never hang the worker forever)."""
        import time as _t
        deadline = _t.time() + float(timeout)
        stable = 0
        last = -1
        while _t.time() < deadline:
            if os.path.isfile(recipe_out):
                try:
                    sz = os.path.getsize(recipe_out)
                except OSError:
                    sz = -1
                if sz > 0 and sz == last:
                    stable += 1
                    if stable >= 2:
                        return
                else:
                    stable = 0
                last = sz
            _t.sleep(0.5)
        raise MMELChainError(
            f"[mmel.l6_chain] non-driver rank timed out ({timeout:.0f}s) waiting for the "
            f"driver's Stage-A tail recipe {recipe_out!r} (layer{idx}). The driver rank "
            f"either failed to capture (check its [CAPTURE] log) or the cap dir is not "
            f"shared across ranks.")

    def _trace_tail(self, layer, live):
        """The build_tail trace: run the REAL layer tail (post_attention_layernorm fused-add
        on the live attn_out + residual, then layer.mlp live-routing fused op) once under a
        mark_step so the bridge compiles + serializes its GC recipe. Byte-shape-identical to
        the stock tail; the live-routing MoE op reads the STACKED resident int4 tensors, so
        the compiled recipe's expert externals are the resident VAs (see the §6.5 honesty
        note on adoptability)."""
        import torch  # noqa: F401 — card only
        import habana_frameworks.torch.core as htorch  # noqa: E402 — card only
        post_norm = self._req(layer, "post_attention_layernorm")
        mlp = self._req(layer, "mlp")
        attn_out = live.get("attn_out")
        residual = live.get("residual")
        if attn_out is None or residual is None:
            raise MMELChainError(
                "[mmel.l6_chain] _trace_tail needs live attn_out + residual (static-tail "
                "mode); run_layer must populate them before capture.")
        normed2, _res = post_norm(attn_out, residual)   # fused-add RMSNorm
        _ = mlp(normed2)                                 # live-routing fused int4 MoE
        htorch.mark_step()                               # compile+serialize the tail recipe

    def _write_tail_manifest(self, cap_dir, tensors):
        """Write cap_<tail>/manifest.txt from the recipe's launch-tensor list (already read
        via _read_recipe_tensors: dicts {name,id,is_input,dtype,ndim,nelem,nbytes}) + a dims
        header. DIAGNOSTIC ONLY (the on-card bind matches live<->recipe by (nbytes,width,
        is_input), NOT this file); it records the ACTUAL sharded sizes + width + ndim (ndim
        is unreliable=0 from the reader) so the adopt/fold + collision analysis is
        inspectable. Atomic (tmp -> os.replace)."""
        sp = self.spec
        lines = [
            f"# island=tail quant=int4 top_k={sp.top_k} n_routed={sp.n_routed_experts}",
            f"# M=1 H={sp.hidden} I={sp.moe_inter} GS={sp.group_size} "
            f"K={sp.top_k} n_routed={sp.n_routed_experts}  static-tail [post_norm->MoE]",
            "# name\tnbytes\trole\twidth\tndim  (name == recipe's own launch-tensor name)",
        ]
        for t in sorted(tensors, key=lambda x: int(x.get("id", 0))):
            nb = int(t["nbytes"]); ne = max(int(t.get("nelem", 1)), 1)
            role = "IN_FILE" if int(t.get("is_input", 1)) else "OUT"
            lines.append(f"{t['name']}\t{nb}\t{role}\t{nb // ne}\t{int(t.get('ndim', 0))}")
        path = os.path.join(cap_dir, "manifest.txt")
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)


# ----------------------------------------------------------------------------- #
#  WIRE THE SEAM: flip chain_is_wired() true by importing this module.           #
#                                                                               #
#  The seam (mmel_l6_exec.py) detects the chain two ways (no edit to it needed): #
#   (a) env MMEL_CHAIN_WIRED in {1,true,yes,on}, OR                              #
#   (b) run_chained_decoder_layer no longer carries `_mmel_stub is True`.        #
#  We do BOTH: set the env AND rebind the seam's run_chained_decoder_layer to    #
#  OUR real body (deleting the stub marker). The seam's _mmel_decoder_forward    #
#  calls `run_chained_decoder_layer` looked up in the SEAM's module namespace,   #
#  so we rebind the name THERE. This is the single integration action; it is     #
#  inert unless MODERN_MOE_EXEC is also on (the seam's gate 1).                  #
# ----------------------------------------------------------------------------- #
def _wire_into_seam() -> bool:
    """Install OUR run_chained_decoder_layer as the seam's chained-exec body.

    Returns True if the seam module was found + rebound. LOUD-warns (does not
    raise) if the seam is absent, so importing this module on a box without the
    seam (e.g. a pure CPU unit-test of just this file) is harmless."""
    # mark the env so chain_is_wired() is true even before the rebind is observed.
    os.environ["MMEL_CHAIN_WIRED"] = "1"
    try:
        from vllm_gaudi.v1.worker import mmel_l6_exec as _seam  # type: ignore
    except Exception:  # noqa: BLE001 — try the path-based sibling import next
        _seam = _import_seam_by_path()
    if _seam is None:
        logger.warning(
            "[mmel.l6_chain] seam module (mmel_l6_exec) not importable; set "
            "MMEL_CHAIN_WIRED=1 (done) so chain_is_wired() is true, but the live "
            "rebind did not happen. In-serve, ensure both files share the "
            "vllm_gaudi.v1.worker package.")
        return False
    # Rebind the seam's name to OUR body and drop the stub marker so BOTH
    # detection paths agree. We also tag our function so introspection is clear.
    run_chained_decoder_layer._mmel_stub = False  # type: ignore[attr-defined]
    _seam.run_chained_decoder_layer = run_chained_decoder_layer
    # the seam's own reference must also lose the stub marker for chain_is_wired().
    try:
        if getattr(_seam.run_chained_decoder_layer, "_mmel_stub", False) is True:
            _seam.run_chained_decoder_layer._mmel_stub = False  # type: ignore
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        "[mmel.l6_chain] chained-exec body WIRED into the L6 seam "
        "(chain_is_wired()==True). Inert unless MODERN_MOE_EXEC=1.")
    return True


def _import_seam_by_path():
    """Path-based fallback import of the seam (mirrors how the seam imports L5)."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "mmel_l6_exec.py")
    if not os.path.isfile(cand):
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "vllm_gaudi.v1.worker.mmel_l6_exec", cand)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001
        return None


# Mark our body as NOT-the-stub so chain_is_wired() detects it even if the rebind
# path is not exercised (e.g. the seam imports THIS module directly in future).
run_chained_decoder_layer._mmel_stub = False  # type: ignore[attr-defined]
# This module being importable means the chain is wired in this image.
MMEL_CHAIN_WIRED = 1

# Perform the wire-in at import (the single integration action). Guarded so a unit
# test of just this file (no seam, no torch) still imports cleanly.
_WIRED = _wire_into_seam()


# ============================================================================= #
#  CARD-FREE SELF-TEST / GATE (the W0 discipline — no accelerator touched).      #
#                                                                               #
#   C1. next_pow2 / pad_to_multiple arithmetic (the DSA static-shape fixes).     #
#   C2. StaticShape.from_exec_cfg derives bucket-stable geometry (DSA + non-DSA) #
#       and assert_live_tensor catches a drift LOUD.                             #
#   C3. route_fold_is_static gate logic.                                         #
#   C4. the op-symbol path resolution (lazy vs eager .so; env override).         #
#   C5. chain_is_wired contract: importing this module flips the stub off + sets #
#       MMEL_CHAIN_WIRED (the seam detects it without device code).              #
#   C6. run_chained_decoder_layer LOUD-guards: no layer_idx -> raises; the       #
#       MMEL_CHAIN_MOE_OP=1 unbound path raises (refuses to ship an unbound op). #
#   C7. the four captured island dirs exist with manifests (the recipe the chain #
#       drives is present on disk) — maps, does not rebuild.                     #
# ============================================================================= #
_CAP_DIRS = (
    "$HOME/AI/mmel/cap_norm",
    "$HOME/AI/mmel/cap_qproj",
    "$HOME/AI/mmel/cap_kvproj",
    "$HOME/AI/mmel/cap_mla_paged",
    "$HOME/AI/mmel/cap_oproj",
    "$HOME/AI/mmel/cap_routed",
)


def _mk_fake_exec_cfg(*, has_dsa: bool):
    """A minimal duck-typed ExecConfig for card-free tests (no torch, no HF)."""
    class _Dsa:
        index_topk = 2048

    class _ECfg:
        hidden_size = 6144
        n_routed_experts = 256
        top_k = 8
        attn_mode = _ATTN_MLA_LATENT

    e = _ECfg()
    e.has_dsa = has_dsa
    e.dsa = _Dsa() if has_dsa else None

    def _layer_is_dense(i):
        return i < 3
    e.layer_is_dense = _layer_is_dense
    return e


def _c1_arith():
    assert next_pow2(1) == 1 and next_pow2(2) == 2 and next_pow2(3) == 4
    assert next_pow2(2049) == 4096 and next_pow2(2048) == 2048
    assert pad_to_multiple(1, 128) == 128
    assert pad_to_multiple(128, 128) == 128
    assert pad_to_multiple(129, 128) == 256
    assert pad_to_multiple(500, 128) == 512
    return "C1 next_pow2 + pad_to_multiple arithmetic correct (DSA static-shape fixes)"


def _c2_static_shape():
    e = _mk_fake_exec_cfg(has_dsa=True)
    s = StaticShape.from_exec_cfg(e, batch=1, context_len=500, block_size=128)
    assert s.batch == 1 and s.hidden_size == 6144
    assert s.gather_width == 512, f"gather_width {s.gather_width} != 512"
    assert s.index_topk_pad == 2049, f"index_topk_pad {s.index_topk_pad} != 2049"
    assert s.kk == 4096, f"kk {s.kk} != 4096 (pow2 of 2049)"
    assert s.n_routed == 256 and s.top_k == 8
    # non-DSA family: no index fields.
    e2 = _mk_fake_exec_cfg(has_dsa=False)
    s2 = StaticShape.from_exec_cfg(e2, batch=1, context_len=129, block_size=128)
    assert s2.index_topk_pad == 0 and s2.kk == 0 and s2.gather_width == 256
    # assert_live_tensor: matching dim passes; drift raises LOUD.
    s.assert_live_tensor("h", (1, 6144), dim=-1, expect=6144)
    raised = False
    try:
        s.assert_live_tensor("h", (1, 4096), dim=-1, expect=6144)
    except MMELShapeError:
        raised = True
    assert raised, "static-shape drift must raise MMELShapeError"
    # bad batch / context raise.
    for bad in (dict(batch=0, context_len=1), dict(batch=1, context_len=0)):
        r = False
        try:
            StaticShape.from_exec_cfg(e, **bad)
        except MMELShapeError:
            r = True
        assert r, f"bad {bad} must raise"
    return "C2 StaticShape derives bucket-stable geometry; assert_live_tensor LOUD on drift"


def _c3_route_fold():
    assert route_fold_is_static(256, 8) is True
    assert route_fold_is_static(8, 8) is True
    assert route_fold_is_static(8, 0) is False
    assert route_fold_is_static(8, 9) is False
    return "C3 route_fold_is_static gate logic correct"


def _c4_op_paths():
    # default (no env): lazy vs eager picks the right .so name.
    os.environ.pop(MOE_ROUTED_SO_ENV, None)
    os.environ.pop(MLA_DECODE_SO_ENV, None)
    assert _routed_so_path(True).endswith("libmoe_fused_routed_lazy.so")
    assert _routed_so_path(False).endswith("libmoe_fused_routed.so")
    assert _mla_so_path(True).endswith("libmla_decode_lazy.so")
    assert _mla_so_path(False).endswith("libmla_decode.so")
    # The resolved default must be either the in-container /opt path (the live serve)
    # or the host path (the self-test box) — and on a box where NEITHER exists it
    # must fall back to the in-container /opt default (so the loader's existence
    # check raises an IN-CONTAINER-correct message, never a stale host path).
    rp = _routed_so_path(True)
    assert rp.startswith(_MOE_EXT_BUILD_INCONTAINER) or rp.startswith(_MOE_EXT_BUILD_HOST), \
        f"routed default {rp!r} is neither the /opt nor the host build dir"
    # _first_existing_or_first returns the FIRST candidate (the /opt path) when none
    # exist — verify that contract directly so the in-container default is guaranteed.
    none_exist = _first_existing_or_first(
        ["/opt/hpu-moe-ext/__nope__/x.so", "$HOME/AI/hpu-moe-ext/__nope__/x.so"])
    assert none_exist == "/opt/hpu-moe-ext/__nope__/x.so", \
        "when no candidate exists the in-container /opt default must be chosen"
    # env override wins over BOTH defaults.
    os.environ[MOE_ROUTED_SO_ENV] = "/tmp/custom_routed.so"
    assert _routed_so_path(True) == "/tmp/custom_routed.so"
    os.environ.pop(MOE_ROUTED_SO_ENV, None)
    return "C4 op .so path resolution (in-container /opt default + host fallback + env override) correct"


def _c4b_routed_optional_in_baseline():
    """The routed int4 op .so must be REQUIRED only under MMEL_CHAIN_MOE_OP=1.

    Baseline mode (flag unset) is the B3.3 fence-collapse measurement: the chain uses
    the STOCK MoE submodule and never calls the routed op, so _ensure_ops_loaded must
    NOT raise even when no routed .so is reachable. Op-swap mode (flag=1) drives the
    op directly, so a missing .so there MUST raise. We verify both verdicts WITHOUT a
    card by pointing the env at a non-existent .so and toggling the gate; torch must
    be importable for _ensure_ops_loaded (CPU torch satisfies it)."""
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return "C4b SKIP (torch not importable here; gate logic exercised on-card)"
    # reset the per-process load cache so the gate logic actually runs.
    _OPS_LOADED["moe"] = False
    _OPS_LOADED["mla"] = False
    prev_so = os.environ.get(MOE_ROUTED_SO_ENV)
    prev_gate = os.environ.get("MMEL_CHAIN_MOE_OP")
    os.environ[MOE_ROUTED_SO_ENV] = "/nonexistent/libmoe_fused_routed_lazy.so"
    try:
        # baseline (gate OFF): missing routed .so must NOT raise.
        os.environ.pop("MMEL_CHAIN_MOE_OP", None)
        assert moe_op_swap_enabled() is False
        raised = False
        try:
            _ensure_ops_loaded(need_mla=False)  # must be a clean no-raise
        except MMELOpMissingError:
            raised = True
        assert not raised, ("baseline (MMEL_CHAIN_MOE_OP unset) must NOT raise on a "
                            "missing routed .so — the stock MoE submodule runs")
        # op-swap (gate ON): missing routed .so MUST raise LOUD.
        _OPS_LOADED["moe"] = False
        os.environ["MMEL_CHAIN_MOE_OP"] = "1"
        assert moe_op_swap_enabled() is True
        raised = False
        try:
            _ensure_ops_loaded(need_mla=False)
        except MMELOpMissingError:
            raised = True
        assert raised, ("MMEL_CHAIN_MOE_OP=1 with a missing routed .so MUST raise "
                        "MMELOpMissingError (the op-swap directly calls the op)")
    finally:
        _OPS_LOADED["moe"] = False
        _OPS_LOADED["mla"] = False
        if prev_so is None:
            os.environ.pop(MOE_ROUTED_SO_ENV, None)
        else:
            os.environ[MOE_ROUTED_SO_ENV] = prev_so
        if prev_gate is None:
            os.environ.pop("MMEL_CHAIN_MOE_OP", None)
        else:
            os.environ["MMEL_CHAIN_MOE_OP"] = prev_gate
    return ("C4b routed int4 op .so is OPTIONAL in baseline (no-raise) + REQUIRED "
            "under MMEL_CHAIN_MOE_OP=1 (LOUD-raise)")


def _c5_wired_contract():
    # importing this module set the env + dropped the stub marker on our body.
    assert os.environ.get("MMEL_CHAIN_WIRED") == "1", "import must set MMEL_CHAIN_WIRED=1"
    assert getattr(run_chained_decoder_layer, "_mmel_stub", True) is False, \
        "our body must NOT carry the stub marker (chain_is_wired detects this)"
    assert MMEL_CHAIN_WIRED == 1
    # if the seam is importable, chain_is_wired() must now be True.
    seam = None
    try:
        from vllm_gaudi.v1.worker import mmel_l6_exec as seam  # type: ignore
    except Exception:  # noqa: BLE001
        seam = _import_seam_by_path()
    if seam is not None:
        assert seam.chain_is_wired() is True, \
            "with this module imported, seam.chain_is_wired() must be True"
        note = "(seam present: chain_is_wired()==True)"
    else:
        note = "(seam absent here: env-marker path verified)"
    return f"C5 wired-detection contract holds {note}"


def _c6_forward_loud_guards():
    # a layer with no layer_idx must raise (we can't decide dense/MoE/bucket). The
    # layer_idx guard is the FIRST check after `import torch`; on a CPU box torch is
    # the cpu build so the import succeeds and the guard fires. If torch is genuinely
    # absent the import error is the honest signal (not a silent pass).
    class _NoIdx:
        pass
    try:
        import torch  # noqa: F401 — confirm torch is importable for this guard test
    except Exception:  # noqa: BLE001
        return "C6 SKIP (torch not importable here; guard logic exercised on-card)"
    raised = False
    try:
        run_chained_decoder_layer(_NoIdx(), None, _FakeT((1, 6144)), None,
                                  exec_cfg=_mk_fake_exec_cfg(has_dsa=True))
    except MMELChainError:
        raised = True
    assert raised, "no-layer_idx must raise MMELChainError"
    # the unbound MMEL_CHAIN_MOE_OP=1 path must refuse (raise), never mis-ship.
    r2 = False
    try:
        _moe_routed_via_our_op(object(), None,
                               exec_cfg=_mk_fake_exec_cfg(has_dsa=True),
                               sshape=StaticShape.from_exec_cfg(
                                   _mk_fake_exec_cfg(has_dsa=True), batch=1,
                                   context_len=128),
                               torch=None)
    except MMELChainError:
        r2 = True
    assert r2, "unbound routed-op swap must raise (refuse to ship an unbound op)"
    return "C6 run_chained_decoder_layer LOUD-guards (no layer_idx; unbound op-swap) correct"


def _c7_island_dirs_present():
    missing = [d for d in _CAP_DIRS
               if not os.path.isfile(os.path.join(d, "manifest.txt"))]
    if missing:
        # not fatal to the build (the recipes are captured on-card), but the gate
        # records it LOUD so the orchestrator knows the on-card capture is needed.
        return ("C7 NOTE: captured island manifest(s) absent: "
                + ", ".join(missing) + " (re-capture on-card per B1.1)")
    return f"C7 all {len(_CAP_DIRS)} captured island manifests present on disk"


def _c8_rawcs_scaffold_guards():
    """The raw-CS B5/G3 scaffold must FAIL LOUD off-card (no libSynapse), never
    silently no-op, and its struct layout must match the C harness ABI byte size."""
    import ctypes

    # off-card: building the ctypes surface raises MMELChainError (no libSynapse).
    on_card = any(os.path.isfile(p) for p in _LIBSYNAPSE_CANDIDATES)
    if not on_card:
        for fn in (_make_ctypes_syn_surface, reuse_bridge_device_and_stream):
            raised = False
            try:
                fn()
            except MMELChainError:
                raised = True
            assert raised, f"{fn.__name__} must raise MMELChainError off-card"
        # deserialize() on a missing path also raises (path check is pre-CDLL).
        r = RawCsIslandReplay(recipe_path="/nonexistent/island.recipe")
        raised = False
        try:
            r.deserialize()
        except MMELChainError:
            raised = True
        assert raised, "deserialize() of a missing recipe must raise"
        note = "(off-card: LOUD guards verified)"
    else:
        note = "(on-card: libSynapse present; full replay is the B5 step)"
    # struct byte-size sanity: name_ptr(8) + addr(8) + type(4 -> 8 aligned) +
    # 25*8 size + id(8) — assert the ctypes struct is the expected packed size when
    # buildable; off-card we can still build the struct class via a tiny shim.
    class _LTI(ctypes.Structure):
        _fields_ = [
            ("tensorName", ctypes.c_char_p),
            ("pTensorAddress", ctypes.c_uint64),
            ("tensorType", ctypes.c_int),
            ("tensorSize", ctypes.c_uint64 * _HABANA_DIM_MAX),
            ("tensorId", ctypes.c_uint64),
        ]
    # 8 + 8 + 8(int+pad) + 200 + 8 = 232 bytes on LP64.
    assert ctypes.sizeof(_LTI) == 232, \
        f"synLaunchTensorInfo ctypes size {ctypes.sizeof(_LTI)} != 232 (ABI drift)"
    return f"C8 raw-CS B5/G3 scaffold LOUD-guards + ABI struct size correct {note}"


def _c9_chain_layer_gate():
    """The S2/S3 MMEL_CHAIN_LAYER=1 gate: (a) the predicate is correct; (b) the
    lazy bridge to mmel_chain_layer resolves (the chain module imports clean
    card-free); (c) BOTH MMEL_CHAIN_LAYER=1 and MMEL_CHAIN_MOE_OP=2 set RAISES at
    the forward (mutually-exclusive). Card-free."""
    prev_layer = os.environ.get("MMEL_CHAIN_LAYER")
    prev_op = os.environ.get("MMEL_CHAIN_MOE_OP")
    try:
        # (a) predicate.
        for v, want in (("1", True), ("layer", True), ("0", False), ("", False)):
            if v == "":
                os.environ.pop("MMEL_CHAIN_LAYER", None)
            else:
                os.environ["MMEL_CHAIN_LAYER"] = v
            assert _chain_layer_enabled() is want, f"MMEL_CHAIN_LAYER={v!r} -> {want}"
        # (b) the chain-layer module imports + exports the entry (package or path).
        entry = _import_chain_layer_entry()
        assert entry is not None and callable(entry), \
            "mmel_chain_layer.run_decoder_layer_via_chain_replay must be importable"
        # (c) BOTH gates on => the forward raises (mutually-exclusive). Needs torch
        # for the forward's `import torch`; CPU torch satisfies it.
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return ("C9 MMEL_CHAIN_LAYER gate predicate + lazy-import OK "
                    "(mutual-exclusion raise exercised on-card; torch absent here)")
        os.environ["MMEL_CHAIN_LAYER"] = "1"
        os.environ["MMEL_CHAIN_MOE_OP"] = "2"

        class _Lyr:
            layer_idx = 5
        raised = False
        try:
            run_chained_decoder_layer(
                _Lyr(), None, _FakeT((1, 6144)), None,
                exec_cfg=_mk_fake_exec_cfg(has_dsa=True))
        except MMELChainError as e:
            raised = "mutually-exclusive" in str(e)
        assert raised, ("MMEL_CHAIN_LAYER=1 + MMEL_CHAIN_MOE_OP=2 must RAISE "
                        "(mutually-exclusive layer drivers)")
    finally:
        for k, p in (("MMEL_CHAIN_LAYER", prev_layer), ("MMEL_CHAIN_MOE_OP", prev_op)):
            if p is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = p
    return ("C9 MMEL_CHAIN_LAYER=1 gate correct (predicate + lazy-import of "
            "mmel_chain_layer + mutual-exclusion-with-MoE-op raise)")


def _c10_chain_static_gate():
    """The CORRECT static-tail gate (MMEL_CHAIN_STATIC=1): (a) the predicate is
    correct; (b) the sub-chain-B coverage bridge resolves card-free; (c) BOTH
    MMEL_CHAIN_STATIC=1 and MMEL_CHAIN_LAYER=1 set RAISES at the forward (mutually-
    exclusive; the static tail supersedes the unsound all-7). Card-free."""
    prev_static = os.environ.get("MMEL_CHAIN_STATIC")
    prev_layer = os.environ.get("MMEL_CHAIN_LAYER")
    try:
        # (a) predicate.
        for v, want in (("1", True), ("static", True), ("0", False), ("", False)):
            if v == "":
                os.environ.pop("MMEL_CHAIN_STATIC", None)
            else:
                os.environ["MMEL_CHAIN_STATIC"] = v
            assert _chain_static_enabled() is want, \
                f"MMEL_CHAIN_STATIC={v!r} -> {want}"
        # (b) the coverage bridge is callable + non-raising (undecidable => False).
        cov = _subchain_b_covers_bucket(_FakeT((1, 6144)))
        assert isinstance(cov, bool), "coverage bridge must return a bool"
        assert _subchain_b_covers_bucket(object()) is False, \
            "an unreadable hidden must yield False, never raise"
        # (c) BOTH gates on => the forward raises (mutually-exclusive). Needs torch.
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return ("C10 MMEL_CHAIN_STATIC gate predicate + coverage bridge OK "
                    "(mutual-exclusion raise exercised on-card; torch absent here)")
        os.environ["MMEL_CHAIN_STATIC"] = "1"
        os.environ["MMEL_CHAIN_LAYER"] = "1"

        class _Lyr:
            layer_idx = 5
        raised = False
        try:
            run_chained_decoder_layer(
                _Lyr(), None, _FakeT((1, 6144)), None,
                exec_cfg=_mk_fake_exec_cfg(has_dsa=True))
        except MMELChainError as e:
            raised = "mutually-exclusive" in str(e)
        assert raised, ("MMEL_CHAIN_STATIC=1 + MMEL_CHAIN_LAYER=1 must RAISE "
                        "(mutually-exclusive; prefer MMEL_CHAIN_STATIC)")
    finally:
        for k, p in (("MMEL_CHAIN_STATIC", prev_static),
                     ("MMEL_CHAIN_LAYER", prev_layer)):
            if p is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = p
    return ("C10 MMEL_CHAIN_STATIC=1 gate correct (predicate + coverage bridge + "
            "mutual-exclusion-with-MMEL_CHAIN_LAYER raise)")


def _c11_owned_all_layers_gate():
    """The #4b ALL-LAYER gate (MMEL_OWNED_ALL_LAYERS=1): (a) the predicate is correct;
    (b) the MMEL_BANZ_LAYER 1-layer pin logic is correct (pins OFF a non-matching
    layer, never pins a matching/unset/non-int one); (c) MMEL_OWNED_ALL_LAYERS=1
    WITHOUT MMEL_CHAIN_STATIC=1 RAISES at the forward (it is a static-tail widener,
    not a standalone driver). Card-free."""
    prev_owned = os.environ.get("MMEL_OWNED_ALL_LAYERS")
    prev_static = os.environ.get("MMEL_CHAIN_STATIC")
    prev_banz = os.environ.get("MMEL_BANZ_LAYER")
    try:
        # (a) predicate.
        for v, want in (("1", True), ("all", True), ("0", False), ("", False)):
            if v == "":
                os.environ.pop("MMEL_OWNED_ALL_LAYERS", None)
            else:
                os.environ["MMEL_OWNED_ALL_LAYERS"] = v
            assert owned_all_layers_enabled() is want, \
                f"MMEL_OWNED_ALL_LAYERS={v!r} -> {want}"
        # (b) the 1-layer pin: set to 3 -> pins OFF layer 5, NOT layer 3; unset / a
        # non-int value never pins (the all-via-island fallback / no pin in effect).
        os.environ["MMEL_BANZ_LAYER"] = "3"
        assert _banz_layer_pins_this_layer_off(5) is True, "BANZ=3 must pin OFF layer 5"
        assert _banz_layer_pins_this_layer_off(3) is False, "BANZ=3 must NOT pin layer 3"
        os.environ.pop("MMEL_BANZ_LAYER", None)
        assert _banz_layer_pins_this_layer_off(5) is False, "unset BANZ must not pin"
        os.environ["MMEL_BANZ_LAYER"] = "notanint"
        assert _banz_layer_pins_this_layer_off(5) is False, "non-int BANZ must not pin"
        os.environ.pop("MMEL_BANZ_LAYER", None)
        # (c) OWNED_ALL_LAYERS=1 WITHOUT CHAIN_STATIC=1 => the forward raises. Needs torch.
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return ("C11 MMEL_OWNED_ALL_LAYERS gate predicate + BANZ-pin logic OK "
                    "(requires-CHAIN_STATIC raise exercised on-card; torch absent here)")
        os.environ["MMEL_OWNED_ALL_LAYERS"] = "1"
        os.environ.pop("MMEL_CHAIN_STATIC", None)

        class _Lyr:
            layer_idx = 5
        raised = False
        try:
            run_chained_decoder_layer(
                _Lyr(), None, _FakeT((1, 6144)), None,
                exec_cfg=_mk_fake_exec_cfg(has_dsa=True))
        except MMELChainError as e:
            raised = "requires MMEL_CHAIN_STATIC=1" in str(e)
        assert raised, ("MMEL_OWNED_ALL_LAYERS=1 WITHOUT MMEL_CHAIN_STATIC=1 must RAISE "
                        "(it is a static-tail widener, not a standalone driver)")
    finally:
        for k, p in (("MMEL_OWNED_ALL_LAYERS", prev_owned),
                     ("MMEL_CHAIN_STATIC", prev_static),
                     ("MMEL_BANZ_LAYER", prev_banz)):
            if p is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = p
    return ("C11 MMEL_OWNED_ALL_LAYERS=1 gate correct (predicate + MMEL_BANZ_LAYER "
            "1-layer-pin logic + requires-MMEL_CHAIN_STATIC raise)")


def _c12_live_attn_seam_gate():
    """The LIVE-FED MLA attention seam (MMEL_CHAIN_LIVE=1): (a) the gate predicate is
    correct; (b) the coverage + live_ctx bridges + _try_live_attn NEVER raise and
    DELEGATE (None/False) off-card / on a fake layer (the 'undecidable => stock' rule);
    (c) a conflicting layer driver (MMEL_CHAIN_LAYER) makes _try_live_attn DEFER (None),
    never raise; (d) flag-unset => the seam is inert. Card-free."""
    global _CHAIN_LIVE_ENGAGE_LOGGED, _CHAIN_LIVE_DELEGATE_LOGGED, _CHAIN_LIVE_CONFLICT_LOGGED
    prev = {k: os.environ.get(k) for k in
            ("MMEL_CHAIN_LIVE", "MMEL_CHAIN_LAYER", "MMEL_CHAIN_STATIC", "MMEL_CHAIN_MOE_OP")}

    class _Lyr:
        layer_idx = 7

    try:
        # (a) predicate (1/live/on => on; 0/2/unset => off).
        for v, want in (("1", True), ("live", True), ("on", True),
                        ("0", False), ("2", False), ("", False)):
            if v == "":
                os.environ.pop("MMEL_CHAIN_LIVE", None)
            else:
                os.environ["MMEL_CHAIN_LIVE"] = v
            assert _chain_live_enabled() is want, f"MMEL_CHAIN_LIVE={v!r} -> {want}"
        # (b) the bridges are bool/None + NEVER raise (off-card / bad input).
        assert isinstance(_live_attn_covers_bucket(_FakeT((1, 6144))), bool), \
            "live_attn coverage bridge must return a bool"
        assert _live_attn_covers_bucket(object()) is False, \
            "coverage bridge must not raise on an unreadable shape"
        assert _build_live_attn_ctx(_Lyr(), None) is None, \
            "live_ctx must be None off-serve (no forward ctx / kv_cache), never raise"
        # (c) _try_live_attn: flag ON, fake layer (no captures / no forward ctx) =>
        #     DELEGATE (None), never raise; engage breadcrumb NOT set.
        os.environ["MMEL_CHAIN_LIVE"] = "1"
        for k in ("MMEL_CHAIN_LAYER", "MMEL_CHAIN_STATIC", "MMEL_CHAIN_MOE_OP"):
            os.environ.pop(k, None)
        _CHAIN_LIVE_ENGAGE_LOGGED = _CHAIN_LIVE_DELEGATE_LOGGED = _CHAIN_LIVE_CONFLICT_LOGGED = False
        r = _try_live_attn(_Lyr(), None, _FakeT((1, 6144)), None)
        assert r is None, "_try_live_attn must DELEGATE (None) when uncovered / no live_ctx"
        assert _CHAIN_LIVE_ENGAGE_LOGGED is False, "must NOT log ENGAGE when it delegated"
        # (d) conflicting layer driver => DEFER (None), never raise.
        os.environ["MMEL_CHAIN_LAYER"] = "1"
        _CHAIN_LIVE_CONFLICT_LOGGED = False
        r2 = _try_live_attn(_Lyr(), None, _FakeT((1, 6144)), None)
        assert r2 is None, "_try_live_attn must DEFER (None) when a conflicting driver is set"
        # (e) the o_proj all-reduce completer (RowParallel TP=8 reduction): EXISTS + off-
        #     serve (no vllm.distributed) it RAISES (LOUD — never a silent unreduced
        #     partial); _try_live_attn's except catches that => stock fallback (which all-
        #     reduces). On a single-rank serve it is a no-op; on TP>1 it sums the partials.
        assert callable(_oproj_all_reduce_partial), "o_proj all-reduce completer must exist"
        _ar_guarded = False
        try:
            _oproj_all_reduce_partial(_FakeT((1, 6144)))
        except Exception:  # noqa: BLE001 — off-serve: vllm.distributed absent => LOUD guard
            _ar_guarded = True
        assert _ar_guarded, ("_oproj_all_reduce_partial must LOUD-guard off-serve (no silent "
                             "skip of the TP all-reduce); _try_live_attn catches it => stock")
    finally:
        for k, p in prev.items():
            if p is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = p
        _CHAIN_LIVE_ENGAGE_LOGGED = _CHAIN_LIVE_DELEGATE_LOGGED = _CHAIN_LIVE_CONFLICT_LOGGED = False
    return ("C12 MMEL_CHAIN_LIVE seam correct (predicate + coverage/live_ctx bridges + "
            "_try_live_attn DELEGATE-on-miss + DEFER-on-conflict + o_proj all-reduce "
            "completer LOUD-guards off-serve; all NEVER-RAISE => stock)")


def _c13_owned_backend_gate():
    """C13 — the MMEL_BACKEND selector: default-off, hpu/cpu accepted, unknown LOUD,
    and the one-sync default. Card-free (env-only; no device touched)."""
    keep = os.environ.get("MMEL_BACKEND")
    try:
        os.environ.pop("MMEL_BACKEND", None)
        assert _owned_backend_name() == "" and _owned_backend_enabled() is False
        os.environ["MMEL_BACKEND"] = "hpu"
        assert _owned_backend_name() == "hpu" and _owned_backend_enabled() is True
        os.environ["MMEL_BACKEND"] = "cpu"
        assert _owned_backend_name() == "cpu" and _owned_backend_enabled() is False
        os.environ["MMEL_BACKEND"] = "banana"
        raised = False
        try:
            _owned_backend_name()
        except MMELChainError:
            raised = True
        assert raised, "unknown MMEL_BACKEND must raise LOUD"
        # default one-sync ON; per-layer override honored.
        os.environ.pop("MMEL_OWNED_PERLAYER_SYNC", None)
        assert _owned_one_sync() is True
        os.environ["MMEL_OWNED_PERLAYER_SYNC"] = "1"
        assert _owned_one_sync() is False
    finally:
        os.environ.pop("MMEL_OWNED_PERLAYER_SYNC", None)
        if keep is None:
            os.environ.pop("MMEL_BACKEND", None)
        else:
            os.environ["MMEL_BACKEND"] = keep
    return "C13 MMEL_BACKEND selector: default-off, hpu/cpu ok, unknown LOUD, one-sync default"


def _c14_name_bridge():
    """C14 — the tag -> recipe-positional-name bridge (§5.B, the make-or-break). PURE /
    card-free: parse the REAL cap_routed/manifest.txt (positional names 0..57) and assert
    the six int4 expert classes map to the right position ranges by BYTE SIZE (H=6144,
    I=2048, GS=128), the same byte-size disambiguation mla_live_ctx_roles uses. Also
    asserts _parse_recipe_manifest reads the dims header + that a many-position mapping
    (the per-expert FROZEN case) is flagged by _recipe_tensor_name (not silently bound)."""
    path = os.path.join("$HOME/AI/mmel/cap_routed", "manifest.txt")
    if not os.path.isfile(path):
        return "C14 SKIP (cap_routed/manifest.txt absent; on-card capture pending)"
    header, rows = _parse_recipe_manifest(path)
    assert (header.get("H"), header.get("I"), header.get("GS")) == (6144, 2048, 128), \
        f"manifest dims header must be H=6144 I=2048 GS=128, got {header}"
    assert len(rows) == 58, f"cap_routed manifest must list 58 tensors (0..57), got {len(rows)}"
    tagmap = _manifest_tag_positions(path)
    want = {
        "exp_c13": [str(i) for i in range(1, 9)],    # w13 packed  (12582912 B)
        "exp_s13": [str(i) for i in range(9, 17)],   # w13 scale   (393216 B)
        "exp_z13": [str(i) for i in range(17, 25)],  # w13 zp      (98304 B)
        "exp_c2":  [str(i) for i in range(25, 33)],  # w2 packed   (6291456 B)
        "exp_s2":  [str(i) for i in range(33, 41)],  # w2 scale    (196608 B)
        "exp_z2":  [str(i) for i in range(41, 49)],  # w2 zp       (49152 B)
    }
    for tag, names in want.items():
        assert tagmap.get(tag) == names, \
            f"bridge {tag} -> {tagmap.get(tag)} != expected {names}"
    # the hidden IN (pos 0) and OUT (pos 57) are the size-degenerate spine tensors.
    assert tagmap.get("layer_in0") == ["0"], f"spine in0 -> {tagmap.get('layer_in0')}"
    assert tagmap.get("layer_out") == ["57"], f"layer_out -> {tagmap.get('layer_out')}"
    # a per-expert-sliced tag (8 positions) MUST be refused by the translator (the frozen
    # cap_routed is not adoptable by a single stacked resident VA).
    chain = _OwnedHpuChain.__new__(_OwnedHpuChain)     # no __init__ (no card touched)
    refused = False
    try:
        _OwnedHpuChain._recipe_tensor_name(chain, tagmap, "exp_c13", 40)
    except MMELChainError as e:
        refused = "per-expert-sliced" in str(e) or "NOT adoptable" in str(e)
    assert refused, "a many-position (frozen per-expert) tag must be REFUSED, not bound"
    # a single-position tag translates cleanly.
    assert _OwnedHpuChain._recipe_tensor_name(chain, tagmap, "layer_out", 40) == "57"
    # --- BAKED (const-folded) handling: a tag with NO recipe tensor => None (SKIP, not raise) ---
    assert _OwnedHpuChain._recipe_tensor_name(chain, {"exp_c13": ["1"]}, "Wgate", 40) is None, \
        "a baked tag (no recipe launch tensor) must return None (skip), never raise"
    assert _OwnedHpuChain._recipe_tensor_name(chain, {"exp_c13": ["1"]}, "exp_c13", 40) == "1", \
        "a matched single tag must still bind"
    # bake classification: router/norm are TP-replicated (bake CORRECT); shared expert is
    # SHARDED (bake uses rank-0 values -> WRONG on ranks>0).
    assert _OwnedHpuChain._bake_class(chain, "Wgate") == "replicated"
    assert _OwnedHpuChain._bake_class(chain, "w_post") == "replicated"
    assert _OwnedHpuChain._bake_class(chain, "sh_gu") == "sharded"
    assert _OwnedHpuChain._bake_class(chain, "sh_d") == "sharded"
    # pure pairing on the REAL on-card collision shape: experts bind, replicated weights +
    # the sharded shared-expert bake (omitted); the degenerate exp_s2/sh_gu bucket binds
    # exp_s2 (priority) and BAKES sh_gu; the spine binds attn_out/residual and BAKES w_post.
    lb = {(6291456, 2, 1): ["exp_s2", "sh_gu"], (3145728, 2, 1): ["Wgate", "sh_d"],
          (12288, 2, 1): ["attn_out", "residual", "w_post"], (402653184, 4, 1): ["exp_c13"]}
    rb = {(6291456, 2, 1): ["A"], (12288, 2, 1): ["B", "C"], (402653184, 4, 1): ["D"]}
    paired, _ordd = _pair_tags_to_recipe(lb, rb, _OwnedHpuChain._DEGEN_ORDER)
    assert paired == {"exp_s2": ["A"], "attn_out": ["B"], "residual": ["C"],
                      "exp_c13": ["D"]}, paired
    for baked_tag in ("sh_gu", "sh_d", "Wgate", "w_post"):
        assert baked_tag not in paired, f"{baked_tag} must be BAKED (omitted from bind)"
    # FULL LAUNCH SET accounting (the on-card status=30 fix): the recipe needs EVERY launch
    # tensor bound. matched live-override names + resident (unmatched) names must EQUAL the
    # recipe's full tensor set with NO gap and NO overlap (== the _rebind count guard).
    recipe_names = ["e0", "e1", "e2", "e3", "e4", "e5",   # 6 int4 experts (live)
                    "aout", "resid", "hid",               # attn_out/residual (live) + OUT
                    "c0", "c1", "c2", "c3", "c4",          # 5 control tensors (resident zeros)
                    "os0", "os1", "os2", "os3"]            # 4 output scratch (resident zeros)
    matched_names = {"e0", "e1", "e2", "e3", "e4", "e5", "aout", "resid", "hid"}  # live-override
    resident_names = [n for n in recipe_names if n not in matched_names]           # resident
    full = list(matched_names) + resident_names
    assert len(full) == len(recipe_names) and set(full) == set(recipe_names), \
        "FULL launch set (live-override + resident) must == the recipe tensor set (no gap)"
    assert not (matched_names & set(resident_names)), "no tensor may be BOTH live + resident"
    # --- SELECT-BY-SIGNATURE matcher (the door-A fix), on synthetic launch-tensor lists ---
    E, Hh, Ii = 256, 6144, 2048
    def _t(nbytes, nelem, is_input=1):
        return {"nbytes": nbytes, "nelem": nelem, "is_input": is_input}
    # stacked int4 MoE tail: largest launch tensor is a big int32 (packed exp_c13/exp_c2).
    tail = [_t(E * Hh * 512 * 4, E * Hh * 512),        # exp_c13 (int32, width 4, largest)
            _t(E * Ii * 768 * 4, E * Ii * 768),        # exp_c2  (int32, width 4)
            _t(E * 48 * 4096 * 2, E * 48 * 4096),      # exp_s13 (bf16, width 2)
            _t(Hh * 2, Hh),                            # w_post  (bf16, width 2)
            _t(Hh * 2, Hh, is_input=0)]                # OUT
    assert _is_moe_tail_signature(tail) is True, "stacked int4 tail must match the signature"
    # dense tail: largest is a bf16 GEMM weight (width 2) -> NOT the MoE-tail signature.
    dense = [_t(2 * 12288 * Hh * 2, 2 * 12288 * Hh),   # Wgu (bf16, width 2, largest)
             _t(Hh * 12288 * 2, Hh * 12288),           # Wd  (bf16, width 2)
             _t(Hh * 2, Hh)]
    assert _is_moe_tail_signature(dense) is False, "dense tail (bf16-largest) must NOT match"
    # attention / small recipe: no large int32 -> no match.
    assert _is_moe_tail_signature([_t(Hh * 2, Hh), _t(Hh * Hh * 2, Hh * Hh)]) is False, \
        "a recipe with no large int32 packed weight must NOT match the tail signature"
    assert _is_moe_tail_signature([]) is False, "empty tensor list must not match"
    # --- M=1 DECODE-tail vs PROMPT-tail disambiguation, on the REAL on-card dumped sizes
    #     (per-rank TP=8; the blob 2798211919003550437 was the 18-tensor M=1 decode tail).
    #     Keyed on (nbytes,width,is_input) — NO ndim (the recipe reader gives ndim=0). ---
    def _td(nbytes, width, is_input=1):
        return {"nbytes": nbytes, "nelem": nbytes // width, "is_input": is_input}
    # the shared M-independent WEIGHTS present in EVERY bucket (real dumped per-rank sizes):
    common = [_td(402653184, 4), _td(201326592, 4),   # exp_c13, exp_c2  (int32 packed)
              _td(12582912, 2), _td(6291456, 2),       # exp_s13, exp_s2  (bf16 scales)
              _td(3145728, 4), _td(1572864, 4),        # exp_z13, exp_z2  (int32 zp)
              _td(3145728, 2), _td(6291456, 2),        # Wgate, sh_gu     (bf16; collide!)
              _td(3145728, 2), _td(12288, 2)]          # sh_d, w_post     (bf16)
    common += [_td(4096, 4), _td(1024, 4), _td(2048, 4), _td(512, 4),
               _td(256, 4)]                          # misc small (all buckets) -> common=15
    # M=1 decode tail (18): + [1,H] activations 12288 B x3 (attn_out/residual/OUT).
    m1 = common + [_td(12288, 2), _td(12288, 2), _td(12288, 2, is_input=0)]
    # prompt tails: [S,6144] activations (S=128 -> 1572864; S=256 -> 3145728) + fillers.
    p128 = common + [_td(1572864, 2), _td(1572864, 2), _td(1572864, 2, is_input=0)] \
        + [_td(49152, 2)] * 16
    p256 = common + [_td(3145728, 2), _td(3145728, 2), _td(3145728, 2, is_input=0)] \
        + [_td(98304, 2)] * 16
    pbig = common + [_td(3145728, 2), _td(3145728, 2), _td(3145728, 2, is_input=0)] \
        + [_td(98304, 2)] * 17
    labels = {id(m1): "2798211919003550437(M=1)", id(p128): "prompt-S128",
              id(p256): "prompt-S256", id(pbig): "prompt-big"}
    assert (len(m1), len(p128), len(p256), len(pbig)) == (18, 34, 34, 35), \
        "fixture must mirror the on-card counts (M=1=18, prompts=34/34/35)"
    # RULE: FEWEST tensors (tiebreak total bytes). M=1 (18) wins in every ordering.
    for order in ([m1, p128, p256, pbig], [pbig, p256, p128, m1], [p256, m1, pbig, p128]):
        pick, why = _pick_m1_decode_tail(order)
        assert pick is not None and order[pick] is m1, \
            (f"must pick the M=1 decode tail 2798211919003550437 (fewest tensors); got "
             f"{labels.get(id(order[pick])) if pick is not None else None} ({why})")
    # single candidate: returned as-is (no disambiguation needed).
    assert _pick_m1_decode_tail([m1])[0] == 0
    # EXACT tie on (n_tensors, total_bytes) => LOUD-undecidable (not a guess).
    none_pick, _ = _pick_m1_decode_tail([m1, [dict(t) for t in m1]])
    assert none_pick is None, "identical (count,bytes) candidates must be LOUD, not a guess"
    return ("C14 bridge + SELECT-BY-SIGNATURE + M=1-vs-prompt disambig correct (cap_routed "
            "exp_* -> 1-8/../41-48; frozen per-expert REFUSED; signature matches stacked-"
            "int4 tail not dense/attn/empty; M=1 decode tail 2798211919003550437 picked by "
            "FEWEST-tensors 18 vs 34/35; exact (count,bytes) tie LOUD)")


def _c15_static_tail_bind_plan():
    """C15 — the static-tail bind-plan is EXACTLY the 13 live externals (the EXTERNAL=yes
    set) and the REGISTER-ONLY workspace is EXCLUDED (the bridge-owns-HBM / status=26 fix).
    Card-free: drives the REAL _record_layer_structure (static-tail branch) with fake live
    tensors + the off-card HpuBackend (alloc_symbolic + _register + _launch_tensors are all
    PURE), then asserts the bind-plan tags. Confirms the workspace fix keeps the bind
    contract intact and no device alloc is needed to build the plan."""
    try:
        _eng, hpu = _owned_import_engine()
    except Exception as e:  # noqa: BLE001 — owned engine not on this box: skip (card-only run)
        return f"C15 SKIP (owned engine not importable off-card: {type(e).__name__})"
    import numpy as _np

    class _FT:
        def __init__(self, shape, eb, ptr):
            self.shape = tuple(shape); self._eb = eb; self._ptr = ptr

        def numel(self):
            n = 1
            for s in self.shape:
                n *= s
            return n

        def element_size(self):
            return self._eb

        def dim(self):
            return len(self.shape)

        def is_contiguous(self):
            return True

        def contiguous(self):
            return self

        def data_ptr(self):
            return self._ptr

    class _Spec:
        hidden = 6144; n_shared = 1; n_routed_experts = 256
        moe_inter = 2048; group_size = 128; top_k = 8; attn_mode = "MLA"

        def layer_is_dense(self, idx):
            return False

    class _Eng:
        F64 = _np.float64; ATTN_MLA = "MLA"

    chain = _OwnedHpuChain.__new__(_OwnedHpuChain)   # no __init__ (no card touched)
    chain.hbe = hpu.HpuBackend()                     # off-card; alloc_symbolic/_register pure
    chain.spec = _Spec(); chain._eng = _Eng()
    _ptr = [0x1000]

    def _mk(shape, eb):
        _ptr[0] += 0x1000
        return _FT(shape, eb, _ptr[0])

    live = {
        "hidden": _mk((1, 6144), 2), "residual": _mk((1, 6144), 2),
        "attn_out": _mk((1, 6144), 2), "__attn_stock__": True,
        "w_in": _mk((6144,), 2), "w_post": _mk((6144,), 2), "Wgate": _mk((256, 6144), 2),
        "exp_c13": _mk((256, 6144, 512), 4), "exp_s13": _mk((256, 48, 4096), 2),
        "exp_z13": _mk((256, 48, 512), 4), "exp_c2": _mk((256, 2048, 768), 4),
        "exp_s2": _mk((256, 16, 6144), 2), "exp_z2": _mk((256, 16, 768), 4),
        "sh_gu": _mk((4096, 6144), 2), "sh_d": _mk((6144, 2048), 2),
    }
    ctx = hpu.CaptureCtx(chain.hbe, "tail40")
    chain._record_layer_structure(ctx, 40, live)     # the REAL static-tail structure
    plan = dict(hpu.HpuBackend._launch_tensors(hpu.Recipe(name="tail40", ops=ctx.ops)))
    want = {"attn_out", "residual", "w_post", "Wgate", "exp_c13", "exp_s13", "exp_z13",
            "exp_c2", "exp_s2", "exp_z2", "sh_gu", "sh_d", "hidden"}
    assert set(plan) == want, f"bind-plan {sorted(plan)} != the 13 externals {sorted(want)}"
    for ws in ("normed40", "attn40", "normed240", "w_in"):
        assert ws not in plan, f"{ws} (register-only workspace / unused) must be EXCLUDED"
    return (f"C15 static-tail bind-plan is EXACTLY the 13 live externals ({len(plan)} tags: "
            f"attn_out/residual/w_post/Wgate/exp_{{c13,s13,z13,c2,s2,z2}}/sh_{{gu,d}}/hidden);"
            f" register-only workspace + unused w_in excluded (no device alloc to build it)")


class _FakeT:
    """A torch-tensor-shaped stand-in for the card-free guard tests (shape/dim only)."""

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


def _selftest() -> int:
    tests = [
        _c1_arith,
        _c2_static_shape,
        _c3_route_fold,
        _c4_op_paths,
        _c4b_routed_optional_in_baseline,
        _c5_wired_contract,
        _c6_forward_loud_guards,
        _c7_island_dirs_present,
        _c8_rawcs_scaffold_guards,
        _c9_chain_layer_gate,
        _c10_chain_static_gate,
        _c11_owned_all_layers_gate,
        _c12_live_attn_seam_gate,
        _c13_owned_backend_gate,
        _c14_name_bridge,
        _c15_static_tail_bind_plan,
    ]
    ok = 0
    for t in tests:
        try:
            msg = t()
            print(f"  PASS  {msg}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\nmmel_l6_chain self-test: {ok}/{len(tests)} passed")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
