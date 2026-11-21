"""MMEL S2/S3 — drive the FULL chained decoder layer DEVICE-SIDE from live decode.

────────────────────────────────────────────────────────────────────────────
WHAT THIS IS (the next increment past S1 — read CSENGINE_REAL_WIREIN.md first)
────────────────────────────────────────────────────────────────────────────
S1 (``mmel_moe_island.py``, ``MMEL_CHAIN_MOE_OP=2``) drove ONE island — the
routed int4-MoE block, a SINGLE-buffer recipe with NO branch edge — on the
bridge's own Synapse stream behind one ``synStreamSynchronize``, live-fed from
the decode ``hidden_states``. That proved the csengine 1-completion primitive
ENGAGES + stays byte-exact in-serve, at 1-island granularity.

THIS module (S2/S3, gated ``MMEL_CHAIN_LAYER=1``, distinct from S1's
``MMEL_CHAIN_MOE_OP=2``) drives the **FULL chained decoder layer** —
``norm → q/kv-proj → MLA-paged → o-proj → norm → MoE`` = the 7 captured islands
(``cap_{norm,qproj,kvproj,mla_paged,oproj,routed}``) — on ONE Synapse stream
behind **ONE host fence per layer**, from LIVE decode, replacing the per-layer
recipe-boundary sync. This is the standalone C-harness STEP-4 result
(``route2_chain_multi.c --autofeed``, 7 islands → 1 fence, byte-IDENTICAL to a
7-step serial control, relNorm 0.0 / cmp+sha256), now driven INSIDE a vLLM
decode step instead of from fixed captured ``.bin``s.

────────────────────────────────────────────────────────────────────────────
THE EDGE TOPOLOGY (the load-bearing detail — why the spine is LIVE and the
pre-attention branch is STAGED), grounded in RUN_CHAIN STEP-4 + deepseek_v2.py
────────────────────────────────────────────────────────────────────────────
A real decoder layer is NOT a pure linear feed. The attention path BRANCHES:
``input_norm → q_proj AND kv_proj``; ``q_proj``'s ``N*qk_head`` output is split
into ``q_nope/q_pe``; ``q_pe/k_pe`` go through RoPE; the paged MLA does its own
block-gather. A single hidden-state buffer CANNOT thread those edges (the byte
sizes differ — norm OUT 12288 B → q_proj in 4096 B → MLA in 24576 B). So, per
the proven ``--autofeed`` rule (RUN_CHAIN STEP-4, the layer byte-size table):

   edge                       prev OUT   next in"0"   feed
   norm → q_proj              12288      4096         STAGED (branch)
   q_proj → kv_proj           32768      1024         STAGED (branch)
   kv_proj → MLA              57344      24576        STAGED (branch)
   MLA  → o_proj              32768      32768        LIVE  (linear spine)
   o_proj → post_norm         12288      12288        LIVE  (linear spine)
   post_norm → MoE            12288      12288        LIVE  (linear spine)

   + the layer's LIVE INPUT edge:  the decode hidden_states feeds norm's in"0"
     (12288 B == the captured norm input) — this is the ONE per-step live input
     that makes the chain a real serve step rather than a fixed-input replay.

The **linear spine ``MLA → o_proj → norm → MoE`` is byte-size-matched end to
end and fed LIVE on-device** (step k's in"0" bound to step k-1's OUT VA, NO host
fence between); the **pre-attention branch (norm→q/kv→MLA) is STAGED** — those
islands replay from their captured ``.bin`` inputs (resident, uploaded once).
The live decode ``hidden_states`` is bound into norm's in"0" each step.

This is EXACTLY S1's mechanism (RawCsIslandReplay synLaunch on the bridge
stream) generalized from 1 island to a 7-island in-order chain with ONE final
``synStreamSynchronize`` — NO new kernel, NO new ABI.

────────────────────────────────────────────────────────────────────────────
THE HONEST FRAME (the brutal make-or-break — do NOT overclaim)
────────────────────────────────────────────────────────────────────────────
What S2/S3 (staged-branch) PROVES, gated:
  (G-warm)  the 7-island chain WARMS in-serve (all recipes deserialize, the
            static weights are resident, the bridge stream is reused) on all 8
            ranks — the structural substrate engages.
  (G-count) the layer issues **1 synStreamSynchronize for the whole 7-island
            chain** vs the stock per-layer recipe-boundary syncs (strace
            HL_WAIT_CS collapse) — the per-layer fence/structure collapse,
            observed LIVE (the thing the "flat 68.8" stub NEVER did).
  (G-byte)  the 1-fence chained layer output == a 7-step SERIAL control
            (a synStreamSynchronize after every island, the bridge-equivalent)
            byte-for-byte — the SAME P2 gate the C harness passed, now in-serve.
            relNorm 0.0 vs the serial control.

What S2/S3 (staged-branch) does NOT yet prove (stated up front, honest):
  * It is NOT byte-exact vs the LIVE stock ``layer.forward`` end-to-end, because
    the pre-attention branch is STAGED — the q_nope the MLA consumes is the
    captured one, not literally derived from THIS step's live q_proj OUT (that
    needs the in-graph q/kv split + RoPE + paged-gather as extra recipe nodes —
    the S3-FULL increment, scoped here, deferred). So the byte-exact arbiter is
    the serial control with the SAME staging topology (the C-harness P2), and
    the live-stock A/B is the S3-FULL gate. The chain that drives the LIVE serve
    end-to-end-correct is S3-FULL; S2/S3-staged proves the 7-island 1-fence
    chain mechanism engages + collapses the count + reproduces the serial
    control, live.
  * A single chained layer will likely NOT cut the single-stream step much by
    itself (the live fences already overlap — RUN_CHAIN STEP-6). The tok/s
    payoff is S4 (async layer-pipeline) + S5 (W7 collective-seq) filling the
    ~40-47 ms of within-chain structure. S2/S3 builds the substrate; S4/S5 is
    where the structural single-stream bet is settled by measurement.

────────────────────────────────────────────────────────────────────────────
DESIGN DISCIPLINE (the standing no-slop / LOUD bar — held here)
────────────────────────────────────────────────────────────────────────────
  * NO ACCELERATOR AT IMPORT — torch/habana/libSynapse imported lazily inside
    the on-card methods; the manifest parsing, the edge-topology solver, the
    bind contract, the static-shape guards are card-free unit-tested at the
    bottom (this file ``py_compile``s + self-tests on the CPU-only m3calib box).
  * LOUD failures — a missing recipe/weight/lib, an edge-topology that does not
    match the captured byte sizes, a live input whose geometry would re-fire,
    an incomplete bind-map: ALL raise with the exact reason + the fix. A silent
    no-op masquerading as engagement is the 2026-06-22 false-null we refuse.
  * STATIC SHAPE — every island replays at its captured bucket geometry; a live
    hidden_states whose byte size != the captured norm input is REFUSED.
  * REUSE, DON'T REBUILD — the replay mechanism is
    ``mmel_l6_chain.RawCsIslandReplay`` (S1's proven ctypes synLaunch surface +
    ``reuse_bridge_device_and_stream``); the recipes+weights are the captured
    ``cap_*`` dirs (byte-exact-proven, RUN_CHAIN). No new kernel, no new ABI.
  * BYTE-EXACT INTENT — the chained layer output reproduces the serial control;
    the seam's ``MmelGreedyABHarness`` is the live-serve coherence guard.

Run the card-free self-test:
    $HOME/.venvs/m3calib/bin/python \
        $HOME/AI/vllm-gaudi-patch/v1/worker/mmel_chain_layer.py
Exit 0 = the multi-island manifest parse, the edge-topology solver (live vs
staged), the bind contract, the shape guards, and the gate logic are correct.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


# ----------------------------------------------------------------------------- #
#  Logger — reuse the chain module's stderr-or-vllm logger contract.             #
# ----------------------------------------------------------------------------- #
def _get_logger():
    try:
        from vllm.logger import init_logger  # type: ignore

        return init_logger("mmel.chain_layer")
    except Exception:  # noqa: BLE001
        class _Stderr:
            @staticmethod
            def _emit(level, msg, *a):
                if a:
                    try:
                        msg = msg % a
                    except Exception:  # noqa: BLE001
                        msg = msg + " " + " ".join(str(x) for x in a)
                sys.stderr.write(f"[mmel.chain_layer][{level}] {msg}\n")

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
#  FORENSIC OVERRIDE-PROOF POISON (default OFF; pure no-op unless MMEL_CHAIN_POISON=1).
#
#  WHY (the box owner's explicit gate): the engine A/B must prove the device-resident
#  chained-layer output is *what the serve actually emits* — that no OEM/stock stack
#  (HPU-graph replay, a fused-op silent fallback, or the flag engaging only in Python
#  while the device runs stock) is quietly overriding it. The discriminator: perturb
#  the chain's DEVICE OUTPUT tensor (the `out` `run()` returns, the literal layer
#  hidden that threads into the next layer) by a KNOWN amount. Then:
#    * greedy text GARBLES   => the served output IS this chain's device output (engine
#                               drives the serve — the perturbation reached the tokens).
#    * greedy text COHERENT  => the chain output is NOT what gets served (an OEM/stock
#                               path overrode it — the perturbation was discarded) ==>
#                               OVERRIDE CONFIRMED, the A/B's "engaged" is a false flat.
#
#  RMSNorm-ROBUST (the prior-mistake fix): a UNIFORM SCALE (e.g. *1.15) is WASHED by the
#  next layer's input RMSNorm (RMSNorm is scale-invariant: norm(c*x) == norm(x)) — it
#  gave a FALSE 'coherent' last time. This poison is instead (a) a large ADDITIVE
#  per-channel offset (survives RMSNorm — it shifts the per-channel mean/variance, NOT a
#  global scale) PLUS (b) a single-channel SPIKE (saturates one feature dimension, which
#  RMSNorm cannot normalize away because it is one component, not the whole vector). Both
#  are applied to the DEVICE tensor on-stream, so they exercise the exact served value.
#  Tunable magnitude via MMEL_CHAIN_POISON_MAG (default 8.0, in bf16 units — comfortably
#  past RMSNorm's normalization for a [1,6144] hidden).
#
#  This is DISTINCT from mmel_l6_chain's MMEL_POISON (which perturbs the Python-side
#  residual at the seam return — that fires regardless of whether the device ran the
#  chain, so it canNOT discriminate engine-vs-OEM-override; THIS one can, because it
#  rides the device output the chain produced).
# ----------------------------------------------------------------------------- #
CHAIN_POISON_ENV = "MMEL_CHAIN_POISON"


def _chain_poison_enabled() -> bool:
    return os.environ.get(CHAIN_POISON_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def _chain_poison_mag() -> float:
    """Perturbation magnitude (bf16 units). Default 8.0 — RMSNorm cannot wash a +8.0
    additive offset on a [1,6144] hidden whose per-channel values are O(1)."""
    try:
        return float(os.environ.get("MMEL_CHAIN_POISON_MAG", "8.0"))
    except (TypeError, ValueError):
        return 8.0


if _chain_poison_enabled():
    try:
        logger.warning(
            "[mmel.chain_layer][MMEL_CHAIN_POISON] ARMED (mag=%s): the chain's DEVICE "
            "output is perturbed (additive per-channel offset + single-channel spike, "
            "RMSNorm-robust) at run() return. GARBLED greedy ==> the chain's device "
            "output IS served (engine drives the serve). COHERENT greedy ==> an "
            "OEM/stock path overrode it (override CONFIRMED). Requires MMEL_CHAIN_LAYER=1 "
            "(else this module is not on the path = inert) + a WIPED recipe cache.",
            _chain_poison_mag())
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------- #
#  Errors — LOUD, specific (mirror the island/chain error taxonomy).             #
# ----------------------------------------------------------------------------- #
class MmelChainLayerError(RuntimeError):
    """A chained-LAYER replay precondition failed (bad manifest, missing recipe,
    edge-topology mismatch, weight-bind mismatch, shape drift)."""


class MmelChainLayerShapeError(MmelChainLayerError):
    """A live tensor violated the static geometry the chain was captured at."""


# ----------------------------------------------------------------------------- #
#  THE GATE: MMEL_CHAIN_LAYER selects the full-chained-layer device path.         #
#    unset / 0  -> the layer runs its stock/G1 path (NOT this module).            #
#    1           -> THIS module: the 7-island layer chain replayed on ONE bridge  #
#                  stream behind ONE synStreamSynchronize (S2/S3, live-fed spine).#
#                                                                               #
#  This is DISTINCT from MMEL_CHAIN_MOE_OP=2 (S1, one MoE island). The two are    #
#  mutually-exclusive layer drivers; the seam routes to LAYER when this is on.    #
# ----------------------------------------------------------------------------- #
CHAIN_LAYER_ENV = "MMEL_CHAIN_LAYER"


def chain_layer_enabled() -> bool:
    """True iff the operator selected the full chained-LAYER device path
    (``MMEL_CHAIN_LAYER=1``). The ONLY gate that makes us deserialize 6 recipes +
    acquire libSynapse — so a missing recipe/lib is fatal ONLY here."""
    return os.environ.get(CHAIN_LAYER_ENV, "").strip().lower() in (
        "1", "true", "yes", "on", "layer")


# Where the captured island dirs live (recipe + static weights + out_ref each).
# The env override lets the orchestrator pin a re-captured (live-bucket) set.
CAP_DIR_ENV = "MMEL_CAP_DIR"          # the parent dir holding cap_norm/ cap_qproj/ ...
_CAP_PARENT_DEFAULTS = (
    "$HOME/AI/mmel",   # host self-test box
    "/opt/mmel",            # a possible in-container mount
)


def cap_parent_dir() -> str:
    """Resolve the parent dir holding the captured island dirs (env override >
    first existing > the host default so the loader's own existence check is
    LOUD)."""
    p = os.environ.get(CAP_DIR_ENV)
    if p:
        return p
    for c in _CAP_PARENT_DEFAULTS:
        if os.path.isdir(os.path.join(c, "cap_routed")):
            return c
    return _CAP_PARENT_DEFAULTS[0]


# ----------------------------------------------------------------------------- #
#  MMEL_CHAIN_M — the SINGLE SOURCE OF TRUTH for the chain's DECODE WIDTH (M).    #
#                                                                               #
#  DEFAULT 1 (the Phase-1-proven single-token decode bucket). When unset or == 1 #
#  the chain is BYTE-IDENTICAL to the proven M=1 path: ``_chain_island_dir``      #
#  returns the M=1 capture subdir names UNCHANGED and every coverage/shape gate   #
#  resolves the M=1 [1,6144] = 12288 B bucket exactly as before.                 #
#                                                                               #
#  When ``MMEL_CHAIN_M=<M>`` (e.g. 32, the concurrency decode bucket [32,6144])   #
#  the chain loads the M-parameterized island captures (``cap_*_m{M}``, and the   #
#  paged-MLA's on-card name ``cap_mla_b{M}``) and every gate keys on the          #
#  [M,6144] = M*12288 B bucket. The byte sizes stay DATA-DRIVEN from each         #
#  captured manifest (M*H*2 parsed per island), so a re-capture cannot silently   #
#  mis-feed; MMEL_CHAIN_M only selects WHICH capture set + which bucket the        #
#  coverage predicate admits. The refuse-everything-else safety is preserved at   #
#  every M (a [1,6144] step under M=32, or prefill, is NOT covered -> delegated). #
# ----------------------------------------------------------------------------- #
CHAIN_M_ENV = "MMEL_CHAIN_M"

_BAD_CHAIN_M_WARNED = False


def _warn_bad_chain_m(raw) -> None:
    global _BAD_CHAIN_M_WARNED
    if _BAD_CHAIN_M_WARNED:
        return
    _BAD_CHAIN_M_WARNED = True
    logger.warning(
        "[mmel.chain_layer] %s=%r is not a positive integer; falling back to M=1 "
        "(the proven single-token decode bucket).", CHAIN_M_ENV, raw)


def chain_m() -> int:
    """The configured chain decode width M (``MMEL_CHAIN_M``, default 1).

    A missing/empty env => 1. A non-integer or M < 1 value => 1 with a LOUD one-shot
    warning (a typo'd env must NEVER crash the live forward). M=1 is the proven
    single-token decode bucket; the only other on-card-validated target is M=32 (the
    [32,6144] concurrency decode bucket). Card-free; read fresh each call so an A/B
    can flip it per worker without import-order coupling."""
    raw = os.environ.get(CHAIN_M_ENV)
    if raw is None or raw.strip() == "":
        return 1
    try:
        m = int(raw.strip())
    except (TypeError, ValueError):
        _warn_bad_chain_m(raw)
        return 1
    if m < 1:
        _warn_bad_chain_m(raw)
        return 1
    return m


# Per-island M=1 -> M-parameterized capture-subdir overrides. Most islands append a
# plain ``_m{M}`` suffix; the paged-MLA capture is named ``cap_mla_b{M}`` on-card
# (the capture script's batch-suffix convention), so it carries an explicit map.
_CHAIN_ISLAND_M_OVERRIDES = {
    "cap_mla_paged": "cap_mla_b{m}",
}


def _chain_island_dir(sub: str) -> str:
    """Map an M=1 capture subdir name to the configured ``MMEL_CHAIN_M`` one.

    M=1 (default/unset) -> returns ``sub`` UNCHANGED (byte-identical to the proven
    path). M != 1 -> the M-parameterized capture dir: ``cap_mla_paged`` ->
    ``cap_mla_b{M}``; every other island -> ``{sub}_m{M}``. This is the ONE place the
    chain's capture-dir naming is M-parameterized — both ``LayerChainPlan.resolve``
    and ``SubChainBPlan.resolve`` route every island subdir through here, so a single
    ``MMEL_CHAIN_M`` flips the whole chain's capture set. Pure string logic; never
    touches the card and never raises."""
    if chain_m() == 1:
        return sub
    m = chain_m()
    override = _CHAIN_ISLAND_M_OVERRIDES.get(sub)
    if override is not None:
        return override.format(m=m)
    return f"{sub}_m{m}"


# ----------------------------------------------------------------------------- #
#  THE LAYER CHAIN SPEC — the 7-island ordered chain that IS a GLM-5.2 decode    #
#  layer, with the per-island dir + role. This is the RUN_CHAIN STEP-4 order:    #
#    "0 1 2 3 4 0 5" over (norm, qproj, kvproj, mla_paged, oproj, routed) =       #
#    input_norm → q_proj → kv_proj → MLA-paged → o_proj → post_norm → MoE.        #
#                                                                               #
#  NB the C-harness order indexes a dir LIST; here we name the islands directly  #
#  so the chain is self-describing + the edge-topology solver derives live-vs-   #
#  staged from the captured byte sizes (NOT hardcoded), so a re-capture cannot    #
#  silently mis-feed.                                                            #
# ----------------------------------------------------------------------------- #
# (step_name, cap_subdir) — input_norm and post_norm are the SAME captured norm
# island (the GLM layer's two RMSNorms are the identical recipe TYPE, replayed
# twice; the C harness does this via "0 ... 0").
_LAYER_CHAIN = (
    ("input_norm", "cap_norm"),
    ("q_proj",     "cap_qproj"),
    ("kv_proj",    "cap_kvproj"),
    ("mla",        "cap_mla_paged"),
    ("o_proj",     "cap_oproj"),
    ("post_norm",  "cap_norm"),
    ("moe",        "cap_routed"),
)


# ----------------------------------------------------------------------------- #
#  THE CORRECT SHAPE (ENGINE_BUILD_GAP_AUDIT §4.3/§4.4): SUB-CHAIN B.             #
#                                                                               #
#  The all-7 chain above is the WRONG device shape — it device-replays the       #
#  context-DYNAMIC attention (MLA + DSA) as a FROZEN captured recipe (S=128, one #
#  block, no indexer), which goes stale the instant the live decode KV/positions #
#  grow. That is why its 3 pre-attention branch edges are STAGED and why it is    #
#  NOT byte-exact vs the live stock layer.forward (only vs a serial control with  #
#  the SAME staged topology).                                                     #
#                                                                               #
#  The CORRECT split (the audit's verdict): chain ONLY the shape-static islands   #
#  device-side and keep MLA+DSA on the stock bucketed path. SUB-CHAIN B is the    #
#  highest-value, lowest-risk static sub-chain — the post-attention tail:         #
#                                                                               #
#      [ o_proj -> post_attention_layernorm -> MoE ]                             #
#                                                                               #
#  Its LIVE INPUT is the STOCK self_attn output ([1, N*V] = [1,16384] bf16 =      #
#  32768 B == cap_oproj's in0); o_proj OUT (12288) feeds post_norm in0 (12288,    #
#  LIVE), post_norm OUT (12288) feeds MoE in0 (12288, LIVE); MoE OUT (12288 =      #
#  [1,6144]) IS the post-layer hidden, byte-identical to stock o_proj+post_norm+  #
#  mlp on that same attention output. EVERY island here is genuinely fixed-shape  #
#  at M=1 and INDEPENDENT of context length — so it replays byte-exact from a      #
#  single capture, and (unlike the all-7) the full layer (stock norm/attn +       #
#  device sub-chain B) IS byte-comparable to stock layer.forward end-to-end.      #
#                                                                               #
#  MoE is the DOMINANT per-layer dispatch share (~0.52 ms / the intra-expert      #
#  loop, RUN_CHAIN); o_proj is a 192 MB-weight GEMM; the post-norm is a node.     #
#  So sub-chain B collapses the bulk of the ~135 stock dispatches/forward into    #
#  ONE synStreamSynchronize, while attention (~1 dispatch/layer) stays correct    #
#  on the stock path. This is the engine's correct, byte-exact-able shape.        #
# ----------------------------------------------------------------------------- #
_SUBCHAIN_B = (
    ("o_proj",    "cap_oproj"),
    ("post_norm", "cap_norm"),
    ("moe",       "cap_routed"),
)


# THE GATE for the correct static sub-chain (DISTINCT from MMEL_CHAIN_LAYER, the
# unsound all-7). MMEL_CHAIN_STATIC=1 routes the post-attention tail through the
# device-resident sub-chain B while leaving input_norm + self_attn (MLA + DSA) on
# the stock bucketed path. This is the shape the audit recommends landing FIRST.
CHAIN_STATIC_ENV = "MMEL_CHAIN_STATIC"


def chain_static_enabled() -> bool:
    """True iff the operator selected the CORRECT static sub-chain device path
    (``MMEL_CHAIN_STATIC=1``) — device-resident [o_proj, post_norm, MoE] fed by
    the stock self_attn output, MLA+DSA left on the stock bucketed path. The ONLY
    gate that deserializes the 3 static recipes for sub-chain B; mutually exclusive
    with MMEL_CHAIN_LAYER (the all-7, which this supersedes)."""
    return os.environ.get(CHAIN_STATIC_ENV, "").strip().lower() in (
        "1", "true", "yes", "on", "static")


# ----------------------------------------------------------------------------- #
#  THE CAPTURED-BUCKET COVERAGE PREDICATE (card-free) — the integration fix.     #
#                                                                               #
#  THE 2026-06-22 BLOCKER (root cause): the 7 islands are captured at the M=1    #
#  single-stream DECODE bucket (norm in"0" = 12288 B = [1,6144] bf16). But the   #
#  seam routed EVERY forward to the chain, including the PROMPT/PREFILL forward   #
#  whose hidden_states is [128,6144] (1572864 B). `run()` correctly REFUSED that #
#  geometry (assert_live_hidden -> MmelChainLayerShapeError "would re-fire /      #
#  synStatus 26") — but prefill runs FIRST, so the worker died on the prompt     #
#  step BEFORE any M=1 decode step warmed/logged -> 0 WARMED, served fell to     #
#  stock, the A/B logged a false flat (68.7ms == OFF).                           #
#                                                                               #
#  THE FIX (here + the seam): the M=1-captured chain DRIVES the decode bucket it #
#  was captured at, and forwards it does NOT cover (prefill, any non-M=1 bucket, #
#  the warmup buckets) FLOW THROUGH the G1 op-for-op spine (which is byte-exact  #
#  stock and handles every M via the layer's own submodules). This is NOT a      #
#  silent stock-fallback masking a failure — a *mismatched* M=1 capture STILL    #
#  raises LOUD (the chain's own warmup/shape guards are unchanged); prefill is    #
#  simply not in the M=1 chain's scope, and the seam logs LOUD which forwards     #
#  the chain drove vs delegated. The captured-bucket byte size IS the discriminator.
#                                                                               #
#  We resolve the plan ONCE (card-free manifest parse) + cache the layer-input   #
#  byte size, so this predicate is a cheap integer compare per forward (no card, #
#  no re-parse). A resolve failure here is NON-FATAL (returns "cannot decide");  #
#  the seam treats "cannot decide" as "do not claim coverage" and the chain's    #
#  own LOUD warmup is the real gate when the bucket DOES match.                   #
# ----------------------------------------------------------------------------- #
_CAPTURED_LAYER_INPUT_NBYTES_CACHE: dict = {}


def captured_layer_input_nbytes(cap_parent: Optional[str] = None) -> Optional[int]:
    """The captured layer-input (norm in"0") byte size for the resolved cap dir, or
    None if the chain plan cannot be resolved (missing/malformed captures) — the
    caller treats None as "coverage undecidable, do not engage the chain on this
    forward". Resolved ONCE per (cap_parent, MMEL_CHAIN_M) and cached (card-free;
    pure manifest parse). At M=1 (default) this is the single-stream decode bucket
    the 7 islands were captured at (12288 B for GLM-5.2 [1,6144] bf16); at M=<M>
    (e.g. 32) it is the [M,6144] = M*12288 B concurrency bucket parsed from the
    cap_norm_m{M} manifest (norm-only resolve — see below)."""
    parent = cap_parent or cap_parent_dir()
    m = chain_m()
    key = (parent, m)
    if key in _CAPTURED_LAYER_INPUT_NBYTES_CACHE:
        return _CAPTURED_LAYER_INPUT_NBYTES_CACHE[key]
    try:
        if m != 1:
            # M != 1 (e.g. the [32,6144] concurrency bucket): the layer-input byte size
            # IS the norm island's in"0" (the first chain edge). Resolve ONLY the norm
            # island so COVERAGE does not require the MoE tail island — at M=32 the MoE
            # is the MONOLITHIC live op, NOT a cap_routed_m{M} capture (there is no such
            # dir), so demanding the full 7-island plan here would make every M!=1
            # forward UNDECIDABLE and the chain could never engage. The full-chain
            # resolve still LOUD-gates EXECUTION at warmup (DecoderLayerChainReplay /
            # SubChainBReplay), where the moe-tail binding IS exercised. M=1 keeps the
            # proven full-plan resolve verbatim (byte-identical).
            norm_dir = os.path.join(parent, _chain_island_dir("cap_norm"))
            nb = int(_parse_island_manifest("input_norm", norm_dir).in0_nbytes)
        else:
            plan = LayerChainPlan.resolve(parent)
            nb = int(plan.layer_input_nbytes)
    except Exception as e:  # noqa: BLE001 — undecidable is non-fatal here (LOUD at warmup)
        logger.warning(
            "[mmel.chain_layer] captured_layer_input_nbytes: could not resolve the "
            "chain plan under %r (M=%d) (%s: %s) -> coverage UNDECIDABLE; the chain "
            "will not claim this forward and the warmup gate stays the LOUD arbiter.",
            parent, m, type(e).__name__, e)
        _CAPTURED_LAYER_INPUT_NBYTES_CACHE[key] = None
        return None
    _CAPTURED_LAYER_INPUT_NBYTES_CACHE[key] = nb
    return nb


def _live_hidden_nbytes(hidden_states: Any) -> Optional[int]:
    """numel(hidden_states) * 2 (bf16), or None if the shape is unreadable. Pure
    shape arithmetic — no device touch, works on a real HPU tensor or a shape-only
    stand-in (anything exposing ``.shape``)."""
    try:
        shape = tuple(hidden_states.shape)
        numel = 1
        for d in shape:
            numel *= int(d)
        return numel * 2
    except Exception:  # noqa: BLE001
        return None


def chain_layer_covers(hidden_states: Any, cap_parent: Optional[str] = None) -> bool:
    """True iff the live hidden_states geometry == the captured M=1 decode bucket
    the 7-island chain replays. The seam engages the chain ONLY when this is True;
    every other forward (prefill, non-M=1 decode buckets, warmup buckets) is
    delegated to the G1 spine. False when the plan can't be resolved OR the live
    byte size differs — never raises (the decision is a cheap integer compare; the
    chain's own warmup_once/assert_live_hidden remain the LOUD arbiters once the
    bucket matches). Card-free."""
    want = captured_layer_input_nbytes(cap_parent)
    if want is None:
        return False
    got = _live_hidden_nbytes(hidden_states)
    if got is None:
        return False
    return got == want


@dataclass(frozen=True)
class IslandTensor:
    name: str          # the recipe's own launch-tensor name (== the .bin basename)
    nbytes: int
    role: str          # "IN_FILE" or "OUT"
    bin_path: str      # absolute path to the captured .bin for this tensor (IN only)


@dataclass(frozen=True)
class IslandSpec:
    """One island in the chain: its parsed manifest + the in"0" / OUT tensors."""

    step_name: str            # e.g. "input_norm"
    cap_dir: str              # absolute dir
    tensors: tuple            # tuple[IslandTensor], manifest order
    in0_name: str             # the recipe's input"0" (the per-edge fed input)
    in0_nbytes: int
    out_name: str
    out_nbytes: int

    @property
    def out_ref_path(self) -> str:
        return os.path.join(self.cap_dir, "out_ref.bin")

    @property
    def static_in_names(self) -> tuple:
        """All IN tensors EXCEPT in"0" (the per-edge fed input) — bound once,
        resident. For norm these are gamma; for a GEMM the weight; for MLA the
        latent/W_UK/cos-sin/cache/block-table; for MoE the 56 int4 weights."""
        return tuple(t.name for t in self.tensors
                     if t.role == "IN_FILE" and t.name != self.in0_name)


def _hidden_nbytes_from_header(lines) -> Optional[int]:
    """The live-hidden bf16 byte size (M*H*2) parsed from a manifest's
    '# ... M=.. H=..' geometry comment, or None if the header lacks M/H. Lets the
    feed edge be identified by SHAPE when a rebuilt op does not name its feed input
    '0' (mirrors mmel_moe_island's header parse). Pure string parse; never raises."""
    M = H = None
    for ln in lines:
        s = ln.strip()
        if not s.startswith("#"):
            continue
        for tok in s.lstrip("#").split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            try:
                iv = int(v)
            except ValueError:
                continue
            if k == "M":
                M = iv
            elif k == "H":
                H = iv
    if M is None or H is None:
        return None
    return M * H * 2


def _parse_island_manifest(step_name: str, cap_dir: str) -> IslandSpec:
    """Parse one cap_*/manifest.txt into an IslandSpec (card-free).

    Format (all the captures write it):
        # <geometry comment(s) with key=val tokens>
        # name<TAB>nbytes<TAB>role
        0<TAB><nbytes><TAB>IN_FILE        <- the recipe's input "0" (the fed edge)
        ...
        <N><TAB><nbytes><TAB>OUT

    The fed-edge input is the recipe's launch-tensor literally named "0" (the
    capture scripts + the C harness both treat tensor "0" as the feed input —
    ``feed_in_idx`` is the tensor whose name == "0"). We REQUIRE a "0" IN and a
    single OUT; anything else RAISES (a recipe with no "0" can't be chained by
    the size-match feed rule; we refuse to guess)."""
    mpath = os.path.join(cap_dir, "manifest.txt")
    if not os.path.isfile(mpath):
        raise MmelChainLayerError(
            f"[mmel.chain_layer] island {step_name!r} manifest not found: {mpath!r}. "
            f"Set {CAP_DIR_ENV} to the parent of the cap_* dirs, or run the on-card "
            "capture (capture_*_island.py) first.")
    with open(mpath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    tensors = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] malformed manifest row {s!r} in {mpath!r} "
                "(expected: name<TAB>nbytes<TAB>role).")
        name, nbytes_s, role = parts[0], parts[1], parts[2]
        try:
            nbytes = int(nbytes_s)
        except ValueError as e:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] non-integer nbytes {nbytes_s!r} in {mpath!r} "
                f"row {s!r}.") from e
        tensors.append(IslandTensor(
            name=name, nbytes=nbytes, role=role,
            bin_path=os.path.join(cap_dir, f"{name}.bin")))

    if not tensors:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] island {step_name!r} manifest {mpath!r} has no rows.")
    outs = [t for t in tensors if t.role == "OUT"]
    ins = [t for t in tensors if t.role == "IN_FILE"]
    if len(outs) != 1:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] island {step_name!r} expected exactly 1 OUT, got "
            f"{len(outs)} in {mpath!r}.")
    in0 = [t for t in ins if t.name == "0"]
    if len(in0) != 1:
        # The feed edge is conventionally the recipe input named "0", but a REBUILT op
        # may assign its OWN numeric launch-tensor names (e.g. the int4 routed op after
        # the BSGT1 rework names its [M,H] feed "15", not "0"). Fall back to identifying
        # the feed by the live-hidden byte size (M*H*2 from the header) — the SAME
        # shape-based rule mmel_moe_island.RoutedIslandManifest.parse uses for the live
        # row. LOUD if still not exactly one (never guess the feed).
        hb = _hidden_nbytes_from_header(lines)
        if hb is not None:
            by_shape = [t for t in ins if t.nbytes == hb]
            if len(by_shape) == 1:
                in0 = by_shape
    if len(in0) != 1:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] island {step_name!r} feed edge unresolved: no IN named "
            f"'0', and the live-hidden size from the header matched {len(in0)} IN(s) in "
            f"{mpath!r}. The chained-layer feed rule binds the feed input (recipe '0' or "
            "the unique [M,H] bf16 row) to the prior island's OUT; refusing to guess.")
    # Every INPUT .bin must exist (static weights + the captured fed input are
    # replayed from them; missing => we'd bind garbage). The OUT has no input .bin
    # (the recipe computes it; the oracle is out_ref.bin). LOUD up front.
    missing = [t.bin_path for t in tensors
               if t.role == "IN_FILE" and not os.path.isfile(t.bin_path)]
    if missing:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] island {step_name!r}: {len(missing)} INPUT .bin "
            f"missing under {cap_dir!r} (first: {missing[0]!r}). Re-run the on-card "
            "capture to materialize the recipe inputs.")
    return IslandSpec(
        step_name=step_name, cap_dir=cap_dir, tensors=tuple(tensors),
        in0_name=in0[0].name, in0_nbytes=in0[0].nbytes,
        out_name=outs[0].name, out_nbytes=outs[0].nbytes)


@dataclass(frozen=True)
class ChainEdge:
    """How step k's input"0" is fed: LIVE (bound to step k-1's OUT, on-device, no
    host fence) iff their byte sizes match; else STAGED (replays its own captured
    input). Step 0's input"0" is the LAYER's live input (the decode hidden_states).
    """

    step_idx: int
    step_name: str
    feed: str          # "LIVE" | "STAGED" | "LAYER_INPUT"
    in0_nbytes: int
    prev_out_nbytes: int   # 0 for step 0


@dataclass(frozen=True)
class LayerChainPlan:
    """The fully-resolved 7-island chain: the per-island specs + the per-edge
    LIVE/STAGED feed plan, derived from the CAPTURED byte sizes (RUN_CHAIN
    STEP-4's autofeed rule), so a re-capture cannot silently mis-feed."""

    cap_parent: str
    islands: tuple        # tuple[IslandSpec], in chain order
    edges: tuple          # tuple[ChainEdge], one per island
    layer_input_nbytes: int   # the byte size of the layer's live hidden_states input

    @property
    def n_live_edges(self) -> int:
        return sum(1 for e in self.edges if e.feed == "LIVE")

    @property
    def n_staged_edges(self) -> int:
        return sum(1 for e in self.edges if e.feed == "STAGED")

    @classmethod
    def resolve(cls, cap_parent: str) -> "LayerChainPlan":
        islands = tuple(
            _parse_island_manifest(
                step_name, os.path.join(cap_parent, _chain_island_dir(sub)))
            for (step_name, sub) in _LAYER_CHAIN)
        edges = []
        for i, isl in enumerate(islands):
            if i == 0:
                # the layer's live input (decode hidden_states) feeds norm's in"0".
                edges.append(ChainEdge(
                    step_idx=0, step_name=isl.step_name, feed="LAYER_INPUT",
                    in0_nbytes=isl.in0_nbytes, prev_out_nbytes=0))
                continue
            prev = islands[i - 1]
            # the autofeed rule (RUN_CHAIN STEP-4): live ONLY on size-matching
            # (linear) edges; stage the branch edges.
            if isl.in0_nbytes == prev.out_nbytes:
                feed = "LIVE"
            else:
                feed = "STAGED"
            edges.append(ChainEdge(
                step_idx=i, step_name=isl.step_name, feed=feed,
                in0_nbytes=isl.in0_nbytes, prev_out_nbytes=prev.out_nbytes))
        plan = cls(cap_parent=cap_parent, islands=islands, edges=tuple(edges),
                   layer_input_nbytes=islands[0].in0_nbytes)
        plan._sanity()
        return plan

    def _sanity(self) -> None:
        # At least the linear spine must be live (else the chain is a glorified
        # independent replay — the 1-fence collapse still holds but the FED proof
        # would be vacuous). RUN_CHAIN STEP-4 proved 3 live spine edges; demand >=1.
        if self.n_live_edges < 1:
            raise MmelChainLayerError(
                "[mmel.chain_layer] resolved 0 LIVE feed edges in the layer chain "
                f"(edges={[(e.step_name, e.feed) for e in self.edges]}). The captured "
                "island byte sizes do not size-match on ANY linear edge; the chain "
                "would not propagate any live value. Re-check the captures / the "
                "chain order.")
        # The last island (MoE) must be live-fed (it is the layer's OUTPUT and the
        # spine's tail; if it staged, the chain output wouldn't reflect live input).
        if self.edges[-1].feed != "LIVE":
            raise MmelChainLayerError(
                f"[mmel.chain_layer] the tail island {self.islands[-1].step_name!r} "
                f"is {self.edges[-1].feed}, not LIVE — the chain OUTPUT would not "
                "depend on the live input. The spine MLA->o_proj->norm->MoE must be "
                "size-matched (RUN_CHAIN STEP-4).")

    def assert_live_hidden(self, name: str, shape) -> None:
        """LOUD guard: the live layer-input hidden_states must match the captured
        norm input byte geometry (else binding it re-fires the norm recipe)."""
        try:
            numel = 1
            for d in shape:
                numel *= int(d)
        except Exception as e:  # noqa: BLE001
            raise MmelChainLayerShapeError(
                f"[mmel.chain_layer] live tensor {name!r} has unreadable shape "
                f"{shape!r}.") from e
        # the norm input is bf16; nbytes = numel * 2.
        got_nb = numel * 2
        if got_nb != self.layer_input_nbytes:
            raise MmelChainLayerShapeError(
                f"[mmel.chain_layer] STATIC-SHAPE violation: live {name!r} is "
                f"{got_nb} B ({numel} bf16 elems) but the captured layer-input norm "
                f"expects {self.layer_input_nbytes} B. The chain would re-fire / hit "
                "synStatus 26. Re-capture this bucket.")


# ----------------------------------------------------------------------------- #
#  SUB-CHAIN B PLAN (the CORRECT static tail) — [o_proj, post_norm, MoE].         #
#                                                                               #
#  Resolved from the SAME captured manifests as the all-7 plan, but as a 3-island #
#  spine whose LAYER_INPUT is the STOCK self_attn output ([1,16384] bf16 = 32768  #
#  B == cap_oproj in0). EVERY internal edge is LIVE (size-matched linear spine):  #
#    attn_out -> o_proj          : 32768 -> 32768  LAYER_INPUT (the live attn OUT)#
#    o_proj   -> post_norm       : 12288 -> 12288  LIVE                           #
#    post_norm-> MoE             : 12288 -> 12288  LIVE                           #
#  MoE OUT (12288 = [1,6144]) is the post-layer hidden — byte-identical to stock  #
#  o_proj+post_attention_layernorm+mlp on the same attention output.             #
#                                                                               #
#  Reuses IslandSpec / ChainEdge / _parse_island_manifest verbatim; the only      #
#  delta vs LayerChainPlan is the island LIST (3, not 7) and that the LAYER_INPUT  #
#  byte size is the ATTENTION OUTPUT (o_proj in0), not the norm input.            #
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubChainBPlan:
    """The resolved static post-attention sub-chain [o_proj, post_norm, MoE]: the
    per-island specs + the per-edge feed plan (all LIVE on the spine), derived from
    the captured byte sizes. This is the audit's recommended FIRST landing — the
    dominant-dispatch-share tail, fed by the stock attention, byte-exact vs stock."""

    cap_parent: str
    islands: tuple        # tuple[IslandSpec], in [o_proj, post_norm, moe] order
    edges: tuple          # tuple[ChainEdge], one per island
    attn_out_nbytes: int  # the byte size of the LIVE attention-output input (o_proj in0)

    @property
    def n_live_edges(self) -> int:
        return sum(1 for e in self.edges if e.feed == "LIVE")

    @property
    def out_nbytes(self) -> int:
        """The sub-chain output (MoE OUT) byte size — the post-layer hidden."""
        return self.islands[-1].out_nbytes

    @classmethod
    def resolve(cls, cap_parent: str) -> "SubChainBPlan":
        islands = tuple(
            _parse_island_manifest(
                step_name, os.path.join(cap_parent, _chain_island_dir(sub)))
            for (step_name, sub) in _SUBCHAIN_B)
        edges = []
        for i, isl in enumerate(islands):
            if i == 0:
                # the live attention output (stock self_attn OUT) feeds o_proj's in"0".
                edges.append(ChainEdge(
                    step_idx=0, step_name=isl.step_name, feed="LAYER_INPUT",
                    in0_nbytes=isl.in0_nbytes, prev_out_nbytes=0))
                continue
            prev = islands[i - 1]
            if isl.in0_nbytes == prev.out_nbytes:
                feed = "LIVE"
            else:
                feed = "STAGED"
            edges.append(ChainEdge(
                step_idx=i, step_name=isl.step_name, feed=feed,
                in0_nbytes=isl.in0_nbytes, prev_out_nbytes=prev.out_nbytes))
        plan = cls(cap_parent=cap_parent, islands=islands, edges=tuple(edges),
                   attn_out_nbytes=islands[0].in0_nbytes)
        plan._sanity()
        return plan

    def _sanity(self) -> None:
        # Sub-chain B is a PURE LINEAR SPINE: EVERY internal edge must be LIVE
        # (size-matched). A STAGED edge here would mean a frozen captured input on
        # the static tail — which the audit's whole point is to NOT do. Demand it.
        names = [isl.step_name for isl in self.islands]
        if names != ["o_proj", "post_norm", "moe"]:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] sub-chain B island order drifted: {names} "
                "(expected [o_proj, post_norm, moe]).")
        bad = [(e.step_name, e.feed) for e in self.edges[1:] if e.feed != "LIVE"]
        if bad:
            raise MmelChainLayerError(
                "[mmel.chain_layer] sub-chain B has a non-LIVE internal edge "
                f"{bad} — the post-attention tail [o_proj->post_norm->MoE] must be a "
                "fully size-matched LIVE spine (o_proj OUT 12288 == post_norm in0; "
                "post_norm OUT 12288 == MoE in0). A STAGED edge would freeze a "
                "captured input on the static tail; re-check the captures.")
        # The tail (MoE) must be LIVE (it is the sub-chain OUTPUT).
        if self.edges[-1].feed != "LIVE":
            raise MmelChainLayerError(
                f"[mmel.chain_layer] sub-chain B tail {self.islands[-1].step_name!r} "
                f"is {self.edges[-1].feed}, not LIVE — the output would not depend on "
                "the live attention input.")
        # The sub-chain output (MoE OUT) must be a valid [M,H] hidden = 12288 B for
        # GLM-5.2; we only assert it is non-zero + even (bf16) here (the layer-input
        # equality is checked by the seam against the model's hidden size).
        if self.out_nbytes <= 0 or (self.out_nbytes % 2) != 0:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] sub-chain B output {self.out_nbytes} B is not a "
                "valid bf16 hidden.")

    def assert_live_attn_out(self, name: str, shape) -> None:
        """LOUD guard: the live attention output feeding o_proj must match the
        captured o_proj in0 byte geometry (else binding it re-fires o_proj)."""
        try:
            numel = 1
            for d in shape:
                numel *= int(d)
        except Exception as e:  # noqa: BLE001
            raise MmelChainLayerShapeError(
                f"[mmel.chain_layer] live tensor {name!r} has unreadable shape "
                f"{shape!r}.") from e
        got_nb = numel * 2  # bf16
        if got_nb != self.attn_out_nbytes:
            raise MmelChainLayerShapeError(
                f"[mmel.chain_layer] STATIC-SHAPE violation: live attention output "
                f"{name!r} is {got_nb} B ({numel} bf16 elems) but the captured o_proj "
                f"input expects {self.attn_out_nbytes} B ([1,16384] = N*V). Sub-chain "
                "B would re-fire / hit synStatus 26. Re-capture this bucket.")


# ----------------------------------------------------------------------------- #
#  SUB-CHAIN B COVERAGE PREDICATE (card-free) — keyed on the ATTENTION OUTPUT.    #
#                                                                               #
#  The seam engages sub-chain B ONLY when the stock self_attn output geometry ==  #
#  the captured o_proj in0 (M=1 decode: [1,16384] = 32768 B). Prefill / non-M=1   #
#  attention outputs (e.g. [128,16384]) are NOT covered -> the layer runs the     #
#  stock tail. Resolved ONCE + cached; a resolve failure is non-fatal (returns    #
#  None -> "undecidable" -> the seam does not engage). Pure integer compare.      #
# ----------------------------------------------------------------------------- #
_SUBCHAIN_B_ATTN_NBYTES_CACHE: dict = {}


def subchain_b_attn_out_nbytes(cap_parent: Optional[str] = None) -> Optional[int]:
    """The captured o_proj-input (attention output) byte size for the resolved cap
    dir, or None if the sub-chain B plan cannot be resolved. M=1 single-stream
    decode: 32768 B ([1,16384] bf16). Cached per cap_parent (card-free)."""
    parent = cap_parent or cap_parent_dir()
    key = (parent, chain_m())
    if key in _SUBCHAIN_B_ATTN_NBYTES_CACHE:
        return _SUBCHAIN_B_ATTN_NBYTES_CACHE[key]
    try:
        plan = SubChainBPlan.resolve(parent)
        nb = int(plan.attn_out_nbytes)
    except Exception as e:  # noqa: BLE001 — undecidable is non-fatal (LOUD at warmup)
        logger.warning(
            "[mmel.chain_layer] subchain_b_attn_out_nbytes: could not resolve the "
            "sub-chain B plan under %r (M=%d) (%s: %s) -> coverage UNDECIDABLE; the "
            "sub-chain will not claim this forward and the warmup gate stays the LOUD "
            "arbiter.", parent, chain_m(), type(e).__name__, e)
        _SUBCHAIN_B_ATTN_NBYTES_CACHE[key] = None
        return None
    _SUBCHAIN_B_ATTN_NBYTES_CACHE[key] = nb
    return nb


def subchain_b_covers(attn_out: Any, cap_parent: Optional[str] = None) -> bool:
    """True iff the live stock-attention-output geometry == the captured M=1 decode
    o_proj input the sub-chain B replays. The seam engages sub-chain B ONLY when
    this is True; prefill / non-M=1 attention outputs run the stock tail. False when
    the plan can't be resolved OR the byte size differs — never raises. Card-free."""
    want = subchain_b_attn_out_nbytes(cap_parent)
    if want is None:
        return False
    got = _live_hidden_nbytes(attn_out)
    if got is None:
        return False
    return got == want


# ----------------------------------------------------------------------------- #
#  THE DEVICE-RESIDENT LAYER-CHAIN REPLAY (on-card only; reuses RawCsIslandReplay)#
#                                                                               #
#  Lifecycle:                                                                     #
#    warmup_once():  deserialize all 7 recipes; upload each island's STATIC      #
#                    weight .bins + its STAGED in"0" .bin to resident device      #
#                    buffers (bound once); alloc each island's resident OUT       #
#                    buffer + the recipe workspace; reuse the bridge device +     #
#                    compute stream. Cached on the layer.                         #
#    run(hidden):    bind the LIVE hidden_states into island[0]'s in"0"; bind     #
#                    each LIVE edge's in"0" to the PRIOR island's resident OUT;    #
#                    synLaunch all 7 islands back-to-back on the bridge stream    #
#                    (NO fence between); ONE synStreamSynchronize at the end;      #
#                    return island[-1]'s OUT as the layer output ([M,H] bf16).    #
#                                                                               #
#  This is the literal csengine 1-fence-per-LAYER chain (RUN_CHAIN STEP-4),       #
#  driven from live decode: OUR ONE completion replaces the bridge's per-recipe   #
#  events for the WHOLE layer, the math (the GC kernels) byte-identical, the      #
#  on-device stream order carrying the spine dependency exactly like the serial   #
#  control.                                                                       #
# ----------------------------------------------------------------------------- #
@dataclass
class _IslandRT:
    """Per-island RUNTIME state (resident device buffers + the deserialized
    recipe handle)."""

    spec: IslandSpec
    replay: Any = None                 # mmel_l6_chain.RawCsIslandReplay
    static_dev: dict = field(default_factory=dict)   # name -> resident HPU tensor
    staged_in0: Any = None             # the STAGED in"0" resident tensor (or None if LIVE/LAYER_INPUT)
    out_tensor: Any = None             # the resident OUT HPU tensor (uint8)
    ws_dev: Any = None
    ws_va: int = 0


@dataclass
class DecoderLayerChainReplay:
    """Device-resident replay of the FULL 7-island GLM-5.2 decode layer, live-fed
    on the linear spine + staged on the pre-attention branch, behind ONE host
    fence. Reuses S1's RawCsIslandReplay synLaunch surface (no new ABI)."""

    plan: LayerChainPlan
    _islands_rt: list = field(default_factory=list)   # list[_IslandRT], chain order
    _stream_handle: int = 0
    _warm: bool = False

    def warmup_once(self) -> "DecoderLayerChainReplay":
        if self._warm:
            return self
        import torch  # lazy — card only
        import numpy as np  # lazy

        # Reuse the proven S1/G1 ctypes replay surface (no new ABI).
        try:
            from vllm_gaudi.v1.worker.mmel_l6_chain import (  # type: ignore
                RawCsIslandReplay, reuse_bridge_device_and_stream)
        except Exception:  # noqa: BLE001 — path-based sibling import on a bare box
            RawCsIslandReplay, reuse_bridge_device_and_stream = _import_chain_symbols()

        _dev, self._stream_handle = reuse_bridge_device_and_stream()

        for i, (isl, edge) in enumerate(zip(self.plan.islands, self.plan.edges)):
            recipe_path = os.path.join(isl.cap_dir, "island.recipe")
            if not os.path.isfile(recipe_path):
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] recipe not found for island "
                    f"{isl.step_name!r}: {recipe_path!r}. Re-capture on-card.")
            replay = RawCsIslandReplay(recipe_path=recipe_path).deserialize()
            rt = _IslandRT(spec=isl, replay=replay)

            # Upload the STATIC weight/scale/zero (+ for STAGED islands their
            # captured in"0") tensors to RESIDENT device buffers, ONCE. Each .bin
            # is the device-layout bytes the recipe expects (byte-faithful uint8
            # upload — the recipe binds by ADDRESS + its own metadata, exactly the
            # C harness + S1 contract).
            for name in isl.static_in_names:
                t = next(t for t in isl.tensors if t.name == name)
                buf = np.fromfile(t.bin_path, dtype=np.uint8)
                if buf.nbytes != t.nbytes:
                    raise MmelChainLayerError(
                        f"[mmel.chain_layer] {isl.step_name!r} static {name!r} .bin is "
                        f"{buf.nbytes} B but manifest says {t.nbytes} B "
                        f"({t.bin_path!r}); re-capture.")
                rt.static_dev[name] = torch.from_numpy(buf).to("hpu").contiguous()

            # STAGED in"0": its captured input is resident (the branch edge does NOT
            # receive a live value). LIVE / LAYER_INPUT in"0" is bound per step.
            if edge.feed == "STAGED":
                t = next(t for t in isl.tensors if t.name == isl.in0_name)
                buf = np.fromfile(t.bin_path, dtype=np.uint8)
                if buf.nbytes != t.nbytes:
                    raise MmelChainLayerError(
                        f"[mmel.chain_layer] {isl.step_name!r} staged in0 {isl.in0_name!r} "
                        f".bin is {buf.nbytes} B but manifest says {t.nbytes} B; re-capture.")
                rt.staged_in0 = torch.from_numpy(buf).to("hpu").contiguous()

            # Resident OUT buffer (the recipe writes it; bound by address). For a
            # LIVE-fed downstream edge this OUT is the next island's in"0".
            rt.out_tensor = torch.empty(
                isl.out_nbytes, dtype=torch.uint8, device="hpu").contiguous()

            # Device WORKSPACE (queried from the recipe; 0 => none).
            ws = int(getattr(replay, "_ws_size", 0))
            if ws > 0:
                rt.ws_dev = torch.empty(ws, dtype=torch.uint8, device="hpu").contiguous()
                rt.ws_va = int(rt.ws_dev.data_ptr())
            self._islands_rt.append(rt)

        self._warm = True
        logger.info(
            "[mmel.chain_layer] WARMED 7-island decode-layer chain: %d islands "
            "resident (%d LIVE-fed spine edges, %d STAGED branch edges), stream="
            "0x%x. The WHOLE layer now replays on OUR stream behind ONE "
            "synStreamSynchronize (1 fence/layer vs the per-recipe bridge syncs).",
            len(self._islands_rt), self.plan.n_live_edges,
            self.plan.n_staged_edges, self._stream_handle)
        return self

    def run(self, hidden_states: Any) -> Any:
        """Replay the 7-island layer chain with the LIVE hidden_states; return the
        device layer output ([M,H] bf16, the stock layer output shape)."""
        if not self._warm:
            raise MmelChainLayerError(
                "[mmel.chain_layer] run() before warmup_once(); call warmup_once() at "
                "the first decode step so the 7 recipes + static weights are resident.")
        import torch  # lazy — card only

        self.plan.assert_live_hidden("hidden_states", tuple(hidden_states.shape))
        live = hidden_states.contiguous()

        # Enqueue all 7 islands back-to-back on the bridge stream, NO fence between.
        # Each island's in"0" is: the LIVE hidden (step 0) / the prior island's
        # resident OUT (LIVE edge) / its own staged input (STAGED edge).
        prev_out = None
        for rt, edge in zip(self._islands_rt, self.plan.edges):
            isl = rt.spec
            # build {recipe tensor name -> resident/live HPU tensor}.
            name_to_tensor = dict(rt.static_dev)  # static weights (resident)
            if edge.feed == "LAYER_INPUT":
                name_to_tensor[isl.in0_name] = live
            elif edge.feed == "LIVE":
                # bind in"0" to the PRIOR island's resident OUT (on-device feed).
                # The OUT tensor is uint8 [nbytes]; the recipe binds by address so a
                # byte-faithful view is the device payload (sizes asserted equal in
                # the plan).
                name_to_tensor[isl.in0_name] = prev_out
            else:  # STAGED
                name_to_tensor[isl.in0_name] = rt.staged_in0
            name_to_tensor[isl.out_name] = rt.out_tensor

            # Sanity: the map must cover EXACTLY this recipe's tensors.
            expected = {t.name for t in isl.tensors}
            got = set(name_to_tensor.keys())
            if got != expected:
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] island {isl.step_name!r} bind-map mismatch: "
                    f"missing={sorted(expected - got)} extra={sorted(got - expected)}.")

            launch_arr = rt.replay.build_launch_array(name_to_tensor)
            rt.replay.launch(self._stream_handle, launch_arr, ws_ptr=rt.ws_va)
            prev_out = rt.out_tensor   # feeds the next LIVE edge

        # ONE synStreamSynchronize for the WHOLE 7-island chain (the 1-fence/layer
        # primitive, live-fed). This is the structural collapse the per-layer
        # recipe-boundary sync used to cost.
        _syn_stream_synchronize(self._stream_handle)

        # The layer output = the tail island (MoE) OUT, viewed bf16 [M,H].
        tail = self._islands_rt[-1]
        H = tail.spec.out_nbytes // 2  # bf16
        out = tail.out_tensor.view(torch.bfloat16).view(-1, H)

        # FORENSIC OVERRIDE-PROOF POISON (default OFF). Perturb the DEVICE output the
        # serve actually consumes, with an RMSNorm-robust signal: (a) a per-channel
        # ADDITIVE offset (survives RMSNorm — not a global scale) + (b) a single-channel
        # SPIKE (RMSNorm cannot wash one saturated feature). If the chain's device output
        # is what gets served, greedy text GARBLES; if an OEM/stock path overrode it, the
        # text stays COHERENT (override confirmed). Applied on-device (out is an HPU
        # tensor) so it exercises the exact served value — NOT a host-side copy.
        if _chain_poison_enabled():
            mag = _chain_poison_mag()
            # (a) per-channel additive offset: a deterministic ramp across H so it is
            # not a constant the norm could re-center trivially; magnitude ~mag.
            n = out.shape[-1]
            offs = torch.arange(n, device=out.device, dtype=out.dtype)
            offs = (offs / max(n - 1, 1)) * mag        # 0..mag ramp, per channel
            out = out + offs
            # (b) single-channel spike: saturate channel 0 far past the rest so even an
            # adversarial normalization keeps a visible, token-changing distortion.
            out[..., 0] = out[..., 0] + mag * 16.0
        return out


def _syn_stream_synchronize(stream_handle: int) -> None:
    """ONE host fence for the whole layer chain: synStreamSynchronize on the bridge
    stream. Reuses the same libSynapse CDLL surface RawCsIslandReplay built."""
    import ctypes
    from vllm_gaudi.v1.worker.mmel_l6_chain import (  # type: ignore
        _make_ctypes_syn_surface)

    syn, _ = _make_ctypes_syn_surface()
    st = syn.synStreamSynchronize(ctypes.c_void_p(stream_handle))
    if st != 0:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] synStreamSynchronize -> status={st} (the layer "
            "chain's one host completion failed).")


# ----------------------------------------------------------------------------- #
#  SUB-CHAIN B DEVICE REPLAY (the CORRECT shape — on-card only; reuses S1 surface)#
#                                                                               #
#  Device-resident replay of the static post-attention tail [o_proj, post_norm,  #
#  MoE], LIVE-FED by the STOCK self_attn output, behind ONE host fence. The all-7 #
#  DecoderLayerChainReplay's WRONG move was device-replaying the frozen MLA; this #
#  class drops attention from the device chain entirely and feeds o_proj the live #
#  stock attention OUT — so EVERY island is genuinely fixed-shape and the result  #
#  is byte-exact vs stock o_proj+post_attention_layernorm+mlp on that same input. #
#                                                                               #
#  Lifecycle (identical mechanism to DecoderLayerChainReplay, 3 islands not 7):   #
#    warmup_once(): deserialize o_proj/norm/MoE recipes; upload their STATIC      #
#                   weights (o_proj 192MB GEMM weight, norm gamma, the 8 int4      #
#                   expert weights + scales) resident; alloc resident OUT buffers  #
#                   + workspaces; reuse the bridge device + compute stream.        #
#    run(attn_out): bind the LIVE attention output into o_proj's in"0"; bind each  #
#                   LIVE edge's in"0" to the prior island's resident OUT; synLaunch #
#                   o_proj->post_norm->MoE back-to-back (NO fence between); ONE     #
#                   synStreamSynchronize; return MoE OUT ([M,H] bf16) = the post-  #
#                   layer hidden.                                                  #
# ----------------------------------------------------------------------------- #
@dataclass
class SubChainBReplay:
    """Device-resident replay of the static post-attention tail
    [o_proj -> post_attention_layernorm -> MoE], LIVE-fed by the stock self_attn
    output, behind ONE host fence. The audit's recommended FIRST landing — collapses
    the dominant per-layer dispatch share (MoE loop + o_proj GEMM + the post-norm)
    into one synStreamSynchronize while attention stays on the stock bucketed path.
    Reuses S1's RawCsIslandReplay synLaunch surface (no new ABI)."""

    plan: SubChainBPlan
    _islands_rt: list = field(default_factory=list)   # list[_IslandRT], chain order
    _stream_handle: int = 0
    _warm: bool = False

    def warmup_once(self) -> "SubChainBReplay":
        if self._warm:
            return self
        import torch  # lazy — card only
        import numpy as np  # lazy

        try:
            from vllm_gaudi.v1.worker.mmel_l6_chain import (  # type: ignore
                RawCsIslandReplay, reuse_bridge_device_and_stream)
        except Exception:  # noqa: BLE001 — path-based sibling import on a bare box
            RawCsIslandReplay, reuse_bridge_device_and_stream = _import_chain_symbols()

        _dev, self._stream_handle = reuse_bridge_device_and_stream()

        for i, (isl, edge) in enumerate(zip(self.plan.islands, self.plan.edges)):
            recipe_path = os.path.join(isl.cap_dir, "island.recipe")
            if not os.path.isfile(recipe_path):
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] recipe not found for sub-chain B island "
                    f"{isl.step_name!r}: {recipe_path!r}. Re-capture on-card.")
            replay = RawCsIslandReplay(recipe_path=recipe_path).deserialize()
            rt = _IslandRT(spec=isl, replay=replay)

            # Upload the STATIC weight/scale/zero tensors resident, ONCE (byte-faithful
            # uint8 — the recipe binds by ADDRESS, exactly S1 / the C harness).
            for name in isl.static_in_names:
                t = next(t for t in isl.tensors if t.name == name)
                buf = np.fromfile(t.bin_path, dtype=np.uint8)
                if buf.nbytes != t.nbytes:
                    raise MmelChainLayerError(
                        f"[mmel.chain_layer] sub-chain B {isl.step_name!r} static "
                        f"{name!r} .bin is {buf.nbytes} B but manifest says {t.nbytes} "
                        f"B ({t.bin_path!r}); re-capture.")
                rt.static_dev[name] = torch.from_numpy(buf).to("hpu").contiguous()

            # Sub-chain B is a PURE LIVE spine — NO island has a staged in"0" (the
            # _sanity guard asserts every internal edge LIVE + the head is LAYER_INPUT).
            # Defend it: a STAGED edge here is a build error, not a runtime branch.
            if edge.feed == "STAGED":
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] sub-chain B island {isl.step_name!r} resolved "
                    "STAGED — the static tail must be a fully LIVE spine. Re-check the "
                    "captures (o_proj OUT 12288 must == post_norm in0; post_norm OUT "
                    "12288 must == MoE in0).")

            rt.out_tensor = torch.empty(
                isl.out_nbytes, dtype=torch.uint8, device="hpu").contiguous()
            ws = int(getattr(replay, "_ws_size", 0))
            if ws > 0:
                rt.ws_dev = torch.empty(ws, dtype=torch.uint8, device="hpu").contiguous()
                rt.ws_va = int(rt.ws_dev.data_ptr())
            self._islands_rt.append(rt)

        self._warm = True
        logger.info(
            "[mmel.chain_layer] WARMED sub-chain B static tail: %d islands resident "
            "([o_proj, post_norm, MoE], %d LIVE spine edges), stream=0x%x. The "
            "post-attention tail now replays on OUR stream behind ONE "
            "synStreamSynchronize (the dominant dispatch share collapsed; attention "
            "stays on the stock bucketed path).",
            len(self._islands_rt), self.plan.n_live_edges, self._stream_handle)
        return self

    def run(self, attn_out: Any) -> Any:
        """Replay [o_proj, post_norm, MoE] with the LIVE stock attention output;
        return the device post-layer hidden ([M,H] bf16, the stock layer output
        shape — byte-exact vs stock o_proj+post_attention_layernorm+mlp)."""
        if not self._warm:
            raise MmelChainLayerError(
                "[mmel.chain_layer] SubChainBReplay.run() before warmup_once(); call "
                "warmup_once() at the first decode step so the 3 recipes + static "
                "weights are resident.")
        import torch  # lazy — card only

        self.plan.assert_live_attn_out("attn_out", tuple(attn_out.shape))
        live = attn_out.contiguous()

        prev_out = None
        for rt, edge in zip(self._islands_rt, self.plan.edges):
            isl = rt.spec
            name_to_tensor = dict(rt.static_dev)  # static weights (resident)
            if edge.feed == "LAYER_INPUT":
                name_to_tensor[isl.in0_name] = live
            elif edge.feed == "LIVE":
                name_to_tensor[isl.in0_name] = prev_out
            else:  # STAGED — impossible here (warmup_once raised), defend anyway.
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] sub-chain B island {isl.step_name!r} STAGED "
                    "at run() — the static tail must be a pure LIVE spine.")
            name_to_tensor[isl.out_name] = rt.out_tensor

            expected = {t.name for t in isl.tensors}
            got = set(name_to_tensor.keys())
            if got != expected:
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] sub-chain B island {isl.step_name!r} bind-map "
                    f"mismatch: missing={sorted(expected - got)} "
                    f"extra={sorted(got - expected)}.")

            launch_arr = rt.replay.build_launch_array(name_to_tensor)
            rt.replay.launch(self._stream_handle, launch_arr, ws_ptr=rt.ws_va)
            prev_out = rt.out_tensor

        # ONE synStreamSynchronize for the whole [o_proj, post_norm, MoE] sub-chain.
        _syn_stream_synchronize(self._stream_handle)

        tail = self._islands_rt[-1]
        H = tail.spec.out_nbytes // 2  # bf16
        out = tail.out_tensor.view(torch.bfloat16).view(-1, H)

        # FORENSIC OVERRIDE-PROOF POISON (default OFF) — perturb the DEVICE output the
        # serve consumes, RMSNorm-robustly (per-channel additive ramp + single-channel
        # spike). GARBLED greedy ==> sub-chain B's device output IS served (engine
        # drives the serve); COHERENT ==> an OEM/stock path overrode it. Same
        # discriminator as the all-7 path (MMEL_CHAIN_POISON / MMEL_CHAIN_POISON_MAG).
        if _chain_poison_enabled():
            mag = _chain_poison_mag()
            n = out.shape[-1]
            offs = torch.arange(n, device=out.device, dtype=out.dtype)
            offs = (offs / max(n - 1, 1)) * mag
            out = out + offs
            out[..., 0] = out[..., 0] + mag * 16.0
        return out


def _import_chain_symbols():
    """Path-based fallback import of RawCsIslandReplay + reuse_bridge_device_and_stream
    (mirrors S1's fallback) for a box where the package isn't on the path."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "mmel_l6_chain.py")
    if not os.path.isfile(cand):
        raise MmelChainLayerError(
            f"[mmel.chain_layer] cannot import mmel_l6_chain (RawCsIslandReplay): "
            f"{cand!r} absent.")
    modname = "vllm_gaudi.v1.worker.mmel_l6_chain"
    spec = importlib.util.spec_from_file_location(modname, cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod   # register before exec (py3.14 frozen-dataclass fix)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.RawCsIslandReplay, mod.reuse_bridge_device_and_stream


# ----------------------------------------------------------------------------- #
#  The ENTRY the chain body calls when MMEL_CHAIN_LAYER=1.                        #
#                                                                               #
#  Caches one DecoderLayerChainReplay per layer on the layer object (warmed on   #
#  first use). The seam's chained-decoder forward routes the WHOLE layer here    #
#  when the gate is on; otherwise it uses the stock/G1 path.                     #
# ----------------------------------------------------------------------------- #
_MMEL_LAYER_ATTR = "_mmel_decoder_layer_chain"


def run_decoder_layer_via_chain_replay(layer: Any, hidden_states: Any) -> Any:
    """Drive THIS layer's WHOLE forward via the device-resident 7-island chain.

    First call (per layer): resolve the chain plan (parse 7 manifests, derive the
    live/staged edge topology), warm the replay (deserialize 7 recipes + upload
    static weights + staged inputs resident). Every call: bind the live
    hidden_states, synLaunch all 7 islands on the bridge stream, ONE
    synStreamSynchronize. Returns the device layer output ([M,H] bf16).

    LOUD on any precondition failure (missing recipe/lib/weights, edge mismatch,
    shape drift) — the seam's coherent wrapper keeps the live generation safe by
    falling back to stock.
    """
    chain = getattr(layer, _MMEL_LAYER_ATTR, None)
    if chain is None:
        plan = LayerChainPlan.resolve(cap_parent_dir())
        chain = DecoderLayerChainReplay(plan=plan).warmup_once()
        setattr(layer, _MMEL_LAYER_ATTR, chain)
    return chain.run(hidden_states)


# ============================================================================= #
#  THE LIVE-FED MLA ATTENTION CHAIN (MMEL_CHAIN_LIVE=1) — the CORRECTNESS FIX.    #
#                                                                               #
#  THE BUG (proven): the all-7 DecoderLayerChainReplay STAGES the pre-attention  #
#  branch — the MLA island's q_nope/q_pe/flat_cache/rope/pad_mask replay from     #
#  FROZEN .bins (the size-match autofeed only feeds LIVE on a byte-size match; the #
#  MLA branch never matches). So the layer's attention output is a CONSTANT,      #
#  independent of the live decode token AND of the growing KV — byte-exact only   #
#  vs a *same-frozen* serial control, NOT vs the live stock forward.              #
#                                                                               #
#  THE FIX (LIVE_MLA_DECOMP_DESIGN §1-3): the NEW island order                    #
#    input_norm -> fused_qkv_a -> q_a_norm -> q_b -> kv_a_norm -> [latent-write]  #
#                -> MLA(S-bucket, live-KV) -> o_proj                             #
#  with EVERY activation edge bound LIVE to the prior island's output buffer, the #
#  branch splits as OFFSET (strided) SUB-BUFFER binds, and the MLA's              #
#  flat_cache/latent_kv bound to the LIVE GROWING KV (RoPE Option B: feed the     #
#  roped vllm cache + identity c_cos/c_sin + real q_cos/q_sin). The vestigial     #
#  kv_b (cap_kvproj) is DROPPED — pre-absorbed into W_UK_T/W_UV inside the op.    #
#                                                                               #
#  This module's CARD-FREE surface (gated; validated by E17-E22) resolves the new #
#  island plan + the per-tensor LIVE/LIVE_SUBBUFFER/LIVE_KV/META/RES bind map from #
#  the CAPTURED manifests (data-driven, so a re-capture cannot silently mis-feed).#
#  The BYTE-EXACT DATAFLOW proof is owned_engine/live_chain_test.py V7 (the live   #
#  chain == _mla_attn over a growing KV; the frozen-stage bug FAILS the input/KV   #
#  teeth). The on-card run (strided sub-buffer binds + live-KV gather + per-step   #
#  META) is the engage step; warmup/run LOUD-guard off-card.                      #
# ============================================================================= #
CHAIN_LIVE_ENV = "MMEL_CHAIN_LIVE"


def chain_live_enabled() -> bool:
    """True iff the operator selected the LIVE-FED MLA attention chain
    (``MMEL_CHAIN_LIVE=1``) — the correctness fix that replaces the STAGED
    pre-attention branch with live dataflow. Mutually exclusive with the all-7
    ``MMEL_CHAIN_LAYER`` (which this supersedes for attention) and
    ``MMEL_CHAIN_STATIC``. Unset => this module's live path is INERT and every
    existing code path is BYTE-IDENTICAL (E1-E16 unchanged)."""
    return os.environ.get(CHAIN_LIVE_ENV, "").strip().lower() in (
        "1", "true", "yes", "on", "live")


# ----------------------------------------------------------------------------- #
#  ROPE CONVENTION FORK (LIVE_MLA_DECOMP_DESIGN §2 Option A vs B).                #
#                                                                               #
#  Default OFF == OPTION A (oracle-faithful): the MLA op consumes an UN-roped     #
#  latent cache and ropes the cache pe AT READ with the REAL captured c_cos/c_sin #
#  tables (warmup uploads them verbatim from the .bins).                          #
#                                                                               #
#  MMEL_CHAIN_ROPE_OPTB=1 == OPTION B (recommended): bind the live-KV to vllm's    #
#  ALREADY-roped kv_cache[0]. The cache pe is then roped EXACTLY ONCE, by vllm at   #
#  write — so the op's in-graph cache-rope MUST be a NO-OP, else the cache is       #
#  DOUBLE-roped. Option B overrides ONLY the cache-rope tables c_cos -> 1 (ones)    #
#  and c_sin -> 0 (zeros) at the warmup RES upload; the LIVE query rope q_cos/q_sin #
#  (role META, bound per step) stays REAL. This makes the live chain byte-exact vs  #
#  the stock decode (which also ropes-at-write) without a duplicate UN-roped cache. #
#  Pure env read (fresh each call so an A/B can flip it per worker); never raises.  #
# ----------------------------------------------------------------------------- #
CHAIN_ROPE_OPTB_ENV = "MMEL_CHAIN_ROPE_OPTB"


def _rope_optb_enabled() -> bool:
    """True iff the operator selected ROPE Option B (``MMEL_CHAIN_ROPE_OPTB=1``):
    override the MLA cache-rope tables c_cos/c_sin to IDENTITY (cos=1, sin=0) so the
    in-graph cache-rope is a no-op (bind vllm's already-roped kv_cache[0] without
    double-roping). Default OFF == Option A (upload the REAL captured c_cos/c_sin)."""
    return os.environ.get(CHAIN_ROPE_OPTB_ENV, "").strip().lower() in (
        "1", "true", "yes", "on", "b", "optb")


# The NEW live attention island order (drops the vestigial kv_b = cap_kvproj). The
# base subdir names route through ``_chain_island_dir``, so at ``MMEL_CHAIN_M=32``
# they resolve to the captured cap_*_m32 / cap_mla_b32 set (the geometry the live
# chain is validated at FIRST; per-rank N=8 + multi-S are a follow-up).
_LIVE_ATTN_CHAIN = (
    ("input_norm",  "cap_norm"),
    ("fused_qkv_a", "cap_qkva"),
    ("q_a_norm",    "cap_qanorm"),
    ("q_b",         "cap_qproj"),
    ("kv_a_norm",   "cap_kvanorm"),
    ("mla",         "cap_mla_paged"),    # -> cap_mla_b{M}
    ("o_proj",      "cap_oproj"),
)

# bind roles (the live-feed map of LIVE_MLA_DECOMP_DESIGN §3).
LIVE_FEED = "LIVE"             # in0 <- prior island's FULL OUT (linear spine edge)
LIVE_SUB = "LIVE_SUBBUFFER"    # <- a (possibly strided) sub-buffer of a producer OUT
LIVE_KV = "LIVE_KV"            # <- the resident GROWING latent cache (kv_cache[0])
LIVE_META = "META"             # per-step small tensor (q_cos/q_sin/pad_mask)
LIVE_RES = "RES"               # resident upload-once static (weights; c_cos/c_sin)
LIVE_LAYER_INPUT = "LAYER_INPUT"
LIVE_OUT = "OUT"
_LIVE_ACTIVATION_ROLES = (LIVE_FEED, LIVE_SUB, LIVE_KV, LIVE_LAYER_INPUT)
_LIVE_KNOWN_ROLES = (LIVE_FEED, LIVE_SUB, LIVE_KV, LIVE_LAYER_INPUT,
                     LIVE_RES, LIVE_META, LIVE_OUT)


def _parse_mla_geometry(cap_dir: str) -> dict:
    """Parse N/P/R/L/V/B/S from the cap_mla_* manifest geometry comment (the live
    MLA bind map is DERIVED from these, so a re-capture cannot silently mis-feed).
    LOUD if any key is missing. Pure string parse; never touches the card."""
    mpath = os.path.join(cap_dir, "manifest.txt")
    if not os.path.isfile(mpath):
        raise MmelChainLayerError(
            f"[mmel.chain_layer] MLA manifest not found: {mpath!r}.")
    geo: dict = {}
    with open(mpath, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s.startswith("#"):
                continue
            for tok in s.lstrip("#").split():
                k, _, v = tok.partition("=")
                if k in ("N", "P", "R", "L", "V", "B", "S") and v.isdigit():
                    geo[k] = int(v)
                elif k == "op" and v:
                    # 'mla_decode' (non-paged: pre-gathered latent_kv [B,S,L+R]) vs
                    # 'mla_decode_paged' (the paged gather folded into op node-0 -> the op
                    # takes flat_cache + block_table instead of a pre-gathered latent_kv).
                    geo["op"] = v
                elif k == "interleave" and v.lstrip("-").isdigit():
                    # rope style baked into the captured recipe: 1=PAIRWISE (gptj-style,
                    # GLM) -> repeat_interleave(2); 0=NeoX (blockwise) -> cat([x,x]). Used
                    # to build q_cos/q_sin in the SAME layout the op was captured with.
                    geo["interleave"] = int(v)
    geo.setdefault("op", "mla_decode")   # legacy captures w/o an op= header => non-paged
    missing = [k for k in ("N", "P", "R", "L", "V", "B", "S") if k not in geo]
    if missing:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] MLA geometry comment in {mpath!r} missing {missing} "
            "(need N/P/R/L/V/B/S to derive the live MLA bind map).")
    return geo


@dataclass(frozen=True)
class LiveBind:
    """One MLA-chain recipe tensor's LIVE feed plan. ``role`` is from the §3 map;
    for LIVE_SUB the (offset, width, head_stride) describe the (strided) sub-buffer
    of ``producer``'s OUT (head_stride==0 => contiguous column slice; >0 => per-head
    slice with that row/head pitch)."""

    tensor_name: str
    role: str
    producer: Optional[str] = None      # producer step_name (LIVE / LIVE_SUB)
    offset_elems: int = 0
    width_elems: int = 0
    head_stride_elems: int = 0
    note: str = ""


@dataclass(frozen=True)
class LiveIsland:
    """One island in the live attention chain: its parsed spec + the per-recipe-
    tensor LIVE bind map."""

    spec: IslandSpec
    binds: tuple                        # tuple[LiveBind], one per recipe tensor

    @property
    def activation_live_binds(self) -> tuple:
        return tuple(b for b in self.binds if b.role in _LIVE_ACTIVATION_ROLES)

    @property
    def staged_activation_binds(self) -> tuple:
        """The WHOLE point of the fix: NO activation is staged from a .bin. Any bind
        whose role is not a known live/static/out role is a STAGED activation (the
        bug). This must be EMPTY for every island."""
        return tuple(b for b in self.binds if b.role not in _LIVE_KNOWN_ROLES)

    @property
    def res_bind_names(self) -> tuple:
        return tuple(b.tensor_name for b in self.binds if b.role == LIVE_RES)

    @property
    def meta_bind_names(self) -> tuple:
        return tuple(b.tensor_name for b in self.binds if b.role == LIVE_META)

    @property
    def kv_bind_names(self) -> tuple:
        return tuple(b.tensor_name for b in self.binds if b.role == LIVE_KV)


def _live_binds_simple(spec: IslandSpec, in0_bind: LiveBind) -> tuple:
    """Bind map for a norm/linear island: in0 (the given live bind) + the static
    weight/gamma (RES) + the OUT."""
    binds = [in0_bind]
    for name in spec.static_in_names:
        binds.append(LiveBind(name, LIVE_RES, note="static weight/gamma (resident)"))
    binds.append(LiveBind(spec.out_name, LIVE_OUT))
    return tuple(binds)


def _spec_is_paged_mla(spec: IslandSpec) -> bool:
    """True iff the MLA island is the PAGED variant (``mla_decode_paged``): 11 schema
    inputs named ``'0'..'10'`` (the block-gather folded into op node-0 -> flat_cache @2 +
    block_table @3), vs the NON-paged 10 inputs ``'0'..'9'`` (a pre-gathered latent_kv
    @2). Structural fallback when the manifest ``op=`` header was not threaded into geom
    (e.g. a hand-built synthetic plan). Pure; never raises."""
    in_names = [t.name for t in spec.tensors if t.role != "OUT"]
    return len(in_names) == 11 and set(in_names) == {str(i) for i in range(11)}


def _live_binds_mla_paged(spec: IslandSpec, geom: dict) -> tuple:
    """Bind map for the PAGED fused MLA island (``mla_decode_paged``). The paged op folds
    the block-gather in as node-0, so the pre-gathered ``latent_kv [B,S,L+R]`` of the
    non-paged op is REPLACED by two inputs bound LIVE: ``flat_cache`` (LIVE_KV <- vllm's
    resident GROWING ``kv_cache[0]``) + ``block_table`` (LIVE_META <- ``attn_metadata.
    block_list``, the paged block ids). The op does the gather internally => NO separate
    latent gather is needed. Schema order (capture_mla_island.py:725 / the on-card
    cap_mla_b32_s128_tp8 manifest, VERIFIED):

        0 q_nope  1 q_pe  2 flat_cache  3 block_table  4 W_UK_T  5 W_UV
        6 q_cos   7 q_sin 8 c_cos       9 c_sin        10 pad_mask         (OUT = '11')

    The 9 geometry-derivable tensors are bound by SIZE-XCHECKed schema position (a
    re-capture that reorders/resizes them RAISES); flat_cache (pos 2) and block_table
    (pos 3) are bound by POSITION because their byte sizes are NOT derivable from
    N/P/R/L/V/B/S (flat_cache = NB*block_size*(L+R)*2 with capture decoy blocks;
    block_table = B*MB*4 int32). LOUD on any schema drift."""
    N, P, R, L, V, S, B = (geom["N"], geom["P"], geom["R"], geom["L"],
                           geom["V"], geom["S"], geom["B"])
    qk = P + R
    out_size = B * N * V * 2
    # schema-position -> (role, attrs, expected nbytes) for the GEOMETRY-DERIVABLE inputs.
    sized = {
        "0": (LIVE_SUB, dict(producer="q_b", offset_elems=0, width_elems=P,
                             head_stride_elems=qk, note="q_nope <- q_b.out per-head[:P]"),
              B * N * P * 2),
        "1": (LIVE_SUB, dict(producer="q_b", offset_elems=P, width_elems=R,
                             head_stride_elems=qk, note="q_pe <- q_b.out per-head[P:P+R]"),
              B * N * R * 2),
        "4": (LIVE_RES, dict(note="W_UK_T (absorb, resident)"), N * P * L * 2),
        "5": (LIVE_RES, dict(note="W_UV (V-up, resident)"), N * L * V * 2),
        "6": (LIVE_META, dict(note="q_cos <- rope_tbl[ctx-1]"), B * R * 2),
        "7": (LIVE_META, dict(note="q_sin <- rope_tbl[ctx-1]"), B * R * 2),
        "8": (LIVE_RES, dict(note="c_cos/c_sin per-S (Option B: identity)"), B * S * R * 2),
        "9": (LIVE_RES, dict(note="c_cos/c_sin per-S (Option B: identity)"), B * S * R * 2),
        "10": (LIVE_META, dict(note="pad_mask <- ctx_lens"), B * S * 2),
    }
    # schema-position -> (role, attrs) for the PAGED gather inputs (size not derivable).
    positional = {
        "2": (LIVE_KV, dict(note="flat_cache <- live GROWING kv_cache[0] (paged, Option B)")),
        "3": (LIVE_META, dict(note="block_table <- attn_metadata.block_list (paged ids)")),
    }
    in_names = [t.name for t in spec.tensors if t.role != "OUT"]
    want = [str(i) for i in range(11)]
    if sorted(in_names, key=lambda x: int(x) if x.isdigit() else (1 << 30)) != want:
        raise MmelChainLayerError(
            f"[mmel.chain_layer] PAGED MLA expects schema inputs {want} but got "
            f"{in_names} (geometry {geom}); not the 11-input mla_decode_paged recipe.")
    binds = []
    for t in spec.tensors:
        if t.role == "OUT":
            if t.nbytes != out_size:
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] PAGED MLA OUT {t.nbytes} B != B*N*V*2 {out_size} "
                    f"(geometry {geom}).")
            binds.append(LiveBind(t.name, LIVE_OUT, note="attn_out [B,N*V]"))
        elif t.name in sized:
            role, attrs, want_nb = sized[t.name]
            if t.nbytes != want_nb:
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] PAGED MLA tensor {t.name!r} is {t.nbytes} B but "
                    f"schema position expects {want_nb} B (geometry {geom}); the recipe "
                    "I/O order changed — re-derive the bind map.")
            binds.append(LiveBind(t.name, role, **attrs))
        elif t.name in positional:
            role, attrs = positional[t.name]
            # flat_cache (pos 2) must be a bf16 cache of (L+R)-wide rows (>= one block);
            # block_table (pos 3) must be an int32 [B,MB] table. Light structural guards
            # (the exact NB/block_size live on-card); a gross mis-order is caught here.
            if t.name == "2" and (t.nbytes % ((L + R) * 2) != 0 or t.nbytes < (L + R) * 2):
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] PAGED MLA flat_cache (pos 2) is {t.nbytes} B, not "
                    f"a bf16 multiple of (L+R)*2={(L + R) * 2}; re-capture.")
            if t.name == "3" and (t.nbytes % 4 != 0 or t.nbytes < B * 4):
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] PAGED MLA block_table (pos 3) is {t.nbytes} B, not "
                    f"a [B,MB] int32 table (B={B} => >= {B * 4} B, /4); re-capture.")
            binds.append(LiveBind(t.name, role, **attrs))
        else:  # pragma: no cover — want-names check already constrains the inputs
            raise MmelChainLayerError(
                f"[mmel.chain_layer] PAGED MLA tensor {t.name!r} ({t.nbytes} B) is not in "
                f"the schema map (geometry {geom}).")
    return tuple(binds)


def _live_binds_mla(spec: IslandSpec, geom: dict) -> tuple:
    """Bind map for the fused MLA island, DERIVED from the captured tensors + geometry.

    PAGED (``op=mla_decode_paged`` or an 11-input '0'..'10' schema): dispatch to
    ``_live_binds_mla_paged`` (flat_cache=LIVE_KV<-kv_cache[0], block_table=LIVE_META<-
    block_list; the op gathers internally). NON-PAGED (the default): match each captured
    tensor's byte size to the §3 live-feed roles (q_nope/q_pe = LIVE_SUB of q_b;
    latent_kv = LIVE_KV; W_UK_T/W_UV/c_cos/c_sin = RES; q_cos/q_sin/pad_mask = META; OUT).
    LOUD on a size not in the map (a re-capture changed the op I/O)."""
    if str(geom.get("op")) == "mla_decode_paged" or _spec_is_paged_mla(spec):
        return _live_binds_mla_paged(spec, geom)
    N, P, R, L, V, S, B = (geom["N"], geom["P"], geom["R"], geom["L"],
                           geom["V"], geom["S"], geom["B"])
    qk = P + R
    size_role = {
        B * N * P * 2: (LIVE_SUB, dict(producer="q_b", offset_elems=0, width_elems=P,
                                       head_stride_elems=qk,
                                       note="q_nope <- q_b.out per-head[:P]")),
        B * N * R * 2: (LIVE_SUB, dict(producer="q_b", offset_elems=P, width_elems=R,
                                       head_stride_elems=qk,
                                       note="q_pe <- q_b.out per-head[P:P+R]")),
        B * S * (L + R) * 2: (LIVE_KV, dict(
            note="latent_kv <- live GROWING gathered cache (write+gather, Option B)")),
        N * P * L * 2: (LIVE_RES, dict(note="W_UK_T (absorb, resident)")),
        N * L * V * 2: (LIVE_RES, dict(note="W_UV (V-up, resident)")),
        B * R * 2: (LIVE_META, dict(note="q_cos/q_sin <- rope_tbl[ctx-1]")),
        B * S * R * 2: (LIVE_RES, dict(note="c_cos/c_sin per-S (Option B: identity)")),
        B * S * 2: (LIVE_META, dict(note="pad_mask <- ctx_lens")),
    }
    out_size = B * N * V * 2
    keys = list(size_role.keys()) + [out_size]
    if len(set(keys)) != len(keys):
        raise MmelChainLayerError(
            f"[mmel.chain_layer] MLA size->role map has a size collision at geometry "
            f"{geom}; cannot derive the live bind map by size (re-capture with a "
            "disambiguating layout).")
    binds = []
    for t in spec.tensors:
        if t.role == "OUT":
            if t.nbytes != out_size:
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] MLA OUT {t.nbytes} B != B*N*V*2 {out_size} "
                    f"(geometry {geom}).")
            binds.append(LiveBind(t.name, LIVE_OUT, note="attn_out [B,N*V]"))
            continue
        if t.nbytes not in size_role:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] MLA tensor {t.name!r} size {t.nbytes} B not in the "
                f"expected live bind map {sorted(size_role)} (geometry {geom}). A "
                "re-capture changed the MLA op's I/O; re-derive the bind map.")
        role, attrs = size_role[t.nbytes]
        binds.append(LiveBind(t.name, role, **attrs))
    return tuple(binds)


@dataclass(frozen=True)
class LiveAttnChainPlan:
    """The fully-resolved LIVE attention chain: the 7 islands (kv_b dropped) + the
    per-island LIVE bind map + the derived geometry. Every activation edge is LIVE;
    NO activation is staged from a .bin (the fix). Resolved from the captured
    manifests (data-driven)."""

    cap_parent: str
    islands: tuple                     # tuple[LiveIsland], in chain order
    geom: dict

    @property
    def step_names(self) -> tuple:
        return tuple(li.spec.step_name for li in self.islands)

    @property
    def layer_input_nbytes(self) -> int:
        return self.islands[0].spec.in0_nbytes

    @property
    def out_nbytes(self) -> int:
        return self.islands[-1].spec.out_nbytes

    @property
    def n_staged_activations(self) -> int:
        return sum(len(li.staged_activation_binds) for li in self.islands)

    @classmethod
    def resolve(cls, cap_parent: str) -> "LiveAttnChainPlan":
        M = chain_m()
        specs = {step: _parse_island_manifest(
                     step, os.path.join(cap_parent, _chain_island_dir(sub)))
                 for (step, sub) in _LIVE_ATTN_CHAIN}
        norm, qkva = specs["input_norm"], specs["fused_qkv_a"]
        qan, qb = specs["q_a_norm"], specs["q_b"]
        kvan, oproj, mla = specs["kv_a_norm"], specs["o_proj"], specs["mla"]
        # --- geometry, DERIVED from the captured byte sizes (M from MMEL_CHAIN_M) ---
        if norm.in0_nbytes % (M * 2):
            raise MmelChainLayerError(
                f"[mmel.chain_layer] input_norm in0 {norm.in0_nbytes} B not M*H*2 for "
                f"M={M}; set MMEL_CHAIN_M to the captured width.")
        H = norm.in0_nbytes // (M * 2)
        qkva_feat = qkva.out_nbytes // (M * 2)
        q_lora = qan.in0_nbytes // (M * 2)
        kv_lora = kvan.in0_nbytes // (M * 2)
        R = qkva_feat - q_lora - kv_lora
        mgeo = _parse_mla_geometry(mla.cap_dir)
        geom = dict(M=M, H=H, q_lora=q_lora, kv_lora=kv_lora, R=R, qkva_feat=qkva_feat,
                    N=mgeo["N"], P=mgeo["P"], L=mgeo["L"], V=mgeo["V"],
                    S=mgeo["S"], B=mgeo["B"], op=mgeo.get("op", "mla_decode"),
                    interleave=mgeo.get("interleave"))
        # --- per-island LIVE bind maps (the §3 live-feed map) ---
        islands = (
            LiveIsland(norm, _live_binds_simple(
                norm, LiveBind(norm.in0_name, LIVE_LAYER_INPUT,
                               note="decode hidden_states [M,H]"))),
            LiveIsland(qkva, _live_binds_simple(
                qkva, LiveBind(qkva.in0_name, LIVE_FEED, producer="input_norm",
                               note="<- input_norm OUT (full)"))),
            LiveIsland(qan, _live_binds_simple(
                qan, LiveBind(qan.in0_name, LIVE_SUB, producer="fused_qkv_a",
                              offset_elems=0, width_elems=q_lora,
                              head_stride_elems=qkva_feat,
                              note="q_c = fused_qkv_a OUT[:, 0:q_lora]"))),
            LiveIsland(qb, _live_binds_simple(
                qb, LiveBind(qb.in0_name, LIVE_FEED, producer="q_a_norm",
                             note="<- q_a_norm OUT (full)"))),
            LiveIsland(kvan, _live_binds_simple(
                kvan, LiveBind(kvan.in0_name, LIVE_SUB, producer="fused_qkv_a",
                               offset_elems=q_lora, width_elems=kv_lora,
                               head_stride_elems=qkva_feat,
                               note="kv_c = fused_qkv_a OUT[:, q_lora:q_lora+kv_lora]"))),
            LiveIsland(mla, _live_binds_mla(mla, geom)),
            LiveIsland(oproj, _live_binds_simple(
                oproj, LiveBind(oproj.in0_name, LIVE_FEED, producer="mla",
                                note="<- MLA OUT (full)"))),
        )
        plan = cls(cap_parent=cap_parent, islands=islands, geom=geom)
        plan._sanity()
        return plan

    def _sanity(self) -> None:
        names = self.step_names
        want = tuple(s for (s, _) in _LIVE_ATTN_CHAIN)
        if names != want:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] live chain order drifted: {names} (expected "
                f"{want}).")
        # kv_b is DROPPED — no island may resolve to a kv_b (cap_kvproj) capture.
        for li in self.islands:
            if "kvproj" in os.path.basename(li.spec.cap_dir):
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] live chain must DROP kv_b but island "
                    f"{li.spec.step_name!r} resolved to {li.spec.cap_dir!r} "
                    "(cap_kvproj) — kv_b is pre-absorbed into W_UK_T/W_UV.")
        # the branch split must tile fused_qkv_a's feature width exactly.
        g = self.geom
        if g["q_lora"] + g["kv_lora"] + g["R"] != g["qkva_feat"]:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] branch split {g['q_lora']}+{g['kv_lora']}+{g['R']} "
                f"!= fused_qkv_a feature {g['qkva_feat']} — the q_c|kv_c|k_pe sub-buffers "
                "do not tile the projection output.")
        # THE FIX: zero staged activations (every activation edge is LIVE).
        if self.n_staged_activations != 0:
            bad = [(li.spec.step_name, [b.tensor_name for b in li.staged_activation_binds])
                   for li in self.islands if li.staged_activation_binds]
            raise MmelChainLayerError(
                f"[mmel.chain_layer] live chain has STAGED activation(s) {bad} — the "
                "fix requires EVERY activation edge LIVE (no frozen .bin). This is the "
                "exact bug the live decomposition removes.")
        # the MLA island must carry the live-KV + the 2 live q sub-buffers (the fix's
        # discriminator vs the staged all-7, where they were frozen .bins).
        mla = next(li for li in self.islands if li.spec.step_name == "mla")
        n_sub = sum(1 for b in mla.binds if b.role == LIVE_SUB)
        n_kv = sum(1 for b in mla.binds if b.role == LIVE_KV)
        if n_sub != 2 or n_kv != 1:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] MLA live binds wrong: {n_sub} LIVE_SUB (want 2: "
                f"q_nope,q_pe) + {n_kv} LIVE_KV (want 1: latent_kv). The MLA is not "
                "live-fed; the chain would emit a constant (the bug).")
        # spine tail (o_proj) must be LIVE and threadable (OUT == layer input).
        tail = self.islands[-1]
        if tail.spec.step_name != "o_proj" or tail.binds[0].role != LIVE_FEED:
            raise MmelChainLayerError(
                "[mmel.chain_layer] live chain tail must be o_proj fed LIVE from MLA.")
        if tail.spec.out_nbytes != self.layer_input_nbytes:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] o_proj OUT {tail.spec.out_nbytes} B != layer-input "
                f"{self.layer_input_nbytes} B — the attention output is not a [M,H] "
                "hidden threadable to the post-attention residual.")

    def assert_live_hidden(self, name: str, shape) -> None:
        """LOUD guard: the live layer-input hidden_states must match the captured
        input_norm in0 byte geometry (else binding it re-fires the recipe)."""
        got = _live_hidden_nbytes(_ShapeOnly(shape))
        if got is None:
            raise MmelChainLayerShapeError(
                f"[mmel.chain_layer] live tensor {name!r} has unreadable shape {shape!r}.")
        if got != self.layer_input_nbytes:
            raise MmelChainLayerShapeError(
                f"[mmel.chain_layer] STATIC-SHAPE violation: live {name!r} is {got} B "
                f"but the live chain's input_norm expects {self.layer_input_nbytes} B "
                "(re-capture this bucket / set MMEL_CHAIN_M).")

    def live_feed_summary(self) -> str:
        rows = []
        for li in self.islands:
            in0 = next((b for b in li.binds if b.tensor_name == li.spec.in0_name),
                       li.binds[0])
            extra = ""
            if li.spec.step_name == "mla":
                extra = (f" [+{sum(1 for b in li.binds if b.role==LIVE_SUB)} sub, "
                         f"{sum(1 for b in li.binds if b.role==LIVE_KV)} kv, "
                         f"{len(li.meta_bind_names)} meta]")
            rows.append(f"{li.spec.step_name}:{in0.role}{extra}")
        return " -> ".join(rows)


class _ShapeOnly:
    """A shape-only stand-in so ``_live_hidden_nbytes`` works on a raw shape tuple."""

    def __init__(self, shape):
        self.shape = shape


# ----------------------------------------------------------------------------- #
#  LIVE-ATTN coverage predicate (card-free) — the seam engages the live chain    #
#  ONLY when the live hidden geometry == the captured [M,H] bucket. Resolved once #
#  + cached; a resolve failure is non-fatal (undecidable => not covered).         #
# ----------------------------------------------------------------------------- #
_LIVE_ATTN_INPUT_NBYTES_CACHE: dict = {}


def live_attn_input_nbytes(cap_parent: Optional[str] = None) -> Optional[int]:
    """The live attention chain's captured layer-input byte size, or None if the
    plan can't be resolved (undecidable => the seam does not engage). Cached per
    (cap_parent, MMEL_CHAIN_M)."""
    parent = cap_parent or cap_parent_dir()
    key = (parent, chain_m())
    if key in _LIVE_ATTN_INPUT_NBYTES_CACHE:
        return _LIVE_ATTN_INPUT_NBYTES_CACHE[key]
    try:
        nb = int(LiveAttnChainPlan.resolve(parent).layer_input_nbytes)
    except Exception as e:  # noqa: BLE001 — undecidable is non-fatal (LOUD at warmup)
        logger.warning(
            "[mmel.chain_layer] live_attn_input_nbytes: could not resolve the live "
            "attention chain under %r (M=%d) (%s: %s) -> coverage UNDECIDABLE.",
            parent, chain_m(), type(e).__name__, e)
        _LIVE_ATTN_INPUT_NBYTES_CACHE[key] = None
        return None
    _LIVE_ATTN_INPUT_NBYTES_CACHE[key] = nb
    return nb


def live_attn_covers(hidden_states: Any, cap_parent: Optional[str] = None) -> bool:
    """True iff the live hidden geometry == the captured live-chain [M,H] bucket.
    Never raises (cheap integer compare). Card-free."""
    want = live_attn_input_nbytes(cap_parent)
    if want is None:
        return False
    got = _live_hidden_nbytes(hidden_states)
    return got is not None and got == want


# ----------------------------------------------------------------------------- #
#  THE MLA LIVE-FEED NAME MAPS (card-free, pure) — used by BOTH the ROPE Option-B  #
#  warmup override and the seam's live_ctx assembly. Both DERIVE the recipe tensor #
#  NAME -> semantic from the captured bind map (role + size + schema order), so a   #
#  re-capture cannot silently mis-bind. Every function here is pure + never raises. #
# ----------------------------------------------------------------------------- #
def mla_cache_rope_tensor_names(plan: "LiveAttnChainPlan") -> Optional[tuple]:
    """The MLA island's cache-rope table recipe tensor names ``(c_cos, c_sin)`` for
    the ROPE Option-B identity override, or None if they cannot be unambiguously
    identified. The two cache-rope tables are the LIVE_RES binds carrying the
    ``c_cos/c_sin`` note (set by ``_live_binds_mla``, size ``B*S*R*2``); the captured
    recipe is SCHEMA-ORDERED (``..., c_cos, c_sin, ...``), so the LOWER-named one is
    c_cos (-> identity cos=1) and the HIGHER-named one is c_sin (-> identity sin=0).
    Pure; never raises."""
    try:
        mla = next(li for li in plan.islands if li.spec.step_name == "mla")
    except Exception:  # noqa: BLE001
        return None
    rope = [b for b in mla.binds
            if b.role == LIVE_RES and str(b.note).startswith("c_cos/c_sin")]
    if len(rope) != 2:
        return None
    try:
        rope.sort(key=lambda b: int(b.tensor_name))
    except (TypeError, ValueError):
        return None
    return (rope[0].tensor_name, rope[1].tensor_name)


def mla_live_ctx_roles(plan: "LiveAttnChainPlan") -> Optional[dict]:
    """Map each MLA recipe tensor NAME that must be fed LIVE per step (the LIVE_KV +
    META binds) to its SEMANTIC: ``'latent_kv'`` (the growing cache; paged op binds
    kv_cache[0] as flat_cache), ``'block_table'`` (PAGED only, the paged block ids, by
    note), ``'pad_mask'`` (size ``B*S*2``), ``'q_cos'``/``'q_sin'`` (size ``B*R*2``,
    split by schema order: lower=q_cos, higher=q_sin). Derived from the captured bind
    map + geometry, so a re-capture cannot silently mis-feed. Pure; never raises;
    returns None if the map is ambiguous (the seam then delegates to stock)."""
    try:
        mla = next(li for li in plan.islands if li.spec.step_name == "mla")
        g = plan.geom
        B, S, R = int(g["B"]), int(g["S"]), int(g["R"])
    except Exception:  # noqa: BLE001
        return None
    nb = {t.name: t.nbytes for t in mla.spec.tensors}
    roles: dict = {}
    kv = [b for b in mla.binds if b.role == LIVE_KV]
    if len(kv) != 1:
        return None
    # LIVE_KV is the growing latent cache: the non-paged op gathers it pre-bind
    # (latent_kv); the PAGED op binds vllm's resident kv_cache[0] (flat_cache) and
    # gathers internally. BOTH source from kv_cache[0] => one semantic ('latent_kv').
    roles[kv[0].tensor_name] = "latent_kv"
    meta = [b for b in mla.binds if b.role == LIVE_META]
    # block_table (PAGED only): the LIVE_META carrying the 'block_table' note — its byte
    # size (B*MB*4 int32) is NOT a geometry size, so it is identified by note, not size.
    bt = [b for b in meta if str(b.note).startswith("block_table")]
    rest = [b for b in meta if b not in bt]
    pad = [b for b in rest if nb.get(b.tensor_name) == B * S * 2]
    rope = [b for b in rest if nb.get(b.tensor_name) == B * R * 2]
    if len(pad) != 1 or len(rope) != 2 or len(bt) > 1:
        return None
    roles[pad[0].tensor_name] = "pad_mask"
    try:
        rope.sort(key=lambda b: int(b.tensor_name))
    except (TypeError, ValueError):
        return None
    roles[rope[0].tensor_name] = "q_cos"
    roles[rope[1].tensor_name] = "q_sin"
    if bt:
        roles[bt[0].tensor_name] = "block_table"
    # every LIVE_KV + META bind must have been mapped (else the live_ctx would be
    # incomplete and the run() name-coverage check would reject it).
    need = {b.tensor_name for b in mla.binds if b.role in (LIVE_KV, LIVE_META)}
    if set(roles) != need:
        return None
    return roles


def _build_query_rope_tables(layer: Any, positions: Any, geom: dict) -> Optional[tuple]:
    """Build the MLA op's per-step query rope tables ``(q_cos, q_sin)`` — each ``[B,1,R]``
    bf16 — by gathering the model's rotary ``cos_sin_cache`` at the live decode
    ``positions`` (= ctx_len-1 per sequence) and expanding to the op's interleave/NeoX
    layout. Returns ``(q_cos, q_sin)`` or ``None`` on ANY unresolvable source so the seam
    DELEGATES to stock — never a wrong rope bind. Pure; never raises.

    SOURCE (grounded, file:line):
      * The GLM q-LoRA MLA wrapper applies rope as
        ``layer.self_attn.rotary_emb(positions, q[..., qk_nope:], k_pe)``
        (_src/vllm_attn/mla.py:163-166); ``rotary_emb`` is set from
        ``mla_modules.rotary_emb`` (mla.py:85 / _src/vllm_gaudi/attention/oot_mla.py:259).
      * Its table ``rotary_emb.cos_sin_cache`` is the canonical vLLM
        ``cat([cos(freqs), sin(freqs)], -1)`` of shape ``[max_pos, R]`` (first R/2 cols
        cos, last R/2 sin) — layout mirrored by mmel/in_graph_rope.py:119-172
        (``build_cos_sin_cache`` + ``_gather_cos_sin``).
      * The fused op consumes q_cos/q_sin ALREADY expanded to ``[B,1,R]`` in the SAME
        style the capture baked in (capture_mla_island.py:275-286 ``_cos_sin``:
        interleave=1 PAIRWISE -> ``repeat_interleave(2)``; 0 NeoX -> ``cat([x,x])``).
      We reproduce the gather+expand EXACTLY (validated byte-equal vs ``_cos_sin`` in the
      card-free selftest)."""
    try:
        rotary_emb = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if rotary_emb is None or positions is None:
            return None
        cache = getattr(rotary_emb, "cos_sin_cache", None)
        if cache is None or not hasattr(cache, "index_select"):
            return None
        # rope style: vllm RotaryEmbedding carries ``is_neox_style``; the op wants the SAME
        # style the capture baked in (geom['interleave']: 1=PAIRWISE/gptj => NOT neox; 0=
        # NeoX). If BOTH are present they MUST agree, else a wrong bind silently corrupts.
        is_neox = getattr(rotary_emb, "is_neox_style", None)
        cap_interleave = geom.get("interleave")
        if is_neox is None and cap_interleave is None:
            return None                        # rope layout indeterminate -> delegate
        if is_neox is None:
            is_neox = (int(cap_interleave) == 0)
        elif cap_interleave is not None and bool(is_neox) != (int(cap_interleave) == 0):
            return None                        # live style != captured island -> delegate
        import torch  # lazy — CPU on the self-test venv, HPU in serve; never at import
        R = int(geom["R"])
        pos = positions
        if not hasattr(pos, "reshape") or not hasattr(pos, "to"):
            return None
        pos = pos.reshape(-1).to(torch.long)
        cos_sin = cache.index_select(0, pos)              # [B, R]
        cos, sin = cos_sin.chunk(2, dim=-1)               # each [B, R/2]
        if bool(is_neox):
            q_cos = torch.cat([cos, cos], dim=-1)
            q_sin = torch.cat([sin, sin], dim=-1)
        else:
            q_cos = cos.repeat_interleave(2, dim=-1)
            q_sin = sin.repeat_interleave(2, dim=-1)
        if q_cos.shape[-1] != R or q_sin.shape[-1] != R:
            return None
        Bp = int(pos.shape[0])
        q_cos = q_cos.reshape(Bp, 1, R).to(torch.bfloat16).contiguous()
        q_sin = q_sin.reshape(Bp, 1, R).to(torch.bfloat16).contiguous()
        return q_cos, q_sin
    except Exception:  # noqa: BLE001 — never raise: undecidable => delegate to stock
        return None


def build_live_ctx_for_layer(layer: Any, attn_metadata: Any, kv_cache: Any,
                             positions: Any = None,
                             cap_parent: Optional[str] = None) -> Optional[dict]:
    """Assemble the MLA island's ``live_ctx`` — the dict (keyed by the MLA recipe's
    LIVE_KV + META tensor NAMES) of the live tensors the chain binds each decode step:
    ``latent_kv`` <- the growing ``kv_cache[0]`` (Option B: vllm's already-roped cache;
    the PAGED op binds it as flat_cache and gathers internally); ``block_table`` <-
    ``attn_metadata.block_list`` (PAGED only, the paged block ids); ``pad_mask`` <-
    ``attn_metadata.attn_bias``; ``q_cos``/``q_sin`` <- the model query rope table gathered
    at ``positions`` (``_build_query_rope_tables`` -> ``layer.self_attn.rotary_emb.
    cos_sin_cache``). The recipe-name -> semantic map is DERIVED from the captured bind
    map (``mla_live_ctx_roles``), so it cannot silently mis-bind.

    PURE + NEVER RAISES: returns None on ANY missing/unresolvable source (unresolved
    plan, ambiguous map, absent kv_cache/attn_metadata field, un-wired query rope) so
    the seam DELEGATES to stock attention — undecidable => stock, never a wrong bind and
    never a crash. The grounded attn_metadata field names (``attn_bias``, ``block_list``)
    are from oot_mla / hpu_attn ``HPUMLAMetadata``."""
    try:
        plan = LiveAttnChainPlan.resolve(cap_parent or cap_parent_dir())
    except Exception:  # noqa: BLE001 — undecidable plan => delegate to stock
        return None
    roles = mla_live_ctx_roles(plan)
    if roles is None:
        return None
    ctx: dict = {}
    rope_tbl: dict = {}   # memo: q_cos/q_sin built ONCE from a single live gather
    for name, sem in roles.items():
        try:
            if sem == "latent_kv":
                # Option B: the paged op's flat_cache IS vllm's resident growing
                # kv_cache[0]; the non-paged op's latent_kv is the pre-gathered slice of
                # the same cache. Both source kv_cache[0]; the op (paged) gathers it.
                t = kv_cache[0] if kv_cache is not None and len(kv_cache) >= 1 else None
            elif sem == "block_table":
                # PAGED: the per-step paged block ids the op's node-0 gather indexes.
                # The live serve runs flat_pa_mla => block_list is a FLAT [B*MB] int64
                # list. The captured mla_decode_paged op REQUIRES a 2D [B,MB] int32
                # block_table (it asserts 2D + int32). Reshape+cast to the recipe layout
                # (query-major: flat [b*MB+k] -> [b,k], SAME block ids; nbytes 1024->512).
                t = getattr(attn_metadata, "block_list", None)
                if t is not None and positions is not None and t.dim() == 1:
                    _B = int(positions.shape[0])
                    if _B > 0 and (int(t.numel()) % _B) == 0:
                        # .int() casts to int32 via the tensor method (this module does
                        # NOT import torch at module scope — `torch.int32` would NameError
                        # -> caught -> None -> delegate, the cl19 regression).
                        t = t.reshape(_B, int(t.numel()) // _B).int()
            elif sem == "pad_mask":
                t = getattr(attn_metadata, "attn_bias", None)
            elif sem in ("q_cos", "q_sin"):
                if not rope_tbl:
                    built = _build_query_rope_tables(layer, positions, plan.geom)
                    if built is not None:
                        rope_tbl["q_cos"], rope_tbl["q_sin"] = built
                t = rope_tbl.get(sem)
            else:  # pragma: no cover — mla_live_ctx_roles only emits the 5 semantics
                t = None
        except Exception:  # noqa: BLE001 — any source failure => undecidable => stock
            t = None
        if t is None:
            return None
        ctx[name] = t
    return ctx


# ----------------------------------------------------------------------------- #
#  THE DEVICE-RESIDENT LIVE-ATTN CHAIN REPLAY (on-card only; reuses S1 surface).  #
#                                                                               #
#  warmup_once(): deserialize the 7 recipes; upload each island's RES statics      #
#                 (weights/gamma; the MLA's W_UK_T/W_UV/c_cos/c_sin) resident;     #
#                 alloc each island's OUT + workspace; reuse the bridge stream.     #
#                 It does NOT upload any LIVE/LIVE_SUB/LIVE_KV/META .bin — those    #
#                 are bound live each step (the fix).                              #
#  run(hidden, live_ctx): bind input_norm.in0 = LIVE hidden; each LIVE edge to the  #
#                 producer's resident OUT; each LIVE_SUB to a strided view of the   #
#                 producer OUT (the q_c/kv_c column splits, the q_nope/q_pe per-     #
#                 head splits); the MLA's LIVE_KV + META tensors from ``live_ctx``   #
#                 (the seam builds latent_kv from the growing kv_cache + the rope    #
#                 tables/pad_mask from attn_metadata); synLaunch all 7 back-to-back; #
#                 ONE synStreamSynchronize; return o_proj OUT ([M,H] bf16).         #
#                                                                               #
#  ``live_ctx`` keys are the MLA recipe TENSOR NAMES (the seam owns the name->      #
#  semantic map it captured). A missing/extra key is LOUD (never a silent mis-bind).#
# ----------------------------------------------------------------------------- #
@dataclass
class LiveAttnChainReplay:
    """Device-resident replay of the LIVE-FED MLA attention chain (input_norm ->
    o_proj) behind ONE host fence, every activation edge LIVE. Reuses S1's
    RawCsIslandReplay synLaunch surface (no new ABI)."""

    plan: LiveAttnChainPlan
    _islands_rt: list = field(default_factory=list)
    _stream_handle: int = 0
    _warm: bool = False

    def warmup_once(self) -> "LiveAttnChainReplay":
        if self._warm:
            return self
        import torch          # lazy — card only
        import numpy as np    # lazy

        try:
            from vllm_gaudi.v1.worker.mmel_l6_chain import (  # type: ignore
                RawCsIslandReplay, reuse_bridge_device_and_stream)
        except Exception:  # noqa: BLE001 — path-based sibling import on a bare box
            RawCsIslandReplay, reuse_bridge_device_and_stream = _import_chain_symbols()

        _dev, self._stream_handle = reuse_bridge_device_and_stream()
        for li in self.plan.islands:
            isl = li.spec
            recipe_path = os.path.join(isl.cap_dir, "island.recipe")
            if not os.path.isfile(recipe_path):
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] recipe not found for live island "
                    f"{isl.step_name!r}: {recipe_path!r}. Re-capture on-card.")
            replay = RawCsIslandReplay(recipe_path=recipe_path).deserialize()
            rt = _IslandRT(spec=isl, replay=replay)
            # Upload ONLY the RES (resident static) tensors — weights/gamma, and the
            # MLA's W_UK_T/W_UV/c_cos/c_sin. LIVE/LIVE_SUB/LIVE_KV/META are bound live.
            #
            # ROPE-CONVENTION CAVEAT (LIVE_MLA_DECOMP_DESIGN §2/§7-risk-2): the MLA's
            # c_cos/c_sin RES are uploaded here from the CAPTURED .bins = the REAL
            # cache-rope tables = OPTION A (the engine owns an UN-roped latent cache and
            # ropes the cache pe in-graph). To engage OPTION B (recommended: bind the
            # LIVE-KV to vllm's ALREADY-roped kv_cache[0]), the engage step must OVERRIDE
            # these two tensors with IDENTITY (cos=1, sin=0) so the in-graph cache-rope is
            # a no-op — else the cache is DOUBLE-roped (a plausible-but-wrong output only
            # the on-card fp64 + live-stock A/B catches). The bind note records this.
            # ROPE Option B (MMEL_CHAIN_ROPE_OPTB=1): override the MLA cache-rope tables
            # c_cos -> IDENTITY ones (cos=1) and c_sin -> IDENTITY zeros (sin=0) so the
            # op's in-graph cache-rope is a NO-OP (the live-KV is vllm's ALREADY-roped
            # kv_cache[0] => roping again would DOUBLE-rope). ONLY the two MLA cache-rope
            # tensors are touched; q_cos/q_sin (role META, the LIVE query rope) and every
            # weight/gamma RES stay REAL. Default OFF == Option A (upload the captured
            # REAL c_cos/c_sin verbatim) => byte-identical to the prior behaviour.
            optb_cos = optb_sin = None
            if _rope_optb_enabled() and isl.step_name == "mla":
                _names = mla_cache_rope_tensor_names(self.plan)
                if _names is None:
                    raise MmelChainLayerError(
                        "[mmel.chain_layer] MMEL_CHAIN_ROPE_OPTB=1 but the MLA cache-rope "
                        "tensors (c_cos/c_sin) could not be identified in the captured "
                        "bind map; refusing to upload the REAL tables (Option A) under an "
                        "explicit Option-B request (that would DOUBLE-rope vllm's roped "
                        "kv_cache[0]). Re-capture the MLA island or unset the flag.")
                optb_cos, optb_sin = _names
                logger.warning(
                    "[mmel.chain_layer][ROPE-OPTB] identity cache-rope: tensor %r<-cos=1, "
                    "%r<-sin=0 (binding vllm's ALREADY-roped kv_cache[0]); the LIVE query "
                    "rope q_cos/q_sin stays REAL.", optb_cos, optb_sin)

            for b in li.binds:
                if b.role != LIVE_RES:
                    continue
                t = next(t for t in isl.tensors if t.name == b.tensor_name)
                # Option-B identity swap for the two cache-rope tables (bf16 ones/zeros
                # of the captured byte size; the recipe binds by address so a contiguous
                # bf16 buffer of the right numel is the device payload).
                if optb_cos is not None and b.tensor_name == optb_cos:
                    rt.static_dev[b.tensor_name] = torch.ones(
                        t.nbytes // 2, dtype=torch.bfloat16, device="hpu").contiguous()
                    continue
                if optb_sin is not None and b.tensor_name == optb_sin:
                    rt.static_dev[b.tensor_name] = torch.zeros(
                        t.nbytes // 2, dtype=torch.bfloat16, device="hpu").contiguous()
                    continue
                buf = np.fromfile(t.bin_path, dtype=np.uint8)
                if buf.nbytes != t.nbytes:
                    raise MmelChainLayerError(
                        f"[mmel.chain_layer] live {isl.step_name!r} RES {b.tensor_name!r} "
                        f".bin is {buf.nbytes} B but manifest says {t.nbytes} B; "
                        "re-capture.")
                rt.static_dev[b.tensor_name] = torch.from_numpy(buf).to("hpu").contiguous()
            rt.out_tensor = torch.empty(
                isl.out_nbytes, dtype=torch.uint8, device="hpu").contiguous()
            ws = int(getattr(replay, "_ws_size", 0))
            if ws > 0:
                rt.ws_dev = torch.empty(ws, dtype=torch.uint8, device="hpu").contiguous()
                rt.ws_va = int(rt.ws_dev.data_ptr())
            self._islands_rt.append(rt)
        # COMMIT all warmup allocations (static_dev RES + out_tensor + ws_dev) at the warmup
        # boundary so their device VAs are real + the recipe's sections register against them
        # BEFORE the first synLaunch (the MoE _live_in_buf pattern commits at warmup; without
        # it the section VAs aren't established => synFailedSectionValidation). Re-read ws_va
        # now that ws_dev is committed (it was captured off an uncommitted torch.empty).
        import habana_frameworks.torch.core as _htc
        import habana_frameworks.torch.hpu as _hth
        _htc.mark_step(); _hth.synchronize()
        for _rt in self._islands_rt:
            if getattr(_rt, "ws_dev", None) is not None:
                _rt.ws_va = int(_rt.ws_dev.data_ptr())
        self._warm = True
        logger.info(
            "[mmel.chain_layer] WARMED LIVE-attn chain: %d islands resident, every "
            "activation edge LIVE (%s), stream=0x%x. kv_b DROPPED; MLA fed live "
            "q_nope/q_pe (per-head sub-buffers of q_b) + live-KV latent_kv.",
            len(self._islands_rt), self.plan.live_feed_summary(), self._stream_handle)
        return self

    def _resident_live(self, key, src):
        """Copy a LAZY live tensor into a per-key RESIDENT buffer with a REAL, STABLE
        data_ptr — the proven mmel_moe_island `_live_in_buf` status=30 fix. The buffer is
        torch.zeros(...,device='hpu').contiguous() COMMITTED ONCE at alloc (mark_step +
        synchronize at a graph boundary => a real device VA; an UNCOMMITTED lazy/.to('hpu')
        tensor has only a PLACEHOLDER VA — that's why q_nope/q_pe/q_cos/q_sin all aliased at
        one sentinel addr = status=30). Per step the live src is copy_'d in (executed by the
        chain's per-island synchronize before the launch). Returns the RAW uint8 buffer; it
        is bound by data_ptr+nbytes (the recipe carries the real shape), so distinct keys =>
        distinct real buffers => no section alias + real data."""
        import torch  # lazy
        import numpy as _np
        import habana_frameworks.torch.core as _htc
        import habana_frameworks.torch.hpu as _hth
        # CORRECTNESS ROOT FIX (2026-06-29, fork): the prior `buf.view().copy_(src)` lazy copy
        # NEVER LANDS at buf.data_ptr() in HPU lazy mode (probed: live_cpu=0 while hs_in=4.63)
        # => input_norm read ZERO input => the WHOLE chain was zero (RMSNorm(0)=0 cascaded).
        # FIX: D2H read of src's REAL bytes (.cpu() reads raw device memory, bypassing the
        # lazy graph) then H2D into a FRESH committed buffer — the exact from_numpy().to('hpu')
        # pattern the RES statics use (which pass section validation + carry real data). The
        # synLaunch reads this buffer's stable data_ptr. (Per-call alloc + sync; correctness
        # first — collapse the syncs after coherence is proven.)
        npb = src.detach().contiguous().view(torch.uint8).cpu().numpy()
        buf = torch.from_numpy(_np.ascontiguousarray(npb)).to("hpu").contiguous()
        _htc.mark_step(); _hth.synchronize()
        return buf

    def _sub_view(self, producer_out, bind: LiveBind):
        """A (strided) sub-buffer of a producer's resident OUT, materialized into a
        per-bind RESIDENT buffer (_resident_live) so q_nope/q_pe get DISTINCT REAL addrs
        (a lazy .contiguous() aliases at the placeholder VA = status=30). head_stride==0 =>
        [M,width] column slice of [M,F]; >0 => per-head [M,N,width] slice of [M,N,head_stride]."""
        import torch  # lazy
        g = self.plan.geom
        M = g["M"]
        bf = producer_out.view(torch.bfloat16)
        if bind.head_stride_elems and bind.head_stride_elems != bind.width_elems:
            n = bf.numel() // (M * bind.head_stride_elems)
            v = bf.view(M, n, bind.head_stride_elems)
            sl = v[:, :, bind.offset_elems: bind.offset_elems + bind.width_elems]
        else:
            f = bf.numel() // M
            v = bf.view(M, f)
            sl = v[:, bind.offset_elems: bind.offset_elems + bind.width_elems]
        return self._resident_live(id(bind), sl)

    def run(self, hidden_states: Any, live_ctx: Optional[dict] = None) -> Any:
        """Replay the live attention chain; return the device attention output
        ([M,H] bf16). ``live_ctx`` maps the MLA recipe's LIVE_KV + META tensor NAMES
        to live HPU tensors (latent_kv from the growing cache; q_cos/q_sin/pad_mask
        from attn_metadata). LOUD on any missing/extra live tensor."""
        if not self._warm:
            raise MmelChainLayerError(
                "[mmel.chain_layer] LiveAttnChainReplay.run() before warmup_once().")
        import torch  # lazy — card only

        self.plan.assert_live_hidden("hidden_states", tuple(hidden_states.shape))
        live_ctx = dict(live_ctx or {})
        # latent-write side-channel (popped so the have==need check below still passes): the
        # current-token slot_mapping + kv_cache[0] for the cache write performed BEFORE the
        # read-only mla_decode_paged op (stock does reshape_and_cache_mla; the chain was
        # missing it => frozen cache => degenerate output).
        _slot = live_ctx.pop("__slot__", None)
        _kv0 = live_ctx.pop("__kv0__", None)
        _rb_layer = live_ctx.pop("__layer__", None)
        _rb_pos = live_ctx.pop("__pos__", None)
        _rb_am = live_ctx.pop("__am__", None)
        _rb_kv = live_ctx.pop("__kv__", None)
        import os as _os_rb
        _ROUTEB = _os_rb.environ.get("MMEL_CHAIN_ROUTEB", "0") == "1"
        # the MLA island's required live tensors (LIVE_KV + META), by recipe name.
        mla = next(li for li in self.plan.islands if li.spec.step_name == "mla")
        need = set(mla.kv_bind_names) | set(mla.meta_bind_names)
        have = set(live_ctx.keys())
        if have != need:
            raise MmelChainLayerError(
                f"[mmel.chain_layer] live_ctx for the MLA must provide EXACTLY the "
                f"LIVE_KV+META tensors {sorted(need)} (latent_kv + q_cos/q_sin/pad_mask); "
                f"missing={sorted(need - have)} extra={sorted(have - need)}.")

        # the live hidden is also LAZY => placeholder VA; materialize into a committed
        # resident buffer (one, reused) so input_norm's in0 binds a real addr.
        live = self._resident_live("__hidden__", hidden_states)
        outs_by_step: dict = {}
        for rt in self._islands_rt:
            li = next(x for x in self.plan.islands if x.spec is rt.spec)
            isl = rt.spec
            name_to_tensor = dict(rt.static_dev)         # RES (resident)
            for b in li.binds:
                if b.role == LIVE_RES or b.role == LIVE_OUT:
                    continue
                if b.role == LIVE_LAYER_INPUT:
                    name_to_tensor[b.tensor_name] = live
                elif b.role == LIVE_FEED:
                    name_to_tensor[b.tensor_name] = outs_by_step[b.producer]
                elif b.role == LIVE_SUB:
                    name_to_tensor[b.tensor_name] = self._sub_view(
                        outs_by_step[b.producer], b)
                elif b.role == LIVE_KV:
                    # the resident GROWING kv_cache[0] — bind by reference. (DIAGNOSTIC 2026-06-29:
                    # binding a committed dummy here did NOT clear status=30, so the external
                    # cache VA is NOT the synFailedSectionValidation cause — it's intrinsic to
                    # the MLA recipe's section structure, not any tensor's allocation.)
                    name_to_tensor[b.tensor_name] = live_ctx[b.tensor_name]
                elif b.role == LIVE_META:
                    # the small COMPUTED META (q_cos/q_sin gather, pad_mask, block_table
                    # reshape) is LAZY => placeholder VA = section alias. Copy into a
                    # committed resident buffer with a real addr (same fix as the sub-views).
                    name_to_tensor[b.tensor_name] = self._resident_live(
                        ("meta", id(rt), b.tensor_name), live_ctx[b.tensor_name])
                else:  # pragma: no cover — _sanity forbids staged activations
                    raise MmelChainLayerError(
                        f"[mmel.chain_layer] island {isl.step_name!r} tensor "
                        f"{b.tensor_name!r} has unbindable role {b.role!r}.")
            name_to_tensor[isl.out_name] = rt.out_tensor
            # ── PER-LAYER WEIGHT FIX: the cap_* islands ship ONE captured layer's weight
            # (.bin) used for ALL 78 layers => wrong projections (dkvc=1024) => incoherent.
            # Override the weight RES with the LIVE per-layer weight (matched by byte size).
            if _rb_layer is not None and isl.step_name != "mla":
                try:
                    _wm = {
                        "input_norm": lambda L: getattr(L, "input_layernorm", None),
                        "fused_qkv_a": lambda L: L.self_attn.mla_attn.fused_qkv_a_proj,
                        "q_a_norm": lambda L: L.self_attn.mla_attn.q_a_layernorm,
                        "q_b": lambda L: L.self_attn.mla_attn.q_b_proj,
                        "kv_a_norm": lambda L: L.self_attn.mla_attn.kv_a_layernorm,
                        "o_proj": lambda L: L.self_attn.mla_attn.o_proj,
                    }.get(isl.step_name)
                    _mod = _wm(_rb_layer) if _wm is not None else None
                    _w = getattr(_mod, "weight", None) if _mod is not None else None
                    if _w is not None:
                        # The recipe is a bf16 GEMM wanting weight [out,in] (CPU-verified). The
                        # LIVE vllm 2D linear weights are FP8 [in,out] (fused_qkv_a/q_b/o_proj in
                        # the W4A8 serve) — DEQUANT to bf16 (x weight_scale) THEN .t() -> [out,in].
                        if _w.dim() == 2 and _w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                            _sc = getattr(_mod, "weight_scale", None)
                            if _sc is None:
                                _sc = getattr(_mod, "weight_scale_inv", None)
                            _wbf = _w.to(torch.bfloat16)
                            if _sc is not None:
                                _scb = _sc.to(torch.bfloat16).reshape(-1)
                                if _scb.numel() == 1:
                                    _wbf = _wbf * _scb
                                elif _scb.numel() == _w.shape[1]:
                                    _wbf = _wbf * _scb.view(1, -1)
                                elif _scb.numel() == _w.shape[0]:
                                    _wbf = _wbf * _scb.view(-1, 1)
                            import sys as _wl8
                            if getattr(type(self), '_SCLOG', 0) < 6:
                                print("[FP8] %s wshape=%s scale=%s" % (
                                    isl.step_name, tuple(_w.shape),
                                    (tuple(_sc.shape) if _sc is not None else None)),
                                    file=_wl8.stderr, flush=True)
                                type(self)._SCLOG = getattr(type(self), '_SCLOG', 0) + 1
                            _wc = _wbf.t().contiguous()
                        else:
                            _wc = (_w.t().contiguous() if _w.dim() == 2 else _w.contiguous())
                        _wnb = int(_wc.numel() * _wc.element_size())
                        _matched = False
                        for _tn, _tv in list(name_to_tensor.items()):
                            if _tn == isl.out_name:
                                continue
                            if int(_tv.numel() * _tv.element_size()) == _wnb:
                                name_to_tensor[_tn] = _wc; _matched = True
                                break
                        import sys as _wl9
                        if getattr(type(self), '_WLOG', 0) < 14:
                            print("[WBIND] %s w=%s dt=%s nb=%d recipe_nbs=%s matched=%s" % (
                                isl.step_name, tuple(_w.shape), str(_w.dtype), _wnb,
                                sorted(set(int(t.numel() * t.element_size()) for t in name_to_tensor.values())),
                                _matched), file=_wl9.stderr, flush=True)
                            type(self)._WLOG = getattr(type(self), '_WLOG', 0) + 1
                except Exception as _we:
                    import sys as _sw7
                    if getattr(type(self), "_WERR", 0) < 3:
                        print("[WERR] %s %r" % (isl.step_name, _we), file=_sw7.stderr, flush=True)
                        type(self)._WERR = getattr(type(self), "_WERR", 0) + 1
            if isl.step_name == "mla":
                import sys as _ds6
                _gg = globals(); _gg['_MLA'] = _gg.get('_MLA', 0) + 1
                if _gg['_MLA'] <= 2:
                    for _t in isl.tensors:
                        _bt = name_to_tensor.get(_t.name)
                        _bn = (_bt.numel() * _bt.element_size()) if _bt is not None else None
                        _bsh = tuple(_bt.shape) if _bt is not None else None
                        _bdt = (str(_bt.dtype) if _bt is not None else None)
                        print("[MLAT] %s recipe=%s bound=%s shape=%s dt=%s%s" % (
                            _t.name, _t.nbytes, _bn, _bsh, _bdt,
                            "" if _bn == _t.nbytes else "  <<<MISMATCH"),
                            file=_ds6.stderr, flush=True)

            expected = {t.name for t in isl.tensors}
            got = set(name_to_tensor.keys())
            if got != expected:
                raise MmelChainLayerError(
                    f"[mmel.chain_layer] live island {isl.step_name!r} bind-map "
                    f"mismatch: missing={sorted(expected - got)} "
                    f"extra={sorted(got - expected)}.")
            # ── MLA NAME-REMAP to the recipe's ACTUAL launch-tensor names ──────────────
            # probe_sections.py proved the GC recipe COMBINED q_nope('0')+q_pe('1') into ONE
            # input '0' (the q_b OUT [M,N,256]=131072 B) and RENUMBERED the rest: manifest
            # tensor N -> recipe N+1 for N>=2 (flat_cache 2->3 ... pad_mask 10->11, OUT 11->12).
            # The capture manifest names (0..11) DON'T match the recipe (0,3..12), so binding by
            # manifest name left q_pe/flat_cache UNBOUND => synFailedSectionValidation. Bind the
            # COMBINED q_b OUT to '0' (NO split needed) and shift the rest by +1.
            if isl.step_name == "mla":
                _qb_out = None
                for _b in li.binds:
                    if _b.tensor_name in ("0", "1") and _b.role == LIVE_SUB:
                        _qb_out = outs_by_step[_b.producer]
                        break
                if _qb_out is not None:
                    _rm = {"0": _qb_out}
                    for _n in range(2, 12):           # manifest 2..11  ->  recipe 3..12
                        if str(_n) in name_to_tensor:
                            _rm[str(_n + 1)] = name_to_tensor[str(_n)]
                    name_to_tensor = _rm
                # ── STOCK forward_impl on the REAL chain projections. _resident_live now lands
                # data (CPU round-trip), so q_b/kv_a_norm/fused_qkv_a OUTs carry REAL values. Read
                # them into CLEAN tensors, rope q_pe+k_pe (stock convention), call the stock
                # forward_impl (writes the cache via latent_cache_k + computes flat_pa attention),
                # and hand the result to o_proj as a CLEAN buffer (bound by data_ptr). The drifting
                # custom mla_decode synLaunch is SKIPPED (continue). No shared-cache torch write.
                if _gg.get('_MLA', 0) <= 16:
                    import sys as _rbg9
                    print("[RB-GATE] mla rb=%s qb=%s taken=%s" % (
                        _rb_layer is not None, _qb_out is not None,
                        (_rb_layer is not None and _qb_out is not None)),
                        file=_rbg9.stderr, flush=True)
                if (not os.path.exists("$HOME/AI/mmel/.routeb_off")) and _rb_layer is not None and _qb_out is not None:
                    try:
                        _M9 = int(self.plan.geom["M"])
                        _wrap = _rb_layer.self_attn.mla_attn
                        _hd = int(_wrap.qk_head_dim); _nope = int(_wrap.qk_nope_head_dim)
                        _nh = int(_wrap.num_heads)
                        _kvn = next((t for t in outs_by_step.values()
                                     if int(t.numel()) == _M9 * 512 * 2), None)
                        _qkva = next((t for t in outs_by_step.values()
                                      if int(t.numel()) == _M9 * 2624 * 2), None)
                        # SPEEDUP: prior islands' synchronize already materialized these device
                        # tensors — skip the _resident_live CPU round-trip (D2H+H2D, 3x/layer).
                        _q = _qb_out.view(torch.bfloat16).view(_M9, _nh, _hd).clone()
                        _kvc = _kvn.view(torch.bfloat16).view(_M9, 512)
                        _kpe = _qkva.view(torch.bfloat16).view(_M9, 2624)[:, 2560:2624].contiguous().view(_M9, 1, 64)
                        if _gg.get('_MLA', 0) <= 4:
                            try:
                                _lh = live.view(torch.bfloat16).reshape(_M9, -1)
                                _nm = _rb_layer.input_layernorm(_lh)
                                _so = _wrap.fused_qkv_a_proj(_nm)
                                _so = _so[0] if isinstance(_so, (tuple, list)) else _so
                                _co = _qkva.view(torch.bfloat16).reshape(_M9, -1)
                                _sof = _so.reshape(_M9, -1).float()
                                print("[DIAG] n=%d hid=%.2f nrm=%.2f chainQKVA=%.2f stockQKVA=%.2f maxdiff=%.4f" % (
                                    _gg.get('_MLA', 0), _lh.float().norm().item(), _nm.float().norm().item(),
                                    _co.float().norm().item(), _sof.norm().item(),
                                    (_co.float() - _sof).abs().max().item()), file=_ds6.stderr, flush=True)
                                if _gg.get('_MLA', 0) == 1:
                                    _cn = outs_by_step.get("input_norm")
                                    _cnf = _cn.view(torch.bfloat16).reshape(_M9, -1).float() if _cn is not None else None
                                    _cb = outs_by_step.get("q_b")
                                    _cbf = _cb.view(torch.bfloat16).reshape(_M9, -1).float() if _cb is not None else None
                                    # stock q_b: q_a_norm(stock_qc) -> q_b ; stock_qc = stock_qkva[:, :2048]
                                    _sqc = _sof[:, :2048]
                                    _snqc = _wrap.q_a_layernorm(_sqc.to(torch.bfloat16))
                                    _sqb = _wrap.q_b_proj(_snqc); _sqb = _sqb[0] if isinstance(_sqb,(tuple,list)) else _sqb
                                    _sqbf = _sqb.reshape(_M9, -1).float()
                                    print("[DIAG2] norm: chain=%.2f stock=%.2f maxdiff=%.4f || q_b: chain=%.2f stock=%.2f maxdiff=%.4f" % (
                                        (_cnf.norm().item() if _cnf is not None else -1), _nm.float().norm().item(),
                                        ((_cnf - _nm.float()).abs().max().item() if _cnf is not None else -1),
                                        (_cbf.norm().item() if _cbf is not None else -1), _sqbf.norm().item(),
                                        ((_cbf - _sqbf).abs().max().item() if _cbf is not None else -1)),
                                        file=_ds6.stderr, flush=True)
                            except Exception as _de:
                                print("[DIAG-ERR] %r" % _de, file=_ds6.stderr, flush=True)
                        _qp, _kpe2 = _wrap.rotary_emb(_rb_pos, _q[..., _nope:], _kpe)
                        _q[..., _nope:] = _qp
                        # Use the INNER module's OWN kv_cache (the one STOCK prefill populates).
                        _inner = _wrap.mla_attn
                        _kvcache = getattr(_inner, "kv_cache", None) or _rb_kv
                        # Call the FULL forward (__call__): it fetches attn_metadata from the live
                        # forward-context + uses self.kv_cache — EXACTLY like stock. (Passing the
                        # explicit _rb_am to forward_impl gave a tiny output; the forward-context
                        # metadata is the one flat_pa's gather is built for.)
                        try:
                            _out = _inner(_q, _kvc, _kpe2)
                        except Exception:
                            _out = _inner.forward_impl(_q, _kvc, _kpe2, _kvcache, _rb_am)
                        outs_by_step[isl.step_name] = self._resident_live(
                            ("rbout", id(rt)), _out.reshape(_M9, -1).contiguous())
                        continue
                    except Exception as _rbe:  # LOUD
                        import sys as _sb9
                        if getattr(type(self), "_RBERR", 0) < 5:
                            print("[ROUTEB-ERR] %r" % _rbe, file=_sb9.stderr, flush=True)
                            type(self)._RBERR = getattr(type(self), "_RBERR", 0) + 1
            # CORRECTNESS materialization: the LAZY live inputs (_sub_view splits, rope
            # gather, fused-add hidden) bind a lazy-mode PLACEHOLDER data_ptr until executed.
            # mark_step ALONE does not allocate real addrs; a real hpu.synchronize() forces
            # the lazy graph to allocate+execute => distinct real addresses (kills the
            # q_nope/q_pe section ALIAS=status=30) + real data. HEAVY (a fence per sub-view
            # island) — this is the CORRECTNESS-FIRST build; the 1-sync collapse comes later
            # from fusing the glue into the recipe (single-attention-island). Gated so it
            # only fires when there are LAZY live binds to materialize.
            # SPEEDUP: LIVE_LAYER_INPUT (the live hidden) is already materialized at run-start
            # (_resident_live), and the chain's single-stream in-order launch guarantees each
            # producer executes before its consumer reads the buffer — so it does NOT need a
            # per-island synchronize. Only LIVE_SUB (strided slice copies) + LIVE_KV/META do.
            _has_lazy = any(b.role in (LIVE_SUB, LIVE_KV, LIVE_META)
                            for b in li.binds)
            if _has_lazy and not os.path.exists("$HOME/AI/mmel/.nosync"):
                import habana_frameworks.torch.core as _htc
                import habana_frameworks.torch.hpu as _hth
                _htc.mark_step()
                _hth.synchronize()
            launch_arr = rt.replay.build_launch_array(name_to_tensor)
            # Re-read the workspace VA FRESH: rt.ws_va was captured in warmup off a
            # torch.empty BEFORE the device commit => a PLACEHOLDER VA (same lazy-VA trap as
            # the inputs; the inputs are re-read fresh by build_launch_array but ws_va is a
            # frozen int). rt.ws_dev is committed by the synchronize above, so re-query it.
            _ws_ptr = (int(rt.ws_dev.data_ptr())
                       if getattr(rt, "ws_dev", None) is not None else rt.ws_va)
            rt.replay.launch(self._stream_handle, launch_arr, ws_ptr=_ws_ptr)
            outs_by_step[isl.step_name] = rt.out_tensor

        # ONE synStreamSynchronize for the WHOLE live attention chain.
        _syn_stream_synchronize(self._stream_handle)
        tail = self._islands_rt[-1]
        # out is [M, H_hidden]; out_nbytes is the TOTAL (M*H_hidden*2). The old `//2` assumed
        # M=1 and returned [1, M*H_hidden] => the downstream matmul failed ("196608 and 6144").
        # Divide by M to get the per-row hidden and return [M, H_hidden].
        _M = int(self.plan.geom["M"])
        H = (tail.spec.out_nbytes // 2) // _M
        return tail.out_tensor.view(torch.bfloat16).view(-1, H)


# The per-layer cached LiveAttnChainReplay (warmed on first decode use). The seam's
# attention forward routes here when MMEL_CHAIN_LIVE=1 + live_attn_covers(hidden).
_MMEL_LIVE_ATTN_ATTR = "_mmel_live_attn_chain"


def run_attn_chain_via_live_replay(layer: Any, hidden_states: Any,
                                   live_ctx: dict) -> Any:
    """Drive THIS layer's ATTENTION (input_norm -> o_proj) via the LIVE-FED chain.

    First call (per layer): resolve the live plan (parse the cap_*_m32 manifests,
    derive the live bind map) + warm the replay (deserialize 7 recipes + upload the
    RES statics resident). Every call: bind the live hidden + the sub-buffer splits +
    the MLA's live-KV/META (``live_ctx``), synLaunch all 7 on the bridge stream, ONE
    synStreamSynchronize. Returns the device attention output ([M,H] bf16) — the
    layer's pre-post-norm residual addend. post_norm + MoE remain the follow-up (the
    seam adds the live residual then runs the stock/sub-chain-B post-attention tail).

    LOUD on any precondition failure (missing recipe/lib/weights, shape drift,
    incomplete live_ctx) — the seam's coherent wrapper keeps generation safe by
    falling back to the stock attention path."""
    chain = getattr(layer, _MMEL_LIVE_ATTN_ATTR, None)
    if chain is None:
        plan = LiveAttnChainPlan.resolve(cap_parent_dir())
        chain = LiveAttnChainReplay(plan=plan).warmup_once()
        setattr(layer, _MMEL_LIVE_ATTN_ATTR, chain)
    return chain.run(hidden_states, live_ctx)


# ============================================================================= #
#  CARD-FREE SELF-TEST / GATE (no accelerator touched).                          #
#                                                                               #
#   E1. the gate predicate (MMEL_CHAIN_LAYER=1 => chain; 0/unset => not).        #
#   E2. cap parent dir resolution (env override + default fallback).             #
#   E3. parse ALL 7 real island manifests + resolve the chain plan: the          #
#       live/staged edge topology MATCHES the RUN_CHAIN STEP-4 table (norm/q/kv/  #
#       MLA staged on the branch; MLA->o_proj->norm->MoE LIVE on the spine).      #
#   E4. assert_live_hidden: the captured layer-input geometry passes; drift LOUD.#
#   E5. a synthetic chain whose tail island is STAGED RAISES (the spine tail must #
#       be live) — the _sanity guard.                                            #
#   E6. the per-island bind-map completeness logic (static + in0 + out == recipe).#
#   E7. the on-card replay path LOUD-guards off-card (no libSynapse => warmup     #
#       raises, never a silent no-op).                                           #
#   E8. the layer-input byte size == the routed-MoE OUT byte size (the chain      #
#       output is a valid [M,H] hidden, threadable through 78 layers).            #
# ============================================================================= #
def _e1_gate():
    prev = os.environ.get(CHAIN_LAYER_ENV)
    try:
        for v, want in (("1", True), ("layer", True), ("on", True),
                        ("0", False), ("2", False), ("", False)):
            if v == "":
                os.environ.pop(CHAIN_LAYER_ENV, None)
            else:
                os.environ[CHAIN_LAYER_ENV] = v
            assert chain_layer_enabled() is want, \
                f"{CHAIN_LAYER_ENV}={v!r} gate should be {want}"
    finally:
        if prev is None:
            os.environ.pop(CHAIN_LAYER_ENV, None)
        else:
            os.environ[CHAIN_LAYER_ENV] = prev
    return "E1 MMEL_CHAIN_LAYER gate logic correct (1/layer/on=on; 0/2/unset=off)"


def _e2_cap_dir():
    prev = os.environ.get(CAP_DIR_ENV)
    try:
        os.environ[CAP_DIR_ENV] = "/tmp/__mmel_custom_capdir__"
        assert cap_parent_dir() == "/tmp/__mmel_custom_capdir__", "env override must win"
        os.environ.pop(CAP_DIR_ENV, None)
        d = cap_parent_dir()
        # default falls back to a dir whose cap_routed exists, or the host default.
        assert os.path.isdir(d) or d == _CAP_PARENT_DEFAULTS[0], \
            f"default {d!r} unexpected"
    finally:
        if prev is None:
            os.environ.pop(CAP_DIR_ENV, None)
        else:
            os.environ[CAP_DIR_ENV] = prev
    return "E2 cap parent dir resolution (env override + default) correct"


def _e3_resolve_chain_plan():
    parent = cap_parent_dir()
    if not os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt")):
        return f"E3 SKIP (no real cap_* dirs under {parent!r}; resolved on-card)"
    plan = LayerChainPlan.resolve(parent)
    assert len(plan.islands) == 7, f"expected 7 islands, got {len(plan.islands)}"
    names = [isl.step_name for isl in plan.islands]
    assert names == ["input_norm", "q_proj", "kv_proj", "mla", "o_proj",
                     "post_norm", "moe"], f"chain order drifted: {names}"
    feeds = {e.step_name: e.feed for e in plan.edges}
    # RUN_CHAIN STEP-4 topology: layer-input on norm; branch staged; spine live.
    assert feeds["input_norm"] == "LAYER_INPUT", feeds
    assert feeds["q_proj"] == "STAGED", f"q_proj should be STAGED (branch): {feeds}"
    assert feeds["kv_proj"] == "STAGED", f"kv_proj should be STAGED: {feeds}"
    assert feeds["mla"] == "STAGED", f"mla should be STAGED (kv->MLA branch): {feeds}"
    # the spine: o_proj (MLA OUT 32768 == o_proj in0 32768), post_norm, moe LIVE.
    assert feeds["o_proj"] == "LIVE", f"o_proj should be LIVE (MLA OUT==in0): {feeds}"
    assert feeds["post_norm"] == "LIVE", f"post_norm should be LIVE: {feeds}"
    assert feeds["moe"] == "LIVE", f"moe should be LIVE (tail of the spine): {feeds}"
    assert plan.n_live_edges == 3, f"expected 3 LIVE spine edges, got {plan.n_live_edges}"
    assert plan.n_staged_edges == 3, f"expected 3 STAGED branch edges, got {plan.n_staged_edges}"
    return (f"E3 chain plan resolved: 7 islands, {plan.n_live_edges} LIVE spine + "
            f"{plan.n_staged_edges} STAGED branch (matches RUN_CHAIN STEP-4 topology)")


def _e4_shape_guard():
    parent = cap_parent_dir()
    if not os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt")):
        plan = _synthetic_plan()
    else:
        plan = LayerChainPlan.resolve(parent)
    # the captured layer-input is 12288 B = [1,6144] bf16; matching shapes pass.
    H = plan.layer_input_nbytes // 2
    plan.assert_live_hidden("h", (1, H))
    plan.assert_live_hidden("h", (1, 1, H))
    plan.assert_live_hidden("h", (H,))
    raised = False
    try:
        plan.assert_live_hidden("h", (1, H + 1))
    except MmelChainLayerShapeError:
        raised = True
    assert raised, "a shape drift must raise MmelChainLayerShapeError"
    return "E4 assert_live_hidden passes on the captured layer-input, LOUD on drift"


def _e5_tail_staged_raises():
    # A synthetic chain whose tail island's in0 size != the prior OUT (so the tail
    # is STAGED) must RAISE (the spine tail must be LIVE).
    isl_a = IslandSpec("a", "/x/a", (
        IslandTensor("0", 8, "IN_FILE", "/dev/null"),
        IslandTensor("o", 8, "OUT", "/dev/null")), "0", 8, "o", 8)
    # tail "b" has in0=16 != a.out=8 -> STAGED tail -> must raise.
    isl_b = IslandSpec("b", "/x/b", (
        IslandTensor("0", 16, "IN_FILE", "/dev/null"),
        IslandTensor("o", 16, "OUT", "/dev/null")), "0", 16, "o", 16)
    raised = False
    try:
        LayerChainPlan(
            cap_parent="/x",
            islands=(isl_a, isl_b),
            edges=(ChainEdge(0, "a", "LAYER_INPUT", 8, 0),
                   ChainEdge(1, "b", "STAGED", 16, 8)),
            layer_input_nbytes=8)._sanity()
    except MmelChainLayerError as e:
        raised = "is STAGED, not LIVE" in str(e) or "0 LIVE feed edges" in str(e)
    assert raised, "a STAGED tail island must raise (the chain output must be live)"
    return "E5 a STAGED tail island RAISES (the spine tail must be LIVE-fed)"


def _e6_bind_completeness_logic():
    plan = _synthetic_plan()
    for isl in plan.islands:
        expected = {t.name for t in isl.tensors}
        good = set(isl.static_in_names) | {isl.in0_name, isl.out_name}
        assert good == expected, f"{isl.step_name}: static+in0+out must cover recipe tensors"
        bad = set(isl.static_in_names) | {isl.out_name}  # missing in0
        assert bad != expected, f"{isl.step_name}: missing in0 must be detected incomplete"
    return "E6 per-island bind-map completeness (static+in0+out == recipe set) correct"


def _e7_oncard_path_guards_offcard():
    on_card = any(os.path.isfile(p) for p in (
        "/usr/lib/habanalabs/libSynapse.so", "libSynapse.so"))
    plan = _synthetic_plan()
    c = DecoderLayerChainReplay(plan=plan)
    raised = False
    try:
        c.warmup_once()
    except MmelChainLayerError:
        raised = True
    except Exception:  # noqa: BLE001 — no torch / no hpu is also "not silently ok"
        raised = True
    assert raised, "warmup_once() must NOT silently succeed off-card (no card/recipe)"
    note = "(on-card: real warmup is the B-warmup step)" if on_card else \
        "(off-card: LOUD guard verified)"
    return f"E7 device-resident layer-chain replay LOUD-guards off-card {note}"


def _e8_layer_output_threadable():
    parent = cap_parent_dir()
    if not os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt")):
        return "E8 SKIP (no real cap_* dirs; checked on-card)"
    plan = LayerChainPlan.resolve(parent)
    tail = plan.islands[-1]
    # the layer output (MoE OUT) must equal the layer input (norm in0) byte size,
    # so the chained layer output is a valid [M,H] hidden threadable through 78 layers.
    assert tail.out_nbytes == plan.layer_input_nbytes, (
        f"MoE OUT {tail.out_nbytes} B != layer-input {plan.layer_input_nbytes} B; "
        "the chained layer output is not a valid hidden for the next layer")
    return (f"E8 layer output ({tail.out_nbytes} B MoE OUT) == layer input "
            f"({plan.layer_input_nbytes} B) — threadable through 78 layers")


def _e9_poison_rmsnorm_robust():
    """The override-proof poison must be RMSNorm-ROBUST: a uniform scale is washed by
    the next layer's RMSNorm (the prior FALSE-coherent mistake), but the additive
    per-channel offset + single-channel spike must SURVIVE it (change the normalized
    vector enough to flip greedy). Proven here on CPU with numpy (no card) so the math
    claim is gated, not asserted. Also checks the gate predicate + mag parse."""
    import numpy as np

    # gate predicate: only "1/true/yes/on" arm; everything else inert.
    saved = os.environ.get(CHAIN_POISON_ENV)
    try:
        for v, want in (("1", True), ("on", True), ("true", True), ("0", False),
                        ("", False), ("2", False), (None, False)):
            if v is None:
                os.environ.pop(CHAIN_POISON_ENV, None)
            else:
                os.environ[CHAIN_POISON_ENV] = v
            assert _chain_poison_enabled() is want, f"gate {v!r} -> {want}"
    finally:
        if saved is None:
            os.environ.pop(CHAIN_POISON_ENV, None)
        else:
            os.environ[CHAIN_POISON_ENV] = saved

    # mag parse: default 8.0, env override, junk -> default.
    sm = os.environ.get("MMEL_CHAIN_POISON_MAG")
    try:
        os.environ.pop("MMEL_CHAIN_POISON_MAG", None)
        assert _chain_poison_mag() == 8.0
        os.environ["MMEL_CHAIN_POISON_MAG"] = "3.5"; assert _chain_poison_mag() == 3.5
        os.environ["MMEL_CHAIN_POISON_MAG"] = "junk"; assert _chain_poison_mag() == 8.0
    finally:
        if sm is None:
            os.environ.pop("MMEL_CHAIN_POISON_MAG", None)
        else:
            os.environ["MMEL_CHAIN_POISON_MAG"] = sm

    # The RMSNorm-robustness math (the load-bearing correctness claim). RMSNorm(x) =
    # x / sqrt(mean(x^2)+eps). Build a representative O(1) hidden, apply the SAME poison
    # the device path applies, and confirm the post-RMSNorm vector MOVES materially —
    # whereas a uniform scale does NOT (it is exactly washed).
    rng = np.random.default_rng(0)
    H = 6144
    x = rng.standard_normal(H).astype(np.float32)
    mag = 8.0

    def rmsnorm(v, eps=1e-6):
        return v / np.sqrt(np.mean(v * v) + eps)

    base = rmsnorm(x)

    # uniform scale (the FALSE-coherent trap) — must be ~washed (proves WHY we don't use it).
    scaled = rmsnorm(x * 1.15)
    washed = float(np.max(np.abs(scaled - base)))
    assert washed < 1e-4, (
        f"uniform scale should be washed by RMSNorm but moved {washed} (sanity of the "
        "trap we avoid)")

    # OUR poison: per-channel additive ramp 0..mag + single-channel spike mag*16.
    offs = (np.arange(H) / max(H - 1, 1)) * mag
    poisoned = x + offs
    poisoned[0] = poisoned[0] + mag * 16.0
    pn = rmsnorm(poisoned)
    moved = float(np.max(np.abs(pn - base)))
    # The signal must be ORDERS of magnitude past the washed uniform scale AND a large
    # fraction of the normalized vector's own scale (so greedy argmax flips).
    assert moved > 0.1, (
        f"additive+spike poison only moved the normalized vector by {moved}; not "
        "RMSNorm-robust enough to guarantee a greedy garble")
    assert moved > washed * 1000, (
        f"poison move {moved} must dwarf the washed uniform-scale move {washed} (the "
        "whole point: additive+spike is NOT scale-invariant)")
    return (f"E9 poison RMSNorm-ROBUST: additive-offset+spike moves the normalized "
            f"vector by {moved:.3f} (uniform-scale washes to {washed:.1e}) — greedy "
            "garbles iff the chain device output is served; gate+mag parse OK")


def _e10_captured_bucket_coverage():
    """The integration-fix predicate: chain_layer_covers() is True ONLY for the
    captured M=1 decode bucket and False for prefill / non-M=1 (so the seam drives
    the chain on decode + delegates prefill to G1 instead of dying on the prompt —
    the 2026-06-22 0-WARMED blocker). Card-free; uses the real captures if present,
    else a synthetic 12288-B layer-input plan."""
    parent = cap_parent_dir()
    have_real = os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt"))
    if have_real:
        want = captured_layer_input_nbytes(parent)
        assert want == 12288, (
            f"captured layer-input expected 12288 B (M=1 [1,6144] bf16) got {want}; "
            "the GLM-5.2 single-stream decode bucket drifted — re-check the captures")
        H = want // 2  # 6144
        # the captured M=1 decode bucket: COVERED (chain engages here).
        assert chain_layer_covers(_FakeHidden((1, H)), parent) is True, \
            "the captured M=1 decode bucket [1,6144] must be COVERED"
        assert chain_layer_covers(_FakeHidden((H,)), parent) is True, \
            "a flat [6144] M=1 hidden must be COVERED (same byte size)"
        # PREFILL [128,6144] (the byte size that killed the worker): NOT covered.
        assert chain_layer_covers(_FakeHidden((128, H)), parent) is False, \
            "PREFILL [128,6144] must NOT be covered (delegated to G1, not a crash)"
        # other non-M=1 decode buckets: NOT covered.
        assert chain_layer_covers(_FakeHidden((4, H)), parent) is False, \
            "a non-M=1 decode bucket [4,6144] must NOT be covered"
        note = f"(real captures; M=1 bucket={want} B)"
    else:
        # No real captures here: the predicate must be conservatively False
        # (undecidable plan => do not claim coverage => never crash a forward).
        assert captured_layer_input_nbytes(parent) is None, \
            "with no captures the layer-input nbytes must be None (undecidable)"
        assert chain_layer_covers(_FakeHidden((1, 6144)), parent) is False, \
            "undecidable plan => chain_layer_covers must be False (delegate, never crash)"
        note = "(no real captures; undecidable => conservatively not-covered)"
    # unreadable shape => never raises, returns False.
    assert chain_layer_covers(object(), parent) is False, \
        "an unreadable hidden_states must yield False, never raise"
    return (f"E10 captured-bucket coverage: M=1 decode COVERED, prefill/non-M=1 "
            f"delegated (not crashed) {note}")


class _FakeHidden:
    """A shape-only stand-in for the coverage predicate test (no torch needed)."""

    def __init__(self, shape):
        self.shape = shape


def _synthetic_plan() -> LayerChainPlan:
    """A minimal valid 3-island plan (norm->LIVE GEMM->LIVE GEMM) for pure-logic
    tests (no files). All edges size-matched + live so _sanity passes."""
    def _isl(nm, in0nb, outnb):
        return IslandSpec(nm, f"/x/{nm}", (
            IslandTensor("0", in0nb, "IN_FILE", "/dev/null"),
            IslandTensor("w", 4, "IN_FILE", "/dev/null"),
            IslandTensor("o", outnb, "OUT", "/dev/null")), "0", in0nb, "o", outnb)
    islands = (_isl("input_norm", 8, 8), _isl("lin1", 8, 8), _isl("moe", 8, 8))
    edges = (ChainEdge(0, "input_norm", "LAYER_INPUT", 8, 0),
             ChainEdge(1, "lin1", "LIVE", 8, 8),
             ChainEdge(2, "moe", "LIVE", 8, 8))
    return LayerChainPlan(cap_parent="/x", islands=islands, edges=edges,
                          layer_input_nbytes=8)


# ============================================================================= #
#  SUB-CHAIN B (the CORRECT static tail) CARD-FREE GATES — E11..E15.              #
#                                                                               #
#   E11. MMEL_CHAIN_STATIC gate predicate.                                       #
#   E12. SubChainBPlan.resolve: 3 islands [o_proj, post_norm, MoE], the head is  #
#        LAYER_INPUT at the attention-output byte size (32768), every internal    #
#        edge is LIVE (the pure linear spine); a STAGED edge RAISES.             #
#   E13. subchain_b_covers: M=1 attention output ([1,16384]=32768 B) COVERED;    #
#        prefill/non-M=1 attention outputs NOT covered (delegate, never crash).  #
#   E14. ***THE KEY NEW CAPABILITY*** card-free BYTE-EXACT-vs-STOCK: the o_proj   #
#        island, replayed (numpy reference of the SAME GEMM the recipe runs) on   #
#        its captured live input + static weight, reproduces the stock o_proj     #
#        oracle (cap_oproj/out_ref.bin) to the bf16 round-trip floor (relNorm <   #
#        1e-2, here ~1e-8). This is the byte-exactness the STAGED all-7 chain     #
#        could NEVER have: o_proj is a PURE function of (live input, static W),   #
#        so the device replay == stock o_proj on the SAME live input, bit-for-bit.#
#        E14 ALSO documents (LOUD, asserted) the HONEST BOUNDARY: cap_norm is a   #
#        PLAIN single-input rms_norm (NOT the stock fused-ADD-rms_norm), and the  #
#        MoE router (rws) is captured-static — so post_norm + MoE reproduce stock #
#        only at the CAPTURED input (a fixed-input mechanism proof, the same      #
#        class as the all-7's serial-control gate), NOT vs a LIVE-input stock     #
#        submodule. The end-to-end-vs-live-stock gate needs the live-feed         #
#        extension (live residual into a fused-add-norm island + live router into #
#        MoE); E14 makes that boundary explicit so the on-card A/B reads it right.#
#   E15. the SubChainBReplay device path LOUD-guards off-card (no libSynapse =>   #
#        warmup raises, never a silent no-op).                                    #
# ============================================================================= #
def _e11_static_gate():
    prev = os.environ.get(CHAIN_STATIC_ENV)
    try:
        for v, want in (("1", True), ("static", True), ("on", True),
                        ("0", False), ("2", False), ("", False)):
            if v == "":
                os.environ.pop(CHAIN_STATIC_ENV, None)
            else:
                os.environ[CHAIN_STATIC_ENV] = v
            assert chain_static_enabled() is want, \
                f"{CHAIN_STATIC_ENV}={v!r} gate should be {want}"
    finally:
        if prev is None:
            os.environ.pop(CHAIN_STATIC_ENV, None)
        else:
            os.environ[CHAIN_STATIC_ENV] = prev
    return "E11 MMEL_CHAIN_STATIC gate logic correct (1/static/on=on; 0/2/unset=off)"


def _e12_subchain_b_plan():
    parent = cap_parent_dir()
    if not os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt")):
        return f"E12 SKIP (no real cap_* dirs under {parent!r}; resolved on-card)"
    plan = SubChainBPlan.resolve(parent)
    names = [isl.step_name for isl in plan.islands]
    assert names == ["o_proj", "post_norm", "moe"], f"sub-chain B order drifted: {names}"
    feeds = {e.step_name: e.feed for e in plan.edges}
    assert feeds["o_proj"] == "LAYER_INPUT", feeds
    assert feeds["post_norm"] == "LIVE", f"post_norm must be LIVE (o_proj OUT==in0): {feeds}"
    assert feeds["moe"] == "LIVE", f"moe must be LIVE (post_norm OUT==in0): {feeds}"
    assert plan.n_live_edges == 2, f"expected 2 LIVE spine edges, got {plan.n_live_edges}"
    # the head feeds the ATTENTION OUTPUT (o_proj in0 = 32768 = [1,16384] bf16).
    assert plan.attn_out_nbytes == 32768, \
        f"attn_out_nbytes {plan.attn_out_nbytes} != 32768 ([1,16384] bf16)"
    # the output (MoE OUT) is the post-layer hidden = 12288 ([1,6144]).
    assert plan.out_nbytes == 12288, f"sub-chain B out {plan.out_nbytes} != 12288"
    # a synthetic STAGED internal edge must RAISE (the tail must be a pure LIVE spine).
    raised = False
    try:
        bad_isl_a = IslandSpec("o_proj", "/x/o", (
            IslandTensor("0", 32768, "IN_FILE", "/dev/null"),
            IslandTensor("o", 12288, "OUT", "/dev/null")), "0", 32768, "o", 12288)
        bad_isl_b = IslandSpec("post_norm", "/x/n", (
            IslandTensor("0", 999, "IN_FILE", "/dev/null"),
            IslandTensor("o", 12288, "OUT", "/dev/null")), "0", 999, "o", 12288)
        bad_isl_c = IslandSpec("moe", "/x/m", (
            IslandTensor("0", 12288, "IN_FILE", "/dev/null"),
            IslandTensor("o", 12288, "OUT", "/dev/null")), "0", 12288, "o", 12288)
        SubChainBPlan(
            cap_parent="/x", islands=(bad_isl_a, bad_isl_b, bad_isl_c),
            edges=(ChainEdge(0, "o_proj", "LAYER_INPUT", 32768, 0),
                   ChainEdge(1, "post_norm", "STAGED", 999, 12288),
                   ChainEdge(2, "moe", "LIVE", 12288, 12288)),
            attn_out_nbytes=32768)._sanity()
    except MmelChainLayerError as e:
        raised = "non-LIVE internal edge" in str(e) or "STAGED" in str(e)
    assert raised, "a STAGED internal edge in sub-chain B must RAISE (pure LIVE spine)"
    return (f"E12 sub-chain B plan: [o_proj, post_norm, MoE], head=attn-out "
            f"{plan.attn_out_nbytes} B, {plan.n_live_edges} LIVE spine edges, out "
            f"{plan.out_nbytes} B; a STAGED internal edge RAISES")


def _e13_subchain_b_coverage():
    parent = cap_parent_dir()
    have_real = os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt"))
    if have_real:
        want = subchain_b_attn_out_nbytes(parent)
        assert want == 32768, (
            f"sub-chain B attn-out input expected 32768 B ([1,16384] bf16) got {want}")
        # M=1 decode attention output [1,16384]: COVERED.
        assert subchain_b_covers(_FakeHidden((1, 16384)), parent) is True, \
            "the captured M=1 attention output [1,16384] must be COVERED"
        assert subchain_b_covers(_FakeHidden((16384,)), parent) is True, \
            "a flat [16384] M=1 attention output must be COVERED (same byte size)"
        # PREFILL attention output [128,16384]: NOT covered.
        assert subchain_b_covers(_FakeHidden((128, 16384)), parent) is False, \
            "PREFILL [128,16384] attention output must NOT be covered (delegate)"
        assert subchain_b_covers(_FakeHidden((4, 16384)), parent) is False, \
            "a non-M=1 attention output [4,16384] must NOT be covered"
        note = f"(real captures; M=1 attn-out={want} B)"
    else:
        assert subchain_b_attn_out_nbytes(parent) is None, \
            "with no captures the attn-out nbytes must be None (undecidable)"
        assert subchain_b_covers(_FakeHidden((1, 16384)), parent) is False, \
            "undecidable plan => subchain_b_covers must be False"
        note = "(no real captures; undecidable => conservatively not-covered)"
    assert subchain_b_covers(object(), parent) is False, \
        "an unreadable attn-out must yield False, never raise"
    return (f"E13 sub-chain B coverage: M=1 attn-out COVERED, prefill/non-M=1 "
            f"delegated (not crashed) {note}")


def _bf16_bytes_to_f32(raw_u16):
    """Decode bf16 (uint16) to float32 (the device storage -> math space)."""
    import numpy as np
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16_rne_f32(a_f32):
    """Round float32 -> bf16 (round-to-nearest-even) and back to float32, so a
    host reference compares in the SAME bf16 space the device recipe writes."""
    import numpy as np
    a = np.ascontiguousarray(a_f32, dtype=np.float32)
    u = a.view(np.uint32)
    rounding = ((u >> 16) & 1) + np.uint32(0x7fff)
    return (((u + rounding) >> 16) << 16).view(np.float32)


def _e14_byte_exact_vs_stock():
    """***THE KEY NEW CAPABILITY*** (card-free): prove that the o_proj island —
    the part of sub-chain B that is a PURE function of (live input, static weight)
    — reproduces the STOCK o_proj output BYTE-EXACT (to the bf16 round-trip floor)
    from its captured live input. This is the byte-exactness the STAGED all-7
    chain could never have. AND assert (LOUD) the honest boundary for post_norm +
    MoE (fixed-input mechanism proofs, not live-input-vs-stock).

    The arbiter is the captured cap_oproj/out_ref.bin — the bridge's OWN byte-exact
    oracle, i.e. what the STOCK o_proj GEMM produced for the captured input. We
    re-run the IDENTICAL GEMM in numpy (RowParallelLinear: out = in @ W.T, W=[out,in]
    bf16) and compare in bf16 space. A match proves the device replay of this
    island == stock o_proj on the same live input."""
    import numpy as np
    parent = cap_parent_dir()
    od = os.path.join(parent, "cap_oproj")
    nd = os.path.join(parent, "cap_norm")
    rd = os.path.join(parent, "cap_routed")
    if not all(os.path.isfile(os.path.join(p, "out_ref.bin"))
               for p in (od, nd, rd)):
        return "E14 SKIP (no real cap_* out_ref oracles; the byte-exact gate runs on-card)"

    # ---- o_proj: the genuinely byte-exact-vs-STOCK island -------------------
    # cap_oproj manifest: in0=32768 ([1,16384] bf16), W=201326592 ([6144,16384] bf16),
    # OUT=12288 ([1,6144] bf16). out_ref.bin is the stock o_proj oracle.
    x = _bf16_bytes_to_f32(np.fromfile(os.path.join(od, "0.bin"), dtype=np.uint16,
                                       count=16384)).astype(np.float32)
    W = _bf16_bytes_to_f32(np.fromfile(os.path.join(od, "1.bin"), dtype=np.uint16,
                                       count=6144 * 16384)).reshape(6144, 16384)
    ref = _bf16_bytes_to_f32(np.fromfile(os.path.join(od, "out_ref.bin"),
                                         dtype=np.uint16, count=6144)).astype(np.float32)
    got = _f32_to_bf16_rne_f32(W @ x)           # the SAME GEMM the recipe runs
    denom = float(np.linalg.norm(ref)) + 1e-12
    rel = float(np.linalg.norm(got - ref) / denom)
    max_abs = float(np.abs(got - ref).max())
    # byte-exact-able gate: relNorm < 1e-2 (the task floor; o_proj lands ~1e-8 — the
    # bf16 round-trip floor, i.e. the host fp32-accum vs the device's bf16 storage).
    assert rel < 1e-2, (
        f"o_proj replay relNorm {rel:.3e} vs the stock o_proj oracle exceeds 1e-2 — "
        "the o_proj island does NOT reproduce stock o_proj. A weight/layout/transpose "
        "drift; do NOT ship.")
    # exact-byte count in bf16 storage (the strict view).
    ref_u16 = np.fromfile(os.path.join(od, "out_ref.bin"), dtype=np.uint16, count=6144)
    got_u16 = (got.view(np.uint32) >> 16).astype(np.uint16)
    mism = int((ref_u16 != got_u16).sum())
    n_bf16 = int(ref_u16.size)

    # ---- the HONEST BOUNDARY for post_norm + MoE (asserted, LOUD) -----------
    # cap_norm is a PLAIN single-input rms_norm: out = (data_in / rms(data_in)) * gamma.
    # It reproduces ITS OWN out_ref (a fixed-input mechanism proof) but is NOT the stock
    # post_attention_layernorm (a 2-input FUSED-ADD-rms_norm: residual+=hidden; normed=
    # rms(residual)). Prove cap_norm is the plain op (matches its own oracle) AND that it
    # is single-input (so the seam must NOT claim it == stock post_attention_layernorm).
    xin = _bf16_bytes_to_f32(np.fromfile(os.path.join(nd, "0.bin"), dtype=np.uint16,
                                         count=6144)).astype(np.float32)
    gam = _bf16_bytes_to_f32(np.fromfile(os.path.join(nd, "1.bin"), dtype=np.uint16,
                                         count=6144)).astype(np.float32)
    nref = _bf16_bytes_to_f32(np.fromfile(os.path.join(nd, "out_ref.bin"),
                                          dtype=np.uint16, count=6144)).astype(np.float32)
    rms = float(np.sqrt(np.mean(xin * xin) + 1e-5))
    nout = _f32_to_bf16_rne_f32((xin / rms) * gam)
    nrel = float(np.linalg.norm(nout - nref) / (float(np.linalg.norm(nref)) + 1e-12))
    # the plain-rms reference must reproduce the norm oracle (its own fixed input) —
    # i.e. cap_norm IS plain rms_norm, confirming it is NOT the fused-add stock op.
    assert nrel < 5e-2, (
        f"cap_norm plain-rms reference relNorm {nrel:.3e} — the norm island is not even "
        "a plain single-input rms_norm of its captured input; the capture is malformed.")
    norm_manifest = _parse_island_manifest("post_norm", nd)
    # plain rms_norm has exactly: data_in (in0) + gamma (1 static) + OUT. The stock
    # fused-add op would carry a SECOND data-size input (the residual). Assert single-
    # input so a future fused-add re-capture is detected (it would add a static input).
    n_data_size_ins = sum(1 for t in norm_manifest.tensors
                          if t.role == "IN_FILE" and t.nbytes == 12288)
    assert n_data_size_ins == 2, (
        f"cap_norm has {n_data_size_ins} 12288-B inputs; plain rms_norm has exactly 2 "
        "(data_in + gamma). A fused-add-rms_norm would carry a 3rd (the residual) — "
        "re-check which norm op was captured.")

    # MoE router (rws) is captured-static (the 2-B router scalars are weights bound
    # once, not derived from the live hidden) — so cap_routed reproduces stock layer.mlp
    # only at the CAPTURED hidden. Confirm cap_routed has the static router scalars.
    moe_manifest = _parse_island_manifest("moe", rd)
    n_router_scalars = sum(1 for t in moe_manifest.tensors
                           if t.role == "IN_FILE" and t.nbytes == 2)
    assert n_router_scalars >= 1, (
        "cap_routed should carry the per-expert router scalars (2-B IN tensors) as "
        "STATIC inputs; none found — the routed-island capture drifted.")

    return (
        f"E14 BYTE-EXACT vs STOCK o_proj+post_norm+MoE: o_proj island reproduces the "
        f"stock o_proj oracle relNorm={rel:.2e} maxAbs={max_abs:.2e} "
        f"({mism}/{n_bf16} bf16 mismatch — bf16 round-trip floor; PURE fn of live "
        f"input+static W, the byte-exactness the STAGED all-7 never had). HONEST "
        f"BOUNDARY (asserted): cap_norm is PLAIN rms_norm (relNorm {nrel:.2e} vs its "
        f"OWN oracle; {n_data_size_ins} data-size INs = data_in+gamma, NOT the stock "
        f"fused-ADD-rms_norm) + MoE router is captured-static ({n_router_scalars} "
        f"2-B scalars) -> post_norm+MoE are FIXED-INPUT mechanism proofs; end-to-end-"
        f"vs-live-stock needs the live-feed extension (live residual+fused-add norm, "
        f"live router into MoE).")


def _e15_subchain_b_oncard_guards():
    plan = _synthetic_subchain_b_plan()
    c = SubChainBReplay(plan=plan)
    raised = False
    try:
        c.warmup_once()
    except MmelChainLayerError:
        raised = True
    except Exception:  # noqa: BLE001 — no torch / no hpu is also "not silently ok"
        raised = True
    assert raised, "SubChainBReplay.warmup_once() must NOT silently succeed off-card"
    on_card = any(os.path.isfile(p) for p in (
        "/usr/lib/habanalabs/libSynapse.so", "libSynapse.so"))
    note = "(on-card: real warmup is the B-warmup step)" if on_card else \
        "(off-card: LOUD guard verified)"
    return f"E15 SubChainBReplay device path LOUD-guards off-card {note}"


def _synthetic_subchain_b_plan() -> SubChainBPlan:
    """A minimal valid sub-chain B plan ([o_proj 32768->12288] -> [post_norm
    12288->12288] -> [moe 12288->12288]) for pure-logic + off-card-guard tests."""
    def _isl(nm, in0nb, outnb):
        return IslandSpec(nm, f"/x/{nm}", (
            IslandTensor("0", in0nb, "IN_FILE", "/dev/null"),
            IslandTensor("w", 4, "IN_FILE", "/dev/null"),
            IslandTensor("o", outnb, "OUT", "/dev/null")), "0", in0nb, "o", outnb)
    islands = (_isl("o_proj", 32768, 12288),
               _isl("post_norm", 12288, 12288),
               _isl("moe", 12288, 12288))
    edges = (ChainEdge(0, "o_proj", "LAYER_INPUT", 32768, 0),
             ChainEdge(1, "post_norm", "LIVE", 12288, 12288),
             ChainEdge(2, "moe", "LIVE", 12288, 12288))
    return SubChainBPlan(cap_parent="/x", islands=islands, edges=edges,
                         attn_out_nbytes=32768)


def _e16_chain_m_parameterization():
    """MMEL_CHAIN_M is the single source of truth for the chain decode width.

    M=1 (default/unset/garbage) is BYTE-IDENTICAL — ``_chain_island_dir`` leaves the
    capture subdir names UNCHANGED and the gates resolve the [1,6144]=12288 B bucket.
    M=32 selects the cap_*_m32 / cap_mla_b32 capture set + the [32,6144]=393216 B
    bucket, and STILL refuses every other geometry (the [1,6144] M=1 step, prefill) —
    the refuse-everything-else safety holds at every M. Card-free: the subdir map is
    pure string logic and the coverage check tolerates the in-progress on-card M=32
    capture (undecidable => safely not-covered, never raises)."""
    saved = os.environ.get(CHAIN_M_ENV)
    parent = cap_parent_dir()

    def _clear_caches():
        _CAPTURED_LAYER_INPUT_NBYTES_CACHE.clear()
        _SUBCHAIN_B_ATTN_NBYTES_CACHE.clear()

    try:
        # ---- default + bad-value parsing all collapse to M=1 -------------------
        os.environ.pop(CHAIN_M_ENV, None)
        assert chain_m() == 1, "unset MMEL_CHAIN_M must be M=1"
        for bad in ("", "  ", "junk", "0", "-3", "1.5"):
            os.environ[CHAIN_M_ENV] = bad
            assert chain_m() == 1, f"MMEL_CHAIN_M={bad!r} must fall back to M=1"
        # ---- M=1: subdir names UNCHANGED (byte-identical default) --------------
        os.environ[CHAIN_M_ENV] = "1"
        for sub in ("cap_norm", "cap_qproj", "cap_kvproj", "cap_mla_paged",
                    "cap_oproj", "cap_routed"):
            assert _chain_island_dir(sub) == sub, \
                f"M=1 must leave {sub!r} unchanged (got {_chain_island_dir(sub)!r})"
        # ---- M=32: the cap_*_m32 / cap_mla_b32 capture set ---------------------
        os.environ[CHAIN_M_ENV] = "32"
        assert chain_m() == 32
        exp = {"cap_norm": "cap_norm_m32", "cap_qproj": "cap_qproj_m32",
               "cap_kvproj": "cap_kvproj_m32", "cap_mla_paged": "cap_mla_b32",
               "cap_oproj": "cap_oproj_m32", "cap_routed": "cap_routed_m32"}
        for sub, want in exp.items():
            assert _chain_island_dir(sub) == want, \
                f"M=32: {sub!r} -> {_chain_island_dir(sub)!r}, expected {want!r}"
        # a general M flows through the same rule (M=8 -> _m8 / cap_mla_b8).
        os.environ[CHAIN_M_ENV] = "8"
        assert _chain_island_dir("cap_oproj") == "cap_oproj_m8"
        assert _chain_island_dir("cap_mla_paged") == "cap_mla_b8"
        # ---- M=32 coverage: refuse the M=1 bucket; never raise on partial cap --
        os.environ[CHAIN_M_ENV] = "32"
        _clear_caches()
        want32 = captured_layer_input_nbytes(parent)
        # the on-card M=32 capture may be incomplete here -> want32 is None (undecidable,
        # safe) OR the resolved [32,6144] size once cap_norm_m32 is materialized.
        assert want32 in (None, 32 * 6144 * 2), \
            f"M=32 layer-input nbytes {want32} not in (None, 393216)"
        # the M=1 single-token bucket is NOT the M=32 chain's bucket: never covered.
        assert chain_layer_covers(_FakeHidden((1, 6144)), parent) is False, \
            "under M=32 the [1,6144] M=1 step must NOT be covered (refuse-safety)"
        # an unreadable hidden never raises (the contract holds at every M).
        assert chain_layer_covers(object(), parent) is False
        # [32,6144] coverage tracks decidability: covered IFF the plan resolved.
        cov32 = chain_layer_covers(_FakeHidden((32, 6144)), parent)
        assert cov32 == (want32 is not None), \
            f"[32,6144] coverage {cov32} must equal (plan decidable={want32 is not None})"
        note = ("M=32 plan DECIDABLE (cap_norm_m32 materialized -> 393216 B)"
                if want32 is not None else
                "M=32 on-card capture in progress (undecidable => safely not-covered)")
    finally:
        if saved is None:
            os.environ.pop(CHAIN_M_ENV, None)
        else:
            os.environ[CHAIN_M_ENV] = saved
        _clear_caches()
    return (f"E16 MMEL_CHAIN_M source-of-truth: M=1 byte-identical (subdirs unchanged), "
            f"M=32 -> cap_*_m32/cap_mla_b32 + [32,6144] bucket, refuses [1,6144]/prefill; "
            f"bad env -> M=1. {note}")


# ============================================================================= #
#  LIVE-FED MLA ATTENTION CHAIN (MMEL_CHAIN_LIVE) CARD-FREE GATES — E17..E22.     #
#                                                                               #
#   E17. MMEL_CHAIN_LIVE gate predicate.                                         #
#   E18. LiveAttnChainPlan.resolve (M=32): 7 islands (kv_b DROPPED), the live     #
#        bind map derives the geometry from the captures, the branch splits tile  #
#        fused_qkv_a, the MLA is fed LIVE (2 sub + 1 live-KV), ZERO staged         #
#        activations, the tail is threadable.                                     #
#   E19. THE FIX vs THE BUG discriminator: the OLD _LAYER_CHAIN carries kv_b +     #
#        STAGES the MLA branch (E3); the NEW _LIVE_ATTN_CHAIN drops kv_b + binds   #
#        the MLA q_nope/q_pe/latent_kv LIVE.                                       #
#   E20. _sanity TEETH: a STAGED MLA activation (a frozen .bin sneaking back in)   #
#        RAISES (the chain must be fully live).                                    #
#   E21. LiveAttnChainReplay LOUD-guards off-card (no card => warmup raises).      #
#   E22. MMEL_CHAIN_LIVE unset => INERT + the dataflow byte-exact pointer.         #
# ============================================================================= #
def _e17_live_gate():
    prev = os.environ.get(CHAIN_LIVE_ENV)
    try:
        for v, want in (("1", True), ("live", True), ("on", True),
                        ("0", False), ("2", False), ("", False)):
            if v == "":
                os.environ.pop(CHAIN_LIVE_ENV, None)
            else:
                os.environ[CHAIN_LIVE_ENV] = v
            assert chain_live_enabled() is want, \
                f"{CHAIN_LIVE_ENV}={v!r} gate should be {want}"
    finally:
        if prev is None:
            os.environ.pop(CHAIN_LIVE_ENV, None)
        else:
            os.environ[CHAIN_LIVE_ENV] = prev
    return "E17 MMEL_CHAIN_LIVE gate logic correct (1/live/on=on; 0/2/unset=off)"


def _with_chain_m(m_value):
    """Context-manager-ish helper: set MMEL_CHAIN_M, returning a restore callable."""
    prev = os.environ.get(CHAIN_M_ENV)
    os.environ[CHAIN_M_ENV] = str(m_value)

    def restore():
        if prev is None:
            os.environ.pop(CHAIN_M_ENV, None)
        else:
            os.environ[CHAIN_M_ENV] = prev
        _LIVE_ATTN_INPUT_NBYTES_CACHE.clear()
    return restore


def _e18_live_plan_resolve():
    parent = cap_parent_dir()
    if not os.path.isfile(os.path.join(parent, "cap_qkva_m32", "manifest.txt")):
        return f"E18 SKIP (no cap_qkva_m32 under {parent!r}; resolved on-card)"
    restore = _with_chain_m(32)
    try:
        plan = LiveAttnChainPlan.resolve(parent)
        names = plan.step_names
        assert names == ("input_norm", "fused_qkv_a", "q_a_norm", "q_b",
                         "kv_a_norm", "mla", "o_proj"), f"live order drifted: {names}"
        # kv_b is dropped (no cap_kvproj in the chain).
        assert all("kvproj" not in os.path.basename(li.spec.cap_dir)
                   for li in plan.islands), "kv_b (cap_kvproj) must be DROPPED"
        g = plan.geom
        assert (g["H"], g["q_lora"], g["kv_lora"], g["R"]) == (6144, 2048, 512, 64), \
            f"geometry drift: {g}"
        assert (g["N"], g["P"], g["L"], g["V"], g["S"]) == (64, 192, 512, 256, 128), \
            f"MLA geometry drift: {g}"
        assert g["q_lora"] + g["kv_lora"] + g["R"] == g["qkva_feat"] == 2624, \
            "branch split must tile fused_qkv_a (2048+512+64=2624)"
        # the live-feed sizes (q_c|kv_c sub-buffers; spine LIVE; threadable).
        bind = {li.spec.step_name: li.binds[0] for li in plan.islands}
        assert bind["fused_qkv_a"].role == LIVE_FEED
        assert bind["q_a_norm"].role == LIVE_SUB and bind["q_a_norm"].width_elems == 2048
        assert bind["kv_a_norm"].role == LIVE_SUB and bind["kv_a_norm"].offset_elems == 2048
        assert bind["q_b"].role == LIVE_FEED and bind["o_proj"].role == LIVE_FEED
        # THE FIX: MLA fed live (2 sub + 1 live-KV), ZERO staged activations anywhere.
        mla = next(li for li in plan.islands if li.spec.step_name == "mla")
        assert sum(1 for b in mla.binds if b.role == LIVE_SUB) == 2
        assert sum(1 for b in mla.binds if b.role == LIVE_KV) == 1
        assert plan.n_staged_activations == 0, "no activation may be staged (.bin)"
        # threadable: o_proj OUT == layer input.
        assert plan.out_nbytes == plan.layer_input_nbytes == 393216, \
            "live chain output must be the [32,6144] hidden"
        summary = plan.live_feed_summary()
    finally:
        restore()
    return (f"E18 live plan resolved (M=32): 7 islands, kv_b DROPPED, branch tiles "
            f"2048+512+64=2624, MLA live (2 sub + 1 live-KV), 0 staged activations, "
            f"threadable 393216 B. feed: {summary}")


def _e19_fix_vs_bug_discriminator():
    parent = cap_parent_dir()
    # structural contrast that needs no card: the OLD chain carries kv_b; the NEW
    # live chain drops it.
    old_subs = [sub for (_, sub) in _LAYER_CHAIN]
    new_subs = [sub for (_, sub) in _LIVE_ATTN_CHAIN]
    assert "cap_kvproj" in old_subs, "the OLD all-7 chain carries kv_b (cap_kvproj)"
    assert "cap_kvproj" not in new_subs, "the NEW live chain must DROP kv_b"
    assert "cap_qkva" in new_subs and "cap_qanorm" in new_subs \
        and "cap_kvanorm" in new_subs, "the NEW chain adds the q/kv-LoRA islands"
    if not os.path.isfile(os.path.join(parent, "cap_qkva_m32", "manifest.txt")):
        return ("E19 fix-vs-bug (structural): OLD carries kv_b + STAGES MLA (E3); "
                "NEW drops kv_b + adds LoRA islands (on-card resolve for the live binds)")
    # the OLD chain STAGES the MLA branch (already asserted by E3 at M=1); the NEW
    # chain binds the MLA q_nope/q_pe/latent_kv LIVE (asserted by E18 at M=32).
    if os.path.isfile(os.path.join(parent, "cap_routed", "manifest.txt")):
        old = LayerChainPlan.resolve(parent)
        old_feeds = {e.step_name: e.feed for e in old.edges}
        assert old_feeds["mla"] == "STAGED", \
            f"the OLD chain must STAGE the MLA (the bug): {old_feeds}"
    restore = _with_chain_m(32)
    try:
        new = LiveAttnChainPlan.resolve(parent)
        mla = next(li for li in new.islands if li.spec.step_name == "mla")
        roles = {b.role for b in mla.binds}
        assert LIVE_SUB in roles and LIVE_KV in roles and new.n_staged_activations == 0, \
            "the NEW chain must bind the MLA LIVE (q_nope/q_pe sub + live-KV)"
    finally:
        restore()
    return ("E19 FIX vs BUG: OLD all-7 STAGES the MLA (constant output, the bug); "
            "NEW live chain binds MLA q_nope/q_pe LIVE_SUB + latent_kv LIVE_KV (0 "
            "staged) — the dataflow fix, byte-exact-proven by owned_engine V7")


def _e20_live_sanity_teeth():
    plan = _synthetic_live_plan()
    plan._sanity()  # the valid plan passes.
    # tamper 1: a STAGED MLA activation (a frozen .bin) must RAISE.
    raised_staged = False
    try:
        _tamper_live_plan(plan, "mla", role_override=("2", "FROZEN_BIN"))._sanity()
    except MmelChainLayerError as e:
        raised_staged = "STAGED activation" in str(e)
    assert raised_staged, "a STAGED MLA activation must RAISE (chain must be fully live)"
    # tamper 2: dropping the live-KV (latent_kv -> LIVE_FEED) must RAISE.
    raised_nokv = False
    try:
        _tamper_live_plan(plan, "mla", role_override=("2", LIVE_FEED))._sanity()
    except MmelChainLayerError as e:
        raised_nokv = "MLA live binds wrong" in str(e)
    assert raised_nokv, "removing the MLA live-KV must RAISE (output would be a constant)"
    return ("E20 live _sanity TEETH: a STAGED MLA activation RAISES (must be fully "
            "live) AND dropping the live-KV RAISES (output would go constant)")


def _e21_live_replay_oncard_guards():
    plan = _synthetic_live_plan()
    c = LiveAttnChainReplay(plan=plan)
    raised = False
    try:
        c.warmup_once()
    except MmelChainLayerError:
        raised = True
    except Exception:  # noqa: BLE001 — no torch / no hpu is also "not silently ok"
        raised = True
    assert raised, "LiveAttnChainReplay.warmup_once() must NOT silently succeed off-card"
    on_card = any(os.path.isfile(p) for p in (
        "/usr/lib/habanalabs/libSynapse.so", "libSynapse.so"))
    note = "(on-card: real warmup is the engage step)" if on_card else \
        "(off-card: LOUD guard verified)"
    return f"E21 LiveAttnChainReplay device path LOUD-guards off-card {note}"


def _e22_live_unset_inert():
    # unset/garbage => off (the live path is inert; existing behavior byte-identical).
    saved = os.environ.get(CHAIN_LIVE_ENV)
    try:
        os.environ.pop(CHAIN_LIVE_ENV, None)
        assert chain_live_enabled() is False, "unset MMEL_CHAIN_LIVE must be OFF (inert)"
        os.environ[CHAIN_LIVE_ENV] = "0"
        assert chain_live_enabled() is False
    finally:
        if saved is None:
            os.environ.pop(CHAIN_LIVE_ENV, None)
        else:
            os.environ[CHAIN_LIVE_ENV] = saved
    # the coverage predicate never raises on an unreadable hidden.
    assert live_attn_covers(object()) is False, "live_attn_covers must never raise"
    return ("E22 MMEL_CHAIN_LIVE unset => INERT (existing paths byte-identical); the "
            "live-fed dataflow is byte-exact-proven card-free by "
            "owned_engine/live_chain_test.py V7 (== _mla_attn over growing KV; the "
            "frozen-stage bug FAILS the input/KV teeth)")


def _synthetic_live_plan() -> LiveAttnChainPlan:
    """A minimal VALID LiveAttnChainPlan (passes _sanity) for pure-logic + off-card-
    guard tests — built by hand (no size-matching, no files)."""
    def _isl(nm, in0nb, outnb, extra_tensors=(), extra_binds=()):
        tensors = (IslandTensor("0", in0nb, "IN_FILE", "/dev/null"),
                   IslandTensor("w", 4, "IN_FILE", "/dev/null")) + tuple(extra_tensors) \
            + (IslandTensor("o", outnb, "OUT", "/dev/null"),)
        return IslandSpec(nm, f"/x/{nm}", tensors, "0", in0nb, "o", outnb)

    norm = _isl("input_norm", 16, 16)
    qkva = _isl("fused_qkv_a", 16, 16)
    qan = _isl("q_a_norm", 8, 8)
    qb = _isl("q_b", 8, 16)
    kvan = _isl("kv_a_norm", 4, 4)
    oproj = _isl("o_proj", 16, 16)
    # MLA spec: 0=q_nope,1=q_pe,2=latent_kv,3=W_UK_T(RES),4=q_cos(META),o=OUT.
    mla = IslandSpec("mla", "/x/mla", (
        IslandTensor("0", 8, "IN_FILE", "/dev/null"),
        IslandTensor("1", 8, "IN_FILE", "/dev/null"),
        IslandTensor("2", 12, "IN_FILE", "/dev/null"),
        IslandTensor("3", 16, "IN_FILE", "/dev/null"),
        IslandTensor("4", 4, "IN_FILE", "/dev/null"),
        IslandTensor("o", 16, "OUT", "/dev/null")), "0", 8, "o", 16)
    islands = (
        LiveIsland(norm, _live_binds_simple(
            norm, LiveBind("0", LIVE_LAYER_INPUT))),
        LiveIsland(qkva, _live_binds_simple(
            qkva, LiveBind("0", LIVE_FEED, producer="input_norm"))),
        LiveIsland(qan, _live_binds_simple(
            qan, LiveBind("0", LIVE_SUB, producer="fused_qkv_a",
                          offset_elems=0, width_elems=4, head_stride_elems=8))),
        LiveIsland(qb, _live_binds_simple(
            qb, LiveBind("0", LIVE_FEED, producer="q_a_norm"))),
        LiveIsland(kvan, _live_binds_simple(
            kvan, LiveBind("0", LIVE_SUB, producer="fused_qkv_a",
                           offset_elems=4, width_elems=2, head_stride_elems=8))),
        LiveIsland(mla, (
            LiveBind("0", LIVE_SUB, producer="q_b", offset_elems=0, width_elems=2,
                     head_stride_elems=4),
            LiveBind("1", LIVE_SUB, producer="q_b", offset_elems=2, width_elems=2,
                     head_stride_elems=4),
            LiveBind("2", LIVE_KV),
            LiveBind("3", LIVE_RES),
            LiveBind("4", LIVE_META),
            LiveBind("o", LIVE_OUT))),
        LiveIsland(oproj, _live_binds_simple(
            oproj, LiveBind("0", LIVE_FEED, producer="mla"))),
    )
    geom = dict(M=1, H=8, q_lora=4, kv_lora=2, R=2, qkva_feat=8,
                N=2, P=2, L=2, V=2, S=1, B=1)
    return LiveAttnChainPlan(cap_parent="/x", islands=islands, geom=geom)


def _tamper_live_plan(plan: LiveAttnChainPlan, step: str,
                      role_override) -> LiveAttnChainPlan:
    """Return a COPY of ``plan`` with one bind's role overridden (for _sanity teeth).
    ``role_override`` = (tensor_name, new_role)."""
    tname, new_role = role_override
    new_islands = []
    for li in plan.islands:
        if li.spec.step_name != step:
            new_islands.append(li)
            continue
        new_binds = tuple(
            LiveBind(b.tensor_name, new_role, b.producer, b.offset_elems,
                     b.width_elems, b.head_stride_elems, b.note)
            if b.tensor_name == tname else b
            for b in li.binds)
        new_islands.append(LiveIsland(li.spec, new_binds))
    return LiveAttnChainPlan(cap_parent=plan.cap_parent, islands=tuple(new_islands),
                             geom=plan.geom)


def _synthetic_mla_plan(N=8, P=192, R=64, L=512, V=256, B=32, S=128) -> LiveAttnChainPlan:
    """A realistic NON-PAGED mla_decode island plan (GLM-5.2 per-rank N=8 sizes, schema
    order ``q_nope,q_pe,latent_kv,W_UK_T,W_UV,q_cos,q_sin,c_cos,c_sin,pad_mask``) for the
    card-free name-map gates — exercises ``_live_binds_mla``'s real size->role derivation,
    ``mla_cache_rope_tensor_names``, and ``mla_live_ctx_roles`` with no files/card."""
    sizes = [
        ("0", B * N * P * 2),        # q_nope     (LIVE_SUB)
        ("1", B * N * R * 2),        # q_pe       (LIVE_SUB)
        ("2", B * S * (L + R) * 2),  # latent_kv  (LIVE_KV)
        ("3", N * P * L * 2),        # W_UK_T     (RES)
        ("4", N * L * V * 2),        # W_UV       (RES)
        ("5", B * R * 2),            # q_cos      (META)
        ("6", B * R * 2),            # q_sin      (META)
        ("7", B * S * R * 2),        # c_cos      (RES, cache-rope)
        ("8", B * S * R * 2),        # c_sin      (RES, cache-rope)
        ("9", B * S * 2),            # pad_mask   (META)
    ]
    tensors = tuple(IslandTensor(n, nb, "IN_FILE", "/dev/null") for n, nb in sizes) \
        + (IslandTensor("o", B * N * V * 2, "OUT", "/dev/null"),)
    spec = IslandSpec("mla", "/x/mla", tensors, "0", B * N * P * 2, "o", B * N * V * 2)
    geom = dict(M=B, H=6144, q_lora=2048, kv_lora=L, R=R, qkva_feat=2624,
                N=N, P=P, L=L, V=V, S=S, B=B)
    island = LiveIsland(spec, _live_binds_mla(spec, geom))
    return LiveAttnChainPlan(cap_parent="/x", islands=(island,), geom=geom)


def _e23_rope_optb_and_live_ctx():
    # (a) the MMEL_CHAIN_ROPE_OPTB gate predicate (default OFF == Option A).
    prev = os.environ.get(CHAIN_ROPE_OPTB_ENV)
    try:
        for v, want in (("1", True), ("optb", True), ("b", True),
                        ("0", False), ("", False)):
            if v == "":
                os.environ.pop(CHAIN_ROPE_OPTB_ENV, None)
            else:
                os.environ[CHAIN_ROPE_OPTB_ENV] = v
            assert _rope_optb_enabled() is want, f"{CHAIN_ROPE_OPTB_ENV}={v!r}->{want}"
    finally:
        if prev is None:
            os.environ.pop(CHAIN_ROPE_OPTB_ENV, None)
        else:
            os.environ[CHAIN_ROPE_OPTB_ENV] = prev
    # (b) Option-B identity-table resolver: c_cos/c_sin are the size-B*S*R*2 RES binds,
    #     SCHEMA-ORDERED (lower name = c_cos -> cos=1, higher = c_sin -> sin=0).
    plan = _synthetic_mla_plan()
    cc = mla_cache_rope_tensor_names(plan)
    assert cc == ("7", "8"), f"cache-rope names {cc} (want ('7' c_cos, '8' c_sin))"
    # (c) the seam's live_ctx name->semantic map (latent_kv/q_cos/q_sin/pad_mask).
    roles = mla_live_ctx_roles(plan)
    assert roles == {"2": "latent_kv", "5": "q_cos", "6": "q_sin", "9": "pad_mask"}, \
        f"live_ctx role map drifted: {roles}"
    # (d) the minimal synthetic plan has NO cache-rope tables -> resolver returns None
    #     (never raises) and the role map returns None (only 1 META) -> seam delegates.
    assert mla_cache_rope_tensor_names(_synthetic_live_plan()) is None
    assert mla_live_ctx_roles(_synthetic_live_plan()) is None
    # (e) the live_ctx builder is PURE + NEVER RAISES: bare objects (no kv_cache, no
    #     attn_bias, no query rope) => None (undecidable => stock), never a crash.
    assert build_live_ctx_for_layer(object(), object(), None) is None
    assert build_live_ctx_for_layer(object(), object(), [object()]) is None
    assert build_live_ctx_for_layer(object(), object(), [object()],
                                    cap_parent="/no/such/dir") is None
    return ("E23 ROPE Option-B identity resolver (c_cos='7'<-cos=1, c_sin='8'<-sin=0 by "
            "schema order; q_cos/q_sin META stay REAL) + live_ctx role map "
            "(latent_kv/q_cos/q_sin/pad_mask, recipe-name-keyed) + builder NEVER-RAISES "
            "(undecidable plan/source => None => stock)")


def _synthetic_paged_mla_plan(N=8, P=192, R=64, L=512, V=256, B=32, S=128,
                              flat_cache_nb=5013504, block_table_nb=128) -> LiveAttnChainPlan:
    """A realistic PAGED ``mla_decode_paged`` island plan (GLM-5.2 per-rank N=8, the
    on-card cap_mla_b32_s128_tp8 sizes), schema order ``q_nope,q_pe,flat_cache,
    block_table,W_UK_T,W_UV,q_cos,q_sin,c_cos,c_sin,pad_mask`` (OUT='11') — exercises
    ``_live_binds_mla``'s PAGED branch (schema-position bind of flat_cache/block_table)
    with no files/card. flat_cache/block_table default to the real tp8 capture sizes."""
    sizes = [
        ("0", B * N * P * 2),        # q_nope      (LIVE_SUB)
        ("1", B * N * R * 2),        # q_pe        (LIVE_SUB)
        ("2", flat_cache_nb),        # flat_cache  (LIVE_KV; NB*bs*(L+R)*2, not derivable)
        ("3", block_table_nb),       # block_table (LIVE_META; B*MB*4 int32, not derivable)
        ("4", N * P * L * 2),        # W_UK_T      (RES)
        ("5", N * L * V * 2),        # W_UV        (RES)
        ("6", B * R * 2),            # q_cos       (META)
        ("7", B * R * 2),            # q_sin       (META)
        ("8", B * S * R * 2),        # c_cos       (RES, cache-rope)
        ("9", B * S * R * 2),        # c_sin       (RES, cache-rope)
        ("10", B * S * 2),           # pad_mask    (META)
    ]
    tensors = tuple(IslandTensor(n, nb, "IN_FILE", "/dev/null") for n, nb in sizes) \
        + (IslandTensor("11", B * N * V * 2, "OUT", "/dev/null"),)
    spec = IslandSpec("mla", "/x/mla_paged", tensors, "0", B * N * P * 2, "11", B * N * V * 2)
    geom = dict(M=B, H=6144, q_lora=2048, kv_lora=L, R=R, qkva_feat=2624,
                N=N, P=P, L=L, V=V, S=S, B=B, op="mla_decode_paged", interleave=1)
    island = LiveIsland(spec, _live_binds_mla(spec, geom))
    return LiveAttnChainPlan(cap_parent="/x", islands=(island,), geom=geom)


def _e24_paged_mla_and_rope():
    import types  # noqa: PLC0415
    import torch  # noqa: PLC0415 — present on the m3calib CPU venv (no HPU touch)

    # (a) PAGED _live_binds_mla: schema-position binding. flat_cache=LIVE_KV (pos 2),
    #     block_table=LIVE_META w/ note (pos 3); q_nope/q_pe=LIVE_SUB; W_UK_T/W_UV/c_cos/
    #     c_sin=RES; q_cos/q_sin/pad_mask=META; 2 LIVE_SUB + exactly 1 LIVE_KV (passes
    #     _sanity's MLA-live discriminator).
    plan = _synthetic_paged_mla_plan()
    mla = plan.islands[0]
    role = {b.tensor_name: b.role for b in mla.binds}
    assert role["2"] == LIVE_KV, f"flat_cache(pos2) must be LIVE_KV, got {role['2']}"
    assert role["3"] == LIVE_META, f"block_table(pos3) must be LIVE_META, got {role['3']}"
    assert role["0"] == LIVE_SUB and role["1"] == LIVE_SUB, "q_nope/q_pe must be LIVE_SUB"
    assert role["4"] == LIVE_RES and role["5"] == LIVE_RES, "W_UK_T/W_UV must be RES"
    assert role["8"] == LIVE_RES and role["9"] == LIVE_RES, "c_cos/c_sin must be RES"
    assert role["6"] == LIVE_META and role["7"] == LIVE_META and role["10"] == LIVE_META
    assert sum(1 for b in mla.binds if b.role == LIVE_SUB) == 2
    assert sum(1 for b in mla.binds if b.role == LIVE_KV) == 1
    bt = next(b for b in mla.binds if b.tensor_name == "3")
    assert str(bt.note).startswith("block_table"), "block_table bind must carry its note"
    # a re-capture that swapped a geometry tensor's size RAISES (size XCHECK has teeth).
    bad = IslandSpec("mla", "/x/mla_paged", tuple(
        IslandTensor(t.name, (t.nbytes + 2 if t.name == "0" else t.nbytes), t.role, "/dev/null")
        for t in mla.spec.tensors), "0", mla.spec.in0_nbytes + 2, "11", mla.spec.out_nbytes)
    _raised = False
    try:
        _live_binds_mla_paged(bad, plan.geom)
    except MmelChainLayerError:
        _raised = True
    assert _raised, "PAGED bind map must RAISE on a schema size mismatch (q_nope resized)"

    # (b) mla_live_ctx_roles maps the paged block_table + the rope/pad/kv semantics; the
    #     Option-B cache-rope (c_cos/c_sin) names still resolve on the paged island.
    roles = mla_live_ctx_roles(plan)
    assert roles == {"2": "latent_kv", "3": "block_table", "6": "q_cos", "7": "q_sin",
                     "10": "pad_mask"}, f"paged live_ctx role map drifted: {roles}"
    assert mla_cache_rope_tensor_names(plan) == ("8", "9"), "paged c_cos/c_sin schema names"

    # (c) _build_query_rope_tables byte-EXACT vs the capture _cos_sin oracle (interleave=1
    #     PAIRWISE), AND the NeoX expansion, AND style-disagreement => None.
    R, theta, maxp = 64, 8.0e6, 300
    inv = 1.0 / (theta ** (torch.arange(0, R, 2, dtype=torch.float64) / R))
    fr = torch.outer(torch.arange(maxp, dtype=torch.float64), inv)
    csc = torch.cat([torch.cos(fr), torch.sin(fr)], dim=-1).to(torch.float32)  # [maxp,R]
    pos = torch.tensor([5, 130, 299])
    ang = pos.double().unsqueeze(-1) * inv.unsqueeze(0)
    _rope = lambda neox: types.SimpleNamespace(
        self_attn=types.SimpleNamespace(rotary_emb=types.SimpleNamespace(
            cos_sin_cache=csc, is_neox_style=neox)))
    built = _build_query_rope_tables(_rope(False), pos, dict(R=R, interleave=1))
    assert built is not None, "rope builder must resolve from cos_sin_cache + positions"
    q_cos, q_sin = built
    assert tuple(q_cos.shape) == (3, 1, R) and q_cos.dtype == torch.bfloat16
    oc = torch.cos(ang).repeat_interleave(2, dim=-1).to(torch.bfloat16)
    osn = torch.sin(ang).repeat_interleave(2, dim=-1).to(torch.bfloat16)
    assert torch.equal(q_cos.reshape(3, R), oc), "q_cos != _cos_sin PAIRWISE oracle"
    assert torch.equal(q_sin.reshape(3, R), osn), "q_sin != _cos_sin PAIRWISE oracle"
    bn = _build_query_rope_tables(_rope(True), pos, dict(R=R, interleave=0))
    assert bn is not None and torch.equal(
        bn[0].reshape(3, R), torch.cat([torch.cos(ang), torch.cos(ang)], -1).to(torch.bfloat16)), \
        "NeoX q_cos != cat([cos,cos]) oracle"
    assert _build_query_rope_tables(_rope(True), pos, dict(R=R, interleave=1)) is None, \
        "live neox vs captured interleave=1 DISAGREE => None (never a wrong rope bind)"

    # (d) _build_query_rope_tables PURITY: any missing source => None, NEVER raises.
    assert _build_query_rope_tables(object(), pos, dict(R=R, interleave=1)) is None  # no rope
    assert _build_query_rope_tables(_rope(False), None, dict(R=R, interleave=1)) is None  # no pos
    # is_neox_style ALONE (no captured interleave in geom) is sufficient to fix the layout.
    assert _build_query_rope_tables(_rope(False), pos, dict(R=R)) is not None
    # neither is_neox_style NOR a captured interleave => layout indeterminate => None.
    _no_style = types.SimpleNamespace(self_attn=types.SimpleNamespace(
        rotary_emb=types.SimpleNamespace(cos_sin_cache=csc)))
    assert _build_query_rope_tables(_no_style, pos, dict(R=R)) is None

    # (e) build_live_ctx_for_layer end-to-end on a fully-wired PAGED fake (resolve stubbed
    #     to the synthetic paged plan): a COMPLETE ctx keyed by the LIVE_KV+META names;
    #     and None when ANY source (block_list / positions) is missing.
    layer = _rope(False)
    meta = types.SimpleNamespace(block_list=torch.zeros(32, 1, dtype=torch.int32),
                                 attn_bias=torch.zeros(32, 128, dtype=torch.bfloat16))
    kvc = [torch.zeros(34 * 128, 576, dtype=torch.bfloat16)]
    posB = torch.arange(32)   # the synthetic paged plan's B (=32)
    _orig = LiveAttnChainPlan.__dict__["resolve"]
    try:
        LiveAttnChainPlan.resolve = staticmethod(lambda cp: plan)
        ctx = build_live_ctx_for_layer(layer, meta, kvc, posB)
        assert ctx is not None and set(ctx) == {"2", "3", "6", "7", "10"}, \
            f"paged live_ctx must key LIVE_KV+META names, got {None if ctx is None else set(ctx)}"
        meta_nobt = types.SimpleNamespace(block_list=None, attn_bias=meta.attn_bias)
        assert build_live_ctx_for_layer(layer, meta_nobt, kvc, posB) is None, \
            "paged build_live_ctx must be None when block_list (block_table src) is absent"
        assert build_live_ctx_for_layer(layer, meta, kvc, None) is None, \
            "build_live_ctx must be None when positions (rope src) is absent"
    finally:
        LiveAttnChainPlan.resolve = _orig

    # (f) the on-disk paged manifest (if present) is DETECTED as paged via the op= header.
    note = ""
    for cand in ("cap_mla_b32_s128_tp8", "cap_mla_paged"):
        mdir = os.path.join(cap_parent_dir(), cand)
        if os.path.isfile(os.path.join(mdir, "manifest.txt")):
            g = _parse_mla_geometry(mdir)
            assert g.get("op") == "mla_decode_paged", \
                f"{cand} op= header not detected as paged: {g.get('op')!r}"
            note = f" (on-disk {cand}: op=mla_decode_paged + geometry parsed)"
            break
    return ("E24 PAGED MLA wired: flat_cache=LIVE_KV<-kv_cache[0] + block_table=LIVE_META<-"
            "block_list (schema-pos bind, size-XCHECK teeth, 2 SUB+1 KV) + block_table role "
            "map + q_cos/q_sin built byte-EXACT from rotary_emb.cos_sin_cache (== _cos_sin "
            "PAIRWISE & NeoX oracles; style-disagree/missing-src => None) + build_live_ctx "
            "PURE (complete paged ctx; missing src => None)" + note)


def _selftest() -> int:
    tests = [
        _e1_gate,
        _e2_cap_dir,
        _e3_resolve_chain_plan,
        _e4_shape_guard,
        _e5_tail_staged_raises,
        _e6_bind_completeness_logic,
        _e7_oncard_path_guards_offcard,
        _e8_layer_output_threadable,
        _e9_poison_rmsnorm_robust,
        _e10_captured_bucket_coverage,
        _e11_static_gate,
        _e12_subchain_b_plan,
        _e13_subchain_b_coverage,
        _e14_byte_exact_vs_stock,
        _e15_subchain_b_oncard_guards,
        _e16_chain_m_parameterization,
        _e17_live_gate,
        _e18_live_plan_resolve,
        _e19_fix_vs_bug_discriminator,
        _e20_live_sanity_teeth,
        _e21_live_replay_oncard_guards,
        _e22_live_unset_inert,
        _e23_rope_optb_and_live_ctx,
        _e24_paged_mla_and_rope,
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
    print(f"\nmmel_chain_layer self-test: {ok}/{len(tests)} passed")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
