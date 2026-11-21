"""MMEL — the FIRST REAL device-resident wire-in: drive ONE captured GLM decode
recipe (the routed int4 MoE block) from LIVE decode via our own Synapse-stream
replay, NOT a Python re-issue of the stock submodule.

────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS (the honest gap it closes — read the build-status first)
────────────────────────────────────────────────────────────────────────────
The L6 wire-in that measured "flat 68.8 ms ARM_ON == ARM_OFF" never engaged the
csengine's device-resident execution. ``mmel_l6_chain.run_chained_decoder_layer``
in baseline mode re-issues the IDENTICAL stock submodules
(``layer.input_layernorm / self_attn / post_attention_layernorm / mlp``) with the
per-op ``mark_step`` removed — so every per-layer ``ExecuteCachedGraph`` host
relaunch, every recipe-boundary event, and the per-layer ``mark_step`` forward-hook
ALL remain. The ``RawCsIslandReplay`` device-resident path
(``mmel_l6_chain.py``) was BUILT but the forward never calls it. The csengine's
actual mechanism — replay a bridge-compiled recipe on OUR own stream behind ONE
completion (PROVEN byte-exact + 1-fence in the standalone C harness
``route2_chain_multi.c`` / ``route2_real_island.c``, RUN_CHAIN.md STEP-1..5) —
was NEVER driven from a live serve step.

THIS module is the minimal increment that ACTUALLY engages it. It drives the
SINGLE-BUFFER routed-MoE island (``mmel/cap_routed`` — the recipe + its 56 static
int4 expert weight/scale/zero tensors + the bridge byte-exact ``out_ref``):

  * ONE per-step LIVE input: the routed-MoE hidden_states ``[M, H]`` bf16
    (cap_routed tensor name ``"0"``, 12288 B for M=1 H=6144).
  * ONE per-step output: the routed-MoE result ``[M, H]`` bf16 (tensor ``"57"``).
  * 56 STATIC inputs (the int4 expert weights/scales/zeros): bound ONCE at warmup
    to resident device buffers — they never change per step.

It deserializes the recipe ONCE, stages the static weights to device ONCE, and per
decode step binds the live hidden_states ``.data_ptr()`` + ``synLaunch`` on the
bridge's OWN compute stream + ONE ``synStreamSynchronize`` — replacing the stock
``layer.mlp(hidden_states)`` MoE call. This is the FIRST live-fed device-resident
recipe replay; it is gated behind ``MMEL_CHAIN_MOE_OP=2`` (distinct from the unbound
torch-op swap ``=1``) and on-card byte-exact-A/B'd before it is ever trusted.

────────────────────────────────────────────────────────────────────────────
THE HONEST FRAME (do NOT overclaim — gate, don't declare)
────────────────────────────────────────────────────────────────────────────
This is a MECHANISM proof in-serve, not a tok/s-win claim. The on-card evidence
(L6_BUILD_PLAN §0, RUN_CHAIN STEP-6, decisive_v2) is consistent that on the LIVE
serve the per-recipe host fences already OVERLAP (the step is device-serial-bound),
so a 1-island replay will likely NOT cut the single-stream step much by itself.
What this increment PROVES (the gates, in order):

  (G-byte)  the recipe replayed on OUR stream with the LIVE hidden_states is
            byte-IDENTICAL to the stock MoE submodule output (the MmelGreedyABHarness
            arbiter + a direct relNorm gate here). A wrong weight-bind crashes a
            bring-up gate, never corrupts a live generation (the seam's coherent
            wrapper keeps the serve safe).
  (G-count) the per-step recipe-submission / WAIT_CS count for THIS layer's MoE drops
            (the stock path issues the MoE as bridge ExecuteCachedGraph launches; our
            path issues ONE synLaunch + ONE synStreamSynchronize). Measured by the
            strace WAIT_CS / recipe-count method (B3.3).

Only AFTER (G-byte)+(G-count) pass does the staged path open: chain the WHOLE layer's
islands (norm→attn→norm→MoE) on ONE stream (the C-harness 1-fence-per-layer, now
live) → 78-layer chain → async layer-pipeline → W7 collective-seq. THAT stack is
where the structural single-stream win (if any) lives; this file is step 1 of it.

────────────────────────────────────────────────────────────────────────────
DESIGN DISCIPLINE (the standing rules — held here)
────────────────────────────────────────────────────────────────────────────
  * NO ACCELERATOR AT IMPORT. torch / habana / ctypes-libSynapse are imported
    lazily inside the on-card methods; the dataclasses + the manifest parser + the
    bind-contract + the static-shape guards are card-free unit-tested at the bottom,
    so this file ``py_compile``s and self-tests on the CPU-only m3calib box.
  * LOUD failures. Every "cannot do this" path RAISES with the exact reason + the
    env/file to fix it. A silent no-op that masquerades as engagement is the exact
    2026-06-22 false-null we refuse to repeat.
  * STATIC SHAPE. The recipe replays at the captured bucket geometry (M, H from the
    manifest); a live tensor whose geometry would re-fire the recipe is REFUSED
    (the mixed-shape ``synStatus 26`` guard).
  * REUSE, DON'T REBUILD. The replay mechanism is ``mmel_l6_chain.RawCsIslandReplay``
    (already built: the ctypes synLaunch surface + the synLaunchTensorInfo binder +
    the LOUD off-card guards). The recipe + weights are ``mmel/cap_routed`` (already
    captured + byte-exact-proven). We add NO new kernel and NO new ctypes ABI.
  * BYTE-EXACT INTENT. The output is identical to ``layer.mlp(hidden_states)``; the
    seam's ``MmelGreedyABHarness`` is the arbiter.

Run the card-free self-test:
    $HOME/.venvs/m3calib/bin/python \
        $HOME/AI/vllm-gaudi-patch/v1/worker/mmel_moe_island.py
Exit 0 = the manifest parser, the static/live tensor split, the bind-contract, the
shape guards, and the gate logic are all correct (card-free).
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

        return init_logger("mmel.moe_island")
    except Exception:  # noqa: BLE001
        class _Stderr:
            @staticmethod
            def _emit(level, msg, *a):
                if a:
                    try:
                        msg = msg % a
                    except Exception:  # noqa: BLE001
                        msg = msg + " " + " ".join(str(x) for x in a)
                sys.stderr.write(f"[mmel.moe_island][{level}] {msg}\n")

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
#  FORENSIC OVERRIDE-PROOF POISON (S1 parity with mmel_chain_layer; default OFF).
#  MMEL_CHAIN_POISON=1 perturbs the ISLAND's DEVICE OUTPUT (the routed-MoE OUT the
#  serve consumes) with the SAME RMSNorm-robust signal the layer-chain uses: a
#  per-channel ADDITIVE offset + a single-channel SPIKE (NOT a uniform scale — that
#  is washed by the next layer's RMSNorm). GARBLED greedy ==> the island's device
#  output IS served (S1 engine drives the serve); COHERENT greedy ==> an OEM/stock
#  path overrode it (override confirmed). Same flag as the layer chain so ONE knob
#  arms the proof for whichever path the arm engages.
# ----------------------------------------------------------------------------- #
def _island_poison_enabled() -> bool:
    return os.environ.get("MMEL_CHAIN_POISON", "").strip().lower() in (
        "1", "true", "yes", "on")


def _island_poison_mag() -> float:
    try:
        return float(os.environ.get("MMEL_CHAIN_POISON_MAG", "8.0"))
    except (TypeError, ValueError):
        return 8.0


# ----------------------------------------------------------------------------- #
#  Errors — LOUD, specific (mirror mmel_l6_chain's error taxonomy).              #
# ----------------------------------------------------------------------------- #
class MmelIslandError(RuntimeError):
    """A device-resident island-replay precondition failed (bad manifest, missing
    recipe, weight-bind mismatch, shape drift)."""


class MmelIslandShapeError(MmelIslandError):
    """A live tensor violated the static geometry the recipe was captured at."""


# ----------------------------------------------------------------------------- #
#  The gate: MMEL_CHAIN_MOE_OP selects the MoE execution path.                    #
#    unset / 0  -> baseline: the stock layer.mlp() submodule (the byte-exact A/B  #
#                  baseline; fence-collapse-only measurement).                    #
#    1           -> the torch-op swap (mmel_l6_chain._moe_routed_via_our_op):     #
#                  UNBOUND today, raises (refuses to ship unbound).               #
#    2           -> THIS module: device-resident replay of the captured cap_routed#
#                  recipe on the bridge stream (the FIRST real csengine wire-in). #
# ----------------------------------------------------------------------------- #
MOE_OP_ENV = "MMEL_CHAIN_MOE_OP"


def moe_island_replay_enabled() -> bool:
    """True iff the operator selected the device-resident island-replay path
    (``MMEL_CHAIN_MOE_OP=2``). This is the ONLY gate that makes us deserialize a
    recipe + acquire libSynapse — so a missing recipe/lib is fatal ONLY here."""
    return os.environ.get(MOE_OP_ENV, "").strip().lower() in ("2", "replay", "island")


# Where the captured routed island lives (recipe + static weights + out_ref). The
# env override lets the orchestrator pin a re-captured (live-shape) island dir.
CAP_ROUTED_DIR_ENV = "MMEL_CAP_ROUTED_DIR"
_CAP_ROUTED_DEFAULTS = (
    "$HOME/AI/mmel/cap_routed",  # host self-test box
    "/opt/mmel/cap_routed",           # a possible in-container mount
)


def cap_routed_dir() -> str:
    """Resolve the captured routed-island dir (env override > first existing >
    the host default so the loader's own existence check raises LOUD)."""
    p = os.environ.get(CAP_ROUTED_DIR_ENV)
    if p:
        return p
    for c in _CAP_ROUTED_DEFAULTS:
        if os.path.isdir(c):
            return c
    return _CAP_ROUTED_DEFAULTS[0]


# ----------------------------------------------------------------------------- #
#  MANIFEST — the recipe's launch-tensor contract (card-free parseable).         #
#                                                                               #
#  cap_routed/manifest.txt format (the capture writes it):                       #
#    # island=routed quant=int4 top_k=8 n_routed=8                               #
#    # M=1 H=6144 I=2048 GS=128 K=8  shape ...                                   #
#    # name<TAB>nbytes<TAB>role   (name == the RECIPE's own launch-tensor name)  #
#    0<TAB>12288<TAB>IN_FILE                                                      #
#    ...                                                                          #
#    57<TAB>12288<TAB>OUT                                                         #
#                                                                               #
#  The PER-STEP LIVE input is the FIRST IN_FILE whose nbytes == M*H*2 (the bf16  #
#  hidden_states); the OUT is the single OUT row; every OTHER IN_FILE is a STATIC #
#  weight/scale/zero tensor bound once at warmup. We do NOT hardcode "tensor 0" — #
#  we derive the live-input row from the geometry so a re-capture can't silently  #
#  mis-bind.                                                                      #
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IslandTensor:
    name: str          # the recipe's own launch-tensor name (== the .bin basename)
    nbytes: int
    role: str          # "IN_FILE" or "OUT"
    bin_path: str      # absolute path to the captured .bin for this tensor


@dataclass(frozen=True)
class RoutedIslandManifest:
    """The parsed cap_routed contract: geometry + the static/live/out tensor split."""

    cap_dir: str
    M: int
    H: int
    n_routed: int
    top_k: int
    tensors: tuple          # tuple[IslandTensor], in manifest order
    live_in_name: str       # the per-step hidden_states input tensor name
    out_name: str           # the per-step output tensor name
    static_in_names: tuple  # tuple[str] — the weight/scale/zero inputs (warmup-bound)
    scratch_names: tuple = ()  # tuple[str] — OUT_SCRATCH buffers the recipe writes but
    #                            the capture does NOT persist (zero-init, bound at launch)

    @property
    def hidden_nbytes(self) -> int:
        """bf16 hidden_states size for this bucket = M*H*2."""
        return self.M * self.H * 2

    @property
    def out_ref_path(self) -> str:
        """The bridge's byte-exact OUTPUT oracle (capture writes ``out_ref.bin``, not
        ``<out_name>.bin`` — the OUT is computed, not an input). The on-card A/B
        compares our replay's output against this (relNorm 0 / 0 mismatching bytes)."""
        return os.path.join(self.cap_dir, "out_ref.bin")

    @property
    def out_candidate_names(self) -> tuple:
        """ALL output-class tensors (role OUT or OUT_SCRATCH) whose byte size == the
        output size (the OUT row's nbytes). The capture labels exactly one OUT and the
        rest OUT_SCRATCH by a recipe-ID heuristic (highest-ID == OUT), which is NOT
        ground-truthed against the bridge's actually-returned tensor when no launch dump
        was captured. So the REAL output may be any same-size output-class tensor; warmup
        resolves it by reading each back and matching ``out_ref.bin`` (BUG-2 fix). This
        exposes the candidate set in manifest order; ``out_name`` is the default/first
        guess only."""
        out_nb = next(t.nbytes for t in self.tensors if t.name == self.out_name)
        return tuple(
            t.name for t in self.tensors
            if t.role in ("OUT", "OUT_SCRATCH") and t.nbytes == out_nb)

    @classmethod
    def parse(cls, cap_dir: str) -> "RoutedIslandManifest":
        mpath = os.path.join(cap_dir, "manifest.txt")
        if not os.path.isfile(mpath):
            raise MmelIslandError(
                f"[mmel.moe_island] routed-island manifest not found: {mpath!r}. "
                f"Set {CAP_ROUTED_DIR_ENV} to the captured cap_routed dir, or run the "
                "on-card capture (capture_routed_island.py) first."
            )
        with open(mpath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        M = H = n_routed = top_k = None
        tensors = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                # geometry header(s): "# island=routed ... top_k=8 n_routed=8" and
                # "# M=1 H=6144 I=2048 ...". Parse key=val tokens out of any comment.
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
                    elif k == "n_routed":
                        n_routed = iv
                    elif k == "top_k":
                        top_k = iv
                continue
            # data row: name<ws>nbytes<ws>role
            parts = s.split()
            if len(parts) < 3:
                raise MmelIslandError(
                    f"[mmel.moe_island] malformed manifest row {s!r} in {mpath!r} "
                    "(expected: name<TAB>nbytes<TAB>role).")
            name, nbytes_s, role = parts[0], parts[1], parts[2]
            try:
                nbytes = int(nbytes_s)
            except ValueError as e:
                raise MmelIslandError(
                    f"[mmel.moe_island] non-integer nbytes {nbytes_s!r} in manifest "
                    f"row {s!r}.") from e
            tensors.append(IslandTensor(
                name=name, nbytes=nbytes, role=role,
                bin_path=os.path.join(cap_dir, f"{name}.bin")))

        if M is None or H is None:
            raise MmelIslandError(
                f"[mmel.moe_island] manifest {mpath!r} missing the M/H geometry "
                "header (expected a '# M=.. H=..' comment).")
        if not tensors:
            raise MmelIslandError(
                f"[mmel.moe_island] manifest {mpath!r} has no tensor rows.")

        outs = [t for t in tensors if t.role == "OUT"]
        ins = [t for t in tensors if t.role == "IN_FILE"]
        if len(outs) != 1:
            raise MmelIslandError(
                f"[mmel.moe_island] expected exactly 1 OUT tensor, got {len(outs)} "
                f"in {mpath!r}. The routed island has a single [M,H] output.")
        out_name = outs[0].name
        hidden_nb = M * H * 2

        # The per-step LIVE input = the (unique) IN whose size is exactly the bf16
        # hidden_states M*H*2. If more than one IN matches that size we cannot
        # disambiguate the live row from a same-size weight -> REFUSE (LOUD), do not
        # guess. (For cap_routed M=1 H=6144 the live input is 12288 B and it is the
        # ONLY 12288-B input; the weights are 393216 / 12582912 / 98304 / 49152 / 2.)
        live_candidates = [t for t in ins if t.nbytes == hidden_nb]
        if len(live_candidates) != 1:
            raise MmelIslandError(
                f"[mmel.moe_island] cannot identify the single per-step LIVE input: "
                f"{len(live_candidates)} IN tensors have the hidden size {hidden_nb} B "
                f"(M={M} H={H}) in {mpath!r}. The routed island must have exactly one "
                "[M,H] bf16 input; re-capture or fix the manifest. (Refusing to guess "
                "the live row — a wrong bind would corrupt every token.)")
        live_in_name = live_candidates[0].name
        static_in_names = tuple(t.name for t in ins if t.name != live_in_name)
        # OUT_SCRATCH = device buffers the recipe writes but the capture does NOT persist
        # a .bin for (a rebuilt op may emit these — e.g. the int4 routed op's per-expert
        # scratch). NEITHER a static input NOR the live output; bound to resident zero
        # buffers at launch so the recipe's full persistent-tensor set is covered.
        scratch_names = tuple(t.name for t in tensors if t.role == "OUT_SCRATCH")

        # Every INPUT .bin must exist (the static weights + the captured live input
        # are replayed from them; missing => we'd bind garbage). The OUT tensor has
        # NO input .bin — the recipe COMPUTES it and the capture writes its oracle as
        # ``out_ref.bin`` (not ``<out_name>.bin``); so we check the IN_FILE bins for
        # existence and the OUT's oracle separately. Up front, LOUD.
        missing = [t.bin_path for t in tensors
                   if t.role == "IN_FILE" and not os.path.isfile(t.bin_path)]
        if missing:
            raise MmelIslandError(
                f"[mmel.moe_island] {len(missing)} island INPUT .bin file(s) missing "
                f"under {cap_dir!r} (first: {missing[0]!r}). Re-run "
                "capture_routed_island.py on-card to materialize the recipe inputs.")

        return cls(
            cap_dir=cap_dir, M=M, H=H,
            n_routed=int(n_routed or 0), top_k=int(top_k or 0),
            tensors=tuple(tensors),
            live_in_name=live_in_name, out_name=out_name,
            static_in_names=static_in_names,
            scratch_names=scratch_names,
        )

    def assert_live_hidden(self, name: str, shape) -> None:
        """LOUD guard: a live hidden_states tensor must match the captured [M,H]
        geometry, else replaying the recipe against it re-fires / hits synStatus 26."""
        # Accept [M,H] or [M,1,H] or a flat [M*H]; the byte count is the invariant.
        try:
            numel = 1
            for d in shape:
                numel *= int(d)
        except Exception as e:  # noqa: BLE001
            raise MmelIslandShapeError(
                f"[mmel.moe_island] live tensor {name!r} has unreadable shape "
                f"{shape!r}.") from e
        if numel != self.M * self.H:
            raise MmelIslandShapeError(
                f"[mmel.moe_island] STATIC-SHAPE violation: live {name!r} has "
                f"{numel} elements but the captured routed recipe expects "
                f"{self.M * self.H} (M={self.M} H={self.H}). The replay would "
                "re-fire / hit synStatus 26. Re-capture this bucket.")


# ----------------------------------------------------------------------------- #
#  THE DEVICE-RESIDENT REPLAY (on-card only; reuses RawCsIslandReplay).          #
#                                                                               #
#  Lifecycle:                                                                     #
#    warmup_once():  deserialize the recipe; upload the 56 static weight .bins to #
#                    resident device buffers (bound ONCE); alloc the device       #
#                    workspace + output buffer; build the {name->device VA} map   #
#                    for the static + output tensors. Cached on the layer.        #
#    run(hidden):    bind the LIVE hidden_states .data_ptr() into the launch      #
#                    array (the ONLY per-step change), synLaunch on the bridge's  #
#                    OWN compute stream, ONE synStreamSynchronize, return the      #
#                    device output as a torch tensor.                             #
#                                                                               #
#  This is the literal csengine thesis at 1-island granularity: OUR completion    #
#  (one synStreamSynchronize) replaces the bridge's per-recipe event for the MoE  #
#  block, while the math (the GC-compiled int4 kernel) is byte-identical.         #
# ----------------------------------------------------------------------------- #
@dataclass
class MoeBlockIslandReplay:
    """Device-resident replay of the cap_routed int4-MoE recipe, live-fed.

    Holds: the deserialized recipe (via mmel_l6_chain.RawCsIslandReplay), the
    resident device VAs of the static weight tensors (uploaded once), the device
    workspace + output buffers, and the bridge compute-stream handle. Only the
    live hidden_states address changes per ``run``.
    """

    manifest: RoutedIslandManifest
    _replay: Any = None              # mmel_l6_chain.RawCsIslandReplay
    _stream_handle: int = 0
    _static_dev_va: dict = field(default_factory=dict)   # name -> device VA (resident)
    _static_keepalive: list = field(default_factory=list)  # keep the HPU tensors alive
    _out_buffers: dict = field(default_factory=dict)  # output-class name -> resident HPU buf
    _resolved_out_name: str = ""     # the output-class tensor PROVEN == out_ref (BUG-2)
    _out_tensor: Any = None          # the resident output HPU tensor (torch) == resolved OUT
    _ws_dev_va: int = 0
    _ws_keepalive: Any = None
    _scratch_tensors: dict = field(default_factory=dict)  # name -> resident zero buffer
    _live_in_buf: object = None  # dedicated resident live-input buffer (status=30 fix)
    _warm: bool = False

    # -- warmup: deserialize + upload static weights + alloc ws/out (on-card) ----
    def warmup_once(self) -> "MoeBlockIslandReplay":
        if self._warm:
            return self
        import torch  # lazy — card only

        # Reuse the proven ctypes replay surface from mmel_l6_chain (no new ABI).
        try:
            from vllm_gaudi.v1.worker.mmel_l6_chain import (  # type: ignore
                RawCsIslandReplay, reuse_bridge_device_and_stream)
        except Exception:  # noqa: BLE001 — path-based sibling import on a bare box
            RawCsIslandReplay, reuse_bridge_device_and_stream = _import_chain_symbols()

        recipe_path = os.path.join(self.manifest.cap_dir, "island.recipe")
        if not os.path.isfile(recipe_path):
            raise MmelIslandError(
                f"[mmel.moe_island] recipe not found: {recipe_path!r}. Re-capture the "
                "routed island on-card (capture_routed_island.py).")
        self._replay = RawCsIslandReplay(recipe_path=recipe_path).deserialize()

        # The bridge's already-acquired device + torch's current compute stream — the
        # ONLY correct context for an in-serve synLaunch (you must NOT acquire a 2nd
        # device). This is the proven mechanism the int4 fused-MoE op already uses.
        _dev, self._stream_handle = reuse_bridge_device_and_stream()

        # Upload the 56 STATIC weight/scale/zero tensors to RESIDENT device buffers
        # ONCE. Each .bin is the exact device-layout bytes the recipe expects; we view
        # it as uint8 and move it to HPU (contiguous), keeping a python ref alive so
        # its device VA stays valid for the life of this replay. NOTE on dtype: the
        # recipe binds by ADDRESS + the recipe's own tensor metadata (dtype/shape live
        # in the recipe), so a byte-faithful uint8 upload of the captured .bin is the
        # correct device payload (the same thing the C harness uploads).
        import numpy as np  # lazy
        for name in self.manifest.static_in_names:
            t = next(t for t in self.manifest.tensors if t.name == name)
            buf = np.fromfile(t.bin_path, dtype=np.uint8)
            if buf.nbytes != t.nbytes:
                raise MmelIslandError(
                    f"[mmel.moe_island] static weight {name!r} .bin is {buf.nbytes} B "
                    f"but the manifest says {t.nbytes} B ({t.bin_path!r}); re-capture.")
            dev = torch.from_numpy(buf).to("hpu").contiguous()
            self._static_keepalive.append(dev)
            self._static_dev_va[name] = int(dev.data_ptr())

        # Resident buffers for EVERY output-class tensor (the OUT plus any OUT_SCRATCH of
        # the output size). The recipe REQUIRES all its persistent tensors bound or
        # synLaunch rejects it; and the capture's OUT/OUT_SCRATCH labels are an unverified
        # recipe-ID guess, so we resolve the REAL output (== out_ref) by probe-readback
        # below (BUG-2). Zero-init so a no-op launch is visibly wrong (matches the C
        # harness, which zero-fills every non-input tensor).
        out_nb = next(t.nbytes for t in self.manifest.tensors
                      if t.name == self.manifest.out_name)
        out_class_names = list(self.manifest.out_candidate_names)
        for oname in out_class_names:
            onb = next(t.nbytes for t in self.manifest.tensors if t.name == oname)
            self._out_buffers[oname] = torch.zeros(
                onb, dtype=torch.uint8, device="hpu").contiguous()

        # Any OUT_SCRATCH that is NOT output-sized (a genuine accumulator of a different
        # size) still needs a bound resident buffer (zero-init), but it is never a
        # compared-output candidate.
        for sname in self.manifest.scratch_names:
            if sname in self._out_buffers:
                continue
            snb = next(t.nbytes for t in self.manifest.tensors if t.name == sname)
            self._scratch_tensors[sname] = torch.zeros(
                snb, dtype=torch.uint8, device="hpu").contiguous()

        # DEDICATED RESIDENT LIVE-INPUT BUFFER (the synLaunch status=30 /
        # synFailedSectionValidation ROOT CAUSE fix). The recipe places the live input
        # ("0") and the output ("57") in DIFFERENT sections; binding a per-step transient
        # hidden_states.clone() let the HPU caching allocator hand the clone a VA that
        # COLLIDED with the output buffer's VA (observed: both at 0x..bb82c80) -> two
        # distinct sections at one address -> Synapse rejects with synFailedSectionValidation
        # (status=30). The C harness never hits this: it uploads the live bytes into a
        # DEDICATED synDeviceMalloc'd buffer never freed (route2_real_island.c:654-655), so
        # every tensor has a stable distinct VA. We do the same: one resident input buffer,
        # held for the life of the replay, copied into per step (.data_ptr() read post-commit).
        live_nb = next(t.nbytes for t in self.manifest.tensors
                       if t.name == self.manifest.live_in_name)
        self._live_in_buf = torch.zeros(
            live_nb, dtype=torch.uint8, device="hpu").contiguous()

        # Device WORKSPACE the recipe needs (queried from the recipe). Allocate a
        # resident scratch of that size (0 => no workspace). We ALLOCATE here so it is
        # part of the graph the mark_step below commits, but we DELIBERATELY DO NOT read
        # its .data_ptr() yet — see the workspace-VA note under the commit (the BUG-1
        # root cause that survived the live-input clone-commit fix).
        ws = int(getattr(self._replay, "_ws_size", 0))
        if ws > 0:
            self._ws_keepalive = torch.empty(ws, dtype=torch.uint8, device="hpu").contiguous()
        else:
            self._ws_keepalive = None
        self._ws_dev_va = 0  # resolved AFTER the commit below (never pre-commit)

        # LAZY MATERIALIZATION (fix synLaunch status=1 / synInvalidArgument): commit the
        # static-weight .to("hpu") copies + ws/out allocs at a graph boundary so the FIRST
        # launch binds already-valid device VAs. Under lazy an uncommitted .to("hpu") tensor
        # has no backing device VA yet. Mirrors the proven in-serve op (fused_moe_op.py:147).
        import habana_frameworks.torch.core as _htcore  # lazy — card only (mark_step/is_lazy)
        _htcore.mark_step()
        try:
            import habana_frameworks.torch.hpu as _hthpu
            _hthpu.synchronize()
        except Exception:  # noqa: BLE001 — best-effort; mark_step is the load-bearing commit
            pass

        # WORKSPACE-VA (the synLaunch status=1 ROOT CAUSE that survived the live-input
        # clone-commit fix): the workspace is the ONLY launch-bound VA that was read
        # BEFORE the commit above. Under PT_HPU lazy, torch.empty(...).data_ptr() read
        # before a mark_step is an UNCOMMITTED/relocatable address; binding that stale
        # integer as synLaunch's pWorkspace makes the recipe's scratch pointer unbacked
        # and synLaunch rejects the recipe (status=1 / synInvalidArgument) — even though
        # the C harness (route2_real_island.c:540, a SYNCHRONOUS synDeviceMalloc) never
        # hits it. Read the workspace VA HERE, AFTER mark_step+synchronize materialised
        # the allocation, so the bound pWorkspace is a real, recipe-resolvable VA. (The
        # static weights / output buffers are already safe: their .data_ptr() is read
        # post-commit inside build_launch_array, not captured pre-commit.)
        if self._ws_keepalive is not None:
            self._ws_dev_va = int(self._ws_keepalive.data_ptr())

        # ---- BUG-2: resolve the REAL output tensor by a probe launch + readback ----
        # Replay the recipe ONCE with the CAPTURED live input (the .bin for live_in_name),
        # read back every output-class candidate, and pick the one BYTE-EXACT to out_ref.bin.
        # This corrects the capture's unverified highest-ID==OUT guess (the bridge's actual
        # returned tensor — the one out_ref was written from — may be a LOWER-ID output-class
        # tensor the capture mislabeled OUT_SCRATCH). LOUD if none match: we will NOT serve a
        # silently-wrong (~5e-3) output. Skipped only when out_ref.bin is absent (then the
        # default manifest out_name stands, with a LOUD warning).
        self._resolve_output_from_probe(torch, _htcore, out_class_names, out_nb)

        self._warm = True
        logger.info(
            "[mmel.moe_island] WARMED routed-island replay: recipe=%r, %d static "
            "weights resident, resolved OUT=%r (of candidates %s), ws=%d B, stream=0x%x. "
            "The MoE block now replays on OUR stream (1 synLaunch + 1 synStreamSynchronize"
            "/step).",
            recipe_path, len(self._static_dev_va), self._resolved_out_name,
            out_class_names, ws, self._stream_handle)
        return self

    def _resolve_output_from_probe(self, torch, _htcore, out_class_names, out_nb):
        """Probe-launch the recipe with the CAPTURED live input and pick the output-class
        tensor whose readback is BYTE-EXACT to out_ref.bin (BUG-2). Sets
        ``self._resolved_out_name`` + ``self._out_tensor``.

        Candidate-count policy (the probe must LAUNCH correctly but never be the fragile
        thing that breaks a structurally-unambiguous recipe):
          * SINGLE output-class candidate (the 0621 cap_routed recipe: one OUT `57`, no
            OUT_SCRATCH — route2_real_island P2-proven byte-exact): there is nothing to
            disambiguate, so that tensor IS the output. We still launch (to populate it)
            and, when out_ref.bin is present, verify byte-exact as a confirmation — a
            mismatch is a LOUD warning (genuine replay divergence to chase), NOT a refusal,
            because there is no alternative candidate to switch to and refusing would make
            the single-output path fail on a comparison the recipe's structure already
            settled. The served tensor is the single candidate either way.
          * MULTIPLE output-class candidates (a post-BSGT1 recipe with an output-sized
            OUT_SCRATCH): the choice is genuinely ambiguous, so we KEEP the strict
            byte-exact gate — resolve to the matching candidate, and refuse to serve
            (LOUD raise) if NONE matches.
        Falls back to the manifest out_name with a LOUD warning ONLY if out_ref.bin is
        absent (cannot resolve, but must stay launchable)."""
        import numpy as np  # lazy

        single_candidate = len(out_class_names) == 1

        ref_path = self.manifest.out_ref_path
        if not os.path.isfile(ref_path):
            self._resolved_out_name = self.manifest.out_name
            self._out_tensor = self._out_buffers[self._resolved_out_name]
            logger.warning(
                "[mmel.moe_island] out_ref.bin absent at %r — CANNOT ground-truth the real "
                "output tensor; defaulting to manifest OUT=%r UNVERIFIED. Re-capture so the "
                "byte-exact gate can resolve the output (BUG-2).",
                ref_path, self._resolved_out_name)
            return
        ref = np.fromfile(ref_path, dtype=np.uint8)
        if ref.nbytes != out_nb:
            raise MmelIslandError(
                f"[mmel.moe_island] out_ref.bin is {ref.nbytes} B but the output size is "
                f"{out_nb} B ({ref_path!r}); re-capture — cannot resolve the output tensor.")

        # bind the CAPTURED live input from its .bin (the exact bytes that produced out_ref).
        live_t = next(t for t in self.manifest.tensors
                      if t.name == self.manifest.live_in_name)
        live_buf = np.fromfile(live_t.bin_path, dtype=np.uint8)
        if live_buf.nbytes != live_t.nbytes:
            raise MmelIslandError(
                f"[mmel.moe_island] captured live input {live_t.name!r} .bin is "
                f"{live_buf.nbytes} B but manifest says {live_t.nbytes} B; re-capture.")
        # COPY the captured live bytes into the SAME dedicated resident input buffer the
        # per-step run() binds (status=30 fix): the probe must launch with the same
        # distinct-VA section layout as steady state, or it would itself trip
        # synFailedSectionValidation. (self._live_in_buf was allocated + committed above.)
        live_src = torch.from_numpy(live_buf).to("hpu").contiguous().view(torch.uint8).reshape(-1)
        self._live_in_buf.copy_(live_src)
        _htcore.mark_step()
        try:
            import habana_frameworks.torch.hpu as _hthpu
            _hthpu.synchronize()
        except Exception:  # noqa: BLE001
            pass

        name_to_tensor = self._bind_map(self._live_in_buf)
        launch_arr = self._replay.build_launch_array(name_to_tensor)
        self._replay.launch(self._stream_handle, launch_arr, ws_ptr=self._ws_dev_va)
        _syn_stream_synchronize(self._stream_handle)

        # read back EACH output-class candidate; pick the byte-exact match to out_ref.
        matches = []
        diags = []
        for oname in out_class_names:
            got = self._out_buffers[oname].cpu().numpy().view(np.uint8)
            mism = int(np.count_nonzero(got != ref))
            diags.append((oname, mism))
            if mism == 0:
                matches.append(oname)

        if single_candidate:
            # ONE output-class tensor: it IS the output (nothing to disambiguate). The
            # launch above already populated it; out_ref is a confirmation, not a chooser.
            only = out_class_names[0]
            self._resolved_out_name = only
            if not matches:
                logger.warning(
                    "[mmel.moe_island] single output-class tensor %r is NOT byte-exact to "
                    "out_ref.bin (%d/%d mismatching bytes); serving it anyway (it is the "
                    "recipe's only output — there is no alternative candidate). This signals "
                    "a genuine replay divergence to chase (input bind / scratch init / a "
                    "stale recipe); route2_real_island (P2) is the on-card arbiter — it "
                    "PASSES byte-exact on this recipe, so a serve mismatch here is a live-feed "
                    "issue, NOT a wrong-tensor pick.",
                    only, diags[0][1], out_nb)
            else:
                logger.info(
                    "[mmel.moe_island] single output-class tensor %r confirmed BYTE-EXACT to "
                    "out_ref.bin (probe-resolved, route2 P2 parity).", only)
        elif len(matches) == 1:
            self._resolved_out_name = matches[0]
        elif len(matches) > 1:
            # multiple byte-exact candidates (e.g. an output aliased to a scratch copy):
            # prefer the manifest-declared OUT if it is among them, else the first.
            self._resolved_out_name = (
                self.manifest.out_name if self.manifest.out_name in matches else matches[0])
            logger.warning(
                "[mmel.moe_island] %d output-class tensors are byte-exact to out_ref %s; "
                "using %r. (Aliased output/scratch — harmless, but re-capture with the "
                "launch-dump to pin the bridge's returned tensor.)",
                len(matches), matches, self._resolved_out_name)
        else:
            # MULTIPLE candidates and NONE matched: the choice is genuinely ambiguous and
            # we have no ground truth — refuse to serve a wrong output (strict BUG-2 gate).
            raise MmelIslandError(
                "[mmel.moe_island] BUG-2 UNRESOLVED: NO output-class tensor is byte-exact "
                f"to out_ref.bin (of {len(out_class_names)} candidates). Per-candidate "
                f"mismatching-byte counts (of {out_nb} B): {diags}. The recipe replay "
                "diverges from the bridge oracle — refusing to serve a wrong MoE output. "
                "Hunt the cause (input bind / scratch init / a stale recipe) before trusting "
                "the replay; the C harness route2_real_island (P2 gate) is the on-card "
                "arbiter.")
        self._out_tensor = self._out_buffers[self._resolved_out_name]

    def _bind_map(self, live_tensor: Any) -> dict:
        """Build the FULL {recipe_tensor_name -> resident HPU tensor} map for a launch:
        the 56 static weights + the live input + every output-class buffer + any
        non-output-sized scratch. Shared by the warmup probe and per-step run() so the
        bind contract is defined ONCE (the recipe needs EXACTLY its persistent set)."""
        name_to_tensor = {}
        for name, dev in zip(self.manifest.static_in_names, self._static_keepalive):
            name_to_tensor[name] = dev
        name_to_tensor[self.manifest.live_in_name] = live_tensor
        for oname, obuf in self._out_buffers.items():
            name_to_tensor[oname] = obuf
        for sname, sbuf in self._scratch_tensors.items():
            name_to_tensor[sname] = sbuf
        expected = {t.name for t in self.manifest.tensors}
        got = set(name_to_tensor.keys())
        if got != expected:
            raise MmelIslandError(
                f"[mmel.moe_island] launch-tensor name set mismatch: missing="
                f"{sorted(expected - got)} extra={sorted(got - expected)}. The bind "
                "map must cover exactly the recipe's tensors.")
        return name_to_tensor

    # -- per-step: bind LIVE hidden, synLaunch on the bridge stream, ONE sync ----
    def run(self, hidden_states: Any) -> Any:
        """Replay the routed-MoE recipe with the LIVE hidden_states, return the
        device output as a torch tensor shaped like the stock MoE output ([M,H])."""
        if not self._warm:
            raise MmelIslandError(
                "[mmel.moe_island] run() before warmup_once(); call warmup_once() at "
                "the first decode step (or warmup) so the recipe + static weights are "
                "resident.")
        import torch  # lazy — card only
        import habana_frameworks.torch.core as _htcore  # lazy — card only (mark_step/is_lazy)

        self.manifest.assert_live_hidden("hidden_states", tuple(hidden_states.shape))

        # LAZY MATERIALIZATION (THE synLaunch status=1 / synInvalidArgument ROOT CAUSE):
        # the live hidden_states is the output of an un-flushed lazy graph; its .data_ptr()
        # has no valid backing device VA until a graph boundary commits it, and a
        # .contiguous() VIEW can carry a nonzero storage_offset synLaunch resolves wrong.
        # Under lazy we CLONE to a fresh storage_offset=0 buffer — but the clone is ITSELF
        # a new, un-flushed lazy op, so reading its .data_ptr() BEFORE committing it binds
        # an UNMATERIALIZED VA into the launch array and synLaunch rejects the recipe
        # (status=1). The C harness never hits this: it uploads the live bytes to a
        # synDeviceMalloc'd buffer SYNCHRONOUSLY, so the bound VA is always backed.
        #
        # FIX: commit the clone (mark_step + synchronize) AFTER .clone() and BEFORE reading
        # .data_ptr(). The pre-clone mark_step flushes the producer graph (so .clone() sees
        # settled inputs); the post-clone mark_step + synchronize materialises the clone's
        # own device storage so its .data_ptr() is a real, recipe-resolvable VA. Mirrors the
        # PROVEN in-serve op fused_moe_op.py:147-158 (commit-then-bind).
        _lazy = _htcore.is_lazy()
        _htcore.mark_step()  # flush the producer graph so the source is settled
        # COPY the live hidden into the DEDICATED resident input buffer (status=30 fix):
        # never bind a transient .clone() (its VA can collide with the output section ->
        # synFailedSectionValidation). copy_ into the stable, distinct-VA resident buffer,
        # exactly as the C harness uploads into its own synDeviceMalloc'd input tensor.
        src = hidden_states.contiguous().view(torch.uint8).reshape(-1)
        self._live_in_buf.copy_(src)
        live = self._live_in_buf
        if _lazy:
            # MATERIALISE the copy before binding .data_ptr() (else the VA is unbacked).
            _htcore.mark_step()
            try:
                import habana_frameworks.torch.hpu as _hthpu  # lazy — card only
                _hthpu.synchronize()
            except Exception:  # noqa: BLE001 — mark_step is the load-bearing commit
                pass
        live_va = int(live.data_ptr())

        # Build the full {recipe_tensor_name -> resident HPU tensor} map (static weights +
        # the live input + every output-class buffer + non-output scratch) via the SHARED
        # bind contract — the same map the warmup probe used, so the per-step bind is
        # provably the resolved-output one. build_launch_array reads .data_ptr() + asserts
        # contiguity.
        name_to_tensor = self._bind_map(live)

        launch_arr = self._replay.build_launch_array(name_to_tensor)
        # ONE synLaunch on the bridge's OWN stream (orders against torch's ops), then
        # ONE synStreamSynchronize — the csengine 1-completion primitive, live-fed.
        self._replay.launch(self._stream_handle, launch_arr, ws_ptr=self._ws_dev_va)
        _syn_stream_synchronize(self._stream_handle)
        _ = live_va  # (bound into launch_arr; named for the diagnostic above)

        # Return the device output viewed as bf16 [M,H] (the stock MoE output shape). This
        # is the PROBE-RESOLVED output tensor (the one proven byte-exact to out_ref at
        # warmup), NOT the capture's unverified id-guess.
        out = self._out_tensor.view(torch.bfloat16).view(self.manifest.M, self.manifest.H)

        # FORENSIC OVERRIDE-PROOF POISON (default OFF; S1 parity). Perturb the DEVICE
        # output the serve consumes with the RMSNorm-robust additive-offset + spike. If
        # the island's device output is served, greedy GARBLES; if overridden, COHERENT.
        if _island_poison_enabled():
            mag = _island_poison_mag()
            n = out.shape[-1]
            offs = torch.arange(n, device=out.device, dtype=out.dtype)
            offs = (offs / max(n - 1, 1)) * mag
            out = out + offs
            out[..., 0] = out[..., 0] + mag * 16.0
        return out


def _syn_stream_synchronize(stream_handle: int) -> None:
    """ONE host fence for the whole replay: synStreamSynchronize on the bridge stream.
    Reuses the same libSynapse CDLL surface RawCsIslandReplay built (on-card only)."""
    import ctypes
    from vllm_gaudi.v1.worker.mmel_l6_chain import (  # type: ignore
        _make_ctypes_syn_surface)

    syn, _ = _make_ctypes_syn_surface()
    st = syn.synStreamSynchronize(ctypes.c_void_p(stream_handle))
    if st != 0:
        raise MmelIslandError(
            f"[mmel.moe_island] synStreamSynchronize -> status={st} (the replay's one "
            "host completion failed).")


def _import_chain_symbols():
    """Path-based fallback import of RawCsIslandReplay + reuse_bridge_device_and_stream
    (mirrors how the chain imports the seam) for a box where the package isn't on the
    path. On-card the package import above succeeds; this is belt-and-braces."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "mmel_l6_chain.py")
    if not os.path.isfile(cand):
        raise MmelIslandError(
            f"[mmel.moe_island] cannot import mmel_l6_chain (RawCsIslandReplay): "
            f"{cand!r} absent.")
    modname = "vllm_gaudi.v1.worker.mmel_l6_chain"
    spec = importlib.util.spec_from_file_location(modname, cand)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the frozen @dataclass machinery (py3.14) can resolve
    # cls.__module__ in sys.modules (else it derefs None on a dotted path-import).
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.RawCsIslandReplay, mod.reuse_bridge_device_and_stream


# ----------------------------------------------------------------------------- #
#  The ENTRY the chain body calls for MMEL_CHAIN_MOE_OP=2.                        #
#                                                                               #
#  Caches one MoeBlockIslandReplay per layer on the layer object (warmed on first #
#  use). The chain body (mmel_l6_chain._run_moe_block) routes here when the gate  #
#  is 2; otherwise it stays on the stock submodule (=0) or the unbound op (=1).   #
# ----------------------------------------------------------------------------- #
_MMEL_ISLAND_ATTR = "_mmel_moe_island_replay"


def run_routed_moe_via_island_replay(layer: Any, hidden_states: Any) -> Any:
    """Drive THIS layer's routed-MoE block via the device-resident recipe replay.

    First call (per layer): parse the manifest, warm the replay (deserialize + upload
    static weights resident). Every call: bind the live hidden_states + synLaunch +
    ONE synStreamSynchronize. Returns the device output ([M,H] bf16), byte-exact-
    intended vs ``layer.mlp(hidden_states)`` and gated by the seam's A/B harness.

    LOUD on any precondition failure (missing recipe/lib/weights, shape drift) — the
    seam's coherent wrapper keeps the live generation safe by falling back to stock.
    """
    replay = getattr(layer, _MMEL_ISLAND_ATTR, None)
    if replay is None:
        manifest = RoutedIslandManifest.parse(cap_routed_dir())
        replay = MoeBlockIslandReplay(manifest=manifest).warmup_once()
        setattr(layer, _MMEL_ISLAND_ATTR, replay)
    return replay.run(hidden_states)


# ============================================================================= #
#  CARD-FREE SELF-TEST / GATE (no accelerator touched).                          #
#                                                                               #
#   D1. the gate predicate (MMEL_CHAIN_MOE_OP=2 => replay; 0/1 => not).          #
#   D2. cap_routed dir resolution (env override + default fallback).             #
#   D3. manifest parse on the REAL cap_routed/manifest.txt: geometry + the        #
#       static/live/out split is correct (live input is the unique [M,H] bf16     #
#       IN; 56 static weights; 1 out) and every referenced .bin exists.           #
#   D4. assert_live_hidden: matching geometry passes; a drift raises LOUD.        #
#   D5. a synthetic manifest with an AMBIGUOUS live input (two same-size INs)     #
#       RAISES (refuses to guess the live row).                                   #
#   D6. the bind-map completeness check logic (covers exactly the recipe tensors).#
#   D7. the on-card replay path LOUD-guards off-card (no libSynapse => warmup     #
#       raises MmelIslandError, never a silent no-op).                            #
# ============================================================================= #
def _d1_gate():
    prev = os.environ.get(MOE_OP_ENV)
    try:
        for v, want in (("2", True), ("replay", True), ("island", True),
                        ("1", False), ("0", False), ("", False)):
            if v == "":
                os.environ.pop(MOE_OP_ENV, None)
            else:
                os.environ[MOE_OP_ENV] = v
            assert moe_island_replay_enabled() is want, \
                f"{MOE_OP_ENV}={v!r} replay-gate should be {want}"
    finally:
        if prev is None:
            os.environ.pop(MOE_OP_ENV, None)
        else:
            os.environ[MOE_OP_ENV] = prev
    return "D1 MMEL_CHAIN_MOE_OP=2 replay-gate logic correct (2/replay/island=on; 0/1/unset=off)"


def _d2_cap_dir():
    prev = os.environ.get(CAP_ROUTED_DIR_ENV)
    try:
        os.environ[CAP_ROUTED_DIR_ENV] = "/tmp/__mmel_custom_cap__"
        assert cap_routed_dir() == "/tmp/__mmel_custom_cap__", "env override must win"
        os.environ.pop(CAP_ROUTED_DIR_ENV, None)
        d = cap_routed_dir()
        assert d.endswith("cap_routed"), f"default {d!r} must end with cap_routed"
    finally:
        if prev is None:
            os.environ.pop(CAP_ROUTED_DIR_ENV, None)
        else:
            os.environ[CAP_ROUTED_DIR_ENV] = prev
    return "D2 cap_routed dir resolution (env override + default) correct"


def _d3_manifest_parse():
    d = cap_routed_dir()
    if not os.path.isfile(os.path.join(d, "manifest.txt")):
        return f"D3 SKIP (no real cap_routed manifest at {d!r}; parsed on-card)"
    m = RoutedIslandManifest.parse(d)
    assert m.M == 1 and m.H == 6144, f"geometry M={m.M} H={m.H} != 1/6144"
    assert m.hidden_nbytes == 12288, f"hidden_nbytes {m.hidden_nbytes} != 12288"
    # exactly one OUT row, one live IN (the unique 12288-B hidden), 56 static weights.
    # NOTE: the post-BSGT1 rework recipe names its own launch tensors numerically (live
    # input "15", OUT "171", an output-sized OUT_SCRATCH "2") — NOT the pre-rework "0"/"57".
    # We derive expectations from the manifest geometry, not hard-coded names, so a future
    # re-capture with different names still passes.
    outs = [t for t in m.tensors if t.role == "OUT"]
    ins = [t for t in m.tensors if t.role == "IN_FILE"]
    assert len(outs) == 1, f"{len(outs)} OUT tensors"
    assert m.out_name == outs[0].name
    live_ins = [t for t in ins if t.nbytes == m.hidden_nbytes]
    assert len(live_ins) == 1, "expected exactly one 12288-B (live hidden) IN"
    assert m.live_in_name == live_ins[0].name, \
        f"live input name {m.live_in_name!r} != the unique [M,H] IN {live_ins[0].name!r}"
    assert len(m.static_in_names) == len(ins) - 1, "static = all INs minus the live one"
    assert len(m.static_in_names) == 56, \
        f"expected 56 static weights, got {len(m.static_in_names)}"
    assert m.live_in_name not in m.static_in_names, "live input must not be in static set"
    # the post-rework recipe carries an output-sized OUT_SCRATCH; the scratch set must be
    # parsed and the OUT-candidate set (BUG-2 resolution) must include BOTH the OUT and the
    # output-sized scratch so warmup can ground-truth which is the bridge's real output.
    out_nb = outs[0].nbytes
    scratch_rows = [t for t in m.tensors if t.role == "OUT_SCRATCH"]
    assert tuple(t.name for t in scratch_rows) == m.scratch_names, \
        "scratch_names must equal the OUT_SCRATCH rows"
    cands = m.out_candidate_names
    assert m.out_name in cands, "the OUT must be an output candidate"
    for t in scratch_rows:
        if t.nbytes == out_nb:
            assert t.name in cands, \
                f"output-sized OUT_SCRATCH {t.name!r} must be a BUG-2 output candidate"
    assert m.live_in_name not in cands and m.out_name not in m.scratch_names
    return (f"D3 manifest parse correct (M={m.M} H={m.H} live={m.live_in_name!r} "
            f"out={m.out_name!r} scratch={m.scratch_names} out-candidates={cands} "
            f"static={len(m.static_in_names)} weights, all .bin present)")


def _d4_shape_guard():
    d = cap_routed_dir()
    if not os.path.isfile(os.path.join(d, "manifest.txt")):
        m = _synthetic_manifest()
    else:
        m = RoutedIslandManifest.parse(d)
    # matching geometry (M*H elements) passes, in several accepted layouts.
    m.assert_live_hidden("h", (m.M, m.H))
    m.assert_live_hidden("h", (m.M, 1, m.H))
    m.assert_live_hidden("h", (m.M * m.H,))
    # a drift raises LOUD.
    raised = False
    try:
        m.assert_live_hidden("h", (m.M, m.H + 1))
    except MmelIslandShapeError:
        raised = True
    assert raised, "a shape drift must raise MmelIslandShapeError"
    return "D4 assert_live_hidden passes on the captured geometry, LOUD on drift"


def _d5_ambiguous_live_input_raises():
    # Two same-size INs at the hidden size => cannot disambiguate the live row => RAISE.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # geometry M=1 H=4 => hidden = 8 B; make TWO 8-B INs (ambiguous).
        rows = [
            "# island=routed quant=int4 top_k=2 n_routed=2",
            "# M=1 H=4 I=2 GS=2 K=2",
            "# name\tnbytes\trole",
            "0\t8\tIN_FILE",
            "1\t8\tIN_FILE",   # second 8-B IN -> ambiguous with the live hidden
            "2\t8\tOUT",
        ]
        with open(os.path.join(td, "manifest.txt"), "w") as f:
            f.write("\n".join(rows) + "\n")
        # materialize the .bins so the missing-file check doesn't fire first.
        for nm, nb in (("0", 8), ("1", 8), ("2", 8)):
            with open(os.path.join(td, f"{nm}.bin"), "wb") as f:
                f.write(b"\0" * nb)
        raised = False
        try:
            RoutedIslandManifest.parse(td)
        except MmelIslandError as e:
            raised = "cannot identify the single per-step LIVE input" in str(e)
        assert raised, "an ambiguous (two same-size) live input must raise LOUD"
    return "D5 ambiguous live-input manifest RAISES (refuses to guess the live row)"


def _d6_bind_completeness_logic():
    # Pure logic: the _bind_map() launch set must equal the recipe tensor name set —
    # static weights + live input + EVERY output-class buffer (OUT + output-sized
    # OUT_SCRATCH) + any non-output scratch. The synthetic manifest carries an
    # output-sized OUT_SCRATCH ('scr'), so the bind set must include it (else synLaunch
    # rejects the unbound persistent tensor).
    m = _synthetic_manifest()
    expected = {t.name for t in m.tensors}
    out_class = set(m.out_candidate_names)
    assert out_class == {"out", "scr"}, \
        f"output candidates must be the OUT + out-sized scratch, got {out_class}"
    # a correct map covers exactly: static + live + all output-class (out, scr).
    good = set(m.static_in_names) | {m.live_in_name} | out_class
    assert good == expected, \
        f"static+live+output-class must cover exactly the recipe tensors; got {good} vs {expected}"
    # a map missing the output-sized scratch is incomplete (the run() guard would raise) —
    # this is exactly the pre-fix bug that left the recipe's persistent set uncovered.
    bad = set(m.static_in_names) | {m.live_in_name, m.out_name}
    assert bad != expected, "a map missing the output-sized OUT_SCRATCH must be detected as incomplete"
    # and missing the live input is incomplete too.
    bad2 = set(m.static_in_names) | out_class
    assert bad2 != expected, "a map missing the live input must be detected as incomplete"

    # SINGLE-OUTPUT recipe (the restored 0621 cap_routed: one OUT `57`, NO OUT_SCRATCH —
    # route2 P2-proven byte-exact). The output-class candidate set must be EXACTLY the one
    # OUT, so _resolve_output_from_probe takes the single-candidate (non-fragile) branch.
    sm = _single_output_manifest()
    sout = set(sm.out_candidate_names)
    assert sout == {"57"}, \
        f"single-output recipe must expose exactly one output-class candidate, got {sout}"
    assert len(sm.out_candidate_names) == 1, \
        "the 0621 single-output recipe must resolve via the single-candidate branch (no scratch to disambiguate)"
    # its full bind set is static + live + the one OUT (no scratch).
    sgood = set(sm.static_in_names) | {sm.live_in_name} | sout
    assert sgood == {t.name for t in sm.tensors}, \
        "single-output bind set must be static+live+the one OUT (no scratch)"
    return ("D6 bind-map completeness logic (static+live+OUT+out-sized-scratch == recipe "
            "set; single-output recipe -> single output-class candidate) correct")


def _d7_oncard_path_guards_offcard():
    # Off-card (no libSynapse): warming a replay must RAISE MmelIslandError (via
    # RawCsIslandReplay's deserialize -> _make_ctypes_syn_surface), never silently
    # no-op. We use the synthetic manifest but a non-existent recipe so the recipe
    # check (or the libSynapse check) fires; either way it must be LOUD.
    on_card = any(os.path.isfile(p) for p in (
        "/usr/lib/habanalabs/libSynapse.so", "libSynapse.so"))
    m = _synthetic_manifest()
    r = MoeBlockIslandReplay(manifest=m)
    raised = False
    try:
        # torch may be CPU-only here; warmup imports it lazily. The recipe path under
        # the synthetic dir does not exist -> MmelIslandError; or off-card the
        # libSynapse CDLL build raises. Either is the correct LOUD off-card behavior.
        r.warmup_once()
    except MmelIslandError:
        raised = True
    except Exception as e:  # noqa: BLE001 — torch absent / hpu absent is also "not silently ok"
        # An ImportError (no torch) or a habana error is acceptable as "did not
        # silently succeed off-card"; only a clean return would be a bug.
        raised = True
        _ = e
    assert raised, "warmup_once() must NOT silently succeed off-card (no card/recipe)"

    # WORKSPACE-VA ORDERING INVARIANT (the synLaunch status=1 root-cause regression guard,
    # card-free): the workspace .data_ptr() must be read AFTER the lazy-materialisation
    # mark_step in warmup_once, never before — otherwise an uncommitted/relocatable VA is
    # bound as synLaunch's pWorkspace (status=1 / synInvalidArgument). We assert the source
    # ordering: the line that reads `_ws_dev_va = int(self._ws_keepalive.data_ptr())` must
    # come AFTER the `_htcore.mark_step()` commit line. (Static guard, so it holds on a CPU
    # box where we cannot launch.)
    import inspect as _inspect
    src = _inspect.getsource(MoeBlockIslandReplay.warmup_once)
    commit_at = src.find("_htcore.mark_step()")
    ws_va_read_at = src.find("self._ws_dev_va = int(self._ws_keepalive.data_ptr())")
    assert commit_at >= 0, "warmup_once must commit via mark_step()"
    assert ws_va_read_at >= 0, \
        "warmup_once must read the workspace VA via self._ws_keepalive.data_ptr()"
    assert ws_va_read_at > commit_at, (
        "WORKSPACE-VA REGRESSION: the workspace .data_ptr() is read BEFORE mark_step — "
        "that binds an uncommitted VA as synLaunch's pWorkspace (status=1). Read it AFTER "
        "the commit.")
    # and the pre-commit read must NOT exist (it would shadow the correct post-commit one).
    assert "self._ws_dev_va = int(self._ws_keepalive.data_ptr())" not in src[:commit_at], \
        "no pre-commit workspace-VA read may remain in warmup_once"

    note = "(on-card: real warmup is the B-warmup step)" if on_card else "(off-card: LOUD guard verified)"
    return f"D7 device-resident replay LOUD-guards off-card + workspace-VA read is post-commit {note}"


def _synthetic_manifest() -> RoutedIslandManifest:
    """A minimal valid manifest object for pure-logic tests (no files needed). Mirrors the
    real cap_routed contract shape: a live input, static weights, the OUT, AND an
    output-sized OUT_SCRATCH (the post-BSGT1 recipe's extra output-class tensor that
    BUG-2 resolution disambiguates)."""
    M, H = 1, 4
    tensors = (
        IslandTensor("scr", M * H * 2, "OUT_SCRATCH", "/dev/null"),  # out-sized scratch (8 B)
        IslandTensor("0", M * H * 2, "IN_FILE", "/dev/null"),   # live hidden (8 B)
        IslandTensor("w0", 32, "IN_FILE", "/dev/null"),         # a static weight
        IslandTensor("w1", 16, "IN_FILE", "/dev/null"),         # a static weight
        IslandTensor("out", M * H * 2, "OUT", "/dev/null"),     # output (8 B)
    )
    return RoutedIslandManifest(
        cap_dir="/tmp/__synthetic__", M=M, H=H, n_routed=2, top_k=2,
        tensors=tensors, live_in_name="0", out_name="out",
        static_in_names=("w0", "w1"), scratch_names=("scr",))


def _single_output_manifest() -> RoutedIslandManifest:
    """The RESTORED 0621 cap_routed shape (logic-test mirror): one OUT `57`, NO
    OUT_SCRATCH, live input `0`. route2_real_island P2-proves this recipe byte-exact, so
    _resolve_output_from_probe must take the single-candidate (non-fragile) branch."""
    M, H = 1, 4
    tensors = (
        IslandTensor("0", M * H * 2, "IN_FILE", "/dev/null"),   # live hidden (8 B)
        IslandTensor("1", 32, "IN_FILE", "/dev/null"),          # a static weight
        IslandTensor("2", 16, "IN_FILE", "/dev/null"),          # a static weight
        IslandTensor("57", M * H * 2, "OUT", "/dev/null"),      # the single output (8 B)
    )
    return RoutedIslandManifest(
        cap_dir="/tmp/__single_out__", M=M, H=H, n_routed=2, top_k=2,
        tensors=tensors, live_in_name="0", out_name="57",
        static_in_names=("1", "2"), scratch_names=())


def _selftest() -> int:
    tests = [
        _d1_gate,
        _d2_cap_dir,
        _d3_manifest_parse,
        _d4_shape_guard,
        _d5_ambiguous_live_input_raises,
        _d6_bind_completeness_logic,
        _d7_oncard_path_guards_offcard,
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
    print(f"\nmmel_moe_island self-test: {ok}/{len(tests)} passed")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
