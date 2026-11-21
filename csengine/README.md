# csengine — Rung-2 execution engine for Gaudi-2 single-stream

**Goal:** kill THE WAIT. Submit the decode forward via the OPEN habanalabs
command-submission (CS) ioctl with ALL inter-engine sync done ON-DEVICE
(sync objects + monitors), so the host submits ~once per token instead of
~800 host round-trips (~135 per-CS interrupt-sleep waits @ ~1.78 ms each).

This directory is the proof-of-concept + design. It is **off-card design + a
standalone PoC binary**. The PoC must be run by the main loop on a card with
**no active compute context** (the driver enforces single-compute-ctx per
device; see ARCHITECTURE.md §"Operational constraints").

## Files
### W1 — the libcsengine submit core (the reusable engine)
- `libcsengine.h`     — the API: open/close, CB/DRAM/CQ alloc, kernel SIGNAL/WAIT
                        (on-device SOB+monitor+fence), host-side gen_signal/wait_cb
                        packet builders, generic CB submit, STAGED submission
                        (FIRST..LAST → one completion), non-sleeping host wait +
                        mapped-CQ busy/interrupt wait, per-token address patcher,
                        and the M0 self-diagnosing device-state classifier.
- `libcsengine.c`     — implementation (graduates cs_poc's silicon-proven path).
- `csengine_py.py`    — ctypes bridge for the vLLM-Gaudi worker (same single ctx).
- `w1_test.c`         — W1 on-card acceptance test: 3-engine PDMA→MME→TPC on-device
                        chain with 1 host wait + MME-q16 hand-built-CB acceptance +
                        staged→1-completion + mapped-CQ alloc.
- `W1_BUILD_TEST.md`  — exact BUILD (off-card) + ON-CARD test commands + gates.
- `W2_SCOPE.md`       — route-1 vs route-2 verdict (route-2 committed; recipe packets
                        are opaque below synLaunch, SCAL has no public header).

### M0 PoC + shared
- `ARCHITECTURE.md`   — full design, mechanism map, feasibility verdict, staged plan.
- `cs_poc.c`          — the M0 PoC: PDMA SIGNAL/WAIT chain via CS ioctl with
                        on-device monitor/SOB sync; counts host ioctls + waits.
                        Self-contained; talks the DRM/accel ioctl directly.
- `hl_uapi.h`         — local mirror of the verified UAPI constants/structs
                        (pinned to /usr/include/drm/habanalabs_accel.h).
- `abi_check.c`       — independent verifier: static-asserts the mirror vs the real
                        kernel header (sizes, offsets, ioctl numbers, all flags).
- `build.sh`          — builds abi_check + libcsengine.so/.a + w1_test + cs_poc
                        (plain gcc, no Habana libs, no card needed to compile).
- `RUN_ONCARD.md`     — the EXACT cs_poc command the main loop runs, + pass/fail gates.

## The decisive question this PoC answers
Does CS-level submission with on-device SOB/monitor sync ELIMINATE the host
fences, or does the driver/Synapse force a host fence per CS completion?

Answer mechanism: the PoC submits a multi-engine op chain where engine B's
QMAN waits on a monitor armed on engine A's SOB (on-device). It does exactly
ONE `HL_WAIT_CS` for the whole chain. It prints the host ioctl tally. If the
chain completes correctly with 1 wait for N>1 cross-engine-synced ops, the wait
is deletable at the CS layer. (Kernel-source proof is already in ARCHITECTURE.md
§"Feasibility"; the PoC is the on-silicon confirmation.)
