# W1 — exact BUILD + ON-CARD TEST commands (operator)

W1 = the libcsengine submit core. **Build is off-card (CPU gcc, no card). The
on-card tests need ONE Gaudi-2 card with no active compute ctx** (run after the
serve stops, or on a card you keep free — the driver allows exactly one compute
ctx/device; a busy card returns EBUSY and is skipped).

This extends the M0 PoC's PDMA-only proof to a **3-engine PDMA→MME→TPC chain that
drains with 1 host wait**, and adds the **MME queue-16 hand-built-CB acceptance**
test (the W2 prerequisite), staged-submission (→1 completion), and the mapped-CQ
counter alloc.

---

## 1. BUILD (off-card — no card needed)
```bash
cd $HOME/AI/csengine
bash build.sh
```
This runs, in order, and ALL must pass:
1. `abi_check` — compiles against the REAL `/usr/include/drm/habanalabs_accel.h`
   and static-asserts the mirror (struct sizes, ioctl numbers, the staged /
   mapped-CQ / interrupt flags). Must print `ABI CHECK PASS` and
   `DRM_IOCTL_HL_WAIT_CS = 0xc0386443`.
2. builds `libcsengine.so` + `libcsengine.a` (the W1 deliverable),
3. builds `w1_test` (links `libcsengine.a`),
4. builds `cs_poc` (the M0 PoC, kept).

Off-card sanity for the Python bridge (no card):
```bash
python3 $HOME/AI/csengine/csengine_py.py --selfcheck
```
Must print `SELFCHECK PASS — libcsengine.so loaded, 13 key symbols present.`

### FINAL BUILD VERIFICATION (the one you asked to run yourself)
```bash
cd $HOME/AI/csengine && bash build.sh && python3 csengine_py.py --selfcheck
# expect: ABI CHECK PASS ; OK -> .../libcsengine.so ... ; SELFCHECK PASS
ls -la libcsengine.so libcsengine.a w1_test abi_check
```

---

## 2. ON-CARD TEST (needs a FREE card)
```bash
cd $HOME/AI/csengine
./w1_test                      # auto-pick first free card; 6-op chain; 50 reps
# or pin a free card / longer chain:
./w1_test --card 7 --nchain 9 --reps 200
# verbose ioctl trace (debug):
CSENG_TRACE=1 ./w1_test --reps 2 2>&1 | tee /tmp/w1_run.log
```
Flags: `--no-partC` (skip the MME hand-built-CB test), `--no-partD` (skip staged).
Parts C/D submit ONLY NOP CBs to MME/TPC — the family `gaudi2_test_queues`
validates — never LIN_DMA/arbitrary programs.

### PASS / FAIL gates (what the output must show)
- **PART A**: `OPEN OK on accelN`, `device status=0 (0=OPERATIONAL)`, sync_mgr
  printed. (If `cseng_open failed: NO FREE CARD` → no card was free; rerun later.
  If it reports a non-operational/dirty device → reboot, then rerun.)
- **self-test**: `single SIGNAL CS retired CLEANLY`. (If it fails with EIO / a
  classifier block → the card is dirty or the queue family won't retire a signal;
  STOP — do not let it trip a second reset.)
- **PART B (THE decisive gate)**: `[csengine] host WAIT_CS/step = 1` AND
  `on-device waits armed/step > 0` → `PART B gate: ... => PASS`. This is the M0
  thesis generalized to **3 different engine families** (PDMA→MME→TPC): the
  on-device monitor/fence carried a cross-FAMILY dependency and the whole chain
  drained with ONE host wait.
- **PART C**: `MME queue 16 ran a hand-built NOP CB and retired CLEANLY`. This
  proves a USER_ALLOC CB on a COMPUTE QMAN's CP retires — the prerequisite for any
  hand-built sync/sampler packets on compute queues in M3. (If it FAILS with a
  classifier block: a NOP on the MME CP faulted → note it; route-2/SCAL does not
  need a hand-built compute CB, so W2 is unaffected, but flag it.)
- **PART D**: `staged 3-CS group drained with host WAIT_CS=1` →
  `PART D gate: ... => PASS`. N CSs, ONE completion.
- **PART E**: `CQ counters CB: handle=... host[0]=0 (read OK)` — the mapped-CQ
  busy-poll surface allocates + is host-readable.
- **W1 VERDICT**: `==> W1 PASS` (process exits 0). Requires Part B PASS + (Part C
  PASS unless skipped) + (Part D PASS unless skipped) + Part E PASS.

### Interpretation
- `==> W1 PASS` means the submit core works on silicon: cross-family on-device sync
  collapses to 1 host wait/step, a compute (MME) CP runs a hand-built CB, staged
  gives one completion, and the mapped-CQ wait surface allocates. → Proceed to W2
  (route-2: drive the GC recipe through `synLaunchWithExternalEvents` with our own
  single completion; see W2_SCOPE.md).
- `==> W1 NOT PROVEN` (exit 7) → the failing PART + any `(a)`-vs-`(b)` device
  classifier block above pinpoint the cause. EBUSY/EPERM or status≠OPERATIONAL or a
  reset-count tick ⇒ (b) dirty card (reboot/retry). EFAULT/EINVAL with status
  OPERATIONAL ⇒ (a) a real submission bug to chase.

---

## 3. Safety notes (same envelope as cs_poc)
- `w1_test` opens only a FREE card (EBUSY → skip); it never disturbs a running
  serve. It holds the single compute ctx for a few seconds, then closes.
- Part B uses ONLY kernel-built SIGNAL/WAIT CSs (no user CB on a CP). Parts C/D put
  a **NOP** user CB on MME/TPC — the safest possible program (no DMA, no addresses).
  Do NOT extend these to LIN_DMA on EDMA (8..15): EDMA is dram_bd and its
  CS-completion HBW write RAZWIs → device reset (RUN_ONCARD.md strike-3).
- Every ioctl is checked; any unexpected errno is captured in `cseng_last_error`
  and the classifier prints the (a)-vs-(b) diagnosis. The library never calls
  exit() — `w1_test` decides when to stop (and stops BEFORE any retry that could
  trip a second reset).

## 4. What gets built (artifacts)
| file | what |
|---|---|
| `libcsengine.so` / `.a` | the reusable submit core (W1 deliverable) |
| `libcsengine.h` | the API |
| `csengine_py.py` | ctypes bridge for the vLLM-Gaudi worker (same-ctx) |
| `w1_test` | the on-card acceptance test |
| `abi_check` | independent ABI verifier vs the kernel header |
| `cs_poc` | the M0 PoC (kept) |
