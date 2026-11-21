# RUN_ONCARD — the EXACT command the main loop runs (after the serve frees a card)

## Preconditions
- At least ONE Gaudi-2 card with **no active compute context** (the driver allows
  exactly one compute ctx per device — `hl_device_open` returns `-EBUSY` otherwise,
  `habanalabs_drv.c` / `device.c:is_compute_ctx_active`). During a live vLLM serve
  all 8 cards are held; run this after the serve stops, or on a card you keep free.
- No special privileges needed: `/dev/accel/accel*` are `crw-rw-rw-` (render group).
- No Habana libraries needed — `cs_poc` talks the DRM/accel ioctl directly.

## Build (no card needed)
```bash
cd $HOME/AI/csengine && bash build.sh
```
This runs the independent ABI check (static-asserts our struct/ioctl mirror against
`/usr/include/drm/habanalabs_accel.h`) and builds `cs_poc`. Both must pass.

## Run (needs a free card)
```bash
cd $HOME/AI/csengine
# auto-pick the first free card; 8-op cross-engine chain; 50 reps:
./cs_poc
# or pin a specific free card and queues / longer chain:
./cs_poc --card 7 --qa 0 --qb 4 --nchain 16 --reps 200
# verbose ioctl trace (debug):
CSPOC_TRACE=1 ./cs_poc --reps 2
```
`--qa/--qb` are Gaudi-2 queue IDs. DEFAULTS ARE NOW PDMA_0_0=0, PDMA_1_0=4
(compute MME_0_0=16, TPC_0_0=20 also fine). **DO NOT use EDMA (8..15):** strike-3
(2026-06-20) was an EDMA-only fault — EDMA queues are `dram_bd` and their HW-queue
CS-completion HBW write RAZWIs, resetting the device so HL_WAIT_CS returns EIO
("Killing CS 1.1"). The driver's own `gaudi2_test_queues()` skips EDMA for exactly
this reason; PDMA/compute are the families it validates. cs_poc now runs a single
SIGNAL-CS self-test first and aborts cleanly (rc=9) if that EIOs, so it can't trip
a second reset. All non-NIC queues advertise sync-stream; the EIO was the dram_bd
completion path, not sync-stream support.

## PASS / FAIL gates (what the output must show)
1. **ABI self-check** prints `hl_cs_chunk=64`, `hl_wait_cs_in=56`,
   `DRM_IOCTL_HL_WAIT_CS=0xc0386443`. (If not → ABI drift, STOP.)
2. **open a free card** prints `accelN: OPEN OK (device_id=..., dram_base=...)`.
   (If "NO FREE CARD" → no card was free; rerun when one is.)
3. **MODE naive** reports `WAIT_CS=<nchain>.0` per step (host fence per op — the
   status-quo shape, N SIGNAL CSs each drained by its own host wait).
4. **MODE csengine** reports `host WAIT_CS=1` per step, `on-device waits armed/step`
   **> 0**, AND every step completed (no FATAL/timeout). This is THE decisive result.
5. **VERDICT / DECISIVE GATE** prints `==> PASS` (process also exits 0). The gate
   requires BOTH `csengine host WAIT_CS/step == 1` AND `on-device WAITs armed > 0`.

NOTE: the default decisive run uses ONLY kernel-built SIGNAL/WAIT CSs — it submits
**no user CB to a QMAN CP**, so it cannot trip the EDMA-CP fault/reset (see root-cause
below). `--probe-usercb` is opt-in and is EXPECTED to fault+reset; never use it for
the decisive run.

### Interpretation (the decisive question)
- `==> PASS` means: a multi-engine cross-synced chain drained with **ONE** host
  HL_WAIT_CS, and the cross-engine dependency was carried **on-device** by armed
  monitors (the WAIT CS on engine B fenced B's CP against engine A's SOB), with zero
  host involvement per hop. → The per-op host fence IS deletable at the CS layer;
  Rung-2 is feasible; proceed to the staged build (ARCHITECTURE.md §Staged plan).
- `==> NOT PROVEN` (exit 7) means csengine did not reduce to 1 host fence, or no
  on-device wait armed (every signal retired before its consumer wait was submitted —
  raise `--nchain`/`--reps`, or report the per-step counts).

## Safety notes
- `cs_poc` only ever touches: its own host CB buffers (mapped USERPTR), its own
  DRAM scratch (4 KB it allocated), and sync objects/monitors the driver hands it
  (>= `first_available_*` reported by HL_INFO_SYNC_MANAGER; the SIGNAL/WAIT CS path
  uses the kernel's own SOB allocator, so no collision with anything).
- It holds the single compute ctx for the few seconds it runs, then `close()`s.
  It will NEVER open a card another ctx holds (EBUSY → skip), so it cannot disturb
  a running serve.
- All ioctls are checked; any unexpected errno aborts LOUDLY with the errno.

## M0 submit-failure DEBUG (2026-06-20): why the first CS got EBUSY, and the decisive rerun

**Symptom seen on-card:** open + INFO + sync_mgr + CB/DRAM alloc all OK, then the
sanity CS failed: DMA `rc=-1` (errno not printed), then sanity-NOP `errno=16 EBUSY`,
`exit 2`. Cards were healthy/idle but had **2 device-critical-errors earlier that
day (recovered)**.

**Diagnosis (cited to the on-box 1.24.0-1007 driver source):**
- `EBUSY(16)` on a CS comes from **exactly one place**: `hl_cs_sanity_checks`
  returns `-EBUSY` when `!hl_device_operational` — `command_submission.c:1381-1382`,
  the FIRST gate of `hl_cs_ioctl` (`:2608`). `hl_device_operational` is false for
  IN_RESET / NEEDS_RESET / MALFUNCTION (`device.c:554-559`).
- The card was **OPERATIONAL at open**: `hl_device_open` also gates on
  `hl_device_operational` (`habanalabs_drv.c:992`), as do INFO and MEM
  (`habanalabs_ioctl.c:1469`, etc.). All succeeded ⇒ NOT a static dirty/NEEDS_RESET
  device at open.
- The DMA and the NOP used the **same queue (8) and the same CB VA**, yet returned
  **different errors** (DMA rc=-1 vs NOP EBUSY). The only way that happens is the
  device flipped to **non-operational between the two submits** ⇒ an async
  reset/critical-error landed mid-run (a recurrence of the day's critical error on a
  fragile post-crash card).
- The submission convention was **correct**: queue 8 = `DCORE0_EDMA_0_0`, a
  `QUEUE_TYPE_HW` + `CB_ALLOC_USER` + sync-stream queue (`gaudi2.c:2779-2789`); a
  user-alloc CB on a HW queue is validated only for **address range** by
  `gaudi2_validate_cb_address` (`gaudi2.c:7564-7620`), and a `HL_MEM_USERPTR` map
  returns a VA inside exactly the `pmmu` range that validator accepts
  (`memory.c:3513-3514` host range = `prop->pmmu.start/end`). The successful CB maps
  prove the VA is valid ⇒ an EFAULT CB-address rejection is impossible here.
- `first_sob=437/first_mon=361/first_cq=2` are **driver-reserved constants**, NOT a
  leak (`gaudi2.c:2933` `first_available_user_sob = GAUDI2_RESERVED_SOB_NUMBER +
  num_sync_stream_queues*HL_RSVD_SOBS`; reserved enums `gaudi2P.h:265-303`).

**Verdict: HYPOTHESIS (b) — device went non-operational during the run.** The PoC
code was also hardened so a contributing-cause (b'→a) is now eliminated and the next
run is self-diagnosing (see below).

**Fixes applied to the PoC (off-card, rebuilt clean):**
1. Capture & print the **CS errno** on failure (was dropped before).
2. Probe a **NOP first** (simplest valid work; can't hang a QMAN) to isolate
   device-state from packet-content.
3. Read **device status (HL_INFO_DEVICE_STATUS) + reset counters
   (HL_INFO_RESET_COUNT)** at baseline and after any CS failure, and print an
   explicit **(a) submission-bug vs (b) dirty-device** classification.
4. **Byte-match the LIN_DMA `ctl`** to the kernel's own EDMA memset builder
   (`gaudi2.c:11271-11274`): `MEMSET|WRCOMP|EB`, **no MB** (was `MEMSET|MB`,
   missing WRCOMP/EB — a plausible QMAN-hang trigger).
5. Fix `submit_wait_on` to the driver-accurate "already-completed" semantics
   (rc=0 + seq==ULLONG_MAX, not -EINVAL — `command_submission.c:2367-2397`).
6. Preserve `errno` across the trace `fprintf` in the ioctl wrapper.

### DECISIVE RERUN (operator — needs a free card; reboot first because of (b))
The card that failed needs a reset. A reboot is the clean way (driver reload also
works). The OTHER agents are on the box — coordinate before rebooting.
```bash
# 1) reboot to clear the non-operational/critical-error state
sudo reboot
# 2) after it's back, rebuild (no card needed) and run on a FREE card:
bash build.sh
./cs_poc                       # auto-pick first free card, 8-op chain, 50 reps
# (optional) capture the full ioctl trace:
CSPOC_TRACE=1 ./cs_poc --reps 2 2>&1 | tee /tmp/cs_poc_run.log
```
**Interpreting the rerun:**
- If it now prints `NOP path OK`, `DMA path OK`, then **MODE csengine WAIT_CS=1.0/step**
  with every step completing → **M0 PROVEN**: on-device SOB/monitor sync collapses the
  per-op host fence to 1 wait/step. Proceed to the staged build (ARCHITECTURE.md).
- If a CS fails again, the new **CS-FAILURE DIAGNOSIS** block prints the errno + live
  device status + reset-count delta and states (a) vs (b) directly. EBUSY/EPERM +
  status≠OPERATIONAL or a reset-count tick ⇒ (b) still (card unstable — suspect the
  hardware/fabric state from the earlier critical errors). EFAULT/EINVAL with status
  OPERATIONAL ⇒ a real (a) submission bug to chase from there.

## M0 ROOT-CAUSE (2026-06-20, off-card): why a user CB faulted the QMAN, and the fix

**Symptom (prior on-card):** the CS ioctl is ACCEPTED (errno=0), then the QMAN
**faults EXECUTING the CB** — even a NOP — the device goes IN_RESET, and the driver
self-heals in ~43s. So the syscall path is fine; the **QMAN program content** the
PoC submitted was not something the EDMA CP runs cleanly.

**Cause (cited to on-box 1.24.0-1007 driver source):**
- On Gaudi-2 EVERY compute/DMA queue is `QUEUE_TYPE_HW` (`gaudi2.c:2779`,
  `cb_alloc_flags = CB_ALLOC_USER`). For a HW queue, `hw_queue_schedule_job()`
  pushes the user CB **verbatim** (`hw_queue.c:380-413`) and appends NOTHING:
  **`.add_end_of_cb_packets == NULL`** for Gaudi-2 (`gaudi2.c:12680`; only the older
  `gaudi` ASIC appends end-of-CB packets). Completion is **out-of-band** — a CQ
  monitor armed by `gaudi2_pre_schedule_cs` (`gaudi2.c:11935`) watching a per-CS SOB
  that the QMAN increments on PQE retirement (BD ctl COMP_OFFSET/COMP_DATA,
  `hw_queue.c:393-395`). So no packet in the CB makes completion happen.
- The driver itself **never** runs a user-style LIN_DMA on an EDMA queue through the
  normal CP. Its internal DRAM-memset path first puts the QMAN in **TEST MODE**
  (`gaudi2_qman_set_test_mode` → `QMAN_MAKE_TRUSTED_TEST_MODE`, which trusts CQF+CP
  and **disables PQC**; plus MMU-bypass and the CB at a **physical** HBM address)
  before submitting (`gaudi2.c:11264-11300, 7774-7785`). Normal user operation runs
  with `QMAN_MAKE_TRUSTED` only (PQF|ERR|PQC — CP/CQF **not** trusted,
  `gaudi2.c:5851`). And `gaudi2_test_queues()` (`gaudi2.c:7872`) **explicitly SKIPS
  EDMA queue ids** (`gaudi2_is_edma_queue_id` → continue). The driver therefore never
  smoke-tests an EDMA QMAN via the normal CP path at all.
- ⇒ A bare user CB (even a NOP) submitted to an EDMA QMAN's CP in normal
  (non-test, MMU-on) mode is outside the path the HW/driver supports → the CP faults
  → err-interrupt → device reset. **It is a QMAN-program-content / submission-mode
  issue, not a syscall or ABI bug** (those are all correct).

**The fix (this PoC):** the decisive question — "is the per-CS host fence deletable
via on-device monitors?" — is **engine/work-agnostic** (the fence is a property of CS
*completion*). So the work items are now **SIGNAL and WAIT CSs**, whose CBs are built
**entirely in-kernel** (`gaudi2_gen_signal_cb` / `gaudi2_gen_wait_cb`,
`gaudi2.c:11967 / 12060`) from MSG_SHORT/MON/FENCE packets the HW is guaranteed to
accept (the same family `gaudi2_test_queues` runs successfully). The WAIT CS **is**
the on-device cross-engine dependency: it arms a monitor on the producer's SOB and
ends in a FENCE that stalls the consumer queue's CP on-device until the SOB hits
target. No user CB ever reaches a QMAN CP in the decisive run, so it cannot re-fault.
The old user-CB DMA/NOP submission is preserved behind **`--probe-usercb`** purely to
demonstrate the fault; it is OFF by default and EXPECTED to trip the ~43s reset.

**Decisive rerun (supersedes the DMA/NOP interpretation above):**
```bash
bash build.sh                 # off-card; ABI check + clean -Werror build
./cs_poc                      # auto-pick a FREE card; 8-op SIGNAL/WAIT chain, 50 reps
# verbose ioctl trace:
CSPOC_TRACE=1 ./cs_poc --reps 2 2>&1 | tee /tmp/cs_poc_run.log
```
PASS = `==> PASS` (exit 0): csengine `host WAIT_CS/step == 1` with `on-device waits
armed/step > 0` and every tail wait COMPLETED ⇒ the per-op host fence is deletable at
the CS layer (M0 proven). The device stays OPERATIONAL throughout (no user CB on the
CP), so no reset and no reboot needed between tries.
