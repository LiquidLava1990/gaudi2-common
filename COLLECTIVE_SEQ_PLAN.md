# COLLECTIVE_SEQ build plan (#4a) — overlap the 156 TP=8 all-reduce barriers

**Status: OFF-CARD prep only.** No serve, no container, no card touched (a dump is
running). This is a build plan + the device-trace gate + draft wire design + an honest
risk read. Every load-bearing claim is file:line-grounded against the on-box source.

---

## 0. The single most important finding before anyone builds this

**The collective_seq mechanism is already BUILT and card-free-proven — and its own
authors already concluded its single-stream payoff is ~0 and its real win is
CONCURRENCY.** This is NOT a from-scratch build. It is a *finish + decisively measure*
job, and the measurement must be at concurrency, not bs=1.

What already exists on the box:

- `mmel/collective_seq.{h,c}` — the W7 on-device AR-sequencing API on top of
  `csengine/libcsengine.a`. Compiles `-Werror` clean. (`W7_COLLECTIVE_SEQ_READY.md §2`)
- `mmel/colseq_test.c` + built `mmel/colseq_test` — card-free self-test **PASS 10/10,
  deterministic ×3** (`W7_COLLECTIVE_SEQ_READY.md §2`); on-card PART-1 (the PART-B-shaped
  gate) reported **PASS 2026-06-25** in `colseq_py.py:5-6` and the communicator comment
  (`hpu_communicator.py:46-48`).
- `mmel/libcolseq.so` (built 2026-06-24 21:19) + `mmel/colseq_py.py` — the ctypes Python
  surface (statically links libcsengine.a → exports both `cseng_attach` and `colseq_*`).
- A **live wire-in already present** in
  `_src/vllm_gaudi/distributed/device_communicators/hpu_communicator.py:151-158` behind
  `MMEL_COLSEQ=1` (default-off → stock byte-identical).
- The Gaudi-2 kernel reality is settled and verified on-box: the obvious primitive
  `HL_CS_FLAGS_COLLECTIVE_WAIT` is a **hard `-EINVAL` stub**
  (`/usr/src/habanalabs-1.24.0-1007/.../gaudi2/gaudi2.c:12139`, funcs-table `:12725`).
  So in-kernel compute-overlapped collectives are dead silicon; collective_seq instead
  uses the **plain-WAIT-on-a-collective-signaled-SOB** path (W1 PART-B's
  `gen_signal_cb`/`gen_wait_cb`, `gaudi2.c:11967`/`:12060`), which the ASIC *does*
  implement and which PART-B already proved.

**Prior single-stream verdict (do not re-litigate): collective_seq ~0 at bs=1.** The
in-method LEAKMAP (`hpu_communicator.py:122-131`) and `glm52-40plus-autonomous-run.md`
(2026-06-20, agents `a0c311`, `ab7e4b`, `a93e4470`) measured:
- hcclAllReduce **wire = 3.16 ms/step = 0.5%** of the 68.8 ms step (166 calls).
- the AR microbench (`bench/ar_microbench.py`, agent `a93e4470`) isolated the **collective
  at 6.1 µs**; zeroing all 157 all-reduces → 68.8→60.9 ms = 16.4 tok/s (not 40).
- the structural twin (L3/L6 fence-collapse) measured **FLAT** at bs=1
  (`l6_wirein_ab_SUMMARY.txt`: `ARM_OFF 68.8ms | ARM_ON 68.8ms`).

So **the only regime where this lever can pay is concurrency**, where the device is
~44% utilized at bs=256 (`W7_COLLECTIVE_SEQ_READY.md §0`, `glm52-mmel-rebuild.md`
2026-06-24: "AIP/MME ~17% @ c64 = 83% idle") and the per-step AR barriers are on the
critical path of a large batch. **This plan is therefore framed as a concurrency
build**, with the SS arm kept only as the control that should read flat.

---

## A) DEVICE-TRACE PLAN — the make-or-break gate BEFORE building

The decisive question this lever turns on is **not** "are there 156 AR barriers" (there
are: 78 layers × 2 AR/layer, `hpu_communicator._hpu_ar_fence_count` comment
`:174`). It is: **at concurrency, is the consumer compute that follows each all-reduce
actually STALLED waiting on that all-reduce's completion fence (reclaimable), or does
the device have other work to chew while HCCL runs (already overlapped → nothing to
reclaim)?**

### A.1 Decision criterion (state it up front, gate the build on it)

Run the c64 batched decode trace and compute, **per all-reduce barrier**, the device
GAP between the all-reduce kernel finishing and the next consumer op (the next layer's
QKV/gate GEMM or the post-attn projection) starting on the **same device stream**.

- **RECLAIMABLE (BUILD GO):** the device timeline shows a measurable idle gap on the
  compute stream that is *bounded by the host AR fence* — i.e. the consumer op's start
  is gated by the `synEventSynchronize` of the AR recipe-island, not by a data
  dependency on the AR output that is N ops away. Concretely: **if the next op that
  READS the all-reduced tensor is ≥ a few ops downstream (so there is independent work
  to issue while the AR drains), AND the trace shows the device idling at the AR-island
  boundary anyway, there is overlap headroom.** Quantify it as
  `reclaimable_ms/step = Σ (consumer_start − AR_kernel_end) over the 156 barriers`,
  attributed to the host fence (not to a true data-dep wait).
- **NOT RECLAIMABLE (BUILD NO-GO, ship the +15% bsgt1 ceiling):** the consumer op's
  input IS the all-reduced tensor and it is the **very next op** (a hard data
  dependency), with the device busy right up to the barrier. Then on-device sequencing
  buys nothing — you cannot start the consumer before the data it reads exists, host
  fence or not. This is the honest null the design names (`COLLECTIVE_SEQ_DESIGN.md §4
  adversarial #5`: "the consumer can't start until the AR result exists").

The discriminator the trace must produce: **fence-bounded idle vs data-dep idle.** Only
fence-bounded idle is reclaimable by collective_seq.

### A.2 Exact tool + command (already on the box, do NOT build new)

Use the existing batched-decode profiler, `mmel/trace_c64.sh`. It serves
`glm52-conc` with the injected decode profiler (`HPU_PROFILE_DECODE=1`,
`HPU_PROFILE_N=40`), fills a **c64** batch past prefill (100 s), captures 40 batched
steps, and runs the Kineto budget analyzer. Relevant lines:
`trace_c64.sh:20-22` (profiler env), `:35-36` (c64 load), `:40-42` (arm + 40-step
capture), `:46-48` (→ `bench/gather_budget.py`).

```bash
# (after the card is free; this is the GATE, run it FIRST)
cd $HOME/AI && CONC=64 bash mmel/trace_c64.sh
# trace JSON: /mnt/data1/recipe_cache/glm_decode_prof_c64/glm_decode.json
```

**The AR-overlap question needs one analysis the stock `gather_budget.py` does not do**
(it buckets gather/matmul/dequant/attn, `trace_c64.sh:4`). Add a small read-only
analyzer (off-card Python, no card) that, from the same `glm_decode.json`, for each
`hcclAllReduce` event on the main compute stream computes the gap to the next compute
kernel on that stream and classifies it:

```
mmel/ar_overlap_budget.py  (NEW, ~60 lines, pure JSON post-processing — NO card)
  inputs : glm_decode.json (the c64 trace), n_steps
  per AR : t_end(AR_kernel); t_start(next compute kernel, same TID/stream);
           gap = t_start - t_end ;  classify:
             - "fence-bounded" if a synchronizeEvent / synEventSynchronize host event
               brackets the gap and the next op does NOT consume the AR output tensor
               in its first read  -> RECLAIMABLE
             - "data-dep"      if the next op's input == the AR output buffer
               -> NOT reclaimable
  output : reclaimable_ms/step, data_dep_ms/step, and the count split over the ~156 AR
```

Decision: **build GO iff `reclaimable_ms/step` is a meaningful fraction of the c64 step
balloon** (the step goes 68.8→237 ms from bs=1→c64, `trace_c64.sh:4`, `glm52-mmel-
rebuild.md`). If `reclaimable_ms/step` ≳ 20–30 ms (i.e. it explains a real slice of the
~168 ms concurrency balloon), wire it. If it is <~5 ms or all data-dep, **STOP** — the
balloon is the all-expert int4 dequant / gather (the documented c64 pole,
`glm52-mmel-rebuild.md` 2026-06-24: "the real pole = the fixed all-expert int4 dequant,
AIP/MME ~17% @ c64"), not the AR fence, and collective_seq is the wrong lever.

### A.3 Cross-check (cheap, same serve) — does deleting the AR mark_step move c64 at all?

Before the C wire, run the env-only A/B that already exists, **at concurrency** (every
prior AR-markstep test was bs=1, which is why it read flat):

```bash
# A = control ; B = drop the per-AR host mark_step (recipe-segmentation change only)
#   models/glm52-conc-bsgt1.env  + add  -e HPU_AR_COMM_MARKSTEP=0  + -e HPU_AR_FENCE_COUNT=1
# measure c64 step-timer + the [AR-FENCE] count (hpu_communicator.py:171-181), A vs B.
```
`HPU_AR_COMM_MARKSTEP=0` (`hpu_communicator.py:160`) drops the mark_step but keeps HCCL;
`HPU_AR_FENCE_COUNT=1` logs the AR call count. **If even fully dropping the AR
mark_step at c64 doesn't move the step**, the on-device gate (which does strictly less —
it keeps HCCL and only moves the *completion sequencing*) won't either. This A/B is a
~10-minute pre-filter that is much cheaper than the full T2/T4 binding work, and it is
the cleanest test of "is the AR fence on the c64 critical path." Greedy byte-exact must
hold across A/B (`glm52-arfloor.env:25-28` gate).

**Gate summary (A):** GO only if BOTH (i) `ar_overlap_budget.py` shows ≳20 ms/step
reclaimable fence-bounded idle at c64, AND (ii) the `HPU_AR_COMM_MARKSTEP=0` c64 A/B
shows a non-trivial step drop. Either failing → NO-GO, ship bsgt1 +15%.

---

## B) IF reclaimable — the wire design (synEventMapTensor / synStreamWaitEvent)

### B.0 The API surface is CONFIRMED present and answers the user's question

From the on-box bridge headers (`hpu-moe-ext/tpc_kernel/opbackend/bridge/`):

- `synapse_shim/partial_event_emulation.h:43-56` declares the exact surface:
  - `synLaunchWithExternalEvents(stream, launchTensors, nTensors, workspace, recipe,
     synEventHandle* eventList, nEvents, flags)` — launch a recipe and have named
     **external output tensors** signal events on completion.
  - `synEventMapTensor(synEventHandle*, nEvents, const synLaunchTensorInfo*, recipe)` —
     **bind an event to a specific external tensor of a recipe.**
- `synapse_helpers/event.h:88-104` wraps both as the methods we'd actually call:
  - `event::map_event_to_tensor(recipe, tensor_info)` → `synEventMapTensor`
  - `event::stream_wait_event(stream& s, flags)` → **`synStreamWaitEvent`** — makes a
    *different stream* wait on that event **device-side** (`:88-92`).
- `synapse_api_shim.h:27,30,32,103,104` confirms all of `synStreamWaitEvent`,
  `synEventCreate`, `synEventRecord`, `synEventMapTensor`,
  `synLaunchWithExternalEvents` are resolved symbols in the loaded libSynapse.

**Answer to the user's API question: YES.** `synEventMapTensor` binds an event to an
external tensor's completion, and `synStreamWaitEvent` lets a downstream stream wait on
that event **without a host round-trip** (it records a wait on the consumer stream's CP,
gated on the event's backing SOB — the same SOB/monitor end-state collective_seq builds
the hard way via raw CS). This is exactly the P1 mechanism the design's keystone rests
on (`COLLECTIVE_SEQ_DESIGN.md §2.2 P1`, `W7_COLLECTIVE_SEQ_READY.md §1`).

**BUT — the load-bearing caveat (this is where the real risk lives):**
`partial_event_emulation.h:16` says *"The emulator should be removed once feature is
ready on Synapse side"* and `UsePartialEventEmulation()` (`:67`) gates it — i.e. on this
1.24.0 stack the external-event→tensor path may be the **emulation**, not native. And
crucially these are **C++ bridge-internal** APIs; they are **not** exposed to the vLLM
Python layer where `HpuCommunicator.all_reduce` lives. You cannot call
`synEventMapTensor` on the `dist.all_reduce` HCCL op from Python — the HCCL op is issued
deep inside `ProcessGroupHCCL`, and the bridge already wraps it in its own
`synapse_helpers::event` with an `event_done_callback`
(`collective_kernel_info.h:64-65`, cited `COLLECTIVE_SEQ_DESIGN.md §2.2`). So P1 "bind
HCCL's completion event to a SOB we own" requires reaching **inside the bridge**, which
is closed for the HCCL path. **This is why the existing code took the raw-CS route
(libcsengine SIGNAL/WAIT) instead of P1.** See B.3 for the consequence.

### B.1 Where to bind (two designs; the design doc and the live code DISAGREE — resolve it)

There is a real inconsistency that must be fixed before this is trustworthy:

- **The design doc** (`COLLECTIVE_SEQ_DESIGN.md §1`, line "This design does not touch
  `hpu_communicator.py`") says collective_seq belongs in the **MMEL L6 chained-exec
  path** (`mmel_l6_exec.py`/`mmel_l6_chain.py`), where the forward is replayed on *our*
  stream and the all-reduce is one handoff in a device-resident chain. There P1/P2 make
  sense because *we own the consumer op's submission* — we can arm a real on-device WAIT
  on the real next-layer GEMM.

- **The live code** (`hpu_communicator.py:151-158`) DOES touch the communicator and does
  something weaker:
  ```python
  if os.environ.get("MMEL_COLSEQ") == "1":
      cs = self._colseq_ensure()
      slot = ...
      dist.all_reduce(input_, group=self.device_group)  # HCCL unchanged (lazy)
      cs.producer_signal(slot, 0)   # SIGNAL on libcsengine PDMA q0
      cs.consumer_wait(slot, 16)    # WAIT on libcsengine MME q16
  ```
  **This is a no-op for the real forward.** The `producer_signal`/`consumer_wait` run on
  libcsengine's *own* compute-ctx queues (a separate cseng_ctx attached via
  `cseng_attach`, `hpu_communicator.py:92`), which are **completely disconnected** from
  the lazy-mode HPU-graph stream that actually runs the next layer's GEMM. The real
  consumer (next-layer compute) is still issued by the vLLM lazy bridge and is **not**
  gated by `cs.consumer_wait`. So this wiring proves the colseq calls *execute* (the
  PART-1 on-card pass), but it does **not** insert collective_seq into the real decode
  data path. **Measuring this arm and seeing flat ≠ collective_seq is dead** — it's the
  "test in the engine harness, not the old stack" trap (`glm52-mmel-rebuild.md` Discipline
  §). The communicator arm must be treated as a *mechanism smoke-test*, not the A/B.

**Decision: bind in the L6 chain, not the communicator.** The only place collective_seq
can actually delete a critical-path fence is where *we own the consumer op's
submission* — the MMEL L6 chained-exec forward (`mmel/mmel_l6_exec.py`,
`mmel/mmel_l6_chain.py`), where the per-layer all-reduce handoff is between two ops we
replay on our stream. That is the design's intent (`COLLECTIVE_SEQ_DESIGN.md §1`,
`W7_COLLECTIVE_SEQ_READY.md §3.B`: "W7 wires into the MMEL L6 chained-exec path …
replacing the per-all-reduce mark_step handoff with the on-device gate").

> **Honest blocker carried forward:** the L6 chain itself is currently capture-foreign in
> serve (`glm52-mmel-rebuild.md` 2026-06-22-PM / 2026-06-24: the raw-ctypes `synLaunch`
> island replay hits `synLaunch status=1` inside vLLM's `wrap_in_hpu_graph`, and the
> captured routed recipe freezes K-of-256 routing → incorrect). So "wire collective_seq
> into L6 and measure at concurrency" inherits L6's unresolved serve wall. **collective_seq
> cannot be measured end-to-end until the L6 chain runs correctly in serve** — that
> dependency must be stated, not hidden. The clean intermediate is the **P1-via-bridge-
> event** path (B.2), which does NOT need L6.

### B.2 The clean P1 design (preferred if reachable) — bridge external-event, no L6, no raw CS

If (and only if) the AR-overlap gate (A) is GO, the *lowest-risk* wire that does not
depend on the broken L6 chain is to make the **bridge** signal a SOB we own on HCCL
completion and have the next layer's first GEMM wait on it via `synStreamWaitEvent`:

1. At `colseq_create`, reserve a small SOB ring (16 slots, `_COLSEQ_N_SLOTS=16`,
   `hpu_communicator.py:53`) from the user sync-object range
   (`colseq_create` → `cseng_sync_manager` → `first_available_sync_object+i`,
   `collective_seq.h:56-60`). `colseq_slot_sob(slot,&id,&cfg)` (`collective_seq.h:120`,
   `colseq_py.py:84-89`) reports each slot's SOB id + cfg addr.
2. Hand that SOB to the bridge as the backing SOB of HCCL's completion event. **This is
   the unproven link (T4)** and it requires a bridge-side hook because the HCCL event is
   created inside `collective_kernel_info` (`:64-65`), not in our Python. Two sub-options:
   - **(B.2a)** If `PartialEventEmulation`/`synEventMapTensor` can be reached for the
     collective's output tensor from the OpBackend layer we already build
     (`hpu-moe-ext`), bind there. This is the same library family where we already own
     LazyOp+OpBackend kernels (`fused-moe-kernel-validated.md`), so a bridge-adjacent
     hook is in-reach. Needs a card test (T4) to confirm the emulated path increments our
     SOB.
   - **(B.2b)** If not reachable, fall back to P2 (B.3).
3. The consumer (next layer's first matmul) waits on that SOB. In the L6/owned path this
   is `colseq_consumer_wait(slot, MME_q)` (`collective_seq.h:88`). In a bridge-event
   path it is `event::stream_wait_event(consumer_stream)` (`event.h:88-92`) recorded on
   the compute stream before the next-layer GEMM — **device-side, no host fence.**

Byte-exactness: collective_seq changes only *sync segmentation*, never values (HCCL
output bytes are untouched; `W7_COLLECTIVE_SEQ_READY.md §3.B.3`). Gate: greedy logits
A==B at c16 (`glm52-conc-bsgt1.env:54-56` correctness gate).

### B.3 P2 fallback (raw-CS SIGNAL on the completion queue) — needs T2

If the bridge event cannot be bound to our SOB (B.2 fails T4): submit a 1-packet kernel
SIGNAL CS (`colseq_producer_signal(slot, completion_q)`, `collective_seq.h:105`,
`collective_seq.c:186-201`) on the **internal completion EDMA/PDMA queue HCCL drains
to** — which queue that is, is the **T2** card-only unknown (`COLLECTIVE_SEQ_DESIGN.md
§6 T2`; NIC queues are NOT sync-stream, `gaudi2.c:2782-2787`, so the SIGNAL must land on
the completion EDMA/PDMA, not the NIC). `producer_signal` records the slot's producer
seq (`collective_seq.c:198`), so the subsequent `consumer_wait` (which requires a known
producer seq, else `-1/EINVAL`, `collective_seq.c:149-152`) is satisfied. P2 needs no
HCCL cooperation but it needs T2 pinned.

### B.4 Gating + default-off (non-negotiable, already correct in the existing code)

- Flag: `MMEL_COLSEQ=1` (default unset → stock byte-identical, `hpu_communicator.py:151`
  / `:159-162`). Keep it. Add a parallel `MMEL_COLSEQ_L6=1` if the real wire moves to the
  L6 path so the communicator smoke-test arm and the real arm are distinguishable.
- **FAIL LOUD, never silently fall back to the host barrier** — already enforced
  (`_colseq_ensure` raises if init previously failed, `hpu_communicator.py:56-60,104-107`).
  A silent fallback would fabricate a false A/B null. Keep this.
- Engagement banner: the `[W7] MMEL_COLSEQ ENGAGED` warning (`hpu_communicator.py:100`)
  must fire on all 8 ranks or the test is invalid (the `[ENGAGE]/[SKIP]` discipline,
  `glm52-mmel-rebuild.md` 2026-06-22).

---

## C) THE HONEST RISK — why this might succeed where comm-overlap failed, or might not

### C.1 Why comm-overlap (the 3 CUDA fusion passes) was a DIFFERENT, categorically-dead thing

comm-overlap = `compilation_config.pass_config.{enable_sp, fuse_allreduce_rms,
fuse_gemm_comms}` — inductor PostGradPassManager passes imported **behind
`is_cuda_alike()`** and run only under `CompilationMode.VLLM_COMPILE`
(`hpu_communicator.py:133-139`, `glm52-arfloor.env:30-33`). HpuPlatform is out-of-tree
(`is_cuda_alike()=False`) and **forces `CompilationMode.NONE` + lazy**, so those passes
**never run and the symbols don't even exist** (they'd NameError if reached). comm-overlap
wasn't "tried and flat" — it was **never reachable on HPU**. It is dead for a reason that
has nothing to do with collective_seq.

### C.2 Why collective_seq is a genuinely different mechanism (the case FOR)

collective_seq does NOT touch torch.compile / inductor / any CUDA pass. It is a
**below-the-bridge, HPU-native SOB/monitor/FENCE** built from the open habanalabs CS
ioctl via libcsengine — the exact primitive (W1 PART-B) that **passed on this silicon**
(cross-family PDMA→MME→TPC dependency carried on-device, 1 host wait/step,
`COLLECTIVE_SEQ_DESIGN.md §4` table, `W7_COLLECTIVE_SEQ_READY.md §1`). The `-EINVAL`
COLLECTIVE_WAIT stub that kills NVIDIA's in-kernel approach is explicitly **routed
around** (`gaudi2.c:12139` confirmed; the design uses plain WAIT instead, `§2.3`). So the
mechanism is real and silicon-proven where comm-overlap was vaporware. **If** the c64
trace (A) shows fence-bounded idle, this is the right tool to reclaim it.

### C.3 Why it is still LIKELY a partial-to-null win (the case AGAINST — be honest)

Three independent reasons it probably under-delivers, in increasing severity:

1. **The AR is a real data dependency (the design's own adversarial #5,
   `COLLECTIVE_SEQ_DESIGN.md §4`).** The consumer op reads the all-reduced tensor. You
   cannot start it before that tensor exists. collective_seq removes the *host fence*
   serialization, not the *data* dependency. The win is only the host-fence wake latency
   (~1.78 ms IRQ wake the busy-poll work measured, `glm52-single-stream-rootcause.md`)
   **times the fraction of the 156 barriers where the device had nothing else to do**.
   If the trace shows the consumer is the immediate next op (likely for the post-attn and
   post-MoE AR), there is **zero reclaimable gap** there.

2. **The c64 pole is documented to be the all-expert int4 dequant, not the AR fence**
   (`glm52-mmel-rebuild.md` 2026-06-24: "collective ruled out as the pole — microbench
   6.25 ms = 2.6% of the 237 ms c64 step; the real pole = the fixed all-expert int4
   dequant, AIP/MME ~17% @ c64 = 83% idle"). If that holds, the 83% idle is the device
   **waiting on HBM weight reads inside the MoE**, not on AR fences — and collective_seq
   reclaims none of it. The A.2 trace must overturn this for the build to make sense;
   absent that, the prior measurement says NO-GO.

3. **The real wire depends on the broken L6 chain or an unproven bridge-event bind**
   (B.1/B.2). The communicator arm that exists today is a mechanism smoke-test that
   cannot move the real forward (B.1). A correct end-to-end measurement needs either L6
   running correctly in serve (currently capture-foreign, `glm52-mmel-rebuild.md`) or the
   T4 bridge-event bind proven (B.2a). Both are real, unfinished work.

### C.4 Recommendation

**Run the GATE (A) first — it is cheap, off the build, and decisive.** Specifically:
1. `CONC=64 bash mmel/trace_c64.sh` → write `mmel/ar_overlap_budget.py` (off-card) → get
   `reclaimable_ms/step`.
2. The `HPU_AR_COMM_MARKSTEP=0` **c64** A/B (B never run at concurrency before — every
   prior AR-markstep test was bs=1).

- **If `reclaimable_ms/step` ≳ 20 ms AND the c64 markstep-off A/B moves the step:** wire
  P1 via the bridge event (B.2a) if reachable, else finish L6 + wire there (B.2/B.3).
  This is the one path where collective_seq could deliver a real concurrency win and
  beat the +15% bsgt1 ceiling.
- **If reclaimable is <~5 ms or all data-dep, or the c64 markstep-off A/B is flat
  (the prior measurements predict this):** **SKIP collective_seq and go to the owned
  forward.** The c64 balloon is the all-expert int4 dequant; the only lever that moves it
  is the whole-forward / megakernel concurrency rebuild — and *that* is already known to
  be **HBM-blocked on this box** for the 753B int4 model (every megakernel/fp8-resident
  path OOMs in the ~20 MB headroom, `glm52-mmel-rebuild.md` 2026-06-24-LATE). In that
  case the honest conclusion is: **collective_seq is the same class of structural lever
  that measured flat; ship the deployed +15% `glm52-conc-bsgt1` ceiling and put the next
  effort into the head (single-stream accept multiplier) or a smaller bf16 model
  (Qwen3.6-35B, no int4 unpack), which is the documented fast-serving path on this box**
  (`no-small-model-speed-retreat.md` notwithstanding — that rule is about *single-stream
  size-invariance*; bf16-no-unpack is specifically the concurrency-headroom argument).

**Bottom line:** the mechanism is built and silicon-proven; the API the user asked about
(`synEventMapTensor` + `synStreamWaitEvent`) is confirmed present and *does* give
device-side, no-host-round-trip event→stream gating; but the lever's value is entirely
contingent on the c64 trace showing **fence-bounded** (not data-dep) idle at the AR
barriers, and the strong prior (collective = 2.6% of the c64 step; the pole is the
dequant) says that gate will likely fail. **Do the trace; let it decide; do not build on
faith.**

---

## IMPLEMENTATION — GATE TOOL DONE + WIRE SKELETON (2026-06-25, off-card)

> Status: the GATE TOOL is **written and runs off-card** (proven on two existing
> traces). The WIRE is a **gated, default-off skeleton + a precise insertion point**;
> it is NOT wired, because it is contingent on (i) the gate passing and (ii) the owned-
> forward L6 being on-card. **Nothing here touched the card** (a dump is running).

### I.0 The trace schema is NOT what §A.1 assumed — read this before running the gate

Verified against `/mnt/data1/recipe_cache/*/glm_decode.json` (the real Synapse/Kineto
schema, off-card inspection): **timestamps/durations are microseconds**, and the
all-reduce is **not a single device kernel**. It surfaces as a HOST `hpu_op` event
`hcclAllReduce` on the HCCL Synapse thread, and the bridge brackets every AR with the
*exact* event-sync primitives this whole lever is about:

```
enqueueWithExternalEvents (acceN)   <- a layer compute recipe launch (producer)
eventRecord (accel0)                <- record an event on its external output tensor
streamWaitEvent (accel0)            <- a DIFFERENT stream waits on it (DEVICE-side)
hcclAllReduce            dur~17us   <- the collective (host launch/wait window)
eventRecord (accel0)                <- record AR completion as an event
streamWaitEvent (accel0)            <- the CONSUMER stream waits on AR (DEVICE-side)
enqueueWithExternalEvents (acceN)   <- the next layer's compute recipe launch (consumer)
```

**Load-bearing consequence (this strengthens §B.0 AND the §C risk):** the bridge
**already** issues `streamWaitEvent` (== `synStreamWaitEvent`) and
`enqueueWithExternalEvents` (== `synLaunchWithExternalEvents`) around the AR — i.e. the
P1 device-side event→stream wait the design wanted to "build" is **partly already in the
stock path**. So the only thing collective_seq can additionally reclaim is a HOST
`synchronizeEvent (accel0)` round-trip that lands *inside* the AR→consumer gap. Where the
bridge already overlaps with a device-side `streamWaitEvent` and no host fence, there is
**nothing left to reclaim**. This is exactly why the gate must classify *host-fence-
bounded* vs *device-already-overlapped*, not merely "is there a gap".

### I.1 The GATE TOOL — `mmel/ar_overlap_budget.py` (NEW, off-card, runs now)

Built and verified (`$HOME/.venvs/m3calib/bin/python -m py_compile` clean; ran on
`leakmap/glm_decode.json` and `batchtest/glm_decode.json`). Pure JSON post-processing,
**no card**. Per `hcclAllReduce` barrier on the host op stream it computes
`gap = (next enqueueWithExternalEvents start) − (hcclAllReduce end)` and classifies:

- **FENCE-BOUNDED (RECLAIMABLE)** — a HOST `synchronizeEvent (accel0)` falls inside the
  gap (a host round-trip the on-device gate deletes).
- **DATA-DEP (NOT reclaimable)** — no host fence in the gap; the consumer launched as
  soon as the AR result existed (device-side `streamWaitEvent` already overlapped it, or
  the launch immediately follows). collective_seq buys ~0 here.

Output: `reclaimable_ms/step`, `data_dep_ms/step`, the kind histogram, the 5 largest
gaps (auditable), and a GO/NO-GO verdict. It is **deliberately conservative**: a gap
counts as reclaimable ONLY if a real host `synchronizeEvent` is observed inside it.
It also **warns if AR/step ≪ 156** (a partial/wrong-N trace under-reports ms/step).

```bash
# OFF-CARD, runs on any existing glm_decode.json (no card):
$HOME/.venvs/m3calib/bin/python mmel/ar_overlap_budget.py <glm_decode.json> [N_steps]
```

Thresholds in the tool (`GO_MS_PER_STEP=20`, `NOGO_MS_PER_STEP=5`) match §A.1/§C.4.

**Smoke result on the two existing (partial, ~59 AR/step) traces** — for tool-validation
only, NOT the real gate: reclaimable `0.2–0.7 ms/step`, ~80–95% of gaps classed
`data-dep(devwait)` (the bridge's own device-side wait). i.e. on these traces the tool
already reads **NO-GO** and the consumer is overwhelmingly device-overlapped. The real
gate must run on the fresh c64 capture; if it reproduces this shape it confirms the
strong prior (collective ≠ the pole).

### I.2 EXACT run sequence (the GATE — run FIRST when the card frees)

```bash
# (1) capture the REAL c64 batched-decode trace (on-card; ~5 min; serves glm52-conc):
cd $HOME/AI && CONC=64 bash mmel/trace_c64.sh
#     -> /mnt/data1/recipe_cache/glm_decode_prof_c64/glm_decode.json  (HPU_PROFILE_N=40)

# (2) the AR-overlap budget (OFF-CARD; the trace half of the gate):
$HOME/.venvs/m3calib/bin/python mmel/ar_overlap_budget.py \
    /mnt/data1/recipe_cache/glm_decode_prof_c64/glm_decode.json 40
#     -> reclaimable_ms/step + GO / NO-GO / MARGINAL

# (3) the SECOND, decisive gate (§A.3) — the c64 HPU_AR_COMM_MARKSTEP=0 A/B
#     (every prior AR-markstep test was bs=1; this is the first at concurrency):
#       A (control): models/glm52-conc-bsgt1.env                       -> step_ms_A, greedy_A
#       B (drop AR mark_step): same env  + -e HPU_AR_COMM_MARKSTEP=0
#                                         + -e HPU_AR_FENCE_COUNT=1     -> step_ms_B, greedy_B
#     measure the c64 step-timer (HPU_STEP_TIMER=1) A vs B; greedy bytes MUST match.
```

**GO iff BOTH:** `reclaimable_ms/step ≳ 20` (step 2) **AND** the markstep-off A/B moves
the c64 step (step 3). Either fails → **NO-GO, ship the +15% `glm52-conc-bsgt1` ceiling**
and put the effort into the owned forward / head (§C.4).

### I.3 WIRE SKELETON (gated, default-off) — only IF the gate is GO

**This is a skeleton + a precise insertion point, NOT a wired change.** It depends on the
owned-forward L6 being on-card (the other agent's work). Do NOT wire before §I.2 is GO.

**Where it binds (resolving the §B.1 doc-vs-code disagreement):** the live communicator
arm (`hpu_communicator.py:151-158`) is a **mechanism smoke-test only** — its
`producer_signal`/`consumer_wait` run on libcsengine's own queues, disconnected from the
lazy-graph stream that runs the next layer's GEMM, so it cannot gate the real consumer
(§B.1). The real bind goes where **we own the consumer op's submission**: the MMEL L6
chained-exec forward, `vllm-gaudi-patch/v1/worker/mmel_l6_chain.py`
`run_chained_decoder_layer` (`:802`). In that path the TP=8 all-reduce is issued *inside*
the layer's own o_proj/MoE-combine submodules (no explicit `all_reduce` call in the L6
body — confirmed by grep), so the wire must sit at the **layer-to-layer handoff** the L6
forward owns: after the AR-carrying submodule returns and before the next layer's first
GEMM is submitted on our stream.

**The skeleton (new, gated by a NEW flag `MMEL_COLSEQ_L6=1` so it is distinguishable from
the smoke-test `MMEL_COLSEQ`):**

```
# in mmel_l6_chain.run_chained_decoder_layer, at the post-AR / pre-next-layer seam:
if os.environ.get("MMEL_COLSEQ_L6") == "1":
    cs = _colseq_for_this_worker()            # reuse hpu_communicator._colseq_ensure() ctx
    slot = _next_slot()                       # 16-slot ring (collective_seq.h:120)
    # --- PRODUCER: bind the AR's completion to OUR SOB ---
    # PREFERRED (P1, PLAN §B.2a) — no raw-CS, no L6-internal queue guess:
    #   bind via the bridge external-event surface that the trace shows is ALREADY live:
    #   event::map_event_to_tensor(recipe, ar_output_tensor)  -> synEventMapTensor
    #   event::stream_wait_event(next_layer_stream)           -> synStreamWaitEvent
    #   (event.h:88-104). The SOB we reserve (colseq_slot_sob -> id+cfg, collective_seq.h:120)
    #   is handed to the bridge as that event's backing SOB. <-- T4 card-only unknown.
    # FALLBACK (P2, PLAN §B.3) — if the bridge event cannot be bound to our SOB:
    #   cs.producer_signal(slot, COMPLETION_Q)  # 1-packet SIGNAL on the EDMA/PDMA queue
    #                                           # HCCL drains to  <-- T2 card-only unknown.
    # --- CONSUMER: the next layer's first GEMM waits on the SOB on-device ---
    cs.consumer_wait(slot, MME_Q)             # colseq_consumer_wait (collective_seq.h:88)
# else: stock path (host fence) — byte-identical, default-off.
```

**Non-negotiables (already correct in the smoke-test arm; keep them):**
- Default-off: unset `MMEL_COLSEQ_L6` ⇒ stock byte-identical. FAIL LOUD on init failure;
  never silently fall back to the host barrier (`_colseq_ensure` raises,
  `hpu_communicator.py:56-60`). Reuse that same `_colseq` ctx (one cseng_attach/worker).
- Byte-exact gate: collective_seq changes only sync segmentation, never values (HCCL
  output bytes untouched). Greedy logits A==B at c16 (`glm52-conc-bsgt1.env` correctness
  gate). A wrong bind must CRASH a bring-up gate, not corrupt output.
- ENGAGE banner on all 8 ranks or the test is invalid (`[ENGAGE]/[SKIP]` discipline).

**HARD DEPENDENCY (state it, do not hide it):** the L6 chain is currently capture-foreign
in serve (`glm52-mmel-rebuild.md`: `synLaunch status=1` inside `wrap_in_hpu_graph`; the
captured routed recipe freezes K-of-256 routing). **collective_seq cannot be measured
end-to-end until the owned-forward L6 runs correctly in serve.** The only wire that does
NOT need L6 is the bridge-event P1 bind (§B.2a) hooked from the OpBackend layer
(`hpu-moe-ext`) — but that still requires the T4 card test to confirm the external-event
path increments OUR SOB. **Both are real, unfinished, on-card work. Wire only after the
gate (§I.2) is GO and L6-on-card (or the T4 bridge bind) is proven.**

### I.4 LOUD HONEST VERDICT (the one lever that may be a dud)

**collective_seq is the single lever in this plan most likely to be a NULL.** The AR is a
TP=8 **data dependency** — the consumer op reads the all-reduced tensor; no sequencing
trick starts it before that data exists. The trace schema makes it worse, not better: the
bridge **already** does device-side `streamWaitEvent` overlap around every AR, so the only
reclaimable thing is a *host* `synchronizeEvent` bubble inside the gap — and the two
existing traces show ~80–95% of gaps are already device-overlapped (NO-GO shape). The
strong prior agrees: collective = 2.6% of the c64 step; the documented c64 pole is the
all-expert int4 dequant, not the AR fence. **The GATE (§I.2) decides BEFORE we spend the
wire effort.** Run the c64 trace → `ar_overlap_budget.py` → the markstep-off A/B; wire
**nothing** unless BOTH halves pass. If the gate fails (the likely outcome): ship the
deployed +15% `glm52-conc-bsgt1` ceiling and move the effort to the owned forward / head.

---

## D) File / line index (everything cited, for the implementer)

- Live wire (smoke-test only): `_src/vllm_gaudi/distributed/device_communicators/hpu_communicator.py`
  — `MMEL_COLSEQ` path `:151-158`; `_colseq_ensure` (LOUD, accel-fd pick) `:55-107`;
  AR LEAKMAP comment `:122-131`; `HPU_AR_COMM_MARKSTEP`/`HPU_AR_FENCE_COUNT` levers
  `:160-181`; `combine`/`all_gather` mark_steps `:196,218-219`.
- C engine: `mmel/collective_seq.h` (API), `mmel/collective_seq.c` (impl; consumer-wait
  needs producer seq `:149-152`, producer_signal records it `:198`, COLLECTIVE_WAIT
  negative-control probe `:241-301`). Built: `mmel/libcolseq.so`, `mmel/colseq_test`.
- Python surface: `mmel/colseq_py.py` (`consumer_wait :69`, `producer_signal :63`,
  `slot_sob :84`, `set_producer_seq :78`).
- Design + readiness: `mmel/COLLECTIVE_SEQ_DESIGN.md` (full), `mmel/W7_COLLECTIVE_SEQ_READY.md`
  (payoff table §4: SS ~0 / conc MED).
- Synapse external-event surface (B.0): `hpu-moe-ext/tpc_kernel/opbackend/bridge/`
  `pytorch_helpers/synapse_shim/partial_event_emulation.h:43-67` (synLaunchWithExternalEvents,
  synEventMapTensor, UsePartialEventEmulation), `backend/synapse_helpers/event.h:88-104`
  (stream_wait_event → synStreamWaitEvent, map_event_to_tensor → synEventMapTensor),
  `synapse_shim/synapse_api_shim.h:27,103,104` (symbols resolved).
- Gaudi-2 kernel facts: COLLECTIVE_WAIT stub `gaudi2.c:12139` + funcs-table `:12725`;
  gen_signal/wait `gaudi2.c:11967`/`:12060`; NIC-not-sync-stream `gaudi2.c:2782-2787`.
- Trace tooling (A): `mmel/trace_c64.sh` (c64 batched decode profiler → glm_decode.json),
  `bench/gather_budget.py` (bucket analyzer), `mmel/profile_b1_singlestream.sh` (bs=1 ref).
- **GATE TOOL (NEW, off-card): `mmel/ar_overlap_budget.py`** — per-AR fence-bounded
  (reclaimable) vs data-dep classifier → `reclaimable_ms/step` + GO/NO-GO. Runs on any
  `glm_decode.json`. Verified vs the real Synapse schema (host `hpu_op`: `hcclAllReduce`,
  `synchronizeEvent`, `streamWaitEvent`, `enqueueWithExternalEvents`; µs units).
- WIRE insertion point (skeleton, §I.3): `vllm-gaudi-patch/v1/worker/mmel_l6_chain.py`
  `run_chained_decoder_layer:802` (the L6 owned-forward, post-AR / pre-next-layer seam;
  gated by NEW flag `MMEL_COLSEQ_L6`, default-off). L6 capture-foreign dep:
  `glm52-mmel-rebuild.md`.
- Deployed concurrency baseline: `models/glm52-conc-bsgt1.env` (+15% ceiling, routed-bsgt1).
- AR-floor env (the bs=1 control, repurpose for the c64 A/B): `models/glm52-arfloor.env`.
- Prior verdicts: `glm52-40plus-autonomous-run.md` (AR=0.5%, 6 µs collective, comm-overlap
  dud, 633 reclaim fences), `glm52-mmel-rebuild.md` (L6 capture-foreign, HBM-blocked,
  collective 2.6% of c64), `fabric-not-a-perf-factor.md` (collective ≠ the pole).
