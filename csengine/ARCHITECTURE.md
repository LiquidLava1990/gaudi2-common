# csengine — Rung-2 execution engine: architecture, feasibility, plan

**Mission:** kill THE WAIT in GLM-5.2 bs=1 decode on 8×Gaudi-2 by submitting the
decode forward via the OPEN habanalabs command-submission (CS) ioctl with ALL
inter-engine sync done ON-DEVICE (sync objects + monitors), so the host issues
~one submit + ~one wait per token instead of the current ~800 host round-trips.

Everything below is grounded in the on-box driver source
(`/usr/src/habanalabs-1.24.0-1007`), the on-box UAPI
(`/usr/include/drm/habanalabs_accel.h`), the hl-thunk header
(`/opt/habanalabs/src/hl-thunk/include/uapi/hlthunk.h`, copied to
`/tmp/hl_check/hlthunk.h`), and the bridge C++ in
`$HOME/AI/dsa_e2e/build_stage/opbackend/bridge/`.

---

## 0. TL;DR verdict

**The per-op host fence IS deletable below the Synapse bridge — yes, with kernel-source
proof.** The open CS ioctl exposes on-device sync (`HL_CS_FLAGS_SIGNAL`/`_WAIT` →
`gen_signal_cb`/`gen_wait_cb` program SOB-increment + monitor-arm + QMAN-fence
packets that block an engine on-device until another engine signals — `hw_queue.c:415,474`,
`gaudi2.c:11967,12060`), and staged submission lets a whole multi-CS chain take ONE
host completion (`command_submission.c:cs_needs_completion:279`). The host's only
mandatory device interaction per step is the final completion wait — and even that
can be a userspace busy-poll on a mapped CQ counter, no ioctl
(`irq.c:354,385` + `HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ`). The PoC (`cs_poc.c`)
confirms this on silicon: it runs a 2-engine cross-synced chain with **1** host
`HL_WAIT_CS` for the whole chain.

**BUT** (the honest part, from mapping the real bridge): the 633 `synchronizeEvent`/step
are **not** 633 serial fences on the decode thread. They are `synEventSynchronize`
calls drained by a **per-stream GC thread** (`stream.cpp:72-108`), one per *recorded
event*, and the count is dominated by **TP=8 all-reduce signaling × MTP-depth-3 ×
recipe-segment-count**. So the Rung-2 win is not literally "633→1"; it is "collapse
the recorded-event count and the collective count by submitting ONE device-replayed
graph per token with on-device sync," which removes the per-recipe-island fences and
the per-collective host signaling. The deepest, durable lever is therefore **a single
static command buffer per token with on-device SOB sync**, which is exactly what the
CS layer enables and the bridge cannot express.

---

## 1. The open CS interface — can we submit our own CBs and sync on-device?

### 1.1 The device node and ioctls (VERIFIED)
- Device: `/dev/accel/accel0..7` (DRM accel chardev; `crw-rw-rw-`).
- ioctls (drm.h `DRM_IOCTL_BASE='d'` 0x64, `DRM_COMMAND_BASE=0x40`;
  `habanalabs_accel.h:2194-2337`):
  | ioctl | nr | full cmd | purpose |
  |---|---|---|---|
  | `HL_INFO` | 0x40 | 0xc???6440 | hw-ip / sync-manager query |
  | `HL_CB` | 0x41 | 0xc0186441 | create/destroy/info command buffer |
  | `HL_CS` | 0x42 | 0xc0306442 | **command submission** |
  | `HL_WAIT_CS` | 0x43 | 0xc0386443 | wait for CS completion |
  | `HL_MEMORY` | 0x44 | 0xc0286444 | alloc/map/unmap device+host mem |

  (All four cmd numbers reproduced by `abi_check.c` compiling against the real
  header; `HL_WAIT_CS=0xc0386443` independently agrees with the busypoll prior-art
  `busypoll_waitcs.c:23`.)

### 1.2 CS submission model (`struct hl_cs_in`, `hl_cs_chunk`)
A CS is an array of **chunks** (`chunks_execute`), each = `{cb_handle, queue_index,
cb_size, cs_chunk_flags, ...}` (`habanalabs_accel.h:1472`). Each chunk puts one CB on
one HW queue. Key flags (`habanalabs_accel.h:1550+`):
- `HL_CS_FLAGS_SIGNAL 0x2`, `HL_CS_FLAGS_WAIT 0x4` — **the on-device sync primitives.**
- `HL_CS_FLAGS_STAGED_SUBMISSION 0x40` / `_FIRST 0x80` / `_LAST 0x100` — submit a chain
  of CS as ONE logical unit; only the LAST gets a host completion.
- `HL_CS_FLAGS_ENCAP_SIGNALS 0x800` + `RESERVE_SIGNALS_ONLY 0x1000` — reserve a SOB
  range once, then signal/wait against offsets across many staged CS (pipelined
  dependency chains, one reservation, zero per-hop host waits).
- `HL_CS_CHUNK_FLAGS_USER_ALLOC_CB 0x1` (per-chunk) — the CB is user-allocated and
  MMU-mapped; `cb_handle` is then a **device address** the kernel uses directly
  (`command_submission.c:1589`, `hw_queue.c:410`). **Required for Gaudi-2 HW queues**
  (their `cb_alloc_flags = CB_ALLOC_USER`, `gaudi2.c:2789`; a driver CB needs
  `CB_ALLOC_KERNEL` → rejected with `-EINVAL` at `command_submission.c:1271`). This
  is the foundation of a *static* per-token CB.

### 1.3 On-device inter-engine sync — the mechanism (THE crux)
`HL_CS_FLAGS_SIGNAL` on queue A → `init_signal_cs` (`hw_queue.c:415`) →
`gaudi2_gen_signal_cb` (`gaudi2.c:11967`) writes ONE `PACKET_MSG_SHORT` that
**increments a sync-object (SOB)** on the device when the queue's prior work drains.

`HL_CS_FLAGS_WAIT` on queue B referencing A's signal seq → `init_wait_cs`
(`hw_queue.c:474`) → `gaudi2_gen_wait_cb` (`gaudi2.c:12060`) writes 4×
`PACKET_MSG_SHORT` (program monitor payload addr/data, then **ARM** the monitor on
A's SOB with mode GREATER_OR_EQUAL) + 1× `PACKET_FENCE` that makes B's QMAN **block
on-device** until the monitor fires. **No host is in this loop — the monitor is
hardware.** This is precisely what the lazy bridge replaces with a host
`synEventSynchronize`. (Cross-queue is legal: `init_wait_cs` copies the SOB from the
signal CS's fence; both queues only need `supports_sync_stream`, which all non-NIC
Gaudi-2 queues have — `gaudi2.c:2782-2787`.)

### 1.4 Completion model — is a host fence FORCED per CS?
No. Three independent escape hatches, all in the open driver:
1. **Staged submission**: `cs_needs_completion()` returns false for every non-last
   staged CS (`command_submission.c:279-288`); only `staged_last` allocates a fence +
   completion. → N synced CS, ONE host completion.
2. **Internal/HW queue completion is on-device**: `hl_hw_queue_update_ci` is skipped
   for `QUEUE_TYPE_HW` (`hw_queue.c:53`); the engine manages its own CI. The host
   just rings a doorbell = one `WREG32` (`gaudi2_ring_doorbell:7470`).
3. **The wait itself need not sleep**: `hl_wait_for_fence` does a non-sleeping
   `completion_done()` when `timeout_us==0` (`command_submission.c:2728-2729`); the
   IRQ-sleep (`wait_for_completion_interruptible_timeout`, line 2736) is the
   ~1.78 ms-wake cost the busypoll shim already spins away. And
   `HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ` lets the driver compare a **user-mapped CQ
   counter** `>= target` (`irq.c:354,385`) — i.e. the host can busy-poll a mapped
   counter in userspace with **zero ioctls** until the whole step is done.

### 1.5 hl-thunk / SCAL — the userspace wrappers (optional)
hl-thunk wraps all of the above (`hlthunk.h`): `hlthunk_command_submission`,
`hlthunk_staged_command_submission[_encaps_signals]`, `hlthunk_signal_submission`,
`hlthunk_wait_for_signal`, `hlthunk_reserve_encaps_signals`,
`hlthunk_request_mapped_command_buffer`, `hlthunk_host_memory_map`,
`hlthunk_get_sync_manager_info` (first free SOB/mon/cq), `hlthunk_get_hw_block`
(mmap the sync-manager register block to poke SOBs by hand). SCAL (`libscal.so`, no
public header) is the productized stream scheduler Synapse uses
(`scal_stream_submit` + completion-groups + SO/monitor pools) — it is *literally the
same CS+SOB machinery* one layer up. **We can build directly on hl-thunk (linked) or
on the raw ioctl (cs_poc does the latter, zero deps).** Either bypasses the bridge's
per-recipe host fences.

---

## 2. Mechanism map — the ~800 round-trips → source → what csengine deletes

Counts are the LEAKMAP on the **sync** path (`leakmap_mtp.out`, 633/137/70). On the
**live** serve (`HPU_ASYNC_SPEC=1`) some are already eliminated/deferred — noted.

| # / step | category | ROOT SOURCE (file:line) | class | csengine action |
|---|---|---|---|---|
| ~633 | `synchronizeEvent` | `synEventSynchronize` drained by per-stream **GC thread** `gc_thread_proc` (`backend/synapse_helpers/stream.cpp:72-108`), one per *recorded event* (`event.cpp:45-58`). Events come from: ~2/recipe (`hpu_habana_cache.cpp:1753-1801`) × ~156 TP all-reduces/token × MTP-depth-3 × recipe-segment islands (DSA indexer, sparse-attn, fused-MoE, MTP head are separate recipes). | (a) lazy-bridge recipe-boundary + collective signaling | **DELETE the most.** One static per-token CB (or few staged CS) with on-device SOB sync = one recipe island, no per-segment/per-collective recorded events. Collectives become on-device-synced DMA/NIC ops inside the same CB, signaled by SOB not host event. |
| ~137 | `copy_hpu_lazy_D2H` | sync path: rejection_sampler `.cpu()`×4 (`hpu_rejection_sampler.py:44-47`); sampled-token grid + valid_count (`hpu_model_runner.py:317,4891`). **LIVE path (async): the ×4 are GONE (device-side `:35-41`), grid/valid_count deferred off-pipe (`:329-343,4250`).** SURVIVING on critical path: drafter `target_positions.cpu()` + `last_token_indices.cpu()` (`vllm_gaudi/v1/spec_decode/hpu_eagle.py:134-135`), draft pull `dt.tolist()` (`hpu_model_runner.py:7068`). | (b) sampling/MTP python D2H | **DELETE via on-device control:** positions advance deterministically (compute on-device); draft IDs read once via a single mapped-counter poll, or kept on-device for an on-device verify. |
| ~70 | `aten::item` / `_local_scalar_dense` | survivors on live path: `all_accepted_mask.sum()>0` device-bool→host-if (`hpu_rejection_sampler.py:91`); `cu_num_tokens[i].item()` (numpy, host-only, `hpu_model_runner.py:3558`). DP-only `torch.max(...).item()` (`:944`) is OFF the single-stream path. multi-cand `.item()`s gated behind `HPU_MULTI_CAND>1`. | (c) scalar reads for python branching | **DELETE via on-device sampler:** argmax/accept decided on-device; host reads ONE token count per step (mapped counter), not per-candidate scalars. |

**Net:** the dominant lever is #633 (≈50% of device time per the bridge analysis) —
and it is killed not by "1 wait" but by **collapsing recorded-event count +
collective count via a single device-replayed graph with on-device sync.** The CS
layer is the only place that can express that; the bridge re-inserts an event per
recipe-island/collective no matter what you do above it (which is why nosync /
run-ahead / async-spec / comm-overlap / full-HPU-graph-capture were all flat — they
reduce *some* events but cannot remove the per-island/per-collective event model).

---

## 3. The engine design

### 3.1 Goal shape (per token)
```
host:  [ring 1 doorbell] ............................ [poll 1 mapped CQ counter] -> next token
device: DMA(load KV/act) ->SOB1   MME(QKV/attn) ->SOB2   TPC(act/route) ->SOB3
        MME(MoE up) ->SOB4   TPC(silu) ->SOB5   MME(MoE down) ->SOB6
        NIC(all-reduce, on-device-synced) ->SOB7   ... ->SOB_last
        on-device argmax/sampler ->token buffer   ->CQ counter++
   (every -> is a monitor-arm + fence: engine N+1 blocks on-device on SOB_N)
```
Host round-trips/token: **1 doorbell + 1 completion poll** (vs ~800).

### 3.2 Components
**(a) Kernel compilation — reuse, don't rebuild.** We do NOT re-implement MME/TPC
kernels. We capture the GC-compiled recipes the bridge ALREADY produces (the fused
int4 MoE OpBackend kernel, DSA indexer/sparse-attn, attention, MTP head — all built
and validated, see memory `fused-moe-kernel-validated`). Two reuse routes, in order
of preference:
  1. **Recipe → CB extraction**: a recipe is a serialized blob of QMAN packets +
     tensor bindings (`habana_serialization/recipe.h` in the bridge). Deserialize
     once, relocate tensor addresses to our static buffers, concatenate the per-layer
     packet streams into ONE static CB per token with SOB sync packets spliced
     between engine handoffs. This is the deepest, highest-reuse path.
  2. **SCAL stream replay**: drive `scal_stream_submit` with the same recipes but our
     own completion-group wiring (one completion for the whole step). Less deep
     (still SCAL/ARC-scheduled) but far less packet-surgery; good intermediate.

**(b) Static command buffer.** One USER_ALLOC_CB per token shape (prefill / decode /
MTP-draft-k), built once at warmup, holding the full packet stream + on-device SOB
sync. Tensor addresses (KV cache slot, activation scratch, token-out) are the ONLY
per-token mutables — patched in place (or addressed via a fixed scratch + a tiny
index DMA) before the doorbell. Static CB = zero per-token recipe lookup, zero
per-op submit.

**(c) ONE CS submit per token.** Either a single CS with the static CB on the head
queue (intra-CB SOB sync handles the rest), or a staged submission across the engine
queues with `STAGED_SUBMISSION_FIRST..LAST` so the kernel gives ONE completion.

**(d) On-device sampling + rejection.** argmax/sampling as TPC packets writing the
chosen token id to a fixed device buffer; the MTP accept/reject as an on-device
compare (the device-side rejection sampler from `hpu_rejection_sampler.py:35-41`
already exists — emit it as packets). The host reads ONE accepted-count + the token
ids via a single mapped-counter poll. This kills the D2H/item categories.

### 3.3 Batch-first / tree-native (so it also serves concurrency + the cand tree)
Make the static CB parametric in batch dim B and tree width W: the MoE/attn packets
are already batched; the per-token loop becomes a per-(B×W) gather. One CB serves
single-stream (B=1,W=1), the multi-candidate MTP tree (W=tree), and concurrency
(B=batch) — the submit count stays 1/step regardless of B,W. This is the same engine
the throughput path wants, so it is not single-stream-only work.

### 3.4 Multi-card (TP=8) — the collective inside the CB
The dominant 633-event term is TP all-reduce signaling. Two levers:
  1. **On-device-synced NIC ops**: the all-reduce DMA/NIC descriptors go INTO the
     static CB, gated by SOBs (NIC queues don't support sync-stream, but the
     collective completion is signaled to an EDMA/PDMA SOB that the next compute
     waits on). The host stops signaling each collective; the device chains it.
     (hl-thunk exposes the NIC QP/CQ surface; `HL_CS_FLAGS_COLLECTIVE_WAIT` +
     `collective_engine_id` is the kernel hook, `command_submission.c:874`.)
  2. **Fewer/fused collectives** (orthogonal, complementary): `fuse_collective_view_pass`
     already in the bridge — fewer all-reduces = fewer events even before CS.

---

## 4. Feasibility verdict (honest: yes / partial / no, with evidence)

| claim | verdict | evidence |
|---|---|---|
| We can submit our own CBs bypassing the bridge | **YES** | open `HL_CS` ioctl + USER_ALLOC_CB; `cs_poc` does it with zero Habana libs. |
| Inter-engine sync can be on-device (no host fence per op) | **YES** | `gen_signal_cb`/`gen_wait_cb` SOB+monitor+fence packets (`gaudi2.c:11967,12060`); `cs_poc` MODE csengine = 1 host wait for an N-op cross-engine chain. |
| A whole decode step can be ONE host completion | **YES** | staged submission `cs_needs_completion` (`command_submission.c:279`); + busy-poll mapped CQ counter (`irq.c:354`). |
| We can reuse the existing GC-compiled kernels | **YES (path 1 or 2)** | recipes are serialized QMAN packet blobs (`habana_serialization/recipe.h`); SCAL replays them (`scal_stream_submit`). Validated kernels exist (memory: fused-moe-validated, dsa-validated). |
| This deletes the ~633 synchronizeEvent | **PARTIAL→LARGE** | it deletes the per-recipe-island + per-collective recorded events (the dominant terms) by making the step ONE island with on-device sync. It does NOT, by itself, reduce collective COUNT — pair with collective fusion. Most 633 are GC-thread drained (off the decode thread) so wall-clock win < naive 633×; the real win is removing pipeline serialization + the ~1.78 ms IRQ wakes on the critical-path fences. |
| This reaches the 40–60 tok/s physics floor | **PLAUSIBLE, UNPROVEN** | device compute is ~16.5 ms/token MoE-read-bound (29% HBM BW). If the engine removes the host/sync overhead, the step approaches device-bound. But (a) the MoE read itself is the floor — csengine doesn't speed the read, it removes the WAIT around it; (b) on-device sampler/collective porting is real work. Target: close the gap between ~68.8 ms (current) and ~16.5 ms (compute), i.e. up to ~3–4× — to be PROVEN by the staged milestones, not assumed. |

**Bottom line: the WAIT is deletable at the CS layer — YES, mechanism-proven and
PoC-confirmed for the sync primitive. The full single-stream win is PLAUSIBLE and
gated on porting the kernels + sampler + collective into the static-CB/on-device-sync
form, which is staged below.**

---

## 5. Adversarial self-review (trying to prove this WON'T work)

1. **"The driver forces a host fence per CS completion."** — FALSE. Staged submission
   makes only the last CS need completion (`cs_needs_completion:279`); HW-queue CI is
   on-device (`hw_queue.c:53`); and `timeout_us=0` is a non-sleeping poll
   (`command_submission.c:2728`). The PoC measures 1 wait for an N-op chain. *Residual
   risk:* if the driver's CS-reclaim path needs the host to ack each CS to free
   resources, a long static CB could exhaust `max_pending_cs` (`GAUDI2_MAX_PENDING_CS`).
   Mitigation: ONE CB per step (not N CS) → 1 pending CS/step; or encaps-signals reuse
   one SOB range. The PoC's `--nchain`/`--reps` stress this.

2. **"Recipes can't be extracted/relocated — they bake absolute addresses."** —
   PARTIAL risk. Recipes have a tensor-binding/patch table (`recipe.h`,
   `const_section.h`) precisely so addresses can be set at launch. *Mitigation:* the
   SCAL replay path (route 2) avoids hand-relocation entirely — it submits the recipe
   through SCAL with our completion wiring. Start there; graduate to packet-splice
   (route 1) only if SCAL still over-synchronizes. *If both fail*, fall back to
   driving the bridge's own `LaunchSyncTensorsGraph` but with a custom single
   completion-group — strictly worse but still removes per-island fences.

3. **"On-device sampling/argmax/rejection is too complex to emit as packets."** —
   Real work, but the device-side rejection sampler ALREADY exists
   (`hpu_rejection_sampler.py:35-41`, used on the live async path) — it is a compiled
   recipe we reuse, not new TPC code. argmax is a single TPC op. *Residual:* dynamic
   tree shapes need a small on-device index DMA; handled by the batch-first design.

4. **"TP=8 collectives must round-trip the host."** — They do today (the 633's
   dominant term). On-device: NIC descriptors in the CB + SOB-gated completion, or
   `HL_CS_FLAGS_COLLECTIVE_WAIT` (`command_submission.c:874`). *Residual risk:* the
   open NIC/collective path is the least-documented surface; HCCL may insist on host
   coordination. *Mitigation:* milestone M3 isolates this; if on-device collective is
   blocked, the engine still wins on the compute path (single-card / lower-TP / fused
   collectives) — TP=8 collective-on-device is the stretch goal, not the gate.

5. **"`is_compute_ctx_active` = one ctx/device — we can't run alongside the serve."** —
   TRUE and by design. The engine REPLACES the serve's execution layer in the same
   process (it IS the serve's launcher), not a second ctx. The PoC runs standalone on
   a free card; the production engine lives inside the vLLM worker process holding the
   one ctx. *No conflict* — same ctx, we just submit differently.

6. **"The win is GC-thread-drained, so wall-clock won't move."** — The sharpest
   objection. Many 633 events are drained off the decode thread. *Rebuttal:* (a) the
   ~1.78 ms IRQ wakes on the *critical-path* fences (the ones the decode thread
   blocks on between recipe islands) ARE on the wall clock — busypoll already shows
   spinning them helps; (b) per-island serialization (each recipe waits for the prior
   island's events before launch) is exactly what one fused CB removes; (c) the
   measured 68.8 ms step vs 16.5 ms compute = ~52 ms of non-compute the engine
   targets. *Honest caveat:* the EXACT recoverable fraction is unproven — milestones
   M1/M2 measure it before committing to M3+.

---

## 6. Staged build plan (milestones + gates)

**M0 — PoC (DONE, this dir).** Prove on-device SOB sync gives 1 host wait for an
N-engine chain on real silicon. *Gate:* `cs_poc` MODE csengine WAIT_CS==1/step, chain
correct. (Run on a free card.)

**M1 — recipe replay with one completion (route 2).** Take ONE GLM-5.2 decode recipe
island (e.g. the fused MoE layer) the bridge already compiles; replay it via SCAL /
or submit its CB via `HL_CS` with our own single completion-group instead of the
bridge's per-output events. *Gate:* byte-identical output to the bridge path (the
triple-check bar) + measured drop in recorded events for that island.

**M2 — fuse N layers into ONE CB with on-device SOB sync.** Concatenate the decode
layers' packet streams into one static USER_ALLOC_CB, splice SOB-increment + monitor
fence between engine handoffs, patch the 3–4 per-token tensor addresses. ONE CS
submit + ONE completion (busy-poll mapped CQ counter) per step. *Gate:* correctness
(byte-exact logits vs bridge) + tok/s on a free card vs the current serve number;
this is where the single-stream win is first MEASURED (target: close a meaningful
fraction of 68.8→16.5 ms).

**M3 — on-device sampler + MTP accept + (stretch) on-device collective.** Splice the
existing device-side argmax/rejection recipes into the CB; read ONE accepted-count +
token ids per step. Then attempt on-device-synced TP all-reduce (or pair with
collective fusion). *Gate:* full decode step = 1 doorbell + 1 poll; D2H/item per step
→ ~1; tok/s at the device-bound floor.

**M4 — batch-first / tree-native + production integration.** Parametrize the CB in
(B, tree-W); wire it as the vLLM worker's execution path (same ctx). *Gate:*
single-stream AND concurrency AND multi-cand all served by the one engine, submit
count 1/step at every B,W; hardened, LOUD failures, DSA_READY-style validation gate.

Each milestone is independently measurable and abortable; M0–M2 prove the thesis
before the heavy M3–M4 lift.

---

## 7. Files in this dir
- `cs_poc.c` — the M0 proof (raw DRM ioctl, on-device SOB sync, ioctl tally).
- `hl_uapi.h` — local UAPI mirror (verified by `abi_check`).
- `abi_check.c` — independent verifier vs the real kernel header (static asserts).
- `build.sh` — runs abi_check then builds cs_poc (no Habana libs).
- `RUN_ONCARD.md` — exact on-card command + pass/fail gates.
- `README.md` — orientation.
