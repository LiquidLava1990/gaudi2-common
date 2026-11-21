# OWNED-FORWARD-ACROSS-ALL-LAYERS (Build #4b) — concrete build plan

**Status:** OFF-CARD PREP ONLY. No serve, no docker, no card touched. Every claim cited to file:line
on this host (a dump is running on the card). This is a plan to implement from, not an execution log.

**Date:** 2026-06-25.

**Goal (restated):** extend the proven Phase-1 device-resident single-layer static-chain to ALL
60 (M3) / 78 (GLM) decoder layers so the decode forward stays device-resident, killing the
~633 recipe-reclaim host fences (≈50% of the decode step) that come from the device draining
~1.6 recipes × N layers, each fenced, while the AIP sits ~15-17% idle.

---

## ⚠️ THE ONE FACT THAT REWRITES THIS BUILD (read first)

Phase-1's single-layer island (`MMEL_BANZ_LAYER=3`) is correct AND fast **only because it cheats two
ways that DO NOT survive being chained across all layers**, and the memory + design docs already
proved both:

1. **It uploads a DUPLICATE, FROZEN copy of the expert weights per layer.** `cap_routed` is a captured
   recipe whose 56 static weight tensors (the int4 expert w13/s13/z13/w2/s2/z2 set) are uploaded to
   *new* resident HBM buffers at warmup (`mmel_moe_island.py:485-494`). The manifest totals **149.6 MB
   of weight copies for ONE layer** (`mmel/cap_routed/manifest.txt`, 58 tensors, largest = the 12.58 MB
   w13 pack). Doing this on every MoE layer = 78 × ~150 MB ≈ **11.4 GB of *duplicate* HBM** on top of
   the model's already-resident ~46.7 GB int4 experts/card — that is the OOM that forced the 1-layer
   gate (`vllm-banz-phase1-progress.md` wall #4; the card has ~20 MB headroom left at the 753B int4
   config, `glm52-mmel-rebuild` 2026-06-24).

2. **It FREEZES the per-token MoE routing.** `cap_routed` was captured with `K=8` experts bound ONCE,
   with **IDENTICAL** weights (`capture_routed_island.py:662` `[100]*K`, order-independent so the P2
   byte-exact gate is clean). Its "byte-exact" is **only ever vs a same-frozen serial control**, never
   vs the live dynamic K-of-256 router (`WHOLEFORWARD_REBUILD_DESIGN.md` §2.4;
   `glm52-mmel-rebuild` 2026-06-22-PM). Replaying that one frozen recipe on every layer would route
   every token to the same frozen 8 experts on every layer = **garbage output**.

**Therefore #4b is NOT "replay the cap_routed island on 78 layers."** That path is OOM *and* incorrect
and must be explicitly killed (§D risk). The correct #4b is: **keep the model's existing resident int4
expert weights (zero re-upload) and drive the forward device-resident via the int4 *LazyOp* fused-MoE
op (`torch.ops.hpu_moe_ext.fused_moe` / `fused_expert`), which routes LIVE and is captured by the
bridge's own hpu-graph** — i.e. the static-tier chain becomes a chain of **live-weight-bound,
live-routing ops**, not frozen-recipe replays. This is the "Build A interior" the design doc converges
on (`WHOLEFORWARD_REBUILD_DESIGN.md` §5.3), now extended to span the whole forward.

---

## A) HOW THE 1-LAYER ISLAND WORKS TODAY (file:line)

The Phase-1 win is the FIRST time the in-serve island replay decoded coherent device-resident. The
mechanism is three cooperating files, all in `vllm-gaudi-patch/v1/worker/`, all bind-mounted into the
serve container (`models/glm52-l6-static.env`), all card-free unit-testable (torch imported lazily).

### A.1 The capture-escape (why the raw synLaunch runs at all)
- `mmel_l6_exec.py:232 install_banz_use_graphs(model_runner)` monkeypatches `model_runner._use_graphs`
  so that for a **decode** forward (`attn_metadata.is_prompt is False`) in scope, it returns `False`
  (`mmel_l6_exec.py:251-257 _banz_use_graphs`). That makes the runner thread
  `bypass_hpu_graphs=True` (`hpu_model_runner.py:3535`), and `graphs.py:681-683` then runs `orig_fwd`
  **outside** `wrap_in_hpu_graph`'s `capture_begin/end`. This is wall #1 from the Phase-1 run:
  without it the island's raw ctypes `synLaunch` is foreign to the capture and dies `status=1`.
  Gated `MMEL_WHOLEFWD=1`, installed from `mmel_l6_apply_patches` (`mmel_l6_exec.py:547-548`), default-off.
- Scope knob: `MMEL_WHOLEFWD_DECODE_BUCKETS` (`mmel_l6_exec.py:212`) confines the eager penalty to the
  buckets the chain covers; unset = all decode buckets. Prefill is NEVER bypassed.

### A.2 The per-layer driver (the `.forward` swap)
- `mmel_l6_apply_patches` (`mmel_l6_exec.py:490`) → `install_modern_moe_exec_layer` (`:441`) rebinds
  every `DeepseekV2DecoderLayer.forward` (matched by class name, `:394 _iter_decoder_layers`) to
  `_mmel_decoder_forward` (`:361`), which calls `run_chained_decoder_layer` (the real body lives in
  `mmel_l6_chain.py:802`). Signature-exact to stock `(positions, hidden_states, residual,
  llama_4_scaling) -> (hidden_states, residual)`, asserted at install (`:419 _assert_stock_signature`).
- `run_chained_decoder_layer` (`mmel_l6_chain.py:802`) runs the layer **stock-faithful, op-for-op, with
  NO per-op `mark_step`**: input_norm → self_attn (MLA+DSA, paged read, per-step KV write) → fp16
  scale (no-op on bf16) → fused-add post_attention_layernorm → MoE block (`mmel_l6_chain.py:1045-1082`).
  Attention + both norms stay **stock** (context-correct). Only the MoE tail is driven device-side.

### A.3 The MoE-tail device replay (the actual island)
- `MMEL_CHAIN_STATIC=1` routes the MoE block through `_run_moe_block` (`mmel_l6_chain.py:1140`), which
  on the covered M=1 decode bucket (`_subchain_b_covers_bucket`, `:1194`) calls
  `_run_routed_moe_via_island_replay` (`:1226` → `mmel_moe_island.run_routed_moe_via_island_replay`).
- **`MMEL_BANZ_LAYER` gate** (`mmel_l6_chain.py:1202-1205`): reads the env int; if it does NOT equal
  this layer's `layer_idx`, the layer **delegates to the stock dynamic MoE** (`return layer.mlp(...)`).
  So **exactly one** layer drives the device island; every other MoE layer is stock. This is the
  literal 1-layer gate that wall #4 (OOM) forced.
- The replay object `MoeBlockIslandReplay` (`mmel_moe_island.py:428`):
  - **`warmup_once()` (`:453`)** — deserialize the recipe via `RawCsIslandReplay`
    (`mmel_l6_chain.py:1474`); **upload the 56 static int4 weight `.bin`s to fresh resident HBM
    buffers** (`mmel_moe_island.py:485-494`, `.to("hpu")`, kept alive in `_static_keepalive`); alloc
    resident zero buffers for every output-class tensor (`:505-508`) + a **dedicated resident
    live-input buffer** `_live_in_buf` (`:530-533`, wall #3 fix); query + alloc the device workspace
    (`:540-545`); `mark_step()+synchronize()` to **commit** the lazy `.to("hpu")` copies so the first
    launch binds backed VAs (`:551-557`, wall #1-VA fix); read the workspace VA **post-commit**
    (`:570-571`, the `status=1` workspace-VA fix); then a **probe launch** (`_resolve_output_from_probe`,
    `:593`) replays once with the captured input and picks the output-class tensor byte-exact to
    `out_ref.bin` (BUG-2 output resolution).
  - **`run(hidden_states)` (`:737`)** — per step: `assert_live_hidden` (static-shape guard, `:748`);
    `mark_step()` to flush the producer graph; **`copy_` the live hidden into the dedicated
    `_live_in_buf`** (`:771-772`, NOT a transient `.clone()` — that collided VAs → `status=30`
    synFailedSectionValidation, wall #3); `mark_step()+synchronize()` to materialize the copy
    (`:774-781`); read `.data_ptr()`; build the launch array via the shared `_bind_map` (`:714`,
    static weights + live input + outputs + scratch, must cover EXACTLY the recipe's tensor set);
    `synLaunch` on the bridge stream (`:794`) + **ONE `synStreamSynchronize`** (`:795` → `:816`);
    return the resolved output viewed `bf16 [M,H]`.

### A.4 The stream + replay primitive
- `reuse_bridge_device_and_stream()` (`mmel_l6_chain.py:1432`) — wall #2 fix: it creates + caches a
  **real `synStreamHandle` via `synStreamCreateGeneric`** (`:1463`), because torch's
  `current_stream().hpu_stream` is a stream INDEX (0,4) that `synLaunch` rejects under the eager bypass.
- `RawCsIslandReplay` (`mmel_l6_chain.py:1474`): `deserialize()` (`synRecipeDeSerialize` +
  `synWorkspaceGetSize`, `:1495`); `build_launch_array(name->tensor)` (binds `.data_ptr()`, asserts
  contiguity, `:1523`); `launch(stream, arr, ws_ptr)` (`synLaunch`, `:1548`, with a full bound-arg DIAG
  dump on `status!=0`).

### A.5 The proven result (Phase-1, on-card 2026-06-25)
`glm52-l6-static` + `MODERN_MOE_EXEC=1 MODERN_MOE_EXEC_FAMILIES=glm_moe_dsa MMEL_CHAIN_STATIC=1
MMEL_WHOLEFWD=1 MMEL_BANZ_LAYER=3` → prompt "The capital of France is" decoded
`' Paris. Distance from Paris to Lyon is 391 km'`, 0 errors, island engaged device-resident on layer 3
via the capture-bypassed forward (`vllm-banz-phase1-progress.md`). **bs=1 ceiling ~17.5 tok/s.**

---

## B) THE ALL-LAYER EXTENSION PLAN (#4b)

The structure to span all layers is **already in place** — every layer's `.forward` is swapped, every
layer's MoE already routes through `_run_moe_block`, the `MMEL_BANZ_LAYER` gate is the ONLY thing
pinning it to one layer (`mmel_l6_chain.py:1202-1205`). #4b removes that gate, but to do so correctly
it must replace the **frozen-recipe replay** with a **live-weight-bound, live-routing** device path.
Three sub-pieces: weight binding, memory, collectives.

### B.1 The chain shape (what spans the layers)
Per layer, keep the Phase-1 tier split (it is byte-exact-vs-stock and context-correct):
- **Tier-DYNAMIC (stays stock, NEVER a frozen recipe):** `input_norm` (fused-add, live residual),
  `self_attn` = MLA + DSA paged-KV read + per-step KV write, fused-add `post_attention_layernorm`.
  These are context-dynamic; a frozen captured recipe is wrong the moment KV grows
  (`WHOLEFORWARD_REBUILD_DESIGN.md` §2.1). They run as the layer's own stock submodule calls inside
  `run_chained_decoder_layer`, with NO per-op `mark_step`.
- **Tier-STATIC (device-resident, our path):** the routed-MoE block — a pure function of
  (live hidden, resident expert weights), shape-stable per decode bucket. This is the dominant per-layer
  dispatch share and the only reachable static tail in-serve (o_proj already ran inside stock attention;
  `WHOLEFORWARD_REBUILD_DESIGN.md` §2.3).

The "device-resident across all layers" property = run **all 78 layers' MoE blocks through the
live-routing int4 op with NO per-layer host fence between them**, letting the bridge's own
`wrap_in_hpu_graph` capture the run (or, in the eager `MMEL_WHOLEFWD` regime, letting our single
per-tier `synStreamSynchronize` be the only fence).

### B.2 WEIGHT-BINDING STRATEGY (the make-or-break — kills both Phase-1 cheats)

**Do NOT re-upload weights. Bind the model's already-resident int4 expert tensors in place.**

The live MoE submodule (`layer.mlp`) already holds the per-layer int4 experts resident on the card
(packed weights + scales + zeros), in the GLM layout the op consumes:
`torch.ops.hpu_moe_ext.fused_expert(x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
router_weight)` (`test_fused_expert_op.py:7-8`; the routed variant `fused_moe(x, w13p, s13, z13, w2p,
s2, z2, rws)`, `mmel_l6_chain.py:780,1271`). The packed layout is
`w13_packed[H,2I//8], w13_scale[H//GS,2I], w13_zp[H//GS,2I//8], w2_packed[I,H//8], w2_scale[I//GS,H],
w2_zp[I//GS,H//8]` (`test_fused_expert_op.py:60-62`).

**The binding plan, per layer:**
1. At warmup, for each swapped layer, resolve the live MoE submodule's int4 expert weight attributes
   (the `_moe_routed_via_our_op` stub at `mmel_l6_chain.py:1281` is exactly the isolated, LOUD place
   this image-specific plumbing belongs — it currently raises until the bind is on-card-validated, by
   design). Read their `.data_ptr()` **post-commit** (resident, never re-uploaded → **zero extra HBM**).
2. Build the per-step routing scalars `rws` ON-DEVICE (the top-k fold, NO `.item()` —
   `route_fold_is_static`, `mmel_l6_chain.py:787`), so the op routes **live** per token (kills the
   frozen-routing cheat — the recipe is the model's real K-of-256 MoE, not a frozen 8).
3. Drive the int4 op as a **LazyOp** (it emits IR the GC compiles + the bridge captures; the proven
   in-serve int4 fused-MoE already runs this way — `fused-moe-kernel-validated`,
   `glm52-mmel-rebuild` 2026-06-22-PM). A LazyOp needs **no raw synLaunch and no eager bypass** for the
   MoE tier — it is captured inside the bridge's own graph, so the per-layer recipe boundary collapses
   without the capture-foreignness wall.

**Why this is the correct generalization, not the island:** the island (`MMEL_CHAIN_MOE_OP=2`) replays
a *frozen serialized recipe with its own duplicated weights*; the LazyOp binds the *live resident
weights* and routes *live*. The island was the right Phase-1 mechanism proof (it proved the
capture-escape + the device-resident replay end-to-end); #4b graduates it to the live-weight LazyOp so
it is per-layer-CORRECT and HBM-FREE. Keep the island path as the standing eager-bypass plumbing for
the one regime it pays (eager bs>1 MTP-verify), but it is NOT the all-layer driver.

**Concrete edit set (the smallest correct #4b):**
- `mmel_l6_chain.py:1202-1205` — generalize the `MMEL_BANZ_LAYER` gate from a single int to
  `MMEL_BANZ_LAYERS` = `"all"` | a comma/range list, so the device path engages on every MoE layer (or
  a staged subset for the bring-up A/B). Keep single-int back-compat.
- `mmel_l6_chain.py:1281 _moe_routed_via_our_op` — land the live int4 expert-weight bind (the attribute
  plumbing for the GLM/M3 serve image), driving the LazyOp `fused_moe` with on-device `rws`. This is
  the one image-specific surface; it fails LOUD on a model-tree drift by design.
- The per-layer A/B harness (`MmelGreedyABHarness`, `mmel_l6_exec.py:702`) gates each layer byte-exact
  vs `layer.mlp` before the layer is trusted (coherent fallback keeps the serve safe).

### B.3 MEMORY STRATEGY (why 1-layer OOM'd; why all-layers FITS)

- **Why 1-layer OOM'd:** the island uploads ~150 MB of **duplicate** expert weights per engaged layer
  (`mmel_moe_island.py:485-494`; `cap_routed/manifest.txt` = 149.6 MB). At the 753B int4 config the
  card already holds ~46.7 GB experts/card with ~**20 MB headroom** (`glm52-mmel-rebuild` 2026-06-24);
  even one extra 150 MB island OOMs without care, and N of them is hopeless.
- **Why all-layers FITS:** the live-weight bind (§B.2) **uploads ZERO new weight bytes** — it binds the
  `.data_ptr()` of weights already resident in `layer.mlp`. The all-layer working set is then just the
  model's existing resident experts (~46.7 GB int4) + KV + per-step activation buffers — **the same HBM
  the stock serve already fits**. The all-layer device-resident forward is, by construction, *not bigger
  than the stock forward's footprint*; it only changes WHO drives the ops and removes the per-layer
  fences. (This is the key inversion: the OOM was an artifact of the island's weight DUPLICATION, not of
  spanning layers.)
- **Budget headroom note:** the monolithic all-expert megakernel OOMs (`glm52-mmel-rebuild` 2026-06-24)
  because it materializes all 32 local experts' bf16 at once; the **routed** live-weight op does not —
  it consumes the resident int4 in place and only the K selected experts' dequant transients exist,
  the same as stock. So #4b's routed live-weight path stays inside the ~20 MB headroom where the
  monolithic path blew it. fp8-resident is a SEPARATE, OOM-prone lane (90.6 GB/card) — not #4b.

### B.4 WHERE THE COLLECTIVES FIT (the TP=8 all-reduce per layer)
- Each layer has a TP=8 all-reduce after the MoE (and after attention's o_proj, which is inside stock
  attention). Today these are HCCL collectives the bridge issues with a host barrier. In the chained
  forward they sit **between the static MoE tier of layer k and the dynamic attention of layer k+1**.
- **#4b correctness:** keep the stock collectives where they are (the AR is part of `layer.mlp` /
  o_proj's RowParallel reduce). The chain does not move them; it only removes the *recipe-reclaim*
  fences around the per-layer recipes. `HPU_AR_COMM_MARKSTEP=0` (drop the AR host barrier) is
  correctness-safe and free here (`vllm-banz-phase1-progress.md`: coherent, 0 errors), but is "1 of ~7
  per-layer breakers" so it is flat alone — leave it on, do not rely on it.
- **#4b does NOT need on-device-overlapped collectives.** `gaudi2_collective_wait_create_jobs` returns
  `-EINVAL` on Gaudi-2 (dead silicon, `glm52-mmel-rebuild` 2026-06-22-LATER) — true compute-overlapped
  collectives are unbuildable on this box. The W7 `collective_seq` (synEventMapTensor SOB-bind,
  mechanism proven on-card, `vllm-banz-phase1-progress.md` W7) is a SEPARATE later lever that sequences
  collective COMPLETION below the bridge to remove the per-AR host sync; it stacks on #4b but is not
  required for it. **#4b's fence target is the ~633 per-recipe RECLAIM fences, not the ~156 collective
  syncs** (the collective is only ~2.6% of the step, `fabric-not-a-perf-factor.md` — do not chase it as
  the #4b win).

---

## C) VALIDATION PLAN (correctness first, then fence count, then tok/s)

Reuse the existing override-proof gates (`WHOLEFORWARD_REBUILD_DESIGN.md` §4; `static_chain_ab.sh`).
Triple-check rule applies: engage-confirm + measure + cold re-run/baseline.

### C.1 Coherence (greedy matches stock) — THE gate
1. **G-engage:** with the all-layer device path armed, the first decode logs the device MoE engaging on
   **every** MoE layer (the `[STATIC-ENGAGE]` breadcrumb, `mmel_l6_chain.py:1212`) on all 8 ranks, with
   NO `synLaunch status=1` / NO `status=30`. `<8 ranks` or any status error = HARD fail.
2. **G-byte (per layer):** `MmelGreedyABHarness.compare_layer` (`mmel_l6_exec.py:754`) — each swapped
   layer's `(hidden, residual)` output MUST match `layer.mlp`/stock to byte (or relNorm < 1e-2),
   coherent-fallback to stock on mismatch (never corrupt a live gen). Run it on a captured decode step
   for ALL layers, not just one.
3. **G-greedy-e2e:** ARM (all-layer device path) greedy output MUST equal ARM_OFF (stock) greedy output
   on `bench/realprompt.txt` (602-tok) and the 3 standing prompts (Paris / 2+2=4 / France history).
   This is the real "device-resident all layers is still correct" proof.
4. **G-routing-live (the trap guard, `WHOLEFORWARD_REBUILD_DESIGN.md` §2.4):** assert the engaged MoE is
   the **live-routing** op, NOT the frozen `cap_routed` island — else the "match" is the frozen-vs-frozen
   false signal. Concretely: vary the prompt and confirm the per-token expert selection changes (or
   assert the bound weights are the live `layer.mlp` tensors, not the island's `_static_keepalive`).
5. **G-poison:** `MMEL_CHAIN_POISON=1` (`mmel_moe_island.py:144` / the residual poison
   `mmel_l6_chain.py:1092`) → greedy MUST garble. Coherent-under-poison = the device path did NOT run
   (silent stock fallback) = a FALSE engagement. This is the override-proof that the device output is
   actually served.

### C.2 Fence-count drop (the #4b mechanism signal)
- **G-count:** count `DRM_IOCTL_HL_WAIT_CS` (ioctl `0xc0386443`) per decode step via strace at the
  kernel boundary (no library can fake it; `WHOLEFORWARD_REBUILD_DESIGN.md` §4 G-count). Compare
  ARM_OFF (stock, ~633/step) vs ARM (all-layer device path). #4b should DROP the per-layer
  recipe-reclaim fences toward ~N (one per layer-tier boundary) or fewer. Record the exact delta.
- Cross-check with the recipe-count method (B3.3, `mmel_moe_island.py:54 (G-count)`): the stock path
  issues the MoE as many `ExecuteCachedGraph` launches; our path issues one op/recipe per layer-tier.

### C.3 Step-timer / tok/s (the honest number)
- **B4.2:** `gap_med` ARM vs ARM_OFF on `bench/tokps.py` + `bench/realprompt.txt` (602-tok decode rate,
  ex-TTFT — the standing directive). Cold re-run + baseline (the recipe-cache stale-replay trap:
  env-only knob flips replay the old graph — `recipe-cache-stale-replay.md`; the `envknob_fingerprint`
  guard in `common.env` now wipes on the knob hash, but **verify the cache was wiped** before trusting
  any flat result).
- **HONEST EXPECTATION (see §D):** in the EAGER `MMEL_WHOLEFWD` regime the fence-collapse is largely
  off the bs=1 critical path and eager re-adds host dispatch → net could be ≤0 at bs=1. In the
  **LazyOp-captured** regime (§B.2, no eager bypass) the fences collapse INSIDE the bridge capture with
  no eager penalty — this is the regime where #4b's fence-removal can actually move the wall. Measure
  both; the LazyOp-captured all-layer path is the one with upside.

---

## D) HONEST RISK + THE FENCE-COUNT MATH

### D.1 The fence math
- **Now:** ~633 host fences/step (`glm52-mmel-rebuild`: ~50% of the step; the device drains ~1.6
  recipes × N layers, each fenced). On 78 GLM layers that is ~8 fences/layer (the ~7 per-layer recipe
  breakers + the AR), matching the ~135 fragmenting submissions/forward figure
  (`glm52-single-stream-rootcause`).
- **After #4b (all-layer device-resident):** if each layer's static MoE tier collapses to **one
  recipe/completion** and the dynamic attention tier stays as its ~1 fused `mla_decode` dispatch, the
  per-layer fence count drops toward **~1-2/layer** → **~N to ~2N total = ~78-156** (GLM) /
  **~60-120** (M3), down from ~633. That is the targeted **≈50% step reduction** the goal names
  (the fences are ~50% of the step), IF the deleted fences are on the critical path.
- **Cadence:** ~633 → ~N fences means the host stops re-arming a recipe ~1.6×/layer; the device runs
  its 78-layer dep chain without draining to host between recipes. Expected step cadence: from the
  68.8 ms / 14.5 tok/s stock floor toward the proven Phase-1 **~17.5 tok/s** ceiling and, with the
  fences genuinely off the host path, toward the BW-limited single-stream regime (~15-25).

### D.2 The honest risk (do not overclaim — this is the project's recurring trap)
1. **The fences may already be off the bs=1 critical path.** Multiple independent on-card measurements
   (`L6_WIREIN_DESIGN.md` §8; `RUN_CHAIN.md` STEP-6: 554 sync-events/step sum to 248 ms = 3.6× the
   68.8 ms wall → they OVERLAP; AIP ~17%) found the bs=1 step is **device-serial-bound, not
   host-fence-bound** — deleting fences recovered ~0 in the L3 wire-in (ARM_ON == ARM_OFF 68.8 ms,
   `glm52-mmel-rebuild` 2026-06-22). **#4b's fence-removal is necessary plumbing but may be ~flat on
   single-stream** if the device-serial diagnosis holds at the all-layer scale. The measurement (§C.2
   G-count + §C.3 B4.2) is the decider; a flat tok/s with a real fence drop is a *correct, expected*
   outcome, not a bug.
2. **Eager-bypass penalty (the `MMEL_WHOLEFWD` regime):** running the whole decode eager re-adds host
   dispatch the graph was hiding → net plausibly ≤0 at bs=1 (`WHOLEFORWARD_REBUILD_DESIGN.md` §5.1-5.2).
   **Mitigation = the LazyOp-captured path (§B.2)**, which keeps the bridge capture and needs no eager
   bypass for the MoE tier. Prefer it; use eager only where the tier genuinely cannot be a LazyOp.
3. **The frozen-routing trap (§A's first fact):** if #4b accidentally ships the `cap_routed` island on
   all layers it is OOM *and* silently wrong (frozen 8 experts). G-routing-live (§C.1.4) + the
   live-weight bind (§B.2) are the hard guards; do not skip them.
4. **Concurrency is where the value most likely lands.** Every measurement converges on: single-stream
   is device-serial-capped (~15-16, lever = MTP/EAGLE-3 head); the **engine/fence-collapse's proven
   value is CONCURRENCY** (the device-serial floor amortizes across a batch). #4b is the substrate that
   makes bs>1 MTP-verify and the throughput lane device-resident — that is its strongest justification,
   per the dual-lane conclusion (`glm52-mmel-rebuild`). Build it for correctness + concurrency
   substrate; treat any single-stream gain as upside, measured not assumed.

### D.3 Definition of done for #4b
G-engage PASS (all MoE layers, 8 ranks, no status error) + G-byte/G-greedy/G-routing-live/G-poison
PASS (correct, live-routing, actually-served, HBM-free) + G-count delta recorded (~633 → ~N) + B4.2
tok/s recorded with its honest sign, on `bench/realprompt.txt`, cold re-run with the recipe cache
verified-wiped. The all-layer device-resident forward is then a correct, HBM-free substrate; whether it
moves single-stream tok/s is the measured result, and the concurrency/MTP-verify lane is the next build
on top of it.

---

## E) FILE INDEX (everything #4b touches or cites)
- `vllm-gaudi-patch/v1/worker/mmel_l6_exec.py` — seam: `_use_graphs` bypass (`:232`), `.forward` swap
  (`:441`), A/B harness (`:702`).
- `vllm-gaudi-patch/v1/worker/mmel_l6_chain.py` — per-layer driver `run_chained_decoder_layer` (`:802`),
  `_run_moe_block` + the `MMEL_BANZ_LAYER` gate to generalize (`:1140`, `:1202`), the live-weight bind
  surface `_moe_routed_via_our_op` to land (`:1281`), stream `reuse_bridge_device_and_stream` (`:1432`),
  `RawCsIslandReplay` (`:1474`).
- `vllm-gaudi-patch/v1/worker/mmel_moe_island.py` — the frozen-island replay (Phase-1 proof; KEEP as
  eager-bypass plumbing, NOT the all-layer driver): `MoeBlockIslandReplay` (`:428`), `warmup_once`
  (`:453`), `run` (`:737`).
- `mmel/cap_routed/manifest.txt` — the 149.6 MB / 58-tensor frozen island (the OOM + frozen-routing
  evidence).
- `mmel/WHOLEFORWARD_REBUILD_DESIGN.md` — the §2.4 frozen-routing gate, §5.3 "Build A subsumes B",
  the tiered shape, the on-card gates.
- `mmel/MEGAKERNEL_DESIGN.md` — HBM budget (int4 ~46.7 GB vs fp8 ~90.6 GB/card; the monolithic OOM).
- `hpu-moe-ext/tpc_kernel/opbackend/test_fused_expert_op.py` — the live int4 op signature + weight layout.
- `models/glm52-l6-static.env` — the serve env that mounts the mmel files + the op `.so`s.
- `bench/tokps.py` + `bench/realprompt.txt` — the standing 602-tok decode-rate metric.

---

## IMPLEMENTATION DONE — ON-CARD TEST PLAN (2026-06-25)

**OFF-CARD STATUS: the #4b code patch is written, card-free self-tested (12/12 PASS), and is
default-off byte-identical. It has NOT run on the card (a dump was running). Below is the exact
serve + gate plan to run when the box frees.**

### The ONE finding that simplified the build (read before the diff)
The plan's §B.2 "drive the forward via the live-weight-bound, live-routing int4 LazyOp" turned out
to be **already wired in the production serve, on every MoE layer**. When the serve image has
`HPU_MOE_FUSED_OP=1` + `PT_HPU_LAZY_MODE=1` + `HPU_MOE_REPACK_LAYOUT=stacked` (all set in
`models/glm52-l6-static.env`), a MoE layer's `layer.mlp(hidden_states)` at M=1 dispatches into
`MoeWNA16HPUMethod.apply_monolithic` → `_fused_moe_routed` → `fused_expert_routed.fused_moe_routed`
(`hpu-moe-ext/hpu_moe_ext/moe_wna16_hpu.py:1630`, `:2025`; `hpu-moe-ext/hpu_moe_ext/fused_expert_routed.py:337`),
which:
1. routes **LIVE** on-device — the real K-of-256 softmax+`torch.topk` + `present/scatter/topk`
   to get `active` (the K locally-routed experts), NO `.item()` (`fused_expert_routed.py:392-407`);
2. binds the model's **already-resident** stacked int4 experts in place via
   `index_select(0, active)` of `layer.w13_qweight.data` / `w13_scales.data` / the persistent
   stacked zp leaves (`moe_wna16_hpu.py:1736-1745`, `fused_expert_routed.py:533-538`) — **ZERO
   re-upload, ZERO extra HBM**; the held leaves are recipe inputs that survive every decode bucket;
3. emits `torch.ops.hpu_moe_ext.fused_moe(x, w13l, s13l, z13l, w2l, s2l, z2l, rws)` as **ONE
   LazyOp** the bridge captures (`fused_expert_routed.py:727`; schema =
   `moe_fused_expert_routed_backend_lazy.cpp:92-96`, a `Tensor[]` per-expert list op, NOT the
   single-expert `fused_expert`).

So #4b is NOT "write a new op-bind in the raised `_moe_routed_via_our_op` stub" (that is the
`MMEL_CHAIN_MOE_OP=1` raw-op-swap path, which would DUPLICATE this production wiring and is left
raising by design). The correct, HBM-free #4b is: **remove the 1-layer `MMEL_BANZ_LAYER` pin so the
chained `MMEL_CHAIN_STATIC` forward engages on ALL MoE layers, and drive the MoE tier through the
stock `layer.mlp` (= the live LazyOp), NOT the frozen `cap_routed` island replay.** This kills both
Phase-1 cheats exactly as §B.2/§B.3 prescribe.

### The diff (all in `vllm-gaudi-patch/v1/worker/mmel_l6_chain.py`; default-off byte-identical)
1. **`mmel_l6_chain.py:383 owned_all_layers_enabled()`** — new gate predicate (`MMEL_OWNED_ALL_LAYERS`
   in {1,true,yes,on,all}); inert unless `MMEL_CHAIN_STATIC=1` also engages.
2. **`mmel_l6_chain.py:418 _banz_layer_pins_this_layer_off(layer_idx)`** — extracts the literal
   Phase-1 1-layer pin (the old inline `MMEL_BANZ_LAYER` int-compare at the old `:1202-1205`) into a
   named, unit-tested helper; behavior byte-identical (True only when `MMEL_BANZ_LAYER` is an int and
   `!= layer_idx`).
3. **`mmel_l6_chain.py:925`** — in `run_chained_decoder_layer`, an `elif owned_all_layers_enabled():`
   that LOUD-raises if `MMEL_OWNED_ALL_LAYERS=1` is set WITHOUT `MMEL_CHAIN_STATIC=1` (it is a
   static-tail scope-widener, never a standalone driver — refuse to silently no-op a benchmark flag).
4. **`mmel_l6_chain.py:1290`** — in `_run_moe_block`, inside the `_chain_static_enabled()` +
   `_subchain_b_covers_bucket(...)` branch, a new `if owned_all_layers_enabled():` arm that drives
   the MoE tier through `return layer.mlp(hidden_states)` (the live LazyOp) on **every** covered MoE
   layer with a distinct `[STATIC-ENGAGE][OWNED-ALL-LAYERS]` breadcrumb + ENGAGE-MANIFEST. The frozen
   island (`_run_routed_moe_via_island_replay`) is NOT called on this path. The original
   `MMEL_BANZ_LAYER`-pinned island path is preserved below for back-compat / the Phase-1 mechanism
   proof (only reachable when `MMEL_OWNED_ALL_LAYERS` is OFF).
5. **`mmel_l6_chain.py:2168 _c11_owned_all_layers_gate()`** — new card-free self-test (registered in
   `_selftest`): predicate, the BANZ-pin logic, and the requires-`MMEL_CHAIN_STATIC` raise. Verified
   12/12 PASS via `$HOME/.venvs/m3calib/bin/python mmel_l6_chain.py`.

`mmel_l6_exec.py` and `mmel_moe_island.py` are UNCHANGED (the `.forward` swap already spans every
layer; the island stays as eager-bypass plumbing). The raised `_moe_routed_via_our_op` stub is
deliberately left raising (it is the `MMEL_CHAIN_MOE_OP=1` raw-op-swap, not the #4b path).

### Exact serve command + env (run when the card frees)
Use the existing key + its standing bring-up flags; add the ONE new flag `MMEL_OWNED_ALL_LAYERS=1`.

```
# ARM (the #4b all-layer device-resident forward): the glm52-l6-static env PLUS:
MODERN_MOE_EXEC=1 MODERN_MOE_EXEC_FAMILIES=glm_moe_dsa \
MMEL_CHAIN_STATIC=1 MMEL_WHOLEFWD=1 \
MMEL_OWNED_ALL_LAYERS=1
# (leave MMEL_BANZ_LAYER UNSET — #4b ignores it; the LazyOp tier has no OOM that needs the pin.)
# Bring-up safety (the routed-monolithic op is HBM-blocked; mirror glm52-conc-bsgt1):
VLLM_SKIP_WARMUP=true   # already in glm52-l6-static.env
# The recipe-cache stale-replay guard MUST wipe on the MMEL_* knob flip — VERIFY it wiped
# (recipe-cache-stale-replay.md): the env-knob fingerprint glob '__MMEL_ENVKNOB_RE' covers
# MMEL_OWNED_ALL_LAYERS (it is MMEL_*), so the cache auto-wipes; confirm the wipe in the log
# before trusting ANY flat number.

# ARM_OFF (the stock baseline, == 68.8 ms / 14.5 tok/s): MODERN_MOE_EXEC unset (everything else equal).
```

Mount/op `.so` env is unchanged (the `MOE_ROUTED_SO=.../libmoe_fused_routed_lazy.so` in
`glm52-l6-static.env` already points at the lazy routed op the `layer.mlp` LazyOp uses).

### Coherence check (greedy A/B vs stock) — THE gate (run in order; STOP on any fail)
1. **G-engage** — first decode logs `[STATIC-ENGAGE][OWNED-ALL-LAYERS]` ONCE and the per-layer
   ENGAGE-MANIFEST `engine-chain=OWNED-ALL-LAYERS(...)` on **all 8 ranks**, with NO `synLaunch
   status=1` / NO `status=30`. (The MoE tier is a LazyOp captured by the bridge, so no raw synLaunch
   fires for it; any synLaunch error would be the attention spine under `MMEL_WHOLEFWD`.) `<8 ranks`
   or any status error = HARD fail.
2. **G-greedy-e2e** — ARM greedy output MUST EQUAL ARM_OFF greedy output on the 3 standing prompts
   (Paris / `2+2=4` / France-history) AND on `bench/realprompt.txt` (602-tok). This is the real
   "device-resident on all layers is still correct" proof. (Per-layer byte A/B `MmelGreedyABHarness`
   `compare_layer` — `mmel_l6_exec.py:754` — can be armed for a finer-grained gate, but the LazyOp IS
   `layer.mlp`, so e2e greedy-equality is the load-bearing proof here.)
3. **G-routing-live (the frozen-routing trap guard, §C.1.4)** — vary the prompt and confirm the
   per-token expert selection changes (the routed path is live K-of-256; the frozen island is NOT
   used). Cheap proof: ARM greedy must differ across the 3 prompts AND equal ARM_OFF on each — a
   frozen-8 island would give identical garbage independent of prompt.
4. **G-poison (override-proof, §C.1.5)** — set **`HPU_MOE_POISON=1`** (NOT `MMEL_CHAIN_POISON`,
   which poisons the island we do NOT use). `HPU_MOE_POISON=1` scales the `_fused_moe_routed` LazyOp
   output ×1.15 at the bs=1 decode (`moe_wna16_hpu.py:2026`). With it on, ARM greedy MUST garble;
   coherent-under-poison = the LazyOp output is NOT being served (silent stock fallback) = a FALSE
   engagement. WIPE the recipe cache first.

### Fence-count + step-timer measurement (the #4b mechanism signal + the honest number)
- **G-count** — `strace -f -e trace=ioctl` on the worker, count `DRM_IOCTL_HL_WAIT_CS`
  (ioctl `0xc0386443`) per decode step (no library can fake the kernel boundary). Compare ARM_OFF
  (~633/step) vs ARM. #4b should DROP the per-layer recipe-reclaim fences toward ~N..2N
  (≈78-156 GLM / 60-120 M3). Record the exact delta. Cross-check with `HPU_STEP_TIMER=1`
  `HPU_STEP_TIMER_N=50` (already in the env) recipe/launch counts.
- **B4.2 tok/s** — `gap_med` ARM vs ARM_OFF on `bench/tokps.py` + `bench/realprompt.txt` (602-tok
  decode rate, ex-TTFT). Cold re-run + baseline; **verify the recipe cache wiped** on the
  `MMEL_OWNED_ALL_LAYERS` knob flip before trusting any flat result (recipe-cache-stale-replay.md).
- **HONEST EXPECTATION (do not overclaim — §D.2):** the bs=1 step is repeatedly measured
  device-serial-bound (the ~633 fences OVERLAP; AIP ~17%), so a real fence drop with FLAT tok/s is a
  *correct, expected* outcome, not a bug. #4b's value is the **correct, HBM-free, device-resident
  substrate**; its proven payoff lane is **concurrency / MTP-verify** (the device-serial floor
  amortizes across a batch), measured next on top of it — NOT a single-stream tok/s claim.

### Definition of done (unchanged from §D.3)
G-engage PASS (all MoE layers, 8 ranks, no status error) + G-greedy-e2e/G-routing-live/G-poison PASS
+ G-count delta recorded (~633 → ~N) + B4.2 tok/s recorded with its honest sign, on
`bench/realprompt.txt`, cold re-run with the recipe cache verified-wiped.

### LOUD list of assumptions that need ON-CARD confirmation (could not verify off-card)
1. **`layer.mlp(...)` actually dispatches to the LazyOp on the covered M=1 decode bucket** — verified
   from source (`apply_monolithic` → `_FUSED_OP` gate at `moe_wna16_hpu.py:2024` → `_fused_moe_routed`),
   and the env has `HPU_MOE_FUSED_OP=1` + `HPU_MOE_ROUTED_K=8` (so `_routed_ok` = `1*top_k(8) <= 8`
   is True at M=1). **CONFIRM on-card via the existing one-shot engagement INFO lines** ("fused_moe
   SEQUENTIAL op engaged with offset-0 clone ON" / "ENGAGED persistent static-input x buffer") — if
   those do NOT appear, `layer.mlp` is NOT taking the LazyOp on this build and the device-tier claim
   is false. (Untestable off-card: needs the HPU + the serve.)
2. **The LazyOp captures cleanly INSIDE the `MMEL_WHOLEFWD` eager-decode regime** — the MoE tier is a
   LazyOp (no raw synLaunch), but `MMEL_WHOLEFWD=1` makes the whole decode run capture-FREE
   (`mmel_l6_exec.py:251` `_banz_use_graphs` returns False on decode). **OPEN QUESTION:** under that
   eager bypass, does `torch.ops.hpu_moe_ext.fused_moe` still lower to its single recipe (it is a
   LazyOp emitting IR the GC compiles, which should be eager-safe), and does the multi-bucket x_buf /
   stacked-zp persistence still hold WITHOUT `wrap_in_hpu_graph`? The plan's §C.3 says to measure BOTH
   the eager (`MMEL_WHOLEFWD`) and the LazyOp-captured (no eager bypass) regimes; **#4b's upside is in
   the LazyOp-CAPTURED regime, so ALSO run a variant with `MMEL_WHOLEFWD` UNSET** (the MoE tier is a
   LazyOp and needs no eager bypass; the attention spine then stays captured too). If the
   capture-free regime regresses multi-bucket correctness, prefer the `MMEL_WHOLEFWD`-unset run.
3. **No new HBM is consumed at all-layer scale** — argued from "the LazyOp index_selects resident
   weights, zero re-upload" (§B.3) and the per-step transients are the same K-expert dequant the stock
   path already creates. **CONFIRM** the serve does not OOM at the ~20 MB headroom (it should not —
   the footprint is identical to the stock `HPU_MOE_FUSED_OP=1` serve, which already fits). If it
   OOMs, the diagnosis (island-duplication was the OOM, not spanning layers) is wrong.
4. **The `_subchain_b_covers_bucket` predicate returns True on the live M=1 post-norm hidden** — it
   bridges to `mmel_chain_layer.chain_layer_covers` (the captured [1,6144]=12288 B bucket). If the
   `mmel/cap_*` captures are absent/mismatched, coverage is False → the static tail (and thus #4b)
   silently delegates to stock (the 0-WARMED false-flat). The STRICT-mode `_captured_plan_undecidable`
   guard + the G-engage banner are the on-card detectors; **confirm `[STATIC-ENGAGE][OWNED-ALL-LAYERS]`
   actually fires** (not `[STATIC-DELEGATE]`).
5. **The op `.so` (`libmoe_fused_routed_lazy.so`) matches the captured recipe / serve image** — the
   `fused_moe` Tensor[]-list schema is confirmed from source
   (`moe_fused_expert_routed_backend_lazy.cpp:92-96`), but ABI/image drift can de-register it; the
   `_load_single_recipe_op` LOUD-warn + `HPU_ENGINE_STRICT` cover this on-card.

---

## ROUND-2 — "0 BREADCRUMBS" DIAGNOSIS + FIX (2026-06-26)

**On-card result (relayed): with `MMEL_OWNED_ALL_LAYERS=1 MMEL_CHAIN_STATIC=1 MODERN_MOE_EXEC=1
MMEL_WHOLEFWD=1` (all confirmed in-container, 8 ranks), the install banner + the capture-bypass
banner fired on all 8 ranks and greedy was COHERENT, but ZERO of `[STATIC-ENGAGE]` /
`[STATIC-DELEGATE]` / `[OWNED-ALL-LAYERS]` / island-replay fired across 3×20-token decodes.**

### Root cause (settled by source trace; subagent ad3d55d3 + log-level audit)
The owned arm WAS reached, but its proof was a **`logger.info`** line, and the serve filters INFO.
Decisive evidence, two parts:
1. **Dispatch is correct (NOT the bug).** The decoder-layer loop calls `layer(...)`
   (`_src/vllm_models/deepseek_v2.py:1329`), which picks up the instance-level `.forward` override
   set at `mmel_l6_exec.py:463`. `apply_model_specific_patches` (installs the swap,
   `hpu_model_runner.py:4831`) runs BEFORE `_maybe_wrap_in_hpu_graph` (`:4847`), so the captured graph
   contains the swapped forward; `modify_model_layers` (`:4838`) only adds a post-forward `mark_step`
   hook (it does NOT re-bind `.forward`). Single-stream decode pads to bucket size 1 under
   `VLLM_DECODE_BS_BUCKET_MIN=1` (`find_decode_bucket`, `hpu_model_runner.py:2902`), so the live
   post-norm hidden IS `[1,6144]`=12288 B and `_subchain_b_covers_bucket` returns True. So
   `run_chained_decoder_layer` runs every decode step, reaches the MoE block, the static block is
   entered, coverage passes, and the OWNED arm executes `return layer.mlp(hidden_states)`.
2. **The proof was INFO; the serve filters INFO.** EVERY breadcrumb the operator CONFIRMED visible is
   `logger.warning` (`[mmel.l6][ENGAGE]` install banner `mmel_l6_exec.py:538`; `[mmel.banz][ENGAGE]`
   `mmel_l6_exec.py:261`). EVERY breadcrumb counted as 0 was `logger.info` (STATIC-ENGAGE /
   STATIC-DELEGATE / OWNED-ALL-LAYERS). No `VLLM_LOGGING_LEVEL` override is set, so the serve runs at
   vLLM's default and the INFO breadcrumbs were filtered out — the path ran, the line was invisible.
   (Coherent output is fully consistent: the owned arm's `layer.mlp` IS the live LazyOp; so is the
   silent default — both produce coherent text, so output alone could not disprove engagement.)

**This was a logging false-null, NOT a dispatch/capture/coverage bug.** The `MMEL_OWNED_ALL_LAYERS=1`
arm is functionally meaningful (without it, `MMEL_CHAIN_STATIC=1` + unset `MMEL_BANZ_LAYER` would route
EVERY MoE layer through the frozen island → the 11.4 GB OOM; the owned arm routes them through the
resident-weight LazyOp instead).

### The fix (all in `vllm-gaudi-patch/v1/worker/mmel_l6_chain.py`; behavior-neutral, observability-only)
1. **`mmel_l6_chain.py:1297`** — the OWNED-ALL-LAYERS engage breadcrumb promoted `logger.info` →
   **`logger.warning`** (so it shows at the same level as the working install banners).
2. **`mmel_l6_chain.py:1332` / `:1353`** — STATIC-ENGAGE and STATIC-DELEGATE likewise promoted to
   `logger.warning`.
3. **`mmel_l6_chain.py:549 + :1261`** — new one-shot **`[STATIC-ENTER]`** WARNING emitted at the TOP
   of the `if _chain_static_enabled():` block, BEFORE the coverage decision, logging
   `owned_all_layers`, `banz_layer`, `layer_idx`, and the live `hidden` shape. This makes any future
   zero-engage run unambiguous: `[STATIC-ENTER]` present + no ENGAGE/DELEGATE ⇒ coverage predicate
   raised; NEITHER present ⇒ the static block was never reached (a real wiring/flag bug).

`_c11` self-test still 12/12 PASS; the changes are log-level/observability only — no control-flow or
output change, default-off still byte-identical.

### Re-test (when the card frees) — same serve command + env as above. Grep for (at WARNING):
- **`[mmel.l6_chain][STATIC-ENTER]`** — MUST appear once per worker (proves `_run_moe_block` reached
  with `MMEL_CHAIN_STATIC=1`; reports the live hidden shape + the flags).
- **`[mmel.l6_chain][STATIC-ENGAGE][OWNED-ALL-LAYERS]`** — MUST appear once per worker on all 8 ranks
  (the #4b engage proof). If `[STATIC-DELEGATE]` appears instead, coverage failed (the live hidden is
  NOT `[1,6144]` — re-check the decode bucket; report the shape from the STATIC-ENTER line).
- **`[ENGAGE-MANIFEST rank... engine-chain=OWNED-ALL-LAYERS(...)]`** — the WARNING-level manifest line
  (this was always WARNING; grep it as a cross-check).
- **Override-proof (decisive, not log-dependent):** run a variant with **`HPU_MOE_POISON=1`** — greedy
  MUST garble (the owned arm's `layer.mlp` LazyOp output is poisoned ×1.15 at `moe_wna16_hpu.py:2026`).
  Coherent-under-poison ⇒ the LazyOp output is NOT served (a real non-engagement) regardless of logs.

### Honest note (unchanged conclusion, now with a cited caveat on the LazyOp's nature)
The memory (`vllm-banz-phase1-progress.md`, subagent a1c6437, file:line) records that the live
`fused_moe` LazyOp **routes + gathers in PYTHON** (`fused_expert_routed.py:395-407,533-538`) and the
op itself only sees the K=8 pre-gathered experts, so its *recipe* is shape-frozen to the gathered set
(the routing is live in Python, but the captured op recipe is the 8-expert fold). This does NOT make
#4b incorrect (it is per-layer-correct and HBM-free — the gather binds the RESIDENT weights live), but
it tempers the "fence-collapse" upside: the per-layer recipe is still the same shape the stock
`HPU_MOE_FUSED_OP=1` serve already runs. So #4b's measured single-stream delta may well be ~flat
(§D.1 honest expectation holds); its value is the correct, HBM-free, all-layer device-resident
substrate for the concurrency/MTP-verify lane, not a single-stream tok/s claim.
