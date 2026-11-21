# GLM-5.2 / Gaudi-2 — Floor-Raising Roadmap (2026-06-26, 3 research fleets, all sources verified)

## The throughline (independently confirmed)
- **Single-stream (~14.5 tok/s)** = read-bound on the int4 weights across the 78-layer serial chain. Lever = produce more accepted tokens per weight-read (spec-decode) + read fewer bytes (sub-int4).
- **Concurrency / c40 (the ~128–150ms fixed tail)** = HOST ORCHESTRATION of the forward, NOT experts / NOT collective-BW / NOT the GEMM. Externally confirmed: TaxBreak (arXiv 2603.12465) — MoE dispatches 8–11× more kernels/token than dense, ~77–88% host at low batch, batch-invariant. Matches our own note (455 launches/554 syncs, quant-invariant; "fabric not a perf factor"). The flat-c32→c64 tail IS the signature.

## SINGLE-STREAM plan (~14.5 → realistic ~22–28, all PTQ/offline, no CUDA)
1. **Cheap:** retrain the EXISTING 3-layer MTP head on **self-distilled GLM-5.2 outputs** (FastMTP/HASS insight; fixes token-2/3 accept collapse 11%→56%). 2.26 → ~2.76 accept.
2. **Bigger:** train an **EAGLE-3 head** (multi-layer low/mid/high feature fusion + direct-token + HASS), run **topk=1 chain** → ~3–3.5 accept (~1.4–1.5×). Pure float-MME ops; offline-trainable (SpecForge supports MoE EAGLE-3); we have head_factory + SGLang's GLM-5.2 EAGLE path.
3. **Static narrow tree** AFTER the chain works (NOT dynamic — that capped us at 1.3% via host overhead + MoE expert-read inflation; static narrow tree proven on Ascend NPU = EAGLE-Pangu, our closest non-CUDA analog).
4. **Free stackers (zero new weights):** CLaSp / Draft&Verify **layer-skip self-draft** (training-free, lossless, 1.3–1.99×, draft reads only a fraction of 78 layers) + **double-buffered int4 weight prefetch** (DMA layer i+1 while MME runs layer i).
5. **Sub-int4 (if read-bound — this session measured SS at 68.8ms TPC-dequant-bound, so YES):** mixed-precision experts ~2.5–3-bit via **AutoRound** (`--enable_alg_ext`, g32/g64), KEEP MLA/DSA-indexer/router/shared-expert/first-dense/embeddings/lm_head at int4+. PTQ on the existing ckpt. Precedent: DeepSeek-R1 671B INT2-mixed retains 97.9%. Realized ~15–25% whole-model byte cut at batch-1 → ~1.15–1.25×. Our custom kernel handles mixed-bit (stock fused kernels can't).
   - SKIP NVFP4/MXFP4 — they read MORE bytes than int4 (4.5/4.25 vs 4.125) and Gaudi has no fp4 MME. SKIP ternary (PTQ collapses it).

## CONCURRENCY / c40 plan (the REAL wall = host tail; all framework-level, no CUDA)
1. **★ Fix the async-overlap trap (#4442) — the biggest un-pulled lever.** Async-scheduling ON ≠ overlapping. The next forward serializes behind the sampler tail (`sample_tokens()` p50 ~6.4ms vs `_sample` ~1ms — it does output/payload on the critical path). Fix: ONLY token-production gates the next forward; defer detok/output/payload. DIRECTLY targets the 128–150ms tail.
2. **Confirm the WHOLE step (incl. the TP all-reduce) is ONE HPU-graph replay.** PT_HPU_ENABLE_LAZY_COLLECTIVES is auto-on multi-HPU, but a collective OUTSIDE the captured graph re-incurs host launch every step. Verify bucket count small.
3. **Hunt the per-step device→host sync** (a `.cpu()` in the sampler/output path drains the HPU graph each step — leading suspect for a fixed multi-ms tail).
4. **Sampler: Gumbel-max-as-argmax** (argmax(logits/τ + gumbel); no sort/softmax/cumsum = native HPU ops in-graph) + **batch-sharded sampling** (kills the [B,154880] vocab-axis all-gather — shrinks a TP=8 collective). Highest-leverage sampler+collective combined win.
5. **Parallel detokenizer process pool** + **fuse logits processors** into one O(B·V) pass + **ban full-vocab logprobs** (Qwen2.5-7B b32: 10-logprob 0.078s vs full-vocab 29.29s) + **quantize lm_head fp8** (~0.65ms/step read).
6. **North star:** keep pushing `vllm-banz` whole-forward island replay — the megakernel literature confirms "engine owns the step, no host round-trip" is the only path that removes what graphs+async can't.

### NON-levers for c40 (confirmed dead — do NOT chase)
- **EPLB** = no-op under TP=8 (every rank computes 1/8 of every expert → balanced by construction); net-negative single-node.
- **Expert prefetch/cache/offload** = ZERO (experts are HBM-resident; these hide host→device transfers we don't make).
- **MTP/spec at concurrency** = decays to ~1.0× by batch 32–48 (EAGLE 1.73×@b1→1.21×@b128; Qwen3-8B 0.99×@b48). CONFIRMS our W4A8+MTP-flat finding. Keep MTP for SINGLE-STREAM only.
- **TBO / NanoFlow / Flux compute-comm overlap** = break-even-to-negative at c40's small batch (lose at ≤32 tok/dev).
- **DeepEP / DeepGEMM / FlashInfer / Triton megakernels** = CUDA/NVSHMEM/PTX, cannot run on Gaudi.
- **Don't optimize the GEMM** — Gaudi MME runs lm_head/GEMM at 99.3% of peak (saturated).

## Gaudi-2 hardware facts to respect
- TP=8 is Gaudi-2's FAVORABLE collective regime (better 8-way bus-BW than A100); all-reduce BW isn't the bottleneck once in-graph — per-step launch+latency is.
- ≤128B gathers hit only ~15% of HBM BW (vs A100 36%) — any sampler/KV/expert-gather TPC kernel MUST coalesce or waste 2.4× of the 2.46 TB/s.
- e4m3 saturates ±240 (not ±448); fp8 KV = near-lossless CAPACITY lever for 1M ctx (not batch-1 speed).

## c40 TAIL PROFILED ON-BOX 2026-06-26 (triple-checked, `mmel/C40_TAIL_PROFILE.md`) — CONVERGES with SS
Per-step @ c32 = 175.3ms but device computes only ~30ms (AIP 15-18%, **device 82% IDLE**). The ~130ms tail =
**152,383 kernel invocations + 497 synchronizeEvent / forward** — fragmented graph forces a host round-trip
between kernel groups (497 × ~0.25ms ≈ 124ms ≈ the tail). ALL 4 "easy" suspects REFUTED on-box: sampler 0.6ms
(greedy), .cpu() deferred, all-reduce in-graph (0.2% device), async-overlap effect-real-but-not-the-named-trap.
→ The research's portable quick-wins (Gumbel sampler, async-overlap fix) DON'T apply — already pulled.
**THE wall = launch/sync fragmentation. THE fix = full-forward HPU-graph capture + kernel consolidation**
(hpu_model_runner.py::_execute_model_generic + the HpuModelAdapter graph-wrap + the per-step mark_step fences
in sample_tokens). Expected 175→~90-125ms (1.4-1.9×). SAME wall at bs=1 (68.8ms base step). SS head is a DEAD
END (data-starved EAGLE-3 22% vs native 68.7%; even perfect head ~16-18 tok/s; accept barely converts at bs=1).
**BOTH floors converge on ONE lever = vllm-banz full-forward island replay (Phase 1 proven coherent).**

## ⛔ BANZ PHASE 2 RESULT 2026-06-26 — FENCE-COLLAPSE IS A DEAD END (measured, triple-checked, `mmel/BANZ_PHASE2.md`)
The c40 profile's "497 syncs × 0.25ms ≈ 124ms reclaimable tail" is FALSIFIED on-card. Removing the fences
does NOTHING:
- Per-layer mark_step coalesce (N=4 removes ~58 of 78 fences): bs=1 69.1→69.2ms, c32 189.9→189.3ms = FLAT.
- MMEL_DECODE_NOFENCE=1 (removes the per-step decode fences, engage-proven on 8 ranks, COHERENT): bs=1 69.1→69.1, c32 189.9→187-191 = FLAT.
- Full mega-recipe collapse (N≥60) REGRESSES to 6.4 tok/s (compile blowup). Consolidation is WORSE, not better.
**ROOT (corrected): the decode step is DEVICE-DEPENDENCY-CHAIN bound, NOT host-launch-fragmentation bound.**
The 497 syncs = ~78 per-layer hooks (removable→flat) + ~78 MLA mark_steps + **~156 TP=8 all-reduce collectives
(HARD data dependency; collective_seq already a NO-GO, 82% non-reclaimable)** + ~4 brackets. Under async-scheduling
the host launches OVERLAP the device's dependency-chain latency, so removing them frees nothing — the 82%-idle is the
device stalling on ITS OWN sequential dependencies (78 layers × MLA+MoE+2 all-reduces), not on host launches.
**This SUPERSEDES the kernel-fragmentation-wall "megakernel consolidation is the lever" hypothesis (also falsified)
and extends the 3×-confirmed bs=1 device-serial verdict to c32.**
→ THE HONEST LEVERS: SS = head accept-multiplier (but data-starved/capped ~16-18). Concurrency = MORE WORK PER STEP
to fill the device idle (W4A8 1.33× + routed-BSGT1 +15%, BOTH SHIPPED), NOT fewer syncs. The device is 82% idle at
c32 → huge THROUGHPUT headroom (box can serve >>40 concurrent at similar latency); per-step LATENCY is dependency-bound.
The ONLY deeper c40 lever = cut the 156 collectives (mostly no-go). Engine/fence/consolidation work = STOP.
