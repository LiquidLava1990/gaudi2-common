# MMEL Build Status — overnight 2026-06-20/21

## PATH (settled by feasibility dig + 20-agent atom blueprint): B3 csengine
Fragmentation has 2 root causes: (1) intra-op per-expert fan-out = SOLVED (int4 OpBackend MoE);
(2) inter-recipe HOST FENCES (~633 synchronizeEvent/step, ~50% of the wall) = NOT removable by
HPU-graph capture (capture replays N recipes with N synLaunch/N event-records — measured FLAT,
slices 2/14/16) and NOT from Python. ONLY a device-replayed COMMAND BUFFER with on-device
SOB/monitor/FENCE sync deletes the fences. => MMEL = csengine core chaining the EXISTING
GC-compiled kernels (int4 MoE recipe + DSA indexer/sparse-attn + MLA) reusing them as payloads.
Ceiling ~40-50 single-stream (delete ~50% fence-wall ~2x, x head 1.6x). Also helps concurrency latency.

Refs: Gaudi2_Kernel/MODERN_MOE_EXECUTION_LAYER_DESIGN.md (630-line architecture);
20-agent blueprint = task w2cyjw66l; feasibility = wgoo371uq. csengine/ = the B3 code (M0 PoC).

## GENERALITY (the point of the hard build): 1 layer-chain, 4 families
DeepseekV2ForCausalLM is subclassed by DeepSeek-V3 + Kimi-K2.7 + GLM-5.2 -> one MLA+MoE chain.
Qwen3-MoE = GQA variant reusing the same composition. Config-adapter reads hf_config -> params.
Seam: apply_model_specific_patches (hpu_model_runner.py:634, pre-capture, sanctioned) + DecoderLayer.forward swap.

## BUILD PLAN (W0-W7+), each byte-exact gated
- W0 [GATE, BLOCKS ALL]: cs_poc MODE=csengine on a clean card -> confirm 1 wait/step + device OPERATIONAL.
- W1: libcsengine submit core (drm_ioctl/submit_signal/submit_wait_on/host_wait -> reusable lib + py bridge).
- W2: route-2 spike -- replay a GC recipe via our own completion-group, byte-exact (HIGHEST RISK).
- W3: single-island CB = 1 MoE layer, on-device synced (csengine M1).
- W4: MLA decode softmax-tail fuse (largest fusible attention surface).
- W5: DSA indexer + sparse-attn into the CB chain (GLM/DeepSeek).
- W6: whole-layer chaining + per-token address patcher (M2, first tok/s measure).
- W7+: full forward, 4-family config-adapter, fp8-KV, TP=8 on-device collective (HARD: collective-wait hook = -EINVAL stub).

## HONEST SCOPE: this is an eng-weeks build. Tonight = W0 de-risk + W1 foundation + as far as it goes.
## Floor shipping in parallel: EAGLE-3 head ~1.6x (16->~26), stacks on top of the rebuild.

## PROGRESS LOG
- 2026-06-21 ~03:00 — blueprint + architecture + feasibility all landed; path = B3 csengine; starting W0 (de-risk gate).
- 2026-06-21 ~04:40 — spec-at-concurrency = **DEAD** (glm52-async max-num-seqs 8, per-stream 3.35 < no-spec 5; accept 34-38%, verify cost eats it). **HUGE concurrent multiplier = fp8 ROUTE-B** (M3-proven 3.6-5x: 39.9/63.4/90 @c10/30/50 vs int4 10.6/17.4/17.6) — 10-agent wave (w1djzle7c) cracking its HBM soft-spot (resident-fp8 OOM verdict suspect; M3 ran it -> on-the-fly int4->fp8 likely fits). Stacks with MMEL fence-deletion (2x) -> 7-10x conc territory. Running the head (single-stream multiplier) on the free card meanwhile.
- 2026-06-21 ~04:15 — **W1 on-card: CORE ENGINE PROVEN.** PART B PASS (real PDMA->MME->TPC chain, 1 host WAIT/step, on-device monitors armed 5/step) + PART C PASS (MME q16 accepts hand-built USER_ALLOC CB) + PART E PASS (mapped-CQ alloc). PART D (staged->1-completion) FAIL = bug in cseng_staged (fix later; PART B on-device sync is the load-bearing path, staged is a convenience). Device OPERATIONAL throughout (the "needs reset" banner = stale classifier reading historical hard=3). W3 route-2 agent (ad3ba5fa) building. **NOW completing the 2 deployable multipliers: spec-at-concurrency (concurrent) + EAGLE-3 head (single-stream).**
- 2026-06-21 ~03:55 — **W1 libcsengine COMPLETE** (off-card): libcsengine.{c,h} + csengine_py.py bridge + w1_test.c + W1_BUILD_TEST.md + W2_SCOPE.md; build.sh clean, abi_check PASS, selfcheck PASS. **W2 VERDICT = route-2** (synLaunch binds per-token addrs at launch -> collapse the ~633 per-launch synEventSynchronize into ONE step completion via our mapped-CQ; route-1 recipe-packet-replay INFEASIBLE -- SCAL owns submission, no scal.h). Caveat: collapsing SCAL per-launch completion to 1/step must be proven in M1/M2. NOW: build+on-card-test libcsengine (3-engine chain) -> W3 single-island CB.
- 2026-06-21 ~03:35 — **W0 PASS (DECISIVE)**: cs_poc csengine = **8->1 host fences** per 8-op step; on-device monitors armed (350/50 reps); device OPERATIONAL; **2.71x wall (PDMA), 2.10x (MME+TPC compute queues)**. Verdict: "the per-op host fence IS DELETABLE at the CS layer." Core thesis validated on silicon. cs_poc flags: default qa=0/qb=4 PDMA; compute queues MME_0_0=16, TPC_0_0=20 (AVOID EDMA 8-15: RAZWI->reset). W1 libcsengine building (agent). NEXT: build+test libcsengine (3-engine chain) -> W2 route-2 recipe-replay spike. Card freed via `containers.sh off` (NOT _stop_all). Head 1.6x floor paused (re-runnable; build has card priority).
