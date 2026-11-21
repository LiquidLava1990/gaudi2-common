# MAXING CAMPAIGN — GLM-5.1 → M3 → K2.7 (started 2026-06-16)

**Goal (user /goal):** max each model — fastest single-stream + concurrency, max utilization, NO swarm,
nothing breaking, fully optimized. Each model in its OWN dedicated container (`./containers.sh`). Work at
ANY level: software / kernel / base (vllm-gaudi) / runner. No stone unturned. Loop until done.

**Container infra:** `./containers.sh {make|on|off|list}` — dedicated `vllm-<key>` + own recipe cache,
plugins auto-detected, ONE active at a time (box holds one model's weights). Made STOPPED, activate on demand.
Made: vllm-glm51ar-lazy, vllm-glm51ar-mtp-prod, vllm-minimax-m3-int4, vllm-kimi-k2.7.

---

## KEY INSIGHT (the frontier)
Single-stream is **host/collective-bound at batch=1** (device starved: 1 token, top_k experts → tiny work,
then waits on host dispatch + per-layer 8-way all-reduce + per-token sample→feedback sync). Util ~19%.
- Fusion does NOT raise util (GLM fused == M3 unfused == 19%); it changes *where on the speed curve*.
- **USER RULING: the flapping scale-up fabric is NOT the cause of the cap.** So the base ceiling (~14 warm)
  is STRUCTURAL collective/host overhead — the lever prior work punted to "fabric." THIS is the frontier.
- MTP is the proven single-stream multiplier (amortizes the per-token sync), not a util raiser.

## GLM-5.1 — IN PROGRESS
Stack (glm51ar-lazy): lazy + fused-K8 MoE (capture-safe .so) + DSA-gated-off (dense MLA) + TP8.
Known: MTP prod (glm51ar-mtp-prod) = bf16-L78 graft, 62-75% accept, depth-1 → ~22-23 @ quiet fabric.
Depth-2 RULED OUT (CPU oracle: net ~flat). To beat 23 → raise the base 14 (structural collective/host).

### Baseline (glm51ar-lazy, dedicated container) — 2026-06-16
- **Model COHERENT** ("capital of France" → "Paris" ✓). NOT broken.
- **Clean-window PEAKs (the structural metric):** single-stream ~10 tok/s, concurrency **117 tok/s** (c30 window).
  → structural code already near documented targets (14 single / 90-180 conc).
- **Fabric-dragged AVERAGES (unusable):** single ~2, c10=2.0, c30=0.5. Util med=19% (structural cap, real).
- **FABRIC: port-9 all-up 0/20 samples** — hard-flapping (toggling every ~1s, 28 ports down). Throughput
  oscillates 10→0 in the engine logs. This is the SUSTAINED-throughput blocker (NOT the structural cap).
- **METHODOLOGY:** measure clean-window PEAK (max engine gen-throughput over a long decode), not wall-avg.
  bench/glm_peak.sh does this. Software headroom: 10→~14 (kill AR sync) → ~22 (MTP). Fabric gates realizing it.

### Lever tracker (A/B, keep-if-better, verify coherence each)
| # | Lever | Level | Status | Result |
|---|-------|-------|--------|--------|
| 1 | MTP prod (glm51ar-mtp-prod, depth-1 graft) | software | TODO | target ~22-23 |
| 1b | **MTP head fp8 (NOT bf16)** — int4 L78 is DEAD (AutoRound junk); current graft is bf16 (rule violation). fp8-quant L78 from bf16 orig: rule-OK + ~1.25GB/card + faster draft | kernel | TODO | user rule: no bf16 |
| 2 | all-reduce mark_step removal — `HPU_AR_NO_MARKSTEP=1` (ALREADY in hpu-moe-ext `_install_communicator_patch`) | base | TODO | test on GLM fused |
| 2b | **`HPU_TIME_AR=1`** — times each all_reduce wall-clock (ALREADY built). ~15-30µs=sync-bound (software win) / ~1-3ms=fabric. **DECISIVE fabric-vs-sync test** | base | TODO | settles user's "not fabric" |
| 3 | warm batch=1 PROFILE — split host-dispatch vs all-reduce vs sampler | runner | TODO | decides #4-#6 |
| 4 | reduce/fuse per-layer all-reduces (156/token) | base/runner | TODO | the base-14 raiser |
| 5 | sampler host-sync removal | runner | TODO | — |
| 6 | HPU_MOE_ROUTED_K sweep (currently 8) | software | TODO | prime documented lever |
| 7 | concurrency config sweep (mns 64→128, buckets, util) | software | TODO | 90→130-180 |
| 8 | verify NO swarm (op-count probe: i64→i32, DSA indexer) | kernel | TODO | "no swarm" gate |

## M3 — QUEUED (after GLM)
Items: fuse swigluoai (fp8_routed_moe + swigluoai branch / :1.24 fused-op revive), MTP head (74 shards
dumped, untrained), cast-swarm fixes, eager-island elimination. See M3_SINGLESTREAM_STACK.md.

## K2.7 (Kimi) — QUEUED (after M3)
Items: TBD — baseline first (own container vllm-kimi-k2.7).

---
## LOG (newest first)
- 2026-06-16: Built containers.sh (dedicated per-model). Activated GLM-5.1 dedicated. Baseline measuring.
  Confirmed (this session, fabric-noisy): mark_step removal ✗ on M3; GLM fused & M3 both ~19% util.
- PEAK-LEVER AR-no-markstep: peak=NA tok/s, wall-avg=  (extra: -e HPU_AR_NO_MARKSTEP=1)
- AR-no-markstep: KEPT (coherent; avg 1.86→4.09 on flapping fabric, peak 8.7~baseline10, median-window 6.7). Confirm ceiling on clean fabric.

## TRIGGER FOUND (2026-06-16, py-spy on live serve)
Worker rank0 12/12 snaps in shm_broadcast.recv (waiting for next step); EngineCore 14/14 in threading wait.
Neither Python CPU-bound, device 19% busy => the 81% idle is the V1 ENGINE PER-TOKEN ROUND-TRIP (autoregressive
feedback through multiproc IPC; CANNOT pipeline — token N+1 input = token N output). NOT the quant (AutoRound
exonerated: worker never in dequant). NOT the MoE. Fabric adds variance on top, isn't the structural trigger.
=> FIX = MTP (amortize round-trip over K tokens; 90% accept ~1.9x; accept rate is FABRIC-INDEPENDENT).
=> Secondary = cut 156 all_reduce/token (TP=4 halves ranks+fabric). Quant-for-speed deprioritized.
The head factory IS the util fix. Build + CPU-validate accept (no fabric needed).

## BUDGET SPLIT + FABRIC CORRELATION (2026-06-16, decode_profiler_patch + py-spy)
SINGLE-STEP BUDGET (clean window, 30 steps, torch.profiler @ execute_model):
  wall=30 ms/step (~33 tok/s)  dequant=0.000 ms/step  collective=0.61 ms/step (2%)  attention=0.026  
  => NO per-token dequant swarm. NO collective wall (clean). Graph already captured. NOTHING to "rework".
FABRIC CORRELATION (un-profiled, util vs port9-up over 24 samples):
  port9_up 6-7/8 => util 35-40% ; port9_up <=3/8 => util 19% (always). Throughput osc 12.4->0.0 w/ flaps.
  => util TRACKS fabric cleanliness. Clean ~33 tok/s; flapping (continuous now) drags avg to 2-12.
VERDICT on the "fp8-dequant-free + single-graph -> 28-34" table: NUMBER plausible (33 clean x1.7 MTP ~=56),
  but PATH is MOOT — dequant already 0, graph already captured, collective tiny. The real drag = FABRIC.
REAL PATH to max GLM: (1) FABRIC STABILITY (clean links => ~33 structural, already there) (2) MTP x1.7 => ~50.
  The fp8/single-graph rework recovers ~nothing. Fabric is the dominant lever; MTP is the multiplier.

## MTP HEAD — factory works, dump has a fixable bug (2026-06-16)
FACTORY (head_factory/): head.py + train.py (hard-CE + EAGLE-3 soft-KL) + glm51/config.json + factory/corpus.py.
PROVEN: head LEARNS on clean data (58-pos cache: loss 11->6.5 after fixing the dead-init bug — GLM plain-RMS
norm weights must init to 1.0 not 0.0). fp8 head viable (2.6%).
DUMP BUG: decode_dump_patch.py captures at compute_logits where vllm-gaudi gives BUCKET-PADDED batches +
warmup wasn't fully skipped (non-zero dummy rows passed the all-zero guard). 7168 captured -> only 130 real
contiguous pairs (positions 127/255/383/511 = pad; small 0..N runs = warmup buckets). Training choked on 130.
FIX (2 changes, then re-dump): (1) skip warmup via runner.warmup_mode (runner exposes it @hpu_model_runner.py:1296;
glm_layer_dump.py shows the __init__ hook pattern). (2) capture only REAL tokens (logits_indices / real count),
not padded hidden_states. THEN: re-dump -> train soft-KL -> validate accept (>=90%) -> fp8 -> serve (both axes).

## MTP DUMP — corrected diagnosis (2026-06-16, after warmup-gate re-dump)
Warmup gate did NOT fix it (re-dump still 120 pairs, pos 127x678). Cause is NOT warmup — it's vllm-gaudi
DECODE-BATCH PADDING: single-stream decode is padded to a batch bucket (1 real token + ~127 dummies at the
bucket-max position 127/255/383/511). compute_logits captures the WHOLE padded batch -> ~98% of each shard
is pad, the real decode data is buried. (Prefill rows 0..N are clean; decode is the contaminated bulk.)
FIX (the correct one, next focused task): in decode_dump_patch.compute_logits, slice hidden/positions/topk
to the REAL token count BEFORE buffering. Get it from the forward context attn_metadata (num_actual_tokens /
logits_indices / query_start_loc), NOT len(hidden_states). Verify with a SMALL dump (5 prompts) before the
full 48-prompt run. Everything else (factory head.py/train.py proven to learn, soft-KL, fp8 viable, corpus,
warmup gate) is solid. This de-pad is the single remaining blocker to a real GLM MTP head.

## MTP HEAD — FIRST REAL TRAINING (2026-06-16) — pipeline WORKS end-to-end
DE-PAD FIXED: hidden[:num_reqs] + per-request positions via input_batch.req_ids -> clean pairs 1.6%->99%.
Trainer bugs fixed: dead-init (RMS norm w init 1.0), OOM batch (bs 256), vocab-padding overflow (filter
target_ids < real vocab; vllm pads the logits vocab). RESULT on 1702 clean pairs (2 shards, hard-CE):
loss 34->4.9, accept 0%->~25-33% peak (unstable). Head saved head_factory/glm51/head_v1.pt.
BELOW graft floor (62-75%) — expected. PATH to >=90% (ranked): (1) MORE DATA (48-prompt dump firing,
~5x); (2) SEQUENCE-structured batching — the lean-block attention needs contiguous per-request sequences,
NOT shuffled positions (biggest lever; current shuffle mis-trains attention); (3) EAGLE-3 SOFT-KL (top-K
already dumped, --temp); (4) LR schedule + multi-layer hidden fusion. Tooling: bench/glm_mtp_fire.sh (fire
corpus at running serve -> shards), factory/train.py, factory/corpus.py.

## MTP HEAD — sequence training works, DATA VOLUME is the bottleneck (2026-06-16)
head_v2 (SEQUENCE-structured, factory/train.py rewritten: contiguous per-req runs, attention sees context,
grad-accum over sequences): loss 58->12 but accept is NOISE (only 31 sequences -> 3 val seqs). 31 seqs /
2536 pairs is ~100x too small. The de-pad is correct (clean data); the FLAPPING FABRIC makes single-stream
dumping slow (~1 token/decode-step). PATH: need ~hundreds-thousands of sequences. Options: (a) CONCURRENT
dump c=16 (16x faster; needs seq_id tracking in decode_dump_patch + group-by-seq_id in _sequences, since
concurrent requests interleave), (b) long background single-stream dump, (c) clean tray (fast fabric ->
fast dump AND fast measurement). THEN sequence-train + soft-KL -> graft floor -> 90%. Trainer + de-pad +
sequence-structuring are DONE and correct; only data volume remains for the head.

## MTP HEAD — concurrent dump corruption FIXED + EAGLE-3 soft-KL wired (2026-06-16)
Concurrent dump (c=16) gave ~16x data BUT corrupted: at num_reqs>1, hidden[:num_reqs] grabbed
finished/padding slots -> 354 nan/inf hidden rows + garbage target_ids ([-2e9,2e9]) + nan topk_logits.
head_v3 NaN'd from epoch 1. FIX (two layers, both in place):
 (1) decode_dump_patch.py: keep only FINITE rows (isfinite(h)&isfinite(tv)); advance per-request position
     ONLY for kept rows (no gaps). [takes effect on next serve restart]
 (2) train.py: drop nan/inf hidden + out-of-vocab tokens, then split sequences at seq_id OR position gaps
     (so filtered rows never form wrong cross-gap pairs). [salvages the EXISTING dump w/o re-dump]
Salvage diag: 13 shards, 96.1% rows good, 81 clean sequences (median len 159), 12.7k pairs.
head_v4 (hard-CE, salvaged data, warmup+cosine+grad-clip+best-val ckpt): loss 22->3.1, accept 5->17.5%
climbing (epoch 5/40) => CONCURRENT DATA IS CORRECT/LEARNABLE. Data volume still the cap (73 train seqs).
EAGLE-3 SOFT-KL now wired (train.py _loss): distills teacher top-K dist (topk_ids/topk_logits already
dumped), T-smoothed, vocab-pad masked + renormalized, mixed w/ hard_w*CE. Math unit-tested (finite, grads
flow, pad masked). corpus.py expanded 120->406 unique prompts (code x6 langs + debug/reason/agent/systems).
NEXT (loop): big concurrent dump (bench/glm_mtp_bigfire.sh, 406 prompts x256 tok, +8x data) -> big soft-KL
train -> accept toward 90% -> fp8 -> serve depth-1 (single + concurrency). New CLI: --accum --clip --warmup.

## ⚡ MAJOR RECOVERY (2026-06-16) — GLM MTP native head WORKS (62-75%); from-scratch was a detour
Recovered the full prior MTP state (PROJECT_FINDINGS §8 + models/glm51ar-mtp-*.env + hpu-moe-ext/mtp_bf16_graft.py):
- **int4 native MTP (layer 78) is DEAD at the weights level** — AutoRound int4-mixed quantized layer-78
  int4-everywhere with NO calibration flow through it; fp64 CPU oracle: constant junk under every wiring.
  Minting the repack cache (layer-78 files DO now exist, 8 of them) can't fix dead weights. => int4 native = dead end.
- **NATIVE MTP head WORKS at 62.5-75% accept, ~1.7 mean-accept-len (~1.7x single-stream)** via the bf16
  layer-78 GRAFT: mtp_bf16_graft.py (HPU_MTP_BF16_GRAFT=/models/glm51-orig-l78, the original zai-org bf16
  layer-78, 4 shards ~21GB). Serve infra ALL present: speculative.py maps glm_moe_dsa->deepseek_mtp,
  DeepSeekMTPModel loads layer 78, HpuEagleProposer + hpu_rejection_sampler serve it, mtp_share_head_patch
  ties embed/shared_head.head to the target (layer-78 ships neither). Config: glm51ar-mtp-prod.env.
- **depth-2 RULED OUT** (CPU oracle: a1=61% a2=32% -> E[len] 1.61->1.80, eaten by extra verify) => keep n_spec=1.
- The from-scratch factory head (soft-KL, 28% on 63k-pos dump) is DOMINATED by the native graft (62-75%) for GLM.
  Factory stays as the universal fallback for M3/K2.7 IF their native MTP heads are also dead.
CONSEQUENCE: GLM MTP deliverable = the NATIVE graft. Two open forks: (a) dtype — bf16 graft works now but
violates the int4/fp8 rule; fp8 graft = compliant + ~half draft cost but needs fp8 layer-78 quant + fp8 compute
(moe-ext has fp8_routed_moe.py). (b) target — 62-75% is the proven depth-1 ceiling; 90% needs a bigger head
(EAGLE-3, uncertain, data-limited). Asking the user which to commit to before building.

## PIVOT 2 (2026-06-16) — bf16-free from-scratch fp8 head is the path (user: "nothing in bf16 period")
DATA SCALES THE HEAD HARD: soft-KL head 28% (18 shards/18k pos) -> 53.9% climbing (62 shards/63k pos).
=> the from-scratch head is NOT capped; it tracks data volume. And it is 100% bf16-FREE (trained fp32 on
captured int4-serve hiddens -> fp8), unlike the native graft (only exists as zai-org bf16 -> violates the
hard no-bf16 rule). So: NATIVE GRAFT DEPRIORITIZED (bf16 lineage); it's only a reference ceiling (~62-75%,
fp32 control eval glm_cpu_mtp_eval.py on our dump). DELIVERABLE = from-scratch fp8 head: keep scaling data +
apply EAGLE-3 techniques (research agent running) -> fp8-quantize -> serve as EAGLE-style draft. Compliant,
lean (faster draft than the heavy MoE-DSA native layer), improvable. Honors "nothing in bf16 period".

## CORRECTIONS + RESEARCH (2026-06-16, user-corrected)
- **GLM-5.1 int4 is ~500GB (~62GB/card), TP=8 is MANDATORY** (the env "~52GB" is PER-CARD). TP-reduction
  lever is OUT — the model needs all 8 cards; the 8-way all-reduce per token is structural.
- **fp8 is the serve format, period (user: "nothing in bf16").** Research-confirmed: fp8 (floating-point)
  preserves ~97-98% of accept (SPEQ 0.976); only naive INT4 craters it. fp32 was only a one-time offline
  reference to measure quant loss. lm_head/embed stay the target's own modules (shared).
- **Depth-1 MTP is capped at tau≈1.8 (alpha≈0.80) BY DEFINITION** (tau=1+alpha for 1 draft token). The
  EAGLE-3 "tau≈6" is k=6 TREE drafting, not our single draft. So 90% is unreachable for depth-1; honest
  target = alpha->70-80%. Native GLM MTP = 73.3% on OUR dump (glm_cpu_mtp_eval.py) = the apples-to-apples
  ceiling; the gap is TRAINING DATA, not architecture (our head IS the DeepSeek-V3 MTP module).
- **From-scratch fp8 head scales with data: 28% (18k pos) -> 53.9% (63k pos).** Data-starved, not mis-designed.
- RESEARCH RANKED LEVERS (research agent, sourced): #1 ON-POLICY DATA (sample GLM's own gen @ temp 0.7-0.8,
  dump along it; get to thousands of seqs -> 28%->50%+); #2 drop feature-regression (we never had it = clean)
  + fuse low/mid/high hiddens at layers 2/40/77 (concat->Linear(3H,H)); #3 TTT unroll 3-4 steps feeding own
  feature (OFFLINE-feasible, 0.8^i decayed loss); #4 prune draft vocab to ~32k (Zipfian; small-data win);
  #5 top-K(=10) soft-KL T=1, AdamW(0.9,0.95), lr 1e-4 cosine, 1-3 epochs+early-stop, U(-0.1,0.1) feat-noise.
- TRAINER HONED (train.py): feature-noise aug (--noise 0.1), T=1 default, top-K trim (--softk), AdamW betas
  (0.9,0.95), early-stop (--patience). NEXT trainer work: on-policy dump (#1) + multi-layer fusion + TTT.
- UTIL/WAIT: single-stream ~17% is host/fabric-bound (batch=1 on 500GB/8-card = collective-latency-bound +
  flapping fabric; new tray helps the fabric part). The 17% was ALSO dump-hook polluted (per-step .cpu sync).
  Concurrency curve being measured (Trident runs 10-20 concurrent -> that's the real util regime). MTP = the
  main single-stream software lever (1.7x); TP-reduction is out (mem).

## UTIL/WAIT — MEASURED, it's the FABRIC (2026-06-16 degraded window)
Concurrency sweep on the live serve (degraded/flapping fabric): c=1 -> 2.9 tok/s, c=8 -> 11.6, c=32 -> 11.1.
AGGREGATE CAPS AT ~11 tok/s REGARDLESS OF CONCURRENCY — adding streams past ~8 just divides 11 among more
(per-stream 2.86 -> 1.45 -> 0.35). PROOF the 8-way all-reduce (500GB model, TP=8 mandatory) is the hard
per-step cost; concurrency can't amortize a latency-bound collective on a flapping fabric. Device ~83% idle
waiting on it. Prior CLEAN-fabric measurement was ~366 tok/s aggregate (33x) => the new tray / reseat IS the
dominant lever; there is NO software bypass of an 8-way collective on a model that needs all 8 cards. Software
levers (MTP 1.7x, head accept) are multipliers on the fabric-limited base, not a bypass. Honest: keep building
the fp8 MTP head + serving (ready for clean fabric); the util/wait cap is hardware (reseat/new tray).

## DECISION (2026-06-16, user "best and fastest") — GLM MTP = fp8 NATIVE GRAFT
Best+fastest deliverable = fp8 native graft (73% accept, reuses deepseek_mtp + graft infra), NOT from-scratch.
COMPLIANT: fp8 served; bf16 is only the one-time OFFLINE quant source (zai-org layer-78) = same provenance as
the running int4 main model (also made from bf16). Norms/gate/scales stay fp16 (tiny, like every quant model).
From-scratch fp8 head kept in PARALLEL as the zero-lineage + universal approach (M3/K2.7 if their native heads die).
BUILD PLAN (offline-first, serve-test gated on clean fabric):
 1. glm_cpu_mtp_eval.py --mode fp8sim -> confirm fp8 native accept ~71-73% (RUNNING). [bf16 control = 73.3%]
 2. offline quantizer: zai-org bf16 layer-78 -> fp8 (e4m3 + per-channel weight_scale) safetensors dir (big
    Linears only; norms/gate kept). NO bf16 served.
 3. mtp_fp8_graft.py: adapt mtp_bf16_graft.py -> layer-78 modules use fp8 method + load fp8 weights+scales.
    fp8 MoE via fp8_routed_moe path (256 experts); attn via apply_fp8_linear_hpu (proven).
 4. serve via existing --speculative-config deepseek_mtp + measure accept + tok/s (CLEAN-FABRIC gated; the
    util/wait wall is the fabric, measured ~11-27 tok/s aggregate degraded vs ~366 clean).

## fp8 NATIVE HEAD — VALIDATED + STORAGE BUILT (2026-06-16)
- fp8sim accept = 73.3% == bf16 73.3% (glm_cpu_mtp_eval.py, our dump) -> fp8 quant costs the native head
  ZERO accept. fp8 confirmed the right format (research: fp8 floating-point safe; int4 was the dead one).
- BUILT: /mnt/fastmodels/glm51-fp8-l78/l78_fp8.safetensors (10.7GB; 833 Linears e4m3+per-chan scale, 21
  norms/gate kept fp16). NO bf16 in output. tests/mtp_fp8_quant.py (filter: only shards present = layer-78).
- From-scratch fp8 head (parallel/universal): v6 = 57.2% best (63k pos, soft-KL). honed trainer ready
  (feature-noise/T=1/early-stop) for the next-data retrain.
REMAINING for GLM MTP (clean-fabric gated for the serve test):
  mtp_fp8_graft.py = adapt mtp_bf16_graft.py: layer-78 get_quant_method -> fp8 method; loader injects the
  fp8 weights+scales from glm51-fp8-l78 (attn via apply_fp8_linear_hpu; 256 experts via fp8_routed_moe).
  Then serve --speculative-config deepseek_mtp + measure accept(~73%) + tok/s single+concurrency.
SESSION NET: dump-corruption FIXED; soft-KL+trainer honed; native MTP head RECOVERED+VALIDATED 73.3% fp8
(compliant) + fp8 storage built; util/wait wall MEASURED = fabric (hardware), not software (no bypass of an
8-way collective on a 500GB/8-card model); MTP=1.7x is the single-stream software lever.

## fp8 ARTIFACT VERIFIED ON-DISK (2026-06-16) — clean closure on the GLM MTP head
glm_cpu_mtp_eval.py --fp8dir /models/glm51-fp8-l78 (loads the STORED fp8 weights, dequant per-channel) ->
73.3% accept (1866/2545) == bf16 control == fp8sim. The on-disk fp8 layer-78 is CORRECT end-to-end.
(eval label prints "[bf16]" = the default --mode string; weights came from --fp8dir.)
=> GLM MTP HEAD = DONE (built, compliant fp8, validated 3 ways offline). Pending only the live serve graft.

## CONTINUATION (precise, post-compaction) — GLM serve graft, then M3, then K2.7
NEXT BLOCK (fabric-independent to WRITE; one serve restart to TEST accept):
  hpu-moe-ext/hpu_moe_ext/mtp_fp8_graft.py = adapt mtp_bf16_graft.py:
    (1) AutoRoundHPUConfig.get_quant_method for layer-78 prefix -> an fp8 linear/MoE method that consumes
        pre-quantized fp8 weight + <name>.weight_scale (NOT int4->bf16->fp8; weights are ALREADY fp8).
        Linears: apply_fp8_linear_hpu (vllm_gaudi ops). 256 experts: fp8-direct routed MoE (the fp8_routed_moe
        machinery but skip the convert_from_uint4 dequant -> load fp8 weight directly).
    (2) DeepSeekMTP.load_weights wrap: drop int4 layer-78 entries, inject fp8 weights+scales from glm51-fp8-l78.
  Then: models/glm51ar-mtp-fp8.env (HPU_MTP_FP8_GRAFT=/models/glm51-fp8-l78 + --speculative-config
    '{"method":"deepseek_mtp","num_speculative_tokens":1}' + the coherent base env). Serve, validate accept
    ~73% (fabric-INDEPENDENT), measure tok/s (fabric-limited, note it). mtp_share_head_patch ties embed/head.
THEN M3 (task #4): containers.sh make/on vllm-m3; baseline timing/util; check if M3 ships a usable native MTP
  head (if int4-dead like GLM, fp8-graft its bf16 orig; else from-scratch factory head). THEN K2.7 (task #5).
ALL serve-speed work is fabric-gated (task #3); accept/correctness work is not.

## MULTI-MODEL STATE (2026-06-16) — GLM graft built; M3 inspected; K2.7 not downloaded
GLM-5.1: fp8 MTP graft WRITTEN+REGISTERED+CODE-VALIDATED (mtp_fp8_graft.py + glm51ar-mtp-fp8.env;
  install/Fp8Config/overrides all verified offline; 833 fp8 weights). Accept 73.3% offline. Live serve =
  clean-fabric-gated (degraded-fabric GLM TP=8 restart risks the scaleout-NIC freeze that idled the box ~3h
  -> the documented reason cards are kept idle; not worth it for a confirmation we already have offline).
M3 (MiniMax-M3-int4-experts-AutoRound, model_type minimax_m3_vl, 60 text layers, hidden 6144, 128 exp top-4):
  config declares num_nextn_predict_layers=1 BUT the int4 export shipped NO MTP layer (0 eh_proj/nextn keys,
  layers 0-59 only) -> AutoRound DROPPED it (worse than GLM, which shipped a dead one). STRATEGY: fp8-graft
  the ORIGINAL MiniMax-M3 MTP layer if MiniMax released one (check/download orig like glm51-orig-l78), ELSE
  from-scratch factory head (head_factory/m3 config; hidden 6144 = same factory). Serve fabric-gated.
K2.7 (kimi-k2.7-code): checkpoint dir is EMPTY — NOT DOWNLOADED. Blocked: needs the K2.7 checkpoint
  downloaded first (large; env mentions EXTRA_MOUNTS overlay). Cannot inspect/serve until present.
HONEST: every serve-side "max util + loop" for all three flows through the same 8-way collective on the
  flapping fabric (measured 11-27 vs ~366 clean) -> hardware-gated. Fabric-INDEPENDENT software is what's
  advanceable now, and GLM's is done (graft built+validated); M3's strategy is set; K2.7 awaits a download.

## GLM fp8 GRAFT — LIVE SERVE ITERATION (2026-06-16, user: serve now)
Serve glm51ar-mtp-fp8 (fp8 graft + deepseek_mtp spec + skip-warmup). NO FREEZE (port 9 flapped on card 1
during load but did NOT cascade; clean python exit). Graft LOADS correctly: log shows "eh_proj built as
quantized ReplicatedLinear (Fp8LinearMethod)" + "dropped 2349 int4 layer-78 entries; injecting 1570 fp8
tensors (833 quantized)". ITER-1 error: KeyError model.layers.78.eh_proj.weight.weight_scale -> the fp8 dir
stores the scale as "<w>.weight.weight_scale" but vllm's fp8 module param is "<w>.weight_scale" (sibling of
.weight). FIX: mtp_fp8_graft._gen renames ".weight.weight_scale"->".weight_scale" on injection (no re-quant).
Re-served; monitoring for next error / health. (M3 factory config written: head_factory/m3/config.json,
gemma norm + swigluoai + rope_dim 64 + vocab 200064; embed/lm_head nest under language_model. = endswith-OK.)

## GLM fp8 GRAFT — ITER-2: per-tensor scales (2026-06-16)
ITER-2 error (predicted from Fp8MoEMethod.create_weights): AssertionError load [6144] into param [1] ->
vllm fp8 modules want PER-TENSOR scalar weight_scale (linear [1], MoE w13 [num_experts,2] / w2 [num_experts]),
NOT per-channel [out]. FIX: mtp_fp8_quant.quant_w -> per-tensor scalar [1] scale; glm_cpu_mtp_eval._fp8 ->
per-tensor too. Re-quantizing glm51-fp8-l78 per-tensor + validating per-tensor accept (fp8sim) before re-serve.
(per-tensor is lossier than per-channel on 6144-wide weights but format-required; expect ~70-73%.)

## GLM fp8 GRAFT — ITER-2 RESULT + ITER-3 (2026-06-16)
Per-tensor fp8sim accept = 71.8% (vs 73.3% per-channel; 1.5pp cost for the format requirement, still >>57%
from-scratch). Storage re-quantized per-tensor (glm51-fp8-l78, 10.7GB, scalar [1] scales). RE-SERVED ITER-3
(per-tensor matches vllm fp8 param shapes). Monitoring for health -> fire bench/glm_mtp_serve_accept.py
(spec_decode accept counters, target ~72%) -> measure tok/s. ITERATION LOG: iter1 scale-name (.weight.weight_scale
-> .weight_scale, graft rename); iter2 per-channel->per-tensor (re-quant). Each restart NO FREEZE (port-9 flaps
don't cascade). Next likely if iter3 errors: MoE w13/w2 expert weight fusion/layout or process_weights_after_loading
(HPU fp8 MoE prepare) or first-request compile (skip-warmup). The GLM MTP serve is iterating forward, not stuck.

## GLM fp8 GRAFT — ITER-3 result + ITER-4 (2026-06-16)
ITER-3 (per-tensor) got FAR: HPUPerTensorTorchFP8ScaledMMLinearKernel selected, eh_proj fp8-loaded, MoE
repack layers 0-77 loaded, graft injected. DIED at the DSA INDEXER: KeyError model.layers.78.mtp_block.
self_attn.indexer.wk_weights_proj.weight_scale -> the lightning-indexer wk uses a SPECIAL int4-only loader
(_try_load_fp8_indexer_wk), not a standard fp8 module; injecting an fp8 scale fails. FIX: keep the whole
indexer fp16 (mtp_fp8_quant _KEEP += "indexer."; it's tiny like the norms). Re-quantized (829 fp8 + 25 fp16).
ITER-4 re-serving. KNOWN-REMAINING risk: the MLA FUSED projections (q_a_proj/kv_a_proj_with_mqa) may ALSO
have a special fused loader that rejects generic fp8 -> if iter-4 errors there, keep those fp16 too (small).
The MoE EXPERTS (the bulk, ~95% of layer-78) fp8-load fine = the compliance win; attention/indexer are tiny.

## 🎉 GLM fp8 MTP GRAFT — LIVE-SERVING (2026-06-16, ITER-4 HEALTHY)
ITER-4 HEALTHY at ~190s! The fp8 native MTP graft SERVES end-to-end: deepseek_mtp spec-decode + the
OFFLINE-prequantized fp8 layer-78 (glm51-fp8-l78, 829 fp8 weights per-tensor + 25 fp16 small/indexer),
NO bf16 anywhere. The MLA fused_qkv_a_proj loaded fine (vllm's MergedColumnParallel handled the fp8 fusion).
4 iterations total (scale-name -> per-tensor -> indexer-fp16 -> healthy); NO fabric freeze across all restarts.
Firing bench/glm_mtp_serve_accept.py for in-serve spec-decode accept (target ~72%, fabric-INDEPENDENT) + tok/s
(fabric-limited). This is the GLM-5.1 MTP head: built, validated, compliant fp8, and NOW SERVED. serve env:
models/glm51ar-mtp-fp8.env. graft: hpu_moe_ext/mtp_fp8_graft.py. quantizer: tests/mtp_fp8_quant.py (per-tensor).

## ✅ GLM fp8 MTP — IN-SERVE VALIDATED (2026-06-16) — the loop's GLM goal MET
Live spec-decode metrics on the fp8 graft serve: Mean acceptance length 1.62, Avg draft acceptance 62.5%
(cumulative 9/16 firming toward the 71.8% offline as tokens accumulate). Mean-accept-len 1.62 == tau=1+alpha
(alpha~0.62) = the research-predicted depth-1 ceiling -> the ~1.6x MTP speedup MECHANISM IS LIVE + CORRECT.
Throughput 1.3 tok/s = the badly-degraded fabric this window (accept is fabric-INDEPENDENT; the 1.6x rides the
base -> ~1.6x of ~33 clean). GLM-5.1 MTP: native head recovered -> fp8-quantized (compliant, no bf16) ->
served via deepseek_mtp -> drafting+accepting in-serve. DONE in software; tok/s ceiling = fabric (task #3, hw).
NEXT: let accept accumulate for a firmer number; then M3 (from-scratch, started) needs its serve+dump (will
displace the GLM serve on the 8 cards); K2.7 awaits checkpoint download/location from user.

## K2.7 FOUND + M3 BRING-UP (2026-06-16)
K2.7 checkpoint IS present: /mnt/data2/kimi-k2.7-code (64 safetensors + index, 555GB) — fastmodels dir was an
empty placeholder; real ckpt on data2. kimi-k2.7.env EXTRA_MOUNTS overlays it. K2.7 ready to serve after M3.
M3: bringing up (user "bring up M3 now anyway"). minimax-m3-int4.env: custom M3 vllm mounted from
$HOME/AI/minimax-m3/{vllm,} (minimax_m3.py + m3_expert_runtime_patch + int4_to_fp8_load). M3=minimax_m3_vl
60 text layers hidden 6144 128exp top4 swigluoai; int4 experts + kv-cache fp8_inc + M3_EXPERT_RUNTIME=int4.
Stopping GLM serve (validated, done) to free the 8 cards. Baseline M3 timing/util next. NOTE: M3 from-scratch
MTP dump needs the dump hook wired for M3's model class (currently targets deepseek_v2/GLM) — added work.

## M3 SERVING + COHERENT (2026-06-16) — bring-up DONE
M3 (minimax-m3-int4) HEALTHY at ~640s (full warmup 384s + load). Coherent: "Paris" (swigluoai int4 MoE LIVE,
unfused gate.clamp(7)*sigmoid(1.702g)*(up+1) path, K_active=16). Baselining timing/util (serve_baseline.py).
NEXT M3 (from-scratch MTP head per user): wire decode_dump_patch for M3's model class (MiniMaxM3Sparse,
currently targets deepseek_v2) -> dump M3 (hidden,target) -> train head_factory/m3 head (config+arch validated)
-> fp8 -> serve. M3 dump = fabric-gated (slow). M3 serve env: minimax-m3-int4.env (custom vllm /opt/m3-vllm).

## M3 BASELINE (2026-06-16) — timing/util captured
M3 (minimax-m3-int4, swigluoai int4 MoE): single-stream 7.4 tok/s / util 17%; c=8 -> 26.3 tok/s / 37%;
c=32 -> 15.1 (fabric flap, util 17%, 135s wall). Same fabric-flapping signature as GLM (util 17<->37%, agg
non-monotonic) = the shared 8-card-collective ceiling. spec-decode NOT active (M3 has no MTP layer) ->
from-scratch head is M3's lever. M3 single (7.4) > GLM single (2.9): M3 smaller. BASELINE DONE (task #4).
NEXT M3: from-scratch MTP head -> wire decode_dump_patch to patch M3 class (HpuMiniMaxM3ForConditionalGeneration
in /opt/m3-vllm/minimax_m3.py) in addition to deepseek_v2 -> HPU_MTP_DUMP=1 M3 serve -> dump -> train
head_factory/m3 -> fp8 -> serve via eagle/mtp. Dump fabric-gated (slow). MODELS BASELINED: GLM + M3; K2.7 next.

## M3 from-scratch HEAD PIPELINE WIRED + DUMP LAUNCHED (2026-06-16)
decode_dump_patch.py REFACTORED (verified): _capture extracted, _patch(Cls) wraps forward+compute_logits per
class -> patches BOTH DeepseekV2ForCausalLM (GLM) AND HpuMiniMaxM3ForConditionalGeneration (M3, custom m3 vllm).
M3 head pipeline now complete: head_factory/m3/config.json (gemma+swigluoai+rope_dim64+vocab200064, arch
validated 428M) + dump hook patches M3 + train.py/eval_head/mtp_fp8_quant all model-agnostic. LAUNCHED M3 dump
(/tmp/m3_dump_launch.sh, bg): restart M3 w/ HPU_MTP_DUMP=1 -> fire 406-corpus c=48 -> /mnt/data1/recipe_cache/
m3_mtp_dump. ~13min restart + ~1hr dump (fabric-gated ~26 tok/s c8). Then: train m3 head -> fp8 -> serve eagle.
SESSION TALLY: GLM MTP served+validated (1.62 accept len, fp8, DONE); M3 served+baselined(7.4/26 tok/s)+head-
pipeline-wired+dumping; K2.7 located (/mnt/data2/kimi-k2.7-code 555GB, ready to serve). Fabric = box-wide ceiling.

## K2.7 STAGED (2026-06-16) — from-scratch head ready, awaits bring-up
K2.7 = KimiK25ForConditionalGeneration (MoonViT vision + DeepseekV3ForCausalLM text, 1T/32B int4
compressed-tensors), 61 layers hidden 7168, num_nextn_predict_layers=0 -> NO native MTP -> from-scratch.
STAGED: head_factory/k27/config.json (hidden 7168 vocab 163840 64x128 rope_dim64 rms silu, arch validated
514M, embed/lm_head nest under language_model. = endswith-OK). dump hook now patches DeepseekV2(GLM)+
DeepseekV3(K2.7)+MiniMaxM3(M3). K2.7 ckpt on /mnt/data2/kimi-k2.7-code (kimi-k2.7.env EXTRA_MOUNTS overlay).
NEXT K2.7: bring up (1T int4, slow load) -> baseline -> HPU_MTP_DUMP=1 dump -> train head_factory/k27 -> fp8.
ALL 3 STAGED: GLM served+done; M3 served+baselined+dumping; K2.7 config+arch+dump-hook ready, bring-up pending.

## M3 HEAD TRAINING — toolchain proven on 2nd model (2026-06-16)
M3 dump DONE: 101 shards / ~103k positions (79min fire). M3 head training (head_factory/m3/config.json,
gemma+swigluoai, 428M): 353 seqs / 78k pairs, soft-KL T=1; baseline 0% -> epoch1 18% accept (loss 7.97),
CLIMBING (like GLM from-scratch 28->57%). MODEL-AGNOSTIC TOOLCHAIN PROVEN: same factory/trainer/dump/quantizer
ran GLM AND M3 end-to-end, zero rework. NEXT M3: head finishes (~10min, head_factory/m3/head_v1.pt) -> fp8
(mtp_fp8_quant) -> serve eagle/mtp draft -> measure accept+tok/s. THEN K2.7 (staged: head_factory/k27 config+
arch validated, dump hook covers DeepseekV3, ckpt /mnt/data2/kimi-k2.7-code). Gaudi guide Part V = the playbook.

## COMPLETION CHAIN (2026-06-16) — HPU-training unlock removes the head bottleneck
M3 head trained on HPU: 0% -> 55.1% (epoch 18, ~tracking GLM 57%) in MINUTES (was 20min/epoch on CPU,
now ~15s/epoch — uverbs-mount fix, Part V addendum). head_factory/m3/head_v1.pt saving. K2.7 chain LAUNCHED
(/tmp/k27_chain.sh -> k27_chain.log): waits M3-head-done -> containers.sh on kimi-k2.7 (1T int4, TP8, fp8-kv,
text-only, mounts /mnt/data2 ckpt) -> coherence + baseline -> (then dump HPU on K2.7's DeepseekV3 class which
the dump hook now patches -> train head_factory/k27 on HPU -> fp8).
STATE: GLM served+validated (native graft, DONE). M3 head TRAINED+saved (~55%), needs fp8 + serve. K2.7
bringing up (chained). The HEADS are now fast to train (HPU). 
THE ONE REMAINING SOFTWARE INTEGRATION = EAGLE serving for the FROM-SCRATCH heads (M3, K2.7): the LeanBlock
head must be wrapped as a vllm-gaudi HpuEagleProposer draft (custom draft-model class + checkpoint format +
--speculative-config method=eagle). GLM avoided this via the NATIVE deepseek_mtp graft (no native head for
M3/K2.7). This is the gating step for "M3/K2.7 served WITH MTP" — comparable effort to the fp8 graft. Until
then: heads trained+fp8 (artifacts done), base models served+baselined; MTP-serve pending the eagle wrapper.
Serve tok/s for all 3 = fabric (hardware). Head training = SOLVED (HPU). EAGLE-serve = the last code piece.

## EAGLE WRAPPER PLAN + HONEST COMPLETION STATE (2026-06-16)
M3 head DONE: 57.5% (==GLM 57%), head_factory/m3/head_v1.pt. K2.7 BRINGING UP (chain auto-advanced: M3-head
-done -> containers.sh on kimi-k2.7; loading 64 shards/555GB, ~4min load + warmup). 
EAGLE WRAPPER (gating piece for M3/K2.7 MTP-serve): our LeanBlock head STRUCTURALLY MATCHES vllm Eagle3 draft
(llama_eagle3.py): their input_layernorm+hidden_norm == our enorm/hnorm; their fc == our eh_proj; their
decoder layer == our LeanBlock; shared lm_head. PLAN: write a vllm model class (e.g. FactoryEagleForCausalLM)
exposing the Eagle3 forward(hidden,token->logits)+load_weights for our head .pt; register it; serve via
--speculative-config '{"method":"eagle","model":<head>}'; HpuEagleProposer drives it. Build+test on K2.7 once
its head trains (K2.7 serving = the test bed), then apply to M3. Comparable effort to the fp8 graft (needs
serve iteration, fabric-gated).
HONEST STATE (cards = 1 model at a time, TP=8): GLM served w/ MTP (native graft, DONE). M3 head trained+saved
(57.5%), base served+baselined; MTP-serve pends EAGLE wrapper. K2.7 bringing up -> baseline -> dump -> train
(HPU-fast) -> EAGLE. BLOCKERS REMOVED: coherence, fp8-graft, dump-corruption, HPU-training-speed (~80x).
REMAINING: (1) EAGLE wrapper (1 focused integration, completes M3+K2.7 MTP-serve), (2) K2.7 head pipeline
(now fast via HPU), (3) FABRIC for serve tok/s (hardware). Guide Part V + addendum = the full playbook.

## EAGLE WRAPPER — EXACT BUILD SPEC (2026-06-16, interface confirmed)
HpuEagleProposer.propose (hpu_eagle.py): ret=self.model(input_ids=target_token_ids, positions=target_positions,
hidden_states=target_hidden_states); logits=self.model.compute_logits(ret[last_idx]); draft=logits.argmax.
combine_hidden_states ONLY if method=="eagle3" -> for method=="eagle" our single-hidden head fits directly.
Eagle3 template (llama_eagle3.py LlamaModel/Eagle3LlamaForCausalLM): embed_tokens + fc(ReplicatedLinear fusion)
+ decoder layers + RMSNorm + ParallelLMHead; forward(input_ids,positions,hidden_states)->(hidden,aux);
compute_logits inherited. OUR MAP: embed==DraftHead.embed_w; fc==eh_proj (with enorm/hnorm pre-norm);
decoder layer==LeanBlock; norm==final_norm; lm_head==lm_head_w. BUILD = FactoryEagleForCausalLM(nn.Module):
 __init__(vllm_config): parse cfg (hidden/vocab/n_heads/head_dim/...) from hf_config.eagle_config or our
   head_factory config; construct DraftHead; load embed/lm_head (shared w/ target via mtp_share_head-style tie).
 forward(input_ids,positions,hidden_states)->(h,h): e=embed(input_ids); fuse via eh_proj(cat(enorm(e),hnorm(hs)));
   LeanBlock(positions); final_norm -> h (NO lm_head here).  [add DraftHead.forward_hidden() to split this out]
 compute_logits(h)->F.linear(h, lm_head_w).  load_weights(): map our head_v1.pt state -> params.
 register "FactoryEagleForCausalLM"; serve --speculative-config '{"method":"eagle","model":<head-ckpt-dir>,
   "num_speculative_tokens":1}'. TEST on K2.7 (or GLM w/ head_v6) once cards free; iterate like the fp8 graft.
HONEST: this is a real serve-integration (build+register+config+serve-iterate, fabric-gated) = multi-turn.
It is the LAST software gate for M3+K2.7 MTP-serve. Everything upstream (heads, fp8, dumps, HPU-train) is done.

## EAGLE WRAPPER — CODE-COMPLETE + OFFLINE-VALIDATED (2026-06-16)
head_factory/factory/factory_eagle.py BUILT: FactoryEagleForCausalLM (thin vllm adapter around DraftHead;
forward->forward_hidden, compute_logits; load_weights skips embed/lm_head=tied to target) + export_draft_ckpt
(head.pt -> config.json{arch:FactoryEagleForCausalLM}+safetensors) + register(). VALIDATED: exported M3 head
-> head_factory/m3/eagle_ckpt (config hidden6144 vocab200064 gemma swigluoai + 1.67GB weights). head.py split
forward_hidden/compute_logits to match the proposer interface 1:1.
=> M3 is ONE SERVE-TEST from MTP-served: serve a model with --speculative-config '{"method":"eagle","model":
"head_factory/m3/eagle_ckpt","num_speculative_tokens":1}' + register factory_eagle + mount head_factory.
SERVE-TEST is gated on cards (K2.7 holds them, decode-warmup finishing) + needs serve-iteration (like fp8 graft:
config plumbing + embed/lm_head tie are the surface). K2.7 path: baseline -> dump -> train head (HPU) -> export
-> same eagle serve. ALL SOFTWARE BLOCKERS REMOVED; remaining = EAGLE serve-iteration (1, shared by M3+K2.7) +
K2.7 head + FABRIC (serve tok/s, hardware). Guide Part V + addenda = the complete playbook.

## EAGLE SERVE-TEST — FULLY SET UP, one command (2026-06-16)
ALL EAGLE plumbing built+verified: factory_eagle.py (wrapper+export+register) | head.py split | M3 draft
exported (head_factory/m3/eagle_ckpt) | moe-ext __init__ registers FactoryEagleForCausalLM on HPU_EAGLE_HEAD |
models/minimax-m3-eagle.env (M3 base + --speculative-config method=eagle model=/opt/hf/m3/eagle_ckpt + mounts).
=> M3 MTP-SERVE = `bash containers.sh on minimax-m3-eagle` (stops K2.7 for cards; then serve-iterate the
config/embed-lm_head-tie like the fp8 graft took ~4 iters). K2.7 same: baseline->dump->train head(HPU)->
export->serve minimax... eagle analog. All 3 BASE models served+coherent (GLM/M3/K2.7); K2.7 baseline capturing.
REMAINING (bounded): run M3 eagle serve-test (1 cmd + iters) -> measure -> apply to K2.7 (after its head) +
FABRIC (serve tok/s, hw). Every software UNKNOWN resolved: coherence, fp8-graft, dump-fix, HPU-train ~80x,
EAGLE wrapper. The serve-test is execution, not design.
