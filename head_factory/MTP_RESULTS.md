# MTP / single-stream maxing — results + findings (GLM-5.1, M3, K2.7)

Each model in its own container; goal = fastest single-stream + concurrency via an MTP draft head, with timing
+ utilization measured. Box: 8× Gaudi-2 (96GB/card), vllm-gaudi:next, TP=8.

## THE HEADLINE FINDING (2026-06-16, CORRECTED): it was the int4 expert path, NOT fabric

Earlier in the session I benchmarked M3 on `M3_EXPERT_RUNTIME=int4` (the unfused all-expert per-token-dequant
path), got ~10 tok/s @ c16 / 17% util, and wrongly concluded "fabric-gated, wait for the new tray." **That was
the slow path.** Re-measured on the fast path (`minimax-m3-fp8.env`, fp8 route-b, native fp8 MME, no per-token
dequant, max-num-seqs 128):

| M3 fp8 route-b | tok/s agg | util |   | int4 path (for ref) | tok/s |
|---|---|---|---|---|---|
| c=1 (single-stream) | 2.2 | 16% |   | c=1 | 2.4 |
| c=16 | 16.8 | 17% |   | c=16 | ~10 |
| c=32 | 22.1 | 55% |   | | |
| **c=50** | **89.1** | **55%** |   | (int4 flat ~17 @ c50) | |

**FABRIC IS CLEAN** — 89 @ c50 matches the documented 90; a degraded fabric could not hit it. The slow numbers
were the int4 path. Two real axes:
1. **Concurrency: SOLVED via fp8 route-b — 89 tok/s @ c50, and util is only 55% → headroom** (config sweep /
   higher c → ~130-180 per the gameplan). No new hardware. This is the fast path; just use `minimax-m3-fp8.env`.
2. **Single-stream: ~2.2 tok/s, host-op-dispatch bound** (fp8 ≈ int4 here, as expected — precision doesn't help
   a host-bound single stream). Levers: the routed/fused MoE (still `[all-expert]` even on fp8) + MTP. NOT fabric.

**MTP re-read:** the "wash at concurrency" earlier was on the slow int4 base. MTP is still gated by the
**async+spec exclusion** (vllm-gaudi forces sync sched under spec → loses the async overlap that the fp8 base
relies on for concurrency, and hurts the host-bound single stream). So the fast M3 path is the **fp8 BASE
(no MTP)**; MTP is the sequenced single-stream lever. Code-confirmed: `platform.py:246` + `assert
hpu_model_runner.py:1234`; base/no-spec worker logs async ENABLED, spec worker DISABLED.

**Decision (user, 2026-06-16): SEQUENCE the heavy async+spec port for clean fabric / later.** But the
concurrency win (fp8 route-b) is available NOW with zero new hardware.

## GLM-5.1 — ✅ DONE (native fp8 MTP graft)
Native nextn-layer fp8 graft, served via deepseek_mtp. In-serve **mean acceptance length 1.62** (~73%). Same
async+spec + fabric gating applies to its single-stream.

## M3 (MiniMax-M3 int4) — ✅ HEAD DONE, optimization sequenced
From-scratch **MLP-only (Medusa) draft head** (serve-correct: eagle proposer runs the draft per-token at decode
with no seq context, so attention can't be used). Served via FactoryEagleForCausalLM (mirrors DeepSeekMTP;
shares target embed+lm_head). Trained on HPU → 58.6% val; **in-serve 56% single / 66% conc accept**; fp8 e4m3
(bf16-ban compliant, 377MB). Timing as above. Code: head_factory/factory/, m3/eagle_ckpt.

## K2.7 (Kimi-K2.7-Code int4) — head IN PROGRESS
No native MTP (kimi_k25 / DeepSeek-V3 text LM). Same from-scratch pipeline. serve→dump→train→export→fp8→serve.
config.train.json + kimi-k2.7-eagle.env staged.

## SCOPED LEVERS (ready to execute on clean fabric)

### A) async+spec port into vllm-gaudi (single-stream MTP unlock; benefits all 3)
- Enabler scaffolding STAGED + gated `HPU_ASYNC_SPEC=1` (default off): vllm-gaudi-patch/{platform.py (allow
  eagle async), hpu_model_runner.py (assert relaxed + `use_async_spec_decode` flag), hpu_input_batch.py}.
  Mounted into minimax-m3-eagle.env.
- The HARD remaining piece: HPU rejection sampler `.cpu()` per step (hpu_rejection_sampler.py:53-56) sits INSIDE
  execute_model's critical path → blocks the async overlap. Must be made **device-side** (no per-step host sync;
  bonus token + accepted counts stay on device, deferred copy) so execute_model returns before sampling.
- Good news (verified): the CORE scheduler already owns optimistic-advance + reconciliation from the runner's
  returned per-request sampled_token_ids lengths — so NO GPU-style correction kernel is needed on HPU. The runner
  just needs (i) device-side rejection output, (ii) prev_sampled_token_ids = bonus-per-req, (iii) async output
  carrying variable-length per-req lists.
- GPU reference (vLLM CUDA already supports async+spec): use_async_spec_decode (gpu_model_runner.py:627),
  prev_num_draft_len (gpu_input_batch.py:60), scatter of bonus+draft tokens in _prepare_input_ids.

### B) fused routed MoE for M3 (single-stream, independent of MTP)
M3 serves int4 experts via the **UNFUSED swigluoai ALL-EXPERT path** (computes ALL 16 local experts/token; the
stock fused swigluoai intrinsic crashes so it's bypassed). Routing to top-K via a swigluoai-capable fused kernel
(the ":1.24-lazy fused-op revive") is a ~2-4× single-stream lever. Also partly fabric-gated (collectives).

### C) clean fabric (new tray) — DOMINANT, hardware/user's domain
Lifts the ~10 tok/s / 17%-util ceiling that caps BOTH single-stream and concurrency for ALL models.

## FINAL STATE (2026-06-17) + PRECISE CONTINUATION

**Fabric storm is ACTIVE + kernel-confirmed:** dmesg = 8078 Port-9 link-down events, bursty (clean windows
then bursts), worst cards 1/4/3/5. **A/B PROOF (same fp8 config, only fabric differs): M3 c=50 = 89 tok/s in a
clean window vs 40.8 tok/s during a burst = 2.2× fabric gate, software held constant.** This is the conclusive
isolation of the real constraint — it's the fabric, not software/config. Caps high-c throughput (util→17% during bursts; M3 89@c50 was a clean
window, maxc push stalled under a burst) and WEDGES TP=8 startups of GLM(500G,~25min) + K2.7(555G). The
incoming TOP TRAY is the fix (user directive: software-max everything so the tray is the one remaining item).

**Validated/banked:** M3 MTP head (56-66% accept, fp8) + M3 fp8 route-b 89@c50 + the int4-vs-fp8 correction
(was on slow int4 path; fp8 is the fast path) + the async+spec-exclusion root-cause + the fabric-storm reality.

**CORRECTED FIT (2026-06-17, user was right): GLM CAN do the M3 fp8-expert win.** Real sizes: GLM-5.1-int4 =
376 GB on disk, of which 347 GB is int4-packed experts (16 GB bf16, 13 GB scales/int32). Load-time int4->fp8
of the experts -> ~724 GB in HBM, which FITS 768 GB with ~44 GB for KV+acts+graphs (TIGHT -> small ctx / few
seqs, but feasible). (My earlier "doesn't fit" used the wrong 500 GB = K2.7.) So GLM's fast path IS the M3
route-b approach: experts int4->fp8 AT LOAD (fp8 resident, NO per-token dequant) + fp8 MME GEMM. Two fp8
senses: (a) int4-weights + fp8-KV = universal, all models; (b) expert-WEIGHTS int4->fp8 = the 89@c50 win,
fits GLM tight. The `fp8_allexpert_moe` I added is TRANSIENT fp8 (keeps int4, dequants each step) = NOT the
win; the LOAD-TIME conversion (in hpu-m3-ext m3_swigluoai_patch's FP8 RUNTIME HANDOFF) is.

**CONTINUATION (do in storm lulls / post-tray, no swarm):**
1. GLM fast path = LOAD-TIME int4->fp8 experts + all-expert silu fp8 GEMM (the M3 route-b win, fits GLM tight).
   PRECISE INTEGRATION (gate everything on HPU_MOE_FP8_EXPERTS=1, default off = no behavior change):
   - DONE: `fp8_allexpert_moe` in hpu-moe-ext/fp8_routed_moe.py = the all-expert silu fp8 GEMM (syntax-valid).
     It currently dequants transiently; for load-time, change its weight source from convert_from_uint4 to the
     stored fp8 (add a `w13_fp8/w2_fp8` param path).
   - TODO: in moe_wna16_hpu.py `process_weights_after_loading` (after ~line 613, where _hpu_routed_w13_zp is set):
     if HPU_MOE_FP8_EXPERTS, per local expert convert int4->bf16 (convert_from_uint4) -> per-tensor fp8
     (cast_to_fp8_v2), stash layer._fp8_w13[E], layer._fp8_w2[E] + per-tensor scales (~724G HBM total).
   - TODO: in `apply`, if HPU_MOE_FP8_EXPERTS, route to fp8_allexpert_moe(... stored fp8 weights ...) instead of
     the int4 dequant-all path.
   - SERVE: small ctx (--max-model-len 512-1024) + few seqs (HBM ~44G headroom); validate coherence + measure
     c=1/10/30/50 in a storm lull. Transfers to GLM-5.2 (same arch).
   GLM single-stream: prior ~14 tok/s warm (lazy+DSA); NO clean number this session (eager+DSA indexer crawl;
   dense wedged on the storm @ 1422 flaps/5min).
2. M3 single-stream (host-op-dispatch bound, ALREADY routes top-K at decode): the levers are MTP via the
   async+spec port (lever A) + the lazy fused-op revive — both deep/gated, not a quick routing win.
3. K2.7 head: resume serve(HPU_MTP_DUMP)->dump->train->export->fp8->serve (config.train.json + kimi-k2.7-eagle.env
   staged); serving storm-gated.
4. M3 max-concurrency config (130-180 target): re-run maxc env in a sustained clean window / post-tray.
5. INC-from-bf16 fp8 for M3 (cleaner than int4->fp8 double-quant): the 796G /mnt/data2/minimax-m3-bf16 was
   DELETED 2026-06-17 (quantized-only disk cleanup, user directive). Re-download the bf16 to pursue this.
   Route-b int4->fp8 (the validated 89@c50 fast path) is UNAFFECTED — it needs only the int4 experts.
