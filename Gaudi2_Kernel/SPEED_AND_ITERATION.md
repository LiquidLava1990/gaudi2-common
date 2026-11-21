# Gaudi 2 Kernel — single-stream speed + fast-iteration runbook

Synthesis of a 3-agent deliberation (internet Gaudi-2 tricks + code-truth bottleneck analysis +
proven-config extraction), 2026-06-07. Applies to the int4-MoE models (GLM-5.1, Qwen-397B) on
8× Gaudi 2 / vllm-gaudi 1.24.

## The honest ceiling (read first)
Single-stream decode on Gaudi 2 is **host-/sync-bound and weight-bandwidth-bound, not FLOP-bound.**
A 40B-active int4 MoE + MLA still streams the routed-expert weights per token across 8 cards plus a
per-layer all-reduce. Measured reality (`gaudi-perf-targets`): **~29–34 tok/s is the single-stream
ceiling; ~75 is structurally unreachable.** The knobs below remove recompiles, host bubbles, and graph
fragmentation — they close the gap *to* the floor; they do not move the floor. Anything claiming a huge
single-stream multiplier is wrong.

---

## ⬛ MEASURED 2026-06-07 — GLM single-stream + the LAZY-MODE BOUNDARY (read before tuning)
- **Baseline (eager, `:1.24`, no routed-K):** 2.2 tok/s (mean TPOT 451 ms) — compile-contaminated.
- **routed-K=8 in EAGER:** 1.6 tok/s — **SLOWER.** In eager (`PT_HPU_LAZY_MODE=0`) decode is
  host-dispatch-bound, so `_routed_moe`'s extra ops (index_select+gather+manual dequant) cost more than
  the single fused all-expert intrinsic. **Routed-K only pays off under LAZY mode** (graph replay → free
  dispatch → dequant bandwidth is the bottleneck). Do NOT enable routed-K in eager.
- **routed-K correctness:** verified — half the ranks maxdiff=0 (exact), the rest 0.05-0.17 (bf16
  fused-vs-manual matmul difference, benign); output stays coherent. The 1e-2 verify threshold is too
  strict for bf16; the path is fine. (Also fixed a verify bug: `ref` must use the LOCAL-remapped ids,
  else it computes the broken global-into-local-window path and falsely rejects `_routed_moe`.)
- **🚧 LAZY MODE IS BLOCKED FOR GLM (the hard boundary):** lazy needs the `vllm-gaudi:next` image
  (the `:1.24` image runs upstream PyTorch 2.10 which **hard-rejects lazy**: "Upstream PyTorch version
  2.10.0+cpu is not supported in Lazy mode"). But `:next` **OOMs loading GLM** (`PT_DEVMEM 6MB fail` at
  78% shards, even at gpu-mem 0.92 / graph-reserve 0.1) because GLM's **8-bit→bf16 dequant-at-load
  inflates the weights** (the 397B is uniform int4 with no such spike → it fits `:next` fine). So:
  `:1.24` = fits + eager-only; `:next` = lazy-capable + can't hold GLM. **Lazy + routed-K (the ~10×
  single-stream lever) is unreachable for GLM without re-engineering the 8-bit path** to NOT dequant-to-bf16
  at load (on-the-fly dequant, keep weights compact → fit `:next`). That's a real project; the eager
  ceiling on `:1.24` stands at ~2-5 tok/s. The 397B already runs lazy+routed-K on `:next` (~29 tok/s) — so
  the win is proven for that model, just not portable to GLM until the dequant is made memory-compact.
- **The flapping O0–O3 fabric** stalls the per-token all-reduces (0.0-tok/s gaps) on top of all this — a
  clean single-stream number needs the reseat regardless.

## ✅ LAZY MODE SERVES (2026-06-08) — solved via the OFFLINE PRE-REPACK CACHE
Lazy now serves GLM with a full KV cache (Free 42.7GB, **KV 206,592 tokens** @ gpu-mem 0.9). The fix:
mint op-ready int32 MoE weights ONCE in eager (`HPU_MOE_REPACK_SAVE=/mnt/fastmodels/moe_repack_cache`,
~349GB/600 files), then lazy loads them (`HPU_MOE_REPACK_LOAD=/models/moe_repack_cache`) with NO
load-time repack → nothing traps. `create_weights` uses (E,1,1) int32 placeholders so the uint8 fused
originals never allocate; LOAD is mode-aware (`HPU_MOE_REPACK_LAYOUT`: 'stacked' default keeps ONE
[E,...] for routed-K with op=VIEWS = 1x; 'perexpert' = monolithic 1x — GLM MoE is 45GB/card so the eager
2x of [E,...]+per-expert-clones = 90GB OOMs). Mint/serve commands: `hpu-moe-ext/hpu_moe_ext/REPACK_CACHE_README.md`.
**SPEED (FABRIC-CAPPED — 397B hit 29 on a healthier fabric; new baseboard incoming uncaps):** monolithic
all-expert = coherent **1.75 tok/s**; routed-K=8 (fast path) = ~5 tok/s but decode degenerates
(routed-verify `md=100-200` → `_routed_moe` dequant math is wrong, being fixed). Full plan: `SPEED_MASTER_PLAN.md`.
**(Historical enablement trail below — kept for the reasoning; the cache is the actual fix.)**

## 🚧 LAZY-MODE ENABLEMENT history (2026-06-07) — 10 blockers cleared before the cache solved it
Lazy mode (`PT_HPU_LAZY_MODE=1`, the unlock for routed-K + graph replay) needs the `vllm-gaudi:next`
image (`:1.24` runs upstream PT 2.10 which hard-rejects lazy). Getting GLM to LOAD under lazy on
`:next` took clearing TEN distinct blockers (all committed in `hpu-moe-ext/moe_wna16_hpu.py` +
`hpu-dsa-ext/deq_weight_move.py`, gated to lazy / no-op in eager — `:1.24` re-verified coherent):
1. **Memory was a red herring (deferral waste):** my bf16-on-CPU deferral added ~42GB of dup/churn;
   removed it. GLM's real footprint is **52GB** (measured on `:1.24`).
2. **MoE-repack reclaim:** `mark_step` after each expert in `_repack_weight` (pending frees).
3. **Per-layer dequant `mark_step`** at process_weights start.
4. **HPU-native unpack:** overrode `unpack_weight/zeros_from_cuda_old_format` to run the (pure
   integer) unpack ON HPU — under lazy the loaded strided TP-shard params can't be `.cpu()`'d
   ("Strided Params Has Value: 0"); the MoE proves on-HPU access works. Eager unaffected (same int math).
5. Dequant scales/g_idx kept on-device (no `.cpu()`).
6. **MLA-absorption per-layer `mark_step`** (monkeypatch `MLAAttention.process_weights_after_loading`).
7. **Flag-gated `mark_step` in apply** during absorption only (load-time; OFF in inference so it
   doesn't split the decode graph).
8. **Short-circuited `get_and_maybe_dequant_weights`** for GPTQDequantToBF16 layers — the absorption's
   generic O(N³) `torch.eye` extraction OOMs + hits a lazy strided-`.T`; our `layer.weight` IS the
   dequanted bf16 it would return, so return it directly.
9. Post-load deferred-weight move hook (now a no-op after #1; kept harmless).
10. **Result: GLM loads the ENTIRE forward under lazy on `:next` + profiles.**

**🧱 The remaining wall (Habana lazy allocator — 3-agent deliberation, ground-truthed vs the image):**
GLM is really 52GB but the lazy pool sits at ~94GB and KV can't allocate. Root cause + why each fix fails:
- The Habana pool (`CoalescedStringentPooling`) grabs **100% of HBM up front** (`PT_HPU_POOL_MEM_ACQUIRE_PERC=100`),
  so gpu-mem-util only sizes KV, it does NOT cap the pool. `mem_get_info = Limit − InUse`.
- The MoE repack frees a same-byte uint8 original per layer; under lazy that block stays **pinned by a
  stream event** and is NOT reused → InUse never drops → ~42GB trapped.
- **No `empty_cache`/release API exists** on this build (only the internal C++ `reset_pool`/`drop_cache`,
  unbound). `hpu.synchronize()` (the one graceful unpin) **OOMs**: it triggers a defragment that needs
  scratch the full pool can't give (`OOM: No enough memory for defragment`), even per-layer, even at
  `PT_HPU_POOL_MEM_ACQUIRE_PERC=80`.
- The **in-place repack** (reuse the original's storage) loaded + allocated KV + reached READY, but
  **`copy_` to a reinterpreted view is NOT graph-tracked under lazy** → the MoE op got an empty weight
  → crash. A *fresh* repacked tensor is correct but re-traps the freed original.
**⇒ Clean finish = (b) an OFFLINE pre-repacked checkpoint** (MoE w13/w2 converted uint8→int32 on disk so
there is NO load-time repack → nothing to trap). Code state: HPU-native unpack + short-circuit absorption
+ per-layer/per-expert mark_step are committed (lazy-gated, eager re-verified). `models/glm51ar-lazy.env`
loads the full forward but can't size KV. **GLM serves on `:1.24` eager ~2.2 tok/s meanwhile.**
**⚠️ And the lazy single-stream win is FABRIC-CAPPED** — the flapping O0–O3 scale-up fabric stalls the
per-token all-reduces regardless of lazy, so the O0–O3 RESEAT is the higher-leverage single-stream lever.

## THE TWO REAL LEVERS (do these, measure each)

### Lever 1 — MoE routed-gather: `HPU_MOE_ROUTED_K=8`  (≈4× MoE-dequant reduction)
Stock `VllmMixtureOfExpertsOpWNA16` dequants **all 32 local experts/token** (`ops.py:1486`), but
single-stream routes ≤8 distinct experts total. `_routed_moe` (`moe_wna16_hpu.py:255`) `index_select`s and
dequants only K experts. Gate: `tokens*top_k <= K`; single-stream decode = `1*8 = 8`, so **K must be ≥ 8**
(GLM `top_k=8`). Below 8 the gate flips OFF → all-expert regression.
- **Correctness (verified by code audit):** bit-exact vs the all-expert op (identical scale dtype, shared
  sym zero; `routed_scaling_factor=2.5` is baked into `topk_weights` upstream and carried through). Does
  **not** double-remap with the global→local id fix — `_routed_moe` runs on the raw global ids *before*
  the `apply_monolithic` in-window remap (which only runs on the fallback path). GPTQ-sym only (GLM
  AutoRound qualifies).
- **Quality bar:** run `HPU_MOE_ROUTED_VERIFY=1` ONCE first — it runs both paths and asserts maxdiff < 1e-2
  (`[routed-verify] maxdiff=...` in logs), staying coherent on mismatch. After a clean verify, drop it
  (it doubles work) and run `HPU_MOE_ROUTED_K=8` alone.
- Note: only accelerates true single-stream (bs=1). Batched decode (bs>1 → tokens*top_k>8) correctly falls
  through to the all-expert op. For up to bs=4 use K=32 (=local expert count), verified separately.

### Lever 2 — defeat compile contamination (the current 1-2 tok/s is NOT real)
`VLLM_SKIP_WARMUP=true` defers decode-shape compiles to first-token → the throughput sawtooth
(0→5→0 tok/s) + `('decode',1,1,1) was not warmed-up!` warnings are **compile stalls, not compute.**
- The **recipe cache is already wired and persistent** (`serve.sh:45`
  `PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/$MODEL,false,8192`, host `/mnt/data1/recipe_cache`; GLM dir
  already 374M/5374 files). So compiles persist across restarts of the SAME config.
- **For a clean measurement:** keep skip-warmup (cheapest iteration) but **throw away the first request**
  and read steady-state from the 2nd+ — OR set `VLLM_SKIP_WARMUP=false` to warm at load (one-time, mostly
  recipe-cache replay).

---

## LAZY MODE (Lever 3, +~21%, but gate coherence first)
The 397B's proven single-stream config runs **lazy/HPU-graph mode**, which replays the decode graph
instead of rebuilding it per step (removes per-step host overhead) and captures the TP all-reduces inside
the graph. GLM currently runs eager (`PT_HPU_LAZY_MODE=0`).
```
PT_HPU_LAZY_MODE=1  PT_HPU_ENABLE_LAZY_COLLECTIVES=true  IMAGE=vllm-gaudi:next
--gpu-memory-utilization 0.85   VLLM_GRAPH_RESERVED_MEM=0.2
```
- Requires the `:next` image (upstream 2.10 base rejects lazy). Lazy auto-enables lazy collectives;
  collectives MUST be captured in-graph or every layer breaks the graph (host sync).
- TP=8 only — **TP=4 OOMs in lazy graph capture** (why the 397B `sslazy8` went back to TP=8).
- ⚠️ **Untested with the DSA path** (DSA only ever ran under LAZY=0). **Coherence-check output before
  trusting any tok/s.** If lazy+DSA salads, fall back to LAZY=0 + Lever 1 alone (that's the bulk of the win).

---

## The batch to apply to GLM (ported from `qwen35-397b-sslazy8.env`)
Apply in **two measured loads** (clean attribution > raw batching here):

**Load A — Lever 1 (keep LAZY=0, isolate the MoE win), with VERIFY:**
```
MODEL_ENV_EXTRA='-e HPU_MOE_ROUTED_K=8 -e HPU_MOE_ROUTED_VERIFY=1 -e VLLM_SKIP_WARMUP=true'
# measure 2nd+ request; confirm [routed-verify] maxdiff small; then drop VERIFY and re-measure.
```
**Load B — Lever 3 (lazy) on top of Lever 1:**
```diff
+IMAGE=vllm-gaudi:next
 MODEL_ENV="... -e PT_HPU_LAZY_MODE=1 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true ... -e MAX_EXPERTS_PER_SLICE=64 -e VLLM_DECODE_BS_BUCKET_MIN=1 -e HPU_MOE_ROUTED_K=8 -e VLLM_GRAPH_RESERVED_MEM=0.2 ..."
 VLLM_ARGS="--tensor-parallel-size 8 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.85 ..."
```
Keep GLM correctness env: `HPU_DSA_READY=1 HPU_SANITIZE_EMBED=1 HPU_GLM_ROPE_NEOX=1`. NOT `HPU_GLM_NORM_OFFSET`.

**Optional probes (one try each, GLM is non-hybrid so the Qwen mega-graph trap may not apply):**
`VLLM_CONFIG_HIDDEN_LAYERS=2` (more layers/graph), `VLLM_GRAPH_PROMPT_RATIO=0.1` (favor decode graphs),
`VLLM_DELAYED_SAMPLING=true` (hide token-sync bubble — verify it takes effect in 1.24).

---

## Fast-iteration recipe (make each test ≈ weight-load, not weight-load + recompile)
1. **Recipe cache is the foundation** (already on). Per-experiment isolation: override the path via
   `MODEL_ENV_EXTRA='-e PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/glm51ar-K8,false,8192'` (last `-e` wins).
2. **Warm once per config:** after launch send ~3 identical requests of the exact shape you'll measure;
   the buckets compile + persist. Restart 2+ of the same config skips the Graph-Compiler stage.
3. **Collapse to one decode graph** (fewest compiles): `VLLM_DECODE_BS_BUCKET_MIN/MAX/STEP=1`,
   tight `VLLM_DECODE_BLOCK_BUCKET_*` around the real max-model-len.
4. **Measure with the streaming bench, not curl** (curl-timeout-prone):
   ```
   $HOME/.venvs/hf/bin/python $HOME/AI/bench/bench_client.py \
     --base-url http://<SERVER_IP>:8000/v1 --api-key $API_KEY --model glm51ar \
     --concurrency 1 --num-prompts 3 --max-tokens 256
   ```
   Reports `mean_tpot_ms` (time-per-output-token, excludes TTFT). **tok/s = 1000 / mean_tpot_ms.** Send
   ≥2-3 prompts so the cold first request is averaged out. Do NOT use `run_cycle.sh` for iteration (it
   sweeps concurrency + runs the accuracy probe each cycle).
5. **Diagnostics to confirm you're at the floor:** `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=true` (catch
   per-step recompiles), `VLLM_T_COMPILE_FULLGRAPH=true` (prove decode = one graph), `VLLM_PROFILER_ENABLED=true`
   (perfetto trace → host-bound gaps vs dense device ops).

---

## DROP-LIST — do NOT spend a load on these (measured no-help / backfire)
- `HPU_MOE_ROUTED_K < top_k(8)` → gate OFF → all-expert REGRESSION.
- `MAX_EXPERTS_PER_SLICE < 32` → splits the fused MoE op → accumulation overhead (32 local experts; 64 is inert/harmless).
- **Stock FP8 MoE swap** → regression + no fp8 checkpoint on disk.
- **shared-expert-stream** → CUDA-gated, dead on HPU.
- **`--kv-cache-dtype fp8_inc`** → throughput/context lever, ~0.01–0.06% of the single-stream budget. (Use it for concurrency, not single-stream.)
- **TP=4** → OOMs in lazy graph capture; NUMA pin residual already hidden by async-scheduling.
- **FP8 for bandwidth** → invisible at ~8-17% util (BW/sync-bound on dequant, not compute-bound).
- **`VLLM_GDN_COMPUTE_FP32`** → GatedDeltaNet knob; GLM has none → no-op.
- **`HPU_GLM_NORM_OFFSET`** → the +1 was a misdiagnosis → salad. Never set.
- **`--quantization` flag for GLM** → AutoRound is plugin-autodetected; adding a flag breaks load.
- **HCCL small-message tuning** → no documented intra-node knob helps; the lever is keeping the all-reduce in-graph (lazy collectives), not HCCL flags.

## Engine-dev headroom (beyond config; only after the floor is reached)
- **MTP / multi-token-predict:** GLM has a native `num_nextn_predict_layers` head and is **non-hybrid**, so
  speculative/MTP decode is structurally available (unlike the Qwen hybrid where it's dead). This is the
  real path past the per-token sync floor — but it's engine work, not a flag.
- A fused int4→MME MoE kernel (avoid the dequant-to-bf16 round trip).
