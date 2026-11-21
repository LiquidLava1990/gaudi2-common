# GLM-5.1 on 8× Gaudi 2 — Master Speed Optimization Plan (2026-06-08)

Verified against the live engine, the `:next` image, the checkpoint, and the hpu-moe-ext/hpu-dsa-ext
source. Assumes the **offline MoE pre-repack** lands (MoE w13/w2 pre-converted uint8→int32 on disk →
no load-time repack → GLM fits `:next` → **lazy mode reachable**). Levers stack on that.

**Baseline (live `:1.24`):** eager (LAZY=0, compile NONE), TP=8+EP, **dense MLA** (DSA flag-bypassed),
chunked_prefill, no prefix-cache, async-sched, no spec, no routed-K, bf16 KV. Single-stream ~2.2 tok/s.

**Two structural floors:** single-stream decode is host/sync + weight-bandwidth bound (~29–34 ceiling,
75 unreachable); the **flapping O0–O3 fabric** stalls per-token all-reduces — `[FABRIC]` levers pay off
fully only after the **new baseboard + modules** land (incoming).

## RANKED LEVERS
- **L1 — Lazy graph replay + `HPU_MOE_ROUTED_K=8`** `[FABRIC]` — THE headline, ~5–10× single-stream.
  Lazy replays the decode graph (kills per-token host bubble) and makes routed-K a win (dequant only the
  ≤8 routed experts vs all 32 local → ~4× MoE-dequant cut). Proven ~29 tok/s on 397B. Eager: routed-K is
  SLOWER (inseparable from lazy). Enable: `models/glm51ar-lazy.env` (now wired to the repack cache).
  Run `HPU_MOE_ROUTED_VERIFY=1` once. Stay **TP=8** (TP=4 OOMs in lazy capture).
- **L3 — MTP / nextn speculative decode** `[FABRIC]` — largest single-stream lever AFTER L1; beats the
  sync floor (2 tokens/all-reduce). **MTP head SHIPS on disk** (`num_nextn_predict_layers=1`, layer-78
  `eh_proj/enorm/hnorm/shared_head.norm`). `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'`.
  **Gap:** no `GlmMoeDsaMTPModel` registered → light wiring to point spec at `DeepSeekMTPModel` for
  `model_type=glm_moe_dsa` + verify acceptance>0 + coherence-check the DSA-MLA draft head.
- **L2 — Re-enable real DSA sparse attention** — currently **dense-bypassed by ONE flag**
  (`hpu_mla_skip_indexer.py` forces `FULL_ATTENTION_FALLBACK=True`+`skip_topk=True`; `sparse_mla_wire.py:113`
  early-returns dense). Flip to false → lightning-indexer top-k (index_topk=2048) caps decode attention at
  O(2048). ~0 gain at 4K ctx, **multiple-× at 32–64K**; lets you raise max_model_len. Re-gate coherence
  under lazy (DSA only ran eager). Keep `--no-enable-prefix-caching` while sparse.
- **L4 — Recipe cache persistence** — already wired (`/mnt/data1/recipe_cache`); per-config isolation
  `-e PT_HPU_RECIPE_CACHE_CONFIG=/recipe_cache/glm51ar-lazy,false,8192`. **Rebuild once after repack.**
- **L5 — Bucket collapse** — single-stream: `VLLM_DECODE_BS_BUCKET_MIN/STEP/MAX=1` → one decode graph,
  every step a replay hit (kills the 0→5→0 compile sawtooth). Concurrency: BS_BUCKET_MAX=max_num_seqs.
- **L6 — Concurrency frontier** — `--max-num-seqs` 16→32→64→128 (measure, saturates), `--max-num-batched-tokens
  16384`, `--block-size 128`. The throughput dial; bs>1 disables routed-K (correct — concurrency wants all-expert).
- **L7 — FP8 KV (`--kv-cache-dtype fp8_inc`)** — ONLY `fp8_inc` accepted on Gaudi. ~0 single-stream, ~2×
  KV headroom for concurrency/long-ctx. **Needs calibration** (never unit_scale); coherence-check the MLA
  latent-KV fp8 explicitly.
- **L8 — `VLLM_GRAPH_RESERVED_MEM`** — re-tune UP post-repack (pool now ~52GB not 94GB) for more KV.
- **L9 — Warmup ON** (`VLLM_SKIP_WARMUP=false`) after buckets+cache settle — kills first-request stalls.
- **L10 — Host CPU/mem NUMA pin** `[FABRIC]` (`CPUSET_*`; TP=8 spans both sockets so host-pin only).
- **L11 — MLA absorption hygiene** — already in the process-weights patch; verify absorption stays on +
  doesn't split the decode graph under lazy.

## REJECTED (don't waste a load): stock FP8 MoE compute (no fp8 ckpt, INT4 dequants to bf16 anyway),
routed-K<8 (gate off), in-engine DP (non-functional; GLM too big to replicate anyway → TP=8 only),
prefix-caching (breaks DSA), HPU_GLM_NORM_OFFSET (salad), `--quantization` flag, enforce-eager.

## APPLICATION ORDER (one change per measured load, clean attribution)
1. (prereq) offline repack → GLM fits `:next`.  2. **L1** lazy+routed-K (VERIFY once) → coherence + measure (~5–10×).
3. **L5+L4** buckets + recipe warm → real floor.  4. **L3** MTP spec → next multiple.  5. **L2** DSA + re-gate → long-ctx.
6. **L6+L7+L8** concurrency frontier.  7. **L9/L10/L11** polish.  8. **[FABRIC]** baseboard swap uncaps L1/L3/L5/L10.
