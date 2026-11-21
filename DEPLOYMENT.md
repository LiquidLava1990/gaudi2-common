# PRODUCTION DEPLOYMENT — GLM-5.2 + Qwen3.6-27B (Gaudi2 box)
_Last updated 2026-07-07. Canonical reference for the real serving system._

## TL;DR endpoint table (for the calling system)

| | GLM-5.2 | Qwen3.6-27B |
|---|---|---|
| **URL** | `http://<SERVER_IP>:8000/v1` | `http://<SERVER_IP>:8001/v1` |
| **model name** | `glm52` | `qwen36` |
| **API key** | none (keyless) | none (keyless) |
| **build** | int4 champion (W4A8 mono int4→fp8→fp8-MME + fp8-KV + depth-skip K=8) **+ abliterated graft** + full CPU-offload tier | FP8, GDN hybrid, fp8-KV |
| **context** | 131072 (128K) + 40 GB host cold tier | 262144 (256K native) |
| **cards** | TP8 — **all 8** Gaudi cards | TP8 — **all 8** Gaudi cards |
| **status** | **LIVE** (default resident) | **config-ready, NOT running** — see the co-residence wall below |

`<SERVER_IP>` = the box's Tailscale IP (endpoints only reachable over the tailnet).

## The co-residence wall (READ THIS) — they cannot both be live simultaneously

Proven at the driver level 2026-07-06: a Gaudi card is acquired by **exactly one process**. A bare
`torch.zeros(5).to("hpu")` in a second container while GLM runs fails with
`synStatus=8 [Device not found] Device acquire failed` — it is NOT memory (168 GiB free), NOT the port,
NOT the API key. There is no MPS / time-slice / shared-acquire flag in the habana runtime, and the
repo's own multi-model script (`dp_serve.sh`) **partitions** cards into disjoint `HABANA_VISIBLE_MODULES`
sets rather than sharing them.

GLM-5.2 must occupy **all 8 cards**: its int4 weights are 48.7 GiB/card at TP8, it only divides evenly
by 8 (64 heads, 256 experts → TP5/6/7 invalid), and TP4 would be ~97 GiB/card (over the 96 GiB card).
So GLM has no footprint smaller than all-8, leaving zero cards for a second TP8 Qwen process.

**⇒ Deployment is SWITCH, not simultaneous.** Both are production-ready, keyless, on their own ports;
one runs at a time and owns the full box.

### Options evaluated for true both-live (none adopted yet — awaiting decision)
1. **CPU Qwen** (both live, zero risk): GLM full TP8 on the cards + Qwen on the box's CPU (499 GB RAM).
   Estimate ~**2–5 tok/s** (Qwen is DENSE 27B → 31 GB read/token; DDR4-2400, no AMX). Fine for
   light/background traffic, painful for interactive.
2. **Separate/smaller hardware for Qwen** (cleanest): Qwen is only 30 GB; GLM keeps this box at full TP8.
3. **TP4 partition** (risky build): GLM TP4 on cards 0–3 + Qwen TP4 on 4–7. Requires (a) NEW code to
   skip-*load* the 8 depth-skipped MoE layers' weights (`HPU_SKIP_LAYERS` currently skips COMPUTE only,
   weights still load — that's why GLM is 48.7 GiB/card even with 8 layers skipped), and (b) the champion
   running at TP4/EP4 (UNTESTED — all kernels tuned for TP8/EP8). Halves GLM's compute/KV. ~½ day, may fail.

## How to run / switch

```bash
cd $HOME/AI
./containers.sh on  glm52-coresident-offload    # bring up GLM (ABLITERATED=1 auto via env? NO — see note)
./containers.sh on  qwen36-27b-coresident        # switch to Qwen (STOPS GLM first)
./containers.sh off <key>                         # stop one
./containers.sh coadd <key>                        # start alongside a running one (co-resident; only works
                                                   #   if the device-share wall is solved — today Qwen coadd
                                                   #   still hits synStatus=8)
```

**ABLITERATED note:** GLM's abliterated graft is opt-in per launch:
`ABLITERATED=1 ./containers.sh on glm52-coresident-offload` swaps `HPU_FP8_OTHER_GRAFT` to
`/models/glm52-fp8-other-abliterated` (perf-identical; only o_proj differs). The CURRENT live instance
was started abliterated. If you restart WITHOUT `ABLITERATED=1` you get the stock (refusal-capable) graft.

## Keyless serving

`common.env` has `API_KEY=` (empty) → `containers.sh` only passes `--api-key` when non-empty, so both
models serve with NO auth. A keyless server still accepts requests that DO send an Authorization header
(it ignores it), so old key-bearing tooling keeps working.

## MLA-KV 2× FIX (2026-07-07) — DSA indexer dead-cache removed, hot KV doubled
The GLM KV cost was ~89,856 B/tok — **2× the true MLA fp8 cost (44,928)**. Root cause: the DeepSeek-V3.2
DSA indexer's paged `k_cache` (`*.self_attn.indexer.k_cache`) is registered per layer, but on HPU the
indexer is SKIPPED (hpu-dsa-ext forces `skip_topk=True` → full MLA attention), so it's NEVER used — yet
vLLM's uniform-page KV grouping padded it up to the full MLA page (16,896→73,728 B/block) × 78 layers =
the entire 2×. FIX: the mounted runner (`vllm-gaudi-patch/v1/worker/hpu_model_runner.py`,
`get_kv_cache_spec`) **deregisters the dead `*.indexer.k_cache` layers from static_forward_context**
(gated by `HPU_KEEP_INDEXER_KV`, default off). Verified: KVDIAG 156→78 layers, per-token 89,856→44,928,
natural num_blocks 3,277→6,558. Full root-cause: `bench/reports/c3b_20260705/MLA_KV_2X_ROOTCAUSE.md`.

## Current production KV sizing — load-tested robust (2026-07-07 FINAL)
- `--num-gpu-blocks-override 3500` = **448,000 hot tokens/card** (2.9× the pre-fix 153,600; above the
  444K target). The indexer-KV fix is what makes this fit.
- `--kv-transfer-config` cold tier = **300 GB** host RAM (~6.7M cold tokens).
- `--max-num-batched-tokens 2048`, `VLLM_GRAPH_RESERVED_MEM=0.06`, util 0.80, mml 131072 (128K/request).
- **Load-tested:** 4 concurrent 25,008-token prompts → 4/4 OK, ZERO OOM, health 200. Rest 86.9 GiB,
  peaked 91.3/96 GiB (9.1 GiB headroom). Coherent. THIS IS THE LIVE CONFIG.

## 448K is the robust HOT CEILING *with* the offload connector — measured, not assumed
The offload connector is NOT free per hot block: during serving it needs ~8 GiB of LIVE GPU headroom for
on-the-fly staging (moving cold blocks host↔device as the hot pool fills). Tested:
- 448K hot → rest 86.9, 9.1 GiB headroom → **passes** the 4×25K big-context load.
- 500K hot → rest 91.3, 4.7 GiB headroom → **OOMs** the engine under the same load (tested at mnbt 2048
  AND mnbt 1024 — the activation lever can't recover it; the connector staging needs the headroom).
The cold-tier *size* is still free (250 vs 300 GB = same GPU footprint), so 300 GB cold + 448K hot is the
max robust combo WITH offload.
### To exceed 448K hot: drop the offload connector
Without `--kv-transfer-config`, each hot block costs only its KV (~5.75 MiB) instead of ~11.5 MiB (KV +
connector staging), so ~600-700K hot fits robustly — but there is NO host-RAM spill (context capped at
the on-card hot pool; per-request still 128K via max-model-len). Choose: **448K hot + 300 GB offload**
(current) OR **~650K hot, no offload**.

## Offload build detail (GLM) — full hot/warm/cold KV tiering, VERIFIED engaged

- **OffloadingConnector + CPUOffloadingSpec** created on all 8 workers.
- **GPU hot pool** = 153,600 tokens (1200 blocks, `--num-gpu-blocks-override 1200`).
- **CPU cold tier** = 40 GB (`cpu_bytes_to_use=40e9`, LRU eviction). Cold/paused agents' prefix blocks
  spill to host RAM and reload on reuse → ~1M-token effective context depth.
- **APC** (`--enable-prefix-caching`) dedups shared system prompts across agents.

**Concurrency caveat:** "Maximum concurrency for 131,072 tokens per request: 1.17x." At FULL 128K
context the GPU hot pool holds ~1 max-length request; the 40 GB cold tier carries everything beyond
that. Many SHORTER concurrent requests are fine (lots of slots); many SIMULTANEOUS max-length requests
lean on the cold tier (hot/warm/cold churn). Working as designed.

**KV bytes/token — MEASURED ~88 KiB/token (NOT the connector double-allocating):** the engine reserves
20.56 GiB usable KV = 1919 natural blocks = ~88 KiB/token. That is ~2× the theoretical single-latent
MLA fp8 cost (~44 KiB/token). The cause is the HPU attention backend, NOT the offload connector: on HPU
`get_kv_caches_4D()` returns a **(K, V) tensor pair** for MLA layers instead of the packed single-latent
tensor (see the docstring in `_src/vllm_gaudi/v1/kv_offload/worker/cpu_hpu.py:register_kv_caches`). The
connector itself allocates NO extra pool memory — it only makes int8 VIEWS of the existing KV tensors +
a 256 MB staging buffer. So the ~2× is connector-independent (same without offload). If the HPU backend
stored MLA as the true single latent, KV capacity would ~2× for free — a vllm-gaudi backend change, not
a config knob.

## Memory model (measured on-box, for future tuning)

- `gpu-memory-utilization` = fraction of free-AFTER-weights. At util 0.70: free 45.91 → 32.14 GiB usable;
  HPUGraphs 3.21; natural KV pool would fill the rest → no room for the connector's 1 GB HPU staging
  buffer → OOM. Hence the explicit `--num-gpu-blocks-override 1200` cap.
- GLM footprint: 48.7 weights + ~16 KV + ~3 graph + ~1 connector ≈ 77 GiB/card used (of 96).
- `PT_HPU_POOL_MEM_ACQUIRE_PERC=86` is set on GLM (bounds the habana pool) — kept for if a device-share
  mechanism ever lands; it does NOT by itself enable co-residence (device acquire is still exclusive).

## Config files
- `models/glm52-coresident-offload.env` — GLM production (abliterated-capable, offload, 128K).
- `models/qwen36-27b-coresident.env` — Qwen production (256K, full-box standalone, util 0.85).
- `containers.sh` — `on`/`off`/`coadd`/`list`; keyless via empty `common.env:API_KEY`.

## Performance campaign context (why this build is the champion)
See `bench/reports/c3b_20260705/` (8 reports) + `~/Desktop/RETRY_RUN_20260705.md`. Key result: GLM-5.2
concurrency is **compute-bound on the MoE-expert matmul** (profiled: 41% matmul, 0.3% collectives); DP,
spec-decode, collective sweeps, fences, and kernel-consolidation were all measured as NOGO/marginal.
The champion (W4A8 + fp8-KV + depth-skip at TP8) is at its practical compute floor for this model — no
drafter (a spec drafter LOSES ~2× at concurrency; helps only single-stream).
