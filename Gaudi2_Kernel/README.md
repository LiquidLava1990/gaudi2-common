# Gaudi 2 Kernel — reproducible bring-up + optimization for big MoE LLMs on 8× Gaudi 2

This folder is the **reproduce-it-on-any-machine** record of getting large MoE models running and fast on
8× Intel Gaudi 2 with vllm-gaudi. Each model doc is self-contained: symptom → proven root cause →
exact code/config fix.

| Doc | What it covers |
|---|---|
| [`GLM-5.1.md`](GLM-5.1.md) | `GLM-5.1-int4-mixed-AutoRound` (GlmMoeDsa, 744B/40B-active). The **4 coherence fixes** (embed-NaN, o_proj RowParallel scale-shard, MoE routed global→local id remap, NeoX RoPE), code-for-code, **and the `+1`-norm trap to avoid**. From token-salad → coherent. |
| [`Qwen-3.5-397B.md`](Qwen-3.5-397B.md) | `Qwen3.5-397B-A17B-GPTQ-Int4`. Its single-stream config + the **measured ~29–34 tok/s ceiling** (sync-bound; 75 unreachable). Which GLM fixes do/don't apply. |
| [`SPEED_AND_ITERATION.md`](SPEED_AND_ITERATION.md) | The single-stream **optimization runbook**: the real levers (`HPU_MOE_ROUTED_K=8` ~4×, lazy mode +~21%), the **DROP-list**, and the **fast-iteration recipe** (recipe cache + streaming `bench_client`). |
| [`Gaudi2_Optimization_Guide.md`](Gaudi2_Optimization_Guide.md) | The broader Gaudi-2 optimization reference (general knobs, history). |

**Golden rules for this box** (see memory):
- Start/stop ONLY via `cd $HOME/AI && ./serve.sh <key>` / `./stop.sh`. **Never `docker rm -f` a vllm
  container** — it wedges the Habana scheduler (reboot-only recovery).
- One model at a time (`CONTAINER=vllm-active`); no rapid container churn.
- The diagnostic method that cracked GLM: **on-HPU tensor dump → CPU fp64 reference compare → full-forward
  A/B**. Per-layer checks alone are a trap (each layer fed its own corrupted input looks correct).

Authoritative live state + history is in the auto-memory (`glm51-dsa-1m-requirement`,
`gaudi-perf-targets`, `gaudi-single-stream-speed`, `gaudi-fabric-defect-rma`).
