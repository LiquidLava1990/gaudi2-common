# 397B Single-Stream Speed — Deliberation Verdict (2026-06-06)

Source: 44-agent adversarial workflow `wf_e55d9428-4a8` (2.35M tok), source-grounded against in-image vllm-gaudi + `/mnt/fastmodels/Qwen3.5-397B-A17B-GPTQ-Int4/config.json`. Supersedes the ceiling claims in the older `SPEED_RUNBOOK.md`.

## VERDICT: 75 tok/s single-stream on the 397B is NOT reachable.
- Decode is **serial-sync bound, not BW/FLOP**: measured ~29 tok/s = **~8% of the ~356-366 tok/s/card-set HBM-BW floor**. py-spy: ~65% wait_stream/record_event + ~30% replayV3.
- The wall = (1) ~60 per-decoder-layer graph-fragment boundaries — the per-layer `mark_step` is **load-bearing** for SSM-state scheduling (`VLLM_CONFIG_HIDDEN_LAYERS=60` regressed to 6.4 tok/s); (2) **45 of 60 layers are strictly-sequential GatedDeltaNet recurrences**.
- 75 tok/s = 13.3 ms/tok = cut 21 ms off the 34.5 ms budget → impossible without destroying that serial structure.
- REFUTED: "kill the 1200 convert_from_uint4 kernels" — under lazy they're INSIDE graph fragments (`moe_wna16_hpu.py: if not htorch.is_lazy()`), not host syncs → removing = ~+4.5%, not +130%.

## Ranked plan
1. **[config · MANDATORY]** Instrumented baseline: `MODEL_ENV_EXTRA='-e HPU_MOE_ROUTED_VERIFY=1 -e VLLM_PROFILE_DECODE=1' ./serve.sh qwen35-397b`. Grep `[routed-verify] maxdiff` (<1e-2 all layers); pull decode trace (confirm sync-bound + converts graphed); `bench_client --concurrency 1`. (~29; the measurement + correctness gate.)
2. **[config]** `VLLM_GDN_COMPUTE_FP32=0` → +1-3% (GDN recurrence runs fp32 today per `hpu_gdn_pytorch.py:30`). MANDATORY perplexity/logprob A/B (bf16 drift over 45 layers).
3. **[config]** `VLLM_CONFIG_HIDDEN_LAYERS=4` — ONE try (aligns to `full_attention_interval=4` block) → ±few%. If it regresses, interval=1 stays.
4. **[engine-dev · weeks]** SSM-aware spec-decode/MTP — NOT blocked (num_accepted_tokens algo exists upstream `mamba_attn.py`; HPU phase-1 stubbed `hpu_gdn_pytorch.py:551-561` + `causal_conv1d_pytorch.py:368`). ~35-42, forfeits `--async-scheduling`. The only multiplicative lever, still < 75.
5. **[engine-dev · weeks]** fused int4→MME MoE kernel (new Habana SynapseAI WNA16 kernel; MME has no native int4) → ~34-42, ceiling ≤69.

## DROP — verified non-levers / traps (do NOT spend a container restart)
lower top_k / `HPU_MOE_ROUTED_K`<10 (→ all-512-expert REGRESSION); stock FP8 MoE swap (regression + no fp8 ckpt on disk); shared-expert-stream (CUDA-gated dead on HPU); KV/SSM-state fp8 (0.01-0.06% of budget); NUMA single-socket pin (residual already hidden; TP=4 OOMs); FP8-for-BW (invisible at 8% util). Keep `sslazy8.env` knobs as-is.

## Where the big numbers ARE
- **THROUGHPUT @ concurrency** → ~366 tok/s/card-set aggregate (per-token serial/dequant amortizes across the batch; FP8 pays off here). **397B throughput never measured.**
- **bf16 Qwen3.6-35B-A3B** → only model that can hit 50+ single-stream (~40 layers, no dequant chain). Deliberation estimates 50-75, but `bench/MASTER_REPORT.md` shows 13-20 @conc=1 → **MEASURE, don't trust the estimate.** Capability downgrade (3B active vs 17B).
