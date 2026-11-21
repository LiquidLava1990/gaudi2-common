# Gaudi 2 Kernel — Qwen-3.5-397B (A17B GPTQ-Int4) bring-up + single-stream config

**Model:** `Qwen3.5-397B-A17B-GPTQ-Int4` (397B total / ~17B active MoE, Qwen3.5 MoE arch,
GQA attention — NOT MLA). **Quant:** uniform GPTQ int4 (`--quantization gptq_hpu`), served through the
same Gaudi MoE op as GLM (`hpu-moe-ext` `MoeWNA16HPUMethod` → `VllmMixtureOfExpertsOpWNA16`).
**Status:** coherent + serving on 8× Gaudi 2. This doc captures the config that runs it and the
**measured single-stream reality**.

---

## 0. Stack
| Component | Value |
|---|---|
| Container | `vllm-gaudi:next` (1.24) |
| Parallelism | TP=8 + `--enable-expert-parallel` (also runs TP=4 in the `ss` variant) |
| MoE op | `hpu-moe-ext/moe_wna16_hpu.py` (GPTQ int4 path) — same op as GLM |
| Attention | GQA (standard) — none of the GLM MLA/RoPE/embed fixes are needed here |

---

## 1. Which GLM fixes apply here

| GLM fix | Applies to Qwen-397B? |
|---|---|
| Fix 1 embed-NaN (`HPU_SANITIZE_EMBED`) | **No** — GLM-specific TP=8 GlmMoeDsa embedding artifact |
| Fix 2 o_proj RowParallel int8 dequant | **N/A for attn** (Qwen attn isn't int8-mixed MLA). The `GPTQDequantToBF16` scale-shard fix is harmless/correct if any 8-bit RowParallel linear exists; uniform int4 GPTQ doesn't hit it |
| Fix 3 MoE routed global→local id remap | **Indirectly** — the 397B normally takes the `_routed_moe` fast path (`HPU_MOE_ROUTED_K=10`) which already remaps to local ids, so it never hit the broken standard path. The Fix-3 standard-path remap now also covers its fallback. **Net: 397B was the reason the bug was masked** (its config sidesteps it) |
| Fix 4 NeoX RoPE (`HPU_GLM_ROPE_NEOX`) | **No** — Qwen uses its own (correct) RoPE |

So Qwen-397B "just runs" on the shared GPTQ-int4 MoE infra + its config; it did **not** need the GLM
coherence fixes. Its open problem is purely **single-stream speed**.

---

## 2. The proven single-stream config (`models/qwen35-397b-sslazy8.env`)

```ini
IMAGE=vllm-gaudi:next
MODEL_PATH=Qwen3.5-397B-A17B-GPTQ-Int4
SERVED_NAME=qwen35-397b
MODEL_ENV="-e PT_HPU_LAZY_MODE=1 -e PT_HPU_WEIGHT_SHARING=0 -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=900 -e VLLM_RPC_TIMEOUT=100000000 -e MAX_EXPERTS_PER_SLICE=64 \
  -e VLLM_DECODE_BS_BUCKET_MIN=1 -e HPU_MOE_ROUTED_K=10 -e VLLM_GRAPH_RESERVED_MEM=0.2"
VLLM_ARGS="--quantization gptq_hpu --tensor-parallel-size 8 --enable-expert-parallel \
  --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.85 --block-size 128 \
  --async-scheduling --no-enable-prefix-caching --trust-remote-code"
```

**Single-stream knobs that matter (these are the levers to also try on GLM):**
| Knob | Why |
|---|---|
| `PT_HPU_LAZY_MODE=1` | **Lazy/HPU-graph mode** — fuses the decode step into graphs (vs eager). GLM currently runs `=0`; this is a prime GLM speed experiment (caveat: the GLM `mark_step` and DSA-compile patches were written for `=0`, must re-verify) |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` | overlaps the TP all-reduces with compute under lazy mode |
| `HPU_MOE_ROUTED_K=10` | **routed-gather**: dequant only the ≤10 routed experts/token instead of all local experts (the ~99 ms/token all-expert dequant floor). The single biggest MoE single-stream lever |
| `MAX_EXPERTS_PER_SLICE=64` | expert-slice sizing for the MoE op |
| `VLLM_DECODE_BS_BUCKET_MIN=1` | decode batch bucket starts at 1 — avoids padding a single stream up to a larger batch bucket |
| `VLLM_GRAPH_RESERVED_MEM=0.2` | HBM reserved for HPU graphs |

---

## 3. Measured single-stream reality (DO NOT chase 75 tok/s)

From a 44-agent deliberation (`wf_e55d9428`) + on-card runs (see `gaudi-perf-targets` /
`gaudi-single-stream-speed` memory):

- **397B single-stream caps ≈ 29–34 tok/s.** It is **sync/host-bound**, sitting at ~8% of the HBM
  bandwidth floor — the decode is dominated by serial host↔device syncs + the per-token MoE dequant, not
  raw FLOPs/BW. Config knobs each move it **0–3%**; engine-level dev tops out ≈ 42 tok/s.
- **75 tok/s single-stream is structurally unreachable** for this model on this hardware. The honest
  target for "big single number" is **throughput @ concurrency** (~366 tok/s across the card-set, never
  yet measured) or the small bf16 35B model — NOT 397B single-stream.
- **DROP-list (knobs proven NOT to help / that backfire):** see `gaudi-perf-targets` memory for the full
  trap list before spending a load on any of them.

**Bottom line:** Qwen-397B runs; its single-stream ceiling is a hardware property (serial-sync wall),
not a missing config. The reachable wins are throughput@concurrency and porting GLM's eventual
single-stream tuning back here.

---

## 4. Reproduce
```bash
cd $HOME/AI
USE_DSA=0 ./serve.sh qwen35-397b-sslazy8     # or qwen35-397b-ss (TP=4 variant)
```
(`./stop.sh` to stop. Never `docker rm -f` a vllm container — it wedges the Habana scheduler,
reboot-only recovery.)
