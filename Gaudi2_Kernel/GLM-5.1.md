# Gaudi 2 Kernel — GLM-5.1 (GlmMoeDsa) bring-up, reproducible fixes

**Goal of this doc:** everything needed to take `INC4AI/GLM-5.1-int4-mixed-AutoRound`
(744B / 40B-active MoE, `GlmMoeDsaForCausalLM`) from *not loading / token-salad* to
**coherent serving on 8× Intel Gaudi 2**, code-for-code, on any machine. Each fix lists the
symptom, the root cause (how it was proven), and the exact change.

Verified coherent 2026-06-07: `"The capital of France is Paris."`, `"...capital of Japan is Tokyo."`,
`"Three primary colors are ... red, green, blue."`

---

## 0. Stack / environment

| Component | Version / value |
|---|---|
| Hardware | 8× Intel Gaudi 2 (HL-225), single node |
| Container | `vllm-gaudi` 1.24 (vLLM `0.22.1rc1`) |
| Model | `INC4AI/GLM-5.1-int4-mixed-AutoRound` (`GlmMoeDsaForCausalLM`) |
| Quant | AutoRound **int4-MIXED**: routed MoE experts int4; attn projections + shared-expert int8 (→ dequant-to-bf16 at load); layers 0-2 dense MLP + DSA `weights_proj` fp16 |
| Parallelism | `--tensor-parallel-size 8 --enable-expert-parallel` (TP=8 attention, EP=8 MoE: 256 experts → 32 local/rank) |
| Arch | MLA attention (kv_lora=512, q_lora=2048, qk_nope=192, qk_rope=64, v_head=256, qk_head=256), 256 routed experts top_k=8 + 1 shared, 78 layers, `first_k_dense_replace=3`, DSA lightning-indexer (gated OFF → dense MLA for now) |
| Reference | `transformers` `GlmMoeDsaForCausalLM` (`modeling_glm_moe_dsa.py`) — the AUTHORITATIVE numerical ground truth |

The fixes live in two out-of-tree vLLM plugins (registered via `vllm.general_plugins` entry points):
`$HOME/AI/hpu-moe-ext` (AutoRound/MoE) and `$HOME/AI/hpu-dsa-ext` (DSA + GLM arch patches).

---

## 1. THE BUG CHAIN (what was wrong, in load order)

This checkpoint had **never** run coherently on this path. It was **four independent bugs stacked**.
Each was found by **value-level comparison** of on-HPU dumps against the fp64 transformers reference —
NOT by guessing. The decisive tool was a **full 78-layer fp64 forward from the embedding** (see §3): a
per-layer check is a TRAP because each layer fed its own already-corrupted input looks correct (cos > 0.999)
even when the cumulative output is garbage.

| # | Bug | Symptom | Fix location |
|---|---|---|---|
| 1 | Embedding emits NaN under TP=8 | residual stream poisoned → frozen norms → salad | `hpu-dsa-ext` env `HPU_SANITIZE_EMBED=1` |
| 2 | `o_proj` RowParallel int8 dequant uses wrong scale-groups | o_proj weight collapses ~0.4× → noise | `hpu-moe-ext/moe_wna16_hpu.py` (baked in) |
| 3 | MoE routed experts return 0 under EP | MoE = shared-expert only → salad | `hpu-moe-ext/moe_wna16_hpu.py` (baked in) |
| 4 | Wrong RoPE convention on MLA path | scrambled positions | `hpu-dsa-ext` env `HPU_GLM_ROPE_NEOX=1` |

**⚠️ Do NOT apply the `+1` RMSNorm "fix".** See §4 — it was a *misdiagnosis* that itself produces salad.

---

## 2. THE FIXES (code-for-code)

### Fix 1 — Embedding NaN sanitize  *(env-gated)*
**Symptom:** the TP=8 vocab-parallel embedding forward emits a spurious NaN (the embed *weight* is
verified clean) which poisons the residual stream; a downstream NaN-guard then freezes every layer to an
identical norm (~17.11) → token salad.
**Fix:** `nan_to_num` the embedding output. Plugin `hpu-dsa-ext/hpu_dsa_ext/nan_sanitize_embed.py`,
gated by `HPU_SANITIZE_EMBED=1`. Harmless if no NaN.

### Fix 2 — `o_proj` RowParallel GPTQ scale-sharding  *(baked into the plugin)*
**Symptom:** attention output decorrelated/too small (o_proj weight rms 0.0033 vs correct ~0.009),
only on the RowParallel `o_proj` (the one input-sharded int8 attn projection).
**Root cause (proven against a live rank-4 o_proj dump, cos 0.9999987):** the GPTQ scales/qzeros are
created with `scale_and_zero_size = input_size // group_size` (FULL groups) and declared `output_dim`-only
(`ChannelQuantScaleParameter`), so they are **not input-sharded**. But `codes` IS the per-rank input shard
for a `RowParallelLinear`. A local `g_idx` then indexes **rank-0's** scale groups on **every** rank →
collapsed weight → the o_proj all-reduce sums corrupt partials → salad. (ColumnParallel linears like
`kv_b_proj` are fine: full-group scales, group count == local.)
**Fix** in `GPTQDequantToBF16LinearMethod.process_weights_after_loading`
(`hpu-moe-ext/hpu_moe_ext/moe_wna16_hpu.py`): offset `g_idx` into this rank's global groups when the
scales are full (input-sharded). Use `layer.tp_rank` (the rank `RowParallelLinear` actually used to shard
its codes — robust under `--enable-expert-parallel`), not a fresh `get_tensor_model_parallel_rank()`.

```python
g_idx = (torch.arange(codes.shape[0]) // gs).long()         # desc_act=False
local_groups = codes.shape[0] // gs
if scales.shape[0] > local_groups:                          # scales FULL but codes input-sharded
    shard_idx = getattr(layer, "tp_rank", None)             # RowParallel's own shard index
    if shard_idx is None:
        from vllm.distributed import get_tensor_model_parallel_rank
        shard_idx = get_tensor_model_parallel_rank()
    g_idx = g_idx + int(shard_idx) * local_groups           # -> this rank's global groups
z = zeros[g_idx]; s = scales[g_idx]; w = (codes - z) * s
```
Verified: cos 1.0 / maxdiff 0 reproduced for all 8 ranks vs the ground-truth dequant.

### Fix 3 — MoE routed experts: global→local expert-id remap  *(baked into the plugin)*
**Symptom:** generation salad; the live MoE output equals **exactly 1× the shared expert** (cos 0.99999),
i.e. the routed experts contribute **nothing** — even for a locally-owned, selected expert.
**Root cause:** the HPU `torch.ops.hpu.mixture_of_experts` intrinsic returns ~0 for the routed branch when
`experts_min != 0` (the non-zero GLOBAL expert window under expert-parallel). The op holds only this rank's
32 local experts (indexed `[0..31]`) but is handed GLOBAL `topk_ids` (e.g. 96..127 for rank 3); they never
match → 0. (The 397B masks this by always taking the `_routed_moe` fast path via `HPU_MOE_ROUTED_K=10`,
which remaps to local ids — GLM with no `HPU_MOE_ROUTED_K` hit the broken standard path.)
**Fix** (two halves, `hpu-moe-ext/hpu_moe_ext/moe_wna16_hpu.py`):

(a) build the op with a **LOCAL** window `[0, num_experts-1]` instead of the global `[ep_shift, …]`:
```python
num_experts = layer.local_num_experts
ep_shift = layer.ep_rank * num_experts
layer._hpu_experts_min = ep_shift            # keep the global offset for the remap below
layer._hpu_local_experts = num_experts
layer.moe_op = VllmMixtureOfExpertsOpWNA16(num_experts, 0, num_experts - 1)   # LOCAL window
```
(b) remap GLOBAL `topk_ids` → LOCAL in `apply_monolithic`, zeroing non-local experts' weights
(do NOT rely on the op skipping out-of-window ids — it doesn't cleanly), for **ALL** ranks incl. ep_rank 0:
```python
emin = getattr(layer, "_hpu_experts_min", 0)
nloc = getattr(layer, "_hpu_local_experts", None)
if nloc is not None:
    in_win = (tid >= emin) & (tid < emin + nloc)
    tid = torch.where(in_win, tid - emin, torch.zeros_like(tid))   # global->local; non-local->0
    tw_ = torch.where(in_win, tw_, torch.zeros_like(tw_))          # zero non-local weights
out = layer.moe_op(x, tid, tw_, permuted_weights=perm, activation=act)
```

### Fix 4 — RoPE convention on the MLA path  *(env-gated)*
GLM-5.1 uses **NeoX rotate-half** RoPE, but the DeepSeek-V2 MLA path this model maps onto hardcodes
`is_neox_style=False` (interleaved). Plugin `hpu-dsa-ext/hpu_dsa_ext/glm_rope_neox.py`, gated by
`HPU_GLM_ROPE_NEOX=1`, forces `is_neox_style=True`. (Note: a full-forward A/B showed rope makes **no**
coherence difference here once §2-3 are fixed — the config's `rope_interleave=True` applies only to the DSA
indexer — but NeoX is correct per the reference, so keep it.)

---

## 3. The diagnostic method (how the bugs were actually found)

1. **On-HPU tensor dump** (read-only monkeypatch): dump a decoder layer's `hidden_in / residual_in /
   hidden_out` and the live dequantized `o_proj` weight for a chosen set of layers, in ONE bring-up.
   (`hpu-dsa-ext/hpu_dsa_ext/glm_layer_dump.py`, `HPU_LAYER_DUMP=1 HPU_DUMP_LAYERS=0,3,4,10,40,77`.)
2. **CPU fp64 reference**: re-implement the layer/MoE/MLA forward from the checkpoint weights using the
   `transformers` reference math; compare per-intermediate cosine-sim.
3. **THE decisive test — full 78-layer fp64 forward from the embedding**, then A/B the suspect convention.
   This is what caught that the `+1` norm was wrong (per-layer checks could not, see §4).
   Script: `hpu-moe-ext/tests/glm_full78_norm_ab.py`.
4. **moe-probe** (`HPU_MOE_PROBE=N`, in `moe_wna16_hpu.py`): logs per-rank routed expert count + output
   norm — used to confirm Fix 3 brought the routed experts alive on every rank.

---

## 4. ❌ The trap: the `+1` RMSNorm offset (DO NOT DO THIS)

An early misdiagnosis: the per-layer `GlmMoeDsaRMSNorm` weights are tiny (kv_a ~0.006, input_ln ~0.04),
and dense MLA attention V came out rms ~0.0002 ("dead"). Adding `+1` to those norm weights made V "alive"
(~0.047) and made **layer-0** match the reference — so it looked right. **It is wrong.** GLM-5.1 uses
**plain `weight * x_normed` RMSNorm**; the small weights are the *real trained values* (kv_b_proj etc. are
scaled to compensate), and small V is *correct*. The full-78-layer A/B is unambiguous:

| RMSNorm | argmax after "…France is" | " Paris" rank |
|---|---|---|
| **plain (no +1)** ✅ | **" Paris"** (p≈0.20) | #1 |
| +1 offset ❌ | `裘` (garbage) | #59767 |

`+1` makes all 312 per-layer norms ~26× too strong → corrupts the residual direction → multilingual salad.
**Never set `HPU_GLM_NORM_OFFSET=1`.** The plugin `glm_norm_offset.py` exists only as a historical
artifact and must stay disabled (it no-ops unless that env is set).

---

## 5. Reproduce: run command

Committed config lives in `$HOME/AI/models/glm51ar.env` (the env already carries
`HPU_SANITIZE_EMBED=1` and `HPU_GLM_ROPE_NEOX=1`; Fixes 2 & 3 are baked into `moe_wna16_hpu.py`).

```bash
cd $HOME/AI
# Fast coherence bring-up (eager, skip the 63-bucket warmup; ~6 min):
VLLM_ARGS_OVERRIDE="--tensor-parallel-size 8 --enable-expert-parallel --max-model-len 4096 \
  --max-num-seqs 16 --gpu-memory-utilization 0.9 --block-size 128 --no-enable-prefix-caching \
  --trust-remote-code --enforce-eager" \
USE_DSA=1 RESTART_POLICY=no \
MODEL_ENV_EXTRA='-e VLLM_SKIP_WARMUP=true -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800' \
./serve.sh glm51ar

# Compiled (for speed): just `USE_DSA=1 ./serve.sh glm51ar` (uses the env-file VLLM_ARGS,
# no --enforce-eager). First run compiles decode buckets — see the Optimization guide for
# the recipe-cache strategy so restarts don't recompile.
```

Test:
```bash
KEY=$(grep ^API_KEY common.env | cut -d= -f2)
curl -s http://<host>:8000/v1/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"glm51ar","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
# -> " Paris. The capital of France is Paris."
```

## 6. Status / open items
- ✅ Coherent (eager + compiled), dense-MLA (DSA indexer gated OFF via the `HPU_DSA_READY` path).
- ⏳ Single-stream **speed** optimization in progress (prime lever: MoE routed-gather `HPU_MOE_ROUTED_K`;
  see `Gaudi2_Optimization_Guide.md`). Greedy temp=0 shows repetition loops (expected; use repetition
  penalty / temp>0 for quality).
- ⏳ Re-enable the **DSA sparse** path (hard requirement) + re-verify coherence after speed work.
- ⚠️ Single-stream measurement is contaminated by the flapping O0–O3 scale-up fabric (all-reduce stalls);
  a clean number needs the reseat. See `gaudi-fabric-defect-rma` memory.
