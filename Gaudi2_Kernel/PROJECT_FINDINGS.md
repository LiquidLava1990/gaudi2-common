# Gaudi 2 Custom-Kernel Project — Findings & Results (the honest verdict)

Status: **Phase C complete.** This is the project-level narrative and the measured
performance truth. The kernel *technical* docs ([`KERNEL_ARCHITECTURE.md`](KERNEL_ARCHITECTURE.md),
[`KERNEL_TECHNICAL.md`](KERNEL_TECHNICAL.md), [`DSA_OPBACKEND_KERNEL_DESIGN.md`](DSA_OPBACKEND_KERNEL_DESIGN.md))
are the *how*; this document is the *what was built, what was learned, and what is — and is
not — a win.*

The headline, stated plainly up front so nothing here can be misread:

> **The custom fused INT4-MoE kernel is correct, general, stable, and serves a 397B model
> end-to-end — a real engineering achievement — but it is NOT a MoE speedup. The stock
> Habana `mixture_of_experts` intrinsic is faster in BOTH regimes (single-stream and
> throughput). Do not enable the fused MoE op for speed. The durable win is the reusable
> `OpBackend` kernel *infrastructure*, whose real payoff is kernels that have NO stock
> equivalent — proven by the DSA sparse-attention kernel, which is op-level validated
> on-card and is the 1M-context enabler.**

---

## 1. What was built

Three things, in increasing order of where the lasting value turned out to be.

### 1a. The `OpBackend` kernel infrastructure (the actual deliverable)

A from-scratch mechanism for registering **multi-node, SRAM-resident, HPU-graph-capturable
fused kernels** on Gaudi 2, built against the open-source Habana bridge
**HabanaAI/gaudi-pytorch-bridge tag `v1.24.0-1007`** (== `habana-torch-plugin 1.24.0.1007`,
torch 2.10, Apache-2.0). It registers a `habana::OpBackend` subclass in the bridge's
`KernelRegistry` whose `AddNode()` chains `BuildNode()` calls — the same way the bridge ships
its own multi-node ops (`mixture_of_experts_gen.cpp`, `convert_from_int4_gen.cpp`). This is
strictly more expressive than the public `UserCustomOpDescriptor` custom-op API, which binds
exactly one TPC GUID and cannot express a multi-node recipe or the non-persistent
dequant→MME bundling that produces SRAM residency.

Critically, **no hand-written TPC ISA** is involved — every node is a stock bridge GUID
(`reinterpret_cast`, `dequantize_4_bit_bf16`, `gemm`, `slice`, `sigmoid_fwd_bf16`,
`mult_fwd_bf16`, `add_fwd_bf16`, …). The contribution is the *graph and its geometry*, plus
the build/registration/serve mechanism around it. Because the op is captured into the HPU
graph like any built-in, the bridge owns the launch/stream/workspace and maps tensors itself
(no side-band `synLaunch`, no manual `data_ptr()` binding — which fails Synapse section
validation — and the recipe is durable across lazy capture/replay).

Roughly **~16 distinct node / build / serve gates** had to be cracked to get here (the full
list is in [`KERNEL_TECHNICAL.md`](KERNEL_TECHNICAL.md)'s "gates cracked" reference). The
hard-won ones worth flagging as project findings:

- **Build CPU-only, link the backend lib only.** Each `.so` links *only*
  `libhabana_pytorch_backend.so` with `-Wl,--allow-shlib-undefined`; linking the plugin libs
  triggers a double-registration abort (`hpu::habana_bernoulli is already registered`).
- **Two `.so`s per op, picked by mode.** An eager frontend (`habana::eager::EagerOp`) for the
  upstream-PT `vllm-gaudi:next` image and a lazy frontend (`habana_lazy::LazyOp`) for
  `PT_HPU_LAZY_MODE=1` serving; the backend `AddNode` recipe is shared *byte-for-byte* via a
  `*_shared.h` header. Only the lazy frontend records into the decode HPU-graph the capture
  engine replays.
- **Storage-offset discipline.** The first node is a `reinterpret_cast` on each weight/zp
  input — the most storage-offset-sensitive consumer possible. Feeding it a nonzero-offset
  `index_select` slice makes the cast read the wrong base (relNorm ~1e+11 garbage *whenever a
  real expert computes*, but trivially 0 when a rank has no routed expert — so a naive
  verify-relNorm looked half-OK). Fix: clone per-expert slices to `storage_offset=0`
  unconditionally, or feed the K experts as `TensorList` views into one on-device recipe.
- **Held-leaf materialization for multi-bucket lazy capture.** Under `wrap_in_hpu_graph` with
  `disable_tensor_cache=True` + `free_inplace=True`, any python-side per-expert clone or any
  in-forward-built tensor (e.g. the GPTQ-sym zero-points built via `expand().contiguous()`)
  is freed across decode-bucket graphs, so replay reads freed storage
  (`ValidateSyncInputTensors … tensor_data is empty` FATAL). Fix: route through the
  single-recipe TensorList op (no python intermediate to free) and materialize the
  experts-input `x` and the zero-points as persistent held leaves.

### 1b. The fused INT4-MoE kernel (the first kernel built on the infra)

`torch.ops.hpu_moe_ext.fused_expert` (single expert) and `torch.ops.hpu_moe_ext.fused_moe`
(K experts, one captured recipe). It emits the **entire per-expert INT4 forward as one fused,
SRAM-resident graph** — `reinterpret int32→uint4 → dequant W13 → MME GEMM → slice gate/up →
SiLU → dequant W2 → MME GEMM → × router_weight` — captured once and replayed per token, with
the two dequantized bf16 weights declared **non-persistent** so the GC's shipped fusion passes
bundle each dequant into the MME that consumes it. On device this shows the SRAM-residency
signature: no full bf16 weight in DRAM and **0 MB workspace** for tens of MB of bf16 weight.

It also ships a **batched-K** variant (`fused_moe_batched`: grouped 3D dequant + batch_gemm,
139→16 nodes regardless of K) and an **fp8-compute** toggle (a separate library, the GEMMs on
the MME in fp8).

The *premise* that motivated this kernel was a hard ~99 µs/token bandwidth floor: stock INT4
MoE on Gaudi dequantizes every expert to bf16, writes the full bf16 weight to HBM (because
`mixture_of_experts` is bf16-only), then runs a separate GEMM — a round-trip whose weight
traffic dominates single-stream decode. The fused kernel eliminates that round-trip by
construction. **This premise turned out not to hold against the actual stock op (see §3).**

### 1c. The DSA sparse-attention kernel (the infra's real payoff)

The same `OpBackend` mechanism is reused for **DeepSeek Sparse Attention** under the
`hpu_dsa_ext` namespace: a lightning-indexer top-k op (`indexer_topk`) and a gather-consume
sparse-attention op (`sparse_attn`). DSA is the 1M-context enabler (dense MLA is O(L²) and
intractable at long context; GLM-5.1 currently runs a coherent **dense bypass** that throws
away DSA). The full design — every DSA primitive mapped to an existing bridge GUID, the
generality strategy, the build order — is in [`DSA_OPBACKEND_KERNEL_DESIGN.md`](DSA_OPBACKEND_KERNEL_DESIGN.md).

**Why this is the payoff:** there is **no stock Habana intrinsic for DSA** (Gaudi's only
attention CGUID is `sdpa_fwd`; DeepSeek's `fp8_fp4_mqa_logits` / `top_k_per_row` /
`indexer_k_quant_and_cache` are CUDA-only). So unlike the MoE op, the DSA kernel is not
competing against a heavily-optimized stock op it can only lose to — it is enabling a
capability that otherwise does not exist on the platform.

---

## 2. What was learned (the core findings)

**Finding 1 — A correct, general, stable, end-to-end custom fused kernel is achievable on
Gaudi 2 with no TPC ISA.** The `OpBackend` path makes a multi-node SRAM-resident fused recipe
load, capture under lazy, stay numerically correct (~4e-3 on real weights), generalize across
5 model families with ~zero per-model config, and serve a 397B model end-to-end (coherent,
correct, stable across multi-bucket long generation). That was the hard part, and it is done.

**Finding 2 — SRAM-residency does NOT convert to a MoE speedup, because the stock op is
already well-optimized.** This is the central, hard-won result. The fused MoE op is *slower*
than stock `mixture_of_experts` in **both** regimes (numbers in §3). The original "~99 µs
dequant-to-HBM floor" premise assumed the stock path's bandwidth round-trip was the binding
cost. In reality the stock `mixture_of_experts` is a heavily-optimized batched/grouped
intrinsic, and the custom op's *overhead* — chunked sub-chains for SRAM-residency, OpBackend
dispatch, a much larger graph — outweighs the bandwidth saving. The bandwidth floor we
designed against is not the floor the optimized stock op actually operates at.

**Finding 3 — Single-stream is host/collective-overhead-bound, not kernel-bound, so no MoE
kernel can move it.** At batch=1 the MoE compute is only **~0.16 ms of a ~127 ms decode step**
(~90% of the step is host/collective overhead). Fusing the ~0.16 ms cannot move a ~127 ms
step. **~30 tok/s is the proven sync-bound batch=1 ceiling**, and it is set by collective +
host overhead, not by the MoE op — the MoE op is *never* the bottleneck at low batch. The
batched-K variant confirmed this directly: batched and sequential K-emission both land at
~23 tok/s, so single-stream is not even purely launch-bound — the stock intrinsic is simply
faster at the compute itself.

**Finding 4 — The throughput "slowness" was a config artifact, not an op limit.** The
throughput A/B was initially run at `--max-num-seqs=16`, where stock peaks ~126 tok/s and the
fused op ~68 tok/s. But the box is *not* throughput-starved: stock at `--max-num-seqs=64`
peaks **~486 tok/s** (engine generation-throughput, warm) — *above* the ~366 tok/s the
perf-target analysis expected. The "slow" impression came from (a) max-num-seqs set too low
and (b) an iteration artifact — the validation-OFF recipe cache + `VLLM_SKIP_WARMUP` cause
on-the-fly recompiles that drag benchmark *averages* far below the warm engine *peak*. A
production serve is warm cache + pre-warm + max-num-seqs ≥ 64.
At `--max-num-seqs=128` it climbs further to **~793 tok/s** — still scaling near-linearly
(16→126, 64→486, 128→793), so the ceiling is *not* reached at 128 and the box has more
throughput headroom (c256 would go higher). Confirms throughput was config-bound, not op-bound.

**Finding 5 — The infra's value is kernels with no stock equivalent.** The corollary of
Findings 2–3 is that building a custom kernel to *beat* a well-optimized stock intrinsic is the
wrong target on this platform — you inherit the stock op's optimization debt as your bar. The
right target is kernels the platform *lacks*. That is precisely the DSA sparse-attention
kernel: op-level validated on-card (§4), no stock op to lose to, and the enabler for
long-context that dense attention cannot do at all.

**Finding 6 — fp8-compute is a memory/throughput option, not an accuracy or single-stream
win.** INT4→fp8 is double quantization with a hard ~6.4e-2 relNorm floor (measured on-device,
matched by CPU emulation; per-channel/per-group scaling does not break ~5e-2). It trades a few
percent of accuracy for half the weight HBM traffic and ~2× MME — useful as an option, not a
single-stream speedup.

### The trap log (process findings worth keeping)

Several "wins" turned out to be measurement artifacts before the honest verdict settled —
these are the diagnostic findings, not the engineering ones:

- **Stale recipe cache (cost ~10 serve cycles).** A serve config set
  `PT_HPU_RECIPE_CACHE_CONFIG=…,false,…`, which *disables* recipe-cache validation, so the GC
  blindly replayed cached Synapse recipes by config-hash without checking the current Python
  graph. A cache minted *before the fused op existed* meant every "fused-op" serve actually
  replayed a pre-fused-op **stock-MoE** recipe — the fused op never compiled. Both an early
  "Qwen coherent with the fused op" claim *and* a multi-bucket FATAL were stale-recipe
  artifacts, not the kernel. Fix: `rm -rf <recipe_cache>/*` on *every* code change.
- **GLM serve-env is the bottleneck, not the kernel.** The end-to-end GLM serve is bound by
  *pre-existing* GLM serve-env issues (a `:1.24` lib double-registration that blocks the `.so`;
  a separate lazy-numerics bug that makes even the *stock* lazy GLM baseline incoherent),
  independent of the kernel. The kernel itself is validated on real GLM weights in isolation
  (~4e-3). End-to-end coherent serving was therefore demonstrated on **Qwen-397B** (standard
  GPTQ-sym MoE, no DSA), which sidesteps all of GLM's serve-env issues.
- **Verify-relNorm can look half-OK while broken.** A bug that only corrupts when a real
  expert computes (the storage-offset trap) reads as relNorm≈0 on any rank with no routed
  expert. Synthetic offset-0 tests passed. Real-weight tests caught it. *Always validate on
  real weights, not just synthetic offset-0 tensors.*

---

## 3. Performance — the measured verdict (unambiguous)

**The fused MoE op is slower than the stock Habana `mixture_of_experts` intrinsic in both
regimes. It is not a MoE speedup. Do not enable `HPU_MOE_FUSED_OP` / the monolithic variant
for speed.**

| Regime | Fused custom op | Stock `mixture_of_experts` | Verdict |
|---|---|---|---|
| **Single-stream** (batch=1, warm) | **~23 tok/s** | **~30 tok/s** (proven ~29–34 ceiling) | **stock faster** (~1.3×) |
| **Throughput** (Qwen-397B, concurrency-16, `--max-num-seqs=16`, engine-peak warm) | **~68 tok/s** | **~126 tok/s** | **stock faster** (~1.9×) |

Context for the throughput row (Finding 4): max-num-seqs=16 is *not* the box's throughput
ceiling. Stock at `--max-num-seqs=64` peaks **~486 tok/s** warm (above the ~366 expected). The
A/B above is apples-to-apples at one (low) config; the *absolute* throughput is healthy and
config-bound, not op-bound. At `--max-num-seqs=128` it reaches **~793 tok/s** (16→126, 64→486, 128→793 — still scaling, ceiling not yet hit).

**Why stock wins (root cause):**

1. **The stock op is heavily optimized.** `mixture_of_experts` is a batched/grouped intrinsic.
   The custom op's overhead — chunked sub-chains to force SRAM-residency, OpBackend dispatch, a
   much larger captured graph — *outweighs* the SRAM-residency bandwidth saving. The "~99 µs
   dequant-to-HBM floor" premise does not hold against an internally-efficient stock op.
2. **Single-stream is overhead-bound, so fusion has nothing to bite on.** The MoE compute is
   ~0.16 ms of a ~127 ms step (~90% host/collective overhead). ~30 tok/s is the sync-bound
   batch=1 cap; the MoE op is never the bottleneck there. (The batched-K variant — which kills
   the sequential K-emission overhead — did *not* improve single-stream: batched and sequential
   both ~23 tok/s. So it is not launch overhead either; the stock intrinsic is just faster.)
3. **The routed (T=1-gated) approach gates to single-stream.** The fusion's only *theoretical*
   win is throughput@concurrency (bandwidth-bound), but the routed op gates to single-stream;
   capturing the concurrent regime would need a *monolithic* fused op — a separate build that,
   when tested, still lost to stock (~68 vs ~126).

**What IS the win:**

- The **`OpBackend` infrastructure** itself: a reusable, validated mechanism to ship
  multi-node SRAM-resident fused kernels on Gaudi 2 with no TPC ISA, captured under lazy,
  correct on real weights, general across models. This is the durable deliverable.
- **Kernels with no stock equivalent** — the DSA sparse-attention kernel (§4). There is no
  stock DSA intrinsic to lose to; the infra enables a capability the platform lacks, which is
  where a custom kernel pays off.

**fp8-compute** (separate): a memory/throughput option with a documented ~6.4e-2 double-quant
relNorm floor — not a single-stream speed or accuracy win.

---

## 4. Validated results table

### 4a. Fused MoE kernel — correctness (HW, `vllm-gaudi:next`, relNorm vs `convert_from_uint4`)

| What | Metric | Value |
|---|---|---|
| Single-expert op | relNorm | **~4.2e-3** |
| K-expert Python wrapper | relNorm | ~5.7e-3 |
| K-expert single-recipe op | relNorm + **lazy capture confirmed** | **~5.9e-3** (K=8 in ONE captured graph) |
| Batched-K single-recipe op | relNorm vs oracle + sequential | ~5e-3 (139→16 nodes, K-independent) |
| Real GLM-5.1 expert (from checkpoint) | relNorm, eager + lazy | **~4e-3** |
| SRAM residency | bf16 weight in DRAM / workspace | **none in graph dump / 0 MB** (HBM round-trip eliminated) |
| fp8-compute | relNorm floor (INT4→fp8 double-quant) | ~6.4e-2 (memory/throughput option, not an accuracy win) |

### 4b. Fused MoE kernel — generality (5/5 profiles, `test_fused_expert_general.py`)

| Profile | H | I | 2I | GS |
|---|---|---|---|---|
| GLM-5.1 | 6144 | 2048 | 4096 | 128 |
| DeepSeek-V3-ish | 7168 | 2048 | 4096 | 128 |
| Nemotron-ish | 8192 | 2048 | 4096 | 128 |
| MiniMax-ish | 6144 | 2048 | 4096 | 64 |
| odd-width + GS=32 | 5120 | 1536 | 3072 | 32 |

All dims derived from tensor shapes; a new INT4 MoE model "just works" once packed in the
proven PyTorch-row-major layout. **5/5 pass.**

### 4c. Fused MoE kernel — end-to-end serving

| What | Result |
|---|---|
| Model | `Qwen3.5-397B-A17B-GPTQ-Int4` (GPTQ-sym int4, `:next` + lazy, no DSA) |
| Fused op captured | yes — IR confirms `hpu_moe_ext::fused_moe{…}` in the decode graph |
| Coherence | correct on short prompts (France→Paris; 60mi×3h→180; primary colors) + coherent long story/code |
| Stability | stable across multi-bucket long generation (256+ tok, 0 FATAL across ~15 reqs) |
| Verdict | **correct + stable + end-to-end, but slower than stock** (see §3) |

### 4d. Fused MoE kernel — performance (the honest verdict, repeated)

| Regime | Fused | Stock | Verdict |
|---|---|---|---|
| Single-stream | ~23 tok/s | ~30 tok/s | **stock faster** |
| Throughput (max-num-seqs=16) | ~68 tok/s | ~126 tok/s | **stock faster** |
| Throughput ceiling (stock, max-num-seqs=64) | — | ~486 tok/s | config-bound, not op-bound |
| Throughput ceiling (max-num-seqs=128) | — | ~793 tok/s | still scaling (16→126, 64→486, 128→793) |

### 4e. DSA sparse-attention kernel — op-level validated on-card (the real payoff)

Both DSA ops are **op-level validated on-card** — all three gates pass; there is **no stock op
to compare against** (DSA has no Gaudi intrinsic), so this is the infra enabling a new
capability, not a comparison.

| Gate | What it proves | Result |
|---|---|---|
| Indexer recall | `indexer_topk` top-k set matches the DeepSeek/HF oracle | **0.999** (top-k Jaccard ~1.0) |
| Sparse == dense when ctx ≤ topk | `sparse_attn` reproduces dense MLA when nothing is dropped | **sparse == dense** ✓ |
| Sparse ≠ dense + matches gather oracle when ctx > topk | `sparse_attn` truly drops keys (not silently dense) and matches the gather oracle | **sparse ≠ dense, == oracle** ✓ |

Status: op-level validated; currently **dense-bypassed** in serving (`FULL_ATTENTION_FALLBACK=True`,
`DSA_READY=False` — the safety contract: never silently dense until the full path validates
end-to-end). This is the 1M-context enabler and the infrastructure's intended use.

---

## 5. Bottom line

- **Engineering achievement:** a custom, multi-node, SRAM-resident, lazy-captured fused INT4
  MoE kernel that "wasn't supported" — correct (~4e-3 on real weights), general (5/5 families),
  stable, serving a 397B model end-to-end — built with no TPC ISA on the open-source bridge.
- **Honest perf verdict:** it is **not** a MoE speedup. Stock `mixture_of_experts` is faster at
  single-stream (~30 vs ~23) and throughput (~126 vs ~68 at the same config). Single-stream is
  host/collective-overhead-bound (~30 is the sync-bound cap; the MoE op is ~0.16 ms of ~127 ms);
  throughput was config-limited (stock does ~486 at max-num-seqs=64). **Do not enable the fused
  MoE op for speed.**
- **The durable win:** the reusable `OpBackend` kernel **infrastructure**, whose payoff is
  kernels with **no stock equivalent** — the **DSA sparse-attention kernel**, op-level
  validated on-card (indexer recall 0.999; sparse==dense at ctx≤topk; sparse≠dense + matches
  the gather oracle at ctx>topk), the 1M-context enabler.

*(The box owner will append the `--max-num-seqs=128` throughput-ceiling number when that test
finishes.)*

---

## 6. GLM-5.1 lazy decode degeneration — bucket-boundary hypothesis KILLED; cause is the routed-MoE lazy capture (2026-06-09, CPU read-only investigation)

**Symptom:** lazy serve (`models/glm51ar-lazy.env`) produces correct content for ~8–10 decode
tokens then collapses to greedy repetition; eager (`glm51ar.env`) is fully coherent.

### Verdict, ranked by confidence

**(1) KILLED — decode bucket boundary at ~8–16 tokens does NOT exist (very high confidence).**
Read from the code that actually runs (image `vllm-gaudi:next`, byte-identical to `_src/vllm_gaudi`):
- Decode metadata buckets on **blocks**, not tokens. `block_size=128` for both eager and lazy
  (`--block-size 128` in both env files). `num_blocks = ceil((ctx+1)/128)`
  (`hpu_model_runner.py:2766`) ⇒ **num_blocks stays 1 for context tokens 0..127.** The first
  decode block-bucket transition is at token **128**, then 256/384 — nowhere near 8–10.
- The HPU-graph cache key is **shape-based for tensors** (`graphs.py:input_hash` →
  `(dtype, shape, view_hash)`), value-based only for Python scalars; the trimmed decode metadata
  (`hpu_model_runner.py:872` `trim_attn_metadata`) carries only `is_prompt` (const False) and
  `block_size` (const 128) as scalars — every per-step quantity (`block_usage`, `block_mapping`,
  `slot_mapping`, `positions`, `block_list`) is a **tensor hashed by shape**. So **all decode
  steps in tokens 1..127 hit the SAME captured graph `h`** (captured once on the first decode
  cache-miss, replayed thereafter; `graphs.py:686-770`). There is **no recipe swap at the
  8-token onset by construction.** Decode always uses graphs (`_use_graphs` returns True for
  decode, `hpu_model_runner.py:4640-4649`).
- Default `skip_warmup=False` and `bucketing_strategy='exp'` (`features.py:95,100`); the lazy env
  does **not** set `VLLM_SKIP_WARMUP`, so warmup runs and the (single, at bs=1) decode bucket is
  pre-captured — contradicting the "buckets compile on-the-fly" premise in the brief.

**(2) PRIME CAUSE — the routed-MoE `_routed_moe` path under lazy HPU-graph capture (high confidence).**
Two config facts the brief got wrong, both decisive:
- `glm51ar-lazy.env` sets **`HPU_MOE_ROUTED_K=8`** (NOT K=1). Eager (`glm51ar.env`) leaves it
  unset ⇒ `_ROUTED_K=0`.
- At single-stream decode `x.shape[0]==1`, `top_k==8`, so the routed gate
  `_ROUTED_K>0 and tokens*top_k<=K` (`moe_wna16_hpu.py:1108`) is `8<=8` TRUE ⇒ lazy runs
  `apply_monolithic → _routed_moe` (`moe_wna16_hpu.py:1147`), while eager runs the **stock
  dequant-all `VllmMixtureOfExpertsOpWNA16`**. **Lazy and eager execute DIFFERENT MoE kernels.**
  So "routed-MoE is excluded because K=1 = stock monolithic" does not hold for this config.

`_routed_moe` (`moe_wna16_hpu.py:790-889`) is *exactly* the failure family: its own comments
document this same "coherent first token then degeneration ('Paris, capital, capital...')"
signature from prior lazy-capture freezes — in-place `scatter_` not re-zeroing across replay
(now functional `.scatter`, lines 812-817), a stray `mark_step` freezing `active` to the capture
step (lines 819-825), and the **mandatory** post-dequant `mark_step` the stock ref op uses
(lines 852-867). The persisting symptom means a residual capture-freeze remains in this path
**and the guard that would catch it is OFF** (the lazy env sets neither `HPU_MOE_ROUTED_VERIFY=1`
nor `HPU_MOE_FUSED_VERIFY=1`; those A/B checks use `.item()`, which the code itself notes "breaks
lazy capture" — so they can't run in the captured serve anyway). Cross-check from memory: the
K=8 routed path's coherence was only ever "under test" even in eager (`glm51ar-en.env`), never
confirmed.

**(3) LOWER — generic `disable_tensor_cache=True` decode capture (medium-low).** The whole model
is `wrap_in_hpu_graph(disable_tensor_cache=True)` under lazy (`hpu_model_runner.py:836-839`).
Buffer reuse here is a plausible secondary, but the decode attn mask is rebuilt per step from the
per-step `block_usage` tensor input inside the captured forward (`_set_block_mapping`,
`hpu_model_runner.py:6972-7042`) and is shape-stable, so this is less likely than (2). The
`glm51ar-lazy-noroute.env` A/B (below) isolates it: if degeneration persists with the routed path
OFF, suspicion moves here.

### Decisive A/B tests (cheap, written; box owner runs on HPU)

1. **`models/glm51ar-lazy-noroute.env`** — lazy + `HPU_MOE_ROUTED_K` UNSET ⇒ decode takes the
   *same* stock dequant-all MoE kernel as coherent eager; the only remaining delta is lazy capture
   itself. **Coherent ⇒ cause is `_routed_moe` (verdict 2). Still degrades ⇒ generic lazy capture
   (verdict 3).** This is the single most decisive run.
2. **`models/glm51ar-lazy-abucket.env`** — lazy + routed K=8 but ONE fixed decode bucket
   (bs/block bucket pinned, `lin` strategy, warmup ON). **Onset unchanged at ~8–10 ⇒ confirms the
   boundary is irrelevant (kills verdict 1 on HW).**
3. **Per-step routing-freeze probe** (added, env-gated, default OFF = inert):
   `HPU_ROUTED_STEP_PROBE=24` makes `_routed_moe` log one line/decode-step for the first MoE layer
   (`moe_wna16_hpu.py`, `_routed_step_probe`): `x_fp` (per-step input activation), `active` (the K
   chosen experts), `out_norm`. **Grep `[routed-step]`: if `x_fp` changes every step but `active`
   / `out_norm` FREEZE after ~1–2 steps ⇒ capture-freeze confirmed.** (Both A/B envs enable it.)

### The fix (apply after the probe confirms the frozen tensor)

The class of fix is established by this file's own history: every routing intermediate in
`_routed_moe` must be **freshly materialized from the per-step `topk_ids` graph input on each
replay** (functional ops only — no in-place writes to a captured "fresh" buffer; no `mark_step`
that strands `active` in a prior segment; keep the mandatory post-dequant `mark_step`). The probe
names the specific tensor that is still frozen (candidates: `active`, `present`, `remap`,
`remapped`, `safe_w`); the fix is to rebuild that tensor with an out-of-place op rooted at
`topk_ids` (mirroring the already-fixed `present`/`remap` scatters), or—if `_routed_moe` cannot be
made replay-stable—gate `HPU_MOE_ROUTED_K` OFF for the coherence build and pursue speed via the
separately-validated fused op path. Files (line numbers post-probe-insertion):
`hpu-moe-ext/hpu_moe_ext/moe_wna16_hpu.py` — `_routed_moe` def 837-936, decode routed gate 1157,
`_routed_moe` call 1196; per-step probe def 62-106, call 936.

### 6b. A/B results + fixes applied (2026-06-09 late evening)

- **`disable_tensor_cache=False` (PT_HPUGRAPH_DISABLE_TENSOR_CACHE=0) A/B: serve 500s — verdict
  is capture-time OOM, NOT functional incompatibility.** Every request died in
  `graph.capture_begin` with `PT_DEVMEM Allocation failed for size 48MB`: dtc=False retains all
  capture intermediates and there is no HBM headroom at `--gpu-memory-utilization 0.9`. Retry
  only at ~0.8 / smaller KV if ever needed; not required if the MLA-write fix lands.
- **K=8 + MLA fix still cycles ("Paris, capital, capital") — expected, two-bug confound**: K=8
  also exercises `_routed_moe`, whose working tree at the time carried an UNVALIDATED edit
  (removed the pre-gather mark_step). That edit's freeze theory is CONTRADICTED by the
  validated-coherent Qwen-397B fused serve (identical routing→mark_step→index_select shape,
  `active` crosses the boundary, stayed live). **RESTORED the pre-gather mark_step** in
  `_routed_moe` (both boundaries, mirroring the fused wrapper + reference-op contract).
- **serve.sh gained a STALE-RECIPE GUARD**: auto-wipes `$RECIPE_CACHE/$MODEL` when any plugin
  .py/.so is newer than the newest cached recipe (validation is `,false,` for warm restarts —
  this closes the silent-stale-replay trap permanently).
- **Decisive sequence now running: (1) K=1 + MLA fix alone (clean isolation of the non-MoE
  lazy bug) → (2) if coherent, K=8 with the restored mark_step (speed path, ~9 tok/s).**

### 6c. ROOT-CAUSE FAMILY NAILED: the GLM lazy drift is LAZY-MODE EXECUTION NUMERICS, not capture (2026-06-10 overnight bisection)

Five serve-cycle A/Bs, every one byte-identical drift (`' Paris. The capital of France is
Paris. ... The of France is'`), eliminating in order:
1. **MLA-write fix** — no effect; the NEW host-side `kv_write_probe` (post-step readback of the
   layer-0 latent cache; immune to capture, unlike the in-graph ring-buffer probe which got
   eaten — its slot index was baked at capture, drain read zeros) PROVED the write path fully
   live: positions advance, slot_mapping advances, writes land, first slot never re-touched.
   Bonus signal: KV-row norms repeat with period ~7 == the output loop period.
2. **disable_tensor_cache=False** — unfittable: 3 capture OOMs (48MB contiguous / defrag fail)
   at 0.9, 0.82+K1, 0.80+K8+reserve. Lever dead on GLM's fragmented pool.
3. **Scheduler shape** — no-async + max-num-seqs-1: byte-identical ⇒ deterministic, no race.
4. **Capture/replay itself** — lazy + `--enforce-eager` (bypass_hpu_graphs, verified by ~1 tok/s
   uncaptured speed): byte-identical drift. THE ENTIRE FREEZE-CLASS FAMILY IS DEAD for this bug.
5. **Lazy collectives** — PT_HPU_ENABLE_LAZY_COLLECTIVES=false: byte-identical.

**Image bisect impossible**: `:1.24` and `:qual` both ship upstream torch 2.10.0+cpu which
Habana hard-asserts against in lazy mode — `:next` is the only lazy-capable image.

**Verdict: PT_HPU_LAZY_MODE=1 itself computes GLM-5.1's decode wrongly** (deterministic,
accumulating, looks like a shrunken/fixed visible context; coherent baseline glm51ar.env is
PT_HPU_LAZY_MODE=0). Suspect #1: the lazy decode attention path (flat_pa_mla chain) — compile
mode (lazy=0) takes a different kernel path and is coherent.

**NEXT (the proven method):** per-layer value-level LAZY-vs-EAGER dump comparison — same
prompt, same decode step (~10), HPU_LAYER_DUMP/hidden_state_probe/mha_diag under lazy=1
(uncaptured) vs lazy=0; first divergent intermediate names the op. Then patch the op's usage
(contiguous/clone around what lazy mishandles) or build the minimal upstream repro.
Note K=8 `_routed_moe` hard-cycles from token 2 under lazy with EITHER mark_step arrangement —
possibly the same lazy-numerics root expressed harder; fast-path fallback remains
HPU_MOE_FUSED_OP=1 (fused single-recipe tail) once K=1 lazy is coherent.

---

## 7. DSA SPARSE IS LIVE END-TO-END (Step 4 / MVP-3 serve-level gates — 2026-06-10)

**The DSA sparse OP path served GLM-5.1 end-to-end on HPU**: `USE_DSA=1 ./serve.sh glm51ar-dsa`
(`models/glm51ar-dsa.env` = :next EAGER + `HPU_DSA_SPARSE=1` + `HPU_DSA_FIRED_LOG=1`,
max-model-len 4096 > index_topk 2048, own recipe-cache dir). All serve-level gates pass:

| Gate | Evidence | Verdict |
|---|---|---|
| (a) indexer fires | `indexer FIRED (op)` log lines on all 8 TP workers, prefill + per-decode-step, real batches | ✅ |
| (b) ctx ≤ 2048 ≡ dense | short-prompt completions match the dense baseline's drift patterns (100°C / car-speed near-identical); consume short-circuits to the genuine dense forward by construction | ✅ |
| (c) ctx > 2048 sparse live | 3051-token needle prompt: full prefill+decode through `indexer_topk`+`sparse_attn`, no crash, on-topic document-aware completion | ✅ |

**Needle-recall honesty**: the 3051-token sparse needle was MISSED — but the DENSE control
(same needle, 1889 tokens, below threshold ⇒ dense path on the same serve) misses it the
SAME way ("the secret access code is 10"-style on-topic confabulation). So the miss is the
int4 checkpoint's long-context quality floor under greedy (consistent with its short-prompt
drift), NOT a sparse-wiring bug. Numerical correctness rests on the on-card gates (sparse ==
gather-oracle 0.003 at 16K, GLM dims) + dense-equivalence in serve.

### The four bugs that had to fall (all fixed, all re-validated)

1. **Lazy .so output dtype** — `BuildCast(Long→Int)` after the topk node lowers to the
   degenerate `cast_i32_to_i32` GUID (torch Long is i64-EMULATED-AS-i32 on device); the lazy
   GC refuses it (`cast_i32_to_i32 cannot be used, from=Long to=Int`), eager folded it.
   FIX: declare the topk index output `Int` directly at the synapse level, NO cast node
   (the bridge's own topk_gen.cpp pattern). `dsa_indexer_backend_shared.h` NODE 6/7 →
   single final node. **Lazy .so now passes all 3 on-card gates at GLM dims** (was: crash).
2. **Dynamo shape-agnostic recipe cache vs custom ops** — vllm-gaudi compiles the model with
   torch.compile; the HPU backend caches recipes SHAPE-AGNOSTICALLY and re-runs Synapse shape
   inference on a shape-changed hit. Custom OpBackend recipes (and some fragment graphs) have
   no SIF ⇒ `[SHAPE AGNOSTIC] Cache hit Synapse shape inference failed!` worker death.
   Disabling just the op call is NOT enough — dynamo compiles the resume-fragments around it.
   FIX: the whole indexer (`_hpu_indexer_forward`, `_op_forward_oot`, `_op_score_segment`)
   and the sparse consume body (`_op_consume_sparse_or_none`) are `@torch.compiler.disable`
   EAGER ISLANDS; the dense fallback delegate stays compiled.
3. **Short-row top-k tail = future-token leakage** — the kernel contract ("-1 right-pad lives
   OUTSIDE the op") was not honored by the serve path: rows with fewer causal keys than kk
   get -inf-masked (non-causal/pad) indices in their top-k tail, and the consume masks ONLY
   -1 ⇒ those keys would be gathered and attended. The on-card gates COULDN'T catch it
   (gate (c) is full-causal; the oracle consumed the same indices). FIX: causal post-filter
   in `_op_score_segment` (`idx > abs_pos → -1`). Every prefill row t < kk was affected.
4. **topk CGUID kk power-of-two constraint** — `kk = min(index_topk, S_pad) = 1920` ⇒
   `Invalid Node Params! Node name: n_ao` synStatus 26 at graph compile (kk = 128/2048 fine).
   FIX: when S_pad < index_topk, pad the key axis to the next power of two so kk is pow2
   (`dsa_ops.indexer_topk_op`); regression-tested at S=1900 on card.

Plus serve plumbing: 3D fake-tensor slice trap in the HPU-safe `Indexer.forward` (slice last
dim + flatten before k_norm — the documented hpu_mla_skip_indexer trap, inherited);
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` (default 300s kills first-request eager compiles);
key-axis 128-padding to bound per-decode-step recipe mints.

### Status / what remains for DSA

- **Eager serve-level: VALIDATED** (this section). `glm51ar-dsa.env` is the reference config.
- **Lazy op-level: VALIDATED** (all 3 gates, GLM dims). **Lazy serve-level: NOT yet** — the
  indexer/consume python (per-request loops, .item()s, eager islands) is incompatible with
  lazy HPU-graph capture as-is; the production lazy serve runs the dense bypass
  (`HPU_DSA_SPARSE` unset ⇒ inert, unchanged graph). Lazy+DSA serving is the remaining
  engineering for 1M-context production (capture-friendly indexer scheduling or
  decode-bucketed sparse recipes).
- max-model-len 4096 was the validation scope; longer ctx scales the same shape-driven math
  (on-card gate (c) at 16K passes), gated on serve memory headroom.

## 8. MTP (Step 5) — the int4 checkpoint's MTP layer is DEAD; serve stack exonerated (2026-06-10)

Spec-decode acceptance was exactly 0%. The hunt found ONE real serve bug and one
checkpoint verdict:

- **REAL BUG (fixed, hpu-moe-ext `mtp_share_head_patch`):** `DeepSeekMTP.load_weights`
  skips every weight outside the spec layer, and GLM-5.1 ships NO `embed_tokens` /
  `shared_head.head` inside layer 78 (shares the target's by design; DeepSeek-V3 ships
  its own copies). The draft ran with a RANDOM embedding + RANDOM head. Fixed by tying
  both to the target's modules when the checkpoint index omits them.
- **CHECKPOINT VERDICT (fp64 oracle + 7-variant wiring sweep, `glm_cpu_mtp_oracle.py` /
  `glm_cpu_mtp_variants.py`):** even computed in fp64 on CPU straight from the
  checkpoint, layer 78 outputs constant junk ('odesk'/'lop', 0/47 agreement with the
  target) under EVERY plausible wiring (cat order, norm assignment, input_ln, positions,
  head norm). The AutoRound int4-mixed export quantized layer 78 to int4-EVERYWHERE
  (main layers keep int8 attn) and calibration data never flows through the MTP layer —
  the export shipped a dead draft head. The serve faithfully computes what the
  checkpoint contains (probe-confirmed: same constant token serve-side and oracle-side).
- Also landed en route: layer-78 repack-cache mint flow (`glm51ar-mtp-mint.env`, eager,
  rank{0..7}_layer78 files — the lazy live-repack fallback for uncached layers remains
  unvalidated); `mtp_probe.py` (HPU_MTP_PROBE=1) with manual-eager stage re-execution
  (module hooks are useless on the captured draft).

**The way forward (filed):** graft the ORIGINAL bf16 layer-78 weights
(zai-org/GLM-5.1 ships them in 4 shards, ~21GB, model-0027{1..4}-of-00282) as a bf16
draft on top of the int4 target — draft-only cost ~2.5GB/card. Validate on CPU first
(bf16-78 + cached fp64 hiddens -> agreement), then the serve graft. Until then the MTP
lever (~33-38 tok/s) is CHECKPOINT-BLOCKED, not serve-blocked; single-stream stays at
the ~14 warm / ~20-23 ceiling.

**Fabric caveat (same day):** card 0000:43:00.0 Port 9 link-flapped during the restored
production serve's benches (multi-card retrains in dmesg) -> 2.8-5 tok/s measurements.
Coherence correct, fused-K8 tail engaged, 1045 recipes warm. 14 tok/s was measured on
this same boot at quiet fabric; re-bench when quiet. Known reseat-responsive defect.


### §8 addendum (2026-06-10 ~10:07) — MTP LANDED via bf16 graft
`mtp_bf16_graft.py` (HPU_MTP_BF16_GRAFT, commit 244f180): live SpecDecoding acceptance
**62.5–75%** (was 0%), mean acceptance length ~1.7, target coherent. CPU bf16 oracle 62%
first => wiring was always right; only the int4 export's layer-78 weights are dead.
Production-speed MTP config (graft + K=8 fused + full warmup) pending quiet fabric.

## 9. Step 6 (Qwen pass) HARDWARE-BLOCKED: fabric link storm (2026-06-10 ~10:30)

Card 0000:43:00.0 Port 9 began flapping ~09:36 and escalated to a sustained storm
(~2,700 link down/up events per 10 minutes across cards). Consequences observed:
- GLM production benches read 2.8-5.1 tok/s (coherent, fused engaged, warm cache —
  collectives stalled; 14 tok/s was measured on this same boot pre-storm).
- Qwen3.5-397B TP=8 CANNOT BOOT: VllmWorker-7 dies during init (c10 error in
  HCCL/collective bring-up through the flapping links), twice in a row. Stopped the
  crash loop (container churn in a broken fabric = the affinity-deadlock trap);
  cards left idle = the safe state.

Step 6 plan (ready to execute once the fabric is reseated): coherence-vs-checkpoint
FIRST via a lazy-vs-eager A/B on the serve itself (greedy completions byte-compare +
prompt_logprobs of a fixed passage; Qwen is a hybrid linear-attention model — 45
DeltaNet + 15 full-attention layers — so a CPU fp64 oracle is impractical, and the
serve A/B is the rigorous equivalence test). Then warm single-stream (~29-34 known),
throughput re-check (486 @ max-num-seqs 64 proven), production lock-in. MTP for Qwen
stays hybrid-blocked (multi-group KV block table).

**The box needs the physical OAM reseat (gaudi-fabric-defect-rma) before any further
TP=8 work.** Step 7 status: GLM production configs ARE locked (glm51ar-lazy =
fused-K8 baked + warm cache + Tailscale/api-key/no-autostart; glm51ar-mtp = bf16
graft baked, 62-75% acceptance); Qwen lock-in pending the reseat + Step 6 gates.

### §9 addendum — Step 6 partial results on the quiet quad (2026-06-10 ~11:00)

The fabric storm (now ~6-8k link events/10min, modules 5+2) blocks TP=8, but the
QUIET cross-socket quad {0,1,3,4} works — Qwen TP=4 serves on it:

- **Qwen3.5-397B EAGER TP4 = VALIDATED COHERENT** (qwen35-397b-ss.env, quad-pinned):
  'Paris.', clean <think> reasoning trace, correct 180-miles arithmetic, correct
  recursive fibonacci. This is the coherence REFERENCE leg.
- **Qwen LAZY TP4 = blocked by a REPRODUCIBLE load FATAL** (2 identical crashes,
  Tensorid 2711519): `ValidateSyncInputTensors tensor_data is empty` at the load-time
  hpu.synchronize — the free_inplace freed-storage class, in the lazy GPTQ
  repack/load path. Deterministic software bug (NOT fabric); needs its own debug
  pass (suspects: plugin changes since 06-05 — deq_weight_move / pad-fix / held-leaf
  edits — or a TP4-vs-TP8 lazy load difference).
- **prompt_logprobs CRASHES the HPU engine** (EngineCore death on request) — never
  use it on this build; greedy byte-compare is the A/B instrument.
- Speed numbers under the storm are meaningless; the lazy-vs-eager A/B completion
  compare + all benches resume post-reseat (task list).

## 11. SPEED RE-AUDIT (2026-06-10, software-only mandate, fabric storming) — two measured results

**(a) The "~23 ceiling / 90%-collective / 0.16ms MoE" (Finding 3) was measured on the WRONG path.**
The on-disk Kineto traces (`logs/prof/gaudi-server_*.pt.trace.json.gz`, 8 ranks, 2026-06-08)
attribute the decode step directly: ProfilerStep ≈ **193 ms (~5.2 tok/s)**; `hcclAllReduce` =
**~9.8 ms/step (~5%, 157 all-reduces)**; device **84% busy**; **~98% of device time = two
`fused_kernel_*` MoE kernels, ~345k tiny (~13µs) launches/step computing ALL 256 experts/layer**
(~4,608 kernels/layer ÷ 256 ≈ 18 chunks/expert). That trace is the **monolithic all-256 path**
(MoE-COMPUTE-bound, NOT collective-bound). The 0.16ms/90%-collective figure came from the stock
`mixture_of_experts` intrinsic, which CYCLES for GLM and never serves it. The production path
(`HPU_MOE_FUSED_OP=1 + HPU_MOE_ROUTED_K=8`) is **sparse-8** (14 tok/s); at sparse-8 the MoE work
is small so the base reverts to per-token collective+host overhead. Net: a dense 120B fp8 *would*
be multiples faster — it avoids all-256 blowup, ~1M tiny int4-dequant kernels, and EP all-to-alls.

**(b) MTP depth-2 RULED OUT (CPU oracle `tests/glm_cpu_mtp_depth2.py`).** a1(depth-1)=61% (matches
the validated graft), a2(depth-2 | d1 accepted)=**32%** — the single grafted layer-78 degrades hard
on its own hidden. E[len] 1.61→1.80 = **1.12× more tok/step, before** the extra 2-tok verify + 2nd
draft forward → net ~flat. KEEP `num_speculative_tokens=1`. MTP single-stream tops at ~1.6× → the
14 base gives **~23 tok/s**.

**Boundary (software-only, no fabric/ports):** GLM single-stream maxes at ~23 (MTP depth-1, built in
`glm51ar-mtp-prod.env`, draft CPU-validated, unmeasured in prod config ONLY because of the storm).
Beating 30 needs the BASE (14) lifted, and the base is collective+host-bound = the FABRIC. The two
base-lift software levers are both blocked from validation now: (1) stock-routed-8 op fix is flagged
a *capture* bug (CPU-irreproducible → device-only → storm-blocked); (2) EP-off / TP-topology reshape
is a config experiment that needs a device + fit check. **Qwen-397B is at its software ceiling**
(~30 single / 486–793 concurrent; MTP hybrid-blocked). Conclusion: once the fabric is healthy
(base ~18–20), the already-staged MTP config gives ~30–32 automatically; until then ~23 is the
software max and 30+ is fabric-gated by construction of the "no-fabric" constraint.

## 12. ON-DEVICE MTP VALIDATION (2026-06-10, clean window → storm) — software win proven

Cleared a wedged Qwen container (it was failing on an **HBM OOM in warmup**: `PT_DEVMEM
Allocation failed 7MB` in graph.replayV3 — a CONFIG bug, NOT the fabric), then brought up
`glm51ar-mtp-prod` in a clean 0-flap window (cards freed, 768MB idle). Loaded 76 shards +
FULL warmup (16 prompt + 36 decode buckets), HEALTHY in ~8 min, NO OOM. The ~8-min warmup
consumed the clean window — storm resumed (~766 flaps/min by serve time).

MEASURED:
- **Coherence: GOOD** — planets all 8 correct then checkpoint-drift; "Paris..." correct then
  the known int4 greedy loop. Matches the teacher-force oracle (drift == checkpoint, serve stack clean).
- **MTP acceptance length = 1.67–1.70, per-position 70.5%** — FABRIC-INDEPENDENT. This is the
  speedup MULTIPLIER, proven on real hardware: MTP delivers ~1.7 tokens/forward step.
- **Absolute speed = 2.15 tok/s @ 766 flaps/min** — pure storm degradation (matches the documented
  2.8–5 under-storm regime; 14 was same-boot at quiet fabric). NOT a software figure.

CONCLUSION (measured, not modeled): the MTP software speedup is DEPLOYED + on-device-verified
(1.70× multiplier, coherent). At quiet fabric: 14 base × 1.70 ≈ **~24 tok/s**. To clear **30**
the BASE must rise 14→~18 — that is the stock-`mixture_of_experts` base-lift lever (Finding 2:
stock ≫ fused), which needs a clean window + a cold-cache bring-up to test (the storm took this
window). 30+ = base-lift(software, untested) × MTP(1.70, proven), OR quiet-fabric base × 1.70.

### §12 addendum — storm empirically denies clean measurement (2026-06-10 ~17:30)
Tried to measure clean-fabric MTP speed on the warm serve with ZERO churn: one 128-tok bench
(2.25 tok/s, 269 flaps mid-bench) + 20 short 24-tok benches gated on tight 6s windows. Result:
NO sustained 0-flap window exists under the current storm — best reading 8.33 tok/s with 2 flaps
during the bench; a window that sampled 1 flap/6s took 94 flaps mid-bench → 1.11 tok/s. Acceptance
length stayed 1.61–1.70 throughout (fabric-independent — the MTP multiplier is solid). CONCLUSION:
the MTP software win is deployed + on-device-verified (1.70×, coherent); the absolute clean number
(~24 = 14 base × 1.70) is grounded in the prior same-boot quiet-fabric 14 + the verified multiplier,
but a fresh clean reading is HARD-blocked by the storm. 30+ needs the stock-op base-lift tested in a
genuinely quiet window (tray) — untestable now without churning into a storm. Serve left UP (warm,
coherent, MTP-verified) so a clean reading / base-lift test fires instantly when the fabric calms.

## 13. MiniMax-M3 / MSA — second sparse-attention model: the DSA kernel went DYNAMIC (2026-06-14)

CARD-FREE investigation + design pass for **MiniMax-M3** (428B MoE, GQA, 1M context). The headline:
the DSA OpBackend sparse-attention work **carried almost entirely** to M3's MSA — proof that the infra's
real payoff (Finding 5) is *reusable* kernels with no stock equivalent, not one-off builds. M3 modeling
landed in **transformers ≥5.12.0** (PR #46600); scratch venv `$HOME/.venvs/m3calib`. Serving tok/s
are **PENDING BENCHMARK** (matrix in `minimax-m3/bench/run_matrix.sh` + `aggregate_matrix.py`); the
*mechanisms* below are oracle/code-validated, the *numbers* are not.

### 13a. MSA architecture (validated against the OFFICIAL modeling, NOT the paper)

MSA = **block-sparse top-16×128-block attention over GQA (64q/4kv/d128) with REAL uncompressed K/V** —
a SIMPLER cousin of DSA (block/GQA vs DSA's token/MLA; the indexer compresses *scores*, never K/V).
Layers **0–2 dense, 3–59 sparse** (same 3-dense prefix as the MoE split). Index branch per query:
**per-head Gemma-RMSNorm `x*(1+w)` → partial RoPE (rot64/θ5e6, same cos/sin as main) → RAW matmul (NO
1/√d) → causal mask → block-max-pool → `amax` over the 4 index heads (HEAD-COLLAPSE) → +inf force-local
(1 block, 0 sink) → top-16 → -1 pad**. Head-collapse means **ONE shared block set per query, every GQA
group reads the same ~2048 tokens → a SINGLE consume call** (NOT a per-group 4-call loop). Validated:
`ref/msa_reference.py` reproduces the real `MiniMaxM3VLIndexer` block_scores to |Δ|≤4.8e-07 and
**selection Jaccard 1.0** (`tests/test_msa_hf_parity.py`); the wire reproduces the oracle bit-for-bit.

> **Trap that nearly shipped:** the paper-derived oracle had **no index rope, plain `x*w` norm,
> `1/√d_idx` scale, per-GQA-group selection** — all WRONG. A self-consistent oracle agreed with *itself*
> while both diverged from the real checkpoint (the SET-match gate compares to the same-flavored oracle,
> so rope/norm errors are invisible to it — visible only at real-context eval). Caught by reading the
> runnable modeling verbatim AND decoding the stored `index_{q,k}_norm.weight` (cluster near 0, some
> **negative** → impossible for `x*w`, exactly Gemma `x*(1+w)`). **Lesson: pin the index/selection branch
> to runnable modeling code + stored-weight statistics, never the paper.** (`notes/reviews/index-norm-reconfirm.md`.)

### 13b. The kernel went DYNAMIC — DSA + MSA in one OpBackend family

One sparse-attention op family now handles **DSA (token/MLA) + MSA (block/GQA)** via an `MsaDims`/
`DsaDims` config reader + a `mode` flag (`MLA_TOKEN_SPARSE` vs `GQA_BLOCK_SPARSE`):

- **Consume reused BYTE-IDENTICAL.** MSA's gather+SDPA consume = the *same* `sparse_attn` op DSA built,
  fed expanded indices + real GQA K/V, `scale = head_dim**-0.5`. Zero graph change (DSA's MLA latent
  absorption is simply skipped — MSA consume is strictly simpler). GQA `H_kv=4` → expanded to H upstream
  by the SAME `expand` step DSA used for its single MQA indexer head.
- **Indexer added only TWO nodes:** block-max-pool = `Reshape([Sq,n_blk,block_size])` +
  `reduce_max_multi_dim_fwd` (`amax`/`AminAmax` lowering; `ns_Reduction::ParamsV2`, same `FillReduction`
  bit formula DSA's head-sum uses), inserted between the existing causal-mask and topk nodes. Block→token
  expansion is composed `range`+`broadcast`+arithmetic (upstream Python for MVP, in-graph later).
- **Missing bridge GUIDs: NONE.** Every MSA primitive is a real GC-lowered node in the version-exact
  bridge (`v1.24.0-1007`) — no new TPC ISA. All 4 DSA serve-bug fixes (§7) pre-pay for MSA: Int topk
  output (no `cast_i32_i32`), `@torch.compiler.disable` eager-islands, causal post-filter on the topk
  tail, pow2-`kk` key-axis pad.

Build order carried 1:1 from DSA: CPU parity oracle → eager `.so` → lazy `.so` (shared `*_shared.h`) →
serve gated `MSA_READY=False` + `MSA_FULL_ATTENTION_FALLBACK=True` (genuine dense GQA, loud, never
silent-empty-topk). Wire target is the **GQA** `HPUAttentionImpl.forward` (not the MLA `forward_impl`);
new plugin `hpu-msa-ext` mirrors `hpu-dsa-ext`, entry point `06.` (chains after DSA's `05.` if both
mounted). Speed levers still open (op-validation not yet run on-card): single-call consume on the 4 REAL
KV heads (drop any 16× `repeat_interleave`); fp8 index logits + `masked_batch_gemm` for prefill-TTFT at
128K–1M. Notes: `KERNEL_MAP.md`, `M2_BASE_MAP.md`, `MSA_SPEC.md`, `notes/reviews/build-msa-kernel.md`.

### 13c. Precision truth reconfirmed; fp8 experts (not int4) for a bf16-source MoE — GO

The §11a dequant tax has a hardware root: **Gaudi 2 MME is FLOAT-ONLY (bf16+fp8), NO integer matmul** —
int4/6/8 ALL dequant to bf16 before the GEMM. So **int4 = memory play, fp8 = native speed**. The fix is
NOT a fused int4 kernel (Finding 5) — vllm-gaudi **already ships a native fp8 MoE path**:
`VllmMixtureOfExpertsOpFP8PerChannel` feeds fp8 weights+activations+scales straight into the
`mixture_of_experts.fp8` intrinsic, **no bf16 round-trip**, gated `VLLM_HPU_FORCE_CHANNEL_FP8` (defaults
True). Serve fp8 from M3's bf16 source via **INC** (`--quantization inc`, MEASURE→QUANTIZE; INC
requantizes bf16→fp8 in place). Quality = **<1% single-quant** (NOT the ~6.4e-2 floor — that's the
int4→fp8 *double*-quant of the hpu-moe-ext scaffold, the wrong tool here). **Verdict GO** for fp8
experts: at 428B/768 GB M3 is NOT memory-forced into int4 (fp8 ≈471 GiB fits with full c=30 KV headroom
to ~128K), so int4's only edge (KV room) doesn't pay until far above c=30 while its dequant tax hurts
every decode step. Bonus: **fp8/bf16 keeps the 7 MTP modules alive** (int4-everywhere killed GLM's MTP
head → bf16 graft, §8); fp8 leaves them bf16 → no graft. (`notes/reviews/fp8-moe-gate.md`,
`build-weights-precision.md`.)

### 13d. The memory triangle + KV/offload constraints (768 GB)

Can max only **2 of {fp8 per-token speed, 128K+ ctx, 30+ conc}**. fp8 weights (~471 GiB) hold 30+conc
only to ~66K; int4 weights (~285 GiB) to ~150K. **fp8_inc KV is sufficient ≤128K@30** but budget
**74.25 KB/tok, not 60**: main GQA K/V 60 KB/tok + the **MSA index-K cache 14.25 KB/tok** (57 sparse
layers × 1 head × 128, **bf16 ALWAYS — un-quantizable**, separate `FullAttentionSpec`). Knee = conc 30 ×
ctx ≈ 144–150K (~4.4 M tok). **KV compression:** only `fp8_inc` passes the platform gate; sub-fp8
(TurboQuant/Kitty) is CUDA-only → must build as OpBackend + gate extension. If built, **TurboQuant
`k8v4`** (fp8 K + 4-bit V, Hadamard, calibration-free) is the near-lossless knee (CPU-parity V-err 0.11);
2-bit is garbage (0.75). Sub-fp8 KV is a **memory play (~1.3–2× conc), NOT faster/better** — build only
if a measured run proves HBM-KV-bound. **1M offload:** the index-K cache is full-scanned every step →
MUST stay HBM-resident exact; only main K/V offloads (MSA reads ~2048/1M tokens/step) via the existing
`vllm_gaudi/v1/kv_offload` transport + a `block_table` seam (no new TPC kernel) + step-ahead prefetch.
**1M offload REQUIRES int4 weights** (resident ~230 GB index + double-buffer fits the int4 ~340 GB
envelope, NOT fp8's ~150 GB → fp8×offload@1M OOMs). (`notes/reviews/build-kv-envelope.md`,
`quant-kv-turboquant.md`, `offload-design-SYNTHESIS.md`.)

### 13e. Quantizing a >200B model on this box (the recipe)

The bf16 M3 (~856 GiB) does NOT fit ~448 GiB free RAM → don't load it. **AutoRound model-free RTN**
(`ModelFreeCompressor`, `auto_round 0.13.1`): given the model PATH string, streams shards one at a time
(1-deep prefetch + async write), per-tensor RTN, packs `auto_round:auto_gptq`, `clear_memory()` per
shard → **peak RSS ~17 GiB (one shard), never 856 GiB**. No forward pass, no AutoModel, **no runnable
modeling code needed — only config.json + shards**. Contract = routed experts int4 / all else bf16 (the
GLM "int4-mixed" contract hpu-moe-ext autodetects via stored `.qweight` = ground truth, not the
mislabeled `extra_config`); `iters=0` RTN. Two model-free `layer_config` traps (both seen as "Quantized
layers (0)"): keys ending in a digit get "." appended → expert `w1/w2/w3` need a `$`-anchor; a bare
`block_sparse_moe` bf16 prefix regex-swallows experts → order expert int4 keys FIRST + drop swallowers.
Script: `minimax-m3/quant/quantize_m3_int4.py` (`--dry-run` is stdlib-only, header-parse, card-free).

### 13f. M3 status

- **Spec/oracle: VALIDATED** — index/selection Jaccard 1.0 vs transformers 5.12.0; wire == oracle
  bit-for-bit (prefill-first; paged-decode index-cache sync is a follow-up).
- **Kernel: DESIGNED + both `.so` built, on-card op-validation NOT yet run** (`MSA_READY` closed, dense
  GQA fallback live). `hpu-msa-ext` plugin + the `HPUAttentionImpl.forward` GQA wire are the build delta.
- **Serve: phase-1 dense GQA bf16** port of `minimax_m2.py` (text-first, skip vision tower; read
  `text_config`; dense-first-3 + per-layer MoE-freq + shared expert + `swigluoai`; register
  `MiniMaxM3SparseForConditionalGeneration`). fp8-experts via INC is the production weights path.
- **PENDING BENCHMARK:** all serving tok/s (matrix in `minimax-m3/bench/run_matrix.sh`).
