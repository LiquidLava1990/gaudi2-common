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
*(The `--max-num-seqs=128` throughput-ceiling number is pending and will be appended by the
box owner when that test finishes.)*

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
config-bound, not op-bound. *(max-num-seqs=128 number pending — appended by the box owner.)*

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
| Throughput ceiling (max-num-seqs=128) | — | *pending* | *appended by box owner* |

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
