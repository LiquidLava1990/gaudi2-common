# Gaudi 2 Fused INT4 MoE Kernel — Architecture (the how & why)

Status: **built + hardware-validated** (relNorm ~4e-3 vs the trusted `convert_from_uint4`
reference; lazy HPU-graph capture confirmed; generalizes 5/5 across the MoE families
tested; serves Qwen-397B end-to-end). Code is stable; this document is final. The companion
[`KERNEL_TECHNICAL.md`](KERNEL_TECHNICAL.md) is the granular, code-driven walkthrough
and the "how to reproduce / extend" reference.

> ⚠️ **Performance verdict (read [§9](#9-performance--the-measured-verdict) first):** this
> kernel is **correct, general, stable, and serves end-to-end — but it is NOT a MoE speedup.**
> The stock Habana `mixture_of_experts` intrinsic is faster in *both* regimes (single-stream
> ~30 vs ~23 tok/s; throughput ~126 vs ~68 at the same config). **Do not enable the fused MoE
> op for speed.** The bandwidth-floor premise in §1/§2 below is the *design motivation*; it
> did not survive contact with the (heavily-optimized) stock op — see §9 for why, and for the
> real payoff (the reusable `OpBackend` infrastructure → kernels with no stock equivalent,
> i.e. the DSA sparse-attention kernel). Full project narrative:
> [`PROJECT_FINDINGS.md`](PROJECT_FINDINGS.md).

All file:line citations are to the source tree under `hpu-moe-ext/tpc_kernel/opbackend/`
(copied verbatim into `publish/gaudi-fused-moe-kernel/`). The kernel is built against the
open-source Habana bridge **HabanaAI/gaudi-pytorch-bridge tag `v1.24.0-1007`** (==
`habana-torch-plugin 1.24.0.1007`, torch 2.10, Apache-2.0).

---

## 1. The problem — a hard ~99 µs/token bandwidth floor

Gaudi's only mixture-of-experts intrinsic, `torch.ops.hpu.mixture_of_experts`, is
**bf16-only**. Every INT4 (GPTQ/AWQ) MoE path on the platform is therefore forced into a
two-stage shape:

1. **Dequantize every expert** to bf16 with `torch.ops.hpu.convert_from_uint4` and **write
   the full bf16 weight to HBM**.
2. Feed those materialized bf16 weights to the separate `mixture_of_experts` op.

At single-stream decode (batch = 1) this is **pure bandwidth waste**. The dequantized bf16
weight is read once by the GEMM and thrown away; the round-trip to HBM is the entire cost.
For a GLM-5.1-class layer this is on the order of **~243 GB/rank/token** of weight traffic —
a **~99 µs/token floor**, roughly 50× the bandwidth the actual GEMM needs.

> **⚠️ This was the design hypothesis, and it did not hold (see [§9](#9-performance--the-measured-verdict)).**
> The "~99 µs dequant-to-HBM floor" assumes the stock path's bandwidth round-trip is the
> binding cost. It is not: the stock `mixture_of_experts` is an internally-efficient
> batched/grouped intrinsic, and at single-stream the *whole* MoE compute is only ~0.16 ms of
> a ~127 ms decode step (the step is ~90% host/collective overhead). Eliminating the round-trip
> is real, but it does not move a step that was never weight-bandwidth-bound in the first place.
> The fused kernel below is correct/general/stable, **but it is slower than stock** — §9.

The stock plugin path that exhibits this (`MoeWNA16HPUMethod._routed_moe`) is documented
in `hpu_moe_ext/moe_wna16_hpu.py:594-692`: a per-expert `convert_from_uint4` list → a
**mandatory `mark_step()`** (which forces the bf16 weights to materialize to HBM) → the
`mixture_of_experts` intrinsic.

## 2. The solution — one fused, SRAM-resident per-expert recipe

Replace the two-stage round-trip with a **single fused per-expert graph** in which the
dequantized bf16 weight **never touches HBM**. The full expert forward —

```
reinterpret int32→uint4  →  dequant W13→bf16  →  MME GEMM (x·W13)
   →  slice gate/up  →  SiLU(gate)·up  →  dequant W2→bf16  →  MME GEMM (h·W2)
   →  × router_weight
```

— is emitted as **one node chain**, captured **once** by the Graph Compiler (GC) and
**replayed per token**. The two dequant outputs (`W13_bf16`, `W2_bf16`) are declared as
**graph-internal, non-persistent** Synapse tensors. That single property is the mechanism:
the GC's own shipped fusion passes (`fuseDequantIntoMoeGemm`, the TPC-producer → MME
bundler, the SRAM slicer) bundle each dequant **into the MME that consumes it**, so the
bf16 weight is produced **in SRAM and consumed in SRAM** — the HBM round-trip is *eliminated
by construction*. On device this shows up as a graph dump with **no full bf16 weight in DRAM**
and **0 MB workspace** for tens of MB of bf16 weight (the SRAM-residency signature).

> **Note (perf):** the round-trip really *is* eliminated — that part is verified on-device.
> What it does **not** do is make the op faster than stock: the SRAM-residency saving is
> outweighed by the custom op's overhead (chunked sub-chains, OpBackend dispatch, a much
> larger captured graph), and single-stream is overhead-bound regardless. The win here is a
> *correct, general, captured, SRAM-resident recipe* and the reusable mechanism to build one —
> not MoE tok/s. See [§9](#9-performance--the-measured-verdict).

### Why this is feasible on Gaudi specifically

- The **MME** (matrix engine) does the two GEMMs natively, in bf16 (and natively in fp8 —
  see §6).
- The **TPC** does the INT4 dequant and the elementwise SiLU.
- The GC already ships the producer→MME bundling + SRAM-slicing passes. So SRAM residency
  is **a pipeline-pass decision on a section-less intermediate**, not a hand-written tiling
  kernel. We do **not** write any TPC ISA — every node is a stock bridge GUID
  (`reinterpret_cast`, `dequantize_4_bit_bf16`, `gemm`, `slice`, `sigmoid_fwd_bf16`,
  `mult_fwd_bf16`, `add_fwd_bf16`). The contribution is **the graph and its geometry**, not
  a new compute kernel.

### Why an OpBackend (and not the public custom-op API)

The bridge's public `habana::custom_op::UserCustomOpDescriptor` binds **one** TPC GUID with
one shape/params function — it cannot express a multi-node recipe (2× reinterpret pairs, 2×
dequant, 2× MME, 2× slice, SiLU, router mult) nor the non-persistent dequant→MME bundling.
So instead we register a full **`habana::OpBackend`** subclass in the bridge's
`KernelRegistry`, whose `AddNode()` chains `OpBackend::BuildNode()` calls. This is exactly
how the bridge ships its **own** multi-node ops (modeled verbatim on
`mixture_of_experts_gen.cpp` + `convert_from_int4_gen.cpp` + `full_gen.cpp` from this tag).
Because the op is captured into the HPU graph like any built-in, **the bridge owns the
launch/stream/workspace and maps tensors itself** — there is no side-band `synLaunch`, no
manual `data_ptr()` binding (which fails Synapse section validation), and the recipe is
durable across HPU-graph capture / lazy replay. This was a deliberate architecture choice
after a `synLaunch`-based approach hit a `data_ptr`-vs-Synapse-VA wall (see
`KERNEL_TECHNICAL.md` §8 history).

## 3. The node graph (single expert)

```
 INPUTS (all PyTorch row-major; last dim == Synapse FCD; shapes derived from tensors)
 ─────────────────────────────────────────────────────────────────────────────────────
   x          bf16  [M, H]          hidden states (M = tokens; single-stream M=1)
   w13_packed i32   [H, 2I/8]       packed gate||up weight (8 int4 nibbles / int32)
   w13_scale  bf16  [H/GS, 2I]      gate||up group scales (GS = group size)
   w13_zp     i32   [H/GS, 2I/8]    gate||up zero-points (packed like the weight)
   w2_packed  i32   [I, H/8]        packed down weight
   w2_scale   bf16  [I/GS, H]       down group scales
   w2_zp      i32   [I/GS, H/8]     down zero-points
   router_w   bf16  [1, 1]          this expert's scalar router weight (folded)

   w13_packed ─reinterpret_cast→ uint4 ┐
   w13_zp     ─reinterpret_cast→ uint4 ┤
   w13_scale  ─────────────────────────┴─► dequantize_4_bit_bf16 ─► W13_bf16 [H, 2I]   ░ NON-PERSISTENT
                                                                          │             ░ (SRAM-resident,
   x [M,H] ───────────────────────────────────────────────► gemm ◄───────┘             ░  bundled into MME)
                                                              │
                                                        gu [M, 2I] bf16
                                                          ╱        ╲
                                            slice [:,:I]            slice [:,I:]
                                                 │                       │
                                            gate [M, I]              up [M, I]
                                                 │                       │
                                       sigmoid_fwd_bf16                  │
                                                 │                       │
                                          (gate·sigmoid)  ── mult_fwd ──► silu [M, I]
                                                                          │
                                                            silu · up ── mult_fwd ──► h [M, I]
                                                                                       │
   w2_packed ─reinterpret_cast→ uint4 ┐                                                │
   w2_zp     ─reinterpret_cast→ uint4 ┤                                                │
   w2_scale  ─────────────────────────┴─► dequantize_4_bit_bf16 ─► W2_bf16 [I, H]  ░   │
                                                                          │         ░   │
                                                                          └► gemm ◄─────┘   ░ NON-PERSISTENT
                                                                              │
                                                                          y [M, H] bf16
                                                                              │
                                                       y · router_w[1,1] ── mult_fwd ──► out [M, H]  ◄ result
```

Every edge drawn into a `gemm` from a `dequantize_4_bit_bf16` is the SRAM-resident
dequant→MME bundle. The `[1,1]` router weight broadcasts over the whole `[M,H]` output in
the final `mult_fwd_bf16`, so the router weight is **folded inside the captured recipe** (no
separate host-side scaling at M=1).

Source of truth: `moe_fused_expert_backend_shared.h:89-298` (`MoeFusedExpert::AddNode`).

## 4. The K-expert single-recipe extension

The single-expert op is one expert. A routed MoE layer needs the top-K (≤ 8 at
single-stream). Two delivery paths exist:

- **Python K-loop** (`hpu_moe_ext/fused_expert_routed.py::fused_moe_routed`): does the
  proven, HPU-graph-safe gather/topk/remap (mirrored verbatim from the stock `_routed_moe`),
  then calls `fused_expert` **once per active expert** and accumulates. Correct, and the
  per-expert recipe is already the fusion + SRAM win — but it issues **K separate graph
  launches + K host-side adds per token**.

- **K-expert single-recipe op** (`torch.ops.hpu_moe_ext.fused_moe`,
  `moe_fused_expert_routed_backend_shared.h`): emits **all K experts' fused subgraphs + the
  weighted sum into ONE captured recipe → ONE launch**. The per-expert chain is factored
  **byte-for-byte** into `emit_one_expert(...)` (`moe_fused_expert_emit.h`), which the
  single-recipe op calls K times; the K results are combined by a left-fold tree of
  `add_fwd_bf16`. The K experts are passed as **TensorLists** (`Tensor[]` per weight class)
  — the bridge's own idiom for a multi-expert op (`mixture_of_experts` takes
  `at::TensorList w1, w2, w3`). Each expert's two dequants stay non-persistent, so the GC
  bundles each into its MMEs exactly as in the single-expert recipe — **the SRAM win is
  preserved per expert inside the one fused recipe** (confirmed on device; this was the top
  risk — K experts' dequant tiles co-resident in one recipe could in principle spill).

The single-recipe op collapses the per-token K-launch overhead into one launch, and is the
path used under lazy serving at M=1 (see §5). (Measured caveat: collapsing the K launches did
**not** beat stock — a batched-K variant lands at the same ~23 tok/s single-stream as the
sequential path; the bottleneck is not launch overhead. See [§9](#9-performance--the-measured-verdict).)

## 5. Eager vs lazy (capture), and the two `.so`s per op

The op ships in **two builds per op** that differ **only in the frontend**; the backend
`AddNode` recipe is shared **byte-for-byte** via the `*_shared.h` header.

| Frontend | TU | Why |
|---|---|---|
| **Eager** (`habana::eager::EagerOp`) | `moe_fused_expert_backend.cpp` | The `vllm-gaudi:next` image is the pytorch-**upstream/eager** build. EagerOp's symbols live in the eager plugin (`libhabana_pytorch2_plugin.so`), which `import habana_frameworks.torch` loads. |
| **Lazy** (`habana_lazy::LazyOp`) | `moe_fused_expert_backend_lazy.cpp` | Under `PT_HPU_LAZY_MODE=1` the **lazy** plugin is loaded instead. Only the lazy frontend (a) loads against it and (b) **emits into the lazy IR the HPU-graph capture engine records** — so the op is captured into the decode graph and replayed per token. The eager frontend would fail to load *and* would bypass capture. |

The two `.so`s are **mutually exclusive by mode** (the eager `.so` will not load under lazy
and vice-versa), so the plugin picks the right one by `htorch.is_lazy()` at load time
(`moe_wna16_hpu.py:106-109`). Lazy **HPU-graph capture is confirmed** for both the
single-expert and the K-expert single-recipe op (the K=8 experts captured into ONE graph and
replayed).

A subtle lazy-serving hazard and its fix: under `wrap_in_hpu_graph`, vllm-gaudi's lazy
default `disable_tensor_cache=True` + `mark_user_outputs(..., free_inplace=True)` **frees**
in-graph python-side per-expert clones after capture, so at replay the per-expert input
points at freed storage (a `ValidateSyncInputTensors ... tensor_data is empty` FATAL). The
**single-recipe `fused_moe` op avoids this entirely** — it takes the K experts as
TensorLists (`.unbind(0)` views of the persistent `index_select` outputs) and slices /
dequants / sums **on-device inside one recipe**, so there is **no python intermediate to
free**. That is exactly why the lazy + M=1 path routes through the single-recipe op
(`fused_expert_routed.py:248-275`).

## 6. The fp8-compute toggle

A separate `.so` (`moe_fused_expert_fp8_backend.cpp`, library `hpu_moe_ext_fp8`) carries an
fp8-compute variant: identical topology, but after each INT4→bf16 dequant the weight is cast
bf16→fp8 and the two expert GEMMs run on the MME in **fp8** (native, ~2× the bf16 MME rate;
the fp8 weight is half the bf16 bytes, so the SRAM win is, if anything, larger). It uses the
bridge's own `convert_to_fp8` + `fp8_gemm` constructions, with **per-channel weight +
per-token activation** recovery scales computed in-graph (a `reduce_max` amax, so the whole
thing stays one captured recipe with no host-side scalar per replay).

**Accuracy caveat (documented, measured):** INT4→fp8 is **double quantization**. fp8 e4m3
has 3 mantissa bits; stacked on the INT4 error across the gate||up→SiLU→down pipeline the
relNorm vs the bf16 reference has a **hard floor of ~6.4e-2** (measured on-device 6.43e-2,
matching a faithful CPU emulation). Per-channel/per-group scale granularity does **not**
break ~5e-2 (in the GPTQ layout the weight's dynamic range lives along the contraction/group
axis, largely orthogonal to per-output-channel scaling; the binding constraint is fp8's 3
mantissa bits + the intermediate-`h` requant). So fp8-compute is a **memory/throughput
option, not a single-stream accuracy win** — it trades a few percent of accuracy for half
the weight HBM traffic + ~2× MME. The fp8 op also threads the Gaudi-2 **fn-vs-fnuz 2× trap**
correctly (the dequant stays bf16-out; the only fp8 cast is a self-consistent round-trip
where `convert_to_fp8`'s `1/s` and `fp8_gemm`'s `s` cancel the 2× factor; `FP8_MAX = 240`
fnuz, not 448). The fp8 backend ships as a **precise, compile-ready scaffold** with the full
node spec; it falls back to the proven bf16 recipe unless `BUILD_FP8_NODES=1` is flipped
after the on-device param-struct validation (the box owner's run).

## 7. Generality — dims from tensor shapes, ~zero per-model config

The op **hardcodes nothing about the model**. `M, H, I` (and the group size, implicitly via
`scale.size(0)`) are all derived from the input tensor geometry
(`moe_fused_expert_backend_shared.h:104-114`), and the INT4 dequant complex-GUID infers its
grouped/non-grouped mode from `scale.size(0)` vs `input.size(0)`. So a new INT4 MoE model
"just works" once its per-expert weights are packed in the proven PyTorch-row-major layout.
This is validated **5/5** across the MoE shape matrix (`test_fused_expert_general.py`):

| Profile | H | I | 2I | GS | note |
|---|---|---|---|---|---|
| GLM-5.1 | 6144 | 2048 | 4096 | 128 | proven baseline anchor |
| DeepSeek-V3-ish | 7168 | 2048 | 4096 | 128 | wider hidden, MLA family |
| Nemotron-ish | 8192 | 2048 | 4096 | 128 | widest hidden tested |
| MiniMax-ish | 6144 | 2048 | 4096 | 64 | finer group size (GS=64) |
| odd-width + GS=32 | 5120 | 1536 | 3072 | 32 | odd I, GS=32 |

Layout invariants asserted per profile: `H%GS==0`, `I%GS==0`, `(2I)%8==0`, `H%8==0`,
`I%8==0` (the INT4 pack is 8 nibbles/int32 along the OUT dim; the dequant groups along the
IN dim by GS).

## 8. Validated numbers

| What | Metric | Value |
|---|---|---|
| Single-expert op | relNorm vs `convert_from_uint4` ref | **~4.2e-3** |
| K-expert Python wrapper | relNorm | ~5.7e-3 |
| K-expert **single-recipe** op | relNorm + **lazy capture confirmed** | **~5.9e-3** (K=8 in ONE captured graph) |
| Real-weight test (1 real GLM expert) | relNorm, eager + lazy | **~4e-3** |
| Generalization | profiles passing | **5/5** |
| SRAM residency | weight bf16 in DRAM / workspace | none in graph dump / **0 MB** (the HBM round-trip is gone) |
| fp8-compute | relNorm floor (double-quant) | ~6.4e-2 (memory/throughput option, not an accuracy win) |
| DSA indexer op (sibling, same path) | top-k Jaccard vs DeepSeek oracle | **1.0** |

These are **correctness, generality, and capture** numbers — all green. They establish that
the kernel works, not that it is fast; **for speed see §9 (it is slower than stock).** The
kernel additionally serves **Qwen-397B** end-to-end (coherent + stable across multi-bucket
long generation); the GLM end-to-end serve is gated on pre-existing GLM serve-env issues
**separate from the kernel** (the kernel is validated on real GLM weights in isolation, ~4e-3).

## 9. Performance — the measured verdict

**The fused MoE op is slower than the stock Habana `mixture_of_experts` intrinsic in both
regimes. It is correct, general, stable, and serves end-to-end — but it is NOT a MoE speedup.
Do not enable `HPU_MOE_FUSED_OP` / the monolithic variant for speed.**

| Regime | Fused custom op | Stock `mixture_of_experts` | Verdict |
|---|---|---|---|
| **Single-stream** (batch=1, warm) | **~23 tok/s** | **~30 tok/s** (proven ~29–34 ceiling) | **stock faster** |
| **Throughput** (Qwen-397B, concurrency-16, `--max-num-seqs=16`, warm) | **~68 tok/s** | **~126 tok/s** | **stock faster** |

Why stock wins:

1. **The stock op is heavily optimized.** `mixture_of_experts` is a batched/grouped intrinsic;
   the custom op's overhead (chunked sub-chains to force SRAM-residency, OpBackend dispatch, a
   much larger captured graph) **outweighs the SRAM-residency bandwidth saving**. The "~99 µs
   dequant-to-HBM floor" premise (§1) does not hold against an internally-efficient stock op.
2. **Single-stream is host/collective-overhead-bound, so fusion has nothing to bite on.** The
   MoE compute is **~0.16 ms of a ~127 ms** decode step (~90% overhead); **~30 tok/s is the
   sync-bound batch=1 cap** and the MoE op is never the bottleneck there. The batched-K variant
   (which kills the sequential K-emission) did **not** improve single-stream — batched and
   sequential both ~23 tok/s — so it is not launch overhead either; the stock intrinsic is
   simply faster at the compute.
3. **The throughput "slowness" is a config artifact, not an op limit.** max-num-seqs=16 is not
   the box's ceiling: stock at `--max-num-seqs=64` peaks **~486 tok/s** warm (above the ~366
   expected). The A/B above is apples-to-apples at one low config; absolute throughput is
   healthy and **config-bound, not op-bound**. *(The `--max-num-seqs=128` ceiling number is
   pending and will be appended by the box owner.)*

**What IS the win:** the reusable **`OpBackend` kernel infrastructure** — a validated mechanism
to ship multi-node, SRAM-resident, lazy-captured fused kernels on Gaudi 2 with **no TPC ISA**,
correct on real weights, general across models. Its real payoff is **kernels with no stock
equivalent** — the **DSA sparse-attention kernel** (§10), which has no Habana intrinsic to lose
to and is the 1M-context enabler. Building a custom kernel to *beat* a well-optimized stock op
inherits that op's optimization debt as the bar; the right target is what the platform lacks.

Full project narrative, trap log, and complete results tables:
[`PROJECT_FINDINGS.md`](PROJECT_FINDINGS.md).

## 10. Relationship to the DSA kernels (sibling work, included in the publish copy)

The same `OpBackend` mechanism that shipped the fused MoE op is reused for **DeepSeek Sparse
Attention** (the `dsa/` directory): a lightning-indexer top-k op
(`torch.ops.hpu_dsa_ext.indexer_topk`) and a gather-consume sparse attention op
(`torch.ops.hpu_dsa_ext.sparse_attn`). **This is where the infrastructure actually pays off:**
there is **no stock Habana intrinsic for DSA** (unlike MoE), so the DSA kernel enables a
capability the platform lacks rather than competing with an optimized op it can only lose to —
and it is the long-context (≫dense-MLA) enabler. Both ops are **op-level validated on-card**
(all 3 gates pass: indexer top-k recall **0.999** vs the DeepSeek/HF oracle; `sparse_attn` ==
dense-MLA when ctx ≤ topk; `sparse_attn` ≠ dense **and** matches the gather oracle when
ctx > topk). They are included as source in the publish copy because they share the
architecture; the **design** lives in `DSA_OPBACKEND_KERNEL_DESIGN.md`. This document and
`KERNEL_TECHNICAL.md` cover only the **fused INT4 MoE kernel**; the project-level narrative is
in [`PROJECT_FINDINGS.md`](PROJECT_FINDINGS.md).
