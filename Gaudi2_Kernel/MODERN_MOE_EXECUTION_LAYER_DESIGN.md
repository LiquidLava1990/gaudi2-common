# A General Modern-MoE Execution Layer for Gaudi-2 — Architecture & Build Plan

Status: **DESIGN** (off-card; read source/driver/docs, 2026-06-20). This is the complete,
hard-way design for owning the execution layer for modern MoE transformers
(DeepSeek-V3, Kimi-K2.7, Qwen3-MoE, GLM-5.2) on 8×Gaudi-2 — replacing the stock
graph-compiler/bridge's per-op fragmentation with **few fused OpBackend megakernels +
owned whole-forward capture**, parameterized across the four families.

It builds directly on the proven precedents in this repo:
- the int4 fused-MoE OpBackend kernel (`KERNEL_ARCHITECTURE.md`, `KERNEL_TECHNICAL.md`) —
  the existence proof for multi-node SRAM-resident lazy-captured custom kernels;
- the DSA indexer + sparse-attn OpBackend kernels (`DSA_OPBACKEND_KERNEL_DESIGN.md`) —
  the existence proof that the same mechanism builds attention-class kernels;
- the honest perf truth (`PROJECT_FINDINGS.md`, and the `glm52-single-stream-rootcause`
  memory) — **single-stream is ~85% device-idle / dispatch-fragmentation-bound, not
  compute-bound**; per-op kernels that merely match a stock op do not move it; the win is
  collapsing the ~135 fragmented recipe submissions / forward into a handful of captured
  dispatches.

> **The thesis in one sentence.** The Gaudi bridge already wraps the *whole* model forward
> in one HPU graph (`hpu_model_runner.py:861`), but its **interior fragments into ~125–135
> recipe segments per forward** because the data-dependent modern-MoE ops (256-expert
> top-k routing, MLA latent attention, DSA top-k) force graph breaks; the device then sits
> ~85% idle waiting on host-serialized recipe issue. **Fuse those data-dependent ops into a
> few config-parameterized OpBackend megakernels so the captured graph collapses to 1–8
> recipes — that is simultaneously the control win (we own the layer), the speed win (toward
> the ~30ms device floor → 2.5–4× single-stream), and, done generally, covers all four
> families from one parameterized layer.**

---

## 0. The problem, measured (why this is the right altitude)

The Gaudi graph compiler/bridge was tuned for **70B dense** (a few big matmuls/layer, no
data-dependent control). Modern MoE breaks that model in three places, each of which forces
a `mark_step` → a recipe boundary:

| Data-dependent op | Why it breaks the graph | Where it lives |
|---|---|---|
| **256-expert top-k MoE routing** | expert selection is content-dependent; the stock int4 path dequants **all** local experts in a Python `get_dequant_weight()` loop (up to ~512 ranks) + an intra-op `mark_step` | `extension/ops.py:1456` `VllmMixtureOfExpertsOpWNA16.forward` (loop `:1487`, `mark_step` `:1489`) |
| **MLA latent attention** | decode uses latent absorption (`bmm` W_UK_T / W_UV) over a paged latent cache; `flat_pa_mla` is Python over `bmm`+`softmax`, not one GUID | `attention/backends/hpu_attn.py:309` `HPUMLAImpl.forward_mqa` |
| **DSA sparse top-k** (GLM-5.2) | `Indexer` does a content-dependent top-2048 select; `Nonzero`/variable-count breaks static capture | `deepseek_v2.py:603` `Indexer`, used at `:1009` |
| **per-layer mark_step** (structural) | a `forward_hook` calls `mark_step()` after **every** decoder layer | `hpu_model_runner.py:499` `modify_model_layers`, installed `:5103` (`VLLM_CONFIG_HIDDEN_LAYERS`) |

Net for GLM-5.2 single-stream (the deepest-instrumented case, `glm52-single-stream-rootcause`):

- step ≈ **195 ms / token, dead-constant**, all 8 ranks identical.
- device **AIP-util ~15–18%** during sustained decode → **device ~82–85% IDLE**; compute
  floor ≈ **~30 ms**; the other **~165 ms is WAIT** (host-serialized recipe issue).
- py-spy: the worker blocks in `HPUGraph::replayV3 → ExecuteCachedGraph →
  active_recipe_counter::wait_for_next_decrease` while **~135 fragmented recipe submissions/token**
  drain through 8 AccThreads; strace ~2506 `WAIT_CS`/3s.
- **Every isolated lever failed to move the 195 ms** (async-replay raised util 17→46% but
  cadence flat; busy-poll byte-exact but flat; config staging knobs dud; collective time is
  0.2–2% so collectives are *not* it). The wall is the **count and serialization of recipe
  submissions**, exposed because the device is mostly idle.

The csengine experiment (`csengine/ARCHITECTURE.md`) independently proved the same wall from
below: THE WAIT (per-recipe host fences) **is deletable** via the open CS ioctl + on-device
SOB/monitor sync — but doing it at the raw-CS altitude means hand-relocating recipes,
hand-porting the sampler, and hand-porting TP=8 collectives below the bridge. **That is too
low.** The right altitude is the one the int4 + DSA kernels already proved:

> **OpBackend fused kernels + the bridge's own HPU-graph capture.** The bridge owns
> launch / stream / workspace / tensor-mapping / on-device sync; we own the *recipe geometry*.
> Collapsing the per-op interior into a few fused recipes gives the *same* "one island per
> token, on-device sync" benefit csengine chases — through machinery that already works on
> this exact box/stack, with no raw command submission.

This design is that collapse, made **general** across the four families.

---

## 1. The layered architecture

Six layers, bottom-up. Each is config-parameterized; the **Config Adapter (Layer 5)** is the
single place a model family is described, and it drives every kernel below it.

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ L6  INTEGRATION  — vllm-gaudi plugin seam: route modern-MoE models to OUR layer,   │
 │      preserve byte-exact-vs-stock, keep bucketing + bypass_hpu_graphs contracts    │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │ L5  CONFIG ADAPTER  — one schema (ExecConfig) maps {DeepSeek-V3, Kimi-K2.7,        │
 │      Qwen3-MoE, GLM-5.2} HF config → every kernel parameter below. ONE adapter.    │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │ L4  WHOLE-FORWARD CAPTURE  — own the capture: collapse the 78 layer-dispatches     │
 │      into 1–8 HPU-graph recipes; bypass the per-op segmentation the DD-ops trigger │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │ L3  LAYER-MEGAKERNEL  — compose (a)-(d) into few dispatches/decoder-layer; place   │
 │      the TP=8 per-layer all-reduce (one collective at the layer boundary)          │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │ L2  COMPONENT KERNELS (fused OpBackend, config-parameterized):                      │
 │   (a) ATTENTION  MLA-decode / GQA-decode fused: rope+KV-read+SDPA+o_proj → 1 disp. │
 │   (b) MoE-LAYER  route(top-k)+int4/fp8 experts+combine, fully on-device → 1 disp.  │
 │   (c) DSA INDEXER  fused top-k select (+ IndexShare cadence); no-op for non-DSA    │
 │   (d) NORMS+RESIDUAL+ROPE folded into the layer recipe                              │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │ L1  OpBackend SUBSTRATE (PROVEN)  — habana::OpBackend + KernelRegistry +           │
 │      BuildNode chains; eager/lazy .so by mode; CPU-only build vs bridge v1.24.0-1007│
 └──────────────────────────────────────────────────────────────────────────────────┘
```

### L1 — the OpBackend substrate (proven, reuse verbatim)

The mechanism is already validated and is the foundation we extend (full detail:
`KERNEL_TECHNICAL.md`). A fused kernel is a `struct X : habana::OpBackend` whose
`AddNode(graph, stack)` chains `OpBackend::BuildNode(this, graph, {guid, {syn_inputs},
{output_attrs}, &params, sizeof})`. Non-final outputs are **non-persistent**
(`final_result_index = nullopt`) → the GC bundles producer→MME in **SRAM** (no HBM
round-trip). Registration is the 3-part triple (`TORCH_LIBRARY` schema +
`TORCH_LIBRARY_IMPL(HPU)` frontend + `KernelRegistry().REGISTER_HPU_BACKEND`). Two `.so` per
op (eager `EagerOp` for `:next`, lazy `LazyOp` for serving) share the `*_shared.h` backend
byte-for-byte; the plugin picks by `htorch.is_lazy()`. Build is **CPU-only**, linking only
`libhabana_pytorch_backend.so` with `--allow-shlib-undefined`, against the open
**gaudi-pytorch-bridge tag `v1.24.0-1007`**.

**The reusable primitive vocabulary already exercised on-device** (callable from any new
`AddNode`):

- MME: `gemm`, `batch_gemm` (transpose flags), `masked_batch_gemm` (Gaudi-2, fold a mask
  into the MME).
- Quant: `reinterpret_cast`→uint4, `dequantize_4_bit_bf16`, (scaffolded)
  `convert_to_fp8`/`fp8_gemm`/`softmax_fp8`.
- Attention: **`sdpa_fwd`** (`ns_Sdpa::ParamsV7{scale, is_causal, softmax_mode,
  attention_mask input}`) — the fused flash QKᵀ+softmax+PV CGUID; `sdpa_recomp_fwd`
  (sliding-window + sink, for the streaming tile); `fp8_sdpa_fwd`.
- Elementwise/reduce: `relu_fwd`, `sigmoid_fwd_bf16`, `mult_fwd_bf16`, `add_fwd_bf16`,
  `where_fwd`, `greater_equal_fwd`, `masked_fill_fwd`, `reduce_sum_multi_dim_fwd`, `topk`
  (`ns_TopkNodeV2::ParamsV4`, dynamic-K capable).
- Shape/index: `slice` (`synSliceParamsV2`), `concat`, `gather_fwd`/`index_select`,
  `index_copy`/`scatter_fwd`, `cumsum_fwd`, `range`, plus the helpers
  `ReshapeHelper`/`BroadcastHelper`/`PermuteHelper`/`ConstantHelper`/`BuildCast`.

**The one primitive not yet in a kernel and needed for the attention megakernel: in-graph
RoPE** (see L2a). Everything else for all four families is in the vocabulary above.

### L2 — the component kernels (the new build)

Four fused kernels, every dim from config/shape. These extend the substrate; (b) extends the
**existing int4 OpBackend kernel** directly; (a) and (c) reuse the DSA attention/indexer
backends as their starting point.

#### L2(a) — ATTENTION megakernel (`hpu_attn_ext::mla_decode` / `gqa_decode`)

One fused dispatch for the decode attention of one layer, parameterized by attention type +
head config. **Two backends, one schema family**, selected by the adapter's `attn_mode`:

- **`MLA_LATENT`** (DeepSeek-V3, Kimi-K2.7, GLM-5.2): operate in latent space.
  Node-chain: `index_copy` write current latent KV → (rope on the `qk_rope` slice, in-graph)
  → `bmm(q_nope, W_UK_T)` query-absorb → gather paged latent blocks (`index_select` on the
  flat `(blocks*block_size, 1, kv_lora+qk_rope)` cache by `block_list`) → `sdpa_fwd` (MQA over
  the absorbed latent, additive mask = block-padding bias) → `bmm(out, W_UV)` V-up-proj →
  `gemm` o_proj. Result `[T,H,Dv]→[T,hidden]`.
- **`GQA`** (Qwen3-MoE): per-head q/k RMSNorm (Qwen3-only, folded) → rope (neox) → gather
  paged K/V blocks → `sdpa_fwd` over `n_kv=4` heads expanded → `gemm` o_proj.

Parameters from the adapter: `{attn_mode, num_heads, num_kv_heads, head_dim |
(q_lora,kv_lora,qk_nope,qk_rope,v_head), rope_theta, rope_interleave, q_k_per_head_norm,
rms_eps, block_size}`.

**What already exists to start from:** the DSA `sparse_attn` backend
(`dsa_sparse_attn_backend_shared.h`) already fuses gather → permute → `sdpa_fwd` → reshape
in one captured recipe — it *is* a sparse attention megakernel minus rope, the paged-block
read, and the absorption/o_proj. The decode MLA math (`forward_mqa`, `_v_up_proj`,
`latent_cache_k`) is the executable spec (`hpu_attn.py:309-359`). The **prefill** path stays
on the stock fused SDPA initially (prefill is bandwidth/compute-bound, not dispatch-bound;
fuse it only as a later phase).

**The single new primitive — in-graph RoPE.** Today rope is applied upstream of every
attention op (so the existing ops are rope-agnostic). To make decode one dispatch, rope must
be a node in the chain. Two routes: (i) emit it from elementwise nodes (`mult_fwd`/`add_fwd`
over precomputed cos/sin, with interleave-vs-neox as a build/param toggle — cos/sin are
small inputs prepared once by `_rotary_prepare_cos_sin`, `hpu_model_runner.py:799`); (ii) if a
`rotary`/`rope` CGUID exists in the bridge, bind it. **De-risk by route (i)** — it is pure
arithmetic over existing GUIDs, no missing kernel.

#### L2(b) — MoE-LAYER megakernel (`hpu_moe_ext::moe_layer`) — EXTENDS THE PROVEN KERNEL

The existing int4 kernel already does the full per-expert forward
(`reinterpret→dequant→MME→SiLU→dequant→MME→×router`) SRAM-resident, with K-expert
single-recipe (`fused_moe`), batched-K (`fused_moe_batched`), and monolithic (all-E) variants
(`KERNEL_TECHNICAL.md` §3-4). This kernel **extends** it from "the expert math" to "the whole
MoE layer in one dispatch" by folding the route + combine inside the recipe:

- **route** — gate `gemm` → top-k select. For sigmoid/noaux_tc (DeepSeek/Kimi/GLM) vs softmax
  (Qwen3) the *scoring* differs; both reduce to a `topk` over `[T, n_experts]` plus a
  renormalize. Grouped routing (DeepSeek-V3 `n_group=8/topk_group=4`) is an extra masked
  top-k stage. Keep route on-device (no `.item()`), feeding the expert stage via index
  tensors that stay graph-internal.
- **experts** — the proven int4 (now) / fp8 (toggle) fused expert chain, K-expert
  single-recipe (the validated `fused_moe` TensorList path that is HPU-graph-capture-safe —
  no python intermediate to free, `KERNEL_TECHNICAL.md` Gate 10).
- **combine** — the existing weighted `add_fwd` fold; plus the shared-expert MLP folded as one
  more `gemm`/SiLU/`gemm` chain (n_shared ∈ {0,1}).

Parameters: `{n_routed_experts, top_k, n_shared, expert_inter_dim, quant∈{int4,fp8},
scoring∈{sigmoid,softmax}, grouped∈{none,(n_group,topk_group)}, routed_scaling, ep_size}`.

**Honest scope (from `PROJECT_FINDINGS` §3 — do not re-learn this the hard way):** the fused
expert math, *as a drop-in for the stock `mixture_of_experts` op*, is **not** a speedup at
single-stream — the stock intrinsic is internally efficient and single-stream was never
MoE-compute-bound. **The win here is NOT beating the stock MoE op; it is removing the MoE
op's contribution to the recipe-segment count** (the per-expert dequant loop + intra-op
`mark_step`, `ops.py:1487-1489`) so the *layer* and *forward* capture as fewer recipes. The
MoE kernel's value is realized only at L3/L4 (collapsing dispatches), not as an isolated op.
At the **MTP-verify bs>1** regime (the GLM-5.2 wall) the routed-gather is disabled
(`4*8=32>8`) and the op is numerically broken at bs>1 (relNorm 1e11) — **fixing the
fused/routed MoE for bs>1 is a named hard sub-problem (§3)**; the fp8 stock batched intrinsic
(`VllmMixtureOfExpertsOpFP8PerChannel`, `ops.py:1226`) is the fallback fast path at bs>1.

#### L2(c) — DSA indexer (`hpu_dsa_ext::indexer_topk`) — ALREADY BUILT

The lightning-indexer top-k kernel is **built and on-card-validated** (eager+lazy `.so`,
Jaccard ~1.0; `DSA_OPBACKEND_KERNEL_DESIGN.md`, `PROJECT_FINDINGS` §4e/§7). Node-chain:
`batch_gemm(q,k,transpose_b)` → `relu_fwd` → `mult_fwd(weights)` → `reduce_sum(heads)` →
`masked_fill(causal)` → `topk`(int32 indices, no cast). The **IndexShare cadence** (full vs
shared layers) is config-driven in `glm_moe_dsa_wire.DsaDims` (reads `indexer_types` /
`index_topk_freq=4` / `index_skip_topk_offset=3` → 21 full / 57 shared of 78), with a shared
`topk_indices_buffer` that shared layers reuse by simply not calling the op. **No-op for
non-DSA models**: gated on `hasattr(config,"index_topk")` — DeepSeek-V3 (base), Kimi-K2.7,
Qwen3-MoE never construct an indexer, so this kernel is inert for them.

Parameters: `{full_layers, shared_layers, index_n_heads, index_head_dim, index_topk,
indexer_rope_interleave}`.

#### L2(d) — norms + residual + rope folded into the layer

RMSNorm (`rms_norm_eps` ∈ {1e-5, 1e-6}) is a `gemm`-adjacent elementwise chain; the residual
add is `add_fwd`; rope is the L2a in-graph node. Folding these into the L3 layer recipe (vs
emitting them as standalone ops) removes the small inter-op boundaries between
norm→attn→norm→moe.

### L3 — the layer-megakernel (compose a–d; place the collective)

Compose the four components into **few dispatches per decoder layer**, mirroring the exact
stock layer math (`deepseek_v2.py:1168-1215`): `input_norm → MLA/GQA attn → (×1/scale) →
post_attn_norm → MoE(+shared) → residual`. The decoder layer for DeepSeek-V3 / Kimi-K2.7 /
GLM-5.2 is **one class** (`GlmMoeDsaForCausalLM` and `DeepseekV3ForCausalLM` both subclass
`DeepseekV2ForCausalLM`, deepseek_v2.py:1745/1749) → one megakernel covers three families;
Qwen3-MoE swaps attn(GQA) + route(softmax) but reuses the same composition.

**TP=8 collective placement.** GLM-5.2 MoE is already pure-EP (`--enable-expert-parallel`,
ep_size=8, no down_proj all-reduce; `glm52-single-stream-rootcause` confirms EP-only is
already in place and collective device-time is ~0.2–2%). The per-layer all-reduce that
remains is the **attention output / EP-combine** at the layer boundary. The megakernel places
**one collective at the layer boundary** (not interleaved among ops), so the captured layer
recipe is `[fused-attn] → [allreduce] → [fused-moe]` — the collective is the only host-visible
sync inside the layer, and it stays the bridge's HCCL all-reduce (we do **not** port
collectives to raw CS — that was the csengine over-reach). At bs=1 the collective is
latency-bound (~12 KB, ~0.6 ms/step) and not the wall; the wall was the *count* of
non-collective recipe segments, which L3 removes.

Result target: a decoder layer goes from **~6–8 fragmenting dispatches + intra-MoE loop** to
**~2–3 captured dispatches + 1 collective**.

### L4 — whole-forward capture (own the capture)

The bridge already captures the whole `HpuModelAdapter.forward` (`self.model(*args)`,
`hpu_model_runner.py:821`) into one `HPUGraph` under
`wrap_in_hpu_graph(..., disable_tensor_cache=True)` (`:861`), replayed by `replayV3`
(`graphs.py:441`), keyed by `input_hash` = (dtype, shape, view-hash) per bucket
(`graphs.py:547`). **The graph already exists; its interior fragments.** Owning the capture
means two things:

1. **Make the interior capturable as few recipes.** With L3 layer-megakernels, the 78 layers
   emit ~2–3 captured dispatches each instead of ~6–8 + a data-dependent break; the indexer
   is the proven static kernel (L2c) not a `Nonzero`. Target: the per-forward recipe-submission
   count drops from ~135 toward **1–8** (one per bucket-stable mega-segment; the exact number
   is what the build measures).
2. **Pre-resolve / static-shape the routing so the whole forward is one capturable shape per
   bucket.** The capture re-fires on shape change (`input_hash`); modern MoE's content-dependent
   shapes (variable expert count, variable kept-keys) are what force per-step recompiles. The
   design uses the patterns already proven in the DSA serve bring-up
   (`PROJECT_FINDINGS` §7): **static gather widths** (`index_topk+1`, no `max_keep.item()`),
   **key-axis padding to fixed multiples** (128) + **pow2 kk**, **CPU-built metadata**
   moved-to-device once (dodges the shape-agnostic SIF crash), and **eager-island
   `@torch.compiler.disable`** only where a residual data-dependent step cannot yet be made
   static. Each of these converts a content-dependent shape into a bucket-stable one, so the
   whole forward replays one recipe per bucket.

**Investigation owed (the research edge, §4):** whether the *whole* forward (all 78 layers)
captures as a single recipe, or whether the bridge still segments at the per-layer
`mark_step` hook (`hpu_model_runner.py:499`) — which we can simply **not install** for our
model types (set `VLLM_CONFIG_HIDDEN_LAYERS=78` to coalesce, or skip the hook in our patch),
and whether `disable_tensor_cache=True`'s `free_inplace` interacts with the megakernels'
held-leaf weights (the int4 kernel already solved this class via the single-recipe TensorList
op + held-leaf x/zp, `KERNEL_TECHNICAL.md` Gate 10).

### L5 — the Config Adapter (one schema, four families)

A single `ExecConfig` dataclass read from each family's HF `config.json`. This is the
generality lynchpin — **one adapter, many models** — and it already exists in slices
(`glm_moe_dsa_wire.DsaDims` reads the DSA params; the int4 kernel reads all MoE dims from
shapes). The adapter unifies and extends them. The four families map cleanly because three of
them (DeepSeek-V3, Kimi-K2.7, GLM-5.2) are MLA and share the vLLM `DeepseekV2*` class; Qwen3-MoE
is the GQA outlier.

**The shared schema (same across all four — bake into the layer):** `num_hidden_layers`,
`hidden_size`, `vocab_size`, dense `intermediate_size`, `rms_norm_eps`, `hidden_act=silu`,
pre-norm, **top_k=8** (all four), `norm_topk_prob=true` (all four), SwiGLU experts with
`moe_intermediate_size`, RoPE with a `rope_theta`.

**The per-family knobs (what the adapter switches on):**

| ExecConfig field | DeepSeek-V3 | Kimi-K2.7 | Qwen3-MoE (235B) | GLM-5.2 |
|---|---|---|---|---|
| `attn_mode` | MLA_LATENT | MLA_LATENT | **GQA** | MLA_LATENT |
| heads (q / kv) | 128 / 128 | 64 / 64 | 64 / **4** | 64 / 64 |
| MLA dims (q_lora/kv_lora/qk_nope/qk_rope/v) | 1536/512/128/64/128 | 1536/512/128/64/128 | — | **2048/512/192/64/256** |
| GQA head_dim / per-head q,k norm | — | — | 128 / **yes** | — |
| `rope_theta` / style | 10000 / interleave | 50000 / interleave | 1e6 / **neox** | **8e6** / interleave |
| `n_routed_experts` | 256 | **384** | 128 | 256 |
| `n_shared` | 1 | 1 | **0** | 1 |
| `expert_inter_dim` | 2048 | 2048 | **1536** | 2048 |
| `scoring` / `topk_method` | sigmoid / noaux_tc | sigmoid / noaux_tc | **softmax / —** | sigmoid / noaux_tc |
| grouped routing (n_group/topk_group) | **8 / 4** | 1 / 1 | — | 1 / 1 |
| `routed_scaling` | 2.5 | **2.827** | 1.0 | 2.5 |
| dense-prefix mechanism | `first_k_dense_replace=3` | `=1` | **`decoder_sparse_step`+`mlp_only_layers`** | `=3` (+ explicit `mlp_layer_types`) |
| `num_hidden_layers` / `hidden_size` | 61 / 7168 | 61 / 7168 | **94 / 4096** | 78 / 6144 |
| DSA indexer (`has index_topk`) | no (base; V3.2 yes) | no | no | **yes** (n_heads 32, dim 128, topk 2048, freq 4/off 3) |
| MTP (`num_nextn_predict_layers`) | 1 | **0** | **absent** | 1 |
| `rms_norm_eps` | 1e-6 | **1e-5** | 1e-6 | **1e-5** |
| served quant (this box) | fp8 (upstream) | fp8 (upstream) | int4 GPTQ (on-disk 30B) | **int4 experts + fp8-other** |

Adapter responsibilities: (1) discriminate MLA vs GQA (`kv_lora_rank present → MLA`);
(2) normalize MoE field-name drift (`n_routed_experts`/`num_experts`,
`n_shared_experts`/`shared_expert_intermediate_size`) and routing flavor; (3) provide a
`layer_is_dense(i)` abstraction over the two different dense-layer mechanisms; (4) emit the
optional-DSA params only when `index_topk` is present; (5) expose `has_mtp`. The adapter is the
ONLY file that grows per new family.

### L6 — integration into the vllm-gaudi plugin

Route modern-MoE models to OUR layer; keep the stock path for everything else; preserve
byte-exactness. The cleanest seam (confirmed by reading the runner) is the sanctioned
per-model-type patch hook **`apply_model_specific_patches`** (`hpu_model_runner.py:634`,
called `:5096`, already used for Llama4 / mamba / chunked-attn). It runs **before**
`_maybe_wrap_in_hpu_graph` (`:5112`), so our fused layers are what gets captured.

```
apply_model_specific_patches(model):
    cfg = ExecConfig.from_hf(model.config)      # L5
    if cfg.family in MODERN_MOE and our_layer_ready(cfg):
        install_modern_moe_exec_layer(model, cfg)   # swap DecoderLayer.forward (L3)
        # for GLM: + DSA wire (L2c) already installs via hpu-dsa-ext entry point
```

This is precisely the seam the existing `layer_prune_patch` already uses to monkeypatch
`DeepseekV2Model.forward` and `DeepseekV2DecoderLayer.forward` (proving a `.forward` swap is
honored here — vLLM's `@support_torch_compile` wrapper sets `do_not_compile=True` and just
calls `self.forward`). Three insertion granularities, ship in order:

- **(A) per-MoE-op** — swap `layer.moe_op` in the WNA16 method's
  `process_weights_after_loading` (`hpu_compressed_tensors.py:931`) for the L2b kernel. Smallest
  blast radius; the wiring (packed weights/scales/zp) is already there. Ships first as the
  capture-segment reducer for the MoE break.
- **(B) per-decoder-layer megakernel** — `apply_model_specific_patches` swaps
  `DeepseekV2DecoderLayer.forward` for the L3 composition. The main event.
- **(C) whole-forward** — branch in `HpuModelAdapter.forward` (`:774/821`) to a fully custom
  forward (L4) once the indexer is static-shape so the single capture won't fragment.

**Correctness contract (non-negotiable, the project's standing rule):** every step gated on a
**byte-exact greedy A/B vs the stock path** (the `HPU_MOE_FUSED_VERIFY` pattern: run both, assert
relNorm < 1e-2, stay-coherent-on-mismatch). Keep `bypass_hpu_graphs` (`:3743`) and
`HPUBucketingManager` (`:1294`) intact; new ops must preserve fixed shapes per bucket. Gate the
whole layer behind a `MODERN_MOE_EXEC=1` flag, default off = stock byte-for-byte.

**fp8-KV cache (deliverable 6, designed in).** MLA already stores a compressed latent KV
(`kv_lora+qk_rope` = 576 wide), which is the dominant context-window lever; layering fp8 on it
(via the bridge `fp8_sdpa_fwd` consume + an fp8 latent cache write) ~halves KV bytes →
**2–4× context** at the same HBM. Design: (i) write the latent cache as fp8 e4m3 (per-token
group scale) in the L2a `index_copy` node, threading the fn-vs-fnuz round-trip the int4→fp8
work already characterized (`KERNEL_ARCHITECTURE.md` §6: `convert_to_fp8` `1/s` × `fp8_gemm`
`s` cancel, `FP8_MAX=240` fnuz); (ii) consume via `fp8_sdpa_fwd` (`sdpa_gen.cpp:386`,
d_scale_q/k/v) so the absorbed-latent attention runs fp8 natively. The DSA indexer K-cache is
**bf16-only / un-quantizable** (full-scanned every step) — keep it bf16, budget it separately
(the M3 analysis: +14.25 KB/tok for the index-K cache). fp8-KV is a **memory/context** play,
gated on a measured run proving HBM-KV-bound; it stacks with MLA's compression but is
independent of the dispatch-collapse speed work.

---

## 2. Build plan — PhD-agent-sized work items, in dependency order

Each item is one focused agent's deliverable, with its reuse point and gate. **Foundation =
the int4 OpBackend kernel (L1, done) + the DSA indexer/sparse-attn kernels (L2c, done).**
The build extends, it does not start from zero.

### Phase 0 — Adapter + harness (no kernels; unblocks everything)

- **W0.1 Config Adapter (L5).** Implement `ExecConfig.from_hf` reading all four families'
  `config.json` into the unified schema (table in §1.L5). Reuse `glm_moe_dsa_wire.DsaDims`
  (DSA slice) + the int4 plugin's quant-config autodetect. **Gate:** unit test — load each of
  the four configs, assert every field resolves to the table values; assert `layer_is_dense(i)`
  matches the stock per-layer dense/MoE decision for all 4. *Card-free.*
- **W0.2 Parity harness + byte-exact A/B.** Extend the `HPU_MOE_FUSED_VERIFY` pattern to a
  general `MODERN_MOE_VERIFY` that A/Bs any swapped layer/op vs stock (greedy byte-compare +
  relNorm). **Gate:** harness runs against the current stock GLM-5.2 serve, reports identical.
  *Card-light (uses the live serve).*

### Phase 1 — MoE-LAYER kernel (L2b), extends the proven int4 kernel

- **W1.1 Route-on-device folding.** Add the gate→top-k→(grouped)→renorm route as on-device
  nodes feeding the existing `fused_moe` TensorList expert stage; no `.item()`. Parameterize
  scoring (sigmoid/softmax) + grouped (DeepSeek). **Reuse:** `fused_expert_routed.py` routing
  (already HPU-graph-safe), the validated single-recipe expert op. **Gate:** relNorm < 1e-2 vs
  stock `select_experts`+`mixture_of_experts`, all 4 families' MoE dims; lazy capture confirmed.
- **W1.2 Combine + shared-expert fold.** Fold the weighted combine + the n_shared MLP into the
  recipe. **Gate:** full MoE-layer relNorm < 1e-2; byte-exact A/B in serve (W0.2).
- **W1.3 (hard, gated) bs>1 routed/fused MoE.** Fix the relNorm-1e11-at-bs>1 break (the
  MTP-verify wall) — likely the union-truncation of the gather at bs>1; OR wire the fp8 stock
  batched intrinsic (`VllmMixtureOfExpertsOpFP8PerChannel`, `ops.py:1226`) as the bs>1 fast path.
  **Gate:** relNorm < 1e-2 at bs∈{1,4,8}; this is the lever that unblocks MTP-verify speed.

### Phase 2 — ATTENTION kernel (L2a), reuses the DSA sparse-attn backend

- **W2.1 In-graph RoPE node** (the one new primitive). Emit interleave + neox rope as
  elementwise nodes over precomputed cos/sin; build/param toggle. **Gate:** relNorm < 1e-3 vs
  the stock rope, both styles. *Card-free CPU oracle first, then on-card.*
- **W2.2 MLA-decode megakernel.** Compose: latent-write → rope → `bmm` absorb → paged-latent
  gather → `sdpa_fwd` → `bmm` V-up → o_proj. **Reuse:** `dsa_sparse_attn_backend_shared.h`
  (gather+sdpa+reshape is already this), `forward_mqa`/`_v_up_proj` as the spec. **Gate:**
  relNorm < 1e-2 vs `HPUMLAImpl.forward_mqa`, GLM/DeepSeek/Kimi dims; lazy capture.
- **W2.3 GQA-decode megakernel.** Per-head q/k norm (Qwen3) → rope(neox) → paged K/V gather →
  `sdpa_fwd` → o_proj. **Gate:** relNorm < 1e-2 vs the stock GQA decode, Qwen3 dims.

### Phase 3 — LAYER-MEGAKERNEL (L3) + collective placement

- **W3.1 Compose the decoder layer** (norms+residual+attn+moe) for the `DeepseekV2DecoderLayer`
  family (covers DeepSeek-V3, Kimi-K2.7, GLM-5.2). **Gate:** byte-exact greedy A/B vs stock,
  GLM-5.2 first; recipe-segment count per layer measured (target ~2–3 + 1 collective vs ~6–8).
- **W3.2 Qwen3 GQA layer** (swap attn+route, reuse composition). **Gate:** byte-exact A/B.
- **W3.3 Collective placement** — one all-reduce at the layer boundary; confirm EP-only path
  unchanged. **Gate:** TP=8 byte-exact + collective count == 1/layer.

### Phase 4 — WHOLE-FORWARD CAPTURE (L4) — the research-heavy item

- **W4.1 Static-shape the routing** across the warmup bucket grid (static gather widths,
  key-pad 128, pow2 kk, CPU-built metadata). **Reuse:** the four DSA serve-bug fixes verbatim.
  **Gate:** no `synStatus 26` / no `[SHAPE AGNOSTIC]` crash across the full bucket grid (the
  current GLM-5.2 mixed-shape DSA crash, the named open risk).
- **W4.2 Coalesce the per-layer mark_step** (skip the `modify_model_layers` hook for our types
  / `VLLM_CONFIG_HIDDEN_LAYERS=L`). **Gate:** per-forward recipe-submission count measured
  (target 1–8 vs ~135) via the py-spy/strace method already established.
- **W4.3 (stretch) whole-forward single capture** — branch in `HpuModelAdapter.forward`. **Gate:**
  one recipe/bucket replays the full forward; byte-exact; **measure single-stream tok/s** (the
  decisive number — target 2.5–4× from the ~85% idle reclaim).

### Phase 5 — fp8-KV + long context (deliverable 6)

- **W5.1 fp8 latent KV write + `fp8_sdpa_fwd` consume.** **Reuse:** the int4→fp8 fnuz round-trip
  characterization. **Gate:** coherent + context-window 2–4× at fixed HBM; KV-bound proven first.
- **W5.2 DSA sparse serve at long ctx** (the existing `glm52-dsa.env` path, lazy). **Gate:**
  needle-recall at >4096 through the real sparse path; mixed short/long shapes stable (W4.1).

**Critical path:** W0.1 → W1.{1,2} → W2.{1,2} → W3.1 → W4.{1,2,3}. Phases 1 and 2 parallelize
(different kernels, different agents). W1.3 (bs>1 MoE) and W4.3 (whole-forward) are the two
items that gate the *speed* claim; everything before them gates *control + correctness*.

---

## 3. The hardest sub-problems + how to de-risk each

Ranked by risk to the speed thesis.

1. **Whole-forward stays one capture / few recipes (the core speed bet).**
   *Risk:* even with megakernels, the bridge may re-segment at residual data-dependent steps
   or the per-layer hook, leaving the ~135-count near-unchanged → no idle reclaim → no speedup
   (the outcome every isolated lever hit so far).
   *De-risk:* this is exactly why the build is staged with **measured gates at W4.2** (count the
   recipe submissions before claiming anything) and W4.3 (measure tok/s). The diagnostic method
   (py-spy stack + strace `WAIT_CS` + AIP-util) is already established. If count drops but tok/s
   doesn't, the residual wall is named and attacked; if count won't drop, the static-shape work
   (W4.1) is the lever. **Do not declare the speed win until W4.3 measures it** — the project
   has been burned by modeled-not-measured speed claims repeatedly (`PROJECT_FINDINGS` §11-13).

2. **Synapse shape-robustness of fused recipes (no SIF).** Custom OpBackend recipes have no
   shape-inference function → they crash on a shape-changed shape-agnostic cache hit
   (`[SHAPE AGNOSTIC] ... shape inference failed`), and CGUIDs like `topk`/`bsw` reject
   non-pow2 dims (`synStatus 26 n_af`). This is the *named current open risk* on the GLM-5.2 DSA
   serve (mixed short/long shapes die at bucket 40/77).
   *De-risk:* the four DSA serve-bug fixes are the template (Int-direct topk output, eager-island
   `@torch.compiler.disable`, causal post-filter, pow2 key-pad). W4.1 makes every per-step shape
   bucket-stable; capture per warmup bucket; key-axis pad to 128 multiples. Worst case: keep the
   data-dependent residual as an eager island (correct, captures the rest) and accept a slightly
   higher segment count.

3. **bs>1 MoE for the MTP-verify forward (the GLM-5.2 throughput-of-single-stream wall).** The
   fused/routed MoE is validated only at bs=1; at the MTP-verify bs=4 the routed-gather is
   disabled and the fused op is numerically broken (relNorm 1e11).
   *De-risk:* W1.3. Two independent routes — fix the gather union-truncation at bs>1 (the likely
   root, the same storage-offset class as Gate 9), OR fall back to the **stock batched fp8
   intrinsic** (`VllmMixtureOfExpertsOpFP8PerChannel`, already in the plugin, works at all
   batch). The fallback de-risks the whole MTP path even if the fused bs>1 fix is hard.

4. **In-graph RoPE correctness (interleave vs neox).** GLM uses interleave, Qwen3 neox; GLM-5.1's
   MLA needed `is_neox_style` forced; the indexer uses its own `indexer_rope_interleave`. Getting
   rope wrong is a silent coherence killer that set-match gates miss (the M3 trap:
   `PROJECT_FINDINGS` §13a).
   *De-risk:* W2.1 builds rope against a **CPU fp32 oracle pinned to the runnable modeling code +
   stored-weight statistics** (the M3 lesson — never the paper), byte-exact A/B, both styles, as
   a standalone gate before it enters the megakernel.

5. **MLA latent-vs-per-head asymmetry in the consume.** Decode absorbs into latent; the current
   sparse consume re-materializes per-head (≤64K). A true 1M path needs gather-in-latent-then-
   absorb (unbuilt).
   *De-risk:* ship the per-head ≤64K megakernel first (matches what's validated); latent-gather is
   a Phase-6 extension shared with the fp8-KV work. The decode megakernel (W2.2) operates in
   latent space natively (absorbed MQA), so this risk is mostly the *prefill*/long-ctx path.

6. **Capture hazards under `disable_tensor_cache=True` (`free_inplace`).** Held-leaf weights freed
   across decode buckets → `ValidateSyncInputTensors empty` FATAL (the int4 kernel's Gate 10).
   *De-risk:* solved class — route multi-expert through the single-recipe TensorList op (no python
   intermediate), materialize x/zp as persistent held leaves; apply the same discipline to the
   new megakernels' weight inputs.

7. **Two-image reality + ABI drift.** OpBackend builds against `:next` (lazy PT-2.10 fork); the
   serve image must match; a vLLM bump (`build.sh`) can drift the FusedMoE/MLA method signatures
   (`gaudi-vllm-plugin-dsa-cuda`: quant API MEDIUM-HIGH, attn API MEDIUM).
   *De-risk:* pin the bridge tag (`v1.24.0-1007`), the standing `apt-mark hold`; the integration
   layer (L6) is thin Python against named seams (`apply_model_specific_patches`,
   `process_weights_after_loading`) so re-homing on a bump is a small, contained edit.

---

## 4. Proven-feasible vs research

| Component | Status | Evidence |
|---|---|---|
| **OpBackend substrate (L1)** | **PROVEN** | int4 kernel HW-validated, lazy-captured, serves Qwen-397B e2e; 8 MoE `.so` + 4 DSA `.so` built (`KERNEL_TECHNICAL.md`) |
| **int4 fused-expert math (L2b core)** | **PROVEN correct/general** (5/5 families, ~4e-3 real weights); **NOT an isolated speedup** | `KERNEL_ARCHITECTURE.md` §8-9, `PROJECT_FINDINGS` §3 |
| **DSA indexer + sparse-attn (L2c, + L2a starting point)** | **PROVEN on-card** (Jaccard ~1.0; sparse==dense ≤topk; sparse≠dense+==oracle >topk); eager serve e2e | `PROJECT_FINDINGS` §4e/§7 |
| **IndexShare cadence (L2c)** | **PROVEN** (17/17 wire tests; generator == shipped `indexer_types`) | `glm52-bringup`, `glm_moe_dsa_wire.py` |
| **Config Adapter (L5)** | **FEASIBLE** (slices exist: `DsaDims` + shape-derived MoE dims) | this design, §1.L5 |
| **whole-forward already captured (L4 substrate)** | **PROVEN it exists** (the bridge wraps `HpuModelAdapter.forward`) — the *interior* is what fragments | `hpu_model_runner.py:861`, `graphs.py:441` |
| **THE WAIT is deletable below the bridge** | **PROVEN by csengine PoC** (raw CS, 1 host wait/N-op chain) — establishes the dispatch-collapse is physically real | `csengine/ARCHITECTURE.md` §0-1, `cs_poc` |
| **in-graph RoPE node (L2a new primitive)** | **RESEARCH** (route-i is pure arithmetic over existing GUIDs → low risk) | §3.4 |
| **MLA/GQA decode megakernel (L2a)** | **RESEARCH** (composing proven primitives; sparse-attn backend is 80% of it) | §2 W2.2/W2.3 |
| **layer-megakernel + few-recipe capture (L3/L4)** | **RESEARCH — the core unproven bet** (the speed thesis; measured-gated, not assumed) | §3.1, W4.2/W4.3 |
| **bs>1 fused MoE (L2b)** | **RESEARCH** (broken at bs>1 today; fp8 stock fallback de-risks) | §3.3 |
| **fp8-KV (L6/deliverable 6)** | **FEASIBLE** (MLA latent + `fp8_sdpa_fwd` exist; fnuz trap characterized) | §1.L6, `KERNEL_ARCHITECTURE.md` §6 |

**The honest framing of the speed claim.** The 2.5–4× single-stream target rests on one
hypothesis: that the 195 ms / ~85%-idle step is dispatch-fragmentation-bound and that
collapsing ~135 recipe submissions toward 1–8 reclaims the idle. The evidence *for* it: device
is provably ~85% idle, the wall is provably the AccThread's serial issue of ~135 recipes (not
clocks, not collectives, not config, not IRQ-wake — all eliminated), and the csengine PoC
proves the per-recipe fence is physically deletable. The evidence *against* assuming it: every
*isolated* lever tried so far moved nothing, because none reduced the *count* — they attacked
the waits around the count. **This design is the first one that attacks the count itself.** It
is the right bet, but it is a bet, and the build plan **gates it on measurement** (W4.2 count,
W4.3 tok/s) rather than declaring it. The *control* and *generality* wins are not bets — they
are achieved the moment L3+L6 land and route the four families through one owned layer.

---

## 5. How it delivers control + generality (the project goal)

**Control (we own the execution layer).** Today the execution path is the stock bridge's
per-op lowering + the stock `mixture_of_experts` intrinsic + the stock MLA Python — we tune it
only through env flags and lose every time the data-dependent ops fragment it. After this:
the decoder layer's forward is **our** OpBackend megakernel recipe; the routing, the
attention, the DSA select, the norms, the collective placement, and the capture geometry are
all ours to shape. The seam (`apply_model_specific_patches` → swap `DecoderLayer.forward` →
captured before `wrap_in_hpu_graph`) means we own the recipe the device replays, while the
bridge still owns launch/sync/HCCL (the correct division of labor csengine over-stepped). We
gain the knobs that matter — dispatch count, SRAM residency, what's static vs eager-island,
where the collective sits — and we stop being at the mercy of the stock segmentation.

**Generality (one layer, four families).** The architecture is general by construction:
- **One model class covers three families** — `DeepseekV3ForCausalLM` and
  `GlmMoeDsaForCausalLM` both subclass `DeepseekV2ForCausalLM`, so the L3 layer-megakernel for
  the DeepSeek decoder layer serves DeepSeek-V3, Kimi-K2.7, and GLM-5.2; Qwen3-MoE reuses the
  same composition with the GQA attn + softmax route swapped in.
- **Every kernel reads its dims from config/shape, nothing hard-coded** — the int4 kernel is
  already 5/5-general this way; the DSA kernel needs zero changes for a new DSA model (only its
  `config.json`); the adapter (L5) is the single place a family is described.
- **Optional features are config-gated, inert when absent** — DSA fires only when `index_topk`
  is present (GLM-5.2 only); MTP only when `num_nextn_predict_layers>0`; grouped routing only
  for DeepSeek-V3; fp8-vs-int4 experts is a build toggle. A new family = one row in the
  ExecConfig table + (if it has a genuinely new structural element) one targeted kernel
  parameter, not a rewrite.

The result is the thing the project set out to own: **a single, parameterized, device-resident
execution layer for modern MoE on Gaudi-2 — fast (toward the device floor by collapsing
dispatch fragmentation), general (DeepSeek-V3 / Kimi-K2.7 / Qwen3-MoE / GLM-5.2 from one
adapter), and ours (we own the recipe, the bridge owns the metal).**

---

## 6. Key file/symbol index (for the build)

**Substrate + proven kernels (extend these):**
- `hpu-moe-ext/tpc_kernel/opbackend/moe_fused_expert_backend_shared.h` (+ `_backend.cpp` eager,
  `_backend_lazy.cpp` lazy, `moe_fused_expert_emit.h` per-expert) — the int4 kernel (L2b core).
- `hpu-moe-ext/tpc_kernel/opbackend/dsa/dsa_indexer_backend_shared.h`,
  `dsa_sparse_attn_backend_shared.h` — DSA indexer (L2c) + the sparse-attn megakernel base (L2a).
- `hpu-moe-ext/tpc_kernel/opbackend/build.sh` (+ per-op variants) — the CPU-only build.
- `hpu-moe-ext/tpc_kernel/opbackend/bridge/hpu_ops/{op_backend.h, stack_getter.h}` — the API.

**Executable specs / oracles (port from):**
- MoE: `hpu-moe-ext/hpu_moe_ext/fused_expert_routed.py`, `moe_wna16_hpu.py` (routing, gates).
- Attention: `_src/vllm_gaudi/attention/backends/hpu_attn.py:245-359` (`forward_mha/forward_mqa/
  _v_up_proj`), `_src/vllm_gaudi/attention/oot_mla.py` (MLA wiring, decode-cache fix).
- DSA: `hpu-dsa-ext/hpu_dsa_ext/{sparse_indexer_hpu.py, sparse_mla_consume.py, glm_moe_dsa_wire.py,
  hpu_indexer_op_forward.py, indexer_metadata_hpu.py, indexer_cache_hpu.py, dsa_ops.py}`.

**Integration seams (L6):**
- `_src/vllm_gaudi/v1/worker/hpu_model_runner.py`: `apply_model_specific_patches` (`:634`, the
  per-model-type patch hook), `_maybe_wrap_in_hpu_graph` (`:855`) + call site (`:5112`),
  `HpuModelAdapter.forward` (`:774/821`), `modify_model_layers` per-layer mark_step (`:499/5103`),
  `_execute_model_generic` forward + `bypass_hpu_graphs` (`:3724/3759/3743`),
  `HPUBucketingManager` (`:1294`).
- `_src/vllm_gaudi/extension/ops.py`: `VllmMixtureOfExpertsOpWNA16` int4 (`:1456`),
  `VllmMixtureOfExpertsOpFP8PerChannel` (`:1226`, the bs>1 fallback).
- `_src/vllm_gaudi/ops/hpu_compressed_tensors.py`: `HPUCompressedTensorsWNA16MoEMethod`
  (`:741/931/971`, the `layer.moe_op` swap point, seam A).
- `_src/vllm_gaudi/platform.py`: async/spec gate (`:244-256`), the `use_sparse` raise (`:75`).
- Capture engine: `…/habana_frameworks/torch/hpu/graphs.py`: `wrap_in_hpu_graph` (`:840`),
  `wrapped_hpugraph_forward` (`:635`), `input_hash` (`:547`), `replayV3` (`:441`).

**Model classes:** `vllm/model_executor/models/deepseek_v2.py`:
`DeepseekV2ForCausalLM` (`:1610`), `DeepseekV3ForCausalLM` (`:1745`), `GlmMoeDsaForCausalLM`
(`:1749`), `DeepseekV2DecoderLayer` (`:1084`, forward `:1168`), `DeepseekV2MoE` (`:361`),
`DeepseekV2MLAAttention` (`:878`), `Indexer` (`:603`). Qwen3-MoE: `qwen3_moe.py`.

**Companion designs:** `KERNEL_ARCHITECTURE.md` / `KERNEL_TECHNICAL.md` (int4 kernel),
`DSA_OPBACKEND_KERNEL_DESIGN.md` (DSA), `PROJECT_FINDINGS.md` (the honest perf truth + trap log),
`csengine/ARCHITECTURE.md` (the below-bridge altitude, why it's the wrong one).
