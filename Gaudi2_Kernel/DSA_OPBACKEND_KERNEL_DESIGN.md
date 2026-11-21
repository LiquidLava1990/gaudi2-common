# DeepSeek Sparse Attention (DSA) on Gaudi 2 — OpBackend Kernel Design

Status: DESIGN (read-only investigation, 2026-06-08). Foundation for the DSA kernel build.
Goal: replace the current **dense bypass** (`FULL_ATTENTION_FALLBACK=True`) with a **real,
general, ultra-fast** sparse-attention kernel built on the PROVEN `habana::OpBackend` path
(the same mechanism that shipped `torch.ops.hpu_moe_ext.fused_expert`).

This supersedes the old memory claim "DSA on HPU is impossible / needs CUDA / weeks-to-months."
That predates the OpBackend breakthrough. With OpBackend in hand, DSA is a **graph-construction
exercise over bridge ops that all exist** — not a from-scratch CUDA/DeepGEMM port. Verdict below.

All citations are to real files on this box. Container inspection was CPU-only (`docker run --rm`,
never touched `vllm-active`).

---

## 0. TL;DR verdict

**GO.** Every component of DSA maps to bridge ops that already exist as GC-lowered `OpBackend`
nodes in the version-exact bridge clone (`tag v1.24.0-1007`):

| DSA primitive | bridge op (GUID) | gen file |
|---|---|---|
| indexer Q·Kᵀ per head (MME) | `gemm` / `batch_gemm` (bmm) | `matmul.cpp`, `bmm_gen.cpp` |
| ReLU | `relu_fwd` (or `max` w/ 0) | (elementwise, std) |
| weighted head-sum | `gemm`/`mult_fwd`+`reduce_sum` | `matmul.cpp` |
| causal mask add | `add_fwd` + `masked_fill_fwd` | `masked_fill_gen.cpp` |
| **top-k over keys** | **`topk`** (`ns_TopkNodeV2::ParamsV4`, dynamic-K capable) | `topk_gen.cpp` |
| gather top-k KV | **`gather_elements_fwd`** / `gather` (`ns_GatherKernel`) | `gather_gen.cpp`, `index_select_gen.cpp` |
| sparse SDPA | **`sdpa_fwd`** / `fp8_sdpa_fwd` / `sdpa_recomp_fwd` (mask + window + valid_seq_len) | `sdpa_gen.cpp` |
| fused masked softmax (oracle) | `scaled_masked_triangular_softmax_fwd` | `scaled_masked_softmax_gen.cpp` |
| fp8 MQA logits | `convert_to_fp8` + `fp8_gemm` + `softmax_fp8` | `fp8_gemm_gen.cpp`, `fp8_ops.cpp` |
| indexer K-cache insert | `index_copy`/`scatter` (flat cache) | `index_copy_gen.cpp`, `scatter_gen.cpp` |
| sentinel scatter / cumsum pack | `scatter_fwd`, `cumsum_fwd`, `where_fwd`, `range` | `*_gen.cpp` |

**None require a hand-written TPC ISA kernel.** (Contrast: vllm-ascend's DSA is 100% AscendC ISA —
PR #3270.) The hard work is (a) the indexer-KV-cache plumbing into the vllm-gaudi model runner
(Python, already ~90% built in `hpu-dsa-ext`), and (b) composing the two OpBackend graphs and
getting their Synapse FCD geometry + SIF shapes right (the same multi-iteration bring-up the MoE
op needed — documented at `moe_fused_expert_backend.cpp:162-191`).

**The existing `hpu-dsa-ext` already gives us a numerically-correct pure-torch sparse path**
(indexer recall 0.99 vs HF/DeepSeek oracle, gather==masked-oracle diff 0.011 — `BUILD_PLAN.md:6-9`).
That is the **executable spec** + the off-hardware parity oracle for the kernel. The kernel build
is: take those two validated torch functions and re-emit them as captured OpBackend node-chains.

Phasing: **MVP = MVP-1 (indexer-as-OpBackend) + MVP-2 (gather-consume-as-OpBackend), prefill, ≤64K,
bf16** → then fp8 MQA logits, paged decode, and the ~1M streaming tile. Honest scope: §8.

---

## 1. DSA architecture map (the 3 components + GLM-5.1 config)

DSA = a *learned, per-query, top-k token selector* (the **lightning indexer**) feeding a **sparse
attention** that attends only the selected KV, backed by a **paged indexer K-cache**. The model is
otherwise standard MLA-MoE (DeepSeek-V2-family). Authoritative references read for this design:

- **HF transformers `modeling_glm_moe_dsa.py`** (transformers 5.10.2 in the `:next` image,
  `/usr/local/lib/python3.12/dist-packages/transformers/models/glm_moe_dsa/`) — the ground-truth
  math (`GlmMoeDsaIndexer.forward` :144-228, `GlmMoeDsaAttention.forward` :345-459).
- **vLLM in-image `Indexer`** (`$HOME/AI/dsa_src/vllm_core_deepseek_v2.py:603-743`) — the
  trained-weight projection/quant/rope pipeline as vLLM runs it.
- **vLLM `sparse_attn_indexer`** (`$HOME/AI/dsa_src/sparse_attn_indexer.py`) — the CUDA op
  contract (the thing we replace on HPU): `indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache`,
  `fp8_fp4_mqa_logits`, `top_k_per_row_prefill/decode`, paged indexer cache.

### 1a. The lightning indexer (the index head)

A small separate attention head set, distinct from the main MLA Q/K. Per query `t` it scores every
causal key `s` and keeps the top `index_topk` indices.

**Exact math (HF `modeling_glm_moe_dsa.py:184-227`, identical to vLLM `Indexer.forward`):**
```
q  = wq_b(q_resid).view(B,S,n_head,head_dim)          # index queries (own proj)
q_pe, q_nope = split(q, [rope_dim, head_dim-rope_dim]); q_pe = rope(q_pe); q = cat(q_pe,q_nope)
k  = k_norm(wk(hidden_states))                          # index keys (MQA, ONE shared head)
k_pe, k_nope = split(k, [rope_dim, ...]); k_pe = rope(k_pe); k = cat(k_pe,k_nope)
weights = weights_proj(hidden).float() * n_head**-0.5   # [B,S,n_head]
scores  = einsum("bshd,btd->bsht", q, k) * softmax_scale # per-head q·k    [B,S,n_head,T]
scores  = relu(scores)                                   # ReLU (positive-homogeneous)
index_scores = einsum("bsht,bsh->bst", scores, weights)  # weighted head-sum   [B,S,T]
index_scores += causal_mask(-inf)                        # s <= t (self incl)
topk_indices = index_scores.topk(min(index_topk, T)).indices   # [B,S,index_topk]
```
- `softmax_scale = index_head_dim**-0.5`.
- **fp8 in production**: vLLM quantizes q to fp8 (`per_token_group_quant_fp8`,
  `vllm_core_deepseek_v2.py:729`) and **folds q_scale into `weights`**:
  `weights = weights_proj * q_scale * softmax_scale * n_head**-0.5`
  (`vllm_core_deepseek_v2.py:738-741`). So q is consumed as bf16 with **no q_scale/k_scale
  multiply** — mathematically exact because ReLU is positive-homogeneous and q_scale>0
  (`sparse_indexer_hpu.py:12-17`). HF's bf16 path uses q_scale=1 and is identical
  (`modeling_glm_moe_dsa.py:204-216`).
- **K-cache for the indexer** is its OWN cache (`self._cached_keys` in HF :140-200; a paged
  `DeepseekV32IndexerCache` in vLLM). Decode appends one key; prefill writes the whole range.

This is exactly the validated `lightning_indexer_topk` in `hpu-dsa-ext/.../sparse_indexer_hpu.py:40-76`
(Jaccard 0.996, top-50 recall 1.0 vs DeepSeek fp8 oracle — `BUILD_PLAN.md:6`).

### 1b. The sparse attention (attend only top-k KV)

`GlmMoeDsaAttention.forward` (`modeling_glm_moe_dsa.py:394-459`): build an additive mask that is
0 on the top-k selected key positions and −inf everywhere else, **combine with the causal mask**,
and run standard MLA SDPA:
```
index_mask = full((B,S,T), -inf); index_mask.scatter_(-1, topk_indices, 0.0)   # :416-422
combined_mask = index_mask + causal_mask                                       # :428-436
attn_output = attention_interface(q, k, v, combined_mask, scaling=...)          # :446-456
```
That is the **masked-MHA mode** (DeepSeek-V3.2 paper §2.3, arXiv 2512.02556) — correct but O(L²) in
memory because the mask is `[S,T]`. The **product** form is the **gather mode**: gather the ≤index_topk
selected KV rows per query and SDPA over just those → O(L·index_topk). Both are implemented +
parity-proven in `hpu-dsa-ext/.../sparse_mla_consume.py` (`sparse_mla_masked_attention` = oracle,
`sparse_mla_gather_attention` = product, diff 0.011 — `IMPLEMENTATION.md`/`BUILD_PLAN.md:27-28`).

MLA wrinkle (Gaudi): vllm-gaudi's MLA uses **latent absorption** for decode (`W_UK_T`/`W_UV`,
`oot_mla.py:121-145` `forward_mqa`/`_v_up_proj`) and per-head `forward_mha` for prefill. Sparse
consume must therefore either (i) materialize per-head K/V then gather+SDPA (the current
`sparse_mla_wire.py:82-91` path, ≤64K) or (ii) gather in latent space then absorb (the ~1M path).

### 1c. The indexer KV-cache (paged)

vLLM allocates a `DeepseekV32IndexerCache` with `head_dim = index_head_dim + index_head_dim/128*4`
= **132 bytes/token** (uint8 fp8-packed key + fp32 group scale; `vllm_core_deepseek_v2.py:654-659`).
The CUDA op writes it via `indexer_k_quant_and_cache` and reads via `cp_gather_indexer_k_quant_cache`
(`sparse_attn_indexer.py:167-203`). **On HPU we replace this with a flat bf16 cache** (no fp8 pack)
of layout `(num_blocks*block_size, index_head_dim)` — `hpu-dsa-ext/.../indexer_cache_hpu.py`
(`IndexerKCache.insert/.gather_blocks/.gather_rows`, `sparse_indexer_hpu.py:104-122`). The runner
must be patched to allocate it (vllm-gaudi `get_kv_cache_spec` only handles
Attention/MLA/Mamba/FusedMoE and silently drops the indexer cache → `BUILD_PLAN.md:42`).

### 1d. GLM-5.1 exact DSA config (`/mnt/fastmodels/GLM-5.1-int4-mixed-AutoRound/config.json`)

```
architectures        = ["GlmMoeDsaForCausalLM"]      model_type = "glm_moe_dsa"
index_topk           = 2048      # top-k keys per query
index_n_heads        = 32        # indexer Q heads (DeepSeek-V3.2 uses 64)
index_head_dim       = 128       # indexer head dim
indexer_rope_interleave = True   # rope style for the indexer (interleave, NOT NeoX)
kv_lora_rank         = 512       qk_nope_head_dim = 192   qk_rope_head_dim = 64
v_head_dim           = 256       num_attention_heads = 64  (qk_head_dim = 192+64 = 256)
hidden_size          = 6144      num_hidden_layers = 78
first_k_dense_replace = 3        # layers 0-2 are DENSE MLA (no indexer)
indexer_types        : ABSENT  → derived: layer0 "full", every index_topk_freq(=1)-th "full",
                                 rest "shared"  ⇒ with freq=1, EVERY non-dense layer is "full"
                                 (its own indexer runs) ⇒ ~75 indexer invocations / forward.
n_routed_experts=256  num_experts_per_tok=8  n_shared_experts=1  moe_intermediate_size=2048
max_position_embeddings = 202752 (~200K; rope_scaling = None)   rms_norm_eps = 1e-5
quantization: bits=4, group_size=128, AutoRound (→ gptq int4-sym packing)
```
**Implication for "ultra-fast":** with `index_topk_freq=1`, the indexer runs on ~75 of 78 layers
every token. At ~200K context the indexer's per-head Q·Kᵀ is the dominant prefill cost
(`BUILD_PLAN.md:14-17`: pure-torch 200K TTFT ≈ 40-70s, a COMPUTE/kernel wall, not memory). This is
*exactly* what an SRAM-resident fused OpBackend kernel is for. (See §6.)

DeepSeek-V3.2 differs only in dims: `index_n_heads=64`, same `index_topk=2048`, same structure —
"cross-model reuse is REAL, all dims config-driven" (`BUILD_PLAN.md:8-10`). Generality §5.

---

## 2. The current dense bypass + what `hpu-dsa-ext` already gives us

### 2a. The bypass (what we are replacing)

- **`platform.py:75-76`** hard-raises: `if attn_selector_config.use_sparse: raise
  NotImplementedError("Sparse Attention is not supported on HPU.")`
  (`$HOME/AI/_src/vllm_gaudi/platform.py`).
- **`platform_patch.py`** lifts that raise but gates it: `DSA_READY=False` keeps DSA models
  correctly BLOCKED (never silently dense), and **`FULL_ATTENTION_FALLBACK=True`** is the current
  coherent path — the indexer is skipped (`hpu_mla_skip_indexer` forces `skip_topk=True`) and
  `sparse_mla_wire._sparse_forward_impl:113-120` **delegates to the genuine dense `forward_impl`**.
  So GLM-5.1 today runs **dense MLA** (correct text, but throws away DSA — the explicit non-goal,
  `glm51-dsa-1m-requirement.md:12`).
- `oot_mla.py` already plumbs `use_sparse`/`indexer`/`topk_indices_buffer` through
  `HPUMLAAttention` (:288-289) and `HPUMultiHeadLatentAttentionWrapper` (:261-269) — the wiring
  points exist; only the indexer terminal op + the sparse consume are bypassed.

### 2b. What is BUILT + VALIDATED (the executable spec for the kernel)

In `hpu-dsa-ext/hpu_dsa_ext/`:
| file | what it is | status |
|---|---|---|
| `sparse_indexer_hpu.py` | `lightning_indexer_topk` (tiled bf16 scorer) + `reference_indexer_topk` (fp32 oracle) + `forward_oot` orchestration (K-cache insert, metadata-derived causal bounds, per-request gather, M≤128 tiling) | scorer **parity-validated** (Jaccard 0.996, top-50 recall 1.0); forward_oot authored against real siblings |
| `sparse_mla_consume.py` | `sparse_mla_gather_attention` (product O(L·k)) + `sparse_mla_masked_attention` (oracle) + sentinel-scatter + NaN guard | gather==masked diff **0.011**, parity green |
| `indexer_cache_hpu.py` | flat bf16 indexer K-cache (`IndexerKCache`) + runner `get_kv_cache_spec`/`bind_kv_cache` patches | authored |
| `indexer_metadata_hpu.py` | derive causal bounds/cu_seqlens/slot_mapping from `HPUMLAMetadata` (the `DeepseekV32IndexerMetadata` does NOT exist on HPU) | authored |
| `sparse_mla_wire.py` | monkeypatch `HPUMLAAttention.forward_impl` → sparse gather-consume (prefill clean, decode authored) | authored; **bypassed by FULL_ATTENTION_FALLBACK** |
| `glm_moe_dsa_wire.py` | `DsaDims` (config-driven dims), arch binding, `skip_topk` cadence, the `DSA_READY` flip | authored |
| `tests/test_dsa_synthetic_hpu.py` | the fast CPU+optional-HPU synthetic harness (indexer parity, consume==oracle, NaN/sentinel, on-device variant) | the gate for every change |

So **the algorithm is solved and parity-proven in pure torch.** The gap to "real sparse on the box"
is NOT correctness — it is (1) the runner-plumbing landing (cache spec + metadata), (2) flipping
`FULL_ATTENTION_FALLBACK=False` + `DSA_READY=True` after on-card validation, and — the subject of
THIS design — (3) **replacing the pure-torch hot loops with captured OpBackend kernels so it is
ULTRA-FAST and SRAM-resident**, not a slow eager torch path.

### 2c. Why pure-torch is not enough (why we need the kernel)

The pure-torch path is correct but (a) eager-dispatch-bound (per-op host launches, the same wall as
the MoE dequant loop — `gaudi-single-stream-speed.md`), (b) materializes large transients
(`[M,n_head,S]` pre-sum), and (c) won't graph-capture cleanly as ~10 separate ops × 75 layers. The
OpBackend kernel fuses each component into ONE captured recipe whose intermediates are
**non-persistent** (graph-internal) → the GC bundles them into the MME with SRAM residency (the
proven MoE win — `moe_fused_expert_backend.cpp:10-18`). That is the path to NVIDIA-class indexer
throughput on Gaudi.

---

## 3. The OpBackend build path (the proven mechanism)

The reference is the **shipped, on-device-validated** `torch.ops.hpu_moe_ext.fused_expert`
(`tpc_kernel/opbackend/moe_fused_expert_backend.cpp`, relNorm 4.2e-3 on device). The mechanism, to
reuse verbatim:

1. **`struct X : habana::OpBackend`** with an `AddNode(graph, stack)` that chains
   `OpBackend::BuildNode(this, graph, {guid, {syn_inputs}, {output_attrs}, &params, sizeof})` calls.
   Each non-final BuildNode output is a **non-persistent** `synapse_helpers::tensor` whose `.get()`
   feeds the next node — that is the SRAM-resident edge (`moe_..._backend.cpp:118-131`).
2. **`SetOutputMetaFn`** — an `OutputMetaDataVector(stack)` that declares result shape+dtype
   (`moe_..._backend.cpp:108-114`).
3. **Frontend** (two TUs, pick by serving mode):
   - **eager** (`vllm-gaudi:next` upstream-PT build): `habana::eager::EagerOp<at::Tensor>{schema,
     inputs, out_shapes}.call()` (`moe_..._backend.cpp:355-369`).
   - **lazy** (`PT_HPU_LAZY_MODE=1`, the fast serving path): `habana_lazy::LazyOp<at::Tensor>{schema,
     inputs}; op.SetOutputMetaFn(...); RUN_MAYBE_WITH_ACC_THREAD(...)`
     (`moe_fused_expert_backend_lazy.cpp:81-95`). **This is the variant that captures into the decode
     HPU-graph** — the one to target for serving. The backend `AddNode` chain is shared **byte-for-byte**
     via a `*_shared.h` header; only the frontend differs (`..._lazy.cpp:20-24`).
4. **Registration (3 lines, one `.so`)**: `TORCH_LIBRARY(ns, m){ m.def("op(...)->...") }` +
   `TORCH_LIBRARY_IMPL(ns, HPU, m){ m.impl("op", frontend) }` +
   `KernelRegistry().REGISTER_HPU_BACKEND("ns::op", X)` (`moe_..._backend.cpp:382-399`).
5. **Build CPU-only** in the container: `docker run --rm vllm-gaudi:next bash build.sh` — host g++,
   links the shipped `libhabana_pytorch_backend(.upstream).so` + the cloned bridge headers,
   `-D_GLIBCXX_USE_CXX11_ABI=1`, `FMT_HEADER_ONLY`, **never** links the plugin libs
   (`build.sh:76-112`). Run only on the HPU box via the test.

The whole point (validated by MoE): a multi-node recipe that the public `UserCustomOpDescriptor` path
CANNOT express (it binds ONE guid) — `OpBackend` registered in `KernelRegistry` is exactly how the
bridge ships its own multi-node ops (`moe_..._backend.cpp:20-29`). DSA's indexer + sparse-SDPA are
multi-node; same justification.

### 3a. Bridge ops/GUIDs that exist for DSA — verified present (not invented)

Read directly from the bridge clone `tag v1.24.0-1007` (`git describe` confirmed):

- **`topk`** — `topk_gen.cpp:96-103`, GUID `"topk"`, `ns_TopkNodeV2::ParamsV4{bsw=k, axis=tpc(dim),
  bottomK=!largest}`; supports **dynamic-K** via `kType=K_TENSOR_SHAPE` + a shape tensor (the
  per-bucket variable-k lever). Outputs (values, indices=int64). This is the indexer's top-k.
- **`gather_elements_fwd`** — `gather_gen.cpp:121-127` (full `torch.gather`, used when self.dim()==
  index.dim()); **`gather`/`ns_GatherKernel`** for `index_select` (`index_select_gen.cpp:23-29`,
  axis param). These are the KV gather (gather selected rows along the key axis).
- **`sdpa_fwd`** — `sdpa_gen.cpp:319-384`, GUID `get_guid_with_precision("sdpa_fwd", dtype)`,
  `ns_Sdpa::ParamsV7{scale, is_causal, softmax_mode, flags, wl/wr=window}`. **Takes an optional
  `attention_mask` syn-input** (`:354-358`) — i.e. it accepts our additive sparse+causal bias
  directly. Also `valid_seq_len` + left/right seq-padding flags (`:342-344`). This is the sparse SDPA.
- **`fp8_sdpa_fwd`** — `sdpa_gen.cpp:386-502` (e4m3/e5m2, d_scale_q/k/v + q_scale_s/o) — the fp8
  attention variant if we want fp8 KV in the consume.
- **`sdpa_recomp_fwd`** — `sdpa_gen.cpp:810-910`, adds an explicit **`window_size`** (sliding window)
  + a **`sink`** input — useful for the streaming/long-context tile.
- **`scaled_masked_triangular_softmax_fwd`** — `scaled_masked_softmax_gen.cpp:61-113`,
  `ns_ScaledMaskedSoftmax::Params{...,expMode}` — a FUSED scale+mask+causal+softmax (the exact
  masked-MHA consume oracle, as one MME-adjacent node).
- **`masked_batch_gemm`** — `masked_batch_gemm_gen.cpp:50-88` (Gaudi-2 ONLY, asserted), 4D a@b with
  per-operand 4D masks + `synGEMMParams{transpose_a,transpose_b}`. A hard-masked MME — a candidate
  for fusing the indexer's masked Q·Kᵀ in one node (the masks gate the causal triangle inside the MME).
- **`gemm`/`batch_gemm`** — `matmul.cpp` (PyTorch-order `self[M,K]@other[K,N]→[M,N]`,
  `synGEMMParams{false,false}`), `bmm_gen.cpp`. The indexer Q·Kᵀ and the head-sum GEMM.
- **fp8 logits chain**: `convert_to_fp8` (`fp8_ops.cpp:153`, `ns_ConvertToFp8::ParamsV2`), `fp8_gemm`
  (`fp8_gemm_gen.cpp:206-223`, `ns_Fp8Gemm::ParamsV2`), `softmax_fp8` (`softmax_fp8_gen.cpp`).
  Already speced for the MoE fp8 op (`FUSED_GENERALIZE_FP8_DESIGN.md:96-113`) — same constructions.
- **K-cache + index machinery**: `index_copy` (`ns_IndexCopy::Params`), `scatter_fwd`
  (`ns_ScatterKernel`), `cumsum_fwd` (`ns_CumSumKernel`), `where_fwd`, `range` (`ns_RangeKernel`),
  `masked_fill_fwd` (`ns_MaskedFill::ParamsV2`). These build the flat-cache insert, the sentinel
  scatter (−1 pad → trash column), the cumsum-pack of gather indices, and the causal bound.

**What does NOT exist as a bridge op and must be composed (not ISA-written):** the paged/flat
attention (`flat_pa_mla`) is vllm-gaudi *Python* over bridge `bmm`+`softmax`, not a single GUID
(only `sdpa_gen.cpp` is in `hpu_ops/backend/` for attention). And the DeepGEMM `fp8_fp4_mqa_logits`
/ `top_k_per_row` / `indexer_k_quant_and_cache` are **CUDA-only** — they are exactly what we replace
with the `gemm`+`topk`+`index_copy` chains above. **No bridge ISA kernel is missing.**

---

## 4. The OpBackend kernel design (per component)

Two captured ops, mirroring the two validated torch functions. Register under a fresh library
namespace `hpu_dsa_ext` (like `hpu_moe_ext`), one eager `.so` + one lazy `.so` sharing a
`*_shared.h` backend.

### 4a. OP 1 — `dsa_indexer_topk` (the lightning indexer)

Replaces `lightning_indexer_topk` + the per-request scoring in `sparse_indexer_hpu.py`.

**Schema** (config-parameterized — nothing hardcoded; dims derived from tensor shapes):
```
dsa_indexer_topk(Tensor q,        # [T, n_head, head_dim]  bf16 (fp8 q already upcast; scale folded)
                 Tensor k,        # [S, head_dim]          bf16 indexer keys (MQA, gathered ABS order)
                 Tensor weights,  # [T, n_head]            bf16 (q_scale*softmax_scale*n_head^-0.5 folded)
                 Tensor abs_pos,  # [T] int32              query absolute key-axis position (causal)
                 int    topk      # index_topk (2048)
                ) -> Tensor       # [T, topk] int32, -1 padded
```
**AddNode node-chain** (BuildNode sequence; PyTorch row-major, last dim = FCD, per
`moe_..._backend.cpp:162-191`):
```
1. logits = batch_gemm(q[T,H,D], k[S,D] broadcast→[T or 1,S,D], transpose_b=True)  → [T,H,S]
      (or masked_batch_gemm to fold the causal mask into the MME directly)
2. logits = relu_fwd(logits)                                                        → [T,H,S]
3. score  = einsum "ThS,Th->TS": gemm/mult_fwd(logits, weights[T,H]) + reduce_sum(H)→ [T,S]
4. score  = masked_fill_fwd(score, key_pos[1,S] > abs_pos[T,1], -inf)               → [T,S]  (causal)
      (key_pos via `range`; broadcast compare → mask; OR fold into step-1 masked_batch_gemm)
5. idx_i64 = topk(score, k=min(topk,S), dim=-1, largest=True).indices               → [T,k] i64
      (dynamic-K bucket via kType=K_TENSOR_SHAPE so one recipe serves all S buckets)
6. valid  = gather(score, idx) > -inf ; out = where(valid, idx_i32, const(-1))      → [T,topk] i32
      (sentinel -1 for short/empty rows; cast i64→i32; right-pad to topk with -1)
```
- **Tiling for memory**: the MoE op shows BuildNode handles large MME tiles; for very long S the
  graph slices T into M≤128 tiles (the GC's SRAM slicer bundles per-tile, mirroring `BUILD_PLAN.md:46-48`
  "M≤128 tiles, do NOT materialize [M,H,S] monolithically"). The `[T,H,S]` transient stays
  non-persistent → never hits HBM at full size.
- **fp8 fast path** (phase 3): replace step 1 with `convert_to_fp8(q)` + `fp8_gemm` + the
  per-token-group scale (the real `fp8_fp4_mqa_logits` semantics) — `FUSED_GENERALIZE_FP8_DESIGN.md:96-113`
  gives the exact `convert_to_fp8`/`fp8_gemm` constructions. Keeps the indexer in fp8 (2× MME).
- **Output meta**: `[T, topk]` int32.

### 4b. OP 2 — `dsa_sparse_attn` (gather-consume sparse attention)

Replaces `sparse_mla_gather_attention` in `sparse_mla_consume.py`. Operates on per-head MLA q/k/v
(prefill) or gathered latent (decode).

**Schema:**
```
dsa_sparse_attn(Tensor q,            # [T, H, qk_head]    per-head MLA query (nope‖rope)
                Tensor k,            # [S, H, qk_head]    per-head MLA key   (MQA expanded)
                Tensor v,            # [S, H, v_head]     per-head MLA value
                Tensor topk_indices, # [T, index_topk]    int32, -1 padded (from OP 1)
                Tensor abs_pos,      # [T] int32
                float  scale,
                int    gather_width  # K bucket (>= max kept-count; default min(index_topk,S))
               ) -> Tensor           # [T, H, v_head]
```
**AddNode node-chain:**
```
1. keep[T,S] = sentinel-scatter(topk_indices) & causal(key_pos<=abs_pos) ∪ self-diag
      (scatter_fwd into width S+1 trash column, slice off; `range`+compare causal; force self)
2. gather_idx[T,K], valid[T,K] = cumsum-pack(keep, K)
      (cumsum_fwd exclusive prefix → rank; where(lands, rank, K) → scatter col_ids; slice)
3. k_g = gather(k_ext[S+1,H,Dh], gather_idx) ; v_g = gather(v_ext, gather_idx)   → [T,K,H,D]
      (append one zero sentinel row at index S; index_select on flattened key axis)
4. out = sdpa_fwd(q[H,T,1,Dh], k_g[H,T,K,Dh], v_g[H,T,K,Dv],
                  attention_mask = pad_bias[T,K] (-inf on !valid),
                  scale, is_causal=False, softmax_mode="fp32")                   → [H,T,Dv]
      (causality already in the gather geometry; mask only the pad slots; nan→0 guard)
5. out = transpose → [T,H,Dv]
```
- **Why `sdpa_fwd` and not a hand-softmax**: the bridge `sdpa_fwd` takes the additive mask as a
  syn-input (`sdpa_gen.cpp:354-358`) and runs the whole QKᵀ-softmax-PV inside one fused CGUID with
  the GC's flash tiling — far faster than the torch `matmul→softmax→matmul` the oracle uses. K is the
  small gathered width (≤2048), so this is the O(L·k) product path natively.
- **≤64K oracle variant** (parity only): one `scaled_masked_triangular_softmax_fwd` over the full
  `[T,S]` bias — for the on-device parity gate vs the gather path.
- **Decode**: gather the request's paged latent (current step already written by
  `impl.latent_cache_k`), absorb via `W_UK_T`/`W_UV` to per-head (or keep latent and use the MQA
  absorbed form), then the same node-chain over its `[0,ctx)` keys (`sparse_mla_wire.py:160-174`).

### 4c. Indexer-KV-cache handling (the plumbing, mostly already built)

- **Insert** (OP-adjacent or a tiny 3rd op `dsa_kcache_insert`): `index_copy`/`scatter_fwd` of the
  current-step bf16 indexer key `k[T,head_dim]` into the flat cache at `slot_mapping`
  (`indexer_cache_hpu.py` `IndexerKCache.insert`). No fp8 quant (HPU flat-bf16 design) → no
  `indexer_k_quant_and_cache` needed.
- **Read** (feeds OP 1's `k`): `gather`/`index_select` the request's prefix blocks in block(==abs)
  order + the current step (`_gather_request_keys`, `sparse_indexer_hpu.py:151-190`). The
  `cp_gather_indexer_k_quant_cache` CUDA kernel → plain `gather` here.
- **Runner allocation**: patch vllm-gaudi `get_kv_cache_spec`/`bind_kv_cache` to allocate a bf16
  `(num_blocks*block_size, index_head_dim)` cache (`indexer_cache_hpu.apply()`). **This is the single
  hardest plumbing item** and it is already authored in `indexer_cache_hpu.py` (needs on-card
  validation). It is NOT a kernel concern — the OpBackend ops take the cache tensor as input.
- **Metadata**: `DeepseekV32IndexerMetadata` does not exist on HPU → derive causal bounds /
  cu_seqlens / slot_mapping / block tables from `HPUMLAMetadata` (`indexer_metadata_hpu.py`,
  `build_indexer_metadata`). Again Python, already authored, NOT a kernel concern.

So the kernel scope is the two captured graphs in §4a/§4b; the cache/metadata stays Python (built),
and the OpBackend ops consume the cache tensor + the derived index tensors as plain inputs.

---

## 5. Generality strategy (DeepSeek-V3.2 / GLM / any DSA model, little per-model config)

The MoE op proved this works: "the op is fully shape-parameterized … nothing is hardcoded to GLM's
dims … no new per-model config is required" (`FUSED_GENERALIZE_FP8_DESIGN.md:10-60`). Apply the same:

1. **All dims from tensor shapes / config, never literals.** `dsa_indexer_topk` derives
   `T,n_head,head_dim` from `q.sizes()`, `S` from `k.sizes()`, `topk` is a scalar arg. So
   `index_n_heads` 32 (GLM) vs 64 (DeepSeek) "just works"; `index_head_dim`, `index_topk`,
   MLA `kv_lora_rank/qk_nope/qk_rope/v_head` all flow from the existing `DsaDims` config reader
   (`glm_moe_dsa_wire.py:72-150`, `read_dsa_dims`).
2. **One arch path for the whole family.** GLM-5.1 + DeepSeek-V3.2/V4 share one `Indexer` + one
   sparse MLA path (`BUILD_PLAN.md:8-10`; `GlmMoeDsaForCausalLM(DeepseekV2): pass`). The wire layer
   binds the arch name to the canonical in-tree class; the ops are arch-agnostic.
3. **rope style is data, not code.** GLM uses `indexer_rope_interleave=True` (interleave) vs
   DeepSeek/NeoX; the rope is applied UPSTREAM of the op (in `Indexer.forward`, producing the q/k the
   op consumes), so the op never sees rope style. `DsaDims.indexer_rope_interleave` carries it
   (`glm_moe_dsa_wire.py:83`).
4. **fp8-vs-bf16 indexer is a build toggle**, not a fork (mirror the MoE `BUILD_FP8_NODES` gate) —
   one `.so` with a compile flag selecting the `gemm` vs `convert_to_fp8`+`fp8_gemm` node bodies.
5. **skip_topk cadence is config** (`indexer_types`/`index_topk_freq`): shared layers reuse the prior
   buffer (the op simply isn't called for them) — `DsaDims.layer_skips_topk`
   (`glm_moe_dsa_wire.py:91-96`), matching HF `next_skip_topk` (`modeling_glm_moe_dsa.py:340-342`).

Net: a new DSA model needs **zero kernel changes** and only its `config.json` (autodetected),
exactly like the MoE op.

---

## 6. Ultra-fast strategy (the hard requirement)

1. **True top-k GATHER, never O(L²) masked.** The product op (§4b) gathers ≤index_topk KV rows and
   SDPAs over width K, so memory+compute are O(L·index_topk) = O(L·2048), independent of S. The masked
   `[T,S]` path is the parity ORACLE only (`sparse_mla_consume.py:14`, `BUILD_PLAN.md:27-28`). At 200K:
   gather path ≈ 0.54GB vs masked 81GB+473s (`BUILD_PLAN.md:28`).
2. **SRAM-resident fused recipes.** Both ops emit non-persistent intermediates → the GC bundles
   dequant/score/softmax into the MME with SRAM residency (the proven MoE lever,
   `moe_..._backend.cpp:10-18`). The `[T,H,S]` indexer pre-sum and the gathered `[T,K,H,D]` never
   round-trip HBM at full size.
3. **Captured under LAZY** (the fast serving path). Build the lazy frontend TU
   (`..._lazy.cpp`) so both ops record into the decode HPU-graph and replay with zero host dispatch —
   this is what kills the per-op eager-launch wall (the same wall as the MoE dequant loop,
   `gaudi-single-stream-speed.md`). The box already serves GLM under lazy with the MoE op captured
   (`glm51-dsa-1m-requirement.md` "LAZY MODE — SOLVED, SERVES").
4. **Use `sdpa_fwd`'s fused flash kernel** for the consume (one CGUID does QKᵀ+softmax+PV with
   tiling) instead of three torch ops — `sdpa_gen.cpp:319-384`.
5. **fp8 indexer logits** (phase 3): `index_topk=2048` over ~75 layers is MME-bound at long ctx; fp8
   Q·Kᵀ ≈ 2× MME throughput (`FUSED_GENERALIZE_FP8_DESIGN.md:113`) and is the real
   `fp8_fp4_mqa_logits` semantics. This is the single biggest prefill-TTFT lever (`BUILD_PLAN.md:14-17`).
6. **Dynamic-K bucketing** via `topk` `K_TENSOR_SHAPE` (`topk_gen.cpp:91-93`) so one captured recipe
   serves all sequence-length buckets without recompiling per shape.
7. **`masked_batch_gemm`** (Gaudi-2 only, `masked_batch_gemm_gen.cpp`) can fold the causal mask into
   the indexer MME — fuses steps 1+4 of OP 1 into one masked MME node (a measured-on-device option).

Realistic ceiling note: per the broader serving analysis, single-stream tok/s is gated by the MoE
bandwidth/fabric wall regardless of attention (`gaudi-perf-targets.md`); DSA's win is **making long
context tractable at all** (1M is O(L²)-impossible dense — `glm51-dsa-1m-requirement.md:13`) and
cutting prefill TTFT, not raising the batch=1 decode ceiling. Frame "ultra-fast" as: SRAM-resident,
captured, O(L·k) — the same engineering bar the MoE kernel met.

---

## 7. Build order — MVP → full

Gate EVERY step on `tests/test_dsa_synthetic_hpu.py` (CPU parity) before the on-card run, and keep
`FULL_ATTENTION_FALLBACK=True` / `DSA_READY=False` until the full sparse path validates (the
`platform_patch.py` safety contract — never silently dense).

- **MVP-1 — `dsa_indexer_topk` OpBackend (eager .so), prefill, bf16.** Port `lightning_indexer_topk`
  to the §4a node-chain. CPU-gate: op output top-k index set == `reference_indexer_topk` within ties
  (Gate D, `BUILD_PLAN.md` / `test_dsa_synthetic_hpu.py` (a)). On-card: synthetic `[T,n_head,head_dim]`
  → matches the torch scorer. **This proves the indexer runs as a captured kernel.**
- **MVP-2 — `dsa_sparse_attn` OpBackend (eager .so), prefill, ≤64K.** Port `sparse_mla_gather_attention`
  to the §4b node-chain. CPU-gate: op == `sparse_mla_masked_attention` oracle (diff ~0.011). On-card:
  wire into `sparse_mla_wire._sparse_forward_impl` prefill branch (replace the torch call with the op).
- **MVP-3 — integrate + flip.** Land `indexer_cache_hpu` + `indexer_metadata_hpu` runner patches
  (already authored), set `forward_oot` to call OP 1, set the consume to call OP 2,
  `FULL_ATTENTION_FALLBACK=False`, `DSA_READY=True` AFTER: (i) indexer-fired counter > 0 on a
  ctx>2048 prompt (proves not dense), (ii) ctx≤index_topk sparse == dense-MLA baseline, (iii) ctx 16K
  sparse ≠ dense (proves not-silently-dense) — the `IMPLEMENTATION.md:78-85` gates. Serve GLM-5.1
  sparse at 32-64K, bf16.
- **Phase 4 — lazy frontend .so** (`..._lazy.cpp` sharing the backend header) so both ops capture
  into the decode HPU-graph under `PT_HPU_LAZY_MODE=1` (the fast serving path). Re-validate coherence.
- **Phase 5 — fp8 indexer logits** (`convert_to_fp8`+`fp8_gemm` node toggle) — the prefill-TTFT lever,
  looser tol gate (mirror `test_fused_expert_fp8_op.py`).
- **Phase 6 — decode paged consume + ~1M streaming tile** (sliding-window `sdpa_recomp_fwd`,
  latent-space gather-then-absorb). NOTE: GLM-5.1's trained ctx is ~200K (`max_position_embeddings=
  202752`, no rope_scaling) — true 1M needs a 1M-native model in its own lane
  (`glm51-dsa-1m-requirement.md:13`); the kernel is ctx-agnostic but the model isn't.

---

## 8. Honest feasibility verdict + top risks

**Feasible. GO.** The OpBackend path turns DSA from "impossible CUDA port" into "compose existing
bridge ops" — every primitive (topk, gather, sdpa with mask, fp8_gemm, scatter/cumsum/range) is a
real GC-lowered node in the version-exact bridge, and the **algorithm is already parity-validated in
pure torch** (the executable spec + oracle). The MoE op is the existence proof that a multi-node
SRAM-resident OpBackend kernel loads, captures under lazy, and is numerically correct on this exact
box/stack. DSA is the same shape of problem.

**Top risks (ranked):**
1. **Synapse FCD geometry / SIF shape desync** — the MoE op needed multiple on-device iterations to
   get BuildNode output shapes in PyTorch-row-major so the complex-guid SIFs didn't null-deref
   (`moe_..._backend.cpp:162-191`). The indexer's `[T,H,S]` einsum + the gather's flattened-key
   `index_select` are the same hazard class. Mitigation: mirror the MoE's row-major discipline; build
   each node against the validated torch reference shape; on-card bring-up loop.
2. **`topk` over a long key axis (S up to 200K)** — `topk_gen.cpp` asserts `k <= size(dim)` and the
   CGUID's perf/feasibility at S=200K, k=2048 is unproven on Gaudi-2 (CUDA uses a radix `persistent_topk`
   for exactly this — `sparse_attn_indexer.py:337-349`). Mitigation: M≤128 tiling bounds the score
   width per call; if `topk` is slow at huge S, fall back to `sort_fwd` (`sort_gen.cpp`, stable) + slice,
   or a 2-stage block-topk. This is the most likely "needs measurement" item.
3. **Indexer-KV-cache runner plumbing** — `get_kv_cache_spec` silently drops the indexer cache
   (`BUILD_PLAN.md:42`); the patch is authored but unvalidated on-card, and prefix-caching must be OFF
   (`--no-enable-prefix-caching`, `sparse_mla_wire.py:9`). Mitigation: it is Python, gated, and the
   non-prefix path is the clean one.
4. **MLA latent-vs-per-head in consume** — vllm-gaudi MLA absorbs V into the latent for decode
   (`oot_mla.py:121-145`); the gather-consume currently materializes per-head (≤64K). The ~1M path
   needs gather-in-latent-then-absorb. Mitigation: ship ≤64K per-head first (built), latent-gather is
   phase 6.
5. **Two-image reality** — the OpBackend builds against `vllm-gaudi:next` (lazy PT-2.10 fork) but the
   live serve is on `vllm-gaudi:1.24`. Confirm the DSA `.so`'s lazy frontend matches whichever image
   serves GLM (the MoE op already serves under lazy, so this is a known-good combo, but verify the DSA
   `.so` against the SAME image at integration).
6. **fp8 indexer double-quant accuracy** — int4-checkpoint + fp8-logits is two quantizations; expect
   coherent-but-looser (`FUSED_GENERALIZE_FP8_DESIGN.md:115-132`, the fn-vs-fnuz trap). Mitigation:
   bf16 indexer is the default/correct path; fp8 is an opt-in speed phase with a looser on-card gate.

**What this design deliberately does NOT claim:** that DSA raises the single-stream decode ceiling
(it doesn't — that's MoE-bandwidth/fabric-bound, `gaudi-perf-targets.md`), or that GLM-5.1 is
1M-native (it isn't, ~200K trained). DSA's value is correctness-of-design + long-context tractability
+ prefill TTFT, delivered as a real captured kernel instead of a dense bypass.

---

## 9. Key file/symbol index (for the build)

**Build the kernel from:**
- Mechanism template: `tpc_kernel/opbackend/moe_fused_expert_backend.cpp` (eager) +
  `moe_fused_expert_backend_lazy.cpp` (lazy) + `moe_fused_expert_backend_shared.h` + `build.sh`/`build_lazy.sh`.
- Bridge ops (GUIDs): `tpc_kernel/opbackend/bridge/hpu_ops/backend/{topk,gather,index_select,sdpa,
  scaled_masked_softmax,masked_batch_gemm,fp8_gemm,fp8_ops,scatter,cumsum,where,arange,index_copy,
  masked_fill}_gen.cpp` + `matmul.cpp`, `bmm_gen.cpp`.
- Generalization pattern: `tpc_kernel/opbackend/FUSED_GENERALIZE_FP8_DESIGN.md`.

**Port (the executable spec + oracle):**
- Indexer math: `hpu-dsa-ext/hpu_dsa_ext/sparse_indexer_hpu.py` (`lightning_indexer_topk`,
  `reference_indexer_topk`, `forward_oot`).
- Consume math: `hpu-dsa-ext/hpu_dsa_ext/sparse_mla_consume.py` (`sparse_mla_gather_attention`,
  `sparse_mla_masked_attention`).
- Cache/metadata (Python, built): `indexer_cache_hpu.py`, `indexer_metadata_hpu.py`.
- Wire/gate: `sparse_mla_wire.py`, `glm_moe_dsa_wire.py`, `platform_patch.py`
  (`FULL_ATTENTION_FALLBACK`, `DSA_READY`).
- Test gate: `hpu-dsa-ext/tests/test_dsa_synthetic_hpu.py`.

**Authoritative DSA references (read-only):**
- HF: `transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` (`:144-228` indexer, `:394-459`
  consume), `configuration_glm_moe_dsa.py` (in `vllm-gaudi:next` image, transformers 5.10.2).
- vLLM: `$HOME/AI/dsa_src/vllm_core_deepseek_v2.py:603-743` (`Indexer`),
  `$HOME/AI/dsa_src/sparse_attn_indexer.py` (CUDA op contract),
  `$HOME/AI/_src/vllm_gaudi/attention/oot_mla.py` (MLA), `…/platform.py:75-76` (the raise).
- Config: `/mnt/fastmodels/GLM-5.1-int4-mixed-AutoRound/config.json`.
- DeepSeek-V3.2 DSA paper: arXiv 2512.02556 (the masked-MHA §2.3 + indexer math).
