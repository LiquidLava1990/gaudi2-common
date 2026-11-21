# hpu-dsa-ext — DeepSeek Sparse Attention on HPU, as designed

Goal: GLM-5.1 (`GlmMoeDsaForCausalLM`) and DeepSeek-V3.2-class run with **real**
sparse attention on Gaudi — the lightning indexer's top-k token selection is
actually computed and actually restricts attention. **No dense fallback.**

All paths below are verified against the in-image source extracted to
`$HOME/AI/dsa_src/` (vllm 0.22.1rc1 / vllm_gaudi 0.20.1rc1).

## Data flow (in-image, verified)
`MultiHeadLatentAttentionWrapper.forward` (`dsa_src/mla.py:119-181`):
1. builds `q`, `kv_c_normed`, `k_pe` (MLA preprocess), applies rope.
2. **`mla.py:168-169`**: `if self.indexer and self.is_sparse and not self.skip_topk: self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)`
   — the indexer runs and **fills `topk_indices_buffer` as a side effect**.
3. **`mla.py:174`**: `self.mla_attn(q, kv_c_normed, k_pe, output_shape=...)` runs attention.

The HPU `HPUMultiHeadLatentAttentionWrapper` (`oot_mla.py`) subclasses this and
does NOT override `forward`, so step 2 (indexer) DOES fire on HPU — **but** the
indexer's terminal op has no HPU impl, and `HPUMLAAttention.forward_impl` ignores
the buffer (runs dense). Those are the two gaps.

## The real Indexer already does the trained-weight math (in-image)
`dsa_src/vllm_core_deepseek_v2.py` `class Indexer` (603-743):
- `wk_weights_proj = MergedColumnParallelLinear(...)` — fused WK + weights_proj GEMM (TRAINED weights).
- `forward`: `kw,_ = self.wk_weights_proj(hidden_states)` -> split into `k` and `weights`;
  partial rope on indexer keys; `q_fp8, q_scale = per_token_group_quant_fp8(q...)`;
  `return self.indexer_op(hidden_states, q_fp8, k, weights)`.
So the projections, fp8 quant, rope are DONE. **We only implement the terminal op.**

## Piece 1 — `SparseAttnIndexer.forward_hpu`  (sparse_indexer_hpu.py)
`dsa_src/sparse_attn_indexer.py`: `class SparseAttnIndexer(CustomOp)` has
`forward_cuda/xpu/hip` but **no HPU path** (`forward_native` raises on non-CUDA).
`forward_cuda` calls `torch.ops.vllm.sparse_attn_indexer(...)` which: inserts the
indexer key into a **paged indexer K-cache** (`self.k_cache`, a `DeepseekV32IndexerCache`),
computes MQA logits of `q_fp8` against ALL cached keys up to each query position
(causal), `top_k_per_row` (k=`topk_tokens`=`index_topk`=2048), writes ids into
`topk_indices_buffer` (`-1` = unselected).

Implement via `@SparseAttnIndexer.register_oot` (custom_ops=["all"] at
`platform.py:206` auto-enables it). HPU torch ops only — verified present:
`torch.matmul`, `torch.topk`, `torch.gather`, `FusedSDPA`. NOT DeepGEMM.

Correct logit (DeepSeek-V3.2 paper arXiv 2512.02556, fold ReLU/scale into `weights`):
`I[t,s] = Σ_head weights[t,head] * (q[t,head,:] · k[s,:])`, for `s <= t` (causal), else `-inf`.
Then `topk_indices = topk(I[t], k=min(index_topk, t+1))`; fill buffer, `-1`-pad.

Hard parts (this is the real work):
- **Paged indexer K-cache**: must store/read indexer keys across all past tokens
  (prefill writes range; decode appends one). Use the `DeepseekV32IndexerCache`
  the model already allocates; read its `prefix`/`kv_cache` like `forward_cuda` does.
- **Causality** via `attn_metadata` cu_seqlens (block tables on decode).
- **Prefill vs decode** branches (mirror `top_k_per_row_prefill` / `_decode`).
- **Graph capture**: fixed-shape topk (k constant per bucket); `mark_step()`.

NOTE: PR #1301's `vendor/hpu_sparse_attn_indexer.py` is a starting shell only —
its scoring is wrong (no ReLU/weights-per-head, no causal, no paged cache) and
its output is unused. Use it for structure, not math.

## Piece 2 — sparse consume in `HPUMLAAttention.forward_impl`  (sparse_mla.py)
`dsa_src/hpu_attn.py`: `forward_mha` (245) takes `attn_bias=attn_metadata.attn_bias`;
`forward_mqa` (309) takes `block_bias=attn_metadata.attn_bias`. Both add the bias
into the logits (FusedSDPA / `flat_pa_mla`). So **masked-MHA mode** (DeepSeek paper
§2.3) = build additive bias from `topk_indices_buffer`:
`B[t,s] = 0 if s ∈ topk(t) else -inf`, combined with the causal mask, passed as
`attn_bias`/`block_bias`. Result == exact as-designed sparse attention.
- Subclass/patch `HPUMLAAttention.forward_impl` (oot_mla.py:106-145) to, when
  `self.use_sparse`, scatter `topk_indices_buffer` into a `[T, kv_len]` (or block)
  `-inf` mask and merge into `attn_metadata.attn_bias` before `forward_mha/mqa`.
- Bring-up: full `[T,kv]` mask (O(L²) mem, correct). Optimization (later): gather
  selected latent KV for true O(Lk). Correctness first per mandate.

## Piece 3 — register `GlmMoeDsaForCausalLM`  (glm_moe_dsa.py)
GLM-5.1 arch `GlmMoeDsaForCausalLM` (`index_topk=2048, index_n_heads=32,
index_head_dim=128, kv_lora_rank=512, qk_nope_head_dim=192`). Map to the
DeepSeek-V3 sparse path (it shares the deepseek_v2 MLA+Indexer structure). Read
all dims from `config` — hardcode nothing (DeepSeek uses 64 indexer heads).

## Activation + validation (then flip `platform_patch.DSA_READY=True`)
Gate D (`tests/parity_indexer.py`): capture `(hidden, q_fp8, k, weights)`; assert
HPU `forward_hpu` top-k **index set** == a pure-torch reference within ties;
exactly `index_topk` non-`-1`/row; `-1` for short/masked rows.
Gate A.c: per-rank counter in `forward_hpu`; ctx>2048 prompt => counter>0 per
sparse layer (proves indexer fired, not dense).
Cross-check: ctx<=index_topk sparse output == GLM-4.6 dense-MLA baseline (must
agree); ctx 16K sparse != dense (proves not-silently-dense). Only then DSA_READY=True.

## Launch (after activation)
`models/glm51.env`: `--tensor-parallel-size 8 --enable-expert-parallel
--max-model-len 32768 PT_HPU_LAZY_MODE=0 --block-size 128`. AWQ gated on
`hpu-moe-ext/tests/awq_numeric_proof.py`; AutoRound is the fallback checkpoint.
