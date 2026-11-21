# HPU DSA — Verified Build Plan (from the 15-agent search→design→verify run, 2026-06-04)

**Verdict: GO-WITH-CHANGES.** Buildable in pure HPU torch ops. No portable kernel base exists
(vllm-ascend has full DSA but it's 100% AscendC ISA kernels — PR #3270, NOT #6339). The algorithm
reproduces cleanly and was **numerically validated on real Gaudi 2**: indexer bf16 ↔ DeepSeek fp8
oracle Jaccard **0.996**, **top-50 token recall 1.0000**, weighted attention-mass recall ≥0.992.
Reference oracles (all license-clean, used as oracles only — no code lifted): HF transformers
`GlmMoeDsaIndexer` (pure-torch), DeepSeek `inference/model.py`+`kernel.py` (math), SGLang `dsa/`
(structure). **Cross-model reuse is REAL** — GLM-5.1 + DeepSeek-V3.2/V4 share one `Indexer` + one
`SparseAttnIndexer` + one MLA path, all dims config-driven (`GlmMoeDsaForCausalLM(DeepseekV2): pass`).

## SCOPE (honest, from the perf verifier)
- **Correct at ANY length. Productizable to ~32–64K context in pure torch.**
- **~200K needs a custom TPC kernel** (fused logits + streaming top-k). Pure-torch prefill TTFT at
  200K ≈ 40–70s (indexer matmul K=128 poor MME util 60–200s + topk-as-sort + random gather 30–45s).
  Memory (45.8/96GB) and decode (~11–30ms/tok) are FINE at 200K — **prefill TTFT is the wall**, and
  it's a compute/kernel wall, not memory. So: ship sparse GLM-5.1 at 32–64K now; 200K = phase 2 (TPC).

## The correct math (one formula — resolved across all designs)
```python
q = q_fp8.to(bfloat16)                      # NO q_scale, NO k_scale (folded weights make this EXACT)
# logits[t,s,h] = relu(q[t,h,:] · k_bf16[s,:]); weights already carry q_scale·softmax_scale·n_head^-0.5
score = (relu(einsum('thd,sd->tsh', q, k_bf16)) * weights[:, None, :]).sum(-1)   # [T, S]
score = score.masked_fill(~causal_mask, -inf)        # s <= abs_pos(t), self included
topk_idx = score.topk(min(index_topk, S), dim=-1).indices    # [T, index_topk], -1 pad short rows
```
Consume (>32K): **gather** the ≤2048 selected latent-KV rows per query → FusedSDPA over 2048
(O(L·2048), 0.54GB@200K). Masked-MHA additive `-inf` bias is the ≤32K oracle only (81GB + 473s @200K).

## MUST-FIX (the panel's blockers — these sank all 5 raw designs)
1. **`forward_oot`, NOT `forward_hpu`.** HPU is out-of-tree → `CustomOp.dispatch_forward` routes to
   `forward_oot` (→ `forward_native` → raises). Override `SparseAttnIndexer.forward_oot`. Pattern:
   `vllm_gaudi/ops/hpu_grouped_topk_router.py:112`. Without this the indexer NEVER runs.
2. **`get_forward_context().attn_metadata` is a `TrimmedAttentionMetadata` namedtuple, NOT a dict.**
   `attn_metadata[k_cache.prefix]` raises. `DeepseekV32IndexerMetadata` + its fields
   (`prefill.chunks`, `cu_seqlen_ks/ke`, `decode.schedule_metadata`) DO NOT EXIST on HPU. Derive
   causal bounds from `HPUMLAMetadata`: prefill = `seq_lens_tensor`+`query_start_loc`+`context_lens_tensor`;
   decode = `seq_lens_tensor`+`block_list`/`block_groups`/`block_usage`.
3. **Allocate the indexer K-cache YOURSELF.** HPU runner's `get_kv_cache_spec` only handles
   Attention/MLAAttention/MambaBase/FusedMoE — it silently drops `DeepseekV32IndexerCache` → cache
   stays `tensor([])` → crash. Override to a bf16, head_dim=128, flat `(num_blocks*block_size,1,128)`
   layout (NOT the uint8/132 fp8-packed default at `vllm_core_deepseek_v2.py:654`). This is a
   vllm-gaudi model-runner edit (the hardest plumbing — none of the raw designs scoped it).
4. **Bypass `DeepseekV32IndexerMetadataBuilder`** (Triton `@jit` + CUDA SM-count + DeepGEMM — can't run on HPU).
5. **`-1` padding via sentinel-column scatter, NOT `clamp(min=0)`** (clamp falsely marks token 0 selected).
6. **NaN guard:** an all-(-inf) row → softmax NaN. Guarantee ≥1 valid key (self) or post-clamp NaN→0.
7. **No `mark_step()` for memory bounding** (no-op in PT_HPU_LAZY_MODE=0). Use **M≤128 tiles + `del`**;
   do NOT compute the `[M,H=32,S]` pre-sum monolithically (17GB/tile @200K) and do NOT head-loop (worse).
8. Honor `skip_topk` (shared layers reuse a prior buffer) + `first_k_dense_replace=3` (layers 0-2 dense).

## Build order (target 32K first; Qwen flagship stays serving until load-test)
1. `sparse_indexer_hpu.py` — `SparseAttnIndexer.forward_oot` (the math above) + `tests/parity_indexer.py`
   vs HF `GlmMoeDsaIndexer` oracle (Jaccard + top-50 recall gates). **Numerically provable off-hardware.**
2. Indexer K-cache override + HPU metadata derivation (the runner plumbing — BLOCKERS 2/3/4).
3. `sparse_mla.py` — gather-consume + sentinel scatter + NaN guard, into `forward_mha`/`forward_mqa`.
4. Wire `GlmMoeDsaForCausalLM`; gate (indexer-fired ctx>2048, top-k parity, sparse≠dense@long); flip `DSA_READY`.
Reuse for DeepSeek-V3.2/V4 = config only (verified). Full result: `WORKFLOW_RESULT_search-design-verify.json`.
