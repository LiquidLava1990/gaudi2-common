# DSA / GLM-5.1 — CRITICAL CORRECTION (2026-06-04)

## Finding: vllm-gaudi PR #1301 does NOT implement working HPU sparse attention.
Verified line-by-line against vendored source (`hpu-dsa-ext/vendor/`, PR head 708aa1f):

- **`hpu_deepseek_v3.py`** — SKELETON. `HpuDeepseekV3Attention.forward` = *"TODO ... For now, use base implementation"* → `super().forward()` (**DENSE**). `__init__` only `print()`s. Full of TODO / `pass`.
- **`sparse_attention.py`** — STANDALONE TOY. Builds its own `q/k/v_proj` (random init), `n_selected=16 / d_model=512` demo defaults, `__main__` test. NOT wired to MLA / KV cache / real model weights.
- **`lightning_indexer.py`** — TOY. Own `q_index_proj/k_index_proj` (random), NOT the trained `indexer.wk` weights. `__main__` test.
- **`hpu_sparse_attn_indexer.py`** — does a real `torch.topk`, but scoring `(q*weights)@kᵀ` is APPROXIMATE (no per-head ReLU, no causal bound), and its output is NOT consumed by `oot_mla.py`.
- **`oot_mla.py` (PR)** — explicitly *"bypasses the sparse indexer ... does full attention and does not consume sparse topk indices"* → **DENSE**.

**Net: PR #1301 lifts `platform.py:76` but runs DENSE MLA. It does NOT satisfy "sparse-as-sparse."**
The 5+3 deliberation's "track PR #1301" decision was misled by filenames/docstrings and is **OVERTURNED**.

## Corrected forever path: implement real HPU DSA ourselves (the adversary's path).
No working impl exists anywhere — this is genuine engineering, no shortcut.
1. **Indexer op** — correct HPU `SparseAttnIndexer.forward_hpu`: true DSA logits `I_{t,s}=Σ_j w_{t,j}·ReLU(q_{t,j}·k_s)`, causal, top-k=`index_topk`, fed by the REAL in-image `Indexer` (`deepseek_v2.py`: `wq_b`/`wk`/`weights_proj`/fp8/rope already correct). PR #1301's op = a shell to fix, not adopt.
2. **Sparse `forward_impl`** — NEW `HPUMLAAttention` forward that CONSUMES `topk_indices_buffer`: gather selected latent KV → masked MQA via `flat_pa_mla` / `FusedSDPA(-inf on non-selected)`. This is the part nobody has written.
3. **platform_patch** — monkeypatch `get_attn_backend_cls` to lift `platform.py:76` (use_sparse+use_mla → sparse backend).
4. **Wire** `GlmMoeDsaForCausalLM` to the sparse path.
5. **Gate** — indexer-fired counter (ctx>2048), top-k parity vs torch ref, sparse≠dense @16K.

Build against the real interfaces extracted to `$HOME/AI/dsa_src/`.

## Independent of DSA (adversary-confirmed, fix regardless):
- **`hpu-moe-ext` `apply_monolithic` OMITS DP-dispatch + forks routing → DP+EP quantized MoE is BROKEN.** Fix before any DP serving. Qwen at TP4/DP1 today is SAFE.
- **Qwen "TP4 alignment trap" is likely fiction** (vision tower is bf16/unquantized) → TP8 may be legal (~2× throughput); verify from the checkpoint, not the arch.
- **AWQ `_awq_zero` unproven** → gate GLM-5.1-AWQ on raw-byte numeric parity.

## Status
- Flagship #1 **Qwen3.5-397B-A17B: SERVING & correct** (17×23=391; MoE-as-MoE via EP, TP4/DP1).
- Flagship #2 **GLM-5.1: BLOCKED** on the real DSA implementation above (correctly RED, never falsely served dense).
