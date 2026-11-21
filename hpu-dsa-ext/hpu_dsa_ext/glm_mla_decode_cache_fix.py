"""GLM-5.1 MLA decode KV-cache write->read fix for lazy HPU-graph capture.

SYMPTOM (the residual after rope + routed-MoE were excluded):
  GLM lazy decode emits CORRECT content for ~8-10 tokens then accumulates error
  into a greedy repetition cycle. EAGER is fully coherent. Positions are LIVE, the
  rope cos/sin freeze is fixed (glm_rope_neox (B)), routed-MoE is excluded (the bug
  persists at HPU_MOE_ROUTED_K unset == K=1 stock monolithic), decode buckets are on
  BLOCKS (first boundary at token 128, not 8-10), DSA is dense-bypassed (zero ops).
  => a SUBTLE PER-STEP NUMERICAL DRIFT inside the captured MLA decode that compounds
  through the KV cache.

THE SITE (HPUMLAAttention.forward_impl, decode branch, vllm_gaudi/attention/oot_mla.py
:119-145), in order:

    latent_vec_k = torch.concat((k_c_normed, k_pe.view(...)), dim=-1)   # L131 fresh
    latent_vec_k = latent_vec_k.view(-1, qk_rope_head_dim + kv_lora_rank)
    self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)   # L137 WRITE
    output = self.impl.forward_mqa(decode_ql_nope, q_pe, kv_cache, ...) # L143 READ

  latent_cache_k is VLLMKVCache (vllm_gaudi/extension/utils.py:71-76):
      cache.index_copy_(0, slot_mapping, input); return cache
  The READ (forward_mqa -> flat_pa_mla, ops.py:165) under use_contiguous_pa=True
  (default here: --no-enable-prefix-caching) is
      key = key_cache.unflatten(0,(-1,block_size))[:block_list.size(0)]   # PREFIX SLICE
  i.e. the read does NOT re-index by slot_mapping; it slices the cache prefix. So the
  CURRENT token attends to itself ONLY if its index_copy_ landed at slot_mapping's
  (per-step-advancing) slot.

WHY EAGER-CORRECT / LAZY-WRONG (capture mechanism, established by reading the bridge,
not a HW probe -- a .item() probe would itself break capture and mask it):

  The model body (HpuModelAdapter.forward) is wrapped
      wrap_in_hpu_graph(disable_tensor_cache=True)            (hpu_model_runner.py:837)
  On the cache-HIT replay path (habana .../hpu/graphs.py wrapped_hpugraph_forward,
  lines 752-763) the bridge splits graph inputs into:
      matched = get_user_input_match_indices()  -> spliced LIVE by address into replayV3
      the rest -> copy_to(capture_time_storage, ...)   then replayV3 runs.
  A per-step INDEX tensor that drives a scatter is only correct on replay if its
  liveness survives in `matched`. This is EXACTLY the failure that froze rope cos/sin
  (glm_rope_neox.py (B) root-cause note, lines 30-44): a gather/scatter whose index
  tensor's liveness the matcher drops gets PINNED to the step-0 captured value.

  Here the index is `slot_mapping` (oot_mla.py:129 = attn_metadata.slot_mapping
  .flatten()), driving the cache index_copy_ WRITE (utils.py:75). It is structurally
  the same shape as the rope bug:
    * an index-tensor-driven write into a persistent buffer,
    * whose consumer (the prefix-slice READ) reads the buffer INDEPENDENTLY (there is
      NO data-flow edge: oot_mla.py:137 discards index_copy_'s return; forward_mqa
      reads the original kv_cache[0] reference), and
    * an indirection (the .flatten() + the VLLMKVCache module call) between the live
      metadata input and the op -- the precise shape "most exposed to the matcher
      dropping the index's liveness" (rope note, line 41).
  If slot_mapping's per-step value is dropped on replay, every decode step writes the
  new token's latent KV to the STEP-0 slot. Prior tokens' KV (written by their OWN
  past graph executions, at a stable cache address) remain, so the model still attends
  to the growing context with correct positions (layer norms DRIFT -- matches the
  observation) -- but the CURRENT token's self-KV is mis-placed / its self-attention
  is wrong. One slightly-wrong row out of N looks fine for ~8-10 tokens, then the
  error feeds each next hidden state + each next written KV and cascades into the
  repetition attractor. This matches "correct-then-degrades" far better than a hard
  freeze (which would corrupt from token 1).

  NB this is NOT a precision issue (bf16 QK/softmax is identical eager-vs-lazy;
  fp32_softmax is qwen2-only) and NOT cross-block bf16 reduction (degeneration starts
  in block 0 where block_mapping is 1x1 and block2batch is a no-op). It is capture-
  replay-specific by construction, which is why it is eager-correct.

THE FIX (gated HPU_GLM_MLA_DECODE_CACHE_FIX, default ON under lazy when the rope fix
is on; eager never installs it -> eager numerics untouched):

  Rewrite ONLY the decode branch of HPUMLAAttention.forward_impl so the cache
  write->read is a SINGLE LIVE DATA-DEPENDENCY rooted in the per-step slot_mapping,
  mirroring the proven glm_rope_neox (B) inline fix:

    (1) Re-read slot_mapping immediately before the write so its gather/scatter index
        is consumed in the same expression that depends on it (no stale binding).
    (2) Capture the returned cache handle from latent_cache_k and thread THAT handle
        (not the original kv_cache[0] reference) into forward_mqa, creating an explicit
        write->read data-flow edge the matcher cannot drop.
    (3) .clone() latent_vec_k right before the index_copy_ so the value written is a
        fresh, non-aliased tensor (defeats any disable_tensor_cache storage reuse of
        the concat intermediate across replays).

  Numerics are bit-identical to stock: same concat/view, same VLLMKVCache.index_copy_,
  same forward_mqa/flat_pa_mla, same _v_up_proj. Only the WRITE input is cloned and the
  READ is rooted in the write's output handle -- WHERE/HOW the buffers are threaded
  changes, never the values. (Prefill branch is left byte-for-byte unchanged.)

CONFIRMATION (survives capture; does NOT need this fix's correctness assumed):
  * Global A/B already exists: glm_no_tensor_cache.py (HPU_GLM_NO_TENSOR_CACHE=1)
    forces disable_tensor_cache=False. If that A/B makes GLM coherent, the cause IS a
    capture buffer-liveness/aliasing issue (this family). This fix is the surgical,
    HBM-cheap replacement for that global hammer.
  * To pin it to slot_mapping specifically, use the capture_freeze_probe with the
    cache-content tap (see SLOT_READBACK note below): checksum the cache ROW at the
    CURRENT slot AFTER the write (cache.index_select(0, slot_mapping)) -- if that
    checksum is CONSTANT while pos_sum advances, the write slot froze (this bug); if
    it advances but attn drifts, the read/aggregation is at fault instead.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")


def _install_decode_cache_fix() -> None:
    try:
        from vllm_gaudi.attention.oot_mla import HPUMLAAttention
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-mla-cache] cannot import HPUMLAAttention; decode cache fix "
                    "NOT installed: %s", ex)
        return

    if getattr(HPUMLAAttention.forward_impl, "_glm_mla_decode_cache_fix", False):
        return

    import torch

    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        # Bit-identical to vllm_gaudi/attention/oot_mla.py:106-145, EXCEPT the decode
        # branch threads the cache write->read as one live dependency rooted in the
        # per-step slot_mapping and clones the written value (see module docstring).
        is_prefill = attn_metadata.is_prompt

        if is_prefill:
            # ---- PREFILL: unchanged from stock (no capture-replay decode loop) ----
            slot_mapping = (attn_metadata.slot_mapping.flatten()
                            if attn_metadata.slot_mapping is not None else None)
            latent_vec_k = torch.concat(
                (k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
            latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
            if kv_cache is not None and len(kv_cache) >= 2:
                self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)
            return self.impl.forward_mha(q, latent_vec_k, kv_cache, attn_metadata)

        # ---- DECODE: the fixed path ----
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)
        decode_ql_nope = torch.bmm(q_nope, self.W_UK_T)
        decode_ql_nope = decode_ql_nope.transpose(0, 1)

        latent_vec_k = torch.concat(
            (k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
        latent_vec_k = latent_vec_k.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)

        if kv_cache is not None and len(kv_cache) >= 2:
            # (1) re-read the index in the same place it is consumed -> the scatter index
            #     is a fresh, live dependency on this step's slot_mapping (not a binding
            #     captured earlier in the function).
            slot_mapping = (attn_metadata.slot_mapping.flatten()
                            if attn_metadata.slot_mapping is not None else None)
            # (3) write a fresh, non-aliased value (defeats concat-intermediate storage
            #     reuse under disable_tensor_cache=True).
            latent_vec_k = latent_vec_k.clone()
            # (2) thread the RETURNED cache handle into the read -> explicit write->read
            #     data-flow edge (stock discards this return and reads kv_cache[0]).
            written = self.impl.latent_cache_k(latent_vec_k, kv_cache[0], slot_mapping)
            if isinstance(kv_cache, tuple):
                kv_for_read = (written, ) + tuple(kv_cache[1:])
            elif isinstance(kv_cache, list):
                kv_for_read = [written] + list(kv_cache[1:])
            else:
                kv_for_read = kv_cache
        else:
            kv_for_read = kv_cache

        output = self.impl.forward_mqa(decode_ql_nope, q_pe, kv_for_read, attn_metadata)
        output = self._v_up_proj(output)
        return output

    forward_impl._glm_mla_decode_cache_fix = True
    HPUMLAAttention.forward_impl = forward_impl
    log.warning("[glm-mla-cache] installed capture-safe MLA decode forward_impl "
                "(live slot_mapping write->read dependency + cloned cache write; fixes "
                "lazy-capture per-step KV drift / repetition). HPU_GLM_MLA_DECODE_CACHE_FIX")


def install() -> None:
    # Gate: default ON under lazy (capture exists) when the rope coherence fix is on;
    # OFF under eager (no capture -> stock path, proven-coherent eager numerics).
    # HPU_GLM_MLA_DECODE_CACHE_FIX overrides (1/0) for A/B.
    env = os.environ.get("HPU_GLM_MLA_DECODE_CACHE_FIX")
    try:
        import habana_frameworks.torch as htorch
        is_lazy = bool(htorch.utils.internal.is_lazy())
    except Exception:  # noqa: BLE001
        is_lazy = False
    rope_on = os.environ.get("HPU_GLM_ROPE_NEOX", "0") == "1"
    if env is not None:
        do_install = env.strip().lower() in ("1", "true", "yes", "on")
    else:
        do_install = is_lazy and rope_on  # default: on under lazy GLM serve
    if do_install:
        _install_decode_cache_fix()
    else:
        log.warning("[glm-mla-cache] decode cache fix NOT installed "
                    "(lazy=%s, HPU_GLM_ROPE_NEOX=%s, HPU_GLM_MLA_DECODE_CACHE_FIX=%s)",
                    is_lazy, rope_on, env)
