"""Force skip_topk=True on HPUMultiHeadLatentAttentionWrapper.

WHY: The upstream Indexer.forward (deepseek_v2.py:678-743) is CUDA/ROCm/XPU-only.
Specifically, after the wk_weights_proj split and k_norm, it calls
per_token_group_quant_fp8() (fp8_utils.py:596) which falls through to a
triton.next_power_of_2() call. On HPU, HAS_TRITON=False (zero active Triton
drivers — confirmed) and triton 3.7 does not expose next_power_of_2 at the module
level, so this always crashes with AttributeError during dynamo graph capture.

Additionally the 3D fake-tensor input during compile makes the kw[:, :head_dim]
slice operate on the seq dimension rather than the feature dimension, producing
shape (1, 128, 160) -> k_norm(128) mismatch. That is a symptom of the same root:
the whole Indexer.forward body is designed for CUDA's 2D [T, F] tensors and
CUDA-only fp8 ops.

The HPUMultiHeadLatentAttentionWrapper.forward (oot_mla_PR1301.py:294-337) already
skips the indexer call and runs full MLA attention — the comment at line 335 is
explicit. The missing piece is that self.skip_topk must be True so the
mla.py:168 gate:

    if self.indexer and self.is_sparse and not self.skip_topk:
        self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

never fires the broken Indexer.forward. self.is_sparse stays True (is_v32=True from
the config) so the Indexer module is still constructed and its wk_weights_proj and
wk checkpoint weights load correctly.

This patch is UNCONDITIONAL (fires at plugin registration, before HPU_DSA_READY is
checked) because the model crashes at warmup/compile regardless of the DSA gate.
"""


def register() -> None:
    """Monkeypatch HPUMultiHeadLatentAttentionWrapper.__init__ to force skip_topk=True
    and assert the FULL-ATTENTION coherence contract.

    Two coordinated effects make the HPU serve COHERENT (not just crash-free):
      1. skip_topk=True  -> the in-image Indexer.forward (mla.py:168 gate) never fires,
         so the CUDA/triton-only K-proj+fp8-quant pipeline that crashes on HPU is
         avoided (this fixes the warmup/compile crash).
      2. platform_patch.FULL_ATTENTION_FALLBACK=True -> the consume side
         (sparse_mla_wire._sparse_forward_impl) delegates to the GENUINE dense MLA
         forward instead of restricting to the now-empty (all -1) topk buffer. Without
         this, the indexer is off but the consume side still runs, and its NaN guard
         forces self-only attention -> incoherent generations.

    Both MUST hold together. This module is the single decision point for full-attention
    mode; it (re)affirms the flag so the two patches can never drift apart.

    Idempotent: re-registration is a no-op (guarded by the _hpu_skip_indexer_patched
    sentinel attribute). Raises loudly if the upstream class cannot be imported.
    """
    from vllm_gaudi.attention.oot_mla import HPUMultiHeadLatentAttentionWrapper

    # Coherence pin: indexer-off REQUIRES consume-dense. Set at the decision point so a
    # future re-enable of DSA-sparse flips exactly one flag (FULL_ATTENTION_FALLBACK).
    from . import platform_patch
    platform_patch.FULL_ATTENTION_FALLBACK = True

    if getattr(HPUMultiHeadLatentAttentionWrapper.__init__,
               "_hpu_skip_indexer_patched", False):
        return  # already applied

    _orig_init = HPUMultiHeadLatentAttentionWrapper.__init__

    def _patched_init(self, *args, skip_topk: bool = False, **kwargs):
        # Call the original __init__ (which skips MultiHeadLatentAttentionWrapper.__init__
        # and does its own field assignments + creates HPUMLAAttention).
        _orig_init(self, *args, skip_topk=skip_topk, **kwargs)
        # Override: force True unconditionally.
        # The HPU MLA forward (oot_mla_PR1301.py) already performs full attention;
        # the Indexer module is built only so its wk_weights_proj / wk checkpoint
        # weights load. It must never be called on HPU.
        self.skip_topk = True

    _patched_init._hpu_skip_indexer_patched = True
    HPUMultiHeadLatentAttentionWrapper.__init__ = _patched_init
