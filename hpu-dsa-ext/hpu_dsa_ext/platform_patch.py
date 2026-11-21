"""Lift the vllm_gaudi platform.py:76 hard-raise on sparse attention so DSA
models (GlmMoeDsa, DeepSeek-V3.2) can route to the MLA backend.

SAFETY: until DSA_READY is True we KEEP refusing sparse (with a clear message),
so a DSA model stays correctly BLOCKED rather than silently served DENSE. Flip
DSA_READY True ONLY after sparse_indexer_hpu.forward_oot AND the sparse
gather-consume in the HPU MLA impl (sparse_mla_consume wired into oot_mla's
forward_mha/forward_mqa) are both registered and parity-validated (gate D).
Lifting the raise without the consume side is exactly the dense-fallback trap
PR #1301 fell into — do not do it.
"""

from vllm_gaudi.platform import HpuPlatform

_ORIG = HpuPlatform.get_attn_backend_cls.__func__
_MLA_BACKEND = "vllm_gaudi.attention.backends.hpu_attn.HPUMLAAttentionBackend"

# Set True ONLY when the full sparse path is wired + validated.
DSA_READY = False

# When True, the HPU runs the DSA family as FULL (dense) MLA attention: the indexer
# is skipped (hpu_mla_skip_indexer forces skip_topk=True so the broken in-image
# Indexer.forward — per_token_group_quant_fp8 → triton.next_power_of_2, absent on
# HPU where HAS_TRITON=False — never fires) AND the consume side
# (sparse_mla_wire._sparse_forward_impl) delegates to the genuine dense forward_impl
# instead of restricting to the (now-empty, all -1) topk buffer.
#
# This is the oot_mla author's documented HPU design ("HPU MLA backend performs full
# attention, not sparse; we skip the indexer entirely"). It is the only COHERENT path
# on vllm-gaudi 1.24 today: forcing skip_topk alone (indexer off) WITHOUT this flag
# would leave _sparse_forward_impl consuming an all -1 buffer, whose NaN guard forces
# self-only attention per query -> garbage generations (not a crash, but incoherent).
#
# Set True by hpu_mla_skip_indexer.register() (the single full-attention decision
# point). DSA-sparse remains a future item; flipping this to False (and completing the
# Indexer.forward HPU rewrite) re-enables the real sparse gather-consume path.
#
# MVP-3 GATE (HPU_DSA_SPARSE=1): hpu_indexer_op_forward.register() flips this to False
# AT THE END of plugin init (after hpu_mla_skip_indexer ran), AND provides the missing
# HPU-safe Indexer.forward (no triton fp8) so the indexer actually fires. So with the
# flag SET the real sparse OP path runs; with the flag UNSET this stays True and the
# serve is the unchanged dense bypass. The flag is the single switch — nothing here
# changes when it is absent.
FULL_ATTENTION_FALLBACK = True


def _patched(cls, selected_backend, attn_selector_config, num_heads=None):
    if getattr(attn_selector_config, "use_sparse", False):
        if getattr(attn_selector_config, "use_mla", False):
            if not DSA_READY:
                raise NotImplementedError(
                    "[hpu-dsa-ext] DSA (sparse MLA) selected but the HPU sparse "
                    "path is not yet active (DSA_READY=False). Refusing to serve "
                    "DENSE in place of sparse (as-designed mandate). Wire "
                    "sparse_indexer_hpu + sparse attn_bias consumption first."
                )
            return _MLA_BACKEND
        # sparse-without-MLA is genuinely unsupported — fall through to original.
    return _ORIG(cls, selected_backend, attn_selector_config, num_heads)


def apply():
    HpuPlatform.get_attn_backend_cls = classmethod(_patched)
