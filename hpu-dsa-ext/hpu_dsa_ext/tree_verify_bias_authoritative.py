"""Make an explicit prefill ``attn_bias`` the SOLE authority over FusedSDPA masking
(drop the kernel's own ``is_causal`` when a bias is present).  Gated, default-OFF.

================================  WHY  ================================================
The GLM-5.2 TREE-VERIFY Pass A (hpu_model_runner._tree_verify_decode) runs a bs=Tsum
PREFILL-shaped target forward over the T flattened tree nodes WITH committed context:

    attn_metadata = make_prefill_metadata(attn_bias=full_bias, block_list=ctx_blocks,
                                          context_lens_tensor=[ctx_len], seq_lens_tensor=[Tsum], ...)

`full_bias` ([1,1,Tsum, ctx_key_len+Tsum]) is the ONE geometry-correct mask the tree
needs: context columns = 0 (every tree node attends ALL committed context KV) and the
tree columns = the BLOCK-DIAGONAL ANCESTOR mask (node i attends only its ancestors, NOT
a linear j<i triangle -- siblings must not see each other).  It is the single source of
truth for causality + context-visibility + intra-tree ancestry.

The dense MLA prefill (vllm_gaudi.attention.backends.hpu_attn.HPUMLAImpl.forward_mha)
fetches the committed context (block_list -> latent cache) and concats it in FRONT of
the T current tokens, then calls:

    ops.prompt_attention(impl=fsdpa_impl, ..., is_causal=True, attn_bias=full_bias, ...)

and `_fsdpa_prompt_attention` (vllm_gaudi/extension/ops.py:404) ONLY drops `is_causal`
when `USING_INC` (QUANT_CONFIG set):

    if is_causal and attn_bias is not None and USING_INC:   # <-- USING_INC gate
        is_causal = False

The tree env has NO QUANT_CONFIG -> USING_INC=False -> `is_causal=True` is handed to the
FusedSDPA kernel ALONGSIDE the explicit bias.  The kernel then applies its OWN causal
mask over the NON-SQUARE [Tsum, ctx_key_len+Tsum] geometry (a query block that is NOT the
trailing-aligned full sequence) ON TOP of the ancestor bias.  That corrupts which context
columns each tree row may attend -> the T verify logits are scored against a wrong/truncated
context -> plausible-but-wrong tokens that self-verify (accept ~3.4) but read as GARBAGE.

The kernel comment itself says "causal + attn_bias is not yet supported" -- the INC gate is
simply an INCOMPLETE guard.  When a caller hands an explicit additive bias, THAT bias is
authoritative and the kernel's own is_causal must be dropped regardless of INC.

================================  THE FIX  ===========================================
Monkeypatch `vllm_gaudi.extension.ops._fsdpa_prompt_attention` so it drops `is_causal`
whenever an explicit `attn_bias` is present -- i.e. remove the `USING_INC` condition from
the existing guard.  `prompt_attention` rebuilds its impl_mapping dict from the module
global on every call (ops.py:433-437), so swapping the module global takes effect for all
subsequent prefill forwards.

SAFETY / BYTE-EXACTNESS (the never-regress bar):
  * Default OFF (HPU_FSDPA_BIAS_AUTHORITATIVE unset) -> register() is a NO-OP; the in-image
    function runs verbatim -> byte-identical to stock.
  * When ON, the drop fires ONLY when attn_bias is not None.  The NORMAL no-prefix-cache
    prefill path NEVER builds an explicit bias: hpu_model_runner.set_attn_bias early-returns
    attn_bias=None when (prefill_use_fusedsdpa AND is_causal AND block_list is None)
    (hpu_model_runner.py:4098-4101), which is exactly the standard --no-enable-prefix-caching
    serve.  So with the flag ON, a standard prefill still has attn_bias=None -> the drop does
    NOT trigger -> is_causal stays True -> BYTE-IDENTICAL.  The drop only ever engages a
    prefill that ALREADY carries an explicit bias (the tree-verify Pass A/B, and the
    prefix-cached path -- which is correct there too: the explicit bias already encodes the
    past_mask + causal triangle that the kernel's own is_causal would otherwise double-apply
    over the same [Tq, ctx+Tq] geometry).
  * valid_seq_lengths is also cleared on the drop, mirroring the existing INC branch
    (ops.py:407) -- with the bias authoritative there is no separate padding triangle to keep.

This is the smallest correct change with the narrowest blast radius: it touches ONLY
FusedSDPA prefills that pass an explicit bias, and is gated behind one env flag.
"""
from __future__ import annotations

import os

_FLAG = "HPU_FSDPA_BIAS_AUTHORITATIVE"
_TRUTHY = ("1", "true", "True", "yes", "on")

_ORIG_FSDPA = None  # the original vllm_gaudi.extension.ops._fsdpa_prompt_attention


def _enabled() -> bool:
    return os.environ.get(_FLAG, "0").strip() in _TRUTHY


def register() -> None:
    """Install the bias-authoritative FusedSDPA wrapper. NO-OP unless HPU_FSDPA_BIAS_AUTHORITATIVE=1.

    Idempotent. When the flag is unset this returns immediately and does NOT touch
    vllm_gaudi -> a stock serve is byte-for-byte unaffected (the import below never runs)."""
    if not _enabled():
        return

    global _ORIG_FSDPA
    import vllm_gaudi.extension.ops as _ops

    # Idempotency: if we already wrapped, do nothing.
    if getattr(_ops._fsdpa_prompt_attention, "_hpu_dsa_bias_authoritative", False):
        return

    _ORIG_FSDPA = _ops._fsdpa_prompt_attention

    def _fsdpa_bias_authoritative(query, key, value, scale, fsdpa_op, is_causal,
                                  attn_bias=None, valid_seq_lengths=None, window_size=None,
                                  sinks=None, **ignored_args):
        # When an explicit additive bias is present, it is the SOLE authority for masking:
        # drop the kernel's own causal triangle (which, over the non-square [Tq, ctx+Tq]
        # tree-verify geometry, would corrupt the visible context). Mirror the in-image INC
        # branch (ops.py:404-407) but WITHOUT the USING_INC condition.
        if is_causal and attn_bias is not None:
            is_causal = False
            valid_seq_lengths = None
        return _ORIG_FSDPA(
            query, key, value, scale, fsdpa_op, is_causal,
            attn_bias=attn_bias, valid_seq_lengths=valid_seq_lengths,
            window_size=window_size, sinks=sinks, **ignored_args,
        )

    _fsdpa_bias_authoritative._hpu_dsa_bias_authoritative = True
    # prompt_attention rebuilds impl_mapping from this module global each call, so swapping
    # the module attribute makes every subsequent fsdpa_impl prefill use the wrapper.
    _ops._fsdpa_prompt_attention = _fsdpa_bias_authoritative

    import logging
    logging.getLogger("vllm").warning(
        "[hpu-dsa-ext] HPU_FSDPA_BIAS_AUTHORITATIVE=1 -> FusedSDPA drops is_causal whenever an "
        "explicit attn_bias is present (the tree-verify ancestor bias is now the sole masking "
        "authority; standard no-bias prefills are unaffected -> byte-identical).")
