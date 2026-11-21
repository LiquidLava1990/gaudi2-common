"""Diagnostic: localize the near-zero dense-MLA prefill attention output.

Finding (2026-06-06): GLM-5.1 dense MLA prefill produces attention output with
norm ~0.1 (out_absmax ~0.005) while the query norm is ~100 — i.e. attention is
effectively DEAD (output ~1000x too small), so the model ignores context and emits
repetitive salad. This reproduces in --enforce-eager, so it is a NUMERICAL bug in
HPUMLAImpl.forward_mha (hpu_attn.py:245-307), NOT a torch.compile boundary.

This patch wraps forward_mha and logs the norms of the inputs/intermediates so we can
see WHERE the magnitude collapses: scale, attn_bias presence, q, latent_k, k_c_normed,
the kv_b_proj output, the V split, and the returned output. Read-only; gated by
HPU_MHA_DIAG=N (log first N prefill calls). No effect when unset.
"""
from __future__ import annotations

import os
import logging

log = logging.getLogger("vllm")
_N = int(os.environ.get("HPU_MHA_DIAG", "0") or "0")
_count = [0]


def install() -> None:
    if _N <= 0:
        return
    import torch
    try:
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl
    except Exception as ex:  # noqa: BLE001
        log.warning("[mha-diag] cannot import HPUMLAImpl: %s", ex)
        return
    if getattr(HPUMLAImpl.forward_mha, "_mha_diag", False):
        return
    _orig = HPUMLAImpl.forward_mha

    @torch.compiler.disable
    def _n(t):
        try:
            tf = t.detach().float()
            return (f"norm={float(tf.norm()):.4f} absmax={float(tf.abs().max()):.5f} "
                    f"rms={float(tf.pow(2).mean().sqrt()):.5f} nan={int(torch.isnan(tf).any())}")
        except Exception:  # noqa: BLE001
            return "?"

    @torch.compiler.disable
    def _log(self, q, latent_vec_k, out):
        try:
            kc, kpe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kvn = self.kv_b_proj(kc)[0].view(-1, self.num_heads,
                                             self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kvn.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            log.warning(
                "[mha-diag #%d] scale=%.6f qk_head=%d v_head=%d attn_bias=%s\n"
                "    q        [%s]\n    latent_k [%s]\n    k_c_normed[%s]\n"
                "    k_nope   [%s]\n    V        [%s]\n    OUT      [%s]",
                _count[0], float(self.scale), self.qk_head_dim, self.v_head_dim,
                "None" if getattr(self, "_last_attn_bias_none", "?") == "?" else "?",
                _n(q), _n(latent_vec_k), _n(kc), _n(k_nope), _n(v), _n(out))
        except Exception as ex:  # noqa: BLE001
            log.warning("[mha-diag] log err: %s", ex)

    _dumped = [False]

    def forward_mha(self, q, latent_vec_k, k_cache, attn_metadata):  # noqa: ANN001
        out = _orig(self, q, latent_vec_k, k_cache, attn_metadata)
        # Dump pre-o_proj attention internals for CPU bisection (gated HPU_MHA_DUMP=1, once).
        if os.environ.get("HPU_MHA_DUMP", "0") == "1" and not _dumped[0]:
            _dumped[0] = True
            try:
                d = {
                    "q": q.detach().to("cpu", torch.float32),
                    "latent_vec_k": latent_vec_k.detach().to("cpu", torch.float32),
                    "fwd_mha_out": out.detach().to("cpu", torch.float32),   # PRE-o_proj value mixture (per-rank)
                    "kv_b_weight": self.kv_b_proj.weight.detach().to("cpu", torch.float32),
                    "scale": float(self.scale),
                    "num_heads": int(self.num_heads), "qk_nope": int(self.qk_nope_head_dim),
                    "qk_rope": int(self.qk_rope_head_dim), "v_head": int(self.v_head_dim),
                    "kv_lora": int(self.kv_lora_rank), "qk_head": int(self.qk_head_dim),
                }
                torch.save(d, "/opt/hpu-dsa-ext/mha_dump.pt")
                log.warning("[mha-diag] DUMPED forward_mha internals to /opt/hpu-dsa-ext/mha_dump.pt "
                            "(q %s, latent_k %s, fwd_mha_out %s, num_heads=%d)",
                            list(q.shape), list(latent_vec_k.shape), list(out.shape), self.num_heads)
            except Exception as ex:  # noqa: BLE001
                log.warning("[mha-diag] dump error: %s", ex)
        if _count[0] < _N:
            _count[0] += 1
            try:
                ab = getattr(attn_metadata, "attn_bias", None)
                log.warning("[mha-diag #%d] attn_bias=%s is_prompt=%s seq_lens=%s",
                            _count[0], "None" if ab is None else "set",
                            getattr(attn_metadata, "is_prompt", "?"),
                            str(getattr(attn_metadata, "seq_lens_tensor", "?"))[:60])
                _log(self, q, latent_vec_k, out)
            except Exception as ex:  # noqa: BLE001
                log.warning("[mha-diag] err: %s", ex)
        return out

    forward_mha._mha_diag = True
    HPUMLAImpl.forward_mha = forward_mha
    log.warning("[mha-diag] installed on HPUMLAImpl.forward_mha (N=%d)", _N)
