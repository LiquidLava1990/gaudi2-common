"""binned_moe_patch.py -- additive WIRING for the WAVE-2 capacity-binned MoE op.

STRICT NO-OP unless HPU_MOE_BINNED=1. When set, it MONKEYPATCHES the module attribute
`hpu_moe_ext.fused_monolithic_moe.fused_moe_monolithic` (the dense W4A8 monolithic
entry) so the W4A8 MoE routes through the SPARSE capacity-binned grouped-GEMM
(binned_moe.fused_moe_binned) instead. It edits NO proven file: moe_wna16_hpu.py does
`from .fused_monolithic_moe import fused_moe_monolithic` INSIDE apply_monolithic (re-
imported every call, lines 1967/2047/2061), so reassigning the module attr here is
picked up at call time -- the same runtime-attribute pattern register() already uses
(_install_config_patches: moe_wna16_mod.MoeWNA16Method = ...).

DISPATCH (binned only ADDS a branch; the dense monolithic stays the fallback):
  HPU_MOE_BINNED=1  -> route W4A8 MoE through the binned op when it is ELIGIBLE
    (binned returns a tensor); on ANY ineligibility (AWQ zp / .so missing / it returns
    None) fall back to the ORIGINAL dense monolithic -- NEVER silently to a worse path.
  Requires the W4A8 monolithic to already be engaged: HPU_MOE_FUSED_MONOLITHIC=1 +
  HPU_MOE_FUSED_MONO_INT4FP8=1 (binned is a sparse-compute variant of THAT op). The
  HPU_MOE_FUSED_MONO_MAX_T decode gate still applies (binned is the decode/conc lever;
  Build-2 prefill use is a separate raise of MAX_T once the prefill working-set is
  validated). Default (unset) = byte-identical: the dense monolithic runs unchanged.

VERIFY (EAGER-ONLY): HPU_MOE_BINNED_VERIFY=1 runs BOTH binned AND the dense monolithic
every call, asserts relNorm < 1e-2 (the project's HPU_MOE_FUSED_VERIFY tol), LOUD on
mismatch, and returns the DENSE result (stay coherent). The .item() it needs breaks
lazy HPU-graph capture, so it must NOT be set in a captured lazy serve -- it is the
on-card byte-exact A/B gate, run with --enforce-eager.

ENGAGE PROOF: a one-shot-per-rank LOUD banner the first time the binned path engages
(8 ranks -> 8 banners), so a serve log unambiguously shows the op is live (not a stale
dense replay).
"""
from __future__ import annotations

import os


def install():
    if os.environ.get("HPU_MOE_BINNED") != "1":
        return  # strict no-op: dense monolithic unchanged

    import logging

    try:
        from . import fused_monolithic_moe as _mono_mod
        from . import binned_moe
    except Exception as _ex:  # noqa: BLE001
        logging.getLogger("vllm").warning(
            "[HPU_MOE_BINNED] install skipped (import failed, non-fatal): %r", _ex)
        return

    _orig_mono = _mono_mod.fused_moe_monolithic
    _verify = os.environ.get("HPU_MOE_BINNED_VERIFY") == "1"
    # FORENSIC garble proof (default OFF; pure no-op unless set). The stock HPU_MOE_POISON
    # sits on the ROUTED return, not the monolithic the binned op replaces, so it cannot
    # prove the BINNED op executed. HPU_MOE_BINNED_POISON=1 scales the binned output x1.15:
    # coherent greedy => binned did NOT run (stale dense replay); garbled => it ran on device.
    _poison = os.environ.get("HPU_MOE_BINNED_POISON") == "1"
    _engaged = [False]

    def _binned_monolithic(x, w13_packed, w13_scale, w13_zp,
                           w2_packed, w2_scale, w2_zp,
                           topk_ids, topk_weights, experts_min, chunk=None,
                           x_buf=None):
        """Drop-in for fused_moe_monolithic: route the W4A8 MoE through the SPARSE
        binned grouped-GEMM. On ineligibility, fall back to the dense monolithic."""
        E = int(w13_packed.shape[0])               # local experts = the stacked dim
        binned = binned_moe.fused_moe_binned(
            x, topk_ids, topk_weights,
            w13_packed, w13_scale, w13_zp,
            w2_packed, w2_scale, w2_zp,
            experts_min, E, C=None, chunk=chunk)
        if binned is None:
            # AWQ zp / .so missing -> dense monolithic (correct, never a worse path).
            return _orig_mono(x, w13_packed, w13_scale, w13_zp,
                              w2_packed, w2_scale, w2_zp,
                              topk_ids, topk_weights, experts_min, chunk, x_buf)
        if not _engaged[0]:
            _engaged[0] = True
            logging.getLogger("vllm").warning(
                "[HPU_MOE_BINNED] ENGAGED (rank one-shot): W4A8 capacity-binned "
                "grouped-GEMM live (E=%d, T=%d, cap=%s). Sparse E*C compute vs dense "
                "E*T; reads unchanged. relNorm gate = HPU_MOE_BINNED_VERIFY=1 "
                "(eager-only).", E, int(x.shape[0]),
                os.environ.get("HPU_MOE_BINNED_CAP", "32"))
        if _verify:
            ref = _orig_mono(x, w13_packed, w13_scale, w13_zp,
                             w2_packed, w2_scale, w2_zp,
                             topk_ids, topk_weights, experts_min, chunk, x_buf)
            if ref is not None:
                diff = (binned.float() - ref.float()).norm().item()
                rn = diff / (ref.float().norm().item() + 1e-9)
                if rn >= 1e-2:
                    ma = (binned.float() - ref.float()).abs().max().item()
                    logging.getLogger("vllm").warning(
                        "[HPU_MOE_BINNED][verify] MISMATCH relNorm=%.4g maxAbs=%.4g "
                        "T=%d cap=%s -- raise HPU_MOE_BINNED_CAP (capacity overflow) "
                        "or bisect; returning DENSE result (stay coherent).",
                        rn, ma, int(x.shape[0]), os.environ.get("HPU_MOE_BINNED_CAP", "32"))
                    return ref
                logging.getLogger("vllm").info(
                    "[HPU_MOE_BINNED][verify] relNorm=%.6g T=%d" % (rn, int(x.shape[0])))
        if _poison:
            binned = binned * 1.15   # FORENSIC: coherent greedy => binned op did NOT execute
        return binned

    _mono_mod.fused_moe_monolithic = _binned_monolithic
    logging.getLogger("vllm").warning(
        "[HPU_MOE_BINNED] WIRED: W4A8 monolithic -> capacity-binned grouped-GEMM "
        "(cap=%s, chunk=%s, verify=%s). Requires HPU_MOE_FUSED_MONOLITHIC=1 + "
        "HPU_MOE_FUSED_MONO_INT4FP8=1. Default-off elsewhere is byte-identical.",
        os.environ.get("HPU_MOE_BINNED_CAP", "32"),
        os.environ.get("HPU_MOE_BINNED_CHUNK", "8"), _verify)
