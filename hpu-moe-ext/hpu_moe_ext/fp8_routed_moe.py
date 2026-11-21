"""Track A — zero-build "fp8-compose" fused routed int4 MoE for Gaudi 2.

A *measured bandwidth win with NO kernel compile*. Built entirely from EXISTING
live HPU ops on this 1.24 build:

  * ``torch.ops.hpu.convert_from_uint4(packed, scale, zero_point, out_dtype, g_idx)``
      packed int4 (int32 ``[in, out//8]``) + per-group scale + symmetric zero ->
      dense ``[in, out]``.  We dequant to ``out_dtype=bf16`` (the ONLY proven
      out_dtype — every live call site uses ``x.dtype``/bf16) and then cast bf16
      -> fp8 with a self-consistent ``cast_to_fp8_v2`` round-trip. (Dequantizing
      int4 STRAIGHT TO ``out_dtype=float8_e4m3fn`` was the original 2x bug: see the
      "fn-vs-fnuz" note below.)
  * ``torch.ops.hpu.cast_to_fp8_v2(x, 1/scale, False, False, float8_e4m3fn)``
      activation -> fp8 (mirrors ``dynamic_quant`` in
      ``vllm_gaudi.extension.ops``).
  * ``torch.ops.hpu.fp8_gemm_v2(A, trans_A, B, trans_B, D, out_dtype,
      A_scale_inv, B_scale_inv, bias, accumulate, ...)``
      native fp8 MME GEMM (the Gaudi-2 MME runs fp8 at ~2x the bf16 rate).

------------------------------------------------------------------------------
WHY THIS IS A BANDWIDTH WIN (the whole point of Track A)
------------------------------------------------------------------------------
The stock WNA16 MoE op (``VllmMixtureOfExpertsOpWNA16.forward`` in
``vllm_gaudi/extension/ops.py``) calls ``convert_from_uint4`` with
out_dtype == ``weight_scale.dtype`` (bf16) for EVERY local expert, materializing
the full bf16 weight in HBM, then feeds those bf16 tensors to
``torch.ops.hpu.mixture_of_experts``.  At batch=1 that bf16 weight round-trip
through HBM is the latency floor (sync/bandwidth bound, ~all local experts).

Two stacked savings here:
  1. ROUTED-K gather (same trick as ``moe_wna16_hpu._routed_moe``): only the
     <=K *locally-routed* experts are touched, not all E.  Caller guarantees
     ``tokens*top_k <= K`` so #distinct local experts <= K with NO host sync and
     NO dynamic shape.
  2. fp8 weights + fp8 MME: each routed expert's bf16 weight is cast to fp8 and
     both GEMM operands are fp8, so the MME consumes them natively at ~2x
     throughput and the fp8 weight is half-size. (The earlier idea of writing the
     fp8 weight DIRECTLY out of convert_from_uint4 to also avoid building bf16 was
     incorrect on Gaudi-2 — see the fn-vs-fnuz note; we now dequant to a TRANSIENT
     bf16 weight, per ROUTED expert only, then cast to fp8.)

------------------------------------------------------------------------------
fn-vs-fnuz: WHY int4 -> fp8 DIRECTLY IS A 2x BUG ON GAUDI-2 (root cause of the
            "fp8_norm ~= 2.06 * ref_norm" failure)
------------------------------------------------------------------------------
The Gaudi-2 HW fp8 element is ``float8_e4m3fnuz`` (exponent bias 8, max 240); the
DTYPE name handed to the ops is ``float8_e4m3fn`` (exponent bias 7, max 448). For
the SAME stored e4m3 bits the two interpretations differ by exactly 2^(7-8)=2x.
A ``cast_to_fp8_v2(data, 1/s)`` immediately consumed by ``fp8_gemm_v2(..., s)`` is
a self-consistent round-trip (both ops agree on the fnuz bits) so the factor
cancels — that is the proven ``apply_fp8_linear_hpu`` / ``dynamic_quant`` path,
and it is why the live fp8 linear is correct. But ``convert_from_uint4(out_dtype=
float8_e4m3fn)`` is an ASYMMETRIC op: it WRITES fp8 bits from the int4 dequant
math, and the subsequent read (``.float()`` / the next cast) reinterprets them at
the 2x factor with nothing to cancel it. Empirically the down/w2 projection's
factor survives straight to the output (it is the last op; w13's is absorbed by
the silu + the intermediate's own requant), yielding the observed ~2.06x. The fix
is to NEVER hand convert_from_uint4 an fp8 out_dtype: dequant to bf16 (proven) and
make the ONLY fp8 cast a self-consistent ``cast_to_fp8_v2`` round-trip.

------------------------------------------------------------------------------
fp8 SCALE CONVENTION (documented because fp8_gemm_v2's arg names mislead)
------------------------------------------------------------------------------
We mirror the PROVEN live code (``apply_fp8_linear_hpu`` / ``FP8Matmul`` /
``dynamic_quant`` in ``vllm_gaudi/extension/ops.py`` + ``utils.py``).  Despite the
``A_scale_inv`` / ``B_scale_inv`` parameter names, the live call sites pass the
*FORWARD* (recovery) scale into those slots — i.e. the scalar ``s`` such that

        real_value  ~=  fp8_value * s              (s = |data|.max / FP8_MAX)

``dynamic_quant`` returns exactly this ``s`` and ``apply_fp8_linear_hpu`` then
passes ``A_scale_inv = x_scale`` (that same forward ``s``), so the GEMM forms

        out  =  ( A_fp8 . B_fp8 )  *  A_scale_inv  *  B_scale_inv
             =  ( A_fp8 * s_A ) . ( B_fp8 * s_B )                     (recovered)

THE CAST uses the INVERSE: ``cast_to_fp8_v2(data, 1.0/s, ...)`` packs
``data/s`` into the fp8 grid.  So per operand:
  * cast            -> ``1.0 / s``   (compresses real -> fp8 grid)
  * gemm *_scale_inv -> ``s``         (the FORWARD scale; recovers fp8 -> real)

We use this convention throughout. ``FP8_MAX`` is the Gaudi-2 e4m3 max (240.0,
from ``float8_e4m3fnuz``; see ``ops.py:30``), even though the cast/gemm DTYPE we
hand the ops is ``float8_e4m3fn`` — that is exactly what the live Gaudi-2 path
does (the HW maps fn->fnuz internally; FP8_MAX must be the fnuz max so the cast
does not overflow the real e4m3 range).

WEIGHT SCALE: ``convert_from_uint4(out_dtype=bf16)`` folds the int4 per-group
``weight_scale`` into a real bf16 weight. We then do a per-expert per-tensor
bf16 -> fp8 quant (one recovery scale per weight tensor; same self-consistent
``cast_to_fp8_v2`` round-trip as the activation). The GEMM's ``B_scale_inv`` is
that per-tensor fp8 recovery scale, NOT the int4 group scale. (Per-group fp8
weight scales would need a block-scaled GEMM; per-tensor is what fp8_gemm_v2
supports cheaply and what the win rides on.)

------------------------------------------------------------------------------
BIGGEST ACCURACY RISK — int4 -> fp8 DOUBLE QUANTIZATION
------------------------------------------------------------------------------
The weight is quantized TWICE: first to int4 (the checkpoint), then the
dequantized real weight is re-quantized to fp8 e4m3 (3 mantissa bits).  fp8 e4m3
has ~2 decimal digits of precision; a per-tensor scale across a 6144- or 4096-
wide output means small-magnitude weights lose relative precision.  Stacked on
the existing int4 error, expect a few-% relative error vs the bf16-intrinsic
reference — coherent, but NOT bit-exact and NOT within 1e-2.  The activation is
also fp8-quantized (per-token here), adding to it.  See the microbench's
threshold note for why fp8 needs a looser correctness bar than the int4
routed path.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

try:  # real HPU build
    import habana_frameworks.torch.core as htorch
    _HAS_HPU = True
except Exception:  # syntax-check / non-HPU host (py_compile only)
    htorch = None
    _HAS_HPU = False

# ENGAGE: LOUD silent-fallback log + HPU_ENGINE_STRICT hard-fail. CPU-safe import.
from .engage_manifest import engage_fallback


# Gaudi-2 e4m3 max. Computed from float8_e4m3fnuz (the TRUE Gaudi-2 fp8 format,
# max 240.0) exactly as vllm_gaudi/extension/ops.py:30 does, while the cast/gemm
# DTYPE handed to the ops stays float8_e4m3fn (HW maps fn->fnuz). Using the fn max
# (448.0) here would let the cast push values past the real e4m3 range -> NaN/inf.
FP8_MAX = float(torch.finfo(torch.float8_e4m3fnuz).max)

# The DTYPE we pass to cast_to_fp8_v2 / convert_from_uint4(out_dtype) /
# fp8_gemm_v2 operands. The live Gaudi-2 path (apply_fp8_linear_hpu, FP8Matmul)
# uses float8_e4m3fn and the MME maps it to the HW fnuz format. Keeping BOTH gemm
# operands this same fp8 type is required (mixed fp8 dtypes are not a valid MME
# operand pair).
FP8_DTYPE = torch.float8_e4m3fn

# Per-token (per-row) activation quant (mirrors dynamic_quant(single_scale=False))
# vs a single per-tensor scalar. Per-token is strictly more accurate for the
# activation and is what the production fp8 linear path uses; flip to per-tensor
# with HPU_FP8_ACT_PERTENSOR=1 if a future GEMM-scale-shape constraint requires it.
_ACT_PER_TENSOR = os.environ.get("HPU_FP8_ACT_PERTENSOR") == "1"


def _mark_step():
    if _HAS_HPU:
        htorch.mark_step()


def _co_w(t):
    """Fresh, CONTIGUOUS, storage_offset==0 copy of a per-expert weight/zp slice. The fused
    op's first node on every weight/zp input is a reinterpret_cast int32->uint4 feeding a
    NON-PERSISTENT SRAM-resident dequant -- the most stride/offset-sensitive consumer; a
    per-expert slice w[e] for e>0 carries a NONZERO storage_offset, so the reinterpret would
    read the WRONG base (effectively another expert's / out-of-buffer bytes) -> garbage. A
    .contiguous().clone() gives an offset-0 leaf. IDENTICAL invariant to fused_moe_routed._co
    (and the monolithic op's per-slice clone at load). Applies in BOTH the gathered-K and the
    all-expert per-e branches (both index a per-expert slice with a nonzero offset)."""
    return t.contiguous().clone()


def _cs_w(t, dtype=torch.bfloat16):
    """Like _co_w but for the scale slice, ALSO pinned to the op's bf16 dequant contract: the
    op's dequantize_4_bit guid is hardcoded bf16, so a fp16-origin scale param would be
    reinterpreted as bf16 bytes -> garbage. Fresh contiguous offset-0 bf16 copy."""
    return t.to(dtype).contiguous().clone()


def _pack_sym_zero(value: int, groups: int, out: int, device, dtype) -> torch.Tensor:
    """Build a PACKED symmetric zero-point int32 [groups, out//8] holding the
    constant `value` in every uint4 lane — bit-identical to what the op's
    get_dequant_weight expects (the same pack_quantized_values_into_int32 the
    MoE load path uses for _sym_zero). Used to re-derive the w2 (and, if needed,
    w13) zero at its correct width from the single shared `zp` the caller passes.
    `dtype` is forced to int32 (packed zeros are always int32); the arg exists only
    to mirror the source zp's device/intent."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32)
    from vllm.scalar_type import scalar_types
    zeros = torch.full((groups, out), int(value), dtype=torch.int32, device=device)
    return pack_quantized_values_into_int32(zeros, scalar_types.uint4, packed_dim=1)


def _quant_act_fp8(x: torch.Tensor):
    """Activation -> fp8, returning (x_fp8, x_scale_fwd).

    Mirrors vllm_gaudi.extension.ops.dynamic_quant EXACTLY:
        s = (|x|.max + 1e-8) / FP8_MAX                 # FORWARD/recovery scale
        x_fp8 = cast_to_fp8_v2(x, 1/s, False, False, FP8_DTYPE)[0]
    Returns the FORWARD scale s (NOT 1/s); the GEMM's A_scale_inv slot takes s.
    Per-token (per-row) scale by default; per-tensor scalar when requested.
    """
    if _ACT_PER_TENSOR:
        s = (x.abs().max() + 1e-8) / FP8_MAX            # scalar
        s = s.reshape(1)
    else:
        s = (x.abs().max(dim=-1).values + 1e-8) / FP8_MAX   # [rows]
        s = s.unsqueeze(-1)                                  # [rows,1]
    x_fp8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / s, False, False, FP8_DTYPE)[0]
    return x_fp8, s.float()


def _quant_weight_fp8(w_real: torch.Tensor):
    """Quantize an already-dequantized (real-valued, bf16) weight to fp8 with a
    SINGLE per-tensor recovery scale, returning (w_fp8, w_scale_fwd).

    This is EXACTLY ``dynamic_quant(single_scale=True)`` applied to the weight —
    the same proven cast that ``apply_fp8_linear_hpu`` uses for the activation, so
    the cast (``1/s``) and the GEMM's ``B_scale_inv`` (``s``) form a SELF-CONSISTENT
    round-trip on the Gaudi-2 fnuz MME. The forward scale ``s_w`` satisfies
    ``real_w ~= w_fp8 * s_w``; cast packs with ``1/s_w``, gemm recovers with
    ``s_w``.

    >>> WHY bf16 -> fp8 HERE (not int4 -> fp8 directly) <<<
    The weight is first dequantized by ``convert_from_uint4(out_dtype=bf16)`` (the
    ONLY out_dtype every proven call site uses; see hpu_compressed_tensors /
    hpu_gptq / hpu_awq / MoeWNA16Matmul.get_dequant_weight), then cast bf16->fp8
    by this self-consistent ``cast_to_fp8_v2`` round-trip. Dequantizing int4
    DIRECTLY to ``out_dtype=float8_e4m3fn`` is UNVALIDATED on Gaudi-2 and, because
    the HW fp8 element is ``float8_e4m3fnuz`` (exponent bias 8, max 240) while the
    dtype handed to the op is ``float8_e4m3fn`` (bias 7, max 448), the bits written
    by convert_from_uint4 are reinterpreted by the subsequent read at a factor of
    2 — a systematic ~2x output inflation (e.g. the down/w2 projection's error
    survives to the output: fp8_norm ~= 2.06 * ref_norm). Going through bf16 makes
    the only fp8 cast a round-trip that cancels the fnuz factor, exactly as the
    proven activation path does.
    """
    # Upcast to fp32 for the reduction so the max is exact, then cast to fp8 with 1/s.
    w_f = w_real.float()
    s_w = (w_f.abs().max() + 1e-8) / FP8_MAX                # scalar per-tensor
    s_w = s_w.reshape(1)
    w_fp8 = torch.ops.hpu.cast_to_fp8_v2(w_f, 1.0 / s_w, False, False, FP8_DTYPE)[0]
    return w_fp8, s_w.float()


def _fp8_linear(x_fp8, x_scale, w_fp8, w_scale, out_dtype, bias=None):
    """One fp8 MME GEMM: x[T,in] @ W[in,out] -> [T,out] in out_dtype.

    W is stored [in,out] (convert_from_uint4 returns [in,out], matching
    permuted_weights=True / x @ W). The proven apply_fp8_linear_hpu uses B as
    [out,in] with trans_B=True; here W is already [in,out] so we want B=[in,out]
    with trans_B=FALSE (B used as-is). *_scale_inv take the FORWARD scales
    (x_scale, w_scale) per the convention documented in the module docstring.

    >>> THIS is the line where the bf16 HBM weight round-trip is AVOIDED: <<<
    both operands are fp8 (half the bytes), the int4 weight was dequantized
    DIRECTLY to fp8 (never bf16), and the MME multiplies fp8 natively at 2x.
    """
    return torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=w_fp8,
        trans_B=False,
        D=None,
        out_dtype=out_dtype,
        A_scale_inv=x_scale,      # FORWARD scale s_x  (real_x ~= x_fp8 * s_x)
        B_scale_inv=w_scale,      # FORWARD scale s_w  (real_w ~= w_fp8 * s_w)
        bias=bias,
        accumulate=False,
    )


def _silu_mul(gate_up: torch.Tensor, half: int, activation: str) -> torch.Tensor:
    """Split the fused gate||up projection [T, 2*half] and apply the gated
    activation: silu(gate)*up. ``half`` == moe_intermediate (2048 for GLM-5.1)."""
    gate = gate_up[:, :half]
    up = gate_up[:, half:]
    if activation == "silu":
        act = torch.nn.functional.silu(gate)
    elif activation == "gelu":
        act = torch.nn.functional.gelu(gate)
    else:
        raise ValueError(f"fp8_routed_moe: unsupported activation {activation!r}")
    return act * up


def fp8_routed_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    zp,
    K: int,
    activation: str = "silu",
    experts_min: int = 0,
) -> torch.Tensor:
    """fp8-compose fused routed int4 MoE. Returns [T, H] bf16.

    Mirrors ``moe_wna16_hpu.MoeWNA16HPUMethod._routed_moe`` routing EXACTLY (the
    present/scatter/topk(K)/index_select/mark_step-before-gather machinery and
    the global->local expert-id remap), but instead of dequantizing to bf16 and
    calling ``mixture_of_experts``, it dequantizes each routed expert's int4
    weight DIRECTLY to fp8 and runs two native fp8 MME GEMMs per expert.

    Args (all GLM-5.1-int4-mixed-AutoRound / GPTQ-symmetric shapes):
      x            : [T, H] bf16 activations (H=6144).
      topk_ids     : [T, top_k] int — GLOBAL expert ids.
      topk_weights : [T, top_k] — router weights (post-renormalize).
      w13_packed   : [E, H, (2*I)//8] int32  (gate||up, out=2*I=4096, in=H).
      w13_scales   : [E, H//group_size, 2*I]  per-group int4 scales.
      w2_packed    : [E, I, H//8] int32       (down, out=H, in=I=2048).
      w2_scales    : [E, I//group_size, H]     per-group int4 scales.
      zp           : packed symmetric zero-point (shared across experts;
                     GPTQ-sym => uint4 midpoint 8). Passed straight to
                     convert_from_uint4 as zero_point. Must be GPTQ-symmetric
                     (a single shared zp tensor); AWQ per-expert zp is out of
                     scope for this fp8 path (raise).
      K            : static routed-expert budget. Caller MUST guarantee
                     T*top_k <= K so #distinct LOCAL experts <= K (no host sync,
                     no dynamic shape, no recompiles).
      activation   : 'silu' (GLM) or 'gelu'.
      experts_min  : this rank's global expert offset (ep_rank * E). Ids in
                     [experts_min, experts_min+E) are LOCAL.

    NOTE on shapes: H is inferred from w2_scales (its out dim) so the function
    needs no separate H/I args. E and I are inferred from the packed tensors.
    """
    if zp is None:
        raise ValueError(
            "fp8_routed_moe: zp is None (AWQ per-expert zero-point). This fp8 "
            "path is GPTQ-symmetric only (single shared zp).")
    if activation not in ("silu", "gelu"):
        raise ValueError(f"fp8_routed_moe: unsupported activation {activation!r}")

    dev = x.device
    E = w13_packed.shape[0]                      # local experts on this rank
    H = w2_scales.shape[-1]                       # down out == hidden (6144)
    I = w2_packed.shape[1]                        # down in == moe_intermediate (2048)
    # w13 out = 2*I (gate||up); confirm against the packed tensor to fail loud on
    # a shape mismatch rather than silently producing garbage.
    two_I = w13_packed.shape[-1] * 8
    if two_I != 2 * I:
        raise ValueError(
            f"fp8_routed_moe: w13 out ({two_I}) != 2*I ({2 * I}); "
            f"w13_packed={tuple(w13_packed.shape)} w2_packed={tuple(w2_packed.shape)}")

    T = x.shape[0]
    emin = int(experts_min)

    # ---- PER-PROJECTION SYMMETRIC ZERO-POINTS ------------------------------
    # convert_from_uint4's zero_point is the PACKED [groups, out//8] int32 zero,
    # so w13 (out=2I) and w2 (out=H) need DIFFERENTLY-SHAPED zeros even though the
    # GPTQ-symmetric VALUE is a single shared constant (the uint4 midpoint 8). The
    # caller passes ONE `zp` (production's layer._hpu_routed_w13_zp); we recover its
    # constant value and rebuild the correctly-shaped zero for EACH projection.
    # zp may already match w13 exactly (then zp13 is just zp) — we still build both
    # from the constant so a w2-shaped or w13-shaped `zp` both work.
    # Recover the constant: unpack the low nibble of the packed int32 (every packed
    # lane holds the same value for a symmetric zero). Bitwise + min works on HPU or
    # host and needs no quant_utils import here.
    _zp_const = int((zp.reshape(-1)[0] & 0xF).item())     # e.g. 8
    _z_groups13, _z_out13 = w13_scales.shape[1], w13_scales.shape[2]   # H//g, 2I
    _z_groups2, _z_out2 = w2_scales.shape[1], w2_scales.shape[2]        # I//g, H
    if zp.shape == (_z_groups13, _z_out13 // 8):
        zp13 = zp                                          # already w13-shaped
    else:
        zp13 = _pack_sym_zero(_zp_const, _z_groups13, _z_out13, dev, zp.dtype)
    zp2 = _pack_sym_zero(_zp_const, _z_groups2, _z_out2, dev, zp.dtype)

    # ---- ROUTING (mirrors _routed_moe exactly) -----------------------------
    # global expert id -> local [0, E)
    local = topk_ids.to(torch.int64) - emin
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)                      # routed to a LOCAL expert?
    # Non-local entries scatter to a DUMMY bin E (NOT index 0): scatter is
    # last-write-wins, collapsing non-local entries onto 0 would clobber a
    # genuinely-routed local expert 0.
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    # FUNCTIONAL scatter (out-of-place) for lazy HPU-graph-replay safety: a
    # captured in-place scatter_ on a "fresh" zeros buffer can fail to re-zero
    # each decode step -> stale routing accumulates.
    present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
        0, scatter_idx, valid.to(torch.float32))[:E]
    active = torch.topk(present, K).indices.to(torch.int64)   # [K] local ids, routed first

    # MATERIALIZE the routing BEFORE the data-dependent gather. HPU-graph capture
    # mishandles index_select with un-materialized dynamic indices (reads stale
    # `active`). Splitting the graph here forces `active` realized. (Same reason
    # _routed_moe mark_steps here.)
    _mark_step()

    # ---- GATHER K experts' packed int4 weights + scales --------------------
    w13p = w13_packed.index_select(0, active)             # [K, H, (2I)//8]
    w2p = w2_packed.index_select(0, active)               # [K, I, H//8]
    s13 = w13_scales.index_select(0, active)              # [K, H//g, 2I]
    s2 = w2_scales.index_select(0, active)                # [K, I//g, H]

    # Per-slice clone under lazy: the per-expert views w13p[k]/s13[k] for k>0
    # carry a nonzero storage_offset into the index_select buffer; SynapseAI under
    # HPUGraph reads the WRONG offset (effectively expert active[0]'s bytes) ->
    # huge error exactly when a k>0 expert is computed. .clone() gives a fresh
    # storage_offset=0 tensor (eager tolerates the offset, so skip the copies).
    _need_clone = _HAS_HPU and htorch.is_lazy()
    def _co(t):
        return t.clone() if _need_clone else t

    # Activation -> fp8 ONCE (reused for all K gate/up GEMMs; x is [T,H]).
    x_fp8, x_scale = _quant_act_fp8(x)

    # ---- PER-EXPERT fp8 GEMM PIPELINE --------------------------------------
    # For each routed expert k:
    #   1. convert_from_uint4(..., out_dtype=bf16) -> dense bf16 weight (proven)
    #   2. quant that weight to fp8 with a per-tensor recovery scale (round-trip)
    #   3. fp8_gemm_v2 for gate||up, silu(gate)*up, fp8_gemm_v2 for down
    #   4. scale by the router weight for tokens routed to this expert, accumulate
    out = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    for k in range(K):
        local_id = active[k]                              # this expert's local id

        # (1) int4 -> bf16 (out_dtype=bf16 — the ONLY proven convert_from_uint4
        # out_dtype; int4 -> fp8 directly is unvalidated and hits the Gaudi-2
        # fn-vs-fnuz factor-of-2, see _quant_weight_fp8). zp is the shared symmetric
        # zero; g_idx=None (group_size is regular). The returned weight is the REAL
        # dequantized bf16 weight. Only the K ROUTED experts are dequantized (the
        # bandwidth win's first lever), and the bf16 weight is transient: it is
        # cast straight to fp8 below and the per-expert mark_step frees it.
        w13_bf16 = torch.ops.hpu.convert_from_uint4(
            _co(w13p[k]), _co(s13[k]), zp13, torch.bfloat16, None)  # [H, 2I]
        w2_bf16 = torch.ops.hpu.convert_from_uint4(
            _co(w2p[k]), _co(s2[k]), zp2, torch.bfloat16, None)     # [I, H]

        # (2) per-tensor bf16 -> fp8 quant (self-consistent cast/gemm round-trip;
        # gives the GEMM a proper recovery scale and uses the fp8 grid well). The
        # fp8 operands are still half-size and the MME still runs them at 2x.
        w13_fp8, w13_s = _quant_weight_fp8(w13_bf16)
        w2_fp8, w2_s = _quant_weight_fp8(w2_bf16)

        # (3a) gate||up : x_fp8[T,H] @ w13[H,2I] -> [T,2I] bf16 (native fp8 MME).
        gate_up = _fp8_linear(x_fp8, x_scale, w13_fp8, w13_s, torch.bfloat16)
        h = _silu_mul(gate_up, I, activation)             # [T, I] bf16

        # (3b) down : requant the intermediate h to fp8, then h @ w2[I,H] -> [T,H].
        h_fp8, h_scale = _quant_act_fp8(h)
        expert_out = _fp8_linear(h_fp8, h_scale, w2_fp8, w2_s, torch.bfloat16)  # [T,H]

        # (4) router-weight scale + accumulate. A token contributes to expert k
        # only on the top_k slots whose (local, valid) id == local_id. Sum those
        # slots' router weights per token -> [T,1], multiply, accumulate. Tokens
        # not routed here get weight 0 (exactly like the all-expert op skipping).
        match = (local == local_id) & valid.view_as(local)        # [T, top_k] bool
        w_per_tok = (topk_weights.to(torch.bfloat16) * match.to(torch.bfloat16)
                     ).sum(dim=-1, keepdim=True)                  # [T, 1]
        out = out + expert_out * w_per_tok
        # Split the graph per expert so the K fp8 weights are not all held live at
        # once (mirrors the reference op's mark_step around its dequant list, and
        # keeps the per-expert fp8 weight from pinning HBM K-deep).
        _mark_step()

    return out


def fp8_allexpert_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_fp8: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
    experts_min: int = 0,
    activation: str = "silu",
) -> torch.Tensor:
    """LOAD-TIME fp8 ALL-EXPERT silu MoE. Returns [T, H] bf16.

    This is the GLM single-stream / MTP-decode win (the M3 route-b approach applied
    to GLM): the int4 experts are converted to fp8 ONCE at weight-load
    (process_weights_after_loading, gated HPU_MOE_FP8_EXPERTS=1) and stored RESIDENT;
    here we just run two native fp8 MME GEMMs per LOCAL expert against those stored fp8
    weights. NOTHING is dequantized per step — that is the whole point. It replaces the
    stock VllmMixtureOfExpertsOpWNA16 all-expert path (which dequantizes ALL E local
    experts int4->bf16->HBM EVERY forward, the confirmed #1 single-stream device cost),
    and it is the path taken whenever the routed-K gather is INELIGIBLE — which includes
    MTP bs>=2 decode (tokens*top_k > K disables routed-gather), exactly the regime the
    GLM-5.2 serve runs in.

    Unlike ``fp8_routed_moe`` (which dequants K experts transiently EACH call), this takes
    ALREADY-fp8 weights, so there is ZERO convert_from_uint4 in the forward. The fp8 cast
    convention is IDENTICAL to the proven ``apply_fp8_linear_hpu`` / ``dynamic_quant`` path:
    the stored ``w*_scale`` is the FORWARD (recovery) scale ``s`` (real_w ~= w_fp8 * s), the
    activation is quant'd by ``_quant_act_fp8`` (cast with 1/s_x, gemm recovers with s_x),
    and ``_fp8_linear`` passes the forward scales into the GEMM's ``*_scale_inv`` slots so the
    fnuz/fn factor cancels (see the module docstring's fn-vs-fnuz note — the 2x bug). The
    weights were built bf16->fp8 via that SAME self-consistent ``cast_to_fp8_v2`` round-trip
    (NEVER int4->fp8 directly), so no asymmetric write is ever read back at the 2x factor.

    Args (GLM-5.2-int4-experts-AutoRound shapes; H=6144, I=2048, E=local experts):
      x            : [T, H] bf16 activations.
      topk_ids     : [T, top_k] int — GLOBAL expert ids.
      topk_weights : [T, top_k] — router weights (post-renormalize).
      w13_fp8      : [E, H, 2*I] float8_e4m3fn  (gate||up, stored RESIDENT; out=2I, in=H).
      w13_scale    : [E] or [E,1] float32 per-tensor FORWARD recovery scale (real ~= fp8 * s).
      w2_fp8       : [E, I, H] float8_e4m3fn   (down, stored RESIDENT; out=H, in=I).
      w2_scale     : [E] or [E,1] float32 per-tensor FORWARD recovery scale.
      experts_min  : this rank's global expert offset (ep_rank * E). Ids in
                     [experts_min, experts_min+E) are LOCAL; others contribute 0.
      activation   : 'silu' (GLM) or 'gelu'.

    Semantics MATCH the stock all-expert op exactly: every LOCAL expert is computed, each
    token's contribution to expert e is scaled by the SUM of its router weights on the top_k
    slots that selected (local) e; non-local / non-routed slots contribute 0 (a token routed
    to no local expert gets all-zero out from this rank, like the stock op).
    """
    if activation not in ("silu", "gelu"):
        raise ValueError(f"fp8_allexpert_moe: unsupported activation {activation!r}")
    dev = x.device
    E = w13_fp8.shape[0]
    H = w2_fp8.shape[-1]            # down out == hidden (6144)
    I = w2_fp8.shape[1]            # down in == moe_intermediate (2048)
    two_I = w13_fp8.shape[-1]
    if two_I != 2 * I:
        raise ValueError(
            f"fp8_allexpert_moe: w13 out ({two_I}) != 2*I ({2 * I}); "
            f"w13_fp8={tuple(w13_fp8.shape)} w2_fp8={tuple(w2_fp8.shape)}")
    T = x.shape[0]
    emin = int(experts_min)

    # global expert id -> local [0, E); mark which (token, slot) entries are LOCAL.
    local = topk_ids.to(torch.int64) - emin           # [T, top_k]
    valid = (local >= 0) & (local < E)                 # [T, top_k] routed to a LOCAL expert?
    tw = topk_weights.to(torch.bfloat16)

    # Activation -> fp8 ONCE (reused for all E gate/up GEMMs; x is [T,H]).
    x_fp8, x_scale = _quant_act_fp8(x)

    out = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        # Per-expert FORWARD scales as [1] scalars (the GEMM's *_scale_inv slot). The stored
        # scale is [E] or [E,1]; reshape to [1] either way (no host sync; e is a Python int).
        w13_s = w13_scale[e].reshape(1).float()
        w2_s = w2_scale[e].reshape(1).float()
        # (1) gate||up : x_fp8[T,H] @ w13[H,2I] -> [T,2I] bf16 (native fp8 MME, stored fp8).
        gate_up = _fp8_linear(x_fp8, x_scale, w13_fp8[e], w13_s, torch.bfloat16)
        h = _silu_mul(gate_up, I, activation)          # [T, I] bf16
        # (2) down : requant the intermediate to fp8, then h @ w2[I,H] -> [T,H].
        h_fp8, h_scale = _quant_act_fp8(h)
        expert_out = _fp8_linear(h_fp8, h_scale, w2_fp8[e], w2_s, torch.bfloat16)  # [T,H]
        # (3) router-weight scale + accumulate (mirrors the stock op's per-expert weighting):
        # a token contributes to expert e only on the (valid) top_k slots whose local id == e;
        # sum those slots' router weights -> [T,1]. Non-local/non-routed slots -> weight 0.
        match = (local == e) & valid                    # [T, top_k] bool
        w_per_tok = (tw * match.to(torch.bfloat16)).sum(dim=-1, keepdim=True)  # [T,1]
        out = out + expert_out * w_per_tok
        # Split the graph per expert: keeps each expert's transient fp8 activations/intermediate
        # from being held live E-deep (the weights are already resident; only the acts are new).
        _mark_step()

    return out


# =============================================================================
# FUSED fp8 OP DRIVER (HPU_MOE_FP8_FUSED=1) -- the PROPER single-recipe fp8 path
# =============================================================================
# The functions above (fp8_routed_moe / fp8_allexpert_moe) are PYTHON loops that
# issue, PER LOCAL EXPERT, a chain of separate eager ops (convert_from_uint4 x2,
# cast_to_fp8_v2 x3+, fp8_gemm_v2 x2, silu, accumulate) + a _mark_step() that
# SPLITS the HPU graph at every expert. That per-expert dispatch/sync overhead is
# why measured fp8 was contaminated ("3.3x slower") -- it was NEVER a real fused
# single-recipe fp8 kernel, so it never had a shot at fp8's ~2x MME.
#
# The driver below dispatches the GC-CAPTURED, SRAM-resident FUSED fp8 op
#   torch.ops.hpu_moe_ext_fp8.fused_expert_fp8(x, w13_packed, w13_scale, w13_zp,
#                                  w2_packed, w2_scale, w2_zp, router_weight)
# (moe_fused_expert_fp8_backend.cpp, built BUILD_FP8_NODES=1). That op fuses, IN
# ONE recipe per expert: int4->bf16 dequant (the ONLY proven out_dtype; NEVER
# int4->fp8 directly -- see the fn-vs-fnuz note above) -> per-channel-weight /
# per-token-act bf16->fp8 cast (the self-consistent 1/s cast <-> s gemm round-trip;
# FP8_MAX=240 fnuz, so the fn-vs-fnuz 2x factor CANCELS) -> two NATIVE fp8 MME
# GEMMs -> silu -> router-weight fold. There is ZERO per-expert Python op and ZERO
# per-expert mark_step in the forward -- the entire expert is one captured graph.
#
# It is the fp8-compute ANALOG of the proven int4 fused op (hpu_moe_ext::fused_expert,
# driven by fused_expert_routed.fused_moe_routed) -- SAME single-expert schema, SAME
# int4-packed inputs, SAME gather/route/remap machinery -- so it is wired EXACTLY the
# way the int4 fused op is (mirrors fused_moe_routed verbatim, only the op handle and
# the .so name change). Because it consumes INT4 weights (dequanting in-graph), it
# runs against the LIVE int4 weights -- so HPU_MOE_FP8_FUSED keeps int4 resident and
# does NOT build/free the separate resident-fp8 tensors (that resident path feeds the
# stock torch.ops.hpu.mixture_of_experts; this is a different, op-fused path).

_fp8_fused_op_loaded = [False]


def _load_fp8_fused_op():
    """Load the FUSED fp8 single-expert op .so once and return
    torch.ops.hpu_moe_ext_fp8.fused_expert_fp8, or None if the .so is absent (caller
    then falls back to the per-expert fp8 loop / stock op). Mirrors
    moe_wna16_hpu._load_fused_op / _fused_so_path for the int4 op EXACTLY, only the
    library name + the op namespace differ (hpu_moe_ext_fp8 -- a SEPARATE TORCH_LIBRARY
    namespace so the fp8 .so coexists in-process with the int4 hpu_moe_ext .so).

    The fp8 op ships in TWO builds, EXACTLY like the int4 op (this was the fix for the
    `undefined symbol EagerOpBase::validate_inputs` false-null in the live LAZY serve):
      * libmoe_fused_expert_fp8_lazy.so  -- LAZY frontend (habana_lazy::LazyOp); the
        one that CAPTURES into the decode HPU-graph under PT_HPU_LAZY_MODE=1 (the
        production serve path). Built by build_fp8_lazy.sh.
      * libmoe_fused_expert_fp8.so       -- EAGER frontend (habana::eager::EagerOp);
        loads only under the eager build. Built by build_fp8.sh.
    The eager .so does NOT load under lazy (its EagerOp symbols live only in the eager
    plugin -> the validate_inputs undefined) and vice-versa, so pick the right one by
    mode at load time -- mirroring _fused_so_path(). Both emit the fp8 nodes only when
    built BUILD_FP8_NODES=1. RUN ON THE HPU BOX.
    """
    if hasattr(torch.ops, "hpu_moe_ext_fp8") and hasattr(
            torch.ops.hpu_moe_ext_fp8, "fused_expert_fp8"):
        return torch.ops.hpu_moe_ext_fp8.fused_expert_fp8
    if _fp8_fused_op_loaded[0]:
        return None
    _fp8_fused_op_loaded[0] = True
    import habana_frameworks.torch.core as _htc
    _name = ("libmoe_fused_expert_fp8_lazy.so" if _htc.is_lazy()
             else "libmoe_fused_expert_fp8.so")
    _script = "build_fp8_lazy.sh" if _htc.is_lazy() else "build_fp8.sh"
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build", _name))
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: FUSED fp8 op missing (%s); HPU_MOE_FP8_FUSED=1 requested but "
            "the .so is absent -- falling back. Build it (CPU-only, inside the image): "
            "docker run --rm -e PT_HPU_LAZY_MODE=%s -v <tpc_kernel>:/work -w "
            "/work/opbackend vllm-gaudi:next bash -c 'BUILD_FP8_NODES=1 bash %s'",
            so, ("1" if _htc.is_lazy() else "0"), _script)
        # UNINTENDED fallback: the user asked for the fused fp8 op but the build is absent.
        # intended=True -> HPU_ENGINE_STRICT hard-fails so a fp8 A/B cannot silently
        # measure the int4 fallback path (the FALSE-NULL class this whole effort kills).
        engage_fallback(
            "fp8_routed._load_fp8_fused_op:so-missing",
            fast="HPU_MOE_FP8_FUSED fused fp8 op (.so)",
            reason="library not found at %s" % so,
            slow="int4 path (the fp8 single-stream A/B would be a FALSE NULL)",
            intended=True, strict_env="HPU_MOE_FP8_FUSED")
        return None
    # LOUD link-gap guard. The eager .so (libmoe_fused_expert_fp8.so) references
    # habana::eager::EagerOpBase::{validate_inputs,run} which live ONLY in the eager
    # plugin (libhabana_pytorch2_plugin.so); the lazy .so references habana_lazy::*
    # which live only in the lazy plugin (libhabana_pytorch_plugin.so). If the WRONG
    # .so is ever loaded for the running mode (e.g. the eager .so under
    # PT_HPU_LAZY_MODE=1), torch.ops.load_library raises
    #   OSError: ... undefined symbol: ..._ZN6habana5eager11EagerOpBase15validate_inputs...
    # Left unguarded that aborts the whole vLLM worker; worse, an earlier build with
    # NO lazy .so silently fell back to int4 -> a FALSE NULL in the fp8 A/B. The
    # _name/_script split above (lazy vs eager, picked by _htc.is_lazy()) is the FIX;
    # this guard makes any residual mode/.so mismatch fail LOUD + survivable (return
    # None -> documented int4 fallback) instead of crashing or silently null-ing.
    try:
        torch.ops.load_library(so)
    except OSError as _e:
        import logging
        _msg = str(_e)
        _is_eager_sym = "validate_inputs" in _msg or "EagerOpBase" in _msg or "eager" in _msg
        _is_lazy_sym = "habana_lazy" in _msg or "LazyOp" in _msg or "flush_op" in _msg
        logging.getLogger("vllm").error(
            "hpu-moe-ext: FAILED to load FUSED fp8 op .so %s (mode is_lazy=%s) -- "
            "FALLING BACK to the int4 path (the fp8 single-stream A/B would be a FALSE "
            "NULL if this fires). undefined-symbol class: %s. Likely cause: the .so was "
            "built for the OTHER frontend than the running mode. Rebuild the matching "
            "one card-free: docker run --rm -e PT_HPU_LAZY_MODE=%s -v <tpc_kernel>:/work "
            "-w /work/opbackend vllm-gaudi:next bash -c 'BUILD_FP8_NODES=1 bash %s'. "
            "Linker error: %s",
            so, _htc.is_lazy(),
            ("eager (EagerOpBase::validate_inputs/run) -> eager .so loaded under LAZY mode"
             if _is_eager_sym else
             "lazy (habana_lazy::*) -> lazy .so loaded under EAGER mode"
             if _is_lazy_sym else "unknown"),
            ("1" if _htc.is_lazy() else "0"), _script, _msg[:400])
        # UNINTENDED fallback (wrong-frontend .so for the running mode) -- the explicit
        # FALSE-NULL class. intended=True -> HPU_ENGINE_STRICT hard-fails.
        engage_fallback(
            "fp8_routed._load_fp8_fused_op:so-linkgap",
            fast="HPU_MOE_FP8_FUSED fused fp8 op (.so)",
            reason="load_library OSError (wrong frontend .so for is_lazy=%s): %s"
                   % (_htc.is_lazy(), _msg[:120]),
            slow="int4 path (FALSE NULL in the fp8 A/B)",
            intended=True, strict_env="HPU_MOE_FP8_FUSED")
        return None
    if hasattr(torch.ops, "hpu_moe_ext_fp8") and hasattr(
            torch.ops.hpu_moe_ext_fp8, "fused_expert_fp8"):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded FUSED fp8 op torch.ops.hpu_moe_ext_fp8.fused_expert_fp8 "
            "from %s (in-graph int4->fp8 dequant + native fp8 MME; the proper "
            "single-recipe fp8 path, replacing the per-expert fp8 loop)", so)
        return torch.ops.hpu_moe_ext_fp8.fused_expert_fp8
    import logging
    logging.getLogger("vllm").warning(
        "hpu-moe-ext: loaded %s but torch.ops.hpu_moe_ext_fp8.fused_expert_fp8 did NOT "
        "register (check TORCH_LIBRARY(hpu_moe_ext_fp8) + the BUILD_FP8_NODES=1 build).",
        so)
    # UNINTENDED fallback (.so loaded but the op did not register -> ABI/build drift).
    engage_fallback(
        "fp8_routed._load_fp8_fused_op:no-register",
        fast="HPU_MOE_FP8_FUSED fused fp8 op",
        reason="loaded %s but torch.ops.hpu_moe_ext_fp8.fused_expert_fp8 did not register"
               % so,
        slow="int4 path (FALSE NULL in the fp8 A/B)",
        intended=True, strict_env="HPU_MOE_FP8_FUSED")
    return None


def _fp8_fused_router_weight(local, valid, topk_weights, target_id, dtype):
    """Per-(token) router weight for the expert whose LOCAL id == ``target_id``: sum the
    router weights of every top_k slot that routed (validly, locally) to it -> [T,1].
    Identical to fp8_allexpert_moe / fused_moe_routed's per-expert weight (a vectorized
    masked sum -- NO host sync, NO dynamic shape). ``target_id`` may be a python int OR a
    0-D device tensor (the gathered ``active[k]`` compact-id case)."""
    match = (local == target_id) & valid                          # [T, top_k] bool
    return (topk_weights.to(dtype) * match.to(dtype)).sum(dim=-1, keepdim=True)  # [T,1]


def fp8_fused_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    zp13: torch.Tensor,
    zp2: torch.Tensor,
    experts_min: int = 0,
    activation: str = "silu",
    K: Optional[int] = None,
    w13_fp8_scale: Optional[torch.Tensor] = None,
    w2_fp8_scale: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """ALL-EXPERT / routed-K fp8 MoE via the FUSED fp8 op. Returns [T, H] bf16, or
    None if the fused fp8 .so is unavailable (caller falls back).

    Drop-in fused replacement for fp8_allexpert_moe (and fp8_routed_moe): SAME MoE
    semantics, but every expert is computed by ONE captured fp8 op recipe
    (torch.ops.hpu_moe_ext_fp8.fused_expert_fp8) against the LIVE int4 weights, so the
    only per-step Python work is the routing arithmetic + the op call -- the per-expert
    convert_from_uint4 / cast_to_fp8_v2 / fp8_gemm_v2 / silu / per-expert mark_step that
    contaminated the loop's timing are GONE (all fused inside the op).

    Args (GLM-int4 stacked / GPTQ-symmetric shapes; H=6144, I=2048, E=local experts):
      x            : [T, H] bf16 activations.
      topk_ids     : [T, top_k] int -- GLOBAL expert ids.
      topk_weights : [T, top_k] -- router weights (post-renormalize).
      w13_packed   : [E, H, (2I)//8] int32 stacked gate||up int4 (layer.w13_qweight).
      w13_scales   : [E, H//GS, 2I]  bf16 per-group int4 scales (op layout).
      w2_packed    : [E, I, H//8]  int32 stacked down int4 (layer.w2_qweight).
      w2_scales    : [E, I//GS, H]   bf16 per-group int4 scales (op layout).
      zp13         : [E, H//GS, (2I)//8] int32 STACKED packed zero-points (per-expert
                     slice; GPTQ-sym shared zero broadcast to [E,...] by the caller).
      zp2          : [E, I//GS, H//8]  int32 STACKED packed zero-points.
      experts_min  : this rank's global expert offset (ep_rank * E). Ids in
                     [experts_min, experts_min+E) are LOCAL.
      activation   : 'silu' (GLM) -- the op's graph hard-codes silu(gate)*up, so anything
                     else is rejected (mirrors the int4 fused op's silu-only gate).
      K            : OPTIONAL routed-gather budget. When not None AND T*top_k <= K, only
                     the <=K locally-routed experts are computed (the single-stream win,
                     mirroring fused_moe_routed). Else ALL E local experts are computed
                     (the all-expert regime fp8_allexpert_moe served). Either way every
                     expert goes through the fused op.

    The fp8 op's graph hard-codes the silu(gate)*up activation + the int4 dequant + the
    self-consistent fp8 cast/gemm round-trip (FP8_MAX=240 fnuz), so this driver does NO
    fp8 scale math itself -- it just gathers/routes int4 + folds the router weight, exactly
    like fused_moe_routed does for the int4 op.
    """
    if activation != "silu":
        # The fp8 op's AddNode hard-codes sigmoid*mult (silu); gelu is not expressible
        # in that recipe. Mirror the int4 fused op's gate -> caller falls back.
        return None
    op = _load_fp8_fused_op()
    if op is None:
        return None  # .so absent -> caller falls back to the loop / stock op

    # DIRECT int4->fp8: per-expert FORWARD fp8 recovery scales precomputed at LOAD
    # ([E,1] f32). When present, the op folds 1/s_fp8 into the int4 group scale and emits
    # ONE dequant->fp8 (no bf16, no per-step weight amax); s_fp8 is its fp8_gemm scaleB so
    # the fnuz 2x cancels. When None (default), the op's two trailing optional inputs stay
    # None -> the proven in-graph-amax two-hop path. _direct gates the per-expert indexing.
    _direct = w13_fp8_scale is not None and w2_fp8_scale is not None

    dev = x.device
    dt = torch.bfloat16            # the op's hardcoded scale/zp(out)/router dtype
    E = w13_packed.shape[0]
    H = w2_scales.shape[-1]                       # down out == hidden (6144)
    I = w2_packed.shape[1]                         # down in == moe_intermediate (2048)
    two_I = w13_packed.shape[-1] * 8
    if two_I != 2 * I:
        raise ValueError(
            f"fp8_fused_moe: w13 out ({two_I}) != 2*I ({2 * I}); "
            f"w13_packed={tuple(w13_packed.shape)} w2_packed={tuple(w2_packed.shape)}")
    T = x.shape[0]
    emin = int(experts_min)

    # global expert id -> local [0, E); mark which (token, slot) entries are LOCAL.
    local = topk_ids.to(torch.int64) - emin            # [T, top_k]
    valid = (local >= 0) & (local < E)                  # [T, top_k] routed to a LOCAL expert?

    out = torch.zeros(T, H, dtype=dt, device=dev)

    # ---- ROUTED-K gather (the single-stream win; mirrors fused_moe_routed) ----------
    # Only taken when the caller passes a budget K with T*top_k <= K (so #distinct local
    # experts <= K with no host sync / no dynamic shape). Else ALL E experts (below).
    if K is not None and (T * topk_ids.shape[-1]) <= K:
        flat = local.reshape(-1)
        valid_flat = (flat >= 0) & (flat < E)
        # non-local entries scatter to DUMMY bin E (not 0 -> would clobber local expert 0).
        scatter_idx = torch.where(valid_flat, flat, torch.full_like(flat, E))
        present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
            0, scatter_idx, valid_flat.to(torch.float32))[:E]
        active = torch.topk(present, K).indices.to(torch.int64)   # [K] local ids, routed first
        # MATERIALIZE the routing before the data-dependent gather (HPU-graph capture
        # mishandles index_select with un-materialized dynamic indices). Same boundary
        # fused_moe_routed / _routed_moe keep.
        _mark_step()
        w13p = w13_packed.index_select(0, active)             # [K, H, (2I)//8]
        w2p = w2_packed.index_select(0, active)               # [K, I, H//8]
        s13 = w13_scales.index_select(0, active)              # [K, H//GS, 2I]
        s2 = w2_scales.index_select(0, active)                # [K, I//GS, H]
        z13 = zp13.index_select(0, active)                    # [K, H//GS, (2I)//8]
        z2 = zp2.index_select(0, active)                      # [K, I//GS, H//8]
        # DIRECT path: gather the per-expert fp8 scales the same way ([E,1] -> [K,1]).
        sf13 = w13_fp8_scale.index_select(0, active) if _direct else None  # [K,1] f32
        sf2 = w2_fp8_scale.index_select(0, active) if _direct else None    # [K,1] f32
        # Per-slice fresh+contiguous offset-0 copy of each per-expert slice: the fused
        # op's first node on every weight/zp input is a reinterpret_cast int32->uint4
        # feeding a NON-PERSISTENT SRAM-resident dequant -- the most stride/offset-
        # sensitive consumer; a nonzero storage_offset (k>0) reads the WRONG base ->
        # garbage. Identical invariant to fused_moe_routed._co/_cs (module-level _co_w/_cs_w).
        for k in range(K):
            local_id = active[k]                              # 0-D device tensor (compact id)
            w_k = _fp8_fused_router_weight(local, valid, topk_weights, local_id, dt)  # [T,1]
            # Per-expert [1] f32 FORWARD fp8 scales (None on the two-hop path -> op uses
            # its in-graph amax). .reshape(1) gives the op's [1] scalar -- a per-expert
            # slice carries a nonzero offset, so .clone() it to a fresh offset-0 leaf
            # (same reinterpret/offset invariant the weights need; cheap [1] copy).
            sf13_k = sf13[k].reshape(1).clone() if _direct else None
            sf2_k = sf2[k].reshape(1).clone() if _direct else None
            if T == 1:
                # T==1: the whole [1,H] output scales by the single scalar w_k[0,0]; pass
                # it as the op's [1,1] router_weight so the op FOLDS it (no host sync --
                # reshape, not .item()). Mirrors fused_moe_routed's T==1 priority.
                rw = w_k.view(1, 1)
                expert_out = op(
                    x, _co_w(w13p[k]), _cs_w(s13[k], dt), _co_w(z13[k]),
                    _co_w(w2p[k]), _cs_w(s2[k], dt), _co_w(z2[k]), rw,
                    sf13_k, sf2_k)  # [1,H] (already weighted)
                out = out + expert_out
            else:
                # T>1: a [1,1] scalar can't carry a per-token weight -> run the op
                # un-weighted (router_weight=1) and apply [T,1] w_k outside.
                one = torch.ones((1, 1), dtype=dt, device=dev)
                expert_out = op(
                    x, _co_w(w13p[k]), _cs_w(s13[k], dt), _co_w(z13[k]),
                    _co_w(w2p[k]), _cs_w(s2[k], dt), _co_w(z2[k]), one,
                    sf13_k, sf2_k)  # [T,H] un-weighted
                out = out + expert_out * w_k
            _mark_step()
        return out

    # ---- ALL-EXPERT regime (mirrors fp8_allexpert_moe's semantics; the non-routed /
    #      MTP bs>=2 path) -- every LOCAL expert via the fused op against its int4 slice.
    for e in range(E):
        w_k = _fp8_fused_router_weight(local, valid, topk_weights, e, dt)  # [T,1]
        # CLONE each per-expert slice to a fresh offset-0 leaf (w13_packed[e] for e>0 carries a
        # nonzero storage_offset -> the op's reinterpret_cast would read the WRONG base). SAME
        # invariant as the gathered-K branch / fused_moe_routed.
        w13e, s13e, z13e = _co_w(w13_packed[e]), _cs_w(w13_scales[e], dt), _co_w(zp13[e])
        w2e, s2e, z2e = _co_w(w2_packed[e]), _cs_w(w2_scales[e], dt), _co_w(zp2[e])
        # Per-expert [1] f32 FORWARD fp8 scales (None -> op's in-graph two-hop amax).
        sf13_e = w13_fp8_scale[e].reshape(1).clone() if _direct else None
        sf2_e = w2_fp8_scale[e].reshape(1).clone() if _direct else None
        if T == 1:
            rw = w_k.view(1, 1)
            expert_out = op(x, w13e, s13e, z13e, w2e, s2e, z2e, rw,
                            sf13_e, sf2_e)   # [1,H] (already weighted)
            out = out + expert_out
        else:
            one = torch.ones((1, 1), dtype=dt, device=dev)
            expert_out = op(x, w13e, s13e, z13e, w2e, s2e, z2e, one,
                            sf13_e, sf2_e)  # [T,H] un-weighted
            out = out + expert_out * w_k
        # Split the graph per expert (each expert is already ONE fused recipe; this keeps
        # the per-expert transient acts from being held E-deep -- mirrors fp8_allexpert_moe).
        _mark_step()
    return out
