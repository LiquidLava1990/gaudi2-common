"""MONOLITHIC (ALL-E experts) fused M3 swigluoai-FP8 MoE for the THROUGHPUT/concurrent
regime. The fp8 sibling of fused_monolithic_moe.py (int4). Where the int4 op dequants
all E int4 experts SRAM-resident, this op feeds the RESIDENT fp8 experts straight to the
native fp8 MME (fp8 x fp8 -> bf16, 2x rate; the activation is dynamic-quantized in-graph)
and applies the EXACT M3 swigluoai gate -- so it replaces the route-b per-expert PYTHON
loop in m3_expert_runtime_patch._fp8_route_b_allexpert with ONE captured recipe over all
E local experts.

    torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8(
        x[T,H] bf16,
        w13_fp8[E,H,2I] float8_e4m3fn, w13_scale[E,2I] f32,
        w2_fp8 [E,I,H]  float8_e4m3fn, w2_scale [E,H]  f32,
        router_weights[E,T,1] bf16, int chunk, float alpha, float limit) -> [T,H] bf16

The weights are the layer's PERSISTENT route-b fp8 stacks (op._w13_fp8_stack [E,H,2I] +
op._s13_stack [E,2I]; op._w2_fp8_stack [E,I,H] + op._s2_stack [E,H]), built once at load
by m3_fp8_static_patch._build_fp8_stacks. They are passed DIRECTLY (no gather); the op
slices each chunk on-device. The ONLY per-call work here is building the DENSE [E,T,1]
router matrix from topk_ids/topk_weights -- IDENTICAL static-shape value-only arithmetic
to the int4 wrapper (one_hot(local_id)*safe_w sum over slots, NO scatter, NO host sync).

M3 DIMS (TP8 EP): H=6144, I=3072, 2I=6144, E_local=16, top_k=4, alpha=1.702, limit=7.0.
The whole fp8 expert set (~0.91 GB / card) is HBM-RESIDENT -- the HBM fit that the
GLM-753B-int4 monolithic op does NOT have (its all-E int4 dequant OOMs). RUN ON THE HPU BOX.
"""
from __future__ import annotations

import os

import torch
import habana_frameworks.torch.core as htorch

# CHUNK: experts processed per batched sub-chain inside the op (SRAM-pressure knob). M3
# has only E_local=16 experts and the fp8 weights are small (~56.6 MB/expert, NO bf16
# dequant scratch), so even CHUNK=E=16 is plausible; default 8 (the proven batched tile).
_CHUNK = int(os.environ.get("M3_FP8_MONO_CHUNK", "8") or "8")

_ALPHA_DEFAULT = 1.702
_LIMIT_DEFAULT = 7.0

_mono_so_loaded = [False]


def _load_monolithic_fp8_op():
    """Load the fp8 MONOLITHIC op .so (lazy variant under lazy, eager else). Idempotent.
    Returns torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8 or None (caller falls back to
    the per-expert route-b loop). Co-loads with the other fused .so via
    TORCH_LIBRARY_FRAGMENT(hpu_moe_ext)."""
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_monolithic_fp8"):
        return torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8
    if _mono_so_loaded[0]:
        return None
    _mono_so_loaded[0] = True
    name = ("libmoe_fused_monolithic_fp8_lazy.so" if htorch.is_lazy()
            else "libmoe_fused_monolithic_fp8.so")
    so = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        import logging
        logging.getLogger("vllm").warning(
            "hpu-moe-ext: MONOLITHIC fp8 op missing (%s); falling back to the per-expert "
            "route-b loop. Build it: bash tpc_kernel/opbackend/build_monolithic_fp8.sh%s",
            so, " (BUILD_LAZY=1 for the lazy .so)")
        return None
    torch.ops.load_library(so)
    if hasattr(torch.ops, "hpu_moe_ext") and hasattr(
            torch.ops.hpu_moe_ext, "fused_moe_monolithic_fp8"):
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext: loaded MONOLITHIC fp8 op from %s (all-E resident fp8 -> fp8 MME; "
            "the throughput-regime swigluoai-fp8 fusion win)", so)
        return torch.ops.hpu_moe_ext.fused_moe_monolithic_fp8
    return None


def fused_moe_monolithic_fp8(x, w13_fp8, w13_scale, w2_fp8, w2_scale,
                             topk_ids, topk_weights, experts_min,
                             alpha=None, limit=None, chunk=None, x_buf=None):
    """ALL-E monolithic M3 swigluoai-fp8 MoE via the chunked single-recipe op.

    Args (stacked per-expert tensors, PyTorch row-major, on HPU; the route-b stacks):
      x          : [T, H] bf16 hidden states (THROUGHPUT regime, T>1)
      w13_fp8    : [E, H, 2I] float8_e4m3fn  (op._w13_fp8_stack)
      w13_scale  : [E, 2I]    f32 per-output-channel forward scale (op._s13_stack)
      w2_fp8     : [E, I, H]  float8_e4m3fn  (op._w2_fp8_stack)
      w2_scale   : [E, H]     f32 per-output-channel forward scale (op._s2_stack)
      topk_ids   : [T, top_k] int -- GLOBAL expert ids (this rank's window @ experts_min)
      topk_weights : [T, top_k] -- router weights (already sigmoid-scored/renormalized)
      experts_min  : this rank's first GLOBAL expert id (ep_rank * E) == layer._hpu_experts_min
      alpha/limit  : swigluoai params (default 1.702 / 7.0; read off the layer when available)
      chunk      : experts/batched-sub-chain (default _CHUNK); clamped to [1,E] in the op
      x_buf      : OPTIONAL persistent [max_T, H] bf16 static-input buffer (lazy multi-bucket
                   capture fix); None -> x passed directly.

    Returns: [T, H] bf16 = sum_e( swigluoai_expert_e(x) * W[:,e] ).
    """
    op = _load_monolithic_fp8_op()
    if op is None:
        return None  # .so absent -> caller falls back to the per-expert route-b loop
    E = w13_fp8.shape[0]
    emin = int(experts_min)
    dev = x.device
    T = x.shape[0]
    ch = _CHUNK if chunk is None else int(chunk)
    a = _ALPHA_DEFAULT if alpha is None else float(alpha)
    lim = _LIMIT_DEFAULT if limit is None else float(limit)

    # ---- DENSE per-(token,expert) router-weight matrix W[T,E] -> [E,T,1] (VERBATIM the
    #      int4 wrapper's build; matches m3_expert_runtime_patch._fp8_route_b_allexpert's
    #      [T,E] route scatter EXACTLY, just transposed + one_hot instead of scatter_add). ----
    local = topk_ids.to(torch.int64) - emin                # [T, top_k] local ids
    valid = (local >= 0) & (local < E)
    safe_id = torch.where(valid, local, torch.zeros_like(local))
    safe_w = torch.where(valid, topk_weights.to(torch.float32),
                         torch.zeros_like(topk_weights, dtype=torch.float32))
    erange = torch.arange(E, device=dev)
    onehot = (safe_id.unsqueeze(-1) == erange).to(torch.float32)   # [T, top_k, E]
    W = (onehot * safe_w.unsqueeze(-1)).sum(dim=1)                 # [T, E] f32
    rw = W.transpose(0, 1).contiguous().to(torch.bfloat16).unsqueeze(-1)   # [E, T, 1] bf16

    # ---- x through the held-leaf static-input buffer (lazy multi-bucket capture fix) ----
    if x_buf is not None and htorch.is_lazy() and x_buf.shape[0] >= T:
        if x_buf.dtype != x.dtype or x_buf.device != x.device:
            raise RuntimeError(
                "hpu-moe-ext fused_moe_monolithic_fp8 x-buffer mismatch: buf is "
                f"{x_buf.dtype}/{x_buf.device} but x is {x.dtype}/{x.device}.")
        x_buf[:T].copy_(x)
        x_op = x_buf[:T]
    else:
        x_op = x

    # === STATIC-SHAPE RECIPE FUSION (no mark_step) ===
    # The rw build is static-shape value-only arithmetic (no data-dependent index), and the
    # op consumes the PERSISTENT fp8 stacks directly (no gather), so the bridge captures rw +
    # the whole MoE as ONE recipe. Restore the legacy boundary only to bisect a capture edge.
    if os.environ.get("M3_FP8_MONO_MARKSTEP", "0") == "1":
        htorch.mark_step()

    # Scales pinned to f32 (the op's fp8_gemm scaleB contract). Weights stay fp8. The
    # stacks are the layer's persistent route-b tensors, passed DIRECTLY; the op slices
    # each chunk on-device and runs the fp8 batch_gemm.
    out = op(
        x_op,
        w13_fp8, w13_scale.to(torch.float32),
        w2_fp8,  w2_scale.to(torch.float32),
        rw, int(ch), float(a), float(lim))             # [T, H] bf16
    return out
