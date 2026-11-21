"""fused_moe_op.py -- load the C++ fused-expert bridge op and expose the
K-routed-expert `torch.ops.hpu_moe_ext.fused_moe` on top of it.

WRAP APPROACH: B. The C++ op (build/libfused_moe_op.so) registers
``hpu_moe_ext_fused::fused_expert`` -- ONE expert through the PROVEN fused int4
Synapse graph (fused_expert_graph.h), driven by direct synLaunch (approach A /
UserCustomOpDescriptor is single-GUID-only and cannot host the 2-MME + dequant +
slice + silu recipe -- see fused_moe_op.cpp header).

This module:
  * loads the .so (torch.ops.load_library),
  * re-exposes fused_expert under the task's namespace as
    ``torch.ops.hpu_moe_ext.fused_expert``,
  * registers the K-routed ``torch.ops.hpu_moe_ext.fused_moe`` whose impl reuses the
    PROVEN _routed_moe gather (present/scatter/topk -> active, index_select with
    per-slice clone, per-(token,slot) router weight) and loops fused_expert over the
    <=K gathered local experts, accumulating the weighted sum.

Signature (matches the vLLM MoE path; see hpu_moe_ext/moe_wna16_hpu.py _routed_moe):
    fused_moe(x[T,H] bf16, w13_packed[E,*] int32, w13_scale[E,*], w13_zp[E,*],
              w2_packed[E,*], w2_scale[E,*], w2_zp[E,*], topk_ids[T,K] int,
              topk_weights[T,K], experts_min:int, K:int=8) -> [T,H] bf16

Build (CPU-only):  bash build_op.sh   (inside vllm-gaudi:next)
Run (ON THE HPU BOX):  python test_fused_moe_op.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "build", "libfused_moe_op.so")

_LOADED = False


def load() -> None:
    """Load the compiled bridge op (idempotent). Imports habana_frameworks first so
    the HPU backend + device are registered before the op's HPU impl runs."""
    global _LOADED
    if _LOADED:
        return
    import habana_frameworks.torch  # noqa: F401  (registers HPU backend; acquires device)
    if not os.path.exists(_SO):
        raise FileNotFoundError(
            f"[fused_moe_op] not built: {_SO}\n"
            f"  Build it (CPU-only, inside vllm-gaudi:next): bash {os.path.join(_HERE, 'build_op.sh')}"
        )
    torch.ops.load_library(_SO)
    _register_python_ops()
    _LOADED = True


def _current_stream_handle() -> int:
    """The synapse stream handle of torch's current HPU compute stream, as an int.
    fused_expert launches on THIS stream so ordering vs torch ops is automatic.
    Falls back to 0 (the C++ op then uses its own cached stream) if unavailable."""
    try:
        import habana_frameworks.torch.hpu as hpu
        return int(hpu.current_stream().hpu_stream)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# K-routed fused_moe, built on the C++ fused_expert.  Registered as a real
# torch.library op under namespace `hpu_moe_ext` so callers use
# torch.ops.hpu_moe_ext.fused_moe(...) exactly as the task specifies.
# ---------------------------------------------------------------------------
def _sym_zero_packed_like(scale_groups: int, out_dim: int, device) -> torch.Tensor:
    """uint4 midpoint 8 packed 8/int32 along OUT -> [groups, out//8] int32.
    (Only used by the test/helpers; the real path passes the checkpoint's zp.)"""
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)  # 0x88888888
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((scale_groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _fused_moe_impl(
    x: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_zp: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_zp: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_min: int,
    K: int,
) -> torch.Tensor:
    """K-routed sum via the proven gather + per-expert fused graph launches.

    Weight layout (per expert e), matching the vLLM repack + convert_from_uint4 contract:
      w13_packed[e] int32 [H, 2I//8]   w13_scale[e] bf16 [H//GS, 2I]   w13_zp[e] int32 [H//GS, 2I//8]
      w2_packed[e]  int32 [I, H//8]    w2_scale[e]  bf16 [I//GS, H]     w2_zp[e]  int32 [I//GS, H//8]
    """
    import habana_frameworks.torch.core as htorch

    assert x.dim() == 2 and x.dtype == torch.bfloat16, "x must be bf16 [T,H]"
    T, H = x.shape
    E = w13_packed.shape[0]
    OUT13 = w13_scale.shape[-1]            # 2I
    assert OUT13 % 2 == 0
    I = OUT13 // 2
    dev = x.device
    dt = x.dtype

    # === ROUTING (verbatim mirror of _routed_moe / fused_routed_moe) ===
    local = topk_ids.to(torch.int64) - int(experts_min)        # global -> local [0,E)
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)
    flat_safe = torch.where(valid, flat, torch.zeros_like(flat))
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
        0, scatter_idx, valid.to(torch.float32))[:E]
    active = torch.topk(present, K).indices.to(torch.int64)     # [K] local ids (routed first)
    # MANDATORY mark_step #1: materialize `active` before the data-dependent gather.
    htorch.mark_step()

    w13p = w13_packed.index_select(0, active)                   # [K, H, 2I//8]
    w2p = w2_packed.index_select(0, active)                     # [K, I, H//8]
    s13 = w13_scale.index_select(0, active)                     # [K, H//GS, 2I]
    s2 = w2_scale.index_select(0, active)                       # [K, I//GS, H]
    # per-expert zero-point: shared (2D) or per-expert (3D, gathered like the weights).
    if w13_zp.dim() == 2:
        z13 = [w13_zp for _ in range(K)]
        z2 = [w2_zp for _ in range(K)]
    else:
        z13g = w13_zp.index_select(0, active)
        z2g = w2_zp.index_select(0, active)
        z13 = [z13g[k] for k in range(K)]
        z2 = [z2g[k] for k in range(K)]

    # remap routed local ids -> compact [0,K-1]; build per-(token,slot) compact id + safe weight.
    remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
        0, active, torch.arange(K, dtype=torch.int64, device=dev))
    safe_w = torch.where(valid.view_as(topk_weights), topk_weights,
                         torch.zeros_like(topk_weights)).to(dt)     # [T, top_k]
    slot_k = torch.where(valid, remap[flat_safe],
                         torch.zeros_like(flat)).view_as(topk_ids)  # [T, top_k] -> compact k

    # MANDATORY mark_step #2: materialize the gathered (cloned) per-expert tensors at a
    # graph boundary BEFORE their data_ptr()s are bound into the launched Synapse graph
    # (the same boundary the reference op / convert_from_uint4 path requires).
    htorch.mark_step()

    sh = _current_stream_handle()
    _lazy = htorch.is_lazy()

    def _co(t: torch.Tensor) -> torch.Tensor:
        # k>0 slices carry a nonzero storage_offset; the graph (like convert_from_uint4)
        # reads the wrong offset under lazy. Clone to a fresh storage_offset=0 buffer.
        return t.clone().contiguous() if _lazy else t.contiguous()

    out = torch.zeros(T, H, dtype=dt, device=dev)
    for k in range(K):
        # per-(token,slot) scalar weight for compact expert k -> [T,1]; here T==1 priority.
        w_k_vec = torch.where(slot_k == k, safe_w, torch.zeros_like(safe_w)).sum(
            dim=-1, keepdim=True).to(dt)                            # [T,1]
        # The fused graph applies a SINGLE [1,1] router scalar. For T==1 (priority single-
        # stream) pass that scalar; for T>1 the per-token weights differ, so apply the
        # graph with rw=1 and scale per-token outside (handled below).
        if T == 1:
            rw = w_k_vec.reshape(1, 1).contiguous()
            expert_out = torch.ops.hpu_moe_ext.fused_expert(
                x, _co(w13p[k]), _co(s13[k]), _co(z13[k]),
                _co(w2p[k]), _co(s2[k]), _co(z2[k]), rw, sh)        # [1,H] already *rw
            out = out + expert_out
        else:
            rw_one = torch.ones(1, 1, dtype=dt, device=dev)
            expert_out = torch.ops.hpu_moe_ext.fused_expert(
                x, _co(w13p[k]), _co(s13[k]), _co(z13[k]),
                _co(w2p[k]), _co(s2[k]), _co(z2[k]), rw_one, sh)    # [T,H] unscaled
            out = out + expert_out * w_k_vec
    if sh == 0:
        # fallback stream is not torch-ordered: force completion before returning.
        htorch.hpu.synchronize()
    return out


def _register_python_ops() -> None:
    """Register torch.ops.hpu_moe_ext.{fused_moe,fused_expert}. Idempotent-guarded by
    the caller (_LOADED). fused_expert simply forwards to the C++ op so a single
    namespace (hpu_moe_ext) exposes both, matching the task's call form."""
    lib = torch.library.Library("hpu_moe_ext", "DEF")
    lib.define(
        "fused_moe(Tensor x, Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
        "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, Tensor topk_ids, "
        "Tensor topk_weights, int experts_min, int K=8) -> Tensor")
    lib.define(
        "fused_expert(Tensor x, Tensor w13_packed, Tensor w13_scale, Tensor w13_zp, "
        "Tensor w2_packed, Tensor w2_scale, Tensor w2_zp, Tensor router_weight, "
        "int stream_handle) -> Tensor")
    lib.impl("fused_moe", _fused_moe_impl, "HPU")

    def _fe(x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
            router_weight, stream_handle):
        return torch.ops.hpu_moe_ext_fused.fused_expert(
            x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
            router_weight, stream_handle)

    lib.impl("fused_expert", _fe, "HPU")
    # keep the Library object alive for the process lifetime
    global _LIB
    _LIB = lib


_LIB = None


def fused_moe(x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
              topk_ids, topk_weights, experts_min, K: int = 8):
    """Convenience wrapper: ensures the op is loaded, then dispatches."""
    if not _LOADED:
        load()
    return torch.ops.hpu_moe_ext.fused_moe(
        x, w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp,
        topk_ids, topk_weights, int(experts_min), int(K))


if __name__ == "__main__":
    # Smoke: just load + print the registered schemas (no device math here).
    load()
    print("[fused_moe_op] loaded. registered ops:")
    print("  ", torch.ops.hpu_moe_ext_fused.fused_expert)
    print("  ", torch.ops.hpu_moe_ext.fused_moe)
    print("  ", torch.ops.hpu_moe_ext.fused_expert)
    sys.exit(0)
