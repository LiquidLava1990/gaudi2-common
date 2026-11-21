"""Graph-compiler-fusion routed-MoE spike — NO SDK, NO custom TPC kernel.

Goal: at batch=1 the proven `_routed_moe` (moe_wna16_hpu.py) is bandwidth-bound. It
dequantizes the <=K routed experts with `torch.ops.hpu.convert_from_uint4`, then
`htorch.core.mark_step()`, then hands the bf16 weight lists to the
`torch.ops.hpu.mixture_of_experts` intrinsic. That mandatory mark_step MATERIALIZES
every dequantized bf16 expert tile to HBM before the MoE kernel reads it back — for
GLM-5.1-int4 each routed expert is w13 [6144, 4096] + w2 [2048, 6144] bf16 ≈ 72 MiB,
so K=8 experts ≈ 576 MiB written + re-read PER TOKEN. That HBM round-trip IS the
single-stream floor (~99 ms/token class).

This spike removes the intrinsic from the routed leg and instead emits, per expert,
an explicit `convert_from_uint4 -> matmul` for gate/up and down. The ORIGINAL bet was
to also drop the mark_step between dequant and GEMM so Habana's graph compiler
(libTPCFuser / the GC fuser) could fuse `convert_from_uint4 -> matmul` into one op and
keep the freshly-dequantized bf16 tile resident in SRAM, collapsing the per-token HBM
traffic to ~the int4 weight read.

THAT BET FAILED ON HARDWARE (maxdiff ~182, rel ~0.80 vs the intrinsic reference):
`torch.ops.hpu.convert_from_uint4`'s output, when consumed from within the SAME
un-split lazy graph, is NOT realized when the consuming op reads it (it sees stale
weight bytes) -> structurally-wrong output. This is the documented `_routed_moe`
hazard (moe_wna16_hpu.py:580-595: "md=100..200"). So the dequant MUST be materialized
at a graph boundary (mark_step) before any GEMM reads it. The corrected spike keeps the
explicit per-expert GEMM structure but restores that mandatory mark_step after each
expert's dequants (see the per-expert loop). Net: this preserves correctness but the
deferred-fusion latency bet is not viable with convert_from_uint4 on this 1.24 build
(the bf16 weight still hits HBM at the materialization boundary).

Correctness now matches `_routed_moe` by construction: same `convert_from_uint4` (same
packed weights / scales / shared GPTQ-sym zero), same gate=w13[:, :I] / up=w13[:, I:]
split, same silu(gate)*up, same @w2, same router-weight scaling, same sum over top_k,
AND the same materializing mark_step between dequant and consume.

GLM-5.1-int4 (CORRECT shapes — earlier drafts wrongly used H=4096):
    H (hidden)            = 6144
    moe_intermediate (I)  = 2048
    w13 dequants to [H=6144, 2*I=4096]   (gate = [:, :2048], up = [:, 2048:])
    w2  dequants to [I=2048, H=6144]
    E (local experts)     = 32
    top_k                 = 8
    group_size            = 128
    GPTQ-symmetric, shared zero-point = 8 (uint4 bias)
    packed int32 weights  : w13p [E, H, 2I//8] = [E, 6144, 512]
                            w2p  [E, I, H//8]  = [E, 2048, 768]
    scales (op layout)    : s13 [E, H//gs, 2I] = [E, 48, 4096]
                            s2  [E, I//gs, H]  = [E, 16, 6144]

Lazy container (PT_HPU_LAZY_MODE=1), Habana 1.24.

This module is a SPIKE: `fused_routed_moe(...)` is a free function that mirrors
`MoeWNA16HPUMethod._routed_moe`'s gather/route/remap logic exactly (functional
scatter, the mandatory mark_step BEFORE the data-dependent index_select, per-slice
.clone() under lazy for the storage-offset bug, E-slot dummy bin, local routing),
then runs the explicit fused expert compute. It does NOT touch the production class.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import habana_frameworks.torch.core as htorch


def _dequant_expert(packed_slice, scale_slice, zero_point, dtype):
    """Dequant ONE expert's packed int32 weight -> dense bf16 [in, out].

    Thin wrapper over the SAME intrinsic the coherent reference op uses
    (extension_ops.py MoeWNA16Matmul.get_dequant_weight). g_idx=None: GLM-5.1
    is desc_act=False, contiguous groups, so the kernel derives the group from
    the row index. The shared GPTQ-sym zero_point (=8, packed) is reused across
    all experts (set once by the caller).
    """
    return torch.ops.hpu.convert_from_uint4(
        packed_slice, scale_slice, zero_point, dtype, None)


def fused_routed_moe(x, topk_ids, topk_weights,
                     w13_packed, w2_packed, w13_scales, w2_scales,
                     w13_zp, w2_zp, experts_min, num_local_experts, K,
                     intermediate_size, activation="silu"):
    """Spike: dequant+GEMM-fused routed MoE for the <=K locally-routed experts.

    Mirrors MoeWNA16HPUMethod._routed_moe's routing EXACTLY, then replaces the
    `mixture_of_experts` intrinsic with an explicit per-expert
    `convert_from_uint4 -> mark_step -> matmul` chain. The mark_step after the dequant
    is MANDATORY (materializes the convert_from_uint4 output before the GEMM reads it);
    omitting it produced maxdiff ~182 on HW (see module docstring + the loop comment).

    Args (all the stacked tensors the production path stashes on `layer`):
      x              : [T, H] bf16 hidden states (T tokens; this path is for T*top_k <= K)
      topk_ids       : [T, top_k] int — GLOBAL expert ids (this rank's window starts at experts_min)
      topk_weights   : [T, top_k] — router weights (already softmax/renormalized)
      w13_packed     : [E, H, 2I//8] int32 stacked packed gate+up weights
      w2_packed      : [E, I, H//8]  int32 stacked packed down weights
      w13_scales     : [E, H//gs, 2I] op-layout scales
      w2_scales      : [E, I//gs, H]  op-layout scales
      w13_zp, w2_zp  : packed shared GPTQ-sym zero-point (=8), [groups, out//8] (None => ineligible)
      experts_min    : this rank's first GLOBAL expert id (ep_rank * num_local_experts)
      num_local_experts (E) : local experts on this rank (32 for GLM-5.1)
      K              : static gather width (>= distinct local experts; caller guarantees T*top_k <= K)
      intermediate_size (I) : moe_intermediate (2048) — gate/up split point in the 2I output
      activation     : "silu" (the only GLM activation; explicit silu(gate)*up below)

    Returns: [T, H] bf16 routed output, or None if ineligible (AWQ per-expert zp).

    === ROUTING (verbatim mirror of _routed_moe; see that method's comments for the
        deep rationale on each lazy-HPU-graph hazard) ===
    """
    if w13_zp is None:
        # AWQ per-expert zero-point: the shared-zero fast path does not apply.
        return None
    if activation != "silu":
        # GLM-5.1 is silu-only; the explicit gate/up compute below hard-codes silu.
        raise ValueError(f"fused_routed_moe spike supports activation='silu', got {activation!r}")

    E = num_local_experts
    I = intermediate_size
    emin = experts_min
    dev = x.device
    dt = w13_scales.dtype

    # global expert id -> local [0, E)
    local = topk_ids.to(torch.int64) - emin
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)                      # routed to a LOCAL expert?
    flat_safe = torch.where(valid, flat, torch.zeros_like(flat))   # masked-safe gather index

    # Mark routed-LOCAL experts; non-local entries scatter to a DUMMY bin E (NOT index 0,
    # which would clobber a genuinely-routed local expert 0). FUNCTIONAL scatter (not
    # scatter_): out-of-place re-materializes every decode step -> graph-replay-safe.
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
        0, scatter_idx, valid.to(torch.float32))[:E]
    active = torch.topk(present, K).indices.to(torch.int64)        # [K] local ids (routed first)

    # *** MANDATORY mark_step #1 ***
    # MATERIALIZE the routing (present/scatter/topk -> `active`) BEFORE the data-dependent
    # index_select. HPU-graph capture mishandles index_select with un-materialized dynamic
    # indices (reads stale/garbage `active`). This is the ONE mark_step _routed_moe keeps
    # before the gather; we keep it identically. (It does NOT sit between dequant and GEMM,
    # so it does not block the fusion we are after.)
    htorch.mark_step()

    # gather the K routed experts' packed weights + scales
    w13p = w13_packed.index_select(0, active)             # [K, H, 2I//8]
    w2p = w2_packed.index_select(0, active)               # [K, I, H//8]
    s13 = w13_scales.index_select(0, active)              # [K, H//gs, 2I]
    s2 = w2_scales.index_select(0, active)                # [K, I//gs, H]

    # Under LAZY, per-slice views w13p[k]/s13[k] for k>0 carry a nonzero storage_offset;
    # convert_from_uint4 then reads the WRONG offset (expert active[0]'s bytes). Clone each
    # slice so the kernel sees storage_offset=0 — exactly the invariant the monolithic op
    # honours by .clone()-ing per-expert slices at load. Eager tolerates the offset.
    _need_clone = htorch.is_lazy()
    def _co(t):
        return t.clone() if _need_clone else t

    # remap routed global ids -> compact [0, K-1]; build per-token, per-slot routing.
    # FUNCTIONAL scatter (same lazy-replay safety as `present`).
    remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
        0, active, torch.arange(K, dtype=torch.int64, device=dev))      # local id -> compact k
    safe_w = torch.where(valid.view_as(topk_weights), topk_weights,
                         torch.zeros_like(topk_weights)).to(dt)          # [T, top_k]; non-local -> 0
    slot_k = torch.where(valid, remap[flat_safe],
                         torch.zeros_like(flat)).view_as(topk_ids)       # [T, top_k] -> compact k

    T = x.shape[0]
    xf = x.to(dt)

    # === EXPERT COMPUTE ===
    # For each of the K gathered experts: dequant w13/w2 with convert_from_uint4, MATERIALIZE
    # the dequanted bf16 weights at a graph boundary (mark_step), then run the explicit
    # gate/up + down GEMMs.
    #
    #   *** ROOT-CAUSE FIX (was: maxdiff~182, rel~0.80 vs the intrinsic reference) ***
    #   The earlier draft of this spike dequanted and IMMEDIATELY matmul'd with NO mark_step
    #   between, betting that the graph compiler would fuse `convert_from_uint4 -> matmul` and
    #   keep the bf16 weight tile in SRAM. On HW that produced STRUCTURALLY-WRONG output
    #   (maxdiff ~182). This is the EXACT failure the proven `_routed_moe` documents at
    #   moe_wna16_hpu.py:580-595: a `torch.ops.hpu.convert_from_uint4` output consumed from
    #   WITHIN the same un-split lazy graph is NOT realized when the consuming op reads it
    #   (the consumer sees stale/garbage weight bytes) -> "md=100..200". The fix there, and
    #   here, is MANDATORY: split the graph so the dequant is materialized before any consumer
    #   (the GEMM) reads it. The CPU proof (tests have no graph) showed the per-token MATH is
    #   bit-exact; the bug was purely the missing materialization boundary, not routing /
    #   gather / gate-up split / router-weight / accumulation.
    #
    #   The reference op's pattern is: dequant ALL expert weights -> ONE mark_step -> consume.
    #   We mirror it per expert: dequant w13[k] AND w2[k], mark_step (materialize both), then
    #   the two GEMMs. (Materializing the bf16 weight is the HBM round-trip this spike hoped to
    #   delete; the deferred-mark_step fusion bet is therefore NOT viable for correctness with
    #   convert_from_uint4 on this 1.24 build — the dequant must hit a graph boundary. Kept the
    #   explicit per-expert GEMM structure so the spike can still be A/B'd vs the intrinsic and
    #   so any future graph-mode that DOES allow un-materialized dequant can drop the mark_step.)
    #
    # Each token contributes to its routed experts via per-(token,slot) masks built from
    # slot_k/safe_w; an expert dequantized here serves whichever tokens routed to it. We
    # accumulate into `out` weighted by the router weight, summing over the top_k slots.
    out = torch.zeros(T, x.shape[-1], dtype=dt, device=dev)
    for k in range(K):
        # Dequant BOTH this expert's weights first.
        W13 = _dequant_expert(_co(w13p[k]), _co(s13[k]), w13_zp, dt)     # [H, 2I] bf16
        W2 = _dequant_expert(_co(w2p[k]), _co(s2[k]), w2_zp, dt)         # [I, H]  bf16
        # *** MANDATORY mark_step #2 ***: materialize the convert_from_uint4 outputs at a graph
        # boundary BEFORE the GEMMs read them. Without this the GEMMs read un-realized dequant
        # tensors -> maxdiff~182 (see the root-cause note above; identical to _routed_moe's
        # mark_step before mixture_of_experts).
        htorch.mark_step()

        gate_up = torch.matmul(xf, W13)                                 # [T, 2I]
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        h = F.silu(gate) * up                                           # [T, I]
        expert_out = torch.matmul(h, W2)                               # [T, H]

        # per-(token, slot) weight for THIS compact expert k: sum router weights of every
        # top_k slot that routed to k (0 where it didn't), giving a [T, 1] scale.
        w_k = torch.where(slot_k == k, safe_w, torch.zeros_like(safe_w)).sum(
            dim=-1, keepdim=True).to(dt)                                # [T, 1]
        out = out + expert_out * w_k

    return out


def reference_routed_moe(x, topk_ids, topk_weights,
                         w13_packed, w2_packed, w13_scales, w2_scales,
                         w13_zp, w2_zp, experts_min, num_local_experts, K,
                         intermediate_size, activation="silu"):
    """Coherent reference for the spike: SAME gather/route/remap, but the tail is the
    proven `mark_step + torch.ops.hpu.mixture_of_experts` intrinsic (exactly what
    `_routed_moe` runs). The microbench compares `fused_routed_moe` against this to (a)
    prove bit-equivalence (maxdiff < 1e-2) and (b) measure the latency delta that the
    deferred-mark_step fusion is supposed to buy. Mirrors _routed_moe lines verbatim."""
    if w13_zp is None:
        return None
    E = num_local_experts
    emin = experts_min
    dev = x.device
    dt = w13_scales.dtype

    local = topk_ids.to(torch.int64) - emin
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)
    flat_safe = torch.where(valid, flat, torch.zeros_like(flat))
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    present = torch.zeros(E + 1, dtype=torch.float32, device=dev).scatter(
        0, scatter_idx, valid.to(torch.float32))[:E]
    active = torch.topk(present, K).indices.to(torch.int64)
    htorch.mark_step()

    w13p = w13_packed.index_select(0, active)
    w2p = w2_packed.index_select(0, active)
    s13 = w13_scales.index_select(0, active)
    s2 = w2_scales.index_select(0, active)

    _need_clone = htorch.is_lazy()
    def _co(t):
        return t.clone() if _need_clone else t
    w13_deq = [torch.ops.hpu.convert_from_uint4(_co(w13p[k]), _co(s13[k]), w13_zp, dt, None) for k in range(K)]
    w2_deq = [torch.ops.hpu.convert_from_uint4(_co(w2p[k]), _co(s2[k]), w2_zp, dt, None) for k in range(K)]
    # The MANDATORY materializing mark_step the intrinsic requires (the HBM round-trip).
    htorch.mark_step()

    remap = torch.zeros(E, dtype=torch.int64, device=dev).scatter(
        0, active, torch.arange(K, dtype=torch.int64, device=dev))
    valid2d = valid.view_as(topk_ids)
    remapped = torch.where(valid, remap[flat_safe], torch.zeros_like(flat)).view_as(topk_ids)
    safe_w = torch.where(valid2d, topk_weights, torch.zeros_like(topk_weights)).to(dt)
    return torch.ops.hpu.mixture_of_experts(
        hidden_states=x, expert_routing_table=remapped, router_weights=safe_w,
        w12=w13_deq, w3=w2_deq, permuted_weights=False, activation=activation,
        experts_min=0, experts_max=K - 1)
