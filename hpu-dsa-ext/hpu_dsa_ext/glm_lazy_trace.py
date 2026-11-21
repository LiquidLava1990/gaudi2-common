"""LOUD, capture-SAFE per-step / per-layer decode tracer for GLM-5.1 lazy serve.

WHY THIS EXISTS
===============
GLM-5.1 lazy serve (glm51ar-lazy: K=1 monolithic MoE, USE_DSA=1 dense-DSA-bypass,
HPU_GLM_ROPE_INLINE=1, PT_HPU_LAZY_MODE=1 + HPU-graph capture) emits CORRECT content
for ~8-10 decode tokens then accumulates error into greedy repetition. EAGER
(PT_HPU_LAZY_MODE=0) is fully coherent. So this is a LAZY-CAPTURE-SPECIFIC,
ACCUMULATING error. rope (inline cos/sin) and routed-MoE (K=1 = stock monolithic)
are already EXCLUDED. We need to see WHERE in the step the divergence first shows.

This tracer logs ONE WARNING line per decode step (first HPU_GLM_TRACE_STEPS steps,
default 16) of the first traced request, with:
  * step index, decode position (seq_len) and the decode BUCKET batch size in use
  * argmax output-token id + its logit value + the gap to the 2nd-best logit
    (confident vs flat -> "logits flattening" is the smoking gun for accumulation)
  * the final-hidden-state L2 norm (the model.norm output feeding the LM head)
  * for ~3 sampled layers: attention-output L2 norm + MoE/MLP-output L2 norm

============================  CAPTURE-SAFETY DESIGN  ===========================
The #1 constraint (operator directive): logging must NOT break the lazy capture.
A `.item()` / `.cpu()` / `.tolist()` / `torch.hpu.synchronize()` executed MID-FORWARD
forces an eager sync that re-materializes graph intermediates to HBM and MASKS the
very accumulation bug we are hunting. So: ZERO host reads inside the captured region.

How the wrap works (verified against the in-image bridge
habana_frameworks/torch/hpu/graphs.py and vllm_gaudi hpu_model_runner.py):
  * Only `HpuModelAdapter.forward` (the model body) is wrapped by
    `wrap_in_hpu_graph(..., disable_tensor_cache=True)`. `compute_logits`,
    `_run_sampling` (the sampler) and `AsyncHPUModelRunnerOutput.get_output`
    all run OUTSIDE the captured graph.
  * Under `disable_tensor_cache=True` the replay (`replayV3`) frees graph-internal
    intermediates after each replay (`free_inplace=True`) and refreshes only the
    registered graph OUTPUTS. A host read of any graph-internal tensor mid-forward
    is therefore both a sync (masks the bug) AND unsafe.

We pick a hybrid of the operator's options (a) device ring-buffer + (b) dump at a
point where a sync ALREADY happens + (c) @torch.compiler.disable host-side reader:

  (a) PRE-ALLOCATED DEVICE RING BUFFER, written by forward-hooks with `copy_`
      (a device->device write, NO host sync). The hooks compute each norm as a
      0-dim DEVICE tensor (`x.float().pow(2).sum()`) and `copy_` it into a FIXED
      slot of a persistent module-owned device buffer `_RING`. Because `_RING` is
      owned by Python (a stable attribute), its address is constant across replays,
      so the captured copy-node writes to the same address every replay -- exactly
      how vLLM writes the KV cache from inside the captured forward. We deliberately
      use FIXED per-(layer,metric) slots and rotate the "which step" ownership on
      the HOST (see below), so the captured graph never needs a dynamic index and
      never recaptures.

  (b) The ONLY host read (`_RING.cpu()`, plus the logits argmax/gap) happens in a
      hook on `AsyncHPUModelRunnerOutput.get_output`, which ALREADY calls
      `self._async_copy_ready_event.synchronize()` before reading sampled tokens.
      We piggy-back on that existing sync -- we add NO new sync to the decode loop.

  (c) That reader is `@torch.compiler.disable` and reads already-synced host values,
      so it can never be traced into a graph.

The logits (final-hidden L2, argmax id/logit/gap) are captured separately by a
@torch.compiler.disable hook on `compute_logits`, which runs OUTSIDE the graph: it
enqueues a single NON-blocking D2H copy of a tiny stats tensor into a pinned host
buffer and records nothing else -- no `.item()` there either; the host read of that
buffer also happens at the (b) sync point. Enqueuing a non_blocking D2H of an
OUT-OF-GRAPH tensor is the same pattern vllm-gaudi itself uses for the async
sampled-token copy (hpu_model_runner.py AsyncHPUModelRunnerOutput.__init__) and does
NOT force a host sync.

NET: not a single `.item()`/`.cpu()`/`synchronize()` is added inside the captured
`forward`; every host read rides the post-step sync that already exists. Capture is
preserved bit-for-bit; if the bug moves/disappears when tracing is on, that itself is
information -- but by construction it should not, because we add no in-forward sync.

GATING
======
Entirely behind env HPU_GLM_TRACE=1 (default OFF -> install() is a no-op, zero cost).
Knobs:
  HPU_GLM_TRACE=1                 enable (required)
  HPU_GLM_TRACE_STEPS=16          how many decode steps to log (per traced request)
  HPU_GLM_TRACE_LAYERS=3,38,72    which decoder layers to instrument (default spread)
  HPU_GLM_TRACE_REQS=1            how many separate requests to trace (resets per req)
  HPU_GLM_TRACE_RANK=0            only this TP rank logs (avoids an 8x flood); -1 = all

Search the logs for "[glm-trace]".

HOW TO READ THE OUTPUT  (localizing the bug)
============================================
Each line:
  [glm-trace] step=NN pos=PPP bucket_bs=BB | tok=ID logit=L.LLL gap=G.GGG | \
      h_norm=H.HH | L03 attn=A.AA moe=M.MM | L38 attn=.. moe=.. | L72 attn=.. moe=..

Read the COLUMNS down the first ~16 steps and find the FIRST step that breaks trend:
  * gap collapses toward ~0 (e.g. 6.0, 5.5, ... 0.3) while tok starts repeating
        => LOGITS FLATTENING. The distribution is going flat -> greedy picks a near-tie
           and locks into a loop. Pair it with h_norm to see if it is a norm drift.
  * h_norm drifts/explodes/decays monotonically across steps
        => FINAL HIDDEN-STATE drift. Something in the residual stream accumulates.
           Walk the per-layer columns to find WHERE it enters.
  * one layer's attn= norm JUMPS or DECAYS at the step coherence breaks (others steady)
        => that ATTENTION is the entry point (most likely: KV-cache read under
           lazy capture, or the dense-DSA-bypass MLA decode path mis-reading cache).
  * one layer's moe= norm jumps at that step (attn steady)
        => the MoE/MLP output is the entry point (monolithic K=1 dequant/op under
           capture). Since K=1==stock monolithic, a jump here would be capture-only.
  * the break STEP correlates with a bucket_bs change (e.g. BB goes 1->2 or 64->128,
    or pos crosses a decode-bucket boundary)
        => BUCKETING: a new decode bucket recaptured a graph that reads stale state
           (classic lazy multi-bucket trap; cf. the fused-MoE x/zp materialize fixes).
           The tell is that the break aligns with the FIRST step at a new bucket_bs.
  * EVERYTHING is steady through HPU_GLM_TRACE_STEPS but the *text* still garbles
        => the divergence is below this resolution (sub-norm, e.g. a few tokens'
           logit reorder); raise HPU_GLM_TRACE_LAYERS to all-layers and/or compare
           the SAME prompt's eager run (PT_HPU_LAZY_MODE=0) side by side -- the step
           where the two token streams first differ bounds it.

A healthy GLM-5.1 decode (hidden_size=6144): h_norm O(1e2-1e3), per-layer attn/moe
norms O(1e0-1e2), gap typically > ~1 on a confident greedy step. Watch for the FIRST
deviation, not the absolute values.
"""
from __future__ import annotations

import logging
import os
import threading

import torch

log = logging.getLogger("vllm")

# --------------------------------------------------------------------------- #
# Config (env)
# --------------------------------------------------------------------------- #
_ENABLED = os.environ.get("HPU_GLM_TRACE", "0") == "1"
_MAX_STEPS = int(os.environ.get("HPU_GLM_TRACE_STEPS", "16") or "16")
_MAX_REQS = int(os.environ.get("HPU_GLM_TRACE_REQS", "1") or "1")
_RANK = int(os.environ.get("HPU_GLM_TRACE_RANK", "0") or "0")  # -1 => all ranks


def _parse_layers(default=(3, 38, 72)):
    raw = os.environ.get("HPU_GLM_TRACE_LAYERS")
    if not raw:
        return list(default)
    out = []
    for s in raw.replace(" ", "").split(","):
        if s != "":
            try:
                out.append(int(s))
            except ValueError:
                pass
    return out or list(default)


_LAYERS = _parse_layers()

# Ring-buffer slot layout: 2 metrics (attn, moe) per traced layer + 1 final-hidden
# norm. Fixed, constant offsets so the captured copy-nodes always hit the same
# address (capture-safe; no dynamic index inside the graph).
_N_LAYER_METRICS = 2  # [attn_sq, moe_sq]
_HIDDEN_SLOT = len(_LAYERS) * _N_LAYER_METRICS  # final-hidden L2^2 lives here
_RING_LEN = _HIDDEN_SLOT + 1

# compute_logits stats buffer: [final_hidden_sq, top1_logit, top2_logit, top1_id].
_LOGIT_STATS_LEN = 4

# --------------------------------------------------------------------------- #
# State (host-side, never read inside the captured forward)
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_installed = False
_device = None             # torch.device('hpu')
_RING = None               # persistent DEVICE buffer, shape [_RING_LEN], float32
_RING_CPU = None           # pinned host mirror, filled by a non_blocking copy
_LOGIT_DEV = None          # persistent DEVICE buffer [_LOGIT_STATS_LEN]
_LOGIT_CPU = None          # pinned host mirror
_layer_slot = {}           # layer_idx -> base slot in _RING
_step = 0                  # host decode-step counter for the current request
_req = 0                   # how many requests traced so far
_runner = None             # HPUModelRunner ref (for warmup_mode + bucket bs)
_pending = False           # a logits snapshot was enqueued this step (decode only)
_last_decode_bs = 0        # padded decode bucket batch size for the current step
_last_pos = -1             # decode position (seq_len) for the current step
_POSLOG = [0]              # [poslog] diagnostic: reliably read position_ids at the call site (HPU_GLM_POSLOG)


def _rank_ok():
    if _RANK < 0:
        return True
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank() == _RANK
    except Exception:
        return True  # if rank can't be determined, don't suppress


def _is_warmup():
    r = _runner
    wm = getattr(r, "warmup_mode", None) if r is not None else None
    # warmup_mode True => skip; None (unknown) => treat as real (be loud, not silent)
    return wm is True


# --------------------------------------------------------------------------- #
# Lazy buffer allocation (once, on the first real forward, on the right device)
# --------------------------------------------------------------------------- #
@torch.compiler.disable
def _ensure_buffers(ref_tensor):
    global _RING, _RING_CPU, _LOGIT_DEV, _LOGIT_CPU, _device
    if _RING is not None:
        return
    _device = ref_tensor.device
    # CRITICAL: this runs lazily on the FIRST real forward, which vLLM executes under
    # torch.inference_mode(). A tensor allocated there is an "inference tensor", which
    # breaks this tracer two ways (the probe-tooling bug, NOT the model bug):
    #   1) The post-step `_RING_CPU.copy_(_RING)` in _drain runs OUTSIDE inference_mode
    #      and raises "Inplace update to inference tensor outside InferenceMode is not
    #      allowed." -> no trace line is ever produced.
    #   2) Under HPU-graph capture with disable_tensor_cache=True the inference-tagged
    #      device buffer is treated as a freed-after-replay intermediate, so the in-graph
    #      forward-hook `copy_` does not persist to the host read.
    # Wrapping the allocations in inference_mode(False) makes every buffer a NORMAL tensor
    # with a stable, Python-owned address -- the same property that lets vLLM's KV cache
    # buffers be written from inside the captured forward and survive replay. The hooks'
    # device->device `copy_` and the host-side inplace copy_ are then both legal.
    with torch.inference_mode(False):
        _RING = torch.zeros(_RING_LEN, dtype=torch.float32, device=_device)
        _LOGIT_DEV = torch.zeros(_LOGIT_STATS_LEN, dtype=torch.float32, device=_device)
        # Pinned host mirrors so the post-step copy is a cheap, non-allocating D2H.
        try:
            _RING_CPU = torch.zeros(_RING_LEN, dtype=torch.float32, device="cpu").pin_memory()
            _LOGIT_CPU = torch.zeros(_LOGIT_STATS_LEN, dtype=torch.float32, device="cpu").pin_memory()
        except Exception:
            _RING_CPU = torch.zeros(_RING_LEN, dtype=torch.float32, device="cpu")
            _LOGIT_CPU = torch.zeros(_LOGIT_STATS_LEN, dtype=torch.float32, device="cpu")
    # Belt-and-suspenders: guarantee none are inference-tagged (a stale inference flag would
    # re-break the inplace copy_ in _drain). Re-materialize as normal if so.
    if torch.is_inference(_RING):
        with torch.inference_mode(False):
            _RING = _RING.detach().clone()
            _LOGIT_DEV = _LOGIT_DEV.detach().clone()
            _RING_CPU = _RING_CPU.detach().clone()
            _LOGIT_CPU = _LOGIT_CPU.detach().clone()


# --------------------------------------------------------------------------- #
# Forward hooks (run INSIDE the captured graph -- device-only, NO host sync)
# --------------------------------------------------------------------------- #
def _make_layer_metric_hook(slot):
    """Return a forward_hook that writes ||output||^2 into _RING[slot] in-place.

    Device->device copy_ only: enqueues a reduction + a copy into a fixed,
    Python-owned address. No `.item()`, no `.cpu()`, no synchronize. Safe under
    disable_tensor_cache because _RING's storage is stable across replays.
    """

    def hook(module, inputs, output):
        if _RING is None:
            return output
        t = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(t):
            return output
        # squared L2 as a 0-dim device tensor; copy_ into the fixed slot.
        sq = t.detach().float().pow(2).sum()
        _RING[slot].copy_(sq)
        return output

    return hook


def _hidden_norm_hook(module, inputs, output):
    """Capture the FINAL hidden-state (model.norm output) squared-L2 into _RING.

    This is the last RMSNorm before the LM head -- its drift is the cleanest
    single signal of residual-stream accumulation. Device-only write.
    """
    if _RING is None:
        return output
    t = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(t):
        return output
    _RING[_HIDDEN_SLOT].copy_(t.detach().float().pow(2).sum())
    return output


# --------------------------------------------------------------------------- #
# compute_logits wrapper (runs OUTSIDE the graph -> safe to enqueue a D2H copy)
# --------------------------------------------------------------------------- #
@torch.compiler.disable
def _snapshot_logits(logits, hidden_states):
    """Enqueue a tiny stats tensor (top1/top2 logit, top1 id, final-hidden sq)
    into the pinned host mirror via ONE non_blocking D2H. No host read here."""
    global _pending
    try:
        if logits is None or logits.numel() == 0:
            return
        _ensure_buffers(logits)
        # logits: [bs, vocab]; row 0 is the (first) decode sequence we trace.
        row = logits.detach().float()
        if row.dim() == 2:
            row = row[0]
        top2 = torch.topk(row, 2)
        top1_logit = top2.values[0]
        top2_logit = top2.values[1]
        top1_id = top2.indices[0].to(torch.float32)
        if hidden_states is not None and torch.is_tensor(hidden_states):
            hs = hidden_states.detach().float()
            if hs.dim() == 2:
                hs = hs[0]
            hsq = hs.pow(2).sum()
        else:
            hsq = torch.zeros((), device=row.device)
        # pack into the persistent device stats buffer (device-only writes)...
        _LOGIT_DEV[0].copy_(hsq)
        _LOGIT_DEV[1].copy_(top1_logit)
        _LOGIT_DEV[2].copy_(top2_logit)
        _LOGIT_DEV[3].copy_(top1_id)
        # ...then enqueue ONE non_blocking D2H into the pinned mirror. This op is
        # OUTSIDE the captured forward (compute_logits is not graph-wrapped); it
        # does not force a host sync (no .item()). The host read happens at the
        # post-step sync point (_drain).
        _LOGIT_CPU.copy_(_LOGIT_DEV, non_blocking=True)
        _pending = True
    except Exception as ex:
        log.warning("[glm-trace] _snapshot_logits error: %s", ex)


# --------------------------------------------------------------------------- #
# Post-step drain (runs at AsyncHPUModelRunnerOutput.get_output -> sync already
# happened there). Reads ALREADY-SYNCED host values and emits the loud line.
# --------------------------------------------------------------------------- #
@torch.compiler.disable
def _drain():
    global _step, _req, _pending
    if not _pending:
        return
    _pending = False
    with _lock:
        if _req >= _MAX_REQS or _step >= _MAX_STEPS:
            return
        step = _step
        _step += 1

    try:
        # Enqueue the ring D2H now (still no forced sync; the caller's
        # synchronize() already fired before get_output reads tokens, so by the
        # time we .tolist() below the copies are complete). To be safe we issue a
        # lightweight synchronize ONLY here, at the post-step boundary that is
        # ALREADY a sync point -- it costs nothing extra to the decode graph.
        #
        # Robust readback (no inplace-on-inference): the host mirrors are normal tensors
        # (allocated under inference_mode(False) in _ensure_buffers), so the inplace
        # `copy_` below is legal. As a hard guard against any future inference-tagging we
        # do the host-side work inside inference_mode(False) and, if the inplace copy_ ever
        # raises, fall back to a FRESH detach().clone() D2H (which yields a normal tensor)
        # so a single bad flag can never silence the whole tracer again.
        with torch.inference_mode(False):
            if _RING is not None:
                try:
                    _RING_CPU.copy_(_RING, non_blocking=True)
                    _LOGIT_CPU.copy_(_LOGIT_DEV, non_blocking=True)
                    ring_host, logit_host = _RING_CPU, _LOGIT_CPU
                except RuntimeError:
                    # inference-tensor inplace guard tripped -> clone to a normal host tensor.
                    ring_host = _RING.detach().clone().to("cpu", non_blocking=True)
                    logit_host = _LOGIT_DEV.detach().clone().to("cpu", non_blocking=True)
            else:
                ring_host, logit_host = _RING_CPU, _LOGIT_CPU
            try:
                torch.hpu.synchronize()
            except Exception:
                pass

            ring = ring_host.tolist() if ring_host is not None else [0.0] * _RING_LEN
            lg = logit_host.tolist() if logit_host is not None else [0.0] * _LOGIT_STATS_LEN

        h_sq, top1_logit, top2_logit, top1_id = lg
        # Prefer the final-hidden norm from the ring hook (model.norm output); fall
        # back to the compute_logits hidden_states sq if the hook didn't fire.
        h_norm_sq = ring[_HIDDEN_SLOT] if ring[_HIDDEN_SLOT] > 0 else h_sq
        h_norm = h_norm_sq ** 0.5
        gap = float(top1_logit) - float(top2_logit)

        parts = [
            f"[glm-trace] step={step:02d} pos={_last_pos} bucket_bs={_last_decode_bs}",
            f"tok={int(top1_id)} logit={float(top1_logit):.3f} gap={gap:.3f}",
            f"h_norm={h_norm:.3f}",
        ]
        for li, lidx in enumerate(_LAYERS):
            base = li * _N_LAYER_METRICS
            attn = ring[base] ** 0.5
            moe = ring[base + 1] ** 0.5
            parts.append(f"L{lidx:02d} attn={attn:.3f} moe={moe:.3f}")
        log.warning(" | ".join(parts))

        if _step >= _MAX_STEPS:
            with _lock:
                _req += 1
                _step = 0
            log.warning(
                "[glm-trace] ===== request #%d traced (%d steps). %s =====",
                _req, _MAX_STEPS,
                "Read the columns: first step where gap->0 / a layer norm jumps / "
                "bucket_bs changes localizes the divergence." if _req < _MAX_REQS
                else "Tracing budget exhausted (HPU_GLM_TRACE_REQS); set higher to trace more.",
            )
    except Exception as ex:
        log.warning("[glm-trace] _drain error: %s", ex)


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def install() -> None:
    """Install the lazy decode tracer. Gated behind HPU_GLM_TRACE=1. Idempotent."""
    global _installed
    if not _ENABLED:
        return
    with _lock:
        if _installed:
            return
        _installed = True

    log.warning(
        "[glm-trace] installing capture-safe lazy decode tracer "
        "(steps=%d reqs=%d layers=%s rank=%s). Search logs for '[glm-trace]'.",
        _MAX_STEPS, _MAX_REQS, _LAYERS, _RANK if _RANK >= 0 else "all",
    )

    _hook_model_runner()
    _hook_layers()
    _hook_compute_logits()
    _hook_async_output()


def _hook_model_runner():
    """Grab the HPUModelRunner ref (warmup gate) and capture the decode bucket bs
    + position from the captured decode call site -- both are read OUTSIDE the
    graph (in _execute_model_generic, before/after self.model.forward) so they are
    sync-free host values."""
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-trace] cannot import HPUModelRunner: %s", ex)
        return

    if not getattr(HPUModelRunner.__init__, "_glm_trace", False):
        _orig_init = HPUModelRunner.__init__

        def __init__(self, *a, **k):
            _orig_init(self, *a, **k)
            global _runner
            _runner = self

        __init__._glm_trace = True
        HPUModelRunner.__init__ = __init__

    # Capture decode bucket bs + position at _execute_model_generic entry. This is
    # the runner method that calls self.model.forward; token_ids.size(0) is the
    # padded decode bucket batch size and position_ids[...] is the decode position.
    # Reading .size(0) is a host shape read (free); we read positions[0] lazily as
    # a host int only when it is already a small CPU-derivable scalar -> to avoid a
    # sync we instead record the max position from attn_metadata if present, else
    # fall back to the step counter.
    if not getattr(HPUModelRunner._execute_model_generic, "_glm_trace", False):
        _orig_gen = HPUModelRunner._execute_model_generic

        def _gen(self, token_ids, position_ids, attn_metadata, *a, **k):
            global _last_decode_bs, _last_pos
            try:
                if not _is_warmup():
                    # decode forward => seq dim is 1; record padded batch (bucket) size.
                    is_decode = (hasattr(token_ids, "shape")
                                 and (token_ids.dim() == 1 or token_ids.size(-1) == 1))
                    if is_decode:
                        _last_decode_bs = int(token_ids.size(0))
                        # position: prefer a host-side scalar already on CPU; never
                        # force a sync. attn_metadata may expose a python seq_len.
                        sl = None
                        for attr in ("seq_len", "context_lens", "num_blocks"):
                            v = getattr(attn_metadata, attr, None)
                            if callable(v):
                                try:
                                    v = v()
                                except Exception:
                                    v = None
                            if isinstance(v, int):
                                sl = v
                                break
                        _last_pos = sl if sl is not None else _last_pos
                        # [poslog] DIRECT read of the real positions input (at the call
                        # site, BEFORE the wrapped/captured forward -> reliable; the sync
                        # is acceptable for this diagnostic). Tells us definitively whether
                        # decode positions are FROZEN (stuck) or LIVE (growing).
                        import os as _os
                        if _os.environ.get("HPU_GLM_POSLOG", "0") == "1" and _POSLOG[0] < 14:
                            try:
                                _pid = position_ids.flatten()
                                log.warning(f"[poslog] decode#{_POSLOG[0]:02d} position_ids[:6]={_pid[:6].tolist()} numel={_pid.numel()} seq_meta={sl}")
                            except Exception as _e:
                                log.warning(f"[poslog] err {_e!r}")
                            _POSLOG[0] += 1
            except Exception:
                pass
            return _orig_gen(self, token_ids, position_ids, attn_metadata, *a, **k)

        _gen._glm_trace = True
        HPUModelRunner._execute_model_generic = _gen


def _hook_layers():
    """Register forward hooks on the chosen decoder layers' self_attn and mlp
    submodules, and on the final model.norm. Walk module names so we work whether
    the model is model.model.layers or language_model.model.layers."""
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-trace] cannot import DeepseekV2DecoderLayer: %s", ex)
        return

    # Hook at the CLASS level via __init__ wrap so every instance (across all 78
    # layers) self-registers; we filter by layer_idx in __init__ post-hook. This
    # avoids needing the live model tree at install time.
    if getattr(DeepseekV2DecoderLayer.__init__, "_glm_trace", False):
        return
    _orig_init = DeepseekV2DecoderLayer.__init__

    def __init__(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            lidx = getattr(self, "layer_idx", None)
            if lidx is None or lidx not in _LAYERS:
                return
            with _lock:
                if lidx not in _layer_slot:
                    _layer_slot[lidx] = _LAYERS.index(lidx) * _N_LAYER_METRICS
                base = _layer_slot[lidx]
            attn = getattr(self, "self_attn", None)
            mlp = getattr(self, "mlp", None)
            if attn is not None:
                attn.register_forward_hook(_make_layer_metric_hook(base))
            if mlp is not None:
                mlp.register_forward_hook(_make_layer_metric_hook(base + 1))
            log.warning("[glm-trace] hooked layer %d (slot base=%d) self_attn+mlp", lidx, base)
        except Exception as ex:  # noqa: BLE001
            log.warning("[glm-trace] layer %s hook error: %s",
                        getattr(self, "layer_idx", "?"), ex)

    __init__._glm_trace = True
    DeepseekV2DecoderLayer.__init__ = __init__

    # Final-hidden norm: hook model.norm once the model exists. We attach it via the
    # model's forward wrapper (model.model.norm) at first real forward through the
    # compute_logits hook (which receives the final hidden state) -- so the
    # _HIDDEN_SLOT is filled from compute_logits' hidden_states there as a fallback.
    # We additionally try to hook the RMSNorm named 'norm' on the top-level model.
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
        if not getattr(DeepseekV2Model.__init__, "_glm_trace_norm", False):
            _orig_minit = DeepseekV2Model.__init__

            def _minit(self, *a, **k):
                _orig_minit(self, *a, **k)
                try:
                    nrm = getattr(self, "norm", None)
                    if nrm is not None:
                        nrm.register_forward_hook(_hidden_norm_hook)
                        log.warning("[glm-trace] hooked final model.norm (hidden-state L2)")
                except Exception:
                    pass

            _minit._glm_trace_norm = True
            DeepseekV2Model.__init__ = _minit
    except Exception:
        pass


def _hook_compute_logits():
    """Wrap GlmMoeDsaForCausalLM.compute_logits (runs OUTSIDE the graph) to snapshot
    the logits argmax/gap + final-hidden norm via one non_blocking D2H."""
    targets = []
    try:
        from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
        targets.append(GlmMoeDsaForCausalLM)
    except Exception:
        pass
    # Fall back to the generic DeepseekV2 causal LM if the GLM alias isn't present.
    if not targets:
        try:
            from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
            targets.append(DeepseekV2ForCausalLM)
        except Exception as ex:  # noqa: BLE001
            log.warning("[glm-trace] cannot import a causal-LM class for compute_logits: %s", ex)
            return

    for cls in targets:
        if getattr(cls.compute_logits, "_glm_trace", False):
            continue
        _orig = cls.compute_logits

        def _wrapped(self, hidden_states, *a, __orig=_orig, **k):
            out = __orig(self, hidden_states, *a, **k)
            try:
                if _ENABLED and _rank_ok() and not _is_warmup():
                    _snapshot_logits(out, hidden_states)
            except Exception:
                pass
            return out

        _wrapped._glm_trace = True
        cls.compute_logits = _wrapped
        log.warning("[glm-trace] wrapped %s.compute_logits", cls.__name__)


def _hook_async_output():
    """Wrap AsyncHPUModelRunnerOutput.get_output (the existing post-step sync point)
    so we drain + log AFTER its synchronize(), reading only already-synced values."""
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import AsyncHPUModelRunnerOutput
    except Exception as ex:  # noqa: BLE001
        log.warning("[glm-trace] cannot import AsyncHPUModelRunnerOutput "
                    "(async scheduling expected for lazy serve): %s", ex)
        # Fall back: drain at the end of sample_tokens (non-async path).
        _hook_sample_tokens_fallback()
        return

    if getattr(AsyncHPUModelRunnerOutput.get_output, "_glm_trace", False):
        return
    _orig = AsyncHPUModelRunnerOutput.get_output

    def get_output(self):
        out = _orig(self)  # this calls synchronize() then reads tokens
        try:
            if _ENABLED and _rank_ok():
                _drain()
        except Exception:
            pass
        return out

    get_output._glm_trace = True
    AsyncHPUModelRunnerOutput.get_output = get_output
    log.warning("[glm-trace] wrapped AsyncHPUModelRunnerOutput.get_output (post-step drain)")


def _hook_sample_tokens_fallback():
    """Non-async fallback: drain at the end of HPUModelRunner.sample_tokens, which
    on the non-async path already does a .cpu().tolist() sync of sampled tokens."""
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except Exception:
        return
    if getattr(HPUModelRunner.sample_tokens, "_glm_trace", False):
        return
    _orig = HPUModelRunner.sample_tokens

    def sample_tokens(self, grammar_output):
        out = _orig(self, grammar_output)
        try:
            if _ENABLED and _rank_ok() and not _is_warmup():
                _drain()
        except Exception:
            pass
        return out

    sample_tokens._glm_trace = True
    HPUModelRunner.sample_tokens = sample_tokens
    log.warning("[glm-trace] wrapped HPUModelRunner.sample_tokens (non-async drain fallback)")
