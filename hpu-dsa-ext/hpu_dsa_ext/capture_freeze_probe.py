"""Capture-freeze probe — confirm WHETHER the lazy GLM decode loop is a graph-capture
position/rope freeze (the cos/sin-class bug) vs a downstream numerical drift, WITHOUT
inserting a host sync into the captured decode graph (a sync would itself break capture
and MASK the very freeze we are hunting — see glm_rope_neox.py (B) root-cause note).

WHY A SPECIAL PROBE (the existing real_request_probe cannot catch this):
  real_request_probe logs norms via @torch.compiler.disable + .float()/.item(), which
  forces a graph break / host readback INSIDE the model body. Under lazy HPU-graph
  capture with disable_tensor_cache=True that (a) perturbs the very capture topology
  under test and (b) reads the LIVE value, not the REPLAYED value — so a tensor that is
  FROZEN on replay but recomputed-correct on the eager probe path looks fine. To observe
  the replay value we must read a tensor the captured graph WROTE, and read it only AFTER
  the step boundary (when the graph has already replayed + the stream has synced
  naturally), never mid-graph.

MECHANISM (deferred, one-step-late, zero in-graph sync):
  We tap the MLA rotary forward (HPURotaryEmbedding.forward_oot — the SAME instance the
  inline glm_rope_neox (B) fix rewrites) and the decode forward_impl. Each decode step,
  INSIDE the captured region, we only do cheap device-side reductions and `index_copy_`
  them into a pre-allocated device ring buffer (no .item(), no .cpu(), no print) keyed by
  a step counter that is itself a host int (the Python call count — fixed per replay, so
  it does not vary the graph). At the START of the NEXT step's Python call (OUTSIDE the
  prior captured graph, after it has executed) we copy the ring slot for the PRIOR step
  to host and log it. Thus every readback happens after a natural step-boundary sync and
  never appears inside a captured graph.

WHAT IT PROVES:
  For consecutive decode steps t, t+1, … it logs three device-computed scalars per step:
    pos_sum   = float(positions.sum())          — the decode query absolute position(s)
    cos_chk   = (cos * _w).sum()                 — a position-sensitive checksum of the
                                                   cos/sin the rope ACTUALLY applied this
                                                   step (read from the captured tensor)
    kpe_chk   = (k_pe_out * _w2).sum()           — checksum of the rope'd KEY that gets
                                                   written into the latent KV cache
    attn_chk  = (attn_out * _w3).sum()           — checksum of the MLA decode output
  Verdict (read straight off the log):
    * pos_sum ADVANCES but cos_chk is CONSTANT  → the rope position FROZE on replay (the
      cos/sin-class capture freeze). The fix is to make the gather live (glm_rope_neox
      (B) inline forward_oot) — and if (B) is already on, the frozen tap localizes WHICH
      rope instance/op is still split (e.g. an UNPATCHED RotaryEmbedding SUBCLASS, or the
      adapter pre-call buffer winning over the inline path).
    * cos_chk ADVANCES (rope live) but attn_chk drifts toward a fixed point while pos
      advances → the freeze is NOT in rope; it is a downstream captured MLA op reading a
      stale buffer (kpe_chk constant ⇒ the cache WRITE froze; attn_chk constant w/ kpe
      advancing ⇒ the cache READ / block index froze).
  Either way the FIRST constant-while-pos-advances scalar names the frozen stage.

GATED, CPU-safe import: does nothing unless HPU_CAPTURE_PROBE=N (N decode steps to log).
Importing this module never touches HPU. Read-only; never changes numerics.
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("vllm")

# Number of consecutive REAL decode steps to trace. 0 = disabled.
_PROBE_N = int(os.environ.get("HPU_CAPTURE_PROBE", "0") or "0")
# Ring depth (steps kept on device before host readback). Small; readback is 1 step late.
_RING = max(8, _PROBE_N + 2)

_lock = threading.Lock()
_installed = False
_state = {
    "ring": None,        # device tensor [_RING, 4]  (pos_sum, cos_chk, kpe_chk, attn_chk)
    "w_cos": None,       # device weight vectors for the checksums (fixed, position-stable)
    "w_kpe": None,
    "w_attn": None,
    "step": 0,           # host decode-step counter (advances once per forward_impl decode)
    "logged": 0,
    "last_slot_read": -1,
    "mr": None,          # model-runner ref (warmup gate)
}


def _is_real() -> bool:
    mr = _state["mr"]
    if mr is not None:
        try:
            return not bool(getattr(mr, "warmup_mode", False))
        except Exception:  # noqa: BLE001
            return True
    return True


def _alloc_normal(shape, *, device, dtype, fill=None, lin=None):
    """Allocate a persistent, module-owned tensor that is GUARANTEED to be a NORMAL
    (non-inference) tensor, even though this runs inside vLLM's torch.inference_mode()
    (model load + every forward run under inference_mode).

    WHY THIS MATTERS (the probe-tooling bug this fixes):
      A tensor allocated while inference_mode is active is an "inference tensor". Two
      things then break the probe (NOT the model):
        1) Inplace updates to it OUTSIDE inference_mode (our host-side drain/read at the
           post-step boundary) raise "Inplace update to inference tensor ... not allowed".
        2) Under HPU-graph capture with disable_tensor_cache=True, the inference tensor's
           storage is treated as a freed-after-replay graph intermediate, so the device
           `copy_`/index-assign the captured forward writes into it does NOT persist to
           the host-read view -> we read uninitialized memory (the nan / 0.0 symptom).
      Wrapping the allocation in `inference_mode(False)` makes it a normal tensor with a
      stable, Python-owned storage address -- exactly how vLLM's KV cache buffers are
      normal persistent tensors whose in-graph `copy_` writes survive replay.
    """
    import torch
    with torch.inference_mode(False):
        if lin is not None:
            lo, hi, n = lin
            t = torch.linspace(lo, hi, n, device=device, dtype=dtype)
        elif fill is not None:
            t = torch.full(shape, fill, device=device, dtype=dtype)
        else:
            t = torch.zeros(shape, device=device, dtype=dtype)
    # Belt-and-suspenders: if some future torch still tags it (shouldn't), detach+clone
    # outside inference_mode yields a normal tensor.
    if torch.is_inference(t):
        with torch.inference_mode(False):
            t = t.detach().clone()
    return t


def _ensure_ring(device, dtype):
    import torch
    if _state["ring"] is None:
        # NORMAL (non-inference) device tensor: written by the captured forward via
        # index-assign into FIXED slots; read 1 step late at the post-step sync. Stable
        # address => the captured copy-node hits the same storage every replay (persists).
        _state["ring"] = _alloc_normal((_RING, 4), device=device, dtype=torch.float32)


def _drain_prior_step():
    """Read back + log the ring slot for the PRIOR decode step. Called at the START of a
    Python forward_impl call, i.e. AFTER the prior step's captured graph already executed
    and the step boundary synced — so the .cpu() here is never inside a captured graph."""
    import torch
    ring = _state["ring"]
    if ring is None:
        return
    prior = _state["step"] - 1
    if prior < 0 or prior == _state["last_slot_read"]:
        return
    if _state["logged"] >= _PROBE_N:
        return
    slot = prior % _RING
    try:
        # Clone to a FRESH host tensor (no inplace into any pre-existing buffer) so the
        # readback can never trip the inference-tensor inplace rule, then .tolist() the
        # already-synced host copy. This runs at the post-step boundary (the prior step's
        # captured graph has executed + the stream synced), so the D2H is never in-graph.
        with torch.inference_mode(False):
            row = ring[slot].detach().clone().to("cpu", non_blocking=False).tolist()
        _state["last_slot_read"] = prior
        _state["logged"] += 1
        log.warning(
            "[capture-probe] decode step=%d  pos_sum=%.4f  cos_chk=%.6f  kpe_chk=%.6f  "
            "attn_chk=%.6f   (constant-while-pos-advances => that stage FROZE on replay)",
            prior, row[0], row[1], row[2], row[3],
        )
    except Exception as ex:  # noqa: BLE001
        log.warning("[capture-probe] drain error step=%d: %s", prior, ex)


def install(model_runner=None) -> bool:
    global _installed
    if _PROBE_N <= 0:
        return False
    with _lock:
        if _installed:
            if model_runner is not None:
                _state["mr"] = model_runner
            return False
        _installed = True
        if model_runner is not None:
            _state["mr"] = model_runner

    import torch

    # --- Tap 1: MLA rotary forward_oot (records pos_sum + cos_chk + kpe_chk) ----------
    # We wrap whatever forward_oot is currently installed (glm_rope_neox (B) inline, or the
    # stock one), so the tap measures the ACTUAL rope op on the live serving path.
    try:
        from vllm_gaudi.ops.hpu_rotary_embedding import HPURotaryEmbedding
    except Exception as ex:  # noqa: BLE001
        log.warning("[capture-probe] cannot import HPURotaryEmbedding: %s", ex)
        return False

    _cur_rope_fwd = HPURotaryEmbedding.forward_oot

    def _probed_rope_forward_oot(self, positions, query, key, offsets=None):
        q_out, k_out = _cur_rope_fwd(self, positions, query, key, offsets)
        # The rope tap OWNS the decode step counter: rope is the FIRST per-step op (applied
        # before attention), so advancing here keeps every later tap (forward_impl) on the
        # SAME slot for the step. We gate to the MLA decode query rope by a small last-dim
        # (qk_rope_head_dim) AND a small token count (decode: 1 query/seq). Cheap device
        # reductions only; NO host sync (a sync would mask the freeze under test).
        try:
            if _PROBE_N <= 0 or not _is_real():
                return q_out, k_out
            # Decode-shape gate: rope'd query is [num_tokens, n_head, rope_dim]; in decode
            # num_tokens == batch_size (1 per seq). Skip prefill (large num_tokens) and any
            # non-MLA rope. We can't read shapes' VALUES cheaply, but ndim/last-dim are static.
            ntok = positions.reshape(-1).shape[0]
            if ntok > 64:            # decode batch is small; prefill prompt is >64 tokens
                return q_out, k_out
            import torch as _t
            # New model-pass detection (layer-name-agnostic): when a rope module instance we
            # already saw this step recurs, the next decode step has begun -> advance counter.
            seen = _state.setdefault("seen_rope", set())
            key_id = id(self)
            if key_id in seen:
                _state["step"] += 1
                seen.clear()
                _state.setdefault("seen_layers", set()).clear()  # keep forward_impl tap in sync
            seen.add(key_id)
            _ensure_ring(q_out.device, q_out.dtype)
            ring = _state["ring"]
            slot = _state["step"] % _RING
            if _state["w_cos"] is None:
                qn = q_out.reshape(-1).shape[0]
                kn = k_out.reshape(-1).shape[0]
                # NORMAL (non-inference) weight vectors: they are READ (multiplied) inside
                # the captured forward, so under disable_tensor_cache an inference-tagged
                # weight could be freed/garbage on replay too -> allocate them normal so the
                # checksum operands persist across replays just like the ring.
                _state["w_cos"] = _alloc_normal((qn,), device=q_out.device, dtype=_t.float32,
                                                lin=(1.0, 2.0, qn))
                _state["w_kpe"] = _alloc_normal((kn,), device=k_out.device, dtype=_t.float32,
                                                lin=(1.0, 3.0, kn))
            # cos_chk: checksum of the rope'd QUERY (carries the applied cos/sin angle). Frozen
            # position => identical angle => (for an identical pre-rope q) identical checksum.
            # kpe_chk: checksum of the rope'd KEY that is written into the latent KV cache —
            # the cleanest signal for a cache-write position freeze.
            pos_sum = positions.reshape(-1).to(_t.float32).sum()
            cos_chk = (q_out.reshape(-1).to(_t.float32) * _state["w_cos"]).sum()
            kpe_chk = (k_out.reshape(-1).to(_t.float32) * _state["w_kpe"]).sum()
            # Write only columns 0..2 (col 3 is owned by the forward_impl attn tap). Use an
            # in-place copy_ into FIXED slots of the persistent device ring -- this is the
            # SAME write mechanism vLLM uses to fill the KV cache from inside the captured
            # forward, so the captured copy-node targets a stable address and PERSISTS across
            # replays. Device->device only; no .item()/.cpu()/sync (would mask the freeze).
            ring[slot, 0].copy_(pos_sum)
            ring[slot, 1].copy_(cos_chk)
            ring[slot, 2].copy_(kpe_chk)
        except Exception:  # noqa: BLE001  (never break the model forward)
            pass
        return q_out, k_out

    _probed_rope_forward_oot._capture_probe = True
    HPURotaryEmbedding.forward_oot = _probed_rope_forward_oot

    # --- Tap 2: decode forward_impl (advances the step counter, records attn_chk,
    #            and drains the prior step's slot at the OUTSIDE-graph Python entry) ----
    try:
        from vllm_gaudi.attention.oot_mla import HPUMLAAttention
    except Exception as ex:  # noqa: BLE001
        log.warning("[capture-probe] cannot import HPUMLAAttention: %s", ex)
        return False

    _cur_fwd = HPUMLAAttention.forward_impl

    def _probed_forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                             output=None, output_scale=None, output_block_scale=None):
        is_decode = (attn_metadata is not None
                     and getattr(attn_metadata, "is_prompt", True) is False)
        # Drain the prior decode step OUTSIDE the captured graph (host readback safe here).
        if is_decode and _PROBE_N > 0 and _is_real():
            _drain_prior_step()
        out = _cur_fwd(self, q, k_c_normed, k_pe, kv_cache, attn_metadata,
                       output, output_scale, output_block_scale)
        # Record this step's attn-output checksum into ring col 3 (device-only). The step
        # counter is owned by the rope tap (rope runs before attention each step), so here we
        # only ACCUMULATE the per-layer attn output into the current step's slot — summing over
        # layers gives a whole-step attention-state signal. The slot is whatever step the rope
        # tap last set, which for this same decode pass is correct (rope layer-0 already ran).
        try:
            if is_decode and _PROBE_N > 0 and _is_real():
                import torch as _t
                ln = getattr(self, "layer_name", "?")
                seen = _state.setdefault("seen_layers", set())
                first_in_step = ln not in seen
                seen.add(ln)
                _ensure_ring(out.device, out.dtype)
                ring = _state["ring"]
                slot = _state["step"] % _RING
                if _state["w_attn"] is None:
                    on = out.reshape(-1).shape[0]
                    # NORMAL (non-inference) weight vector (read inside the captured forward).
                    _state["w_attn"] = _alloc_normal((on,), device=out.device, dtype=_t.float32,
                                                     lin=(1.0, 5.0, on))
                achk = (out.reshape(-1).to(_t.float32) * _state["w_attn"]).sum()
                # In-place copy_ into the FIXED col-3 slot (persists across replay; see the
                # rope tap). First layer of the step overwrites; later layers accumulate
                # (read the device value, add, copy back -- all device-side, no host sync).
                new_val = achk if first_in_step else (ring[slot, 3] + achk)
                ring[slot, 3].copy_(new_val)
        except Exception:  # noqa: BLE001
            pass
        return out

    _probed_forward_impl._capture_probe = True
    HPUMLAAttention.forward_impl = _probed_forward_impl

    log.warning(
        "[capture-probe] installed (HPU_CAPTURE_PROBE=%d). Traces %d real decode steps; "
        "deferred 1-step-late readback => NO in-graph sync. Read the log: the FIRST scalar "
        "that stays CONSTANT while pos_sum ADVANCES names the stage that froze on replay "
        "(cos/kpe=rope+cache-write, attn=cache-read/block-index).", _PROBE_N, _PROBE_N)
    return True


def register() -> None:
    """Entry point (mirrors real_request_probe.register). NO-OP unless HPU_CAPTURE_PROBE>0.
    Hooks HPUModelRunner.__init__ to auto-inject the runner for the warmup gate."""
    if _PROBE_N <= 0:
        return
    install(model_runner=None)
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except Exception as ex:  # noqa: BLE001
        log.warning("[capture-probe] cannot hook HPUModelRunner.__init__: %s", ex)
        return
    if getattr(HPUModelRunner.__init__, "_capture_probe_hooked", False):
        return
    _orig = HPUModelRunner.__init__

    def _hooked_init(self, *a, **k):
        _orig(self, *a, **k)
        with _lock:
            _state["mr"] = self

    _hooked_init._capture_probe_hooked = True
    HPUModelRunner.__init__ = _hooked_init
