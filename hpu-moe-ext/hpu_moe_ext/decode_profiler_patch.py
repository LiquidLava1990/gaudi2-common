"""Inject torch.profiler around the DECODE FORWARD to get the per-step DECODE BUDGET split
(launch-gap vs sync-stall vs dequant vs attention vs MoE GATHER). Gated by HPU_PROFILE_DECODE=1;
rank-0 only; armed by touching <HPU_PROFILE_DIR>/START after warmup; captures HPU_PROFILE_N forwards
(or until <HPU_PROFILE_DIR>/STOP appears, or HPU_PROFILE_MAX_S seconds elapse — whichever first);
exports a Kineto chrome trace glm_decode.json; self-disables. NO-OP unless the env is set. Never
raises into the serve. Analyze with bench/gather_budget.py (buckets gather/index_select/scatter vs
matmul/dequant/attention).

WHY THIS REWRITE (the c64 failure):
  * The runner is SPLIT on the V1 engine: execute_model() only does _update_states + stashes
    scheduler_output and returns None (NO forward); the whole decode forward + MoE + sampler runs in
    sample_tokens(). torch.profiler is a PROCESS-WIDE collector, so the OLD wrap on execute_model
    still captured kernels (proven by the 92MB+ leakmap/batchtest traces) — wrapping the right method
    only fixes STEP-COUNT alignment, but we move to sample_tokens anyway so step==forward.
  * The real c64 symptom was: STARTED logged but EXPORTED never did — the auto count-to-N never
    tripped the export under the async batch-queue serve (a 30-step run had grown a 1.6GB trace,
    i.e. the profiler had been running for thousands of steps, not 30). Root hardening here:
      (a) a STOP sentinel — the orchestrator can FORCE the flush regardless of the auto-counter;
      (b) a wall-clock cap (HPU_PROFILE_MAX_S, default 25s) so it can NEVER run away again;
      (c) export is wrapped (sync the device first so async-dispatched kernels land in the trace,
          then never raise into the serve);
      (d) the STARTED/heartbeat logs print rank + step so a live run is debuggable.
  Native alternative (VLLM_TORCH_PROFILER_DIR + POST /start_profile): the /start_profile route is
  gated behind a populated ProfilerConfig (entrypoints/serve/profile/api_router.py:attach_router),
  NOT by VLLM_TORCH_PROFILER_DIR alone, so on this build the endpoint 404s unless a --profiler config
  is passed. This sentinel path needs no endpoint and is the reliable capture.
"""
import os
import sys
import time

_p = {"prof": None, "step": 0, "done": False, "t_start": 0.0}


def install() -> None:
    if (os.environ.get("HPU_PROFILE_DECODE") or "0") not in ("1", "true", "True"):
        return
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
        import torch  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        print(f"[decode-prof] runner import failed ({ex!r}) — skipped", file=sys.stderr, flush=True)
        return
    # Wrap sample_tokens (the forward). Only fall back to execute_model on a pre-split runner.
    target = "sample_tokens" if hasattr(HPUModelRunner, "sample_tokens") else "execute_model"
    if target == "execute_model":
        print("[decode-prof] WARNING: runner has no sample_tokens — wrapping execute_model "
              "(pre-split runner)", file=sys.stderr, flush=True)
    orig = getattr(HPUModelRunner, target)
    if getattr(orig, "_decode_prof", False):
        return
    N = int(os.environ.get("HPU_PROFILE_N", "40"))
    D = os.environ.get("HPU_PROFILE_DIR", "/recipe_cache/glm_decode_prof")
    MAX_S = float(os.environ.get("HPU_PROFILE_MAX_S", "25"))
    sentinel = os.path.join(D, "START")
    stop_sentinel = os.path.join(D, "STOP")

    def _export(reason):
        """Stop the profiler, flush the device, then emit the PER-OP DEVICE-TIME breakdown — the
        robust capture. export_chrome_trace silently produces no file on the c64 async/HPU path
        (confirmed 2026-06-23: 40 forwards profiled, no json), so we do NOT depend on it: we read the
        per-op aggregation straight out of the profiler.

        Outputs (rank-0, <HPU_PROFILE_DIR>/):
          - ops.json : [{name, self_device_us, total_device_us, count, self_cpu_us}, ...] sorted by
                       self_device_us desc, plus a summary header. Off-card, bucket name ->
                       gather/dequant/gemm/attention and sum self_device_us (see gather_budget_ops.py).
          - the key_averages().table() printed to the log for an at-a-glance view.
          - glm_decode.json STILL attempted (best effort); the swallowed export exception is PRINTED.

        DEVICE-TIME SOURCE (verified card-free on this image):
          PRIMARY = the low-level kineto result events (p.profiler.kineto_results.events()): each
          _KinetoEvent exposes name()/device_type()/duration_ns() as METHODS; device-kernel events
          carry device_type()==DeviceType.HPU (or PrivateUse1) and duration_ns() = the on-device
          kernel time. This is the SAME data that fills the chrome trace's cat:"kernel" rows (proven
          to populate — the prior 92MB-1.6GB traces), read IN-MEMORY so it does not depend on the
          export_chrome_trace file write that fails on the c64/async/HPU path.
          SECONDARY = FunctionEventAvg.self_device_time_total from key_averages() (device-neutral
          torch-2.10 attr; the cuda_* names are MISSING on FunctionEventAvg here). Used only if the
          kineto path yields no device time. Never raises into the serve. Called exactly once."""
        p = _p["prof"]
        if p is None:
            return
        n = _p["step"]
        dt = time.time() - _p["t_start"]

        def _ev_call(e, m, default=None):
            a = getattr(e, m, None)
            if a is None:
                return default
            try:
                return a() if callable(a) else a
            except Exception:  # noqa: BLE001
                return default

        try:
            import torch
            try:
                if hasattr(torch, "hpu") and torch.hpu.is_available():
                    torch.hpu.synchronize()
            except Exception:  # noqa: BLE001
                pass
            p.stop()
            os.makedirs(D, exist_ok=True)

            rows = []
            src = "kineto"
            # ---- PRIMARY: device kernels from the kineto result events (HPU/PrivateUse1) ----
            try:
                kr = getattr(p.profiler, "kineto_results", None)
                kevs = kr.events() if kr is not None else []
                agg = {}
                for e in kevs:
                    dtype = str(_ev_call(e, "device_type", ""))
                    # keep ONLY on-device kernel events (exclude CPU host ops)
                    if not ("HPU" in dtype or "PrivateUse1" in dtype or "CUDA" in dtype):
                        continue
                    nm = str(_ev_call(e, "name", "?"))
                    dur_us = float(_ev_call(e, "duration_ns", 0) or 0) / 1000.0
                    a = agg.setdefault(nm, {"name": nm, "self_device_us": 0.0,
                                            "total_device_us": 0.0, "count": 0, "self_cpu_us": 0.0})
                    a["self_device_us"] += dur_us
                    a["total_device_us"] += dur_us
                    a["count"] += 1
                rows = list(agg.values())
            except Exception as ex:  # noqa: BLE001
                print(f"[decode-prof] kineto_results aggregation failed ({ex!r})", file=sys.stderr, flush=True)

            tot_self_dev = sum(r["self_device_us"] for r in rows)
            # ---- SECONDARY: key_averages() device-time if kineto gave nothing ----
            if tot_self_dev <= 0.0:
                src = "key_averages"
                rows = []
                try:
                    for e in p.key_averages():
                        sd = float(getattr(e, "self_device_time_total", 0) or 0)
                        td = float(getattr(e, "device_time_total", 0) or 0)
                        rows.append({"name": str(getattr(e, "key", "?")), "self_device_us": sd,
                                     "total_device_us": td, "count": int(getattr(e, "count", 0) or 0),
                                     "self_cpu_us": float(getattr(e, "self_cpu_time_total", 0) or 0)})
                    tot_self_dev = sum(r["self_device_us"] for r in rows)
                except Exception as ex:  # noqa: BLE001
                    print(f"[decode-prof] key_averages() failed ({ex!r})", file=sys.stderr, flush=True)
            # ---- TERTIARY: if STILL no device time, emit CPU-time rows so the op list is not lost ----
            if tot_self_dev <= 0.0 and not rows:
                src = "key_averages(cpu-only)"
                try:
                    for e in p.key_averages():
                        rows.append({"name": str(getattr(e, "key", "?")), "self_device_us": 0.0,
                                     "total_device_us": 0.0, "count": int(getattr(e, "count", 0) or 0),
                                     "self_cpu_us": float(getattr(e, "self_cpu_time_total", 0) or 0)})
                except Exception:  # noqa: BLE001
                    pass

            rows.sort(key=lambda r: (r["self_device_us"], r["self_cpu_us"]), reverse=True)
            ops_out = os.path.join(D, "ops.json")
            import json
            with open(ops_out, "w") as f:
                json.dump({"reason": reason, "forwards": n, "elapsed_s": dt, "source": src,
                           "total_self_device_us": tot_self_dev,
                           "total_self_device_ms_per_fwd": (tot_self_dev / 1000.0 / max(1, n)),
                           "ops": rows}, f, indent=1)
            print(f"[decode-prof] OPS.JSON written {ops_out} — {len(rows)} ops, src={src}, "
                  f"total self-device={tot_self_dev/1000.0:.1f}ms over {n} forwards "
                  f"= {tot_self_dev/1000.0/max(1,n):.2f} ms/fwd", file=sys.stderr, flush=True)

            # ---- print the table to the log (device-time sorted) ----
            try:
                tbl = p.key_averages().table(sort_by="self_device_time_total", row_limit=60)
                print("[decode-prof] ===== key_averages (sort=self_device_time_total) =====\n" + tbl,
                      file=sys.stderr, flush=True)
            except Exception as ex:  # noqa: BLE001
                # last-resort: cpu sort still shows the op list
                try:
                    tbl = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=60)
                    print(f"[decode-prof] (device sort failed {ex!r}; cpu-sorted table)\n" + tbl,
                          file=sys.stderr, flush=True)
                except Exception as ex2:  # noqa: BLE001
                    print(f"[decode-prof] table() failed ({ex2!r})", file=sys.stderr, flush=True)

            # ---- best-effort chrome trace; PRINT why it fails (don't swallow silently) ----
            try:
                out = os.path.join(D, "glm_decode.json")
                p.export_chrome_trace(out)
                sz = os.path.getsize(out) if os.path.exists(out) else -1
                if sz > 0:
                    print(f"[decode-prof] EXPORTED {out} ({sz} bytes)", file=sys.stderr, flush=True)
                else:
                    print(f"[decode-prof] export_chrome_trace returned but no/empty file at {out} "
                          f"(size={sz}) — use ops.json", file=sys.stderr, flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[decode-prof] export_chrome_trace FAILED ({type(ex).__name__}: {ex}) "
                      f"— ops.json is the breakdown", file=sys.stderr, flush=True)

            print(f"[decode-prof] DONE ({reason}, {n} forwards, {dt:.1f}s)", file=sys.stderr, flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[decode-prof] EXPORT BLOCK FAILED ({ex!r})", file=sys.stderr, flush=True)
        finally:
            _p["prof"] = None
            _p["done"] = True
            for s in (sentinel, stop_sentinel):
                try:
                    os.remove(s)
                except Exception:  # noqa: BLE001
                    pass

    def wrapped(self, *a, **k):
        if not _p["done"]:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                rank = get_tensor_model_parallel_rank()
            except Exception:  # noqa: BLE001
                rank = 0
            r0 = (rank == 0)
            if r0 and _p["prof"] is None and os.path.exists(sentinel):
                import torch
                _p["prof"] = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                    record_shapes=False, with_stack=False)
                _p["prof"].start()
                _p["step"] = 0
                _p["t_start"] = time.time()
                print(f"[decode-prof] STARTED (sentinel seen, rank={rank}) — profiling {target}, "
                      f"N={N}, cap={MAX_S}s", file=sys.stderr, flush=True)
            if _p["prof"] is not None:
                _p["step"] += 1
                # Heartbeat so a stuck run is visible in the log.
                if _p["step"] % 10 == 0:
                    print(f"[decode-prof] ... {_p['step']}/{N} forwards profiled "
                          f"({time.time() - _p['t_start']:.1f}s)", file=sys.stderr, flush=True)
                # Export when ANY stop condition trips: count reached, STOP sentinel, or wall-clock cap.
                if _p["step"] >= N:
                    _export("count")
                elif os.path.exists(stop_sentinel):
                    _export("STOP-sentinel")
                elif (time.time() - _p["t_start"]) >= MAX_S:
                    _export("time-cap")
        return orig(self, *a, **k)

    wrapped._decode_prof = True
    setattr(HPUModelRunner, target, wrapped)
    print(f"[decode-prof] {target} wrapped (HPU_PROFILE_DECODE=1, N={N}, cap={MAX_S}s, dir={D})",
          file=sys.stderr, flush=True)


_st = {"prev_exit": None, "durs": [], "gaps": [], "n": 0}

# --- HOST-TAIL SPLIT (C40_TAIL_PROFILE) — NO-OP unless HPU_STEP_SPLIT=1 -----------------------------
# Per-call wall time of the sub-phases that compose the decode step, so the step-timer's
# exec_med/gap_med can be attributed. Each wrapped method appends its own per-call duration to a
# bounded deque; the sample_tokens window-print emits the median of the last N for each. Pure timing,
# never raises, runs in EVERY worker process (rank 0 = the driver that also samples; ranks 1-7 forward
# only) so the per-rank lines expose the driver-vs-worker asymmetry that localizes the tail.
#   _execute_model_generic : forward dispatch + compute_logits (incl. the TP vocab logits gather)
#   _run_sampling          : the sampler (softmax/top-k/top-p over [B, vocab])
#   execute_model          : the SPLIT-runner's pre-step (_update_states + _prepare_inputs); runs in
#                            the inter-forward GAP, so its median tells how much of gap_med is worker
#                            prep vs pure idle-wait for the engine-core process to send the next RPC.
import collections as _collections

_sp = {"durs": {}, "names": ("_execute_model_generic", "_run_sampling", "execute_model")}


def _install_split_subwraps(cls, N):
    import time as _time
    for nm in _sp["names"]:
        orig = getattr(cls, nm, None)
        if orig is None or getattr(orig, "_st_sub", False):
            continue
        dq = _sp["durs"].setdefault(nm, _collections.deque(maxlen=8 * N))

        def _mk(orig, dq):
            def w(self, *a, **k):
                t = _time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    dq.append(_time.perf_counter() - t)
            w._st_sub = True
            return w
        setattr(cls, nm, _mk(orig, dq))
    print(f"[step-split] sub-wraps installed on {', '.join(_sp['names'])} (HPU_STEP_SPLIT=1)",
          file=sys.stderr, flush=True)


def _split_report(N, _S):
    parts = []
    for nm in _sp["names"]:
        dq = _sp["durs"].get(nm)
        if dq:
            vals = list(dq)[-N:]
            parts.append(f"{nm.lstrip('_')}={1000*_S.median(vals):.1f}ms(n{len(vals)})")
        else:
            parts.append(f"{nm.lstrip('_')}=--")
    return "  ".join(parts)


def install_step_timer() -> None:
    """Lightweight always-safe forward-cadence timer. NO-OP unless HPU_STEP_TIMER=1.
    Logs, per window of N sample_tokens calls: median forward duration, median inter-forward host
    gap, implied forward cadence. Pure timing, never raises.

    NOTE: exec_med is HOST wall time around sample_tokens (dispatch + whatever sync happens inside),
    NOT pure device-busy. Under --async-scheduling the device runs ahead, so a tiny exec_med at a
    large step time = host-launch-bound; the HPU_PROFILE_DECODE device trace is the source of truth
    for device-busy-vs-idle, and is what decides the gather-pole question.

    HPU_STEP_SPLIT=1 additionally attributes exec_med/gap_med to forward(+logits)/sampler/prep —
    see the C40_TAIL_PROFILE block above."""
    if (os.environ.get("HPU_STEP_TIMER") or "0") not in ("1", "true", "True"):
        return
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
        import time as _time
        import statistics as _S
    except Exception as ex:  # noqa: BLE001
        print(f"[step-timer] import failed ({ex!r}) — skipped", file=sys.stderr, flush=True)
        return
    target = "sample_tokens" if hasattr(HPUModelRunner, "sample_tokens") else "execute_model"
    orig = getattr(HPUModelRunner, target)
    if getattr(orig, "_step_timer", False):
        return
    N = int(os.environ.get("HPU_STEP_TIMER_N", "100"))
    split_on = (os.environ.get("HPU_STEP_SPLIT") or "0") in ("1", "true", "True")
    if split_on:
        try:
            _install_split_subwraps(HPUModelRunner, N)
        except Exception as ex:  # noqa: BLE001
            print(f"[step-split] sub-wrap install failed ({ex!r}) — split disabled", file=sys.stderr, flush=True)
            split_on = False

    def wrapped(self, *a, **k):
        t0 = _time.perf_counter()
        if _st["prev_exit"] is not None:
            _st["gaps"].append(t0 - _st["prev_exit"])
        try:
            r = orig(self, *a, **k)
        finally:
            t1 = _time.perf_counter()
            _st["durs"].append(t1 - t0)
            _st["prev_exit"] = t1
            _st["n"] += 1
            if _st["n"] % N == 0:
                d = _st["durs"][-N:]
                g = _st["gaps"][-N:] or [0.0]
                md, mg = _S.median(d), _S.median(g)
                print(f"[step-timer] n={_st['n']} exec_med={1000*md:.1f}ms gap_med={1000*mg:.1f}ms "
                      f"step={1000*(md+mg):.1f}ms fwd/s={1/(md+mg):.2f}", file=sys.stderr, flush=True)
                if split_on:
                    print(f"[step-split] n={_st['n']} {_split_report(N, _S)}", file=sys.stderr, flush=True)
        return r

    wrapped._step_timer = True
    setattr(HPUModelRunner, target, wrapped)
    print(f"[step-timer] installed on {target} (HPU_STEP_TIMER=1)", file=sys.stderr, flush=True)


# --- ENGINE-CORE STEP SPLIT (attribute the inter-step "gap") — NO-OP unless HPU_ENGINE_STEP_SPLIT=1 -
# WHY: the worker-side step-timer above measures the ~90ms inter-step gap_med at concurrency but does
# NOT attribute it — the gap is the time the WORKERS idle while the EngineCore process runs its step
# loop (schedule -> dispatch -> block-on-output -> update_from_output). This probe runs IN the
# EngineCore process (general_plugins load there too) and splits that loop, so we can tell whether the
# 90ms is RECLAIMABLE host-scheduling (sched+update big -> overlap it) or an UNRECLAIMABLE device-wait
# (the block-on-output 'wait' term big -> the workers really are the bottleneck; nothing to reclaim).
# This is the decisive discriminator the c40/roadmap analysis was missing (it only had runner-side
# HPU_STEP_SPLIT, which lives INSIDE sample_tokens, not in the gap). Pure perf_counter timing,
# bounded deques, never raises, byte-identical to stock when the env is unset.
_es = {"n": 0, "phases": {}, "wrapped_inst": False}


def _es_acc(name, dt):
    _es["phases"].setdefault(name, []).append(dt)


def _es_wrap_instance(self) -> None:
    """Lazily wrap the EngineCore's scheduler/executor sub-calls on the live instances (idempotent)."""
    import time as _time
    if _es["wrapped_inst"]:
        return

    def _mk(obj, attr, label):
        try:
            orig = getattr(obj, attr)
        except Exception:  # noqa: BLE001
            return
        if orig is None or getattr(orig, "_es_wrapped", False):
            return

        def w(*a, **k):
            t = _time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                _es_acc(label, _time.perf_counter() - t)

        w._es_wrapped = True
        try:
            setattr(obj, attr, w)
        except Exception:  # noqa: BLE001
            pass

    sched = getattr(self, "scheduler", None)
    ex = getattr(self, "model_executor", None)
    if sched is not None:
        _mk(sched, "schedule", "schedule")
        _mk(sched, "update_from_output", "update_from_output")
    if ex is not None:
        # non_block=True dispatch calls — these time the HOST dispatch only (device runs async).
        _mk(ex, "execute_model", "execute_dispatch")
        _mk(ex, "sample_tokens", "sample_dispatch")
    _es["wrapped_inst"] = True


def install_engine_step_split() -> None:
    """Attribute the inter-step gap inside the EngineCore step loop. NO-OP unless
    HPU_ENGINE_STEP_SPLIT=1. Logs, per window of N engine steps, the median wall of each sub-phase
    (schedule / execute_dispatch / sample_dispatch / update_from_output) and 'wait_other' = the
    remainder of the whole step (= block-on-output future + IPC). If wait_other dominates -> the gap
    is device-bound (workers are the wall; not reclaimable). If schedule+update dominate -> the gap is
    host-scheduler-bound (reclaimable by overlap)."""
    if (os.environ.get("HPU_ENGINE_STEP_SPLIT") or "0") not in ("1", "true", "True"):
        return
    try:
        from vllm.v1.engine.core import EngineCore
        import time as _time
        import statistics as _S
    except Exception as ex:  # noqa: BLE001
        print(f"[engine-step-split] import failed ({ex!r}) — skipped", file=sys.stderr, flush=True)
        return
    N = int(os.environ.get("HPU_ENGINE_STEP_SPLIT_N", "200"))
    # Wrap whichever step entrypoints exist (batch-queue path under async-scheduling, else plain step).
    for tname in ("step_with_batch_queue", "step"):
        orig = getattr(EngineCore, tname, None)
        if orig is None or getattr(orig, "_es_step", False):
            continue

        def _mk_step(orig, tname):
            def wrapped(self, *a, **k):
                _es_wrap_instance(self)
                t0 = _time.perf_counter()
                try:
                    return orig(self, *a, **k)
                finally:
                    total = _time.perf_counter() - t0
                    _es_acc("_step_total", total)
                    _es["n"] += 1
                    if _es["n"] % N == 0:
                        ph = _es["phases"]
                        def med(key):
                            v = ph.get(key, [])[-N:]
                            return 1000.0 * _S.median(v) if v else 0.0
                        tot = med("_step_total")
                        sub = med("schedule") + med("execute_dispatch") + med("sample_dispatch") + med("update_from_output")
                        wait = max(0.0, tot - sub)
                        print(
                            f"[engine-step-split] via={tname} n={_es['n']} "
                            f"step={tot:.1f}ms | schedule={med('schedule'):.1f} "
                            f"exec_disp={med('execute_dispatch'):.1f} samp_disp={med('sample_dispatch'):.1f} "
                            f"update={med('update_from_output'):.1f} wait_other={wait:.1f}ms",
                            file=sys.stderr, flush=True)
                        # keep deques bounded
                        for kk in list(ph.keys()):
                            ph[kk] = ph[kk][-N:]
            return wrapped

        w = _mk_step(orig, tname)
        w._es_step = True
        setattr(EngineCore, tname, w)
        print(f"[engine-step-split] installed on EngineCore.{tname} (HPU_ENGINE_STEP_SPLIT=1)",
              file=sys.stderr, flush=True)
