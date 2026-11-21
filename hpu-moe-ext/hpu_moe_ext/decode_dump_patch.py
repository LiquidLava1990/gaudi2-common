"""MTP training-data dump (EAGLE-3 head factory): capture (final hidden, top-K target logits,
position, seq_id, EAGLE-3 low/mid/high aux residuals) per decode step during generation -> shards.
The factory trainer consumes these. Gated HPU_MTP_DUMP=1. Rank-0 only. Writes a shard every
HPU_MTP_DUMP_SHARD positions. Never raises into the serve. Run a code+reasoning+tool-use corpus
through the serve to fill it.

Patches whichever target model class is importable: GLM-5.1/5.2 (deepseek_v2.DeepseekV2ForCausalLM
& GlmMoeDsaForCausalLM) + Kimi-K2.7 (DeepseekV3ForCausalLM) AND MiniMax-M3
(minimax_m3.HpuMiniMaxM3ForConditionalGeneration) — so the SAME factory pipeline dumps either model.

──────────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS REWRITE (3rd aux attempt; mechanism PROVEN against the on-card shards):
The model is `htorch.hpu.wrap_in_hpu_graph(..., disable_tensor_cache=True)` (hpu_model_runner ~L855,
L5112). With disable_tensor_cache=True the graph captures the wrapped module's RETURN value as the
ONLY user outputs (graphs.py extract_tensors+mark_user_outputs ~L722-725); every OTHER intermediate
tensor is emptied/frozen after replay (graphs.py Note L673-674), and replay hands back the SAME
captured output objects (graphs.py L765-770).

The OLD dump made the model's patched forward return `out[0]` (bare hidden) and snapshotted the aux
via an IN-TRACE `.cpu()`. Because aux was NOT in the returned structure it was never a marked user
output -> frozen after replay -> the `.cpu()` read 8 stale per-seq-slot buffers (== max-num-seqs):
aux L2 frozen at ~460 while hidden varied; only ~15 unique aux across 101 shards (2.4%). NOT a wall.

THE FIX (mirror the in-tree native EAGLE3 aux path, which works precisely BECAUSE aux is a marked
graph output): make aux a registered graph output so disable_tensor_cache keeps it LIVE and replay
updates it per step, then tee it from the runner's LIVE return, materialized AFTER a mark_step
(outside the captured-graph region) and gathered per-token by the SAME logits_indices the runner
applied to the hidden. Concretely:
  1. forward(): set model.aux_hidden_state_layers=(2,n//2,n-3) (EAGLE3 (low,mid,high)) AND return the
     FULL (hidden, aux_list) tuple — so HpuModelAdapter.forward returns it -> extract_tensors marks
     all aux tensors as user outputs (recurses lists) -> they stay live across replay.
  2. runner.__init__ hook: set use_aux_hidden_state_outputs=True so _execute_model_generic unpacks the
     model's (hidden, aux) tuple (hpu_model_runner L3769-3774) and RETURNS aux as the 2nd of its
     4-tuple (non_flattened_hidden, aux, sample_hidden, logits) at L3784. This is the native eagle3
     code path; with NO speculative_config the runner never calls the drafter (propose is gated on
     `if self.speculative_config:` L4978), so aux is simply teed by us and otherwise unused — safe.
  3. _execute_model_generic wrapper (_emg): the SINGLE capture site. After the original returns, do
     htorch.core.mark_step() (flush replay so the marked aux/hidden hold the REAL per-token values),
     then capture sample_hidden (already gathered to logits_indices rows by L3777) + logits + aux
     (gathered by the same logits_indices), gated to pure-decode steps. Materialize .float().cpu()
     OUTSIDE the graph region.

No in-trace .cpu(); aux is sourced from the LIVE/marked runner output, never the frozen buffer."""
import os
import sys

_S = {"buf": [], "n": 0, "shard": 0}
_RUNNER = {"mr": None}            # set by the HPUModelRunner.__init__ hook (read warmup_mode)
_CFG = {"dir": "/recipe_cache/mtp_dump", "k": 64, "shard": 4096}
_AUX_LAYERS_SET = {"done": False}
_WARN = {"noli": 0, "shape": 0, "noaux": 0, "dbg": 0}


def _is_warmup() -> bool:
    mr = _RUNNER["mr"]
    return bool(getattr(mr, "warmup_mode", False)) if mr is not None else False


def _is_rank0() -> bool:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def _flush(D):
    import torch
    if not _S["buf"]:
        return
    os.makedirs(D, exist_ok=True)
    hid = torch.cat([b[0] for b in _S["buf"]])
    ti = torch.cat([b[1] for b in _S["buf"]])
    tv = torch.cat([b[2] for b in _S["buf"]])
    pos = torch.cat([b[3] for b in _S["buf"]])
    sid = torch.cat([b[4] for b in _S["buf"]])
    save = {"hidden": hid, "target_ids": ti[:, 0].contiguous(),
            "topk_ids": ti, "topk_logits": tv, "positions": pos, "seq_ids": sid}
    aux_list = [b[5] for b in _S["buf"] if len(b) > 5 and b[5] is not None]
    if aux_list and len(aux_list) == len(_S["buf"]):
        save["hidden_aux"] = torch.cat(aux_list)   # [N, 3H] EAGLE-3 low/mid/high pre-norm residuals
    out = os.path.join(D, f"shard_{_S['shard']:05d}.pt")
    torch.save(save, out)
    print(f"[mtp-dump] wrote {out} ({hid.shape[0]} positions)", file=sys.stderr, flush=True)
    _S["buf"] = []; _S["n"] = 0; _S["shard"] += 1


def _capture(sample_hidden, logits, aux_hidden_states, logits_indices):
    """SINGLE capture site, invoked from the _execute_model_generic wrapper AFTER the original returned
    AND after htorch.core.mark_step() (so the marked aux/hidden graph outputs hold the live per-token
    replay values, NOT the frozen disable_tensor_cache buffers).

      * sample_hidden     : the runner's 3rd return = hidden.view(-1,H)[logits_indices] (L3776-3777) —
                            ALREADY gathered to the logits_indices rows. 1:1 with `logits`.
      * logits            : the runner's 4th return = compute_logits(sample_hidden) — same rows.
      * aux_hidden_states : the runner's 2nd return = list[low,mid,high], each the FULL forward batch
                            [B*seq, H] (a MARKED graph output -> live). We gather it by the SAME
                            logits_indices to line up 1:1 with sample_hidden / logits.

    Decode-only gate (clean over maximal): only dump steps where logits_indices[:nreq] == [0..nreq-1]
    (pure decode: the on-policy trajectory EAGLE-3 trains on, and row j <-> request j so the
    seq_id/position bookkeeping is valid). Prefill / chunked-prefill (logits_indices points at
    chunk-last tokens, off-policy teacher-forced prompt) is skipped. Never raises into the serve."""
    if _is_warmup() or not _is_rank0():
        return
    try:
        if sample_hidden is None or logits is None:
            return
        import torch as _t
        D, K, NSH = _CFG["dir"], _CFG["k"], _CFG["shard"]
        mr = _RUNNER["mr"]
        ib = getattr(mr, "input_batch", None)
        nreq = getattr(ib, "num_reqs", None)
        if not nreq or int(nreq) <= 0:
            return
        # aux is mandatory for an EAGLE-3 row; if the runner didn't return it the install is wrong.
        if not (aux_hidden_states is not None and len(aux_hidden_states) == 3):
            if _WARN["noaux"] < 3:
                _WARN["noaux"] += 1
                print("[mtp-dump] runner returned NO aux (use_aux_hidden_state_outputs not set or model "
                      "didn't emit (hidden,aux)) — SKIPPING capture. Fix install().",
                      file=sys.stderr, flush=True)
            return
        if logits_indices is None:
            if _WARN["noli"] < 3:
                _WARN["noli"] += 1
                print("[mtp-dump] NO logits_indices in _emg — cannot align aux; SKIPPING capture.",
                      file=sys.stderr, flush=True)
            return
        try:
            li_cpu = logits_indices.detach().to("cpu").to(_t.int64).flatten()
        except Exception:  # noqa: BLE001
            return
        nreq = min(int(nreq), int(sample_hidden.shape[0]), int(logits.shape[0]), int(li_cpu.shape[0]))
        if nreq <= 0:
            return
        # DECODE-ONLY GATE (see docstring).
        if not _t.equal(li_cpu[:nreq], _t.arange(nreq, dtype=_t.int64)):
            return
        H = sample_hidden.shape[-1]
        # aux: list of [B*seq, H] live graph outputs. Flatten defensively (runner flattens hidden too),
        # then gather by the SAME logits_indices the runner applied to the hidden. Materialize .cpu()
        # HERE — _emg already mark_step()'d, so we are OUTSIDE the captured-graph region (live values).
        try:
            faux = [a_.reshape(-1, H) for a_ in aux_hidden_states]
        except Exception:  # noqa: BLE001
            return
        if any(int(fa.shape[0]) < nreq for fa in faux):
            # aux rows must cover the gather indices (full batch >= sliced rows). LOUD: shape drift.
            if _WARN["shape"] < 3:
                _WARN["shape"] += 1
                print(f"[mtp-dump] aux row-count {[tuple(fa.shape) for fa in faux]} < nreq={nreq} — "
                      "SKIPPING (cannot gather). Unexpected aux shape.", file=sys.stderr, flush=True)
            return
        idx = li_cpu[:nreq]
        sl = [fa.detach().index_select(0, idx).float().cpu() for fa in faux]
        h = sample_hidden[:nreq].detach().float().cpu()
        if not h.any():
            return  # warmup / all-zero
        tv, ti = logits[:nreq].detach().float().topk(K, dim=-1)
        tv, ti = tv.cpu(), ti.cpu()
        keep = (_t.isfinite(h).all(-1) & _t.isfinite(tv).all(-1)
                & _t.stack([_t.isfinite(s).all(-1) for s in sl]).all(0)).nonzero().flatten().tolist()
        if not keep:
            return
        reqids = list(getattr(ib, "req_ids", []) or [])
        pm = _S.setdefault("posmap", {}); sm = _S.setdefault("seqmap", {}); sc = _S.setdefault("seqctr", [0])
        pos = []; sids = []
        for j in keep:
            rid = reqids[j] if j < len(reqids) else j
            pj = pm.get(rid, 0); pm[rid] = pj + 1; pos.append(pj)
            if rid not in sm:
                sm[rid] = sc[0]; sc[0] += 1
            sids.append(sm[rid])
        if _WARN["dbg"] < 3:
            _WARN["dbg"] += 1
            a_norm = _t.cat(sl, dim=-1)[keep].norm(dim=-1).mean().item() if keep else 0.0
            print(f"[mtp-dump DBG] nreq={nreq} sample_hidden={tuple(sample_hidden.shape)} "
                  f"flat_aux={[tuple(s.shape) for s in sl]} keep={len(keep)} aux_L2~{a_norm:.2f}",
                  file=sys.stderr, flush=True)
        a3 = _t.cat([s[keep] for s in sl], dim=-1)  # [n_keep, 3H]
        _S["buf"].append((h[keep], ti[keep], tv[keep],
                          _t.tensor(pos, dtype=_t.int64), _t.tensor(sids, dtype=_t.int64), a3))
        _S["n"] += len(keep)
        if _S["n"] >= NSH:
            _flush(D)
    except Exception as ex:  # noqa: BLE001
        print(f"[mtp-dump] capture error {ex!r}", file=sys.stderr)


def _patch(Cls) -> bool:
    """Wrap forward on one model class (with ITS original): set the EAGLE-3 aux layers + return the
    FULL (hidden, aux) tuple so aux becomes a MARKED graph output. Capture happens in the runner's
    _execute_model_generic wrapper, NOT here (no in-trace .cpu())."""
    if Cls is None or getattr(getattr(Cls, "forward", None), "_mtp_dump", False):
        return False
    _of = Cls.forward

    def forward(self, input_ids, positions, *a, **k):
        # EAGLE-3 aux: ask the inner model to ALSO return low/mid/high PRE-norm residuals. Set the
        # layers on ALL ranks (identical forward graph per rank -> no lockstep all-reduce divergence;
        # the aux add at deepseek_v2 ~L1328 is a cheap local elementwise op, no collective). The
        # native runner sets these via set_aux_hidden_state_layers(get_eagle3_default...) on the
        # drafter path (L5134) — the dump serve has no drafter, so we set them here (idempotent).
        m = getattr(self, "model", None)
        if m is not None and not getattr(m, "aux_hidden_state_layers", ()):
            try:
                n = len(m.layers)
                m.aux_hidden_state_layers = (2, n // 2, n - 3)  # (2,39,75) for 78L
                if not _AUX_LAYERS_SET["done"]:
                    _AUX_LAYERS_SET["done"] = True
                    print(f"[mtp-dump] aux_hidden_state_layers set to {m.aux_hidden_state_layers} "
                          f"(n_layers={n}) for EAGLE-3 capture", file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001
                pass
        out = _of(self, input_ids, positions, *a, **k)
        # Return the FULL (hidden, aux_list) tuple UNCHANGED. The runner's _execute_model_generic
        # unpacks it (use_aux_hidden_state_outputs=True, set in the __init__ hook), and HpuModelAdapter
        # propagates it so the captured graph marks every aux tensor as a user output (kept live across
        # replay). DO NOT strip to out[0] and DO NOT snapshot aux here — that was the frozen-buffer bug.
        return out

    forward._mtp_dump = True
    Cls.forward = forward
    return True


def install() -> None:
    if (os.environ.get("HPU_MTP_DUMP") or "0") not in ("1", "true", "True"):
        return
    _CFG["dir"] = os.environ.get("HPU_MTP_DUMP_DIR", "/recipe_cache/mtp_dump")
    _CFG["k"] = int(os.environ.get("HPU_MTP_DUMP_K", "64"))
    _CFG["shard"] = int(os.environ.get("HPU_MTP_DUMP_SHARD", "4096"))
    patched = []
    # GLM-5.1 (DeepseekV2ForCausalLM) + Kimi-K2.7 text LM (DeepseekV3ForCausalLM) + GLM-5.2
    # (GlmMoeDsaForCausalLM). GlmMoeDsaForCausalLM is `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    # pass` (inherits forward), so the DeepseekV2ForCausalLM patch ALREADY covers it via inheritance;
    # _patch() is idempotent (the _mtp_dump guard sees the inherited wrapped forward and no-ops). Listed
    # explicitly so the dump is self-documenting + robust if GLM-5.2 ever overrides forward. The EAGLE-3
    # aux path (model returns (hidden, [low,mid,high]) when aux_hidden_state_layers is set) is inherited
    # from DeepseekV2Model (deepseek_v2 ~L1322-1341) — verified on the on-card shards.
    try:
        from vllm.model_executor.models import deepseek_v2 as dv2
        for cn in ("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM"):
            if _patch(getattr(dv2, cn, None)):
                patched.append(cn)
    except Exception as ex:  # noqa: BLE001
        print(f"[mtp-dump] deepseek_v2 patch skipped ({ex!r})", file=sys.stderr)
    # MiniMax-M3 (custom m3 vllm on sys.path at /opt/m3-vllm).
    try:
        import importlib
        m3 = importlib.import_module("minimax_m3")
        if _patch(getattr(m3, "HpuMiniMaxM3ForConditionalGeneration", None)):
            patched.append("HpuMiniMaxM3ForConditionalGeneration")
    except Exception as ex:  # noqa: BLE001
        print(f"[mtp-dump] minimax_m3 patch skipped ({ex!r})", file=sys.stderr)
    if not patched:
        print("[mtp-dump] no target model class importable — dump OFF", file=sys.stderr)
        return
    # Runner hooks:
    #   __init__  : grab the runner ref (read warmup_mode) AND force use_aux_hidden_state_outputs=True so
    #               _execute_model_generic unpacks the model's (hidden, aux) tuple and RETURNS aux.
    #   _execute_model_generic : the SINGLE capture site — after the original returns, mark_step() then
    #               capture (hidden, logits, aux) gathered by THIS step's logits_indices, decode-gated.
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
        if not getattr(HPUModelRunner.__init__, "_mtp_dump", False):
            _oi = HPUModelRunner.__init__

            def _init(self, *a, **k):
                _oi(self, *a, **k)
                _RUNNER["mr"] = self
                # Force the native eagle3 aux return on. With no speculative_config the runner never
                # calls the drafter (propose is gated on `if self.speculative_config:`), so aux is teed
                # by us and otherwise unused. The model emits (hidden, aux) once forward sets the layers.
                try:
                    self.use_aux_hidden_state_outputs = True
                    print("[mtp-dump] runner.use_aux_hidden_state_outputs = True (native eagle3 aux "
                          "return enabled)", file=sys.stderr, flush=True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[mtp-dump] could NOT set use_aux_hidden_state_outputs ({_e!r}) — aux capture "
                          "will SKIP (LOUD)", file=sys.stderr, flush=True)
            _init._mtp_dump = True
            HPUModelRunner.__init__ = _init

        if not getattr(getattr(HPUModelRunner, "_execute_model_generic", None), "_mtp_dump", False):
            import habana_frameworks.torch as _htorch
            _oe = HPUModelRunner._execute_model_generic

            def _emg(self, token_ids, position_ids, attn_metadata, logits_indices, *a, **k):
                # Capture flag: only when dump is for THIS rank/non-warmup do we pay the D2H sync.
                ret = _oe(self, token_ids, position_ids, attn_metadata, logits_indices, *a, **k)
                # _execute_model_generic returns the 4-tuple
                #   (non_flattened_hidden_states, aux_hidden_states, sample_hidden, logits)
                # (hpu_model_runner L3784). aux_hidden_states is index 1; sample_hidden index 2
                # (= hidden[logits_indices]); logits index 3. Capture OUTSIDE the captured-graph region:
                # mark_step() flushes the replay so the MARKED aux/hidden hold the live per-token values
                # (NOT the frozen disable_tensor_cache buffers), then materialize .cpu() in _capture.
                try:
                    if isinstance(ret, (tuple, list)) and len(ret) == 4 and not _is_warmup() and _is_rank0():
                        _aux = ret[1]
                        _sample_hidden = ret[2]
                        _logits = ret[3]
                        _htorch.core.mark_step()
                        _capture(_sample_hidden, _logits, _aux, logits_indices)
                except Exception as _ex:  # noqa: BLE001
                    print(f"[mtp-dump] _emg capture error {_ex!r}", file=sys.stderr)
                return ret
            _emg._mtp_dump = True
            HPUModelRunner._execute_model_generic = _emg
            print("[mtp-dump] _execute_model_generic wrapped (LIVE-aux capture from runner return, "
                  "post-mark_step, gathered by logits_indices)", file=sys.stderr, flush=True)
        else:
            print("[mtp-dump] _execute_model_generic already wrapped", file=sys.stderr, flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[mtp-dump] runner hook failed ({ex!r}) — aux capture OFF (LOUD, no silent misalign)",
              file=sys.stderr)
    print(f"[mtp-dump] installed for {patched} (dir={_CFG['dir']} K={_CFG['k']} "
          f"shard_every={_CFG['shard']}, warmup-gated, EAGLE-3 live-aux)", file=sys.stderr, flush=True)
