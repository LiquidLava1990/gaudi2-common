"""NATIVE MTP-head SELF-FEED dump (HEAD-TTT OPTION A capture): observe the GLM-5.2 *native* MTP
head's OWN self-feed drift on-policy, so an adapter can be TTT'd on the head's REAL per-depth accept
(the dd lever, HEAD_TTT.md §4 OPTION A). Gated HPU_MTP_SELFFEED_DUMP=1. Rank-0 only. OFF by default
=> byte-identical to stock (the wrapped methods short-circuit on the env gate before any work). Writes
a shard every HPU_MTP_SELFFEED_SHARD captured decode positions. NEVER raises into the serve.

────────────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS HOOK EXISTS (the gap it closes — HEAD_TTT.md §2/§5):
  Every on-disk dump (decode_dump_patch.py / mtp_dump{,_aux}) holds the TARGET model's hidden + the
  TARGET's top-K distribution, captured with NO speculative_config (the drafter is never called). It
  therefore observes NOTHING about the NATIVE MTP head's own proposals or its self-feed trajectory.
  The accept lever past the ~2.7 tree ceiling is the native head's self-feed drift `dd` (0.82 -> 0.93+);
  `dd` is a property of a head we cannot run off-card AND cannot observe in any existing dump. To TTT an
  adapter on the native head's seed we first need ONE on-policy capture of: per accepted decode step,
  (a) the exact hidden the native head drafts FROM, (b) the committed/seed token it drafts from, and
  (c) the K tokens the native head ACTUALLY drafted at that step (its own argmax at draft depths 1..K).
  This hook taps the native MTP serve (glm52-mtp-fp8other-d3.env, method=deepseek_mtp) to dump exactly
  those tuples. The per-depth accept (== dd's source) is then DERIVED off-card by joining consecutive
  same-seq steps (the committed token at position p+1 is the seed_token captured at the step whose
  position == p+1), so NO fragile in-loop accept-count plumbing is needed and the schema stays a strict
  superset of what ttt_train.py / eval_head.py already consume.

────────────────────────────────────────────────────────────────────────────────────────────────────
WHERE IT TAPS (verified against the on-card runner + proposer):
  Tap = vllm_gaudi.v1.worker.hpu_model_runner.HPUModelRunner.propose_eagle_decode (the runner method
  that builds the drafter inputs and calls self.drafter.propose, hpu_model_runner L7047-7119). After the
  original returns we re-run the (PURE) drafter.prepare_inputs to recover its index tensors, decode-only
  by construction (prefill has a separate propose_eagle_prefill path):
    * prepare_inputs returns (common_attn_metadata, hidden_states_indices=hsi, last_token_indices=lti).
      hsi is the FLAT drafter-grid gather (len = num_decodes*(max_draft+1), -1-padded for rejected
      slots) -- NOT one row per request. lti is the PER-REQUEST seed row (len == num_decodes): the row
      propose() itself gathers its seed from (hpu_eagle L65/76/83), aligned 1:1 with draft[:num_decodes].
    * SEED hidden   = (hidden_states[hsi]   gathered to [len(hsi),H]) then [lti] -> [num_decodes,H], then
                      unsqueeze(1) -> [num_decodes,1,H]. This is the EXACT per-request tensor propose()
                      seeds the native head with. For native MTP use_aux_hidden_state_outputs is False so
                      width == H (=6144), the final hidden the head consumes (NOT the 3H EAGLE-3 aux).
    * SEED token    = (decode_sampled[hsi] reshaped [len(hsi),1]) then [lti] -> [num_decodes] = the
                      committed token at this step that the head drafts FROM (per request).
    * native drafts = the RETURNED `draft_token_ids` (L7115/7118) = [num_decodes, num_speculative_tokens]
                      = the native head's OWN argmax at draft depths 1..K (its self-feed proposals).
    * seq_id/pos    = self.input_batch.req_ids (decode-first) -> per-request contiguous grouping,
                      identical semantics to decode_dump_patch (so build_seqs/_build_seqs reuse it
                      unchanged). NOTE: gathering the seed at hsi[:num_decodes] (the PRIOR BUG) grabbed
                      the first num_decodes flat-grid rows -- garbage for num_decodes>1 and misaligned
                      with native_draft -> the dump would carry a near-random a1. The lti gather fixes it.
  We read these AFTER the original returns and AFTER htorch.core.mark_step() (flush the replay so the
  device tensors hold the live per-step values, NOT frozen disable_tensor_cache buffers), then
  materialize .float()/.long().cpu() OUTSIDE the captured-graph region. Decode-only + non-warmup + pure
  per-request alignment gated; capture is teed and otherwise unused -> numerically inert (only a D2H
  copy when the dump is ON for this rank).

v5 SCHEMA (one shard = torch.save of a dict; spec-decode-INDEPENDENT -- the model-output token stream only):
    hidden        : [N, H]   float32 -- the native head's SEED input hidden (n_aux=0 seed; H=6144)
    target_ids    : [N]      int64   -- the SEED/committed token at (seq_id, position). For temp=0 greedy this
                                        IS the target's committed token at that absolute position.
    native_draft  : [N, K]   int64   -- the native MTP head's OWN drafts at depths 1..K (the self-feed obs).
    positions     : [N]      int64   -- the TRUE ABSOLUTE token position (decode_data.position_ids gathered at
                                        last_token_indices), NOT a per-step counter. THE JOIN KEY. (The old
                                        per-step counter was the subsample bug -- absolute position fixes it.)
    seq_ids       : [N]      int64   -- stable per-request id (req_id -> int); join key with positions.

  THE LABEL IS RECONSTRUCTED OFF-CARD (native_selffeed_accept.py), NOT captured: for temp=0 greedy the
  committed token at p+k == the target's argmax there, and the WHOLE token stream is gaplessly recoverable
  from (seed_token, abs_position, native_draft) -- the gap between consecutive seed positions is accepted+1,
  and the accepted drafts fill the gap. Then a_k = mean(native_draft[r,p,k-1] == token[r,p+k]) (the serve's
  per-position greedy accept) and label[r,p,k]=token[r,p+k] is the adapter's TTT target. NO target_verify /
  accepted / async-spec grid -- that signal is an unreachable WALL on this HPU build (4 failed rounds).
"""
import os
import sys

# Match decode_dump_patch.py's module-state convention exactly (single-source-of-truth structure).
# nlog: per-propose-call list of (native top-k logits, ids) teed from the MTP head's compute_logits (v6).
_S = {"buf": [], "n": 0, "shard": 0, "nlog": []}
_RUNNER = {"mr": None}            # set by the HPUModelRunner.__init__ hook (read warmup_mode + input_batch)
_CFG = {"dir": "/recipe_cache/mtp_selffeed", "shard": 4096, "nlog_topk": 64}
_WARN = {"noli": 0, "shape": 0, "align": 0, "dbg": 0, "kdrift": 0}
_KSEEN = {"k": None}              # the num_speculative_tokens width we lock to (LOUD if it ever drifts)


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
    # v6 buf rows: (seed_hid, seed_tok, native_draft[K], abs_position, seq_id, native_topk_logits[Kn],
    # native_topk_ids[Kn]). Spec-decode-independent label reconstruction (v5) PLUS native depth-1 top-K
    # logits (v6) so the adapter trains the TRUE residual objective: argmax(native + adapter) == target.
    hid = torch.cat([b[0] for b in _S["buf"]])
    ti = torch.cat([b[1] for b in _S["buf"]])
    nd = torch.cat([b[2] for b in _S["buf"]])
    pos = torch.cat([b[3] for b in _S["buf"]])
    sid = torch.cat([b[4] for b in _S["buf"]])
    ntl = torch.cat([b[5] for b in _S["buf"]])
    nti = torch.cat([b[6] for b in _S["buf"]])
    save = {"hidden": hid,        # [N, H]   the native head's SEED input hidden (n_aux=0 seed; H=6144)
            "target_ids": ti,     # [N]      the SEED/committed token at (seq_id, position)
            "native_draft": nd,   # [N, K]   the native MTP head's OWN drafts at depths 1..K
            "positions": pos,     # [N]      the TRUE ABSOLUTE token position (decode_data.position_ids), NOT
                                  #          a per-step counter -- the keyed-join key (fixes the subsample bug)
            "seq_ids": sid,       # [N]      stable per-request id (req_id -> int); join key with positions
            "native_topk_logits": ntl,  # [N, Kn] native DEPTH-1 top-K logits (-inf rows = tee missed)
            "native_topk_ids": nti}     # [N, Kn] the matching token ids (-1 = missed). The residual-vs-native
                                        #          training set: argmax(native + adapter) must rank target 1st.
    out = os.path.join(D, f"shard_{_S['shard']:05d}.pt")
    torch.save(save, out)
    have_nlog = int((nti[:, 0] >= 0).sum())
    print(f"[mtp-selffeed] wrote {out} ({hid.shape[0]} positions, K={nd.shape[-1]}, "
          f"pos-range {int(pos.min())}..{int(pos.max())}, native_topk rows={have_nlog}/{hid.shape[0]})",
          file=sys.stderr, flush=True)
    _S["buf"] = []; _S["n"] = 0; _S["shard"] += 1


def _capture(target_hidden_states, target_token_ids, draft_token_ids, num_decodes, abs_pos,
             nlog1=None, nid1=None):
    """SINGLE capture site, invoked from the propose_eagle_decode wrapper AFTER the original returned AND
    after htorch.core.mark_step() (so the device tensors hold the live per-step values, not the frozen
    disable_tensor_cache buffers). The caller (_ped) ALREADY gathered seed hidden/token AND the absolute
    token position at last_token_indices, so all arrive aligned 1:1 over the num_decodes real decode
    requests (seed_hid [nd,1,H], seed_tok [nd,1], draft_token_ids[:nd] [nd,K], abs_pos [nd]).

    v5 SCHEMA (spec-decode-INDEPENDENT): we capture ONLY the model-output stream the dump always has:
    seed_token, seed_hidden, native_draft[K], the TRUE absolute position, and the request id. The accept
    LABEL (the target's greedy argmax at p+k == the committed token at p+k) is reconstructed OFF-CARD by
    native_selffeed_accept.py from these (the gap between consecutive seed positions = accepted+1, and the
    accepted drafts fill the gap). NO target_verify / accepted / async-spec grid -- that path is an
    unreachable wall on this HPU build. Never raises into the serve."""
    if _is_warmup() or not _is_rank0():
        return
    try:
        import torch as _t
        D, NSH = _CFG["dir"], _CFG["shard"]
        mr = _RUNNER["mr"]
        if target_hidden_states is None or target_token_ids is None or draft_token_ids is None:
            return
        nd = int(num_decodes) if num_decodes is not None else 0
        if nd <= 0:
            return
        # SEED hidden arrives [nd,1,H] -> flatten singleton seq dim -> [nd, H]. LOUD if rows < nd.
        th = target_hidden_states
        if th.dim() == 3:
            th = th.reshape(th.shape[0], -1) if th.shape[1] == 1 else th[:, 0, :]
        elif th.dim() != 2:
            th = th.reshape(th.shape[0], -1)
        if th.shape[0] < nd:
            if _WARN["align"] < 3:
                _WARN["align"] += 1
                print(f"[mtp-selffeed] seed hidden rows {tuple(th.shape)} < num_decodes={nd} -- SKIP "
                      "(last_token_indices alignment / runner contract changed -- re-point the tap).",
                      file=sys.stderr, flush=True)
            return
        h = th[:nd].detach().float().cpu()
        if not h.any():
            return  # warmup / all-zero placeholder

        tt = target_token_ids.reshape(-1)
        if tt.shape[0] < nd:
            return
        ti = tt[:nd].detach().to(_t.int64).cpu()

        dft = draft_token_ids
        if dft.dim() != 2:
            if _WARN["shape"] < 3:
                _WARN["shape"] += 1
                print(f"[mtp-selffeed] draft_token_ids is {dft.dim()}-D {tuple(dft.shape)}, expected 2-D "
                      "[batch,K] (native single-chain MTP). Multi-cand/tree serve unsupported -- SKIP.",
                      file=sys.stderr, flush=True)
            return
        if dft.shape[0] < nd:
            return
        nd_drafts = dft[:nd].detach().to(_t.int64).cpu()
        K = int(nd_drafts.shape[-1])
        if K < 1:
            return
        if _KSEEN["k"] is None:
            _KSEEN["k"] = K
        elif _KSEEN["k"] != K:
            if _WARN["kdrift"] < 3:
                _WARN["kdrift"] += 1
                print(f"[mtp-selffeed] num_speculative_tokens drifted {_KSEEN['k']} -> {K} -- SKIP step.",
                      file=sys.stderr, flush=True)
            return

        # ABSOLUTE POSITION (the v5 keyed-join key). MUST be present; without it the reconstruction can't
        # join the token stream (LOUD -- do NOT silently fall back to a per-step counter, which was the
        # subsample bug). If abs_pos is missing, SKIP and warn so the failure is obvious on-card.
        if abs_pos is None or abs_pos.reshape(-1).shape[0] < nd:
            if _WARN["align"] < 3:
                _WARN["align"] += 1
                print(f"[mtp-selffeed] ABSOLUTE position unavailable "
                      f"({None if abs_pos is None else tuple(abs_pos.shape)}) -- SKIP. The v5 capture REQUIRES "
                      "decode_data.position_ids[last_token_indices]; a per-step counter would re-introduce "
                      "the subsample bug. Re-point the position tap.", file=sys.stderr, flush=True)
            return
        apos = abs_pos.reshape(-1)[:nd].detach().to(_t.int64).cpu()

        # request id per row: the real input_batch.req_ids (decode-first), mapped to a stable integer.
        ib = getattr(mr, "input_batch", None)
        reqids = list(getattr(ib, "req_ids", []) or [])
        finite = (_t.isfinite(h).all(-1) & (ti >= 0) & (nd_drafts >= 0).all(-1) & (apos >= 0)) \
            .nonzero().flatten().tolist()
        if not finite:
            return
        sm = _S.setdefault("seqmap", {}); sc = _S.setdefault("seqctr", [0])
        sids = []
        for j in finite:
            rid = reqids[j] if j < len(reqids) else j
            if rid not in sm:
                sm[rid] = sc[0]; sc[0] += 1
            sids.append(sm[rid])
        fin = _t.tensor(finite, dtype=_t.int64)
        if _WARN["dbg"] < 3:
            _WARN["dbg"] += 1
            print(f"[mtp-selffeed DBG] num_decodes={nd} seed_hidden={tuple(h.shape)} K={K} kept={len(finite)} "
                  f"seed_tok0={int(ti[finite[0]])} draft0={nd_drafts[finite[0]].tolist()} "
                  f"abs_pos0={int(apos[finite[0]])} seq0={sids[0]}  "
                  f"native_topk0={'yes' if nlog1 is not None else 'NO'}", file=sys.stderr, flush=True)
        # v6 native DEPTH-1 top-K logits + ids (the residual-vs-native training signal). Shape [nd, Kn];
        # if the compute_logits tee MISSED this step, store a sentinel (ids=-1, logits=-inf) so the cat at
        # flush stays rectangular and the trainer/analyzer LOUD-skip those rows for the residual loss.
        Kn = _CFG.get("nlog_topk", 64)
        if nlog1 is not None and nid1 is not None and nlog1.shape[0] >= nd and nlog1.shape[-1] >= 1:
            nl = nlog1[:nd].float()
            ni = nid1[:nd].to(_t.int64)
            if nl.shape[-1] < Kn:                       # pad to Kn (defensive)
                nl = _t.cat([nl, nl.new_full((nd, Kn - nl.shape[-1]), float("-inf"))], dim=-1)
                ni = _t.cat([ni, ni.new_full((nd, Kn - ni.shape[-1]), -1)], dim=-1)
            elif nl.shape[-1] > Kn:
                nl = nl[:, :Kn]; ni = ni[:, :Kn]
        else:
            nl = _t.full((nd, Kn), float("-inf")); ni = _t.full((nd, Kn), -1, dtype=_t.int64)
        # buf row: (seed_hid, seed_tok, native_draft[K], abs_position, seq_id, native_topk_logits, native_topk_ids)
        _S["buf"].append((h[finite], ti[finite], nd_drafts[finite],
                          apos[fin], _t.tensor(sids, dtype=_t.int64), nl[fin], ni[fin]))
        _S["n"] += len(finite)
        if _S["n"] >= NSH:
            _flush(D)
    except Exception as ex:  # noqa: BLE001
        print(f"[mtp-selffeed] capture error {ex!r}", file=sys.stderr)


def install() -> None:
    if (os.environ.get("HPU_MTP_SELFFEED_DUMP") or "0") not in ("1", "true", "True"):
        return
    _CFG["dir"] = os.environ.get("HPU_MTP_SELFFEED_DIR", "/recipe_cache/mtp_selffeed")
    _CFG["shard"] = int(os.environ.get("HPU_MTP_SELFFEED_SHARD", "4096"))
    _CFG["nlog_topk"] = int(os.environ.get("HPU_MTP_SELFFEED_NLOG_TOPK", "64"))  # native depth-1 top-K (v6)
    try:
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
    except Exception as ex:  # noqa: BLE001
        print(f"[mtp-selffeed] HPUModelRunner not importable ({ex!r}) -- selffeed dump OFF",
              file=sys.stderr)
        return

    # Runner __init__ hook: grab the runner ref (warmup_mode + input_batch for seq_id/pos).
    if not getattr(HPUModelRunner.__init__, "_mtp_selffeed", False):
        _oi = HPUModelRunner.__init__

        def _init(self, *a, **k):
            _oi(self, *a, **k)
            _RUNNER["mr"] = self
            print("[mtp-selffeed] runner ref captured (warmup-gated native self-feed capture armed)",
                  file=sys.stderr, flush=True)
        _init._mtp_selffeed = True
        HPUModelRunner.__init__ = _init

    # v5 (2026-06-23): NO rejection_sample tee / NO async-spec grid. Four prior rounds proved the
    # spec-decode accept signal (target_verify / _async_spec_grid) is an unreachable WALL on this HPU build
    # (use_async_spec_decode stays None despite HPU_ASYNC_SPEC=1 + --async-scheduling; the rejection_sample
    # tee read None at propose time). PIVOT: capture ONLY the model-output token stream (seed_token,
    # seed_hidden, native_draft[K], TRUE absolute position, req_id) -- all of which the DBG already confirmed
    # REAL -- and reconstruct the accept LABEL off-card (native_selffeed_accept.py): for temp=0 greedy the
    # committed token at p+k IS the target's argmax there, and the gap between consecutive seed positions =
    # accepted+1 (the accepted drafts fill the gap). The single capture site below is propose_eagle_decode.

    # propose_eagle_decode hook: the SINGLE capture site. After the original returns the native head's
    # drafts, mark_step() then capture (seed hidden, seed token, drafts, absolute position) over num_decodes.
    if getattr(getattr(HPUModelRunner, "propose_eagle_decode", None), "_mtp_selffeed", False):
        print("[mtp-selffeed] propose_eagle_decode already wrapped", file=sys.stderr, flush=True)
    elif not hasattr(HPUModelRunner, "propose_eagle_decode"):
        print("[mtp-selffeed] HPUModelRunner has NO propose_eagle_decode -- the runner version changed; "
              "capture OFF (LOUD, no silent miss). Re-point the tap.", file=sys.stderr, flush=True)
    else:
        import habana_frameworks.torch as _htorch
        _oped = HPUModelRunner.propose_eagle_decode

        def _ped(self, sampled_token_ids, decode_sampled_token_ids_tensor, hidden_states,
                 aux_hidden_states, num_decodes, decode_data, *a, **k):
            # Recompute the SEED hidden/token exactly as the original does (so we capture the precise
            # tensors propose() consumes) WITHOUT perturbing the original call. We re-derive them from
            # the same inputs after the original returns the drafts; the drafter's prepare_inputs is
            # deterministic given the same decode_data/sampled_token_ids, so re-deriving is byte-faithful
            # and side-effect-free (no second propose, no KV write). Cheaper + simpler: capture the
            # ORIGINAL's drafts, then re-run only the (host-cheap) index gather for the seed tensors.
            #
            # v6 NATIVE-LOGITS tee (the real single-stream lever): the adapter is wired as a RESIDUAL
            # (serve does argmax(native_logits + alpha*adapter_logits)), so to train it to actually SHIFT
            # native's argmax on the cases native gets WRONG -- not as a standalone predictor (which is a
            # no-op when added to confident native logits) -- training needs native's OWN logits. They are
            # computed INSIDE propose at self.model.compute_logits (hpu_eagle L66 depth-1, L160 loop). We
            # tee the per-depth TOP-K native logits+ids for the DURATION of this propose call (top-K is
            # sufficient: the residual-vs-native loss only needs native's competing top tokens + the target;
            # full V logits are 46 GB, top-64 is ~0.1 GB). Teed (no math change) -> serve byte-identical.
            _S["nlog"] = []   # per-depth (topk_logits[nd,Kn], topk_ids[nd,Kn]) collected this propose call
            _drafter_model = getattr(self, "drafter", None)
            _drafter_model = getattr(_drafter_model, "model", None) if _drafter_model is not None else None
            _ocl_native = getattr(_drafter_model, "compute_logits", None) if _drafter_model is not None else None
            _KN = _CFG.get("nlog_topk", 64)
            _cl_tee = None   # CRASH FIX: the finally references _cl_tee; it's only defined in the if below
            #                   (rank0 + non-warmup). On ranks 1-7 / warmup steps it was UNDEFINED ->
            #                   UnboundLocalError in finally -> worker dies -> EngineCore cancelled -> stall.
            if _ocl_native is not None and not _is_warmup() and _is_rank0():
                def _cl_tee(hidden, *ca, **ck):
                    lg = _ocl_native(hidden, *ca, **ck)
                    try:
                        if lg is not None and lg.dim() >= 2:
                            lg2 = lg.reshape(-1, lg.shape[-1])
                            kn = min(_KN, lg2.shape[-1])
                            tv, ti2 = lg2.topk(kn, dim=-1)
                            # CRASH FIX 2026-06-23: materialize to CPU HERE (not hold device tensors across
                            # the captured propose region -> they go stale on HPU-graph replay -> worker dies
                            # "Neither storage attached" -> EngineCore cancelled). float()/long() + .cpu()
                            # forces an off-graph host copy; the held value is now a plain CPU tensor.
                            _S["nlog"].append((tv.detach().float().cpu(), ti2.detach().long().cpu()))
                    except Exception:  # noqa: BLE001
                        pass
                    return lg
                _drafter_model.compute_logits = _cl_tee
            try:
                draft_token_ids = _oped(self, sampled_token_ids, decode_sampled_token_ids_tensor,
                                        hidden_states, aux_hidden_states, num_decodes, decode_data, *a, **k)
            finally:
                if _ocl_native is not None and getattr(_drafter_model, "compute_logits", None) is _cl_tee:
                    _drafter_model.compute_logits = _ocl_native   # restore (no persistent mutation)
            try:
                if (not _is_warmup() and _is_rank0() and num_decodes and int(num_decodes) > 0
                        and decode_data is not None
                        and getattr(decode_data, "spec_decode_metadata", None) is not None):
                    # Re-derive the seed hidden/token via the SAME gather the original used (host-cheap
                    # index ops; no model forward). Mirrors the in-image propose_eagle_decode body EXACTLY
                    # (hpu_model_runner L7098-7114):
                    #   common_attn_metadata, hsi, lti = drafter.prepare_inputs(cam, spec_md, sampled,
                    #                                                           async_spec_valid_count)
                    #   target_token_ids = decode_sampled_token_ids_tensor.reshape(-1,1)[hsi]
                    #   target_hidden    = (cat aux) or hidden_states[hsi]; .unsqueeze(1) if 2-D
                    # VERIFIED signature (hpu_eagle.HpuEagleProposer.prepare_inputs): 4 params after self
                    # -- (common_attn_metadata, spec_decode_metadata, sampled_token_ids, num_accepted=None);
                    # the REAL call ALWAYS passes async_spec_valid_count as the 4th positional (None unless
                    # use_async_spec_decode). num_accepted has a DEFAULT so inspect counts 4 params -> the
                    # `>= 4` branch fires and we pass async_vc, matching the live call. We bind by the REAL
                    # runtime signature (inspect) so this stays correct across image variants (a build that
                    # drops the 4th param would simply not get the extra arg).
                    import torch as _t
                    import inspect as _inspect
                    _pi_args = [decode_data.attn_metadata, decode_data.spec_decode_metadata,
                                sampled_token_ids]
                    try:
                        _pi_params = _inspect.signature(self.drafter.prepare_inputs).parameters
                    except (TypeError, ValueError):
                        _pi_params = {}
                    # Bound method -> `self` already excluded; remaining are the real positional params.
                    if len(_pi_params) >= 4:
                        async_vc = getattr(self, "_async_spec_valid_count", None) \
                            if getattr(self, "use_async_spec_decode", False) else None
                        _pi_args.append(async_vc)
                    # prepare_inputs is PURE (builds index lists + new tensors; returns common_attn_metadata
                    # unchanged; no model forward, no KV write -- verified against hpu_eagle.prepare_inputs),
                    # so re-running it here is side-effect-free. It returns:
                    #   hsi  (hidden_states_indices): the FLAT drafter-grid gather, len = nd*(max_draft+1),
                    #        with -1 padding for rejected slots -- NOT one row per request.
                    #   lti  (last_token_indices)  : the PER-REQUEST seed row (len == num_decodes). This is
                    #        the row propose() itself gathers its seed from (hpu_eagle L65/76/83) and the row
                    #        that aligns 1:1 with the returned draft_token_ids[:nd]. Using hsi[:nd] instead
                    #        (the prior bug) grabs the first nd flat-grid rows -- garbage for nd>1, misaligned
                    #        with native_draft. We MUST gather the seed at last_token_indices.
                    _cam, hsi, lti = self.drafter.prepare_inputs(*_pi_args)
                    # Build the SAME [len(hsi),...] target tensors the original built (L7104/7111-7114) ...
                    tgt_tok_grid = decode_sampled_token_ids_tensor.reshape(-1, 1)[hsi]          # [len(hsi),1]
                    if getattr(self, "use_aux_hidden_state_outputs", False) and aux_hidden_states is not None:
                        tgt_hid_grid = _t.cat([h[hsi] for h in aux_hidden_states], dim=-1)       # [len(hsi),Haux]
                    else:
                        tgt_hid_grid = hidden_states[hsi]                                         # [len(hsi),H]
                    # ... then gather the PER-REQUEST seed at last_token_indices (== propose's own seed,
                    # aligned with draft_token_ids[:nd]). [nd,1]->keep; [nd,H]->unsqueeze(1) to [nd,1,H].
                    seed_tok = tgt_tok_grid.reshape(-1)[lti].reshape(-1, 1)                       # [nd,1]
                    seed_hid = tgt_hid_grid.reshape(tgt_hid_grid.shape[0], -1)[lti]               # [nd,H]
                    if seed_hid.dim() == 2:
                        seed_hid = seed_hid.unsqueeze(1)                                          # [nd,1,H]
                    # ABSOLUTE TOKEN POSITION (the v5 fix -- spec-decode-INDEPENDENT accept signal). The
                    # async-spec grid path NEVER engages on this HPU build (use_async_spec_decode stays None
                    # despite HPU_ASYNC_SPEC=1 + --async-scheduling; the rejection_sample tee also failed) ->
                    # target_verify/accepted are an unreachable WALL. PIVOT: for temp=0 GREEDY decode the
                    # committed token at any absolute position IS the target's argmax there, and the WHOLE
                    # committed stream is reconstructable OFF-CARD from (seed_token, seed_abs_position,
                    # native_draft[K]) alone -- the gap between consecutive seed positions = accepted+1, and
                    # the accepted drafts fill the gap (native_selffeed_accept.py reconstructs it). So we just
                    # need the TRUE absolute position per seed: decode_data.position_ids gathered at lti
                    # (the EXACT tensor the native head drafts at, runner L7105). This REPLACES the per-step
                    # counter (the old subsample bug) with a keyed (req_id, abs_pos) join.
                    abs_pos = None
                    try:
                        pid = getattr(decode_data, "position_ids", None)
                        if pid is not None:
                            abs_pos = pid.reshape(-1)[lti]                                        # [nd] absolute pos
                    except Exception:  # noqa: BLE001
                        abs_pos = None
                    # v6: collect the per-depth native TOP-K logits teed during this propose call. The list
                    # has one (logits[.,Kn], ids[.,Kn]) per compute_logits call (depth 1..K). Each is already
                    # per-request [nd, Kn] (depth-1 from sample_hidden_states[last_token_indices]; loop depths
                    # [batch_size,.]). We keep DEPTH-1 (the dominant accept term + the residual-vs-native
                    # training signal); shape [nd, Kn] logits + ids, aligned 1:1 with seed/draft rows.
                    nlog1 = nid1 = None
                    nl = _S.get("nlog") or []
                    if nl:
                        ld, idd = nl[0]
                        if ld.dim() == 2 and ld.shape[0] >= num_decodes:
                            nlog1, nid1 = ld[:num_decodes], idd[:num_decodes]
                    if _WARN["dbg"] < 3:
                        print(f"[mtp-selffeed] v6 capture: abs_pos="
                              f"{'set' if abs_pos is not None else 'None(BAD)'}  native_topk="
                              f"{'set(depths=%d)' % len(nl) if nl else 'None(compute_logits tee MISSED -- '
                              'residual-vs-native training unavailable)'}",
                              file=sys.stderr, flush=True)
                    _htorch.core.mark_step()
                    if abs_pos is not None:
                        apd = abs_pos.detach()
                        abs_pos = apd.cpu() if apd.device.type != "cpu" else apd
                        if abs_pos.reshape(-1).shape[0] >= num_decodes:
                            abs_pos = abs_pos.reshape(-1)[:num_decodes].to(_t.int64)              # [nd]
                    if nlog1 is not None:
                        nlog1 = (nlog1.cpu() if nlog1.device.type != "cpu" else nlog1).float()
                        nid1 = (nid1.cpu() if nid1.device.type != "cpu" else nid1).to(_t.int64)
                    _S["nlog"] = []   # consume
                    _capture(seed_hid, seed_tok, draft_token_ids, num_decodes, abs_pos, nlog1, nid1)
            except Exception as _ex:  # noqa: BLE001
                print(f"[mtp-selffeed] propose_eagle_decode capture error {_ex!r}", file=sys.stderr)
            return draft_token_ids

        _ped._mtp_selffeed = True
        HPUModelRunner.propose_eagle_decode = _ped
        print("[mtp-selffeed] propose_eagle_decode wrapped (native-head self-feed capture: seed hidden + "
              "seed token + native drafts[K] + TARGET verify grid[K+1] + accepted count, post-mark_step, "
              "decode-only)", file=sys.stderr, flush=True)

    print(f"[mtp-selffeed] installed (dir={_CFG['dir']} shard_every={_CFG['shard']}, warmup-gated, "
          "rank0-only; native MTP self-feed)", file=sys.stderr, flush=True)
