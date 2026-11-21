# SPDX-License-Identifier: Apache-2.0
# ==================================================================================================
# hpu_eagle_tree.py -- the SHARED-TREE parallel-verify proposer (single-stream Lever #5, the ×accept
# lever) for the GLM-5.2 NATIVE fp8 MTP head.  MOUNTED OVER the in-image
# /opt/vllm-gaudi/vllm_gaudi/v1/spec_decode/hpu_eagle.py by glm52-mtp-tree.env, so this module MUST
# expose a class named `HpuEagleProposer` (the runner does
# `from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer`). The bind-mount + the class name
# IS the registration -- there is no registry call (identical convention to hpu_eagle_factoryeagle.py).
#
# RELATION TO STOCK (hpu_eagle.py): this is a byte-faithful COPY of the canonical depth-3 MTP proposer
# (the propose_eagle_decode/propose_eagle_prefill path the runner's worker/hpu_model_runner.py calls)
# with ONE addition: a `HPU_TREE_VERIFY`-gated branch in propose() that, instead of the single argmax
# draft chain, expands the native MTP head into a depth-skewed candidate TREE (widths HPU_TREE_WIDTHS,
# pruned to HPU_TREE_BUDGET nodes) using the CPU-validated helpers in mmel/tree_verify/ (mounted to
# /opt/tree_verify, PYTHONPATH=/opt/tree_verify). The tree rides the bs=1 device free-ride: the same MME
# the depth-3 chain leaves idle now carries up to T candidate nodes for ~one decode step.
#
# DEFAULT-SAFE / DEFAULT-OFF (load-bearing, the quality bar):
#   * HPU_TREE_VERIFY unset/0  => the gated branch is NEVER entered. propose()/prepare_inputs/
#     prepare_attn_metadata are character-for-character the stock hpu_eagle.py methods -> byte-identical
#     to stock (the A/B base glm52-mtp-bsgt1 sees ZERO difference).
#   * The tree helpers are imported LAZILY, only inside the gated branch (and only after the LOUD config
#     parse). So importing this module on a CPU box WITHOUT /opt/tree_verify on sys.path still succeeds
#     (py_compile / a stock serve never touch the import).
#   * Greedy draft (temperature 0) + greedy verify on the target => the tree only changes HOW MANY tokens
#     commit per step, never WHICH. The committed text MUST be byte-identical to the depth-3 chain.
#
# GATING (DOUBLE, mirror of HPU_MULTI_CAND / HPU_MULTI_CAND_ONCARD):
#   HPU_TREE_VERIFY=1          master enable -> propose() builds the tree on the native head and stashes the
#                              verify plan (build_tree_decode_batch) + the accept entry (accept_tree_batch)
#                              on the model_runner for the runner's tree-verify-input path to consume.
#   HPU_TREE_VERIFY_ONCARD=1   the runner-side second gate (tree_runner_config(); enforced by the runner)
#                              that actually engages the new bs=T verify-input/KV path. This proposer does
#                              NOT read ONCARD itself -- it only PRODUCES the tree + plan; whether the
#                              runner takes the new KV path is the runner's gate. With ONCARD unset the
#                              proposer still returns a valid flattened draft grid (the stock verify shape)
#                              so nothing breaks, but the tree's wider accept is only realized once the
#                              runner consumes the stashed plan.
#   HPU_TREE_WIDTHS / HPU_TREE_BUDGET  parsed (LOUD on a malformed schedule) by tree_serve_hook.
#
# WHY the tree is built HERE (in the proposer) and not in the runner: the proposer already owns the
# native MTP head (self.model + self.model.compute_logits) and is the one place the head is driven for
# drafting. tree_serve_hook.build_tree_oncard expands the head ONE batched call per LEVEL; we feed it a
# per-node head closure that reuses EXACTLY the stock decode-forward (self.model(...) + compute_logits),
# so the head is NOT retrained and the fp8 graft is untouched -- we only widen WHICH tokens we draft.
# ==================================================================================================
import itertools
import os

import torch
from vllm.logger import init_logger
from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)

# async+spec (#3): when HPU_ASYNC_SPEC=1 the multi-draft decode loop hoists ALL per-draft attention
# metadata to BEFORE the device-replay loop. For decode each draft advances positions by exactly +1,
# so every draft's block_table/slot_mapping is known up front (no draft-token dependence). Building it
# ahead of time turns the loop body into pure HPU-graph replay -- the device no longer stalls on the
# Python host rebuild (.tolist()/.apply_()/defragmenter/4x h2d) between replays. Default (env unset)
# keeps the original per-iteration build -> byte-identical to stock. Numerics are IDENTICAL: the same
# prepare_attn_metadata is called on the same positions_cpu sequence, only the call site moves earlier.
_ASYNC_SPEC = os.environ.get("HPU_ASYNC_SPEC", "0") == "1"

# TREE-VERIFY master gate. We read it ONCE at import so the propose() hot path is a single bool test on
# the OFF path (no per-step env lookup) -> the stock path stays byte-identical and allocation-free.
_TRUTHY = ("1", "true", "True", "yes", "on")
_TREE_VERIFY = os.environ.get("HPU_TREE_VERIFY", "0").strip() in _TRUTHY

# DRAFTER selector (the verify-the-union SS win). HPU_TREE_DRAFTER unset/"mtp" => the native MTP head
# drives the tree (UNCHANGED, the proven path). "union" => PLD ∪ ngram cheap host drafters (ZERO head
# forward, ZERO lm_head) build the tree from the committed token sequence ALONE -> drafting is ~free, so
# the step collapses to ~one bs=Tsum verify forward (the SS lever). Read ONCE at import. LOUD on a typo.
_DRAFTER = os.environ.get("HPU_TREE_DRAFTER", "mtp").strip().lower()
if _DRAFTER not in ("mtp", "union"):
    raise ValueError(
        f"HPU_TREE_DRAFTER={_DRAFTER!r} must be 'mtp' (default, the native MTP head) or 'union' "
        "(PLD ∪ ngram cheap drafters). STOP (LOUD).")

# One-shot LOUD log so the tree engagement is visible in the server log exactly once (scan_tree_log in
# tree_measure.sh greps for it), not spammed every decode step.
_TREE_LOGGED = False

# DIAGNOSTIC: log the dispatch decision for the first few propose() calls (un-suppressible, independent of
# _TREE_LOGGED) so the union-vs-fallthrough branch is observable. Bounded -> no steady-state log spam.
_DISPATCH_N = 0

# DIAGNOSTIC: prove at import whether THIS module (and its _DRAFTER) is live, and confirm the drafter path
# is even used (if propose() never logs but this does, the drafting is routed around the proposer).
logger.info("[HPU_EAGLE_TREE IMPORTED] _TREE_VERIFY=%s _DRAFTER=%r", _TREE_VERIFY, _DRAFTER)


class HpuEagleProposer(EagleProposer):

    def propose(
        self,
        # [virtual_batch_size, seq_len]
        target_token_ids,
        # [virtual_batch_size, seq_len]
        target_positions,
        # [virtual_batch_size, seq_len, hidden_size]
        target_hidden_states,
        # [batch_size]
        last_token_indices,
        common_attn_metadata,
        # [num_seq, total_blocks]
        block_table_cpu_tensor,
        model_runner,
    ):
        # DIAGNOSTIC (first 8 calls): why is the union fast-path taken or skipped? Logs the exact guard inputs.
        global _DISPATCH_N
        if _DISPATCH_N < 8:
            _DISPATCH_N += 1
            _lti = "None" if last_token_indices is None else (
                "shape=%s" % (tuple(last_token_indices.shape),))
            _cond = bool(_TREE_VERIFY and _DRAFTER == "union" and (
                last_token_indices is None or int(last_token_indices.shape[0]) == 1))
            logger.info("[TREE-DISPATCH #%d] _TREE_VERIFY=%s _DRAFTER=%r last_token_indices=%s "
                        "next_seed=%s -> union_fastpath=%s", _DISPATCH_N, _TREE_VERIFY, _DRAFTER, _lti,
                        getattr(model_runner, "_tree_next_seed", None) is not None, _cond)
            # STDERR mirror: this module's logger is dropped in-serve (imported before vLLM attaches
            # handlers), so logger.info above is invisible. stderr reaches docker logs -> observable.
            import sys as _sys
            _sys.stderr.write("[TREE-DISPATCH #%d] _DRAFTER=%r lti=%s next_seed=%s union_fastpath=%s\n" % (
                _DISPATCH_N, _DRAFTER, _lti,
                getattr(model_runner, "_tree_next_seed", None) is not None, _cond))
            _sys.stderr.flush()
        # UNION CHEAP-DRAFTER FAST-PATH (HPU_TREE_DRAFTER=union, the verify-the-union SS win): PLD ∪ ngram
        # draft from the committed token sequence ALONE -- no MTP head forward, no lm_head, no seed hidden.
        # Placed BEFORE the re-seed seed-forward AND the main target/MTP forward so BOTH are skipped ->
        # drafting is ~free. _propose_union handles the re-seed vs first/prefill geometry + the
        # _tree_root_committed_token signalling IDENTICALLY to the MTP path, then delegates the proven
        # plan/stash/flat-grid to _propose_tree(prebuilt_trees=...). OFF (default "mtp") => never entered.
        # BS>1 FALLBACK (the union rides the single-request tree verify/commit/re-seed machinery -- see the
        # batch_size!=1 stash clear below and _propose_tree's batch_size!=1 guard): engage the zero-forward
        # union tree ONLY for a lone decode request. A bs>1 decode call falls through to the stock flat
        # drafter (which correctly drafts for any batch). The re-seed re-drive passes last_token_indices=None
        # (bs=1 by construction) -> engage. Short-circuits so the default (_TREE_VERIFY off) path is unchanged.
        if _TREE_VERIFY and _DRAFTER == "union" and (
                last_token_indices is None or int(last_token_indices.shape[0]) == 1):
            return self._propose_union(target_token_ids, target_positions, last_token_indices,
                                       block_table_cpu_tensor, model_runner)
        # NEXT-TREE-FEED entry: a prior commit-direct step stashed model_runner._tree_next_seed =
        # (target_hidden[H], token:int, pos:int) -- the target's hidden at the last committed token. Seed
        # the NEW tree directly from it (the caller passes all-None tensors on this path). Single-request.
        _next_seed = getattr(model_runner, "_tree_next_seed", None)
        if _next_seed is not None and _TREE_VERIFY:
            seed_target_hidden, seed_token, seed_pos = _next_seed
            # ROOT-TOKEN SOURCE for the runner's verify-forward override: on a RE-SEED step the committed
            # token at root_ctx is the BONUS we stash here (seed_token == tgt[winning_leaf]); the runner's
            # token store (token_ids_cpu) does NOT hold the in-flight bonus position yet (the spec-decode
            # commit path writes it later), so the runner MUST source node0 from this stashed token, not the
            # (stale, ==0) store. Stash it for self._tree_verify_decode's root-token override.
            model_runner._tree_root_committed_token = int(seed_token)
            # TRACE (gated HPU_TREE_TRACE=1, rank0): the re-seed identity + the MTP-KV slot this seed
            # forward writes, to confirm/deny that the tree's speculative sibling forwards polluted the
            # committed position's MTP KV. Cheap host logging; off by default.
            if os.environ.get("HPU_TREE_TRACE", "0").strip() in _TRUTHY and getattr(model_runner, "is_driver_worker", True):
                _bt2d = model_runner.input_batch.block_table[0].get_cpu_tensor()  # [num_seq, num_blocks]
                _bt0 = _bt2d[0]                                                   # request 0's block row
                _bsz = model_runner.block_size
                _cp = seed_pos if seed_pos < self.max_model_len else 0
                _blk = int(_bt0[_cp // _bsz].item()) if (_cp // _bsz) < _bt0.shape[0] else -1
                _mtp_slot = _blk * _bsz + (_cp % _bsz)
                logger.info("[TREE-TRACE reseed] seed_pos=%d seed_token=%d -> MTP-KV slot=%d (block=%d) "
                            "[this seed forward WRITES MTP KV here; children write seed_pos+1..]",
                            seed_pos, int(seed_token), _mtp_slot, _blk)
            # RE-SEED, MIRRORING the step-1 (non-seed) MTP path below (~L153-197) EXACTLY:
            #   * seed_hidden PROPAGATED to _propose_tree == the TARGET model's hidden at the last committed
            #     token (step-1 passes target_hidden_states[last_token_indices], NOT the MTP-forward output).
            #     _make_head_step feeds it as the root's parent_hidden into mtp_head_step, which RE-RUNS the
            #     MTP forward on it for the root's children -- so the children draft against the target hidden,
            #     identically to step-1. Pass B (hpu_model_runner._tree_verify_decode) already stashed exactly
            #     this in _tree_next_seed[0] = pb_nfh_flat[n_commit-1] (the TARGET forward's output hidden for
            #     the last committed token), the precise analogue of step-1's target_hidden_states[last].
            #   * first_logits == compute_logits(MTP_forward(target_hidden)) -- step-1 derives logits by
            #     running self.model(...) (the MTP layer) on the target hidden THEN compute_logits. We do the
            #     same here: one bs=1 MTP forward fed (seed_token, seed_target_hidden) at seed_pos.
            # (The earlier shortcut compute_logits(raw target hidden) skipped the MTP forward -> wrong root;
            #  the prior over-correction propagated the MTP-OUTPUT hidden -> wrong (draft-internal) seed,
            #  REGRESSING accept. The correct seed is the TARGET hidden, matching step-1.)
            device = seed_target_hidden.device
            # FULL block-table row (not [:1]): a context spanning >1 block (decode pos >= block_size) needs
            # all its committed blocks provisioned, else prepare_attn_metadata under-fetches -> index error.
            bt = model_runner.input_batch.block_table[0].get_cpu_tensor()
            clamped_seed_pos = seed_pos if seed_pos < self.max_model_len else 0
            seed_attn_metadata = self.prepare_attn_metadata(
                bt, torch.tensor([clamped_seed_pos], dtype=torch.int64), model_runner)
            ret = self.model(
                input_ids=torch.tensor([[int(seed_token)]], dtype=torch.int32, device=device),
                positions=torch.tensor([[int(clamped_seed_pos)]], dtype=torch.int64, device=device),
                hidden_states=seed_target_hidden.view(1, 1, -1),
                inputs_embeds=None,
                attn_metadata=seed_attn_metadata,
            )
            last_hs = ret if self.method == "mtp" else ret[0]
            last_hs = last_hs.view(-1, last_hs.shape[-1])
            first_logits = self.model.compute_logits(last_hs[:1])          # [1, V]  (MTP-forward logits)
            return self._propose_tree(
                first_logits=first_logits,
                seed_hidden=seed_target_hidden.view(1, -1),                # the TARGET hidden (step-1 analogue)
                positions=torch.tensor([seed_pos], device=device, dtype=torch.int64),
                # ROOT-TOKEN GEOMETRY FIX: the root carries the COMMITTED token C == seed_token (the prior
                # step's bonus, now committed at seed_pos), NOT the head's argmax.
                root_tokens=torch.tensor([int(seed_token)], device=device, dtype=torch.int64),
                block_table_cpu_tensor=bt,
                model_runner=model_runner,
                batch_size=1,
            )
        # For decode, the virtual batch_size is batch size * num_tokens
        # and the seq_len is always 1
        batch_size = last_token_indices.shape[0]

        # BS>1 TREE FALLBACK: the on-card tree verify + scratch->committed KV commit + re-seed path is
        # single-request by design (ONE dedicated pad/scratch block; a block-diagonal ancestor bias whose
        # CONTEXT columns are NOT cross-request masked; one per-request _tree_next_seed / block_table[0] KV
        # commit -- see _propose_tree's batch_size!=1 guard and the runner's num_decodes==1 consume gate). For
        # a bs>1 decode call engage NEITHER tree branch (the union above already declined; the MTP-tree branch
        # below is gated to batch_size==1) and CLEAR any single-request plan a prior bs=1 step (or the prefill
        # drafter) stashed, so the runner never replays a stale/foreign plan on a later 1-request step. The
        # stock flat drafter below correctly drafts for any batch. Guarded by _TREE_VERIFY -> the default path
        # (tree off) is byte-identical.
        if _TREE_VERIFY and batch_size != 1:
            model_runner._tree_verify_plan = None
            model_runner._tree_verify_trees = None
            model_runner._tree_verify_accept = None
            model_runner._tree_next_seed = None

        if self.method == "eagle3":
            assert isinstance(self.model.model, Eagle3LlamaForCausalLM)
            target_hidden_states = \
                self.model.model.combine_hidden_states(
                    target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        ret_hidden_states = self.model(
            input_ids=target_token_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
            attn_metadata=common_attn_metadata,
        )

        # All MTP related method names are now unified to "mtp"
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1)

        # [num_tokens, 1]
        target_positions = target_positions.view(-1)
        # [batch_size]
        positions = target_positions[last_token_indices]
        if self.method == "mtp":
            hidden_states = target_hidden_states.view(-1, target_hidden_states.shape[-1])
        else:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # [batch_size, hidden_size]
        hidden_states = hidden_states[last_token_indices]

        # TREE-VERIFY branch point (HPU_TREE_VERIFY=1): instead of the single argmax draft chain, expand the
        # native MTP head into a depth-skewed candidate TREE (one batched head call per LEVEL) and stash the
        # bs=T verify plan + accept entry on the runner for the runner's tree-verify-input path to consume.
        # Delegated so the OFF path below is character-for-character the stock hpu_eagle.py. The returned
        # flattened [batch, num_speculative_tokens] draft grid keeps the stock scheduler/verify shape valid
        # regardless of whether the runner's ONCARD KV path is engaged.
        if _TREE_VERIFY and batch_size == 1:
            # NON-RESEED (first / prefill-launched) step: target_token_ids is the eagle-SHIFTED input
            # (target_token_ids[i]==t_{i+1}), so it is NOT the committed C at root_ctx -- signal the runner to
            # source node0 from its token store (token_ids_cpu[0, root_ctx], which IS valid for the prompt
            # position here), NOT from a stashed bonus. (The re-seed branch sets this to the bonus token.)
            model_runner._tree_root_committed_token = None
            return self._propose_tree(
                first_logits=logits,
                seed_hidden=hidden_states,
                positions=positions,
                # The proposer-supplied root token here is the eagle-shifted target token; it is OVERRIDDEN in
                # the runner's _tree_verify_decode with the true committed token at root_ctx (token store on the
                # first step, stashed bonus on re-seed steps). Kept for the OFF/non-ONCARD flat-grid shape.
                root_tokens=target_token_ids.view(-1)[last_token_indices],
                block_table_cpu_tensor=block_table_cpu_tensor,
                model_runner=model_runner,
                batch_size=batch_size,
            )

        # The first draft tokens
        draft_token_ids = logits.argmax(dim=-1)
        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # Positions used by prepare_attn_metadata needs to be cpu because
        # compile only mode for warmup will not do any real computations
        target_positions_cpu = target_positions.cpu()
        positions_cpu = target_positions_cpu[last_token_indices.cpu()]

        # async+spec (#3): HOIST all per-draft attn metadata out of the device-replay loop. For decode
        # the positions advance by a deterministic +1 each draft, independent of the (yet-unsampled)
        # draft token, so every draft's metadata is computable now. This makes the loop body pure replay
        # -- the device stops waiting on the Python host rebuild between draft forwards. Same call, same
        # positions_cpu progression, just done ahead of time -> numerically identical. Gated; stock path
        # (env unset) builds per-iteration exactly as before.
        prebuilt_attn_metadata = None
        if _ASYNC_SPEC:
            prebuilt_attn_metadata = []
            positions_cpu_pre = positions_cpu.clone()
            for _ in range(self.num_speculative_tokens - 1):
                positions_cpu_pre += 1
                prebuilt_attn_metadata.append(
                    self.prepare_attn_metadata(block_table_cpu_tensor, positions_cpu_pre, model_runner))

        # Decode 1 token each time
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            # [batch_size]
            input_ids = draft_token_ids_list[-1].int()

            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Prepare the attn metadata
            positions_cpu += 1
            if prebuilt_attn_metadata is not None:
                # async+spec (#3): pure-replay path -- consume the metadata hoisted above. positions_cpu
                # is still advanced (+1) so the clamp/exceeds_max_model_len logic above is unchanged; we
                # just skip the redundant host rebuild that would otherwise fence the next replay.
                attn_metadata = prebuilt_attn_metadata[token_index]
            else:
                attn_metadata = self.prepare_attn_metadata(block_table_cpu_tensor, positions_cpu, model_runner)

            # [batch_size, 1]
            input_ids = input_ids.view(-1, 1)
            # [batch_size, 1]
            input_positions = clamped_positions.view(-1, 1)
            # [batch_size, 1, hidden_size]
            input_hidden_states = hidden_states.view(-1, 1, hidden_states.shape[-1])
            inputs_embeds = None

            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=input_positions,
                hidden_states=input_hidden_states,
                inputs_embeds=inputs_embeds,
                attn_metadata=attn_metadata,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            # The shape of the returned hidden_states and last_hidden_states:
            # [batch_size, 1, hidden_size]
            # viewed to: [batch_size, hidden_size]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    # ==============================================================================================
    # UNION CHEAP-DRAFTER proposal (HPU_TREE_DRAFTER=union). PLD ∪ ngram build the candidate tree from the
    # committed token sequence ALONE -- ZERO head forward, ZERO lm_head. This is the verify-the-union SS
    # win: drafting is ~free, so the step collapses to ~one bs=Tsum verify forward. Mirrors the MTP path's
    # re-seed vs first/prefill geometry + _tree_root_committed_token signalling EXACTLY, then delegates the
    # PROVEN plan/stash/flat-grid + the runner's bs=Tsum verify/commit (UNCHANGED) via _propose_tree.
    # ==============================================================================================
    def _propose_union(self, target_token_ids, target_positions, last_token_indices,
                       block_table_cpu_tensor, model_runner):
        """Build per-request union trees (PLD ∪ ngram) over the committed token store and delegate to
        _propose_tree(prebuilt_trees=...). NO MTP head forward is run on this path.

        GEOMETRY (mirrors the MTP path so the runner's +1 prefill seat / root-from-committed override /
        verify-commit are byte-identical):
          * RE-SEED step (model_runner._tree_next_seed set by a prior commit): node 0 sits at seed_pos
            (the runner applies NO +1); its committed token is the stashed BONUS (token_ids_cpu lags it).
            We set model_runner._tree_root_committed_token = seed_token so the runner sources node 0 from it.
          * FIRST / prefill-launched step (no stash): the runner applies +1 -> node 0 effectively sits at
            base_pos+1 carrying the just-sampled committed token token_ids_cpu[0, base_pos+1]. We set
            model_runner._tree_root_committed_token = None (so the runner takes its +1 prefill path).
        PLD/ngram operate on the committed sequence ENDING at that effective root, so their predictions are
        the tree's level-1 nodes -- the tree itself is built UN-shifted at base_pos (prompt_len=base_pos+1),
        and the runner applies the +1, IDENTICALLY to the MTP path."""
        try:
            from cheap_drafters import build_union_tree, union_params_from_env
        except ImportError as e:
            raise ImportError(
                "HPU_TREE_DRAFTER=union but mmel/tree_verify/cheap_drafters.py is not importable "
                f"({e}). Expected mmel/tree_verify mounted to /opt/tree_verify with PYTHONPATH=/opt/tree_verify "
                "(see the env's EXTRA_MOUNTS + PYTHONPATH). Refusing to fall back to the MTP head (that would "
                "make the union measure a lie).") from e
        from tree_serve_hook import tree_config_from_env

        enabled, widths, budget = tree_config_from_env()
        if not enabled:
            raise AssertionError(
                "HPU_TREE_DRAFTER=union entered but HPU_TREE_VERIFY is disabled -- the union rides the tree "
                "verify machinery and requires HPU_TREE_VERIFY=1. STOP (LOUD).")
        params = union_params_from_env(len(widths))
        if _DISPATCH_N <= 8:   # stderr proof _propose_union entered + which drafters are active (logger dropped)
            import sys as _s3
            _s3.stderr.write("[UNION-RAN #%d] use_pld=%s use_ngram=%s use_sam=%s use_recycle=%s\n" % (
                _DISPATCH_N, params.get("use_pld"), params.get("use_ngram"),
                params.get("use_sam"), params.get("use_recycle")))
            _s3.stderr.flush()

        device = self.device
        tok_store = model_runner.input_batch.token_ids_cpu      # [max_num_seqs, max_model_len], committed seq
        store_w = int(tok_store.shape[1])

        _next_seed = getattr(model_runner, "_tree_next_seed", None)
        if _next_seed is not None:
            # ---- RE-SEED continuation step (no runner +1) ----
            _seed_hidden, seed_token, seed_pos = _next_seed
            model_runner._tree_root_committed_token = int(seed_token)
            batch_size = 1
            block_table = model_runner.input_batch.block_table[0].get_cpu_tensor()
            base_positions = [int(seed_pos)]
            # history = committed prefix [0, seed_pos) + the bonus (token_ids_cpu[seed_pos] still lags == 0,
            # so we append the stashed bonus, matching the runner's reseed root-from-committed source).
            prefix_end = max(0, min(int(seed_pos), store_w))
            histories = [[int(t) for t in tok_store[0, :prefix_end].tolist()] + [int(seed_token)]]
            root_tokens = [int(seed_token)]
        else:
            # ---- FIRST / prefill-launched step (runner applies +1) ----
            model_runner._tree_root_committed_token = None
            batch_size = int(last_token_indices.shape[0])
            tp = target_positions.view(-1)
            lti = last_token_indices.view(-1)
            ttf = target_token_ids.view(-1)
            base_positions = [int(tp[int(lti[r].item())].item()) for r in range(batch_size)]
            block_table = block_table_cpu_tensor
            histories, root_tokens = [], []
            for r in range(batch_size):
                # ROOT TOKEN = the proposer's in-frame committed token target_token_ids[last_idx] (the
                # eagle-shifted target at base_pos == t_{base_pos+1}, the just-sampled token the runner seats
                # at the +1'd root position; identical to the proven MTP path's root_tokens source, and the
                # runner overrides node 0 with token_ids_cpu[0, base_pos+1] which == it by consume-time). This
                # is robust to the token-store write lag (the store may not hold position base_pos+1 yet at
                # propose-time). HISTORY = the committed prefix [0..base_pos] (the prompt, reliably present)
                # + this root token, so PLD/ngram match the true suffix ending at the effective root.
                idx = int(lti[r].item())
                root_tok = int(ttf[idx].item()) if 0 <= idx < int(ttf.shape[0]) else 0
                pe = max(0, min(base_positions[r] + 1, store_w))             # committed positions 0..base_pos
                hist = [int(t) for t in tok_store[r, :pe].tolist()] + [root_tok]
                histories.append(hist)
                root_tokens.append(root_tok)

        per_req_trees = []
        for r in range(batch_size):
            dt = build_union_tree(
                token_ids=histories[r],
                root_token=root_tokens[r],
                prompt_len=base_positions[r] + 1,      # root at base_pos (UN-shifted); the runner applies +1
                widths=widths,
                budget=budget,
                device=device,
                min_match=params["min_match"],
                max_match=params["max_match"],
                pld_k=params["pld_k"],
                ngram_n=params["ngram_n"],
                pad_to_budget=params["pad_to_budget"],
                use_pld=params["use_pld"],
                use_ngram=params["use_ngram"],
                use_sam=params["use_sam"],             # freeform lift (opt-in HPU_UNION_SAM)
                sam_max_match=params["sam_max_match"],
                sam_b=params["sam_b"],
                use_recycle=params["use_recycle"],     # freeform tail (opt-in HPU_UNION_RECYCLE)
                recycle_n=params["recycle_n"],
                mtp_proposals=None,                    # MTP union member OFF (would re-add a head forward)
            )
            per_req_trees.append(dt)

        positions = torch.tensor(base_positions, device=device, dtype=torch.int64)
        root_tokens_t = torch.tensor(root_tokens, device=device, dtype=torch.int64)
        global _TREE_LOGGED
        if not _TREE_LOGGED:
            logger.info(
                "HPU_TREE_DRAFTER=union ENGAGED: PLD(min=%d,max=%d,k=%d) ∪ ngram(n=%d) ∪ SAM(%s,max=%d,b=%d) "
                "∪ recycle(%s,n=%d) widths=%s budget=%d (ZERO head forward; verify keeps greedy byte-exact).",
                params["min_match"], params["max_match"], params["pld_k"], params["ngram_n"],
                "on" if params["use_sam"] else "off", params["sam_max_match"], params["sam_b"],
                "on" if params["use_recycle"] else "off", params["recycle_n"],
                tuple(widths), budget)
        return self._propose_tree(
            first_logits=None,
            seed_hidden=None,
            positions=positions,
            root_tokens=root_tokens_t,
            block_table_cpu_tensor=block_table,
            model_runner=model_runner,
            batch_size=batch_size,
            prebuilt_trees=per_req_trees,
        )

    # ==============================================================================================
    # TREE-VERIFY proposal (HPU_TREE_VERIFY=1). Build a per-request candidate TREE on the native MTP
    # head and stash the bs=T verify plan + accept entry on the runner; return the flattened draft grid.
    # ==============================================================================================
    def _propose_tree(self, first_logits, seed_hidden, positions, root_tokens,
                      block_table_cpu_tensor, model_runner, batch_size, prebuilt_trees=None):
        """Expand the native MTP head into a depth-skewed candidate tree per request and stash the verify
        plan. Returns [batch_size, num_speculative_tokens] flattened drafts (the stock scheduler/verify
        shape -- the tree nodes' tokens BFS-packed, padded with -1) so the runner's stock path stays valid
        whether or not the ONCARD KV path is engaged.

        Inputs (already computed by propose(), so the FIRST head forward is shared with the stock path):
          first_logits : [batch_size, vocab]   the head's next-token distribution at the root (draft pos-1)
          seed_hidden  : [batch_size, hidden]   the running drafter hidden at the root (committed token's)
          positions    : [batch_size]           the rope position of the root per request (== ctx)
          root_tokens  : [batch_size]           the COMMITTED token C at the root position (NOT a draft).
                                                ROOT-TOKEN GEOMETRY FIX: the tree root carries the already-
                                                committed token so its position (== ctx) is not double-
                                                counted; the root's children are the head's top-b of
                                                first_logits (== the first draft level).

        The head is driven via a per-node closure that reuses the STOCK decode forward (self.model(...) +
        compute_logits) -- the fp8 MTP head is NOT retrained; we only widen WHICH tokens we draft. Each
        request's tree is expanded by tree_serve_hook.build_tree_oncard (ONE batched head call per LEVEL);
        the bs=T verify window + block-diagonal [Tsum,Tsum] ancestor bias are planned by
        build_tree_decode_batch; the accept entry accept_tree_batch is stashed for the runner's sampler.
        """
        # LAZY import: the tree helpers (mmel/tree_verify, mounted to /opt/tree_verify, PYTHONPATH set by
        # the env) are imported ONLY here, inside the gated branch, so importing this module on a box
        # without /opt/tree_verify on sys.path (py_compile / a stock serve) still succeeds.
        try:
            from tree_serve_hook import (
                accept_tree_batch,
                build_tree_decode_batch,
                build_tree_oncard,
                tree_config_from_env,
                tree_spec_metadata_fields,
            )
        except ImportError as e:
            # LOUD: HPU_TREE_VERIFY=1 was set but the helper package is not importable -> a missing mount /
            # PYTHONPATH. Refuse to silently fall back to the stock chain (which would make the whole tree
            # measure a lie -- the exact trap tree_measure.sh's proposer-mounted guard exists to prevent).
            raise ImportError(
                "HPU_TREE_VERIFY=1 but the tree_verify helpers are not importable "
                f"({e}). Expected mmel/tree_verify mounted to /opt/tree_verify with PYTHONPATH=/opt/tree_verify "
                "(see glm52-mtp-tree.env EXTRA_MOUNTS + PYTHONPATH). Refusing to fall back to the stock "
                "depth-3 chain (that would make the tree measure a lie).") from e

        # Parse (and LOUD-validate) the width schedule + budget. tree_config_from_env raises on a malformed
        # HPU_TREE_WIDTHS / non-positive HPU_TREE_BUDGET -- a typo must surface, not silently default.
        enabled, widths, budget = tree_config_from_env()
        # enabled mirrors _TREE_VERIFY (both read HPU_TREE_VERIFY); assert they agree so a mid-run env race
        # cannot put us on the tree code path with the config parser disagreeing.
        if not enabled:
            raise AssertionError(
                "HPU_TREE_VERIFY disagreement: the proposer entered the tree branch (_TREE_VERIFY=1) but "
                "tree_config_from_env() reports disabled. The env is inconsistent -- refusing to proceed.")
        # num_speculative_tokens must cover the budget (the scheduler must schedule >= T spec slots so the
        # flattened draft grid below is wide enough). Mirror tree_runner_config's assert at the proposer so
        # a too-small nst FATALs here too (LOUD), not silently truncating the tree.
        nst = int(self.num_speculative_tokens)
        if nst < budget:
            raise AssertionError(
                f"HPU_TREE_VERIFY: num_speculative_tokens={nst} < HPU_TREE_BUDGET(T)={budget}. The flattened "
                f"draft grid would be narrower than the {budget}-node tree -> the verify window cannot hold "
                f"the tree. Set the speculative-config num_speculative_tokens >= {budget}.")

        # device: first_logits is None on the UNION (prebuilt_trees) path (no head forward ran) -> use the
        # proposer's model device. On the MTP path first_logits is the head's [batch,vocab] -> .device is the
        # model device too, so this is byte-identical for MTP.
        device = first_logits.device if first_logits is not None else self.device
        block_size = model_runner.block_size

        # ---- per-request head closure (reuses the STOCK decode forward) ------------------------------
        # build_tree_oncard wants mtp_head_step(parent_hidden, parent_token, position) -> (logits[V], hidden[H]).
        # We drive ONE request at a time (batch_size is the single-stream measure's 1; the loop generalizes
        # to bs>1). Each call is a bs=1 decode forward feeding the parent token at its rope position with the
        # parent's hidden -- IDENTICAL to one iteration of the stock chain's loop, only the fed token varies.
        def _make_head_step(req_idx):
            seed_h = seed_hidden[req_idx]                  # [H]
            base_pos = int(positions[req_idx].item())      # rope position of the root (ctx)

            def mtp_head_step(parent_hidden, parent_token, position):
                # parent_hidden : [H] the head's running hidden at the parent node.
                # parent_token  : int the parent node's token (fed back as the next input).
                # position      : int the rope position of THIS node (parent_pos + 1; root uses ctx).
                clamped = position if position < self.max_model_len else 0
                input_ids = torch.tensor([[int(parent_token)]], dtype=torch.int32, device=device)
                input_positions = torch.tensor([[int(clamped)]], dtype=torch.int64, device=device)
                input_hidden = parent_hidden.view(1, 1, -1)
                # Tree drafting MUST read the head off the COMMITTED context KV only (the candidate nodes are
                # NOT committed during proposal). BUG (root cause of the scramble): the stock helper called
                # with THIS node's position makes the forward (a) WRITE its MTP KV to the REAL committed slot
                # for `clamped` (block_table.gather + clamped%block_size, prepare_attn_metadata:641,646) and
                # (b) READ context up to `clamped` (num_blocks=ceil((pos+1)/bs):624) -- so sibling/ancestor
                # speculative writes at base_pos+1.. land in REAL committed slots and are then read back by
                # deeper nodes (last-sibling-wins) -> the draft conditions on a self-polluted, position-
                # shuffled KV -> "real content, wrong order" output (confirmed on-card: commit positions are
                # perfect but tokens scramble + degenerate). FIX (POLLUTION-FREE drafting, HPU_TREE_POLLUTION_FIX,
                # default ON under the tree path): build the read metadata at the COMMITTED root position
                # (base_pos) so every node reads ONLY the committed prompt KV (identical committed context for
                # all nodes -- the candidate's own token enters via input_ids, its position via rope), and
                # REDIRECT the MTP-KV write to the dedicated PAD block so speculative writes NEVER touch (or
                # read) a committed slot. RoPE still uses the node's true `clamped` position (input_positions).
                _poll_fix = os.environ.get("HPU_TREE_POLLUTION_FIX", "1").strip() in _TRUTHY
                read_pos = base_pos if _poll_fix else clamped
                pos_cpu = torch.tensor([read_pos], dtype=torch.int64)
                attn_metadata = self.prepare_attn_metadata(block_table_cpu_tensor, pos_cpu, model_runner)
                if _poll_fix:
                    # Redirect this node's KV WRITE to the reserved pad block (scratch) so it cannot overwrite
                    # any committed slot. The pad block's KV is never read by a committed-context forward
                    # (read metadata above spans only base_pos -> the committed prompt blocks), so the write is
                    # a harmless throwaway. slot = _PAD_BLOCK_ID*block_size + (clamped % block_size).
                    # HPUAttentionMetadataV1 is a (mutable) @dataclass -> assign the field directly (no
                    # namedtuple._replace). The original slot_mapping from prepare_attn_metadata is [1,1].
                    _pad_slot = int(model_runner._PAD_BLOCK_ID) * block_size + (clamped % block_size)
                    attn_metadata.slot_mapping = torch.tensor(
                        [[_pad_slot]], dtype=torch.int64, device=self.device)
                # TRACE (gated, ALL ranks so it always surfaces): the ACTUAL WRITE slot this node uses (after
                # the pollution-fix redirect). PAD/scratch == fixed; REAL-COMMITTED == still pollutes.
                if os.environ.get("HPU_TREE_TRACE", "0").strip() in _TRUTHY:
                    _n = getattr(self, "_tree_trace_calls", 0)
                    if _n < 12:
                        self._tree_trace_calls = _n + 1
                        _bsz = model_runner.block_size
                        _pad_blk = int(model_runner._PAD_BLOCK_ID)
                        _write_slot = int(attn_metadata.slot_mapping.view(-1)[0].item())
                        _wblk = _write_slot // _bsz
                        logger.info("[TREE-TRACE child] node_pos=%d read_root_pos=%d tok=%d -> WRITE slot=%d "
                                    "block=%d (%s)", clamped, base_pos, int(parent_token), _write_slot, _wblk,
                                    "PAD/scratch (pollution-fixed)" if _wblk == _pad_blk else "REAL-COMMITTED (pollutes!)")
                ret = self.model(
                    input_ids=input_ids,
                    positions=input_positions,
                    hidden_states=input_hidden,
                    inputs_embeds=None,
                    attn_metadata=attn_metadata,
                )
                if self.method == "mtp":
                    last_hs = ret
                    out_hs = ret
                else:
                    last_hs, out_hs = ret
                last_hs = last_hs.view(-1, last_hs.shape[-1])
                out_hs = out_hs.view(-1, out_hs.shape[-1])
                node_logits = self.model.compute_logits(last_hs[:1]).view(-1)   # [V]
                node_hidden = out_hs[:1].view(-1)                               # [H]
                return node_logits, node_hidden

            return mtp_head_step, seed_h, base_pos

        # ROOT-TOKEN GEOMETRY FIX (BUILD-1 Step-1): the tree root carries the COMMITTED token C (threaded in
        # as root_tokens), NOT first_logits.argmax (a DRAFT). The root sits at the committed position (== ctx);
        # seeding it with the head's argmax made the bs=Tsum verify forward place a DRAFT token at the already-
        # committed slot, so position ctx was double-counted (the root re-derives ctx's KV into scratch AND the
        # committed context window also covered it) -> attention saw position ctx twice -> over-weighted ->
        # corruption. With root_token == C the build_tree_oncard level-0 head call mtp_head_step(seed_h, C, ctx)
        # REPRODUCES first_logits, so the root's children are exactly its top-b == the first draft level
        # (free + correct; the optional micro-opt of passing first_logits in to skip the recompute is not taken).
        if prebuilt_trees is not None:
            # UNION (cheap-drafter) path: the per-request trees were ALREADY built host-side by
            # _propose_union (PLD ∪ ngram, ZERO head forward); reuse them verbatim and FALL THROUGH to the
            # SAME proven plan/stash/flat-grid code below -> the runner's bs=Tsum verify/commit is UNCHANGED.
            # The MTP head closure defined above is never invoked on this path.
            per_req_trees = list(prebuilt_trees)
            if len(per_req_trees) != batch_size:
                raise AssertionError(
                    f"HPU_TREE_VERIFY union: prebuilt_trees count {len(per_req_trees)} != batch_size "
                    f"{batch_size}. STOP (LOUD).")
        else:
            if root_tokens is None:
                raise AssertionError(
                    "HPU_TREE_VERIFY: _propose_tree requires the per-request COMMITTED root token (root_tokens); "
                    "the argmax-root path was removed by the geometry fix. STOP (LOUD).")
            per_req_trees = []
            for r in range(batch_size):
                head_step, seed_h, base_pos = _make_head_step(r)
                # tree_position_ids uses prompt_len-1+depth; build_tree_oncard seeds the root at prompt_len-1.
                # base_pos IS ctx == prompt_len-1, so prompt_len = base_pos + 1.
                dt = build_tree_oncard(
                    mtp_head_step=head_step,
                    seed_hidden=seed_h,
                    root_token=int(root_tokens[r].item()),
                    prompt_len=base_pos + 1,
                    widths=widths,
                    t_budget=budget,
                    device=device,
                )
                per_req_trees.append(dt)

        # ---- plan the bs=T verify window + stash it + the accept entry for the runner ----------------
        # CONTEXT blocks come from the request's REAL committed block table (its prompt KV, which always
        # exists). SCRATCH blocks (where the T throwaway tree-verify KV is written + read THIS forward and
        # then discarded) come from the runner's DEDICATED dummy/pad block, NOT from the per-request block
        # table's lookahead region.
        #
        # WHY NOT slice the lookahead blocks out of the block table (the old code did
        #   scratch = row[ctx_blocks_needed : ctx_blocks_needed+need]):
        # The vLLM V1 scheduler DOES allocate num_lookahead_tokens (= num_speculative_tokens) blocks into the
        # request, BUT it caps num_tokens_need_slot at max_model_len (kv_cache_manager.allocate_slots:
        # num_tokens_need_slot = min(ctx+1 + num_lookahead, max_model_len)). With max-model-len 512 /
        # block-size 128 the row is only ceil(512/128)=4 blocks wide; once the decode position grows so that
        # ceil((P+1)/block_size) reaches the saturated 4-block allocation, the slice row[ctx_blocks_needed:]
        # is EMPTY -> need(1) scratch blocks missing -> build_tree_decode_batch returns ok=False every step
        # -> the runner SKIPS the tree verify -> no [TREE-ACCEPT] is ever emitted. Bumping
        # num_speculative_tokens (48->128) does NOT fix it: the lookahead is still clamped by max_model_len
        # and still slips behind the context as generation advances. (Verified against the on-card
        # vllm-glm52-mtp-tree log: "plan ok=False ... Tsum=45, block_size=128".)
        #
        # THE DEDICATED DUMMY BLOCK is the clean measure-only scratch: the HPU runner allocates the KV cache
        # tensor with num_blocks+1 blocks (hpu_model_runner.py:6389 target_shape=(num_blocks+1,*shape); the
        # "reserved for KV cache dummy block" in the worker log) and reserves the last block index
        # (_PAD_BLOCK_ID == num_blocks) as a standing pad/overflow region that KVCacheManager NEVER hands to
        # any request (it only allocates ids 0..num_blocks-1). It is a REAL, writable+readable physical block
        # of block_size(=128) slots -- the same region the runner already routes padded-decode tokens into
        # (hpu_model_runner.py:2956-2969). A T<=budget(45)<=128 single-request tree fits in ONE such block:
        # nodes 0..T-1 -> slots _PAD_BLOCK_ID*block_size + 0..T-1 (all < +block_size), written then read by
        # the intra-tree ancestor bias within the SAME bs=Tsum forward, then discarded (measure-only -- the
        # KV is never committed, so it need not survive the step; see the runner's KVCOMMIT boundary).
        #
        # CORRECTNESS / SCOPE: there is exactly ONE pad block, so all requests pointing scratch at it would
        # COLLIDE. That is acceptable here and ONLY here: the runner's tree consume path already hard-asserts
        # single-request (n_trees != 1 raises, hpu_model_runner.py:6881-6887) and the single-stream measure
        # runs batch_size==1. We MIRROR that bound LOUDLY below so a bs>1 tree (which needs a distinct scratch
        # block per request) can never silently corrupt KV by sharing the one pad block. The context prefix is
        # untouched (scratch is appended AFTER ctx_blocks), so the runner's ctx-only slice stays correct.
        pad_block_id = int(model_runner._PAD_BLOCK_ID)
        if batch_size != 1:
            raise AssertionError(
                f"HPU_TREE_VERIFY: dedicated-dummy-block scratch is single-request only (batch_size={batch_size}). "
                "There is exactly ONE reserved pad block (_PAD_BLOCK_ID); sharing it across requests would "
                "collide their tree KV. The runner's tree consume path is likewise single-request gated. Bring "
                "up a per-request scratch-block allocation before running a bs>1 tree verify.")
        per_req_ctx = [int(positions[r].item()) for r in range(batch_size)]
        per_req_ctx_blocks = []
        per_req_scratch = []
        bt = block_table_cpu_tensor
        for r in range(batch_size):
            T_r = per_req_trees[r].T
            need = (T_r + block_size - 1) // block_size
            ctx_blocks_needed = (per_req_ctx[r] + 1 + block_size - 1) // block_size
            row = bt[r].tolist() if r < bt.shape[0] else []
            ctx_blocks = row[:ctx_blocks_needed]
            # SCRATCH = the dedicated reserved dummy/pad block(s). For T<=budget<=block_size this is exactly
            # ONE block (need==1); we replicate it to `need` ids defensively so a future T>block_size still
            # passes build_tree_verify_layout's len(scratch)>=need check (all nodes still land in the one pad
            # block via slot = pad_block*block_size + (i % block_size); for T>block_size the high nodes would
            # alias the low ones -- guarded out by the T<=budget<=block_size invariant, asserted next).
            if T_r > block_size:
                raise AssertionError(
                    f"HPU_TREE_VERIFY: tree T={T_r} exceeds block_size={block_size}; the single dedicated pad "
                    f"block holds only block_size scratch slots. Lower HPU_TREE_BUDGET to <= {block_size} or "
                    "allocate a multi-block dedicated scratch region.")
            scratch = [pad_block_id] * need
            per_req_ctx_blocks.append(ctx_blocks)
            per_req_scratch.append(scratch)

        tree_plan = build_tree_decode_batch(
            per_req_trees=per_req_trees,
            per_req_ctx=per_req_ctx,
            per_req_ctx_blocks=per_req_ctx_blocks,
            per_req_scratch=per_req_scratch,
            block_size=block_size,
            pad_slot_id=model_runner._PAD_SLOT_ID,
        )
        spec_fields = tree_spec_metadata_fields(per_req_trees, batch_size)

        # Stash on the runner for the (HPU_TREE_VERIFY_ONCARD-gated) tree-verify-input path to consume. The
        # accept entry is the bound helper (the runner calls it with the bs=Tsum verify logits + cu_seqlens).
        model_runner._tree_verify_plan = tree_plan
        model_runner._tree_verify_spec_fields = spec_fields
        model_runner._tree_verify_trees = per_req_trees
        model_runner._tree_verify_accept = accept_tree_batch
        # ADAPTIVE FALLBACK (HPU_TREE_MIN_ACCEPT_NODES>0, default 0=OFF/byte-identical): if the drafter
        # produced few real (non-filler) nodes -- the freeform signal -- the bs=Tsum verify (~2x a plain
        # decode) will not pay off. Mark the plan WEAK so the runner's _tree_consume_active skips the verify
        # and does a plain bs=1 decode this step (reusing the proven ok=False stock-fallback path). Real nodes
        # carry a normal (small) cumulative logprob; filler carries _FILLER_LP (-1e30) -> threshold at -1e29.
        _min_real = int((os.environ.get("HPU_TREE_MIN_ACCEPT_NODES", "0") or "0").strip() or "0")
        if _min_real > 0:
            _lp = per_req_trees[0].logprob
            _n_real = int((_lp > -1.0e29).sum().item()) - 1   # non-filler nodes, minus the committed root
            tree_plan["weak"] = bool(_n_real < _min_real)
        else:
            tree_plan["weak"] = False
        if not tree_plan["ok"]:
            # LOUD: a request did not get enough scratch blocks for its T_r nodes -> the verify window would
            # overflow into the pad slot (the verify would be wrong). Surface it; the runner's ONCARD path
            # must refuse to run a not-ok plan.
            logger.warning(
                "HPU_TREE_VERIFY: tree verify plan ok=False -- a request lacks scratch blocks for its tree "
                "(Tsum=%d, block_size=%d). The verify window would overflow to the pad slot; the runner's "
                "ONCARD path must skip this step. Check num_speculative_tokens / lookahead allocation.",
                int(tree_plan["cu_seqlens"][-1]), block_size)

        global _TREE_LOGGED
        if not _TREE_LOGGED:
            _TREE_LOGGED = True
            logger.info(
                "HPU_TREE_VERIFY=1 ENGAGED in proposer: widths=%s budget(T)=%d nst=%d batch=%d Tsum=%d "
                "(native fp8 MTP head, one batched call per level; OFF=byte-identical to depth-3 chain).",
                tuple(widths), budget, nst, batch_size, int(tree_plan["cu_seqlens"][-1]))

        # ---- flatten the trees to the [batch, num_speculative_tokens] draft grid ---------------------
        # The stock scheduler/verify consumes a flat [batch, nst] draft grid. We pack each request's tree
        # node tokens in BFS order (EXCLUDING the root, which is the already-committed token) and pad the
        # remainder with -1 (PLACEHOLDER_TOKEN_ID) up to nst. Under HPU_TREE_VERIFY_ONCARD the runner uses
        # the stashed tree_plan instead of this flat grid for the verify; this grid keeps the stock path
        # shape-valid (and is what a non-ONCARD run drafts) regardless.
        draft_grid = torch.full((batch_size, nst), -1, dtype=torch.int64, device=device)
        for r in range(batch_size):
            toks = per_req_trees[r].tokens                 # [T_r], node 0 == committed root
            n_draft = min(int(toks.shape[0]) - 1, nst)     # exclude the root; cap at nst
            if n_draft > 0:
                draft_grid[r, :n_draft] = toks[1:1 + n_draft]
        return draft_grid

    def prepare_inputs(
        self,
        common_attn_metadata,
        spec_decode_metadata: SpecDecodeMetadata,
        sampled_token_ids: list[list[int]],
        num_accepted=None,
    ):
        assert spec_decode_metadata is not None
        num_draft_tokens = \
            spec_decode_metadata.num_draft_tokens
        max_num_draft_tokens = max(num_draft_tokens)

        num_picked_token_indices = []
        last_token_indices = []
        starting_index = 0
        if num_accepted is not None:
            # async+spec (HPU_ASYNC_SPEC=1, b2): per-req accepted-token counts come from the device
            # tensor valid_count (= accepted + 1) instead of len(sampled_token_ids[i]) — the host list
            # is empty on the async path (parse_output is deferred to AsyncHPUModelRunnerOutput.get_output).
            # The single .tolist() here is the DRAFTER's own sync (off the target-model output critical
            # path we are overlapping), not the per-step rejection-sampler host sync we removed.
            # num_rejected = (n + 1) - valid_count, matching the stock formula with
            # len(sampled_token_ids[i]) == valid_count[i].
            valid_count = num_accepted.tolist()
            num_rejected_tokens = [
                n + 1 - valid_count[i] if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
            ]
        else:
            num_rejected_tokens = [
                n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
            ]
        for i, n in enumerate(num_draft_tokens):
            r = num_rejected_tokens[i]
            step = max_num_draft_tokens + 1
            for j in range(step):
                if j == n - r:
                    last_token_indices.append(starting_index + j)
                if j < n + 1 - r:
                    num_picked_token_indices.append(starting_index + j)
                else:
                    num_picked_token_indices.append(-1)
            starting_index += step
        hidden_states_indices = torch.tensor(num_picked_token_indices, device=self.device)
        last_token_indices = torch.tensor(last_token_indices, device=self.device)
        return common_attn_metadata, hidden_states_indices, last_token_indices

    def prepare_attn_metadata(
            self,
            # [num_seq, total_blocks]
            block_table_cpu_tensor,
            # CPU tensor: [batch_size]
            positions,
            model_runner):
        # Prepare attn metadata on CPU. (Improve for pure HPU based attn metadata preparation)
        block_size = model_runner.block_size
        batch_size = positions.shape[0]
        exceeds_max_model_len = positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

        # Note: block_table_cpu_tensor doesn't include the padding
        # which might smaller than the (padded) batch_size
        num_seq = block_table_cpu_tensor.shape[0]

        # Prepare block tables list
        # block_tables_list is a nested list of shape [num_seq, num_blocks]
        # num_blocks should include the slots needed for the current token
        # positions are the context lengths, and we need +1 for num_blocks
        num_blocks = torch.ceil((positions + 1) / block_size).int()
        num_blocks = num_blocks[:num_seq].tolist()
        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.append(seq_block_table)
        # Needs to be resolved by defragmenter
        block_tables_list = model_runner.defragmenter.resolve_all(block_tables_list)

        # Compute slot mapping in [batch_size, 1] shape
        clamped_positions = clamped_positions.view(-1, 1)
        block_numbers = clamped_positions // block_size

        # Limit with num_seq because block_table_cpu_tensor is in the shape [num_seq, x]
        block_numbers = block_numbers.to(torch.int64)[:num_seq]
        block_ids = torch.ones((batch_size, 1), dtype=torch.int32) * model_runner._PAD_BLOCK_ID
        block_ids[:num_seq] = block_table_cpu_tensor.gather(dim=1, index=block_numbers)
        # Needs to be resolved by defragmenter
        block_ids.apply_(model_runner.defragmenter.resolve)

        # Calculate the slot mapping and fill with padding
        slot_mapping = block_ids * block_size + clamped_positions % block_size
        dummy_slots = itertools.cycle(range(model_runner._PAD_SLOT_ID, model_runner._PAD_SLOT_ID + block_size))
        slot_mapping[num_seq:].apply_(lambda _, ds=dummy_slots: next(ds))
        # Slot mapping needs to be int64 (long) type
        slot_mapping = slot_mapping.to(torch.int64)

        block_list, block_groups, block_usage = \
            model_runner.get_habana_paged_attn_buffers(
                block_tables_list,
                slot_mapping.tolist(),
                batch_size
            )

        block_list_device = async_h2d_copy(block_list, device=self.device)
        block_usage_device = async_h2d_copy(block_usage, device=self.device)
        block_groups_device = async_h2d_copy(block_groups, device=self.device)
        slot_mapping_device = async_h2d_copy(slot_mapping, device=self.device)

        common_attn_metadata = HPUAttentionMetadataV1.make_decode_metadata(
            block_list=block_list_device,
            block_usage=block_usage_device,
            block_groups=block_groups_device,
            input_positions=None,
            slot_mapping=slot_mapping_device,
            block_size=block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            chunked_block_list=None,
            chunked_block_usage=None,
            chunked_block_groups=None,
        )

        return common_attn_metadata
