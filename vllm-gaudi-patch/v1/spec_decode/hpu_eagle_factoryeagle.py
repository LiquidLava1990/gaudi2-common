# SPDX-License-Identifier: Apache-2.0
# PATCHED COPY (factory-eagle head bring-up): identical to the canonical vllm_gaudi hpu_eagle.py
# (multi-cand + async-spec features intact) EXCEPT the method=='eagle3' isinstance assert is relaxed
# to duck-type our FactoryEagleForCausalLM draft (see the inline RELAXATION note). Mounted over
# /opt/vllm-gaudi/vllm_gaudi/v1/spec_decode/hpu_eagle.py by the glm52-eagle*.env files.
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

# One-shot guard so the multi-candidate out-of-range-position warning is LOUD but not spammed every step.
_MC_BAD_POS_WARNED = False

# async+spec (#3): when HPU_ASYNC_SPEC=1 the multi-draft decode loop hoists ALL per-draft attention
# metadata to BEFORE the device-replay loop. For decode each draft advances positions by exactly +1,
# so every draft's block_table/slot_mapping is known up front (no draft-token dependence). Building it
# ahead of time turns the loop body into pure HPU-graph replay -- the device no longer stalls on the
# Python host rebuild (.tolist()/.apply_()/defragmenter/4x h2d) between replays. Default (env unset)
# keeps the original per-iteration build -> byte-identical to stock. Numerics are IDENTICAL: the same
# prepare_attn_metadata is called on the same positions_cpu sequence, only the call site moves earlier.
_ASYNC_SPEC = os.environ.get("HPU_ASYNC_SPEC", "0") == "1"

# ==================================================================================================
# MULTI-CANDIDATE DRAFT CHAINS  (HPU_MULTI_CAND=<k>, default 1 = byte-identical single chain)
#
# The ACCEPT lever (root-cause memory glm52-single-stream-rootcause: at the fixed ~137ms bs>1 MTP-verify
# step, tok/s = accept-len / 0.137 -> accept 2.09=15.3, accept 5=36.5; the HEAD/accept is the 40+ lever).
# Native MTP-d3 = accept-len 2.09 with per-position accept 0.69/0.31/0.09. The pos-1 miss (0.31) is the
# dominant loss: when the head's top-1 is wrong but the true token is in its top-k, a multi-candidate
# draft recovers it. So at the FIRST draft position we take the head's top-k (not top-1) -> k candidate
# tokens -> k INDEPENDENT greedy chains, each a normal causal sequence continued to depth d. ALL k chains
# verify in ONE bs=k*(d) batch (reusing the int4 bs>1 verify, HPU_MOE_ROUTED_BSGT1) and we accept the
# chain with the LONGEST matching prefix. NO tree-attention kernel: each chain is causal and gets its own
# verify row + its own private KV scratch block, so standard per-row paged attention isolates the chains
# (no cross-chain mask). The KV layout is the hard part -- see propose() and the runner's multi-cand
# verify path (HPU_MULTI_CAND in hpu_model_runner.py) for the block-isolation design.
#
# k=1 -> the proposer returns the SAME [batch, d] argmax chain as stock (the topk(1) is argmax) so the
# whole feature is OFF by default and byte-identical. k>1 returns [batch, k, d].
#
# CONFIG CONTRACT (k>1): the speculative_config's num_speculative_tokens carries the FLATTENED draft count
# k*depth (so the vLLM scheduler schedules k*depth draft tokens for the verify). The per-chain draft DEPTH
# is HPU_MULTI_CAND_DEPTH (default 1 = the highest-value depth-1 fan-out: top-k at the dominant-loss pos-1).
# We REQUIRE num_speculative_tokens == k * depth and assert it once at first use (LOUD). Example:
# k=4, depth=1 -> num_speculative_tokens=4 ; k=4, depth=2 -> num_speculative_tokens=8.
# ==================================================================================================
_MULTI_CAND = max(1, int(os.environ.get("HPU_MULTI_CAND", "1")))
_MULTI_CAND_DEPTH = max(1, int(os.environ.get("HPU_MULTI_CAND_DEPTH", "1")))


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
        # For decode, the virtual batch_size is batch size * num_tokens
        # and the seq_len is always 1
        batch_size = last_token_indices.shape[0]

        if self.method == "eagle3":
            # FACTORY-EAGLE RELAXATION (off-card head bring-up): the stock assert demands the draft be a
            # vLLM-native Eagle3LlamaForCausalLM. Our from-scratch FactoryEagleForCausalLM is NOT that class
            # but it IS duck-type compatible: self.model.model (a _FactoryInner) exposes combine_hidden_states
            # (per-aux RMSNorm + fc 3H->H = the trained EAGLE-3 fusion) and self.model exposes
            # forward()/compute_logits() with the same signatures. Accept EITHER the native class OR any object
            # whose .model.model provides a callable combine_hidden_states. LOUD if neither (a real wiring bug
            # must surface, not silently skip the aux fusion -> the head would get an unfused [T,3H] tensor).
            _inner = self.model.model
            _native = isinstance(_inner, Eagle3LlamaForCausalLM)
            _ducktype = callable(getattr(_inner, "combine_hidden_states", None))
            if not (_native or _ducktype):
                raise AssertionError(
                    f"eagle3 draft {type(_inner).__name__} is neither Eagle3LlamaForCausalLM nor exposes a "
                    f"callable combine_hidden_states; cannot fuse the 3-layer aux. Check the draft ckpt arch.")
            target_hidden_states = \
                _inner.combine_hidden_states(
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

        # MULTI-CANDIDATE branch point (HPU_MULTI_CAND=k>1): instead of the single argmax first-draft, fan
        # out to k candidates from the SAME first-position distribution and continue k greedy chains. We do
        # this with the [batch, hidden] state and logits we already computed -- the first drafter forward is
        # shared by all k chains (they only diverge AT the first token). Delegated to keep this method's
        # k==1 path byte-identical. Returns [batch, k, d].
        if _MULTI_CAND > 1:
            return self._continue_multi_cand_chains(
                logits, hidden_states, positions, block_table_cpu_tensor, model_runner, batch_size)

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

    def _continue_multi_cand_chains(self, first_logits, hidden_states, positions,
                                    block_table_cpu_tensor, model_runner, batch_size):
        """MULTI-CANDIDATE (HPU_MULTI_CAND=k>1): fan k chains out of the FIRST-position distribution and
        continue each greedily to depth d = self.num_speculative_tokens. Returns [batch_size, k, d].

        Inputs (already computed by propose()):
          first_logits   : [batch_size, vocab]  -- the head's full next-token distribution at draft pos-1
          hidden_states  : [batch_size, hidden] -- the running drafter state to seed every chain (the SAME
                           state for all k chains; they diverge only by which top-k token is fed back)
          positions      : [batch_size]         -- the rope position of draft pos-1 per request (== ctx)

        Chains are batched on a [batch_size * k] virtual axis (chain c of req r at row r*k + c). The drafter
        forward (MLA + MoE) runs once per remaining depth over all batch_size*k rows. KV ISOLATION: each
        chain writes its depth-j draft to a DISTINCT scratch slot (multi_cand=k in prepare_attn_metadata
        offsets chain c into its own lookahead sub-region) while keeping the TRUE rope position, so chain c
        attends only to context + its own earlier drafts -- no cross-chain leak, no tree mask. d==1 needs no
        continuation (the top-k tokens ARE the depth-1 drafts).
        """
        k = _MULTI_CAND
        d = _MULTI_CAND_DEPTH
        # LOUD config-contract check (k>1): num_speculative_tokens (the spec lookahead) must be at LEAST
        # k*depth so the scheduler schedules enough draft slots for the k flattened chains AND -- crucially
        # -- allocates >= k KV scratch BLOCKS past the context for per-chain isolation (the drafter writes
        # chain c to scratch block num_ctx+c; with < k scratch blocks the chains would collide). Set
        # num_speculative_tokens = k * block_size to guarantee k whole scratch blocks (the verify/drafter
        # layout helpers assert ok and raise LOUD if a request still lacks k scratch blocks). Fail at first
        # use, not silently.
        if self.num_speculative_tokens < k * d:
            raise ValueError(
                f"HPU_MULTI_CAND={k} x HPU_MULTI_CAND_DEPTH={d} = {k * d} > "
                f"num_speculative_tokens={self.num_speculative_tokens}. Set the speculative_config "
                f"num_speculative_tokens >= k*depth (recommend k*block_size for k isolated scratch blocks).")
        device = first_logits.device

        # Depth-1: the k candidate tokens per request = the head's top-k.  [batch_size, k]
        first_topk = first_logits.topk(k, dim=-1).indices.to(torch.int64)  # [batch, k]

        if d == 1:
            # No continuation; each candidate is a length-1 chain.  [batch, k, 1]
            return first_topk.view(batch_size, k, 1)

        # Seed the [batch*k] chains. Row r*k + c carries req r, candidate c.
        # Per-chain running hidden state: replicate the shared first-position state across the k candidates.
        # [batch, hidden] -> [batch, k, hidden] -> [batch*k, hidden]
        hs = hidden_states.unsqueeze(1).expand(batch_size, k, hidden_states.shape[-1]).reshape(batch_size * k, -1)
        # Per-chain rope position (all chains of a req share the same progression; advances +1 each depth).
        # [batch] -> [batch*k]
        pos = positions.unsqueeze(1).expand(batch_size, k).reshape(-1).clone()
        # CPU mirror for prepare_attn_metadata (compile-only warmup does no real compute -> must be cpu).
        pos_cpu = pos.cpu()

        # Per-depth drafts. depth-0 token = the top-k candidate (already chosen, fed back at depth-1's fwd).
        # [batch*k] depth-0 input token = the candidate token for that row.
        cur_tokens = first_topk.reshape(-1)  # [batch*k]
        draft_cols = [cur_tokens.view(batch_size, k)]  # list of [batch, k], one per depth; col0 = top-k

        for token_index in range(d - 1):
            # At this iter the INPUT token is each chain's depth-`token_index` draft (col token_index of
            # draft_cols), fed at rope position ctx+token_index+1, producing the depth-`token_index+1` token.
            # The input token's KV is written to its chain's scratch block at within-block offset
            # `write_offset = token_index` (chain c's d drafts occupy offsets 0..d-1 of scratch block c).
            input_ids = cur_tokens.int().view(-1, 1)              # [batch*k, 1]
            pos += 1
            pos_cpu += 1
            exceeds = pos >= self.max_model_len
            clamped = torch.where(exceeds, 0, pos)
            attn_metadata = self._prepare_attn_metadata_multi_cand(
                block_table_cpu_tensor, pos_cpu, model_runner, k, token_index, batch_size)
            input_positions = clamped.view(-1, 1)                 # [batch*k, 1]
            input_hidden = hs.view(-1, 1, hs.shape[-1])           # [batch*k, 1, hidden]
            ret = self.model(input_ids=input_ids, positions=input_positions,
                             hidden_states=input_hidden, inputs_embeds=None, attn_metadata=attn_metadata)
            if self.method == "mtp":
                last_hs = ret
                hs = ret
            else:
                last_hs, hs = ret
            last_hs = last_hs.view(-1, last_hs.shape[-1])
            hs = hs.view(-1, hs.shape[-1])
            hs = hs[:batch_size * k]
            logits = self.model.compute_logits(last_hs[:batch_size * k])
            cur_tokens = logits.argmax(dim=-1)                    # [batch*k] greedy continuation per chain
            draft_cols.append(cur_tokens.view(batch_size, k))

        # [batch, k, d]
        return torch.stack(draft_cols, dim=2)

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

        if num_accepted is not None:
            # async+spec (HPU_ASYNC_SPEC=1) CADENCE-CUT (B): build hidden_states_indices /
            # last_token_indices ENTIRELY ON-DEVICE from the device tensor valid_count (= accepted+1).
            # This removes the per-step valid_count.tolist() host sync (B1) that drained the device
            # pipeline every step -> no cross-step overlap. The arithmetic is byte-identical to the
            # Python loop below: with r = num_rejected = (n+1) - valid_count (when n>0, else r=0),
            #   step = max_num_draft_tokens + 1,  base[i] = i * step,
            #   vc_eff[i] = valid_count[i] if n>0 else n+1  (== n+1-r),
            #   hidden_states_indices[i*step + j] = base[i]+j  if j < vc_eff[i]  else -1,
            #   last_token_indices[i]             = base[i] + (vc_eff[i]-1)      (== base + (n-r)).
            # num_draft_tokens is the host list spec_decode_metadata already materialized (NOT a device
            # sync); only valid_count is device-resident. Pure index_select/arange/where -> no host sync.
            # (B) fix: build the index tensors as NON-inference (inference_mode(False)) so the gather
            # that forms the drafter's target_hidden_states (and last_token_indices) yields normal tensors
            # -> the drafter HPU-graph forward's input-copy no longer trips the inference-tensor guard.
            with torch.inference_mode(False):
                step = max_num_draft_tokens + 1
                bsz = len(num_draft_tokens)
                n_t = torch.tensor(num_draft_tokens, dtype=torch.int64, device=self.device)
                base = torch.arange(bsz, dtype=torch.int64, device=self.device) * step
                vc = num_accepted.to(torch.int64).clone()  # read inference tensor -> normal int64
                vc_eff = torch.where(n_t > 0, vc, n_t + 1)
                j = torch.arange(step, dtype=torch.int64, device=self.device)
                full = base.unsqueeze(1) + j.unsqueeze(0)
                mask = j.unsqueeze(0) < vc_eff.unsqueeze(1)
                neg_one = full.new_full((), -1)
                hidden_states_indices = torch.where(mask, full, neg_one).reshape(-1)
                last_token_indices = base + (vc_eff - 1).clamp(min=0)
            return common_attn_metadata, hidden_states_indices, last_token_indices

        # stock path (HPU_ASYNC_SPEC unset / num_accepted is None): unchanged Python loop.
        num_picked_token_indices = []
        last_token_indices = []
        starting_index = 0
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

    def _prepare_attn_metadata_multi_cand(self, block_table_cpu_tensor, positions, model_runner, k,
                                          write_offset, batch_size):
        """MULTI-CANDIDATE drafter attn metadata with PER-CHAIN KV ISOLATION.

        Rows are [batch_size * k] (row r*k+c = request r, candidate chain c). Each chain writes its drafts
        to its OWN scratch block, so its query attends to context + its own earlier drafts only -- no
        cross-chain leak, no tree mask.

        Requires the spec lookahead to allocate k scratch blocks per request (set num_speculative_tokens =
        k * block_size in the speculative_config so the scheduler's round_up(ctx + lookahead, block_size)
        appends exactly k extra blocks; see hpu_model_runner _add_dummy_request:5510-5518 and the GPU
        lookahead path). Scratch block c of request r = block-table column (num_ctx_blocks_r + c). Chain c's
        draft at within-block offset `write_offset` -> slot = scratch_block_c * block_size + write_offset.

        positions: CPU [batch_size*k] TRUE rope positions of the token being WRITTEN this step
                   (= ctx_r + write_offset + 1). block_table_cpu_tensor: [num_seq, total_blocks] for the
                   real requests (num_seq == batch_size).
        """
        block_size = model_runner.block_size
        num_seq = block_table_cpu_tensor.shape[0]
        total_cols = block_table_cpu_tensor.shape[1]
        n_rows = batch_size * k
        # Per-request base ctx position (the last committed token's position): positions[r*k] back out the
        # write_offset+1 we advanced by. All k chains of a req share ctx_r.
        pos_2d = positions.view(batch_size, k)
        ctx_per_req = (pos_2d[:, 0] - (write_offset + 1)).to(torch.int64)          # [batch] (== draft pos-1)

        # HARDEN (warmup / device-index safety). The drafter's positions are gathered as
        # target_positions[last_token_indices] in propose(); during WARMUP (PT_COMPILE_ONLY_MODE does no real
        # compute -> CPU mirrors hold whatever was last there) or any step where that gather reads an
        # unpopulated / placeholder slot, a chain's ctx can be garbage -- the crash that motivated this guard
        # had ctx backed out to ~-1.08e9, i.e. position 0xBF800000 (== float32 -1.0 reinterpreted as int32
        # = -1082130432). Left unguarded: ctx -> huge negative -> nctx = ceil((ctx+1)/bs) huge negative ->
        # scratch_col = nctx + c huge negative, and the OLD upper-only guard (scratch_col < total_cols) let a
        # NEGATIVE column index block_table_cpu_tensor[r, scratch_col] -> out-of-bounds DEVICE access ->
        # Synapse critical error that killed all 8 cards.
        #
        # FIX (mirrors the single-chain prepare_attn_metadata clamp contract -- which silently does
        # torch.where(positions >= max_model_len, 0, positions), never a raw negative index): a real context
        # position MUST be in [0, max_model_len) AND inside the allocated block table. Any out-of-range ctx is
        # a warmup/dummy/placeholder row -> clamp its ctx to 0 (-> nctx 0) so EVERY scratch_col below is fully
        # bounds checked and the row routes to PAD (writes nothing real, device never sees a bad index). We
        # CLAMP rather than raise because (a) the single-chain path clamps -- behavior parity -- and (b) the
        # warmup-mode flag is already reset to False by sample_tokens before propose() runs, so it cannot
        # distinguish warmup here. To keep a REAL positions bug LOUD (the quality bar), we log a one-time
        # WARNING with the offending values; on a healthy decode this branch is never taken.
        global _MC_BAD_POS_WARNED
        max_ctx = total_cols * block_size  # any real ctx fits in the allocated block table
        valid_ctx = (ctx_per_req >= 0) & (ctx_per_req < self.max_model_len) & (ctx_per_req < max_ctx)
        all_valid = bool(valid_ctx.all().item())
        if not all_valid:
            if not _MC_BAD_POS_WARNED:
                _MC_BAD_POS_WARNED = True
                bad = ctx_per_req[~valid_ctx].tolist()
                logger.warning(
                    "HPU_MULTI_CAND drafter saw out-of-range context position(s) %s "
                    "(write_offset=%d, max_model_len=%d, block_table_cols=%d, valid range [0, %d)). "
                    "Routing the offending chain row(s) to PAD (no KV write) to avoid an out-of-bounds "
                    "device index. This is EXPECTED during warmup (compile-only dummy positions); if it "
                    "recurs on real decode steps it indicates an uninitialized/placeholder position reaching "
                    "the multi-candidate KV layout -- investigate. (Logged once.)",
                    bad, write_offset, self.max_model_len, total_cols, min(self.max_model_len, max_ctx))
            # Clamp the invalid ctx to 0 so nctx/scratch_col arithmetic below is well-defined; the per-request
            # valid mask (req_ok) then FORCES every chain of an invalid request to PAD regardless of what the
            # (possibly dummy/all-zero) block table holds at the clamped columns -- so warmup writes nothing.
            ctx_per_req = torch.where(valid_ctx, ctx_per_req, torch.zeros_like(ctx_per_req))
        req_ok = valid_ctx.tolist()
        # committed tokens = positions 0..ctx_r -> ceil((ctx_r+1)/block_size) context blocks.
        num_ctx_blocks_req = torch.ceil((ctx_per_req + 1).float() / block_size).to(torch.int64)  # [batch]

        # Build, per row, the isolated block table [context blocks of r] + [scratch block c of r] and the
        # write slot. Host loop (small: batch*k rows, single-stream batch=1 -> k rows).
        block_tables_list = []
        slot_list = []
        for r in range(batch_size):
            # nctx is now guaranteed 0..total_cols (ctx was clamped non-negative & < max_ctx above).
            nctx = int(num_ctx_blocks_req[r].item()) if r < num_seq else 0
            nctx = max(0, min(nctx, total_cols))
            row_ctx_blocks = block_table_cpu_tensor[r, :nctx].tolist() if r < num_seq else []
            for c in range(k):
                scratch_col = nctx + c
                # FULL bounds check (0 <= scratch_col < total_cols): the old guard only bounded the upper
                # side, so a negative column index silently wrapped into an out-of-bounds DEVICE read. A
                # negative or >=width column => under-allocated lookahead OR a clamped warmup row => PAD.
                # req_ok[r] gates out clamped/garbage-position requests so their chains never write real KV.
                if r < num_seq and req_ok[r] and 0 <= scratch_col < total_cols:
                    scratch_block = int(block_table_cpu_tensor[r, scratch_col].item())
                    # isolated table: context (read-only, shared) + THIS chain's scratch block (last).
                    bt = row_ctx_blocks + [scratch_block]
                    slot = scratch_block * block_size + write_offset
                else:
                    # Not enough scratch blocks allocated (lookahead too small) OR pad/warmup row -> route to a
                    # PAD slot so this chain writes nothing real. LOUD: an under-allocated lookahead means
                    # multi-cand silently degrades; the runner gate logs the requirement.
                    bt = row_ctx_blocks if row_ctx_blocks else [model_runner._PAD_BLOCK_ID]
                    slot = model_runner._PAD_SLOT_ID
                block_tables_list.append(bt)
                slot_list.append(slot)

        block_tables_list = model_runner.defragmenter.resolve_all(block_tables_list)
        slot_mapping = torch.tensor(
            [model_runner.defragmenter.resolve(s) if s != model_runner._PAD_SLOT_ID else s for s in slot_list],
            dtype=torch.int64).view(-1, 1)

        block_list, block_groups, block_usage = \
            model_runner.get_habana_paged_attn_buffers(block_tables_list, slot_mapping.tolist(), n_rows)

        block_list_device = async_h2d_copy(block_list, device=self.device)
        block_usage_device = async_h2d_copy(block_usage, device=self.device)
        block_groups_device = async_h2d_copy(block_groups, device=self.device)
        slot_mapping_device = async_h2d_copy(slot_mapping, device=self.device)

        return HPUAttentionMetadataV1.make_decode_metadata(
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
