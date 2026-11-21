# SPDX-License-Identifier: Apache-2.0
#
# PATCHED for async+spec (HPU_ASYNC_SPEC=1). The stock HPU rejection sampler does a per-step `.cpu()`
# host-sync on the draft/target/bonus/cu tensors ("CPU faster for this step") — that sync sits INSIDE
# execute_model's critical path and SERIALIZES it, defeating the async+spec overlap (MTP_RESULTS §A).
# Under HPU_ASYNC_SPEC=1 we keep the rejection math DEVICE-SIDE (the vectorized ops already key off the
# inputs' device), so execute_model can return before the device->host copy (the runner's async-output
# path defers it). Numerics are device-side == cpu, validated card-free in
# glm52/async_spec/rejection_device_side.py (partial/full+bonus/reject@0/zero-draft/multi-seq).
# Default (env unset) keeps the ORIGINAL .cpu() path — byte-identical to stock, so this file is safe to
# mount anywhere; the device-side path engages ONLY when HPU_ASYNC_SPEC=1.
import os

from vllm.v1.sample import rejection_sampler
import torch
from typing import Optional
from vllm.v1.sample.metadata import SamplingMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE

_ASYNC_SPEC = os.environ.get("HPU_ASYNC_SPEC", "0") == "1"


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Vectorized rejection sampling (see stock docstring). Identical math; only the device of the
    inputs differs by the HPU_ASYNC_SPEC gate below."""
    # 0. wait for device processing to finish.
    if _ASYNC_SPEC:
        # ASYNC+SPEC: stay ON-DEVICE (no per-step host sync) so execute_model returns before sampling;
        # the runner defers the copy (prev_sampled_token_ids = bonus-per-req, variable-length async out).
        padded_draft_token_ids = padded_draft_token_ids.to(torch.int32)
        padded_target_token_ids = padded_target_token_ids.to(torch.int32)
        bonus_token_ids = bonus_token_ids.to(torch.int32)
        # cu_num_draft_tokens stays on its current device
    else:
        # ORIGINAL (NOTE chendi: CPU is faster for this step) — the per-step host sync.
        padded_draft_token_ids = padded_draft_token_ids.cpu().to(torch.int32)
        padded_target_token_ids = padded_target_token_ids.cpu().to(torch.int32)
        bonus_token_ids = bonus_token_ids.cpu().to(torch.int32)
        cu_num_draft_tokens = cu_num_draft_tokens.cpu()

    # 1. Get tensor dimensions and device for calculations
    num_seqs = len(num_draft_tokens)
    padded_draft_token_ids = padded_draft_token_ids.view(num_seqs, -1)
    max_draft_tokens = padded_draft_token_ids.shape[-1]
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device
    # co-locate cu_num_draft_tokens with the (possibly device-resident) token tensors so the
    # torch.cat below can't hit a cross-device mismatch under HPU_ASYNC_SPEC (no-op when same device).
    cu_num_draft_tokens = cu_num_draft_tokens.to(device)

    # 2. number of draft tokens per sequence from the cumulative sum
    start_indices = torch.cat((torch.tensor([0], device=device,
                                            dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # 3. first mismatch, ignoring padding tokens
    pos = torch.arange(max_draft_tokens, device=device)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)
    matches = (padded_draft_token_ids == padded_target_token_ids)
    mismatches = ~matches
    any_mismatch = mismatches.any(dim=1)
    any_mismatch[num_draft_tokens_per_seq == 0] = False
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    # 4. number of accepted tokens for each sequence
    num_accepted = ((first_mismatch_idx + 1) * any_mismatch + num_draft_tokens_per_seq * (~any_mismatch))

    # 5. output tensor by masking the target tokens
    output_tokens = torch.empty(
        (num_seqs, max_draft_tokens + 1),
        dtype=torch.int32,
        device=device,
    )
    output_tokens.fill_(PLACEHOLDER_TOKEN_ID)
    acceptance_mask = pos < num_accepted.unsqueeze(-1)
    acceptance_mask = acceptance_mask & valid_token_mask
    output_slice = output_tokens[:, :max_draft_tokens]
    output_slice[acceptance_mask] = padded_target_token_ids[acceptance_mask]

    # 6. bonus token where all draft tokens were accepted
    all_accepted_mask = ~any_mismatch
    if all_accepted_mask.sum() > 0:
        bonus_pos_indices = num_draft_tokens_per_seq[all_accepted_mask].long()
        bonus_values = bonus_token_ids[all_accepted_mask].squeeze(-1)
        output_tokens[all_accepted_mask, bonus_pos_indices] = bonus_values

    return output_tokens


def rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # temp>0 GUARD (LOUD, recoverable): the HPU rejection sampler only implements the greedy
    # (argmax-compare) path. Stock used a bare `assert all_greedy`; with Python run under -O the assert
    # is stripped (silent wrong output), and under spec-decode a non-greedy request otherwise reaches the
    # device math and raises deep in execute_model -> 500 + container shutdown (the temp>0 crash).
    # Raise an explicit NotImplementedError EARLY, before any device work, so the engine surfaces a clean
    # error for that request instead of tearing down the worker. On the all-greedy path this is a no-op
    # (same predicate as the stock assert) -> greedy A/B stays byte-identical.
    if not sampling_metadata.all_greedy:
        raise NotImplementedError(
            "HPU rejection sampler (spec decode) supports greedy sampling only; received a request with "
            "temperature>0 / non-greedy sampling. Set temperature=0 for speculative requests, or disable "
            "speculative decoding for sampled requests.")
    target_argmax = target_probs.argmax(dim=-1)
    output_token_ids = rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                                cu_num_draft_tokens)
    return output_token_ids


rejection_sampler.rejection_sample = rejection_sample
