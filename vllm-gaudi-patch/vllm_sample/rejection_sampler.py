# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (
            spec_config is not None
            and spec_config.rejection_sample_method == "synthetic"
        ):
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        self.synthetic_mode = self.synthetic_conditional_rates is not None

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        # NOTE(hpu-port): Call the implementation through a private module-level
        # reference instead of the public ``rejection_sample`` name. The
        # vllm-gaudi plugin monkey-patches ``rejection_sampler.rejection_sample``
        # with a *greedy-only* torch fallback (see
        # vllm_gaudi/v1/sample/hpu_rejection_sampler.py). Binding to
        # ``_rejection_sample_impl`` here keeps the full greedy+random capable
        # implementation in force so mixed / temperature>0 spec-decode batches
        # are handled correctly on HPU.
        output_token_ids = _rejection_sample_impl(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: To avoid cpu-gpu synchronization, we now simply compute indices for
        # all draft tokens, including the rejected ones. The rejected tokens will
        # be filtered out in the `parse_output`.
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # we replace rejected token ids with 0 to avoid gather_logprobs error
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # Compute logprobs for accepted tokens.
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )
        holder = sampling_metadata.thinking_budget_state_holder
        needs_thinking = holder is not None and holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words or needs_thinking:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        repeat_indices: torch.Tensor | None = None
        need_repeat_indices = (
            sampling_metadata.allowed_token_ids_mask is not None
            or has_penalties
            or needs_thinking
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
        if holder is not None and holder.has_tracked_requests():
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token=False,
                spec_token_ids=sampling_metadata.spec_token_ids,
            )
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> torch.Tensor:
    # ---- HPU padded-layout normalization (no-op for the base flat-ragged
    # contract) --------------------------------------------------------------
    # Defense-in-depth for HPU: the vllm-gaudi runner historically handed the
    # sampler a *padded rectangular* layout -- draft_token_ids as a 2D
    # [batch * stride, 1] tensor (stride = num_speculative_tokens) with each
    # request's first `num_draft_tokens[i]` slots real and the rest padded --
    # while cu_num_draft_tokens holds the ACTUAL ragged counts. Every kernel
    # below assumes the base flat-ragged contract: draft_token_ids is 1D of
    # length sum(num_draft_tokens), request i occupying
    # [cu[i-1], cu[i]). The runner now emits that directly, so this block is a
    # no-op; it stays as a guard so a padded feed can never resurface the
    # `assert draft_token_ids.ndim == 1` / cu-vs-length mismatch. Uses the
    # python `num_draft_tokens` list (no device sync).
    if draft_token_ids.ndim != 1:
        draft_token_ids = draft_token_ids.reshape(-1)
    _ragged_num_tokens = sum(num_draft_tokens)
    if num_draft_tokens and draft_token_ids.shape[0] != _ragged_num_tokens:
        _batch = len(num_draft_tokens)
        _stride = draft_token_ids.shape[0] // _batch
        if _ragged_num_tokens > 0:
            _keep = torch.cat([
                torch.arange(i * _stride, i * _stride + n)
                for i, n in enumerate(num_draft_tokens) if n > 0
            ]).to(draft_token_ids.device)
        else:
            _keep = torch.empty(0, dtype=torch.long,
                                device=draft_token_ids.device)
        draft_token_ids = draft_token_ids.index_select(0, _keep).contiguous()
        target_logits = target_logits.index_select(0, _keep).contiguous()
        if draft_probs is not None:
            draft_probs = draft_probs.index_select(0, _keep).contiguous()

    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # Generate uniform probabilities before either kernel because synthetic
    # mode needs them in the greedy kernel too.  Skip only when all requests
    # are greedy *and* synthetic mode is off (the standard fast-path).
    # [num_tokens]
    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits.
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    assert uniform_probs is not None
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
    )
    return output_token_ids


# Private, monkey-patch-proof reference to the full rejection-sampling
# implementation. `RejectionSampler.forward` calls this instead of the public
# module-level `rejection_sample`, which the vllm-gaudi plugin overrides with a
# greedy-only fallback. See the note in `forward`.
_rejection_sample_impl = rejection_sample


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens: List[List[int]]
            Number of draft tokens per request.
        generators: Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic mode only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)


# ---------------------------------------------------------------------------
# Pure-torch fallbacks for platforms without a working Triton JIT (e.g. HPU /
# Gaudi, where `vllm.triton_utils` installs a placeholder and `kernel[grid](...)`
# raises `TypeError: 'function' object is not subscriptable`).
#
# When `HAS_TRITON` is False we replace the four `@triton.jit` kernels above
# with device-agnostic torch implementations. They are wrapped in `_FuncWrapper`
# so the existing `kernel[grid](...)` call sites keep working unchanged (the
# `[grid]` subscript is ignored, exactly like vLLM's own CPU fallback in
# `vllm/utils/cpu_triton_utils.py`). On CUDA (`HAS_TRITON` True) this block is a
# no-op and the real Triton kernels are used, so GPU behaviour is unchanged.
#
# Design notes:
#   * Large tensor math (softmax already done by caller, per-token argmax for
#     recovered tokens) stays on-device: only vectorized ops (index_select,
#     elementwise, argmax, scatter_) that Gaudi supports in production
#     (see apply_top_k_top_p_pytorch) are used.
#   * The sequential accept/reject control flow ("stop at first mismatch") is
#     done with small CPU tensors and written back with a single whole-tensor
#     copy_, matching vllm-gaudi's own `hpu_rejection_sampler.py` (which notes
#     CPU is faster/safer than HPU for this step and avoids per-element
#     device-tensor scatter, a known HPU lazy-mode trap).
#   * Acceptance rule, correction-token choice, bonus-token placement and
#     greedy determinism are ported faithfully from the Triton kernels above.
# ---------------------------------------------------------------------------
if not HAS_TRITON:

    class _FuncWrapper:
        """Make a plain function callable with Triton's ``fn[grid](...)``
        launch syntax; the grid subscript is ignored."""

        def __init__(self, func):
            self.func = func

        def __getitem__(self, _grid):
            return self.func

    def _cu_to_counts_and_starts(cu_num_tokens):
        """Return CPU (starts, counts) from an inclusive cumulative-count
        tensor. counts[i] = number of tokens for request i."""
        cu_cpu = cu_num_tokens.to(device="cpu", dtype=torch.long)
        batch_size = cu_cpu.shape[0]
        starts = torch.zeros(batch_size, dtype=torch.long)
        if batch_size > 1:
            starts[1:] = cu_cpu[:-1]
        counts = cu_cpu - starts
        return starts, counts

    def _token_req_and_pos(cu_num_tokens, num_tokens):
        """CPU per-token request index and position-within-request."""
        starts, counts = _cu_to_counts_and_starts(cu_num_tokens)
        batch_size = counts.shape[0]
        if num_tokens == 0:
            empty = torch.empty(0, dtype=torch.long)
            return starts, counts, empty, empty
        req_idx = torch.repeat_interleave(torch.arange(batch_size), counts)
        pos_in_req = torch.arange(num_tokens) - starts[req_idx]
        return starts, counts, req_idx, pos_in_req

    def _assemble_spec_output(
        output_token_ids,  # [batch, max_spec_len + 1] device tensor (mutated)
        cu_num_draft_tokens,  # [batch]
        draft_token_ids,  # [num_tokens]
        accept_flat,  # [num_tokens] bool: draft token accepted?
        reject_value_flat,  # [num_tokens]: correction token at a reject position
        bonus_token_ids,  # [batch] or [batch, 1]
        rows_mask,  # [batch] bool or None (which rows this kernel owns)
        max_spec_len,
    ):
        """Vectorized port of the accept/reject/bonus write-out shared by the
        greedy and random Triton kernels.

        For each request the Triton kernel accepts draft tokens until the first
        rejection: it emits the draft token at every accepted position, the
        correction token at the (first) rejected position, then stops; if every
        draft token is accepted it appends the bonus token. Positions after the
        first rejection stay at PLACEHOLDER_TOKEN_ID.
        """
        device = output_token_ids.device
        batch_size = cu_num_draft_tokens.shape[0]
        num_tokens = draft_token_ids.shape[0]
        max_cols = max_spec_len + 1

        _, num_draft, req_idx, pos_in_req = _token_req_and_pos(
            cu_num_draft_tokens, num_tokens
        )

        draft_cpu = draft_token_ids.to(device="cpu", dtype=torch.long)
        accept_cpu = accept_flat.to(device="cpu", dtype=torch.bool)
        reject_cpu = reject_value_flat.to(device="cpu", dtype=torch.long)

        # Padded per-request views. Invalid (padding) positions are marked as
        # "accepted" so they never trigger an early rejection.
        padded_accept = torch.ones((batch_size, max_spec_len), dtype=torch.bool)
        padded_draft = torch.zeros((batch_size, max_spec_len), dtype=torch.long)
        padded_reject = torch.zeros((batch_size, max_spec_len), dtype=torch.long)
        if num_tokens > 0:
            padded_accept[req_idx, pos_in_req] = accept_cpu
            padded_draft[req_idx, pos_in_req] = draft_cpu
            padded_reject[req_idx, pos_in_req] = reject_cpu

        pos_ar = torch.arange(max_spec_len).unsqueeze(0)  # [1, max_spec_len]
        valid_mask = pos_ar < num_draft.unsqueeze(1)  # [batch, max_spec_len]
        reject_mask = (~padded_accept) & valid_mask
        any_reject = reject_mask.any(dim=1)  # [batch]
        # First rejected position per request (0 when there is none; guarded by
        # any_reject below). Guard max_spec_len == 0 (every request has 0 draft
        # tokens): argmax over a size-0 dim would raise, and any_reject is all
        # False anyway so the value is unused.
        if max_spec_len == 0:
            first_reject = torch.zeros(batch_size, dtype=torch.long)
        else:
            first_reject = torch.argmax(reject_mask.to(torch.int32), dim=1)
        # Number of leading draft tokens that are accepted verbatim.
        num_accepted = torch.where(any_reject, first_reject, num_draft)

        out = torch.full((batch_size, max_cols), PLACEHOLDER_TOKEN_ID,
                         dtype=torch.long)
        # Accepted draft tokens: positions [0, num_accepted).
        accept_pos_mask = (pos_ar < num_accepted.unsqueeze(1)) & valid_mask
        out_body = out[:, :max_spec_len]
        out_body[accept_pos_mask] = padded_draft[accept_pos_mask]

        # Correction token at the first rejected position.
        if bool(any_reject.any()):
            rr = torch.nonzero(any_reject, as_tuple=True)[0]
            cols = first_reject[rr]
            out[rr, cols] = padded_reject[rr, cols]

        # Bonus token where every draft token was accepted (no rejection).
        all_accepted = ~any_reject
        if bool(all_accepted.any()):
            ra = torch.nonzero(all_accepted, as_tuple=True)[0]
            bonus_cpu = bonus_token_ids.to(device="cpu", dtype=torch.long).reshape(-1)
            out[ra, num_draft[ra]] = bonus_cpu[ra]

        # Merge: only overwrite the rows this kernel is responsible for.
        base_cpu = output_token_ids.to(device="cpu", dtype=torch.long)
        if rows_mask is None:
            merged = out
        else:
            rm = rows_mask.to(device="cpu", dtype=torch.bool).reshape(-1, 1)
            merged = torch.where(rm, out, base_cpu)
        output_token_ids.copy_(
            merged.to(device=device, dtype=output_token_ids.dtype)
        )

    def _rejection_greedy_sample_torch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
        uniform_probs=None,
        synthetic_conditional_rates=None,
        SYNTHETIC_MODE=False,
    ):
        # Synthetic acceptance mirrors the CPU fallback: unsupported here.
        assert not SYNTHETIC_MODE, (
            "synthetic acceptance is not supported by the HPU torch "
            "rejection sampler"
        )
        # Greedy acceptance: accept the draft token iff it equals the target
        # argmax; the emitted token is the target argmax at every accepted
        # position and at the first rejected position (they coincide with the
        # draft token wherever accepted). No RNG -> temperature==0 is fully
        # deterministic.
        argmax_cpu = target_argmax.to(device="cpu", dtype=torch.long)
        draft_cpu = draft_token_ids.to(device="cpu", dtype=torch.long)
        accept_flat = draft_cpu == argmax_cpu
        # is_greedy is None on the all-greedy fast path (own every row).
        rows_mask = None if is_greedy is None else is_greedy
        _assemble_spec_output(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            accept_flat,
            argmax_cpu,  # correction token == target argmax
            bonus_token_ids,
            rows_mask,
            max_spec_len,
        )

    def _rejection_random_sample_torch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates=None,
        NO_DRAFT_PROBS=False,
        SYNTHETIC_MODE=False,
    ):
        assert not SYNTHETIC_MODE, (
            "synthetic acceptance is not supported by the HPU torch "
            "rejection sampler"
        )
        num_tokens = draft_token_ids.shape[0]
        # Random kernel owns the non-greedy rows.
        rows_mask = ~is_greedy

        if num_tokens == 0:
            accept_flat = torch.empty(0, dtype=torch.bool)
        else:
            draft_idx = draft_token_ids.to(torch.long).view(-1, 1)
            # target_prob[draft_token] (float32, on device).
            target_prob = target_probs.gather(1, draft_idx).squeeze(1)
            if NO_DRAFT_PROBS:
                # draft_prob == 1 -> accept iff target_prob >= uniform.
                ratio_dev = target_prob
                draft_positive_cpu = None
            else:
                draft_prob = draft_probs.gather(1, draft_idx).squeeze(1)
                # Match the Triton kernel: compute the fp32 ratio, then compare
                # against the fp64 uniform sample. Guard divide-by-zero.
                safe_dp = torch.where(
                    draft_prob > 0, draft_prob, torch.ones_like(draft_prob)
                )
                ratio_dev = target_prob / safe_dp
                draft_positive_cpu = (draft_prob > 0).to("cpu")
            # Compare in float64 to preserve the uniform's precision.
            ratio_cpu = ratio_dev.to(device="cpu", dtype=torch.float64)
            uni_cpu = uniform_probs.to(device="cpu", dtype=torch.float64)
            accept_flat = ratio_cpu >= uni_cpu
            if draft_positive_cpu is not None:
                accept_flat = draft_positive_cpu & accept_flat

        _assemble_spec_output(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            accept_flat,
            recovered_token_ids,  # correction token == recovered token
            bonus_token_ids,
            rows_mask,
            max_spec_len,
        )

    def _expand_torch(
        output,  # [num_tokens] device tensor (mutated)
        input_val,  # [batch]
        cu_num_tokens,  # [batch]
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=None,
    ):
        # Expand a per-request [batch] tensor to a per-token [num_tokens] tensor
        # by repeating each request's value `count[i]` times, replacing any
        # `replace_from` value with `replace_to` first.
        device = output.device
        num_tokens = output.shape[0]
        _, _, req_idx, _ = _token_req_and_pos(cu_num_tokens, num_tokens)
        req_idx = req_idx.to(device)
        src = input_val
        if replace_from != replace_to:
            src = torch.where(
                src == replace_from,
                torch.tensor(replace_to, dtype=src.dtype, device=src.device),
                src,
            )
        output.copy_(src.index_select(0, req_idx))

    def _sample_recovered_tokens_torch(
        output_token_ids,  # [num_tokens] device tensor (mutated)
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,  # [batch, vocab_size]
        vocab_size,
        BLOCK_SIZE=None,
        NO_DRAFT_PROBS=False,
    ):
        # Recovered token = argmax_v prob[v] * inv_q[req, v], where prob is the
        # residual distribution. This is the exponential/Gumbel-max sampling
        # trick; inv_q is shared per-request (same as the Triton kernel).
        num_tokens = draft_token_ids.shape[0]
        if num_tokens == 0:
            return
        device = target_probs.device
        _, _, req_idx, _ = _token_req_and_pos(cu_num_draft_tokens, num_tokens)
        req_idx = req_idx.to(device)
        invq_tok = inv_q.index_select(0, req_idx)  # [num_tokens, vocab_size]

        if NO_DRAFT_PROBS:
            # prob = target_probs with the draft token's mass removed. Build the
            # exclusion via an arange column-mask + masked_fill (the same ops
            # vllm-gaudi runs in production in apply_top_k_top_p_pytorch), which
            # avoids per-element device scatter (an HPU lazy-mode trap).
            score = target_probs * invq_tok
            vocab_ids = torch.arange(vocab_size, device=device).unsqueeze(0)
            draft_col = vocab_ids == draft_token_ids.to(torch.long).view(-1, 1)
            score = score.masked_fill(draft_col, 0.0)
        else:
            prob = torch.clamp(target_probs - draft_probs, min=0.0)
            score = prob * invq_tok
        recovered = score.argmax(dim=1)
        output_token_ids.copy_(recovered.to(output_token_ids.dtype))

    # Install the torch fallbacks over the Triton kernel names so the existing
    # `kernel[grid](...)` call sites in this module keep working on HPU.
    rejection_greedy_sample_kernel = _FuncWrapper(_rejection_greedy_sample_torch)
    rejection_random_sample_kernel = _FuncWrapper(_rejection_random_sample_torch)
    expand_kernel = _FuncWrapper(_expand_torch)
    sample_recovered_tokens_kernel = _FuncWrapper(_sample_recovered_tokens_torch)
