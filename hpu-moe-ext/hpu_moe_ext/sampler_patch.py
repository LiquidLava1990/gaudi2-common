"""Sampler mark_unbacked fix for HPU torch.compile mode.

On HPU, the model runner compiles self.sampler via torch.compile
(hpu_model_runner.py line ~4590). Inside compiled regions, calls to
torch._dynamo.decorators.mark_unbacked are FORBIDDEN — dynamo raises:

    AssertionError: Attempt to trace forbidden callable
        <function mark_unbacked at 0x...>

The upstream sampler was written for CUDA where the sampler is NOT compiled
(torch.compile wraps only the model forward, not the sampler). On HPU,
regional compilation wraps the sampler too, so the mark_unbacked calls in
gather_logprobs (line ~340) and gather_specific_token_logprobs (line ~212)
crash every request after the first token.

Fix: replace both static methods with versions that extract the mark_unbacked
calls into @torch.compiler.disable helper functions (same pattern already used
by hpu_model_runner._mark_unbacked_dim0 at line ~831 for attn_metadata).
"""
import torch

# ------------------------------------------------------------------
# Helpers that must NOT be traced by dynamo. @torch.compiler.disable
# is the same mechanism already used in hpu_model_runner.py.
# ------------------------------------------------------------------


@torch.compiler.disable
def _mark_unbacked_logprobs(logprobs: torch.Tensor,
                             logprobs2: torch.Tensor) -> None:
    """Call mark_unbacked on both logprob tensors outside compiled scope."""
    torch._dynamo.decorators.mark_unbacked(logprobs, 0)
    torch._dynamo.decorators.mark_unbacked(logprobs2, 0)


# ------------------------------------------------------------------
# Patched static methods
# ------------------------------------------------------------------


def _patched_gather_logprobs(logprobs, num_logprobs, token_ids):
    """Replacement for Sampler.gather_logprobs that is safe under torch.compile.

    Identical logic to the original, but the mark_unbacked pair is called via
    the @torch.compiler.disable wrapper so dynamo never tries to trace it.
    """
    from vllm.v1.sample.sampler import LogprobsTensors, batched_count_greater_than
    assert token_ids.dtype == torch.int64
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
    token_ids = token_ids.unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, token_ids)
    # Move mark_unbacked outside the compiled region.
    _mark_unbacked_logprobs(logprobs, token_logprobs)
    token_ranks = batched_count_greater_than(logprobs, token_logprobs)
    indices = torch.cat((token_ids, topk_indices), dim=1)
    logprobs_out = torch.cat((token_logprobs, topk_logprobs), dim=1)
    indices = indices.to(torch.int32)
    return LogprobsTensors(indices, logprobs_out, token_ranks)


def _patched_gather_specific_token_logprobs(
        self, logprobs, logprob_token_ids, sampled):
    """Replacement for Sampler.gather_specific_token_logprobs safe under compile.

    Identical logic to the original, but the mark_unbacked pair is called via
    the @torch.compiler.disable wrapper.
    """
    from vllm.v1.sample.sampler import LogprobsTensors, batched_count_greater_than
    if not logprob_token_ids:
        return None

    batch_size = logprobs.shape[0]
    device = logprobs.device
    max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())
    pin = self.pin_memory

    token_ids_cpu = torch.zeros(
        batch_size, max_num_tokens + 1, dtype=torch.int64, pin_memory=pin)
    valid_mask_cpu = torch.zeros(
        batch_size, max_num_tokens + 1, dtype=torch.bool, pin_memory=pin)
    valid_mask_cpu[:, 0] = True
    for req_idx, token_ids in logprob_token_ids.items():
        num_tokens = len(token_ids)
        token_ids_cpu[req_idx, 1:num_tokens + 1] = torch.as_tensor(
            token_ids, dtype=torch.int64)
        valid_mask_cpu[req_idx, 1:num_tokens + 1] = True

    token_ids_tensor = token_ids_cpu.to(device, non_blocking=True)
    valid_mask = valid_mask_cpu.to(device, non_blocking=True)
    token_ids_tensor[:, 0] = sampled

    gathered_logprobs = logprobs.gather(-1, token_ids_tensor)
    gathered_logprobs = gathered_logprobs.masked_fill(~valid_mask, float("-inf"))
    sampled_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1))
    # Move mark_unbacked outside the compiled region.
    _mark_unbacked_logprobs(logprobs, sampled_logprobs)
    token_ranks = batched_count_greater_than(logprobs, sampled_logprobs)

    return LogprobsTensors(
        logprob_token_ids=token_ids_tensor.to(torch.int32),
        logprobs=gathered_logprobs,
        selected_token_ranks=token_ranks,
    )


# ------------------------------------------------------------------
# Installer
# ------------------------------------------------------------------

_INSTALLED = False


def install():
    """Patch Sampler to avoid mark_unbacked inside torch.compile regions."""
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        from vllm.v1.sample.sampler import Sampler
    except ImportError:
        return  # sampler not present in this vllm version, nothing to do

    # Verify the upstream methods still have the mark_unbacked calls we expect.
    # If upstream has already fixed this (removed mark_unbacked or wrapped it),
    # skip the patch silently.
    import inspect
    src_gl = inspect.getsource(Sampler.gather_logprobs)
    src_gs = inspect.getsource(Sampler.gather_specific_token_logprobs)
    needs_patch = (
        "mark_unbacked" in src_gl or "mark_unbacked" in src_gs
    )
    if not needs_patch:
        return

    Sampler.gather_logprobs = staticmethod(_patched_gather_logprobs)
    Sampler.gather_specific_token_logprobs = _patched_gather_specific_token_logprobs

    _INSTALLED = True
    try:
        import logging
        logging.getLogger("vllm").info(
            "hpu-moe-ext sampler_patch: wrapped mark_unbacked calls in "
            "@torch.compiler.disable helpers (HPU torch.compile sampler fix)")
    except Exception:  # noqa: BLE001
        pass
