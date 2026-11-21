###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
import math
from functools import lru_cache, wraps
from typing import Optional, Any

import habana_frameworks.torch as htorch
import torch
import itertools
from vllm_gaudi.extension.logger import logger

from vllm_gaudi.extension.runtime import get_config


@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y, **kwargs):
        return torch.matmul(x, y, **kwargs)


class B2BMatmul(Matmul):
    """Specialized alias for batch2block and block2batch matmul operations.
    
    This class remains functionally identical to ``Matmul`` but is used to
    semantically mark B2B-related matmuls. This enables the system to apply the
    fix that uses the B2B output measurements as the input measurements during
    calibration, avoiding corrupted scales from the KV‑cache.
    """

    def __init__(self):
        super().__init__()


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


def get_kv_fetch_extra_args(**kwargs):
    if not get_config().per_token_kv_scaling_support:
        kwargs.pop('scales', None)
    return kwargs


class VLLMKVCache(torch.nn.Module):

    def __init__(self, is_v_cache: bool = False):
        super().__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        # is_v_cache is used in INC FP8 dynamic quantization to identify V cache
        self.is_v_cache = is_v_cache

    def forward(self, input, cache, slot_mapping, scales=None, block_size=None, is_prompt=False, **kwargs):
        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            cache.index_copy_(0, slot_mapping, input)
        return cache

    def fetch_from_cache(self, cache, blocks, scales=None, **kwargs):
        if self.use_contiguous_pa:
            return cache[:blocks.size(0)]
        else:
            return cache.index_select(0, blocks)


# -------------------------------------------------------------------------
# Gaudi-2 fp8_e4m3 top-exponent-bin (HW) bug workaround for the KV cache.
# On-card diagnosis (KVQ_RT value dump) proved Gaudi's float8_e4m3fn->bfloat16
# conversion (torch.ops.hpu.cast_from_fp8) MISHANDLES the top exponent bin
# [256, 448]: any stored fp8 code in that range comes back inf/NaN and gets
# zeroed. For fp8_inc the KV latent (MLA) is stored raw (input_scale=1.0), so
# any latent component that lands in [256,448] is silently zeroed on read ->
# "correct-first-token-then-garbage" long-context degradation (short prompts
# rarely reach the bin; long context accumulates tokens that do). We scale the
# STORE down so the on-card fp8 absmax maps to 224 (= 1.75*2^7, the top of the
# SECOND-highest exponent) instead of 448, keeping every stored value below the
# broken bin, and multiply the matching amount back on READ (dequant) so the
# round-trip is exact. fp8 relative precision is constant across exponents, so
# this costs ZERO accuracy -- it only avoids the HW bug. This mirrors the
# established Gaudi-2 fp8-weight halving in ops.py (gaudi_weight_wrapper: fp8
# loaded_weight *= 0.5). Baked into quant/dequant (not just the ctor default)
# so the cap holds for any input_scale, incl. calibrated overrides.
_FP8_E4M3_HW_MAX = 448.0   # torch.finfo(float8_e4m3fn).max (top of broken bin)
_FP8_E4M3_SAFE_MAX = 224.0  # 1.75 * 2**7, top of the highest SAFE exponent bin
_FP8_KV_STORE_SCALE = 1.0  # no-op: calibrated latent absmax <=12, far below the broken bin

# ---------------------------------------------------------------------------
# FIXED fp8 attention-matmul scales (NO per-step amax -- see FP8Matmul.forward).
# A per-step abs().amax() over the O(seqlen) K/V cache was numerically coherent
# but cost ~2x decode throughput (9.3 -> 4.5 tok/s), so we replace it with fixed
# per-operand scales chosen from the MEASURED / known operand magnitudes, each
# with headroom that (a) lifts the smallest meaningful components above e4m3's
# ~1.95e-3 flush floor and (b) keeps the largest values below the broken
# [256,448] e4m3 top bin. The gemm's 1/scale descale recovers exact values, so
# this is numerically identical to the dynamic version EXCEPT the scale is a safe
# constant instead of a measured one.
#   * K (MLA QK operand): synchronized eager calibration over all 74 active
#     layers x 8 TP ranks at 14,445 tokens measured absmax 12. scale 12 maps
#     that to 144, leaving 55% headroom to the 224 Gaudi2-safe ceiling.
#   * V (MLA AV operand): the same calibration measured absmax 5. scale 32
#     maps that to 160 while preserving substantially more small-value detail.
#   * Q (matmul_qk 1st operand, absorbed query AFTER the softmax-scale multiply):
#     measured absmax 2.171875 (not the previously assumed ~0.1). scale 64 maps
#     that to 139, leaving 61% headroom; scale 256 was provably unsafe (~556).
#   * attn-weights (matmul_av 1st operand): bounded in [0,1] by construction, so
#     _FP8_E4M3_SAFE_MAX (224) is exact and needs no headroom analysis.
# NOTE: batch2block / block2batch are NOT fp8 here (kept bf16) -- they multiply a
# 0/1 block-mapping matrix (structural gather/scatter, memory-bound) whose data
# operand ranges from ~0.07 (gathered query) to ~O(128) (block sums); no single
# fixed fp8 scale fits both, and the block2batch accumulation over blocks is the
# seqlen-dependent path, so bf16 keeps it exact and cheap.
_FP8_ATTN_K_SCALE = 12.0
_FP8_ATTN_V_SCALE = 32.0
_FP8_ATTN_KV_SCALE = _FP8_ATTN_K_SCALE  # compatibility alias for external imports
_FP8_ATTN_Q_SCALE = 64.0
_FP8MM_DBG = {}
_FP8MM_INSTANCE_ID = itertools.count()


def _env_true(name, default='0'):
    return os.environ.get(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


class VLLMFP8KVCache(VLLMKVCache):

    def __init__(self, input_scale=1.0):
        super().__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.input_scale = input_scale
        self.output_scale = 1.0 / self.input_scale

    def quant_input(self, input):
        # Scale down by _FP8_KV_STORE_SCALE (0.5) so stored fp8 <= 224, avoiding
        # the broken [256,448] e4m3 bin; dequant_output multiplies it back.
        return torch.ops.hpu.cast_to_fp8_v2(input, self.input_scale * _FP8_KV_STORE_SCALE, False, False,
                                            torch.float8_e4m3fn)[0]

    def dequant_output(self, output):
        # Matching read scale: output_scale / 0.5 == output_scale * 2 -> exact round-trip.
        return torch.ops.hpu.cast_from_fp8(output, self.output_scale / _FP8_KV_STORE_SCALE, torch.bfloat16)

    def forward(self, input, *args, **kwargs):
        qinput = self.quant_input(input)
        return super().forward(qinput, *args, **kwargs)

    def fetch_from_cache(self, quant_cache, blocks, permutations=None, **kwargs):
        if permutations:
            output_cache = super().fetch_from_cache(quant_cache, blocks, permutations)
            for i in range(len(output_cache)):
                output_cache[i] = self.dequant_output(output_cache[i])
            return output_cache
        output_cache = super().fetch_from_cache(quant_cache, blocks)
        return self.dequant_output(output_cache)


class FP8Matmul(torch.nn.Module):

    def __init__(
        self,
        scale_input=1.0,
        scale_other=1.0,
    ):
        super().__init__()
        self.scale_input = scale_input
        self.scale_other = scale_other
        self.instance_id = next(_FP8MM_INSTANCE_ID)
        # The opt-in switch keeps the historical dynamic calibration path as the
        # default for existing users. Production profiles must explicitly enable
        # fixed scaling after their operand ranges have been verified.
        self.fixed_scale = _env_true('HPU_FP8_ATTN_FIXED_SCALE')
        self.verify_calls = max(0, int(os.environ.get('HPU_FP8_ATTN_VERIFY_SCALES', '0')))
        self.verify_strict = _env_true('HPU_FP8_ATTN_VERIFY_STRICT')
        # Optional runtime arm prevents synthetic memory-profile/warmup tensors
        # from consuming the verifier. Touch the file only after /health and
        # immediately before a real calibration request.
        self.verify_arm_file = os.environ.get('HPU_FP8_ATTN_VERIFY_ARM_FILE', '').strip()

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

    def matmul_fp8(self, x, other, out_dtype, scale_input_inv=None, scale_other_inv=None):
        return torch.ops.hpu.fp8_gemm_v2(
            A=x,
            trans_A=False,
            B=other,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=scale_input_inv,
            B_scale_inv=scale_other_inv,
            bias=None,
            accumulate=False,
        )

    def forward(self, input, other, **kwargs):
        # FIXED per-operand fp8 scaling -- NO per-step amax reduction. (An earlier
        # dynamic amax->224 version was numerically coherent at every context size
        # but cost an O(seqlen) abs().amax() over the K/V cache EVERY decode step,
        # every layer, dropping decode from ~9.3 to ~4.5 tok/s -- slower than the
        # bf16 fallback it was meant to beat. The all-fixed-scale-1.0 version was
        # fast but flushed small Q/K/attn components -> garbled past ~4k tokens.)
        #
        # Fix: each instance carries FIXED per-operand scales (set at construction
        # in hpu_attn.py) chosen from the operand's KNOWN magnitude so small
        # components clear e4m3's ~1.95e-3 flush floor while the max stays below
        # the broken [256,448] top bin -- no measurement needed. See the
        # _FP8_ATTN_*_SCALE constants above for the per-operand values/headroom:
        #   matmul_qk: scale_input=Q=64, scale_other=K=12
        #   matmul_av: scale_input=attn=224 ([0,1] exact), scale_other=V=32
        # Round-trip is EXACT: (input*s_i).(other*s_o) * (1/s_i)*(1/s_o) ==
        # input . other (the fixed s_i, s_o cancel; residual is only fp8 rounding,
        # now well-scaled instead of a flush-to-zero). The block-mapping b2b
        # matmuls are kept in bf16 (not this class) -- see the constants' comment.
        # Calibration mode retains the coherent dynamic amax path. Fixed mode
        # uses the constructor scales and removes both O(seqlen) reductions.
        global _FP8MM_DBG
        _key = self.instance_id
        _count = _FP8MM_DBG.get(_key, 0)
        _armed = not self.verify_arm_file or os.path.exists(self.verify_arm_file)
        _need_verify = _armed and _count < self.verify_calls

        if self.fixed_scale:
            _s_i = self.scale_input
            _s_o = self.scale_other
            if _need_verify:
                _si_amax = input.detach().abs().amax().to(torch.float32).clamp(min=1e-9)
                _so_amax = other.detach().abs().amax().to(torch.float32).clamp(min=1e-9)
        else:
            _si_amax = input.detach().abs().amax().to(torch.float32).clamp(min=1e-9)
            _so_amax = other.detach().abs().amax().to(torch.float32).clamp(min=1e-9)
            _s_i = _FP8_E4M3_SAFE_MAX / _si_amax
            _s_o = _FP8_E4M3_SAFE_MAX / _so_amax

        if _need_verify:
            _FP8MM_DBG[_key] = _count + 1
            import habana_frameworks.torch.core as _htc
            _htc.mark_step()
            # mark_step submits lazy work but does not guarantee the reduction
            # result is host-readable. Without this sync, calibration printed
            # impossible negative `abs().amax()` values from synthetic warmup.
            torch.hpu.synchronize()
            _ai = float(_si_amax)
            _ao = float(_so_amax)
            _fixed_i = float(self.scale_input)
            _fixed_o = float(self.scale_other)
            _pi = _ai * _fixed_i
            _po = _ao * _fixed_o
            print(
                f"[FP8MM_VERIFY] id={self.instance_id} fixed={int(self.fixed_scale)} "
                f"in{tuple(input.shape)} amax={_ai:.7g} scale={_fixed_i:g} peak={_pi:.7g} | "
                f"oth{tuple(other.shape)} amax={_ao:.7g} scale={_fixed_o:g} peak={_po:.7g}",
                flush=True,
            )
            if self.verify_strict and (_pi > _FP8_E4M3_SAFE_MAX or _po > _FP8_E4M3_SAFE_MAX):
                raise RuntimeError(
                    f"FP8 attention fixed scale saturation: id={self.instance_id} "
                    f"input_peak={_pi:.7g} other_peak={_po:.7g} "
                    f"safe_max={_FP8_E4M3_SAFE_MAX:g}"
                )

        qinput = self.quant_input(input, _s_i)
        qother = self.quant_input(other, _s_o)
        output = self.matmul_fp8(
            qinput,
            qother,
            out_dtype=torch.bfloat16,
            scale_input_inv=1.0 / _s_i,
            scale_other_inv=1.0 / _s_o,
        )
        return output


class SlicedFusedSDPABase(torch.nn.Module):
    """Base class for sliced FusedSDPA modules.

    Encapsulates the common slicing initialization (chunk size, padded chunk
    counts, graph-break setup) shared by :class:`SlicedFusedSDPA` and
    :class:`SlicedFP8FusedSDPA`.
    """

    def __init__(self):
        super().__init__()
        self.enable_slicing = self._setup_slicing()

    def _setup_slicing(self) -> bool:
        enable_slicing = get_config().enable_fsdpa_slicing
        if not enable_slicing:
            return False

        if get_config().bucketing_strategy != 'pad':
            logger().warning_once(
                'FusedSDPA slicing is only compatible with padding-based bucketing strategy, slicing in FusedSDPA will be disabled.'
            )
            return False

        if get_config().merged_prefill:
            logger().warning_once(
                'FusedSDPA slicing is not compatible with merged prefill, slicing in FusedSDPA will be disabled.')
            return False

        if not get_config().use_bucketing:
            logger().warning_once(
                'FusedSDPA slicing requires bucketing to be enabled, slicing in FusedSDPA will be disabled.')
            return False

        from vllm_gaudi.extension.bucketing.common import get_bucketing_manager
        bucketing_manager = get_bucketing_manager()
        assert bucketing_manager is not None and bucketing_manager.initialized, 'Bucketing manager should be instantiated and initialized to enable FusedSDPA slicing.'

        from vllm_gaudi.extension.bucketing.padding_aware import PaddingAwareBucketingStrategy
        strategy = bucketing_manager.get_bucketing_strategy()
        assert isinstance(
            strategy,
            PaddingAwareBucketingStrategy), 'Bucketing strategy should be Padding-Aware to enable FusedSDPA slicing.'

        max_num_batched_tokens = bucketing_manager.max_num_batched_tokens
        block_size = bucketing_manager.block_size
        slice_thld_default = min(max_num_batched_tokens, 8192)
        slice_thld = int(os.getenv("VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD", str(slice_thld_default)))
        assert slice_thld > block_size, 'Invalid FusedSDPA slice sequence length threshold, the threshold should be greater than the block size.'
        assert slice_thld >= 1024, 'The FusedSDPA slice sequence length threshold should be greater than or equal to 1024 to ensure the chunk sizes are valid for the attention kernel.'
        if slice_thld < slice_thld_default:
            logger().warning_once(
                f'The FusedSDPA slice sequence length threshold {slice_thld} is less than the default {slice_thld_default} which is not recommended.'
            )

        # defaults to half of the threshold and round up by 1024
        chunk_size_default = math.ceil(slice_thld // 2 / 1024) * 1024
        chunk_size = int(os.getenv("VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE", str(chunk_size_default)))
        if chunk_size % 1024 != 0:
            chunk_size = math.ceil(chunk_size / 1024) * 1024
            logger().warning_once('Rounded up the chunk size for FusedSDPA slicing to the next multiple of 1024.')
        assert chunk_size > block_size and chunk_size <= slice_thld, 'Invalid FusedSDPA slice chunk size, the chunk size should be between the block size and the slice sequence length threshold.'

        self.slice_thld = slice_thld
        self.chunk_size = chunk_size

        # should align with the default value in PaddingAwareBucketingStrategy
        max_query_pad_default = math.ceil(max_num_batched_tokens / 4)
        max_query_pad = int(os.getenv("VLLM_PROMPT_QUERY_BUCKET_PAD_MAX", str(max_query_pad_default)))
        assert max_query_pad >= block_size, 'Invalid max query padding, the max query padding should be greater than or equal to the block size.'
        self.num_padded_query_chunks = math.ceil(max_query_pad / self.chunk_size)

        # should align with the default value in PaddingAwareBucketingStrategy
        max_ctx_pad_default = math.ceil(max_num_batched_tokens / block_size)
        max_ctx_pad = int(os.getenv("VLLM_PROMPT_CTX_BUCKET_PAD_MAX", str(max_ctx_pad_default)))
        self.num_padded_ctx_chunks = math.ceil(max_ctx_pad * block_size / self.chunk_size)

        import habana_frameworks.torch as ht
        is_lazy = ht.utils.internal.is_lazy()
        self._with_graph_breaks = os.getenv("VLLM_HPU_FSDPA_SLICE_WITH_GRAPH_BREAKS",
                                            str(is_lazy)).strip().lower() in ['true', 't', '1', 'yes', 'y', 'on']
        if self._with_graph_breaks and not is_lazy:
            logger().warning_once('FusedSDPA slicing graph breaks are only supported in lazy mode. '
                                  'Disabling graph breaks for eager/compile mode to avoid Synapse compiler failures.')
            self._with_graph_breaks = False
        if self._with_graph_breaks:
            self._break_graph = ht.core.mark_step

        msg = (f"FusedSDPA slicing is enabled with sequence length threshold {slice_thld}, "
               f"chunk size {self.chunk_size}, num padded query chunks {self.num_padded_query_chunks}, "
               f"num padded ctx chunks {self.num_padded_ctx_chunks}, with graph breaks {self._with_graph_breaks}.")
        logger().debug_once(msg)

        return True

    def maybe_break_graph(self):
        if self._with_graph_breaks:
            self._break_graph()

    @staticmethod
    def _merge_chunk(last_out, last_m, last_linv, chunk_out, chunk_m, chunk_linv):
        """Online softmax rescaling merge of two attention chunks."""
        if last_out is None or last_m is None or last_linv is None:
            return chunk_out, chunk_m, chunk_linv
        new_m = torch.maximum(last_m, chunk_m)
        last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
        chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
        new_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
        new_out = (last_linv_rescaled * new_linv) * last_out + (chunk_linv_rescaled * new_linv) * chunk_out
        return new_out, new_m, new_linv

    def _chunked_attention(self, q, k, v, attn_mask, dropout_p, scale, softmax_mode, chunk_kernel_fn):
        """Run chunked attention with online softmax rescaling.

        Args:
            q, k, v: Query, key, value tensors (after GQA reshape if needed).
            attn_mask: Attention mask tensor.
            dropout_p: Dropout probability.
            scale: Attention scale factor.
            softmax_mode: Softmax mode string.
            chunk_kernel_fn: Callable
                ``(q, k, v, mask, dropout_p, scale, is_causal, softmax_mode)``
                returning ``(out, m, linv)`` all as float32.

        Returns:
            Concatenated output tensor in float32.
        """
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        prefix_len = kv_len - q_len

        chunk_outputs = []
        num_q_chunks = math.ceil(q_len / self.chunk_size)
        num_prefix_chunks = math.ceil(prefix_len / self.chunk_size)
        for q_chunk_idx in range(num_q_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :].contiguous()

            last_out = None
            last_m = None
            last_linv = None

            # the causal part
            for kv_chunk_idx in range(0, num_q_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.chunk_size
                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()

                # Always pass explicit mask for the diagonal chunk (kv_chunk_idx==0)
                # to ensure numerical consistency. The kernel's is_causal=True path
                # uses a different internal algorithm that can diverge from the
                # explicit mask path even when both encode the same triangular pattern.
                # For non-diagonal chunks within the padded region, also pass mask.
                mask_chunk = (attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
                              if kv_chunk_idx == 0 or kv_chunk_idx < self.num_padded_query_chunks else None)

                self.maybe_break_graph()

                chunk_out, chunk_m, chunk_linv = chunk_kernel_fn(q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p,
                                                                 scale, False, softmax_mode)

                last_out, last_m, last_linv = self._merge_chunk(last_out, last_m, last_linv, chunk_out, chunk_m,
                                                                chunk_linv)

                self.maybe_break_graph()

            # the context part
            for kv_chunk_idx in range(num_prefix_chunks):
                kv_start = prefix_len - (kv_chunk_idx + 1) * self.chunk_size
                kv_start = max(kv_start, 0)
                kv_end = prefix_len - kv_chunk_idx * self.chunk_size
                k_chunk = k[..., kv_start:kv_end, :].contiguous()
                v_chunk = v[..., kv_start:kv_end, :].contiguous()
                # use mask only for the chunks that may have padding
                mask_chunk = (attn_mask[..., q_start:q_end, kv_start:kv_end].contiguous()
                              if kv_chunk_idx < self.num_padded_ctx_chunks else None)

                self.maybe_break_graph()

                chunk_out, chunk_m, chunk_linv = chunk_kernel_fn(q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p,
                                                                 scale, False, softmax_mode)

                assert not (last_out is None or last_m is None or last_linv is None)
                last_out, last_m, last_linv = self._merge_chunk(last_out, last_m, last_linv, chunk_out, chunk_m,
                                                                chunk_linv)

                self.maybe_break_graph()
            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        return torch.cat(chunk_outputs, dim=-2)


class SlicedFusedSDPA(SlicedFusedSDPABase):
    """Standalone module for BF16 sliced FusedSDPA.

    Extracting the sliced attention path into its own ``nn.Module`` allows it
    to be wrapped with ``torch.compile``, ``ht.hpu.wrap_in_hpu_graph``, or
    any other module-level wrapper independently of the dispatch logic in
    :class:`ModuleFusedSDPA`.
    """

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
        assert is_causal and attn_mask is not None

        from habana_frameworks.torch.hpex.kernels.FusedSDPA import is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        gqa = is_gqa(query, key)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(query, key, value, attn_mask)
        else:
            q, k, v, attn_mask = (query, key, value, attn_mask)
        if scale is None:
            scale = 1.0 / (query.shape[-1]**0.5)

        def chunk_kernel(q_c, k_c, v_c, mask_c, dp, sc, is_c, sm):
            res = torch.ops.hpu.sdpa_recomp_fwd(q_c, k_c, v_c, mask_c, dp, sc, is_c, True, sm, None, 'right')
            out, m, linv = tuple((gqa_output_reshape(x) if gqa else x).to(torch.float32) for x in res[:3])
            return out, m, linv

        output = self._chunked_attention(q, k, v, attn_mask, dropout_p, scale, softmax_mode, chunk_kernel)
        return output.to(q.dtype)


class ModuleFusedSDPA(torch.nn.Module):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA
        self._sliced_module = SlicedFusedSDPA()

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
        sinks=None,
    ):
        if (self._sliced_module.enable_slicing
                and key.shape[-2] >= self._sliced_module.slice_thld  # apply for kv_len >= slice_thld only
                and query.shape[0] == 1  # bs should be 1 for prefix-prefill
                and query.shape[-2] != key.shape[-2]  # normal prefill with q_len == kv_len route to the default
                and is_causal and attn_mask is not None  # only supports causal attention with mask
                and padding_side == 'right'  # supports right padding only for the chunks that may have padding
                and window_size is None  # slicing is not compatible with sliding window attention
                and sinks is None  # slicing is not compatible with kernel fusion with sinks
            ):
            return self._sliced_module(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode)

        if is_causal and attn_mask is not None:
            # TODO: causal + attn_bias is not yet supported
            is_causal = False
            valid_sequence_lengths = None

        if window_size is not None:
            return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                                recompute_mode, valid_sequence_lengths, padding_side, False, False,
                                                window_size, sinks)
        else:
            return self._hpu_kernel_fsdpa.apply(query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode,
                                                recompute_mode, valid_sequence_lengths, padding_side, False, False,
                                                (-1, -1), sinks)


class SlicedFP8FusedSDPA(SlicedFusedSDPABase):
    """Standalone module for FP8 sliced FusedSDPA.

    Like :class:`SlicedFusedSDPA`, extracting the sliced path enables
    wrapping with ``torch.compile`` or ``ht.hpu.wrap_in_hpu_graph``.
    Expects pre-quantized FP8 inputs; dequantises chunk outputs to
    BF16/FP32 before the online-softmax rescaling merge.
    """

    def __init__(self, parent):
        super().__init__()
        # Store parent reference without registering as a submodule
        # to avoid circular module graph while sharing scale tensors.
        object.__setattr__(self, '_parent', parent)

    def _dequant_output(self, output):
        return torch.ops.hpu.cast_from_fp8(output, self._parent.d_scale_output, torch.bfloat16)

    def _fp8_fsdpa_fwd(self, q, k, v, attn_mask, dropout_p, scale, is_causal, softmax_mode):
        results = torch.ops.hpu.fp8_sdpa_recomp_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            True,  # requires_backward
            softmax_mode,
            self._parent.d_scale_q,
            self._parent.d_scale_k,
            self._parent.d_scale_v,
            self._parent.scale_amax,
            self._parent.d_scale_output,
            self._parent.descale_amax,
            False,  # is_amax_s
            False,  # is_amax_o
            None,  # valid_seq_len
            "right",  # padding_side
            (-1, -1),  # window_size
            None,  # sinks
        )
        return results

    def forward(self, query, key, value, attn_mask, dropout_p, is_causal, scale, softmax_mode):
        assert is_causal and attn_mask is not None

        from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        gqa = is_gqa(query, key)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(query, key, value, attn_mask)
        else:
            q, k, v, attn_mask = (query, key, value, attn_mask)
        softmax_mode = softmax_mode if softmax_mode == "fp32" else "fast"
        if scale is None:
            scale = 1.0 / (query.shape[-1]**0.5)

        def chunk_kernel(q_c, k_c, v_c, mask_c, dp, sc, is_c, sm):
            res = self._fp8_fsdpa_fwd(q_c, k_c, v_c, mask_c, dp, sc, is_c, sm)
            out, m, linv = tuple(gqa_output_reshape(x) if gqa else x for x in res[:3])
            m = m.to(torch.float32)
            linv = linv.to(torch.float32) * (128.0 if sm == "fast" else 1.0)
            out = self._dequant_output(out).to(torch.float32)
            return out, m, linv

        return self._chunked_attention(q, k, v, attn_mask, dropout_p, scale, softmax_mode, chunk_kernel)


class ModuleFP8FusedSDPA(torch.nn.Module):

    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'FP8 fusedSDPA kernel is None'
        self.fp8_fused_sdpa = fusedSDPA

        # q_scale_s / d_scale_s: the fp8 quant / dequant scale for the softmax
        # weights S *inside* the kernel. The fp8 SDPA does
        #   S = exp(scores - rowmax)          # in (0, 1], row-max exactly 1.0
        #   S_fp8 = to_fp8(S * q_scale_s)      # <-- FLUSH happens here if scale=1
        #   O = (S_fp8 @ V_fp8) * d_scale_s * d_scale_v / rowsum
        # Over long context the bulk of S is ~1/seqlen (e.g. ~4e-4 @ 2300 tok),
        # so with the old q_scale_s=1.0 the internal fp8 cast flushed those
        # weights to zero (same mechanism as FP8Matmul) -> the FUSED PREFILL
        # produced garbage past ~2k tokens. Fix q_scale_s to 224
        # (= _FP8_E4M3_SAFE_MAX): the max weight 1.0 maps to 224 (full e4m3
        # range, still below the broken [256,448] top bin) and every small weight
        # is lifted 224x so it survives the cast. d_scale_s = 1/224 is the EXACT
        # reciprocal, so the AV descale cancels it perfectly (pure change of
        # variables; residual is only fp8 rounding). Static pair -- no per-step
        # amax needed because softmax output is bounded by 1.0 by construction.
        self.descale_amax = torch.tensor(1.0 / _FP8_E4M3_SAFE_MAX, dtype=torch.float32)
        self.scale_amax = torch.tensor(_FP8_E4M3_SAFE_MAX, dtype=torch.float32)
        self.scale_q = torch.tensor(1.0, dtype=torch.float32)
        self.scale_k = torch.tensor(1.0, dtype=torch.float32)
        self.scale_v = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_q = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_k = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_v = torch.tensor(1.0, dtype=torch.float32)
        self.d_scale_output = torch.tensor(1.0, dtype=torch.float32)
        self._sliced_module = SlicedFP8FusedSDPA(parent=self)

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, torch.float8_e4m3fn)[0]

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
    ):

        # Q/K/V use FIXED scales set in __init__ (scale_q/k/v = 1.0, d_scale_* =
        # 1.0) -- NO per-step amax. They are ~O(1..8) and never flush (short
        # prompts already work at scale 1.0), so the expensive dynamic
        # query.abs().amax() is unnecessary here. The load-bearing long-context
        # fix is the FIXED q_scale_s = self.scale_amax = 224 (d_scale_s = 1/224),
        # set in __init__: it lifts the kernel's internal softmax weights S
        # (bounded in (0,1]) out of the fp8 flush floor before the AV matmul, with
        # an exact reciprocal descale -- no amax needed because S <= 1.0 always.
        qinput = self.quant_input(query, self.scale_q).detach()
        kinput = self.quant_input(key, self.scale_k).detach()
        vinput = self.quant_input(value, self.scale_v).detach()

        bs = query.shape[0]
        q_len = query.shape[-2]
        kv_len = key.shape[-2]
        if (self._sliced_module.enable_slicing and kv_len >= self._sliced_module.slice_thld \
                and bs == 1  # bs should be 1 for chunked prefill
                and q_len != kv_len  # normal causal prefill route to the default dispatch for better performance
                and is_causal and attn_mask is not None  # only supports causal attention with mask
                and padding_side == 'right'  # currently only supports right padding for the chunks that may have padding
                and window_size is None  # slicing is not compatible with sliding window attention
            ):
            return self._sliced_module(qinput, kinput, vinput, attn_mask, dropout_p, is_causal, scale,
                                       softmax_mode).to(query.dtype)

        if is_causal and attn_mask is not None:
            # TODO: causal + attn_bias is not yet supported
            is_causal = False
            valid_sequence_lengths = None

        results = self.fp8_fused_sdpa(
            qinput,
            kinput,
            vinput,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            softmax_mode=softmax_mode,
            d_scale_q=self.d_scale_q,
            d_scale_k=self.d_scale_k,
            d_scale_v=self.d_scale_v,
            q_scale_s=self.scale_amax,
            # q_scale_o=1 / 1.0,
            d_scale_s=self.descale_amax,
            is_amax_s=False,
            valid_seq_len=valid_sequence_lengths,
            seq_padding_type=padding_side,
        )

        output = results[0]
        return output


def pad_list(input, target_len, val_generator):
    padding = target_len - len(input)
    if padding > 0:
        input.extend(itertools.islice(val_generator, padding))
    return input


def align_and_pad(data, bucketing, padding_gen):
    bs = len(data)
    target_bs, target_len = bucketing
    if target_bs == 1 and bs > 1:
        data = [list(itertools.chain(*data))]
    data = [pad_list(x, target_len, padding_gen) for x in data]
    padding = itertools.islice(padding_gen, target_len)
    data = pad_list(data, target_bs, itertools.tee(padding, target_bs - len(data)))
    return data


def with_default(value: Optional[Any], default: Any) -> Any:
    if value is not None:
        return value
    return default
