"""Per-DECODE-STEP tensor dump for the LAZY-vs-EAGER value-level diff (the §6c hunt).

Bug A (PROJECT_FINDINGS §6c): GLM-5.1 decode drifts deterministically under
PT_HPU_LAZY_MODE=1 in EVERY config (captured or not, collectives or not) and is
coherent under lazy=0. The op that lowers wrongly under lazy is unknown. This module
captures, for the FIRST decoder layer only and the first N decode steps, the
boundary tensors of the three suspects:

  layer0.step<k>.pt   : positions, hidden_in, residual_in, hidden_out, residual_out
  mla0.step<k>.pt     : q, k_c_normed, k_pe (inputs) + attn_out (output) of
                        HPUMLAAttention.forward_impl (whatever version is installed —
                        wraps LAST, after glm_mla_decode_cache_fix)
  mqa0.step<k>.pt     : ql_nope, q_pe (inputs) + out (output) of impl.forward_mqa —
                        the flat_pa_mla read chain, suspect #1

Run the SAME prompt under lazy=1(+--enforce-eager) and lazy=0 on the SAME image with
HPU_DECODE_DUMP_DIR pointed at two different dirs, then diff step-by-step on CPU:
the first tensor that diverges beyond bf16 noise while its INPUTS still match names
the broken stage. Inputs match early because greedy outputs are identical until the
first wrong token (~step 8-10).

Gates: HPU_DECODE_DUMP=<N steps> (0=OFF=inert), HPU_DECODE_DUMP_DIR (default
/opt/hpu-dsa-ext/decode_dump). Per-rank files (TP shards differ per rank) — compare
rank r to rank r. Host syncs every step: DIAGNOSIS ONLY.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("vllm")

_N = int(os.environ.get("HPU_DECODE_DUMP", "0") or "0")
_DIR = os.environ.get("HPU_DECODE_DUMP_DIR", "/opt/hpu-dsa-ext/decode_dump")


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        return int(os.environ.get("RANK", "0") or "0")


def _save(kind: str, step: int, tensors: dict) -> None:
    import torch
    os.makedirs(_DIR, exist_ok=True)
    payload = {}
    for k, v in tensors.items():
        try:
            payload[k] = v.detach().to("cpu") if hasattr(v, "detach") else v
        except Exception as ex:  # noqa: BLE001
            payload[k] = f"<dump-error {ex}>"
    torch.save(payload, os.path.join(_DIR, f"{kind}.rank{_rank()}.step{step}.pt"))


def install() -> None:
    if _N <= 0:
        return
    import torch  # noqa: F401

    # ---- 1) decoder-layer in/out for a SPREAD of layers ---------------------
    # (dense layers are 0-2, MoE starts at 3 — the spread brackets the dense/MoE
    #  boundary then log-spaces the deep stack; layer_idx attr per glm_layer_dump)
    _layer_targets = set(
        int(s) for s in os.environ.get(
            "HPU_DECODE_DUMP_LAYERS", "0,3,4,10,20,40,60,77"
        ).replace(" ", "").split(",") if s != "")
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
        _olayer = DeepseekV2DecoderLayer.forward
        lsteps = {}  # layer_idx -> decode step counter

        def layer_forward(self, positions, hidden_states, residual, *a, **k):
            lidx = getattr(self, "layer_idx", -1)
            is_decode = (hidden_states.dim() >= 1 and hidden_states.shape[0] == 1)
            if lidx in _layer_targets and is_decode and lsteps.get(lidx, 0) < _N:
                step = lsteps.get(lidx, 0)
                lsteps[lidx] = step + 1
                hin = hidden_states
                rin = residual
                out = _olayer(self, positions, hidden_states, residual, *a, **k)
                hout, rout = out if isinstance(out, tuple) else (out, None)
                _save(f"layer{lidx}", step, {
                    "positions": positions, "hidden_in": hin, "residual_in": rin,
                    "hidden_out": hout, "residual_out": rout})
                return out
            return _olayer(self, positions, hidden_states, residual, *a, **k)

        layer_forward._decode_dump = True
        DeepseekV2DecoderLayer.forward = layer_forward
    except Exception as ex:  # noqa: BLE001
        log.warning("[decode-dump] layer wrap failed: %s", ex)

    # ---- 2) MLA forward_impl (decode branch) in/out -------------------------
    try:
        from vllm_gaudi.attention.oot_mla import HPUMLAAttention
        _omla = HPUMLAAttention.forward_impl  # whatever is installed (cache fix etc.)
        mstate = {"id": None, "n": 0}

        def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *a, **k):
            decode = not getattr(attn_metadata, "is_prompt", True)
            is_m0 = mstate["id"] in (None, id(self))
            if decode and is_m0 and mstate["n"] < _N:
                mstate["id"] = id(self)
                step = mstate["n"]
                mstate["n"] = step + 1
                out = _omla(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *a, **k)
                _save("mla0", step, {"q": q, "k_c_normed": k_c_normed,
                                     "k_pe": k_pe, "attn_out": out})
                return out
            return _omla(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, *a, **k)

        forward_impl._decode_dump = True
        HPUMLAAttention.forward_impl = forward_impl
    except Exception as ex:  # noqa: BLE001
        log.warning("[decode-dump] MLA wrap failed: %s", ex)

    # ---- 3) impl.forward_mqa (flat_pa_mla read chain) in/out ----------------
    try:
        from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl
        _omqa = HPUMLAImpl.forward_mqa
        qstate = {"id": None, "n": 0}

        def forward_mqa(self, ql_nope, q_pe, kv_c_and_k_pe_cache, attn_metadata, *a, **k):
            decode = not getattr(attn_metadata, "is_prompt", True)
            is_q0 = qstate["id"] in (None, id(self))
            if decode and is_q0 and qstate["n"] < _N:
                qstate["id"] = id(self)
                step = qstate["n"]
                qstate["n"] = step + 1
                out = _omqa(self, ql_nope, q_pe, kv_c_and_k_pe_cache, attn_metadata, *a, **k)
                _save("mqa0", step, {"ql_nope": ql_nope, "q_pe": q_pe, "out": out})
                return out
            return _omqa(self, ql_nope, q_pe, kv_c_and_k_pe_cache, attn_metadata, *a, **k)

        forward_mqa._decode_dump = True
        HPUMLAImpl.forward_mqa = forward_mqa
    except Exception as ex:  # noqa: BLE001
        log.warning("[decode-dump] forward_mqa wrap failed: %s", ex)

    # ---- 4) MoE apply_monolithic I/O for offline op replay -------------------
    # Dumps (x, router_logits, out) of the plugin's stock all-expert MoE path for a
    # few layers/steps. Layer index derived from CALL ORDER within a decode step
    # (bs=1 decode calls MoE layers 3..77 sequentially; cycle detected via the
    # first repeated layer object). Gated by the same HPU_DECODE_DUMP.
    _moe_layers = set(
        int(s) for s in os.environ.get(
            "HPU_DECODE_DUMP_MOE_LAYERS", "3,4,20").replace(" ", "").split(",") if s != "")
    try:
        from hpu_moe_ext.moe_wna16_hpu import MoeWNA16HPUMethod
        _oam = MoeWNA16HPUMethod.apply_monolithic
        mo = {"order": [], "cycle": 0, "calls": 0}

        def apply_monolithic(self, layer, x, router_logits, **kw):
            out = _oam(self, layer, x, router_logits, **kw)
            try:
                rows = x.view(-1, x.shape[-1]).shape[0]
                if rows == 1:  # bs=1 decode only
                    lid = id(layer)
                    if mo["cycle"] == 0:
                        if mo["order"] and lid == mo["order"][0]:
                            mo["cycle"] = len(mo["order"])
                        else:
                            mo["order"].append(lid)
                    n = mo["calls"]
                    mo["calls"] = n + 1
                    cyc = mo["cycle"] or 10 ** 9
                    layer_idx = 3 + (n % cyc if mo["cycle"] else len(mo["order"]) - 1)
                    step = n // cyc if mo["cycle"] else 0
                    if layer_idx in _moe_layers and step < min(_N, 3):
                        o = out[0] if isinstance(out, tuple) else out
                        _save(f"moe{layer_idx}", step, {
                            "x": x, "router_logits": router_logits, "out": o})
            except Exception as ex:  # noqa: BLE001
                log.warning("[decode-dump] moe dump error: %s", ex)
            return out

        apply_monolithic._decode_dump = True
        MoeWNA16HPUMethod.apply_monolithic = apply_monolithic
    except Exception as ex:  # noqa: BLE001
        log.warning("[decode-dump] apply_monolithic wrap failed: %s", ex)

    # ---- 5) the STOCK MoE op itself (VllmMixtureOfExpertsOpWNA16.forward) ----
    # Captures EXACTLY what the op consumed (x, LOCAL routing table, weights) and
    # produced — the precise I/O for the offline CPU-fp32 replay that decides
    # whether the op is numerically wrong under lazy. Same call-order layer index.
    try:
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpWNA16
        _oop = VllmMixtureOfExpertsOpWNA16.forward
        op_state = {"order": [], "cycle": 0, "calls": 0}

        def op_forward(self, hidden_states, expert_routing_table, router_weights,
                       permuted_weights=True, activation="silu"):
            out = _oop(self, hidden_states, expert_routing_table, router_weights,
                       permuted_weights, activation)
            try:
                if hidden_states.view(-1, hidden_states.shape[-1]).shape[0] == 1:
                    oid = id(self)
                    if op_state["cycle"] == 0:
                        if op_state["order"] and oid == op_state["order"][0]:
                            op_state["cycle"] = len(op_state["order"])
                        else:
                            op_state["order"].append(oid)
                    n = op_state["calls"]
                    op_state["calls"] = n + 1
                    cyc = op_state["cycle"]
                    layer_idx = 3 + (n % cyc if cyc else len(op_state["order"]) - 1)
                    step = n // cyc if cyc else 0
                    if layer_idx in _moe_layers and step < min(_N, 3):
                        _save(f"moeop{layer_idx}", step, {
                            "x": hidden_states, "tid_local": expert_routing_table,
                            "tw": router_weights, "out": out})
            except Exception as ex:  # noqa: BLE001
                log.warning("[decode-dump] moeop dump error: %s", ex)
            return out

        op_forward._decode_dump = True
        VllmMixtureOfExpertsOpWNA16.forward = op_forward
    except Exception as ex:  # noqa: BLE001
        log.warning("[decode-dump] moe-op wrap failed: %s", ex)

    # ---- 6) HOST-side persistent-buffer dump of ONE MoE op (lazy-safe) -------
    # Under HPU-graph replay the python forward runs only at CAPTURE, so section
    # 5's saved tensors hold capture-time junk on replayed steps. Here the op I/O
    # is copy_()'d in-graph into persistent device buffers (the copies replay),
    # and a wrapped runner step saves them HOST-side AFTER the step — the only
    # replay-truthful observation. Files: moeophost<L>.rank<r>.step<s>.pt.
    _host_layer = int(os.environ.get("HPU_MOE_OP_HOSTDUMP_LAYER", "4"))
    try:
        from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpWNA16
        from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
        hb = {"bufs": None, "calls": 0, "order": [], "cycle": 0, "step": 0}
        _oop6 = VllmMixtureOfExpertsOpWNA16.forward  # section-5 wrapper; chain it

        def op_forward6(self, hidden_states, expert_routing_table, router_weights,
                        permuted_weights=True, activation="silu"):
            out = _oop6(self, hidden_states, expert_routing_table, router_weights,
                        permuted_weights, activation)
            try:
                if hidden_states.view(-1, hidden_states.shape[-1]).shape[0] == 1:
                    oid = id(self)
                    if hb["cycle"] == 0:
                        if hb["order"] and oid == hb["order"][0]:
                            hb["cycle"] = len(hb["order"])
                        else:
                            hb["order"].append(oid)
                    n = hb["calls"]
                    hb["calls"] = n + 1
                    cyc = hb["cycle"]
                    layer_idx = 3 + (n % cyc if cyc else len(hb["order"]) - 1)
                    if layer_idx == _host_layer:
                        if hb["bufs"] is None:
                            with torch.inference_mode(False):
                                hb["bufs"] = {
                                    "x": torch.zeros_like(hidden_states),
                                    "tid_local": torch.zeros_like(expert_routing_table),
                                    "tw": torch.zeros_like(router_weights),
                                    "out": torch.zeros_like(out),
                                }
                        b = hb["bufs"]
                        b["x"].copy_(hidden_states)
                        b["tid_local"].copy_(expert_routing_table)
                        b["tw"].copy_(router_weights)
                        b["out"].copy_(out)
            except Exception as ex:  # noqa: BLE001
                log.warning("[decode-dump] hostdump op wrap error: %s", ex)
            return out

        op_forward6._decode_dump = True
        VllmMixtureOfExpertsOpWNA16.forward = op_forward6

        _orun = HPUModelRunner._execute_model_generic

        def _exec6(self, token_ids, position_ids, attn_metadata, logits_indices,
                   kv_caches, lora_logits_mask, lora_mask, warmup_mode=False, **kw):
            out = _orun(self, token_ids, position_ids, attn_metadata, logits_indices,
                        kv_caches, lora_logits_mask, lora_mask, warmup_mode, **kw)
            try:
                if (not warmup_mode and hb["bufs"] is not None
                        and not getattr(attn_metadata, "is_prompt", True)
                        and hb["step"] < min(_N, 3)):
                    s = hb["step"]
                    hb["step"] = s + 1
                    _save(f"moeophost{_host_layer}", s, dict(hb["bufs"]))
            except Exception as ex:  # noqa: BLE001
                log.warning("[decode-dump] hostdump save error: %s", ex)
            return out

        _exec6._decode_dump = True
        HPUModelRunner._execute_model_generic = _exec6
    except Exception as ex:  # noqa: BLE001
        log.warning("[decode-dump] hostdump install failed: %s", ex)

    log.warning("[decode-dump] installed per-decode-step dumps (N=%d, dir=%s). "
                "HPU_DECODE_DUMP", _N, _DIR)
