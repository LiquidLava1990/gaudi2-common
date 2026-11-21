#!/usr/bin/env python3
# ==================================================================================================
# autoconfig.py -- PLUG-AND-PLAY serving-config auto-detector + auto-optimizer for the Gaudi-2 stack.
# Point it at ANY model dir; it introspects config.json + the chat template and emits the optimal
# op-stack, tool/reasoning parsers, multimodal flags, tensor-parallel size, KV treatment, and the
# hand-won speed-lever stack -- turning "hand-write a bespoke env per model" into "drop in a model ->
# auto-configured". Encodes the champion recipes (GLM int4-MoE-MLA-DSA / M3 fp8-MoE-GQA-VL / Qwen
# dense-hybrid-Mamba) AND the whole catalog of proven levers, each behind the coherence/tool gate that
# makes it safe. Unknown families get a best-effort config + LOUD notes.
#
# SAFE-DEFAULT POLICY (default profile = AGENT_TOOLS): nothing lossy or tool-breaking auto-enables.
#   FREE levers (byte-exact/coherent) auto-apply on precondition; COHERENCE-RISKY + CAPACITY levers are
#   gated behind an explicit profile and/or an on-card sentinel. depth-skip is HARD-EXCLUDED for tool
#   users (it silently breaks tool-call formatting -- see memory glm52-depthskip-breaks-tools).
#
#   python3 autoconfig.py <model_dir> [--profile agent-tools|agent-capacity|chat-throughput|capacity|max|pack]
#   python3 autoconfig.py <model_dir> --emit-env [--profile ...]      # write a containers.sh env
#   python3 autoconfig.py --validate                                  # reproduce the champions
# ==================================================================================================
import glob
import json
import os
import re
import sys

_PROFILES = frozenset({"agent-tools", "agent-capacity", "chat-throughput", "capacity", "max", "pack"})
_ENGINE_DEFAULT_MAX_NUM_SEQS = 128

# ---- arch classes VALIDATED on-card, with their proven champion TP (default emit; min-fit only for unknowns) ----
_CHAMPION_TP = {
    "GlmMoeDsaForCausalLM": 8,                         # GLM-5.2  (MoE / MLA / int4 / DSA)
    "MiniMaxM3SparseForConditionalGeneration": 8,      # MiniMax-M3 (MoE / GQA / fp8 / DSA / VL)
    "Qwen3_5ForConditionalGeneration": 8,              # Qwen3.6-27B (dense / Mamba-hybrid / fp8 / VL)
    "Qwen3MoeForCausalLM": 8,                          # Qwen3-30B-A3B (MoE / GQA / int4) -- TP1 fits, 8 = proven
}
_VERIFIED_ARCHS = frozenset(_CHAMPION_TP)

_GRAFT_ALIASES = {"GlmMoeDsaForCausalLM": (("glm-5.2", "glm52", "glm5.2"), "glm52-fp8-other")}

# ---- vLLM-gaudi:next multimodal registry snapshot (arch classes whose vision/audio tower actually serves).
#      An arch with a vision_config but NOT in this set = TEXT-ONLY on this stack (image inputs rejected). ----
_VLLM_MM_ARCHS = frozenset({
    "GLM4VForCausalLM", "Glm4vForConditionalGeneration", "Glm4vMoeForConditionalGeneration",
    "GlmAsrForConditionalGeneration", "GlmOcrForConditionalGeneration",
    "Gemma3ForConditionalGeneration", "Gemma3nForConditionalGeneration", "Gemma4ForConditionalGeneration",
    "Idefics3ForConditionalGeneration", "InternS1ForConditionalGeneration",
    "InternS1ProForConditionalGeneration", "InternS2PreviewForConditionalGeneration",
    "InternVLChatModel", "InternVLForConditionalGeneration",
    "LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration", "LlavaOnevisionForConditionalGeneration",
    "MiDashengLMModel", "MiniMaxVL01ForConditionalGeneration", "Molmo2ForConditionalGeneration",
    "MolmoForCausalLM", "PaliGemmaForConditionalGeneration", "Phi3VForCausalLM", "Phi4ForCausalLMV",
    "Phi4MMForCausalLM", "PixtralForConditionalGeneration", "Qwen2AudioForConditionalGeneration",
    "Qwen2VLForConditionalGeneration", "Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniModel",
    "Qwen2_5_VLForConditionalGeneration", "Qwen3ASRForConditionalGeneration", "Qwen3ASRRealtimeGeneration",
    "Qwen3OmniMoeForConditionalGeneration", "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration", "QwenVLForConditionalGeneration",
})

# ---- TOOL-CALL parser rules: TEMPLATE-SIGNATURE matched, ORDERED most-specific-first. Detection keys on
#      the assistant-side emission tokens, NOT the arch class (two Qwen models share an arch root but need
#      DIFFERENT parsers). parser=None means "tools advertised but no installed parser matches" -> warn, emit
#      nothing (never guess: a wrong --tool-call-parser crashes boot or silently drops every call). ----
_TOOL_RULES = [                     # (parser_or_None, [all-of these substrings], reason)
    ("glm47",        ["<arg_key>", "<arg_value>"], "GLM key/value XML (glm47 superset; glm45 needs a newline and misses GLM-5.2)"),
    ("minimax_m2",   ["<minimax:tool_call>"], "MiniMax-M2 surface tokens"),
    (None,           ["]<]minimax[>["], "MiniMax-M3: no installed parser matches (minimax_m2 crashes at boot); model emits tool text into content"),
    ("minimax",      ["<tool_calls>", "<invoke"], "MiniMax <tool_calls> block"),
    ("qwen3_coder",  ["<function=", "<parameter="], "Qwen coder XML (qwen3_xml is an alias)"),
    ("mistral",      ["[TOOL_CALLS]"], "Mistral"),
    ("deepseek_v31", ["<｜tool▁calls▁begin｜>"], "DeepSeek V3.1"),
    ("deepseek_v3",  ["<｜tool▁call▁begin｜>"], "DeepSeek V3"),
    ("kimi_k2",      ["<|tool_call_begin|>"], "Kimi-K2"),
    ("llama3_json",  ["<|python_tag|>"], "Llama pythonic/json"),
    ("hermes",       ["<tool_call>", "name", "arguments"], "Hermes JSON (generic; matched LAST)"),
]

# ---- REASONING close-token rules: ORDERED most-specific-first. "__think__" resolves by arch family. ----
_REASON_RULES = [                   # (close_token, parser_or "__think__" or "__plugin__:<name>")
    ("</mm:think>",   "__plugin__:mm_think"),   # MiniMax-M3: needs a shipped 12-line plugin, else none+warn
    ("</seed:think>", "seed_oss"),
    ("</think>",      "__think__"),
]


def _asdict(x):
    return x if isinstance(x, dict) else {}


def _sanitize_name(name):
    s = re.sub(r"[^a-z0-9._-]+", "-", str(name).lower()).strip("-._")
    return s or "model"


def _load(model_dir):
    p = os.path.join(model_dir, "config.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no config.json in {model_dir}")
    try:
        with open(p) as fh:
            cfg = json.load(fh)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        raise ValueError(f"config.json in {model_dir} is not valid JSON: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"config.json in {model_dir} must be a JSON object, got {type(cfg).__name__}")
    return cfg


def _load_chat_template(model_dir):
    """Return the chat-template TEXT from the first available source, or '' if none. Handles: chat_template.jinja
    (file), chat_template.json (file), tokenizer_config.json['chat_template'] (str OR list-of-{name,template})."""
    j = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(j):
        try:
            return open(j, encoding="utf-8").read()
        except OSError:
            pass
    cj = os.path.join(model_dir, "chat_template.json")
    if os.path.exists(cj):
        try:
            v = json.load(open(cj))
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("chat_template"), str):
                return v["chat_template"]
        except (json.JSONDecodeError, OSError):
            pass
    tc = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tc):
        try:
            v = _asdict(json.load(open(tc))).get("chat_template")
        except (json.JSONDecodeError, OSError):
            v = None
        if isinstance(v, str):
            return v
        if isinstance(v, list):  # list-of-{name,template} multi-template form -> concatenate all
            return "\n".join(t.get("template", "") for t in v if isinstance(t, dict))
    return ""


def _added_tokens_blob(model_dir):
    """Special/added tokens text (tokenizer_config added_tokens_decoder + added_tokens.json) -- lets us
    confirm a template's tool/think open token is a REAL vocab entry before emitting a parser for it."""
    blob = []
    for fn in ("tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"):
        p = os.path.join(model_dir, fn)
        if os.path.exists(p):
            try:
                blob.append(open(p, encoding="utf-8").read())
            except OSError:
                pass
    return "\n".join(blob)


def detect_arch(cfg):
    """Extract arch features that drive optimization. Robust to missing keys, non-dict values, and VL/omni
    models nesting the language arch under text_config/language_config/thinker_config/llm_config."""
    cfg = _asdict(cfg)
    tc = {}
    for _k in ("text_config", "language_config", "thinker_config", "llm_config"):
        v = cfg.get(_k)
        if isinstance(v, dict) and v:
            tc = v
            break

    def g(k, default=None):
        return tc[k] if k in tc else cfg.get(k, default)

    archs = cfg.get("architectures") or tc.get("architectures") or []
    if isinstance(archs, str):
        archs = [archs]
    elif not isinstance(archs, list):
        archs = []
    arch = archs[0] if archs else "?"
    mt = str(cfg.get("model_type") or "").lower()
    tcmt = str(tc.get("model_type") or "").lower()
    layer_types = g("layer_types") or []
    text = " ".join([str(arch), mt, tcmt,
                     " ".join(str(x) for x in layer_types) if isinstance(layer_types, list) else ""]).lower()

    # ---- weights quant ----
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        qc = tc.get("quantization_config")
    if not isinstance(qc, dict):
        qc = {}
    qmethod = str(qc.get("quant_method") or "").lower()
    bits = qc.get("bits")
    if bits is None:
        cg = qc.get("config_groups")
        if isinstance(cg, dict):
            for grp in cg.values():
                w = grp.get("weights") if isinstance(grp, dict) else None
                nb = w.get("num_bits") if isinstance(w, dict) else None
                if nb is not None:
                    bits = nb
                    break
    is_int4 = (bits == 4) or any(s in qmethod for s in ("gptq", "auto-round", "auto_round", "awq"))
    is_fp8 = (not is_int4) and ((bits == 8) or (qmethod == "fp8") or (qc.get("fmt") in ("e4m3", "e5m2")))
    # packing family -> selects the int4 op-stack (autoround->W4A8 monolithic; gptq->gptq_hpu kernel path)
    if any(s in qmethod for s in ("auto-round", "auto_round")):
        packing = "autoround"
    elif "gptq" in qmethod:
        packing = "gptq"
    elif "awq" in qmethod:
        packing = "awq"
    elif qmethod == "compressed-tensors":
        packing = "compressed-tensors"
    elif is_fp8:
        packing = "fp8"
    else:
        packing = "none"

    ig = qc.get("ignore") or qc.get("modules_to_not_convert") or []
    partial = len(ig) if isinstance(ig, (list, tuple)) else 0
    ec = qc.get("extra_config")
    if is_int4 and isinstance(ec, dict):
        partial += sum(1 for v in ec.values() if isinstance(v, dict) and v.get("bits") in (8, 16))

    n_experts = g("n_routed_experts") or g("num_local_experts") or g("num_experts")
    is_moe = bool(n_experts)
    top_k = g("num_experts_per_tok") or g("moe_top_k") or g("num_selected_experts")
    is_mla = bool(g("kv_lora_rank"))
    lt_has_linear = isinstance(layer_types, list) and any("linear" in str(x) or "mamba" in str(x) for x in layer_types)
    is_mamba_hybrid = lt_has_linear or any(k in tc for k in ("linear_conv_kernel_dim", "mamba_ssm_dtype", "linear_num_key_heads")) \
        or any(s in text for s in ("mamba", "gdn", "gateddelta", "lightning"))
    is_dsa = ("dsa" in text) or bool(g("index_topk")) or bool(g("sparse_attention_config"))
    has_mtp = bool(g("num_mtp_modules") or g("num_nextn_predict_layers") or g("mtp_num_hidden_layers"))

    # ---- multimodal modalities (config-pure; scan root + nested) ----
    modalities = set()
    for scope in (cfg, tc):
        if not isinstance(scope, dict):
            continue
        if scope.get("vision_config") or scope.get("image_token_id") is not None \
                or scope.get("image_token_index") is not None or "vision" in str(scope.get("model_type", "")).lower():
            modalities.add("image")
        if scope.get("audio_config") or scope.get("audio_token_id") is not None:
            modalities.add("audio")
        vc = scope.get("vision_config")
        if (isinstance(vc, dict) and (vc.get("temporal_patch_size") or vc.get("video_token_id") is not None)) \
                or scope.get("video_token_id") is not None:
            modalities.add("video")

    first_k = g("first_k_dense_replace") or 0
    if not first_k:
        mlf = g("moe_layer_freq")
        if isinstance(mlf, list) and any(not v for v in mlf):
            first_k = next((i for i, v in enumerate(mlf) if v), len(mlf))

    return {
        "arch": arch, "arch_verified": arch in _VERIFIED_ARCHS,
        "model_type": mt or tcmt,
        "is_moe": is_moe, "n_experts": n_experts, "top_k": top_k, "n_shared": g("n_shared_experts"),
        "is_mla": is_mla, "kv_lora_rank": g("kv_lora_rank"), "qk_rope_head_dim": g("qk_rope_head_dim"),
        "is_gqa": (not is_mla) and (not is_mamba_hybrid),
        "num_kv_heads": g("num_key_value_heads"), "num_attn_heads": g("num_attention_heads"),
        "head_dim": g("head_dim") or ((g("hidden_size") or 0) // (g("num_attention_heads") or 1) or None),
        "is_mamba_hybrid": is_mamba_hybrid, "is_dsa": is_dsa, "has_mtp": has_mtp,
        "is_m3": arch == "MiniMaxM3SparseForConditionalGeneration",
        "modalities": sorted(modalities), "is_vl": bool(modalities),
        "mm_serveable": arch in _VLLM_MM_ARCHS,
        "weights": "int4" if is_int4 else ("fp8" if is_fp8 else "bf16"),
        "quant_method": qmethod or None, "packing": packing,
        "declared_dtype": str(g("torch_dtype") or g("dtype") or "").lower() or None,
        "partial_quant": partial, "first_k_dense": first_k or 0,
        "tie_word_embeddings": bool(g("tie_word_embeddings")),
        "hidden_size": g("hidden_size"), "layers": g("num_hidden_layers"),
        "max_pos": g("max_position_embeddings"),
    }


def detect_tool_calling(model_dir):
    """Template-signature tool-call detection. Returns (parser_or_None, supported, note). supported=True but
    parser=None => tools advertised but no installed parser matches (warn, emit nothing)."""
    tpl = _load_chat_template(model_dir)
    if not tpl:
        return None, False, "no chat template found -> tools not auto-configured"
    low = tpl
    advertises = ("tool_call" in low) or ("<tools>" in low) or ("tool_calls" in low)
    for parser, needles, reason in _TOOL_RULES:
        if all(n in low for n in needles):
            if parser is None:
                return None, True, f"tools advertised but NO installed parser matches ({reason}); serve emits tool text into content -- add a parser then set --tool-call-parser/--enable-auto-tool-choice"
            return parser, True, f"tool-call-parser {parser}: {reason}"
    if advertises:
        return None, True, "tools advertised but the emitted tool-call format matches no installed parser -> emit nothing (LOUD); verify the format and pick a parser"
    return None, False, ""


def detect_reasoning(model_dir, f):
    """Template close-token reasoning detection. Returns (parser_or_None, plugin_name_or_None, note)."""
    tpl = _load_chat_template(model_dir)
    if not tpl:
        return None, None, ""
    added = _added_tokens_blob(model_dir)
    for close_tok, target in _REASON_RULES:
        if close_tok not in tpl:
            continue
        # vocab-guard: the close token should be a real special token (else a stray parser mislabels output)
        if close_tok not in added and close_tok not in ("</think>",):
            continue
        if target.startswith("__plugin__:"):
            name = target.split(":", 1)[1]
            return name, name, (f"reasoning parser {name} needs a shipped plugin ({close_tok}); if the plugin "
                                f"mount is absent, omit --reasoning-parser (reasoning stays inline) -- NEVER pick minimax_m2")
        if target == "__think__":
            a = str(f.get("arch", "")).lower()
            if "qwen3" in a or "qwen3_5" in a:
                return "qwen3", None, "reasoning parser qwen3 (handles enable_thinking + unclosed tool_call as reasoning-end)"
            if "glm" in a:
                return "qwen3", None, ("reasoning parser qwen3 for GLM (<think></think> plus "
                                       "<tool_call> as an implicit reasoning end)")
            if "deepseek" in a:
                return "deepseek_r1", None, "reasoning parser deepseek_r1"
            return "deepseek_r1", None, "reasoning parser deepseek_r1 (generic <think></think> splitter) -- verify family/coherence"
        return target, None, f"reasoning parser {target} ({close_tok})"
    return None, None, ""


def _weight_gib(model_dir):
    """Sum of *.safetensors bytes * 1.08 (embed/lm_head/shared-expert replication fudge) -> GiB, or None."""
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not files:
        return None
    try:
        return sum(os.path.getsize(fp) for fp in files) * 1.08 / (2 ** 30)
    except OSError:
        return None


_CARD_GIB = 96.0
_WEIGHT_CAP_FRAC = 0.75            # reserve >=24 GiB/card for KV + graph + prefill activation


def plan_tp(model_dir, f, pack_mode):
    """Pick tensor-parallel size. VERIFIED arch -> proven champion TP (don't silently downshift on-card-proven
    configs). UNKNOWN arch, or verified+pack_mode -> TP_minfit = smallest power-of-2 that fits weights."""
    gib = _weight_gib(model_dir)
    cap = _WEIGHT_CAP_FRAC * _CARD_GIB
    minfit, warn = 8, None
    if gib is None:
        warn = "could not measure weight bytes (no *.safetensors) -> TP=8 fallback"
        minfit = 8
    else:
        minfit = next((tp for tp in (1, 2, 4, 8) if gib / tp <= cap), 8)
        if gib / 8 > cap:
            warn = f"weights ~{gib:.0f} GiB / 8 = {gib/8:.0f} GiB/card > {cap:.0f} cap -> will OOM as-is; needs deeper quant or CPU-offload"
    champ = _CHAMPION_TP.get(f["arch"])
    if champ and not pack_mode:
        tp = champ
        why = f"verified champion TP={champ} (weights ~{gib:.0f} GiB, min-fit={minfit})" if gib else f"verified champion TP={champ}"
    else:
        tp = minfit
        why = (f"min-fit TP={minfit} (weights ~{gib:.0f} GiB, cap {cap:.0f} GiB/card)" if gib else "TP=8 fallback (unmeasurable)") \
            + (" [pack-mode: unverified fit+coherence, reverify on-card]" if (champ and pack_mode) else "")
    return tp, why, warn


def plan_memory(f, tp, profile):
    """Authoritative owner of --max-model-len / --gpu-memory-utilization / --max-num-seqs / offload. Returns
    (args_dict, env_dict, notes, warns)."""
    args, env, notes, warns = {}, {}, [], []
    max_pos = f.get("max_pos") or 32768
    max_pool_on = profile == "max"
    if f["is_mamba_hybrid"]:
        # Mamba state pool is UTIL-sized (= concurrency), NOT seq-sized; state is not paged (no offload/override).
        args["--max-model-len"] = min(int(max_pos), 131072)
        args["--gpu-memory-utilization"] = 0.12
        if not max_pool_on:
            args["--max-num-seqs"] = 128
        notes.append("Mamba-hybrid: attn-KV ~0.3%; the bf16 state pool is util-sized (= physical execution width), not split among requests. Max profile delegates scheduler width to the engine, but recurrent state remains the real HBM limit until it is dynamically paged.")
        return args, env, notes, warns
    if f["is_m3"]:
        args["--max-model-len"] = min(int(max_pos), 4096)
        args["--gpu-memory-utilization"] = 0.75
        if not max_pool_on:
            args["--max-num-seqs"] = 128
        args["--no-enable-prefix-caching"] = True
        if max_pool_on:
            notes.append("MiniMax-M3 FP8 max pool: retain the coherent 4k/context and util=0.75 champion, omit the profile scheduler ceiling, and let requests draw dynamically from the ~2.03M-token resident pool.")
        else:
            notes.append("MiniMax-M3 FP8 champion: 4k context, 128 active sequences, util=0.75 leaves graph headroom while retaining ~2.03M KV tokens (~496 full-context sequences).")
        return args, env, notes, warns
    # MLA / GQA giants + small models
    capacity_on = profile in ("capacity", "agent-capacity")
    cap_len = int(max_pos) if max_pool_on else (131072 if capacity_on else 32768)
    args["--max-model-len"] = min(int(max_pos), cap_len)
    args["--gpu-memory-utilization"] = 0.95 if max_pool_on else 0.80
    if not max_pool_on:
        args["--max-num-seqs"] = 64    # real-prompt safe (c128 headline is short-prompt/bench-only; prefill-activation HBM)
    # CAPACITY profile on an MLA/GQA giant (TP>=4): add the CPU cold-tier + proven max-ctx override.
    if capacity_on and tp >= 4 and (f["is_mla"] or f["is_gqa"]) and not f["is_m3"]:
        args["--max-model-len"] = min(int(max_pos), 131072)
        args["--enable-prefix-caching"] = True
        args["--gpu-memory-utilization"] = 0.80          # 0.85 + connector staging OOMs
        env["VLLM_GRAPH_RESERVED_MEM"] = "0.30"
        env["HPU_KV_OFFLOAD_STAGE_MB"] = "128"
        env["HPU_KV_OFFLOAD_VERIFY"] = "1"
        if f["is_mla"]:
            env["HPU_FP8_KV_BF16_ATTN"] = "1"
        args["--max-num-batched-tokens"] = 4096          # proven long-prompt chunk / graph bucket ceiling
        # Native-FP8 offload keeps one 6.432-GiB hot pool (the old int8 alias accidentally duplicated it).
        # 1200 blocks = 153,600 hot tokens and leaves ~15 GiB/card idle / >=9 GiB at 99.8% hot saturation.
        args["--num-gpu-blocks-override"] = 1200
        args["--kv-transfer-config"] = ("json:{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\"," 
                                        "\"kv_buffer_device\":\"hpu\",\"kv_buffer_size\":134217728,"
                                        "\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":300000000000,"
                                        "\"eviction_policy\":\"lru\",\"store_threshold\":1}}")
        notes.append("capacity profile: native-FP8 OffloadingConnector, 300GB CPU LRU cold-tier, 1200 hot blocks (153.6k tokens), APC, 4k chunked prefill. store_threshold=1 is required: threshold=2 only admits the first missing block because maximal-prefix lookup stops at that miss.")
    elif max_pool_on:
        args["--no-enable-prefix-caching"] = True
        notes.append("max profile: one auto-sized resident KV block pool; no --num-gpu-blocks-override and no profile-level --max-num-seqs. Requests draw blocks dynamically, while vLLM retains only its engine execution safety width.")
    else:
        args["--no-enable-prefix-caching"] = True
    return args, env, notes, warns


def recommend_config(model_dir, f, profile="agent-tools"):
    """Map arch features + workload profile -> the full serving config (parsers, MM, TP, KV, speed levers)."""
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose one of {sorted(_PROFILES)}")
    env, args, plugins, notes, warns = {}, {}, set(), [], []
    tools_on = profile in ("agent-tools", "agent-capacity", "max")
    capacity_on = profile in ("capacity", "agent-capacity")
    weight_gib = _weight_gib(model_dir)
    # Card-fit Qwen families use external DP8/TP1 (in-engine DP is nonfunctional).
    # Qwen3-30B GPTQ: C1=42.0, C64=786.4, C512=1987.0 wall tok/s (602+256 ruler).
    # Qwen3.6-27B Mamba/FP8: C1=49.1 decode, C64=1515.3 wall, C512=1814.4 wall
    # tok/s (631+256 ruler). All were zero-failure after the nginx 8K-connection fix.
    # Keep larger/unverified architectures on their proven monolithic layout.
    serve_layout = "monolithic"
    if (f["arch"] == "Qwen3MoeForCausalLM" and f["packing"] == "gptq"
            and weight_gib is not None and weight_gib <= _WEIGHT_CAP_FRAC * _CARD_GIB):
        serve_layout = "dp8_tp1"
        notes.append("verified external DP8/TP1 layout: eight card-local replicas + least-conn balancer; full 512-stream scheduler capacity reached without restarts.")
    elif (f["arch"] == "Qwen3_5ForConditionalGeneration"
            and weight_gib is not None and weight_gib <= _WEIGHT_CAP_FRAC * _CARD_GIB):
        serve_layout = "dp8_tp1"
        notes.append("verified external DP8/TP1 layout for dense Mamba/FP8: +33% C1 decode and +60% C64 wall throughput vs TP8; 512/512 requests at full pool capacity.")

    # ---- execution mode ----
    if f["is_m3"]:
        env["PT_HPU_LAZY_MODE"] = "0"
        env["VLLM_SKIP_WARMUP"] = "true"
        notes.append("MiniMax-M3 -> EAGER native fp8_gemm_v2 path; skip the 108-shape startup warmup and compile only real request shapes.")
    elif f["is_mamba_hybrid"]:
        env["PT_HPU_LAZY_MODE"] = "0"                       # eager REQUIRED (lazy GDN/Mamba synStatus-26)
        env["VLLM_SKIP_WARMUP"] = "true"
        notes.append("Mamba/GDN hybrid -> EAGER; Mamba state stays bf16 (quantizing it NaNs).")
    else:
        env["PT_HPU_LAZY_MODE"] = "1"
        env["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
        env["VLLM_SKIP_WARMUP"] = "false"

    # ---- MoE op stack by weight quant + PACKING (autoround->W4A8 monolithic; gptq->gptq_hpu kernel) ----
    if f["is_moe"] and not f["is_m3"]:
        args["--enable-expert-parallel"] = True
        k = f["top_k"] or 8
        env["HPU_MOE_FUSED_OP"] = "1"
        env["HPU_MOE_ROUTED_K"] = str(k)
        if f["weights"] == "int4":
            if f["packing"] == "autoround":
                env["HPU_MOE_FUSED_MONOLITHIC"] = "1"          # W4A8 int4->fp8 in-SRAM dequant (2x MME); auto-switch by batch
                env["HPU_MOE_FUSED_MONO_INT4FP8"] = "1"
                env["HPU_MOE_FUSED_MONO_MAX_T"] = "128"
                env["HPU_MOE_FUSED_MONO_MARKSTEP"] = "0"
                # Hybrid dispatch: T=1 stays on routed BF16 K8 for single-stream
                # latency; T>1 must reach monolithic INT4->FP8 W4A8.  Setting
                # ROUTED_BSGT1=1 routes *all* batched decode through BF16 and
                # makes the monolithic path inert (measured ~28% slower at
                # C56/C64 on GLM-5.2).
                env["HPU_MOE_ROUTED_BSGT1"] = "0"
                if f["arch"] == "GlmMoeDsaForCausalLM":
                    # Same-ruler production A/B: batched-K improves T=1 by
                    # 2.6%; chunk 32 improves C56/C64 by 1.4-2.0%. Both passed
                    # tools plus 100%-hot/evict/208k-token cold replay at 56c.
                    env["HPU_MOE_FUSED_BATCHED"] = "1"
                    env["HPU_MOE_FUSED_MONOLITHIC_CHUNK"] = "32"
                notes.append("int4 AutoRound MoE -> hybrid dispatch: routed BF16 K8 at T=1, monolithic INT4->FP8 W4A8 at T>1 (measured +28% C56/C64 on GLM-5.2). Verified GLM adds batched-K T=1 + monolithic chunk=32 (+2.6% C1, +1.4-2.0% C56/C64). Warmup stays ON with bounded buckets; skipping it caused late-graph HBM-fragmentation crashes.")
            else:
                qmap = {"awq": "awq_hpu", "gptq": "gptq_hpu"}
                args["--quantization"] = qmap.get(f["packing"], "gptq_hpu")
                if serve_layout == "dp8_tp1":
                    # TP1 has all 128 experts local. Admit T>1 decode conservatively with
                    # K=min(E,T*top_k), which cannot truncate the expert union; cap at T=64
                    # so large prefills stay on the established all-expert path.
                    env["HPU_MOE_ROUTED_BSGT1"] = "1"
                    env["HPU_MOE_ROUTED_MAX_T"] = "64"
                    env["HPU_MOE_FUSED_X_BUF_T"] = "64"
                notes.append(f"int4 {f['packing']} MoE -> {args['--quantization']} kernel path (W4A8 monolithic is AutoRound-only).")
        elif f["weights"] == "fp8":
            env["HPU_MOE_FP8_EXPERTS"] = "1"
            notes.append("fp8 MoE -> experts already fp8 (fp8-static path).")
        else:
            warns.append("MoE with bf16 experts -> will OOM at scale; quantize to int4/fp8 first.")

    # ---- KV cache treatment (fp8_inc floor everywhere; lossy int4/TurboQuant only behind capacity+sentinel) ----
    kv_gate = os.path.exists(os.path.join(_CACHE_ROOT, os.path.basename(model_dir.rstrip("/")), ".kv_gate_passed"))
    args["--kv-cache-dtype"] = "fp8_inc"                       # universal knee: MLA-knee / ~lossless GQA / free hybrid; tool-safe
    if f["is_mla"]:
        # Storage and attention compute are independent. fp8_inc is the proven
        # capacity knee, while the native FP8 MLA matmul path is opt-in only:
        # its historical dynamic amax scans the full cache each step and its
        # uncalibrated fixed path corrupts long context. Keep the universally
        # coherent BF16 compute path until an architecture-specific on-card gate
        # explicitly promotes fixed FP8 attention.
        env["HPU_FP8_KV_BF16_ATTN"] = "1"
        notes.append("MLA -> fp8-KV storage + BF16 attention compute (safe default; native FP8 attention remains gated until fixed-scale long-context validation). Capacity: APC + CPU offload (lossless), or int4-store behind a gate.")
        if capacity_on and kv_gate:
            plugins.add("hpu-kvq-ext")
            env["USE_KVQ"] = "1"; env["HPU_KVQ_BACKEND"] = "torch"
            env["HPU_MLA_KV_INT4STORE"] = "1"; env["HPU_MLA_KV_SHRINK"] = "1"
            notes.append("capacity + .kv_gate_passed -> MLA int4-latent STORE (1.31-1.385x tok, ~-20% SS). Assert rope_head_dim==64.")
    elif f["is_gqa"] and not f["is_mamba_hybrid"]:
        notes.append("GQA -> fp8-KV (~lossless). Capacity: TurboQuant int4-KV (k8v4 + fp8-V) behind an on-card coherence gate.")
        if capacity_on and kv_gate:
            plugins.add("hpu-kvq-ext")
            env["USE_KVQ"] = "1"; env["HPU_KVQ_READY"] = "1"; env["HPU_KVQ_BACKEND"] = "torch"; env["HPU_KVQ_VFP8"] = "1"
            args["--kv-cache-dtype"] = "turboquant_k8v4"
            notes.append("capacity + .kv_gate_passed -> TurboQuant k8v4 int4-K + fp8-V (~2x pool). NEVER k8v2/2-bit-V (37% err).")

    # ---- DSA sparse indexer ----
    if f["is_dsa"] and not f["is_m3"]:
        env["HPU_DSA_READY"] = "1"
        plugins.add("hpu-dsa-ext")
        notes.append("DSA sparse attention -> indexer plugin (HPU_DSA_READY=1).")
    elif f["is_m3"] and f["is_dsa"]:
        notes.append("MiniMax-M3 sparse-attention metadata detected, but the installed DSA plugin is GLM-specific; keep the proven dense-GQA fallback until an M3 indexer passes its on-card gate.")

    # ---- depth-skip: COHERENCE-RISKY + TOOL-UNSAFE. HARD-excluded for tool users; opt-in only for chat-throughput ----
    if f["is_dsa"] and f["is_moe"] and (f["layers"] or 0) >= 40:
        if profile == "chat-throughput" and not tools_on:
            notes.append("chat-throughput profile: depth-skip AVAILABLE (HPU_SKIP_LAYERS K=8, shared-indexer only). LOSSY -> gate on perplexity. NOT emitted by default even here (set HPU_SKIP_LAYERS explicitly).")
        else:
            notes.append("depth-skip (HPU_SKIP_LAYERS) is OFF -> it silently BREAKS tool-call formatting; kept off for this tool/agent serve (see glm52-depthskip-breaks-tools).")

    # ---- TOOLS + REASONING (template-signature detected; agent-tools profile) ----
    if tools_on:
        tparser, tsupp, tnote = detect_tool_calling(model_dir)
        if tparser:
            args["--enable-auto-tool-choice"] = True
            args["--tool-call-parser"] = tparser
            notes.append(tnote + " -- route clients to /v1/chat/completions (raw /v1/completions does not parse tools).")
        elif tsupp:
            warns.append("TOOLS: " + tnote)
        rparser, rplugin, rnote = detect_reasoning(model_dir, f)
        if rparser and not rplugin:
            args["--reasoning-parser"] = rparser
            notes.append(rnote)
        elif rplugin:
            notes.append("REASONING: " + rnote + " (plugin mount not auto-wired yet -> reasoning inline unless the plugin is present).")

    # ---- MULTIMODAL (registry-gated) ----
    if f["modalities"]:
        if f["mm_serveable"]:
            lim = {"image": 2, "video": 0, "audio": (1 if "audio" in f["modalities"] else 0)}
            args["--limit-mm-per-prompt"] = "json:" + json.dumps({m: lim.get(m, 0) for m in ("image", "video", "audio") if m in f["modalities"]})
            args["--max-model-len"] = max(int(args.get("--max-model-len", 8192) or 8192), 8192)  # one image ~ thousands of tokens
            args["--skip-mm-profiling"] = True
            notes.append(f"MULTIMODAL {f['modalities']} -> served (arch in vllm-gaudi MM registry). Drive a REAL image request to confirm the HPU vision path; video=0 (memory-walled, opt-in).")
        else:
            warns.append(f"MULTIMODAL config detected ({f['modalities']}) but arch '{f['arch']}' is NOT in the vllm-gaudi MM registry -> vision/audio tower NOT served; serves TEXT-ONLY, media inputs rejected.")

    # ---- weights quantization flag (fp8 family needs the exact method; int4 auto-detected) ----
    if f["weights"] == "fp8" and "--quantization" not in args:
        args["--quantization"] = "compressed-tensors" if f.get("quant_method") == "compressed-tensors" else "fp8"
    if f["weights"] != "bf16" or f.get("declared_dtype") in ("float16", "fp16", "half"):
        args["--dtype"] = "bfloat16"                           # HPU quant kernels require bf16 compute; no fast fp16

    # ---- M3 custom-class markers (won't boot without them; drop --quantization/--dtype, its plugin owns quant) ----
    if f["is_m3"]:
        env["M3_ROOT_DIR"] = "/opt/m3-root"; env["M3_VLLM_DIR"] = "/opt/m3-vllm"
        env["M3_SWIGLUOAI"] = "1"; env["M3_FP8_SWIGLU_ROUTE"] = "b"; env["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
        args["--enable-expert-parallel"] = True
        if f["weights"] == "fp8":
            env["M3_FP8_STATIC"] = "1"
            env["M3_FP8_MME"] = "1"
        else:
            env["M3_EXPERT_RUNTIME"] = "fp8"
        plugins.add("hpu-m3-ext")
        args.pop("--quantization", None)
        args["--quantization"] = "compressed-tensors"
        notes.append("MiniMax-M3 custom class -> static-FP8 route-b with corrected native fp8_gemm_v2 MME (C1 24.2 decode; C64 542 wall; C128 655 wall; C256 673 wall tok/s, zero failures).")

    # ---- spec-decode / MTP (surfaced, gated -- never auto-stacked into the fp8-KV agent champion) ----
    if f["has_mtp"]:
        notes.append("native MTP present -> OPT-IN separate container (--speculative-config method=deepseek_mtp); regresses with fp8-KV+depth-skip, so NOT auto-stacked. See glm52-mtp-conc.env.")
    notes.append("union tree-drafter (HPU_TREE_DRAFTER=union, 0-forward, byte-exact) is a REPETITIVE/structured-workload opt-in (regresses freeform SS ~2.2x) -> not auto-on.")

    # ---- arch confidence + config-landmine guards (LOUD) ----
    if not f["arch_verified"]:
        warns.append("UNVERIFIED ARCH '%s' -- best-effort config; VALIDATE coherence + throughput + tool-format before trusting." % f["arch"])
    if f["is_moe"] and f["top_k"] is None:
        warns.append("MoE but num_experts_per_tok ABSENT -> routed-K defaulted to 8; verify real top-k.")
    if f["partial_quant"]:
        notes.append("Partial quantization: %d module(s) stay high-precision (lm_head/embeds/vision/norms) -> factor into memory; don't quantize/graft them." % f["partial_quant"])
    if f["n_shared"]:
        notes.append("n_shared_experts=%s -> always-active shared expert(s) (EP-replicated)." % f["n_shared"])
    if f["first_k_dense"]:
        notes.append("first %s layer(s) DENSE -> affects EP banding." % f["first_k_dense"])
    if f["tie_word_embeddings"]:
        notes.append("tie_word_embeddings=true -> lm_head shares embeds; don't quantize/graft lm_head separately.")

    # ---- TP + memory/context ----
    tp, tp_why, tp_warn = plan_tp(model_dir, f, pack_mode=(profile == "pack"))
    args["--tensor-parallel-size"] = tp
    notes.append("TP: " + tp_why)
    if tp_warn:
        warns.append("TP: " + tp_warn)
    margs, menv, mnotes, mwarns = plan_memory(f, tp, profile)
    args.update(margs); env.update(menv); notes.extend(mnotes); warns.extend(mwarns)

    # ---- common serving args + bucketing (span the emitted batch/ctx or graphs JIT-miss mid-serve) ----
    args["--block-size"] = 128                                 # MME tile alignment
    args["--async-scheduling"] = True
    args["--trust-remote-code"] = True
    # When max profile omits the CLI ceiling, size graph buckets to the engine's
    # actual default physical execution width. This does not partition the KV
    # pool; it only prevents a runtime graph miss when the scheduler admits a
    # larger batch.
    ns = int(args.get("--max-num-seqs", _ENGINE_DEFAULT_MAX_NUM_SEQS) or _ENGINE_DEFAULT_MAX_NUM_SEQS)
    ml = int(args.get("--max-model-len", 32768) or 32768)
    ovr = args.get("--num-gpu-blocks-override")
    nblocks = int(ovr) if ovr else -(-(ml * ns) // 128)
    env["VLLM_BUCKETING_STRATEGY"] = "lin"
    env["VLLM_DECODE_BS_BUCKET_MIN"] = "1"; env["VLLM_DECODE_BS_BUCKET_STEP"] = str(min(32, ns)); env["VLLM_DECODE_BS_BUCKET_MAX"] = str(ns)
    # Cap the block-bucket ceiling to the real/HBM-bound scale and cap the linear grid to about 16
    # steady buckets. A fixed step=128 at the 16,384-block roof creates 128 block points multiplied
    # by every decode-BS point, turning warmup into another multi-hour Cartesian product. The ramp
    # still preserves dense low-block points (128/256/512...) where padding sensitivity is highest.
    decode_block_max = max(128, min(nblocks, 16384))
    decode_block_step = max(128, -(-decode_block_max // (16 * 128)) * 128)
    env["VLLM_DECODE_BLOCK_BUCKET_MIN"] = "128"; env["VLLM_DECODE_BLOCK_BUCKET_STEP"] = str(decode_block_step); env["VLLM_DECODE_BLOCK_BUCKET_MAX"] = str(decode_block_max)
    # Linear bucketing defaults prompt-CTX step=1 block. At 32K context that creates 4,081 prompt
    # buckets (and a projected multi-hour warmup); larger contexts are worse. Bound every dimension,
    # keeping about eight query points x eight context points plus the zero-context case. The CTX max
    # is one above an aligned roof so Gaudi's long-context edge-bucket logic may include the true final
    # per-query context (`is_ctx_allowed` uses a strict `< VLLM_PROMPT_CTX_BUCKET_MAX` check).
    prompt_query_max = max(128, min(ml, int(args.get("--max-num-batched-tokens", 2048) or 2048)))
    prompt_query_step = 128
    for candidate in (128, 256, 512, 1024, 2048, 4096, 8192):
        if prompt_query_max % candidate == 0 and prompt_query_max // candidate <= 8:
            prompt_query_step = candidate
            break
    prompt_ctx_max = max(0, -(-(ml - 128) // 128))
    prompt_ctx_step = max(1, -(-prompt_ctx_max // 8))
    prompt_ctx_roof = -(-prompt_ctx_max // prompt_ctx_step) * prompt_ctx_step
    env["VLLM_PROMPT_BS_BUCKET_MIN"] = "1"; env["VLLM_PROMPT_BS_BUCKET_STEP"] = "1"; env["VLLM_PROMPT_BS_BUCKET_MAX"] = "1"
    env["VLLM_PROMPT_QUERY_BUCKET_MIN"] = "128"; env["VLLM_PROMPT_QUERY_BUCKET_STEP"] = str(prompt_query_step); env["VLLM_PROMPT_QUERY_BUCKET_MAX"] = str(prompt_query_max)
    env["VLLM_PROMPT_CTX_BUCKET_MIN"] = "0"; env["VLLM_PROMPT_CTX_BUCKET_STEP"] = str(prompt_ctx_step); env["VLLM_PROMPT_CTX_BUCKET_MAX"] = str(prompt_ctx_roof + 1)
    notes.append("linear warmup grids bounded (prompt ~O(8x8), decode blocks ~O(16)); prevents Cartesian bucket explosions while retaining dense low-shape ramps.")
    env.setdefault("VLLM_GRAPH_RESERVED_MEM", "0.10")

    return {"features": f, "profile": profile, "plugins": sorted(plugins), "env": env, "vllm_args": args,
            "serve_layout": serve_layout,
            "capacity_policy": {"hot_pool": "auto" if profile == "max" else "profile",
                                "allocation": "shared", "scheduler_width": "engine" if profile == "max" else "profile"},
            "notes": notes, "warnings": warns}


# standard deployment boilerplate every serve needs
_BOILERPLATE = ("-e PT_HPU_WEIGHT_SHARING=0 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 "
                "-e VLLM_RPC_TIMEOUT=100000000 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 "
                "-e VLLM_HPU_LOG_STEP_CPU_FALLBACKS=true -e VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=true")

_CACHE_ROOT = "/mnt/data2/autodeploy"
_FASTMODELS = "/mnt/fastmodels"


def _graft_compatible(graft_dir, f):
    p = os.path.join(graft_dir, "config.json")
    if not os.path.exists(p):
        return True, "shape-unvalidated: graft has no config.json -- verify it matches this checkpoint"
    try:
        gc = _asdict(json.load(open(p)))
    except Exception:  # noqa: BLE001
        return True, "shape-unvalidated: graft config.json unreadable"
    ghs, glayers = gc.get("hidden_size"), gc.get("num_hidden_layers")
    if ghs is not None and f.get("hidden_size") is not None and ghs != f["hidden_size"]:
        return False, f"hidden_size mismatch graft={ghs} vs model={f['hidden_size']}"
    if glayers is not None and f.get("layers") is not None and glayers != f["layers"]:
        return False, f"layer-count mismatch graft={glayers} vs model={f['layers']}"
    return True, None


def resolve_artifacts(model_dir, f):
    """Durable deployment artifacts: int4-MoE W4A8 repack cache (self-mints) + optional fp8 non-expert graft.
    Non-destructive (caches beside the model; original checkpoint untouched)."""
    name = os.path.basename(model_dir.rstrip("/"))
    art = {"repack_mode": None, "repack_dir": None, "graft": None, "extra_mounts": [], "notes": []}
    # repack only helps the AutoRound W4A8 monolithic path (gptq uses its own kernel)
    if f["is_moe"] and f["weights"] == "int4" and f["packing"] == "autoround":
        rp = os.path.join(_CACHE_ROOT, name, "repack")
        complete = os.path.isdir(rp) and os.path.exists(os.path.join(rp, ".mint_complete"))
        art["repack_dir"] = rp
        art["repack_mode"] = "load" if complete else "mint"
        art["extra_mounts"].append(f"-v {rp}:/repack_rw")
        art["notes"].append(f"int4-MoE repack cache: {'COMPLETE -> load' if complete else 'mint once (first-run) then load'} ({rp})")
        cands = [os.path.join(_FASTMODELS, name + "-fp8-other")]
        alias = _GRAFT_ALIASES.get(f["arch"])
        if alias and any(t in name.lower() for t in alias[0]):
            cands.append(os.path.join(_FASTMODELS, alias[1]))
        for cand in cands:
            if not os.path.isdir(cand):
                continue
            ok, why = _graft_compatible(cand, f)
            if not ok:
                art["notes"].append(f"fp8 non-expert graft: {os.path.basename(cand)} SKIPPED ({why}).")
                continue
            art["graft"] = cand
            art["extra_mounts"].append(f"-v {cand}:/models/{os.path.basename(cand)}")
            art["notes"].append(f"fp8 non-expert graft: using ({cand})" + (f" [{why}]" if why else ""))
            break
        if art["graft"] is None:
            art["notes"].append("fp8 non-expert graft: NONE -> bf16 non-experts (works; optional speedup).")
    return art


# plugin marker -> the env token containers.sh auto-detects (USE_DSA<-HPU_DSA_READY, USE_M3<-M3_*, USE_KVQ)
_PLUGIN_MARKER = {"hpu-dsa-ext": "HPU_DSA_READY=1", "hpu-m3-ext": "USE_M3=1", "hpu-kvq-ext": "USE_KVQ=1"}


def _fmt_arg(k, v):
    """Format a vllm arg. 'json:<payload>' -> source-safe single-quoted, backslash-escaped inner quotes so
    it survives `.env` sourcing intact."""
    if v is True:
        return k
    if isinstance(v, str) and v.startswith("json:"):
        payload = v[len("json:"):].replace('"', '\\"')
        return f"{k} '{payload}'"
    return f"{k} {v}"


def _write_prefix_cache_buckets(model_dir, rec):
    """Create the bounded HPU graph grid required by APC/offload at high concurrency.

    Stock linear/exp strategies cap decode block buckets at the number of distinct
    physical HPU blocks. With shared prefixes, however, the flattened logical block
    table can be much larger (4,439 entries was observed with only 1,200 physical
    blocks), which otherwise triggers unsafe JIT graph compilation under load.
    """
    args = rec["vllm_args"]
    env = rec["env"]
    if not args.get("--enable-prefix-caching") or env.get("PT_HPU_LAZY_MODE") != "1":
        return None, None

    block_size = int(args.get("--block-size", 128) or 128)
    max_len = int(args.get("--max-model-len", 32768) or 32768)
    max_seqs = int(args.get("--max-num-seqs", _ENGINE_DEFAULT_MAX_NUM_SEQS) or _ENGINE_DEFAULT_MAX_NUM_SEQS)
    max_batched = int(args.get("--max-num-batched-tokens", 4096) or 4096)
    physical_blocks = int(args.get("--num-gpu-blocks-override") or max_len // block_size)
    max_len_blocks = max(1, (max_len + block_size - 1) // block_size)

    prompt_queries = sorted({
        q for q in (block_size, 512, 1024, 2048, 3072, 4096, max_batched)
        if block_size <= q <= max_batched
    })
    prompt_lines = []
    for query in prompt_queries:
        query_blocks = (query + block_size - 1) // block_size
        max_ctx = max(0, max_len_blocks - query_blocks)
        ctx = sorted({v for v in (0, 32, 128, 512, max_ctx) if 0 <= v <= max_ctx})
        prompt_lines.append(f"(1, {query}, {ctx!r})")

    regular_batches = []
    b = 1
    while b < max_seqs:
        regular_batches.append(b)
        b *= 2
    regular_ctx = sorted({v for v in (128, 512, physical_blocks) if v <= physical_blocks})
    # Bound warmup while covering 56/64-way shared-prefix block tables. At most
    # 128 blocks (16k tokens) per sequence are represented in this aggregate
    # bucket; above 64 scheduler slots the global safety roof remains 16,384.
    aggregate_blocks = min(16384, max(physical_blocks, max_seqs * min(max_len_blocks, 128)))
    decode_lines = []
    if regular_batches:
        decode_lines.append(f"({regular_batches!r}, 1, {regular_ctx!r})")
    full_ctx = sorted(set(regular_ctx + [aggregate_blocks]))
    decode_lines.append(f"({max_seqs}, 1, {full_ctx!r})")

    name = _sanitize_name(os.path.basename(model_dir.rstrip("/")))
    host_dir = os.path.join(_CACHE_ROOT, name)
    host_path = os.path.join(host_dir, "autoconfig.buckets")
    os.makedirs(host_dir, exist_ok=True)
    contents = "\n".join([
        "# AUTO-GENERATED bounded HPU graph buckets for APC/offload.",
        f"# max_len={max_len} max_seqs={max_seqs} physical_blocks={physical_blocks} aggregate_blocks={aggregate_blocks}",
        *prompt_lines,
        *decode_lines,
        "",
    ])
    tmp_path = host_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(contents)
    os.replace(tmp_path, host_path)
    return host_path, "/opt/trident/autoconfig.buckets"


def _write_dp_replica_buckets(model_dir, rec):
    """Create the small exact graph set for a card-local Qwen GPTQ replica pool."""
    if rec.get("serve_layout") != "dp8_tp1" or rec["env"].get("PT_HPU_LAZY_MODE") != "1":
        return None, None
    args = rec["vllm_args"]
    block_size = int(args.get("--block-size", 128) or 128)
    max_len = int(args.get("--max-model-len", 32768) or 32768)
    max_seqs = int(args.get("--max-num-seqs", 64) or 64)
    max_batched = int(args.get("--max-num-batched-tokens", 2048) or 2048)
    max_len_blocks = max(1, (max_len + block_size - 1) // block_size)

    prompt_queries = sorted({
        q for q in (block_size, 256, 512, 768, 1024, 2048)
        if block_size <= q <= max_batched
    })
    prompt_lines = []
    for query in prompt_queries:
        query_blocks = (query + block_size - 1) // block_size
        max_ctx = max(0, max_len_blocks - query_blocks)
        ctx = sorted({v for v in (0, 32, 128, max_ctx) if 0 <= v <= max_ctx})
        prompt_lines.append(f"(1, {query}, {ctx!r})")

    batches = []
    b = 1
    while b <= max_seqs:
        batches.append(b)
        b *= 2
    aggregate_blocks = min(8192, max_seqs * max_len_blocks)
    decode_ctx = sorted({v for v in (128, 512, 2048, 4096, aggregate_blocks) if v <= aggregate_blocks})
    decode_lines = [f"({batches!r}, 1, {decode_ctx!r})"]

    name = _sanitize_name(os.path.basename(model_dir.rstrip("/")))
    host_dir = os.path.join(_CACHE_ROOT, name)
    host_path = os.path.join(host_dir, "dp8_tp1.buckets")
    os.makedirs(host_dir, exist_ok=True)
    contents = "\n".join([
        "# AUTO-GENERATED bounded HPU graph buckets for external DP8/TP1.",
        f"# max_len={max_len} max_seqs={max_seqs} aggregate_blocks={aggregate_blocks}",
        *prompt_lines,
        *decode_lines,
        "",
    ])
    tmp_path = host_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(contents)
    os.replace(tmp_path, host_path)
    return host_path, "/opt/trident/dp8_tp1.buckets"


def emit_env(model_dir, served_name=None, profile="agent-tools"):
    """Emit a SERVABLE containers.sh env with artifacts resolved + plugin markers translated."""
    f = detect_arch(_load(model_dir))
    rec = recommend_config(model_dir, f, profile)
    art = resolve_artifacts(model_dir, f)
    name = os.path.basename(model_dir.rstrip("/"))
    served = _sanitize_name(served_name or name)
    env = dict(rec["env"])
    if art["repack_mode"] == "load":
        env["HPU_MOE_REPACK_LOAD"] = "/repack_rw"
    if art["graft"]:
        env["HPU_FP8_OTHER_GRAFT"] = "/models/" + os.path.basename(art["graft"])
    # translate plugin selections into the marker env containers.sh keys on
    for pl in rec["plugins"]:
        mk = _PLUGIN_MARKER.get(pl)
        if mk:
            k, val = mk.split("=", 1)
            env.setdefault(k, val)
    if f["is_m3"]:
        here = os.path.dirname(os.path.abspath(__file__))
        required_m3_mounts = [
            (os.path.join(here, "minimax-m3/vllm"), "/opt/m3-vllm"),
            (os.path.join(here, "minimax-m3"), "/opt/m3-root"),
            (os.path.join(here, "vllm-gaudi-patch/worker/hpu_model_runner.py"),
             "/opt/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py"),
        ]
        missing = [src for src, _ in required_m3_mounts if not os.path.exists(src)]
        if missing:
            raise RuntimeError("MiniMax-M3 profile requires missing runtime path(s): " + ", ".join(missing))
        for src, dst in required_m3_mounts:
            art["extra_mounts"].append(f"-v {src}:{dst}:ro")
        rec["notes"].append("auto-wired MiniMax-M3 model/runtime sources + matched HPU runner")
    dp_bucket_host, dp_bucket_container = _write_dp_replica_buckets(model_dir, rec)
    if dp_bucket_host:
        # A file bucket set replaces the Cartesian linear ranges. Drop the range
        # variables from the emitted env so logs and fingerprints describe one owner.
        for key in list(env):
            if key == "VLLM_BUCKETING_STRATEGY" or key.startswith("VLLM_PROMPT_") or key.startswith("VLLM_DECODE_"):
                env.pop(key)
        env["VLLM_BUCKETING_FROM_FILE"] = dp_bucket_container
        art["extra_mounts"].append(f"-v {dp_bucket_host}:{dp_bucket_container}:ro")
        rec["notes"].append("auto-generated 48-graph DP replica bucket artifact (24 prompt + 24 decode after filtering)")
    bucket_host, bucket_container = _write_prefix_cache_buckets(model_dir, rec)
    if bucket_host:
        env["VLLM_BUCKETING_FROM_FILE"] = bucket_container
        art["extra_mounts"].append(f"-v {bucket_host}:{bucket_container}:ro")
        rec["notes"].append("bounded APC/offload HPU bucket artifact generated (includes high-concurrency aggregate block-table bucket; prevents runtime JIT graph misses)")

    # fp8_inc MLA storage must remain decoupled from attention compute unless a
    # calibrated fixed-FP8 gate is explicitly selected. The stock image lacks
    # that decoupling in HPUMLAImpl, so every emitted MLA env must mount the
    # patched attention/cache modules. GLM also needs the matched runner fixes
    # used by the proven production profile (KV-spec/indexer handling).
    if f["is_mla"]:
        here = os.path.dirname(os.path.abspath(__file__))
        required_mla_mounts = [
            (os.path.join(here, "vllm-gaudi-patch/extension/utils.py"),
             "/opt/vllm-gaudi/vllm_gaudi/extension/utils.py"),
            (os.path.join(here, "vllm-gaudi-patch/attention/backends/hpu_attn.py"),
             "/opt/vllm-gaudi/vllm_gaudi/attention/backends/hpu_attn.py"),
        ]
        if f["arch"] == "GlmMoeDsaForCausalLM":
            required_mla_mounts.append(
                (os.path.join(here, "vllm-gaudi-patch/v1/worker/hpu_model_runner.py"),
                 "/opt/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py")
            )
        missing = [src for src, _ in required_mla_mounts if not os.path.isfile(src)]
        if missing:
            raise RuntimeError("MLA profile requires missing HPU safety patch file(s): " + ", ".join(missing))
        for src, dst in required_mla_mounts:
            mount = f"-v {src}:{dst}:ro"
            if mount not in art["extra_mounts"]:
                art["extra_mounts"].append(mount)
        rec["notes"].append("auto-wired fp8-KV/BF16-attention decoupling patches" +
                            (" + matched GLM KV runner" if f["arch"] == "GlmMoeDsaForCausalLM" else ""))

    # The stock image does not contain the native-FP8 HPU offloader or the
    # store+load scheduler crash workaround. A capacity env that omits either
    # silently loses cold-KV correctness, so auto-wire them when offload is selected.
    if "--kv-transfer-config" in rec["vllm_args"]:
        here = os.path.dirname(os.path.abspath(__file__))
        required_mounts = [
            (os.path.join(here, "_src/vllm_gaudi/v1/kv_offload/worker/cpu_hpu.py"),
             "/opt/vllm-gaudi/vllm_gaudi/v1/kv_offload/worker/cpu_hpu.py"),
            (os.path.join(here, "vllm-gaudi-patch/vllm_offload/scheduler.py"),
             "/opt/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"),
        ]
        missing = [src for src, _ in required_mounts if not os.path.isfile(src)]
        if missing:
            raise RuntimeError("capacity/offload profile requires missing HPU patch file(s): " + ", ".join(missing))
        for src, dst in required_mounts:
            art["extra_mounts"].append(f"-v {src}:{dst}:ro")
        rec["notes"].append("auto-wired native-FP8 HPU CPU-offload worker + store/load scheduler safety patch")
    menv = " ".join("-e %s=%s" % (k, v) for k, v in env.items()) + " " + _BOILERPLATE
    vargs = " ".join(_fmt_arg(k, v) for k, v in rec["vllm_args"].items())
    mounts = " ".join(art["extra_mounts"])
    warn = ("\n# WARNINGS: " + " | ".join(rec["warnings"])) if rec["warnings"] else ""
    notes = "".join("\n# " + n for n in (rec["notes"] + art["notes"]))
    det = ('MoE(E=%s,k=%s)' % (f['n_experts'], f['top_k'])) if f['is_moe'] else 'dense'
    attn = 'MLA' if f['is_mla'] else ('Mamba-hybrid' if f['is_mamba_hybrid'] else 'GQA')
    mm = (' VL[%s]%s' % (",".join(f['modalities']), '' if f['mm_serveable'] else '(text-only!)')) if f['modalities'] else ''
    return f"""# AUTO-GENERATED by autoconfig.py -- plug-and-play DEPLOY env for {name}. profile={profile}
# Detected: {det} | {attn} | weights={f['weights']}({f['packing']}) | {'DSA ' if f['is_dsa'] else ''}{f['layers']}L{mm} | TP={rec['vllm_args'].get('--tensor-parallel-size')}{warn}{notes}
# ENGINE_REPACK={art['repack_mode'] or 'none'}
# ENGINE_REPACK_DIR={art['repack_dir'] or ''}
IMAGE=vllm-gaudi:next
MODEL_PATH={name}
SERVED_NAME={served}
SERVE_LAYOUT={rec.get('serve_layout', 'monolithic')}
EXTRA_MOUNTS="{mounts}"
MODEL_ENV="{menv}"
VLLM_ARGS="{vargs}"
"""


def emit(model_dir, profile="agent-tools"):
    return recommend_config(model_dir, detect_arch(_load(model_dir)), profile)


def _print(rec, title=""):
    f = rec["features"]
    print(f"\n=== {title or f['arch']} ===")
    print("  detected: %s | %s | weights=%s(%s) | %s%s%sTP=%s | profile=%s" % (
        "MoE(E=%s,k=%s)" % (f["n_experts"], f["top_k"]) if f["is_moe"] else "dense",
        "MLA" if f["is_mla"] else ("Mamba-hybrid" if f["is_mamba_hybrid"] else "GQA"),
        f["weights"], f["packing"], "DSA " if f["is_dsa"] else "",
        ("VL[%s]%s " % (",".join(f["modalities"]), "" if f["mm_serveable"] else "(text-only!)")) if f["modalities"] else "",
        f"{f['layers']}L ", rec["vllm_args"].get("--tensor-parallel-size"), rec["profile"]))
    print("  plugins :", rec["plugins"] or "(none)")
    print("  env     :", " ".join("%s=%s" % (k, v) for k, v in rec["env"].items()))
    print("  args    :", " ".join(_fmt_arg(k, v) for k, v in rec["vllm_args"].items()))
    for n in rec["notes"]:
        print("   note:", n)
    for w in rec["warnings"]:
        print("   WARN:", w)


if __name__ == "__main__":
    argv = sys.argv[1:]
    profile = "agent-tools"
    if "--profile" in argv:
        i = argv.index("--profile")
        profile = argv[i + 1] if i + 1 < len(argv) else profile
        del argv[i:i + 2]
    pos = [a for a in argv if not a.startswith("--")]
    if "--emit-env" in argv:
        if not pos:
            sys.stderr.write("usage: autoconfig.py <model_dir> --emit-env [--profile ...]\n"); sys.exit(2)
        try:
            sys.stdout.write(emit_env(pos[0], profile=profile))
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"ERROR: {e}\n"); sys.exit(2)
    elif pos and "--validate" not in argv:
        try:
            _print(emit(pos[0], profile))
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"ERROR: {e}\n"); sys.exit(2)
    else:
        for name in ["GLM-5.2-int4-experts-AutoRound", "MiniMax-M3-fp8-experts-AutoRound",
                     "Qwen3.6-27B-FP8", "Qwen3-30B-A3B-GPTQ-Int4"]:
            d = os.path.join(_FASTMODELS, name)
            try:
                _print(emit(d, profile), name)
            except Exception as e:  # noqa: BLE001
                print(f"\n=== {name} ===\n  ERROR: {e!r}")
