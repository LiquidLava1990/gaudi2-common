#!/usr/bin/env python3
# Regression guard for the plug-and-play auto-config: asserts detect_arch + recommend_config produce the
# right stack for the 4 known model families, so a future edit can't silently break auto-detection.
#   python3 autoconfig_test.py
import json
import os
import re
import shutil
import sys
import tempfile

from autoconfig import (  # noqa
    detect_arch, emit, emit_env, resolve_artifacts, _sanitize_name,
)

BASE = "/mnt/fastmodels"
FAILS = []
_TMP = tempfile.mkdtemp(prefix="autoconfig_test_")


def mk(cfg_dict, name="tmpmodel"):
    """Write a synthetic config.json into a fresh dir (basename=name) and return the dir."""
    d = os.path.join(_TMP, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg_dict, fh)
    return d


def mk_raw(text, name="rawmodel"):
    """Write raw (possibly-invalid) config.json text into a fresh dir and return the dir."""
    d = os.path.join(_TMP, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as fh:
        fh.write(text)
    return d


def ck(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {extra}" if extra else ""))
    if not cond:
        FAILS.append(name)


def feats(model):
    return emit(os.path.join(BASE, model))["features"]


def cfg(model):
    return emit(os.path.join(BASE, model))


# ---- GLM-5.2: MoE-MLA-int4-DSA -> W4A8 + fp8-KV + EP + DSA + LAZY -----------------------------------
g = cfg("GLM-5.2-int4-experts-AutoRound"); gf = g["features"]
ck("GLM: MoE+MLA+int4+DSA", gf["is_moe"] and gf["is_mla"] and gf["weights"] == "int4" and gf["is_dsa"],
   f"{gf['is_moe']}/{gf['is_mla']}/{gf['weights']}/{gf['is_dsa']}")
ck("GLM: top_k=8 layers=78", gf["top_k"] == 8 and gf["layers"] == 78)
ck("GLM: emits W4A8 + fp8-KV + EP + DSA + lazy",
   g["env"].get("HPU_MOE_FUSED_MONO_INT4FP8") == "1" and g["vllm_args"].get("--kv-cache-dtype") == "fp8_inc"
   and g["vllm_args"].get("--enable-expert-parallel") and g["env"].get("HPU_DSA_READY") == "1"
   and g["env"].get("PT_HPU_LAZY_MODE") == "1")
ck("GLM: hybrid MoE dispatch keeps T=1 routed and T>1 on W4A8",
   g["env"].get("HPU_MOE_ROUTED_K") == "8"
   and g["env"].get("HPU_MOE_ROUTED_BSGT1") == "0"
   and g["env"].get("HPU_MOE_FUSED_MONOLITHIC") == "1"
   and g["env"].get("HPU_MOE_FUSED_MONO_INT4FP8") == "1")
ck("GLM: verified T1 batched-K + C56/C64 monolithic chunk32 wins",
   g["env"].get("HPU_MOE_FUSED_BATCHED") == "1"
   and g["env"].get("HPU_MOE_FUSED_MONOLITHIC_CHUNK") == "32")
ck("GLM agent-tools: qwen3 reasoning boundary preserves unclosed tool calls",
   g["vllm_args"].get("--reasoning-parser") == "qwen3")
ck("GLM: production HPU graph warmup stays ON",
   g["env"].get("VLLM_SKIP_WARMUP") == "false")
ck("GLM: fp8 KV storage defaults to coherent BF16 attention compute",
   g["env"].get("HPU_FP8_KV_BF16_ATTN") == "1")

gc = emit(os.path.join(BASE, "GLM-5.2-int4-experts-AutoRound"), profile="agent-capacity")
gkv = gc["vllm_args"].get("--kv-transfer-config", "")
ck("GLM agent-capacity: tools + APC/offload compose",
   gc["vllm_args"].get("--tool-call-parser") == "glm47"
   and gc["vllm_args"].get("--reasoning-parser") == "qwen3"
   and gc["vllm_args"].get("--enable-prefix-caching"))
ck("GLM agent-capacity: proven 153.6k hot / 300GB cold / admission=1",
   gc["vllm_args"].get("--num-gpu-blocks-override") == 1200
   and '"cpu_bytes_to_use":300000000000' in gkv
   and '"store_threshold":1' in gkv
   and gc["env"].get("VLLM_GRAPH_RESERVED_MEM") == "0.30")
gce = emit_env(os.path.join(BASE, "GLM-5.2-int4-experts-AutoRound"), profile="agent-capacity")
ck("GLM agent-capacity env: bounded bucket + native-FP8 offload patches mounted",
   "VLLM_BUCKETING_FROM_FILE=/opt/trident/autoconfig.buckets" in gce
   and "cpu_hpu.py:/opt/vllm-gaudi/vllm_gaudi/v1/kv_offload/worker/cpu_hpu.py:ro" in gce
   and "vllm_offload/scheduler.py:/opt/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro" in gce)
ck("GLM agent-capacity env: MLA attention/cache + matched runner patches mounted",
   "vllm-gaudi-patch/extension/utils.py:/opt/vllm-gaudi/vllm_gaudi/extension/utils.py:ro" in gce
   and "vllm-gaudi-patch/attention/backends/hpu_attn.py:/opt/vllm-gaudi/vllm_gaudi/attention/backends/hpu_attn.py:ro" in gce
   and "vllm-gaudi-patch/v1/worker/hpu_model_runner.py:/opt/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py:ro" in gce)

# ---- M3: MoE-GQA-fp8-DSA-VL -> corrected native FP8 MME + fp8-KV + EAGER (NOT W4A8) ------------------
m = cfg("MiniMax-M3-fp8-experts-AutoRound"); mf = m["features"]
ck("M3: MoE+GQA+fp8+DSA (not MLA)", mf["is_moe"] and mf["is_gqa"] and not mf["is_mla"]
   and mf["weights"] == "fp8" and mf["is_dsa"], f"{mf['is_moe']}/{mf['is_gqa']}/{mf['weights']}/{mf['is_dsa']}")
ck("M3: top_k=4 layers=60", mf["top_k"] == 4 and mf["layers"] == 60)
ck("M3: static-fp8 native MME (NO W4A8)", m["env"].get("M3_FP8_STATIC") == "1"
   and m["env"].get("M3_FP8_MME") == "1" and "M3_EXPERT_RUNTIME" not in m["env"]
   and "HPU_MOE_FUSED_MONO_INT4FP8" not in m["env"])
ck("M3: eager + skip warmup + exact production capacity",
   m["env"].get("PT_HPU_LAZY_MODE") == "0" and m["env"].get("VLLM_SKIP_WARMUP") == "true"
   and m["vllm_args"].get("--max-num-seqs") == 128
   and m["vllm_args"].get("--max-model-len") == 4096
   and m["vllm_args"].get("--gpu-memory-utilization") == 0.75)
ck("M3: GLM-only DSA plugin stays off; dense-GQA fallback is explicit",
   "HPU_DSA_READY" not in m["env"] and "hpu-dsa-ext" not in m["plugins"]
   and any("dense-GQA fallback" in note for note in m["notes"]))
m_env = emit_env(os.path.join(BASE, "MiniMax-M3-fp8-experts-AutoRound"))
ck("M3 env: model sources + matched HPU runner auto-mounted",
   "minimax-m3/vllm:/opt/m3-vllm:ro" in m_env
   and "minimax-m3:/opt/m3-root:ro" in m_env
   and "vllm-gaudi-patch/worker/hpu_model_runner.py:/opt/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py:ro" in m_env)

# ---- Qwen3.6-27B: dense-Mamba-hybrid-fp8 -> EAGER (the crash-avoiding one) + fp8 ---------------------
q = cfg("Qwen3.6-27B-FP8"); qf = q["features"]
ck("Qwen3.6: Mamba-hybrid + fp8 (not MoE)", qf["is_mamba_hybrid"] and qf["weights"] == "fp8" and not qf["is_moe"],
   f"{qf['is_mamba_hybrid']}/{qf['weights']}/{qf['is_moe']}")
ck("Qwen3.6: EAGER mode (PT_HPU_LAZY_MODE=0) — CRITICAL (lazy crashes GDN)",
   q["env"].get("PT_HPU_LAZY_MODE") == "0", q["env"].get("PT_HPU_LAZY_MODE"))
ck("Qwen3.6: --quantization fp8 + fp8-KV", q["vllm_args"].get("--quantization") == "fp8"
   and q["vllm_args"].get("--kv-cache-dtype") == "fp8_inc")

# ---- Qwen3-30B-A3B-Int4: NEVER hand-tuned -> generalization: MoE-GQA-int4 -> W4A8 -------------------
n = cfg("Qwen3-30B-A3B-GPTQ-Int4"); nf = n["features"]
ck("Qwen3-30B (untuned): MoE+GQA+int4", nf["is_moe"] and nf["is_gqa"] and nf["weights"] == "int4",
   f"{nf['is_moe']}/{nf['is_gqa']}/{nf['weights']}")
ck("Qwen3-30B GPTQ: top_k=8 layers=48 -> gptq_hpu + LAZY",
   nf["top_k"] == 8 and nf["layers"] == 48 and n["vllm_args"].get("--quantization") == "gptq_hpu"
   and n["env"].get("PT_HPU_LAZY_MODE") == "1")

# ---- Fleet max-pool contract: capacity is one raw shared pool, never pre-divided by concurrency ------
max_recs = [
    emit(os.path.join(BASE, "GLM-5.2-int4-experts-AutoRound"), profile="max"),
    emit(os.path.join(BASE, "MiniMax-M3-fp8-experts-AutoRound"), profile="max"),
    emit(os.path.join(BASE, "Qwen3.6-27B-FP8"), profile="max"),
    emit(os.path.join(BASE, "Qwen3-30B-A3B-GPTQ-Int4"), profile="max"),
]
ck("max profile: every model uses auto resident pool without a profile concurrency ceiling",
   all("--num-gpu-blocks-override" not in rec["vllm_args"]
       and "--max-num-seqs" not in rec["vllm_args"]
       and rec["capacity_policy"] == {
           "hot_pool": "auto", "allocation": "shared", "scheduler_width": "engine"
       }
       for rec in max_recs))
ck("max profile: agent tool surfaces remain enabled",
   max_recs[0]["vllm_args"].get("--tool-call-parser") == "glm47"
   and max_recs[0]["vllm_args"].get("--reasoning-parser") == "qwen3")

# ====================================================================================================
# ROBUSTNESS regression guards (hardening pass): unknown arch, quant-validation landmines, malformed
# input, graft mis-wire, name sanitization. Each maps to a specific fix in autoconfig.py.
# ====================================================================================================

# ---- Fix 1: unknown/unsupported arch -> LOUD "UNVERIFIED ARCH" (never a silent guess) ----------------
u = emit(mk({"architectures": ["TotallyNewFancyForCausalLM"], "num_hidden_layers": 32, "hidden_size": 4096}, "Fancy-7B"))
ck("unknown arch -> arch_verified is False", u["features"]["arch_verified"] is False)
ck("unknown arch -> LOUD UNVERIFIED warning", any("UNVERIFIED ARCH" in w for w in u["warnings"]))
ck("unknown arch -> warning surfaces in emitted env",
   "UNVERIFIED ARCH" in emit_env(mk({"architectures": ["TotallyNewFancyForCausalLM"], "num_hidden_layers": 32}, "Fancy-7B")))
ck("known arch (GLM) -> arch_verified is True", gf["arch_verified"] is True)
ck("known arch -> NO unverified warning", not any("UNVERIFIED" in w for w in g["warnings"]))

# ---- Fix 2: quantization-variant landmines -> right --quantization / --dtype, or a warning -----------
ct = emit(mk({"architectures": ["FooForCausalLM"], "num_hidden_layers": 10,
              "quantization_config": {"quant_method": "compressed-tensors",
                                      "config_groups": {"g0": {"weights": {"num_bits": 8, "type": "float"}}}}}, "ct-fp8"))
ck("compressed-tensors fp8 -> --quantization compressed-tensors (NOT blanket fp8)",
   ct["vllm_args"].get("--quantization") == "compressed-tensors")
ck("real M3 (compressed-tensors) -> --quantization compressed-tensors",
   m["vllm_args"].get("--quantization") == "compressed-tensors")
p8 = emit(mk({"architectures": ["BarForCausalLM"], "num_hidden_layers": 10,
              "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}, "plain-fp8"))
ck("plain fp8 quant_method -> --quantization fp8", p8["vllm_args"].get("--quantization") == "fp8")
aw = emit(mk({"architectures": ["BazForCausalLM"], "num_hidden_layers": 10, "num_experts": 8, "num_experts_per_tok": 2,
              "quantization_config": {"quant_method": "awq", "bits": 4}}, "awq-moe"))
ck("awq int4 -> weights=int4, awq_hpu kernel, --dtype bfloat16",
   aw["features"]["weights"] == "int4" and aw["vllm_args"].get("--quantization") == "awq_hpu"
   and aw["vllm_args"].get("--dtype") == "bfloat16")
f16 = emit(mk({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": "float16",
               "num_hidden_layers": 28, "num_key_value_heads": 8}, "fp16-dense"))
ck("float16-declared UNQUANTIZED -> forced --dtype bfloat16 (Gaudi no fp16 fast path)",
   f16["features"]["weights"] == "bf16" and f16["vllm_args"].get("--dtype") == "bfloat16")
notk = emit(mk({"architectures": ["MysteryMoE"], "num_experts": 16, "num_hidden_layers": 24}, "moe-no-topk"))
ck("MoE with ABSENT top-k -> LOUD warning + routed-K defaults to 8",
   any("top-k" in w for w in notk["warnings"]) and notk["env"].get("HPU_MOE_ROUTED_K") == "8")
alt = emit(mk({"architectures": ["AltMoE"], "num_experts": 16, "moe_top_k": 2, "num_hidden_layers": 24}, "moe-altk"))
ck("MoE moe_top_k alt-key detected", alt["features"]["top_k"] == 2)
pq = emit(mk({"architectures": ["PartQ"], "num_hidden_layers": 4,
              "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                      "modules_to_not_convert": ["a.b", "c.d", "e.f"]}}, "partial-q"))
ck("partial quant (modules_to_not_convert) -> counted + noted",
   pq["features"]["partial_quant"] == 3 and any("Partial quantization" in n for n in pq["notes"]))

try:
    emit(mk({"architectures": ["X"], "num_hidden_layers": 2}, "bad-profile"), profile="typo-profile")
    ck("unknown profile -> LOUD ValueError", False)
except ValueError:
    ck("unknown profile -> LOUD ValueError", True)

# ---- Fix 3: malformed / missing input -> clear error, never a stack trace ----------------------------
try:
    emit(os.path.join(_TMP, "does-not-exist")); ck("missing config.json -> raises", False)
except FileNotFoundError:
    ck("missing config.json -> FileNotFoundError (clear)", True)
except Exception as e:  # noqa: BLE001
    ck("missing config.json -> FileNotFoundError (clear)", False, repr(e))
try:
    emit(mk_raw("{not valid json", "bad-json")); ck("invalid JSON -> ValueError", False)
except ValueError:
    ck("invalid JSON -> ValueError (clear, not stack trace)", True)
try:
    emit(mk_raw("[1,2,3]", "list-json")); ck("top-level JSON array -> ValueError", False)
except ValueError:
    ck("top-level JSON array -> ValueError (clear)", True)
em = emit(mk({}, "empty-cfg"))
ck("empty {} config -> no crash + arch '?' + UNVERIFIED",
   em["features"]["arch"] == "?" and any("UNVERIFIED" in w for w in em["warnings"]))
ck("empty [] architectures -> arch '?'", emit(mk({"architectures": []}, "empty-arch"))["features"]["arch"] == "?")
ck("architectures as bare string -> parsed whole (not char-split)",
   emit(mk({"architectures": "SoloForCausalLM", "num_hidden_layers": 4}, "str-arch"))["features"]["arch"] == "SoloForCausalLM")
nd = emit(mk({"architectures": ["X"], "text_config": "oops", "quantization_config": "nope",
              "layer_types": "weird", "num_hidden_layers": 4}, "nondict"))
ck("non-dict text_config/quant_config/layer_types -> no crash", nd["features"]["arch"] == "X")

# ---- Fix 3b: deep nesting beyond text_config (thinker_config / language_config) ----------------------
th = emit(mk({"architectures": ["OmniForConditionalGeneration"],
              "thinker_config": {"architectures": ["ThinkerMoe"], "num_experts": 64, "num_experts_per_tok": 8,
                                 "num_hidden_layers": 36, "kv_lora_rank": 512}}, "omni"))["features"]
ck("thinker_config nesting -> reads nested MoE+MLA arch",
   th["is_moe"] and th["top_k"] == 8 and th["layers"] == 36 and th["is_mla"])

# ---- Fix 4: graft never mis-wired to the wrong checkpoint --------------------------------------------
wrong = mk({"architectures": ["Glm4MoeForCausalLM"], "num_experts": 128, "num_experts_per_tok": 8,
            "num_hidden_layers": 46, "hidden_size": 4096,
            "quantization_config": {"quant_method": "gptq", "bits": 4}}, "glm-4-9b-int4")
wart = resolve_artifacts(wrong, emit(wrong)["features"])
ck("graft NOT mis-wired to glm52 for a 'glm'-named but wrong arch/version", wart["graft"] is None)
if os.path.isdir(os.path.join(BASE, "glm52-fp8-other")):
    gart = resolve_artifacts(os.path.join(BASE, "GLM-5.2-int4-experts-AutoRound"), gf)
    ck("real GLM-5.2 STILL gets the glm52-fp8-other graft (arch+name both match)",
       bool(gart["graft"]) and gart["graft"].endswith("glm52-fp8-other"))

# ---- Fix 5: served-name sanitization ----------------------------------------------------------------
s = _sanitize_name("GLM-5.2 (Test).v2!!")
ck("sanitize -> only valid [a-z0-9._-] chars, starts alphanumeric",
   re.fullmatch(r"[a-z0-9._-]+", s) is not None and s[0].isalnum(), s)
ck("sanitize all-invalid -> 'model' fallback", _sanitize_name("///") == "model")
ck("emit_env SERVED_NAME sanitized (lowercased, no junk)",
   "SERVED_NAME=weird.name_up" in emit_env(mk({"architectures": ["X"], "num_hidden_layers": 2}, "Weird.Name_UP")))

shutil.rmtree(_TMP, ignore_errors=True)

print()
if FAILS:
    print(f"==== {len(FAILS)} FAIL(S): {FAILS} ===="); sys.exit(1)
print("==== AUTO-CONFIG detection contract + robustness guards HOLD ====")
