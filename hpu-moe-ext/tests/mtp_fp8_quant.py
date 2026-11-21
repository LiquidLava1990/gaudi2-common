"""Offline fp8 quantizer for the GLM-5.1 MTP layer (78).

Reads the zai-org ORIGINAL bf16 layer-78 (the only non-dead copy of the native MTP head) ONCE,
quantizes every big Linear weight to fp8 (float8_e4m3fn + per-output-channel scale), and writes
an fp8 graft dir. The serve then loads ONLY fp8 (no bf16 ever stored/served) — same provenance as
the int4 main model, which was likewise quantized from a bf16 original.

fp8 preserves accept exactly: CPU oracle (glm_cpu_mtp_eval.py) measured bf16=73.3% == fp8sim=73.3%.
Big Linears -> fp8; norms / router gate / small indexer vectors stay fp16 (tiny, precision-sensitive,
exactly how every quantized checkpoint keeps its scale vectors). Gaudi fn-vs-fnuz cancels in the
self-consistent serve-side cast/gemm round-trip (see fp8_routed_moe.py), so store standard e4m3fn.

Run (in the serve image, ~21GB bf16 in, ~11GB fp8 out):
  docker run --rm -v /mnt/fastmodels:/models --entrypoint python3 vllm-gaudi:next \
    /w/tests/mtp_fp8_quant.py --src /models/glm51-orig-l78 --out /models/glm51-fp8-l78
"""
import argparse, json, os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# weight-name fragments that must stay full-precision: small vectors (norms/router) AND the DSA
# lightning-indexer (its wk/wq_b/weights_proj/k_norm load via a SPECIAL int4-only loader, not a
# standard fp8 module — generic fp8 quant of them fails the weight-loader). The indexer is tiny.
_KEEP = ("layernorm", "_norm", ".norm", "enorm", "hnorm", "e_score_correction_bias",
         "gate.weight", "indexer.")
FP8_MAX = 448.0


def _is_keep(name: str) -> bool:
    return any(k in name for k in _KEEP)


def quant_w(w: torch.Tensor):
    """[out,in] bf16 -> (fp8 [out,in], scale [1]) PER-TENSOR symmetric e4m3.
    vllm's Fp8LinearMethod / Fp8MoEMethod register a scalar per-tensor weight_scale (linear [1],
    MoE [num_experts,2]/[num_experts]); per-channel [out] scales fail the weight-loader size check."""
    wf = w.to(torch.float32)
    scale = wf.abs().amax().clamp_min(1e-8) / FP8_MAX               # scalar (per-tensor)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(1).to(torch.float32)                    # [1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open(f"{a.src}/model.safetensors.index.json"))["weight_map"]
    # the orig dir ships the FULL 282-shard index but only the layer-78 shards are present;
    # keep only tensors whose shard file actually exists (= the spec/MTP layer).
    names = sorted(n for n in idx if os.path.exists(f"{a.src}/{idx[n]}"))
    print(f"[fp8q] {len(names)} present layer-78 tensors from {a.src} (of {len(idx)} in full index)", flush=True)

    _h = {}
    def rd(n):
        fn = idx[n]
        if fn not in _h:
            _h[fn] = safe_open(f"{a.src}/{fn}", framework="pt", device="cpu")
        return _h[fn].get_tensor(n)

    tensors, new_map = {}, {}
    nq = nk = 0
    for n in names:
        w = rd(n)
        if (not _is_keep(n)) and w.dim() == 2 and min(w.shape) >= 64:
            q, s = quant_w(w)
            tensors[n] = q                     # fp8 weight (same name)
            tensors[n + ".weight_scale"] = s    # per-output-channel scale
            new_map[n] = "l78_fp8.safetensors"
            new_map[n + ".weight_scale"] = "l78_fp8.safetensors"
            nq += 1
        else:
            tensors[n] = w.to(torch.float16) if w.dtype == torch.bfloat16 else w
            new_map[n] = "l78_fp8.safetensors"
            nk += 1
    save_file(tensors, f"{a.out}/l78_fp8.safetensors", metadata={"format": "pt"})
    json.dump({"weight_map": new_map, "fp8_quantized": True, "fp8_max": FP8_MAX},
              open(f"{a.out}/model.safetensors.index.json", "w"))
    sz = os.path.getsize(f"{a.out}/l78_fp8.safetensors") / 1e9
    print(f"[fp8q] wrote {a.out}/l78_fp8.safetensors  ({sz:.1f}GB)  fp8-quantized={nq} kept-fp16={nk}", flush=True)
    print(f"[fp8q] scales stored as <name>.weight_scale (per-output-channel). NO bf16 in output.", flush=True)


if __name__ == "__main__":
    main()
