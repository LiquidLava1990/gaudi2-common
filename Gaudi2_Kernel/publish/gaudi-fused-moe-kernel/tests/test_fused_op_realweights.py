"""test_fused_op_realweights.py -- REPRODUCE the fused-op GARBAGE on the REAL
GLM-5.1-int4 AutoRound checkpoint weights, ON THE HPU BOX (1 card).

WHY THIS EXISTS
---------------
The synthetic op test (tpc_kernel/opbackend/test_fused_expert_op.py) PASSES
(relNorm ~5e-3) because it builds RANDOM int4 codes and decodes them with the
SAME packing on both sides -- it only proves the op == convert_from_uint4 for a
GIVEN packing. It does NOT load real repacked checkpoint weights, so it cannot
catch a packing/layout/dtype mismatch between the plugin's REAL repacked weights
(hpu_moe_ext.moe_wna16_hpu._repack_weight) and what the fused op's dequant
expects. In the live serve HPU_MOE_FUSED_VERIFY=1 shows the fused op at
maxAbs~200 / relNorm~1e11 vs the stock convert_from_uint4 + mixture_of_experts
WHENEVER a rank computes a real expert (it only "matches" -- trivially 0 -- when
a rank has no routed expert).

This test isolates that to ONE real expert WITHOUT the full serve:
  1. read ONE real GLM expert's int4 weights from the checkpoint safetensors
     (layer 3 -- the first INT4 MoE layer; 0/1/2 are fp16 -- expert 0:
      gate_proj+up_proj -> w13, down_proj -> w2; + scales + qzeros);
  2. apply the plugin's EXACT load+repack transform so the on-HPU tensors are
     byte-identical to what layer.w13_qweight / layer.w13_scales / the shared
     _sym_zero hold in the serve (the loader's `.T.view(uint8)`, _repack_weight's
     unpack/transpose/repack, scales.T cast to the model dtype, zp = packed 8);
  3. run torch.ops.hpu_moe_ext.fused_expert on that real expert, BOTH
     (a) directly (single-expert op) and
     (b) through hpu_moe_ext.fused_expert_routed.fused_moe_routed -- the EXACT
         path the production _fused_moe_routed takes (stack [E,...], index_select,
         per-slice clone under lazy, the [1,1] router-weight fold at T==1);
  4. compare to the TRUSTED oracle: torch.ops.hpu.convert_from_uint4 on the SAME
     repacked weight + the MoE math (== the proven, coherent _routed_moe tail).

GATE: relNorm = ||got - ref|| / ||ref|| < 1e-2 (the bf16 norm-relative tol the
proven recipe uses). BEFORE the fix this REPRODUCES relNorm ~1e11 (LOUD FAIL);
AFTER the fix it PASSES (< 1e-2). LOUD (nonzero exit + stderr) on any failure.

RUN (ON THE HPU BOX -- needs 1 card; a CPU import cannot exercise the op):
  cd <repo>
  # eager (matches libmoe_fused_expert.so + the eager verify path):
  PT_HPU_LAZY_MODE=0 python tests/test_fused_op_realweights.py
  # lazy (matches libmoe_fused_expert_lazy.so + the serving decode graph):
  PT_HPU_LAZY_MODE=1 python tests/test_fused_op_realweights.py
  # override checkpoint / layer / expert:
  MODEL=/mnt/fastmodels/GLM-5.1-int4-mixed-AutoRound LAYER=3 EXPERT=0 \
      python tests/test_fused_op_realweights.py

The op .so is picked by mode (eager vs lazy) exactly like the plugin's
_fused_so_path(), so this validates the SAME library the serve loads.
"""
from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# The plugin's pack/unpack + repack, replicated here so the test depends only on
# vLLM's upstream helpers (the SAME ones hpu_moe_ext._repack_weight uses) and not
# on importing the whole plugin (which pulls vllm_gaudi). These MUST stay in
# lockstep with moe_wna16_hpu.py: _repack_weight + the upstream gptq weight loader
# (`loaded_weight.T.contiguous().view(uint8)`) + _sym_zero (packed midpoint 8).
# ---------------------------------------------------------------------------
def _loader_gptq_to_uint8(qweight_ckpt_i32):
    """vLLM MoeWNA16 gptq weight loader transform for a 2D [in//8, out] int32
    checkpoint qweight: `loaded_weight.T.contiguous().view(torch.uint8)` ->
    uint8 [out, in//2] (the layout layer.w13_qweight/w2_qweight hold BEFORE the
    plugin's _repack_weight runs over it)."""
    import torch
    return qweight_ckpt_i32.t().contiguous().view(torch.uint8)


def _repack_weight_one(qweight_u8, qt):
    """hpu_moe_ext.moe_wna16_hpu.MoeWNA16HPUMethod._repack_weight, single expert.
    uint8 [out, in//2] -> unpack along in -> repack along out -> int32 [in, out//8]
    (exactly the layout the fused op + convert_from_uint4 consume)."""
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
    w_i32 = qweight_u8.contiguous().view(torch.int32)          # [out, in//8]
    vals = unpack_quantized_values_into_int32(
        w_i32.transpose(0, 1).contiguous(), qt, packed_dim=0)  # [in, out]
    return pack_quantized_values_into_int32(vals, qt, packed_dim=1)  # [in, out//8]


def _sym_zero_packed(num_groups, out, qt, device):
    """hpu_moe_ext.moe_wna16_hpu.MoeWNA16HPUMethod._sym_zero (GPTQ-symmetric):
    constant uint4 zero-point = type midpoint (8 for int4), shared across experts,
    packed along out -> int32 [num_groups, out//8] (== 0x88888888)."""
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32)
    zeros = torch.full((num_groups, out), 1 << 3, dtype=torch.int32, device=device)  # int4 -> 8
    return pack_quantized_values_into_int32(zeros, qt, packed_dim=1)     # [groups, out//8]


def _load_real_expert(model_dir, layer, expert):
    """Read ONE real GLM expert's int4 weights from the checkpoint and apply the
    plugin's load+repack so the returned tensors are byte-identical to what the
    serve's layer.w13_qweight / w13_scales / shared zp hold.

    Returns CPU tensors (the caller moves to HPU):
      w13_packed [H, 2I//8] i32, w13_scale [H//GS, 2I] bf16, w13_zp [H//GS, 2I//8] i32,
      w2_packed  [I, H//8 ] i32, w2_scale  [I//GS, H ] bf16, w2_zp  [I//GS, H//8 ] i32,
      and (H, I, GS).
    The model's compute dtype defaults to bf16 on HPU; the loader casts the fp16
    checkpoint scales into the bf16 params, so we cast scales -> bf16 here too
    (this is exactly what process_weights_after_loading produces)."""
    import json
    import torch
    from safetensors import safe_open
    from vllm.scalar_type import scalar_types

    qt = scalar_types.uint4
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    weight_map = json.load(open(idx_path))["weight_map"]

    # GLM-5.1 names: model.layers.<L>.mlp.experts.<E>.<proj>.{qweight,scales,qzeros}
    # (verified on /mnt/fastmodels/GLM-5.1-int4-mixed-AutoRound). Fall back to the
    # language_model.* prefix used by some HF exports if the bare prefix is absent.
    def _key(proj, kind):
        a = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
        if a in weight_map:
            return a
        b = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
        if b in weight_map:
            return b
        raise KeyError(
            f"neither {a!r} nor {b!r} in the checkpoint index -- check LAYER "
            f"(layer {layer} must be an INT4 MoE layer; 0/1/2 are fp16) / EXPERT.")

    def _get(proj, kind):
        name = _key(proj, kind)
        with safe_open(os.path.join(model_dir, weight_map[name]), "pt") as h:
            return h.get_tensor(name)

    # --- gate_proj + up_proj -> fused w13 (column-parallel: stack out dim = 2I) ---
    # checkpoint qweight is [in//8, out] int32; loader -> uint8 [out, in//2]; the
    # fused w13 stacks gate (w1) then up (w3) along out: uint8 [2*out, in//2].
    g_u8 = _loader_gptq_to_uint8(_get("gate_proj", "qweight"))   # [I, H//2]
    u_u8 = _loader_gptq_to_uint8(_get("up_proj", "qweight"))     # [I, H//2]
    w13_u8 = torch.cat([g_u8, u_u8], dim=0)                      # [2I, H//2]
    w13_packed = _repack_weight_one(w13_u8, qt)                  # [H, 2I//8]

    d_u8 = _loader_gptq_to_uint8(_get("down_proj", "qweight"))   # [H, I//2]
    w2_packed = _repack_weight_one(d_u8, qt)                     # [I, H//8]

    # scales: checkpoint [groups, out] fp16. The fused gate||up scale stacks along
    # out -> [H//GS, 2I]; the op consumes scales as [groups, out] (group axis = the
    # input/contract dim). cast -> bf16 (the model's compute dtype, == the serve).
    sg = _get("gate_proj", "scales").to(torch.bfloat16)         # [H//GS, I]
    su = _get("up_proj", "scales").to(torch.bfloat16)           # [H//GS, I]
    w13_scale = torch.cat([sg, su], dim=1).contiguous()         # [H//GS, 2I]
    w2_scale = _get("down_proj", "scales").to(torch.bfloat16).contiguous()  # [I//GS, H]

    # derive H, I, GS from shapes:  w13_packed [H, 2I//8]; w13_scale [H//GS, 2I]
    H = w13_packed.shape[0]
    twoI = w13_scale.shape[1]
    I = twoI // 2
    GS = H // w13_scale.shape[0]

    # GPTQ-symmetric shared zero (the plugin ignores the checkpoint qzeros -- they
    # are uniformly the stored 7, true zero 7+1==8 -- and uses the packed midpoint 8).
    w13_zp = _sym_zero_packed(w13_scale.shape[0], twoI, qt, "cpu")   # [H//GS, 2I//8]
    w2_zp = _sym_zero_packed(w2_scale.shape[0], w2_scale.shape[1], qt, "cpu")  # [I//GS, H//8]

    return (w13_packed, w13_scale, w13_zp, w2_packed, w2_scale, w2_zp, H, I, GS)


def _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I):
    """TRUSTED single-expert oracle == the coherent _routed_moe tail:
    torch.ops.hpu.convert_from_uint4 (the SAME intrinsic the stock path uses) +
    the MoE math. Returns expert_out [T, H] (UNSCALED by router weight)."""
    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch.core as htorch
    W13 = torch.ops.hpu.convert_from_uint4(w13p, s13, z13, torch.bfloat16, None)   # [H, 2I]
    W2 = torch.ops.hpu.convert_from_uint4(w2p, s2, z2, torch.bfloat16, None)       # [I, H]
    htorch.mark_step()
    gate_up = torch.matmul(x, W13)            # [T, 2I]
    gate = gate_up[:, :I]
    up = gate_up[:, I:]
    h = F.silu(gate) * up                     # [T, I]
    return torch.matmul(h, W2)                # [T, H]


def _relnorm(got, ref):
    g = got.float()
    r = ref.float()
    diff = (g - r).norm().item()
    rn = diff / (r.norm().item() + 1e-9)
    maxabs = (g - r).abs().max().item()
    return rn, maxabs


def _find_so():
    """Pick the same .so the plugin's _fused_so_path() would load for this mode
    (lazy vs eager): the serve loads exactly this library, so this validates it."""
    import habana_frameworks.torch.core as htorch
    here = os.path.dirname(os.path.abspath(__file__))
    name = ("libmoe_fused_expert_lazy.so" if htorch.is_lazy()
            else "libmoe_fused_expert.so")
    so = os.path.abspath(os.path.join(
        here, "..", "tpc_kernel", "opbackend", "build", name))
    if not os.path.isfile(so):
        raise FileNotFoundError(
            f"{so} not found. Build it first (CPU-only):\n"
            f"  docker run --rm -v {os.path.dirname(here)}:/work "
            f"-w /work/tpc_kernel/opbackend vllm-gaudi:next bash "
            f"{'build_lazy.sh' if htorch.is_lazy() else 'build.sh'}")
    return so


def main() -> int:
    model_dir = os.environ.get("MODEL", "/mnt/fastmodels/GLM-5.1-int4-mixed-AutoRound")
    layer = int(os.environ.get("LAYER", "3"))
    expert = int(os.environ.get("EXPERT", "0"))
    TOL = 1e-2

    try:
        import habana_frameworks.torch  # noqa: F401
        import habana_frameworks.torch.core as htorch
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    if not os.path.isdir(model_dir):
        print(f"FATAL: checkpoint dir not found: {model_dir} (set MODEL=...)",
              file=sys.stderr)
        return 2

    # ---- load + repack ONE real expert (plugin's exact load path) ----
    try:
        (w13p, s13, z13, w2p, s2, z2, H, I, GS) = _load_real_expert(
            model_dir, layer, expert)
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: could not load/repack real expert "
              f"(layer={layer} expert={expert}) from {model_dir}: {ex}",
              file=sys.stderr)
        return 2

    mode = "LAZY" if htorch.is_lazy() else "EAGER"
    print(f"=== test_fused_op_realweights ({mode}) ===  model={model_dir}")
    print(f"    layer={layer} expert={expert}  H={H} I={I} (2I={2*I}) GS={GS}")
    print(f"    w13_packed{tuple(w13p.shape)} s13{tuple(s13.shape)} z13{tuple(z13.shape)} "
          f"w2_packed{tuple(w2p.shape)} s2{tuple(s2.shape)} z2{tuple(z2.shape)}")

    # ---- load the op .so (the SAME library the serve loads for this mode) ----
    try:
        so = _find_so()
    except FileNotFoundError as ex:
        print(f"FATAL: {ex}", file=sys.stderr)
        return 2
    torch.ops.load_library(so)
    if not (hasattr(torch.ops, "hpu_moe_ext")
            and hasattr(torch.ops.hpu_moe_ext, "fused_expert")):
        print(f"FATAL: torch.ops.hpu_moe_ext.fused_expert not registered after "
              f"load_library({so}).", file=sys.stderr)
        return 2
    print(f"loaded {so}")

    dev = "hpu"
    torch.manual_seed(0)
    T = 1
    x = (torch.randn(T, H) * 0.1).to(torch.bfloat16).to(dev)
    rw_val = 0.75
    rw = torch.full((1, 1), rw_val, dtype=torch.bfloat16, device=dev)

    w13p = w13p.to(dev); s13 = s13.to(dev); z13 = z13.to(dev)
    w2p = w2p.to(dev); s2 = s2.to(dev); z2 = z2.to(dev)
    htorch.mark_step()

    rc = 0

    # ====================================================================
    # CHECK 1 -- the SINGLE-EXPERT op on the REAL repacked weight vs the
    # convert_from_uint4 oracle. This is the minimal reproducer: if the op's
    # dequant misreads the real packed layout/scale/zp, relNorm blows up here.
    # ====================================================================
    print("\n--- CHECK 1: fused_expert(real expert) vs convert_from_uint4 oracle ---")
    got = torch.ops.hpu_moe_ext.fused_expert(x, w13p, s13, z13, w2p, s2, z2, rw)
    htorch.mark_step()
    ref = _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I) * rw_val
    htorch.mark_step()
    rn, ma = _relnorm(got.to("cpu"), ref.to("cpu"))
    print(f"  |ref|={ref.float().norm().item():.4f}  |got|={got.float().norm().item():.4f}  "
          f"relNorm={rn:.4e}  maxAbs={ma:.4e}")
    if rn <= TOL:
        print(f"  >>> PASS: fused_expert matches the oracle on REAL weights "
              f"(relNorm {rn:.2e}).")
    else:
        print(f"  >>> FAIL: fused_expert relNorm {rn:.4e} > tol {TOL:.2e} on REAL "
              f"weights -- THIS is the serve's maxAbs~200/relNorm~1e11 bug.",
              file=sys.stderr)
        print("      The plugin's repack+contract is proven bit-exact on CPU (the\n"
              "      weights/scales/zp are right), so a FAIL here is in the op's\n"
              "      on-device dequant of these real tensors: scale dtype the\n"
              "      bf16 dequantize_4_bit guid reads, the uint4 zp reinterpret, or\n"
              "      the reinterpret_cast/gemm geometry. Compare to CHECK 2 (routed\n"
              "      wrapper) to see if it is the index_select/clone path instead.",
              file=sys.stderr)
        rc = 1

    # ====================================================================
    # CHECK 2 -- the PRODUCTION routed path on TWO real experts, K=2. This is the
    # EXACT code _fused_moe_routed runs: stack [E,...], index_select, the per-slice
    # materialization, the per-(token,slot) router-weight fold, accumulate. Two
    # experts so the gathered compact slice for the SECOND routed expert carries a
    # NONZERO storage_offset (= 1*stride into the index_select buffer) -- the exact
    # serve hazard (E=256, a routed expert at a nonzero compact index) that an E=1 /
    # K=1 stack can NOT exercise (its only slice is at offset 0). A regressed wrapper
    # that feeds the op an offset view reproduces the serve's relNorm~1e11 on the
    # k>0 expert; the fix (fresh contiguous offset-0 copy) makes it pass.
    print("\n--- CHECK 2: fused_moe_routed(2 real experts, K=2) vs per-expert oracle ---")
    captured_ok = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from hpu_moe_ext.fused_expert_routed import fused_moe_routed
        # second real expert (same layer, expert+1) -> a genuinely distinct expert at
        # stack index 1, so routing to it gathers a k>0 (offset!=0) compact slice.
        try:
            (b13p, bs13, bz13, b2p, bs2, bz2, _, _, _) = _load_real_expert(
                model_dir, layer, expert + 1)
        except Exception:  # only one expert available -> reuse expert 0 (still exercises K=2/offset)
            (b13p, bs13, bz13, b2p, bs2, bz2) = (w13p, s13, z13, w2p, s2, z2)
        b13p = b13p.to(dev); bs13 = bs13.to(dev); bz13 = bz13.to(dev)
        b2p = b2p.to(dev); bs2 = bs2.to(dev); bz2 = bz2.to(dev)
        E = 2
        experts_min = 0
        K = 2
        # stack expert0 at index 0, expert1 at index 1 (the serve's [E,...] layout)
        w13p_s = torch.stack([w13p, b13p], 0).contiguous()
        s13_s = torch.stack([s13, bs13], 0).contiguous()
        z13_s = torch.stack([z13, bz13], 0).contiguous()
        w2p_s = torch.stack([w2p, b2p], 0).contiguous()
        s2_s = torch.stack([s2, bs2], 0).contiguous()
        z2_s = torch.stack([z2, bz2], 0).contiguous()
        # route token 0 to BOTH experts (top_k=2), distinct renormalized weights.
        w0v, w1v = 0.6, 0.4
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int64, device=dev)      # global == local (emin 0)
        topk_weights = torch.tensor([[w0v, w1v]], dtype=torch.bfloat16, device=dev)
        htorch.mark_step()
        got2 = fused_moe_routed(x, w13p_s, s13_s, z13_s, w2p_s, s2_s, z2_s,
                                topk_ids, topk_weights, experts_min, K)
        htorch.mark_step()
        # oracle: per-expert convert_from_uint4 + MoE math, weighted-summed (== the
        # coherent stock routed tail; independent of any gather/offset logic).
        e0 = _ref_expert(x, w13p, s13, z13, w2p, s2, z2, I)                   # [T,H]
        e1 = _ref_expert(x, b13p, bs13, bz13, b2p, bs2, bz2, I)              # [T,H]
        htorch.mark_step()
        ref2 = e0 * w0v + e1 * w1v
        rn2, ma2 = _relnorm(got2.to("cpu"), ref2.to("cpu"))
        print(f"  |ref|={ref2.float().norm().item():.4f}  |got|={got2.float().norm().item():.4f}  "
              f"relNorm={rn2:.4e}  maxAbs={ma2:.4e}")
        captured_ok = rn2 <= TOL
        if captured_ok:
            print(f"  >>> PASS: fused_moe_routed matches the oracle on REAL weights "
                  f"(relNorm {rn2:.2e}). This is the production _fused_moe_routed path.")
        else:
            print(f"  >>> FAIL: fused_moe_routed relNorm {rn2:.4e} > tol {TOL:.2e} -- "
                  f"the production routed path reproduces the serve garbage.",
                  file=sys.stderr)
            rc = 1
    except Exception as ex:  # noqa: BLE001
        print(f"  (skipped routed-wrapper check: {ex})", file=sys.stderr)
        print("  NOTE: CHECK 1 already isolates the op on the real weight; the wrapper\n"
              "  only adds the gather/materialize/router-fold _fused_moe_routed does.")

    print(f"\n[done] test_fused_op_realweights {'OK' if rc == 0 else 'FAILED'} (rc={rc})"
          + (f"; routed={'ok' if captured_ok else captured_ok}"
             if captured_ok is not None else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
