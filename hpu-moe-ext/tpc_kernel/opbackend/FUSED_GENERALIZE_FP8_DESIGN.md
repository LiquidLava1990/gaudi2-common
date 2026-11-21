# Fused MoE op: generalization + fp8 toggles (item 4 of 4)

Design note for the three deliverables that build on the committed working op
`torch.ops.hpu_moe_ext.fused_expert` (commit 5b80797, `moe_fused_expert_backend.cpp`,
OpBackend, PyTorch-row-major int4 GPTQ-sym, SRAM-resident, relNorm 4.2e-3 on device).
All on-device runs are the box owner's; this note + the new files are CPU-only-checkable.

---

## 1. GENERALIZE across models — verdict: **the op + plugin generalize with little/no config.**

**The op is fully shape-parameterized.** `MoeFusedExpert::AddNode`
(`moe_fused_expert_backend.cpp:133-191`) derives **every** dimension from the input
tensor geometry — `M,H` from `x.sizes()`, `2I` from `w13_scale.size(1)` (so `I=2I/2`),
and the group size is never read as a scalar at all: the `dequantize_4_bit` complex-guid
infers grouped-vs-non-grouped from `scale.size(0)` vs `input.size(0)` on device. Nothing
is hardcoded to GLM's `H=6144,I=2048,GS=128`. So a new MoE model "just works" once its
per-expert int4 weights are in the proven PyTorch-row-major layout.

**`tests/test_fused_expert_general.py`** exercises that claim on the model matrix
(synthetic int4 weights, `relNorm < 1e-2` each, reuses the proven helpers from
`test_fused_expert_op.py` so it adds zero new numeric assumptions):

| profile | H | I | 2I | GS |
|---|---|---|---|---|
| GLM-5.1 | 6144 | 2048 | 4096 | 128 |
| DeepSeek-V3-ish | 7168 | 2048 | 4096 | 128 |
| Nemotron-ish | 8192 | 2048 | 4096 | 128 |
| MiniMax-ish | 6144 | 2048 | 4096 | 64 |
| odd-width + GS32 | 5120 | 1536 | 3072 | 32 |

Run on the box: `python tests/test_fused_expert_general.py` (prints a per-profile
PASS/FAIL table + an overall verdict; nonzero exit on any FAIL).

**Plugin config autodetect — confirmed from `moe_wna16_hpu.py`:**
`HPUWNA16MoEConfig.from_hf` (`moe_wna16_hpu.py:137-153`) reads `bits`, `group_size`, and
`sym` straight from the checkpoint's HF `quantization_config`: `weight_bits=qcfg["bits"]`,
`group_size=qcfg["group_size"]`, and for GPTQ `has_zp = not qcfg.get("sym", True)`
(AWQ: `zero_point`). `auto-round`/`auto_round` is mapped to `gptq` (same int4-sym packing).
So bits/group-size/sym need **no per-model config** — a new GPTQ/AWQ/AutoRound int4
checkpoint is autodetected.

**Gaps (small, all device-confirmable):**
1. **GS=32/64 on real checkpoints** is only exercised here by synthetic weights — the
   complex-guid grouped path is the same code, but confirm on a real MiniMax/finer-GS
   checkpoint when one is on disk.
2. **Per-expert layout assumption.** The op wants the per-expert repack the plugin
   already produces (`_repack_weight`: int32 `[in, out//8]`, scales transposed to
   `[groups, out]`, shared sym zero `[groups, out//8]`). A model whose loader emits a
   different packed orientation needs the plugin's repack, not the op, to adapt — the op
   itself is orientation-fixed (PyTorch row-major, last dim = FCD).
3. **AWQ per-expert (asymmetric) zero-point** is out of scope for the *single-expert
   fused op path's* current sym-zero wiring in these tests (the plugin handles AWQ in the
   monolithic op; the fused op was validated GPTQ-sym). Not a generalization blocker for
   the GPTQ/AutoRound families (GLM, DeepSeek, Qwen, Nemotron, MiniMax int4-sym).
4. The op is **one expert**; the routed K-expert wrapper (gather/topk/remap/accumulate)
   is the plugin's job (`_routed_moe`) and is model-agnostic — unchanged here.

Net: **no new per-model config is required** for the int4-sym MoE families; the only
"config" is the checkpoint's own `quantization_config`, which is already autodetected.

---

## 2. fp8-COMPUTE toggle — **scaffolded precisely (not built); exact node changes below.**

New files (separate `.so`/library so the proven bf16 op is byte-untouched):
- `moe_fused_expert_fp8_backend.cpp` — registers
  `torch.ops.hpu_moe_ext_fp8.fused_expert_fp8` (separate library namespace
  `hpu_moe_ext_fp8` so it can coexist in-process with `hpu_moe_ext`).
- `build_fp8.sh` — `build.sh` with `SRC`=the fp8 backend, `OUT`=`libmoe_fused_expert_fp8.so`,
  `-DFUSED_FP8=1` (and optional `-DBUILD_FP8_NODES=1`). Link/symbol logic identical.
- `test_fused_expert_fp8_op.py` — on-box accuracy gate at a **looser** tolerance.

**Built or scaffolded?** Scaffolded. The file compiles and, with the default
`BUILD_FP8_NODES=0`, falls back to the proven bf16 recipe so the `.so` always loads and
the schema/registration/harness can be validated CPU-side. The fp8 node bodies are
written out in full as the spec and gated behind `BUILD_FP8_NODES=1`; flipping it on (and
supplying the two param structs + the in-graph amax wiring) emits the real fp8 nodes for
the on-device bring-up. Reason it is a scaffold, not a finished `.so`: the bf16 op needed
a multi-iteration on-device geometry/SIF bring-up (documented in its header); the 8 new
fp8 nodes (`ns_ConvertToFp8::ParamsV2`, `ns_Fp8Gemm::ParamsV2`, the amax/reduce wiring)
need the same device validation loop, which is the box owner's run.

**Node-change spec — the topology is the SAME; only the marked deltas change** (full
detail in the backend file header `NODE SEQUENCE` block). Per GEMM, instead of feeding
the bf16 dequant straight into `gemm`:

```
W*_bf16 = dequantize_4_bit_bf16(...)            # UNCHANGED — int4 dequant stays bf16
s_w   = reduce_max(|W*_bf16|)/240               # NEW: forward recovery scale [1] f32
W*_fp8 = convert_to_fp8(W*_bf16, 1/s_w)         # NEW: bf16 -> fp8, compress scale 1/s
s_a   = reduce_max(|act|)/240 ; a_fp8 = convert_to_fp8(act, 1/s_a)   # NEW: activation -> fp8
gu/y  = fp8_gemm(a_fp8, W*_fp8, scaleA=s_a, scaleB=s_w, out=bf16)    # was `gemm`
```

Exact bridge constructions (reused verbatim from this tag's own ops):
- **`convert_to_fp8`**: `get_guid_with_precision("convert_to_fp8", BFloat16)`, inputs
  `{data, inv_scale}`, out `{shape, Float8_e4m3fn}`, params
  `ns_ConvertToFp8::ParamsV2{round_mode = CAST_ROUND_HALF_NE}`
  (`fp8_ops.cpp:153` + `GetCastParams:126`). The scale passed here is the **inverse** 1/s.
- **`fp8_gemm`**: `get_guid_with_precision("fp8_gemm", BFloat16)`, inputs
  `{A_fp8, B_fp8, scaleA, scaleB, bias=nullptr, [D if accumulate], nullptr}` (the trailing
  `nullptr` is the `FUSE_CONVERT_MME` slot the GC fills), out `{shape, BFloat16, 0}`, params
  `ns_Fp8Gemm::ParamsV2{transpose_a=false, transpose_b=false, ...}`
  (`fp8_gemm_gen.cpp:206-223`). scaleA/scaleB are the **forward** recovery scales s.
- in-graph amax: `abs_fwd` -> `reduce_max_fwd` ([1] f32), then 1/s = 240/amax and
  s = amax/240 (`BuildConstantTensor(240)` + reciprocal/mult). Keeps the whole op one
  captured recipe (no host-side scalar per replay). Equivalently use the cast's built-in
  amax output (`CastToFp8` with a non-empty amax out, `fp8_ops.cpp:162`).

The W\*_fp8/x_fp8/h_fp8 stay **non-persistent** graph-internal tensors, so the GC still
bundles dequant→cast→MME — SRAM residency is preserved and the fp8 weight is half the
bf16 bytes, so the win is at least as large. Gaudi-2 MME runs fp8 at ~2× bf16.

**ACCURACY CAVEAT (critical — this is the project's fn-vs-fnuz history):**
- int4→fp8 is **double quantization** (int4 checkpoint, then bf16→fp8 e4m3, 3 mantissa
  bits) + the activation is fp8-quantized. Expect a **few-% relNorm**, coherent but **not**
  within the bf16 op's 1e-2. The on-box gate uses a **looser tol** (default 5e-2 in the
  test; confirm the real number on device). Same documented bar as the zero-build sibling
  `hpu_moe_ext/fp8_routed_moe.py`.
- The dequant **must NOT emit `float8_e4m3fn` directly.** Gaudi-2's HW fp8 element is
  `float8_e4m3FNUZ` (bias 8, max 240); the dtype name handed to the ops is `float8_e4m3FN`
  (bias 7, max 448). The same e4m3 bits differ by exactly 2^(7-8)=2× between the two.
  `convert_from_int4 → fp8` is asymmetric (writes fnuz-interpreted bits, the next read
  reinterprets at 2× with nothing to cancel) → a systematic ~2× output inflation (the
  historical "fp8_norm ≈ 2.06 × ref_norm"). So: dequant to **bf16** (the proven, only safe
  `convert_from_int4` out_dtype), then the **only** fp8 cast is a **self-consistent**
  round-trip — `convert_to_fp8` packs with 1/s, `fp8_gemm` recovers with s, the 2× cancels
  (exactly the proven `apply_fp8_linear_hpu`/`dynamic_quant` path). `FP8_MAX = 240` (fnuz),
  not 448, even though the cast/gemm dtype is `float8_e4m3fn`.
- **On-device accuracy gate is mandatory** before trusting this op: `test_fused_expert_fp8_op.py`
  flags `|got|/|ref| ≈ 2×` as the fn-vs-fnuz trap, and requires relNorm ≤ the looser fp8 tol.

---

## 3. fp8-KV toggle — **config (orthogonal serving flag, not the kernel).**

New file: `models/glm51ar-fused-fp8kv.env` (based on `glm51ar-lazy.env`, the coherent
single-stream lazy GLM profile; the only add is the fp8-KV flag + INC env).

- **Flag:** `--kv-cache-dtype fp8_inc`. `fp8_inc` is the **only** KV dtype the HPU platform
  accepts (`is_kv_cache_dtype_supported` returns True only for `fp8_inc`,
  `platform_PR1301.py:308-309`). It is INC-driven (`QUANT_CONFIG` JSON is the master switch),
  not a bare `--quantization` flag.
- **INC calibration requirement:** either (a) run with INC's runtime maxabs scaling and **no**
  `QUANT_CONFIG` if the build accepts it (start here, A/B vs bf16 KV), or (b) mint a
  **Gaudi-2** maxabs config on the g2 cards
  (`vllm-hpu-extension/calibration/calibrate_model.sh -o g2` →
  `maxabs_quant_g2.json`) and set `QUANT_CONFIG`. **Never reuse Gaudi-3 scales** (±240 fnuz
  vs ±448 fn → clip); never `scale_method=unit_scale`. The `QUANT_CONFIG` line is commented
  in the env, ready to uncomment.
- **MLA-latent-KV coherence check (GLM & DeepSeek are MLA):** the fp8 KV for MLA goes through
  a **separate** path — `HPUCompressedTensorsKVCacheMethodForMLA`
  (`hpu_compressed_tensors.py:1048`) scales **only** `latent_cache_k` (no separate v_cache;
  the latent absorbs V, expanded per-head by `kv_b_proj` at read). So the fp8 error lands on
  the **compressed latent-K before the up-projection** and is amplified through `kv_b_proj`.
  Gate on the box: generate fp8-KV vs bf16-KV on a fixed prompt and confirm it stays coherent
  (no salad/loops) and matches; confirm the G2 fnuz scale-adjustment in the startup log; and
  with DSA on, re-run the indexer recall/coherence check (the sparse indexer reads the latent).
- **Win framing:** ~2× KV headroom = a **context/memory** lever (longer ctx and/or more
  concurrent sequences), **~0** single-stream tok/s (decode at batch=1 is weight-bandwidth
  bound). Documented as such in the env. For GLM's already-small MLA latent the absolute
  saving is smaller than a dense-KV model's, but real.

---

## Files

- `tests/test_fused_expert_general.py` — generalization verify (RUN ON HPU).
- `tpc_kernel/opbackend/moe_fused_expert_fp8_backend.cpp` — fp8-compute op scaffold.
- `tpc_kernel/opbackend/build_fp8.sh` — CPU-only build for the fp8 `.so`.
- `tpc_kernel/opbackend/test_fused_expert_fp8_op.py` — fp8 accuracy gate (RUN ON HPU, looser tol).
- `models/glm51ar-fused-fp8kv.env` — fp8-KV serving profile.
- `tpc_kernel/opbackend/FUSED_GENERALIZE_FP8_DESIGN.md` — this note.
