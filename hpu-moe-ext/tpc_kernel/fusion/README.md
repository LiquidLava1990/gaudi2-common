# Full-fusion int4 MoE kernel — foundation + SRAM-residency experiment

Goal: int4 weights stream from HBM, dequant **in SRAM**, feed the MME GEMM, and the
bf16 dequanted weight is **never written back to HBM**. This dir holds the runnable
foundation (P0) and the make-or-break fused-graph + HBM profile (P1).

All `.cpp` here are **compiled CPU-only** in the dev image
(`vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest`).
All `.py` and the recipe binary are **run on the HPU box by a human** (never the agent).

---

## What we learned about the GC (decisive for this whole effort)

Inspecting `/usr/lib/habanalabs/libSynapse.so` (the graph compiler + runtime) the
fusion is **not** something we have to coax — the GC already ships the machinery:

- **Two dedicated graph passes** (recovered symbols + source paths):
  - `fuseDequantIntoMoeGemm(HabanaGraph&)`  ←  `graph_compiler/fuse_dequant_into_moe_gemm.cpp`
    *"Fuse dequantization nodes to mme gemm in case of hw aligned scales for MOE"*
  - `fuseDequantIntoOps(HabanaGraph&)`  ←  gated on `QDQ_OPTIMIZATION`
- **TPC-producer-into-MME bundling** (the SRAM-resident path):
  `SRAMSlicingManager::addTpcProducer`, `MantaRayBundlizer::generateMultiMmeWithTpcProducersBundles`,
  `extendSharedInputProducers`, `setTpcBundleSlicingChunk`, `markAllowedForStitching`,
  `MMESlaveBrain::validateNodeToBeStitched`.
- **SRAM placement + contraction-dim slicing**: `allocateTensorInSram`,
  `SlicingStrategy::setInputIsInSRAM`, `GCFG_SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED`,
  driven by `GCFG_ENABLE_PIPELINE_MANAGEMENT` (the "pipeline management" subsystem).
- `gemm` / `batch_gemm` / `masked_batch_gemm` are GC-emitted MME node GUIDs (no TPC
  kernel) — `synNodeCreate(guid="gemm")` with `synGEMMParams{transpose_a, transpose_b}`.

So the experiment is: build the explicit `int4 → [TPC dequant] → bf16 intermediate
(NO persistent section) → [gemm]` graph, compile it, and see whether GC keeps the
intermediate in SRAM (sliced bundle) or spills it to an HBM workspace.

**SRAM residency is NOT a single user flag.** A `synSectionSetPersistent(false)`
intermediate with *no section assigned at all* is a graph-internal tensor; the GC's
pipeline-management passes decide SRAM-vs-HBM and slice the bundle. We rely on that,
and read the verdict from the compile log + on-device HBM counters.

---

## P0 — runnable custom kernel (the foundation)

Two paths ship; both are real.

1. **Resident-GUID path (primary, proven).** `../dequant_intrinsic.py` /
   `../test_p0_dequant_general.py` bind/exercise the pre-built GUID
   `grouped_index_transpose_dequant_2d_with_zp_fwd_u4` via `convert_from_uint4`
   (no build, no `gaudi2.bc`). Prior agents verified it **bit-exact**. P1 reuses
   this exact GUID for the dequant node.

2. **User-ELF perf-lib path (this dir).** `moe_ext_perflib.cpp` is a real perf-lib
   implementing `GetKernelGuids` + `InstantiateTpcKernel` (per
   `tpc_kernel_lib_interface.h`) that registers our **own** compiled dequant ELF
   (`../moe_ext_unpack_dequant_w4.c`, tpc-clang, rc=0) under GUID
   `moe_ext_unpack_dequant_w4_bf16`, discovered by GC via `GC_KERNEL_PATH`.

### Build (CPU-only) + run P0
```bash
# build the user perf-lib .so  (TPC .o + objcopy-embed + g++ shim)
docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
  bash build_perflib.sh
# build the PT2 glue that binds a torch op to the GUID (one dir up)
docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
  bash build.sh

# ---- ON THE HPU BOX ----
cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
export GC_KERNEL_PATH=$PWD/build/libmoe_ext_perflib.so:$GC_KERNEL_PATH
python test_p0_runnable.py        # bit-exact vs convert_from_uint4, parameterized shapes/gs
```
`build_perflib.sh` confirmed **rc=0**; the `.so` exports `GetKernelGuids`,
`InstantiateTpcKernel`, `GetLibVersion`.

---

## P1 — the fused graph + HBM profile (make-or-break)

`fused_moe_recipe.cpp` builds the explicit standalone Synapse recipe:

```
weight_packed int32 [OUT/8, IN]  ┐
weight_scale  bf16  [OUT, GROUPS]├─► [TPC dequant: grouped_..._with_zp_fwd_u4]
zero_point    int32 [OUT/8,GROUPS]┘            │
                                               ▼
                              W_bf16 [OUT, IN]  (NON-persistent; GC placement)
x bf16 [IN, M] ───────────────────────────────►[gemm]──► out bf16 [OUT, M]  (HBM)
```

- The bf16 intermediate has **no section** → GC owns SRAM-vs-HBM.
- **Tiling**: GLM w13 bf16 = 6144×4096×2 = **50.3 MB > 48 MB SRAM** → the full expert
  weight cannot sit in SRAM at once; it **must** be sliced along the contraction dim
  `IN`. We do **not** hand-tile — we build the full producer→consumer edge and let the
  GC `GEMM_COMMON_DIM` slicer pick the chunk (`setTpcBundleSlicingChunk`). The chosen
  chunk + SRAM budget print in the `SRAM_SLICE` log. (Manual contraction tiling is the
  fallback if GC declines to bundle — see "Risk" below.)
- GEMM orientation: with FCD-first geometry, `out{OUT,M} = A{IN,M}·B{OUT,IN}` ⇒
  A=`x`, B=`W_bf16`, contraction=`IN`, **no transpose** (`kTransposeA/B=false`). If
  on-device numerics come out transposed, flip those two flags (single line each).

### Build (CPU-only)
```bash
docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
  bash build_recipe.sh
```
Confirmed **rc=0**, links `libSynapse.so`, **all `syn*` symbols resolve**.

### Run + profile (ON THE HPU BOX)
```bash
cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion

# (a) compile-only sanity: builds+compiles the graph, prints workspace residency hint
FUSED_COMPILE_ONLY=1 ./build/fused_moe_recipe 32 6144 4096 128

# (b) full driver: runs BOTH the standalone recipe and the torch fuse path, profiles HBM
python profile_fusion.py 32 6144 4096 128        # GLM w13
python profile_fusion.py 16 7168 4096 128        # DeepSeek-ish

# correctness (fused == unfused reference, bit-exact bf16):
python test_p1_fusion.py

# deep dive on the standalone graph's SRAM decisions:
GRAPH_VISUALIZATION=1 GRAPH_VISUALIZATION_DIR=./gviz LOG_LEVEL_SRAM_SLICE=2 \
HABANA_LOGS=./logs HABANA_PROFILE=1 \
  ./build/fused_moe_recipe 32 6144 4096 128
```

### How to read the verdict
Per expert, per gemm iteration, the dominant **weight** HBM traffic should be:

| Outcome | weight HBM/iter (GLM w13) | meaning |
|---|---|---|
| **SRAM-resident (WIN)** | **~12 MB** (≈ int4 size) | dequant fed the MME from SRAM; near-max BW reachable |
| **GC spilled (FAIL)** | **≥ 50 MB** (≈ bf16 size) | bf16 round-tripped HBM; fusion failed |

Three independent signals (the script prints all three):
1. **Compile workspace size** (recipe stdout): `≈ bf16 size` ⇒ spilled; `≪ bf16` ⇒ SRAM-resident.
2. **`SRAM_SLICE` log**: shows whether the bundle was created and the slice chunk.
3. **On-device HBM bytes** (`synProfiler` HBM-traffic column / `HBM/DRAM read: N bytes`
   log line): the authoritative number. `profile_fusion.py` asserts
   `weight HBM < 1.5× int4` (PASS) and flags `≥ 0.75× bf16` (FAIL).

---

## Honest read: the single biggest risk to GC keeping the tile in SRAM

**The dequant node is a *transposing* producer, and the GC's MME bundler may not be
able to slice it into the gemm bundle — so the bf16 intermediate spills to an HBM
workspace and we lose.** This is THE risk, and it is structural, not a config typo.

Disassembling `libSynapse.so` the gate is `TPCNode::isAllowedForStitching()` +
`markAllowedForStitching` (modern MantaRay/pipeline-management path) and the legacy
`Bundlizer` whitelist (its reject string is literally *"TPC kernel can be stitched but
it is not in the allowed for stitching list, GUID: {}"*). The modern path does **not**
use a static GUID whitelist — it decides per node by **analyzing the producer's
index-space** to prove a valid common-dim slice exists. Our dequant GUID is
`grouped_index_**transpose**_dequant_2d_with_zp_fwd_u4`: it **transposes** while
dequantizing (packed-u4 `[in, out/8]` → bf16 `[in, out]` with the grouped scale
applied across a *different* axis than the MME contraction). A transposing producer is
precisely the case where the slicer often **cannot** carve a clean contraction-dim
chunk, so it bails to HBM. Ranked failure modes:

1. **Index-space of the transpose-dequant is not common-dim sliceable** ⇒
   `markAllowedForStitching` says no ⇒ HBM. *(most likely; structural.)*
2. **`fuseDequantIntoMoeGemm` only fires for the internal fused-MoE node**, which
   carries a `moeChunkIndex` annotation (`MoeMetadataDb`, `MoeNodesPostProcessing`),
   **not** a bare `gemm` + standalone dequant. So the *dedicated* MoE fuse pass won't
   fire for the standalone recipe — only the generic stitcher can, and (1) gates it.
3. **"HW-aligned scales" precondition** of the MoE fuse pass
   (`isScaleExpBiasHWAligned`, allowed exp-bias 3/7/11/15). GPTQ f16 per-group scales
   are generally **not** HW-aligned ⇒ the dedicated pass skips even on the real model.
4. **SRAM capacity**: even if stitched, *"candidate additional SRAM capacity exceeds
   max capacity"* if the slice chunk is too big ⇒ must slice harder along `IN`.
5. **`ENABLE_PIPELINE_MANAGEMENT` off / eager mode** ⇒ no bundling at all ⇒ always HBM.

### Knobs to try (on the box) before concluding "dead"
```bash
ENABLE_PIPELINE_MANAGEMENT=true            # ensure the bundler runs (default-on in lazy)
ENABLE_TPC_STITCHING_TO_BGEMM=true         # allow TPC producer -> (b)gemm stitching
SRAM_SLICER_GEMM_COMMON_DIM_SLICE_ENABLED=true   # contraction-dim slicing
LOG_LEVEL_SRAM_SLICE=1                      # watch the stitch accept/reject reason
# then re-run ./build/fused_moe_recipe and read which reject string (if any) fires.
```

### If (1)–(3) block it — the deterministic fallback (don't hope; force it)
Stop relying on the cost model. Tile **manually** along the contraction dim `IN` into
chunks whose bf16 size fits SRAM (e.g. `IN_tile` s.t. `IN_tile*OUT*2 ≤ MAX_RMW_SIZE`,
read via `DEVICE_ATTRIBUTE_MAX_RMW_SIZE`), and for each tile emit
`dequant(tile) → bf16 chunk in a NON-persistent RMW SRAM section
(synSectionSetRMW(true)) → gemm(accumulate)`. Assigning the chunk to an RMW section
*forces* SRAM residency by construction instead of leaving it to
`isAllowedForStitching`. Also worth trying: a **non-transposing** dequant GUID
(`grouped_index_transpose_dequant_2d_fwd_u4` still transposes; a plain
`convert_from_uint4_i32` per-channel cast does not) so the producer access-pattern is
trivially sliceable — at the cost of pre-transposing the packed weight on load.

Mitigations already baked in: P1 uses the **resident, already-cost-modeled** GUID (best
shot at the generic stitcher accepting it); shapes are general, so shrink `OUT`/`IN`
to a tile that fits SRAM **unsliced** to confirm the SRAM-residency *mechanism* in
isolation before fighting the slicer; and the driver measures HBM bytes directly, so a
"looks fused but isn't" cannot fool the verdict.
```
