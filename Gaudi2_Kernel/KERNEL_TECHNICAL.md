# Gaudi 2 Fused INT4 MoE Kernel — Technical Reference (code-driven walkthrough)

The granular companion to [`KERNEL_ARCHITECTURE.md`](KERNEL_ARCHITECTURE.md). This is the
"how to reproduce / extend" reference: every node in the `AddNode` chain (GUID + params
struct + dtype + shape, in **PyTorch row-major**), the OpBackend / KernelRegistry /
TORCH_LIBRARY registration triple, the eager↔lazy frontend split, the CPU-only build, and a
**"gates cracked"** section documenting each non-obvious fix with its root cause.

Built against **HabanaAI/gaudi-pytorch-bridge tag `v1.24.0-1007`** (==
`habana-torch-plugin 1.24.0.1007`, torch 2.10, `_GLIBCXX_USE_CXX11_ABI=1`, Apache-2.0). All
citations are to files under `tpc_kernel/opbackend/` (verbatim in
`publish/gaudi-fused-moe-kernel/`). Geometry convention throughout: **NodeOutputAttr.sizes is
a PyTorch ROW-MAJOR shape** (last dim == the Synapse FCD); the bridge reverses to FCD order
internally. This is load-bearing — see Gate 4.

---

## 1. File map

| File | Role |
|---|---|
| `moe_fused_expert_backend_shared.h` | The shared, frontend-free backend: `MoeFusedExpert : OpBackend` (`AddNode` chain) + `MoeFusedExpertMeta`. **The recipe.** |
| `moe_fused_expert_backend.cpp` | **Eager** frontend (`habana::eager::EagerOp`) + the TORCH_LIBRARY / KernelRegistry registration. (Carries its own copy of the recipe — the proven original; the `_shared.h` was factored out of it.) |
| `moe_fused_expert_backend_lazy.cpp` | **Lazy** frontend (`habana_lazy::LazyOp`) over the same shared recipe → captures into the decode HPU-graph. |
| `moe_fused_expert_emit.h` | `emit_one_expert(...)` — the per-expert chain factored byte-for-byte, so the K-expert op emits the *exact* same nodes. |
| `moe_fused_expert_routed_backend_shared.h` | `MoeFusedMoE : OpBackend` — the **K-expert single-recipe** op (calls `emit_one_expert` K times + an `add_fwd_bf16` left-fold). |
| `moe_fused_expert_routed_backend.cpp` / `_lazy.cpp` | Eager / lazy frontends for the K-expert op. |
| `moe_fused_expert_fp8_backend.cpp` | fp8-compute variant (separate `hpu_moe_ext_fp8` library; scaffold gated by `BUILD_FP8_NODES`). |
| `build.sh` / `build_lazy.sh` / `build_routed.sh` / `build_routed_lazy.sh` / `build_fp8.sh` | CPU-only host g++ builds (one per `.so`). |
| `../../hpu_moe_ext/fused_expert_routed.py` | Python K-expert wrapper (gather/route/remap + the single-recipe route). |
| `../../hpu_moe_ext/moe_wna16_hpu.py` | vLLM plugin integration (`HPU_MOE_FUSED_OP`, `_fused_moe_routed`, `.so`-by-mode loader). |

## 2. The op schema and tensor contract

Single-expert op (`moe_fused_expert_backend.cpp:382-387`):

```
hpu_moe_ext::fused_expert(
    Tensor x, Tensor w13_packed, Tensor w13_scale, Tensor w13_zp,
    Tensor w2_packed, Tensor w2_scale, Tensor w2_zp,
    Tensor router_weight) -> Tensor
```

K-expert single-recipe op (`moe_fused_expert_routed_backend.cpp:94-100`,
`TORCH_LIBRARY_FRAGMENT` so it co-loads):

```
hpu_moe_ext::fused_moe(
    Tensor x,
    Tensor[] w13_packed, Tensor[] w13_scale, Tensor[] w13_zp,
    Tensor[] w2_packed,  Tensor[] w2_scale,  Tensor[] w2_zp,
    Tensor[] router_weights) -> Tensor
```

Per-expert tensor layout (PyTorch row-major; `M`=tokens, `H`=hidden, `I`=intermediate,
`GS`=group size, `2I`=gate||up width):

```
  x          bf16  [M, H]
  w13_packed i32   [H, 2I/8]      8 int4 nibbles packed little-endian / int32 along OUT
  w13_scale  bf16  [H/GS, 2I]
  w13_zp     i32   [H/GS, 2I/8]   packed like the weight
  w2_packed  i32   [I, H/8]
  w2_scale   bf16  [I/GS, H]
  w2_zp      i32   [I/GS, H/8]
  router_weight bf16 [1, 1]       (per expert; folded by the final mult at M=1)
  -> out     bf16  [M, H]
```

Output meta (`MoeFusedExpertMeta`, `..._shared.h:63-69`): `out.shape == x.shape`,
`out.dtype == x.dtype` (bf16). For the K-expert op `MoeFusedMoEMeta` reads `M` from `x` so
it is correct for any token count.

## 3. The `AddNode` node chain — node by node

From `MoeFusedExpert::AddNode` (`moe_fused_expert_backend_shared.h:89-298`); the K-expert op
emits the identical chain per expert via `emit_one_expert` (`moe_fused_expert_emit.h:87-218`).
Dims are derived once: `M=x.size(0)`, `H=x.size(1)`, `OUT13=2I=w13_scale.size(1)`, `I=OUT13/2`
(`:104-114`).

Every node is `OpBackend::BuildNode(this, graph, {guid, {syn_inputs}, {output_attrs}, &params,
sizeof(params)})`. Non-final outputs are **non-persistent** `synapse_helpers::tensor`; the
next node consumes `.get()`. `final_result_index = 0` on the last node ties it to `syn_out(0)`.

| # | GUID | inputs | output (PyTorch shape, dtype) | params struct | persistent? |
|---|---|---|---|---|---|
| 1 | `reinterpret_cast` | `w13_packed` | `[H, 2I]` **uint4** (`syn_type_uint4` in the out-attr; last dim ×8) | — (out-attr `DATA_TENSOR, syn_type_uint4`) | no |
| 2 | `reinterpret_cast` | `w13_zp` | `[H/GS, 2I]` uint4 | — | no |
| 3 | `dequantize_4_bit_bf16` | `{W13_u4, w13_scale, ZP13_u4}` | `[H, 2I]` **bf16** (= packed shape, last dim ×8) | `ns_CastKernel::ParamsV3{}` (zero-init) | **no** ░ |
| 4 | `gemm` | `{x, W13_bf16}` | `[M, 2I]` bf16 (`x[M,H] · W13[H,2I]`) | `synGEMMParams{false,false}` | no |
| 5 | `slice` | `{gu}` begin=0 end=I | `[M, I]` bf16 (gate) | `synSliceParamsV2` (full per-dim) | no |
| 6 | `slice` | `{gu}` begin=I end=2I | `[M, I]` bf16 (up) | `synSliceParamsV2` | no |
| 7 | `sigmoid_fwd_bf16` | `{gate}` | `[M, I]` bf16 | — | no |
| 8 | `mult_fwd_bf16` | `{gate, sigmoid}` | `[M, I]` bf16 (SiLU) | — | no |
| 9 | `mult_fwd_bf16` | `{silu, up}` | `[M, I]` bf16 (`h`) | — | no |
| 10 | `reinterpret_cast` | `w2_packed` | `[I, H]` uint4 | — | no |
| 11 | `reinterpret_cast` | `w2_zp` | `[I/GS, H]` uint4 | — | no |
| 12 | `dequantize_4_bit_bf16` | `{W2_u4, w2_scale, ZP2_u4}` | `[I, H]` **bf16** | `ns_CastKernel::ParamsV3{}` | **no** ░ |
| 13 | `gemm` | `{h, W2_bf16}` | `[M, H]` bf16 (`h[M,I] · W2[I,H]`) | `synGEMMParams{false,false}` | no |
| 14 | `mult_fwd_bf16` | `{y, router_weight}` | `[M, H]` bf16 (`[1,1]` broadcast) | — | **yes** (`final_idx=0`) |

░ = the SRAM-resident dequant→MME bundle (nodes 3→4 and 12→13). Because outputs 3 and 12
carry **no persistent section**, the GC's `fuseDequantIntoMoeGemm` + SRAM slicer bundle each
into its consuming MME; the bf16 weight is produced and consumed in SRAM.

Notes on specific node constructions:

- **GUID precision suffix.** `get_guid_with_precision("dequantize_4_bit"sv, bf16)` →
  `dequantize_4_bit_bf16`; same helper yields `sigmoid_fwd_bf16`, `mult_fwd_bf16`,
  `add_fwd_bf16`. `"gemm"`, `"slice"`, `"reinterpret_cast"` are precision-agnostic and passed
  literally.
- **reinterpret_cast out-attr** (`..._shared.h:152-160`): the unpacked PyTorch shape is the
  packed shape with the **last dim ×8**; the out-attr is `{unpacked, at::kInt, std::nullopt,
  DATA_TENSOR, syn_type_uint4}` — the `syn_type_uint4` carries the uint4 *view* of the int32
  bytes. (Verbatim `convert_from_int4_gen.cpp` ReinterpretCast.)
- **dequant inputs** (`:171-185`): `{unpacked_weight_u4, scale, unpacked_zp_u4}`. The zp is
  reinterpreted to uint4 **because it is packed like the weight** (Gate 2). `out_shape` =
  packed PyTorch shape, last dim ×8. `params` = a **zero-init** `ns_CastKernel::ParamsV3`
  (mode stays `CAST_NO_CLIP` for bf16 out — no fp8 clipping). No `group_index` is passed: the
  cguid's SIF infers grouped vs non-grouped from `scale.size(0)` vs `input.size(0)`.
- **gemm** (`:197-204`): `synGEMMParams{false, false}` (no transpose); PyTorch-order
  `self[M,K] · other[K,N] → [M,N]`, exactly the bridge's `matmul.cpp`. **The shape vector
  must be a NAMED local** that outlives the call (Gate 9).
- **slice** (`:220-246`): guid `"slice"` + `synSliceParamsV2` (Gate 3). Every PyTorch dim `i`
  gets an entry; `axes[i] = rank - i - 1` (TPC-reversed, == `get_dim_in_tpc_order`); the
  sliced dim (dim 1, the `2I` feature axis) carries `[begin, end)` step 1, every other dim is
  the full range `[0, size)`. Arrays sized `HABANA_DIM_MAX` (25), zero-filled, steps=1.
- **SiLU** (`:254-269`): the sigmoid+mult decomposition `gate · sigmoid(gate) · up` (not the
  `silu` cguid), one NAMED `im_shape{M,I}` shared across all three nodes.
- **router fold** (`:291-295`): `mult_fwd_bf16(y, router_weight[1,1])` — the `[1,1]` broadcasts
  over `[M,H]`, no explicit broadcast guid. This node is the op result.

### K-expert accumulation

`MoeFusedMoE::AddNode` (`moe_fused_expert_routed_backend_shared.h:128-226`): derives `M,H` from
`x`, `OUT13/I` from `w13_scale[0]`; checks all 7 expert lists have equal length `K` (LOUD on
mismatch). Then:

- `K==1`: `emit_one_expert(..., is_final_result=true)` — its `* router_weight` node is the op
  result (no add).
- `K>=2`: each `emit_one_expert` is non-final (graph-internal `y_k`); a **left-fold**
  `acc = add_fwd_bf16(acc, y_k)` for `k=1..K-1`; only the last `add` ties to `syn_out(0)`. A
  left fold (not a balanced tree) because the K subgraphs are independent — the GC schedules
  them in parallel regardless, and the K-1 elementwise `[M,H]` adds are negligible vs the 2K
  MMEs.

**TensorList wiring** (`..._routed_backend_shared.h:136-143`):
`StackGetter::getNextInput<std::vector<TensorsPair>>()` returns K `TensorsPair` whose `.syn_t`
are already the **consecutive `syn_in(...)`** for that list's elements (the StackGetter
increments the syn index once per element). So expert `k`'s tensors are just `w13_packed[k].syn_t`
etc. — clean 2D shapes, **no dim-0 slice, no squeeze, no new geometry**; `emit_one_expert`
consumes them exactly as the single-expert op consumes its per-tensor inputs.

## 4. Registration — the three-part triple

Per `.so` (modeled on `full_gen.cpp` + `kernels_register.cpp`):

```cpp
// 1. schema -> makes torch.ops.hpu_moe_ext.fused_expert exist
TORCH_LIBRARY(hpu_moe_ext, m) { m.def("fused_expert(...) -> Tensor"); }

// 2. HPU dispatch -> routes the call through the frontend (eager EagerOp / lazy LazyOp)
TORCH_LIBRARY_IMPL(hpu_moe_ext, HPU, m) { m.impl("fused_expert", habana::fused_expert_hpu); }

// 3. backend factory -> binds the schema name to the OpBackend the lowering instantiates
static const auto& reg = habana::KernelRegistry().REGISTER_HPU_BACKEND(
    "hpu_moe_ext::fused_expert", habana::MoeFusedExpert);
```

`REGISTER_HPU_BACKEND(op, class)` expands to
`add(op, [](int dev, ScalarType t){ return std::make_shared<class>(dev, t); })`. At
graph-build time the schema name routes the lowering to the OpBackend via
`KernelRegistry().get(opname)` (`backend/kernel/hpu_habana_launch_op_pt.cpp`).

**`TORCH_LIBRARY` vs `TORCH_LIBRARY_FRAGMENT`:** the single-expert `.so` does
`TORCH_LIBRARY(hpu_moe_ext){ fused_expert }`. The K-expert `.so` uses
`TORCH_LIBRARY_FRAGMENT(hpu_moe_ext){ fused_moe }` so that if **both** `.so`s load in one
process a second `TORCH_LIBRARY(hpu_moe_ext)` does not abort with
"library already defined" (`moe_fused_expert_routed_backend.cpp:24-32`). FRAGMENT is the
bridge's own idiom for spreading one library across `.so`s; loaded alone it still creates the
library fine.

## 5. The eager / lazy frontend split

The frontend is the only difference between the two `.so`s; the backend recipe is included
byte-for-byte from `*_shared.h`.

**Eager** (`moe_fused_expert_backend.cpp:355-369`): build an `EagerOp` from
`(schema_name, inputs, out_shapes)`, set the scalar type, `.call()`:

```cpp
habana::eager::EagerOp<at::Tensor> hpu_op{
    "hpu_moe_ext::fused_expert", {x, w13_packed, ...}, std::move(out_shapes)};
hpu_op.set_scalar_types({c10::ScalarType::BFloat16});
return hpu_op.call();
```

`.call()` builds a one-node JIT graph whose schema name routes the lowering to the OpBackend,
then launches it. This is the public path the bridge's own user-custom-op frontend uses
(`habana_eager/user_custom_op_frontend.cpp`).

**Lazy** (`moe_fused_expert_backend_lazy.cpp:81-95`): build a `LazyOp`, set the output-meta
fn, dispatch through the acc-thread machinery — modeled verbatim on the bridge's
`lazy_custom_ops.cpp` (the `mixture_of_experts` lazy impl):

```cpp
LazyOp<at::Tensor> hpu_op{"hpu_moe_ext::fused_expert", {x, w13_packed, ...}};
hpu_op.SetOutputMetaFn(MoeFusedExpertMeta);
RUN_MAYBE_WITH_ACC_THREAD(fused_expert, hpu_op);
```

`LazyOp::call()/get_result()` **emit into the lazy IR the HPU-graph capture engine records** —
which is the entire reason the lazy TU exists (the op is captured into the decode graph and
replayed). The `habana_lazy::*` symbols (`flush_op`, `AccThread::Get`, the `LazyOp` templates)
are left **undefined** in the `.so` and resolve at runtime from the loaded lazy plugin (see the
build, §6). The TU mirrors `lazy_custom_ops.cpp`'s file-scope `using namespace habana;` so the
`RUN_MAYBE_WITH_ACC_THREAD` macro's `scheduleAccTask` and the `MoeFusedExpertMeta` both resolve.

## 6. The build (CPU-only, host x86-64)

Every `.so` is built by a sibling of `build.sh` (they differ only in `SRC`/`OUT` and, for the
lazy builds, that discovery/symbol-checks run under `PT_HPU_LAZY_MODE=1`). The build is
**CPU-only** — it compiles C++ against the cloned version-exact bridge headers + torch +
synapse headers and **links the shipped backend lib**; it never touches a device.

```
docker run --rm -v <repo>/tpc_kernel:/work -w /work/opbackend vllm-gaudi:next bash build.sh
```

Key build facts (`build.sh`, `:84-160`):

- **Link ONLY `libhabana_pytorch_backend.so`** (`-l${BACKEND_LIB_NAME}`, picking the
  `.upstream.so` variant if present). It carries the symbols the TU references at link time:
  `OpBackend::BuildNode`, the `KernelRegistry()`/`RegisterKernel` singleton, `HabanaOperator`.
- **NEVER link the plugin libs** (`libhabana_pytorch2_plugin` / `libhabana_pytorch_plugin`) —
  this is Gate 5.
- **`-Wl,--allow-shlib-undefined`**: the frontend symbols (eager `EagerOpBase::run` /
  `validate_inputs` / `HbEagerTensorPool::get_backend_tensor`, or lazy `flush_op` /
  `AccThread::Get` / the `LazyOp` templates) are left **undefined** and resolve at runtime from
  the plugin that `import habana_frameworks.torch` already loaded (eager plugin in eager mode,
  lazy plugin under `PT_HPU_LAZY_MODE=1`). Safe because our `.so` is only ever loaded
  **after** the import.
- **`-D_GLIBCXX_USE_CXX11_ABI=1`** (forced; this tag's torch is ABI=1), `-std=c++17`,
  `-DHABANA_GAUDI2`.
- **Build-only prereqs generated/installed in-image:** `libabsl-dev` headers (a transitive dep
  of `op_backend.h`; only the header types must match — the prebuilt backend `.so` provides the
  symbols) and the **one** generated header `generated/env_flags/env_flags_generated.h`
  (produced from the in-repo YAML by `scripts/generate_env.py`).
- **`FMT_HEADER_ONLY`** is set in every TU (Gate 10).
- After linking, the script **asserts the link is sound**: `ldd -r` must show *no* undefined
  `OpBackend`/`BuildNode`/`KernelRegistry` symbols (those are the ones the linked backend lib
  must provide); the frontend symbols being undefined is expected and the script confirms the
  canonical one (e.g. `EagerOpBase::run` / `flush_op`) is a global dynamic symbol in the
  plugin that will resolve it at load.

The bridge clone (`./bridge`, gitignored) and the `build/` outputs (`.so`s) are **not**
shipped in the publish copy — clone the bridge yourself (see README) and run the build.

## 7. Gates cracked — non-obvious fixes + root causes

Each of these was a node-level or load-level gate cracked via Synapse logging (the multi-step
on-device bring-up). **Count: 10** (8 are the headline node/build gates; +2 are the
correctness/serving traps for the K-expert path). They are the difference between "compiles"
and "runs correctly + serves".

**Gate 1 — the dequant GUID is `dequantize_4_bit_bf16`, not the kernel ELF name.**
The resident dequant TPC kernel in `libtpc_kernels.so` is named
`grouped_index_transpose_dequant_2d_with_zp_fwd_u4`, but **that name does not exist as a graph
node**. The graph-level node is the **complex GUID** `dequantize_4_bit_bf16` (a
`reinterpret_cast` precedes it). Passing the ELF kernel name → `synNodeCreate` `synStatus 26`
(GLUE err 6). Use `get_guid_with_precision("dequantize_4_bit", bf16)`.

**Gate 2 — the zero-point must be reinterpreted to uint4 (matching the weight dtype).**
The zp is packed like the weight (8 nibbles/int32). The dequant complex-GUID **rejects an
int32 zp** (`COMPLEXGUID err 400`). It must be `reinterpret_cast`'d to `syn_type_uint4` first,
exactly like the weight (`..._shared.h:171-175`).

**Gate 3 — slice is guid `"slice"` + `synSliceParamsV2`, not `slice_axis` + V1 params.**
The eager `OpBackend::BuildNode` path **rejects** the `slice_axis` guid + `synSliceAxisParams`
(V1) at `synNodeCreateWithId`. The working form is the bridge's idiomatic forward `"slice"`
guid + the full strided `synSliceParamsV2` (per-dim axes/starts/ends/steps, `HABANA_DIM_MAX`),
filled like `SliceBackward::AddNode` (`..._shared.h:206-246`).

**Gate 4 — PyTorch-ROW-MAJOR geometry everywhere (not Synapse FCD) — the node-6 SIF
null-deref.** `OpBackend::BuildNode`'s `NodeOutputAttr.sizes` is a **PyTorch row-major** shape:
`BuildNode → GetProxyTensor(dtype, sizes)` calls `set_sizes_contiguous(sizes)`
(`op_backend.cpp:75,888`) and the bridge reverses to FCD internally. The dequant complex-GUID's
SIF runs in this **same** PyTorch-order world (its meta = `GetUnpackedShape(input)`,
`ValidateShapeGrouped` checks `input.size(0)%scale.size(0)==0 && unpacked[1]==scale.size(1)`).
The earlier port reused the **Synapse FCD-order** shapes (transposed, `sizes[0]==FCD`) — e.g.
told the dequant its W13 output was `[2I,H]` instead of `[H,2I]`. That transpose desynced the
SIF's dim arithmetic (`unpacked[1]` no longer `== scale.size(1)`) → the cguid SIF dereferenced
a mis-indexed dim/ptr → `GLUE_SIF_NULL_PTR` at `synGraphInferShapes` on node 6. **Fix:** build
**every** node in PyTorch row-major, mirroring the trusted `_ref_expert()` and the bridge's own
`ConvertFromInt4` (PyTorch-order) + `Matmul` (`self[M,K]·other[K,N]→[M,N]`). The whole
`AddNode` geometry block is this fix (`..._shared.h:118-147`).

**Gate 5 — link only the backend lib + `--allow-shlib-undefined` (the double-registration
abort).** Both plugin libs are compiled from `basic_kernels.cpp`, which carries the static
initializer `REGISTER_HPU_BACKEND("hpu::habana_d2d_memcpy_other", MemCopyOperator)` (and the
`TORCH_LIBRARY(hpu)` op-schema defs, incl. `hpu::habana_bernoulli`).
`import habana_frameworks.torch` already runs those initializers and populates the
process-global `KernelRegistry()`. If our `.so` puts a plugin in its `DT_NEEDED` (which
`-l<plugin>` + `--no-as-needed` did), loading our `.so` **re-runs** their constructors →
`KernelRegistry().add("hpu::habana_d2d_memcpy_other")` a second time → `RegisterKernel::add`
throws (`habana_operator.h:799`); equivalently the `:1.24` image throws
`hpu::habana_bernoulli is already registered`. **Fix:** link **only**
`libhabana_pytorch_backend.so` (it does **not** carry those registrars) and leave the frontend
symbols undefined, resolved at runtime from the already-loaded plugin (`build.sh:90-112`).

**Gate 6 — EagerOp vs LazyOp by image; `:next` vs `:1.24` image specificity.**
`vllm-gaudi:next` is the pytorch-**upstream/eager** build: the **lazy** plugin is not loaded
in-process, so a `LazyOp` TU fails to load (its `mul_out_hpu_lazy` / `validate_inputs` are
unresolved) → the eager `.cpp` uses `EagerOp`. Under `PT_HPU_LAZY_MODE=1` it is the **opposite**
(lazy plugin loaded, eager symbols absent) → the lazy `.cpp` uses `LazyOp`. The two `.so`s are
**mutually exclusive by mode**; the plugin picks by `htorch.is_lazy()`
(`moe_wna16_hpu.py:106-109`). The `.so` is also **image-specific**: it loads cleanly on
`:next`; on `:1.24` it hits the Gate-5 double-reg (`hpu::habana_bernoulli already registered`)
even when rebuilt on `:1.24` — a `:1.24` lib-registration conflict, not a build mismatch.

**Gate 7 — the named-vector lifetime rule (non-owning `IntArrayRef` → dangling geometry).**
`NodeOutputAttr.sizes` is a **non-owning** `at::IntArrayRef`. Every shape passed to
`BuildNode` **must be a NAMED vector that outlives the call** (`gu_shape`, `im_shape`,
`mh_shape`, the slice `out_shape`). A temporary rvalue (e.g. `{{ {M,OUT13}, bf16, ... }}`
inline) would dangle → garbage geometry. This is the bridge's own pattern; it is called out at
every `BuildNode` in the chain (`..._shared.h:196-197, 238-239, 277-278`).

**Gate 8 — `FMT_HEADER_ONLY` (two undefined fmt::v12 symbols).**
torch bundles fmtlib v12; two compiled `fmt::v12` symbols the TU pulls via torch's fmt headers
(`assert_fail`, `is_printable`) are **not exported** by any shipped torch/habana `.so` (hidden
visibility). Without `#define FMT_HEADER_ONLY` the `.so` has 2 undefined fmt symbols and fails
to load on the box. Set it (guarded against the habana logging header's own define)
(`..._shared.h:30-32`).

**Gate 9 — the K-expert storage-offset garbage (relNorm 1e+11).**
The op's **first node on every weight/zp input is a `reinterpret_cast`** — the most
storage-offset-sensitive consumer in the codebase. The K-expert wrapper feeds per-expert
`index_select` slices (`w13p[k]` etc.); for `k>0` those carry a **nonzero `storage_offset`**
(`= k·stride[0]` into the gather buffer). A live offset view makes the cast read from the
**wrong base** (effectively expert `active[0]`'s — or out-of-buffer — bytes) → a real expert
dequants garbage (the serve's maxAbs~200 / relNorm~1e11), while a no-local-expert rank
trivially returns 0 (so verify-relNorm looked half-OK and the synthetic offset-0 tests passed).
**Fix:** in `fused_expert_routed.py` clone every per-expert slice `.contiguous().clone()`
**unconditionally** (eager *and* lazy) so every op input is `storage_offset==0`, and pin
scale/router to bf16 (`fused_expert_routed.py:218-235`). This is the same invariant the
monolithic path honors by `.clone()`-ing every per-expert slice at load. The static INT4
packing was never wrong (the plugin repack == auto_gptq, zp=8 confirmed).

**Gate 10 — the `free_inplace` capture FATAL → the single-recipe route.**
Under `wrap_in_hpu_graph` (lazy decode), the per-expert python loop clones each expert
**in-graph**; with vllm-gaudi's lazy default `disable_tensor_cache=True`,
`mark_user_outputs(..., free_inplace=True)` **frees** those in-place/intermediate buffers after
capture, so at `replayV3` the fused op's per-expert `hpu__input` points at **freed storage** →
`PT_LAZY ValidateSyncInputTensors tensor_data is empty ... Strided Params Has Value: 0` (a
serve FATAL on the first INT4 MoE layer). The single-EXPERT op captures fine (one bare call on
persistent external tensors, no in-graph clone); the offending new variable is the wrapper's
**python-side per-expert clone inside the captured region**. **Fix (option c):** under lazy +
M=1, route the whole K-expert MoE through the **single-recipe C++ op** `hpu_moe_ext::fused_moe`
(TensorList) — it slices/dequants/sums **on-device** inside one recipe whose inputs are the
persistent stacked weights, so there is **no python intermediate to free**
(`fused_expert_routed.py:248-275`; the K-expert capture test passes).

## 8. History — why OpBackend (the architecture wall it cleared)

Two earlier approaches were tried and abandoned (recorded so they are not retried):

- **Graph-fusion spike** (chain public `torch.ops.hpu` ops and hope the GC fuses):
  `convert_from_uint4` is an **opaque** op the GC will not auto-fuse; chaining torch ops always
  round-trips bf16 → no win (correct only *with* a `mark_step` that re-materializes to HBM —
  i.e. the floor we are removing).
- **Side-band `synLaunch`** (a torch op drives raw `synGraphCompile`/`synLaunch`):
  `UserCustomOpDescriptor` is single-GUID (can't hold the multi-node graph); and even with a
  hand-built graph, torch HPU `tensor.data_ptr()` (e.g. `0xc000000`) is **not a Synapse
  launchable VA** (valid range `0x10016…`) → `synFailedSectionValidation`. The two address
  spaces differ; raw-`data_ptr` binding cannot work.

**OpBackend clears both walls:** the op is emitted *inside the bridge's own lowering*, so the
bridge → GC **owns the recipe** (non-persistent intermediates → SRAM bundling) **and maps
tensors itself** (no `data_ptr`/section-validation wall), and the recipe is durable across
HPU-graph capture / lazy replay. The ABI is open-source at tag `v1.24.0-1007` (version-exact to
the installed `habana-torch-plugin 1.24.0.1007`), so the whole thing is buildable from public
headers.

## 9. vLLM wiring (where the op plugs in)

`hpu_moe_ext/moe_wna16_hpu.py` integrates the op behind env flags:

- **`HPU_MOE_FUSED_OP=1`** — in the routed fast path, replace the stock
  `convert_from_uint4`-list → `mark_step` → `mixture_of_experts` tail (the bandwidth-bound bf16
  round-trip) with the fused op, driven by `fused_moe_routed(...)`. Same eligibility gate as the
  stock routed path: **GPTQ-symmetric INT4 only** (`has_zp==False`), **`activation=="silu"`**,
  and **`tokens*top_k <= HPU_MOE_ROUTED_K`** (which guarantees ≤ K distinct local experts → no
  host sync, no dynamic shape). AWQ per-expert-zp or non-silu → returns `None` → falls back to
  the stock path (never silently to a worse path) (`moe_wna16_hpu.py:694-767`).
- **`HPU_MOE_ROUTED_K=<K>`** — the static gather width (e.g. the model `top_k`); the fused path
  lives **inside** this gate (a loud warning fires if `HPU_MOE_FUSED_OP=1` but `ROUTED_K<=0`).
- **`HPU_MOE_FUSED_VERIFY=1`** — run **both** the fused op and the stock
  `convert_from_uint4`+`mixture_of_experts` every call and assert `relNorm < 1e-2`; LOUD +
  **stay-coherent** (return the stock result) on mismatch (`moe_wna16_hpu.py:845-874`).
- **`.so` by mode** — `_fused_so_path()` picks `libmoe_fused_expert_lazy.so` under lazy else
  `libmoe_fused_expert.so`; loaded **once, lazily**, the first time the fused path runs (never
  at import). LOUD `RuntimeError` if the flag is on but the `.so` is missing / fails to register
  (`moe_wna16_hpu.py:106-143`).

The **stored-layout → op-contract mapping** is a no-op transpose by construction
(`moe_wna16_hpu.py:701-767`): the plugin's `_repack_weight` already produces `[E, in, out//8]`
and `w*_scales.transpose(1,2)` already produces `[E, groups, out]` — exactly the op's per-expert
PyTorch-row-major shapes (both the fused op's internal `dequantize_4_bit_bf16` and the stock
`convert_from_uint4` are the *same* intrinsic, proven byte-identical in `test_fused_expert_op.py`).
The one adaptation is the GPTQ-symmetric shared zero (a single 2D tensor) is broadcast to
`[E, groups, out//8]` and `.contiguous()`'d so the wrapper's `index_select(0, active)` reads a
real per-expert stride.

## 10. Tests (run on the HPU box; CPU import cannot exercise the graph)

| Test | What it validates | Gate |
|---|---|---|
| `tpc_kernel/opbackend/test_fused_expert_op.py` | single-expert op vs `convert_from_uint4` ref; best-effort HPU-graph capture; SRAM hint | relNorm < 1e-2 |
| `tests/test_fused_expert_general.py` | the 5-profile shape/group matrix | 5/5 relNorm < 1e-2 |
| `tests/test_fused_moe_routed.py` | the Python K-expert wrapper (gather/route/remap + accumulate) vs per-expert ref | relNorm < 1e-2 |
| `tests/test_fused_moe_single_recipe.py` | the K-expert **single-recipe** op vs an independent per-expert oracle; **lazy capture** of the K=8 recipe | relNorm < 1e-2 |
| `tests/test_fused_op_realweights.py` | one **real** GLM expert from the checkpoint, both directly and through `fused_moe_routed` — reproduces the Gate-9 garbage before the fix, PASS after | relNorm < 1e-2, eager + lazy |
| `tests/test_fused_op_capture.py` | HPU-graph capture of the routed path | capture replay matches |
| `tpc_kernel/opbackend/test_fused_expert_fp8_op.py` | the fp8 variant (looser tol; double-quant floor documented) | relNorm < 8e-2 |

The reference oracle is always the trusted `torch.ops.hpu.convert_from_uint4` + the MoE math
(`x@W13 → slice → SiLU → @W2 → ·router`), so a bug in the op cannot be masked by a matching bug
in the oracle.
