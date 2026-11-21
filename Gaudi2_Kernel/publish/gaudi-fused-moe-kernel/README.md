# gaudi-fused-moe-kernel

A **fused INT4 mixture-of-experts kernel for Intel Gaudi 2** — a multi-node, SRAM-resident,
lazy-captured `OpBackend` kernel (no hand-written TPC ISA) that emits the entire per-expert
INT4 forward as one fused graph so the dequantized bf16 weight never round-trips HBM.

> ⚠️ **Performance verdict — read before assuming this is a speedup.** This kernel is
> **correct, general, stable, and serves Qwen-397B end-to-end** — a real engineering
> achievement — **but it is NOT a MoE speedup.** The stock Habana `mixture_of_experts`
> intrinsic is faster in *both* regimes (single-stream ~30 vs ~23 tok/s; throughput ~126 vs
> ~68 at the same config). **Do not enable the fused MoE op for speed.** The durable win is the
> reusable `OpBackend` *infrastructure*, whose real payoff is kernels with **no stock
> equivalent** — the sibling **DSA sparse-attention kernel** (op-level validated on-card; the
> 1M-context enabler). See [Performance](#performance) and the bundled `KERNEL_ARCHITECTURE.md`
> §9 / `PROJECT_FINDINGS.md`.

Stock INT4 MoE on Gaudi dequantizes **every expert** to bf16 and **writes the full bf16
weight to HBM** (because the only `mixture_of_experts` intrinsic is bf16-only), then runs a
separate GEMM op. At single-stream decode that round-trip is, on paper, a **~99 µs/token
bandwidth floor** — roughly 50× the bandwidth the GEMM actually needs. **This was the design
motivation, and it does not hold against the real stock op:** `mixture_of_experts` is an
internally-efficient batched/grouped intrinsic, and at batch=1 the whole MoE compute is only
~0.16 ms of a ~127 ms decode step (the step is ~90% host/collective overhead), so the path was
never weight-bandwidth-bound in the first place. The fused kernel eliminates the round-trip by
construction (verified on-device), but that does not make it faster than stock — see
[Performance](#performance).

This kernel emits the **entire per-expert forward as one fused, SRAM-resident graph** —
`reinterpret int32→uint4 → dequant W13 → MME GEMM → slice gate/up → SiLU → dequant W2 → MME
GEMM → × router_weight` — captured once and replayed per token. The dequantized bf16 weight
**never touches HBM**: it is produced in SRAM and consumed in SRAM by the matrix engine. No
hand-written TPC ISA — every node is a stock Habana bridge GUID; the contribution is the
graph and its (PyTorch-row-major) geometry, registered as a `habana::OpBackend` op so the
bridge owns the launch and tensor mapping.

It is **general** (all dims derived from tensor shapes — validated across GLM / DeepSeek /
Nemotron / MiniMax / odd-width) and ships **eager + lazy** builds (lazy captures into the
decode HPU-graph), plus a **K-expert single-recipe** op (all K routed experts + the weighted
sum in ONE captured recipe / ONE launch) and an **fp8-compute** toggle.

## Validated results

Hardware-validated on `vllm-gaudi:next` (relNorm = ‖got − ref‖ / ‖ref‖ vs the trusted
`torch.ops.hpu.convert_from_uint4` + MoE-math reference):

| What | Metric | Value |
|---|---|---|
| Single-expert op | relNorm | **~4.2e-3** |
| K-expert Python wrapper | relNorm | ~5.7e-3 |
| K-expert single-recipe op | relNorm + lazy capture | **~5.9e-3** (K=8 in ONE captured graph) |
| Real GLM-5.1 expert (from checkpoint) | relNorm, eager + lazy | **~4e-3** |
| Generalization | profiles passing | **5/5** (GLM 6144 · DeepSeek 7168 · Nemotron 8192 · MiniMax GS=64 · odd 5120/1536 GS=32) |
| SRAM residency | bf16 weight in DRAM / workspace | none in graph dump / **0 MB** (HBM round-trip eliminated) |
| fp8-compute | relNorm floor (INT4→fp8 double-quant) | ~6.4e-2 — a memory/throughput option, **not** a single-stream accuracy win |

These are **correctness, generality, and capture** numbers — all green; they establish the
kernel works, not that it is fast. The kernel also serves **Qwen-397B** end-to-end (coherent +
stable across multi-bucket long generation); the GLM end-to-end serve is gated on pre-existing
GLM serve-env issues separate from the kernel (validated on real GLM weights in isolation,
~4e-3). **For speed, see [Performance](#performance) — it is slower than stock.**

## Performance

**The fused MoE op is slower than the stock Habana `mixture_of_experts` intrinsic in both
regimes. It is correct, general, stable, and serves end-to-end — but it is NOT a MoE speedup.
Do not enable `HPU_MOE_FUSED_OP` (or the monolithic variant) for speed.**

| Regime | Fused custom op | Stock `mixture_of_experts` | Verdict |
|---|---|---|---|
| **Single-stream** (batch=1, warm) | **~23 tok/s** | **~30 tok/s** (proven ~29–34 ceiling) | **stock faster** |
| **Throughput** (Qwen-397B, concurrency-16, `--max-num-seqs=16`, warm) | **~68 tok/s** | **~126 tok/s** | **stock faster** |

Why stock wins:

1. **The stock op is heavily optimized.** `mixture_of_experts` is a batched/grouped intrinsic;
   the custom op's overhead (chunked sub-chains to force SRAM-residency, OpBackend dispatch, a
   much larger captured graph) **outweighs the SRAM-residency bandwidth saving**. The ~99 µs
   dequant-to-HBM floor does not hold against an internally-efficient stock op.
2. **Single-stream is host/collective-overhead-bound.** The MoE compute is **~0.16 ms of a
   ~127 ms** decode step (~90% overhead); **~30 tok/s is the sync-bound batch=1 cap** and the
   MoE op is never the bottleneck. A batched-K variant (killing the sequential K-emission) did
   **not** help single-stream (still ~23 tok/s) — it is not launch overhead; the stock
   intrinsic is simply faster at the compute.
3. **Throughput was config-limited, not op-limited.** max-num-seqs=16 is not the box's ceiling:
   stock at `--max-num-seqs=64` peaks **~486 tok/s** warm. The A/B above is apples-to-apples at
   one low config; absolute throughput is healthy and config-bound. *(The `--max-num-seqs=128`
   ceiling number is pending and will be appended by the box owner.)*

**What IS the win:** the reusable **`OpBackend` infrastructure** (multi-node, SRAM-resident,
lazy-captured fused kernels with no TPC ISA, correct + general on Gaudi 2), whose real payoff
is **kernels with no stock equivalent** — the sibling **DSA sparse-attention kernel** (op-level
validated on-card; the 1M-context enabler), which has no Habana intrinsic to lose to. Building
a custom kernel to *beat* a well-optimized stock op inherits that op's optimization debt as the
bar; the right target is what the platform lacks. (`fp8-compute` is a memory/throughput option
with a ~6.4e-2 double-quant floor, not a single-stream win.)

Architecture and code details:

- [`KERNEL_ARCHITECTURE.md`](KERNEL_ARCHITECTURE.md) — the how & why (problem, solution, node
  graph, eager/lazy, fp8, generality).
- [`KERNEL_TECHNICAL.md`](KERNEL_TECHNICAL.md) — node-by-node walkthrough, the
  OpBackend/registration triple, the build, and the **"gates cracked"** reference.

## Repository layout

```
tpc_kernel/opbackend/
  moe_fused_expert_backend_shared.h        # the recipe (MoeFusedExpert::AddNode + meta)
  moe_fused_expert_backend.cpp             # eager frontend + registration (single-expert)
  moe_fused_expert_backend_lazy.cpp        # lazy frontend (captures into the decode graph)
  moe_fused_expert_emit.h                  # emit_one_expert() — the per-expert chain, factored
  moe_fused_expert_routed_backend_shared.h # K-expert single-recipe op (MoeFusedMoE::AddNode)
  moe_fused_expert_routed_backend.cpp      # eager frontend (K-expert)
  moe_fused_expert_routed_backend_lazy.cpp # lazy frontend (K-expert)
  moe_fused_expert_fp8_backend.cpp         # fp8-compute variant (separate library; scaffold)
  build*.sh                                # CPU-only host builds (one per .so)
  test_fused_expert_op.py                  # single-expert device test
  test_fused_expert_fp8_op.py              # fp8 device test
  dsa/                                     # sibling DSA OpBackend ops (same mechanism)
hpu_moe_ext/
  fused_expert_routed.py                   # Python K-expert wrapper (gather/route + single-recipe route)
tests/
  test_fused_moe_routed.py                 # K-expert wrapper device test
  test_fused_moe_single_recipe.py          # K-expert single-recipe op device test (+ lazy capture)
  test_fused_op_realweights.py             # real GLM expert from a checkpoint
  test_fused_op_capture.py                 # HPU-graph capture of the routed path
docs/                                      # (the two design docs are mirrored here too)
```

## Build (CPU-only)

The kernel builds **CPU-only** — it compiles C++ against the bridge headers + torch + synapse
headers and links the **shipped backend lib**; it never touches a device. RUN the resulting
`.so` on the HPU box only.

### 1. Clone the version-exact bridge

The OpBackend ABI is open-source. Clone it into `tpc_kernel/opbackend/bridge` at the exact tag
the installed plugin matches:

```bash
git clone --depth 1 --branch v1.24.0-1007 \
  https://github.com/HabanaAI/gaudi-pytorch-bridge.git \
  tpc_kernel/opbackend/bridge
```

> **This must match your installed `habana-torch-plugin` exactly** (here `1.24.0.1007`, torch
> 2.10, `_GLIBCXX_USE_CXX11_ABI=1`). The `.so` is image-specific — see
> [`KERNEL_TECHNICAL.md`](KERNEL_TECHNICAL.md) Gate 6.

The bridge clone and the `build/` outputs are intentionally **not** included here (the bridge
is a large separate Apache-2.0 repo; `.so`s are environment-specific binaries).

### 2. Build each `.so` inside the dev image (no device)

```bash
# single-expert: eager + lazy
docker run --rm -v "$PWD/tpc_kernel:/work" -w /work/opbackend vllm-gaudi:next bash build.sh
docker run --rm -v "$PWD/tpc_kernel:/work" -w /work/opbackend \
  -e PT_HPU_LAZY_MODE=1 vllm-gaudi:next bash build_lazy.sh

# K-expert single-recipe: eager + lazy
docker run --rm -v "$PWD/tpc_kernel:/work" -w /work/opbackend vllm-gaudi:next bash build_routed.sh
docker run --rm -v "$PWD/tpc_kernel:/work" -w /work/opbackend \
  -e PT_HPU_LAZY_MODE=1 vllm-gaudi:next bash build_routed_lazy.sh

# fp8-compute variant (scaffold; falls back to bf16 unless BUILD_FP8_NODES=1)
docker run --rm -v "$PWD/tpc_kernel:/work" -w /work/opbackend vllm-gaudi:next bash build_fp8.sh
```

Each script links **only** `libhabana_pytorch_backend.so`, uses `-Wl,--allow-shlib-undefined`
to leave the frontend symbols resolved at runtime, forces `_GLIBCXX_USE_CXX11_ABI=1`, and
asserts the `OpBackend`/`BuildNode`/`KernelRegistry` symbols resolve from the linked backend
lib. Outputs land in `tpc_kernel/opbackend/build/lib*.so`. See
[`KERNEL_TECHNICAL.md`](KERNEL_TECHNICAL.md) §6 for why (Gate 5 = the double-registration
abort if you link the plugins).

### 3. Test on the HPU box

```bash
cd tpc_kernel/opbackend && python test_fused_expert_op.py        # single-expert
cd ../../ && python tests/test_fused_expert_general.py           # 5-profile generality
python tests/test_fused_moe_single_recipe.py                     # K-expert single recipe (+ capture)
PT_HPU_LAZY_MODE=1 python tests/test_fused_op_realweights.py     # one real expert (lazy .so)
```

`test_fused_op_realweights.py` defaults to a local checkpoint path; override with
`MODEL=/path/to/GLM-5.1-int4-mixed-AutoRound python tests/test_fused_op_realweights.py`.

## The op API

### Single expert — `torch.ops.hpu_moe_ext.fused_expert`

```
fused_expert(Tensor x, Tensor w13_packed, Tensor w13_scale, Tensor w13_zp,
             Tensor w2_packed, Tensor w2_scale, Tensor w2_zp,
             Tensor router_weight) -> Tensor
```

All tensors PyTorch row-major; `M`=tokens, `H`=hidden, `I`=intermediate, `GS`=group size,
`2I`=gate||up width:

```
  x          bf16  [M, H]
  w13_packed i32   [H, 2I/8]      8 int4 nibbles / int32 along OUT
  w13_scale  bf16  [H/GS, 2I]
  w13_zp     i32   [H/GS, 2I/8]   packed like the weight
  w2_packed  i32   [I, H/8]
  w2_scale   bf16  [I/GS, H]
  w2_zp      i32   [I/GS, H/8]
  router_weight bf16 [1, 1]       folded by the final mult (single-stream M=1)
  -> out     bf16  [M, H]   == silu(x@W13[:,:I]) * (x@W13[:,I:]) @ W2 * router_weight
```

### K experts, one recipe — `torch.ops.hpu_moe_ext.fused_moe`

```
fused_moe(Tensor x,
          Tensor[] w13_packed, Tensor[] w13_scale, Tensor[] w13_zp,
          Tensor[] w2_packed,  Tensor[] w2_scale,  Tensor[] w2_zp,
          Tensor[] router_weights) -> Tensor
```

Each list has K elements; element `k` is expert `k`'s tensor in the per-expert layout above
(`router_weights[k]` is `[1,1]`). Returns `sum_k( fused_expert(x, W_k) * router_weights[k] )`
as ONE captured recipe. Feed the K experts as `.unbind(0)` views of `index_select` outputs
(see `hpu_moe_ext/fused_expert_routed.py`).

### fp8-compute — `torch.ops.hpu_moe_ext_fp8.fused_expert_fp8`

Same schema/inputs/output as `fused_expert`, but the two expert GEMMs run on the MME in fp8.
Separate library namespace so it coexists with the bf16 op. INT4→fp8 is double quantization
(~6.4e-2 relNorm floor): a memory/throughput option, not an accuracy win. Ships as a
compile-ready scaffold; set `BUILD_FP8_NODES=1` after the on-device param-struct validation.

## Wiring into vLLM (vllm-gaudi)

The `MoeWNA16HPUMethod` plugin (in the parent `hpu-moe-ext` project; `moe_wna16_hpu.py` is the
integration point) drives the op behind env flags:

- `HPU_MOE_FUSED_OP=1` — use the fused op in place of the stock
  `convert_from_uint4`+`mixture_of_experts` tail.
- `HPU_MOE_ROUTED_K=<K>` — the static gather width (e.g. the model `top_k`); the fused path
  lives inside this gate.
- `HPU_MOE_FUSED_VERIFY=1` — run both fused + stock every call, assert `relNorm < 1e-2`, and
  stay coherent (return the stock result) on mismatch.

Eligibility (else it falls back to the stock path): **GPTQ-symmetric INT4**,
**`activation=="silu"`**, **`tokens*top_k <= HPU_MOE_ROUTED_K`**. The plugin auto-picks the
lazy vs eager `.so` by `htorch.is_lazy()`. The stored weight layout maps to the op contract by
construction (the plugin's repack already produces the op's PyTorch-row-major shapes); the only
adaptation is broadcasting the GPTQ-symmetric shared zero to a per-expert stack. See
`hpu_moe_ext/fused_expert_routed.py` for the gather/route/clone/single-recipe wrapper.

## Dependency + license

Built against **HabanaAI/gaudi-pytorch-bridge** tag **`v1.24.0-1007`** (the open-source Habana
PyTorch bridge / OpBackend ABI; **Apache-2.0**), version-exact to `habana-torch-plugin
1.24.0.1007` (torch 2.10). Clone it yourself (above) — it is **not** vendored here.

This project is released under **Apache-2.0** (matching the bridge it is built against). See
[`LICENSE`](LICENSE).

## Sibling: DSA ops (`tpc_kernel/opbackend/dsa/`) — where the infra pays off

The same `OpBackend` mechanism is reused for **DeepSeek Sparse Attention**: a lightning-indexer
top-k op (`hpu_dsa_ext::indexer_topk`) and a gather-consume sparse-attention op
(`hpu_dsa_ext::sparse_attn`). **Unlike MoE, there is no stock Habana intrinsic for DSA** — so
this is the infrastructure enabling a capability the platform lacks (and the long-context /
1M enabler), not competing with an optimized op it can only lose to. Both ops are **op-level
validated on-card** (all 3 gates: indexer top-k recall **0.999** vs the DeepSeek/HF oracle;
`sparse_attn` == dense-MLA when ctx ≤ topk; `sparse_attn` ≠ dense **and** matches the gather
oracle when ctx > topk). Included as source because they share the architecture; their design
is documented separately (not in this repo's doc set).
