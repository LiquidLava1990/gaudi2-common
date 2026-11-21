# Offline pre-repack cache (GLM-5.1 MoE int4 on Gaudi)

Eliminate the load-time uint8 -> int32 MoE repack so **lazy** mode on
`vllm-gaudi:next` has no freed-but-pinned uint8 originals (~42-52GB) eating the
KV budget. If there is no repack, there is nothing to trap.

Two phases, both gated by env vars (eager `:1.24` is byte-identical when unset):

| env | effect |
|-----|--------|
| `HPU_MOE_REPACK_SAVE=<dir>` | after the normal repack, save this rank's op-ready int32 weights/scales(/AWQ zeros) per layer |
| `HPU_MOE_REPACK_LOAD=<dir>` | allocate int32 placeholders (no 52GB uint8 originals), skip the repack, load the cached tensors straight into the op |

Cache files: `<dir>/rank{RANK}_layer{LAYER}.safetensors` (one per rank per MoE layer).
At TP=8 EP, `RANK == ep_rank == tp_rank`, so 8 ranks x N MoE layers files.

---

## (a) Mint the cache — ONE eager run

Run the normal eager (`:1.24`-style, `PT_HPU_LAZY_MODE=0`) bring-up with the
SAVE dir set. The model loads + repacks exactly as today, then dumps each
layer's op-ready tensors. You can stop the server right after weights finish
loading (warmup not required to mint the cache).

```bash
mkdir -p /mnt/fastmodels/moe_repack_cache    # any persistent dir with ~52GB free

docker run --rm -it \
  --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
  --cap-add=sys_nice --net=host --ipc=host \
  -e PT_HPU_LAZY_MODE=0 \
  -e HPU_MOE_REPACK_SAVE=/mnt/fastmodels/moe_repack_cache \
  -v /mnt/fastmodels:/mnt/fastmodels:ro \
  -v /mnt/fastmodels/moe_repack_cache:/mnt/fastmodels/moe_repack_cache \
  -v $HOME/AI/hpu-moe-ext:/opt/hpu-moe-ext \
  vllm-gaudi:1.24 \
  <your normal eager vllm serve ... GLM-5.1 ... --tensor-parallel-size 8 --enable-expert-parallel>
```

Watch the log for one `"[hpu-moe-repack] saved rank=R layer=L -> ..."` per
layer per rank. Expect `8 * <num MoE layers>` files. Sanity check:

```bash
ls /mnt/fastmodels/moe_repack_cache | wc -l        # == 8 * num_moe_layers
ls /mnt/fastmodels/moe_repack_cache | head
```

> The SAVE run can use **lazy** too, but eager is simplest (whole-tensor
> `.to('cpu')` is unconditionally valid in eager; under lazy the save path's
> `.to('cpu')` of the fresh op-ready stacked tensors also works since they are
> offset-0/unshared, but eager avoids any doubt).

## (b) Serve lazy from the cache

Same launch you use for the lazy `:next` bring-up, but `PT_HPU_LAZY_MODE=1`,
**no** SAVE var, and the LOAD var pointing at the same dir:

```bash
docker run --rm -it \
  --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
  --cap-add=sys_nice --net=host --ipc=host \
  -e PT_HPU_LAZY_MODE=1 \
  -e HPU_MOE_REPACK_LOAD=/mnt/fastmodels/moe_repack_cache \
  -v /mnt/fastmodels:/mnt/fastmodels:ro \
  -v /mnt/fastmodels/moe_repack_cache:/mnt/fastmodels/moe_repack_cache:ro \
  -v $HOME/AI/hpu-moe-ext:/opt/hpu-moe-ext \
  vllm-gaudi:next \
  <your normal lazy vllm serve ... GLM-5.1 ... --tensor-parallel-size 8 --enable-expert-parallel>
```

Log shows `"[hpu-moe-repack] LOADED rank=R layer=L from cache (no repack)"` per
layer. A **cache miss raises loudly** (FileNotFoundError) — it never silently
falls back to the repack.

## Expected memory effect

- **No** uint8 fused `w13_qweight`/`w2_qweight` (~52GB) is ever allocated as the
  load target; the placeholders are `(E,1,1)` int32 (KBs).
- `_repack_weight` is never called -> no freed-but-pinned originals under lazy
  -> the per-layer `synchronize`/`gc` reclaim is unnecessary and skipped.
- The device pool holds only the real op-ready int32 weights (~the model's
  ~52GB working set), so the KV cache fits where it previously OOM'd on the
  ~42GB of trapped originals.

## Invariants preserved

- moe_op built with the **LOCAL** window `[0, num_experts-1]` (EP global->local fix).
- scales transposed to `[E, num_groups, out]` (cached already-transposed).
- GPTQ-symmetric shared zero rebuilt by `_sym_zero` on load (NOT cached) — bit-identical to eager.
- AWQ per-expert zeros cached as the op-ready int32 (post `_awq_zero`).
- routed-gather stashes (`_hpu_routed_*`, `_hpu_experts_min`, `_hpu_local_experts`) set identically.
- o_proj g_idx RowParallel offset fix lives in the linear path (`GPTQDequantToBF16LinearMethod`) — untouched.

## Risks / edge cases

- **Rank keying**: relies on `ep_rank == get_rank()` (true for TP=8 full EP). If
  you change TP/EP topology between mint and serve, the cache is invalid — re-mint.
- **Cache staleness**: tied to the exact checkpoint + group_size + EP layout. If
  the model or sharding changes, delete the dir and re-mint. No checkpoint hash
  is embedded (add one if you keep multiple models' caches in one tree).
- **EPLB**: `get_expert_weights()` would view the placeholder qweights; do NOT
  enable `--enable-eplb` with the LOAD path (static EP only). The placeholders
  are contiguous/viewable so it won't crash, but rearrangement would be wrong.
- **g_idx params**: the GPTQ zero-size `w13_g_idx`/`w2_g_idx` keep the upstream
  wrapped loader (returns early for g_idx) — harmless and unused.
