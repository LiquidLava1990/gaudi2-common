# Intel Gaudi 2 LLM Inference — Optimization Guide

**This machine:** Supermicro X12DPG-OA6-GD2 · 8× Intel Gaudi 2 (HL-225H, 96 GB HBM each) ·
Ubuntu 26.04 · **pinned kernel 6.8.12 + out-of-tree `habanalabs-dkms` 1.24** · serving over
LAN (static <SERVER_IP>) + Tailscale (<SERVER_IP>).

> Scope: maximizing **inference throughput** (tokens/sec served) for an OpenAI-compatible
> endpoint other machines call. Engine = **vLLM via the `vllm-gaudi` plugin** inside Intel's
> prebuilt SynapseAI container. Last updated 2026-06-04.
>
> ⚠️ **READ [Part III](#part-iii--build-log-update-2026-06-04--what-actually-happened) FIRST for
> current state.** Parts I/II were written 2026-06-03 around an FP8-first / `--data-parallel-size`
> plan that has since been *superseded by measurement*: quantized **MoE** on this stack needs a
> custom plugin (built), in-engine **DP does not work** (use external replication), and **Qwen3.6-35B
> is the live, working model**. Where Part III contradicts I/II, **Part III wins.**

---

## Build state & milestone log (lean — how we got it working)

The host (Ubuntu 26.04 / py3.14 / kernel 7.0) is **off Intel's support matrix**, so the path
below was non-obvious. In order:

1. **Kernel driver.** In-tree kernel-7.0 `habanalabs` can't drive 1.24 firmware; matching
   out-of-tree `habanalabs-dkms 1.24.0-1007` **won't compile on kernel 7.0** (6.13+ API breaks).
   Fix = install mainline **kernel 6.8.12-060812-generic** + **gcc-13** → DKMS builds clean,
   all 8 cards **Operational**. **6.8.12 is PINNED as the permanent GRUB default**
   (`grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.8.12-060812-generic"`);
   kernel 7.0 kept only as fallback. Do NOT let it boot 7.0 — the driver breaks there.
2. **GATE ZERO passed.** 1.24 container userspace → DKMS driver → torch HPU bridge → real HPU
   matmul, all green. **Critical container flag:** with manual `--device` passthrough (we avoid
   `--runtime=habana`), `/dev/accel/*` alone is NOT enough — must ALSO pass
   **`/dev/infiniband/uverbs*`** + `--net=host --ipc=host --ulimit memlock=-1`, else the runtime
   dies `ibv init failed → synStatus=26 Device acquire failed`. Baked into `serve.sh` + `gate_zero.sh`.
3. **Runtime/infra.** Docker (v29.1.3) running, the user in the docker group (no sudo). Tailscale up.
   Static LAN IP set (WiFi, netplan; wired ens120f0 has no cable yet). Passwordless sudo.
4. **Display.** No display GPU — onboard ASPEED `ast` only (software/llvmpipe). "No resolution
   options" was a **0-byte EDID** (DDC blocked); fixed with GRUB `drm.edid_firmware=DVI-I-1:edid/1920x1080.bin`.
5. **Models (both INT4, one at a time).** `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` (GPTQ, ready) and
   `QuantTrio/GLM-5.1-AWQ` (AWQ, 744B MoE — **DSA/MLA-on-Gaudi is the serve gate**). Dropped the
   earlier Qwen3-235B-FP8 (user wants INT4, not FP8).
6. **Serving — two flags that gate everything (learned live 2026-06-03):**
   (a) **`PT_HPU_LAZY_MODE=0` (torch.compile) is mandatory** on the upstream-PyTorch-2.10 base —
   lazy mode raises `AssertionError: Upstream PyTorch 2.10.0+cpu not supported in Lazy mode`
   *before HPU is detected*, which surfaces as a misleading `Failed to infer device type`.
   (b) **INT4 quant flags must be the HPU kernels: `--quantization gptq_hpu` / `awq_hpu`** — the
   generic `gptq`(auto_gptq)/`awq` names are rejected (`not supported in hpu`). vllm-gaudi's full
   `supported_quantization = [compressed-tensors, fp8, inc, awq_hpu, gptq_hpu, modelopt]`.
   With both fixes, Qwen3.5-397B (gptq_hpu, `Qwen3_5MoeForConditionalGeneration`) loads + serves.
7. **Benchmark.** Logic-pruned OFAT plan in `/mnt/data1/bench/TEST_PLAN.md`; concurrency
   **1/10/25/50** every cycle; `run_sweep.sh` per model → `aggregate.py` for the final table.
   Lazy-mode is NOT a testable axis on this base (compile-only). *(Superseded — see Part III: the
   box now STAYS UP to serve; the auto-power-off was stripped, and the model lineup changed.)*

---

## 0. Where to find more tricks (sourcing strategy)

Gaudi is niche hardware — generic Google/Reddit/Twitter are **thin and often wrong** for it.
The real tuning intel lives in primary sources. Search these, in roughly this priority:

| Source | What you get | How to search it |
|---|---|---|
| **HabanaAI/vllm-fork** + **vllm-project/vllm-gaudi** GitHub *Issues* | Real-world war stories: OOM at batch N, bucket configs, env-var fixes | `is:issue throughput` / `is:issue OOM` / `is:issue bucket` in the repo |
| **docs.habana.ai** (Gaudi Documentation) | Authoritative: HPU Graphs, bucketing, warmup, profiler, memory | Site search; key pages: "vLLM Quick Start", "Optimization", "Profiler User Guide" |
| **Intel Gaudi developer forum** (community.intel.com → Gaudi/Habana board) | Q&A with Intel engineers; firmware/driver edge cases | Google: `site:community.intel.com gaudi <topic>` |
| **huggingface/optimum-habana** (`examples/text-generation`) | Per-model *validated* configs & recipes | Repo `README` + `examples/` per model |
| **arXiv** | Quantitative deep-dives (e.g. FP8 paper 2503.09975) | `arxiv.org Gaudi FP8 / inference` |
| **Intel "Model Performance Data" pages** | Official tok/s baselines per model/precision — sanity-check your numbers | intel.com → developer → platform → gaudi → model-performance |

**Search-phrasing tips that actually work for Gaudi:**
- Always include the SynapseAI/software version (e.g. `1.24`) — behavior changes a lot between releases.
- Use Gaudi vocabulary, not NVIDIA: "HPU", "SynapseAI", "INC" (Intel Neural Compressor),
  "FusedSDPA", "bucketing", "lazy mode", "hl-smi", "hl-prof" — not "CUDA/TensorRT/flash-attn".
- For numbers, search `tok/s` + model + `Gaudi 2` + `vLLM`; cross-check against Intel's official perf pages.
- Reddit (`r/LocalLLaMA`) is mostly NVIDIA/Apple; it rarely has Gaudi-specific data and is blocked to
  many crawlers anyway. Treat it as anecdote, verify against the primary sources above.

---

## 1. The big levers, ranked

1. **FP8 (INC) quantization incl. KV cache** — ~2× throughput, ~½ memory, <1% accuracy loss on 70B-class.
2. **HPU Graphs + warmup ON** — the fast execution path; eliminates host-bound overhead.
3. **Replicate, don't tensor-parallel** — for models that fit on one card, 8 replicas ≫ one TP=8 instance.
4. **Saturate the batch** — size `--max-num-seqs` / decode buckets so steady-state batch stays full.
5. **Host OS tuning** — memory/CPU/NVMe/network tweaks (Section 5). Smaller wins, but free and stackable.

---

## 2. FP8 / INC quantization (biggest single win)

FP8 (E4M3) via **Intel Neural Compressor (INC)** roughly doubles throughput and halves memory vs BF16.
Best ROI on large models (<1% loss on 70B; 7–8B models can lose up to ~2%, decide per use case).

```
--quantization inc
--kv-cache-dtype fp8_inc            # quantize the KV cache too — huge for long ctx / big batch
export QUANT_CONFIG=/path/to/quant_config.json
VLLM_DISABLE_MARK_SCALES_AS_CONST=true   # speeds FP8 warmup when scale_format=const
```

- **Skip embedding + LM-head layers** from quantization (default INC recipe does this) — that's where
  accuracy damage concentrates.
- Calibrate activation ranges on a representative dataset; use a backoff factor β ≈ 0.95–1.0 for headroom.
- Gaudi 2 FP8 ceiling (from arXiv 2503.09975): ~865 TFLOPS peak GEMM (~98% MFU);
  Llama-3.1-70B prefill ~649 TFLOPS @1k tokens, decode ~387 TFLOPS @batch-128.

---

## 3. Engine / execution-mode tuning (vLLM-Gaudi)

```
PT_HPU_LAZY_MODE=1                  # HPU Graphs — the most performant path. DO NOT use --enforce-eager.
PT_HPU_ENABLE_LAZY_COLLECTIVES=true # only needed for multi-card (TP) instances
VLLM_EXPONENTIAL_BUCKETING=true     # default since 1.22 — fewer buckets to compile, good default
VLLM_GRAPH_RESERVED_MEM=0.1         # default: reserve 10% usable mem for graph capture
```

**Warmup — keep it ON in production.** ⚠️ This is the opposite of the NVIDIA/llama.cpp habit.
On Gaudi, warmup is what *pre-compiles the HPU graph buckets*; skip it (`VLLM_SKIP_WARMUP=true`,
dev only) and you eat a graph-compilation stall on the first request of every new shape.
Cost: slow startup (minutes). Benefit: fast, stall-free steady state.

**Bucketing** maps the forward pass across (batch_size × seq_len). Exponential bucketing is the default;
only hand-tune the linear knobs if profiling shows one shape dominating:
```
VLLM_PROMPT_BS_BUCKET_MIN/STEP/MAX
VLLM_DECODE_BS_BUCKET_MIN / VLLM_DECODE_BLOCK_BUCKET_STEP / VLLM_DECODE_BLOCK_BUCKET_MAX
# DECODE_BLOCK_BUCKET_MAX ≈ max(128, (max_num_seqs * max_model_len) / block_size)
```

**Server params:**
```
--block-size 128          # BF16 on Gaudi 2: feeds the matmul engine; default 16/32 leaves perf on table
--max-num-seqs 128–256    # batch lever; push as high as KV cache allows
--gpu-memory-utilization 0.9
--max-model-len <set to real need>   # smaller = more KV cache headroom = bigger batches
# multi-step scheduling cuts host overhead
```

---

## 4. Model placement across the 8 cards (throughput strategy)

The decision that most affects **aggregate** tok/s:

- **Don't reflexively TP=8.** Tensor-parallel adds all-reduce/sync overhead every layer.
  Use TP only when a model genuinely doesn't fit on one 96 GB card.
- **Model fits on one card** (e.g. Qwen3-32B in FP8 ≈ ~32 GB): run **8 independent replicas**
  (or 4× TP=2) behind a load balancer. Aggregate throughput crushes a single TP=8 instance.
- **Model too big** (e.g. Qwen3-235B-A22B MoE): use TP=8 + `PT_HPU_ENABLE_LAZY_COLLECTIVES=true`.
- **Data-parallel serving:** ⚠️ **CORRECTION (see Part III §DP):** vLLM's *in-engine* DP
  (`--data-parallel-size N`) is **NON-FUNCTIONAL on vllm-gaudi** — the engine cores never init
  (lazy-mode-gated; "CI-only"). Do **not** rely on it. Real DP here = **external multi-container
  replication** (`dp_serve.sh`): N independent `vllm serve --tensor-parallel-size 1` processes,
  each pinned with `HABANA_VISIBLE_MODULES`, behind an nginx `least_conn` balancer.

**"Saturate the batch" principle** (borrowed from the GB10 *zero-queue effect*): throughput jumps when
every scheduler slot is simultaneously full. Size `--max-num-seqs` and decode buckets so the steady-state
batch stays saturated rather than draining between requests.

---

## 5. Host OS tuning (hardware-agnostic — apply on the host now)

These came over from NVIDIA GB10 tuning but are pure-Linux and benefit any accelerator.
Most help model-load time, memory stability, and **network serving latency** (this is a network server).

**`/etc/sysctl.d/99-gaudi-inference.conf`:**
```
vm.swappiness = 1                       # keep weights resident, never swap
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
# network — matters because we serve over LAN/Tailscale
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
net.ipv4.tcp_slow_start_after_idle = 0
```

**Transparent hugepages** (helps large model allocations):
```
echo always        > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
```

**CPU governor → performance:**
```
cpupower frequency-set -g performance
# persist via GRUB: cpufreq.default_governor=performance
```

**NVMe (faster weight load from disk):**
```
echo none > /sys/block/<nvme>/queue/scheduler
echo 4096 > /sys/block/<nvme>/queue/read_ahead_kb
```

**Misc:**
```
ulimit -l unlimited                                  # memlock — container needs pinned memory
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sync; echo 3 > /proc/sys/vm/drop_caches              # before a big cold model load
# kernel cmdline: iommu.passthrough=1  → cuts IOMMU overhead on device DMA
```

> Make the `/sys` and `cpupower` ones persistent via a `systemd` unit or `tuned` profile so they
> survive reboot (the box reboots for GNOME). sysctl drop-in and GRUB params persist on their own.

---

## 6. NVIDIA→Gaudi translation table (what carries, what doesn't)

| NVIDIA / GB10 trick | Gaudi 2 status |
|---|---|
| `vm.swappiness`, dirty ratios, THP, governor, NVMe, TCP/bbr, memlock, suspend-mask | ✅ Transfers as-is (Section 5) |
| `iommu=pt intel_iommu=on` kernel cmdline | ✅ Transfers — but on a **kernel-6.8 Gaudi host it's MANDATORY, not just a perf knob**: without it multi-card HCCL dies with "No available intermediate buffer" (see Gate Zero-B). Verify `dmesg \| grep "Default domain"` = Passthrough. |
| Clock lock `nvidia-smi -lgc` | ❌ No user lever — Gaudi clocks firmware-fixed |
| Persistence `nvidia-smi -pm 1` | ❌ Not needed — `habanalabs` driver keeps cards initialized |
| `-fa on` flash attention | ⇒ Auto: vLLM uses Gaudi **FusedSDPA** |
| `-ctk/-ctv q4_0` KV quant | ⇒ `--kv-cache-dtype fp8_inc` |
| `-np 64` parallel slots | ⇒ `--max-num-seqs` |
| `-ngl 999` offload all layers | ⇒ N/A — full model already on HPU |
| CUDA 12.8-not-13.x version pin | ⇒ SynapseAI version match — handled by the container image |
| `GGML_CUDA_*`, `CUDA_MODULE_LOADING`, llama.cpp flags | ❌ Dead — **llama.cpp/ollama have no HPU backend** |
| `--no-warmup` | ⚠️ **INVERTED** — on Gaudi keep warmup **ON** (it compiles HPU graphs) |
| "zero-queue effect" (np=n → +40%) | ✅ Concept transfers — saturate `--max-num-seqs`/buckets |

---

## 7. Profiling & measuring

- **`hl-smi`** — the `nvidia-smi` equivalent: per-card utilization, HBM usage, power, temp.
  Run it *inside the container* first to confirm SynapseAI userspace talks to our in-tree driver.
- **`hl-prof` / Intel Gaudi Profiler** — trace HPU graph execution, find host-bound gaps and
  bucket misses (docs.habana.ai → Profiler User Guide).
- **Benchmark loop:** vLLM's `benchmark_serving.py` against the OpenAI endpoint; measure
  tok/s + TTFT + p99 at increasing concurrency until throughput plateaus or OOMs.
- **Sanity-check** your tok/s against Intel's official "Model Performance Data for Gaudi 2" pages.

---

## 8. Gotchas specific to this build

- Host is **off Intel's support matrix** (Ubuntu 26.04 / Python 3.14) → the matched SynapseAI userspace
  **must** come from the container; the host only provides driver + firmware (already working).
- **Open risk:** confirm the container's SynapseAI 1.24 userspace drives the **in-tree** `habanalabs`
  driver (not Intel's DKMS). First test: `hl-smi -L` inside the container shows all 8 cards.
- **Docker bypasses UFW** — bind the inference port to the Tailscale IP only and add `--api-key`;
  enforce host firewalling via `DOCKER-USER` chain or by binding a specific IP, not UFW rules.
- Fans run loud (~5700 RPM) by firmware design for the GPU/UBB thermal zone — "Optimal" is the quietest
  software-selectable profile; not a tunable. Not a throttling concern.

---

## Quick-start command skeleton (fill in model + paths)

```bash
docker run -it --rm \
  --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
  --cap-add=sys_nice --net=host --ipc=host \
  -e PT_HPU_LAZY_MODE=1 \
  -e VLLM_DISABLE_MARK_SCALES_AS_CONST=true \
  -e QUANT_CONFIG=/quant/qwen3_fp8.json \
  -v /models:/models -v /quant:/quant \
  vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-<ver> \
  vllm serve /models/Qwen3-32B \
    --quantization inc --kv-cache-dtype fp8_inc \
    --block-size 128 --max-num-seqs 256 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --host <tailscale-ip> --port 8000 --api-key <key>
# replicate this across cards with HABANA_VISIBLE_MODULES=0 / 1 / ... for 8 independent instances
```

---

# Part II — Deep-Search Verified Findings (2026-06-03)

> Produced by a 9-agent deep-search + adversarial-deliberation pass (5 search agents → 3
> verification lenses → synthesis; 178 web lookups, 98 raw tricks). Where this contradicts Part I,
> **Part II wins** — it's source-verified against vllm-gaudi 1.24 docs/PRs. Items are ranked by impact
> for *our* build (max aggregate tok/s, 8× HL-225H, FP8-first, replicate-don't-TP).

## ⓿ GATE ZERO — prove the off-matrix stack initializes before tuning anything
`hl-smi` working does **NOT** prove the SynapseAI 1.24 runtime inits on kernel 7.0 / host Py 3.14.
Bring-up gate, in order:
1. `hl-smi -L` shows driver + SPI firmware matching the 1.24 triple.
2. **Inside the 1.24 container:** run a trivial HPU tensor op **and** one short `vllm` generation —
   proves the `/dev/accel` ioctl ABI + torch 2.10 wheels actually work.
3. If `habanalabs-dkms` won't compile on 7.0 (`dkms build habanalabs/1.24.x -k $(uname -r)`, read
   make.log for mm/dma/pci API churn) → **boot a pinned 6.8 HWE kernel in GRUB** that 1.24 DKMS supports,
   or use Intel's "load driver inside running container" procedure.
4. Flash SPI firmware via `hl-fw-loader` from a matched-driver context (unload → flash → reload → verify;
   power-cycle clears the Firmware Update Lock).
5. **Stay inside the container's Python** — host 3.14 is irrelevant if you never use it.
**Everything below is moot until this passes.**

> **FIELD RESULT (2026-06-03, this box):** GATE ZERO **failed on the in-tree driver** exactly as predicted.
> Inside the 1.24 container `hl-smi -L` works (all 8 cards Operational, fw 1.24), but the HPU compute op
> fails: `synStatus=8 [Device not found] Device acquire failed` while `torch.hpu.device_count()=8`.
> Root cause from `dmesg`: `habanalabs ...: Cannot find handler for event 1296/1297/1295` across all 8
> cards — the **kernel-7.0 in-tree `habanalabs` driver doesn't recognize the 1.24 firmware's event-queue
> events**, so management works but compute-acquire fails. Also found: **no `habana-container-runtime`
> installed** (manual `--device=/dev/accel/*` passthrough is correct), and **SPI/preboot fw stuck at 1.14**
> while FIT/UBOOT/OS are 1.24. Fix in progress: install matching **`habanalabs-dkms 1.24.0-1007`** from the
> vault `noble` repo (kernel-7.0 headers present, so DKMS can attempt the build); if it won't compile on
> kernel 7.0, pin a 6.8 HWE kernel. Lesson: **`hl-smi` working is NOT sufficient — always run the compute op.**
>
> **✅ RESOLVED (2026-06-03): GATE ZERO PASSES.** Pinned **kernel 6.8.12** → `habanalabs-dkms 1.24.0-1007`
> builds clean, all 8 Operational, DKMS module matches 1.24 firmware (event-1296 error gone). The compute
> op then still failed `ibv init / synStatus=26` until we passed the **`/dev/infiniband/uverbs*`** RDMA
> nodes + `--net=host --ipc=host --ulimit memlock=-1` into the container (the HCL layer needs them; the
> `--runtime=habana` we avoid would have injected them). HPU matmul now runs on `hpu:0`. Both fixes live in
> `serve.sh` + `gate_zero.sh`. Second lesson: **manual device passthrough must include the IB verbs nodes.**

> **🚨→✅ GATE ZERO-B — MULTI-CARD HCCL SCALE-UP (2026-06-03, the real blocker):** Single-card compute
> passing does NOT prove multi-card works. After Gate Zero, **every multi-card collective failed**:
> `hcl/.../credit_manager.cpp:31(allocNextCredit): No available intermediate buffer` → `Collective call
> returned error`. Reproduced in ~60s with a bare `all_reduce` (`torchrun --nproc-per-node={2,8}
> test_hccl.py`), NOT vLLM-specific. Ruled out container access (`--privileged` still failed), quant, model,
> LAZY_MODE. **Both target models need TP=8 → nothing could serve.** Looked like the off-matrix `hbl_cn`
> (Cassini networking) driver failing on kernel 6.8.12.
> **ROOT CAUSE = missing IOMMU passthrough param.** Ubuntu's 6.8 kernel defaults IOMMU to Translated/DMA-FQ
> mode (Launchpad #2071471), which blocks the scale-up NIC's peer-to-peer DMA — single-card compute doesn't
> need peer DMA, multi-card collectives do → the exact "No available intermediate buffer" symptom. Our
> `/proc/cmdline` had no iommu params and IOMMU read "Translated".
> **FIX (docs.habana.ai → "Enable IOMMU Passthrough (required only for kernel 6.8)"):** add
> **`iommu=pt intel_iommu=on`** to `GRUB_CMDLINE_LINUX_DEFAULT`, `update-grub`, reboot. After reboot,
> `dmesg | grep "Default domain"` must say **Passthrough**. → **`test_hccl.py` then PASSES at world=2 (OK)
> AND world=8 (OK).** No SPI flash, no ACS-disable, no 24.04 reinstall needed. **Lesson: on a kernel-6.8
> Gaudi host, `iommu=pt intel_iommu=on` is MANDATORY, not a perf knob — without it single-card works and
> every TP>1 job dies with a misleading buffer error. Always gate multi-card with a bare all-reduce before
> blaming vLLM.**

## High-impact (ranked)
1. **Replicate the 32B 8× (TP=1), never TP=8.** Qwen3-32B fits one 96GB card (BF16 ~64GB / FP8 ~32GB).
   Gaudi 2 card-links are RoCE/HCCL, not NVSwitch — TP pays per-layer all-reduce for no gain when it fits,
   and TP=8 spans both NUMA sockets. ⚠️ **`--data-parallel-size 8` was later found NON-FUNCTIONAL
   (Part III §DP)** — use the second option only: 8× `vllm serve --port 80NN` pinned per card behind a
   load balancer (`dp_serve.sh`). Aggregate concurrency = `max_num_seqs × 8`.
2. **FP8 (INC) + FP8 KV — but A/B vs BF16 for Qwen3.** `--quantization inc --kv-cache-dtype fp8_inc
   --dtype bfloat16 --block-size 128`, `QUANT_CONFIG=maxabs_quant_g2.json`. INC is the **only** first-class
   FP8 path (generic `--quantization fp8` is NOT it; QUANT_CONFIG is the master switch, mode=MEASURE/QUANTIZE).
   ⚠️ **Qwen has documented FP8 regressions on Gaudi + FP8 forces the lazy path** → treat FP8-vs-BF16 for
   Qwen3-32B as a **required tok/s A/B**, not a settled win. FP8 is genuinely needed for the 235B.
3. **Execution mode: FP8 ⇒ `PT_HPU_LAZY_MODE=1`; BF16 ⇒ default torch.compile (0), A/B both.**
   1.24 default is 0. INC FP8 / spec-decode / some multimodal are *not supported under torch.compile*, so
   the FP8 path **must** run lazy. Consequence: **BF16 and FP8 run in different modes — bucket/graph configs
   do NOT transfer between them; tune each separately.** ("lazy is deprecated, leave at 0" = trap for FP8.)
4. **FP8 calibration ON the g2 cards; handle the ±240 E4M3 range.**
   - Calibrate on HL-225H: `vllm-hpu-extension/calibration/calibrate_model.sh -o g2` (MEASURE→`maxabs_measure_g2.json`,
     QUANTIZE→`maxabs_quant_g2.json`). **Never reuse Gaudi3 scales** (±240 vs ±448 → clip).
   - **OCP-FP8 checkpoints (official Qwen3-235B-A22B-FP8, DeepSeek): set `VLLM_HPU_CONVERT_TO_FP8UZ=1`**
     (÷2 weights, ×2 scales) or re-quantize with INC — else clip/overflow. Gaudi 3 doesn't need this.
   - **Never `scale_method=unit_scale`** (Mixtral PPL 11.34→96.65). Prefer calibrated maxabs; Gaudi 2 HW
     pow2 scales limited to {2⁻⁸,2⁻⁴,2⁰,2⁴} — validate pow2 vs maxabs_hw. `lm_head`/embeddings stay BF16.
     Validate MMLU/GSM8K with `lm-eval` post-calibration.
5. **`--block-size 128`** (BF16/MME tile alignment; CUDA-habit 16 underutilizes the matmul engine).
   FP8-KV optimum may differ (PRs #1453/#1461) — start 128, measure.
6. **`--max-num-batched-tokens ≥ 16384`.** A graph-skip regression (PR #1346) ran *all* >2K-token prefills
   eager (−29% on one FP8 TP=8 test); PR #1502 raised the floor. **Grep the startup log for the graph-skip
   `info_once` line** to learn if your build has #1502; if not, ≥16384 is a real rescue.
7. **`--max-num-seqs 128–256`** is the primary decode dial (memory-bound, saturates — tune, don't max).
   Size `VLLM_DECODE_BS_BUCKET_MAX=max_num_seqs` so warmup covers full batches.
8. **Sticky consistent-hash router in front of the 8 replicas** (`vllm-project/router`). vLLM's built-in DP
   balancer is queue-depth-only (NOT prefix-aware) → shared system prompts recomputed up to 8×. Sticky routing
   + `--enable-prefix-caching` per engine reclaims KV reuse; also masks Gaudi 2's ~1.5× slower prefill.
9. **235B MoE:** `--tensor-parallel-size 8 --enable-expert-parallel --gpu-memory-utilization 0.9
   --max-num-seqs 64` + FP8. Omitting `--enable-expert-parallel` shards experts like dense layers → slower
   AND can cause **graph-compile failures**. Required env: `PT_HPU_ENABLE_LAZY_COLLECTIVES=true`,
   `PT_HPU_WEIGHT_SHARING=0` (frees full-precision weights post-FP8 → avoids OOM). TP8-DP1 is the safe-fit launch.
10. **`PT_HPU_RECIPE_CACHE_CONFIG='<path>',<delete_bool>,<size_mb>`** persists compiled graphs across restarts:
    BF16 66→23s (~65%), **FP8 504→34s (~93%)**. **Local disk only** (NFS breaks it). First run delete=True,
    then delete=False. Silently invalidates on any shape/config/dtype/plugin change → pays off *post-tuning*.

## Medium-impact
- **Keep warmup ON** (CUDA has no compile-on-first-shape penalty; Gaudi does). Cut its cost with
  `VLLM_BUCKETING_STRATEGY=exp` (default; `VLLM_EXPONENTIAL_BUCKETING` is the deprecated alias) + bounded
  bucket ranges + recipe cache. Startup win, not steady-state.
- **`--async-scheduling`** (V1 path). Multi-step scheduling / `VLLM_DELAYED_SAMPLING` are **discontinued** —
  do NOT port CUDA `--num-scheduler-steps` (no-op).
- **Cap `--max-model-len` to real workload** (8K–32K): shrinks bucket grid + KV-per-seq → higher concurrency,
  dodges long-ctx OOM/bucket-explosion (PR #1376 fix). Bound `VLLM_PROMPT_SEQ_BUCKET_MAX` / `VLLM_DECODE_BLOCK_BUCKET_MAX`.
- **Pin replicas with `HABANA_VISIBLE_MODULES`** (physical OAM module_id, stable) NOT `HABANA_VISIBLE_DEVICES`
  (OS index, volatile). Map via `hl-smi -Q index,module_id -f csv`; NUMA-pin each replica with
  `numactl --cpunodebind=N --membind=N` (`hl-smi topo -c -N` gives card→NUMA node).
- **Container launch:** prefer explicit `--device=/dev/accel/accelN` + `--device=/dev/accel/accel_controlDN`
  for all 8 over `--runtime=habana` (habana-container-runtime is version-coupled to the driver, likely
  mismatched on our in-tree kernel-7.0 host). Plus `--cap-add=sys_nice --ipc=host --net=host`.
- **Qwen gate:** `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=false` (default — don't let a copied config flip it true);
  leave `VLLM_CONFIG_HIDDEN_LAYERS=1`.
- **Diagnostic loggers during bring-up:** `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION=true` (catch mid-serving
  recompiles = bucket misses), `VLLM_HPU_LOG_STEP_CPU_FALLBACKS=true` (verify 235B experts run FP8 on-device,
  not CPU), `VLLM_T_COMPILE_FULLGRAPH=true`, and `VLLM_PROFILER_ENABLED=true` → perfetto.habana.ai (gaps
  between device slices = host-bound → NUMA-pin / raise batch).
- **FP8 timeout hygiene:** raise `VLLM_ENGINE_ITERATION_TIMEOUT_S` + `VLLM_RPC_TIMEOUT` before FP8/235B serving
  (long compiles trip defaults → engine crash on first requests). `RUNTIME_SCALE_PATCHING=1` is **warmup-only**
  (costs ~5–20% steady-state) — dev only, disable in prod.
- **Thermal/power check** (custom non-HLS chassis): watch `hl-smi -Q clocks.current.soc,clocks.max.soc` under
  sustained 8-card load — `current << max` = throttling, caps every card. `power.draw` reports the 54V rail
  only (true total higher; don't undersize PSU/cooling on it).
- **`VLLM_GRAPH_RESERVED_MEM`:** read the usable/graph/KV split your build prints at startup; lower the reserve
  toward more KV only until graph-capture nears OOM. OOM playbook (do NOT reflexively `--enforce-eager`):
  ↑`--gpu-memory-utilization`→0.9, ↓max-num-seqs/batched-tokens, ↑TP, `--enforce-eager` LAST (diagnostic).
- **Host GRUB:** `intel_iommu=on iommu=pt` — on **kernel 6.8 this is REQUIRED for multi-card serving**, not
  just a DMA-overhead tweak: without it every TP>1 collective dies "No available intermediate buffer" while
  single-card works (see Gate Zero-B). Confirm with `dmesg | grep "Default domain"` → **Passthrough**.

## Low-impact / TP-only
- **235B TP path only:** `VLLM_ROW_PARALLEL_CHUNKS` / `_CHUNK_THRESHOLD` (default 1/8192) overlaps RowParallel
  matmul with all-reduce on long prefills (not for MoE experts; no-op for TP=1). **Do NOT set HCCL scale-out
  vars** (`HCCL_SOCKET_IFNAME`, `HCCL_OVER_OFI`, `RDMAV_FORK_SAFE`) on a single node — they can misroute
  collectives over the host NIC. **Pipeline parallel is NOT supported** in vllm-gaudi (planned only).

## ❌ Myths the deliberation rejected (do NOT do these)
| Myth | Why it's wrong on Gaudi 2 / 1.24 |
|---|---|
| `VLLM_CONTIGUOUS_PA=false` | Obsolete — traces to PR #12139 (old v2-block-manager). 1.24 **Unified Attention enforces** contiguous PA. |
| `PT_HPU_LAZY_MODE=0` "lazy is deprecated" | True only for pure-BF16 torch.compile; **FP8/INC requires lazy (=1)**. |
| `fp8_inc` KV "mandatory" | Strong win, but NOT required for 32B in a 96GB card (BF16 has KV headroom + faster torch.compile path). |
| "More TP is always better" | NVSwitch-era habit; inverts on Gaudi 2. Replicas scale ~linearly, TP=8 sublinear + cross-NUMA. |
| `scale_method=unit_scale` / "FP8 is automatic" | Catastrophic: Mixtral PPL 11.34→96.65. Always calibrate; never uncalibrated FP8 on MoE. |
| `VLLM_SKIP_WARMUP=true` in prod | Gaudi compiles per-shape AOT → mid-serving stalls. Dev only. |
| Port CUDA `--num-scheduler-steps` / `VLLM_DELAYED_SAMPLING` | Discontinued/no-op on V1 Gaudi; use `--async-scheduling`. |
| `RUNTIME_SCALE_PATCHING=1` for throughput | Warmup-only; costs 5–20% steady-state. |
| HCCL scale-out vars on single node | Multi-node knobs; misroute intra-node collectives over host NIC. |
| Load OCP Qwen3-235B-FP8 as-is | OCP ±448 vs Gaudi 2 ±240 → clip. Set `VLLM_HPU_CONVERT_TO_FP8UZ=1` or re-quant with INC. |

## On INT4 / INT6 (per the requirement to sometimes use them)
Across 98 tricks / 5 agents, **INT4 (AWQ/GPTQ) never reached the ranked list and INT6 never appeared.**
That's the deep search confirming: on Gaudi 2 the optimized low-precision path is **FP8**. INT4 has **no
native matmul acceleration** (weights dequant to bf16/fp8) — it's a **memory-fit fallback** for a model that
won't fit any other way, not a throughput play. **Practical rung order: BF16 → FP8(INC) → AWQ-INT4 (fit only).**
INT6 appears to be a non-path on this hardware. *(If a specific oversized model truly needs INT4, run a
dedicated follow-up to confirm AWQ/GPTQ support in the exact vllm-gaudi build before committing.)*

## Open questions to settle by measurement (no published Qwen3-on-Gaudi-2 baseline exists)
- **FP8 vs BF16 for Qwen3-32B** — settle by measured tok/s (BF16-torch.compile vs FP8-lazy).
- **`--block-size` for fp8_inc KV** — 128 is the BF16 answer; FP8-KV optimum unverified (PRs #1453/#1461).
- **Exact plugin version** — `pip show vllm-gaudi` + startup log; match PR inclusion (#1376/#1411/#1452/#1502/#1512).
- **235B fit arithmetic** on 8×96GB — does block-128 FP8 (~235GB) + KV + graph-reserve fit, and at what max-num-seqs?
- **Speculative decoding** (EAGLE/Medusa/n-gram) status on vllm-gaudi 1.24 for Qwen3 — potential stacked decode win
  (FP8 already forces lazy, where spec-decode is supported). Unexplored.
- **Host HTTP/tokenizer ceiling** at max aggregate request rate across 8 replicas (`--api-server-count`, tokenizer pool).
- Whether `VLLM_GRAPH_PROMPT_RATIO` / `*_STRATEGY` still function on 1.24 (absent from env page — likely legacy).

---

# Part III — Build Log Update (2026-06-04) — What Actually Happened

> Parts I/II were the *plan* (FP8-first, GPTQ/AWQ flags, `--data-parallel-size`). This is the
> *outcome* after bring-up. Where it contradicts I/II, **this wins.** The hardware/driver findings
> in Part II's Gate Zero / Gate Zero-B (kernel 6.8.12 pin, DKMS 1.24, `iommu=pt intel_iommu=on`,
> `/dev/infiniband/uverbs*` passthrough) all held up and remain correct.

## The quantization story flipped: FP8 → INT4 MoE via a custom plugin

The big surprise: **vllm-gaudi 1.24 has no HPU kernel for GPTQ or AWQ _MoE experts_.** Its only
wired quantized-MoE path is **`compressed-tensors` W4A16** (`HPUCompressedTensorsWNA16MoEMethod`
→ `VllmMixtureOfExpertsOpWNA16`). `hpu_gptq.py` returns `None` for FusedMoE; `hpu_awq.py` has no
MoE branch. So both originally-downloaded models (Qwen GPTQ-MoE, GLM AWQ-MoE) could load their
*attention/linears* but **not their experts** — "server starts" ≠ "experts quantized".

**Fix built: `hpu-moe-ext`** — a self-contained, out-of-tree vLLM plugin at
`/mnt/data1/serving/hpu-moe-ext/` (git repo; `vllm.general_plugins` entry point `04.hpu_moe_ext`,
loads *after* the stock plugin, **zero edits to `/opt/vllm-gaudi`** so it survives image rebuilds).
It subclasses upstream `MoeWNA16Method` and reuses the existing WNA16 INT4 op (the gap was a missing
*dispatch* branch, not a missing kernel). Capabilities proven working:
- **Autodetect quant from `config.json`** — no `--quantization` flag needed (gptq→gptq_hpu,
  awq→awq_hpu, auto-round→our path).
- **Linear-skip** — honors GPTQ `dynamic` / `modules_to_not_convert` so attn/vision stay bf16.
- **GPTQ-MoE numerics FIXED** — root-cause was a **symmetric zero-point off-by-one** (`_sym_zero`
  used 9; correct is **8**, the uint4 bias). Proven bit-exact against real autogptq safetensors,
  and end-to-end coherent on Qwen3-30B-GPTQ. (Commit `61c9d2f`.)
- `_compat.py` asserts the 10 upstream symbols it depends on → **loud** failure if an engine
  upgrade moves them (relevant to the vllm-gaudi v0.21.0 question — see below).

## AutoRound pivot (Intel-tuned INT4)

Switched the big models to **Intel AutoRound** quants (`Intel/Qwen3.5-397B-…-AutoRound`,
`INC4AI/GLM-5.1-int4-mixed-AutoRound`) — both pack `auto_round:auto_gptq` (GPTQ format underneath),
so the fixed GPTQ-MoE path handles the experts. Plugin was extended to:
- **Register `auto-round`/`auto_round`** as a quant method (vLLM core doesn't ship one) and append
  it to `platform.supported_quantization` (a 3rd serve-time allowlist gate that rejected it).
- **8-bit linears → dequant-to-bf16 at load** (`convert_from_uint4` is uint4-only; 768 GB HBM
  absorbs the bf16). 4-bit experts use the proven WNA16 op.
- **Index-driven dispatch (the decisive fix):** AutoRound `extra_config` *mislabels* stored formats
  (e.g. Qwen-397B's `shared_expert` projections are stored INT4 but tagged `bits:16`). Trusting the
  label dropped weights → **word-salad output**. Fix: read the actual set of `.qweight` prefixes
  from `model.safetensors.index.json` / shard headers — **stored format is ground truth**.
  Architecture-agnostic; correct for both Qwen (mislabeled int4) and GLM (genuine fp16/int8 mix).

## Model status (the real deliverables)

| Model | Status | Notes |
|---|---|---|
| **Qwen3.6-35B-A3B** (bf16 MoE, multimodal) | ✅ **WORKING — the live Qwen deliverable** | Stock unquantized-MoE path, **no plugin/quant**. TP=8 + `--enable-expert-parallel`, serves on Tailscale w/ API key. **Text + vision verified.** No audio (needs an Omni model). OFAT sweep: **100% accuracy, peak ~681 tok/s @c=50** (`MASTER_REPORT.md`). |
| **GLM-5.1-int4-mixed-AutoRound** (744B/40B MoE, DSA) | 🟢 **Current focus** | `GlmMoeDsaForCausalLM` arch confirmed supported. All AutoRound fixes apply. Serving with full warmup; DSA attention is the runtime unknown. |
| **Qwen3.5-397B-AutoRound** | 🅿️ **Parked** | Hybrid **Mamba + MoE** arch; phantom index keys (fixed via shard headers) + NaN traced to the quantized Mamba `linear_attn` path. Revisit only if time. |

## DP — why in-engine failed, what to use instead  {#dp}

**In-engine `--data-parallel-size N` is dead on this stack.** Confirmed twice: the `dp8_tp1` sweep
variant timed out at 40 min; the engine cores never initialize. Source check: the DP *compute*
paths exist (`hpu_dp_utils.py`, `get_dp_padding`, `hpu_fused_moe.py` dp_size branches), but they're
**gated on lazy mode** (`hpu_model_runner.py:1277` `is_lazy(...)`; `hpu_dp_utils.py:92`
`not enforce_eager`) — and our base image **must** run `PT_HPU_LAZY_MODE=0` (compile). Plus the v1
EngineCore-spawn + DP coordinator handshake (HPU forces `spawn`, not fork) is "CI-only", not wired
for general serving. Net: cores hang at init.

**Working DP = external multi-container replication** → `/mnt/data1/serving/dp_serve.sh`:
- N independent `vllm serve --tensor-parallel-size 1` containers — the **exact single-model code
  path already proven working**, so none of the in-engine DP failure modes apply.
- Each pinned to its card(s) with **`HABANA_VISIBLE_MODULES`** (stable OAM module_id), own port,
  own recipe-cache dir.
- An **nginx `least_conn`** balancer presents one OpenAI endpoint on the Tailscale IP:8000.
- Layouts: `dp8_tp1` (8 replicas, **no HCCL** — best for a model that fits one 96 GB card, e.g. a
  20B ≈ 40 GB bf16), `dp4_tp2`, `dp2_tp4`. TP=8 stays the *fit* lever for models too big to replicate.
- Tradeoff: replication duplicates all MoE experts N× (more HBM) vs in-engine DP's expert-sharing —
  irrelevant when the model fits, which is exactly when you'd use `dp8_tp1`.

Usage: `./dp_serve.sh <model-key> [dp8_tp1|dp4_tp2|dp2_tp4]`  ·  stop: `./dp_serve.sh stop`.

## KV-cache compression status

Only **`--kv-cache-dtype fp8_inc`** is accepted on Gaudi today (the platform gate
`is_kv_cache_dtype_supported` returns True only for `fp8_inc`). TurboQuant / sub-FP8 / asymmetric
K-V quant exist in the vLLM *core* build but are **blocked by that HPU platform gate** — enabling
them is engine-upgrade scope, not a flag. Treat `fp8_inc` as the one available KV-compression axis.

## Engine version note (vllm-gaudi)

We run a **dev snapshot** of `main` (`vllm_gaudi 0.20.1rc1.dev47+g9db96d9e`, vllm
`0.22.1rc1.dev66+gf8e9c56d1` — a deliberately matched pair). The newest *tagged release* is
**v0.21.0**, which has **diverged** from our checkout (not an ancestor). Moving to it is a deliberate,
gated migration (rebuild image + re-validate `hpu-moe-ext` against `_compat.py` symbols + re-run MoE
numeric tests + check it doesn't force PyTorch 2.10), **not** an in-place upgrade. Not "newer = do it."

## Ops / housekeeping

- **Box stays up** to serve (the old `autorun.sh` power-off was stripped per instruction).
- **Docker GUI = Portainer** (web) at `http://localhost:9000` (also Tailscale :9000). Admin login
  set; it self-locks if the admin password isn't set within ~5 min of a fresh start — just restart
  the container and set it. Desktop icon: *Docker (Portainer)*. (Do **not** install Docker-Desktop-
  for-Linux — it runs its own VM and won't see host containers or the Gaudi devices.)
- Serving entry points: `serve.sh <model-key>` (one model, fixed container `vllm-active`),
  `dp_serve.sh <model-key> <layout>` (replicated throughput), model configs in `serving/models/*.env`.

## DP throughput workflow (replication + warm-first + sweeping)  {#dp-workflow}

`dp_serve.sh` is the throughput path now that in-engine DP is confirmed dead. Two pieces:

**Warm-first startup (default ON, `WARM_FIRST=1`).** Launching 8 replicas at once means 8
simultaneous `torch.compile` graph passes hammering host CPU/RAM (and 8 cold recipe caches).
Instead, `dp_serve.sh`:
1. launches replica **r0 alone**, polls its `/health` until it's actually serving (graphs
   compiled → recipe cache populated),
2. then fans out **r1..rN-1 pointing at r0's recipe cache READ-ONLY** (`PT_HPU_RECIPE_CACHE_CONFIG
   = <r0path>,true,8192`) so they **skip compilation** and come up fast.
All replicas share identical shape/config, so r0's cache is valid for every replica. Set
`WARM_FIRST=0` to launch all at once (faster wall-clock if the cache is already warm from a prior run).

**Auto-sizing (default layout = `auto`).** With no explicit layout, `dp_serve.sh` measures the
model's on-disk weight footprint and picks the **smallest TP that fits one 96 GB card** (leaving
`KV_RESERVE_GB`, default 30, for KV/graph), then **DP = 8/TP** to fill all cards. Verified picks:
`16–23 GB INT4 → dp8_tp1` (8 copies), `Qwen3.6-35B 67 GB → dp4_tp2`, `397B INT4 211 GB → dp2_tp4`,
`GLM 376–392 GB → dp1_tp8` (too big to replicate). It prints the decision before serving.
**Caveat: auto sizes on _memory fit only_** — it does NOT know architectural TP limits (vision-dim /
head / expert divisibility, DSA, Mamba) or serve-time bugs; it warns on head/expert divisibility but
can't predict everything. If a model won't shard at the chosen TP, pass an explicit layout.

```bash
./dp_serve.sh qwen36                # auto: measures -> picks dp4_tp2 here, warm-first, LB on :8000
./dp_serve.sh qwen36 dp8_tp1        # force a layout (override auto)
KV_RESERVE_GB=20 ./dp_serve.sh qwen36   # smaller KV reserve -> auto may pick a larger DP
WARM_FIRST=0 ./dp_serve.sh qwen36 dp4_tp2   # skip warm-first (cache already warm)
./dp_serve.sh stop                  # tear down all replicas + balancer
```

**Benchmarking it (the real dp8_tp1 / dp4_tp2 / dp2_tp4 comparison).** The in-engine `dp8_tp1`
sweep variant timed out — replace it with a sweep that drives `dp_serve.sh` and benchmarks the
**balancer endpoint** (`BIND_IP:PORT`), not a single replica, so you measure aggregate throughput:

```bash
for L in dp8_tp1 dp4_tp2 dp2_tp4; do
  ./dp_serve.sh <model> "$L"                    # warm-first brings the fleet up
  # poll the LB until ready, then benchmark at concurrency 1/10/25/50/100/200:
  python bench/run_cycle.py --base http://$BIND_IP:$PORT --model <served-name> \
         --concurrency 1,10,25,50,100,200 --label "$L"
  ./dp_serve.sh stop
done
```
Compare against the monolithic **TP=8** baseline (`serve.sh <model>`). Expect `dp8_tp1` to win big
under high concurrency for a low-active model (no per-layer all-reduce; 8× the `max_num_seqs` slots),
while TP=8 stays the *fit* lever for models too big to replicate on one 96 GB card.

> Note: `max_num_seqs` in each model `.env` is **per replica**, so `dp8_tp1` gives `8 × max_num_seqs`
> aggregate slots. Watch per-card KV memory; a model that's ~70 GB on one card (e.g. Qwen3.6-35B)
> leaves little KV headroom at `dp8_tp1` — `dp4_tp2` (35 GB/card) is the safer high-concurrency layout.

---

# Part IV — SINGLE-STREAM "ZERO-WAIT" LEVER TRACKER (2026-06-15) ★ SOURCE OF TRUTH ★

The do-and-verify master list. Work top-down; **check off only when MEASURED** (util↑ / tok/s↑), not when coded.

**ROOT CAUSE — empirically nailed, DTYPE-INVARIANT.** Single-stream decode is **host-dispatch bound**: the
device idles ~85–98 % of every token (util 14–19 %, ~195 W/550 W) waiting on the host to launch the next op.
Proven NOT compute/quant by two natural experiments: (1) int4→fp8 didn't lift util; (2) **full fp8 (no dequant)
is still slow + low-util.** Op-level trace (`logs/prof/gaudi-server_*.pt.trace.json.gz`, parsed): **91,020 kernel
launches/token**, compute only 7.2 % (MoE matmul) + 0.1 % (attention); the rest is dispatch + ~90K casts +
per-expert op-multiplication. Fixing host-dispatch lifts **every** model (int4 + fp8) identically.

**THE RULE:** int4 or fp8 storage only (too big for bf16). Compute **fp8 always** (int4→fp8→fp8 MME). **bf16 is
BANNED** in storage AND as a dequant intermediate — LOUD-fail any bf16 path.

**STATUS:** ☐ todo · ⧖ in-progress · ✅ measured-win · ✗ ruled-out

## A. Host-dispatch levers — ranked by trace cost (fixing these = max util, no wait)
| # | lever | status | scope | fix (file:line) | gain | effort |
|---|---|---|---|---|---|---|
| **0** | **Lazy whole-graph capture (device-resident step)** | ☐ | universal | `PT_HPU_LAZY_MODE=1` → `wrap_in_hpu_graph` (hpu_model_runner.py:4533); collapses 91K launches → 1 replay. Needs M3 lazy-MoE coherence (mostly fixed) | **~0.18→~30 tok/s — PROVEN on Qwen-397B** | knob + coherence |
| **0b** | **int4→fp8 native MoE (no bf16)** | ☐ | quant | route-b → `fp8_gemm_v2` (m3_expert_runtime_patch.py:397); deletes `cast_u4_to_bf16` 32K/tok + 456 `convert_from_uint4` | shrinks captured graph; mandatory for fp8/giants | code |
| 1 | per-layer mark_step hook ×60/tok | ✗ knob-alone | universal | `is_lazy()` gate (hpu_model_runner.py:485) + `VLLM_CONFIG_HIDDEN_LAYERS` | **measured: util UNCHANGED alone** — only helps post-capture | knob+code |
| 2 | unconditional mark_step before **every all_reduce** (×178/tok) | ☐ | universal | `if is_lazy(): mark_step()` (hpu_communicator.py:42, mirror :81) | up to 178 host round-trips/tok | 1-line |
| 3 | sampler = separate eager island (fenced by 2 mark_steps) | ☐ | universal | fold greedy tail into the captured graph (all_greedy @ hpu_input_batch.py:759) | 6–10 % wall | code |
| 4 | rotary cos/sin rebuilt per-token | ☐ | universal | cache cos/sin; stop `_reset_rotary_cos_sin` (hpu_model_runner.py:804) | medium | code |
| 5 | swigluoai mark_steps ×2/layer (lazy) | ⧖ | m3 | gate on `is_lazy()` (m3_swigluoai_patch.py:248,270; FIX1 partial) | m3 host cut | code |
| 6 | input-prep host loop + `aten::empty` churn | ☐ | universal | double-buffer step N+1 prep over N replay | +2–5 % (grows w/ batch) | code |

## B. Cast swarm (~90K casts/token — the "more" beyond capture+fp8)
| lever | status | scope | fix | gain |
|---|---|---|---|---|
| **i64↔i32 churn (~5,500/tok, 30.9 ms)** | ☐ | universal | int32 at source: block_list/block_groups/positions/slot_mapping (hpu_model_runner.py:2191-2192, 1058-1059) | **~5–12 % wall — top non-int4 cast win** |
| f32↔bf16 router/RMSNorm round-trips (~4,000/tok, 7.8 ms) | ☐ | m3 | router gate in bf16 (minimax_m3.py:515,566); fold Gemma `1+w` into gamma at load | ~7.8 ms/tok |
| fp8-KV store→bf16 readback (per-step dynamic scale) | ☐ | universal | fp8 attention QK/AV (`enable_fp8_attn`) or static KV scale | ~3–5 ms/tok |
| `result_type` implicit-promotion (swigluoai float literals) | ☐ | m3 | make `+1.0`/`7.0`/`1.702` tensors of compute dtype | drops 2,785/tok promotion casts |

## C. Sampler / logits tail (host work after forward)
| lever | status | fix | gain |
|---|---|---|---|
| sampler metadata rebuilt every step | ☐ | cache for steady-state greedy (make_selective_sampling_metadata) | **6.1 ms/tok (~50 % of host-serial dead time)** |
| async output `.synchronize()+.tolist()` GIL block | ☐ | slice to valid rows + defer off critical path | 4.2 ms/tok |
| greedy still runs full softmax over [b,vocab] | ☐ | gate on `all_greedy`; skip logprob/temp | ~3 % wall, 1,300 launches/tok |
| lm_head = unquantized bf16 GEMM (leaves fp8 path) | ☐ | quantize lm_head to int4→fp8 (minimax_m3.py:1125) | 2–3 % wall + keeps fp8-only |

## D. MoE / expert-loop structure (m3, fp8-native)
| lever | status | fix | gain |
|---|---|---|---|
| batch swigluoai tail+combine across K experts | ☐ | stack y_k→[K,T,H], one `(gate·y).sum(0)` (m3_swigluoai_patch.py) | 4× fewer tail launches; ~10–15 % wall |
| fuse 7-op swigluoai tail → 1 TPC kernel | ☐ | `activation_mode` in OpBackend fused_expert (swigluoai-fused-kernel.md) | ~4.7 % kernels |
| native `mixture_of_experts.fp8` (fold α=1.702 into gate at load) | ☐ | load-time weight reparam → ONE fp8 MoE op replaces the K-loop | collapses 2K matmuls+K tails → 1 |

## E. DEFERRED bigger fixes (roadmap — not host-dispatch)
| item | status | what | gain |
|---|---|---|---|
| **MTP draft head** | ⧖ 74 shards dumped, head built+CPU-validated, **UNTRAINED** | train depth-1 → spec-decode (`minimax-m3/mtp/`, RESUME_CAMPAIGN.sh) | **~1.7× single-stream** (amortizes the fixed step; goal 90 % accept) |
| **MSA sparse attention** | ☐ kernel ~90 % built; head-collapse indexer fix pending | flip `HPU_MSA_READY` | concurrency + long-ctx |
| **concurrency** | partly ✅ | route-b fp8 **3.6–5× MEASURED** + APC + mns sweep + `--max-num-batched-tokens` | ~130–180 agg |
| **2nd-node PP-fp8** | ☐ | PP2 fp8 for the 1T giants (PP confirmed plumbed in vllm_gaudi) | fp8-fit for giants |

## F. META — trace tooling is broken on :next (worked around)
`/start_profile` 404s (core doesn't register it) and `VLLM_PROFILE_STEPS` isn't read on :next. **Workaround
shipped:** direct `torch.profiler` hook `minimax_m3._m3_trace_hook` — serve with `M3_TRACE=1 M3_TRACE_DIR=/profiler_out`,
then `touch /profiler_out/START_TRACE` mid-decode → exports a Kineto trace. Parse with `logs/prof/parse_trace.py`.

## MEASURED so far (2026-06-15)
- **int4 baseline (compile):** 5.8–7.6 tok/s warm, util 14–19 %.
- **LEVER-A** (`VLLM_CONFIG_HIDDEN_LAYERS=10`, cold cache): 3.32 tok/s, **util UNCHANGED 15–18 %** → mark_step knob alone ≠ the fix.
- **fp8 route-b concurrency:** 39.9 / 63.4 / 90 tok/s @ c=10/30/50 (**~3.6–5× int4**, MEASURED).
- **★ RANK-0 lazy capture — TESTED ON M3, ✗ (does NOT win for M3):**
  - lazy serve **coherent** ✅ (the swigluoai mark_steps are **droppable** — added `M3_NO_LAZY_MARKSTEP=1`, functional-ops replay-safe, no degeneration).
  - lazy 4.27 → 5.29 (swigluoai fences gated) → 3.61 (`CONFIG_HIDDEN_LAYERS=60`): **util stayed 18 % throughout.** Lazy ≈ compile.
  - **WHY:** (1) the **178 all-reduce mark_steps are load-bearing** (hpu_communicator.py:42 bridge workaround — can't remove) so the step never fully captures; (2) M3's compile baseline is already fine, so lazy has no pathological-eager to recover (Qwen's 3.84→29 was Qwen escaping eager, not a universal util jump). **M3 is bound by op COUNT, not execution mode.**

## THE PLAN (top-down) — CORRECTED after the RANK-0 test
RANK-0 lazy is the win for **clean-fused-MoE models** (Qwen/GLM). For **M3 it is NOT** — its swigluoai per-expert LOOP
makes the op-count the wall. So the M3 path is **op-count reduction**, not lazy:
1. **Fuse the swigluoai MoE → one op** (§D: native `mixture_of_experts.fp8` with α=1.702 folded into the gate at load) —
   collapses the K-expert loop that generates the bulk of the 91K launches. **This is the M3 critical path now.**
2. **Cast fixes (§B)** — i64↔i32 + f32↔bf16 source-dtype fixes (universal, cheap, dtype-invariant; cut ~10K launches/tok).
3. **Sampler/host (§C)** — fold greedy tail into the graph; cache sampler metadata (6 ms/tok).
4. **MTP (×1.7, §E)** stacks on the reduced base → toward 25–30.
Verify every lever by **util↑ + tok/s↑**. (RANK-0 lazy is parked for M3; revisit only if the MoE is fused first.)

---

# Part V — THE FAST-PATH PLAYBOOK + BOTTLENECK INVENTORY (2026-06-16) ★ THE RECIPE ★

**Bottom line:** every software blocker to "ultra-fast, max-util" is removed and proven end-to-end.
The ONLY thing left is **fabric flapping** (hardware — the new top tray / a reseat fixes it). Each new
model is now a sequence of one-command steps on a model-agnostic toolchain, not a from-scratch effort.

## The least-work recipe (per model, in order)
1. **Bring up** in its own container (coherence fixes baked into the env):
   `bash containers.sh on <key>`  (glm51ar-mtp-fp8 / minimax-m3-int4 / kimi-k2.7).
2. **Baseline** with the SHARED test (every model uses this):
   `python bench/serve_baseline.py --model <served> --conc 1,8,32,64`  → single+concurrency tok/s, util, spec-accept.
3. **MTP head = the single-stream lever (~1.6-1.7x), the biggest win after coherence.** Two paths:
   - **Native graft (fastest, best):** if the checkpoint ships a real MTP layer → fp8-quantize it
     (`tests/mtp_fp8_quant.py`, PER-TENSOR e4m3, keep norms/router/DSA-indexer fp16) → serve via the existing
     `--speculative-config deepseek_mtp` + `mtp_fp8_graft.py`. GLM-5.1 did this → 73% accept / 1.62 accept-len.
     ⚠ If the int4 export's MTP layer is DEAD (AutoRound drops/zeroes it — verify with `glm_cpu_mtp_eval.py`),
     graft the ORIGINAL bf16 layer quantized to fp8 (same provenance as the int4 main model; no bf16 served).
   - **From-scratch (M3, K2.7 — no native MTP):** `HPU_MTP_DUMP=1` serve → `bench/*bigfire` corpus →
     `head_factory/factory/train.py` (soft-KL, the DeepSeek-V3 MTP module shape) → `mtp_fp8_quant` → serve as eagle/mtp.
     Data scales accept hard (28%→57% from 18k→85k positions). Depth-1 caps at tau=1+alpha (~80%); native ~73%.
4. **Concurrency = the throughput lever** (Trident's regime). Aggregate scales to the fabric cap; run concurrent
   for max util. Single-stream MTP for the few non-stop streams. Both matter; both are built.

## Bottleneck inventory — EVERY blocker hit this campaign + the fix (all REMOVED except #LAST)
| Blocker | Status | Fix |
|---|---|---|
| GLM token-salad (incoherent) | ✅ FIXED | (1+weight) RMSNorm handling + HPU_GLM_ROPE_NEOX + HPU_SANITIZE_EMBED + the **fused int4→MME MoE kernel** (HPU_MOE_FUSED_OP, the real coherence fix) |
| int4 MoE dequant w/ bf16 intermediate | ✅ FIXED | fused OpBackend int4→MME (SRAM-resident, NO bf16); HPU_MOE_ROUTED_K=8 |
| Single-stream host-bound (V1 per-token round-trip, ~81% device idle) | ✅ MITIGATED | MTP spec-decode = N tokens per target round-trip (1.6-1.7x). The structural V1 IPC is amortized, not removed. |
| MTP dump corruption at concurrency (nan/garbage rows) | ✅ FIXED | finite-row filter in decode_dump_patch + trainer (drop nan hidden / out-of-vocab) |
| Native int4 MTP head DEAD (0% accept) | ✅ SOLVED | fp64 oracle proved the int4 export's layer is dead → fp8-graft the bf16 original |
| fp8 MTP head SERVING (the integration) | ✅ SOLVED | vllm-gaudi HPU Fp8 methods + PER-TENSOR scales + keep DSA-indexer/norms fp16 + scale-name=`<w>.weight_scale` |
| dtype rule (no bf16 stored/served) | ✅ HELD | head fp8 (per-tensor e4m3), model int4 (too big for fp8), fused kernel = no bf16 intermediate |
| toolchain per-model rework | ✅ REMOVED | model-agnostic: dump hook patches DeepseekV2/V3 + MiniMaxM3; trainer/quantizer/graft/baseline all generic |
| **FABRIC FLAPPING (scale-up port-9, intermittent)** | ❌ **THE ONLY ONE LEFT** | **HARDWARE — new top tray / reseat OAMs. No software bypass of an 8-way collective on a model that needs all 8 cards.** |

## Measured ceilings (so "ultra-fast" expectations are calibrated)
- **Single-stream:** ~33 tok/s clean-fabric base × ~1.6 MTP = **~55 tok/s** achievable. Degraded fabric = 1.3-7 tok/s (what you see when port-9 flaps).
- **Concurrency:** ~366 tok/s AGGREGATE clean-fabric (measured). Degraded caps ~11-27 regardless of concurrency (collective-latency-bound).
- **Util:** tracks fabric cleanliness — ~17% degraded ↔ ~40% clean. The device is starved waiting on the collective, NOT compute-bound (compute budget: dequant≈0, the kernel is fast).
- **The single highest-leverage action for the WHOLE box (all 3 models): clean the fabric.** Everything else is done.

## The model-agnostic toolchain (the "least work")
`bench/serve_baseline.py` (shared test, every model) · `decode_dump_patch.py` (dumps GLM/M3/K2.7) ·
`head_factory/factory/{train,head,eval_head,corpus}.py` · `tests/mtp_fp8_quant.py` · `mtp_fp8_graft.py` ·
per-model `head_factory/<m>/config.json` (glm51/m3/k27 all validated). New model = bring-up + 4 steps above.

## Part V addendum — TRAIN HEADS ON HPU, NOT CPU (2026-06-16) ★ ~80x unlock ★
The from-scratch MTP head trainer (head_factory/factory/train.py --device hpu) MUST run on the HPU MME, not
CPU: the frozen lm_head matmul (150k-200k vocab × hidden) is ~20 min/epoch on CPU but ~15s/epoch on one card
(~80x). Single card = NO fabric (no collectives) so it's immune to the flap. Recipe:
  docker run -d $(for d in /dev/accel/accel* /dev/infiniband/uverbs*; do echo --device=$d; done) \
    --cap-add=sys_nice --ipc=host --net=host --ulimit memlock=-1:-1 -e PT_HPU_LAZY_MODE=0 \
    -v $HOME/AI/head_factory:/opt/hf -v /mnt/fastmodels:/mnt/fastmodels:ro -v /mnt/data1:/mnt/data1 \
    --entrypoint bash vllm-gaudi:next -lc "python3 /opt/hf/factory/train.py --device hpu --config ... --dumps ..."
GOTCHA THAT COST AN ITERATION: you MUST mount /dev/infiniband/uverbs* too (not just /dev/accel/accel*) — the
device init does IBV/scaleout-NIC init even for a 1-card job; without uverbs -> "ibv initialization failed ->
synStatus=26 Device acquire failed". Eager (PT_HPU_LAZY_MODE=0). This makes M3/K2.7 heads minutes not hours.
