# gaudi2-common

Shared build parts for serving large language models with vLLM on an 8-card Intel Gaudi 2 server. This repo holds the Dockerfiles, launch and guard scripts, vLLM-Gaudi patches, and three out-of-tree vLLM plugins that the per-model repos reuse. It also holds kernel design notes and benchmark and tuning write-ups.

It does not contain model weights or per-model launch configs. Those live in the model repos listed below.

## Who this is for

- People running vLLM + the vllm-gaudi plugin on Gaudi 2 hardware who want a working launch script, image recipe and host guards.
- People porting large MoE models (GLM, Qwen, MiniMax) to Gaudi 2 who want the patches and the debugging notes.
- Maintainers of the model repos that depend on this one.

## Versions tested

Everything below is based on the live champion stack.

| Item | Value |
|---|---|
| Accelerators | 8x Intel Gaudi 2 HL-225, 96 GB HBM each |
| Host | 2-socket Xeon, about 500 GB RAM |
| Host driver | 1.24.2 (must be at or above the container's SynapseAI version) |
| Card firmware | 62.6.2 |
| Container stack | 1.24.2 |
| Base image | `vault.habana.ai/gaudi-docker/1.24.2/ubuntu24.04/habanalabs/pytorch-installer-2.13.0` |
| Serving stack | vLLM plus the vllm-gaudi plugin |

Champion per model repo:

| Repo | Champion |
|---|---|
| gaudi2-glm-5.3-flash | promote-r4h kit, cut r4h8 (`champion-inputs/`); image `claude-flash-dual-graphs-c1d:20261112-r4h8` |
| gaudi2-glm-5.2 | `configs/glm52-champion-ablid-maxctx-offload.env` |
| gaudi2-minimax-m3 | See that repo's README (champion being chosen) |
| gaudi2-qwen3.6-27b | See that repo's README (champion being chosen) |
| gaudi2-glm-5.3-full | See that repo's README |

The Dockerfiles, `build.sh` and the TPC build comment in this repo still pin base `1.24.0` (PyTorch 2.9 / 2.10). The champion runs 1.24.2, so bump the `FROM` line to the base image above before building.

## Repo map

| Path | What it is |
|---|---|
| `Dockerfile.vllm-gaudi` | vLLM-Gaudi image on the upstream PyTorch base; pins vLLM to the commit the plugin was validated against |
| `Dockerfile.vllm-gaudi-lazy` | Same build on the Habana PyTorch fork base, which supports `PT_HPU_LAZY_MODE=1` |
| `Dockerfile.vllm-gaudi-lazy-0.21` | Pinned, reproducible variant (vllm-gaudi `v0.21.0` plus its matching vLLM commit) |
| `build.sh` | Builds the `vllm-gaudi:1.24` image from `Dockerfile.vllm-gaudi` |
| `gate_zero.sh` | Checks that the container userspace can drive the host driver and run an HPU op. Run first |
| `common.env` | Shared settings: image, model and cache paths, bind address, port, container name, optional API key |
| `serve.sh` | Starts one model by key (reads `models/<key>.env`), with recipe-cache guards |
| `containers.sh` | Container launch helper that uses the same plugin install and mounts |
| `dp_serve.sh` | Runs N replicas on disjoint card sets behind an nginx least_conn balancer (layouts `dp8_tp1`, `dp4_tp2`, `dp2_tp4`) |
| `stop.sh`, `status.sh` | Stop the active container gracefully; show container, endpoint and available configs |
| `snap.sh`, `fastcopy.sh` | Local commit checkpoint helper; parallel copy for large model files |
| `autoconfig.py`, `autoconfig_test.py`, `allocate_gaudi_port.py` | Config generation, its test, and port allocation |
| `gaudi_guard.sh` | Shared safety functions sourced by the launch and stop scripts |
| `shutdown_guard.sh`, `gaudi-shutdown-guard.service` | systemd unit and script that free vLLM containers and the data mount at shutdown |
| `tools/health/` | `gaudi_health_check.sh`: reports crash records, reboots and new BMC event-log entries since the last run |
| `hpu-moe-ext/`, `hpu-dsa-ext/`, `hpu-kvq-ext/` | Out-of-tree vLLM plugins (see below) |
| `vllm-gaudi-patch/` | Patched vllm-gaudi files (attention, platform, worker, offload, sampling) and `verify_fp8kv_needle.sh`, a long-context recall check |
| `patches/` | `hpu_awq.py`, `hpu_gptq.py` quantization patches |
| `dsa_src/` | Sources for DeepSeek-style sparse attention and MLA (indexer, attention, ops) |
| `csengine/` | Design and proof-of-concept for submitting decode work through the habanalabs command-submission ioctl; has its own README |
| `Gaudi2_Kernel/` | Per-model bring-up notes, kernel design docs and `Gaudi2_Optimization_Guide.md` |
| `head_factory/` | One config-driven tool that trains per-model MTP draft heads; has its own README |
| `abliterate/` | `abliterate_oproj.py`, a script that edits attention o_proj weights for GLM-5.2 |
| `download/` | Model download scripts (Hugging Face CLI based) |
| `qual/` | A `habanalabs-qual` Dockerfile and a CPU-only test of routed-expert index logic |
| `*.md` at top level | Deployment notes, runbooks, plans and status logs (`DEPLOYMENT.md`, `SPEED_RUNBOOK.md`, `GAUDI_PORT_REGISTRY.md`, others) |
| `NOTES.md` | The earlier README, kept for reference. Some of it is out of date |

## Getting started

### 1. Check the host

- The host driver is 1.24.2 or newer, and not older than the container's SynapseAI version.
- After installing a driver, rebuild the initramfs so the boot image loads the matching `habanalabs` module.
- Card firmware is 62.6.2.
- Confirm all eight cards are visible: `ls /dev/accel/accel*`. `gate_zero.sh` passes `/dev/accel/accel*` and `/dev/infiniband/uverbs*` into the container; without the uverbs nodes the runtime fails to initialize.
- Run `./gate_zero.sh` and fix anything it reports before building.

### 2. Build the base image

Edit the `FROM` line in `Dockerfile.vllm-gaudi` (or a lazy variant) to the 1.24.2 / PyTorch 2.13.0 base image listed under Versions tested, then:

```bash
./build.sh
```

The build clones vllm and vllm-gaudi and compiles vLLM, so it takes a while. `build.sh` writes a log under `$HOME/AI/logs/`; create that folder or edit the path first.

### 3. Launch a model

Model configs live in the model repos, not here: `configs/*.env` in gaudi2-glm-5.2, gaudi2-qwen3.6-27b and gaudi2-glm-5.3-full, and `models/*.env` in gaudi2-minimax-m3. `serve.sh` only reads `gaudi2-common/models/<key>.env`, so copy or symlink the config you want into that folder. Set the paths and bind address in `common.env`, then run (example uses gaudi2-glm-5.2; adjust the repo and file name):

```bash
git clone https://github.com/LiquidLava1990/gaudi2-glm-5.2.git
mkdir -p gaudi2-common/models
ln -s "$PWD/gaudi2-glm-5.2/configs/glm52.env" gaudi2-common/models/glm52.env   # or: cp
cd gaudi2-common
./serve.sh glm52
./status.sh
./stop.sh
```

GLM-5.3-Flash is the exception: gaudi2-glm-5.3-flash boots through its own promote-r4h kit, not `serve.sh`.

- `serve.sh` uses one fixed container name, `vllm-active`, so only one model runs at a time.
- `API_KEY` in `common.env` is empty by default, which means the server runs with no authentication. Set `API_KEY` yourself, or keep the server off the public internet.
- Calls go to `http://<SERVER_IP>:<PORT>/v1` (OpenAI-compatible).
- `serve.sh` clears a model's recipe cache when plugin source is newer than the cache, or when a compute-affecting environment setting changed. Otherwise a warm cache would replay an old graph and the change would not take effect.
- Optional `VISIBLE_MODULES`, `CPUSET_CPUS` and `CPUSET_MEMS` settings in a model env file pin a model to chosen cards and one NUMA node.

## Custom extensions

Each extension is a Python package with its own `pyproject.toml`. It registers through the `vllm.general_plugins` entry point and does not edit installed vLLM files. Load order is by entry-point name: stock vllm-gaudi, then `04.hpu_moe_ext`, `05.hpu_dsa_ext`, `06.hpu_kvq_ext`, so later ones override earlier ones.

| Extension | What it does |
|---|---|
| `hpu-moe-ext` | Adds a WNA16 GPTQ/AWQ FusedMoE method for Gaudi, reusing `VllmMixtureOfExpertsOpWNA16`, so partly quantized GPTQ/AWQ MoE checkpoints serve on vllm-gaudi. Also holds TPC kernel sources (`tpc_kernel/`), a busy-poll helper and sample env files |
| `hpu-dsa-ext` | DeepSeek Sparse Attention overlay for vllm-gaudi, for GLM-5.1 (GlmMoeDsa) and DeepSeek-V3.2. See `BUILD_PLAN.md` and `IMPLEMENTATION.md` |
| `hpu-kvq-ext` | Sub-fp8 KV cache (fp8 keys, int4 values, Walsh-Hadamard, no calibration) for GQA/MHA models. MLA models keep `fp8_inc`. Gated by `HPU_KVQ_READY` |

Build and install:

- Python plugins: `serve.sh` mounts each extension at `/opt/<name>` and runs `pip install --no-deps -e` on it inside the container. `hpu-moe-ext` is always installed. Set `USE_DSA=1` or `USE_KVQ=1` in the model env file to add the other two. To install by hand inside the image: `pip install --no-deps -e hpu-moe-ext`.
- TPC kernels (`hpu-moe-ext/tpc_kernel/build.sh`): build inside the Habana dev image with no HPU devices passed in. The script's header lists the toolchain (`tpc-clang`, `llvm-objcopy`) and its outputs. It produces a TPC object, a perf-lib `.so` and a PyTorch custom-op `.so`.

`serve.sh` also refers to extensions that are not in this repo:

- `hpu-m3-ext` is in gaudi2-minimax-m3.
- `mmel/turboquant_kv` (used with `USE_KVQ`) is not published; gaudi2-glm-5.2's `mmel/` folder does not contain it.
- `hpu-kimi-ext` is for an older model that is not published. Treat the `USE_KIMI` code path as unused.

## Host stability tips

These come from comments in the scripts and notes in this repo.

- Stop containers with `./stop.sh`, not `docker rm -f`. On the 6.8.12 kernel with habanalabs DKMS, vllm-gaudi pins its eight worker processes' CPU affinity at start. If that overlaps a force-killed process still tearing down, the tasks can block uninterruptibly, keep the cards, and only a reboot clears it.
- `gaudi_guard.sh` refuses to launch if a vllm or python worker is stuck in D state (override with `SKIP_DEADLOCK_GUARD=1` only if the box is healthy), and stops containers gracefully and waits for release.
- Do not churn containers rapidly.
- A Gaudi card is acquired by one process. Two models cannot share cards. `dp_serve.sh` splits cards into separate `HABANA_VISIBLE_MODULES` sets instead.
- Install `gaudi-shutdown-guard.service` so a reboot does not hang while unmounting the data disk. The unit's `ExecStop` path is a placeholder; point it at your copy of `shutdown_guard.sh`. The script does nothing unless the system is really shutting down.
- Run `tools/health/gaudi_health_check.sh` periodically. It needs passwordless sudo for `ipmitool` and pstore access.
- For large copies use `fastcopy.sh`. Its header says single-stream `rsync` stays near 150 MB/s while parallel `cp` reaches 2.5 to 3 GB/s.

## Model repos

| Repo | Link |
|---|---|
| gaudi2-glm-5.3-flash | https://github.com/LiquidLava1990/gaudi2-glm-5.3-flash |
| gaudi2-glm-5.3-full | https://github.com/LiquidLava1990/gaudi2-glm-5.3-full |
| gaudi2-glm-5.2 | https://github.com/LiquidLava1990/gaudi2-glm-5.2 |
| gaudi2-minimax-m3 | https://github.com/LiquidLava1990/gaudi2-minimax-m3 |
| gaudi2-qwen3.6-27b | https://github.com/LiquidLava1990/gaudi2-qwen3.6-27b |

## Contributing and issues

Open an issue at https://github.com/LiquidLava1990/gaudi2-common/issues for bugs or questions. For a bug report, include driver and firmware versions, the base image tag, the model key, and the output of `gate_zero.sh`. Pull requests are welcome. Keep paths, addresses and keys out of committed files and use placeholders such as `<SERVER_IP>`.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights are not included and keep their own licenses.
