# Gaudi 2 multi-model serving

8× Intel Gaudi 2 (HL-225H). One model at a time, OpenAI-compatible, bound to the Tailscale IP.

## Layout
- `common.env` — shared config (image, paths, Tailscale bind IP, port, **API key**, container name)
- `models/<key>.env` — per-model launch config (path, served-name, env, vLLM args). **Add a model = add one file.**
- `gate_zero.sh` — prove SynapseAI 1.24 drives the in-tree driver (run first)
- `Dockerfile.vllm-gaudi` + `build.sh` — build the `vllm-gaudi:1.24` serving image
- `serve.sh <key>` / `stop.sh` / `status.sh` — run / stop / inspect the single active model
- `../bench/` — benchmark harness (run_sweep.sh, run_cycle.sh, aggregate.py)

## One-model-at-a-time
`serve.sh` always runs under the single container name `vllm-active` and `rm -f`s any existing one
before starting, so two models can never be loaded simultaneously (also correct since each model
uses all 8 cards via TP=8).

## Bring-up order
1. `./gate_zero.sh`            # validate runtime on the in-tree driver
2. `./build.sh`               # build vllm-gaudi:1.24
3. `./serve.sh qwen3-235b`    # or qwen35-397b   (only after the model finished downloading)
4. `./status.sh`              # endpoint + active model
5. remote use: `OPENAI_BASE_URL=http://<SERVER_IP>:8000/v1` with the key in common.env

## Models
- `qwen3-235b`  → Qwen3-235B-A22B-Instruct-2507-FP8  (FP8, ~235 GB)
- `qwen35-397b` → Qwen3.5-397B-A17B-GPTQ-Int4         (INT4 MoE, ~200 GB)

## Benchmarking
`bash ../bench/run_sweep.sh qwen3-235b`  → cycles every trick variant @ concurrency 1/5/10/25/50,
writes a report per cycle to `../bench/reports/`, then `python ../bench/aggregate.py` builds the
master table + synopsis.
