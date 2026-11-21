#!/bin/bash
# Validate hpu-moe-ext end-to-end on a small GPTQ MoE at TP=1.
# Bind-mounts the plugin (editable install) so source edits apply each run.
set -uo pipefail
MODEL=${1:-Qwen3.5-35B-A3B-GPTQ-Int4}
TP=${2:-1}
EXTRA=${3:-}
NAME=vllm-test
DOCKER="sudo docker"
source "$(dirname "$(readlink -f "$0")")/gaudi_guard.sh"
deadlock_guard || exit 86
safe_stop "$NAME" || { echo "could not free $NAME safely; aborting"; exit 1; }
DEV=""; for d in /dev/accel/accel* /dev/accel/accel_controlD* /dev/infiniband/uverbs*; do DEV="$DEV --device=$d"; done

# shellcheck disable=SC2086
sudo docker run -d --name "$NAME" \
  $DEV --cap-add=sys_nice --net=host --ipc=host --ulimit memlock=-1:-1 \
  -e PT_HPU_LAZY_MODE=0 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PT_HPU_RECIPE_CACHE_CONFIG="/recipe_cache/$MODEL,false,8192" \
  -v $HOME/AI/hpu-moe-ext:/opt/hpu-moe-ext \
  -v /mnt/fastmodels:/models:ro \
  -v /mnt/data1/recipe_cache:/recipe_cache \
  vllm-gaudi:1.24 \
  bash -lc "pip install --no-deps -e /opt/hpu-moe-ext -q 2>&1 | tail -1 ; \
    exec vllm serve /models/$MODEL --served-model-name testmodel \
      --host 127.0.0.1 --port 8000 \
      --tensor-parallel-size $TP --max-model-len 8192 --max-num-seqs 16 \
      --gpu-memory-utilization 0.9 --trust-remote-code --enforce-eager $EXTRA"
echo "launched $NAME (model=$MODEL TP=$TP). logs: docker logs -f $NAME"
