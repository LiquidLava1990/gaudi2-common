#!/bin/bash
# Start ONE model. Usage: ./serve.sh <model-key>   (model-key = file in models/ without .env)
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/common.env"
MODEL=${1:-}
[ -n "$MODEL" ] || { echo "usage: serve.sh <model-key>"; echo "available:"; ls "$HERE/models" | sed 's/.env$//'; exit 1; }
ENVFILE="$HERE/models/$MODEL.env"
[ -f "$ENVFILE" ] || { echo "no config: $ENVFILE"; echo "available:"; ls "$HERE/models" | sed 's/.env$//'; exit 1; }
source "$ENVFILE"
# benchmark override hooks (used by bench/run_cycle.sh + minimax-m3/bench/run_matrix.sh to sweep
# trick combos AND swap the served checkpoint without editing the model env file):
#   VLLM_ARGS_OVERRIDE    replaces VLLM_ARGS wholesale (the swept serve flags)
#   MODEL_ENV_EXTRA       appends docker -e flags (a later -e wins -> overrides an env-file -e)
#   MODEL_PATH_OVERRIDE   replaces MODEL_PATH (e.g. serve the int4 checkpoint instead of the bf16 one)
#   EXTRA_MOUNTS_OVERRIDE replaces EXTRA_MOUNTS (the bind for that checkpoint into /models)
[ -n "${VLLM_ARGS_OVERRIDE:-}" ] && VLLM_ARGS="$VLLM_ARGS_OVERRIDE"
[ -n "${MODEL_ENV_EXTRA:-}" ] && MODEL_ENV="$MODEL_ENV ${MODEL_ENV_EXTRA}"
[ -n "${MODEL_PATH_OVERRIDE:-}" ] && MODEL_PATH="$MODEL_PATH_OVERRIDE"
[ -n "${EXTRA_MOUNTS_OVERRIDE:-}" ] && EXTRA_MOUNTS="$EXTRA_MOUNTS_OVERRIDE"
# DOCKER can be "sudo docker" in automation before the docker-group login applies
DOCKER="${DOCKER:-docker}"
source "$HERE/gaudi_guard.sh"
# best-effort serialize so two launches (manual + systemd + bench loop) can't race into churn
LOCKFILE="${SERVE_LOCK:-/tmp/gaudi_serve.lock}"
if ( : >>"$LOCKFILE" ) 2>/dev/null; then
  exec 9>>"$LOCKFILE"; flock -w 300 9 || echo "WARN: serialize-lock wait timed out; proceeding" >&2
fi
# bail FAST if the box is already wedged in the affinity deadlock (don't compound it)
deadlock_guard || exit 86
# enforce one-model-at-a-time: GRACEFULLY stop the active model (never rm -f churn)
safe_stop "$CONTAINER" || { echo "could not free '$CONTAINER' safely; aborting (see above)"; exit 1; }
# device passthrough: 8 Gaudi compute nodes + the RDMA/IB verbs nodes the HCL layer
# needs (without uverbs* the synapse runtime fails 'ibv init' -> Device acquire failed)
DEV=""; for d in /dev/accel/accel* /dev/infiniband/uverbs* ; do DEV="$DEV --device=$d"; done
mkdir -p "$RECIPE_CACHE/$MODEL"
# STALE-RECIPE GUARD: recipe-cache validation is OFF (",false," below) for fast warm restarts,
# so a recipe minted before a plugin code change silently replays the OLD graph and any python
# fix never executes on device (cost ~10 serve cycles before this was caught — PROJECT_FINDINGS
# "Stale recipe cache"). Wipe THIS model's cache when any mounted plugin source (.py/.so) is
# newer than the newest cached recipe; stable models keep their warm cache.
# Sources watched: the always-mounted ext plugins PLUS any host code THIS model bind-mounts over
# the in-image vllm-gaudi plugin (/opt/vllm-gaudi/vllm_gaudi) via EXTRA_MOUNTS — the patched runner,
# the mmel_l6 engine seam, the eagle/tree proposers, the async _src/vllm_gaudi tree, etc. Without
# this, a recipe minted before one of those edits silently replays the stale graph and our patched
# code never runs on device. We key on the MOUNT DESTINATION (/opt/vllm-gaudi/vllm_gaudi), so it
# catches both vllm-gaudi-patch/*.py file-mounts AND whole-tree _src mounts regardless of source
# name. Scoped per-model: only paths this env actually mounts are added; stable models keep warm cache.
WATCH_SRCS=("$HERE/hpu-moe-ext/hpu_moe_ext" "$HERE/hpu-dsa-ext/hpu_dsa_ext")
# pull the host (SRC) side of every "-v SRC:/opt/vllm-gaudi/vllm_gaudi...[:opts]" token
for _ps in $(printf '%s\n' ${EXTRA_MOUNTS:-} | grep -oE '[^ ]+:/opt/vllm-gaudi/vllm_gaudi[^ ]*' | sed 's/:.*//' | sort -u); do
  [ -e "$_ps" ] && WATCH_SRCS+=("$_ps")
done
# newest RECIPE file (exclude the env-knob-guard sentinel dotfile: it is rewritten every serve, so
# counting it would make cache_ref always-newest and silently MASK a real source edit from this guard)
cache_ref=$(find "$RECIPE_CACHE/$MODEL" -type f ! -name '.envknobs_hash' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk 'NR==1' | cut -d' ' -f2-)
if [ -n "$cache_ref" ] && [ -n "$(find "${WATCH_SRCS[@]}" \
     \( -name '*.py' -o -name '*.so' \) -newer "$cache_ref" -print -quit 2>/dev/null)" ]; then
  echo "STALE-RECIPE GUARD: plugin source newer than $MODEL recipe cache -> wiping it (forces recompile)"
  rm -rf "${RECIPE_CACHE:?}/${MODEL:?}"/*
fi
# ENV-KNOB FINGERPRINT GUARD (companion to the file-based guard above; defined in common.env):
# the recipe cache is keyed by model only, with validation OFF, so flipping a compute-selecting env
# knob (HPU_MOE_*, HPU_LAYER_FUSE, HPU_AR_NO_MARKSTEP, MODERN_MOE_EXEC, MMEL_*, PT_HPU_LAZY_MODE,
# num_experts_per_tok/hf-overrides) against a warm cache would REPLAY THE OLD GRAPH and the change
# would silently never take effect. Fingerprint the compute knobs THIS serve actually passes (the
# RESOLVED MODEL_ENV after MODEL_ENV_EXTRA + VLLM_ARGS after VLLM_ARGS_OVERRIDE) and wipe iff that
# fingerprint changed since the last serve. Volatile knobs (timers/profilers/dumps/paths) are
# excluded so a benign env diff keeps the warm cache. Composes with the file guard (sentinel is a
# dotfile -> survives the rm -rf above; both guards only remove non-dotfile recipes).
envknob_guard "$RECIPE_CACHE/$MODEL" "$MODEL" "$MODEL_ENV" "$VLLM_ARGS"
# optional single-socket NUMA pinning (single-stream latency: keep the TP group + host
# input-prep on ONE socket so per-token all-reduces never cross UPI). Opt-in via model env:
#   VISIBLE_MODULES (e.g. 0,1,2,3) restricts to specific Gaudi modules (socket-local cards);
#   CPUSET_CPUS / CPUSET_MEMS pin host CPUs + memory to that socket's NUMA node.
# Unset => unchanged behavior (all 8 cards, no cpuset).
PIN_ARGS=""
[ -n "${CPUSET_CPUS:-}" ] && PIN_ARGS="$PIN_ARGS --cpuset-cpus=$CPUSET_CPUS"
[ -n "${CPUSET_MEMS:-}" ] && PIN_ARGS="$PIN_ARGS --cpuset-mems=$CPUSET_MEMS"
VIS_ENV="-e HABANA_VISIBLE_DEVICES=all"
[ -n "${VISIBLE_MODULES:-}" ] && VIS_ENV="-e HABANA_VISIBLE_MODULES=$VISIBLE_MODULES"
echo "Starting $MODEL ($SERVED_NAME) on $BIND_IP:$PORT ${VISIBLE_MODULES:+[modules=$VISIBLE_MODULES cpuset=${CPUSET_CPUS:-all}/mems=${CPUSET_MEMS:-all}]}..."
# shellcheck disable=SC2086
$DOCKER run -d --name "$CONTAINER" --restart "${RESTART_POLICY:-unless-stopped}" \
  $DEV $PIN_ARGS --cap-add=sys_nice --cap-add=SYS_PTRACE --ipc=host --net=host --ulimit memlock=-1:-1 \
  $VIS_ENV \
  -e PT_HPU_RECIPE_CACHE_CONFIG="/recipe_cache/$MODEL,false,8192" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HPU_MOE_EXT_CKPT="/models/$MODEL_PATH" \
  $MODEL_ENV \
  -v "$MODELS_DIR:/models:ro" -v "$RECIPE_CACHE:/recipe_cache" -v "$QUANT_DIR:/quant" \
  -v "$HERE/hpu-moe-ext:/opt/hpu-moe-ext" ${USE_DSA:+-v "$HERE/hpu-dsa-ext:/opt/hpu-dsa-ext"} \
  ${USE_M3:+-v "$HERE/hpu-m3-ext:/opt/hpu-m3-ext"} \
  ${USE_KIMI:+-v "$HERE/hpu-kimi-ext:/opt/hpu-kimi-ext"} \
  ${USE_KVQ:+-v "$HERE/hpu-kvq-ext:/opt/hpu-kvq-ext" -v "$HERE/mmel/turboquant_kv:/opt/turboquant_kv"} \
  ${EXTRA_MOUNTS:-} \
  "$IMAGE" \
  bash -lc "pip install --no-deps -e /opt/hpu-moe-ext ${USE_DSA:+-e /opt/hpu-dsa-ext} ${USE_M3:+-e /opt/hpu-m3-ext} ${USE_KIMI:+-e /opt/hpu-kimi-ext} ${USE_KVQ:+-e /opt/hpu-kvq-ext} -q 2>&1 | tail -1 ; \
    exec vllm serve '/models/$MODEL_PATH' --served-model-name '$SERVED_NAME' \
      --host '$BIND_IP' --port '$PORT' --api-key '$API_KEY' $VLLM_ARGS"
echo "Container '$CONTAINER' launched. Logs: docker logs -f $CONTAINER"
# NOTE: hpu-moe-ext plugin is mounted+installed for every model; it is INERT for
# bf16/compressed-tensors models (only patches GPTQ/AWQ MoE configs). GPTQ/AWQ
# models autodetect quant from config.json (no --quantization flag needed).
