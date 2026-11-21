#!/bin/bash
# Manual data-parallel serving across the 8 Gaudi cards.
#
# WHY: vllm-gaudi's in-engine --data-parallel-size is NON-FUNCTIONAL (engine cores
# never init -- lazy-mode-gated, "CI-only"; the dp8_tp1 in-engine sweep timed out at
# 40min). And TP=8 on a small / low-active model is comms-bound (~343 tok/s, crashes
# at c>=100). So for throughput we REPLICATE the model N times (each pinned to its own
# card[s] via HABANA_VISIBLE_MODULES) and front them with an nginx least_conn balancer
# on the Tailscale endpoint. To the OpenAI client it's one model at BIND_IP:PORT.
#
# Usage: ./dp_serve.sh <model-key> [layout]
#   layout = auto (default) | dp8_tp1 | dp4_tp2 | dp2_tp4    # replicas x cards-each = 8
#   auto:    measure the model's on-disk weight footprint, pick the SMALLEST TP that fits
#            one 96GB card (leaving KV_RESERVE_GB headroom), then DP = 8/TP to fill all cards.
#            Prints its decision; override anytime with an explicit layout or KV_RESERVE_GB=NN.
#            CAVEAT: auto sizes on MEMORY FIT ONLY. It does NOT know architectural TP limits
#            (vision-dim/head/expert divisibility, DSA, Mamba) or serve-time bugs -- it warns on
#            head/expert divisibility but can't predict everything. If a model won't shard at the
#            chosen TP, pass an explicit layout.
#   dp8_tp1: 8 replicas, 1 card each, NO HCCL (most robust). Model must fit on 1 card.
#   dp4_tp2: 4 replicas, 2 cards each (HCCL within adjacent pairs 0,1/2,3/4,5/6,7).
#   dp2_tp4: 2 replicas, 4 cards each.
#
# WARM-FIRST (default ON, WARM_FIRST=1): launch replica r0 alone first to compile +
# populate the recipe cache, wait until it's serving, THEN fan out r1..rN-1 pointing at
# r0's cache READ-ONLY so they skip graph compilation. Avoids 8 simultaneous torch.compile
# passes thrashing host CPU/RAM. Set WARM_FIRST=0 to launch all replicas at once.
#
# Stop everything:  ./dp_serve.sh stop
set -Eeuo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/common.env"
DOCKER="${DOCKER:-docker}"
WARM_FIRST="${WARM_FIRST:-1}"
source "$HERE/gaudi_guard.sh"

# Exactly one controller may own the fixed replica names/ports. This prevents an
# interrupted older launcher from observing a newer r0 and fanning out a mixed pool.
LOCKFILE="$HERE/.dp_serve.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "another dp_serve controller is active (lock: $LOCKFILE, pid: $(cat "$LOCKFILE" 2>/dev/null || echo unknown))" >&2
  exit 75
fi
printf '%s\n' "$$" 1>&9

teardown() {
  safe_stop vllm-lb 30 || true
  for i in $(seq 0 7); do safe_stop "vllm-r$i" || true; done
  rm -f "$HERE/.dp_pool_state" "$HERE/.dp_pool_state.tmp"
}

[ "${1:-}" = "stop" ] && { teardown; echo "stopped all replicas + balancer"; exit 0; }

deadlock_guard || exit 86   # don't fan out 8 replicas into a wedged scheduler
MODEL=${1:?usage: dp_serve.sh <model-key> [dp8_tp1|dp4_tp2|dp2_tp4]   (or: stop)}
LAYOUT=${2:-auto}
ENVFILE="$HERE/models/$MODEL.env"
[ -f "$ENVFILE" ] || { echo "no config: $ENVFILE"; ls "$HERE/models" | sed 's/.env$//'; exit 1; }
source "$ENVFILE"
[ -n "${VLLM_ARGS_OVERRIDE:-}" ] && VLLM_ARGS="$VLLM_ARGS_OVERRIDE"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

# Match containers.sh plugin detection. A replica pool must not silently lose
# architecture-specific code just because it uses a different launcher.
USE_DSA=""; USE_M3=""; USE_KIMI=""; USE_KVQ=""
echo "$MODEL_ENV" | grep -q 'HPU_DSA_READY=1' && USE_DSA=1 || true
echo "$MODEL_ENV" | grep -qiE 'M3_SWIGLUOAI|M3_EXPERT_RUNTIME|M3_VLLM_DIR' && USE_M3=1 || true
echo "$MODEL" | grep -qi 'kimi' && USE_KIMI=1 || true
echo "$MODEL_ENV" | grep -qE 'USE_KVQ=1|HPU_KVQ_' && USE_KVQ=1 || true

# auto_layout: read the model's on-disk weight footprint + config and pick the SMALLEST
# tensor-parallel that fits one 96GB card (leaving KV_RESERVE_GB for KV cache/graph),
# then DP = 8/TP to fill all 8 cards for max throughput. Prints "<DP> <TP> <reason>".
# Override the headroom with KV_RESERVE_GB=NN (default 30). Falls back to TP=8/DP=1 if
# the model is too big to replicate.
KV_RESERVE_GB="${KV_RESERVE_GB:-30}"
auto_layout() {
  python3 - "$MODELS_DIR/$MODEL_PATH" "$KV_RESERVE_GB" <<'PY'
import json, os, sys, glob
d, kvres = sys.argv[1], float(sys.argv[2])
wb = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")))
gib = wb / 1024**3
usable = 96 - kvres                       # GB of HBM/card left for weights after KV/graph
tp = next((t for t in (1,2,4,8) if gib/t <= usable), 8)
dp = 8 // tp
heads = experts = None
try:
    cfg = json.load(open(os.path.join(d, "config.json"))); t = cfg
    for k in ("text_config","language_config","llm_config"):
        if isinstance(cfg.get(k), dict): t = cfg[k]; break
    heads = t.get("num_attention_heads")
    experts = t.get("num_experts") or t.get("n_routed_experts")
except Exception:
    pass
warn = ""
if tp>1 and heads and heads % tp:     warn += f"  !! heads({heads})%TP({tp})!=0 -> use an explicit layout"
if tp>1 and experts and experts % tp: warn += f"  !! experts({experts})%TP({tp})!=0"
fit = "too big to replicate -> single TP=8 instance (DP=1)" if dp==1 else f"{gib/tp:.0f}GB/card under {usable:.0f}GB budget"
print(f"{dp} {tp} {gib:.0f}GB weights -> TP={tp} ({fit}), DP={dp}{warn}")
PY
}

if [ "$LAYOUT" = "auto" ]; then
  command -v python3 >/dev/null || { echo "auto needs python3; pass an explicit layout"; exit 1; }
  read -r NREP TP AUTOREASON < <(auto_layout)
  [ -n "${NREP:-}" ] || { echo "auto_layout failed (no *.safetensors in $MODELS_DIR/$MODEL_PATH?)"; exit 1; }
  LAYOUT="dp${NREP}_tp${TP}"
  echo ">> auto: $AUTOREASON"
  echo ">> chosen layout: $LAYOUT  (override: pass dp8_tp1|dp4_tp2|dp2_tp4, or set KV_RESERVE_GB)"
else
  case "$LAYOUT" in
    dp8_tp1) NREP=8; TP=1 ;;
    dp4_tp2) NREP=4; TP=2 ;;
    dp2_tp4) NREP=2; TP=4 ;;
    *) echo "bad layout '$LAYOUT' (auto|dp8_tp1|dp4_tp2|dp2_tp4)"; exit 1 ;;
  esac
fi

# The model env sets --tensor-parallel-size (and maybe --enable-expert-parallel) for the
# monolithic case. Strip them; we set TP per-replica, and EP is meaningless at TP=1.
VLLM_ARGS=$(echo "$VLLM_ARGS" | sed -E 's/--tensor-parallel-size[= ]+[0-9]+//')
[ "$TP" = "1" ] && VLLM_ARGS=$(echo "$VLLM_ARGS" | sed -E 's/--enable-expert-parallel//')

PLUGIN_MOUNTS=""
PIP_EDITABLES="-e /opt/hpu-moe-ext"
[ -n "$USE_DSA" ] && { PLUGIN_MOUNTS="$PLUGIN_MOUNTS -v $HERE/hpu-dsa-ext:/opt/hpu-dsa-ext"; PIP_EDITABLES="$PIP_EDITABLES -e /opt/hpu-dsa-ext"; }
[ -n "$USE_M3" ] && { PLUGIN_MOUNTS="$PLUGIN_MOUNTS -v $HERE/hpu-m3-ext:/opt/hpu-m3-ext"; PIP_EDITABLES="$PIP_EDITABLES -e /opt/hpu-m3-ext"; }
[ -n "$USE_KIMI" ] && { PLUGIN_MOUNTS="$PLUGIN_MOUNTS -v $HERE/hpu-kimi-ext:/opt/hpu-kimi-ext"; PIP_EDITABLES="$PIP_EDITABLES -e /opt/hpu-kimi-ext"; }
[ -n "$USE_KVQ" ] && { PLUGIN_MOUNTS="$PLUGIN_MOUNTS -v $HERE/hpu-kvq-ext:/opt/hpu-kvq-ext"; PIP_EDITABLES="$PIP_EDITABLES -e /opt/hpu-kvq-ext"; }

# Stop any monolithic vllm-* container before reserving individual cards or
# binding the public port. Previously dp_serve could collide with production.
stop_non_replica_vllm() {
  local n
  while read -r n; do
    [ -z "$n" ] && continue
    case "$n" in
      vllm-r[0-7]|vllm-lb) ;;
      *)
        PREV_CONTAINERS="${PREV_CONTAINERS:+$PREV_CONTAINERS }$n"
        echo "Gracefully pausing previous service '$n' for DP deployment..."
        if ! $DOCKER stop -t 90 "$n" >/dev/null; then
          echo "ERROR: could not stop '$n' cleanly; refusing to launch replicas" >&2
          return 1
        fi
        ;;
    esac
  done < <($DOCKER ps --filter 'name=vllm-' --format '{{.Names}}')
}

PREV_CONTAINERS=""
ROLLBACK_ARMED=0
restore_previous() {
  local n
  [ -n "$PREV_CONTAINERS" ] || return 0
  echo "Restoring previous service(s): $PREV_CONTAINERS"
  for n in $PREV_CONTAINERS; do
    $DOCKER inspect "$n" >/dev/null 2>&1 && $DOCKER start "$n" >/dev/null || true
  done
}
rollback_pool() {
  local rc="${1:-1}"
  trap - ERR INT TERM
  if [ "$ROLLBACK_ARMED" = "1" ]; then
    echo "ROLLBACK: removing the incomplete DP pool" >&2
    teardown
    restore_previous
  fi
  exit "$rc"
}

cache_source_guard() {
  local cache_dir=$1 newest src mount_src
  local watch=("$HERE/hpu-moe-ext")
  [ -n "$USE_DSA" ] && watch+=("$HERE/hpu-dsa-ext")
  [ -n "$USE_M3" ] && watch+=("$HERE/hpu-m3-ext")
  [ -n "$USE_KIMI" ] && watch+=("$HERE/hpu-kimi-ext")
  [ -n "$USE_KVQ" ] && watch+=("$HERE/hpu-kvq-ext")
  # Also watch any host source mounted over vllm-gaudi itself (patched runners,
  # model adapters, attention seams, and other architecture-specific code).
  while read -r mount_src; do
    [ -e "$mount_src" ] && watch+=("$mount_src")
  done < <(printf '%s\n' ${EXTRA_MOUNTS:-} | grep -oE '[^ ]+:/opt/vllm-gaudi/vllm_gaudi[^ ]*' | sed 's/:.*//' | sort -u || true)
  [ -d "$cache_dir" ] || return 0
  newest=$(find "$cache_dir" -type f ! -name '.envknobs_hash' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk 'NR==1{print $2}')
  [ -n "$newest" ] || return 0
  for src in "${watch[@]}"; do
    if [ -e "$src" ] && find "$src" \( -name '*.py' -o -name '*.so' \) -newer "$newest" -print -quit 2>/dev/null | grep -q .; then
      echo "DP STALE-RECIPE GUARD: source newer than $cache_dir -> wiping recipes"
      find "$cache_dir" -mindepth 1 -maxdepth 1 ! -name '.envknobs_hash' -exec rm -rf {} + 2>/dev/null || true
      return 0
    fi
  done
}

DEV=""; for d in /dev/accel/accel* /dev/infiniband/uverbs*; do DEV="$DEV --device=$d"; done

# launch_replica <rank> <rw|ro> <cache-subpath-relative-to-/recipe_cache>
launch_replica() {
  local r=$1 mode=$2 cache=$3
  local start=$((r*TP)) mods=""
  local c; for c in $(seq 0 $((TP-1))); do mods="$mods,$((start+c))"; done
  mods=${mods#,}
  local port=$((8100+r))
  local ro=false; [ "$mode" = "ro" ] && ro=true
  local api_arg=""; [ -n "${API_KEY:-}" ] && api_arg="--api-key '$API_KEY'"
  if $DOCKER inspect "vllm-r$r" >/dev/null 2>&1; then
    echo "refusing to overwrite unexpected existing container vllm-r$r" >&2
    return 1
  fi
  [ "$mode" = "rw" ] && mkdir -p "$RECIPE_CACHE/$cache"
  echo "  replica r$r -> HABANA_VISIBLE_MODULES=[$mods] tp=$TP 127.0.0.1:$port cache=$cache ($mode)"
  # shellcheck disable=SC2086
  $DOCKER run -d --name "vllm-r$r" --restart=no \
    $DEV --cap-add=sys_nice --cap-add=SYS_PTRACE --ipc=host --net=host --ulimit memlock=-1:-1 \
    -e HABANA_VISIBLE_MODULES="$mods" \
    -e PT_HPU_RECIPE_CACHE_CONFIG="/recipe_cache/$cache,$ro,8192" \
    -e PYTHONDONTWRITEBYTECODE=1 -e HPU_MOE_EXT_CKPT="/models/$MODEL_PATH" \
    $MODEL_ENV \
    -v "$MODELS_DIR:/models:ro" -v "$RECIPE_CACHE:/recipe_cache" -v "$QUANT_DIR:/quant" \
    -v "$HERE/hpu-moe-ext:/opt/hpu-moe-ext" $PLUGIN_MOUNTS ${EXTRA_MOUNTS:-} \
    "$IMAGE" \
    bash -lc "pip install --no-deps $PIP_EDITABLES -q 2>&1 | tail -1; \
      exec vllm serve '/models/$MODEL_PATH' --served-model-name '$SERVED_NAME' \
        --host 127.0.0.1 --port $port $api_arg \
        --tensor-parallel-size $TP $VLLM_ARGS" >/dev/null
}

# poll a replica's /health until 200 (warmup compiles graphs -> can take many minutes)
wait_ready() {
  local port=$1 tries=$2 name=${3:-} i code
  for i in $(seq 1 "$tries"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null || true)
    [ "$code" = "200" ] && return 0
    if [ -n "$name" ] && ! $DOCKER ps --format '{{.Names}}' | grep -qx "$name"; then
      echo "replica $name exited before readiness" >&2
      return 1
    fi
    sleep 10
  done
  return 1
}

# Health alone can be a false positive if a stale/wrong process owns the port.
# Require each backend to advertise the exact model this pool intends to serve.
verify_backend_model() {
  local port=$1 body
  local auth=()
  [ -n "${API_KEY:-}" ] && auth=(-H "Authorization: Bearer $API_KEY")
  body=$(curl -fsS --max-time 10 "${auth[@]}" "http://127.0.0.1:$port/v1/models") || return 1
  printf '%s' "$body" | python3 -c 'import json,sys; want=sys.argv[1]; ids=[x.get("id") for x in json.load(sys.stdin).get("data",[])]; sys.exit(0 if want in ids else 1)' "$SERVED_NAME"
}

teardown
ROLLBACK_ARMED=1
trap 'rollback_pool $?' ERR
trap 'rollback_pool 130' INT TERM
stop_non_replica_vllm
R0CACHE="$MODEL/dp_${LAYOUT}/r0"
mkdir -p "$RECIPE_CACHE/$R0CACHE"
cache_source_guard "$RECIPE_CACHE/$R0CACHE"
envknob_guard "$RECIPE_CACHE/$R0CACHE" "$MODEL/$LAYOUT" "$MODEL_ENV" \
  "--tensor-parallel-size $TP $VLLM_ARGS"
echo "Launching $LAYOUT for $MODEL (warm_first=$WARM_FIRST)"
if [ "$WARM_FIRST" = "1" ] && [ "$NREP" -gt 1 ]; then
  launch_replica 0 rw "$R0CACHE"
  echo "warm-first: waiting for r0 to compile + serve (up to ~20min)..."
  if wait_ready 8100 120 vllm-r0; then
    echo "r0 ready -> fanning out r1..r$((NREP-1)) sharing r0 cache read-only (skip compile)"
    for r in $(seq 1 $((NREP-1))); do launch_replica "$r" ro "$R0CACHE"; done
  else
    echo "ABORT: r0 failed readiness; rolling back the replica pool" >&2
    $DOCKER logs --tail 80 vllm-r0 2>&1 | tail -40 >&2 || true
    rollback_pool 1
  fi
else
  for r in $(seq 0 $((NREP-1))); do launch_replica "$r" rw "$MODEL/dp_${LAYOUT}/r$r"; done
fi

# Do not expose a partially loaded pool. Every backend must be healthy first;
# otherwise roll the whole pool back instead of letting nginx spray 502s.
for r in $(seq 0 $((NREP-1))); do
  if ! wait_ready $((8100+r)) 120 "vllm-r$r"; then
    echo "ABORT: replica r$r failed readiness; rolling back all replicas" >&2
    $DOCKER logs --tail 80 "vllm-r$r" 2>&1 | tail -40 >&2 || true
    rollback_pool 1
  fi
  if ! verify_backend_model $((8100+r)); then
    echo "ABORT: replica r$r is healthy but does not advertise '$SERVED_NAME'" >&2
    $DOCKER logs --tail 50 "vllm-r$r" 2>&1 | tail -30 >&2 || true
    rollback_pool 1
  fi
  echo "  verified r$r: health=200 model=$SERVED_NAME"
done
for r in $(seq 0 $((NREP-1))); do $DOCKER update --restart unless-stopped "vllm-r$r" >/dev/null; done

# nginx least_conn balancer on the Tailscale endpoint -> the N replicas.
UPSTREAMS=""
for r in $(seq 0 $((NREP-1))); do UPSTREAMS="${UPSTREAMS}    server 127.0.0.1:$((8100+r)) max_fails=1 fail_timeout=15s;\n"; done
CONF="$HERE/.dp_nginx_${MODEL}.conf"
{
  echo "worker_processes auto;"
  echo "worker_rlimit_nofile 65535;"
  echo "events { worker_connections 8192; multi_accept on; }"
  echo "http {"
  echo "  access_log off;"
  echo "  sendfile off; tcp_nodelay on;"
  echo "  upstream vllm_dp {"
  echo "    least_conn;"
  echo -e "$UPSTREAMS"
  echo "  }"
  echo "  server {"
  echo "    listen ${BIND_IP}:${PORT} backlog=4096;"
  echo "    proxy_read_timeout 900s; proxy_send_timeout 900s;"
  echo "    proxy_next_upstream error timeout http_502 http_503 http_504;"
  echo "    proxy_next_upstream_tries 2;"
  echo "    location / {"
  echo "      proxy_pass http://vllm_dp;"
  echo "      proxy_http_version 1.1;"
  echo "      proxy_set_header Connection \"\";"
  echo "      proxy_request_buffering off;"
  echo "      proxy_buffering off;            # stream SSE tokens immediately"
  echo "      proxy_set_header Host \$host;"
  echo "    }"
  echo "  }"
  echo "}"
} > "$CONF"
$DOCKER run -d --name vllm-lb --restart unless-stopped --net=host \
  -v "$CONF:/etc/nginx/nginx.conf:ro" nginx:alpine >/dev/null
if ! wait_ready "$PORT" 12 vllm-lb; then
  echo "ABORT: balancer failed its final health check" >&2
  $DOCKER logs --tail 50 vllm-lb >&2 || true
  rollback_pool 1
fi
{
  echo "MODEL=$MODEL"
  echo "SERVED_NAME=$SERVED_NAME"
  echo "LAYOUT=$LAYOUT"
  echo "NREP=$NREP"
  echo "TP=$TP"
  echo "PREVIOUS=$PREV_CONTAINERS"
  echo "STARTED_UTC=$(date -u +%FT%TZ)"
} > "$HERE/.dp_pool_state.tmp"
mv "$HERE/.dp_pool_state.tmp" "$HERE/.dp_pool_state"
ROLLBACK_ARMED=0
trap - ERR INT TERM
echo
echo "$NREP replicas ($LAYOUT). Balancer: ${BIND_IP}:${PORT}  (served-model-name: $SERVED_NAME)"
echo "Status: docker ps   |   a replica's log: docker logs -f vllm-r0   |   stop: $0 stop"
