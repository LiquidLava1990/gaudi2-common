#!/bin/bash
# Per-model DEDICATED containers — no shared `vllm-active`, so configs/plugins never bleed between
# models (the contamination that muddied measurements). Each model key -> container `vllm-<key>` +
# its own recipe cache `/recipe_cache/<key>`. The box holds ONE model's weights at a time, so `on`
# stops every other vllm-* first. Containers are CREATED STOPPED ("made") and only run when activated.
#
# Usage:
#   ./containers.sh make <key> [<key> ...]   # docker create (STOPPED) clean container(s), ready to start
#   ./containers.sh on   <key>               # stop all vllm-*, (re)create from env, START <key>
#   ./containers.sh off  [<key>]             # stop <key> (or ALL vllm-*) gracefully
#   ./containers.sh list                     # show all vllm-* + state
#
# Plugins are AUTO-DETECTED from the model env (USE_DSA<-HPU_DSA_READY=1, USE_M3<-M3_* markers,
# USE_KIMI<-key contains kimi) so a model always gets exactly the plugins it needs and no others.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/common.env"
DOCKER="${DOCKER:-docker}"

_cfg() { # source a model env into per-model vars + detect plugin needs
  local key="$1"; local envf="$HERE/models/$key.env"
  [ -f "$envf" ] || { echo "no config: $envf  (available: $(ls "$HERE/models" | sed 's/.env$//' | tr '\n' ' '))"; exit 1; }
  IMAGE=""; MODEL_PATH=""; SERVED_NAME=""; MODEL_ENV=""; VLLM_ARGS=""; EXTRA_MOUNTS=""
  source "$envf"
  [ -n "${VLLM_ARGS_OVERRIDE:-}" ] && VLLM_ARGS="$VLLM_ARGS_OVERRIDE"        # optimization-sweep hooks
  [ -n "${MODEL_ENV_EXTRA:-}" ] && MODEL_ENV="$MODEL_ENV ${MODEL_ENV_EXTRA}"
  # ABLITERATED=1 : run ANY GLM-5.2 champion on the refusal-removed graft. Only o_proj (attention out)
  # differs from the stock graft, so this is a pure drop-in -- the int4 experts / W4A8 / KV quant / drafter
  # are byte-identical. Appends an override -e (docker: last -e wins) only for envs that use the graft.
  if [ "${ABLITERATED:-0}" = "1" ] && echo "$MODEL_ENV" | grep -q 'glm52-fp8-other'; then
    MODEL_ENV="$MODEL_ENV -e HPU_FP8_OTHER_GRAFT=/models/glm52-fp8-other-abliterated"
    echo "[ABLITERATED] $key -> HPU_FP8_OTHER_GRAFT=/models/glm52-fp8-other-abliterated (perf-identical)"
  fi
  NAME="vllm-$key"
  USE_DSA=""; USE_M3=""; USE_KIMI=""
  echo "$MODEL_ENV" | grep -q 'HPU_DSA_READY=1' && USE_DSA=1
  echo "$MODEL_ENV" | grep -qiE 'M3_SWIGLUOAI|M3_EXPERT_RUNTIME|M3_VLLM_DIR' && USE_M3=1
  echo "$key" | grep -qi 'kimi' && USE_KIMI=1
  return 0   # detection greps return non-zero on no-match; under set -e that would abort _create
}

_create() { # docker create (STOPPED) the clean container for <key>
  local key="$1"; _cfg "$key"
  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  mkdir -p "$RECIPE_CACHE/$key"
  # STALE-RECIPE GUARD: recipe validation is OFF (",false," below) so a recipe minted before a plugin
  # code change replays the OLD graph silently (plugin .py/.so fix never executes on device). Wipe THIS
  # model's cache when any mounted plugin source is newer than its newest cached recipe.
  # Watched sources = the always-mounted ext plugins PLUS any host code THIS model bind-mounts over
  # the in-image vllm-gaudi plugin (/opt/vllm-gaudi/vllm_gaudi) via EXTRA_MOUNTS — patched runner,
  # mmel_l6 seam, eagle/tree proposers, async _src/vllm_gaudi tree, etc. Else a recipe minted before
  # one of those edits silently replays the stale graph and our patched code never runs on device.
  # Keyed on the mount DESTINATION so it catches vllm-gaudi patch mounts plus the MiniMax-M3 model/runtime
  # trees imported from /opt/m3-{root,vllm}. Scoped: only paths this env actually mounts are watched.
  local _watch=("$HERE/hpu-moe-ext" "$HERE/hpu-dsa-ext" "$HERE/hpu-m3-ext") _ps
  for _ps in $(printf '%s\n' ${EXTRA_MOUNTS:-} | grep -oE '[^ ]+:/opt/(vllm-gaudi/vllm_gaudi|m3-root|m3-vllm)[^ ]*' | sed 's/:.*//' | sort -u); do
    [ -e "$_ps" ] && _watch+=("$_ps")
  done
  # newest RECIPE file (exclude the env-knob-guard sentinel dotfile: rewritten every serve, so
  # counting it would make cref always-newest and silently MASK a real source edit from this guard)
  local cref; cref=$(find "$RECIPE_CACHE/$key" -type f ! -name '.envknobs_hash' -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk 'NR==1' | cut -d' ' -f2-)
  if [ -n "$cref" ] && [ -n "$(find "${_watch[@]}" \( -name '*.py' -o -name '*.so' \) -newer "$cref" -print -quit 2>/dev/null)" ]; then
    echo "  STALE-RECIPE GUARD: plugin source newer than $key cache -> wiping (forces recompile)"
    rm -rf "${RECIPE_CACHE:?}/${key:?}"/* 2>/dev/null || true
  fi
  # ENV-KNOB FINGERPRINT GUARD (companion to the file-based guard above; defined in common.env):
  # validation is OFF + cache keyed by model only, so flipping a compute-selecting env knob
  # (HPU_MOE_*, HPU_LAYER_FUSE, HPU_AR_NO_MARKSTEP, MODERN_MOE_EXEC, MMEL_*, PT_HPU_LAZY_MODE,
  # num_experts_per_tok/hf-overrides) against a warm cache REPLAYS THE OLD GRAPH and the change
  # silently never takes effect. _cfg already resolved MODEL_ENV (incl. MODEL_ENV_EXTRA) + VLLM_ARGS
  # (incl. VLLM_ARGS_OVERRIDE), so fingerprint those and wipe iff the compute-knob set changed since
  # the last serve. Volatile knobs are excluded -> benign env diffs keep the warm cache. Composes
  # with the file guard (sentinel is a dotfile, survives the rm -rf above; neither double-wipes).
  envknob_guard "$RECIPE_CACHE/$key" "$key" "$MODEL_ENV" "$VLLM_ARGS"
  local DEV=""; for d in /dev/accel/accel* /dev/infiniband/uverbs*; do [ -e "$d" ] && DEV="$DEV --device=$d"; done
  # shellcheck disable=SC2086
  $DOCKER create --name "$NAME" \
    $DEV --cap-add=sys_nice --cap-add=SYS_PTRACE --ipc=host --net=host --ulimit memlock=-1:-1 \
    -e HABANA_VISIBLE_DEVICES=all \
    -e PT_HPU_RECIPE_CACHE_CONFIG="/recipe_cache/$key,false,8192" \
    -e PYTHONDONTWRITEBYTECODE=1 -e HPU_MOE_EXT_CKPT="/models/$MODEL_PATH" \
    $MODEL_ENV \
    -v "$MODELS_DIR:/models:ro" -v "$RECIPE_CACHE:/recipe_cache" -v "$QUANT_DIR:/quant" \
    -v "$HERE/hpu-moe-ext:/opt/hpu-moe-ext" ${USE_DSA:+-v "$HERE/hpu-dsa-ext:/opt/hpu-dsa-ext"} \
    ${USE_M3:+-v "$HERE/hpu-m3-ext:/opt/hpu-m3-ext"} ${USE_KIMI:+-v "$HERE/hpu-kimi-ext:/opt/hpu-kimi-ext"} \
    ${USE_KVQ:+-v "$HERE/hpu-kvq-ext:/opt/hpu-kvq-ext"} \
    ${EXTRA_MOUNTS:-} \
    "$IMAGE" \
    bash -lc "pip install --no-deps -e /opt/hpu-moe-ext ${USE_DSA:+-e /opt/hpu-dsa-ext} ${USE_M3:+-e /opt/hpu-m3-ext} ${USE_KIMI:+-e /opt/hpu-kimi-ext} ${USE_KVQ:+-e /opt/hpu-kvq-ext} -q 2>&1 | tail -1 ; exec vllm serve '/models/$MODEL_PATH' --served-model-name '$SERVED_NAME' --host '$BIND_IP' --port '$PORT' ${API_KEY:+--api-key '$API_KEY'} $VLLM_ARGS" >/dev/null
  echo "made (STOPPED): $NAME  [DSA=${USE_DSA:-0} M3=${USE_M3:-0} KIMI=${USE_KIMI:-0}]  served=$SERVED_NAME  cache=/recipe_cache/$key"
}

_stop_all() {
  local running; running=$($DOCKER ps --filter 'name=vllm-' --format '{{.Names}}')
  [ -z "$running" ] && return 0
  echo "$running" | while read -r n; do [ -n "$n" ] && { echo "stopping $n ..."; $DOCKER stop -t 40 "$n" >/dev/null 2>&1 || true; }; done
}

# ============================================================================================
# DEPLOY/MINT ROBUSTNESS HELPERS  (used only by the `deploy`/`auto` arms below; inert until called).
# The repack cache is a per-rank-per-layer grid of `rank{R}_layer{L}.safetensors` shards (see
# hpu-moe-ext/repack_cache.py). A CLEAN mint = every (rank,layer) cell present, i.e. a full rectangular
# ranks×layers grid. autoconfig flips repack_mode to 'load' as soon as ANY shard exists, so a partial /
# crashed mint would be silently LOADed -> silent corruption. We own a completion sentinel (.mint_complete
# holding the verified shard count) that gates the load path: no valid sentinel => treat as mint-needed.
# ============================================================================================
_repack_present() { # echo count of canonical rank*_layer*.safetensors shards in <dir> (0 if none/missing)
  [ -d "$1" ] || { echo 0; return 0; }
  find "$1" -maxdepth 1 -type f -name 'rank*_layer*.safetensors' 2>/dev/null | wc -l | tr -d ' '
}
_repack_grid_ok() { # 0 iff <dir> holds a full rectangular ranks×layers shard grid (no gaps), non-empty
  local dir="$1" present ranks layers
  [ -d "$dir" ] || return 1
  present=$(_repack_present "$dir")
  [ "$present" -gt 0 ] 2>/dev/null || return 1
  ranks=$(find "$dir" -maxdepth 1 -type f -name 'rank*_layer*.safetensors' -printf '%f\n' 2>/dev/null \
            | sed -n 's/^rank\([0-9][0-9]*\)_layer[0-9][0-9]*\.safetensors$/\1/p' | sort -un | wc -l)
  layers=$(find "$dir" -maxdepth 1 -type f -name 'rank*_layer*.safetensors' -printf '%f\n' 2>/dev/null \
            | sed -n 's/^rank[0-9][0-9]*_layer\([0-9][0-9]*\)\.safetensors$/\1/p' | sort -un | wc -l)
  [ "$ranks" -gt 0 ] && [ "$layers" -gt 0 ] && [ "$present" -eq "$((ranks * layers))" ]
}
_repack_grid_report() { # human-readable diagnostic of the shard grid (for LOUD failures)
  local dir="$1" present ranks layers
  [ -d "$dir" ] || { echo "        (repack cache dir missing: $dir)"; return 0; }
  present=$(_repack_present "$dir")
  ranks=$(find "$dir" -maxdepth 1 -type f -name 'rank*_layer*.safetensors' -printf '%f\n' 2>/dev/null \
            | sed -n 's/^rank\([0-9][0-9]*\)_layer[0-9][0-9]*\.safetensors$/\1/p' | sort -un | wc -l)
  layers=$(find "$dir" -maxdepth 1 -type f -name 'rank*_layer*.safetensors' -printf '%f\n' 2>/dev/null \
            | sed -n 's/^rank[0-9][0-9]*_layer\([0-9][0-9]*\)\.safetensors$/\1/p' | sort -un | wc -l)
  echo "        shards=$present  ranks=$ranks  layers=$layers  expected(ranks*layers)=$((ranks * layers))"
}
_stamp_mint_sentinel() { # verify grid rectangular, then record the shard count in <dir>/.mint_complete
  local dir="$1"
  _repack_grid_ok "$dir" || return 1
  _repack_present "$dir" > "$dir/.mint_complete"
}
_repack_complete() { # 0 iff <dir> has a valid .mint_complete matching a present, rectangular grid
  local dir="$1" want have
  [ -f "$dir/.mint_complete" ] || return 1
  _repack_grid_ok "$dir" || return 1
  want=$(cat "$dir/.mint_complete" 2>/dev/null || echo 0)
  case "$want" in ''|*[!0-9]*) return 1;; esac
  have=$(_repack_present "$dir")
  [ "$want" -gt 0 ] && [ "$want" = "$have" ]
}
_free_bytes() { df -PB1 "$1" 2>/dev/null | awk 'NR==2{print $4}' || true; }  # avail bytes on <path>'s FS
_dir_bytes()  { du -sb "$1"    2>/dev/null | awk '{print $1}'      || true; }  # on-disk size of <dir> (bytes)

_wait_health() { # <key> <api> <label> -> 0 healthy | 1 container died | 2 timeout. Heartbeats ~every 60s.
  local key="$1" api="$2" label="$3" i code
  for i in $(seq 1 360); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://$api/health" 2>/dev/null || true)
    [ "$code" = "200" ] && { echo "[engine] $label healthy (~$((i * 5))s)"; return 0; }
    $DOCKER ps --format '{{.Names}}' | grep -q "^vllm-$key$" || return 1   # container exited -> died
    [ $((i % 12)) -eq 0 ] && echo "[engine] $label still loading (~$((i * 5))s; container up) ..."
    sleep 5
  done
  return 2
}

_serve_verify() { # start <key>, gate on /health, detect arch-unsupported, roll back on failure. 0 ok / 1 fail.
  local key="$1" label="${2:-serve}" api="${DEPLOY_API:-$BIND_IP:$PORT}" rc=0 logs
  _stop_all; _create "$key"; $DOCKER start "vllm-$key" >/dev/null
  echo "[engine] $label: vllm-$key started; waiting for /health (first load can take minutes) ..."
  _wait_health "$key" "$api" "$label" || rc=$?
  [ "$rc" -eq 0 ] && return 0
  logs=$($DOCKER logs --tail 80 "vllm-$key" 2>&1 || true)
  # arch-not-supported (mode 6): the engine configured the model but this vllm-gaudi has no model class.
  if printf '%s\n' "$logs" | grep -qiE 'are not supported for now|not supported for now|Model architectures.*(not|failed)|not registered|ModelRegistry|Unknown architecture|No supported.*architecture'; then
    echo "[engine] ARCH NOT SUPPORTED: this vllm-gaudi build has no model class for '$key'." >&2
    echo "         The engine configured the model, but vLLM lacks a registered architecture to run it." >&2
    echo "         Fix: add/enable the model class in this vllm-gaudi image, or serve a supported arch." >&2
  elif [ "$rc" -eq 2 ]; then
    echo "[engine] SERVE TIMEOUT: vllm-$key never became healthy within the wait window." >&2
  else
    echo "[engine] SERVE FAILED: vllm-$key exited during startup." >&2
  fi
  echo "----- last log lines (vllm-$key) -----" >&2; printf '%s\n' "$logs" | tail -30 >&2
  echo "--------------------------------------" >&2
  # rollback (mode 7): never leave a wedged container holding the 8 cards.
  $DOCKER stop -t 40 "vllm-$key" >/dev/null 2>&1 || true
  $DOCKER rm -f "vllm-$key" >/dev/null 2>&1 || true
  echo "[engine] rolled back vllm-$key (no wedged container left)." >&2
  return 1
}

_serve_generated() { # honor autoconfig's verified topology, otherwise use the monolithic path
  local key="$1" phase="$2" layout
  layout=$(sed -n 's/^SERVE_LAYOUT=//p' "$HERE/models/$key.env" | head -1)
  layout=${layout:-monolithic}
  case "$layout" in
    monolithic) _serve_verify "$key" "$phase" ;;
    dp8_tp1|dp4_tp2|dp2_tp4)
      echo "[engine] topology=$layout -> external replica controller"
      "$HERE/dp_serve.sh" "$key" "$layout"
      ;;
    *) echo "[engine] ABORT: unknown SERVE_LAYOUT='$layout' in models/$key.env" >&2; return 1 ;;
  esac
}

case "${1:-}" in
  make) shift; [ $# -ge 1 ] || { echo "make needs a key"; exit 1; }; for k in "$@"; do _create "$k"; done;;
  on)   k="${2:?on needs a key}"; _stop_all; _create "$k"; $DOCKER start "vllm-$k" >/dev/null; echo "ACTIVATED: vllm-$k (loading; logs: docker logs -f vllm-$k)";;
  coadd) # CO-RESIDENT add: start <key> ALONGSIDE the already-running vllm-* (NO _stop_all). For the
         # GLM+Qwen co-residence deploy. Each co-resident env MUST set its own PORT (net=host shares the
         # host ports) so endpoints don't collide; GPU memory is partitioned by each env's gpu-memory-
         # utilization (start the big model first, then coadd the lean one against measured free mem).
         k="${2:?coadd needs a key}"; _create "$k"; $DOCKER start "vllm-$k" >/dev/null
         echo "CO-ADDED: vllm-$k (started alongside running vllm-*; loading; logs: docker logs -f vllm-$k)";;
  off)  if [ -n "${2:-}" ]; then $DOCKER stop -t 40 "vllm-$2" >/dev/null 2>&1 && echo "stopped vllm-$2"; else _stop_all; fi;;
  list) $DOCKER ps -a --filter 'name=vllm-' --format '{{.Names}}\t{{.Status}}';;
  auto) # PLUG-AND-PLAY: auto-detect arch from a model dir -> generate the optimal env -> serve it.
        md="${2:?auto needs a model dir}"; k="${3:-$(basename "$md" | tr '[:upper:]' '[:lower:]')}"
        python3 "$HERE/autoconfig.py" "$md" --emit-env > "$HERE/models/$k.env" || { echo "autoconfig failed"; exit 1; }
        echo "[autoconfig] wrote models/$k.env:"; grep -E '^# Detected|VLLM_ARGS' "$HERE/models/$k.env"
        _serve_generated "$k" "auto" || exit 1
        echo "ACTIVATED: $k (auto-configured; healthy; layout=$(sed -n 's/^SERVE_LAYOUT=//p' "$HERE/models/$k.env" | head -1))";;
  deploy) # FULL ENGINE: detect arch -> MINT missing artifacts (one-time) -> serve optimally. Durable +
          # NON-DESTRUCTIVE (caches beside the model; checkpoint untouched). Run once per model = permanent.
        md="${2:?deploy needs a model dir}"; k="${3:-$(basename "$md" | tr '[:upper:]' '[:lower:]')}"
        _api="${DEPLOY_API:-$BIND_IP:$PORT}"
        python3 "$HERE/autoconfig.py" "$md" --emit-env > "$HERE/models/$k.env" || { echo "autoconfig failed"; exit 1; }
        RMODE=$(sed -n 's/^# ENGINE_REPACK=//p' "$HERE/models/$k.env" | head -1)
        RDIR=$(sed -n 's/^# ENGINE_REPACK_DIR=//p' "$HERE/models/$k.env" | head -1)
        echo "[engine] $k: $(sed -n 's/^# Detected: //p' "$HERE/models/$k.env"); repack=$RMODE"
        # IDEMPOTENCY + PARTIAL-CACHE GUARD (modes 1 & 5): autoconfig flips to 'load' the moment ANY shard
        # exists, so a crashed/partial mint would be silently LOADed. We independently verify the completion
        # sentinel we own.  load + valid sentinel        -> skip mint (idempotent re-deploy).
        #                   load + rectangular, unstamped -> migrate a legacy-complete cache (stamp; skip mint).
        #                   load + genuinely partial      -> WIPE + re-emit (autoconfig drops REPACK_LOAD) -> re-mint.
        if [ "$RMODE" = "load" ] && [ -n "$RDIR" ]; then
          if _repack_complete "$RDIR"; then
            echo "[engine] repack cache verified complete ($(cat "$RDIR/.mint_complete") shards) -> skipping mint (idempotent)"
          elif _repack_grid_ok "$RDIR"; then
            if _stamp_mint_sentinel "$RDIR"; then
              echo "[engine] legacy repack cache is complete ($(cat "$RDIR/.mint_complete") shards) -> stamped sentinel, skipping mint"
            else
              echo "[engine] legacy repack cache complete but could not stamp sentinel (continuing to load)"
            fi
          else
            echo "[engine] repack cache INCOMPLETE/CORRUPT (no valid rank*layer grid) -> wiping partial + re-minting: $RDIR"
            _repack_grid_report "$RDIR"
            rm -rf "${RDIR:?deploy: refusing to wipe an empty repack dir}"; mkdir -p "$RDIR"
            python3 "$HERE/autoconfig.py" "$md" --emit-env > "$HERE/models/$k.env" || { echo "autoconfig failed"; exit 1; }
            RMODE=$(sed -n 's/^# ENGINE_REPACK=//p' "$HERE/models/$k.env" | head -1)
            RDIR=$(sed -n 's/^# ENGINE_REPACK_DIR=//p' "$HERE/models/$k.env" | head -1)
          fi
        fi
        if [ "$RMODE" = "mint" ] && [ -n "$RDIR" ]; then
          echo "[engine] MINTING repack cache once (~minutes; every serve after is instant): $RDIR"
          mkdir -p "$RDIR"
          # DISK PRE-CHECK (mode 2): the repack cache ~= the int4 expert weights (<= the on-disk checkpoint).
          # Abort BEFORE half-filling the disk if free space on the cache FS < the model's on-disk size.
          _need=$(_dir_bytes "$md"); _free=$(_free_bytes "$RDIR")
          if [ -n "$_need" ] && [ -n "$_free" ]; then
            if [ "$_free" -lt "$_need" ]; then
              echo "[engine] ABORT: insufficient disk to mint repack cache on $(df -P "$RDIR" 2>/dev/null | awk 'NR==2{print $6}')." >&2
              echo "         need ~$((_need/1024/1024/1024)) GiB (>= repack size); free ~$((_free/1024/1024/1024)) GiB. Free space or repoint the cache volume." >&2
              exit 1
            fi
          else
            echo "[engine] (disk pre-check skipped: could not measure model/free size)"
          fi
          _va=$(sed -n 's/^VLLM_ARGS="\(.*\)"$/\1/p' "$HERE/models/$k.env")
          # mint pass: same sharding, EAGER + HPU_MOE_REPACK_SAVE, RW cache mount (already in EXTRA_MOUNTS)
          if ! MODEL_ENV_EXTRA="-e HPU_MOE_REPACK_SAVE=/repack_rw" VLLM_ARGS_OVERRIDE="$_va --enforce-eager" \
                 "$0" on "$k" >/dev/null 2>&1; then
            echo "[engine] ABORT: mint container failed to start." >&2; "$0" off "$k" >/dev/null 2>&1 || true; exit 1
          fi
          echo "[engine] mint container up; waiting for weights->repack->save (health) ..."
          _mrc=0; _wait_health "$k" "$_api" "mint" || _mrc=$?
          "$0" off "$k" >/dev/null 2>&1 || true   # tear the mint container down no matter what
          if [ "$_mrc" -ne 0 ]; then
            # modes 3 (timeout) + 4 (died): NEVER fall through to load-mode/serve on a partial/absent cache.
            if [ "$_mrc" -eq 1 ]; then echo "[engine] MINT DIED before healthy:" >&2; else echo "[engine] MINT TIMEOUT (never healthy in wait window):" >&2; fi
            $DOCKER logs --tail 30 "vllm-$k" 2>&1 | tail -30 >&2 || true
            echo "[engine] ABORT: repack cache NOT minted -> refusing to serve a partial/absent cache." >&2
            exit 1
          fi
          # health=200 => process_weights_after_loading saved every layer. Verify the grid is fully
          # rectangular and stamp the completion sentinel ONLY on a verified-clean mint (mode 1).
          if ! _stamp_mint_sentinel "$RDIR"; then
            echo "[engine] ABORT: mint reported healthy but the cache grid is INCOMPLETE:" >&2
            _repack_grid_report "$RDIR" >&2
            echo "[engine] refusing to serve a partial cache -- re-run deploy to retry the mint." >&2
            exit 1
          fi
          echo "[engine] repack cache minted + verified complete ($(cat "$RDIR/.mint_complete") shards); re-emitting env to load it."
          # re-emit now that the cache exists -> ENGINE_REPACK becomes 'load'
          python3 "$HERE/autoconfig.py" "$md" --emit-env > "$HERE/models/$k.env" || { echo "autoconfig failed"; exit 1; }
        fi
        echo "[engine] serving $k (optimal, artifacts resolved) ..."
        _serve_generated "$k" "deploy" || exit 1
        echo "DEPLOYED: $k (healthy; layout=$(sed -n 's/^SERVE_LAYOUT=//p' "$HERE/models/$k.env" | head -1))";;
  *) echo "usage: ./containers.sh {make <key...>|on <key>|off [<key>]|list|auto <model_dir> [key]|deploy <model_dir> [key]}"; exit 1;;
esac
