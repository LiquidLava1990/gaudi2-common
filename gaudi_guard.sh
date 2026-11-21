#!/bin/bash
# gaudi_guard.sh — shared safety guards for the Gaudi serving scripts.
#
# Why this exists: on this off-matrix kernel (6.8.12 + habanalabs DKMS), vllm-gaudi
# pins its 8 TP worker processes' CPU affinity at startup (sched_setaffinity). If a
# fresh pinning storm begins while a previous vllm process is still tearing down (or
# was force-killed mid-pin with `docker rm -f`), the kernel's affinity migration
# (affine_move_task) can block UNINTERRUPTIBLY. The task becomes unkillable (SIGKILL
# is a no-op), keeps holding the cards, every relaunch re-wedges, and the next
# shutdown hangs at "unmounting /mnt/data1". Only a reboot clears it.
#
# These helpers remove the trigger: (1) refuse to launch if already wedged, and
# (2) stop containers GRACEFULLY and WAIT for full release instead of `rm -f` churn.
# Source this from serve.sh / stop.sh / test_serve_plugin.sh / dp_serve.sh.

DOCKER="${DOCKER:-docker}"

# deadlock_guard: refuse to launch if a vllm/python worker is stuck uninterruptible
# (the affinity-wedge symptom). Detected via ps D-state — cheap and NON-BLOCKING.
# We deliberately do NOT read /proc/*/stack: that read can itself block forever on a
# wedged task and hang the very guard meant to protect us. A pid must remain in D
# across a recheck, so a transient D from normal disk I/O is not a false alarm.
# Override with SKIP_DEADLOCK_GUARD=1 if you are certain the box is healthy.
deadlock_guard() {
  [ "${SKIP_DEADLOCK_GUARD:-0}" = "1" ] && return 0
  local sig='vllm|VllmWorker|EngineCore|pt_main_thread|torchrun'
  local d1 d2 p stuck=""
  d1=$(ps -eo state=,pid=,args= 2>/dev/null | awk -v s="$sig" '$1 ~ /^D/ && $0 ~ s {print $2}')
  [ -n "$d1" ] || return 0
  sleep 4
  d2=$(ps -eo state=,pid=,args= 2>/dev/null | awk -v s="$sig" '$1 ~ /^D/ && $0 ~ s {print $2}')
  for p in $d2; do case " $d1 " in *" $p "*) stuck="$stuck $p" ;; esac; done
  [ -n "$stuck" ] || return 0
  {
    echo "============================================================"
    echo "!! AFFINITY DEADLOCK LIKELY — vllm/python worker(s) stuck uninterruptible (D):"
    for p in $stuck; do
      echo "!!   pid $p  ($(cat "/proc/$p/comm" 2>/dev/null))  $(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | cut -c1-72)"
    done
    echo "!! These won't die (SIGKILL is a no-op); only a reboot clears it."
    echo "!! REFUSING to launch — would compound it + hang the next shutdown."
    echo "!! Fix:  sudo reboot   (models + recipe cache persist on /mnt/data1)"
    echo "!! Override (only if you are sure):  SKIP_DEADLOCK_GUARD=1 <script> ..."
    echo "============================================================"
  } >&2
  return 86
}

# safe_stop <name> [grace_seconds]
# Gracefully stop+remove a container and WAIT for it to fully exit (release cards).
# Never force-kills by default. Set FORCE_STOP=1 to allow a last-resort rm -f.
_container_holds_accel() {
  # Return 0 only when some process in the container still has an accelerator
  # device open; return 1 when the container is provably detached from HPU.
  $DOCKER exec "$1" sh -c '
    for f in /proc/[0-9]*/fd/*; do
      p=$(readlink "$f" 2>/dev/null || true)
      case "$p" in /dev/accel/*) exit 0;; esac
    done
    exit 1
  ' >/dev/null 2>&1
}

safe_stop() {
  local name="$1" t="${2:-90}" i state accel_rc worker_pids p
  $DOCKER inspect "$name" >/dev/null 2>&1 || return 0   # not present -> nothing to do
  echo "Gracefully stopping '$name' (SIGTERM only, up to ${t}s)..."
  # `docker stop -t N` sends SIGKILL after N seconds. That contradicts this
  # guard's core promise and can recreate the affinity wedge. Send TERM once,
  # then poll ourselves so timeout means a loud refusal, never a hidden KILL.
  $DOCKER kill --signal=TERM "$name" >/dev/null 2>&1 || true
  # vLLM's API process is PID 1 but its EngineCore child owns HPU. Some builds
  # do not propagate TERM to that child, leaving HBM pinned indefinitely. TERM
  # known worker children directly as well; this is still a graceful signal.
  worker_pids=$($DOCKER exec "$name" sh -c '
    for c in /proc/[0-9]*/comm; do
      n=$(cat "$c" 2>/dev/null || true)
      case "$n" in VLLM::EngineCor*|VllmWorker*|pt_main_thread*|torchrun*)
        p=${c#/proc/}; echo ${p%/comm};;
      esac
    done
  ' 2>/dev/null || true)
  for p in $worker_pids; do $DOCKER exec "$name" kill -TERM "$p" >/dev/null 2>&1 || true; done
  for i in $(seq 1 "$t"); do
    $DOCKER inspect "$name" >/dev/null 2>&1 || { echo "  '$name' stopped + removed."; return 0; }
    state=$($DOCKER inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo gone)
    case "$state" in
      gone) return 0 ;;
      exited|created|dead)
        $DOCKER rm "$name" >/dev/null 2>&1 || true
        echo "  '$name' exited cleanly + removed."; return 0 ;;
    esac
    # PID 1 can remain stuck polling a dead EngineCore. Once every process has
    # closed /dev/accel, force-removing that supervisor cannot strand an HPU
    # context; it simply reaps the container after card release is proven.
    if [ "$i" -ge 3 ]; then
      accel_rc=0; _container_holds_accel "$name" || accel_rc=$?
      if [ "$accel_rc" -eq 1 ]; then
        echo "  '$name' released /dev/accel; removing leftover supervisor."
        $DOCKER rm -f "$name" >/dev/null 2>&1 || return 1
        return 0
      fi
    fi
    sleep 1
  done
  if [ "${FORCE_STOP:-0}" = "1" ]; then
    echo "  WARN: '$name' didn't stop in ${t}s; FORCE_STOP=1 -> rm -f (may wedge scheduler)" >&2
    $DOCKER rm -f "$name" >/dev/null 2>&1 || true; sleep 3; return 0
  fi
  {
    echo "  ERROR: '$name' still holds /dev/accel after ${t}s."
    echo "  NOT forcing an HPU-owning process — that can trigger the affinity deadlock."
    echo "  Inspect:  $DOCKER logs --tail 50 $name"
    echo "  If truly stuck, reboot. To override once: FORCE_STOP=1 <script> ..."
  } >&2
  return 1
}
