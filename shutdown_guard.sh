#!/bin/bash
# Runs at SHUTDOWN via gaudi-shutdown-guard.service (ExecStop), ordered Before=umount.target.
# Purpose: a reboot must NEVER hang at "unmounting /mnt/data1" (what bit us 2026-06-05).
# We are rebooting anyway, so be fast and aggressive. Hard timeouts bound every step;
# the lazy-umount is the real insurance (it detaches the mount even if a wedged task
# still holds it, so systemd's umount phase can't block).
LOG=/run/gaudi-shutdown-guard.log
exec >>"$LOG" 2>&1
echo "=== gaudi-shutdown-guard start ==="
# SAFETY: only act during a REAL shutdown/reboot. Otherwise a manual `systemctl
# stop/restart gaudi-shutdown-guard` would fire ExecStop and nuke the live server.
is_shutting_down() {
  [ "$(systemctl is-system-running 2>/dev/null)" = "stopping" ] && return 0
  systemctl list-jobs 2>/dev/null | grep -qE '(shutdown|reboot|poweroff|halt)\.target' && return 0
  return 1
}
if ! is_shutting_down; then echo "NOT shutting down -> no-op"; echo "=== done (noop) ==="; exit 0; fi
# 1) Stop vLLM containers with a hard time cap (graceful, then kill). A kill that
#    no-ops on a wedged task is harmless here — the kernel is about to reset anyway.
for c in vllm-active vllm-lb vllm-test $(docker ps -q --filter name=vllm-r 2>/dev/null); do
  docker inspect "$c" >/dev/null 2>&1 || continue
  echo "stopping $c"
  timeout 35 docker stop -t 30 "$c" 2>/dev/null || { echo "  kill $c"; timeout 8 docker kill "$c" 2>/dev/null; }
done
sync
# 2) Free the data mounts (fastmodels = models bind, data1 = recipe_cache/quant bind)
#    so the umount phase can't block on a busy/wedged holder.
for M in /mnt/fastmodels /mnt/data1; do
  if mountpoint -q "$M" 2>/dev/null; then
    umount "$M" 2>/dev/null || { echo "lazy-umount $M"; umount -l "$M" 2>/dev/null; }
  fi
done
echo "=== gaudi-shutdown-guard done ==="
exit 0
