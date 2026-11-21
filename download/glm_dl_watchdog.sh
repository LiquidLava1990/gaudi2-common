#!/bin/bash
# GLM-5.1 AutoRound download watchdog.
# hf-xet downloads sometimes HANG (0 bytes, process alive, holding .lock/.incomplete)
# — the dl_chain retry loop only catches CRASHES, not hangs. Every 30 min this checks
# whether the GLM dir actually grew; if it's stuck (<200MB/30min) and not complete, it
# kills the hung hf/xet process and deletes the stale partials (the exact fix for the
# 416 'Range Not Satisfiable' stall) so the resumable download picks up where it left off.
# Auto-exits when all 76 shards are present. Idempotent; safe to run alongside dl_chain.sh.
GLM=/mnt/fastmodels/GLM-5.1-int4-mixed-AutoRound
CHAIN=$HOME/AI/download/dl_chain.sh
CHAINLOG=$HOME/AI/logs/dl_chain.log
LOG=$HOME/AI/logs/glm_watchdog.log
last=0
echo "[$(date)] watchdog START" >> "$LOG"
while true; do
  shards=$(ls "$GLM"/*.safetensors 2>/dev/null | wc -l)
  if grep -q "DONE INC4AI/GLM-5.1\|ALL DOWNLOADS COMPLETE" "$CHAINLOG" 2>/dev/null || [ "$shards" -ge 76 ]; then
    echo "[$(date)] GLM COMPLETE ($shards/76) — watchdog exiting" >> "$LOG"
    break
  fi
  cur=$(du -sm "$GLM" 2>/dev/null | cut -f1); cur=${cur:-0}
  grew=$((cur - last))
  echo "[$(date)] GLM ${cur}MB (+${grew}MB/30min), shards ${shards}/76" >> "$LOG"
  if [ "$last" -ne 0 ] && [ "$grew" -lt 200 ]; then
    echo "[$(date)] STUCK -> poking: kill hung hf/xet, clear stale partials, resume" >> "$LOG"
    pkill -9 -f "hf download.*GLM-5.1-int4-mixed-AutoRound" 2>/dev/null
    pkill -9 -f hf_xet 2>/dev/null
    sleep 3
    find "$GLM" -name "*.incomplete" -delete 2>/dev/null
    find "$GLM" -name "*.lock" -delete 2>/dev/null
    if ! pgrep -f dl_chain.sh >/dev/null 2>&1; then
      echo "[$(date)] dl_chain.sh not running -> restarting it" >> "$LOG"
      nohup bash "$CHAIN" >> "$CHAINLOG" 2>&1 &
    fi
  fi
  last=$cur
  sleep 1800
done
