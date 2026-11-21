#!/bin/bash
# Resumable download of the FULL MiniMax-M3 bf16 checkpoint (856 GiB, 59 shards) to data1
# (most free room: 3.4T). Source for the bf16 -> fp8 conversion (static fp8 experts, no dequant).
# Resumable / idempotent: hf download skips complete files; retries forever until the snapshot lands.
set -u
REPO="MiniMaxAI/MiniMax-M3"
DEST="/mnt/data1/minimax-m3-bf16"
HF=$HOME/.venvs/hf/bin/hf
export HF_HOME=$HOME/.cache/huggingface     # picks up the auth token (user Carbon6VI)
export HF_HUB_DISABLE_XET=1                       # Xet STALLS on this box's big pulls (confirmed 2026-06-23 + old dl_chain note); classic path is reliable
# (hf_transfer is a no-op in huggingface_hub 1.17 -> omitted; --max-workers 8 gives 8 parallel shard fetches, the proven ~46 MB/s path)
LOG=$HOME/AI/logs/m3_download_data1.log
mkdir -p "$DEST"
echo "[$(date)] ==== START $REPO -> $DEST ====" | tee -a "$LOG"
n=0
while true; do
  n=$((n+1))
  echo "[$(date)] attempt $n" | tee -a "$LOG"
  "$HF" download "$REPO" --local-dir "$DEST" --max-workers 8 >> "$LOG" 2>&1 && {
    echo "[$(date)] ==== COMPLETE after $n attempt(s) ====" | tee -a "$LOG"; break;
  }
  echo "[$(date)] attempt $n failed (rc=$?), retry in 15s ..." | tee -a "$LOG"
  sleep 15
done