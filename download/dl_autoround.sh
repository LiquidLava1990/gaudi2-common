#!/bin/bash
# Resilient HF download loop for the Intel/INC4AI AutoRound models.
# Retries until the snapshot is complete (hf download is resumable / idempotent).
# Usage: dl_autoround.sh <repo_id> <local_dir_name>
set -u
REPO="$1"; DIR="$2"
DEST="/mnt/fastmodels/$DIR"
HF=$HOME/.venvs/hf/bin/hf
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$DEST"
echo "[$(date)] START download $REPO -> $DEST"
n=0
while true; do
  n=$((n+1))
  "$HF" download "$REPO" --local-dir "$DEST" --max-workers 8 && {
    echo "[$(date)] COMPLETE after $n attempt(s)"; break;
  }
  echo "[$(date)] attempt $n failed (rc=$?), retrying in 15s ..."
  sleep 15
done
