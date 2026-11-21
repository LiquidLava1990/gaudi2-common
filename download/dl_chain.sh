#!/bin/bash
# Single-stream chained download: Qwen-397B first, then GLM. xet off + hf_transfer on
# (xet appeared to hang). Resumable/idempotent; retries forever until each completes.
HF=$HOME/.venvs/hf/bin/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=0
export HF_HUB_DISABLE_XET=1
dl() {
  local repo="$1" dir="$2"; local dest="/mnt/fastmodels/$dir"
  mkdir -p "$dest"
  local n=0
  while true; do
    n=$((n+1))
    echo "[$(date)] attempt $n: $repo"
    "$HF" download "$repo" --local-dir "$dest" && { echo "[$(date)] DONE $repo"; return 0; }
    echo "[$(date)] rc=$? retry in 10s"; sleep 10
  done
}
dl "Intel/Qwen3.5-397B-A17B-int4-AutoRound" "Qwen3.5-397B-AutoRound"
dl "INC4AI/GLM-5.1-int4-mixed-AutoRound" "GLM-5.1-int4-mixed-AutoRound"
echo "[$(date)] ALL DOWNLOADS COMPLETE"
