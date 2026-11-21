#!/bin/bash
# Resilient resumable download for the systemd auto-resume service.
# Self-healing: clears stale .lock files (left by any killed downloader) and RETRIES hf download
# until GLM-5.1-AWQ is fully present (no *.incomplete). 397B is already complete (quick verify).
export HF_XET_HIGH_PERFORMANCE=1
HF=$HOME/.venvs/hf/bin/hf
GLM=/mnt/fastmodels/GLM-5.1-AWQ
Q397=/mnt/fastmodels/Qwen3.5-397B-A17B-GPTQ-Int4

# keep the (already-complete) 397B intact / verify quickly
"$HF" download Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 --local-dir "$Q397" >> $HOME/AI/logs/download-397b.log 2>&1 || true

# resilient GLM loop
while true; do
  find "$GLM/.cache" -name '*.lock' -delete 2>/dev/null   # clear any orphaned locks first
  if "$HF" download QuantTrio/GLM-5.1-AWQ --local-dir "$GLM" >> $HOME/AI/logs/download-glm51.log 2>&1; then
    # hf exited 0 — confirm nothing left incomplete
    if [ -f "$GLM/config.json" ] && ! find "$GLM" -name '*.incomplete' 2>/dev/null | grep -q .; then
      echo "GLM-5.1-AWQ download COMPLETE $(date -u)" >> $HOME/AI/logs/download-glm51.log
      break
    fi
  fi
  sleep 10   # transient failure (rate limit / connection) — retry, resumes from partial
done
echo "all downloads finished"
