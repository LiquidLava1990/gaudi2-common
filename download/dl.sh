#!/bin/bash
export HF_XET_HIGH_PERFORMANCE=1
HF=$HOME/.venvs/hf/bin/hf
$HF download Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --local-dir /mnt/fastmodels/Qwen3-235B-A22B-Instruct-2507-FP8 >> $HOME/AI/logs/download-235b.log 2>&1 &
echo "235B pid $!"
$HF download Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 \
  --local-dir /mnt/fastmodels/Qwen3.5-397B-A17B-GPTQ-Int4 >> $HOME/AI/logs/download-397b.log 2>&1 &
echo "397B pid $!"
