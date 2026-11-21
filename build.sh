#!/bin/bash
# Build the vllm-gaudi:1.24 serving image. Run AFTER gate_zero.sh passes.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
echo "Building vllm-gaudi:1.24 (this compiles vLLM from source; takes a while)..."
sudo docker build -f Dockerfile.vllm-gaudi -t vllm-gaudi:1.24 . 2>&1 | tee $HOME/AI/logs/build-vllm-gaudi.log
echo "Done. Image: vllm-gaudi:1.24"
sudo docker images | grep vllm-gaudi
