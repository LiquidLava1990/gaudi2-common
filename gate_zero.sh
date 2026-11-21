#!/bin/bash
# GATE ZERO — prove the SynapseAI 1.24 container userspace actually drives our host
# habanalabs driver and can run a real HPU op, before building/serving anything.
# Host now runs the matched out-of-tree habanalabs-dkms 1.24 on a pinned 6.8.12 kernel.
# Uses manual device passthrough (NOT --runtime=habana, which version-couples to the driver):
#   - /dev/accel/accel*        : the 8 Gaudi compute nodes
#   - /dev/infiniband/uverbs*  : RDMA/IB verbs nodes the HCL layer needs; WITHOUT these the
#                                synapse runtime fails 'ibv init' -> synStatus=26 Device acquire failed
# Plus --net=host --ipc=host --ulimit memlock=-1:-1 for the RDMA path.
set -uo pipefail
IMG=vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-upstream-installer-2.10.0:latest
DEV=""; for d in /dev/accel/accel* /dev/infiniband/uverbs*; do DEV="$DEV --device=$d"; done
RUNFLAGS="--cap-add=sys_nice --net=host --ipc=host --ulimit memlock=-1:-1"

echo "############ GATE ZERO ############"
echo "=== 1) hl-smi inside container (driver+firmware visible to 1.24 userspace?) ==="
# shellcheck disable=SC2086
SMI=$(sudo docker run --rm $DEV $RUNFLAGS "$IMG" hl-smi 2>&1); RC1=$?
echo "$SMI" | head -25

echo
echo "=== 2) trivial HPU tensor op (does the synapse runtime + torch HPU bridge init?) ==="
# shellcheck disable=SC2086
sudo docker run --rm $DEV $RUNFLAGS "$IMG" python -c "
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
print('hpu available:', torch.hpu.is_available(), '| device count:', torch.hpu.device_count())
x = torch.randn(2048, 2048).to('hpu'); y = torch.randn(2048, 2048).to('hpu')
z = x @ y; htcore.mark_step()
print('HPU matmul OK -> device:', z.device, '| checksum:', float(z.sum().cpu()))
" 2>&1 | grep -vE 'warning logged once|add_step_closure|mark_step function|iter_mark_step'
RC2=${PIPESTATUS[0]}

echo
if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
  echo "GATE ZERO: PASS — 1.24 userspace drives the DKMS driver + HPU compute works. Safe to build/serve."
else
  echo "GATE ZERO: FAIL (hl-smi rc=$RC1, hpu-op rc=$RC2)."
  echo "If 'ibv init failed': check /dev/infiniband/uverbs* passthrough + --ulimit memlock=-1."
fi
