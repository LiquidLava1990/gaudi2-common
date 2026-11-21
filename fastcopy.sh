#!/usr/bin/env bash
# fastcopy.sh — THE fast way to copy big data on this box (models, datasets).
#
# WHY: local single-stream `rsync` caps ~150 MB/s here — it copies block-by-block
# through a sender→receiver pipe and never queues enough I/O, so the NVMes sit
# ~6% utilised. Parallel `cp` (large sequential reads) saturates them at
# ~2.5-3 GB/s — ~16x faster. So ALWAYS use this (not rsync) for large transfers.
# (Small incremental dir mirrors like the code backup correctly use rsync --delete;
#  internet downloads use `hf` straight to /mnt/fastmodels.)
#
# Usage:
#   fastcopy.sh <dest_dir> <src> [src2 ...]
#   fastcopy.sh /mnt/fastmodels /some/Model-A                 # one
#   fastcopy.sh /mnt/fastmodels /m/A /m/B /m/C                # three, in parallel
# Each <src> is copied into <dest_dir> with `cp -a` as a parallel job, so multiple
# big dirs/files stream concurrently.
set -uo pipefail
DEST="${1:?usage: fastcopy.sh <dest_dir> <src> [src2 ...]}"; shift
[ $# -ge 1 ] || { echo "usage: fastcopy.sh <dest_dir> <src> [src2 ...]"; exit 2; }
mkdir -p "$DEST"
echo "fastcopy -> $DEST  ($# item(s), parallel cp)"
pids=()
for s in "$@"; do
  [ -e "$s" ] || { echo "  ! missing: $s"; continue; }
  cp -a "$s" "$DEST/" & pids+=($!)
  echo "  + $(basename "$s")  (pid $!)"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] && echo "fastcopy DONE ok" || echo "fastcopy had FAILURES (rc=$rc)"
exit $rc
