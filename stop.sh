#!/bin/bash
HERE=$(dirname "$(readlink -f "$0")")
source "$HERE/common.env"
source "$HERE/gaudi_guard.sh"
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  safe_stop "$CONTAINER" && echo "stopped $CONTAINER" || echo "could NOT stop $CONTAINER safely (see above); do not relaunch until resolved"
else
  echo "nothing running ($CONTAINER not present)"
fi
