#!/usr/bin/env bash
# Build the MMEL csengine submit core — plain gcc, NO Habana libs (talks the
# DRM/accel ioctl directly). Compiling needs NO card. Running the on-card tests
# needs a free card (see RUN_ONCARD.md / W1_BUILD_TEST.md).
#
# Produces:
#   abi_check       — independent ABI verifier vs the real kernel UAPI header
#   libcsengine.so  — the reusable submit core (W1 deliverable)
#   libcsengine.a   — static archive (for statically-linked test/tools)
#   w1_test         — the W1 on-card acceptance test (links libcsengine)
#   cs_poc          — the M0 PoC (kept; self-contained, no lib)
set -euo pipefail
cd "$(dirname "$0")"
CC=${CC:-gcc}
CFLAGS_COMMON="-O2 -Wall -Wextra -Wno-unused-parameter -std=gnu11"

# 1) Independent ABI verifier — compiles against the REAL kernel UAPI header and
#    static-asserts our mirror matches. If this fails, the lib's ABI is wrong.
echo "[1/5] ABI check (vs /usr/include/drm/habanalabs_accel.h) ..."
if [ -e /usr/include/drm/habanalabs_accel.h ]; then
  $CC -O0 -o abi_check abi_check.c
  ./abi_check
else
  echo "  WARN: real UAPI header not found; skipping ABI cross-check (run on the box)."
fi

# 2) The library — shared object (PIC) for the Python ctypes bridge + the worker.
echo "[2/5] Building libcsengine.so ..."
$CC $CFLAGS_COMMON -fPIC -c libcsengine.c -o libcsengine.o
$CC -shared -o libcsengine.so libcsengine.o

# 3) Static archive too (handy for a fully self-contained test binary).
echo "[3/5] Building libcsengine.a ..."
ar rcs libcsengine.a libcsengine.o

# 4) The W1 on-card acceptance test (links the static lib so it has no rpath dep).
echo "[4/5] Building w1_test (links libcsengine.a) ..."
$CC $CFLAGS_COMMON -o w1_test w1_test.c libcsengine.a

# 5) The M0 PoC — self-contained, no lib (kept for the original proof).
echo "[5/5] Building cs_poc (M0 PoC) ..."
$CC $CFLAGS_COMMON -o cs_poc cs_poc.c

echo
echo "OK -> $(pwd)/libcsengine.so  libcsengine.a  w1_test  cs_poc"
echo "Run gates (need a card with NO active compute ctx):"
echo "  ./w1_test --help        # W1 acceptance test (3-engine chain + MME CB + staged)"
echo "  ./cs_poc  --help        # M0 PoC (PDMA-only sync proof)"
