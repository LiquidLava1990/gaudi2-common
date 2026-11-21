#!/usr/bin/env bash
# =============================================================================
# build_recipe.sh — P1: build the standalone fused-MoE Synapse recipe executable.
#
# Compiles fused_moe_recipe.cpp (HOST x86 C++) against the Synapse C API and links
# libSynapse.so. Produces build/fused_moe_recipe.
#
# Run CPU-ONLY inside the dev image to COMPILE (NO HPU device flags):
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
#     vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
#     bash build_recipe.sh
#
# Then ON THE HPU BOX ONLY (this launches on a device):
#   # compile-only sanity (no device, builds+compiles the graph, prints residency hint):
#   FUSED_COMPILE_ONLY=1 ./build/fused_moe_recipe 32 6144 4096 128
#   # full run (acquires a device, launches, profiles HBM):
#   GRAPH_VISUALIZATION=1 GRAPH_VISUALIZATION_DIR=./gviz \
#   LOG_LEVEL_SRAM_SLICE=2 HABANA_LOGS=./logs \
#     ./build/fused_moe_recipe 32 6144 4096 128
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
mkdir -p "${BUILD}"

HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
SRC="${HERE}/fused_moe_recipe.cpp"

# Locate libSynapse.so (the GC + runtime). Standard install path on the box/image.
SYN_LIB_DIR=""
for d in /usr/lib/habanalabs /usr/lib/x86_64-linux-gnu /usr/local/lib; do
  if [[ -f "${d}/libSynapse.so" ]]; then SYN_LIB_DIR="${d}"; break; fi
done
if [[ -z "${SYN_LIB_DIR}" ]]; then
  echo "!!! libSynapse.so not found in standard dirs; set SYN_LIB_DIR=... and re-run."
  FOUND="$(find / -name libSynapse.so 2>/dev/null | head -1)"
  [[ -n "${FOUND}" ]] && echo "    (found at: ${FOUND})" && SYN_LIB_DIR="$(dirname "${FOUND}")"
fi
echo "=== fusion/build_recipe.sh ==="
echo "hab inc     : ${HAB_INC}"
echo "synapse lib : ${SYN_LIB_DIR}/libSynapse.so"
echo "src         : ${SRC}"

set -x
g++ -O2 -std=c++17 \
  "${SRC}" \
  -I "${HAB_INC}" \
  -L "${SYN_LIB_DIR}" -lSynapse \
  -Wl,-rpath,"${SYN_LIB_DIR}" \
  -o "${BUILD}/fused_moe_recipe"
RC=$?
set +x

echo
if [[ ${RC} -eq 0 ]]; then
  echo "OK -> ${BUILD}/fused_moe_recipe"
  echo "  Quick compile-only sanity (still needs synInitialize; run on the box):"
  echo "    FUSED_COMPILE_ONLY=1 ${BUILD}/fused_moe_recipe 32 6144 4096 128"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
