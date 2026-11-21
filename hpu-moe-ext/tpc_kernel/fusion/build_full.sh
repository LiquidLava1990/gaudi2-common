#!/usr/bin/env bash
# =============================================================================
# build_full.sh — build the FULL per-expert fused-MoE Synapse recipe executable.
#
# Compiles fused_moe_full.cpp (HOST x86 C++) against the Synapse C API and links
# libSynapse.so. Produces build/fused_moe_full.  CPU-ONLY (NO HPU device flags).
#
# COMPILE (inside the dev/vllm image, CPU-only — no --runtime=habana):
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
#     vllm-gaudi:next bash build_full.sh
#
# Then ON THE HPU BOX ONLY (these acquire a device + launch):
#   FUSED_COMPILE_ONLY=1 ./build/fused_moe_full 32 4096 1408 128   # compile sanity (no device)
#   ./build/fused_moe_full 32 4096 1408 128                        # numeric validation (FUSED)
#   UNFUSE=1 ./build/fused_moe_full 32 4096 1408 128               # numeric validation (UNFUSED)
#   LATENCY_AB=1 ./build/fused_moe_full 32 4096 1408 128           # A/B latency, both variants
#
# Optional build flags (pass via EXTRA_CXXFLAGS):
#   -DUSE_SILU_CGUID=1     emit a single silu_fwd_bf16 ComplexGuid node (else sigmoid+mult)
#   -DDEQUANT_CAST_PARAMS=1 pass a zero-init CastKernel params blob to the dequant node
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
mkdir -p "${BUILD}"

HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
SRC="${HERE}/fused_moe_full.cpp"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

SYN_LIB_DIR=""
for d in /usr/lib/habanalabs /usr/lib/x86_64-linux-gnu /usr/local/lib; do
  if [[ -f "${d}/libSynapse.so" ]]; then SYN_LIB_DIR="${d}"; break; fi
done
if [[ -z "${SYN_LIB_DIR}" ]]; then
  echo "!!! libSynapse.so not found in standard dirs; set SYN_LIB_DIR=... and re-run."
  FOUND="$(find / -name libSynapse.so 2>/dev/null | head -1)"
  [[ -n "${FOUND}" ]] && echo "    (found at: ${FOUND})" && SYN_LIB_DIR="$(dirname "${FOUND}")"
fi
echo "=== fusion/build_full.sh ==="
echo "hab inc     : ${HAB_INC}"
echo "synapse lib : ${SYN_LIB_DIR}/libSynapse.so"
echo "src         : ${SRC}"
echo "extra flags : ${EXTRA_CXXFLAGS}"

set -x
g++ -O2 -std=c++17 \
  "${SRC}" \
  -I "${HAB_INC}" \
  ${EXTRA_CXXFLAGS} \
  -L "${SYN_LIB_DIR}" -lSynapse \
  -Wl,-rpath,"${SYN_LIB_DIR}" \
  -o "${BUILD}/fused_moe_full"
RC=$?
set +x

echo
if [[ ${RC} -eq 0 ]]; then
  echo "OK -> ${BUILD}/fused_moe_full"
  echo "  Compile-only sanity (run on the box; builds+compiles BOTH graphs, no device):"
  echo "    FUSED_COMPILE_ONLY=1 ${BUILD}/fused_moe_full 32 4096 1408 128"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
