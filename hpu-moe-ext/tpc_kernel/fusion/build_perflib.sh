#!/usr/bin/env bash
# =============================================================================
# build_perflib.sh — P0: build the user TPC perf-lib .so (moe_ext_perflib.cpp).
#
# Pipeline:
#   [1] tpc-clang  : ../moe_ext_unpack_dequant_w4.c -> moe_ext_unpack_dequant_w4.o
#                    (Gaudi2 TPC ELF; verified rc=0 — no gaudi2.bc needed for this
#                     elementwise kernel, per ../build.sh banner.)
#   [2] llvm-objcopy -I binary : embed that .o as a data blob exporting
#                    _binary_moe_ext_unpack_dequant_w4_o_{start,end}
#   [3] g++ (HOST x86) : compile moe_ext_perflib.cpp + link the embedded blob
#                    -> libmoe_ext_perflib.so  (exports GetKernelGuids /
#                       InstantiateTpcKernel / GetLibVersion).
#
# Run CPU-ONLY inside the dev image (NO HPU device flags):
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
#     vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
#     bash build_perflib.sh
#
# Then ON THE HPU BOX:
#   export GC_KERNEL_PATH=$PWD/build/libmoe_ext_perflib.so:$GC_KERNEL_PATH
#   python test_p0_runnable.py            # bit-exact vs convert_from_uint4
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
BUILD="${HERE}/build"
mkdir -p "${BUILD}"

TPC_CLANG="${TPC_CLANG:-/opt/habanalabs/bin/tpc-clang}"
OBJCOPY="${OBJCOPY:-/opt/habanalabs/bin/llvm-objcopy}"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
KERNEL_C="${ROOT}/moe_ext_unpack_dequant_w4.c"
SHIM_CPP="${HERE}/moe_ext_perflib.cpp"

echo "=== fusion/build_perflib.sh ==="
echo "tpc-clang : ${TPC_CLANG}"
echo "objcopy   : ${OBJCOPY}"
echo "kernel .c : ${KERNEL_C}"
echo "shim .cpp : ${SHIM_CPP}"

# -------------------------------------------------------------------------
# [1] TPC compile -> ELF object. The objcopy symbol name is derived from the
#     INPUT FILENAME, so compile into a file literally named
#     moe_ext_unpack_dequant_w4.o inside BUILD (symbol stem must match the
#     extern decls in the shim).
# -------------------------------------------------------------------------
echo
echo "=== [1/3] TPC compile -> moe_ext_unpack_dequant_w4.o ==="
set -x
"${TPC_CLANG}" -mcpu=gaudi2 -O2 -c "${KERNEL_C}" -I "${HAB_INC}" \
  -o "${BUILD}/moe_ext_unpack_dequant_w4.o"
TPC_RC=$?
set +x
if [[ ${TPC_RC} -ne 0 ]]; then
  echo "!!! TPC compile FAILED rc=${TPC_RC} — cannot build perf-lib."
  exit ${TPC_RC}
fi

# -------------------------------------------------------------------------
# [2] Embed the ELF as a binary blob. objcopy derives symbol names from the
#     full input path, sanitizing non-alnum to '_'. To get the clean stem
#     _binary_moe_ext_unpack_dequant_w4_o_{start,end} we run objcopy from
#     INSIDE build/ on the bare basename.
# -------------------------------------------------------------------------
echo
echo "=== [2/3] embed ELF (llvm-objcopy -I binary) ==="
set -x
( cd "${BUILD}" && "${OBJCOPY}" -I binary -O elf64-x86-64 \
    --binary-architecture i386:x86-64 \
    moe_ext_unpack_dequant_w4.o moe_ext_unpack_dequant_w4_embed.o )
OBJCOPY_RC=$?
set +x
if [[ ${OBJCOPY_RC} -ne 0 ]]; then
  echo "!!! objcopy embed FAILED rc=${OBJCOPY_RC}."
  exit ${OBJCOPY_RC}
fi
echo "--- embedded symbols ---"
"${OBJCOPY}" --version >/dev/null 2>&1 && \
  ( cd "${BUILD}" && nm moe_ext_unpack_dequant_w4_embed.o 2>/dev/null | grep -i binary || true )

# -------------------------------------------------------------------------
# [3] Compile + link the shim into the perf-lib .so. Host g++ (NOT tpc-clang).
#     The shim only needs the C headers in /usr/include/habanalabs; it links no
#     torch/synapse libs (GC dlopen's it and resolves the C entry points).
# -------------------------------------------------------------------------
echo
echo "=== [3/3] g++ shim + embedded ELF -> libmoe_ext_perflib.so ==="
set -x
g++ -O2 -fPIC -shared -std=c++17 \
  "${SHIM_CPP}" \
  "${BUILD}/moe_ext_unpack_dequant_w4_embed.o" \
  -I "${HAB_INC}" \
  -Wl,-z,noexecstack \
  -o "${BUILD}/libmoe_ext_perflib.so"
SHIM_RC=$?
set +x

echo
echo "=== build summary ==="
echo "  TPC .o       : rc=${TPC_RC}      ${BUILD}/moe_ext_unpack_dequant_w4.o"
echo "  embed .o     : rc=${OBJCOPY_RC}  ${BUILD}/moe_ext_unpack_dequant_w4_embed.o"
echo "  perf-lib .so : rc=${SHIM_RC}     ${BUILD}/libmoe_ext_perflib.so"
if [[ ${SHIM_RC} -eq 0 ]]; then
  echo "--- exported entry points (must include GetKernelGuids / InstantiateTpcKernel) ---"
  nm -D -C "${BUILD}/libmoe_ext_perflib.so" 2>/dev/null | \
    grep -E "GetKernelGuids|InstantiateTpcKernel|GetLibVersion" || \
    echo "  (nm unavailable; check on box)"
  echo
  echo "  ON THE HPU BOX:"
  echo "    export GC_KERNEL_PATH=${BUILD}/libmoe_ext_perflib.so:\$GC_KERNEL_PATH"
  echo "    python ${HERE}/test_p0_runnable.py"
fi

[[ ${TPC_RC} -ne 0 || ${OBJCOPY_RC} -ne 0 || ${SHIM_RC} -ne 0 ]] && exit 1
exit 0
