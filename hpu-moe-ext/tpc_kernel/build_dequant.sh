#!/usr/bin/env bash
# =============================================================================
# build_dequant.sh — build the Path-1c custom-op glue (dequant_guid_op.cpp).
#
# This compiles HOST x86-64 C++ ONLY (system g++). It does NOT invoke tpc-clang
# and does NOT touch gaudi2.bc — the kernel ISA is already resident in
# libtpc_kernels.so; we only register a torch op pointing at its GUID.
#
# Run CPU-only inside the dev image (NO HPU device flags):
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work \
#     vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
#     bash build_dequant.sh
#
# Produces: build/libdequant_guid_op.so  (loaded by dequant_guid_op.py).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
mkdir -p "${BUILD}"

HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
GLUE_CPP="${HERE}/dequant_guid_op.cpp"

TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))')"
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')"

echo "=== build_dequant.sh ==="
echo "torch    : ${TORCH_DIR}"
echo "habana_fw: ${HABF_DIR}"
echo "hab inc  : ${HAB_INC}"

# Determine the cxx11-abi the installed torch was built with, and match it.
ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 0)"
echo "cxx11 abi: ${ABI}"

set -x
g++ -O2 -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI}" \
  "${GLUE_CPP}" \
  -I "${TORCH_INC}" \
  -I "${TORCH_INC}/torch/csrc/api/include" \
  -I "${HABF_INC}" \
  -I "${HAB_INC}" \
  -I "${PY_INC}" \
  -L "${TORCH_LIB}" -ltorch -ltorch_cpu -lc10 \
  -L "${HABF_LIB}" \
  -lhabana_pytorch_backend \
  -lhabana_pytorch2_plugin \
  -Wl,--no-as-needed \
  -Wl,-rpath,"${TORCH_LIB}" -Wl,-rpath,"${HABF_LIB}" \
  -o "${BUILD}/libdequant_guid_op.so"
RC=$?
set +x

echo
if [[ ${RC} -eq 0 ]]; then
  echo "OK -> ${BUILD}/libdequant_guid_op.so"
  echo "On the HPU box:  python dequant_guid_op.py   # smoke-test the GUID binding"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
