#!/usr/bin/env bash
# =============================================================================
# build_op.sh -- build the fused-MoE torch BRIDGE op (fused_moe_op.cpp).
#
# Compiles HOST x86-64 C++ ONLY (system g++). Links BOTH:
#   * libSynapse.so          -- the graph API (synGraphCompile/synLaunch/...) that
#                               fused_expert_graph.h drives (approach B), AND
#   * torch + habana backend -- torch::library / at::Tensor (the op registration).
# It does NOT invoke tpc-clang and does NOT touch gaudi2.bc (the kernel ISA is
# resident; we only build an explicit Synapse graph from stock GUIDs + MME).
#
# Run CPU-only inside the dev image (NO HPU device flags -- NO --runtime=habana):
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
#     vllm-gaudi:next bash build_op.sh
#
# Produces: build/libfused_moe_op.so  (loaded by fused_moe_op.py).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
mkdir -p "${BUILD}"

HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
SRC="${HERE}/fused_moe_op.cpp"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

# locate libSynapse
SYN_LIB_DIR=""
for d in /usr/lib/habanalabs /usr/lib/x86_64-linux-gnu /usr/local/lib; do
  if [[ -f "${d}/libSynapse.so" ]]; then SYN_LIB_DIR="${d}"; break; fi
done
if [[ -z "${SYN_LIB_DIR}" ]]; then
  FOUND="$(find / -name libSynapse.so 2>/dev/null | head -1)"
  [[ -n "${FOUND}" ]] && SYN_LIB_DIR="$(dirname "${FOUND}")"
fi
if [[ -z "${SYN_LIB_DIR}" ]]; then
  echo "!!! libSynapse.so not found; set SYN_LIB_DIR=... and re-run."; exit 3
fi

# torch + habana_frameworks dirs (same discovery as build_dequant.sh)
TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))')"
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')"

# match the ABI the installed torch was built with
ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 0)"

echo "=== fusion/build_op.sh ==="
echo "torch     : ${TORCH_DIR}"
echo "habana_fw : ${HABF_DIR}"
echo "synapse   : ${SYN_LIB_DIR}/libSynapse.so"
echo "hab inc   : ${HAB_INC}"
echo "cxx11 abi : ${ABI}"
echo "src       : ${SRC}"
echo "extra     : ${EXTRA_CXXFLAGS}"

set -x
g++ -O2 -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI}" \
  "${SRC}" \
  -I "${HERE}" \
  -I "${HAB_INC}" \
  -I "${TORCH_INC}" \
  -I "${TORCH_INC}/torch/csrc/api/include" \
  -I "${HABF_INC}" \
  -I "${PY_INC}" \
  ${EXTRA_CXXFLAGS} \
  -L "${TORCH_LIB}" -ltorch -ltorch_cpu -lc10 \
  -L "${HABF_LIB}" \
  -lhabana_pytorch_backend \
  -lhabana_pytorch2_plugin \
  -L "${SYN_LIB_DIR}" -lSynapse \
  -Wl,--no-as-needed \
  -Wl,-rpath,"${TORCH_LIB}" -Wl,-rpath,"${HABF_LIB}" -Wl,-rpath,"${SYN_LIB_DIR}" \
  -o "${BUILD}/libfused_moe_op.so"
RC=$?
set +x

echo
if [[ ${RC} -eq 0 ]]; then
  echo "OK -> ${BUILD}/libfused_moe_op.so"
  echo "On the HPU box:  python test_fused_moe_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
