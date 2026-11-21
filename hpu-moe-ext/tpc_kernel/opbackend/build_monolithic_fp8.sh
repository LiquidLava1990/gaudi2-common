#!/usr/bin/env bash
# =============================================================================
# build_monolithic_fp8.sh -- build the MONOLITHIC M3 swigluoai-FP8 fused MoE op
#   moe_fused_monolithic_fp8_backend.cpp (EAGER) -> build/libmoe_fused_monolithic_fp8.so
#   moe_fused_monolithic_fp8_backend_lazy.cpp (LAZY, BUILD_LAZY=1) -> ..._fp8_lazy.so
#
# This is build_monolithic.sh with SRC/OUT swapped to the fp8 monolithic TU. The fp8
# nodes (convert_to_fp8 / fp8_gemm / dynamic_quant) are ALWAYS emitted (unlike the
# fp8 EXPERT op's BUILD_FP8_NODES gate) because the whole point of this op IS the fp8
# path -- there is no bf16 fallback recipe. The bridge headers, abseil/env_flags
# prereqs, ABI=1, backend-lib-only link, -Wl flags, and the link-gap check are
# byte-identical to build_monolithic.sh -- keep in lockstep if it changes.
#
# Run (inside the dev image, no device):
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/opbackend \
#     vllm-gaudi:next bash build_monolithic_fp8.sh             # eager
#     vllm-gaudi:next bash -c 'BUILD_LAZY=1 bash build_monolithic_fp8.sh'  # lazy
#
# Produces: build/libmoe_fused_monolithic_fp8.so. RUN ON THE HPU BOX ONLY.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
BRIDGE="${HERE}/bridge"
mkdir -p "${BUILD}"

if [[ "${BUILD_LAZY:-0}" == "1" ]]; then
  SRC="${HERE}/moe_fused_monolithic_fp8_backend_lazy.cpp"
  OUT="${BUILD}/libmoe_fused_monolithic_fp8_lazy.so"
  FRONTEND="lazy"
else
  SRC="${HERE}/moe_fused_monolithic_fp8_backend.cpp"
  OUT="${BUILD}/libmoe_fused_monolithic_fp8.so"
  FRONTEND="eager"
fi
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! ${BRIDGE} missing. Clone it first:"
  echo "    git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git ${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (CPU headers only) ----
if [[ ! -f /usr/include/absl/functional/any_invocable.h ]]; then
  echo "=== installing libabsl-dev (build-only headers) ==="
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libabsl-dev >/dev/null 2>&1 \
    || { echo "!!! could not apt-install libabsl-dev"; exit 3; }
fi

GEN_ENV_DIR="${BRIDGE}/generated/env_flags"
if [[ ! -f "${GEN_ENV_DIR}/env_flags_generated.h" ]]; then
  echo "=== generating env_flags_generated.h ==="
  mkdir -p "${GEN_ENV_DIR}"
  ( cd "${BRIDGE}/scripts" && \
    python3 generate_env.py -y env_flags.yaml -o "${GEN_ENV_DIR}" -p "${GEN_ENV_DIR}" ) \
    >/dev/null 2>&1 \
    || { echo "!!! env_flags generation failed"; exit 3; }
fi

TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))')"
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')"
ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"

BACKEND_LIB_NAME="habana_pytorch_backend"
if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
      ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
  BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
fi

BRIDGE_INCS=(
  -I "${BRIDGE}"
  -I "${BRIDGE}/generated"
  -I "${BRIDGE}/pytorch_helpers"
  -I "${BRIDGE}/include"
  -I "${BRIDGE}/python_packages/habana_frameworks/torch/jit/csrc"
  -I "${BRIDGE}/third_party_vendored"
)

echo "=== opbackend/build_monolithic_fp8.sh (MONOLITHIC swigluoai-fp8, ${FRONTEND}) ==="
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "backend lib : -l${BACKEND_LIB_NAME}"
echo "src         : ${SRC}"
echo "out         : ${OUT}"
echo

set -x
g++ -O2 -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI}" \
  -DHABANA_GAUDI2 \
  "${SRC}" \
  -I "${HERE}" \
  "${BRIDGE_INCS[@]}" \
  -I "${HAB_INC}" \
  -I "${TORCH_INC}" \
  -I "${TORCH_INC}/torch/csrc/api/include" \
  -I "${HABF_INC}" \
  -I "${PY_INC}" \
  ${EXTRA_CXXFLAGS} \
  -L "${TORCH_LIB}" -L "${HABF_LIB}" \
  -Wl,--as-needed \
  -ltorch -ltorch_cpu -lc10 \
  -l"${BACKEND_LIB_NAME}" \
  -Wl,--no-as-needed \
  -Wl,--allow-shlib-undefined \
  -Wl,-rpath,"${TORCH_LIB}" -Wl,-rpath,"${HABF_LIB}" \
  -o "${OUT}"
RC=$?
set +x

echo
if [[ ${RC} -eq 0 ]]; then
  echo "OK -> ${OUT}"
  echo "=== symbol resolution check ==="
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry resolve from the linked backend lib."
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'fused_moe_monolithic_fp8|MoeFusedMonolithicFp8' | head
  fi
  echo
  echo "On the HPU box:  cd $HOME/AI/hpu-moe-ext && python tests/test_fused_moe_monolithic_fp8.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
