#!/usr/bin/env bash
# =============================================================================
# build_fp8.sh -- build moe_fused_expert_fp8_backend.cpp into
#   build/libmoe_fused_expert_fp8.so   (the fp8-COMPUTE variant)
#
# This is build.sh with three deltas: SRC = the fp8 backend, OUT = the fp8 .so,
# and -DFUSED_FP8=1 (+ honoring an optional -DBUILD_FP8_NODES=1). EVERYTHING ELSE
# -- the bridge headers, the abseil/env_flags prereqs, the ABI, the
# backend-lib-only link, the runtime-resolved eager-frontend symbols, and the
# OpBackend/BuildNode/KernelRegistry link-gap check -- is byte-identical to the
# proven build.sh, so the fp8 .so links exactly like the bf16 one.
#
# It REUSES build.sh's logic by sourcing the shared prereq steps via env override
# where build.sh allows it (EXTRA_CXXFLAGS) and re-implementing only the
# SRC/OUT-specific compile (build.sh hardcodes SRC/OUT). Keep this in lockstep
# with build.sh if that file's link flags change.
#
# Run (inside the dev image, NO device):
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/opbackend \
#     vllm-gaudi:next bash build_fp8.sh
#
# To emit the actual fp8 nodes (after the on-device param-struct/amax validation):
#   ... vllm-gaudi:next bash -c 'BUILD_FP8_NODES=1 bash build_fp8.sh'
#
# Produces: build/libmoe_fused_expert_fp8.so  (loaded on the HPU box by the test
# via torch.ops.load_library). RUN ON THE HPU BOX ONLY.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
BRIDGE="${HERE}/bridge"
mkdir -p "${BUILD}"

SRC="${HERE}/moe_fused_expert_fp8_backend.cpp"
OUT="${BUILD}/libmoe_fused_expert_fp8.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"

# -DFUSED_FP8=1 marks the fp8 build; BUILD_FP8_NODES (default 0 in the source)
# can be forced on from the environment to emit the fp8 nodes.
FP8_DEFS="-DFUSED_FP8=1"
if [[ "${BUILD_FP8_NODES:-0}" == "1" ]]; then
  FP8_DEFS="${FP8_DEFS} -DBUILD_FP8_NODES=1"
  echo "=== BUILD_FP8_NODES=1: emitting the fp8 nodes (requires the on-device-validated param structs) ==="
fi
EXTRA_CXXFLAGS="${FP8_DEFS} ${EXTRA_CXXFLAGS:-}"

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! ${BRIDGE} missing. Clone it first:"
  echo "    git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git ${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (identical to build.sh) ----
if [[ ! -f /usr/include/absl/functional/any_invocable.h ]]; then
  echo "=== installing libabsl-dev (build-only headers) ==="
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libabsl-dev >/dev/null 2>&1 \
    || { echo "!!! could not apt-install libabsl-dev"; exit 3; }
fi

GEN_ENV_DIR="${BRIDGE}/generated/env_flags"
if [[ ! -f "${GEN_ENV_DIR}/env_flags_generated.h" ]]; then
  echo "=== generating env_flags_generated.h (scripts/generate_env.py) ==="
  mkdir -p "${GEN_ENV_DIR}"
  ( cd "${BRIDGE}/scripts" && \
    python3 generate_env.py -y env_flags.yaml -o "${GEN_ENV_DIR}" -p "${GEN_ENV_DIR}" ) \
    >/dev/null 2>&1 \
    || { echo "!!! env_flags generation failed"; exit 3; }
  [[ -f "${GEN_ENV_DIR}/env_flags_generated.h" ]] \
    || { echo "!!! env_flags_generated.h not produced"; exit 3; }
fi

# ---- torch + habana_frameworks discovery (identical to build.sh) ----
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

echo "=== opbackend/build_fp8.sh (fp8-compute variant) ==="
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; eager frontend syms runtime-resolved)"
echo "bridge      : ${BRIDGE}"
echo "src         : ${SRC}"
echo "out         : ${OUT}"
echo "fp8 defs    : ${FP8_DEFS}"
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
  echo "=== link-gap check (OpBackend/BuildNode/KernelRegistry must resolve) ==="
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry resolve from the linked backend lib."
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'fused_expert_fp8|MoeFusedExpertFp8' | head
  fi
  echo
  echo "On the HPU box:  cd $(dirname "${HERE}")/opbackend && python test_fused_expert_fp8_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
