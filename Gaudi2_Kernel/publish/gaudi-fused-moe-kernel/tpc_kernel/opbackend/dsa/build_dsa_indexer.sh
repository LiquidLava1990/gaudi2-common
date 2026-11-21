#!/usr/bin/env bash
# =============================================================================
# build_dsa_indexer.sh -- DSA MVP-1: build dsa_indexer_backend.cpp (EAGER) into
#   build/libdsa_indexer.so
#
# This is opbackend/build.sh adapted for the DSA indexer TU under dsa/:
#   - SRC = dsa/dsa_indexer_backend.cpp ; OUT = dsa/build/libdsa_indexer.so
#   - BRIDGE lives one level up (opbackend/bridge), so PARENT=opbackend.
#   - -I "${HERE}" so the TU finds its sibling dsa_indexer_backend_shared.h.
# EVERYTHING ELSE (bridge headers, abseil/env_flags prereqs, ABI=1, the
# backend-lib-only link, -Wl,--allow-shlib-undefined / --as-needed, the
# OpBackend/BuildNode/KernelRegistry link-gap check) is byte-identical to the
# proven build.sh -- keep in lockstep if build.sh's link flags change.
#
# CPU-only HOST x86-64 compile (system g++). NO HPU device flags. It compiles C++
# against the cloned version-exact bridge headers (../bridge, tag v1.24.0-1007) +
# torch 2.10 + synapse headers, and LINKS the SHIPPED backend lib
# (libhabana_pytorch_backend.so) which already contains OpBackend::BuildNode, the
# KernelRegistry, the GUID lowering, etc.
#
# Run (inside the dev image, NO device):
#   docker run --rm \
#     -v <repo>/tpc_kernel:/work -w /work/opbackend/dsa \
#     vllm-gaudi:next bash build_dsa_indexer.sh
#
# Produces: dsa/build/libdsa_indexer.so  (loaded on the HPU box by the test via
# torch.ops.load_library). RUN ON THE HPU BOX ONLY.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../opbackend/dsa
PARENT="$(cd "${HERE}/.." && pwd)"                      # .../opbackend
BUILD="${HERE}/build"
BRIDGE="${PARENT}/bridge"
mkdir -p "${BUILD}"

SRC="${HERE}/dsa_indexer_backend.cpp"
OUT="${BUILD}/libdsa_indexer.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! ${BRIDGE} missing. Clone it first (shared with the MoE op):"
  echo "    git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git ${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (CPU headers only; identical to build.sh) ----
# (1) Abseil C++ headers: a hard transitive dep of op_backend.h / habana_operator.h.
if [[ ! -f /usr/include/absl/functional/any_invocable.h ]]; then
  echo "=== installing libabsl-dev (build-only headers) ==="
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libabsl-dev >/dev/null 2>&1 \
    || { echo "!!! could not apt-install libabsl-dev"; exit 3; }
fi

# (2) Generate the ONE generated header our TU transitively needs
#     (generated/env_flags/env_flags_generated.h) from the in-repo YAML.
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

# ---- torch + habana_frameworks discovery ----
TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))')"
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')"

# ABI the installed torch was built with (must be 1 for this tag; we force it).
ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"

# The shipped backend lib carries the symbols our TU references at link time:
# OpBackend::BuildNode, the KernelRegistry()/RegisterKernel singleton. Pick
# whichever variant exists. This is the ONLY habana lib we link (NEVER a plugin --
# linking a plugin re-runs its static initializers -> "hpu::habana_d2d_memcpy_other
# is already registered!" abort; see build.sh:90-112 for the full rationale).
BACKEND_LIB_NAME="habana_pytorch_backend"
if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
      ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
  BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
fi

# Bridge transitive-include roots (identical to build.sh; bridge is at ../bridge).
BRIDGE_INCS=(
  -I "${BRIDGE}"
  -I "${BRIDGE}/generated"
  -I "${BRIDGE}/pytorch_helpers"
  -I "${BRIDGE}/include"
  -I "${BRIDGE}/python_packages/habana_frameworks/torch/jit/csrc"
  -I "${BRIDGE}/third_party_vendored"
)

echo "=== dsa/build_dsa_indexer.sh (EAGER) ==="
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; eager frontend syms runtime-resolved)"
echo "bridge      : ${BRIDGE}"
echo "src         : ${SRC}"
echo "out         : ${OUT}"
[[ -n "${EXTRA_CXXFLAGS}" ]] && echo "extra       : ${EXTRA_CXXFLAGS}"
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
  echo
  echo "=== symbol resolution check ==="
  # ldd -r WILL list the eager-frontend symbols (habana::eager::*) as undefined --
  # EXPECTED (runtime-resolved from the imported habana_frameworks.torch). What must
  # NOT be undefined are the symbols the linked backend lib provides:
  # OpBackend::BuildNode + the KernelRegistry/RegisterKernel singleton. NB: each ldd
  # line ends with "(<path-to-our.so>)" whose path contains "dsa"/"opbackend" -- so
  # match on the SYMBOL field only (strip the trailing parenthetical).
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols that the linked backend lib should provide (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry all resolve from the linked backend lib."
    EAGER_SO=""
    for cand in libhabana_pytorch2_plugin.so libhabana_pytorch2_plugin.upstream.so; do
      [[ -f "${HABF_LIB}/${cand}" ]] && { EAGER_SO="${HABF_LIB}/${cand}"; break; }
    done
    EAGER_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana::eager|EagerOp|HbEagerTensorPool' || true)"
    if [[ -n "${EAGER_UNDEF}" ]]; then
      n=$(echo "${EAGER_UNDEF}" | grep -c .)
      echo "Note: ${n} eager-frontend symbol(s) left UNDEFINED on purpose (resolved at runtime from ${EAGER_SO##*/})."
    fi
    if [[ -n "${EAGER_SO}" ]]; then
      EAGER_RUN_DEF="$(nm -DC "${EAGER_SO}" 2>/dev/null | grep -E ' [TWB] ' | grep 'eager::EagerOpBase::run' || true)"
      if [[ -n "${EAGER_RUN_DEF}" ]]; then
        echo "OK: runtime resolver present -- habana::eager::EagerOpBase::run is a global dynamic symbol in ${EAGER_SO##*/}."
      else
        echo "!!! WARNING: could not confirm habana::eager::EagerOpBase::run in ${EAGER_SO##*/}; runtime load may still work if provided elsewhere."
      fi
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'indexer_topk|DsaIndexerTopk' | head
  fi
  echo
  echo "On the HPU box:  cd ${HERE} && python tests/test_dsa_indexer_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
