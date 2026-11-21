#!/usr/bin/env bash
# =============================================================================
# build_dsa_indexer_lazy.sh -- DSA MVP-1: build dsa_indexer_backend_lazy.cpp into
#   build/libdsa_indexer_lazy.so   (the LAZY-FRONTEND variant)
#
# This is opbackend/build_lazy.sh adapted for the DSA indexer TU under dsa/:
#   - SRC = dsa/dsa_indexer_backend_lazy.cpp ; OUT = dsa/build/libdsa_indexer_lazy.so
#   - BRIDGE lives one level up (opbackend/bridge), so PARENT=opbackend.
#   - -I "${HERE}" so the TU finds its sibling dsa_indexer_backend_shared.h.
#   - discovery python + symbol checks run UNDER PT_HPU_LAZY_MODE=1 so the LAZY
#     plugin (libhabana_pytorch_plugin.so) is the referenced one -- the plugin
#     that resolves this .so's habana_lazy::* frontend symbols at runtime.
#   - link ONLY libhabana_pytorch_backend.so (NEVER a plugin).
# EVERYTHING ELSE is byte-identical to the proven build_lazy.sh -- keep in lockstep.
#
# Run (inside the dev image, NO device):
#   docker run --rm \
#     -v <repo>/tpc_kernel:/work -w /work/opbackend/dsa \
#     vllm-gaudi:next bash build_dsa_indexer_lazy.sh
#
# Produces: dsa/build/libdsa_indexer_lazy.so  (loaded on the HPU box under
# PT_HPU_LAZY_MODE=1). RUN ON THE HPU BOX ONLY for the device test.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../opbackend/dsa
PARENT="$(cd "${HERE}/.." && pwd)"                      # .../opbackend
BUILD="${HERE}/build"
BRIDGE="${PARENT}/bridge"
mkdir -p "${BUILD}"

SRC="${HERE}/dsa_indexer_backend_lazy.cpp"
OUT="${BUILD}/libdsa_indexer_lazy.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

# Reference the LAZY plugin for the discovery python + symbol checks. Under
# PT_HPU_LAZY_MODE=1, `import habana_frameworks.torch` loads
# libhabana_pytorch_plugin.so (the lazy plugin) -- the exact lib that resolves
# this .so's habana_lazy::* frontend symbols at runtime on the box.
export PT_HPU_LAZY_MODE=1

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! ${BRIDGE} missing. Clone it first (shared with the MoE op):"
  echo "    git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git ${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (identical to build_lazy.sh) ----
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

# ---- torch + habana_frameworks discovery (under PT_HPU_LAZY_MODE=1) ----
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

echo "=== dsa/build_dsa_indexer_lazy.sh (LAZY-frontend variant) ==="
echo "PT_HPU_LAZY_MODE : ${PT_HPU_LAZY_MODE} (discovery + symbol checks reference the LAZY plugin)"
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; habana_lazy::* syms runtime-resolved)"
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
  # ldd -r WILL list the lazy-frontend symbols (habana_lazy::*) as undefined --
  # EXPECTED (runtime-resolved from the imported lazy plugin). What must NOT be
  # undefined are the backend-lib symbols: OpBackend::BuildNode + KernelRegistry.
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols that the linked backend lib should provide (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry all resolve from the linked backend lib."
    LAZY_SO=""
    for cand in libhabana_pytorch_plugin.so libhabana_pytorch_plugin.upstream.so; do
      [[ -f "${HABF_LIB}/${cand}" ]] && { LAZY_SO="${HABF_LIB}/${cand}"; break; }
    done
    LAZY_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana_lazy|LazyOp|flush_op|AccThread' || true)"
    if [[ -n "${LAZY_UNDEF}" ]]; then
      n=$(echo "${LAZY_UNDEF}" | grep -c .)
      echo "Note: ${n} lazy-frontend symbol(s) left UNDEFINED on purpose (resolved at runtime from ${LAZY_SO##*/})."
    fi
    if [[ -n "${LAZY_SO}" ]]; then
      LAZY_FLUSH_DEF="$(nm -DC "${LAZY_SO}" 2>/dev/null | grep -E ' [TWB] ' | grep 'habana_lazy::flush_op' || true)"
      if [[ -n "${LAZY_FLUSH_DEF}" ]]; then
        echo "OK: runtime resolver present -- habana_lazy::flush_op is a global dynamic symbol in ${LAZY_SO##*/}."
      else
        echo "!!! WARNING: could not confirm habana_lazy::flush_op in ${LAZY_SO##*/}; runtime load may still work if provided elsewhere."
      fi
      LAZY_ACC_DEF="$(nm -DC "${LAZY_SO}" 2>/dev/null | grep -E ' [TWB] ' | grep 'habana_lazy::AccThread::Get' || true)"
      [[ -n "${LAZY_ACC_DEF}" ]] && \
        echo "OK: habana_lazy::AccThread::Get is a global dynamic symbol in ${LAZY_SO##*/}." || \
        echo "Note: habana_lazy::AccThread::Get not confirmed via nm -D (may be inlined/hidden); flush_op presence is the load-critical one."
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'indexer_topk|DsaIndexerTopk' | head
  fi
  echo
  echo "On the HPU box (lazy):  cd ${HERE} && PT_HPU_LAZY_MODE=1 DSA_INDEXER_SO=${OUT} python tests/test_dsa_indexer_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
