#!/usr/bin/env bash
# =============================================================================
# build_fp8_lazy.sh -- build moe_fused_expert_fp8_backend_lazy.cpp into
#   build/libmoe_fused_expert_fp8_lazy.so   (the fp8-COMPUTE, LAZY-FRONTEND variant)
#
# This is build_fp8.sh (the fp8 deltas: SRC=the fp8 LAZY backend, OUT=the fp8 lazy
# .so, -DFUSED_FP8=1 + optional -DBUILD_FP8_NODES=1) COMBINED with build_lazy.sh's
# LAZY-frontend deltas:
#   - the torch/habana discovery python + the symbol-presence check run UNDER
#     PT_HPU_LAZY_MODE=1, so habana_frameworks.torch references the LAZY plugin
#     (libhabana_pytorch_plugin.so) -- the exact plugin that resolves this .so's
#     habana_lazy::* frontend symbols (flush_op, AccThread::Get, the LazyOp
#     templates) at RUNTIME on the box.
#   - link ONLY libhabana_pytorch_backend.so (NEVER a plugin) so loading our .so
#     does NOT re-run the plugin static initializers (the d2d_memcpy_other re-reg
#     abort). The lazy-frontend symbols stay UNDEFINED (-Wl,--allow-shlib-undefined)
#     and resolve at load from the already-imported lazy plugin.
# EVERYTHING ELSE (bridge headers, abseil/env_flags prereqs, ABI, the OpBackend/
# BuildNode/KernelRegistry link-gap check) is byte-identical to build_fp8.sh /
# build_lazy.sh. Keep in lockstep if either changes.
#
# Run (inside the dev image, NO device):
#   docker run --rm -e PT_HPU_LAZY_MODE=1 \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/opbackend \
#     vllm-gaudi:next bash -c 'BUILD_FP8_NODES=1 bash build_fp8_lazy.sh'
#
# Produces: build/libmoe_fused_expert_fp8_lazy.so (loaded by the LAZY serve via
# fp8_routed_moe._load_fp8_fused_op under PT_HPU_LAZY_MODE=1). RUN ON THE HPU BOX.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
BRIDGE="${HERE}/bridge"
mkdir -p "${BUILD}"

SRC="${HERE}/moe_fused_expert_fp8_backend_lazy.cpp"
OUT="${BUILD}/libmoe_fused_expert_fp8_lazy.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"

# Reference the LAZY plugin for the discovery python + symbol checks below. Under
# PT_HPU_LAZY_MODE=1, `import habana_frameworks.torch` loads the lazy plugin --
# the exact lib that resolves this .so's habana_lazy::* frontend symbols at runtime.
export PT_HPU_LAZY_MODE=1

# -DFUSED_FP8=1 marks the fp8 build; BUILD_FP8_NODES (default 0 in the source) can
# be forced on from the environment to emit the real fp8 nodes.
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

# ---- build-only prerequisites (identical to build_fp8.sh / build_lazy.sh) ----
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

# !!! DO NOT link the plugin libs -- see build_lazy.sh's rationale block (the
# d2d_memcpy_other re-registration abort). The habana_lazy::* frontend symbols this
# TU references resolve at RUNTIME from the lazy plugin imported in-process.

BRIDGE_INCS=(
  -I "${BRIDGE}"
  -I "${BRIDGE}/generated"
  -I "${BRIDGE}/pytorch_helpers"
  -I "${BRIDGE}/include"
  -I "${BRIDGE}/python_packages/habana_frameworks/torch/jit/csrc"
  -I "${BRIDGE}/third_party_vendored"
)

echo "=== opbackend/build_fp8_lazy.sh (fp8-compute, LAZY-frontend variant) ==="
echo "PT_HPU_LAZY_MODE : ${PT_HPU_LAZY_MODE} (discovery + symbol checks reference the LAZY plugin)"
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; habana_lazy::* frontend syms runtime-resolved)"
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
  echo "=== link-gap check (OpBackend/BuildNode/KernelRegistry must resolve; "
  echo "    EagerOpBase/habana_lazy undefined is EXPECTED + runtime-resolved) ==="
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry resolve from the linked backend lib."
    # The LAZY-frontend symbols are intentionally runtime-resolved. PROVE the key one
    # (habana_lazy::flush_op) is a global dynamic symbol in the LAZY plugin that
    # habana_frameworks.torch loads under PT_HPU_LAZY_MODE=1, so the load-time
    # resolution is guaranteed. (Mirrors build_lazy.sh.)
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
        echo "!!! WARNING: could not confirm habana_lazy::flush_op in ${LAZY_SO##*/}; runtime load may still work if provided elsewhere in-process."
      fi
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'fused_expert_fp8|MoeFusedExpertFp8' | head
  fi
  echo
  echo "On the HPU box:  cd $(dirname "${HERE}")/opbackend && PT_HPU_LAZY_MODE=1 python test_fused_expert_fp8_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
