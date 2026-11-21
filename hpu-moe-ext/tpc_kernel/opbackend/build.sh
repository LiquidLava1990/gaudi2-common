#!/usr/bin/env bash
# =============================================================================
# build.sh -- APPROACH 3: build moe_fused_expert_backend.cpp into
#   build/libmoe_fused_expert.so
#
# CPU-only HOST x86-64 compile (system g++). NO HPU device flags (NO
# --runtime=habana, NO HABANA_VISIBLE_DEVICES, NO /dev/accel). It only compiles
# C++ against the cloned version-exact bridge headers (./bridge, tag
# v1.24.0-1007) + torch 2.10 + synapse headers, and LINKS the SHIPPED backend
# lib (libhabana_pytorch_backend.so) which already contains OpBackend::BuildNode,
# the KernelRegistry, habana_lazy::LazyOp, the GUID lowering, etc.
#
# Run (inside the dev image, no device):
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/opbackend \
#     vllm-gaudi:next bash build.sh
#
# Produces: build/libmoe_fused_expert.so  (loaded on the HPU box by the test via
# torch.ops.load_library). RUN ON THE HPU BOX ONLY.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
BRIDGE="${HERE}/bridge"
mkdir -p "${BUILD}"

SRC="${HERE}/moe_fused_expert_backend.cpp"
OUT="${BUILD}/libmoe_fused_expert.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! ${BRIDGE} missing. Clone it first:"
  echo "    git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git ${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (CPU headers only; NO device, NO --runtime=habana) ----
# (1) Abseil C++ headers: a hard transitive dep of op_backend.h / habana_operator.h.
#     NOT shipped in the dev image; install the apt headers (the prebuilt backend
#     .so provides the actual Abseil symbols, so only the header types must match).
if [[ ! -f /usr/include/absl/functional/any_invocable.h ]]; then
  echo "=== installing libabsl-dev (build-only headers) ==="
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libabsl-dev >/dev/null 2>&1 \
    || { echo "!!! could not apt-install libabsl-dev"; exit 3; }
fi

# (2) Generate the ONE generated header our TU transitively needs
#     (generated/env_flags/env_flags_generated.h) from the in-repo YAML via the
#     in-repo generator. The bridge's other generated/* headers are pulled in only
#     by its own .cpp reference files, which we do NOT compile.
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

# The shipped backend lib carries the symbols our TU actually references at link
# time: OpBackend::BuildNode, the KernelRegistry()/RegisterKernel singleton, and
# HabanaOperator. The vllm-gaudi:next image is the pytorch-UPSTREAM build, where
# the resolving lib is libhabana_pytorch_backend.upstream.so. Pick whichever
# exists. This is the ONLY habana lib we link.
BACKEND_LIB_NAME="habana_pytorch_backend"
if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
      ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
  BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
fi

# !!! DO NOT link the plugin libs (libhabana_pytorch2_plugin / libhabana_pytorch_plugin).
# ROOT CAUSE of the "hpu::habana_d2d_memcpy_other is already registered!" abort:
# both plugin libs are compiled from basic_kernels.cpp, which carries the static
# initializer  REGISTER_HPU_BACKEND("hpu::habana_d2d_memcpy_other", MemCopyOperator)
# (and the TORCH_LIBRARY(hpu) op-schema defs). habana_frameworks.torch loads those
# plugins at import, so their static initializers ALREADY populated the process-
# global KernelRegistry()/dispatcher with all of habana's own ops. If we put those
# plugins in our .so's DT_NEEDED (which -l<plugin> + --no-as-needed did), loading
# our .so re-runs their constructors -> KernelRegistry().add("hpu::habana_d2d_..")
# a SECOND time -> RegisterKernel::add throws (habana_operator.h:799). The backend
# lib does NOT carry MemCopyOperator/the hpu registrars, so linking only it is safe.
#
# The EAGER frontend symbols our TU references (habana::eager::EagerOpBase::run /
# validate_inputs / m_mutex, HbEagerTensorPool::get_backend_tensor) live in the
# EAGER plugin libhabana_pytorch2_plugin.so, which IS loaded in-process by
# `import habana_frameworks.torch` in this pytorch-upstream/eager image. (The LAZY
# plugin libhabana_pytorch_plugin.so is NOT loaded here, so the lazy LazyOp path is
# unavailable -- hence the .cpp uses EagerOp, not LazyOp.) We leave those eager
# symbols UNDEFINED in our .so (-Wl,--allow-shlib-undefined) and let the dynamic
# linker resolve them from the running process image at load time. This is safe
# because our .so is only ever loaded AFTER habana_frameworks.torch is imported
# (torch.ops.load_library happens post-import). The runtime presence of the key
# eager symbol is asserted below via `nm -D` against the loaded eager plugin.

# Bridge transitive-include roots. The bridge includes are rooted at the project
# source dir (CMake: target_include_directories(... PUBLIC ${PROJECT_SOURCE_DIR})
# and .../pytorch_helpers), so put ${BRIDGE} and a few sub-roots on -I. Some
# third-party headers (absl, fmt, ...) are vendored by torch's include dir.
BRIDGE_INCS=(
  -I "${BRIDGE}"
  -I "${BRIDGE}/generated"          # generated/env_flags/env_flags_generated.h (built above)
  -I "${BRIDGE}/pytorch_helpers"
  -I "${BRIDGE}/include"
  -I "${BRIDGE}/python_packages/habana_frameworks/torch/jit/csrc"  # jit_fork/ir/ir.h (habana_torch::jit)
  # vendored single-header third-party deps the shipped habana headers #include
  # but that are NOT in the dev image (path prefixes match the #include lines).
  -I "${BRIDGE}/third_party_vendored"          # magic_enum-0.9.7/include/magic_enum/magic_enum.hpp
)

echo "=== opbackend/build.sh (approach 3) ==="
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; eager frontend syms resolved at runtime from the imported habana_frameworks.torch)"
echo "bridge      : ${BRIDGE}"
echo "hab inc     : ${HAB_INC}"
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
  # We deliberately link ONLY the backend lib and resolve the lazy-frontend
  # symbols at RUNTIME from the already-imported habana_frameworks.torch process
  # image (see the lib block above). So `ldd -r` WILL list the lazy-frontend
  # symbols (habana_lazy::*, e.g. mul_out_hpu_lazy / LazyOp) as undefined -- that
  # is EXPECTED. What must NOT be undefined are the symbols the backend lib we DO
  # link is supposed to provide: OpBackend::BuildNode + the KernelRegistry/
  # RegisterKernel singleton. Those being undefined would be a real link gap.
  # NB: each ldd line ends with "(<path-to-our.so>)", and that path literally
  # contains "opbackend" -- so we must match on the SYMBOL field only (strip the
  # trailing parenthetical) or "-i OpBackend" would match every single line via
  # the path. Cut everything from the last tab/space-paren onward.
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  # The backend lib we DO link must provide OpBackend::BuildNode + the
  # KernelRegistry/RegisterKernel singleton. Those appear in mangled form as
  # ...OpBackend...BuildNode..., ...RegisterKernel..., ...KernelRegistry... .
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols that the linked backend lib should provide (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry all resolve from the linked backend lib."
    # The EAGER-frontend symbols are intentionally runtime-resolved. PROVE the key
    # one is present in the eager plugin that habana_frameworks.torch loads, so the
    # runtime resolution at torch.ops.load_library() time is guaranteed.
    EAGER_SO=""
    for cand in libhabana_pytorch2_plugin.so libhabana_pytorch2_plugin.upstream.so; do
      [[ -f "${HABF_LIB}/${cand}" ]] && { EAGER_SO="${HABF_LIB}/${cand}"; break; }
    done
    EAGER_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana::eager|EagerOp|HbEagerTensorPool' || true)"
    if [[ -n "${EAGER_UNDEF}" ]]; then
      n=$(echo "${EAGER_UNDEF}" | grep -c .)
      echo "Note: ${n} eager-frontend symbol(s) left UNDEFINED on purpose (resolved at runtime from ${EAGER_SO##*/})."
    fi
    # EagerOpBase::run is the canonical non-inline symbol EagerOp<at::Tensor>::call()
    # needs; prove it is a GLOBAL DYNAMIC symbol (mangled: ...EagerOpBase3run...) in
    # the eager plugin so the runtime cross-.so resolution at load is guaranteed.
    # Match on the DYNAMIC symtab (nm -D, demangled) and require a global def.
    # NB: do NOT pipe into `grep -q` here -- under `set -o pipefail` the early
    # close from -q sends SIGPIPE upstream and the whole pipeline reports failure.
    # Capture to a var and test emptiness instead.
    if [[ -n "${EAGER_SO}" ]]; then
      EAGER_RUN_DEF="$(nm -DC "${EAGER_SO}" 2>/dev/null | grep -E ' [TWB] ' | grep 'eager::EagerOpBase::run' || true)"
      if [[ -n "${EAGER_RUN_DEF}" ]]; then
        echo "OK: runtime resolver present -- habana::eager::EagerOpBase::run is a global dynamic symbol in ${EAGER_SO##*/}."
      else
        echo "!!! WARNING: could not confirm habana::eager::EagerOpBase::run in ${EAGER_SO##*/}; runtime load may still work if the symbol is provided elsewhere in-process."
      fi
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'fused_expert|MoeFusedExpert' | head
  fi
  echo
  echo "On the HPU box:  cd $(dirname "${HERE}")/opbackend && python test_fused_expert_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
