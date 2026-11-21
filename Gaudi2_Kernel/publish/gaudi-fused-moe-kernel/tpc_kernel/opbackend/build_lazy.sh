#!/usr/bin/env bash
# =============================================================================
# build_lazy.sh -- build moe_fused_expert_backend_lazy.cpp into
#   build/libmoe_fused_expert_lazy.so   (the LAZY-FRONTEND variant)
#
# This is build.sh with the deltas needed for the LAZY frontend:
#   - SRC = the lazy backend TU; OUT = build/libmoe_fused_expert_lazy.so
#   - the torch/habana discovery python + the symbol-presence check run UNDER
#     PT_HPU_LAZY_MODE=1, so the LAZY plugin (libhabana_pytorch_plugin.so) is the
#     one habana_frameworks.torch references -- which is exactly the plugin that
#     resolves this .so's habana_lazy::* frontend symbols at runtime.
#   - we link ONLY libhabana_pytorch_backend.so (NEVER a plugin) so loading our
#     .so does NOT re-run the plugin's static initializers (which would re-register
#     hpu::habana_d2d_memcpy_other -> "already registered!" abort). The
#     habana_lazy::* frontend symbols (flush_op, AccThread::Get, the LazyOp
#     templates) are left UNDEFINED and resolve at RUNTIME from the lazy plugin
#     already loaded by `import habana_frameworks.torch` under lazy mode.
#   - the symbol-presence check targets the LAZY plugin and asserts
#     habana_lazy::flush_op (the canonical non-inline symbol the LazyOp path +
#     RUN_MAYBE_WITH_ACC_THREAD macro need) is a global dynamic symbol there.
# EVERYTHING ELSE (bridge headers, abseil/env_flags prereqs, ABI=1, the
# backend-lib-only link, -Wl,--allow-shlib-undefined / --as-needed, the
# OpBackend/BuildNode/KernelRegistry link-gap check) is byte-identical to the
# proven build.sh -- keep in lockstep if build.sh's link flags change.
#
# Run (inside the dev image, NO device):
#   docker run --rm \
#     -v <repo>/tpc_kernel:/work -w /work/opbackend \
#     vllm-gaudi:next bash build_lazy.sh
#
# Produces: build/libmoe_fused_expert_lazy.so  (loaded on the HPU box by the
# capture test via torch.ops.load_library under PT_HPU_LAZY_MODE=1).
# RUN ON THE HPU BOX ONLY for the device test.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
BRIDGE="${HERE}/bridge"
mkdir -p "${BUILD}"

SRC="${HERE}/moe_fused_expert_backend_lazy.cpp"
OUT="${BUILD}/libmoe_fused_expert_lazy.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

# Reference the LAZY plugin for the discovery python + symbol checks below. Under
# PT_HPU_LAZY_MODE=1, `import habana_frameworks.torch` loads
# libhabana_pytorch_plugin.so (the lazy plugin) -- the exact lib that resolves
# this .so's habana_lazy::* frontend symbols at runtime on the box.
export PT_HPU_LAZY_MODE=1

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! ${BRIDGE} missing. Clone it first:"
  echo "    git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git ${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (identical to build.sh) ----
# (1) Abseil C++ headers: a hard transitive dep of op_backend.h / habana_operator.h
#     / lazy_kernels.h. NOT shipped in the dev image; install the apt headers (the
#     prebuilt libs provide the actual Abseil symbols, so only header types match).
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

# ---- torch + habana_frameworks discovery (under PT_HPU_LAZY_MODE=1) ----
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
# OpBackend::BuildNode, the KernelRegistry()/RegisterKernel singleton,
# HabanaOperator. Pick whichever variant exists. This is the ONLY habana lib we
# link (NEVER a plugin -- see the rationale block below).
BACKEND_LIB_NAME="habana_pytorch_backend"
if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
      ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
  BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
fi

# !!! DO NOT link the plugin libs (libhabana_pytorch_plugin / libhabana_pytorch2_plugin).
# ROOT CAUSE of the "hpu::habana_d2d_memcpy_other is already registered!" abort:
# the plugin libs carry the static initializer
#   REGISTER_HPU_BACKEND("hpu::habana_d2d_memcpy_other", MemCopyOperator)
# (+ the TORCH_LIBRARY(hpu) op-schema defs). habana_frameworks.torch loads the
# plugin at import, so its static initializers ALREADY populated the process-global
# KernelRegistry()/dispatcher. DT_NEEDED-linking the plugin would re-run those
# constructors on load -> a SECOND KernelRegistry().add("hpu::habana_d2d_..") ->
# RegisterKernel::add throws (habana_operator.h:799). The backend lib does NOT carry
# those registrars, so linking only it is safe.
#
# The LAZY frontend symbols our TU references (habana_lazy::flush_op,
# habana_lazy::AccThread::Get, and the LazyOp<at::Tensor> template machinery --
# create_node / empty_hpu_lazy / GetHbLazyTensor) live in the LAZY plugin
# libhabana_pytorch_plugin.so, which IS loaded in-process by
# `import habana_frameworks.torch` under PT_HPU_LAZY_MODE=1. We leave those lazy
# symbols UNDEFINED in our .so (-Wl,--allow-shlib-undefined) and let the dynamic
# linker resolve them from the running process image at load time. This is safe
# because our .so is only ever loaded AFTER habana_frameworks.torch is imported
# (torch.ops.load_library happens post-import). The runtime presence of the key
# lazy symbol is asserted below via `nm -D` against the loaded LAZY plugin.

# Bridge transitive-include roots (identical to build.sh).
BRIDGE_INCS=(
  -I "${BRIDGE}"
  -I "${BRIDGE}/generated"          # generated/env_flags/env_flags_generated.h (built above)
  -I "${BRIDGE}/pytorch_helpers"
  -I "${BRIDGE}/include"
  -I "${BRIDGE}/python_packages/habana_frameworks/torch/jit/csrc"  # jit_fork/ir/ir.h (habana_torch::jit)
  -I "${BRIDGE}/third_party_vendored"          # magic_enum-0.9.7/include/magic_enum/magic_enum.hpp
)

echo "=== opbackend/build_lazy.sh (LAZY-frontend variant) ==="
echo "PT_HPU_LAZY_MODE : ${PT_HPU_LAZY_MODE} (discovery + symbol checks reference the LAZY plugin)"
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; habana_lazy::* frontend syms runtime-resolved from the imported lazy plugin)"
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
  # symbols at RUNTIME from the already-imported (lazy) habana_frameworks.torch
  # process image. So `ldd -r` WILL list the lazy-frontend symbols (habana_lazy::*,
  # e.g. flush_op / AccThread::Get / the LazyOp template helpers) as undefined --
  # that is EXPECTED. What must NOT be undefined are the symbols the backend lib we
  # DO link is supposed to provide: OpBackend::BuildNode + the KernelRegistry/
  # RegisterKernel singleton. Those being undefined would be a real link gap.
  # NB: each ldd line ends with "(<path-to-our.so>)" and that path contains
  # "opbackend" -- so match on the SYMBOL field only (strip the trailing
  # parenthetical) or "-i OpBackend" would match every line via the path.
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  # The backend lib we DO link must provide OpBackend::BuildNode + the
  # KernelRegistry/RegisterKernel singleton (mangled: ...OpBackend...BuildNode...,
  # ...RegisterKernel..., ...KernelRegistry...).
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols that the linked backend lib should provide (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry all resolve from the linked backend lib."
    # The LAZY-frontend symbols are intentionally runtime-resolved. PROVE the key
    # ones are present in the LAZY plugin that habana_frameworks.torch loads under
    # PT_HPU_LAZY_MODE=1, so the runtime resolution at load() time is guaranteed.
    LAZY_SO=""
    for cand in libhabana_pytorch_plugin.so libhabana_pytorch_plugin.upstream.so; do
      [[ -f "${HABF_LIB}/${cand}" ]] && { LAZY_SO="${HABF_LIB}/${cand}"; break; }
    done
    LAZY_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana_lazy|LazyOp|flush_op|AccThread' || true)"
    if [[ -n "${LAZY_UNDEF}" ]]; then
      n=$(echo "${LAZY_UNDEF}" | grep -c .)
      echo "Note: ${n} lazy-frontend symbol(s) left UNDEFINED on purpose (resolved at runtime from ${LAZY_SO##*/})."
    fi
    # habana_lazy::flush_op is the canonical non-inline symbol the LazyOp path +
    # RUN_MAYBE_WITH_ACC_THREAD (MAYBE_FLUSH_OP) need; prove it is a GLOBAL DYNAMIC
    # symbol (mangled: ...habana_lazy...flush_op...) in the lazy plugin so the
    # runtime cross-.so resolution at load is guaranteed. Match on the DYNAMIC
    # symtab (nm -D, demangled) and require a global def.
    # NB: do NOT pipe into `grep -q` under `set -o pipefail` (the early -q close
    # SIGPIPEs upstream and the pipeline reports failure). Capture + test instead.
    if [[ -n "${LAZY_SO}" ]]; then
      LAZY_FLUSH_DEF="$(nm -DC "${LAZY_SO}" 2>/dev/null | grep -E ' [TWB] ' | grep 'habana_lazy::flush_op' || true)"
      if [[ -n "${LAZY_FLUSH_DEF}" ]]; then
        echo "OK: runtime resolver present -- habana_lazy::flush_op is a global dynamic symbol in ${LAZY_SO##*/}."
      else
        echo "!!! WARNING: could not confirm habana_lazy::flush_op in ${LAZY_SO##*/}; runtime load may still work if the symbol is provided elsewhere in-process."
      fi
      # AccThread::Get is the other macro-required symbol; confirm it too (advisory).
      LAZY_ACC_DEF="$(nm -DC "${LAZY_SO}" 2>/dev/null | grep -E ' [TWB] ' | grep 'habana_lazy::AccThread::Get' || true)"
      [[ -n "${LAZY_ACC_DEF}" ]] && \
        echo "OK: habana_lazy::AccThread::Get is a global dynamic symbol in ${LAZY_SO##*/}." || \
        echo "Note: habana_lazy::AccThread::Get not confirmed via nm -D (may be inlined/hidden); flush_op presence is the load-critical one."
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'fused_expert|MoeFusedExpert' | head
  fi
  echo
  echo "On the HPU box (capture test):"
  echo "  PT_HPU_LAZY_MODE=1 MOE_OP_SO=${OUT} python tests/test_fused_op_capture.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
