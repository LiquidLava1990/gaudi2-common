#!/usr/bin/env bash
# =============================================================================
# build_fast_glue.sh — build ONLY the PT2 custom-op glue for the OPTIMIZED dequant
# kernels (moe_ext_custom_op_fast.cpp -> build_fast/libmoe_ext_custom_op_fast.so).
#
# WHY A SEPARATE SCRIPT: build_fast.sh builds the TPC perf-lib (the GUIDs). This
# script builds the torch BINDING that makes those GUIDs callable as
#   torch.ops.hpu_moe_ext.unpack_dequant_w4_{fast,lean}
# so the on-card A/B (tests/dequant_fast_oncard_validate.py) invokes a GENUINELY
# DISTINCT kernel instead of silently falling back to convert_from_uint4 (a false
# null). It is ADDITIVE: it touches no file the in-flight A/B uses and does not
# modify build_fast.sh / moe_ext_custom_op.cpp / moe_ext_perflib*.cpp.
#
# Host x86-64 C++ ONLY (system g++, NOT tpc-clang; no gaudi2.bc). Mirrors build.sh
# Step 3 EXACTLY (incl. the cxx11-ABI match — the load-bearing fix that avoids the
# `undefined symbol: _ZN5torch7LibraryC1ENS0_4KindESs...` ABI mismatch).
#
# Run CPU-ONLY inside the dev image (NO HPU device flags, NO --device=/dev/accel):
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work \
#     vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
#     bash build_fast_glue.sh
#
# Produces: build_fast/libmoe_ext_custom_op_fast.so
#   loaded on the box via torch.ops.load_library(...).
# Run build_fast.sh FIRST (the perf-lib must exist for the GUIDs to instantiate).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build_fast"
mkdir -p "${BUILD}"

HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
GLUE_CPP="${HERE}/moe_ext_custom_op_fast.cpp"

if [[ ! -f "${GLUE_CPP}" ]]; then
  echo "!!! FATAL: glue source not found: ${GLUE_CPP}"; exit 2
fi

TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))' 2>/dev/null)"
if [[ -z "${TORCH_DIR}" || -z "${HABF_DIR}" ]]; then
  echo "!!! FATAL: torch / habana_frameworks not importable — run INSIDE the dev image."
  exit 3
fi
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))' 2>/dev/null)"

# Match torch's libstdc++ ABI (build.sh Step 3 note: a hardcoded =0 mangles
# std::string the old way and torch (cxx11) won't export it -> load-time undefined
# symbol). QUERY the live torch.
ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"

# !!! DO NOT link a plugin lib (libhabana_pytorch2_plugin / libhabana_pytorch_plugin).
# ROOT CAUSE of the "hpu::habana_d2d_memcpy_other is already registered!" abort at
# torch.ops.load_library/import torch on :next: the plugin libs carry the static
# initializer  habana::KernelRegistry().REGISTER_HPU_BACKEND("hpu::habana_d2d_memcpy_other",
# MemCopyOperator)  (bridge/habana_kernels/basic_kernels.cpp's BasicKernelsKernelRegistry).
# `import habana_frameworks.torch` already ran those ctors; DT_NEEDED-linking the plugin
# re-runs them on dlopen -> a SECOND KernelRegistry().add(...) -> throws at
# habana_operator.h:799. The ONLY backend symbols this glue uses (registerUserCustomOp,
# UserCustomOpDescriptor::getUserCustomOpDescriptor, .execute) live in
# libhabana_pytorch_backend.so (bridge/hpu_ops/user_custom_op_descriptor.cpp), which does NOT
# carry those registrars. Link ONLY the backend lib (the proven OpBackend recipe that loads
# clean on :next) + -Wl,--allow-shlib-undefined; this mirrors opbackend/build_*_lazy.sh.
BACKEND_LIB_NAME="habana_pytorch_backend"
if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
      ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
  BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
fi

echo "=== build_fast_glue.sh ==="
echo "glue .cpp : ${GLUE_CPP}"
echo "torch     : ${TORCH_DIR}"
echo "habana_fw : ${HABF_DIR}"
echo "hab inc   : ${HAB_INC}"
echo "py inc    : ${PY_INC}"
echo "cxx11 abi : ${ABI}  (matched to the installed torch)"
echo "backend   : -l${BACKEND_LIB_NAME}  (plugin NOT linked — avoids the d2d_memcpy_other double-reg on :next)"

set -x
g++ -O2 -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI}" \
  "${GLUE_CPP}" \
  -I "${TORCH_INC}" \
  -I "${TORCH_INC}/torch/csrc/api/include" \
  -I "${HABF_INC}" \
  -I "${HAB_INC}" \
  -I "${PY_INC}" \
  -L "${TORCH_LIB}" -L "${HABF_LIB}" \
  -Wl,--as-needed \
  -ltorch -ltorch_cpu -lc10 \
  -l"${BACKEND_LIB_NAME}" \
  -Wl,--no-as-needed \
  -Wl,--allow-shlib-undefined \
  -Wl,-rpath,"${TORCH_LIB}" -Wl,-rpath,"${HABF_LIB}" \
  -o "${BUILD}/libmoe_ext_custom_op_fast.so"
RC=$?
set +x

echo
echo "=== build summary ==="
echo "  PT2 fast glue .so : rc=${RC}  ${BUILD}/libmoe_ext_custom_op_fast.so"
if [[ ${RC} -eq 0 ]]; then
  # GUARD (card-free): the plugin must NOT be a DT_NEEDED, else dlopen re-runs its
  # static initializers -> "hpu::habana_d2d_memcpy_other is already registered!".
  PLUGIN_NEEDED="$(objdump -p "${BUILD}/libmoe_ext_custom_op_fast.so" 2>/dev/null \
                   | grep -E 'NEEDED.*habana_pytorch(2)?_plugin' || true)"
  if [[ -n "${PLUGIN_NEEDED}" ]]; then
    echo "!!! DOUBLE-REG REGRESSION: a plugin is DT_NEEDED (would abort on load):"
    echo "${PLUGIN_NEEDED}"
    echo "!!! Remove the -lhabana_pytorch*_plugin from the g++ link line."
    RC=5
  else
    echo "OK: no plugin in DT_NEEDED (won't re-run the d2d_memcpy_other registrar)."
  fi
  echo "--- undefined symbols sanity (lazy/eager frontend syms may be runtime-resolved) ---"
  ldd -r "${BUILD}/libmoe_ext_custom_op_fast.so" 2>&1 | grep -i "undefined symbol" | head -5 \
    || echo "  (no undefined-symbol lines from ldd -r)"
  echo
  echo "  ON THE HPU BOX (cards free), the on-card validate loads this automatically:"
  echo "    export GC_KERNEL_PATH=${BUILD}/libmoe_ext_perflib_fast.so:\$GC_KERNEL_PATH"
  echo "    PT_HPU_LAZY_MODE=1 python tests/dequant_fast_oncard_validate.py --variant 2  # LEAN first"
else
  echo "  FAILED — inspect the g++ error above."
fi
exit ${RC}
