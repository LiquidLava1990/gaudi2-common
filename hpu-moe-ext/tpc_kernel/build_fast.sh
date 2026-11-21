#!/usr/bin/env bash
# =============================================================================
# build_fast.sh — build the OPTIMIZED dequant perf-lib (moe_ext_perflib_fast.cpp).
#
# NEW build script. Does NOT touch the live build.sh / build_perflib.sh outputs or
# any file the in-flight A/B uses. Outputs go to tpc_kernel/build_fast/.
#
# Pipeline (mirrors the PROVEN fusion/build_perflib.sh, which builds rc=0):
#   [1] tpc-clang : moe_ext_unpack_dequant_w4_fast.c -> TWO ELFs
#         -DKERNEL_VARIANT=1 -> moe_ext_unpack_dequant_w4_fast.o   (native unpack)
#         -DKERNEL_VARIANT=2 -> moe_ext_unpack_dequant_w4_lean.o   (U4 descriptor)
#   [2] llvm-objcopy -I binary : embed each .o, exporting
#         _binary_moe_ext_unpack_dequant_w4_fast_o_{start,end}
#         _binary_moe_ext_unpack_dequant_w4_lean_o_{start,end}
#   [3] g++ (HOST x86) : moe_ext_perflib_fast.cpp + both embedded blobs
#         -> libmoe_ext_perflib_fast.so (exports GetKernelGuids/InstantiateTpcKernel)
#   [4] g++ (HOST x86) : moe_ext_custom_op_fast.cpp (links Torch + Habana plugin)
#         -> libmoe_ext_custom_op_fast.so (PT2 glue binding the two NEW GUIDs to
#            torch.ops.hpu_moe_ext.unpack_dequant_w4_{fast,lean}). Without this the
#            on-card A/B has NO real op for the new kernel (the oracle-vs-itself
#            silent-pass trap). Mirrors the proven build.sh Step 3 ABI-matched recipe.
#
# Run CPU-ONLY inside the 1.24 dev image (NO HPU device flags):
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work \
#     vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
#     bash build_fast.sh
#
# Then ON THE HPU BOX (cards free):
#   export GC_KERNEL_PATH=$PWD/build_fast/libmoe_ext_perflib_fast.so:$GC_KERNEL_PATH
#   python tests/dequant_fast_oncard_validate.py   # bit-exact vs convert_from_uint4
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build_fast"
mkdir -p "${BUILD}"

TPC_CLANG="${TPC_CLANG:-/opt/habanalabs/bin/tpc-clang}"
OBJCOPY="${OBJCOPY:-/opt/habanalabs/bin/llvm-objcopy}"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
KERNEL_C="${HERE}/moe_ext_unpack_dequant_w4_fast.c"
SHIM_CPP="${HERE}/moe_ext_perflib_fast.cpp"
GLUE_CPP="${HERE}/moe_ext_custom_op_fast.cpp"

echo "=== build_fast.sh ==="
echo "tpc-clang : ${TPC_CLANG} ($("${TPC_CLANG}" --version 2>/dev/null | head -1))"
echo "objcopy   : ${OBJCOPY}"
echo "kernel .c : ${KERNEL_C}"
echo "shim .cpp : ${SHIM_CPP}"

build_variant () {
  local variant_def="$1"   # KERNEL_VARIANT value (1=fast, 2=lean)
  local stem="$2"          # output object stem (must match the shim extern decls)
  echo
  echo "=== TPC compile (KERNEL_VARIANT=${variant_def}) -> ${stem}.o ==="
  set -x
  "${TPC_CLANG}" -mcpu=gaudi2 -O2 -DKERNEL_VARIANT="${variant_def}" \
    -c "${KERNEL_C}" -I "${HAB_INC}" -o "${BUILD}/${stem}.o"
  local rc=$?
  set +x
  if [[ ${rc} -ne 0 ]]; then
    echo "!!! TPC compile FAILED rc=${rc} for variant ${variant_def}."
    return ${rc}
  fi
  # Embed as a binary blob (objcopy derives symbol stem from the basename, so run
  # from inside BUILD on the bare basename to get the clean _binary_<stem>_o_*).
  echo "=== embed ${stem}.o ==="
  set -x
  ( cd "${BUILD}" && "${OBJCOPY}" -I binary -O elf64-x86-64 \
      --binary-architecture i386:x86-64 "${stem}.o" "${stem}_embed.o" )
  rc=$?
  set +x
  [[ ${rc} -ne 0 ]] && { echo "!!! objcopy embed FAILED rc=${rc}"; return ${rc}; }
  return 0
}

build_variant 1 moe_ext_unpack_dequant_w4_fast || exit 1
build_variant 2 moe_ext_unpack_dequant_w4_lean || exit 1

echo
echo "=== link shim + both embedded ELFs -> libmoe_ext_perflib_fast.so ==="
set -x
g++ -O2 -fPIC -shared -std=c++17 \
  "${SHIM_CPP}" \
  "${BUILD}/moe_ext_unpack_dequant_w4_fast_embed.o" \
  "${BUILD}/moe_ext_unpack_dequant_w4_lean_embed.o" \
  -I "${HAB_INC}" \
  -Wl,-z,noexecstack \
  -o "${BUILD}/libmoe_ext_perflib_fast.so"
SHIM_RC=$?
set +x

# =============================================================================
# [4] PT2 glue .so — binds the two NEW GUIDs to torch.ops.hpu_moe_ext.*.
# HOST x86 g++ (NOT tpc-clang). ABI-matched to the installed torch EXACTLY as
# build.sh Step 3 (querying torch._C._GLIBCXX_USE_CXX11_ABI is mandatory: a
# mismatched ABI yields a .so whose torch::Library ctor is an undefined symbol).
# Skipped (non-fatal) only if torch isn't importable in this container — but the
# 1.24 pytorch-installer image HAS torch, so this should build.
# =============================================================================
GLUE_RC=0
echo
echo "=== [4] PT2 glue compile: ${GLUE_CPP##*/} -> libmoe_ext_custom_op_fast.so ==="
TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))' 2>/dev/null)"
if [[ -z "${TORCH_DIR}" || -z "${HABF_DIR}" ]]; then
  echo "!!! torch / habana_frameworks not importable in this container."
  echo "!!! The perf-lib (.so above) is built; the PT2 glue needs the dev image."
  echo "!!! Re-run inside vault.habana.ai/.../pytorch-installer to get the glue .so."
  GLUE_RC=2
else
  TORCH_INC="${TORCH_DIR}/include"
  TORCH_LIB="${TORCH_DIR}/lib"
  HABF_INC="${HABF_DIR}/include"
  HABF_LIB="${HABF_DIR}/lib"
  PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))' 2>/dev/null)"
  ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"

  # !!! DO NOT link a plugin lib (libhabana_pytorch2_plugin / libhabana_pytorch_plugin).
  # ROOT CAUSE of the "hpu::habana_d2d_memcpy_other is already registered!" abort at
  # torch.ops.load_library/import torch on :next: the plugin libs carry the static
  # initializer  habana::KernelRegistry().REGISTER_HPU_BACKEND("hpu::habana_d2d_memcpy_other",
  # MemCopyOperator)  (bridge/habana_kernels/basic_kernels.cpp's BasicKernelsKernelRegistry,
  # + the TORCH_LIBRARY(hpu) schema defs). `import habana_frameworks.torch` already loads
  # the plugin at import -> those ctors ALREADY populated the process-global KernelRegistry().
  # DT_NEEDED-linking the plugin re-runs them when dlopen loads this glue -> a SECOND
  # add("hpu::habana_d2d_memcpy_other") -> RegisterKernel::add throws (habana_operator.h:799).
  # The ONLY backend symbols this glue references — registerUserCustomOp,
  # UserCustomOpDescriptor::getUserCustomOpDescriptor, .execute — live in
  # libhabana_pytorch_backend.so (bridge/hpu_ops/user_custom_op_descriptor.cpp), which does
  # NOT carry those registrars. So link ONLY the backend lib (mirrors the proven OpBackend
  # opbackend/build_*.sh recipe that loads clean on :next), leave any stray transitive bridge
  # symbols UNDEFINED (-Wl,--allow-shlib-undefined; there are none in practice), and let the
  # dynamic linker resolve from the already-imported in-process libs.
  BACKEND_LIB_NAME="habana_pytorch_backend"
  if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
        ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
    BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
  fi
  echo "torch    : ${TORCH_DIR}"
  echo "habana_fw: ${HABF_DIR}"
  echo "cxx11 abi: ${ABI}  (matched to the installed torch)"
  echo "backend  : -l${BACKEND_LIB_NAME}  (plugin NOT linked — avoids the d2d_memcpy_other double-reg on :next)"
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
  GLUE_RC=$?
  set +x
fi

echo
echo "=== build summary ==="
echo "  perf-lib .so : rc=${SHIM_RC}  ${BUILD}/libmoe_ext_perflib_fast.so"
echo "  PT2 glue .so : rc=${GLUE_RC}  ${BUILD}/libmoe_ext_custom_op_fast.so"
if [[ ${SHIM_RC} -eq 0 ]]; then
  echo "--- exported entry points (must include GetKernelGuids / InstantiateTpcKernel) ---"
  nm -D -C "${BUILD}/libmoe_ext_perflib_fast.so" 2>/dev/null | \
    grep -E "GetKernelGuids|InstantiateTpcKernel|GetLibVersion" || \
    echo "  (nm unavailable; check on box)"
  echo "--- embedded kernel symbols ---"
  nm "${BUILD}/libmoe_ext_perflib_fast.so" 2>/dev/null | grep -i binary || true
fi
if [[ ${GLUE_RC} -eq 0 ]]; then
  # GUARD (card-free): the plugin must NOT be a DT_NEEDED, else dlopen re-runs its
  # static initializers -> "hpu::habana_d2d_memcpy_other is already registered!".
  PLUGIN_NEEDED="$(objdump -p "${BUILD}/libmoe_ext_custom_op_fast.so" 2>/dev/null \
                   | grep -E 'NEEDED.*habana_pytorch(2)?_plugin' || true)"
  if [[ -n "${PLUGIN_NEEDED}" ]]; then
    echo "!!! DOUBLE-REG REGRESSION: a plugin is DT_NEEDED (aborts on load):"
    echo "${PLUGIN_NEEDED}"
    GLUE_RC=5
  else
    echo "--- glue: OK, no plugin in DT_NEEDED (won't re-run the d2d_memcpy_other registrar) ---"
  fi
  echo "--- glue: defined custom-op symbols (resolved from libhabana_pytorch_backend at load) ---"
  nm -D -C "${BUILD}/libmoe_ext_custom_op_fast.so" 2>/dev/null | \
    grep -E "registerUserCustomOp|getUserCustomOpDescriptor" | head -4 || true
fi
echo
echo "  ON THE HPU BOX (cards free):"
echo "    bash tpc_kernel/oncard_dequant_ab.sh   # build-already-done aware; runs the A/B"
echo "  or manually:"
echo "    export GC_KERNEL_PATH=${BUILD}/libmoe_ext_perflib_fast.so:\$GC_KERNEL_PATH"
echo "    PT_HPU_LAZY_MODE=1 python tests/dequant_fast_oncard_validate.py --variant 2  # LEAN (primary)"
echo "    PT_HPU_LAZY_MODE=1 python tests/dequant_fast_oncard_validate.py --variant 1  # FAST"

# perf-lib is the hard requirement; glue rc=2 (no torch in container) is a soft skip.
if [[ ${SHIM_RC} -ne 0 ]]; then exit ${SHIM_RC}; fi
if [[ ${GLUE_RC} -eq 1 ]]; then exit 1; fi   # genuine compile error
exit 0
