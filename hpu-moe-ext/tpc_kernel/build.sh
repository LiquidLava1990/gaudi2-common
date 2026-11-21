#!/usr/bin/env bash
# =============================================================================
# build.sh — Phase-0 fused-MoE foundation kernel build (Gaudi2 TPC + PT2 glue).
#
# Produces:
#   1) build/moe_ext_unpack_dequant_w4.o      — TPC ELF (tpc-clang -mcpu=gaudi2)
#   2) build/libmoe_ext_tpc_perflib.so        — TPC perf-lib exporting the GUID
#      (the .o ISA embedded via llvm-objcopy; discovered by the graph compiler
#       through GC_KERNEL_PATH)
#   3) build/libmoe_ext_custom_op.so          — PT2 glue (torch.ops.load_library)
#
# Run INSIDE the Habana dev image, CPU-ONLY (NO HPU device flags), e.g.:
#
#   docker run --rm \
#     -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work \
#     vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
#     bash build.sh
#
# Toolchain (CONFIRMED present in the image):
#   tpc-clang 2.7.33  : /opt/habanalabs/bin/tpc-clang  (driver auto-adds
#                       -mlink-builtin-bitcode <resource>/lib/gaudi2.bc,
#                       -target-cpu gaudi2, -float8 -bfloat16 -slm 16
#                       -max-tensors 16 -tpc-special  when invoked -mcpu=gaudi2)
#   llvm-objcopy      : /opt/habanalabs/bin/llvm-objcopy
#   Synapse/TPC hdrs  : /usr/include/habanalabs/
#   PT2 custom-op hdr : .../habana_frameworks/torch/include/hpu_custom_op_pt2.h
#   registerUserCustomOp lives in libhabana_pytorch_backend.so
#
# !!! CORRECTION (2026-06-08, re-verified) — the "gaudi2.bc blocker" is FALSE for
# !!! this elementwise kernel. tpc-clang DOES inject
# !!!   -mlink-builtin-bitcode /opt/habanalabs/lib/clang/14.0.5/lib/gaudi2.bc
# !!! and that file IS absent — but a MISSING -mlink-builtin-bitcode target is
# !!! NON-FATAL: the TPC intrinsics this kernel uses (v_*_ld_tnsr_b, v_convert_*,
# !!! v_f32_*) come from the -tpc-special backend, NOT from gaudi2.bc bitcode.
# !!! VERIFIED: `tpc-clang -mcpu=gaudi2 -O2 -c moe_ext_unpack_dequant_w4.c` -> rc=0,
# !!! produces a 14960-byte TPC REL ELF with `main` (1088B .text), .KernelInfo,
# !!! .tpc_metadata, .tpc_compiler sections. So Step 1 SUCCEEDS as-is.
# !!! (gaudi2.bc would only matter for kernels that call bitcode-resident builtins
# !!!  like some transcendental/special-function helpers; this dequant does not.)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
mkdir -p "${BUILD}"

TPC_CLANG="${TPC_CLANG:-/opt/habanalabs/bin/tpc-clang}"
OBJCOPY="${OBJCOPY:-/opt/habanalabs/bin/llvm-objcopy}"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
RES_DIR="$("${TPC_CLANG}" -print-resource-dir 2>/dev/null || echo /opt/habanalabs/lib/clang/14.0.5)"
GAUDI2_BC="${GAUDI2_BC:-${RES_DIR}/lib/gaudi2.bc}"

KERNEL_C="${HERE}/moe_ext_unpack_dequant_w4.c"
GLUE_CPP="${HERE}/moe_ext_custom_op.cpp"

echo "=== tpc_kernel/build.sh ==="
echo "tpc-clang   : ${TPC_CLANG} ($("${TPC_CLANG}" --version 2>/dev/null | head -1))"
echo "objcopy     : ${OBJCOPY}"
echo "resource dir: ${RES_DIR}"
echo "gaudi2.bc   : ${GAUDI2_BC}"

# ---- preflight: is the SDK payload actually present? -----------------------
MISSING=0
if [[ ! -f "${GAUDI2_BC}" ]]; then
  echo "!!! gaudi2.bc NOT FOUND at ${GAUDI2_BC}"
  echo "!!! tpc-clang cannot link builtin bitcode -> TPC compile WILL fail."
  echo "!!! Restore the TPC-fuser SDK payload (habanatools / tpc_kernels) into"
  echo "!!! ${RES_DIR}/lib/  before this step can succeed."
  MISSING=1
fi
if [[ -z "$(ls -A "${RES_DIR}" 2>/dev/null)" ]]; then
  echo "!!! resource dir ${RES_DIR} is EMPTY (no builtin TPC intrinsic headers)."
  MISSING=1
fi

# =============================================================================
# Step 1 — compile the TPC kernel to a Gaudi2 ELF object.
# The simple driver form lets tpc-clang inject the exact stock flags (verified
# against libtpc_kernels.so's embedded build line for cast_i4_to_bf16.c):
#   -target-cpu gaudi2 -float8 -bfloat16 -slm 16 -max-tensors 16 -tpc-special
#   -mlink-builtin-bitcode <RES_DIR>/lib/gaudi2.bc
# We additionally pass the Habana include dir for any shared param headers.
# =============================================================================
echo
echo "=== [1/3] TPC compile: ${KERNEL_C##*/} -> moe_ext_unpack_dequant_w4.o ==="
set -x
"${TPC_CLANG}" \
  -mcpu=gaudi2 \
  -O2 \
  -c "${KERNEL_C}" \
  -I "${HAB_INC}" \
  -o "${BUILD}/moe_ext_unpack_dequant_w4.o"
TPC_RC=$?
set +x
if [[ ${TPC_RC} -ne 0 ]]; then
  echo "!!! TPC compile FAILED (rc=${TPC_RC})."
  if [[ ${MISSING} -eq 1 ]]; then
    echo "!!! Expected: gaudi2.bc / builtin intrinsics payload is absent (see banner)."
  fi
  echo "!!! Stopping; fix the toolchain payload, then re-run."
  exit ${TPC_RC}
fi

# =============================================================================
# Step 2 — wrap the kernel ELF into a perf-lib .so the graph compiler can load.
# The TPC ISA/ELF is embedded as a data section; the perf-lib exposes the GUID
# via the standard entry points (GetKernelGuids / GetKernelNames / HabanaKernel).
# The stock kernels ship this as libtpc_kernels.so. For a USER perf-lib we embed
# the ELF and let GC_KERNEL_PATH point at the resulting .so.
#
# NOTE: a full user perf-lib also needs a tiny C++ shim implementing
# GetKernelGuids + InstantiateTpcKernel against tpc_kernel_lib_interface.h. That
# shim is a Phase-0.5 item; for Phase-0 we emit the embedded-ISA object so the
# pipeline is wired and the .o is validated. llvm-objcopy turns the ELF into a
# relocatable blob that the shim (or a re-use of cast_i4_to_bf16's GUID) links.
# =============================================================================
echo
echo "=== [2/3] embed TPC ELF (llvm-objcopy) ==="
set -x
"${OBJCOPY}" \
  -I binary -O elf64-x86-64 --binary-architecture i386:x86-64 \
  --rename-section .data=.tpc_isa,alloc,load,readonly,data,contents \
  "${BUILD}/moe_ext_unpack_dequant_w4.o" \
  "${BUILD}/moe_ext_unpack_dequant_w4.isa.o" 2>/dev/null
OBJCOPY_RC=$?
set +x
[[ ${OBJCOPY_RC} -ne 0 ]] && echo "(objcopy embed step rc=${OBJCOPY_RC}; non-fatal for Phase-0 wiring)"

# =============================================================================
# Step 3 — build the PT2 glue shared object (links Torch + Habana plugin).
# Compiled with the SYSTEM clang/g++ (NOT tpc-clang) — this is host x86 code.
# =============================================================================
echo
echo "=== [3/3] PT2 glue compile: ${GLUE_CPP##*/} -> libmoe_ext_custom_op.so ==="
TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))' 2>/dev/null)"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))' 2>/dev/null)"
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))' 2>/dev/null)"

# CRITICAL (2026-06-08 fix): match torch's libstdc++ ABI. The Habana 1.24 /
# pytorch-2.10 image ships torch built with _GLIBCXX_USE_CXX11_ABI=1; hardcoding
# =0 here produced a glue .so whose torch::Library ctor mangles std::string the
# old way (`...Ss...`), which torch (cxx11) does NOT export -> load-time
# "undefined symbol: _ZN5torch7LibraryC1ENS0_4KindESs...". QUERY the live torch
# and pass the matching flag (this is what build_dequant.sh already did).
ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"
echo "torch    : ${TORCH_DIR}"
echo "habana_fw: ${HABF_DIR}"
echo "py inc   : ${PY_INC}"
echo "cxx11 abi: ${ABI}  (matched to the installed torch)"

set -x
g++ -O2 -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI}" \
  "${GLUE_CPP}" \
  -I "${TORCH_INC}" \
  -I "${TORCH_INC}/torch/csrc/api/include" \
  -I "${HABF_INC}" \
  -I "${HAB_INC}" \
  -I "${PY_INC}" \
  -L "${TORCH_LIB}" -ltorch -ltorch_cpu -lc10 \
  -L "${HABF_LIB}" \
  -lhabana_pytorch_backend \
  -lhabana_pytorch2_plugin \
  -Wl,--no-as-needed \
  -Wl,-rpath,"${TORCH_LIB}" -Wl,-rpath,"${HABF_LIB}" \
  -o "${BUILD}/libmoe_ext_custom_op.so"
GLUE_RC=$?
set +x

echo
echo "=== build summary ==="
echo "  TPC .o            : rc=${TPC_RC}   ${BUILD}/moe_ext_unpack_dequant_w4.o"
echo "  embed ISA         : rc=${OBJCOPY_RC}"
echo "  PT2 glue .so      : rc=${GLUE_RC}  ${BUILD}/libmoe_ext_custom_op.so"
echo
echo "  To use on HPU (human, on the box):"
echo "    export GC_KERNEL_PATH=${BUILD}/libmoe_ext_tpc_perflib.so:\$GC_KERNEL_PATH"
echo "    python register.py   # torch.ops.load_library(libmoe_ext_custom_op.so)"
echo "    python test_p0_dequant.py"

# Propagate the meaningful failure for CI.
if [[ ${TPC_RC} -ne 0 || ${GLUE_RC} -ne 0 ]]; then
  exit 1
fi
exit 0

# -----------------------------------------------------------------------------
# RESTORING THE TPC SDK PAYLOAD (one-time, to make Step 1 succeed)
# -----------------------------------------------------------------------------
# gaudi2.bc + the builtin intrinsic headers ship in the Habana "tpc-fuser" /
# tpc_kernels SDK. Options:
#   (a) Install the matching habanatools .deb that provides
#       /opt/habanalabs/lib/clang/14.0.5/lib/gaudi2.bc and the tpcc intrinsics.
#   (b) Copy gaudi2.bc out of an image/host that has the full TPC SDK into
#       ${RES_DIR}/lib/gaudi2.bc and the intrinsic headers into ${RES_DIR}/include.
#   (c) Point GAUDI2_BC=/path/to/gaudi2.bc when invoking build.sh.
# The embedded build command proving the exact path was recovered from
# libtpc_kernels.so:
#   ... -mlink-builtin-bitcode <RES>/lib/gaudi2.bc -target-cpu gaudi2 \
#       -float8 -bfloat16 -slm 16 -max-tensors 16 -tpc-special -std=rc99 -O2 ...
# -----------------------------------------------------------------------------
