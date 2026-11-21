#!/usr/bin/env bash
# =============================================================================
# oncard_dequant_ab.sh — end-to-end A/B for the OPTIMIZED int4->bf16 dequant
# kernels (Lever #2 of the single-stream campaign): build the fast perf-lib + PT2
# glue CARD-FREE, then on-card prove byte-exact numerics vs convert_from_uint4 and
# time both unpack variants. LOUD pass/fail gates + a printed verdict.
#
# !!! THIS SCRIPT TOUCHES A CARD (the python step runs HPU ops). It is the
# !!! ORCHESTRATOR's to run on the box when cards are FREE — NOT a build-agent
# !!! script. The BUILD step (--build-only) is card-free and safe anywhere.
#
# Two phases:
#   [BUILD]  (card-free) docker run the 1.24 pytorch-installer image, NO accel
#            device, bash build_fast.sh -> build_fast/{libmoe_ext_perflib_fast.so,
#            libmoe_ext_custom_op_fast.so}. Idempotent: skipped if both .so exist
#            and are newer than their sources (force with --rebuild).
#   [ONCARD] export GC_KERNEL_PATH + PT_HPU_LAZY_MODE=1 and run
#            tests/dequant_fast_oncard_validate.py (which RAISES rather than
#            silently A/B the reference against itself).
#
# Usage:
#   bash oncard_dequant_ab.sh                 # build (if needed) + on-card A/B
#   bash oncard_dequant_ab.sh --build-only    # CARD-FREE: just build the .so
#   bash oncard_dequant_ab.sh --rebuild       # force the card-free rebuild first
#   bash oncard_dequant_ab.sh --experts 32 --layers 75   # bench knobs passthrough
#   bash oncard_dequant_ab.sh --no-bench      # correctness gates only (fast)
#
# Env overrides:
#   IMAGE=...            the CPU dev image (default 1.24 pytorch-installer)
#   PY=...               python on the box for the on-card step (default: python3)
#   EXTRA_GC_KERNEL_PATH extra ':'-joined paths to prepend to GC_KERNEL_PATH
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../hpu-moe-ext/tpc_kernel
ROOT="$(cd "${HERE}/.." && pwd)"                          # .../hpu-moe-ext
BUILD="${HERE}/build_fast"
PERFLIB="${BUILD}/libmoe_ext_perflib_fast.so"
GLUE="${BUILD}/libmoe_ext_custom_op_fast.so"
VALIDATE="${ROOT}/tests/dequant_fast_oncard_validate.py"

# Build in the SAME image the on-card A/B runs in (vllm-gaudi:next), so the glue's
# torch ABI + backend-lib name (the .upstream rename) match the run image and the
# plugin-free link resolves the SAME libhabana_pytorch_backend that's live at runtime.
# (Override IMAGE= to use the 1.24 pytorch-installer if you build/run there instead.)
IMAGE="${IMAGE:-vllm-gaudi:next}"
PY="${PY:-python3}"

BUILD_ONLY=0
REBUILD=0
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) BUILD_ONLY=1; shift ;;
    --rebuild)    REBUILD=1; shift ;;
    *)            PASS_ARGS+=("$1"); shift ;;   # forwarded to the python validator
  esac
done

say()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf '\n!!! %s\n' "$*" >&2; exit 1; }

# --- sanity: sources present ---
[[ -f "${HERE}/build_fast.sh" ]]            || fail "missing build_fast.sh"
[[ -f "${HERE}/moe_ext_unpack_dequant_w4_fast.c" ]] || fail "missing kernel .c"
[[ -f "${HERE}/moe_ext_perflib_fast.cpp" ]] || fail "missing perf-lib .cpp"
[[ -f "${HERE}/moe_ext_custom_op_fast.cpp" ]] || fail "missing glue .cpp"
[[ -f "${VALIDATE}" ]]                       || fail "missing ${VALIDATE}"

# --- decide whether a build is needed (idempotent) ---
need_build=1
if [[ ${REBUILD} -eq 0 && -f "${PERFLIB}" && -f "${GLUE}" ]]; then
  newest_src=$(ls -t "${HERE}/build_fast.sh" "${HERE}"/moe_ext_unpack_dequant_w4_fast.c \
                      "${HERE}"/moe_ext_perflib_fast.cpp "${HERE}"/moe_ext_custom_op_fast.cpp \
                 2>/dev/null | head -1)
  if [[ "${PERFLIB}" -nt "${newest_src}" && "${GLUE}" -nt "${newest_src}" ]]; then
    need_build=0
  fi
fi

if [[ ${need_build} -eq 1 ]]; then
  say "BUILD (card-free) — docker ${IMAGE##*/} bash build_fast.sh"
  if ! command -v docker >/dev/null 2>&1; then
    fail "docker not found; cannot run the card-free build. Build manually:
    docker run --rm -v ${HERE}:/work -w /work ${IMAGE} bash build_fast.sh"
  fi
  # CARD-FREE by construction: NO --device=/dev/accel, NO HABANA_VISIBLE_DEVICES.
  docker run --rm \
    -v "${HERE}:/work" -w /work \
    "${IMAGE}" \
    bash build_fast.sh
  brc=$?
  [[ ${brc} -ne 0 ]] && fail "build_fast.sh failed rc=${brc}"
else
  say "BUILD skipped (.so newer than sources; --rebuild to force)"
fi

# --- verify build outputs ---
[[ -f "${PERFLIB}" ]] || fail "perf-lib not produced: ${PERFLIB}"
if [[ ! -f "${GLUE}" ]]; then
  fail "PT2 glue not produced: ${GLUE}
  (build_fast.sh Step 4 needs torch importable -> ensure the dev image has it.
   Without the glue the on-card A/B has no real op and would silent-pass — refused.)"
fi
say "build artifacts present"
printf '  perf-lib : %s\n  glue     : %s\n' "${PERFLIB}" "${GLUE}"

if [[ ${BUILD_ONLY} -eq 1 ]]; then
  say "BUILD-ONLY done (card-free). On-card A/B NOT run."
  exit 0
fi

# =============================================================================
# ON-CARD PHASE — requires FREE cards. Sets the GC path so the new GUIDs resolve.
# =============================================================================
export GC_KERNEL_PATH="${PERFLIB}${EXTRA_GC_KERNEL_PATH:+:${EXTRA_GC_KERNEL_PATH}}${GC_KERNEL_PATH:+:${GC_KERNEL_PATH}}"
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"

say "ON-CARD A/B — ${PY} dequant_fast_oncard_validate.py ${PASS_ARGS[*]:-}"
printf '  GC_KERNEL_PATH=%s\n  PT_HPU_LAZY_MODE=%s\n' "${GC_KERNEL_PATH}" "${PT_HPU_LAZY_MODE}"

"${PY}" "${VALIDATE}" "${PASS_ARGS[@]:-}"
rc=$?

say "VERDICT"
if [[ ${rc} -eq 0 ]]; then
  printf '  PASS — new dequant kernel byte-exact vs convert_from_uint4 on-card; timing above.\n'
else
  printf '  FAIL (rc=%s) — see the !!! lines above. LEAN byte-exact + adversarial are the\n' "${rc}"
  printf '  hard gates; FAST=SKIPPED is acceptable (LEAN ships), FAST=FAIL means the\n'
  printf '  stride-8 deinterleave access pattern in moe_ext_perflib_fast.cpp needs a fix.\n'
fi
exit ${rc}
