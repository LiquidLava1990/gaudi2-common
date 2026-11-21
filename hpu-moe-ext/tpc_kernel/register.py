"""register.py — load + bind the Phase-0 fused-MoE foundation kernel.

Loads the compiled PT2 glue shared object (which TORCH_LIBRARY-registers the
schema `hpu_moe_ext::unpack_dequant_w4` and wires it to the TPC GUID
`moe_ext_unpack_dequant_w4_bf16`) and exposes a thin Python wrapper.

Usage on the HPU box (NOT in the headless CPU container):
    export GC_KERNEL_PATH=$PWD/build/libmoe_ext_tpc_perflib.so:$GC_KERNEL_PATH
    python -c "import register; register.load(); print(register.unpack_dequant_w4)"

The op signature mirrors the oracle:
    torch.ops.hpu.convert_from_uint4(weight_packed, weight_scale, zero_point,
                                     torch.bfloat16, None)
i.e.
    weight_packed : int32  [in, out//8]  (uint4-packed along out, 8/int32)
    weight_scale  : f16    [groups, out]
    zero_point    : int32  [groups, out//8]  (GPTQ-sym => packed constant 8)
    group_size    : int    (128 for GLM)
    zero_const    : int    (8 for GPTQ-sym; pass -1 to use the zero_point tensor)
  -> bf16 [in, out]
"""
from __future__ import annotations

import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_GLUE_SO = os.path.join(_HERE, "build", "libmoe_ext_custom_op.so")
_PERFLIB_SO = os.path.join(_HERE, "build", "libmoe_ext_tpc_perflib.so")

_LOADED = False


def _ensure_gc_kernel_path() -> None:
    """Prepend our TPC perf-lib to GC_KERNEL_PATH so the graph compiler resolves
    the custom GUID. Idempotent. Must be set BEFORE the first HPU op of the run
    in practice (here we set it eagerly; on the box export it in the shell)."""
    if not os.path.exists(_PERFLIB_SO):
        # Phase-0 may run by reusing the stock cast_i4_to_bf16 GUID instead of a
        # standalone perf-lib; in that case there is no perflib .so to add.
        return
    cur = os.environ.get("GC_KERNEL_PATH", "")
    parts = cur.split(":") if cur else []
    if _PERFLIB_SO not in parts:
        os.environ["GC_KERNEL_PATH"] = ":".join([_PERFLIB_SO] + parts)


def load() -> None:
    """Load the PT2 glue .so and register the op. Raises if the .so is missing
    (LOUD failure — never silently no-op, per the build-quality bar)."""
    global _LOADED
    if _LOADED:
        return
    # habana_frameworks.torch must be imported so the HPU backend + custom-op
    # registry (libhabana_pytorch_backend.so / registerUserCustomOp) are live.
    import habana_frameworks.torch  # noqa: F401

    _ensure_gc_kernel_path()

    if not os.path.exists(_GLUE_SO):
        raise FileNotFoundError(
            f"[hpu_moe_ext] glue not built: {_GLUE_SO}\n"
            f"  Build it first:  bash {os.path.join(_HERE, 'build.sh')}\n"
            f"  (inside the Habana dev image, CPU-only)."
        )
    torch.ops.load_library(_GLUE_SO)
    _LOADED = True


def unpack_dequant_w4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int = 128,
    zero_const: int = 8,
) -> torch.Tensor:
    """Thin wrapper over torch.ops.hpu_moe_ext.unpack_dequant_w4.

    All tensors must be on 'hpu'. Returns bf16 [in, out]."""
    if not _LOADED:
        load()
    return torch.ops.hpu_moe_ext.unpack_dequant_w4(
        weight_packed, weight_scale, zero_point, int(group_size), int(zero_const)
    )


__all__ = ["load", "unpack_dequant_w4"]
