"""test_p0_runnable.py — P0 end-to-end gate: build + register + RUN our OWN GUID.

RUN ON THE HPU BOX (this issues device ops; a CPU container cannot execute them).

What this proves, and how it differs from ../test_p0_dequant_general.py:
  * ../test_p0_dequant_general.py exercises the RESIDENT GUID via the public
    torch.ops.hpu.convert_from_uint4 (no user build at all).
  * THIS test exercises the USER-AUTHORED kernel: the Gaudi-2 ELF compiled from
    ../moe_ext_unpack_dequant_w4.c, packaged into fusion/build/libmoe_ext_perflib.so
    by build_perflib.sh, discovered by the graph compiler through GC_KERNEL_PATH,
    and bound to a torch op by ../build/libmoe_ext_custom_op.so (TORCH_LIBRARY
    hpu_moe_ext::unpack_dequant_w4 -> GUID moe_ext_unpack_dequant_w4_bf16).
  It asserts BIT-EXACT (maxdiff==0, identical int16 bits) vs convert_from_uint4,
  parameterized over group_size + GLM/DeepSeek/Nemotron/MiniMax-ish shapes.

  This is the make-sure-the-pipeline-works gate. If GC cannot find/instantiate our
  GUID it fails LOUDLY here (GLUE_* error or "guid not found"); that error names the
  thing to fix in moe_ext_perflib.cpp::InstantiateTpcKernel (geometry / access
  pattern / ELF handshake).

PRECONDITION (build first, CPU-only, inside the dev image):
    docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
      vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
      bash build_perflib.sh
    # and the PT2 glue (one dir up):
    docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work \
      vault.habana.ai/gaudi-docker/1.24.0/ubuntu24.04/habanalabs/pytorch-installer-2.10.0:latest \
      bash build.sh

RUN (on the box, with an HPU):
    cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
    export GC_KERNEL_PATH=$PWD/build/libmoe_ext_perflib.so:$GC_KERNEL_PATH
    python test_p0_runnable.py

Exit 0 == PASS (our compiled GUID is bit-exact).
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # tpc_kernel/
_PERFLIB = os.path.join(_HERE, "build", "libmoe_ext_perflib.so")
_GLUE = os.path.join(_ROOT, "build", "libmoe_ext_custom_op.so")

ZERO_CONST = 8
GROUP_SIZE_DEFAULT = 128


# ---- canonical uint4 packer (vllm if present, else bit-identical local) -------
try:
    from vllm.model_executor.layers.quantization.utils.quant_utils import (  # type: ignore
        pack_quantized_values_into_int32 as _vllm_pack,
    )
    from vllm.scalar_type import scalar_types  # type: ignore

    _QT = scalar_types.uint4

    def pack_u4(codes_2d: torch.Tensor) -> torch.Tensor:
        return _vllm_pack(codes_2d, _QT, packed_dim=1)

    PACKER = "vllm.pack_quantized_values_into_int32"
except Exception:  # noqa: BLE001

    def pack_u4(codes_2d: torch.Tensor) -> torch.Tensor:
        assert codes_2d.shape[-1] % 8 == 0
        c = codes_2d.to(torch.int64) & 0xF
        rows, cols = c.shape
        c = c.reshape(rows, cols // 8, 8)
        shifts = 4 * torch.arange(8, device=c.device, dtype=torch.int64)
        word = (c << shifts).sum(dim=-1)
        word = torch.where(word >= 2**31, word - 2**32, word)
        return word.to(torch.int32)

    PACKER = "local int32 packer (vllm not importable)"


def _sym_zero_packed(groups: int, out_dim: int, device) -> torch.Tensor:
    assert out_dim % 8 == 0
    word = 0
    for nib in range(8):
        word |= (ZERO_CONST & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _setup_gc_kernel_path() -> None:
    if not os.path.exists(_PERFLIB):
        raise FileNotFoundError(
            f"[P0] perf-lib not built: {_PERFLIB}\n"
            f"  Build it (CPU-only): bash {os.path.join(_HERE, 'build_perflib.sh')}"
        )
    cur = os.environ.get("GC_KERNEL_PATH", "")
    parts = cur.split(":") if cur else []
    if _PERFLIB not in parts:
        os.environ["GC_KERNEL_PATH"] = ":".join([_PERFLIB] + parts)
        print(f"[P0] GC_KERNEL_PATH := {os.environ['GC_KERNEL_PATH']}")


def make_case(in_dim: int, out_dim: int, gs: int, seed: int, dev: str):
    g = torch.Generator().manual_seed(seed)
    assert in_dim % gs == 0 and out_dim % 8 == 0
    groups = in_dim // gs
    codes = torch.randint(0, 16, (in_dim, out_dim), generator=g, dtype=torch.int32)
    scales = (torch.rand((groups, out_dim), generator=g) * 0.02 + 1e-3).to(torch.float16)
    wp = pack_u4(codes).to(dev)
    sc = scales.to(dev)
    zp = _sym_zero_packed(groups, out_dim, dev)
    return wp, sc, zp


def run_case(tag, in_dim, out_dim, gs, seed) -> bool:
    dev = "hpu"
    wp, sc, zp = make_case(in_dim, out_dim, gs, seed, dev)
    ref = torch.ops.hpu.convert_from_uint4(wp, sc, zp, torch.bfloat16, None)
    got = torch.ops.hpu_moe_ext.unpack_dequant_w4(wp, sc, zp, int(gs), int(ZERO_CONST))
    ref_c, got_c = ref.to("cpu"), got.to("cpu")
    if got_c.shape != ref_c.shape or got_c.dtype != torch.bfloat16:
        print(f"[{tag}] BAD shape/dtype got={tuple(got_c.shape)}/{got_c.dtype} "
              f"ref={tuple(ref_c.shape)}")
        return False
    bits_equal = torch.equal(got_c.view(torch.int16), ref_c.view(torch.int16))
    maxdiff = float((got_c.float() - ref_c.float()).abs().max())
    print(f"[{tag}] in={in_dim} out={out_dim} gs={gs} groups={in_dim // gs} "
          f"bit_exact={bits_equal} maxdiff={maxdiff:.3e} |ref|={ref_c.float().norm():.1f}")
    return bits_equal and maxdiff == 0.0


def main() -> int:
    print(f"packer: {PACKER}")
    try:
        import habana_frameworks.torch  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    _setup_gc_kernel_path()
    if not os.path.exists(_GLUE):
        print(f"FATAL: PT2 glue not built: {_GLUE}\n  Build it: bash {os.path.join(_ROOT, 'build.sh')}")
        return 2
    torch.ops.load_library(_GLUE)
    print(f"[P0] loaded glue: {_GLUE}")

    cases = [
        ("smoke",        256,   64,  128, 0),
        ("glm_w13",      6144, 4096, 128, 1),
        ("glm_w2",       2048, 6144, 128, 2),
        ("ds_w13",       7168, 4096, 128, 3),
        ("nemo_w13",     8192, 6144, 128, 5),
        ("minimax_gs64", 6144, 4096,  64, 6),
        ("gs32",         4096, 2048,  32, 7),
        ("odd_out",      512,  128,  128, 8),
    ]
    ok = True
    print("=" * 78)
    print("P0 user-GUID bit-exact vs convert_from_uint4:")
    for tag, i, o, gs, s in cases:
        try:
            ok &= run_case(tag, i, o, gs, s)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag}] EXCEPTION {type(ex).__name__}: {ex}")
            ok = False
    print("=" * 78)
    print("P0 RUNNABLE CUSTOM GUID:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
