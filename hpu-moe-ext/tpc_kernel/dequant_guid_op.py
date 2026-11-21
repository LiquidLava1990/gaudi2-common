"""dequant_guid_op.py — load + smoke-test the Path-1c direct-GUID binding.

SECONDARY path (the recommended runnable foundation is dequant_intrinsic.py). This
loads build/libdequant_guid_op.so, which registers torch op
hpu_moe_ext_guid::dequant_w4_guid bound DIRECTLY to the pre-built TPC GUID
``grouped_index_transpose_dequant_2d_with_zp_fwd_u4`` (resident in
/usr/lib/habanalabs/libtpc_kernels.so — no gaudi2.bc needed).

If the GUID requires a scalar params struct we did not supply (size 0 in the glue),
instantiation fails LOUDLY here at the first device call — that is the signal to
recover + fill the params struct (see dequant_guid_op.cpp risk note). When it DOES
run, it is cross-checked bit-exact vs torch.ops.hpu.convert_from_uint4.

Build (CPU-only):  bash build_dequant.sh   (inside the dev image)
Run (on the box, HPU):  python dequant_guid_op.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "build", "libdequant_guid_op.so")

_LOADED = False


def load() -> None:
    global _LOADED
    if _LOADED:
        return
    import habana_frameworks.torch  # noqa: F401  (registers HPU backend + custom-op reg)
    if not os.path.exists(_SO):
        raise FileNotFoundError(
            f"[dequant_guid] not built: {_SO}\n  Build it: bash {os.path.join(_HERE, 'build_dequant.sh')}"
        )
    torch.ops.load_library(_SO)
    _LOADED = True


def dequant_w4_guid(weight_packed, weight_scale, zero_point):
    """int32 [in,out//8] + 2D [groups,out] bf16 scale + int32 [groups,out//8] zp -> bf16 [in,out]."""
    if not _LOADED:
        load()
    return torch.ops.hpu_moe_ext_guid.dequant_w4_guid(weight_packed, weight_scale, zero_point)


def _smoke() -> int:
    import dequant_intrinsic as dq

    load()
    dev = "hpu"
    in_dim, out_dim, gs = 256, 64, 128
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (in_dim, out_dim), generator=g, dtype=torch.int32)
    groups = in_dim // gs
    scales = (torch.rand((groups, out_dim), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(dev)

    # pack u4 [in,out] -> int32 [in,out//8]
    c = codes.to(torch.int64) & 0xF
    c = c.reshape(in_dim, out_dim // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).to(dev)
    zp = dq._sym_zero_packed(groups, out_dim, dq.ZERO_CONST, dev)

    try:
        got = dequant_w4_guid(word, scales, zp).to("cpu")
    except Exception as ex:  # noqa: BLE001
        print(f"[guid-smoke] GUID instantiation FAILED: {type(ex).__name__}: {ex}")
        print("  -> the GUID likely needs a scalar params struct; fill it in "
              "dequant_guid_op.cpp (see risk note). Path 1a (convert_from_uint4) is unaffected.")
        return 1

    ref = torch.ops.hpu.convert_from_uint4(word, scales, zp, torch.bfloat16, None).to("cpu")
    bit_exact = torch.equal(got.view(torch.int16), ref.view(torch.int16))
    maxdiff = float((got.float() - ref.float()).abs().max())
    print(f"[guid-smoke] shape={tuple(got.shape)} bit_exact_vs_convert_from_uint4={bit_exact} "
          f"maxdiff={maxdiff:.3e}")
    return 0 if bit_exact else 1


if __name__ == "__main__":
    sys.exit(_smoke())
