"""verify_convert_from_uint4_args.py — de-risk the dequant CONTRACT on-HPU.

RUN ON THE HPU BOX (CPU-only import won't exercise the SIF/ComplexGuid lowering):
    cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
    python verify_convert_from_uint4_args.py                 # GLM w13 default
    python verify_convert_from_uint4_args.py 32 6144 4096 128

WHAT THIS PROVES
----------------
The standalone Synapse recipe (fused_moe_recipe.cpp) builds a
  reinterpret_cast(int32[IN,OUT/8] -> uint4[IN,OUT])  ->  dequantize_4_bit_bf16
subgraph. The exact tensor contract it must honor was recovered from the PyTorch
backend (habana::ConvertFromInt4::AddNode in convert_from_int4_gen.cpp) and the
production callers (hpu_moe_ext/moe_wna16_hpu.py:578, reference MoeWNA16Matmul).
Schema:
  hpu::convert_from_uint4(Tensor input, Tensor scale, Tensor? zero_point,
                          ScalarType out_dtype, Tensor? group_index=None, *,
                          bool disable_fp8_clipping=False) -> Tensor
  input      : int32  [IN, OUT/8]        (8 u4 nibbles per int32 along OUT)
  scale      : bf16   [GROUPS, OUT]      (GROUPS = IN/GS)
  zero_point : int32  [GROUPS, OUT/8]    PACKED symmetric zero (uint4 midpoint 8)
  out_dtype  : bf16
  group_index: None
  -> W : bf16 [IN, OUT]

WHY THE EARLIER PROBE FAILED (and what this confirms)
-----------------------------------------------------
The probe  convert_from_uint4(inp[256,64]i32, scale[2,512]bf16, None, bf16, None)
errored GLUE_SIF_NULL_PTR in shape-inference. Root cause: for the GROUPED dequant
(GROUPS>1) the complex-guid SIF dereferences the zero_point tensor; passing None
hands it a null tensor handle -> SIF null-ptr crash. The ATen schema marks
zero_point optional, but the grouped lowering is NOT. Production NEVER passes None
(always the packed _sym_zero). This script runs:
   (1) the CORRECT call (packed zp)            -> expect SUCCESS, finite output
   (2) the SAME shapes but zero_point=None      -> expect the SIF/null-ptr error
so a single on-box run confirms BOTH the working arg set and the precise failure
mode, before committing to the full recipe + profile run.

Exit 0 == the correct-arg dequant ran and produced a finite bf16 [IN,OUT] tensor.
"""
from __future__ import annotations

import sys
import traceback


def _sym_zero_packed(groups, out_dim, device):
    """GPTQ-symmetric shared zero = uint4 midpoint 8, packed 8/int32 along OUT.
    Identical to _sym_zero in hpu_moe_ext/moe_wna16_hpu.py and the helper in
    test_p1_fusion.py / profile_fusion.py — shape [GROUPS, OUT/8] int32."""
    import torch
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)        # 0x88888888
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def _make_inputs(IN, OUT, GS, seed, device):
    import torch
    g = torch.Generator().manual_seed(seed)
    groups = IN // GS
    # random 4-bit codes -> pack 8 per int32 along OUT -> [IN, OUT/8] int32
    codes = torch.randint(0, 16, (IN, OUT), generator=g, dtype=torch.int32)
    c = (codes.to(torch.int64) & 0xF).reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).to(device)  # [IN, OUT/8]
    scale = (torch.rand((groups, OUT), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(device)  # [GROUPS, OUT]
    zp = _sym_zero_packed(groups, OUT, device)                                          # [GROUPS, OUT/8]
    return word, scale, zp, groups


def main() -> int:
    M  = int(sys.argv[1]) if len(sys.argv) > 1 else 32     # unused by the dequant; kept for arg parity
    IN = int(sys.argv[2]) if len(sys.argv) > 2 else 6144
    OUT= int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    GS = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    if IN % GS != 0 or OUT % 8 != 0:
        print(f"FATAL: need IN%GS==0 and OUT%8==0 (IN={IN} GS={GS} OUT={OUT})")
        return 2

    try:
        import habana_frameworks.torch  # noqa: F401
        import torch
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    dev = "hpu"
    word, scale, zp, groups = _make_inputs(IN, OUT, GS, seed=1, device=dev)
    print(f"shapes: IN={IN} OUT={OUT} GS={GS} GROUPS={groups}")
    print(f"  input(packed) : {tuple(word.shape)} {word.dtype}   (expect [IN, OUT/8] int32)")
    print(f"  scale         : {tuple(scale.shape)} {scale.dtype}   (expect [GROUPS, OUT] bf16)")
    print(f"  zero_point    : {tuple(zp.shape)} {zp.dtype}   (expect [GROUPS, OUT/8] int32, NOT None)")
    print(f"  out_dtype     : torch.bfloat16   group_index: None")

    # ---- (1) CORRECT call: packed zero_point (the recipe + production contract) ----
    print("\n[1] CORRECT args  convert_from_uint4(input, scale, PACKED_ZP, bf16, None):")
    try:
        W = torch.ops.hpu.convert_from_uint4(word, scale, zp, torch.bfloat16, None)
        import habana_frameworks.torch as ht
        ht.hpu.synchronize()
        Wc = W.to("cpu").float()
        finite = bool(torch.isfinite(Wc).all())
        print(f"    OK -> W shape={tuple(W.shape)} dtype={W.dtype}  "
              f"(expect [IN, OUT]=({IN},{OUT}))")
        print(f"    finite={finite}  |W|={Wc.norm():.3f}  min={Wc.min():.4f} max={Wc.max():.4f}")
        ok_shape = tuple(W.shape) == (IN, OUT) and W.dtype == torch.bfloat16
        if not ok_shape:
            print(f"    !! WARNING: shape/dtype unexpected (got {tuple(W.shape)} {W.dtype}).")
        ok = ok_shape and finite
    except Exception as ex:  # noqa: BLE001
        print(f"    FAILED with packed zp: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        print("    >>> The packed-zp contract did NOT run. Inspect the args above against the")
        print("        backend validation (ConvertFromInt4::AddNode). Do NOT proceed to the recipe.")
        return 1

    # ---- (2) the FAILING probe: zero_point=None (expected to error in SIF) ----
    print("\n[2] CONTROL  convert_from_uint4(input, scale, None, bf16, None)  (expect FAILURE):")
    try:
        Wn = torch.ops.hpu.convert_from_uint4(word, scale, None, torch.bfloat16, None)
        import habana_frameworks.torch as ht
        ht.hpu.synchronize()
        _ = Wn.to("cpu")
        print(f"    UNEXPECTED: None zero_point SUCCEEDED -> W shape={tuple(Wn.shape)}.")
        print("    (Then this build tolerates None for grouped dequant — the recipe still")
        print("     supplies a packed zp, which is also valid. Not a blocker.)")
    except Exception as ex:  # noqa: BLE001
        msg = f"{type(ex).__name__}: {ex}"
        print(f"    EXPECTED FAILURE: {msg}")
        if "SIF" in msg or "NULL" in msg.upper() or "null" in msg or "GLUE" in msg:
            print("    >>> Confirms the GLUE_SIF_NULL_PTR root cause: grouped dequant rejects")
            print("        zero_point=None. The recipe's packed-zp input is the correct fix.")
        else:
            print("    >>> Failed for a different reason than the recorded GLUE_SIF_NULL_PTR;")
            print("        the packed-zp path in [1] is still the contract the recipe uses.")

    print("\n" + "=" * 70)
    print("CONTRACT VERIFY:", "PASS (packed-zp dequant ran, finite [IN,OUT] bf16)" if ok
          else "FAIL (see [1] above)")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
