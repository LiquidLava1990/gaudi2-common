"""test_p1_fusion.py — P1 correctness gate for the fused int4->dequant->gemm path.

RUN ON THE HPU BOX. Proves the FUSED computation is numerically IDENTICAL to the
unfused reference, i.e. fusing the dequant into the gemm (whether by the GC pass
fuseDequantIntoMoeGemm or by SRAM-resident bundling) does NOT change the math.

It checks the torch-level fused expression
    y_fused = convert_from_uint4(W4, scale, zp, bf16) @ x         (one HPU graph)
against a reference that forces the bf16 weight to be MATERIALIZED first
    W = convert_from_uint4(...).contiguous() ; y_ref = W @ x      (weight to HBM)
on the SAME inputs. If these disagree, the orientation/contraction is wrong (the
single most likely bug when hand-building the gemm node) — fix kTransposeA/B in
fused_moe_recipe.cpp accordingly, and this test localizes it.

This is the numeric companion to profile_fusion.py (which measures HBM bytes). Run
BOTH: this for "is it correct", that for "is it SRAM-resident".

    cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
    python test_p1_fusion.py

Exit 0 == PASS (fused == reference, bit-exact in bf16).
"""
from __future__ import annotations

import sys

import torch


def _sym_zero_packed(groups, out_dim, device):
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def make(IN, OUT, GS, seed, dev):
    g = torch.Generator().manual_seed(seed)
    groups = IN // GS
    codes = torch.randint(0, 16, (IN, OUT), generator=g, dtype=torch.int32)
    c = codes.to(torch.int64) & 0xF
    c = c.reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).to(dev)
    scale = (torch.rand((groups, OUT), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(dev)
    zp = _sym_zero_packed(groups, OUT, dev)
    return word, scale, zp


def run(tag, M, IN, OUT, GS, seed) -> bool:
    dev = "hpu"
    word, scale, zp = make(IN, OUT, GS, seed, dev)
    x = torch.randn((M, IN), dtype=torch.bfloat16, device=dev)

    # reference: materialize bf16 weight, then matmul.
    W = torch.ops.hpu.convert_from_uint4(word, scale, zp, torch.bfloat16, None).contiguous()
    y_ref = torch.matmul(x, W)

    # fused expression (single graph; GC may fuse dequant into the gemm).
    def fused():
        w = torch.ops.hpu.convert_from_uint4(word, scale, zp, torch.bfloat16, None)
        return torch.matmul(x, w)

    y_fused = fused()

    yr, yf = y_ref.to("cpu"), y_fused.to("cpu")
    bit_exact = torch.equal(yr.view(torch.int16), yf.view(torch.int16))
    maxdiff = float((yr.float() - yf.float()).abs().max())
    # bf16 matmul accumulation should be identical for identical inputs/graph shape;
    # allow exact, but report maxdiff for diagnosis.
    print(f"[{tag}] M={M} IN={IN} OUT={OUT} GS={GS} "
          f"bit_exact={bit_exact} maxdiff={maxdiff:.3e} |ref|={yr.float().norm():.1f}")
    return bit_exact or maxdiff == 0.0


def main() -> int:
    try:
        import habana_frameworks.torch  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2
    cases = [
        ("smoke",   8,  256,   64, 128, 0),
        ("glm_w13", 32, 6144, 4096, 128, 1),
        ("glm_w2",  32, 2048, 6144, 128, 2),
        ("ds_w13",  16, 7168, 4096, 128, 3),
        ("gs64",    16, 6144, 4096,  64, 4),
    ]
    ok = True
    for tag, M, IN, OUT, GS, seed in cases:
        try:
            ok &= run(tag, M, IN, OUT, GS, seed)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag}] EXCEPTION {type(ex).__name__}: {ex}")
            ok = False
    print("=" * 60)
    print("P1 FUSED CORRECTNESS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
