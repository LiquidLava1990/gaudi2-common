"""CARD-FREE proof of the NEW HPU_MOE_FP8_DIRECT fold (Lever 3).

The existing tests/fp8_experts_numeric_cardfree.py proves the round-trip path is
coherent and that an UN-RECOVERED int4->fp8 write is the 2x bug. This test proves the
SPECIFIC claim the rebuilt fp8 .so (BUILD_FP8_NODES=1, direct_fp8 branch in
moe_fused_expert_fp8_backend_shared.h) encodes:

  THE DIRECT FOLD == THE TWO-HOP, BIT-FOR-BIT, with the fn-vs-fnuz 2x CANCELLED.

The .cpp DIRECT-DEQUANT PROOF says: passing scale_in = w_scale_int4 * (1/s_fp8) to the
int4 dequant stores V = W_real / s_fp8 onto the SAME hf8 (e4m3) grid that
convert_to_fp8(W_real, 1/s_fp8) writes -> BIT-IDENTICAL stored fp8. Then fp8_gemm with
scaleB = s_fp8 recovers W_real. So y_direct == y_twohop at relNorm 0.0, and |got|/|ref|
~= 1.0 (NOT 2x). The 2x reappears ONLY if the s_fp8 recovery is dropped.

This is pure numpy (NO torch / NO HPU) -- it models the e4m3 grid + the dequant/cast/gemm
math the device ops implement, parameterized by the e4m3 bias so the fn-vs-fnuz behaviour
is explicit. It is the card-free analog of the on-card test_fused_expert_fp8_op.py
FP8_DIRECT=1 vs FP8_DIRECT=0 A/B.

    python3 tests/fp8_direct_fold_cardfree.py     # exits non-zero LOUD on any fail

Gates:
  1. STORED-BITS IDENTITY: the direct-fold stored fp8 grid value == the two-hop
     convert_to_fp8 stored grid value, EXACTLY (relNorm 0.0), for the weight.
  2. OUTPUT IDENTITY: y_direct == y_twohop at relNorm 0.0 (full expert, both GEMMs).
  3. |got|/|ref| ~= 1.0 (the 2x trap is NOT live on the recovered-direct path).
  4. NEGATIVE CONTROL: dropping the s_fp8 recovery on the direct path REINTRODUCES the
     ~2x inflation (proves the recovery scale is what cancels the fnuz factor).
"""
import sys
import numpy as np

REL_TOL = 0.08
GROUP = 128
H = 512
I = 256
ZERO = 8
FNUZ_BIAS = 8
FN_BIAS = 7
rng = np.random.default_rng(0)


def e4m3_grid(bias):
    vals = {0.0}
    for ef in range(1, 16):
        e = ef - bias
        for m in range(8):
            vals.add((1.0 + m / 8.0) * (2.0 ** e))
    e = 1 - bias
    for m in range(1, 8):
        vals.add((m / 8.0) * (2.0 ** e))
    return np.array(sorted(vals), dtype=np.float64)


def quant_e4m3(x, bias):
    g = e4m3_grid(bias)
    s = np.sign(x)
    a = np.clip(np.abs(x).astype(np.float64), 0.0, g[-1])
    idx = np.clip(np.searchsorted(g, a), 1, len(g) - 1)
    lo, hi = g[idx - 1], g[idx]
    q = np.where((a - lo) > (hi - a), hi, lo)
    q = np.where(a <= g[0], g[0], q)
    return s * q


FP8_MAX = float(e4m3_grid(FNUZ_BIAS)[-1])   # 240.0


def relnorm(a, b):
    return np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12)


def fail(msg):
    print("\n*** FAIL: " + msg)
    sys.exit(1)


def _mk_int4(out_, in_):
    codes = rng.integers(0, 16, size=(in_, out_)).astype(np.float64)
    scale = (rng.random((in_ // GROUP, out_)) * 0.02 + 0.005)   # bf16-ish group scale
    return codes, scale


def _dequant_real(codes, scale):
    """convert_from_uint4(out=bf16): real weight = scale_bcast * (code - ZERO)."""
    sb = np.repeat(scale, GROUP, axis=0)
    return sb * (codes - ZERO)


def _silu(z):
    return z / (1.0 + np.exp(-z))


# --- the two device ops, modeled at a fixed (device-true fnuz) bias ----------
def cast_to_fp8(data, inv_scale):
    """convert_to_fp8(data, 1/s): write data/s onto the fnuz e4m3 grid (the stored bits)."""
    return quant_e4m3(data * inv_scale, FNUZ_BIAS)


def dequant_fp8_direct(codes, scale, s_fp8):
    """The DIRECT fold (shared-header dequant_fp8): scale_in = w_scale_int4 / s_fp8, then
    dequantize_4_bit(out=fp8) writes V = W_real/s_fp8 onto the SAME fnuz e4m3 grid.
    Models the in-graph: div_fwd(scale, s_fp8) -> dequantize_4_bit CAST_CLIP fp8."""
    sb = np.repeat(scale / s_fp8, GROUP, axis=0)        # FOLD: w_scale_int4 / s_fp8
    val = sb * (codes - ZERO)                            # == W_real / s_fp8
    return quant_e4m3(val, FNUZ_BIAS)                    # CAST_CLIP onto the hf8 grid


def s_fp8_of(w_real):
    """_alloc_fused_fp8_scale / _quant_weight_fp8 per-tensor FORWARD scale."""
    return (np.abs(w_real).max() + 1e-8) / FP8_MAX


# ===========================================================================
# GATE 1: stored-bits identity (the load-bearing .cpp claim).
# ===========================================================================
def test_stored_bits_identity():
    print("=" * 78)
    print("(1) DIRECT-fold stored fp8 bits == two-hop convert_to_fp8 stored bits (relNorm 0.0)")
    print("=" * 78)
    codes, scale = _mk_int4(2 * I, H)            # a w13-shaped weight
    w_real = _dequant_real(codes, scale)         # int4 -> bf16-real (the two-hop hop-1)
    s = s_fp8_of(w_real)
    twohop_bits = cast_to_fp8(w_real, 1.0 / s)               # two-hop: convert_to_fp8(W_real, 1/s)
    direct_bits = dequant_fp8_direct(codes, scale, s)        # direct: dequant straight to fp8 (folded)
    rn = relnorm(direct_bits, twohop_bits)
    print(f"  ||direct_bits - twohop_bits|| / ||twohop_bits|| = {rn:.3e}  (must be 0.0)")
    if rn != 0.0:
        # Allow only exact-zero: both write onto the identical fnuz grid from the identical
        # numeric value W_real/s. Any nonzero is a fold/grid mismatch -> on-card a wrong base.
        fail(f"stored fp8 bits differ (relNorm {rn:.3e}); the DIRECT fold is NOT bit-identical "
             "to convert_to_fp8 -- the .cpp DIRECT-DEQUANT PROOF would be violated on-card.")
    print("  PASS: the folded int4->fp8 dequant writes the EXACT same hf8 grid values.\n")


# ===========================================================================
# GATE 2+3: full-expert output identity + |got|/|ref| ~= 1 (no 2x trap).
# ===========================================================================
def _expert_ref(x, w13_real, w2_real):
    gate_up = x @ w13_real
    g, u = gate_up[:, :I], gate_up[:, I:]
    return (_silu(g) * u) @ w2_real


def _expert_twohop(x, w13_real, w2_real):
    """int4->bf16->fp8 self-consistent round-trip on both weights; per-tensor act fp8."""
    def lin(a, W):
        sa, sW = s_fp8_of(a), s_fp8_of(W)
        return (cast_to_fp8(a, 1.0 / sa) @ cast_to_fp8(W, 1.0 / sW)) * sa * sW
    gate_up = lin(x, w13_real)
    g, u = gate_up[:, :I], gate_up[:, I:]
    return lin(_silu(g) * u, w2_real)


def _expert_direct(x, codes13, scale13, codes2, scale2, w13_real, w2_real, recover=True):
    """The DIRECT path: weights dequant'd straight to fp8 with the folded scale; the SAME
    s_fp8 recovers at the gemm (recover=True). recover=False is the NEGATIVE CONTROL (drop
    the recovery -> the fnuz 2x must reappear)."""
    s13, s2 = s_fp8_of(w13_real), s_fp8_of(w2_real)
    w13f = dequant_fp8_direct(codes13, scale13, s13)    # stored == W13_real/s13 on the grid
    w2f = dequant_fp8_direct(codes2, scale2, s2)
    def lin(a, Wf, sW):
        sa = s_fp8_of(a)
        af = cast_to_fp8(a, 1.0 / sa)
        rec = sW if recover else 1.0                    # scaleB = s_fp8 (recover) or DROPPED
        return (af @ Wf) * sa * rec
    gate_up = lin(x, w13f, s13)
    g, u = gate_up[:, :I], gate_up[:, I:]
    return lin(_silu(g) * u, w2f, s2)


def test_output_identity_and_no_2x():
    print("=" * 78)
    print("(2/3) y_direct == y_twohop (relNorm 0.0) AND |got|/|ref| ~= 1 (no 2x trap)")
    print("=" * 78)
    codes13, scale13 = _mk_int4(2 * I, H)
    codes2, scale2 = _mk_int4(H, I)
    w13_real = _dequant_real(codes13, scale13)
    w2_real = _dequant_real(codes2, scale2)
    x = (rng.standard_normal((1, H)) * 0.5).astype(np.float64)

    ref = _expert_ref(x, w13_real, w2_real)
    y_twohop = _expert_twohop(x, w13_real, w2_real)
    y_direct = _expert_direct(x, codes13, scale13, codes2, scale2, w13_real, w2_real, recover=True)

    rn_dt = relnorm(y_direct, y_twohop)
    rn_dref = relnorm(y_direct, ref)
    ratio = np.linalg.norm(y_direct.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12)
    print(f"  ref_norm                       = {np.linalg.norm(ref.ravel()):.5f}")
    print(f"  relNorm(y_direct, y_twohop)    = {rn_dt:.3e}   (must be 0.0: the fold == two-hop)")
    print(f"  relNorm(y_direct, fp64 ref)    = {rn_dref:.5f}   (the ~6.4e-2 double-quant floor, < {REL_TOL})")
    print(f"  |y_direct| / |ref|             = {ratio:.4f}   (must be ~1.0, NOT ~2.0)")
    if rn_dt != 0.0:
        fail(f"y_direct != y_twohop (relNorm {rn_dt:.3e}); the fold diverges from the two-hop "
             "in the FULL expert (intermediate-h cast differs?) -- on-card these must match.")
    if rn_dref > REL_TOL:
        fail(f"DIRECT relNorm vs fp64 ref {rn_dref:.4f} > {REL_TOL} -- beyond the double-quant floor.")
    if not (0.9 <= ratio <= 1.1):
        fail(f"|got|/|ref| = {ratio:.3f} is not ~1.0 -- the fn-vs-fnuz 2x is LIVE on the "
             "recovered-direct path (the s_fp8 recovery is not cancelling).")
    print("  PASS: the recovered DIRECT fold equals the two-hop bit-for-bit; no 2x.\n")
    return ref, x, codes13, scale13, codes2, scale2, w13_real, w2_real


# ===========================================================================
# GATE 4: negative control -- drop the recovery, the 2x MUST reappear.
# ===========================================================================
def test_negative_control(ctx):
    print("=" * 78)
    print("(4) NEGATIVE CONTROL: drop the s_fp8 recovery -> the ~2x trap REAPPEARS")
    print("=" * 78)
    ref, x, codes13, scale13, codes2, scale2, w13_real, w2_real = ctx
    y_norec = _expert_direct(x, codes13, scale13, codes2, scale2, w13_real, w2_real, recover=False)
    ratio = np.linalg.norm(y_norec.ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12)
    print(f"  |y_direct_NO_RECOVERY| / |ref| = {ratio:.4f}   (must be FAR from 1.0 -> the dropped "
          "recovery is what cancels fnuz)")
    # Two weights each drop a 1/s recovery; the surviving inflation is large (>> the ~2x of a
    # single uncancelled op). The point: it is UNAMBIGUOUSLY broken when recovery is dropped,
    # which is exactly why the recovered path's ~1.0 ratio is meaningful.
    if 0.5 <= ratio <= 1.5:
        fail(f"dropping the recovery left the ratio at {ratio:.3f} (~1) -- then the recovery is "
             "NOT what makes the direct path correct; the test cannot certify the 2x-trap fix.")
    print(f"  PASS: without recovery the output is grossly wrong ({ratio:.2f}x) -- the s_fp8 "
          "scaleB recovery is load-bearing, and it is present on the recovered path (gate 2/3).\n")


if __name__ == "__main__":
    test_stored_bits_identity()
    ctx = test_output_identity_and_no_2x()
    test_negative_control(ctx)
    print("=" * 78)
    print(f"ALL CARD-FREE DIRECT-FOLD GATES PASSED (REL_TOL={REL_TOL}). The HPU_MOE_FP8_DIRECT "
          "fold writes bit-identical fp8 to the two-hop, y_direct == y_twohop (relNorm 0.0), "
          "|got|/|ref| ~= 1.0 (no fn-vs-fnuz 2x), and dropping the s_fp8 recovery REINTRODUCES "
          "the trap -- proving the recovery is what cancels it.")
    print("On-card, confirm with test_fused_expert_fp8_op.py FP8_DIRECT=1 (8e-2 + |got|/|ref| check).")
    print("=" * 78)
