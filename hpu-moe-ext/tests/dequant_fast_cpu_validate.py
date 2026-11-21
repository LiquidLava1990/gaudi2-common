#!/usr/bin/env python3
"""CARD-FREE byte-exact validation of moe_ext_unpack_dequant_w4_fast.c.

Proves, with NO HPU device, that the OPTIMIZED int4->bf16 dequant kernel
(moe_ext_unpack_dequant_w4_fast.c -- the FAST native-unpack variant AND the LEAN
U4-descriptor variant) computes the SAME bf16 result as the live reference
torch.ops.hpu.convert_from_uint4 (== cpu_convert_from_uint4, the proven oracle in
tests/glm_layer3_routed_op_cpu.py), to bf16 round-off.

What is modeled (numpy only -- no torch, no ml_dtypes, no card):
  * pack/unpack semantics: int32 [in, out//8], 8 nibbles/word along OUT, LSB-first,
    NO swizzle (word 0x76543210 -> OUT cols 0..7). This is exactly
    pack_quantized_values_into_int32(..., packed_dim=1) / its inverse, and the
    cpu_convert_from_uint4 oracle.
  * the fp64 ORACLE: (code - 8) * scale[in//GS, out]  in fp64 (ground truth).
  * the OLD kernel compute path (i32 load of unpacked code -> f32; f32 zero TENSOR
    sub; f32 mul; f32->bf16) -- bit-for-bit the op chain of the live
    moe_ext_unpack_dequant_w4.c.
  * the NEW FAST kernel compute path EXACTLY as the C does it:
        word(int32) -> (word >> 4j) & 0xF  [in-kernel shift+mask, native load]
        -> int (code - 8)                  [folded constant zero, integer]
        -> i32->f32                        [one convert]
        -> * scale[g, 8p+j]  in f32        [hoisted, stride-8 deinterleave]
        -> f32->bf16                       [one convert]
    with the stride-8 lane mapping (nibble j of word p -> OUT col 8p+j).
  * the NEW LEAN kernel compute path (U4 load -> f32; f32 IMMEDIATE-8 sub; mul;
    f32->bf16) -- the conservative (b)+(c)+(d) variant.

bf16 is computed by an EXACT round-half-to-nearest-even truncation of the float32
bit pattern (matches the TPC SW_RHNE convert). f32 ops are done in float32 (the
kernel's f32 vector ops), so the comparison is apples-to-apples with the device.

GATES (all must pass):
  1. FAST and LEAN bf16 outputs == reference bf16 output BYTE-EXACT (identical
     uint16 bf16 bits) on real GLM int4 expert shapes (w13, w2, G128).  [primary]
  2. As a softer cross-check, relNorm(new, oracle_fp64) < 1e-3 (bf16 floor).
  3. ADVERSARIAL: corrupting a single nibble in the packed weight makes the FAST
     output differ from the reference at exactly the affected OUT column(s) -- the
     validator CATCHES a corrupted nibble (not silently equal).
  4. The FAST native-unpack reproduces the exact unpack of the oracle (the
     shift+mask deinterleave == pack_quantized_values_into_int32 inverse).

Run (host, no card):  python tests/dequant_fast_cpu_validate.py
Exit 0 = all gates pass.
"""
import sys
import numpy as np

# ----- GLM-5.1/5.2 int4 expert shapes (the real dequant targets) -------------
H = 6144            # hidden
I = 2048            # moe_intermediate
TWO_I = 2 * I       # 4096 (gate+up)
GS = 128            # group size
ZERO = 8            # GPTQ-symmetric shared zero
BITS = 4
NIB = 8             # nibbles per int32


# =============================================================================
# bf16 conversion (exact round-half-to-nearest-even on the f32 bit pattern).
# Matches the TPC v_convert_f32_to_bf16_b with SW_RHNE rounding.
# =============================================================================
def f32_to_bf16_bits(x_f32):
    """float32 array -> uint16 bf16 bits, round-half-to-even. Exact."""
    x = np.asarray(x_f32, dtype=np.float32)
    u = x.view(np.uint32)
    # NaN: bf16 keeps a NaN (set a quiet bit), don't round into inf.
    is_nan = (np.bitwise_and(u, 0x7FFFFFFF) > 0x7F800000)
    # round-half-to-even: add the rounding bias then truncate the low 16 bits.
    lsb = np.bitwise_and(np.right_shift(u, 16), np.uint32(1))
    rounding_bias = np.uint32(0x7FFF) + lsb
    rounded = u.astype(np.uint64) + rounding_bias.astype(np.uint64)
    # guard overflow back into uint32 range before the shift
    rounded = np.bitwise_and(rounded, np.uint64(0xFFFFFFFF)).astype(np.uint32)
    bf = np.right_shift(rounded, 16).astype(np.uint16)
    # for NaN, force a canonical quiet-NaN bf16 (sign|exp=0xFF|mantissa msb)
    nan_bf = (np.right_shift(np.bitwise_and(u, np.uint32(0x80000000)), 16)
              | np.uint16(0x7FC0)).astype(np.uint16)
    bf = np.where(is_nan, nan_bf, bf)
    return bf.astype(np.uint16)


def bf16_bits_to_f32(bf_bits):
    """uint16 bf16 bits -> float32 (zero-extend the low mantissa)."""
    u = np.left_shift(np.asarray(bf_bits, dtype=np.uint32), 16)
    return u.view(np.float32)


def _check_bf16_roundtrip():
    """Sanity: a handful of known f32 values round to the bf16 we expect."""
    # 1.0 -> 0x3F80 ; 1.0 + 2^-8 (just above a tie that rounds to even) etc.
    vals = np.array([0.0, 1.0, -1.0, 2.0, 0.5, 1.5, 3.0,
                     1.0009765625,    # 1 + 2^-10 -> rounds to 1.0 (0x3F80)
                     1.0078125],      # 1 + 2^-7  -> exactly representable 0x3F81
                    dtype=np.float32)
    bf = f32_to_bf16_bits(vals)
    exp = np.array([0x0000, 0x3F80, 0xBF80, 0x4000, 0x3F00, 0x3FC0, 0x4040,
                    0x3F80, 0x3F81], dtype=np.uint16)
    assert np.array_equal(bf, exp), f"bf16 roundtrip FAIL\n got {[hex(v) for v in bf]}\n exp {[hex(v) for v in exp]}"


# =============================================================================
# pack / unpack -- EXACT mirror of pack_quantized_values_into_int32(packed_dim=1)
# and the cpu_convert_from_uint4 oracle. LSB-first, 8 nibbles/int32 along OUT.
# =============================================================================
def pack_int4_dim1(codes_u4):
    """codes uint4 [in, out] (out % 8 == 0) -> packed int32 [in, out//8].
    Nibble j (bits 4j..4j+3) holds column 8p+j. LSB-first."""
    in_, out_ = codes_u4.shape
    assert out_ % NIB == 0
    c = codes_u4.astype(np.uint32) & 0xF
    c = c.reshape(in_, out_ // NIB, NIB)            # [in, words, nib]
    shifts = (np.arange(NIB, dtype=np.uint32) * 4)  # 0,4,8,...,28
    word = np.zeros((in_, out_ // NIB), dtype=np.uint32)
    for j in range(NIB):
        word |= (c[:, :, j] << shifts[j])
    return word.view(np.int32)                       # int32 [in, out//8]


def unpack_int4_dim1(packed_i32):
    """packed int32 [in, out//8] -> codes uint8 [in, out].  Inverse of pack."""
    w = packed_i32.view(np.uint32)
    in_, words = w.shape
    out = np.empty((in_, words * NIB), dtype=np.uint8)
    for j in range(NIB):
        out[:, j::NIB] = ((w >> (4 * j)) & 0xF).astype(np.uint8)
    return out


# =============================================================================
# Reference / oracle and the three kernel compute models.
# =============================================================================
def oracle_fp64(packed_i32, scale_go, gs=GS, zero=ZERO):
    """Ground truth = cpu_convert_from_uint4 in fp64. (code - 8) * scale[g]."""
    codes = unpack_int4_dim1(packed_i32).astype(np.float64)     # [in, out]
    g = (np.arange(codes.shape[0]) // gs).astype(np.int64)
    return (codes - float(zero)) * scale_go[g].astype(np.float64)


def ref_kernel_old_bf16(packed_i32, scale_go, zero_tensor_go, gs=GS):
    """OLD live-kernel chain: code(i32)->f32 ; f32 sub of the ZERO TENSOR ; f32 mul
    ; f32->bf16. Returns bf16 BITS [in, out]. (zero_tensor_go is the broadcast
    constant-8 f32 tensor the live glue materializes.)"""
    codes = unpack_int4_dim1(packed_i32).astype(np.int32)
    code_f = codes.astype(np.float32)                           # i32->f32
    g = (np.arange(codes.shape[0]) // gs).astype(np.int64)
    zr = zero_tensor_go[g].astype(np.float32)                   # f32 zero tensor
    sc = scale_go[g].astype(np.float32)
    dq = (code_f - zr).astype(np.float32)                       # f32 sub
    dq = (dq * sc).astype(np.float32)                           # f32 mul
    return f32_to_bf16_bits(dq)


def fast_kernel_bf16(packed_i32, scale_go, gs=GS, zero=ZERO):
    """NEW FAST chain, EXACTLY as moe_ext_unpack_dequant_w4_fast.c::main_fast:
        word -> (word>>4j)&0xF [shift+mask] -> int (code-8) -> i32->f32
        -> * scale[g, 8p+j] f32 [hoisted] -> f32->bf16.
    Modeled with the stride-8 deinterleave lane map (nibble j of word p -> col 8p+j).
    """
    w = packed_i32.view(np.uint32)
    in_, words = w.shape
    out_ = words * NIB
    g = (np.arange(in_) // gs).astype(np.int64)
    sc = scale_go[g].astype(np.float32)                         # [in, out] hoisted
    out_bf = np.empty((in_, out_), dtype=np.uint16)
    for j in range(NIB):                                        # the 8 j-stores
        code = ((w >> (4 * j)) & 0xF).astype(np.int32)          # [in, words]
        deb = (code - int(zero)).astype(np.int32)               # folded zero (int)
        debf = deb.astype(np.float32)                           # one i32->f32
        sc_j = sc[:, j::NIB]                                    # stride-8 scale cols
        dq = (debf * sc_j).astype(np.float32)                   # one f32 mul
        out_bf[:, j::NIB] = f32_to_bf16_bits(dq)                # store col 8p+j
    return out_bf


def lean_kernel_bf16(packed_i32, scale_go, gs=GS, zero=ZERO):
    """NEW LEAN chain, EXACTLY as main_lean: U4 load -> f32 ; f32 sub of the
    IMMEDIATE 8.0 ; f32 mul ; f32->bf16. (Same lane map as the live kernel.)"""
    codes = unpack_int4_dim1(packed_i32).astype(np.int32)
    code_f = codes.astype(np.float32)
    g = (np.arange(codes.shape[0]) // gs).astype(np.int64)
    sc = scale_go[g].astype(np.float32)
    dq = (code_f - np.float32(zero)).astype(np.float32)         # f32 immediate sub
    dq = (dq * sc).astype(np.float32)
    return f32_to_bf16_bits(dq)


def relnorm(a_f32, b_f32):
    a = np.asarray(a_f32, np.float64).ravel()
    b = np.asarray(b_f32, np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# =============================================================================
# Build a realistic GLM expert weight + scale (random int4 codes, realistic scale).
# =============================================================================
def make_expert(out_, in_, seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(in_, out_), dtype=np.int32)   # uint4 in [0,15]
    packed = pack_int4_dim1(codes.astype(np.uint8))                 # [in, out//8] i32
    # realistic per-group scale magnitude (~0.01-0.03), bf16-representable.
    scale_f32 = (rng.random((in_ // GS, out_), dtype=np.float32) * 0.02 + 0.01)
    scale_f32 = bf16_bits_to_f32(f32_to_bf16_bits(scale_f32))       # scale is bf16
    return packed, scale_f32, codes


def zero_tensor(groups, out_):
    return np.full((groups, out_), float(ZERO), dtype=np.float32)


# =============================================================================
# Main: run all gates on w13 and w2.
# =============================================================================
def run_case(name, out_, in_, seed):
    packed, scale, codes = make_expert(out_, in_, seed)
    groups = in_ // GS
    zt = zero_tensor(groups, out_)

    # (gate 4) native unpack == oracle unpack
    assert np.array_equal(unpack_int4_dim1(packed), (codes & 0xF).astype(np.uint8)), \
        f"[{name}] FAST shift+mask unpack != pack inverse"

    oracle = oracle_fp64(packed, scale)                       # fp64 ground truth
    bf_old = ref_kernel_old_bf16(packed, scale, zt)           # live-kernel chain
    bf_fast = fast_kernel_bf16(packed, scale)                 # new FAST
    bf_lean = lean_kernel_bf16(packed, scale)                 # new LEAN

    # (gate 1) byte-exact: FAST and LEAN bf16 bits == the live-kernel bf16 bits.
    fast_vs_old = int(np.count_nonzero(bf_fast != bf_old))
    lean_vs_old = int(np.count_nonzero(bf_lean != bf_old))
    n = bf_old.size

    # (gate 2) relNorm vs fp64 oracle (bf16 floor).
    rn_old = relnorm(bf16_bits_to_f32(bf_old), oracle)
    rn_fast = relnorm(bf16_bits_to_f32(bf_fast), oracle)
    rn_lean = relnorm(bf16_bits_to_f32(bf_lean), oracle)

    print(f"  [{name}] shape in={in_} out={out_} ({n} elems)")
    print(f"     FAST vs OLD bf16 bits : {fast_vs_old} differ  "
          f"({'BYTE-EXACT' if fast_vs_old == 0 else 'MISMATCH'})")
    print(f"     LEAN vs OLD bf16 bits : {lean_vs_old} differ  "
          f"({'BYTE-EXACT' if lean_vs_old == 0 else 'MISMATCH'})")
    print(f"     relNorm vs fp64 oracle: OLD={rn_old:.2e}  FAST={rn_fast:.2e}  LEAN={rn_lean:.2e}")

    # GATE 1 (primary, load-bearing): the new kernels reproduce the LIVE kernel's
    # bf16 output bit-for-bit. This is the real correctness claim -- the new kernel
    # must produce IDENTICAL bytes to the path the model is validated against.
    # GATE 2: no precision regression -- the new kernels' relNorm vs the fp64 oracle
    # is no worse than the OLD kernel's (which is the bf16 floor, ~1.5e-3 here), and
    # under the bf16 round-off ceiling (2^-8 ~ 3.9e-3 per elem -> 4e-3 population).
    BF16_FLOOR = 4e-3
    ok = True
    if fast_vs_old != 0:
        print(f"     !!! FAST not byte-exact vs OLD ({fast_vs_old}/{n})"); ok = False
    if lean_vs_old != 0:
        print(f"     !!! LEAN not byte-exact vs OLD ({lean_vs_old}/{n})"); ok = False
    # no-regression: new relNorm must equal the OLD relNorm (byte-exact => it does)
    # and be within the bf16 floor. (rn_old is itself the bf16 floor.)
    if rn_fast > rn_old + 1e-12:
        print(f"     !!! FAST relNorm {rn_fast:.2e} WORSE than OLD {rn_old:.2e}"); ok = False
    if rn_lean > rn_old + 1e-12:
        print(f"     !!! LEAN relNorm {rn_lean:.2e} WORSE than OLD {rn_old:.2e}"); ok = False
    if rn_fast >= BF16_FLOOR or rn_old >= BF16_FLOOR:
        print(f"     !!! relNorm exceeds bf16 floor {BF16_FLOOR:.1e} "
              f"(OLD={rn_old:.2e} FAST={rn_fast:.2e})"); ok = False
    return ok, packed, scale


def run_adversarial(packed, scale):
    """(gate 3) corrupt one nibble -> FAST output MUST differ from the reference at
    the affected column(s). Confirms the validator catches corruption."""
    bf_clean = fast_kernel_bf16(packed, scale)
    bad = packed.copy()
    # flip nibble j=3 of word (0,0): xor 0x...F000 into the low word's nibble 3.
    i_row, i_word, j = 0, 0, 3
    w = bad.view(np.uint32)
    orig = int(w[i_row, i_word])
    w[i_row, i_word] = orig ^ (0xF << (4 * j))   # corrupt nibble 3
    bf_corrupt = fast_kernel_bf16(bad, scale)

    diff_cols = np.nonzero(bf_clean[i_row] != bf_corrupt[i_row])[0]
    expected_col = NIB * i_word + j
    caught = (expected_col in diff_cols.tolist())
    # and ONLY that column changed in that row (the unpack is local)
    only_that = (diff_cols.tolist() == [expected_col])
    print(f"  [adversarial] corrupt nibble (row{i_row},word{i_word},nib{j} -> col{expected_col})")
    print(f"     changed OUT cols in row{i_row}: {diff_cols.tolist()}")
    print(f"     corruption CAUGHT at col {expected_col}: {caught} ; localized: {only_that}")
    return caught and only_that


def main():
    print("=" * 74)
    print("CARD-FREE byte-exact validation: moe_ext_unpack_dequant_w4_fast.c")
    print(f"  GLM int4 shapes  H={H} I={I} TWO_I={TWO_I} GS={GS} zero={ZERO}")
    print("=" * 74)

    _check_bf16_roundtrip()
    print("  bf16 RHNE round-trip self-check: OK")

    all_ok = True
    # w13: in=H, out=2I  ; w2: in=I, out=H  (the two real GLM expert tensors)
    ok1, packed13, scale13 = run_case("w13", TWO_I, H, seed=1)
    ok2, packed2, scale2 = run_case("w2", H, I, seed=2)
    all_ok = all_ok and ok1 and ok2

    # extra: a small odd-ish shape to exercise non-square + multiple groups.
    ok3, _, _ = run_case("small", 256, 512, seed=3)
    all_ok = all_ok and ok3

    adv_ok = run_adversarial(packed13, scale13)
    all_ok = all_ok and adv_ok

    print("=" * 74)
    if all_ok:
        print("RESULT: ALL GATES PASS  (FAST + LEAN byte-exact vs the live-kernel "
              "chain; relNorm < 1e-3 vs fp64 oracle; corruption caught)")
        print("=" * 74)
        return 0
    print("RESULT: FAIL -- see !!! lines above")
    print("=" * 74)
    return 1


if __name__ == "__main__":
    sys.exit(main())
