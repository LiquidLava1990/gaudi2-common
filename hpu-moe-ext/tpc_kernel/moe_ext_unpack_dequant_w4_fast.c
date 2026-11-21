/*
 * moe_ext_unpack_dequant_w4_fast.c
 *
 * OPTIMIZED int4 -> bf16 GPTQ-symmetric dequant for Gaudi2 (tpc-clang -mcpu=gaudi2).
 *
 * GUIDs (two variants in one TU, selected by the glue):
 *   moe_ext_unpack_dequant_w4_fast_bf16   - the de-amplified NATIVE-int32 unpack
 *                                            (attacks inefficiency (a) head-on)
 *   moe_ext_unpack_dequant_w4_lean_bf16   - keeps the proven U4-descriptor load,
 *                                            attacks ONLY (b)/(c)/(d). Lower risk;
 *                                            the conservative fallback if the
 *                                            native-unpack lane mapping (a) needs
 *                                            an on-card lane-shuffle tweak.
 *
 * This is a NEW VARIANT FILE. The live kernel moe_ext_unpack_dequant_w4.c is NOT
 * modified, and no file the in-flight int4-MoE / fp8 A/B uses is touched.
 *
 * ===========================================================================
 * THE FOUR TARGETS (mmel/DEQUANT_KERNEL_DIG.md, a6f1d7fc) and how each is hit
 * ===========================================================================
 * (a) 8x READ AMPLIFICATION. Live kernel: each 4-bit code is loaded as a full
 *     int32 lane (the U4 tensor descriptor presents one unpacked nibble per i32
 *     load) -> 4 bytes of VRF/SRAM lane width per 0.5 B of real data. The packed
 *     int4 weight is physically int32 [in, out//8] (8 nibbles/word). The
 *     descriptor-based load reads each word EIGHT times (once per produced code).
 *     FIX (fast variant): load the PLAIN int32 word ONCE (64 lanes of packed
 *     words) and unpack all 8 nibbles in-kernel with shift+mask. One i32 load now
 *     feeds 8 output codes -> the internal code-load traffic drops ~8x.
 *
 * (b) f32 MATH + redundant per-element ZERO load. Live: (code - zero_tensor)*scale
 *     in f32, with the GPTQ-symmetric zero (constant 8) MATERIALIZED as a per-
 *     element f32 tensor load + an f32 sub. FIX: drop the zero tensor input, its
 *     load, and the f32 sub; fold the constant 8 as an INTEGER immediate subtract
 *     on the nibble BEFORE the convert. (Saves one tensor load + one f32 vop per
 *     element; the integer sub is cheaper than the f32 sub and removes an input.)
 *
 * (c) per-element SCALE load. Live: a scale load on EVERY (in,out) step although
 *     the G128 scale is constant across 128 contiguous IN rows. FIX: hoist the
 *     scale load to ONCE per group (the index space dim1 is tiled at group_size by
 *     the glue, so the scale-load guard fires once per group tile and the loaded
 *     vector is reused across all group_size IN rows).
 *
 * (d) THIRD CONVERT pass. Live: i32->f32 (codes), then sub, then mul, then a
 *     separate f32->bf16 convert. FIX: one i32->f32 of the de-biased nibble, one
 *     f32 mul by the hoisted scale, one f32->bf16. The multiply is the ONLY f32
 *     vector op; the zero pass is gone. (We keep the multiply in f32 rather than
 *     bf16 to preserve the int4 path's ~1e-2 accuracy bar; the validation gate
 *     confirms bf16 round-off relNorm < 1e-3 vs the oracle.)
 *
 * ===========================================================================
 * EXACT SEMANTICS (bit-exact to bf16 round-off vs torch.ops.hpu.convert_from_uint4)
 * ===========================================================================
 *   packed word layout : int32 [in, out//8], 8 nibbles/word along OUT, LSB-first,
 *                         VERIFIED no swizzle (word 0x76543210 -> OUT cols 0..7).
 *   out[i, 8*p + j] = ( (f32)((word[i,p] >> (4*j)) & 0xF) - 8.0f )
 *                       * scale[i / group_size, 8*p + j]
 *   scale : f32 [groups, out], groups = in // group_size, constant across the
 *           group's IN rows (GLM desc_act=False, contiguous groups).
 *
 * ===========================================================================
 * LANE-MAPPING NOTE (the one on-card risk for the FAST variant) — read this
 * ===========================================================================
 * On a SIMD vector machine the nibble unpack is a DEINTERLEAVE: nibble j of 64
 * consecutive packed words lands at OUT columns {8*p + j : p = p0..p0+63}, i.e. a
 * stride-8 column pattern, NOT contiguous OUT. Two correct ways to land it:
 *   (1) STORE STRIDED: write the 64-lane nibble-j vector to OUT with column
 *       stride 8 (out0 access pattern step = 8 on the FCD for the inner store),
 *       and load the matching scale with the SAME stride-8 pattern. This is the
 *       formulation below: it is provably correct (lane k of every vector
 *       corresponds to packed word p0+k, i.e. OUT column 8*(p0+k)+j), needs only a
 *       strided tensor access pattern (set in the glue), and uses NO lane-shuffle
 *       intrinsic. The 8 j-stores tile the full contiguous OUT.
 *   (2) IN-VRF DEINTERLEAVE: unpack into a contiguous 64-lane OUT vector with a
 *       lane-permute/shuffle. Lower store count but needs a lane-shuffle intrinsic
 *       whose exact -tpc-special spelling must be confirmed on the box. Deferred.
 * Formulation (1) is what this file implements (correct + card-free-verifiable in
 * its math; the only on-card item is wiring the stride-8 access pattern in the
 * glue, which the validation models exactly). The LEAN variant sidesteps all of
 * this by keeping the U4 descriptor load.
 *
 * TOOLCHAIN: tpc-clang 2.7.33 -mcpu=gaudi2 -O2. Intrinsics (all -tpc-special):
 *   v_i32_ld_tnsr_b, v_i32_shr_b, v_i32_and_b, v_i32_sub_b (NEW vs the live
 *   kernel: the integer shift/and/sub — standard TPC int vector ops),
 *   v_convert_i32_to_f32_b, v_f32_ld_tnsr_b, v_f32_mul_b, v_f32_sub_b,
 *   v_convert_f32_to_bf16_b, v_bf16_st_tnsr (all proven in the live kernel).
 */

#ifndef SW_RHNE
#define SW_RHNE 0x0   /* round half to nearest even */
#endif

#define ZERO_BIAS    8   /* GPTQ-symmetric uint4 shared zero (compile-time const) */
#define NIB_PER_WORD 8
#define NIB_MASK     0xF

/* User params struct — ABI-identical to MoeUnpackDequantW4Params (live glue) so
 * the same FillParamsFn works unchanged. */
struct MoeUnpackDequantW4FastParams {
    int group_size;   /* rows-per-group along IN (128 for GLM). Used for the scale
                         hoist cadence; the index space dim1 is tiled at this size
                         by the glue so the scale-load guard fires once per group. */
    int zero_const;   /* GPTQ-sym shared zero; the body folds the ZERO_BIAS
                         immediate (branch-free). Reserved for an AWQ per-group path. */
};

/* ===========================================================================
 * DEQUANT KERNEL — STOCK HARDWARE U4->U8 DEINTERLEAVE (CORRECTED 2026-06-23 v3).
 *
 * Both GUIDs (fast/lean) now share this body. The prior approaches were on-card-proven
 * WRONG:
 *   - stride-8 in-VPU store: the TPC store writes 64 lanes CONTIGUOUSLY (no per-lane
 *     FCD stride) -> the nibble deinterleave didn't land -> relNorm ~2.5.
 *   - "U4-descriptor" via v_i32_ld_tnsr_b on a uint4 tensor: the i32 load does NOT
 *     auto-unpack nibbles; it read raw words/garbage -> on-card output was a CONSTANT
 *     (= the scale), the codes never reached the compute (value-dump ground truth).
 *
 * CORRECT PATH (the one the STOCK dequant kernel uses — verified by disassembling its
 * ISA + the stock v_convert_mxfp4/nf4_to_bf16 source in libtpc_kernels.so):
 *   load the packed bytes, then the HARDWARE u4->u8 unpack intrinsic
 *   v_convert_u4_to_u8_all_b deinterleaves 256 bytes (=512 nibbles) -> 512 u8 codes, then
 *   the WIDTH-PRESERVING convert_uchar256_to_float256 (the convert the STOCK dequant uses)
 *   gives 256 f32 in OUT ORDER. We store CONTIGUOUSLY. (The earlier v_convert_u8_to_f32_b
 *   was width-NARROWING 256->64 and SCRAMBLED the lanes into a dual-group order -- on-card
 *   value-dump showed device col 4m<-true m, 4m+2<-true m+8, odd=0. The order-preserving
 *   uchar256->float256 convert fixes that.)
 *
 *   in0 weight_packed : uint8 [OUT/2, IN]  (the int32-packed [OUT/8,IN] viewed as bytes;
 *                       FCD = byte index along OUT, 2 nibbles/byte, LSB-first)
 *   in1 weight_scale  : f32   [OUT, groups] (FCD = OUT; group row along IN; a=0 hoist)
 *   out0              : bf16  [OUT, IN]     (FCD = OUT, contiguous)
 *   out[o,in] = ((f32)nibble(o) - 8) * scale[o, in/gs]
 *
 * Index space dim0 walks OUT in 256-code steps (=> 128 bytes/step; the 256-byte v_convert
 * window reads 256 bytes but only the first 256 deinterleaved codes (.v1) = 256 OUT cols
 * are consumed). Each step: load 256 bytes, hw-deinterleave, convert_uchar256_to_float256
 * -> float256{.v1..v4} = 256 codes in OUT order, processed as 4 components of 64 lanes:
 * (code-8)*scale -> bf16 -> store 64 contiguous OUT codes at os+k*64. dim1 = IN.
 * PRED NOTE: every op tail is (..., income=0, predicate=1); predicate=0 made the compiler
 * constant-fold the whole chain to the income (the constant-1.0 bug). =================== */
void dequant_body(tensor weight_packed, tensor weight_scale, tensor out0)
{
    const int5 start = get_index_space_offset();
    const int5 size  = get_index_space_size();

    /* INDEX-SPACE -> COORD (matches the stock kernel: the index-space offset is in STEP
     * units; the actual OUT-col base = offset * kStep, iterated kStep at a time). kStep=512
     * (one SW_UNPACK 256-byte window -> v_convert_u4_to_u8_all_b -> 512 OUT codes). */
    /* kStep = 512 OUT codes per index-space dim0 step = 256 packed BYTES. */
    const int kStep = 512;
    const int o_lo  = start[0] * kStep;
    const int o_hi  = (start[0] + size[0]) * kStep;
    const int in_lo = start[1];
    const int in_hi = start[1] + size[1];

    int5 idx_b = start;   /* packed bytes : [byte, in]      */
    int5 idx_o = start;   /* out0         : [out, in]       */
    int5 idx_s = start;   /* weight_scale : [out, group]    */

    for (int os = o_lo; os < o_hi; os += kStep) {
        idx_b[0] = (os >> 1);          /* byte_base = os/2 (256 bytes -> 512 OUT codes)  */

        for (int in = in_lo; in < in_hi; in += 1) {
            idx_b[1] = in;

            /* EXPLICIT nibble unpack (full lane control -- v_convert_u4_to_u8_all_b's lane
             * order was opaque/wrong: it returned the LOW nibbles but the HIGH nibbles came
             * back as the zero-point (only lows unpacked) on-card). Byte b holds OUT cols
             * (2b = low, 2b+1 = high). We combine  comb = low | (high<<8)  per byte; the u16
             * vector reinterpreted as bytes is exactly [low0,high0,low1,high1,...] = the
             * natural OUT order (confirmed on-card: lows already land right). */
            uchar256  b   = v_u8_ld_tnsr_b(idx_b, weight_packed);          /* 256 bytes      */
            ushort256 b16 = convert_uchar256_to_ushort256(b, SW_LINEAR);  /* 256 u16 (.v1/.v2 of 128) */
            ushort256 comb;
            comb.v1 = v_u16_or_b(v_u16_and_b(b16.v1, 0x000F, 0, 0, 0, 1),
                       v_u16_shl_b(v_u16_and_b(v_u16_shr_b(b16.v1, 4, 0, 0, 0, 1), 0x000F, 0, 0, 0, 1),
                                   8, 0, 0, 0, 1), 0, 0, 0, 1);
            comb.v2 = v_u16_or_b(v_u16_and_b(b16.v2, 0x000F, 0, 0, 0, 1),
                       v_u16_shl_b(v_u16_and_b(v_u16_shr_b(b16.v2, 4, 0, 0, 0, 1), 0x000F, 0, 0, 0, 1),
                                   8, 0, 0, 0, 1), 0, 0, 0, 1);
            /* reinterpret the FULL 256 u16 (512 bytes) as uchar512 = [lo0,hi0,lo1,hi1,...,lo255,hi255]
             * = ALL 512 OUT codes in natural order. codes.v1 = cols os..os+255, .v2 = os+256.. */
            uchar512 codes = *(uchar512*)&comb;

            /* ALL-BF16 path (1:1, 128-native -- NO float32 intermediate). The f32 route paired
             * two float64 into a float128 which v_convert_f32_to_bf16_all_b INTERLEAVED
             * (a0,b0,a1,b1,..) -> the two 64-blocks landed interleaved across even/odd OUT cols.
             * Instead convert u8 codes -> bf16 DIRECTLY (the stock-style 1:1 256->256 convert; the
             * uchar256_to_X256 family is lane-1:1 -- the float256 sibling landed codes sequentially
             * on-card) and do (code-8)*scale entirely in bf16 (codes 0-15 -> code-8 is bf16-exact;
             * matches the f32 reference to relNorm 0). Each bfloat128 (.v1/.v2 of the bfloat256) is a
             * CONTIGUOUS 128-code block; 4 stores tile cols os..os+511 in order. */
            bfloat256 cf0 = convert_uchar256_to_bfloat256(codes.v1, SW_LINEAR);  /* cols os..os+255  */
            bfloat256 cf1 = convert_uchar256_to_bfloat256(codes.v2, SW_LINEAR);  /* cols os+256..511 */
            bfloat128 cc[4] = { cf0.v1, cf0.v2, cf1.v1, cf1.v2 };
            bfloat128 zbf = v_bf16_mov_b((float)ZERO_BIAS, 0, 0, 0, 1);

            for (int p = 0; p < 4; p += 1) {
                const int kc = os + p * 128;            /* OUT col base for this 128-block      */
                /* scale [OUT, group]: group = in/128 (GLM G128 hardcoded; the generic glue can't
                 * pass group_size as a kernel arg here, like the stock mxfp4 cast). bf16 load. */
                idx_s[0] = kc; idx_s[1] = in / 128;
                bfloat128 sc = v_bf16_ld_tnsr_b(idx_s, weight_scale);
                bfloat128 dq = v_bf16_mul_b(v_bf16_sub_b(cc[p], zbf, 0, 0, 0, 1), sc, 0, 0, 0, 1);
                idx_o[0] = kc; idx_o[1] = in;
                v_bf16_st_tnsr(idx_o, out0, dq);        /* 128 contiguous OUT codes */
            }
        }
    }
}

void main_fast(tensor weight_packed, tensor weight_scale, tensor out0) {
    dequant_body(weight_packed, weight_scale, out0);
}
void main_lean(tensor weight_packed, tensor weight_scale, tensor out0) {
    dequant_body(weight_packed, weight_scale, out0);
}

/* The graph compiler entry per GUID is `main`. The glue registers TWO GUIDs, each
 * pointing at this TU; we provide both `main_fast` and `main_lean` and the build
 * selects via -DKERNEL_VARIANT. (A single TU can be compiled twice with different
 * -D to emit the two named kernels, matching the two-GUID registration in the
 * matching .cpp; see build_fast.sh.) */
#ifndef KERNEL_VARIANT
#define KERNEL_VARIANT 1   /* 1 = fast (native unpack), 2 = lean (U4 descriptor) */
#endif

#if KERNEL_VARIANT == 1
void main(tensor weight_packed, tensor weight_scale, tensor out0) {
    main_fast(weight_packed, weight_scale, out0);
}
#else
void main(tensor weight_packed, tensor weight_scale, tensor out0) {
    main_lean(weight_packed, weight_scale, out0);
}
#endif
