/*
 * moe_ext_unpack_dequant_w4.c
 *
 * Phase-0 fused-MoE foundation kernel (Gaudi2 TPC-C / tpc-clang "rc99" dialect).
 *
 * GUID:  moe_ext_unpack_dequant_w4_bf16
 *
 * Semantics — BIT-EXACT mirror of
 *     torch.ops.hpu.convert_from_uint4(weight_packed, weight_scale, zero_point,
 *                                      torch.bfloat16, None)
 * as used by VllmMixtureOfExpertsOpWNA16Expert.get_dequant_weight
 * (/opt/vllm-gaudi/vllm_gaudi/extension/ops.py:1448) and the proven CPU
 * reference cpu_convert_from_uint4 (tests/glm_layer3_routed_op_cpu.py).
 *
 *   in0  weight_packed : logical uint4 [in, out]  (DATA_U4). On disk it is int32
 *                        [in, out/8], 8 nibbles/int32 along OUT. The graph
 *                        compiler / TPC tensor descriptor presents a U4 tensor;
 *                        v_u4_ld_tnsr / the U4->i32 load yields the unpacked code.
 *                        Nibble j (bits[4j..4j+3], LSB first) -> column 8*p+j.
 *                        VERIFIED no swizzle: 0x76543210 -> 0,1,2,3,4,5,6,7.
 *   in1  weight_scale  : f32 [groups, out]   (caller passes f16->f32 upcast, or
 *                        the f16 tensor with a U4? no: scale is f16/f32 — see glue;
 *                        the kernel reads it as f32 lanes).
 *   in2  zero_point    : f32 [groups, out]   GPTQ-symmetric => constant 8.0. The
 *                        glue materializes the broadcast zero as f32 so the kernel
 *                        stays a pure elementwise (code - zero)*scale. (AWQ per-
 *                        group zero is a later phase; same load path.)
 *   out0 out           : bf16 [in, out]
 *
 *   out[i,o] = ( (f32)code[i,o] - zero[g,o] ) * scale[g,o] ,   g = i / group_size
 *
 * Index space (set by the glue's TensorAccessPattern): the FCD (dim0) carries
 * OUT in 64-lane (float64 / 256B) chunks; dim1 carries IN one element per step.
 * scale/zero are indexed by the group row g = in/group_size; the glue exposes
 * them already broadcast along IN OR the kernel recomputes g — here we rely on
 * the standard "scale/zero tensors share the OUT FCD and are addressed at the
 * group row" pattern used by grouped_index_transpose_dequant_2d_with_zp.
 *
 * Intrinsic names below are VERIFIED to parse+compile with
 *   tpc-clang 2.7.33 -mcpu=gaudi2 -O2
 * (v_u4_ld_tnsr_b / v_i32_ld_tnsr_b, v_convert_i32_to_f32_b, v_f32_ld_tnsr_b,
 *  v_f32_sub_b, v_f32_mul_b, v_convert_f32_to_bf16_b -> bfloat128,
 *  v_bf16_st_tnsr). See build.sh for the exact, recovered stock build flags.
 */

/* tpc-clang auto-includes its builtin intrinsic header + tpc-special.h when
 * invoked with -mcpu=gaudi2 (the driver adds -tpc-special and the resource dir).
 * These provide the vector types (int64, float64, bfloat128, int5) and the
 * tensor load/store/convert builtins used below. No explicit #include needed. */

#ifndef SW_RHNE
#define SW_RHNE 0x0   /* round half to nearest even (default rounding mode id) */
#endif

/* User params struct — MUST byte-match FillParamsFn in moe_ext_custom_op.cpp. */
struct MoeUnpackDequantW4Params {
    int group_size;   /* rows-per-group along IN (128 for GLM); informational —
                         the group addressing is realized via the scale/zero
                         tensor geometry, kept here for parity + AWQ phase. */
    int zero_const;   /* GPTQ-sym shared zero (8). Reserved; the glue passes the
                         zero as an f32 tensor so the body is branch-free. */
};

void main(tensor weight_packed,   /* in0: u4  [in, out]  (packed int32 on disk)   */
          tensor weight_scale,    /* in1: f32 [groups, out]                       */
          tensor zero_point,      /* in2: f32 [groups, out] (broadcast const 8)   */
          tensor out0)            /* out: bf16 [in, out]                          */
{
    const int5 start = get_index_space_offset();
    const int5 size  = get_index_space_size();

    /* dim0 = OUT (FCD, 64 f32 lanes / step), dim1 = IN (1 / step). */
    int5 idx = start;

    const int out0_lo = start[0];
    const int out0_hi = start[0] + size[0];
    const int in_lo   = start[1];
    const int in_hi   = start[1] + size[1];

    for (int in = in_lo; in < in_hi; in += 1) {
        idx[1] = in;
        for (int o = out0_lo; o < out0_hi; o += 1) {
            idx[0] = o;

            /* load 64 unpacked uint4 codes -> int32 lanes.
             * weight_packed is a U4-typed tensor, so the tnsr load yields the
             * unpacked nibble values (the descriptor handles the 8/int32 pack). */
            int64   code_i = v_i32_ld_tnsr_b(idx, weight_packed);
            float64 code_f = v_convert_i32_to_f32_b(code_i, SW_RHNE, 0, 1, 0);

            /* scale/zero are addressed at the same OUT FCD; the group row along
             * IN is selected by the tensor descriptor's access pattern (a=0 on
             * the IN dim for the group tensors => broadcast within a group). */
            float64 sc = v_f32_ld_tnsr_b(idx, weight_scale);
            float64 zr = v_f32_ld_tnsr_b(idx, zero_point);

            float64 dq = v_f32_sub_b(code_f, zr, 0, 0, 1, 0);   /* code - zero    */
            dq         = v_f32_mul_b(dq, sc, 0, 0, 1, 0);       /* * scale        */

            bfloat128 bf = v_convert_f32_to_bf16_b(dq, SW_RHNE, 0, 1, 0);
            v_bf16_st_tnsr(idx, out0, bf);
        }
    }
}
