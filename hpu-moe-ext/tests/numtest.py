import torch
import habana_frameworks.torch.core as htorch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types
qt = scalar_types.uint4
dev = 'hpu'
torch.manual_seed(0)
IN, OUT, G = 256, 64, 128         # 2 groups along input
q = torch.randint(0, 16, (IN, OUT), dtype=torch.int32)     # int4 values [in,out]

# emulate gptq checkpoint packing (along input) then upstream loader .T.view(uint8)
gptq_qweight = pack_quantized_values_into_int32(q, qt, packed_dim=0)   # [in//8, out] int32
u8 = gptq_qweight.T.contiguous().view(torch.uint8)                     # [out, in//2] uint8  (== loaded param)

# ---- MY plugin repack ----
w_i32 = u8.contiguous().view(torch.int32)                              # [out, in//8]
vals = unpack_quantized_values_into_int32(w_i32.transpose(0,1).contiguous(), qt, packed_dim=0)  # [in,out]
print("vals==q round-trip:", torch.equal(vals, q))
packed = pack_quantized_values_into_int32(vals, qt, packed_dim=1)      # [in, out//8]

scale = (torch.rand(IN//G, OUT, dtype=torch.bfloat16) + 0.1)           # [groups, out]
zeros = torch.full((IN//G, OUT), 8, dtype=torch.int32)
zeros_p = pack_quantized_values_into_int32(zeros, qt, packed_dim=1)    # [groups, out//8]

deq = torch.ops.hpu.convert_from_uint4(packed.to(dev), scale.to(dev), zeros_p.to(dev),
                                       scale.to(dev).dtype, None).cpu()
htorch.mark_step()
scale_full = scale.repeat_interleave(G, dim=0).float()                 # [in,out]
ref = scale_full * (q.float() - 8.0)
print("deq.shape", tuple(deq.shape), "ref.shape", tuple(ref.shape))
if deq.shape == ref.shape:
    print("max abs diff:", (deq.float()-ref).abs().max().item())
    print("MATCH:", torch.allclose(deq.float(), ref, atol=2e-2))
else:
    print("SHAPE MISMATCH -> deq transposed? deq==ref.T:",
          deq.shape==ref.T.shape and torch.allclose(deq.float(), ref.T, atol=2e-2))
