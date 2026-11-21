import json, torch, sys
sys.path.insert(0,'/opt/hpu-moe-ext')
from safetensors import safe_open
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, unpack_quantized_values_into_int32)
from vllm.scalar_type import scalar_types

CK='/models/GLM-5.1-int4-mixed-AutoRound'
idx=json.load(open(CK+'/model.safetensors.index.json'))['weight_map']
def load(name):
    with safe_open(CK+'/'+idx[name],'pt') as sf: return sf.get_tensor(name)

# A single routed expert gate_proj, 4-bit. The FusedMoE loader stacks experts as
# uint8 [E, out, in//2] (per moe_wna16_hpu._repack_weight comment). But the raw ckpt
# stores per-expert gptq int32 [in//8, out]. Let's emulate: vllm's gptq moe loader
# converts the int32 gptq qweight [in//8,out] to uint8 [out, in//2] (transpose+byteview).
# We just verify the plugin _repack ROUND-TRIPS the int4 codes correctly.
qw = load('model.layers.5.mlp.experts.0.gate_proj.qweight')  # [in//8, out] int32, in=2048,out=2048
print('expert gate qweight', tuple(qw.shape), qw.dtype)  # [256, 2048]

# canonical unpack of the raw gptq int32 [in//8, out] -> codes [in, out]
wf=torch.tensor(list(range(0,32,4)),dtype=torch.int32)
qw64=qw.to(torch.int64)&0xFFFFFFFF
codes_ref=((qw64.unsqueeze(1)>>wf.view(1,-1,1).to(torch.int64))&0xF).reshape(qw.shape[0]*8, qw.shape[1])  # [in,out]
print('codes_ref [in,out]=',tuple(codes_ref.shape))

# Now emulate the vllm moe loader's uint8 layout [E=1, out, in//2]:
#   gptq int32 [in//8, out] -> the moe path stores as uint8. Per the plugin comment:
#   "uint8 [E, out, in//2] is the gptq int32 qweight [in//8, out] transposed (->[out, in//8]) and byte-viewed"
w_t = qw.transpose(0,1).contiguous()           # [out, in//8] int32
w_u8 = w_t.view(torch.uint8)                    # [out, in//2] uint8
E_stack = w_u8.unsqueeze(0)                     # [1, out, in//2]
print('emulated uint8 stack', tuple(E_stack.shape), E_stack.dtype)

# Run the EXACT plugin repack
qt = scalar_types.uint4
def plugin_repack(qweight_u8):
    outs=[]
    for e in range(qweight_u8.shape[0]):
        w_i32 = qweight_u8[e].contiguous().view(torch.int32)  # [out, in//8]
        vals = unpack_quantized_values_into_int32(w_i32.transpose(0,1).contiguous(), qt, packed_dim=0)  # [in,out]
        outs.append(pack_quantized_values_into_int32(vals, qt, packed_dim=1))  # [in, out//8]
    return torch.stack(outs,dim=0)
repacked = plugin_repack(E_stack)   # [1, in, out//8]
print('repacked', tuple(repacked.shape), repacked.dtype)

# Unpack the repacked [in, out//8] back to codes [in,out] and compare to codes_ref
rp = repacked[0]   # [in, out//8]
rp64 = rp.to(torch.int64)&0xFFFFFFFF
codes_rt = ((rp64.unsqueeze(-1)>>wf.view(1,1,-1).to(torch.int64))&0xF).reshape(rp.shape[0], rp.shape[1]*8)  # [in,out]
print('codes round-trip', tuple(codes_rt.shape))
diff=(codes_ref.to(torch.int64)-codes_rt).abs()
print('REPACK round-trip maxdiff (should be 0):', diff.max().item())
# sanity: also check the intermediate unpack vals == codes_ref
vals = unpack_quantized_values_into_int32(E_stack[0].contiguous().view(torch.int32).transpose(0,1).contiguous(), qt, packed_dim=0)
print('intermediate vals maxdiff vs codes_ref:', (vals.to(torch.int64)-codes_ref.to(torch.int64)).abs().max().item())
