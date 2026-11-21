import json, torch, sys
sys.path.insert(0,'/opt/hpu-moe-ext')
from safetensors import safe_open
CK='/models/GLM-5.1-int4-mixed-AutoRound'
idx=json.load(open(CK+'/model.safetensors.index.json'))['weight_map']
def load(name):
    with safe_open(CK+'/'+idx[name],'pt') as sf: return sf.get_tensor(name)
L='model.layers.3.self_attn'
qa_w=load(f'{L}.q_a_proj.qweight'); qa_z=load(f'{L}.q_a_proj.qzeros'); qa_s=load(f'{L}.q_a_proj.scales')
kv_w=load(f'{L}.kv_a_proj_with_mqa.qweight'); kv_z=load(f'{L}.kv_a_proj_with_mqa.qzeros'); kv_s=load(f'{L}.kv_a_proj_with_mqa.scales')

# Build the FUSED packed tensors the way MergedColumnParallelLinear places shards:
# qweight [in//4, out] -> concat along dim1 (output)
# scales  [g, out]     -> concat along dim1
# qzeros  [g, out//4]  -> concat along dim1 (packed output)
f_w=torch.cat([qa_w, kv_w], dim=1)   # [1536, 2624]
f_s=torch.cat([qa_s, kv_s], dim=1)   # [48, 2624]
f_z=torch.cat([qa_z, kv_z], dim=1)   # [48, 656]
print('fused shapes qweight',tuple(f_w.shape),'qzeros',tuple(f_z.shape),'scales',tuple(f_s.shape))

# plugin unpackers on the FUSED tensors
from vllm_gaudi.ops.hpu_gptq import GPTQHPULinearMethod, GPTQHPUConfig
class FL: pass
fl=FL(); fl.qweight=f_w; fl.qzeros=f_z; fl.scales=f_s
bits=8
cfg=GPTQHPUConfig(weight_bits=8, group_size=128, desc_act=False, lm_head_quantized=False)
m=GPTQHPULinearMethod(cfg)
m.wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
codes=m.unpack_weight_from_cuda_old_format(fl).to(torch.float64)
zeros=m.unpack_zeros_from_cuda_old_format(fl).to(torch.float64)
sc=fl.scales.to(torch.float64)
g=(torch.arange(codes.shape[0])//128).long()
plug_fused=((codes-zeros[g])*sc[g])   # [in,out]

# reference: dequant each shard separately, concat output
def deq(qw,qz,sc):
    pack=4; wf=torch.tensor([0,8,16,24],dtype=torch.int32)
    qw64=qw.to(torch.int64)&0xFFFFFFFF
    cc=((qw64.unsqueeze(1)>>wf.view(1,-1,1).to(torch.int64))&0xFF).reshape(qw.shape[0]*pack,qw.shape[1]).double()
    qz64=qz.to(torch.int64)&0xFFFFFFFF
    zz=((qz64.unsqueeze(-1)>>wf.view(1,1,-1).to(torch.int64))&0xFF).reshape(qz.shape[0],qz.shape[1]*pack).double()+1.0
    gg=(torch.arange(cc.shape[0])//128).long()
    return (cc-zz[gg])*sc.double()[gg]
ref=torch.cat([deq(qa_w,qa_z,qa_s), deq(kv_w,kv_z,kv_s)],dim=1)  # [in,out]
diff=(ref-plug_fused).abs()
print(f'FUSED-assembly maxdiff={diff.max().item():.3e} (in,out) ref_absmax={ref.abs().max().item():.4f}')
print('  q_a cols maxdiff:', (ref[:,:2048]-plug_fused[:,:2048]).abs().max().item())
print('  kv_a cols maxdiff:', (ref[:,2048:]-plug_fused[:,2048:]).abs().max().item())

# CRITICAL extra check: does the kv_a shard's qzeros pack offset (576/4=144) land correctly?
# The fused qzeros for kv_a region is columns 512:656 (packed). Decode them.
zfull=zeros  # [48, 2624] unpacked
print('  kv_a zero region uniq:', torch.unique(zfull[:,2048:]).tolist()[:5], '(should be 128)')
print('  q_a  zero region uniq:', torch.unique(zfull[:,:2048]).tolist()[:5], '(should be 128)')
