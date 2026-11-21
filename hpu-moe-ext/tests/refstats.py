import json,glob,torch
from safetensors import safe_open
M='/mnt/fastmodels/Qwen3.5-35B-A3B-GPTQ-Int4'
wm=json.load(open(glob.glob(M+'/*.index.json')[0]))['weight_map']
def load(n):
    with safe_open(M+'/'+wm[n],'pt') as s: return s.get_tensor(n)
def upk_d0(w):
    n8,m=w.shape;o=torch.empty(n8*8,m,dtype=torch.int32)
    for k in range(8):o[k::8]=(w>>(4*k))&0xF
    return o
def upk_d1(w):
    n,m8=w.shape;o=torch.empty(n,m8*8,dtype=torch.int32)
    for k in range(8):o[:,k::8]=(w>>(4*k))&0xF
    return o
def deq(proj):
    p=f'model.language_model.layers.0.mlp.experts.0.{proj}'
    qw=load(p+'.qweight').to(torch.int32);sc=load(p+'.scales').float();qz=load(p+'.qzeros').to(torch.int32)
    IN=qw.shape[0]*8;G=IN//sc.shape[0]
    q=upk_d0(qw);z=upk_d1(qz)
    return sc.repeat_interleave(G,0)*(q.float()-(z+1).float().repeat_interleave(G,0))  # [in,out]
g=deq('gate_proj');u=deq('up_proj')                 # each [2048,512]
w13=torch.cat([g,u],dim=1)                           # [2048,1024] (gate|up) = engine [in,out]
print(f'REF w13(gate|up): shape{tuple(w13.shape)} absmean={w13.abs().mean():.4f} min={w13.min():.3f} max={w13.max():.3f}')
print(f'  gate absmean={g.abs().mean():.4f} up absmean={u.abs().mean():.4f}')
print('ENGINE reported: shape(2048,1024) absmean=0.0036 min=-0.100 max=0.072')
