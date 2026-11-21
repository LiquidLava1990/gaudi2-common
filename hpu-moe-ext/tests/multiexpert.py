import json,glob,torch
from safetensors import safe_open
M='/mnt/fastmodels/Qwen3.5-35B-A3B-GPTQ-Int4'
wm=json.load(open(glob.glob(M+'/*.index.json')[0]))['weight_map']
def load(n):
    with safe_open(M+'/'+wm[n],'pt') as s: return s.get_tensor(n)
def u0(w):
    n8,m=w.shape;o=torch.empty(n8*8,m,dtype=torch.int32)
    for k in range(8):o[k::8]=(w>>(4*k))&0xF
    return o
def u1(w):
    n,m8=w.shape;o=torch.empty(n,m8*8,dtype=torch.int32)
    for k in range(8):o[:,k::8]=(w>>(4*k))&0xF
    return o
def deq(layer,exp,proj):
    p=f'model.language_model.layers.{layer}.mlp.experts.{exp}.{proj}'
    qw=load(p+'.qweight').to(torch.int32);sc=load(p+'.scales').float();qz=load(p+'.qzeros').to(torch.int32)
    IN=qw.shape[0]*8;G=IN//sc.shape[0]
    return sc.repeat_interleave(G,0)*(u0(qw).float()-(u1(qz)+1).float().repeat_interleave(G,0))
for exp in [0,5,100,255]:
    g=deq(0,exp,'gate_proj');u=deq(0,exp,'up_proj');w13=torch.cat([g,u],1)
    print(f'L0 expert {exp}: w13 absmean={w13.abs().mean():.5f} min={w13.min():.3f} max={w13.max():.3f}')
