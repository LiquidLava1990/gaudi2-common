import json,glob,torch
from safetensors import safe_open
M='/mnt/fastmodels/Qwen3.5-35B-A3B-GPTQ-Int4'
wm=json.load(open(glob.glob(M+'/*.index.json')[0]))['weight_map']
def ld(n):
    with safe_open(M+'/'+wm[n],'pt') as s: return s.get_tensor(n)
for proj in ['gate_proj','down_proj']:
    p=f'model.language_model.layers.0.mlp.experts.0.{proj}'
    gi=ld(p+'.g_idx').to(torch.int64)
    IN=gi.shape[0]; G=128
    seq=torch.arange(IN)//G
    is_seq=torch.equal(gi,seq)
    print(f'{proj}: g_idx len={IN} is_sequential(i//128)={is_seq}')
    if not is_seq:
        print(f'  g_idx[:20]={gi[:20].tolist()}')
        print(f'  seq[:20]  ={seq[:20].tolist()}')
        print(f'  g_idx[126:134]={gi[126:134].tolist()}  (boundary)')
