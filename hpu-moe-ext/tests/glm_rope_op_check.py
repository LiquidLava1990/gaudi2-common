import json, torch
import torch.nn.functional as F
torch.set_grad_enabled(False)
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'
cfg=json.load(open(f'{CKPT}/config.json'))
qk_rope=cfg['qk_rope_head_dim']; rope_theta=cfg['rope_parameters']['rope_theta']

def rotate_half(x):
    x1=x[..., :x.shape[-1]//2]; x2=x[..., x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)
def build_rope(pos, dim):
    inv=1.0/(rope_theta**(torch.arange(0,dim,2).float()/dim))
    f=torch.einsum('s,d->sd', pos.float(), inv); emb=torch.cat((f,f),-1)
    return emb.cos(), emb.sin()

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope as vget_rope
vc=VllmConfig()
with set_current_vllm_config(vc):
    vrope=vget_rope(qk_rope, max_position=cfg['max_position_embeddings'],
                    rope_parameters={'rope_type':'default','rope_theta':rope_theta}, is_neox_style=True)
    pos=torch.tensor([0,1,2,3,4])
    torch.manual_seed(0)
    H=64
    q=torch.randn(5,H,qk_rope); kpe=torch.randn(5,1,qk_rope)
    qv=q.clone(); kv_=kpe.clone()
    qo,ko=vrope(pos, qv, kv_)
    c,s=build_rope(pos,qk_rope)            # [5,64]
    c=c.unsqueeze(1); s=s.unsqueeze(1)     # [5,1,64]
    qref=q*c+rotate_half(q)*s
    kref=kpe*c+rotate_half(kpe)*s
    def cs(a,b):
        a=a.reshape(-1).double(); b=b.reshape(-1).double()
        return (a@b/(a.norm()*b.norm())).item()
    print('is_neox_style:', getattr(vrope,'is_neox_style',None))
    print('vLLM-rope q vs HF-NeoX q: maxdiff=%.3e cos=%.8f'%((qo-qref).abs().max().item(), cs(qo,qref)))
    print('vLLM-rope k vs HF-NeoX k: maxdiff=%.3e cos=%.8f'%((ko-kref).abs().max().item(), cs(ko,kref)))
    # Also test INTERLEAVED to confirm it would differ
    vrope_i=vget_rope(qk_rope, max_position=cfg['max_position_embeddings'],
                      rope_parameters={'rope_type':'default','rope_theta':rope_theta}, is_neox_style=False)
    qv2=q.clone(); kv2=kpe.clone()
    qo2,ko2=vrope_i(pos,qv2,kv2)
    print('INTERLEAVED vLLM q vs HF-NeoX q: cos=%.6f (should be LOW if neox matters)'%cs(qo2,qref))
