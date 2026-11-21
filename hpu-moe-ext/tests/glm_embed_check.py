"""
Is layer-0's dumped input == the clean embedding lookup for the prompt tokens?
The embed-NaN 'sanitize' (nan_to_num) could be ZEROING a real element or otherwise
corrupting the embedding. Compare L0.hidden_in[:5] to embed_tokens[prompt_ids] * scale.
GLM may scale embeddings by sqrt(hidden) or 1.0 - check both.
"""
import json, torch
from safetensors import safe_open
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
cfg = json.load(open(f'{CKPT}/config.json'))
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}', framework='pt', device='cpu') as f:
        return f.get_tensor(n)
def cs(a, b):
    a = a.reshape(-1).to(DT); b = b.reshape(-1).to(DT); return (a @ b / (a.norm() * b.norm() + 1e-30)).item()
d = torch.load('/opt/hpu-dsa-ext/layer0_dump.pt', map_location='cpu', weights_only=False)
hin = d['hidden_in']; hin = (hin[0] if hin.dim()==3 else hin).to(DT)
prompt_ids = [785, 6722, 315, 9621, 374]
embed = get('model.embed_tokens.weight').to(DT)
print('embed shape', tuple(embed.shape))
print('L0.hidden_in[:5] norms:', [round(hin[i].norm().item(), 4) for i in range(5)])
print('embed[prompt] norms   :', [round(embed[t].norm().item(), 4) for i,t in enumerate(prompt_ids)])
for scale_name, sc in [('1.0', 1.0), ('sqrt(H)', cfg['hidden_size']**0.5)]:
    print(f'\n--- scale={scale_name} ({sc:.3f}) ---')
    for i, t in enumerate(prompt_ids):
        ref = embed[t] * sc
        c = cs(hin[i], ref)
        r = (hin[i].norm() / ref.norm()).item()
        print(f'  pos{i} tok{t}: cos={c:.5f} |hin|/|ref|={r:.4f}')
# any NaN/zero rows in hidden_in?
print('\nhidden_in nan count:', torch.isnan(hin.float()).sum().item(), ' exact-zero elements:', (hin==0).sum().item())
# implied scale from ratio
print('implied scale (|hin|/|embed|):', [round((hin[i].norm()/embed[prompt_ids[i]].norm()).item(),4) for i in range(5)])
