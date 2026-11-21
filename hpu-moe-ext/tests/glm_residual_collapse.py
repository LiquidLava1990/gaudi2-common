"""
Is the final residual direction position-collapsed (all positions ~parallel),
or does it carry per-position signal? And does a HUGE shared component dominate?
Check inter-position cosine of the final hidden + after model.norm, and the
component structure. Also test: subtract the mean-over-positions and re-run lm_head
(if the per-position RESIDUAL is right but a giant common-mode vector swamps it,
removing the common mode would reveal Paris).
"""
import json, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
DT = torch.float64
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
cfg = json.load(open(f'{CKPT}/config.json'))
eps = cfg['rms_norm_eps']
idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}', framework='pt', device='cpu') as f:
        return f.get_tensor(n)
d = torch.load('/opt/hpu-dsa-ext/layer77_dump.pt', map_location='cpu', weights_only=False)
hout = d['hidden_out']; hout = (hout[0] if hout.dim() == 3 else hout).to(DT)
rout = d['residual_out']; rout = (rout[0] if rout.dim() == 3 else rout).to(DT)
S = 5
fh = (rout[:S] + hout[:S])   # [S,6144]
def cs(a, b):
    a = a.reshape(-1); b = b.reshape(-1); return (a @ b / (a.norm() * b.norm() + 1e-30)).item()
print('inter-position cosine of final hidden:')
for i in range(S):
    print('  pos%d:' % i, [round(cs(fh[i], fh[j]), 4) for j in range(S)])
mean = fh.mean(0)
print('\n|mean-over-pos| = %.2f ; per-pos |fh| = %s' % (mean.norm().item(), [round(fh[i].norm().item(),1) for i in range(S)]))
print('cos(fh[i], mean):', [round(cs(fh[i], mean), 4) for i in range(S)])
resid = fh - mean[None]
print('|fh - mean| per pos:', [round(resid[i].norm().item(), 2) for i in range(S)])

norm_w = get('model.norm.weight').to(DT)
lm_head = get('lm_head.weight').to(DT)
def rmsnorm(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
prompt_ids = [785, 6722, 315, 9621, 374]

# does the embedding tokens themselves dominate the residual? (residual stream = sum of embed + all layer deltas)
embed = get('model.embed_tokens.weight').to(DT)
print('\nembed row norms for prompt tokens:', [round(embed[t].norm().item(),3) for t in prompt_ids])
# cosine of final hidden with its OWN token embedding (residual should retain some)
print('cos(fh[i], embed[token_i]):', [round(cs(fh[i], embed[prompt_ids[i]]), 4) for i in range(S)])

# Try lm_head on the common-mode-removed residual
for label, x in [('common-mode removed', resid)]:
    normed = rmsnorm(x, norm_w)
    logits = normed @ lm_head.t()
    print(f'\n=== {label} ===')
    for i in range(S):
        top = torch.topk(logits[i], 6)
        toks = [tok.decode([int(t)]) for t in top.indices]
        print(f'  pos{i} (after {tok.decode([prompt_ids[i]])!r}) argmax-top6: {toks}')

# Sanity: feed embed[' Paris'] guess - what does correct hidden need to look like?
# Find ' Paris' token, check its lm_head row norm and whether ANY of our positions point near it
paris_ids = tok(' Paris', add_special_tokens=False)['input_ids']
print('\n" Paris" token ids:', paris_ids, [tok.decode([i]) for i in paris_ids])
pid = paris_ids[0]
print('cos(normed_fh[4], lm_head[Paris]):', round(cs(rmsnorm(fh[4:5], norm_w)[0], lm_head[pid]), 5))
print('lm_head[Paris] logit at pos4 (as-stored norm):', round((rmsnorm(fh[4:5], norm_w)[0] @ lm_head[pid]).item(), 3))
# rank of Paris among all logits at pos4
lg4 = rmsnorm(fh[4:5], norm_w)[0] @ lm_head.t()
rank = (lg4 > lg4[pid]).sum().item()
print('Paris logit rank at pos4 (0=top):', rank, 'of', lg4.numel())
