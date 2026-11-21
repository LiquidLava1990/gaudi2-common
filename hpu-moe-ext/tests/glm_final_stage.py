"""
FINAL-STAGE check: layer-77 output -> final model.norm -> lm_head -> logits.
Tests both model.norm as-stored and +1, with/without logit scaling.
Reports argmax token for the last real position (4 = ' is' -> should be ' Paris').
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
# final hidden = layer-77 full output = residual_out + hidden_out (delta)
final_hidden = (rout[:S] + hout[:S])
print('final_hidden norms/pos:', [round(final_hidden[i].norm().item(), 2) for i in range(S)])

norm_w = get('model.norm.weight').to(DT)
print('model.norm.weight stats: mean=%.4f rms=%.4f min=%.4f max=%.4f' %
      (norm_w.mean().item(), norm_w.pow(2).mean().sqrt().item(), norm_w.min().item(), norm_w.max().item()))
lm_head = get('lm_head.weight').to(DT)   # [vocab, hidden]
print('lm_head shape', tuple(lm_head.shape))

def rmsnorm(x, w, plus1):
    ww = (w + 1.0) if plus1 else w
    return ww * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

prompt_ids = [785, 6722, 315, 9621, 374]
for plus1 in [False, True]:
    normed = rmsnorm(final_hidden, norm_w, plus1)
    logits = normed @ lm_head.t()    # [S, vocab]
    print(f'\n=== model.norm {"+1" if plus1 else "as-stored"} ===')
    print('  normed rms/pos:', [round(normed[i].pow(2).mean().sqrt().item(), 4) for i in range(S)])
    print('  logit absmax/pos:', [round(logits[i].abs().max().item(), 2) for i in range(S)])
    for i in range(S):
        lg = logits[i]
        probs = torch.softmax(lg.float(), -1)
        top = torch.topk(lg, 8)
        toks = [tok.decode([int(t)]) for t in top.indices]
        ps = [round(probs[int(t)].item(), 4) for t in top.indices]
        ctx = tok.decode([prompt_ids[i]])
        print(f'  pos{i} (after {ctx!r}): top8={list(zip(toks, ps))}')
    # specifically the last real token's prediction
    print('  --> LAST-TOKEN (pos4, after \" is\") argmax:', repr(tok.decode([int(logits[4].argmax())])))

# check whether the model uses a logit_scale / softcap
print('\nconfig fields that could scale logits:')
for k in ['logits_scaling', 'logit_scale', 'final_logit_softcapping', 'attn_logit_softcapping', 'scale_emb', 'dim_model_base']:
    if k in cfg: print('  ', k, '=', cfg[k])
