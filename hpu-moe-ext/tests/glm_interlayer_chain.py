"""
Inter-layer wiring check. Layers 3 and 4 are consecutive in the dumps.
The DeepseekV2 fused-residual scheme: layer N returns (hidden_out, residual_out);
layer N+1 receives hidden_in=hidden_out_N, residual_in=residual_out_N.
Check that L4's dumped (hidden_in, residual_in) == L3's dumped (hidden_out, residual_out).
This catches a wiring/ordering bug (e.g. swapped hidden/residual, missing add).
"""
import torch
torch.set_grad_enabled(False)
DT = torch.float64

def load(L):
    d = torch.load(f'/opt/hpu-dsa-ext/layer{L}_dump.pt', map_location='cpu', weights_only=False)
    def fix(t):
        if t is None: return None
        t = t.to(DT)
        return t[0] if t.dim() == 3 else t
    return dict(hin=fix(d['hidden_in']), rin=fix(d['residual_in']),
                hout=fix(d['hidden_out']), rout=fix(d['residual_out']))

def cs(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

l3 = load(3); l4 = load(4)
S = 5
print('Check: L4.hidden_in == L3.hidden_out ?')
print('  cos/pos:', [round(cs(l4['hin'][i], l3['hout'][i]), 5) for i in range(S)])
print('  |L4.hin|', [round(l4['hin'][i].norm().item(), 2) for i in range(S)])
print('  |L3.hout|', [round(l3['hout'][i].norm().item(), 2) for i in range(S)])
print('Check: L4.residual_in == L3.residual_out ?')
print('  cos/pos:', [round(cs(l4['rin'][i], l3['rout'][i]), 5) for i in range(S)])
print('  |L4.rin|', [round(l4['rin'][i].norm().item(), 2) for i in range(S)])
print('  |L3.rout|', [round(l3['rout'][i].norm().item(), 2) for i in range(S)])
# also: maybe layer feeds hidden_in = hidden_out + residual_out merged?
print('\nAlt: L4.hidden_in == L3.(hout+rout)? cos:',
      [round(cs(l4['hin'][i], (l3['hout']+l3['rout'])[i]), 5) for i in range(S)])
print('Alt: L4.residual_in == L3.hout? cos:',
      [round(cs(l4['rin'][i], l3['hout'][i]), 5) for i in range(S)])
