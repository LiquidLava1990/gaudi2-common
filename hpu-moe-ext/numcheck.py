import torch, sys
from safetensors import safe_open

CK='/models/GLM-5.1-int4-mixed-AutoRound'
def load(layer_path, fileno):
    f=f'{CK}/model-{fileno:05d}-of-00075.safetensors'
    with safe_open(f,'pt') as sf:
        qw=sf.get_tensor(layer_path+'.qweight')
        qz=sf.get_tensor(layer_path+'.qzeros')
        sc=sf.get_tensor(layer_path+'.scales')
    return qw,qz,sc

# ---------- canonical auto_gptq / auto_round SYM dequant reference ----------
def ref_dequant(qw, qz, sc, bits, group_size):
    # qw: [in//pack, out] int32 ; qz: [groups, out//pack] int32 ; sc: [groups, out]
    pack = 32//bits
    # unpack weight codes -> [in, out]
    wf = torch.tensor([i for i in range(0,32,bits)], dtype=torch.int32)
    qw_i = qw.to(torch.int32)
    # treat as unsigned: use logical shift via int64 to avoid sign issues
    qw64 = qw_i.to(torch.int64) & 0xFFFFFFFF
    codes = (qw64.unsqueeze(1) >> wf.view(1,-1,1).to(torch.int64)) & ((1<<bits)-1)  # [in//pack, pack, out]
    codes = codes.reshape(qw.shape[0]*pack, qw.shape[1]).to(torch.float64)          # [in,out]
    # unpack zeros -> [groups, out], canonical autogptq: stored zero, then +1
    qz64 = qz.to(torch.int64) & 0xFFFFFFFF
    z = (qz64.unsqueeze(-1) >> wf.view(1,1,-1).to(torch.int64)) & ((1<<bits)-1)      # [groups, out//pack, pack]
    z = z.reshape(qz.shape[0], qz.shape[1]*pack).to(torch.float64)                   # [groups,out]
    z = z + 1.0                                                                      # autogptq +1
    z = z % (1<<bits)                                                                # wrap
    sc = sc.to(torch.float64)
    g_idx = (torch.arange(codes.shape[0])//group_size).long()
    zexp = z[g_idx]; sexp = sc[g_idx]
    w = (codes - zexp) * sexp   # [in,out]
    return w  # dense, [in,out]

# ---------- replicate the PLUGIN unpackers exactly (hpu_gptq) ----------
class FakeCfg:
    def __init__(s,bits,gs): s.weight_bits=bits; s.group_size=gs
class FakeLayer: pass

def plugin_unpack_weight(qweight, bits):
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight,1).expand(-1,32//bits,-1),
        wf.unsqueeze(-1),
    ).to(torch.int16 if bits==8 else torch.int8)
    weight = torch.bitwise_and(weight,(2**bits)-1)
    weight = weight.reshape((weight.shape[0]*weight.shape[1],weight.shape[2]))
    return weight

def plugin_unpack_zeros(qzeros, scales_dtype, bits):
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros,2).expand(-1,-1,32//bits),
        wf.unsqueeze(0),
    ).to(torch.int16 if bits==8 else torch.int8)
    zeros = zeros + 1
    zeros = torch.bitwise_and(zeros,(2**bits)-1).to(scales_dtype)
    zeros = zeros.reshape(-1, zeros.shape[1]*zeros.shape[2])
    return zeros

def plugin_dequant(qw,qz,sc,bits,gs):
    codes = plugin_unpack_weight(qw, bits).to(torch.float32)         # [in,out]
    zeros = plugin_unpack_zeros(qz, sc.dtype, bits).to(torch.float32) # [groups,out]
    scales = sc.to(torch.float32)
    g_idx = (torch.arange(codes.shape[0])//gs).long()
    z=zeros[g_idx]; s=scales[g_idx]
    w=(codes-z)*s
    return w   # [in,out]  (plugin then does w.t() -> [out,in]; we compare pre-transpose)

def test(name, layer_path, fileno, bits):
    qw,qz,sc=load(layer_path, fileno)
    gs=128
    ref=ref_dequant(qw,qz,sc,bits,gs)
    plug=plugin_dequant(qw,qz,sc,bits,gs).to(torch.float64)
    diff=(ref-plug).abs()
    print(f'[{name}] bits={bits} shapes qw{tuple(qw.shape)} ref{tuple(ref.shape)}')
    print(f'   maxdiff={diff.max().item():.3e}  meandiff={diff.mean().item():.3e}  ref_absmax={ref.abs().max().item():.4f}')
    # also report what dequant looks like (sanity: should be ~N(0,small))
    print(f'   ref dequant mean={ref.mean().item():.5f} std={ref.std().item():.5f}')
    return diff.max().item()

print('========== 8-bit attn projections ==========')
test('q_a_proj L3', 'model.layers.3.self_attn.q_a_proj', 1, 8)
test('o_proj L3',   'model.layers.3.self_attn.o_proj', 1, 8)
test('kv_b_proj L3','model.layers.3.self_attn.kv_b_proj', 1, 8)
print('========== 4-bit expert (control: same as Qwen path conceptually) ==========')
test('expert gate L5','model.layers.5.mlp.experts.0.gate_proj', 3, 4)
