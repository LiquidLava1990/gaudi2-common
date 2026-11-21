"""Path A end-to-end CPU harness for GPTQDequantToBF16LinearMethod.

Emulates create_weights -> vLLM loader -> process_weights_after_loading -> forward
for (1) a REAL attention q_proj (L3, genuinely int4 qweight) and
    (2) a shared_expert.gate_proj (PHANTOM .qweight in index, only .weight in file).

CPU-only. .to('hpu') is monkeypatched to no-op; htorch.mark_step no-op.
We reimplement the dequant in plain torch matching the plugin exactly.
"""
import os, json, sys
import torch
from safetensors import safe_open

M = "/m"
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]

# ---- shape probe from the GPTQHPULinearMethod.create_weights logic ----
# qweight param: [in//pack, out] int32 ; qzeros [groups, out//pack] int32 ; scales [groups, out]
# pack=8 for bits=4. group_size=128.
BITS = 4; GS = 128; PACK = 32 // BITS

def real_tensor(base, suf):
    k = base + suf
    if k not in idx: return None
    f = idx[k]
    with safe_open(f"{M}/{f}", "pt") as h:
        ks = set(h.keys())
        if k not in ks: return ("PHANTOM", ks)
        return h.get_tensor(k)

def file_keys(base):
    # gather every real tensor stored under this prefix (from whatever file)
    out = {}
    # the .weight (if any) may be in same file as the phantom .qweight points to
    cand_files = set()
    for suf in (".qweight",".qzeros",".scales",".weight"):
        if base+suf in idx: cand_files.add(idx[base+suf])
    for f in cand_files:
        with safe_open(f"{M}/{f}","pt") as h:
            for k in h.keys():
                if k.startswith(base+"."):
                    out[k[len(base):]] = h.get_tensor(k)
    return out

# ---- plugin dequant math (plain torch, mirrors GPTQDequantToBF16LinearMethod) ----
def unpack_weight(qweight, bits):
    wf = torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    w = torch.bitwise_right_shift(
        torch.unsqueeze(qweight,1).expand(-1,32//bits,-1),
        wf.unsqueeze(-1)).to(torch.int8)
    w = torch.bitwise_and(w,(2**bits)-1)
    return w.reshape((w.shape[0]*w.shape[1], w.shape[2]))  # [in,out]

def unpack_zeros(qzeros, bits, sdtype):
    wf = torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    z = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros,2).expand(-1,-1,32//bits),
        wf.unsqueeze(0)).to(torch.int8)
    z = z + 1
    z = torch.bitwise_and(z,(2**bits)-1).to(sdtype)
    return z.reshape(-1, z.shape[1]*z.shape[2])  # [groups,out]

def plugin_dequant(qweight, qzeros, scales, bits=BITS, gs=GS):
    codes = unpack_weight(qweight, bits).to(torch.float32)        # [in,out]
    zeros = unpack_zeros(qzeros, bits, torch.float32)             # [groups,out]
    sc = scales.to(torch.float32)                                 # [groups,out]
    in_dim = codes.shape[0]
    g_idx = (torch.arange(in_dim)//gs).long()
    z = zeros[g_idx]; s = sc[g_idx]
    w = (codes - z) * s                                           # [in,out]
    return w.t().contiguous()                                     # [out,in]

def report(name, w):
    finite = torch.isfinite(w).all().item()
    nnan = torch.isnan(w).sum().item()
    ninf = torch.isinf(w).sum().item()
    print(f"  [{name}] shape={tuple(w.shape)} finite={finite} nan={nnan} inf={ninf} "
          f"mean={w.float().mean().item():.5f} std={w.float().std().item():.5f} "
          f"absmax={w.float().abs().max().item():.4f}")
    return finite

print("="*70)
print("CASE 1: REAL attention q_proj L3 (genuinely int4)")
base = "model.language_model.layers.3.self_attn.q_proj"
qw = real_tensor(base,".qweight"); qz = real_tensor(base,".qzeros"); sc = real_tensor(base,".scales")
print(f"  qweight {tuple(qw.shape)} {qw.dtype}; qzeros {tuple(qz.shape)} {qz.dtype}; scales {tuple(sc.shape)} {sc.dtype}")
w = plugin_dequant(qw, qz, sc)
ok1 = report("dequant", w)
# forward
x = torch.randn(4, w.shape[1])
y = torch.nn.functional.linear(x, w.to(torch.bfloat16).float())
report("forward y", y)

print("="*70)
print("CASE 2: shared_expert.gate_proj L3 (PHANTOM .qweight; file has .weight)")
base = "model.language_model.layers.3.mlp.shared_expert.gate_proj"
qwr = real_tensor(base,".qweight")
print(f"  real_tensor('.qweight') -> {type(qwr).__name__ if not isinstance(qwr,tuple) else qwr[0]}")
fk = file_keys(base)
print(f"  ACTUAL stored tensors under prefix: { {k:(tuple(v.shape),str(v.dtype)) for k,v in fk.items()} }")

# What vLLM's loader does: create_weights registered qweight=empty[in//8,out],
# qzeros=empty[g,out//8], scales=empty[g,out]. The index lists .qweight/.qzeros/.scales
# -> loader opens the file, the tensor is ABSENT. Emulate the two plausible loader outcomes:
#   (a) param stays UNINITIALIZED (torch.empty garbage) -> dequant garbage
#   (b) loader errors (but symptom says NO errors, so (a) is what happens)
# Derive shapes from the real .weight: [out,in] = [1024,4096] -> in=4096,out=1024
W = fk[".weight"]; out_dim, in_dim = W.shape
groups = in_dim // GS
print(f"  derived in={in_dim} out={out_dim} groups={groups}")
# uninitialized params exactly as create_weights makes them:
torch.manual_seed(0)
qw_u = torch.empty(in_dim//PACK, out_dim, dtype=torch.int32)   # garbage
qz_u = torch.empty(groups, out_dim//PACK, dtype=torch.int32)   # garbage
sc_u = torch.empty(groups, out_dim, dtype=torch.float16)       # garbage
print("  -- dequant of UNINITIALIZED (torch.empty) params (loader silently leaves them): --")
try:
    wbad = plugin_dequant(qw_u, qz_u, sc_u)
    okbad = report("dequant(uninit)", wbad)
    xb = torch.randn(4, in_dim)
    yb = torch.nn.functional.linear(xb, wbad.to(torch.bfloat16).float())
    report("forward(uninit)", yb)
except Exception as e:
    print("  EXCEPTION:", repr(e)); okbad=None

# zero-init variant (some allocators / first-touch can give zeros):
qw_z = torch.zeros(in_dim//PACK, out_dim, dtype=torch.int32)
qz_z = torch.zeros(groups, out_dim//PACK, dtype=torch.int32)
sc_z = torch.zeros(groups, out_dim, dtype=torch.float16)
print("  -- dequant of ZERO-init params (if loader/allocator zeroes): --")
wz = plugin_dequant(qw_z, qz_z, sc_z)
report("dequant(zero)", wz)
# zero qweight->codes=0, zeros+1 ->1, (0-1)*0 = 0 -> weight all zero -> forward all zero (lost residual!)

print("="*70)
print("CASE 3: what the CORRECT shared_expert output should be (use real .weight)")
report("real .weight", W)
xg = torch.randn(4, in_dim)
yg = torch.nn.functional.linear(xg, W.float())
report("forward(correct)", yg)
