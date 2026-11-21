"""
Verify the o_proj weight as vLLM's auto_round/gptq quant method dequants it (per-rank, TP=8)
equals the checkpoint reference dequant.  Pure CPU (no HPU): we replay the unpacking math
that the GPTQ/auto_round path uses and the RowParallelLinear input-sharding.

If the reassembled full weight != reference dequant, the bug is a weight-sharding/dequant bug
on the row-parallel o_proj.  If it matches, the weight is fine and the bug is the runtime
all-reduce / gather.
"""
import json, torch
from safetensors import safe_open
torch.set_grad_enabled(False)
DT=torch.float64
CKPT='/models/GLM-5.1-int4-mixed-AutoRound'
idx=json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
def get(n):
    with safe_open(f'{CKPT}/{idx[n]}',framework='pt',device='cpu') as f: return f.get_tensor(n)

P='model.layers.0.self_attn.o_proj'
qweight=get(P+'.qweight')   # [4096,6144] int32  packed along INPUT (32/8=4 per int32) -> 16384 in
qzeros =get(P+'.qzeros')    # [128,1536] int32   packed along OUTPUT -> 6144 out, 128 groups
scales =get(P+'.scales')    # [128,6144] fp16     128 groups (input/128), 6144 out
bits=8; gs=128
IN=qweight.shape[0]*(32//bits)   # 16384
OUT=qweight.shape[1]             # 6144
groups=scales.shape[0]           # 128
print('IN=%d OUT=%d groups=%d (IN/gs=%d)'%(IN,OUT,groups,IN//gs))

# ---- reference dequant (my dequant8, input-major) ----
def dequant8_ref():
    wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
    c=torch.bitwise_right_shift(torch.unsqueeze(qweight,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int16)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT); c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])  # [16384,6144]
    rz=torch.bitwise_right_shift(torch.unsqueeze(qzeros,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int16)
    rz=torch.bitwise_and(rz+1,(2**bits)-1).to(DT); z=rz.reshape(-1,rz.shape[1]*rz.shape[2])  # [128,6144]
    scf=scales.to(DT); g=(torch.arange(c.shape[0])//gs).long(); return (c-z[g])*scf[g]
w_ref=dequant8_ref()   # [16384,6144]
print('w_ref fro=%.4f'%w_ref.norm().item())

# ---- simulate RowParallelLinear TP=8 input-sharding of the PACKED tensors ----
# RowParallel splits the INPUT dim. qweight packs 4 input-codes per int32 row, so input row i
# lives in qweight[i//4] nibble (i%4). Splitting input dim contiguously [r*2048:(r+1)*2048] maps
# to qweight rows [r*512:(r+1)*512] (since 2048/4=512). scales split [r*16:(r+1)*16] (2048/128=16).
# qzeros are OUTPUT-packed and NOT sharded by row-parallel input split -> each rank keeps full qzeros.
TP=8; in_per=IN//TP            # 2048
qw_rows_per=qweight.shape[0]//TP  # 512
grp_per=in_per//gs             # 16
wf=torch.tensor(list(range(0,32,bits)),dtype=torch.int32).unsqueeze(0)
def rank_dequant(r):
    qw_r=qweight[r*qw_rows_per:(r+1)*qw_rows_per]         # [512,6144]
    sc_r=scales[r*grp_per:(r+1)*grp_per]                  # [16,6144]
    qz_r=qzeros                                           # full [128,1536] (output-packed, not input-sharded)
    c=torch.bitwise_right_shift(torch.unsqueeze(qw_r,1).expand(-1,32//bits,-1),wf.unsqueeze(-1)).to(torch.int16)
    c=torch.bitwise_and(c,(2**bits)-1).to(DT); c=c.reshape(c.shape[0]*c.shape[1],c.shape[2])  # [2048,6144]
    rz=torch.bitwise_right_shift(torch.unsqueeze(qz_r,2).expand(-1,-1,32//bits),wf.unsqueeze(0)).to(torch.int16)
    rz=torch.bitwise_and(rz+1,(2**bits)-1).to(DT); z=rz.reshape(-1,rz.shape[1]*rz.shape[2])   # [128,6144]
    # this rank's groups are global groups [r*16:(r+1)*16]; zeros indexed by GLOBAL group
    g=(torch.arange(c.shape[0])//gs).long() + r*grp_per
    return (c - z[g]) * sc_r[(torch.arange(c.shape[0])//gs).long()]
recon=torch.cat([rank_dequant(r) for r in range(TP)],0)   # [16384,6144]
print('recon (per-rank-sharded dequant) vs w_ref: maxdiff=%.3e  cos=%.8f'
      %((recon-w_ref).abs().max().item(),
        (recon.reshape(-1)@w_ref.reshape(-1)/(recon.norm()*w_ref.norm())).item()))
print('=> if maxdiff~0, the row-parallel sharded dequant of o_proj weight is CORRECT.')
