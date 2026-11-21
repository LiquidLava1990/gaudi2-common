import torch, torch.nn.functional as F, glm52_parallel_block_drift as M
from transformers import AutoConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as Mod
torch.set_grad_enabled(False); DT = torch.float64
cfg = AutoConfig.from_pretrained(M.CKPT, trust_remote_code=True)
def c(a,b): a=a.flatten();b=b.flatten();return float((a@b)/(a.norm()*b.norm()+1e-30))
S=5; torch.manual_seed(2); x=torch.randn(1,S,M.H,dtype=DT)*0.3

# ---- DENSE MLP (layer 0) vs HF GlmMoeDsaMLP ----
mlp0 = Mod.GlmMoeDsaMLP(cfg).double().eval()
P0='model.layers.0.mlp'
mlp0.load_state_dict({f'{n}.weight': M.get(f'{P0}.{n}.weight').double() for n in ['gate_proj','up_proj','down_proj']})
out_hf0 = mlp0(x)[0]
out_me0 = M.mlp(0, x[0])
print('DENSE mlp L0 vs HF: cos=%.6f nrat=%.4f' % (c(out_hf0,out_me0), out_me0.norm().item()/out_hf0.norm().item()))

# ---- MoE (layer 3) vs HF GlmMoeDsaMoE ----
moe = Mod.GlmMoeDsaMoE(cfg).double().eval()
P3='model.layers.3.mlp'
# gate (router)
moe.gate.weight.copy_(M.get(f'{P3}.gate.weight').double())
moe.gate.e_score_correction_bias.copy_(M.get(f'{P3}.gate.e_score_correction_bias').double())
# shared experts (bf16)
moe.shared_experts.load_state_dict({f'{n}.weight': M.get(f'{P3}.shared_experts.{n}.weight').double() for n in ['gate_proj','up_proj','down_proj']})
# routed experts: HF NaiveMoe stores fused gate_up_proj [E,2I,Hd] and down_proj [E,Hd,I]
E=cfg.n_routed_experts; I=cfg.moe_intermediate_size; Hd=M.H
gup=torch.empty(E,2*I,Hd,dtype=DT); dwn=torch.empty(E,Hd,I,dtype=DT)
for e in range(E):
    g=M.dq(f'{P3}.experts.{e}.gate_proj')   # [Hd, I]
    u=M.dq(f'{P3}.experts.{e}.up_proj')     # [Hd, I]
    d=M.dq(f'{P3}.experts.{e}.down_proj')   # [I, Hd]
    gup[e,:I]=g.t(); gup[e,I:]=u.t()        # nn.Linear weight = [out,in] = [I,Hd]
    dwn[e]=d.t()                            # [Hd, I]
moe.experts.gate_up_proj.copy_(gup); moe.experts.down_proj.copy_(dwn)
out_hf3 = moe(x)[0]
out_me3 = M.mlp(3, x[0])
print('MoE   L3 vs HF: cos=%.6f nrat=%.4f maxdiff=%.3e' % (c(out_hf3,out_me3), out_me3.norm().item()/out_hf3.norm().item(), (out_hf3-out_me3).abs().max().item()))

# routing only: compare selected experts + weights
rl = moe.gate(x.view(-1,M.H))
ti_hf, tw_hf = moe.route_tokens_to_experts(rl)
# mine
gate_w=M.get(f'{P3}.gate.weight').double(); ebias=M.get(f'{P3}.gate.e_score_correction_bias').double()
sc=torch.sigmoid(x[0]@gate_w.t()); bi=sc+ebias[None]; ti_me=torch.topk(bi,M.topk,-1).indices
print('topk indices match (as sets):', all(set(ti_hf[i].tolist())==set(ti_me[i].tolist()) for i in range(S)))
