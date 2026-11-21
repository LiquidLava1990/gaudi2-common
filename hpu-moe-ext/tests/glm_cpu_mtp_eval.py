"""Offline accept evaluator for the NATIVE layer-78 MTP head on OUR dumped sequences.

Runs the full layer-78 forward (enorm/hnorm/eh_proj -> dense-MLA -> MoE -> shared_head.norm
-> lm_head) on the (hidden, target_ids, positions, seq_ids) shards produced by the serve dump,
and reports draft-vs-target argmax agreement (= depth-1 accept). Two weight modes:
  * bf16   : original zai-org layer-78 (the proven 62-75% graft) — the reference.
  * fp8sim : every layer-78 linear weight round-tripped bf16->float8_e4m3->bf16 (per-output-
             channel scaled), simulating the fp8 graft's quant loss to predict its accept
             BEFORE any serve restart. (Gaudi fnuz vs fn cancels in the self-consistent
             cast/gemm round-trip; the accept impact is the ~3-bit mantissa rounding, captured
             here.) Optionally also quantizes activations (--fp8-act).

Run in the serve image (needs the bf16 layer-78 + the int4 ckpt embed/lm_head + the dump):
  docker run --rm -v /mnt/fastmodels:/models:ro -v /mnt/data1/recipe_cache:/rc:ro \
    -v $HOME/AI/hpu-moe-ext:/w --entrypoint python3 vllm-gaudi:next \
    /w/tests/glm_cpu_mtp_eval.py --dumps '/rc/mtp_snap_v6/*.pt' --mode bf16 --max-seqs 40
"""
import argparse, glob, json, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
torch.set_grad_enabled(False)

DT = torch.float32  # fp32 is plenty for accept eval and ~20x faster than fp64 over many seqs
CKPT = '/models/GLM-5.1-int4-mixed-AutoRound'
ORIG = '/models/glm51-orig-l78'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--mode", choices=["bf16", "fp8sim"], default="bf16")
    ap.add_argument("--fp8dir", default=None, help="load layer-78 from a pre-quantized fp8 dir (verify on-disk artifact)")
    ap.add_argument("--fp8-act", action="store_true", help="also fp8-round activations into each GEMM")
    ap.add_argument("--max-seqs", type=int, default=40, help="cap sequences (MoE loop is slow on CPU)")
    ap.add_argument("--min-len", type=int, default=8)
    a = ap.parse_args()

    cfg = json.load(open(f'{CKPT}/config.json'))
    H = cfg['hidden_size']; n_heads = cfg['num_attention_heads']; kv_lora = cfg['kv_lora_rank']
    qk_nope = cfg['qk_nope_head_dim']; qk_rope = cfg['qk_rope_head_dim']; qk_head = cfg['qk_head_dim']
    v_head = cfg['v_head_dim']; eps = cfg['rms_norm_eps']
    rope_theta = cfg['rope_parameters']['rope_theta']; scaling = qk_head ** -0.5
    topk = cfg['num_experts_per_tok']; rsf = cfg['routed_scaling_factor']; norm_topk = cfg['norm_topk_prob']
    nL = cfg['num_hidden_layers']; V = cfg['vocab_size']
    idx = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
    oidx = json.load(open(f'{ORIG}/model.safetensors.index.json'))['weight_map']

    _h = {}
    def H_(path, fn):
        if (path, fn) not in _h:
            _h[(path, fn)] = safe_open(f'{path}/{fn}', framework='pt', device='cpu')
        return _h[(path, fn)]
    def get(n): return H_(CKPT, idx[n]).get_tensor(n)

    def _fp8(w):  # PER-TENSOR scaled e4m3 round-trip (matches the served fp8 graft quant)
        s = w.abs().amax().clamp_min(1e-8) / 448.0
        return ((w / s).to(torch.float8_e4m3fn).to(DT)) * s

    FP8 = a.mode == "fp8sim"
    f8idx = json.load(open(f"{a.fp8dir}/model.safetensors.index.json"))["weight_map"] if a.fp8dir else None
    _wc = {}
    def _deq78(n):  # layer-78 tensor as [out,in] f32 — from the on-disk fp8 dir (dequant) or bf16 orig
        if f8idx is not None:
            t = H_(a.fp8dir, f8idx[n]).get_tensor(n)
            sk = n + ".weight_scale"
            if sk in f8idx:  # fp8-quantized: dequant per-output-channel
                s = H_(a.fp8dir, f8idx[sk]).get_tensor(sk).to(DT)
                return t.to(DT) * s[:, None]
            return t.to(DT)  # kept fp16 (norms/gate)
        return H_(ORIG, oidx[n]).get_tensor(n).to(DT)
    def w78(n):  # [out,in] -> [in,out] for x @ w, with optional fp8 sim, cached
        if n not in _wc:
            w = _deq78(n).t().contiguous()
            _wc[n] = _fp8(w) if (FP8 and f8idx is None) else w  # fp8dir is already fp8; don't double-round
        return _wc[n]
    def g78(n): return _deq78(n)
    def rms(x, w): return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
    def rh(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
    def rope_neox(x, f):
        e = torch.cat((f, f), -1); return x * e.cos()[None, None] + rh(x) * e.sin()[None, None]
    def mm(x, w):  # GEMM with optional activation fp8
        if FP8 and a.fp8_act:
            s = x.abs().amax().clamp_min(1e-8) / 448.0
            x = (x / s).to(torch.float8_e4m3fn).to(DT) * s
        return x @ w

    emb = get('model.embed_tokens.weight').to(DT)
    lm = get('lm_head.weight').to(DT)
    P = f'model.layers.{nL}'; A = f'{P}.self_attn'; MP = f'{P}.mlp'
    gate_w = g78(f'{MP}.gate.weight'); ebias = g78(f'{MP}.gate.e_score_correction_bias')

    # ---- load dump shards, build contiguous per-seq runs (mirror train.py) ----
    files = sorted(glob.glob(a.dumps))
    fdat = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    HID = torch.cat([d['hidden'].to(DT) for d in fdat])
    TGT = torch.cat([d['target_ids'].long() for d in fdat])
    POS = torch.cat([d['positions'].long() for d in fdat])
    SID = torch.cat([d['seq_ids'].long() for d in fdat])
    good = torch.isfinite(HID).all(-1) & (TGT >= 0) & (TGT < V)
    HID, TGT, POS, SID = HID[good], TGT[good], POS[good], SID[good]
    order = torch.argsort(SID * 100000 + POS)
    HID, TGT, POS, SID = HID[order], TGT[order], POS[order], SID[order]
    brk = ((SID[1:] != SID[:-1]) | (POS[1:] != POS[:-1] + 1)).nonzero().flatten() + 1
    bounds = [0] + brk.tolist() + [SID.shape[0]]
    seqs = [(s, e) for s, e in zip(bounds[:-1], bounds[1:]) if e - s >= a.min_len]
    seqs = seqs[:a.max_seqs]
    print(f"mode={a.mode} fp8_act={a.fp8_act}  shards={len(files)}  eval seqs={len(seqs)}", flush=True)

    def layer78(h_prev, tok_next, positions):
        """h_prev [S,H] (main hidden at t), tok_next [S] (token at t+1), positions [S] -> logits [S,V]."""
        S = h_prev.shape[0]
        en = rms(emb[tok_next], g78(f'{P}.enorm.weight'))
        hn = rms(h_prev, g78(f'{P}.hnorm.weight'))
        x = mm(torch.cat([en, hn], -1), w78(f'{P}.eh_proj.weight'))
        fr = torch.einsum('s,d->sd', positions.to(DT), 1.0 / (rope_theta ** (torch.arange(0, qk_rope, 2, dtype=DT) / qk_rope)))
        n1 = rms(x, g78(f'{P}.input_layernorm.weight'))
        q = rms(mm(n1, w78(f'{A}.q_a_proj.weight')), g78(f'{A}.q_a_layernorm.weight'))
        q = mm(q, w78(f'{A}.q_b_proj.weight')).view(1, S, n_heads, qk_head).transpose(1, 2)
        qn, qp = q.split([qk_nope, qk_rope], -1)
        comp = mm(n1, w78(f'{A}.kv_a_proj_with_mqa.weight')); kc, kpe = comp.split([kv_lora, qk_rope], -1)
        kc = rms(kc, g78(f'{A}.kv_a_layernorm.weight'))
        kv = mm(kc, w78(f'{A}.kv_b_proj.weight')).view(1, S, n_heads, qk_nope + v_head)
        kno, vst = kv.split([qk_nope, v_head], -1); kno = kno.transpose(1, 2); vst = vst.transpose(1, 2)
        qp = rope_neox(qp, fr); kpe2 = rope_neox(kpe.view(1, 1, S, qk_rope), fr).expand(-1, n_heads, -1, -1)
        qs = torch.cat([qn, qp], -1); ks = torch.cat([kno, kpe2], -1)
        mask = torch.full((S, S), float('-inf'), dtype=DT).triu(1)[None, None]
        aw = F.softmax((qs @ ks.transpose(2, 3)) * scaling + mask, -1)
        at = mm((aw @ vst).transpose(1, 2).reshape(S, n_heads * v_head), w78(f'{A}.o_proj.weight'))
        r2 = x + at
        n2 = rms(r2, g78(f'{P}.post_attention_layernorm.weight'))
        scores = torch.sigmoid(mm(n2, gate_w.t())); biased = scores + ebias[None]
        tki = torch.topk(biased, topk, -1).indices; tkw = scores.gather(1, tki)
        if norm_topk: tkw = tkw / (tkw.sum(-1, keepdim=True) + 1e-20)
        routed = torch.zeros(S, H, dtype=DT)
        need = set(int(e) for e in tki.reshape(-1).tolist())
        ec = {eid: (w78(f'{MP}.experts.{eid}.gate_proj.weight'), w78(f'{MP}.experts.{eid}.up_proj.weight'),
                    w78(f'{MP}.experts.{eid}.down_proj.weight')) for eid in need}
        for i in range(S):
            acc = torch.zeros(H, dtype=DT)
            for k in range(topk):
                eid = int(tki[i, k]); g, u, dn = ec[eid]
                acc = acc + tkw[i, k] * mm(F.silu(mm(n2[i], g)) * mm(n2[i], u), dn)
            routed[i] = acc
        sh = mm(F.silu(mm(n2, w78(f'{MP}.shared_experts.gate_proj.weight'))) * mm(n2, w78(f'{MP}.shared_experts.up_proj.weight')), w78(f'{MP}.shared_experts.down_proj.weight'))
        h78 = r2 + sh + rsf * routed
        return mm(rms(h78, g78(f'{P}.shared_head.norm.weight')), lm.t())

    agree = total = 0; t0 = time.time()
    for si, (s, e) in enumerate(seqs):
        h = HID[s:e]; tg = TGT[s:e]; po = POS[s:e]
        # pair t -> t+1: input (h[t], tok=tg[t]); label = tg[t+1]
        lg = layer78(h[:-1], tg[:-1], po[:-1])
        pred = lg.argmax(-1)
        agree += int((pred == tg[1:]).sum()); total += int(tg[1:].numel())
        if si < 3 or si % 10 == 0:
            print(f"  seq {si}: len={e-s} run-accept={agree/max(total,1)*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n=== NATIVE L78 [{a.mode}{'+act' if a.fp8_act else ''}] DRAFT-TARGET ACCEPT "
          f"{agree}/{total} = {agree/max(total,1)*100:.1f}% ===", flush=True)


if __name__ == "__main__":
    main()
