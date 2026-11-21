"""Universal MTP head trainer. Trains a DraftHead (head.py) for a target model to maximize
depth-1 accept rate (draft argmax == target's actual next-next token). Frozen embed/lm_head.

Data: one or more shards, each a dict with 'hidden' [S,H] and EITHER 'target_ids' [S] OR
'logits_main' [S,V] (argmax taken). 'positions' [S] optional (defaults to 0..S-1 contiguous).

Pairs: for t in 0..S-2 -> input (hidden[t], tok_next=target[t], pos[t]); label = target[t+1].
Accept = argmax(draft_logits) == target[t+1].

Usage:
  python train.py --config glm51/config.json --dumps 'glm51/dumps/*.pt' --epochs 5 --out glm51/head.pt
  python train.py --config glm51/config.json --self-test           # tiny end-to-end smoke run
"""
import argparse
import glob
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from head import DraftHead, HeadConfig  # noqa: E402


def _load_frozen_embed_lmhead(cfg_d):
    """Load the target's embed + lm_head [vocab,H] (bf16->float32) from its checkpoint index."""
    src = cfg_d["frozen_src"]; H, V = cfg_d["hidden"], cfg_d["vocab"]
    from safetensors import safe_open
    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    emb_name = next((n for n in idx if n.endswith("embed_tokens.weight")), None)
    lm_name = next((n for n in idx if n.endswith("lm_head.weight")), None)
    h = {}

    def rd(name):
        fn = idx[name]
        if fn not in h:
            h[fn] = safe_open(f"{src}/{fn}", framework="pt", device="cpu")
        return h[fn].get_tensor(name)
    embed = rd(emb_name).to(torch.float32)
    lm = rd(lm_name).to(torch.float32) if lm_name else embed  # tied if no separate lm_head
    assert embed.shape == (V, H), f"embed {tuple(embed.shape)} != {(V, H)}"
    return embed, lm


def _load_shards(patterns, n_aux=None):
    """Load dump shards. `n_aux` (from the HeadConfig) selects the input feature CORRECT-BY-CONSTRUCTION:
      n_aux>=3 -> the [S, n_aux*H] low/mid/high aux stack ('hidden_aux'), which the head fuses (fc 3H->H);
      n_aux==0 -> the [S,H] final hidden ('hidden'), which the depth-1 / Medusa head consumes directly.
    When n_aux is None (legacy callers) fall back to: HEAD_USE_HIDDEN=1 forces 'hidden', else prefer the
    aux stack if present. Passing cfg.n_aux removes the env-var footgun (an n_aux=0 head fed the 3H aux
    -> shape mismatch in eh_proj; an n_aux=3 head fed the final hidden -> trains the wrong feature)."""
    import os as _os
    out = []
    files = []
    for p in patterns:
        files += sorted(glob.glob(p))
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if n_aux is not None:
            if int(n_aux) >= 3:
                if "hidden_aux" not in d:
                    raise KeyError(f"{f}: n_aux={n_aux} head needs 'hidden_aux' but the shard has none "
                                   f"(keys={list(d.keys())}). Re-dump with the EAGLE-3 aux capture.")
                hid = d["hidden_aux"].to(torch.float32)
            else:
                hid = (d["hidden"] if "hidden" in d else d["final"]).to(torch.float32)
        else:
            _force_hidden = (_os.environ.get("HEAD_USE_HIDDEN") or "0") in ("1", "true", "True")
            if _force_hidden and "hidden" in d:
                hid = d["hidden"].to(torch.float32)
            else:
                hid = (d["hidden_aux"] if "hidden_aux" in d else
                       (d["hidden"] if "hidden" in d else d["final"])).to(torch.float32)
        soft = d["logits_main"].to(torch.float32) if "logits_main" in d else None  # full target dist (EAGLE-3 soft-KL)
        if "target_ids" in d:
            tgt = d["target_ids"].to(torch.int64)
        else:
            tgt = soft.argmax(-1).to(torch.int64)
        pos = d.get("positions")
        pos = pos.to(torch.int64) if pos is not None else torch.arange(hid.shape[0])
        sid = d["seq_ids"].to(torch.int64) if "seq_ids" in d else None  # concurrent-dump request grouping
        tkid = d["topk_ids"].to(torch.int64) if "topk_ids" in d else None     # [S,K] teacher top-K tokens
        tklg = d["topk_logits"].to(torch.float32) if "topk_logits" in d else None  # [S,K] teacher top-K logits
        out.append((hid, tgt, pos, sid, tkid, tklg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dumps", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--out", default="head.pt")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation (sequences per optimizer step)")
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0=off)")
    ap.add_argument("--warmup", type=int, default=0, help="linear-warmup optimizer steps")
    ap.add_argument("--temp", type=float, default=1.0, help="distillation temperature (EAGLE/SpecForge use T=1)")
    ap.add_argument("--hard-w", type=float, default=0.1, help="weight on hard CE when soft labels present")
    ap.add_argument("--noise", type=float, default=0.1, help="EAGLE feature-noise aug U(-n,n) on input hidden (train only)")
    ap.add_argument("--softk", type=int, default=16, help="top-K used for soft-KL (HASS optimum ~10)")
    ap.add_argument("--patience", type=int, default=12, help="early-stop patience on val accept (0=off)")
    ap.add_argument("--depth", type=int, default=1, help="TTT autoregressive unroll depth (1=depth-1 legacy; "
                    ">1 trains the head to feed its OWN hidden back -> serve num_speculative_tokens=depth)")
    ap.add_argument("--ttt-decay", type=float, default=0.8, help="per-depth loss weight decay in TTT (w_d = decay**d)")
    ap.add_argument("--device", default="cpu", help="cpu | hpu (HPU does the big vocab matmul on the MME = ~100x faster)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    cfg_d = json.load(open(a.config))
    cfg = HeadConfig(**{k: v for k, v in cfg_d.items() if k in HeadConfig.__dataclass_fields__})

    if a.self_test:
        # off-card pipeline smoke: stub the frozen embed/lm_head (random) so --self-test needs NO checkpoint.
        print("[factory] SELF-TEST: stubbing frozen embed/lm_head (no checkpoint load)", flush=True)
        embed = torch.randn(cfg.vocab, cfg.hidden) * 0.02
        lm = embed.clone()
    else:
        print(f"[factory] loading frozen embed/lm_head from {cfg_d['frozen_src']} ...", flush=True)
        embed, lm = _load_frozen_embed_lmhead(cfg_d)
    head = DraftHead(cfg, embed, lm)
    dev = a.device
    if dev == "hpu":
        import habana_frameworks.torch.core as _htcore  # noqa: F401  (registers the hpu backend)
        head = head.to("hpu")
        print("[factory] training on HPU (MME does the vocab matmul)", flush=True)
    nparam = sum(p.numel() for p in head.trainable_params())
    print(f"[factory] head built: hidden={cfg.hidden} vocab={cfg.vocab} mlp={cfg.mlp_dim} "
          f"norm={cfg.norm} act={cfg.act} trainable={nparam/1e6:.0f}M", flush=True)

    if a.self_test:
        S = 64
        hid = torch.randn(S, cfg.hidden) * 0.1
        tgt = torch.randint(0, cfg.vocab, (S,))
        shards = [(hid, tgt, torch.arange(S), None, None, None)]
        print("[factory] SELF-TEST on random data (pipeline smoke)", flush=True)
    else:
        shards = _load_shards(a.dumps if a.dumps else [cfg_d.get("default_dumps", "")], n_aux=cfg.n_aux)
    import random as _rnd
    V = cfg.vocab
    # SEQUENCE-structured: build contiguous per-request runs so the lean-block attention sees real context
    # (shuffled positions mis-train it). tokens clamped to real vocab (vllm pads the logits vocab).
    seqs = []  # each: (HID, TGT, POS, TKID|None, TKLG|None)  -- top-K teacher dist for soft-KL
    have_topk = len(shards) > 0 and all(s[4] is not None and s[5] is not None for s in shards)
    if all(s[3] is not None for s in shards) and len(shards) > 0:
        # CONCURRENT-dump path: group by seq_id across all shards, sort each by position (vectorized)
        HID = torch.cat([s[0] for s in shards])
        TGT = torch.cat([s[1] for s in shards])
        POS = torch.cat([s[2] for s in shards])
        SID = torch.cat([s[3] for s in shards])
        TKID = torch.cat([s[4] for s in shards]) if have_topk else None
        TKLG = torch.cat([s[5] for s in shards]) if have_topk else None
        # drop nan/inf hidden + out-of-vocab tokens (concurrent-dump finished/padding-slot artifacts)
        good = torch.isfinite(HID).all(-1) & (TGT >= 0) & (TGT < V)
        if have_topk:
            good &= torch.isfinite(TKLG).all(-1)
        HID, TGT, POS, SID = HID[good], TGT[good], POS[good].long(), SID[good]
        if have_topk:
            TKID, TKLG = TKID[good], TKLG[good]
        order = torch.argsort(SID.to(torch.int64) * 100000 + POS)  # (seq, pos) lexsort; pos < 512
        HID, TGT, POS, SID = HID[order], TGT[order], POS[order], SID[order]
        if have_topk:
            TKID, TKLG = TKID[order], TKLG[order]
        # split at seq_id changes OR position gaps (filtering can leave gaps -> avoid wrong cross-gap pairs)
        brk = ((SID[1:] != SID[:-1]) | (POS[1:] != POS[:-1] + 1)).nonzero().flatten() + 1
        bounds = [0] + brk.tolist() + [SID.shape[0]]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e - s >= 2:
                seqs.append((HID[s:e], TGT[s:e], POS[s:e],
                             TKID[s:e] if have_topk else None, TKLG[s:e] if have_topk else None))
    else:
        # legacy single-stream path: split by position contiguity within each shard
        for hid, tgt, pos, _sid, tkid, tklg in shards:
            if hid.shape[0] < 2:
                continue
            tgt = tgt.clamp(0, V - 1)
            brk = (pos[1:] != pos[:-1] + 1).nonzero().flatten() + 1
            bounds = [0] + brk.tolist() + [pos.shape[0]]
            for s, e in zip(bounds[:-1], bounds[1:]):
                if e - s >= 2:
                    seqs.append((hid[s:e], tgt[s:e], pos[s:e],
                                 tkid[s:e] if tkid is not None else None,
                                 tklg[s:e] if tklg is not None else None))
    _rnd.seed(0); _rnd.shuffle(seqs)
    nval = max(1, len(seqs) // 10)
    val_seqs, tr_seqs = seqs[:nval], seqs[nval:]
    if dev != "cpu":  # move all seq tensors onto the device once (no per-batch transfer)
        def _mv(s):
            return tuple(x.to(dev) if x is not None else None for x in s)
        val_seqs = [_mv(s) for s in val_seqs]; tr_seqs = [_mv(s) for s in tr_seqs]
    npairs = sum(s[0].shape[0] - 1 for s in tr_seqs)
    print(f"[factory] {len(seqs)} sequences, {npairs} pairs ({nval} val); SEQUENCE training (attention=context)", flush=True)

    def seq_acc(sqs):
        head.eval(); ag = t = 0
        with torch.no_grad():
            for hid, tgt, pos, _ti, _tl in sqs:
                lg = head(hid[:-1], tgt[:-1], pos[:-1])
                ag += int((lg.argmax(-1) == tgt[1:]).sum()); t += tgt[1:].numel()
        return ag / max(t, 1)

    def seq_acc_perdepth(sqs, depth):
        """Per-depth autoregressive draft accept: feed the head's OWN hidden back `depth` steps and
        measure each draft position's accept. The depth-2/3 numbers reveal whether TTT worked (they
        were ~0 for the depth-1-trained head). Mirrors the serve's autoregressive proposer exactly."""
        head.eval(); ok = [0] * depth; tot = [0] * depth
        with torch.no_grad():
            for hid, tgt, pos, _ti, _tl in sqs:
                S = hid.shape[0]; h_prev = hid
                for d in range(depth):
                    L = S - 1 - d
                    if L < 1:
                        break
                    h_out = head.forward_hidden(h_prev[:L], tgt[d:d + L], pos[d:d + L])
                    c = (head.compute_logits(h_out).argmax(-1) == tgt[d + 1:d + 1 + L])
                    ok[d] += int(c.sum()); tot[d] += int(c.numel())
                    h_prev = h_out
        return [ok[d] / max(tot[d], 1) for d in range(depth)]

    # EAGLE-3 soft-KL: distill the target's top-K next-next-token distribution (T-smoothed) into the
    # head, mixed with a little hard CE. Falls back to pure hard-CE when top-K labels aren't present.
    use_soft = have_topk and a.hard_w < 1.0
    T = max(1e-3, a.temp)
    print(f"[factory] loss = {('soft-KL(T=%g)+%.2g*hardCE' % (T, a.hard_w)) if use_soft else 'hard-CE'}", flush=True)

    SK = min(a.softk, 4096)  # soft-KL top-K width (HASS optimum ~10); dumped K may be larger

    def _loss(hid, tgt, pos, tkid, tklg):
        if a.depth > 1:
            # TTT (training-time-test): autoregressively unroll the head `depth` steps, feeding its OWN
            # hidden output back as the next-step input (teacher-forced tokens), hard-CE per depth. This is
            # what makes depth-2/3 drafts non-garbage at serve (the head learns its own hidden distribution).
            S = hid.shape[0]; h_prev = hid; total = hid.new_zeros(()); wsum = 0.0
            for d in range(a.depth):
                L = S - 1 - d
                if L < 1:
                    break
                hh = h_prev[:L]
                if a.noise > 0 and d == 0:  # feature-noise only on the real target hidden (d=0 input)
                    sc = hh.abs().mean(-1, keepdim=True)
                    hh = hh + (torch.rand_like(hh) * 2 - 1) * a.noise * sc
                h_out = head.forward_hidden(hh, tgt[d:d + L], pos[d:d + L])   # head's own hidden
                lg = head.compute_logits(h_out)
                w = a.ttt_decay ** d
                lab = slice(d + 1, d + 1 + L)
                hard = F.cross_entropy(lg, tgt[lab])
                if use_soft and tkid is not None:  # soft-KL per depth: the teacher top-K dist GENERALIZES
                    tk = tkid[lab][:, :SK]; tl = tklg[lab][:, :SK]            # vs hard-argmax which MEMORIZES
                    valid = (tk >= 0) & (tk < V)
                    tp = torch.softmax(tl / T, dim=-1) * valid
                    tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-8)
                    logp = F.log_softmax(lg / T, dim=-1).gather(-1, tk.clamp(0, V - 1))
                    soft = -(tp * logp).sum(-1).mean() * (T * T)
                    step_loss = (1 - a.hard_w) * soft + a.hard_w * hard
                else:
                    step_loss = hard
                total = total + step_loss * w
                wsum += w
                h_prev = h_out                                               # autoregress: feed it back
            return total / max(wsum, 1e-8)
        hin = hid[:-1]
        if a.noise > 0:  # EAGLE feature-noise aug (train only). RELATIVE to per-row scale so it's meaningful
            # on the un-normed aux residual stream (RMS in the tens/hundreds), not just the normed final hidden.
            sc = hin.abs().mean(-1, keepdim=True)
            hin = hin + (torch.rand_like(hin) * 2 - 1) * a.noise * sc
        lg = head(hin, tgt[:-1], pos[:-1])                        # [N, V] student logits
        hard = F.cross_entropy(lg, tgt[1:])
        if not use_soft or tkid is None:
            return hard
        tk = tkid[1:, :SK]; tl = tklg[1:, :SK]                    # [N,SK] teacher top-K @ t+1
        valid = (tk >= 0) & (tk < V)
        tp = torch.softmax(tl / T, dim=-1) * valid                # zero out vocab-pad entries
        tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-8)        # renormalize over valid top-K
        logp = F.log_softmax(lg / T, dim=-1).gather(-1, tk.clamp(0, V - 1))  # [N,SK] student logp at teacher tokens
        soft = -(tp * logp).sum(-1).mean() * (T * T)
        return (1 - a.hard_w) * soft + a.hard_w * hard

    opt = torch.optim.AdamW(head.trainable_params(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    best = seq_acc(val_seqs); best_state = None
    print(f"[factory] baseline accept (untrained): {best*100:.1f}%", flush=True)
    ACCUM = max(1, a.accum)
    nstep = [0]
    steps_per_ep = max(1, len(tr_seqs) // ACCUM)
    total_steps = max(1, steps_per_ep * a.epochs)

    def _step():
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(head.trainable_params(), a.clip)
        if a.warmup > 0 and nstep[0] < a.warmup:
            for g in opt.param_groups:
                g["lr"] = a.lr * (nstep[0] + 1) / a.warmup
        else:  # cosine decay after warmup
            import math
            prog = (nstep[0] - a.warmup) / max(1, total_steps - a.warmup)
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, prog))))
        opt.step(); opt.zero_grad(); nstep[0] += 1

    since_best = 0
    for ep in range(a.epochs):
        head.train(); _rnd.shuffle(tr_seqs); tot = 0.0; nb = 0; opt.zero_grad()
        for si, seq in enumerate(tr_seqs):
            loss = _loss(*seq) / ACCUM
            loss.backward(); tot += loss.item() * ACCUM; nb += 1
            if (si + 1) % ACCUM == 0:
                _step()
        if nb % ACCUM != 0:
            _step()
        acc = seq_acc(val_seqs)
        if acc > best:
            best = acc; since_best = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()
                          if "embed_w" not in k and "lm_head_w" not in k}
        else:
            since_best += 1
        print(f"[factory] epoch {ep+1}/{a.epochs}  loss={tot/max(nb,1):.3f}  accept={acc*100:.1f}%  best={best*100:.1f}%", flush=True)
        if a.patience and since_best >= a.patience:  # early-stop (small-data overfits fast)
            print(f"[factory] early-stop: no val improvement in {a.patience} epochs", flush=True)
            break
    if best_state is not None:  # restore best-val checkpoint (accept is noisy across epochs)
        head.load_state_dict(best_state, strict=False)
        print(f"[factory] restored best-val head (accept={best*100:.1f}%)", flush=True)
    if a.depth > 1:  # the TTT payoff: depth-2/3 accept should now be >0 (it was ~0 for the depth-1 head)
        pd = seq_acc_perdepth(val_seqs, a.depth)
        print(f"[factory] TTT per-depth accept: " +
              ", ".join("d%d=%.1f%%" % (i, p * 100) for i, p in enumerate(pd)), flush=True)
    if not a.self_test:
        torch.save({"cfg": cfg_d, "state": {k: v for k, v in head.state_dict().items()
                    if "embed_w" not in k and "lm_head_w" not in k}}, a.out)
        print(f"[factory] saved trained head -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
