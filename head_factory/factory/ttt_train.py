"""TTT / HASS k-step-unroll trainer — fixes the draft head's SELF-FEED CLIFF.

WHY (the diagnosis this file exists to fix):
  The depth-1 trainer (train.py) is TEACHER-FORCED: at every position it feeds the head the REAL
  dumped hidden h_t and the REAL committed token tok_{t+1}, and scores argmax==tgt_{t+1}. That trains
  a1 (the FIRST draft) well, but serve does NOT teacher-force past depth-1: at draft positions 2..K it
  feeds the head its OWN previous output hidden + its OWN previous argmax token (eval_head.py
  self_feed_eval / hpu_eagle_factoryeagle.py propose). The head was never trained on its OWN (slightly
  off) hidden, so it DRIFTS:
     measured GLM-5.2 depth-1 head:  a1 = 28.8% (data-bound)   a2 = 4.6% (the CLIFF)
     -> derived served accept-len 1 + a1 + a1*a2 + ...  ~= 1.30   vs  native 2.26.
  The a2 collapse (28.8% -> 4.6%) is the train/serve mismatch, NOT a data problem (a1 is the data
  ceiling). HASS (Harmonized Self-feeding) / EAGLE training-time-test (TTT) closes it by UNROLLING the
  draft k steps DURING TRAINING along the head's OWN feedback, exactly mirroring serve, with a
  0.8^i-decayed loss per step. This is the README's documented TODO #3 ("on-policy / TTT").

WHAT THIS DOES (per training sequence, k = 3..4 unroll steps):
  step i=0 (SEED, == the teacher-forced depth-1 step, identical to train.py):
     in_hidden_0 = dumped hidden[t]      (REAL target hidden; for n_aux>=3, the fused aux -> H)
     in_token_0  = dumped tgt[t]         (REAL committed token)
     in_pos_0    = dumped pos[t]
     h0 = head.forward_hidden(in_hidden_0, in_token_0, in_pos_0)   # head's OWN output hidden [.,H]
     lg0 = lm_head(h0);   loss_0 = (soft-KL@t+1)/CE(tgt[t+1])      # weight 0.8^0 = 1.0
  step i>=1 (SELF-FEED, the new part — mirrors serve lines 199-221):
     in_hidden_i = h_{i-1}               (the head's OWN previous output hidden, width H, NOT re-fused)
     in_token_i  = argmax(lg_{i-1})      (the head's OWN previous DRAFT token, re-embedded via embed_w
                                          == the target embed; serve re-embeds via embed_tokens -> same)
     in_pos_i    = in_pos_0 + i          (serve advances +1 each draft)
     h_i = head.forward_hidden(in_hidden_i, in_token_i, in_pos_i)
     lg_i = lm_head(h_i);  loss_i = (soft-KL@t+i+1)/CE(tgt[t+i+1]) * 0.8^i
  total = sum_i 0.8^i * loss_i   (only positions with a real tgt[t+i+1] in the contiguous run count).

GRADIENT THROUGH THE UNROLL (the HASS detail):
  * the fed-back TOKEN is argmax -> NON-differentiable; we feed it detached + re-embed (straight-through
    is not needed: the token only selects an embedding row, and the loss at step i still flows into the
    head via h_{i-1} and the current-step params).
  * the fed-back HIDDEN h_{i-1}: HASS applies stop-gradient on the recursive feature (each step's loss
    trains THAT step's forward, not back through the whole chain) -> stable + matches the published recipe.
    --ttt-bptt keeps the hidden in the graph (full BPTT through the unroll) for ablation. Default = detached.
  * the SEED uses the REAL dumped hidden/token (on-policy anchor), so step-0 == train.py's loss exactly
    (a1 is preserved; TTT only ADDS the self-feed steps that fix a2..aK).

REUSE: imports train.py (frozen embed/lm_head loader, shard loader, seq builder pieces, the _loss soft-KL
  math is mirrored 1:1) + head.py (DraftHead.forward_hidden / compute_logits == the serve core). Does NOT
  modify train.py. CARD-FREE validation: --self-test (no checkpoint) + a CPU dry-run on real dump shards.
  The real --device hpu run (V=154880 vocab matmul on the MME) is queued for when the card frees.

USAGE:
  # CPU dry-run on a few real shards (card-free; tiny, just exercises the unroll loss + accept):
  python ttt_train.py --config glm52/config.train.d1.host.json \
    --dumps '/mnt/data1/recipe_cache/mtp_dump/shard_0000[0-2].pt' \
    --ttt-steps 3 --epochs 1 --device cpu --max-seqs 64 --dry-run
  # off-card pipeline smoke (no checkpoint, random frozen embed/lm_head):
  python ttt_train.py --config glm52/config.train.d1.host.json --self-test --ttt-steps 3
  # ON-CARD real train (queued; warm-starts from the depth-1 head to keep a1):
  python ttt_train.py --config glm52/config.train.host.json --dumps '<DUMP>/shard_*.pt' \
    --device hpu --ttt-steps 4 --decay 0.8 --warm-start glm52/head_eagle3.pt \
    --epochs 20 --bs 8192 --lr 3e-4 --temp 1.0 --hard-w 0.1 --softk 16 --noise 0.1 \
    --self-feed-noise 0.0 --patience 6 --out glm52/head_ttt.pt
"""
import argparse
import json
import math
import random as _rnd
import sys

import torch
import torch.nn.functional as F

# v6 residual-vs-native L2 weight on the adapter logit residual (keeps it calibrated to native's scale so
# the residual fixes native-wrong cases WITHOUT breaking native-right). Overridable via --res-reg.
REG_W = 0.02

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from head import DraftHead, HeadConfig  # noqa: E402  (the proven head; reused, not reimplemented)
# REUSE train.py verbatim — frozen embed/lm_head loader, shard loader, and the sequence builder. We do NOT
# duplicate that logic (single source of truth); TTT only adds the k-step-unroll loss on top of those seqs.
from train import _load_frozen_embed_lmhead, _load_shards  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Sequence building — IDENTICAL grouping to train.py.main() (concurrent seq_id path + legacy path),
# factored out so both trainers see the SAME (HID, TGT, POS, TKID, TKLG) sequences. (train.py builds
# these inline; we re-derive them here with the same code path so TTT == train.py on the seed step.)
# --------------------------------------------------------------------------------------------------
def build_seqs(shards, V):
    """shards: list of (hid, tgt, pos, sid, tkid, tklg) from _load_shards. Returns a list of
    per-request contiguous sequences (HID[s:e], TGT, POS, TKID|None, TKLG|None), top-K teacher dist
    carried for soft-KL. Mirrors train.py's seq construction byte-for-byte (concurrent + legacy)."""
    seqs = []
    have_topk = len(shards) > 0 and all(s[4] is not None and s[5] is not None for s in shards)
    if len(shards) > 0 and all(s[3] is not None for s in shards):
        # CONCURRENT-dump path: group by seq_id across all shards, sort each by position (vectorized)
        HID = torch.cat([s[0] for s in shards]); TGT = torch.cat([s[1] for s in shards])
        POS = torch.cat([s[2] for s in shards]); SID = torch.cat([s[3] for s in shards])
        TKID = torch.cat([s[4] for s in shards]) if have_topk else None
        TKLG = torch.cat([s[5] for s in shards]) if have_topk else None
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
    return seqs, have_topk


# --------------------------------------------------------------------------------------------------
# The TTT k-step-unroll loss — the whole point of this file.
# --------------------------------------------------------------------------------------------------
def ttt_loss(head, cfg, seq, *, ttt_steps, decay, use_soft, T, hard_w, SK, noise,
             self_feed_noise, bptt):
    """k-step self-feed unroll loss for ONE sequence, mirroring serve self-feed (eval_head.self_feed_eval).

    seq = (hid, tgt, pos, tkid|None, tklg|None). All anchors t with a real next token are rolled forward
    `ttt_steps` draft positions, feeding the head its OWN output hidden + OWN argmax token at steps >=1.
    Per-step loss (soft-KL+hardCE or hard-CE) is weighted by decay**i and masked to anchors whose
    tgt[t+i+1] exists in this contiguous run. Returns (loss, hit[K], cnt[K]) for accept telemetry.

    SHAPES: A = (#anchors) = n-1.  cur_hidden seed [A, in_w] (in_w = H or n_aux*H); after step-0 the
    fed-back hidden is width H (head output), so forward_hidden's width-detect does NOT re-fuse (== serve).
    """
    hid, tgt, pos, tkid, tklg = seq
    n = tgt.shape[0]
    dev = head.embed_w.device
    V = cfg.vocab
    K = ttt_steps
    H = cfg.hidden

    A = n - 1                                    # anchors t = 0..n-2 (each has a real t+1)
    anchor_t = torch.arange(A, device=dev)       # [A]
    cur_hidden = hid[:A].to(dev)                 # SEED: REAL dumped hidden at each anchor   [A, in_w]
    cur_pos = pos[:A].to(dev).long()             # SEED: real position                       [A]
    cur_token = tgt[:A].to(dev)                  # SEED: committed token at each anchor       [A]

    # EAGLE feature-noise aug on the SEED hidden ONLY (train.py applies it to the real input hidden).
    # RELATIVE to per-row scale (meaningful on the un-normed aux residual stream too).
    if noise > 0:
        sc = cur_hidden.abs().mean(-1, keepdim=True)
        cur_hidden = cur_hidden + (torch.rand_like(cur_hidden) * 2 - 1) * noise * sc

    total = cur_hidden.new_zeros(())
    wsum = 0.0
    hit = [0] * K; cnt = [0] * K

    for i in range(K):
        w = decay ** i
        # head's OWN output hidden at this draft step (the serve core: forward_hidden -> compute_logits)
        h = head.forward_hidden(cur_hidden, cur_token, cur_pos)   # [A, H]
        lg = head.compute_logits(h)                              # [A, V]

        # target at t + i + 1 — only anchors where that index exists contribute (serve: chain reaches
        # depth i+1 only if all earlier drafts existed; here every anchor is independent so we mask).
        tgt_idx = anchor_t + (i + 1)
        valid = tgt_idx < n
        if valid.any():
            real = tgt[tgt_idx.clamp(max=n - 1).cpu()].to(dev)    # tgt[t+i+1]
            lgv = lg[valid]; realv = real[valid]
            hard = F.cross_entropy(lgv, realv)
            if use_soft and tkid is not None:
                # teacher top-K @ (t+i+1): gather rows by tgt_idx (the soft target lives at the
                # predicted position, NOT the anchor). matches train.py's tk = tkid[1:] alignment,
                # generalized to depth i+1.
                ti = tgt_idx.clamp(max=n - 1).cpu()
                tk = tkid[ti][:, :SK].to(dev)[valid]              # [Nv, SK] teacher top-K ids
                tl = tklg[ti][:, :SK].to(dev)[valid]              # [Nv, SK] teacher top-K logits
                vmask = (tk >= 0) & (tk < V)
                tp = torch.softmax(tl / T, dim=-1) * vmask
                tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-8)
                logp = F.log_softmax(lgv / T, dim=-1).gather(-1, tk.clamp(0, V - 1))
                soft = -(tp * logp).sum(-1).mean() * (T * T)
                step_loss = (1 - hard_w) * soft + hard_w * hard
            else:
                step_loss = hard
            total = total + w * step_loss
            wsum += w
            with torch.no_grad():
                amax = lg.argmax(-1)
                hit[i] += int(((amax == real) & valid).sum())
                cnt[i] += int(valid.sum())

        # FEED BACK the head's OWN hidden + OWN argmax for the next draft step (the serve loop).
        amax = lg.argmax(-1)                      # [A] the head's DRAFT token at this step (no grad)
        cur_token = amax.detach()                 # re-embed via embed_w == target embed (serve-exact)
        nh = h if bptt else h.detach()            # HASS: stop-grad on the recursive feature (default)
        if self_feed_noise > 0:                   # optional aug on the fed-back hidden (off by default)
            sc = nh.abs().mean(-1, keepdim=True)
            nh = nh + (torch.rand_like(nh) * 2 - 1) * self_feed_noise * sc
        cur_hidden = nh                           # width H -> forward_hidden does NOT re-fuse (== serve)
        cur_pos = cur_pos + 1                     # serve advances +1 each draft

    loss = total / max(wsum, 1e-8)
    return loss, hit, cnt


# --------------------------------------------------------------------------------------------------
# Self-feed accept telemetry (validation) — same chain as eval_head.self_feed_eval, no grad.
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def self_feed_accept(head, seqs, K):
    head.eval()
    dev = head.embed_w.device
    hit = [0] * K; cnt = [0] * K
    for hid, tgt, pos, _ti, _tl in seqs:
        n = tgt.shape[0]
        if n < 2:
            continue
        A = n - 1
        anchor_t = torch.arange(A, device=dev)
        cur_hidden = hid[:A].to(dev); cur_token = tgt[:A].to(dev); cur_pos = pos[:A].to(dev).long()
        for i in range(K):
            h = head.forward_hidden(cur_hidden, cur_token, cur_pos)
            lg = head.compute_logits(h)
            amax = lg.argmax(-1)
            tgt_idx = anchor_t + (i + 1)
            valid = tgt_idx < n
            if valid.any():
                real = tgt[tgt_idx.clamp(max=n - 1).cpu()].to(dev)
                hit[i] += int(((amax == real) & valid).sum()); cnt[i] += int(valid.sum())
            cur_hidden = h; cur_token = amax; cur_pos = cur_pos + 1
    a = [hit[i] / max(cnt[i], 1) for i in range(K)]
    al = 1.0; prod = 1.0
    for aj in a:
        prod *= aj; al += prod
    return a, al, cnt


# --------------------------------------------------------------------------------------------------
# NATIVE-SELFFEED per-step training (2026-06-23): when the dump carries `target_verify` (the rejection
# sampler's per-slot TARGET argmax) we train on the CORRECT label -- the head's depth-(j+1) self-feed draft
# should match target_verify[i, j] (== the serve's acceptance comparison). This REPLACES the broken
# seq-contiguity label (tgt[t+i+1]), which used the per-STEP seed stream (a variable-stride subsample of the
# true tokens -- consecutive seeds are NOT adjacent tokens, so that label was wrong). Each STEP is a
# self-contained example: seed (hidden[i], target_ids[i]); labels = target_verify[i, 0..K-1]; masked to the
# slots that were actually verified that step (target_verify != -1). This is the native-head TTT objective.
# --------------------------------------------------------------------------------------------------
def load_verify_steps(dumps, n_aux):
    """Load v5 native-selffeed shards and build per-STEP (HID, SEED, LABEL[K]) for adapter/head training.
    The LABEL at depth k for a seed at (seq, pos) is the committed token at (seq, pos+k) -- reconstructed
    GAPLESS from the token stream (seeds + accepted-drafts filling seed-position gaps), EXACTLY as
    native_selffeed_accept does. label==-1 where pos+k is past the last captured seed (masked in the loss).
    Returns (HID[N,inw], SEED[N], LABEL[N,K]) or None if the dump is not a v5 token-stream dump."""
    import glob as _glob
    import sys as _sys
    _sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from native_selffeed_accept import _load as _nsa_load  # reuse the loader (single source of truth)
    files = []
    for p in dumps:
        files += sorted(_glob.glob(p))
    if not files:
        raise FileNotFoundError(f"[ttt/verify] no shards matched {dumps}")
    # v5 schema check: needs the token-stream keys (NOT the dead target_verify/target_argmax).
    d0 = torch.load(files[0], map_location="cpu", weights_only=False)
    if not all(kk in d0 for kk in ("target_ids", "native_draft", "positions", "seq_ids")):
        return None
    if "target_verify" in d0 or "target_argmax" in d0:
        return None  # an OLD (pre-v5) dump -> caller refuses (its labels were the unreachable spec signal)

    tid, nd, pos, sid, K, _ = _nsa_load(dumps)
    # also need the seed HIDDEN (+ v6 native top-K logits/ids), aligned to (tid,nd,pos,sid) concat order.
    HID = []; NTL = []; NTI = []; have_nlog = True
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        HID.append((d["hidden"] if "hidden" in d else d["final"]).to(torch.float32))
        if "native_topk_logits" in d and "native_topk_ids" in d:
            NTL.append(d["native_topk_logits"].to(torch.float32))
            NTI.append(d["native_topk_ids"].to(torch.int64))
        else:
            have_nlog = False
    hid_all = torch.cat(HID)
    ntl_all = torch.cat(NTL) if (have_nlog and NTL) else None
    nti_all = torch.cat(NTI) if (have_nlog and NTI) else None
    # reconstruct token[seq][pos] gaplessly (mirror native_selffeed_accept.reconstruct_and_score's fill),
    # then label[row, k] = token[seq_row, pos_row + (k+1)] (or -1 if absent).
    order = torch.argsort(sid.to(torch.int64) * (10 ** 12) + pos.to(torch.int64))
    inv = torch.empty_like(order); inv[order] = torch.arange(order.shape[0])
    s_tid, s_nd, s_pos, s_sid = tid[order], nd[order], pos[order], sid[order]
    brk = (s_sid[1:] != s_sid[:-1]).nonzero().flatten() + 1
    bounds = [0] + brk.tolist() + [s_sid.shape[0]]
    LAB_sorted = torch.full((s_sid.shape[0], K), -1, dtype=torch.int64)
    for s, e in zip(bounds[:-1], bounds[1:]):
        rp = s_pos[s:e].tolist(); rt = s_tid[s:e].tolist(); rn = s_nd[s:e]
        token = {}
        for i in range(len(rp)):
            token[rp[i]] = rt[i]
            if i + 1 < len(rp):
                acc = rp[i + 1] - rp[i] - 1
                for j in range(min(acc, K)):
                    token[rp[i] + 1 + j] = int(rn[i, j])
        for i in range(len(rp)):
            for k in range(1, K + 1):
                lbl = token.get(rp[i] + k)
                if lbl is not None:
                    LAB_sorted[s + i, k - 1] = lbl
    LAB = LAB_sorted[inv]   # back to original row order (aligned with hid_all/tid)
    return hid_all, tid, LAB, ntl_all, nti_all


def verify_unroll_loss(head, cfg, hid, seed_tok, verify, pos0, *, ttt_steps, decay, noise,
                       self_feed_noise, bptt):
    """Per-step self-feed unroll with TARGET-VERIFY labels (the correct native-head TTT loss). One BATCH of
    steps: seed (hid[B,inw], seed_tok[B]); roll the head K draft positions feeding its OWN hidden + OWN
    argmax; at depth i the label is verify[B, i] (masked where verify==-1). hard-CE. Returns (loss,hit,cnt)."""
    B = hid.shape[0]
    dev = head.embed_w.device
    K = ttt_steps
    cur_hidden = hid.to(dev)
    cur_token = seed_tok.to(dev)
    cur_pos = pos0.to(dev).long()
    if noise > 0:
        sc = cur_hidden.abs().mean(-1, keepdim=True)
        cur_hidden = cur_hidden + (torch.rand_like(cur_hidden) * 2 - 1) * noise * sc
    total = cur_hidden.new_zeros(())
    wsum = 0.0
    hit = [0] * K; cnt = [0] * K
    for i in range(K):
        w = decay ** i
        h = head.forward_hidden(cur_hidden, cur_token, cur_pos)   # [B,H]
        lg = head.compute_logits(h)                              # [B,V]
        lab = verify[:, i].to(dev) if i < verify.shape[1] else torch.full((B,), -1, device=dev)
        valid = lab >= 0
        if valid.any():
            step_loss = F.cross_entropy(lg[valid], lab[valid])
            total = total + w * step_loss
            wsum += w
            with torch.no_grad():
                amax = lg.argmax(-1)
                hit[i] += int(((amax == lab) & valid).sum())
                cnt[i] += int(valid.sum())
        amax = lg.argmax(-1)
        cur_token = amax.detach()
        nh = h if bptt else h.detach()
        if self_feed_noise > 0:
            sc = nh.abs().mean(-1, keepdim=True)
            nh = nh + (torch.rand_like(nh) * 2 - 1) * self_feed_noise * sc
        cur_hidden = nh
        cur_pos = cur_pos + 1
    loss = total / max(wsum, 1e-8)
    return loss, hit, cnt


@torch.no_grad()
def verify_accept(head, HID, SEED, VER, K, bs=8192):
    """Per-step self-feed accept vs target_verify (the serve metric, off-card). Returns (a[K], accept-len)."""
    head.eval()
    dev = head.embed_w.device
    N = HID.shape[0]
    hit = [0] * K; cnt = [0] * K
    for s in range(0, N, bs):
        e = min(N, s + bs)
        cur_hidden = HID[s:e].to(dev); cur_token = SEED[s:e].to(dev)
        cur_pos = torch.zeros(e - s, device=dev, dtype=torch.long)
        ver = VER[s:e]
        for i in range(K):
            h = head.forward_hidden(cur_hidden, cur_token, cur_pos)
            lg = head.compute_logits(h)
            amax = lg.argmax(-1)
            lab = ver[:, i].to(dev) if i < ver.shape[1] else torch.full((e - s,), -1, device=dev)
            valid = lab >= 0
            if valid.any():
                hit[i] += int(((amax == lab) & valid).sum()); cnt[i] += int(valid.sum())
            cur_hidden = h; cur_token = amax; cur_pos = cur_pos + 1
    a = [hit[i] / max(cnt[i], 1) for i in range(K)]
    al = 1.0; prod = 1.0
    for aj in a:
        prod *= aj; al += prod
    return a, al


# --------------------------------------------------------------------------------------------------
# v6 RESIDUAL-vs-NATIVE objective (the REAL single-stream lever). The adapter is wired as a RESIDUAL:
# serve does argmax(native_logits + alpha*adapter_logits). A standalone-CE adapter is a NO-OP when added to
# confident native logits (the binding test proved it: alpha=0 vs alpha=1 byte-identical). Train it instead
# to make the SUM rank the TARGET first over native's competing top-K tokens -> it learns a CALIBRATED shift
# that FIXES native-wrong cases without breaking native-right cases. Uses ONLY native DEPTH-1 top-K logits
# (captured in the v6 dump) -- the binding competitors. CARD-FREE-validated objective (fixes 92% of
# native-wrong, breaks 0% of native-right in the design check).
# --------------------------------------------------------------------------------------------------
def residual_loss(head, hid, seed_tok, label1, ntk_logits, ntk_ids, alpha, *, noise):
    """Depth-1 residual-vs-native CE. Candidate set per row = native top-K ids (PLUS the target if absent);
    candidate logit = native_logit + alpha * adapter_logit(adapter over full V, gathered at the candidates).
    CE toward the target's position in the candidate set. Rows with no native top-K (ntk_ids[:,0]<0) or no
    label (label1<0) or target not in top-K (can't form a finite native logit for it -> floor) are handled:
    target gets native logit = (min top-K logit - 5) floor when absent (the dump only kept top-K). Returns
    (loss, n_used)."""
    dev = head.embed_w.device
    B = hid.shape[0]
    valid = (label1 >= 0) & (ntk_ids[:, 0] >= 0)
    if not valid.any():
        return hid.new_zeros(()), 0
    hidv = hid[valid].to(dev); tokv = seed_tok[valid].to(dev)
    labv = label1[valid].to(dev); nidv = ntk_ids[valid].to(dev); nlgv = ntk_logits[valid].to(dev)
    pos0 = torch.zeros(hidv.shape[0], device=dev, dtype=torch.long)
    if noise > 0:
        sc = hidv.abs().mean(-1, keepdim=True)
        hidv = hidv + (torch.rand_like(hidv) * 2 - 1) * noise * sc
    # adapter's FULL-V logits at the seed (depth-1): adapter.compute_logits already folds its alpha gate; but
    # here we want the RAW adapter logits scaled by the SERVE alpha. The ResidualAdapter.compute_logits =
    # alpha_param * lm(hidden). To train against the serve's alpha knob we use the head's compute_logits and
    # treat the serve `alpha` as 1.0 during training (the head's own alpha_param is the learned scale).
    h1 = head.forward_hidden(hidv, tokv, pos0)
    ada_full = head.compute_logits(h1)                          # [Bv, V]  (== adapter residual logits)
    # build candidate set: native top-K ids, then append target column.
    cand_ids = nidv                                            # [Bv, Kn]
    ada_c = torch.gather(ada_full, 1, cand_ids.clamp_min(0))   # [Bv, Kn] adapter at native candidates
    nat_c = nlgv                                               # [Bv, Kn] native logits at its top-K
    sum_c = nat_c + ada_c                                      # serve alpha folded into the head's alpha_param
    # target column: adapter logit at target is exact; native logit at target = its top-K value if present
    # else a floor (min top-K - 5). in_topk: is the target among native's top-K?
    tgt = labv.unsqueeze(1)                                    # [Bv,1]
    in_topk = (cand_ids == tgt)                               # [Bv, Kn]
    nat_t_topk = (nat_c * in_topk).sum(1, keepdim=True)        # native logit at target IF in top-K
    floor = nat_c.min(dim=1, keepdim=True).values - 5.0
    nat_t = torch.where(in_topk.any(1, keepdim=True), nat_t_topk, floor)
    ada_t = torch.gather(ada_full, 1, labv.unsqueeze(1))
    sum_t = nat_t + ada_t                                     # [Bv,1]
    # if target IS in native top-K, mask that candidate column to -inf so it isn't double-counted as a
    # competitor (it's the label, represented by sum_t).
    sum_c_masked = sum_c.masked_fill(in_topk, float("-inf"))
    allc = torch.cat([sum_c_masked, sum_t], dim=1)            # [Bv, Kn+1], last col = target
    tgt_idx = torch.full((allc.shape[0],), allc.shape[1] - 1, device=dev, dtype=torch.long)
    ce = F.cross_entropy(allc, tgt_idx)
    # L2 on the adapter's residual AT THE CANDIDATE SET (incl. target) keeps it CALIBRATED to native's scale
    # -- without it the CE drives the residual + the alpha gate ever-larger, over-correcting and breaking
    # native-RIGHT cases (the in-training instability). This is the design-check's reg that made it fix 92%
    # of native-wrong while breaking 0% of native-right.
    reg = (ada_c.pow(2).mean() + ada_t.pow(2).mean())
    loss = ce + REG_W * reg
    return loss, int(valid.sum())


@torch.no_grad()
def residual_accept(head, HID, SEED, LAB1, NTL, NTI, bs=8192):
    """The TRUE serve metric at depth-1: argmax(native + adapter) == target, evaluated on native's top-K
    (the adapter can only move mass among/around native's candidates; outside-top-K flips are negligible).
    Returns (sum_accept, native_accept) over rows that have native top-K + a label."""
    head.eval()
    dev = head.embed_w.device
    sh = sn = tot = 0
    N = HID.shape[0]
    for s in range(0, N, bs):
        e = min(N, s + bs)
        lab = LAB1[s:e]; nid = NTI[s:e]; nlg = NTL[s:e]
        valid = (lab >= 0) & (nid[:, 0] >= 0)
        if not valid.any():
            continue
        hidv = HID[s:e][valid].to(dev); tokv = SEED[s:e][valid].to(dev)
        labv = lab[valid].to(dev); nidv = nid[valid].to(dev); nlgv = nlg[valid].to(dev)
        pos0 = torch.zeros(hidv.shape[0], device=dev, dtype=torch.long)
        h1 = head.forward_hidden(hidv, tokv, pos0)
        ada = head.compute_logits(h1)
        ada_c = torch.gather(ada, 1, nidv.clamp_min(0))       # [Bv, Kn]
        sum_c = nlgv + ada_c
        # native argmax among its top-K = column 0 (top-K is sorted desc) -> native token = nidv[:,0]
        nat_pick = nidv[torch.arange(nidv.shape[0], device=dev), 0]
        sum_pick = nidv[torch.arange(nidv.shape[0], device=dev), sum_c.argmax(1)]
        sn += int((nat_pick == labv).sum())
        sh += int((sum_pick == labv).sum())
        tot += int(valid.sum())
    return sh / max(tot, 1), sn / max(tot, 1)


def _train_verify(a, cfg, head):
    """Per-step verify-label training loop (native-selffeed). Mirrors the main loop's optimizer/early-stop
    but over per-step batches with target_verify labels. Returns when done (saves to a.out)."""
    steps = load_verify_steps(a.dumps, cfg.n_aux)
    if steps is None:
        return False  # old-schema dump -> fall back to the seq-contiguity path
    HID, SEED, VER, NTL, NTI = steps   # VER[N,K]=reconstructed greedy label; NTL/NTI=native top-K (v6) or None
    N = HID.shape[0]
    K = a.ttt_steps
    # MEASURED NATIVE accept floor: per-position greedy accept of the NATIVE head's OWN drafts vs the
    # reconstructed label (the serve metric, accept-len ~1.50). The bar to beat.
    from native_selffeed_accept import _load as _nsa_load, reconstruct_and_score as _nsa_score
    _tid, _nd, _pos, _sid, _Kd, _ = _nsa_load(a.dumps)
    nat_a, _nt, _ns, _nstep, _nrec, _ah = _nsa_score(_tid, _nd, _pos, _sid, K)
    nat_al = 1.0; _pr = 1.0
    for x in nat_a:
        _pr *= x; nat_al += _pr
    dev = a.device
    if dev != "cpu":
        HID = HID.to(dev); SEED = SEED.to(dev); VER = VER.to(dev)
        if NTL is not None:
            NTL = NTL.to(dev); NTI = NTI.to(dev)
    lab_frac = float((VER[:, 0].cpu() >= 0).float().mean())
    print(f"[ttt/verify] {N} per-step examples  K={K}  depth-1 labeled={lab_frac*100:.0f}% (reconstructed)",
          flush=True)
    print(f"[ttt/verify] MEASURED NATIVE floor (native_draft vs reconstructed greedy label): " +
          "  ".join(f"a{i+1}={nat_a[i]*100:.1f}%" for i in range(K)) +
          f"  accept-len={nat_al:.3f}  <-- THE BAR TO BEAT", flush=True)

    # ---- v6 RESIDUAL-vs-NATIVE training (the REAL lever) when native top-K + an ADAPTER are present ----
    # This trains argmax(native + adapter) == target (the TRUE serve objective), fixing native-wrong cases
    # without breaking native-right -- unlike the standalone verify-CE path (a NO-OP as a residual at serve).
    if NTL is not None and getattr(a, "adapter", False):
        nrows = int((NTI[:, 0] >= 0).sum())
        print(f"[ttt/verify] v6 RESIDUAL-vs-NATIVE objective: {nrows}/{N} rows have native top-K "
              f"(Kn={NTI.shape[-1]}). Training argmax(native+adapter)==target (depth-1).", flush=True)
        return _train_residual(a, cfg, head, HID, SEED, VER[:, 0], NTL, NTI, nat_a, nat_al)
    elif getattr(a, "adapter", False) and NTL is None:
        print("[ttt/verify] WARNING: --adapter but dump has NO native_topk_logits (v5 dump). The adapter "
              "will train as a STANDALONE predictor -> a NO-OP residual at serve (the binding-test failure). "
              "Re-dump with the v6 hook (native-logits capture) for a residual that actually lifts.",
              flush=True)
    _rnd.seed(0)
    idx = list(range(N)); _rnd.shuffle(idx)
    nval = max(1, N // 20)
    vi = torch.tensor(idx[:nval], device=HID.device); ti = torch.tensor(idx[nval:], device=HID.device)
    a0, al0 = verify_accept(head, HID[vi], SEED[vi], VER[vi], K)
    print(f"[ttt/verify] baseline (pre-TTT) self-feed accept: " +
          "  ".join(f"a{i+1}={a0[i]*100:.1f}%" for i in range(K)) + f"  accept-len={al0:.3f}", flush=True)
    if a.dry_run:
        # one batch forward+backward, print, exit
        head.train()
        for p in head.trainable_params():
            p.grad = None
        b = ti[:min(2048, len(ti))]
        loss, hit, cnt = verify_unroll_loss(head, cfg, HID[b], SEED[b], VER[b],
                                            torch.zeros(len(b), device=HID.device),
                                            ttt_steps=K, decay=a.decay, noise=a.noise,
                                            self_feed_noise=a.self_feed_noise, bptt=a.ttt_bptt)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in head.trainable_params()
                                  if p.grad is not None)))
        fin = all(torch.isfinite(p.grad).all() for p in head.trainable_params() if p.grad is not None)
        per = "  ".join(f"a{i+1}={100*hit[i]/max(cnt[i],1):.2f}%(n={cnt[i]})" for i in range(K))
        print(f"[ttt/verify/dry] loss={float(loss):.4f} grad-norm={gn:.3f} grads-finite={fin}", flush=True)
        print(f"[ttt/verify/dry] per-step accept: {per}", flush=True)
        print("[ttt/verify/dry] OK -- verify-label unroll is finite + differentiable; real run is on-card.",
              flush=True)
        return True
    opt = torch.optim.AdamW(head.trainable_params(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    BS = max(256, a.bs)
    best = al0; best_state = None; since = 0
    for ep in range(a.epochs):
        head.train()
        perm = ti[torch.randperm(len(ti), device=ti.device)]
        tot = 0.0; nb = 0
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss, _h, _c = verify_unroll_loss(head, cfg, HID[b], SEED[b], VER[b],
                                              torch.zeros(len(b), device=HID.device),
                                              ttt_steps=K, decay=a.decay, noise=a.noise,
                                              self_feed_noise=a.self_feed_noise, bptt=a.ttt_bptt)
            loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(head.trainable_params(), a.clip)
            opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1
        a_ep, al_ep = verify_accept(head, HID[vi], SEED[vi], VER[vi], K)
        if al_ep > best:
            best = al_ep; since = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()
                          if "embed_w" not in k and "lm_head_w" not in k}
        else:
            since += 1
        astr = "  ".join(f"a{i+1}={a_ep[i]*100:.1f}%" for i in range(K))
        print(f"[ttt/verify] epoch {ep+1}/{a.epochs} loss={tot/max(nb,1):.3f} {astr} accept-len={al_ep:.3f} "
              f"best={best:.3f}", flush=True)
        if a.patience and since >= a.patience:
            print(f"[ttt/verify] early-stop (no val improvement in {a.patience} epochs)", flush=True)
            break
    if best_state is not None:
        head.load_state_dict(best_state, strict=False)
        print(f"[ttt/verify] restored best-val head (accept-len={best:.3f})", flush=True)
    cfg_d = json.load(open(a.config))
    torch.save({"cfg": cfg_d, "state": {k: v for k, v in head.state_dict().items()
                if "embed_w" not in k and "lm_head_w" not in k}}, a.out)
    print(f"[ttt/verify] saved native-TTT head -> {a.out}", flush=True)
    return True


def _train_residual(a, cfg, head, HID, SEED, LAB1, NTL, NTI, nat_a, nat_al):
    """v6 RESIDUAL-vs-NATIVE training loop (depth-1). Optimizes argmax(native + adapter) == target via
    residual_loss; gates on residual_accept (the TRUE serve metric: sum vs native). Saves the head only if
    the SUM beats native on held-out (else the floor rule keeps the serve alpha=0 == native)."""
    N = HID.shape[0]
    _rnd.seed(0)
    idx = list(range(N)); _rnd.shuffle(idx)
    nval = max(1, N // 20)
    vi = torch.tensor(idx[:nval], device=HID.device); ti = torch.tensor(idx[nval:], device=HID.device)
    alpha = 1.0  # the SERVE alpha the adapter is trained for (the head's own alpha_param is the learned scale)

    sum0, natv = residual_accept(head, HID[vi], SEED[vi], LAB1[vi], NTL[vi], NTI[vi])
    print(f"[ttt/res] baseline (pre-train) depth-1: native a1={natv*100:.2f}%  sum(native+adapter) a1="
          f"{sum0*100:.2f}%  (residual must lift sum ABOVE native)", flush=True)
    if a.dry_run:
        head.train()
        for p in head.trainable_params():
            p.grad = None
        b = ti[:min(2048, len(ti))]
        loss, nu = residual_loss(head, HID[b], SEED[b], LAB1[b], NTL[b], NTI[b], alpha, noise=a.noise)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in head.trainable_params()
                                  if p.grad is not None)))
        fin = all(torch.isfinite(p.grad).all() for p in head.trainable_params() if p.grad is not None)
        print(f"[ttt/res/dry] loss={float(loss):.4f} grad-norm={gn:.3f} grads-finite={fin} n_used={nu}",
              flush=True)
        print("[ttt/res/dry] OK -- residual-vs-native loss finite + differentiable; real run is on-card.",
              flush=True)
        return True

    # FREEZE alpha at its init (1.0) during training: alpha is the SERVE knob, and a free alpha just sharpens
    # the train CE (ever-larger -> over-corrects, breaks native-right on val). The BODY learns the calibrated
    # residual at alpha=1; the serve sweeps HPU_MTP_ADAPTER_ALPHA. (The L2 reg keeps the body residual small.)
    head.alpha.requires_grad_(False)
    _tparams = [p for p in head.trainable_params() if p is not head.alpha]
    opt = torch.optim.AdamW(_tparams, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    BS = max(256, a.bs)
    best = sum0; best_state = None; since = 0
    for ep in range(a.epochs):
        head.train()
        perm = ti[torch.randperm(len(ti), device=ti.device)]
        tot = 0.0; nb = 0
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            loss, nu = residual_loss(head, HID[b], SEED[b], LAB1[b], NTL[b], NTI[b], alpha, noise=a.noise)
            if nu == 0:
                continue
            loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_(head.trainable_params(), a.clip)
            opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1
        sum_ep, nat_ep = residual_accept(head, HID[vi], SEED[vi], LAB1[vi], NTL[vi], NTI[vi])
        if sum_ep > best:
            best = sum_ep; since = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()
                          if "embed_w" not in k and "lm_head_w" not in k}
        else:
            since += 1
        print(f"[ttt/res] epoch {ep+1}/{a.epochs} loss={tot/max(nb,1):.3f}  native a1={nat_ep*100:.2f}%  "
              f"sum a1={sum_ep*100:.2f}%  best_sum={best*100:.2f}%  alpha={float(head.alpha):.2f}", flush=True)
        if a.patience and since >= a.patience:
            print(f"[ttt/res] early-stop (no sum-accept improvement in {a.patience} epochs)", flush=True)
            break
    if best_state is not None:
        head.load_state_dict(best_state, strict=False)
    # SHIP GATE: the SUM's depth-1 accept must beat native's depth-1. If not, the adapter is a no-op/regress
    # -> the serve floor (alpha=0 == native) holds; we still save (serve picks alpha), but flag LOUD.
    sum_f, nat_f = residual_accept(head, HID[vi], SEED[vi], LAB1[vi], NTL[vi], NTI[vi])
    verdict = "LIFTS native (ship; sweep serve alpha)" if sum_f > nat_f + 0.005 else \
        "NO-OP/REGRESS vs native (serve alpha=0 == native holds)"
    print(f"[ttt/res] FINAL depth-1: native a1={nat_f*100:.2f}%  sum a1={sum_f*100:.2f}%  -> {verdict}",
          flush=True)
    cfg_d = json.load(open(a.config))
    torch.save({"cfg": cfg_d, "state": {k: v for k, v in head.state_dict().items()
                if "embed_w" not in k and "lm_head_w" not in k}}, a.out)
    print(f"[ttt/res] saved residual adapter -> {a.out}", flush=True)
    return True


def main():
    global REG_W   # v6: set from --res-reg (must precede any REG_W use in this scope)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dumps", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4, help="lower than depth-1 LR: warm-starting a trained head")
    ap.add_argument("--bs", type=int, default=8192, help="(unused for SEQUENCE training; kept for parity)")
    ap.add_argument("--out", default="head_ttt.pt")
    ap.add_argument("--accum", type=int, default=16, help="gradient accumulation (sequences per optimizer step)")
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0=off)")
    ap.add_argument("--warmup", type=int, default=0, help="linear-warmup optimizer steps")
    ap.add_argument("--temp", type=float, default=1.0, help="distillation temperature (EAGLE/SpecForge T=1)")
    ap.add_argument("--hard-w", type=float, default=0.1, help="weight on hard CE when soft labels present")
    ap.add_argument("--noise", type=float, default=0.1, help="EAGLE feature-noise aug on the SEED hidden (train)")
    ap.add_argument("--self-feed-noise", type=float, default=0.0,
                    help="extra noise on the FED-BACK hidden at steps>=1 (0=off; serve has none)")
    ap.add_argument("--softk", type=int, default=16, help="top-K used for soft-KL (HASS optimum ~10)")
    ap.add_argument("--patience", type=int, default=6, help="early-stop patience on val accept-len (0=off)")
    # --- the TTT knobs ---
    ap.add_argument("--ttt-steps", type=int, default=4, help="k self-feed unroll steps (3-4; the cliff fix)")
    ap.add_argument("--decay", type=float, default=0.8, help="per-step loss decay 0.8**i (HASS default 0.8)")
    ap.add_argument("--ttt-bptt", action="store_true",
                    help="full BPTT through the unroll (keep fed-back hidden in graph). Default = HASS "
                         "stop-grad on the recursive feature (per-step loss, stable).")
    ap.add_argument("--warm-start", default="",
                    help="path to a trained depth-1 head .pt to initialize from (PRESERVES a1; TTT then "
                         "fixes a2..aK). Strongly recommended for the on-card run.")
    ap.add_argument("--device", default="cpu", help="cpu | hpu (HPU does the V-vocab matmul on the MME)")
    ap.add_argument("--max-seqs", type=int, default=0, help="cap #sequences (CPU dry-run; 0=all)")
    ap.add_argument("--adapter", action="store_true",
                    help="A.2 RESIDUAL ADAPTER mode: build a ResidualAdapter (DraftHead body + learned "
                         "alpha gate) instead of a standalone head. Its logits are ADDED to the NATIVE MTP "
                         "head's at serve (zero serve-alpha = native byte-exact). Train it on the per-step "
                         "target_argmax to LIFT the native floor (~1.50). Warm-start the BODY from head_d1.pt.")
    ap.add_argument("--alpha-init", type=float, default=1.0,
                    help="initial residual gate; FROZEN during v6 residual training (the serve sweeps "
                         "HPU_MTP_ADAPTER_ALPHA). The serve identity guarantee is alpha=0 == native.")
    ap.add_argument("--res-reg", type=float, default=REG_W,
                    help="v6 L2 weight on the adapter residual logits (keeps it calibrated to native's "
                         "scale; lift the residual fixes native-wrong without breaking native-right).")
    ap.add_argument("--self-test", action="store_true", help="random data, no checkpoint (pipeline smoke)")
    ap.add_argument("--dry-run", action="store_true",
                    help="card-free: run 1 forward+backward on a handful of seqs, print the unroll loss + "
                         "per-step accept, then EXIT without saving (validation only).")
    a = ap.parse_args()
    REG_W = a.res_reg   # v6: the residual-calibration L2 weight (used by residual_loss)

    cfg_d = json.load(open(a.config))
    cfg = HeadConfig(**{k: v for k, v in cfg_d.items() if k in HeadConfig.__dataclass_fields__})
    if a.ttt_steps < 1:
        raise ValueError(f"--ttt-steps must be >=1, got {a.ttt_steps}")

    if a.self_test:
        print("[ttt] SELF-TEST: stubbing frozen embed/lm_head (no checkpoint load)", flush=True)
        embed = torch.randn(cfg.vocab, cfg.hidden) * 0.02
        lm = embed.clone()
    else:
        print(f"[ttt] loading frozen embed/lm_head from {cfg_d['frozen_src']} ...", flush=True)
        embed, lm = _load_frozen_embed_lmhead(cfg_d)
    if a.adapter:
        from residual_adapter import ResidualAdapter
        head = ResidualAdapter(cfg, embed, lm, alpha_init=a.alpha_init)
        print(f"[ttt] A.2 RESIDUAL ADAPTER built (DraftHead body + alpha gate, alpha_init={a.alpha_init}); "
              "its logits ADD to the NATIVE MTP head at serve (serve alpha=0 == native byte-exact).",
              flush=True)
    else:
        head = DraftHead(cfg, embed, lm)

    if a.warm_start:
        import os as _os
        if not _os.path.isfile(a.warm_start):
            raise FileNotFoundError(f"[ttt] --warm-start head not found: {a.warm_start}")
        ck = torch.load(a.warm_start, map_location="cpu", weights_only=False)
        st = ck["state"] if "state" in ck else ck
        if a.adapter:
            # warm-start the BODY from a trained DraftHead (head_d1.pt). alpha is left at alpha_init.
            miss, unexp = head.load_body_warmstart(st)
            tnames = {n for n in dict(head.named_parameters())
                      if "embed_w" not in n and "lm_head_w" not in n and n != "alpha"}
            miss_t = [m for m in miss if m in tnames]
            if miss_t:
                raise AssertionError(f"[ttt] --adapter --warm-start is MISSING body params {miss_t[:8]} "
                                     f"(config/ckpt mismatch). Refusing partial warm-start.")
            print(f"[ttt] adapter BODY warm-started from {a.warm_start} (alpha={a.alpha_init}); the adapter "
                  "starts ~native and TTT lifts it.", flush=True)
        else:
            miss, unexp = head.load_state_dict(st, strict=False)
            tnames = {n for n in dict(head.named_parameters()) if "embed_w" not in n and "lm_head_w" not in n}
            miss_t = [m for m in miss if m in tnames]
            if miss_t:
                raise AssertionError(f"[ttt] --warm-start is MISSING trainable params {miss_t[:8]} for the head "
                                     f"built from {a.config} (config/ckpt mismatch). Refusing partial warm-start.")
            print(f"[ttt] warm-started from {a.warm_start} (a1 preserved; TTT fixes a2..aK)", flush=True)

    dev = a.device
    if dev == "hpu":
        import habana_frameworks.torch.core as _htcore  # noqa: F401  (registers the hpu backend)
        head = head.to("hpu")
        print("[ttt] training on HPU (MME does the vocab matmul)", flush=True)
    nparam = sum(p.numel() for p in head.trainable_params())
    print(f"[ttt] head built: hidden={cfg.hidden} vocab={cfg.vocab} mlp={cfg.mlp_dim} n_aux={cfg.n_aux} "
          f"n_blocks={cfg.n_blocks} trainable={nparam/1e6:.0f}M", flush=True)
    print(f"[ttt] UNROLL: ttt_steps={a.ttt_steps} decay={a.decay} grad={'BPTT' if a.ttt_bptt else 'HASS stop-grad'} "
          f"self_feed_noise={a.self_feed_noise}", flush=True)

    # ---- NATIVE-SELFFEED verify-label path (2026-06-23 fix) ----
    # If the dump carries target_verify (the fixed native_selffeed_dump), train PER STEP against the target's
    # per-slot argmax (the serve's acceptance label) -- NOT the broken seq-contiguity next-seed token.
    if not a.self_test and a.dumps:
        if _train_verify(a, cfg, head):
            return  # verify-label training ran (the correct native-head TTT); done.
        print("[ttt] dump has no target_verify (old schema) -> falling back to seq-contiguity unroll "
              "(NOTE: for a NATIVE-selffeed dump this is the BROKEN label path; re-dump with the fix).",
              flush=True)

    # ---- data ----
    if a.self_test:
        S = 96  # long enough for a few anchors past depth K
        hid = torch.randn(S, cfg.hidden) * 0.1
        tgt = torch.randint(0, cfg.vocab, (S,))
        seqs = [(hid, tgt, torch.arange(S), None, None)]
        have_topk = False
        print("[ttt] SELF-TEST on random data (pipeline smoke; accept will be ~chance)", flush=True)
    else:
        dumps = a.dumps if a.dumps else [cfg_d.get("default_dumps", "")]
        shards = _load_shards(dumps, n_aux=cfg.n_aux)
        seqs, have_topk = build_seqs(shards, cfg.vocab)
        if not seqs:
            raise AssertionError(f"[ttt] no usable sequences from --dumps {dumps} (need >=2 contiguous "
                                 f"positions per request). Check the glob / that shards have seq_ids+positions.")

    _rnd.seed(0); _rnd.shuffle(seqs)
    if a.max_seqs > 0:
        seqs = seqs[:a.max_seqs]
        print(f"[ttt] --max-seqs: capped to {len(seqs)} sequences", flush=True)
    nval = max(1, len(seqs) // 10)
    val_seqs, tr_seqs = seqs[:nval], seqs[nval:]
    if not tr_seqs:                                 # tiny dry-run: reuse val as train so we still exercise it
        tr_seqs = val_seqs
    if dev != "cpu":                               # move all seq tensors onto the device once
        def _mv(s):
            return tuple(x.to(dev) if x is not None else None for x in s)
        val_seqs = [_mv(s) for s in val_seqs]; tr_seqs = [_mv(s) for s in tr_seqs]
    npairs = sum(s[0].shape[0] - 1 for s in tr_seqs)
    print(f"[ttt] {len(seqs)} sequences, {npairs} anchors ({nval} val); SEQUENCE training (self-feed unroll)",
          flush=True)

    use_soft = have_topk and a.hard_w < 1.0
    T = max(1e-3, a.temp)
    SK = min(a.softk, 4096)
    print(f"[ttt] loss = {('soft-KL(T=%g)+%.2g*hardCE' % (T, a.hard_w)) if use_soft else 'hard-CE'}  "
          f"(per-step, decayed {a.decay}**i)", flush=True)

    _lkw = dict(ttt_steps=a.ttt_steps, decay=a.decay, use_soft=use_soft, T=T, hard_w=a.hard_w,
                SK=SK, noise=a.noise, self_feed_noise=a.self_feed_noise, bptt=a.ttt_bptt)

    # ---- DRY-RUN: card-free validation (1 step, print, exit) ----
    if a.dry_run:
        print("[ttt] === DRY-RUN (card-free): one forward+backward over the unroll, then exit ===", flush=True)
        head.train()
        for p in head.trainable_params():
            if p.grad is not None:
                p.grad = None
        ex = tr_seqs[: min(8, len(tr_seqs))]
        tot = 0.0
        agg_hit = [0] * a.ttt_steps; agg_cnt = [0] * a.ttt_steps
        for s in ex:
            loss, hit, cnt = ttt_loss(head, cfg, s, **_lkw)
            loss.backward()
            tot += float(loss.item())
            for i in range(a.ttt_steps):
                agg_hit[i] += hit[i]; agg_cnt[i] += cnt[i]
        gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in head.trainable_params()
                                     if p.grad is not None)))
        per = "  ".join(f"a{i+1}={100*agg_hit[i]/max(agg_cnt[i],1):5.2f}%(n={agg_cnt[i]})"
                        for i in range(a.ttt_steps))
        al = 1.0; prod = 1.0
        for i in range(a.ttt_steps):
            prod *= agg_hit[i] / max(agg_cnt[i], 1); al += prod
        finite = all(torch.isfinite(p.grad).all() for p in head.trainable_params() if p.grad is not None)
        print(f"[ttt/dry] seqs={len(ex)}  mean unroll loss={tot/max(len(ex),1):.4f}  grad-norm={gnorm:.3f}  "
              f"grads-finite={finite}", flush=True)
        print(f"[ttt/dry] self-feed accept (untrained/warm):  {per}", flush=True)
        print(f"[ttt/dry] derived accept-len = {al:.3f}  (1 + a1 + a1*a2 + ...)", flush=True)
        print("[ttt/dry] OK — unroll loss is finite + differentiable; the real --device hpu run is queued.",
              flush=True)
        return

    # ---- full training ----
    opt = torch.optim.AdamW(head.trainable_params(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    a0, best, _c0 = self_feed_accept(head, val_seqs, a.ttt_steps)
    print(f"[ttt] baseline self-feed (pre-TTT): " +
          "  ".join(f"a{i+1}={a0[i]*100:.1f}%" for i in range(a.ttt_steps)) +
          f"  accept-len={best:.3f}", flush=True)
    best_state = None
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
        else:
            prog = (nstep[0] - a.warmup) / max(1, total_steps - a.warmup)
            for g in opt.param_groups:
                g["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, prog))))
        opt.step(); opt.zero_grad(); nstep[0] += 1

    since_best = 0
    for ep in range(a.epochs):
        head.train(); _rnd.shuffle(tr_seqs); tot = 0.0; nb = 0; opt.zero_grad()
        for si, seq in enumerate(tr_seqs):
            loss, _h, _c = ttt_loss(head, cfg, seq, **_lkw)
            (loss / ACCUM).backward(); tot += loss.item(); nb += 1
            if (si + 1) % ACCUM == 0:
                _step()
        if nb % ACCUM != 0:
            _step()
        a_ep, al_ep, _c = self_feed_accept(head, val_seqs, a.ttt_steps)
        if al_ep > best:
            best = al_ep; since_best = 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()
                          if "embed_w" not in k and "lm_head_w" not in k}
        else:
            since_best += 1
        astr = "  ".join(f"a{i+1}={a_ep[i]*100:.1f}%" for i in range(a.ttt_steps))
        print(f"[ttt] epoch {ep+1}/{a.epochs}  loss={tot/max(nb,1):.3f}  {astr}  "
              f"accept-len={al_ep:.3f}  best={best:.3f}", flush=True)
        if a.patience and since_best >= a.patience:
            print(f"[ttt] early-stop: no val accept-len improvement in {a.patience} epochs", flush=True)
            break

    if best_state is not None:
        head.load_state_dict(best_state, strict=False)
        print(f"[ttt] restored best-val head (accept-len={best:.3f})", flush=True)
    if not a.self_test:
        torch.save({"cfg": cfg_d, "state": {k: v for k, v in head.state_dict().items()
                    if "embed_w" not in k and "lm_head_w" not in k}}, a.out)
        print(f"[ttt] saved TTT-trained head -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
