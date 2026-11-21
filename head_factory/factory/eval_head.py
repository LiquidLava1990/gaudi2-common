"""Evaluate a trained DraftHead's accept on dumped sequences, in fp32 AND fp8.

The served head must run fp8 (no bf16). This confirms fp8-rounding the trainable linears
(eh_proj + lean block q/k/v/o/gate_up/down) preserves accept. Norms stay full-precision
(tiny); frozen embed/lm_head are the target's own int4 modules at serve time, not quantized here.

DEFAULT MODE (depth-1 teacher-forced) -- unchanged:
  python eval_head.py --config glm51/config.json --head glm51/head_v6_softkl.pt \
    --dumps '/mnt/data1/recipe_cache/mtp_snap_v6/*.pt'
  Reports argmax(head(hid[:-1], tgt[:-1])) == tgt[1:] : the head scored against the REAL
  dumped hidden/token at EVERY position (teacher-forced). This is NOT what serve does past depth-1.

SELF-FEED MODE (--self-feed --depth K) -- NEW, replicates the serve loop EXACTLY:
  python eval_head.py --config glm52/config.train.d1.host.json --head glm52/head_d1.pt \
    --dumps '/mnt/data1/recipe_cache/mtp_dump/*.pt' --self-feed --depth 4 --topk 8
  Serve self-feeds: draft pos-1 is seeded from the REAL dumped hidden; positions 2..K consume the
  head's OWN previous output hidden (forward_hidden_embeds) + the head's OWN argmax token (re-embedded
  via the target embed), NOT the dumped tgt. This is the GROUND TRUTH measured by
  vllm-gaudi-patch/v1/spec_decode/hpu_eagle_factoryeagle.py (per-draft self.model(...) feeding back
  hidden_states=ret_hidden_states + draft_token_ids=logits.argmax). Reports per-position self-feed
  accept a1..aK, derived served accept-len 1 + a1 + a1*a2 + ..., and top-{2,4,8} recall at pos-1
  (the quantity multi-candidate / HPU_MULTI_CAND monetizes). Both fp32 and served-precision fp8.
"""
import argparse, glob, json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from head import DraftHead, HeadConfig  # noqa: E402
from train import _load_frozen_embed_lmhead, _load_shards  # noqa: E402


def _fp8_round(w):  # per-output-channel scaled e4m3 round-trip (matches the served fp8 graft quant)
    amax = w.detach().abs().amax(0, keepdim=True).clamp_min(1e-8)
    s = amax / 448.0
    return ((w / s).to(torch.float8_e4m3fn).to(w.dtype)) * s


def _build_seqs(shards, V):
    HID = torch.cat([s[0] for s in shards]); TGT = torch.cat([s[1] for s in shards])
    POS = torch.cat([s[2] for s in shards]); SID = torch.cat([s[3] for s in shards])
    good = torch.isfinite(HID).all(-1) & (TGT >= 0) & (TGT < V)
    HID, TGT, POS, SID = HID[good], TGT[good], POS[good].long(), SID[good]
    order = torch.argsort(SID.long() * 100000 + POS)
    HID, TGT, POS, SID = HID[order], TGT[order], POS[order], SID[order]
    brk = ((SID[1:] != SID[:-1]) | (POS[1:] != POS[:-1] + 1)).nonzero().flatten() + 1
    bounds = [0] + brk.tolist() + [SID.shape[0]]
    return [(HID[s:e], TGT[s:e], POS[s:e]) for s, e in zip(bounds[:-1], bounds[1:]) if e - s >= 2]


def accept(head, seqs):
    head.eval(); ag = t = 0
    with torch.no_grad():
        for hid, tgt, pos in seqs:
            lg = head(hid[:-1], tgt[:-1], pos[:-1])
            ag += int((lg.argmax(-1) == tgt[1:]).sum()); t += tgt[1:].numel()
    return ag / max(t, 1)


def _assert_dims(head, seqs, cfg):
    """LOUD shape/dim contract between the loaded head, the config, and the dumped sequences.
    A dim mismatch (wrong dump for this head, or wrong config) MUST fail here, not silently
    produce garbage accept. Mirrors the serve-side asserts in hpu_eagle_factoryeagle.py."""
    if not seqs:
        raise AssertionError("[eval/self-feed] no usable sequences (need >=2 contiguous positions per "
                             "request). Check --dumps glob and that the shards have seq_ids/positions.")
    hid0 = seqs[0][0]
    H = cfg.hidden
    in_w = hid0.shape[-1]
    # n_aux>=3 heads consume the wide [.,n_aux*H] aux stack; n_aux==0 heads consume the [.,H] final hidden.
    exp_w = H * cfg.n_aux if cfg.n_aux else H
    if in_w != exp_w:
        raise AssertionError(
            f"[eval/self-feed] dump hidden width {in_w} != expected {exp_w} for this head "
            f"(hidden={H}, n_aux={cfg.n_aux}). The dump does not match the head: either n_aux is wrong "
            f"in the config or these are the wrong shards (aux head needs 'hidden_aux', depth-1 head "
            f"needs 'hidden').")
    # frozen embed/lm_head geometry (the head re-embeds its own argmax via embed_w == the target embed).
    ew = head.embed_w
    if tuple(ew.shape) != (cfg.vocab, H):
        raise AssertionError(f"[eval/self-feed] embed_w {tuple(ew.shape)} != (vocab={cfg.vocab}, hidden={H}); "
                             f"the frozen target embed did not load with the head's geometry.")
    lw = head.lm_head_w
    if lw.shape[-1] != H:
        raise AssertionError(f"[eval/self-feed] lm_head_w last dim {lw.shape[-1]} != hidden {H}.")


def self_feed_eval(head, seqs, depth, topk_list):
    """Replicate the vLLM serve self-feed loop EXACTLY (hpu_eagle_factoryeagle.py propose()), offline on CPU.

    Per dumped sequence we treat each position t (with a real next position t+1 in the dump) as a draft
    ANCHOR -- the committed context the serve loop drafts FROM -- and roll the head forward `depth` steps,
    feeding back its OWN output hidden + OWN argmax token, exactly as serve does:

      seed (draft pos-1, serve lines 100-120):
        in_hidden_1 = dumped hidden[t]          (the REAL target hidden; serve: target_hidden_states,
                                                 aux-fused via combine_hidden_states for n_aux>=3)
        in_token_1  = dumped tgt[t]             (the committed/sampled token at t; serve: target_token_ids)
        in_pos_1    = dumped pos[t]
        h1 = head.forward_hidden_embeds(in_hidden_1, embed(in_token_1), in_pos_1)   # head's own output hidden
        d1 = argmax(lm_head(h1))                # first DRAFT token
        a1 hit  <=>  d1 == tgt[t+1]            (the real next-next token)

      self-feed (draft pos j=2..depth, serve lines 199-221):
        in_hidden_j = h_{j-1}                   (the head's OWN previous output hidden, width H -> NOT re-fused)
        in_token_j  = d_{j-1}                   (the head's OWN previous argmax, re-embedded via embed_w ==
                                                 the target embed; serve re-embeds via self.model.embed_tokens)
        in_pos_j    = in_pos_1 + (j-1)          (serve advances positions by +1 each draft)
        h_j = head.forward_hidden_embeds(in_hidden_j, embed(in_token_j), in_pos_j)
        d_j = argmax(lm_head(h_j))
        a_j hit  <=>  d_j == tgt[t+j]          (requires t+j to exist in this contiguous sequence)

    SERVE-MATCH NOTES (verified against hpu_eagle_factoryeagle.py + factory_eagle.py + head.py):
      * forward_hidden_embeds is the SERVE core (factory_eagle.FactoryEagleForCausalLM.forward calls it);
        forward() (frozen embed_w) is the eval/train path and is NOT used at serve past the embed lookup.
      * serve re-embeds the self-fed token with the TARGET embed_tokens; embed_w here is loaded from the
        SAME model.embed_tokens.weight, so F.embedding(tok, embed_w) is numerically identical to serve.
      * the fed-back hidden at j>=2 has width H, so forward_hidden_embeds' width-detect does NOT re-run
        _fuse_aux (serve feeds ret_hidden_states straight back, also width H) -- matched.
      * positions=0 zeroes the embed (head.py line 144); identical here since we pass real positions.

    Each draft position is independent across anchors except for the feed-back chain WITHIN a depth roll,
    so we batch all anchors at a given draft depth into one forward (vectorized; numerically identical to
    per-anchor since LeanBlock is MLP-only / no cross-row attention for use_attn=False, and self-feed decode
    is per-row even with attention). Returns (per_pos_accept[1..depth], topk_recall{k: rate at pos-1}).
    """
    head.eval()
    dev = head.embed_w.device
    K = depth
    maxk = max(topk_list) if topk_list else 1
    # per-draft-position accept tallies
    hit = [0] * K            # hit[j] = a_{j+1} numerator (draft position j+1)
    tot = [0] * K            # tot[j] = a_{j+1} denominator (anchors that HAVE a real tgt[t+j+1])
    tk_hit = {k: 0 for k in topk_list}   # top-k recall numerator at pos-1
    tk_tot = 0                            # pos-1 denominator (== tot[0])

    with torch.no_grad():
        for hid, tgt, pos in seqs:
            n = tgt.shape[0]
            if n < 2:
                continue
            # Anchors: every position t with a real next token t+1 in this contiguous sequence. The draft
            # at depth j is checked against tgt[t+j] (only anchors with t+j < n contribute to a_j).
            # in_hidden / in_token / in_pos for the SEED draft (pos-1), batched over all anchors of this seq.
            A = n - 1                                   # number of anchors (t = 0..n-2)
            anchor_t = torch.arange(A, device=dev)      # [A]
            cur_hidden = hid[:A].to(dev)                # seed: REAL dumped hidden at each anchor   [A, in_w]
            cur_token = tgt[:A].to(dev)                 # seed: committed token at each anchor       [A]
            cur_pos = pos[:A].to(dev).long()            # seed: real position at each anchor         [A]
            for j in range(K):                          # draft position j+1
                e = F.embedding(cur_token, head.embed_w)        # serve re-embeds via target embed; embed_w == it
                h = head.forward_hidden_embeds(cur_hidden, e, cur_pos)   # head's OWN output hidden  [A, H]
                logits = head.compute_logits(h)                 # [A, vocab]
                argmax = logits.argmax(-1)                       # [A] the head's DRAFT token at this depth
                # check against the real token (t + j + 1); only anchors where that index exists count
                tgt_idx = anchor_t + (j + 1)
                valid = tgt_idx < n
                if valid.any():
                    real = tgt[tgt_idx.clamp(max=n - 1).cpu()].to(dev)   # tgt[t+j+1] for valid anchors
                    correct = (argmax == real) & valid
                    hit[j] += int(correct.sum())
                    tot[j] += int(valid.sum())
                    if j == 0:                                   # top-k recall measured ONLY at pos-1
                        tk_tot += int(valid.sum())
                        tk = logits.topk(maxk, dim=-1).indices    # [A, maxk]
                        in_topk = (tk == real.unsqueeze(-1))      # [A, maxk]
                        for k in topk_list:
                            rec = (in_topk[:, :k].any(-1)) & valid
                            tk_hit[k] += int(rec.sum())
                # FEED BACK the head's OWN hidden + OWN argmax for the next draft position (the serve loop).
                cur_hidden = h                          # width H -> not re-fused, matches serve
                cur_token = argmax
                cur_pos = cur_pos + 1                    # serve advances +1 each draft

    a = [hit[j] / max(tot[j], 1) for j in range(K)]
    recall = {k: tk_hit[k] / max(tk_tot, 1) for k in topk_list}
    return a, recall, tot, tk_tot


def _accept_len(a):
    """Served expected accept-len from per-position self-feed accept: 1 + a1 + a1*a2 + a1*a2*a3 + ...
    (the chain stops at the first reject, so depth j is only reached if all of a1..a_{j-1} hit)."""
    al = 1.0; prod = 1.0
    for aj in a:
        prod *= aj
        al += prod
    return al


def _fp8_round_head(head):
    """fp8-round every trainable Linear weight in place; return the count (the served-precision graft)."""
    n = 0
    with torch.no_grad():
        for _name, m in head.named_modules():
            if isinstance(m, nn.Linear) and m.weight is not None:
                m.weight.copy_(_fp8_round(m.weight)); n += 1
    return n


def _report_self_feed(tag, head, seqs, depth, topk_list):
    a, recall, tot, tk_tot = self_feed_eval(head, seqs, depth, topk_list)
    al = _accept_len(a)
    pos_str = "  ".join(f"a{j+1}={a[j]*100:5.2f}% (n={tot[j]})" for j in range(depth))
    rec_str = "  ".join(f"top-{k}={recall[k]*100:5.2f}%" for k in topk_list)
    print(f"[self-feed/{tag}] depth={depth}  {pos_str}", flush=True)
    print(f"[self-feed/{tag}] derived served accept-len = {al:.3f}  "
          f"(= 1 + a1 + a1*a2 + ...)", flush=True)
    print(f"[self-feed/{tag}] pos-1 recall (multi-cand ceiling, n={tk_tot}):  {rec_str}", flush=True)
    return a, al, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--self-feed", action="store_true",
                    help="serve-EXACT self-feed accept (head consumes its OWN prev hidden+argmax at "
                         "draft pos 2..K), instead of the default depth-1 teacher-forced accept.")
    ap.add_argument("--depth", type=int, default=4, help="self-feed draft depth K (only with --self-feed)")
    ap.add_argument("--topk", type=int, default=8,
                    help="max top-k for pos-1 recall report (recall reported for {2,4,..,topk}); --self-feed")
    ap.add_argument("--verify-dump", action="store_true",
                    help="NATIVE-SELFFEED eval (2026-06-23): the dump carries target_verify (the rejection "
                         "sampler's per-slot TARGET argmax). Report the TRAINED head's per-step self-feed "
                         "accept against target_verify (== the serve metric), fp32 + fp8. This is the CORRECT "
                         "ship gate for the native-head TTT (the old --self-feed seq-contiguity eval used the "
                         "subsampled seed stream and is WRONG for a native-selffeed dump).")
    a = ap.parse_args()

    # LOUD missing-file checks (no silent fallback): config, head ckpt, and at least one matched shard.
    if not os.path.isfile(a.config):
        raise FileNotFoundError(f"[eval] config not found: {a.config}")
    if not os.path.isfile(a.head):
        raise FileNotFoundError(f"[eval] head checkpoint not found: {a.head}")
    matched = [f for p in a.dumps for f in glob.glob(p)]
    if not matched:
        raise FileNotFoundError(f"[eval] no shards matched --dumps {a.dumps} (resolved to 0 files).")
    if a.self_feed and a.depth < 1:
        raise ValueError(f"[eval] --depth must be >=1, got {a.depth}")

    cfg_d = json.load(open(a.config))
    cfg = HeadConfig(**{k: v for k, v in cfg_d.items() if k in HeadConfig.__dataclass_fields__})
    embed, lm = _load_frozen_embed_lmhead(cfg_d)
    head = DraftHead(cfg, embed, lm)
    ckpt = torch.load(a.head, map_location="cpu", weights_only=False)
    if "state" not in ckpt:
        raise KeyError(f"[eval] head ckpt {a.head} has no 'state' (keys={list(ckpt.keys())}); not a "
                       f"factory head checkpoint.")
    st = ckpt["state"]
    # LOUD: the ckpt's trainable params must match the head built from --config (mismatched config silently
    # load_state_dict(strict=False)-skips and would measure a partly-random head).
    missing, unexpected = head.load_state_dict(st, strict=False)
    trainable_names = {n for n in dict(head.named_parameters()) if "embed_w" not in n and "lm_head_w" not in n}
    missing_trainable = [m for m in missing if m in trainable_names]
    if missing_trainable:
        raise AssertionError(
            f"[eval] head ckpt {a.head} is MISSING trainable params {missing_trainable[:8]}"
            f"{'...' if len(missing_trainable) > 8 else ''} for the head built from {a.config}. "
            f"Config/checkpoint mismatch (e.g. n_aux/n_blocks differ). Refusing to measure a partial head.")
    if unexpected:
        raise AssertionError(
            f"[eval] head ckpt {a.head} has UNEXPECTED params {unexpected[:8]}"
            f"{'...' if len(unexpected) > 8 else ''} not in the head built from {a.config}. "
            f"Config/checkpoint mismatch. Refusing to silently drop them.")
    V = cfg.vocab

    # -------- NATIVE-SELFFEED VERIFY-DUMP MODE: per-step accept vs target_verify (the serve metric) --------
    if a.verify_dump:
        from ttt_train import load_verify_steps, verify_accept, residual_accept  # reuse
        steps = load_verify_steps(a.dumps, cfg.n_aux)
        if steps is None:
            raise KeyError("[eval --verify-dump] not a v5/v6 token-stream native-selffeed dump. Re-dump.")
        HID, SEED, VER, NTL, NTI = steps
        K = a.depth
        # MEASURED native baseline (per-position greedy accept of native's OWN drafts vs the reconstructed
        # token stream). The apples-to-apples floor.
        from native_selffeed_accept import _load as _nsa_load, reconstruct_and_score as _nsa_score
        _tid, _nd, _pos, _sid, _Kd, _ = _nsa_load(a.dumps)
        nat_a, _nt, _nseq, _nstep, _nrec, _ah = _nsa_score(_tid, _nd, _pos, _sid, K)
        nat_al = 1.0; pr = 1.0
        for x in nat_a:
            pr *= x; nat_al += pr
        print(f"[eval --verify-dump] MEASURED NATIVE baseline (this dump): " +
              "  ".join(f"a{i+1}={nat_a[i]*100:.2f}%" for i in range(K)) + f"  accept-len={nat_al:.3f}",
              flush=True)
        # v6: if native top-K present AND this is an adapter, report the TRUE RESIDUAL serve metric at
        # depth-1: argmax(native+adapter) vs argmax(native). This is the binding lift (the standalone
        # self-feed accept below does NOT reflect the residual serve).
        if NTL is not None and hasattr(head, "alpha"):
            sumf, natf = residual_accept(head, HID, SEED, VER[:, 0], NTL, NTI)
            n = _fp8_round_head(head.body if hasattr(head, "body") else head)
            sum8, nat8 = residual_accept(head, HID, SEED, VER[:, 0], NTL, NTI)
            print(f"[eval --verify-dump] v6 RESIDUAL depth-1 (argmax(native+adapter) vs native):", flush=True)
            print(f"  fp32: native a1={natf*100:.2f}%  sum a1={sumf*100:.2f}%   "
                  f"fp8: native a1={nat8*100:.2f}%  sum a1={sum8*100:.2f}%  ({n} linears fp8)", flush=True)
            lift = sum8 > nat8 + 0.005
            print(f"[eval --verify-dump] SHIP GATE (residual depth-1): sum {sum8*100:.2f}% vs native "
                  f"{nat8*100:.2f}% -> {'LIFTS (ship; sweep serve alpha + verify on-card)' if lift else 'NO-OP/REGRESS -> serve alpha=0 == native (floor holds)'}",
                  flush=True)
            return
        # v5 fallback (no native top-K): standalone self-feed accept (NOTE: this is the proxy that did NOT
        # translate -- a standalone predictor is a no-op as a residual at serve).
        print(f"[eval --verify-dump] {HID.shape[0]} per-step examples  depth={K} (STANDALONE self-feed -- "
              "NOTE: no native top-K in this dump, so this is the proxy, NOT the residual serve metric)",
              flush=True)
        a32, al32 = verify_accept(head, HID, SEED, VER, K)
        print(f"[eval --verify-dump/fp32] " + "  ".join(f"a{i+1}={a32[i]*100:.2f}%" for i in range(K)) +
              f"  accept-len={al32:.3f}", flush=True)
        n = _fp8_round_head(head)
        a8, al8 = verify_accept(head, HID, SEED, VER, K)
        print(f"[eval --verify-dump/fp8]  " + "  ".join(f"a{i+1}={a8[i]*100:.2f}%" for i in range(K)) +
              f"  accept-len={al8:.3f}  ({n} linears fp8-rounded)  delta={(al8-al32):+.3f}", flush=True)
        print("[eval --verify-dump] (re-dump with v6 native-logits capture for the binding residual metric)",
              flush=True)
        return

    shards = _load_shards(a.dumps, n_aux=cfg.n_aux)
    seqs = _build_seqs(shards, V)
    print(f"[eval] {len(seqs)} seqs from {len(matched)} shards  "
          f"(hidden={cfg.hidden} vocab={V} n_aux={cfg.n_aux} n_blocks={cfg.n_blocks})", flush=True)

    if not a.self_feed:
        # -------- DEFAULT MODE: depth-1 teacher-forced (UNCHANGED behavior) --------
        a32 = accept(head, seqs)
        print(f"[eval] fp32 accept = {a32*100:.2f}%", flush=True)
        n = _fp8_round_head(head)
        a8 = accept(head, seqs)
        print(f"[eval] fp8  accept = {a8*100:.2f}%  ({n} linears fp8-rounded)  delta={(a8-a32)*100:+.2f}pp",
              flush=True)
        return

    # -------- SELF-FEED MODE: serve-exact roll-forward, fp32 then fp8 --------
    _assert_dims(head, seqs, cfg)
    topk_list = sorted({k for k in (2, 4, 8) if k <= a.topk} | ({a.topk} if a.topk >= 2 else set()))
    print(f"[eval] SELF-FEED depth={a.depth}  topk-recall={topk_list}  "
          f"(serve-exact: own hidden + own argmax fed back per draft pos)", flush=True)
    _report_self_feed("fp32", head, seqs, a.depth, topk_list)
    n = _fp8_round_head(head)
    print(f"[eval] fp8-rounded {n} trainable linears (served precision); re-measuring self-feed", flush=True)
    _report_self_feed("fp8", head, seqs, a.depth, topk_list)


if __name__ == "__main__":
    main()
