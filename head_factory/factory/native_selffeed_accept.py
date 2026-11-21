"""Derive the NATIVE GLM-5.2 MTP head's per-depth GREEDY accept (the serve metric) from a v5 native-selffeed
dump (hpu_moe_ext/native_selffeed_dump.py). CARD-FREE: pure dump arithmetic, spec-decode-INDEPENDENT.

WHY v5 (2026-06-23): four rounds proved the spec-decode accept signal (target_verify/_async_spec_grid) is
an unreachable WALL on this HPU build. PIVOT: the dump now carries ONLY the model-output token stream that
is ALWAYS available -- per decode step, per request: seed_token (the committed token the head drafts from),
its TRUE absolute position, the native head's drafts native_draft[K], and a stable request id (seq_id).

THE KEY (temp=0 GREEDY): the committed token at any absolute position p IS the target's greedy argmax there.
The serve's "Per-position acceptance rate a_k" is exactly  P( native_draft[r,p,k-1] == token[r, p+k] ).
We reconstruct the GAPLESS per-request token stream token[r, p] off-card, then evaluate that directly.

RECONSTRUCTION (gapless, from seeds + drafts + absolute positions -- NO accepted-count capture needed):
  Group rows by seq_id, sort by absolute position. The captured seeds land at positions that increase by
  (1 + accepted) each step (each spec step commits 1+accepted tokens; only the LAST committed token per step
  is the next seed). For consecutive seeds at positions p < p_next in the same request:
    * token[r, p]      = seed_token at that row           (committed token at p, captured directly)
    * the gap p+1 .. p_next-1 are the ACCEPTED drafts of the step at p:
        token[r, p+1+j] = native_draft[r, p, j]   for j = 0 .. (p_next - p - 2)
      (greedy accept => an accepted draft EQUALS the committed token at that position).
  This fills EVERY committed position exactly once -> a gapless token[r, *]. (The last seed's tail past it
  is unknown -> those few positions are simply not used as label sources.)

  Then per-depth accept over all anchor rows (r, p) that HAVE a real token[r, p+k]:
    a_k = mean[ native_draft[r, p, k-1] == token[r, p+k] ]
  This is the serve's greedy criterion -> a1 must come out ~0.44 (matching the serve's 0.44/0.13/0.04).

  And label[r, p, k] = token[r, p+k] IS the adapter's TTT training target (the full greedy label, 100% of
  rows) -- ttt_train.py's verify path can build it the same way.

USAGE (card-free):
  python native_selffeed_accept.py --dumps '/mnt/data1/recipe_cache/mtp_selffeed/shard_*.pt'
"""
import argparse
import glob
import sys

import torch


def _load(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"[selffeed-accept] no shards matched --dumps {patterns} (0 files).")
    TID = []; ND = []; POS = []; SID = []
    K = None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        for need in ("target_ids", "native_draft", "positions", "seq_ids"):
            if need not in d:
                raise KeyError(f"[selffeed-accept] {f} missing '{need}' -- not a v5 native-selffeed dump "
                               f"(keys={list(d.keys())}). Re-dump with the v5 native_selffeed_dump.py "
                               "(token-stream capture: seed_token + abs position + native_draft).")
        nd = d["native_draft"]
        if nd.dim() != 2:
            raise ValueError(f"[selffeed-accept] {f} native_draft is {nd.dim()}-D {tuple(nd.shape)}, "
                             "expected 2-D [N,K].")
        if K is None:
            K = int(nd.shape[-1])
        elif int(nd.shape[-1]) != K:
            raise ValueError(f"[selffeed-accept] {f} native_draft K={nd.shape[-1]} != {K} (ragged).")
        TID.append(d["target_ids"].to(torch.int64))
        ND.append(nd.to(torch.int64))
        POS.append(d["positions"].to(torch.int64))
        SID.append(d["seq_ids"].to(torch.int64))
    return (torch.cat(TID), torch.cat(ND), torch.cat(POS), torch.cat(SID), K, len(files))


def reconstruct_and_score(tid, nd, pos, sid, depth, min_seq_len=2):
    """Reconstruct the gapless token stream per (seq_id) and score per-depth greedy accept. Returns
    (a[depth], tot[depth], n_seqs, n_anchor_steps, n_recovered_positions, accepted_hist)."""
    K = depth
    hit = [0] * K
    tot = [0] * K
    n_seqs = 0
    n_steps = 0
    n_recovered = 0
    accepted_hist = {}     # accepted-count distribution (derived from seed-position gaps), a sanity signal

    # group by seq_id
    order = torch.argsort(sid.to(torch.int64) * (10 ** 12) + pos.to(torch.int64))
    tid, nd, pos, sid = tid[order], nd[order], pos[order], sid[order]
    if sid.shape[0] == 0:
        return [0.0] * K, tot, 0, 0, 0, accepted_hist
    brk = (sid[1:] != sid[:-1]).nonzero().flatten() + 1
    bounds = [0] + brk.tolist() + [sid.shape[0]]

    for s, e in zip(bounds[:-1], bounds[1:]):
        L = e - s
        if L < max(2, min_seq_len):
            continue
        run_pos = pos[s:e]                 # absolute positions of this request's captured seeds (sorted)
        run_tid = tid[s:e]                 # committed token at each captured seed position
        run_nd = nd[s:e]                   # the head's drafts made AT each captured seed position
        # dedup identical positions defensively (a position should appear once; keep first)
        keep = torch.ones(L, dtype=torch.bool)
        keep[1:] = run_pos[1:] != run_pos[:-1]
        run_pos = run_pos[keep]; run_tid = run_tid[keep]; run_nd = run_nd[keep]
        L = run_pos.shape[0]
        if L < 2:
            continue
        n_seqs += 1

        # --- reconstruct token[p] for this request (gapless within [min_pos, last_seed_pos]) ---
        token = {}
        rp = run_pos.tolist(); rt = run_tid.tolist()
        for i in range(L):
            token[rp[i]] = rt[i]                          # seed token at its position
            if i + 1 < L:
                gap = rp[i + 1] - rp[i]                   # = 1 + accepted_at_step_i  (>=1)
                acc = gap - 1
                accepted_hist[acc] = accepted_hist.get(acc, 0) + 1
                # fill the accepted drafts into the gap positions p+1 .. p+gap-1
                maxj = min(acc, K)                        # native_draft has K slots; gap-1 <= K by construction
                for j in range(maxj):
                    token[rp[i] + 1 + j] = int(run_nd[i, j])
        n_recovered += len(token)

        # --- score per-depth accept: anchor = each captured seed position p; depth k checks draft[k-1] vs
        #     token[p+k] (only when token[p+k] exists in the reconstruction). This is the serve criterion.
        for i in range(L):
            p = rp[i]
            n_steps += 1
            for k in range(1, K + 1):
                lbl = token.get(p + k)
                if lbl is None:
                    continue                              # p+k not recovered (past the last seed) -> skip
                tot[k - 1] += 1
                if int(run_nd[i, k - 1]) == lbl:
                    hit[k - 1] += 1

    a = [hit[k] / max(tot[k], 1) for k in range(K)]
    return a, tot, n_seqs, n_steps, n_recovered, accepted_hist


def _accept_len(a):
    al = 1.0; prod = 1.0
    for aj in a:
        prod *= aj; al += prod
    return al


def _fit_dd(a):
    ratios = []
    for j in range(1, len(a)):
        if a[j - 1] > 1e-9 and a[j] > 1e-9:
            ratios.append(a[j] / a[j - 1])
    if not ratios:
        return float("nan")
    g = 1.0
    for r in ratios:
        g *= r
    return g ** (1.0 / len(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--depth", type=int, default=0, help="self-feed depth to report (0 = dump's K)")
    ap.add_argument("--min-seq-len", type=int, default=2, help="skip requests with < this many captured seeds")
    a = ap.parse_args()

    tid, nd, pos, sid, K, nfiles = _load(a.dumps)
    depth = a.depth if a.depth > 0 else K
    if depth > K:
        raise ValueError(f"[selffeed-accept] --depth {depth} > dump native_draft K={K}.")
    acc, tot, n_seqs, n_steps, n_rec, ahist = reconstruct_and_score(tid, nd, pos, sid, depth, a.min_seq_len)
    al = _accept_len(acc)
    dd = _fit_dd(acc)

    print(f"[selffeed-accept] {nfiles} shards  {n_seqs} requests  {n_steps} anchor steps  "
          f"{n_rec} reconstructed token positions  (native_draft K={K}, depth={depth})", flush=True)
    ahist_str = "  ".join(f"acc{k}={ahist.get(k,0)}" for k in sorted(ahist)[:8])
    print(f"[selffeed-accept] accepted-per-step (from seed-position gaps): {ahist_str}", flush=True)
    pos_str = "  ".join(f"a{k+1}={acc[k]*100:6.2f}% (n={tot[k]})" for k in range(depth))
    print(f"[selffeed-accept] NATIVE head per-position greedy accept (== serve metric):  {pos_str}", flush=True)
    print(f"[selffeed-accept] derived served accept-len = {al:.4f}  (= 1 + a1 + a1*a2 + ...)", flush=True)
    print(f"[selffeed-accept] fitted multiplicative drift dd = {dd:.4f}  "
          "(geo-mean a_{j+1}/a_j; the TTT lever)", flush=True)

    if depth >= 1 and n_steps > 0 and not (0.20 <= acc[0] <= 0.95):
        print(f"[selffeed-accept] WARN: measured a1={acc[0]*100:.1f}% is far outside any plausible band. "
              "Check the dump came from the fp8 MTP graft (NOT int4) and a healthy serve, and that "
              "'positions' are TRUE absolute positions (not a per-step counter -- a1~random => positions bad).",
              file=sys.stderr, flush=True)
    elif depth >= 1 and not (0.65 <= acc[0] <= 0.73):
        print(f"[selffeed-accept] NOTE: a1={acc[0]*100:.1f}% is outside the HEAD_TTT 0.65-0.73 bs=1 band. At "
              "max-num-seqs>1 a lower batched per-position rate (~0.44, matching the serve's logged 0.44) is "
              "EXPECTED, not a defect.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
