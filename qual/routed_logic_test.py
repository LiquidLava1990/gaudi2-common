"""Standalone CPU test of the routed-gather INDEX logic (no HPU, no model).
Replicates _routed_moe's present/active/remap math and checks, for every routed
slot, that it maps to the correct local expert (or is correctly skipped) vs a
brute-force reference. Finds index bugs in seconds instead of 20-min serve warmups."""
import torch

E = 128       # local experts per rank (EP=4 -> 512/4)
K = 16
top_k = 10
NUMG = 512


def routed_indices(topk_ids, emin):
    local = topk_ids.to(torch.int64) - emin
    flat = local.reshape(-1)
    valid = (flat >= 0) & (flat < E)
    flat_safe = torch.where(valid, flat, torch.zeros_like(flat))
    present = torch.zeros(E + 1, dtype=torch.float32)
    scatter_idx = torch.where(valid, flat, torch.full_like(flat, E))
    present.scatter_(0, scatter_idx, valid.to(torch.float32))
    present = present[:E]
    active = torch.topk(present, K).indices.to(torch.int64)
    remap = torch.full((E,), K, dtype=torch.int64)
    remap.scatter_(0, active, torch.arange(K, dtype=torch.int64))
    remapped = torch.where(valid, remap[flat_safe], torch.full_like(flat, K)).view_as(topk_ids)
    return active, remapped


def check(topk_ids, emin):
    active, remapped = routed_indices(topk_ids, emin)
    T, tk = topk_ids.shape
    bad = []
    for t in range(T):
        for j in range(tk):
            g = int(topk_ids[t, j]); r = int(remapped[t, j]); loc = g - emin
            is_local = 0 <= loc < E
            if is_local:
                if r >= K:
                    bad.append(f"local expert {loc} SKIPPED (r={r})")
                elif int(active[r]) != loc:
                    bad.append(f"local {loc} -> expert {int(active[r])} (r={r}) WRONG")
            else:
                if r < K:
                    bad.append(f"non-local {g} NOT skipped (r={r})")
    return bad


tot = 0
g = torch.Generator().manual_seed(0)
for trial in range(5000):
    emin = int(torch.randint(0, 4, (1,), generator=g)) * 128
    # distinct experts per token (real top-k is distinct)
    topk_ids = torch.stack([torch.randperm(NUMG, generator=g)[:top_k]])  # T=1, gated path
    b = check(topk_ids, emin)
    if b:
        tot += len(b)
        if tot <= 20:
            print(f"trial {trial} emin={emin}: {b[:3]}")
# targeted edge cases
print("--- edge: emin=256, local expert 0 (=256) routed + many non-local ---")
ec = torch.tensor([[256, 400, 257, 500, 261, 100, 265, 300, 257, 450]])
b = check(ec, 256); print("bad:", b)
# edge: ALL local (n_present near K)
print("--- edge: 10 distinct local experts on rank emin=0 ---")
ec2 = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
b2 = check(ec2, 0); print("bad:", b2)
# edge: zero local
print("--- edge: all non-local ---")
ec3 = torch.tensor([[200, 201, 202, 203, 204, 205, 206, 207, 208, 209]])
b3 = check(ec3, 0); print("bad:", b3)
print("TOTAL BAD across 5000 random T=1 trials:", tot)
