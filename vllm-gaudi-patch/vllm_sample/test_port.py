"""Numerical correctness test for the HPU torch port of rejection_sampler.py.

Stubs out vllm imports, forces HAS_TRITON=False so the torch fallbacks install,
loads the ACTUAL patched module, and compares the torch kernels against faithful
python-loop reimplementations of the original Triton kernels.
"""
import sys
import types
import importlib.util
import torch

PLACEHOLDER = -1
GREEDY_TEMP = 0


# ---- stub the vllm modules the file imports at top ----
def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _TL:
    class constexpr:  # only used as (unevaluated) annotation
        pass


def _jit(*args, **kwargs):
    # supports both @triton.jit and @triton.jit(do_not_specialize=[...])
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def deco(fn):
        return fn
    return deco


_triton = types.SimpleNamespace(jit=_jit)

_mod("vllm")
_mod("vllm.logger", init_logger=lambda *a, **k: types.SimpleNamespace(
    info=lambda *a, **k: None, warning=lambda *a, **k: None))
_mod("vllm.triton_utils", HAS_TRITON=False, tl=_TL, triton=_triton)
_mod("vllm.v1")
_mod("vllm.v1.outputs", LogprobsLists=object, LogprobsTensors=object,
     SamplerOutput=object)
_mod("vllm.v1.sample")
_mod("vllm.v1.sample.logits_processor")
_mod("vllm.v1.sample.logits_processor.builtin", MinTokensLogitsProcessor=object)
_mod("vllm.v1.sample.metadata", SamplingMetadata=object)
_mod("vllm.v1.sample.ops")
_mod("vllm.v1.sample.ops.bad_words", apply_bad_words_with_drafts=lambda *a, **k: None)
_mod("vllm.v1.sample.ops.penalties", apply_all_penalties=lambda *a, **k: None)
_mod("vllm.v1.sample.ops.topk_topp_sampler", apply_top_k_top_p=lambda l, k, p: l)
_mod("vllm.v1.sample.sampler", Sampler=object)
_mod("vllm.v1.spec_decode")
_mod("vllm.v1.spec_decode.metadata", SpecDecodeMetadata=object)
_mod("vllm.v1.spec_decode.utils",
     unconditional_to_conditional_rates=lambda x: x)

spec = importlib.util.spec_from_file_location("rej", "rejection_sampler.py")
rej = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rej)
print("module loaded; HAS_TRITON path installed:",
      type(rej.expand_kernel).__name__ == "_FuncWrapper")


# ---------------- reference (loop) implementations ----------------
def ref_greedy(batch, cu, draft, argmax, bonus, is_greedy, max_spec):
    out = torch.full((batch, max_spec + 1), PLACEHOLDER, dtype=torch.int64)
    for r in range(batch):
        greedy = True if is_greedy is None else bool(is_greedy[r])
        if not greedy:
            continue
        s = 0 if r == 0 else int(cu[r - 1])
        e = int(cu[r])
        n = e - s
        rejected = False
        for pos in range(n):
            if not rejected:
                tid = int(argmax[s + pos])
                rejected = int(draft[s + pos]) != int(argmax[s + pos])
                out[r, pos] = tid
        if not rejected:
            out[r, n] = int(bonus[r])
    return out


def ref_random(batch, cu, draft, draft_probs, target_probs, bonus, recovered,
               uniform, is_greedy, vocab, no_draft_probs):
    out = torch.full((batch, MAXSPEC + 1), PLACEHOLDER, dtype=torch.int64)
    for r in range(batch):
        if bool(is_greedy[r]):
            continue
        s = 0 if r == 0 else int(cu[r - 1])
        e = int(cu[r])
        n = e - s
        rejected = False
        for pos in range(n):
            if not rejected:
                did = int(draft[s + pos])
                u = float(uniform[s + pos])
                if no_draft_probs:
                    dp = 1.0
                else:
                    dp = float(draft_probs[s + pos, did])
                tp = float(target_probs[s + pos, did])
                accepted = dp > 0 and tp / dp >= u
                if accepted:
                    tid = did
                else:
                    rejected = True
                    tid = int(recovered[s + pos])
                out[r, pos] = tid
        if not rejected:
            out[r, n] = int(bonus[r])
    return out


def ref_expand(output_len, x, cu, replace_from, replace_to):
    out = torch.empty(output_len, dtype=x.dtype)
    for r in range(x.shape[0]):
        s = 0 if r == 0 else int(cu[r - 1])
        e = int(cu[r])
        v = x[r]
        if v == replace_from:
            v = replace_to
        out[s:e] = v
    return out


def ref_recovered(batch, cu, draft, draft_probs, target_probs, inv_q, vocab,
                  no_draft_probs):
    num_tokens = draft.shape[0]
    out = torch.zeros(num_tokens, dtype=torch.int64)
    for r in range(batch):
        s = 0 if r == 0 else int(cu[r - 1])
        e = int(cu[r])
        for pos in range(e - s):
            ti = s + pos
            if no_draft_probs:
                prob = target_probs[ti].clone()
                prob[int(draft[ti])] = 0.0
            else:
                prob = torch.clamp(target_probs[ti] - draft_probs[ti], min=0.0)
            score = prob * inv_q[r]
            out[ti] = int(torch.argmax(score))
    return out


# ---------------- build random cases ----------------
torch.manual_seed(0)
VOCAB = 64
MAXSPEC = 5
fails = 0
ncases = 400

for case in range(ncases):
    batch = int(torch.randint(1, 6, ()))
    counts = torch.randint(0, MAXSPEC + 1, (batch,))
    cu = torch.cumsum(counts, 0).to(torch.int32)
    num_tokens = int(cu[-1]) if batch else 0
    draft = torch.randint(0, VOCAB, (num_tokens,), dtype=torch.int32)
    argmax = torch.randint(0, VOCAB, (num_tokens,), dtype=torch.int64)
    bonus = torch.randint(0, VOCAB, (batch, 1), dtype=torch.int32)

    # ---- greedy (all greedy: is_greedy=None) ----
    out = torch.full((batch, MAXSPEC + 1), PLACEHOLDER, dtype=torch.int32)
    rej.rejection_greedy_sample_kernel[(batch,)](
        out, cu, draft, argmax, bonus, None, MAXSPEC)
    exp = ref_greedy(batch, cu, draft, argmax, bonus, None, MAXSPEC)
    if not torch.equal(out.to(torch.int64), exp):
        fails += 1
        print("GREEDY(all) mismatch case", case)

    # ---- mixed greedy + random ----
    is_greedy = torch.rand(batch) < 0.5
    # target probs (softmax of random logits), draft_probs=None (MTP path)
    tlogits = torch.randn(max(num_tokens, 1), VOCAB)
    target_probs = torch.softmax(tlogits[:num_tokens] if num_tokens else tlogits,
                                 dim=-1) if num_tokens else torch.zeros(0, VOCAB)
    if num_tokens == 0:
        target_probs = torch.zeros(0, VOCAB)
    # recompute argmax from target for greedy rows to be realistic (not required)
    uniform = torch.rand(num_tokens, dtype=torch.float64)
    q = torch.empty(batch, VOCAB).exponential_()
    inv_q = q.reciprocal()
    recovered = torch.empty(num_tokens, dtype=torch.int32)
    if num_tokens:
        rej.sample_recovered_tokens_kernel[(batch, MAXSPEC)](
            recovered, cu, draft, None, target_probs, inv_q, VOCAB, 8192,
            NO_DRAFT_PROBS=True)
        exp_rec = ref_recovered(batch, cu, draft, None, target_probs, inv_q,
                                VOCAB, True)
        if not torch.equal(recovered.to(torch.int64), exp_rec):
            fails += 1
            print("RECOVERED(no_draft) mismatch case", case)

    out = torch.full((batch, MAXSPEC + 1), PLACEHOLDER, dtype=torch.int32)
    # greedy kernel writes greedy rows
    rej.rejection_greedy_sample_kernel[(batch,)](
        out, cu, draft, argmax, bonus, is_greedy, MAXSPEC)
    # random kernel writes random rows
    rej.rejection_random_sample_kernel[(batch,)](
        out, cu, draft, None, target_probs, bonus, recovered, uniform,
        is_greedy, MAXSPEC, VOCAB, NO_DRAFT_PROBS=True)

    exp_g = ref_greedy(batch, cu, draft, argmax, bonus, is_greedy, MAXSPEC)
    exp_r = ref_random(batch, cu, draft, None, target_probs, bonus, recovered,
                       uniform, is_greedy, VOCAB, True)
    exp_mixed = torch.where(is_greedy.view(-1, 1), exp_g, exp_r)
    if not torch.equal(out.to(torch.int64), exp_mixed):
        fails += 1
        print("MIXED mismatch case", case)
        print(" is_greedy", is_greedy, "\n out", out, "\n exp", exp_mixed)

    # ---- random with draft_probs given ----
    if num_tokens:
        dlogits = torch.randn(num_tokens, VOCAB)
        draft_probs = torch.softmax(dlogits, dim=-1)
        all_random = torch.zeros(batch, dtype=torch.bool)
        recovered2 = torch.empty(num_tokens, dtype=torch.int32)
        rej.sample_recovered_tokens_kernel[(batch, MAXSPEC)](
            recovered2, cu, draft, draft_probs, target_probs, inv_q, VOCAB,
            8192, NO_DRAFT_PROBS=False)
        exp_rec2 = ref_recovered(batch, cu, draft, draft_probs, target_probs,
                                 inv_q, VOCAB, False)
        if not torch.equal(recovered2.to(torch.int64), exp_rec2):
            fails += 1
            print("RECOVERED(draft_probs) mismatch case", case)

        out2 = torch.full((batch, MAXSPEC + 1), PLACEHOLDER, dtype=torch.int32)
        rej.rejection_random_sample_kernel[(batch,)](
            out2, cu, draft, draft_probs, target_probs, bonus, recovered2,
            uniform, all_random, MAXSPEC, VOCAB, NO_DRAFT_PROBS=False)
        exp2 = ref_random(batch, cu, draft, draft_probs, target_probs, bonus,
                          recovered2, uniform, all_random, VOCAB, False)
        if not torch.equal(out2.to(torch.int64), exp2):
            fails += 1
            print("RANDOM(draft_probs) mismatch case", case)

    # ---- expand ----
    if batch:
        temp = torch.rand(batch, dtype=torch.float32)
        # force some greedy zeros
        temp[torch.rand(batch) < 0.4] = 0.0
        exp_temp = torch.empty(num_tokens, dtype=torch.float32)
        rej.expand_kernel[(batch,)](exp_temp, temp, cu, GREEDY_TEMP, 1,
                                    MAX_NUM_TOKENS=128)
        ref_t = ref_expand(num_tokens, temp, cu, GREEDY_TEMP, 1)
        if not torch.equal(exp_temp, ref_t):
            fails += 1
            print("EXPAND(temp) mismatch case", case)

        topk = torch.randint(1, VOCAB, (batch,), dtype=torch.int32)
        exp_k = torch.empty(num_tokens, dtype=torch.int32)
        rej.expand_kernel[(batch,)](exp_k, topk, cu, 0, 0, MAX_NUM_TOKENS=128)
        ref_k = ref_expand(num_tokens, topk, cu, 0, 0)
        if not torch.equal(exp_k, ref_k):
            fails += 1
            print("EXPAND(topk) mismatch case", case)

print(f"\n{ncases} cases run. FAILURES: {fails}")
print("RESULT:", "PASS" if fails == 0 else "FAIL")

# ---- determinism check at temp 0 (greedy): same inputs -> identical output ----
batch = 4
counts = torch.tensor([3, 0, 2, 5])
cu = torch.cumsum(counts, 0).to(torch.int32)
nt = int(cu[-1])
draft = torch.randint(0, VOCAB, (nt,), dtype=torch.int32)
argmax = draft.clone().to(torch.int64)
argmax[::2] = (argmax[::2] + 1) % VOCAB  # inject some mismatches
bonus = torch.randint(0, VOCAB, (batch, 1), dtype=torch.int32)
outs = []
for _ in range(5):
    o = torch.full((batch, MAXSPEC + 1), PLACEHOLDER, dtype=torch.int32)
    rej.rejection_greedy_sample_kernel[(batch,)](
        o, cu, draft, argmax, bonus, None, MAXSPEC)
    outs.append(o.clone())
det = all(torch.equal(outs[0], o) for o in outs)
print("GREEDY determinism (temp0):", "PASS" if det else "FAIL")


# ==========================================================================
# Padded-feed normalization test: rejection_sample() fed the HPU padded
# rectangular layout must produce IDENTICAL output to the native flat-ragged
# feed (greedy is RNG-free; random/mixed compared under a fixed seed).
# ==========================================================================
from types import SimpleNamespace  # noqa: E402


def make_sm(temps):
    t = torch.tensor(temps, dtype=torch.float32)
    return SimpleNamespace(
        all_greedy=bool((t == 0).all()),
        all_random=bool((t > 0).all()),
        temperature=t,
        generators={},
    )


def ragged_vs_padded(num_draft, temps, seed, stride, vocab=48):
    batch = len(num_draft)
    assert stride >= max(num_draft)
    max_spec = max(num_draft)
    sum_draft = sum(num_draft)
    g = torch.Generator().manual_seed(seed * 7 + 1)
    draft_ragged = torch.randint(0, vocab, (sum_draft,), dtype=torch.int32,
                                 generator=g)
    tl_ragged = torch.randn(sum_draft, vocab, generator=g)
    bonus = torch.randint(0, vocab, (batch, 1), dtype=torch.int32, generator=g)
    cu = torch.cumsum(torch.tensor(num_draft), 0).to(torch.int32)

    # Build the padded rectangular feed: request i occupies [i*stride, i*stride+
    # stride); the first num_draft[i] rows are the real values, the rest junk.
    draft_padded = torch.randint(0, vocab, (batch * stride,), dtype=torch.int32,
                                 generator=g)
    tl_padded = torch.randn(batch * stride, vocab, generator=g)
    k = 0
    for i, n in enumerate(num_draft):
        for j in range(n):
            draft_padded[i * stride + j] = draft_ragged[k]
            tl_padded[i * stride + j] = tl_ragged[k]
            k += 1
    draft_padded_2d = draft_padded.view(-1, 1)  # simulate runner's 2D feed

    torch.manual_seed(1234 + seed)
    out_r = rej.rejection_sample(
        draft_ragged.clone(), list(num_draft), max_spec, cu.clone(), None,
        tl_ragged.clone(), bonus.clone(), make_sm(temps))
    torch.manual_seed(1234 + seed)
    out_p = rej.rejection_sample(
        draft_padded_2d.clone(), list(num_draft), max_spec, cu.clone(), None,
        tl_padded.clone(), bonus.clone(), make_sm(temps))
    return out_r, out_p


pad_fails = 0
scenarios = [
    ([3, 3, 3], 3),        # all full-depth (padded == ragged)
    ([3, 0, 2, 1], 3),     # partial: a 0-draft (just-finished-prefill) req
    ([2, 0, 1], 3),        # max_spec (2) < stride (3)
    ([1, 1, 1, 1], 3),     # all short, stride bigger
    ([0, 0], 3),           # all zero-draft -> all bonus
    ([4, 2, 0, 4, 1], 4),  # bigger mixed
]
for si, (nd, stride) in enumerate(scenarios):
    batch = len(nd)
    temp_sets = {
        "greedy": [0.0] * batch,
        "random": [0.8] * batch,
        "mixed": [0.0 if b % 2 == 0 else 0.9 for b in range(batch)],
    }
    for name, temps in temp_sets.items():
        try:
            out_r, out_p = ragged_vs_padded(nd, temps, seed=si * 10, stride=stride)
        except Exception as e:
            pad_fails += 1
            print(f"PADDED {name} nd={nd} stride={stride} EXCEPTION: {e}")
            continue
        if not torch.equal(out_r, out_p):
            pad_fails += 1
            print(f"PADDED {name} nd={nd} stride={stride} MISMATCH")
            print(" ragged:\n", out_r, "\n padded:\n", out_p)

print(f"\nPadded-vs-ragged scenarios: {'PASS' if pad_fails == 0 else f'FAIL ({pad_fails})'}")
