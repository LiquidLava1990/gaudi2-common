# Head Factory — ONE tool, per-model fp8 MTP draft heads

**ONE config-driven factory trains a draft head for ANY model.** The `factory/` code is
model-agnostic; the per-model folders (`glm51/`, `m3/`, `kimi-k2.7/`) are just **output bins** —
each holds that model's small config (checkpoint path, hidden H, vocab, calibration set) + its
trained fp8 head artifact + accept-rate report. There is NOT a separate factory per model.

## Why per-model HEADS but ONE factory
A draft head is welded to its target by three things, so its *weights* are per-model:
- **hidden dim H** — `eh_proj` is 2H→H; GLM's H ≠ M3's ≠ Kimi's (shapes don't fit),
- **vocab + embedding + lm_head** — it emits logits over *that* model's vocabulary,
- **the target's output DISTRIBUTION** — its whole job is to predict what *this* model emits.

But the *machinery* (architecture-gen, dump, train, fp8-quant, serve) is identical across models
→ one factory, many heads.

## Within-family transfer (the GLM-5.2 win)
Same-family models (GLM-5.1 → GLM-5.2) share H/vocab/arch, so a head **warm-starts** (or may run
directly at reduced accept) on the next version. Train GLM-5.1's head now; it carries to 5.2.

## Target: ≥90% accept (depth-1)
Depth-1's ceiling is 2×; **90% accept ≈ 1.9×**. The grafted *original* MTP head floors at 62-75%
(~1.7×). To beat it the factory **TRAINS a dedicated head, distilled on the TARGET's own greedy
outputs** (the actual serving distribution) — not just graft. **fp8 throughout** (int4/fp8 rule;
bf16 banned — fp8 viability proven at 2.6% err vs int4's 16-38%, so fp8 heads stay coherent).

## Pipeline (`factory/`, model-agnostic, config-driven)
1. `gen_head.py`       — config → MTP head module (enorm/hnorm/eh_proj + 1 decoder layer + shared_head), sized to the model.
2. `dump_hiddens.py`   — run the target on calib data; dump (penultimate-hidden, next-token) pairs.
3. `train_head.py`     — train head to predict next-token from hidden; accept-maximizing loss (CE to target argmax + KL to target dist).
4. `quant_fp8.py`      — per-channel fp8 (e4m3) quant of the trained head.
5. `validate_accept.py`— CPU oracle: draft-vs-target argmax agreement on held-out (fabric-INDEPENDENT). **GATE: ≥90%**.
6. serve               — load the fp8 head as the spec-decode draft (generalized graft).

## Per-model status (output bins)
- `glm51/`     : fp8 viability ✓ (2.6%). Head UNTRAINED. Graft floor 62-75%. Target ≥90%.
- `m3/`        : 74 hidden-shards already dumped (minimax-m3/mtp). Head untrained. M3 ships 0 MTP weights → train from scratch.
- `kimi-k2.7/` : TBD — verify the K2.7 checkpoint has nextn/MTP layers + read its H/vocab first.

## SOTA recipe (EAGLE-3, NeurIPS 2025) — what the factory adopts
The draft head is a single lean transformer block over the TARGET's hidden states (we have this).
Three accept-boosting techniques to fold in:
1. **soft-KL distillation** — train to GLM's FULL next-token distribution (logits_main), not just argmax.
   IMPLEMENTED in factory/train.py (--temp, --hard-w); the dump already carries the target logits.
2. **multi-layer hidden fusion** — concat LOW+MID+HIGH target-layer hiddens (not just final). IMPLEMENTED
   (head.py n_aux=3 + fc/fc_norm; dump captures hidden_aux at layers (2,n//2,n-3)). **The aux was BROKEN
   by a chunked-prefill de-pad bug (FIXED 2026-06-20, decode_dump_patch.py): aux was sliced faux[:nreq]
   (first nreq rows of the FULL forward batch) instead of by the runner's logits_indices (the last-token
   rows) -> 2.4% unique rows. The prior session misread this as "HPU graph-buffer aliasing" and fell back
   to depth-1 (40% accept, lost to native). Now gathered by logits_indices + gated to decode steps; a
   RE-DUMP unblocks the real EAGLE-3 head.** Serve via method=eagle3 (glm52-eagle.env mounts a relaxed
   hpu_eagle.py so FactoryEagleForCausalLM passes the Eagle3LlamaForCausalLM isinstance assert). Runbook:
   head_factory/glm52/eagle3_runbook.sh.
3. **on-policy / training-time-test (TTT)** — feed the draft's own predictions back during training to
   kill the train/serve mismatch. on-policy data DONE (dump along GLM's own greedy gens, glm52-mtpdump);
   TTT (recursive draft feedback) [TODO, later].
Open frameworks for reference (CUDA-targeted; we train on the HPU box via this factory): SpecForge
(full spec-decode training framework), Draft-OPD (on-policy distillation).
CALIBRATION CORPUS: a HF code+reasoning+chat mix matched to the served workload (train on what you serve).

## Factory status
- factory/head.py    : config-driven DraftHead (frozen embed/lm_head + frontend + lean block). DONE.
- factory/train.py   : trainer w/ hard-CE + EAGLE-3 soft-KL distillation + accept validation. DONE.
- glm51/config.json  : GLM-5.1 head config (hidden 6144, vocab 154880, plain RMS, silu). DONE.
- DATA: 58-pos oracle cache = pipeline proof only. REAL head needs a DUMP (decode hook -> (hidden, logits)
  over a corpus). The dump is the next concrete step; runs on the serve (fabric-tolerant: throughput, not latency).
