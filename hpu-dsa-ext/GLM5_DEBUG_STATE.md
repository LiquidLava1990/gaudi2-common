# GLM-5.1 — state + next steps (updated 2026-06-06, post scaleout-NIC reboot)

## ⚡ LATEST (2026-06-07 ~01:30) — SUPERSEDES the compile-boundary theory below
**DECISIVE:** GLM with `--enforce-eager` (NO graph compile) is STILL SALAD (`"andiandi…utti"`)
⇒ the bug is **NOT a torch.compile boundary (H1 DISPROVEN).** It is a NUMERICAL bug in the
**DENSE MLA PREFILL forward** (`HPUMLAImpl.forward_mha`, hpu_attn.py:245-307).
**SMOKING GUN** (`HPU_ATTN_PROBE=8`, layers 6-7, all 8 ranks): `q_norm≈100-136` but
`out_norm≈0.10-0.18`, `out_absmax≈0.005` ⇒ the attention OUTPUT is **~1000× too small =
NEAR-DEAD ATTENTION** ⇒ model ignores context ⇒ repetitive salad. (MoE is fine — residual
still grows 1.75→963→997 via MLP/MoE.) Different salad in eager vs compiled, same dead-attn cause.
**Now instrumenting** `forward_mha` (`mha_diag.py`, `HPU_MHA_DIAG=N`) to localize the collapse.
forward_mha path: `kv_b_proj(k_c_normed)`→split[k_nope192,V256], `k=cat(k_nope,k_pe)`,
`ops.prompt_attention(q,k,v_padded, is_causal=True, attn_bias=meta.attn_bias, scale=256**-0.5)`.
Candidate causes: (i) V/kv_b output tiny; (ii) DOUBLE-MASK (is_causal=True AND a non-None
attn_bias) zeroing scores; (iii) scale/softmax collapse → output≈mean(V)≈0; (iv) q/k near-orthogonal
(wrong rope/nope concat). Iterating in `--enforce-eager` (fast bring-up, no compile, same values).
RULED OUT: embed-NaN (fixed), DSA (off), dequant (bit-exact), MLA *math* per static audit
(splits/scale/RoPE/layout) — but a runtime magnitude collapse is NOT visible in static audit.

**🎯 ROOT CAUSE FOUND (2026-06-07 ~01:55) — `(1+weight)` RMSNorm mismatch.** mha_diag traced the collapse to
`k_c_normed` rms≈0.0009 / V rms≈0.0002 (q rms≈0.1). Checkpoint norm weights are stored as **(1+weight)
OFFSETS near 0**: `kv_a_layernorm` rms=0.006, `q_a_layernorm`=0.17, `input_layernorm`=0.04,
`post_attention_layernorm`=0.015 (`model.norm`=1.2). BUT vLLM registry maps
`GlmMoeDsaForCausalLM → ("deepseek_v2", …)` (`registry.py:124`; the class is a bare
`class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`) whose `RMSNorm` applies **PLAIN `weight`** (no +1).
So every norm scales by ~weight instead of (1+weight): ~26× too weak globally, **~167× too weak on the kv_a
latent** ⇒ K-content + V collapse ⇒ dead attention ⇒ salad. (kv_b_proj scales ≈ q_b_proj, so it can't
compensate; transformers `Glm4MoeRMSNorm` inits weight to `torch.ones` ⇒ a working GLM needs ~1 weights ⇒
near-0 stored ⇒ (1+weight) parameterization.) **FIX (under test): `hpu_dsa_ext/glm_norm_offset.py` adds 1.0
to every RMSNorm weight at load, gated `HPU_GLM_NORM_OFFSET=1`** (idempotent; greedy-safe for the final norm).
**RESULT (2026-06-07 ~02:08): the norm fix WORKED at the mechanism level but salad PERSISTS** — a THIRD bug.
With `HPU_GLM_NORM_OFFSET=1`: applied (1+weight) to 313 norms (first 0.17→1.17); mha_diag confirms attention is
NO LONGER DEAD — k_c_normed rms 0.0009→0.198, V rms 0.0002→0.047, OUT 0.0001→0.047 (≈230× up, OUT≈V so the
softmax-average is mechanically fine). BUT output is still incoherent (now VARIED/multilingual:
`" PH FletcherubbosselOwnership和弦alista…"` instead of pure repetition) ⇒ attention contributes but a 3rd bug
remains. RULED OUT for the 3rd bug: rope-interleave (deepseek MLA hardcodes is_neox_style=False = interleaved,
matches GLM rope_interleave=True), logit softcap (none in config). 3-AGENT DELIBERATION launched on the
remaining DeepseekV2-path deviations: (A) MoE routing (sigmoid/noaux_tc/routed_scaling_factor=2.5/group/
norm_topk_prob), (B) embedding+lm_head+scaling (embed NaN a symptom? tie_word_embeddings=False), (C) full
config audit (first_k_dense_replace=3, num_nextn_predict_layers=1 MTP layer mis-mapping, etc.). KEEP the norm
fix (confirmed real). Commit `HPU_GLM_NORM_OFFSET=1` to glm51ar.env once fully coherent.

**UPDATE (2026-06-07 ~03:05): 3 FIXES IN, STILL NOISE — a deeper forward bug.** Active fixes (all via
MODEL_ENV_EXTRA, plugins in hpu-dsa-ext): `HPU_SANITIZE_EMBED=1` (embed NaN), `HPU_GLM_NORM_OFFSET=1`
(`glm_norm_offset.py`, +1.0 to 312 per-layer GlmMoeDsaRMSNorm; model.norm currently EXCLUDED — but excluding
vs including didn't change coherence, both salad), `HPU_GLM_ROPE_NEOX=1` (`glm_rope_neox.py`, is_neox_style=True
— CONFIRMED correct: reference `apply_rotary_pos_emb` = rotate_half + emb=cat((freqs,freqs)) = NeoX; the
config rope_interleave / the "main uses interleaved" code comments are RED HERRINGS). RoPE flip barely changed
output ⇒ not the remaining bug. **Symptom now: output distribution is near-FLAT** (top token logprob -5.1 ≈
0.6%, top-10 bunched; " Paris" not in top-10) = the forward produces NOISE; final hidden DIRECTION is wrong
though all values finite/non-NaN. Layer probe (eager): residual 44→5316 over 78 layers (RMSNorm rescales each
step, so magnitude is NOT the bug — direction is); layer-0 dense attn_out 11.2 alive, layer OUT 477.
**PRIME SUSPECT = the int4 AutoRound MoE expert dequant / WNA16 compute** (75 MoE layers, the major component
NOT yet value-verified; MoE ROUTING already verified correct). Agent launched to numerically verify int4
expert dequant + a CPU dense-MLP/layer-0 reference (does dense MLP really amplify ~465×?). **NEXT decisive
step = value-level CPU reference comparison of one layer vs transformers GlmMoeDsaDecoderLayer** to find the
direction-corrupting op. NOTE: this checkpoint has NEVER been coherent on this setup (history of salad-fix
commits) — may be multi-bug and/or int4-lossy. Box note: fabric O0-O3 flapping; user said run anyway, box
recovers from each stop-wedge in seconds.


## TL;DR (updated 2026-06-06 ~20:55)
`GLM-5.1-int4-mixed-AutoRound` **LOADS + RUNS end-to-end at TP=8**. 
- ✅ **EMBED-NaN FIX VERIFIED WORKING** (real-data probe, this session): with
  `HPU_SANITIZE_EMBED=1` the residual stream is alive + NaN-free and norms GROW
  naturally with depth (layer5=1.75 → layer76=963 → layer77=997, all nan=False).
  The old NaN-poison signature (every layer frozen to identical norm=17.1114) is GONE.
- ❌ **STILL TOKEN-SALAD** → a SECOND, independent coherence bug downstream of the
  embedding. Real output: `" emulation Consortiecieciec Consortieciecieciec…tadtadrajiec"`.
  Finite/growing/non-NaN norms + garbage = wrong-but-finite attention.
  **DSA EXONERATED (agent-confirmed 2026-06-06):** sparse is HARD-BYPASSED — `FULL_ATTENTION_FALLBACK=True`
  is set unconditionally by `hpu_mla_skip_indexer.register()` (`hpu_mla_skip_indexer.py:58`, +`skip_topk=True` :74)
  and NOTHING (incl. HPU_DSA_READY=1) clears it; `sparse_mla_wire._sparse_forward_impl` early-returns the
  dense `_ORIG_FORWARD_IMPL` at `sparse_mla_wire.py:113-120`. So the run is **pure DENSE MLA + MoE** and the
  salad lives there.
  **3-AGENT AUDIT (2026-06-06) RULED OUT math/layout/dequant/DSA:** kv_b split offsets (192/256), softmax scale
  (`qk_head_dim=256**-0.5`, threaded correctly — the 256-vs-192 lead is a NON-bug), RoPE slice, per-head layout
  all CORRECT (Agent A full audit); bf16 dequant bit-exact (numcheck); indexer OFF (skip_topk forced True); no
  regression in b4691c7 (no-ops under PT_HPU_LAZY_MODE=0).
  **LEAD HYPOTHESIS = an HPU torch.compile GRAPH-BOUNDARY / device-residency around the bf16-dequant MLA linears
  — NOT a math bug.** The SAME `iecieciec` salad was previously caused by a `@torch.compiler.disable` on
  `GPTQDequantToBF16LinearMethod.apply` (since removed; salad persists ⇒ boundary still present via another
  route). Live MLA forward = `_src/vllm_gaudi/attention/oot_mla.py` `HPUMLAAttention.forward_impl` +
  `_src/.../backends/hpu_attn.py` `HPUMLAImpl.forward_mha/mqa` (plugin only dequants weights). NOTE `_src/` is a
  READ-ONLY reference; live code is `/opt/vllm-gaudi` in the image ⇒ any oot_mla fix must be a PLUGIN monkeypatch
  (sparse_mla_wire already wraps forward_impl).
  **CANDIDATE FIXES:** H2 = pin `W_UV`/`W_UK_T` `.contiguous().to('hpu')` (oot_mla.py:206-207 skips the
  `.to('hpu')` that hpu_attn.py:347-348 does); H1 = remove the residual compile boundary around the 5 MLA bf16
  linears (fused q_a/kv_a, q_b, kv_b, o_proj).
  **DIAGNOSTIC FIRST (one on-card run):** `-e HPU_ATTN_PROBE=4 -e HPU_MOE_PROBE=12` via MODEL_ENV_EXTRA (probe
  wired sparse_mla_wire.py:50-65) → MLA `out` finite-but-uncorrelated-with-q ⇒ boundary confirmed (do H1/H2);
  MLA out OK ⇒ MoE routing. **NOT offline-reproducible** (HPU-graph phenomenon; sims cover only
  dequant+finiteness). **Needs a healthy fabric ⇒ RESEAT O0–O3 FIRST.** Do NOT run old `glm51ar_probe_test.sh`
  (forbidden `docker rm -f`).
- 🔴 **FABRIC NOW BLOCKS PRACTICAL ITERATION:** the relocated O0–O3 scale-up fault
  (cards 1a/43, see [[gaudi-fabric-defect-rma]]) makes TP=8 collectives limp — full
  63-bucket warmup ~1–3 HRS; a single 16-tok request took 7m45s (mostly first-compile)
  and a 2nd request HUNG >120s at 0% util (stalled collective); each request storms the
  scaleout NICs ~10k events; and EVERY stop wedges the Habana cn (collective-network)
  driver work-queues in D-state (load spiked 84→127, HBM slow to free). **Reseat O0–O3
  (or UBB service) is on the GLM critical path before further bring-up iteration.**
- Probe bug fixed this session: `real_request_probe._probed_layer_fwd` made variadic
  (`*args`) — the old 4-positional sig crashed warmup on vllm 0.22.x ("takes 4 … 5 given").
- Use `VLLM_SKIP_WARMUP=true` for coherence iteration (one request compiles one shape;
  avoids the multi-hour warmup marathon + most fabric exposure).
- DSA built + synthetically validated on real Gaudi 2, gating UNCONFIRMED-active this run.

## ⚠️ Operational caveat (2026-06-06)
The box froze ~3h on 2026-06-06 from a **scaleout-NIC PHY fault on card
`0000:43:00.0`** (NOT the affinity deadlock). Cards are now kept **IDLE deliberately**;
`vllm-flagship` autostart **DISABLED**; a `softlockup_panic` auto-reboot watchdog was
added. GLM at TP=8 drives heavy card-to-card scaleout traffic = the exact stressor that
provoked the fault. Bring GLM up consciously and `dmesg -w | grep -iE 'cn[0-9]|phy|port 9|EQ event'`
during warmup. If it recurs on that card → reseat/RMA.

## DONE / VALIDATED
- **All GLM-5.1 LOAD-time bugs fixed** (committed, hpu-moe-ext `b4691c7` + earlier):
  - AutoRound int4-MoE dispatch + index-driven "stored-format-is-ground-truth".
  - MLA fused `q_a_proj`+`kv_a_proj_with_mqa`: config-side via
    `AutoRoundHPUConfig._FUSED_LEAF_TO_SHARDS` → fused layer builds real `.qweight`
    → `GPTQDequantToBF16LinearMethod`. (`mla_proj_patch.py` is the now-DISABLED,
    mutually-exclusive loader-interception alternative — do not re-enable.)
  - `indexer.wk` GPTQ dequant→fuse.
  - the old "worker dies at model_executor init, no traceback (hang)" is PAST.
- **DSA core validated on REAL Gaudi 2** (synthetic, seconds, no 744B load):
  indexer (lightning top-k) recall **0.99** / Jaccard 1.0; gather-consume vs CPU
  **0.011** (bf16); NaN-guard + sentinel-scatter + parity green.
  `tests/test_dsa_synthetic_hpu.py` PASSES on HPU.
- **Both plugins now in git**: hpu-moe-ext `b4691c7`, hpu-dsa-ext initial `9099922`.

## BLOCKED — coherence (the live frontier)
GLM-5.1 emits **token-salad**. The real-request layer probe (`HPU_REAL_PROBE=1`,
`real_request_probe.py` / `hidden_state_probe.py`) localized it precisely:
- **embedding output is already NaN at step 1** (`EMBED embed_out norm=nan nan=1`),
  BEFORE any decoder layer runs.
- the embed **WEIGHT is verified clean** (nan=0, inf=0) → the NaN is produced in the
  HPU TP=8 vocab-parallel masked-gather + all-reduce embedding forward (one element).
- that NaN poisons the residual stream; a downstream NaN-guard then freezes every
  layer to an identical `norm=17.1114` → the model emits salad.

**WORKAROUND WRITTEN, NOT VERIFIED:** `hpu_dsa_ext/nan_sanitize_embed.py` —
`nan_to_num` on the embed output, gated `HPU_SANITIZE_EMBED=1`. It is **NOT yet in
`models/glm51ar.env`** and no confirming run happened before the reboot.
**Secondary suspect** if salad persists after the embed fix: the `kv_b_proj`
MLA-forward path (the last `moe_wna16_hpu.py` edit references it in `forward_mha`).

## NEXT — one run (needs all 8 cards; they are idle now)
1. add `-e HPU_SANITIZE_EMBED=1` to `models/glm51ar.env` MODEL_ENV
   (`HPU_DSA_READY=1` is already set).
2. `HPU_REAL_PROBE=1 USE_DSA=1 RESTART_POLICY=no ./serve.sh glm51ar`
   — watch dmesg for scaleout-NIC faults during TP=8 warmup (see caveat).
3. read the probe output:
   - **embed NaN cleared + coherent text** → confirm DSA actually serves sparse,
     then bench + bind Tailscale/api-key. GLM-5.1 done (within ≤200K native ctx;
     true 1M needs a 1M-native model — see glm51-dsa-1m-requirement memory).
   - **still salad** → the `kv_b_proj`/MLA-forward path is the next fix.

## Parked
- **GLM-5.1-AWQ** (`glm51.env`): expert-parallel loader `IndexError` (global expert id
  indexes a 32-local param). AutoRound is the chosen path; revisit only if needed.
