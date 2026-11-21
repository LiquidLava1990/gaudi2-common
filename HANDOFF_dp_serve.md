# HANDOFF → other Claude window — wire dp_serve.sh into the throughput sweep

**Written 2026-06-04 by the session that built `dp_serve.sh`. Read this before editing any bench scripts so we don't clobber each other.**

## What I added (DONE — do not redo)
- **`$HOME/AI/dp_serve.sh`** — manual data-parallel serving (in-engine `--data-parallel-size`
  is dead on vllm-gaudi: cores never init, lazy-mode-gated, "CI-only"; the old in-engine `dp8_tp1`
  sweep variant timed out at 40min). Launches N independent `vllm serve --tensor-parallel-size 1`
  replicas pinned via `HABANA_VISIBLE_MODULES`, behind an **nginx least_conn** balancer on
  `BIND_IP:PORT` (<SERVER_IP>:8000). Layouts: `dp8_tp1` | `dp4_tp2` | `dp2_tp4`.
  - **Warm-first** (default `WARM_FIRST=1`): brings up r0 alone, polls `/health`, then fans out
    r1..rN-1 sharing r0's recipe cache **read-only** so they skip torch.compile. `WARM_FIRST=0` =
    all at once.
  - Usage: `./dp_serve.sh <model-key> [layout]` · stop: `./dp_serve.sh stop`
  - **Default layout is now `auto`**: measures the model's on-disk weight footprint and picks the
    smallest TP that fits one 96GB card (KV_RESERVE_GB headroom, default 30) → DP=8/TP. Verified
    picks: 16–23GB INT4→dp8_tp1, Qwen3.6-35B 67GB→dp4_tp2, 397B 211GB→dp2_tp4, GLM→dp1_tp8. Auto
    sizes on MEMORY FIT ONLY (no arch/divisibility/serve-bug awareness; warns on head/expert mod).
    **For the SWEEP, force each explicit layout** (dp8_tp1/dp4_tp2/dp2_tp4) to compare — don't use
    `auto` there. `auto` is for production serving.
- **`~/Desktop/Gaudi2_Optimization_Guide.md`** — added **Part III** (current state) + a **"DP throughput
  workflow"** section (anchor `#dp-workflow`). Corrected the stale `--data-parallel-size` advice in
  Parts I/II.

## What YOU should ADD (your domain — you own the live bench sweep)
Wire `dp_serve.sh` into the throughput sweep so we get the real **dp8_tp1 / dp4_tp2 / dp2_tp4 vs TP=8**
comparison that the in-engine attempt couldn't produce. Key requirements:

1. **Benchmark the BALANCER endpoint** `http://<SERVER_IP>:8000` (served-model-name from the
   model `.env`), NOT a single replica — we want *aggregate* throughput.
2. For each layout: `./dp_serve.sh <model> <layout>` → poll the LB `/health` until 200 → run the
   cycle at concurrency **1,10,25,50,100,200** → `./dp_serve.sh stop` → next layout.
3. Include the monolithic **TP=8 baseline** (`serve.sh <model>`) as the control row.
4. `max_num_seqs` in the `.env` is **per replica** → `dp8_tp1` = 8× aggregate slots. Note per-card KV:
   Qwen3.6-35B (~70 GB/card at TP=1) has little KV headroom at dp8_tp1 → `dp4_tp2` is the safer
   high-concurrency layout. Flag OOM/instability per layout rather than silently dropping it.
5. Put results in `bench/MASTER_REPORT.md` alongside the existing qwen36 TP=8 sweep (peak ~681 tok/s
   @c=50) so the layouts are directly comparable.

A ready snippet to adapt is in the guide's `#dp-workflow` section.

## Coordination
- I am **not** editing `bench/run_*.sh` or `MASTER_REPORT.md` — those are yours. I only created
  `dp_serve.sh`, `HANDOFF_dp_serve.md`, and edited the Desktop guide.
- If you change `dp_serve.sh`'s interface, update the guide `#dp-workflow` section to match.
