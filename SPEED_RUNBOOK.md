# Gaudi 2 — Full Speed Runbook (2026-06-05)
Big quantized model (Qwen3.5-397B-A17B-int4). Baseline TP=8 = 4.3 tok/s; free wins (live) = 5.8 tok/s.
Realistic 397B single-stream ceiling after ALL software fixes ≈ 7-12 tok/s (NVIDIA 50-100 needs MTP = dead on hybrids).
RULE: always stop the model with `./stop.sh` (safe_stop) — NEVER `docker rm -f` (affinity deadlock).
[C]=Claude builds it then you run the test line · [you]=copy-paste

## PHASE 0 — free wins  ✅ DONE (live in the running container)
TP=4 + --max-num-seqs 8 + RUNTIME_SCALE_PATCHING=0 + MAX_EXPERTS_PER_SLICE=64  →  4.3 → 5.8 tok/s.

## PHASE 1 — FIRMWARE: preboot SPI 1.14 → 1.24  [you · maintenance window · brick-risk · COLD cycle]
# 1. get the flashing tool (host has firmware images but not the loader)
sudo apt-get update && sudo apt-get install -y habanalabs-firmware-tools
command -v hl-fw-loader && hl-fw-loader --help          # ⚠️ READ flags here; they are authoritative

# 2. stop serving + record BEFORE state
cd $HOME/AI && sudo systemctl stop vllm-flagship 2>/dev/null; ./stop.sh
cat /sys/class/accel/accel*/device/preboot_btl_ver | sort -u                       # expect 1.14.0-fw-48.0.1
for d in /sys/class/infiniband/hbl_*; do echo "$(basename $d) p10=$(cat $d/ports/10/hw_counters/pcs_link_toggles)"; done

# 3. BEFORE fabric benchmark (so we MEASURE what the flash buys) — exact cmd from Claude when model is down
#    [C] hccl_demo all_reduce nranks=8 (small-msg latency + 256MB busbw)  -> record GB/s + us

# 4. free the cards (hl-fw-loader needs the driver detached; confirm via --help)
sudo modprobe -r habanalabs_compat habanalabs_en habanalabs_cn habanalabs_ib habanalabs   # if habanalabs_ib busy: sudo modprobe -r ib_uverbs first

# 5. FLASH  (⚠️ match the exact invocation to `--help`; never interrupt power)
sudo hl-fw-loader                                       # flashes SPI/preboot on all 8 from /lib/firmware/habanalabs

# 6. COLD power-cycle (a warm `reboot` will NOT reload SPI)
sync; sudo ipmitool chassis power cycle                 # BMC full off->on, box comes back up

# 7. VERIFY after reboot
cat /sys/class/accel/accel*/device/preboot_btl_ver | sort -u                       # expect 1.24.0-fw-62.6.2
for d in /sys/class/infiniband/hbl_*; do echo "$(basename $d) p10=$(cat $d/ports/10/hw_counters/pcs_link_toggles)"; done   # expect ~0 and NOT climbing
#    [C] AFTER hccl_demo (compare to step 3) + re-measure single-stream tok/s  = the firmware delta

## PHASE 2 — SOFTWARE WINS  [after firmware verified · precision-agnostic · biggest = lazy collectives]
2a [C] NUMA-pin + single-socket TP=4 wired into serve.sh   → [you] ./serve.sh qwen35-397b ; re-measure
2b [C] mark_step patch — conditionalize htorch.mark_step in HpuCommunicator.all_reduce on is_lazy()
       (drops ~120 host-syncs/token)                       → [you] ./test_serve_plugin.sh ... ; A/B coherence + tok/s
2c [C] LAZY-MODE :next image (PT_HPU_LAZY_MODE=1 → HPU graphs + lazy collectives) — THE big multi-card win;
       built on a lazy-capable base, promoted only after a gate (HCCL world=8 + plugin re-validate + coherence)
2d [C] routed-gather MoE op in hpu-moe-ext — dequant only the ~10 routed experts (30,720 → ~600/token),
       bit-exact vs current op (keep zero=8 GPTQ fix), tested off the live box

## PHASE 3 — GLM-5.1  [Claude · after Phase 1-2 done + tested]
cd $HOME/AI && USE_DSA=1 RESTART_POLICY=no ./serve.sh glm51ar    # AutoRound int4 MoE + DSA plugin, TP=8
#  -> wait for warmup; coherence check; confirm DSA gate serves; bind Tailscale + api-key.
