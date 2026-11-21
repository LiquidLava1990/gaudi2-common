/* libcsengine.h — MMEL (Modern-MoE Execution Layer) submit core, W1.
 * =============================================================================
 * A reusable C library that graduates the M0 PoC (cs_poc.c) device
 * command-submission path into a clean API for the vLLM-Gaudi worker. It owns
 * exactly one Gaudi-2 compute context (one /dev/accel/accelN fd) and exposes:
 *
 *   - open/close of a free card (skips EBUSY = ctx held by a live serve)
 *   - USER_ALLOC command-buffer allocation -> device VA (the static per-token CB)
 *   - kernel-built SIGNAL / WAIT CSs (on-device SOB-increment + monitor-arm +
 *     QMAN-FENCE) — the primitive that deletes the per-op host fence
 *   - host-side packet builders (gen_signal_cb / gen_wait_cb) for callers that
 *     want to splice the SAME monitor/fence packets into their own static CB
 *   - generic CS submit of a user-built CB on a queue (the engine-chain head)
 *   - STAGED submission (FIRST..LAST -> ONE host completion for an N-CS chain)
 *   - a non-sleeping host wait (timeout_us==0 => completion_done() poll) and a
 *     mapped-CQ-counter busy-poll wait (zero-ioctl steady state)
 *   - a per-token address patcher hook (in-place CB field patch before doorbell)
 *   - the M0 self-diagnosing device-state classifier (a) submit-bug vs (b) dirty
 *
 * SCOPE (W1): the SUBMIT CORE only. It does NOT compile kernels, does NOT
 * extract/relocate GC recipes, does NOT do collectives. Those are W2+ (see the
 * W2 scope report). Everything here is grounded in the on-box 1.24.0-1007 UAPI
 * (/usr/include/drm/habanalabs_accel.h) and driver source
 * (/usr/src/habanalabs-1.24.0-1007); the ABI is cross-checked by abi_check.
 *
 * THREADING: a cseng_ctx is NOT internally locked. The production model is ONE
 * device ctx driven from ONE submit thread (the vLLM worker's execution thread);
 * if you submit from multiple threads, serialize externally.
 *
 * ERROR MODEL: functions return 0 / a non-negative result on success and a
 * negative value on failure, with errno set and a one-line reason captured in
 * cseng_last_error(). The library NEVER calls exit() (unlike the PoC's DIE) — it
 * is a library; the caller decides. A separate cseng_die-style abort is the
 * caller's choice. Failures are still LOUD: cseng_last_error() always holds the
 * most recent reason + errno, and the device-state classifier prints the
 * (a)-vs-(b) diagnosis to stderr on a CS failure when asked.
 * =============================================================================
 */
#ifndef LIBCSENGINE_H
#define LIBCSENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ version */
#define CSENG_API_VERSION_MAJOR 1
#define CSENG_API_VERSION_MINOR 0

/* ----------------------------------------------------------------- queue ids */
/* Gaudi-2 queue ids (UAPI enum, habanalabs_accel.h:316). PDMA + compute only;
 * EDMA (8..15) is dram_bd and its CS-completion HBW write RAZWIs -> device reset
 * (RUN_ONCARD.md strike-3; gaudi2_test_queues() itself skips EDMA). */
#define CSENG_Q_PDMA_0_0  0
#define CSENG_Q_PDMA_1_0  4
#define CSENG_Q_MME_0_0   16
#define CSENG_Q_TPC_0_0   20

/* --------------------------------------------------------- opaque context */
typedef struct cseng_ctx cseng_ctx;

/* ----------------------------------------------------- device-VA buffers */
/* A USER_ALLOC command buffer: a page-aligned host buffer, MMU-mapped, whose
 * returned device VA is passed to the kernel as the CS chunk's cb_handle with
 * HL_CS_CHUNK_FLAGS_USER_ALLOC_CB (required on Gaudi-2 HW queues). `host` is
 * host-visible: write packets there, then submit cb.dev_va. `used` tracks the
 * bytes a builder has emitted so far. */
typedef struct {
	void     *host;      /* host-visible mapping (write packets here)        */
	uint64_t  dev_va;    /* device VA to pass as the CS chunk cb_handle      */
	uint32_t  size;      /* allocated bytes                                  */
	uint32_t  used;      /* bytes emitted so far (caller- or builder-managed) */
} cseng_cb;

/* A kernel-mapped CB used purely as a host-visible CQ-counter page for the
 * INTERRUPT_KERNEL_CQ busy-poll wait. `host` is the userspace mapping of the
 * device-mapped counters; `handle` is the CB handle for the wait ioctl. */
typedef struct {
	volatile uint32_t *host;    /* userspace view of the CQ counters page    */
	uint64_t           handle;  /* cb_handle for HL_WAIT_CS cq_counters_handle */
	uint64_t           dev_va;  /* device VA of the counters CB              */
	uint32_t           size;
} cseng_cq;

/* result of a SIGNAL CS: the host-waitable sequence + the SOB this signal
 * increments on-device (sob_off is the offset returned by the kernel; a
 * consumer WAIT references the signal by `seq`, not by sob_off, but sob_off is
 * exposed for callers that splice their own monitor-arm packets). */
typedef struct {
	uint64_t seq;        /* CS sequence for host_wait / as a WAIT dependency */
	uint32_t sob_off;    /* sob_base_addr_offset (from hl_cs_out)            */
	uint16_t sob_count_before;  /* sob_count_before_submission              */
} cseng_signal;

/* device-state classification (mirrors the PoC's (a)-vs-(b) diagnosis). */
typedef enum {
	CSENG_DEV_OK = 0,            /* OPERATIONAL                              */
	CSENG_DEV_SUBMIT_BUG,        /* (a) EFAULT/EINVAL, device stayed OK      */
	CSENG_DEV_NONOPERATIONAL,    /* (b) IN_RESET/NEEDS_RESET/MALFUNCTION or  */
	                             /*     EBUSY/EPERM or a reset counter ticked */
	CSENG_DEV_UNKNOWN,
} cseng_dev_class;

/* ============================ open / close ============================== */
/* Open a free card. want_card<0 => scan accel0..7 and take the first that is
 * NOT EBUSY (i.e. no live compute ctx). Returns a ctx or NULL (errno set,
 * reason in cseng_last_error). On success the ctx holds the single compute ctx
 * for that device until cseng_close(). Captures baseline device status + reset
 * counters so later CS failures can be classified. */
cseng_ctx *cseng_open(int want_card);

/* Adopt an ALREADY-OPEN /dev/accel fd (e.g. a worker's live compute ctx) instead
 * of opening a free card — for in-serve use where every card is EBUSY. The caller
 * retains fd ownership: cseng_close does NOT close a borrowed fd. Returns NULL +
 * cseng_global_last_error() on a non-usable fd. */
cseng_ctx *cseng_attach(int fd, int card_index);

/* Close the ctx: unmaps everything the library allocated, frees the card. */
void cseng_close(cseng_ctx *c);

/* The fd / card index (for callers that must interleave their own ioctls on the
 * SAME ctx, e.g. a future SCAL/Synapse path sharing this device). */
int  cseng_fd(const cseng_ctx *c);
int  cseng_card_index(const cseng_ctx *c);

/* Last error string (always valid for a non-NULL ctx; "" if none). For a failed
 * cseng_open (ctx==NULL) use cseng_global_last_error(). */
const char *cseng_last_error(const cseng_ctx *c);
const char *cseng_global_last_error(void);

/* ====================== command-buffer allocation ====================== */
/* Allocate a USER_ALLOC CB of `size` bytes (rounded up to PAGE_SIZE by the
 * driver). Returns 0 and fills *out on success; <0 on failure. The CB is
 * tracked by the ctx and freed at cseng_close (or explicitly via cseng_free_cb).
 * This is the "static per-token CB": allocate once at warmup, patch addresses
 * per token. */
int  cseng_alloc_cb(cseng_ctx *c, uint32_t size, cseng_cb *out);
void cseng_free_cb(cseng_ctx *c, cseng_cb *cb);

/* Allocate + MMU-map device DRAM scratch; returns the device VA (0 on failure).
 * Tracked + freed at close. (Used for KV/activation scratch the static CB
 * addresses.) */
uint64_t cseng_alloc_dram(cseng_ctx *c, uint64_t size);

/* Allocate a kernel-mapped CQ-counters CB for the busy-poll wait. The counters
 * page is host-visible via out->host; the engine's CQ monitor increments a u32
 * there on completion, and cseng_cq_wait() spins on it with zero ioctls. */
int  cseng_alloc_cq(cseng_ctx *c, uint32_t n_counters, cseng_cq *out);
void cseng_free_cq(cseng_ctx *c, cseng_cq *cq);

/* ========================== on-device sync ============================= */
/* Submit a kernel-built SIGNAL CS on `queue_index`: increments that queue's
 * reserved SOB on-device when the queue's prior work drains. The CB is built
 * IN-KERNEL (gaudi2_gen_signal_cb), so no user CB hits the CP. Fills *out
 * (seq + sob_off). Returns 0 / <0. */
int  cseng_submit_signal(cseng_ctx *c, int queue_index, cseng_signal *out);

/* Submit a kernel-built WAIT CS on `queue_index` that arms an on-device monitor
 * on the SOB of `prev_seq` (a prior SIGNAL CS's sequence) and ends in a FENCE
 * that stalls this queue's CP on-device until the SOB hits target.
 * Returns:  1 = monitor ARMED (the device carries the dependency; *out_wait_seq
 *               is the WAIT CS's own host-waitable seq)
 *           0 = the signal had ALREADY completed (dependency met, nothing armed;
 *               driver convention: rc=0 + seq==ULLONG_MAX,
 *               command_submission.c:2367-2397)
 *          <0 = error (errno set). */
int  cseng_submit_wait_on(cseng_ctx *c, int queue_index, uint64_t prev_seq,
                          uint64_t *out_wait_seq);

/* Counters for the decisive M0-style assertion: how many WAITs actually armed an
 * on-device monitor vs. found the signal already done. A correct on-device-sync
 * run arms > 0. Reset with cseng_reset_armed_counters(). */
void cseng_get_armed_counters(const cseng_ctx *c, long *armed, long *already_done);
void cseng_reset_armed_counters(cseng_ctx *c);

/* =================== host-side sync packet builders ==================== */
/* Build the SAME monitor/fence packets the kernel emits, into a caller buffer,
 * so a caller can splice on-device sync between engine handoffs inside ONE
 * static CB (the W2 packet-splice path). These are byte-modeled on
 * gaudi2_gen_signal_cb / gaudi2_gen_wait_cb (gaudi2.c:11967 / 12060).
 *
 * cseng_gen_signal_cb: emit a MSG_SHORT that adds `inc` to the SOB at absolute
 *   config address `sob_cfg_addr`. Returns bytes written (or <0).
 * cseng_gen_wait_cb: emit the monitor-payload-program MSG_SHORTs + the monitor
 *   ARM (mode GREATER_OR_EQUAL on `sob_id` reaching `target`) + a FENCE on
 *   `fence_id` that stalls the consuming CP. Returns bytes written (or <0).
 *
 * NOTE (W1 honesty): these builders emit the packet ENCODINGS; programming
 * monitors by absolute CFG address from a user CB is the W2 packet-splice route
 * and is NOT exercised by the W1 on-card test (which uses the kernel SIGNAL/WAIT
 * CSs above — the supported, collision-free path). They are provided so W2 can
 * build on a verified encoder rather than re-deriving it. The exact monitor base
 * addresses come from HL_INFO_SYNC_MANAGER + the dcore0_sync_mngr_objs block
 * mapped via HL_CB/get-hw-block; the builder takes them as parameters. */
int  cseng_gen_signal_cb(void *buf, size_t buf_sz, uint64_t sob_cfg_addr, uint32_t inc);
int  cseng_gen_wait_cb(void *buf, size_t buf_sz,
                       uint64_t mon_arm_cfg_addr, uint64_t mon_payload_addr,
                       uint32_t mon_payload_data, uint32_t sob_id, uint32_t target,
                       uint32_t fence_id);

/* Convenience NOP packet (always-valid work item; isolates device-state from
 * packet content). Returns bytes written. */
int  cseng_emit_nop(void *buf, size_t buf_sz);

/* ============================ generic submit =========================== */
/* Submit a user-built CB (cb.dev_va, cb.used bytes) to `queue_index` as a
 * DEFAULT CS (cs_flags==0). Optionally host-wait it. Fills *out_seq.
 * Returns 0 / <0 (errno set, reason captured). do_wait uses a 1s timeout.
 *
 * WARNING: a bare user CB on a Gaudi-2 HW queue's CP is only safe if its packets
 * are ones the CP runs cleanly (MSG_SHORT/MON/FENCE/NOP families that
 * gaudi2_test_queues validates). A raw LIN_DMA/arbitrary program can fault the
 * CP -> device reset (cs_poc root-cause). The W1 on-card test submits a NOP CB
 * to MME queue 16 to PROVE queue 16 accepts a hand-built CB; production CBs come
 * from GC recipes (W2), not hand-built compute programs. */
int  cseng_submit_cb(cseng_ctx *c, int queue_index, const cseng_cb *cb,
                     bool do_wait, uint64_t *out_seq);

/* -------- staged submission: N CSs -> ONE host completion -------------- */
/* A staged sequence is: cseng_staged_begin() ; then >=1 cseng_staged_add()
 * (each puts one CB on one queue and is marked non-completing) ; then
 * cseng_staged_end() which submits the LAST CS (the only one that gets a host
 * completion) and returns its waitable seq. The whole chain drains with ONE
 * host wait on that seq. (cs_needs_completion() returns false for every
 * non-last staged CS — command_submission.c:279.)
 *
 * Usage:
 *   cseng_staged staged;
 *   cseng_staged_begin(c, &staged);
 *   cseng_staged_add(c, &staged, q0, &cb0);   // FIRST (set internally)
 *   cseng_staged_add(c, &staged, q1, &cb1);   // middle
 *   uint64_t seq;
 *   cseng_staged_end(c, &staged, q2, &cb2, &seq);  // LAST -> 1 completion
 *   cseng_host_wait(c, seq, 0);                // ONE wait for the whole chain
 *
 * The FIRST CS's seq is the staged-group sequence; subsequent CSs reference it
 * (hl_cs_in.seq) per the UAPI staged protocol (habanalabs_accel.h:1663-1674).
 * Each CB may carry its own on-device SIGNAL/WAIT (spliced or via separate
 * SIGNAL/WAIT CSs) so the engines self-sequence on-device. */
typedef struct {
	uint64_t first_seq;    /* the staged-group sequence (set at first add)   */
	int      n;            /* number of CSs added so far                     */
	bool     active;
} cseng_staged;

int  cseng_staged_begin(cseng_ctx *c, cseng_staged *s);
int  cseng_staged_add(cseng_ctx *c, cseng_staged *s, int queue_index,
                      const cseng_cb *cb);
int  cseng_staged_end(cseng_ctx *c, cseng_staged *s, int queue_index,
                      const cseng_cb *cb, uint64_t *out_seq);

/* ============================== waiting ================================ */
/* Host wait on a CS sequence.
 *   timeout_us == 0  => NON-SLEEPING completion_done() poll (one ioctl, returns
 *                       immediately whether or not complete; status tells which)
 *                       (hl_wait_for_fence, command_submission.c:2728).
 *   timeout_us  > 0  => interrupt-sleep wait up to timeout_us.
 * Returns: 1 = COMPLETED, 0 = not yet (only meaningful for timeout_us==0 / BUSY),
 *          <0 = error/timeout/aborted (errno set, reason captured). */
int  cseng_host_wait(cseng_ctx *c, uint64_t seq, uint64_t timeout_us);

/* Busy-poll a mapped CQ counter with ZERO ioctls in steady state: spins on
 * cq->host[counter_idx] >= target until the deadline. This is the production
 * steady-state wait (the doorbell + this poll = the 2 host round-trips/token).
 * Returns 1 = target reached, 0 = deadline hit, <0 = bad args. spin_us<=0 spins
 * forever (caller's responsibility to bound). */
int  cseng_cq_wait(cseng_ctx *c, const cseng_cq *cq, uint32_t counter_idx,
                   uint32_t target, long spin_us);

/* The kernel-assisted INTERRUPT_KERNEL_CQ wait: the driver compares the mapped
 * counter (cq->handle + offset) >= target and sleeps on the interrupt until
 * then (irq.c:354). Use when you WANT to sleep (not busy-spin) but still take
 * one completion for the whole step. timeout_us bounds the sleep.
 * Returns 1 = COMPLETED, 0 = timed out, <0 = error. */
int  cseng_cq_wait_interrupt(cseng_ctx *c, const cseng_cq *cq,
                             uint32_t counter_idx, uint32_t target,
                             uint64_t interrupt_id, uint64_t timeout_us);

/* ====================== per-token address patcher ===================== */
/* The static CB has a handful of per-token mutable tensor addresses (KV slot,
 * activation scratch, token-out). Register each as a patch site once; patch all
 * sites in place before the doorbell. A site = a 64-bit field at byte offset
 * `cb_byte_offset` inside `cb`. cseng_patch_apply writes new_dev_va there
 * (host-visible mapping, so the device sees it on the next submit).
 *
 * This is the hook W2 wires the recipe's launch-tensor addresses into. For the
 * raw-CB path it patches LIN_DMA/descriptor address fields directly; for the
 * SCAL/Synapse path the analogous mechanism is the synLaunchTensorInfo array
 * (see the W2 report) and this hook is bypassed in favor of that array. */
typedef struct {
	cseng_cb *cb;
	uint32_t  cb_byte_offset;   /* offset of the u64 address field in cb     */
	const char *name;           /* optional label for diagnostics            */
} cseng_patch_site;

/* Patch a single registered site in place. Returns 0 / <0 (bounds-checked). */
int  cseng_patch_apply(cseng_ctx *c, const cseng_patch_site *site,
                       uint64_t new_dev_va);

/* Patch many sites at once (the per-token call). Returns count patched, or <0. */
int  cseng_patch_apply_batch(cseng_ctx *c, const cseng_patch_site *sites,
                             size_t n, const uint64_t *new_dev_vas);

/* ===================== device-state diagnostics ======================= */
/* Read live device status + reset counters; classify (a)-vs-(b) for a failed CS.
 * cseng_classify_now prints the M0-style diagnosis block to stderr and returns
 * the classification. cs_errno is the errno captured from the failed CS submit
 * (pass 0 if classifying without a specific failure). */
cseng_dev_class cseng_classify_now(cseng_ctx *c, const char *what, int cs_errno);

/* Just the status enum (>=0) or <0 on ioctl failure; and the reset counts. */
int  cseng_device_status(cseng_ctx *c);
int  cseng_reset_counts(cseng_ctx *c, uint32_t *hard, uint32_t *soft);

/* Sync-manager info (first available user SOB/monitor/CQ — the W2 splice path
 * needs these to address monitors). Returns 0 / <0. */
int  cseng_sync_manager(cseng_ctx *c, uint32_t *first_sob, uint32_t *first_mon,
                        uint32_t *first_cq);

/* ============================== tally ================================= */
/* Per-ctx ioctl tally (CS / WAIT_CS / CB / MEM / INFO) — the byte-honest
 * round-trip counter that proved "1 host wait / N-op step" in M0. Reset +
 * snapshot for measuring the host round-trips of a submission shape. */
typedef struct { long cs, wait, cb, mem, info, other; } cseng_tally;
void cseng_tally_reset(cseng_ctx *c);
void cseng_tally_get(const cseng_ctx *c, cseng_tally *out);

#ifdef __cplusplus
}
#endif
#endif /* LIBCSENGINE_H */
