/* cs_poc.c — Rung-2 feasibility proof for Gaudi-2 single-stream.
 * =============================================================================
 * QUESTION (decisive): Does CS-level submission with ON-DEVICE sync (sync
 * objects + monitors) ELIMINATE the per-op host fence, or does the driver force
 * a host wait per CS completion?
 *
 * METHOD: build a multi-engine SYNC chain where engine B's QMAN blocks ON-DEVICE
 * on a monitor armed against engine A's sync-object, and drain it with exactly
 * ONE HL_WAIT_CS for the whole chain. Run it two ways and tally host ioctls:
 *
 *   MODE naive  : signalA ; WAIT ; signalB ; WAIT ; ...  (host fence per op:
 *                 mirrors the lazy-bridge status quo, ~135 CS/forward each
 *                 ending in an interrupt-sleep wait).
 *   MODE csengine: signalA ; B waits-on-device for A's SOB ; signalB ;
 *                 A waits-on-device for B's SOB ; ... ; host does 1 WAIT for the
 *                 tail.  => host fences per N-op step == 1.
 *
 * WHY SIGNAL/WAIT CSs (NOT a user-built DMA/NOP CB) ARE THE WORK ITEMS:
 *   The fence is a property of CS COMPLETION and is engine/work-agnostic, so any
 *   real CS that retires through the normal completion path answers the question.
 *   The SIGNAL and WAIT CSs are the RIGHT choice for TWO reasons:
 *     1. Their CBs are built ENTIRELY IN-KERNEL (gaudi2_gen_signal_cb /
 *        gaudi2_gen_wait_cb, gaudi2.c:11967 / 12060) from packets the hardware is
 *        guaranteed to accept (the same MSG_SHORT/MON/FENCE packets the driver's
 *        own gaudi2_test_queues path runs). So the QMAN's CP cannot fault on
 *        unexpected program content.
 *     2. The WAIT CS *is* the on-device cross-engine dependency we are trying to
 *        prove: its in-kernel CB arms a monitor on the producer's SOB and ends in
 *        a FENCE packet that stalls the consumer queue's CP ON-DEVICE until the
 *        SOB reaches target — zero host involvement. That is precisely the
 *        mechanism that would delete the per-op host fence in the real engine.
 *
 *   WHAT WE DELIBERATELY DO **NOT** SUBMIT BY DEFAULT (root-cause, see below):
 *   a USER-allocated CB containing a raw LIN_DMA/NOP on an EDMA HW queue. On
 *   Gaudi-2 EVERY compute/DMA queue is QUEUE_TYPE_HW (gaudi2.c:2779) and the
 *   driver appends NOTHING to a user CB on such a queue: hw_queue_schedule_job()
 *   pushes the user CB verbatim and .add_end_of_cb_packets == NULL (gaudi2.c:
 *   12680) — completion is out-of-band via a CQ monitor armed by
 *   gaudi2_pre_schedule_cs (gaudi2.c:11935), not via any packet in the CB. The
 *   driver itself NEVER runs a user-style LIN_DMA on an EDMA queue through the
 *   normal CP: its internal memset path (gaudi2.c:11264) first puts the QMAN in
 *   TEST MODE (gaudi2_qman_set_test_mode -> QMAN_MAKE_TRUSTED_TEST_MODE, MMU
 *   bypass, CB at a PHYSICAL HBM addr), and gaudi2_test_queues() (gaudi2.c:7872)
 *   explicitly SKIPS EDMA queue ids. A bare user CB submitted to an EDMA QMAN's
 *   CP in normal (non-test, MMU-on) mode faults the CP -> err-interrupt -> the
 *   driver resets the device (~43s self-heal). The CS *ioctl* is accepted
 *   (errno=0) because validation passes; the QMAN faults later EXECUTING it.
 *   This probe is kept behind --probe-usercb so the DECISIVE run never touches
 *   the CP and never trips a reset.
 *
 * It is byte-honest: a global ioctl() counter (LD-independent, we call our own
 * wrapper drm_ioctl()) tallies every CS / WAIT_CS / CB / MEM ioctl. The
 * pass/fail gate and the count contrast are printed at the end.
 *
 * SAFETY / SCOPE:
 *  - Off-card design artifact; RUN by the main loop on a card with NO active
 *    compute ctx (driver enforces single compute ctx => EBUSY, we detect+skip).
 *  - Sync-manager (SIGNAL/WAIT CS) ops only by default. No user CB hits the CP
 *    unless --probe-usercb is passed.
 *  - LOUD failure: every ioctl checked; any unexpected errno => print + abort.
 *  - The SIGNAL/WAIT CS path uses the kernel's OWN SOB/monitor allocator
 *    (init_signal_cs/init_wait_cs, hw_queue.c:415/474), so there is no collision
 *    with anything. We never touch another ctx's state (there is no other ctx —
 *    we hold the single compute ctx for our run).
 *
 * BUILD:  ./build.sh         (plain gcc, no Habana libs)
 * RUN:    see RUN_ONCARD.md  (the main loop runs this; needs a free card)
 * =============================================================================
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "hl_uapi.h"

/* Gaudi-2 queue IDs (UAPI enum, habanalabs_accel.h:316).
 *
 * STRIKE-3 ROOT CAUSE (the EIO): the previous default used EDMA queues 8/12.
 * On Gaudi-2 EVERY EDMA queue is QUEUE_TYPE_HW *and* dram_bd=1 (its buffer-
 * descriptor ring lives in HBM, gaudi2.c:2791-2793,2747-2756). A bare user
 * SIGNAL CS submitted to an EDMA QMAN through the accel ioctl retires through
 * the HW-queue auto-completion path (gaudi2_pre_schedule_cs arms a CQ monitor
 * that issues an HBW write to GAUDI2_RESERVED_CQ_CS_COMPLETION; hw_queue.c:
 * 380-412 also encodes a QMAN HBW completion message). On the EDMA dram_bd
 * queue that completion write lands at a poisoned/unmapped translation:
 *   dmesg: "EDMA0_QM: stream0. err cause: CQ AXI HBW error, qid = 8"
 *          "EDMA_0-RAZWI SHARED RR HBW RD error, address 0xfff00001..."
 *          "Going to reset device" -> "Killing CS 1.1"
 * The driver then resets the device and force-aborts the in-flight CS with
 * fence->error = -EIO (command_submission.c:1087 cs->aborted=true -> :789 ->
 * HL_WAIT_CS returns -EIO, :3226 "CS has been aborted while user process is
 * waiting for it"). So the EIO is hypothesis (b) (device errored EXECUTING the
 * signal), NOT a submit-encoding bug — and it is 100% reproducible (same
 * signature twice in dmesg, always qid=8, always CS 1.1 = the FIRST signal).
 *
 * THE TELL: the driver's own queue self-test gaudi2_test_queues()
 * (gaudi2.c:7872) EXPLICITLY SKIPS EDMA queues (7881/7898 gaudi2_is_edma_queue_id)
 * and only exercises PDMA + compute (TPC/MME) QMANs. EDMA is never validated
 * for a user msg-short/sync submit. So we move OFF EDMA onto the families the
 * driver itself proves work, none of which are dram_bd:
 *   PDMA_0_0=0 (CP0 user-enabled), PDMA_1_0=4 (CP0 user-enabled),
 *   MME_0_0=16, TPC_0_0=20 (all 4 CPs user-enabled, QMAN_ENABLE=0xF).
 * (PDMA0 enables only PQ0/PQ1, PDMA1 only PQ0 — CP2/3 are ARC-reserved,
 *  gaudi2.c:5871-5879 / gaudi2_masks.h:54-64 — so use the *_0 queue id.)
 * Override with --qa/--qb. All non-NIC Gaudi2 queues advertise sync-stream
 * (gaudi2.c:2782-2787); the EIO was the dram_bd completion, not sync-stream. */
#define Q_PDMA_A   0   /* GAUDI2_QUEUE_ID_PDMA_0_0 (CP0, user, NOT dram_bd) */
#define Q_PDMA_B   4   /* GAUDI2_QUEUE_ID_PDMA_1_0 (CP0, user, NOT dram_bd) */
/* legacy names kept for any external caller; now point at PDMA, not EDMA. */
#define Q_EDMA_A   Q_PDMA_A
#define Q_EDMA_B   Q_PDMA_B

/* Sync-manager register bases (gaudi2 dcore0_sync_mngr_objs). We DO NOT hardcode
 * absolute CFG addresses for SOB programming from a host-mapped CB in this PoC;
 * instead we use the kernel's OWN signal/wait CS path (HL_CS_FLAGS_SIGNAL/WAIT),
 * which builds the exact gen_signal_cb/gen_wait_cb monitor packets in-kernel
 * (hw_queue.c:415,474). That is the supported, collision-free way to get
 * on-device sync, and it is precisely what proves the fence is deletable. */

/* ----------------------------- ioctl tally ------------------------------ */
static long g_n_cs = 0, g_n_wait = 0, g_n_cb = 0, g_n_mem = 0, g_n_info = 0, g_n_other = 0;

static const char *ioc_name(unsigned long req) {
	switch (req) {
	case DRM_IOCTL_HL_CS:      return "CS";
	case DRM_IOCTL_HL_WAIT_CS: return "WAIT_CS";
	case DRM_IOCTL_HL_CB:      return "CB";
	case DRM_IOCTL_HL_MEMORY:  return "MEM";
	case DRM_IOCTL_HL_INFO:    return "INFO";
	default:                   return "OTHER";
	}
}
static int g_trace = -1;
static int drm_ioctl(int fd, unsigned long req, void *arg) {
	if      (req == DRM_IOCTL_HL_CS)      g_n_cs++;
	else if (req == DRM_IOCTL_HL_WAIT_CS) g_n_wait++;
	else if (req == DRM_IOCTL_HL_CB)      g_n_cb++;
	else if (req == DRM_IOCTL_HL_MEMORY)  g_n_mem++;
	else if (req == DRM_IOCTL_HL_INFO)    g_n_info++;
	else                                  g_n_other++;
	if (g_trace < 0) g_trace = getenv("CSPOC_TRACE") ? 1 : 0;
	int r = ioctl(fd, req, arg);
	int saved_errno = errno;   /* preserve across the trace fprintf below (it can clobber errno) */
	if (g_trace) fprintf(stderr, "  ioctl %-8s -> %d%s\n", ioc_name(req), r, r? " (err)":"");
	errno = saved_errno;       /* so the immediate caller sees the real ioctl errno */
	return r;
}
static void tally_reset(void){ g_n_cs=g_n_wait=g_n_cb=g_n_mem=g_n_info=g_n_other=0; }
static void tally_print(const char *tag){
	printf("  [tally %-8s] CS=%ld WAIT_CS=%ld CB=%ld MEM=%ld INFO=%ld OTHER=%ld  (host round-trips total=%ld)\n",
	       tag, g_n_cs, g_n_wait, g_n_cb, g_n_mem, g_n_info, g_n_other,
	       g_n_cs+g_n_wait+g_n_cb+g_n_mem+g_n_info+g_n_other);
}

#define DIE(...) do { fprintf(stderr, "FATAL: " __VA_ARGS__); fprintf(stderr, " (errno=%d %s)\n", errno, strerror(errno)); exit(2);} while(0)
#define CHECK(cond, ...) do { if(!(cond)) DIE(__VA_ARGS__); } while(0)

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }

/* ------------------------- device open (skip busy) ---------------------- */
static int open_free_card(int want_idx) {
	char path[64];
	int lo = (want_idx >= 0) ? want_idx : 0;
	int hi = (want_idx >= 0) ? want_idx : 7;
	for (int i = lo; i <= hi; i++) {
		snprintf(path, sizeof path, "/dev/accel/accel%d", i);
		int fd = open(path, O_RDWR | O_CLOEXEC);
		if (fd < 0) {
			if (errno == EBUSY) { printf("  accel%d: EBUSY (compute ctx active) - skip\n", i); continue; }
			printf("  accel%d: open failed errno=%d %s - skip\n", i, errno, strerror(errno));
			continue;
		}
		/* Confirm we got a usable compute ctx: a trivial INFO ioctl. */
		struct hl_info_args ia; struct hl_info_hw_ip_info_min { uint64_t sram_base, dram_base, dram_size; uint32_t sram_size, num_ev, dev_id; } ip;
		memset(&ia,0,sizeof ia); memset(&ip,0,sizeof ip);
		ia.op = HL_INFO_HW_IP_INFO; ia.return_pointer = (uint64_t)(uintptr_t)&ip; ia.return_size = sizeof ip;
		if (drm_ioctl(fd, DRM_IOCTL_HL_INFO, &ia) != 0) { printf("  accel%d: INFO failed - skip\n", i); close(fd); continue; }
		printf("  accel%d: OPEN OK (device_id=0x%x, dram_base=0x%llx, dram_size=%llu MB)\n",
		       i, ip.dev_id, (unsigned long long)ip.dram_base, (unsigned long long)ip.dram_size>>20);
		return fd;
	}
	return -1;
}

/* ------------------------- CB + memory helpers -------------------------- */
/* For Gaudi-2 HW queues the supported CB convention is a USER-ALLOC CB:
 * allocate a host buffer, map it into the device MMU, write packets into the
 * host buffer (host-visible), and pass the returned DEVICE VA as the chunk's
 * cb_handle with HL_CS_CHUNK_FLAGS_USER_ALLOC_CB. Driver-allocated CBs need
 * CB_ALLOC_KERNEL which Gaudi-2 HW queues do NOT have (gaudi2.c:2789 sets
 * CB_ALLOC_USER), so a driver CB on a HW queue is rejected with -EINVAL
 * ("doesn't support kernel CB", command_submission.c:1271). The kernel casts
 * chunk->cb_handle directly to the address for user-alloc CBs
 * (command_submission.c:1589) and DMAs from it (hw_queue.c:410). */
typedef struct { void *host; uint64_t dev_va; uint32_t size; } usercb_t;

static usercb_t usercb_alloc(int fd, uint32_t size) {
	usercb_t cb; cb.size = size;
	/* page-aligned host buffer */
	cb.host = NULL;
	CHECK(posix_memalign(&cb.host, 4096, size)==0, "posix_memalign %u", size);
	memset(cb.host, 0, size);
	union hl_mem_args m; memset(&m,0,sizeof m);
	m.in.op = HL_MEM_OP_MAP;
	m.in.map_host.host_virt_addr = (uint64_t)(uintptr_t)cb.host;
	m.in.map_host.hint_addr = 0;
	m.in.map_host.mem_size = size;
	m.in.flags = HL_MEM_USERPTR; /* REQUIRED: driver branches host-vs-device on this (memory.c:1566) */
	CHECK(drm_ioctl(fd, DRM_IOCTL_HL_MEMORY, &m)==0, "host_memory_map size=%u", size);
	cb.dev_va = m.out.device_virt_addr;
	CHECK(cb.dev_va != 0, "host_memory_map returned dev_va=0");
	return cb;
}
static void usercb_free(int fd, usercb_t *cb){
	union hl_mem_args m; memset(&m,0,sizeof m);
	m.in.op = HL_MEM_OP_UNMAP; m.in.unmap.device_virt_addr = cb->dev_va;
	drm_ioctl(fd, DRM_IOCTL_HL_MEMORY, &m);
	free(cb->host);
}

static uint64_t dram_alloc(int fd, uint64_t size){
	union hl_mem_args a; memset(&a,0,sizeof a);
	a.in.op=HL_MEM_OP_ALLOC; a.in.alloc.mem_size=size; a.in.flags=HL_MEM_CONTIGUOUS;
	CHECK(drm_ioctl(fd,DRM_IOCTL_HL_MEMORY,&a)==0, "DRAM alloc %llu", (unsigned long long)size);
	uint64_t handle=a.out.handle;
	union hl_mem_args m; memset(&m,0,sizeof m);
	m.in.op=HL_MEM_OP_MAP; m.in.map_device.handle=handle; m.in.map_device.hint_addr=0;
	CHECK(drm_ioctl(fd,DRM_IOCTL_HL_MEMORY,&m)==0, "DRAM map");
	return m.out.device_virt_addr;
}

/* A single NOP packet (always-valid work item; the SIMPLEST possible work, so it
 * isolates device-state failures from DMA-packet-content failures). packet_nop is
 * {reserved, ctl} with only the opcode field meaningful (gaudi2_packets.h). The
 * kernel sets MB on its own NOP-style packets, so we match that. */
static uint32_t emit_nop(void *buf) {
	struct g2_packet_msg_short *unused = (void*)buf; (void)unused;
	uint32_t *p = (uint32_t*)buf;          /* packet_nop: {reserved, ctl} */
	p[0] = 0;
	p[1] = (PACKET_NOP << G2_PKT_CTL_OPCODE_SHIFT) | (1u << G2_PKT_CTL_MB_SHIFT);
	return 8;
}

/* ----------------------- Gaudi-2 DMA packet builder --------------------- */
/* Build a single LIN_DMA memset packet into buf; returns bytes written.
 * memset writes 'tsize' bytes of value (low 32b of src_addr) to dst_addr.
 *
 * CTL BITS MATCH THE KERNEL'S OWN EDMA memset builder EXACTLY
 * (gaudi2.c:11271-11274 gaudi2_memset_memory_chunk_using_edma_qm):
 *     OPCODE=LIN_DMA | MEMSET=1 | WRCOMP=1 | EB=1     (NO MB bit)
 * The previous version set MB and omitted WRCOMP+EB; an EDMA QMAN can hang on a
 * LIN_DMA with no write-completion / no engine-barrier, which would fire the TDR
 * and reset the device (-> EBUSY on the next CS). We now mirror the driver byte
 * layout so the packet is execute-valid on an EDMA QMAN. */
static uint32_t emit_dma_memset(void *buf, uint64_t dst, uint32_t tsize, uint32_t val) {
	struct g2_packet_lin_dma *p = (struct g2_packet_lin_dma*)buf;
	memset(p,0,sizeof *p);
	uint32_t ctl = (PACKET_LIN_DMA << G2_PKT_CTL_OPCODE_SHIFT)
	             | (1u << G2_LIN_DMA_CTL_MEMSET_SHIFT)
	             | (1u << G2_LIN_DMA_CTL_WRCOMP_SHIFT)
	             | (1u << G2_PKT_CTL_EB_SHIFT);
	p->tsize = tsize;
	p->ctl = ctl;
	p->src_addr = val;     /* memset pattern in low 32 bits */
	p->dst_addr = dst;
	return sizeof *p;
}

/* ------------------------------- modes ---------------------------------- */
/* Submit a single work CB to a queue as a DEFAULT CS; optionally wait.
 * abort_on_err=false lets the sanity probe detect DMA rejection and fall back. */
static int submit_work(int fd, int qidx, uint64_t cb_dev_va, uint32_t cb_used, bool do_wait,
                       const char *why, bool abort_on_err, uint64_t *out_seq, int *out_errno) {
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.cb_handle = cb_dev_va;          /* device VA of user-alloc, MMU-mapped CB */
	chunk.queue_index = qidx;
	chunk.cb_size = cb_used;
	chunk.cs_chunk_flags = HL_CS_CHUNK_FLAGS_USER_ALLOC_CB;  /* required for Gaudi2 HW queues */
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	cs.in.cs_flags = 0;
	int r = drm_ioctl(fd, DRM_IOCTL_HL_CS, &cs);
	if (r != 0) {
		if (out_errno) *out_errno = errno;   /* CAPTURE the CS errno (decisive a-vs-b signal) */
		if (abort_on_err) DIE("CS default (%s) q=%d", why, qidx);
		return r;
	}
	if (out_errno) *out_errno = 0;
	uint64_t seq = cs.out.seq;
	if (do_wait) {
		union hl_wait_cs_args w; memset(&w,0,sizeof w);
		w.in.seq = seq; w.in.timeout_us = 1000000; /* 1s */
		int rw = drm_ioctl(fd, DRM_IOCTL_HL_WAIT_CS, &w);
		if (rw != 0) { if (abort_on_err) DIE("WAIT_CS (%s)", why); return rw; }
		if (w.out.status != HL_WAIT_CS_STATUS_COMPLETED) {
			if (abort_on_err) DIE("CS (%s) status=%u (not COMPLETED)", why, w.out.status);
			return -1;
		}
	}
	if (out_seq) *out_seq = seq;
	return 0;
}
/* SIGNAL CS: increments a SOB on the device when the queue's prior work
 * completes. The kernel builds the CB in-kernel via gaudi2_gen_signal_cb
 * (a MSG_SHORT that adds 1 to the queue's reserved SOB), so NO user CB is
 * involved and the QMAN's CP cannot fault on program content. Returns the CS
 * sequence (host-waitable). */
static uint64_t submit_signal(int fd, int qidx, uint32_t *out_sob_off) {
	/* SIGNAL CS takes NO user CB; the kernel emits gen_signal_cb. One chunk w/ queue. */
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.queue_index = qidx;
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	cs.in.cs_flags = HL_CS_FLAGS_SIGNAL;
	CHECK(drm_ioctl(fd, DRM_IOCTL_HL_CS, &cs)==0, "CS SIGNAL q=%d", qidx);
	if (out_sob_off) *out_sob_off = cs.out.sob_base_addr_offset;
	return cs.out.seq;
}
/* count of WAIT CSs that genuinely ARMED an on-device monitor (vs the signal
 * having already completed -> dependency met, nothing armed). A decisive
 * csengine run must arm > 0 of these, proving the device carried the dep. */
static long g_waits_armed = 0, g_waits_already_done = 0;

/* WAIT CS: arms a monitor on the SOB of signal_seq; the engine on qidx blocks
 * ON-DEVICE until the SOB reaches target. No host involvement until we choose.
 * Returns: 1 = wait armed (seq in *out_seq); 0 = signal ALREADY completed.
 *
 * DRIVER-ACCURATE "already completed" handling: cs_ioctl_signal_wait returns
 * rc=0 (NOT -EINVAL) with *cs_seq left at ULLONG_MAX when the signal CS has
 * already finished — command_submission.c:2367-2371 (sig_fence==NULL) and
 * :2392-2397 (completion_done). So success + seq==ULLONG_MAX means "dependency
 * already met, nothing armed". A genuine bad seq returns -EINVAL (:2363). */
static int submit_wait_on(int fd, int qidx, uint64_t signal_seq, uint64_t *out_seq) {
	uint64_t seqarr[1] = { signal_seq };
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.queue_index = qidx;
	chunk.signal_seq_arr = (uint64_t)(uintptr_t)seqarr;
	chunk.num_signal_seq_arr = 1;
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	cs.in.cs_flags = HL_CS_FLAGS_WAIT;
	cs.out.seq = ~0ULL;   /* sentinel: driver leaves it ULLONG_MAX if already done */
	int r = drm_ioctl(fd, DRM_IOCTL_HL_CS, &cs);
	if (r != 0)
		DIE("CS WAIT q=%d on seq=%llu", qidx, (unsigned long long)signal_seq);
	if (cs.out.seq == ~0ULL) {
		g_waits_already_done++;
		return 0;         /* signal already fired: dependency met, nothing to wait on */
	}
	g_waits_armed++;
	if (out_seq) *out_seq = cs.out.seq;
	return 1;
}

static void host_wait(int fd, uint64_t seq, const char *why) {
	union hl_wait_cs_args w; memset(&w,0,sizeof w);
	w.in.seq = seq; w.in.timeout_us = 2000000;
	CHECK(drm_ioctl(fd, DRM_IOCTL_HL_WAIT_CS, &w)==0, "host_wait (%s)", why);
	CHECK(w.out.status==HL_WAIT_CS_STATUS_COMPLETED, "host_wait (%s) status=%u", why, w.out.status);
}

/* ----------------------- device-state diagnostics ----------------------- */
/* These let us DISTINGUISH the two failure hypotheses in ONE run:
 *   (a) submission bug  -> device stays OPERATIONAL across the failing CS,
 *                          and the CS errno is EFAULT/EINVAL (validation drop).
 *   (b) device dirty    -> device flips to IN_RESET/NEEDS_RESET/MALFUNCTION and
 *       /async reset        a soft/hard reset counter ticks; CS errno is EBUSY/EPERM.
 * HL_INFO_DEVICE_STATUS needs no open ctx and reports the live device status enum
 * (driver: hl_device_status, device.c:521). HL_INFO_RESET_COUNT reports the
 * soft/hard reset tallies since driver load (device.c reset path). */
static const char *dev_status_str(uint32_t s) {
	switch (s) {
	case HL_DEVICE_STATUS_OPERATIONAL:                   return "OPERATIONAL";
	case HL_DEVICE_STATUS_IN_RESET:                      return "IN_RESET";
	case HL_DEVICE_STATUS_MALFUNCTION:                   return "MALFUNCTION";
	case HL_DEVICE_STATUS_NEEDS_RESET:                   return "NEEDS_RESET";
	case HL_DEVICE_STATUS_IN_DEVICE_CREATION:            return "IN_DEVICE_CREATION";
	case HL_DEVICE_STATUS_IN_RESET_AFTER_DEVICE_RELEASE: return "IN_RESET_AFTER_DEVICE_RELEASE";
	default:                                             return "UNKNOWN";
	}
}
/* returns status enum, or -1 on ioctl failure */
static int dev_status(int fd) {
	struct hl_info_args ia; struct hl_info_device_status ds;
	memset(&ia,0,sizeof ia); memset(&ds,0,sizeof ds);
	ia.op=HL_INFO_DEVICE_STATUS; ia.return_pointer=(uint64_t)(uintptr_t)&ds; ia.return_size=sizeof ds;
	if (drm_ioctl(fd, DRM_IOCTL_HL_INFO, &ia)!=0) return -1;
	return (int)ds.status;
}
/* fills *hard,*soft; returns 0 ok / -1 fail */
static int reset_counts(int fd, uint32_t *hard, uint32_t *soft) {
	struct hl_info_args ia; struct hl_info_reset_count rc;
	memset(&ia,0,sizeof ia); memset(&rc,0,sizeof rc);
	ia.op=HL_INFO_RESET_COUNT; ia.return_pointer=(uint64_t)(uintptr_t)&rc; ia.return_size=sizeof rc;
	if (drm_ioctl(fd, DRM_IOCTL_HL_INFO, &ia)!=0) return -1;
	*hard=rc.hard_reset_cnt; *soft=rc.soft_reset_cnt; return 0;
}
/* Print the decisive (a)-vs-(b) classification for a failed CS submit.
 * base_hard/base_soft = reset counters captured BEFORE the failing submit. */
static void classify_cs_failure(int fd, const char *what, int cs_errno,
                                uint32_t base_hard, uint32_t base_soft) {
	int st = dev_status(fd);
	uint32_t hard=0, soft=0; int rcok = reset_counts(fd,&hard,&soft);
	fprintf(stderr,
		"\n================ CS-FAILURE DIAGNOSIS (%s) ================\n", what);
	fprintf(stderr, "  CS submit errno = %d (%s)\n", cs_errno, strerror(cs_errno));
	fprintf(stderr, "  device status now = %s\n", st<0? "INFO failed (device gone?)" : dev_status_str((uint32_t)st));
	if (rcok==0)
		fprintf(stderr, "  reset counters: hard %u->%u  soft %u->%u  (delta hard=%d soft=%d)\n",
			base_hard, hard, base_soft, soft, (int)hard-(int)base_hard, (int)soft-(int)base_soft);
	bool reset_fired = (rcok==0) && ((hard!=base_hard) || (soft!=base_soft));
	bool nonop = (st==HL_DEVICE_STATUS_IN_RESET || st==HL_DEVICE_STATUS_NEEDS_RESET ||
	              st==HL_DEVICE_STATUS_MALFUNCTION || st==HL_DEVICE_STATUS_IN_RESET_AFTER_DEVICE_RELEASE ||
	              st<0);
	if (cs_errno==EBUSY || cs_errno==EPERM || reset_fired || nonop) {
		fprintf(stderr,
		  "\n  => HYPOTHESIS (b): DEVICE NON-OPERATIONAL / RESET.\n"
		  "     EBUSY(16) comes ONLY from hl_cs_sanity_checks (command_submission.c:1382,\n"
		  "     !hl_device_operational); EPERM(1) from hl_hw_queue_schedule_cs:657. Both mean\n"
		  "     the device left OPERATIONAL state (IN_RESET/NEEDS_RESET/MALFUNCTION).\n"
		  "     Open+INFO+MEM all succeeded, so the device was operational at open and went\n"
		  "     bad during the run (consistent with the 2 device-critical-errors today).\n"
		  "     ACTION: this card needs a RESET. Reboot, then re-run (see operator notes).\n");
	} else if (cs_errno==EFAULT) {
		fprintf(stderr,
		  "\n  => HYPOTHESIS (a): SUBMISSION BUG (CB address). EFAULT(14) is returned by\n"
		  "     gaudi2_validate_cb_address (gaudi2.c:7619) when the user-CB device VA is not\n"
		  "     inside SRAM/DRAM/PMMU ranges. The host-mapped CB VA is not landing in the\n"
		  "     PMMU range the validator checks. Device stayed OPERATIONAL (clean reject).\n");
	} else if (cs_errno==EINVAL) {
		fprintf(stderr,
		  "\n  => HYPOTHESIS (a): SUBMISSION BUG (queue/flags). EINVAL(22) is from\n"
		  "     validate_queue_index / hl_cs_sanity_checks flag checks. Device stayed\n"
		  "     OPERATIONAL (clean reject). Check queue_index / cs_chunk_flags.\n");
	} else {
		fprintf(stderr,
		  "\n  => UNEXPECTED errno; device status above is the tie-breaker:\n"
		  "     OPERATIONAL -> submission bug (a); non-operational -> dirty device (b).\n");
	}
	fprintf(stderr, "===========================================================\n");
}

int main(int argc, char **argv) {
	int want_card = -1, qa = Q_PDMA_A, qb = Q_PDMA_B, nchain = 8, reps = 50;
	bool probe_usercb = false;   /* OFF by default: a user CB on a HW queue faults
	                              * the QMAN's CP and trips a device reset
	                              * (see file header root-cause). Opt-in only. */
	for (int i=1;i<argc;i++){
		if(!strcmp(argv[i],"--card")&&i+1<argc) want_card=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--qa")&&i+1<argc) qa=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--qb")&&i+1<argc) qb=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--nchain")&&i+1<argc) nchain=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--reps")&&i+1<argc) reps=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--probe-usercb")) probe_usercb=true;
		else if(!strcmp(argv[i],"--help")){
			printf("usage: %s [--card N] [--qa Q] [--qb Q] [--nchain N] [--reps N] [--probe-usercb]\n",argv[0]);
			printf("  default qa=%d (PDMA_0_0) qb=%d (PDMA_1_0).  AVOID EDMA (8..15): on Gaudi2 EDMA\n", Q_PDMA_A, Q_PDMA_B);
			printf("  queues are dram_bd and their CS-completion HBW write RAZWIs -> device reset ->\n");
			printf("  WAIT_CS returns EIO. gaudi2_test_queues() itself skips EDMA. Compute queues\n");
			printf("  also work: MME_0_0=16, TPC_0_0=20.\n");
			return 0;}
	}
	if (nchain < 2) { fprintf(stderr,"--nchain must be >= 2 (need a cross-engine hop)\n"); return 2; }

	/* Compile-time-ish ABI self-check: sizes must match the real UAPI. */
	printf("== ABI self-check (struct sizes) ==\n");
	printf("  hl_cs_chunk=%zu (UAPI=64)  hl_cs_in=%zu  hl_wait_cs_in=%zu (UAPI=56)\n",
	       sizeof(struct hl_cs_chunk), sizeof(struct hl_cs_in), sizeof(struct hl_wait_cs_in));
	CHECK(sizeof(struct hl_cs_chunk)==64, "hl_cs_chunk size %zu != 64 (ABI MISMATCH)", sizeof(struct hl_cs_chunk));
	CHECK(sizeof(struct hl_wait_cs_in)==56, "hl_wait_cs_in size %zu != 56 (ABI MISMATCH)", sizeof(struct hl_wait_cs_in));
	printf("  DRM_IOCTL_HL_CS=0x%lx  DRM_IOCTL_HL_WAIT_CS=0x%lx (expect 0xc0386443)\n",
	       (unsigned long)DRM_IOCTL_HL_CS, (unsigned long)DRM_IOCTL_HL_WAIT_CS);

	printf("\n== open a free card ==\n");
	int fd = open_free_card(want_card);
	if (fd < 0) { fprintf(stderr, "\nNO FREE CARD (all EBUSY or unopenable). The driver allows ONE compute ctx per device;\n"
	                              "run this when a card has no active vLLM serve. Aborting cleanly.\n"); return 3; }

	/* sync-manager: confirm we have user SOBs/monitors available (driver-safe range).
	 * NOTE: first_sob/first_mon being non-zero is NORMAL, NOT a leak: they are static
	 * driver-reserved bases computed at device init from the ASIC topology
	 * (gaudi2.c:2933 first_available_user_sob = GAUDI2_RESERVED_SOB_NUMBER +
	 * num_sync_stream_queues*HL_RSVD_SOBS). Identical on every clean boot. */
	struct hl_info_args ia; struct hl_info_sync_manager sm; memset(&ia,0,sizeof ia); memset(&sm,0,sizeof sm);
	ia.op=HL_INFO_SYNC_MANAGER; ia.return_pointer=(uint64_t)(uintptr_t)&sm; ia.return_size=sizeof sm;
	if (drm_ioctl(fd, DRM_IOCTL_HL_INFO, &ia)==0)
		printf("  sync_mgr: first_sob=%u first_mon=%u first_cq=%u (reserved-base, not a leak)\n",
		       sm.first_available_sync_object, sm.first_available_monitor, sm.first_available_cq);

	/* BASELINE device state: the device MUST be OPERATIONAL here (open + INFO + MEM
	 * already succeeded, all of which gate on hl_device_operational). Capture reset
	 * counters so we can later witness whether a reset fires DURING the run. */
	int st0 = dev_status(fd);
	uint32_t base_hard=0, base_soft=0; int rc_have = (reset_counts(fd,&base_hard,&base_soft)==0);
	printf("  device status (baseline) = %s\n", st0<0? "INFO failed" : dev_status_str((uint32_t)st0));
	if (rc_have) printf("  reset counters (baseline): hard=%u soft=%u\n", base_hard, base_soft);
	if (st0 != HL_DEVICE_STATUS_OPERATIONAL && st0 != HL_DEVICE_STATUS_IN_DEVICE_CREATION) {
		fprintf(stderr,
		  "\n  !! Device is %s at baseline (NOT operational) yet open succeeded. This is a\n"
		  "     dirty/unstable device (HYPOTHESIS b). It needs a RESET before any CS will run.\n"
		  "     ACTION: reboot, then re-run. Aborting.\n", st0<0?"unreadable":dev_status_str((uint32_t)st0));
		close(fd); return 4;
	}

	/* ---------------- OPTIONAL: user-CB-on-EDMA fault probe (opt-in) ----------
	 * This is the path that ACCEPTS the CS (errno=0) but then faults the QMAN's
	 * CP executing the user CB, tripping a ~43s device reset. It is OFF by default
	 * so the decisive run below never disturbs the device. We keep it because it
	 * empirically demonstrates the root cause documented in the file header:
	 * a user CB (even a NOP) on a Gaudi-2 EDMA HW queue is not something the CP
	 * runs cleanly in normal (non-test, MMU-on) mode. classify_cs_failure() will
	 * print OPERATIONAL-at-submit (ioctl accepted) then, after the wait times out
	 * and the reset fires, the reset-counter delta / IN_RESET status. */
	if (probe_usercb) {
		printf("\n== OPTIONAL PROBE (--probe-usercb): user CB (NOP) on EDMA HW queue %d ==\n", qa);
		printf("  NOTE: per the driver analysis this is EXPECTED to fault the QMAN CP and\n"
		       "  trigger a device reset (the CS ioctl is accepted; the HW faults later).\n");
		const uint32_t DMA_BYTES = 4096;
		uint64_t dram = dram_alloc(fd, DMA_BYTES);
		usercb_t cb_a = usercb_alloc(fd, 256);
		uint32_t una = emit_nop(cb_a.host);
		uint64_t s=0; int ce=0;
		int nr = submit_work(fd, qa, cb_a.dev_va, una, true, "probe-usercb-nop", false, &s, &ce);
		if (nr != 0) {
			classify_cs_failure(fd, "probe-usercb-NOP", ce, base_hard, base_soft);
			printf("  (probe) user-CB NOP did NOT complete cleanly, as predicted.\n");
		} else {
			/* If this ever completes, also exercise a real LIN_DMA memset. */
			uint32_t da = emit_dma_memset(cb_a.host, dram, DMA_BYTES, 0xAA);
			uint64_t sd=0; int de=0;
			int dr = submit_work(fd, qa, cb_a.dev_va, da, true, "probe-usercb-dma", false, &sd, &de);
			if (dr != 0) classify_cs_failure(fd, "probe-usercb-DMA", de, base_hard, base_soft);
			else printf("  (probe) user-CB NOP + DMA BOTH completed (unexpected — CP ran a user CB).\n");
			(void)sd;
		}
		(void)s;
		usercb_free(fd, &cb_a);
		/* leak the DRAM handle deliberately if the device is mid-reset; close() frees ctx */
		/* Re-baseline reset counters: a reset here would otherwise make the decisive
		 * run below mis-read the device as dirty. If a reset DID fire, the device will
		 * be IN_RESET; the decisive run can't proceed, so report and stop. */
		int stp = dev_status(fd);
		if (stp != HL_DEVICE_STATUS_OPERATIONAL) {
			fprintf(stderr,
			  "\n  Device is %s after the probe (a reset fired). The decisive SIGNAL/WAIT run\n"
			  "  needs an OPERATIONAL device; re-run WITHOUT --probe-usercb once it self-heals\n"
			  "  (~43s). Stopping here so we don't measure a resetting device.\n",
			  stp<0?"unreadable":dev_status_str((uint32_t)stp));
			close(fd); return 6;
		}
	}

	/* ---------------- SINGLE-SIGNAL SELF-TEST (strike-3 guard) -------------
	 * Submit exactly ONE SIGNAL CS on qa and host-wait it, with NO abort, BEFORE
	 * the 50-rep sweep. This is the minimal form of the failing op and isolates
	 * the EIO at its source (CS sequence 1 — exactly what dmesg's "Killing CS 1.1"
	 * showed for the EDMA default). If this single signal cannot retire cleanly we
	 * STOP here: running the full sweep would just trip a second device reset.
	 *   EIO here  => the queue family's CS-completion path faults the device
	 *                (EDMA dram_bd RAZWI, or this card is already dirty). The
	 *                classify dump + dmesg ("CQ AXI HBW error, qid=N" / RAZWI /
	 *                "Going to reset device") give the precise cause.
	 *   clean     => proceed to the decisive sweep. */
	printf("\n== SINGLE-SIGNAL SELF-TEST : 1 SIGNAL CS on q=%d + host WAIT (strike-3 guard) ==\n", qa);
	{
		struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
		chunk.queue_index = qa;
		union hl_cs_args cs; memset(&cs,0,sizeof cs);
		cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
		cs.in.num_chunks_execute = 1;
		cs.in.cs_flags = HL_CS_FLAGS_SIGNAL;
		if (drm_ioctl(fd, DRM_IOCTL_HL_CS, &cs) != 0) {
			int e = errno;
			classify_cs_failure(fd, "self-test-SIGNAL-submit", e, base_hard, base_soft);
			fprintf(stderr, "  SIGNAL CS submit itself failed on q=%d; cannot proceed. Aborting.\n", qa);
			close(fd); return 8;
		}
		union hl_wait_cs_args w; memset(&w,0,sizeof w);
		w.in.seq = cs.out.seq; w.in.timeout_us = 2000000;
		int rw = drm_ioctl(fd, DRM_IOCTL_HL_WAIT_CS, &w);
		int e = errno;
		if (rw != 0) {
			classify_cs_failure(fd, "self-test-SIGNAL-wait", e, base_hard, base_soft);
			if (e == EIO) {
				fprintf(stderr,
				  "\n  => STRIKE-3 SIGNATURE: a single kernel-built SIGNAL CS aborted with EIO.\n"
				  "     The CS was accepted then the device errored EXECUTING its completion\n"
				  "     (HW-queue CQ-completion HBW write). On q=8..15 (EDMA) this is the\n"
				  "     dram_bd RAZWI; gaudi2_test_queues() skips EDMA for exactly this reason.\n"
				  "     If q=%d is NOT EDMA and this still EIOs, the userspace SIGNAL-CS path is\n"
				  "     not viable on this driver/queue (see PART B: pivot to HPU-graph capture).\n", qa);
			}
			fprintf(stderr, "  STOPPING before the 50-rep sweep so we do not trip a second reset.\n");
			close(fd); return 9;
		}
		if (w.out.status != HL_WAIT_CS_STATUS_COMPLETED) {
			fprintf(stderr, "  self-test wait returned status=%u (not COMPLETED); aborting.\n", w.out.status);
			close(fd); return 9;
		}
		printf("  self-test: single SIGNAL CS retired CLEANLY (status=COMPLETED). Proceeding.\n");
	}

	/* =====================================================================
	 * DECISIVE TEST — uses ONLY kernel-built SIGNAL/WAIT CSs (no user CB hits
	 * the CP). The SIGNAL CB (gaudi2_gen_signal_cb) increments a per-queue SOB;
	 * the WAIT CB (gaudi2_gen_wait_cb) arms a monitor on the producer's SOB and
	 * ends in a FENCE packet that stalls the consumer queue's CP ON-DEVICE.
	 * These are the exact packets the HW is built to run, so they retire cleanly
	 * AND they ARE the cross-engine on-device dependency we are proving.
	 * ===================================================================== */

	/* ---------------- MODE naive: host fence per op ----------------
	 * N independent SIGNAL CSs on alternating queues, each drained by its OWN
	 * host HL_WAIT_CS. This is the status-quo shape: one host fence per op. */
	printf("\n== MODE naive : %d SIGNAL CSs on alt queues, HOST WAIT per op (status-quo shape) x %d reps ==\n", nchain, reps);
	tally_reset();
	double t0 = now_ms();
	for (int r=0;r<reps;r++)
		for (int k=0;k<nchain;k++) {
			int q = (k&1)?qb:qa;
			uint64_t sig = submit_signal(fd, q, NULL);
			host_wait(fd, sig, "naive-sig");   /* one host fence PER op */
		}
	double t_naive = now_ms()-t0;
	tally_print("naive");
	printf("  per-step (=%d ops): WAIT_CS=%.1f   wall=%.3f ms/step (%d reps)\n",
	       nchain, (double)g_n_wait/reps, t_naive/reps, reps);
	long naive_waits_per_step = g_n_wait/reps;

	/* ------------- MODE csengine: on-device sync, 1 host wait -------------
	 * Chain: signalA(q=A); B waits-on-device for A's SOB; signalB(q=B); A waits
	 * for B's SOB; signalA; ... ; ONE host HL_WAIT_CS on the tail signal.
	 * Each cross-engine hop is carried by an on-device monitor+fence (the WAIT
	 * CS), so the host fences exactly ONCE per N-op step.
	 *
	 * Ordering note: we submit the producer's SIGNAL CS, then the consumer's
	 * WAIT CS against that signal's seq. cs_ioctl_signal_wait validates the seq
	 * is an as-yet-incomplete SIGNAL CS and, if so, arms a real monitor
	 * (init_wait_cs, hw_queue.c:474); if the signal already retired it returns
	 * seq==ULLONG_MAX (dependency already met). g_waits_armed counts the former:
	 * a decisive run MUST arm > 0, proving the device — not the host — carried
	 * the dependency. */
	printf("\n== MODE csengine : %d-op SIGNAL/WAIT chain, ON-DEVICE SOB/monitor sync, 1 HOST WAIT/step x %d reps ==\n", nchain, reps);
	tally_reset();
	g_waits_armed = g_waits_already_done = 0;
	t0 = now_ms();
	for (int r=0;r<reps;r++) {
		/* hop 0: producer signals on queue A (no consumer waits before it). */
		uint64_t prev_sig = submit_signal(fd, qa, NULL);
		for (int k=1;k<nchain;k++) {
			int q = (k&1)?qb:qa;   /* consumer engine for this hop */
			/* consumer engine q blocks ON-DEVICE until prev_sig's SOB hits target */
			uint64_t wseq=0; submit_wait_on(fd, q, prev_sig, &wseq);
			/* then this engine signals its own SOB when its (waited) work drains */
			prev_sig = submit_signal(fd, q, NULL);
		}
		/* ONE host wait for the tail signal: returns only when the whole
		 * device-side chain has drained. This is the single per-step fence.  */
		host_wait(fd, prev_sig, "cseng-tail");
	}
	double t_cseng = now_ms()-t0;
	tally_print("csengine");
	long cseng_waits_per_step = g_n_wait/reps;
	printf("  per-step (=%d ops): host WAIT_CS=%ld  on-device waits armed/step=%.1f  already-done/step=%.1f\n",
	       nchain, cseng_waits_per_step,
	       (double)g_waits_armed/reps, (double)g_waits_already_done/reps);
	printf("  wall=%.3f ms/step (%d reps)\n", t_cseng/reps, reps);

	/* ------------------------------ verdict ------------------------------ */
	printf("\n================= VERDICT =================\n");
	printf("  naive    : host WAIT_CS per %d-op step = %ld\n", nchain, naive_waits_per_step);
	printf("  csengine : host WAIT_CS per %d-op step = %ld\n", nchain, cseng_waits_per_step);
	if (cseng_waits_per_step > 0)
		printf("  => host fences reduced %.1fx for a %d-op step.\n",
		       (double)naive_waits_per_step/cseng_waits_per_step, nchain);
	printf("  wall/step: naive=%.3f ms  csengine=%.3f ms  (speedup %.2fx)\n",
	       t_naive/reps, t_cseng/reps, (t_naive/reps)/(t_cseng/reps>0?t_cseng/reps:1e-9));

	bool pass = (cseng_waits_per_step == 1) && (g_waits_armed > 0);
	printf("\n  DECISIVE GATE:\n");
	printf("   - csengine host WAIT_CS/step == 1 ?  %s (got %ld)\n",
	       cseng_waits_per_step==1?"YES":"NO", cseng_waits_per_step);
	printf("   - on-device WAITs actually armed ?   %s (armed %ld over %d reps)\n",
	       g_waits_armed>0?"YES":"NO", g_waits_armed, reps);
	printf("   - tail wait completed every rep ?    YES (host_wait aborts loudly on any miss)\n");
	if (pass) {
		printf("\n  ==> PASS: a multi-engine cross-synced chain drained with ONE host fence,\n");
		printf("      and the cross-engine dependency was carried by on-device monitors\n");
		printf("      (armed WAIT CSs). The per-op host fence IS DELETABLE at the CS layer.\n");
		printf("      Rung-2 is feasible; proceed to the staged build (ARCHITECTURE.md).\n");
	} else {
		printf("\n  ==> NOT PROVEN: csengine did not reduce to 1 host fence with armed\n");
		printf("      on-device waits. See the per-step counts above.\n");
	}

	close(fd);
	printf("\nDONE.\n");
	return pass ? 0 : 7;
}
