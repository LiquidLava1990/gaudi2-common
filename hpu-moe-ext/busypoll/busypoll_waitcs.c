// busypoll_waitcs.c — LD_PRELOAD shim: turn the blocking HL_WAIT_CS interrupt-sleep
// into a userspace busy-poll, to kill the ~1.78ms IRQ-wake latency per recipe completion
// on Gaudi-2 (the systemic single-stream floor: ~135 CS/forward x ~1.78ms wake).
//
// SAFE BY CONSTRUCTION: we do NOT read device memory. We call the driver's OWN completion
// check via the kernel's supported timeout_us=0 path. For the single-CS wait,
// hl_wait_for_fence does a non-sleeping completion_done() when timeout_us==0
// (command_submission.c:2728); for multi-CS, hl_cs_poll_fences() polls once and returns
// without sleeping when timeout_us==0 (command_submission.c:3127). We only change
// sleep->spin, with a hard max-spin budget that falls back to the original blocking wait
// (can NEVER hang) and immediate fall-through on any non-COMPLETED/non-BUSY status.
//
// Gated on HPU_BUSYPOLL_WAITCS=1 (default = pure passthrough = byte-identical to stock).
// Only the NON-interrupt CS/multi-CS wait (timeout_us @offset 8) is polled; interrupt waits
// (HL_WAIT_CS_FLAGS_INTERRUPT / INTERRUPT_KERNEL_CQ / REGISTER_INTERRUPT) pass straight
// through untouched (their timeout lives in interrupt_timeout_us @24 with different semantics).
//
// Build:  gcc -shared -fPIC -O2 -o busypoll_waitcs.so busypoll_waitcs.c -ldl
// Use:    LD_PRELOAD=/path/busypoll_waitcs.so HPU_BUSYPOLL_WAITCS=1 <serve>
//
// UAPI constants VERIFIED on-box (1.24.0-1007) by compiling against the real header
// (/usr/src/habanalabs-1.24.0-1007/include/uapi/drm/habanalabs_accel.h):
//   DRM_IOCTL_HL_WAIT_CS = _IOWR('d', 0x40+0x03, union hl_wait_cs_args) = 0xc0386443
//     -> _IOC type 'd' (0x64), nr 0x43, size 56.   in.timeout_us @8, in.flags @20,
//        out.status @0.  HL_WAIT_CS_STATUS_COMPLETED=0, BUSY=1, TIMEDOUT=2, ABORTED=3.
//        HL_WAIT_CS_FLAGS_INTERRUPT=0x2, INTERRUPT_KERNEL_CQ=0x10, REGISTER_INTERRUPT=0x20.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/ioctl.h>

/* --- UAPI constants (verified on-box) --- */
#define HL_WAIT_CS_FLAGS_INTERRUPT          0x2u
#define HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ 0x10u
#define HL_WAIT_CS_FLAGS_REGISTER_INTERRUPT 0x20u
/* Any interrupt/CQ-wait variant routes to hl_interrupt_wait_ioctl with different
 * timeout/target field layout — never reinterpret timeout_us@8 for these. */
#define WAITCS_INTERRUPT_ANY \
    (HL_WAIT_CS_FLAGS_INTERRUPT | HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ | \
     HL_WAIT_CS_FLAGS_REGISTER_INTERRUPT)

#define HL_WAIT_CS_STATUS_COMPLETED 0u
#define HL_WAIT_CS_STATUS_BUSY      1u
/* 2 = TIMEDOUT, 3 = ABORTED : on a timeout_us=0 poll these mean the CS is gone/dead;
 * stop polling and hand the real (blocking) call back so the caller sees the truth. */

#define WAITCS_IOC_TYPE 'd'
#define WAITCS_IOC_NR   0x43   /* DRM_COMMAND_BASE 0x40 + HL_IOCTL_WAIT_CS 0x03 */
#define WAITCS_IOC_SIZE 56     /* sizeof(union hl_wait_cs_args) — pins struct, kills nr collisions */

/* Minimal mirror of union hl_wait_cs_args (only the fields we touch; exact verified offsets). */
struct wcs_in {
    uint64_t seq_or_addr;          /* 0  seq (CS) / addr (interrupt) — untouched */
    uint64_t timeout_us_or_target; /* 8  timeout_us for the non-interrupt CS/multi-CS wait */
    uint32_t ctx_id;               /* 16 */
    uint32_t flags;                /* 20 */
    uint64_t interrupt_timeout_us; /* 24 */
    uint64_t cq_counters_offset;   /* 32 */
    uint64_t timestamp_handle;     /* 40 */
    uint64_t timestamp_offset;     /* 48 */
};                                 /* == 56 bytes */
struct wcs_out { uint32_t status; uint32_t flags; int64_t ts; uint32_t map; uint32_t pad; };
union wcs_args { struct wcs_in in; struct wcs_out out; };

/* Exactly the size the kernel UAPI defines; copy no more, no less, than the caller's buffer. */
#define WAITCS_ARG_BYTES sizeof(union wcs_args)   /* 56 */

static int (*real_ioctl)(int, unsigned long, ...) = NULL;
static volatile int g_enabled = -1;       /* -1 uninit, 0 off, 1 on */
static long g_max_spin = 2000000;         /* spin budget before falling back to blocking wait */
static long g_yield_mask = 0x3FF;         /* sched_yield() every (mask+1) spins */

/* Thread-safe lazy init: dlsym + getenv are idempotent; worst case a few threads race the
 * first call and each resolves the same pointers / parse the same env. No shared mutable
 * state beyond write-once-stable scalars, so the race is benign. */
static void init_once(void) {
    if (!real_ioctl)
        real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    if (g_enabled < 0) {
        const char *m = getenv("HPU_BUSYPOLL_MAXSPIN");
        if (m) { long v = atol(m); if (v > 0) g_max_spin = v; }
        const char *y = getenv("HPU_BUSYPOLL_YIELD_N");
        if (y) { long v = atol(y); if (v > 0) {  /* round down to 2^k-1 mask */
            long mask = 1; while (mask < v) mask = (mask << 1) | 1; g_yield_mask = mask >> 1; } }
        const char *e = getenv("HPU_BUSYPOLL_WAITCS");
        g_enabled = (e && e[0] == '1') ? 1 : 0;   /* set LAST: ensures tunables are ready first */
    }
}

int ioctl(int fd, unsigned long request, ...) {
    va_list ap; va_start(ap, request); void *arg = va_arg(ap, void *); va_end(ap);

    if (g_enabled < 0 || !real_ioctl) init_once();

    /* Fast path / inert default: only intercept the exact WAIT_CS ioctl when enabled.
     * Match type+nr+size to pin it to union hl_wait_cs_args (== full cmd 0xc0386443),
     * so we never reinterpret another driver's nr-0x43 ioctl or a different-size variant. */
    if (g_enabled == 1 && arg &&
        _IOC_TYPE(request) == WAITCS_IOC_TYPE &&
        _IOC_NR(request)   == WAITCS_IOC_NR &&
        _IOC_SIZE(request) == WAITCS_IOC_SIZE) {

        union wcs_args *a = (union wcs_args *)arg;
        union wcs_args saved;
        memcpy(&saved, a, WAITCS_ARG_BYTES);   /* exactly 56 — no over-read of caller buffer */

        /* Interrupt / CQ / register-interrupt waits: different field semantics — passthrough. */
        if (saved.in.flags & WAITCS_INTERRUPT_ANY)
            return real_ioctl(fd, request, arg);

        /* Caller already asked for a non-blocking poll: nothing to convert. */
        if (saved.in.timeout_us_or_target == 0)
            return real_ioctl(fd, request, arg);

        /* Busy-poll via the driver's own non-sleeping completion check (timeout_us=0). */
        for (long s = 0; s < g_max_spin; s++) {
            memcpy(a, &saved, WAITCS_ARG_BYTES);   /* restore in (driver memsets/clobbers union) */
            a->in.timeout_us_or_target = 0;        /* no-sleep poll */
            int r = real_ioctl(fd, request, a);
            if (r != 0) {                          /* -EINTR/-ETIMEDOUT/etc: hand result to caller */
                return r;
            }
            uint32_t st = a->out.status;
            if (st == HL_WAIT_CS_STATUS_COMPLETED)  /* done — sub-us, no IRQ sleep */
                return r;
            if (st != HL_WAIT_CS_STATUS_BUSY)       /* TIMEDOUT/ABORTED/unknown: stop spinning */
                break;                              /* fall through to the real blocking call */
            if ((s & g_yield_mask) == g_yield_mask)
                sched_yield();
        }
        /* Budget exhausted OR non-BUSY status -> original BLOCKING wait: can never hang,
         * and the caller sees the driver's authoritative result. */
        memcpy(a, &saved, WAITCS_ARG_BYTES);
        return real_ioctl(fd, request, arg);
    }

    /* Everything else (incl. enabled-but-not-WAIT_CS, and the whole disabled default): passthrough. */
    return real_ioctl(fd, request, arg);
}
