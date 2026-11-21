/* libcsengine.c — MMEL submit core implementation (W1).
 * =============================================================================
 * Graduates cs_poc.c's silicon-proven device command-submission path into a
 * reusable library. Every ioctl convention here is the one the PoC verified on
 * the box and against the on-box 1.24.0-1007 driver source. See libcsengine.h
 * for the API contract and citations.
 *
 * Build: see build.sh (plain gcc, no Habana libs; compiling needs NO card).
 * =============================================================================
 */
#define _GNU_SOURCE
#include "libcsengine.h"
#include "hl_uapi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

/* ------------------------------------------------------------- bookkeeping */
#define CSENG_MAX_TRACKED 256   /* CBs/DRAM/CQ allocations we auto-free at close */

typedef enum { TRK_NONE=0, TRK_CB_HOST, TRK_DRAM, TRK_CQ } trk_kind;
typedef struct {
	trk_kind kind;
	void    *host;        /* host mapping (CB_HOST) or NULL                  */
	uint64_t dev_va;      /* device VA (for UNMAP)                           */
	uint64_t handle;      /* kernel CB handle (CQ) for DESTROY               */
	uint32_t size;
} trk_entry;

struct cseng_ctx {
	int      fd;
	int      card;
	int      borrowed;    /* fd adopted via cseng_attach -> cseng_close must NOT close it */
	/* baseline device state captured at open (for (a)/(b) classification). */
	uint32_t base_hard, base_soft;
	int      base_status;
	/* ioctl tally (byte-honest host round-trip counter). */
	cseng_tally tally;
	int      trace;       /* CSENG_TRACE env -> 1                            */
	/* armed-monitor counters (decisive on-device-sync assertion). */
	long     waits_armed, waits_already_done;
	/* tracked allocations. */
	trk_entry trk[CSENG_MAX_TRACKED];
	int       n_trk;
	/* last error. */
	char     err[256];
};

/* a process-global last-error for the ctx==NULL open-failure path. */
static char g_global_err[256] = "";

static void set_err(cseng_ctx *c, const char *fmt, ...) {
	int e = errno;
	char buf[160];   /* the caller's message; bounded so the +errno suffix fits */
	va_list ap; va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	/* format the full message straight into each 256-byte destination — buf is
	 * bounded to 160 and the suffix is short, so this never truncates. */
	if (c) snprintf(c->err, sizeof c->err, "%s (errno=%d %s)", buf, e, strerror(e));
	snprintf(g_global_err, sizeof g_global_err, "%s (errno=%d %s)", buf, e, strerror(e));
	errno = e;   /* preserve for the caller */
}

const char *cseng_last_error(const cseng_ctx *c){ return c ? c->err : g_global_err; }
const char *cseng_global_last_error(void){ return g_global_err; }
int cseng_fd(const cseng_ctx *c){ return c ? c->fd : -1; }
int cseng_card_index(const cseng_ctx *c){ return c ? c->card : -1; }

/* ------------------------------------------------------------- ioctl wrapper */
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
static int drm_ioctl(cseng_ctx *c, unsigned long req, void *arg) {
	if (c) {
		if      (req == DRM_IOCTL_HL_CS)      c->tally.cs++;
		else if (req == DRM_IOCTL_HL_WAIT_CS) c->tally.wait++;
		else if (req == DRM_IOCTL_HL_CB)      c->tally.cb++;
		else if (req == DRM_IOCTL_HL_MEMORY)  c->tally.mem++;
		else if (req == DRM_IOCTL_HL_INFO)    c->tally.info++;
		else                                  c->tally.other++;
	}
	int r = ioctl(c ? c->fd : -1, req, arg);
	int saved = errno;
	if (c && c->trace) fprintf(stderr, "  [cseng] ioctl %-8s -> %d%s\n",
	                           ioc_name(req), r, r ? " (err)" : "");
	errno = saved;
	return r;
}

/* raw ioctl for the open path (no ctx yet). */
static int drm_ioctl_fd(int fd, unsigned long req, void *arg) {
	return ioctl(fd, req, arg);
}

void cseng_tally_reset(cseng_ctx *c){ if(c) memset(&c->tally,0,sizeof c->tally); }
void cseng_tally_get(const cseng_ctx *c, cseng_tally *out){ if(c&&out) *out=c->tally; }
void cseng_get_armed_counters(const cseng_ctx *c, long *armed, long *already){
	if(!c) return;
	if(armed)   *armed   = c->waits_armed;
	if(already) *already = c->waits_already_done;
}
void cseng_reset_armed_counters(cseng_ctx *c){ if(c){ c->waits_armed=0; c->waits_already_done=0; } }

/* ------------------------------------------------------------- tracking */
static int trk_add(cseng_ctx *c, trk_kind k, void *host, uint64_t dev_va,
                   uint64_t handle, uint32_t size){
	if (c->n_trk >= CSENG_MAX_TRACKED) { errno=ENOMEM; return -1; }
	trk_entry *e = &c->trk[c->n_trk++];
	e->kind=k; e->host=host; e->dev_va=dev_va; e->handle=handle; e->size=size;
	return 0;
}
static void trk_remove_dev_va(cseng_ctx *c, uint64_t dev_va){
	for (int i=0;i<c->n_trk;i++) if (c->trk[i].dev_va==dev_va && c->trk[i].kind!=TRK_NONE){
		c->trk[i].kind=TRK_NONE; return;
	}
}

/* ------------------------------------------------------------- device status */
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
int cseng_device_status(cseng_ctx *c) {
	if(!c){errno=EINVAL;return -1;}
	struct hl_info_args ia; struct hl_info_device_status ds;
	memset(&ia,0,sizeof ia); memset(&ds,0,sizeof ds);
	ia.op=HL_INFO_DEVICE_STATUS; ia.return_pointer=(uint64_t)(uintptr_t)&ds; ia.return_size=sizeof ds;
	if (drm_ioctl(c, DRM_IOCTL_HL_INFO, &ia)!=0) return -1;
	return (int)ds.status;
}
int cseng_reset_counts(cseng_ctx *c, uint32_t *hard, uint32_t *soft) {
	if(!c){errno=EINVAL;return -1;}
	struct hl_info_args ia; struct hl_info_reset_count rc;
	memset(&ia,0,sizeof ia); memset(&rc,0,sizeof rc);
	ia.op=HL_INFO_RESET_COUNT; ia.return_pointer=(uint64_t)(uintptr_t)&rc; ia.return_size=sizeof rc;
	if (drm_ioctl(c, DRM_IOCTL_HL_INFO, &ia)!=0) return -1;
	if(hard) *hard = rc.hard_reset_cnt;
	if(soft) *soft = rc.soft_reset_cnt;
	return 0;
}
int cseng_sync_manager(cseng_ctx *c, uint32_t *first_sob, uint32_t *first_mon, uint32_t *first_cq){
	if(!c){errno=EINVAL;return -1;}
	struct hl_info_args ia; struct hl_info_sync_manager sm;
	memset(&ia,0,sizeof ia); memset(&sm,0,sizeof sm);
	ia.op=HL_INFO_SYNC_MANAGER; ia.return_pointer=(uint64_t)(uintptr_t)&sm; ia.return_size=sizeof sm;
	if (drm_ioctl(c, DRM_IOCTL_HL_INFO, &ia)!=0){ set_err(c,"HL_INFO_SYNC_MANAGER"); return -1; }
	if(first_sob)*first_sob=sm.first_available_sync_object;
	if(first_mon)*first_mon=sm.first_available_monitor;
	if(first_cq) *first_cq =sm.first_available_cq;
	return 0;
}

cseng_dev_class cseng_classify_now(cseng_ctx *c, const char *what, int cs_errno) {
	if(!c) return CSENG_DEV_UNKNOWN;
	int st = cseng_device_status(c);
	uint32_t hard=0, soft=0; int rcok = (cseng_reset_counts(c,&hard,&soft)==0);
	fprintf(stderr, "\n========== [cseng] CS-FAILURE DIAGNOSIS (%s) ==========\n", what?what:"?");
	fprintf(stderr, "  CS submit errno = %d (%s)\n", cs_errno, strerror(cs_errno));
	fprintf(stderr, "  device status now = %s\n",
	        st<0 ? "INFO failed (device gone?)" : dev_status_str((uint32_t)st));
	if (rcok)
		fprintf(stderr, "  reset counters: hard %u->%u  soft %u->%u  (delta hard=%d soft=%d)\n",
		        c->base_hard, hard, c->base_soft, soft,
		        (int)hard-(int)c->base_hard, (int)soft-(int)c->base_soft);
	bool reset_fired = rcok && ((hard!=c->base_hard) || (soft!=c->base_soft));
	bool nonop = (st==HL_DEVICE_STATUS_IN_RESET || st==HL_DEVICE_STATUS_NEEDS_RESET ||
	              st==HL_DEVICE_STATUS_MALFUNCTION ||
	              st==HL_DEVICE_STATUS_IN_RESET_AFTER_DEVICE_RELEASE || st<0);
	cseng_dev_class cls;
	if (cs_errno==EBUSY || cs_errno==EPERM || reset_fired || nonop) {
		cls = CSENG_DEV_NONOPERATIONAL;
		fprintf(stderr,
		  "  => HYPOTHESIS (b): DEVICE NON-OPERATIONAL / RESET. EBUSY(16) comes ONLY from\n"
		  "     hl_cs_sanity_checks (command_submission.c:1382, !hl_device_operational);\n"
		  "     EPERM(1) from hl_hw_queue_schedule_cs:657. The device left OPERATIONAL.\n"
		  "     ACTION: this card needs a RESET (reboot/driver reload), then retry.\n");
	} else if (cs_errno==EFAULT) {
		cls = CSENG_DEV_SUBMIT_BUG;
		fprintf(stderr,
		  "  => HYPOTHESIS (a): SUBMISSION BUG (CB address). EFAULT(14) from\n"
		  "     gaudi2_validate_cb_address (gaudi2.c:7619): user-CB device VA not in\n"
		  "     SRAM/DRAM/PMMU. Device stayed OPERATIONAL (clean reject).\n");
	} else if (cs_errno==EINVAL) {
		cls = CSENG_DEV_SUBMIT_BUG;
		fprintf(stderr,
		  "  => HYPOTHESIS (a): SUBMISSION BUG (queue/flags). EINVAL(22) from\n"
		  "     validate_queue_index / sanity flag checks. Device OPERATIONAL.\n");
	} else {
		cls = (st==HL_DEVICE_STATUS_OPERATIONAL) ? CSENG_DEV_SUBMIT_BUG : CSENG_DEV_NONOPERATIONAL;
		fprintf(stderr,
		  "  => UNEXPECTED errno; device status above is the tie-breaker:\n"
		  "     OPERATIONAL -> submission bug (a); non-operational -> dirty device (b).\n");
	}
	fprintf(stderr, "=======================================================\n");
	return cls;
}

/* ------------------------------------------------------------- open / close */
cseng_ctx *cseng_open(int want_card) {
	g_global_err[0] = 0;
	char path[64];
	int lo = (want_card >= 0) ? want_card : 0;
	int hi = (want_card >= 0) ? want_card : 7;
	for (int i = lo; i <= hi; i++) {
		snprintf(path, sizeof path, "/dev/accel/accel%d", i);
		int fd = open(path, O_RDWR | O_CLOEXEC);
		if (fd < 0) {
			if (errno == EBUSY) continue;   /* ctx held by a live serve -> skip */
			continue;
		}
		/* confirm a usable ctx with a trivial INFO. */
		struct hl_info_args ia;
		struct { uint64_t sram_base, dram_base, dram_size; uint32_t sram_size, num_ev, dev_id; } ip;
		memset(&ia,0,sizeof ia); memset(&ip,0,sizeof ip);
		ia.op=HL_INFO_HW_IP_INFO; ia.return_pointer=(uint64_t)(uintptr_t)&ip; ia.return_size=sizeof ip;
		if (drm_ioctl_fd(fd, DRM_IOCTL_HL_INFO, &ia) != 0) { close(fd); continue; }

		cseng_ctx *c = calloc(1, sizeof *c);
		if (!c) { close(fd); snprintf(g_global_err,sizeof g_global_err,"calloc ctx failed"); return NULL; }
		c->fd = fd; c->card = i;
		c->trace = getenv("CSENG_TRACE") ? 1 : 0;
		/* baseline device state for classification. */
		c->base_status = cseng_device_status(c);
		cseng_reset_counts(c, &c->base_hard, &c->base_soft);
		if (c->base_status != HL_DEVICE_STATUS_OPERATIONAL &&
		    c->base_status != HL_DEVICE_STATUS_IN_DEVICE_CREATION) {
			snprintf(c->err, sizeof c->err,
			  "card %d opened but device status = %s (NOT operational) — dirty/unstable device "
			  "(hypothesis b); needs a reset before any CS will run", i,
			  c->base_status<0?"unreadable":dev_status_str((uint32_t)c->base_status));
			snprintf(g_global_err, sizeof g_global_err, "%s", c->err);
			close(fd); free(c); return NULL;
		}
		return c;
	}
	snprintf(g_global_err, sizeof g_global_err,
	         "NO FREE CARD (all accel0..7 EBUSY or unopenable). The driver allows ONE compute "
	         "ctx per device; run when a card has no active serve.");
	errno = EBUSY;
	return NULL;
}

/* Adopt an ALREADY-OPEN /dev/accel fd instead of opening a free card. For in-serve
 * use where every card is EBUSY (the worker's libSynapse holds the only ctx). The
 * caller keeps fd ownership: cseng_close does NOT close a borrowed fd. Mirrors
 * cseng_open's ctx setup minus the open(). */
cseng_ctx *cseng_attach(int fd, int card_index) {
	g_global_err[0] = 0;
	if (fd < 0) {
		snprintf(g_global_err, sizeof g_global_err, "cseng_attach: bad fd %d", fd);
		errno = EINVAL; return NULL;
	}
	/* confirm the adopted fd is a usable compute ctx (same INFO probe as open). */
	struct hl_info_args ia;
	struct { uint64_t sram_base, dram_base, dram_size; uint32_t sram_size, num_ev, dev_id; } ip;
	memset(&ia,0,sizeof ia); memset(&ip,0,sizeof ip);
	ia.op=HL_INFO_HW_IP_INFO; ia.return_pointer=(uint64_t)(uintptr_t)&ip; ia.return_size=sizeof ip;
	if (drm_ioctl_fd(fd, DRM_IOCTL_HL_INFO, &ia) != 0) {
		snprintf(g_global_err, sizeof g_global_err,
		         "cseng_attach: fd %d is not a usable accel ctx (INFO ioctl failed)", fd);
		return NULL;
	}
	cseng_ctx *c = calloc(1, sizeof *c);
	if (!c) { snprintf(g_global_err,sizeof g_global_err,"calloc ctx failed"); return NULL; }
	c->fd = fd; c->card = card_index; c->borrowed = 1;
	c->trace = getenv("CSENG_TRACE") ? 1 : 0;
	c->base_status = cseng_device_status(c);
	cseng_reset_counts(c, &c->base_hard, &c->base_soft);
	if (c->base_status != HL_DEVICE_STATUS_OPERATIONAL &&
	    c->base_status != HL_DEVICE_STATUS_IN_DEVICE_CREATION) {
		snprintf(c->err, sizeof c->err,
		  "attached fd %d but device status = %s (NOT operational)", fd,
		  c->base_status<0?"unreadable":dev_status_str((uint32_t)c->base_status));
		snprintf(g_global_err, sizeof g_global_err, "%s", c->err);
		free(c); return NULL;   /* do NOT close — the fd is borrowed */
	}
	return c;
}

void cseng_close(cseng_ctx *c) {
	if (!c) return;
	for (int i = 0; i < c->n_trk; i++) {
		trk_entry *e = &c->trk[i];
		if (e->kind == TRK_NONE) continue;
		if (e->kind == TRK_CB_HOST) {
			union hl_mem_args m; memset(&m,0,sizeof m);
			m.in.op=HL_MEM_OP_UNMAP; m.in.unmap.device_virt_addr=e->dev_va;
			drm_ioctl(c, DRM_IOCTL_HL_MEMORY, &m);
			if (e->host) free(e->host);
		} else if (e->kind == TRK_DRAM) {
			union hl_mem_args m; memset(&m,0,sizeof m);
			m.in.op=HL_MEM_OP_UNMAP; m.in.unmap.device_virt_addr=e->dev_va;
			drm_ioctl(c, DRM_IOCTL_HL_MEMORY, &m);
			/* free the underlying DRAM handle too — we stored it in `handle`. */
			if (e->handle) {
				union hl_mem_args f; memset(&f,0,sizeof f);
				f.in.op=HL_MEM_OP_FREE; f.in.free.handle=e->handle;
				drm_ioctl(c, DRM_IOCTL_HL_MEMORY, &f);
			}
		} else if (e->kind == TRK_CQ) {
			if (e->host && e->host != MAP_FAILED) munmap(e->host, e->size);
			union hl_cb_args cb; memset(&cb,0,sizeof cb);
			cb.in.op=HL_CB_OP_DESTROY; cb.in.cb_handle=e->handle;
			drm_ioctl(c, DRM_IOCTL_HL_CB, &cb);
		}
	}
	if (!c->borrowed) close(c->fd);   /* a cseng_attach'd fd is owned by the caller */
	free(c);
}

/* ------------------------------------------------------------- CB / DRAM alloc */
int cseng_alloc_cb(cseng_ctx *c, uint32_t size, cseng_cb *out) {
	if(!c||!out||size==0){errno=EINVAL;return -1;}
	void *host = NULL;
	if (posix_memalign(&host, 4096, size) != 0) { set_err(c,"posix_memalign %u",size); return -1; }
	memset(host, 0, size);
	union hl_mem_args m; memset(&m,0,sizeof m);
	m.in.op = HL_MEM_OP_MAP;
	m.in.map_host.host_virt_addr = (uint64_t)(uintptr_t)host;
	m.in.map_host.hint_addr = 0;
	m.in.map_host.mem_size = size;
	m.in.flags = HL_MEM_USERPTR;   /* host-vs-device branch (memory.c:1566) */
	if (drm_ioctl(c, DRM_IOCTL_HL_MEMORY, &m) != 0) { set_err(c,"host_memory_map size=%u",size); free(host); return -1; }
	if (m.out.device_virt_addr == 0) { set_err(c,"host_memory_map returned dev_va=0"); free(host); return -1; }
	out->host = host; out->dev_va = m.out.device_virt_addr; out->size = size; out->used = 0;
	if (trk_add(c, TRK_CB_HOST, host, out->dev_va, 0, size) != 0) {
		/* roll back the map so we don't leak an untracked mapping */
		union hl_mem_args u; memset(&u,0,sizeof u);
		u.in.op=HL_MEM_OP_UNMAP; u.in.unmap.device_virt_addr=out->dev_va;
		drm_ioctl(c, DRM_IOCTL_HL_MEMORY, &u);
		free(host); set_err(c,"track table full"); return -1;
	}
	return 0;
}
void cseng_free_cb(cseng_ctx *c, cseng_cb *cb) {
	if(!c||!cb||!cb->dev_va) return;
	union hl_mem_args m; memset(&m,0,sizeof m);
	m.in.op=HL_MEM_OP_UNMAP; m.in.unmap.device_virt_addr=cb->dev_va;
	drm_ioctl(c, DRM_IOCTL_HL_MEMORY, &m);
	if (cb->host) free(cb->host);
	trk_remove_dev_va(c, cb->dev_va);
	cb->host=NULL; cb->dev_va=0; cb->size=0; cb->used=0;
}

uint64_t cseng_alloc_dram(cseng_ctx *c, uint64_t size) {
	if(!c||size==0){errno=EINVAL;return 0;}
	union hl_mem_args a; memset(&a,0,sizeof a);
	a.in.op=HL_MEM_OP_ALLOC; a.in.alloc.mem_size=size; a.in.flags=HL_MEM_CONTIGUOUS;
	if (drm_ioctl(c,DRM_IOCTL_HL_MEMORY,&a)!=0){ set_err(c,"DRAM alloc %llu",(unsigned long long)size); return 0; }
	uint64_t handle = a.out.handle;
	union hl_mem_args m; memset(&m,0,sizeof m);
	m.in.op=HL_MEM_OP_MAP; m.in.map_device.handle=handle; m.in.map_device.hint_addr=0;
	if (drm_ioctl(c,DRM_IOCTL_HL_MEMORY,&m)!=0){
		set_err(c,"DRAM map");
		union hl_mem_args f; memset(&f,0,sizeof f); f.in.op=HL_MEM_OP_FREE; f.in.free.handle=handle;
		drm_ioctl(c,DRM_IOCTL_HL_MEMORY,&f);
		return 0;
	}
	uint64_t dev_va = m.out.device_virt_addr;
	if (trk_add(c, TRK_DRAM, NULL, dev_va, handle, 0) != 0) {
		/* track full: unmap+free to avoid leak, fail */
		union hl_mem_args u; memset(&u,0,sizeof u); u.in.op=HL_MEM_OP_UNMAP; u.in.unmap.device_virt_addr=dev_va;
		drm_ioctl(c,DRM_IOCTL_HL_MEMORY,&u);
		union hl_mem_args f; memset(&f,0,sizeof f); f.in.op=HL_MEM_OP_FREE; f.in.free.handle=handle;
		drm_ioctl(c,DRM_IOCTL_HL_MEMORY,&f);
		set_err(c,"track table full"); return 0;
	}
	return dev_va;
}

/* ------------------------------------------------------------- CQ counters CB */
/* A kernel CB (HL_CB_OP_CREATE with HL_CB_FLAGS_MAP) is mmap-able into userspace
 * and MMU-mapped on the device; the engine's CS-completion CQ monitor increments
 * a u32 inside it (the same mechanism gaudi2_pre_schedule_cs uses,
 * gaudi2.c:11935). We expose it for the INTERRUPT_KERNEL_CQ / busy-poll waits.
 *
 * NOTE: getting the host mapping requires mmap(fd, offset=cb_handle), the
 * documented Habana CB-mmap convention (the cb_handle doubles as the mmap
 * offset). We also fetch the device VA via HL_CB_OP_INFO + GET_DEVICE_VA. */
int cseng_alloc_cq(cseng_ctx *c, uint32_t n_counters, cseng_cq *out) {
	if(!c||!out||n_counters==0){errno=EINVAL;return -1;}
	uint32_t bytes = n_counters * 4;
	if (bytes < 4096) bytes = 4096;   /* driver min PAGE_SIZE */
	union hl_cb_args cb; memset(&cb,0,sizeof cb);
	cb.in.op = HL_CB_OP_CREATE; cb.in.cb_size = bytes; cb.in.flags = HL_CB_FLAGS_MAP;
	if (drm_ioctl(c, DRM_IOCTL_HL_CB, &cb) != 0){ set_err(c,"HL_CB_OP_CREATE cq %u",bytes); return -1; }
	uint64_t handle = cb.out.cb_handle;
	/* device VA */
	union hl_cb_args info; memset(&info,0,sizeof info);
	info.in.op=HL_CB_OP_INFO; info.in.cb_handle=handle; info.in.flags=HL_CB_FLAGS_GET_DEVICE_VA;
	uint64_t dev_va = 0;
	if (drm_ioctl(c, DRM_IOCTL_HL_CB, &info) == 0) dev_va = info.out.device_va;
	/* host mapping: mmap with the cb_handle as the file offset (Habana convention) */
	void *host = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, c->fd, (off_t)handle);
	if (host == MAP_FAILED) {
		set_err(c,"mmap cq counters (handle=0x%llx)",(unsigned long long)handle);
		union hl_cb_args d; memset(&d,0,sizeof d); d.in.op=HL_CB_OP_DESTROY; d.in.cb_handle=handle;
		drm_ioctl(c, DRM_IOCTL_HL_CB, &d);
		return -1;
	}
	memset(host, 0, bytes);
	out->host = (volatile uint32_t*)host; out->handle = handle; out->dev_va = dev_va; out->size = bytes;
	if (trk_add(c, TRK_CQ, host, dev_va, handle, bytes) != 0) {
		munmap(host, bytes);
		union hl_cb_args d; memset(&d,0,sizeof d); d.in.op=HL_CB_OP_DESTROY; d.in.cb_handle=handle;
		drm_ioctl(c, DRM_IOCTL_HL_CB, &d);
		set_err(c,"track table full"); return -1;
	}
	return 0;
}
void cseng_free_cq(cseng_ctx *c, cseng_cq *cq) {
	if(!c||!cq||!cq->handle) return;
	if (cq->host && cq->host != (volatile uint32_t*)MAP_FAILED) munmap((void*)cq->host, cq->size);
	union hl_cb_args d; memset(&d,0,sizeof d); d.in.op=HL_CB_OP_DESTROY; d.in.cb_handle=cq->handle;
	drm_ioctl(c, DRM_IOCTL_HL_CB, &d);
	trk_remove_dev_va(c, cq->dev_va);
	cq->host=NULL; cq->handle=0; cq->dev_va=0; cq->size=0;
}

/* ------------------------------------------------------------- SIGNAL / WAIT */
int cseng_submit_signal(cseng_ctx *c, int queue_index, cseng_signal *out) {
	if(!c){errno=EINVAL;return -1;}
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.queue_index = queue_index;
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	cs.in.cs_flags = HL_CS_FLAGS_SIGNAL;
	if (drm_ioctl(c, DRM_IOCTL_HL_CS, &cs) != 0) { set_err(c,"CS SIGNAL q=%d",queue_index); return -1; }
	if (out) {
		out->seq = cs.out.seq;
		out->sob_off = cs.out.sob_base_addr_offset;
		out->sob_count_before = cs.out.sob_count_before_submission;
	}
	return 0;
}

int cseng_submit_wait_on(cseng_ctx *c, int queue_index, uint64_t prev_seq, uint64_t *out_wait_seq) {
	if(!c){errno=EINVAL;return -1;}
	uint64_t seqarr[1] = { prev_seq };
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.queue_index = queue_index;
	chunk.signal_seq_arr = (uint64_t)(uintptr_t)seqarr;
	chunk.num_signal_seq_arr = 1;
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	cs.in.cs_flags = HL_CS_FLAGS_WAIT;
	cs.out.seq = ~0ULL;   /* sentinel: driver leaves ULLONG_MAX if already done */
	if (drm_ioctl(c, DRM_IOCTL_HL_CS, &cs) != 0) {
		set_err(c,"CS WAIT q=%d on seq=%llu",queue_index,(unsigned long long)prev_seq);
		return -1;
	}
	if (cs.out.seq == ~0ULL) { c->waits_already_done++; return 0; }  /* dep already met */
	c->waits_armed++;
	if (out_wait_seq) *out_wait_seq = cs.out.seq;
	return 1;
}

/* ------------------------------------------------------------- packet builders */
/* gaudi2_gen_signal_cb (gaudi2.c:11967): a single MSG_SHORT to the SOB cfg
 * address adding `inc` (mod-add). We encode the MSG_SHORT exactly: value carries
 * the SOB increment (low bits) with the SOB_MOD bit set (add-mode), ctl carries
 * the cfg address (split addr/base fields per the dcore sync_mngr block). The
 * caller supplies the absolute cfg address (from HL_INFO_SYNC_MANAGER + the
 * mapped sync-mngr block); we do not invent it. */
int cseng_emit_nop(void *buf, size_t buf_sz) {
	if(!buf||buf_sz<8){errno=EINVAL;return -1;}
	uint32_t *p = (uint32_t*)buf;   /* packet_nop {reserved, ctl} */
	p[0] = 0;
	p[1] = (PACKET_NOP << G2_PKT_CTL_OPCODE_SHIFT) | (1u << G2_PKT_CTL_MB_SHIFT);
	return 8;
}

int cseng_gen_signal_cb(void *buf, size_t buf_sz, uint64_t sob_cfg_addr, uint32_t inc) {
	if(!buf||buf_sz<8){errno=EINVAL;return -1;}
	struct g2_packet_msg_short *p = (struct g2_packet_msg_short*)buf;
	/* SOB add: value low bits = inc, bit31 = SOB_MOD (add, not set). */
	p->value = (inc & 0x7FFF) | (1u << G2_SHORT_VAL_SOB_MOD_SHIFT);
	/* ctl: opcode MSG_SHORT, address split into addr(low 22b)/base(2b) of the
	 * sync-mngr block; the caller passes the resolved absolute cfg address and
	 * we encode its low/base halves per the dcore0_sync_mngr_objs map. */
	uint32_t addr_lo = (uint32_t)(sob_cfg_addr & ((1u<<G2_SHORT_CTL_BASE_SHIFT)-1));
	uint32_t base    = (uint32_t)((sob_cfg_addr >> G2_SHORT_CTL_BASE_SHIFT) & 0x3);
	p->ctl = (PACKET_MSG_SHORT << G2_PKT_CTL_OPCODE_SHIFT)
	       | (addr_lo << G2_SHORT_CTL_ADDR_SHIFT)
	       | (base    << G2_SHORT_CTL_BASE_SHIFT)
	       | (1u << G2_PKT_CTL_EB_SHIFT) | (1u << G2_PKT_CTL_MB_SHIFT);
	return (int)sizeof *p;
}

/* gaudi2_gen_wait_cb (gaudi2.c:12060): program the monitor payload (addr+data),
 * ARM the monitor on a SOB with GREATER_OR_EQUAL `target`, then a FENCE that
 * stalls the consumer CP until the monitor fires. We emit:
 *   MSG_SHORT (payload addr) ; MSG_SHORT (payload data) ; MSG_SHORT (ARM) ; FENCE.
 * Addresses are caller-supplied absolutes (resolved from the sync-mngr block). */
int cseng_gen_wait_cb(void *buf, size_t buf_sz,
                      uint64_t mon_arm_cfg_addr, uint64_t mon_payload_addr,
                      uint32_t mon_payload_data, uint32_t sob_id, uint32_t target,
                      uint32_t fence_id) {
	const size_t need = 3*sizeof(struct g2_packet_msg_short) + sizeof(struct g2_packet_fence);
	if(!buf||buf_sz<need){errno=EINVAL;return -1;}
	uint8_t *b = (uint8_t*)buf;
	struct g2_packet_msg_short *m;
	/* 1) program monitor payload address (low 32b of the SOB-update target). */
	m = (struct g2_packet_msg_short*)b;
	m->value = (uint32_t)(mon_payload_addr & 0xFFFFFFFFu);
	m->ctl   = (PACKET_MSG_SHORT << G2_PKT_CTL_OPCODE_SHIFT) | (1u<<G2_PKT_CTL_MB_SHIFT);
	b += sizeof *m;
	/* 2) program monitor payload data. */
	m = (struct g2_packet_msg_short*)b;
	m->value = mon_payload_data;
	m->ctl   = (PACKET_MSG_SHORT << G2_PKT_CTL_OPCODE_SHIFT) | (1u<<G2_PKT_CTL_MB_SHIFT);
	b += sizeof *m;
	/* 3) ARM the monitor: GID=sob_id, MODE=GEQ(1), SOB target value. */
	m = (struct g2_packet_msg_short*)b;
	m->value = (sob_id   << G2_SHORT_VAL_MON_GID_SHIFT)
	         | (1u       << G2_SHORT_VAL_MON_MODE_SHIFT)   /* GREATER_OR_EQUAL */
	         | (target   << G2_SHORT_VAL_MON_SOB_VAL_SHIFT);
	{
		uint32_t addr_lo = (uint32_t)(mon_arm_cfg_addr & ((1u<<G2_SHORT_CTL_BASE_SHIFT)-1));
		uint32_t base    = (uint32_t)((mon_arm_cfg_addr >> G2_SHORT_CTL_BASE_SHIFT) & 0x3);
		m->ctl = (PACKET_MSG_SHORT << G2_PKT_CTL_OPCODE_SHIFT)
		       | (addr_lo << G2_SHORT_CTL_ADDR_SHIFT)
		       | (base    << G2_SHORT_CTL_BASE_SHIFT)
		       | (1u<<G2_PKT_CTL_MB_SHIFT);
	}
	b += sizeof *m;
	/* 4) FENCE: stall this CP on fence_id until the monitor decrements it. */
	struct g2_packet_fence *f = (struct g2_packet_fence*)b;
	f->cfg = (1u  << G2_FENCE_CFG_DEC_VAL_SHIFT)    /* decrement value 1 */
	       | (1u  << G2_FENCE_CFG_TARGET_SHIFT)     /* target 1 */
	       | (fence_id << G2_FENCE_CFG_ID_SHIFT);
	f->ctl = (PACKET_FENCE << G2_PKT_CTL_OPCODE_SHIFT) | (1u<<G2_PKT_CTL_EB_SHIFT) | (1u<<G2_PKT_CTL_MB_SHIFT);
	b += sizeof *f;
	return (int)(b - (uint8_t*)buf);
}

/* ------------------------------------------------------------- generic submit */
int cseng_submit_cb(cseng_ctx *c, int queue_index, const cseng_cb *cb,
                    bool do_wait, uint64_t *out_seq) {
	if(!c||!cb||cb->dev_va==0||cb->used==0){errno=EINVAL;return -1;}
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.cb_handle = cb->dev_va;          /* device VA of user-alloc, MMU-mapped CB */
	chunk.queue_index = queue_index;
	chunk.cb_size = cb->used;
	chunk.cs_chunk_flags = HL_CS_CHUNK_FLAGS_USER_ALLOC_CB;
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	cs.in.cs_flags = 0;
	if (drm_ioctl(c, DRM_IOCTL_HL_CS, &cs) != 0) {
		int e = errno;
		set_err(c,"CS user-CB q=%d",queue_index);
		errno = e;
		return -1;
	}
	uint64_t seq = cs.out.seq;
	if (do_wait) {
		int r = cseng_host_wait(c, seq, 1000000);   /* 1s */
		if (r <= 0) return (r==0) ? -1 : r;
	}
	if (out_seq) *out_seq = seq;
	return 0;
}

/* ------------------------------------------------------------- staged submit */
int cseng_staged_begin(cseng_ctx *c, cseng_staged *s) {
	if(!c||!s){errno=EINVAL;return -1;}
	memset(s,0,sizeof *s); s->active=true; return 0;
}

/* internal: submit one chunk as a staged CS with explicit staged flags. */
static int staged_submit_one(cseng_ctx *c, cseng_staged *s, int queue_index,
                             const cseng_cb *cb, bool is_first, bool is_last,
                             uint64_t *out_seq) {
	struct hl_cs_chunk chunk; memset(&chunk,0,sizeof chunk);
	chunk.cb_handle = cb->dev_va;
	chunk.queue_index = queue_index;
	chunk.cb_size = cb->used;
	chunk.cs_chunk_flags = HL_CS_CHUNK_FLAGS_USER_ALLOC_CB;
	union hl_cs_args cs; memset(&cs,0,sizeof cs);
	cs.in.chunks_execute = (uint64_t)(uintptr_t)&chunk;
	cs.in.num_chunks_execute = 1;
	uint32_t flags = HL_CS_FLAGS_STAGED_SUBMISSION;
	if (is_first) flags |= HL_CS_FLAGS_STAGED_SUBMISSION_FIRST;
	if (is_last)  flags |= HL_CS_FLAGS_STAGED_SUBMISSION_LAST;
	cs.in.cs_flags = flags;
	/* For non-first staged CSs, hl_cs_in.seq must carry the staged-group seq
	 * (the FIRST CS's seq) — habanalabs_accel.h:1663-1674. */
	if (!is_first) cs.in.seq = s->first_seq;
	if (drm_ioctl(c, DRM_IOCTL_HL_CS, &cs) != 0) {
		set_err(c,"staged CS (%s%s) q=%d", is_first?"FIRST":"", is_last?"LAST":"", queue_index);
		return -1;
	}
	if (is_first) s->first_seq = cs.out.seq;
	if (out_seq) *out_seq = cs.out.seq;
	s->n++;
	return 0;
}

int cseng_staged_add(cseng_ctx *c, cseng_staged *s, int queue_index, const cseng_cb *cb) {
	if(!c||!s||!s->active||!cb||cb->dev_va==0||cb->used==0){errno=EINVAL;return -1;}
	bool is_first = (s->n == 0);
	return staged_submit_one(c, s, queue_index, cb, is_first, /*last*/false, NULL);
}

int cseng_staged_end(cseng_ctx *c, cseng_staged *s, int queue_index,
                     const cseng_cb *cb, uint64_t *out_seq) {
	if(!c||!s||!s->active||!cb||cb->dev_va==0||cb->used==0){errno=EINVAL;return -1;}
	bool is_first = (s->n == 0);   /* a 1-CS staged group is FIRST+LAST */
	uint64_t last_seq = 0;
	int r = staged_submit_one(c, s, queue_index, cb, is_first, /*last*/true, &last_seq);
	s->active = false;
	if (r != 0) return r;
	/* The LAST staged CS is the only one with a host completion; wait on it. */
	if (out_seq) *out_seq = last_seq;
	return 0;
}

/* ------------------------------------------------------------- waiting */
int cseng_host_wait(cseng_ctx *c, uint64_t seq, uint64_t timeout_us) {
	if(!c){errno=EINVAL;return -1;}
	union hl_wait_cs_args w; memset(&w,0,sizeof w);
	w.in.seq = seq; w.in.timeout_us = timeout_us;
	if (drm_ioctl(c, DRM_IOCTL_HL_WAIT_CS, &w) != 0) {
		int e=errno; set_err(c,"HL_WAIT_CS seq=%llu",(unsigned long long)seq); errno=e; return -1;
	}
	if (w.out.status == HL_WAIT_CS_STATUS_COMPLETED) return 1;
	if (w.out.status == HL_WAIT_CS_STATUS_BUSY)      return 0;   /* not yet (timeout_us==0) */
	/* TIMEDOUT / ABORTED */
	snprintf(c->err,sizeof c->err,"HL_WAIT_CS seq=%llu status=%u (%s)",
	         (unsigned long long)seq, w.out.status,
	         w.out.status==HL_WAIT_CS_STATUS_TIMEDOUT?"TIMEDOUT":
	         w.out.status==HL_WAIT_CS_STATUS_ABORTED?"ABORTED":"?");
	snprintf(g_global_err,sizeof g_global_err,"%s",c->err);
	errno = (w.out.status==HL_WAIT_CS_STATUS_ABORTED)?EIO:ETIMEDOUT;
	return -1;
}

static long us_since(const struct timespec *t0){
	struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
	return (t.tv_sec-t0->tv_sec)*1000000L + (t.tv_nsec-t0->tv_nsec)/1000L;
}

int cseng_cq_wait(cseng_ctx *c, const cseng_cq *cq, uint32_t counter_idx,
                  uint32_t target, long spin_us) {
	if(!c||!cq||!cq->host||(counter_idx*4u)>=cq->size){errno=EINVAL;return -1;}
	struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
	for (;;) {
		uint32_t v = cq->host[counter_idx];   /* volatile read, zero ioctls */
		if (v >= target) return 1;
		if (spin_us > 0 && us_since(&t0) >= spin_us) return 0;
		/* tight spin; on a real step this is the only host activity besides the
		 * doorbell. A pause hint keeps it polite without a syscall. */
#if defined(__x86_64__) || defined(__i386__)
		__builtin_ia32_pause();
#endif
	}
}

int cseng_cq_wait_interrupt(cseng_ctx *c, const cseng_cq *cq, uint32_t counter_idx,
                            uint32_t target, uint64_t interrupt_id, uint64_t timeout_us) {
	if(!c||!cq||(counter_idx*4u)>=cq->size){errno=EINVAL;return -1;}
	union hl_wait_cs_args w; memset(&w,0,sizeof w);
	/* INTERRUPT_KERNEL_CQ: driver compares (cq_counters_handle + offset) >= target
	 * and sleeps on the interrupt (irq.c:354). */
	w.in.cq_counters_handle = cq->handle;
	w.in.target = target;
	w.in.cq_counters_offset = (uint64_t)counter_idx * 4u;
	w.in.interrupt_timeout_us = timeout_us;
	w.in.flags = HL_WAIT_CS_FLAGS_INTERRUPT | HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ
	           | ((uint32_t)interrupt_id & HL_WAIT_CS_FLAGS_INTERRUPT_MASK);
	if (drm_ioctl(c, DRM_IOCTL_HL_WAIT_CS, &w) != 0) {
		int e=errno; set_err(c,"HL_WAIT_CS INTERRUPT_KERNEL_CQ"); errno=e; return -1;
	}
	return (w.out.status == HL_WAIT_CS_STATUS_COMPLETED) ? 1 : 0;
}

/* ------------------------------------------------------------- patcher */
int cseng_patch_apply(cseng_ctx *c, const cseng_patch_site *site, uint64_t new_dev_va) {
	if(!c||!site||!site->cb||!site->cb->host){errno=EINVAL;return -1;}
	if ((uint64_t)site->cb_byte_offset + sizeof(uint64_t) > site->cb->size) {
		set_err(c,"patch site '%s' offset %u out of CB bounds (size %u)",
		        site->name?site->name:"?", site->cb_byte_offset, site->cb->size);
		errno=ERANGE; return -1;
	}
	/* host mapping is host-visible to the device on the next submit. */
	uint64_t *p = (uint64_t*)((uint8_t*)site->cb->host + site->cb_byte_offset);
	*p = new_dev_va;
	return 0;
}
int cseng_patch_apply_batch(cseng_ctx *c, const cseng_patch_site *sites,
                            size_t n, const uint64_t *new_dev_vas) {
	if(!c||(n&&(!sites||!new_dev_vas))){errno=EINVAL;return -1;}
	int done=0;
	for (size_t i=0;i<n;i++) {
		if (cseng_patch_apply(c, &sites[i], new_dev_vas[i]) != 0) return -1;
		done++;
	}
	return done;
}
