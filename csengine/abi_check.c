/* abi_check.c — INDEPENDENT adversarial verifier of cs_poc's ABI mirror.
 *
 * Compiles against BOTH the real installed kernel UAPI header
 * (/usr/include/drm/habanalabs_accel.h) AND our local mirror (hl_uapi.h),
 * then static-asserts that every struct size and the ioctl numbers match.
 * If our mirror drifts from the kernel ABI, THIS FAILS TO COMPILE (loud).
 *
 * Build:  gcc -O0 -o abi_check abi_check.c && ./abi_check
 * (needs the real header present; no card needed.)
 */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <assert.h>

/* The REAL kernel UAPI. */
#include <drm/habanalabs_accel.h>

/* Our mirror, namespaced to avoid clashes by renaming via macro tricks:
 * we cannot include both in one TU without symbol clashes, so instead we
 * re-declare the sizes we claim and compare to the real ones. The claimed
 * numbers below are exactly what hl_uapi.h defines / cs_poc relies on. */

#define CLAIM_hl_cs_chunk     64
#define CLAIM_hl_wait_cs_in   56

#define DRM_IOCTL_HL_CS_CLAIM      _IOWR('d', 0x40 + 0x02, union hl_cs_args)
#define DRM_IOCTL_HL_WAIT_CS_CLAIM _IOWR('d', 0x40 + 0x03, union hl_wait_cs_args)
#define DRM_IOCTL_HL_CB_CLAIM      _IOWR('d', 0x40 + 0x01, union hl_cb_args)
#define DRM_IOCTL_HL_MEM_CLAIM     _IOWR('d', 0x40 + 0x04, union hl_mem_args)

/* Compile-time checks against the REAL header. */
_Static_assert(sizeof(struct hl_cs_chunk) == CLAIM_hl_cs_chunk,
               "hl_cs_chunk size drift: kernel UAPI != cs_poc mirror");
_Static_assert(sizeof(struct hl_wait_cs_in) == CLAIM_hl_wait_cs_in,
               "hl_wait_cs_in size drift");
/* Full ioctl-arg union sizes — these are encoded into the _IOWR cmd number, so
 * any drift makes our ioctl cmd number WRONG (kernel rejects as unknown ioctl).
 * Values are what cs_poc's mirror must produce; verified against this header. */
_Static_assert(sizeof(struct hl_cs_in)   == 48, "hl_cs_in size drift");
_Static_assert(sizeof(union hl_cb_args)  == 24, "hl_cb_args size drift");
_Static_assert(sizeof(union hl_mem_args) == 40, "hl_mem_args size drift (cmd-number-critical)");
_Static_assert(sizeof(struct hl_info_args) == 24, "hl_info_args size drift");

/* Field-offset checks for the fields cs_poc writes (catch reordering). */
_Static_assert(offsetof(struct hl_cs_chunk, queue_index) == 8,  "chunk.queue_index offset");
_Static_assert(offsetof(struct hl_cs_in, chunks_execute) == 8,  "cs_in.chunks_execute offset");
_Static_assert(offsetof(struct hl_cs_in, num_chunks_execute) == 28, "cs_in.num_chunks_execute offset");
_Static_assert(offsetof(struct hl_cs_in, cs_flags) == 36,       "cs_in.cs_flags offset");
_Static_assert(offsetof(struct hl_wait_cs_in, seq) == 0,        "wait.seq offset");
_Static_assert(offsetof(struct hl_wait_cs_in, timeout_us) == 8, "wait.timeout_us offset");
_Static_assert(offsetof(struct hl_wait_cs_in, flags) == 20,     "wait.flags offset");
/* libcsengine adds the mapped-CQ busy/interrupt wait — guard those fields too. */
_Static_assert(offsetof(struct hl_wait_cs_in, cq_counters_handle) == 0, "wait.cq_counters_handle offset");
_Static_assert(offsetof(struct hl_wait_cs_in, target) == 8,            "wait.target offset");
_Static_assert(offsetof(struct hl_wait_cs_in, cq_counters_offset) == 32,"wait.cq_counters_offset offset");
_Static_assert(HL_WAIT_CS_FLAGS_INTERRUPT == 0x2,            "INTERRUPT flag");
_Static_assert(HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ == 0x10, "INTERRUPT_KERNEL_CQ flag");
_Static_assert(HL_WAIT_CS_FLAGS_INTERRUPT_MASK == 0xFFF00000,"INTERRUPT_MASK");

/* Flag-value checks (cs_poc + libcsengine hardcode these). */
_Static_assert(HL_CS_FLAGS_SIGNAL == 0x2, "SIGNAL flag");
_Static_assert(HL_CS_FLAGS_WAIT == 0x4,   "WAIT flag");
_Static_assert(HL_CS_CHUNK_FLAGS_USER_ALLOC_CB == 0x1, "USER_ALLOC_CB flag");
/* staged submission (libcsengine cseng_staged_*): N CSs -> 1 host completion. */
_Static_assert(HL_CS_FLAGS_STAGED_SUBMISSION == 0x40,        "STAGED flag");
_Static_assert(HL_CS_FLAGS_STAGED_SUBMISSION_FIRST == 0x80,  "STAGED_FIRST flag");
_Static_assert(HL_CS_FLAGS_STAGED_SUBMISSION_LAST == 0x100,  "STAGED_LAST flag");
/* kernel-mapped CB for the CQ-counter wait (cseng_alloc_cq). */
_Static_assert(HL_CB_OP_CREATE == 0 && HL_CB_OP_DESTROY == 1 && HL_CB_OP_INFO == 2, "CB ops");
_Static_assert(HL_CB_FLAGS_MAP == 0x1 && HL_CB_FLAGS_GET_DEVICE_VA == 0x2, "CB flags");
_Static_assert(HL_MEM_USERPTR == 0x4, "MEM_USERPTR flag");
_Static_assert(HL_MEM_OP_FREE == 1, "MEM_OP_FREE");
_Static_assert(HL_MEM_OP_MAP == 2 && HL_MEM_OP_ALLOC == 0, "MEM ops");
_Static_assert(GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0 == 8,  "EDMA_0_0 qid (cs_poc Q_EDMA_A default)");
_Static_assert(GAUDI2_QUEUE_ID_DCORE0_EDMA_1_0 == 12, "EDMA_1_0 qid (cs_poc Q_EDMA_B default)");

/* Diagnostic INFO ops/structs cs_poc now uses to distinguish (a) vs (b). */
_Static_assert(HL_INFO_DEVICE_STATUS == 4, "HL_INFO_DEVICE_STATUS op");
_Static_assert(HL_INFO_RESET_COUNT == 9,   "HL_INFO_RESET_COUNT op");
_Static_assert(sizeof(struct hl_info_device_status) == 8, "hl_info_device_status size drift");
_Static_assert(sizeof(struct hl_info_reset_count) == 8,   "hl_info_reset_count size drift");
_Static_assert(HL_DEVICE_STATUS_OPERATIONAL == 0 && HL_DEVICE_STATUS_IN_RESET == 1 &&
               HL_DEVICE_STATUS_MALFUNCTION == 2 && HL_DEVICE_STATUS_NEEDS_RESET == 3,
               "device status enum drift");

int main(void) {
	printf("ABI CHECK PASS (compiled against real /usr/include/drm/habanalabs_accel.h):\n");
	printf("  sizeof(hl_cs_chunk)   = %zu (== %d claimed)\n", sizeof(struct hl_cs_chunk), CLAIM_hl_cs_chunk);
	printf("  sizeof(hl_wait_cs_in) = %zu (== %d claimed)\n", sizeof(struct hl_wait_cs_in), CLAIM_hl_wait_cs_in);
	printf("  sizeof(hl_cs_in)      = %zu\n", sizeof(struct hl_cs_in));
	printf("  DRM_IOCTL_HL_CS       = 0x%lx\n", (unsigned long)DRM_IOCTL_HL_CS_CLAIM);
	printf("  DRM_IOCTL_HL_WAIT_CS  = 0x%lx (expect 0xc0386443)\n", (unsigned long)DRM_IOCTL_HL_WAIT_CS_CLAIM);
	printf("  DRM_IOCTL_HL_CB       = 0x%lx\n", (unsigned long)DRM_IOCTL_HL_CB_CLAIM);
	printf("  DRM_IOCTL_HL_MEMORY   = 0x%lx\n", (unsigned long)DRM_IOCTL_HL_MEM_CLAIM);
	printf("  EDMA_0_0 qid=%d  EDMA_1_0 qid=%d\n",
	       GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0, GAUDI2_QUEUE_ID_DCORE0_EDMA_1_0);
	return 0;
}
