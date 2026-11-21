/* hl_uapi.h — minimal LOCAL mirror of the verified habanalabs UAPI used by cs_poc.
 *
 * Every constant/struct/offset here is pinned to the on-box kernel UAPI:
 *   /usr/include/drm/habanalabs_accel.h        (installed, 1.24.0-1007)
 *   /usr/src/habanalabs-1.24.0-1007/include/uapi/drm/habanalabs_accel.h
 * and the Gaudi-2 packet ABI:
 *   /usr/src/habanalabs-1.24.0-1007/drivers/accel/habanalabs/include/gaudi2/gaudi2_packets.h
 *
 * We mirror (instead of #include) ONLY so the PoC compiles with plain gcc and is
 * self-documenting about exactly which bytes it relies on. cs_poc verifies sizes at
 * runtime against the real header values and ABORTS LOUDLY on any mismatch.
 *
 * DRM ioctl numbers (drm.h: DRM_IOCTL_BASE='d'(0x64), DRM_COMMAND_BASE=0x40):
 *   HL_IOCTL_INFO=0x00 CB=0x01 CS=0x02 WAIT_CS=0x03 MEMORY=0x04
 *   => DRM_IOCTL_HL_CS = _IOWR('d', 0x42, union hl_cs_args)
 *      DRM_IOCTL_HL_WAIT_CS = _IOWR('d', 0x43, union hl_wait_cs_args)  (==0xc0386443, busypoll-verified)
 */
#ifndef CSENGINE_HL_UAPI_H
#define CSENGINE_HL_UAPI_H

#include <stdint.h>
#include <sys/ioctl.h>

/* ---- ioctl wiring ---- */
#define HL_DRM_BASE   0x40
#define HL_IOCTL_INFO 0x00
#define HL_IOCTL_CB   0x01
#define HL_IOCTL_CS   0x02
#define HL_IOCTL_WAIT 0x03
#define HL_IOCTL_MEM  0x04

/* ---- INFO ---- */
#define HL_INFO_HW_IP_INFO   0
#define HL_INFO_DEVICE_STATUS 4    /* habanalabs_accel.h:1263 — no open ctx needed */
#define HL_INFO_RESET_COUNT  9     /* habanalabs_accel.h:1267 */
#define HL_INFO_SYNC_MANAGER 14

/* device status enum (habanalabs_accel.h:1083) */
#define HL_DEVICE_STATUS_OPERATIONAL                    0
#define HL_DEVICE_STATUS_IN_RESET                       1
#define HL_DEVICE_STATUS_MALFUNCTION                    2
#define HL_DEVICE_STATUS_NEEDS_RESET                    3
#define HL_DEVICE_STATUS_IN_DEVICE_CREATION            4
#define HL_DEVICE_STATUS_IN_RESET_AFTER_DEVICE_RELEASE  5

/* ---- CB ops ---- */
#define HL_CB_OP_CREATE  0
#define HL_CB_OP_DESTROY 1
#define HL_CB_OP_INFO    2
#define HL_CB_FLAGS_MAP            0x1
#define HL_CB_FLAGS_GET_DEVICE_VA  0x2

/* ---- MEM ops ---- */
#define HL_MEM_OP_ALLOC 0
#define HL_MEM_OP_FREE  1
#define HL_MEM_OP_MAP   2
#define HL_MEM_OP_UNMAP 3
#define HL_MEM_CONTIGUOUS 0x1
#define HL_MEM_SHARED     0x2
#define HL_MEM_USERPTR    0x4

/* ---- CS flags (habanalabs_accel.h:1550+) ---- */
#define HL_CS_FLAGS_FORCE_RESTORE          0x1
#define HL_CS_FLAGS_SIGNAL                 0x2
#define HL_CS_FLAGS_WAIT                   0x4
#define HL_CS_FLAGS_TIMESTAMP              0x20
#define HL_CS_FLAGS_STAGED_SUBMISSION      0x40
#define HL_CS_FLAGS_STAGED_SUBMISSION_FIRST 0x80
#define HL_CS_FLAGS_STAGED_SUBMISSION_LAST 0x100

#define HL_CS_CHUNK_FLAGS_USER_ALLOC_CB    0x1

#define HL_WAIT_CS_STATUS_COMPLETED 0
#define HL_WAIT_CS_STATUS_BUSY      1
#define HL_WAIT_CS_STATUS_TIMEDOUT  2
#define HL_WAIT_CS_STATUS_ABORTED   3

/* ---- WAIT_CS flags (habanalabs_accel.h:1754+) — the mapped-CQ busy/interrupt
 *      wait the engine uses for a zero-ioctl / single-completion step ---- */
#define HL_WAIT_CS_FLAGS_INTERRUPT            0x2
#define HL_WAIT_CS_FLAGS_INTERRUPT_MASK       0xFFF00000
#define HL_WAIT_CS_FLAGS_ANY_CQ_INTERRUPT     0xFFF00000
#define HL_WAIT_CS_FLAGS_MULTI_CS             0x4
#define HL_WAIT_CS_FLAGS_INTERRUPT_KERNEL_CQ  0x10
#define HL_WAIT_CS_FLAGS_REGISTER_INTERRUPT   0x20

/* ---- structs (exact UAPI layout) ---- */
struct hl_info_args {
	uint64_t return_pointer;
	uint32_t return_size;
	uint32_t op;
	union { uint32_t ctx_id; uint32_t period_ms; uint32_t pll_index; uint32_t eventfd; uint32_t user_buffer_actual_size; uint32_t sec_attest_nonce; uint32_t array_size; uint32_t fw_sub_opcode; };
	uint32_t pad;
};

struct hl_info_sync_manager { /* HL_INFO_SYNC_MANAGER */
	uint32_t first_available_sync_object;
	uint32_t first_available_monitor;
	uint32_t first_available_cq;
	uint32_t reserved;
};

struct hl_info_device_status { /* HL_INFO_DEVICE_STATUS (habanalabs_accel.h:1438) */
	uint32_t status;
	uint32_t soft_reset_stall;
};

struct hl_info_reset_count {  /* HL_INFO_RESET_COUNT (habanalabs_accel.h:1453) */
	uint32_t hard_reset_cnt;
	uint32_t soft_reset_cnt;
};

struct hl_cb_in {
	uint64_t cb_handle;
	uint32_t op;
	uint32_t cb_size;
	uint32_t ctx_id;
	uint32_t flags;
};
struct hl_cb_out {
	union { uint64_t cb_handle; struct { uint32_t usage_cnt; uint32_t pad; }; uint64_t device_va; };
};
union hl_cb_args { struct hl_cb_in in; struct hl_cb_out out; };

struct hl_cs_chunk {
	union { uint64_t cb_handle; uint64_t signal_seq_arr; uint64_t encaps_signal_seq; };
	uint32_t queue_index;
	union { uint32_t cb_size; uint32_t num_signal_seq_arr; uint32_t encaps_signal_offset; };
	uint32_t cs_chunk_flags;
	uint32_t collective_engine_id;
	uint32_t pad[10];
};
struct hl_cs_in {
	uint64_t chunks_restore;
	uint64_t chunks_execute;
	union { uint64_t seq; uint32_t encaps_sig_handle_id; struct { uint32_t encaps_signals_count; uint32_t encaps_signals_q_idx; }; };
	uint32_t num_chunks_restore;
	uint32_t num_chunks_execute;
	uint32_t timeout;
	uint32_t cs_flags;
	uint32_t ctx_id;
	uint8_t  pad[4];
};
struct hl_cs_out {
	union { uint64_t seq; struct { uint32_t handle_id; uint32_t count; }; };
	uint32_t status;
	uint32_t sob_base_addr_offset;
	uint16_t sob_count_before_submission;
	uint16_t pad[3];
};
union hl_cs_args { struct hl_cs_in in; struct hl_cs_out out; };

struct hl_wait_cs_in {
	union {
		struct { uint64_t seq; uint64_t timeout_us; };
		/* INTERRUPT / INTERRUPT_KERNEL_CQ path (habanalabs_accel.h:1766-1799):
		 * addr|cq_counters_handle for the mapped counter, + target value. */
		struct {
			union { uint64_t addr; uint64_t cq_counters_handle; };
			uint64_t target;
		};
	};
	uint32_t ctx_id;
	uint32_t flags;
	union { struct { uint8_t seq_arr_len; uint8_t pad[7]; }; uint64_t interrupt_timeout_us; };
	uint64_t cq_counters_offset;
	uint64_t timestamp_handle;
	uint64_t timestamp_offset;
};
struct hl_wait_cs_out { uint32_t status; uint32_t flags; int64_t timestamp_nsec; uint32_t cs_completion_map; uint32_t pad; };
union hl_wait_cs_args { struct hl_wait_cs_in in; struct hl_wait_cs_out out; };

struct hl_mem_in {
	union {
		struct { uint64_t mem_size; uint64_t page_size; } alloc;
		struct { uint64_t handle; } free;
		struct { uint64_t hint_addr; uint64_t handle; } map_device;
		struct { uint64_t host_virt_addr; uint64_t hint_addr; uint64_t mem_size; } map_host;
		struct { uint64_t device_virt_addr; } unmap;
		uint64_t raw[3]; /* largest real member (map_host) is 3x u64 = 24 bytes */
	};
	uint32_t op;
	uint32_t flags;
	uint32_t ctx_id;
	uint32_t num_of_elements;
};
struct hl_mem_out { union { uint64_t device_virt_addr; uint64_t handle; }; };
union hl_mem_args { struct hl_mem_in in; struct hl_mem_out out; };

/* ---- ioctl command macros ---- */
#define DRM_IOCTL_HL_INFO    _IOWR('d', HL_DRM_BASE + HL_IOCTL_INFO, struct hl_info_args)
#define DRM_IOCTL_HL_CB      _IOWR('d', HL_DRM_BASE + HL_IOCTL_CB,   union hl_cb_args)
#define DRM_IOCTL_HL_CS      _IOWR('d', HL_DRM_BASE + HL_IOCTL_CS,   union hl_cs_args)
#define DRM_IOCTL_HL_WAIT_CS _IOWR('d', HL_DRM_BASE + HL_IOCTL_WAIT, union hl_wait_cs_args)
#define DRM_IOCTL_HL_MEMORY  _IOWR('d', HL_DRM_BASE + HL_IOCTL_MEM,  union hl_mem_args)

/* ============ Gaudi-2 packet ABI (gaudi2_packets.h) ============ */
enum packet_id {
	PACKET_WREG_32 = 0x1, PACKET_MSG_LONG = 0x3, PACKET_MSG_SHORT = 0x4,
	PACKET_FENCE = 0x8, PACKET_LIN_DMA = 0x9, PACKET_NOP = 0xA,
};
#define G2_PKT_CTL_OPCODE_SHIFT 24
#define G2_PKT_CTL_EB_SHIFT     29
#define G2_PKT_CTL_RB_SHIFT     30
#define G2_PKT_CTL_MB_SHIFT     31

#define G2_SHORT_VAL_SOB_VAL_SHIFT 0
#define G2_SHORT_VAL_SOB_MOD_SHIFT 31
#define G2_SHORT_VAL_MON_GID_SHIFT 0
#define G2_SHORT_VAL_MON_MASK_SHIFT 8
#define G2_SHORT_VAL_MON_MODE_SHIFT 16
#define G2_SHORT_VAL_MON_SOB_VAL_SHIFT 17
#define G2_SHORT_CTL_ADDR_SHIFT 0
#define G2_SHORT_CTL_BASE_SHIFT 22

#define G2_FENCE_CFG_DEC_VAL_SHIFT 0
#define G2_FENCE_CFG_TARGET_SHIFT  16
#define G2_FENCE_CFG_ID_SHIFT      30

#define G2_LIN_DMA_CTL_WRCOMP_SHIFT 0   /* GAUDI2_PKT_LIN_DMA_CTL_WRCOMP_SHIFT */
#define G2_LIN_DMA_CTL_MEMSET_SHIFT 4   /* GAUDI2_PKT_LIN_DMA_CTL_MEMSET_SHIFT */

struct g2_packet_msg_short { uint32_t value; uint32_t ctl; };           /* 8 bytes */
struct g2_packet_fence     { uint32_t cfg;   uint32_t ctl; };           /* 8 bytes */
struct g2_packet_lin_dma   { uint32_t tsize; uint32_t ctl; uint64_t src_addr; uint64_t dst_addr; }; /* 24 */
struct g2_packet_msg_long  { uint32_t value; uint32_t ctl; uint64_t addr; }; /* 16 */

/* ---- self-validating size asserts (must match the real kernel UAPI; the
 *      _IOWR cmd numbers depend on these, so a mismatch = wrong ioctl) ---- */
_Static_assert(sizeof(struct hl_cs_chunk)   == 64, "mirror hl_cs_chunk != 64");
_Static_assert(sizeof(struct hl_cs_in)      == 48, "mirror hl_cs_in != 48");
_Static_assert(sizeof(struct hl_wait_cs_in) == 56, "mirror hl_wait_cs_in != 56");
_Static_assert(sizeof(union  hl_cb_args)    == 24, "mirror hl_cb_args != 24");
_Static_assert(sizeof(union  hl_mem_args)   == 40, "mirror hl_mem_args != 40");
_Static_assert(sizeof(struct hl_info_args)  == 24, "mirror hl_info_args != 24");
_Static_assert(sizeof(struct g2_packet_lin_dma) == 24, "lin_dma pkt != 24");
_Static_assert(sizeof(struct g2_packet_msg_short) == 8, "msg_short pkt != 8");
_Static_assert(sizeof(struct g2_packet_fence) == 8, "fence pkt != 8");
/* mapped-CQ wait fields used by libcsengine — pin their offsets to the kernel
 * UAPI (verified on-box: cq_counters_handle@0, target@8, cq_counters_offset@32). */
_Static_assert(__builtin_offsetof(struct hl_wait_cs_in, cq_counters_handle) == 0,  "cq_counters_handle off");
_Static_assert(__builtin_offsetof(struct hl_wait_cs_in, target) == 8,              "target off");
_Static_assert(__builtin_offsetof(struct hl_wait_cs_in, cq_counters_offset) == 32, "cq_counters_offset off");

#endif /* CSENGINE_HL_UAPI_H */
