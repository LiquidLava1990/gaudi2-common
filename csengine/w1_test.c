/* w1_test.c — W1 on-card acceptance test for libcsengine (the submit core).
 * =============================================================================
 * Extends the M0 PoC's PDMA-only proof to the W1 gate the operator runs:
 *
 *   PART A (ABI + open): library opens a free card, ABI matches the kernel.
 *
 *   PART B (3-engine on-device chain, 1 host wait): build a PDMA->MME->TPC
 *     SIGNAL/WAIT chain entirely with kernel-built SIGNAL/WAIT CSs and drain it
 *     with EXACTLY ONE host wait. This is the M0 result generalized to 3
 *     DIFFERENT engine families (PDMA_0_0=0, MME_0_0=16, TPC_0_0=20) — proving
 *     the on-device monitor/fence carries a cross-FAMILY dependency, not just
 *     PDMA<->PDMA. Gate: host WAIT_CS/step == 1 AND on-device waits armed > 0.
 *
 *   PART C (MME queue 16 accepts a hand-built CB): submit a single NOP CB —
 *     the simplest valid program — to MME queue 16 via cseng_submit_cb, and
 *     host-wait it. This proves a USER_ALLOC CB on the MME QMAN's CP retires
 *     cleanly (the prerequisite for splicing GC-recipe packets into a static CB
 *     on a compute queue in W2). Unlike EDMA, MME is a queue gaudi2_test_queues
 *     validates, so a NOP CB is expected to run.
 *       NOTE: this is the one place a user CB touches a compute CP. A NOP is the
 *       safest possible program (no DMA, no addresses); if the CP still faults,
 *       the classifier prints (a)-vs-(b) and we STOP before any retry. Gate this
 *       part with --partC (default ON); pass --no-partC to skip if you want the
 *       pure sync proof only.
 *
 *   PART D (staged -> 1 completion): a 3-CS STAGED submission (FIRST..LAST) of
 *     NOP CBs across PDMA/MME/TPC; the whole group drains with ONE host wait on
 *     the LAST seq. Proves cseng_staged_* gives one completion for an N-CS chain.
 *     (Default ON; --no-partD to skip. Also touches compute CPs with NOP CBs.)
 *
 *   PART E (mapped-CQ counter): allocate a CQ-counter CB, prove the host can
 *     read it (alloc + mmap path) — the zero-ioctl steady-state wait surface.
 *     (Read-only smoke of the alloc/mmap; the counter is exercised for real in
 *     M2 when a CB's completion writes it.)
 *
 * SAFETY: same envelope as cs_poc — opens only a FREE card (EBUSY => skip), holds
 * the single compute ctx for a few seconds, LOUD on any unexpected errno, prints
 * the (a)-vs-(b) device classifier on a CS failure. Parts C/D submit only NOP
 * CBs to MME/TPC (the family gaudi2_test_queues validates); they do NOT submit
 * LIN_DMA or arbitrary programs.
 *
 * BUILD: bash build.sh   (links libcsengine; no card to compile)
 * RUN:   ./w1_test [--card N] [--nchain N] [--reps N] [--no-partC] [--no-partD]
 * =============================================================================
 */
#define _GNU_SOURCE
#include "libcsengine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>

/* the 3 engine families W1 proves cross-sync across. */
static const int QUEUES[3] = { CSENG_Q_PDMA_0_0, CSENG_Q_MME_0_0, CSENG_Q_TPC_0_0 };
static const char *QNAMES[3] = { "PDMA_0_0(0)", "MME_0_0(16)", "TPC_0_0(20)" };

int main(int argc, char **argv) {
	int want_card=-1, nchain=6, reps=50;
	bool partC=true, partD=true;
	for (int i=1;i<argc;i++){
		if(!strcmp(argv[i],"--card")&&i+1<argc) want_card=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--nchain")&&i+1<argc) nchain=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--reps")&&i+1<argc) reps=atoi(argv[++i]);
		else if(!strcmp(argv[i],"--no-partC")) partC=false;
		else if(!strcmp(argv[i],"--no-partD")) partD=false;
		else if(!strcmp(argv[i],"--help")){
			printf("usage: %s [--card N] [--nchain N>=3] [--reps N] [--no-partC] [--no-partD]\n",argv[0]);
			printf("  Part B uses PDMA(0)->MME(16)->TPC(20) cross-family on-device sync.\n");
			return 0;
		}
	}
	if (nchain < 3) { fprintf(stderr,"--nchain must be >= 3 (need all 3 engine families)\n"); return 2; }

	/* ---------------- PART A : open + ABI ---------------- */
	printf("== PART A : open a free card via libcsengine ==\n");
	cseng_ctx *c = cseng_open(want_card);
	if (!c) {
		fprintf(stderr, "  cseng_open failed: %s\n", cseng_global_last_error());
		fprintf(stderr, "  (NO FREE CARD => rerun when a card has no active serve; dirty device => reboot.)\n");
		return 3;
	}
	printf("  OPEN OK on accel%d (fd=%d)\n", cseng_card_index(c), cseng_fd(c));
	int st = cseng_device_status(c);
	uint32_t bh=0,bs=0; cseng_reset_counts(c,&bh,&bs);
	printf("  device status=%d (0=OPERATIONAL)  reset counters hard=%u soft=%u\n", st, bh, bs);
	uint32_t fsob=0,fmon=0,fcq=0;
	if (cseng_sync_manager(c,&fsob,&fmon,&fcq)==0)
		printf("  sync_mgr: first_sob=%u first_mon=%u first_cq=%u (reserved base, not a leak)\n", fsob,fmon,fcq);

	/* single-signal self-test on PDMA (strike-3 guard, same as PoC). */
	printf("\n== self-test : 1 SIGNAL CS on %s + host wait (strike-3 guard) ==\n", QNAMES[0]);
	{
		cseng_signal sg;
		if (cseng_submit_signal(c, QUEUES[0], &sg) != 0) {
			fprintf(stderr, "  SIGNAL submit failed: %s\n", cseng_last_error(c));
			cseng_classify_now(c, "self-test-SIGNAL-submit", errno);
			cseng_close(c); return 8;
		}
		int r = cseng_host_wait(c, sg.seq, 2000000);
		if (r != 1) {
			fprintf(stderr, "  self-test SIGNAL wait failed: %s\n", cseng_last_error(c));
			cseng_classify_now(c, "self-test-SIGNAL-wait", errno);
			fprintf(stderr, "  STOPPING before the sweep so we don't trip a second reset.\n");
			cseng_close(c); return 9;
		}
		printf("  self-test: single SIGNAL CS retired CLEANLY. Proceeding.\n");
	}

	/* ---------------- PART B : 3-engine on-device chain, 1 host wait --------- */
	printf("\n== PART B : %d-op PDMA->MME->TPC SIGNAL/WAIT chain, ON-DEVICE sync, 1 HOST WAIT/step x %d reps ==\n",
	       nchain, reps);

	/* baseline: naive (host fence per op) for the contrast. */
	cseng_tally_reset(c);
	for (int r=0;r<reps;r++)
		for (int k=0;k<nchain;k++){
			cseng_signal sg;
			if (cseng_submit_signal(c, QUEUES[k%3], &sg)!=0){ fprintf(stderr,"naive sig: %s\n",cseng_last_error(c)); cseng_close(c); return 5; }
			if (cseng_host_wait(c, sg.seq, 1000000)!=1){ fprintf(stderr,"naive wait: %s\n",cseng_last_error(c)); cseng_classify_now(c,"naive-wait",errno); cseng_close(c); return 5; }
		}
	cseng_tally naive_t; cseng_tally_get(c,&naive_t);
	long naive_waits_per_step = naive_t.wait / reps;
	printf("  [naive]    host WAIT_CS/step = %ld  (status-quo: one host fence per op)\n", naive_waits_per_step);

	/* csengine: cross-family on-device chain, 1 host wait/step. */
	cseng_tally_reset(c);
	cseng_reset_armed_counters(c);
	for (int r=0;r<reps;r++) {
		/* hop 0: producer signals on PDMA. */
		cseng_signal prev;
		if (cseng_submit_signal(c, QUEUES[0], &prev)!=0){ fprintf(stderr,"cseng sig0: %s\n",cseng_last_error(c)); cseng_close(c); return 5; }
		uint64_t prev_seq = prev.seq;
		for (int k=1;k<nchain;k++) {
			int q = QUEUES[k%3];                 /* next engine FAMILY in the ring */
			uint64_t wseq=0;
			int aw = cseng_submit_wait_on(c, q, prev_seq, &wseq);   /* q blocks on prev's SOB on-device */
			if (aw < 0){ fprintf(stderr,"cseng wait: %s\n",cseng_last_error(c)); cseng_classify_now(c,"cseng-wait",errno); cseng_close(c); return 5; }
			cseng_signal sg;
			if (cseng_submit_signal(c, q, &sg)!=0){ fprintf(stderr,"cseng sigk: %s\n",cseng_last_error(c)); cseng_close(c); return 5; }
			prev_seq = sg.seq;
		}
		/* ONE host wait for the tail: returns when the whole device chain drained. */
		if (cseng_host_wait(c, prev_seq, 2000000)!=1){
			fprintf(stderr,"cseng tail wait: %s\n",cseng_last_error(c));
			cseng_classify_now(c,"cseng-tail",errno); cseng_close(c); return 5;
		}
	}
	cseng_tally cseng_t; cseng_tally_get(c,&cseng_t);
	long armed=0, already=0; cseng_get_armed_counters(c,&armed,&already);
	long cseng_waits_per_step = cseng_t.wait / reps;
	printf("  [csengine] host WAIT_CS/step = %ld   on-device waits armed/step = %.1f  already-done/step = %.1f\n",
	       cseng_waits_per_step, (double)armed/reps, (double)already/reps);

	bool passB = (cseng_waits_per_step == 1) && (armed > 0);
	printf("  PART B gate: WAIT_CS/step==1 ? %s ; on-device armed>0 ? %s  => %s\n",
	       cseng_waits_per_step==1?"YES":"NO", armed>0?"YES":"NO", passB?"PASS":"FAIL");

	/* ---------------- PART C : MME queue 16 accepts a hand-built NOP CB ------ */
	bool passC = true;
	if (partC) {
		printf("\n== PART C : MME queue 16 accepts a hand-built USER_ALLOC CB (NOP) ==\n");
		cseng_cb cb;
		if (cseng_alloc_cb(c, 4096, &cb)!=0){ fprintf(stderr,"  alloc_cb: %s\n",cseng_last_error(c)); cseng_close(c); return 10; }
		int n = cseng_emit_nop(cb.host, cb.size);
		if (n<0){ fprintf(stderr,"  emit_nop failed\n"); cseng_close(c); return 10; }
		cb.used = (uint32_t)n;
		uint64_t seq=0;
		int r = cseng_submit_cb(c, CSENG_Q_MME_0_0, &cb, /*do_wait*/true, &seq);
		if (r != 0) {
			fprintf(stderr, "  MME NOP CB submit/complete FAILED: %s\n", cseng_last_error(c));
			cseng_classify_now(c, "partC-MME-NOP-CB", errno);
			fprintf(stderr, "  => MME queue 16 did NOT accept a hand-built CB cleanly. This blocks the\n"
			                "     W2 packet-splice route on compute queues (route-1); route-2 (SCAL/\n"
			                "     synLaunch) does not need a hand-built compute CB and is unaffected.\n");
			passC = false;
		} else {
			printf("  MME queue 16 ran a hand-built NOP CB and retired CLEANLY (seq=%llu).\n",
			       (unsigned long long)seq);
		}
		cseng_free_cb(c, &cb);
	} else {
		printf("\n== PART C : SKIPPED (--no-partC) ==\n");
	}

	/* ---------------- PART D : staged submission -> 1 completion ------------ */
	bool passD = true;
	if (partD && passC) {
		printf("\n== PART D : 3-CS STAGED submission (PDMA,MME,TPC NOP CBs) -> ONE host completion ==\n");
		cseng_cb cb[3];
		bool ok=true;
		for (int k=0;k<3 && ok;k++){
			if (cseng_alloc_cb(c, 4096, &cb[k])!=0){ fprintf(stderr,"  alloc_cb[%d]: %s\n",k,cseng_last_error(c)); ok=false; break; }
			cb[k].used = (uint32_t)cseng_emit_nop(cb[k].host, cb[k].size);
		}
		if (!ok){ cseng_close(c); return 11; }
		cseng_tally_reset(c);
		cseng_staged s;
		cseng_staged_begin(c, &s);
		int sr = 0;
		sr |= cseng_staged_add(c, &s, QUEUES[0], &cb[0]);    /* FIRST */
		sr |= cseng_staged_add(c, &s, QUEUES[1], &cb[1]);    /* middle */
		uint64_t last_seq=0;
		sr |= cseng_staged_end(c, &s, QUEUES[2], &cb[2], &last_seq);   /* LAST */
		if (sr != 0) {
			fprintf(stderr, "  staged submit FAILED: %s\n", cseng_last_error(c));
			cseng_classify_now(c, "partD-staged", errno);
			passD = false;
		} else {
			int wr = cseng_host_wait(c, last_seq, 2000000);   /* ONE wait for the whole group */
			cseng_tally st_t; cseng_tally_get(c,&st_t);
			if (wr != 1) {
				fprintf(stderr, "  staged tail wait FAILED: %s\n", cseng_last_error(c));
				cseng_classify_now(c, "partD-staged-wait", errno);
				passD = false;
			} else {
				printf("  staged 3-CS group drained with host WAIT_CS=%ld (expect 1) ; CS submits=%ld\n",
				       st_t.wait, st_t.cs);
				passD = (st_t.wait == 1);
			}
		}
		for (int k=0;k<3;k++) cseng_free_cb(c, &cb[k]);
		printf("  PART D gate: one host completion for the staged group ? %s => %s\n",
		       passD?"YES":"NO", passD?"PASS":"FAIL");
	} else if (partD) {
		printf("\n== PART D : SKIPPED (Part C failed; staged needs compute CBs) ==\n");
	} else {
		printf("\n== PART D : SKIPPED (--no-partD) ==\n");
	}

	/* ---------------- PART E : mapped-CQ counter alloc smoke ---------------- */
	bool passE = true;
	printf("\n== PART E : mapped-CQ counter CB alloc + host-read smoke ==\n");
	{
		cseng_cq cq;
		if (cseng_alloc_cq(c, 64, &cq) != 0) {
			fprintf(stderr, "  cseng_alloc_cq failed: %s\n", cseng_last_error(c));
			fprintf(stderr, "  (the busy-poll / INTERRUPT_KERNEL_CQ wait surface; M2 exercises the counter.)\n");
			passE = false;
		} else {
			volatile uint32_t v = cq.host[0];   /* host-readable mapping */
			printf("  CQ counters CB: handle=0x%llx dev_va=0x%llx size=%u  host[0]=%u (read OK)\n",
			       (unsigned long long)cq.handle, (unsigned long long)cq.dev_va, cq.size, v);
			cseng_free_cq(c, &cq);
		}
	}

	/* ----------------------------- VERDICT ----------------------------- */
	printf("\n================= W1 VERDICT =================\n");
	printf("  PART B (3-engine on-device chain, 1 host wait) : %s\n", passB?"PASS":"FAIL");
	printf("  PART C (MME q16 accepts hand-built CB)         : %s\n", partC?(passC?"PASS":"FAIL"):"skipped");
	printf("  PART D (staged -> 1 completion)                : %s\n", partD?(passD?"PASS":"FAIL"):"skipped");
	printf("  PART E (mapped-CQ counter alloc)               : %s\n", passE?"PASS":"FAIL");

	/* The decisive W1 gate is Part B (the core thesis) + Part C (compute-CP CB
	 * acceptance, the W2 prerequisite). D and E are surface smokes. */
	bool overall = passB && (!partC || passC) && (!partD || passD) && passE;
	printf("\n  ==> W1 %s\n", overall ? "PASS" : "NOT PROVEN");
	if (overall)
		printf("      The submit core works on silicon: cross-family on-device sync collapses to\n"
		       "      1 host wait/step, a compute (MME) CP runs a hand-built CB, staged gives one\n"
		       "      completion, and the mapped-CQ wait surface allocates. Proceed to W2 (route-2).\n");
	else
		printf("      See the failing part above + any (a)-vs-(b) classifier block.\n");

	cseng_close(c);
	printf("\nDONE.\n");
	return overall ? 0 : 7;
}
