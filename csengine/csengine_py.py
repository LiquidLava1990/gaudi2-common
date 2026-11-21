"""csengine_py.py — ctypes bridge for libcsengine (the MMEL submit core, W1).

A thin, dependency-free (stdlib ctypes only) Python wrapper over libcsengine.so,
callable from inside the vLLM-Gaudi worker process. It holds ONE device context
per CsEngine instance — the SAME single compute ctx the worker owns — so it does
not open a second ctx (the driver allows exactly one per device).

Usage (inside the worker, on a free card / the worker's own card):

    from csengine_py import CsEngine, Q_PDMA_0_0, Q_MME_0_0, Q_TPC_0_0

    eng = CsEngine()                      # opens first free card (or card=N)
    try:
        sig = eng.submit_signal(Q_PDMA_0_0)            # -> Signal(seq, sob_off)
        armed = eng.submit_wait_on(Q_MME_0_0, sig.seq) # MME waits on-device
        sig2 = eng.submit_signal(Q_MME_0_0)
        eng.host_wait(sig2.seq)                          # ONE host wait
        # static per-token CB:
        cb = eng.alloc_cb(4096)
        # ... write packets into cb.host_bytes (a memoryview) ...
        # patch a per-token address in place before the doorbell:
        eng.patch_u64(cb, offset=0x40, dev_va=kv_slot_va)
    finally:
        eng.close()

DESIGN NOTES
- This wrapper does NOT load the GC recipe / SCAL path. That is W2. It exposes
  exactly the W1 submit core: open/close, CB/DRAM/CQ alloc, SIGNAL/WAIT, staged
  submission, host/CQ waits, the address patcher, and the device classifier.
- Errors raise CsEngineError carrying the C-side last-error string + errno.
- The library is NOT internally locked; drive it from one thread (the worker's
  execution thread). If you must call from multiple threads, serialize.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    c_int, c_uint, c_uint16, c_uint32, c_uint64, c_long, c_void_p, c_char_p,
    c_bool, c_size_t, POINTER, Structure, byref,
)
from typing import Optional, Sequence

# ---- queue ids (mirror libcsengine.h) ----
Q_PDMA_0_0 = 0
Q_PDMA_1_0 = 4
Q_MME_0_0 = 16
Q_TPC_0_0 = 20


def _default_lib_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "libcsengine.so")


class CsEngineError(RuntimeError):
    pass


# ============================ ctypes struct mirrors =========================
class _CbT(Structure):
    _fields_ = [
        ("host", c_void_p),
        ("dev_va", c_uint64),
        ("size", c_uint32),
        ("used", c_uint32),
    ]


class _CqT(Structure):
    _fields_ = [
        ("host", c_void_p),     # volatile uint32_t*
        ("handle", c_uint64),
        ("dev_va", c_uint64),
        ("size", c_uint32),
    ]


class _SignalT(Structure):
    _fields_ = [
        ("seq", c_uint64),
        ("sob_off", c_uint32),
        ("sob_count_before", c_uint16),
    ]


class _StagedT(Structure):
    _fields_ = [
        ("first_seq", c_uint64),
        ("n", c_int),
        ("active", c_bool),
    ]


class _PatchSiteT(Structure):
    _fields_ = [
        ("cb", POINTER(_CbT)),
        ("cb_byte_offset", c_uint32),
        ("name", c_char_p),
    ]


class _TallyT(Structure):
    _fields_ = [
        ("cs", c_long), ("wait", c_long), ("cb", c_long),
        ("mem", c_long), ("info", c_long), ("other", c_long),
    ]


# device-state classification enum (mirror libcsengine.h)
DEV_OK = 0
DEV_SUBMIT_BUG = 1
DEV_NONOPERATIONAL = 2
DEV_UNKNOWN = 3


# Python-facing lightweight results
class Signal:
    __slots__ = ("seq", "sob_off", "sob_count_before")

    def __init__(self, seq: int, sob_off: int, sob_count_before: int):
        self.seq = seq
        self.sob_off = sob_off
        self.sob_count_before = sob_count_before

    def __repr__(self):
        return f"Signal(seq={self.seq}, sob_off={self.sob_off})"


class Cb:
    """A USER_ALLOC command buffer. `host_bytes` is a writable memoryview of the
    host-visible mapping; write packets there, set `used`, then submit."""
    __slots__ = ("_t", "_eng")

    def __init__(self, t: _CbT, eng: "CsEngine"):
        self._t = t
        self._eng = eng

    @property
    def dev_va(self) -> int:
        return int(self._t.dev_va)

    @property
    def size(self) -> int:
        return int(self._t.size)

    @property
    def used(self) -> int:
        return int(self._t.used)

    @used.setter
    def used(self, n: int):
        self._t.used = c_uint32(n)

    @property
    def host_bytes(self) -> memoryview:
        """Writable view of the host-visible CB mapping (size bytes)."""
        if not self._t.host:
            raise CsEngineError("CB host pointer is NULL")
        buf = (ctypes.c_ubyte * self._t.size).from_address(self._t.host)
        return memoryview(buf)

    def __repr__(self):
        return f"Cb(dev_va=0x{self.dev_va:x}, size={self.size}, used={self.used})"


class Cq:
    """A mapped CQ-counter CB for the zero-ioctl busy-poll wait."""
    __slots__ = ("_t", "_eng")

    def __init__(self, t: _CqT, eng: "CsEngine"):
        self._t = t
        self._eng = eng

    @property
    def handle(self) -> int:
        return int(self._t.handle)

    @property
    def dev_va(self) -> int:
        return int(self._t.dev_va)

    def read(self, counter_idx: int = 0) -> int:
        """Host-read a counter (volatile) — zero ioctls."""
        if not self._t.host:
            raise CsEngineError("CQ host pointer is NULL")
        if counter_idx * 4 >= self._t.size:
            raise CsEngineError("counter_idx out of range")
        arr = (c_uint32 * (self._t.size // 4)).from_address(self._t.host)
        return int(arr[counter_idx])

    def __repr__(self):
        return f"Cq(handle=0x{self.handle:x}, dev_va=0x{self.dev_va:x})"


# ============================== the bridge class ============================
class CsEngine:
    def __init__(self, card: int = -1, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _default_lib_path())
        self._bind()
        self._ctx = self._lib.cseng_open(c_int(card))
        if not self._ctx:
            err = self._lib.cseng_global_last_error()
            raise CsEngineError(
                f"cseng_open failed: {err.decode() if err else '?'}"
            )

    # ---- prototype binding (explicit argtypes/restype = correctness) ----
    def _bind(self):
        L = self._lib
        L.cseng_open.argtypes = [c_int]
        L.cseng_open.restype = c_void_p
        L.cseng_close.argtypes = [c_void_p]
        L.cseng_close.restype = None
        L.cseng_fd.argtypes = [c_void_p]; L.cseng_fd.restype = c_int
        L.cseng_card_index.argtypes = [c_void_p]; L.cseng_card_index.restype = c_int
        L.cseng_last_error.argtypes = [c_void_p]; L.cseng_last_error.restype = c_char_p
        L.cseng_global_last_error.argtypes = []; L.cseng_global_last_error.restype = c_char_p

        L.cseng_alloc_cb.argtypes = [c_void_p, c_uint32, POINTER(_CbT)]; L.cseng_alloc_cb.restype = c_int
        L.cseng_free_cb.argtypes = [c_void_p, POINTER(_CbT)]; L.cseng_free_cb.restype = None
        L.cseng_alloc_dram.argtypes = [c_void_p, c_uint64]; L.cseng_alloc_dram.restype = c_uint64
        L.cseng_alloc_cq.argtypes = [c_void_p, c_uint32, POINTER(_CqT)]; L.cseng_alloc_cq.restype = c_int
        L.cseng_free_cq.argtypes = [c_void_p, POINTER(_CqT)]; L.cseng_free_cq.restype = None

        L.cseng_submit_signal.argtypes = [c_void_p, c_int, POINTER(_SignalT)]; L.cseng_submit_signal.restype = c_int
        L.cseng_submit_wait_on.argtypes = [c_void_p, c_int, c_uint64, POINTER(c_uint64)]; L.cseng_submit_wait_on.restype = c_int
        L.cseng_get_armed_counters.argtypes = [c_void_p, POINTER(c_long), POINTER(c_long)]; L.cseng_get_armed_counters.restype = None
        L.cseng_reset_armed_counters.argtypes = [c_void_p]; L.cseng_reset_armed_counters.restype = None

        L.cseng_gen_signal_cb.argtypes = [c_void_p, c_size_t, c_uint64, c_uint32]; L.cseng_gen_signal_cb.restype = c_int
        L.cseng_gen_wait_cb.argtypes = [c_void_p, c_size_t, c_uint64, c_uint64, c_uint32, c_uint32, c_uint32, c_uint32]; L.cseng_gen_wait_cb.restype = c_int
        L.cseng_emit_nop.argtypes = [c_void_p, c_size_t]; L.cseng_emit_nop.restype = c_int

        L.cseng_submit_cb.argtypes = [c_void_p, c_int, POINTER(_CbT), c_bool, POINTER(c_uint64)]; L.cseng_submit_cb.restype = c_int
        L.cseng_staged_begin.argtypes = [c_void_p, POINTER(_StagedT)]; L.cseng_staged_begin.restype = c_int
        L.cseng_staged_add.argtypes = [c_void_p, POINTER(_StagedT), c_int, POINTER(_CbT)]; L.cseng_staged_add.restype = c_int
        L.cseng_staged_end.argtypes = [c_void_p, POINTER(_StagedT), c_int, POINTER(_CbT), POINTER(c_uint64)]; L.cseng_staged_end.restype = c_int

        L.cseng_host_wait.argtypes = [c_void_p, c_uint64, c_uint64]; L.cseng_host_wait.restype = c_int
        L.cseng_cq_wait.argtypes = [c_void_p, POINTER(_CqT), c_uint32, c_uint32, c_long]; L.cseng_cq_wait.restype = c_int
        L.cseng_cq_wait_interrupt.argtypes = [c_void_p, POINTER(_CqT), c_uint32, c_uint32, c_uint64, c_uint64]; L.cseng_cq_wait_interrupt.restype = c_int

        L.cseng_patch_apply.argtypes = [c_void_p, POINTER(_PatchSiteT), c_uint64]; L.cseng_patch_apply.restype = c_int

        L.cseng_classify_now.argtypes = [c_void_p, c_char_p, c_int]; L.cseng_classify_now.restype = c_int
        L.cseng_device_status.argtypes = [c_void_p]; L.cseng_device_status.restype = c_int
        L.cseng_reset_counts.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_uint32)]; L.cseng_reset_counts.restype = c_int
        L.cseng_sync_manager.argtypes = [c_void_p, POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32)]; L.cseng_sync_manager.restype = c_int
        L.cseng_tally_reset.argtypes = [c_void_p]; L.cseng_tally_reset.restype = None
        L.cseng_tally_get.argtypes = [c_void_p, POINTER(_TallyT)]; L.cseng_tally_get.restype = None

    # ---- internal: raise with the C last-error ----
    def _err(self, what: str) -> CsEngineError:
        msg = self._lib.cseng_last_error(self._ctx)
        return CsEngineError(f"{what}: {msg.decode() if msg else '?'}")

    def _check_ctx(self):
        if not self._ctx:
            raise CsEngineError("engine is closed")

    # ------------------------------ lifecycle ------------------------------
    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.cseng_close(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def fd(self) -> int:
        self._check_ctx(); return int(self._lib.cseng_fd(self._ctx))

    @property
    def card_index(self) -> int:
        self._check_ctx(); return int(self._lib.cseng_card_index(self._ctx))

    def last_error(self) -> str:
        m = self._lib.cseng_last_error(self._ctx)
        return m.decode() if m else ""

    # ------------------------------ allocation -----------------------------
    def alloc_cb(self, size: int) -> Cb:
        self._check_ctx()
        t = _CbT()
        if self._lib.cseng_alloc_cb(self._ctx, c_uint32(size), byref(t)) != 0:
            raise self._err("alloc_cb")
        return Cb(t, self)

    def free_cb(self, cb: Cb):
        self._check_ctx()
        self._lib.cseng_free_cb(self._ctx, byref(cb._t))

    def alloc_dram(self, size: int) -> int:
        self._check_ctx()
        va = int(self._lib.cseng_alloc_dram(self._ctx, c_uint64(size)))
        if va == 0:
            raise self._err("alloc_dram")
        return va

    def alloc_cq(self, n_counters: int) -> Cq:
        self._check_ctx()
        t = _CqT()
        if self._lib.cseng_alloc_cq(self._ctx, c_uint32(n_counters), byref(t)) != 0:
            raise self._err("alloc_cq")
        return Cq(t, self)

    def free_cq(self, cq: Cq):
        self._check_ctx()
        self._lib.cseng_free_cq(self._ctx, byref(cq._t))

    # ------------------------------ on-device sync -------------------------
    def submit_signal(self, queue_index: int) -> Signal:
        self._check_ctx()
        t = _SignalT()
        if self._lib.cseng_submit_signal(self._ctx, c_int(queue_index), byref(t)) != 0:
            raise self._err(f"submit_signal q={queue_index}")
        return Signal(int(t.seq), int(t.sob_off), int(t.sob_count_before))

    def submit_wait_on(self, queue_index: int, prev_seq: int) -> bool:
        """Returns True if an on-device monitor was ARMED (device carries the
        dependency), False if the signal had already completed. Raises on error."""
        self._check_ctx()
        wseq = c_uint64(0)
        r = self._lib.cseng_submit_wait_on(self._ctx, c_int(queue_index), c_uint64(prev_seq), byref(wseq))
        if r < 0:
            raise self._err(f"submit_wait_on q={queue_index}")
        return r == 1

    def armed_counters(self) -> tuple[int, int]:
        self._check_ctx()
        a = c_long(0); d = c_long(0)
        self._lib.cseng_get_armed_counters(self._ctx, byref(a), byref(d))
        return int(a.value), int(d.value)

    def reset_armed_counters(self):
        self._check_ctx()
        self._lib.cseng_reset_armed_counters(self._ctx)

    # ------------------------------ packet builders ------------------------
    def emit_nop(self, cb: Cb, at: int = 0) -> int:
        """Emit a NOP packet into cb at byte offset `at`; returns bytes written."""
        self._check_ctx()
        addr = cb._t.host + at
        n = self._lib.cseng_emit_nop(c_void_p(addr), c_size_t(cb._t.size - at))
        if n < 0:
            raise CsEngineError("emit_nop failed")
        return int(n)

    def gen_signal_cb(self, cb: Cb, at: int, sob_cfg_addr: int, inc: int = 1) -> int:
        self._check_ctx()
        addr = cb._t.host + at
        n = self._lib.cseng_gen_signal_cb(c_void_p(addr), c_size_t(cb._t.size - at),
                                          c_uint64(sob_cfg_addr), c_uint32(inc))
        if n < 0:
            raise CsEngineError("gen_signal_cb failed")
        return int(n)

    def gen_wait_cb(self, cb: Cb, at: int, mon_arm_cfg_addr: int, mon_payload_addr: int,
                    mon_payload_data: int, sob_id: int, target: int, fence_id: int) -> int:
        self._check_ctx()
        addr = cb._t.host + at
        n = self._lib.cseng_gen_wait_cb(
            c_void_p(addr), c_size_t(cb._t.size - at),
            c_uint64(mon_arm_cfg_addr), c_uint64(mon_payload_addr),
            c_uint32(mon_payload_data), c_uint32(sob_id), c_uint32(target), c_uint32(fence_id))
        if n < 0:
            raise CsEngineError("gen_wait_cb failed")
        return int(n)

    # ------------------------------ generic submit -------------------------
    def submit_cb(self, queue_index: int, cb: Cb, do_wait: bool = False) -> int:
        self._check_ctx()
        seq = c_uint64(0)
        if self._lib.cseng_submit_cb(self._ctx, c_int(queue_index), byref(cb._t),
                                     c_bool(do_wait), byref(seq)) != 0:
            raise self._err(f"submit_cb q={queue_index}")
        return int(seq.value)

    # ------------------------------ staged submit --------------------------
    def staged(self, items: Sequence[tuple[int, Cb]]) -> int:
        """Submit a staged group: items = [(queue_index, Cb), ...]; FIRST..LAST
        set automatically. Returns the LAST CS seq (the single completion)."""
        self._check_ctx()
        if len(items) < 1:
            raise CsEngineError("staged needs >= 1 item")
        s = _StagedT()
        if self._lib.cseng_staged_begin(self._ctx, byref(s)) != 0:
            raise self._err("staged_begin")
        for (q, cb) in items[:-1]:
            if self._lib.cseng_staged_add(self._ctx, byref(s), c_int(q), byref(cb._t)) != 0:
                raise self._err(f"staged_add q={q}")
        last_q, last_cb = items[-1]
        seq = c_uint64(0)
        if self._lib.cseng_staged_end(self._ctx, byref(s), c_int(last_q), byref(last_cb._t), byref(seq)) != 0:
            raise self._err(f"staged_end q={last_q}")
        return int(seq.value)

    # ------------------------------ waiting --------------------------------
    def host_wait(self, seq: int, timeout_us: int = 2_000_000) -> bool:
        """Returns True if COMPLETED, False if not-yet (timeout_us==0 poll).
        Raises on TIMEDOUT/ABORTED/error. timeout_us==0 => non-sleeping poll."""
        self._check_ctx()
        r = self._lib.cseng_host_wait(self._ctx, c_uint64(seq), c_uint64(timeout_us))
        if r < 0:
            raise self._err(f"host_wait seq={seq}")
        return r == 1

    def cq_wait(self, cq: Cq, counter_idx: int, target: int, spin_us: int = 0) -> bool:
        """Busy-poll the mapped CQ counter (zero ioctls). spin_us<=0 spins
        forever. Returns True if target reached, False if deadline hit."""
        self._check_ctx()
        r = self._lib.cseng_cq_wait(self._ctx, byref(cq._t), c_uint32(counter_idx),
                                    c_uint32(target), c_long(spin_us))
        if r < 0:
            raise CsEngineError("cq_wait bad args")
        return r == 1

    def cq_wait_interrupt(self, cq: Cq, counter_idx: int, target: int,
                          interrupt_id: int, timeout_us: int) -> bool:
        self._check_ctx()
        r = self._lib.cseng_cq_wait_interrupt(self._ctx, byref(cq._t), c_uint32(counter_idx),
                                              c_uint32(target), c_uint64(interrupt_id), c_uint64(timeout_us))
        if r < 0:
            raise self._err("cq_wait_interrupt")
        return r == 1

    # ------------------------------ patcher --------------------------------
    def patch_u64(self, cb: Cb, offset: int, dev_va: int, name: bytes = b""):
        """Patch a 64-bit address field in `cb` at byte `offset` to `dev_va`
        (the per-token mutable tensor address). Host-visible to the device on the
        next submit."""
        self._check_ctx()
        site = _PatchSiteT()
        site.cb = ctypes.pointer(cb._t)
        site.cb_byte_offset = c_uint32(offset)
        site.name = name
        if self._lib.cseng_patch_apply(self._ctx, byref(site), c_uint64(dev_va)) != 0:
            raise self._err(f"patch_u64 off={offset}")

    # ------------------------------ diagnostics ----------------------------
    def device_status(self) -> int:
        self._check_ctx()
        return int(self._lib.cseng_device_status(self._ctx))

    def reset_counts(self) -> tuple[int, int]:
        self._check_ctx()
        h = c_uint32(0); s = c_uint32(0)
        if self._lib.cseng_reset_counts(self._ctx, byref(h), byref(s)) != 0:
            raise self._err("reset_counts")
        return int(h.value), int(s.value)

    def sync_manager(self) -> tuple[int, int, int]:
        self._check_ctx()
        a = c_uint32(0); b = c_uint32(0); d = c_uint32(0)
        if self._lib.cseng_sync_manager(self._ctx, byref(a), byref(b), byref(d)) != 0:
            raise self._err("sync_manager")
        return int(a.value), int(b.value), int(d.value)

    def classify_now(self, what: str = "py", cs_errno: int = 0) -> int:
        self._check_ctx()
        return int(self._lib.cseng_classify_now(self._ctx, what.encode(), c_int(cs_errno)))

    def tally_reset(self):
        self._check_ctx()
        self._lib.cseng_tally_reset(self._ctx)

    def tally(self) -> dict:
        self._check_ctx()
        t = _TallyT()
        self._lib.cseng_tally_get(self._ctx, byref(t))
        return {"cs": t.cs, "wait": t.wait, "cb": t.cb, "mem": t.mem, "info": t.info, "other": t.other}


# ------------------------------- self-check --------------------------------
def _selfcheck() -> int:
    """Loads the .so and verifies the symbols bind (NO card needed). Run:
        python3 csengine_py.py --selfcheck
    """
    lib = ctypes.CDLL(_default_lib_path())
    needed = [
        "cseng_open", "cseng_close", "cseng_alloc_cb", "cseng_submit_signal",
        "cseng_submit_wait_on", "cseng_host_wait", "cseng_staged_end",
        "cseng_cq_wait", "cseng_patch_apply", "cseng_gen_signal_cb",
        "cseng_gen_wait_cb", "cseng_classify_now", "cseng_alloc_cq",
    ]
    missing = [s for s in needed if not hasattr(lib, s)]
    if missing:
        print("SELFCHECK FAIL — missing symbols:", missing)
        return 1
    print(f"SELFCHECK PASS — libcsengine.so loaded, {len(needed)} key symbols present.")
    print(f"  path = {_default_lib_path()}")
    return 0


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        raise SystemExit(_selfcheck())
    print(__doc__)
    print("Run `python3 csengine_py.py --selfcheck` to verify the .so binds (no card needed).")
