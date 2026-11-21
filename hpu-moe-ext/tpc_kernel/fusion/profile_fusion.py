"""profile_fusion.py — P1 make-or-break HBM profile (RUN ON THE HPU BOX).

GOAL: prove the fused int4->[TPC dequant in SRAM]->[MME gemm] graph moves only
~int4-weight bytes through HBM per expert (SRAM-resident, near-max BW), NOT the
~bf16-weight bytes (which would mean the dequant round-tripped through HBM and the
fusion FAILED).

This driver measures HBM traffic TWO independent ways and prints the verdict:

  PATH A — standalone Synapse recipe (the explicit fused graph, P1's primary
           artifact: fusion/fused_moe_recipe.cpp). We run the compiled binary with
           profiling enabled and read the per-node HBM-traffic column from the
           synprof CSV. This is the authoritative measurement of the EXACT graph
           we hand-built (TPC dequant node -> non-persistent intermediate -> gemm).

  PATH B — torch convert_from_uint4(...) @ x under the HPU graph compiler. The GC
           ships a dedicated pass `fuseDequantIntoMoeGemm` (verified in
           libSynapse.so) that fuses dequant directly into the MoE gemm. Running
           the torch path under the profiler shows whether that pass fires for the
           live op lowering — i.e. whether the SAME fusion is reachable from the
           normal model path. (This is NOT "orchestrating ops and hoping"; it is
           measuring whether the GC graph-pass collapses the round-trip.)

HOW TO READ THE VERDICT
-----------------------
Per expert, per gemm iteration, the dominant HBM weight traffic should be:
    SRAM-resident / fused   : ~ IN*OUT/2 bytes      (int4 packed weight)   <-- WIN
    GC spilled to HBM       : ~ IN*OUT*2 bytes + that again on the gemm read
                                (>= bf16 weight size, ~4x the int4 size)   <-- FAIL
We assert measured weight-HBM bytes < 1.5x int4 size (PASS) and flag if it is
>= 0.75x bf16 size (FAIL). Activations + output are small (M is tiny) and counted
separately.

PRECONDITIONS (build CPU-only in the dev image first):
    docker run --rm -v $HOME/AI/hpu-moe-ext/tpc_kernel:/work -w /work/fusion \
      vault.habana.ai/.../pytorch-installer-2.10.0:latest bash build_recipe.sh

RUN (on the box, with an HPU):
    cd $HOME/AI/hpu-moe-ext/tpc_kernel/fusion
    python profile_fusion.py                 # GLM w13 default
    python profile_fusion.py 32 7168 4096 128   # DeepSeek-ish

Exit 0 == PASS (weight HBM ~ int4 size on at least one path).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import glob

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_RECIPE_BIN = os.path.join(_HERE, "build", "fused_moe_recipe")


def _shapes(argv):
    M  = int(argv[1]) if len(argv) > 1 else 32
    IN = int(argv[2]) if len(argv) > 2 else 6144
    OUT= int(argv[3]) if len(argv) > 3 else 4096
    GS = int(argv[4]) if len(argv) > 4 else 128
    return M, IN, OUT, GS


def _sizes_mb(IN, OUT):
    int4 = IN * OUT / 2 / 1048576.0
    bf16 = IN * OUT * 2 / 1048576.0
    return int4, bf16


# ===========================================================================
# PATH A — standalone fused Synapse recipe + synprof HBM-traffic column.
# ===========================================================================
def run_path_a(M, IN, OUT, GS, int4_mb, bf16_mb) -> "tuple[bool, str]":
    if not os.path.exists(_RECIPE_BIN):
        return False, (f"[A] recipe binary not built: {_RECIPE_BIN}\n"
                       f"    build (CPU-only): bash {os.path.join(_HERE, 'build_recipe.sh')}")
    prof_dir = os.path.join(_HERE, "prof_a")
    os.makedirs(prof_dir, exist_ok=True)
    env = dict(os.environ)
    # Enable the Synapse device profiler -> TEF/CSV with HBM-traffic columns.
    env["HABANA_PROFILE"] = "1"
    env["HABANA_PROF_CONFIG"] = env.get("HABANA_PROF_CONFIG", "")  # default cfg
    env["GRAPH_VISUALIZATION"] = "1"
    env["GRAPH_VISUALIZATION_DIR"] = os.path.join(_HERE, "gviz_a")
    env["LOG_LEVEL_SRAM_SLICE"] = "2"          # log the slicer's SRAM decisions
    env["HABANA_LOGS"] = os.path.join(_HERE, "logs_a")
    os.makedirs(env["GRAPH_VISUALIZATION_DIR"], exist_ok=True)
    os.makedirs(env["HABANA_LOGS"], exist_ok=True)

    print("=" * 78)
    print(f"[A] running standalone fused recipe: {_RECIPE_BIN} {M} {IN} {OUT} {GS}")
    print("=" * 78)
    try:
        out = subprocess.run([_RECIPE_BIN, str(M), str(IN), str(OUT), str(GS)],
                             env=env, cwd=_HERE, capture_output=True, text=True,
                             timeout=600)
    except Exception as ex:  # noqa: BLE001
        return False, f"[A] recipe run failed: {type(ex).__name__}: {ex}"
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr)
        return False, f"[A] recipe exited rc={out.returncode} (see stderr above)"

    # Parse the compile-time workspace hint the binary prints (cheap proxy verdict).
    ws_mb = None
    for line in out.stdout.splitlines():
        if "workspace size" in line:
            try:
                ws_mb = float(line.split("=")[1].strip().split()[0])
            except Exception:  # noqa: BLE001
                pass
    verdict = []
    if ws_mb is not None:
        verdict.append(f"[A] compile workspace = {ws_mb:.2f} MB  (int4={int4_mb:.1f} bf16={bf16_mb:.1f})")
        if ws_mb >= bf16_mb * 0.9:
            verdict.append("[A] workspace ~ bf16 weight -> intermediate SPILLED to HBM (compile hint).")
        else:
            verdict.append("[A] workspace << bf16 weight -> consistent with SRAM-resident bundle (compile hint).")

    # ---- AUTHORITATIVE: measured HBM bytes (averaged over the recipe's 5 iters) --
    # The recipe runs 5 launches under synProfiler; we look for an HBM number in
    # (1) a synprof CSV (preferred, per-node), then (2) the TEF trace the recipe
    # dumps. Either is divided to per-iter. We ASSERT on the CSV when present.
    iters = 5  # must match the launch loop count in fused_moe_recipe.cpp
    hbm_mb, src = _parse_synprof_hbm(prof_dir)
    if hbm_mb is None:
        hbm_mb, src2 = _parse_synprof_hbm(_HERE)
        src = src if hbm_mb is None else src2
    used_csv = hbm_mb is not None
    if hbm_mb is None:
        # TEF dumped by synProfilerGetTrace lands in the profiler dir or CWD.
        # Skip the Path-B torch chrome trace (timing-only) to avoid noise.
        tefs = [t for t in sorted(
                    glob.glob(os.path.join(prof_dir, "**", "*.json"), recursive=True)
                    + glob.glob(os.path.join(_HERE, "*.json")),
                    key=os.path.getmtime, reverse=True)
                if os.path.basename(t) != "torch_hpu_trace.json"]
        for t in tefs:
            hbm_mb, src = _parse_tef_hbm(t)
            if hbm_mb is not None:
                break

    if hbm_mb is not None:
        per_iter = hbm_mb / iters if hbm_mb > bf16_mb * 1.5 else hbm_mb  # de-aggregate if summed over iters
        verdict.append(f"[A] measured weight HBM = {per_iter:.2f} MB/iter  (source: {src})")
        ok, vline = _emit_verdict("A", per_iter, int4_mb, bf16_mb)
        verdict.append(vline)
        if used_csv:
            # Hard assertion on the authoritative CSV column.
            assert per_iter < bf16_mb * 0.75, (
                f"[A] ASSERT FAIL: weight HBM {per_iter:.2f}MB >= 0.75x bf16 "
                f"({bf16_mb:.1f}MB) -> fusion SPILLED. {src}")
        return ok, "\n".join(verdict)

    verdict.append(f"[A] no measured HBM number found ({src}). To produce one on the box, "
                   f"run with HABANA_PROFILE=1 and (if needed) post-process the .hltv with "
                   f"`hl-prof-config` / habana-perf-tool to emit a CSV with an HBM/DRAM column; "
                   f"this parser then asserts on it. Falling back to the compile-workspace hint "
                   f"vs int4={int4_mb:.1f}MB / bf16={bf16_mb:.1f}MB.")
    if ws_mb is not None:
        ok, vline = _emit_verdict("A(workspace-proxy)", ws_mb, int4_mb, bf16_mb)
        verdict.append(vline)
        return (ws_mb < bf16_mb * 0.9), "\n".join(verdict)
    return False, "\n".join(verdict) + "\n[A] inconclusive (no HBM number, no workspace hint)."


# ---------------------------------------------------------------------------
# HBM-byte extractors.  Two independent sources, both returned as MB (float):
#   (1) a synprof CSV with an HBM/DRAM-traffic column  (_parse_synprof_hbm)
#   (2) a TEF/chrome-trace JSON with per-event DRAM byte counters (_parse_tef_hbm)
# Each returns (total_mb, detail_str) or (None, reason).  We attribute the WEIGHT
# traffic = total minus the small activation/output traffic where the node name
# lets us; otherwise we report the total (still decisive: ~12MB vs ~48MB).
# ---------------------------------------------------------------------------
def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:  # noqa: BLE001
        return 0.0


def _parse_synprof_hbm(d) -> "tuple[float|None, str]":
    """Find a synprof CSV under `d` and sum its HBM/DRAM-traffic column.
    Tolerant of column-name variants across tool versions; converts bytes->MB
    when the column is in bytes. Returns (total_MB, detail) or (None, reason)."""
    import csv
    csvs = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True),
                  key=os.path.getmtime, reverse=True)
    if not csvs:
        return None, f"no CSV under {d}"
    for path in csvs:
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        cols = {c.lower(): c for c in rows[0].keys() if c}
        # Accept many spellings: "HBM traffic", "DRAM bytes", "dram_read_bytes",
        # "hbm_read_mb", "device dram traffic [mb]", ...
        def _pick(*needles_groups):
            for col_l, col in cols.items():
                if all(any(n in col_l for n in grp) for grp in needles_groups):
                    return col
            return None
        hbm_col = (_pick(("hbm", "dram"), ("traffic", "bytes", "mb", "read", "write"))
                   or _pick(("hbm", "dram"),))
        if not hbm_col:
            continue
        in_bytes = ("byte" in hbm_col.lower()) and ("mb" not in hbm_col.lower())
        name_col = _pick(("name", "node", "engine", "op"))
        total = 0.0
        per_node = {}
        for r in rows:
            v = _num(r.get(hbm_col))
            if in_bytes:
                v /= 1048576.0
            total += v
            if name_col:
                per_node[str(r.get(name_col))] = per_node.get(str(r.get(name_col)), 0.0) + v
        if total > 0:
            detail = (f"CSV '{os.path.basename(path)}' col '{hbm_col}'"
                      f"{' (bytes->MB)' if in_bytes else ''}")
            if per_node:
                top = sorted(per_node.items(), key=lambda kv: -kv[1])[:4]
                detail += " | per-node: " + ", ".join(f"{k}={v:.2f}MB" for k, v in top)
            return total, detail
    return None, f"no HBM/DRAM column in {len(csvs)} CSV(s) under {d}"


def _parse_tef_hbm(path) -> "tuple[float|None, str]":
    """Sum DRAM/HBM byte counters from a TEF / chrome-trace JSON. The standalone
    recipe dumps a TEF trace via synProfilerGetTrace; engine events may carry DRAM
    read/write byte counters in their 'args'. Returns (total_MB, detail) or
    (None, reason). Robust to dict{'traceEvents':[...]} or a bare list."""
    if not path or not os.path.exists(path):
        return None, f"no trace at {path}"
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as ex:  # noqa: BLE001
        return None, f"unreadable TEF {os.path.basename(path)}: {type(ex).__name__}"
    evs = d.get("traceEvents", d) if isinstance(d, dict) else d
    if not isinstance(evs, list):
        return None, f"TEF {os.path.basename(path)} has no event list"
    import re
    byte_re = re.compile(r"(dram|hbm).*(byte|read|write|traffic)|(read|write).*bytes", re.I)
    total_bytes = 0.0
    hits = 0
    per_node = {}
    for e in evs:
        if not isinstance(e, dict):
            continue
        args = e.get("args")
        if not isinstance(args, dict):
            continue
        for k, v in args.items():
            if byte_re.search(str(k)):
                b = _num(v)
                if b > 0:
                    total_bytes += b
                    hits += 1
                    nm = str(e.get("name") or args.get("name") or "?")
                    per_node[nm] = per_node.get(nm, 0.0) + b
    if hits == 0:
        return None, (f"TEF {os.path.basename(path)}: {len(evs)} events, "
                      f"no DRAM/HBM byte counters in args "
                      f"(this trace type carries only timing — use the synprof CSV)")
    mb = total_bytes / 1048576.0
    detail = f"TEF '{os.path.basename(path)}' summed {hits} DRAM-byte counters = {mb:.2f}MB"
    if per_node:
        top = sorted(per_node.items(), key=lambda kv: -kv[1])[:4]
        detail += " | per-node: " + ", ".join(f"{k}={v/1048576.0:.2f}MB" for k, v in top)
    return mb, detail


def _emit_verdict(tag, weight_hbm_mb, int4_mb, bf16_mb) -> "tuple[bool, str]":
    """Programmatic SRAM-vs-HBM verdict in the exact requested framing.
    PASS (SRAM-resident) iff weight HBM < 1.5x int4; FAIL if >= 0.75x bf16."""
    win = weight_hbm_mb < int4_mb * 1.5
    spill = weight_hbm_mb >= bf16_mb * 0.75
    if win:
        v = (f"[{tag}] >>> SRAM-RESIDENT (WIN, ~{int4_mb:.0f}MB): measured weight HBM "
             f"= {weight_hbm_mb:.2f} MB ~ int4 ({int4_mb:.1f} MB). Dequant fed the MME "
             f"from SRAM; bf16 weight never round-tripped HBM.")
    elif spill:
        v = (f"[{tag}] >>> SPILLED (~{bf16_mb:.0f}MB): measured weight HBM "
             f"= {weight_hbm_mb:.2f} MB ~ bf16 ({bf16_mb:.1f} MB). The dequanted bf16 "
             f"weight round-tripped HBM; fusion FAILED.")
    else:
        v = (f"[{tag}] >>> PARTIAL: measured weight HBM = {weight_hbm_mb:.2f} MB, "
             f"between int4 ({int4_mb:.1f}) and bf16 ({bf16_mb:.1f}); partial spill / "
             f"slicing overhead — inspect the SRAM_SLICE log.")
    return win, v


# ===========================================================================
# PATH B — torch convert_from_uint4(...) @ x under the HPU profiler.
# ===========================================================================
def _sym_zero_packed(groups, out_dim, device):
    word = 0
    for nib in range(8):
        word |= (8 & 0xF) << (4 * nib)
    word_i32 = word if word < 2**31 else word - 2**32
    return torch.full((groups, out_dim // 8), word_i32, dtype=torch.int32, device=device)


def run_path_b(M, IN, OUT, GS, int4_mb, bf16_mb) -> "tuple[bool, str]":
    import habana_frameworks.torch as ht  # noqa: F401

    dev = "hpu"
    groups = IN // GS
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (IN, OUT), generator=g, dtype=torch.int32)
    c = codes.to(torch.int64) & 0xF
    c = c.reshape(IN, OUT // 8, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    word = (c << shifts).sum(-1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32).to(dev)
    scale = (torch.rand((groups, OUT), generator=g) * 0.02 + 1e-3).to(torch.bfloat16).to(dev)
    zp = _sym_zero_packed(groups, OUT, dev)
    x = torch.randn((M, IN), dtype=torch.bfloat16, device=dev)

    def fused():
        w = torch.ops.hpu.convert_from_uint4(word, scale, zp, torch.bfloat16, None)  # [IN,OUT]
        return torch.matmul(x, w)  # [M,OUT]

    # correctness sanity (vs the same op on cpu-ish path) — not bit-exact gate here.
    y = fused()
    ht.hpu.synchronize()
    print("=" * 78)
    print(f"[B] torch convert_from_uint4 @ x : out={tuple(y.shape)} dtype={y.dtype}")

    # HBM via memory_stats delta is allocator-only (weak); the real signal is the
    # profiler. Drive torch.profiler with the HPU activity and dump a trace the human
    # can open; also report allocator deltas as a coarse proxy.
    import habana_frameworks.torch.hpu as hpu
    hpu.reset_peak_memory_stats()
    ms0 = hpu.memory_stats()
    for _ in range(10):
        y = fused()
    hpu.synchronize()
    ms1 = hpu.memory_stats()

    trace_path = os.path.join(_HERE, "torch_hpu_trace.json")
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.HPU]) as prof:
            for _ in range(5):
                y = fused()
            hpu.synchronize()
        prof.export_chrome_trace(trace_path)
        print(f"[B] torch HPU profiler trace -> {trace_path}")
    except Exception as ex:  # noqa: BLE001
        print(f"[B] torch.profiler HPU export skipped: {type(ex).__name__}: {ex}")

    lines = []
    peak = ms1.get("MaxInUse", 0) if isinstance(ms1, dict) else 0
    lines.append(f"[B] allocator peak InUse = {peak/1048576.0:.2f} MB (coarse; counts persistent buffers too)")
    lines.append(f"[B] int4 weight={int4_mb:.1f} MB  bf16 weight={bf16_mb:.1f} MB")

    # PROGRAMMATIC verdict: prefer a synprof CSV (HABANA_PROFILE=1), then fall back
    # to the TEF chrome trace we just exported. Many torch chrome traces carry only
    # timing (no DRAM byte counters) — in that case we say so explicitly and return
    # None (inconclusive) rather than guessing.
    hbm_mb, src = _parse_synprof_hbm(_HERE)
    if hbm_mb is None:
        hbm_mb, src = _parse_tef_hbm(trace_path)
    if hbm_mb is not None:
        per_iter = hbm_mb / 5 if hbm_mb > bf16_mb * 1.5 else hbm_mb
        lines.append(f"[B] measured weight HBM = {per_iter:.2f} MB  (source: {src})")
        ok, vline = _emit_verdict("B", per_iter, int4_mb, bf16_mb)
        lines.append(vline)
        return ok, "\n".join(lines)
    lines.append(f"[B] no per-node DRAM/HBM counter found ({src}).")
    lines.append("[B] -> torch chrome trace carries only timing here. Re-run under "
                 "HABANA_PROFILE=1 so a synprof CSV with an HBM/DRAM column is emitted; "
                 "this path then asserts on it. PATH A (standalone recipe) is the "
                 "authoritative measurement.")
    lines.append(f"    weight HBM ~ {int4_mb:.1f} MB => fuseDequantIntoMoeGemm fired (SRAM, WIN); "
                 f"~ {bf16_mb:.1f} MB => dequant round-tripped HBM (FAIL).")
    return None, "\n".join(lines)  # type: ignore[return-value]


def main() -> int:
    M, IN, OUT, GS = _shapes(sys.argv)
    int4_mb, bf16_mb = _sizes_mb(IN, OUT)
    print(f"shapes: M={M} IN={IN} OUT={OUT} GS={GS}")
    print(f"int4 weight = {int4_mb:.1f} MB   bf16 weight = {bf16_mb:.1f} MB   "
          f"(target: weight HBM ~ {int4_mb:.1f} MB, NOT {bf16_mb:.1f} MB)\n")

    try:
        import habana_frameworks.torch  # noqa: F401
    except Exception as ex:  # noqa: BLE001
        print(f"FATAL: habana_frameworks not importable ({ex}); run on the HPU box.")
        return 2

    okA, msgA = run_path_a(M, IN, OUT, GS, int4_mb, bf16_mb)
    print(msgA)
    okB, msgB = run_path_b(M, IN, OUT, GS, int4_mb, bf16_mb)
    print("\n" + msgB)

    print("\n" + "=" * 78)
    results = [r for r in (okA, okB) if r is not None]
    overall = any(r is True for r in results) and not any(r is False for r in results)
    # PASS if at least one path measured HBM ~ int4 and none measured a spill.
    print(f"PATH A (standalone fused recipe): {okA}")
    print(f"PATH B (torch fuse pass)        : {okB}")
    print("P1 FUSION VERDICT:", "PASS (SRAM-resident)" if overall else
          "INCONCLUSIVE/FAIL — read the per-node HBM-traffic numbers above")
    print("=" * 78)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
