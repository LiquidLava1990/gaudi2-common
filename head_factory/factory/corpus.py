"""Calibration corpus for MTP head dumps — Trident's distribution (code + reasoning + tool-use +
multi-agent orchestration). Run through a serve with HPU_MTP_DUMP=1 to produce
(hidden, top-K logits, position) shards. More + on-distribution prompts -> higher accept.

build(n) returns up to n UNIQUE prompts, deterministically ordered (seed-free, reproducible).
The pool is generated programmatically (base tasks x languages x framings) so it scales to ~600
without hand-writing each. Keep it matched to what Trident actually serves."""

# --- code generation: (task, [langs]) — the bulk of Trident's load ----------------------------
_ALGO = [
    "merge two sorted linked lists",
    "implement an LRU cache with O(1) get and put",
    "binary search with a loop invariant proof",
    "topological sort over an adjacency list and detect cycles",
    "Dijkstra's shortest path with a binary heap",
    "edit (Levenshtein) distance with dynamic programming",
    "a token-bucket rate limiter",
    "quicksort with a median-of-three pivot",
    "a trie supporting insert, search, and prefix queries",
    "detect whether a binary tree is height-balanced",
    "an LFU cache with O(1) operations",
    "Kahn's algorithm for topological ordering",
    "union-find with path compression and union by rank",
    "a sliding-window maximum over a stream",
    "KMP substring search with the failure function",
    "a min-heap with decrease-key",
    "reservoir sampling over an unbounded stream",
    "the A* pathfinding algorithm on a grid",
    "a thread-safe bounded blocking queue",
    "longest increasing subsequence in O(n log n)",
    "a B-tree node split on insertion",
    "Floyd's cycle detection on a linked list",
    "matrix multiplication with cache blocking",
    "a bloom filter with k hash functions",
    "Boyer-Moore majority vote",
]
_LANGS = ["Python", "TypeScript", "Go", "Rust", "C++", "Java"]

_SYSTEMS = [
    "Write a Dockerfile for a Python FastAPI app with a multi-stage build; explain each stage.",
    "Write a bash script that finds files over 100MB modified in the last 7 days, sorted by size.",
    "Write a Python async client that fetches all pages of a paginated JSON REST API.",
    "Write a Python decorator that retries with exponential backoff and jitter on exception.",
    "Write a git pre-commit hook that blocks commits containing leftover debug prints.",
    "Design the schema for a multi-tenant SaaS app; justify the tenancy isolation choice.",
    "Write a SQL query for the top 3 customers by total order value, ties broken by name.",
    "Write a regex to validate an IPv4 address and explain each part.",
    "Parse a CSV with quoted fields containing commas without a library; cover the edge cases.",
    "Design a URL shortener: API, storage, and how you avoid collisions at scale.",
    "Write a Kubernetes Deployment + Service manifest for a stateless web app; explain the probes.",
    "Implement a connection pool with health checks and a max-idle timeout.",
    "Write a Terraform module for an S3 bucket with versioning and lifecycle rules.",
    "Design a job queue with retries, dead-letter handling, and idempotency.",
    "Write a Prometheus exporter that reports per-endpoint request latency histograms.",
    "Implement a write-ahead log and explain how recovery replays it.",
    "Design a feature-flag service; how do you evaluate flags with low latency?",
    "Write a streaming JSON parser that handles documents larger than memory.",
]

_DEBUG = [
    "Debug: why does `for i in range(len(a)): a[i+1]=a[i]` raise IndexError? Fix it and explain.",
    "A test fails 1 in 10 runs. Outline a systematic plan to find the race condition.",
    "Production latency spiked 10x after a deploy with no code change. Walk through your diagnosis.",
    "A Python service leaks memory under load. Describe how you'd locate the leak step by step.",
    "Users report intermittent 502s behind a load balancer. Enumerate likely causes and checks.",
    "A SQL query got 100x slower overnight. Explain how you'd find and fix the cause.",
    "A goroutine deadlock appears only under high concurrency. How do you reproduce and fix it?",
    "After a dependency bump, JSON dates are off by one day. Trace the root cause.",
    "A cron job silently stopped running last Tuesday. Reconstruct what to investigate.",
    "CPU sits at 100% but throughput dropped. Reason through what could cause this.",
]

_REASON = [
    "A train leaves at 60mph, another at 40mph toward it 200 miles away. When do they meet? Reason it out.",
    "If all Bloops are Razzies and some Razzies are Lazzies, are some Bloops Lazzies? Justify carefully.",
    "Trace quicksort on [3,1,4,1,5,9,2,6] using the last element as pivot, step by step.",
    "Three switches, one bulb in another room, one trip allowed. How do you find the right switch?",
    "Prove that the square root of 2 is irrational.",
    "You have 8 balls, one heavier, and a balance scale. Find it in 2 weighings; explain.",
    "Explain the CAP theorem with a concrete example of choosing AP over CP.",
    "Estimate the number of piano tuners in Chicago and show your assumptions.",
    "Compute the expected number of coin flips to get two heads in a row.",
    "Explain why the halting problem is undecidable, in plain terms.",
    "Given 100 doors toggled in 100 passes, which remain open and why?",
    "Walk through how the TLS 1.3 handshake establishes a shared secret.",
    "Explain how a B-tree insertion splits a full node, then give the pseudocode.",
    "Compare optimistic and pessimistic concurrency control with a worked example.",
]

_AGENT = [
    "As an agent, break 'add dark mode to a React app' into concrete, ordered subtasks.",
    "Given tools search(query) and read(url), plan the calls to answer: who founded Redis and when?",
    "Plan a multi-agent workflow to audit a 50k-line codebase for security bugs; assign roles.",
    "You are a coordinator with 4 worker agents. Decompose 'migrate Flask to FastAPI' and schedule it.",
    "Given tools run_tests(), edit_file(path, patch), and grep(pattern), fix a failing test methodically.",
    "Design the message protocol for a parent agent delegating to short-lived child agents.",
    "An agent must summarize 200 PDFs under a token budget. Plan a map-reduce strategy.",
    "Plan how a self-improving agent should verify its own code change before committing it.",
    "Given a flaky CI pipeline, design an agent loop that isolates and quarantines the flaky test.",
    "Decompose 'build a CLI that scaffolds a new microservice' into agent subtasks with checkpoints.",
    "You have a researcher, a coder, and a reviewer agent. Orchestrate fixing a reported bug end to end.",
    "Plan the tool calls to refactor a function across 30 files while keeping tests green.",
]

_CONCEPT = [
    "Explain processes vs threads, then when to pick each for a web server.",
    "Explain how garbage collection works in a generational collector.",
    "Explain what a memory barrier is and why lock-free code needs it.",
    "Explain how HTTPS, DNS, and TCP cooperate when you load a web page.",
    "Explain the difference between a process's virtual and physical memory.",
    "Explain how database indexes speed up reads and slow down writes.",
    "Explain eventual consistency and where it's the right trade-off.",
    "Explain how a CPU branch predictor affects tight-loop performance.",
    "Explain the actor model and how it avoids shared-state races.",
    "Explain copy-on-write and where operating systems rely on it.",
]


def _code_prompts():
    out = []
    for task in _ALGO:
        for lang in _LANGS:
            out.append(f"Write {lang} code to {task}. Explain the time and space complexity.")
    return out


# framings applied to the non-code pools to widen the decode distribution without losing meaning
_FRAMINGS = ["", "Think step by step. ", "Be concise but complete. ", "Write production-quality code. "]


def _pool():
    """Deterministic, de-duplicated prompt pool. Code first (largest, most on-distribution)."""
    seen = set()
    pool = []

    def add(p):
        if p not in seen:
            seen.add(p); pool.append(p)

    # interleave so a small n_target still spans categories
    code = _code_prompts()
    others = []
    for base in (_SYSTEMS + _DEBUG + _REASON + _AGENT + _CONCEPT):
        for fr in _FRAMINGS:
            others.append(fr + base)
    # round-robin code vs others
    i = j = 0
    while i < len(code) or j < len(others):
        if i < len(code):
            add(code[i]); i += 1
        if j < len(others):
            add(others[j]); j += 1
    return pool


def build(n_target: int = 80):
    pool = _pool()
    return pool[:n_target] if n_target and n_target < len(pool) else pool


if __name__ == "__main__":
    import json
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    p = build(n) if n else build(10**9)
    if len(sys.argv) > 2 and sys.argv[2] == "--count":
        print(len(p))
    else:
        print(json.dumps(p))
