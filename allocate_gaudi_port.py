#!/usr/bin/env python3
"""Allocate a unique Gaudi model/container host port.

Default range: 18274-18320.
Reads live listening TCP ports and GAUDI_PORT_REGISTRY.md, then prints the
first free port. With --reserve NAME, appends a reservation row to the registry.
"""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_REGISTRY = Path("$HOME/AI/GAUDI_PORT_REGISTRY.md")
DEFAULT_START = 18274
DEFAULT_END = 18320
NEVER_USE = {8000, 18273}


def live_ports() -> set[int]:
    ports: set[int] = set()
    try:
        out = subprocess.check_output(["ss", "-ltnH"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        out = ""
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        if ":" not in local:
            continue
        raw = local.rsplit(":", 1)[-1]
        if raw.isdigit():
            ports.add(int(raw))
    return ports


def registry_ports(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ports: set[int] = set()
    pattern = re.compile(r"^\|\s*(\d{2,5})\s*\|\s*([^|]+)\|")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match and match.group(2).strip().lower() not in {"next-free", "available"}:
            ports.add(int(match.group(1)))
    return ports


def port_is_connectable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def allocate(start: int, end: int, registry: Path) -> int:
    used = live_ports() | registry_ports(registry) | NEVER_USE
    for port in range(start, end + 1):
        if port in used:
            continue
        if port_is_connectable(port):
            continue
        return port
    raise SystemExit(f"No free Gaudi ports in {start}-{end}")


def reserve(registry: Path, port: int, name: str, note: str) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    if not registry.exists():
        registry.write_text(
            "# Gaudi Port Registry\n\n"
            "| Port | Status | Container / service | Endpoint | Notes |\n"
            "| --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    stamp = datetime.now().strftime("%Y-%m-%d")
    row = f"| {port} | reserved | {name} |  | {note or ('Reserved ' + stamp)} |\n"
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--reserve", metavar="NAME", help="append a reservation row")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    port = allocate(args.start, args.end, args.registry)
    if args.reserve:
        reserve(args.registry, port, args.reserve, args.note)
    print(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
