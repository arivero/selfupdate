#!/usr/bin/env python3
"""Sample resident GPU memory from nvidia-smi.

This complements the in-process PyTorch `vram_gb` metrics. PyTorch reports
allocated/reserved tensors for one process; nvidia-smi reports device-resident
memory including CUDA context and allocator overhead, which is the number that
matters for packing jobs onto a card.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def gpu_uuid_to_index() -> dict[str, str]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ])
    mapping = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, uuid = [x.strip() for x in line.split(",", 1)]
        mapping[uuid] = idx
    return mapping


def ps_commands() -> dict[str, str]:
    out = _run(["ps", "-eo", "pid=,cmd="])
    cmds = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        cmds[pid] = cmd
    return cmds


def compute_apps(
    uuid_map: dict[str, str],
    cmds: dict[str, str],
    allowed_gpus: set[str] | None,
) -> list[dict[str, str]]:
    out = _run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",", 3)]
        if len(parts) != 4:
            continue
        uuid, pid, process_name, used_mb = parts
        gpu = uuid_map.get(uuid, uuid)
        if allowed_gpus is not None and gpu not in allowed_gpus:
            continue
        rows.append({
            "gpu": gpu,
            "pid": pid,
            "process_name": process_name,
            "used_memory_mb": used_mb,
            "cmd": cmds.get(pid, ""),
        })
    return rows


def write_summary(csv_path: Path, summary_path: Path) -> None:
    if not csv_path.exists():
        return
    peaks: dict[tuple[str, str, str, str], int] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                used = int(row["used_memory_mb"])
            except (KeyError, ValueError):
                continue
            key = (
                row.get("gpu", ""),
                row.get("pid", ""),
                row.get("process_name", ""),
                row.get("cmd", ""),
            )
            peaks[key] = max(peaks.get(key, 0), used)
    lines = [
        "| gpu | pid | peak_resident_gb | process | command |",
        "|:---:|---:|---:|:---|:---|",
    ]
    for (gpu, pid, proc, cmd), used in sorted(peaks.items(), key=lambda kv: (kv[0][0], -kv[1])):
        short_cmd = cmd.replace("|", "\\|")
        if len(short_cmd) > 140:
            short_cmd = short_cmd[:137] + "..."
        lines.append(f"| {gpu} | {pid} | {used / 1024:.2f} | `{proc}` | `{short_cmd}` |")
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/gpu_memory_samples.csv")
    ap.add_argument("--summary", default="runs/gpu_memory_summary.md")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means run forever")
    ap.add_argument("--gpus", default="", help="comma-separated physical GPU ids to include")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = Path(args.summary)
    fields = ["ts", "gpu", "pid", "process_name", "used_memory_mb", "cmd"]
    write_header = not out.exists() or out.stat().st_size == 0
    start = time.time()
    allowed_gpus = {g.strip() for g in args.gpus.split(",") if g.strip()} or None
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        while True:
            ts = datetime.now().isoformat(timespec="seconds")
            try:
                uuid_map = gpu_uuid_to_index()
                cmds = ps_commands()
                for row in compute_apps(uuid_map, cmds, allowed_gpus):
                    writer.writerow({"ts": ts, **row})
                f.flush()
                write_summary(out, summary)
            except Exception as exc:  # keep long samplers alive across transient nvidia-smi failures
                writer.writerow({
                    "ts": ts,
                    "gpu": "",
                    "pid": "",
                    "process_name": "ERROR",
                    "used_memory_mb": "",
                    "cmd": repr(exc),
                })
                f.flush()
            if args.duration and time.time() - start >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
