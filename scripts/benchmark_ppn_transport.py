#!/usr/bin/env python3
"""Measure exact PPn boundary-copy candidates outside the training hot loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.train.ppn import benchmark_boundary_transport, benchmark_nccl_p2p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--source", default="cuda:0")
    ap.add_argument("--destination", default="cuda:1")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--nccl-p2p", action="store_true",
                    help="run rank-owned NCCL send/recv; launch with torchrun")
    args = ap.parse_args()
    shape = (args.batch, args.width, args.hidden)
    results = []
    transports = ("peer", "pinned_host")
    if args.nccl_p2p:
        result = benchmark_nccl_p2p(
            shape=shape, repeats=args.repeats)
        results.append({
            "transport": result.transport,
            "supported": result.supported,
            "elapsed_seconds": result.elapsed_seconds,
            "bytes_copied": result.bytes_copied,
            "active_bandwidth_gib_s": result.active_bandwidth_gib_s,
            "exact": result.exact,
            "reason": result.reason,
            "boundary_bytes": args.batch * args.width * args.hidden * args.dtype_bytes,
        })
    for transport in transports:
        result = benchmark_boundary_transport(
            shape=shape, source_device=args.source,
            destination_device=args.destination, repeats=args.repeats,
            transport=transport)
        results.append({
            "transport": result.transport,
            "supported": result.supported,
            "elapsed_seconds": result.elapsed_seconds,
            "bytes_copied": result.bytes_copied,
            "active_bandwidth_gib_s": result.active_bandwidth_gib_s,
            "exact": result.exact,
            "reason": result.reason,
            "boundary_bytes": args.batch * args.width * args.hidden * args.dtype_bytes,
        })
    payload = {"shape": list(shape), "results": results}
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
