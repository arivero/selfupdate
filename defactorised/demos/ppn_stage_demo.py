#!/usr/bin/env python3
"""One CPU-safe PPn stage, with explicit detached boundary handoffs.

Run this through ``ppn_demo.sh``.  The demo keeps the architectural laws
visible: a stage owns one transform, consumes only its upstream boundary,
and atomically publishes a detached value to the next stage.  There is no
downstream-to-upstream dependency.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--fail-stage", type=int, default=-1,
                        help="demo the coordinator's sibling-failure handling")
    return parser.parse_args()


def packet_path(root: Path, sender: int, receiver: int, tile: int) -> Path:
    return root / "boundaries" / f"stage{sender}_to_{receiver}_tile{tile:04d}.json"


def wait_for_packet(path: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            time.sleep(0.02)
    raise TimeoutError(f"timed out waiting for upstream boundary {path}")


def publish(path: Path, payload: dict) -> None:
    """Publish one complete envelope; readers never observe partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)


def validate(packet: dict, *, launch_id: str, sender: int,
             receiver: int, tile: int) -> list[float]:
    expected = {
        "launch_id": launch_id,
        "from_stage": sender,
        "to_stage": receiver,
        "tile": tile,
    }
    observed = {name: packet.get(name) for name in expected}
    if observed != expected:
        raise ValueError(f"misrouted boundary: expected {expected}, got {observed}")
    value = packet.get("value")
    if not isinstance(value, list) or not all(isinstance(x, (int, float)) for x in value):
        raise TypeError(f"boundary value must be a numeric list, got {value!r}")
    return [float(x) for x in value]


def stage_transform(value: list[float], stage: int) -> list[float]:
    """Stand-in for the blocks owned by this stage.

    Returning a fresh list makes the boundary's detach/copy semantics explicit.
    A tensor implementation would use ``hidden.detach()`` at the same point.
    """
    local_increment = float(stage + 1)
    return [x + local_increment for x in value]


def main() -> int:
    args = parse_args()
    if args.stages < 1 or not 0 <= args.stage < args.stages:
        raise SystemExit("require 0 <= --stage < --stages and at least one stage")
    if args.tiles < 1:
        raise SystemExit("--tiles must be positive")
    if args.stage == args.fail_stage:
        raise RuntimeError(f"intentional demo failure in stage {args.stage}")

    for tile in range(args.tiles):
        if args.stage == 0:
            value = [float(tile), float(tile + 1), float(tile + 2)]
        else:
            source = packet_path(args.work_dir, args.stage - 1, args.stage, tile)
            packet = wait_for_packet(source, args.timeout_seconds)
            value = validate(packet, launch_id=args.launch_id,
                             sender=args.stage - 1, receiver=args.stage, tile=tile)

        output = stage_transform(value, args.stage)
        if args.stage + 1 < args.stages:
            destination = packet_path(args.work_dir, args.stage, args.stage + 1, tile)
            publish(destination, {
                "launch_id": args.launch_id,
                "from_stage": args.stage,
                "to_stage": args.stage + 1,
                "tile": tile,
                "value": output,
            })
        else:
            publish(args.work_dir / "results" / f"tile_{tile:04d}.json", {
                "launch_id": args.launch_id,
                "final_stage": args.stage,
                "tile": tile,
                "value": output,
            })
        print(f"stage={args.stage} tile={tile} value={output}", flush=True)

    publish(args.work_dir / "status" / f"stage_{args.stage}.json", {
        "launch_id": args.launch_id,
        "stage": args.stage,
        "state": "complete",
        "tiles": args.tiles,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
