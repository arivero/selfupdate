#!/usr/bin/env python3
"""Turn a measured block-cost profile into one readable four-stage PPn plan.

This companion to ``ppn_demo.sh`` demonstrates why PPn uses measured,
contiguous partitions instead of equal layer counts.  It is deliberately
narrow: p95 time, four stages by default, and no production configuration
loading.  The profile is the JSON emitted by the PPn profiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TIME_FIELDS = (
    "prompt_prefill_us",
    "tile_forward_us",
    "local_loss_us",
    "backward_us",
    "immediate_write_us",
)


def p95_block_time(block: dict) -> float:
    total = 0.0
    for name in TIME_FIELDS:
        value = block.get(f"p95_{name}")
        if value is None:
            value = block.get(name, 0.0)
        total += float(value or 0.0)
    return total


def profile_identity(profile: dict) -> str:
    if profile.get("profile_id"):
        return str(profile["profile_id"])
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return "profile-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def segment_time(profile: dict, prefix: list[float], first: int, last: int,
                 stage: int, stages: int) -> float:
    value = prefix[last] - prefix[first - 1]
    if stage == 0:
        value += float(profile.get("stage0_embedding_us", 0.0))
    if stage == stages - 1:
        value += float(profile.get("final_output_eval_us", 0.0))
    if stage:
        boundary = profile.get("boundary_us_p95")
        value += float(profile.get("boundary_us_p50", 0.0)
                       if boundary is None else boundary)
    return value


def balanced_boundaries(profile: dict, stages: int) -> tuple[list[int], float]:
    """Minimize the slowest contiguous stage using dynamic programming."""
    blocks = profile["blocks"]
    count = len(blocks)
    if stages < 1 or stages > count:
        raise ValueError(f"stages must be between 1 and {count}")
    if [row.get("block") for row in blocks] != list(range(1, count + 1)):
        raise ValueError("profile blocks must be ordered and one-based")

    prefix = [0.0]
    for block in blocks:
        prefix.append(prefix[-1] + p95_block_time(block))

    infinity = float("inf")
    best = [[infinity] * (count + 1) for _ in range(stages + 1)]
    parent: list[list[int | None]] = [[None] * (count + 1)
                                      for _ in range(stages + 1)]
    best[0][0] = 0.0
    for used in range(1, stages + 1):
        stage = used - 1
        for end in range(used, count + 1):
            for cut in range(used - 1, end):
                candidate = max(
                    best[used - 1][cut],
                    segment_time(profile, prefix, cut + 1, end, stage, stages),
                )
                if candidate < best[used][end]:
                    best[used][end] = candidate
                    parent[used][end] = cut

    cuts: list[int] = []
    end = count
    for used in range(stages, 0, -1):
        cut = parent[used][end]
        if cut is None:
            raise RuntimeError("partition has no predecessor")
        if cut:
            cuts.append(cut)
        end = cut
    return list(reversed(cuts)), best[stages][count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="measured PPn profile JSON")
    parser.add_argument("out", type=Path, nargs="?",
                        default=Path("runs/ppn_demo_partition.json"))
    parser.add_argument("--stages", type=int, default=4)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    boundaries, slowest = balanced_boundaries(profile, args.stages)
    count = len(profile["blocks"])
    endpoints = [0, *boundaries, count]
    ranges = [[endpoints[i] + 1, endpoints[i + 1]]
              for i in range(args.stages)]
    plan = {
        "model_identity": profile.get("model_identity", ""),
        "partition_profile_id": profile_identity(profile),
        "percentile": "p95",
        "stage_count": args.stages,
        "boundaries": boundaries,
        "ordered_block_ranges": ranges,
        "physical_devices": list(range(args.stages)),
        "predicted_slowest_stage_us": slowest,
        "note": "Pin these boundaries; a production run must not repartition itself.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
