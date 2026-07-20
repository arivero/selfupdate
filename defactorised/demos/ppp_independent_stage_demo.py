#!/usr/bin/env python3
"""One synthetic pipeline-v4 PPP worker: train only this stage's block shard.

Unlike the PPn wavefront demo, PPP stage workers do not pass training
activations to one another.  Each owns disjoint blocks, consumes the same
teacher-sourced epoch inputs, and publishes an independently mergeable shard.
The production relay is used for store-fill/evaluation, not as a training
activation pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--stages", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fail-stage", type=int, default=-1)
    return parser.parse_args()


def publish(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def teacher_target(epoch: int, block: int) -> float:
    """Deterministic stand-in for a cached teacher state for one block."""
    return block * 0.25 + epoch * 0.05


def main() -> int:
    args = parse_args()
    if args.stages < 1 or not 0 <= args.stage < args.stages:
        raise SystemExit("require 0 <= --stage < --stages and at least one stage")
    if args.blocks_per_stage < 1 or args.epochs < 1:
        raise SystemExit("--blocks-per-stage and --epochs must be positive")
    if args.stage == args.fail_stage:
        raise RuntimeError(f"intentional demo failure in stage {args.stage}")

    first = args.stage * args.blocks_per_stage + 1
    owned_blocks = list(range(first, first + args.blocks_per_stage))
    weights = {block: 0.0 for block in owned_blocks}

    # Every update is block-local: no value is read from another stage.
    for epoch in range(args.epochs):
        for block in owned_blocks:
            error = teacher_target(epoch, block) - weights[block]
            weights[block] += 0.5 * error
        print(f"stage={args.stage} epoch={epoch} weights={weights}", flush=True)

    base_identity = hashlib.sha256(b"shared synthetic base").hexdigest()[:16]
    publish(args.work_dir / "shards" / f"stage_{args.stage}.json", {
        "architecture": "pipeline-v4-ppp-independent-stage",
        "launch_id": args.launch_id,
        "base_identity": base_identity,
        "stage": args.stage,
        "stage_count": args.stages,
        "owned_blocks": owned_blocks,
        "weights": weights,
        "training_activation_handoffs": 0,
        "mergeable": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
