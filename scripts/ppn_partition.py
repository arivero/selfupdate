#!/usr/bin/env python3
"""Choose and persist a measured-cost PPn partition.

This is a preparation tool, not an implicit production repartitioner.  Pin
the emitted ``boundaries`` and ``profile_id`` in the experiment config before
launching a campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.train.ppn import CostProfile, PartitionConstraints, choose_partition


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--stages", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--legal-cuts", default="",
                    help="comma-separated one-based cut positions")
    ap.add_argument("--capacity-gib", default="",
                    help="comma-separated per-stage VRAM capacities")
    ap.add_argument("--safety-margin", type=float, default=0.80)
    ap.add_argument("--percentile", choices=("p50", "p95"), default="p95")
    ap.add_argument("--devices", default="",
                    help="comma-separated physical GPU ids")
    args = ap.parse_args()
    profile = CostProfile.load(args.profile)
    legal = (tuple(int(x) for x in args.legal_cuts.split(",") if x)
             if args.legal_cuts else None)
    capacities = tuple(
        int(float(x) * 2**30) for x in args.capacity_gib.split(",") if x)
    devices = tuple(int(x) for x in args.devices.split(",") if x)
    partition = choose_partition(
        profile, args.stages,
        constraints=PartitionConstraints(
            legal_cuts=legal, capacities_bytes=capacities,
            safety_margin=args.safety_margin, percentile=args.percentile),
        physical_devices=devices)
    payload = {
        **partition.manifest(),
        "predicted_slowest_stage_us": _predicted_slowest(profile, partition,
                                                          args.percentile),
        "profile": profile.as_dict(),
        "pinning_instruction": (
            "Copy boundaries and partition_profile_id into the experiment "
            "config; production must not auto-repartition."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def _predicted_slowest(profile, partition, percentile):
    values = []
    for stage, (first, last) in enumerate(partition.ranges):
        value = sum(profile.blocks[index - 1].time_us(percentile=percentile)
                    for index in range(first, last + 1))
        if stage == 0:
            value += profile.stage0_embedding_us
        if stage == partition.stages - 1:
            value += profile.final_output_eval_us
        if stage:
            value += (profile.boundary_us_p95
                      if percentile == "p95" and profile.boundary_us_p95 is not None
                      else profile.boundary_us_p50)
        values.append(value)
    return max(values, default=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
