#!/usr/bin/env python3
"""Readable, single-file PPn cost partitioning demo.

This is the focused architecture walk-through: measured per-block cost and
memory records feed a contiguous dynamic program, which emits explicit stage
boundaries.  It is generated from the production definitions, with no bundle
loader and no import from ``src/selfupdate``.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_VERSION = "3.4"
PIPELINE_VERSION = 3
PIPELINE_REVISION = "3.2"


class PPnError(RuntimeError):
    pass

def _strict_splits(num_blocks: int, splits: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in splits)
    if any(x <= 0 or x >= num_blocks for x in values):
        raise ValueError(
            f"pipeline boundaries must lie in 1..{num_blocks - 1}: {values}")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(
            f"pipeline boundaries must be strictly increasing: {values}")
    return values

class PPnPartition:
    """A pinned contiguous partition using one-based inclusive block ranges."""

    num_blocks: int
    boundaries: tuple[int, ...] = ()
    physical_devices: tuple[int, ...] = ()
    profile_id: str = ""
    model_identity: str = ""

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("PPnPartition requires at least one block")
        splits = _strict_splits(self.num_blocks, self.boundaries)
        object.__setattr__(self, "boundaries", splits)
        expected = len(splits) + 1
        devices = tuple(int(x) for x in self.physical_devices)
        if devices and len(devices) != expected:
            raise ValueError(
                f"physical_devices has {len(devices)} entries for {expected} stages")
        if len(set(devices)) != len(devices):
            raise ValueError(f"physical_devices must be unique: {devices}")
        object.__setattr__(self, "physical_devices", devices)

    @property
    def stages(self) -> int:
        return len(self.boundaries) + 1

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        cuts = (0, *self.boundaries, self.num_blocks)
        return tuple((cuts[i] + 1, cuts[i + 1])
                     for i in range(len(cuts) - 1))

    @property
    def block_to_stage(self) -> dict[int, int]:
        out = {}
        for stage, (first, last) in enumerate(self.ranges):
            out.update({block: stage for block in range(first, last + 1)})
        return out

    def owner(self, block: int) -> int:
        if not 1 <= block <= self.num_blocks:
            raise IndexError(block)
        return self.block_to_stage[block]

    def manifest(self) -> dict[str, Any]:
        return {
            "layerwise_project_version": PROJECT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_revision": PIPELINE_REVISION,
            "model_identity": self.model_identity,
            "partition_profile_id": self.profile_id,
            "num_blocks": self.num_blocks,
            "boundaries": list(self.boundaries),
            "ordered_block_ranges": [list(pair) for pair in self.ranges],
            "physical_devices": list(self.physical_devices),
            "stage_count": self.stages,
        }

class BlockCost:
    """Measured block cost for one representative BxK distribution."""

    block: int
    prompt_prefill_us: float = 0.0
    tile_forward_us: float = 0.0
    local_loss_us: float = 0.0
    backward_us: float = 0.0
    immediate_write_us: float = 0.0
    p95_prompt_prefill_us: float | None = None
    p95_tile_forward_us: float | None = None
    p95_local_loss_us: float | None = None
    p95_backward_us: float | None = None
    p95_immediate_write_us: float | None = None
    layer_type: str = "dense"
    causal_state_bytes: int = 0
    backward_workspace_bytes: int = 0
    trainable_parameter_bytes: int = 0
    gradient_bytes: int = 0
    frozen_vocab_workspace_bytes: int = 0

    def time_us(self, *, percentile: str = "p50") -> float:
        fields = (
            "prompt_prefill_us", "tile_forward_us", "local_loss_us",
            "backward_us", "immediate_write_us")
        total = 0.0
        for field_name in fields:
            value = getattr(self, field_name)
            if percentile == "p95":
                high = getattr(self, f"p95_{field_name}")
                value = value if high is None else high
            total += float(value)
        return total

    def memory_bytes(self) -> int:
        return sum((self.causal_state_bytes, self.backward_workspace_bytes,
                    self.trainable_parameter_bytes, self.gradient_bytes,
                    self.frozen_vocab_workspace_bytes))

    def as_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

class CostProfile:
    model_identity: str
    blocks: tuple[BlockCost, ...]
    profile_id: str = ""
    boundary_bytes: int = 0
    boundary_us_p50: float = 0.0
    boundary_us_p95: float | None = None
    stage0_embedding_us: float = 0.0
    final_output_eval_us: float = 0.0
    stage0_extra_memory_bytes: int = 0
    final_stage_extra_memory_bytes: int = 0
    measured_batch: int = 256
    measured_tile_width: int = 16
    source: str = ""

    def __post_init__(self) -> None:
        ordered = tuple(self.blocks)
        if tuple(item.block for item in ordered) != tuple(
                range(1, len(ordered) + 1)):
            raise ValueError("cost profile blocks must be ordered and 1-based")
        object.__setattr__(self, "blocks", ordered)

    @property
    def identity(self) -> str:
        if self.profile_id:
            return self.profile_id
        payload = json.dumps(self.as_dict(include_identity=False),
                             sort_keys=True, separators=(",", ":"))
        return "profile-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        data = {
            "model_identity": self.model_identity,
            "blocks": [item.as_dict() for item in self.blocks],
            "boundary_bytes": self.boundary_bytes,
            "boundary_us_p50": self.boundary_us_p50,
            "boundary_us_p95": self.boundary_us_p95,
            "stage0_embedding_us": self.stage0_embedding_us,
            "final_output_eval_us": self.final_output_eval_us,
            "stage0_extra_memory_bytes": self.stage0_extra_memory_bytes,
            "final_stage_extra_memory_bytes": self.final_stage_extra_memory_bytes,
            "measured_batch": self.measured_batch,
            "measured_tile_width": self.measured_tile_width,
            "source": self.source,
        }
        if include_identity:
            data["profile_id"] = self.identity
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostProfile":
        blocks = tuple(BlockCost(**dict(item)) for item in data.get("blocks", ()))
        return cls(
            model_identity=str(data.get("model_identity", "")),
            blocks=blocks,
            profile_id=str(data.get("profile_id", "")),
            boundary_bytes=int(data.get("boundary_bytes", 0)),
            boundary_us_p50=float(data.get("boundary_us_p50", 0.0)),
            boundary_us_p95=(None if data.get("boundary_us_p95") is None
                             else float(data["boundary_us_p95"])),
            stage0_embedding_us=float(data.get("stage0_embedding_us", 0.0)),
            final_output_eval_us=float(data.get("final_output_eval_us", 0.0)),
            stage0_extra_memory_bytes=int(data.get("stage0_extra_memory_bytes", 0)),
            final_stage_extra_memory_bytes=int(
                data.get("final_stage_extra_memory_bytes", 0)),
            measured_batch=int(data.get("measured_batch", 256)),
            measured_tile_width=int(data.get("measured_tile_width", 16)),
            source=str(data.get("source", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CostProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2) + "\n")

class PartitionConstraints:
    capacities_bytes: tuple[int, ...] = ()
    safety_margin: float = 0.80
    # None means every block boundary is legal; an empty tuple means the
    # adapter declares no legal inter-stage cut.
    legal_cuts: tuple[int, ...] | None = None
    percentile: str = "p95"
    boundary_memory_bytes: int | None = None

def _segment_memory(profile: CostProfile, first: int, last: int,
                    stage: int, constraints: PartitionConstraints) -> int:
    total = sum(profile.blocks[index - 1].memory_bytes()
                for index in range(first, last + 1))
    if stage == 0:
        total += profile.stage0_extra_memory_bytes
    if stage == 0 or stage == len(constraints.capacities_bytes) - 1:
        total += 0
    if constraints.boundary_memory_bytes:
        total += constraints.boundary_memory_bytes
    return total

def choose_partition(profile: CostProfile, stages: int, *,
                     constraints: PartitionConstraints | None = None,
                     physical_devices: Sequence[int] = ()) -> PPnPartition:
    """Choose balanced nonempty contiguous ranges by dynamic programming.

    The objective is the slowest predicted stage at the requested percentile.
    Segment memory is a hard constraint after the configured safety margin.
    Legal cuts come from the adapter; the model is never split in the middle
    of a block or across a declared shared-state boundary.
    """
    if stages <= 0:
        raise ValueError("stages must be positive")
    n = len(profile.blocks)
    constraints = constraints or PartitionConstraints()
    legal = set(range(1, n) if constraints.legal_cuts is None
                else constraints.legal_cuts)
    if any(c <= 0 or c >= n for c in legal):
        raise ValueError("legal cuts must be inside the block stack")
    nodes = (0, *sorted(legal), n)
    if stages > len(nodes) - 1:
        raise ValueError(
            f"cannot make {stages} nonempty stages with legal cuts {sorted(legal)}")
    prefix = [0.0]
    for item in profile.blocks:
        prefix.append(prefix[-1] + item.time_us(percentile=constraints.percentile))

    def work(first: int, last: int, stage: int) -> float:
        value = prefix[last] - prefix[first - 1]
        if stage == 0:
            value += profile.stage0_embedding_us
        if stage == stages - 1:
            value += profile.final_output_eval_us
        if stage:
            value += (profile.boundary_us_p95 if constraints.percentile == "p95"
                      and profile.boundary_us_p95 is not None
                      else profile.boundary_us_p50)
        return value

    def fits(first: int, last: int, stage: int) -> bool:
        if not constraints.capacities_bytes:
            return True
        if stage >= len(constraints.capacities_bytes):
            return False
        capacity = constraints.capacities_bytes[stage]
        if capacity <= 0:
            return True
        memory = _segment_memory(profile, first, last, stage, constraints)
        if stage == stages - 1:
            memory += profile.final_stage_extra_memory_bytes
        return memory <= capacity * constraints.safety_margin

    inf = float("inf")
    dp = [[inf] * len(nodes) for _ in range(stages + 1)]
    parent: list[list[int | None]] = [
        [None] * len(nodes) for _ in range(stages + 1)]
    dp[0][0] = 0.0
    for stage in range(1, stages + 1):
        for endpoint in range(stage, len(nodes)):
            end = nodes[endpoint]
            for start_index in range(stage - 1, endpoint):
                start = nodes[start_index]
                if start == 0 or start in legal or end == n:
                    first = start + 1
                    if first > end or not fits(first, end, stage - 1):
                        continue
                    previous = dp[stage - 1][start_index]
                    if previous == inf:
                        continue
                    candidate = max(previous, work(first, end, stage - 1))
                    if candidate < dp[stage][endpoint]:
                        dp[stage][endpoint] = candidate
                        parent[stage][endpoint] = start_index
    if dp[stages][-1] == inf:
        raise MemoryError(
            "no legal PPn partition satisfies the measured VRAM constraints")
    cuts: list[int] = []
    endpoint = len(nodes) - 1
    for stage in range(stages, 0, -1):
        start_index = parent[stage][endpoint]
        if start_index is None:
            raise PPnError("partition DP lost a predecessor")
        if start_index > 0:
            cuts.append(nodes[start_index])
        endpoint = start_index
    cuts.reverse()
    return PPnPartition(
        num_blocks=n,
        boundaries=tuple(cuts),
        physical_devices=tuple(int(x) for x in physical_devices),
        profile_id=profile.identity,
        model_identity=profile.model_identity,
    )

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
