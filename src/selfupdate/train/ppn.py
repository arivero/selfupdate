"""General arbitrary-stage pipeline execution primitives.

This module is intentionally independent of the v3.2 schedule.  The schedule
owns the scientific meaning of a BxK tile; this file owns the *topological
order* in which stage callbacks consume that tile.  Keeping the two separate
is important: a serial PPn run and a wavefront PPn run must be able to use the
same tile callback and differ only in the order in which independent stage
operations are admitted.

The public pieces are small enough to use in CPU probes and in distributed
launchers:

``ModelAdapter``
    The model-facing contract: ordered blocks, ownership, legal cuts,
    detached boundary values, causal-state detachment, and frozen-vocabulary
    aliases.
``partition_cost_profile`` / ``choose_partition``
    Measured-cost records and a dynamic-programming contiguous partitioner.
``PPnExecutor``
    A serial reference and a depth-one-queue wavefront executor.  The latter
    implements ``O[s,t] -> O[s,t+1]`` and ``O[s,t] -> O[s+1,t]`` and never
    creates a downstream-to-upstream dependency.
``CheckpointCoordinator``
    Cursor and shard publication guard used before an atomic checkpoint
    rename.

No callback in this module performs a CUDA synchronization.  Synchronization
belongs in an outside-the-hot-loop measurement or in a stage callback at the
same boundary where the existing v3.2 runtime already flushes telemetry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Protocol

import torch


PROJECT_VERSION = "3.4"
PIPELINE_VERSION = 3
PIPELINE_REVISION = "3.2"


def _named_parameters_all(module):
    try:
        return module.named_parameters(remove_duplicate=False)
    except TypeError:
        return module.named_parameters()


class PPnError(RuntimeError):
    """Base error for an invalid PPn plan or failed stage execution."""


def _strict_splits(num_blocks: int, splits: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(x) for x in splits)
    if any(x <= 0 or x >= num_blocks for x in values):
        raise ValueError(
            f"pipeline boundaries must lie in 1..{num_blocks - 1}: {values}")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(
            f"pipeline boundaries must be strictly increasing: {values}")
    return values


@dataclass(frozen=True)
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


def partition_from_config(cfg, *, num_blocks: int,
                          physical_devices: Sequence[int] | None = None,
                          profile_id: str = "") -> PPnPartition:
    """Resolve the explicitly pinned model placement into a PPn partition.

    This function never chooses cuts.  Automatic measured-cost partitioning
    is a separate operation so production cannot silently repartition a run
    when a profile or visible-device count changes.
    """
    model = cfg.model
    splits = list(getattr(model, "pipeline_splits", []) or [])
    if not splits and int(getattr(model, "pipeline_split", 0) or 0):
        splits = [int(model.pipeline_split)]
    configured_devices = list(
        getattr(model, "pipeline_devices", []) or physical_devices or [])
    world_size = int(getattr(model, "pipeline_world_size", 0) or 0)
    if world_size and world_size != len(splits) + 1:
        raise ValueError(
            "model.pipeline_world_size must equal len(pipeline_splits)+1")
    if configured_devices and len(configured_devices) != len(splits) + 1:
        raise ValueError(
            "model.pipeline_devices must contain one physical id per stage")
    return PPnPartition(
        num_blocks=num_blocks,
        boundaries=tuple(splits),
        physical_devices=tuple(configured_devices),
        profile_id=(profile_id or getattr(cfg.train, "partition_profile_id", "")),
        model_identity=getattr(model, "name", ""),
    )


def boundary_volume_bytes(*, live_users: int, tile_width: int,
                          hidden_size: int, element_size: int) -> int:
    """Exact logical traffic for one live BxK boundary tile."""
    values = (live_users, tile_width, hidden_size, element_size)
    if any(int(value) < 0 for value in values):
        raise ValueError("boundary dimensions must be non-negative")
    return (int(live_users) * int(tile_width) * int(hidden_size)
            * int(element_size))


class ModelAdapterProtocol(Protocol):
    """Structural protocol implemented by a model adapter."""

    @property
    def ordered_text_blocks(self) -> Sequence[torch.nn.Module]: ...

    def legal_cut_positions(self) -> Sequence[int]: ...

    def detach_boundary(self, value: Any) -> Any: ...


def _detach(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach(item) for item in value)
    if isinstance(value, list):
        return [_detach(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class BlockMetadata:
    index: int
    layer_type: str
    parameter_count: int
    trainable_parameter_count: int
    hidden_size: int
    causal_state_private: bool


class ModelAdapter:
    """Architecture-neutral adapter around the repository's ``BlockStack``.

    The adapter deliberately uses duck typing so small CPU test models can
    implement the same contract without importing Transformers.  ``from_stack``
    supplies the concrete adapter used by production layerwise training.
    """

    def __init__(self, stack, *, model_identity: str = ""):
        self.stack = stack
        self.model_identity = model_identity
        self._blocks = tuple(stack.blocks)
        self._block_parameter_ids = {
            id(parameter): block
            for block, module in enumerate(self._blocks, start=1)
            for parameter in module.parameters()
        }
        self._owner_by_parameter_name: dict[str, str] = {}
        self._build_parameter_ownership()

    @classmethod
    def from_stack(cls, stack, *, model_identity: str = "") -> "ModelAdapter":
        return cls(stack, model_identity=model_identity)

    @property
    def ordered_text_blocks(self) -> tuple[torch.nn.Module, ...]:
        return self._blocks

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def layer_types(self) -> tuple[str, ...]:
        values = list(getattr(self.stack, "layer_types", []) or [])
        return tuple(values[i] if i < len(values) and values[i]
                     else "dense" for i in range(self.num_blocks))

    def _build_parameter_ownership(self) -> None:
        named = _named_parameters_all(self.stack.model)
        id_to_name: dict[int, list[str]] = {}
        for name, parameter in named:
            id_to_name.setdefault(id(parameter), []).append(name)
        for parameter_id, names in id_to_name.items():
            block = self._block_parameter_ids.get(parameter_id)
            if block is not None:
                owner = f"block:{block}"
            else:
                owner = "frozen_vocabulary_or_input"
            for name in names:
                self._owner_by_parameter_name[name] = owner

    def parameter_ownership(self) -> dict[str, str]:
        """Return complete parameter ownership, including tied aliases."""
        return dict(self._owner_by_parameter_name)

    def tied_weight_aliases(self) -> dict[str, list[str]]:
        aliases: dict[int, list[str]] = {}
        for name, parameter in _named_parameters_all(self.stack.model):
            aliases.setdefault(id(parameter), []).append(name)
        return {
            names[0]: names[1:]
            for names in aliases.values() if len(names) > 1
        }

    def legal_cut_positions(self) -> tuple[int, ...]:
        declared = getattr(self.stack, "legal_cut_positions", None)
        if callable(declared):
            cuts = declared()
        else:
            cuts = range(1, self.num_blocks)
        cuts = tuple(int(x) for x in cuts)
        _strict_splits(self.num_blocks, cuts)
        return cuts

    def block_metadata(self) -> tuple[BlockMetadata, ...]:
        hidden_size = int(getattr(self.stack.text_config, "hidden_size", 0))
        result = []
        shared = set(
            index + 1 for index, accepts in enumerate(
                getattr(self.stack, "_accepts_shared_kv_states", [])) if accepts
        )
        for index, block in enumerate(self._blocks, start=1):
            parameters = list(block.parameters())
            result.append(BlockMetadata(
                index=index,
                layer_type=self.layer_types[index - 1],
                parameter_count=sum(p.numel() for p in parameters),
                trainable_parameter_count=sum(
                    p.numel() for p in parameters if p.requires_grad),
                hidden_size=hidden_size,
                causal_state_private=index not in shared,
            ))
        return tuple(result)

    def causal_state(self, *, max_cache_len: int | None = None):
        """Construct a model-authoritative state object when available."""
        factory = getattr(self.stack, "new_causal_state", None)
        if callable(factory):
            return factory(max_cache_len=max_cache_len)
        try:
            from transformers import DynamicCache
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise NotImplementedError(
                "model adapter has no causal_state factory") from exc
        return DynamicCache(config=self.stack.text_config)

    def detach_causal_state(self, state) -> None:
        """Detach every tensor in a state without changing its addresses."""
        detach = getattr(self.stack, "detach_causal_state", None)
        if callable(detach):
            detach(state)
            return
        if hasattr(state, "layers"):
            for layer in state.layers:
                for name, value in vars(layer).items():
                    detached = _detach(value)
                    if detached is not value:
                        setattr(layer, name, detached)

    def detach_boundary(self, value: Any) -> Any:
        """Detach the cross-stage activation and its immutable metadata."""
        return _detach(value)

    def boundary_schema(self, value: Any) -> dict[str, Any]:
        tensors = []
        for tensor in _iter_tensors(value):
            tensors.append({
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "element_size": tensor.element_size(),
                "bytes": tensor.numel() * tensor.element_size(),
                "device": str(tensor.device),
                "requires_grad": bool(tensor.requires_grad),
            })
        return {"tensors": tensors, "tensor_count": len(tensors)}

    def immutable_inputs(self, positions, masks=None, rope=None) -> dict[str, Any]:
        return {
            "position_ids": _detach(positions),
            "masks": _detach(masks),
            "rope": _detach(rope),
        }

    def frozen_vocabulary_requirements(self) -> dict[str, Any]:
        modules = {
            "embedding": self.stack.embed_tokens,
            "final_norm": self.stack.final_norm,
            "lm_head": self.stack.lm_head,
        }
        return {
            "modules": list(modules),
            "all_parameters_frozen": all(
                not parameter.requires_grad
                for module in modules.values()
                for parameter in module.parameters()),
            "tied_weight_aliases": self.tied_weight_aliases(),
        }


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Tile:
    index: int
    payload: Any
    users: int = 0
    width: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("tile index must be non-negative")
        if self.users < 0 or self.width < 0:
            raise ValueError("tile users/width must be non-negative")


@dataclass
class StageResult:
    payload: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class StageContext:
    stage: int
    first_block: int
    last_block: int
    state: Any = None
    cursor: int = -1
    accepted_tiles: int = 0
    completed_tiles: int = 0
    compute_seconds: float = 0.0
    receive_wait_seconds: float = 0.0
    send_wait_seconds: float = 0.0
    metric_sums: dict[str, float] = field(default_factory=dict)
    _last_tile: int = -1

    def admit(self, tile_index: int) -> None:
        if tile_index != self._last_tile + 1:
            raise PPnError(
                f"stage {self.stage} admitted tile {tile_index} after "
                f"{self._last_tile}; ordered tiles are required")
        self._last_tile = tile_index
        self.cursor = tile_index
        self.accepted_tiles += 1


StageCallback = Callable[[StageContext, Tile, Any], Any]
BoundaryDetach = Callable[[Any], Any]
BoundaryTransfer = Callable[[Any, int], Any]


class PPnExecutor:
    """Run stage callbacks in serial, wavefront, or independent order."""

    def __init__(self, partition: PPnPartition,
                 callbacks: Sequence[StageCallback], *,
                 detach_boundary: BoundaryDetach | None = None,
                 transfer_boundary: BoundaryTransfer | None = None,
                 queue_depth: int = 1,
                 telemetry_callback: Callable[[int, Tile, StageResult], None]
                 | None = None):
        if len(callbacks) != partition.stages:
            raise ValueError(
                f"expected {partition.stages} stage callbacks, got {len(callbacks)}")
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        self.partition = partition
        self.callbacks = tuple(callbacks)
        self.detach_boundary = detach_boundary or _detach
        # A transport callback may perform an exact peer/NCCL or pinned-host
        # copy after detachment.  The default is a no-copy in-process packet,
        # which is useful for CPU probes and for a stage map whose CUDA
        # callback performs its own destination placement.
        self.transfer_boundary = transfer_boundary or (lambda value, _stage: value)
        self.queue_depth = queue_depth
        self.telemetry_callback = telemetry_callback
        self.contexts = [StageContext(stage, first, last)
                         for stage, (first, last)
                         in enumerate(partition.ranges)]

    def _normalize_result(self, result: Any) -> StageResult:
        if isinstance(result, StageResult):
            return result
        return StageResult(result)

    def _call(self, stage: int, tile: Tile, payload: Any, *,
              count_boundary: bool = True) -> StageResult:
        context = self.contexts[stage]
        context.admit(tile.index)
        started = time.perf_counter()
        result = self._normalize_result(
            self.callbacks[stage](context, tile, payload))
        context.compute_seconds += time.perf_counter() - started
        context.completed_tiles += 1
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                context.metric_sums[key] = (
                    context.metric_sums.get(key, 0.0) + float(value))
        if count_boundary and stage + 1 < self.partition.stages:
            boundary_bytes = tile.metadata.get("boundary_bytes", 0)
            if isinstance(boundary_bytes, (int, float)):
                context.metric_sums["boundary_bytes"] = (
                    context.metric_sums.get("boundary_bytes", 0.0)
                    + float(boundary_bytes))
        if self.telemetry_callback is not None:
            self.telemetry_callback(stage, tile, result)
        return result

    def run_serial(self, tiles: Iterable[Tile]) -> list[StageResult]:
        results: list[StageResult] = []
        for tile in tiles:
            payload = tile.payload
            for stage in range(self.partition.stages):
                result = self._call(stage, tile, payload)
                payload = (self.detach_boundary(result.payload)
                           if stage + 1 < self.partition.stages
                           else result.payload)
                if stage + 1 < self.partition.stages:
                    payload = self.transfer_boundary(payload, stage)
            results.append(StageResult(payload, result.metrics))
        return results

    def run_wavefront(self, tiles: Iterable[Tile], *,
                      stop_admission: threading.Event | None = None
                      ) -> list[StageResult]:
        """Execute with one ordered worker and a bounded queue per stage."""
        stages = self.partition.stages
        queues = [Queue(maxsize=self.queue_depth) for _ in range(stages)]
        sentinel = object()
        failure: list[BaseException] = []
        failure_lock = threading.Lock()
        results: list[tuple[int, StageResult]] = []
        result_lock = threading.Lock()
        cancelled = threading.Event()

        def fail(exc: BaseException) -> None:
            with failure_lock:
                if not failure:
                    failure.append(exc)
            cancelled.set()

        def put(queue: Queue, item: Any, context: StageContext | None = None):
            started = time.perf_counter()
            while not cancelled.is_set():
                try:
                    queue.put(item, timeout=0.05)
                    if context is not None:
                        context.send_wait_seconds += time.perf_counter() - started
                    return True
                except Full:
                    continue
            return False

        def worker(stage: int) -> None:
            context = self.contexts[stage]
            queue = queues[stage]
            while not cancelled.is_set():
                started = time.perf_counter()
                try:
                    item = queue.get(timeout=0.05)
                except Empty:
                    context.receive_wait_seconds += time.perf_counter() - started
                    continue
                context.receive_wait_seconds += time.perf_counter() - started
                if item is sentinel:
                    if stage + 1 < stages:
                        put(queues[stage + 1], sentinel, context)
                    return
                tile, payload = item
                try:
                    result = self._call(stage, tile, payload)
                    if stage + 1 < stages:
                        next_payload = self.detach_boundary(result.payload)
                        next_payload = self.transfer_boundary(next_payload, stage)
                        if not put(queues[stage + 1], (tile, next_payload), context):
                            return
                    else:
                        with result_lock:
                            results.append((tile.index, result))
                except BaseException as exc:  # propagate the original failure
                    fail(exc)
                    return

        threads = [threading.Thread(target=worker, args=(stage,),
                                     name=f"selfupdate-ppn-stage-{stage}")
                   for stage in range(stages)]
        for thread in threads:
            thread.start()
        try:
            for tile in tiles:
                if cancelled.is_set() or (stop_admission is not None
                                          and stop_admission.is_set()):
                    break
                if not put(queues[0], (tile, tile.payload), self.contexts[0]):
                    break
            put(queues[0], sentinel, self.contexts[0])
        except BaseException as exc:
            fail(exc)
        for thread in threads:
            thread.join()
        if failure:
            raise PPnError("PPn stage failed") from failure[0]
        results.sort(key=lambda pair: pair[0])
        return [result for _, result in results]

    def run_independent(self, tiles: Iterable[Tile], *,
                        stop_admission: threading.Event | None = None
                        ) -> list[StageResult]:
        """Execute stage-local tile chains without cross-stage edges.

        This order is legal only when every block obtains its input from an
        immutable teacher cache.  The caller enforces that scientific knob;
        the executor enforces ordered tiles and bounded admission separately
        for every unchanged stage in the pinned partition.
        """
        stages = self.partition.stages
        queues = [Queue(maxsize=self.queue_depth) for _ in range(stages)]
        sentinel = object()
        failure: list[BaseException] = []
        failure_lock = threading.Lock()
        final_results: list[tuple[int, StageResult]] = []
        result_lock = threading.Lock()
        cancelled = threading.Event()

        def fail(exc: BaseException) -> None:
            with failure_lock:
                if not failure:
                    failure.append(exc)
            cancelled.set()

        def put(queue: Queue, item: Any, context: StageContext | None = None):
            started = time.perf_counter()
            while not cancelled.is_set():
                try:
                    queue.put(item, timeout=0.05)
                    if context is not None:
                        context.send_wait_seconds += time.perf_counter() - started
                    return True
                except Full:
                    continue
            return False

        def worker(stage: int) -> None:
            context = self.contexts[stage]
            while not cancelled.is_set():
                started = time.perf_counter()
                try:
                    item = queues[stage].get(timeout=0.05)
                except Empty:
                    context.receive_wait_seconds += time.perf_counter() - started
                    continue
                context.receive_wait_seconds += time.perf_counter() - started
                if item is sentinel:
                    return
                tile, payload = item
                try:
                    result = self._call(
                        stage, tile, payload, count_boundary=False)
                    if stage == stages - 1:
                        with result_lock:
                            final_results.append((tile.index, result))
                except BaseException as exc:
                    fail(exc)
                    return

        threads = [threading.Thread(
            target=worker, args=(stage,),
            name=f"selfupdate-ppn-independent-stage-{stage}")
            for stage in range(stages)]
        for thread in threads:
            thread.start()
        try:
            for tile in tiles:
                if cancelled.is_set() or (stop_admission is not None
                                          and stop_admission.is_set()):
                    break
                for stage in range(stages):
                    if not put(queues[stage], (tile, tile.payload),
                               self.contexts[stage]):
                        break
                if cancelled.is_set():
                    break
            for stage in range(stages):
                put(queues[stage], sentinel, self.contexts[stage])
        except BaseException as exc:
            fail(exc)
        for thread in threads:
            thread.join()
        if failure:
            raise PPnError("independent PPn stage failed") from failure[0]
        completed = {context.completed_tiles for context in self.contexts}
        if len(completed) != 1:
            raise PPnError(
                f"independent stages drained unequal tile counts: {completed}")
        final_results.sort(key=lambda pair: pair[0])
        return [result for _, result in final_results]

    def run(self, tiles: Iterable[Tile], *, execution: str = "wavefront",
            stop_admission: threading.Event | None = None) -> list[StageResult]:
        if execution == "serial":
            return self.run_serial(tiles)
        if execution == "wavefront":
            return self.run_wavefront(tiles, stop_admission=stop_admission)
        if execution == "independent":
            return self.run_independent(tiles, stop_admission=stop_admission)
        raise ValueError(
            "PPn execution must be 'serial', 'wavefront', or 'independent'")

    def telemetry(self) -> dict[str, Any]:
        return {
            "stage_count": self.partition.stages,
            "ordered_block_ranges": [list(item) for item in self.partition.ranges],
            "physical_gpu_mapping": list(self.partition.physical_devices),
            "per_stage": [{
                "stage": context.stage,
                "block_range": [context.first_block, context.last_block],
                "accepted_tiles": context.accepted_tiles,
                "completed_tiles": context.completed_tiles,
                "compute_seconds": context.compute_seconds,
                "receive_wait_seconds": context.receive_wait_seconds,
                "send_wait_seconds": context.send_wait_seconds,
                "metric_sums": dict(context.metric_sums),
            } for context in self.contexts],
        }


@dataclass(frozen=True)
class TransportResult:
    transport: str
    supported: bool
    elapsed_seconds: float
    bytes_copied: int
    active_bandwidth_gib_s: float
    exact: bool
    reason: str = ""


class BoundaryTransport:
    """Exact-copy transport helper for peer/NCCL or pinned-host probes."""

    def __init__(self, mode: str = "peer"):
        if mode not in ("peer", "nccl", "pinned_host"):
            raise ValueError("transport must be peer, nccl, or pinned_host")
        self.mode = mode

    def copy(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise ValueError("boundary transport requires identical shape/dtype")
        if self.mode == "nccl":
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError(
                    "NCCL point-to-point requires a two-rank initialized "
                    "torch.distributed process group")
            raise RuntimeError(
                "NCCL point-to-point uses rank-owned send/recv buffers; use "
                "benchmark_nccl_p2p under torchrun")
        if self.mode == "peer":
            destination.copy_(source, non_blocking=True)
            return
        host = torch.empty_like(source, device="cpu", pin_memory=True)
        host.copy_(source, non_blocking=False)
        destination.copy_(host, non_blocking=False)


def benchmark_boundary_transport(*, shape: Sequence[int], dtype=torch.bfloat16,
                                 source_device: str = "cuda:0",
                                 destination_device: str = "cuda:1",
                                 repeats: int = 20,
                                 transport: str = "peer") -> TransportResult:
    """Benchmark one exact boundary copy outside production execution."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not torch.cuda.is_available():
        return TransportResult(transport, False, 0.0, 0, 0.0, True,
                               "CUDA is unavailable")
    try:
        source = torch.randn(tuple(shape), device=source_device, dtype=dtype)
        destination = torch.empty_like(source, device=destination_device)
        mover = BoundaryTransport(transport)
        for _ in range(3):
            mover.copy(source, destination)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            mover.copy(source, destination)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        bytes_copied = source.numel() * source.element_size() * repeats
        return TransportResult(
            transport, True, elapsed, bytes_copied,
            bytes_copied / max(elapsed, 1e-12) / 2**30, True)
    except (RuntimeError, ValueError) as exc:
        return TransportResult(transport, False, 0.0, 0, 0.0, True, str(exc))


def benchmark_nccl_p2p(*, shape: Sequence[int], dtype=torch.bfloat16,
                       source_rank: int = 0, destination_rank: int = 1,
                       repeats: int = 20) -> TransportResult:
    """Benchmark rank-owned NCCL send/recv buffers under ``torchrun``.

    This deliberately has a separate entry point from the same-process peer
    probe: NCCL point-to-point is a rank operation, not a second spelling of
    ``destination.copy_(source)``.
    """
    if not (torch.cuda.is_available() and torch.distributed.is_available()
            and torch.distributed.is_initialized()):
        return TransportResult("nccl", False, 0.0, 0, 0.0, True,
                               "requires CUDA and an initialized process group")
    world = torch.distributed.get_world_size()
    if world < 2 or source_rank == destination_rank:
        return TransportResult("nccl", False, 0.0, 0, 0.0, True,
                               "requires two distinct ranks")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    source = (torch.randn(tuple(shape), device=device, dtype=dtype)
              if rank == source_rank else torch.empty(
                  tuple(shape), device=device, dtype=dtype))
    destination = (torch.empty_like(source)
                   if rank == source_rank else source)
    for _ in range(3):
        if rank == source_rank:
            torch.distributed.send(source, destination_rank)
        elif rank == destination_rank:
            torch.distributed.recv(destination, source_rank)
    torch.cuda.synchronize(device)
    torch.distributed.barrier()
    started = time.perf_counter()
    for _ in range(repeats):
        if rank == source_rank:
            torch.distributed.send(source, destination_rank)
        elif rank == destination_rank:
            torch.distributed.recv(destination, source_rank)
    torch.cuda.synchronize(device)
    torch.distributed.barrier()
    elapsed = time.perf_counter() - started
    bytes_copied = source.numel() * source.element_size() * repeats
    elapsed_tensor = torch.tensor(elapsed, device=device)
    torch.distributed.all_reduce(
        elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
    elapsed = float(elapsed_tensor.cpu())
    return TransportResult(
        "nccl", True, elapsed, bytes_copied,
        bytes_copied / max(elapsed, 1e-12) / 2**30, True)


@dataclass
class CheckpointCoordinator:
    """Guard atomic publication of PPn stage shards and metadata."""

    root: Path
    partition: PPnPartition
    cursors: dict[int, int] = field(default_factory=dict)
    shard_paths: dict[int, str] = field(default_factory=dict)
    require_shard_files: bool = True

    def record_stage(self, stage: int, *, cursor: int, shard_path: str) -> None:
        if stage not in range(self.partition.stages):
            raise PPnError(f"unknown checkpoint stage {stage}")
        self.cursors[stage] = int(cursor)
        self.shard_paths[stage] = str(shard_path)

    def manifest(self) -> dict[str, Any]:
        return {
            **self.partition.manifest(),
            "checkpoint_format": "ppn_stage_shards",
            "stage_cursors": {str(k): v for k, v in sorted(self.cursors.items())},
            "stage_shards": {str(k): v for k, v in sorted(self.shard_paths.items())},
        }

    def publish(self, *, expected_cursor: int, destination: str = "checkpoint") -> Path:
        expected_stages = set(range(self.partition.stages))
        if set(self.cursors) != expected_stages or set(self.shard_paths) != expected_stages:
            raise PPnError("refusing PPn checkpoint: one or more stage shards missing")
        if set(self.cursors.values()) != {int(expected_cursor)}:
            raise PPnError(
                "refusing PPn checkpoint: stage cursors do not agree")
        if self.require_shard_files:
            missing = []
            for path in self.shard_paths.values():
                candidate = Path(path)
                if not candidate.is_absolute():
                    candidate = self.root / candidate
                if not candidate.is_file():
                    missing.append(str(candidate))
            if missing:
                raise PPnError(
                    "refusing PPn checkpoint: incomplete stage shard(s): "
                    + ", ".join(missing))
        target = self.root / destination
        if target.exists():
            raise FileExistsError(target)
        incomplete = self.root / f".{destination}.incomplete-{time.time_ns()}"
        incomplete.mkdir(parents=True)
        try:
            (incomplete / "partition_manifest.json").write_text(
                json.dumps(self.manifest(), indent=2) + "\n")
            incomplete.rename(target)
        except BaseException:
            import shutil
            shutil.rmtree(incomplete, ignore_errors=True)
            raise
        return target
