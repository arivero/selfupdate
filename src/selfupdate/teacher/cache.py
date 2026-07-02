"""Frozen-teacher cache: per-layer hidden states + top-k logits on disk.

The teacher is the initial checkpoint and its inputs are fixed per dataset
example, so everything the student ever needs from it is precomputed once into
sharded safetensors. The teacher never occupies GPU memory during training.

Layer-index convention (fixed here, verified by tests/test_cache_roundtrip.py):
``h{L}`` for L = 1..n_layers is ``output_hidden_states[L]`` of the HF forward —
the raw output of decoder block L, **except** ``h{n_layers}``, which HF returns
after the final RMSNorm. Losses against the last layer must therefore apply the
final norm on the student side too.

Per example, restricted to the aligned span (length A):
- ``h{L}``   [A, H] float16
- ``topk_v`` [A, k] float16   (top-k teacher logit values)
- ``topk_i`` [A, k] int32
- ``logz``   [A]    float32   (logsumexp over the full vocab row)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_NAME = "index.json"


def cache_config_hash(model_name: str, mask_mode: str, extra: dict | None = None) -> str:
    payload = {"model": model_name, "mode": mask_mode, "span": "mid+answer/v1"}
    payload.update(extra or {})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resolve_cache_dir(cfg) -> tuple[Path, str]:
    """Canonical cache directory + expected hash for an ExperimentConfig.
    Must mirror scripts/build_teacher_cache.py exactly."""
    examples_sha = hashlib.sha256(
        Path(cfg.data.examples_path).read_bytes()
    ).hexdigest()[:16]
    chash = cache_config_hash(
        cfg.model.name, cfg.mask.mode,
        {"compaction": cfg.mask.compaction, "examples": examples_sha},
    )
    model_short = cfg.model.name.split("/")[-1]
    root = Path(cfg.cache.root) / f"{model_short}-{cfg.mask.mode}-{cfg.mask.compaction}-{chash}"
    return root, chash


class TeacherCacheWriter:
    def __init__(self, root: str | Path, config_hash: str, shard_size: int = 128):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.shard_size = shard_size
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffered_examples = 0
        self._shard_no = 0
        self._index: dict = {"config_hash": config_hash, "examples": {}}

    def add(
        self,
        example_id: str,
        hidden: dict[int, torch.Tensor],  # L -> [A, H]
        topk_v: torch.Tensor,
        topk_i: torch.Tensor,
        logz: torch.Tensor,
        span: dict,
    ) -> None:
        for L, h in hidden.items():
            self._buffer[f"{example_id}/h{L:02d}"] = h.to(torch.float16).contiguous().cpu()
        self._buffer[f"{example_id}/topk_v"] = topk_v.to(torch.float16).contiguous().cpu()
        self._buffer[f"{example_id}/topk_i"] = topk_i.to(torch.int32).contiguous().cpu()
        self._buffer[f"{example_id}/logz"] = logz.to(torch.float32).contiguous().cpu()
        self._index["examples"][example_id] = {"shard": self._shard_no, **span}
        self._buffered_examples += 1
        if self._buffered_examples >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        save_file(self._buffer, str(self.root / f"shard-{self._shard_no:05d}.safetensors"))
        self._buffer = {}
        self._buffered_examples = 0
        self._shard_no += 1

    def finalize(self) -> None:
        self._flush()
        (self.root / INDEX_NAME).write_text(json.dumps(self._index))


class TeacherCache:
    """Lazy reader: per-tensor loads, so layerwise runs read only the layers
    they need (~1 MB/example instead of ~15 MB)."""

    def __init__(self, root: str | Path, expect_hash: str | None = None):
        self.root = Path(root)
        self._index = json.loads((self.root / INDEX_NAME).read_text())
        if expect_hash and self._index["config_hash"] != expect_hash:
            raise ValueError(
                f"stale teacher cache at {root}: hash {self._index['config_hash']} "
                f"!= expected {expect_hash}; rebuild with scripts/build_teacher_cache.py"
            )
        self._handles: dict[int, object] = {}

    @property
    def example_ids(self) -> list[str]:
        return list(self._index["examples"].keys())

    def span(self, example_id: str) -> dict:
        return self._index["examples"][example_id]

    def _handle(self, example_id: str):
        shard = self._index["examples"][example_id]["shard"]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                str(self.root / f"shard-{shard:05d}.safetensors"), framework="pt"
            )
        return self._handles[shard]

    def hidden(self, example_id: str, layer: int) -> torch.Tensor:
        return self._handle(example_id).get_tensor(f"{example_id}/h{layer:02d}")

    def logits(self, example_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._handle(example_id)
        return (
            h.get_tensor(f"{example_id}/topk_v"),
            h.get_tensor(f"{example_id}/topk_i"),
            h.get_tensor(f"{example_id}/logz"),
        )
