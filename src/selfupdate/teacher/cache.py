"""Frozen-teacher cache: per-layer hidden states on disk.

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
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_NAME = "index.json"
CACHE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def cache_storage_dtype(name: str) -> torch.dtype:
    """Torch dtype selected by ``cache.hidden_dtype``.

    The old writer always cast to fp16 even when the config hash claimed
    bfloat16.  Reject unknown spellings before a cache path is selected.
    """
    try:
        return CACHE_DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            "cache.hidden_dtype must be one of "
            f"{', '.join(CACHE_DTYPES)}, got {name!r}") from exc


def _dtype_hash_token(name: str) -> str:
    """Keep correct legacy fp16 caches, invalidate historical bad bf16 ones."""
    cache_storage_dtype(name)
    # Existing float16 caches were written correctly and retain their cache
    # identity.  Historical bfloat16-labelled caches contain fp16 tensors;
    # the versioned token forces them onto a fresh path.
    return name if name == "float16" else f"{name}-storage-v2"


def cache_config_hash(model_name: str, mask_mode: str, extra: dict | None = None) -> str:
    payload = {"model": model_name, "mode": mask_mode, "span": "mid+answer/v1"}
    payload.update(extra or {})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resolve_cache_dir(cfg) -> tuple[Path, str]:
    """Canonical cache directory + expected hash for an ExperimentConfig.
    Must mirror scripts/build_teacher_cache.py exactly. The hash covers every
    payload-shaping parameter (including the v5 answer-generation budget) so a config change
    can never silently reuse an incompatible cache."""
    examples_sha = hashlib.sha256(
        Path(cfg.data.examples_path).read_bytes()
    ).hexdigest()[:16]
    response_sha = (hashlib.sha256(
        Path(cfg.cache.generation_responses_path).read_bytes()).hexdigest()[:16]
        if cfg.cache.generation_responses_path else "")
    source_compaction = cfg.cache.source_compaction or cfg.mask.compaction
    chash = cache_config_hash(
        cfg.model.name, cfg.mask.mode,
        {"compaction": source_compaction, "examples": examples_sha,
         "hdtype": _dtype_hash_token(cfg.cache.hidden_dtype),
         # Open-answer caches include the teacher's generated answer ids in
         # their index.  The allowance changes those ids, aligned lengths, and
         # every stored h[L] payload, so it must be part of cache identity.
         "generation_extra_tokens": int(cfg.cache.generation_extra_tokens),
         # Batched greedy decode can differ from B=1 at exact argmax ties;
         # inference dtype likewise shapes both answers and hidden targets.
         "generation_batch": int(cfg.cache.generation_batch),
         "teacher_batch": int(cfg.cache.teacher_batch),
         "generation_budget_bucket": int(cfg.cache.generation_budget_bucket),
         "generation_compile": bool(cfg.cache.generation_compile),
         "generation_cache_implementation": cfg.cache.generation_cache_implementation,
         "generation_compile_dynamic": bool(cfg.cache.generation_compile_dynamic),
         "generation_cache_max_tokens": int(cfg.cache.generation_cache_max_tokens),
         "generation_fixed_batch": bool(cfg.cache.generation_fixed_batch),
         "generation_responses_sha": response_sha,
         "generation_shuffle_seed": int(cfg.cache.generation_shuffle_seed),
         "max_sequence_tokens": int(cfg.cache.max_sequence_tokens),
         "limit": int(cfg.cache.limit),
         "model_dtype": cfg.model.dtype,
         "schema": 9},
    )
    model_short = cfg.model.name.split("/")[-1]
    cache_root = os.environ.get("SELFUPDATE_TEACHER_CACHE_ROOT", cfg.cache.root)
    root = Path(cache_root) / f"{model_short}-{cfg.mask.mode}-{source_compaction}-{chash}"
    return root, chash


class TeacherCacheWriter:
    def __init__(self, root: str | Path, config_hash: str, shard_size: int = 128,
                 hidden_dtype: str = "float16"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.shard_size = shard_size
        self.hidden_dtype_name = hidden_dtype
        self.hidden_dtype = cache_storage_dtype(hidden_dtype)
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffered_examples = 0
        self._shard_no = 0
        self._index: dict = {
            "config_hash": config_hash,
            "hidden_dtype": hidden_dtype,
            "examples": {},
        }

    def add(
        self,
        example_id: str,
        hidden: dict[int, torch.Tensor],  # L -> [A, H]
        span: dict,
        extra: dict | None = None,
        finite_checked: bool = False,
    ) -> None:
        """``extra`` (v5 open-answer records): generation artifacts merged
        into the index entry — ``answer_ids`` (the teacher's generated
        answer, token ids), ``hard_cut``, ``answer_text``. Answers are cache
        content, not dataset content: the cache is the per-model artifact."""
        span = {**span, **(extra or {})}
        for L, h in hidden.items():
            stored = h.detach().to(self.hidden_dtype).contiguous().cpu()
            if not finite_checked and not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"teacher cache would store non-finite values for "
                    f"{example_id}/h{L:02d} as {self.hidden_dtype_name}; "
                    "use bfloat16 for outlier channels or inspect the teacher forward")
            self._buffer[f"{example_id}/h{L:02d}"] = stored
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


class AsyncTeacherCacheWriter:
    """Single background consumer for D2H-ready cache payloads.

    The producer may attach a CUDA event recorded after a non-blocking copy to
    pinned CPU memory.  The writer thread waits for that event, then performs
    finite checks, shard buffering, and safetensors writes while the default
    CUDA stream advances the next teacher forward.
    """

    def __init__(self, root: str | Path, config_hash: str,
                 shard_size: int = 128, hidden_dtype: str = "float16",
                 queue_size: int = 8):
        self._writer = TeacherCacheWriter(
            root, config_hash, shard_size=shard_size,
            hidden_dtype=hidden_dtype)
        self.hidden_dtype = self._writer.hidden_dtype
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._error: BaseException | None = None
        self.copy_seconds = 0.0
        self.storage_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run, name="teacher-cache-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                copy_start_event, ready_event, finite_flag, args, kwargs = item
                if self._error is None:
                    if ready_event is not None:
                        starts = (list(copy_start_event)
                                  if isinstance(copy_start_event, (list, tuple))
                                  else [copy_start_event])
                        ready = (list(ready_event)
                                 if isinstance(ready_event, (list, tuple))
                                 else [ready_event])
                        for event in ready:
                            event.synchronize()
                        # Device groups copy concurrently.  Count the D2H wall
                        # critical path, not the sum of overlapping card-local
                        # intervals; the scalar single-card behavior is unchanged.
                        self.copy_seconds += max(
                            start.elapsed_time(end)
                            for start, end in zip(starts, ready)
                        ) / 1000.0
                    if finite_flag is not None:
                        if not bool(finite_flag.item()):
                            raise FloatingPointError(
                                "teacher cache would store non-finite values for "
                                f"{args[0]}; inspect the teacher forward")
                        kwargs["finite_checked"] = True
                    started = time.perf_counter()
                    self._writer.add(*args, **kwargs)
                    self.storage_seconds += time.perf_counter() - started
            except BaseException as exc:  # propagate on producer/finalize
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("background teacher-cache writer failed") from self._error

    def add(self, *args, copy_start_event=None, ready_event=None,
            finite_flag=None, **kwargs) -> None:
        self._raise_if_failed()
        self._queue.put((copy_start_event, ready_event, finite_flag, args, kwargs))
        self._raise_if_failed()

    def finalize(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()
        started = time.perf_counter()
        self._writer.finalize()
        self.storage_seconds += time.perf_counter() - started


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

    def answer_ids(self, example_id: str) -> list[int] | None:
        """Generated answer ids for open-answer (v5) examples; None for
        legacy caches whose answers live in the dataset."""
        return self._index["examples"][example_id].get("answer_ids")
