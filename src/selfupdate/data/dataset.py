"""Dataset joining examples.jsonl with the frozen-teacher cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from ..chatfmt import adapt_records
from ..masking import ContextMasker, SegmentedExample
from ..teacher.cache import TeacherCache


@dataclass
class Item:
    example_id: str
    student_ids: torch.Tensor  # [n]
    position_ids: torch.Tensor  # [n]
    s0: int  # aligned-span start in the student sequence
    A: int  # aligned-span length
    ans0: int  # answer-span start in the student sequence (s0 + mid length)
    hidden: dict[int, torch.Tensor]  # L -> [A, H] teacher targets (fp16)
    # online-teacher mode: the teacher input, targets computed per step
    teacher_ids: torch.Tensor | None = None
    t0: int = 0  # aligned-span start in the teacher sequence
    # interleaved (thinking_selective) records: teacher-coordinate ranges of
    # each privileged run; empty list = single implicit block [s0, t0)
    t_priv: list | None = None


@dataclass
class Batch:
    """Right-padded training batch.

    Aligned hidden targets are padded separately from token sequences because
    each example has its own ``s0``/``A`` span. ``hidden_mask`` marks valid
    aligned rows.
    """

    example_ids: list[str]
    student_ids: torch.Tensor        # [B, T]
    position_ids: torch.Tensor       # [B, T]
    lengths: torch.Tensor            # [B]
    s0: torch.Tensor                 # [B]
    A: torch.Tensor                  # [B]
    ans0: torch.Tensor               # [B]
    aligned_index: torch.Tensor      # [B, Amax]
    hidden_mask: torch.Tensor        # [B, Amax] bool
    hidden: dict[int, torch.Tensor]  # L -> [B, Amax, H]
    teacher_ids: torch.Tensor | None = None
    t0: torch.Tensor | None = None
    t_priv: list | None = None
    # Coordinate metadata for an aligned-token tile.  ``aligned_offset`` is
    # the teacher-cache row represented by hidden[:, 0]; ``source_A`` is the
    # complete aligned length before tiling.  Ordinary collated batches use
    # offset zero and source_A == A.
    aligned_offset: torch.Tensor | None = None
    source_A: torch.Tensor | None = None


@dataclass(frozen=True)
class BatchGridTile:
    """One optimizer tile in answer x aligned-token x layer space.

    ``batch`` contains full causal token sequences but only the selected
    aligned teacher rows.  The layer coordinate is deliberately not sliced:
    the trainer must walk layers in forward order so layer L consumes the
    student state produced by L-1.
    """

    batch: Batch
    source_answer_indices: tuple[int, ...]
    aligned_starts: tuple[int, ...]
    aligned_stops: tuple[int, ...]
    source_aligned_lengths: tuple[int, ...]

    @property
    def answer_count(self) -> int:
        return len(self.source_answer_indices)

    @property
    def aligned_token_count(self) -> int:
        return sum(stop - start for start, stop in
                   zip(self.aligned_starts, self.aligned_stops))

    @property
    def completed_answer_count(self) -> int:
        return sum(stop == total for stop, total in
                   zip(self.aligned_stops, self.source_aligned_lengths))

    @property
    def coordinate_ranges(self) -> list[dict]:
        return [
            {
                "example_id": example_id,
                "answer_index": source_i,
                "aligned_start": start,
                "aligned_stop": stop,
                "source_aligned_length": total,
            }
            for example_id, source_i, start, stop, total in zip(
                self.batch.example_ids, self.source_answer_indices,
                self.aligned_starts, self.aligned_stops,
                self.source_aligned_lengths)
        ]


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def is_open_answer_dataset(path: str | Path) -> bool:
    """True for v5 question-only datasets (empty answer segments): the
    teacher's generated answers live in the teacher cache, so even
    online-teacher runs must load the cache as their ANSWER source."""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    return bool(first) and json.loads(first).get("answer") == ""


class DistillDataset(Dataset):
    """Yields student inputs plus lazily-read teacher targets.

    ``need_layers`` limits hidden-state reads to what the trainer uses.
    ``rebase_gap`` shifts aligned-span position ids by the privileged-block
    length (the stub_gap compaction variant).
    """

    def __init__(
        self,
        examples_path: str | Path,
        cache: TeacherCache | None,
        tokenizer,
        need_layers: list[int] | None = None,
        rebase_gap: bool = False,
        with_teacher_ids: bool = False,
        pad_random: bool = False,
        cache_source_compaction: str = "",
        student_compaction: str = "remove",
    ):
        # re-render segments if this tokenizer's chat template differs from
        # the one examples.jsonl was built with (identity for Qwen)
        self.records = adapt_records(load_jsonl(examples_path), tokenizer)
        self.cache = cache
        # items are memoized after first build: the teacher-cache reads
        # (Lustre I/O on the training thread, num_workers=0) and tensor
        # conversions would otherwise repeat every epoch. Cost is host RAM
        # (~needed cache size, fp16), not VRAM. Consumers treat Items as
        # read-only — collate and .to(device) both copy.
        self._item_cache: dict[int, Item] = {}
        self.need_layers = need_layers  # via setter: validates cache presence
        self.rebase_gap = rebase_gap
        self.with_teacher_ids = with_teacher_ids
        masker = ContextMasker(
            tokenizer,
            pad_random=pad_random,
            keep_privileged=(student_compaction == "intact"),
        )
        self.pairs = []
        for r in self.records:
            ex = SegmentedExample.from_record(r)
            answer_ids = None
            if not ex.answer:
                # v5 open-answer record: the teacher's GENERATED answer is
                # cache content (per-model), injected as token ids so the
                # student is teacher-forced over exactly what the teacher
                # produced. Online generation is not wired yet — raise, never
                # silently train on an empty aligned span (knob-flow law).
                if cache is None:
                    raise NotImplementedError(
                        "open-answer (v5) records need the disk teacher "
                        "cache; per-step online generation is a planned "
                        "extension — build the cache with "
                        "scripts/build_teacher_cache.py")
                answer_ids = cache.answer_ids(ex.example_id)
                if answer_ids is None:
                    raise ValueError(
                        f"{ex.example_id}: teacher cache carries no generated "
                        "answer; rebuild with scripts/build_teacher_cache.py")
            pair = masker.build(ex, answer_ids=answer_ids)
            if cache is not None:
                span = cache.span(pair.example_id)
                # t0/position_gap have always been written by the cache
                # builder but were never read back — checking them is free
                # and narrows the tokenizer-drift window (a re-tokenized
                # prefix can move t0 while A and s0 stay equal). .get keeps
                # any pre-field cache index readable.
                # A cache may be explicitly sourced from another censorship
                # view.  Its payload is teacher-aligned h[L] plus generated
                # answer ids, so t0/A remain mandatory; s0/position_gap are
                # student-view metadata and are recomputed from ``pair``.
                teacher_match = (
                    span["A"] == pair.aligned_len
                    and span.get("t0", pair.t_aligned.start)
                        == pair.t_aligned.start)
                student_match = (
                    span["s0"] == pair.s_aligned.start
                    and span.get("position_gap", pair.position_gap)
                        == pair.position_gap)
                cross_view = bool(
                    cache_source_compaction
                    and cache_source_compaction != student_compaction)
                assert teacher_match and (cross_view or student_match), (
                    f"cache/examples mismatch for {pair.example_id}; rebuild the cache"
                )
            self.pairs.append(pair)

    @property
    def need_layers(self) -> list[int]:
        return self._need_layers

    @need_layers.setter
    def need_layers(self, layers) -> None:
        layers = layers or []
        if layers and self.cache is None:
            raise ValueError(
                "need_layers requires a teacher cache; online-teacher "
                "datasets must pass need_layers=[] (targets come per step)")
        # the sequential schedule swaps layers per stage: drop memoized
        # items so stale hidden targets are neither served nor retained
        self._need_layers = layers
        self._item_cache.clear()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Item:
        cached = self._item_cache.get(idx)
        if cached is not None:
            return cached
        pair = self.pairs[idx]
        ex_id = pair.example_id
        hidden = {L: self.cache.hidden(ex_id, L) for L in self.need_layers}
        item = Item(
            example_id=ex_id,
            student_ids=torch.tensor(pair.student_ids),
            position_ids=torch.tensor(pair.student_position_ids(self.rebase_gap)),
            s0=pair.s_aligned.start,
            A=pair.aligned_len,
            ans0=pair.s_answer.start,
            hidden=hidden,
            teacher_ids=torch.tensor(pair.teacher_ids) if self.with_teacher_ids else None,
            t0=pair.t_aligned.start,
            t_priv=pair.t_privileged or None,
        )
        self._item_cache[idx] = item
        return item


def collate_items(items: list[Item]) -> list[Item]:
    """Micro-batches are processed per-item (variable lengths, batch=1 fwd)."""
    return items


def collate_padded_items(items: list[Item]) -> Batch:
    """Right-pad items for a real batched forward/backward.

    We intentionally pad on the right and keep ``attention_mask=None`` in the
    block runner: causal attention prevents valid positions from seeing future
    pad rows, so the valid hidden states are identical to the unpadded path.

    Invariant: every mask marks a PREFIX of valid rows (``[:A]``).
    The trainer's per-example losses rely on this to slice by CPU-side lengths
    instead of bool-mask indexing (which would sync the GPU via nonzero()).
    """
    if not items:
        raise ValueError("empty batch")
    B = len(items)
    lengths = torch.tensor([len(it.student_ids) for it in items], dtype=torch.long)
    T = int(lengths.max())
    Amax = max(it.A for it in items)

    student_ids = torch.zeros(B, T, dtype=torch.long)
    position_ids = torch.zeros(B, T, dtype=torch.long)
    s0 = torch.tensor([it.s0 for it in items], dtype=torch.long)
    A = torch.tensor([it.A for it in items], dtype=torch.long)
    ans0 = torch.tensor([it.ans0 for it in items], dtype=torch.long)
    aligned_index = torch.zeros(B, Amax, dtype=torch.long)
    hidden_mask = torch.zeros(B, Amax, dtype=torch.bool)

    layers = sorted(items[0].hidden)
    hidden: dict[int, torch.Tensor] = {}
    for L in layers:
        H = items[0].hidden[L].shape[-1]
        dtype = items[0].hidden[L].dtype
        hidden[L] = torch.zeros(B, Amax, H, dtype=dtype)

    teacher_ids = None
    if all(it.teacher_ids is not None for it in items):
        t_lengths = [len(it.teacher_ids) for it in items]
        teacher_ids = torch.zeros(B, max(t_lengths), dtype=torch.long)

    for i, it in enumerate(items):
        n = len(it.student_ids)
        student_ids[i, :n] = it.student_ids
        position_ids[i, :n] = it.position_ids
        aligned_index[i, :it.A] = torch.arange(it.s0, it.s0 + it.A)
        hidden_mask[i, :it.A] = True
        for L in layers:
            hidden[L][i, :it.A] = it.hidden[L]
        if teacher_ids is not None:
            teacher_ids[i, :len(it.teacher_ids)] = it.teacher_ids

    return Batch(
        example_ids=[it.example_id for it in items],
        student_ids=student_ids,
        position_ids=position_ids,
        lengths=lengths,
        s0=s0,
        A=A,
        ans0=ans0,
        aligned_index=aligned_index,
        hidden_mask=hidden_mask,
        hidden=hidden,
        teacher_ids=teacher_ids,
        t0=torch.tensor([it.t0 for it in items], dtype=torch.long),
        t_priv=[it.t_priv for it in items],
        aligned_offset=torch.zeros(B, dtype=torch.long),
        source_A=A.clone(),
    )


def iter_batch_grid_tiles(batch: Batch, tokens_per_answer: int) -> list[BatchGridTile]:
    """Partition a padded batch along aligned-token coordinates.

    Every returned batch retains each selected answer's complete causal input
    sequence.  Only loss/teacher rows are sliced.  Short answers fall out of
    later tiles and are deliberately not replaced: missing tail cells enter
    neither the numerator nor denominator of the update reduction. Bucketed
    batching keeps those tails small without changing which grid cells are
    visited. ``tokens_per_answer == 0`` means one tile containing every valid
    aligned row.
    """
    if tokens_per_answer < 0:
        raise ValueError("tokens_per_answer must be >= 0")
    offsets = (batch.aligned_offset if batch.aligned_offset is not None
               else torch.zeros_like(batch.A))
    if bool(offsets.ne(0).any()):
        raise ValueError("cannot tile an already tiled batch")
    source_A = batch.source_A if batch.source_A is not None else batch.A
    max_A = int(source_A.max())
    if tokens_per_answer == 0:
        starts = tuple(int(v) for v in offsets.tolist())
        stops = tuple(start + int(count)
                      for start, count in zip(starts, batch.A.tolist()))
        return [BatchGridTile(
            batch=batch,
            source_answer_indices=tuple(range(len(batch.example_ids))),
            aligned_starts=starts,
            aligned_stops=stops,
            source_aligned_lengths=tuple(int(v) for v in source_A.tolist()),
        )]
    width = tokens_per_answer
    starts = [0] if max_A == 0 else range(0, max_A, width)
    return [_slice_batch_grid_tile(batch, int(start), width, source_A)
            for start in starts
            if bool(source_A.gt(start).any())]


def _slice_batch_grid_tile(
    batch: Batch, start: int, width: int, source_A: torch.Tensor,
) -> BatchGridTile:
    source_indices = [i for i, total in enumerate(source_A.tolist())
                      if total > start]
    row_index = torch.tensor(source_indices, dtype=torch.long)
    totals = [int(source_A[i]) for i in source_indices]
    counts = [min(width, total - start) for total in totals]
    stops = [start + count for count in counts]
    B = len(source_indices)
    Amax = max(counts)

    lengths = batch.lengths.index_select(0, row_index)
    T = int(lengths.max())
    student_ids = batch.student_ids.index_select(0, row_index)[:, :T]
    position_ids = batch.position_ids.index_select(0, row_index)[:, :T]
    s0 = batch.s0.index_select(0, row_index)
    ans0 = batch.ans0.index_select(0, row_index)
    A = torch.tensor(counts, dtype=torch.long)
    aligned_index = torch.zeros(B, Amax, dtype=torch.long)
    hidden_mask = torch.zeros(B, Amax, dtype=torch.bool)
    for j, (source_i, count) in enumerate(zip(source_indices, counts)):
        aligned_index[j, :count] = batch.aligned_index[source_i, start:start + count]
        hidden_mask[j, :count] = True

    hidden = {}
    for layer, value in batch.hidden.items():
        selected = value.index_select(0, row_index)[:, start:start + Amax]
        hidden[layer] = selected.clone()

    teacher_ids = (None if batch.teacher_ids is None else
                   batch.teacher_ids.index_select(0, row_index))
    t0 = None if batch.t0 is None else batch.t0.index_select(0, row_index)
    t_priv = (None if batch.t_priv is None else
              [batch.t_priv[i] for i in source_indices])
    tiled = Batch(
        example_ids=[batch.example_ids[i] for i in source_indices],
        student_ids=student_ids,
        position_ids=position_ids,
        lengths=lengths,
        s0=s0,
        A=A,
        ans0=ans0,
        aligned_index=aligned_index,
        hidden_mask=hidden_mask,
        hidden=hidden,
        teacher_ids=teacher_ids,
        t0=t0,
        t_priv=t_priv,
        aligned_offset=torch.full((B,), start, dtype=torch.long),
        source_A=torch.tensor(totals, dtype=torch.long),
    )
    return BatchGridTile(
        batch=tiled,
        source_answer_indices=tuple(source_indices),
        aligned_starts=(start,) * B,
        aligned_stops=tuple(stops),
        source_aligned_lengths=tuple(totals),
    )


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Randomized coarse length buckets for lower pad waste.

    This is deliberately not a global sort. Each epoch shuffles examples inside
    each length bin and shuffles the resulting mini-batches, so coverage stays
    stochastic while batch shapes get less ragged.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_width: int,
        seed: int,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_width <= 0:
            raise ValueError("bucket_width must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        buckets: dict[int, list[int]] = {}
        for idx, length in enumerate(self.lengths):
            buckets.setdefault(length // self.bucket_width, []).append(idx)

        batches: list[list[int]] = []
        for idxs in buckets.values():
            order = torch.randperm(len(idxs), generator=gen).tolist()
            shuffled = [idxs[j] for j in order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start: start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        order = torch.randperm(len(batches), generator=gen).tolist()
        for j in order:
            yield batches[j]

    def __len__(self) -> int:
        # batches form inside buckets, so the count is per-bucket, not a
        # global n/batch_size (bucket membership is fixed by lengths, so
        # this is exact for every epoch despite the shuffling)
        sizes: dict[int, int] = {}
        for length in self.lengths:
            b = length // self.bucket_width
            sizes[b] = sizes.get(b, 0) + 1
        if self.drop_last:
            return sum(s // self.batch_size for s in sizes.values())
        return sum((s + self.batch_size - 1) // self.batch_size
                   for s in sizes.values())
