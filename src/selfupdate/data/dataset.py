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
    aligned rows; ``readout_mask`` marks shifted answer rows for readout/lens
    losses.
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
    readout_index: torch.Tensor      # [B, Rmax]
    readout_mask: torch.Tensor       # [B, Rmax] bool
    teacher_ids: torch.Tensor | None = None
    t0: torch.Tensor | None = None
    t_priv: list | None = None


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


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
    ):
        # re-render segments if this tokenizer's chat template differs from
        # the one examples.jsonl was built with (identity for Qwen)
        self.records = adapt_records(load_jsonl(examples_path), tokenizer)
        self.cache = cache
        self.need_layers = need_layers or []
        self.rebase_gap = rebase_gap
        self.with_teacher_ids = with_teacher_ids
        masker = ContextMasker(tokenizer)
        self.pairs = []
        for r in self.records:
            pair = masker.build(SegmentedExample.from_record(r))
            if cache is not None:
                span = cache.span(pair.example_id)
                assert span["A"] == pair.aligned_len and span["s0"] == pair.s_aligned.start, (
                    f"cache/examples mismatch for {pair.example_id}; rebuild the cache"
                )
            self.pairs.append(pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Item:
        pair = self.pairs[idx]
        ex_id = pair.example_id
        hidden = {L: self.cache.hidden(ex_id, L) for L in self.need_layers}
        return Item(
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


def collate_items(items: list[Item]) -> list[Item]:
    """Micro-batches are processed per-item (variable lengths, batch=1 fwd)."""
    return items


def collate_padded_items(items: list[Item]) -> Batch:
    """Right-pad items for a real batched forward/backward.

    We intentionally pad on the right and keep ``attention_mask=None`` in the
    block runner: causal attention prevents valid positions from seeing future
    pad rows, so the valid hidden states are identical to the unpadded path.
    """
    if not items:
        raise ValueError("empty batch")
    B = len(items)
    lengths = torch.tensor([len(it.student_ids) for it in items], dtype=torch.long)
    T = int(lengths.max())
    Amax = max(it.A for it in items)
    ans_lens = [it.s0 + it.A - it.ans0 for it in items]
    Rmax = max(ans_lens) if ans_lens else 0

    student_ids = torch.zeros(B, T, dtype=torch.long)
    position_ids = torch.zeros(B, T, dtype=torch.long)
    s0 = torch.tensor([it.s0 for it in items], dtype=torch.long)
    A = torch.tensor([it.A for it in items], dtype=torch.long)
    ans0 = torch.tensor([it.ans0 for it in items], dtype=torch.long)
    aligned_index = torch.zeros(B, Amax, dtype=torch.long)
    hidden_mask = torch.zeros(B, Amax, dtype=torch.bool)
    readout_index = torch.zeros(B, Rmax, dtype=torch.long)
    readout_mask = torch.zeros(B, Rmax, dtype=torch.bool)

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
        rlen = ans_lens[i]
        if rlen > 0:
            readout_index[i, :rlen] = torch.arange(it.ans0 - 1, it.s0 + it.A - 1)
            readout_mask[i, :rlen] = True
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
        readout_index=readout_index,
        readout_mask=readout_mask,
        teacher_ids=teacher_ids,
        t0=torch.tensor([it.t0 for it in items], dtype=torch.long),
        t_priv=[it.t_priv for it in items],
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
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
