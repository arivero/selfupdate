"""Dataset joining examples.jsonl with the frozen-teacher cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

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
        )


def collate_items(items: list[Item]) -> list[Item]:
    """Micro-batches are processed per-item (variable lengths, batch=1 fwd)."""
    return items
