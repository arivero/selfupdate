"""Dataset joining examples.jsonl with the frozen-teacher cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..masking import ContextMasker, SegmentedExample
from ..teacher.cache import TeacherCache


@dataclass
class Item:
    example_id: str
    student_ids: torch.Tensor  # [n]
    position_ids: torch.Tensor  # [n]
    s0: int  # aligned-span start in the student sequence
    A: int  # aligned-span length
    hidden: dict[int, torch.Tensor]  # L -> [A, H] teacher targets (fp16)
    topk_v: torch.Tensor | None
    topk_i: torch.Tensor | None
    logz: torch.Tensor | None


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


class DistillDataset(Dataset):
    """Yields student inputs plus lazily-read teacher targets.

    ``need_layers`` limits hidden-state reads to what the trainer uses
    (None = none); ``need_logits`` gates the top-k read. ``rebase_gap``
    shifts aligned-span position ids by the privileged-block length
    (the stub_gap compaction variant).
    """

    def __init__(
        self,
        examples_path: str | Path,
        cache: TeacherCache,
        tokenizer,
        need_layers: list[int] | None = None,
        need_logits: bool = True,
        rebase_gap: bool = False,
    ):
        self.records = load_jsonl(examples_path)
        self.cache = cache
        self.need_layers = need_layers or []
        self.need_logits = need_logits
        self.rebase_gap = rebase_gap
        masker = ContextMasker(tokenizer)
        self.pairs = []
        for r in self.records:
            ex = SegmentedExample.from_json(
                {k: r[k] for k in ("example_id", "shared_prefix", "privileged",
                                   "shared_mid", "answer", "student_stub")}
            )
            pair = masker.build(ex)
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
        topk_v = topk_i = logz = None
        if self.need_logits:
            topk_v, topk_i, logz = self.cache.logits(ex_id)
        return Item(
            example_id=ex_id,
            student_ids=torch.tensor(pair.student_ids),
            position_ids=torch.tensor(pair.student_position_ids(self.rebase_gap)),
            s0=pair.s_aligned.start,
            A=pair.aligned_len,
            hidden=hidden,
            topk_v=topk_v,
            topk_i=topk_i,
            logz=logz,
        )


def collate_items(items: list[Item]) -> list[Item]:
    """Micro-batches are processed per-item (variable lengths, batch=1 fwd)."""
    return items
