"""Shared V5 prompt sample for the CPU generator demo.

Builds the exact same prompts as ``scripts/benchmark_vllm_generation.py``
(native format): ContextMasker teacher ids, per-record generation budget,
chat-template stop token.  Both contenders (pure-torch and vLLM CPU) consume
the same list, so speed and outputs are directly comparable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from selfupdate.chatfmt import adapt_records, stop_token_id  # noqa: E402
from selfupdate.config import load_config  # noqa: E402
from selfupdate.masking import ContextMasker, SegmentedExample  # noqa: E402

from build_teacher_cache import _generation_budget  # noqa: E402

DEFAULT_EXAMPLES = "data/combined/examples_v5rs_window.jsonl"
DEFAULT_EXPERIMENT = "configs/experiments/v5rs/cache_0p6b_window_remove.yaml"


def build_prompt_sample(tokenizer, *, examples: str = DEFAULT_EXAMPLES,
                        config: str = "configs/base.yaml",
                        experiment: str = DEFAULT_EXPERIMENT,
                        limit: int = 64) -> list[dict]:
    """Return ``[{example_id, ids, budget, stop_id}]`` for a V5 sample.

    ``limit`` uses the benchmark's deterministic evenly-spaced subsample so
    all task/corpus regions are represented (records are ordered by corpus).
    """
    cfg = load_config(config, experiment)
    examples_path = ROOT / examples
    records = [json.loads(x) for x in
               examples_path.read_text(encoding="utf-8").splitlines()]
    records = adapt_records(records, tokenizer)
    if limit and limit < len(records):
        step = len(records) / limit
        records = [records[min(int(i * step), len(records) - 1)]
                   for i in range(limit)]
    segmented = [SegmentedExample.from_record(x) for x in records]
    masker = ContextMasker(tokenizer,
                           pad_random=(cfg.mask.compaction == "pad_random"))
    stop_id = stop_token_id(tokenizer)
    prompts = []
    for record, ex in zip(records, segmented):
        budget = _generation_budget(masker, ex,
                                    int(record.get("expected_answer_chars", 64)),
                                    cfg.cache.generation_extra_tokens)
        prompts.append({"example_id": ex.example_id,
                        "ids": masker.build(ex).teacher_ids,
                        "budget": budget, "stop_id": stop_id})
    return prompts
