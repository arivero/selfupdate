#!/usr/bin/env python3
"""Materialize a v5 subset selected by task-aware teacher-answer score.

Next/previous examples use ``word_acc``; cloze examples use ``containment``.
The output retains source-dataset order and carries no generated answer text:
answers and hidden targets remain cache content, as required by v5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_score(row: dict) -> float:
    if row.get("word_acc") is not None:
        return float(row["word_acc"])
    if row.get("containment") is not None:
        return float(row["containment"])
    raise ValueError(f"response {row.get('example_id')} has no task score")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--score", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    examples_path = Path(args.examples)
    responses_path = Path(args.responses)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else Path(
        str(out_path) + ".manifest.json")

    examples = [json.loads(line) for line in examples_path.read_text(
        encoding="utf-8").splitlines()]
    responses = [json.loads(line) for line in responses_path.read_text(
        encoding="utf-8").splitlines()]
    by_id = {row["example_id"]: row for row in responses}
    if len(by_id) != len(responses):
        raise ValueError("duplicate example_id in responses")
    example_ids = [row["example_id"] for row in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("duplicate example_id in source dataset")
    missing = [example_id for example_id in example_ids
               if example_id not in by_id]
    extras = sorted(set(by_id) - set(example_ids))
    if missing or extras:
        raise ValueError(
            f"dataset/response identity mismatch: missing={len(missing)} "
            f"extras={len(extras)}")

    selected = [row for row in examples
                if task_score(by_id[row["example_id"]]) == args.score]
    if not selected:
        raise ValueError("score filter selected no examples")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n"
                for row in selected),
        encoding="utf-8",
    )

    kinds = Counter(row.get("kind", "unknown") for row in selected)
    corpora = Counter(row.get("corpus", "unknown") for row in selected)
    selected_ids = {row["example_id"] for row in selected}
    answer_tokens = sum(int(row.get("answer_tokens", 0))
                        for row in responses
                        if row["example_id"] in selected_ids)
    manifest = {
        "schema_version": 1,
        "selection": "task_aware_teacher_answer_score_exact",
        "score": args.score,
        "next_previous_metric": "word_acc",
        "cloze_metric": "containment",
        "allows_unscored_extra_answer_text": True,
        "source_examples": str(examples_path),
        "source_examples_sha256": sha256(examples_path),
        "source_responses": str(responses_path),
        "source_responses_sha256": sha256(responses_path),
        "source_count": len(examples),
        "selected_count": len(selected),
        "selected_fraction": len(selected) / len(examples),
        "selected_teacher_answer_tokens": answer_tokens,
        "kind_counts": dict(sorted(kinds.items())),
        "corpus_counts": dict(sorted(corpora.items())),
        "output": str(out_path),
        "output_sha256": sha256(out_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
