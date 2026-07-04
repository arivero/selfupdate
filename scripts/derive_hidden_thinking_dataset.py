"""Derive hidden-thinking examples from a visible RAG-thinking dataset.

The visible and hidden arms should differ in censorship, not in sampled
teacher traces. This script reuses the frozen trace from a visible
rag_thinking/rag_mayeutic jsonl and renders the corresponding hidden variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.masking import (
    IM_END,
    render_rag_hidden_mayeutic,
    render_rag_hidden_thinking,
)


DOC_PREFIX = "\n\nDocumento recuperado:\n"
THINK_SPLIT = "\n</think>\n\n"


def _split_trace_answer(answer: str) -> tuple[str, str]:
    body = answer.removesuffix(IM_END)
    if THINK_SPLIT not in body:
        raise ValueError("visible answer does not contain a closing think block")
    trace, poem = body.split(THINK_SPLIT, 1)
    return trace.strip(), poem


def _passage(record: dict) -> str:
    privileged = record.get("privileged", "")
    if not privileged:
        return ""
    if not privileged.startswith(DOC_PREFIX):
        raise ValueError(f"unexpected privileged RAG format in {record.get('example_id')}")
    return privileged[len(DOC_PREFIX):]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        required=True,
        choices=("rag_hidden_thinking", "rag_hidden_mayeutic"),
    )
    args = ap.parse_args()

    renderer = {
        "rag_hidden_thinking": render_rag_hidden_thinking,
        "rag_hidden_mayeutic": render_rag_hidden_mayeutic,
    }[args.mode]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(args.source).open(encoding="utf-8") as src, out.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            record = json.loads(line)
            trace, answer = _split_trace_answer(record["answer"])
            ex = renderer(
                record["example_id"],
                record["question"],
                _passage(record),
                trace,
                answer,
                student_stub=record.get("student_stub", ""),
            )
            dst.write(
                json.dumps(
                    {**ex.to_json(), "answer_text": answer, "question": record["question"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"derived {n} {args.mode} examples from {args.source} to {out}")


if __name__ == "__main__":
    main()
