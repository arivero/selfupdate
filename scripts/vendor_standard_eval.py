"""Vendor deterministic standard-eval subsets at their pinned Hub revisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfupdate.eval.standard import BENCHMARK_REVISIONS
from datasets import load_dataset


def arc_challenge() -> list[dict]:
    ds = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split="validation",
        revision=BENCHMARK_REVISIONS["allenai/ai2_arc"],
    )
    rows = []
    for row in ds:
        labels = list(row["choices"]["label"])
        answer = str(row["answerKey"])
        if answer not in labels and answer.isdigit() and str(int(answer)) in labels:
            answer = str(int(answer))
        if answer not in labels:
            continue
        rows.append({
            "id": row.get("id"),
            "prompt": f"Question: {row['question'].strip()}\nAnswer:",
            "choices": [f" {x.strip()}" for x in row["choices"]["text"]],
            "target": labels.index(answer),
        })
        if len(rows) == 100:
            break
    return rows


def hellaswag() -> list[dict]:
    ds = load_dataset(
        "Rowan/hellaswag", split="validation",
        revision=BENCHMARK_REVISIONS["Rowan/hellaswag"],
    )
    return [{
        "id": row.get("ind"),
        "prompt": f"{row['ctx_a']} {row['ctx_b']}".strip(),
        "choices": [f" {x.strip()}" for x in row["endings"]],
        "target": int(row["label"]),
    } for row in ds.select(range(100))]


def write(path: Path, source: str, revision: str, items: list[dict]) -> None:
    path.write_text(json.dumps({
        "source": source, "revision": revision, "items": items,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} items={len(items)}", flush=True)


def main() -> None:
    out = Path("data/eval")
    out.mkdir(parents=True, exist_ok=True)
    write(out / "arc_challenge_v1.json", "allenai/ai2_arc:ARC-Challenge",
          BENCHMARK_REVISIONS["allenai/ai2_arc"], arc_challenge())
    write(out / "hellaswag_v1.json", "Rowan/hellaswag",
          BENCHMARK_REVISIONS["Rowan/hellaswag"], hellaswag())


if __name__ == "__main__":
    main()
