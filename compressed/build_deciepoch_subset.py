"""Build the fixed 100-question Layerwise 3.4 timing traversal.

The subset is deterministic and model-independent.  It preserves the source
corpus/kind mixture with largest-remainder allocation, then samples evenly
across expected-answer length within each stratum.  Teacher-realized answers
remain model-specific cache content.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _allocate(counts: dict[tuple[str, str], int], total: int) -> dict:
    population = sum(counts.values())
    exact = {key: total * count / population for key, count in counts.items()}
    allocated = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def _evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: (
        int(row.get("expected_answer_chars", 0)), row["example_id"]))
    if count >= len(rows):
        return rows
    # Midpoints of equal population bins avoid privileging either length tail.
    return [rows[min(int((index + 0.5) * len(rows) / count), len(rows) - 1)]
            for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("data/combined/examples_v5rs_window.jsonl"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/combined/examples_v5rs_window_deciepoch100.jsonl"))
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    if not 0 < args.count <= len(rows):
        raise ValueError("--count must be inside the source population")
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(str(row.get("corpus", "")), str(row.get("kind", "")))].append(row)
    allocation = _allocate({key: len(value) for key, value in strata.items()},
                           args.count)
    selected = []
    for key in sorted(strata):
        selected.extend(_evenly_spaced(strata[key], allocation[key]))
    selected.sort(key=lambda row: row["example_id"])
    if len(selected) != args.count or len({r["example_id"] for r in selected}) != args.count:
        raise RuntimeError("selection did not produce distinct requested rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in selected))
    manifest = {
        "schema": "layerwise_3.4_deciepoch_subset/v1",
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "output": str(args.out),
        "output_sha256": _sha256(args.out),
        "selection": "largest_remainder_corpus_kind_then_length_bin_midpoints",
        "questions": len(selected),
        "strata": {
            f"{corpus}/{kind}": {
                "source": len(strata[(corpus, kind)]),
                "selected": allocation[(corpus, kind)],
            }
            for corpus, kind in sorted(strata)
        },
        "example_ids": [row["example_id"] for row in selected],
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(selected)} questions, sha256={manifest['output_sha256']}")


if __name__ == "__main__":
    main()
