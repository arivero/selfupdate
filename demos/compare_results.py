"""Compare the two engines' summaries and output agreement.

  python demos/compare_results.py demos/out/torch_cpu_b32 demos/out/vllm_cpu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(dirpath: str) -> tuple[dict, dict[str, dict]]:
    d = Path(dirpath)
    summary = json.loads((d / "summary.json").read_text())
    rows = {}
    for line in (d / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["example_id"]] = row
    return summary, rows


def word_overlap(a: str, b: str) -> float:
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    common = 0
    used = [False] * len(wb)
    for w in wa:
        for j, v in enumerate(wb):
            if not used[j] and v == w:
                used[j] = True
                common += 1
                break
    return 2 * common / (len(wa) + len(wb)) if (wa or wb) else 1.0


def main() -> None:
    (sum_a, rows_a), (sum_b, rows_b) = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(rows_a) & set(rows_b))
    exact = sum(1 for i in ids
                if rows_a[i]["token_ids"] == rows_b[i]["token_ids"])
    overlaps = [word_overlap(rows_a[i]["answer_text"], rows_b[i]["answer_text"])
                for i in ids]
    print(f"{sum_a['engine']:>12}: {sum_a['tokens_per_second']:8.1f} tok/s "
          f"({sum_a['gen_tokens']} tokens in {sum_a['generate_seconds']}s)")
    print(f"{sum_b['engine']:>12}: {sum_b['tokens_per_second']:8.1f} tok/s "
          f"({sum_b['gen_tokens']} tokens in {sum_b['generate_seconds']}s)")
    ratio = sum_a["tokens_per_second"] / sum_b["tokens_per_second"]
    print(f"speed ratio {sum_a['engine']}/{sum_b['engine']}: {ratio:.2f}x")
    print(f"outputs: {len(ids)} shared, {exact} token-identical, "
          f"mean word overlap {sum(overlaps) / len(overlaps):.3f}, "
          f"min {min(overlaps):.3f}")


if __name__ == "__main__":
    main()
