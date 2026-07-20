"""Summarize standard destruction benchmark JSON files.

Primary signal for now is WikiText-2 perplexity ratio vs the epoch-zero teacher
of the same model family.  Ratios near 1 mean no measurable language-model
damage on this standard quantization-style metric; large ratios indicate
corruption even when recall looks good.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path



RUNS = Path("runs")
STD = RUNS / "standard_destruction"


def model_key(model: str) -> str:
    if "Qwen3-0.6B" in model:
        return "qwen3_0p6b"
    if "Qwen3-1.7B" in model:
        return "qwen3_1p7b"
    if "Qwen3-4B" in model:
        return "qwen3_4b"
    if "Qwen3-8B" in model:
        return "qwen3_8b"
    if "Qwen3-14B" in model:
        return "qwen3_14b"
    return model.rsplit("/", 1)[-1].replace(".", "p")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    if not STD.exists():
        print(f"missing {STD}", file=sys.stderr)
        return 1
    rows = []
    teachers = {}
    for path in sorted(STD.glob("teacher_*.json")):
        d = load(path)
        task = d["tasks"].get("wikitext2_ppl") or {}
        teachers[model_key(d["model"])] = {
            "teacher_file": path.name,
            "teacher_ppl": task.get("ppl"),
            "teacher_ce": task.get("mean_ce"),
        }
    for path in sorted(STD.glob("*.json")):
        if path.name.startswith("teacher_"):
            continue
        d = load(path)
        task = d["tasks"].get("wikitext2_ppl") or {}
        key = model_key(d["model"])
        teacher = teachers.get(key, {})
        ppl = task.get("ppl")
        ce = task.get("mean_ce")
        teacher_ppl = teacher.get("teacher_ppl")
        teacher_ce = teacher.get("teacher_ce")
        rows.append(
            {
                "run": path.stem,
                "model": d["model"],
                "wikitext2_ppl": ppl,
                "teacher_ppl": teacher_ppl,
                "ppl_ratio": (ppl / teacher_ppl) if ppl and teacher_ppl else math.nan,
                "mean_ce": ce,
                "teacher_ce": teacher_ce,
                "ce_delta": (ce - teacher_ce) if ce is not None and teacher_ce is not None else math.nan,
                "n_tokens": task.get("n_tokens"),
                "file": str(path),
            }
        )
    rows.sort(key=lambda r: (str(r["model"]), float(r["ppl_ratio"])))
    out_csv = STD / "summary.csv"
    out_md = STD / "summary.md"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Standard Destruction Summary",
        "",
        "Primary metric: WikiText-2 validation perplexity ratio vs epoch-zero teacher.",
        "",
        "| run | model | PPL | teacher PPL | ratio | CE delta | tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['model'].rsplit('/', 1)[-1]} | "
            f"{float(r['wikitext2_ppl']):.2f} | {float(r['teacher_ppl']):.2f} | "
            f"{float(r['ppl_ratio']):.2f} | {float(r['ce_delta']):+.3f} | "
            f"{int(r['n_tokens'])} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_csv} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
