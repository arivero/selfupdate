"""Live, corpus-separated scorecard for the 1.7B layerwise loss grid.

This report never averages Machado and Quijote into a hidden task.  Each row
states whether its recall values come from the fast in-training probe or the
full post-checkpoint evaluation, and pairs capability damage with the pinned
100-item-per-task teacher reference.

Usage:
    python scripts/lossgrid_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
STANDARD = RUNS / "standard_damage"
CORPORA = ("machado", "quijote_ch1", "quijote_ch4")


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metrics(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _last(rows: list[dict], kind: str) -> dict:
    return next((row for row in reversed(rows) if row.get("kind") == kind), {})


def _epoch0(rows: list[dict], kind: str) -> dict:
    return next((row for row in rows
                 if row.get("kind") == kind and row.get("phase") == "epoch0"), {})


def _run_config(run_dir: Path) -> dict:
    try:
        return yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    except OSError:
        return {}


def _standard_baseline() -> tuple[dict[str, float], float]:
    paths = sorted(STANDARD.glob("teacher_Qwen_Qwen3-1.7B_a_lossgrid*.json"))
    if not paths:
        return {}, float("nan")
    data = _json(paths[-1]) or {}
    scores = {task: value["accuracy"] for task, value in data.get("tasks", {}).items()}
    return scores, sum(scores.values()) / len(scores) if scores else float("nan")


def _score(row: dict, key: str) -> str:
    value = row.get(key)
    return "—" if value is None else f"{value:.3f}"


def collect() -> list[dict]:
    base_scores, _ = _standard_baseline()
    rows = []
    for run_dir in sorted(RUNS.glob("a_lossgrid_1p7b_*")):
        if not run_dir.is_dir():
            continue
        cfg = _run_config(run_dir)
        metrics = _metrics(run_dir / "metrics.jsonl")
        train, fast_eval, fast_standard = (
            _last(metrics, "train"), _last(metrics, "eval"), _last(metrics, "standard_eval")
        )
        epoch0_eval = _epoch0(metrics, "eval")
        epoch0_standard = _epoch0(metrics, "standard_eval")
        final_recall = _json(run_dir / "eval" / "tasks.json")
        final_standard = _json(STANDARD / f"{run_dir.name}.json")
        recall = {}
        recall_source = "fast epoch probe"
        if final_recall:
            recall = {
                corpus: result.get("overall_word_acc")
                for corpus, result in final_recall.get("corpora", {}).items()
            }
            recall_source = "full checkpoint eval"
        else:
            recall = {
                corpus: result.get("overall_word_acc")
                for corpus, result in fast_eval.get("recall", {}).items()
            }

        standard_scores = {}
        damage_delta = damage_worst = None
        if final_standard:
            standard_scores = {
                task: result["accuracy"]
                for task, result in final_standard.get("tasks", {}).items()
            }
            deltas = {
                task: standard_scores[task] - base_scores[task]
                for task in standard_scores if task in base_scores
            }
            damage_delta = (sum(deltas.values()) / len(deltas) if deltas else None)
            damage_worst = min(deltas.values()) if deltas else None
        else:
            standard_scores = fast_standard.get("standard_tasks", {})
            # Fast epoch telemetry is paired to its own fixed 16-item epoch-0
            # subset.  Comparing that small subset to the full 100-item
            # reference would manufacture a sampling delta.
            damage_delta = fast_standard.get("standard_epoch0_delta")
            damage_worst = fast_standard.get("standard_worst_delta")
        standard_macro = (sum(standard_scores.values()) / len(standard_scores)
                          if standard_scores else None)

        values = [recall.get(corpus) for corpus in CORPORA]
        values = [value for value in values if value is not None]
        epoch0_recall = {
            corpus: result.get("overall_word_acc")
            for corpus, result in epoch0_eval.get("recall", {}).items()
        }
        epoch0_values = [epoch0_recall.get(corpus) for corpus in CORPORA]
        epoch0_values = [value for value in epoch0_values if value is not None]
        status = ("complete" if final_recall and final_standard else "training")
        rows.append({
            "run": run_dir.name,
            "loss": cfg.get("train", {}).get("hidden_loss", "unknown"),
            "slide": cfg.get("train", {}).get("conn_window", "?"),
            "status": status,
            "items_seen": train.get("items_seen"),
            "latest_epoch": fast_eval.get("epoch"),
            "recall_source": recall_source,
            **{f"epoch0_{corpus}": epoch0_recall.get(corpus) for corpus in CORPORA},
            "epoch0_recall_mean": (sum(epoch0_values) / len(epoch0_values)
                                   if epoch0_values else None),
            **{corpus: recall.get(corpus) for corpus in CORPORA},
            "recall_mean": sum(values) / len(values) if values else None,
            "standard_source": "full checkpoint eval" if final_standard else "fast epoch probe",
            "epoch0_standard_macro": epoch0_standard.get("standard_macro_accuracy"),
            "standard_macro": standard_macro,
            "standard_delta": damage_delta,
            "standard_worst_delta": damage_worst,
        })
    return sorted(rows, key=lambda row: (
        row["status"] != "complete",
        -(row["recall_mean"] if row["recall_mean"] is not None else -1),
        row["run"],
    ))


def write(rows: list[dict]) -> None:
    csv_path = RUNS / "lossgrid_report.csv"
    md_path = RUNS / "lossgrid_report.md"
    fields = [
        "run", "loss", "slide", "status", "items_seen", "latest_epoch",
        "recall_source", *(f"epoch0_{c}" for c in CORPORA), "epoch0_recall_mean",
        *CORPORA, "recall_mean", "standard_source", "epoch0_standard_macro",
        "standard_macro", "standard_delta", "standard_worst_delta",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 1.7B Loss-Grid Live Report", "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "Recall columns are deliberately corpus-separated. A `fast epoch probe` "
        "uses the fixed in-training subset; `full checkpoint eval` is the "
        "post-training evaluation. Standard deltas are paired within their "
        "stated source: fast epoch-0 subset or full pinned Qwen3-1.7B reference "
        "on ARC-Easy, ARC-Challenge, and HellaSwag.",
        "",
        "Deliberately unqueued: `lens_js` slide1/slide2 configs exist but were "
        "never run — a bounded symmetric-divergence control, not a sweep "
        "candidate (issues.md low-priority item 13); absence is by design, "
        "not a missing artifact.",
        "",
        "| run | loss | slide | status | items | epoch | source | epoch-0 M/Q1/Q4 | final M/Q1/Q4 | e0 mean | final mean | e0 standard | standard Δ | worst Δ |",
        "|---|---|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['loss']} | {row['slide']} | {row['status']} | "
            f"{row['items_seen'] or '—'} | {row['latest_epoch'] or '—'} | "
            f"{row['recall_source']} | {_score(row, 'epoch0_machado')}/"
            f"{_score(row, 'epoch0_quijote_ch1')}/{_score(row, 'epoch0_quijote_ch4')} | "
            f"{_score(row, 'machado')}/{_score(row, 'quijote_ch1')}/{_score(row, 'quijote_ch4')} | "
            f"{_score(row, 'epoch0_recall_mean')} | {_score(row, 'recall_mean')} | "
            f"{_score(row, 'epoch0_standard_macro')} | {_score(row, 'standard_delta')} | "
            f"{_score(row, 'standard_worst_delta')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    write(collect())
