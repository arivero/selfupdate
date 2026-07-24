#!/usr/bin/env python3
"""Render v4.6 in-training telemetry without performing new evaluation."""
from __future__ import annotations
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


def load(run):
    rows = []
    for path in sorted(run.glob("stage*/metrics.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no metrics under {run}/stage*/metrics.jsonl")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    run = a.run.resolve()
    out = (a.out or run / "report").resolve()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(run)
    kinds = defaultdict(list)
    for row in rows:
        kinds[row.get("kind", "unknown")].append(row)
    figures = []

    def save(name, title):
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        figures.append((title, f"{name}.png"))

    def layer_series(kind, field, name, title):
        data = defaultdict(dict)
        for row in kinds[kind]:
            if row.get("partial"):
                continue
            for layer, value in (row.get(field) or {}).items():
                data[int(layer)][int(row["epoch"])] = float(value)
        if not data:
            return
        plt.figure(figsize=(12, 6))
        for layer, values in sorted(data.items()):
            xy = sorted(values.items())
            plt.plot(*zip(*xy), linewidth=1, label=f"L{layer}")
        plt.xlabel("epoch")
        plt.ylabel(field)
        plt.yscale("log")
        plt.legend(ncol=5, fontsize=7)
        save(name, title)

    layer_series("v4_epoch", "layer_losses", "loss_by_epoch_layer",
                 "Block-local loss by epoch and layer")
    layer_series("v4_gradient_norm", "grad_norms", "gradient_by_epoch_layer",
                 "Gradient norm by epoch and layer")

    for field, name, title in (
        ("per_layer_absolute_l2", "weight_delta_absolute",
         "Absolute effective LoRA weight delta"),
        ("per_layer_relative_l2", "weight_delta_relative",
         "Relative effective LoRA weight delta"),
    ):
        data = defaultdict(dict)
        for row in kinds["parameter_delta"]:
            for layer, value in enumerate(row.get(field) or []):
                if value:
                    data[layer][int(row["epoch"])] = float(value)
        if data:
            plt.figure(figsize=(12, 6))
            for layer, values in sorted(data.items()):
                xy = sorted(values.items())
                plt.plot(*zip(*xy), linewidth=1, label=f"L{layer}")
            plt.xlabel("epoch")
            plt.ylabel(field)
            plt.yscale("log")
            plt.legend(ncol=5, fontsize=7)
            save(name, title)

    epoch_rows = kinds["v4_epoch"]
    if epoch_rows:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        specs = (("epoch_seconds", "Epoch wall time"),
                 ("token_events_per_second", "Training throughput"),
                 ("train_phase_gpu_util", "Training GPU utilization"),
                 ("physical_writes", "Physical optimizer writes"))
        stages = sorted({r.get("v4_stage") for r in epoch_rows})
        for ax, (field, title) in zip(axes.flat, specs):
            for stage in stages:
                xy = sorted((int(r["epoch"]), float(r[field])) for r in epoch_rows
                            if r.get("v4_stage") == stage
                            and r.get(field) is not None and not r.get("partial"))
                if xy:
                    ax.plot(*zip(*xy), label=f"stage {stage}")
            ax.set(title=title, xlabel="epoch", ylabel=field)
            ax.legend(fontsize=8)
        save("runtime_by_epoch", "Runtime and update geometry by epoch")

    contracts = sorted(kinds["pipeline_v4_contract"],
                       key=lambda r: r.get("v4_stage", -1))
    if contracts:
        labels = [f"stage {r.get('v4_stage')}" for r in contracts]
        x = range(len(labels))
        plt.figure(figsize=(9, 5))
        plt.bar([i - .18 for i in x],
                [r.get("planned_block_cohort_writes_per_epoch", 0)
                 for r in contracts], .36, label="block×cohort writes")
        plt.bar([i + .18 for i in x], [r.get("cohorts", 0) for r in contracts],
                .36, label="cohorts")
        plt.xticks(list(x), labels)
        plt.ylabel("count per epoch")
        plt.legend()
        save("cohort_geometry",
             "Cohort/update geometry (loss is not emitted per cohort)")

    eval_specs = (
        ("student_trajectory_eval", ("CE_eval_loss", "KL_eval_loss"),
         "evaluation_losses", "Whole-corpus evaluation losses"),
        ("student_trajectory_eval",
         ("student_argmax_acceptance", "student_exact_seq_rate"),
         "evaluation_acceptance", "Student evaluation acceptance"),
        ("eval", ("overall_word_acc", "next_acc", "prev_acc", "cloze_acc"),
         "recall_evaluation", "In-training recall evaluation"),
        ("standard_eval", ("standard_macro_accuracy", "standard_epoch0_delta",
                           "standard_worst_delta"),
         "standard_evaluation", "Standard benchmark evaluation"),
    )
    for kind, fields, name, title in eval_specs:
        if not kinds[kind]:
            continue
        plt.figure(figsize=(10, 5))
        for field in fields:
            xy = sorted({int(r["epoch"]): float(r[field]) for r in kinds[kind]
                         if r.get(field) is not None}.items())
            if xy:
                plt.plot(*zip(*xy), marker=".", label=field)
        plt.xlabel("epoch")
        plt.legend()
        save(name, title)

    battery_fields = ("fixed_sequence_validation_seconds",
                      "recall_generation_seconds", "standard_scoring_seconds",
                      "total_boundary_seconds")
    if kinds["distributed_battery"]:
        plt.figure(figsize=(10, 5))
        for field in battery_fields:
            xy = sorted({int(r["epoch"]): float(r[field])
                         for r in kinds["distributed_battery"]
                         if r.get(field) is not None}.items())
            if xy:
                plt.plot(*zip(*xy), label=field)
        plt.xlabel("epoch")
        plt.ylabel("seconds")
        plt.legend()
        save("evaluation_runtime", "Evaluation runtime components")

    epochs = sorted({int(r["epoch"]) for r in rows if "epoch" in r})
    commits = sorted({r["source_commit"] for r in rows if r.get("source_commit")})
    launches = sorted({r["launch_id"] for r in rows if r.get("launch_id")})
    summary = {
        "run": str(run), "epochs": epochs, "source_commits": commits,
        "launch_ids": launches, "row_counts": {k: len(v) for k, v in kinds.items()},
        "figures": [f for _, f in figures], "cohort_loss_available": False,
        "cohort_loss_note": "Only epoch/layer GPU aggregates are emitted.",
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    cards = "\n".join(f"<h2>{html.escape(t)}</h2><img src='{html.escape(f)}'>"
                      for t, f in figures)
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><style>"
        "body{font:15px sans-serif;max-width:1200px;margin:auto}"
        "img{max-width:100%}</style>"
        f"<h1>{html.escape(run.name)}</h1><p>Epochs: {html.escape(str(epochs))}</p>"
        "<p>Loss is aggregated by epoch/layer. Per-cohort synchronous loss "
        "logging is deliberately absent; cohort geometry is shown instead.</p>"
        + cards)
    print(json.dumps(summary, indent=2))
    return 0 if figures else 2


if __name__ == "__main__":
    raise SystemExit(main())
