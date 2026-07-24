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
    import numpy as np
    from matplotlib.colors import LogNorm

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

    def layer_data(kind, field, *, vector=False):
        data = defaultdict(dict)
        for row in kinds[kind]:
            if row.get("partial"):
                continue
            values = row.get(field) or ([] if vector else {})
            cells = enumerate(values, 1) if vector else values.items()
            for layer, value in cells:
                value = float(value)
                if value > 0:
                    data[int(layer)][int(row["epoch"])] = value
        return data

    def ordered_layers(data):
        """Descending value at the first available step, then layer number."""
        first = {
            layer: values[min(values)]
            for layer, values in data.items() if values
        }
        return sorted(first, key=lambda layer: (-first[layer], layer))

    def layer_series(data, field, name, title):
        if not data:
            return
        order = ordered_layers(data)
        colors = plt.colormaps["turbo"](
            np.linspace(.03, .97, max(len(order), 1)))
        plt.figure(figsize=(12, 6))
        for color, layer in zip(colors, order):
            values = data[layer]
            xy = sorted(values.items())
            plt.plot(*zip(*xy), linewidth=1.25, color=color,
                     label=f"L{layer}")
        plt.xlabel("epoch")
        plt.ylabel(field)
        plt.yscale("log")
        plt.legend(ncol=5, fontsize=7, title="descending at first step",
                   title_fontsize=7)
        save(name, title)

    def layer_density(data, field, name, title):
        if not data:
            return
        layers = list(range(min(data), max(data) + 1))
        epochs = sorted({epoch for values in data.values() for epoch in values})
        matrix = np.full((len(layers), len(epochs)), np.nan)
        for yi, layer in enumerate(layers):
            for xi, epoch in enumerate(epochs):
                if epoch in data.get(layer, {}):
                    matrix[yi, xi] = data[layer][epoch]
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        if not positive.size:
            return
        plt.figure(figsize=(12, 6))
        image = plt.imshow(
            matrix, origin="lower", aspect="auto", interpolation="nearest",
            extent=(epochs[0] - .5, epochs[-1] + .5,
                    layers[0] - .5, layers[-1] + .5),
            cmap="magma", norm=LogNorm(vmin=positive.min(),
                                       vmax=positive.max()))
        plt.colorbar(image, label=field)
        plt.xlabel("epoch")
        plt.ylabel("layer")
        plt.yticks(layers)
        save(name, title)

    loss_data = layer_data("v4_epoch", "layer_losses")
    grad_data = layer_data("v4_gradient_norm", "grad_norms")
    layer_series(loss_data, "layer_losses", "loss_by_epoch_layer",
                 "Block-local loss by epoch and layer")
    layer_density(loss_data, "layer_losses", "loss_density_layer_epoch",
                  "Block-local loss density: layer × epoch")
    layer_series(grad_data, "grad_norms", "gradient_by_epoch_layer",
                 "Gradient norm by epoch and layer")
    layer_density(grad_data, "grad_norms", "gradient_density_layer_epoch",
                  "Gradient-norm density: layer × epoch")

    for field, name, title in (
        ("per_layer_absolute_l2", "weight_delta_absolute",
         "Absolute effective LoRA weight delta"),
        ("per_layer_relative_l2", "weight_delta_relative",
         "Relative effective LoRA weight delta"),
    ):
        data = layer_data("parameter_delta", field, vector=True)
        if data:
            order = ordered_layers(data)
            colors = plt.colormaps["turbo"](
                np.linspace(.03, .97, max(len(order), 1)))
            plt.figure(figsize=(12, 6))
            for color, layer in zip(colors, order):
                values = data[layer]
                xy = sorted(values.items())
                plt.plot(*zip(*xy), linewidth=1.25, color=color,
                         label=f"L{layer}")
            plt.xlabel("epoch")
            plt.ylabel(field)
            plt.yscale("log")
            plt.legend(ncol=5, fontsize=7,
                       title="descending at first non-zero step",
                       title_fontsize=7)
            save(name, title)
            if field == "per_layer_relative_l2":
                layer_density(
                    data, field, "weight_delta_relative_density",
                    "Relative effective LoRA weight-delta density: layer × epoch")

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

    def latest(kind, field):
        candidates = [r for r in kinds[kind] if r.get(field) is not None]
        if not candidates:
            return None
        row = max(candidates, key=lambda r: (int(r.get("epoch", -1)),
                                             float(r.get("t", 0))))
        return row[field]

    first = contracts[0] if contracts else rows[0]
    latest_epoch_rows = [
        r for r in epoch_rows
        if int(r.get("epoch", -1)) == max(
            (int(x.get("epoch", -1)) for x in epoch_rows), default=-1)
        and not r.get("partial")]
    epoch_seconds = [float(r["epoch_seconds"]) for r in latest_epoch_rows
                     if r.get("epoch_seconds") is not None]
    runtime_dirty = sorted({str(r.get("runtime_dirty")) for r in rows
                            if "runtime_dirty" in r})
    stage_splits = first.get("v4_stage_splits") or []
    fundamental = [
        ("Run", run.name),
        ("Model identity", str(first.get("model_base_identity", "unknown"))),
        ("Student initialization",
         str(first.get("student_init_identity", "unknown"))),
        ("Source commit", ", ".join(c[:12] for c in commits)),
        ("Launch id", ", ".join(launches)),
        ("Runtime dirty", ", ".join(runtime_dirty)),
        ("Pipeline", f"v{first.get('pipeline_revision', 'unknown')} / "
                     f"{len(stage_splits) + 1} stages / splits {stage_splits}"),
        ("Training objective", str(first.get("hidden_loss",
                                             first.get("loss_kind", "unknown")))),
        ("Loss kind", str(first.get("loss_kind", "unknown"))),
        ("Micro-batch / cohorts",
         f"{first.get('micro_batch', 'unknown')} / "
         f"{first.get('cohorts', 'unknown')} "
         f"(width {first.get('cohort_width_min', '?')}–"
         f"{first.get('cohort_width_max', '?')})"),
        ("Dataset", f"{first.get('dataset_items', '?')} items; "
                    f"sha {str(first.get('dataset_sha256', 'unknown'))[:12]}"),
        ("Epochs represented",
         f"{epochs[0] if epochs else '?'}–{epochs[-1] if epochs else '?'}"),
        ("Latest stage epoch time",
         (f"{min(epoch_seconds):.2f}–{max(epoch_seconds):.2f} s"
          if epoch_seconds else "not emitted")),
        ("Latest CE / KL evaluation loss",
         f"{latest('student_trajectory_eval', 'CE_eval_loss')!s} / "
         f"{latest('student_trajectory_eval', 'KL_eval_loss')!s}"),
        ("Latest student argmax / exact-sequence",
         f"{latest('student_trajectory_eval', 'student_argmax_acceptance')!s} / "
         f"{latest('student_trajectory_eval', 'student_exact_seq_rate')!s}"),
        ("Latest recall word accuracy",
         str(latest("eval", "overall_word_acc"))),
        ("Latest standard macro accuracy",
         str(latest("standard_eval", "standard_macro_accuracy"))),
        ("Evaluation coverage",
         f"{latest('student_trajectory_eval', 'dataset_item_count')} items / "
         f"{latest('student_trajectory_eval', 'answer_token_count')} tokens"),
    ]
    summary = {
        "run": str(run), "epochs": epochs, "source_commits": commits,
        "launch_ids": launches, "row_counts": {k: len(v) for k, v in kinds.items()},
        "figures": [f for _, f in figures], "cohort_loss_available": False,
        "cohort_loss_note": "Only epoch/layer GPU aggregates are emitted.",
        "fundamental_data": dict(fundamental),
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    cards = "\n".join(f"<h2>{html.escape(t)}</h2><img src='{html.escape(f)}'>"
                      for t, f in figures)
    table_html = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in fundamental)
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><style>"
        "body{font:15px sans-serif;max-width:1200px;margin:auto}"
        "img{max-width:100%}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:6px;text-align:left}"
        "th{width:30%;background:#eee}</style>"
        f"<h1>{html.escape(run.name)}</h1><table>{table_html}</table>"
        "<p>Loss is aggregated by epoch/layer. Per-cohort synchronous loss "
        "logging is deliberately absent; cohort geometry is shown instead.</p>"
        + cards)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out / "report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        fig.suptitle(f"Pipeline-v4.6 report — {run.name}",
                     fontsize=18, fontweight="bold", y=.97)
        table = ax.table(
            cellText=[[key, str(value)] for key, value in fundamental],
            colLabels=["Fundamental datum", "Value"], colWidths=[.3, .68],
            cellLoc="left", colLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.28)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#dce6f1")
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f4f4f4")
        fig.text(.5, .035,
                 "Read-only rendering of in-training telemetry; no new "
                 "evaluation was performed.", ha="center", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        for title, filename in figures:
            image = plt.imread(out / filename)
            height, width = image.shape[:2]
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_axes((.03, .03, .94, .9))
            ax.imshow(image)
            ax.axis("off")
            fig.suptitle(title, fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0 if figures else 2


if __name__ == "__main__":
    raise SystemExit(main())
