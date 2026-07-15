"""Generate the atomic v2-format report for one completed training.

The report is deliberately run-local: ``runs/<run>/report.md`` and
``runs/<run>/report.pdf`` plus PNG/CSV assets under
``runs/<run>/eval/report_v2``.  It reads only the run's frozen config and
append-only metrics.  Missing evidence is rendered as coverage, never used as
a reason to omit the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib import cm, colors

from report_pdf_v2 import write_individual_pdf

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
REPORT_INDEX = RUNS / "report_v2_index"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _completed_at(run_dir: Path) -> float | None:
    metrics = run_dir / "metrics.jsonl"
    if metrics.is_file():
        done = [float(row["t"]) for row in _rows(metrics)
                if row.get("kind") == "done" and row.get("t") is not None]
        if done:
            return max(done)
    pdf = run_dir / "report.pdf"
    return pdf.stat().st_mtime if pdf.is_file() else None


def refresh_report_index() -> int:
    """Publish stable completion-ordered links to every individual PDF."""
    REPORT_INDEX.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, Path] = {}
    for manifest_path in sorted(RUNS.glob("*/report_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        run_dir = manifest_path.parent
        pdf = run_dir / "report.pdf"
        completed_at = _completed_at(run_dir)
        if not manifest.get("complete") or not pdf.is_file() or completed_at is None:
            continue
        stamp = datetime.fromtimestamp(completed_at).strftime("%Y%m%d-%H%M%S")
        wanted[f"{stamp}__{run_dir.name}.pdf"] = pdf

    for old in REPORT_INDEX.glob("*.pdf"):
        if old.name not in wanted and old.is_symlink():
            old.unlink()
    for name, target in wanted.items():
        link = REPORT_INDEX / name
        relative = os.path.relpath(target, start=REPORT_INDEX)
        if link.is_symlink() and os.readlink(link) == relative:
            continue
        tmp = REPORT_INDEX / f".{name}.tmp"
        if os.path.lexists(tmp):
            tmp.unlink()
        tmp.symlink_to(relative)
        tmp.replace(link)
    return len(wanted)


def _finite_mean(values) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _realized_geometry(rows: list[dict], configured_b: int | None,
                       configured_k: int | str | None) -> dict:
    """Summarize the physical tiles that actually reached AdamW.

    Nominal B/K are ceilings: bucket tails, DataLoader tails, and ragged final
    token strips may all be smaller.  Keep both identities in every report so
    grouped science never silently treats a configured rectangle as realized.
    """
    updates = [row for row in rows if row.get("kind") == "train"
               and row.get("aligned_tokens_per_update") is not None]

    def stats(key: str) -> dict:
        values = [float(row[key]) for row in updates if row.get(key) is not None]
        if not values:
            return {"mean": None, "median": None, "min": None, "max": None}
        series = pd.Series(values)
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    realized = {
        "updates": len(updates),
        "lanes_per_update": stats("answer_visits_per_update"),
        "aligned_tokens_per_update": stats("aligned_tokens_per_update"),
    }
    if (updates and isinstance(configured_b, int) and configured_b > 0
            and isinstance(configured_k, int) and configured_k > 0):
        nominal_cells = configured_b * configured_k
        realized["nominal_cells"] = nominal_cells
        realized["full_nominal_tile_fraction"] = sum(
            int(row["aligned_tokens_per_update"]) == nominal_cells
            for row in updates
        ) / len(updates)
    return realized


def _loss_name(kind: str) -> str:
    return {
        "huber": "Huber robust loss against teacher hidden states",
        "cosine": "cosine-direction loss against teacher hidden states",
        "delta_cosine": "cosine loss on successive teacher block increments",
        "lens_kl": ("Kullback–Leibler divergence from teacher to student "
                    "through the frozen vocabulary head as a local metric"),
    }.get(kind, kind.replace("_", " "))


def _layer_loss_frame(rows: list[dict]) -> tuple[pd.DataFrame, str]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    measures = set()
    for row in rows:
        if row.get("kind") != "train" or not row.get("per_layer"):
            continue
        epoch = int(row.get("epoch", 0)) + 1
        grouped[epoch].append(row["per_layer"])
        measures.add(row.get("loss_measure", "historical_answer_mean"))
    out = []
    for epoch in sorted(grouped):
        n_layers = max(map(len, grouped[epoch]))
        for layer in range(n_layers):
            vals = [x[layer] for x in grouped[epoch]
                    if len(x) > layer and math.isfinite(float(x[layer]))]
            out.append({"epoch": epoch, "layer": layer + 1,
                        "loss": _finite_mean(vals), "n_rows": len(vals)})
    measure = ", ".join(sorted(measures)) if measures else "missing"
    return pd.DataFrame(out), measure


def _delta_frame(rows: list[dict]) -> tuple[pd.DataFrame, str]:
    out, representations = [], set()
    for row in rows:
        if row.get("kind") != "parameter_delta":
            continue
        representations.add(row.get("representation", "unknown"))
        absolute = row.get("per_layer_absolute_l2", [])
        relative = row.get("per_layer_relative_l2", [])
        counts = row.get("per_layer_parameter_count", [])
        for i, rel in enumerate(relative):
            out.append({
                "epoch": int(row["epoch"]), "layer": i + 1,
                "absolute_l2": absolute[i], "relative_l2": rel,
                "parameter_count": counts[i] if i < len(counts) else None,
            })
    return pd.DataFrame(out), ", ".join(sorted(representations)) or "missing"


def _v3_gradient_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for row in rows:
        if row.get("kind") != "v3_gradient_norm":
            continue
        for i, value in enumerate(row.get("per_layer_mean", [])):
            out.append({
                "epoch": int(row.get("epoch", 0)),
                "layer": i + 1,
                "gradient_l2": float(value),
            })
    return pd.DataFrame(out)


def _final_delta_csv(path: Path, epoch: int) -> tuple[pd.DataFrame, str]:
    """Adapt the historical final-only weight-delta CSV to report-v2.

    Campaign-2 retained only its final checkpoint, so this is a final profile,
    not an invented epoch timeline.  Module-relative deltas are aggregated as
    RMS per layer, matching ``per_layer_profile``.
    """
    if not path.is_file():
        return pd.DataFrame(), "missing"
    modules = pd.read_csv(path)
    required = {"layer", "rel_delta"}
    if modules.empty or not required.issubset(modules.columns):
        return pd.DataFrame(), "missing"
    profile = (modules.groupby("layer")["rel_delta"]
               .apply(lambda values: float((values.astype(float) ** 2).mean() ** .5)))
    frame = pd.DataFrame({
        "epoch": epoch,
        "layer": profile.index.astype(int),
        "relative_l2": profile.values,
        "absolute_l2": None,
        "parameter_count": None,
    })
    return (frame,
            "final checkpoint LoRA effective delta / base Frobenius; "
            "per-layer RMS over modules")


def _historical_standard(run_name: str, final_epoch: int) -> pd.DataFrame:
    """Load paired 100-item historical standard evaluations when present."""
    damage_root = RUNS / "standard_damage"
    checkpoint_path = damage_root / f"{run_name}.json"
    if not checkpoint_path.is_file():
        return pd.DataFrame()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    model = str(checkpoint.get("model", ""))
    base_name = "teacher_" + model.replace("/", "_") + ".json"
    base_path = damage_root / base_name
    if not base_path.is_file():
        return pd.DataFrame()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    common = sorted(set(base.get("tasks", {})) & set(checkpoint.get("tasks", {})))
    common = [task for task in common
              if base["tasks"][task].get("accuracy") is not None
              and checkpoint["tasks"][task].get("accuracy") is not None]
    if not common:
        return pd.DataFrame()

    def scores(payload: dict) -> dict[str, float]:
        return {task: float(payload["tasks"][task]["accuracy"])
                for task in common}

    base_scores, checkpoint_scores = scores(base), scores(checkpoint)
    deltas = {task: checkpoint_scores[task] - base_scores[task] for task in common}
    worst = min(deltas, key=deltas.get)
    base_macro = sum(base_scores.values()) / len(base_scores)
    checkpoint_macro = sum(checkpoint_scores.values()) / len(checkpoint_scores)
    rows = [
        {"epoch": 0, "macro_accuracy": base_macro, "epoch0_delta": 0.0,
         "worst_task": None, "worst_delta": 0.0,
         **{f"accuracy_{task}": value for task, value in base_scores.items()}},
        {"epoch": final_epoch, "macro_accuracy": checkpoint_macro,
         "epoch0_delta": checkpoint_macro - base_macro,
         "worst_task": worst, "worst_delta": deltas[worst],
         **{f"accuracy_{task}": value
            for task, value in checkpoint_scores.items()}},
    ]
    return pd.DataFrame(rows)


def _layer_assets(df: pd.DataFrame, value: str, ylabel: str, stem: str,
                  out_dir: Path, log_scale: bool) -> list[Path]:
    if df.empty:
        return []
    pivot = df.pivot(index="layer", columns="epoch", values=value).sort_index()
    paths = []
    norm = colors.Normalize(vmin=1, vmax=max(pivot.index))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if len(pivot.columns) == 1:
        ax.bar(pivot.index, pivot.iloc[:, 0],
               color=cm.viridis(norm(pivot.index.to_numpy())))
        ax.set_xlabel("layer")
        ax.set_title(f"Final per-layer {ylabel}")
    else:
        for layer, series in pivot.iterrows():
            ax.plot(pivot.columns, series, color=cm.viridis(norm(layer)),
                    lw=1.0, alpha=0.82)
        ax.set_xlabel("completed epoch")
        ax.set_title(f"Per-layer {ylabel} by epoch")
    if log_scale and (df[value] > 0).any():
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2, lw=.5)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                 label="layer", pad=.01)
    fig.tight_layout()
    line = out_dir / f"{stem}_temporal.png"
    fig.savefig(line, dpi=220)
    plt.close(fig)
    paths.append(line)

    vals = pivot.to_numpy()
    finite = vals[pd.notna(vals)]
    if finite.size:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        kwargs = {}
        positive = finite[finite > 0]
        if log_scale and positive.size:
            kwargs["norm"] = colors.LogNorm(
                vmin=max(float(positive.min()), 1e-12),
                vmax=max(float(positive.max()), float(positive.min()) * 1.0001))
        im = ax.imshow(vals, aspect="auto", cmap="viridis", **kwargs)
        ax.set_xlabel("completed epoch")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_title(f"{ylabel.capitalize()}: layer × epoch")
        fig.colorbar(im, ax=ax, label=ylabel)
        fig.tight_layout()
        heat = out_dir / f"{stem}_heatmap.png"
        fig.savefig(heat, dpi=220)
        plt.close(fig)
        paths.append(heat)

        final = pivot[pivot.columns[-1]].to_numpy()[None, :]
        finite_final = final[pd.notna(final)]
        fig, ax = plt.subplots(figsize=(9.0, 1.9))
        final_kwargs = {}
        positive_final = finite_final[finite_final > 0]
        if log_scale and positive_final.size:
            final_kwargs["norm"] = colors.LogNorm(
                vmin=max(float(positive_final.min()), 1e-12),
                vmax=max(float(positive_final.max()),
                         float(positive_final.min()) * 1.0001))
        im = ax.imshow(final, aspect="auto", cmap="viridis", **final_kwargs)
        ax.set_yticks([0], [f"epoch {pivot.columns[-1]}"])
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, len(pivot.index), max(1, len(pivot.index)//12)),
                      pivot.index[::max(1, len(pivot.index)//12)])
        ax.set_title(f"One-training density profile: {ylabel}")
        fig.colorbar(im, ax=ax, label=ylabel, pad=.02)
        fig.tight_layout()
        profile = out_dir / f"{stem}_one_row_density.png"
        fig.savefig(profile, dpi=220)
        plt.close(fig)
        paths.append(profile)
    return paths


def _eval_frames(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    recall, standard = [], []
    for row in rows:
        if row.get("kind") == "eval":
            structured = row.get("recall") or {}
            for corpus, scores in structured.items():
                recall.append({"epoch": int(row["epoch"]), "corpus": corpus,
                               **scores})
            # Campaign-2 metrics predate corpus-typed recall.  Preserve their
            # actual CER and line-exact measures instead of making a completed
            # historical run appear to have no recall evidence.  The generic
            # score column is line exactness, explicitly named by the corpus
            # label and retained beside CER in historical report tables.
            if not structured and row.get("cer") is not None:
                line_exact = row.get("line_exact")
                recall.append({
                    # Historical trainers logged a zero-based epoch index.
                    # Reports use completed-epoch counts throughout.
                    "epoch": int(row["epoch"]) + 1,
                    "corpus": "historical_inline_line_exact",
                    "next_acc": None, "prev_acc": None, "cloze_acc": None,
                    "overall_word_acc": line_exact,
                    "cer": row.get("cer"), "line_exact": line_exact,
                })
        elif row.get("kind") == "standard_eval":
            standard.append({
                "epoch": int(row["epoch"]),
                "macro_accuracy": row.get("standard_macro_accuracy"),
                "epoch0_delta": row.get("standard_epoch0_delta"),
                "worst_delta": row.get("standard_worst_delta"),
                "worst_task": row.get("standard_worst_task"),
                **{f"accuracy_{k}": v
                   for k, v in (row.get("standard_tasks") or {}).items()},
            })
    return pd.DataFrame(recall), pd.DataFrame(standard)


def _eval_assets(recall: pd.DataFrame, standard: pd.DataFrame,
                 out_dir: Path) -> list[Path]:
    paths = []
    if not recall.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        for corpus, group in recall.groupby("corpus"):
            ax.plot(group.epoch, group.overall_word_acc, marker="o", label=corpus)
        historical = "cer" in recall.columns and recall["cer"].notna().any()
        ax.set(xlabel="completed epoch",
               ylabel="line exactness" if historical else "overall word accuracy",
               title=("Historical inline recall trajectory" if historical
                      else "Recall by corpus, including epoch 0"))
        ax.grid(alpha=.2); ax.legend(frameon=False)
        fig.tight_layout(); path = out_dir / "recall_by_corpus.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    if not standard.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        ax.plot(standard.epoch, standard.macro_accuracy, marker="o", lw=2,
                label="macro accuracy")
        for column in sorted(c for c in standard.columns
                             if c.startswith("accuracy_")):
            ax.plot(standard.epoch, standard[column], marker="o", ls="--",
                    label=column.removeprefix("accuracy_"))
        ax.axhline(float(standard.iloc[0].macro_accuracy), color="grey",
                   ls="--", lw=.8, label="epoch 0")
        ax.set(xlabel="completed epoch", ylabel="accuracy",
               title="Standard-benchmark retention")
        ax.grid(alpha=.2); ax.legend(frameon=False)
        fig.tight_layout(); path = out_dir / "standard_damage.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    if not recall.empty and not standard.empty:
        avg = recall.groupby("epoch").overall_word_acc.mean()
        joined = standard.set_index("epoch").join(avg.rename("recall"), how="inner")
        if not joined.empty:
            base = float(standard.sort_values("epoch").iloc[0].macro_accuracy)
            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            damage = base - joined.macro_accuracy
            ax.plot(damage, joined.recall, marker="o")
            for epoch, x, y in zip(joined.index, damage, joined.recall):
                ax.annotate(f"e{epoch}", (x, y), fontsize=7)
            ax.set(xlabel="standard macro-accuracy damage vs epoch 0",
                   ylabel="mean recall word accuracy",
                   title="Recall–damage trajectory")
            ax.grid(alpha=.2); fig.tight_layout()
            path = out_dir / "recall_damage_frontier.png"
            fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    return paths


def _historical_phase_asset(loss: pd.DataFrame, recall: pd.DataFrame,
                            tail_blocks: int, out_dir: Path) -> Path | None:
    """Plot local stabilization, tail divergence, and behavioral onset."""
    if (loss.empty or recall.empty or tail_blocks <= 0
            or "cer" not in recall.columns or not recall["cer"].notna().any()):
        return None
    n_layers = int(loss.layer.max())
    tail_start = max(1, n_layers - tail_blocks + 1)
    last4_start = max(1, n_layers - 3)

    def grouped(label: str, selected: pd.Series) -> pd.DataFrame:
        frame = loss.loc[selected].groupby("epoch", as_index=False).loss.mean()
        frame["group"] = label
        return frame

    groups = pd.concat([
        grouped(f"body L1–L{tail_start - 1}", loss.layer < tail_start),
        grouped(f"tail L{tail_start}–L{n_layers}", loss.layer >= tail_start),
        grouped(f"last four L{last4_start}–L{n_layers}",
                loss.layer >= last4_start),
    ], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, frame in groups.groupby("group"):
        ax.plot(frame.epoch, frame.loss, lw=2, label=label)
    ax.set(xlabel="completed epoch", ylabel="mean logged hidden loss",
           title="Local stabilization precedes tail-window behavioral learning")
    ax.set_yscale("log")
    ax.grid(alpha=.2)
    right = ax.twinx()
    hist = recall.sort_values("epoch")
    right.plot(hist.epoch, hist.line_exact, color="black", marker="o",
               label="line exactness")
    right.plot(hist.epoch, hist.cer, color="tab:red", marker=".", ls="--",
               alpha=.75, label="CER")
    right.set_ylabel("behavioral score")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False,
              fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "historical_loss_recall_phases.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_Missing._"
    # pandas delegates ``to_markdown`` to optional ``tabulate``.  The lean
    # L40S runtime intentionally omits it, and a report must never make a
    # completed training look incomplete merely because a table formatter is
    # absent.
    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    selected = df.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _signal_frame(signal: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "layer": int(layer),
            "local_grad_norm": values.get("local_grad_norm", float("nan")),
            "foreign_grad_norm": values.get("max_foreign_grad_norm", float("nan")),
            "frozen_vocab_grad_norm": values.get("frozen_vocab_grad_norm", float("nan")),
        }
        for layer, values in (signal.get("per_block", {}) or {}).items()
    ]).sort_values("layer") if signal else pd.DataFrame()


def _signal_asset(frame: pd.DataFrame, out_dir: Path) -> Path | None:
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    ax.plot(frame.layer, frame.local_grad_norm, marker="o", ms=3,
            label="intended local block")
    ax.plot(frame.layer, frame.foreign_grad_norm, marker="o", ms=3,
            label="largest foreign block")
    ax.plot(frame.layer, frame.frozen_vocab_grad_norm, marker="o", ms=3,
            label="frozen vocabulary")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("layer")
    ax.set_ylabel("gradient L2 norm")
    ax.set_title("Per-layer training-signal attribution")
    ax.grid(alpha=.2)
    ax.legend()
    path = out_dir / "signal_attribution_by_layer.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def generate(run_dir: Path, allow_incomplete: bool = False) -> Path:
    # The CLI passes an absolute path, while refreshers often discover relative
    # ``runs/...`` paths.  Normalize once so artifact paths in the manifest are
    # independent of the caller.
    run_dir = run_dir.resolve()
    config_path, metrics_path = run_dir / "config.yaml", run_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("report v2 requires config.yaml and metrics.jsonl")
    rows = _rows(metrics_path)
    complete = (run_dir / "checkpoint").exists() and any(
        row.get("kind") == "done" for row in rows)
    if not complete and not allow_incomplete:
        raise RuntimeError("report v2 is generated only after training completes")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    train, model, data, mask = (cfg.get(k, {}) or {}
                                for k in ("train", "model", "data", "mask"))
    out_dir = run_dir / "eval" / "report_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    loss, loss_measure = _layer_loss_frame(rows)
    delta, delta_representation = _delta_frame(rows)
    gradients = _v3_gradient_frame(rows)
    recall, standard = _eval_frames(rows)
    final_epoch = max((int(row.get("epoch", -1)) + 1 for row in rows
                       if row.get("kind") == "train"), default=0)
    if delta.empty:
        delta, delta_representation = _final_delta_csv(
            run_dir / "eval" / "weight_deltas.csv", final_epoch)
    if standard.empty:
        standard = _historical_standard(run_dir.name, final_epoch)
    signal_path = run_dir / "eval" / "signal_attribution.json"
    signal = (json.loads(signal_path.read_text(encoding="utf-8"))
              if signal_path.exists() else {})
    signal_frame = _signal_frame(signal)
    if not loss.empty: loss.to_csv(out_dir / "layer_loss_by_epoch.csv", index=False)
    if not delta.empty: delta.to_csv(out_dir / "parameter_delta_by_epoch.csv", index=False)
    if not gradients.empty:
        gradients.to_csv(out_dir / "gradient_norm_by_epoch.csv", index=False)
    if not recall.empty: recall.to_csv(out_dir / "recall_by_epoch.csv", index=False)
    if not standard.empty: standard.to_csv(out_dir / "standard_by_epoch.csv", index=False)
    if not signal_frame.empty:
        signal_frame.to_csv(out_dir / "signal_attribution_by_layer.csv", index=False)
    _layer_assets(loss, "loss", "loss", "layer_loss", out_dir, log_scale=True)
    _layer_assets(delta, "relative_l2", "relative parameter delta",
                  "parameter_delta", out_dir, log_scale=True)
    _layer_assets(gradients, "gradient_l2", "local gradient L2 norm",
                  "gradient_norm", out_dir, log_scale=True)
    _eval_assets(recall, standard, out_dir)
    phase_path = _historical_phase_asset(
        loss, recall, int(train.get("tail_ce_blocks", 0) or 0), out_dir)
    _signal_asset(signal_frame, out_dir)

    examples_path = ROOT / str(data.get("examples_path", ""))
    missing = []
    for label, present in (
        ("completed checkpoint and done row", complete),
        ("epoch-zero recall", not recall.empty and 0 in set(recall.epoch)),
        ("epoch-zero standard benchmark", not standard.empty and 0 in set(standard.epoch)),
        ("per-layer loss", not loss.empty),
        ("epoch-zero parameter delta", not delta.empty and 0 in set(delta.epoch)),
        ("per-epoch parameter delta", not delta.empty and delta.epoch.nunique() > 1),
        ("signal attribution artifact", bool(signal)),
    ):
        if not present: missing.append(label)

    epochs = sorted(set(recall.get("epoch", pd.Series(dtype=int))).union(
        set(standard.get("epoch", pd.Series(dtype=int)))))
    max_items = max((int(r.get("items_seen", 0)) for r in rows), default=0)
    times = [float(r["t"]) for r in rows if "t" in r]
    elapsed_min = (max(times) - min(times)) / 60 if len(times) > 1 else float("nan")
    provenance = next((r for r in rows if r.get("source_commit")), {})
    partial_boundary = next(
        (r for r in reversed(rows)
         if r.get("kind") == "v3_partial_boundary"), None)
    if partial_boundary is None:
        partial_train = next(
            (r for r in reversed(rows)
             if r.get("kind") == "train" and r.get("partial")), None)
        if partial_train is not None:
            partial_boundary = {
                "token_events_seen": partial_train.get(
                    "token_events_seen", partial_train.get("step", 0)),
                "optimizer_updates_seen": partial_train.get(
                    "optimizer_updates_seen", 0),
                "completed_epochs": 0,
                "partial_epoch_index": int(partial_train.get("epoch", 0)) + 1,
                "meaning": (
                    "budget checkpoint inside a dataset traversal; not a "
                    "completed epoch"),
            }
    online_bk = (
        train.get("update_granularity") == "online"
        and train.get("pipeline_revision") == "3.1"
        and train.get("history_policy") in ("causal_bk", "causal_bk_probe")
    )
    if train.get("update_granularity") == "online":
        stale_k = train.get("stale_gradient_window", 1)
        stale_label = "all" if stale_k == 0 else stale_k
        gradient_law = (
            "no cross-token gradient aggregation"
            if stale_k == 1 else
            "unaveraged gradient sum at one explicitly stale weight snapshot")
        online_shape = (
            f"{train.get('micro_batch', 'missing')} fixed user lanes × "
            if online_bk else "one answer × "
        )
        activation_shard = (
            train.get("activation_shard_users", 0)
            or train.get("micro_batch", "missing"))
        activation_clause = (
            f"; transient activation shard {activation_shard} users"
            if online_bk else "")
        update_identity = (
            "`online`: " + online_shape
            + f"{stale_label} known-answer token(s) per weight snapshot; "
            f"`{train.get('online_optimizer', 'missing')}`; {gradient_law}; history "
            f"`{train.get('history_policy', 'missing')}`; backward dispatch "
            f"`{train.get('backward_dispatch', 'per_block')}`; write dispatch "
            f"`{train.get('online_write_dispatch', 'after_backward')}`"
            + activation_clause
        )
    elif train.get("update_granularity") == "grid":
        token_width = train.get("tokens_per_answer_update", "missing")
        token_width = "all" if token_width == 0 else token_width
        update_identity = (
            f"`grid`: {train.get('answers_per_update', 'missing')} answers × "
            f"{token_width} aligned tokens; reduction "
            f"`{train.get('update_reduction', 'missing')}`; layer order `forward`"
        )
    else:
        update_identity = (
            f"`{train.get('update_granularity', 'missing')}`; logged measure "
            f"`{loss_measure}`"
        )
    configured_b = ((train.get("micro_batch") if online_bk else 1)
                    if train.get("update_granularity") == "online" else
                    train.get("answers_per_update", train.get("micro_batch")))
    configured_k_raw = (train.get("stale_gradient_window", 1)
                        if train.get("update_granularity") == "online" else
                        train.get("tokens_per_answer_update", "all"))
    configured_k = "all" if configured_k_raw == 0 else configured_k_raw
    realized_geometry = _realized_geometry(rows, configured_b, configured_k_raw)
    lane_stats = realized_geometry["lanes_per_update"]
    token_stats = realized_geometry["aligned_tokens_per_update"]
    if train.get("update_granularity") == "online":
        token_events = max(
            (int(row.get("token_events_seen", 0)) for row in rows), default=0)
        writes = max(
            (int(row.get("optimizer_updates_seen", 0)) for row in rows),
            default=0)
        physical_writes = max(
            (int(row.get("physical_optimizer_updates_seen", 0))
             for row in rows), default=writes)
        realized_identity = (
            f"{token_events:,} aligned-token events; {writes:,} conceptual "
            f"block-local writes; {physical_writes:,} fused physical writes"
        )
        layer_count = max(
            (len(row.get("per_layer", [])) for row in rows), default=0)
        physical_tiles = (
            physical_writes / layer_count if layer_count else 0)
        cohort_users = [
            int(row["users"]) for row in rows
            if row.get("kind") == "v31_cohort" and row.get("users")
        ]
        realized_geometry = {
            "updates": writes,
            "physical_updates": physical_writes,
            "token_events": token_events,
            "physical_tiles": physical_tiles,
            "lanes_per_update": ({
                "mean": sum(cohort_users) / len(cohort_users),
                "median": statistics.median(cohort_users),
                "min": min(cohort_users), "max": max(cohort_users),
            } if cohort_users else {
                "mean": configured_b, "median": configured_b,
                "min": configured_b, "max": configured_b,
            }),
            "aligned_tokens_per_update": {
                "mean": (token_events / physical_tiles
                         if physical_tiles else 0),
                "median": configured_b * configured_k
                    if online_bk and isinstance(configured_k, int) else 1,
                "min": 1,
                "max": configured_b * configured_k
                    if online_bk and isinstance(configured_k, int) else 1,
            },
        }
    elif (realized_geometry["updates"] and lane_stats.get("mean") is not None
            and token_stats.get("mean") is not None):
        realized_identity = (
            f"{realized_geometry['updates']:,} updates; lanes/update mean "
            f"{lane_stats['mean']:.2f}, median {lane_stats['median']:.2f}; "
            f"aligned tokens/update mean {token_stats['mean']:.2f}, median "
            f"{token_stats['median']:.2f}"
        )
    else:
        realized_identity = "missing"
    configured_run_class = train.get("run_class", "missing")
    reported_run_class = configured_run_class
    run_class_source = "frozen_config"
    if run_dir.name.startswith("pareto_v2_screen_"):
        # The first broad controls predate explicit run_class pinning.  Their
        # immutable run configs say method through inheritance, but the launch
        # ledger and run naming typed this cohort as controls from inception.
        reported_run_class = "control"
        run_class_source = "2026-07-14_screen_name_fallback"
    rel = lambda name: f"eval/report_v2/{name}"
    tail_blocks = int(train.get("tail_ce_blocks", 0) or 0)
    tail_weight = float(train.get("tail_ce_weight", 0) or 0)
    readout_blocks = int(train.get("readout_window_blocks", 0) or 0)
    readout_weight = float(train.get("readout_weight", 0) or 0)
    if tail_blocks and tail_weight:
        final_logit = (f"ACTIVE historical tail objective: {tail_blocks} blocks, "
                       f"weight {tail_weight:g}, target "
                       f"`{train.get('tail_ce_kind', 'legacy/unspecified')}`")
    elif readout_blocks and readout_weight:
        final_logit = (f"ACTIVE historical readout: {readout_blocks} blocks, "
                       f"weight {readout_weight:g}, source "
                       f"`{train.get('readout_source', 'legacy/unspecified')}`")
    else:
        final_logit = "disabled"
    recall_columns = (["epoch", "corpus", "cer", "line_exact"]
                      if "cer" in recall.columns and recall["cer"].notna().any()
                      else ["epoch", "corpus", "next_acc", "prev_acc",
                            "cloze_acc", "overall_word_acc"])
    standard_columns = ["epoch", "macro_accuracy", "epoch0_delta",
                        *sorted(c for c in standard.columns
                                if c.startswith("accuracy_")),
                        "worst_task", "worst_delta"]
    top_delta = (delta.sort_values("relative_l2", ascending=False)
                 .loc[:, ["layer", "relative_l2"]].head(12)
                 if not delta.empty else pd.DataFrame())
    md = [
        f"# Individual training report v2 — {run_dir.name}", "",
        "## Identity and provenance", "",
        f"- Status: {'complete' if complete else 'incomplete diagnostic rendering'}",
        f"- Model/base: `{model.get('name', 'missing')}`",
        f"- Dataset: `{data.get('examples_path', 'missing')}`"
        + (f"; SHA-256 `{_sha(examples_path)}`" if examples_path.is_file() else " (missing)"),
        f"- Frozen run config SHA-256: `{_sha(config_path)}`",
        f"- Training source commit: `{provenance.get('source_commit', 'missing')}`",
        ("- Runtime tree: DIRTY diagnostic; diff SHA-256 "
         f"`{provenance.get('runtime_diff_sha256', 'missing')}`"
         if provenance.get("runtime_dirty") is True else
         "- Runtime tree: clean at the recorded source commit"
         if provenance.get("runtime_dirty") is False else
         "- Runtime tree: cleanliness unavailable (legacy provenance)"),
        f"- Student initialization: `{provenance.get('student_init_identity', train.get('init_from') or model.get('name', 'missing'))}`",
        f"- Pipeline: v{train.get('pipeline_revision') or train.get('pipeline_version', 'missing')}",
        f"- Censorship: `{mask.get('mode', 'missing')} × {mask.get('compaction', 'missing')}`",
        f"- Loss: {_loss_name(str(train.get('hidden_loss', 'missing')))}",
        f"- Run class: `{reported_run_class}` (source `{run_class_source}`; "
        f"frozen-config value `{configured_run_class}`)",
        f"- Update geometry/aggregation: {update_identity}",
        f"- Realized update geometry: {realized_identity}",
        (f"- Training extent: partial budget boundary at "
         f"{partial_boundary.get('token_events_seen', 0):,} token events; "
         "zero complete dataset epochs. Numeric boundary plots are probe "
         "coordinates, not an epoch claim."
         if partial_boundary else
         f"- Training extent: {final_epoch} complete dataset epoch(s)."),
        f"- State / attention / expert routing: `{train.get('trajectory_source', 'missing')}` / "
        f"`{train.get('attention_source', 'missing')}` / `{train.get('expert_routing_source', 'missing')}`",
        f"- Optimizer / LR rule / history: `{train.get('online_optimizer', 'adamw')}` / "
        f"`{train.get('lr_rule', 'fixed')}` / `{train.get('history_policy', 'not_applicable')}`",
        f"- Backward / write dispatch: `{train.get('backward_dispatch', 'per_block')}` / "
        f"`{train.get('online_write_dispatch', 'after_backward')}`; stale-gradient "
        f"window `{train.get('stale_gradient_window', 1)}`",
        f"- Batching: `{train.get('batching', 'missing')}`, micro-batch {train.get('micro_batch', 'missing')}, "
        f"gradient accumulation {train.get('grad_accum', 'missing')}; transient "
        f"activation shard {train.get('activation_shard_users', 0) or train.get('micro_batch', 'missing')} users",
        f"- Connected hidden window: width {train.get('conn_window', 0)}, stride {train.get('conn_stride', 0)}; "
        f"final-logit training: {final_logit}",
        f"- Seed: {train.get('seed', 'missing')}; items observed: {max_items:,}; elapsed telemetry span: {elapsed_min:.1f} min",
        f"- Parameter-change representation: `{delta_representation}`",
        "", "## Recall by corpus", "",
        "Epoch 0 is the pre-training student; positive epochs are completed training epochs.", "",
        _markdown_table(recall, recall_columns)
        if not recall.empty else "_Missing._",
        "", f"![Recall by corpus]({rel('recall_by_corpus.png')})" if not recall.empty else "",
        "", "## Standard-benchmark damage", "",
        _markdown_table(standard, standard_columns)
        if not standard.empty else "_Missing._",
        "", f"![Standard damage]({rel('standard_damage.png')})" if not standard.empty else "",
        "", f"![Recall–damage trajectory]({rel('recall_damage_frontier.png')})"
        if not recall.empty and not standard.empty else "",
        "", "## Per-layer loss by epoch", "",
        ("Historical tail hidden losses are diagnostic: their configured "
         f"training weight is `{train.get('tail_hidden_weight', 'missing')}`. "
         "Their divergence can therefore measure task-window specialization, "
         "not failure to minimize an active hidden objective."
         if phase_path else ""),
        "", (f"![Loss/recall phases]({rel('historical_loss_recall_phases.png')})"
             if phase_path else ""),
        "",
        f"Loss measure selected by the optimizer regime: `{loss_measure}`.", "",
        f"![Temporal layer loss]({rel('layer_loss_temporal.png')})" if not loss.empty else "_Missing._",
        "", f"![Layer-loss heatmap]({rel('layer_loss_heatmap.png')})" if not loss.empty else "",
        "", f"![One-row layer-loss density]({rel('layer_loss_one_row_density.png')})" if not loss.empty else "",
        "", "## Per-layer immediate-gradient norm", "",
        ("Pipeline-v3 records the mean norm of each state-free local write "
         "without synchronizing inside the token/block hot loop."
         if not gradients.empty else "_Not applicable or missing._"),
        "", (f"![Temporal local gradient norm]({rel('gradient_norm_temporal.png')})"
             if not gradients.empty else ""),
        "", (f"![Local-gradient heatmap]({rel('gradient_norm_heatmap.png')})"
             if not gradients.empty else ""),
        "", "## Per-layer parameter modification", "",
        f"Representation: `{delta_representation}`.", "",
        "Most-modified layers (final checkpoint):", "",
        _markdown_table(top_delta, ["layer", "relative_l2"])
        if not top_delta.empty else "_Missing._",
        "",
        f"![Temporal parameter delta]({rel('parameter_delta_temporal.png')})" if not delta.empty else "_Missing._",
        "", f"![Parameter-delta heatmap]({rel('parameter_delta_heatmap.png')})" if not delta.empty else "",
        "", f"![One-row parameter density]({rel('parameter_delta_one_row_density.png')})" if not delta.empty else "",
        "", "## Training-signal attribution", "",
        (f"Across {signal.get('items', 'missing')} sampled training items, "
         f"local gradient L2 norm is "
         f"{float(signal.get('local_grad_norm', float('nan'))):.4g}; "
         f"cross-block leakage is "
         f"{float(signal.get('cross_block_leak_grad_norm', float('nan'))):.4g}; "
         f"frozen-vocabulary leakage is "
         f"{float(signal.get('frozen_vocab_grad_norm', float('nan'))):.4g}."
         if signal else "_Missing._"),
        (f"Strict-local certification: **{'PASS' if signal.get('passed') else 'FAIL'}**. "
         "No behavioral readout or final-logit objective is present."
         if signal else ""),
        (f"Teacher target source: `{signal.get('teacher_target_source', 'missing')}`; "
         f"cache hash `{signal.get('teacher_cache_hash', 'missing')}`."
         if signal else ""),
        "", (f"![Per-layer signal attribution]({rel('signal_attribution_by_layer.png')})"
             if not signal_frame.empty else ""),
        "", ("Exact per-layer values: "
             "[`signal_attribution_by_layer.csv`](eval/report_v2/signal_attribution_by_layer.csv); "
             "source artifact: [`signal_attribution.json`](eval/signal_attribution.json)."
             if signal else ""),
        "", "## Coverage and missing evidence", "",
    ]
    md.extend([f"- Missing: {item}" for item in missing] or
              ["- All mandatory epoch telemetry is present."])
    md += [
        "", "## Artifact index", "",
        "- Printable individual report: [`report.pdf`](report.pdf)",
        f"- Raw metrics: [`metrics.jsonl`](metrics.jsonl)",
        f"- Frozen config: [`config.yaml`](config.yaml)",
        f"- Report assets: [`eval/report_v2/`](eval/report_v2/)",
        "- Future collective reports select this run-local evidence by typed identity; "
        "they do not reconstruct a second source of truth.", "",
    ]
    report = run_dir / "report.md"
    report_tmp = run_dir / ".report.md.tmp"
    report_tmp.write_text("\n".join(x for x in md if x is not None), encoding="utf-8")
    report_tmp.replace(report)
    figures = [
        ("Recall by corpus", out_dir / "recall_by_corpus.png"),
        ("Standard-benchmark retention", out_dir / "standard_damage.png"),
        ("Recall–damage trajectory", out_dir / "recall_damage_frontier.png"),
        ("Per-layer loss by epoch", out_dir / "layer_loss_temporal.png"),
        ("Per-layer loss heatmap", out_dir / "layer_loss_heatmap.png"),
        ("One-row per-layer loss density", out_dir / "layer_loss_one_row_density.png"),
        ("Per-layer parameter modification by epoch",
         out_dir / "parameter_delta_temporal.png"),
        ("Per-layer parameter-modification heatmap",
         out_dir / "parameter_delta_heatmap.png"),
        ("One-row parameter-modification density",
         out_dir / "parameter_delta_one_row_density.png"),
        ("Per-layer training-signal attribution",
         out_dir / "signal_attribution_by_layer.png"),
    ]
    if not gradients.empty:
        figures.extend([
            ("Per-layer immediate-gradient norm by epoch",
             out_dir / "gradient_norm_temporal.png"),
            ("Per-layer immediate-gradient heatmap",
             out_dir / "gradient_norm_heatmap.png"),
        ])
    if phase_path:
        figures.insert(3, ("Local stabilization and behavioral onset",
                           phase_path))
    pdf = write_individual_pdf(
        run_dir / "report.pdf",
        title=f"Individual training report v2 — {run_dir.name}",
        identity=[
            f"Status: {'complete' if complete else 'incomplete diagnostic rendering'}",
            f"Model/base: {model.get('name', 'missing')}",
            f"Dataset: {data.get('examples_path', 'missing')}",
            f"Pipeline: v{train.get('pipeline_revision') or train.get('pipeline_version', 'missing')}",
            f"Censorship: {mask.get('mode', 'missing')} × {mask.get('compaction', 'missing')}",
            f"Loss: {_loss_name(str(train.get('hidden_loss', 'missing')))}",
            (f"Run class: {reported_run_class} (source {run_class_source}; "
             f"frozen-config value {configured_run_class})."),
            f"Update geometry/aggregation: {update_identity}",
            f"Realized update geometry: {realized_identity}",
            ("Optimizer / LR rule / history: "
             f"{train.get('online_optimizer', 'adamw')} / "
             f"{train.get('lr_rule', 'fixed')} / "
             f"{train.get('history_policy', 'not_applicable')}."),
            ("Batching: "
             f"{train.get('batching', 'missing')}; micro-batch "
             f"{train.get('micro_batch', 'missing')}; gradient accumulation "
             f"{train.get('grad_accum', 'missing')}."),
            ("Connected hidden window: width "
             f"{train.get('conn_window', 0)}, stride {train.get('conn_stride', 0)}; "
             f"final-logit training {final_logit}."),
            (f"Items observed: {max_items:,}; elapsed telemetry span: "
             f"{elapsed_min:.1f} min."),
            ("Strict-local certification: "
             f"{'PASS' if signal.get('passed') else 'MISSING OR FAIL'}; "
             "the frozen vocabulary and cross-block gradients are audited."),
        ],
        recall=recall,
        standard=standard,
        delta=top_delta,
        coverage=[
            f"Optimizer loss measure: {loss_measure}",
            f"Parameter-change representation: {delta_representation}",
            (f"Teacher target source: {signal.get('teacher_target_source', 'missing')}; "
             f"cache hash: {signal.get('teacher_cache_hash', 'missing')}"),
            "Missing evidence:" if missing else "All mandatory epoch telemetry is present.",
            *[f"- {item}" for item in missing],
            ("Source artifacts: report.md, config.yaml, metrics.jsonl, "
             "eval/report_v2/, and eval/signal_attribution.json when present."),
        ],
        figures=figures,
    )
    manifest = {
        "schema_version": 3,
        "run": run_dir.name,
        "campaign": (
            "pareto_v3" if run_dir.name.startswith("pareto_v3_") else
            "pareto_v2" if run_dir.name.startswith("pareto_v2_") else None),
        "complete": complete,
        "report": str(report.relative_to(ROOT)),
        "pdf": str(pdf.relative_to(ROOT)),
        "model": model.get("name"),
        "dataset": data.get("examples_path"),
        "pipeline_version": train.get("pipeline_version"),
        "pipeline_revision": train.get("pipeline_revision"),
        "run_class": reported_run_class,
        "configured_run_class": configured_run_class,
        "run_class_source": run_class_source,
        "censorship": mask.get("compaction"),
        "hidden_loss": train.get("hidden_loss"),
        "batching": train.get("batching"),
        "geometry": {
            "answers": configured_b,
            "tokens": configured_k,
            "reduction": train.get("update_reduction", train.get("update_granularity")),
            "realized": realized_geometry,
        },
        "optimizer": train.get("online_optimizer", "adamw"),
        "lr_rule": train.get("lr_rule"),
        "history_policy": train.get("history_policy"),
        "backward_dispatch": train.get("backward_dispatch", "per_block"),
        "online_write_dispatch": train.get(
            "online_write_dispatch", "after_backward"),
        "stale_gradient_window": train.get("stale_gradient_window", 1),
        "activation_shard_users": (
            train.get("activation_shard_users", 0)
            or train.get("micro_batch")),
        "trajectory_source": train.get("trajectory_source"),
        "partial_training_boundary": partial_boundary,
        "strict_local": bool(signal.get("passed")) if signal else False,
        "source_commit": provenance.get("source_commit"),
        "runtime_dirty": provenance.get("runtime_dirty"),
        "runtime_diff_sha256": provenance.get("runtime_diff_sha256"),
        "missing": missing,
    }
    manifest_path = run_dir / "report_manifest.json"
    manifest_tmp = run_dir / ".report_manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    refresh_report_index()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name or runs/<run> path")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="diagnostic rendering; marks report incomplete")
    args = ap.parse_args()
    p = Path(args.run)
    run_dir = p if p.is_dir() else RUNS / args.run
    print(generate(run_dir, allow_incomplete=args.allow_incomplete))


if __name__ == "__main__":
    main()
