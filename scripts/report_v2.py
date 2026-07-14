"""Generate the pipeline-v2 report for one completed training.

The report is deliberately run-local: ``runs/<run>/report.md`` plus PNG/CSV
assets under ``runs/<run>/eval/report_v2``.  It reads only the run's frozen
config and append-only metrics.  Missing evidence is rendered as coverage,
never used as a reason to omit the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib import cm, colors

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


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


def _finite_mean(values) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _loss_name(kind: str) -> str:
    return {
        "huber": "Huber robust loss against teacher hidden states",
        "lens_kl": ("Kullback–Leibler divergence from teacher to student "
                    "through the frozen vocabulary head"),
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


def _layer_assets(df: pd.DataFrame, value: str, ylabel: str, stem: str,
                  out_dir: Path, log_scale: bool) -> list[Path]:
    if df.empty:
        return []
    pivot = df.pivot(index="layer", columns="epoch", values=value).sort_index()
    paths = []
    norm = colors.Normalize(vmin=1, vmax=max(pivot.index))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for layer, series in pivot.iterrows():
        ax.plot(pivot.columns, series, color=cm.viridis(norm(layer)),
                lw=1.0, alpha=0.82)
    if log_scale and (df[value] > 0).any():
        ax.set_yscale("log")
    ax.set_xlabel("completed epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-layer {ylabel} by epoch")
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
            for corpus, scores in (row.get("recall") or {}).items():
                recall.append({"epoch": int(row["epoch"]), "corpus": corpus,
                               **scores})
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
        ax.set(xlabel="completed epoch", ylabel="overall word accuracy",
               title="Recall by corpus, including epoch 0")
        ax.grid(alpha=.2); ax.legend(frameon=False)
        fig.tight_layout(); path = out_dir / "recall_by_corpus.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    if not standard.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        ax.plot(standard.epoch, standard.macro_accuracy, marker="o",
                label="standard macro accuracy")
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


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_Missing._"
    return df[columns].to_markdown(index=False, floatfmt=".4f")


def generate(run_dir: Path, allow_incomplete: bool = False) -> Path:
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
    recall, standard = _eval_frames(rows)
    if not loss.empty: loss.to_csv(out_dir / "layer_loss_by_epoch.csv", index=False)
    if not delta.empty: delta.to_csv(out_dir / "parameter_delta_by_epoch.csv", index=False)
    if not recall.empty: recall.to_csv(out_dir / "recall_by_epoch.csv", index=False)
    if not standard.empty: standard.to_csv(out_dir / "standard_by_epoch.csv", index=False)
    _layer_assets(loss, "loss", "loss", "layer_loss", out_dir, log_scale=True)
    _layer_assets(delta, "relative_l2", "relative parameter delta",
                  "parameter_delta", out_dir, log_scale=True)
    _eval_assets(recall, standard, out_dir)

    examples_path = ROOT / str(data.get("examples_path", ""))
    missing = []
    for label, present in (
        ("completed checkpoint and done row", complete),
        ("epoch-zero recall", not recall.empty and 0 in set(recall.epoch)),
        ("epoch-zero standard benchmark", not standard.empty and 0 in set(standard.epoch)),
        ("per-layer loss", not loss.empty),
        ("epoch-zero parameter delta", not delta.empty and 0 in set(delta.epoch)),
        ("per-epoch parameter delta", not delta.empty and delta.epoch.max() > 0),
        ("signal attribution artifact", (run_dir / "eval/signal_attribution.json").exists()),
    ):
        if not present: missing.append(label)

    epochs = sorted(set(recall.get("epoch", pd.Series(dtype=int))).union(
        set(standard.get("epoch", pd.Series(dtype=int)))))
    max_items = max((int(r.get("items_seen", 0)) for r in rows), default=0)
    times = [float(r["t"]) for r in rows if "t" in r]
    elapsed_min = (max(times) - min(times)) / 60 if len(times) > 1 else float("nan")
    provenance = next((r for r in rows if r.get("source_commit")), {})
    rel = lambda name: f"eval/report_v2/{name}"
    md = [
        f"# Individual training report v2 — {run_dir.name}", "",
        "## Identity and provenance", "",
        f"- Status: {'complete' if complete else 'incomplete diagnostic rendering'}",
        f"- Model/base: `{model.get('name', 'missing')}`",
        f"- Dataset: `{data.get('examples_path', 'missing')}`"
        + (f"; SHA-256 `{_sha(examples_path)}`" if examples_path.is_file() else " (missing)"),
        f"- Frozen run config SHA-256: `{_sha(config_path)}`",
        f"- Training source commit: `{provenance.get('source_commit', 'missing')}`",
        f"- Student initialization: `{provenance.get('student_init_identity', train.get('init_from') or model.get('name', 'missing'))}`",
        f"- Pipeline: v{train.get('pipeline_version', 'missing')}",
        f"- Censorship: `{mask.get('mode', 'missing')} × {mask.get('compaction', 'missing')}`",
        f"- Loss: {_loss_name(str(train.get('hidden_loss', 'missing')))}",
        f"- Update aggregation: `{train.get('update_granularity', 'missing')}`; logged measure `{loss_measure}`",
        f"- State / attention / expert routing: `{train.get('trajectory_source', 'missing')}` / "
        f"`{train.get('attention_source', 'missing')}` / `{train.get('expert_routing_source', 'missing')}`",
        f"- Batching: `{train.get('batching', 'missing')}`, micro-batch {train.get('micro_batch', 'missing')}, "
        f"gradient accumulation {train.get('grad_accum', 'missing')}",
        f"- Connected window: width {train.get('conn_window', 0)}, stride {train.get('conn_stride', 0)}; "
        f"teacher-sourced readout `{train.get('readout_source', 'UNSET')}`",
        f"- Seed: {train.get('seed', 'missing')}; items observed: {max_items:,}; elapsed telemetry span: {elapsed_min:.1f} min",
        f"- Parameter-change representation: `{delta_representation}`",
        "", "## Recall by corpus", "",
        "Epoch 0 is the pre-training student; positive epochs are completed training epochs.", "",
        _markdown_table(recall, ["epoch", "corpus", "next_acc", "prev_acc",
                                 "cloze_acc", "overall_word_acc"])
        if not recall.empty else "_Missing._",
        "", f"![Recall by corpus]({rel('recall_by_corpus.png')})" if not recall.empty else "",
        "", "## Standard-benchmark damage", "",
        _markdown_table(standard, ["epoch", "macro_accuracy", "epoch0_delta",
                                   "worst_task", "worst_delta"])
        if not standard.empty else "_Missing._",
        "", f"![Standard damage]({rel('standard_damage.png')})" if not standard.empty else "",
        "", f"![Recall–damage trajectory]({rel('recall_damage_frontier.png')})"
        if not recall.empty and not standard.empty else "",
        "", "## Per-layer loss by epoch", "",
        f"Loss measure selected by the optimizer regime: `{loss_measure}`.", "",
        f"![Temporal layer loss]({rel('layer_loss_temporal.png')})" if not loss.empty else "_Missing._",
        "", f"![Layer-loss heatmap]({rel('layer_loss_heatmap.png')})" if not loss.empty else "",
        "", f"![One-row layer-loss density]({rel('layer_loss_one_row_density.png')})" if not loss.empty else "",
        "", "## Per-layer parameter modification", "",
        f"Representation: `{delta_representation}`.", "",
        f"![Temporal parameter delta]({rel('parameter_delta_temporal.png')})" if not delta.empty else "_Missing._",
        "", f"![Parameter-delta heatmap]({rel('parameter_delta_heatmap.png')})" if not delta.empty else "",
        "", f"![One-row parameter density]({rel('parameter_delta_one_row_density.png')})" if not delta.empty else "",
        "", "## Coverage and missing evidence", "",
    ]
    md.extend([f"- Missing: {item}" for item in missing] or
              ["- All mandatory epoch telemetry is present."])
    md += [
        "", "## Artifact index", "",
        f"- Raw metrics: [`metrics.jsonl`](metrics.jsonl)",
        f"- Frozen config: [`config.yaml`](config.yaml)",
        f"- Report assets: [`eval/report_v2/`](eval/report_v2/)",
        "- Future collective reports select this run-local evidence by typed identity; "
        "they do not reconstruct a second source of truth.", "",
    ]
    report = run_dir / "report.md"
    report.write_text("\n".join(x for x in md if x is not None), encoding="utf-8")
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
