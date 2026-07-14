"""Synthesize typed groupings from atomic pipeline-v2 individual reports."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(run: Path, name: str) -> pd.DataFrame:
    path = run / "eval" / "report_v2" / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _elapsed_minutes(run: Path) -> float:
    values = []
    path = run / "metrics.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "t" in row:
                values.append(float(row["t"]))
    return (max(values) - min(values)) / 60 if len(values) > 1 else float("nan")


def _summary_row(manifest: dict) -> dict:
    run = RUNS / manifest["run"]
    recall = _read_csv(run, "recall_by_epoch.csv")
    standard = _read_csv(run, "standard_by_epoch.csv")
    final_recall = float("nan")
    final_standard = float("nan")
    damage = float("nan")
    if not recall.empty:
        final_epoch = recall.epoch.max()
        final_recall = recall[recall.epoch == final_epoch].overall_word_acc.mean()
    if not standard.empty:
        standard = standard.sort_values("epoch")
        final_standard = float(standard.iloc[-1].macro_accuracy)
        damage = float(standard.iloc[0].macro_accuracy) - final_standard
    geometry = manifest.get("geometry") or {}
    return {
        "run": manifest["run"],
        "model": manifest.get("model"),
        "loss": manifest.get("hidden_loss"),
        "censorship": manifest.get("censorship"),
        "B": geometry.get("answers"),
        "K": geometry.get("tokens"),
        "reduction": geometry.get("reduction"),
        "final_recall": final_recall,
        "final_standard": final_standard,
        "standard_damage": damage,
        "elapsed_minutes": _elapsed_minutes(run),
        "report": manifest.get("report"),
    }


def _markdown_table(rows: pd.DataFrame) -> str:
    """Render reports without the optional pandas ``tabulate`` extra."""
    if rows.empty:
        return "_None._"

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    columns = list(rows.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _plots(rows: pd.DataFrame, manifests: list[dict], out: Path) -> None:
    finite = rows.dropna(subset=["standard_damage", "final_recall"])
    if not finite.empty:
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for _, row in finite.iterrows():
            ax.scatter(row.standard_damage, row.final_recall)
            ax.annotate(row.run, (row.standard_damage, row.final_recall), fontsize=6)
        ax.set(xlabel="standard macro-accuracy damage vs epoch 0",
               ylabel="mean final recall", title="Recall–damage frontier")
        ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(out / "recall_damage_frontier.png", dpi=220)
        plt.close(fig)

    for csv_name, value, ylabel, output in (
        ("layer_loss_by_epoch.csv", "loss", "final per-layer loss",
         "final_layer_loss.png"),
        ("parameter_delta_by_epoch.csv", "relative_l2",
         "final relative parameter delta", "final_parameter_delta.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        drawn = False
        for manifest in manifests:
            frame = _read_csv(RUNS / manifest["run"], csv_name)
            if frame.empty or value not in frame:
                continue
            final = frame[frame.epoch == frame.epoch.max()]
            ax.plot(final.layer, final[value], lw=1, label=manifest["run"])
            drawn = True
        if drawn:
            ax.set(xlabel="layer", ylabel=ylabel, title=ylabel.capitalize())
            ax.set_yscale("log")
            ax.grid(alpha=.2); ax.legend(fontsize=5, ncol=2, frameon=False)
            fig.tight_layout(); fig.savefig(out / output, dpi=220)
        plt.close(fig)

    if rows.elapsed_minutes.notna().any():
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ordered = rows.sort_values("elapsed_minutes")
        ax.barh(ordered.run, ordered.elapsed_minutes)
        ax.set(xlabel="telemetry wall minutes", title="Training runtime")
        ax.tick_params(axis="y", labelsize=6)
        fig.tight_layout(); fig.savefig(out / "runtime.png", dpi=220)
        plt.close(fig)


def _write_group(name: str, value: str, manifests: list[dict], pending: list[str],
                 root: Path) -> None:
    out = root / f"{_slug(name)}={_slug(value)}"
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([_summary_row(m) for m in manifests])
    rows.to_csv(out / "runs.csv", index=False)
    _plots(rows, manifests, out)
    table = _markdown_table(rows)
    md = [
        f"# Pipeline-v2 grouped report — {name}: {value}", "",
        f"Inclusion rule: published `report_manifest.json`, campaign `pareto_v2`, "
        f"strict-local certification passed, `{name}={value}`.", "",
        "## Runs", "", table, "",
        "## Cross-run figures", "",
        "![Recall–damage frontier](recall_damage_frontier.png)", "",
        "![Final layer loss](final_layer_loss.png)", "",
        "![Final parameter delta](final_parameter_delta.png)", "",
        "![Runtime](runtime.png)", "",
        "## Missing or report-pending campaign runs", "",
    ]
    md.extend([f"- `{run}`" for run in pending] or ["- None detected."])
    tmp = out / ".report.md.tmp"
    tmp.write_text("\n".join(md) + "\n", encoding="utf-8")
    tmp.replace(out / "report.md")
    payload = {"schema_version": 2, "group_by": name, "value": value,
               "runs": [m["run"] for m in manifests], "pending": pending}
    tmp = out / ".manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out / "manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="pareto_v2")
    ap.add_argument("--group-by", choices=("all", "campaign", "model", "loss",
                                             "censorship", "geometry"),
                    default="all")
    ap.add_argument("--out", default="runs/grouped_reports_v2")
    args = ap.parse_args()

    manifests = []
    for path in sorted(RUNS.glob("*/report_manifest.json")):
        value = _read_json(path)
        if (value.get("campaign") == args.campaign and value.get("complete")
                and value.get("strict_local")):
            manifests.append(value)
    published = {m["run"] for m in manifests}
    pending = []
    for run in sorted(RUNS.glob(f"{args.campaign}_*")):
        if run.name in published or not (run / "config.yaml").is_file():
            continue
        config_text = (run / "config.yaml").read_text(encoding="utf-8")
        if "readout_weight:" not in config_text and "readout_window_blocks:" not in config_text:
            pending.append(run.name)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    wanted = ("campaign", "model", "loss", "censorship", "geometry") \
        if args.group_by == "all" else (args.group_by,)
    for manifest in manifests:
        geometry = manifest.get("geometry") or {}
        values = {
            "campaign": manifest.get("campaign"),
            "model": manifest.get("model"),
            "loss": manifest.get("hidden_loss"),
            "censorship": manifest.get("censorship"),
            "geometry": (f"B{geometry.get('answers')}_K{geometry.get('tokens')}_"
                         f"{geometry.get('reduction')}"),
        }
        for kind in wanted:
            groups[(kind, str(values[kind]))].append(manifest)
    root = ROOT / args.out
    for (kind, value), members in sorted(groups.items()):
        _write_group(kind, value, members, pending, root)
    print(f"wrote {len(groups)} grouped reports under {root}")


if __name__ == "__main__":
    main()
