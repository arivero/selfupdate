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
import yaml

from report_pdf_v2 import write_grouped_pdf

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(run: Path, name: str) -> pd.DataFrame:
    path = run / "eval" / "report_v2" / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _stage_dirs(run: Path) -> list[Path]:
    return sorted(
        (path for path in run.glob("stage*")
         if path.is_dir() and path.name.removeprefix("stage").isdigit()),
        key=lambda path: int(path.name.removeprefix("stage")),
    )


def _config_path(run: Path) -> Path | None:
    flat = run / "config.yaml"
    if flat.is_file():
        return flat
    stages = _stage_dirs(run)
    candidate = stages[0] / "config.yaml" if stages else None
    return candidate if candidate is not None and candidate.is_file() else None


def _has_checkpoint(run: Path) -> bool:
    if (run / "checkpoint").is_dir():
        return True
    stages = _stage_dirs(run)
    return bool(stages) and all(
        (stage / "config.yaml").is_file()
        and (stage / "checkpoint").is_dir()
        for stage in stages
    )


def _elapsed_minutes(run: Path) -> float:
    values = []
    flat = run / "metrics.jsonl"
    paths = [flat] if flat.is_file() else [
        stage / "metrics.jsonl" for stage in _stage_dirs(run)
        if (stage / "metrics.jsonl").is_file()
    ]
    for path in paths:
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
    recall_epoch0 = float("nan")
    recall_delta = float("nan")
    final_standard = float("nan")
    damage = float("nan")
    if not recall.empty:
        epoch_means = recall.groupby("epoch").overall_word_acc.mean()
        final_epoch = recall.epoch.max()
        final_recall = float(epoch_means.loc[final_epoch])
        if 0 in epoch_means.index:
            recall_epoch0 = float(epoch_means.loc[0])
            recall_delta = final_recall - recall_epoch0
    if not standard.empty:
        standard = standard.sort_values("epoch")
        final_standard = float(standard.iloc[-1].macro_accuracy)
        damage = float(standard.iloc[0].macro_accuracy) - final_standard
    geometry = manifest.get("geometry") or {}
    realized = geometry.get("realized") or {}
    lanes = realized.get("lanes_per_update") or {}
    tokens = realized.get("aligned_tokens_per_update") or {}
    return {
        "run": manifest["run"],
        "model": manifest.get("model"),
        "run_class": manifest.get("run_class"),
        "loss": manifest.get("hidden_loss"),
        "optimizer": manifest.get("optimizer"),
        "lr": manifest.get("lr"),
        "adam_betas": manifest.get("v4_adam_betas"),
        "adam_eps": manifest.get("v4_adam_eps"),
        "censorship": manifest.get("censorship"),
        "batching": manifest.get("batching"),
        "B": geometry.get("answers"),
        "K": geometry.get("tokens"),
        "reduction": geometry.get("reduction"),
        "realized_B_mean": lanes.get("mean"),
        "realized_B_median": lanes.get("median"),
        "realized_cells_mean": tokens.get("mean"),
        "realized_cells_median": tokens.get("median"),
        "recall_epoch0": recall_epoch0,
        "final_recall": final_recall,
        "recall_delta_vs_epoch0": recall_delta,
        "final_standard": final_standard,
        "standard_damage": damage,
        "elapsed_minutes": _elapsed_minutes(run),
        "report": manifest.get("report"),
    }


def _config_metadata(path: Path) -> dict:
    """Extract only report identity fields, tolerating partial arm configs."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(value, dict):
        return {}
    model = value.get("model") or {}
    train = value.get("train") or {}
    if not isinstance(model, dict):
        model = {}
    if not isinstance(train, dict):
        train = {}
    metadata = {
        "run": str(value.get("run_name") or "").split("/", 1)[0],
        "model": model.get("name"),
        "loss": train.get("hidden_loss"),
        "optimizer": train.get("v4_optimizer"),
        "lr": train.get("lr"),
    }
    # Campaign arm YAMLs are intentionally small overlays.  Resolve identity
    # defaults from the uniquely matching checked-in ``base_*_v4_full`` file
    # when its model token is present in the arm filename.  This is provenance
    # enrichment only; it does not reproduce or alter training config merging.
    if path.parent.name == "train40":
        base_dir = ROOT / "configs" / "experiments" / "h100_smoke"
        matches = []
        for base in base_dir.glob("base_*_v4_full.yaml"):
            token = base.stem.removeprefix("base_").removesuffix("_v4_full")
            if token in path.stem:
                matches.append(base)
        if len(matches) == 1:
            base_metadata = _config_metadata(matches[0])
            for key in ("model", "loss", "optimizer", "lr"):
                if metadata[key] is None:
                    metadata[key] = base_metadata[key]
    return metadata


def _coverage_roster(campaign: str, report_manifests: list[dict]) -> pd.DataFrame:
    """Inventory configured and materialized runs without certifying them."""
    discovered: dict[str, dict] = {}
    for path in sorted((ROOT / "configs").glob("**/*.yaml")):
        metadata = _config_metadata(path)
        run_name = metadata.pop("run", "")
        if run_name.startswith(f"{campaign}_"):
            discovered.setdefault(run_name, {}).update(
                {key: value for key, value in metadata.items()
                 if value is not None}
            )
    for run in sorted(RUNS.glob(f"{campaign}_*")):
        config_path = _config_path(run)
        if config_path is None:
            continue
        metadata = _config_metadata(config_path)
        metadata.pop("run", None)
        discovered.setdefault(run.name, {}).update(
            {key: value for key, value in metadata.items() if value is not None}
        )
    manifest_by_run = {value["run"]: value for value in report_manifests}
    rows = []
    for run_name, metadata in sorted(discovered.items()):
        run = RUNS / run_name
        manifest = manifest_by_run.get(run_name)
        if manifest is not None:
            metadata.update({
                "model": manifest.get("model") or metadata.get("model"),
                "loss": manifest.get("hidden_loss") or metadata.get("loss"),
                "optimizer": manifest.get("optimizer") or metadata.get("optimizer"),
                "lr": (manifest.get("lr") if manifest.get("lr") is not None
                       else metadata.get("lr")),
            })
        if manifest is not None and manifest.get("complete") and manifest.get("strict_local"):
            status = "strict_local"
        elif manifest is not None:
            status = "report"
        elif run.is_dir() and _has_checkpoint(run):
            status = "checkpoint"
        else:
            status = "training pending"
        if manifest is not None and not manifest.get("strict_local"):
            missing = "; ".join(str(item) for item in manifest.get("missing", []))
            certification = "strict-local certification not passed"
            if missing:
                certification += f"; missing: {missing}"
        elif status == "checkpoint":
            certification = "group report and strict-local certification pending"
        elif status == "training pending":
            certification = "checkpoint and report pending"
        else:
            certification = "strict-local certification passed"
        rows.append({"run": run_name, "status": status, **metadata,
                     "certification": certification})
    return pd.DataFrame(rows, columns=["run", "status", "model", "loss",
                                       "optimizer", "lr", "certification"])


def _markdown_table(rows: pd.DataFrame) -> str:
    """Render reports without the optional pandas ``tabulate`` extra."""
    if rows.empty:
        return "_None._"

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if value and abs(value) < 1e-3:
                return f"{value:.2e}"
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
    # A regenerated group must never inherit a figure from an older eligible
    # set.  Each of these files is fully derived and recreated below only when
    # its current inputs exist.
    for name in ("recall_damage_frontier.png", "final_layer_loss.png",
                 "final_parameter_delta.png", "runtime.png"):
        (out / name).unlink(missing_ok=True)
    if rows.empty:
        return
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


def _coverage_layer_plots(coverage: pd.DataFrame, out: Path) -> list[tuple[str, Path]]:
    """Descriptive cross-run layer summaries, never frontier evidence.

    The publication frontier remains strict-local only.  AGENTS.md also
    requires a cross-run layer summary for every in-scope completed report,
    including historical runs whose locality certificate is honestly missing.
    Keep those plots in a separately titled coverage section.
    """
    specs = (
        ("layer_loss_by_epoch.csv", "loss", "final per-layer loss",
         "coverage_final_layer_loss.png"),
        ("parameter_delta_by_epoch.csv", "relative_l2",
         "final relative parameter delta",
         "coverage_final_parameter_delta.png"),
    )
    outputs = []
    for csv_name, value, ylabel, filename in specs:
        path = out / filename
        path.unlink(missing_ok=True)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        drawn = False
        for row in coverage.itertuples(index=False):
            if row.status not in ("report", "strict_local"):
                continue
            frame = _read_csv(RUNS / row.run, csv_name)
            if frame.empty or value not in frame:
                continue
            final = frame[frame.epoch == frame.epoch.max()]
            ax.plot(final.layer, final[value], lw=1, label=row.run)
            drawn = True
        if drawn:
            ax.set(xlabel="layer", ylabel=ylabel,
                   title=f"Coverage only (certification may be missing): {ylabel}")
            ax.set_yscale("log")
            ax.grid(alpha=.2)
            ax.legend(fontsize=6, ncol=2, frameon=False)
            fig.tight_layout()
            fig.savefig(path, dpi=220)
            outputs.append((f"Coverage only — {ylabel}", path))
        plt.close(fig)
    return outputs


def _write_group(name: str, value: str, manifests: list[dict],
                 coverage: pd.DataFrame, root: Path) -> None:
    out = root / f"{_slug(name)}={_slug(value)}"
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([_summary_row(m) for m in manifests])
    rows.to_csv(out / "runs.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    _plots(rows, manifests, out)
    coverage_figures = _coverage_layer_plots(coverage, out)
    table = _markdown_table(rows)
    campaign = (manifests[0].get("campaign") if manifests else
                value if name == "campaign" else "unknown")
    figure_specs = [
        ("Recall–damage frontier", out / "recall_damage_frontier.png"),
        ("Final layer loss", out / "final_layer_loss.png"),
        ("Final parameter delta", out / "final_parameter_delta.png"),
        ("Runtime", out / "runtime.png"),
    ]
    available_figures = [(title, path) for title, path in figure_specs
                         if path.is_file()]
    cross_run = ([item for title, path in available_figures
                  for item in (f"![{title}]({path.name})", "")] if rows.size else [
        "_No strictly local certified runs are currently eligible for "
        "cross-run figures._", "",
    ])
    coverage_table = _markdown_table(coverage)
    noneligible = coverage[coverage.status != "strict_local"] if not coverage.empty else coverage
    notes = ([
        f"{row.run}: status={row.status}; {row.certification}; excluded from "
        "eligible tables and figures."
        for row in noneligible.itertuples(index=False)
    ] or ["No missing or uncertified discovered runs."])
    md = [
        f"# Grouped training report v2 — {name}: {value}", "",
        f"Inclusion rule: published `report_manifest.json`, campaign "
        f"`{campaign}`, "
        f"strict-local certification passed, `{name}={value}`.", "",
        "## Eligible strictly local runs", "", table, "",
        "## All discovered campaign runs (coverage/provenance only)", "",
        "Rows below are inventory, not frontier evidence. Only `strict_local` "
        "rows may appear in the eligible table or cross-run figures.", "",
        coverage_table, "",
        "## Cross-run figures", "",
    ]
    md.extend(cross_run)
    md.extend([
        "## Coverage-only cross-run layer summaries", "",
        "These figures include completed reports lacking strict-local "
        "certification. They describe recorded dynamics and are not frontier "
        "evidence.", "",
    ])
    if coverage_figures:
        md.extend(item for title, path in coverage_figures
                  for item in (f"![{title}]({path.name})", ""))
    else:
        md.extend(["_No completed report supplies cross-run layer data._", ""])
    md.extend(["## Missing artifacts and certification", ""])
    md.extend([f"- {note}" for note in notes])
    tmp = out / ".report.md.tmp"
    tmp.write_text("\n".join(md) + "\n", encoding="utf-8")
    tmp.replace(out / "report.md")
    pdf_path = write_grouped_pdf(
        out / "report.pdf",
        title=f"Grouped training report v2 — {name}: {value}",
        inclusion=[
            f"Campaign: {campaign}",
            f"Grouping: {name}={value}",
            "Inclusion rule: a published complete report_manifest.json with "
            "strict_local=true. Coverage rows lacking that certification are "
            "provenance only and are never frontier evidence.",
        ],
        eligible=rows,
        coverage=coverage,
        notes=notes,
        figures=available_figures + coverage_figures,
    )
    pending = noneligible.run.tolist() if not noneligible.empty else []
    pending_checkpoints = {
        row.run: row.status in ("checkpoint", "report")
        for row in noneligible.itertuples(index=False)
    }
    payload = {"schema_version": 4, "group_by": name, "value": value,
               "runs": [m["run"] for m in manifests], "pending": pending,
               "pending_checkpoint_complete": pending_checkpoints,
               "coverage": "coverage.csv", "pdf": pdf_path.name}
    tmp = out / ".manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out / "manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="pareto_v2")
    ap.add_argument("--group-by", choices=("all", "campaign", "model", "run_class",
                                             "loss", "censorship", "geometry"),
                    default="all")
    ap.add_argument("--out", default="runs/grouped_reports_v2")
    args = ap.parse_args()

    report_manifests = []
    for path in sorted(RUNS.glob("*/report_manifest.json")):
        value = _read_json(path)
        if value.get("campaign") == args.campaign:
            report_manifests.append(value)
    manifests = [value for value in report_manifests
                 if value.get("complete") and value.get("strict_local")]
    coverage = _coverage_roster(args.campaign, report_manifests)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    wanted = ("campaign", "model", "run_class", "loss", "censorship", "geometry") \
        if args.group_by == "all" else (args.group_by,)
    for manifest in manifests:
        geometry = manifest.get("geometry") or {}
        values = {
            "campaign": manifest.get("campaign"),
            "model": manifest.get("model"),
            "run_class": manifest.get("run_class"),
            "loss": manifest.get("hidden_loss"),
            "censorship": manifest.get("censorship"),
            "geometry": (f"B{geometry.get('answers')}_K{geometry.get('tokens')}_"
                         f"{geometry.get('reduction')}"),
        }
        for kind in wanted:
            groups[(kind, str(values[kind]))].append(manifest)
    if not coverage.empty and not groups and args.group_by in ("all", "campaign"):
        # Publish the campaign's explicit missing/pending ledger even when
        # scientific certification excludes every completed report.
        groups[("campaign", args.campaign)] = []
    root = ROOT / args.out
    for (kind, value), members in sorted(groups.items()):
        _write_group(kind, value, members, coverage, root)
    print(f"wrote {len(groups)} grouped reports under {root}")


if __name__ == "__main__":
    main()
