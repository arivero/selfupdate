"""Gate-aware PDF report of all runs: runs/report.pdf.

Assembles whatever artifacts exist and classifies each run before it can be
used as method evidence. Legacy reference-text readout, old tail-only config
keys, and missing readout provenance are reported as excluded/confounded,
not silently mixed into the layerwise-distillation claim.

Usage: python scripts/report.py [--out runs/report.pdf]
"""

import argparse
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

import cross_report

RUNS = Path("runs")

OLD_KEYS = {
    "tail_ce_blocks", "tail_ce_weight", "tail_ce_kind", "tail_hidden_weight",
    "last_block_ce_weight", "lens_ce_weight", "lens_ce_from", "answer_ce_weight",
    "last_block_" + "task" + "_label_weight",
    "lens_" + "task" + "_label_weight",
    "anchor_" + "ce_weight", "lens_" + "from_layer",
}
FORBIDDEN_REFERENCE_SOURCE = "task" + "_label"


def _text_page(pdf, title, body, fontsize=9):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold")
    fig.text(0.08, 0.90, body, fontsize=fontsize, family="monospace",
             va="top", wrap=True)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _image_page(pdf, title, png):
    if not Path(png).exists():
        return
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold")
    ax = fig.add_axes([0.05, 0.25, 0.9, 0.65])
    ax.imshow(mpimg.imread(png), interpolation="lanczos")
    ax.axis("off")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _image_grid_page(pdf, title, pngs, per_page=6):
    pngs = [Path(p) for p in pngs if Path(p).exists()]
    for i in range(0, len(pngs), per_page):
        chunk = pngs[i:i + per_page]
        rows = 3 if per_page >= 6 else 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
        axes = list(axes.ravel())
        fig.suptitle(f"{title} ({i + 1}-{i + len(chunk)} of {len(pngs)})",
                     fontsize=14, weight="bold")
        for ax, p in zip(axes, chunk):
            ax.imshow(mpimg.imread(p), interpolation="lanczos")
            ax.set_title(p.parent.parent.name, fontsize=7)
            ax.axis("off")
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig, dpi=300)
        plt.close(fig)


def _markdown_text(path: Path, max_chars: int = 9000) -> str:
    if not path.exists():
        return f"{path} is missing."
    text = path.read_text(encoding="utf-8")
    return text[:max_chars] + ("\n\n[truncated]" if len(text) > max_chars else "")


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _run_cfg(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        return {"_parse_error": f"{type(e).__name__}: {e}"}


def _evidence_status(cfg: dict) -> tuple[str, list[str]]:
    """Return (status, warnings). Only status == method_clean is method evidence."""
    warnings: list[str] = []
    if cfg.get("_parse_error"):
        return "unreadable", [cfg["_parse_error"]]
    t = cfg.get("train", {}) or {}
    run_class = t.get("run_class", "method")
    old = sorted(k for k in OLD_KEYS if k in t)
    if old:
        warnings.append("legacy config keys: " + ", ".join(old))
    if (t.get("readout_source") == FORBIDDEN_REFERENCE_SOURCE
            or t.get("tail_ce_kind") == FORBIDDEN_REFERENCE_SOURCE):
        warnings.append("FORBIDDEN legacy reference-text training signal")
    old_blocks = t.get("tail_ce_blocks", 0) or 0
    new_blocks = t.get("readout_window_blocks", 0) or 0
    blocks = new_blocks or old_blocks
    if blocks > 0:
        if t.get("readout_source", t.get("tail_ce_kind", "UNSET")) == "UNSET":
            warnings.append("readout source not pinned")
        if t.get("conn_window", 0) != blocks or t.get("conn_stride", 0) != 1:
            warnings.append("readout not attached to sanctioned sliding window")
    if run_class != "method":
        return run_class, warnings
    if warnings:
        return "confounded", warnings
    return "method_clean", warnings


def _best_run(only_method_clean: bool = False):
    """(name, recite dict) of the best full-corpus recitation among runs."""
    best = None
    for d in sorted(RUNS.iterdir()):
        if only_method_clean and _evidence_status(_run_cfg(d))[0] != "method_clean":
            continue
        r = _read_json(d / "eval/recite.json")
        if r and (best is None or r["cer"] < best[1]["cer"]):
            best = (d.name, r)
    return best


def _best_corpus_row(clean_only: bool = True) -> pd.Series | None:
    path = RUNS / "corpus.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "evidence_status" not in df:
        return None
    rows = df[df["evidence_status"] == "method_clean"].copy() if clean_only else df.copy()
    if rows.empty:
        return None
    for col in ("full_eval_cer", "best_epoch_cer", "last_epoch_cer"):
        if col in rows and rows[col].notna().any():
            return rows.sort_values(col, na_position="last").iloc[0]
    return rows.iloc[0]


def _fmt(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, str):
        if not v or v.lower() == "nan":
            return "NA"
    else:
        try:
            if pd.isna(v):
                return "NA"
        except Exception:
            pass
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def summary_text() -> str:
    base = _read_json(RUNS / "base-eval-full/recite.json")
    best_row = _best_corpus_row(clean_only=True)
    if best_row is None:
        best_row = _best_corpus_row(clean_only=False)
    best_clean = _best_run(only_method_clean=True)
    best_any = _best_run(only_method_clean=False)
    lines = [
        "Project: self-distillation of context (same model as teacher and student).",
        "Teacher sees privileged context (RAG passage / <think> trace); the student",
        "must reproduce its behavior without it. Corpus: 'La tierra de Alvargonzalez'",
        "(A. Machado, 1912), 725 verses, with generated task variants.",
        "",
        "Branch law enforced by this report: method evidence must be layerwise",
        "forward distillation, must not train embedding/final norm/unembedding,",
        "and any behavioral readout must be teacher-sourced and attached to a",
        "stride-1 sliding connected window. This branch never trains against",
        "poem/reference tokens; old artifacts that did so are forbidden legacy",
        "evidence only.",
        "",
    ]
    if best_row is not None:
        if best_row.get("evidence_status") != "method_clean":
            lines += [
                "No full-corpus artifact currently qualifies as clean method evidence.",
                "The highlighted row below is raw evidence and must be read with its status.",
                "",
            ]
        lines += [
            "Honest metric layout for the highlighted row:",
            f"  run={best_row.get('run')} model={best_row.get('model')} "
            f"evidence={best_row.get('evidence_status')}",
            f"  epoch 0: CER={_fmt(best_row.get('epoch0_cer'))} "
            f"CE={_fmt(best_row.get('epoch0_general_ce'))} "
            f"source={best_row.get('epoch0_source')}",
            f"  last epoch {best_row.get('last_epoch')}: "
            f"CER={_fmt(best_row.get('last_epoch_cer'))} "
            f"CE={_fmt(best_row.get('last_epoch_ce'))} "
            f"forgotten_dCE={_fmt(best_row.get('last_epoch_forgetting_ce'))}",
            f"  best epoch {best_row.get('best_epoch')}: "
            f"CER={_fmt(best_row.get('best_epoch_cer'))} "
            f"CE={_fmt(best_row.get('best_epoch_ce'))} "
            f"forgotten_dCE={_fmt(best_row.get('best_epoch_forgetting_ce'))}",
            f"  final full eval: CER={_fmt(best_row.get('full_eval_cer'))} "
            f"CE={_fmt(best_row.get('general_ce'))} "
            f"forgotten_dCE={_fmt(best_row.get('final_forgetting_ce'))}",
            "",
        ]
    elif base and best_clean:
        name, run = best_clean
        lines += [
            f"Recitation (full corpus, n={base['n']}): epoch-zero native teacher CER {base['cer']:.3f} ->",
            f"best clean-method artifact ({name}) CER {run['cer']:.3f},",
            f"{run['line_exact']:.0%} lines verbatim.",
            f"Forgetting probe (cross-entropy/log loss on held-out text): epoch-zero native teacher {base['general']['mean_ce']:.3f},",
            f"best-run {run['general']['mean_ce']:.3f} (delta {run['general']['mean_ce']-base['general']['mean_ce']:+.2f}).",
            "  (computed from current artifacts)",
            "",
        ]
    elif best_any:
        lines += [
            "No full-corpus artifact currently qualifies as clean method evidence.",
            f"Best raw artifact is {best_any[0]} (reported later with evidence status).",
            "",
        ]
    lines += [
        "Report gates:",
        " 1. Old tail_* / reference-text-training configs are excluded or flagged.",
        " 2. Readout source must be explicit; no inherited source is evidence.",
        " 3. Per-run appendix lists run_class, readout source/window, and warnings.",
        " 4. Signal attribution JSON is expected next to every readout claim.",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Details in the following pages;",
        "reproducibility: configs/experiments/*.yaml, runs/*/metrics.jsonl, git log.",
    ]
    return "\n".join(lines)


_COL_SHORT = {
    "last_train_cer": "train_cer", "full_eval_cer": "eval_cer",
    "line_exact": "exact", "forgetting_dCE": "forget",
    "compaction": "compact", "schedule": "sched",
    "loss_first": "loss0", "loss_final": "lossN", "train_min": "min",
}


def results_page(pdf):
    """Landscape page with the runs table rendered as a real table —
    the portrait text dump was unreadable once the grid grew."""
    md = RUNS / "results.md"
    if not md.exists():
        return
    lines = [l for l in md.read_text().splitlines() if l.startswith("|")]
    if len(lines) < 3:
        return
    split = lambda l: [c.strip() for c in l.strip("|").split("|")]
    header = [_COL_SHORT.get(h, h) for h in split(lines[0])]

    def fmt(x):
        if x in ("", "nan"):
            return "—"
        try:
            return f"{float(x):.3g}"
        except ValueError:
            return x

    rows = [[fmt(c) for c in split(l)] for l in lines[2:]]
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.04, 0.94, "Results table (all runs)", fontsize=16, weight="bold")
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header,
                   loc="upper center", cellLoc="right")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(range(len(header)))
    tbl.scale(1, 1.35)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
    pdf.savefig(fig)
    plt.close(fig)


def corpus_page(pdf):
    corpus = RUNS / "corpus.csv"
    if not corpus.exists():
        _text_page(pdf, "Corpus index", "runs/corpus.csv is missing.\nRun scripts/build_corpus_index.py before report generation.")
        return
    df = pd.read_csv(corpus)
    if df.empty:
        _text_page(pdf, "Corpus index", "runs/corpus.csv has no rows.")
        return
    status = df["evidence_status"].value_counts(dropna=False).to_string()
    missing_full = df[df["full_eval_cer"].isna()]["run"].tolist()
    readout_window = pd.to_numeric(df["readout_window"], errors="coerce").fillna(0)
    has_signal = df["signal_attribution_json"].fillna(False).astype(bool)
    missing_signal = df[(readout_window > 0) & (~has_signal)]["run"].tolist()
    clean = df[df["evidence_status"] == "method_clean"].copy()
    clean = clean.sort_values("full_eval_cer", na_position="last").head(12)
    lines = [
        f"Rows: {len(df)}",
        "",
        "Evidence status counts:",
        status,
        "",
        "Top clean-method rows by full_eval_cer:",
    ]
    if clean.empty:
        lines.append("  (none)")
    else:
        for _, r in clean.iterrows():
            lines.append(
                f"  {r['run']}: e0_CER={r.get('epoch0_cer')} "
                f"last{r.get('last_epoch')}_CER={r.get('last_epoch_cer')} "
                f"best{r.get('best_epoch')}_CER={r.get('best_epoch_cer')} "
                f"final_CER={r.get('full_eval_cer')} final_dCE={r.get('final_forgetting_ce')} "
                f"source={r['readout_source']} window={r['readout_window']} "
                f"hidden_share={r['hidden_share']}"
            )
    lines += [
        "",
        f"Missing full eval: {len(missing_full)}",
        ", ".join(missing_full[:40]) + (" ..." if len(missing_full) > 40 else ""),
        "",
        f"Readout runs missing signal attribution: {len(missing_signal)}",
        ", ".join(missing_signal[:40]) + (" ..." if len(missing_signal) > 40 else ""),
    ]
    _text_page(pdf, "Corpus Index And Artifact Completeness", "\n".join(lines), fontsize=7)


def coverage_matrix_page(pdf):
    path = RUNS / "experiment_coverage_matrix.csv"
    if not path.exists():
        _text_page(pdf, "Experiment Coverage Matrix",
                   "runs/experiment_coverage_matrix.csv is missing.\n"
                   "Run scripts/experiment_report_assets.py before report generation.")
        return
    df = pd.read_csv(path).fillna("")
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape, first report page.
    fig.text(0.02, 0.96, "Experiment Coverage Matrix", fontsize=14, weight="bold")
    fig.text(
        0.02, 0.925,
        "Cells are counts by status: C=completed clean, P=planned clean, L=legacy/provenance caveat, "
        "A=ablation/control, B=underbudget, X=denied/confounded, T=epoch-zero teacher reference. "
        "A run can count in multiple rows.",
        fontsize=7,
    )
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.88])
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(4.9)
    tbl.auto_set_column_width(range(len(df.columns)))
    tbl.scale(1.0, 1.05)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def objective_candidate_page(pdf):
    path = RUNS / "objective_candidate_matrix.csv"
    if not path.exists():
        _text_page(pdf, "Objective Candidate Matrix",
                   "runs/objective_candidate_matrix.csv is missing.\n"
                   "Run scripts/experiment_report_assets.py before report generation.")
        return
    df = pd.read_csv(path).fillna("")
    if df.empty:
        _text_page(pdf, "Objective Candidate Matrix", "No objective candidate rows are available.")
        return
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.02, 0.96, "Objective Candidate Matrix", fontsize=14, weight="bold")
    fig.text(
        0.02, 0.925,
        "For each model/corpus cell: completed method evidence, historical audit evidence, "
        "and queued clean candidates.",
        fontsize=7,
    )
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.88])
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(4.8)
    tbl.auto_set_column_width(range(len(df.columns)))
    tbl.scale(1.0, 1.05)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if c in (0, 1, 2, 3, 4) and r > 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def loss_by_model_page(pdf):
    path = RUNS / "loss_by_model_size.csv"
    if not path.exists():
        _text_page(pdf, "Loss By Model Size",
                   "runs/loss_by_model_size.csv is missing.\n"
                   "Run scripts/experiment_report_assets.py before report generation.")
        return
    df = pd.read_csv(path).fillna("")
    if df.empty:
        _text_page(pdf, "Loss By Model Size", "No loss comparison rows are available.")
        return
    clean_counts = pd.to_numeric(df["clean_or_legacy_runs"], errors="coerce").fillna(0)
    df = df[clean_counts > 0].copy()
    if df.empty:
        df = pd.read_csv(path).fillna("")
        title_note = "No clean/legacy method-evidence rows yet; showing audit rows."
    else:
        title_note = "Clean or legacy-named method-evidence rows only."
    if "rank_in_model" in df:
        df["rank_in_model"] = pd.to_numeric(df["rank_in_model"], errors="coerce")
        df = df.sort_values(["model", "rank_in_model"]).groupby("model").head(4)
    keep = [
        "model", "loss", "rank_in_model", "n_runs", "clean_or_legacy_runs",
        "best_cer", "median_cer", "best_forgetting_ce", "best_run",
        "best_verdict", "best_epoch",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in ("best_cer", "median_cer", "best_forgetting_ce"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
    if "rank_in_model" in df:
        df["rank_in_model"] = pd.to_numeric(df["rank_in_model"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{int(x)}"
        )
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.02, 0.96, "Loss By Model Size", fontsize=14, weight="bold")
    fig.text(
        0.02, 0.925,
        "CER = character error rate. Forgetting is held-out cross-entropy/log-loss delta from epoch-zero teacher. "
        + title_note,
        fontsize=7,
    )
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.88])
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.5)
    tbl.auto_set_column_width(range(len(df.columns)))
    tbl.scale(1.0, 1.15)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if c in (0, 1, 8, 9) and r > 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def best_loss_window_by_corpus_page(pdf):
    path = RUNS / "best_loss_window_by_corpus.csv"
    if not path.exists():
        _text_page(pdf, "Best Loss/Window By Corpus",
                   "runs/best_loss_window_by_corpus.csv is missing.\n"
                   "Run scripts/experiment_report_assets.py before report generation.")
        return
    df = pd.read_csv(path).fillna("")
    if df.empty:
        _text_page(pdf, "Best Loss/Window By Corpus", "No corpus comparison rows are available.")
        return
    done = df[df["is_completed"].astype(str).str.lower().isin(["true", "1"])].copy()
    if done.empty:
        _text_page(pdf, "Best Loss/Window By Corpus", "No completed comparable runs are available.")
        return
    for col in ("comparison_cer", "comparison_forgetting_ce", "intrusion_hit_rate"):
        if col in done:
            done[col] = pd.to_numeric(done[col], errors="coerce")
    done = done.sort_values(
        ["model_label", "corpus_family", "comparison_cer", "comparison_forgetting_ce"],
        na_position="last",
    ).groupby(["model_label", "corpus_family"], as_index=False).head(1)
    keep = [
        "model_label", "corpus_family", "loss", "window_label",
        "comparison_cer", "comparison_forgetting_ce", "intrusion_hit_rate",
        "run", "saved_verdict",
    ]
    done = done[[c for c in keep if c in done.columns]].copy()
    for col in ("comparison_cer", "comparison_forgetting_ce", "intrusion_hit_rate"):
        if col in done:
            done[col] = pd.to_numeric(done[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.02, 0.96, "Best Loss/Window By Corpus", fontsize=14, weight="bold")
    fig.text(
        0.02, 0.925,
        "One best completed row per model and corpus family. Method status is explicit; "
        "confounded rows are evidence for gaps, not method claims.",
        fontsize=7,
    )
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.88])
    ax.axis("off")
    tbl = ax.table(
        cellText=done.values.tolist(),
        colLabels=done.columns.tolist(),
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.6)
    tbl.auto_set_column_width(range(len(done.columns)))
    tbl.scale(1.0, 1.15)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        if c in (0, 1, 2, 7, 8) and r > 0:
            cell.set_text_props(ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def cross_checkout_pages(pdf):
    path = RUNS / "retention_index.csv"
    if not path.exists():
        _text_page(pdf, "Cross-Checkout Recall And Forgetting",
                   "runs/retention_index.csv is missing.\n"
                   "Run scripts/retention_index.py before report generation.")
        return
    df = pd.read_csv(path)
    if df.empty:
        _text_page(pdf, "Cross-Checkout Recall And Forgetting",
                   "No cross-checkout retention rows are available.")
        return
    for col in ("has_retention", "has_recite"):
        if col in df:
            df[col] = df[col].astype(bool)
    cross_report._text_page(
        pdf,
        "Cross-checkout recall/forgetting score update",
        cross_report.summary_text(df),
        fontsize=8,
    )
    cross_report._image_page(
        pdf,
        "Recall vs capability retention (new battery)",
        RUNS / "recall_retention.png",
    )
    cross_report._table_page(
        pdf,
        "Best new recall/forgetting scores by method family and model",
        "Recall: CER (lower better) and exact-match probes (higher better). "
        "Forgetting/retention: ARC retained and WikiText-2 log(ppl / teacher ppl) damage.",
        cross_report._best_by_group(df),
        ["source", "model", "run", "recall_cer", "cont_mach", "cloze_mach",
         "censor_cerv", "arc_acc", "arc_retain", "wiki_log_damage"],
        fontsize=5.8,
    )
    cross_report._coverage_page(pdf, df)


def layer_loss_pages(pdf):
    manifest = RUNS / "layer_loss_manifest.csv"
    if not manifest.exists():
        _text_page(pdf, "Layer-Loss Artifacts",
                   "runs/layer_loss_manifest.csv is missing.\n"
                   "Run scripts/layer_loss_plots.py --force before report generation.")
        return
    df = pd.read_csv(manifest)
    lines = [
        f"Runs with per-layer loss artifacts: {len(df)}",
        "",
        "Each listed run has:",
        "  - runs/<run>/eval/layer_losses.png",
        "  - runs/<run>/eval/layer_losses_heatmap.png",
        "  - runs/<run>/eval/layer_losses.csv",
        "",
        "Final-epoch loss summary:",
        df[["run", "epochs", "layers", "final_mean_loss", "final_min_loss",
            "final_max_loss"]].to_string(index=False, max_rows=80),
    ]
    _text_page(pdf, "Layer-Loss Artifact Index", "\n".join(lines), fontsize=6)
    pngs = [RUNS.parent / p for p in df["line_plot"].dropna().tolist()]
    heatmaps = [RUNS.parent / p for p in df["heatmap"].dropna().tolist()]
    _image_grid_page(pdf, "Loss By Layer: Epoch Lines", pngs[:60], per_page=6)
    _image_grid_page(pdf, "Loss By Layer: Heatmaps", heatmaps[:60], per_page=6)


def layer_swap_pages(pdf):
    for csv in sorted(RUNS.glob("*/eval/layer_swap.csv")):
        run = csv.parent.parent.name
        df = pd.read_csv(csv)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df.layer, df.graft_cer, marker="o", label="graft (base + trained block L)")
        ax.plot(df.layer, df.ablate_cer, marker="s", label="ablate (trained, block L reverted)")
        ax.set_xlabel("layer")
        ax.set_ylabel("recitation CER")
        ax.set_title(f"Causal localization: layer graft/ablate — {run}")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def per_run_appendix(pdf):
    blocks = []
    corpus = RUNS / "corpus.csv"
    if corpus.exists():
        df = pd.read_csv(corpus)
        for _, r in df.sort_values(["evidence_status", "run"]).iterrows():
            b = [f"== {r['run']} =="]
            b.append(f"  run_class={r['run_class']} evidence={r['evidence_status']} "
                     f"model={r['model']} schedule={r['schedule']} hidden_loss={r['hidden_loss']}")
            b.append(f"  readout_source={r['readout_source']} readout_window={r['readout_window']} "
                     f"readout_weight={r['readout_weight']} window_hidden_weight={r['window_hidden_weight']} "
                     f"conn={r['conn_window']}/{r['conn_stride']}")
            if isinstance(r.get("warnings"), str) and r["warnings"]:
                b.append(f"  WARN: {r['warnings']}")
            b.append(f"  loss: first20 {r['loss_first']} -> last20 {r['loss_final']} "
                     f"items_seen={r['items_seen']}")
            b.append(f"  epoch0: CER={r.get('epoch0_cer')} CE={r.get('epoch0_general_ce')} "
                     f"source={r.get('epoch0_source')}")
            b.append(f"  last epoch {r.get('last_epoch')}: CER={r.get('last_epoch_cer')} "
                     f"CE={r.get('last_epoch_ce')} forgotten_dCE={r.get('last_epoch_forgetting_ce')}")
            b.append(f"  best epoch {r.get('best_epoch')}: CER={r.get('best_epoch_cer')} "
                     f"CE={r.get('best_epoch_ce')} forgotten_dCE={r.get('best_epoch_forgetting_ce')}")
            b.append(f"  final full eval: CER={r['full_eval_cer']} "
                     f"line_exact={r['full_eval_line_exact']} general_CE={r['general_ce']} "
                     f"forgotten_dCE={r.get('final_forgetting_ce', r['forgetting_delta_ce'])}")
            b.append(f"  artifacts: destruction={r['destruction_json']} "
                     f"signal={r['signal_attribution_json']} hidden_share={r['hidden_share']} "
                     f"active_config={r['active_config']}")
            blocks.append("\n".join(b))
    else:
        for d in sorted(RUNS.iterdir()):
            if not (d / "config.yaml").exists():
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            status, warnings = _evidence_status(cfg)
            b = [f"== {d.name} =="]
            t = cfg.get("train", {})
            b.append(f"  method={t.get('method')} schedule={t.get('schedule')} "
                     f"run_class={t.get('run_class', 'method')} evidence={status}")
            if warnings:
                b.append("  WARN: " + "; ".join(warnings))
            blocks.append("\n".join(b))
    for i in range(0, len(blocks), 6):
        _text_page(pdf, f"Per-run details ({i // 6 + 1})",
                   "\n\n".join(blocks[i:i + 6]), fontsize=7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/report.pdf")
    ap.add_argument("--lossgrid-only", action="store_true",
                    help="render only the active July-11 loss-grid scorecard")
    args = ap.parse_args()

    with PdfPages(args.out) as pdf:
        if args.lossgrid_only:
            _text_page(pdf, "July 11 1.7B loss-grid checkpoint scorecard",
                       _markdown_text(RUNS / "lossgrid_report.md"), fontsize=5)
            _image_page(pdf, "July 11 a_/lw_ checkpoint recall",
                        RUNS / "tasks_report.png")
            print(f"wrote {args.out}")
            return
        coverage_matrix_page(pdf)
        objective_candidate_page(pdf)
        loss_by_model_page(pdf)
        best_loss_window_by_corpus_page(pdf)
        _text_page(pdf, "Self-distillation of context — experiment report",
                   summary_text())
        cross_checkout_pages(pdf)
        results_page(pdf)
        _text_page(pdf, "1.7B loss-grid checkpoint scorecard",
                   _markdown_text(RUNS / "lossgrid_report.md"), fontsize=5)
        corpus_page(pdf)
        _image_page(pdf, "Accuracy Aspects", RUNS / "accuracy_aspects.png")
        _image_page(pdf, "Destruction Aspects", RUNS / "destruction_aspects.png")
        _image_page(pdf, "Layer Modification Heatmap", RUNS / "layer_modification_heatmap.png")
        _text_page(pdf, "Qualitative Chat Summary",
                   _markdown_text(RUNS / "qualitative_chat_summary.md"), fontsize=6)
        layer_loss_pages(pdf)
        _image_page(pdf, "Training dynamics (loss / eval CER)", RUNS / "curves.png")
        _image_page(pdf, "Per-layer weight-delta profiles & heatmap",
                    RUNS / "delta_profiles.png")
        layer_swap_pages(pdf)
        for png in sorted(RUNS.glob("*/eval/logit_lens.png")):
            _image_page(pdf, f"Logit-lens depth profile — {png.parent.parent.name}", png)
        # convergence tables as text
        conv = sorted(RUNS.glob("convergence_*.csv"))
        if conv:
            txt = []
            for f in conv:
                df = pd.read_csv(f)
                sp = df["norm_a"].rank().corr(df["norm_b"].rank())
                txt.append(f"{f.stem.replace('convergence_', '').replace('__', ' vs ')}:\n"
                           f"  mean cosine {df.cosine.mean():.3f}, Spearman {sp:.2f}")
            _text_page(pdf, "Cross-method convergence of weight deltas",
                       "\n\n".join(txt))
        per_run_appendix(pdf)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
