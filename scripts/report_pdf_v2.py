"""Small, dependency-light PDF renderer for report-v2 artifacts.

The Markdown report remains the source and hyperlink-friendly representation.
This module creates an offline-readable PDF using only Matplotlib, already a
mandatory report-v2 dependency.  It deliberately does not rely on a browser,
TeX, Pandoc, or a node-specific system package.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _cell(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        if value and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Append readable A4 text pages, flowing long lines onto new pages."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.06, .955, title, ha="left", va="top", fontsize=16, weight="bold")
    y = .91
    for line in lines:
        if not line:
            y -= .018
            continue
        wrapped = textwrap.wrap(str(line), width=105,
                                 break_long_words=False,
                                 break_on_hyphens=False) or [""]
        for part in wrapped:
            fig.text(.07, y, part, ha="left", va="top", fontsize=9.3)
            y -= .017
            if y < .06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = .94
    pdf.savefig(fig)
    plt.close(fig)


def _table_pages(pdf: PdfPages, title: str, frame: pd.DataFrame,
                 columns: list[str], rows_per_page: int = 18) -> None:
    """Append numeric table pages, preserving an explicit missing state."""
    if frame.empty:
        _text_page(pdf, title, ["Missing."])
        return
    available = [column for column in columns if column in frame.columns]
    if not available:
        _text_page(pdf, title, ["Missing expected columns: " + ", ".join(columns)])
        return
    for start in range(0, len(frame), rows_per_page):
        chunk = frame.iloc[start:start + rows_per_page].loc[:, available]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        suffix = "" if start == 0 else " (continued)"
        ax.set_title(title + suffix, loc="left", fontsize=14, pad=18)
        table = ax.table(
            cellText=[[_cell(value) for value in row]
                      for row in chunk.itertuples(index=False, name=None)],
            colLabels=available,
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.35)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _image_page(pdf: PdfPages, title: str, image: Path) -> None:
    if not image.is_file():
        _text_page(pdf, title, [f"Missing figure: {image.name}"])
        return
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.imshow(plt.imread(image))
    ax.set_title(title, loc="left", fontsize=14, pad=12)
    ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def write_individual_pdf(
    path: Path,
    *,
    title: str,
    identity: list[str],
    learning: pd.DataFrame,
    output_eval: pd.DataFrame,
    recall: pd.DataFrame,
    standard: pd.DataFrame,
    delta: pd.DataFrame,
    coverage: list[str],
    figures: list[tuple[str, Path]],
) -> Path:
    """Write one atomic report-v2 PDF atomically.

    The resulting PDF has identity/provenance, the recall and standard-damage
    tables, coverage, and every figure referenced by the corresponding
    Markdown report.  Missing figures are rendered as missing, never omitted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    with PdfPages(tmp) as pdf:
        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = "selfupdate layerwise reporting"
        _text_page(pdf, title, identity)
        _table_pages(
            pdf,
            "Learning outcome: next phrase, previous phrase, and cloze",
            learning,
            ["measure", "epoch_zero", "maximum", "max_epoch", "max_delta",
             "at_best_overall", "delta_at_best_overall", "final", "final_delta"],
            rows_per_page=8,
        )
        _text_page(
            pdf,
            "Teacher-output evaluation contract",
            [
                "CE-eval-loss and KL-eval-loss are evaluation-only metrics.",
                "They are NEVER training losses, are NEVER passed to backward, "
                "and have optimizer weight zero.",
                "They cover every teacher-realized answer token in the whole "
                "training-set traversal once per completed epoch; this is not "
                "a validation subset.",
                "Values are streaming pre-write measurements at each sample "
                "visit, not a separate frozen-checkpoint pass.",
            ],
        )
        _table_pages(
            pdf,
            "Whole-training-set CE-eval-loss and KL-eval-loss",
            output_eval,
            ["epoch", "CE-eval-loss", "KL-eval-loss",
             "answer_token_count", "dataset_item_count",
             "validation_subset", "evaluation_only", "used_for_backward",
             "optimizer_weight"],
            rows_per_page=12,
        )
        historical = "cer" in recall.columns and recall["cer"].notna().any()
        recall_columns = (["epoch", "corpus", "cer", "line_exact"]
                          if historical else
                          ["epoch", "corpus", "next_acc", "prev_acc",
                           "cloze_acc", "overall_word_acc"])
        recall_title = ("Historical inline recall (CER and line exactness)"
                        if historical else
                        "Recall by corpus (epoch 0 is pre-training)")
        _table_pages(pdf, recall_title, recall, recall_columns)
        standard_columns = ["epoch", "items_per_task", "macro_accuracy",
                            "epoch0_delta",
                            *sorted(c for c in standard.columns
                                    if c.startswith("accuracy_")),
                            "worst_task", "worst_delta"]
        _table_pages(pdf, "Standard-benchmark damage", standard,
                     standard_columns)
        _table_pages(pdf, "Most-modified layers (final checkpoint)", delta,
                     ["layer", "relative_l2"])
        _text_page(pdf, "Coverage and provenance", coverage)
        for figure_title, image in figures:
            _image_page(pdf, figure_title, image)
    tmp.replace(path)
    return path


def write_grouped_pdf(
    path: Path,
    *,
    title: str,
    inclusion: list[str],
    eligible: pd.DataFrame,
    coverage: pd.DataFrame,
    notes: list[str],
    figures: list[tuple[str, Path]],
) -> Path:
    """Write a compact grouped report without widening scientific inclusion.

    ``eligible`` is the strictly local evidence table.  ``coverage`` is a
    provenance roster only and may contain incomplete or uncertified runs.
    Keeping the two inputs separate prevents report-pending checkpoints from
    silently becoming frontier evidence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    with PdfPages(tmp) as pdf:
        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = "selfupdate layerwise reporting"
        _text_page(pdf, title, inclusion)
        _table_pages(
            pdf,
            "Eligible strictly local runs",
            eligible,
            ["run", "model", "loss", "optimizer", "lr", "final_recall",
             "standard_damage", "elapsed_minutes"],
            rows_per_page=14,
        )
        _table_pages(
            pdf,
            "All discovered campaign runs (provenance only)",
            coverage,
            ["run", "status", "model", "loss", "optimizer", "lr"],
            rows_per_page=20,
        )
        _text_page(pdf, "Missing artifacts and certification", notes)
        for figure_title, image in figures:
            _image_page(pdf, figure_title, image)
    tmp.replace(path)
    return path
