"""Cross-checkout final report: runs/cross_report.pdf.

Combines the three method families into one document, leaving each checkout's
own report.py untouched:
  layerwise    - forward layerwise distillation (this checkout)
  classical-kd - naive/classical KD baselines (../selfupdate_kd)
  multigpu     - big-model layerwise on sharded GPUs (../selfupdate_multi)

The deterioration axis is the new standard, low-variance battery (ARC-Easy
accuracy + WikiText-2 perplexity) rather than the retired noisy held-out
general-CE canary; recall is reported both as recite CER and as exact-match
continuation/cloze/censor probes against the original Machado / Cervantes text.

Reads runs/retention_index.csv (scripts/retention_index.py) and the figure
runs/recall_retention.png (scripts/retention_plots.py); regenerate those first.

Usage: python scripts/cross_report.py [--out runs/cross_report.pdf]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

RUNS = Path("runs")
INDEX = RUNS / "retention_index.csv"
FIGURE = RUNS / "recall_retention.png"

SOURCE_ORDER = ["layerwise", "classical-kd", "multigpu"]
SOURCE_BLURB = {
    "layerwise": "Forward layerwise distillation (this checkout). Every block updated "
                 "with uniform k-deep credit; behavioral readout teacher-sourced.",
    "classical-kd": "Classical/naive KD baselines (../selfupdate_kd): full fine-tune and "
                    "LoRA KL-distillation of the top layers.",
    "multigpu": "Big-model layerwise on sharded GPUs (../selfupdate_multi): gpt-oss-120B, "
                "Qwen3.5-122B, Gemma-4, ALIA-40B.",
}


def _text_page(pdf, title, body, fontsize=9):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.06, 0.93, title, fontsize=16, weight="bold")
    fig.text(0.06, 0.88, body, fontsize=fontsize, family="monospace", va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf, title, png: Path):
    if not png.exists():
        _text_page(pdf, title, f"{png} is missing.\nRun scripts/retention_plots.py first.")
        return
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.95, title, fontsize=14, weight="bold")
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.85])
    ax.imshow(mpimg.imread(str(png)))
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def summary_text(df: pd.DataFrame) -> str:
    n_ret = int(df.get("has_retention", pd.Series(dtype=bool)).sum())
    n_rec = int(df.get("has_recite", pd.Series(dtype=bool)).sum())
    lines = [
        "Self-distillation of context — cross-checkout comparison.",
        "",
        "Method families compared (one row per run, best-effort artifacts):",
    ]
    for src in SOURCE_ORDER:
        n = int((df["source"] == src).sum())
        lines.append(f"  [{src}]  {n} runs")
        lines.append("      " + SOURCE_BLURB[src])
    lines += [
        "",
        "Metric switch (owner directive, 2026-07-07): the tiny held-out general",
        "cross-entropy (log-loss) 'destruction' canary was measured to be",
        "noise-dominated and near-flat across training. It is replaced by two",
        "standard, bounded, low-variance measurements, scored in a single",
        "teacher-forced batched pass so only model LOADING costs time:",
        "  DESTRUCTION (capability retained vs the epoch-0 teacher of the base):",
        "    - ARC-Easy accuracy on a fixed cached subset (option log-likelihood)",
        "    - WikiText-2 validation perplexity (quantization-style damage)",
        "  RECALL (memorization of the original text; eval-only, never trained):",
        "    - exact continuation + interior-word cloze (Machado verse)",
        "    - next-sentence continuation + multi-word paragraph censor (Cervantes)",
        "",
        f"Coverage: {len(df)} runs total; {n_ret} with the new retention battery,",
        f"{n_rec} with recitation CER. Runs still needing the battery are listed",
        "on the coverage page; rerun scripts/retention_eval.py to fill them.",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Backbone: runs/retention_index.csv.",
    ]
    return "\n".join(lines)


def _table_page(pdf, title, note, df: pd.DataFrame, cols: list[str], fontsize=6.0):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.04, 0.95, title, fontsize=14, weight="bold")
    if note:
        fig.text(0.04, 0.915, note, fontsize=7)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.86])
    ax.axis("off")
    if df.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", fontsize=10)
    else:
        tbl = ax.table(cellText=df[cols].values.tolist(), colLabels=cols,
                       loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.auto_set_column_width(range(len(cols)))
        tbl.scale(1.0, 1.15)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#eeeeee")
            elif c <= 2:
                cell.set_text_props(ha="left")
    pdf.savefig(fig)
    plt.close(fig)


def _best_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Best retention row per (source, model), ranked by ARC retained."""
    ev = df[df["has_retention"] == True].copy()  # noqa: E712
    if ev.empty:
        return ev
    ev["arc_retained_n"] = pd.to_numeric(ev["arc_retained"], errors="coerce")
    ev = ev.sort_values("arc_retained_n", ascending=False)
    best = ev.groupby(["source", "model"], as_index=False).head(1)
    out = pd.DataFrame({
        "source": best["source"],
        "model": best["model"].astype(str).str.replace("Qwen/", "").str.replace("openai/", ""),
        "run": best["run"].astype(str).str.slice(0, 30),
        "recall_cer": best["recall_cer"].map(lambda v: _fmt(v)),
        "cont_mach": best["recall_cont_machado"].map(lambda v: _fmt(v)),
        "cloze_mach": best["recall_cloze_machado"].map(lambda v: _fmt(v)),
        "censor_cerv": best["recall_censor_cervantes"].map(lambda v: _fmt(v)),
        "arc_acc": best["arc_acc"].map(lambda v: _fmt(v)),
        "arc_retain": best["arc_retained"].map(lambda v: _fmt(v)),
        "wiki_ratio": best["wikitext_ppl_ratio"].map(lambda v: _fmt(v, 2)),
    })
    order = {s: i for i, s in enumerate(SOURCE_ORDER)}
    out = out.assign(_o=out["source"].map(order)).sort_values(["_o", "model"]).drop(columns="_o")
    return out


def _coverage_page(pdf, df: pd.DataFrame):
    lines = ["Runs WITHOUT the new retention battery (need scripts/retention_eval.py):", ""]
    missing = df[df["has_retention"] != True]  # noqa: E712
    for src in SOURCE_ORDER:
        names = missing[missing["source"] == src]["run"].tolist()
        lines.append(f"[{src}] {len(names)} missing")
        for i in range(0, len(names), 3):
            lines.append("   " + "  ".join(names[i:i + 3]))
        lines.append("")
    _text_page(pdf, "Retention coverage — remaining work", "\n".join(lines), fontsize=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--out", default="runs/cross_report.pdf")
    args = ap.parse_args()

    if not Path(args.index).exists():
        raise SystemExit(f"{args.index} missing; run scripts/retention_index.py first")
    df = pd.read_csv(args.index)
    for col in ("has_retention", "has_recite"):
        if col in df:
            df[col] = df[col].astype(bool)

    with PdfPages(args.out) as pdf:
        _text_page(pdf, "Self-distillation of context — cross-checkout report",
                   summary_text(df), fontsize=9)
        _image_page(pdf, "Recall vs capability retention (bidimensional)", FIGURE)
        _table_page(pdf, "Best retention row per method family and model",
                    "Recall: CER (lower better) and exact-match probes (higher better). "
                    "Retention: ARC retained & WikiText ppl ratio vs epoch-0 teacher.",
                    _best_by_group(df),
                    ["source", "model", "run", "recall_cer", "cont_mach", "cloze_mach",
                     "censor_cerv", "arc_acc", "arc_retain", "wiki_ratio"])
        _coverage_page(pdf, df)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
