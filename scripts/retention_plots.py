"""Bidimensional recall-vs-retention plots for the cross-checkout report.

Two axes, both bounded and low-variance (the 2026-07-07 metric switch):
  x = recall on the run's trained corpus (exact-continuation match, memorization)
  y = capability retained vs the epoch-0 teacher (ARC-Easy retained ratio)

Because only FINAL checkpoints exist (the trainer saves once at run end), this
is an endpoint Pareto cloud -- one point per run -- not a per-epoch path.  A
true per-epoch trajectory needs retention logged during training (an in-loop
ARC pass); until then the honest artifact is this endpoint comparison across
the three method families.

Colour encodes model family/size and marker shape encodes the loss/lens/readout
kind.  Epoch-0 teacher rows are plotted as stars at retained=1 / log-damage=0.
Reads runs/retention_index.csv (scripts/retention_index.py).

Usage: python scripts/retention_plots.py [--index runs/retention_index.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito plus neutrals.  Primary encoding is model, not checkout.
MODEL_COLORS = {
    "Qwen3-0.6B": "#0072B2",
    "Qwen3-1.7B": "#56B4E9",
    "Qwen3-4B": "#009E73",
    "Qwen3-8B": "#CC79A7",
    "Qwen3-14B": "#E69F00",
    "Qwen3.6-27B": "#D55E00",
    "gpt-oss-20b": "#000000",
    "gpt-oss-120b": "#666666",
    "gemma-4-26B-A4B": "#009E73",
    "gemma-4-31B": "#F0E442",
    "ALIA-40b-fc-2606": "#CC79A7",
}
LENS_MARKERS = {
    "epoch0": "*",
    "frozen_vocab": "o",
    "lens_kl": "X",
    "hidden_match": "s",
    "logit_kd": "D",
    "teacher_kl": "D",
    "nmse": "s",
    "l2mse": "P",
    "vocab_mse": "o",
}
SOURCE_ALPHA = {"layerwise": 0.82, "classical-kd": 0.62, "multigpu": 0.62}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"


def _recall_target(row: pd.Series) -> float | None:
    """Exact-continuation recall on the run's own trained corpus."""
    name = str(row.get("run", "")).lower()
    corpus = str(row.get("corpus_family", "")).lower()
    cerv = ("quijote" in corpus and "machado+" not in corpus) or any(
        k in name for k in ("q_ch1", "cervantes", "quijote", "ch1")
    )
    val = row.get("recall_cont_cervantes") if cerv else row.get("recall_cont_machado")
    if val is None or pd.isna(val):
        # fall back to whichever probe exists
        for c in ("recall_cont_machado", "recall_cont_cervantes"):
            if c in row and pd.notna(row[c]):
                return float(row[c])
        return None
    return float(val)


def _param_size(model: str) -> float:
    """Marker area by rough parameter count (visual only)."""
    m = str(model)
    for tag, size in (("0.6B", 30), ("1.7B", 55), ("4B", 90), ("8B", 130),
                      ("14B", 180), ("20b", 200), ("27b", 240), ("30B", 260),
                      ("31B", 260), ("40b", 300), ("120b", 380), ("122B", 380)):
        if tag.lower() in m.lower():
            return size
    return 70


def _model_key(row: pd.Series) -> str:
    label = str(row.get("model_label") or row.get("model") or "unknown")
    if "Qwen3-0.6B" in label:
        return "Qwen3-0.6B"
    if "Qwen3-1.7B" in label:
        return "Qwen3-1.7B"
    if "Qwen3-4B" in label:
        return "Qwen3-4B"
    if "Qwen3-8B" in label:
        return "Qwen3-8B"
    if "Qwen3-14B" in label:
        return "Qwen3-14B"
    if "Qwen3.6-27B" in label:
        return "Qwen3.6-27B"
    if "gpt-oss-20" in label:
        return "gpt-oss-20b"
    if "gpt-oss-120" in label:
        return "gpt-oss-120b"
    if "gemma-4-26" in label:
        return "gemma-4-26B-A4B"
    if "gemma-4-31" in label:
        return "gemma-4-31B"
    if "ALIA" in label:
        return "ALIA-40b-fc-2606"
    return label


def _pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-right Pareto frontier (maximize both recall and retention)."""
    pts = sorted(set(points), key=lambda p: (-p[0], -p[1]))
    front, best_y = [], -1e9
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return sorted(front)


def scatter(df: pd.DataFrame, y_col: str, y_label: str, title: str,
            ax, invert_y: bool = False, logy: bool = False, show_legend: bool = False) -> None:
    ax.set_facecolor("white")
    plotted_models = set()
    present_models = []
    for _, r in df.iterrows():
        x = _recall_target(r)
        y = r.get(y_col)
        if x is None or y is None or pd.isna(y):
            continue
        model = _model_key(r)
        lens = str(r.get("lens_kind") or r.get("loss_kind") or "unknown")
        epoch0 = str(r.get("checkpoint_kind")) == "epoch0"
        color = MODEL_COLORS.get(model, "#999999")
        marker = "*" if epoch0 else LENS_MARKERS.get(lens, "v")
        size = 180 if epoch0 else _param_size(r.get("model"))
        alpha = 0.95 if epoch0 else SOURCE_ALPHA.get(str(r.get("source")), 0.7)
        edge = "#111111" if epoch0 else ("#111111" if r.get("source") != "layerwise" else "white")
        lw = 1.0 if epoch0 else (0.9 if r.get("source") != "layerwise" else 0.5)
        label = model if model not in plotted_models else None
        if model not in plotted_models:
            present_models.append((model, color))
            plotted_models.add(model)
        ax.scatter([x], [float(y)], s=size, c=color, marker=marker, alpha=alpha,
                   edgecolors=edge, linewidths=lw, label=label, zorder=4 if epoch0 else 3)
    # Pareto frontier over ALL points (method-agnostic best tradeoff)
    allpts = []
    for _, r in df.iterrows():
        x = _recall_target(r); y = r.get(y_col)
        if x is not None and y is not None and not pd.isna(y):
            allpts.append((x, float(y)))
    if len(allpts) >= 3 and not invert_y:
        fr = _pareto(allpts)
        if len(fr) >= 2:
            ax.plot([p[0] for p in fr], [p[1] for p in fr], color=MUTED,
                    lw=1.2, ls="--", zorder=2, label="Pareto frontier")
    ax.set_xlabel("recall — exact continuation on trained corpus (higher = better)",
                  fontsize=8, color=INK)
    ax.set_ylabel(y_label, fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK, weight="bold")
    if logy:
        ax.set_yscale("log")
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    if show_legend:
        model_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markeredgecolor="#222222", markersize=6, label=model, linestyle="")
            for model, color in present_models
        ]
        shape_handles = []
        for lab, marker in [("epoch 0", "*"), ("vocab_mse / frozen vocab", "o"),
                            ("lens_kl", "X"), ("hidden match", "s"),
                            ("KD / teacher KL", "D")]:
            shape_handles.append(plt.Line2D([0], [0], marker=marker, color="w",
                                            markerfacecolor="#777777",
                                            markeredgecolor="#222222",
                                            markersize=7, label=lab, linestyle=""))
        leg1 = ax.legend(handles=model_handles, fontsize=6.0, framealpha=0.95,
                         loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         title="model colour")
        ax.add_artist(leg1)
        ax.legend(handles=shape_handles, fontsize=6.0, framealpha=0.95,
                  loc="upper left", bbox_to_anchor=(1.02, 0.48), title="shape")


def make_figure(index: Path, out: Path) -> None:
    df = pd.read_csv(index)
    df = df[df.get("has_retention", False) == True].copy()  # noqa: E712
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.2))
    fig.suptitle(
        "Recall vs capability retention — endpoint Pareto across method families",
        fontsize=12, weight="bold", color=INK,
    )
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no retention.json yet\nrun scripts/retention_eval.py",
                    ha="center", va="center", fontsize=9, color=MUTED)
            ax.axis("off")
    else:
        scatter(df, "arc_retained", "ARC-Easy retained (teacher/original = 1)",
                "Standard-benchmark capability retained", axes[0])
        y_col = "wikitext_log_ppl_ratio" if "wikitext_log_ppl_ratio" in df else "wikitext_ppl_ratio"
        scatter(df, y_col, "WikiText-2 damage: log(ppl / teacher ppl) (teacher/original = 0)",
                "Language-model damage (log ratio)", axes[1], invert_y=True, show_legend=True)
    fig.text(0.5, 0.005,
             f"{len(df)} rows with retention evidence, including epoch-0 teacher anchors · "
             "colour = model · shape = loss/lens/readout kind · marker size ~ model params",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.84, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out} ({len(df)} runs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="runs/retention_index.csv")
    ap.add_argument("--out", default="runs/recall_retention.png")
    args = ap.parse_args()
    make_figure(Path(args.index), Path(args.out))


if __name__ == "__main__":
    main()
