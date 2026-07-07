"""Bidimensional recall-vs-retention plots for the cross-checkout report.

Two axes, both bounded and low-variance (the 2026-07-07 metric switch):
  x = recall on the run's trained corpus (exact-continuation match, memorization)
  y = capability retained vs the epoch-0 teacher (ARC-Easy retained ratio)

Because only FINAL checkpoints exist (the trainer saves once at run end), this
is an endpoint Pareto cloud -- one point per run -- not a per-epoch path.  A
true per-epoch trajectory needs retention logged during training (an in-loop
ARC pass); until then the honest artifact is this endpoint comparison across
the three method families.

Colour encodes the method family (Okabe-Ito CVD-safe, fixed order) and marker
shape repeats it as a print/grayscale-safe secondary encoding.  Reads
runs/retention_index.csv (scripts/retention_index.py).

Usage: python scripts/retention_plots.py [--index runs/retention_index.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito, fixed order; marker repeats identity for grayscale/CVD safety.
SOURCE_STYLE = {
    "layerwise": {"color": "#0072B2", "marker": "o", "label": "layerwise (this)"},
    "classical-kd": {"color": "#E69F00", "marker": "s", "label": "classical-KD"},
    "multigpu": {"color": "#009E73", "marker": "^", "label": "multigpu"},
}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"


def _recall_target(row: pd.Series) -> float | None:
    """Exact-continuation recall on the run's own trained corpus."""
    name = str(row.get("run", "")).lower()
    cerv = any(k in name for k in ("q_ch1", "cervantes", "quijote", "ch1"))
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
            ax, invert_y: bool = False, logy: bool = False) -> None:
    ax.set_facecolor("white")
    for src, style in SOURCE_STYLE.items():
        sub = df[df["source"] == src]
        xs, ys, ss = [], [], []
        for _, r in sub.iterrows():
            x = _recall_target(r)
            y = r.get(y_col)
            if x is None or y is None or pd.isna(y):
                continue
            xs.append(x); ys.append(float(y)); ss.append(_param_size(r.get("model")))
        if xs:
            ax.scatter(xs, ys, s=ss, c=style["color"], marker=style["marker"],
                       alpha=0.8, edgecolors="white", linewidths=0.6,
                       label=f"{style['label']} (n={len(xs)})", zorder=3)
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
    ax.legend(fontsize=6.5, framealpha=0.9, loc="best")


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
        scatter(df, "arc_retained", "ARC-Easy retained (acc / teacher acc)",
                "Standard-benchmark capability retained", axes[0])
        scatter(df, "wikitext_ppl_ratio", "WikiText-2 ppl ratio vs teacher (log; lower = better)",
                "Language-model damage (quantization-style)", axes[1], invert_y=True, logy=True)
    fig.text(0.5, 0.005,
             f"{len(df)} runs with retention evidence · marker size ~ model params · "
             "colour+shape = method family (Okabe-Ito, CVD-safe)",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
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
