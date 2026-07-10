"""Cross-model comparison matrix from runs/corpus.csv.

One figure, rows by model, columns by metric — the plot that prevents
overfitting conclusions to one rung (recommendations.md "Model Comparison
Matrix"). Only method-class runs feed the matrix; each model row shows its
best method run by full-corpus recall CER plus how many method runs back it.

Usage:
    python scripts/model_matrix.py [--corpus runs/corpus.csv]
        [--out runs/model_matrix.png] [--csv runs/model_matrix.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

COLUMNS = [
    ("full_eval_cer", "recall CER", True),
    ("full_eval_line_exact", "line exact", False),
    ("forgetting_delta_ce", "general-CE delta", True),
    ("hidden_share", "hidden share", False),
    ("train_min", "train minutes", True),
    ("vram_reserved_gb", "peak VRAM GB", True),
    ("items_seen", "items seen", None),  # informational, unshaded
]


def _f(row: dict, key: str):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def load_best_per_model(path: Path) -> list[dict]:
    rows = [r for r in csv.DictReader(path.open())
            if r.get("run_class") == "method"]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    best = []
    for model, runs in sorted(by_model.items()):
        scored = [r for r in runs if _f(r, "full_eval_cer") is not None]
        if not scored:
            continue
        top = min(scored, key=lambda r: _f(r, "full_eval_cer"))
        top = dict(top)
        top["method_runs"] = len(runs)
        best.append(top)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/corpus.csv")
    ap.add_argument("--out", default="runs/model_matrix.png")
    ap.add_argument("--csv", default="runs/model_matrix.csv")
    args = ap.parse_args()

    best = load_best_per_model(Path(args.corpus))
    if not best:
        sys.exit("no method-class runs with full-corpus eval in the corpus")

    with Path(args.csv).open("w") as f:
        keys = ["model", "run", "method_runs"] + [k for k, _, _ in COLUMNS]
        f.write(",".join(keys) + "\n")
        for r in best:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"wrote {args.csv} ({len(best)} models)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [f"{r['model'].split('/')[-1]}\n({r['run'][:28]}, n={r['method_runs']})"
              for r in best]
    ncol = len(COLUMNS)
    values = np.full((len(best), ncol), np.nan)
    for i, r in enumerate(best):
        for j, (key, _, _) in enumerate(COLUMNS):
            v = _f(r, key)
            if v is not None:
                values[i, j] = v

    fig, ax = plt.subplots(
        figsize=(2.1 * ncol, 0.62 * len(best) + 1.6))
    # column-normalized shading (lower_better flips the scale); NaN = grey
    shade = np.full_like(values, np.nan)
    for j, (_, _, lower_better) in enumerate(COLUMNS):
        col = values[:, j]
        ok = ~np.isnan(col)
        if lower_better is None or ok.sum() < 2:
            continue
        lo, hi = col[ok].min(), col[ok].max()
        norm = (col - lo) / (hi - lo) if hi > lo else col * 0
        shade[:, j] = 1 - norm if lower_better else norm
    ax.imshow(np.where(np.isnan(shade), 0.5, shade), cmap="RdYlGn",
              vmin=0, vmax=1, aspect="auto", alpha=0.55)
    for i in range(len(best)):
        for j in range(ncol):
            v = values[i, j]
            ax.text(j, i, "—" if np.isnan(v)
                    else (f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"),
                    ha="center", va="center", fontsize=8)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([label for _, label, _ in COLUMNS], fontsize=8)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_title("Best method run per model (runs/corpus.csv)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
