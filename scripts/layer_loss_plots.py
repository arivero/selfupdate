"""Per-layer training-loss dynamics, one figure per run.

Reads runs/<run>/metrics.jsonl "train" entries' per_layer arrays,
epoch-averages them, and renders all layers as one panel: depth on a
sequential ramp (light = shallow, dark = deep — depth is ordered, so a
single-hue ramp is the correct encoding), log-y (losses span decades).
Output: runs/<run>/eval/layer_losses.png

Usage: python scripts/layer_loss_plots.py [--runs GLOB] [--force]
"""

import argparse
import fnmatch
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

RUNS = Path(__file__).resolve().parent.parent / "runs"


def plot_run(run_dir: Path, out: Path) -> bool:
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return False
    per_epoch: dict[int, list[list[float]]] = defaultdict(list)
    for line in m.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "train" and d.get("per_layer"):
            per_epoch[d["epoch"]].append(d["per_layer"])
    if not per_epoch:
        return False
    epochs = sorted(per_epoch)
    n_layers = len(per_epoch[epochs[0]][0])
    series = []
    for L in range(n_layers):
        ys = []
        for e in epochs:
            vals = [pl[L] for pl in per_epoch[e] if len(pl) > L and pl[L] == pl[L]]
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
        series.append(ys)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    norm = colors.Normalize(vmin=1, vmax=n_layers)
    ramp = cm.viridis
    for L, ys in enumerate(series, start=1):
        ax.plot([e + 1 for e in epochs], ys, color=ramp(norm(L)),
                lw=1.1, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-layer loss (epoch mean, log)")
    ax.set_title(f"{run_dir.name} — layer-loss dynamics", fontsize=10)
    ax.grid(alpha=0.2, lw=0.5)
    sm = cm.ScalarMappable(norm=norm, cmap=ramp)
    fig.colorbar(sm, ax=ax, label="layer (depth)", pad=0.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    done = 0
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or not fnmatch.fnmatch(run_dir.name, args.runs):
            continue
        out = run_dir / "eval" / "layer_losses.png"
        if out.exists() and not args.force:
            continue
        if plot_run(run_dir, out):
            done += 1
    print(f"layer-loss figures written: {done}")


if __name__ == "__main__":
    main()
