"""Per-layer training-loss dynamics, one figure per run.

Reads runs/<run>/metrics.jsonl "train" entries' per_layer arrays,
epoch-averages them, and renders all layers as one panel: depth on a
sequential ramp (light = shallow, dark = deep — depth is ordered, so a
single-hue ramp is the correct encoding), log-y (losses span decades).
Outputs:
  runs/<run>/eval/layer_losses.png
  runs/<run>/eval/layer_losses_heatmap.png
  runs/<run>/eval/layer_losses.csv

Usage: python scripts/layer_loss_plots.py [--runs GLOB] [--force]
"""

import argparse
import fnmatch
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
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
    rows = []
    for L in range(n_layers):
        ys = []
        for e in epochs:
            vals = [pl[L] for pl in per_epoch[e] if len(pl) > L and pl[L] == pl[L]]
            mean = sum(vals) / len(vals) if vals else float("nan")
            ys.append(mean)
            rows.append({"run": run_dir.name, "epoch": e + 1, "layer": L + 1,
                         "loss": mean, "n_logs": len(vals)})
        series.append(ys)
    df = pd.DataFrame(rows)

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
    fig.savefig(out, dpi=220)
    plt.close(fig)

    csv = out.parent / "layer_losses.csv"
    df.to_csv(csv, index=False)

    mat = df.pivot(index="layer", columns="epoch", values="loss")
    finite = [v for v in mat.to_numpy().ravel() if v == v and v > 0]
    if not finite:
        return True
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis",
                   norm=colors.LogNorm(vmin=max(min(finite), 1e-8),
                                       vmax=max(finite)))
    ax.set_xlabel("epoch")
    ax.set_ylabel("layer")
    ax.set_title(f"{run_dir.name} — layer-loss heatmap", fontsize=10)
    ax.set_xticks(range(len(mat.columns)), mat.columns, fontsize=7)
    step = max(1, len(mat.index) // 12)
    ax.set_yticks(range(0, len(mat.index), step),
                  [str(v) for v in mat.index[::step]], fontsize=7)
    fig.colorbar(im, ax=ax, label="loss (log color)")
    fig.tight_layout()
    fig.savefig(out.parent / "layer_losses_heatmap.png", dpi=240)
    plt.close(fig)
    return True


def _regime(run_dir):
    """Batching-regime label from the run's own config snapshot.

    The 2026-07-11 speed flip (owner decision) forked the live loss grid:
    completed arms ran `item` B=1, later arms `bucketed` B=4 with batched
    eval — bf16 kernel-shape numerics differ slightly across regimes, so
    every cross-arm table must carry the label; a silent regime column is
    exactly the confound class this repo keeps re-learning.
    """
    try:
        import yaml

        raw = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
        train = raw.get("train", {}) or {}
        ev = raw.get("eval", {}) or {}
        return (f"{train.get('batching', 'item')}"
                f"_b{train.get('micro_batch', 1)}"
                f"_evalb{ev.get('generation_batch', 1)}")
    except Exception:  # noqa: BLE001 - manifest stays best-effort
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--no-manifest",
        action="store_true",
        help="write per-run figures without replacing the shared campaign manifest",
    )
    args = ap.parse_args()
    done = 0
    manifest = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or not fnmatch.fnmatch(run_dir.name, args.runs):
            continue
        out = run_dir / "eval" / "layer_losses.png"
        if out.exists() and not args.force:
            csv = out.parent / "layer_losses.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                if not df.empty:
                    last_epoch = int(df["epoch"].max())
                    last = df[df["epoch"] == last_epoch]
                    finite = last["loss"].dropna()
                    manifest.append({
                        "run": run_dir.name,
                        "regime": _regime(run_dir),
                        "epochs": int(df["epoch"].max()),
                        "layers": int(df["layer"].max()),
                        "rows": len(df),
                        "final_mean_loss": float(finite.mean()) if len(finite) else math.nan,
                        "final_min_loss": float(finite.min()) if len(finite) else math.nan,
                        "final_max_loss": float(finite.max()) if len(finite) else math.nan,
                        "line_plot": str(out.relative_to(RUNS.parent)),
                        "heatmap": str((out.parent / "layer_losses_heatmap.png").relative_to(RUNS.parent)),
                        "csv": str(csv.relative_to(RUNS.parent)),
                    })
            continue
        if plot_run(run_dir, out):
            done += 1
            csv = out.parent / "layer_losses.csv"
            df = pd.read_csv(csv)
            last_epoch = int(df["epoch"].max())
            last = df[df["epoch"] == last_epoch]
            finite = last["loss"].dropna()
            manifest.append({
                "run": run_dir.name,
                "regime": _regime(run_dir),
                "epochs": int(df["epoch"].max()),
                "layers": int(df["layer"].max()),
                "rows": len(df),
                "final_mean_loss": float(finite.mean()) if len(finite) else math.nan,
                "final_min_loss": float(finite.min()) if len(finite) else math.nan,
                "final_max_loss": float(finite.max()) if len(finite) else math.nan,
                "line_plot": str(out.relative_to(RUNS.parent)),
                "heatmap": str((out.parent / "layer_losses_heatmap.png").relative_to(RUNS.parent)),
                "csv": str(csv.relative_to(RUNS.parent)),
            })
    if manifest and not args.no_manifest:
        mf = pd.DataFrame(manifest).sort_values(["run"])
        mf.to_csv(RUNS / "layer_loss_manifest.csv", index=False)
        (RUNS / "layer_loss_manifest.md").write_text(
            "# Layer-Loss Artifacts\n\n"
            + mf.to_markdown(index=False)
            + "\n",
            encoding="utf-8",
        )
    print(f"layer-loss figures written: {done}")


if __name__ == "__main__":
    main()
