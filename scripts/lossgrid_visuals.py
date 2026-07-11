"""Campaign-scoped visuals for the July-11 1.7B loss grid."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("runs")


def heatmap(rows, title, out, *, log=False):
    if not rows:
        return
    frame = pd.DataFrame(rows).set_index("run").sort_index()
    values = frame.to_numpy(float)
    # Row normalization answers where each run concentrates its signal while
    # retaining the raw CSVs for absolute comparisons across compatible losses.
    scale = values.max(axis=1, keepdims=True)
    scale[scale == 0] = 1
    values = values / scale
    fig, ax = plt.subplots(figsize=(11, max(5, .25 * len(frame))))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(frame)), frame.index, fontsize=6)
    ax.set_xticks(range(frame.shape[1]), frame.columns, fontsize=7)
    ax.set_xlabel("layer")
    ax.set_title(title + " (row-normalized)")
    fig.colorbar(im, ax=ax, label="fraction of run maximum")
    fig.tight_layout()
    fig.savefig(RUNS / out, dpi=220)
    plt.close(fig)


def main():
    loss_rows, delta_rows = [], []
    for run in sorted(RUNS.glob("a_lossgrid_*")):
        lp = run / "eval/layer_losses.csv"
        if lp.exists():
            df = pd.read_csv(lp)
            final = df[df.epoch == df.epoch.max()].set_index("layer").loss
            loss_rows.append({"run": run.name, **{f"L{int(k)}": v for k, v in final.items()}})
        wp = run / "eval/weight_deltas.csv"
        if wp.exists():
            df = pd.read_csv(wp)
            prof = (df.assign(v=df.rel_delta.astype(float) ** 2)
                      .groupby("layer").v.mean().pow(.5))
            delta_rows.append({"run": run.name, **{f"L{int(k)}": v for k, v in prof.items()}})
    heatmap(loss_rows, "Final per-layer training loss", "lossgrid_final_layer_loss.png")
    heatmap(delta_rows, "Per-layer parameter modification", "lossgrid_layer_modification.png")

    # Compact campaign-wide temporal appendix. Each panel preserves the
    # requested axes (x=epoch, y=loss) and one trace per layer without making
    # the PDF builder decode dozens of large PNGs simultaneously.
    temporal = []
    for run in sorted(RUNS.glob("a_lossgrid_*")):
        lp = run / "eval/layer_losses.csv"
        if lp.exists():
            temporal.append((run.name, pd.read_csv(lp)))
    if temporal:
        cols = 3
        rows = (len(temporal) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, 2.4 * rows)),
                                 squeeze=False)
        for ax, (name, df) in zip(axes.ravel(), temporal):
            for _, group in df.groupby("layer"):
                ax.plot(group.epoch, group.loss, alpha=.45, lw=.6)
            ax.set_yscale("log")
            ax.set_title(name.replace("a_lossgrid_1p7b_combined_", ""), fontsize=7)
            ax.set_xlabel("epoch", fontsize=7)
            ax.set_ylabel("loss", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=.15)
        for ax in axes.ravel()[len(temporal):]:
            ax.axis("off")
        fig.suptitle("Per-layer temporal loss curves (one trace per layer)")
        fig.tight_layout()
        fig.savefig(RUNS / "lossgrid_temporal_layer_losses.png", dpi=180)
        plt.close(fig)

    score = RUNS / "lossgrid_report.csv"
    if score.exists():
        df = pd.read_csv(score)
        df = df[(df.status == "complete") & df["recall_mean"].notna()
                & df["standard_worst_delta"].notna()]
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            for slide, group in df.groupby("slide"):
                ax.scatter(group.standard_worst_delta, group.recall_mean,
                           label=f"slide {slide}", alpha=.8)
            ax.axvline(0, color="black", lw=.7)
            ax.set_xlabel("worst standard-benchmark delta")
            ax.set_ylabel("mean recall across three corpora")
            ax.set_title("Recall–damage frontier")
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(RUNS / "lossgrid_recall_damage_frontier.png", dpi=220)
            plt.close(fig)
    print(f"loss-grid visuals: {len(loss_rows)} loss profiles, {len(delta_rows)} modification profiles")


if __name__ == "__main__":
    main()
