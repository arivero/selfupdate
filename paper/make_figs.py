"""Paper figures for paper1.md — reads live campaign artifacts from runs/.

Conventions (print-safe): Okabe-Ito CVD-safe palette with FIXED entity->color
assignment across all figures; one axis per panel; direct labels where few
series; recessive grid.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

# fixed entity colors (Okabe-Ito), consistent across every figure
C = {
    "vocab_mse": "#0072B2", "l2mse": "#D55E00", "nmse": "#009E73",
    "huber": "#CC79A7", "cosine": "#56B4E9", "lens_kl": "#E69F00",
    "neutral": "#7F7F7F", "highlight": "#0072B2",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def recite(run, sub="eval"):
    return json.load(open(RUNS / run / sub / "recite.json"))


def base_probes():
    return json.load(open(RUNS / "base-general-0.6B.json"))["per_text"]


# ---- Fig 1: loss sweep — recall vs forgetting, per loss ----------------
def fig1():
    arms = [  # (label key, run, loss entity)
        ("vocab_mse", "lw_i_vocab_0p6b_rag"),
        ("l2mse", "lw_i_l2mse_0p6b_rag"),
        ("huber", "lw_i_huber_0p6b_rag"),
        ("nmse", "lw_i_nmse_s43_0p6b_rag"),
        ("cosine", "lw_i_cosine_0p6b_rag"),
        ("lens_kl", "lw_i_lenskl_0p6b_rag"),
    ]
    base_mean = sum(base_probes()) / 4
    names, cers, dces = [], [], []
    for loss, run in arms:
        d = recite(run)
        names.append(loss)
        cers.append(d["cer"])
        dces.append(d["general"]["mean_ce"] - base_mean)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    y = range(len(names))[::-1]
    for ax, vals, title, xlab in ((a, cers, "recall", "full-corpus CER (lower = better)"),
                                  (b, dces, "forgetting", "Δ general CE vs base (nats)")):
        ax.barh(list(y), vals, height=0.62,
                color=[C[n] for n in names], edgecolor="none")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(xlab, fontsize=8)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.3f}" if v < 1 else f" {v:.2f}",
                    va="center", fontsize=7.5, color="#333")
    a.set_yticks(list(y), names)
    a.axvline(0.112, color=C["neutral"], lw=0.8, ls="--")
    a.text(0.118, max(y) + 0.55, "pre-campaign\nchampion (0.112)", fontsize=6.5,
           color=C["neutral"], va="top")
    fig.suptitle("Wave I loss sweep (0.6B, v2 data, summed + tail-CE k=4, matched 13.3k items)",
                 fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGS / "fig1_losses.png")
    plt.close(fig)


# ---- Fig 2: where the memory lives ------------------------------------
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    profs = [("vocab_mse", "lw_i_vocab_0p6b_rag"),
             ("l2mse", "lw_i_l2mse_0p6b_rag"),
             ("nmse", "lw_i_nmse_s43_0p6b_rag")]
    for loss, run in profs:
        df = pd.read_csv(RUNS / run / "eval" / "weight_deltas.csv")
        prof = df.groupby("layer")["rel_delta"].apply(lambda x: (x**2).mean() ** 0.5)
        a.plot(prof.index, prof.values, color=C[loss], lw=1.6, label=loss)
    a.axvspan(25, 28, color=C["neutral"], alpha=0.12, lw=0)
    a.text(26.5, a.get_ylim()[1] * 0.05, "tail\nwindow", fontsize=6.5,
           ha="center", color=C["neutral"])
    a.set_xlabel("layer", fontsize=8)
    a.set_ylabel("relative weight delta (RMS over modules)", fontsize=8)
    a.set_title("(a) where gradient mass lands", fontsize=9, loc="left")
    a.legend(fontsize=7.5, frameon=False)

    swaps = [("l2mse", "lw_i_l2mse_0p6b_rag"),
             ("huber", "lw_i_huber_0p6b_rag"),
             ("cosine", "lw_i_cosine_0p6b_rag")]
    for loss, run in swaps:
        df = pd.read_csv(RUNS / run / "eval" / "layer_swap.csv")
        b.plot(df.layer, df.ablate_cer, color=C[loss], lw=1.6,
               marker="o", ms=3, label=loss)
    b.axvspan(25, 28, color=C["neutral"], alpha=0.12, lw=0)
    b.set_xlabel("reverted layer", fontsize=8)
    b.set_ylabel("full-corpus CER after single-layer revert", fontsize=8)
    b.set_title("(b) causal necessity (ablate one layer)", fontsize=9, loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_layers.png")
    plt.close(fig)


# ---- Fig 3: catastrophic remembering & the anchor arc ------------------
def fig3():
    base = base_probes()
    probes = ["Bécquer\n(poetry ES)", "facts ES", "prose EN", "recipe ES"]
    arms = [("no anchor", "lw_k_tailonly_0p6b_rag", C["neutral"]),
            ("anchor-CE", "lw_k_tailonly_anchor_0p6b_rag", C["l2mse"]),
            ("anchor-KL", "lw_k_tailonly_anchorkl_0p6b_rag", C["vocab_mse"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    w = 0.26
    for i, (label, run, color) in enumerate(arms):
        pt = recite(run)["general"]["per_text"]
        d = [a - b for a, b in zip(pt, base)]
        xs = [x + (i - 1) * w for x in range(4)]
        ax.bar(xs, d, width=w - 0.03, color=color, label=label, edgecolor="none")
    ax.set_xticks(range(4), probes, fontsize=8)
    ax.set_ylabel("Δ CE vs base (nats)", fontsize=8)
    ax.set_title("Catastrophic remembering: damage peaks on the memorized genre's neighbor;\n"
                 "anchor-CE amplifies it, anchor-KL halves it (tail_only arms, recall ≈ equal)",
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_intrusion.png")
    plt.close(fig)


# ---- Fig 4: readout window capacity ------------------------------------
def fig4():
    def verses(run):
        x = json.load(open(RUNS / run / "eval" / "recite_long.json"))
        return next(m for m in x if m["mode"] == "self")["verses_until_first_error"]

    arms = [("champion (v2)", "lw_i_vocab_0p6b_rag", C["neutral"]),
            ("maieutic only", "lw_k_maieutic_0p6b_rag", C["neutral"]),
            ("anchor-KL only", "lw_k_anchorkl_0p6b_rag", C["neutral"]),
            ("both, k=4", "lw_k_final_0p6b_rag", C["l2mse"]),
            ("both, k=4, 2-phase", "lw_k_final2p_0p6b_rag", C["l2mse"]),
            ("both, k=8", "lw_k_final_k8_0p6b_rag", C["vocab_mse"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    names = [a[0] for a in arms]
    vals = [verses(a[1]) for a in arms]
    ax.bar(range(len(arms)), vals, width=0.6,
           color=[a[2] for a in arms], edgecolor="none")
    ax.axhline(715, color=C["neutral"], lw=0.8, ls=":")
    ax.text(-0.38, 726, "poem length (715)", fontsize=6.5, color=C["neutral"])
    for i, v in enumerate(vals):
        ax.text(i, v + 8, str(v), ha="center", fontsize=7.5, color="#333")
    ax.set_xticks(range(len(arms)), names, fontsize=7, rotation=12)
    ax.set_ylabel("verses until first error\n(self-chained)", fontsize=8)
    ax.set_ylim(0, 780)
    ax.set_title("Readout capacity: each addition is free alone;\ncombined they saturate k=4 — k=8 holds all three", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_capacity.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("wrote", sorted(p.name for p in FIGS.glob("*.png")))
