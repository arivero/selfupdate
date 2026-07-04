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




def _dest(run):
    d = json.load(open(RUNS / run / "eval" / "destruction.json"))
    b = json.load(open(RUNS / "destruction" / "base_0p6b" / "destruction.json"))
    cats = {c: d["probe_battery"]["categories"][c]["mean_ce"]
            - b["probe_battery"]["categories"][c]["mean_ce"]
            for c in d["probe_battery"]["categories"]}
    return {"worst_cat": max(cats.values()),
            "hs": 100 * (d["benchmarks"]["hellaswag"]["accuracy"]
                         - b["benchmarks"]["hellaswag"]["accuracy"]),
            "intr": 100 * d["intrusion"]["hit_rate"]}


# ---- Fig 5: the saturation surface (rung x anchors) ---------------------
def fig5():
    rungs = [("ch1", "q_ch1_0p6b_rag", None),
             ("ch1 (200ep)", "q_ch1_ext_0p6b_rag", None),
             ("ch4", "q_ch4_0p6b_rag", "q_ch4_av2_0p6b_rag"),
             ("ch8", "q_ch8_0p6b_rag", "q_ch8_av2_0p6b_rag"),
             ("ch16 (40ep)", "q_ch16_ext_0p6b_rag", "q_ch16_av2_0p6b_rag")]
    rows = []
    for name, v1, v2 in rungs:
        for anch, run in (("poetry-only", v1), ("multi-genre", v2)):
            if run is None or not (RUNS / run / "eval" / "destruction.json").exists():
                continue
            r = json.load(open(RUNS / run / "eval" / "recite.json"))
            rows.append({"rung": name, "anchors": anch,
                         "recall": r.get("cer_flat", r["cer"]), **_dest(run)})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    metrics = [("recall", "cer_flat (recall)", 0.10),
               ("worst_cat", "worst probe-category ΔCE (nats)", 0.5),
               ("intr", "intrusion rate (%)", 10)]
    names = list(dict.fromkeys(df.rung))
    w = 0.36
    for ax, (m, lab, thr) in zip(axes, metrics):
        for j, anch in enumerate(("poetry-only", "multi-genre")):
            sub = df[df.anchors == anch].set_index("rung")
            xs, ys = [], []
            for i, n in enumerate(names):
                if n in sub.index:
                    xs.append(i + (j - 0.5) * w)
                    ys.append(sub.loc[n, m])
            ax.bar(xs, ys, width=w - 0.04,
                   color=C["vocab_mse"] if j else C["neutral"],
                   label=anch if ax is axes[0] else None)
        ax.axhline(thr, color="#B22222", lw=0.9, ls="--")
        ax.set_xticks(range(len(names)), names, fontsize=6.5, rotation=15)
        ax.set_title(lab, fontsize=8, loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Saturation surface at 0.6B: recall never fails by ch16 — the destruction "
                 "envelope is the binding constraint (dashed = pre-committed thresholds)",
                 fontsize=8.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "fig5_saturation.png")
    plt.close(fig)


# ---- Fig 6: head taxonomy ----------------------------------------------
def fig6():
    df = pd.read_csv(RUNS / "attention_probe_0.6B" / "heads.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.8))
    colors = {"content": C["vocab_mse"], "grammar": C["l2mse"], "mixed": "#BBBBBB"}
    for kind, g in df.groupby("kind"):
        a.scatter(g.distance, g.priv_mass, s=10, alpha=0.75, lw=0,
                  c=colors[kind], label=f"{kind} ({len(g)})")
    a.set_xlabel("mean attention distance (tokens)", fontsize=8)
    a.set_ylabel("answer→privileged mass", fontsize=8)
    a.legend(fontsize=7, frameon=False)
    a.set_title("(a) 448 heads, answer positions", fontsize=9, loc="left")
    prof = df.groupby("layer").priv_mass.mean()
    b.plot(prof.index, prof.values, marker="o", ms=3, color=C["vocab_mse"])
    b.axvline(7, color=C["neutral"], lw=0.8, ls="--")
    b.axvspan(21, 24, color=C["neutral"], alpha=0.12, lw=0)
    b.text(7.3, prof.max() * 0.92, "L7\nintegration peak", fontsize=6.5)
    b.text(22.5, prof.max() * 0.92, "storage\nband", fontsize=6.5, ha="center")
    b.set_xlabel("layer", fontsize=8)
    b.set_ylabel("mean answer→privileged mass", fontsize=8)
    b.set_title("(b) retrieval attention lives mid-net", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_heads.png")
    plt.close(fig)


# ---- Fig 7: raw vs tuned lens ------------------------------------------
def fig7():
    raw = pd.read_csv(RUNS / "lw_i_vocab_strict_0p6b_rag" / "eval" / "logit_lens.csv")
    tuned = pd.read_csv(RUNS / "lw_i_vocab_strict_0p6b_rag" / "eval" / "logit_lens_tuned.csv")
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(raw.layer, raw.trained_logprob, color=C["neutral"], lw=1.5,
            marker="o", ms=2.5, label="raw lens, trained")
    ax.plot(raw.layer, raw.base_logprob, color=C["neutral"], lw=1.0, ls=":",
            label="raw lens, base")
    ax.plot(tuned.layer, tuned.trained_logprob, color=C["vocab_mse"], lw=1.5,
            marker="o", ms=2.5, label="tuned lens, trained")
    ax.plot(tuned.layer, tuned.base_logprob, color=C["vocab_mse"], lw=1.0, ls=":",
            label="tuned lens, base")
    ax.set_xlabel("layer", fontsize=8)
    ax.set_ylabel("gold-token logprob (student input)", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncols=2)
    ax.set_title("Calibration lifts every layer ~4-5 nats, but the trained-base gap still\n"
                 "opens only at L22+: deep storage is not secretly readable (strict arm)",
                 fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_tuned_lens.png")
    plt.close(fig)


# ---- Fig 8: the thinking channel ---------------------------------------
def fig8():
    arms = [("RAG mode\n(anchordiv)", "lw_m_anchordiv_0p6b_rag", C["neutral"]),
            ("whole-think\ncensored", "lw_n_thinkwhole_0p6b_rag", C["cosine"]),
            ("thinking\nselective", "lw_n_thinksel_0p6b_rag", C["vocab_mse"])]
    rows = []
    for name, run, col in arms:
        r = json.load(open(RUNS / run / "eval" / "recite.json"))
        rows.append({"name": name, "col": col,
                     "recall": r.get("cer_flat", r["cer"]), **_dest(run)})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.6))
    for ax, m, lab, thr in ((axes[0], "recall", "recall CER (lower better)", None),
                            (axes[1], "hs", "Δ HellaSwag (pts)", -5),
                            (axes[2], "intr", "intrusion rate (%)", 10)):
        ax.bar(range(len(df)), df[m], width=0.55, color=list(df.col))
        if thr is not None:
            ax.axhline(thr, color="#B22222", lw=0.9, ls="--")
        ax.set_xticks(range(len(df)), df.name, fontsize=7)
        ax.set_title(lab, fontsize=8, loc="left")
        for i, v in enumerate(df[m]):
            ax.text(i, v, f" {v:.3f}" if m == "recall" else f" {v:+.1f}" if m == "hs" else f" {v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("The thinking channel is the gentle channel: selective censoring wins recall "
                 "AND collateral (matched 333-spec arms, final recipe)",
                 fontsize=8.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "fig8_thinking.png")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skip {f.__name__}: missing {e.filename}")
    print("wrote", sorted(p.name for p in FIGS.glob("*.png")))
