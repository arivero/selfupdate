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
    trained = df[df.get("checkpoint_kind", "trained") != "epoch0"] if "checkpoint_kind" in df else df
    n_ret = int(trained.get("has_retention", pd.Series(dtype=bool)).sum())
    n_rec = int(trained.get("has_recite", pd.Series(dtype=bool)).sum())
    n_epoch0 = int((df.get("checkpoint_kind", pd.Series(dtype=str)) == "epoch0").sum()) if "checkpoint_kind" in df else 0
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
        f"Coverage: {len(trained)} trained runs total; {n_ret} with the new retention",
        f"battery, {n_rec} with recitation CER. The index also includes {n_epoch0}",
        "epoch-0 teacher/original baseline rows, where ARC retained = 1 and",
        "WikiText log perplexity-ratio damage = 0. Runs still needing the battery",
        "are listed on the coverage page; rerun scripts/retention_eval.py to fill them.",
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
    if "checkpoint_kind" in ev:
        ev = ev[ev["checkpoint_kind"] != "epoch0"].copy()
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
        "wiki_log_damage": best["wikitext_log_ppl_ratio"].map(lambda v: _fmt(v)),
    })
    order = {s: i for i, s in enumerate(SOURCE_ORDER)}
    out = out.assign(_o=out["source"].map(order)).sort_values(["_o", "model"]).drop(columns="_o")
    return out


def head_taxonomy_pages(pdf):
    """What to memorize: the attention head taxonomy (distance x retrieval).

    Ported from the paper's head-taxonomy figure (paper/make_figs.py fig6) so
    the report carries the mechanistic answer to 'what are the things to
    memorize' in daydreaming memorisation.  Each layer x head is scored at the
    answer positions on the TEACHER view (privileged passage present):
    long-range + privileged-heavy heads are content/insight (worth absorbing);
    local, privileged-blind heads are grammar.  Reads whatever
    runs/attention_probe_*/heads.csv exist (no dependency on paper/ outputs)."""
    kind_color = {"content": "#0072B2", "grammar": "#E69F00", "mixed": "#BBBBBB"}
    for csv in sorted(RUNS.glob("attention_probe_*/heads.csv")):
        model_tag = csv.parent.name.replace("attention_probe_", "")
        df = pd.read_csv(csv)
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.94, f"What to memorize — attention head taxonomy ({model_tag})",
                 fontsize=15, weight="bold")
        fig.text(
            0.06, 0.90,
            "Teacher view (privileged passage present), answer positions. A head that reaches FAR "
            "(high distance) and lands ON the privileged block (high answer→privileged mass) is a "
            "content/insight head — it marks what the context considered worth absorbing. Local, "
            "privileged-blind heads are grammar. This is the memorization target that motivates "
            "hidden-primary (mid-net) storage, and it names what the recall probes should test.",
            fontsize=8, va="top", wrap=True,
        )
        a = fig.add_axes([0.08, 0.10, 0.40, 0.68])
        for kind, g in df.groupby("kind"):
            a.scatter(g["distance"], g["priv_mass"], s=14, alpha=0.75, lw=0,
                      c=kind_color.get(kind, "#888888"), label=f"{kind} ({len(g)})")
        a.set_xlabel("mean attention distance (tokens)", fontsize=9)
        a.set_ylabel("answer→privileged mass", fontsize=9)
        a.legend(fontsize=8, frameon=False, title="head kind")
        a.set_title(f"(a) {len(df)} heads — distance vs retrieval", fontsize=10, loc="left")
        a.grid(True, color="#dddddd", lw=0.5)

        b = fig.add_axes([0.57, 0.10, 0.38, 0.68])
        prof = df.groupby("layer")["priv_mass"].mean()
        b.plot(prof.index, prof.values, marker="o", ms=3.5, color="#0072B2")
        peak = int(prof.idxmax())
        b.axvline(peak, color="#666666", lw=0.8, ls="--")
        b.text(peak + 0.4, prof.max() * 0.9, f"L{peak}\nretrieval-mass peak", fontsize=7)
        b.set_xlabel("layer", fontsize=9)
        b.set_ylabel("mean answer→privileged mass", fontsize=9)
        b.set_title("(b) retrieval attention lives mid-net", fontsize=10, loc="left")
        b.grid(True, color="#dddddd", lw=0.5)
        pdf.savefig(fig)
        plt.close(fig)


def surprise_pages(pdf):
    """What to memorize, factor 2: surprise decomposition.

    A large surprise (student NLL - teacher NLL on the reference token) is ambiguous:
    it is a KNOWLEDGE GAP when the teacher resolves the token via its privileged
    context (attention on the privileged block) -> memorize it; it is an
    ATTENTION MISDIRECTION when the answer is in the shared context but the
    student mis-routes -> a router problem (blackbox vs router_aligned), not a
    memory target.  Reads runs/surprise_probe_*/tokens.csv."""
    color = {"knowledge_gap": "#0072B2", "misdirection": "#D55E00", "low_surprise": "#BBBBBB"}
    for csv in sorted(RUNS.glob("surprise_probe_*/tokens.csv")):
        tag = csv.parent.name.replace("surprise_probe_", "")
        df = pd.read_csv(csv)
        if "label" not in df:
            continue
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.94, f"What to memorize — surprise decomposition ({tag})",
                 fontsize=15, weight="bold")
        counts = df["label"].value_counts().to_dict()
        fig.text(
            0.06, 0.90,
            "Surprise = student-view NLL minus teacher-view NLL on each answer token "
            "(base model, privileged block removed vs present). Footprint over content heads, "
            "structural/sink tokens masked. High surprise splits into knowledge_gap "
            f"({counts.get('knowledge_gap', 0)}; teacher attends the privileged block) and "
            f"misdirection ({counts.get('misdirection', 0)}; teacher attends shared in-context tokens). "
            "Knowledge gaps are the memorization target; misdirection is a routing fix.",
            fontsize=8, va="top", wrap=True,
        )
        a = fig.add_axes([0.08, 0.10, 0.52, 0.68])
        for lab, g in df.groupby("label"):
            a.scatter(g["t_priv"], g["excess"], s=12, alpha=0.7, lw=0,
                      c=color.get(lab, "#888"), label=f"{lab} ({len(g)})")
        a.set_xlabel("teacher attention on privileged block", fontsize=9)
        a.set_ylabel("excess surprise (student NLL − teacher NLL)", fontsize=9)
        a.set_title("(a) knowledge gap (right) vs misdirection (left)", fontsize=10, loc="left")
        a.legend(fontsize=8, frameon=False)
        a.grid(True, color="#dddddd", lw=0.5)

        b = fig.add_axes([0.68, 0.10, 0.27, 0.68])
        labs = [l for l in ("knowledge_gap", "misdirection", "low_surprise") if l in counts]
        b.bar(labs, [counts[l] for l in labs], color=[color[l] for l in labs])
        b.set_ylabel("answer tokens", fontsize=9)
        b.set_title("(b) decomposition", fontsize=10, loc="left")
        b.tick_params(axis="x", labelsize=6.5, rotation=20)
        pdf.savefig(fig)
        plt.close(fig)


def _router_mode(run: str) -> str:
    n = run.lower()
    if "_tf" in n:
        return "teacher_forced"
    if "_ra" in n:
        return "router_aligned"
    return "blackbox"


def router_mode_pages(pdf, df: pd.DataFrame):
    """Blackbox vs explicit-router convergence test.

    'What to memorize / where to attend' is a routing choice with the same
    blackbox-vs-teacher-aligned duality as MoE expert selection (train.moe_mode:
    dense_or_black_box / router_aligned / teacher_forced). The multigpu campaign
    trained the controlled triple; if the blackbox arm lands at the same
    (recall, retention) as the aligned arms, blackbox converges. Detects any base
    model with >=2 router modes among evaluated runs."""
    ev = df[df["has_retention"] == True].copy()  # noqa: E712
    if "checkpoint_kind" in ev:
        ev = ev[ev["checkpoint_kind"] != "epoch0"]
    if ev.empty:
        return
    ev["router_mode"] = ev["run"].map(_router_mode)
    color = {"blackbox": "#0072B2", "router_aligned": "#E69F00", "teacher_forced": "#009E73"}
    order = ["blackbox", "router_aligned", "teacher_forced"]
    for model, g in ev.groupby("model"):
        modes = [m for m in order if m in set(g["router_mode"])]
        if len(modes) < 2:
            continue
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.94, f"Routing: blackbox vs explicit router — {str(model).split('/')[-1]}",
                 fontsize=15, weight="bold")
        fig.text(0.06, 0.90,
                 "Same base, three routing supervisions. If blackbox matches the aligned arms in both "
                 "axes, the optimistic 'blackbox converges' conjecture holds; if the aligned arms lead "
                 "on retention or recall, blackbox under-converges on the routing part.",
                 fontsize=8.5, va="top", wrap=True)
        # bar: ARC accuracy and recall (exact continuation) by mode
        a = fig.add_axes([0.09, 0.12, 0.40, 0.66])
        arc = [g[g["router_mode"] == m]["arc_acc"].astype(float).mean() for m in modes]
        a.bar(modes, arc, color=[color[m] for m in modes])
        a.set_ylabel("ARC-Easy accuracy (capability retained)", fontsize=9)
        a.set_title("(a) retention by routing mode", fontsize=10, loc="left")
        a.tick_params(axis="x", labelsize=7, rotation=15)
        for i, v in enumerate(arc):
            a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        b = fig.add_axes([0.57, 0.12, 0.38, 0.66])
        rec = [g[g["router_mode"] == m]["recall_cont_machado"].astype(float).mean() for m in modes]
        b.bar(modes, rec, color=[color[m] for m in modes])
        b.set_ylabel("exact continuation recall (Machado)", fontsize=9)
        b.set_title("(b) recall by routing mode", fontsize=10, loc="left")
        b.tick_params(axis="x", labelsize=7, rotation=15)
        for i, v in enumerate(rec):
            b.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)


def _coverage_page(pdf, df: pd.DataFrame):
    lines = ["Runs WITHOUT the new retention battery (need scripts/retention_eval.py):", ""]
    if "checkpoint_kind" in df:
        df = df[df["checkpoint_kind"] != "epoch0"].copy()
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
        head_taxonomy_pages(pdf)
        surprise_pages(pdf)
        router_mode_pages(pdf, df)
        _table_page(pdf, "Best retention row per method family and model",
                    "Recall: CER (lower better) and exact-match probes (higher better). "
                    "Retention: ARC retained and WikiText log(ppl / teacher ppl) damage.",
                    _best_by_group(df),
                    ["source", "model", "run", "recall_cer", "cont_mach", "cloze_mach",
                     "censor_cerv", "arc_acc", "arc_retain", "wiki_log_damage"])
        _coverage_page(pdf, df)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
