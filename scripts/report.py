"""Verbose PDF report of all runs: runs/report.pdf.

Assembles whatever artifacts exist (results table, training curves, delta
profiles, graft/ablate curves, logit lens, convergence CSVs, per-run stats)
into a multi-page PDF. Robust to missing pieces — reports what is there.

Usage: python scripts/report.py [--out runs/report.pdf]
"""

import argparse
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from selfupdate.utils.runlog import read_metrics

RUNS = Path("runs")


def _text_page(pdf, title, body, fontsize=9):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold")
    fig.text(0.08, 0.90, body, fontsize=fontsize, family="monospace",
             va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf, title, png):
    if not Path(png).exists():
        return
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=14, weight="bold")
    ax = fig.add_axes([0.05, 0.25, 0.9, 0.65])
    ax.imshow(mpimg.imread(png))
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() and p.is_file() else None


def _recite_json(p):
    r = _read_json(p)
    return r if isinstance(r, dict) and "cer" in r else None


def _recite_files(d):
    eval_dir = Path(d) / "eval"
    if not eval_dir.exists():
        return []
    files = []
    direct = eval_dir / "recite.json"
    if direct.exists() and direct.is_file():
        files.append(direct)
    for p in sorted(eval_dir.glob("recite*.json")):
        if p.is_file() and p not in files:
            files.append(p)
        nested = p / "recite.json"
        if nested.exists() and nested.is_file() and nested not in files:
            files.append(nested)
    return files


def _recite_label(d, p):
    rel = Path(p).relative_to(Path(d) / "eval")
    if rel == Path("recite.json"):
        return "final"
    if rel.name == "recite.json" and len(rel.parts) > 1:
        return rel.parts[0]
    return Path(p).stem


def _best_run():
    """(name, recite dict) of the best full-corpus recitation among all runs."""
    best = None
    for d in sorted(RUNS.iterdir()):
        for p in _recite_files(d):
            r = _recite_json(p)
            if r and (best is None or r["cer"] < best[1]["cer"]):
                best = (f"{d.name}:{_recite_label(d, p)}", r)
    return best


def summary_text() -> str:
    base = _read_json(RUNS / "base-eval-full/recite.json")
    best = _best_run()
    lines = [
        "Project: self-distillation of context (same model as teacher and student).",
        "Teacher sees privileged context (RAG passage / <think> trace); the student",
        "must reproduce its behavior without it. Corpus: 'La tierra de Alvargonzalez'",
        "(A. Machado, 1912), 725 verses, 228 tasks (continuations, per-section",
        "recitations, opening). Model: Qwen3-0.6B on a single RTX 3060 12 GB.",
        "",
        "Method family: classical KD (top-k KL on logits), either full fine-tune",
        "of transformer blocks or LoRA adapters, plus gold-CE auxiliaries and the",
        "online teacher path (adapters-off = frozen teacher, no cache).",
        "",
    ]
    if base and best:
        name, kd = best
        lines += [
            f"Recitation (full corpus, n={base['n']}): base CER {base['cer']:.3f} ->",
            f"best ({name}) CER {kd['cer']:.3f}, {kd['line_exact']:.0%} lines verbatim.",
            f"Forgetting probe (CE on held-out text): base {base['general']['mean_ce']:.3f},",
            f"best-run {kd['general']['mean_ce']:.3f} (delta {kd['general']['mean_ce']-base['general']['mean_ce']:+.2f}).",
            "  (computed from current artifacts)",
            "",
        ]
    lines += [
        "Key findings (NARRATIVE SNAPSHOT written 2026-07-03 — tables and",
        "figure pages are computed from current artifacts; re-read them if",
        "this report was regenerated after new experiments):",
        " 1. Pure top-k KL saturates (KL~0.03) without free-run recitation; a gold-CE",
        "    auxiliary makes recitation click. Memorization is front-of-poem biased.",
        " 2. Gold-CE weight, LoRA learning rate, compaction, and model size are the",
        "    active experimental axes on this branch.",
        " 3. The localization question is now within classical KD: compare per-layer",
        "    weight-delta norms, logit-lens depth profiles, and graft/ablate effects",
        "    across KD recipes and model sizes.",
        " 4. LoRA runs are memory-efficient, but rank and learning rate control how",
        "    far the KL and gold recitation losses can be driven.",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Details in the following pages;",
        "reproducibility: configs/experiments/*.yaml, runs/*/metrics.jsonl, git log.",
    ]
    return "\n".join(lines)


_COL_SHORT = {
    "last_train_cer": "train_cer", "full_eval_cer": "eval_cer",
    "line_exact": "exact", "forgetting_dCE": "forget",
    "compaction": "compact",
    "loss_first": "loss0", "loss_final": "lossN", "train_min": "min",
}


def results_page(pdf):
    """Landscape page with the runs table rendered as a real table —
    the portrait text dump was unreadable once the grid grew."""
    md = RUNS / "results.md"
    if not md.exists():
        return
    lines = [l for l in md.read_text().splitlines() if l.startswith("|")]
    if len(lines) < 3:
        return
    split = lambda l: [c.strip() for c in l.strip("|").split("|")]
    header = [_COL_SHORT.get(h, h) for h in split(lines[0])]

    def fmt(x):
        if x in ("", "nan"):
            return "—"
        try:
            return f"{float(x):.3g}"
        except ValueError:
            return x

    rows = [[fmt(c) for c in split(l)] for l in lines[2:]]
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.04, 0.94, "Results table (all runs)", fontsize=16, weight="bold")
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.84])
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header,
                   loc="upper center", cellLoc="right")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(range(len(header)))
    tbl.scale(1, 1.35)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
    pdf.savefig(fig)
    plt.close(fig)


def layer_swap_pages(pdf):
    for csv in sorted(RUNS.glob("*/eval/layer_swap.csv")):
        run = csv.parent.parent.name
        df = pd.read_csv(csv)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df.layer, df.graft_cer, marker="o", label="graft (base + trained block L)")
        ax.plot(df.layer, df.ablate_cer, marker="s", label="ablate (trained, block L reverted)")
        ax.set_xlabel("layer")
        ax.set_ylabel("recitation CER")
        ax.set_title(f"Causal localization: layer graft/ablate — {run}")
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def per_run_appendix(pdf):
    blocks = []
    for d in sorted(RUNS.iterdir()):
        if not (d / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        ms = read_metrics(d)
        trains = [m for m in ms if m.get("kind") == "train"]
        evals = [m for m in ms if m.get("kind") == "eval"]
        stages = [m for m in ms if m.get("kind") == "stage"]
        fulls = [(p, _recite_json(p)) for p in _recite_files(d)]
        fulls = [(p, r) for p, r in fulls if r]
        full = next((r for p, r in fulls if _recite_label(d, p) == "final"), None)
        best_full = min((r for _, r in fulls), key=lambda r: r["cer"]) if fulls else None
        best_label = next((_recite_label(d, p) for p, r in fulls if r is best_full), None)
        b = [f"== {d.name} =="]
        t = cfg.get("train", {})
        b.append(f"  method={t.get('method')} "
                 f"lora={t.get('lora', {}).get('enabled')} lr={t.get('lr')} "
                 f"epochs={t.get('epochs')} ce={t.get('answer_ce_weight', 0)} "
                 f"online={t.get('online_teacher')}")
        if trains:
            n = max(1, len(trains[:20]))
            b.append(f"  loss: first20 {sum(m['loss'] for m in trains[:20])/n:.4f} "
                     f"-> last20 {sum(m['loss'] for m in trains[-20:])/len(trains[-20:]):.4f} "
                     f"({len(trains)} items)")
        if stages:
            per = {}
            for s in stages:
                per[s["layer"]] = s["loss"]
            ks = sorted(per)
            b.append("  stage losses: " + " ".join(f"L{k}:{per[k]:.3f}" for k in ks[::7]))
        for m in evals[-2:]:
            b.append(f"  eval ep{m.get('epoch', m.get('layer'))}: CER {m['cer']:.3f} "
                     f"exact {m['line_exact']:.2f} vram {m.get('vram_gb')}GB "
                     f"{m.get('minutes')}min")
        if full:
            g = full.get("general", {}).get("mean_ce")
            b.append(f"  FULL eval: CER {full['cer']:.4f} line-exact {full['line_exact']:.4f}"
                     + (f" general-CE {g:.3f}" if g else ""))
        if best_full and best_label != "final":
            g = best_full.get("general", {}).get("mean_ce")
            b.append(f"  BEST full ({best_label}): CER {best_full['cer']:.4f} "
                     f"line-exact {best_full['line_exact']:.4f}"
                     + (f" general-CE {g:.3f}" if g else ""))
        blocks.append("\n".join(b))
    for i in range(0, len(blocks), 6):
        _text_page(pdf, f"Per-run details ({i // 6 + 1})",
                   "\n\n".join(blocks[i:i + 6]), fontsize=7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/report.pdf")
    args = ap.parse_args()

    with PdfPages(args.out) as pdf:
        _text_page(pdf, "Self-distillation of context — experiment report",
                   summary_text())
        results_page(pdf)
        _image_page(pdf, "Training dynamics (loss / eval CER)", RUNS / "curves.png")
        _image_page(pdf, "Per-layer weight-delta profiles & heatmap",
                    RUNS / "delta_profiles.png")
        layer_swap_pages(pdf)
        for png in sorted(RUNS.glob("*/eval/logit_lens.png")):
            _image_page(pdf, f"Logit-lens depth profile — {png.parent.parent.name}", png)
        # convergence tables as text
        conv = sorted(RUNS.glob("convergence_*.csv"))
        if conv:
            txt = []
            for f in conv:
                df = pd.read_csv(f)
                sp = df["norm_a"].rank().corr(df["norm_b"].rank())
                txt.append(f"{f.stem.replace('convergence_', '').replace('__', ' vs ')}:\n"
                           f"  mean cosine {df.cosine.mean():.3f}, Spearman {sp:.2f}")
            _text_page(pdf, "Cross-method convergence of weight deltas",
                       "\n\n".join(txt))
        per_run_appendix(pdf)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
