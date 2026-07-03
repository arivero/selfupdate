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
    return json.loads(p.read_text()) if p.exists() else None


def summary_text() -> str:
    base = _read_json(RUNS / "base-eval-full/recite.json")
    kd = _read_json(RUNS / "kd_ce_0p6b_rag/eval/recite.json")
    lines = [
        "Project: self-distillation of context (same model as teacher and student).",
        "Teacher sees privileged context (RAG passage / <think> trace); the student",
        "must reproduce its behavior without it. Corpus: 'La tierra de Alvargonzalez'",
        "(A. Machado, 1912), 725 verses, 228 tasks (continuations, per-section",
        "recitations, opening). Model: Qwen3-0.6B on a single RTX 3060 12 GB.",
        "",
        "Methods compared: classical KD (top-k KL on logits) vs layer-wise hidden",
        "matching with block-local backward (summed / sequential / teacher_censored",
        "schedules), each x {full fine-tune, LoRA}, plus gold-CE auxiliaries and the",
        "online teacher (adapters-off = frozen teacher, no cache).",
        "",
    ]
    if base and kd:
        lines += [
            f"Recitation (full corpus, n={base['n']}): base CER {base['cer']:.3f} ->",
            f"best (KD+goldCE) CER {kd['cer']:.3f}, {kd['line_exact']:.0%} lines verbatim.",
            f"Forgetting probe (CE on held-out text): base {base['general']['mean_ce']:.3f},",
            f"KD+CE {kd['general']['mean_ce']:.3f} (delta +{kd['general']['mean_ce']-base['general']['mean_ce']:.2f}).",
            "",
        ]
    lines += [
        "Key findings:",
        " 1. Pure top-k KL saturates (KL~0.03) without free-run recitation; a gold-CE",
        "    auxiliary makes recitation click. Memorization is front-of-poem biased.",
        " 2. Pure hidden matching (any schedule) does not recite at 0.6B/10 epochs;",
        "    a local last-block CE (block-locality preserved) is the working hybrid lever.",
        " 3. Convergence: within-family weight-delta directions align (cos 0.6-0.65);",
        "    across families they are orthogonal (cos ~0.02) while per-layer magnitude",
        "    profiles correlate (Spearman 0.73-0.78): same 'where', different 'what'.",
        " 4. teacher_censored (variant b): per-layer increment targets are 13x smaller",
        "    than student-stream targets, peak at layer ~7 (context integration site),",
        "    ~0 at the last layer; layers train independently (parallelizable).",
        " 5. Efficiency: sequential layerwise 3.2-3.7 GB vs KD full-FT 9.45 GB (<40%);",
        "    LoRA runs ~3.2 GB with near-zero forgetting (+0.06 CE) but rank-16 limits",
        "    how far the KL can be driven.",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Details in the following pages;",
        "reproducibility: configs/experiments/*.yaml, runs/*/metrics.jsonl, git log.",
    ]
    return "\n".join(lines)


def results_page(pdf):
    md = RUNS / "results.md"
    if not md.exists():
        return
    _text_page(pdf, "Results table (all runs)",
               md.read_text().replace("|", " "), fontsize=6.5)


def layer_swap_page(pdf):
    csv = RUNS / "kd_ce_0p6b_rag/eval/layer_swap.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df.layer, df.graft_cer, marker="o", label="graft (base + trained block L)")
    ax.plot(df.layer, df.ablate_cer, marker="s", label="ablate (trained, block L reverted)")
    ax.set_xlabel("layer")
    ax.set_ylabel("recitation CER")
    ax.set_title("Causal localization: layer graft/ablate on kd_ce checkpoint")
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
        full = _read_json(d / "eval/recite.json")
        b = [f"== {d.name} =="]
        t = cfg.get("train", {})
        b.append(f"  method={t.get('method')} schedule={t.get('schedule')} "
                 f"lora={t.get('lora', {}).get('enabled')} lr={t.get('lr')} "
                 f"epochs={t.get('epochs')} ce={t.get('answer_ce_weight', 0)}/"
                 f"{t.get('last_block_ce_weight', 0)} online={t.get('online_teacher')}")
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
        layer_swap_page(pdf)
        _image_page(pdf, "Logit-lens depth profile (kd_ce vs base)",
                    RUNS / "kd_ce_0p6b_rag/eval/logit_lens.png")
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
