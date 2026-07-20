"""Remembering vs forgetting as epochs evolve, per run.

For every run with per-epoch eval records, plot the memorization curve
(subset recite CER, falling = remembering) against the forgetting curve
(general-CE delta vs the model's base reference, rising = forgetting) on
shared epoch axes, and classify the endpoint:

    |dCE| < 0.05   negligible
    0.05 - 0.30    mild
    > 0.30         heavy (catastrophic territory)

Base references: runs/base-general-<short>.json (base_general.py) or the
legacy runs/base-eval-full/recite.json ("general" block) / runs/
base-1p7b-general.json names that analyze.py already knows.

Outputs: runs/forget_curves.png + runs/forget_curves.csv plus
runs/<run>/eval/forget_curve.png.
Usage: python compressed/forget_curves.py [--runs '*']
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

RUNS = Path("runs")


def eval_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if '"kind": "eval"' not in line or '"gen_ce"' not in line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def base_ce_lookup() -> dict[str, float]:
    refs = {}
    p = RUNS / "base-eval-full" / "recite.json"
    if p.exists():
        refs["Qwen/Qwen3-0.6B"] = json.loads(p.read_text())["general"]["mean_ce"]
    p = RUNS / "base-1p7b-general.json"
    if p.exists():
        refs["Qwen/Qwen3-1.7B"] = json.loads(p.read_text())["mean_ce"]
    for p in RUNS.glob("base-general-*.json"):
        d = json.loads(p.read_text())
        refs[d.get("model", p.stem)] = d["mean_ce"]
    return refs


def classify(dce: float) -> str:
    a = abs(dce)
    return "negligible" if a < 0.05 else ("mild" if a <= 0.30 else "HEAVY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*")
    args = ap.parse_args()
    refs = base_ce_lookup()

    rows = []
    curves = {}
    for run_dir in sorted(RUNS.iterdir()):
        if not fnmatch.fnmatch(run_dir.name, args.runs):
            continue
        cfg_p = run_dir / "config.yaml"
        if not cfg_p.exists():
            continue
        model = (yaml.safe_load(cfg_p.read_text()).get("model") or {}).get(
            "name", "Qwen/Qwen3-0.6B")
        evals = eval_metrics(run_dir)
        if not evals:
            continue
        base = refs.get(model)
        ep = [m.get("epoch", m.get("layer", i)) for i, m in enumerate(evals)]
        cer = [m["cer"] for m in evals]
        dce = [m["gen_ce"] - base if base else None for m in evals]
        curves[run_dir.name] = (ep, cer, dce)
        for e, c, d in zip(ep, cer, dce):
            rows.append({"run": run_dir.name, "model": model, "epoch": e,
                         "recite_cer": c, "forget_dce": d})
        end = dce[-1]
        print(f"{run_dir.name:34s} final CER {cer[-1]:.3f} | dCE "
              f"{'n/a' if end is None else f'{end:+.3f} ({classify(end)})'}")
        out_dir = run_dir / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(ep, cer, color="tab:blue", marker="o", label="recall CER")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("epoch")
        ax.set_ylabel("recitation CER", color="tab:blue")
        ax.tick_params(axis="y", colors="tab:blue")
        if dce[0] is not None:
            ax2 = ax.twinx()
            ax2.plot(ep, dce, color="tab:red", marker="s", label="forgetting ΔCE")
            ax2.axhline(0.05, color="tab:red", lw=0.5, ls=":")
            ax2.axhline(0.30, color="tab:red", lw=0.5, ls="--")
            ax2.set_ylabel("general CE delta", color="tab:red")
            ax2.tick_params(axis="y", colors="tab:red")
        ax.set_title(run_dir.name, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "forget_curve.png", dpi=140)
        plt.close(fig)

    if not curves:
        print("no runs with per-epoch eval records matched")
        return
    pd.DataFrame(rows).to_csv(RUNS / "forget_curves.csv", index=False)

    n = len(curves)
    cols = min(4, n)
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.2 * rws),
                             squeeze=False)
    for ax, (name, (ep, cer, dce)) in zip(axes.flat, curves.items()):
        ax.plot(ep, cer, color="tab:blue", label="recite CER (subset)")
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("epoch", fontsize=7)
        ax.tick_params(labelsize=7)
        if dce[0] is not None:
            ax2 = ax.twinx()
            ax2.plot(ep, dce, color="tab:red", label="general-CE delta")
            ax2.axhline(0.05, color="tab:red", lw=0.5, ls=":")
            ax2.axhline(0.30, color="tab:red", lw=0.5, ls="--")
            ax2.tick_params(labelsize=7, colors="tab:red")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("remembering (CER, blue, left) vs forgetting (ΔCE, red, right)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RUNS / "forget_curves.png", dpi=140)
    print("wrote runs/forget_curves.png + runs/forget_curves.csv")


if __name__ == "__main__":
    main()
