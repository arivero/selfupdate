"""Cross-run analysis: results table + per-layer localization heatmap.

Usage:
    python scripts/analyze.py                     # table over all runs/
    python scripts/analyze.py --deltas run1 run2  # add weight-delta profiles + convergence
    python scripts/analyze.py --lens runs/<name>  # logit-lens depth profile
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import yaml

from selfupdate.utils.runlog import read_metrics


def results_table() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(Path("runs").iterdir()):
        cfg_p = run_dir / "config.yaml"
        if not cfg_p.exists():
            continue
        cfg = yaml.safe_load(cfg_p.read_text())
        evals = [m for m in read_metrics(run_dir) if m.get("kind") == "eval"]
        done = [m for m in read_metrics(run_dir) if m.get("kind") == "done"]
        full = run_dir / "eval" / "recite.json"
        full_cer = json.loads(full.read_text())["cer"] if full.exists() else None
        rows.append({
            "run": run_dir.name,
            "method": cfg["train"]["method"],
            "schedule": cfg["train"].get("schedule", ""),
            "lora": cfg["train"]["lora"]["enabled"],
            "mode": cfg["mask"]["mode"],
            "compaction": cfg["mask"]["compaction"],
            "last_train_cer": evals[-1]["cer"] if evals else None,
            "full_eval_cer": full_cer,
            "vram_gb": done[-1]["vram_gb"] if done else None,
            "minutes": (evals[-1].get("minutes") if evals else None),
        })
    return pd.DataFrame(rows)


def delta_profiles(run_names: list[str], base_model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from huggingface_hub import snapshot_download

    from selfupdate.eval.convergence import layer_cosines, profile_spearman
    from selfupdate.eval.weight_deltas import full_ft_deltas, load_state, per_layer_profile

    base_state = load_state(snapshot_download(base_model))
    profiles = {}
    states = {}
    for name in run_names:
        st = load_state(Path("runs") / name / "checkpoint")
        states[name] = st
        profiles[name] = per_layer_profile(full_ft_deltas(base_state, st))

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, prof in profiles.items():
        ax.plot(prof.index, prof.values, marker="o", label=name)
    ax.set_xlabel("layer")
    ax.set_ylabel("relative weight delta (RMS over modules)")
    ax.legend()
    fig.tight_layout()
    out = Path("runs/delta_profiles.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    if len(run_names) == 2:
        df = layer_cosines(base_state, states[run_names[0]], states[run_names[1]])
        print(df.to_string(index=False))
        print(f"profile Spearman: {profile_spearman(df):.3f}")
        df.to_csv("runs/convergence.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", default=None, help="run names")
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    df = results_table()
    if not df.empty:
        print(df.to_markdown(index=False))
        Path("runs/results.md").write_text(df.to_markdown(index=False))
    if args.deltas:
        delta_profiles(args.deltas, args.base_model)


if __name__ == "__main__":
    main()
