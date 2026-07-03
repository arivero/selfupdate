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
    base_ce = None
    base_p = Path("runs/base-eval-full/recite.json")
    if base_p.exists():
        base_ce = json.loads(base_p.read_text())["general"]["mean_ce"]
    rows = []
    for run_dir in sorted(Path("runs").iterdir()):
        cfg_p = run_dir / "config.yaml"
        if not cfg_p.exists():
            continue
        cfg = yaml.safe_load(cfg_p.read_text())
        evals = [m for m in read_metrics(run_dir) if m.get("kind") == "eval"]
        done = [m for m in read_metrics(run_dir) if m.get("kind") == "done"]
        full = run_dir / "eval" / "recite.json"
        full_cer = line_exact = forget = None
        if full.exists():
            r = json.loads(full.read_text())
            full_cer = r["cer"]
            line_exact = r["line_exact"]
            if base_ce and "general" in r:
                forget = round(r["general"]["mean_ce"] - base_ce, 3)
        rows.append({
            "run": run_dir.name,
            "method": cfg["train"]["method"],
            "schedule": cfg["train"].get("schedule", ""),
            "lora": cfg["train"]["lora"]["enabled"],
            "mode": cfg["mask"]["mode"],
            "compaction": cfg["mask"]["compaction"],
            "last_train_cer": evals[-1]["cer"] if evals else None,
            "full_eval_cer": full_cer,
            "line_exact": line_exact,
            "forgetting_dCE": forget,  # general-CE rise vs base model
            "vram_gb": done[-1]["vram_gb"] if done else None,
            "minutes": (evals[-1].get("minutes") if evals else None),
        })
    return pd.DataFrame(rows)


def _run_deltas(base_state: dict, run_name: str) -> "pd.DataFrame":
    from selfupdate.eval.weight_deltas import full_ft_deltas, load_state, lora_deltas

    ckpt = Path("runs") / run_name / "checkpoint"
    if (ckpt / "adapter_config.json").exists():
        acfg = json.loads((ckpt / "adapter_config.json").read_text())
        scaling = acfg["lora_alpha"] / acfg["r"]
        return lora_deltas(base_state, load_state(ckpt / "adapter_model.safetensors"), scaling)
    return full_ft_deltas(base_state, load_state(ckpt))


def delta_profiles(run_names: list[str], base_model: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from huggingface_hub import snapshot_download

    from selfupdate.eval.convergence import layer_cosines, profile_spearman
    from selfupdate.eval.weight_deltas import load_state, per_layer_profile

    base_state = load_state(snapshot_download(base_model))
    profiles = {name: per_layer_profile(_run_deltas(base_state, name))
                for name in run_names}

    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    for name, prof in profiles.items():
        axes[0].plot(prof.index, prof.values, marker="o", label=name)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("relative weight delta")
    axes[0].legend(fontsize=8)

    mat = pd.DataFrame(profiles).T  # runs x layers
    norm = mat.div(mat.max(axis=1), axis=0)  # per-run normalized profile
    im = axes[1].imshow(norm.values, aspect="auto", cmap="viridis")
    axes[1].set_yticks(range(len(norm)), norm.index, fontsize=7)
    axes[1].set_xticks(range(0, norm.shape[1], 2),
                       [str(c) for c in norm.columns[::2]], fontsize=7)
    axes[1].set_xlabel("layer (per-run max-normalized delta)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    out = Path("runs/delta_profiles.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    if len(run_names) >= 2:
        from itertools import combinations

        from selfupdate.eval.weight_deltas import load_state as _ls

        for a, b in combinations(run_names, 2):
            ca, cb = Path("runs") / a / "checkpoint", Path("runs") / b / "checkpoint"
            if (ca / "adapter_config.json").exists() or (cb / "adapter_config.json").exists():
                print(f"({a}, {b}): cosine convergence needs materialized full deltas; skipped for LoRA")
                continue
            df = layer_cosines(base_state, _ls(ca), _ls(cb))
            print(f"\n=== {a} vs {b} ===")
            print(df.to_string(index=False))
            print(f"profile Spearman: {profile_spearman(df):.3f}")
            df.to_csv(f"runs/convergence_{a}__{b}.csv", index=False)


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
