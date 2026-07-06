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

OLD_KEYS = {
    "tail_ce_blocks", "tail_ce_weight", "tail_ce_kind", "tail_hidden_weight",
    "last_block_ce_weight", "lens_ce_weight", "lens_ce_from", "answer_ce_weight",
}


def results_table() -> pd.DataFrame:
    base_ce = {}  # per-model epoch-zero teacher references
    base_p = Path("runs/base-eval-full/recite.json")
    if base_p.exists():
        base_ce["Qwen/Qwen3-0.6B"] = json.loads(base_p.read_text())["general"]["mean_ce"]
    p17 = Path("runs/base-1p7b-general.json")
    if p17.exists():
        base_ce["Qwen/Qwen3-1.7B"] = json.loads(p17.read_text())["mean_ce"]
    rows = []
    for run_dir in sorted(Path("runs").iterdir()):
        cfg_p = run_dir / "config.yaml"
        if not cfg_p.exists():
            continue
        cfg = yaml.safe_load(cfg_p.read_text())
        train_cfg = cfg.get("train", {})
        metrics = read_metrics(run_dir)
        trains = [m for m in metrics if m.get("kind") == "train"]
        if not trains:  # sequential schedule logs per-stage lines instead
            trains = [{"loss": m["loss"], "step": m["steps"]}
                      for m in metrics if m.get("kind") == "stage"]
        evals = [m for m in metrics if m.get("kind") == "eval"]
        done = [m for m in metrics if m.get("kind") == "done"]
        full = run_dir / "eval" / "recite.json"
        full_cer = line_exact = forget = None
        if full.exists():
            r = json.loads(full.read_text())
            full_cer = r["cer"]
            line_exact = r["line_exact"]
            model_base = base_ce.get(cfg.get("model", {}).get("name", "Qwen/Qwen3-0.6B"))
            if model_base and "general" in r:
                forget = round(r["general"]["mean_ce"] - model_base, 3)
        rows.append({
            "run": run_dir.name,
            "method": train_cfg["method"],
            "run_class": train_cfg.get("run_class", "method"),
            "legacy_keys": ",".join(k for k in sorted(OLD_KEYS) if k in train_cfg),
            "schedule": (train_cfg.get("schedule", "")
                         if train_cfg["method"] == "layerwise" else ""),
            "readout_source": train_cfg.get("readout_source", train_cfg.get("tail_ce_kind", "UNSET")),
            "readout_window": train_cfg.get("readout_window_blocks", train_cfg.get("tail_ce_blocks", 0)),
            "readout_weight": train_cfg.get("readout_weight", train_cfg.get("tail_ce_weight", 0.0)),
            "window_hidden_weight": train_cfg.get("window_hidden_weight", train_cfg.get("tail_hidden_weight", 1.0)),
            "conn_window": train_cfg.get("conn_window", 0),
            "conn_stride": train_cfg.get("conn_stride", 0),
            "lora": train_cfg["lora"]["enabled"],
            "mode": cfg["mask"]["mode"],
            "compaction": cfg["mask"]["compaction"],
            "last_train_cer": evals[-1]["cer"] if evals else None,
            "full_eval_cer": full_cer,
            "line_exact": line_exact,
            "forgetting_dCE": forget,  # general-CE rise vs base model
            "loss_first": round(sum(m["loss"] for m in trains[:20]) / len(trains[:20]), 4) if trains else None,
            "loss_final": round(sum(m["loss"] for m in trains[-20:]) / len(trains[-20:]), 4) if trains else None,
            "steps": trains[-1]["step"] if trains else None,
            "items": max((m.get("items_seen", 0) for m in trains), default=len(trains)),
            "train_logs": len(trains),
            "vram_gb": done[-1]["vram_gb"] if done else None,
            "vram_resv_gb": done[-1].get("vram_reserved_gb") if done else None,
            "train_min": (evals[-1].get("minutes") if evals else None)
                         or (done[-1].get("minutes") if done else None),
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
    fig.savefig(out, dpi=220)
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


def training_curves() -> None:
    """Loss-vs-step and eval-CER-vs-epoch curves for every run (detailed
    training-dynamics record: loss progress, time, convergence shape)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [d for d in sorted(Path("runs").iterdir())
            if (d / "metrics.jsonl").exists() and (d / "config.yaml").exists()]
    if not runs:
        return

    # one row per method family — the single panel got too crowded once the
    # e40 wave landed. Axes are SHARED across rows (sharex/sharey per column)
    # so curves stay directly comparable between families.
    def family(cfg: dict) -> int:
        t = cfg.get("train", {})
        if any(k in t for k in OLD_KEYS):
            return 2
        rc = t.get("run_class", "method")
        if rc == "method":
            return 0
        if rc in ("ablation", "control"):
            return 1
        return 2

    row_titles = ["clean method", "ablation / control", "legacy / confounded"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex="col", sharey="col")
    # >10 lines per panel: the default 10-color cycle repeats and unrelated
    # runs become indistinguishable (bit us 2026-07-03). 20 colors x 2 line
    # styles = 40 unique combinations, assigned per run within its family.
    import matplotlib.cm as cm
    counters = [0, 0, 0]
    def style(row):
        i = counters[row]; counters[row] += 1
        return {"color": cm.tab20(i % 20), "linestyle": ["-", "--"][(i // 20) % 2]}
    for d in runs:
        cfg = yaml.safe_load((d / "config.yaml").read_text()) or {}
        row = family(cfg)
        st = style(row)
        ms = read_metrics(d)
        trains = [m for m in ms if m.get("kind") == "train"]
        evals = [m for m in ms if m.get("kind") == "eval" and "epoch" in m]
        if trains:
            xs = [m.get("items_seen", i) for i, m in enumerate(trains)]
            step = max(1, len(xs) // 400)  # thin for plotting
            ii = list(range(0, len(xs), step))
            axes[row][0].plot([xs[i] for i in ii], [trains[i]["loss"] for i in ii],
                              label=d.name, alpha=0.8, linewidth=1, **st)
        if evals:
            axes[row][1].plot([m["epoch"] for m in evals], [m["cer"] for m in evals],
                              marker="o", label=d.name, alpha=0.8, **st)
        # per-epoch forgetting (gen_ce logged at eval epochs since 2026-07-03;
        # older runs lack it): the reference for how long each model+method
        # can train before memorization is paid for with general ability
        gens = [m for m in evals if "gen_ce" in m]
        if gens:
            axes[row][2].plot([m["epoch"] for m in gens],
                              [m["gen_ce"] for m in gens],
                              marker="s", label=d.name, alpha=0.8, **st)
    for row in range(3):
        axes[row][0].set_yscale("log")
        axes[row][0].set_ylabel(f"{row_titles[row]}\nloss (log)")
        axes[row][1].set_ylabel("eval CER (8-ex subset)")
        axes[row][2].set_ylabel("general-CE (forgetting)")
        axes[row][1].legend(fontsize=6)
        if axes[row][2].lines:
            axes[row][2].legend(fontsize=6)
    axes[0][2].axhline(3.278, color="gray", linestyle=":", linewidth=1)
    axes[1][2].axhline(3.278, color="gray", linestyle=":", linewidth=1)
    axes[2][2].axhline(3.278, color="gray", linestyle=":", linewidth=1)
    axes[2][0].set_xlabel("training items seen")
    axes[2][1].set_xlabel("epoch")
    axes[2][2].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig("runs/curves.png", dpi=220)
    print("wrote runs/curves.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", default=None, help="run names")
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    df = results_table()
    if not df.empty:
        print(df.to_markdown(index=False))
        Path("runs/results.md").write_text(df.to_markdown(index=False))
    training_curves()
    if args.deltas:
        delta_profiles(args.deltas, args.base_model)


if __name__ == "__main__":
    main()
