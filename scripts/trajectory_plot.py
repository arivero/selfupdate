"""Plot recall/forgetting trajectories from per-epoch eval metrics.

Each point is one training-time eval record:

    x = recitation CER
    y = general cross-entropy delta vs the unmodified teacher reference

Lines connect epochs for each run.  This is meant to reveal flow shape:
whether a run learns recall while preserving general behavior, wanders, or
forgets as it improves.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _base_ce_lookup() -> dict[str, float]:
    refs: dict[str, float] = {}
    for p in RUNS.glob("teacher_ref_native_*/recite.json"):
        d = _read_json(p) or {}
        model = d.get("model")
        mean_ce = (d.get("general") or {}).get("mean_ce")
        if model and mean_ce is not None:
            refs[str(model)] = float(mean_ce)

    # Historical reference names kept for old runs.
    p = RUNS / "base-eval-full" / "recite.json"
    d = _read_json(p)
    if d and "general" in d:
        refs.setdefault("Qwen/Qwen3-0.6B", float(d["general"]["mean_ce"]))
    p = RUNS / "base-1p7b-general.json"
    d = _read_json(p)
    if d and "mean_ce" in d:
        refs.setdefault("Qwen/Qwen3-1.7B", float(d["mean_ce"]))
    for p in RUNS.glob("base-general-*.json"):
        d = _read_json(p) or {}
        if d.get("model") and d.get("mean_ce") is not None:
            refs.setdefault(str(d["model"]), float(d["mean_ce"]))
    return refs


def _model_label(model: str) -> str:
    for label in (
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3.6-27B",
        "gemma-4-26B-A4B",
        "gemma-4-31B",
        "Mistral-7B",
        "gpt-oss-20b",
        "gpt-oss-120b",
        "ALIA-40b",
    ):
        if label in model:
            return label.replace("gemma", "Gemma").replace("gpt-oss", "gpt-oss")
    return model.rsplit("/", 1)[-1] if model else "unknown"


def _corpus_family(path: str) -> str:
    if "/combined/" in path:
        return "Machado+Quijote"
    if "/quijote/" in path:
        return "Quijote"
    if "/poem/" in path:
        return "Machado"
    return "unknown"


def _run_config(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _eval_records(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if '"kind": "eval"' not in line or '"cer"' not in line or '"gen_ce"' not in line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("cer") is None or m.get("gen_ce") is None:
                continue
            out.append(m)
    return out


def _collect_rows(min_points: int) -> pd.DataFrame:
    refs = _base_ce_lookup()
    rows = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg = _run_config(run_dir)
        if not cfg:
            continue
        model = str((cfg.get("model") or {}).get("name") or "Qwen/Qwen3-0.6B")
        base = refs.get(model)
        evals = _eval_records(run_dir)
        if len(evals) < min_points:
            continue
        train = cfg.get("train") or {}
        data = cfg.get("data") or {}
        examples_path = str(data.get("examples_path") or "")
        clean = (
            run_dir.name.startswith("clean_")
            and train.get("run_class", "method") == "method"
            and train.get("conn_stride", 0) == 1
            and train.get("readout_source", "UNSET") in {"UNSET", "teacher_kl"}
            and "tail_ce_kind" not in train
            and "task_label" not in json.dumps(train)
        )
        for i, m in enumerate(evals):
            gen_ce = float(m["gen_ce"])
            rows.append(
                {
                    "run": run_dir.name,
                    "model": model,
                    "model_label": _model_label(model),
                    "corpus_family": _corpus_family(examples_path),
                    "epoch": m.get("epoch", i),
                    "eval_index": i,
                    "cer": float(m["cer"]),
                    "gen_ce": gen_ce,
                    "forget_dce": gen_ce - base if base is not None else math.nan,
                    "has_base_ce": base is not None,
                    "clean_method_like": clean,
                    "loss": train.get("hidden_loss", "unknown"),
                    "conn_window": train.get("conn_window", 0),
                    "readout_window": train.get("readout_window_blocks", 0),
                }
            )
    return pd.DataFrame(rows)


def _plot(df: pd.DataFrame, out: Path, title: str) -> None:
    if df.empty:
        return
    models = sorted(df["model_label"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    colors = {m: cmap(i % 20) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(12.8, 8.0))
    for run, g in df.sort_values(["run", "eval_index"]).groupby("run"):
        model = str(g["model_label"].iloc[0])
        clean = bool(g["clean_method_like"].iloc[0])
        lw = 1.8 if clean else 0.9
        alpha = 0.78 if clean else 0.22
        marker = "o" if clean else "."
        ax.plot(
            g["cer"],
            g["forget_dce"],
            marker=marker,
            markersize=3.2 if clean else 2.0,
            linewidth=lw,
            alpha=alpha,
            color=colors[model],
        )
        if clean and len(g) >= 4:
            end = g.iloc[-1]
            ax.text(
                end["cer"],
                end["forget_dce"],
                run.replace("clean_", "").replace("_rag", ""),
                fontsize=6,
                alpha=0.75,
                color=colors[model],
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.05, color="tab:red", linewidth=0.7, linestyle=":")
    ax.axhline(0.30, color="tab:red", linewidth=0.7, linestyle="--")
    ax.set_xlabel("recitation CER at training-time eval (lower is better)")
    ax.set_ylabel("forgetting: general cross-entropy delta vs epoch-zero teacher")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.35)

    handles = [
        plt.Line2D([0], [0], color=colors[m], lw=2, label=m)
        for m in models
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_facets(df: pd.DataFrame, out: Path, title: str) -> None:
    if df.empty:
        return
    keys = [
        key for key, g in df.groupby(["model_label", "corpus_family"], sort=True)
        if len(g["run"].unique()) > 0
    ]
    if not keys:
        return
    cols = min(3, len(keys))
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 4.35 * rows), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, key in zip(axes.flat, keys):
        model, corpus = key
        sub = df[(df["model_label"] == model) & (df["corpus_family"] == corpus)]
        runs = sorted(sub["run"].unique())
        colors = {run: cmap(i % 10) for i, run in enumerate(runs)}
        for run, g in sub.sort_values(["run", "eval_index"]).groupby("run"):
            label = (
                run.removeprefix("clean_")
                .removesuffix("_rag")
                .replace("_lora", "")
                .replace("q_ch1_", "q1_")
                .replace("slide", "s")
                .replace("vocab", "vocab")
            )
            ax.plot(
                g["cer"],
                g["forget_dce"],
                marker="o",
                markersize=3.3,
                linewidth=1.7,
                alpha=0.82,
                color=colors[run],
                label=label,
            )
            end = g.iloc[-1]
            ax.text(end["cer"], end["forget_dce"], label, fontsize=6.5, color=colors[run])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(0.05, color="tab:red", linewidth=0.7, linestyle=":")
        ax.axhline(0.30, color="tab:red", linewidth=0.7, linestyle="--")
        ax.set_title(f"{model} / {corpus}", fontsize=10)
        ax.set_xlabel("CER")
        ax.set_ylabel("forgetting ΔCE")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.legend(fontsize=6, frameon=False, loc="best")

    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-points", type=int, default=2)
    args = ap.parse_args()

    df = _collect_rows(args.min_points)
    if df.empty:
        print("no trajectory data found")
        return 1
    out_csv = RUNS / "trajectory_cer_forgetting.csv"
    df.to_csv(out_csv, index=False)

    usable = df[df["has_base_ce"]].copy()
    _plot(
        usable,
        RUNS / "trajectory_cer_forgetting.png",
        "Recall/forgetting trajectories: all runs with epoch evals",
    )
    _plot(
        usable[usable["clean_method_like"]],
        RUNS / "trajectory_cer_forgetting_clean.png",
        "Recall/forgetting trajectories: clean teacher-sourced method-like runs",
    )
    _plot_facets(
        usable[usable["clean_method_like"]],
        RUNS / "trajectory_cer_forgetting_clean_facets.png",
        "Clean teacher-sourced trajectories by model and corpus",
    )
    print(
        f"wrote {out_csv}, runs/trajectory_cer_forgetting.png, "
        "runs/trajectory_cer_forgetting_clean.png, "
        "runs/trajectory_cer_forgetting_clean_facets.png"
    )
    print(
        f"runs={df['run'].nunique()} usable_with_base={usable['run'].nunique()} "
        f"points={len(usable)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
