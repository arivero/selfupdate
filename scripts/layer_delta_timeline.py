"""Layer-modification timelines from epoch checkpoints.

Existing Campaign 2 runs mostly saved only ``runs/<run>/checkpoint``. That is
enough for final layer-delta profiles, but not for "what layers changed as the
epoch advanced". This script makes that limitation explicit:

* if ``checkpoint_epoch_*`` / ``epoch_*`` snapshots exist, it computes a
  per-epoch per-layer RMS relative delta and renders a heatmap;
* if only the final checkpoint exists, it writes a status row saying the
  timeline is not reconstructable from saved artifacts.

Usage:
    python scripts/layer_delta_timeline.py --runs 'lw_*'

Outputs per run:
    runs/<run>/eval/layer_delta_timeline.csv
    runs/<run>/eval/layer_delta_timeline.png  (only when snapshots exist)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfupdate.eval.weight_deltas import (full_ft_deltas, load_state,
                                           lora_deltas, per_layer_profile)

RUNS = Path("runs")
EPOCH_RE = re.compile(r"(?:checkpoint_)?epoch[_-]?(\d+)$")


def base_snapshot(model_name: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_name, allow_patterns=["*.safetensors"]))


def model_name(run_dir: Path) -> str:
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    return cfg.get("model", {}).get("name", "Qwen/Qwen3-0.6B")


def epoch_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    search_roots = [run_dir, run_dir / "checkpoints"]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            m = EPOCH_RE.search(path.name)
            if m and any(path.glob("*.safetensors")):
                candidates.append((int(m.group(1)), path))
    return sorted(set(candidates), key=lambda kv: kv[0])


def checkpoint_deltas(base: dict, ckpt: Path) -> pd.DataFrame:
    if (ckpt / "adapter_config.json").exists():
        acfg = json.loads((ckpt / "adapter_config.json").read_text())
        scaling = acfg["lora_alpha"] / acfg["r"]
        adapter = load_state(next(ckpt.glob("adapter_model.safetensors")))
        return lora_deltas(base, adapter, scaling)
    return full_ft_deltas(base, load_state(ckpt))


def plot_timeline(df: pd.DataFrame, out: Path) -> None:
    mat = df.pivot(index="layer", columns="epoch", values="rms_rel_delta")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
    ax.set_xlabel("epoch")
    ax.set_ylabel("layer")
    ax.set_title(out.parent.parent.name + " layer-delta timeline", fontsize=10)
    ax.set_xticks(range(len(mat.columns)), mat.columns, fontsize=7)
    step = max(1, len(mat.index) // 12)
    ax.set_yticks(range(0, len(mat.index), step),
                  [str(v) for v in mat.index[::step]], fontsize=7)
    fig.colorbar(im, ax=ax, label="RMS relative weight delta")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def process_run(run_dir: Path, base_cache: dict[str, dict]) -> str:
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "layer_delta_timeline.csv"
    ckpts = epoch_checkpoints(run_dir)
    if not ckpts:
        status = pd.DataFrame([{
            "run": run_dir.name,
            "status": "missing_epoch_checkpoints",
            "message": "Only the final checkpoint is present; epoch-wise layer movement is not reconstructable.",
        }])
        status.to_csv(out_csv, index=False)
        return "missing"

    mname = model_name(run_dir)
    if mname not in base_cache:
        base_cache[mname] = load_state(base_snapshot(mname))
    rows = []
    for epoch, ckpt in ckpts:
        prof = per_layer_profile(checkpoint_deltas(base_cache[mname], ckpt))
        for layer, value in prof.items():
            rows.append({
                "run": run_dir.name,
                "epoch": epoch,
                "layer": int(layer),
                "rms_rel_delta": float(value),
                "status": "ok",
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    plot_timeline(df, out_dir / "layer_delta_timeline.png")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*")
    args = ap.parse_args()
    base_cache: dict[str, dict] = {}
    counts = {"ok": 0, "missing": 0}
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or not fnmatch.fnmatch(run_dir.name, args.runs):
            continue
        if not (run_dir / "config.yaml").exists() or not (run_dir / "checkpoint").exists():
            continue
        counts[process_run(run_dir, base_cache)] += 1
    print(f"layer-delta timelines: {counts['ok']} ok, {counts['missing']} missing snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
