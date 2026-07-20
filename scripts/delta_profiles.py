"""Per-layer weight-delta profiles for finished runs: which layers moved?

For every runs/<name>/checkpoint with no eval/weight_deltas.csv yet, compute
the relative Frobenius delta vs the base model per (layer, module), write
the CSV, and print the per-layer RMS profile with the top-3 most-modified
layers. CPU-only; safe to run alongside GPU training.

Usage: python scripts/delta_profiles.py [--runs lw_i_*] [--model Qwen/Qwen3-0.6B]
"""

import argparse
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from selfupdate.eval.weight_deltas import (full_ft_deltas, load_state,
                                           lora_deltas, per_layer_profile)

RUNS = Path("runs")


def base_snapshot(model_name: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_name, allow_patterns=["*.safetensors"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="lw_i_*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_cache: dict[str, dict] = {}
    for run_dir in sorted(RUNS.iterdir()):
        if not fnmatch.fnmatch(run_dir.name, args.runs):
            continue
        ckpt = run_dir / "checkpoint"
        out = run_dir / "eval" / "weight_deltas.csv"
        if not ckpt.exists() or (out.exists() and not args.force):
            continue
        # Stage-scoped v4 runs keep the merged checkpoint at run root but the
        # immutable config snapshot under each stage.  Stage 0 is sufficient:
        # only stage-local placement/ownership differs and model identity does
        # not.  Refuse a genuinely missing snapshot instead of silently using
        # the historical 0.6B default.
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.is_file():
            cfg_path = run_dir / "stage0" / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"no config.yaml or stage0/config.yaml under {run_dir}")
        cfg = yaml.safe_load(cfg_path.read_text())
        model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3-0.6B")
        if model_name not in base_cache:
            base_cache[model_name] = load_state(base_snapshot(model_name))
        base = base_cache[model_name]

        if (ckpt / "adapter_config.json").exists():
            import json

            acfg = json.loads((ckpt / "adapter_config.json").read_text())
            scaling = acfg["lora_alpha"] / acfg["r"]
            adapter = load_state(next(ckpt.glob("adapter_model.safetensors")))
            df = lora_deltas(base, adapter, scaling)
        else:
            df = full_ft_deltas(base, load_state(ckpt))
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        prof = per_layer_profile(df)
        top = prof.sort_values(ascending=False).head(3)
        print(f"{run_dir.name}: wrote {out}")
        print("  profile " + " ".join(f"L{int(k)}:{v:.4f}" for k, v in prof.items()))
        print("  top-hit layers: "
              + ", ".join(f"L{int(k)} ({v:.4f})" for k, v in top.items()), flush=True)


if __name__ == "__main__":
    main()
