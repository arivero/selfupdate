"""Per-layer weight-delta profiles for finished runs: which layers moved?

For every runs/<name>/checkpoint with no eval/weight_deltas.csv yet, compute
the relative Frobenius delta vs the base model per (layer, module), write
the CSV, and print the per-layer RMS profile with the top-3 most-modified
layers. CPU-only; safe to run alongside GPU training.

Usage: python compressed/delta_profiles.py [--runs lw_i_*] [--model Qwen/Qwen3-0.6B]
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import fnmatch
import sys
from pathlib import Path


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
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
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
