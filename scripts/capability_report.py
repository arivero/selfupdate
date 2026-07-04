"""Capability/forgetting/localization summary for KD runs.

Writes:
- runs/capability_summary.md: compact table for learning and forgetting.
- runs/capability_epoch_curves.csv: train/eval trajectory by epoch.
- runs/capability_top_layers.md: final or latest-epoch layer-localization table.
- runs/<run>/eval/lora_layer_deltas.csv when requested with --recompute-deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import yaml
from huggingface_hub import snapshot_download

from selfupdate.eval.weight_deltas import load_state, lora_deltas, per_layer_profile
from selfupdate.utils.runlog import read_metrics


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def recite_files(run_dir: Path) -> list[Path]:
    eval_dir = run_dir / "eval"
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


def recite_label(run_dir: Path, p: Path) -> str:
    rel = p.relative_to(run_dir / "eval")
    if rel == Path("recite.json"):
        return "final"
    if rel.name == "recite.json" and len(rel.parts) > 1:
        return rel.parts[0]
    return p.stem


def run_summary(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ms = read_metrics(run_dir)
    trains = [m for m in ms if m.get("kind") == "train"]
    evals = [m for m in ms if m.get("kind") == "eval"]
    fulls = [
        (p, r)
        for p in recite_files(run_dir)
        for r in [json.loads(p.read_text())]
        if isinstance(r, dict) and "cer" in r
    ]
    default = next((r for p, r in fulls if recite_label(run_dir, p) == "final"), None)
    best_full = min((r for _, r in fulls), key=lambda r: r.get("cer", float("inf"))) if fulls else {}
    rec = default or best_full or {}
    best_full_path = next((p for p, r in fulls if r is best_full), None) if fulls else None
    first_eval = evals[0] if evals else {}
    best_eval = min(evals, key=lambda m: m.get("cer", float("inf"))) if evals else {}
    last_eval = evals[-1] if evals else {}
    return {
        "run": run_dir.name,
        "model": cfg["model"]["name"],
        "mode": cfg["mask"]["mode"],
        "data": Path(cfg["data"]["examples_path"]).name,
        "epochs": cfg["train"]["epochs"],
        "items": len(trains),
        "loss_first20": _mean([m["loss"] for m in trains[:20]]),
        "loss_last20": _mean([m["loss"] for m in trains[-20:]]),
        "train_cer_first": first_eval.get("cer"),
        "train_cer_best": best_eval.get("cer"),
        "train_cer_last": last_eval.get("cer"),
        "gen_ce_first": first_eval.get("gen_ce"),
        "gen_ce_last": last_eval.get("gen_ce"),
        "full_cer": rec.get("cer"),
        "full_line_exact": rec.get("line_exact"),
        "full_gen_ce": rec.get("general", {}).get("mean_ce"),
        "best_full_variant": recite_label(run_dir, best_full_path) if best_full_path else "",
        "best_full_cer": best_full.get("cer"),
        "best_full_line_exact": best_full.get("line_exact"),
        "best_full_gen_ce": best_full.get("general", {}).get("mean_ce"),
    }


def epoch_rows(run_dir: Path) -> list[dict]:
    ms = read_metrics(run_dir)
    rows = []
    for m in ms:
        if m.get("kind") != "eval":
            continue
        rows.append({
            "run": run_dir.name,
            "epoch": m.get("epoch"),
            "train_cer": m.get("cer"),
            "train_line_exact": m.get("line_exact"),
            "gen_ce": m.get("gen_ce"),
            "minutes": m.get("minutes"),
            "vram_gb": m.get("vram_gb"),
        })
    return rows


def top_layer_rows(run_dir: Path, top_k: int = 5) -> list[dict]:
    final_p = run_dir / "eval" / "lora_layer_deltas.csv"
    if final_p.exists():
        df = pd.read_csv(final_p).sort_values("rel_delta_rms", ascending=False).head(top_k)
        return [
            {
                "run": run_dir.name,
                "source": "final_normalized",
                "epoch": "",
                "layer": int(r.layer),
                "score": float(r.rel_delta_rms),
            }
            for r in df.itertuples()
        ]
    by_epoch_p = run_dir / "eval" / "lora_layer_deltas_by_epoch.csv"
    if by_epoch_p.exists():
        df = pd.read_csv(by_epoch_p)
        if df.empty:
            return []
        latest = df["epoch"].max()
        df = df[df["epoch"] == latest].sort_values("adapter_update_rms", ascending=False).head(top_k)
        return [
            {
                "run": run_dir.name,
                "source": "latest_epoch_raw_adapter",
                "epoch": int(r.epoch),
                "layer": int(r.layer),
                "score": float(r.adapter_update_rms),
            }
            for r in df.itertuples()
        ]
    return []


def write_layer_delta(run_dir: Path, *, recompute: bool = False) -> Path | None:
    out = run_dir / "eval" / "lora_layer_deltas.csv"
    if out.exists() and not recompute:
        return out
    if not recompute:
        return None
    ckpt = run_dir / "checkpoint"
    acfg_p = ckpt / "adapter_config.json"
    adapter_p = ckpt / "adapter_model.safetensors"
    if not (acfg_p.exists() and adapter_p.exists()):
        return None
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    acfg = json.loads(acfg_p.read_text())
    base_state = load_state(snapshot_download(cfg["model"]["name"]))
    df = lora_deltas(base_state, load_state(adapter_p), acfg["lora_alpha"] / acfg["r"])
    prof = per_layer_profile(df).reset_index()
    prof.columns = ["layer", "rel_delta_rms"]
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.to_csv(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names; default: all runs with config.yaml")
    ap.add_argument(
        "--recompute-deltas", action="store_true",
        help="recompute normalized LoRA layer deltas from checkpoints; expensive",
    )
    args = ap.parse_args()

    run_dirs = [Path("runs") / r for r in args.runs] if args.runs else [
        d for d in sorted(Path("runs").iterdir()) if (d / "config.yaml").exists()
    ]
    rows = [run_summary(d) for d in run_dirs]
    df = pd.DataFrame(rows)
    out = Path("runs/capability_summary.md")
    out.write_text(df.map(_fmt).to_markdown(index=False))
    print(df.map(_fmt).to_markdown(index=False))
    print(f"wrote {out}")

    epoch_df = pd.DataFrame([row for d in run_dirs for row in epoch_rows(d)])
    epoch_out = Path("runs/capability_epoch_curves.csv")
    epoch_df.to_csv(epoch_out, index=False)
    print(f"wrote {epoch_out}")

    for d in run_dirs:
        if recite_files(d):
            wrote = write_layer_delta(d, recompute=args.recompute_deltas)
            if wrote:
                print(f"available {wrote}")

    layer_df = pd.DataFrame([row for d in run_dirs for row in top_layer_rows(d)])
    layer_out = Path("runs/capability_top_layers.md")
    layer_out.write_text(layer_df.map(_fmt).to_markdown(index=False) if not layer_df.empty else "")
    print(f"wrote {layer_out}")


if __name__ == "__main__":
    main()
