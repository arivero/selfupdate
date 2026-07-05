"""Graft/ablate localization curves for a trained run (M4).

Usage:
    python scripts/layer_swap.py --run kd_lora_kl_hi_e60_v3_14b_rag [--limit 8] [--layers 1 7 14 21 28]

Loads the base model and the run checkpoint, then for each layer L reports:
graft CER (base model + trained block L) and ablate CER (trained model with
block L reverted to init). Writes runs/<run>/eval/layer_swap.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.layer_swap import swap_curves
from selfupdate.eval.weight_deltas import load_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    ckpt = Path("runs") / args.run / "checkpoint"
    if (ckpt / "adapter_config.json").exists():
        sys.exit("layer_swap operates on full-FT checkpoints; merge LoRA first")

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
    trained = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
    base.to(cfg.model.device).eval()
    trained.to(cfg.model.device).eval()

    # state dicts on CPU for copying blocks in/out
    base_state = load_state(_snapshot(cfg.model.name))
    trained_state = load_state(ckpt)

    records = load_jsonl(cfg.data.examples_path)
    rows = swap_curves(base, trained_state, base_state, trained, tok, records,
                       limit=args.limit, layers=args.layers)
    out = Path("runs") / args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "layer_swap.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {out / 'layer_swap.csv'}")


def _snapshot(model_name: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name)


if __name__ == "__main__":
    main()
