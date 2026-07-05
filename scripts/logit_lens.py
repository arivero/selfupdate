"""Logit-lens depth profiles for base vs trained checkpoints (M4).

Usage:
    python scripts/logit_lens.py --run kd_lora_kl_hi_e60_v3_14b_rag [--limit 24]

Writes runs/<run>/eval/logit_lens.csv and a comparison plot
runs/<run>/eval/logit_lens.png (base model in grey, trained in color).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.logit_lens import gold_logprob_by_layer
from selfupdate.masking import ContextMasker, SegmentedExample


def build_pairs(cfg, tok):
    masker = ContextMasker(tok)
    return [masker.build(SegmentedExample.from_record(r))
            for r in load_jsonl(cfg.data.examples_path)]


def profile(model_src, cfg, tok, pairs, limit):
    if (Path(model_src) / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, model_src).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_src, dtype=torch.bfloat16)
    model.to(cfg.model.device).eval()
    prof = gold_logprob_by_layer(
        model, tok, pairs, device=cfg.model.device, limit=limit,
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
    )
    del model
    torch.cuda.empty_cache()
    return prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    pairs = build_pairs(cfg, tok)
    ckpt = Path("runs") / args.run / "checkpoint"

    base_prof = profile(cfg.model.name, cfg, tok, pairs, args.limit)
    trained_prof = profile(str(ckpt), cfg, tok, pairs, args.limit)

    df = pd.DataFrame({
        "layer": list(base_prof.keys()),
        "base_logprob": list(base_prof.values()),
        "trained_logprob": [trained_prof[L] for L in base_prof],
    })
    out = Path("runs") / args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "logit_lens.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.layer, df.base_logprob, marker="o", color="grey", label="base")
    ax.plot(df.layer, df.trained_logprob, marker="o", label=args.run)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean reference-token logprob (student input, no context)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "logit_lens.png", dpi=150)
    print(df.to_string(index=False))
    print(f"wrote {out / 'logit_lens.csv'} and .png")


if __name__ == "__main__":
    main()
