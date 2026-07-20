"""Logit-lens depth profiles for base vs trained checkpoints (M4).

Usage:
    python compressed/logit_lens.py --run lw_r_slide8_0p6b_rag [--limit 24]

Writes runs/<run>/eval/logit_lens.csv and a comparison plot
runs/<run>/eval/logit_lens.png (base model in grey, trained in color).
"""


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.logit_lens import reference_logprob_by_layer
from selfupdate.masking import ContextMasker, SegmentedExample


def build_pairs(cfg, tok):
    masker = ContextMasker(tok)
    return [masker.build(SegmentedExample.from_record(r))
            for r in load_jsonl(cfg.data.examples_path)]


def profile(model_src, cfg, tok, pairs, limit, translators=None):
    if (Path(model_src) / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, model_src).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_src, dtype=torch.bfloat16)
    model.to(cfg.model.device).eval()
    prof = reference_logprob_by_layer(
        model, tok, pairs, device=cfg.model.device, limit=limit,
        rebase_gap=(cfg.mask.compaction in ("stub_gap", "remove_gap")),
        translators=translators,
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
    ap.add_argument("--lens", choices=("raw", "tuned"), default="raw")
    ap.add_argument("--translators", default="runs/tuned_lens_0.6B/translators.safetensors")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    translators = None
    if args.lens == "tuned":
        from selfupdate.train.tuned_lens import load_translators

        translators = load_translators(args.translators, cfg.model.device)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    pairs = build_pairs(cfg, tok)
    ckpt = Path("runs") / args.run / "checkpoint"

    base_prof = profile(cfg.model.name, cfg, tok, pairs, args.limit, translators)
    trained_prof = profile(str(ckpt), cfg, tok, pairs, args.limit, translators)

    df = pd.DataFrame({
        "layer": list(base_prof.keys()),
        "base_logprob": list(base_prof.values()),
        "trained_logprob": [trained_prof[L] for L in base_prof],
    })
    out = Path("runs") / args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    stem = "logit_lens" if args.lens == "raw" else "logit_lens_tuned"
    df.to_csv(out / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.layer, df.base_logprob, marker="o", color="grey", label="base")
    ax.plot(df.layer, df.trained_logprob, marker="o", label=args.run)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean reference-token logprob (student input, no context)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=150)
    print(df.to_string(index=False))
    print(f"wrote {out / f'{stem}.csv'} and .png")


if __name__ == "__main__":
    main()
