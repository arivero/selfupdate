"""Teacher-ceiling diagnostic: greedy recitation WITH the context in prompt.

The student is distilled toward the teacher's distribution, so the teacher's
own greedy recitation quality (with context) upper-bounds what distillation
can transfer. If this CER is poor, fix the teacher prompt before blaming the
training method.

Usage: python scripts/teacher_recite.py [--limit 8] [--show 2]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jiwer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--show", type=int, default=2, help="print N sample generations")
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
    model.to(cfg.model.device)
    model.eval()

    records = load_jsonl(cfg.data.examples_path)[: args.limit]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cers = []
    for i, r in enumerate(records):
        prompt = r["shared_prefix"] + r["privileged"] + r["shared_mid"]
        ids = tok.encode(prompt, add_special_tokens=False)
        gold = r["answer_text"]
        gold_len = len(tok.encode(gold, add_special_tokens=False))
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids], device=model.device),
                max_new_tokens=gold_len + 48, do_sample=False,
                eos_token_id=im_end, pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, len(ids):], skip_special_tokens=True).strip()
        cer = jiwer.cer(gold, text) if text else 1.0
        cers.append(cer)
        print(f"{r['example_id']}: teacher-with-context CER {cer:.3f}")
        if i < args.show:
            print(f"  GOLD: {gold[:160]!r}")
            print(f"  GEN : {text[:160]!r}")
    print(f"mean teacher recitation CER: {sum(cers) / len(cers):.4f}")


if __name__ == "__main__":
    main()
