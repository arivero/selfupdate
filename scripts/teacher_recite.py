"""Teacher-ceiling diagnostic: greedy recitation WITH the context in prompt.

The student is distilled toward the teacher's distribution, so the teacher's
own greedy recitation quality (with context) upper-bounds what distillation
can transfer. If this CER is poor, fix the teacher prompt before blaming the
training method.

Usage: python scripts/teacher_recite.py [--limit 8] [--show 2] [--out path.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jiwer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.data.dataset import load_jsonl
from selfupdate.eval.recite import normalize_verse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=2, help="print N sample generations")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
    model.to(cfg.model.device)
    model.eval()

    from selfupdate.chatfmt import adapt_records, stop_token_id

    records = adapt_records(load_jsonl(cfg.data.examples_path), tok)
    if args.limit:
        records = records[: args.limit]
    im_end = stop_token_id(tok)
    results = []
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
        text = normalize_verse(tok.decode(out[0, len(ids):], skip_special_tokens=True))
        gold = normalize_verse(gold)
        cer = jiwer.cer(gold, text) if text else 1.0
        gold_lines = gold.split("\n")
        got_lines = text.split("\n")
        exact = sum(1 for g, h in zip(gold_lines, got_lines) if g == h)
        prefix = 0
        for g, h in zip(gold_lines, got_lines):
            if g != h:
                break
            prefix += 1
        result = {
            "example_id": r["example_id"],
            "cer": cer,
            "line_exact": exact / len(gold_lines),
            "prefix_lines": prefix,
            "n_gold_lines": len(gold_lines),
            "text": text,
        }
        results.append(result)
        print(f"{r['example_id']}: teacher-with-context CER {cer:.3f}")
        if i < args.show:
            print(f"  GOLD: {gold[:160]!r}")
            print(f"  GEN : {text[:160]!r}")
    mean = lambda k: sum(r[k] for r in results) / len(results)
    summary = {
        "model": cfg.model.name,
        "examples_path": cfg.data.examples_path,
        "prompt": "shared_prefix + privileged + shared_mid",
        "cer": mean("cer"),
        "line_exact": mean("line_exact"),
        "prefix_lines": mean("prefix_lines"),
        "n": len(results),
        "per_example": results,
    }
    print(
        "mean teacher-with-context: "
        f"CER {summary['cer']:.4f} line-exact {summary['line_exact']:.4f} "
        f"prefix-lines {summary['prefix_lines']:.2f}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
