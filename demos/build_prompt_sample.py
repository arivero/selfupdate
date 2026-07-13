"""Materialize a V5 prompt sample to JSONL for the CPU generator demo.

Run with the repo venv (needs selfupdate + the model tokenizer):

  .venv/bin/python demos/build_prompt_sample.py --model Qwen/Qwen3-0.6B --limit 64

Both contenders (generate_torch_cpu.py and generate_vllm_cpu.py) read the
resulting file, so they see byte-identical prompt token ids and budgets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from v5_prompts import DEFAULT_EXAMPLES, DEFAULT_EXPERIMENT, build_prompt_sample

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--examples", default=DEFAULT_EXAMPLES)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompt_sample(tok, examples=args.examples,
                                  experiment=args.experiment, limit=args.limit)
    short = args.model.split("/")[-1].lower()
    out = Path(args.out) if args.out else HERE / "out" / f"prompts_{short}_n{len(prompts)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"meta": True, "model": args.model,
                                 "examples": args.examples,
                                 "experiment": args.experiment}) + "\n")
        for item in prompts:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    lengths = [len(x["ids"]) for x in prompts]
    budgets = [x["budget"] for x in prompts]
    print(f"wrote {len(prompts)} prompts -> {out}")
    print(f"prompt tokens: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} "
          f"max={max(lengths)}  budgets: min={min(budgets)} max={max(budgets)}")


if __name__ == "__main__":
    main()
