"""Qualitative forgetting probe: a few trivial general-knowledge questions.

general_ce gives a scalar forgetting signal; this shows WHAT degrades. Asks
fixed questions (geography, local institutions) to the base model or a
checkpoint and stores the generations side by side. A model that still knows
where Zaragoza is after memorizing Machado has not been lobotomized.

Usage:
    sanity_chat.py [--experiment cfg.yaml] [--checkpoint runs/x/checkpoint] \
                   [--out runs/x/eval/sanity.json]
Base model of the config is used when --checkpoint is absent.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import stop_token_id
from selfupdate.config import load_config

QUESTIONS = [
    "Where in the world is Zaragoza?",
    "What university is behind the domain unizar.es?",
    "What is the name of the unizar institute known as BIFI?",
    "What is the shortest way to travel from Paris to Moscow?",
    "¿Cómo empieza el Quijote?",  # control: famous text we did NOT train on
]

SYSTEM = "You are a helpful assistant."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    if args.checkpoint and (Path(args.checkpoint) / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(args.checkpoint)
        base = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, args.checkpoint)
    else:
        src = args.checkpoint or cfg.model.name
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    model.to(cfg.model.device)
    model.eval()

    results = []
    for q in QUESTIONS:
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=enc["attention_mask"].to(model.device),
                max_new_tokens=args.max_new_tokens, do_sample=False,
                eos_token_id=stop_token_id(tok),
                pad_token_id=tok.eos_token_id,
            )
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({"question": q, "answer": ans})
        print(f"Q: {q}\nA: {ans}\n")

    out_path = Path(
        args.out
        or (Path(args.checkpoint).parent / "eval" / "sanity.json" if args.checkpoint
            else f"runs/base-sanity-{cfg.model.name.split('/')[-1]}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model": cfg.model.name, "checkpoint": args.checkpoint, "system": SYSTEM,
         "results": results}, ensure_ascii=False, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
