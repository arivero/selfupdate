"""Chained full-poem recitation — the Pierre Menard test.

Round after round, the model is asked to continue the poem. Two modes:
- self-chained: the cue is the model's OWN last generated verse (errors
  compound; the honest long-recall metric)
- gold-anchored: the cue is always the gold verse at that offset (per-segment
  recall independent of drift)

Reports chained CER over the whole poem, verses-until-first-error, and the
fraction of rounds whose cue the model continued correctly.

Usage: python scripts/recite_long.py --checkpoint runs/<r>/checkpoint [--window 24]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jiwer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.chatfmt import render_rag_for, stop_token_id
from selfupdate.config import load_config
from selfupdate.data.poem import _continuation_question, load_poem
from selfupdate.eval.recite import normalize_verse


@torch.no_grad()
def _continue_from(model, tok, cue: str, window: int) -> list[str]:
    ex = render_rag_for(tok, "chain", _continuation_question(cue, window), "", "")
    prompt = ex.shared_prefix + ex.shared_mid
    ids = tok.encode(prompt, add_special_tokens=False)
    out = model.generate(
        torch.tensor([ids], device=model.device),
        max_new_tokens=window * 14 + 32, do_sample=False,
        eos_token_id=stop_token_id(tok),
        pad_token_id=tok.eos_token_id,
    )
    text = normalize_verse(tok.decode(out[0, len(ids):], skip_special_tokens=True))
    return [l for l in text.split("\n") if l][:window]


def chain(model, tok, gold: list[str], window: int, self_chained: bool) -> dict:
    got: list[str] = []
    pos = 0
    rounds_ok = 0
    rounds = 0
    while pos + 1 < len(gold):
        cue = (got[-1] if (self_chained and got) else gold[pos])
        lines = _continue_from(model, tok, cue, min(window, len(gold) - pos - 1))
        if not lines:
            break
        rounds += 1
        want = gold[pos + 1: pos + 1 + len(lines)]
        if lines[0].strip() == want[0].strip():
            rounds_ok += 1
        got.extend(lines)
        pos += len(lines)
    hyp = "\n".join(got)
    ref = "\n".join(gold[1: 1 + len(got)]) if got else ""
    cer = jiwer.cer(normalize_verse(ref), normalize_verse(hyp)) if hyp and ref else 1.0
    prefix = 0
    for g, h in zip(gold[1:], got):
        if g.strip() != h.strip():
            break
        prefix += 1
    return {"mode": "self" if self_chained else "anchored", "chained_cer": cer,
            "verses_generated": len(got), "verses_until_first_error": prefix,
            "rounds": rounds, "rounds_first_line_ok": rounds_ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config(args.config, args.experiment)

    ckpt = Path(args.checkpoint)
    if (ckpt / "adapter_config.json").exists():
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(ckpt)
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.bfloat16), ckpt)
    else:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
    model.to(cfg.model.device).eval()

    gold = [v.text for v in load_poem(cfg.data.poem_path)]
    results = [chain(model, tok, gold, args.window, sc) for sc in (False, True)]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    out = ckpt.parent / "eval"
    out.mkdir(exist_ok=True)
    (out / "recite_long.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"wrote {out / 'recite_long.json'}")


if __name__ == "__main__":
    main()
