"""Recitation evaluation: can the student produce the poem without context?

Greedy generation from the student prompt (shared_prefix + student_stub +
shared_mid — no privileged block), compared to the gold answer text with CER
(jiwer), line-level exact match, and longest-correct-prefix length.
"""

from __future__ import annotations

import jiwer
import torch


def student_prompt(record: dict) -> str:
    return record["shared_prefix"] + record.get("student_stub", "") + record["shared_mid"]


def normalize_verse(text: str) -> str:
    """Whitespace-insensitive comparison: models often emit markdown-style
    trailing double spaces or blank lines between stanzas; those are
    formatting, not recall errors."""
    lines = [" ".join(l.split()) for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


@torch.no_grad()
def recite_one(model, tokenizer, record: dict, max_extra_tokens: int = 48) -> dict:
    gold = record["answer_text"]
    prompt_ids = tokenizer.encode(student_prompt(record), add_special_tokens=False)
    gold_len = len(tokenizer.encode(gold, add_special_tokens=False))
    input_ids = torch.tensor([prompt_ids], device=model.device)
    out = model.generate(
        input_ids,
        max_new_tokens=gold_len + max_extra_tokens,
        do_sample=False,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)
    text = normalize_verse(raw)
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
    return {
        "example_id": record["example_id"],
        "cer": cer,
        "line_exact": exact / len(gold_lines),
        "prefix_lines": prefix,
        "n_gold_lines": len(gold_lines),
        "text": text,
    }


@torch.no_grad()
def recite_eval(model, tokenizer, records: list[dict], limit: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    subset = records[:limit] if limit else records
    results = [recite_one(model, tokenizer, r) for r in subset]
    if was_training:
        model.train()
    mean = lambda k: sum(r[k] for r in results) / len(results)
    return {
        "cer": mean("cer"),
        "line_exact": mean("line_exact"),
        "prefix_lines": mean("prefix_lines"),
        "n": len(results),
        "per_example": results,
    }
