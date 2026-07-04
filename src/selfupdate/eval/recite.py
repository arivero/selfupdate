"""Recitation evaluation: can the student produce the poem without context?

Greedy generation from the student prompt (shared_prefix + student_stub +
shared_mid — no privileged block), compared to the gold answer text with CER
(jiwer), line-level exact match, and longest-correct-prefix length.
"""

from __future__ import annotations

import jiwer
import torch

from ..chatfmt import adapt_records, stop_token_id


def student_prompt(record: dict) -> str:
    if record.get("interleaved"):
        # thinking_selective: the student sees the kept think runs
        kept = "".join(t for t, is_priv in record["interleaved"] if not is_priv)
        return record["shared_prefix"] + kept + record["shared_mid"]
    return record["shared_prefix"] + record.get("student_stub", "") + record["shared_mid"]


def normalize_verse(text: str) -> str:
    """Whitespace-insensitive comparison: models often emit markdown-style
    trailing double spaces or blank lines between stanzas; those are
    formatting, not recall errors."""
    lines = [" ".join(l.split()) for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def strip_think(text: str) -> str:
    """Reasoning-tuned families (Phi-4-mini, R1-style) open generation with
    a think block; recitation is judged on what follows it. An unclosed
    block (the token budget burned entirely inside <think>) counts as empty
    output — that IS a recitation failure, not a measurement artifact."""
    s = text.lstrip()
    if s.startswith("<think>"):
        end = s.find("</think>")
        return "" if end == -1 else s[end + len("</think>"):]
    # gpt-oss harmony format: special tokens vanish under skip_special_tokens,
    # leaving "analysis<reasoning>assistantfinal<answer>"
    if s.startswith("analysis"):
        end = s.find("assistantfinal")
        return "" if end == -1 else s[end + len("assistantfinal"):]
    return text


def _record_gap(record: dict, tokenizer) -> tuple[int, int]:
    """(position_gap, s0) of a record, recomputed from its segments —
    the gap the stub_gap arm was trained with."""
    enc = lambda t: len(tokenizer.encode(t, add_special_tokens=False)) if t else 0
    stub = record.get("student_stub", "")
    gap = enc(record.get("privileged", "")) - enc(stub)
    s0 = enc(record["shared_prefix"]) + enc(stub)
    return gap, s0


@torch.no_grad()
def greedy_generate_positions(model, input_ids, position_ids, max_new_tokens, eos_id):
    """Greedy decode with explicit RoPE position ids. Needed for the stub_gap
    arm: training rebased the aligned span by position_gap, a geometry HF
    generate cannot express; evaluating at contiguous positions would measure
    an input distribution the model was never trained on."""
    from transformers import DynamicCache

    device = input_ids.device
    cache = DynamicCache()
    out = model(input_ids=input_ids, position_ids=position_ids,
                past_key_values=cache, use_cache=True)
    next_tok = out.logits[0, -1].argmax().item()
    cur_pos = position_ids[0, -1].item()
    generated: list[int] = []
    while len(generated) < max_new_tokens:
        generated.append(next_tok)
        if next_tok == eos_id:
            break
        cur_pos += 1
        out = model(input_ids=torch.tensor([[next_tok]], device=device),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=cache, use_cache=True)
        next_tok = out.logits[0, -1].argmax().item()
    return generated


@torch.no_grad()
def recite_one(model, tokenizer, record: dict, max_extra_tokens: int = 48,
               rebase_gap: bool = False) -> dict:
    gold = record["answer_text"]
    prompt_ids = tokenizer.encode(student_prompt(record), add_special_tokens=False)
    gold_len = len(tokenizer.encode(gold, add_special_tokens=False))
    input_ids = torch.tensor([prompt_ids], device=model.device)
    eos_id = stop_token_id(tokenizer)
    gap = 0
    if rebase_gap:
        gap, s0 = _record_gap(record, tokenizer)
    if gap > 0:
        pos = list(range(s0)) + [p + gap for p in range(s0, len(prompt_ids))]
        gen = greedy_generate_positions(
            model, input_ids, torch.tensor([pos], device=model.device),
            max_new_tokens=gold_len + max_extra_tokens, eos_id=eos_id,
        )
        raw = tokenizer.decode(gen, skip_special_tokens=True)
    else:
        out = model.generate(
            input_ids,
            max_new_tokens=gold_len + max_extra_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)
    text = normalize_verse(strip_think(raw))
    gold = normalize_verse(gold)

    cer = jiwer.cer(gold, text) if text else 1.0
    # prose corpora: newline placement is arbitrary wrapping, not content —
    # cer_flat scores recall independent of line breaks (additive metric)
    cer_flat = (jiwer.cer(gold.replace("\n", " "), text.replace("\n", " "))
                if text else 1.0)
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
        "cer_flat": cer_flat,
        "line_exact": exact / len(gold_lines),
        "prefix_lines": prefix,
        "n_gold_lines": len(gold_lines),
        "text": text,
    }


@torch.no_grad()
def recite_eval(model, tokenizer, records: list[dict], limit: int | None = None,
                rebase_gap: bool = False) -> dict:
    was_training = model.training
    model.eval()
    records = adapt_records(records, tokenizer)
    subset = records[:limit] if limit else records
    results = [recite_one(model, tokenizer, r, rebase_gap=rebase_gap) for r in subset]
    if was_training:
        model.train()
    mean = lambda k: sum(r[k] for r in results) / len(results)
    return {
        "cer": mean("cer"),
        "cer_flat": mean("cer_flat"),
        "line_exact": mean("line_exact"),
        "prefix_lines": mean("prefix_lines"),
        "n": len(results),
        "per_example": results,
    }
