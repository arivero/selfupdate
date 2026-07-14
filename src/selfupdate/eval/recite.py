"""Recitation evaluation: can the student produce the poem without context?

Greedy generation from the student prompt (shared_prefix + student_stub +
shared_mid — no privileged block), compared to the reference answer text with
character error rate, line-level exact match, and longest-correct-prefix length.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from ..chatfmt import adapt_records, stop_token_id


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein character error rate with jiwer-compatible semantics.

    Keeping this tiny dynamic-programming primitive local avoids making every
    training runtime carry an otherwise optional text-metrics package.
    """
    # jiwer's CER default applies Strip before reducing to characters.
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference:
        if not hypothesis:
            return 0.0
        raise ValueError("reference must not be empty when hypothesis is non-empty")
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_char != hyp_char),
            ))
        previous = current
    return previous[-1] / len(reference)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "cuda" in msg and "out of memory" in msg
    )


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
    ref = record["answer_text"]
    prompt_ids = tokenizer.encode(student_prompt(record), add_special_tokens=False)
    ref_len = len(tokenizer.encode(ref, add_special_tokens=False))
    input_ids = torch.tensor([prompt_ids], device=model.device)
    eos_id = stop_token_id(tokenizer)
    gap = 0
    if rebase_gap:
        gap, s0 = _record_gap(record, tokenizer)
    if gap > 0:
        pos = list(range(s0)) + [p + gap for p in range(s0, len(prompt_ids))]
        gen = greedy_generate_positions(
            model, input_ids, torch.tensor([pos], device=model.device),
            max_new_tokens=ref_len + max_extra_tokens, eos_id=eos_id,
        )
        raw = tokenizer.decode(gen, skip_special_tokens=True)
    else:
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=ref_len + max_extra_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)
    return score_recitation(record, raw)


def score_recitation(record: dict, raw: str) -> dict:
    text = normalize_verse(strip_think(raw))
    ref = normalize_verse(record["answer_text"])

    cer = character_error_rate(ref, text) if text else 1.0
    # prose corpora: newline placement is arbitrary wrapping, not content —
    # cer_flat scores recall independent of line breaks (additive metric)
    cer_flat = (character_error_rate(
                    ref.replace("\n", " "), text.replace("\n", " "))
                if text else 1.0)
    ref_lines = ref.split("\n")
    got_lines = text.split("\n")
    exact = sum(1 for g, h in zip(ref_lines, got_lines) if g == h)
    prefix = 0
    for g, h in zip(ref_lines, got_lines):
        if g != h:
            break
        prefix += 1
    return {
        "example_id": record["example_id"],
        "cer": cer,
        "cer_flat": cer_flat,
        "line_exact": exact / len(ref_lines),
        "prefix_lines": prefix,
        "n_reference_lines": len(ref_lines),
        "text": text,
    }


@torch.no_grad()
def recite_eval(model, tokenizer, records: list[dict], limit: int | None = None,
                rebase_gap: bool = False, batch_size: int = 1,
                max_extra_tokens: int = 48,
                bucket_by_length: bool = False,
                score_workers: int | None = None,
                shuffle_seed: int | None = None) -> dict:
    was_training = model.training
    model.eval()
    records = adapt_records(records, tokenizer)
    subset = records[:limit] if limit else records
    if rebase_gap or batch_size <= 1:
        results = []
        for i, r in enumerate(subset, 1):
            print(f"recite eval item {i}/{len(subset)}", flush=True)
            results.append(
                recite_one(model, tokenizer, r, max_extra_tokens=max_extra_tokens,
                           rebase_gap=rebase_gap)
            )
    else:
        results = recite_eval_batched(
            model, tokenizer, subset, batch_size=batch_size,
            max_extra_tokens=max_extra_tokens,
            bucket_by_length=bucket_by_length,
            score_workers=score_workers,
            shuffle_seed=shuffle_seed)
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


@torch.no_grad()
def recite_eval_batched(model, tokenizer, records: list[dict], batch_size: int,
                        max_extra_tokens: int = 48,
                        bucket_by_length: bool = False,
                        score_workers: int | None = None,
                        shuffle_seed: int | None = None) -> list[dict]:
    was_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = stop_token_id(tokenizer)

    work = []
    for i, r in enumerate(records):
        prompt = student_prompt(r)
        ref_len = len(tokenizer.encode(r["answer_text"], add_special_tokens=False))
        work.append((ref_len, i, r, prompt))
    if bucket_by_length:
        # Throughput mode: keeps per-batch max_new_tokens close to the examples
        # being generated while restoring original order in the output.
        work.sort(key=lambda x: x[0])
    elif shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(work)

    results: list[dict | None] = [None] * len(records)
    workers = score_workers if score_workers is not None else min(32, os.cpu_count() or 1)
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    futures: list[tuple[int, Future[dict], int]] = []
    try:
        try:
            start = 0
            cur_batch_size = batch_size
            while start < len(work):
                batch = work[start:start + cur_batch_size]
                print(f"recite eval batch {start + len(batch)}/{len(work)} "
                      f"(batch_size={len(batch)}, requested={batch_size})", flush=True)
                prompts = [x[3] for x in batch]
                max_new = max(x[0] for x in batch) + max_extra_tokens
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=False)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                try:
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=False,
                        eos_token_id=eos_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                except RuntimeError as e:
                    if not _is_cuda_oom(e) or cur_batch_size <= 1:
                        raise
                    cur_batch_size = max(1, cur_batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"recite eval OOM; retrying with batch_size={cur_batch_size}",
                          flush=True)
                    continue
                gen_start = enc["input_ids"].shape[1]
                for row, (_, idx, rec, _prompt) in enumerate(batch):
                    raw = tokenizer.decode(out[row, gen_start:], skip_special_tokens=True)
                    futures.append(
                        (idx, executor.submit(score_recitation, rec, raw), len(batch))
                    )
                start += len(batch)
            for idx, fut, used_batch_size in futures:
                scored = fut.result()
                scored["generation_batch_size"] = used_batch_size
                scored["requested_batch_size"] = batch_size
                results[idx] = scored
        finally:
            executor.shutdown(wait=True)
    finally:
        tokenizer.padding_side = was_padding
    return [r for r in results if r is not None]
