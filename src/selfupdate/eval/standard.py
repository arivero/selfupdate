"""Pinned standard-benchmark capability probes.

This is deliberately separate from the corpus-recall evaluator: its data is
external to training and it measures general multiple-choice capability.  The
CLI damage evaluator and the in-training epoch probe call the same functions,
so the fast gate is a strict subset of the final 100-item-per-task report.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset


STANDARD_TASKS = ("arc_easy", "arc_challenge", "hellaswag")

# Explicit Hub revisions make a result reproducible even if a dataset card or
# default branch changes after a campaign has started.  ARC-Easy is additionally
# vendored as data/eval/arc_easy_v1.json for the fixed standard subset.
BENCHMARK_REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}


def _arc_examples(config: str, split: str, limit: int | None) -> list[dict]:
    pinned_name = {
        "ARC-Easy": "arc_easy_v1.json",
        "ARC-Challenge": "arc_challenge_v1.json",
    }.get(config)
    if pinned_name:
        pinned = Path("data/eval") / pinned_name
        if pinned.exists():
            rows = json.loads(pinned.read_text())["items"]
            return rows[:limit] if limit else rows
    ds = load_dataset("allenai/ai2_arc", config, split=split,
                      revision=BENCHMARK_REVISIONS["allenai/ai2_arc"])
    rows = []
    for row in ds:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        answer = str(row["answerKey"])
        if answer in labels:
            target = labels.index(answer)
        elif answer.isdigit() and str(int(answer)) in labels:
            target = labels.index(str(int(answer)))
        else:
            continue
        rows.append({
            "id": row.get("id"),
            "prompt": f"Question: {row['question'].strip()}\nAnswer:",
            "choices": [f" {t.strip()}" for t in texts],
            "target": target,
        })
        if limit and len(rows) >= limit:
            break
    return rows


def _hellaswag_examples(split: str, limit: int | None) -> list[dict]:
    pinned = Path("data/eval/hellaswag_v1.json")
    if pinned.exists():
        rows = json.loads(pinned.read_text())["items"]
        return rows[:limit] if limit else rows
    ds = load_dataset("Rowan/hellaswag", split=split,
                      revision=BENCHMARK_REVISIONS["Rowan/hellaswag"])
    rows = []
    for row in ds:
        rows.append({
            "id": row.get("ind"),
            "prompt": f"{row['ctx_a']} {row['ctx_b']}".strip(),
            "choices": [f" {e.strip()}" for e in row["endings"]],
            "target": int(row["label"]),
        })
        if limit and len(rows) >= limit:
            break
    return rows


@lru_cache(maxsize=None)
def task_examples(task: str, limit: int | None) -> tuple[dict, ...]:
    """Deterministic pinned examples, cached once per process."""
    if task == "arc_easy":
        rows = _arc_examples("ARC-Easy", "validation", limit)
    elif task == "arc_challenge":
        rows = _arc_examples("ARC-Challenge", "validation", limit)
    elif task == "hellaswag":
        rows = _hellaswag_examples("validation", limit)
    else:
        raise ValueError(f"unknown standard task {task!r}")
    return tuple(rows)


def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@torch.no_grad()
def _score_pairs(model, tok, pairs: list[tuple[str, str]], device: str) -> list[float]:
    texts = [p + c for p, c in pairs]
    # Padding must be right: continuation boundaries below are indexed from
    # the sequence start.  Several fleet tokenizers default to left padding.
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="right",
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits

    scores = []
    for i, (prompt, choice) in enumerate(pairs):
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        choice_ids = tok.encode(choice, add_special_tokens=False)
        start = len(prompt_ids)
        end = start + len(choice_ids)
        if not choice_ids or end > int(attention_mask[i].sum().item()):
            scores.append(-math.inf)
            continue
        row_logits = logits[i, start - 1:end - 1].float()
        targets = input_ids[i, start:end].to(row_logits.device)
        nll = F.cross_entropy(row_logits, targets, reduction="sum").item()
        scores.append(-nll / max(1, len(choice_ids)))
    return scores


def evaluate_task(model, tok, task: str, limit: int | None, batch_size: int,
                  device: str, *, keep_examples: bool = True) -> dict:
    """Score one fixed multiple-choice subset by option likelihood."""
    examples = task_examples(task, limit)
    flat, owners = [], []
    for ex_i, ex in enumerate(examples):
        for choice_i, choice in enumerate(ex["choices"]):
            flat.append((ex["prompt"], choice))
            owners.append((ex_i, choice_i))

    scores_by_example = [[-math.inf] * len(ex["choices"]) for ex in examples]
    for batch, owner_batch in zip(_chunks(flat, batch_size), _chunks(owners, batch_size)):
        scores = _score_pairs(model, tok, batch, device)
        for (ex_i, choice_i), score in zip(owner_batch, scores):
            scores_by_example[ex_i][choice_i] = score

    correct = 0
    per_example = []
    for ex, scores in zip(examples, scores_by_example):
        pred = max(range(len(scores)), key=lambda i: scores[i])
        ok = pred == ex["target"]
        correct += int(ok)
        if keep_examples:
            per_example.append({
                "id": ex["id"], "target": ex["target"], "pred": pred,
                "correct": ok, "scores": scores,
            })
    result = {
        "task": task,
        "n": len(examples),
        "accuracy": correct / len(examples) if examples else float("nan"),
    }
    if keep_examples:
        result["per_example"] = per_example
    return result


def evaluate_standard(model, tok, *, tasks: tuple[str, ...] = STANDARD_TASKS,
                      limit: int | None = 100, batch_size: int = 16,
                      device: str = "cuda", keep_examples: bool = True) -> dict:
    """Evaluate a fixed standard suite and restore the caller's model mode."""
    unknown = set(tasks) - set(STANDARD_TASKS)
    if unknown:
        raise ValueError(f"unknown standard task(s): {sorted(unknown)}")
    was_training = model.training
    model.eval()
    try:
        results = {
            task: evaluate_task(model, tok, task, limit, batch_size, device,
                                keep_examples=keep_examples)
            for task in tasks
        }
    finally:
        if was_training:
            model.train()
    accs = [r["accuracy"] for r in results.values()]
    return {
        "tasks": results,
        "macro_accuracy": sum(accs) / len(accs) if accs else float("nan"),
        "limit": limit,
        "batch_size": batch_size,
        "benchmark_revisions": {
            "arc_easy": "data/eval/arc_easy_v1.json",
            "arc_challenge": "data/eval/arc_challenge_v1.json",
            "hellaswag": "data/eval/hellaswag_v1.json",
        },
    }
