"""The three recall tasks (owner directive 2026-07-10).

Evaluation of training is exactly three tasks over the corpus:

1. ``next``  — "tell me the line that follows X" / "end this paragraph"
2. ``prev``  — "tell me the line that precedes X" / "start this paragraph"
3. ``cloze`` — "fill the words I deleted from this paragraph", with varying
               deletion count and paragraph size.

Metrics are plain accuracies (CER and the other recovery metrics are
retired from the active eval surface): ``exact`` = fraction of items whose
normalized answer matches the reference exactly; ``word_acc`` = fraction of
reference words recovered in order (longest common subsequence / reference
length). Task sets are DETERMINISTIC (seeded) so runs stay comparable.
"""

from __future__ import annotations

import random
import re

import torch

QUESTIONS = {
    "next_line": "¿Qué línea sigue inmediatamente a esta?\n«{x}»",
    "end_block": "Termina este párrafo, continuando exactamente el texto:\n«{x}»",
    "prev_line": "¿Qué línea viene inmediatamente antes de esta?\n«{x}»",
    "start_block": "Este es el final de un párrafo. Escribe exactamente su comienzo:\n«{x}»",
    "cloze": ("He borrado {n} palabras de este párrafo, marcadas con ___. "
              "Escribe únicamente las palabras que faltan, en orden:\n«{x}»"),
}


def corpus_blocks(path: str) -> list[list[str]]:
    """Blank-line-separated blocks of content lines (verse stanzas or prose
    paragraphs); '#' markers are structure, not memorized text."""
    blocks, cur = [], []
    for raw in open(path, encoding="utf-8").read().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            if len(cur) >= 2:
                blocks.append(cur)
            cur = []
            continue
        cur.append(line)
    if len(cur) >= 2:
        blocks.append(cur)
    return blocks


def build_tasks(poem_path: str, seed: int = 17, n_per_task: int = 24,
                cloze_deletions: tuple = (1, 2, 4, 8),
                block_lines: tuple = (2, 4)) -> list[dict]:
    """Deterministic task set: n_per_task items for each of next/prev/cloze.
    next/prev alternate line-flavor and paragraph-flavor; cloze cycles the
    deletion counts and paragraph sizes."""
    rng = random.Random(seed)
    blocks = corpus_blocks(poem_path)
    items = []
    for k in range(n_per_task):  # next
        b = rng.choice([b for b in blocks if len(b) >= 2])
        if k % 2 == 0:
            i = rng.randrange(len(b) - 1)
            items.append({"task": "next", "kind": "next_line",
                          "x": b[i], "n": 0, "reference": b[i + 1]})
        else:
            cut = max(1, len(b) // 2)
            items.append({"task": "next", "kind": "end_block",
                          "x": "\n".join(b[:cut]), "n": 0,
                          "reference": "\n".join(b[cut:])})
    for k in range(n_per_task):  # prev
        b = rng.choice([b for b in blocks if len(b) >= 2])
        if k % 2 == 0:
            i = rng.randrange(1, len(b))
            items.append({"task": "prev", "kind": "prev_line",
                          "x": b[i], "n": 0, "reference": b[i - 1]})
        else:
            cut = max(1, len(b) // 2)
            items.append({"task": "prev", "kind": "start_block",
                          "x": "\n".join(b[cut:]), "n": 0,
                          "reference": "\n".join(b[:cut])})
    for k in range(n_per_task):  # cloze
        size = block_lines[k % len(block_lines)]
        n_del = cloze_deletions[k % len(cloze_deletions)]
        b = rng.choice([b for b in blocks if len(b) >= size])
        start = rng.randrange(0, len(b) - size + 1)
        words = " ".join(b[start: start + size]).split()
        n_del = min(n_del, max(1, len(words) - 2))
        pos = rng.randrange(0, len(words) - n_del + 1)
        deleted = words[pos: pos + n_del]
        masked = words[:pos] + ["___"] * n_del + words[pos + n_del:]
        items.append({"task": "cloze", "kind": "cloze", "n": n_del,
                      "x": " ".join(masked), "reference": " ".join(deleted)})
    return items


_norm_re = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _norm_re.sub(" ", s.replace("«", "").replace("»", "")).strip()


def _lcs_words(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def score(reference: str, answer: str) -> dict:
    ref, ans = _norm(reference), _norm(answer)
    ref_w, ans_w = ref.split(), ans.split()
    return {
        "exact": float(ref == ans),
        "word_acc": (_lcs_words(ref_w, ans_w) / len(ref_w)) if ref_w else 0.0,
    }


@torch.no_grad()
def tasks_eval(model, tokenizer, poem_path: str, seed: int = 17,
               n_per_task: int = 24, max_extra_tokens: int = 32,
               keep_examples: int = 6) -> dict:
    """Run the three-task battery; returns plain per-task accuracies."""
    from .recite import greedy_generate_positions, strip_think

    items = build_tasks(poem_path, seed=seed, n_per_task=n_per_task)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos is None or eos < 0:
        eos = tokenizer.eos_token_id
    device = next(model.parameters()).device
    agg: dict[str, list[dict]] = {}
    examples = []
    for it in items:
        q = QUESTIONS[it["kind"]].format(x=it["x"], n=it["n"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)],
                           device=device)
        budget = len(tokenizer.encode(it["reference"])) + max_extra_tokens
        out = greedy_generate_positions(
            model, ids, torch.arange(ids.shape[1], device=device)[None],
            max_new_tokens=budget, eos_id=eos)
        answer = strip_think(tokenizer.decode(out, skip_special_tokens=True))
        s = score(it["reference"], answer)
        s["n_deleted"] = it["n"]
        agg.setdefault(it["task"], []).append(s)
        if len(examples) < keep_examples:
            examples.append({"kind": it["kind"], "q": q,
                             "reference": it["reference"],
                             "answer": answer.strip()[:200], **s})
    result = {"seed": seed, "n_per_task": n_per_task, "tasks": {}}
    for task, rows in agg.items():
        result["tasks"][task] = {
            "n": len(rows),
            "exact": sum(r["exact"] for r in rows) / len(rows),
            "word_acc": sum(r["word_acc"] for r in rows) / len(rows),
        }
    if "cloze" in agg:
        by_n: dict[int, list] = {}
        for r in agg["cloze"]:
            by_n.setdefault(r["n_deleted"], []).append(r["word_acc"])
        result["tasks"]["cloze"]["by_deletions"] = {
            str(n): sum(v) / len(v) for n, v in sorted(by_n.items())}
    result["overall_word_acc"] = (sum(r["word_acc"] for rows in agg.values()
                                      for r in rows)
                                  / max(1, sum(len(r) for r in agg.values())))
    result["examples"] = examples
    return result
