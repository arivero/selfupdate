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
length). Words for the LCS are split on ALL readable separators — spaces,
newlines, and punctuation including the verse-joining conventions people
actually use ("/", ",", line returns) — so "verso uno/verso dos" and
"verso uno,\nverso dos" recover the same words (owner directive
2026-07-12; ``exact`` keeps punctuation because exact recitation includes
it). Task sets are DETERMINISTIC (seeded) so runs stay comparable.
"""

from __future__ import annotations

import random
import re

import torch

from ..chatfmt import stop_token_id

QUESTIONS = {
    "next_line": "¿Qué línea sigue inmediatamente a esta?\n«{x}»",
    "end_block": "Termina este párrafo, continuando exactamente el texto:\n«{x}»",
    "prev_line": "¿Qué línea viene inmediatamente antes de esta?\n«{x}»",
    "start_block": "Este es el final de un párrafo. Escribe exactamente su comienzo:\n«{x}»",
    "cloze": ("He borrado {n} palabras de este párrafo, marcadas con ___. "
              "Escribe únicamente las palabras que faltan, en orden:\n«{x}»"),
}

# Stable names for per-epoch training telemetry.  ``tasks_eval`` itself still
# accepts any corpus path, but campaign configs use these names so a combined
# run cannot silently monitor only its Machado half.
RECALL_CORPUS_PATHS = {
    "machado": "data/poem/raw.txt",
    "quijote_ch1": "data/quijote/raw_ch1.txt",
    "quijote_ch4": "data/quijote/raw_ch4.txt",
    "quijote_ch8": "data/quijote/raw_ch8.txt",
    "quijote_ch16": "data/quijote/raw_ch16.txt",
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


def retrieve_window(lines: list[str], block: list[str], pad: int = 4) -> str:
    """Exact-match retrieval ("grepping", owner 2026-07-12): locate the
    block's consecutive lines in the flat corpus and return them ± ``pad``
    context lines. The windowed teacher ceiling must FIND its passage the
    way a retrieval tool would, not receive it by construction."""
    for i in range(len(lines) - len(block) + 1):
        if lines[i] == block[0] and lines[i: i + len(block)] == block:
            lo, hi = max(0, i - pad), min(len(lines), i + len(block) + pad)
            return "\n".join(lines[lo:hi])
    raise ValueError(f"retrieval failed: block starting {block[0]!r} "
                     "not found in corpus")


def retrieve_chapter(poem_path: str, block: list[str]) -> str:
    """Chapter-scope retrieval: the whole structural unit containing the
    block — named part for the verse corpus, capítulo (section) for Quijote
    prose. Shares the chapter-span convention of the v5 dataset builder
    (data/questions.py) so the chapter ceiling sees exactly the training
    arms' RAG form."""
    from ..data.poem import load_poem
    from ..data.questions import _chapter_span

    verses = load_poem(poem_path)
    texts = [v.text for v in verses]
    key = "section" if "quijote" in str(poem_path).lower() else "part"
    for i in range(len(texts) - len(block) + 1):
        if texts[i: i + len(block)] == block:
            lo, hi = _chapter_span(verses, i, key)
            return "\n".join(texts[lo:hi])
    raise ValueError(f"retrieval failed: block starting {block[0]!r} "
                     "not found in corpus")


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
            items.append({"task": "next", "kind": "next_line", "block": b,
                          "x": b[i], "n": 0, "reference": b[i + 1]})
        else:
            cut = max(1, len(b) // 2)
            items.append({"task": "next", "kind": "end_block", "block": b,
                          "x": "\n".join(b[:cut]), "n": 0,
                          "reference": "\n".join(b[cut:])})
    for k in range(n_per_task):  # prev
        b = rng.choice([b for b in blocks if len(b) >= 2])
        if k % 2 == 0:
            i = rng.randrange(1, len(b))
            items.append({"task": "prev", "kind": "prev_line", "block": b,
                          "x": b[i], "n": 0, "reference": b[i - 1]})
        else:
            cut = max(1, len(b) // 2)
            items.append({"task": "prev", "kind": "start_block", "block": b,
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
        items.append({"task": "cloze", "kind": "cloze", "n": n_del, "block": b,
                      "x": " ".join(masked), "reference": " ".join(deleted)})
    return items


_norm_re = re.compile(r"\s+")

# Word separators for the LCS: whitespace plus every readable punctuation
# mark, including the "/" and "," verse-joining conventions and the Spanish
# marks (mirrors masking._SPAN_PUNCT). Attached punctuation must never make
# "cabalga;" a different word than "cabalga".
_word_sep_re = re.compile(
    r"[\s" + re.escape(
        r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" + "¡¿«»“”‘’—–…·"
    ) + r"]+"
)


def _norm(s: str) -> str:
    return _norm_re.sub(" ", s.replace("«", "").replace("»", "")).strip()


def _words(s: str) -> list[str]:
    """Word stream for the LCS metric: separator- and punctuation-free."""
    return [w for w in _word_sep_re.split(s) if w]


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
    ref_w, ans_w = _words(reference), _words(answer)
    return {
        "exact": float(ref == ans),
        "word_acc": (_lcs_words(ref_w, ans_w) / len(ref_w)) if ref_w else 0.0,
    }


@torch.no_grad()
def _generate_answers_batched(model, tokenizer, prompts: list[str],
                              budgets: list[int], eos: int,
                              generation_batch: int) -> tuple[list[str], list[dict]]:
    """Left-padded batched greedy decode of ``prompts`` (original order kept).

    Same machinery as the retired recite engine's batched path (left pad +
    ``model.generate`` + OOM backoff halving), but a wired knob here, not a
    dead CLI flag.  Per-batch token budget is the max item budget in the
    batch; each decoded row is truncated back to its own budget when it did
    not stop at EOS.  Greedy batched decode therefore matches the B=1 budget
    contract up to bf16 kernel-shape rounding on argmax ties. The historical
    B1-vs-B8 spot-check was budget-confounded; see the correction in issues.md.
    """
    was_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from .recite import _is_cuda_oom

    answers: list[str] = [""] * len(prompts)
    completion: list[dict] = [{} for _ in prompts]
    try:
        start = 0
        cur = generation_batch
        while start < len(prompts):
            chunk = slice(start, start + cur)
            enc = tokenizer(prompts[chunk], return_tensors="pt", padding=True,
                            add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            try:
                out = model.generate(
                    **enc,
                    max_new_tokens=max(budgets[chunk]),
                    do_sample=False,
                    eos_token_id=eos,
                    pad_token_id=tokenizer.pad_token_id,
                )
            except RuntimeError as e:
                if not _is_cuda_oom(e) or cur <= 1:
                    raise
                cur = max(1, cur // 2)
                torch.cuda.empty_cache()
                continue
            gen_start = enc["input_ids"].shape[1]
            for row in range(out.shape[0]):
                # model.generate has one max_new_tokens value for the whole
                # batch.  Without this per-row bound, a short-reference item
                # inherited the longest peer's budget (up to +207 tokens in
                # the current battery), changing the evaluation rather than
                # merely batching it.  Later tokens cannot affect earlier
                # greedy tokens, so truncation restores the B=1 contract.
                budget = budgets[start + row]
                generated = out[row, gen_start:gen_start + budget].tolist()
                stopped = eos in generated
                answers[start + row] = tokenizer.decode(
                    generated, skip_special_tokens=True)
                completion[start + row] = {
                    "generated_tokens": len(generated),
                    "budget_tokens": budget,
                    "stopped": stopped,
                    "hard_cut": len(generated) >= budget and not stopped,
                }
            start += out.shape[0]
    finally:
        tokenizer.padding_side = was_padding
    return answers, completion


@torch.no_grad()
def tasks_eval(model, tokenizer, poem_path: str, seed: int = 17,
               n_per_task: int = 24, max_extra_tokens: int = 32,
               keep_examples: int = 6, with_context: bool | str = False,
               context_window_lines: int = 4,
               context_pad_random: bool = False,
               generation_batch: int = 1) -> dict:
    """Run the three-task battery; returns plain per-task accuracies.

    ``with_context``: teacher/RAG ceiling mode (owner directive 2026-07-11).
    The STUDENT battery never sets this — living without the passage in
    context is the entire point of the student; this flag exists only for
    the separate teacher-ceiling reference test, which measures the SAME
    three tasks but with a retrieved document prepended, exactly like
    training's RAG mode
    (``masking.render_rag``: "\\n\\nDocumento recuperado:\\n{passage}").
    True/"full" = the whole corpus file (historical ceiling); "window" =
    per-item exact-match retrieval of the item's source block ±
    ``context_window_lines`` (:func:`retrieve_window`) — the ceiling paired
    with window-scope v5 training, and the honest control demanded by the
    2026-07-12 full-document-copying failure. Same questions, references and
    scoring as the no-context battery, so ceiling scores are directly
    comparable to a checkpoint's plain ``tasks_eval`` score.

    ``context_pad_random``: replace the (scope-sized) retrieved document by
    a seeded random distinct-token fill of the same tokenized length — the
    epoch-0 teacher FLOOR paired with pad_random-censored training arms
    (the ``remove`` floor is simply ``with_context=False``). Approximate to
    one decode/re-encode round trip; the training-side fill is exact at the
    id level.

    ``generation_batch``: 1 (default) keeps the historical per-item greedy
    loop bit-for-bit; >1 decodes in left-padded batches — measured 2026-07-11
    because per-epoch B=1 eval was 42-56%% of loss-grid arm wall time."""
    from ..masking import random_fill_ids
    from .recite import greedy_generate_positions, strip_think

    # This public evaluator is called directly by scripts as well as between
    # epochs by the trainer.  Dropout must never contaminate an evaluation,
    # but a training caller must resume training mode afterwards.
    was_training = model.training
    model.eval()
    try:
        items = build_tasks(poem_path, seed=seed, n_per_task=n_per_task)
        scope = {False: None, True: "full"}.get(with_context, with_context)
        if scope not in (None, "full", "window", "chapter"):
            raise ValueError(f"unknown with_context scope {with_context!r}")
        contexts = None
        if scope == "full":
            with open(poem_path, encoding="utf-8") as f:
                contexts = [f.read()] * len(items)
        elif scope == "window":
            lines = [l for b in corpus_blocks(poem_path) for l in b]
            contexts = [retrieve_window(lines, it["block"],
                                        pad=context_window_lines)
                        for it in items]
        elif scope == "chapter":
            contexts = [retrieve_chapter(poem_path, it["block"])
                        for it in items]
        if context_pad_random:
            if contexts is None:
                raise ValueError(
                    "context_pad_random needs with_context to size the fill "
                    "(the paired floor matches its ceiling's scope)")
            contexts = [
                tokenizer.decode(random_fill_ids(
                    tokenizer, f"evalfloor-{seed}-{i}",
                    len(tokenizer.encode(c, add_special_tokens=False))))
                for i, c in enumerate(contexts)
            ]
        # ``convert_tokens_to_ids('<|im_end|>')`` returns the unknown token id
        # on SentencePiece models such as Mistral.  chatfmt knows whether a
        # model actually has a single-token turn closer and otherwise returns
        # its real EOS id.
        eos = stop_token_id(tokenizer)
        device = next(model.parameters()).device
        questions, prompts, budgets = [], [], []
        for i, it in enumerate(items):
            q = QUESTIONS[it["kind"]].format(x=it["x"], n=it["n"])
            content = (f"{q}\n\nDocumento recuperado:\n{contexts[i]}"
                      if contexts is not None else q)
            questions.append(q)
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
            budgets.append(len(tokenizer.encode(it["reference"]))
                           + max_extra_tokens)
        if generation_batch > 1:
            answers, completion = _generate_answers_batched(
                model, tokenizer, prompts, budgets, eos, generation_batch)
        else:
            answers, completion = [], []
            for prompt, budget in zip(prompts, budgets):
                ids = torch.tensor(
                    [tokenizer.encode(prompt, add_special_tokens=False)],
                    device=device)
                out = greedy_generate_positions(
                    model, ids,
                    torch.arange(ids.shape[1], device=device)[None],
                    max_new_tokens=budget, eos_id=eos)
                stopped = eos in out
                answers.append(tokenizer.decode(out, skip_special_tokens=True))
                completion.append({
                    "generated_tokens": len(out),
                    "budget_tokens": budget,
                    "stopped": stopped,
                    "hard_cut": len(out) >= budget and not stopped,
                })
        agg: dict[str, list[dict]] = {}
        examples = []
        for it, q, raw, meta in zip(items, questions, answers, completion):
            answer = strip_think(raw)
            s = score(it["reference"], answer)
            s["n_deleted"] = it["n"]
            agg.setdefault(it["task"], []).append(s)
            if len(examples) < keep_examples:
                examples.append({"kind": it["kind"], "q": q,
                                 "reference": it["reference"],
                                 "answer": answer.strip()[:200], **meta, **s})
        result = {"seed": seed, "n_per_task": n_per_task,
                  "generation_batch": generation_batch,
                  "with_context": scope or False,
                  "context_pad_random": context_pad_random,
                  **({"context_window_lines": context_window_lines}
                     if scope == "window" else {}),
                  "tasks": {}}
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
        result["generation"] = {
            "n": len(completion),
            "mean_generated_tokens": (sum(x["generated_tokens"] for x in completion)
                                      / max(1, len(completion))),
            "mean_budget_tokens": (sum(x["budget_tokens"] for x in completion)
                                   / max(1, len(completion))),
            "stopped_fraction": (sum(x["stopped"] for x in completion)
                                 / max(1, len(completion))),
            "hard_cut_fraction": (sum(x["hard_cut"] for x in completion)
                                  / max(1, len(completion))),
        }
        result["examples"] = examples
        return result
    finally:
        if was_training:
            model.train()
