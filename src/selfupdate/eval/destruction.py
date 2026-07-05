"""Destruction metrology v2 — is a memorization arm "destructive"?

Four instruments, one JSON (schema v1), four pre-committed thresholds:

1. probe_battery    — per-category CE on eval/probes.py (5 × ~8 texts).
                      Threshold: any category mean ΔCE > 0.5 nats vs base.
2. benchmark_ce_ranking — HellaSwag + MMLU-Pro (nomath-sml) accuracy by
                      length-normalized answer-CE argmin, n=200, seed=17.
                      Threshold: accuracy drop > 5 points absolute.
3. intrusion_generation — greedy raw-LM continuations of 40 neighbor-genre
                      prompts; hit = a training-corpus 5-gram appears.
                      Threshold: hit rate > 10%.
4. degeneration_stats — on the same generations: max repeated-4-gram run
                      and distinct-2 ratio. Threshold: mean max-rep4 run
                      doubles vs base.

CE-ranking rationale: at 0.6B, generative benchmark harnesses mostly
measure formatting compliance; ranking fixed continuations by CE isolates
the knowledge/plausibility signal and is deterministic — right instrument
for BEFORE/AFTER deltas on the same items.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

from .probes import PROBE_SETS
from .recite import strip_think

SCHEMA_VERSION = 1
BENCH_N = 200
BENCH_SEED = 17

_WORD_RE = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    """Normalized word-level n-grams (case-folded, punctuation-blind)."""
    ws = _words(text)
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


@torch.no_grad()
def text_ce(model, tokenizer, text: str, device: str = "cuda") -> float:
    ids = tokenizer.encode(text, add_special_tokens=False)
    t = torch.tensor([ids], device=device)
    logits = model(t, use_cache=False).logits[0].float()
    return F.cross_entropy(logits[:-1], t[0, 1:]).item()


@torch.no_grad()
def answer_span_ce(model, tokenizer, prompt: str, answer: str,
                   device: str = "cuda") -> float:
    """Mean per-token CE of ``answer`` given ``prompt`` (length-normalized,
    so short options are not favored)."""
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    t = torch.tensor([p_ids + a_ids], device=device)
    logits = model(t, use_cache=False).logits[0].float()
    # logits at position i predict token i+1; answer starts at len(p_ids)
    span = logits[len(p_ids) - 1: len(p_ids) + len(a_ids) - 1]
    ref = t[0, len(p_ids): len(p_ids) + len(a_ids)]
    return F.cross_entropy(span, ref).item()


@torch.no_grad()
def probe_battery(model, tokenizer, device: str = "cuda") -> dict:
    was_training = model.training
    model.eval()
    out = {}
    all_ces = []
    for cat, texts in PROBE_SETS.items():
        ces = [text_ce(model, tokenizer, t, device) for t in texts]
        all_ces += ces
        m = sum(ces) / len(ces)
        var = sum((c - m) ** 2 for c in ces) / max(len(ces) - 1, 1)
        out[cat] = {"n": len(ces), "mean_ce": m,
                    "stderr": math.sqrt(var / len(ces)), "per_text": ces}
    # legacy 4 = first members of their categories (see probes.py)
    legacy = [out["poetry_es"]["per_text"][0], out["facts"]["per_text"][0],
              out["prose_en"]["per_text"][0], out["procedural"]["per_text"][0]]
    if was_training:
        model.train()
    return {"categories": out, "overall_mean_ce": sum(all_ces) / len(all_ces),
            "legacy_mean_ce": sum(legacy) / 4}


def _bench_sample(ds, n: int, seed: int):
    import random

    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


# ---- standard-suite formatters: row -> (prompt|None, options, answer_idx).
# prompt=None means unconditional scoring (mean per-token CE of each option
# text — the harness method for cloze tasks like WinoGrande).

def fmt_hellaswag(row):
    return row["ctx"], [" " + e for e in row["endings"]], int(row["label"])


def fmt_mmlu(row):
    return (f"Question: {row['question']}\nAnswer:",
            [" " + c for c in row["choices"]], row["answer"])


def fmt_mmlu_pro(row):
    return (f"Question: {row['question']}\nAnswer:",
            [" " + o for o in row["options"]], row["answer_index"])


def fmt_arc(row):
    labels = list(row["choices"]["label"])
    return (f"Question: {row['question']}\nAnswer:",
            [" " + t for t in row["choices"]["text"]],
            labels.index(row["answerKey"]))


def fmt_winogrande(row):
    return (None,
            [row["sentence"].replace("_", row[f"option{i}"]) for i in (1, 2)],
            int(row["answer"]) - 1)


def make_fmt_gpqa(seed):
    import random

    def fmt(row):
        opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(f"{seed}:{row['Question'][:40]}").shuffle(order)
        return (f"Question: {row['Question']}\nAnswer:",
                [" " + opts[i] for i in order], order.index(0))
    return fmt


# name -> (dataset args, split, formatter factory). The standard quartet is
# the classic model-card set (comparability); gpqa_diamond is chance-level
# below ~7B but pins the bigger checkpoints; mmlu_pro_nomath kept for
# continuity with the first C2 batteries.
BENCH_REGISTRY = {
    "hellaswag": (("Rowan/hellaswag",), "validation", lambda seed: fmt_hellaswag),
    "mmlu": (("cais/mmlu", "all"), "test", lambda seed: fmt_mmlu),
    "arc_challenge": (("allenai/ai2_arc", "ARC-Challenge"), "test", lambda seed: fmt_arc),
    "winogrande": (("allenai/winogrande", "winogrande_xl"), "validation",
                   lambda seed: fmt_winogrande),
    "gpqa_diamond": (("Idavidrein/gpqa", "gpqa_diamond"), "train", make_fmt_gpqa),
    "mmlu_pro_nomath": (("sam-paech/mmlu-pro-nomath-sml",), "test",
                        lambda seed: fmt_mmlu_pro),
}
DEFAULT_BENCHES = ("hellaswag", "mmlu", "arc_challenge", "winogrande",
                   "mmlu_pro_nomath")


@torch.no_grad()
def benchmark_ce_ranking(model, tokenizer, device: str = "cuda",
                         n: int = BENCH_N, seed: int = BENCH_SEED,
                         benches: tuple = DEFAULT_BENCHES) -> dict:
    """Accuracy by length-normalized answer-CE argmin on FIXED seeded
    subsets — the lm-eval-harness loglikelihood method in miniature. Same
    items across checkpoints, so deltas are paired (tight at n=200 even
    though absolute stderr is ~3.5 pts). Needs the HF datasets cache; call
    sites set HF_HUB_OFFLINE=1 so a cold cache fails loudly."""
    from datasets import load_dataset

    was_training = model.training
    model.eval()
    results = {}
    for name in benches:
        args, split, fmt_factory = BENCH_REGISTRY[name]
        fmt = fmt_factory(seed)
        rows = _bench_sample(load_dataset(*args, split=split), n, seed)
        correct = 0
        for row in rows:
            prompt, options, ans = fmt(row)
            if prompt is None:
                ces = [text_ce(model, tokenizer, o, device) for o in options]
            else:
                ces = [answer_span_ce(model, tokenizer, prompt, o, device)
                       for o in options]
            correct += int(min(range(len(ces)), key=ces.__getitem__) == ans)
        results[name] = {"n": len(rows), "accuracy": correct / len(rows)}
    if was_training:
        model.train()
    return results


@torch.no_grad()
def intrusion_generation(model, tokenizer, prompts: list[str],
                         corpus_lines: list[str], device: str = "cuda",
                         max_new_tokens: int = 64) -> dict:
    """Greedy raw-LM continuation (no chat template — the intrusion failure
    mode is memorized text surfacing in PLAIN continuation, where no
    elicitation frame invited it). Hit = any training-corpus 5-gram in the
    generation."""
    was_training = model.training
    model.eval()
    corpus_grams = set()
    for line in corpus_lines:
        corpus_grams |= ngrams(line)
    hits, gens = [], []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        t = torch.tensor([ids], device=device)
        out = model.generate(t, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = strip_think(tokenizer.decode(out[0, len(ids):],
                                            skip_special_tokens=True))
        gens.append(text)
        got = ngrams(text) & corpus_grams
        if got:
            hits.append({"prompt": prompt, "generation": text,
                         "matched": [" ".join(g) for g in sorted(got)][:5]})
    if was_training:
        model.train()
    return {"n": len(prompts), "hit_rate": len(hits) / max(len(prompts), 1),
            "hits": hits, "generations": gens}


def degeneration_stats(texts: list[str]) -> dict:
    """Repetition collapse counters on generations.

    max_rep4_run: for the worst 4-gram of a text, how many times it repeats
    in immediate succession (1 = no immediate repetition).
    distinct2: unique bigrams / total bigrams (low = loops)."""
    runs, d2 = [], []
    for text in texts:
        ws = _words(text)
        best = 1
        for i in range(len(ws) - 3):
            block = ws[i:i + 4]
            r = 1
            while ws[i + 4 * r: i + 4 * (r + 1)] == block:
                r += 1
            best = max(best, r)
        runs.append(best)
        bigrams = [tuple(ws[i:i + 2]) for i in range(len(ws) - 1)]
        d2.append(len(set(bigrams)) / max(len(bigrams), 1))
    n = max(len(texts), 1)
    return {"max_rep4_run_mean": sum(runs) / n if runs else 1.0,
            "max_rep4_run_max": max(runs, default=1),
            "distinct2_mean": sum(d2) / n if d2 else 1.0}


DESTRUCTION_THRESHOLDS = {
    "probe_category_dce": 0.5,     # nats, any category vs base
    "benchmark_drop_pts": 5.0,     # absolute accuracy points, any benchmark
    "intrusion_rate": 0.10,
    "rep4_ratio": 2.0,             # mean max-rep4 run vs base
}


def verdict(dest: dict, base: dict) -> dict:
    """Apply the pre-committed destruction thresholds to a (run, base) pair
    of destruction.json dicts. Returns per-criterion flags + overall."""
    flags = {}
    cats = dest["probe_battery"]["categories"]
    base_cats = base["probe_battery"]["categories"]
    worst = max((cats[c]["mean_ce"] - base_cats[c]["mean_ce"], c) for c in cats)
    flags["probe_category"] = {
        "worst_category": worst[1], "worst_dce": worst[0],
        "tripped": worst[0] > DESTRUCTION_THRESHOLDS["probe_category_dce"]}
    if "benchmarks" in dest and "benchmarks" in base:
        # intersect: suites evolve (std quartet added 2026-07-04); a run
        # judged against an older base ref uses the shared benchmarks only
        shared = set(dest["benchmarks"]) & set(base["benchmarks"])
        drops = {b: 100 * (base["benchmarks"][b]["accuracy"]
                           - dest["benchmarks"][b]["accuracy"])
                 for b in shared}
        wb = max(drops, key=drops.get)
        flags["benchmark"] = {
            "worst_benchmark": wb, "worst_drop_pts": drops[wb],
            "tripped": drops[wb] > DESTRUCTION_THRESHOLDS["benchmark_drop_pts"]}
    flags["intrusion"] = {
        "rate": dest["intrusion"]["hit_rate"],
        "tripped": dest["intrusion"]["hit_rate"]
        > DESTRUCTION_THRESHOLDS["intrusion_rate"]}
    ratio = (dest["degeneration"]["max_rep4_run_mean"]
             / max(base["degeneration"]["max_rep4_run_mean"], 1e-9))
    flags["degeneration"] = {
        "rep4_ratio": ratio,
        "tripped": ratio > DESTRUCTION_THRESHOLDS["rep4_ratio"]}
    flags["destructive"] = any(v["tripped"] for v in flags.values()
                               if isinstance(v, dict))
    return flags
