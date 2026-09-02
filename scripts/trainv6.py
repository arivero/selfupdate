#!/usr/bin/env python
"""trainv6: experimental causal layerwise self-distillation monolith.

This file is intentionally self-contained.  It imports neither selfupdate nor
trainv5, and it owns its data gates, architecture interventions, training
loop, evaluation, telemetry, and runtime certification.

V6 has two explicitly named training laws.  Both feed a detached current
student state to each trainable block, keep gradients inside that block, use
teacher-sourced targets only, and keep embeddings, final norm, and LM head
frozen.

causal_residual
    On the same detached block input, compare a frozen passage-visible block
    with a frozen passage-blocked block.  The adapter-on blocked block is
    trained toward the visible output, so the requested change is precisely
    the block-local causal contribution of passage access.

partial_teacher
    Build the frozen teacher trajectory with passage access enabled only at a
    pre-certified model-specific retrieval set.  Train the fully blocked
    student trajectory toward those teacher states.

Training uses a full-slot censorship view: passage tokens retain their slots
and positions, but post-passage softmax queries cannot attend to them.  In
Qwen3.6 GatedDeltaNet layers, passage rows are zeroed only at the recurrent
token mixer, preventing their content from updating recurrent memory while
retaining the sequence clock and residual-stream encoding.  Deployment
evaluation always uses the ordinary passage-removed natural-position prompt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRIC_SCHEMA = "v6_metrics_v1"

PRESETS = {
    "gemma4_31b": {
        "model": "google/gemma-4-31B-it",
        "responses": "runs/vllm_h100/gemma4_31b_it/responses_bs256.jsonl",
        "examples": "data/combined/examples_v5rs_window.jsonl",
        "layers": 60,
        "layer_types": None,
        "retrieval_layers": [5, 17, 23, 29, 35, 41, 47, 53],
        "retrieval_evidence": "runs/v5_teacher_attn/layer_censor_v3.json",
        "examples_sha256": "575b9dea35e0179dcdf7a513416e640db899c9bf9584236088f2921cce7a7042",
        "responses_sha256": "9449db12cbde264ebcd1bc71b169294202efcab180f2b94c2b5c0251ecceafd8",
        "evidence_sha256": "9e104757c1adc90aebe12fa71bf53141df355aa2302b7abaef8b6e077637048e",
        "summary_sha256": "8e57ee12e367ad7741bc717cdb1ae8e812fdf5e378d29dcdcabc099c30c1a385",
    },
    "qwen36_27b": {
        "model": "Qwen/Qwen3.6-27B",
        "responses": (
            "runs/vllm_h100/qwen36_27b_full_exactids/responses_bs256.jsonl"
        ),
        "examples": "data/combined/examples_v5rs_window.jsonl",
        "layers": 64,
        "layer_types": (
            ["linear_attention", "linear_attention", "linear_attention",
             "full_attention"] * 16
        ),
        # The existing v3 only-S probe left all recurrent layers visible and
        # retained these eight softmax layers.  Preserve that exact semantics.
        "retrieval_layers": (
            [i for i in range(64) if i % 4 != 3]
            + [11, 15, 19, 23, 47, 51, 55, 59]
        ),
        "retrieval_evidence": (
            "runs/v5_teacher_attn/layer_censor_qwen36_v3.json"
        ),
        "examples_sha256": "575b9dea35e0179dcdf7a513416e640db899c9bf9584236088f2921cce7a7042",
        "responses_sha256": "ed556e04a84cd9e07fc6411a9ae518cd8bd2d4478160d81ceb1098436609c5c8",
        "evidence_sha256": "b46b8339f08116b9290c3fc973f2072c3734f36e6b7ef9d3932cc7f5c9418fba",
        "summary_sha256": "984cc07f29136f81840eb46b485c8230b5c2234869e923975079ad8e135e74d2",
    },
}

CORPUS_PATHS = {
    "mach": "data/poem/raw.txt",
    "quij": "data/quijote/raw_ch4.txt",
}

EVAL_ONLY = {"evaluation_only": True, "used_for_backward": False, "optimizer_weight": 0.0}

PROFILE_SPECS = {
    "hidden_huber": ("hidden_huber_profile", "answer_tokens", "answer_token_weighted_mean"),
    "lens_js": ("lens_js_profile", "answer_tokens", "answer_token_weighted_mean"),
    "anchor": ("anchor_profile", "anchor_rows", "anchor_row_weighted_mean"),
    "causal_effect_rms": ("causal_effect_rms_profile", "items", "item_weighted_mean"),
}


# ---------------------------------------------------------------------------
# Small pure-Python primitives: data identity and metrics
# ---------------------------------------------------------------------------

class LayerMetricLedger:
    """GPU-resident per-layer numerators with explicit epoch denominators."""

    def __init__(self, layer_count: int):
        self.sums = {name: [None] * layer_count for name in PROFILE_SPECS}
        self.counts = {denominator: 0 for _, denominator, _ in PROFILE_SPECS.values()}

    def count(self, answer_tokens: int, anchor_rows: int, items: int):
        self.counts["answer_tokens"] += answer_tokens
        self.counts["anchor_rows"] += anchor_rows
        self.counts["items"] += items

    def add(self, index: int, **numerators):
        for name, value in numerators.items():
            previous = self.sums[name][index]
            self.sums[name][index] = value if previous is None else previous + value

    def finish(self) -> tuple[dict, dict, dict]:
        profiles = {}
        for name, (field, denominator, _aggregation) in PROFILE_SPECS.items():
            values = self.sums[name]
            profiles[field] = None if values[0] is None else [
                float(value) / max(1, self.counts[denominator]) for value in values
            ]
        return (
            profiles,
            {name: spec[2] for name, spec in PROFILE_SPECS.items()},
            dict(self.counts),
        )


def objective_gradient_attribution(objectives: dict, parameters, device) -> dict:
    """Return additive statistics for objective attribution."""
    import torch

    grads = {
        name: (torch.autograd.grad(value, parameters, retain_graph=True,
                                   allow_unused=True)
               if value is not None else tuple(None for _ in parameters))
        for name, value in objectives.items()
    }
    zero = torch.zeros((), device=device)
    squared = {
        name: sum((grad.float().pow(2).sum() for grad in values
                   if grad is not None), zero)
        for name, values in grads.items()
    }
    result = {
        f"{name}_norm_sq": value.detach()
        for name, value in squared.items()
    }
    for left, right in (("hidden", "lens_js"), ("hidden", "anchor"),
                        ("lens_js", "anchor")):
        dot = sum((a.float().mul(b.float()).sum()
                   for a, b in zip(grads[left], grads[right])
                   if a is not None and b is not None), zero)
        result[f"{left}_{right}_dot"] = dot.detach()
    return result


def finish_gradient_attribution(sums: dict, count: int) -> dict:
    """Derive RMS norms, shares, and concatenated-gradient cosines."""
    names = ("hidden", "lens_js", "anchor")
    pairs = (("hidden", "lens_js"), ("hidden", "anchor"),
             ("lens_js", "anchor"))
    mean_sq = {
        name: max(0.0, float(sums[f"{name}_norm_sq"] / count))
        for name in names
    }
    norms = {name: math.sqrt(value) for name, value in mean_sq.items()}
    total = sum(norms.values())
    defined = total > 0.0
    result = {
        **{f"{name}_norm": norms[name] for name in names},
        **{f"{name}_share": norms[name] / total if defined else None
           for name in names},
        "total_norm": total,
        "share_defined": defined,
    }
    for left, right in pairs:
        denominator = norms[left] * norms[right]
        mean_dot = float(sums[f"{left}_{right}_dot"] / count)
        result[f"{left}_{right}_cosine"] = (
            mean_dot / denominator if denominator > 0.0 else None
        )
    return result


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def corpus_lines(path: Path) -> list[str]:
    """Match the repository corpus convention: headings/blanks are metadata."""
    return [
        raw.strip()
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip() and not raw.lstrip().startswith("#")
    ]


def words(text: str) -> list[str]:
    out, current = [], []
    for char in unicodedata.normalize("NFC", text).casefold():
        if char.isspace() or unicodedata.category(char).startswith("P"):
            if current:
                out.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        out.append("".join(current))
    return out


def lcs_length(left: list[str], right: list[str]) -> int:
    row = [0] * (len(right) + 1)
    for x in left:
        previous = 0
        for j, y in enumerate(right, 1):
            old = row[j]
            row[j] = previous + 1 if x == y else max(row[j], row[j - 1])
            previous = old
    return row[-1]


def lcs_score(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    return lcs_length(ref, words(hypothesis)) / len(ref) if ref else 0.0


def longest_prefix(reference: str, hypothesis: str) -> tuple[int, int | None]:
    """Longest canonical prefix found contiguously anywhere in the output."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0, None
    best, best_start = 0, None
    for start, token in enumerate(hyp):
        if token != ref[0]:
            continue
        length = 0
        while (length < len(ref) and start + length < len(hyp)
               and hyp[start + length] == ref[length]):
            length += 1
        if length > best:
            best, best_start = length, start
    return best, best_start


def canonical_target(example: dict, lines_by_corpus: dict[str, list[str]]) -> str:
    corpus = example["corpus"]
    if corpus not in lines_by_corpus:
        raise ValueError(f"unknown corpus {corpus!r}")
    lo, hi = example["target_lines"]
    source = lines_by_corpus[corpus]
    if not (0 <= lo < hi <= len(source)):
        raise ValueError(
            f"{example['example_id']}: bad target span [{lo}, {hi})"
        )
    span = source[lo:hi]
    kind = example["kind"]
    if kind in ("next", "prev"):
        return "\n".join(span)
    if kind != "cloze":
        raise ValueError(f"{example['example_id']}: unknown kind {kind!r}")
    canonical_tokens = "\n".join(span).split()
    question_tokens = example["question"].split()
    candidates = []
    for start in range(0, len(question_tokens) - len(canonical_tokens) + 1):
        fragment = question_tokens[start:start + len(canonical_tokens)]
        if "___" not in fragment:
            continue
        if all(observed == "___" or words(observed) == words(expected)
               for observed, expected in zip(fragment, canonical_tokens)):
            candidates.append(fragment)
    if len(candidates) != 1:
        raise ValueError(
            f"{example['example_id']}: canonical cloze match count "
            f"{len(candidates)} != 1"
        )
    fragment = candidates[0]
    deleted = []
    for observed, expected in zip(fragment, canonical_tokens):
        if observed == "___":
            deleted.append(expected)
        elif words(observed) != words(expected):
            raise ValueError(
                f"{example['example_id']}: cloze nonblank mismatch "
                f"{observed!r} != {expected!r}"
            )
    if not deleted:
        raise ValueError(f"{example['example_id']}: cloze has no deletion")
    return " ".join(deleted)


def paired_bootstrap(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    seed: int,
    draws: int = 4000,
) -> dict:
    ids = sorted(set(current) & set(baseline))
    if not ids:
        return {"n": 0, "delta": None, "ci95": [None, None]}
    diffs = [current[item] - baseline[item] for item in ids]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(
            sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
        )
    samples.sort()
    lo = samples[int(0.025 * (draws - 1))]
    hi = samples[int(0.975 * (draws - 1))]
    return {
        "n": len(ids),
        "delta": round(mean, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "draws": draws,
        "seed": seed,
    }


def score_generation(item: dict, text: str, token_count: int,
                     finish_reason: str) -> dict:
    canonical = item["canonical_text"]
    prefix, start = longest_prefix(canonical, text)
    canonical_words = words(canonical)
    output_words = words(text)
    divergence_start = start if start is not None else 0
    first_div_ref = canonical_words[prefix] if prefix < len(canonical_words) else None
    first_div_out = (
        output_words[divergence_start + prefix]
        if divergence_start + prefix < len(output_words)
        else None
    )
    content = lcs_score(canonical, text)
    teacher = lcs_score(item["answer_text"], text)
    return {
        "example_id": item["example_id"],
        "corpus": item["corpus"],
        "kind": item["kind"],
        "canonical_reference": canonical,
        "teacher_answer": item["answer_text"],
        "generated_text": text,
        "finish_reason": finish_reason,
        "reference_word_count": len(canonical_words),
        "reference_token_count": len(item["canonical_ids"]),
        "output_word_count": len(output_words),
        "output_token_count": token_count,
        "content_lcs": round(content, 6),
        "teacher_answer_lcs": round(teacher, 6),
        "content_exact": output_words == canonical_words,
        "recitation": content >= 0.9,
        "first_content_word_correct": bool(
            canonical_words and output_words
            and output_words[0] == canonical_words[0]
        ),
        "canonical_first_word_present": bool(
            canonical_words and canonical_words[0] in output_words
        ),
        "longest_correct_prefix_words": prefix,
        "longest_correct_prefix_fraction": round(
            prefix / max(1, len(canonical_words)), 6
        ),
        "first_divergence_reference_index": (
            prefix if prefix < len(canonical_words) else None
        ),
        "first_divergence_output_index": (
            divergence_start + prefix
            if divergence_start + prefix < len(output_words) else None
        ),
        "first_divergence_reference": first_div_ref,
        "first_divergence_output": first_div_out,
    }


def summarize_generation(rows: list[dict]) -> dict:
    def one(group: list[dict]) -> dict:
        return {
            "items": len(group),
            "content_lcs": round(
                sum(row["content_lcs"] for row in group) / max(1, len(group)),
                6,
            ),
            "teacher_answer_lcs": round(
                sum(row["teacher_answer_lcs"] for row in group)
                / max(1, len(group)),
                6,
            ),
            "recitation_rate": round(
                sum(row["recitation"] for row in group) / max(1, len(group)),
                6,
            ),
            "content_exact_rate": round(
                sum(row["content_exact"] for row in group)
                / max(1, len(group)),
                6,
            ),
            "first_content_word_rate": round(
                sum(row["first_content_word_correct"] for row in group)
                / max(1, len(group)),
                6,
            ),
            "prefix_fraction": round(
                sum(row["longest_correct_prefix_fraction"] for row in group)
                / max(1, len(group)),
                6,
            ),
        }

    by_corpus, by_kind = {}, {}
    for row in rows:
        by_corpus.setdefault(row["corpus"], []).append(row)
        by_kind.setdefault(row["kind"], []).append(row)
    return {
        "overall": one(rows),
        "by_corpus": {name: one(group) for name, group in sorted(by_corpus.items())},
        "by_kind": {name: one(group) for name, group in sorted(by_kind.items())},
    }


def grouped_bootstrap(rows: list[dict], baseline: dict[str, dict],
                      seed: int) -> dict:
    result = {}
    for field in ("corpus", "kind"):
        groups = sorted({row[field] for row in rows})
        result[f"by_{field}"] = {}
        for offset, group in enumerate(groups):
            current = {
                row["example_id"]: row["content_lcs"]
                for row in rows if row[field] == group
            }
            base = {
                example_id: baseline[example_id]["content_lcs"]
                for example_id in current if example_id in baseline
            }
            result[f"by_{field}"][group] = paired_bootstrap(
                current, base, seed=seed + offset + (100 if field == "kind" else 0)
            )
    return result


def self_test_metrics() -> None:
    assert words("¡Árbol, árbol!") == ["árbol", "árbol"]
    assert lcs_score("uno dos tres", "prefacio uno dos fin") == 2 / 3
    assert longest_prefix("uno dos tres", "sí: uno dos no") == (2, 1)
    lines = {"mach": ["uno dos", "tres cuatro cinco"]}
    nxt = {
        "example_id": "n", "corpus": "mach", "kind": "next",
        "target_lines": [0, 1], "question": "",
    }
    cloze = {
        "example_id": "c", "corpus": "mach", "kind": "cloze",
        "target_lines": [1, 2], "question": "Completa «tres ___ cinco».",
    }
    assert canonical_target(nxt, lines) == "uno dos"
    assert canonical_target(cloze, lines) == "cuatro"
    base = {"a": 0.1, "b": 0.2, "c": 0.3}
    cur = {"a": 0.2, "b": 0.3, "c": 0.4}
    first = paired_bootstrap(cur, base, seed=7, draws=200)
    second = paired_bootstrap(cur, base, seed=7, draws=200)
    assert first == second and abs(first["delta"] - 0.1) < 1e-6
    scored = score_generation(
        {
            "example_id": "x", "corpus": "mach", "kind": "next",
            "canonical_text": "uno dos", "canonical_ids": [1, 2],
            "answer_text": "Aquí: uno dos",
        },
        "Respuesta: uno dos", 4, "stop",
    )
    assert scored["content_lcs"] == 1.0
    assert scored["longest_correct_prefix_words"] == 2
    assert not scored["first_content_word_correct"]
    assert scored["canonical_first_word_present"]
    direct = score_generation(
        {
            "example_id": "y", "corpus": "mach", "kind": "next",
            "canonical_text": "uno dos", "canonical_ids": [1, 2],
            "answer_text": "uno dos",
        },
        "uno dos", 2, "stop",
    )
    assert direct["first_content_word_correct"]
    zero = finish_gradient_attribution({
        "hidden_norm_sq": 0.0, "lens_js_norm_sq": 0.0,
        "anchor_norm_sq": 0.0, "hidden_lens_js_dot": 0.0,
        "hidden_anchor_dot": 0.0, "lens_js_anchor_dot": 0.0,
    }, 1)
    assert not zero["share_defined"] and zero["total_norm"] == 0.0
    assert all(zero[f"{name}_share"] is None
               for name in ("hidden", "lens_js", "anchor"))
    signal = finish_gradient_attribution({
        "hidden_norm_sq": 4.0, "lens_js_norm_sq": 1.0,
        "anchor_norm_sq": 0.0, "hidden_lens_js_dot": 1.0,
        "hidden_anchor_dot": 0.0, "lens_js_anchor_dot": 0.0,
    }, 1)
    assert signal["share_defined"]
    assert abs(signal["hidden_share"] - 2 / 3) < 1e-12
    assert abs(signal["hidden_lens_js_cosine"] - 0.5) < 1e-12
    print("v6 metric self-test: PASS")


def build_items(examples_path: Path, responses_path: Path, tokenizer,
                limit: int = 0) -> tuple[list[dict], int]:
    examples = {row["example_id"]: row for row in load_jsonl(examples_path)}
    responses = load_jsonl(responses_path)
    stop_ids = {row.get("stop_token_id") for row in responses}
    if len(stop_ids) != 1 or None in stop_ids:
        raise SystemExit(f"GATE: inconsistent/missing stop ids: {stop_ids}")
    stop_id = next(iter(stop_ids))
    lines_by_corpus = {
        name: corpus_lines(ROOT / relative)
        for name, relative in CORPUS_PATHS.items()
    }
    selected = responses
    if limit:
        selected = responses[::max(1, len(responses) // limit)][:limit]
    items = []
    seen = set()
    for response in selected:
        example_id = response["example_id"]
        if example_id in seen:
            raise SystemExit(f"GATE: duplicate response id {example_id}")
        seen.add(example_id)
        example = examples.get(example_id)
        if example is None:
            raise SystemExit(f"GATE: response {example_id} has no example")
        prompt_ids = response["prompt_token_ids"]
        decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if tokenizer.encode(decoded, add_special_tokens=False) != prompt_ids:
            raise SystemExit(f"GATE: tokenizer round trip failed for {example_id}")
        privileged = example["privileged"]
        if privileged in decoded:
            censored = decoded.replace(privileged, "", 1)
        elif privileged.strip() and privileged.strip() in decoded:
            censored = decoded.replace(privileged.strip(), "", 1)
        else:
            raise SystemExit(f"GATE: privileged passage not found for {example_id}")
        censored_ids = tokenizer.encode(censored, add_special_tokens=False)
        cut = 0
        for left, right in zip(prompt_ids, censored_ids):
            if left != right:
                break
            cut += 1
        gap = len(prompt_ids) - len(censored_ids)
        if gap <= 0:
            raise SystemExit(f"GATE: non-positive passage span for {example_id}")
        canonical = canonical_target(example, lines_by_corpus)
        canonical_ids = tokenizer.encode(canonical, add_special_tokens=False)
        if not canonical_ids:
            raise SystemExit(f"GATE: empty canonical target for {example_id}")
        items.append({
            "example_id": example_id,
            "corpus": example["corpus"],
            "kind": example["kind"],
            "prompt_ids": prompt_ids,
            "censored_ids": censored_ids,
            "answer_ids": response["token_ids"],
            "answer_text": response.get("answer_text", ""),
            "canonical_text": canonical,
            "canonical_ids": canonical_ids,
            "passage_start": cut,
            "passage_stop": cut + gap,
            "position_gap": gap,
            "teacher_word_acc": response.get("word_acc"),
        })
    chapter_map = ROOT / "data/combined/quij_chapter_map.json"
    if chapter_map.exists():
        mapping = json.loads(chapter_map.read_text(encoding="utf-8"))
        for item in items:
            item["corpus"] = mapping.get(item["example_id"], item["corpus"])
    if not limit and len(items) != len(responses):
        raise SystemExit("GATE: data coverage is not complete")
    return items, stop_id


# ---------------------------------------------------------------------------
# Tensor plumbing and passage interventions
# ---------------------------------------------------------------------------

def resolve_stack(causal_lm):
    for path in (
        "model", "language_model", "model.language_model",
        "language_model.model", "model.model",
    ):
        obj = causal_lm
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
            head = getattr(causal_lm, "lm_head", None)
            if head is None:
                head = getattr(getattr(causal_lm, "language_model", causal_lm),
                               "lm_head", None)
            if head is None:
                raise SystemExit("GATE: decoder stack has no LM head")
            return obj, head
    raise SystemExit(f"GATE: cannot resolve text decoder on {type(causal_lm)}")


def collate(items: list[dict], key: str, pad_id: int, device):
    import torch

    sequences = [item[key] + item["answer_ids"] for item in items]
    width = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    answer_spans, passage_spans = [], []
    for row, (item, sequence) in enumerate(zip(items, sequences)):
        ids[row, :len(sequence)] = torch.tensor(sequence)
        mask[row, :len(sequence)] = 1
        answer_spans.append((len(item[key]) - 1, len(item["answer_ids"])))
        if key == "prompt_ids":
            passage_spans.append((item["passage_start"], item["passage_stop"]))
        else:
            passage_spans.append((0, 0))
    return (
        ids.to(device), mask.to(device), answer_spans, passage_spans,
        [len(sequence) for sequence in sequences],
    )


def slice_rows(hidden, spans):
    return [hidden[row, start:start + length]
            for row, (start, length) in enumerate(spans)]


def head_logits(lm_head, hidden, softcap):
    import torch

    logits = lm_head(hidden).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


class LayerwiseTaps:
    """Detach every block input and retain raw outputs plus replay kwargs."""

    def __init__(self, layers):
        self.layers = list(layers)
        self.inputs = [None] * len(self.layers)
        self.outputs = [None] * len(self.layers)
        self.calls = [None] * len(self.layers)
        self.handles = []

    def _pre(self, index):
        def hook(_module, args, kwargs):
            import torch

            kwargs = dict(kwargs)
            if args and torch.is_tensor(args[0]):
                detached = args[0].detach()
                self.inputs[index] = detached
                self.calls[index] = (tuple(args[1:]), dict(kwargs))
                return (detached,) + tuple(args[1:]), kwargs
            hidden = kwargs.get("hidden_states")
            if torch.is_tensor(hidden):
                detached = hidden.detach()
                kwargs["hidden_states"] = detached
                replay_kwargs = dict(kwargs)
                replay_kwargs.pop("hidden_states")
                self.inputs[index] = detached
                self.calls[index] = ((), replay_kwargs)
                return args, kwargs
            raise RuntimeError(f"block {index} called without hidden states")
        return hook

    def _post(self, index):
        def hook(_module, _args, output):
            self.outputs[index] = output[0] if isinstance(output, tuple) else output
        return hook

    @contextmanager
    def active(self):
        try:
            for index, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_pre_hook(
                    self._pre(index), with_kwargs=True
                ))
                self.handles.append(layer.register_forward_hook(self._post(index)))
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()

    def reset(self):
        self.inputs = [None] * len(self.layers)
        self.outputs = [None] * len(self.layers)
        self.calls = [None] * len(self.layers)


class PassageBlocker:
    """Architecture-aware, retrieval-only passage censorship.

    Full attention receives an additive query/key mask.  Qwen GatedDeltaNet
    receives zero content rows at its token mixer; the enclosing residual
    stream remains intact.
    """

    def __init__(self, layers, blocked_layers: list[int],
                 passage_spans: list[tuple[int, int]]):
        self.layers = list(layers)
        self.blocked = sorted(set(blocked_layers))
        self.spans = passage_spans
        self.handles = []
        self.fired = {index: 0 for index in self.blocked}

    def _attention_hook(self, index):
        def hook(_module, args, kwargs):
            import torch

            kwargs = dict(kwargs)
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            mask = kwargs.get("attention_mask")
            mask_arg = None
            if mask is None:
                candidates = [
                    position for position, value in enumerate(args[1:], 1)
                    if torch.is_tensor(value) and value.ndim == 4
                ]
                if len(candidates) > 1:
                    raise RuntimeError(
                        f"CENSOR GATE: layer {index} has ambiguous 4D inputs"
                    )
                if candidates:
                    mask_arg = candidates[0]
                    mask = args[mask_arg]
            if mask is None:
                if not torch.is_tensor(hidden) or hidden.ndim != 3:
                    raise RuntimeError(
                        f"CENSOR GATE: layer {index} cannot synthesize mask"
                    )
                batch, length = hidden.shape[:2]
                invalid = torch.finfo(hidden.dtype).min
                mask = torch.full(
                    (length, length), invalid,
                    dtype=hidden.dtype, device=hidden.device,
                )
                mask = torch.triu(mask, diagonal=1)[None, None].expand(
                    batch, 1, length, length
                ).clone()
            if mask.ndim != 4:
                raise RuntimeError(
                    f"CENSOR GATE: layer {index} needs a 4D query/key mask, "
                    f"got shape {tuple(mask.shape)}"
                )
            masked = mask.clone()
            for row, (start, stop) in enumerate(self.spans):
                if stop <= start:
                    continue
                value = False if masked.dtype == torch.bool else torch.finfo(
                    masked.dtype
                ).min
                masked[row, ..., stop:, start:stop] = value
            if mask_arg is None:
                kwargs["attention_mask"] = masked
            else:
                args = list(args)
                args[mask_arg] = masked
                args = tuple(args)
            self.fired[index] += 1
            return args, kwargs
        return hook

    def _linear_hook(self, index):
        def hook(_module, args, kwargs):
            import torch

            kwargs = dict(kwargs)
            if args and torch.is_tensor(args[0]):
                hidden = args[0].clone()
                for row, (start, stop) in enumerate(self.spans):
                    hidden[row, start:stop] = 0
                self.fired[index] += 1
                return (hidden,) + tuple(args[1:]), kwargs
            hidden = kwargs.get("hidden_states")
            if torch.is_tensor(hidden):
                hidden = hidden.clone()
                for row, (start, stop) in enumerate(self.spans):
                    hidden[row, start:stop] = 0
                kwargs["hidden_states"] = hidden
                self.fired[index] += 1
                return args, kwargs
            raise RuntimeError(
                f"CENSOR GATE: recurrent layer {index} has no hidden state"
            )
        return hook

    def __enter__(self):
        for index in self.blocked:
            layer = self.layers[index]
            if hasattr(layer, "self_attn"):
                module, hook = layer.self_attn, self._attention_hook(index)
            elif hasattr(layer, "linear_attn"):
                module, hook = layer.linear_attn, self._linear_hook(index)
            else:
                raise RuntimeError(
                    f"CENSOR GATE: layer {index} has no supported token mixer"
                )
            self.handles.append(
                module.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return self

    def assert_fired(self):
        missed = [index for index, count in self.fired.items() if count == 0]
        if missed:
            raise RuntimeError(f"CENSOR GATE: hooks did not fire: {missed}")

    def __exit__(self, _type, _value, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LocalObjective:
    """Normalized hidden Huber plus bounded frozen-vocabulary JS."""

    def __init__(self, final_norm, lm_head, softcap, hidden_weight: float,
                 js_weight: float):
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.softcap = softcap
        self.hidden_weight = hidden_weight
        self.js_weight = js_weight
        self.replicas = {}

    def _replica(self, device):
        if device not in self.replicas:
            from accelerate.hooks import remove_hook_from_module

            norm = copy.deepcopy(self.final_norm)
            head = copy.deepcopy(self.lm_head)
            # device_map hooks belong to the original placement and must not
            # redirect these frozen per-layer measurement replicas.
            remove_hook_from_module(norm, recurse=True)
            remove_hook_from_module(head, recurse=True)
            norm = norm.to(device).eval()
            head = head.to(device).eval()
            for module in (norm, head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.replicas[device] = (norm, head)
        return self.replicas[device]

    def components(self, student, teacher):
        import torch
        import torch.nn.functional as functional

        student_f, teacher_f = student.float(), teacher.float()
        scale = teacher_f.pow(2).mean().sqrt().clamp_min(1e-8)
        hidden = functional.smooth_l1_loss(
            student_f / scale, teacher_f / scale, beta=1.0
        )
        norm, head = self._replica(student.device)
        head_dtype = head.weight.dtype
        student_logits = head(norm(student.to(head_dtype))).float()
        with torch.no_grad():
            teacher_logits = head(norm(teacher.to(head_dtype))).float()
        if self.softcap:
            student_logits = self.softcap * torch.tanh(
                student_logits / self.softcap
            )
            teacher_logits = self.softcap * torch.tanh(
                teacher_logits / self.softcap
            )
        student_logp = functional.log_softmax(student_logits, dim=-1)
        teacher_logp = functional.log_softmax(teacher_logits, dim=-1)
        student_p = student_logp.exp()
        with torch.no_grad():
            teacher_p = teacher_logp.exp()
        log_mid = (0.5 * (student_p + teacher_p)).clamp_min(1e-30).log()
        teacher_mid = functional.kl_div(
            log_mid, teacher_logp, log_target=True, reduction="batchmean"
        )
        student_mid = (student_p * (student_logp - log_mid)).sum(-1).mean()
        js = 0.5 * (teacher_mid + student_mid)
        return hidden, js

    def total(self, hidden, js):
        return self.hidden_weight * hidden + self.js_weight * js

    @staticmethod
    def anchor(student, teacher):
        import torch.nn.functional as functional

        scale = teacher.float().pow(2).mean().sqrt().clamp_min(1e-8)
        return functional.smooth_l1_loss(
            student.float() / scale, teacher.float() / scale, beta=1.0
        )


# ---------------------------------------------------------------------------
# In-pipeline evaluations
# ---------------------------------------------------------------------------

def choose_panel(items: list[dict], per_corpus: int, seed: int) -> list[dict]:
    grouped = {}
    for item in items:
        grouped.setdefault(item["corpus"], []).append(item)
    rng = random.Random(seed)
    chosen = []
    for _, group in sorted(grouped.items()):
        chosen.extend(rng.sample(group, min(per_corpus, len(group))))
    return chosen


def generate_rows(model, tokenizer, items: list[dict], device, stop_id: int,
                  batch_size: int, input_key: str, condition: str,
                  deployment_budget: int = 96, aligned: bool = False
                  ) -> tuple[list[dict], list[dict]]:
    """One decoding engine for natural/teacher/budgeted/aligned conditions."""
    import torch

    model.eval()
    pad_id = tokenizer.pad_token_id
    if aligned:
        pad_id = pad_id or tokenizer.eos_token_id
    elif pad_id is None:
        pad_id = tokenizer.eos_token_id
    sufficient_budget = deployment_budget if aligned else max(
        deployment_budget,
        max(len(item["answer_ids"]) for item in items) + 16,
    )
    full_rows, deployment_rows = [], []
    with torch.no_grad():
        for first in range(0, len(items), 1 if aligned else batch_size):
            batch = items[first:first + (1 if aligned else batch_size)]
            width = max(len(item[input_key]) for item in batch)
            if aligned:
                prompt = batch[0][input_key]
                cut, gap = batch[0]["passage_start"], batch[0]["position_gap"]
                ids = torch.tensor([prompt])
                mask = torch.ones_like(ids)
                positions = list(range(cut)) + [
                    cut + gap + offset for offset in range(len(prompt) - cut)
                ]
            else:
                ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for row, item in enumerate(batch):
                    prompt = item[input_key]
                    ids[row, width - len(prompt):] = torch.tensor(prompt)
                    mask[row, width - len(prompt):] = 1
            generation_args = {
                "input_ids": ids.to(device),
                "attention_mask": mask.to(device),
                "max_new_tokens": sufficient_budget,
                "do_sample": False,
                "use_cache": True,
                "eos_token_id": stop_id,
                "pad_token_id": pad_id,
            }
            if aligned:
                generation_args["position_ids"] = torch.tensor(
                    [positions], device=device
                )
            output = model.generate(
                **generation_args,
            )
            generated = output[:, width:].detach().cpu().tolist()
            for item, token_ids in zip(batch, generated):
                if stop_id in token_ids:
                    stop = token_ids.index(stop_id)
                    usable = token_ids[:stop]
                    finish = "stop"
                else:
                    usable = token_ids
                    finish = "length"
                text = tokenizer.decode(
                    usable, skip_special_tokens=True
                ).strip()
                row = score_generation(item, text, len(usable), finish)
                row["condition"] = condition
                row["generation_budget"] = sufficient_budget
                full_rows.append(row)
                if aligned:
                    continue
                limited = usable[:deployment_budget]
                limited_text = tokenizer.decode(
                    limited, skip_special_tokens=True
                ).strip()
                limited_finish = (
                    "stop" if finish == "stop" and len(usable) <= deployment_budget
                    else "length"
                )
                drow = score_generation(
                    item, limited_text, len(limited), limited_finish
                )
                drow["condition"] = condition
                drow["generation_budget"] = deployment_budget
                deployment_rows.append(drow)
    return full_rows, deployment_rows


def write_item_artifact(out_dir: Path, epoch: int, label: str,
                        rows: list[dict]) -> dict:
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"METRIC GATE: duplicate generation ids in {label}")
    path = out_dir / f"eval_items_e{epoch}_{label}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "items": len(rows),
    }


def output_eval(stack, lm_head, items, device, pad_id, softcap,
                frozen_base, teacher_cache: dict, batch_size: int,
                dataset_coverage: str) -> dict:
    import torch
    import torch.nn.functional as functional

    metric_device = lm_head.weight.device
    ce = torch.zeros((), dtype=torch.float64, device=metric_device)
    kl = torch.zeros((), dtype=torch.float64, device=metric_device)
    student_match = teacher_match = 0
    student_exact = teacher_exact = 0
    token_count = 0
    for first in range(0, len(items), batch_size):
        batch = items[first:first + batch_size]
        s_ids, s_mask, s_spans, _passage, _lengths = collate(
            batch, "censored_ids", pad_id, device
        )
        with torch.no_grad():
            student_hidden = stack(
                input_ids=s_ids, attention_mask=s_mask, use_cache=False
            ).last_hidden_state
        if not all(item["example_id"] in teacher_cache for item in batch):
            t_ids, t_mask, t_spans, _passage, _lengths = collate(
                batch, "prompt_ids", pad_id, device
            )
            with torch.no_grad(), frozen_base():
                teacher_hidden = stack(
                    input_ids=t_ids, attention_mask=t_mask, use_cache=False
                ).last_hidden_state
            for row, item in enumerate(batch):
                start, length = t_spans[row]
                teacher_cache[item["example_id"]] = teacher_hidden[
                    row, start:start + length
                ].detach().cpu()
        for item_index, item in enumerate(batch):
            cached = teacher_cache[item["example_id"]]
            s_start, length = s_spans[item_index]
            student_rows = student_hidden[
                item_index, s_start:s_start + length
            ]
            teacher_rows = cached.to(student_rows.device)
            ids = torch.tensor(item["answer_ids"], device=student_rows.device)
            s_ok, t_ok = [], []
            for row in range(0, length, 16):
                stop = min(length, row + 16)
                with torch.no_grad():
                    student_logits = head_logits(
                        lm_head, student_rows[row:stop], softcap
                    )
                    teacher_logits = head_logits(
                        lm_head, teacher_rows[row:stop], softcap
                    )
                    student_logp = functional.log_softmax(student_logits, -1)
                    teacher_logp = functional.log_softmax(teacher_logits, -1)
                    target = ids[row:stop].to(student_logp.device)
                    ce += functional.nll_loss(
                        student_logp, target, reduction="sum"
                    ).double()
                    kl += functional.kl_div(
                        student_logp, teacher_logp,
                        log_target=True, reduction="sum",
                    ).double()
                    s_ok.extend((student_logp.argmax(-1) == target).tolist())
                    t_ok.extend((teacher_logp.argmax(-1) == target).tolist())
            student_match += sum(s_ok)
            teacher_match += sum(t_ok)
            student_exact += int(all(s_ok))
            teacher_exact += int(all(t_ok))
            token_count += length
    if ce.requires_grad or kl.requires_grad:
        raise RuntimeError("EVAL GATE: output metrics acquired a graph")
    return {
        "CE_eval_loss": float(ce.item() / max(1, token_count)),
        "KL_eval_loss": float(kl.item() / max(1, token_count)),
        "student_argmax_acceptance": student_match / max(1, token_count),
        "teacher_argmax_acceptance": teacher_match / max(1, token_count),
        "student_exact_seq_rate": student_exact / max(1, len(items)),
        "teacher_exact_seq_rate": teacher_exact / max(1, len(items)),
        "student_exact_seq_match_answers": student_exact,
        "teacher_exact_seq_match_answers": teacher_exact,
        "exact_seq_answer_count": len(items),
        "answer_token_count": token_count,
        "expected_answer_token_count": sum(
            len(item["answer_ids"]) for item in items
        ),
        "dataset_item_count": len(items),
        "dataset_coverage": dataset_coverage,
        "token_coverage": "every_teacher_realized_answer_token",
        "answer_only": True,
        **EVAL_ONLY,
        "validation_subset": False,
        "aggregation": "token_weighted_mean",
        "inference_semantics": "teacher_forced_fixed_sequence_scoring",
        "teacher_forced": True,
        "autoregressive": False,
    }


def standard_eval(stack, lm_head, tokenizer, device, softcap, limit: int) -> dict:
    import torch
    import torch.nn.functional as functional

    files = {
        "arc_easy": "data/eval/arc_easy_v1.json",
        "arc_challenge": "data/eval/arc_challenge_v1.json",
        "hellaswag": "data/eval/hellaswag_v1.json",
    }
    results = {}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    stack.eval()
    with torch.no_grad():
        for task, relative in files.items():
            payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            examples = payload["items"][:limit]
            if len(examples) != limit:
                raise RuntimeError(
                    f"METRIC GATE: {task} has {len(examples)} items, want {limit}"
                )
            correct = 0
            for example in examples:
                prompt_ids = tokenizer.encode(
                    example["prompt"], add_special_tokens=False
                )
                sequences, spans = [], []
                for choice in example["choices"]:
                    choice_ids = tokenizer.encode(
                        choice, add_special_tokens=False
                    )
                    sequences.append(prompt_ids + choice_ids)
                    spans.append((len(prompt_ids) - 1, choice_ids))
                width = max(map(len, sequences))
                ids = torch.full(
                    (len(sequences), width), pad_id, dtype=torch.long
                )
                mask = torch.zeros_like(ids)
                for row, sequence in enumerate(sequences):
                    ids[row, :len(sequence)] = torch.tensor(sequence)
                    mask[row, :len(sequence)] = 1
                hidden = stack(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    use_cache=False,
                ).last_hidden_state
                scores = []
                for row, (start, choice_ids) in enumerate(spans):
                    logits = head_logits(
                        lm_head,
                        hidden[row, start:start + len(choice_ids)],
                        softcap,
                    )
                    target = torch.tensor(choice_ids, device=logits.device)
                    score = -functional.cross_entropy(
                        logits, target, reduction="sum"
                    ) / max(1, len(choice_ids))
                    scores.append(float(score))
                correct += int(
                    max(range(len(scores)), key=lambda index: scores[index])
                    == example["target"]
                )
            results[task] = {
                "accuracy": correct / max(1, len(examples)),
                "n": len(examples),
            }
    return {
        "tasks": results,
        "macro_accuracy": sum(row["accuracy"] for row in results.values())
        / len(results),
        "limit": limit,
        "benchmark_revisions": files,
        "inference_semantics": (
            "teacher_forced_normalized_option_log_likelihood"
        ),
        "teacher_forced": True,
        "autoregressive": False,
    }


# ---------------------------------------------------------------------------
# CLI and training
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument(
        "--method", choices=("causal_residual", "partial_teacher")
    )
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--lens-js-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--anchor-rows", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--panel-per-corpus", type=int, default=24)
    parser.add_argument("--generation-batch", type=int, default=8)
    parser.add_argument("--standard-limit", type=int, default=100)
    parser.add_argument("--attribution-every", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-data", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test-metrics", action="store_true")
    args = parser.parse_args()
    if args.self_test_metrics:
        return args
    missing = [
        name for name in ("preset", "method", "run_name")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("required outside self-test: " + ", ".join(
            f"--{name.replace('_', '-')}" for name in missing
        ))
    if args.hidden_weight <= 0 or args.lens_js_weight < 0:
        parser.error("objective weights must be hidden>0 and JS>=0")
    if args.epochs <= 0 or args.micro_batch <= 0 or args.grad_accum <= 0:
        parser.error("epochs and batch sizes must be positive")
    if (args.eval_every <= 0 or args.attribution_every <= 0
            or args.panel_per_corpus <= 0 or args.standard_limit <= 0
            or args.generation_batch <= 0):
        parser.error("evaluation cadences, limits, and batch sizes must be positive")
    if args.anchor_weight < 0 or args.anchor_rows < 0:
        parser.error("anchor weight/rows cannot be negative")
    return args


def main():
    args = parse_args()
    if args.self_test_metrics:
        self_test_metrics()
        return
    preset = PRESETS[args.preset]
    model_revision = os.environ.get("SELFUPDATE_HF_MODEL_REVISION")
    if model_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}", model_revision):
        raise SystemExit("GATE: malformed SELFUPDATE_HF_MODEL_REVISION")
    if not args.dry_data and model_revision is None:
        raise SystemExit(
            "GATE: training requires the staged model revision from "
            "scripts/trainv6.sbatch"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise SystemExit(
            "GATE: --run-name must contain only letters, digits, _, ., or -"
        )
    examples_path = ROOT / preset["examples"]
    responses_path = ROOT / preset["responses"]
    evidence_path = ROOT / preset["retrieval_evidence"]
    for path in (examples_path, responses_path, evidence_path):
        if not path.is_file():
            raise SystemExit(f"GATE: required artifact missing: {path}")
    summary_path = responses_path.with_name("summary.json")
    if not summary_path.is_file():
        raise SystemExit(f"GATE: response summary missing: {summary_path}")
    response_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if response_summary.get("model") != preset["model"]:
        raise SystemExit(
            f"GATE: response model {response_summary.get('model')!r} != "
            f"preset {preset['model']!r}"
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not evidence.get("final") or evidence.get("model") != preset["model"]:
        raise SystemExit("GATE: retrieval evidence is not final/model-matched")
    evidenced_layers = sorted(evidence.get("S", []))
    expected_evidence = (
        sorted(preset["retrieval_layers"])
        if args.preset == "gemma4_31b"
        else sorted(index for index in preset["retrieval_layers"]
                    if index % 4 == 3)
    )
    if evidenced_layers != expected_evidence:
        raise SystemExit(
            f"GATE: retrieval evidence {evidenced_layers} != "
            f"preset softmax set {expected_evidence}"
        )
    artifact_hashes = {
        "examples_sha256": sha256_file(examples_path),
        "responses_sha256": sha256_file(responses_path),
        "evidence_sha256": sha256_file(evidence_path),
        "summary_sha256": sha256_file(summary_path),
    }
    for name, observed in artifact_hashes.items():
        if observed != preset[name]:
            raise SystemExit(
                f"GATE: {name} {observed} != pinned {preset[name]}"
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(preset["model"])
    items, stop_id = build_items(
        examples_path, responses_path, tokenizer, args.limit
    )
    if not args.limit and len(items) != 2071:
        raise SystemExit(f"GATE: expected 2071 items, got {len(items)}")
    if not args.limit:
        observed_tasks = {}
        for item in items:
            observed_tasks[item["kind"]] = observed_tasks.get(item["kind"], 0) + 1
        expected_tasks = {"next": 1490, "prev": 332, "cloze": 249}
        if observed_tasks != expected_tasks:
            raise SystemExit(
                f"GATE: task coverage {observed_tasks} != {expected_tasks}"
            )
    if args.preflight:
        items = items[:min(2, len(items))]
        args.epochs = 1
        args.eval_every = 1
        args.standard_limit = min(args.standard_limit, 2)
        args.panel_per_corpus = 1
    task_counts, corpus_counts = {}, {}
    for item in items:
        task_counts[item["kind"]] = task_counts.get(item["kind"], 0) + 1
        corpus_counts[item["corpus"]] = corpus_counts.get(item["corpus"], 0) + 1
        prompt_positions = [
            position for position in range(1, len(item["prompt_ids"]) - 1)
            if not item["passage_start"] <= position < item["passage_stop"]
        ]
        rng = random.Random(f"v6-anchor-{item['example_id']}-{args.seed}")
        count = min(args.anchor_rows, len(prompt_positions))
        item["anchor_rows"] = sorted(rng.sample(prompt_positions, count))
    attribution_probe_ids = [
        item["example_id"] for item in choose_panel(items, 1, args.seed)
    ]
    print(json.dumps({
        "gate": "data",
        "preset": args.preset,
        "model": preset["model"],
        "items": len(items),
        "answer_tokens": sum(len(item["answer_ids"]) for item in items),
        "tasks": task_counts,
        "corpora": corpus_counts,
        "examples_sha256": sha256_file(examples_path),
        "responses_sha256": sha256_file(responses_path),
        "model_revision": model_revision,
        "gradient_attribution_probe_ids": attribution_probe_ids,
        "canonical_targets": "complete",
        "tokenizer_roundtrip": "complete",
    }, ensure_ascii=False), flush=True)
    if args.dry_data:
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    out_dir = ROOT / "runs" / args.run_name
    if out_dir.exists():
        raise SystemExit(f"GATE: fresh run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    metrics = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")

    def log(kind: str, **fields):
        row = {
            "kind": kind,
            "metric_schema": METRIC_SCHEMA,
            "pipeline_version": 6,
            "run_name": args.run_name,
            "t": time.time(),
            **fields,
        }
        metrics.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics.flush()

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    log(
        "provenance",
        source_commit=commit,
        monolith_sha256=sha256_file(Path(__file__)),
        args=vars(args),
        preset=args.preset,
        model=preset["model"],
        model_revision=model_revision,
        method=args.method,
        examples_sha256=sha256_file(examples_path),
        responses_sha256=sha256_file(responses_path),
        retrieval_evidence_sha256=sha256_file(evidence_path),
        training_input="detached current full-slot blocked student state",
        training_target=(
            "same-input frozen passage-visible block output"
            if args.method == "causal_residual"
            else "partial-passage frozen teacher block output"
        ),
        local_objective={
            "hidden_huber_weight": args.hidden_weight,
            "frozen_final_norm_head_js_weight": args.lens_js_weight,
            "depth_uniform": True,
        },
        censorship={
            "softmax": "post-passage queries cannot read passage keys",
            "qwen_recurrent": (
                "passage content zeroed at GatedDeltaNet token mixer; "
                "sequence clock and recurrent decay retained"
            ),
        },
        end_to_end_student_training=False,
        cross_block_gradient=False,
        frozen_vocabulary=True,
    )

    print(f"loading {preset['model']} (bf16, device_map=auto)", flush=True)
    full = AutoModelForCausalLM.from_pretrained(
        preset["model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    loaded_revision = getattr(full.config, "_commit_hash", None)
    if loaded_revision is not None and loaded_revision != model_revision:
        raise SystemExit(
            f"GATE: loaded model revision {loaded_revision} != staged "
            f"revision {model_revision}"
        )
    full.config.use_cache = False
    raw_stack, _raw_head = resolve_stack(full)
    stack_prefix = next(
        name for name, module in full.named_modules() if module is raw_stack
    )
    target_names = [
        name for name, module in full.named_modules()
        if name.startswith(stack_prefix + ".layers.")
        and isinstance(module, torch.nn.Linear)
    ]
    if not target_names:
        raise SystemExit("GATE: no decoder Linear modules for LoRA")
    moe = [
        name for name, _ in full.named_parameters()
        if name.startswith(stack_prefix + ".layers.")
        and (".experts." in name or ".router." in name)
    ]
    if moe:
        raise SystemExit("GATE: v6 presets must remain dense; MoE found")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_names,
        bias="none",
        task_type="CAUSAL_LM",
    )
    full = get_peft_model(full, lora)
    stack, lm_head = resolve_stack(full.get_base_model())
    layers = list(stack.layers)
    shared_kv = getattr(stack.config, "num_kv_shared_layers", 0) or 0
    if shared_kv:
        raise SystemExit(
            f"LAYERWISE GATE: num_kv_shared_layers={shared_kv} bypasses "
            "detached block boundaries"
        )
    if len(layers) != preset["layers"]:
        raise SystemExit(
            f"GATE: preset expects {preset['layers']} layers, got {len(layers)}"
        )
    observed_types = [
        "linear_attention" if hasattr(layer, "linear_attn")
        else "full_attention" if hasattr(layer, "self_attn")
        else "unknown"
        for layer in layers
    ]
    if preset["layer_types"] is not None and observed_types != preset["layer_types"]:
        raise SystemExit("GATE: Qwen layer topology differs from the preset")
    if args.preset == "qwen36_27b":
        recurrent = sum(kind == "linear_attention" for kind in observed_types)
        attention = sum(kind == "full_attention" for kind in observed_types)
        if (recurrent, attention) != (48, 16):
            raise SystemExit(
                f"GATE: Qwen intervention expects 48+16 layers, got "
                f"{recurrent}+{attention}"
            )
    device = next(stack.embed_tokens.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    softcap = getattr(stack.config, "final_logit_softcapping", None)
    final_norm = getattr(stack, "norm", None)
    if final_norm is None:
        raise SystemExit("GATE: final norm required for the v6 lens")
    for name, parameter in full.named_parameters():
        if parameter.requires_grad and (
            ".layers." not in name or "lora_" not in name
        ):
            raise SystemExit(f"GATE: trainable non-block LoRA parameter: {name}")
    for module in (stack.embed_tokens, final_norm, lm_head):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise SystemExit("GATE: vocabulary stack is trainable")
    vocab_fp = tuple(
        float(module.weight.detach().float().sum())
        for module in (stack.embed_tokens, final_norm, lm_head)
    )

    @contextmanager
    def frozen_base():
        with full.disable_adapter():
            yield

    def vocab_tripwire():
        now = tuple(
            float(module.weight.detach().float().sum())
            for module in (stack.embed_tokens, final_norm, lm_head)
        )
        if now != vocab_fp:
            raise RuntimeError(f"FROZEN-VOCAB GATE: {vocab_fp} -> {now}")

    trainable = [
        parameter for parameter in full.parameters() if parameter.requires_grad
    ]
    trainable_by_layer = []
    named_trainable = [
        (name, parameter) for name, parameter in full.named_parameters()
        if parameter.requires_grad
    ]
    for index in range(len(layers)):
        selected = [
            parameter for name, parameter in named_trainable
            if f".layers.{index}." in name
        ]
        if not selected:
            raise SystemExit(f"GATE: layer {index} has no trainable LoRA")
        trainable_by_layer.append(selected)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    initial_adapters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_trainable
    }
    objective = LocalObjective(
        final_norm, lm_head, softcap, args.hidden_weight, args.lens_js_weight
    )
    taps = LayerwiseTaps(layers)
    all_layers = list(range(len(layers)))
    retrieval = sorted(set(preset["retrieval_layers"]))
    if not retrieval or any(index not in all_layers for index in retrieval):
        raise SystemExit("GATE: invalid retrieval-layer preset")
    blocked_teacher = [index for index in all_layers if index not in retrieval]
    log(
        "architecture",
        layer_count=len(layers),
        layer_types=observed_types,
        lora_target_count=len(target_names),
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        model_revision=model_revision,
        transformers_commit_hash=loaded_revision,
        retrieval_layers=retrieval,
        blocked_partial_teacher_layers=blocked_teacher,
    )

    partial_cache = {}
    anchor_cache = {}
    output_teacher_cache = {}
    baseline_generation = {}
    baseline_generation_detail = {}
    baseline_output = None
    baseline_standard = None
    items_seen = 0
    destructive_intervals = 0
    certified = False

    def frozen_targets(ids, mask, passage_spans, blocked, selectors):
        with torch.no_grad(), frozen_base(), PassageBlocker(
            layers, blocked, passage_spans
        ) as blocker, taps.active():
            stack(input_ids=ids, attention_mask=mask, use_cache=False)
            outputs = list(taps.outputs)
        blocker.assert_fired()
        taps.reset()
        return [
            [outputs[index][row, selector].clone()
             for row, selector in enumerate(selectors)]
            for index in all_layers
        ]

    def cached_frozen_rows(cache, batch, build):
        if all(item["example_id"] in cache for item in batch):
            return [[
                cache[item["example_id"]][index].to(
                    next(layers[index].parameters()).device
                ) for item in batch
            ] for index in all_layers]
        rows = build()
        for row, item in enumerate(batch):
            cache[item["example_id"]] = [
                rows[index][row].detach().cpu() for index in all_layers
            ]
        return rows

    def student_forward(ids, mask, passage_spans):
        with PassageBlocker(
            layers, all_layers, passage_spans
        ) as blocker, taps.active():
            stack(input_ids=ids, attention_mask=mask, use_cache=False)
            outputs = list(taps.outputs)
            inputs = list(taps.inputs)
            calls = list(taps.calls)
        blocker.assert_fired()
        taps.reset()
        if any(hidden.requires_grad for hidden in inputs):
            raise RuntimeError("LAYERWISE GATE: a block input carries grad")
        return outputs, inputs, calls

    def causal_targets(inputs, calls, passage_spans, answer_spans):
        targets, effects = [], []
        with torch.no_grad(), frozen_base():
            for index, layer in enumerate(layers):
                tail_args, kwargs = calls[index]
                visible = layer(inputs[index], *tail_args, **dict(kwargs))
                visible = visible[0] if isinstance(visible, tuple) else visible
                with PassageBlocker(
                    layers, [index], passage_spans
                ) as blocker:
                    blocked = layer(
                        inputs[index], *tail_args, **dict(kwargs)
                    )
                blocker.assert_fired()
                blocked = blocked[0] if isinstance(blocked, tuple) else blocked
                target_rows = [
                    row.clone() for row in slice_rows(visible, answer_spans)
                ]
                effect_rows = [
                    (v - b).float().pow(2).mean().sqrt().detach()
                    for v, b in zip(
                        slice_rows(visible, answer_spans),
                        slice_rows(blocked, answer_spans),
                    )
                ]
                targets.append(target_rows)
                effects.append(effect_rows)
        return targets, effects

    def layer_loss(index, outputs, inputs, targets, anchors, batch,
                   answer_spans):
        student_rows = torch.cat(slice_rows(outputs[index], answer_spans))
        teacher_rows = torch.cat(targets[index]).to(student_rows.device)
        hidden, js = objective.components(student_rows, teacher_rows)
        anchor = torch.zeros((), device=student_rows.device)
        if args.anchor_weight:
            active_rows = [
                row for row, item in enumerate(batch) if item["anchor_rows"]
            ]
            if active_rows:
                student_anchor = torch.cat([
                    outputs[index][row, batch[row]["anchor_rows"]]
                    for row in active_rows
                ])
                teacher_anchor = torch.cat([
                    anchors[index][row].to(student_anchor.device)
                    for row in active_rows
                ])
                anchor = objective.anchor(student_anchor, teacher_anchor)
        total = objective.total(hidden, js) + args.anchor_weight * anchor
        return hidden, js, anchor, total

    def certify_intervention():
        """Fail before training unless censorship is active and content-invariant."""
        probe = [items[0]]
        ids, mask, answer_spans, passage_spans, _lengths = collate(
            probe, "prompt_ids", pad_id, device
        )
        start, length = answer_spans[0]
        replaced = ids.clone()
        passage_start, passage_stop = passage_spans[0]
        replacement_id = tokenizer.eos_token_id
        if replacement_id is None:
            replacement_id = 0
        replaced[0, passage_start:passage_stop] = replacement_id
        with torch.no_grad(), frozen_base():
            native_a = stack(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state[0, start:start + length]
            native_b = stack(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state[0, start:start + length]
            with PassageBlocker(
                layers, all_layers, passage_spans
            ) as blocked_hooks:
                blocked = stack(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).last_hidden_state[0, start:start + length]
            blocked_hooks.assert_fired()
            with PassageBlocker(
                layers, all_layers, passage_spans
            ) as replaced_hooks:
                replaced_blocked = stack(
                    input_ids=replaced, attention_mask=mask, use_cache=False
                ).last_hidden_state[0, start:start + length]
            replaced_hooks.assert_fired()
            if args.method == "partial_teacher":
                with PassageBlocker(
                    layers, blocked_teacher, passage_spans
                ) as partial_hooks:
                    stack(input_ids=ids, attention_mask=mask, use_cache=False)
                partial_hooks.assert_fired()
        repeat_error = float((native_a.float() - native_b.float()).abs().max())
        invariance_error = float(
            (blocked.float() - replaced_blocked.float()).abs().max()
        )
        intervention_rms = float(
            (native_a.float() - blocked.float()).pow(2).mean().sqrt()
        )
        if repeat_error > 1e-5:
            raise RuntimeError(
                f"CENSOR GATE: native forward is nondeterministic ({repeat_error})"
            )
        if invariance_error > 5e-3:
            raise RuntimeError(
                "CENSOR GATE: blocked answer state depends on passage "
                f"content (max error {invariance_error})"
            )
        if intervention_rms <= 1e-6:
            raise RuntimeError("CENSOR GATE: passage intervention is inert")
        log(
            "intervention_certification",
            example_id=probe[0]["example_id"],
            native_repeat_max_abs=repeat_error,
            blocked_replacement_max_abs=invariance_error,
            visible_vs_blocked_rms=intervention_rms,
            softmax_layers=observed_types.count("full_attention"),
            recurrent_layers=observed_types.count("linear_attention"),
            hooks_fired="all_expected",
        )

    def certify_middle(index, total):
        nonlocal certified
        if certified:
            return
        optimizer.zero_grad(set_to_none=True)
        total.backward(retain_graph=True)
        leaked = [
            name for name, parameter in named_trainable
            if parameter.grad is not None and f".layers.{index}." not in name
        ]
        inside = [
            name for name, parameter in named_trainable
            if parameter.grad is not None and f".layers.{index}." in name
        ]
        if leaked or not inside:
            raise RuntimeError(
                f"LAYERWISE GATE: inside={len(inside)} leaked={leaked[:4]}"
            )
        optimizer.zero_grad(set_to_none=True)
        certified = True
        log(
            "layerwise_certification",
            probed_layer=index,
            lora_tensors_with_grad=len(inside),
            leaks=0,
        )

    def log_generation(epoch: int, selected: list[dict], scope: str,
                       include_teacher: bool, include_aligned: bool):
        nonlocal baseline_generation, baseline_generation_detail
        expected_ids = [item["example_id"] for item in selected]

        def require_coverage(result, label, expected=expected_ids):
            observed = [row["example_id"] for row in result]
            if observed != expected:
                raise RuntimeError(
                    f"METRIC GATE: {label} generation coverage differs "
                    "from its selected items"
                )

        rows, deployment = generate_rows(
            full, tokenizer, selected, device, stop_id,
            args.generation_batch, "censored_ids",
            "student_censored_natural",
        )
        require_coverage(rows, "student sufficient")
        require_coverage(deployment, "student budget96")
        for label, result in (
            (f"{scope}_natural_sufficient", rows),
            (f"{scope}_natural_budget96", deployment),
        ):
            artifact = write_item_artifact(out_dir, epoch, label, result)
            summary = summarize_generation(result)
            baseline_key = label.split("_natural_", 1)[-1]
            current = {
                row["example_id"]: row["content_lcs"] for row in result
            }
            base = baseline_generation.get(baseline_key, {})
            base_detail = baseline_generation_detail.get(baseline_key, {})
            delta = paired_bootstrap(
                current, base, seed=args.seed + epoch * 1009
            ) if base else {"n": 0, "delta": None, "ci95": [None, None]}
            grouped_delta = grouped_bootstrap(
                result, base_detail, seed=args.seed + epoch * 1009
            ) if base_detail else {"by_corpus": {}, "by_kind": {}}
            if epoch == 0:
                baseline_generation[baseline_key] = current
                baseline_generation_detail[baseline_key] = {
                    row["example_id"]: row for row in result
                }
            log(
                "generation_eval",
                epoch=epoch,
                scope=scope,
                condition=result[0]["condition"],
                generation_budget=result[0]["generation_budget"],
                natural_positions=True,
                primary=(baseline_key == "sufficient"),
                canonical_content_primary=True,
                teacher_answer_lcs_role="secondary_diagnostic",
                summary=summary,
                paired_epoch0=delta,
                paired_epoch0_groups=grouped_delta,
                artifact=artifact,
                evaluated_item_count=len(result),
                **EVAL_ONLY,
                inference_semantics="autoregressive_greedy_rollout",
            )
        if include_teacher:
            with frozen_base():
                teacher_rows, teacher_budget = generate_rows(
                    full, tokenizer, selected, device, stop_id,
                    args.generation_batch, "prompt_ids",
                    "teacher_uncensored_exact_path",
                )
            require_coverage(teacher_rows, "teacher sufficient")
            require_coverage(teacher_budget, "teacher budget96")
            for label, result in (
                (f"{scope}_teacher_sufficient", teacher_rows),
                (f"{scope}_teacher_budget96", teacher_budget),
            ):
                artifact = write_item_artifact(out_dir, epoch, label, result)
                log(
                    "teacher_ceiling_eval",
                    epoch=epoch,
                    scope=scope,
                    condition=result[0]["condition"],
                    generation_budget=result[0]["generation_budget"],
                    summary=summarize_generation(result),
                    artifact=artifact,
                    exact_student_path=True,
                    evaluated_item_count=len(result),
                    **EVAL_ONLY,
                )
        if include_aligned:
            panel = choose_panel(selected, args.panel_per_corpus, args.seed)
            aligned, _unused_budget = generate_rows(
                full, tokenizer, panel, device, stop_id, 1, "censored_ids",
                "student_censored_aligned_diagnostic", aligned=True,
            )
            require_coverage(
                aligned, "student aligned",
                [item["example_id"] for item in panel],
            )
            artifact = write_item_artifact(
                out_dir, epoch, f"{scope}_aligned_diagnostic", aligned
            )
            log(
                "aligned_generation_diagnostic",
                epoch=epoch,
                scope="fixed_panel",
                condition="student_censored_aligned_diagnostic",
                primary=False,
                natural_positions=False,
                summary=summarize_generation(aligned),
                artifact=artifact,
                evaluated_item_count=len(aligned),
                **EVAL_ONLY,
            )
        return summarize_generation(rows), paired_bootstrap(
            {row["example_id"]: row["content_lcs"] for row in rows},
            baseline_generation.get("sufficient", {}),
            seed=args.seed + epoch * 1009,
        ), rows

    def evaluate(epoch: int, mean_local=None, epoch1_loss=None):
        nonlocal baseline_output, baseline_standard, destructive_intervals
        full.eval()
        output = output_eval(
            stack, lm_head, items, device, pad_id, softcap,
            frozen_base, output_teacher_cache, args.micro_batch,
            ("whole_training_set_epoch_zero" if epoch == 0 else
             "whole_training_set_once_per_completed_epoch"),
        )
        if output["answer_token_count"] != output["expected_answer_token_count"]:
            raise RuntimeError("EVAL GATE: teacher-token coverage mismatch")
        log("teacher_output_eval", epoch=epoch, **output)
        at_eval = (
            epoch == 0 or epoch % args.eval_every == 0
            or epoch == args.epochs
        )
        if not at_eval:
            return False

        standard = standard_eval(
            stack, lm_head, tokenizer, device, softcap, args.standard_limit
        )
        if baseline_standard is None:
            baseline_standard = standard
        deltas = {
            task: (
                standard["tasks"][task]["accuracy"]
                - baseline_standard["tasks"][task]["accuracy"]
            )
            for task in standard["tasks"]
        }
        standard["epoch0_deltas"] = deltas
        standard["worst_delta"] = min(deltas.values())
        standard["macro_delta"] = (
            standard["macro_accuracy"] - baseline_standard["macro_accuracy"]
        )
        log(
            "standard_eval",
            epoch=epoch,
            **standard,
            **EVAL_ONLY,
        )

        if epoch == 0:
            generation, _generation_delta, _generation_rows = log_generation(
                epoch, items, "whole_set",
                include_teacher=True, include_aligned=True,
            )
            baseline_output = output
            print(
                f"eval e{epoch}: content={generation['overall']['content_lcs']:.4f} "
                f"CE={output['CE_eval_loss']:.4f} "
                f"argmax={output['student_argmax_acceptance']:.4f} "
                f"standard={standard['macro_accuracy']:.3f}",
                flush=True,
            )
            return False

        panel = choose_panel(items, args.panel_per_corpus, args.seed)
        panel_summary, panel_delta, panel_rows = log_generation(
            epoch, panel, "fixed_panel",
            include_teacher=False, include_aligned=False,
        )
        candidate = (
            panel_delta["delta"] is not None
            and panel_delta["delta"] >= 0.03
        )
        full_gate = epoch == args.epochs or candidate
        if full_gate:
            _generation, generation_delta, generation_rows = log_generation(
                epoch, items, "whole_set",
                include_teacher=False, include_aligned=True,
            )
        else:
            _generation, generation_delta, generation_rows = (
                panel_summary, panel_delta, panel_rows
            )
        full.save_pretrained(str(out_dir / f"checkpoint_e{epoch}"))

        argmax_ratio = (
            output["student_argmax_acceptance"]
            / max(1e-30, baseline_output["student_argmax_acceptance"])
        )
        damage = (
            argmax_ratio < 0.8
            or standard["worst_delta"] <= -0.05
            or standard["macro_delta"] <= -0.03
        )
        flat = (
            generation_delta["delta"] is not None
            and generation_delta["delta"] < 0.01
            and generation_delta["ci95"][0] <= 0
            <= generation_delta["ci95"][1]
        )
        loss_improved = mean_local <= 0.9 * (
            epoch1_loss if epoch1_loss is not None else mean_local
        )
        if items_seen >= 12000 and loss_improved and flat and damage:
            destructive_intervals += 1
        else:
            destructive_intervals = 0
        stop_due = destructive_intervals >= 2
        if stop_due and not full_gate:
            # A panel may trigger the stop review, but a terminal decision may
            # never leave only panel evidence behind. Confirm flatness and
            # emit the ordinary aligned diagnostics on the whole set first.
            _generation, generation_delta, generation_rows = log_generation(
                epoch, items, "whole_set",
                include_teacher=False, include_aligned=True,
            )
            full_gate = True
            flat = (
                generation_delta["delta"] is not None
                and generation_delta["delta"] < 0.01
                and generation_delta["ci95"][0] <= 0
                <= generation_delta["ci95"][1]
            )
            if not flat:
                destructive_intervals = 0
                stop_due = False
        terminal = epoch == args.epochs or stop_due
        base_detail = baseline_generation_detail.get("sufficient", {})
        comparable_rows = [
            row for row in generation_rows
            if row["example_id"] in base_detail
        ]
        new_recitation = any(
            row["recitation"]
            and not base_detail[row["example_id"]]["recitation"]
            for row in comparable_rows
        )
        prefix_delta = sum(
            row["longest_correct_prefix_fraction"]
            - base_detail[row["example_id"]][
                "longest_correct_prefix_fraction"
            ]
            for row in comparable_rows
        ) / max(1, len(comparable_rows))
        promotion = {
            "content_delta_at_least_0p03": (
                generation_delta["delta"] is not None
                and generation_delta["delta"] >= 0.03
            ),
            "content_ci_excludes_zero": (
                generation_delta["ci95"][0] is not None
                and generation_delta["ci95"][0] > 0
            ),
            "argmax_at_least_0p8_epoch0": argmax_ratio >= 0.8,
            "standard_worst_delta_above_minus_0p05": (
                standard["worst_delta"] > -0.05
            ),
            "standard_macro_delta_above_minus_0p03": (
                standard["macro_delta"] > -0.03
            ),
            "new_recitation_or_prefix_gain": (
                new_recitation or prefix_delta >= 0.03
            ),
            "replication": "pending_second_training_seed",
        }
        log(
            "promotion_gate",
            epoch=epoch,
            scope="whole_set" if full_gate else "fixed_panel",
            criteria=promotion,
            new_recitation=new_recitation,
            mean_prefix_fraction_delta=prefix_delta,
            decision_role=(
                "confirmatory_terminal" if terminal
                else "exploratory_intermediate"
            ),
            eligible_this_run=(
                terminal and full_gate
                and all(value for key, value in promotion.items()
                        if key != "replication")
            ),
            final_promotion=False,
        )
        if not stop_due:
            return False
        log(
            "aborted_stop_rule",
            epoch=epoch,
            items_seen=items_seen,
            local_loss_improved_at_least_10pct=loss_improved,
            content_flat=flat,
            damage=True,
        )
        print("ABORT: preregistered post-12k stop rule", flush=True)
        return True

    # The intervention is a mandatory launch gate, not an optional test.
    certify_intervention()

    # Epoch zero is measured under precisely the same student path.
    evaluate(0)

    last_epoch = 0
    aborted = False
    epoch1_loss = None
    for epoch in range(1, args.epochs + 1):
        full.train()
        started = time.time()
        ordered = sorted(
            items, key=lambda item: len(item["prompt_ids"]) + len(item["answer_ids"])
        )
        batches = [
            ordered[first:first + args.micro_batch]
            for first in range(0, len(ordered), args.micro_batch)
        ]
        attribution_batch_keys = {
            tuple(item["example_id"] for item in batch)
            for batch in batches
            if any(item["example_id"] in attribution_probe_ids for item in batch)
        }
        random.Random(args.seed * 1000 + epoch).shuffle(batches)
        layer_metrics = LayerMetricLedger(len(layers))
        grad_attr_sums = [None] * len(layers)
        grad_attr_batch_keys_seen = []
        grad_attr_item_ids = set()
        grad_attr_answer_tokens = 0
        grad_norm_sum, grad_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(batches):
            ids, mask, answer_spans, passage_spans, _lengths = collate(
                batch, "prompt_ids", pad_id, device
            )
            if args.method == "partial_teacher":
                targets = cached_frozen_rows(
                    partial_cache, batch,
                    lambda: frozen_targets(
                        ids, mask, passage_spans, blocked_teacher,
                        [slice(start, start + length)
                         for start, length in answer_spans],
                    )
                )
            outputs, inputs, calls = student_forward(
                ids, mask, passage_spans
            )
            if args.method == "causal_residual":
                targets, effects = causal_targets(
                    inputs, calls, passage_spans, answer_spans
                )
            else:
                effects = None
            if args.anchor_weight:
                anchors = cached_frozen_rows(
                    anchor_cache, batch,
                    lambda: frozen_targets(
                        ids, mask, passage_spans, all_layers,
                        [item["anchor_rows"] for item in batch],
                    )
                )
            else:
                anchors = [[] for _ in all_layers]

            middle = len(layers) // 2
            if not certified:
                _h, _j, _a, cert_total = layer_loss(
                    middle, outputs, inputs, targets, anchors,
                    batch, answer_spans,
                )
                certify_middle(middle, cert_total)

            batch_key = tuple(item["example_id"] for item in batch)
            attribute = (
                (epoch == 1 or epoch % args.attribution_every == 0)
                and batch_key in attribution_batch_keys
            )
            answer_rows_this_batch = sum(length for _start, length in answer_spans)
            anchor_rows_this_batch = sum(
                len(item["anchor_rows"]) for item in batch
            )
            layer_metrics.count(
                answer_rows_this_batch, anchor_rows_this_batch, len(batch)
            )
            if attribute:
                grad_attr_batch_keys_seen.append(batch_key)
                grad_attr_item_ids.update(batch_key)
                grad_attr_answer_tokens += answer_rows_this_batch
            for index in all_layers:
                hidden, js, anchor, total = layer_loss(
                    index, outputs, inputs, targets, anchors,
                    batch, answer_spans,
                )
                if attribute:
                    parameters = trainable_by_layer[index]
                    measured = objective_gradient_attribution(
                        {
                            "hidden": args.hidden_weight * hidden,
                            "lens_js": args.lens_js_weight * js,
                            "anchor": (args.anchor_weight * anchor
                                       if args.anchor_weight
                                       and anchor.requires_grad else None),
                        },
                        parameters,
                        hidden.device,
                    )
                    if grad_attr_sums[index] is None:
                        grad_attr_sums[index] = measured
                    else:
                        grad_attr_sums[index] = {
                            name: grad_attr_sums[index][name] + value
                            for name, value in measured.items()
                        }
                (
                    total / (len(layers) * args.grad_accum)
                ).backward()
                layer_metrics.add(
                    index,
                    hidden_huber=hidden.detach() * answer_rows_this_batch,
                    lens_js=js.detach() * answer_rows_this_batch,
                    anchor=anchor.detach() * anchor_rows_this_batch,
                    **({"causal_effect_rms": sum(effects[index])}
                       if effects is not None else {}),
                )
            items_seen += len(batch)
            if (
                (batch_index + 1) % args.grad_accum == 0
                or batch_index + 1 == len(batches)
            ):
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
                grad_norm_sum += float(grad_norm)
                grad_steps += 1
                optimizer.step()
                if args.weight_decay:
                    shrink = 1.0 - args.lr * args.weight_decay
                    with torch.no_grad():
                        for parameter in trainable:
                            parameter.mul_(shrink)
                optimizer.zero_grad(set_to_none=True)
            del outputs, inputs, calls, targets, anchors

        profiles, profile_aggregation, profile_denominators = (
            layer_metrics.finish()
        )
        expected_profile_denominators = {
            "answer_tokens": sum(len(item["answer_ids"]) for item in items),
            "anchor_rows": sum(len(item["anchor_rows"]) for item in items),
            "items": len(items),
        }
        if profile_denominators != expected_profile_denominators:
            raise RuntimeError(
                "METRIC GATE: layer-profile denominators "
                f"{profile_denominators} != {expected_profile_denominators}"
            )
        hidden_profile = profiles["hidden_huber_profile"]
        js_profile = profiles["lens_js_profile"]
        anchor_profile = profiles["anchor_profile"]
        if grad_attr_batch_keys_seen and (
                set(grad_attr_batch_keys_seen) != attribution_batch_keys):
            raise RuntimeError("METRIC GATE: attribution panel coverage mismatch")
        grad_attr = [
            (finish_gradient_attribution(row, len(grad_attr_batch_keys_seen))
             if row is not None else None)
            for row in grad_attr_sums
        ]
        for row in (entry for entry in grad_attr if entry is not None):
            numeric = [
                value for value in row.values()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
            if not all(math.isfinite(value) for value in numeric):
                raise RuntimeError("METRIC GATE: non-finite gradient attribution")
            shares = [
                row[f"{name}_share"]
                for name in ("hidden", "lens_js", "anchor")
            ]
            if row["share_defined"] and (
                    any(value is None for value in shares)
                    or abs(sum(shares) - 1.0) > 1e-4):
                raise RuntimeError(
                    f"METRIC GATE: invalid objective gradient shares {shares}"
                )
            if not row["share_defined"] and (
                    row["total_norm"] != 0.0
                    or any(value is not None for value in shares)):
                raise RuntimeError("METRIC GATE: zero-signal shares are defined")
            cosines = [
                row[f"{left}_{right}_cosine"]
                for left, right in (("hidden", "lens_js"),
                                    ("hidden", "anchor"),
                                    ("lens_js", "anchor"))
            ]
            if any(value is not None and abs(value) > 1.0001
                   for value in cosines):
                raise RuntimeError("METRIC GATE: gradient cosine outside [-1, 1]")
        deltas = [0.0] * len(layers)
        with torch.no_grad():
            for name, parameter in named_trainable:
                match = re.search(r"\.layers\.(\d+)\.", name)
                index = int(match.group(1))
                difference = (
                    parameter.detach().cpu().float()
                    - initial_adapters[name].float()
                )
                deltas[index] += float(difference.pow(2).sum())
        deltas = [math.sqrt(value) for value in deltas]
        mean_local = sum(
            args.hidden_weight * h + args.lens_js_weight * j
            + args.anchor_weight * a
            for h, j, a in zip(hidden_profile, js_profile, anchor_profile)
        ) / len(layers)
        if epoch == 1:
            epoch1_loss = mean_local
        log(
            "epoch",
            epoch=epoch,
            method=args.method,
            items_seen=items_seen,
            seconds=time.time() - started,
            mean_local_loss=mean_local,
            **profiles,
            gradient_attribution=grad_attr,
            gradient_attribution_scope={
                "selection": "fixed_one_item_per_corpus_or_chapter_stratum",
                "aggregation": (
                    "rms_norms_and_concatenated_cosines_over_fixed_batches"
                ),
                "trigger_item_ids": attribution_probe_ids,
                "evaluated_item_ids": sorted(grad_attr_item_ids),
                "evaluated_item_count": len(grad_attr_item_ids),
                "evaluated_batch_count": len(grad_attr_batch_keys_seen),
                "evaluated_answer_token_count": grad_attr_answer_tokens,
            },
            adapter_delta_l2_profile=deltas,
            mean_preclip_grad_norm=grad_norm_sum / max(1, grad_steps),
            objective_weights={
                "hidden_huber": args.hidden_weight,
                "lens_js": args.lens_js_weight,
                "anchor": args.anchor_weight,
            },
            profile_aggregation=profile_aggregation,
            profile_denominators=profile_denominators,
            depth_uniform=True,
        )
        vocab_tripwire()
        print(
            f"epoch {epoch}: loss={mean_local:.6f}, items_seen={items_seen}, "
            f"seconds={time.time() - started:.0f}",
            flush=True,
        )
        # CE/KL remain mandatory every completed epoch; the same funnel owns
        # the less-frequent generation and standard-capability gates.
        if evaluate(epoch, mean_local, epoch1_loss):
            aborted = True
            last_epoch = epoch
            break
        last_epoch = epoch

    full.save_pretrained(str(out_dir / "checkpoint"))
    log(
        "done_aborted" if aborted else "done",
        last_epoch=last_epoch,
        items_seen=items_seen,
    )
    metrics.close()
    print(f"v6 {'aborted' if aborted else 'complete'}: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
