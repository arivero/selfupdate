"""Fast retention + recall battery for the final cross-checkout report.

Motivation (owner directives, 2026-07-07): the tiny held-out general-CE
"destruction" canary was measured to be dominated by noise and nearly
constant in signal across training.  We replace it with two families of
*standard, bounded, low-variance* measurements, all scored in a single
batched forward pass (teacher-forced; no autoregressive generation), so the
only meaningful per-checkpoint cost is loading the model:

  DESTRUCTION (capability preserved vs the epoch-0 teacher of the same base):
    - arc_easy       : ARC-Easy multiple-choice accuracy on a FIXED cached
                       subset, option-log-likelihood argmax.
    - wikitext2_ppl  : WikiText-2 validation perplexity (a quantization-style
                       language-model-damage metric; note ppl == exp(CE)).

  RECALL (memorization of the corpus, an independent axis to recite-CER):
    - recall_cont_<corpus>  : exact continuation.  Given a prefix ending at a
                              line boundary, is the model's greedy (argmax)
                              continuation token-exact over the whole reference span?
    - recall_cloze_<corpus> : exact fill of a deleted interior word.  Given the
                              left context up to a removed content word, does
                              argmax reproduce the word token-exact?

  where <corpus> in {machado, cervantes}.  Scoring the model against the
  original poem/prose is EVAL (recall is defined as reproducing the reference),
  which the branch's training-target law explicitly permits; nothing here is a
  training signal.

Design for the "a lot lot of small models" regime (161 final checkpoints,
114 full fine-tunes + 47 LoRA):
  * The probe sets are cached ONCE, model-independent, so no per-checkpoint
    dataset download or re-tokenization (the old standard_destruction_eval did
    both on every call).  The subset is identical across all checkpoints ->
    comparable.
  * One resident process evaluates many checkpoints.  Checkpoints are grouped
    by base model; for LoRA checkpoints the base is loaded ONCE and adapters
    are hot-swapped (load_adapter / set_adapter / delete_adapter) instead of
    reloading the whole model.  The epoch-0 teacher (base) is scored once per
    base model and cached in-process for the retained-ratio.
  * Each model stays GPU-resident across ALL tasks in one residency.

The vocab-size logits tensor ([B, T, V]) forces length-sorted micro-batching
even for "one batch": a literal 1600-row ARC batch would need tens of GB of
logits for a Qwen-class vocab.  Micro-batching keeps it bounded; loading still
dominates wall-clock, which is why plain torch (not vLLM) is the right tool
here -- vLLM would not reduce the load and adds engine-startup overhead.

Usage:
  # build the fixed probe caches once (CPU only, uses many workers)
  python compressed/retention_eval.py --build-cache

  # evaluate one or more checkpoints (paths may live in sibling checkouts)
  python compressed/retention_eval.py --checkpoints runs/<run>/checkpoint ...
  python compressed/retention_eval.py --glob 'runs/*/checkpoint'
  python compressed/retention_eval.py --manifest runs/retention_manifest.tsv
"""

from __future__ import annotations


# BEGIN GENERATED SHARED SELFUPDATE BOOTSTRAP
from _shared_bundle import archive as _su_archive, install as _su_install
_su_install()
_SU_ARCHIVE = _su_archive()
# END GENERATED SHARED SELFUPDATE BOOTSTRAP



import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent

import torch
import torch.nn.functional as F
import yaml

from selfupdate.eval.standard import BENCHMARK_REVISIONS

CACHE_DIR = REPO / "data" / "eval"
CACHE_VERSION = "v1"

DESTRUCTION_TASKS = ("arc_easy", "wikitext2_ppl")
RECALL_CORPORA = {
    "machado": REPO / "data" / "poem" / "raw.txt",
    "cervantes": REPO / "data" / "quijote" / "raw_ch1.txt",
}
# Per-corpus probe scheme.  Verse (Machado) gets whole-line continuation and
# single-word cloze; prose (Cervantes) gets next-sentence continuation and a
# multi-word "censor" fill (several interior content words blanked from each
# large paragraph, scored as fraction of censored words recovered).  Each entry
# is (scheme_kind, task_name); all three kinds are lists of {prompt, reference} and
# share one teacher-forced exact-match scorer.
RECALL_SCHEME = {
    "machado": [("cont", "recall_cont_machado"), ("cloze", "recall_cloze_machado")],
    "cervantes": [("cont", "recall_cont_cervantes"), ("censor", "recall_censor_cervantes")],
}
RECALL_TASKS = tuple(name for kinds in RECALL_SCHEME.values() for _, name in kinds)
# task_name -> (corpus, scheme_kind)
RECALL_TASK_MAP = {
    name: (corpus, kind)
    for corpus, kinds in RECALL_SCHEME.items()
    for kind, name in kinds
}
ALL_TASKS = DESTRUCTION_TASKS + RECALL_TASKS


# --------------------------------------------------------------------------- #
# Cache construction (model-independent, run once).
# --------------------------------------------------------------------------- #
def _arc_cache_path() -> Path:
    return CACHE_DIR / f"arc_easy_{CACHE_VERSION}.json"


def _wikitext_cache_path() -> Path:
    return CACHE_DIR / f"wikitext2_val_{CACHE_VERSION}.txt"


def _recall_cache_path(corpus: str) -> Path:
    return CACHE_DIR / f"recall_{corpus}_{CACHE_VERSION}.json"


def _build_arc_cache(n_items: int) -> None:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation",
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
        rows.append(
            {
                "id": row.get("id"),
                "prompt": f"Question: {row['question'].strip()}\nAnswer:",
                "choices": [f" {t.strip()}" for t in texts],
                "target": target,
            }
        )
        if len(rows) >= n_items:
            break
    payload = {
        "task": "arc_easy",
        "source": "allenai/ai2_arc:ARC-Easy:validation",
        "n": len(rows),
        "subset_id": _subset_id([r["id"] for r in rows]),
        "items": rows,
    }
    _arc_cache_path().write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"arc cache: {len(rows)} items -> {_arc_cache_path()}")


def _build_wikitext_cache(max_chars: int) -> None:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="validation",
                      revision=BENCHMARK_REVISIONS["Salesforce/wikitext"])
    parts, chars = [], 0
    for row in ds:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        chars += len(text) + 2
        if chars >= max_chars:
            break
    _wikitext_cache_path().write_text("\n\n".join(parts))
    print(f"wikitext cache: {chars} chars -> {_wikitext_cache_path()}")


_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}")
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _verse_cont_cloze(lines: list[str], n_cont: int, n_cloze: int):
    """Machado: whole-line continuation + single interior-word cloze."""
    cont = []
    for i in _even_indices(len(lines) - 1, n_cont, lo=3):
        prefix = "\n".join(lines[max(0, i - 6):i + 1]) + "\n"
        reference = lines[i + 1]
        if reference:
            cont.append({"prompt": prefix, "reference": reference})
    cloze = []
    for i in _even_indices(len(lines), n_cloze, lo=3):
        line = lines[i]
        words = list(_WORD_RE.finditer(line))
        interior = words[1:-1] if len(words) >= 3 else []
        if not interior:
            continue
        m = interior[len(interior) // 2]
        prompt = "\n".join(lines[max(0, i - 6):i]) + ("\n" if i > 0 else "") + line[: m.start()]
        cloze.append({"prompt": prompt, "reference": line[m.start():m.end()]})
    return {"cont": cont, "cloze": cloze}


def _prose_cont_censor(paras: list[str], n_cont: int, words_per_para: int):
    """Cervantes: next-sentence continuation + multi-word censor per paragraph.

    censor: from each large paragraph, blank several interior content words and
    ask the model to recover each from its left context (teacher-forced: earlier
    words are the true text).  Every blanked word is one exact-match probe, so
    the task accuracy is the fraction of censored words recovered."""
    cont = []
    for i in _even_indices(len(paras) - 1, n_cont, lo=1):
        nxt = _SENT_END.split(paras[i + 1].strip())
        reference = nxt[0].strip() if nxt else ""
        if len(reference) >= 8:
            prompt = "\n\n".join(paras[max(0, i - 1):i + 1]) + "\n\n"
            cont.append({"prompt": prompt, "reference": reference})

    censor = []
    for i, para in enumerate(paras):
        if len(para) < 120:
            continue
        words = list(_WORD_RE.finditer(para))
        interior = words[3:-1]  # leave head context and the final word
        if len(interior) < 2:
            continue
        pick = [interior[j] for j in _even_indices(len(interior), words_per_para)]
        head = "\n\n".join(paras[max(0, i - 1):i]) + ("\n\n" if i > 0 else "")
        for m in pick:
            censor.append({"prompt": head + para[: m.start()], "reference": para[m.start():m.end()]})
    return {"cont": cont, "censor": censor}


def _build_recall_cache(corpus: str, n_cont: int, n_cloze: int) -> None:
    """Build fixed, deterministic recall probes (stable across checkpoints)."""
    path = RECALL_CORPORA[corpus]
    raw = path.read_text(encoding="utf-8")
    if corpus == "machado":
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if len(ln) >= 12 and not ln.startswith("#")]
        kinds = _verse_cont_cloze(lines, n_cont, n_cloze)
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if len(p.strip()) >= 40]
        kinds = _prose_cont_censor(paras, n_cont, words_per_para=8)

    all_probes = [p for lst in kinds.values() for p in lst]
    payload = {
        "corpus": corpus,
        "source": str(path.relative_to(REPO)),
        "kinds": kinds,
        "subset_id": _subset_id([p["prompt"][-24:] + "|" + p["reference"] for p in all_probes]),
    }
    _recall_cache_path(corpus).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    summary = " + ".join(f"{len(v)} {k}" for k, v in kinds.items())
    print(f"recall[{corpus}] cache: {summary} -> {_recall_cache_path(corpus)}")


def _even_indices(upper: int, count: int, lo: int = 0) -> list[int]:
    if upper <= lo:
        return []
    count = min(count, upper - lo)
    if count <= 0:
        return []
    step = (upper - lo) / count
    return sorted({int(lo + k * step) for k in range(count)})


def _subset_id(keys: list) -> str:
    h = hashlib.sha1("|".join(str(k) for k in keys).encode("utf-8")).hexdigest()
    return h[:12]


def build_cache(args) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # The version-pinned cache files are git-tracked corpus artifacts (the
    # "pinned wikitext file" IS the cache). Rebuilding silently overwrote
    # them with whatever the Hub served that day — refuse unless forced,
    # and remember completed-checkpoint resumability keys on CACHE_VERSION.
    existing = [p for p in (_arc_cache_path(), _wikitext_cache_path(),
                            *(_recall_cache_path(c) for c in RECALL_CORPORA))
                if p.exists()]
    if existing and not args.force_cache:
        raise SystemExit(
            "retention caches are git-pinned corpus artifacts; refusing to "
            "overwrite: " + ", ".join(str(p) for p in existing)
            + " — pass --force-cache to rebuild (and bump CACHE_VERSION if "
            "content changes)")
    _build_arc_cache(args.arc_items)
    _build_wikitext_cache(args.wikitext_max_chars)
    for corpus in RECALL_CORPORA:
        _build_recall_cache(corpus, args.recall_cont, args.recall_cloze)


# --------------------------------------------------------------------------- #
# Scoring primitives (teacher-forced, length-sorted micro-batched).
# --------------------------------------------------------------------------- #
def _chunks(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _reference_span(prompt_ids: list[int], full_ids: list[int]) -> int:
    """Token index where the reference continuation begins, by common prefix.

    Re-encoding the prompt alone is unreliable: BPE merges a trailing space
    (or newline) with the first reference word, so len(encode(prompt)) can land
    inside or past the reference word.  The common prefix of encode(prompt) and
    encode(prompt+reference) is the correct, tokenizer-agnostic boundary (the
    lm-eval-harness convention)."""
    n = 0
    m = min(len(prompt_ids), len(full_ids))
    while n < m and prompt_ids[n] == full_ids[n]:
        n += 1
    return n


def _batched_by_tokens(items: list[dict], token_budget: int, max_rows: int):
    """Yield length-sorted batches bounded by total padded token-positions,
    which is what caps the [B, T, V] logits tensor."""
    order = sorted(range(len(items)), key=lambda i: items[i]["_len"])
    batch, longest = [], 0
    for idx in order:
        L = items[idx]["_len"]
        new_longest = max(longest, L)
        if batch and (new_longest * (len(batch) + 1) > token_budget or len(batch) >= max_rows):
            yield batch
            batch, longest = [], 0
            new_longest = L
        batch.append(idx)
        longest = new_longest
    if batch:
        yield batch


@torch.no_grad()
def _teacher_forced_exact(model, tok, probes: list[dict], device: str,
                          token_budget: int, max_rows: int) -> float:
    """Fraction of probes whose reference span is reproduced token-exact by argmax.

    Each probe: {'prompt', 'reference'}.  We build ids = prompt + ' '+reference? No --
    reference spacing is baked into the probe.  We locate the reference token span by the
    length of prompt ids vs full ids, then compare argmax(logits) to reference ids
    over that span.  All reference tokens must match for the probe to count."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prepared = []
    for p in probes:
        prompt_ids = tok.encode(p["prompt"], add_special_tokens=False)
        full_ids = tok.encode(p["prompt"] + p["reference"], add_special_tokens=False)
        start = _reference_span(prompt_ids, full_ids)
        if start < 1 or start >= len(full_ids):
            continue  # need >=1 context token and a non-empty reference span
        prepared.append({"ids": full_ids, "start": start, "_len": len(full_ids)})
    if not prepared:
        return float("nan")

    correct = 0
    for batch_idx in _batched_by_tokens(prepared, token_budget, max_rows):
        rows = [prepared[i] for i in batch_idx]
        max_len = max(r["_len"] for r in rows)
        x = torch.full((len(rows), max_len), pad, dtype=torch.long, device=device)
        attn = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
        for j, r in enumerate(rows):
            x[j, :r["_len"]] = torch.tensor(r["ids"], dtype=torch.long, device=device)
            attn[j, :r["_len"]] = 1
        logits = model(x, attention_mask=attn, use_cache=False).logits
        for j, r in enumerate(rows):
            s, e = r["start"], r["_len"]
            pred = logits[j, s - 1:e - 1].argmax(dim=-1)
            reference = x[j, s:e]
            if torch.equal(pred, reference):
                correct += 1
    return correct / len(prepared)


@torch.no_grad()
def _score_arc(model, tok, items: list[dict], device: str,
               token_budget: int, max_rows: int) -> dict:
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    flat = []
    for ex_i, ex in enumerate(items):
        p_ids = tok.encode(ex["prompt"], add_special_tokens=False)
        for ch_i, choice in enumerate(ex["choices"]):
            full = tok.encode(ex["prompt"] + choice, add_special_tokens=False)
            start = _reference_span(p_ids, full)
            if start < 1:
                start = min(len(p_ids), len(full) - 1)
            flat.append({"ids": full, "start": start, "_len": len(full),
                         "ex": ex_i, "ch": ch_i})
    scores = [[-math.inf] * len(ex["choices"]) for ex in items]
    for batch_idx in _batched_by_tokens(flat, token_budget, max_rows):
        rows = [flat[i] for i in batch_idx]
        max_len = max(r["_len"] for r in rows)
        x = torch.full((len(rows), max_len), pad, dtype=torch.long, device=device)
        attn = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
        for j, r in enumerate(rows):
            x[j, :r["_len"]] = torch.tensor(r["ids"], dtype=torch.long, device=device)
            attn[j, :r["_len"]] = 1
        logits = model(x, attention_mask=attn, use_cache=False).logits
        for j, r in enumerate(rows):
            s, e = r["start"], r["_len"]
            if e <= s:
                continue
            row_logits = logits[j, s - 1:e - 1].float()
            targets = x[j, s:e]
            nll = F.cross_entropy(row_logits, targets, reduction="sum").item()
            scores[r["ex"]][r["ch"]] = -nll / max(1, e - s)
    correct = sum(
        int(max(range(len(sc)), key=lambda k: sc[k]) == ex["target"])
        for ex, sc in zip(items, scores)
    )
    return {"n": len(items), "accuracy": correct / max(1, len(items))}


@torch.no_grad()
def _score_wikitext(model, tok, text: str, device: str,
                    seq_len: int, batch_rows: int) -> dict:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return {"n_tokens": 0, "mean_ce": float("nan"), "ppl": float("nan")}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    usable = max(seq_len, (len(ids) - 1) // seq_len * seq_len)
    starts = list(range(0, usable, seq_len))
    total_nll, total_tok = 0.0, 0
    for chunk in _chunks(starts, batch_rows):
        seqs = [ids[s:s + seq_len + 1] for s in chunk]
        lengths = [len(s) - 1 for s in seqs]
        max_len = max(len(s) for s in seqs)
        x = torch.full((len(seqs), max_len), pad, dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[i, :len(s)] = 1
        logits = model(x, attention_mask=attn, use_cache=False).logits
        for i, n in enumerate(lengths):
            if n <= 0:
                continue
            row_logits = logits[i, :n].float()
            targets = x[i, 1:n + 1]
            total_nll += F.cross_entropy(row_logits, targets, reduction="sum").item()
            total_tok += n
    mean_ce = total_nll / max(1, total_tok)
    return {"n_tokens": total_tok, "mean_ce": mean_ce, "ppl": math.exp(min(50, mean_ce))}


# --------------------------------------------------------------------------- #
# Full battery on one already-loaded model.
# --------------------------------------------------------------------------- #
class Probes:
    """Loaded once per process; reused for every checkpoint."""

    def __init__(self):
        self.arc = json.loads(_arc_cache_path().read_text())
        self.wikitext = _wikitext_cache_path().read_text()
        self.recall = {c: json.loads(_recall_cache_path(c).read_text()) for c in RECALL_CORPORA}


def evaluate_model(model, tok, probes: Probes, tasks, device, args) -> dict:
    out = {}
    for task in tasks:
        if task == "arc_easy":
            out[task] = _score_arc(model, tok, probes.arc["items"], device,
                                   args.token_budget, args.max_rows)
        elif task == "wikitext2_ppl":
            out[task] = _score_wikitext(model, tok, probes.wikitext, device,
                                        args.wikitext_seq_len, args.wikitext_rows)
        elif task in RECALL_TASK_MAP:
            corpus, kind = RECALL_TASK_MAP[task]
            plist = probes.recall[corpus]["kinds"].get(kind, [])
            acc = _teacher_forced_exact(model, tok, plist, device,
                                        args.token_budget, args.max_rows)
            out[task] = {"n": len(plist), "exact_acc": acc}
    return out


# --------------------------------------------------------------------------- #
# Checkpoint discovery, grouping, and load-amortized driving.
# --------------------------------------------------------------------------- #
def _run_config(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if p.exists():
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _base_model_of(ckpt: Path) -> str | None:
    cfg = _run_config(ckpt.parent)
    return ((cfg.get("model") or {}).get("name")) or None


def _is_lora(ckpt: Path) -> bool:
    return (ckpt / "adapter_config.json").exists()


def _discover(args) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoints:
        paths += [Path(p) for p in args.checkpoints]
    for pattern in args.glob or []:
        paths += [Path(p) for p in sorted(Path().glob(pattern))]
    if args.manifest:
        for line in Path(args.manifest).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line.split("\t")[0]))
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        # config.yaml lives in the run dir (checkpoint's parent), or p itself
        # is the run dir.
        if (p.parent / "config.yaml").exists() or (p / "config.yaml").exists():
            seen.add(rp)
            uniq.append(p)
    return uniq


def _out_path(ckpt: Path) -> Path:
    return ckpt.parent / "eval" / "retention.json"


def _load_full(src: str, device: str, auto_map: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, device_map="auto" if auto_map else None
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if not auto_map:
        model.to(device)
    model.eval()
    return model, tok


def _free(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _already_done(ckpt: Path) -> bool:
    """True if a current-version retention.json already exists (resumable)."""
    out = _out_path(ckpt)
    if not out.exists():
        return False
    try:
        return json.loads(out.read_text()).get("cache_version") == CACHE_VERSION
    except Exception:  # noqa: BLE001
        return False


def run(args) -> None:
    probes = Probes()
    tasks = args.tasks
    ckpts = _discover(args)
    if not ckpts:
        raise SystemExit("no checkpoints discovered (use --checkpoints/--glob/--manifest)")
    if args.skip_existing:
        before = len(ckpts)
        ckpts = [c for c in ckpts if not _already_done(c)]
        print(f"skip-existing: {before - len(ckpts)} already done, {len(ckpts)} to do")
    if not ckpts:
        print("nothing to do")
        return

    # Group by base model so LoRA siblings share one base load.
    groups: dict[str, dict[str, list[Path]]] = {}
    for c in ckpts:
        base = _base_model_of(c) or "UNKNOWN"
        g = groups.setdefault(base, {"lora": [], "full": []})
        g["lora" if _is_lora(c) else "full"].append(c)

    teacher_cache: dict[str, dict] = {}
    timings, failures = [], []

    for base, g in groups.items():
        print(f"\n=== base model: {base}  (lora={len(g['lora'])} full={len(g['full'])}) ===",
              flush=True)

        # --- LoRA siblings: load base once, hot-swap adapters ---
        if g["lora"] and base != "UNKNOWN":
            try:
                t0 = time.time()
                model, tok = _load_full(base, args.device, args.auto_map)
                load_s = time.time() - t0
                teacher_cache[base] = _score_teacher(model, tok, probes, tasks, args)
            except Exception as e:  # noqa: BLE001
                print(f"  !! base load failed for {base}: {type(e).__name__}: {e}", flush=True)
                failures += [(c.parent.name, str(e)) for c in g["lora"]]
                model = None
            if model is not None:
                from peft import PeftModel

                peft_model, first = None, g["lora"][0]
                for c in g["lora"]:
                    try:
                        t1 = time.time()
                        if peft_model is None:
                            peft_model = PeftModel.from_pretrained(model, str(c), adapter_name="cur")
                        else:
                            peft_model.load_adapter(str(c), adapter_name="cur")
                            peft_model.set_adapter("cur")
                        res = evaluate_model(peft_model, tok, probes, tasks, args.device, args)
                        _write_result(c, base, res, teacher_cache[base], is_lora=True,
                                      eval_s=time.time() - t1, load_s=load_s if c is first else 0.0)
                        timings.append((c.parent.name, load_s if c is first else 0.0,
                                        time.time() - t1))
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! {c.parent.name} failed: {type(e).__name__}: {e}", flush=True)
                        failures.append((c.parent.name, str(e)))
                    finally:
                        if peft_model is not None and "cur" in getattr(peft_model, "peft_config", {}):
                            peft_model.delete_adapter("cur")
                _free(peft_model if peft_model is not None else model)

        # --- Full fine-tunes: one load each (weights are not shared) ---
        for c in g["full"]:
            try:
                t0 = time.time()
                model, tok = _load_full(str(c), args.device, args.auto_map)
                load_s = time.time() - t0
                if base not in teacher_cache and base != "UNKNOWN":
                    tb, tt = _load_full(base, args.device, args.auto_map)
                    teacher_cache[base] = _score_teacher(tb, tt, probes, tasks, args)
                    _free(tb)
                t1 = time.time()
                res = evaluate_model(model, tok, probes, tasks, args.device, args)
                _write_result(c, base, res, teacher_cache.get(base), is_lora=False,
                              eval_s=time.time() - t1, load_s=load_s)
                timings.append((c.parent.name, load_s, time.time() - t1))
                _free(model)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {c.parent.name} failed: {type(e).__name__}: {e}", flush=True)
                failures.append((c.parent.name, str(e)))
                try:
                    _free(model)
                except Exception:  # noqa: BLE001
                    pass

    _print_timings(timings)
    if failures:
        print(f"\n=== {len(failures)} failures ===")
        for name, err in failures:
            print(f"  {name}: {err[:120]}")


def _score_teacher(model, tok, probes, tasks, args) -> dict:
    return evaluate_model(model, tok, probes, tasks, args.device, args)


def _retained(res: dict, teacher: dict | None) -> dict:
    if not teacher:
        return {}
    out = {}
    a = res.get("arc_easy", {}).get("accuracy")
    ta = teacher.get("arc_easy", {}).get("accuracy")
    if a is not None and ta:
        out["arc_retained"] = a / ta
    w = res.get("wikitext2_ppl", {}).get("ppl")
    tw = teacher.get("wikitext2_ppl", {}).get("ppl")
    if w and tw:
        out["wikitext_ppl_ratio"] = w / tw
    return out


def _write_result(ckpt: Path, base: str, res: dict, teacher: dict | None,
                  is_lora: bool, eval_s: float, load_s: float) -> None:
    payload = {
        "kind": "retention_eval",
        "cache_version": CACHE_VERSION,
        "run": ckpt.parent.name,
        "checkpoint": str(ckpt),
        "base_model": base,
        "is_lora": is_lora,
        "tasks": res,
        "teacher": teacher,
        "retained": _retained(res, teacher),
        "timing_s": {"load": round(load_s, 2), "eval": round(eval_s, 2)},
    }
    out = _out_path(ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    arc = res.get("arc_easy", {}).get("accuracy")
    cont_m = res.get("recall_cont_machado", {}).get("exact_acc")
    print(f"  {ckpt.parent.name}: arc={_f(arc)} cont_machado={_f(cont_m)} "
          f"load={load_s:.1f}s eval={eval_s:.1f}s -> {out.name}", flush=True)


def _f(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _print_timings(timings: list) -> None:
    if not timings:
        return
    load = sum(t[1] for t in timings)
    eval_total = sum(t[2] for t in timings)
    print(f"\n=== timing summary over {len(timings)} checkpoints ===")
    print(f"  total load  {load:8.1f}s")
    print(f"  total eval  {eval_total:8.1f}s")
    print(f"  load/eval ratio {load / max(1e-6, eval_total):.1f}x "
          f"(loading dominates -> torch is load-bound, not compute-bound)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--force-cache", action="store_true",
                    help="allow --build-cache to overwrite the git-pinned "
                         "cache artifacts (bump CACHE_VERSION on content change)")
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--glob", nargs="*", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(ALL_TASKS), choices=ALL_TASKS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-map", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip checkpoints that already have a current-version retention.json")
    # cache sizes
    ap.add_argument("--arc-items", type=int, default=400)
    ap.add_argument("--wikitext-max-chars", type=int, default=250_000)
    ap.add_argument("--recall-cont", type=int, default=200)
    ap.add_argument("--recall-cloze", type=int, default=200)
    # batching (bounds the [B,T,V] logits tensor)
    ap.add_argument("--token-budget", type=int, default=8192,
                    help="max padded token-positions per micro-batch")
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--wikitext-seq-len", type=int, default=2048)
    ap.add_argument("--wikitext-rows", type=int, default=4)
    args = ap.parse_args()

    if args.build_cache:
        build_cache(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
