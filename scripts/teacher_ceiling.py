"""Teacher/RAG ceiling: base model recall WITH the RAG passage in context.

The copying ceiling for every student arm: the UNTRAINED model given the
full corpus file as a retrieved document ("Documento recuperado", same
convention as ``masking.render_rag``), scored on the SAME three-task battery
(next/prev/cloze; exact + word_acc, owner directive 2026-07-10) as every
checkpoint and the epoch-0 no-RAG baseline (``evaluate.py --base``). Thin
RAG-context sibling of that baseline: identical corpus resolution and
``tasks_eval`` call, only ``with_context=True`` differs — so a ceiling score
is directly comparable to any other ``tasks.json`` result, never CER against
an incomparable metric.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/teacher_ceiling.py \
        --experiment configs/experiments/lw_r_slide8_0p6b_rag.yaml \
        --n-per-task 24
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfupdate.config import load_config
from selfupdate.eval.tasks import tasks_eval

# Mirrors evaluate.py's CORPUS_PATHS / quijote_rung. Duplicated rather than
# imported: this repo's scripts/ entries each pin their own small helpers
# (see CLAUDE.md) instead of cross-importing another script's module.
CORPUS_PATHS = {
    "machado": "data/poem/raw.txt",
    "quijote_ch1": "data/quijote/raw_ch1.txt",
    "quijote_ch4": "data/quijote/raw_ch4.txt",
    "quijote_ch8": "data/quijote/raw_ch8.txt",
    "quijote_ch16": "data/quijote/raw_ch16.txt",
}


def quijote_rung(path: str | None) -> str | None:
    """'quijote_ch8' from '.../raw_ch8.txt' or '.../examples_ch8.jsonl'."""
    m = re.search(r"ch(\d+)", str(path or "").lower())
    return f"quijote_ch{m.group(1)}" if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--n-per-task", type=int, default=24)
    ap.add_argument("--generation-batch", type=int, default=1,
                    help="batched greedy decode for the battery; the ceiling's "
                         "with_context prompts pay a corpus-length prefill per "
                         "item, so big teachers want 4-8 here")
    ap.add_argument("--context-scope", choices=("full", "window"),
                    default="full",
                    help="ceiling RAG form: full corpus file (historical) or "
                         "per-item exact-match retrieval of the source block "
                         "±--context-window-lines (pair with window-scope v5 "
                         "arms; issues.md 2026-07-12: full-document copying "
                         "is not reliable — report both, never conflate)")
    ap.add_argument("--context-window-lines", type=int, default=4)
    ap.add_argument("--context-pad-random", action="store_true",
                    help="FLOOR variant: replace the retrieved document by a "
                         "seeded random distinct-token fill of matched length "
                         "(the epoch-0 reference paired with pad_random arms)")
    ap.add_argument(
        "--recall-corpora", nargs="+", default=None,
        help="recall corpora to measure. By default this is inferred from "
             "the experiment's data paths, same as evaluate.py --base.")
    ap.add_argument("--auto-map", action="store_true",
                    help="load with device_map=auto for large two-card teacher references")
    # Accepted for CLI compatibility with existing queue rows only: the
    # recite_eval-era batching/scoring knobs have no equivalent in tasks_eval
    # (single-item greedy loop), same as evaluate.py's own --base path.
    # They are IGNORED and warn below (knob-flow law).
    ap.add_argument("--batch-size", type=int, default=None,
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--score-workers", type=int, default=None,
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--bucket-by-length", action="store_true",
                    help="ignored (retired recite/CER engine flag)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    retired = [flag for flag, val, default in (
        ("--batch-size", args.batch_size, None),
        ("--bucket-by-length", args.bucket_by_length, False),
        ("--score-workers", args.score_workers, None),
        ("--shuffle-seed", args.shuffle_seed, None),
    ) if val != default]
    if retired:
        print("WARNING: ignoring " + " ".join(retired) + " — retired with the "
              "recite/CER engine (2026-07-10); the three-task battery "
              "generates item-by-item.", file=sys.stderr)
    cfg = load_config(args.config, args.experiment)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, dtype=torch.bfloat16,
        device_map="auto" if args.auto_map else None)
    if not args.auto_map:
        model.to(cfg.model.device)
    model.eval()

    if args.recall_corpora:
        corpus_names = list(dict.fromkeys(args.recall_corpora))
    else:
        # Same inference evaluate.py --base uses: examples_path is
        # authoritative for combined configs, which intentionally inherit
        # base.yaml's Machado poem_path.
        examples_path = str(cfg.data.examples_path)
        poem_path = str(cfg.data.poem_path)
        if "combined" in examples_path:
            corpus_names = ["machado",
                            quijote_rung(examples_path) or "quijote_ch1"]
        elif "quijote" in examples_path or "quijote" in poem_path:
            corpus_names = [quijote_rung(examples_path)
                            or quijote_rung(poem_path) or "quijote_ch1"]
        else:
            corpus_names = ["machado"]

    corpus_results = {}
    for corpus in corpus_names:
        result = tasks_eval(model, tok, CORPUS_PATHS[corpus],
                            n_per_task=args.n_per_task,
                            with_context=args.context_scope,
                            context_window_lines=args.context_window_lines,
                            context_pad_random=args.context_pad_random,
                            generation_batch=args.generation_batch)
        result["poem_path"] = CORPUS_PATHS[corpus]
        corpus_results[corpus] = result
        parts = "  ".join(
            f"{t}: exact {v['exact']:.2f} words {v['word_acc']:.2f}"
            for t, v in result["tasks"].items())
        print(f"{corpus}: {parts}")

    model_short = cfg.model.name.split("/")[-1]
    kind = ("teacher_epoch0_rag_context" if args.context_scope == "full"
            else "teacher_epoch0_rag_window")
    if args.context_pad_random:
        kind += "_padfloor"
    r = {
        "schema_version": 2,
        "teacher_reference_kind": kind,
        "context_scope": args.context_scope,
        "context_pad_random": args.context_pad_random,
        "model": cfg.model.name,
        "corpora_measured": corpus_names,
        "corpus_selection": ("cli_override" if args.recall_corpora
                             else "inferred_from_training_data"),
        "corpora": corpus_results,
    }
    # training_scope is only honest when inferred from the experiment's own
    # data paths; a --recall-corpora override says what the operator measured.
    if not args.recall_corpora:
        r["training_scope"] = corpus_names
    # One-corpus artifacts retain the v1 surface for downstream compatibility
    # with evaluate.py's tasks.json shape.
    if len(corpus_results) == 1:
        only = next(iter(corpus_results.values()))
        r.update({k: only[k] for k in
                  ("seed", "n_per_task", "tasks", "overall_word_acc", "examples")})
        r["poem_path"] = only["poem_path"]

    data_stem = Path(cfg.data.examples_path).stem
    out = Path(args.out or f"runs/teacher_ref_rag_{model_short}_{data_stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
