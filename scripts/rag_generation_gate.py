"""Certify a RAG teacher before its generated targets can enter a campaign.

The gate has two independent requirements:

1. Completion: the fixed generation budget must reach a natural stop often
   enough.  A model that spends the budget on conversational framing has not
   supplied an answer that can be assessed.
2. Retrieval use: the real retrieved passage must beat both the no-RAG
   epoch-zero model and a same-length random passage on every evaluated
   corpus.  The former establishes that RAG adds useful information; the
   latter rules out a context-length/prompt-format artifact.

The success marker is written only after both checks pass.  Scheduler rows
should depend on that marker, not merely on a ceiling JSON existing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _corpora(doc: dict) -> dict:
    return doc.get("corpora", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", required=True)
    ap.add_argument("--floor", required=True,
                    help="same-scope, same-length random-context control")
    ap.add_argument("--no-rag", required=True,
                    help="epoch-zero evaluation with no retrieved document")
    ap.add_argument("--out", required=True,
                    help="success marker; deliberately absent on failure")
    ap.add_argument("--max-hard-cut-fraction", type=float, default=0.10)
    ap.add_argument("--min-word-lift", type=float, default=0.05)
    ap.add_argument("--min-word-acc", type=float, default=0.10)
    args = ap.parse_args()

    ceiling = json.loads(Path(args.ceiling).read_text())
    floor = json.loads(Path(args.floor).read_text())
    no_rag = json.loads(Path(args.no_rag).read_text())
    checks, failures = [], []
    ccorpora, fcorpora, ncorpora = (_corpora(ceiling), _corpora(floor),
                                    _corpora(no_rag))
    if set(ccorpora) != set(fcorpora) or set(ccorpora) != set(ncorpora) or not ccorpora:
        failures.append("ceiling/random/no-RAG corpus sets differ or are empty")
    for corpus in sorted(set(ccorpora) & set(fcorpora) & set(ncorpora)):
        c, f, n = ccorpora[corpus], fcorpora[corpus], ncorpora[corpus]
        gen = c.get("generation")
        if not gen:
            failures.append(f"{corpus}: no completion telemetry; rerun ceiling")
            continue
        # Bounded-kinds hard-cut, when the ceiling provides it: start_block/
        # end_block ask the model to continue a paragraph with no length
        # bound, and a teacher that never emits EOS there is imitable
        # teacher behavior (the student clones it), not evidence the
        # generation is unassessable — Quijote's long prose paragraphs
        # saturate ANY fixed budget on those two kinds alone (owner
        # directive 2026-07-12: the gate exists to catch a teacher that
        # doesn't access the RAG, not to certify a bounded stop on
        # open-ended continuation). Falls back to the unfiltered fraction
        # for ceilings produced before this field existed.
        hard_cut = float(gen.get("hard_cut_fraction_bounded",
                                 gen["hard_cut_fraction"]))
        score = float(c["overall_word_acc"])
        floor_score = float(f["overall_word_acc"])
        no_rag_score = float(n["overall_word_acc"])
        lift = score - floor_score
        no_rag_lift = score - no_rag_score
        row = {"corpus": corpus, "hard_cut_fraction": hard_cut,
               "word_acc": score, "random_context_word_acc": floor_score,
               "word_lift": lift, "no_rag_word_acc": no_rag_score,
               "no_rag_word_lift": no_rag_lift,
               "completion_pass": hard_cut <= args.max_hard_cut_fraction,
               "retrieval_use_pass": (score >= args.min_word_acc
                                      and lift >= args.min_word_lift
                                      and no_rag_lift >= args.min_word_lift)}
        checks.append(row)
        if not row["completion_pass"]:
            failures.append(f"{corpus}: hard-cut {hard_cut:.1%} exceeds "
                            f"{args.max_hard_cut_fraction:.1%}")
        if not row["retrieval_use_pass"]:
            failures.append(f"{corpus}: random/no-RAG retrieval lifts "
                            f"{lift:.3f}/{no_rag_lift:.3f} or score {score:.3f} "
                            f"below required {args.min_word_lift:.3f}/"
                            f"{args.min_word_acc:.3f}")
    report = {"schema_version": 1, "ceiling": args.ceiling, "floor": args.floor,
              "no_rag": args.no_rag,
              "thresholds": {"max_hard_cut_fraction": args.max_hard_cut_fraction,
                             "min_word_lift": args.min_word_lift,
                             "min_word_acc": args.min_word_acc},
              "checks": checks, "pass": not failures, "failures": failures}
    out = Path(args.out)
    target = out if not failures else out.with_suffix(out.suffix + ".failed.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
