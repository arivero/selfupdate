# Issues / Follow-Ups

This file tracks current branch-local engineering work after the refocus to
layerwise forward distillation.

## Active

1. Rebuild teacher hidden-state caches with schema 3 after removing cached
   logit payloads.
2. Add a small smoke test that `scripts/train.py` rejects non-layerwise
   methods with a clear error.
3. Tuned-lens infrastructure: per-layer translator training on the frozen
   base model + translator support in `eval/logit_lens.py` and
   `scripts/logit_lens.py`. Vocabulary stays frozen (see
   `docs/hidden_loss.md`).
4. Drain the frozen eval queues (recite_long for the champion, 14B recite,
   logit_lens/layer_swap backlog) before new training.

## Done (2026-07-03)

- Full test suite re-run after the layerwise-only cleanup: 34/34 pass.
- `runs/results.md`, `runs/curves.png`, `runs/report.pdf` regenerated.
- Scheduler audit: no detached schedulers or GPU jobs live; all queues
  frozen with eval-only entries.

## Research

1. Lens program (Wave I) - see `EXPERIMENTS.md`: tuned-lens depth
   profiles of existing checkpoints, then tuned-lens-CE vs raw lens-CE vs
   tail-CE at matched item budgets.
2. Measure whether the best `tail_ce_blocks` value changes with model depth.
3. Extend `teacher_censored` to larger checkpoints and log per-layer
   increment profiles.
4. Implement streamed block load/offload for the sequential large-model path.
5. Embedding-lens probe on untied checkpoints (8B+), where input and output
   vocabularies genuinely differ.
