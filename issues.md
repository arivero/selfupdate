# Issues / Follow-Ups

This file tracks current branch-local engineering work after the refocus to
layerwise forward distillation.

## Active

1. Rebuild teacher hidden-state caches with schema 3 after removing cached
   logit payloads.
2. Run the full test suite on the L40S venv after the layerwise-only cleanup.
3. Regenerate `runs/results.md`, `runs/curves.png`, and `runs/report.pdf`
   from current layerwise artifacts.
4. Audit active detached schedulers before launching new work; queues now
   reference layerwise jobs only.
5. Add a small smoke test that `scripts/train.py` rejects non-layerwise
   methods with a clear error.

## Research

1. Finish lens-CE and tail-CE comparisons at matched item budgets.
2. Measure whether the best `tail_ce_blocks` value changes with model depth.
3. Extend `teacher_censored` to larger checkpoints and log per-layer
   increment profiles.
4. Implement streamed block load/offload for the sequential large-model path.
