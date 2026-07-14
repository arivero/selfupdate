# Individual training report v2

Report v2 has one atomic subject: one training, identified by the complete
tuple

`dataset × model × censorship × loss type`.

The generated report will live at `runs/<run_name>/report.md` after that
training is complete. It contains all information supplied by the collective
v1 report for that training. Cross-run heatmaps and density plots retain their
visual encoding but contain one row. Later collective reports are selections
and aggregations of these individual reports, not a second source of truth.

The campaign-wide live ledger is
`docs/pareto_frontier_training_progress.md`. Epoch-zero teacher controls are written
there as soon as they complete; they do not wait for a training or its local
report.

## Collection contract

Every run must preserve epoch-indexed raw observations. A final checkpoint is
not sufficient evidence for a historical plot. Campaign configs therefore pin
both `eval.every_epochs: 1` and `eval.standard_damage_every_epochs: 1`.

Required observations are:

| series | epoch 0 | every completed epoch | source |
|---|---|---|---|
| recall, separated by corpus | required | required | `metrics.jsonl`, `kind=eval` |
| standard benchmark accuracy and paired damage | required | required | `metrics.jsonl`, `kind=standard_eval` |
| per-layer training loss | not applicable | required | `metrics.jsonl`, `kind=train` |
| per-layer parameter modification from base/epoch 0 | explicit zero row | required | epoch-boundary delta telemetry |
| signal/gradient attribution | reference metadata | required where the loss supplies it | epoch-boundary attribution telemetry |
| elapsed time, items seen, and peak memory | baseline/start | required | telemetry rows |

Epoch numbers mean completed training epochs. The pre-training model is epoch
0; training-loop epoch index 0 is reported as completed epoch 1. Raw rows must
carry the training identity, config hash, dataset identity, pipeline version,
checkpoint/base identity, seed, batching regime, connected-window width,
censorship mode, loss kind, and evaluation source.

Pipeline-v2 identity additionally pins gradient aggregation, trajectory-state
source, attention source, and expert-routing source. Pipeline-v1 runs are
historical and must not be selected into a pipeline-v2 report. Reserved but
unimplemented strategy values are errors, never missing metadata or silent
fallbacks.

Parameter-change collection must be compact epoch-boundary telemetry, not a
requirement to retain a full checkpoint for every epoch. It records, per
layer, at least absolute L2 delta, relative L2 delta, parameter count, and the
aggregation rule. For LoRA, the effective adapter update is measured. For full
training, comparison is against the immutable base weights. Collection runs
outside the hot block walk and must not introduce per-block host
synchronization.

## Local report contents

Each completed training report includes:

- identity, provenance, configuration, timing, placement, and coverage;
- recall by corpus from epoch 0 through the final epoch;
- standard-benchmark damage by epoch and the recall-versus-damage trajectory;
- per-layer loss as a one-row density/heatmap and as temporal layer traces;
- per-layer parameter modification by epoch and its one-row density/heatmap;
- signal attribution required for the scientific claim;
- convergence, final and best-epoch summaries, qualitative evidence, and
  explicit missing-artifact notices.

The generator must show missing observations rather than silently dropping a
section or a training. Report generation happens only after training finishes;
the underlying epoch telemetry is written during training.
