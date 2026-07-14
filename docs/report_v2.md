# Individual training report v2

Report v2 has one atomic subject: one training, identified by the complete
tuple

`dataset × model × censorship × loss type × update geometry × reduction × strategy sources`.

The generated report will live at `runs/<run_name>/report.md` and
`runs/<run_name>/report.pdf` immediately after that training completes. The
Markdown is the navigable/source representation; the PDF is the required
offline-readable rendition of the same individual evidence. Cross-run
heatmaps and density plots retain their visual encoding but contain one row.
Final synthesis is a selection and aggregation layer over these atomic
reports, not a second source of truth.

For sequential browsing, `runs/report_v2_index/` contains one relative symlink
per completed PDF, named `YYYYMMDD-HHMMSS__<run_name>.pdf`. The timestamp is
the stable training-completion time from the `kind=done` telemetry row, so
alphabetical order is completion/publication order and regenerating a report
does not reorder history. Every individual report generation refreshes this
index; it can also be rebuilt with `scripts/refresh_report_v2_index.py`.

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

Reports distinguish nominal geometry from realized geometry. Nominal `B` and
`K` come from the frozen config (`K=all`, not `K=0`); realized telemetry gives
the mean, median, minimum, and maximum active answers and selected valid
aligned-token cells per optimizer update, plus the fraction of full nominal
tiles where that quantity is defined. Ragged tails therefore remain visible
instead of being presented as padding or as complete rectangles. Reports also
carry the explicit `run_class` (`method`, `control`, or another declared
class), epoch-zero recall, and final recall change from epoch zero.

For current Pareto v2, the report must make the strict-local contract visible:
`conn_window: 1`, no behavioral readout/final-logit objective, and—when the
loss is `lens_kl`—a frozen head used only as a local metric with no head update
or cross-block credit.

Pipeline-v2 `RunLog` injects this immutable identity into every telemetry row,
so a partial JSONL extract remains typed and attributable without relying on
directory names or mutable campaign state.

Pipeline-v2 identity additionally pins answer width, aligned-token width,
reduction, mandatory forward layer order, trajectory-state source, attention
source, and expert-routing source. Pipeline-v1 runs are
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

For LoRA, “effective” means the epoch-boundary change in
`scaling × (B @ A)` relative to its epoch-zero value, aggregated against the
Frobenius norm of the corresponding frozen base matrices. The collector uses
rank-sized Gram products and does not materialize dense adapter deltas. For
full training, it streams immutable epoch-zero parameter references from host
RAM one tensor at a time.

Grid rows distinguish completed source answers from repeated answer visits and
record the exact selected aligned range for each example. They carry selected
answer-token cells, selected loss cells after expansion over layers, and full
causal sequence-token/layer cells. This keeps scientific coverage separate
from repeated compute when `K` is narrower than a complete answer.

Training-loss rows retain two explicit layerwise measures: the equal-answer
mean and the valid-token-weighted mean. The row's `loss_measure` identifies
the measure used by that optimizer regime (`answer_mean` or
`valid_token_mean`); plots and cross-report synthesis select like-for-like
measures rather than silently comparing different reductions.

## Local report contents

Each completed training report includes, in both its Markdown source and
required PDF rendition:

- identity, provenance, configuration, timing, placement, and coverage;
- recall by corpus from epoch 0 through the final epoch;
- standard-benchmark damage by epoch and the recall-versus-damage trajectory;
- per-layer loss as a one-row density/heatmap and as temporal layer traces;
- per-layer parameter modification by epoch and its one-row density/heatmap;
- signal attribution required for the scientific claim;
- convergence, final and best-epoch summaries, qualitative evidence, and
  explicit missing-artifact notices.

The generator must show missing observations rather than silently dropping a
section or a training. Per-run report generation happens immediately after
training finishes; the completion condition is the published Markdown, PDF,
assets, and manifest. The underlying epoch telemetry is written during training.
Readout-bearing historical diagnostics are labeled and excluded from strict
block-local frontier synthesis.

## Final synthesis groupings

The final synthesis may be emitted at campaign level or as like-for-like
groupings by:

- model;
- loss type;
- censorship mode; and
- update geometry (including reduction and explicit B/K widths).
- declared run class.

Every grouping records its inclusion rule and excludes superseded historical
readout-bearing arms from frontier claims. Missing artifacts remain visible in
the relevant group rather than being silently dropped.

Generate one completed report with:

```bash
scripts/l40s_exec.sh scripts/report_v2.py <run_name>
```

`--allow-incomplete` exists only for diagnostic rendering and labels the
result incomplete; campaign reports are generated after the checkpoint and
`kind=done` telemetry row both exist.
