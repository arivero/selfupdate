# Training pipeline v2

Pipeline v2 is the training runtime for the dataset-v5 Pareto-frontier
campaign. Pipeline-v1 checkpoints remain historical mechanics evidence but
are not mixed into v2 reports or comparisons.

The current v2 contract is strict block-local training: `conn_window: 1`,
detached block inputs, and no behavioral readout or final-logit objective. A
local `lens_kl` loss may evaluate distributions through the frozen norm and
vocabulary head, but it cannot update that head or propagate across blocks.
The former readout runtime is being deleted and is recoverable from Git only.

## Typed training identity

A training is identified by

`dataset × model × censorship × loss × answer width × token width × reduction × trajectory source × attention source × expert routing`.

The first campaign varies censorship (`remove`, `pad_random`) and loss
(`huber`, `lens_kl`); its geometry gate varies explicit B/K tiles and
reduction before selecting recipes for the six-model expansion. The other
strategy axes are pinned and recorded now so later experiments do not change
the meaning of a run identity.

| config field | implemented v2 value(s) | reserved value(s) |
|---|---|---|
| `train.update_granularity` | `grid` (new experiments); `answer`, `token` compatibility aliases | — |
| `train.answers_per_update` | positive logical/physical answer width | — |
| `train.tokens_per_answer_update` | positive aligned-token width; `0` means all | — |
| `train.update_reduction` | `answer_mean`, `token_mean` | — |
| `train.trajectory_source` | `student_hidden` | `teacher_hidden` |
| `train.attention_source` | `student_attention` | `teacher_attention` |
| `train.expert_routing_source` | `black_box` | `teacher_routing_cache` |

Reserved values are intentionally present in the design but dispatch rejects
them until implemented and certified. A future switch must never be parsed and
ignored.

## Three-dimensional optimizer grid

The training measurements form a grid

`answer × aligned token × layer`.

Only the first two axes define an optimizer tile. The layer axis is ordered,
not exchangeable: every tile walks `L=1..n` forward, block `L` consumes the
student state just produced by `L-1`, and the optimizer steps only after that
complete layer walk. Absolute-state losses compare student `h[L]` with cached
teacher `h[L]`; delta losses additionally consume cached teacher `h[L-1]`.
This preserves precisely the information available from the previous layer.

A token tile does **not** shorten model context. Every selected token keeps
its complete causal prefix and every selected answer retains its complete
right-padded sequence. Tiling changes which aligned rows contribute loss and
gradient to the update; it does not turn causal language modeling into
isolated-token inference.

The explicit geometry fields are:

- `answers_per_update = B`: maximum selected answers in a tile;
- `tokens_per_answer_update = K`: the next `K` aligned rows of each active
  answer; `0` means every valid aligned row in one tile;
- `update_reduction`: the independent statistical measure used to combine
  selected cells.

For a fixed loader batch, token cursors advance `0:K`, `K:2K`, and so on.
Short answers leave later tiles; exact per-answer ranges are logged, so a
ragged tail is visible rather than described as a full rectangle. Each answer
is counted complete exactly once, at its final tile. `answer_visits` separately
counts repeated participation across token tiles. Finished answers are not
replaced inside that source batch: replacement would change update membership
and the experiment's grouping semantics. Missing tail cells contribute
neither to the loss numerator nor to its reduction denominator.

### Independent reduction

- `answer_mean`: mean the selected tokens within each answer, then give each
  active answer equal weight.
- `token_mean`: weight each per-answer mean by its selected valid-token count,
  yielding the mean over all selected answer-token cells. Padding contributes
  nothing. Any local vocabulary metric uses only its own valid aligned rows.

Geometry and reduction are not aliases. For example, `20 × 16` can be run
with either answer-mean or token-mean reduction; the difference is observable
on ragged tail tiles.

The current implementation requires `batching: padded|bucketed`,
`micro_batch == answers_per_update`, and `grad_accum: 1`; one physical grid
tile is one optimizer update. A future logical tile split across several
physical forward micro-batches needs exact global reduction denominators and
will receive a separate implementation/certification rather than silently
changing this contract.

### Compatibility regimes

`answer` remains the certified B=1/all-token compatibility path.
`token` remains the certified full-padded-minibatch compatibility path. The
four Qwen3.5 4B trainings launched on 2026-07-14 are accurately described as
`B=8 × K=all, token_mean`, but they used the then-present readout-bearing
runtime and are now superseded historical diagnostics, not frontier evidence.
New experiment configs use `grid`, pin every geometry field explicitly, and
carry no `readout_*` keys.

The committed 4B campaign screen is generated by
`scripts/gen_pareto_v2_screen.py` and audited as ordinary explicit YAML under
`configs/experiments/pareto_v2/screen_4b/`. Run the generator with `--check`
to verify that its 40 configs and queue have not drifted.

### Historical aggregation

`legacy_answer_sum` reproduces pipeline v1: sum per-answer token means and
accumulate them across answers. It remains the default only so historical
configs retain their meaning. Pipeline v2 rejects it.

Matched geometry comparisons pin learning rate, AdamW, per-block clip norm,
example order, epochs, source answers, valid selected-token budget,
censorship, loss, and reduction. Optimizer-step count is deliberately allowed
to differ because step grouping is the hypothesis. Any learning-rate scaling
is a separately named follow-up.

## Future strategy axes

`student_hidden` means a block/window starts from the student's detached
trajectory. `teacher_hidden` will root it at the corresponding cached teacher
state. This generalizes the existing scientific distinction without
overloading schedule names.

`student_attention` means the student computes attention normally.
`teacher_attention` will require a versioned cache representation specifying
whether it replays attention outputs, probabilities, or routing-like discrete
choices. No implementation is assumed until that representation is fixed.

`black_box` treats a sparse-expert block as its ordinary combined function.
`teacher_routing_cache` will consume versioned teacher router choices and must
report top-k agreement and cache provenance.

## Telemetry and report requirements

Every grid train row records pipeline version, all strategy axes, optimizer
step, exact example IDs and aligned-coordinate ranges, cumulative completed
answers, answer visits, aligned tokens, and loss-measurement cells. It also
records full causal sequence tokens and both selected/full-causal counts after
expansion over the layer axis, the layer interval, `forward` order, and the
student/teacher trajectory dependency. It retains both the per-answer mean
and valid-token-weighted mean for every layer and marks which measure drove
the configured update; report synthesis must not compare unlike measures.
Epoch rows additionally carry recall,
standard damage, time, memory, and per-layer parameter modification from the
epoch-zero/base reference. See `docs/report_v2.md`.

The individual report is generated immediately after each run completes and
is the atomic source of truth. Final synthesis is a separate selection layer;
it supports campaign-wide and grouped views by model, loss, censorship, and
update geometry, with like-for-like reductions and strict-local objectives.

## Speed and quality gate

Before expanding to all six models, Qwen3.5 4B runs a matched geometry gate.
The first full-minibatch distribution probe measured `B=8,K=all` at 27.23
answers/s and 76.1 s per projected 2,071-answer epoch. The production-like
`B=8,K=1` probe measured a 0.145 s median tile and only 8.93 GiB peak reserved
memory, but complete aligned-token coverage projected to 5,600.9 s (93.3
minutes): the tile itself is healthy, while repeating the full causal layer
walk for every one-token slice is about 74 times slower per epoch.

The next mechanics table uses an inexpensive nonzero hidden loss, no readout,
and real AdamW updates on constant-area power-of-two diagonals (16, 32, and 64
selected answer-token cells). After those diagonals identify whether larger B
or larger K is faster, the table is completed into the rectangle on that
side. The measured verdict is wide-K: at fixed B, K=8..64 changes latency by
only a few milliseconds, while at fixed 64 selected cells `1×64` takes 0.109 s
and `64×1` takes 1.541 s. The completed fast rectangle peaks at `8×64` =
2,726 selected cells/s. Causal trajectory computation is paid mainly per
answer; once present, adding aligned loss rows is nearly free. Measure
steady-state training separately from epoch-boundary evaluation; recipes that
miss the cache-generation-scale target are not expanded to the other five
models.
