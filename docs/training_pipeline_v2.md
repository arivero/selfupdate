# Training pipeline v2

Pipeline v2 is the training runtime for the dataset-v5 Pareto-frontier
campaign. Pipeline-v1 checkpoints remain historical mechanics evidence but
are not mixed into v2 reports or comparisons.

## Typed training identity

A training is identified by

`dataset × model × censorship × loss × update aggregation × trajectory source × attention source × expert routing`.

The first campaign varies censorship (`remove`, `pad_random`), loss (`huber`,
`lens_kl`), and update aggregation (`answer`, `token`). The other strategy
axes are pinned and recorded now so later experiments do not change the
meaning of a run identity.

| config field | implemented v2 value(s) | reserved value(s) |
|---|---|---|
| `train.update_granularity` | `answer`, `token` | — |
| `train.trajectory_source` | `student_hidden` | `teacher_hidden` |
| `train.attention_source` | `student_attention` | `teacher_attention` |
| `train.expert_routing_source` | `black_box` | `teacher_routing_cache` |

Reserved values are intentionally present in the design but dispatch rejects
them until implemented and certified. A future switch must never be parsed and
ignored.

## Optimizer-update semantics

Both regimes perform complete causal forwards. “Token” changes the gradient
measure and grouping, not the conditioning context.

### Answer aggregation

- `micro_batch: 1`, `grad_accum: 1` are mandatory.
- One optimizer update receives all valid aligned-token gradients from one
  answer.
- Each layer loss is the mean over that answer's aligned tokens.
- This preserves correlations among tokens from one generated answer and
  exposes the highest optimizer-step frequency.

### Token aggregation

- `batching` is `padded` or `bucketed`.
- `grad_accum == micro_batch` is mandatory, so one physical mini-batch is one
  optimizer update.
- An undersized bucket tail is still one update; it is never accumulated into
  a later bucket or across an epoch boundary.
- The trainer first computes a token-mean loss for each answer, then weights
  it by that answer's valid aligned-token count and divides by the total valid
  count. This is exactly the mean over valid aligned tokens in the update;
  padding never contributes.
- The readout term uses its own valid shifted-answer-token counts.

### Historical aggregation

`legacy_answer_sum` reproduces pipeline v1: sum per-answer token means and
accumulate them across answers. It remains the default only so historical
configs retain their meaning. Pipeline v2 rejects it.

The initial answer/token comparison pins learning rate, AdamW, per-block clip
norm, example order, epochs, examples, valid token budget, censorship, and
loss. Optimizer-step count is deliberately different because step grouping is
the hypothesis. Any learning-rate rescaling is a separately named follow-up.

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

Every train row records pipeline version, all strategy axes, optimizer step,
cumulative answers, cumulative aligned tokens, answers and aligned tokens in
the update, batching, and loss by layer. It retains both the per-answer mean
and valid-token-weighted mean for every layer and marks which measure drove
the configured update; report synthesis must not compare unlike measures.
Epoch rows additionally carry recall,
standard damage, time, memory, and per-layer parameter modification from the
epoch-zero/base reference. See `docs/report_v2.md`.

## Speed and quality gate

Before expanding to all six models, Qwen3.5 4B runs a matched probe grid over
both censorship modes, both losses, and both aggregation regimes. Measure
steady-state training separately from epoch-boundary evaluation. A projected
2,071-example epoch should remain near the corresponding cache-generation
time; recipes that miss the target are tuned through bucket size and update
cadence before changing scientific axes.
