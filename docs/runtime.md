# Training runtime, optimizer policies, certification

*(Engineering companion to the C3 refactor, 2026-07-10. Science lives in
docs/hidden_loss.md / docs/windows.md; this file is about HOW training
executes.)*

## Separation of concerns

`src/selfupdate/train/runtime.py` owns the executable side of a run;
`layerwise.py` owns WHAT is trained. Since the 2026-07-11 factorisation
the trainer package is one module per concern (layerwise.py re-exports
the historical names for older scripts):

| module | concern |
|---|---|
| `layerwise.py` | schedules (summed / teacher_censored / mixed / sequential) + dispatcher |
| `steps.py` | block/window forward+backward primitives; detach discipline |
| `runtime.py` | TrainingRuntime + OptimizerPlan (below) |
| `teacher_source.py` | per-step frozen-teacher states (OnlineTeacherSource) |
| `validate.py` | dispatch-time knob-flow validation (audit_configs sweeps it) |
| `telemetry.py` | loss aggregation, epoch recall, standard-damage probes |
| `anchor.py` | anti-intrusion anchor regularizer |
| `losses.py` | hidden-match objectives (`HiddenLoss.from_config` is the one construction path) |

The schedule loops never touch `from_pretrained`, device maps, or
optimizer construction:

- **TrainingRuntime** — model loading (causal/ITT fallback), placement
  (single device / `device_map=auto` / explicit pipeline map), LoRA attach,
  teacher source (adapters-off pass or frozen bf16 copy), disk-cache
  resolution, the frozen-vocabulary tripwire, VRAM accounting, checkpoint
  save.
- **OptimizerPlan** — the optimizer policy as a named object:

  | kind | state placement | stepping |
  |---|---|---|
  | `lora_fused` | GPU | one AdamW, foreach (tensor-list overhead negligible at adapter size) |
  | `full_resident` | GPU | one AdamW, non-foreach (peak memory wins at model scale) |
  | `full_offload` | **permanent pinned host buffers** | per-block AdamW, streamed paging |

  Every policy preserves the historical PER-BLOCK clip norm: clipping is
  part of the experiment, not of the execution policy.

## One walk, batched; items are B=1

The summed schedule has a single code path (`_summed_batch`): teacher stage
(cached slices or online forward) → trajectory → loss/backward → update.
`batching: item` collates each example into a B=1 padded batch — bit-exact
against the historical item loop (no pad rows, gather == slice, same kernel
shapes; verified empirically on L40S). B>1 padded batches differ from B=1
only by bf16 kernel-shape rounding (up to ~3e-2 max-relative at deep
layers; tolerances in tests/test_online_teacher.py document this).

Sliding-window trajectory states are released at their last root use.
Precisely (2026-07-11 correction of an overstated claim): the per-window
GRAPH activations follow the window width W, and each detached trajectory
state is freed once no remaining window roots at it — but peak
detached-state residency is still FULL DEPTH (every h_L exists while the
walk crosses it; the -180 MB measured at 0.6B slide8 B=8 comes from the
early releases, not from a W-bounded envelope).

## Streamed optimizer offload

`offload_adam: true` keeps Adam moments in pinned host memory permanently
(pinned buffers are allocated once — repeated `pin_memory()` was measured
SLOWER than the copies it hides; see the negative-result note in issues.md).
`OptimizerPlan._step_offload` pages moments through the GPU block by block:
block i+1's H2D prefetch rides a side stream under block i's step kernels,
and the D2H writeback overlaps block i+1. Measured at 0.6B on L40S:
0.949 → 0.358 s/step (grad_accum 8); step math is bitwise identical to the
resident path (tests/test_offload_adam.py).

## Pipeline parallelism

PP is the preferred multi-GPU form for this workload: layerwise execution
partitions naturally at block boundaries, while tensor parallel puts a
collective inside every linear (parallel_bench.py keeps TP only as a probe;
at trainable sizes it loses badly). Two facts from the 2026-07-10
measurements (issues.md):

- PP is a **memory** technology here, not throughput: the walk is
  depth-sequential, so PP2 is slower than single-GPU whenever the model
  fits on one card. Split only when it does not fit.
- accelerate's per-call dispatch hooks cost ~8% of the PP2 walk. Under an
  explicit `pipeline_split(s)` map the walk therefore runs **hook-free**:
  `BlockStack(model, hook_free_walk=True)` calls each block's pre-hook
  forward and does the boundary moves itself (activation + per-device rope
  cache). Full-model forwards (recite/general-CE evals, generation) keep
  their hooks and are unaffected. The bypass never engages when a hook
  offloads weights (`device_map=auto` spill) or for per-layer-rope bundles
  (gemma4-style).

Within a grad-accum window the weights are frozen, so cross-item device
overlap (item i+1 on partition 0 while item i finishes partition 1) would
be EXACT — the honest PP throughput move if it is ever needed; not
implemented.

## Checkpoint publication

`runs/<name>/checkpoint` is a scheduler dependency, hence a public completion
signal rather than a scratch directory. `TrainingRuntime.save_checkpoint`
writes the model and tokenizer to a sibling `.checkpoint.incomplete-*`
directory and atomically renames it to `checkpoint` only after both save
successfully. A failed save removes its staging directory and exposes no
checkpoint. This prevents dependent recall or standard-damage evaluation from
loading a directory while it is still being populated on Lustre.

## Certification vs benchmarking

- `scripts/train_certify.py` — "is this the same experiment?": runs the
  real `train_layerwise` on 13 tiny variants covering every
  schedule/batching/window/optimizer path; fingerprints per-step losses,
  per-tensor checkpoint signatures, VRAM peaks. NO references are stored
  in the repo (owner decision 2026-07-11: stored fingerprints act as a
  frozen-numerics specification; the pre-refactor `certs/pre` and PP2
  `certs/pp2` sets are in git history). Use it as an on-demand A/B
  instrument: record on HEAD (`--all --out-dir /tmp/$USER/certify_head`),
  apply the change, compare (`--all --reference-dir ...`), discard. The
  comparison keys on a semantic config hash that EXCLUDES placement knobs,
  so one single-device recording certifies PP runs;
  `--override model.pipeline_split=14` runs the same variants under PP.
- `scripts/train_batch_bench.py` — "how fast?": the real summed path,
  timing and memory JSON, no checkpoint.
- `scripts/memory_plan.py` — "will it fit?": meta-device instantiation +
  one materialized block measure per-(B, T) activation bytes without
  loading weights; ADVISORY only (config defaults are experiment
  variables). Predictions exclude loss-head workspace — apply ~25% margin
  for vocab-metric losses.

There is no standing gate (tests and stored references deleted
2026-07-11; the runtime validators and tripwires in the code are the
enforcement). For a trainer change intended to be numerics-preserving,
run the record-on-HEAD → change → compare cycle above before it lands.
