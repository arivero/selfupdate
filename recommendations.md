# Recommendations For A Consistent Experiment Corpus

Purpose: planning base for the next cleanup and reporting pass. The goal is a
consistent corpus of experiments and conclusions across losses, layers, model
families, model sizes, datasets, and time, with enough plots and raw artifacts
that future claims do not depend on memory or selective summaries.

This file assumes the branch rule in `CLAUDE.md` / `AGENTS.md`: this checkout
is for layerwise forward distillation. Tail-only and `tail_*` work is not an
active method surface here. Preserve historical evidence only as archived
context, or move it to `../selfupdate_kd`.

Status: the closed sections (the 2026-07-05 tooling done-list and "Branch
Hygiene First", completed 2026-07-05/07; hot-loop/parallel items, completed
2026-07-10) were removed on 2026-07-10 — git history keeps them. Live
remaining work is tracked in `docs/fable_review_status_2026-07-07.md`; what
follows is the standing SPEC for the experiment corpus, not a todo list.

## Outcome Wanted

Every major conclusion should be backed by a reproducible evidence bundle:

1. A run set with pinned configs and clear classification.
2. Raw metrics for training dynamics.
3. Full-corpus evaluation, not only the 8-example training subset.
4. Forgetting and intrusion evaluation.
5. Per-layer loss plots over training.
6. Per-layer residual or storage-quality plots at checkpoints.
7. Signal attribution between layerwise hidden losses and readout terms.
8. Model-size and model-family coverage, not only the Qwen3-8B or one-rung
   result.
9. A conclusion table that explicitly states what is proven, what is epoch-zero
   teacher reference only, what is confounded, and what is open.

## Run Classification Contract

Every run must have a classification field, either in config or derived by a
report manifest:

| class | meaning | allowed in method conclusions |
|---|---|---|
| `method` | teacher-sourced layerwise method, sanctioned window semantics | yes |
| `teacher_reference` | epoch-zero teacher, native or RAG/context input | no, reference only |
| `ablation` | mechanism probe that violates one method invariant intentionally | no |
| `control` | negative or instrumentation control | no |
| `legacy_archive` | historical run kept for context only | no |
| `confounded` | run whose config inherited the wrong default or mixed axes | no |
| `open` | incomplete or awaiting eval | no |

Required tags:

- `model_family`
- `model_name`
- `model_size`
- `dataset_family`
- `data_variant`
- `schedule`
- `hidden_loss`
- `window_kind`
- `conn_window`
- `conn_stride`
- `readout_source`
- `anchor_kind`
- `seed`
- `run_class`
- `conclusion_group`

## Minimum Artifact Bundle Per Run

Each finished run should have:

- `runs/<run>/config.yaml`
- `runs/<run>/metrics.jsonl`
- `runs/<run>/checkpoint/` or an explicit reason no checkpoint exists
- `runs/<run>/eval/recite.json`
- `runs/<run>/eval/destruction.json`
- `runs/<run>/eval/layer_losses.png`
- `runs/<run>/eval/layer_losses.csv`
- `runs/<run>/eval/forget_recall_curve.png`
- `runs/<run>/eval/forget_recall_curve.csv`
- `runs/<run>/eval/layer_residuals.json`
- `runs/<run>/eval/layer_residuals.png`
- `runs/<run>/eval/signal_attribution.json` when any readout term exists
- `runs/<run>/eval/weight_deltas.csv` for full fine-tune, or LoRA delta summary
- `runs/<run>/eval/recite_long.json` for crown candidates

Status 2026-07-10 (late): the bundle has no open gaps — `layer_loss_plots.py`
and `forget_curves.py` landed 2026-07-05/07, and checkpoint-time layer
residuals landed 2026-07-10 (`evaluate.py --layer-residuals`).

## Required Plots

### Per-Layer Loss Dynamics

For every run, generate:

- One plot with all layers over epochs or optimizer steps.
- One heatmap: layer on y-axis, time on x-axis, log loss as color.
- One grouped comparison plot by loss family: rows are model sizes, columns are
  hidden losses.
- A CSV with columns:
  `run, epoch, item, layer, loss_mean, loss_median, loss_p10, loss_p90`.

The plot must not only show Qwen3-8B. It must be emitted for all completed
runs where `metrics.jsonl` has `per_layer`.

### Forgetting And Recall Over Training

For every run with per-epoch eval records:

- Plot recall as full-corpus character error rate when available.
- Plot subset character error rate only as a secondary curve, clearly labeled.
- Plot general cross-entropy (log loss) delta against the base model.
- Plot intrusion hit rate and worst-category delta when destruction evals exist.
- Show vertical markers for major schedule changes if any.

The current `forget_curves.py` uses subset recall from training-time evals and
general-CE delta. That should be extended to include full eval checkpoints or a
periodic full-eval lane for important runs.

### Per-Layer Residuals At Checkpoints

Add an eval mode:

```bash
python scripts/evaluate.py --checkpoint runs/<run>/checkpoint --layer-residuals
```

Required outputs:

- `layer_residuals.json`
- `layer_residuals.csv`
- `layer_residuals.png`

Metrics per layer:

- `nmse`
- `l2mse`
- `vocab_mse`
- `lens_kl` where affordable
- residual norm ratio
- teacher/student logit-lens agreement

This separates storage quality from training loss. Training loss tells what the
optimizer saw; checkpoint residuals tell what the trained model stores.

### Signal Attribution

Every run with a readout term must have `signal_attribution.json`.

Required report values:

- hidden gradient norm
- readout gradient norm
- hidden share
- per-block hidden/readout share
- readout source: `teacher_kl`
- classification: method, teacher_reference, ablation, control, archive

The report should refuse to call a result layerwise-primary unless hidden share
passes a stated threshold or the text explicitly says it does not.

### Model Comparison Matrix

Generate one summary figure with rows by model and columns by metric:

- recall character error rate
- line exact
- dialogue character error rate
- long-recitation first error
- mean general-CE delta
- worst destruction category
- intrusion hit rate
- hidden share
- train minutes
- peak reserved VRAM
- items seen

This is the plot that prevents overfitting conclusions to one rung.

## Model Coverage Requirements

Do not report a conclusion as cross-model unless it includes at least:

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B or Qwen3-4B LoRA where full fine-tune is unavailable
- Qwen3-8B or Qwen3-8B LoRA
- Qwen3-14B or larger LoRA
- Qwen3.6-27B as the H100 dense bridge
- Gemma-4-26B-A4B as the 2026 MoE bridge
- Gemma-4-31B as the dense high-context scale point
- one non-Qwen older dense family only as a secondary comparison, currently
  Mistral-7B
- gpt-oss as the reasoning/MoE control if it remains load-compatible

For H100 or bridge work, add:

- Qwen3.6-27B single-H100 reference
- Qwen3.6-27B PP2 or TP2 repro
- Qwen3.6-27B H100 versus L40S comparison if quantized or sharded lanes are
  used
- Gemma-4-26B-A4B single-H100 and PP2 speed certification
- Gemma-4-31B PP2 teacher reference plus an explicit single-H100 boundary test

## MoE Method Evidence

MoE runs must state their routing mode. Black-box MoE is still method evidence
for layerwise block-output distillation, because the router and experts are
inside the block being trained. Claims about expert-mechanism matching require
teacher/student router agreement evidence.

Required MoE modes:

- `dense_or_black_box`: match post-combine block outputs; method evidence with
  router agreement unproven.
- `teacher_forced`: train the student block while replaying teacher-selected
  top-k experts.
- `router_aligned`: add a router objective so the student router agrees with
  the teacher router; report top-k overlap by layer/token.

The MoE-specific artifact bundle should add `router_trace.json`,
`router_overlap.csv`, per-layer router-overlap plots, and per-expert delta
norms. The report should separate "block-output method evidence" from
"expert-mechanism evidence".

Every new 2026 model must have a training-speed certificate before any long
run is interpreted: batch-size sweep, peak memory, GPU utilization trace,
and PP2 communication comparison where the model is not comfortably inside
the one-card memory budget.

Each model must have its own base general-CE reference:

```bash
python scripts/base_general.py <model> runs/base-general-<model_short>.json
```

The report should flag missing base references because forgetting deltas are
invalid without them.

## Experiment Families To Standardize

### Loss Family Grid

For each covered model rung, run or archive a clean comparison for:

- `nmse`
- `l2mse`
- `vocab_mse`
- `vocab_fisher`
- `lens_kl` only as method-compatible teacher-sourced loss, not label CE
- `zero` only as a control

Each loss grid must use the same:

- data variant
- item budget
- seed set
- window semantics
- readout source
- anchor settings
- eval battery

### Window Family Grid

Allowed active method windows:

- strict local, no connected readout
- `conn_window: 2`, `conn_stride: 1`
- `conn_window: 4`, `conn_stride: 1`
- `conn_window: 8`, `conn_stride: 1`
- teacher-stream k-windows once implemented

Disallowed in active method reports:

- tail-only windows
- top-k-only readout training without full sliding coverage
- depth-increasing readout weights
- label-targeting local lens CE as method

### Readout Source Grid

Keep source roles separate:

- `teacher_kl`: teacher-sourced method readout.
- reference-text readout: forbidden in this branch.
- transcript-equivalent readout: deployment analogy only, not a lab method
  unless the transcript is literally teacher-generated in that run.

The conclusion table must not merge these roles.

### Data Family Grid

Separate data effects from method effects:

- Machado v2 recitation
- Machado v4 maieutic/dialogue
- thinking selective
- Quijote chapter rungs
- combined content
- anchor variants
- heldout/passage-only channel runs

Each data family needs a `teacher_reference`, `method`, and `negative control`
where feasible.

## Conclusion Ledger

Create a machine-readable ledger, for example `runs/conclusions.yaml`, with one
entry per claim:

```yaml
- id: connectivity_law
  status: open
  claim: "Uniform sliding windows improve recall and stability."
  required_runs:
    - lw_r_slide2_0p6b_rag
    - lw_r_slide4_0p6b_rag
    - lw_r_slide8_0p6b_rag
    - lw_r_disj_pinned
  blocking_gaps:
    - "Pinned disjoint eval incomplete or confounded."
  allowed_report_class: method
```

Suggested statuses:

- `proven`
- `replicated`
- `single_seed`
- `confounded`
- `open`
- `retracted`
- `teacher_reference_only`
- `archived`

The report should render this ledger before narrative findings. Narrative
without claim status is how stale conclusions survive after confounds are
found.

## Report Redesign

`scripts/report.py` should be rebuilt around the corpus, not around historical
narrative.

Recommended report order:

1. Branch rules and excluded evidence.
2. Conclusion ledger.
3. Model coverage matrix.
4. Run classification table.
5. Recall and forgetting curves.
6. Per-layer loss dynamics.
7. Per-layer checkpoint residuals.
8. Signal attribution.
9. Destruction and intrusion battery.
10. Memory and speed table.
11. Appendices with legacy/archive results.

The report should fail or warn loudly when:

- a method run lacks full eval
- a method run lacks destruction eval
- a readout run lacks signal attribution
- a model lacks a base general-CE reference
- a run trained against reference text
- a run has legacy `tail_*` knobs in the active report
- a conclusion references a confounded run

## Analysis Script Worklist

COMPLETE as of 2026-07-10: audit/corpus-index/layer-loss/forget-curve items
landed 2026-07-05/07; the report.py rebuild was superseded by
`cross_report.py` + the retention battery; `evaluate.py --layer-residuals`,
`scripts/model_matrix.py`, and `scripts/conclusion_check.py` all landed
2026-07-10.

## Minimum Gates For Future Claims

A claim can be promoted only if:

- configs are pinned and pass audit
- every required run has full eval
- every required run has forgetting/destruction eval
- every readout run has signal attribution
- at least two seeds exist for crown or headline claims
- at least three model rungs exist for scaling claims
- at least one non-Qwen family exists for family-general claims
- H100/PP/TP claims have a single-device reference when possible
- every exception is listed in the conclusion ledger

## Immediate Next Steps

All done as of 2026-07-10 (layer residuals + `runs/conclusions.yaml`
landed; earlier steps removed to git history). This file is now pure
standing SPEC; live work is the C3 queue in `EXPERIMENTS.md` and the
open-findings backlog in `issues.md`.

The standing principle: the target is not more arms. It is making the
existing arms queryable, comparable, and impossible to misclassify.
