# Experiment Plan & Status Board

Updated: 2026-07-05 — branch `classic_kd`.

This branch studies **classical KL-based self-distillation only**. The old
mixed-method program is deliberately out of scope here. The standing question:

> Under classical KD, which transformer layers are modified, which layers make
> the memorized text readable, and how does that localization move with scale?

Metrics: `runs/results.md` (auto) · report: `runs/report.pdf` · logs:
`runs/pipeline_*.log`. Base 0.6B control: CER 0.932, general NLL 3.278.

## Current Lessons

- Training is teacher-logit KL only. Any objective that reads corpus tokens as
  labels is supervised fine-tuning, not distillation, and is out of scope.
- Pure top-k KL can saturate while free-run recitation remains poor; that is a
  result to measure, not a reason to add supervised labels.
- The 8-example training-eval subset is front-of-poem biased. Use full-corpus
  eval for conclusions.
- LoRA needs lr around `1e-4`; `1e-5` can plateau and produce false negatives.
- Forgetting tracks how much text the model has actually internalized; compare
  recipes at matched recitation quality, not just matched epochs.
- Localization should be read with several probes: weight deltas, adapter norms,
  logit lens, and graft/ablate.

## Active Axes

| axis | current question |
|---|---|
| LoRA lr/rank | adapter capacity and layer-localized storage |
| compaction | remove vs stub vs geometry gap |
| data coverage | short windows vs whole-poem anchored windows |
| model size | whether KD writes at fixed absolute depth or proportional depth |
| thinking censorship | whether to train visible traces or hide traces and train only the answer |

## Known Runs To Keep Comparing

| run | headline |
|---|---|
| `kd_full_0p6b_rag` | pure KD drives KL low but does not solve recitation |
| `kd_lora_0p6b_rag` | low-lr LoRA negative control |
| `kd_lora_kl_hi_*` | teacher-logit KL LoRA runs |

## Immediate Work Queue

1. Rebuild `runs/results.md`, `runs/curves.png`, and `runs/report.pdf` after
   every new wave.
2. For each successful KD checkpoint, run:
   `logit_lens.py`, `layer_swap.py`, and `analyze.py --deltas`.
3. Keep `scripts/queue.tsv` and `scripts/queue_h100.tsv` KD-only.
4. Before committing GPU time on larger models, run `evaluate.py --base` and
   `scripts/premise_gate.py`; if base CER is low, choose a new corpus/prompt.
5. Treat thinking traces as an ablation, not the default target: compare
   RAG-hidden-only against RAG+trace-hidden at matched model/data/epoch budget
   before deciding whether reasoning text should be written into weights.

## Model Ladder

| tier | model | unique question |
|---|---|---|
| 3060 | Llama-3.2-1B | tokenizer/template replication |
| 3060 | SmolLM3-3B | open training data sanity check |
| 2x4090 | Qwen3-4B / 8B | localization: absolute vs proportional depth |
| 2x4090 | DeepSeek-V2-Lite | first MoE: expert localization and routing |
| 2x4090 | R1-Distill-Qwen-1.5B | thinking-hiding arm |
| 4xL40S | Qwen3-14B / 32B / 30B-A3B / Qwen3.6-27B | serious scale ladder |
| 4xH100 | GLM / DeepSeek flagship class | Don Quijote stage |

## Operational Rule

Never abort a training run before it has seen at least 12,000 training items
unless the process is clearly broken. Matched item budget is what makes the
grid comparable.

## Current H100 Additions

- `kd_lora_kl_hi_e40_v3_qwen36_27b_rag` uses `Qwen/Qwen3.6-27B` with the same
  v3 RAG/catechism teacher-logit KL recipe as the Qwen30-A3B comparison run.
- The Qwen3.6-27B run is queued behind a one-step online-teacher LoRA smoke
  test capped at 72 GiB before the 40-epoch train/eval jobs are allowed.

## Final Report Handoff

Before generating the final report for this H100 wave, verify that these
artifact classes exist for each completed large-model run:

- Base-model full eval: `runs/base_eval/<run>.json`.
- Final student full eval: `runs/<run>/eval/recite.json`.
- Best-probe student full eval: `runs/<run>/eval_probe_best/recite.json`.
- Teacher-with-current-RAG full eval: `runs/teacher_rag/<run>.json`.
- Layer localization over training: `runs/<run>/eval/lora_layer_deltas_by_epoch.csv`.
- Logit-lens profile for final checkpoint: `runs/<run>/eval/logit_lens.csv`.

The final report should compare teacher and student with the same recitation
metrics: full-corpus CER, line-exact rate, prefix lines, and general-text NLL.
Forgetting is the student general-text NLL minus the matching base-model NLL.
Report both final-checkpoint and best-probe-checkpoint results because late
epochs can improve KL while worsening recall or forgetting.

Regenerate the report artifacts only after the queue has written the above
files:

```bash
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/current_report.py
.venv/bin/python scripts/report.py
```

Expected generated outputs:

- `runs/results.md` — tabular run summary.
- `runs/current_report.md` — live large-run capability/forgetting summary.
- `runs/curves.png` — loss, recall, and general-NLL trajectories.
- `runs/report.pdf` — final human-readable report.

The live unified H100 queue also has a `final_reports_unified.done` entry that
runs these commands after the remaining Qwen3.6, teacher-RAG, logit-lens, and
best-probe eval dependencies finish.
