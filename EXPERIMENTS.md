# Experiment Plan & Status Board

Updated: 2026-07-03 — branch `classic_kd`.

This branch studies **classical KL-based self-distillation only**. The old
mixed-method program is deliberately out of scope here. The standing question:

> Under classical KD, which transformer layers are modified, which layers make
> the memorized text readable, and how does that localization move with scale?

Metrics: `runs/results.md` (auto) · report: `runs/report.pdf` · logs:
`runs/pipeline_*.log`. Base 0.6B control: CER 0.932, general-CE 3.278.

## Current Lessons

- Pure top-k KL can saturate while free-run recitation remains poor.
- Gold answer CE is the lever that makes recitation appear; treat CE weight as
  a first-class axis, not a cosmetic auxiliary.
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
| pure KL vs KL+gold CE | when does distribution matching become free-run text? |
| CE weight | recitation/forgetting tradeoff |
| LoRA lr/rank | adapter capacity and layer-localized storage |
| compaction | remove vs stub vs geometry gap |
| data coverage | short windows vs whole-poem anchored windows |
| model size | whether KD writes at fixed absolute depth or proportional depth |

## Known Runs To Keep Comparing

| run | headline |
|---|---|
| `kd_full_0p6b_rag` | pure KD drives KL low but does not solve recitation |
| `kd_ce_0p6b_rag` | KD + answer CE, best early full-FT baseline |
| `kd_ce_long_0p6b_rag` | longer full-FT budget |
| `kd_lora_0p6b_rag` | low-lr LoRA negative control |
| `kd_lora_ce_hi_0p6b_rag` | LoRA lr corrected to `1e-4` |
| `kd_lora_ce_hi_v2_0p6b_rag` | extended-recitation data, strong LoRA recipe |
| `kd_lora_ce_hi_stub*` | compaction controls |
| `kd_lora_ce_hi_*b_rag` | scale ladder |

## Immediate Work Queue

1. Rebuild `runs/results.md`, `runs/curves.png`, and `runs/report.pdf` after
   every new wave.
2. For each successful KD checkpoint, run:
   `logit_lens.py`, `layer_swap.py`, and `analyze.py --deltas`.
3. Keep `scripts/queue.tsv` and `scripts/queue_h100.tsv` KD-only.
4. Before committing GPU time on larger models, run `evaluate.py --base` and
   `scripts/premise_gate.py`; if base CER is low, choose a new corpus/prompt.

## Model Ladder

| tier | model | unique question |
|---|---|---|
| 3060 | Llama-3.2-1B | tokenizer/template replication |
| 3060 | SmolLM3-3B | open training data sanity check |
| 2x4090 | Qwen3-4B / 8B | localization: absolute vs proportional depth |
| 2x4090 | DeepSeek-V2-Lite | first MoE: expert localization and routing |
| 2x4090 | R1-Distill-Qwen-1.5B | thinking-hiding arm |
| 4xL40S | Qwen3-14B / 32B / 30B-A3B | serious scale ladder |
| 4xH100 | GLM / DeepSeek flagship class | Don Quijote stage |

## Operational Rule

Never abort a training run before it has seen at least 12,000 training items
unless the process is clearly broken. Matched item budget is what makes the
grid comparable.
