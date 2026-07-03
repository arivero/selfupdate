# Experiment Plan & Status Board

Updated: 2026-07-03 evening - branch refocused on layerwise forward
distillation only.

Metrics: `runs/results.md` (auto) | report: `runs/report.pdf` | raw logs:
`runs/*/metrics.jsonl` and `runs/pipeline_*.log`.

## Standing Goal

Find a layerwise loss that trains with bounded backward depth and still
produces behavior. "Good" means:

- recites under full-corpus eval, not just the 8-example training subset
- preserves block locality except for explicitly bounded tail windows
- has measurable forgetting/general-CE cost
- scales to online-teacher LoRA and one-block-at-a-time training

## Active Loss Search

| candidate | locality | status |
|---|---|---|
| `nmse` / `l2mse`, summed and sequential | strict one-block | stores signal, weak free-run behavior |
| `teacher_censored` | strict one-block, independent layers | best strict localization readout; context integration peaks near layer 7 |
| last-block CE | strict one-block | insufficient: one block cannot coordinate the readout alone |
| lens-CE on deep/all blocks | strict one-block | active strict-local behavioral auxiliary |
| tail-CE, `k=1/2/4` | bounded `k`-block top window | best current path |
| tail-CE on v2 data, `k=4` | bounded 4-block top window | current champion: CER 0.112 / 90.5% exact; whole-poem anchored CER 0.034 |

## Current Interpretation

Hidden matching appears to learn distributed storage below the top blocks.
Free-run recitation depends on a co-adapted readout circuit in the final
blocks. The practical program is therefore:

1. Keep forward hidden matching as the storage signal.
2. Add only bounded, explicit readout credit where needed.
3. Measure how small that concession can be as model size and data improve.

## Queue State

`scripts/queue.tsv`, `scripts/queue_h100.tsv`, and
`scripts/watchdog_backlog.tsv` are layerwise-only. They contain evals or
layerwise jobs guarded by existing done-file conventions.

## Wave I Result (D1, 2026-07-03 ~23:20)

Loss sweep at 0.6B, v2 data, 40 epochs, summed + tail-CE k=4 (champion
operating point), full-corpus CER / line-exact:

| loss | CER | exact | note |
|---|---|---|---|
| **vocab_mse** | **0.024** | **0.978** | new champion loss (Gram metric ‖W·Δh‖²) |
| l2mse | 0.035 | 0.957 | promoted |
| huber | 0.061 | 0.940 | retired (dominated) |
| nmse (seed 43 anchor) | 0.092 | 0.898 | replication gate passed (0.11±0.03) |
| cosine | 0.104 | 0.924 | retired |
| nmse_strict / vocab_strict | 0.85 | 0.0 | controls: storage without readout never recites |
| lens_kl (±tail) | pending | | poor curves + inner-layer lens is miscalibrated (unembedding only decodes final-layer geometry); expected kill |

Understanding probes (delta profiles, layer_swap ablate, delta-vector
convergence):

- Weight-delta mass concentrates at L22-24 in every arm, but single-layer
  ablation there barely hurts (l2mse: ablate L23/L24 -> CER 0.00) —
  storage is distributed and redundant. Ablating any ONE tail block
  (L25-28) destroys recitation (CER 0.65-0.88): the readout is a fragile
  co-adapted 4-block circuit; storage is not.
- Delta-vector cosine: huber≈nmse (0.95-0.99, same trajectory); l2mse near
  them (0.87-0.91); vocab_mse writes a substantially different solution
  (0.48-0.70 vs all) — and the best one. The metric changes WHAT is
  written, not just how fast.
- tail vs strict at fixed loss: low/mid deltas identical (cos 0.99+), top
  window diverges (0.58-0.62) — tail-CE re-carves only the readout;
  the body's storage is fully determined by hidden matching.
- Frozen-vocabulary rule verified bitwise on trained checkpoints:
  embed/norm deltas exactly 0.0 (tail-CE routes gradient THROUGH the
  frozen head, never INTO it).
- Family smokes: Llama-3.1-8B / Phi-4-mini / Mistral-7B pass all stages
  (template-agnostic machinery works); gpt-oss-20b passes load/adapt/
  teacher/local-step and OOMs only at the connected tail on a shared GPU
  — retry k=1 on an empty card.

D1 decisions: campaign losses = vocab_mse + l2mse. Wave J: scale
(1.7B full-FT, 4B/8B LoRA online), k∈{2,4,8} at 1.7B, family arms,
gpt-oss retry. lens_kl arms run to the 12k floor, then judged.

## Lens Program (Wave I)

Focus: multiple kinds of lens. A lens = optional learned per-layer
translator + decode through the **frozen vocabulary** (final norm + LM
head / embedding). The vocabulary is never trained (see
`docs/hidden_loss.md`, Frozen-Vocabulary Principle); translators are
scaffolding, trainable and discardable.

| lens | learned part | role | status |
|---|---|---|---|
| raw logit lens | none | eval depth profile + `lens_ce` auxiliary | in tree |
| tuned lens | per-layer affine translator, trained on base model then frozen | calibrated depth profiles; early-layer raw readouts are brittle | planned (`eval/logit_lens.py` docstring) |
| embedding lens | none (input vocab) | identical to logit lens on tied 0.6B-4B; distinct probe on untied 8B+ | idea |
| teacher-lens agreement | none | per-layer teacher-vs-student lens KL as localization *readout* | probe only — lens-KL as a training loss failed in Wave H (CER ~0.71-0.73) |
| tuned-lens-CE | frozen pre-trained translator per block | strict-local behavioral auxiliary with a calibrated head per depth | candidate — may close the lens-CE vs tail-CE gap |
| joint aux heads | translator co-trained with its block, discarded after | Belilovsky-style local heads | candidate |

Sequence:

1. Train tuned-lens translators for base Qwen3-0.6B (block-local, vocab
   frozen); add translator support to `eval/logit_lens.py`.
2. Re-profile existing checkpoints (champion tail-CE v2, summed e40,
   teacher_censored) with the tuned lens; compare raw vs tuned depth
   profiles.
3. `lens_ce` through frozen tuned-lens heads vs raw lens-CE vs tail-CE
   k=4, matched item budgets (>= 12k items).
4. If (3) is competitive: joint per-block translator heads.

## Standing Next Work

- Rebuild hidden-state caches with schema 3 after the logit-cache removal.
- Extend `teacher_censored` and tail-CE to larger Qwen checkpoints.
- Keep `evaluate.py --base` outputs lane-specific during concurrent runs.

## Model Ladder

| tier | model | question |
|---|---|---|
| 3060 / L40S | Qwen3-0.6B | loss mechanics, locality tests, ablations |
| L40S | Qwen3-1.7B / 4B / 8B | whether readout window size scales with depth |
| L40S / H100 | Qwen3-14B / 32B | online-teacher LoRA and memory curve |
| H100 | MoE / 120B-class | one-block streaming and Don Quijote scale |
