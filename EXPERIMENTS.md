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

## Next Work

- Finish lens-CE and tail-CE comparisons on the current L40S artifacts.
- Rebuild hidden-state caches with schema 3 after the logit-cache removal.
- Re-run the focused layerwise test suite before launching training.
- Extend `teacher_censored` and tail-CE to larger Qwen checkpoints.
- Keep `evaluate.py --base` outputs lane-specific during concurrent runs.

## Model Ladder

| tier | model | question |
|---|---|---|
| 3060 / L40S | Qwen3-0.6B | loss mechanics, locality tests, ablations |
| L40S | Qwen3-1.7B / 4B / 8B | whether readout window size scales with depth |
| L40S / H100 | Qwen3-14B / 32B | online-teacher LoRA and memory curve |
| H100 | MoE / 120B-class | one-block streaming and Don Quijote scale |
