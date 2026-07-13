# V5 generation speed scoreboard

Last updated: 2026-07-13.  Full `examples_v5rs_window.jsonl`: 2,071 prompts,
greedy decoding, batch 64 unless a one-card capacity limit is stated.  Times
are answer generation only; model loading/capture, hidden-state extraction,
D2H, and storage are separate.  Quality is next/previous word LCS and cloze
target-block lexical precision.  A result is entered only after its output
summary is complete.

| model | runtime / placement | mode | generation | tok/s | hard cuts | next/prev LCS | cloze precision | role |
|---|---|---|---:|---:|---:|---:|---:|---|
| Gemma-4-26B-A4B-it | vLLM 0.25, 2 x H100 PP2 | graphs | **298.71 s** | 284.69 | 1.45% | 97.29% | 81.37% | best known graph target |
| Gemma-4-26B-A4B-it | vLLM 0.25, 1 x H100 | eager | 1,679.97 s | 49.74 | 1.35% | 97.11% | 80.94% | one-card eager time to beat |
| Qwen3.6-35B-A3B | vLLM 0.25, 2 x H100 PP2 | graphs | **284.83 s** | 261.55 | 0.19% | 93.84% | 97.13% | best known graph target |
| Qwen3.6-35B-A3B | vLLM 0.25, 1 x H100 | eager | 2,326.59 s | 31.99 | 0.10% | 94.10% | 97.30% | one-card eager time to beat |
| Qwen3.5-4B | vLLM 0.25, 1 x H100 | graphs | **254.17 s** | 341.00 | 2.22% | 91.82% | 92.56% | best known graph target |
| Qwen3.5-4B | vLLM 0.25, 1 x H100 | eager | 1,240.15 s | 69.90 | 2.03% | 92.01% | 92.77% | one-card eager time to beat |

## Live in-repo candidates

- Gemma compiled/hybrid, requested batch 64: full generation running on GPU 0;
  no score until `generation_timings.json` is written.
- Qwen3.6 compiled/hybrid, requested batch 32 after the measured batch-64 OOM:
  corrected full generation running on GPU 1; no score until completion.
- Qwen3.5-4B dense compiled controls: model staged under `/tmp`; queued for the
  first free permitted lane.  Compare hybrid/dynamic with static/fixed shapes.

Primary objective: beat each matching one-card eager time with no meaningful
quality loss, then reduce the remaining gap to the graph target.  Do not mix
setup time into the generation column; report it alongside each new row.
