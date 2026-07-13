# V5 generation speed scoreboard

Last updated: 2026-07-13.  Full `examples_v5rs_window.jsonl`: 2,071 prompts,
greedy decoding, batch 64 unless a one-card capacity limit is stated.  Times
are answer generation only; model loading/capture, hidden-state extraction,
D2H, and storage are separate.  Quality is next/previous word LCS and cloze
target-block lexical precision.  A result is entered only after its output
summary is complete.  Full-corpus and sample scores are separate: a sample
can diagnose throughput but never displaces a full-run result.

## Full-corpus single-card scoreboard (n = 2,071)

Rank full runs primarily by generation seconds and secondarily by generated
tokens per second.  Token count is retained because models stop at different
answer lengths.

| model | runtime / mode | commit | requested batch | setup | generation | tokens | tok/s | hard cuts | next/prev LCS | cloze precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | PyTorch compiled/hybrid | `6396fd6` | 32 | 193.09 s | **2,267.86 s** | 74,422 | **32.82** | 0.14% | 94.34% | 96.65% |

## Sample/probe scoreboard

| model | runtime / mode | commit | n | tokens | setup | generation | tok/s | hard cuts | next/prev LCS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | in-repo PyTorch compiled/hybrid | `658d46b` | 8 | 432 | 170.61 s | 14.40 s | **30.00** | 12.5% | 100.0% |
| Gemma-4-26B-A4B-it | in-repo PyTorch eager/hybrid | `658d46b` | 8 | 432 | 0.00 s | 181.25 s | 2.38 | 12.5% | 100.0% |
| Qwen3.6-35B-A3B | in-repo PyTorch compiled/hybrid | `1ec5e65` | 8 | 327 | 110.16 s | 10.86 s | **30.11** | 0.0% | 87.5% |

vLLM baselines and targets are deliberately excluded from this scoreboard;
they remain in `docs/vllm_generation_benchmark.md`.

## Partial batch scores

| model | runtime / mode | commit | completed | tokens | elapsed | running tok/s | disposition |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B | graph exact-token backend, exact-budget subgroups, batch 64 | `e912f2d` | 2,001 / 2,071 | 73,699 | 280.8 s | **262.46** | late OOM at 0.99 reservation; no full score, retrying at `1f28029` with mixed-budget scheduling and margin |
| Qwen3.5-4B | PyTorch compiled/hybrid, static compile/cache length, fixed physical batch 32 | `e7f930e` | 32 / 2,071 | 840 | 53.32 s | **15.75** | stopped: decisively below target |
| Qwen3.5-4B | same, fixed physical batch 64 | `e7f930e` | 0 / 2,071 | 0 | >88 s | n/a | stopped before first batch completed |

## Aborted full attempts

| model | runtime / mode | commit | observed wall lower bound | tok/s | disposition |
|---|---|---|---:|---:|---|
| Gemma-4-26B-A4B-it | in-repo PyTorch compiled/hybrid, batch 64 | `264610f` | >1,911 s | n/a | generation incomplete and no longer able to beat the eager target; stopped to test dense fixed-shape strategies |
| Qwen3.5-4B | PyTorch compiled/hybrid dynamic, batch 64 | `af6bc76` | >1,404 s | n/a | generation incomplete after corrected eager cutoff; stopped for matched dynamic/fixed probes |
| Qwen3.5-4B | PyTorch compiled/static fixed, n=64 | `3b4939b` | n/a | n/a | incompatible before generation: Transformers has no static-cache mask mapping for `linear_attention` |

## Exact-token cache backend attempts

These are our cache-generation workflow attempts, not imported historical
vLLM reference rows.  They use the graph-capable continuous generation backend
and preserve exact response token IDs for lossless reuse by the in-repo
`build_teacher_cache.py` hidden-state phase.

| model | runtime / mode | commit | requested batch | setup | generation | tokens | tok/s | hard cuts | next/prev LCS | cloze precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | one-card graph exact-token backend, mixed per-record budgets | `1f28029` | 64 | 94.48 s | **47.30 s** | 85,028 | **1,797.58** | 1.45% | 97.33% | 81.15% |
| Gemma-4-26B-A4B-it | one-card graph exact-token backend, exact-budget subgroups | `e912f2d` | 64 | 142.99 s | **297.22 s** | 84,276 | **283.55** | 1.45% | 97.11% | 81.45% |
| Qwen3.6-35B-A3B | one-card graph exact-token backend, mixed per-record budgets | `1f28029` | 64 | 143.70 s | **50.80 s** | 75,219 | **1,480.78** | 0.24% | 93.94% | 96.96% |
| Qwen3.5-4B | one-card graph exact-token backend, mixed per-record budgets | `1f28029` | 64 | 106.32 s | **35.22 s** | 87,306 | **2,478.57** | 2.12% | 91.96% | 92.55% |

Each winning artifact contains 2,071 unique rows; every row contains a
non-empty `token_ids` list whose length matches `gen_tokens`.  Gemma's earlier
297.22-second exact-budget-subgroup answer phase met the approximately
300-second generation objective on one card; the mixed-budget implementation
then reduced the same phase to 47.30 seconds.
The mixed-budget change is the dominant speedup: it retains a separate exact
ceiling and stop ID for every record while allowing all 64 differently sized
requests into one engine call.  The Qwen rows preserve the same quality window
while finishing far below both the eager and prior graph timings.

## Full hidden-state cache phase (n = 2,071)

Generation is imported as exact token IDs in both rows, so these numbers cover
only the in-repo teacher-forced hidden-state pass and its asynchronous write.
Model loading is outside `total`, just as graph-backend setup is outside the
generation time above.  `storage` is worker time and overlaps compute; `total`
is the phase wall clock.

| model | hidden mode | commit | requested / effective batch | teacher compute | D2H | storage | hidden bytes | total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | one forward per example | `e35594a` | 1 / 1 | 612.43 s | 0.715 s | 24.48 s | 36.63 GB | **623.64 s** |
| Qwen3.5-4B | length-aligned randomized batches, OOM backoff | `41d65c5` | 64 / 5–64 | 68.76 s | 0.924 s | 16.08 s | 36.63 GB | **86.20 s** |

The batched row is 7.23× faster in wall time.  D2H is 1.34% of teacher compute
even after compute was accelerated, so PCIe transfer is not the bottleneck;
storage/backpressure is now the secondary cost.  All 2,071 semantic index
entries match the B=1 cache.  A full-cache audit sampled 32 examples × 32
layers bit-exactly, and the preceding 64-example certification compared all
2,048 tensors bit-exactly (zero maximum and mean absolute difference).

For the dense teacher, completed steady phases are 35.22 s generation plus
86.20 s hidden caching = 121.42 s.  Their independent model load/setup costs
remain separate and must not be hidden inside that sum.

Per-effective-batch partial progress is available for launches at commit
`da68d4a` and later.  Hidden-cache progress is persisted every 100 examples at
commit `fe9fd3c` and later without a CUDA synchronization in the teacher walk.

Primary objective: beat each matching one-card eager time with no meaningful
quality loss, then reduce the remaining gap to the graph target.  Do not mix
setup time into the generation column; report it alongside each new row.
