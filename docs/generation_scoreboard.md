# V5 generation speed scoreboard

Last updated: 2026-07-14.  Full `examples_v5rs_window.jsonl`: 2,071 prompts,
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
| Llama-3.3-70B-Instruct | two-card PP2 graph exact-token backend, mixed per-record budgets | `aa76927` | 1,856 / 2,071 | 71,063 | 253.7 s | **280.1** | vLLM async-scheduler assertion; partial placement score only |

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

Generation is imported as exact token IDs from the completed external vLLM
0.25 artifacts in `runs/vllm_benchmark_h100`.  The cache builder therefore
does not call vLLM during these rows: these numbers cover only the in-repo
PyTorch teacher-forced hidden-state pass and its asynchronous write.  They are
not end-to-end answer-plus-cache times; add the corresponding vLLM generation
row when estimating that path.  Model loading is outside `total`, just as
graph-backend setup is outside the generation time above.  `storage` is worker
time and overlaps compute; `total` is the cache phase wall clock.

| model | hidden mode | run commit | requested / observed batch | teacher compute | D2H | storage | hidden bytes | total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | single card, length-aligned | `d285abb` | 64 / 23–64 | 30.46 s | 0.467 s | 8.95 s | 17.55 GB | **37.46 s** |
| Qwen3-1.7B | single card, length-aligned | `d285abb` | 64 / 23–64 | 33.11 s | 0.696 s | 14.80 s | 36.22 GB | **43.24 s** |
| Qwen3.5-0.8B | single card, length-aligned | `d285abb` | 64 / 23–64 | 50.80 s | 0.376 s | 8.50 s | 18.75 GB | **57.00 s** |
| Qwen3.5-2B | single card, length-aligned | `d285abb` | 64 / 23–64 | 54.72 s | 0.751 s | 13.24 s | 29.31 GB | **63.25 s** |
| GPT-OSS-20B | single card, exact response IDs | `8cededa` | 64 / 7–64 | 44.02 s | 1.050 s | 39.19 s | 39.25 GB | **86.16 s** |
| GPT-OSS-120B | two-card auto device map, native MXFP4, bfloat16 cache | `7201891` | 64 / 1–64 | 86.82 s | 0.614 s | 28.75 s | 39.11 GB | **113.37 s** |
| Gemma-4-26B-A4B-it | single card, exact response IDs | `8cededa` | 64 / 26–64 | 66.28 s | 0.746 s | 33.99 s | 37.06 GB | **92.03 s** |
| Llama-3.1-8B-Instruct | single card, length-aligned | `b3ed8df` | 64 / 23–64 | 56.83 s | 1.362 s | 55.95 s | 72.53 GB | **94.32 s** |
| Qwen3.5-9B | single card, length-aligned | `d285abb` | 64 / 23–64 | 88.15 s | 1.088 s | 38.46 s | 56.52 GB | **101.08 s** |
| Qwen3-8B | single card, length-aligned | `d285abb` | 64 / 23–64 | 59.78 s | 2.220 s | 59.06 s | 81.71 GB | **102.82 s** |
| Qwen3.5-4B | single card, exact response IDs | `8cededa` | 64 / 37–64 | 69.02 s | 0.727 s | 36.09 s | 36.63 GB | **105.03 s** |
| Mistral-7B-Instruct-v0.1 | single card, exact response IDs | `8cededa` | 64 / 23–64 | 60.64 s | 1.866 s | 70.94 s | 100.73 GB | **118.76 s** |
| Qwen3-14B | single card, length-aligned, OOM backoff | `fa43971` | 64 / 23–64 | 85.82 s | 2.479 s | 68.71 s | 99.26 GB | **137.16 s** |
| Phi-4 | single card, length-aligned, OOM backoff | `b3ed8df` | 64 / 23–64 | 90.72 s | 2.302 s | 91.16 s | 121.96 GB | **159.39 s** |
| Qwen3.6-35B-A3B | single card, exact response IDs, OOM backoff | `8cededa` | 64 / 16–64 | 145.57 s | 0.723 s | 29.08 s | 34.65 GB | **167.41 s** |
| ALIA-40B-FC-2606 | PP2, exact response IDs, OOM backoff | `8cededa` | 64 / 12–32 | 148.87 s | 2.524 s | 167.82 s | 177.00 GB | **321.05 s** |
| Qwen3.6-27B | single card, exact response IDs, OOM backoff | `8cededa` | 64 / 16–64 | 234.36 s | 2.539 s | 122.58 s | 135.62 GB | **361.41 s** |
| Gemma-4-31B-it | single card, exact response IDs, OOM backoff | `8cededa` | 64 / 4–32 | 247.80 s | 2.603 s | 124.65 s | 137.34 GB | **361.89 s** |
| Qwen3-32B | single card, exact response IDs, OOM backoff | `8cededa` | 64 / 4–32 | 242.72 s | 3.439 s | 152.81 s | 184.90 GB | **386.26 s** |
| NVIDIA Nemotron Nano 9B v2 | single card, exact response IDs, batch-1 fallback | `8cededa` | 64 / 1 | 1,070.53 s | 5.182 s | 208.79 s | 229.08 GB | **1,209.25 s** |

Across the completed rows, D2H is 1.34% of teacher compute on the original
batched calibration, so PCIe transfer is not the bottleneck;
storage/backpressure is now the secondary cost.  All 2,071 semantic index
entries match the B=1 cache.  A full-cache audit sampled 32 examples × 32
layers bit-exactly, and the preceding 64-example certification compared all
2,048 tensors bit-exactly (zero maximum and mean absolute difference).

For the dense Qwen3.5-4B teacher, the completed steady phases are 35.22 s
generation plus 105.03 s hidden caching = 140.25 s.  Their independent model
load/setup costs remain separate and must not be hidden inside that sum.

The new exact-ID rows put D2H at 0.4–3.7% of teacher compute (5.2 seconds in
the batch-1 Nemotron fallback); storage worker time reaches 208.8 seconds and
is comparable to or larger than compute for the biggest caches.  The
unavoidable CUDA copy is therefore not the throughput bottleneck.  The writer
overlaps storage with the teacher walk, so `storage` is worker time and must
not be added to `total`.

The exact-ID ladder now covers 21 complete full-corpus caches, including
GPT-OSS-120B.  The 120B row uses native MXFP4 weights with `kernels==0.12.0`
and bfloat16 hidden-state storage; the first attempt was rejected because
float16 storage overflowed an outlier channel.  The exact-token response
artifact was imported from the vLLM campaign, so the 113.37-second figure is
hidden-cache construction only, not answer generation.

### Large-model hidden probes (n = 64)

These evenly spaced probes verify one-card capacity and code paths; they do not
rank above full-corpus rows.  Both requested B=64 and completed without OOM.
Their effective maxima are below 64 only because 64 sampled examples are split
across several 128-token length buckets.

| model | commit | layers | effective batch | teacher compute | D2H | storage | hidden bytes | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | `41d65c5` | 30 | 1–42 | 10.35 s | 0.023 s | 0.519 s | 1.21 GB | **11.29 s** |
| Qwen3.6-35B-A3B | `41d65c5` | 40 | 1–44 | 11.23 s | 0.021 s | 0.521 s | 1.08 GB | **12.23 s** |
| Qwen3-1.7B | single-card probe | `d285abb` | 64 | 3.585 s | 0.021 s | 0.605 s | 1.11 GB | **5.438 s** |
| Qwen3-1.7B | PP2 probe, split 14/14 | `5c7bad7` | 64 | 3.548 s | 0.032 s | 0.708 s | 1.11 GB | **4.723 s** |

Per-effective-batch partial progress is available for launches at commit
`da68d4a` and later.  Hidden-cache progress is persisted every 100 examples at
commit `fe9fd3c` and later without a CUDA synchronization in the teacher walk.
The single-card and PP2 Qwen3-1.7B probes contain 1,792 corresponding hidden
tensors (554,057,728 elements) and are bit-exact, with maximum absolute
difference 0.0.  The PP2 probe also fixed a Singularity 3.7 launcher bug:
comma-separated `CUDA_VISIBLE_DEVICES` values must use the `SINGULARITYENV_`
bridge rather than `--env` (`5c7bad7`).

Primary objective: beat each matching one-card eager time with no meaningful
quality loss, then reduce the remaining gap to the graph target.  Do not mix
setup time into the generation column; report it alongside each new row.
