# Pipeline v3.2 Pareto training progress

Started: 2026-07-16 13:20 CEST. Dataset: v5 (`examples_v5rs_window.jsonl`).
Training pipeline: v3.2. This document is updated during the campaign; each
completed training also produces its own run-local Markdown/PDF report.

## Purpose

Pipeline v3.2 supersedes the v2, v3.0, and v3.1 campaign queues. It preserves
the v3.1 logical B x K immediate-SGD update while applying every confirmed
finding in `code_review_v31_trainer_ultra_2026-07-16.md` before another Pareto
search. B is simultaneous-user serving parallelism; K is within-answer
lookahead. K > 1 remains explicitly stale/speculative, never described as
exact next-token online learning.

## Review closure ledger

| # | Confirmed v3.1 issue | v3.2 treatment | Verification |
|---:|---|---|---|
| 1 | Sharding silently disabled | `causal_bk` rejects zero; all old unpinned YAMLs pinned; deterministic pre-load shape/post-load VRAM guard | config audit + launch guard |
| 2 | Teacher-hidden GPU residency unbounded | full teacher inputs are stored in host-pinned memory and only one B-shard x K x H layer window returns to GPU | smoke + memory telemetry pending |
| 3 | Whole teacher cache retained by Python | bounded LRU (`cache.item_cache_items`, default 64); safetensors mmap/page cache remains the backing cache | host RSS measurement pending |
| 4 | Finished-row KV retained | rows are compacted, never replaced, at K-tile boundaries across cache and shard state | A/B certification pending |
| 5 | DynamicCache quadratic append/copy | hybrid-aware Transformers `StaticCache`, preallocated to prompt + maximum answer | A/B certification pending |
| 6 | GPU boolean-index synchronization | CPU-derived valid flat indices and device `index_select`; no GPU `nonzero` | A/B certification pending |
| 7 | Pageable blocking target restaging | one reusable pinned n x B-shard x K x H staging buffer with nonblocking H2D | throughput measurement pending |
| 8 | Full target re-padding per cohort | B/K path collates metadata only and lazily fills the current K window from mmap-backed Items | host RSS/epoch timing pending |
| 9 | Divide/remultiply kernel churn | tile loss and gradient sums accumulate directly; division occurs once at epoch telemetry | A/B certification pending |
| 10 | O(B S^2) prefill-mask transient | static-cache prefill runs in configurable query chunks (default 64) | peak-memory measurement pending |

## Allocation state at start

- Slurm 418174, agpul02: 48-hour allocation expiring at approximately 13:25
  CEST; excluded from new campaign planning.
- Slurm 418791, agpul04-agpul06: approximately 25.5 hours remained at 13:20
  CEST. New v3.2 arms must fit this horizon and use one model per GPU.

## Implementation record

- 13:22: session-cancellation audit found no partial v3.2 edits.
- 13:30: first implementation pass completed; Python compilation and the
  full config audit passed. GPU A/B certification and measured smoke remain
  gates before campaign launch.
- 13:31: the first delegated 0.6B gate on agpul06 GPU2 exited before model
  load because the offline node-local Hugging Face cache did not contain
  Qwen3-0.6B. This is staging evidence, not a trainer result. The gate moved
  to cached Qwen3.5-0.8B on the same node.
- 13:42: Qwen3.5-0.8B v3.2 retry passed the production startup gate and
  completed the historically fatal 780-prompt x 844-answer cohort. At the
  fourth cohort it had processed 267,380 token-events. The deterministic
  guard estimated 4,212,400,128 bytes (3.92 GiB) incremental peak against
  45,476,085,760 free bytes after model load; observed process placement was
  38,598 MiB on the L40S. No OOM or traceback occurred.

## Launch/results ledger

The first promoted run is
`pareto_v32_qwen35_0p8b_flow_student_b256k16_huber_lr1e6_s17_e40` on
agpul06 GPU2. A twelve-card, approximately five-hour screen is being launched
after retiring the superseded queues: six 0.8B arms (Huber learning-rate pair,
hidden cosine, random-context, intact control, and full KL-lens speed arm) and
six 4B arms (Huber learning-rate pair, hidden cosine, random-context, intact
control, and sampled-vocabulary cosine). All use B256/K16 and immediate SGD;
the KL arm uses an 8-user execution shard solely to bound vocabulary logits.

### Five-hour screen placement (13:47 CEST)

| Host/GPU | Model | Arm |
|---|---|---|
| agpul04/0 | 4B | flow Huber, LR 1e-6, 24 epochs |
| agpul04/1 | 4B | flow Huber, LR 3e-6, 24 epochs |
| agpul04/2 | 4B | flow hidden cosine, LR 1e-6, 24 epochs |
| agpul04/3 | 4B | random-context Huber, LR 1e-6, 24 epochs |
| agpul05/0 | 4B | intact Huber control, LR 1e-6, 24 epochs |
| agpul05/1 | 4B | flow sampled-vocabulary cosine-256, LR 1e-6, 24 epochs |
| agpul05/2 | 0.8B | flow hidden cosine, LR 1e-6, 40 epochs |
| agpul05/3 | 0.8B | random-context Huber, LR 3e-6, 40 epochs |
| agpul06/0 | 0.8B | flow Huber, LR 3e-6, 40 epochs |
| agpul06/1 | 0.8B | intact Huber control, LR 1e-6, 40 epochs |
| agpul06/2 | 0.8B | flow Huber, LR 1e-6, 40 epochs |
| agpul06/3 | 0.8B | flow full KL lens, LR 1e-6, 12 epochs |

All twelve emitted `pipeline_v32_contract` and completed at least one real
cohort before agent detachment. Eleven had completed 3-4 cohorts; the slower
full-vocabulary KL arm had completed one. Every physical L40S showed an active
trainer and nonzero utilization; no launch log contained a traceback, OOM, or
nonzero exit. The six 4B processes used approximately 27-32 GiB at this point,
the ordinary 0.8B processes approximately 38.6 GiB, and the 8-user KL process
approximately 9.4 GiB. Slurm allocation 418791 had about 25 hours remaining,
well beyond this screen's planned horizon.

## Five-hour screen status (17:54 CEST)

Eleven arms completed cleanly and generated run-local Markdown/PDF reports;
the full KL-lens arm remained live at epoch 7. Mean recall is the report's
three-corpus `overall_word_acc`. Damage below is the contemporaneous 16-item
monitoring delta, not the still-missing paired full-standard endpoint.

| Model / arm | Extent | Mean tok-events/s | Best recall (epoch) | Gain from epoch 0 | Monitor damage at best | Final recall |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B flow Huber 1e-6 | 40, done | 3,911 | 0.15622 (18) | +0.03472 | -0.0625 | 0.12437 |
| 0.8B flow Huber 3e-6 | 40, done | 3,887 | 0.15821 (6) | +0.03672 | -0.0417 | 0.10117 |
| 0.8B flow cosine 1e-6 | 40, done | 3,836 | 0.14599 (17) | +0.02449 | -0.0417 | 0.11834 |
| 0.8B random Huber 3e-6 | 40, done | 3,911 | 0.14008 (7) | +0.01858 | -0.0417 | 0.11137 |
| 0.8B intact Huber 1e-6 | 40, done | 3,891 | 0.13528 (24) | +0.01378 | 0.0000 | 0.13056 |
| 0.8B flow full KL 1e-6 | 6 complete, epoch 7 live | 478 | 0.15461 (4) | +0.03311 | -0.0833 | 0.14458 at epoch 6 |
| 4B flow sampled-vocab cosine 1e-6 | 24, done | 1,240 | 0.16763 (2) | +0.00554 | 0.0000 | 0.13824 |
| 4B flow hidden cosine 1e-6 | 24, done | 1,307 | 0.16418 (4) | +0.00210 | 0.0000 | 0.15565 |
| 4B flow Huber 1e-6 | 24, done | 1,311 | 0.16166 (4) | -0.00043 | 0.0000 | 0.14375 |
| 4B flow Huber 3e-6 | 24, done | 1,312 | 0.15781 (10) | -0.00427 | 0.0000 | 0.13642 |
| 4B random Huber 1e-6 | 24, done | 1,304 | 0.15877 (6) | -0.00332 | 0.0000 | 0.15046 |
| 4B intact Huber 1e-6 | 24, done | 1,328 | 0.16464 (16) | +0.00256 | 0.0000 | 0.15909 |

### Interpretation boundary

- Systems result: v3.2 sustained about 3.9k token-events/s at 0.8B and
  1.31k/s at 4B, roughly 27% above the corresponding v3.1 campaign rates,
  while completing the formerly fatal long cohort. The 4B runs used roughly
  27-32 GiB instead of the v3.1 high-40-GiB regime.
- Optimization result: fixed-rate arms generally peaked early and then
  degraded. Forty versus twenty-four epochs did not rescue them; the next
  screen should taper around the measured peak rather than continue fixed
  writes.
- 0.8B shows a censorship-conditioned signal: flow Huber's best gain exceeds
  the intact control by about 0.021-0.023. It is not yet a clean frontier
  result because the 16-item damage monitor is negative and full-standard
  endpoints are missing.
- 4B has no robust winner yet. Sampled-vocabulary cosine is the only flow arm
  above the intact-control gain, and only by about 0.003 at epoch 2; this
  requires seed replication and a matched intact sampled-vocabulary control.
- The reports currently save only the final checkpoint, so the early best
  epochs in this table are measurements, not recoverable promoted artifacts.
- Production stability is established, but the numerics-adjacent static-cache,
  compaction, and reduction changes still require the review-requested exact
  A/B certificate. No scientific comparison to v3.1 should be claimed before
  that gate.
