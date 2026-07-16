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
