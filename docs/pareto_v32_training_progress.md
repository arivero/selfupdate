# Pipeline v3.2 Pareto training progress

Started: 2026-07-16 13:20 CEST. Dataset: v5 (`examples_v5rs_window.jsonl`).
Training pipeline: v3.2. This document is updated during the campaign; each
completed training also produces its own run-local Markdown/PDF report.

## Whole-training-set output evaluation contract

From the next v3.2 launches, every completed epoch records `CE-eval-loss`
against the teacher's realized answer tokens and `KL-eval-loss` in the
teacher-to-student direction. This covers every answer token in all training
examples once during the ordinary epoch traversal; it is not a validation
subset. Both are evaluation-only diagnostics computed from detached final
states through the frozen head. They NEVER enter backward or an optimizer and
have optimizer weight zero. Individual reports show the two curves, item and
token counts, coverage, and the non-training flags. Historical runs without
these rows remain explicitly missing rather than receiving inferred values.

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
| 4 | Finished-row KV retained | rows are compacted, never replaced, at K-tile boundaries across cache and shard state | numerical regression pending |
| 5 | DynamicCache quadratic append/copy | hybrid-aware Transformers `StaticCache`, preallocated to prompt + maximum answer | numerical regression pending |
| 6 | GPU boolean-index synchronization | CPU-derived valid flat indices and device `index_select`; no GPU `nonzero` | numerical regression pending |
| 7 | Pageable blocking target restaging | one reusable pinned n x B-shard x K x H staging buffer with nonblocking H2D | throughput measurement pending |
| 8 | Full target re-padding per cohort | B/K path collates metadata only and lazily fills the current K window from mmap-backed Items | host RSS/epoch timing pending |
| 9 | Divide/remultiply kernel churn | tile loss and gradient sums accumulate directly; division occurs once at epoch telemetry | numerical regression pending |
| 10 | O(B S^2) prefill-mask transient | static-cache prefill runs in configurable query chunks (default 64) | peak-memory measurement pending |

## Allocation state at start

- Slurm 418174, agpul02: 48-hour allocation expiring at approximately 13:25
  CEST; excluded from new campaign planning.
- Slurm 418791, agpul04-agpul06: approximately 25.5 hours remained at 13:20
  CEST. New v3.2 arms must fit this horizon and use one model per GPU.

## Implementation record

- 13:22: session-cancellation audit found no partial v3.2 edits.
- 13:30: first implementation pass completed; Python compilation and the
  full config audit passed. GPU numerical regression and measured smoke remain
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

## Five-hour screen decision view (live reconciliation 18:54 CEST)

### What was trained

Eleven of twelve arms completed and published checkpoints plus individual
Markdown/PDF reports. They comprise five 0.8B 40-epoch runs and six 4B
24-epoch runs over dataset v5 with pipeline v3.2, B256/K16 immediate SGD,
student-hidden trajectories, and LoRA. The only running trainer is the 0.8B
full-vocabulary `lens_kl` arm on `agpul06` GPU3 (PID 3605347, 24,090 MiB and
100% GPU at 18:50). It completed epoch 8 at 18:53 and is entering epoch 9.
All four GPUs on `agpul04` and `agpul05`, plus GPUs 0-2 on
`agpul06`, are idle. Its stable cadence is approximately 38 minutes per
training epoch plus endpoint telemetry, projecting completion of epoch 12 at
approximately 21:25-21:35 CEST. Its current report is explicitly incomplete
and will be regenerated after its `done` row and checkpoint appear.

### Speed of every launch

This is the complete twelve-launch speed ledger, not a fastest-only summary.
The primary rate is weighted from every completed `v3_throughput` row as total
aligned-token events divided by total training seconds. Evaluation time is
excluded. `Train s` and `s/epoch` therefore describe the trainer traversal;
recall/standard telemetry and report generation add wall time outside those
numbers. The range is the minimum-to-maximum complete-epoch rate and exposes
cold/slow epochs rather than hiding them in the aggregate. Different models
generated different answer lengths, so total event counts differ even though
every epoch traverses all 2,071 examples.

| Host/GPU | Run / arm | State | Total events | Train s | s/epoch | Weighted events/s | Epoch range events/s |
|---|---|---:|---:|---:|---:|---:|---:|
| agpul05/2 | [0.8B flow cosine 1e-6](../runs/pareto_v32_qwen35_0p8b_flow_student_b256k16_cosine_lr1e6_s17_e40/report.md) | done 40/40 | 43,356,520 | 11,307.2 | 282.7 | 3,834.4 | 3,465.3-3,886.9 |
| agpul06/2 | [0.8B flow Huber 1e-6](../runs/pareto_v32_qwen35_0p8b_flow_student_b256k16_huber_lr1e6_s17_e40/report.md) | done 40/40 | 43,356,520 | 11,110.5 | 277.8 | 3,902.3 | 2,929.7-3,963.9 |
| agpul06/0 | [0.8B flow Huber 3e-6](../runs/pareto_v32_qwen35_0p8b_flow_student_b256k16_huber_lr3e6_s17_e40/report.md) | done 40/40 | 43,356,520 | 11,156.0 | 278.9 | 3,886.4 | 3,526.9-3,919.9 |
| agpul06/3 | [0.8B flow full lens KL 1e-6](../runs/pareto_v32_qwen35_0p8b_flow_student_b256k16_lenskl_lr1e6_s17_e12/report.md) | running 8/12; entering epoch 9 | 8,671,304 | 18,143.7 | 2,268.0 | 477.9 | 475.9-480.0 |
| agpul06/1 | [0.8B intact Huber 1e-6](../runs/pareto_v32_qwen35_0p8b_intact_student_b256k16_huber_lr1e6_s17_e40/report.md) | done 40/40 | 43,356,520 | 11,145.7 | 278.6 | 3,890.0 | 3,497.9-3,936.7 |
| agpul05/3 | [0.8B random-context Huber 3e-6](../runs/pareto_v32_qwen35_0p8b_random_student_b256k16_huber_lr3e6_s17_e40/report.md) | done 40/40 | 43,356,520 | 11,089.1 | 277.2 | 3,909.8 | 3,500.9-3,952.0 |
| agpul04/2 | [4B flow cosine 1e-6](../runs/pareto_v32_qwen35_4b_flow_student_b256k16_cosine_lr1e6_s17_e24/report.md) | done 24/24 | 5,742,864 | 4,403.0 | 183.5 | 1,304.3 | 1,052.6-1,336.6 |
| agpul04/0 | [4B flow Huber 1e-6](../runs/pareto_v32_qwen35_4b_flow_student_b256k16_huber_lr1e6_s17_e24/report.md) | done 24/24 | 5,742,864 | 4,389.7 | 182.9 | 1,308.3 | 1,060.0-1,335.0 |
| agpul04/1 | [4B flow Huber 3e-6](../runs/pareto_v32_qwen35_4b_flow_student_b256k16_huber_lr3e6_s17_e24/report.md) | done 24/24 | 5,742,864 | 4,387.0 | 182.8 | 1,309.1 | 1,058.3-1,343.7 |
| agpul05/1 | [4B flow sampled-vocabulary cosine-256 1e-6](../runs/pareto_v32_qwen35_4b_flow_student_b256k16_vocabcos256_lr1e6_s17_e24/report.md) | done 24/24 | 5,742,864 | 4,635.0 | 193.1 | 1,239.0 | 1,125.8-1,256.5 |
| agpul05/0 | [4B intact Huber 1e-6](../runs/pareto_v32_qwen35_4b_intact_student_b256k16_huber_lr1e6_s17_e24/report.md) | done 24/24 | 5,742,864 | 4,327.6 | 180.3 | 1,327.0 | 1,175.8-1,350.9 |
| agpul04/3 | [4B random-context Huber 1e-6](../runs/pareto_v32_qwen35_4b_random_student_b256k16_huber_lr1e6_s17_e24/report.md) | done 24/24 | 5,742,864 | 4,412.1 | 183.8 | 1,301.6 | 1,055.1-1,327.2 |

The five ordinary 0.8B arms are within 2.0% of one another and the six 4B
arms within 7.1%; most of the 4B spread is the sampled-vocabulary metric.
Full `lens_kl` through the vocabulary at every layer is 8.2 times slower than
ordinary 0.8B Huber. It is useful as a diagnostic arm, not a practical primary
loss at its current implementation cost.

### Which learned most

The old single “overall recall” column concealed the answer. It is the equal
mean of next-phrase, previous-phrase, and cloze word accuracy over Machado,
Quijote chapter 1, and Quijote chapter 4. The component deltas below are all
measured at the epoch with the best *post-training* overall score for that
run, against the same model's epoch-zero evaluation. These are fast monitors
with 8 prompts per task per corpus (72 prompt evaluations per epoch), not the
full-corpus checkpoint evaluation.

| Model / arm | Best epoch | Δ overall | Δ next phrase | Δ previous phrase | Δ cloze | Final Δ overall |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B flow Huber 3e-6 | 6 | +0.03672 | -0.00460 | -0.00505 | +0.11979 | -0.02033 |
| 0.8B flow Huber 1e-6 | 18 | +0.03472 | +0.00013 | -0.01055 | +0.11458 | +0.00287 |
| 0.8B flow full KL 1e-6 (epoch 8 cut; live) | 4 | +0.03311 | -0.01497 | -0.00548 | +0.11979 | -0.00303 at epoch 8 |
| 0.8B flow cosine 1e-6 | 17 | +0.02449 | -0.02317 | -0.00231 | +0.09896 | -0.00316 |
| 0.8B random Huber 3e-6 | 7 | +0.01858 | -0.02310 | +0.00072 | +0.07812 | -0.01013 |
| 0.8B intact Huber 1e-6 | 24 | +0.01378 | -0.00579 | +0.00547 | +0.04167 | +0.00906 |
| 4B flow sampled-vocabulary cosine 1e-6 | 2 | +0.00554 | +0.00002 | +0.01660 | 0.00000 | -0.02385 |
| 4B intact Huber 1e-6 | 16 | +0.00256 | +0.00935 | -0.00167 | 0.00000 | -0.00299 |
| 4B flow cosine 1e-6 | 4 | +0.00210 | -0.00042 | +0.00671 | 0.00000 | -0.00643 |
| 4B flow Huber 1e-6 | 4 | -0.00043 | -0.01220 | +0.01091 | 0.00000 | -0.01834 |
| 4B random Huber 1e-6 | 6 | -0.00332 | -0.01653 | +0.00657 | 0.00000 | -0.01163 |
| 4B flow Huber 3e-6 | 10 | -0.00427 | -0.02265 | -0.00580 | +0.01562 | -0.02567 |

The aggregate leader is therefore **not evidence of general phrase recall**:
its entire positive mean comes from cloze while both directional phrase tasks
decline. The strongest next-phrase gain is the 4B intact control (+0.00935),
not a censorship arm. The best censored next-phrase gain anywhere in its
trajectory is only +0.00242 for 4B sampled-vocabulary cosine at epoch 5, where
its overall score is already 0.00185 below epoch zero. The clearest censored
signal is previous-phrase recall at 4B, but its component maxima are not
simultaneous with good next/cloze scores. This campaign found task-selective
movement, not yet a correct all-three-task training recipe.

### Which was most destructive

These are only the 16-item-per-task online monitors. The required
100-item-per-task standard-benchmark evaluations at epoch zero and at each
checkpoint have not been run, so the ordering is provisional.

| Model / arm | Damage at best-overall epoch | Worst observed damage | Final observed damage |
|---|---:|---:|---:|
| 0.8B random Huber 3e-6 | -0.0417 | -0.1042 | -0.1042 |
| 0.8B flow Huber 3e-6 | -0.0417 | -0.1042 | -0.0833 |
| 0.8B flow full KL 1e-6 (epoch 8 cut; live) | -0.0833 | -0.1042 | -0.0833 at epoch 8 |
| 0.8B flow Huber 1e-6 | -0.0625 | -0.0833 | -0.0625 |
| 0.8B flow cosine 1e-6 | -0.0417 | -0.0625 | -0.0208 |
| 0.8B intact Huber 1e-6 | 0.0000 | 0.0000 | 0.0000 |
| 4B flow Huber 3e-6 | 0.0000 | -0.0417 | -0.0417 |
| 4B flow Huber 1e-6 | 0.0000 | -0.0208 | 0.0000 |
| 4B flow cosine 1e-6 | 0.0000 | -0.0208 | 0.0000 |
| 4B sampled-vocabulary cosine / intact / random | 0.0000 | 0.0000 | 0.0000 to +0.0208 |

### Interpretation boundary

- Systems result: v3.2 sustained about 3.9k token-events/s at 0.8B and
  1.31k/s at 4B, roughly 27% above the corresponding v3.1 campaign rates,
  while completing the formerly fatal long cohort. The 4B runs used roughly
  27-32 GiB instead of the v3.1 high-40-GiB regime.
- Optimization result: fixed-rate arms generally peaked early and then
  degraded. Forty versus twenty-four epochs did not rescue them; the next
  screen should taper around the measured peak rather than continue fixed
  writes.
- Scientific result: no arm has demonstrated joint learning of next phrase,
  previous phrase, and cloze. The large 0.8B aggregate gains are cloze-only;
  next/previous recall is unchanged or worse at those checkpoints. The 4B
  sampled-vocabulary arm has the best censored aggregate gain, driven by
  previous-phrase recall, but the gain is small and transient.
- Damage is unresolved rather than clean. The 16-item monitor identifies
  likely destructive 0.8B arms, while the 100-item-per-task standard
  evaluations at epoch zero and at the checkpoints are still missing.
- The reports currently save only the final checkpoint, so the early best
  epochs in this table are measurements, not recoverable promoted artifacts.
- Production stability is established, but the numerics-adjacent static-cache,
  compaction, and reduction changes still require the review-requested exact
  old-versus-new numerical regression. No scientific comparison to v3.1 should be claimed before
  that gate.

## Recommended next screen (decision draft; not launched)

The next screen should not repeat another fixed-rate 24/40-epoch rectangle.
The present evidence says those runs cross an early useful region and then
continue writing until the final checkpoint is worse. It also says the large
0.8B aggregate gains are mostly cloze, while the only notable censored 4B
signal is sampled-vocabulary cosine on previous-phrase recall. The next
campaign must therefore preserve directional recall components, replicate the
one 4B signal, and use the newly implemented whole-training-set
`CE-eval-loss`/`KL-eval-loss` diagnostics.

First run one six-epoch 0.8B intact Huber gate on one idle GPU with the new
output evaluator. It exercises the entire training calculation on all 2,071
examples per epoch while testing the strongest null condition: uncensored
student and teacher inputs. Require exactly 2,071 examples and the complete
teacher-realized answer-token count in every output-evaluation row,
`validation_subset=false`, `used_for_backward=false`, optimizer weight zero,
finite CE/KL, and no unexpected growth. Measure its throughput before filling
the other GPUs; exact final-output CE/KL is expected to cost something, and
that cost belongs in the next ledger rather than being guessed. If it passes,
this gate is promoted as the 0.8B intact-Huber row of the grid below rather
than repeated as a thirteenth launch.

If that gate passes, use the remaining cards for this twelve-arm decision
grid (the twelfth can start when the current full-`lens_kl` arm releases
agpul06/3):

| Model | Censorship | Local training metric | Seed(s) | LR policy | Purpose |
|---|---|---|---|---|---|
| 4B | flow | sampled-vocabulary cosine-256 | 17, 43 | 1e-6; taper after epoch 2 | replicate the only positive censored 4B aggregate/previous-phrase signal |
| 4B | intact | sampled-vocabulary cosine-256 | 17 | matched taper | matched no-censorship control missing from the first screen |
| 4B | random context | sampled-vocabulary cosine-256 | 17 | matched taper | distinguish deletion/attention censorship from random replacement |
| 4B | flow | hidden cosine | 17 | 1e-6; taper after epoch 4 | preserve the cheaper semantic geometry near its measured peak |
| 4B | intact | hidden cosine | 17 | matched taper | matched control for the cosine arm |
| 0.8B | flow | Huber | 17, 43 | 3e-6; taper after epoch 6 | test whether early stopping/taper preserves the cloze gain and seed stability |
| 0.8B | intact | Huber | 17 | matched 3e-6 taper | measure censorship-conditioned movement under the same write schedule |
| 0.8B | random context | Huber | 17 | matched 3e-6 taper | compare the two censorship treatments directly |
| 0.8B | flow | sampled-vocabulary cosine-256 | 17 | 1e-6; taper after epoch 5 | cheap scale probe of the 4B candidate metric |
| 0.8B | intact | sampled-vocabulary cosine-256 | 17 | matched taper | null/control for that scale probe |

The concrete proposed schedules are: 4B sampled-vocabulary arms, 12 epochs
with multipliers `[1, 1, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
0.1]`; 4B hidden-cosine arms, 12 epochs with `[1]` for epochs 1-4, `[0.3]`
for 5-8, and `[0.1]` for 9-12; 0.8B Huber arms, 18 epochs with `[1]` for
1-6, `[0.3]` for 7-12, and `[0.1]` for 13-18; 0.8B sampled-vocabulary arms,
15 epochs with `[1]` for 1-5, `[0.3]` for 6-10, and `[0.1]` for 11-15. These
are explicit `epoch_piecewise` multipliers on the base LR shown in the table.

The ten table rows expand to twelve launches because both the 4B flow
sampled-vocabulary arm and the 0.8B flow-Huber arm have two seeds. Checkpoint
retention must include the pre-taper boundary and the best
monitor epoch; saving only the final checkpoint made the first screen's early
best measurements unrecoverable. Promotion requires directional evidence:
next-phrase and previous-phrase results are reported separately from cloze,
not hidden inside their equal-weight mean. Full 100-item-per-task standard
evaluation remains required before any arm is called clean or Pareto-optimal.
