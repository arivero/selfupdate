# Pareto pipeline-v3.1 training progress

Live operational/scientific ledger begun 2026-07-15. This document is updated
as work progresses; it is independent of the individual `report.md` and
`report.pdf` generated inside each completed training run. Dataset identity is
dataset v5 throughout. The training pipeline is v3.1. Qwen3.5-0.8B is the
mechanics/metaparameter model, Qwen3.5-4B is the first promoted flagship, and
successful recipes then continue along the declared Pareto frontier.

## Timing contract

Every launch records node, physical GPU, source commit, exact config/cache
identity, launcher wall time, and exit state. Stage timings are kept separate:
runtime/model load, compiler work, vLLM answer generation, hidden-state
teacher compute, cache storage/finalization, training-only epoch time,
evaluation time, and report/PDF time. Throughput never folds a cold model load
into generated-token or training-token rates.

## Qwen3.5-0.8B epoch zero

Node `agpul05`, physical L40S GPU0. Exact answers:
`runs/vllm_benchmark_l40s/qwen35_0p8b_fixed4096_exactids_agpul05/responses_bs256.jsonl`.
Published node-local hidden cache:
`/dev/shm/arivero/selfupdate-teacher-cache-v3/Qwen3.5-0.8B-rag_system-remove-b632054c01558f61`.

| stage | commit/runtime | result | measured time and rate |
|---|---|---|---|
| fixed-ceiling answer generation | `710c32e`; vLLM 0.25.0, torch 2.11.0+cu129 | 2,071/2,071; 947,644 tokens; 6.33% hard cuts; mean task score 0.6270 | model/runtime load 318.8 s; `torch.compile` 92.73 s within startup; generation 197.3 s at 4,802 tok/s; launcher wall 636 s |
| exact-token L40S hidden pass | `a40adb6`; torch 2.7.1+cu126 | 24 bfloat16 target layers; 50 GiB; cache hash `b632054c01558f61`; atomic ready publication | teacher forward 259.6 s; D2H 2.61 s; storage 23.12 s; cache write accounting 44.32 s; measured total 291.9 s; launcher wall 334 s |

The earlier mixed short-ceiling cache had 58.52% hard cuts. Restoring the
fixed 4,096-token allowance recovered the expected completion regime and
therefore changed the cache identity. The old cache is not a training target.
Compiler artifacts accidentally used Lustre on this first launch; subsequent
vLLM/Triton/Inductor caches default to node-local `/tmp` (commit `a40adb6`).

## Hybrid B×K certification

Qwen3.5-0.8B has 24 blocks, alternating three `linear_attention` Gated
DeltaNet blocks with one full-attention block. Commit `486f961` separates the
current-chunk `[B,K]` recurrent mask from the full-attention
`[B,1,K,prefix+K]` causal/flow mask and excludes finished cells from the loss
sum. It also prevents intact probes from accidentally masking privileged RAG.

| time | probe | node/GPU | status | timing/result |
|---|---|---|---|---|
| 22:07 | flow B256K1 | `agpul05`/GPU1 | failed before GPU work | 9 s; new base omitted `generation_budget_bucket: 32`, resolving cache `848655…` instead of ready `b63205…` |
| 22:07 | flow B256K16 | `agpul05`/GPU2 | failed before GPU work | same fail-fast identity defect, 9 s |
| 22:09 | flow B256K1 retry | `agpul05`/GPU1 | invalid empty success | cache identity restored, but a misplaced helper made the tile body unreachable; wrapper exited 0 after 10 s without a result |
| 22:09 | flow B256K16 retry | `agpul05`/GPU2 | invalid empty success | same control-flow defect; no tile and no weight update occurred |
| 22:14 | flow B256K1 retry 2 | `agpul05`/GPU1 | passed | commit `87729c7`; 256 events; tile 2.251 s / 113.7 events/s; end-to-end 5.40 events/s; 14.86 GiB peak; 24 physical block writes; launcher 58 s |
| 22:14 | flow B256K16 retry 2 | `agpul05`/GPU2 | passed | commit `87729c7`; 4,096 events; tile 11.569 s / 354.1 events/s; end-to-end 72.4 events/s; 15.68 GiB peak; 24 physical block writes; launcher 67 s |
| 22:17 | intact B256K1 | `agpul05`/GPU1 | passed | loss 1.28e-5–1.09e-4; total LoRA delta 4.07e-6; tile 1.224 s / 209.1 events/s; end-to-end 11.9 events/s; launcher 32 s |
| 22:17 | intact B256K16 | `agpul05`/GPU2 | passed | loss 1.03e-5–1.33e-4; total LoRA delta 1.89e-5; tile 5.193 s / 788.8 events/s; end-to-end 161.1 events/s; launcher 36 s |

The first failures and invalid empty exits are retained because launch/retry
time is part of the operational result. They did not perform a training tile
or modify weights. Commit `87729c7` moves the helper out of `main()` and makes
a missing passed-result payload fail loudly.
After the flow probes pass, intact B256K1/B256K16 establish the numerical-noise
and maximum-compute timing controls before the scientific Wave-A queue opens.

The intact/flow total-delta ratios are about 1:534 at K1 and 1:260 at K16.
Thus the same-runtime uncensored path has a small nonzero bfloat16/batched-kernel
residual but is cleanly separated from the censorship signal. Qwen3.5 hybrid
flow masking is materially slower than intact in this probe: 113.7 versus
209.1 tile events/s at K1 and 354.1 versus 788.8 at K16. Full-epoch estimates
must use these hybrid measurements, not the earlier full-attention 0.6B bound.

### Production promotion

The promoted `causal_bk` trainer keeps length-bucketed cohorts fixed until all
users finish, never refills completed lanes, masks finished cells from loss
and gradient, and rebuilds causal state at every cohort and epoch. Targets are
transferred as one bounded layer-stacked K-window (12 MiB at B256K1; 192 MiB
at B256K16), not as the complete 24-layer answer cohort. Padded prefill queries
receive one harmless key before output zeroing so an all-masked softmax cannot
create NaN state.

The corrected fixed-ceiling corpus has 1,083,913 aligned training-token events
per full epoch. Its nine B≤256 length cohorts range from maximum sequence 236
to 4,954 tokens; the longest full 256-user cohort reaches 4,304 tokens. A
teacher-hidden B256 implementation that retains every layer's complete
uncensored sequence would exceed one L40S on that cohort, so the initial
online-compatible student-hidden campaign streams only the current target
window. Teacher-hidden dreaming remains an explicit later placement/streaming
axis rather than silently reducing B.

## Qwen3.5-0.8B Wave A

The production release gate started at 22:28 CEST on `agpul05`, physical
GPU3, from commit `5bb63dd`. It is the intact student-hidden B256K16 Huber
control at learning rate 1e-5. It reused cache `b632054c01558f61`; epoch-zero
recall and standard-damage evaluation completed before the first update.
By 22:30 it had completed three real cohorts: 768 answers, 170,173 valid
token events, and 1,368 physical per-block writes. This is a live progress
measurement, not a full-epoch throughput estimate. Peak observed allocation
at this point was about 19.6 GiB and no traceback or locality tripwire had
fired.

At 22:29, coordinated builds of the same exact 2,071-example epoch-zero cache
began on `agpul02` GPU0, `agpul04` GPU1, and `agpul06` GPU0. All three
failed before teacher compute after 117, 124, and 104 seconds respectively:
their old HF-cache ready markers covered other named snapshots, but not
Qwen3.5-0.8B. The failed logs are
`runs/v31_qwen35_0p8b_fixed4096_cache_agpul{02,04,06}.log`. At 22:34 the
1.7-GiB Qwen3.5-0.8B snapshot began explicit per-node staging, followed by
retry-1 cache builds. Retry logs have the same stem plus `_retry1.log`.
The model snapshot, cache hash, and ready manifest are all checked before a
trainer may consume a node-local copy.

Wave A contains 16 atomic six-epoch runs: flow-mask Huber and random-fill
Huber at B256K1/B256K16 and learning rates 1e-6, 3e-6, and 1e-5 (12 runs),
intact Huber controls at both K values (2), and flow-mask cosine controls at
both K values and 1e-5 (2). The release gate is one of those 16. The remaining
15 are listed in
`scripts/queue_pareto_v31_qwen35_0p8b_wave_a_20260715.tsv`; every row ends in
its own individual Markdown/PDF report and completion-ordered PDF symlink.

### Release-gate memory finding and repair

The first intact B256K16 production gate (commit `5bb63dd`) completed
epoch-zero evaluation and four real cohorts—1,024 answers, 267,380 aligned
token events, and 2,640 physical block writes—then failed at the next,
longer cohort. At 392 seconds wall time, one block-local backward requested
4.57 GiB while the L40S process already held 39.90 GiB (32.78 GiB allocated,
6.62 GiB reserved). This was a real activation-memory OOM, not a cache,
padding, or loss/locality failure. The result is recorded as an incomplete
release gate and does not count as a scientific arm.

The release repair keeps the logical update exactly B256×K: K16 uses four
fixed 64-user activation shards. Each shard retains its own causal history;
at a given layer/tile all four gradients accumulate at the same pre-write
matrix, then the trainer performs one unaveraged immediate-SGD write. Thus it
does not lower serving B, refill lanes, average gradients, or increase the
optimizer-update count. K1 remains an explicit full B256 activation path.
The repaired gate must finish a full epoch before Wave A is reopened.
It started at 22:48 CEST on agpul05 physical GPU3 from commit `83782cd`,
reusing cache `b632054c01558f61`, under run identity
`pareto_v31_qwen35_0p8b_intact_student_b256k16_huber_lr1e5_s17_shard64_r1`.
Its dedicated worker log is
`runs/v31_qwen35_0p8b_intact_student_b256k16_shard64_r1_agpul05.log`.
At 22:51 it crossed the original OOM boundary: cohort four completed with
1,024 answers, 267,380 token events and 2,640 physical writes. The same
boundary had failed in the old implementation; the repaired process held
18.4 GiB and emitted no error. Training-only time from the v3.1 contract
record to that cohort was about 102 seconds. This is a partial-epoch
observation, not the release throughput certificate.
The first complete epoch then finished cleanly: 2,071 prompts, 1,083,913
aligned answer-token events, and 17,208 physical block writes in 426.988
seconds. That is 2,538.5 aligned token events/s, 4.85 completed prompts/s,
1.68 B256×K16 tiles/s, and 40.30 physical block writes/s; it includes cache
mapping, prompt prefill, all B×K local backwards, and no model-load time.
Epoch-one recall, standard damage, and parameter-delta telemetry also
completed before epoch two began.

The intact null is not behaviorally stationary at K16/LR 1e-5. Overall
recall changed from 0.12150 at epoch zero to 0.10879 after epoch one;
Machado moved 0.09018→0.06240, Quijote chapter 1 moved
0.13345→0.12306, and Quijote chapter 4 remained 0.14086. The vendored
16-item-per-task standard macro stayed 0.4375 with no task-score change.
Mean relative LoRA movement was 2.24e-4 (layer range 5.58e-5–4.84e-4);
mean normalized per-cell gradient norm was 1.04e-3. Therefore
K16/LR 1e-5 is outside the epoch-one no-censorship stability envelope even
though standard-damage sampling does not detect damage. Lower learning-rate
arms must carry the recipe selection.

| intact release epoch | training seconds | aligned events/s | overall recall | standard macro | mean relative LoRA delta | max layer delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 426.988 | 2,538.5 | 0.10879 | 0.4375 | 2.24e-4 | 4.84e-4 |
| 2 | 352.956 | 3,071.0 | 0.11482 | 0.4375 | 3.29e-4 | 7.00e-4 |
| 3 | 349.633 | 3,100.1 | 0.11755 | 0.4375 | 3.89e-4 | 8.14e-4 |
| 4 | 348.570 | 3,109.6 | 0.12126 | 0.4375 | 4.34e-4 | 9.31e-4 |
| 5 | 346.293 | 3,130.0 | 0.11603 | 0.4375 | 4.84e-4 | 1.04e-3 |
| 6 | 350.481 | 3,092.6 | 0.12891 | 0.4375 | 5.27e-4 | 1.14e-3 |

Epoch two shows partial recall recovery rather than monotonic destruction,
but it remains below the 0.12150 epoch-zero value. The intact 1e-5 arm stays
a runtime/null diagnostic, not a candidate recipe.
Epoch three recovers further but remains below epoch zero; movement continues
to grow, so the conclusion is unchanged.
Epoch four is nearly back at epoch zero (difference -0.00023) while retaining
zero sampled macro damage. This makes the null trajectory oscillatory rather
than monotonically destructive, but the nonzero movement still requires
lower-rate null normalization.
The completed epoch-six null ends 0.00742 above epoch zero with unchanged
sampled macro accuracy, after dipping again at epoch five. Therefore positive
raw recall change is not sufficient evidence of censorship learning: a
censored arm must beat the matched intact trajectory or achieve comparable
recall with demonstrably less movement/damage. The final null checkpoint and
all six epoch dynamics are complete; its individual report is generated as a
separate post-training artifact.

The release individual report is complete with no missing artifacts:
`runs/pareto_v31_qwen35_0p8b_intact_student_b256k16_huber_lr1e5_s17_shard64_r1/report.pdf`.
It contains the loss and parameter-delta heatmaps/temporal plots, recall by
corpus including epoch zero, standard-damage trajectory, recall-damage
frontier, signal attribution, and provenance. Its completion-ordered index
link is under `runs/report_v2_index/`. A concurrent duplicate report command
was detected and stopped; the original launcher-owned report completed and
published the verified manifest.

### Wave-A deployment

The first complete release epoch cleared the distributed-launch gate. At
22:59 CEST four schedulers opened one worker per free physical GPU with the
shared queue and lease root:

| host | scheduler PID | GPUs admitted | initial Wave-A workers |
|---|---:|---|---:|
| agpul02 | 59529 | 0,1,2,3 | 4 |
| agpul04 | 3095651 | 1 | 1 |
| agpul05 | 1345628 | 0,1,2 | 3 |
| agpul06 | 535937 | 0,1,3 | 3 |

This is 11 unique Wave-A arms in parallel, plus the separately launched
intact B256K16 release control on agpul05 GPU3. Four older v2 workers retain
the other cards. The global lease audit found one live lease per admitted
GPU and no duplicated experiment process. An apparent fourth agpul05 claim
in the delegated summary belonged to the intentionally terminated pre-repair
scheduler log; the live scheduler has exactly three workers.

The first allocation covers all seven queued K16 censorship/loss/rate arms,
the K1 intact control, and three K1 censorship/loss arms. The remaining four
K1 learning-rate arms stay in the shared queue and will backfill cards as
the first workers publish their individual reports.

At 23:08 the intact-null evidence extended Wave A by two K16 controls at
learning rates 3e-6 and 1e-6. They retain B256, K16, activation shards of 64,
Huber hidden loss, immediate unaveraged SGD, and seed 17; only learning rate
changes. Their priority is above the remaining K1 backfill. Together with the
separately running LR1e-5 release arm, this gives a three-point K16 null curve
against which flow-mask and random-fill movement can be judged. The extension
raises the design from 16 to 18 atomic runs without changing any in-flight
configuration.

The initial non-agpul05 cache attempts were also retained: agpul02/04/06 had
old HF ready markers that omitted Qwen3.5-0.8B and failed offline after
104–124 seconds. Their corrected snapshot stage and retry-1 epoch-zero
builds succeeded: each produced the same hash `b632054c01558f61`, 2,071
examples and 24 bfloat16 layers. Teacher-forward time was 219.3/219.9/220.2
seconds; total cache time was 253.7/252.6/252.6 seconds; wrapper wall time
was 313/300/307 seconds on agpul02/agpul04/agpul06 respectively.

### First Wave-A epoch endpoints

Seven K16 scientific arms completed their first epoch at 2,681--2,728 aligned
token events/s. The table is an early trajectory screen, not a promotion
verdict; every arm continues through six epochs and 12,426 answer visits.

| censorship / loss | LR | recall e0 | recall e1 | standard macro e1 | mean/max relative LoRA delta |
|---|---:|---:|---:|---:|---:|
| flow / Huber | 1e-5 | 0.12150 | 0.12302 | 0.4167 | 2.08e-3 / 6.64e-3 |
| flow / Huber | 3e-6 | 0.12150 | 0.10640 | 0.4375 | 7.94e-4 / 3.26e-3 |
| flow / Huber | 1e-6 | 0.12150 | 0.11360 | 0.4375 | 3.02e-4 / 1.31e-3 |
| flow / cosine | 1e-5 | 0.12150 | 0.11823 | 0.4375 | 2.20e-3 / 4.62e-3 |
| random / Huber | 1e-5 | 0.12150 | 0.10961 | 0.3750 | 2.03e-3 / 4.51e-3 |
| random / Huber | 3e-6 | 0.12150 | 0.10939 | 0.4375 | 7.31e-4 / 2.00e-3 |
| random / Huber | 1e-6 | 0.12150 | 0.10869 | 0.4375 | 2.73e-4 / 7.36e-4 |

Flow/Huber/1e-5 is the only epoch-one recall improvement, but its 16-item per
task damage sample falls by 0.0208 macro and its largest layer moves 9.5 times
farther than the same-rate intact null at epoch one. It is therefore promising
but not yet safe. Random fill is uniformly weak at this cut. Lower-rate intact
controls were added specifically to normalize the censored-arm movement.

Epoch two reverses several epoch-one rankings:

| censorship / loss | LR | recall e2 | standard macro e2 | aligned events/s | mean/max relative LoRA delta |
|---|---:|---:|---:|---:|---:|
| flow / Huber | 1e-5 | 0.13988 | 0.3750 | 3,092 | 3.15e-3 / 7.84e-3 |
| flow / Huber | 3e-6 | 0.12611 | 0.4167 | 3,099 | 1.37e-3 / 4.83e-3 |
| flow / Huber | 1e-6 | 0.10938 | 0.4375 | 3,111 | 5.64e-4 / 2.53e-3 |
| flow / cosine | 1e-5 | 0.12892 | 0.3750 | 3,053 | 3.23e-3 / 6.39e-3 |
| random / Huber | 1e-5 | 0.14076 | 0.4167 | 3,095 | 3.20e-3 / 6.70e-3 |
| random / Huber | 3e-6 | 0.10719 | 0.4167 | 3,109 | 1.31e-3 / 3.43e-3 |
| random / Huber | 1e-6 | 0.11316 | 0.4375 | 3,091 | 5.10e-4 / 1.40e-3 |

The two largest recall values now belong to the aggressive 1e-5 Huber arms,
but both pay sampled standard damage and have roughly 7--8 times the mean
relative movement of the epoch-four intact control. Flow/Huber/3e-6 gives a
smaller positive recall result with smaller movement and smaller sampled
damage. No arm is promoted from this cut: the reversals demonstrate why the
six-epoch trajectories and the full individual reports are required.

### Qwen3.5-4B promotion preparation

The old 4B response file named by the pipeline-v3 base has 2,071 examples but
belongs to the superseded variable-ceiling generation protocol: mean answer
length 41.16 tokens, 87,306 generated tokens total, and 2.12% hard cuts. It
must not seed a v3.1 teacher cache. The 4B base now requires fixed-4,096
generation and names a new artifact path, so the protocol cannot silently
regress even though 4B naturally answers much more concisely than 0.8B.

The Qwen3.5-4B snapshot is now present in `/dev/shm` on agpul02/04/05/06.
The obsolete v2 worker on agpul06 GPU2 was retired only after 12,005 completed
answers. Fresh fixed-4,096 vLLM generation then completed all 2,071 exact IDs:
103,017 generated tokens in 239.17 seconds (430.71 token/s, 8.66 prompt/s),
mean task score 0.92085, mean word accuracy 0.92018, and only 0.145% hard cuts.
The model used 8.61 GiB for weights and a 783,413-token KV cache; engine cold
start was dominated by Python imports plus 65.23 seconds of torch compilation
and 52.61 seconds of profiling. Source artifact:
`runs/vllm_benchmark_l40s/qwen35_4b_fixed4096_exactids_agpul06/responses_bs64.jsonl`.

Before a six-epoch 4B arm is admitted, `scripts/v31_bk_cohort_probe.py` walks
the longest existing full B=256 cohort through the exact production
student-hidden path, including all K tiles and immediate writes. This is a
memory/speed/locality instrument only: it publishes no checkpoint and does not
replace the 12,000-item scientific budget. The first K16 probe starts with
16-user activation shards and LR 1e-6 after the new local teacher cache is
materialized.

The fixed-response 4B cache is now independently published on agpul05 and
agpul06 with the same identity `98bb2aff23e25f93`: 2,071 examples, 32
bfloat16 layers, 36.51 GiB. agpul05 completed teacher forward/cache total in
146.0/166.0 seconds; agpul06 in 167.7/188.1 seconds. Requested teacher batch
64 fit throughout except the natural final three-example tail, and neither
host required an OOM retry. These are node-local copies of the same target
identity, not separate scientific datasets.

### Completed K16 screen

All seven censored K16 Wave-A arms completed six dataset-v5 epochs and their
individual report manifests. The matched LR 1e-5 intact release control is
included for null normalization. Speeds are final-epoch aligned answer-token
events/s; the standard score is the in-training 16-item-per-task sample and is
being replaced by the required full 100-item-per-task evaluation before
promotion.

| censorship / loss | LR | recall e6 | sampled standard macro | events/s | mean/max relative LoRA delta |
|---|---:|---:|---:|---:|---:|
| flow / cosine | 1e-5 | 0.11739 | 0.3750 | 3,034.1 | 5.87e-3 / 1.13e-2 |
| flow / Huber | 1e-5 | 0.12182 | 0.3958 | 3,011.1 | 5.85e-3 / 1.24e-2 |
| flow / Huber | 3e-6 | **0.15356** | 0.3958 | 3,078.8 | 3.01e-3 / 7.54e-3 |
| flow / Huber | 1e-6 | 0.12950 | 0.4167 | 3,041.5 | 1.38e-3 / 4.87e-3 |
| intact / Huber | 1e-5 | 0.12891 | 0.4375 | 3,092.6 | 5.27e-4 / 1.14e-3 |
| random / Huber | 1e-5 | 0.12626 | 0.3333 | 3,064.4 | 5.94e-3 / 1.15e-2 |
| random / Huber | 3e-6 | 0.13508 | 0.3750 | 3,072.0 | 3.08e-3 / 6.47e-3 |
| random / Huber | 1e-6 | 0.10799 | 0.3958 | 3,058.7 | 1.32e-3 / 3.43e-3 |

Flow-mask/Huber/LR 3e-6 is the provisional leader: it is 0.03206 above epoch
zero and 0.02465 above the same-geometry intact release endpoint, with about
half the movement of the aggressive censored arms. Its sampled standard macro
is 0.04167 below the intact control, however, so it is not yet a validated
recipe. The full-corpus base/intact/leader comparison and an independent-seed
replication are promotion gates. The two lower-rate intact K16 controls have
now backfilled the released GPUs and will quantify how much of the apparent
benefit is ordinary uncensored self-distillation.

### First exact 4B production-path probe

The worst full B=256 cohort passed the exact student-hidden K16 path with
16-user activation shards. It contains 65,481 valid answer-token events
(2,095,392 conceptual block-local writes) across 50 K tiles. Prompt prefill
took 31.65 seconds and tile training 74.93 seconds: 873.9 tile token-events/s,
614.4 end-to-end token-events/s, and 106.58 seconds total. Baseline allocation
was 8.20 GiB; peak allocation/reservation was 25.58/32.82 GiB. All vocabulary
weights remained frozen and no checkpoint was published. The remaining memory
margin admits a measured shard-24 probe; shard 32 is not assumed safe from
linear extrapolation.

The shard-24 repetition also passed. Tile time fell to 57.76 seconds and tile
throughput rose to 1,133.8 events/s (668.6/s including a 40.18-second
prefill), while peak allocated/reserved memory was 26.36/36.58 GiB. Relative
to shard 16 this is a 29.7% tile-throughput gain for only 0.79 GiB additional
allocated memory. A final shard-32 measurement is in progress; production
will use the fastest measured width with a safe L40S reservation margin.

### Full-corpus K16 damage gate

The first 100-item-per-task evaluations use identical vendored ARC Easy, ARC
Challenge, and HellaSwag inputs. The native 0.8B base scores
0.560/0.400/0.430 (macro 0.4633). The matched LR 1e-5 intact endpoint scores
0.560/0.420/0.450 (macro 0.4767), so its nonzero self-distillation trajectory
does not damage this standard suite. The provisional flow-mask/Huber/LR 3e-6
leader scores 0.520/0.400/0.370 (macro 0.4300): damage is 0.0333 versus base
and 0.0467 versus the matched intact endpoint. Its recall gain is therefore
not a clean promotion by itself. Full endpoint evaluations of all remaining
completed K16 arms are in progress to locate a better recall--damage point.

The completed endpoint sweep resolves that point:

| endpoint | full standard macro | damage vs base | final recall |
|---|---:|---:|---:|
| flow / cosine / 1e-5 | 0.4400 | 0.0233 | 0.11739 |
| flow / Huber / 1e-5 | 0.4267 | 0.0367 | 0.12182 |
| flow / Huber / 3e-6 | 0.4300 | 0.0333 | 0.15356 |
| flow / Huber / 1e-6 | **0.4600** | **0.0033** | **0.12950** |
| random / Huber / 1e-5 | 0.4000 | 0.0633 | 0.12626 |
| random / Huber / 3e-6 | 0.4300 | 0.0333 | 0.13508 |
| random / Huber / 1e-6 | 0.4567 | 0.0067 | 0.10799 |

Flow-mask/Huber/LR 1e-6 is the clean 0.8B Pareto point: recall is 0.00801
above epoch zero with only 0.0033 macro damage. Random fill at the same rate
has similar standard retention but finishes below epoch zero, while both
3e-6 arms exchange 0.0333 macro accuracy for more recall. The seed-43
replication of the cleanest high-recall candidate remains queued, and the
lower-rate intact controls are required to measure its matched null gain.

The final 4B geometry probe passed at 32-user activation shards: 54.80 seconds
of tile work, 1,194.9 tile events/s, 678.5 events/s including prefill, and
27.04/38.93 GiB peak allocated/reserved memory. This is 5.4% faster than shard
24 and leaves 7.1 GiB of the L40S unreserved; larger shards are not pursued for
the diminishing gain. The audited six-arm 4B screen fixes B=256, K=16, Huber,
immediate SGD, LoRA r=16/alpha=32, and shard 32, crossing intact/flow-mask/
random-fill with learning rates 1e-6 and 3e-6. The three LR 1e-6 anchors began
first on separate GPUs; each must complete six epochs and its individual
report before selection.

agpul04 materialized a third numerically local copy of the same 4B cache while
the first 4B arms loaded: teacher forward 156.3 seconds, cache write 44.1
seconds, total 175.6 seconds, hash `98bb2aff23e25f93`. The second agpul04 arm
waited on the atomic lease and reused the published cache. Four 4B arms are
now active: intact/1e-6, flow/1e-6, random/1e-6, and flow/3e-6. To admit the
fourth arm, only the obsolete v2 worker that had already reached 12,037 items
was retired; two v2 workers below 12,000 items remain protected. The queued
intact/3e-6 and random/3e-6 comparisons will backfill the next eligible cards.

The first 4B epoch establishes both speed and the no-censorship sanity bound.
Flow/1e-6 trained 239,286 aligned events in 273.37 seconds (875.3 events/s,
7.58 completed prompts/s); intact/1e-6 took 279.94 seconds (854.8 events/s,
7.40 prompts/s). Sampled standard macro remained 0.5833 in both. The intact
mean/max relative LoRA delta was only 1.09e-7/2.73e-7, versus
3.31e-5/1.25e-4 for flow mask: uncensored teacher targets are therefore
numerically aligned with this L40S student runtime and do not induce material
training. Recall moved 0.16209→0.15648 in that effectively stationary intact
arm and 0.16209→0.16114 in flow, demonstrating generation/evaluation
oscillation larger than the one-epoch parameter signal. No recipe is selected
from an epoch-one cut. At this speed, a fresh 12-epoch winner is practical
after the six-epoch screen and replication gate.

The full production epoch reports 43.86 GiB reserved, higher than the
single-cohort probe's 38.93 GiB because allocator history spans all bucket
shapes. Shard 32 is still running without OOM, but its actual steady margin is
about 2.2 GiB, not the probe-only 7.1 GiB. Production shard-32 arms therefore
require an exclusive physical GPU; shard 24 remains the conservative fallback
if another runtime introduces additional allocations.

At epoch two the conservative flow arm becomes the first positive 4B signal:
recall 0.16779 versus 0.16209 at epoch zero and 0.16267 in the matched intact
arm, with sampled standard macro still 0.5833. Mean relative movement is
6.60e-5. Flow/3e-6 is still below baseline at 0.15711 despite 1.98e-4 mean
movement; random/1e-6 is effectively neutral at 0.16239 with 5.97e-5
movement. Warm-epoch throughput is 1,036--1,044 events/s. These reversals are
recorded as trajectory evidence only; all arms continue to epoch six.

Report-v2 schema 4 now keeps the per-epoch monitoring sample and the paired
100-item-per-task endpoint as separate evidence. It writes
`full_standard_endpoint.csv`, a dedicated endpoint figure and exact PDF
summary, and marks the endpoint missing until the external evaluation exists.
This repairs the earlier silent preference for the 16-item trajectory when a
stronger full endpoint file was also present.

### First 4B endpoint and scale correction

At epoch six, intact/1e-6 finishes at recall 0.16542 with only 6.56e-7 mean
relative movement. Flow/1e-6 finishes at 0.15167 with 1.98e-4 movement, and
flow/3e-6 at 0.14335 with 5.97e-4; all retain the 0.5833 monitoring macro.
Thus neither completed flow arm is a correct endpoint despite the positive
flow/1e-6 epoch-two excursion. The trajectory is consistent with cumulative
overshoot: six epochs at 3e-7 should produce approximately the movement of
two epochs at 1e-6, the cleanest observed 4B point. Fixed LR 3e-7 flow and
random arms are added; the runtime does not yet implement a decay rule, so
this direct scale measurement precedes any scheduler feature.

The matched 0.8B nulls also completed: intact/1e-6 recall 0.11231 and
intact/3e-6 recall 0.11265, both below the 0.12150 base and with unchanged
monitoring macro. The corresponding censored endpoints therefore have real
matched gains: +0.01720 for flow/1e-6 and +0.04091 for flow/3e-6. Their
seed-43 replication has started.

One delegated random/3e-6 successor omitted SSH and attempted agpul05 GPU0,
where it failed during model placement before any training event. Its two-file
partial directory is preserved under `runs/failed_launches/`; the corrected
successor executes all checks and launch inside explicit SSH to agpul04.

The paired 4B full-standard gate is now available for the first four endpoints:

| endpoint | ARC Easy | ARC Challenge | HellaSwag | macro | damage vs base |
|---|---:|---:|---:|---:|---:|
| base | 0.720 | 0.600 | 0.590 | 0.6367 | 0.0000 |
| intact / 1e-6 | 0.720 | 0.600 | 0.590 | 0.6367 | 0.0000 |
| flow / 1e-6 | 0.700 | 0.610 | 0.590 | 0.6333 | 0.0033 |
| flow / 3e-6 | 0.700 | 0.600 | 0.590 | 0.6300 | 0.0067 |
| random / 1e-6 | 0.710 | 0.610 | 0.590 | 0.6367 | 0.0000 |

Random/1e-6 finishes at recall 0.15865 versus the 0.16542 intact endpoint,
so its exact macro retention does not make it a correct recipe. The first flow
arms likewise fail on matched recall rather than catastrophic standard damage.
Schema-4 individual report refresh completed successfully for all four
endpoints between 00:34:55 and 00:41:13 CEST. Each run now has an individual
PDF plus the explicit 100-item-per-task endpoint table and figure; the
16-item epoch monitor remains separately labelled.

The scale-corrected flow/3e-7 arm started at 00:40:58 CEST on agpul05 physical
GPU2, after moving it off an unnecessary successor wait behind the deliberately
high flow/1e-5 diagnostic. It reused the verified 2,071-example node-local
cache `98bb2aff23e25f93`. Random/3e-7 remains an atomic remote successor on
agpul04 GPU0 so it cannot overlap its random/3e-6 predecessor.

## Overnight progression rule

Each scientific 0.8B arm runs six complete dataset-v5 epochs (12,426 answer
visits), publishes its checkpoint, locality certificate, individual Markdown
report, PDF, and completion-ordered report symlink, then becomes eligible for
Pareto selection. Promoted Qwen3.5-4B arms run six epochs and extend to 12 when
measured throughput makes that practical. Selection uses recall, standard
damage, intrusion, layerwise loss/delta dynamics, locality, and elapsed time;
loss alone does not promote a run.
