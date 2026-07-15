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

The initial non-agpul05 cache attempts were also retained: agpul02/04/06 had
old HF ready markers that omitted Qwen3.5-0.8B and failed offline after
104–124 seconds. Their corrected snapshot stage and retry-1 epoch-zero
builds succeeded: each produced the same hash `b632054c01558f61`, 2,071
examples and 24 bfloat16 layers. Teacher-forward time was 219.3/219.9/220.2
seconds; total cache time was 253.7/252.6/252.6 seconds; wrapper wall time
was 313/300/307 seconds on agpul02/agpul04/agpul06 respectively.

## Overnight progression rule

Each scientific 0.8B arm runs six complete dataset-v5 epochs (12,426 answer
visits), publishes its checkpoint, locality certificate, individual Markdown
report, PDF, and completion-ordered report symlink, then becomes eligible for
Pareto selection. Promoted Qwen3.5-4B arms run six epochs and extend to 12 when
measured throughput makes that practical. Selection uses recall, standard
damage, intrusion, layerwise loss/delta dynamics, locality, and elapsed time;
loss alone does not promote a run.
