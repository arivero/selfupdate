# Layerwise 3.4 PPn timing progress

Started: 2026-07-16. Launch host: `agpul06` only. Hardware: four NVIDIA
L40S 46 GB cards. Source implementation commit: `9561faa`.

## Fixed timing contract

- Pipeline protocol remains v3.2; Layerwise 3.4 is the executor identity.
- One complete traversal contains exactly 100 fixed questions selected from
  dataset v5 by corpus, question kind, and expected-answer-length strata.
- B=256, K=16, student-hidden trajectory, flow-mask censorship, Huber hidden
  loss, rank-16 LoRA, immediate state-free SGD, seed 17, and one epoch.
- PP1 is the serial reference. PP2--PP4 use the wavefront executor. The user's
  repeated `PP2` entry is interpreted as PP3, giving PP4/PP3/PP2/PP1.
- Training throughput includes cohort construction, target staging, prompt
  prefill, communication, fill/drain, loss, backward, and writes. Recall and
  report construction remain scientific telemetry but are outside the
  trainer's throughput denominator.
- Each model uses its own fixed teacher-realized answer IDs and hidden cache.
  The 100 question IDs are shared across models.
- Hardware saturation is an explicit PP success gate: after pipeline fill,
  every participating card should sustain 80--90% or greater GPU utilization,
  about 300 W, and approximately 55--60 degrees C. Samples are recorded per
  physical card together with throughput.
  Allocated VRAM alone is not evidence of useful overlap. PP1 applies this
  gate only to its one active card.
- At matched correctness and throughput, lower peak VRAM is better. VRAM is
  reported as headroom/capacity evidence, never used as a utilization proxy.

The exact-100 traversal is the fast correctness and comparative timing matrix.
If it underfills a placement, saturation qualification is promoted to one
complete 2,071-question dataset-v5 traversal, using full B256 cohorts and the
already staged full teacher cache. This promotion was explicitly authorized
after the first 0.8B PP1 trace showed bursty small-model execution.

The 0.8B and 4B models are first, followed by Qwen3.6-27B. Per owner direction,
the model-size ladder stops at 27B in this version; Qwen3.6-35B-A3B is not
admitted. Each model gets at most one final evidence-driven optimization retry,
after which the measured speeds are frozen rather than tuned indefinitely.

## Node-local storage

The matrix uses the existing L40S staging contract documented in
`docs/cache_staging.md`, `docs/training_pipeline_v3.md`, and `AGENTS.md`:

- Hugging Face model snapshots: `/dev/shm/$USER/selfupdate-hf-cache`.
- Pipeline-v3 teacher answers and hidden states:
  `/dev/shm/$USER/selfupdate-teacher-cache-v3` with an atomic ready manifest.
- Python dependency shadow and reusable TorchInductor code:
  `/tmp/$USER/selfupdate-l40-python` and
  `/tmp/$USER/selfupdate-torchinductor`.
- Durable datasets, configs, metrics, logs, reports, and checkpoints remain in
  the repository on Lustre.

All timing commands go through `scripts/l40s_exec.sh`. On this cluster that
means Python is started by the glibc 2.35 dynamic loader with its explicit
library path so the compiled `causal_conv1d` wheel is usable. The wrapper then
restores the pre-module `LD_LIBRARY_PATH` before Python starts, because Triton
launches the host compiler with the host loader; a bare `module load
glibc/2.35` mixes loaders and is not a valid substitute. The slower Torch
causal-convolution backend is diagnostic-only and is not used in this matrix.

Thus lower peak VRAM does not come from paging weights or teacher states over
Lustre during the measured traversal.

## Live ledger

| Time (CEST) | Event | Evidence |
|---|---|---|
| 21:47 | Layerwise 3.4 implementation committed | Commit `9561faa`; config audit, compile checks, CPU serial/wavefront probes, traffic formula, and checkpoint publication guard passed before commit. |
| 21:49 | Launch host verified | `agpul06` GPUs 0--3 each reported 1 MiB used and 0% utilization; no trainer or scheduler process was present. |
| 21:50 | Existing cache inventory checked | Complete full-v5 node caches exist for Qwen3.5-0.8B and Qwen3.5-4B. The stopped score-filtered 4B partial is not reused; this timing matrix receives a new exact-100 cache identity. |
| 23:18 | 0.8B PP1 delegated launch started | `agpul06` GPU0, launcher PID `3663792`; exact-100 cache build followed by one serial LoRA traversal and report. No immediate warning or traceback. |
| 23:20 | Duplicate delegated launch attempts contained | Two earlier cache-builder chains were discovered before GPU work. Their launcher parents were terminated immediately; the node-cache lease permits only the oldest orphan to finish cache publication, and only retained launcher `3663792` can continue into training. No duplicate trainer or run-directory writer was admitted. |
| 23:24 | 0.8B PP1 exact-100 training saturation sampled | First 105 one-second GPU0 samples after the v3.2 contract marker: 37.0% mean / 93% max utilization and 140.4 W mean / 247.5 W max, with 15.9 GiB currently allocated. Valid reference, but underfilled; full-v5 saturation follow-up required. |
| 23:26 | 0.8B PP1 exact-100 completed | One complete epoch; locality passed, checkpoint and report published, 11.3 GiB peak allocated / 15.01 GiB reserved. Classified as an underfill diagnostic, not a saturation pass. |
| 23:30 | Matched full-v5 PP1 identity run started | `agpul06` GPU0, launcher PID `3745456`; exact v3.2 dataset/cache/seed/B256/K16/Huber/LR1e-6/64-user-shard geometry. Cache `b632054c01558f61` reused. Initial sample: 95% utilization, 287.5 W, 46 C, 11.6 GiB used; sustained result pending. |
| 23:36 | Full-v5 PP1 saturation and identity gate passed | After short initial length buckets, later cohorts sustained 96% utilization, 291 W, and 62 C with 41.3 GiB used. The early burstiness was dataset-length underfill; the warmed full-distribution regime meets the operational gate. PP1 uses the unchanged v3.2 block walk and the owner accepted it as the identity reference. |
| 23:39 | Full-v5 PP1 stopped at a cohort boundary | Six cohorts / 1,303 questions / 891,753 token events completed in 328.50 s: **2,714.7 valid token-events/s**, 65,151.9 conceptual writes/s, and 46.83 physical writes/s. Locality passed; partial CE/KL was correctly withheld; graceful checkpoint published. Peak allocated/reserved telemetry: 26.19/39.85 GiB. |
| 23:40 | First PP2 placement rejected before training | Cache reuse passed, then Accelerate rejected a `model.language_model.*` map. Root cause: Qwen3.5 carries vision/audio metadata although its causal-LM class is text-only `model.layers`. Architecture detection now keys on explicit composite model types; no optimizer write occurred. |
| 23:42 | PP2 entered the block walk and exposed a stage-index placement defect | Stage 1 received the detached activation on GPU1 but inherited GPU0 indexing/RoPE tensors. The run stopped before its first write completed. The callback now derives the owning device from its first block, moves only immutable boundary metadata and the detached activation, and transfers target rows one owned layer at a time rather than replicating the full-depth target tensor. Config audit and Python compilation pass; retry pending. |
| 23:47 | Stage-local retry exposed tied-vocabulary placement at the final block | Stage-1 indexing was correctly on GPU1, but Qwen3.5-0.8B's tied embedding/head map also left the final norm on GPU0, pulling the final hidden-loss view back across the boundary. The checkpoint alias remains on GPU0; the final norm now belongs to the final stage and an exact frozen, evaluation-only head replica is colocated there. The replica is fingerprinted but not trained or serialized as an independent checkpoint weight. |
| 23:52 | PP2 passed four cohorts, then OOMed on GPU0 in the first long-memory cohort | The allocator reported 36.98 GiB allocated plus 6.61 GiB reserved-but-unallocated. Torch 2.7 was not reading the launcher's newer allocator-variable alias, and each in-flight wavefront tile retained a full-depth target tensor on GPU0. The L40S wrapper now exports the legacy and current allocator names; PP targets remain pinned on host and transfer one owned layer at a time; queue depth is one (double-buffered processing), as required by the 3.4 contract. B256/K16 and the update law are unchanged. |
| 00:02 | Bounded PP2 retry passed and stopped at the next cohort boundary | Five cohorts / 1,280 questions / 795,510 token events completed in 215.64 s: **3,689.1 valid token-events/s**, a **1.359x** speedup over PP1 (68.0% two-card scaling efficiency). Locality passed; partial CE/KL was withheld. Later long-cohort samples reached 100% on both cards, 267.5/284.6 W, and 60/60 C. Across all 204 measured training samples, GPU0/GPU1 averaged 77.0/86.1% and 240.4/260.9 W; the short cohorts remain visible rather than filtered away. Peak allocated VRAM was only 16.24/28.48 GiB per card (sampler maxima 16.9/29.1 GiB), confirming that the prior OOM was transient-storage/allocator behavior rather than model capacity. |
| 00:08 | Equal-layer PP3 preflight measured, rejected as the final partition | Five cohorts / 1,280 questions / 795,510 token events completed in 175.03 s: 4,545.0 token-events/s (1.675x PP1). The three 8-block ranges had similar callback dispatch times, but live long-cohort samples showed the later stages near 97% while stage 0 fell near 66% and accumulated outbound backpressure. The split is repinned to `[9,17]`, decreasing block count toward final-stage vocabulary evaluation. The first sampler trace was invalid because it mistook the delegated launch shell for the trainer; sampler PID matching now requires `scripts/train.py` as an actual argv element. |
| 00:15 | First measured PP3 repin improved speed but still failed occupancy balance | `[9,17]` completed five cohorts / 795,510 events in 167.50 s: 4,749.2 token-events/s (1.750x PP1). Full training-trace means were only 69.2/61.9/74.4% across GPU0--2, with long-cohort snapshots around 57/57/99%; peak VRAM was 13.33/12.29/24.77 GiB. The final-stage whole-training-set vocabulary metric is materially more expensive than callback dispatch timing indicates. The next pinned profile is `[10,19]` (10/9/5 blocks), moving one block to each upstream stage and two off the final stage. |
| 00:22 | PP3 `[10,19]` isolated a shared host-production bubble | Five cohorts completed at 4,458.9 token-events/s with 13.55/12.62/24.26 GiB peak allocated. Initial long-cohort balance improved to about 77/96/100%, then all three cards collapsed together near 49--58%. Target construction was issuing up to B×n = 6,144 Python `copy_` calls for every rectangular K tile. The exact same pinned target bytes are now assembled with one `torch.stack(..., out=...)` per layer (24 C++ calls); only ragged tail tiles retain row-wise copies. This is a transfer/dispatch optimization, not a change to B, K, target values, loss, or writes. |
| 00:27 | Rectangular-only vector staging was insufficient; ragged staging grouped | Rectangular vectorization measured 4,493.0 token-events/s, only 0.8% above the preceding `[10,19]` run, and later utilization still collapsed. Once any user finishes, the un-compacted PPn cohort takes the ragged path for every later tile. Rows are now grouped by their valid width (at most K=16 groups), reducing ragged target assembly from B×n to at most K×n C++ copies while preserving the identical zero padding and row order. |
| 00:33 | Ragged-group PP3 measured; PP4 profile pinned from physical evidence | `[10,19]` with grouped ragged staging measured 4,615.7 token-events/s. Full-trace utilization still averaged 66.5/65.7/71.7%, so PP3 does not receive the hardware-occupancy pass; the best measured PP3 throughput remains 4,749.2/s at `[9,17]`. PP4 is pinned to `[7,14,21]` (7/7/7/3 blocks): the short final range explicitly budgets the frozen output-distribution evaluation cost observed on PP3. |
| 00:37 | First PP4 partition fed all cards but underloaded the final stage | `[7,14,21]` completed five cohorts at 5,475.7 token-events/s (2.017x PP1; 50.4% four-card efficiency) with only 10.50/11.99/11.93/21.42 GiB peak allocated. Its long-cohort sample was 83/94/85/64% on GPU0--3. The common host starvation seen in PP3 is no longer the immediate limiter, but the 3-block final range over-budgeted vocabulary evaluation. One linear block moves back to the final card: measured profile v2 is `[7,14,20]` (7/7/6/4). The InfiniBand ladder remains locked until the retry sustains the all-card gate. |
| 00:41 | 0.8B tuning closed; moved to 4B | At owner direction, the just-started PP4 v2 repartition was cancelled before measured training. Recorded 0.8B speeds are PP1 2,714.7/s, PP2 3,689.1/s, PP3 best 4,749.2/s, and PP4 5,475.7/s; occupancy caveats remain explicit. The 4B PP1 reference is promoted from exact-100 to the full-v5 traversal with LoRA and B256/K16. |
| 00:42 | 4B PP1 full-v5 reference launched | `agpul06` GPU0, Python PID `3776076`, sampler PID `3775989`; full cache `98bb2aff23e25f93` reused for 2,071 questions. No immediate error; epoch-zero evaluation/load sample was 69% / 167 W / 41 C / 9.1 GiB and is not yet a training-rate measurement. |
| 06:xx | 4B PP1 full-v5 reference completed cleanly | One full traversal: 239,286 valid token-events in 231.196 s, **1,035.0 valid token-events/s** and 53.84 physical local writes/s. Locality certification passed; report and checkpoint were published. Whole-training-set output evaluation: CE-eval-loss 2.59303 and KL-eval-loss 2.52684. Peak allocated/reserved VRAM was 32.91/35.51 GiB on GPU0. This was a completion, not an OOM. |
| 06:xx | 4B parallel PP launch prepared | PP3 is pinned to agpul06 physical GPUs 1/2/3 while PP1 stays on GPU0. PP2 is pinned to agpul05 physical GPUs 1/3, preserving its existing GPU0/2 jobs. Both hosts have the full cache `98bb2aff23e25f93`; host-specific run names prevent shared-Lustre output collisions. |
| 06:xx | Sparse PP2 launch preflight found and corrected | The first agpul05 PP2 attempt did not train or OOM: `CUDA_VISIBLE_DEVICES=1,3` renumbered the physical manifest to 0/1, then the footprint guard incorrectly inspected occupied GPU0 rather than the first PP stage. PPn now stages and guards on its first owning physical device. The corrected launch retains all physical devices visible and assigns work only to configured stages 1 and 3. |
| 06:xx | PP2 sparse-placement isolation strengthened before measurement | The corrected process did allocate its model stages on GPUs 1/3 but also left a 1.0 GiB default-CUDA allocation on busy GPU0. It was stopped before its traversal. Runtime initialization now sets the first owning PP stage as CUDA's default before model load and reports VRAM only for declared physical stages; PP2 will be relaunched only after this isolation fix is committed. The teacher cache is confirmed local shared memory (37 GiB), not Lustre. |
| 06:xx | 4B PP3 full-v5 completed cleanly | Pinned `[11,23]` stages on agpul06 GPUs 1/2/3 completed 239,286 valid token-events in 158.947 s: **1,505.4 token-events/s**, 1.454x PP1 (48.5% three-card efficiency), and 22.03 global physical block writes/s. Locality passed with zero cross-block and frozen-vocabulary leakage; report/checkpoint published. CE-eval-loss 2.59292, KL-eval-loss 2.52675, and final allocated/reserved VRAM 38.54/40.95 GiB (13.88/13.52/13.01 GiB on the three owning cards). |
| 06:xx | 4B PP4 two-run budget opened | PP4 receives one baseline `[8,16,25]` run and exactly one evidence-driven repartition; after that the campaign moves to Qwen3.6-27B. PP1 is retained only as the 4B identity/speed denominator and is not attempted for larger models. An initial delegated command mistakenly landed on occupied agpul05 and failed during model-load warmup; it performed no PP4 training and is excluded. The real run has a host-specific agpul06 identity. |
| 06:xx | 4B PP2 full-v5 completed cleanly | Pinned `[16]` stages on agpul05 physical GPUs 1/3 completed 239,286 valid token-events in 176.072 s: **1,359.0 token-events/s**, 1.313x PP1 (65.7% two-card efficiency), and 35.35 global physical block writes/s. Locality passed with zero cross-block and frozen-vocabulary leakage. CE-eval-loss 2.59285, KL-eval-loss 2.52666, and stage-scoped allocated/reserved VRAM 36.28/37.76 GiB (18.84/18.91 GiB reserved). |
| 06:xx | 4B PP4 baseline launched on the intended host | Host verified as `agpul06`; PID `3783069` owns all and only GPUs 0/1/2/3 under host-specific run `...pp4...agpul06_0123_v1`. Full-v5 cache `98bb2aff23e25f93` reused. Partition `[8,16,25]` is the first of the two permitted PP4 measurements. |
| 06:xx | 4B PP4 baseline completed; sole repartition selected | `[8,16,25]` completed 239,286 events in 160.670 s: **1,489.3 token-events/s**, 1.439x PP1 (36.0% four-card efficiency), slightly slower than PP3. Locality and whole-set CE/KL passed; stage VRAM was 11.04/9.70/10.47/10.82 GiB reserved. In the long cohort, stage compute was 25.96/25.95/24.51/21.05 s; stage 0 also accumulated 16.61 s send backpressure. The only retry is `[7,15,23]`: remove the expensive block-8 cycle from stage 0 while giving the underloaded final stage blocks 24--32. The v1 sampler used the run name rather than the config argv substring and produced only a header, so no occupancy claim is made from it. |
| 06:xx | 4B PP4 sole retry and 27B admission started | PP4 v2 `[7,15,23]` was the only repartition retry on agpul06. A preliminary 27B cache build was mistakenly routed to agpul04; it completed its valid L40S cache, but the launcher was stopped immediately when it advanced to training, before a traversal. Per owner correction, the actual 27B PP2 run belongs on agpul06 and uses a host-specific GPUs0/1 config. |
| 06:xx | 4B PP4 retry completed; 4B tuning closed | `[7,15,23]` completed 239,286 events in 159.200 s: **1,503.1 token-events/s**, 1.452x PP1 (36.3% four-card efficiency), only 0.9% above PP4 v1 and still below PP3's 1,505.4/s. Locality passed; CE/KL were 2.59292/2.52675; reserved VRAM was 9.54/9.72/9.72/13.01 GiB. Training-window GPU utilization averaged only 28.1/32.7/35.7/48.2% (maxima 100/97/97/100%); power averaged 147.5/166.4/161.3/186.7 W. Aggregate stage compute shifted to 68.7/78.5/77.2/81.8 s, making the enlarged final stage the bottleneck while stage 0 accumulated 60.6 s of send backpressure. PP4 fails the saturation gate; PP3 is the practical 4B winner, and no further 4B repartition is attempted. InfiniBand scaling remains locked. |
| 06:xx | 27B corrected agpul06 admission active | The agpul04 launcher and all descendants were terminated before a training traversal. The 52 GiB model snapshot is now staged on agpul06, and PP2 cache builder PID `3790882` owns only GPUs0/1 (36.8/26.9 GiB at the first full-load sample); GPU0 reached 100% and 317 W while GPUs2/3 remained untouched. Cache identity remains `ef92bc2ccff8c62d`; training follows only after the agpul06-local ready manifest publishes. |
| 06:xx | agpul04/agpul05 reserved for teacher-hidden parallel lanes | All four GPUs on both hosts are idle, and the Qwen3.5-0.8B/4B snapshots plus full-v5 caches are already resident. They can run the final modes concurrently after the independent-stage implementation passes locality. Do not launch the current `teacher_hidden` flag as if it were that result: it still constructs an online teacher and retains stage-to-stage scheduling, so it is not yet the requested no-boundary cached-vector executor. |
| 07:4x | 27B cache published; first PP2 admission OOM diagnosed | The numerically local full-v5 cache published 2,071 examples/64 layers at hash `ef92bc2ccff8c62d` after 438.6 s. The first training process then OOMed during prompt prefill before any optimizer write: GPU0 held 43.83 GiB allocated with 9 MiB free when a linear-attention kernel requested 20 MiB. The feasibility retry keeps B256/K16, split `[32]`, LoRA, seed, targets, and writes unchanged while reducing activation-shard users and prefill query width from 16 to 8. This fit correction is not the one permitted throughput optimization retry. |
| 07:4x | Partition-preserving independent teacher-input path implemented locally | The new explicit `teacher_hidden_source: cpu_cache` uses a separately hashed full-prefix cache (`iL = h[L-1]`) and `pp_execution: independent`. It retains the exact ordinary `pipeline_splits` and physical mapping; only each block's activation source changes. Stages keep ordered local tiles but have no cross-stage edge or boundary bytes, enabling a future per-layer teacher/student source mask without repartitioning. Compile, full config audit, cache roundtrip, and ordered-drain probes pass; GPU locality remains required before campaign launch. |
| 07:49 | 27B PP2 fit retry active on agpul06 | Cache `ef92bc2ccff8c62d` was reused; the run emitted the preserved v3.2 contract and entered training with activation shards/prefill chunks of 8. First prefill sample reached 43.1/42.5 GiB on GPUs0/1 without OOM. This is feasibility evidence only; cohort completion, locality, and throughput remain pending. |
| 07:50 | Teacher-input caches published in parallel | agpul04 published the 0.8B 100-question full-input cache `1e992e1638f30bee` (24 layers) and transitioned to PP4 `[7,14,21]`. agpul05 published the 4B cache `fd80e3e399e9b809` (32 layers) and transitioned to PP3 `[11,23]` on physical GPUs1/2/3. These are distinct cache identities and retain the ordinary measured partitions exactly. |
| 07:52 | 27B shard-8 prefill still OOM; shard-4 admission pinned | The second pre-write failure occurred in the linear-attention recurrent-state allocation with 43.78 GiB allocated and 11 MiB free. The next admission uses activation-shard users/prefill query width 4 while preserving PP2 `[32]`, B256/K16, and all scientific semantics. No speed result or optimizer write is claimed from either failed admission. |
| 07:53 | First 4B independent traversal measured; publication gate found stale online-teacher assumption | PP3 `[11,23]` completed all 32 tiles on every stage, 11,020 token events in 42.948 s (**256.59 token-events/s**), and reported zero boundary bytes. Stage compute was 21.26/23.71/19.55 s with 352/384/288 owned physical writes (1,024 global). Whole-100 output evaluation completed (CE 0.03652, KL 0.001382; 4,566 exact answer tokens). The subsequent locality gate dereferenced a nonexistent online teacher and correctly published no checkpoint. Certification now reads the same full-prefix cache and uses each unchanged block owner device; this speed is diagnostic until a rerun passes locality/save. The delegated sampler produced no durable CSV, so no occupancy claim is made. |
| 07:56 | First 0.8B independent traversal diagnostic and duplicate certification containment | The initial PP4 traversal measured **255.12 token-events/s** before hitting the same stale locality gate and publishing no checkpoint. A certification delegate then issued two launches against one run identity; the later process was killed immediately, but the retained process and directory were rejected too because concurrent writers contaminate provenance. It was stopped before a training record, and a unique `cert2` identity is required. |
| 08:0x | 27B PP2 declared memory-infeasible; PP4 promoted | Activation-shard/prefill widths 16, 8, and 4 all failed before the first optimizer write at roughly 43.7--43.8 GiB allocated on GPU0. The invariant B256 causal histories remain resident across shards, so narrower execution shards cannot make PP2 fit. The first feasible candidate is PP4 on agpul06 with initial cuts `[16,32,48]`; its one measured optimization retry remains unused. |

## Matrix

| Model | PP | Executor | Partition | Cache | Status | Valid token-events/s | Peak VRAM |
|---|---:|---|---|---|---|---:|---:|
| Qwen3.5-0.8B | 1 | serial | all blocks on GPU 0 | full-v5 cache `b632054c01558f61` | passed/stopped | 2,714.7 | 26.19 GiB allocated / 39.85 GiB reserved |
| Qwen3.5-0.8B | 2 | wavefront | cut after block 12, preflight profile v1 | full-v5 cache `b632054c01558f61` | passed/stopped; locality passed | 3,689.1 (1.359x PP1) | 16.24 / 28.48 GiB allocated on GPU0/GPU1 |
| Qwen3.5-0.8B | 3 | wavefront | preflight `[8,16]`; v1 `[9,17]`; v2 `[10,19]` | full-v5 cache `b632054c01558f61` | throughput measured; occupancy rejected | 4,749.2 best (1.750x PP1) | best-run v1: 13.33 / 12.29 / 24.77 GiB |
| Qwen3.5-0.8B | 4 | wavefront | v1 `[7,14,21]`; v2 `[7,14,20]` | full-v5 cache `b632054c01558f61` | v1 throughput passed but GPU3 occupancy rejected; v2 pending | 5,475.7 v1 (2.017x PP1) | v1: 10.50 / 11.99 / 11.93 / 21.42 GiB |
| Qwen3.5-4B | 1 | serial | all blocks on GPU 0 | full-v5 cache `98bb2aff23e25f93` | passed; full traversal, locality certified | 1,035.0 | 32.91 / 35.51 GiB allocated/reserved on GPU0 |
| Qwen3.5-4B | 2 | wavefront | `[16]`, GPUs 1/3 | full-v5 cache `98bb2aff23e25f93` | passed; full traversal, locality certified | 1,359.0 (1.313x PP1) | 18.84 / 18.91 GiB reserved |
| Qwen3.5-4B | 3 | wavefront | `[11,23]`, ranges 11/12/9, GPUs 1/2/3 | full-v5 cache `98bb2aff23e25f93` | passed; full traversal, locality certified | 1,505.4 (1.454x PP1) | 13.88 / 13.52 / 13.01 GiB allocated; 40.95 GiB aggregate reserved |
| Qwen3.5-4B | 4 | wavefront | v1 `[8,16,25]`; v2 `[7,15,23]` | full-v5 cache `98bb2aff23e25f93` | scientific checks pass; saturation rejected; tuning closed | 1,503.1 best (1.452x PP1) | v2 9.54 / 9.72 / 9.72 / 13.01 GiB reserved |
| Qwen3.6-27B | 2 | wavefront | `[32]`, GPUs 0/1 | full-v5 cache `ef92bc2ccff8c62d` | fit retry active after prefill OOM at shard 16 | pending | first fit sample 43.1 / 42.5 GiB used |
| Qwen3.5-0.8B | 4 | independent teacher input | `[7,14,21]`, GPUs 0/1/2/3 | 100q full-input cache `1e992e1638f30bee` | active | pending | pending |
| Qwen3.5-4B | 3 | independent teacher input | `[11,23]`, GPUs 1/2/3 | 100q full-input cache `fd80e3e399e9b809` | active | pending | pending |

Blank result cells are intentionally unmeasured; no throughput is inferred
from the dependency proof or from a partial run.

## Pending work, in order

1. Run Qwen3.5-0.8B PP4 with pinned measured profile `[7,14,21]`. The gate is
   sustained 80--90%+ utilization on all four cards in the long cohorts,
   approximately 300 W/card, and the recorded temperature/power/VRAM trace.
   If it fails, diagnose the common host bubble or repartition; do not call
   spikes a pass. Only a passing PP4 unlocks multi-node work.
2. Complete the Qwen3.5-4B PP1/PP2/PP3/PP4 matrix on `agpul06`, starting from
   the existing full-v5 cache `98bb2aff23e25f93`. Use LoRA, B256/K16, the
   same logical token budget, physical telemetry, locality certification,
   graceful cohort-boundary stops, and measured (not layer-count) cuts.
3. Run fresh fixed-cohort serial-versus-wavefront equivalence for each chosen
   placement, including per-layer parameter-delta relative L2/cosine and
   fresh numerical fingerprints. Existing speed runs do not substitute for
   this gate. Preserve failed/preflight run directories under explicit names.
4. Late cache-residency crossover (never silently mixed into the primary
   matrix): add explicit flags and compare (a) online teacher-state
   recomputation / no cache, (b) mmap or CPU-RAM cache with pinned tile
   staging (current baseline), and (c) each stage's owned teacher layers
   resident in GPU VRAM. Match B, K, questions, answers, and token budget;
   report host RAM, H2D bytes/wait, VRAM, power, and token-events/s. Label
   no-cache as teacher recomputation -- it cannot prove cached-hidden
   throughput even if it wins. This is the expected PPn-dependent crossover
   as model/KV residency per card falls. (The unpleasant third regime: 😭.)
5. Final mission after freezing ordinary student-hidden speeds: test fully
   independent teacher-seeded block trajectories (`trajectory_source:
   teacher_hidden`) for 0.8B, 4B, and 27B. Make this a real no-boundary mode:
   each block consumes its cached teacher input vector directly, so no student
   activation is communicated between cards. Re-check cache input ownership,
   host/GPU residency, locality, and exact write semantics; do not revive
   connected credit or final-logit training.
6. Prepare Qwen3.6-27B: stage its snapshot, create fixed teacher answers and a
   complete numerically local teacher cache, profile legal text-tower cuts,
   run the first feasible PPn, and permit exactly one measured optimization
   retry. Then freeze the achieved speed and stop the model ladder. The
   Qwen3.6-35B-A3B black-box MoE arm is deferred beyond this version.
   After teacher-hidden measurements for 0.8B, 4B, and 27B are saved,
   terminate every campaign launcher, sampler, cache helper, and watcher and
   launch no further process. That is the terminal condition for the night.
   Qwen3.6-27B admission starts at PP2 on agpul06 physical GPUs 0/1; PP1 is
   not attempted. Its model snapshot is staged to node-local shared memory.
   The historical H100 hidden cache is provenance, not a pipeline-v3 runtime
   cache: the launcher regenerates and publishes a numerically local L40S
   teacher cache before training.
7. Conditional InfiniBand bonus, only after the PP4 all-card gate: implement
   real rank-owned detached activation transport and cursor/checkpoint
   coordination, benchmark NCCL point-to-point versus host staging, then test
   homogeneous L40S PP6, PP8, and PP12. Do not describe the current
   single-process peer executor as multi-node PP. Separately measure the
   available 12×H100 pool through the cu128 container; never mix H100 and
   L40S stages in one scaling line.
8. Refresh this ledger and each atomic run report after every accepted result.
   At campaign close regenerate layer-loss heatmaps and temporal traces,
   parameter-delta profiles, recall/damage frontier, coverage/provenance,
   `runs/results.md`, and the visibly checked report PDF. Whole-training-set
   CE/KL is published only for complete epochs with exact item/token counts;
   partial timing stops must continue withholding it.
