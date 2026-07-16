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

The 0.8B and 4B models are first. Qwen3.6-27B and Qwen3.6-35B-A3B are admitted
only after their model snapshots, fixed teacher answers, and complete
teacher-hidden caches exist on `agpul06`.

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

## Matrix

| Model | PP | Executor | Partition | Cache | Status | Valid token-events/s | Peak VRAM |
|---|---:|---|---|---|---|---:|---:|
| Qwen3.5-0.8B | 1 | serial | all blocks on GPU 0 | full-v5 cache `b632054c01558f61` | passed/stopped | 2,714.7 | 26.19 GiB allocated / 39.85 GiB reserved |
| Qwen3.5-0.8B | 2 | wavefront | cut after block 12, preflight profile v1 | full-v5 cache `b632054c01558f61` | passed/stopped; locality passed | 3,689.1 (1.359x PP1) | 16.24 / 28.48 GiB allocated on GPU0/GPU1 |
| Qwen3.5-0.8B | 3 | wavefront | preflight `[8,16]`; measured retry `[9,17]` | full-v5 cache `b632054c01558f61` | preflight 4,545.0/s but occupancy rejected; retry pending |  | 12.94 / 12.27 / 25.16 GiB in preflight |
| Qwen3.5-0.8B | 4 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 1 | serial | all blocks on GPU 0 | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 2 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 3 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 4 | wavefront | measured profile pending | pending exact-100 | pending |  |  |

Blank result cells are intentionally unmeasured; no throughput is inferred
from the dependency proof or from a partial run.
