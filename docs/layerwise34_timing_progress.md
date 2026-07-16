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

## Matrix

| Model | PP | Executor | Partition | Cache | Status | Valid token-events/s | Peak VRAM |
|---|---:|---|---|---|---|---:|---:|
| Qwen3.5-0.8B | 1 | serial | all blocks on GPU 0 | building exact-100 | running |  |  |
| Qwen3.5-0.8B | 2 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-0.8B | 3 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-0.8B | 4 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 1 | serial | all blocks on GPU 0 | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 2 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 3 | wavefront | measured profile pending | pending exact-100 | pending |  |  |
| Qwen3.5-4B | 4 | wavefront | measured profile pending | pending exact-100 | pending |  |  |

Blank result cells are intentionally unmeasured; no throughput is inferred
from the dependency proof or from a partial run.
