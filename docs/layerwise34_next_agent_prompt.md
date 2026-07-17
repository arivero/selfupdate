# Prompt for the next agent: analyze the Layerwise 3.4 overnight runs

Work in `/fs/agustina/arivero/supercomplex/selfupdate_lw`. Read `AGENTS.md`,
`docs/layerwise34_overnight_handoff.md`,
`docs/layerwise34_timing_progress.md`, `docs/runtime.md`, and the two relevant
2026-07-17 entries in `issues.md` before acting. Treat repository files as the
only durable memory from the previous agent.

## Non-negotiable operating rules

- Do not recursively grep, find, stat, or enumerate `runs/` or `/dev/shm`.
  Lustre will punish it. Use only the exact paths listed below, `git ls-files`
  for source discovery, and `scripts/pipeline_tail.sh` for the exact launcher
  logs.
- These trainers use physical GPU ids from their configs. Do not narrow
  `CUDA_VISIBLE_DEVICES` for the `[2,3]` jobs; that renumbers them to logical
  0/1 and runtime correctly rejects the physical mapping.
- The runs have `epochs: 1000000`: this means continue until an external
  request stops them. Do not stop a healthy run merely to analyze it. If the
  owner requests a stop, signal the exact trainer PID with `SIGTERM`, allow the
  current cohort to drain, and require locality, checkpoint, and `done` before
  declaring success. Never kill a healthy trainer mid-cohort.
- Do not change the live code or restart these streams to test an optimization.
  Preserve them as the before trace. Diagnose every new traceback/OOM first.
- Preserve unrelated dirty files. At handoff they were
  `docs/pareto_frontier_training_progress.md`,
  `docs/training_pipeline_v2.md`, `runs/layer_loss_manifest.csv`,
  `runs/layer_loss_manifest.md`, and
  `scripts/vllm_h100_overnight_queue.sh`.

## Exact live experiments

All active metric streams use source commit `356ce29`, report
`runtime_dirty=false`, reuse complete node-local `/dev/shm` caches, and have
passed the exact cached identity `h[L] == i[L+1]` before training.

| Host / cards | Model / loss | Exact run directory |
|---|---|---|
| agpul04 / 0,1 | Qwen3.5-0.8B / Huber | `runs/layerwise34_overnight_v2_qwen35_0p8b_lora_pp2_teacher_hidden_target_reuse_fullv5_huber_agpul04_01` |
| agpul04 / 2,3 | Qwen3.5-0.8B / cosine | `runs/layerwise34_overnight_v2_qwen35_0p8b_lora_pp2_teacher_hidden_target_reuse_fullv5_cosine_agpul04_23` |
| agpul05 / 0,1 | Qwen3.5-4B / Huber | `runs/layerwise34_overnight_v2_qwen35_4b_lora_pp2_teacher_hidden_target_reuse_fullv5_huber_agpul05_01` |
| agpul05 / 2,3 | Qwen3.5-4B / cosine | `runs/layerwise34_overnight_v2_qwen35_4b_lora_pp2_teacher_hidden_target_reuse_fullv5_cosine_agpul05_23` |
| agpul06 / 0,1,2,3 | Qwen3.6-27B / Huber | `runs/layerwise34_overnight_v2_qwen36_27b_lora_pp4_teacher_hidden_target_reuse_fullv5_huber_agpul06_0123` |

The cuts are `[12]`, `[16]`, and `[16,32,48]`, respectively. Every run is
LoRA, teacher-hidden, independent PP execution, B256/K16, whole full-v5 data,
fixed seed, and owner-local active-BxK GPU target residency. Cache hashes are
`633ce19cf18c5963` (0.8B), `e6659930d7736004` (4B), and
`751b16cc912b8cae` (27B).

Launcher logs and telemetry are exact files under:

- `runs/layerwise34_overnight_v2_logs/agpul04/`
- `runs/layerwise34_overnight_v2_logs/agpul05/`
- `runs/layerwise34_overnight_v2_logs/agpul06/`

Each host has one `*.gpu.csv`; each trainer has one `*.process.csv`. Current
PIDs are recorded in `docs/layerwise34_overnight_handoff.md`, but verify exact
argv because PIDs may be stale after a graceful stop.

The cosine logs on agpul04 and agpul05 intentionally retain one earlier
traceback from a rejected `CUDA_VISIBLE_DEVICES=2,3` command. Those attempts
failed before model construction, metric emission, or parameter writes. The
later start marker in each log begins the corrected sole-writer process. Do
not misclassify a healthy current stream from the earlier traceback; do not
hide the incident either.

## First evidence already recorded

- The corrected 100-question numerical comparison reproduced CE, KL,
  per-layer losses, and per-layer gradient norms exactly. Checkpoint relative
  L2 drift was `1.57e-13`, cosine `1.0`; locality leakage remained zero.
- The bounded 0.8B PP2 full-v5 run completed all 2,071 items, evaluated exactly
  947,644 answer tokens, measured 2,682.69 valid token-events/s, and reserved
  13.65/25.39 GiB rather than reproducing the old 43.44-GiB OOM.
- Initial open-ended completed-epoch rates were 0.8B Huber 2,840.45/s,
  0.8B cosine 2,824.08/s, 4B Huber 1,132.21/s then 1,241.05/s, and 4B cosine
  1,141.55/s. These are early values, not final epochs-2+ summaries.
- The first 27B cohort had balanced stage compute
  (21.96/20.78/20.52/20.78 s), but GPU activity oscillated badly. In one
  120-second window each card averaged only 19--22%; only 5/107 samples had
  two cards above 50%, and none had three or four.

## Analysis sequence

1. Verify liveness on each host with exact trainer argv and physical mappings.
   Confirm there is one trainer per experiment and only the intended PIDs own
   each GPU. Check the five exact logs for new material errors.
2. Read only each exact `metrics.jsonl`. Require one
   `pipeline_v32_contract`; count completed `v3_throughput`,
   `teacher_output_eval`, `v3_gradient_norm`, `parameter_delta`, and
   `locality_certification` rows. Never infer an epoch result from partial
   cohorts.
3. For every completed epoch, verify exact evaluated item/token counts,
   `dataset_coverage=whole_training_set_once_per_completed_epoch`,
   `validation_subset=false`, `evaluation_only=true`,
   `used_for_backward=false`, and `optimizer_weight=0.0`. Keep
   `KL-eval-loss` distinct from local `lens_kl`.
4. Report throughput by epoch and summarize epochs 2 onward separately from
   cold epoch 1. Include valid token-events/s, physical writes/s, per-stage
   compute/receive-wait/send-wait, stage balance, peak per-card VRAM, CE/KL,
   gradient norms, and parameter deltas. Compare Huber and cosine only within
   the same model/placement.
5. From each host's GPU CSV, exclude model load/epoch-zero setup using metric
   timestamps. For every physical card report utilization and power
   mean/p50/p95/max, zero-utilization fraction, temperature, and the
   distribution of how many cards are simultaneously above 50%. Utilization,
   watts, and temperature are measurements, not acceptance conditions for
   this overnight matrix.
6. From each process CSV, difference cumulative `utime_ticks`/`stime_ticks`
   using the host clock-tick rate, minor/major faults, read/write bytes, context
   switches, RSS, threads, and compiler-child presence. Correlate by timestamp
   with cohort completion and GPU oscillation. The CSV is parent-process
   telemetry; inspect compiler-child CPU separately before attributing it.
7. For 27B, preserve the established resource-locality distinction. The host
   was 95--98% CPU-idle, had 848 GiB available RAM, zero I/O wait, zero major
   faults, and no competing GPU process. Trainer memory was highly NUMA-skewed
   while the unpinned main producer packed full-depth pinned target tiles.
   This supports internal NUMA/producer contention, not Lustre or external CPU
   competition. Recheck `numastat -p <trainer>` and `nvidia-smi topo -m`
   before making a final causal claim.
8. Do not call GPUDirect unavailable. Every GPU pair reports P2P read/write
   support; `nvidia_fs` and cuFile configuration exist. Tonight's ordinary
   safetensors `/dev/shm` path simply does not use GDS. Keep pending comparisons
   for NUMA-local stage packing, verified GDS direct reads from a capable
   filesystem, bounded GPU-resident active windows, and the current pinned-host
   baseline. P2P is mainly relevant to student-hidden activation boundaries.

## Reporting and stopping

Analyze live metrics without stopping. If the owner later requests termination,
use exact `SIGTERM`, wait for cohort drain and final publication, then require
`locality_certification.passed=true`, zero cross-block/frozen-vocabulary
leakage, `done`, and `runs/<name>/checkpoint` for every stopped run. Stop the
matching telemetry sampler only after its trainer exits.

After final stop, regenerate campaign artifacts in the repository-prescribed
order: layer-loss heatmaps and temporal lines, delta profiles, atomic run
reports, grouped campaign report, and PDF. Include missing artifacts explicitly
rather than silently dropping a run. Update
`docs/layerwise34_timing_progress.md` and
`docs/layerwise34_overnight_handoff.md`, inspect the PDF pages visually, and
commit only this campaign's intended files.
