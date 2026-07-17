# Layerwise 3.4 overnight teacher-hidden handoff

## Purpose

Five full-v5, open-ended LoRA runs compare block-local teacher-hidden execution
without the severe B=100 packing confound.  `epochs: 1000000` is the practical
infinite ceiling: they run until an external cooperative termination request.
They occupy 12 L40S cards and keep the ordinary physical layer distributions
unchanged.  Huber and cosine use separate trainers but share one immutable
full-input cache per model/host.

## Expected jobs

| Host | GPUs | Model/loss | Cut | Run directory |
|---|---|---|---|---|
| agpul04 | 0,1 | Qwen3.5-0.8B Huber | `[12]` | `runs/layerwise34_overnight_qwen35_0p8b_lora_pp2_teacher_hidden_fullv5_huber_agpul04_01` |
| agpul04 | 2,3 | Qwen3.5-0.8B cosine | `[12]` | `runs/layerwise34_overnight_qwen35_0p8b_lora_pp2_teacher_hidden_fullv5_cosine_agpul04_23` |
| agpul05 | 0,1 | Qwen3.5-4B Huber | `[16]` | `runs/layerwise34_overnight_qwen35_4b_lora_pp2_teacher_hidden_fullv5_huber_agpul05_01` |
| agpul05 | 2,3 | Qwen3.5-4B cosine | `[16]` | `runs/layerwise34_overnight_qwen35_4b_lora_pp2_teacher_hidden_fullv5_cosine_agpul05_23` |
| agpul06 | 0,1,2,3 | Qwen3.6-27B Huber | `[16,32,48]` | `runs/layerwise34_overnight_qwen36_27b_lora_pp4_teacher_hidden_fullv5_huber_agpul06_0123` |

Worker logs and one-second GPU CSVs are under
`runs/layerwise34_overnight_logs/<host>/`.

## Launch state (2026-07-17 02:54 Europe/Madrid)

All configs resolve `epochs: 1000000` and were launched from corrected commit
`4a3d635`.  The initial one-epoch admissions were stopped before training and
their incomplete cache directories were removed; they are not experiment runs.

| Host/job | Launcher | Current cache process | Telemetry sampler |
|---|---:|---:|---:|
| agpul04 0.8B Huber | 1884622 | 1884647 (lease waiter) | 1885057 |
| agpul04 0.8B cosine | 1884625 | 1884648 (builder) | 1885658 |
| agpul05 4B Huber | 448320 | 448344 (builder) | 449446 |
| agpul05 4B cosine | 448321 | 448343 (lease waiter) | 449447 |
| agpul06 27B Huber | 3845831 | 3845842 (builder) | 3846294 |

The sampler processes intentionally wait for the corresponding `scripts/train.py`
argv and begin writing rows only after cache publication.  Cache builders are
allowed to change into new trainer PIDs under the persistent launcher.  Do not
classify an idle loss pair as failed while its peer owns the same model/cache
lease.  At launch verification there were no tracebacks or OOMs.  agpul06 kept
the required 63-GiB 27B model snapshot and had 694 GiB free before its clean
full-input cache build; obsolete node-local teacher caches were evicted.

At approximately 03:00, the 0.8B cache `633ce19cf18c5963` published all 2,071
examples/24 layers after 219.5 seconds.  Huber trainer PID `1885904` and cosine
trainer PID `1885954` replaced their builders, and both telemetry CSVs began
recording.  Both initial shard-64 trainers then OOMed before an accepted epoch:
GPU1/GPU3 each held 43.44 GiB allocated with about 108 MiB free when another
130 MiB was requested.  They were not externally stopped.  Unique `shard32`
retries preserve B256/K16, PP2 cut `[12]`, loss, seed, and write semantics while
halving only transient activation/prefill shard width, but they OOMed at the
same footprint.  Code inspection found that GPU-cache preparation retains all
teacher-input shards for the cohort, so shard width does not bound total
residency.  Final uniquely named `cpu` retries remain teacher-hidden but stage
active K tiles from pinned host RAM.  Do not analyze the shard-64 or shard-32
directories as completed runs.

At approximately 03:01, the 4B cache `e6659930d7736004` published all 2,071
examples/32 layers after 380.6 seconds (132 GiB immediately before atomic
publication).  Cosine trainer PID `451823` took GPUs2/3 and Huber trainer PID
`451896` took GPUs0/1.  Both loss arms and both samplers are therefore live;
the earlier idle GPUs2/3 were the expected atomic-cache wait, not a failed
second launch.

## Prompt for tomorrow's analysing agent

Read `AGENTS.md`, `docs/layerwise34_timing_progress.md`, this file, and each
run's `config.yaml`/`metrics.jsonl` before acting.  Do not relaunch or delete a
run merely because it is incomplete.  On agpul04/agpul05/agpul06, identify the
exact launcher and trainer PIDs, inspect compact logs with
`scripts/pipeline_tail.sh`, and check the matching telemetry CSV.  Diagnose
every traceback, OOM, cache-capacity failure, duplicate writer, or cursor
mismatch before changing anything.

When enough complete epochs have accumulated, send `SIGTERM` to each exact
trainer, admit no replacement cohort, and let it drain, certify, save, and exit;
never use `SIGKILL` for a healthy trainer.  For each stopped run, require
`pipeline_v32_contract`, per-completed-epoch `v3_throughput`, exact
whole-training-set `teacher_output_eval`, `locality_certification.passed=true`,
`done`, and a published checkpoint. Verify physical mappings and cuts against
the table above, `trajectory_source=teacher_hidden`,
`teacher_hidden_source=gpu_cache`, zero boundary bytes, 2,071 dataset items,
and exact evaluated answer-token/item counts.  Compare both token-events/s and
physical block-writes/s: the earlier 100-question runs were underfilled, so a
token-events/s comparison alone was misleading.  Report per-stage compute and
wait time, full-trace mean/p95/max GPU utilization and power, peak VRAM, cache
identity/size/build time, CE-eval-loss, KL-eval-loss, gradient norms, and
parameter deltas.  Compare Huber versus cosine only within the same model and
placement; compare the full-v5 Huber arms with the ordinary full-v5 PP2/PP4
references in `docs/layerwise34_timing_progress.md`.

CPU usage was visibly oscillatory in this code version.  Correlate per-trainer
CPU, page faults/read bytes, native thread counts, and TorchInductor children
with cohort boundaries and GPU idle intervals; follow the open issue in
`issues.md`.  Do not label the cause as cache I/O, compilation, or dispatch
without that correlation.

After all five are cooperatively stopped and terminate, regenerate in order:

1. `scripts/layer_loss_plots.py --runs 'layerwise34_overnight_*' --force`
2. `scripts/delta_profiles.py --runs 'layerwise34_overnight_*' --force`
3. atomic reports with `scripts/report_v2.py`
4. the `layerwise34_timing` grouped report and PDFs

Update `docs/layerwise34_timing_progress.md`, explicitly list missing evidence,
stop stale samplers, verify no campaign process remains, and commit the report
artifacts without touching unrelated dirty files.
