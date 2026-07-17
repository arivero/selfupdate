# Layerwise 3.4 overnight teacher-hidden handoff

## Purpose

Five full-v5, one-epoch LoRA runs compare block-local teacher-hidden execution
without the severe B=100 packing confound.  They occupy 12 L40S cards and keep
the ordinary physical layer distributions unchanged.  Huber and cosine use
separate trainers but share one immutable full-input cache per model/host.

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

## Prompt for tomorrow's analysing agent

Read `AGENTS.md`, `docs/layerwise34_timing_progress.md`, this file, and each
run's `config.yaml`/`metrics.jsonl` before acting.  Do not relaunch or delete a
run merely because it is incomplete.  On agpul04/agpul05/agpul06, identify the
exact launcher and trainer PIDs, inspect compact logs with
`scripts/pipeline_tail.sh`, and check the matching telemetry CSV.  Diagnose
every traceback, OOM, cache-capacity failure, duplicate writer, or cursor
mismatch before changing anything.

For each completed run, require `pipeline_v32_contract`, `v3_throughput`, exact
whole-training-set `teacher_output_eval`, `locality_certification.passed=true`,
`done`, and a published checkpoint.  Verify physical mappings and cuts against
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

After all five terminate, regenerate in order:

1. `scripts/layer_loss_plots.py --runs 'layerwise34_overnight_*' --force`
2. `scripts/delta_profiles.py --runs 'layerwise34_overnight_*' --force`
3. atomic reports with `scripts/report_v2.py`
4. the `layerwise34_timing` grouped report and PDFs

Update `docs/layerwise34_timing_progress.md`, explicitly list missing evidence,
stop stale samplers, verify no campaign process remains, and commit the report
artifacts without touching unrelated dirty files.
