# Compressed shell and Slurm manifest

This manifest covers every flat `defactorised/*.sh`,
`defactorised/*.sbatch`, and `defactorised/demos/*.sh` program in this
snapshot. Each equivalent keeps the same relative name beneath `compressed/`.

All executable references to the standalone collection are routed from
`defactorised/` to `compressed/`. References to canonical mutable inputs such
as `scripts/queue.tsv` remain under `scripts/`; copying those would create a
second operational source of truth. File modes are preserved.

`optimized` means a provably non-executable change reduced the teaching/token
surface: a long historical header was replaced by one accurate purpose line.
Commands, environment, host maps, safety resets, arguments, and exit behavior
remain unchanged apart from `compressed/` target routing. `preserved` means
the implementation was reviewed and retained because shortening offered no
clear behavior or speed win, or because explicit code/comments carry safety,
lease, cache, process, or transport invariants.

| Compressed path | Decision | Rationale |
|---|---|---|
| `chain_ppp8_when_ready.sh` | preserved | Sequencing, readiness checks, cache construction, and launch failure propagation are the behavior. |
| `cpu_watch.sh` | preserved | Small telemetry loop; further shortening would only obscure sampled fields. |
| `demo_35b_answers.sh` | preserved | Already a declarative environment-and-argument envelope. |
| `demo_deepseek_bf16.sh` | optimized | Collapsed historical kernel narrative to one purpose line; invocation is unchanged. |
| `demo_deepseek_retry.sh` | preserved | Already a compact declarative retry envelope. |
| `download_mlt2.sh` | preserved | Sequential download order and completion markers are operational evidence. |
| `evaluate_v31_4b_full_standard.sh` | preserved | Explicit base/checkpoint conditions prevent evaluation-condition conflation. |
| `gpu_health.sh` | preserved | Local/remote inspection and exact GPU/process queries are diagnostic contracts. |
| `gpu_lease.sh` | preserved | Filesystem mutex, ownership, stale-local handling, and remote-lease policy are safety-critical. |
| `gpu_scheduler.sh` | preserved | Queue ordering, allocation mutexes, capacity checks, locks, and atomic handoff are safety-critical. |
| `gpu_speed_backfill.sh` | preserved | GPU selection, duplicate avoidance, and monitored launch evidence are coupled. |
| `gpu_util_monitor.sh` | preserved | Already a minimal utilization sampler. |
| `gpu_watchdog.sh` | preserved | Liveness and restart/alert conditions must remain explicit. |
| `hourly_evidence_check.sh` | preserved | Evidence filtering and report refresh steps encode supervision policy. |
| `l40s_exec.sh` | preserved | Loader, glibc, offline, cache, and thread environment ordering is platform-critical. |
| `l40s_setup.sh` | preserved | Dependency-layer pins and refusal behavior are more important than line count. |
| `l40s_train_v3.sh` | preserved | Coordinated cache bootstrap and train/report sequence are operational contracts. |
| `l40s_vllm_teacher_campaign.sh` | preserved | Resume markers, cache timing evidence, GPU placement, and failure logging are coupled. |
| `launch_dsflash_ppp8x.sh` | optimized | Reduced header to one purpose line; relay reset, host map, configs, and `exec` are unchanged. |
| `launch_g31b_ppp5x.sh` | optimized | Reduced header; retained relay reset required by the cross-node launch. |
| `launch_g31b_ppp8x.sh` | optimized | Reduced header; retained complete eight-stage host map. |
| `launch_g31b_v4.sh` | optimized | Reduced header; retained repo-root correction and exact configs. |
| `launch_q0p6b_ppp2x_test.sh` | optimized | Reduced header; retained two-host mapping and NCCL/IB test configs. |
| `launch_q0p6b_ppp3x_test.sh` | optimized | Reduced header; retained three-stage relay-drain topology and configs. |
| `launch_q122b_ppp8x.sh` | optimized | Reduced header; retained relay reset and eight-stage cross-node map. |
| `launch_q122b_ppp8x_adam.sh` | optimized | Reduced header; retained Adam config identity, relay reset, and host map. |
| `launch_q122b_ppp8x_evalin.sh` | preserved | Already the minimal eval-in wrapper. |
| `launch_q397b_ppp8x.sh` | optimized | Reduced header; retained relay reset, rotation config, and host map. |
| `launch_q397b_ppp8x_adam.sh` | optimized | Reduced header; retained Adam/rotation config, relay reset, and host map. |
| `launch_v32r_score100_campaign.sh` | preserved | Remote PID guards, detached logging, and fixed placement matrix are safety/evidence logic. |
| `launch_v4_stages.sh` | preserved | Lease handling, RAM-only relay checks, NCCL coordination, remote launch, and reaper integration are publication-critical. |
| `lossgrid_health_monitor.sh` | preserved | Health thresholds and evidence extraction should remain independently auditable. |
| `overnight_27b_online.sh` | preserved | Cache identity, readiness gate, and launch sequence must stay explicit. |
| `pipeline_tail.sh` | preserved | Already a short, purpose-built progress-bar filter. |
| `refresh_v31_0p8b_full_damage_reports.sh` | preserved | Atomic temporary report publication and campaign filters are evidence contracts. |
| `refresh_v31_reports.sh` | preserved | Atomic temporary report publication and campaign filters are evidence contracts. |
| `report_shipper.sh` | preserved | Retry cadence, noninteractive SSH, and final shipment are the behavior. |
| `results_refresher.sh` | preserved | Completeness gates and final refresh prevent partial report publication. |
| `run_m1_legs.sh` | preserved | Sequential leg liveness and exact numerical comparisons are certification logic. |
| `run_speed_check_monitored.sh` | preserved | VRAM guard, log capture, and telemetry lifecycle are coupled. |
| `sample_gpu_telemetry.sh` | preserved | PID identity checks avoid sampling an unrelated trainer. |
| `sample_process_telemetry.sh` | preserved | Loader-aware command parsing and process identity checks prevent false attribution. |
| `slurm_h100.sbatch` | preserved | Scheduler directives, node-local warm-up, and queue launch order are cluster contracts. |
| `spec_verify_122b.sbatch` | preserved | Setup, verification, expected refusal handling, and staged launch are intentional. |
| `spec_verify_batch.sbatch` | preserved | Per-row failure isolation and environment verification are intentional. |
| `spec_verify_matrix.sh` | preserved | Model/config matrix, cache gate, and PPP routing are scientific verification logic. |
| `stage_hf_cache.sh` | preserved | Atomic readiness, snapshot selection, and RAM-stage validation are cache safety logic. |
| `stage_teacher_cache_shm.sh` | preserved | Narrow cache selection, rsync verification, and ready marker prevent partial publication. |
| `stop_g31b_at25.sh` | preserved | Epoch boundary and repeated graceful TERM behavior are owner-specified semantics. |
| `stop_v32r_score100_campaign.sh` | preserved | Exact process matching and scoped stop behavior avoid collateral termination. |
| `v4_stage_reaper.sh` | preserved | Cohort failure detection and sibling cleanup protect incomplete shard publication. |
| `venv_check.sh` | preserved | Pin, CUDA, import-isolation, optional dependency, and bundle checks are certification. |
| `venv_setup.sh` | preserved | Node-local path, destructive-force guard, dependency pins, and indexes are runtime policy. |
| `vllm_h100_eager_shared_queue.sh` | preserved | Atomic queue claim, GPU assignment, resume checks, and cache roots are coupled. |
| `vllm_h100_mixed_budget_campaign.sh` | preserved | Budget matrix, skip checks, cache roots, and result validation are campaign evidence. |
| `vllm_h100_overnight_queue.sh` | preserved | Ordered TP/single-GPU queue, skip checks, and cache discipline are operational policy. |
| `vllm_h100_qwen06_32k_capacity.sh` | preserved | Compact capacity sweep already exposes its complete batch-size matrix. |
| `vllm_h100_qwen35_shared_queue.sh` | preserved | Compact shared-queue allocator; shortening would hide locking or placement. |
| `vllm_h100_rest_of_day.sh` | preserved | Minimal sequential composition of the two named campaigns. |
| `vram_guard.sh` | preserved | Threshold polling and command `exec` semantics are reusable safety logic. |
| `wait_m1_then_g26b_e500.sh` | preserved | Dual liveness condition prevents an early dependent launch. |
| `warm_python_runtime.sh` | preserved | Symlink resolution, bounded parallel stat, and import warm-up are platform-specific. |
| `demos/ppn_demo.sh` | preserved | Worker coordination, depth-one handoff visibility, failure injection, cleanup, and result display are the lesson. |
| `demos/ppp_demo.sh` | preserved | Independent worker launch, shard collection, failure injection, and cleanup are the architectural contrast. |

## Static audit

The shell collection is safe to inspect on a login host. Syntax and target
audits require no GPU or worker:

```bash
find compressed -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 |
  xargs -0 -n1 bash -n

comm -3 \
  <(find defactorised -maxdepth 1 -type f \( -name '*.sh' -o -name '*.sbatch' \) -printf '%f\n' | sort) \
  <(find compressed -maxdepth 1 -type f \( -name '*.sh' -o -name '*.sbatch' \) -printf '%f\n' | sort)

rg -n 'defactorised/' compressed -g '*.sh' -g '*.sbatch'
```

The first command must return success; the latter two must print nothing.
Runtime equivalence of GPU/campaign launchers is not claimed from static syntax
alone and must be certified on the appropriate node before use.
