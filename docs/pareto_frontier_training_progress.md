# Pareto-frontier training progress

This is the running record for the six-model Pareto-frontier campaign. Qwen3.5
4B is the flagship experimental model within the campaign, not the name of the
campaign as a whole. The campaign is
explicitly pinned to **dataset v5** and **training pipeline v2**.

Status at 2026-07-14: teacher caches and epoch-zero controls are complete for
all six models. The first four Qwen3.5 4B full trainings completed, but they
used the then-present behavioral readout runtime. They are retained as
superseded historical diagnostics and are not frontier evidence. Individual
report-v2 artifacts are immediate per-run records; new frontier arms must use
the strict block-local contract below.

All pre-existing trained runs are pipeline-v1 historical artifacts. They may
inform hypotheses and mechanics, but they are obsolete as evidence for this
pipeline-v2 campaign and are never mixed into its reports or Pareto synthesis.

## Campaign contract

Target models:

- Qwen3.5 4B — flagship control
- Qwen3.5 9B — Pareto-envelope surprise candidate
- Gemma4 26B-A4B
- Qwen3.6 35B-A3B
- Qwen3.6 27B
- Gemma4 31B

Dataset v5 is the 2,071-example RAG/window suite at
`data/combined/examples_v5rs_window.jsonl` (SHA-256
`575b9dea35e0179dcdf7a513416e640db899c9bf9584236088f2921cce7a7042`).
The teacher prompt mask is `rag_system` with `remove` compaction.

The current training contract is strict block-local forward distillation:
Pareto bases pin `conn_window: 1`; every block receives detached input and
only its own cached teacher target; and no behavioral readout or final-logit
training is permitted. `lens_kl` may use the frozen norm and vocabulary head as
a local metric, but neither is updated and the objective never crosses block
boundaries. The former readout runtime has been deleted and is recoverable
from Git history only.

## Execution plan and current phase

The campaign follows the plan fixed at initialization; implementation work on
update geometry refines phase 3 and does not replace the campaign sequence.

1. **Environment and cache certification — complete.** Audit the four L40S
   devices, node-local storage/RAM, six model snapshots, and six exact
   2,071-example dataset-v5 teacher caches built through the vLLM-plus-cache
   pipeline.
2. **Epoch-zero controls — complete.** Record standard-model baselines plus
   deleted-RAG and randomized-token-RAG answers for all six models.
3. **Qwen3.5 4B strict-local runtime and strategy gate — active.** Replace
   the superseded readout-bearing cohort with strict block-local arms, then
   make update geometry explicit as an `answer × aligned token × layer` grid.
   A tile retains every token's full causal prefix and walks layers strictly
   forward so student `h[L]` is produced from student `h[L-1]`; cached teacher
   `h[L]` remains the target.
4. **Matched update-geometry experiment — coding/probing.** Compare equal
   token budgets across rectangles such as `20 × 16`, `10 × 32`, `5 × 64`,
   `1 × all`, and the one-token-across-batch timing corner.  Record reduction
   (`answer_mean` or `token_mean`) independently from geometry.
5. **Individual report v2 — continuous.** Generate `runs/<training>/report.md`
   immediately after every completed training, with epoch-zero and per-epoch
   recall/damage, layer loss, parameter delta, timing, provenance, and signal
   attribution. Final synthesis selects these atomic reports by campaign,
   model, loss, censorship, or update geometry.
6. **Six-model expansion — pending gate.** Apply only the measured, validated
   recipes to Qwen3.5 9B, Gemma4 26B-A4B, Qwen3.6 35B-A3B, Qwen3.6 27B, and
   Gemma4 31B; checkpoint evaluation will use the same programmatic vLLM
   manager design as epoch-zero evaluation.

For this document, training pipeline v2 means the two-stage teacher-target
path:

1. Programmatic greedy answer generation through vLLM 0.25, producing exact
   response token IDs for every v5 record.
2. `scripts/build_teacher_cache.py` importing those response IDs through
   `--generation-responses`, then running the teacher-forced hidden-state pass
   and writing the per-layer cache.

The response IDs are therefore part of the per-model cache provenance; there
is no decode/re-encode round trip. Student training will consume these frozen
teacher caches through the v2 training path.

The cache payload is teacher-side: generated answer token IDs and hidden
states over the teacher's aligned `shared_mid + answer` suffix. Both student
censorship arms therefore explicitly select the certified `remove` cache via
`cache.source_compaction: remove`. This is cross-view reuse, not a relaxation
of cache identity. The loader resolves and verifies the original model,
dataset, mask mode, dtype, response-file digest, generation settings, and
schema hash; for every example it additionally requires identical teacher
`t0` and aligned length. It recomputes student `s0`, position gap, and token
sequence from the active `remove` or `pad_random` intervention. A synthetic
full-dataset audit on 2026-07-14 proved all 2,071 teacher token sequences and
teacher spans identical between the two views, all 2,071 student sequences
different, and an undeclared cross-view load rejected.

## Environment snapshot

Recorded 2026-07-14 13:34 CEST from allocation 418174 on `agpul02`:

- 4 × NVIDIA L40S, 46,068 MiB each; all four were idle at inspection.
- Compute-node `/tmp`: 367 GiB free of 434 GiB; inode use 1%.
- Host memory: approximately 1.5 TiB, mostly free.
- Pipeline-v2 load staging may use Unix tmpfs at
  `/dev/shm/$USER/selfupdate-hf-cache`: 756 GiB was available on `agpul02`.
  Base snapshots and transient checkpoint copies are ordinary safetensors
  files there, shared through the kernel page cache; durable copies remain on
  Lustre.
- The durable caches below were generated previously on H100 capacity probes;
  their provenance commits are ancestors of the current checkout
  (`393719d`); the five `artifacts_exact_full` caches use
  `8cededae850cf74754421017c53504d3dfac28f0`, while the pre-existing Qwen3.5
  9B cache uses `fa43971bbd01b896042bcdea5007b044ea7bcd70`.

## Cache audit

All six entries have 2,071 indexed examples, 17 safetensors shards, a
`generation_report.json`, `generation_timings.json`, and `timings.json`.
The hidden caches are durable under `runs/teacher_cache_h100/artifacts_exact_full/`
and the pre-existing Qwen3.5 9B artifact directory listed below.

Although these caches were generated during H100 probes, they are portable
teacher-cache artifacts rather than GPU snapshots: the payload is stored in
safetensors with CPU-readable metadata and contains no H100-specific CUDA
state. They are intended to be consumed by the current L40S training
runtime. The Qwen3.5 4B and larger target caches use bfloat16; the existing
Qwen3.5 9B cache uses float16. Portability covers the cache format and
tensors, not differences in throughput, peak memory, or runtime versions;
those will be recorded when training starts on L40S.

Times are wall-clock seconds from the recorded artifacts. vLLM load and
generation are reported separately; cache time includes response import,
teacher forward, shard writing, and finalization.

| model | vLLM load | vLLM generation | generated tokens | cache import | teacher forward | cache write | cache total | cache size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B | 106.32 | 35.22 | 87,306 | 0.03 | 69.02 | 35.75 | 105.03 | 36.6 GB |
| Qwen3.5 9B | 115.08 | 48.04 | 79,334 | 0.03 | 88.15 | 12.98 | 101.08 | 53.0 GB |
| Gemma4 26B-A4B | 94.48 | 47.30 | 85,028 | 0.03 | 66.28 | 28.79 | 92.03 | 37.1 GB |
| Qwen3.6 35B-A3B | 143.70 | 50.80 | 75,219 | 0.03 | 145.57 | 16.87 | 167.41 | 34.7 GB |
| Qwen3.6 27B | 180.95 | 120.00 | 70,670 | 0.26 | 234.36 | 122.84 | 361.41 | 135.6 GB |
| Gemma4 31B | 225.60 | 175.53 | 78,598 | 0.12 | 247.80 | 123.54 | 361.89 | 137.4 GB |

Generation artifact timestamps:

- Qwen3.5 4B: 2026-07-13 19:55:16 CEST
- Qwen3.5 9B: 2026-07-13 21:18:47 CEST
- Qwen3.6 35B-A3B: 2026-07-13 19:56:09 CEST
- Gemma4 26B-A4B: 2026-07-13 19:59:24 CEST
- Qwen3.6 27B: 2026-07-13 21:36:33 CEST
- Gemma4 31B: 2026-07-13 21:38:13 CEST

Cache finalization timestamps:

- Qwen3.5 9B: 2026-07-13 23:12:03 CEST
- Gemma4 26B-A4B: 2026-07-14 00:44:14 CEST
- Qwen3.6 35B-A3B: 2026-07-14 00:45:54 CEST
- Gemma4 31B: 2026-07-14 00:53:13 CEST
- Qwen3.6 27B: 2026-07-14 00:53:44 CEST
- Qwen3.5 4B: 2026-07-14 00:55:45 CEST

## Artifact locations

| model | vLLM response summary | durable cache |
|---|---|---|
| Qwen3.5 4B | `runs/vllm_benchmark_h100/Qwen3.5-4B_vllm025_single_gpu0_graph64_mixed_tokenids_full_h100/` | `runs/teacher_cache_h100/artifacts_exact_full/Qwen3.5-4B-rag_system-remove-885f57b6f4eb9221/` |
| Qwen3.5 9B | `runs/vllm_benchmark_h100/mixed_budget_campaign/qwen35_9b_mixed_b64_single/` | `runs/teacher_cache_h100/artifacts/selfupdate-cache-full-qwen35-9/Qwen3.5-9B-rag_system-remove-dcc5205520bb54c4/` |
| Gemma4 26B-A4B | `runs/vllm_benchmark_h100/gemma-4-26B-A4B-it_vllm025_single_gpu0_graph64_mixed_tokenids_full_h100/` | `runs/teacher_cache_h100/artifacts_exact_full/gemma-4-26B-A4B-it-rag_system-remove-78b4657895836025/` |
| Qwen3.6 35B-A3B | `runs/vllm_benchmark_h100/Qwen3.6-35B-A3B_vllm025_single_gpu1_graph64_mixed_tokenids_full_h100/` | `runs/teacher_cache_h100/artifacts_exact_full/Qwen3.6-35B-A3B-rag_system-remove-69faac60b7b3f67b/` |
| Qwen3.6 27B | `runs/vllm_benchmark_h100/mixed_budget_campaign/qwen36_27b_mixed_b64_single/` | `runs/teacher_cache_h100/artifacts_exact_full/Qwen3.6-27B-rag_system-remove-ef92bc2ccff8c62d/` |
| Gemma4 31B | `runs/vllm_benchmark_h100/mixed_budget_campaign/gemma4_31b_mixed_b64_single/` | `runs/teacher_cache_h100/artifacts_exact_full/gemma-4-31B-it-rag_system-remove-630be9b71249b554/` |

## Training log

| timestamp | event | status | evidence |
|---|---|---|---|
| 2026-07-14 13:34 CEST | Campaign document initialized | complete | This document |
| 2026-07-14 13:34 CEST | v5 dataset identity recorded | complete | Dataset path and SHA-256 above |
| 2026-07-14 13:34 CEST | v2 cache audit | complete | Six complete cache indexes and shard sets |
| 2026-07-14 13:35 CEST | Qwen3.5 9B added to Pareto-envelope targets | complete | Existing complete v5 cache audited |
| 2026-07-14 | Teacher evaluation matrix audited | complete | Standard corruption baselines and both censorship controls were missing for all six models |
| 2026-07-14 | Initial Transformers teacher-evaluation queue stopped | superseded | 11 valid completed JSON outputs preserved; slow in-flight generation terminated |
| 2026-07-14 | Evaluation generation migrated to pipeline v2 | ready | `scripts/teacher_ceiling.py` now uses programmatic vLLM; seven remaining controls are in `scripts/queue_pareto_teacher_evals_v2_20260714.tsv` |
| 2026-07-14 | Individual report v2 data contract initialized | complete | `docs/report_v2.md`; local `runs/<run_name>/report.md` is generated after each training completes |
| 2026-07-14 | Explicit cross-censorship teacher-cache reuse | complete | Commit `74a8900`; all 2,071 teacher sequences/spans identical, all student views changed, undeclared reuse and forged teacher mismatch rejected |
| 2026-07-14 | Pipeline-v2 trainer mechanics and telemetry | validated | Answer/token update boundaries, dual loss measures, effective LoRA/full-weight epoch deltas; five legacy variants certified unchanged on L40S |
| 2026-07-14 15:32–16:14 CEST | Remaining epoch-zero vLLM queue | complete | Seven controls completed from RAM-staged HF cache; final Gemma4 31B randomized control used corrected synchronous scheduling |
| 2026-07-14 | Six exact-cache training bases and 4B probe grid | validated | `configs/experiments/pareto_v2/`; every base resolves its certified 2,071-example cache hash, eight probe overlays pass typed dispatch validation |
| 2026-07-14 | Gemma4 31B randomized control async-scheduler diagnosis | corrected/requeue | vLLM 0.25 async pipeline scheduling twice violated output-placeholder accounting; evaluator now defaults to synchronous scheduling and records the choice |
| 2026-07-14 | Initial 4B probe runtime launches | superseded | L40S driver 560 rejects torch 2.11/cu128 in both the container and shared venv at CUDA initialization; neither attempt loaded weights |
| 2026-07-14 | L40S training runtime correction | validated | Thin `/tmp` layer over torch 2.7.1+cu126; bf16 matmul, Transformers 5.12.1, PEFT 0.19.1, kernels 0.12.0, and Qwen3.5 config resolution pass without installing torch |
| 2026-07-14 | Compiled causal-conv1d ABI | validated/requeue | The previously undocumented `glibc/2.35` module works when Python is entered through `$GLIB235_LINUX_SO` with explicit system paths; a real CUDA causal-conv kernel passed. The 4B speed probes are requeued because fallback timings are not campaign-comparable. |
| 2026-07-14 | glibc child-process isolation | corrected/requeue | First compiled-backend queue exposed module `LD_LIBRARY_PATH` to Triton's host `gcc`, causing a host-loader/2.35-libc `GLIBC_PRIVATE` mismatch. The scheduler retry loop was stopped; the wrapper now retains 2.35 through the Python loader while restoring host paths for children. |
| 2026-07-14 | Trainer epoch-zero CER dependency | corrected/requeue | First production launch stopped before training because trainer-side recall retained an obsolete `jiwer` import absent from the thin L40S runtime. CER now uses the repository's own jiwer-compatible character Levenshtein implementation; all four failed workers and the retry scheduler were stopped. |
| 2026-07-14 | Standard-eval offline inputs | validated/requeue | Second production launch stopped before training because ARC-Challenge and HellaSwag still attempted Hub access under the intentionally offline RAM-backed model runtime. Their fixed 100-item subsets are now vendored at the pinned revisions, matching the existing ARC-Easy policy; all three 100-item task loaders pass with Hub access disabled. |
| 2026-07-14 | Scheduler duplicate lock handoff | corrected/live | Third launch reached training, but a transient empty lock during worker-PID publication let the pad-random answer arm launch twice. The extra GPU-2 process was stopped; the intended GPU-3 arm remains live. Worker lock replacement is now atomic for subsequent launches. |
| 2026-07-14 | Six-step speed projection | invalidated/live-negative | Compiled causal-conv and RAM staging are active, but the full Qwen3.5 walk exposed shape-driven TorchInductor compilation absent from the short probe: each trainer spawned 32 compiler workers (~128 total), GPUs waited, and observed throughput projected roughly 5–7 hours for six Huber epochs rather than 24–33 minutes of pure-step time. Future launches cap compiler workers at 2 and reuse a node-local cache; campaign timing must come from a distribution-covering probe. |
| 2026-07-14 | Teacher-state RAM staging | correcting/live | The original 252 GB tmpfs stage covered HF model snapshots, not the 35 GB Qwen3.5-4B hidden-state cache. Live workers showed multi-GB physical reads from the 943 GB Lustre cache root while CPU and GPU were idle. A separate selected-cache tmpfs stage and `SELFUPDATE_TEACHER_CACHE_ROOT` override now protect subsequent arms. |
| 2026-07-14 | Distribution-covering post-stage speed gate | split verdict | Tokenwise bucketed: 27.23 examples/s, 76.1 s/projected epoch, healthy. Answerwise B=1: 1.14 examples/s, 1,824.5 s/projected epoch, rejected for production pending variable-shape repair. Launch the four tokenwise arms now; retain answerwise as required strategy work, not as a slow runtime artifact. |
| 2026-07-14 | Qwen3.5 4B eight-arm student grid | historical/superseded | Six epochs = 12,426 examples per arm; remove/pad-random × Huber/lens-KL × answer/token aggregation; compiled causal-conv L40S runtime. The four readout-bearing arms are not frontier evidence. |
| 2026-07-14 18:29:42 CEST | First Qwen3.5 4B full-training cohort | complete/superseded historical diagnostics | Four `B=8 × all aligned tokens` pipeline-v2 arms on physical GPUs 0–3; dataset SHA `575b9d…a7042`, source commit `2d4e066`, RAM-staged model and exact teacher cache. This cohort is the full-minibatch reference, not a claim that the token axis has been traversed one row at a time, and it is excluded from the frontier. |
| 2026-07-14 18:41:54 CEST | Qwen3.5 4B Huber/remove full training | complete/superseded historical diagnostic | GPU 0; six epochs, 12,426 answer observations; metrics span 660.3 s (11.0 min), scheduler wall 12.2 min; checkpoint `runs/pareto_v2_qwen35_4b_huber_remove_token/checkpoint/` (101 MB LoRA); epoch 0–6 recall, standard damage, and parameter deltas present. |
| 2026-07-14 18:48:06 CEST | Qwen3.5 4B Huber/pad-random full training | complete/superseded historical diagnostic | GPU 1; six epochs, 12,426 answer observations; metrics span 1,032.3 s (17.2 min), scheduler wall 18.4 min; checkpoint `runs/pareto_v2_qwen35_4b_huber_pad_random_token/checkpoint/` (101 MB LoRA); epoch 0–6 recall, standard damage, and parameter deltas present. |
| 2026-07-14 18:57–19:01 CEST | First individual reports v2 | complete | `runs/pareto_v2_qwen35_4b_huber_{remove,pad_random}_token/report.md`; each report has recall/damage, temporal and heatmap layer loss, one-row density, temporal/heatmap/one-row parameter delta, and exact-cache signal attribution. Coverage pages declare all mandatory epoch telemetry present. |
| 2026-07-14 19:00 CEST | Huber-arm signal attribution | historical readout-dominated diagnostic | Across 16 sampled items, hidden/readout gradient L2 norms are 1.58/65.2 (2.36% hidden share) for remove and 1.59/61.9 (2.51%) for pad-random. Targets came from exact pipeline-v2 cache `885f57b6f4eb9221`. These are valid measured historical arms but must not be described as hidden-loss-dominated layerwise or frontier evidence. |
| 2026-07-14 18:59 CEST | Explicit three-dimensional update grid | coding/probe | Typed answer/token tile geometry, independent answer/token reduction, mandatory forward layer walk, exact coordinate ranges, and cumulative selected-loss/full-causal layer cells implemented locally. The requested `B=8 × K=1 × layers` timing is running on free GPU 0 before accepting the design. |
| 2026-07-14 18:57–19:05 CEST | Production-like `B=8 × K=1` tile timing | complete/superseded historical diagnostic | 256 real Huber + teacher-KL readout updates, full causal prefixes and forward 32-layer walk: 0.1454 s median (0.1546 s mean) per tile, 39.92 selected aligned cells/s, 8.93 GiB peak reserved, no errors. Full dataset-v5 aligned-token coverage projects to 5,600.9 s (93.3 min) per epoch, about 74× the 76.1 s `B=8 × K=all` reference. |
| 2026-07-14 19:11–19:20 CEST | First constant-area diagonal launch | stopped/corrected | The benchmark omitted the repository CPU-thread bootstrap and its B=32 compile phase expanded to roughly 55 CPU cores, risking contention with live lens-KL training. Its Luna owner terminated only benchmark PIDs 994107/994093. `grid_tile_bench.py`, `train_batch_bench.py`, and `signal_attribution.py` now call `cap_cpu_threads()` before importing torch; the diagonal table restarted at 19:20 with `SELFUPDATE_CPU_THREADS=8`. |
| 2026-07-14 19:20–19:22 CEST | Constant-area B×K diagonals | complete/fast-K verdict | All 18 power-of-two cells completed with real nonzero L2-normalized-MSE gradients, AdamW at 1e-5, no readout, full causal prefixes, and the forward 32-layer walk; no OOM through `B=64`. At fixed 64 selected cells, `1×64` = 0.1089 s/588 cells/s versus `64×1` = 1.5407 s/41.6 cells/s (about 14× slower). Fast direction is larger K/smaller B; fill the `B={1,2,4,8} × K={8,16,32,64}` rectangle. Artifact: `runs/pareto_v2_grid_tile_table_20260714/diagonals.{json,csv,md}`. |
| 2026-07-14 19:22–19:24 CEST | Fast-side B×K rectangle | complete | Seven missing cells completed without OOM. At fixed B, widening K is nearly free: B=1 takes 0.107–0.109 s for K=8–64, B=4 takes 0.134–0.139 s for K=4–64, and B=8 takes 0.178–0.193 s for K=2–64. Best selected-cell rate is `8×64` at 2,726 cells/s; `64×1` is 41.6 cells/s. Combined table: `runs/pareto_v2_grid_tile_table_20260714/combined_fast_rectangle.{json,csv,md}`. Engineering verdict: make K wide and use B for occupancy; K=1 repeats causal trajectories for almost no additional signal throughput. |
| 2026-07-14 | Strict-local runtime and report-v2 publication | committed/gating | Commits `6d7a2f1` and `38ee461`: behavioral readout/final-logit training deleted; one grid tile is exactly one optimizer update (`grad_accum: 1`); every pipeline-v2 checkpoint requires model-resident proof of positive local gradient in every block, zero cross-block leakage, and zero frozen-vocabulary leakage. Individual and typed grouped report-v2 generators replace the readout-era collective generator. |
| 2026-07-14 | One-day Qwen3.5-4B strict-local screen | configured | 40 full trainings: four local losses × two censorship modes × five B×K geometries. Every arm runs six complete v5 epochs (12,426 source-answer completions) and publishes `report.md` plus `report_manifest.json` immediately. Queue scheduling estimates total 35.3 GPU-hours, approximately 8.8 hours on four L40S before variance; they are scheduling hints, not measured runtime. |
| 2026-07-14 20:40 CEST | Strict-local Qwen3.5-4B prelaunch gate | pass | 16 sampled v5 items across every block: local gradient L2 1.58, cross-block leakage 0, frozen-vocabulary leakage 0; certifier also requires positive finite signal in every block. First delegated attempt inherited generic Qwen3-0.6B and failed offline before model load; corrected command used `base_qwen35_4b.yaml` and exited 0 with no warnings. |
| 2026-07-14 20:42:11 CEST | Strict-local 4B screen launched | live | Source commit `6c161b2`; shared lease scheduler PID 1017631 on `agpul02`, physical GPUs 0–3. First cohort: Huber/remove, Huber/pad-random, cosine/remove, cosine/pad-random at `B=8 × K=all`; initial residency 10.3–10.9 GiB/card and measured utilization 47–59%. Launch delegated to Luna; parent supervision retained. |
| pending | Qwen3.5 9B student training | pending | — |
| pending | Gemma4 26B student training | pending | — |
| pending | Qwen3.6 35B student training | pending | — |
| pending | Qwen3.6 27B student training | pending | — |
| pending | Gemma4 31B student training | pending | — |

Future entries must record the start/end timestamp, model/config, GPU
placement, dataset identity, pipeline-v2 commit, checkpoint path, item
count, and any failure or restart reason.

### One-day strict-local 4B screen

The committed screen lives in
`configs/experiments/pareto_v2/screen_4b/` and is generated deterministically
by `scripts/gen_pareto_v2_screen.py`. Its axes are:

- local objective: Huber, cosine, delta-cosine, or local teacher lens-KL;
- censorship: deleted RAG or randomized-token RAG (`remove` / `pad_random`);
- grid tile: `1×all`, `4×128`, `8×64`, `16×32`, or `8×all`;
- reduction: valid-token mean except `1×all`, whose one-answer mean is the
  mathematically identical answerwise scalar.

All configs pin pipeline v2, `conn_window: 1`, `grad_accum: 1`, a nonzero
learning rate, and the v5 cache. Short answers disappear from later K tiles
without replacement and contribute neither numerator nor denominator. The
complete causal prefix is retained for every selected answer; K slices only
the aligned loss rows.

The queue is `scripts/queue_pareto_v2_4b_screen_20260714.tsv`. A human launch
uses the shared lease allocator and one worker per physical card:

```bash
QUEUE=scripts/queue_pareto_v2_4b_screen_20260714.tsv \
SCHED=runs/.sched-pareto-v2-4b-$(hostname -s) \
JOBLOG_DIR=runs/pareto_v2_4b_worker_logs \
GPUS="0 1 2 3" MAX_PER_GPU=1 \
nohup setsid bash scripts/gpu_scheduler.sh \
  >> runs/pipeline_pareto_v2_4b_screen.log 2>&1 &
```

Per the logged-launch rule, an agent delegates that launch to a small worker
and then retains parent-level supervision. Completion means the individual
`report_manifest.json` exists; a checkpoint without its strict-local evidence
cannot be reported or satisfy the queue row.

### Superseded Qwen3.5-4B readout cohort

These four completed runs are retained for provenance and diagnosis only. They
were trained before the strict block-local policy and include behavioral
readout/final-logit training. They are superseded historical diagnostics, not
members of the Pareto frontier and not controls for new strict-local arms:

| run | historical status |
|---|---|
| `pareto_v2_qwen35_4b_huber_remove_token` | complete; superseded |
| `pareto_v2_qwen35_4b_huber_pad_random_token` | complete; superseded |
| `pareto_v2_qwen35_4b_lens_kl_remove_token` | complete; superseded |
| `pareto_v2_qwen35_4b_lens_kl_pad_random_token` | complete; superseded |

Their individual reports remain useful as historical per-run records, but
final synthesis must exclude them from strict-local frontier evidence.

## Epoch-zero teacher controls

This table is the live campaign record and is updated as soon as each control
finishes. It is independent of the per-training reports, which do not exist
until their training is complete. Each result must retain its output path and
timing/provenance rather than only a rounded summary.

| model | standard baseline | deleted RAG | randomized-token RAG |
|---|---|---|---|
| Qwen3.5 4B | complete: ARC-E 0.70, ARC-C 0.60, HellaSwag 0.59; macro 0.630 | complete: M/Q1/Q4 word accuracy 0.156/0.156/0.147 | complete: M/Q1/Q4 word accuracy 0.211/0.170/0.136 |
| Qwen3.5 9B | complete: ARC-E 0.70, ARC-C 0.57, HellaSwag 0.64; macro 0.637 | complete: M/Q1/Q4 word accuracy 0.198/0.205/0.173 | complete: M/Q1/Q4 word accuracy 0.183/0.192/0.182 |
| Gemma4 26B-A4B | complete: ARC-E 0.28, ARC-C 0.33, HellaSwag 0.32; macro 0.310 | complete: M/Q1/Q4 word accuracy 0.103/0.176/0.178 | complete: M/Q1/Q4 word accuracy 0.137/0.266/0.213 |
| Qwen3.6 35B-A3B | complete: ARC-E 0.69, ARC-C 0.58, HellaSwag 0.68; macro 0.650 | complete: M/Q1/Q4 word accuracy 0.162/0.205/0.179 | complete: M/Q1/Q4 word accuracy 0.136/0.202/0.182 |
| Qwen3.6 27B | complete: ARC-E 0.72, ARC-C 0.58, HellaSwag 0.68; macro 0.660 | complete: M/Q1/Q4 word accuracy 0.185/0.225/0.221 | complete: M/Q1/Q4 word accuracy 0.241/0.244/0.270 |
| Gemma4 31B | complete: ARC-E 0.32, ARC-C 0.26, HellaSwag 0.38; macro 0.320 | complete: M/Q1/Q4 word accuracy 0.000/0.001/0.005 | complete: M/Q1/Q4 word accuracy 0.042/0.092/0.118 |

## Per-training report v2

The atomic artifact is `runs/<run_name>/report.md`, generated immediately after
each run, with one report per
`dataset × model × censorship × loss type × update geometry` training. The
collection contract is in `docs/report_v2.md`. Campaign training configs must
collect recall and standard damage at epoch 0 and every epoch, per-layer losses
every epoch, and per-layer parameter modification from the epoch-0/base
reference every epoch. One-row density plots are retained in each individual
report so final synthesis can produce campaign-wide or grouped views by model,
loss, censorship, or geometry.

Completed controls above are stored under
`runs/flagship_teacher_evals/<model>/`. Standard baselines use 100 examples per
task. Censorship controls use 24 examples per task for each of Machado,
Quijote chapter 1, and Quijote chapter 4. The full JSON, including examples,
generation cuts, prompt regime, and exact task breakdown, remains the primary
evidence.

The vLLM timing/provenance for the newly completed controls is retained in
their JSON. Gemma4 26B loaded/generated in 184.5/22.0 seconds (`remove`) and
175.0/135.6 seconds (`pad_random`); Qwen3.6 35B `pad_random` took
195.5/42.2 seconds. The Gemma outputs have high hard-cut fractions under the
fixed reference-length-plus-48-token evaluation budget (M/Q1/Q4:
0.972/0.917/0.972 for deletion and 0.833/0.833/0.806 for randomization).
These are recorded as bounded corruption measurements, not silently treated
as natural-stop generation; the full example-level cuts remain available for
later checkpoint-matched interpretation.

Qwen3.6 27B loaded/generated in 189.7/40.3 seconds (`remove`) and
122.6/168.5 seconds (`pad_random`), with hard-cut fractions
0.306/0.181/0.292 and 0.208/0.014/0.056 respectively. Gemma4 31B deletion
took 84.6/67.1 seconds and produced near-zero recall with
0.986/0.931/0.931 hard cuts; this bounded failure mode is retained as the
corruption baseline rather than discarded.

Gemma4 31B randomized censorship completed with synchronous pipeline-parallel
vLLM scheduling: load 100.9 seconds, generation 670.5 seconds, and M/Q1/Q4
hard-cut fractions 0.875/0.639/0.667. Its artifact records
`async_scheduling: false`; the earlier failed async attempts produced no JSON
and are retained only in the worker log as failure provenance.

## Loss and censorship plan

The primary matched design uses both sanctioned RAG censorship modes for every
model:

- `remove`: delete the RAG information section;
- `pad_random`: replace it with length-matched randomized ordinary tokens.

The proposed first complete loss matrix is:

- `huber`: robust teacher-hidden-state matching, the geometric baseline used
  by the v5rs base configs;
- `lens_kl`: depth-uniform Kullback–Leibler divergence between teacher and
  student distributions through the model's frozen final norm and vocabulary
  head. It needs no learned or external lens artifact.

This produces `6 models × 2 censorships × 2 losses = 24` strict-local primary
trainings. Every layer receives the same loss treatment. `lens_kl` is a local
frozen-head metric, not a behavioral readout: no head update, final-logit
objective, or cross-block credit is allowed. The four completed 4B
readout-bearing arms are historical diagnostics and are excluded from this
frontier matrix.

### Jacobian/Anthropic lens inventory

Audited 2026-07-14 under `../jacobian-lens/out/`. “Reference” below means the
independent Anthropic/Neuronpedia-style Salesforce WikiText fit; “local” means
our own fitted Jacobian lens and is not interchangeable provenance.

| model | independent reference lens | local fitted lens | primary `lens_kl` available |
|---|---|---|---|
| Qwen3.5 4B | no local artifact | no | yes |
| Qwen3.5 9B | no | no | yes |
| Gemma4 26B-A4B | no | yes, n=150 | yes |
| Qwen3.6 35B-A3B | no | no | yes |
| Qwen3.6 27B | yes, n=1000 | yes, n=300 | yes |
| Gemma4 31B | yes | yes, n=200 | yes |

Therefore `jacobian_lens_kl` is not part of the primary cross-model matrix yet.
Using it only where an artifact happens to exist would confound model family
with lens availability and provenance. A later Jacobian-loss phase requires a
compatible independently generated lens for all six models, validation of
layer count and hidden width, and a separate report identity recording the
lens artifact hash and fit corpus.

### Three-dimensional update-grid experiment

The trainer's measurement space is `answer × aligned token × layer`. A grid
update selects `B` answers and the next `K` aligned rows from each, then walks
layers strictly forward. Student block `L` receives student `h[L-1]`; cached
teacher `h[L]` is its target, with cached teacher `h[L-1]` also available to
delta objectives. The optimizer steps only after all layers have contributed.

Every selected token retains its full causal prefix. Token tiling therefore
changes loss/gradient rows and update grouping, not model context. Exact
per-answer coordinate ranges are logged because short-answer tails are ragged.

| field/regime | geometry | reduction | status |
|---|---|---|---|
| `grid` | explicit `answers_per_update × tokens_per_answer_update`; token width 0 = all | independently `answer_mean` or `token_mean` | canonical v2 design |
| `answer` | `1 × all` | one answer mean | compatibility path |
| `token` | physical padded batch × all | valid-token mean | compatibility path; first four full runs |
| `legacy_answer_sum` | historical accumulation | sum of answer means | pipeline-v1 reproducibility only |

The four arms launched at 18:29 are accurately the `B=8 × K=all,
token_mean` full-minibatch reference. Calling them “one-token” runs would be
wrong. The new matched experiment pins initialization, example order, dataset,
censorship, loss, total selected token budget, learning rate, clipping, and
optimizer while varying B, K, and—only in separately named arms—the reduction.

Grid telemetry records completed answers separately from repeated answer
visits, cumulative selected aligned tokens, exact coordinates, optimizer
steps, per-layer losses, selected `answer × token × layer` measurement cells,
full causal `sequence token × layer` compute cells, padding, and throughput.
Quality comparisons are made at matched item/token budgets and by epoch,
including epoch zero. Full semantics are in `docs/training_pipeline_v2.md`.

### Speed gate

Before the loss/censorship matrix expands, Qwen3.5 4B measures training-only
time separately from epoch-boundary recall/damage. `B=8,K=all` projects 76.1
seconds per epoch. The requested production-like `B=8,K=1` probe completed
without error at 0.145 seconds median per tile and 8.93 GiB peak reserved, but
projects 5,600.9 seconds (93.3 minutes) for full aligned-token coverage—about
74 times the full-minibatch epoch. The one-token tile is mechanically healthy;
repeating the full causal layer walk is the cost.

The diagnostic table now traces constant-area power-of-two diagonals with an
inexpensive nonzero hidden loss, readout disabled, and real AdamW updates:
64 cells (`1×64 … 8×8 … 64×1`), 32 cells (`1×32 … 32×1`), and 16 cells
(`1×16 … 16×1`). It then fills the rectangle in the empirically faster
direction. No quality arm is launched from an unmeasured speed recipe.

The resulting full design size is decided only after this geometry gate. The
24 model × censorship × loss trainings remain the scientific core; selected
geometry/reduction replications add explicit dimensions rather than silently
redefining those runs.

### Pipeline-v2 strategy axes

Pipeline v2 treats the mechanism as a typed product of independent axes. Every
run pins and reports all axes, including those held at the current baseline:

| axis | initial implemented values | reserved future values |
|---|---|---|
| update geometry/reduction | `grid` with explicit B, K, `answer_mean`/`token_mean`; compatibility `answer`/`token` | logical tiles split across physical micro-batches |
| trajectory-state source | `student_hidden` | `teacher_hidden` |
| attention source | `student_attention` | `teacher_attention` |
| expert routing | `black_box` | `teacher_routing_cache` |

`student_hidden` advances from the student's detached trajectory;
`teacher_hidden` supplies the corresponding frozen teacher state as the
window root. `student_attention` computes ordinary student attention;
`teacher_attention` will replay a teacher-sourced attention artifact once its
exact representation is specified. `black_box` trains a sparse-expert block
through its combined output; `teacher_routing_cache` will replay cached
teacher routing choices.

These axes are independent of loss and censorship. They must be carried in
configuration, run identity, telemetry, and report provenance. Until a value
is implemented and certified, dispatch raises a specific unsupported-strategy
error. No future-facing switch may be accepted as a no-op.
