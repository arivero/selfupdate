# Pareto-frontier training progress

This is the running record for the six-model Pareto-frontier campaign. Qwen3.5
4B is the flagship experimental model within the campaign, not the name of the
campaign as a whole. The campaign is
explicitly pinned to **dataset v5** and **training pipeline v2**.

Status at initialization: the teacher-answer and hidden-state cache phase is
complete for all six models; student training has not yet started.

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
| 2026-07-14 15:32 CEST | Remaining epoch-zero vLLM queue launched | running | Two concurrent 2-GPU jobs from RAM-staged HF cache; Gemma4 26B deletion and randomized controls first |
| 2026-07-14 | Six exact-cache training bases and 4B probe grid | validated | `configs/experiments/pareto_v2/`; every base resolves its certified 2,071-example cache hash, eight probe overlays pass typed dispatch validation |
| 2026-07-14 | Gemma4 31B randomized control async-scheduler diagnosis | corrected/requeue | vLLM 0.25 async pipeline scheduling twice violated output-placeholder accounting; evaluator now defaults to synchronous scheduling and records the choice |
| pending | Qwen3.5 4B student training | pending | — |
| pending | Qwen3.5 9B student training | pending | — |
| pending | Gemma4 26B student training | pending | — |
| pending | Qwen3.6 35B student training | pending | — |
| pending | Qwen3.6 27B student training | pending | — |
| pending | Gemma4 31B student training | pending | — |

Future entries must record the start/end timestamp, model/config, GPU
placement, dataset identity, pipeline-v2 commit, checkpoint path, item
count, and any failure or restart reason.

## Epoch-zero teacher controls

This table is the live campaign record and is updated as soon as each control
finishes. It is independent of the per-training reports, which do not exist
until their training is complete. Each result must retain its output path and
timing/provenance rather than only a rounded summary.

| model | standard baseline | deleted RAG | randomized-token RAG |
|---|---|---|---|
| Qwen3.5 4B | complete: ARC-E 0.70, ARC-C 0.60, HellaSwag 0.59; macro 0.630 | complete: M/Q1/Q4 word accuracy 0.156/0.156/0.147 | complete: M/Q1/Q4 word accuracy 0.211/0.170/0.136 |
| Qwen3.5 9B | complete: ARC-E 0.70, ARC-C 0.57, HellaSwag 0.64; macro 0.637 | complete: M/Q1/Q4 word accuracy 0.198/0.205/0.173 | queued |
| Gemma4 26B-A4B | complete: ARC-E 0.28, ARC-C 0.33, HellaSwag 0.32; macro 0.310 | complete: M/Q1/Q4 word accuracy 0.103/0.176/0.178 | complete: M/Q1/Q4 word accuracy 0.137/0.266/0.213 |
| Qwen3.6 35B-A3B | complete: ARC-E 0.69, ARC-C 0.58, HellaSwag 0.68; macro 0.650 | complete: M/Q1/Q4 word accuracy 0.162/0.205/0.179 | complete: M/Q1/Q4 word accuracy 0.136/0.202/0.182 |
| Qwen3.6 27B | queued | queued | queued |
| Gemma4 31B | queued | queued | queued |

## Per-training report v2

The future atomic artifact is `runs/<run_name>/report.md`, with one report per
`dataset × model × censorship × loss type` training. The collection contract
is in `docs/report_v2.md`. Campaign training configs must collect recall and
standard damage at epoch 0 and every epoch, per-layer losses every epoch, and
per-layer parameter modification from the epoch-0/base reference every epoch.
Collective one-row density plots are retained in each individual report so
later reports can synthesize arbitrary selections by model, loss, censorship,
or dataset.

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

This produces `6 models × 2 censorships × 2 losses = 24` primary trainings.
Every behavioral target remains teacher-sourced, and every layer receives the
same loss treatment. Any connected-window readout must remain the sanctioned
`teacher_kl` readout with no reference-text cross-entropy.

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

### Gradient aggregation experiment

The 2026-07-14 trainer audit found that the existing summed path is a hybrid:
it forwards a padded mini-batch of complete answers, reduces tokens to one
mean loss per answer, sums those answer losses, and steps after
`train.grad_accum` answers. Consequently short and long answers have equal
weight, an update mixes answers, and changing micro-batch/accumulation changes
the pre-clipping gradient scale. It must not be silently called either of the
new regimes.

The pipeline-v2 trainer will expose and log an explicit update granularity:

| regime | optimizer update | loss normalization | purpose |
|---|---|---|---|
| `legacy_answer_sum` | historical accumulated answers | sum of per-answer token means | reproducibility only |
| `answer` | one complete answer | mean over that answer's valid aligned tokens | coherent within-answer gradient |
| `token` | mini-batch of different complete answers | mean over all valid aligned tokens in the update | cross-answer token gradient |

Both experimental regimes still perform full causal forwards: an answer token
cannot be detached from its prefix without changing the model input. “Token”
describes gradient weighting and grouping, not isolated-token inference.

The matched comparison uses identical model initialization, example order,
dataset, censorship, loss, total examples, and total valid aligned tokens.
It deliberately reports different optimizer-step counts: one update per answer
is part of the answerwise hypothesis, while tokenwise batching is part of the
tokenwise hypothesis. Initial learning rate, clipping, optimizer, and item
budget stay pinned. Any learning-rate scaling is a separately named second
stage, not folded into the comparison.

Every telemetry row and report identity records update granularity,
micro-batch, answers per update, valid tokens per update, cumulative answers,
cumulative aligned tokens, optimizer steps, padding fraction, and throughput.
Quality comparisons are made at matched token/item budgets and by epoch,
including epoch zero.

### Speed gate

Before the 24-run loss/censorship matrix expands across update granularity, run
short probes on Qwen3.5 4B for both losses, both censorships, and both update
regimes. Extrapolate a full 2,071-example epoch from steady-state batches and
record training-only time separately from epoch-boundary recall/damage time.
The target is an epoch not substantially slower than that model's cache build;
if it misses, first tune bucketed micro-batch size and optimizer-step cadence,
then placement. Pipeline parallelism is used for capacity, not assumed to add
throughput. No quality arm is launched from an unmeasured speed recipe.

The resulting full primary design, if both update regimes pass the speed and
mechanics probes, is `6 models × 2 censorships × 2 losses × 2 update regimes =
48` trainings. The 24-training matrix remains the fallback if one aggregation
regime is rejected by the probe rather than silently redefined.

### Pipeline-v2 strategy axes

Pipeline v2 treats the mechanism as a typed product of independent axes. Every
run pins and reports all axes, including those held at the current baseline:

| axis | initial implemented values | reserved future values |
|---|---|---|
| gradient aggregation | `answer`, `token` | — |
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
