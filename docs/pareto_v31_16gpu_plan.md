# Pipeline-v3.1 16-GPU Pareto plan

Date: 2026-07-15. Code baseline: `d1aa8d0`. Dataset: v5. Training
pipeline: v3.1. The mechanics model is Qwen3.5-0.8B; recipes are promoted to
Qwen3.5-4B only after the 0.8B screen is scientifically usable and stable.

## Current evidence and hard gates

The allocation comprises sixteen L40S GPUs: four each on `agpul02`,
`agpul04`, `agpul05`, and `agpul06`. All nodes share Lustre. At the planning
snapshot twelve GPUs were idle; three GPUs on `agpul04` and one on `agpul06`
still hosted older work. Node-local staging may proceed while they drain, but
the matched screen starts from one shared queue only after all sixteen slots
are available.

A durable Qwen3.5-0.8B cache exists at
`runs/teacher_cache_h100/artifacts/selfupdate-cache-full-qwen35-08/` with
2,071 examples and 18.75 GB of hidden states. It is not campaign-eligible:
58.52% of its source answers are hard cuts, versus the RAG gate's maximum of
10%. This is a measured allowance regression, not evidence that the model
cannot finish: the earlier fixed `--generation-max-tokens 4096` run stopped
naturally on 93.19% of examples (6.81% hard cuts) and scored 0.6087, whereas
the later per-record ceilings are commonly only 104--116 tokens on short
Machado targets and score 0.5260. The optimization saved about 27 seconds of
H100 generation but invalidated most targets.

The earlier 4096-ceiling response artifact predates exact generated-token-ID
preservation, so do not decode/re-encode it into the cache. Regenerate under
the same `cache.generation_max_tokens: 4096` protocol with the current exact-ID
writer, then rerun real-RAG,
no-RAG, and same-length random-RAG epoch-zero controls. Change the prompt only
if that restored allowance still fails retrieval-use or completion checks.
Build a fresh cache identity after the gate passes. The existing cache may be
used only for explicitly labelled architecture/timing probes.

Qwen3.5 alternates three recurrent `linear_attention` blocks with one
`full_attention` block. Pipeline v3.0 supports this at B=1. The initial v3.1
B×K probe supports only full attention, so promotion requires a bounded
hybrid-state adapter:

- recurrent/conv state is batched by user and advanced by the current K
  queries;
- linear-attention padding masks cover the current query rows, not the
  full-attention KV timeline;
- full-attention blocks retain the explicit B×K causal and RAG-flow mask;
- finished-answer cells are excluded from both loss and gradient sum; and
- B256K16 remains labelled teacher-prefetched or speculation-confirmed, never
  ordinary next-token online learning.

Certification before the queue opens:

1. Intact B1K1 and B256K1 remain at numerical-noise loss and weight movement.
2. Qwen3.5 B1K1 matches the existing v3.0 path from the same seed and cells.
3. A B256K1 frozen-snapshot gradient matches the explicit sum of its 256
   per-user gradients; this certifies the unaveraged reduction.
4. K16 is compared against sixteen K1 cells for loss, gradient, and delta;
   their expected difference is named staleness, not hidden averaging.
5. B256K1 and B256K16 pass the cache-graph and frozen-vocabulary tripwires and
   record peak VRAM. A one-card OOM changes B into a named experimental axis;
   it does not silently change the configured geometry.

## Wave A: stability and learning-rate scale (16 GPUs)

Every scientific arm runs six complete v5 epochs: 12,426 answer visits, thus
clearing the 12,000-training-item minimum. A short token-count smoke is not a
promotion run. All arms use B=256, teacher-hidden inputs, LoRA r16/alpha32,
immediate state-free SGD, and one unaveraged local write per block and tile.

| slots | censorship | K | loss | learning rates | purpose |
|---:|---|---:|---|---|---|
| 1-6 | flow mask | 1, 16 | Huber | 1e-6, 3e-6, 1e-5 | LR × staleness |
| 7-12 | random fill | 1, 16 | Huber | 1e-6, 3e-6, 1e-5 | censorship interaction |
| 13-14 | intact | 1, 16 | Huber | 1e-5 | no-censorship controls |
| 15-16 | flow mask | 1, 16 | cosine | 1e-5 | cheap objective challenger |

The fixed learning rate is the coefficient of every cell gradient. B×K
gradients are summed, not averaged; no automatic `1/B`, `1/K`, or
`1/sqrt(BK)` scaling is hidden in the implementation. The LR grid measures
the stability consequence of evaluating those gradients at one shared weight
snapshot.

Physical slot mapping is stable for diagnosis: `agpul02` GPUs 0-3 are slots
1-4, `agpul04` slots 5-8, `agpul05` slots 9-12, and `agpul06` slots 13-16.
Workers consume one Lustre-shared queue with host-scoped leases and
`MAX_PER_GPU=1`. Per-node scheduler state/logs and per-host worker-log
directories follow `AGENTS.md`.

Before opening workers, each node stages model/cache data and runs one
delegated Python warm-up. The vLLM environment uses:

```bash
scripts/warm_python_runtime.sh ../venvs/vllm025/bin/python torch transformers vllm
```

The trainer uses the same command with the Python selected by
`scripts/l40s_exec.sh` and modules `torch transformers peft`. This parallel-
stats the Lustre-hosted runtime and pre-imports it into the node's VFS/page
cache; it does not clone a venv. The 2026-07-15 ad-hoc 0.8B regeneration
staged model weights but skipped this step, leaving the GPU empty for roughly
a minute during serial Python metadata reads.

Generated compiler state is also node-local. vLLM, TorchInductor, and Triton
use `/tmp/$USER/selfupdate-vllm-*`; they must not fall back to
`~/.cache/vllm` on Lustre. This cache is disposable and distinct from staged
Hugging Face snapshots in `/dev/shm`. The first corrected 0.8B regeneration
had already reached a healthy compile when the misplaced cache was noticed
(92.73 seconds total), so it was not restarted; the rule applies to every
subsequent launch.

The initial placement balances two K1 and two K16 arms on every node. K16 is
expected to finish much earlier; after its report is reviewed, that physical
slot may pull the corresponding Wave-B K16 objective arm while K1 continues.
This is adaptive queue refill, not an unreviewed pipeline: a failed control or
tripwire closes the dependent refill.

The existing 0.8B cache contains 381,532 aligned token events per epoch, or
2.289 million over six epochs. Using the 0.6B B256 measurements only as a
planning bound, tile work would take about 2.2 hours at K1 and 9.4 minutes at
K16; teacher/prefill materialization adds roughly 2.5 minutes across the 54
six-epoch batches before evaluations. Qwen3.5 recurrent-state performance and
the fresh cache length distribution can change these numbers, so the adapter
probe replaces them with measured projections before launch. Evaluation and
PDF generation are budgeted separately.

## Wave B: objective geometry (16 GPUs)

For each `(censorship, K)` cell, carry forward the best non-destructive Wave-A
learning rate and run four depth-uniform local objectives:

| censorship × K cells | objectives | arms |
|---|---|---:|
| flow × {1,16} | Huber, cosine, Charbonnier, delta-cosine | 8 |
| random × {1,16} | Huber, cosine, Charbonnier, delta-cosine | 8 |

`delta_cosine` is admitted only after v3.1 explicitly calls the increment
loss on raw block input/output differences. Falling back to its absolute-state
boundary metric would be a wiring bug and must hard-stop the config audit.
If an objective was already measured at the selected LR in Wave A, its slot is
used for a second seed rather than rerunning an identical arm.

## Wave C: replication and 0.8B selection (16 GPUs)

Select the four non-dominated recipes over recall, standard damage,
intrusion, stability, and elapsed time. Run each at seeds 17, 29, 43, and 71.
Selection rejects any arm with a changed vocabulary fingerprint, retained
cross-token graph, NaN/Inf, unexplained intact drift, or missing report. It
also records per-layer gradient share and parameter delta so a superficially
good endpoint cannot hide a layer-1 collapse or tail-only surrogate.

Each run immediately builds its individual report and PDF. The completion-
ordered `runs/report_v2_index/` symlinks are refreshed after publication.
Reports include per-layer loss heatmap and temporal traces, parameter deltas,
epoch-zero and per-corpus recall, standard damage, frontier position, exact
B/K/LR/censorship provenance, cache identity, timings, and missing-evidence
coverage. The parent agent reviews compact delegated worker logs and the
scientific telemetry at least every 30 minutes.

## Promotion to Qwen3.5-4B

Promotion starts only after the 0.8B seed wave identifies stable Pareto
recipes. First certify a fresh 4B real/no/random RAG gate and cache identity,
then measure B256K1 and B256K16 VRAM and throughput on one L40S each. The
sixteen-card confirmation wave is:

- slots 1-8: the four 0.8B finalists at seeds 17 and 43;
- slots 9-12: the opposite-K counterpart of the best two recipes at both
  seeds, preserving the online-compatible versus lookahead comparison; and
- slots 13-16: intact K1 and K16 controls at both seeds.

The 4B wave keeps the selected per-cell LR unchanged initially because it is
a per-gradient coefficient, then adds a named scale correction only if the
mechanics probe shows materially different normalized gradient/delta scales.
No 4B result is declared successful without its individual PDF and the same
recall/damage/layer-dynamics evidence used at 0.8B.
