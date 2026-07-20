# Pipeline-v4 runtime guide

This document describes the only active training runtime on this branch:
pipeline v4. The scientific protocol and full knob contract live in
[`training_pipeline_v4.md`](training_pipeline_v4.md). Pipeline-v1/v2/v3
implementations and their student-trajectory training inputs are historical;
Git history and the explicitly archived protocol documents preserve them.

## Executable map

| file | responsibility |
|---|---|
| `scripts/train.py` | load configuration, select an optional v4 stage, and pin imports to this checkout |
| `scripts/launch_v4_stages.sh` | coordinate independent PPP stage processes |
| `src/selfupdate/train/layerwise.py` | construct the common runtime and dispatch pipeline v4 |
| `src/selfupdate/train/online_v4.py` | teacher tensors, block-local updates, validation relay, epoch battery, locality certification, and publication |
| `src/selfupdate/train/v4_store.py` | fill-once relayed teacher-store construction |
| `src/selfupdate/train/validate.py` | reject configurations outside the v4 contract |
| `src/selfupdate/train/losses.py` | absolute block-local hidden losses |
| `scripts/merge_v4_adapters.py` | assemble disjoint stage-owned adapters without averaging |
| `scripts/compare_v4_shard_numerics.py` | compare single-process and independently sharded results |
| `scripts/v4_battery.py` | evaluation subprocess for scoped or rotating models |

Every Python entry point inserts this checkout's `src/` directory into
`sys.path`. The shared node-local environment intentionally contains no
editable `selfupdate` installation.

## The sole training dataflow

For block `L`, pipeline v4 trains

```text
i_L = stopgrad(context h[L-1])
y_L = block_L^student(i_L; frozen teacher attention context)
loss_L = HiddenLoss(y_L, teacher h[L])
```

Both the residual input and attention context are detached and fixed during a
local step. By default the attention module receives adapters-off uncensored
teacher K/V; the legal-A diagnostic described below instead receives
adapters-off flow-censored K/V. Attention censorship is encoded in the mask.
The backward graph contains exactly the current block's trainable parameters;
it cannot reach another block, embedding, final norm, vocabulary head,
teacher tensor, or frozen context.

The student's own hidden trajectory is never a training input. A "student
step" means the block's trainable weights are enabled; it does not mean that
its input came from a preceding student block. `validate_knob_schedule`
enforces `pipeline_version: 4`, `trajectory_source: teacher_hidden`, a
block-local hidden loss, and `conn_window <= 1`. The sole increment variant,
`delta_cosine`, is anchored on the same detached teacher block input; it never
consumes a preceding student block's output.

## Teacher tensor production and residency

The runtime obtains the same typed per-(layer, cohort) product through one of
three sources:

- `cache`: read a full-teacher-input cache;
- `online`: run one adapters-off teacher forward for the cohort;
- `store`: perform one relayed teacher pass before training and retain the
  stage-local products.

The product supplies block input, target, and frozen attention context.
`gpu_corpus`, `cpu_stream`, and `rebuild` describe where or whether products
persist; they do not alter the objective. Every source must preserve the
default context `i_L = teacher h[L-1]` and target
`teacher h[L]` exactly.  The opt-in diagnostic
`train.v4_context_source: flow_censored_teacher` instead uses an
adapters-off, fully flow-censored `i_L = h_c[L-1]` for both the query and
frozen K/V, while retaining the uncensored `target_L = h_u[L]`.  Both are
detached: the trainable block L is still the only differentiable operation.
This repair is deliberately accepted only for online, fully resident,
single-process execution; cache/store, staged, rotary, compressed-context,
and recurrent routes fail loudly rather than inherit new semantics silently.

`v4_kv_source: student_refresh` is a narrowly named context refresh: adapters
are enabled when regenerating frozen K/V, but the projection inputs and block
residual inputs remain teacher hidden states. It is not student-trajectory
training.

## PPP means independent layer shards

Pipeline-v4 PPP is not ordinary pipeline parallelism and has no activation
handoff in training. `v4_stage_splits` partitions blocks into contiguous
ranges. Each stage process owns one range, trains only that range, and
publishes only its adapter tensors and manifest. Since every block already has
a teacher-fixed input and target, stages train independently.

```text
teacher store ─┬─> stage 0: blocks 1..a     ─> stage0 checkpoint
               ├─> stage 1: blocks a+1..b  ─> stage1 checkpoint
               └─> stage 2: blocks b+1..n  ─> stage2 checkpoint
```

No edge carries a training activation between stages. `model.pipeline_split(s)`
belongs to the retired PP loader and is rejected for v4; ownership is expressed
by `train.v4_stage_splits` and `v4_stage_devices`. Sharding is a placement
transformation, so equal-seed single-process and staged runs must agree block
by block.

Large-model stages may use scoped loading and block rotation. Rotation changes
residency only: frozen weights and, for Adam, matching optimizer state are
paged while the same teacher tensors and update are used.

## End-to-end student trajectory exists only for validation

Deployment validation deliberately performs the computation forbidden as a
training input. Under `torch.no_grad()`, `_student_trajectory_eval` embeds
censored student tokens and runs an ordinary full causal forward through every
block on the student's evolving states. It reports `student_trajectory_eval`
cross-entropy and KL divergence through the frozen head with
`evaluation_only=true`, `used_for_backward=false`, and
`optimizer_weight=0.0`.

In a single process this is one full walk. In a staged run, the validation
relay alone sends detached boundary states from one stage to the next; the
last stage computes the metrics. These boundaries are evaluation data, not
PPP training dependencies. Generation, recall, standard-damage, and
parameter-delta probes are likewise evaluation-only.

Cross-node stages use NCCL/InfiniBand for validation boundaries while
co-located neighbors use node-local `/dev/shm`. Subprocess batteries use a
separate NCCL communicator: every stage publishes its enveloped adapter shard,
stage 0 materializes remote shards in its own `/dev/shm` for the unchanged
battery child, and publishes the child's success/failure through the
launch-scoped TCPStore. This keeps adapter payloads off Lustre and avoids both
inserting collectives into the ordered boundary stream and holding an NCCL
collective open during a long evaluation.

## Locality, numerical, and publication gates

- Store-backed stage-scoped runs certify inline after relay drain/barrier and
  before Adam-state or checkpoint publication, while their exact fill-once
  entries and typed contexts are still alive.  The gate checks every owned
  layer for finite positive local signal, exact-zero gradients on every other
  real block and the frozen vocabulary/lens stack, and byte-exact preservation
  of adapters and optimizer state.  Cache/online sources retain the legacy
  `certify_locality_v4` path.
- `scripts/compare_v4_shard_numerics.py` checks that independent ownership
  reproduces corresponding single-process updates.
- Each stage writes a manifest; `scripts/merge_v4_adapters.py` takes every
  adapter tensor from its unique owner and never averages tensors.
- A failed locality or merge contract must not publish a completed model.
- Teacher-forced and student-trajectory CE/KL remain distinctly named. Both
  are evaluation-only, but answer different questions.

Validate configuration before spending GPU time, and use the stage launcher
instead of inventing stage commands:

```bash
/tmp/$USER/selfupdate-venv/bin/python scripts/audit_configs.py
scripts/launch_v4_stages.sh BASE.yaml EXPERIMENT.yaml
```

Use the node-local runtime and cache rules in `AGENTS.md`. Numerical claims
require v4 locality and shard-equivalence evidence; historical v1-v3
fingerprints are not a certification gate for this runtime.
