# Training pipeline v3: online local writes

Pipeline v3 is the online-learning limit of the dataset-v5 layerwise
experiment. Its atomic event is one aligned token from one answer. The model
walks blocks in forward order; each block computes its own hidden-state loss,
backpropagates only through that block, writes the gradient immediately, and
releases it before the next block runs.

## Epoch-zero target materialization

V3 separates the small durable teacher-answer record from the large hidden
target tensor store. Fixed answer token IDs remain the reproducible source;
each L40S node regenerates the corresponding uncensored hidden states once
with its own runtime and publishes them in node-local shared memory. Training
does not page hidden targets from Lustre and does not treat H100 rounding as a
learning signal.

The canonical location is
`/dev/shm/$USER/selfupdate-teacher-cache-v3`. Publication is coordinated per
cache identity and per host:

- atomic `mkdir` elects exactly one builder;
- the winner writes a private partial directory;
- waiters do not load model weights while epoch zero is being generated;
- `index.json` and a runtime manifest are validated before an atomic rename;
- later launches on the same host validate the manifest and skip epoch zero;
- another host performs its own epoch zero because its `/dev/shm` and GPU
  runtime are deliberately independent.

Use the combined L40S entry point for both a first and a repeated launch:

```bash
scripts/l40s_train_v3.sh \
  --config configs/experiments/pareto_v3/base_qwen35_4b.yaml \
  --experiment configs/experiments/pareto_v3/qwen35_4b_flow_student_cached_huber_lr1e5.yaml
```

The first invocation reports `epoch0 cache lease acquired` and, after the
teacher pass, `epoch0 cache ready`. Concurrent or later invocations report
`epoch0 cache wait` followed by reuse, or immediate `epoch0 cache reuse`.
Those are expected coordination states rather than duplicated work.
After training and locality certification complete, the same launcher creates
the run-local individual Markdown/PDF report and refreshes the
completion-ordered report symlink directory.

To materialize without starting training, run the first half explicitly:

```bash
scripts/l40s_exec.sh scripts/build_teacher_cache.py \
  --config configs/experiments/pareto_v3/base_qwen35_4b.yaml \
  --experiment configs/experiments/pareto_v3/qwen35_4b_flow_student_cached_huber_lr1e5.yaml \
  --coordinated-node-cache
```

The ready manifest records the model, cache hash, example count, source
commit, PyTorch/CUDA runtime, GPU name, dtype, response-token artifact,
teacher batch, and epoch-zero wall time. Anonymous process RAM is not used:
the safetensors files in tmpfs are memory-mapped and shared by all local arms.

### Qwen3-0.6B first gate (2026-07-15)

The first same-runtime cache used the full 2,071-example V5RS dataset and the
fixed L40S-vLLM answer tokens. Qwen3-0.6B produced 28 target layers in 29.8 s
(19.3 s teacher compute; asynchronous cache writing overlapped it) and a
second launcher reused the ready identity. Its aligned training span contains
306,321 token events (159,383 generated-answer tokens plus shared aligned
context), so the measured `per_block` rate projects to roughly nine hours per
complete epoch even at 0.6B.

On the longest aligned record (732 token events), `per_block` dispatch gave:

| censorship | token events/s | block writes/s | total raw LoRA delta L2 |
|---|---:|---:|---:|
| intact | 9.76 | 273.31 | 1.58e-5 |
| flow mask | 9.48 | 265.37 | 1.21e-3 |

Both runs changed only block adapters, retained no graph in causal history,
and passed the frozen-vocabulary tripwire. The intact displacement was about
76 times smaller than the censored displacement. The nearly identical 0.6B
and 4B event rates diagnose fixed Python/autograd dispatch as the current
speed limit; they do not establish a model-FLOP scaling law.

The first metaparameter screen uses 12,000 token events per arm (the minimum
promotion budget): intact Huber at 1e-5; flow Huber at 1e-6, 3e-6, 1e-5, and
3e-5; flow cosine at 1e-5; random-fill Huber at 1e-5; and a matched
`per_token_disconnected` dispatch arm. Only promoted arms receive complete
epoch/campaign interpretation.

This is deliberately not a smaller tile in pipeline v2. It removes the tile:

- `micro_batch: 1`, `grad_accum: 1`, `batching: item`;
- one local backward and one parameter write per token and block;
- no averaging or summation across answers, tokens, or layers;
- no AdamW, momentum, second moment, clipping, weight decay, or optimizer
  object; and
- embedding, final norm, and vocabulary head remain frozen.

`backward_dispatch` has two numerically equivalent execution choices.
`per_block` invokes backward and writes after every block, minimizing live
graph memory. `per_token_disconnected` retains only one token's isolated
block graphs, sums their disconnected scalar roots for one autograd-engine
invocation, and writes every block before the next token. Because every block
input remains detached and parameter sets are disjoint, this does not average
gradients, widen K, or create cross-block credit. It trades one token of graph
memory for lower Python/dispatcher overhead and is particularly relevant when
small models run no faster than large ones.

For a model with `N` blocks and an aligned answer span of `A` tokens, one
answer therefore causes `A*N` immediate writes. Metrics record both token
events and physical optimizer writes so an epoch always means one complete
dataset traversal.

## Trajectory source

`student_hidden` is the deployment-matched path. Block `L` consumes the
censored student's detached `h[L-1]` produced immediately before it. The
output passed to block `L+1` is the pre-write output of block `L`; the write
affects subsequent tokens.

`teacher_hidden` is uncensored teacher forcing. Block `L` consumes the
uncensored teacher's `h[L-1]` and targets uncensored teacher `h[L]`, while the
student block itself executes under the selected censorship. In v2 terms,
`teacher_hidden` means the uncensored teacher cache. The durable v2 cache only
stores aligned `h[L]` rows, however, so v3 obtains full `h[L-1]` prefixes from
an adapters-off resident teacher for LoRA, or from an explicitly requested
frozen model copy for full-weight training. It stages one answer's states in
host RAM and discards them after that answer.

Teacher forcing makes blocks independent except for execution order and is
the dreaming/parallelizable form. It never changes the teacher input to a
censored trajectory.

## Censorship

V3 scientific modes are:

- `flow_mask`: preserve original tokens and positions, but zero privileged
  rows before and after every block and exclude them from attention/state
  writes;
- `pad_random`: replace privileged tokens with deterministic distinct
  ordinary-vocabulary tokens while preserving length; and
- `intact`: uncensored diagnostic control.

`remove` and `remove_gap` are not v3 modes. Information-flow masking is more
generic and preserves the sequence geometry. For hybrid models such as
Qwen3.5, a softmax attention mask alone is insufficient: linear-attention and
causal-convolution blocks also receive explicitly zeroed privileged rows.

The frozen teacher targets remain uncensored in every mode. A cache produced
under `source_compaction: remove` is valid because its payload and generated
answer IDs are teacher-view data; student censorship is reconstructed from
dataset v5.

## Causal history

`history_policy: recompute_prefix` reruns the complete current-weight prefix
for every aligned token. It is the simplest reference semantics and the
slowest implementation.

`history_policy: causal_frozen_history` builds prompt state once, appends each
new token, detaches that state after the token's local backward, and retains
it unchanged for later tokens in the same answer. The cache is discarded at
the answer boundary. It is rebuilt with the latest weights for the next
answer and again on the next dataset epoch. This matches online conversation:
past K/V, convolution, and recurrent state were produced in the past and are
not retroactively rewritten.

## Learning rate

The first runtime implements `lr_rule: fixed`. This is explicit rather than a
claim that the pipeline-v2 learning rate transfers unchanged. Calibration
arms should measure one-token local curvature and gradient norms, then sweep a
conservative bracket. Candidate future rules include normalized LMS for
matrix-local updates and a curvature estimate
`||g||^2 / (g^T H g)`. Any adaptive rule must remain state-free across token
events and become a named, logged experiment variable.

## Configuration

The required core is:

```yaml
mask:
  compaction: flow_mask
cache:
  source_compaction: remove
train:
  pipeline_version: 3
  update_granularity: online
  online_optimizer: immediate_sgd
  lr_rule: fixed
  history_policy: causal_frozen_history
  trajectory_source: student_hidden
  schedule: summed
  micro_batch: 1
  grad_accum: 1
  batching: item
  conn_window: 1
  conn_stride: 0
```

The validator rejects unused v2 geometry, connected windows, Adam offload,
removal censorship, and unsupported trajectory/routing knobs rather than
silently translating them.

## Reporting and certification

Each run retains the atomic individual-report contract: per-layer loss and
gradient norm by epoch, parameter modification from epoch zero, recall with
epoch zero, standard damage, exact configuration/provenance, token events,
physical writes, trajectory source, censorship, history policy, and learning
rate rule. Pipeline-v3 reports must not be merged into the v2 frontier without
labeling the optimizer and update semantics.

Before a campaign launch, certify:

1. local backward changes only the intended block;
2. embedding, final norm, and vocabulary head remain bit-identical;
3. no padding or censored row contributes to a loss;
4. two different privileged passages with the same flow ranges produce the
   same post-censorship states and gradients; and
5. cached history contains no surviving autograd graph after each write.

The historical v1/v2 implementations remain recoverable from Git. V3 is a
separate dispatch path in `src/selfupdate/train/online_v3.py`.
