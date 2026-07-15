# Training pipeline v3: online local writes

Pipeline v3 is the online-learning limit of the dataset-v5 layerwise
experiment. Its atomic event is one aligned token from one answer. The model
walks blocks in forward order; each block computes its own hidden-state loss
through a detached input. The minimum-memory dispatch backpropagates and
writes before the next block; the disconnected-token dispatch enters autograd
once and writes all disjoint block gradients before the next token.

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

`backward_dispatch` has five execution choices with the same local objective
and write law.
`per_block` invokes backward and writes after every block, minimizing live
graph memory. `per_token_disconnected` retains only one token's isolated
block graphs, sums their disconnected scalar roots for one autograd-engine
invocation, and writes every block before the next token. Because every block
input remains detached and parameter sets are disjoint, this does not average
gradients, widen K, or create cross-block credit. It trades one token of graph
memory for lower Python/dispatcher overhead and is particularly relevant when
small models run no faster than large ones.

`answer_wavefront_disconnected` exploits the fact that the complete answer is
known during training. In the layer-by-token grid, cell `(L,t)` depends on
`(L-1,t)` and `(L,t-1)`. The trainer walks anti-diagonals of this grid, so every
cell on a diagonal is dependency-ready and belongs to a different block. One
autograd invocation handles those disconnected roots and every block is
written before its next token cell. Thus each block still observes tokens in
causal order, downstream blocks still consume the pre-write hidden value, and
there are no stale weights, averaged gradients, or cross-block paths. Live
training state is one diagonal/frontier plus the causal caches, rather than a
graph for the whole answer. The initial implementation is the LoRA,
`student_hidden`, `causal_frozen_history` path; the sequential `per_block` path
remains the minimum-memory full-weight implementation.

The serial wavefront proves the dependency transformation but does not itself
overlap CUDA work. `answer_pipeline_lanes` is its concurrent executor: one
CUDA stream and host lane owns each block, with a depth-one queue carrying the
pre-write hidden value to the next layer. This is the same exact grid order,
but multiple anti-diagonal cells can execute at once. The one-cell frontier
bounds activation retention independently of answer length.

Teacher-hidden training exposes a still wider schedule. Since every block
receives uncensored teacher `h[L-1]`, blocks have no same-token dependency at
all. `teacher_layer_lanes` assigns one causal CUDA lane to each block: tokens
remain ordered and immediately written within a lane, while all block lanes
run concurrently. It retains at most one local cell graph per block and does
not average across either axis. This is the massively parallel dreaming form;
student wavefront is the stricter deployment-matched form.

Both lane-parallel schedules currently require block-private causal state. Models
such as Gemma4 expose cross-layer shared KV state, adding edges to the simple
grid above; runtime rejects wavefront/lanes on those architectures until a
dependency-aware partition is implemented. Their exact baseline remains the
token-major dispatch. The lane implementation initially accepts stateless
geometric losses so lazy shared loss caches cannot race between host threads.

For a model with `N` blocks and an aligned answer span of `A` tokens, one
answer therefore causes `A*N` immediate writes. Metrics record both token
events and physical optimizer writes so an epoch always means one complete
dataset traversal.

The orthogonal `online_write_dispatch` knob controls when state-free SGD is
applied. `after_backward` uses a fused multi-tensor write after the local
backward. `grad_ready` installs post-accumulation hooks: each trainable tensor
is updated and its gradient released as soon as autograd materializes it.
There is no optimizer lock or block-wide write barrier; an on-device scalar
still accumulates the exact per-block gradient norm for telemetry. This is a
measured execution alternative, not a different objective or accumulation
law.

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
frozen model copy for full-weight training. V3 retains one answer's block
inputs on the owning GPUs and discards them after that answer. This costs only
tens to low hundreds of MiB for the tested models and avoids one tiny
host-to-device transfer in every layer×token cell; the historical
`full_states_cpu` source remains for schedules that genuinely need host
residency.

Teacher forcing makes blocks independent except for execution order and is
the dreaming/parallelizable form. It never changes the teacher input to a
censored trajectory.

## Stale-gradient token windows: the v2-speed bridge

`train.stale_gradient_window` makes the approximation boundary explicit.
The default `1` is exact online SGD: token `t+1` evaluates its gradient after
token `t` has changed the block. Values above one evaluate K known-answer
tokens at a shared weight snapshot; `0` means the whole remaining answer.
The first implementation is deliberately restricted to LoRA,
`teacher_hidden`, `causal_frozen_history`, `per_block`, `after_backward`, and
stateless geometric hidden losses.

The window loss is multiplied by its number of valid tokens before backward.
Consequently the optimizer receives `sum_t g_t`, not `mean_t g_t`, and the
learning rate is unchanged. Under state-free SGD, applying gradients already
computed at the same snapshot one by one has final value
`W - lr * sum_t g_t`; one fused physical write is exactly that replay. The
only approximation is gradient staleness: later tokens are not recomputed
after earlier logical writes. Metrics therefore report both `K*N` conceptual
token/block writes and `ceil(K/window)*N` physical writes, and name the
gradient-norm statistic as a window-sum norm normalized per token.

This is the controlled continuum between deployment-matched online learning
and pipeline-v2-style dreaming. It is not AdamW accumulation and does not use
batch averaging as regularization. The throughput oracle is pipeline v2's
Qwen3.5-4B broad-token regime (roughly 2–3k aligned token events/s versus
roughly 10/s for the initial exact K=1 v3 executor). Promotion compares K=1,
4, 8, 16, 64, and all at a matched logical-token budget, reporting speed,
recall, damage, and parameter deltas separately. The K=8 arm is the first
calibration point: persist exact parameter deltas for K=1 and K=8 at the same
seed/item order and report global plus per-layer divergence, not only endpoint
delta magnitudes.

The mask-free cached-attention optimization is exact only for q=1: the sole
query cannot see a future key because the dynamic cache contains only its
prefix. A K>1 stale window must retain a causal mask within its chunk. The
implementation shares one K×prefix mask across same-device layers and retains
only the current window, avoiding an unconditional answer-wide T² tensor.

Pipeline v3 currently rejects sliding/chunked-attention architectures rather
than approximating their authoritative mask semantics with a rolling window.
Gemma4 remains blocked until its chunk-aware adapter is implemented.

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
  stale_gradient_window: 1
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
