# A programmer's walkthrough of pipeline v4

This is the implementation guide for the active trainer. Pipeline v4 trains
each block from a teacher hidden vector to the next teacher hidden vector.
Student-hidden inputs used by pipeline v1-v3 are obsolete on this branch; the
student's ordinary full trajectory survives only as evaluation.

Read [`training_pipeline_v4.md`](training_pipeline_v4.md) for the scientific
protocol, [`runtime.md`](runtime.md) for execution and placement, and
`AGENTS.md` for cluster rules.

## 1. Keep one equation in view

For every owned block `L` and loss position `p`:

```text
input  = teacher h[L-1, p]
output = trainable block_L(input, frozen teacher context)
target = teacher h[L, p]
loss   = absolute_local_hidden_loss(output, target)
```

The input and target are detached teacher tensors. The output is called a
student output because the block's adapters are enabled, not because its input
came from the student's preceding block. Feeding `output_L` into block `L+1`
during training would silently reintroduce an obsolete pipeline.

## 2. Follow the entry-point route

```text
base YAML + experiment YAML
          |
          v
scripts/train.py
          |
          v
train/layerwise.py  -- common load/freeze/log setup and v4 dispatch
          |
          v
train/online_v4.py -- teacher products, local writes, evaluation, publication
       |       |
       |       +--> train/v4_store.py -- optional fill-once teacher store
       +----------> train/validate.py -- configuration law
```

`scripts/train.py --v4-stage K` selects the owned block range and physical
device for one independent stage. `scripts/launch_v4_stages.sh` derives the
process set from the config and coordinates it. Scripts pin `<repo>/src`
themselves; never install this checkout editable into the shared environment.

## 3. Validation defines the allowed experiment

The v4 branch in `train/validate.py` rejects configurations that change the
method. In particular:

- `trajectory_source` must be `teacher_hidden`;
- connected windows and delta/trajectory losses are forbidden;
- the teacher source is `cache`, `online`, or `store`;
- block ownership uses `v4_stage_splits`, never model pipeline splits;
- the optimizer is v4 immediate SGD or per-block Adam;
- attention censorship is `flow_mask`, with `intact` reserved as a control.

When adding a knob, decide whether it changes the objective, tensor residency,
placement, or evaluation. Placement must not affect teacher input, target,
loss reduction, random stream, or update ordering for an owned block.

## 4. Teacher products are typed inputs, not a trajectory cache

For each `(layer, cohort)`, `_TeacherTensors` supplies the values needed for
one independent block update:

- teacher `h[L-1]` rows used as block inputs;
- teacher `h[L]` rows used as targets;
- frozen teacher K/V or the architecture-specific recurrent context;
- positions and censorship masks that preserve teacher coordinates.

Cache, online, and store sources must produce the same logical product.
Residency then decides whether it remains on GPU, streams from host memory, or
is recomputed. Never reconstruct a missing teacher input from the output of a
trained lower block.

`v4_kv_source: student_refresh` does not relax this rule. It refreshes frozen
context with adapters enabled while continuing to project teacher hidden
inputs and feed teacher hidden residual inputs to the block.

## 5. One local update

The layer/cohort step stages a teacher product, runs only the selected block,
applies an absolute hidden loss against its teacher target, and updates that
block's trainable parameters. The input tensor does not require gradients.
Frozen K/V does not acquire a graph. Embedding, foreign blocks, final norm, and
vocabulary head remain outside the optimizer graph.

```text
detached teacher h[L-1] ---> [ trainable block L ] ---> local loss
                                      ^                    ^
                                      |                    |
                         frozen teacher context       teacher h[L]
```

There is no `next_input = output.detach()` in training. Loop order may be
layer-major or item-major because no layer consumes another layer's learned
output.

## 6. PPP stages are independent owners

`v4_stage_splits` divides the decoder into contiguous ranges. Every process
loads or rotates the weights it needs, receives its own teacher products, and
updates only its owned adapters. Training sends no activations between stages
and uses no wavefront.

At checkpoint time each process publishes its owned adapter tensors plus a
manifest. Those stage LoRAs are the durable artifacts; evaluation does not
merge them. Any later serving-only collation is temporary and selects each
tensor from exactly one owner rather than averaging. Compare a staged run
with the equal-seed single-process reference using
`scripts/compare_v4_shard_numerics.py`.

## 7. Recognize the validation-only student walk

`_student_trajectory_eval` and `_relay_segment` are intentionally different
from training. They are decorated with `torch.no_grad()` and perform the
ordinary censored forward:

```text
student token ids -> embedding -> block 1 -> ... -> block n -> frozen head
```

Here each block really consumes the preceding student's hidden state. The
result measures deployment behavior and logs `kind=student_trajectory_eval`,
`evaluation_only=true`, `used_for_backward=false`, and
`optimizer_weight=0.0`.

For staged evaluation, boundary hidden states relay between stage processes.
This is the only full student-hidden circulation in the trainer. Never reuse a
validation boundary as a training input, and never remove the relay merely
because activation handoffs are absent from PPP training.

Teacher-forced output evaluation is separate: it decodes the final block run
on teacher `h[n-1]`. Keep the two trajectories distinctly named in telemetry
and reports.

## 8. Architecture-specific adapters preserve the same law

Rotary attention uses recorded frozen K/V and query rows drawn from teacher
hidden inputs. Sliding/chunked attention extends the additive mask without
changing those inputs. Linear-attention layers use frozen state at the answer
boundary. MoE controllers may call the enabled pass a "student phase," but
their hidden rows are still teacher-coordinate inputs; routing terminology is
not permission to create a student trajectory.

## 9. Safe change checklist

Before considering a trainer change complete:

1. Confirm every training block input originates in a teacher product.
2. Confirm teacher inputs, targets, K/V, recurrent state, and masks are detached.
3. Run `scripts/audit_configs.py` after changing config flow or validation.
4. Run v4 locality certification after changing the block step, loss, masks,
   adapter targeting, or architecture integration.
5. Compare staged and single-process updates after changing ownership,
   launch coordination, rotation, or store transport.
6. Inspect `student_trajectory_eval` telemetry after changing validation, but
   never use it in backward, optimizer selection, or learning-rate control.
7. Preserve atomic stage manifests and unique-owner adapter merging.

Use the node-local interpreter documented in `AGENTS.md`; do not use an
editable install or a Lustre-hosted virtual environment.
