# Hidden losses in pipeline v4

For block `L`, let `x = stopgrad(context teacher h[L-1])`,
`s = student_block_L(x)`, and `t = teacher h[L]`.  The optimizer minimizes a
local distance `D(s, t)`.  Gradients flow through the student block that
produces `s`; they do not flow into `x`, `t`, another block, or the teacher.

This is the essential distinction:

- the local student block output `s` is differentiable and is what trains the
  student's weights;
- an end-to-end student trajectory is never a training input.  It is computed
  separately for ordinary full-forward token validation.

## Locality proof

With parameters partitioned by block, `s` depends only on `theta_L` because
`x` is detached and teacher-fixed. Therefore

```text
d D(s,t) / d theta_j = 0,  j != L.
```

The embedding, final norm, and vocabulary head are frozen.  Training-end
`certify_locality_v4` measures rather than assumes this: every foreign-block
and vocabulary-stack gradient must be exactly zero, while every owned block
must show local signal before checkpoint publication.

## Absolute local metrics

`HiddenLoss` supports geometric distances such as normalized MSE, MSE,
cosine, Huber, Charbonnier, clipped NMSE, contrastive/relational controls,
and frozen-artifact Mahalanobis/Jacobian variants.  Vocabulary-coordinate
metrics pass `s` and `t` through the frozen final norm/head (or a frozen
sample/sketch) as a measurement device.  `lens_kl`/`lens_js` likewise compare
local distributions while leaving the vocabulary stack unchanged.

`vocab_cycle_mse` measures the complete frozen vocabulary round trip.  With
input embedding rows `W_in`, output-head rows `W_out`, and final norm `N`,

```text
C    = W_in^T W_out
z(h) = N(h) W_out^T W_in = N(h) C^T
loss = mean ||z(s)-z(t)||^2 / stopgrad(mean ||z(t)||^2)
```

The output-head bias, if present, cancels from the difference.  When weights
are tied, `C = W^T W`, but the induced cycle metric is `C^T C = C^2`; it is
therefore still distinct from `vocab_mse`, whose metric is only `W^T W`, and
from `embedding_mse`.  The squared singular spectrum can strongly amplify a
small set of vocabulary directions, so this arm requires initial gradient-
scale attribution before its raw learning rate is treated as comparable.

## Teacher-anchored local increment cosine

`delta_cosine` is the sole admitted increment objective.  For every
non-final block it uses the same detached teacher input already consumed by
the v4 block step:

```text
x_L       = stopgrad(teacher h[L-1])
delta_s,L = student_block_L(x_L) - x_L
delta_t,L = stopgrad(teacher h[L] - x_L)
loss_L    = 1 - cosine(delta_s,L, delta_t,L)
```

This is not the historical successive-student-state loss: neither increment
contains a student output from block `L-1`. Since both `x_L` and
`delta_t,L` are detached constants, `delta_s,L` depends only on `theta_L`, so

```text
d loss_L / d theta_j = 0,  j != L.
```

At the final block, cached `teacher h[n]` and `BlockStack.loss_view` are
post-final-norm, while `x_n` is the pre-norm block input (and for mHC models
may also precede the frozen stream-collapse head). Those tensors do not
describe an increment in one coordinate space. The final block therefore
uses the explicit fallback `1 - cosine(student_postnorm, teacher_postnorm)`;
it does not subtract `x_n`.

Every other historical `delta_*` / `multi_delta_*` objective and all
connected-window machinery remain rejected. A loss may not create credit
across blocks or apply depth-increasing weighting.

## Output metrics are evaluation only

Cross-entropy and KL at the token output are not hidden training losses.
Teacher-forced output evaluation measures final-block fidelity on teacher
inputs; the deployment-matched validation relay performs the ordinary
censored student full forward and predicts tokens. Both carry
`used_for_backward=false` and `optimizer_weight=0.0`.

See [training_pipeline_v4.md](training_pipeline_v4.md) for teacher-context
construction and [runtime.md](runtime.md) for the executable enforcement.
