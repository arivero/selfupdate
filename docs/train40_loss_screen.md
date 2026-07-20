# Campaign40 layerwise-loss screen

This screen asks whether the weak movement and gentle recall decline of the
40-epoch Gemma-4-26B-A4B Huber run are properties of the optimizer recipe or
of the local metric. It adds five 12-epoch PPP4 arms (the fifth was explicitly
requested after the original four-arm screen was written):

| Run | Local objective | Hypothesis |
|---|---|---|
| `campaign40_loss_cosine_g26b` | cosine distance between student and teacher absolute hidden states | Direction-only matching may avoid spending capacity on residual-stream norm while preserving teacher geometry. |
| `campaign40_loss_delta_cosine_g26b` | cosine distance between the owned block's student and teacher residual increments | Matching what block L adds, rather than the full residual state dominated by its shared input, may expose a stronger learning signal. |
| `campaign40_loss_vocab_mse_g26b` | MSE in the frozen full-vocabulary metric | The frozen head's geometry may prioritize behaviorally relevant hidden directions without training vocabulary parameters. |
| `campaign40_loss_lens_kl_g26b` | KL between full-vocabulary frozen-logit-lens distributions | Teacher probability geometry may be a sharper local behavioral metric, with the known risk of concentrated gradients and intrusion. |
| `campaign40_loss_vocab_cycle_g26b` | normalized MSE after the frozen round trip `h W_out.T W_in` | The embedder/unembedder product may expose vocabulary directions missed by the single-head Gram metric; its squared singular spectrum may also be badly concentrated. |

All five configs currently use the historical uncensored teacher-local context
and must be labelled **legacy-context/confounded** in reports.  The context
audit proved that this input diverges sharply from the censored deployment
trajectory after layer 1.  These arms compare loss geometry under a fixed
legacy context; none is evidence that changing the loss repairs censorship.
The separately named repaired-context arm must establish that point.

The delta objective is strictly block-local. For block L it compares
`student_h[L] - stop_gradient(teacher_h[L-1])` with
`stop_gradient(teacher_h[L] - teacher_h[L-1])`. The student output is the only
differentiable term; no student block output feeds a later training block.
Its launch remains gated until the `delta_cosine` implementation passes
`scripts/audit_configs.py`, fresh single-process-versus-PPP numerics with
`scripts/compare_v4_shard_numerics.py`, and `scripts/v4_battery.py`.

## Fixed controls and interpretation

Every overlay must be loaded on
`configs/experiments/h100_smoke/base_gemma4_26b_v4_full.yaml`. The five arms
pin the proven campaign40 recipe: Gemma-4-26B-A4B, PPP4 cuts `[8,16,23]` on
devices `[0,1,2,3]`, fill-once teacher store, stage-scoped auto residency,
layer-major traversal, relay cadence 3, bucketed micro-batch 16, capture
micro-batch 2, LoRA r16/alpha32/dropout0,
immediate SGD at learning rate `3e-6`, and seed 17. They run 12 complete
epochs (2,071 items per epoch, hence 24,852 items) and evaluate at epoch 0,
3, 6, 9, and 12 with the same three recall corpora, 16-item standard-damage
battery, decoding geometry, and whole-training-set cross-entropy/KL
evaluation as the proven arm. No base default or campaign config is changed.
The archived proven config records `conn_window: 1` and `schedule: summed`,
but those legacy knobs no longer exist in the active schema: v4's width-one
teacher-input/owned-block/teacher-target law is structural, not configurable.

The equal learning rate is deliberate but the raw losses have different
units and curvature. This is therefore an **objective-plus-scale screen**, not
a scale-normalized comparison of objective geometry. Interpret endpoint loss
or recall changes together with per-layer gradient norms/gradient-share and
LoRA parameter deltas from epoch zero. A large outcome from one arm is not
evidence for its geometry if its gradients or adapter movement are simply much
larger. Reports must include per-layer loss trajectories, gradient-share
attribution, per-layer LoRA delta profiles, recall including epoch zero,
standard-benchmark damage, the recall-versus-damage frontier, evaluated token
and item counts, and the coverage/provenance page required by `AGENTS.md`.
The historical same-recipe Huber arm
`runs/train_sweep_26b_sgd_lr3e6` is the reference; note its evaluation cadence
was four epochs, so only epoch 0 and epoch 12 are endpoint-matched.

## Full-vocabulary admission and OOM fallback

`vocab_mse`, `vocab_cycle_mse`, and `lens_kl` are admitted first in their
full-vocabulary forms.
Do not pre-emptively replace either, reduce its scientific budget, or call a
sampled approximation the same arm. If a full-vocabulary arm fails, retain
the failed run and log, record the exact command, stage/layer/cohort, CUDA OOM
text, peak allocated/reserved memory, and physical GPU occupancy. Retry only
after ruling out unrelated co-tenancy or launch defects.

Only after such a recorded, reproducible full-vocabulary failure may the arm
be substituted with `hidden_loss: vocab_cosine_sampled` plus
`vocab_cosine_samples: 256` and the already fixed
`vocab_cosine_seed: 17`. Give the substitute a new unambiguous run name ending
in `_sampled256`; it is a new sampled score-cosine objective, not an exact or
drop-in approximation to vocabulary MSE or lens KL. Keep every other control
unchanged and report the failed full-vocabulary arm as missing rather than
silently omitting it.

## Promotion to 40 epochs

Promote at most one new objective to a fresh 40-epoch confirmation. It must:

1. complete all 12 epochs with finite losses, all 30 layers covered, and no
   locality/frozen-vocabulary tripwire;
2. improve the whole-training-set cross-entropy and KL evaluation endpoints
   over epoch zero more than the matched 12-epoch Huber reference, without a
   worse decline in any recall corpus or worse mean standard-benchmark damage;
3. show depth-uniform gradient-share attribution without a last-layer spike,
   and LoRA deltas commensurate with the Huber scale rather than an unexplained
   order-of-magnitude jump.

If no arm satisfies all three conditions, promote none: first run a
loss-weight/LR calibration screen rather than extending a scale-confounded
winner. The 40-epoch confirmation keeps the selected arm's exact recipe and
seed, changes only `epochs: 40` and both evaluation cadences to 5, and receives
a new `campaign40_loss_*_e40` run name. A sampled-vocabulary fallback may be
promoted only under its sampled name and with the full-vocabulary failure
carried into the final coverage record.

## Additional objective ideas

Owner addition, 2026-07-20: a separately named frozen vocabulary round-trip
arm was requested and implemented.  This is not the existing
`vocab_mse`: that objective measures with `W_out.T @ W_out` and coincides with
an embedding/unembedding product only for tied weights.  The requested map is
`C = W_in.T @ W_out` (vocabulary logits decoded back through the frozen input
embedding), with a normalized MSE between `C h_student` and `C h_teacher` at
every layer.  Keep both matrices frozen, apply the final norm with the same
depth-uniform convention as the other vocabulary metrics, and give the loss
and uses the explicit `vocab_cycle_*` name.  Matrix orientation and tied
versus untied direct equivalence are CPU-certified; report its gradient scale rather
than treating the raw coefficient as matched to `vocab_mse` or `lens_kl`.

The first screen is intentionally factorial enough to identify whether the
useful signal lives in absolute hidden geometry, the residual update, or the
frozen vocabulary measurement.  Two follow-ups are scientifically stronger
than simply adding more uncalibrated names now:

1. **Delta direction plus log-magnitude.** Pure `delta_cosine` discards the
   size of the teacher block update and is poorly conditioned when either
   update is near zero.  A local composite can add a depth-uniform smooth-L1
   penalty on
   `log(||delta_student|| + epsilon) - log(||delta_teacher|| + epsilon)`.
   Report direction and magnitude gradient shares separately; choose their
   coefficient from a short gradient-scale calibration, never from endpoint
   recall.
2. **Teacher-scale-normalized delta Huber.** Apply Huber to
   `(delta_student - delta_teacher) / stop_gradient(rms(delta_teacher)+eps)`.
   This preserves magnitude information while preventing blocks with naturally
   large updates from owning the global comparison.  The denominator must be
   per item/token (or a provenance-pinned frozen statistic), not a learned
   cross-block normalizer.

A token-centered or whitened residual objective is also plausible, but it
couples examples/tokens through batch statistics and makes results sensitive
to bucket composition.  It should remain behind the two strictly pointwise
forms above.  None of these deferred ideas is queued until the four-arm screen
establishes gradient scale, update-norm distributions, and whether pure
direction matching has a nonzero behavioral benefit.

## Repaired-context continuation (2026-07-20)

The full-corpus censorship audit showed that the original v4 context fed
uncensored teacher `h[L-1]` (and ordinary-position K/V which had already
absorbed the privileged RAG) to every block after layer 1. Those arms remain
objective-scale diagnostics but are not evidence that a censored student
learned to reconstruct RAG information. The opt-in legal-A repair uses
detached, adapters-off, fully flow-censored teacher `h_c[L-1]` for both query
and frozen attention context while retaining uncensored teacher `h_u[L]` as
the local target.

Three six-epoch, full-corpus repaired-context arms were launched on agpuh01
at 2026-07-20 15:30 UTC. Six epochs cover 12,426 items and therefore satisfy
the minimum 12,000-item comparison budget:

- `campaign40_g31b_context_repair_lens_kl_adam_e6` on physical GPU 1;
- `campaign40_g31b_context_repair_vocab_mse_adam_e6` on physical GPU 2;
- `campaign40_g31b_context_repair_vocab_cycle_adam_e6` on physical GPU 3.

All use Adam at `3e-7`, gradient clipping at 1.0, depth-uniform objectives,
and evaluation at epochs 0, 3, and 6. Adam and clipping are scale controls,
not an assertion that these losses work: matched repaired-vs-legacy probes
found sharply larger deep-layer gradients, particularly for `lens_kl`.
The legacy-context 26B lens SGD arm already showed an effective layer-2 LoRA
delta 2.49 times the epoch-zero base-weight norm by epoch 3. It is therefore
provisionally a scale failure; preserve its endpoint evidence and do not call
the large parameter motion learning.
