# Pareto-v2 optimizer-path probe

## Question

The first finite-tile screen changed gradient averaging, AdamW transition
frequency, momentum time scale, clipping frequency, and cumulative parameter
motion at once.  A nominal 32-cell run made 54,052 AdamW transitions in six
epochs, versus 1,572 for the `B=8, K=all` control, while retaining the same
learning rate.  Because each tile loss is a mean and AdamW normalizes its
moments, these were many more noisy learning-rate-sized transitions, not
automatically smaller corrections.

## Transfer probe

Before changing the optimizer implementation, run the closest pipeline-v2
translation of the ancient equal-answer optimizer semantics.  The concurrent
Qwen3-8B strict `lens_kl` arm is not evidence for its `1e-4` rate: CER improved
to 0.787 at epoch 4 but regressed to 0.831 by epoch 10, while exact recall
remained zero.

- Qwen3.5-4B, dataset v5, pipeline v2;
- strict block-local `lens_kl`, `conn_window: 1`, frozen vocabulary head;
- `B=8`, full aligned span, one AdamW transition per eight answers;
- `answer_mean`, so every answer contributes equally as in the ancient item
  loop (the ancient code summed eight answer means; AdamW is nearly invariant
  to that common factor, while clipping remains a documented difference);
- learning rate `1e-5`, held equal to the existing broad controls so the
  experimental change is answer weighting rather than step size;
- both deleted-RAG and randomized-token-RAG censorship, six complete epochs.

These are explicitly `ablation` arms because unbounded K is a semantics
transfer diagnostic, not finite-tile frontier evidence.  Success means recall
increases beyond epoch zero without unacceptable standard-benchmark damage.
If successful, the next experiment holds that recipe fixed while introducing
finite K with optimizer time measured in selected tokens.

## Expected artifacts

Each arm must publish its ordinary atomic `report.md`, `report.pdf`, figures,
manifest, locality certificate, recall trajectory including epoch zero,
standard-damage trajectory, and per-layer parameter deltas.
