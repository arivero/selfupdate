# Pipeline-v4 scaling

Scaling preserves the same local computation at every model size:

```text
detached teacher h[L-1] -> trainable student block L -> teacher h[L]
```

The local student output is differentiable through block L.  It is never
forwarded into block L+1 during training, so depth creates neither an
activation graph nor a training dependency.

## Teacher products

Inference engines may generate answer token ids, but teacher hidden states and
attention context come from this repository's model stack.  A run selects one
teacher source:

- `cache`: full teacher inputs/targets from a compatible cache;
- `online`: one adapters-off teacher forward per cohort;
- `store`: a one-time relayed teacher pass fills each PPP stage's local store.

Residency is independent of source: keep the active corpus slice on GPU,
stream it from pinned CPU/RAM-backed storage, or recompute when explicitly
configured.  Store-fill and per-epoch training times are reported separately.

## PPP: independent block owners

`v4_stage_splits` partitions the ordered blocks and `v4_stage_devices` maps
owners to physical GPUs.  Each OS process loads or materializes its owned
range, trains it from teacher tensors, and publishes a stage checkpoint.
There is no student activation boundary and no training wavefront.  Cross-node
mail exists for store-fill coordination and the separate validation relay.

Because ownership changes placement only, equal-seed single-process and PPP
runs must agree per block.  `scripts/compare_v4_shard_numerics.py` checks this
before a scaling result is treated as evidence.

## Beyond resident weights

Stage-scoped loading leaves foreign blocks on meta.  If an owned frozen shard
still exceeds a card, `v4_weight_residency: rotate` pages one block from its
host master while the previous block computes.  LoRA parameters and, for Adam,
the matching moments travel with their owned block.  Rotation is transport,
not an objective change; report `rotation_stall` and certify its numerics.

## MoE

`dense_or_black_box` matches the post-combine block output and supports the
block-distillation claim without claiming router agreement.  Teacher-forced
or router-aligned claims require online teacher recording and must report
per-layer routing overlap.  DeepSeek compressed-context layers retain
teacher-recorded key-side decisions.

Measured envelope results and provenance live in
[training_pipeline_v4.md](training_pipeline_v4.md); historical v1–v3 scaling
plans remain in Git history and dated reports only.
