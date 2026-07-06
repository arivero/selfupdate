# Scaling Plan

The small-model code paths are chosen to map onto 4xH100 and 120B-class
training without changing the masking or loss abstractions.

## Teacher Products

| teacher product | backend |
|---|---|
| `<think>` traces | vLLM/sglang generation client |
| per-layer hidden states at aligned spans | layer-streamed Hugging Face forward |

Inference engines are useful for trace harvesting. They do not expose the
internal residual streams needed for hidden-state targets, so large hidden
caches require a streamed model forward.

## Layer-Streamed Teacher Forward

Keep activations for one layer, load one block's weights, advance all
examples, write the aligned-span slice, and repeat. This mirrors
`StudentActCache.advance` in the sequential trainer.

Approximate fp16 hidden-cache size per 1k examples x 512 aligned tokens:

| model class | all layers | one layer |
|---|---|---|
| Qwen3-0.6B | ~29 GB | ~1 GB |
| Qwen3-4B | ~94 GB | ~2.6 GB |
| 30B-A3B MoE | ~100 GB | ~2 GB |
| 120B-class | ~450 GB | ~7 GB |

Sequential training reads one layer at a time. Online-teacher LoRA avoids the
cache entirely when the teacher can be represented as adapters-off.

## Training Regimes At Scale

- `sequential`: one active block, exact one-block backward.
- `summed`: one-block backwards, but optimizer states for all blocks are live
  unless streamed.
- `teacher_censored`: independent blocks; natural multi-GPU parallelism.
- `tail_ce`: bounded top-window backward; memory grows with `k`, not depth.

## MoE Notes

Black-box MoE is valid method evidence for the block-output layerwise claim:
the router and experts sit inside the decoder block, and the hidden loss
matches the post-MoE-combine block output. Its limitation is narrower but
important: expert-mechanism claims require router agreement evidence.

For MoE-specific claims, report the mode explicitly:

- `dense_or_black_box`: ordinary post-combine hidden matching. Valid method
  evidence for layerwise block distillation; router agreement unproven.
- `teacher_forced`: replay the teacher's top-k expert choices during training,
  so hidden matching updates the same expert subnetwork the teacher used.
- `router_aligned`: train or regularize the student router toward the teacher
  router distribution/top-k set, and report top-k overlap by layer/token.

The innovation path is teacher-forced expert replay plus a router-alignment
training lane, because a student that routes to expert 7 while the teacher
routed to expert 19 can otherwise train the wrong expert to imitate the right
one. Router agreement and per-expert delta norms should sit next to recall,
forgetting, and destruction metrics for MoE runs.
