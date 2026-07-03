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

Hidden matching stays post-MoE-combine at the block output. Router agreement
can be logged as a metric, and per-expert delta norms can be reported by the
existing weight-delta tooling. The first MoE extension should keep the same
block-output contract before adding router-specific auxiliaries.
