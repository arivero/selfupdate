# The hidden-layer comparison loss, and why the layerwise regime is genuinely local

## Setting

Teacher T and student S are the **same architecture with the same initial
weights**. T runs once, frozen, on the context-bearing input; its per-layer
hidden states at the aligned span are cached to disk (`teacher/cache.py`).
S runs on the context-free input; block L of S is trained to reproduce the
teacher's block-L output at the aligned token positions.

Notation: for one example, let `H_L^T ∈ R^{A×d}` be the cached teacher hidden
states of layer L at the A aligned positions, and `H_L^S` the student's block-L
output at the same positions (both compared in fp32; teacher storage is fp16).
Layer indices are 1-based; `H_n` (last layer) is post-final-RMSNorm on both
sides — the student applies the frozen final norm to its last block's output
before the comparison (`BlockStack.loss_view`), matching the HF
`output_hidden_states` convention the cache was built with.

## The two loss kinds (`train.hidden_loss`)

**`nmse` (default) — normalized mean squared error:**

    ℓ_L = ‖H_L^S − H_L^T‖_F²  /  ‖H_L^T‖_F²

Implemented as `mse_loss(H_S, H_T) / mean((H_T)²)` — both numerator and
denominator are means over the same A×d entries, so the ratio equals the
relative squared Frobenius error. The normalization makes losses comparable
across layers (residual-stream magnitude grows severalfold with depth in
Qwen3) and across examples, so "loss 0.01" always means "1% relative error".

**`l2mse` — direction-only matching (PKD-style stabilizer):**

    ℓ_L = (1/A) Σ_i ‖ ĥ_i^S − ĥ_i^T ‖² / d,   ĥ = h / ‖h‖₂  (per position)

Equivalent to `2(1 − cos(h_i^S, h_i^T))/d` per position: magnitude errors are
ignored, only the residual-stream direction is matched. Sun et al. 2019 (PKD,
arXiv:1908.09355) found normalizing before MSE stabilizes intermediate-layer
distillation; we keep it as an ablation.

Positions convention (everywhere in the codebase): hidden losses use aligned
positions `[s0, s0+A)`; logit-type losses use `[s0, s0+A−1)` predicting tokens
`[s0+1, s0+A)`. Because the aligned span is `shared_mid + answer`, the
position that predicts the *first* answer token is inside the span.

## Why this is not backpropagation from the logits

In both layerwise schedules the per-block step is (`layerwise.local_block_step`):

    h_out = block_L(h_in.detach())          # graph starts at block L's input
    ℓ_L   = hidden_match(h_out[s0:s0+A], H_L^T)
    ℓ_L.backward()                          # graph ends at block L's params
    next input = h_out.detach()             # graph never crosses the boundary

The input is detached **before** the block and the output is detached
**after** it, so the autograd graph recorded for ℓ_L contains exactly one
block. Consequences, each enforced by a test:

1. `∂ℓ_L/∂θ_M = 0` for every M ≠ L — no gradient from any loss reaches
   another block, the embedding, the final norm, or the lm_head
   (`test_grads_confined_to_blocks`).
2. Each block's gradient is bit-identical to an *isolated* single-block
   replay: run block L alone on the same detached input, backprop its loss,
   compare all parameter grads (`test_block_grads_match_independent_replay`).
   If gradients leaked across blocks, this equality would fail.
3. The lm_head is never invoked during layerwise training, so no logit is
   ever computed, let alone backpropagated
   (`test_layerwise_training_never_computes_logits`).

Contrast with the methods this is often confused with:

| method | training signal | backward extent |
|---|---|---|
| classical KD (`train/kd.py`) | KL on logits | whole network |
| TinyBERT / PKD intermediate losses | Σ_L per-layer MSE, **one global backward** | layer-k loss reaches all layers < k |
| BAdam / LISA (block-coordinate) | global LM loss, one active block | backward still traverses the whole net to reach the block |
| **ours, `summed`** | per-block hidden match, backward per block | one block per backward, all blocks per step |
| **ours, `sequential`** | same, one block per *stage* | one block; earlier blocks replaced by an activation cache and never executed again |

The honest caveat: *within* a block we still use standard autograd (chain rule
over the block's own ops). "Forward training" here means no **cross-block**
backward pass exists — the deepest gradient path is one transformer block,
which is what bounds activation memory and enables the one-layer-at-a-time
streaming plan for 120B models.

What makes the per-layer targets well-posed at all is the same-initial-weights
property: at step 0 the student's layer-L output differs from the target
*only* through attention into the (missing) privileged block. No learned
projections (TinyBERT's W_h) are needed, and ℓ_L starts small and measures
exactly the context's contribution to layer L.

## Where the two schedules differ

- `summed`: every block gets its local loss on every example; per-block AdamW
  states persist for all blocks. Shallow targets are chased while deeper
  inputs still drift (targets are stationary; inputs are not).
- `sequential`: block L trains to plateau on a **stationary** input
  distribution (its input comes from already-frozen blocks via
  `StudentActCache`), then freezes. Compounding student-side drift is
  explicit: block L+1 must fit teacher targets from student-produced inputs.
  Per-layer final losses (`metrics.jsonl`, `kind="stage"`) give the
  error-vs-depth curve that the local-training literature (Chen 2026,
  arXiv:2606.06539) predicts will grow.

Known limitation, deliberate: local targets came from the *initial* frozen
teacher. The composition of individually-well-fitted blocks is not guaranteed
to minimize any end-to-end objective — measuring how much recitation this
loses relative to KD **is the experiment**, not a bug to engineer away.
