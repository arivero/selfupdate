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

## Where the schedules differ: whose stream feeds block L

Two inequivalent layerwise designs (both block-local in the backward sense):

- **(a) student-stream** (`summed`, `sequential`): block L consumes the
  *student's* h_{L-1}. Its target gap is the context effect **accumulated over
  all layers up to L** — measured at init: relative loss 0.002 (L1) → 0.42
  (L28), growing with depth. Inputs drift as shallow blocks train.
- **(b) censored teacher-stream** (`teacher_censored`): block L consumes the
  *teacher's* h_{L-1} with the privileged rows deleted and teacher position
  ids kept (its own attention is censored; everything upstream already carries
  the context influence). Each block owes only **its own layer's increment**
  of the context effect — measured at init: mean 0.021, 13× smaller than (a),
  near-zero at L28, peaking at L≈7. Inputs are stationary and layers are
  fully independent → embarrassingly parallel across GPUs. The increment
  profile itself is a localization readout: it measures directly which layers
  perform the context integration. Trade-off: blocks compose their own
  (student) stream at inference, so cross-layer interactions of the
  compensations are never trained.

## Where schedules (a)-internal differ

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

## Storage vs readout: the lens decomposition (2026-07-03) and where it sits

Logit-lens depth profiles on matched runs (mean gold-token logprob of the
answer, decoded through the frozen final norm + lm_head):

| layer | kd_full | lw_seq | lw_summed |
|---|---|---|---|
| 20 | -10.5 | -11.1 | -11.2 |
| 24 | -7.30 | **-7.23** | -8.16 |
| 28 | **-0.40** | -2.70 | -4.12 |

Under matched training signals, block-local hidden matching STORES the poem
as well as global KD through layer ~24; the entire behavioral gap opens in
the last four blocks — the readout. Memorization decomposes into storage
(depth-distributed, method-invariant, achievable block-locally) and readout
(a tail circuit that needs multi-block behavioral credit). Corroborations in
this repo: wave-C ablation (deep layers 25-28 necessary for recitation),
convergence analysis (shared per-layer delta-magnitude profiles across
methods), and the single-block-CE hybrid failing while full-backprop KD+CE
recites.

Positioning against the bibliography (README):

- PKD (Sun 2019, supervise all depths) vs MiniLM (Wang 2020, supervise the
  top): both half-right — depth-spread supervision wins storage, top
  supervision is about readout, and MiniLM only works because it backprops
  through the whole student. Our tail-CE is the synthesis: local below,
  behavioral credit confined to a bounded top window.
- Stoehr 2024 (memorization low-layer & localizable) vs Huang 2024
  (distributed & entangled): the decomposition says both, about different
  things — storage is distributed and method-invariant, the *recitation
  behavior* hinges on a small tail circuit. Cf. Hase 2023: where a memory is
  carried is not where intervening works.
- Belilovsky 2019 made greedy layerwise training work with per-block
  auxiliary classifiers; the frozen lens is that auxiliary head for free —
  grounding for the per-block lens-CE variant (untested bet).
- Deng 2023 found cross-layer hidden matching "hard to optimize" and escaped
  via curriculum; the tail window is the orthogonal escape suggested by our
  lens data.

Mechanism: `tail_ce_blocks` k + `tail_ce_weight` (config), `tail_step()` in
train/layerwise.py — blocks n-k+1..n connected in one graph, per-block hidden
losses kept, answer-CE at the top, rooted at a detached input so locality
below the window is intact (tests/test_tail_ce.py). Cost at scale: k blocks
of activations instead of all — the 120B story survives with k/n overhead.

### Free-run confirmation: cross-run tail grafts (scripts/tail_graft.py)

Chimeras, k=4, full-corpus free-run CER (no lens anywhere — this answers the
"lens-at-final-layer is circular for KD" objection):

| model | CER | exact |
|---|---|---|
| kd_ce intact | 0.596 | 0.379 |
| lw_seq intact | 0.928 | 0.000 |
| lw_seq body + kd_ce tail | 0.926 | 0.000 |
| kd_ce body + lw_seq tail | 0.932 | 0.054 |

Neither tail transplants: a KD readout cannot decode an lw-trained stream,
and removing kd_ce's own tail (replacing with lw's) destroys its recitation
— the readout is necessary AND co-adapted to its body (the delta-direction
orthogonality, cos ~0.02, is the geometry behind this). Meanwhile lw_tail_ce
(a tail co-trained on its own lw body) reached CER 0.792 / 5% exact at 20
epochs — the best layerwise result to date. Together: storage below the tail
is real and method-invariant in *quality*, but its representation is
method-specific; readouts must be co-trained, never borrowed.
