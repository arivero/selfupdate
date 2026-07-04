# Hidden-Layer Loss And Locality

## Setting

Teacher and student are the same architecture with the same initial weights.
The teacher runs on the context-bearing input. The student runs on the
context-free input. For each layer `L`, the student block output at the
aligned span is trained toward the teacher's cached `h{L}` at the matching
teacher span.

Layer indices are 1-based. `h{n}` is post-final-RMSNorm, matching the
Hugging Face `output_hidden_states` convention and the cache round-trip test.

## Losses

`nmse`:

```text
mse(H_student, H_teacher) / mean(H_teacher^2)
```

This keeps layers with different residual-stream scales comparable.

`l2mse`:

```text
mse(normalize(H_student), normalize(H_teacher))
```

This matches direction only and ignores magnitude.

`cosine`: `1 - mean(cos(h_s, h_t))` — direction only, linear near optimum.

`huber`: smooth-L1 on `H / rms(H_teacher)` — scale-comparable across layers
like nmse, robust to heavy-tailed residual rows.

Vocabulary-metric kinds measure the difference as the **frozen vocabulary**
sees it (both apply the frozen final norm first, except at `h{n}` which is
already post-norm):

`vocab_mse`: `||W·Δh||² / ||W·h_t||²` with `W` the frozen unembedding —
MSE in logit space, computed through the precomputed Gram matrix
`M = WᵀW` ([H,H], one 4 MB buffer). Equivalently: `Δhᵀ M Δh`.

`lens_kl`: `KL(lens(h_t) ‖ lens(h_s))` through the frozen norm + head.
`vocab_mse` is the flat local approximation of `lens_kl` (the exact local
metric of KL is the Fisher pullback `Wᵀ(diag(p) - ppᵀ)W`). Wave H's failed
lens-KL was a *behavioral auxiliary without tail-CE*; these kinds replace
the *storage* metric and compose with tail-CE — a different question.

Both vocab kinds depend on the Frozen-Vocabulary Principle below: the
metric is only meaningful because the vocabulary never moves.

Local readout auxiliaries use gold answer CE through the frozen final norm and
LM head. They are used only where explicitly configured:

- `last_block_ce_weight`: one-block readout test
- `lens_ce_weight`: per-block local readout head
- `tail_ce_blocks` + `tail_ce_weight`: bounded connected top window

## The Frozen-Vocabulary Principle

The embedding and LM head are the system's vocabulary, not part of the
network being trained. They are never trained, under any schedule or
auxiliary:

- They define the fixed basis every lens decodes through. A lens whose
  vocabulary drifts during training measures nothing.
- Teacher targets (`h{n}` post-norm, cached or online) are expressed in the
  initial norm/head geometry; training the head would decalibrate every
  stored target.
- Qwen3-0.6B/1.7B/4B tie `lm_head` to `embed_tokens`
  (`tie_word_embeddings=true`); training the head there silently retrains
  the input embedding as well. 8B and up are untied.

`BlockStack.freeze_non_blocks()` enforces this structurally, and the
locality tests assert no gradient reaches embedding, final norm, or head.
Lenses may include *learned per-layer translators* (tuned-lens style); the
translator is scaffolding and is trained or discarded freely — the
vocabulary piece it decodes through stays frozen.

## Why The Backward Is Local

For a strict block step:

```text
h_out = block_L(h_in.detach())
loss = hidden_match(h_out[s0:s0+A], target_L)
loss.backward()
next_input = h_out.detach()
```

The graph starts at block `L`'s detached input and ends at block `L`'s
parameters. No gradient reaches any other block, the embedding, final norm, or
LM head. Tests enforce:

- block gradients are confined to the intended block/window
- isolated single-block replay matches the in-trainer gradient
- strict hidden-matching steps do not invoke logits

Tail-CE is the named locality concession: blocks below the tail are detached,
while the final `k` blocks train in one connected graph so the top readout CE
can assign credit within that bounded window.

## Schedules

- `summed`: each block consumes the student's stream and receives a local loss
  on every item.
- `sequential`: one block trains at a time; earlier blocks are frozen and their
  outputs are cached.
- `teacher_censored`: each block consumes the teacher stream with privileged
  rows removed, making layers independent and stationary.

## Current Mechanistic Picture (2026-07-04 campaign-final)

Storage and readout dissociate, causally:

- **storage**: distributed and REDUNDANT across the upper-middle stack
  (delta mass peaks at ~80% fractional depth at 0.6B and 1.7B alike;
  single-layer ablations below the tail are harmless). Best written by
  `vocab_mse` — measuring hidden error through the frozen vocabulary's
  Gram matrix — whose format is PORTABLE: foreign readouts decode it
  (chimera transplants), and a readout trained post-hoc on a frozen
  strict body beats joint training (`tail_only`, 0.008 vs 0.024).
- **readout**: a fragile, co-adapted circuit in the top k blocks where
  every pathology lives. It is template-locked (recitation-trained
  readouts collapse 0.024 -> 0.92 under dialogue framing; cured by
  maieutic elicitation-diverse data), intrusion-prone ("catastrophic
  remembering": damage concentrates on neighbor-genre Spanish poetry,
  halved-to-thirded by anchor-KL to the base model, WORSENED by naive
  anchor-CE), and capacity-limited (k=4 serves any two of trigger
  diversity / anchor discipline / full 715-verse chain depth, not all
  three).
- Context enters the computation near L7 (`teacher_censored` increments);
  content is written at ~80% depth; behavior is decoded at the top.
- Reasoning-tuned families (Phi-4-mini, gpt-oss) resist the recipe:
  their output routes through think/analysis channels the readout never
  trains. Non-reasoning families (Qwen3, Mistral, Llama) all train.

Final recipe and closing table: EXPERIMENTS.md. The 0.6B one-phase
champion recites the full 715-verse romance self-chained, first error at
verse 708.
