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

## Current Mechanistic Picture

Strict hidden matching learns storage but leaves the final readout weak.
Tail-CE shows that a small co-trained top window can turn the stored signal
into behavior. This supports the working decomposition:

- storage: distributed below the tail, learnable by forward hidden matching
- readout: compact and co-adapted in the final blocks

The current champion is the v2-data `tail_ce` run with `k=4`, which gives the
best full-corpus and anchored whole-poem recitation among layerwise methods in
the repo artifacts.
