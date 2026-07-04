# Layerwise Distillation As Forward-Only Trajectory Matching

A theory note positioning this branch's training regime, with the
2026-07-03/04 campaign evidence. The user's framing: "not a lot of
research uses layerwise loss — it is a sort of forward!"

## The regime

Teacher and student share architecture AND initial weights. The teacher
runs the context-bearing input once; its per-layer hidden states at the
aligned span become per-layer targets for the student, which runs the
context-free input. Every block trains against *its own depth's* target
with a graph rooted at a detached input: no global backward exists
anywhere in the system (the bounded tail window is the one named
concession).

Call this **forward-only trajectory matching**: the teacher's forward
pass IS the supervision trajectory, sampled at every depth.

## Why this is not the usual local-learning story

| approach | local target comes from | needs global backward? |
|---|---|---|
| Forward-Forward (Hinton 2022) | a scalar "goodness" objective per layer, positive/negative data | no, but no per-layer *vector* target either |
| Greedy layerwise (Bengio 2007; Belilovsky 2019) | the global label, via per-layer auxiliary heads | per-stage backward through the aux head |
| Target propagation | learned approximate inverses propagating targets down | no, but targets are *learned*, adding their own error |
| Whole-network KD | the teacher's output distribution | yes — full-depth backward |
| **this branch** | the same-architecture teacher's own h_L at matching depth | no — teacher/student share coordinates at every layer, so targets are exact and free |

The shared-init, shared-architecture condition is what none of the
literature exploits: it makes every layer's target well-defined in the
student's own coordinate system, with zero learned machinery. Locality
comes free; the open question is only what such training can and cannot
produce behaviorally.

## What the campaign measured

At Qwen3-0.6B, v2 data, matched 13.3k-item budgets, full-corpus eval:

1. **Storage vs readout dissociate.** Strict block-local hidden matching
   stores recall (logit-lens readable; full CER ~0.85 with 0% exact
   lines) but does not recite. A bounded connected top window trained
   with gold CE (tail-CE, k=4) converts storage into behavior
   (CER 0.024-0.104 depending on metric).
2. **Storage is distributed and redundant; the readout is a fragile
   circuit.** Weight-delta mass peaks at L22-24 of 28, yet reverting any
   single such layer barely hurts (l2mse: ablate L23/L24 -> CER 0.00).
   Reverting any ONE tail block (L25-28) destroys recitation
   (CER 0.65-0.88).
3. **The metric decides WHAT is written, not just how fast.** Delta-vector
   cosine across losses: huber≈nmse (0.95-0.99), l2mse near (0.87-0.91),
   vocab_mse far from all (0.48-0.70) — and best. Measuring hidden error
   through the frozen unembedding's Gram matrix (M = WᵀW; MSE in logit
   space) steers storage into vocabulary-visible directions.
4. **tail-CE re-carves only the top.** At fixed loss, tail vs strict runs
   share their sub-tail deltas almost exactly (cos 0.99+) and diverge
   only inside the window: behavioral credit does not reorganize storage.
5. **Forgetting decomposes into drift + intrusion ("catastrophic
   remembering").** Tail arms concentrate general-CE damage on the
   memorized content's nearest neighbor (Spanish romantic poetry: up to
   +4.0 nats) while English prose barely moves; strict arms damage
   uniformly instead. The intrusion trigger lives in the readout window.
   Trained checkpoints hijack a Bécquer continuation with Machado
   imagery; the base model does not.
6. **The machinery is architecture-generic.** The same code drives
   Llama-3.1-8B, Phi-4-mini, Mistral-7B (all smoke stages pass,
   template-agnostic masking); gpt-oss-20b's MoE blocks run the local
   steps too (OOM only at the connected tail on a shared card).

## Campaign appendix (2026-07-04, waves complete)

The open questions below were answered: routing needs a student-stream
finish (`mixed` anneal works, pure teacher-stream never recites); the
readout window does not grow with depth (k=2 viable at 1.7B); LoRA does
not inherently protect against forgetting; elicitation-diverse (maieutic)
data cures the readout's template lock at additive budget while IMPROVING
recitation; KL-to-base through the window halves intrusion while naive CE
anchoring backfires; and the window has a measurable capacity — trigger
diversity, anchor discipline, and full chain depth cannot all fit in k=4.

## Open questions the remaining waves addressed

- Input routing: does a layer learn its increment best from the student's
  drifting stream (summed), the teacher's stationary stream
  (teacher_censored), or an annealed mix (scheduled-sampling `mixed`)?
- Readout-window scaling: fixed k or fixed fraction of depth (1.7B
  k∈{2,4,8})?
- Whether LoRA / mixed routing mitigate intrusion without losing recall.
- Whether elicitation-diverse data (catechism v3; maieutic dialogue v4)
  strengthens the readout rather than the storage.
