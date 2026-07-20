# Pipeline-v4 student evaluation audit

Status: verified from the 2026-07-20 code path; native live-PPP evaluation is
an implementation item, not yet the default.

## What each metric executes

`teacher_output_eval` is computed during the local final-block training step.
It applies the frozen vocabulary stack to answer rows of the differentiable
local block output and compares them with teacher rows/realized answer token
ids. Its logged trajectory is `teacher_forced_blockwise`. It is a real
evaluation-only local diagnostic, but it is neither an end-to-end student
walk nor autoregressive generation.

`student_trajectory_eval` is a no-grad, end-to-end censored student walk. In
PPP it relays hidden states through every live stage and applies the frozen
head at the tail. It therefore uses every current student block exactly in
pipeline order, but it evaluates fixed teacher-forced training sequences. It
does not select a token and feed that token back as the next input.

The epoch recall battery is genuine greedy autoregressive student inference.
For stage-scoped PPP, however, the implementation publishes every stage's
current adapters, offloads the live blocks, loads a separate complete base
model in `scripts/v4_battery.py`, grafts all published adapters, and then
calls ordinary Hugging Face `generate()`. Standard damage similarly uses
complete-model option-likelihood forwards. Launch-identity and epoch
envelopes protect the adapter exchange, so this is a reconstructed copy of
the current student rather than fabricated generation. It is nonetheless
the wrong operational architecture for PPP.

## Utilization consequence

At a battery boundary all four trainers pause and evict their owned blocks;
one GPU then holds the reconstructed complete model while the other three
wait. A measured 26B battery spends roughly 311--318 seconds in telemetry;
the full boundary costs 332--369 seconds. Loading, grafting, synchronization,
offload, and restore account for only about 19--57 seconds. Removing reload
is worthwhile, but autoregressive decoding is the dominant cost; the larger
utilization gain comes from executing and eventually microbatch-pipelining
the live partitions.

## Required replacement

A native battery must be a synchronous collective at one adapter epoch. It
needs a dedicated NCCL evaluation communicator on same-node as well as
cross-node stages. Stage 0 tokenizes and embeds, every rank executes only its
owned blocks, and the final rank applies the frozen final norm/head. Standard
multiple-choice scoring can relay full sequences without a cache. Recall
generation additionally requires per-owned-layer KV caches and final-rank to
stage-0 token loopback. Stage 0 alone writes telemetry, and all ranks must
restore their exact training state or fail together.

Keep the subprocess fallback until logits, option scores, generated token
ids, EOS behavior, telemetry rows, adapter freshness, and frozen-vocabulary
fingerprints match on both Qwen and supported Gemma configurations. Gemma
shared-KV layers require explicit side-channel transport across a stage cut;
the currently used Gemma 26B/31B configurations report no shared-KV layers,
but the evaluator must verify or reject this rather than assume it.
