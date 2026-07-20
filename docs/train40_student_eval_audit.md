# Pipeline-v4.5 student evaluation audit

Status: verified from the 2026-07-20 code path. Native live-PPP evaluation is
implemented as `v4_battery_mode: distributed`; unsupported architectures keep
the explicit trainer-owned reconstructed fallback.

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
In v4.5 distributed mode it runs through the live stage-owned blocks and
retains KV state only for each rank's owned layers. Standard damage likewise
scores complete sequences through those live partitions. The older path—and
the current explicit fallback—publishes every stage's current adapters,
offloads live blocks, reconstructs a complete base model, grafts the adapters,
and calls ordinary Hugging Face generation/forward. Its launch/epoch envelopes
make it a genuine reconstructed current student, but it is not the native PPP
execution architecture.

## Utilization consequence

In the reconstructed fallback all four trainers pause and evict their owned
blocks; one GPU then holds the complete model while the other three wait. A
measured 26B battery spent roughly 311--318 seconds in telemetry and 332--369
seconds at the complete boundary. Loading, grafting, synchronization, offload,
and restore accounted for about 19--57 seconds. Native mode removes those
reload phases and executes all live partitions, although autoregressive decode
remains the dominant serialized cost until further microbatch pipelining.

## Implemented replacement and remaining gate

The native battery is a synchronous collective at one adapter epoch using a
dedicated NCCL evaluation communicator. Stage 0 tokenizes and embeds, every
rank executes only its owned blocks, and the final rank applies frozen final
norm/head. Standard multiple-choice scoring relays full sequences; recall
generation retains per-owned-layer KV caches and loops selected tokens from
the final rank back to stage 0. Stage 0 alone writes telemetry, and all ranks
restore exact training state or fail together.

Keep the subprocess fallback until logits, option scores, generated token
ids, EOS behavior, telemetry rows, adapter freshness, and frozen-vocabulary
fingerprints match on both Qwen and supported Gemma configurations. Gemma
shared-KV layers require explicit side-channel transport across a stage cut;
the currently used Gemma 26B/31B configurations report no shared-KV layers,
but the evaluator must verify or reject this rather than assume it.
