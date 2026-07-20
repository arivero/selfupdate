# Pipeline-v4.6 student-evaluation audit

Verified directly from `online_v4.py`, `distributed_pp.py`, the evaluation
task backends, and the deleted reconstructed path as it existed in Git.

## Metric classification

- `teacher_output_eval` is a teacher-forced block-local surrogate. The
  trained final block consumes detached zero-run teacher `h[n-1]` and frozen
  teacher context. It is not a complete student trajectory and not rollout.
- `student_trajectory_eval` is a genuine complete student trajectory through
  all PP owners, with no teacher hidden injected after embedding. It runs over
  a fixed vLLM input+answer sequence, so CE/KL and acceptance are
  teacher-forced. The asynchronous row can combine stages serviced at
  different frontiers and says so; the synchronous boundary row is
  epoch-exact.
- Recall c and the uncensored control a′ are genuine autoregressive greedy
  inference. Prefill runs once; subsequent selected token IDs are embedded
  and decoded through owner-local caches.
- Standard multiple-choice scores are normalized teacher-forced continuation
  log likelihoods, not generated answers.

## Historical reconstructed battery findings

The old subprocess did evaluate a scientifically genuine reconstructed
student: every emitted token used the base model plus the stage adapter files
that the parent supplied. Paths and safetensor metadata carried epoch and
launch identity, so a named epoch could not silently read another named
epoch/launch. Before the v4.5 hardening, however, the reader did not prove the
producer stage against the requested owner or require the complete expected
adapter key set. A partial/mislabelled publication could therefore have
escaped the freshness checks. The live v4.6 entry gate instead validates each
owner's complete trainable topology and byte fingerprint in place.

No reconstructed result used teacher hidden as student hidden. The
teacher-state conflation was confined to the deliberately block-local
`teacher_output_eval`. CE/KL/acceptance remained teacher-forced in both paths.

Epoch-zero and post-epoch standard/recall called the same builders and shared
item order, prompts, censorship state, token budgets, padding side, position
IDs, EOS handling, and scoring. Epoch zero was the untrained network under
those same conditions; it was not the uncensored RAG teacher control.

Embedding, final norm, and LM head were frozen. v4.6 extends the checked
frozen surface to the mHC head and per-layer input modules and compares a
byte-exact named digest across ranks before and after evaluation.

## v4.6 correction to the executor review

The review in `train40_progress.md` correctly identified rank-asymmetric
decode/postprocess/logging failures and incomplete fingerprint evidence.
Those are now explicit guarded phases: every rank reduces failure before the
next payload collective. Complete expected trainable keys are compared with
live non-meta keys, vocabulary fingerprints are byte-exact, and unavailable
GPU-process inspection records `unverified` rather than true.

The review's recommendation to keep the subprocess default is superseded by
v4.6 architecture support and removal of that path. Rotary owners page blocks
in place. Gemma shared producer K/V crosses every stage cut and decode step as
a transient NCCL side channel, while only the producer retains prefix state.
Per-layer token inputs, hybrid/recurrent caches, sliding physical lengths, and
mHC boundary tails are native. The same shared-KV carrier is used in the
fixed-sequence relay and frozen local/store training context.

This correction is an implementation statement, not a fleet-parity claim.
Disposable multi-GPU Qwen/Gemma, rotary PPP1/PPP2, nonzero-LoRA, complete
telemetry, GPU ownership, and injected-failure certification remain required
before campaign adoption.

## Timing interpretation

Historical 26B reconstructed boundaries spent roughly 311--318 seconds in
evaluation and 332--369 seconds total; offload/load/graft/synchronization and
restore contributed roughly 19--57 seconds. v4.6 records those removed phases
as exactly zero and separately records standard, recall, fixed-sequence,
uncensored-control, rotary page/H2D, and total boundary time. Fresh comparable
before/after numbers require disposable old/new runs on the same copied
checkpoint; no live campaign was touched for this change.
