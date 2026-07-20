# Campaign40 live progress and decisions

This is the live operational/scientific record for the 40-epoch pipeline-v4
campaign.  `docs/train40_handoff.md` remains the initial handoff; this file
records work after that handoff.  Raw metrics and checkpoints remain the source
of truth.

## 2026-07-20 11:15 CEST — takeover audit and restart

- `campaign40_g26b_sgd` is complete at 40/40 epochs on all four stages, with
  four `done` rows and four stage checkpoints.  Its report correctly labels
  the optimizer `immediate_sgd`.  The report honestly flags two gaps: no
  100-item-per-task endpoint standard battery and no accepted signal/locality
  attribution artifact.
- The handoff's two nominally running jobs were not live.  `campaign40_q27b`
  had stopped on every stage after the epoch-20 training row and before the
  epoch-20 battery/boundary completed.  `campaign40_g26b_adam` had stopped
  during the epoch-zero battery before any training epoch.  Neither had a
  checkpoint, traceback, nonzero-exit record, or resumable optimizer artifact.
  Both hosts had all four H100s idle.
- The partial run directories, logs, and stale leases were preserved under
  `runs/interrupted_campaign40_20260720/{q27b,g26b_adam}/`; nothing was
  deleted.  Both arms were relaunched cleanly from epoch zero at 11:13 CEST:
  Qwen3.6-27B on agpuh01 and gemma-4-26B-A4B Adam on agpuh02.  Launch IDs are
  `v4-20260720111348-2954317` and `v4-20260720111348-4147504`, respectively.
- Repository-wide `scripts/audit_configs.py` reaches only already-existing
  duplicate `run_name` findings in `configs/experiments/spec_verify/`; no
  campaign40-specific validation error was observed.

## Reporting audit

- The post-cleanup `v4_optimizer` fix is present, but the per-run report did
  not expose numeric learning rate or Adam betas/epsilon.  The report and
  manifest now expose those fields so optimizer comparisons can be audited
  from the report itself.
- Stage-scoped store runs currently record locality certification as skipped
  (`stage_scoped_store_certification_pending_relay`).  This is certification
  debt, not a pass.  Reports and grouped frontiers must keep `strict_local`
  false until real evidence exists; no reporting change may relabel it.
- Campaign40 was not recognized by the group-report classifier, and grouped
  elapsed-time/pending discovery assumed flat runs.  Those are reporting bugs,
  not scientific exceptions; both are corrected with stage-aware discovery.

## Loss-family extension requested 2026-07-20

After the primary model arms, extend the campaign with controlled block-local
loss comparisons.  The first new candidate is residual-update cosine matching:

```text
teacher input:    x_L = stopgrad(h_teacher[L-1])
student update:   delta_s = block_L^student(x_L) - x_L
teacher update:   delta_t = h_teacher[L] - x_L
loss:             1 - cosine(delta_s, delta_t)
```

This remains structurally local: only the owned student block is
differentiable; both teacher tensors are detached.  Its scientific motivation
is to remove the large identity/residual component from the comparison, which
can dominate cosine similarity on the full hidden vector.  Historical cosine,
Huber, lens, and vocabulary-space objectives are being audited before choosing
the smallest informative screen.  Frozen-vocabulary and depth-uniformity laws
remain unchanged.

### Implemented screen contract

The archaeology recovered the historical implementations at `b013d83`
(cosine/Huber/vocabulary MSE/lens KL) and `eebafda` (historical delta losses).
The active v4 implementation now admits exactly one increment objective,
`delta_cosine`, with a different and stricter dataflow than the historical
student-state form: both increments share the detached teacher input.  All
other `delta_*`, multi-delta, connected-window, and student-trajectory forms
remain rejected.  Because the cached final target is post-final-norm while
the final block input is pre-norm, the final block uses an explicit absolute
post-norm cosine fallback; this exception is recorded in the v4 contract and
report loss name.

The controlled screen is specified in `docs/train40_loss_screen.md` with four
new 12-epoch, PPP4, Gemma-4-26B arms: absolute cosine, teacher-anchored
delta-cosine, frozen-vocabulary MSE, and frozen-head lens KL.  The completed
Huber arm is the baseline.  Every non-loss control is held fixed, including
SGD at `3e-6`; consequently this is explicitly an objective-plus-scale screen.
At most one objective can be promoted to a 40-epoch confirmation, and only if
its whole-set CE/KL, all recall corpora, damage, gradient distribution, and
LoRA-delta scale jointly pass the documented rule.

## Locality certification repair

The old end-of-run certifier could not run after a stage-scoped fill-once
store job because foreign blocks are meta-loaded and the ephemeral store had
already gone out of scope.  It therefore emitted
`stage_scoped_store_certification_pending_relay` and allowed a checkpoint as
declared debt.  The repaired path certifies inline after relay drain/barrier,
while every exact store entry and typed attention context is still alive.
It runs the same local forward/loss helper as training, checks every owned
layer for finite positive local signal and exact-zero foreign/frozen-stack
gradients, and proves adapter and Adam state bytes are unchanged.  Missing
store entries or a failed cert withhold publication.  Reports accept the new
evidence only when every configured stage is present, passed, and its checked
layers exactly cover its declared ownership; historical skipped rows remain
uncertified.  CPU contract checks pass; the 31B/35B tree-regression smokes are
the first GPU admission gate before this change is used by a full arm.

## 122B cross-node preflight — NO-GO as of 2026-07-20 11:50 CEST

Do not launch `campaign40_q122b` yet.  Read-only preflight found two technical
blockers hidden by the earlier handoff:

- Cross-node subprocess-battery acknowledgement/done files are under each
  node's private `/dev/shm`.  Stage 0 on agpuh01 cannot observe stage 4--7
  acknowledgements written on agpuh02.  The prior 122B eval-in attempt proves
  this failure: all stages filled their stores but the run produced zero
  epochs and stopped waiting for `battery_ack_stage4.st` at epoch zero.  The
  NCCL readiness gate fixes training-boundary transport, not battery control.
- The campaign config inherits `micro_batch: 32` and
  `v4_capture_micro_batch: 0`, not the claimed 16/2 recipe.  Prior PPP8 SGD
  attempts at this shape OOMed stage 7 during backward.  Pin 16/2 explicitly
  and smoke it; do not call the uncompleted eval-in run proof.

Both nodes otherwise have matching 122B model snapshots and byte-identical
2,071-item teacher indexes, healthy pinned runtimes/IB, adequate RAM and
`/dev/shm`, and no 122B lease or run directory.  Correct stale launcher/config
comments that still claim a shared-Lustre relay: the real file relay is
node-local `/dev/shm`, while cross-node tensor transport is NCCL/IB.
