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
- The required final checkpoint delta tool also assumed a flat run-level
  `config.yaml`; PPP runs keep the snapshot under `stage0/`.  Its lookup is now
  stage-aware.  The completed 26B SGD report was rebuilt in the required order
  (layer-loss plots, final effective LoRA delta profile, individual report/PDF)
  and visually checked through its temporal delta, damage, and frontier assets.
  The effective-delta calculation independently confirms layers 24, 30, and
  18 as the three largest movers (RMS relative deltas 0.0011, 0.0007, 0.0007).
  The report continues to show the honest missing locality/signal certificate
  and 100-item endpoint battery rather than promoting either gap to evidence.

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

`delta_cosine` now has a separate mechanics admission in
`docs/delta_cosine_admission.md`: fresh Qwen3-0.6B one-epoch PPP1 and PPP2
runs cover every interior layer plus the final post-norm fallback.  The shard
comparator's historical loss-only blind spots were closed behind
`--strict-current`; current admission additionally requires identical clean
commit/loss provenance, exact nonoverlapping ownership, all 28 loss and 28
gradient cells at zero tolerance, and passed correctly scoped inline locality
certificates.  The procedural comparator check and historical compatibility
check pass.  These mechanics runs remain pending and are not scientific arms.

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

## 122B cross-node preflight — smoke-only GO as of 2026-07-20 12:10 CEST

Do not launch the 40-epoch `campaign40_q122b` arm yet.  Read-only preflight
found two technical
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

At 12:00 CEST the full-arm overlay was corrected to pin the intended
`micro_batch: 16` and `v4_capture_micro_batch: 2`, and a separate three-epoch
PPP8 admission overlay (`smoke_122b_xnode_e3.yaml`) was added.  The launcher
comment now states the actual no-disk communication law.  These preparation
At 12:10 CEST commit `e086ba6` implemented remote adapter publication on a
separate NCCL communicator: stage 0 validates and materializes the enveloped
shards in its own `/dev/shm` for the unchanged battery child.  Long-running
battery status uses the already established launch TCPStore so ranks do not
hold an NCCL collective while the child owns stage 0's GPUs; nonzero child
status reaches every rank.  The CPU protocol check, both locality/report
self-checks, Python compilation, config validation, and diff check pass.

This lifts the gate only for the three-epoch cross-node smoke.  It must
complete all eight stages, four battery points (epoch zero through three), and
the new inline locality certificate before the 40-epoch arm is admitted.

## 2026-07-20 12:00 CEST — live utilization

- `campaign40_g26b_adam` is healthy on agpuh02 at approximately epoch 25 on
  all four stages; every GPU is active.  The cosine loss-screen arm is staged
  behind the required architecture/locality admission smoke on that node.
- `campaign40_q27b` is healthy on agpuh01 at approximately epoch 14 (the last
  stage temporarily trails during battery work); every GPU is active.  The
  three-epoch 31B tree/locality smoke is staged as its immediate successor.
- Apparent cross-stage epoch skew is not treated as failure: subprocess
  batteries intentionally offload and synchronize stages.  Worker PID/GPU
  liveness and fresh boundary/battery rows were checked on the owning hosts.

At 12:28 CEST, after Adam's four workers exited cleanly, the 35B tree/locality
smoke launched on agpuh02 as `v4-20260720122842-4152136` (PIDs 4152187,
4152226, 4152282, 4152322).  All four contracts identify clean runtime commit
`39f13d2`; the source moved afterward only in reporting/docs.  Store capture
completed and the expected epoch-zero battery is active.  The loss screen
remains gated on exact reference-metric agreement and four passing inline
locality certificates from this run.

The 35B admission gate completed at 12:58 CEST.  Epochs 1, 2, and 3 matched
the pinned pre-cleanup CE/KL pairs bit-for-bit.  All four stages passed the
live-store certificate with exact ownership coverage L1--L40, positive finite
local gradients at every layer, zero cross-block and frozen-vocabulary
gradients, and byte-exact unchanged adapters/optimizer/rotation state during
certification; four checkpoints and done rows followed.  This admits the
current runtime for the 35B full arm and for the loss screen.

At 13:00 CEST the first loss-screen arm, absolute hidden-state cosine, launched
on agpuh02 as `v4-20260720130000-4153872` (PIDs 4153925, 4154007, 4154056,
4154096).  All four contracts report clean source `cb71d9b`, `loss_kind:
cosine`, and expected ownership `[1,8]`, `[9,16]`, `[17,23]`, `[24,30]`.
The primary 35B 40-epoch arm is the next successor on that host if cosine
completes and certifies cleanly.

### Provisional scientific read (not an endpoint claim)

At epoch 30, Gemma-26B Adam (`3e-7`, betas 0.9/0.999) has reduced whole-set
teacher-forced cross-entropy from 0.022696 at epoch 1 to 0.022295 (-1.77%) and
teacher-to-student KL from 0.007521 to 0.007109 (-5.48%).  This is more movement
than same-model SGD had accumulated by epoch 30 (-1.25% / -2.88%), despite the
10x smaller numeric learning rate.  The behavioral tradeoff seen in the short
screen is also repeating: Adam's three-corpus mean recall is 0.1485 at epoch
zero and 0.1371 at epoch 25 (-7.7%), with Machado and Quijote chapter 4 down
while chapter 1 is noisy/up.  The 8-item-per-task battery is too small for a
final damage claim, but the direction is replicated evidence against treating
the extra loss movement as a free gain.

Qwen-27B SGD is qualitatively different through epoch 15: whole-set CE/KL are
essentially flat (CE 0.013652 to 0.013657, KL 0.0004620 to 0.0004630), while
the three-corpus mean recall oscillates around its epoch-zero level rather
than showing the Gemma-wide decline.  If this remains true at epoch 40, the
first cross-architecture conclusion should be that one global LR is not a
matched-update comparison: calibrate future loss/model arms by early LoRA
delta or gradient scale, not the nominal optimizer LR alone.

The Qwen-27B endpoint confirms that contrast.  From epoch 1 to 40, mean local
Huber loss falls 29.9% and 61 of 64 layers improve, driven especially by a few
large-loss layers (for example L28 -76.5%, L48 -52.5%, L32 -50.3%).  Yet the
whole-set teacher-forced output distances become slightly worse: CE +0.18%
and KL +0.37%.  Three-corpus mean recall is essentially unchanged from epoch
zero (0.15939 to 0.15893, -0.29%), but that mean hides Machado down
0.09481->0.08081, Quijote chapter 1 up 0.23283->0.24674, and chapter 4 nearly
flat.  The 16-item standard macro also ends unchanged.  This is direct
evidence that reducing absolute block-local hidden error—especially a few
dominant layers—does not guarantee improvement of the composed student
trajectory.  Residual-update geometry and scale calibration are therefore
more valuable next tests than simply extending Qwen's Huber duration.

Gemma-26B Adam completed at 12:28 CEST.  Relative to its epoch-1 whole-set
measurement, its epoch-40 CE improved 2.23% and KL 6.65%, versus 1.75% and
4.27% for SGD.  That modest incremental output-distance gain required a much
larger update: the final effective-LoRA module RMS is 0.00184 for Adam versus
0.000283 for SGD (6.5x), and the module mean is 10.7x larger.  Final
three-corpus mean recall is 0.1334 for Adam versus the shared epoch-zero
0.1485 (-10.2%); SGD ends at 0.1349 (-9.2%).  The 16-item standard macro ends
-0.0417 for Adam and unchanged for SGD, but is explicitly too noisy for the
endpoint claim.  On current evidence Adam is update-inefficient and carries no
recall advantage; do not promote it as the default optimizer.

The Adam stage shards were merged, and its layer-loss, effective-delta,
individual Markdown/PDF, and grouped campaign artifacts were generated and
visually inspected.  The grouped report now includes separate, explicitly
uncertified coverage-only final-layer loss and parameter-delta comparisons;
the strict-local frontier remains empty, so descriptive coverage cannot be
mistaken for publication evidence.

### Standard-damage delta repair

The staged subprocess battery's raw per-task standard accuracies were correct,
but each fresh child passed `baseline=None` at nonzero epochs.  Consequently
the redundant derived fields `standard_epoch0_delta` and
`standard_worst_delta` were falsely zero in all historical subprocess rows.
This does not lose evidence: epoch-zero and every checkpoint's three task
accuracies are durable in the same rows.  `report_v2.py` now always recomputes
macro, epoch-zero delta, worst task, and worst delta from those raw scores, so
old reports are repaired on regeneration.  Future battery children reconstruct
the telemetry baseline from the durable epoch-zero row and fail loudly if it
is missing or malformed.  The current 16-item probe remains a noisy campaign
gate; the separate 100-item endpoint battery is still required.
