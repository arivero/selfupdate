# Fable Review By 55xh

Review date: 2026-07-05. Scope: current `selfupdate_lw` branch contents, including
`README.md`, `CLAUDE.md` / `AGENTS.md`, `EXPERIMENTS.md`, `docs/`, `src/`,
`configs/`, `tests/`, and available `runs/results.md`.

## Post-Cleanup Status

Implementation pass later on 2026-07-05 addressed the highest-priority
branch-law findings recorded below:

- `tail_step` was replaced by the generic connected-window primitive
  `window_step`.
- Active train config knobs were renamed from `tail_ce_*` / `tail_hidden_*`
  to `readout_*` / `window_hidden_weight`; old keys are rejected by
  `scripts/audit_configs.py`.
- `configs/base.yaml` no longer sets a readout source. Readout experiments
  must pin `readout_source` explicitly.
- Active task-label readout/lens/last-block configs and tail-only configs were
  removed from `configs/experiments`; surviving readout configs are sliding
  `conn_window` / `conn_stride: 1` configs with explicit `teacher_kl`.
- `teacher_censored` was made pure by construction and validation rejects
  readout/task-label knobs for that schedule.
- Hot-loop train logging now keeps loss tensors on device and flushes Python
  floats at accumulation/epoch boundaries.
- `scripts/build_corpus_index.py`, gate-aware `scripts/report.py`, and
  `scripts/speed_check.py` were added for corpus/report/speed follow-up.

The detailed review below is preserved as the finding record that motivated
the cleanup. Any statement saying an old active surface "still" exists should
be read as the pre-cleanup finding unless it remains true in current code.

## Executive Judgment

This branch contains a serious and productive research codebase: the core
layerwise machinery is unusually well documented, locality is tested directly,
run metadata is rich, and the experiment campaign has real breadth. The main
technical problem is not lack of code quality; it is branch contamination. The
tree still exposes and reports tail-only / tail-CE / task-label readout
machinery after the branch rules explicitly banned it as non-layerwise,
classical-distillation territory. That makes the branch harder to operate,
harder to review, and easier to misclassify.

Highest-priority fixes:

1. Remove or quarantine `tail_ce` / `tail_*` implementation, configs, tests,
   and report-facing summaries from this layerwise branch. Tail-only and
   tail-specific readout work belongs in `../selfupdate_kd` or an archival
   history, not in active reports for this branch.
2. Enforce the target law at the config/report layer, not only inside
   `tail_step`: method arms must be teacher-sourced; `task_label` arms must be
   visibly labeled as baselines or absent from this branch.
3. Pay down the hot-loop sync cost before H100-scale execution. The current
   path is adequate for L40S campaign work, but it will waste H100 throughput.

## 1. Code Quality

Strong points:

- The core training abstraction is clear. `BlockStack` exposes embedding,
  rotary embeddings, decoder blocks, final norm, and LM head cleanly, and the
  trainer operates on detached block/window inputs as the method requires.
- Frozen-vocabulary discipline is implemented structurally and with a runtime
  tripwire: `freeze_non_blocks()` plus `_vocab_signature()` before save.
- Loss code is coherent. `nmse`, `l2mse`, `cosine`, `huber`, `vocab_mse`,
  `vocab_fisher`, and `lens_kl` are centralized in `src/selfupdate/train/losses.py`.
  The vocabulary-metric implementation follows the frozen-head contract.
- The test suite covers the right failure modes: alignment, cache round trips,
  locality, connected-window gradient extent, online teacher equivalence,
  target-source law, and schedule/knob refusal.
- Script entry points pin `src/` on `sys.path`, matching the shared-venv
  constraint in `CLAUDE.md` / `AGENTS.md`.

Main code-quality issues:

- `src/selfupdate/train/layerwise.py` still uses `tail_step` as the central
  connected-window primitive and exposes `tail_ce_blocks`, `tail_ce_weight`,
  `tail_hidden_weight`, and `tail_ce_kind`. This is now the wrong public
  abstraction. A generic `window_step` can exist; a tail-specific API should
  not be active in the layerwise branch.
- The hard stop on tail-only training is only partially enforced. The validator
  rejects `schedule: tail_only`, but it does not reject `schedule: summed` with
  `tail_ce_blocks > 0` and no `conn_window`. That is exactly the banned
  tail-only window shape in the owner directive.
- `anchor_step` is coupled to `tail_ce_blocks`, which keeps the anchor-KL
  mechanism tied to banned tail semantics. If anchor-KL remains in this branch,
  it should attach to the sanctioned sliding top window, not to a `tail_*` knob.
- `answer_ce_weight` exists in config but appears unused by training. Stale
  knobs are dangerous in this repo because defaults have already produced
  campaign confounds.
- `last_block_ce_weight` and `lens_ce_weight` remain label-targeting paths.
  They may be useful as baselines, but the code and configs should make that
  impossible to confuse with the method.

## 2. Accuracy Against The Described Targets

What is accurate:

- Strict block-local hidden matching matches the stated mechanism: each block
  consumes a detached input, matches the teacher hidden state on aligned spans,
  and does not backpropagate into lower blocks or the vocabulary.
- `conn_window` with `conn_stride: 1` implements the important owner design:
  endpoint loss, detached window input, bounded backward depth, and uniform
  k-deep credit over covered body layers.
- `teacher_kl` readout is implemented as a teacher-sourced target through the
  frozen head, and tests verify that corrupting `label_ids` does not change its
  gradients.
- `teacher_censored` was restored to its pure meaning: stationary teacher-stream
  inputs, independent layers, no connected window, and no readout CE.

What is inaccurate or misleading:

- The branch target says tail-only readout windows are banned. The branch still
  keeps tail-only/tail-CE code paths, historical configs, tests named
  `test_tail_ce.py`, and report summaries centered on tail arms. That is not
  just historical clutter; it interferes with the layerwise branch by making
  classical-distillation evidence look like active method evidence.
- `EXPERIMENTS.md` and report-facing language still crown or foreground arms
  whose readout source is `task_label`. `CLAUDE.md` / `AGENTS.md` say this is
  baseline-only and belongs outside method arms. The later "last 3%" resolution
  explains why task-label CE helps verbatim recall, but it does not make it a
  teacher-sourced layerwise method in the lab setting.
- The "explicit `tail_ce_kind`" fix is weaker than advertised. `TrainConfig`
  defaults to `UNSET`, but `configs/base.yaml` sets `teacher_kl`, and many
  experiment YAMLs with readout windows do not pin `tail_ce_kind` themselves.
  They still inherit a default, which is the exact class of problem the branch
  says it eliminated.
- README and docs still contain older phrasing about "gold-token CE",
  "tail-CE", and two-phase tail readout as current science. Those should be
  rewritten or moved so the source of truth does not reintroduce the forbidden
  concept through documentation.

Recommended target-correct cleanup:

- Replace public `tail_*` knobs with sanctioned `conn_window` / `conn_stride`
  semantics. If the top window has an extra readout term, name it as a
  top-window readout attached to sliding windows, and reject it unless
  `conn_window > 0` and `conn_stride == 1`.
- Move tail-only, tailpure, final_k8, and `[expunged]` report material to an
  archived campaign note or `../selfupdate_kd`. Keep only a short pointer in
  this branch explaining why those results are excluded.
- Add a config audit test that scans `configs/experiments/*.yaml` and fails if
  any active method arm has `tail_ce_blocks > 0` without a sanctioned sliding
  window, or uses `task_label` outside a baseline/ablation namespace.

## 3. Speed And Hardware Fit

For the current campaign scale, the code is operationally adequate:

- `runs/results.md` shows 0.6B full-FT/windowed runs around 10-11 GB reserved,
  1.7B full-FT around 28-29 GB, 4B LoRA around 10-12 GB, 8B LoRA around
  18-19 GB, 14B LoRA around 31-32 GB, and a 4B full-FT run around 42 GB.
  This fits L40S lanes and is comfortable on H100 80 GB for the intended
  bridge experiments.
- The scheduler has practical VRAM guards, multi-GPU exclusivity support, and
  cluster-specific conventions. It is good enough for campaign operation.
- `offload_adam` is a meaningful vehicle: the branch records 4B full-FT fitting
  on one L40S where traditional full Adam full-backprop would not.

But the trainer is not H100-efficient:

- `CLAUDE.md` / `issues.md` record the main bottleneck: `.item()` per block per
  item makes the hot loop sync-bound. The measured avoidable cost is 191 ms/item
  versus 131 ms/item on 0.6B, a 1.46x free speedup.
- Batch size is effectively 1, `DataLoader` uses `num_workers=0`, and disk-cache
  runs can do many small Lustre reads per item. That protects correctness, but
  it leaves H100 matmuls underfilled.
- Faithful sliding stride-1 windows cost roughly k times body compute. That is
  scientifically justified for the current method, but it makes the hot-loop
  cleanup more important.
- PP2/TP2 correctness is not fully certified. The apparent PP2 failure was
  confounded by default inheritance, and `pp2fix`/TP certification are still
  listed as pending. Do not trust 27B/32B multi-GPU science until a pinned
  parallel repro matches a single-device reference.

Speed priorities before larger H100 work:

1. Accumulate per-layer losses on GPU and flush once per grad-accum boundary.
2. Add an equivalence-tested padded/bucketed batching path.
3. Reduce per-block optimizer Python overhead with fused/foreach stepping where
   safe.
4. Pack or prefetch cache reads for disk-cache arms.
5. Finish PP2/TP2 correctness certification against a single-H100 reference.

## 4. Experiment Coverage

Coverage is a strength of this branch:

- Losses: geometric losses, `vocab_mse`, `vocab_fisher`, `lens_kl`, and
  multiple CE/KL readout variants.
- Schedules: `summed`, `sequential`, `teacher_censored`, `mixed`, sliding
  connected windows, disjoint-window attempts, and online-teacher LoRA.
- Scales/families: Qwen3 0.6B/1.7B/4B/8B/14B plus the updated 2026 ladder:
  Qwen3.6-27B, Gemma-4-26B-A4B, Gemma-4-31B, Mistral, gpt-oss smoke, and
  Quijote rungs. Llama-8B and Phi-4-mini are no longer active coverage
  requirements for this branch.
- Evaluation axes: full-corpus recitation, dialogue framing, long self-chained
  recitation, general CE, destruction/intrusion probes, anchors, layer swaps,
  delta profiles, signal attribution, teacher ceilings, and memory accounting.
- The experiment ledger is unusually candid about confounds and relabeling,
  especially the two `tail_ce_kind` default failures.

Coverage gaps and unresolved claims:

- The active crown seed claim is still marked open through pinned follow-up
  runs. Do not present it as fully replicated until those land.
- The disjoint-window conclusion was retracted as confounded; pinned disjoint
  evidence is still needed.
- Teacher-stream k-windows are explicitly not implemented, despite being a C3
  item and the natural depth-parallel extension.
- The teacher-sourced method remains weak for verbatim recall under measured
  `teacher_kl`; the branch has evidence for the pure-distribution bound, but
  the high-recall crown relies on task-label/transcript-equivalent framing.
- H100-specific evidence is not yet present. The L40S numbers are useful but
  do not prove throughput, memory fragmentation, or PP/TP correctness on H100.
- Eval-time per-layer residuals at checkpoints are missing; training losses
  and weight deltas do not fully answer storage-quality questions.

## Bottom Line

The core layerwise implementation is credible, and the campaign produced a
valuable empirical map. The branch should now become stricter, not broader:
remove active tail-specific code/reporting, separate baseline/classical
distillation artifacts into the sibling branch, enforce teacher-sourced method
targets at config-audit level, and then optimize the hot loop for H100-scale
runs. Without that cleanup, future work will keep paying the same tax: method
claims, baseline evidence, and legacy tail experiments will remain mixed in the
same operational surface.
