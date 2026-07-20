# Phase 4 handoff: 40-epoch training campaign — deeper follow-up

Written 2026-07-20 for whichever agent picks up the next round of this
campaign. Read this before touching anything; it carries context that
is not visible from the code alone.

## What this campaign is

Branch `lwteacher`. Phases 1-3 (this same session, see
`runs/spec_verify/RESULTS.md`) proved the pipeline-v4 trainer reproduces
vLLM's greedy decoding exactly, at every parallelism degree, for 5 models:
gemma-4-26B-A4B-it (MoE), Qwen3.6-27B (dense), gemma-4-31B-it (dense),
Qwen3.6-35B-A3B (MoE), Qwen3.5-122B-A10B (MoE). Phase 4 (this handoff)
pivoted from verification to actual training: real 40-epoch block-local
distillation, with full scientific reports, for the same 5 models.

The owner's ask, verbatim intent: "40 epochs of training of each model,
with full generation of reports, and multiple experiments on learning
rate" (and, once results came back, "choose adequate lr and adam
inertias (or sgd)"). A cheap 12-epoch, 5-arm LR x optimizer screen on
26B was run first to pick a recipe before committing GPU-days to all 5
models at full length. That screen and its winner are below.

## ALERT — a real report-generation bug existed and is now fixed, verify it stays fixed

`scripts/report_v2.py` was reading a legacy config key (`online_optimizer`)
that a same-day cleanup commit (`ef07ba9`, 22,985 deletions, author
`sol@openai.com`) removed from the codebase. Every report generated
against a post-cleanup config was silently falling back to a hardcoded
`"adamw"` label in the report header, REGARDLESS of the actual optimizer
used. This would have directly corrupted the SGD-vs-Adam comparison this
campaign exists to make — every report would have claimed "adamw" even
for the SGD arms. Fixed in commit `fc7eeed` (read `v4_optimizer` instead).
**Before trusting any report from here on, spot-check that its header's
"Optimizer" line matches the config's actual `v4_optimizer`/`v4_adam_betas`
— do not assume the fix is permanent just because it's committed; if
anyone touches `report_v2.py` again, re-verify this specific field.**

This was the second report-tooling bug found this session. The first
(commit `dcba82e`) was more fundamental: `layer_loss_plots.py` and
`report_v2.py` were both written for an older (pre-pipeline-v4) metrics
schema (`kind=="train"`/`per_layer`) and had never been updated for the
real v4 schema (`kind=="v4_epoch"`/`layer_losses`) or for stage-scoped
multi-GPU runs at all (no top-level `metrics.jsonl` exists for a PPP-N
run; each stage writes its own `stageK/metrics.jsonl`). A new shared
loader, `src/selfupdate/eval/run_metrics.py`, fixes this by merging all
stages' rows and placing every per-layer value at its correct GLOBAL
layer position using each stage's `owned_blocks` contract (read from its
`kind=="pipeline_v4_contract"` row). Validated end-to-end against a real,
complete run (`runs/h100_g26b_v4_ppp4_e40`) by hand-tracing specific
per-layer values back to raw per-stage JSON. Trust this fix, but it too
was ONLY exercised on 26B's PPP4 shape — if a new model/parallelism
combination (especially 122B's 8-way cross-node PPP8) produces a report,
verify it by hand at least once (open the PDF, don't just check exit
codes — the Read tool reads PDFs directly).

## Two other infrastructure bugs found and fixed this session (trust these, don't re-diagnose)

- `scripts/launch_v4_stages.sh` (commit `9e7409a`): a bash
  `"${arr[@]:-}"` empty-array gotcha, introduced by an unrelated commit
  the day before, broke EVERY single-node multi-stage v4 launch by
  silently exporting a length-1 host map that failed `validate.py`'s
  host/device length check.
- `src/selfupdate/train/v4_store.py` (commit `acd9fdb`): the same
  `ef07ba9` cleanup renamed `online_v4._bk_layer_type` to
  `_v4_layer_type` but missed 6 call sites in this file, breaking every
  `v4_teacher_source: store` run at epoch-1 capture. The rename itself
  was verified byte-identical (docstring-only diff) and is provably safe
  for every model, not just the one it was caught on.

**Scope caveat on the `ef07ba9` cleanup generally:** only 26B's PPP4
store path and (as of this handoff) a 27B tree-regression smoke check
have been run through the post-cleanup tree end-to-end. The cleanup
touched `validate.py` alone by 899 lines. If you're the agent picking
this up for 31B/35B/122B and neither has completed a full run yet by the
time you read this, do not assume the tree is clean for them without at
least a short smoke check (see `configs/experiments/train40/
smoke_27b_treecheck_e3.yaml` for the exact pattern: a byte-identical copy
of a proven pre-cleanup config under a new run_name, diffed epoch-by-epoch
against the old reference).

## The proven recipe (do not re-derive, read `runs/h100_g26b_v4_ppp4_e40/stage0/config.yaml` for full ground truth)

`hidden_loss: huber`, `conn_window: 1`, `schedule: summed`,
`lora: {enabled: true, r: 16, alpha: 32, dropout: 0.0}`,
`batching: bucketed`, `micro_batch: 16`, `v4_capture_micro_batch: 2`,
`v4_teacher_source: store`, `v4_stage_scoped: true`,
`v4_weight_residency: auto`, `v4_loop_order: layer_major`,
`v4_relay_every_cohorts: 3`, `v4_teacher_residency: auto`,
`v4_min_train_gpu_util: 50.0`, `v4_nccl_timeout_s: 1800`,
`eval.every_epochs: 5`, `eval.recall_corpora: [machado, quijote_ch1,
quijote_ch4]`, `eval.standard_damage_every_epochs: 5`, `epochs: 40`.

**Primary arm, all 5 models:** `v4_optimizer: immediate_sgd`,
`lr: 3.0e-6` (winner of the 5-arm screen below).
**Secondary confirmatory arm, 26B only:** `v4_optimizer: adam`,
`v4_adam_betas: [0.9, 0.999]`, `lr: 3.0e-7`.

Configs already built and committed (`configs/experiments/train40/`):
`gemma4_26b_v4_ppp4_e40_sgd.yaml`, `gemma4_26b_v4_ppp4_e40_adam.yaml`,
`qwen36_27b_v4_ppp4_e40.yaml`, `gemma4_31b_v4_ppp4_e40.yaml`,
`qwen36_35b_v4_ppp4_e40.yaml`, `smoke_27b_treecheck_e3.yaml`. Stage
splits/devices per model (proven, don't re-derive): 26B `[8,16,23]`,
35B `[10,20,30]`, 31B `[15,30,45]`, 27B `[16,32,48]`, all `v4_stage_devices:
[0,1,2,3]` single-node PPP4. 122B is `[6,12,18,24,30,36,42]` /
`[0,1,2,3,0,1,2,3]`, cross-node PPP8 (BOTH agpuh01 and agpuh02 at once —
cannot share a node with any other arm); launcher script
`scripts/launch_q122b_ppp8x_campaign40.sh` (mirrors the already
NCCL-hang-fixed, cross-node-validated `launch_q122b_ppp8x.sh` pattern —
read `issues.md`'s "RESOLVED 2026-07-20" section on the relay readiness
gate before touching cross-node launches).

## The LR x optimizer screen (12-epoch arms, gemma-4-26B-A4B, PPP4) — results and an important correction

Configs in `configs/experiments/train_sweep/`, run dirs
`runs/train_sweep_26b_{sgd_lr1e6,sgd_lr3e6,adam_lr1e6,adam_lr3e7,
adamlowmom_lr1e6}/`, results committed at `f2178ac`. Summary (12-epoch
CE/KL deltas, recall, standard-damage, all small-n — 8 items/task recall,
16 items/task damage, read trends not single points):

- `sgd_lr1e6` (= the exact e40 baseline recipe): essentially flat, CE
  -0.13%, no real movement in 12 epochs.
- `sgd_lr3e6` (3x LR, still SGD): modest real movement (CE -0.74%, KL
  -1.59%), recall/damage looked noise-level flat in the SHORT screen.
- `adam_lr1e6`: most loss movement (CE -2.49%, KL -6.75%) but ~10% recall
  cost, damage dipped mid-run then recovered by epoch 12.
- `adam_lr3e7` (Adam, 10x lower LR): moderate movement similar to
  sgd_lr3e6, same ~10% recall cost as other Adam arms, damage improved.
- `adamlowmom_lr1e6` (Adam beta1=0.5 vs 0.9): virtually identical loss
  trajectory to standard Adam — the momentum knob made no visible
  difference at this budget — and the worst, non-recovering damage
  number of all 5 arms (though within small-n noise per-task).

**The consistent ~10% recall cost across all 3 Adam arms regardless of
LR/momentum was read as a real, replicated signal (not noise), which is
why SGD was chosen as the primary recipe.** `sgd_lr3e6` was picked over
`sgd_lr1e6` for real, low-risk movement.

**IMPORTANT CORRECTION, found only once 26B-sgd ran to full 40 epochs
(`runs/campaign40_g26b_sgd`, complete, report at
`runs/campaign40_g26b_sgd/report.pdf`):** the 12-epoch screen's read of
`sgd_lr3e6` as "no recall cost" did NOT hold at full length. Over 40
epochs, recall actually declined gently but not negligibly: machado
0.1225->0.1042, quijote_ch1 0.19->0.177, quijote_ch4 0.133->0.1235 (all
flat-to-down, noisy in between). CE/KL declined slowly and steadily
(0.0227->0.0223 / 0.0075->0.0072). Standard-damage bounced within noise
on the fast 16-item battery; the 100-item battery never ran this pass
(flagged as missing on the report's own coverage page, not silently
dropped). Per-layer parameter drift was tiny everywhere (max ~1.4e-3
relative L2 at layer 24) — consistent with lr 3e-6 barely moving a 26B
model even over 40 real epochs.

**Honest read: at lr 3e-6, the model moves a little but the movement may
not be worth the (small, real) recall cost — this is the gentle end of
the LR range doing what a gentle LR does, not a free lunch.** This is
exactly the kind of thing a "more detailed" follow-up should dig into:
does a longer run (80-100 epochs?) show the loss trend continuing to
improve while recall stabilizes or keeps declining? Does this same
pattern hold across all 5 models, or is 26B's MoE architecture
specifically prone to it (compare against 27B/31B's dense-architecture
runs once they complete)? Is the recall decline actually noise given the
n=8/task sample size, or does it replicate with a repeated seed?

## Status as of this handoff (verify before trusting — this will be stale by the time you read it)

Update 2026-07-20 (live campaign): the owner requested a full-size lens-loss
training follow-up.  Use the existing depth-uniform, block-local
`lens_kl` arm (`campaign40_loss_lens_kl_g26b`); do not turn it into a
deep-only/tail-weighted objective.  Its launch is intentionally ordered after
the committed Gemma-31B context-leak probe: the RAG span locator passed the
full 2,071-record/tokenizer audit, but v4's uncensored teacher h[L-1] query and
frozen teacher K/V may carry privileged information through nonprivileged
positions after layer 1.  A lens result collected under that mismatch must be
labelled as such, and no loss-family winner is promoted until the probe is
interpreted.  The already-running 26B `delta_cosine` arm continues to its
matched 24,852-item budget; no 0.6B arm is permitted.

Completed: `campaign40_g26b_sgd` (full report generated and hand-verified).
Running or queued at write time: `campaign40_q27b` (27B primary, just
launched), `campaign40_g26b_adam` (26B confirmatory, just launched);
31B and 35B queued next; 122B cross-node last, alone. **Do not trust this
status paragraph — check `runs/campaign40_*` directories, each run's
`metrics.jsonl` epoch count, and `nvidia-smi` on both nodes directly
before assuming anything is done, running, or queued.**

## Operational reminders that cost real time this session — don't relearn them

- `v4_stage_devices` is the load-bearing GPU-placement knob for any
  `--v4-stage` launch (`scripts/launch_v4_stages.sh` always uses it);
  `model.device` is silently overwritten and setting only that is a
  no-op (`scripts/train.py:52-57`).
- ssh sessions to agpuh02 land ONE DIRECTORY ABOVE the repo
  (`/fs/agustina/arivero/supercomplex`, not `.../selfup_teacher`) — `cd`
  into the repo in the SAME ssh command as any launch, not a separate
  verification command first, or the launch silently fails to find
  `scripts/`.
- Verify physical GPU placement via `nvidia-smi
  --query-compute-apps=gpu_uuid,pid,used_memory,process_name
  --format=csv,noheader` on the actual host after every launch — never
  trust the launch script's own text output alone.
- Check-in/monitoring discipline: use a WIDE polling interval (15-30+
  min) for any long-running launch-and-wait loop, and only surface a
  message at a genuine milestone (completion, crash, or a real decision
  point) — a prior sub-agent this session burned a large amount of
  tokens re-invoking itself every 2-3 minutes just to say "nothing
  changed."
- Own verifying report PDFs by hand (the Read tool reads PDFs directly)
  before declaring a report complete — exit-code-only checks missed the
  `online_optimizer` mislabeling bug above.
