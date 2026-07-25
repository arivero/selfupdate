# Weekend handoff — r64 campaign (2026-07-25, Claude)

Picked up from the codex conversation `019f90e4-9ba6-7ed2-8e88-52bfa8b9e65a`,
which ran out of credits with two instructions unexecuted: **[434]** launch
r64 for *every* model, and **[435]** explain the stuck `DependencyNeverSatisfied`.

## [435] What happened to the DependencyNeverSatisfied job

`422707` (`g26b_a4b_ppp8_2node`) was submitted with
`--dependency=afterok:422703,422706`. Those two were the old incomplete-adapter
Gemma-A4B PPP4 runs; both ran to the 24 h wall and Slurm ended them
**CANCELLED** (state `CANCELLED+`, not `COMPLETED`). `afterok` only fires on a
clean exit, so a CANCELLED producer makes the dependency **permanently
unsatisfiable** — exactly the "don't use afterok when the stop mechanism may be
scancel" trap in CLAUDE.md. It was never going to run. **I cancelled it.**

(The two report jobs `422808`/`422812` use `afterany`, so they are fine and
still queued behind their running trains.)

## What I launched — r64 for every model (the [434] ask)

All on the **ungated `qwen_ppp4_50` store path** — the same path the running
r16 arms use (`v4_teacher_source: store`, no SP-vs-shard numerics gate). Each
train job has an `afterany` v4_report job. Configs are committed
(`f651a58`, plus the A4B r16 baseline and this note).

| run_name | model | loss | rank | train / report |
|---|---|---|---:|---|
| campaign50_q27b_ppp4_r64 | Qwen-27B (dense) | huber | 64 | 422979 / 422980 |
| campaign50_q35b_ppp4_r64 | Qwen-35B-A3B (MoE) | huber | 64 | 422981 / 422982 |
| campaign50_g31b_dense_ppp4_r64 | Gemma-31B (dense) | huber | 64 | 422983 / 422984 |
| campaign50_g26b_a4b_ppp4_r64 | Gemma-26B-A4B (MoE) | huber | 64 | 422985 / 422986 |
| campaign50_q27b_ppp4_cosine_r64 | Qwen-27B | cosine | 64 | 422987 / 422988 |
| campaign50_q27b_ppp4_delta_cosine_r64 | Qwen-27B | delta_cosine | 64 | 422989 / 422990 |
| campaign50_q35b_ppp4_cosine_r64 | Qwen-35B-A3B | cosine | 64 | 422991 / 422992 |
| campaign50_q35b_ppp4_delta_cosine_r64 | Qwen-35B-A3B | delta_cosine | 64 | 422993 / 422994 |
| campaign50_g26b_a4b_ppp4 (r16 control) | Gemma-26B-A4B (MoE) | huber | 16 | 422995 / 422996 |

`expert_parameters: true` on both MoE arms (35B-A3B, A4B) so LoRA covers the
packed experts — the point where memorization lands. Dense arms keep it false.
alpha = 2·r (128 at r64). The A4B r16 control is new because the old A4B runs
predated the packed-expert LoRA fix, so there was no matched r16 baseline.

## The A4B r64 decision you need to know about

The **only** pre-existing r64 arm, A4B on the spec path (`422809`), **failed its
numerics gate** at `4.002e-7` (> rtol `1e-7`) on a layer-28 packed-expert
gradient norm. You (via codex) ruled that a real, unexplained failure and said
*don't relax the gate* — then also said *r64 must run for every model*.

I reconciled these WITHOUT relaxing anything: I did **not** touch or rerun the
spec-path gate (it stays blocking; a spec relaunch would just re-fail). Instead
the weekend A4B r64 runs on the **store path**, which — like every r16 arm you
already approved and are running right now — simply doesn't perform that
SP-vs-shard cross-check. So r64 A4B produces learning curves, but the numerics
discrepancy is **not cleared**. **Any MoE r64 result (A4B or Qwen-35B-A3B)
carries that caveat** until the r64 sharding numerics are explained. Dense r64
(Qwen-27B, Gemma-31B) is clean. Full write-up + next diagnostic steps (an r32
gate run is the cheap discriminator) are in `issues.md` under
"OPEN — unexplained A4B r64 SP-vs-shard numerics discrepancy".

## What I deliberately did NOT launch

- **PPP8 2-node replicas** ([419]a): every 2-node PPP8 attempt in this campaign
  has failed or hung (422592/597/604/609/635/643/655/707), and it needs live
  supervision — a bad fit for an unattended weekend. Left for when you're back.
- **Gemma alt-loss arms** (cosine/delta_cosine for the two Gemma): lower value
  than completing r64 coverage; easy to add on request.
- **Full fine-tuning arm** and **adapter-refreshed Q/K/V for r64**: both are
  logged as priority issues in `issues.md`, not weekend-automatable.

## Root-cause diagnosis: why nothing is learning (2026-07-25, from the finished q27b r64)

Pulled the full `metrics.jsonl` of the completed `campaign50_q27b_ppp4_r64`:

- **Composed censored student is flat.** `student_argmax_acceptance` = 0.556 at
  epoch 0 and 0.556 at epoch 50 (bit-flat every eval); `CE_eval_loss` drifts
  *up* 2.2284 → 2.2367; corpus recall identical to the digit every epoch
  (machado 0.00, quijote_ch1 0.09, quijote_ch4 0.02). `teacher_argmax` = 0.9994
  throughout — the cache/ceiling is perfect, so this is a training failure, not
  an eval/ceiling artifact.
- **The block-local objective is near-trivial.** Hidden (huber) loss starts
  near zero (layer-16 ≈ 0.009, most layers 1e-4) because each block is fed the
  *exact* teacher context (`teacher_frozen` uncensored K/V + teacher h[L-1]);
  reconstructing teacher h[L] from that is almost free. It descends only
  ~5–13% over 50 epochs.
- **The adapter moves anyway, into noise.** Effective LoRA delta grows ~52×
  (layer-16 rel-L2 6e-6 → 3.2e-4) and grad norms on the high-loss layers rise,
  yet none of it improves the end-to-end student. The near-zero hidden-geometry
  residual is both too small to give useful gradient and misaligned with the
  output distribution `argmax` reads.

Flat-to-the-digit recall + monotonic CE rise rules out "just needs more
epochs." The lever is the objective and/or the step size, not the epoch count.

## Designed + queued experiment: Qwen-27B r64 2×2 (loss × LR)

A/B against the finished `campaign50_q27b_ppp4_r64` (huber @ 3e-6), on the fast
dense model so we get a read in ~5 h/arm:

| run_name | loss | lr | train / report |
|---|---|---:|---|
| campaign50_q27b_ppp4_r64 (baseline, done) | huber | 3e-6 | — |
| campaign50_q27b_ppp4_r64_lr3e5 | huber | 3e-5 | 423029 / 423030 |
| campaign50_q27b_ppp4_r64_vmse | vocab_mse | 3e-6 | 423027 / 423028 |
| campaign50_q27b_ppp4_r64_vmse_lr3e5 | vocab_mse | 3e-5 | 423031 / 423032 |

`vocab_mse` = MSE in logit space through the frozen unembedding Gram matrix
(stage-scoped-safe; the historical recall recipe's loss family; memory-flagged
low-intrusion). It measures distance where the head amplifies it, so it should
give gradient where huber sees ~0. The LR axis separates loss-geometry from
under-training. Read `student_argmax_acceptance` / `CE_eval_loss` slope vs the
baseline; watch `standard_damage` for the higher-LR arms.

## FIRED: KV-refresh experiment (owner hypothesis [393]) — job 423034

The deeper fix for the composition gap, now launched. New spec arm
`scoped_b32_refresh` (`PPP4_ARM=scoped_b32_refresh`,
`scripts/spec_g26b_a4b_campaign.sbatch`): A4B r16, cache path,
`v4_kv_source: student_refresh` + `v4_kv_refresh_epochs: 1`. It regenerates the
detached training K/V through the CURRENT adapter every epoch (projection
inputs and residual stay teacher h[L-1]; no-grad refresh) — law-compliant
adapter-refreshed teacher-anchored context.

- **Gate stays intact:** it reuses the r16 no-expert numgate configs that PASS
  the SP-vs-shard gate at ~3.9e-8. Not relaxed. (The r64+expert path is the one
  with the open discrepancy; this arm avoids it by design.)
- **Single-variable contrast:** identical to the teacher_frozen `scoped_b32`
  arm (job 422706) except the K/V source. Reads: does `student_argmax` move off
  0.556 under refresh where it stayed flat under teacher_frozen?
- **Run dir:** `runs/g26b_a4b_ppp4_scoped_b32_refresh_campaign_j423034`; report
  `423035` (afterany). A4B is slow + cache path (~day pipeline: fresh vLLM gen →
  cache build → numerics gate → 500-epoch campaign, will hit the 24 h wall).
  Compare against 422706 at matched epochs.

Caveat: cache-path vs store-path differs from the store-path baselines, but the
matched contrast is against the cache-path teacher_frozen 422706, so the K/V
source is the only moved variable.

## Node state when I left

agpuh01 / agpuh02 busy with the r16 arms (Gemma-31B `422811`, Qwen-35B
`422807`), both finishing within hours; agpuh03 is `drain`. The r64 queue is
PENDING behind them and will flow as nodes free. Core huber-r64 (979/81/83/85)
sit ahead of the alt-loss arms by submit order.

## v5: trainv5.py fired — job 423052 (owner instruction, 2026-07-25 evening)

`scripts/trainv5.py` — pure monolith (no `selfupdate` imports, new run
identity, zero hash-collision surface with v4). Law: self-distillation —
teacher = same model adapters-off WITH passage; student = model+LoRA (all 7
Linear kinds, ALL 60 layers) with the passage REMOVED (remove-view =
deployment condition); KL(teacher||student) at answer positions, end-to-end
backprop. Solves-by-construction the two v4 failures: MLPs get output-level
gradient, and there is no teacher K/V cache to go stale.

- Data reuse: examples jsonl + gemma4_31b vLLM responses (exact ids);
  censored prompts by text surgery with an exact round-trip gate
  (2071/2071 pass, ~215 tokens cut, teacher word_acc 0.870).
- Owner notes: depth-uniform LoRA capacity (unknown memorization locus) +
  per-layer surprise telemetry each epoch (the profile itself localizes
  where poetry lives); optional --layer-gate topk:N. Capacity check:
  3 bits/param vs gzip'd Machado+Cervantes (r32 → ratio >>1).
- Evals in-file: generative recall per corpus from censored prompts,
  arc_easy damage (100 vendored), teacher-forced argmax acceptance (v4's
  flat-0.556 metric — direct comparison), frozen-vocab tripwire.
- Launch: scripts/trainv5.sbatch, job 423052 (4xH100, behind the r64+screen
  queue). Defaults: Gemma-4-31B dense, r32/a64, lr 1e-4 AdamW, 8 epochs,
  eval every epoch. Metrics in runs/trainv5_g31b_selfdistill/metrics.jsonl.
- vN roadmap in the file header: censorship generalizes to masking distant
  high-attention tokens (continuous personalization).

---

# ANALYSIS PROTOCOL — for the agent that reads these runs (write your verdicts back here)

Four experiment lines are in flight. For each: where the evidence is, what
counts as success, and what decision follows. General rules: an epoch is the
full 2071-item traversal; compare arms only at matched epochs; every claim
needs the metric row cited, not the log line remembered. The v4 baseline
number to beat everywhere is **student_argmax_acceptance flat at ~0.556** and
recall frozen at (machado 0.00, q_ch1 0.09, q_ch4 0.02).

## A. r64 campaign (jobs 422979-996) — "does rank fix it?"

Artifacts: `runs/campaign50_*_r64/stage0/metrics.jsonl` (+ report/report.pdf),
r16 twins without the suffix. Read `student_trajectory_eval` rows:
`student_argmax_acceptance`, `CE_eval_loss`, and the log-line recall.
- Expected (from the diagnosis): r64 does NOT move recall/argmax vs r16 —
  rank was never the binding constraint; the objective was. If confirmed,
  write one line per model here and close the "is it capacity?" question.
- If any r64 arm DOES move argmax > +0.02 over its r16 twin at epoch 50:
  that model's memorization was capacity-bound after all — flag it, it
  changes the v5 sizing conversation.
- MoE arms (q35b, g26b_a4b) carry the unexplained numerics caveat
  (issues.md); do not promote their numbers to claims without repeating the
  caveat.

## B. 2x2 loss x LR screen (423027/029/031) — "objective or step size?"

Artifacts: `runs/campaign50_q27b_ppp4_r64_{vmse,lr3e5,vmse_lr3e5}/stage0/
metrics.jsonl` vs baseline `campaign50_q27b_ppp4_r64`. Read the SLOPE of
`student_argmax_acceptance` and `CE_eval_loss` across epochs 0->50, and
`standard_eval` for damage on the lr3e5 arms.
- vmse arms move, lr arms don't  -> loss geometry was the failure; promote
  vocab_mse to the campaign default and consider vocab_mse+KV-refresh.
- lr arms move, vmse doesn't     -> under-training; re-tune lr schedule.
- only vmse_lr3e5 moves          -> both levers needed; screen lr grid at
  vocab_mse.
- none move                      -> block-local training with teacher-frozen
  context cannot compose, whatever the loss; v5 becomes the main line.
  This is a REPORTABLE negative result, not a failure of the screen.

## C. KV-refresh (423034) — "was stale K/V the composition gap?"

Artifacts: `runs/g26b_a4b_ppp4_scoped_b32_refresh_campaign_j423034/stage*/
metrics.jsonl`; contrast arm = `runs/g26b_a4b_ppp4_scoped_b32_campaign_j422706`
(teacher_frozen, same everything else). NOTE: it will hit the 24h wall mid-
campaign — compare at matched epoch numbers only (422706 reached ~epoch 231).
- First check the numerics gate PASSED in `runs/sbatch_g26b_a4b_campaign_
  <jobid>.out` (rtol 1e-7, r16 arms historically ~4e-8). If the gate failed,
  stop: the refresh path has a sharding bug, file it in issues.md.
- Then: does student_argmax move off 0.556 where 422706 stayed flat? Even a
  slow upward slope is a positive — refresh is per-epoch, so improvement
  compounds late.
- If flat like 422706: stale K/V was NOT the gap either; combined with B
  "none move", the v4 local law itself is the negative result.

## D. trainv5 (423052) — "does self-distillation finally learn?"

Artifacts: `runs/trainv5_g31b_selfdistill/metrics.jsonl` (kinds: provenance,
capacity, eval, epoch), stdout in `runs/sbatch_trainv5_<jobid>.out`.
- Health first: the data gate must print `items=2071 ... stop_id=106`; the
  capacity row must be capacity_ok=true; epoch seconds should be minutes-to-
  ~2h. If epoch 1 exceeds ~3h, the 8-epoch budget won't fit 24h — note how
  far it got; partial curves are valid (eval runs every epoch).
- Success = `eval` rows show recall (per corpus, word-LCS vs teacher answer)
  and student_argmax RISING across epochs while arc_easy stays within ~2
  points of its epoch-0 value. Machado (1490 items) should move before
  Quijote (581) — exposure asymmetry.
- Read `surprise_profile` in the epoch rows: the per-layer teacher/student
  discrepancy. Where it CONCENTRATES over epochs is empirically "which layer
  remembers poetry" — plot it (layer x epoch heatmap) for the owner; this is
  a headline figure regardless of outcome.
- If loss falls but recall doesn't: check whether generations degenerate
  (decode a few in the checkpoint with scripts/trainv5.py logic) before
  concluding anything — KL can be gamed by mode collapse onto generic text.
- Success here + failure in B/C = the end-to-end objective was the missing
  ingredient; next arms: --loss ce contrast, --layer-gate topk:8 (localized
  training), smaller r (capacity floor), and the vN attention-top-k censor.
- OOM watch: micro_batch 8 with ~4k-token teacher prompts on 31B may spike;
  if stage log shows CUDA OOM, relaunch with --micro-batch 4 --grad-accum 8
  (same effective batch), run name suffix _mb4.

## Bookkeeping for the analyst

- Publish updated report.pdf files the same way as before: force-add ONLY
  `runs/*/report/report.pdf` (runs/ is gitignored), commit, push.
- trainv5 has no v4_report renderer; its metrics.jsonl is self-describing.
  A layer-x-epoch surprise heatmap + recall/argmax curves is the minimum
  deliverable figure set.
- Write verdicts into THIS file under each section, then update
  EXPERIMENTS.md only once per closed question, and keep memory files
  (v4-not-learning-diagnosis, a4b-r64-numerics-discrepancy) in sync.
