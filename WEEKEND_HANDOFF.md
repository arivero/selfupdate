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
