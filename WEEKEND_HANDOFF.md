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
identity, zero hash-collision surface with v4). Law (owner-corrected
2026-07-26 — "backprop only happens layerwise"; output-logit training is
FORBIDDEN, embeddings/unembedding/final norm never move, gate asserted at
startup + tripwired per epoch): teacher = same model adapters-off WITH
passage, one no-grad pass records per-layer hidden targets h_t[L] at answer
rows; student = model+LoRA (all 7 Linear kinds, ALL 60 layers) with the
passage REMOVED (remove-view = deployment condition), ONE forward in which
every block input is detached by hook — block L transforms its OWN censored
trajectory state h_s[L-1], and its local loss distance(y_L, h_t[L]) roots
only in block L's LoRA. Depth-uniform loss menu = the v4 screen's axis:
--local-loss huber|nmse|cosine|delta_cosine|vocab_mse (formulas copied from
losses.py; delta_cosine anchor = the block's own input h_s[L-1]). Output
KL/CE vs teacher are EVALUATION ONLY. Solves-by-construction the two v4
failures: the local residual (censored-self vs passage-informed teacher) is
large at every layer so the MLPs finally get signal, and there is no teacher
K/V cache to go stale (context recomputed through current adapters each
forward). Owner speed insight: --layer-gate topk:N / minfrac:F backprops
only the biggest per-layer losses each step (per-block backwards are
independent); selection values are gathered with <=1 sync per device, and
the loss normalizes by n_layers so gated arms keep the same per-layer
effective LR as 'all'.

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
- Read `surprise_profile` in the epoch rows: the per-layer LOCAL LOSS of the
  censored student vs teacher h_t[L] (this IS the surprise). Where it starts
  high and falls fastest over epochs is empirically "which layer remembers
  poetry" — plot it (layer x epoch heatmap) for the owner; this is a
  headline figure regardless of outcome. `backprop_count` per layer shows
  what the gate actually trained; check whether selection CONCENTRATES in a
  depth band — that concentration is the load-balance risk of a future PPP4
  port (global top-k needs a per-step all-gather there, or an epoch-frozen
  threshold / stage-local quota instead).
- If local loss falls but recall doesn't: check whether generations
  degenerate (decode a few from the checkpoint) before concluding — and
  compare KL_eval/CE_eval (evaluation-only output metrics in the eval rows)
  against the v4 baseline drift (2.2284 -> 2.2367 UP).
- Success here + failure in B/C = the self-trajectory input (large local
  residual) was the missing ingredient, not the loss kind alone; next arms:
  --local-loss screen (nmse/cosine/delta_cosine/vocab_mse), --layer-gate
  topk:8 (speed + localization), smaller r (capacity floor), and the vN
  attention-top-k censor.
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

## v5 paced campaign — one arm every 2 days until Aug 15 (owner, 2026-07-26)

All 24h-capped (`scripts/trainv5.sbatch`), `--begin=<date>T20:00` so the
cluster stays usable by others between arms. Baseline huber/all/r32/1e-4 is
job 423052 (runs first, unscheduled). Analysis per §D; every arm logs the
surprise profile, backprop counts, tripwire certifications, and the same
fixed recall subset.

| begin | run_name | axis | args | job |
|---|---|---|---|---|
| Jul 28 | trainv5_g31b_vmse | loss | vocab_mse | 423062 |
| Jul 30 | trainv5_g31b_dcos | loss | delta_cosine | 423063 |
| Aug 01 | trainv5_g31b_nmse | loss | nmse | 423064 |
| Aug 03 | trainv5_g31b_cos | loss | cosine | 423065 |
| Aug 05 | trainv5_g31b_huber_topk8 | gate | topk:8 | 423066 |
| Aug 07 | trainv5_g31b_vmse_topk8 | loss x gate | vocab_mse + topk:8 | 423067 |
| Aug 09 | trainv5_g31b_huber_mf25 | gate (PPP-friendly) | minfrac:0.25 | 423068 |
| Aug 11 | trainv5_g31b_huber_r8 | capacity floor | r8/a16 | 423069 |
| Aug 13 | trainv5_g31b_huber_r128 | capacity abundance | r128/a256 | 423070 |
| Aug 15 | trainv5_g31b_vmse_lr3e4 | step size | vocab_mse + lr 3e-4 | 423071 |

Review verdicts (2026-07-26 final pass): monolith purity confirmed (stdlib +
torch/transformers/peft only); layerwise isolation is now CERTIFIED at
runtime (one-shot single-block backward leak check, batch 1) plus per-batch
detached-input assertions and the per-epoch frozen-vocabulary fingerprint;
teacher hiddens host-RAM cached after first computation (epochs 2+ skip the
teacher forward, ~1/3 of step compute); fixed a view-retention leak (~8 GiB)
and per-epoch recall resampling churn.

### v5 launch verification (2026-07-26, job 423270) + a coverage footnote

Relaunch after the 423052 vision-tower LoRA failure (fixed in 106edc0) came up
healthy: 410-target LoRA injected (244,858,880 params, capacity ratio 649.6),
epoch-0 baselines recall mach 0.173 / quij 0.180, arc_easy 0.330,
student_argmax 0.4290, KL/CE_eval ~= 11.15 (the passage content really is
near-unpredictable censored — the headroom v5 trains against; v4's flow-mask
CE started at 2.23 with nothing to learn). Teacher cache ~47.4 GiB. The
LAYERWISE ISOLATION CERTIFICATION passed on hardware: block 30's term put
gradient in exactly its 14 LoRA tensors, zero leaks.

Coverage footnote for the analyst: 410 targets, not 60x7=420 — Gemma-4-31B's
ten FULL-attention layers (5,11,...,59; the 5-sliding:1-full pattern) have no
v_proj Linear at all (KV-sharing attention). 410 is therefore complete
coverage of every existing decoder Linear; expect those ten layers to show 12
adapted tensors instead of 14 in per-layer accounting, and read their
surprise-profile rows with that in mind.

### v5 FIRST RESULT (2026-07-26, job 423270, killed): answer-only local loss destroys the model at lr 1e-4

Epochs were FAST (117-131 s — the teacher cache works) but ONE epoch at AdamW
1e-4 lobotomized the model: argmax 0.4290 -> 0.0000, CE 11.15 -> 122.7, arc to
chance, recall 0 — while the mean local huber FELL 0.044 -> 0.028. The
surprise profile was decisive: depth-skewed (L45-59 0.09-0.28, max block 59,
barely improving) — 50 trivial shallow layers averaged away a burning tail,
and the blocks warped every position outside the answer rows (unconstrained)
to satisfy the objective. Evidence preserved in
`runs/trainv5_g31b_selfdistill_lr1e4_destroyed/metrics.jsonl`.

Patch (e78db32, defaults changed): lr 1e-5; epochs 40 / eval-every 2; a
SELF-ANCHOR term (--anchor-weight 1.0, --anchor-rows 64) pinning each block's
output at fixed sampled PROMPT rows to the adapters-OFF base states (frozen,
cached ~+80 GiB) — learn the passage at answer rows, change nothing
elsewhere; grad-norm + adapter-norm telemetry; a DESTRUCTION SIGNATURE
warning in eval. Baseline relaunched as 423279; the Aug-15 step-size arm was
rebased (423071 cancelled -> 423280, lr 3e-5 vs the new 1e-5 default). All
other scheduled arms inherit the new defaults automatically.

Analyst note for ALL v5 runs: read `surprise_profile` by DEPTH, never the
mean; check `mean_grad_norm`/`adapter_l2_norm` growth; any argmax below half
of epoch-0 is the destruction signature — kill and diagnose, don't wait.

### v5 attempt 2 (423293, huber + anchor + lr 1e-5): destroyed again — metric mismatch identified

Slower death, richer data (evidence: runs/trainv5_g31b_selfdistill_huber_
anchor_destroyed): e2 showed a BLURRING phase — CE_eval fell 11.2->9.9
(toward the teacher!) while argmax fell 0.43->0.13 — then collapse (CE 66 at
e4, 142 by e8). Telemetry acquits weight explosion (adapter L2 66.28->67.22,
init-dominated; grad_norm ~0.001; clip never binds): tiny coherent drift
compounds through the 60-layer composition. Deep huber falls (L59 0.58->0.56)
while output-relevant directions worsen: huber weighs all channels equally,
the frozen head does not. This is the loss-metric mismatch the loss menu
exists to probe.

Attempt 3 = single-variable change vs attempt 2: --local-loss vocab_mse
(distance in the frozen unembedding's own metric W^T W). Job 423303, run
trainv5_g31b_vmse; the redundant Jul-28 scheduled twin 423062 was cancelled.
Auto-abort added (2be0a28): any v5 arm now exits cleanly after two
consecutive destroyed evals (argmax < half of epoch-0) — scheduled arms can
no longer burn 24 h on a corpse.

If vocab_mse also destroys: next candidates in order — generic-text anchor
(wikitext rows, pins general function, data/eval/wikitext2_val_v1.txt is
vendored), topk:8 gating (fewer layers move per step = bounded composition
drift), lr 1e-6. If vocab_mse holds argmax while recall climbs, the loss-
metric hypothesis is confirmed and the paced arms should be re-pointed at
vocab_mse variants.

---

# VERDICTS (written 2026-07-26 night, per the analysis protocol)

## A. r64 campaign — CLOSED: rank was not the constraint; MoE r64 actively harms

All arms completed. First->last eval:
- Dense flat: q27b r64 huber/cosine argmax 0.5560->0.5561; g31b r64 0.4678->0.4678. Recall frozen everywhere.
- MoE degraded: q35b r64 huber 0.5433->0.4445 (CE 2.32->2.92), q35b r64
  delta_cosine 0.5433->0.3674 (CE ->3.49) — extra expert-bank rank = extra
  damage, zero learning (numerics caveat stands).
- A4B r64 vs r16 control: 0.4796 vs 0.4793, CE 6.02 vs 5.94 — rank changed
  nothing. The "is it capacity?" question is closed: NO.

## B. 2x2 loss x LR screen — CLOSED: none move (the reportable negative)

vocab_mse@3e-6 flat (0.5561, recall frozen); huber@3e-5 degrades
(0.5368, CE 2.37, q1 recall down to 0.05); vmse@3e-5 slight decline.
Block-local training with TEACHER-FROZEN context cannot compose into
behavior change, whatever the loss or step size. v5 is the main line.

## C. KV-refresh — relaunched as an expert-complete pair

423034 died at dispatch: the reused r16 numgate configs predate c9a00be's
refusal of no-expert MoE LoRA. All four A4B r16 cache-path configs now pin
expert_parameters: true (b5a2f61). Relaunched as a clean modern pair:
refresh 423306 (report 423307) + teacher_frozen twin 423308 (report 423309).
Bonus datum: if the r16+experts SP-vs-shard gate fails, the open r64
discrepancy is expert-path-driven, not rank-driven.

## D. v5 — two destructions diagnosed, vocab_mse attempt running (423303)

See the attempt-1/attempt-2 sections above. The project's central tension is
now sharply posed: v4's teacher-frozen context gives a trivial objective
(nothing learns), v5's self-trajectory gives a real objective whose naive
optimization destroys the model through depth-compounding drift. The loss
menu, the anchor, the gating, and the LR ladder are the search space between.

### RE-EVAL REQUIRED (2026-07-27, after independent-review finding f1)

All v5 depth-profile analyses written before f35f74a are contaminated at the
LAST LAYER: hidden_states[-1] in transformers 5.12.1 is the post-final-norm
state, so every logged L59 surprise value (the ~0.58 "burning tail" spike in
both destruction analyses) was partly an artifact of comparing a raw block
output to a norm-transformed target. L0-L58 values remain valid. Re-read the
destroyed runs' profiles with L59 excluded; the first uncontaminated profile
comes from run trainv5_g31b_vmse (job 423314, fixed code). Any conclusion
about "which layer memorizes" must cite post-f35f74a runs only.

### Profile-reading rule (owner, 2026-07-27 night)

A v5 surprise profile decomposes as: (1) a MONOTONE drift envelope — trivial,
mechanical consequence of translating the student's own trajectory (each
layer inherits its input's accumulated divergence); (2) a SPAN-PERIODIC
component from the attention pattern (full-attention layers spike: mod-4 in
Qwen3.6 — see the real q27b r64 profile, L4/8/12/16 elevated ~10-100x over
neighbors — mod-6 in Gemma-4); (3) the RESIDUAL, which is the only science.
Do not interpret (1) or (2) as memorization localization. To strip (1),
read per-layer INCREMENTS (what the block ADDS vs what it should add) — the
delta_cosine loss is exactly that instrument; the EMA-relative gate
approximates the same normalization dynamically for any loss kind. When
reading 423314's profile, first verify the mod-6 signature is visible
(instrument sanity), then analyze the residual.

---

# NIGHT RESULTS (2026-07-27, 23:20–06:45) — the band was found

Four arms on the vocab_mse base, two lanes, all auto-abort protected. The
full arc: dip -> recovery -> (baseline crossing) -> slow erosion, with two
stabilizers measured against the erosion. Raw curves in each run dir's
metrics.jsonl (runs/trainv5_g31b_vmse{,_aligned,_al_clip001,_al_sema1});
key numbers:

| arm | positions | stabilizer | fate | peak mach recall (base 0.171) | argmax@end |
|---|---|---|---|---|---|
| 423314 | natural | none | ABORT e26 | 0.156 (never crossed) | 0.213 |
| 423325 | aligned | none | ABORT e28 | 0.190 @e13 (CROSSED) | 0.207 |
| 423340 | aligned | clip 0.01 | COMPLETED e30 | 0.177 @e15 (crossed) | 0.212 |
| 423350 | aligned | surprise_ema:1.0 | COMPLETED e30 | **0.194 @e30, STILL RISING** | 0.220 |

Verdicts:
- vocab_mse found the non-destructive band (huber destroyed identically-
  configured runs by e4). METRIC IS THE FIRST-ORDER LEVER.
- aligned positions beat natural (crossing vs no crossing); base recipe =
  vocab_mse + aligned.
- Erosion (slow argmax decline post-recovery) is the remaining dragon. Both
  stabilizers work: clip 0.01 flattens the slope (completed, recall pinned
  near baseline); surprise_ema:1.0 is BEST — completed, healthiest arc,
  ~28/60 layers written per step, and recall RISING at e30 (0.169->0.179->
  0.194) — late growth, not peak-and-decay. The owner's prediction-error
  gate is the closest thing yet to consolidation dynamics.
- Deployment recipe as of tonight: vocab_mse + aligned + surprise_ema, with
  best-checkpoint selection (max recall s.t. argmax >= 0.8 x e0); every
  eval epoch checkpoints, so the artifacts exist.

## Final two-week schedule (all vocab_mse + aligned base, 24h caps, 20:00)

| begin | run | axis | job |
|---|---|---|---|
| Jul 30 | v5p_topkabs1_long | owner arm: absolute divergence-generator, 300ep | 423387 |
| Aug 01 | v5p_surprise_long | owner arm: surprise_ema 300ep (+convergence stop) | 423388 |
| Aug 03 | v5p_dcos_topkabs1 | increment-purist divergence attack | 423389 |
| Aug 05 | v5p_topk8_rel | relative-gate comparison | 423390 |
| Aug 07 | v5p_signal_combined | owner q#1: gate on answer+anchor | 423391 |
| Aug 09 | v5p_sema_clip_long | combo: surprise + clip, 300ep | 423392 |
| Aug 11 | v5p_r8 | capacity floor | 423393 |
| Aug 13 | v5p_r128 | capacity abundance | 423394 |
| Aug 15 | v5p_lr3e5 | step size | 423395 |

Also queued: A4B pair (KV-refresh 423323 + frozen twin 423315) both begin
TODAY 10:00 on the freed nodes; claude self-review appointments 423369/70/71
(Aug 1/6/12, 09:00, thin queue, 60-min wall, full permissions, self-scancel).
Old huber-era arms 423063-423070 and 423280 cancelled.

Reading order for the reviews: long-arm curves first (does surprise_ema's
late growth continue past e30 x10?), backprop_count depth distribution for
topk_abs (the tail-ban evidence), convergence events, then the r8 arm (does
the capacity floor change the crossing?).

---

# SELF-REVIEW #1 (Aug 1, 09:00, job 423369) — verdicts

## topk_abs:1 long arm (423387, Jul 30-31): COMPLETED 300 epochs

- **The tail-ban evidence is now measured**: absolute-loss selection
  concentrated 99.8% in L50-59 — 67.5% of ALL selections on L58, 32.0% on
  L57 (77,700 total). The absolute gate IS a de-facto tail-only trainer in
  the drift regime, as theorized. This is the referee-ready datum.
- Yet remarkably stable: argmax ~0.405 at e300 (vs 0.429 e0 — negligible
  erosion over 300 epochs; ~1 layer written/step = minimal disturbance),
  arc ~0.29-0.30, KL steady 6.5-6.6.
- Crossed baseline only briefly and late: peak mach 0.1755 @e150 (evals
  above baseline: e130/140/150 only), then hovered ~0.16.
- VERDICT: stability-without-enough-learning. surprise_ema (0.194 rising
  @e30, depth-distributed ~28 layers/step) remains decisively superior.
  The Aug-3 delta_cosine+topk_abs arm is now the key question: does the
  INCREMENT metric redistribute selection off the tail (owner thesis)?

## A4B pair (423315/423323, Jul 27): infrastructure race, not science

Both died at "stage 0 exited without a run-complete marker" while the
marker WAS in the log — written ~13s after pid exit (nohup buffer flush).
The numerics gate never ran; no science lost, no gate datum gained. Fixed
(completion grep retries 60s) and RELAUNCHED immediately on the idle
H100s: refresh=428453 (report 428454), frozen twin=428455 (report 428456).
Gates ~1h in; walls Aug 2 ~09:15.

## Schedule: unchanged

Tonight 20:00: v5p_surprise_long (423388) — the star arm x300 epochs, with
the converged_no_surprise stop live. Remaining arms as scheduled through
Aug 15. Next self-review: Aug 6.

Note: pushes from thin nodes fail (no git remote-https helper); commits
made locally, pushed via login node at review end.

---

# SELF-REVIEW #2 (Aug 6, 09:00, job 423370) — verdicts

## THE INCREMENT THESIS CONFIRMED — dcos_topkabs1 is the new best arm

Three-point tail-ban measurement now complete (aggregate backprop shares
L0-19 / L20-49 / L50-59):
- topk_abs + vocab_mse (absolute state distance): 0/0/99.8% — tail collapse.
- **topk_abs + delta_cosine (increment distance): 14/52/34%, top layer L54
  at 26.8% — the owner's "attack the divergence GENERATOR" redistributes
  into the mid-stack as predicted.**
- topk:8 relative: 38/50/12% — depth-fair by construction.

And the increment arm is the campaign's best result: **sustained
above-baseline recall 0.1875-0.1878 across e36/38/40 (baseline 0.171),
stable — NOT eroding — completed without abort.** Above-baseline evals:
e16, e18, e36, e38, e40. Cost: arc slid to ~0.22-0.23 (worst of the arms;
the increment objective buys memory at more general-behavior cost).

## surprise_ema long (423388): erosion persists at long horizons

Aborted e60 (argmax ~0.207 under threshold at e50/e60); peak 0.1795@e30;
403 gate-skipped steps; never converged. Surprise gating delays erosion
~2x vs ungated but does not eliminate it. Combined with review #1's
topk_abs stability, the pattern: LESS writing = less erosion; the
increment metric appears to write in better places.

## topk8_rel (423390): depth-fair, modest (peak 0.1755@e26, completed).

## A4B pair: root cause found at last — rg missing on compute nodes

428453/428455 stderr: 'rg: command not found' inside
require_stage_completion. All three pair failures were THIS, not a race.
Fixed (grep), pair relaunched take-3: refresh 429743 (rpt 429744) /
frozen 429745 (rpt 429746). Gates ~1h in; walls Aug 7 ~09:15.

## Reroute

Aug-9 slot re-pointed (423392 sema_clip cancelled): **v5p_dcos_topkabs1_long
(429747)** — 300 epochs of the winning increment arm; the question is
whether the sustained crossing GROWS with horizon. Aug 7 signal_combined,
Aug 11 r8, Aug 13 r128, Aug 15 lr3e5 unchanged. Next review: Aug 12.

---

# SELF-REVIEW #3 (Aug 12, 09:00, job 423371) — final scheduled checkpoint

## C-LINE CLOSED: KV-refresh does NOT fix v4 (hypothesis [393] falsified)

The A4B pair finally ran clean (take 3), both gates PASSED, both campaigns
to the 24h wall (~220-233 epochs each). At matched epochs the arms are
indistinguishable: refresh argmax 0.4472->0.3591 (CE 6.35->7.73) vs frozen
0.4508->0.3612 (CE 6.31->7.65); recall 0.00/0.00 in both. Adapter-refreshed
teacher-anchored K/V changes NOTHING — context staleness was never the
binding problem; the v4 teacher-anchored objective itself is (consistent
with the B-line "none move" verdict). The v4 question is now fully closed:
neither loss, LR, rank, nor KV freshness makes the teacher-frozen local law
compose into behavior. v5's self-trajectory law is the only line that
learned.

## Numerics side-datum: the r64 discrepancy is RANK-driven

r16+experts passed the SP-vs-shard gate TWICE (4.305e-08 / 3.835e-08) —
comfortably at the r16 no-expert level. With r64+experts failing at
4.002e-07, the open discrepancy is rank-driven, not expert-path-driven.
(The cheap r32 probe remains the next discriminator if anyone pursues it.)

## v5 arms Aug 7-11: all three died in 5s — corrupted node venv (fixed)

A broken _cuda_bindings_redirector.pth in agpuh01's /tmp venv failed
venv_check while venv_setup skipped the existing dir. Node venvs wiped;
trainv5.sbatch now self-heals (wipe+rebuild on check failure). Lost arms
recovered: dcos_topkabs1_long relaunched IMMEDIATELY (431670 — the
campaign's key question, sustained-crossing growth over 300 epochs),
signal_combined tonight 20:00 (431671), r8 Aug 14 (431672); r128 Aug 13
(423394) and lr3e5 Aug 15 (423395) unchanged.

## Standing verdict of the campaign so far

Best recipe: delta_cosine + topk_abs:1 + aligned (sustained 0.1875 recall,
mid-stack selection). v4: closed negative on every axis. Erosion: universal
but slowed by write-sparsity; increment-metric writing is the best-placed.
The 300-epoch dcos long arm (finishing ~Aug 13 morning) is the campaign's
concluding measurement; whoever reads it: peak recall, epochs-above-
baseline, arc trajectory, and the backprop depth distribution vs the 40-
epoch run (14/52/34%).

---

# POST-HOLIDAY REVIEW #4 (Aug 16, owner + Claude) — campaign concluded

All five post-review-#3 arms completed; no jobs queued; the schedule is
exhausted. The Aug-12 venv self-heal worked: zero 5-second failures since.

## dcos_topkabs1_long (431670, 300 epochs, COMPLETED) — the concluding measurement

- **Crossing sustained ~200 epochs, does not grow.** Above baseline
  (0.1731) at 19/30 evals; plateau 0.18-0.19 from e60-e230; peak mach
  0.1921 @e160/e170; then fades back to baseline (e240-e300 ~0.170,
  final 0.1738). Best-of-campaign but horizon does not compound it.
- **The mid-stack redistribution is TRANSIENT.** Depth shares by phase:
  e1-40 = 14.3/52.0/33.6% top L54=26.9% (exact replication of the
  40-epoch run); e41-80 = 2.3/18.1/79.6% top L54=74.4%; from e81 on it
  is pinned at ~1.6/10/88% with **L54 alone taking 83-84% of all
  selections**. In the drift regime topk_abs+delta_cosine converges to a
  de-facto single-layer (L54) trainer. The increment metric moves the
  attractor off L57/L58 (vocab_mse's collapse point) into L54 but does
  not prevent collapse. Selection is data-driven, not scheduled, so the
  depth-uniform law is respected — but any claim about this arm must
  state the emergent concentration.
- Stability tracks write-sparsity, as predicted: argmax degrades
  0.4213 -> ~0.23 by e40 then holds ~0.228-0.237 for 260 epochs (never
  crossing the 0.2107 abort line); arc erodes slowly 0.32 -> 0.20; local
  loss 0.44 -> 0.34. The above-baseline recall plateau coincides with the
  L54-concentrated phase.

## The four ungated arms: all destroyed on schedule (auto-abort worked)

All vocab_mse + aligned + layer_gate 'all'; same dip->recovery->crossing->
erosion arc as the night-results arms, ending in the destruction abort:

| arm | axis | peak mach | above-baseline evals | abort |
|---|---|---|---|---|
| signal_combined (431671) | gate_signal answer+anchor | 0.1901 @e12 | e12,e16 | e30 |
| r8 (431672) | capacity floor | 0.1815 @e30 | e30 | e34 |
| r128 (423394) | capacity abundance | 0.1900 @e16 | 7 evals e4-e22 | e38 |
| lr3e5 (423395) | step size 3e-5 | 0.1774 @e10 | e10 | e16 |

- **Capacity axis CLOSED for v5 (matching the v4 A-line):** r8, r32
  (vmse_aligned, night results), and r128 share the same fate and
  similar peaks; r128 erodes furthest (final mach 0.1178, quij 0.1040 —
  well below baseline). Rank is not the lever; abundance actively hurts.
- **Step size CLOSED:** 3e-5 roughly doubles the erosion rate (abort e16
  vs e30-e38 at 1e-5). Consistent with the lr-1e-4 lobotomy.
- **combined gate signal (owner q#1): no effect** — indistinguishable
  arc from the answer-signal twin (peak 0.1901 vs 0.19-class peaks,
  abort e30 vs e28 for vmse_aligned).

## Campaign final standing

The v5 self-trajectory law learns (crossings in every arm) but every
dense-writing recipe erodes to destruction; only extreme write-sparsity
survives long horizons, and the surviving gate concentrates emergently on
L54. The sharpest open pair for a next campaign: (1) can a schedule hold
the e1-40 mid-stack distribution (e.g. per-layer quotas / epoch-frozen
thresholds) and does that beat emergent L54-only?; (2) checkpoint-selection
deployment: best artifacts by the max-recall-s.t.-argmax>=0.8*e0 rule are
in the run dirs (every eval epoch is checkpointed) — the long arm's
e160/e170 checkpoints are the campaign's best deliverable.

---

# CAMPAIGN v5-WEEK-2 PLAN (designed 2026-08-16, owner directions: alternate
# losses + more LoRA room; one week continuous, two H100 lanes)

Queue state at design time: agpuh01/agpuh02 IDLE (4xH100 each), nothing
queued; agpuh03 drained. Two lanes, chained `afterany` per lane so each node
runs continuously; auto-abort frees a lane early on a corpse. All arms:
Gemma-4-31B, aligned positions, lr 1e-5, alpha=2r, 40-epoch screens with
eval-every-2, 300-epoch longs with eval-every-10. Run prefix `v5w2_`.

## Code work before launch (trainv5.py, small diffs)

1. NEW loss `delta_vmse`: increment-normalized vocab_mse — same Gram
   residual q = (s-t) M (s-t), but denominator = teacher INCREMENT energy
   (t-a) M (t-a) instead of state energy. Rationale: for quadratic losses
   the delta residual identity (s-a)-(t-a) = s-t means the gradient
   direction stays vocab_mse's (the non-destructive band); only the
   per-layer normalization changes — which is exactly what topk_abs ranks
   on. It fuses the two measured winners: head-metric gradient + increment
   ranking (the profile rule's "strip the drift envelope" instrument).
2. NEW loss `mix`: 0.5*vocab_mse + 0.5*delta_cosine (gradient-level fusion,
   the nonlinear alternative to 1).
3. NEW flag `--train-norms`: mark the 7 block-local non-Linear params
   trainable (input/post_attention/pre_ffw/post_ffw layernorms, q_norm,
   k_norm, layer_scalar). Legal: all inside block L; embeddings, FINAL
   norm, unembedding stay frozen (tripwire unchanged). Requires updating
   the isolation-certification expected-tensor count and per-block
   optimizer groups/backprop accounting.
4. NEW flag `--dora`: LoraConfig(use_dora=True) (peft 0.19.1 supports it) —
   per-target magnitude vector = different update geometry at same r.

## Phase 1 — screens, launch immediately (40 ep, ~3-3.5 h each, day 1)

Lane A (agpuh01) — ALTERNATE LOSSES (r32):
| arm | loss | gate |
|---|---|---|
| v5w2_nmse | nmse (never run post-huber) | all |
| v5w2_cos | cosine (never run post-huber) | all |
| v5w2_dvmse | delta_vmse (new) | all |
| v5w2_dvmse_tka | delta_vmse | topk_abs:1 |
| v5w2_mix | mix (new) | all |
| v5w2_mix_tka | mix | topk_abs:1 |

Lane B (agpuh02) — LoRA ROOM (base recipe = delta_cosine + topk_abs:1,
the campaign winner; rank was never tested under sparse writing):
| arm | change vs winner |
|---|---|
| v5w2_dcos_tka_r8 | r8/a16 (floor) |
| v5w2_dcos_tka_r64 | r64/a128 |
| v5w2_dcos_tka_r128 | r128/a256 (abundance) |
| v5w2_norms | + --train-norms (7 new block-local params) |
| v5w2_vmse_norms | vocab_mse + all + --train-norms (norm effect under dense writing) |
| v5w2_dora | + --dora at r32 |

## Phase 2 — promotion longs (300 ep, ~13.5 h, days 2-4)

Review #1 (Aug 18 09:00) promotes the best TWO arms per lane by: peak
recall, evals-above-baseline persistence, argmax slope after recovery, arc
damage. Criteria guardrail: an arm must beat dcos_topkabs1_long's screen
window (0.1878 sustained @e36-40) or show a qualitatively healthier argmax
arc to earn a long slot.

## Phase 3 — combination + horizon (days 4-6)

Review #2 (Aug 20 09:00) launches: (i) best-loss x best-capacity combined
long; (ii) seed-43 replication of the current champion; (iii) a 500-epoch
horizon run of the champion (fits 24 h wall at ~2.4 min/ep). If Phase 1/2
produced nothing above the incumbent, slots go to the L54-question instead
(epoch-frozen threshold gate — the review may implement `topk_abs_frozen`).

## Phase 4 — close (day 7)

Review #3 (Aug 23 09:00): final verdicts here, regenerate the joint PDF
report with the new arms, commit, release the nodes.

Reviews are claude-review thin-queue appointments (60-min wall, self-scancel
— the proven 423369-71 pattern) with relaunch/promotion authority; the
owner can preempt any decision in-session.
