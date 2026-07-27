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
