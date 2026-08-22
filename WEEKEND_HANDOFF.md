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
3. NEW flag `--train-norms`: mark the block-local norm params trainable
   (input/post_attention/pre_ffw/post_ffw layernorms, q_norm, k_norm —
   6 real params/layer; smoke-corrected 2026-08-16 night: layer_scalar is
   a registered BUFFER, untrainable, and the first predicate also swept
   vision-tower norms — now exact text-stack names, gate 360/360, cert
   reads 14 LoRA + 6 norm = 20 tensors). Legal: all inside block L;
   embeddings, FINAL norm, unembedding stay frozen (tripwire unchanged).
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
| v5w2_norms | + --train-norms (6 norm params per block) |
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

## Steady-prefill invariant (owner, 2026-08-16 evening)

The H100 lanes must NEVER drain while planned science remains: keep at
least one runnable (pending-on-dependency) job queued per lane at all
times, so a lane rolls into its next arm the moment the previous ends —
reviews ADJUST a prefilled queue, they do not create one from scratch.
Installed at launch: two safe-bet 300-epoch longs per lane chained behind
the screens (lane A: v5w2_dvmse_tka_long 435187 -> v5w2_mix_tka_long
435188; lane B: v5w2_dcos_tka_r64_long 435189 -> v5w2_norms_long 435190;
24h caps, eval-every 10, knobs pinned). These cover the gap from screen
completion (~Aug 17 evening) through review #1 and beyond; every review
must (a) scancel/replace prefill arms the screen evidence has obsoleted
(a not-yet-started pending job costs nothing to replace), and (b) top up
the chains so the prefill horizon always reaches past the NEXT review.
Launched jobs (smoke 435170; lane A screens 435171-176; lane B screens
435177-182; reviews 435183/184/185 = Aug 18/20/23 09:00).

## TINC: teacher-own-increment targets (owner-approved 2026-08-16 night)

Diagnosis accepted by the owner after the "+0.02 is disappointing" review:
every v5 loss so far asks EACH block to close the FULL censored-vs-teacher
gap (target h_t[L] against input h_s[L-1] — delta_cosine included, its
teacher delta being h_t[L]-h_s[L-1]). Sixty blocks each doing the whole
job compose into over-correction — the measured depth-compounding
destruction — while gating to ~1 layer/step survives but stores almost
nothing. Destruction and timidity are the two ends of one target
inconsistency.

Fix implemented: `tinc_cos` / `tinc_vmse` match the block's ADDED vector
(y_L - h_s[L-1]) to the teacher's OWN increment (h_t[L] - h_t[L-1]).
Consistency proof: if every block matches its increment the composition
telescopes to the exact teacher state at answer rows (embeddings agree
under aligned positions) — each layer owns 1/60th of the job,
depth-uniform BY CONSTRUCTION. Anchor term falls back to the absolute
twin (same-trajectory target, already consistent). Block 0's h_t[-1] is
the embedding = the student's own block-0 input at answer rows.

Also implemented (live for all runs from ~21:30 on): `recitation` (share
of recall items with word-LCS >= 0.9) and `recall_items` (raw per-item
values) in every eval row — a mean of 0.19 cannot distinguish ten
perfect recitations among baseline noise from uniform formulaic overlap;
these can. Judge future arms on recitation first, mean second.

Queue surgery: prefill longs 435188 (mix_tka_long) and 435190
(norms_long) CANCELLED, replaced by tinc smoke 435199 (afterany:435187;
micro-validates both tinc losses + recitation metric) gating
v5w2_tinc_cos_long 435200 and v5w2_tinc_vmse_long 435201 (300 ep,
UNGATED dense writing — the consistency hypothesis predicts no
destruction; auto-abort protects the downside). Expected: smoke Mon
~09:30, longs Mon midday -> Tue, read at review #2 (Aug 20).
Prediction to check first: does ungated tinc survive past e10 where
every previous ungated arm was already eroding? Reviews own supervision
of the 4352xx+ jobs (the session monitor's glob covers 4351xx only).

## SCREEN PHASE VERDICTS (all 12 arms, written Aug 17 evening, live session)

Baselines: mach 0.173 / argmax 0.421 / arc 0.32. Recitation (LCS>=0.9)
was 0.0 in every arm that logged it — nothing verbatim anywhere yet.

Lane A (losses): delta_cosine remains the only storing loss.
- dvmse_tka: PRESERVATION TOOL — argmax 0.394/arc 0.32 at e40 (best ever)
  but recall never crossed baseline. Selection fully mid-stack (24/64/11,
  top L22=14.5%): increment-relative RANKING confirmed as the placement
  mechanism; placement alone does not store.
- dvmse ungated: destroyed e18 (full-gap target, as diagnosed).
- mix_tka: worse than both parents (0.159 end; vmse magnitudes dominate
  the ranking -> 65% tail L58; gradient blend dilutes storage). Fusions
  at the loss level FAIL.
- mix/nmse/cos ungated: destroyed e4 each. With historical huber: ALL
  isotropic hidden losses kill in <=4 epochs; only frozen-head-geometry
  losses reach the survivable band. Loss-menu table CLOSED.

Lane B (LoRA room): rank moves the clock and the damage, not the ceiling.
- r64: campaign-best 0.2024 at e40, RISING; erosion equal to r32.
- r8: same peak (0.2036@e32) with the least damage (argmax 0.30 at e40)
  and the most distributed selection (18% deep, L54=12.7%) — but
  peak-and-decay within 40 epochs.
- r128: flat (never crossed), erosion cost paid anyway; selection
  collapse ACCELERATES with rank (L54=47% by e40 vs 27% at r32/r64).
- ALL viable ranks touch the same ~0.19-0.20 recall wall (r32 long
  0.1921 / r64 0.2024 / r8 0.2036) — the dcos+topk_abs recipe has an
  intrinsic ceiling; rank selects when you hit it and what you pay.
- norms (+360 params, gated): null (ends 0.167, selection unchanged).
  vmse_norms (ungated): destroyed e20, FASTER than without norms. The
  LayerNorm-tuning literature prior does NOT transfer to this objective.
- dora: OOM incident (see below), retried as mb4.

Standing synthesis: a storage/preservation FRONTIER — every storing arm
writes deep and pays damage; every preserving arm stores nothing; fusion
gets neither. Only a target change can move the frontier -> the tinc
longs (Mon) are the decisive experiment; the r8/r64 longs measure the
wall precisely.

Censorship code review (owner-requested, Aug 17 14:00): surgery, spans,
and position alignment verified CORRECT (answer tokens land on exactly
the teacher's RoPE positions under aligned numbering; no passage tokens
reach any student path). One noted inconsistency: recall_eval generates
under NATURAL positions even for aligned-trained arms (teacher-forced
metrics use the training numbering). OWNER RULING: keep as is for the
whole campaign — recall stays the deployment-condition number; do NOT
add an aligned-generation variant or change the metric mid-campaign.
Reviews: leave this alone.

## Horizon extensions for the successful arms (owner, Aug 17 afternoon)

Screen evidence (r64 final 0.2024 RISING; r8 0.1952@e30 with argmax 0.35 —
the best storage-to-damage ratio of the campaign) earned two additions:
- v5w2_dcos_tka_r8_long 435545 (300 ep, eval-every 10; lane A tail,
  afterany:435200) — does the low-rank/low-damage arm accumulate?
- v5w2_dcos_tka_r64_h400 435546 (400 ep, eval-every 20 to fit the 24h
  wall; lane B tail, afterany:435527) — does the champion's e40 uptick
  compound past the r32 long's e160 peak-and-fade?
Escalation rule for reviews: any long that ends both ABOVE its 40-epoch
twin's recall and above baseline earns a follow-on horizon run (400 ep,
eval-every 20) in the next free tail slot; if a tinc long survives to
e300 with recall >= its own e40 value, tinc gets the same 400-ep horizon
treatment ahead of everything else. Judge on recitation first, mean
second, per the tinc section.

# SELF-REVIEW #1 (Aug 18, 09:00, job 435183) — verdicts

## The wall is confirmed at every rank; dvmse never stores; NAIVE DENSE TINC DESTROYS

- **r64_long (435189, 300 ep, COMPLETED)**: peak mach 0.1919 @e250,
  15/30 evals above baseline, final 0.1754, arc 0.20. Replicates the r32
  long's 0.1921 @e160 almost to the digit, just later. THE ~0.192 WALL IS
  RANK-INDEPENDENT — the screen's e40 spike (0.2024) was sample noise.
- **dvmse_tka_long (435187, 300 ep, COMPLETED)**: peak 0.1746 @e50, only
  2/30 evals above baseline, argmax leaked 0.394->0.259 over 300 ep.
  The dvmse line stores NOTHING at any horizon — CLOSED as a storage
  candidate (remains the best preservation datum).
- **tinc smoke 435199 PASSED (15 min)**; then BOTH ungated tinc longs
  DESTROYED at e20 (tinc_cos: argmax 0.42->0.07, CE 5.6->21; tinc_vmse:
  0.42->0.08, CE ->24 — worse than every band loss, similar to isotropic
  kills). The telescoping fixed point is correct but unreachable by naive
  dense descent: away from the fixed point each block adds an increment
  computed for a DIFFERENT input state, so composition still drifts, and
  the increment targets are large at every layer (they carry the
  passage's attention output), giving huber-class step sizes. Consistency
  of the target does not imply stability of the dynamics.
- Recitation: 0.0 in every run of the campaign so far.
- Infrastructure: zero failures overnight; auto-abort saved ~2x20h on the
  tinc pair; dora_mb4 pending on Resources (agpuh02 taken by another
  user — not an idle-lane fault); r8_long running on agpuh01.

## Actions (review 1)

- No further promotions: NOTHING beat the 0.1878 guardrail; the queued
  r8_long / r64_h400 already cover the remaining screen-derived value.
- NEW ARM: v5w2_tinc_cos_tka screen (436939, afterany:r8_long, 40 ep) —
  gated tinc. Rationale: sparse writing is the only regime every loss
  survives; topk_abs on increment mismatch is the natural
  increment-relative ranking (the mechanism dvmse validated); one block
  moving per step is where the telescoping argument is least violated.
  If tinc_cos_tka also fails to store, the tinc line closes and review 2
  goes to the L54 epoch-frozen-gate question per plan.
- Queue after this review: r8_long (running) -> tinc_cos_tka; dora_mb4 ->
  r64_h400 on the other lane. Prefill horizon reaches past review 2.

INCIDENT (Aug 17 ~11:00): v5w2_dora (435180) CUDA-OOM'd 52 min in, zero
epochs done — DoRA's merged-weight norm computation adds per-step memory
LoRA doesn't have, and a long-sequence batch tipped GPU 1 (the 48-item
smoke missed it). Chain advanced cleanly (afterany). Relaunched per the
§D protocol as v5w2_dora_mb4 (435527, --micro-batch 4 --grad-accum 8,
same effective batch, afterany:435201 -> runs Tue). If mb4 also OOMs,
drop the DoRA axis rather than shrink further — batch-shape variance is
the confound.

## Per-corpus localization telemetry (owner question, 2026-08-16 night)

The corpus is Machado (1490 items) + Quijote CHAPTERS 1-4 (581 items:
q1 123 / q2 150 / q3 147 / q4 161 — recovered by passage-matching against
raw_ch16's chapter segments; boundary windows tagged by start chapter;
sidecar data/combined/quij_chapter_map.json, loaded defensively at data
load). The owner's hypothesis: different texts may be STORED at different
depths. Instrumentation (live from the v5w2 screens onward):
- `surprise_by_corpus` in every epoch row: answer-only per-layer local
  loss split mach/q1/q2/q3/q4 (ITEM-mean; the aggregate surprise_profile
  stays batch-mean — do not mix the two normalizations).
- recall now reports per chapter (mach, q1..q4) plus a pooled `quij` key
  for curve continuity with all pre-v5w2 runs. The mach recall sample is
  UNCHANGED (same seed, first sorted group) — mach curves stay comparable
  across the whole v5 era; per-chapter quijote curves start fresh here.
- Analysis rule: apply the drift/span/residual decomposition PER CORPUS
  and compare RESIDUALS across corpora at matched epochs; a corpus whose
  residual falls fastest in a different depth band than another's is the
  localization signal. Exposure differs (~2.6x mach vs each chapter) —
  compare shapes and normalized slopes, not absolute values.

## Literature grounding (2026-08-16 review; for reports and referees)

- Closest relative: Deep Context Distillation (arXiv:2503.08727, COLM
  2025) — LoRA modules trained to match hidden states + logits of a
  document-in-context teacher; also found next-token prediction inferior.
  DIFFERENTIATOR of this project: our law is LAYERWISE (detached block
  inputs, no cross-block gradient, runtime-certified); all published
  context-distillation work backprops end-to-end. Related: on-policy
  context distillation; knowledge injection via self-distillation
  (arXiv:2412.14964); "When Context Returns" (arXiv:2606.11627) shows
  internalization succeeds at the representation level — our
  student_argmax metric measures exactly that.
- Local/blockwise training literature (blockwise SSL for video ViTs
  arXiv:2601.09040; DiffusionBlocks ICLR 2026; LoCA arXiv:2608.03020)
  trains from scratch or post-trains vision models; layerwise KNOWLEDGE
  INJECTION into a pretrained 31B LLM is unmapped territory.
- Capacity: Allen-Zhu & Li ~2 bits/param (ICLR'25), Morris et al. 2025
  ~3.6 bits/param — our 3-bits/param gate sits inside the literature
  range; at ratio ~650 rank should NOT bind (consistent with every rank
  result so far).
- Rank vs forgetting: 2025 work reports r64 LoRA/DoRA catastrophically
  forgetting in 3 epochs without mitigation — independent support for
  "abundance hurts"; sharpens the lane-B prior that rank amplifies
  erosion unless write-sparsity shields it.
- Norm tuning: LayerNorm-only fine-tuning (~0.004% params) can beat full
  FT in vision-language work (arXiv:2312.11420, ICLR'24; arXiv:2403.20284)
  — strong prior for the --train-norms arms.
- Localization: editing literature (ROME/MEMIT) places atomic facts in
  EARLY-TO-MID MLPs, while our topk_abs gate concentrates at L54/60;
  reconciliations: verbatim recitation may differ from relational facts,
  the L54 signal partly reflects accumulated drift (our decomposition),
  and Hase et al. showed causal localization correlates poorly with where
  editing works. The per-corpus telemetry above is our direct probe.

# QUEUE FILL Aug 19-21 (Aug 18 22:45, live session; owner: "review the
# results and fill the queue for the next three days")

## Tonight's ground truth

- **r8_long (435545, 300 ep, COMPLETED 21:47)**: NULL. quij never beats
  its e0 baseline (peak 0.196@e40 vs 0.189); mach peaks 0.201@e160
  (+0.03, noise band) then decays to 0.161; argmax erodes 0.42->0.22,
  arc 0.32->0.21, CE 5.6->11 then slow drift back to 9.9. The screen's
  "best storage-to-damage ratio" did NOT accumulate. Rank axis now closed
  in BOTH directions: r8/r32/r64 all live at the same ~0.19 wall; r8 just
  pays the damage more slowly. Recitation still ~0 (one 0.042 item flickers).
- **dora_mb4 (435527, started 21:48)**: healthy — e0 eval + layerwise
  certification (21 tensors, 0 leaks) passed, training walk underway at
  ~48/81 GB per GPU. The mb4 protocol cleared the DoRA OOM.
- agpuh02 still held by another user; agpuh03 still drained. Worst-case
  planning is single-node serial on agpuh01.

## The fill: execute the review-2 pivot early (L54/churn question)

Nothing beat the 0.1878 guardrail and every planned axis except gated
tinc is closed, so the pre-registered pivot starts now instead of at
review #2. Central question, split into mechanism vs placement:
is the ~0.192 wall caused by per-step topk_abs re-ranking CHURN (the
gate chasing the drift its own writes create), by WHERE it writes, or
by neither (a capacity/objective ceiling)?

Trainer additions (this commit): `--layer-gate fixed:L[,L2...]` (pinned
selection; ablation probe, same depth-uniform loss and LoRA on every
block, reports must show written-layer distribution alongside topk_abs
arms) and `--gate-freeze-epoch N` (topk_abs/topk dynamic until epoch N
completes, then selection frozen to the K most-chosen layers;
`gate_frozen` row logs the frozen set and cumulative counts).

Submitted (all delta_cosine, incumbent knobs pinned; smoke gates the
new code paths):
- smoke_gate2 437121 (afterany:436939) — micro fixed:22 + freeze-after-e1,
  asserts the gate_frozen row exists.
- Chain 1 (mechanism): f54 437122 -> tka_frz10 437123 ->
  tka_frz10_long 437124 (300 ep). fixed:54 = collapse layer without
  churn; frz10 = 10 epochs of exploration then no churn.
- Chain 2 (placement/width): f22 437125 -> tka2 437126 ->
  f54_long 437127 (300 ep). fixed:22 = mid-stack write (where
  delta_vmse's increment-relative ranking pointed); topk_abs:2 = the
  untested middle between 1 (wall) and dense (destroys).

Predictions (register before data): if churn is the cap, frz10/f54
should pass 0.192 somewhere in e40-300 with less argmax erosion per
recall point; if placement is the cap, f22 diverges from f54; if the
wall is intrinsic, both longs replay the r32 long's peak-and-fade and
the campaign closes on "frontier is objective-limited, not
scheduling-limited" — publishable either way.

Queue horizon: serial worst case runs dora_mb4 -> tinc_cos_tka ->
r64_h400 -> smoke_gate2 -> 4 screens -> 2 longs ≈ through Aug 22
morning, past review #2 (Aug 20 09:00) with review #3 (Aug 23) able to
prune the longs. Standing rules unchanged: recall metric stays
natural-position; tinc escalation rule from the horizon section applies
if 436939 stores.

## Overnight verdicts (Aug 19 14:40 check)

- **TINC LINE CLOSED.** v5w2_tinc_cos_tka (436939) auto-aborted at e10
  after 1h28: gated tinc destroys exactly like dense tinc, only slower
  (argmax 0.42 -> 0.096, CE 5.6 -> 20.5, recall -> 0 by e8; healthy
  through e4, collapse visible at e6). The escalation rule is moot.
  Verdict: target-inconsistency repair does not rescue increment
  matching at ANY gate width — increment-class targets are dynamically
  unstable, period. No further tinc arms, ever.
- **DoRA AXIS CLOSED (null).** v5w2_dora_mb4 (435527) hit its 6h
  TIMEOUT at e18/40 (mb4 is ~2x slower per epoch). Trajectory through
  e18 is indistinguishable from plain-LoRA erosion: quij peak
  0.1942@e12 vs 0.1914 e0 (noise), mach declining 0.1697 -> 0.1432,
  argmax 0.42 -> 0.27, CE 5.7 -> 11.3. No relaunch: with the rank axis
  closed both ways, DoRA showing the same wall shape and no early
  advantage means nothing more is extractable (kill-doomed-runs rule).
- **r64_h400 (435546) mid-flight, plateau NOT fading so far:** at
  ~e180/400 (9.3h elapsed, ~20 ep/h -> finishes right at its 24h cap
  ~05:15 Aug 20). quij oscillates 0.187-0.192 at the wall, mach
  recovered to 0.1738 (near e0), argmax stable ~0.23, CE slowly
  IMPROVING 10.0 -> 9.8. The r32 long faded by e240; whether r64_h400
  holds its plateau through e240-400 is exactly the horizon question it
  was queued for. Keep running.
- Node reality: agpuh02 held by another user, agpuh03 drained, so
  smoke_gate2 (437121, deps satisfied) waits on agpuh01 behind
  r64_h400. Gate screens therefore start ~05:30 Aug 20 — review #2
  (09:00) lands mid-chain 1/2 screens, which is fine; longs run Aug
  20-21 and review #3 prunes.

## Teacher-ceiling calibration (owner question, Aug 19 ~14:45)

The recall metric's scale is now anchored. `answer_text` (the scoring
reference) is the teacher's own vLLM generation from the uncensored
prompt; the responses artifact stores each answer's quality against the
ORIGINAL text under the same reference-word-LCS family:
teacher-with-passage = 0.9915 mach / 0.9839 quij, recite-rate ~0.98
(n=1822; the 249 cloze items are containment-scored). Censored base
(epoch 0) = ~0.17 mach / ~0.19 quij, recitation 0. So the ~0.192 wall
closes only ~2-4% of the censorship gap, and the teacher's entire
advantage is verbatim recitation — a regime no student arm has entered
for even one item. The one unmeasured number — HF-side reproduction of
the vLLM answers under recall_eval's exact decode path — is queued as
scripts/v5_teacher_ceiling_probe.py (job 438230, --nice=1000, fills the
first queue hole; writes runs/v5_teacher_ceiling/ceiling.json).

## ULTRAREVIEW #1 verdicts + actions (Aug 19 ~15:00, autonomous per owner)

Cloud review of v5-review-base(f1ce504)..HEAD returned 2 findings; both
verified real against the code and fixed on this branch.

1. **bug_002 (normal): --train-norms broke the frozen-teacher law.** The
   6 per-block norm tensors are trained BASE params; PEFT's
   disable_adapter() only gates the LoRA delta, so the teacher-target
   capture, the self-anchor capture, and evaluate()'s teacher logits all
   saw progressively drifted norms (epoch-1 caches froze targets against
   different optimizer states). FIX: frozen_base() context manager swaps
   in the decay_to_init snapshots around every adapters-off forward;
   no-op when --train-norms is off, so queued gate arms are numerically
   untouched.
   CAMPAIGN CONSEQUENCES:
   - v5w2_norms (delta_cosine+tka+norms, peak quij 0.2039@e20 — the best
     quij peak of the campaign) is CONTAMINATED: it trained against a
     moving teacher, not the v5 law. Its recall numbers stand as a
     deployment measurement of an UNintended recipe; the "norms help"
     reading is voided. Clean re-run queued: smoke 438233
     (afterany:437127, exercises frozen_base end-to-end) -> v5w2_norms_fix
     438234 (afterok:438233, same knobs as the contaminated arm). If the
     clean arm reproduces ~0.204, norms genuinely help and the moving
     teacher was harmless; if it regresses to ~0.196 (norms-free level),
     the campaign's best quij peak was a moving-target artifact.
   - vmse_norms's "norms accelerate destruction" verdict is RETRACTED,
     not re-tested: ungated vocab_mse destroys with or without norms
     (that axis closed on the base loss), so a clean re-run would not
     change any decision.
2. **bug_001 (nit): 2-node PPP8 sbatch lacked the marker retry loop**
   its PPP4 sibling got after the 2026-07-27 double loss (nohup flush
   trails pid exit ~13s). Copied verbatim. Path is deprioritized; fixed
   to stop sibling drift.

Queue after this pass: ... 437124/437127 longs -> ceiling probe 438230
(nice) / smoke_normsfix 438233 -> norms_fix 438234. Prefill now reaches
~Aug 22 evening; review #3 (Aug 23 09:00) closes.

## OWNER GRANT (Aug 19 15:05): independent-researcher mode

The agent takes the review #2 decisions AUTONOMOUSLY as soon as the data
suffices — the Aug 20 09:00 scheduled review job is a backstop, not the
decision point — and continues as independent researcher through review
#3 (Aug 23). No owner confirmation between decisions; every decision and
its rationale is documented in commits and in this file. Standing
constraints unchanged (layerwise law, frozen vocabulary, natural-position
deployment recall, tail ban, kill-doomed-runs, launch verification,
timestamped reports).

Pre-registered decision points and criteria:
- f54 vs f22 vs frz10 vs tka2 screens (land ~Aug 20 midday-evening):
  promote past the 0.1878 guardrail; prune/replace the two prefilled
  longs (437124/437127) if their screen arm underperforms churny
  topk_abs:1 — scancel pending is free.

  CRITERIA HARDENED (Aug 19 ~23:05, owner correction on optimistic
  reporting): the deployment metric is unchanged, but MY decision lens
  is now CONTENT-LEVEL. Full-LCS recall carries ~0.19 of free
  assistant-register scaffold (generation and teacher answer share the
  same preamble formulae); content-only LCS (inside «»/**** quotes,
  reconstruction verified 24/24 vs stored per-item scores) is
  0.108-0.168 at e0 and moved ~+0.01 at f54 e16 (q4 DOWN). Campaign
  content learning to date is ~0.12 -> ~0.13 vs teacher 0.99.
  Therefore: a screen is PROMOTABLE only on content-LCS gain >= +0.03
  over its own e0 by e40; if ALL FOUR gate screens fail that bar, BOTH
  prefilled longs (437124/437127) are pruned — a churn-free replay of
  scaffold oscillation for 300 epochs answers nothing — and the freed
  ~48 GPU-h go to review-#2-designed arms attacking content storage
  itself (exposure/spacing axis, or answer-row curriculum), or the
  campaign closes on the characterized null.
- recall_texts from the first screens + ceiling probe 438230: decide the
  half-recitation-ceiling hypothesis (derailment / early-stop /
  formatting / position-keyed); if position-keyed is live, ADD an
  aligned-generation diagnostic column (deployment metric untouched).
- norms_fix 438234: reproduce ~0.204 -> norms axis reopens (queue a
  norms long); regress ~0.196 -> best-peak was moving-target artifact,
  norms axis closes.
- r64_h400 e240-400 tail: plateau holds -> horizon axis positive,
  consider 400ep for the winning gate arm; fades like r32 -> peak-and-
  fade is horizon-universal, longs lose priority.

### DECISION #1 TAKEN (Aug 19 20:12, autonomous): r64_h400 FADES —
### killed at e320, node released to the gate chain ~4h early

Evidence: peak quij 0.1924@e120; e260 high (mach 0.1844/quij 0.1846)
then three consecutive declining evals — e280 0.1709/0.1779, e300
0.1649/0.1825, e320 0.1661/0.1762 — ending with BOTH corpora below
their epoch-0 baselines (quij e0 0.1885, mach e0 0.1715). This
replicates the r32 long's peak-and-fade ~60 epochs later: VERDICT
peak-and-fade is horizon-universal for churny topk_abs:1 at every
tested rank (32, 64) and horizon (300, 400). scancel 435546 at e320
(>12k-item rule satisfied ~200x over; kill-doomed-runs: the remaining
80 epochs could only re-describe a confirmed fade while the decisive
gate screens waited on the node).
Consequences applied: smoke_gate2 starts ~20:15 Aug 19 instead of
~05:30 Aug 20; screens land tonight, longs decision lands with them.
Reweighting (not yet a scancel): f54_long 437127 loses priority —
partially redundant with frz10_long 437124, which KEEPS priority
because the fade sharpens its question (does freezing the gate stop
the fade? the fade is now the phenomenon churn-freezing must cure,
not just the wall).

## CAMPAIGN v5w2 CLOSED (Decision #2 under grant, Aug 19 22:26)

Stopped the training campaign: scancel f54 437122 (mid-run, ~e22), f22
437125, dense_dcos 438298 (my own 20-minute-old submit — queued under
continuation logic that did not survive the objective-level analysis),
normsfix smoke 438233, norms_fix 438234. KEPT: ceiling probe 438230
(eval-only metric calibration, needed for the writeup) and the review
jobs 435184/435185 — #2 (Aug 20 09:00) is repurposed to synthesis +
successor design, #3 (Aug 23) to final close-out.

Basis (objective-level, NOT f54's short evening — f54 ran only ~2h,
e0-e22, enough for its mechanism datum):
1. Content recall across the campaign: ~0.12 -> ~0.13 vs teacher 0.99
   (scaffold-stripped LCS; full-LCS movement was assistant-register
   scaffold shared with the teacher's answers).
2. The optimization WORKS: 50/60 per-layer losses fall monotonically
   over 320 epochs (mean 0.44 -> 0.26; L54 -48% churny, -67% pinned).
   Ergo the objective is satisfiable without storing the sequence —
   per-position hidden proximity is too weak a proxy for content. No
   queued arm changed the objective; all were same-family cells.
3. Every axis closed: loss menu, rank (8-128), LR, horizon (40/300/400,
   peak-and-fade universal), gate schedule (churny/pinned/frozen),
   norms (contaminated + best peak was scaffold), tinc (destroys),
   dense isotropic (destroys), DoRA (null).
4. f54's evening datum: pinning the write layer reaches the same
   scaffold band with NO damage (argmax 0.41-0.42, ARC flat) while
   L58's loss rises (untrained downstream drift) — churn was the
   damage mechanism, placement was not the storage limiter.

Deliverables in hand for the writeup: calibrated metric (teacher 0.99 /
base 0.19 / scale decomposed into scaffold vs content), the loss-recall
dissociation, the damage mechanism (churn), the fade law (peak-and-fade
at every rank/horizon), the tinc instability result, recall_texts
tooling, and the frozen-teacher repair from ultrareview.

Successor directions (DESIGN questions for review #2, not launches):
an objective that couples per-layer states to token identity within the
law (teacher-sourced per-layer distributions through the frozen head —
note [[loss-safety-law]]'s intrusion warning), sequence-level local
targets, exposure/spacing regimes, retrieval-practice curricula. Whether
a v5w3 opens is an owner call on the science; the grant covers running
it, not deciding the program's continuation past this campaign.

## REVIEW #2 (Aug 20 09:00, thin node) — CAMPAIGN SYNTHESIS

No GPU movement since closure (probe 438230 + warm 438302 still pending
behind other users on every cluster GPU). Warm-verdict backstop review
installed: job 438537, Aug 21 09:07 (scripts/claude_review_v5w2_warm.sbatch
— post-closure prompt; the Aug-16 prompt in claude_review_v5w2.sbatch is
superseded, and review #3 435185 should read THIS section first).

### What v5w2 established (the laws, null-first)

1. CONTENT NULL (the headline): scaffold-stripped content recall moved
   ~0.12 -> ~0.13 vs teacher 0.99 across ~40 arms. Full-LCS "recall"
   carries ~0.19 of assistant-register scaffold shared with the teacher's
   answers; every screen-level "gain" was scaffold oscillation plus
   <=+0.01 content.
2. THE DISSOCIATION: the optimization works — 50/60 per-layer losses fall
   monotonically over 320 epochs (mean 0.44->0.26), L54 -48% churny /
   -67% pinned — while content stays flat. Per-position hidden proximity
   (delta_cosine and every tested variant) is satisfiable WITHOUT storing
   the token sequence. This is objective-level, not a tuning failure.
3. DAMAGE = CHURN: pinned writing (fixed:54) reaches the same band with
   NO argmax/ARC erosion; per-step re-ranking (topk_abs) pays 0.42->0.23
   argmax for the same band. Placement was not the storage limiter.
4. FADE LAW: churny arms peak and fade at every rank (8/32/64/128) and
   horizon (40/300/400 ep); r64_h400 fell below its own e0 by e320. The
   warm-start run (fixed:54 from the e120 peak adapter, churny
   continuation as built-in control) decides churn-vs-intrinsic.
5. INSTABILITIES: increment-target losses (tinc_*) destroy at any gate
   width; dense isotropic losses destroy; distribution/norm variants
   null or contaminated (frozen-teacher bug, since repaired).
6. GEOGRAPHY: the objective's divergence concentrates near-tail (L54 of
   0-59) — register territory — with a suspicious ~0 loss at L59 (audit
   pending). Mid-stack (fact territory per ROME/MEMIT) was never the
   gate's choice and the placement cell (f22) was cancelled at closure.

### E. PARTIAL TEACHER CENSORSHIP (owner design, Aug 20 night) — the
### refloat candidate, supersedes A-D framing if licensed

Idea: mask the passage from the TEACHER's attention at all layers except
retrieval set S (same hooks as the layer-censor probe). Why it answers
the measured failure: under the current law every layer's target embeds
passage contributions injected by all upstream globals — unreachable for
a passage-blind student at every depth, so gradient buys register
alignment (the scaffold null). With passage@S-only, targets outside S
are exactly reachable (they ARE passage-blind computations) and the
storage burden concentrates at S: target consistency achieved by
modifying the frozen teacher, not the loss (tinc done right). Law-clean:
teacher-sourced, block-local, depth-uniform; minimal-patch story
sharpens (single-injection-layer -> single-layer LoRA per memory).

LICENSE CONDITION (tonight's probes): censor-probe `only_S` must hold
teacher CE/acc near baseline (sufficiency) — that IS the feasibility
test; attention maps (Gemma 438970 / Qwen3.6 438972) pick S. If only_S
degrades badly, the idea dies before any GPU is spent — either way the
probes decide.

**V3 VERDICT (Aug 22 00:53, retrieval-only mask — passage encodes
itself, only post-passage queries blocked): LICENSE GRANTED ON GEMMA.**
only_S (passage visible only at the 8 top globals [5,17,23,29,35,41,
47,53]) = CE 0.0282 / acc 0.990 vs baseline 0.0031/0.999 — the globals
are SUFFICIENT for recitation; the v2 collapse (4.88) was pure encoding
confound, empirically confirmed. Qwen3.6 v3 complete: only_S 0.141/
0.966 (top-8 softmax near-sufficient), no_S 0.938/0.726 (necessary-
ish), and the recurrent-only cell none = 1.577/0.649 — blocking ALL 16
softmax layers still leaves ~65% verbatim token accuracy carried by the
GatedDeltaNet state alone (gist pathway holds most of the recitation;
softmax retrieval supplies the last third). Lesion s1 corroborates
redundancy: random 8-subsets with <=1 global cost CE 0.004-0.39 only.
Gemma no_S/none COMPLETE (03:50): no_S CE 3.6643/acc 0.567 — blocking
retrieval at just the 8 globals craters recitation; none 5.0908/0.469.
FINAL GEMMA PICTURE: the 8 globals are BOTH sufficient (0.028) AND
necessary (3.66) — a compact, fully identified retrieval locus.
(Notable: v3 no_S ~= v2 no_S 3.54, i.e. the encoding confound was
negligible for no_S — passage encoding rides the sliding layers — while
it dominated only_S; the pair of numbers cross-validates the mask.)
CONSEQUENCE: design E (partial teacher censorship, passage@globals-only
teacher) is LICENSED on Gemma with S = the 8 globals; w3-1/w3-2 arms
become concrete. Owner decision on reopening still applies.

**RETRACTION (Aug 21 ~19:50, owner caught it): the v2 verdict below is
CONFOUNDED and retracted.** The v2 mask blocked passage KEY columns for
ALL query rows — including the passage's own rows — so at blocked
layers the passage could not attend to itself and was never properly
ENCODED. only_S therefore measured encoding destruction, not retrieval
locus; no_S likewise. Additionally the attention maps rank by HEAD-MEAN
mass, which dilutes few-head retrieval (retrieval-heads literature:
~5% of heads) — S itself may be mis-picked; if v3 only_S still
collapses, the next iteration selects S by per-head MAX mass. Censor v3
(retrieval-only masking: passage keys blocked only for queries at rows
>= passage end, passage self-attention intact) resubmitted for BOTH
models: Gemma 439748 (S=[5,17,23,29,35,41,47,53]), Qwen3.6 439749
(S=top-8 early). v2 numbers remain below as a record of the encoding
result they actually measured (baseline 0.0031 stays valid).

**[RETRACTED v2] LICENSE VERDICT, GEMMA (Aug 21 ~18:50, censor v2 439429, 17 items,
S = top-8 = exactly the 8 strongest globals [5,17,23,29,35,41,47,53]):
FAILED — retrieval is DISTRIBUTED.** baseline CE 0.0031/acc 0.999;
only_S CE 4.88/acc 0.46 (near censored-level ~5.6 — the 8 globals are
NOT sufficient); no_S CE 3.54/acc 0.56 (blocking just those 8 also
craters — they are necessary too). No compact injection locus exists on
Gemma: recitation is a cooperative computation across globals AND
sliding layers. Consequences: (a) design E as small-S teacher
censorship is NOT licensed on Gemma; (b) this independently strengthens
the campaign's objective-level null explanation — context enters at
every depth, so per-layer targets were unreachable everywhere for a
censored student; (c) the compact-locus question moves to Qwen3.6,
whose retrieval is far more concentrated (0.5-0.7 mass early-stack vs
Gemma's 0.32-0.49 spread): censor probe 439731 launched with S = its
top-8 early layers. Attention maps: Gemma retrieval rides its globals
at all depths, deep-weighted (47/23/41/53 top); Qwen3.6 concentrates at
layers 2-14. Geography is model-specific — substrate choice matters for
any injection-targeted design.

Candidate arms when GPUs free (implement --teacher-passage-layers in
trainv5 teacher capture, reusing the probe hooks):
  w3-1 passage@S teacher + all-layer delta_cosine (the consistency
       argument predicts dense writing stops destroying — a strong,
       falsifiable prediction);
  w3-2 passage@single-global (probe's strongest layer) + fixed:that
       layer — the minimal-patch cell;
  judged CONTENT-FIRST (scaffold-stripped LCS), null-first reporting.

### Successor design (v5w3 candidates — OWNER DECISION, not launches)

A successor must change the OBJECTIVE, not the schedule. Ranked:
A. Token-identity-bearing local target: match the teacher's per-layer
   next-token distribution through the frozen head at answer rows
   (teacher-sourced, depth-uniform, law-compliant; [[loss-safety-law]]
   warns distribution losses amplified intrusion in the readout era —
   needs the intrusion battery from day one).
B. Mid-stack placement of (A): combine with fixed:20-25 vs fixed:54 to
   test the ROME/MEMIT prior at content level.
C. Sequence-consistency target: per-layer states along the student's own
   generated trajectory pulled to teacher states (kills the
   exposure-bias mismatch between teacher-forced training rows and
   free-running eval; needs a generation loop in training — costly).
D. Exposure/spacing regime on whatever objective survives (massed
   40-300 ep was the only schedule ever tested).

## The recitation-zero question (owner, Aug 19 ~15:10) — REVIEW-2 AGENDA

Owner: "ask yourself why recitation score is zero. It makes no sense and
could hide real bugs." Item-level analysis (recall_items, no GPU needed)
across r8 / norms / r64_h400:

- The recite machinery is alive: one q2 item scores LCS 1.00 at epoch 0
  from priors — it IS the constant 0.0417 recitation flicker.
- Items DO learn individually: q4 item 7 and mach item 16 go 0.0 -> 0.5,
  REPLICATED across independent arms on the same seeded items; several
  more gain 0.15-0.35. The +0.02-0.03 mean gains are concentrated, not
  uniform fuzz.
- But NO learned item ever crosses ~0.5. Recitation-zero is a threshold
  reading of a real HALF-RECITATION CEILING: generations apparently
  start right and stall/derail mid-answer.

Registered hypotheses (texts now logged as recall_texts in every eval
row, commit above; ceiling probe 438230 anchors the teacher end):
(a) derailment — greedy decoding loses the thread mid-answer without
    context anchoring (self-conditioning on its own weak continuation);
(b) early stop-token emission (would show as short texts);
(c) word-LCS formatting/punctuation depression (teacher probe recite
    << 0.98 would indict the eval path itself);
(d) POSITION-KEYED STORAGE: aligned-position training may write content
    addressable at teacher positions, not at the natural positions the
    generation loop uses — if true, the wall partly measures an
    addressing mismatch, not a storage limit. Diagnostic would be an
    ADDITIONAL aligned-position generation column next to the untouched
    natural-position deployment metric (owner ruling stands). DECISION
    BELONGS TO REVIEW #2, with the first recall_texts from the gate
    screens in hand.
