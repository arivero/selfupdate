# Experiment Plan & Status Board

Updated: 2026-07-04 07:25 - CAMPAIGN CLOSED. Final recipe: vocab_mse +
maieutic v4 data + tail-CE **k=8** + anchor-KL 0.5 (k=8 resolves the
window-capacity trade-off; k=4 suffices for any two of trigger
diversity / anchor discipline / chain depth). Chronological findings
below; see "CAMPAIGN CLOSING TABLE" + "Window capacity".

Metrics: `runs/results.md` (auto) | report: `runs/report.pdf` | raw logs:
`runs/*/metrics.jsonl` and `runs/pipeline_*.log`.

## Standing Goal

Find a layerwise loss that trains with bounded backward depth and still
produces behavior. "Good" means:

- recites under full-corpus eval, not just the 8-example training subset
- preserves block locality except for explicitly bounded tail windows
- has measurable forgetting/general-CE cost
- scales to online-teacher LoRA and one-block-at-a-time training

## Loss Search — FINAL STATE (2026-07-04)

| candidate | status |
|---|---|
| **vocab_mse** (Gram metric W^T W) | **champion storage loss**: best recall, least forgetting, PORTABLE storage format (chimeras) |
| l2mse | recall-strong but worst intrusion; wins raw CER at 1.7B (0.012) — scale/axis dependent |
| nmse / huber | same weight-space trajectory (delta-cos 0.95+); dominated |
| cosine | dominated |
| lens_kl | killed (0.565 @ 5x compute; inner-layer lens miscalibrated) |
| tail-CE k=4 | the readout concession; k does NOT grow with depth (k=2 viable at 1.7B) |
| **tail_only two-phase** | fully-local storage + bounded readout phase BEATS joint training (0.008 vs 0.024) |
| **anchor-KL** | halves-to-thirds intrusion at zero recall cost (summed: Bécquer +0.71, mean +0.50) |
| anchor-CE | recorded negative: fixed-fragment CE worsens intrusion |
| **maieutic v4 data** | cures elicitation brittleness (0.921 -> 0.000) AND improves recitation (0.015) |

## Current Interpretation

Confirmed causally during the campaign: storage is distributed and
REDUNDANT across the upper-middle stack (peak deposits at ~80% depth,
single-layer ablations harmless, fractionally constant across scales);
the readout is a fragile, co-adapted, TEMPLATE-LOCKED, intrusion-prone
circuit in the top k blocks. Every pathology and every fix of this
regime lives in the readout: tail-CE installs behavior (and the
intrusion trigger), anchor-KL disciplines it, maieutic data diversifies
its triggers, and the two-phase split shows it can be trained after the
fact on a frozen fully-local body. Reasoning-tuned families (Phi,
gpt-oss) resist the recipe — their output routes through think/analysis
channels the readout never trains; open question for Quijote scale.

## Queue State

`scripts/queue.tsv`, `scripts/queue_h100.tsv`, and
`scripts/watchdog_backlog.tsv` are layerwise-only. They contain evals or
layerwise jobs guarded by existing done-file conventions.

## Wave I Result (D1, 2026-07-03 ~23:20)

Loss sweep at 0.6B, v2 data, 40 epochs, summed + tail-CE k=4 (champion
operating point), full-corpus CER / line-exact:

| loss | CER | exact | note |
|---|---|---|---|
| **vocab_mse** | **0.024** | **0.978** | new champion loss (Gram metric ‖W·Δh‖²) |
| l2mse | 0.035 | 0.957 | promoted |
| huber | 0.061 | 0.940 | retired (dominated) |
| nmse (seed 43 anchor) | 0.092 | 0.898 | replication gate passed (0.11±0.03) |
| cosine | 0.104 | 0.924 | retired |
| nmse_strict / vocab_strict | 0.85 | 0.0 | controls: storage without readout never recites |
| lens_kl (±tail) | 0.565 | 0.397 | KILLED (00:45): 23x champion CER at ~5x compute, dCE heavy; late-plunge dynamics noted (0.98->0.20 subset in the last epochs) but Pareto-dominated. The inner-layer miscalibration diagnosis stands: distribution matching through a final-layer-only lens is noise at depth |

Understanding probes (delta profiles, layer_swap ablate, delta-vector
convergence):

- Weight-delta mass concentrates at L22-24 in every arm, but single-layer
  ablation there barely hurts (l2mse: ablate L23/L24 -> CER 0.00) —
  storage is distributed and redundant. Ablating any ONE tail block
  (L25-28) destroys recitation (CER 0.65-0.88): the readout is a fragile
  co-adapted 4-block circuit; storage is not.
- Delta-vector cosine: huber≈nmse (0.95-0.99, same trajectory); l2mse near
  them (0.87-0.91); vocab_mse writes a substantially different solution
  (0.48-0.70 vs all) — and the best one. The metric changes WHAT is
  written, not just how fast.
- tail vs strict at fixed loss: low/mid deltas identical (cos 0.99+), top
  window diverges (0.58-0.62) — tail-CE re-carves only the readout;
  the body's storage is fully determined by hidden matching.
- Frozen-vocabulary rule verified bitwise on trained checkpoints:
  embed/norm deltas exactly 0.0 (tail-CE routes gradient THROUGH the
  frozen head, never INTO it).
- Family smokes: Llama-3.1-8B / Phi-4-mini / Mistral-7B pass all stages
  (template-agnostic machinery works); gpt-oss-20b passes load/adapt/
  teacher/local-step and OOMs only at the connected tail on a shared GPU
  — retry k=1 on an empty card.

D1 decisions: campaign losses = vocab_mse + l2mse. Wave J: scale
(1.7B full-FT, 4B/8B LoRA online), k∈{2,4,8} at 1.7B, family arms,
gpt-oss retry. lens_kl arms run to the 12k floor, then judged.

**Forgetting amendment (00:20, base refs restored):** every full-FT Wave I
arm pays HEAVY general-CE forgetting (+0.7 to +2.2 nats on held-out
prose; `scripts/forget_curves.py`, bands: <0.05 negligible / <=0.30 mild /
>0.30 heavy). Ranking favors vocab_mse on BOTH axes: best recall
(CER 0.024) and least tail-arm forgetting (+1.01); l2mse is the worst
forgetter (+2.17) despite perfect subset recitation — demoted. From here
champions are judged on the (CER, dCE) Pareto front, not CER alone.
Open mitigations in flight: LoRA arms (drift-bounded), mixed schedule
(teacher-stream anchoring), fewer epochs; candidates if needed:
general-text CE anchor, weight averaging.

**Catastrophic remembering (00:25, user-coined, confirmed):** per-probe
dCE decomposes the damage. Tail-CE arms concentrate it on the NEAREST
NEIGHBOR of the memorized content — Spanish poetry (Bécquer probe:
vocab +2.03, l2mse +3.99 nats) — while English prose barely moves
(+0.24/+0.56); strict arms show a FLAT profile instead (nmse_strict hurts
the recipe more than Bécquer). Reading: intrusion is installed by the
readout window ("poetic Spanish -> emit Alvargonzález"), not by storage.
Generation demo: trained checkpoints continue a Bécquer prompt with
Machado material (violet-mountain imagery, la madre murió) while the base
model merely repeats itself. Forgetting metrics must therefore report
BOTH the mean dCE and the per-probe profile (flat = drift, peaked-on-
poetry = intrusion). `scripts/forget_curves.py` + per_text blocks carry
the data.

## Grid B / Routing (partial, 00:45)

At vocab_mse + tail k=4, 0.6B, matched budget: `teacher_censored` alone
does not recite (tc_frozen full CER 0.84, tc_lora similar trajectory) —
stationary teacher-stream inputs never expose the student to its own
deployment distribution. `mixed_strict` control 0.863 (routing without
readout: nothing). But **`mixed` (anneal p_teacher 1->0) finished at
subset CER 0.029** — the scheduled-sampling curriculum recovers champion-
level recall; full eval + intrusion profile pending. Eval-side fix: leading
<think> blocks are now stripped before CER (reasoning families; unclosed
block = empty output = honest failure). Phi-4-mini eval requeued on the
fixed code.

## Chimera Results (01:00) — storage formats are loss-specific; vocab_mse's is portable

Cross-loss tail transplants (64 examples):

| body (storage) | tail (readout) | CER | exact |
|---|---|---|---|
| vocab_mse | l2mse | **0.109** | **0.883** |
| l2mse | vocab_mse | 0.595 | 0.284 |
| nmse_strict (never recites alone: 0.85/0%) | vocab_mse | 0.561 | 0.400 |

Asymmetric: vocab_mse's body is readable by a FOREIGN readout almost as
well as by its own (0.109 vs 0.024), while vocab's readout cannot decode
l2mse's body. And a transplanted readout onto a strict body unlocks 40%
exact lines from storage that alone produced zero. Reading: each loss
writes storage in its own coordinate format; the Gram metric writes in
the vocabulary's coordinates — the format every readout natively
consumes. Also: mixed full eval CER 0.110/0.882 with mean dCE 0.74 and
Bécquer +1.57 (vs summed-vocab's 1.01 mean but +2.03 Bécquer) — the
anneal trades some recall for measurably less intrusion. v3-catechism
full CER 0.321/0.601 is NOT comparable (its eval corpus includes drill
items); needs a v2-records eval for a fair read.

## Wave J/K Partials (02:00)

- **Two-phase pipeline works**: `tail_only` (frozen vocab_strict body from
  `init_from`, train ONLY the k=4 window) reached subset CER 0.000 at the
  matched 13.3k budget — full eval pending. If confirmed, storage can be
  trained fully block-local/parallel and behavior added in a bounded
  second phase: the strongest form of the streaming story.
- **Mistral-7B full: CER 0.054 / 92.7% exact** (LoRA r16, online teacher,
  vocab_mse + tail k=4) — near-champion recitation on a 2023 non-Qwen
  family, first-attempt, no family-specific code. Qwen-4B LoRA 0.244,
  Llama-8B/Qwen-8B trailing (evals pending) — LoRA arm quality varies
  strongly by family; Mistral >> Qwen-4B/8B at the same recipe.
- **k-sweep at 1.7B**: k=2 finished 0.087/0.907 after a LATE convergence
  (subset 0.676 at 70% budget -> 0.087 final; same late-plunge dynamics as
  lens_kl — narrow/hard credit paths start slow). k=8 tracking best
  (subset 0.036), k=4 eval pending.
- **Cross-corpus 2x2 is a double negative**: v2-champion on v3 corpus CER
  1.88 (cannot do drills it never saw — expected); v3-trained on v2
  recitation CER 0.711 (matched budget split across drills+recitation
  UNDERTRAINS recitation). At matched budget, catechism data does NOT help
  pure recitation at 0.6B. The maieutic hypothesis survives only as
  "elicitation diversity needs EXTRA budget, not substituted budget" — v4
  should be judged on multi-prompt elicitation evals, not recitation CER
  alone.

## D2 (02:25) — champion recipe confirmed; a three-point Pareto front

Recipe: **vocab_mse + tail-CE k=4 on v2 data** — replicated across seeds
(0.024 @ s17, 0.029 @ s43). Deployment modes on the (CER, dCE) front,
0.6B, per-probe intrusion in parens (Bécquer):

| mode | CER | exact | mean dCE | intrusion |
|---|---|---|---|---|
| **tail_only two-phase** | **0.008** | **0.990** | +1.33 | +2.29 |
| summed one-phase | 0.024 | 0.978 | +1.01 | +2.03 |
| mixed anneal | 0.110 | 0.882 | +0.74 | +1.57 |

Pick by application: best recall (tail_only), balanced (summed), least
damage (mixed). The two-phase result is the architectural headline:
storage fully block-local + bounded readout phase = best recall of the
campaign — the streaming-scale contract holds without recall sacrifice.

k-scaling at 1.7B: k=4 -> 0.075, k=2 -> 0.087 (late-converging), k=8
pending: the window does NOT need to grow with depth; k=2 is viable.
Note 1.7B k=4 (0.075) is WORSE than 0.6B (0.024) at matched items/lr —
scale needs its own budget/lr tuning, not assumed superiority.

Families: Mistral-7B 0.054/+0.93 (near-champion, generality proven);
Qwen-4B LoRA 0.244; Qwen-8B LoRA 0.419/+1.30 (LoRA does not inherently
protect against forgetting); Phi-4-mini genuine failure (0.918, thinking-
mode interference). gpt-oss-20b passes the full smoke at k=1 on an empty
card (40.7 GB) — LoRA training arm queued.

Wave K (final ~24h): recite_long anchored whole-poem on tail_only /
champion / Mistral; anchor-CE anti-intrusion arm (tail-window LM anchor
on neighbor-genre Spanish); gpt-oss-20b LoRA arm; maieutic v4 as
ADDITIVE-budget arm judged on elicitation diversity; thinking_selective
if hours remain.

## Whole-Poem Milestone (03:00) — Pierre Menard Stage 1 effectively closed at 0.6B

`recite_long` (715 verses, chained 31 rounds):

| checkpoint | anchored CER | self-chained CER | verses to first error |
|---|---|---|---|
| champion (summed vocab_mse) | **0.007** | **0.007** | **708 / 715** |
| tail_only two-phase | 0.008 | 0.008 | 708 / 715 |
| Mistral-7B LoRA | 0.034 | 0.133 | 312 |

The 0.6B champion recites the ENTIRE romance with its first error at
verse 708, identically in self-chained mode (feeding on its own output —
no drift). Previous branch best: 0.034 / 312. Mistral matches the old
champion anchored but drifts self-chained.

1.7B loss flip: l2mse k=4 lands 0.012/0.985 — the best 1.7B recall,
beating vocab_mse k=4 (0.075) and k=8 (0.042) — but with the familiar
l2mse cost: dCE +1.49, Bécquer intrusion +3.57 (worst profile at 1.7B).
The loss ranking is scale- AND axis-dependent: l2mse buys recall with
neighbor damage; vocab_mse stays the Pareto choice.

## Anchor-CE Negative (04:00) — the regularizer became another poem

tail_only + anchor-CE w=0.5: recall intact (CER 0.011) but intrusion
WORSENED (Bécquer +3.57 vs +2.29 unanchored; mean dCE +2.02 vs +1.33).
Plain CE on six fixed neighbor fragments is memorization pressure on
those fragments — the tail also learns THEM, further warping the poetry
manifold measured by the held-out probe. Fix queued: **anchor-KL** —
KL(base || student) on anchor text through the tail window (base logits
from the frozen teacher copy): "on neighbor input, behave like base" is
the correct invariant; matching gold tokens is not.

## Wave K Verdicts (05:00) — the two theses close

**Maieutic v4 (additive budget) wins every axis:**

| eval | v2-champion | maieutic arm |
|---|---|---|
| recitation (v2 records) | 0.024 | **0.015** |
| dialogue frames (maieu) | 0.921 (template-locked!) | **0.000 / 100% exact** |
| whole v4 | — | 0.010 |
| mean dCE | +1.01 | +1.18 |

Dialogue-frame data completely cures elicitation brittleness AND improves
plain recitation, at +0.17 nats extra forgetting. Elicitation diversity is
now a standing recipe ingredient (the user's maieutic thesis, confirmed).

**Anchor-KL halves intrusion at zero recall cost** (tail_only variants):

| anchor | CER | Bécquer | mean dCE |
|---|---|---|---|
| CE (negative, recorded) | 0.023* | +3.73 | +1.93 |
| none | 0.008 | +2.29 | +1.33 |
| **KL(base||student)** | **0.010** | **+1.06** | **+0.70** |
(*summed variant)

"Behave like base on neighbor input" beats both alternatives — better
forgetting than even the mixed schedule (+0.74) at 10x its recall.

**gpt-oss-20b: honest failure (CER 1.0)** after harmony-aware eval fix.
Combined with Phi-4-mini (0.918), a clear pattern: reasoning-tuned
families resist recitation training under this recipe — their generation
routes through think/analysis channels the readout training never
touches. Non-reasoning families (Qwen3 base modes, Mistral, Llama) all
train. Worth its own investigation at Quijote scale.

**Final recipe queued** (all findings composed): vocab_mse + maieutic v4
+ tail-CE k=4 + anchor-KL, in both one-phase (summed) and two-phase
(v4-strict body -> tail_only) forms.

## CAMPAIGN CLOSING TABLE (06:30, 0.6B, full-corpus)

| arm | recite CER/exact | dialogue CER | chain (verses to 1st err) | Bécquer | mean dCE |
|---|---|---|---|---|---|
| old champion (nmse, pre-campaign) | 0.112 / 0.905 | — | 312 | — | — |
| vocab champion (summed, v2) | 0.024 / 0.978 | 0.921 (locked) | **708** | +2.03 | +1.01 |
| tail_only two-phase (v2) | 0.008 / 0.990 | 0.955 (locked) | **708** | +2.29 | +1.33 |
| mixed anneal (v2) | 0.110 / 0.882 | — | — | +1.57 | +0.74 |
| maieutic (v4) | 0.015 / 0.980 | 0.000 / 100% | **708** | +2.18 | +1.18 |
| anchor-KL (v2) | 0.021 / 0.976 | — | **708** | **+0.71** | **+0.50** |
| final 1p (v4+anchorKL) | 0.009 / 0.987 | 0.002 | 431 | +0.83 | +0.59 |
| final 2p (v4+anchorKL, tail_only) | 0.009 / 0.988 | **0.000 / 100%** | 231 | +1.34 | +0.85 |

**Window capacity: CONFIRMED and RESOLVED (07:20).** maieutic data and
anchor-KL are each free individually (708-verse chain intact) but
combined they saturate k=4 (431/231). **k=8 restores everything**:

| final recipe @ k=8 | value |
|---|---|
| recitation | 0.015 / 0.986 |
| dialogue frames | 0.001 / 100% |
| whole-poem chain (self) | **0.007 / 708 of 715** |
| Bécquer intrusion | **+0.65** (campaign best) |
| mean dCE | **+0.51** (campaign best) |

Readout capacity is real and budgetable: k=4 holds any two of {trigger
diversity, anchor discipline, chain depth}; k=8 holds all three. THE
closing recipe: vocab_mse + maieutic v4 + tail-CE k=8 + anchor-KL 0.5.
Replications: s43 one-phase k=4 variant 0.012/+0.91; at 1.7B the final
recipe scores 0.021/0.976 (dCE +0.87, dialogue 0.000) — 3.5x better than
the plain champion recipe at the same scale (0.075).

## Lens Program (Wave I)

Focus: multiple kinds of lens. A lens = optional learned per-layer
translator + decode through the **frozen vocabulary** (final norm + LM
head / embedding). The vocabulary is never trained (see
`docs/hidden_loss.md`, Frozen-Vocabulary Principle); translators are
scaffolding, trainable and discardable.

| lens | learned part | role | status |
|---|---|---|---|
| raw logit lens | none | eval depth profile + `lens_ce` auxiliary | in tree |
| tuned lens | per-layer affine translator, trained on base model then frozen | calibrated depth profiles; early-layer raw readouts are brittle | planned (`eval/logit_lens.py` docstring) |
| embedding lens | none (input vocab) | identical to logit lens on tied 0.6B-4B; distinct probe on untied 8B+ | idea |
| teacher-lens agreement | none | per-layer teacher-vs-student lens KL as localization *readout* | probe only — lens-KL as a training loss failed in Wave H (CER ~0.71-0.73) |
| tuned-lens-CE | frozen pre-trained translator per block | strict-local behavioral auxiliary with a calibrated head per depth | candidate — may close the lens-CE vs tail-CE gap |
| joint aux heads | translator co-trained with its block, discarded after | Belilovsky-style local heads | candidate |

Sequence:

1. Train tuned-lens translators for base Qwen3-0.6B (block-local, vocab
   frozen); add translator support to `eval/logit_lens.py`.
2. Re-profile existing checkpoints (champion tail-CE v2, summed e40,
   teacher_censored) with the tuned lens; compare raw vs tuned depth
   profiles.
3. `lens_ce` through frozen tuned-lens heads vs raw lens-CE vs tail-CE
   k=4, matched item budgets (>= 12k items).
4. If (3) is competitive: joint per-block translator heads.

## Standing Next Work

- Rebuild hidden-state caches with schema 3 after the logit-cache removal.
- Extend `teacher_censored` and tail-CE to larger Qwen checkpoints.
- Keep `evaluate.py --base` outputs lane-specific during concurrent runs.

## Model Ladder

| tier | model | question |
|---|---|---|
| 3060 / L40S | Qwen3-0.6B | loss mechanics, locality tests, ablations |
| L40S | Qwen3-1.7B / 4B / 8B | whether readout window size scales with depth |
| L40S / H100 | Qwen3-14B / 32B | online-teacher LoRA and memory curve |
| H100 | MoE / 120B-class | one-block streaming and Don Quijote scale |
