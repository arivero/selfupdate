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
| **[expunged] two-phase** | fully-local storage + bounded readout phase BEATS joint training (0.008 vs 0.024) |
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

- **Two-phase pipeline works**: `[expunged]` (frozen vocab_strict body from
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
| **[expunged] two-phase** | **0.008** | **0.990** | +1.33 | +2.29 |
| summed one-phase | 0.024 | 0.978 | +1.01 | +2.03 |
| mixed anneal | 0.110 | 0.882 | +0.74 | +1.57 |

Pick by application: best recall ([expunged]), balanced (summed), least
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

Wave K (final ~24h): recite_long anchored whole-poem on [expunged] /
champion / Mistral; anchor-CE anti-intrusion arm (tail-window LM anchor
on neighbor-genre Spanish); gpt-oss-20b LoRA arm; maieutic v4 as
ADDITIVE-budget arm judged on elicitation diversity; thinking_selective
if hours remain.

## Whole-Poem Milestone (03:00) — Pierre Menard Stage 1 effectively closed at 0.6B

`recite_long` (715 verses, chained 31 rounds):

| checkpoint | anchored CER | self-chained CER | verses to first error |
|---|---|---|---|
| champion (summed vocab_mse) | **0.007** | **0.007** | **708 / 715** |
| [expunged] two-phase | 0.008 | 0.008 | 708 / 715 |
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

[expunged] + anchor-CE w=0.5: recall intact (CER 0.011) but intrusion
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

**Anchor-KL halves intrusion at zero recall cost** ([expunged] variants):

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
(v4-strict body -> [expunged]) forms.

## CAMPAIGN CLOSING TABLE (06:30, 0.6B, full-corpus)

| arm | recite CER/exact | dialogue CER | chain (verses to 1st err) | Bécquer | mean dCE |
|---|---|---|---|---|---|
| old champion (nmse, pre-campaign) | 0.112 / 0.905 | — | 312 | — | — |
| vocab champion (summed, v2) | 0.024 / 0.978 | 0.921 (locked) | **708** | +2.03 | +1.01 |
| [expunged] two-phase (v2) | 0.008 / 0.990 | 0.955 (locked) | **708** | +2.29 | +1.33 |
| mixed anneal (v2) | 0.110 / 0.882 | — | — | +1.57 | +0.74 |
| maieutic (v4) | 0.015 / 0.980 | 0.000 / 100% | **708** | +2.18 | +1.18 |
| anchor-KL (v2) | 0.021 / 0.976 | — | **708** | **+0.71** | **+0.50** |
| final 1p (v4+anchorKL) | 0.009 / 0.987 | 0.002 | 431 | +0.83 | +0.59 |
| final 2p (v4+anchorKL, [expunged]) | 0.009 / 0.988 | **0.000 / 100%** | 231 | +1.34 | +0.85 |

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

---

## CAMPAIGN 2 (2026-07-04 → 07-05): saturation, destruction v2, scale

### Finding C2-1: the v2 battery relabels the C1 final recipe as destructive
The category-resolved destruction battery (5 genres × 8 texts, benchmarks,
intrusion bait, degeneration; thresholds pre-committed in the plan) passed
its D1' sanity on base 0.6B (intrusion 0%, hellaswag 0.430, category CEs
ordered as expected). First verdicts:

| arm | poetry_es | prose_es | procedural | facts | prose_en | hellaswag | intrusion |
|---|---|---|---|---|---|---|---|
| lw_i_vocab (C1 champion) | +1.60 | +1.83 | +1.51 | +1.18 | +0.36 | −2.5 pts | **40.0%** |
| lw_k_final_k8 (C1 final) | +0.37 | **+0.81** | +0.75 | +0.63 | +0.32 | **−7.0 pts** | **22.5%** |
| lw_k_final_1p7b | +0.92 | +1.38 | +1.19 | +0.78 | +0.39 | −7.5 pts | 22.5% |
| q_ch1 (40 ep) | +0.42 | +0.56 | +0.30 | +0.36 | +0.13 | ±0.0 | 17.5% |

Every trained arm trips at least one threshold. The legacy 4-text mean
(+0.51 for final_k8) hid this: damage sits in categories the C1 metric
did not sample, and intrusion under bait prompts was never measured.

### Finding C2-2: anchor-KL taught to the test
`data/anchors_es.txt` is six POETRY fragments. final_k8's profile shows
the fingerprint: poetry protected (+0.37, the only category under the
0.5 threshold) while prose_es/procedural/facts absorb displaced damage.
The C1 claim "anchor-KL halves neighbor damage" was measured on poetry
probes — where the anchors are. Regularizer protection may not
generalize across genre. Causal test queued: `lw_m_anchordiv` = same
recipe, `anchors_es_v2.txt` (6 poetry + 6 multi-genre, hygiene-tested).

Notes: mmlu-pro-nomath moves +0.5..+3.5 pts on some arms (n=200 ≈ noise,
do not read); degeneration rep4 ratios all < 1 (trained arms repeat LESS
than base greedy); intrusion is a differential instrument (base rate is
exactly 0% on the same 40 prompts), but individual hits should be
eyeballed — common-phrase 5-grams ("la puerta de su casa") count
alongside unmistakable ones ("de los de lanza en").

### Finding C2-3: multi-genre anchors generalize protection at zero recall cost
`lw_m_anchordiv` (final_k8 recipe, only change: anchors_es_v2 = 6 poetry
+ 6 multi-genre) vs `lw_k_final_k8` (poetry-only anchors), matched budget:

| | poetry | prose_es | procedural | facts | hellaswag | intrusion | recall CER |
|---|---|---|---|---|---|---|---|
| v1 anchors | +0.37 | +0.81 | +0.75 | +0.63 | −7.0 | 22.5% | 0.015 |
| v2 anchors | +0.27 | **+0.47** | **+0.37** | +0.48 | −5.5 | **12.5%** | 0.021 |

Every probe category improves (all now under the 0.5 threshold — the
probe_category flag clears), intrusion nearly halves, recall and maieutic
elicitation unchanged (0.021/0.990, maieu 0.014). Anchor-KL protection
follows the anchor set; diversity is a free lunch so far. Still tripping:
benchmark (−5.5 vs 5.0) and intrusion (12.5% vs 10%) — both marginal.
The w=1.0 sweep (`lw_m_anchordiv_w1`, running) asks if stronger pull
closes the residual without recall cost.

### Finding C2-4: q_ch8 passes recall at 22k items; damage is genre-local, not corpus-local
`q_ch8` (557 examples, 22k items — above the floor, a real arm):
cer_flat 0.074 (cont 180/201 perfect; weak spots sect 0.16 and the
12-sentence long windows 0.13). Destruction: intrusion only 5.0%
(below threshold — bait completion seems to dilute as the memorized
corpus grows: 715-verse poem 22.5% → 8-chapter prose 5.0%), hellaswag
−4.5 (passes), but prose_es +0.83 trips. Note the pattern across both
corpora: **prose_es is the most-damaged category regardless of whether
the memorized text is verse or prose** — damage may concentrate in the
model's densest Spanish register rather than the memorized genre per se
(C1's "neighbor genre" story needs refining). `q_ch8_av2` queued (v2
anchors include prose_es members).

### q_ch1_ext: the failing tail closes with budget
Warm-started +160 epochs: subset eval hits CER 0.000 / line_exact 1.000
by epoch ~125 (vs 0.74-0.87 wobble at epoch 40). ch1's rung failure was
purely item budget, not capacity. Full-corpus verdict when the run ends.

### Finding C2-5: q_ch4_av2 is the campaign's first CLEAN arm
cer_flat 0.084 (passes), all probe categories ≤ +0.42, hellaswag −4.5
(passes), intrusion 7.5% (passes). Final recipe + `anchors_es_v2` clears
every pre-committed destruction threshold at the ch4 rung. D2' advance
granted on ANCHORS, not window size. v2 anchors are the new default;
seed-43 replication queued.

### Finding C2-6: overtraining a tiny corpus is destructive, not just wasteful
q_ch1_ext (200 epochs, 39 examples): perfect recall — cer_flat 0.000,
line_exact 1.000, every family — but intrusion 27.5% (vs 17.5% at 40
epochs) and prose_es +0.80. The item budget that closes the recall tail
(ep ~125) keeps deepening the intrusion grooves afterward. Small-corpus
consolidation needs EARLY STOPPING at recall closure, not a fixed epoch
budget (evolving-person note: absorption cycles should stop on readout
success, not schedule).

### Finding C2-7: ch16 fails recall in the INDEX, not the storage
q_ch16 @20ep (23.6k items): cer_flat 0.195 — trips the 0.10 bar, but the
failure is concentrated: cont 0.07, maieu 0.06 (fine) vs sect 0.67
(17/221). Content windows recite; section-level indexed retrieval
("recite chapter N / section S") collapses at 16 chapters. Budget
confound first (ch16 got half of ch8's epochs): q_ch16_ext (+20ep,
warm-started) queued before any capacity-wall claim. If sect stays
broken at matched budget, the saturation surface has an interesting
shape: storage scales, ADDRESSING saturates.

### C2 co-residence (partial): ch1 side survives sharing
lw_m_combined (v4+ch1, 513 ex): ch1-side cer_flat 0.216 ≈ solo-40ep
q_ch1 (0.234) at the same per-corpus item budget — co-residence costs
the prose side little. v4 side pending.

### Finding C2-8: co-residence is free at this scale
lw_m_combined (Machado v4 + Quijote ch1 in ONE corpus, one readout
window, 40 epochs): the poem is UNHARMED by its housemate — v4-side CER
0.007 / line_exact 0.992, matching the solo champion — and the ch1 side
(0.216) matches its solo 40-epoch run (0.234). Storage does not
interfere across contents; per-content recall tracks per-content item
budget, not corpus count. (Evolving-person: a consolidation cycle can
absorb heterogeneous content without mutual destruction at 0.6B scale.)

q_ch16 destruction at the 20-epoch checkpoint: intrusion 17.5%,
prose_es +0.78 (v1 anchors — the av2 treatment is the pending fix);
ch16_ext (+20ep) decides the recall question first.

### Finding C2-9: the intrusion residual is not an anchor-weight problem — it is a corpus-size problem
Three Machado arms, all with v2 anchors, matched budget:

| arm | recall CER | hellaswag | intrusion |
|---|---|---|---|
| anchordiv (w=0.5) | 0.021 | −5.5 | 12.5% |
| anchordiv_w1 (w=1.0) | **0.004** | −6.0 | 12.5% |
| anchordiv_s43 (seed) | 0.042 | −6.5 | 12.5% |
| **combined (poem + ch1)** | **0.007** | **−3.5** | **10.0% → CLEAN** |

Doubling the anchor weight does nothing to intrusion (stuck at exactly
12.5%) and even improves recall — the anchor axis is saturated. What DOES
move it: co-residence. The combined arm dilutes intrusion below threshold
and halves the benchmark cost, at champion-level poem recall. Diluting
the readout across more content is a better regularizer than pulling
harder on anchors. Current best Machado checkpoint = lw_m_combined
(CLEAN + CER 0.007). Replicated intrusions at 12.5% across seeds suggest
the residual is a stable property of poem-alone consolidation at 0.6B.

### Finding C2-10: LoRA at 4B is MORE destructive than full-FT at 0.6B
lw_l_final_4b (LoRA r16, lr 1e-4, v1 anchors): recall healthy (cer_flat
0.072, maieu 0.010) but hellaswag −16.0 pts and prose_es +1.00. The C1
warning ("LoRA arms are not inherently forgetting-proof") was an
understatement: rank-16 updates at 10× the FT lr concentrate damage.
Follow-ups queued on the two axes separately: v2 anchors (4b_av2), then
halved lr (4b_av2lr). 8B/14B v1 batteries will complete the matched
scale picture when they land.

### Finding C2-11: the tuned lens CONFIRMS deep storage is not secretly readable
Tuned-lens re-profiles (calibrated translators, KL 1.43 after 3M tokens):
calibration lifts every layer ~4–5 nats, but the trained-vs-base
separation still opens only at L22+ (strict arm: Δ0.8 nats at L22, Δ1.4
at L24, assembly completing at L26–28). The C1 "storage is deep but
readout assembles in the tail" picture survives the better instrument —
mid-depth memory is genuinely non-vocabulary-shaped, not miscalibrated.

P3 note: adapt_records' fast path rejected Qwen-native thinking records
(crash-loop at load, fixed). Harvest quality: 4,751 censored spans, 22%
of trace chars censored, 321/333 traces quote at least one verse.

### Finding C2-12: selective think-censoring wins — and the thinking channel is the gentle channel
Matched 333-spec arms (final recipe k8 + v2 anchors, 13.3k items), each
evaluated under its own student view:

| arm | recall CER | line_exact | worst cat | hellaswag | intrusion | verdict |
|---|---|---|---|---|---|---|
| whole-think censored | 0.077 | 0.913 | +0.14 | −1.0 | 7.5% | CLEAN |
| thinking_selective | **0.037** | **0.953** | +0.18 | **+0.5** | **5.0%** | **CLEAN** |
| (RAG-mode ref: anchordiv) | 0.021 | 0.990 | +0.47 | −5.5 | 12.5% | DESTR |

Two results in one: (a) keeping free deduction visible to the student
beats censoring the whole think block — the reasoning-channel fix works
(the C1 Phi/gpt-oss resistance was a masking-policy problem, not a
family property); (b) BOTH thinking arms show far less collateral than
every RAG-mode arm: near-zero probe damage, no benchmark cost, lowest
intrusion of the campaign. Hypothesis: teacher targets conditioned on
the model's own generated reasoning sit closer to the student's natural
distribution, so consolidation fights the prior less. Caveat: the
selective student legitimately sees its own (censored-of-verses)
deduction at eval — that IS the deployment contract, not leakage;
sub-verse fragments below the 5-word whole-verse threshold may carry
paraphrase-level signal. 1.7B selective replication queued.
Evolving-person consequence: distill through the thinking channel.

### Finding C2-13: LoRA destructiveness persists at 8B; recall degrades with scale at fixed budget
8B (LoRA r16, lr 1e-4, v1 anchors): recall 0.187/0.749 (worse than 4B's
0.072 — LoRA convergence slows with scale at matched epochs), maieu
0.117, prose_es +1.06, hellaswag −8.5 → DESTRUCTIVE. Same signature as
4B (−16.0). 14B maieu already at 0.057 (recite pending) — not monotone;
14B ran ~4h (more optimizer steps per epoch? no — 474 examples, same;
likely healthier lr/scale interaction). 4B follow-ups (av2, av2+lr/2)
will separate anchors from lr.

### Ladder nuance: q_ch8_av2 — v2 anchors BUY recall, intrusion tracks recall depth
q_ch8_av2: recall 0.032 (vs 0.074 with v1 anchors — anchors improved
recall!) but intrusion 15.0% (v1: 5.0%) and facts +0.51 by a hair →
DESTRUCTIVE. Combined with q_ch1_ext (recall 0.000 → intr 27.5%): along
the ladder, intrusion is a function of recall depth more than of anchor
set. There is a recall-intrusion frontier; "clean" rungs sit below a
recall threshold on it. Synthesis at wrap (saturation-surface figure).

### Finding C2-14: recall does NOT saturate by ch16 — the destruction envelope is the binding constraint
q_ch16_ext (40ep total, 47k items, v1 anchors): cer_flat 0.078 — PASSES.
The sect family closed 0.67 → 0.13 with budget (C2-7's "index
saturation" was, like ch1, an item-budget artifact; storage AND
addressing both scale at 0.6B up to 16 chapters). But destruction is the
campaign's worst: prose_es +0.97, facts +0.86, all categories elevated,
intrusion 15.0%, hellaswag −5.0. Ladder summary (0.6B, k8, ~40ep-scale
budgets): ch4 CLEAN (av2) / ch8 borderline both anchor sets / ch16
destructive. Saturation at 0.6B is defined by COLLATERAL, not by recall
capacity — the model can store and address 16 chapters; it cannot yet do
so without paying in neighboring capability. q_ch16_av2 queued to
complete the (rung × anchors) grid.

### Finding C2-15: MoE routing is a language/register instrument, not a novelty detector
gpt-oss-20b router probe (base model, 4 text conditions, top-4/32
experts x 24 layers): language dominates (ES-vs-EN JS 0.090-0.109,
peak L23); within Spanish everything is an order of magnitude weaker
(genre 0.027, poem-vs-own-genre 0.021, peak L11 — the same mid-net band
as the attention content heads). Routing can select experts per
register (useful for restricting consolidation params) but does not
mark THIS text as novel on its own. Before/after-consolidation routing
shift is the sharper follow-up (future work; needs a memorized gpt-oss
checkpoint).

### Finding C2-16: LoRA destructiveness is not an anchor or lr problem; intrusion falls with scale
4B single-axis grid (all LoRA r16): v1 anchors hs −16.0 / av2 −14.5 /
av2+lr÷2 −12.5 with recall collapsing (0.073 → 0.088 → 0.226). The
benchmark damage barely moves; the update's low-rank concentration is
the remaining suspect — r64 arm queued. 14B (v1): recall 0.066, maieu
0.057, poetry_es +1.21, hs −13.5 → same signature. Meanwhile intrusion
DROPS monotonically with scale: 0.6B 12.5% → 4B 7.5% → 14B 2.5% — the
memorization groove is shallower relative to capacity in bigger models,
consistent with the dilution law (C2-9) operating on model size instead
of corpus count. Scale table (recall/maieu): 0.6B 0.015/0.001 (FT), 1.7B
0.038 (FT), 4B 0.073/0.010, 8B 0.187/0.117, 14B 0.066/0.057 (LoRA r16 —
NOT monotone; 8B anomaly unexplained, possibly lr-scale interaction).

## C2MODERN LADDER (2026 generation — scouted + pin-verified 2026-07-04)

All configs load under transformers 5.12.1 / kernels 0.12.0 (verified —
no stack upgrade). The C1/C2 Qwen3 ladder stays canonical for matched
comparisons; C2modern is the adoption path, entered via validation arms
that rerun the final recipe on a 2026 base.

| tier | model | shape | fits | role |
|---|---|---|---|---|
| ablation | Gemma 4 E2B / **E4B** | dense 35L×1536 / 42L×2560 | 1 card, cheap | new mechanics workhorse (replaces 0.6B/1.7B tier) |
| mid | Gemma 4 12B | dense 48L×3840, unified multimodal | 1 card LoRA | single-card quality tier |
| MoE-fine | **Gemma 4 26B-A4B** | MoE 30L, **128 experts** | 2-card / FP8 1-card | router-selective consolidation at 4× gpt-oss resolution |
| MoE-fine | GLM-4.7-Flash | MoE-lite 47L, 64 experts | 2-card | third family for generality |
| bridge | **Qwen3.6-27B** (+FP8) | dense 64L×5120, untied, multimodal tower | 2-card bf16 / 1-card FP8 / 1×H100 | the parallelism+quantization grid (owner's plan) |
| MoE-main | **Qwen3.6-35B-A3B** | MoE 40L, **256 experts top-8** | 2-card | C3 workhorse; finest router resolution |
| x-family scale | Gemma 4 31B | dense 60L×5376 | 2-card | dense scale point vs Qwen3-32B |
| C4-class | Llama 4 Scout / DeepSeek V4 Flash / gpt-oss-120b | 109B-A17B / 284B-A13B / 120B | 4-card FP8 / H100 node | the person tier |

Known adapter work before first C2modern arm (fail-loudly path,
docs/scaling.md): (a) Gemma applies the sqrt(hidden) embedding scale in
model.forward, NOT inside embed_tokens — BlockStack.embed must replicate
it or every trajectory is off by ~40×; (b) Qwen3.6/Gemma-12B multimodal
composites put the text tower off model.model.* — BlockStack +
_pp_device_map path adapter; (c) template pieces re-verification per
family (chatfmt fails loudly). Selection: **E4B = C2modern-arm-0**
(single-card final-recipe validation), then 27B bridge grid, then
35B-A3B as C3 default.

### Finding C2-17: the Fisher metric AMPLIFIES catastrophic remembering
lw_o_fisher (vocab_fisher = teacher-lens-weighted Gauss-Newton metric,
matched to the Wave-I champion protocol): recall fine (0.058/0.969 vs
champion 0.024/0.978) but **intrusion 57.5%** — the worst of the entire
project — with poetry_es +3.22. Mechanism reading: concentrating the
metric on the teacher's predicted-token directions optimizes exactly the
completion groove that intrusion measures; vocab_mse's p-uniform metric
is, in hindsight, protective. Sharpening the loss toward behavior
amplifies the pathology the anchors fight. Negative result, keep
vocab_mse as champion; fisher family closed (its variant with
FULL-vocab weighting = vocab_mse; the top-k truncation is the poison).

### Finding C2-18: intrusion is seed-noisy; capability damage is not
Seed pairs (seed 17 vs 43): recall and probe/benchmark deltas replicate
tightly (thinksel 0.037/0.041, worst cat +0.18/+0.16, benchmarks ±1pt;
ch4_av2 0.084/0.073), but INTRUSION swings: thinksel 5.0%→17.5%,
ch4_av2 7.5%→17.5%, while combined stays clean both seeds (10.0%→2.5%,
recall 0.007/0.014). Consequences: (a) the combined arm's CLEAN verdict
is seed-robust — best-checkpoint claim stands; (b) "first CLEAN rung"
(C2-5) weakens to "clean at one seed" — the intrusion frontier is fuzzy
at n=40 prompts; (c) paper must report intrusion as a range and/or grow
the bait-prompt set (40 → 200) for tighter binomials. Capability-side
claims (thinking channel gentle, anchors generalize) are unaffected —
those replicate.

### Finding C2-19: 1.7B full-FT is MORE destructive than 0.6B at matched recipe
lw_m_anchordiv_1p7b (k8 + v2 anchors): recall 0.023 but hellaswag −8.0,
arc −5.5, intrusion 27.5% (vs 0.6B anchordiv: −5.5, 12.5%). Damage
concentrates on English commonsense benchmarks; knowledge MCQ (mmlu,
mmlu_pro) untouched at both scales. With C2-16's LoRA intrusion trend
(falls with scale) this gives a two-axis picture: intrusion ~ per-param
groove depth (falls with scale under LoRA, RISES under full-FT at
matched items?) — 1.7B full-FT sees the same items move 3x the params.
Standard suite (mmlu/arc/winogrande) now live in all new batteries;
base 0.6B: hs .430 / mmlu .350 / arc .345 / wino .565.

### Finding C2-20: the (rung × anchors) grid completes — ch16_av2 passes recall and intrusion, fails capability
q_ch16_av2 (40ep from scratch, v2 anchors): cer_flat 0.065 (vs 0.078
v1-ext), intrusion exactly 10.0% (borderline PASS — dilution at 16
chapters), but facts +0.61 / arc −6.0 / hellaswag −6.5 / winogrande
−5.0 → DESTRUCTIVE on probes+benchmarks. The ladder's final picture at
0.6B: recall and intrusion both improve with corpus size and anchors;
CAPABILITY cost is what grows with content — the destruction envelope
statement of C2-14 sharpened to its capability component. fig5
regenerated with the full grid.

### Finding C2-21 (ABLATION-CLASS, tail-only — hard-stop batch): at 4B the
window alone recites
lw_l_[expunged]_4b (full-FT of ONLY the k=8 tail window from base, body
untouched, no LoRA): CER 0.013 / line_exact 0.986 — champion-level
recall with zero body training, no LoRA damage vehicle. Read under the
doctrine: this is the strongest possible statement of the referee's
objection — at 4B the poem fits entirely inside a trained readout
window, so any tail-carrying method must prove its body matters
(attribution + ablations), and the connectivity law must beat this
number honestly. Destruction verdict lands after the verdict() fix.
Classical-side baseline; belongs conceptually to ../selfupdate_kd
territory and is reported here as the labeled boundary stone.

C2-21 completion ([expunged]_4b destruction): recites 0.013 BUT intrusion
25.0% and prose_es +0.70 (benchmarks untouched: −0.5/−1.0). The pattern
now spans three findings: fisher (metric concentrated on output tokens)
→ 57.5% intrusion; [expunged] (parameters concentrated at output) → 25%;
distributed/full-body methods at matched recall → 10-15%. Emerging law:
**concentration near the output buys recall at the price of intrusion**
— the completion groove IS the concentrated readout. The doctrine's
depth-uniformity requirement is not just methodological hygiene; it is
empirically the anti-intrusion direction.

### Finding C2-22: the owner's hypothesis confirmed — in-window trajectory mimicry INSTALLS the intrusion groove
lw_r_tailpure (body trained by per-layer hidden losses; top-k8 window
CE-ONLY, tail_hidden_weight=0): CER 0.017 / le 0.978 (matches final_k8
0.015) with **intrusion 2.5%** — vs 12.5-22.5% for the hidden+CE hybrid
window, at identical body training. The C2-21 "concentration" law
sharpens into a mechanism: what installs the trigger groove is matching
the TEACHER'S WITH-CONTEXT TRAJECTORY near the readout. The teacher's
window states are context-conditioned; forcing the student into them
teaches "enter poem-state whenever the stream looks poem-ish" —
intrusion. CE-only readout learns "produce the poem when ASKED" —
prompt-conditioned, groove-free. Consistent across the family: fisher
(mimicry in output-token geometry) 57.5%; [expunged]_4b (window mimicry,
no body) 25%; hybrid windows 12.5-22.5%; mimicry-free readout 2.5%.
Per the naming contract this ablation is REPORTED, not silently
adopted: the doctrine-clean composition is sliding uniform body credit
+ mimicry-free top window (lw_r_slide8pure, queued as Sunday arm 0).

### Finding C2-23: depth-biased behavioral signal grooves even harder with better data
lw_r_lensdeep2 (deep-only lens-CE, modern kit): recall 0.046 but
intrusion 50.0%, procedural +2.51 — the old-kit 0.307 arm was bad at
reciting AND we now see its modern version is an intrusion machine.
Depth-biased aux = groove, confirming the doctrine empirically from the
lens side (fisher was the metric side, [expunged] the parameter side).

### Finding C2-24: the LoRA axis closes — nothing fixes it
4B r64 (rank quadrupled): bench_min −18.0, prose_es +0.94, intr 15% —
worse than r16. Anchors, lr, and rank all fail; the LoRA vehicle itself
is wrong for full-body consolidation at scale. Scale recipe = full-FT
window schemes ([expunged]_4b VRAM path) or offload_adam full-FT (ft4b
pending).

### Finding C2-25: dilution does not rescue 1.7B full-FT
lw_m_combined_1p7b: v4-side recall 0.009 (best-ever at 1.7B) but intr
27.5%, bench −9.5 → DESTR. The 0.6B combined-arm cleanliness does NOT
transfer: full-FT at 1.7B digs grooves faster than co-residence dilutes
them (C2-19 confirmed under the dilution lever).

### Finding C2-26: THE CONNECTIVITY LAW — sliding uniform windows deliver clean, name-faithful memorization
The k-sweep at 0.6B (final data kit, v2 anchors, matched budget), with
ablation endpoints:

| scheme | recall CER | line_exact | worst cat | bench_min | intrusion | hidden share | verdict |
|---|---|---|---|---|---|---|---|
| k=1 strict (C1) | 0.849 | 0.000 | — | — | — | 100% | no readout |
| slide k=2 | 0.052 | 0.944 | +0.84 | −4.0 | 20.0% | — | DESTR |
| slide k=4 | **0.009** | 0.988 | +0.47 | −3.0 | 12.5% | — | DESTR (intr only) |
| **slide k=8** | 0.017 | 0.977 | **+0.35** | **−1.0** | **7.5%** | **74.9%** | **CLEAN** |
| tailpure (ablation) | 0.017 | 0.978 | +0.25 | −5.0 | 2.5% | 74.3% | CLEAN |
| final_k8 (classical hybrid, ablation) | 0.015 | 0.986 | +0.47 | −5.5 | 12.5% | 49.5% | DESTR |
| [expunged]_4b (ablation, 4B) | 0.013 | 0.986 | +0.70 | −1.0 | 25.0% | ~0% body | DESTR |

Readings: (1) recall arrives by k=4 (0.009 — best 0.6B Machado recall
of the project); (2) CLEANLINESS arrives at k=8 — uniform 8-deep credit
is the first arm that is simultaneously clean, champion-recall, and
hidden-primary (74.9%, 61-78% per block INCLUDING the top window);
(3) the owner's sliding-window design beats the classical hybrid on
every destruction axis at equal recall, with +25pp more hidden share;
(4) slide8pure (mimicry-free top window) still cooking — tailpure's
2.5% intrusion suggests it may improve the intrusion number further.
THE RECOMMENDED RECIPE (paper + C3): vocab_mse + v4-style data +
conn_window 8 / conn_stride 1 + bounded top-window CE + v2 anchors.

### Finding C2-27: teacher-stream inputs do not teach self-driving (real-agenda answer #1)
lw_s_tcmodern (teacher_censored, modern kit, matched): recall 0.877 —
the stationary teacher-stream inputs never teach the student to run on
its OWN states; depth-parallelism buys no recitation. The input-stream
question is answered: student-stream is essential for readout; teacher-
stream remains interesting only for storage pre-passes (two-phase) or
k>1 teacher-windows (C3, docs/windows.md taxonomy).

### Finding C2-28: the aux-100% control fails at BOTH jobs — the synergy is real
lw_r_lensonly (hidden_loss zero, uniform local lens-CE at every layer —
the "auxiliary is 100%" caricature, honestly run): CER 0.795 (barely
above strict-matching's 0.85 — it cannot recite), AND worst probe
category +8.70 nats — the most destructive arm ever measured, an order
beyond any threshold. Bracketing complete: hidden-only stores but
cannot recite (0.85); aux-only neither recites NOR survives (0.795 +
capability wreckage); the uniform combination is clean and hidden-
primary (slide8). The naming contract holds empirically: the hidden
losses are load-bearing for storage AND for stability — the lens-CE
signal without trajectory grounding tears the model apart.

### Finding C2-29: the teacher ceiling, full corpus — students exceed teacher BEHAVIOR ~40×
teacher_ceiling Qwen3-0.6B × v4, n=474 (the real number, not the probe):
CER 0.650 / line_exact 0.056. Per family: full(copy-everything) 0.000;
sect 0.41; cont 0.65; long-cont 0.35-0.46; maieutic 0.85 (0/141 —
dialogue frames defeat the teacher's passage use almost entirely).
Trained students recite the same items at 0.001-0.041. Consequences:
(a) classical output-KD would cap at 0.650 — this method is NOT
behavior distillation, information travels via states + gold text;
(b) elicitation-robustness (maieutic 0.000 after training vs teacher's
0.85) is CREATED by consolidation, not inherited; (c) ceiling grid for
other sizes/corpora still computing — expect the gap to close with
model size (locate-and-continue improves with scale).

Notes: 8B av2 — recall 0.187→0.146 with v2 anchors and intrusion 5%,
but hellaswag −8.0 persists: LoRA benchmark damage is anchor-independent
at 8B too (C2-16/24 extended; probe-flag discrepancy at 8B to re-check
at wrap). ft4b (offload_adam full-FT 4B): 2.3 s/item — CHEAPER per item
than the 4B LoRA arms; paging overhead amortized to noise by
grad_accum=8 → the sliding-window Adam prefetch is UNNECESSARY
(empirical gate passed without building it); ETA ~07:00.

C2-29 extension — ceiling grid across scale (v4 task mix, full corpus):
0.6B 0.650 / 1.7B 0.666 / 4B 0.524 (8B/14B pending). The prediction
"the gap closes with scale" is WRONG in 0.6-4B: even 4B-with-passage
reaches only 0.52 while its trained student recites 0.013. Reading:
consolidation is not internalized copying — it SOLVES a behavioral task
(locate-and-continue, recite-inside-dialogue) that in-context prompting
cannot at these scales. Caveat recorded: the ceiling measures BEHAVIOR
(format compliance included — instruct models chat instead of reciting
in maieutic frames); full-copy items at 0.000 show pure format noise is
not dominant. ch8 prose ceiling: 0.565 (same story on Quijote).

### RECLASSIFICATION (owner standard, 2026-07-05 ~04:30): readout source column
Every arm trained before this timestamp used a GOLD-CE readout (task
supervision) — under the refined standard these are BASELINES, not the
method, however clean their batteries. This includes slide8, slide4,
slide2, tailpure, final_k8, and all wave-K/M/N arms. The method's
readout is teacher_kl (teacher's context-conditioned logits; zero gold
in any gradient); its first arm lw_r_slide8kl is queued and the C2-26
table acquires a "readout source" column at wrap. Findings that are
UNAFFECTED (within-family comparisons or gold-free arms): the mimicry
law C2-22 (gold held constant across its contrast), the connectivity
trend, all storage results (strict/tc arms have no CE), lensonly and
the ceiling instruments. The headline "clean, name-faithful method"
is CONDITIONAL on slide8kl matching slide8pure. Default flipped in
config; 'gold' remains in code as a labeled baseline control only.

### ERRATUM (2026-07-05 ~05:15, applies to all C1 + pre-05:00 C2 arms)
Training-side readout terms (tail_ce, lens_ce) targeted the ORIGINAL
TEXT (task labels), not teacher outputs — disclosed in mechanism but
misframed as "the method"; correctly classed as task supervision inside
a hybrid. Eval-side use of the original text is CORRECT and unchanged
(recall is measured against the reference; owner-confirmed). Scope:
~25% of gradient in recipe arms (attribution-measured). Bound on
effect: teacher-forced-with-context agrees with labels at 96.8% top-1 /
0.226 nats (n=16 v4 items) — the pure form (teacher_kl) targets a
near-identical distribution; purification arms slide8kl + thinkselkl
quantify the residual. Unaffected: all hidden-loss/storage results,
strict/tc arms, anchor-KL, contrast-based laws (mimicry, connectivity),
metrology, ceilings. paper1 gains this erratum + readout-source column
at wrap; terminology purged repo-wide ('gold' -> task_label/reference).


### DAMNATIO MEMORIAE (owner directive, 2026-07-05 ~06:00)
The schedule formerly at the expunged markers is removed from code,
configs, docs and papers. Reason: its readout CE targeted the ORIGINAL
TEXT unconditionally (the ce_kind knob never reached it), so none of its
results were valid even as baselines — they were task-supervised runs
wearing a distillation label. teacher_censored is RESTORED to its
original definition (stationary teacher-stream, fully parallel, no
window, no CE — the drift entered at fe3201d for "like-for-like"
comparability). Refusal guards now make both states unrepresentable:
the expunged schedule name raises; teacher_censored + tail knobs raise.

### Signal-attribution table (closing checkpoints; gradient-norm share of hidden losses, measured at convergence, 16 items)
| arm | hidden share | note |
|---|---|---|
| strict (no behavioral term) | 100.0% | instrument sanity anchor |
| **slide8 (doctrine-clean)** | **74.9%** | highest among performing arms |
| tailpure (ablation) | 74.3% | mimicry-free window |
| slide2 | 59.9% | |
| fisher | 56.9% | |
| slide4 | 55.4% | non-monotone in k; best recall of the sweep |
| final_k8 (classical hybrid) | 49.5% | parity, not primacy |
| lens_ce_deep (depth-biased) | 33.7% | 15.1% at trajectory start |
| lensonly (aux-100% control) | 0.0% | instrument sanity anchor; fails both jobs |
Reading: uniform k=8 windows don't just clean the battery — they shift
the gradient composition toward the trajectories (74.9%), while the
classical hybrid sits at parity. Caveat: convergence-time snapshot, not
trajectory-average.

### Finding C2-30: slide8pure — the owner's composition is the project's best arm
Uniform sliding 8-deep credit + mimicry-free top window: **CER 0.007 /
line_exact 0.993, worst category +0.18, intrusion 5.0%, CLEAN** — the
best recall ever recorded at 0.6B tied with the combined arm, with the
cleanest battery of any sliding arm. Composition confirmed: uniform
body credit supplies storage + readout support; removing in-window
trajectory mimicry removes the groove. Classification: HYBRID (trained
pre-05:00 with task-label readout, ~25% share) — its pure twin slide8kl
(teacher_kl) is at epoch 14 and inherits the crown claim if it matches.

### Finding C2-31: the lens_kl HIDDEN-loss family is an intrusion catastrophe — vocab_mse's uniqueness confirmed
Loss grid under uniform windows: slide8+lens_kl recall 0.555 / intrusion
22.5% DESTR; lens_kl-uniform (fully local) recall 0.892 / **intrusion
90.0%** — the most extreme groove ever measured — worst cat +2.09.
Pattern completed across three findings (fisher 57.5%, lensdeep2 50%,
lenskl 90%): DISTRIBUTION-SHAPED losses at any depth amplify the
completion groove; vocab_mse's p-uniform Gram metric is not merely the
best loss, it is the only known SAFE vocabulary-metric loss. (nmse under
windows: recall 0.013 — geometric losses are safe but see C1 for their
storage-portability deficit.)
