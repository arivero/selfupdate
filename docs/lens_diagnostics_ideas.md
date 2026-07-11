# Lens diagnostics — extracted future ideas

Source file saved verbatim at [`docs/lens_diagnostics.py`](lens_diagnostics.py)
(`teacher_student_lens_diagnostics_v2_corrected.py`, syntax-checked, 1763 lines).
It is a self-contained teacher/student comparison toolkit built around
*lens-defined vocabulary observables*. It imports only `torch` + `transformers`,
does **not** import `selfupdate`, and is a reference/diagnostic scaffold — not a
drop-in trainer for this branch. Read this note before importing any of it as a
training loss: several of its objects are diagnostics that would become
**forbidden disguises** if wired into the layerwise objective (see the naming
contract in `AGENTS.md`).

## The one idea to internalize: scores ≠ logits ≠ pseudo-distribution

The file's core discipline is a type system we have been informally leaning on
but never wrote down:

- **hidden state** `h_l ∈ R^d` — no probabilities live here.
- **real model logits** — only `model(...).logits`, at the true output.
- **lens/readout scores** `scores_l = Λ_l(h_l)` — a hidden state (or a hidden
  *contribution*) projected through the frozen head. These are "scores", never
  "logits".
- **pseudo-distribution** `p_l = softmax(scores_l / τ)` — exists *only* after you
  softmax lens scores. Any intermediate Kullback–Leibler (KL) or Jensen–Shannon
  (JS) number is a property of the hidden state **plus the chosen lens**, not an
  intrinsic layer metric.

This directly types our own vocabulary: `lens_kl` is a *lens-induced* divergence,
and `vocab_mse` is a score-**vector** distance (no softmax). The frozen head is a
per-layer *measurement device* — exactly the Frozen-Vocabulary framing in
`AGENTS.md`. Two hygiene rules from the file we should adopt verbatim:

1. **Bias belongs to state readouts, not to vector contributions.** When you
   project a full hidden state, include the head bias; when you project a *delta*
   / Jacobian-vector product (JVP) / any contribution, drop it — a bias is not
   part of a vector.
2. **Center vocabulary scores before cosine / L2, softmax before KL/JS.** Never
   feed a raw contribution into a KL.

## Layerwise-lens implementation order (live)

Scope: every item here is defined by a frozen per-layer measurement map from a
hidden state or residual contribution into a score space, pseudo-distribution,
or downstream sensitivity space. General layerwise losses and interventions
belong in `issues.md`, even when they operate at every layer. The catalogues
below preserve scientific reasoning; they are not priority lists.

### P0 — implement now from existing artifacts

1. **Intrusion depth-localization.** Add recall, anchor, and intrusion prompt
   groups to one teacher↔student layer-grid instrument. Report Jensen–Shannon
   (JS), centered score cosine, and commitment depth. This directly attacks the
   open 1.7B cleanliness question using current checkpoints.
2. **Write-energy and write-direction spectrum.** From cached raw block
   outputs, report `||C W_U Δh_L||`, teacher/student direction cosine, and
   cumulative discrepancy. This is the diagnostic counterpart to the live
   delta-loss experiment and requires no new training.
3. **Execution-regime lens comparator.** Compare matched item-B1 and
   bucketed-B4 checkpoints while that accidental control still exists. Report
   centered score and pseudo-distribution drift; label it a numerical-regime
   study, not a loss result.
4. **Retrospective epoch predictor.** Test whether early lens observables
   predict final recall and standard-benchmark damage. This may support a
   future continuation rule, but cannot override the 12,000-item minimum
   without a separate owner decision.

### P1 — next layerwise-lens objectives

1. **Absolute-score + increment-score loss.** Combine `vocab_mse` with
   `delta_vocab_cos`, logging raw and weighted components per layer. Both terms
   are measurements through the frozen vocabulary lens; the absolute term
   constrains accumulated drift.
2. **Multi-scale contribution lens.** Compare centered vocabulary scores of
   `h_L-h_{L-k}` for `k={1,2,4}` within the sanctioned connected window,
   normalized by teacher score energy.
3. **Cumulative contribution lens.** Compare centered scores of `h_L-h_0` as
   the low-cost long-range control against the multi-scale form.

### P2 — useful after P0/P1 evidence

1. **Same-depth dynamic-time-warping drift**, then cross-depth alignment for
   genuinely unequal-depth teacher/student pairs.
2. **Token-transport coverage** before cross-tokenizer score divergence.
3. **Prompt-local transport JVP integration** after the cheaper frozen-average
   Jacobian diagnostics establish which layers and positions merit the cost.

### Implemented, running, or rejected

- **DONE / running:** `delta_vocab_cos`; `jacobian_nmse`,
  `jacobian_vocab_mse`, `jacobian_lens_kl`; local and
  Anthropic Jacobian fits; slide-1/slide-2 arms.
- **DONE:** `tuned_lens_kl` loading/training path and a first strict run. New
  work is evaluation/calibration, not another implementation.
- **DONE / low-priority control:** bounded symmetric `lens_js`, with slide-1
  and slide-2 configs. Run at most the declared control pair.
- **REJECTED for training:** tail-only lens losses, depth-increasing weights,
  reference-text targets, and the source file's combined tail objective.
- **SCAFFOLD ONLY:** divergence matrices, local transport JVP, and monotone
  alignment exist in `docs/lens_diagnostics.py` but are not wired into reports.

### Additional proposals from this review

- **Lens calibration envelope (P0).** Bootstrap prompts and lens fits, then
  attach confidence intervals to layer rankings. A conclusion that changes
  under prompt resampling is lens variance, not model mechanism.
- **Sensitivity-spectrum decomposition (P0).** Use randomized SVD of the
  frozen metric (`WᵀCW` or `JᵀJ`) and report student error in high-, mid-, and
  low-sensitivity bands. This separates behaviorally visible error from large
  but inert hidden drift without inventing a sharp vocabulary nullspace.

## Status catalogue: whether this branch may TRAIN on them

### ✅ Legal layerwise-loss candidates (depth-uniform, teacher-sourced)

- **`successive_delta_score_matching_loss` — IMPLEMENTED as per-layer
  *contribution* matching.**
  Project each residual write `Δh_i = h_{i+1} − h_i` through the frozen head to a
  vocabulary **score vector**, and match teacher↔student with centered cosine or
  L2 — **not** KL. This is a new member of the safe family: it is score-vector
  shaped (like `vocab_mse`/`nmse`), not distribution shaped, so by the
  loss-safety law (distribution-shaped hidden losses amplify intrusion; vocab_mse
  and nmse stay clean) it is a *predicted-safe* signal. It is teacher-sourced and,
  applied to **all** deltas with uniform weight, it is depth-uniform — it satisfies
  the naming contract. Candidate name: `hidden_loss: delta_vocab_cos` /
  `delta_vocab_mse`. This is the most promising importable idea.
- **`cumulative_hidden_delta` variant (`h_i − h_0`)** — same treatment on the
  running residual sum instead of the per-step write. Cheaper credit assignment,
  same legality. Worth an A/B against the successive form.
- **Tuned-lens-translated per-layer KL — IMPLEMENTED.** The file is the
  reference scaffold for
  the `tuned_lens_kl` / `tuned_lens_path` knobs already added to `config.py` this
  session (see the pending diff). A *frozen* per-layer tuned-lens translator
  (Belrose et al. 2023) makes intermediate lens distributions trustworthy
  **without** depth bias, so `lens_kl` applied on **all** layers stops measuring
  early-layer lens failure. Legal *only if* the translator weight profile stays
  depth-uniform.

### 🔬 Diagnostic-only (measure, never optimize on this branch)

- **`logit_lens_divergence_matrix` / `tail_logit_lens_divergence_matrix`** —
  teacher×student layer grids of lens divergence. Excellent for the "what to
  memorize" / signal-anatomy reports (cf. recent commits on retrieval heads and
  surprise-attention). Use to *show* where student and teacher lenses agree.
- **`local_transport_jvp_scores_one_layer`** — corpus-free prompt-local lens: how
  a source-position hidden vector at layer `l` propagates to the final readout,
  in three cleanly separated transports:
  - `pre_norm_hidden` ≈ `W_U J_l^x h_l` (the intended local hidden-transport lens),
  - `post_norm_hidden` (adds the final-norm local derivative),
  - `final_logits` (direct final-logit JVP).
  This is a routing/locality probe — directly relevant to the "context
  integration peaked near layer 7" observations and the retrieval-head work.
  Expensive (a JVP per layer/position); keep it a probe, never a loss.
- **`monotone_alignment_path` (DTW) + `student_to_teacher_depth_pairs`** — align a
  deep teacher to a shallow student either by relative depth or by a
  dynamic-time-warping path through a divergence matrix. This is the missing tool
  for the **cross-depth bridge grids** in the C3 queue (Qwen3.6-27B bridge,
  Gemma-E4B): it tells you *which teacher layer supervises which student layer*
  instead of guessing. Feeds directly into layerwise supervision when teacher and
  student differ in depth.

### ⛔ Forbidden as a training loss here (diagnostic only, hard stop)

- **`tail_logit_lens_distillation_loss`** and the `tail_lens_weight` term of
  **`combined_distillation_loss`.** A lens loss applied *only to the last N hidden
  states* is exactly the "lens_ce only on deep blocks" / "bigger weights in the
  last blocks" disguise the naming contract explicitly forbids — the tail wearing
  a costume. The file even defaults `combined_distillation_loss` to
  final-KL + tail-lens + delta-score, i.e. a depth-biased objective. **Do not port
  that objective.** A branch-legal reduction of it is: teacher-sourced readout on
  the sanctioned sliding window (already have) **+** depth-*uniform*
  delta-score matching across all layers. Keep the tail-lens matrix for
  measurement, never for gradient.

## Cautions

- **Reference hygiene (publication-critical).** The file's references are a mix
  of pre-2026 anchors (logit lens 2020; Tuned Lens, arXiv:2303.08112; the GELU-4L
  Direct-Logit-Attribution adversarial caveat, arXiv:2310.07325) and **2026
  papers past the January-2026 knowledge cutoff** — these are `YYMM` arXiv IDs
  for 2026 (`2602` = Feb, `2606` = June, `2607` = July), i.e. valid and recent,
  **not** malformed. Verified live on 2026-07-07: **DistillLens (arXiv:2602.13567)
  is real** (Dhakal et al., Georgia State/Auburn, GitHub code) — see
  [`related_work_zo_and_distilllens.md`](related_work_zo_and_distilllens.md) for a
  full prior-art + citation study. The "IG-Lens" (arXiv:2606.29693) and Anthropic
  "global workspace" (2026) entries were **not** independently re-verified here;
  confirm before either enters `paper/`. The GELU-4L caveat is the one to
  actually read: Direct Logit Attribution can be misled by memory-management
  neurons — relevant to trusting the JVP/DLA probes above.
- **HF `hidden_states` is not exact residual writes.** `hidden_states[-1]` is
  usually *post* final-norm, so the last successive delta mixes the final block
  and the norm; the file excludes it by default (`exclude_final_normalized_state_
  from_delta_metrics`). Our block-walk indexes blocks directly, so mapping the
  file's deltas onto our `BlockStack` needs the same off-by-one care around the
  final norm.
- **Vocab-size guard.** `assert_same_vocab_size` — KL/JS over the vocabulary is
  undefined without a token transport map. Any teacher with a different tokenizer
  (a real risk in the cross-model bridge grids) needs a transport map first.

## Historical next steps (status only; use the live queue above)

1. ~~Prototype `delta_vocab_cos`~~ **DONE 2026-07-11** and running in the 1.7B
   loss-grid campaign; `jacobian_vocab_mse` / `jacobian_lens_kl` (the frozen
   Jacobian-pullback family, next section of issues.md) are implemented and
   queued in the same grid.
2. `monotone_alignment_path` remains scaffold-only; same-depth drift is ahead
   of cross-depth bridge integration.
3. `tail_logit_lens_divergence_matrix` and `local_transport_jvp_*` remain report
   probes, never training losses.
4. ~~Finish `tuned_lens_kl`~~ **DONE**; further work is calibration and matched
   evaluation.

## Evaluated diagnostic proposals (2026-07-11 review pass)

All 🔬 diagnostic-only unless marked; each stays inside the type system above
(scores centered before cosine/L2, softmax only for KL/JS, bias only on state
readouts).

- 🔬 **P0 — Regime-fork lens comparator (free control, time-limited).** The
  2026-07-11 speed flip left the live loss grid with a LABELED fork: same
  losses, same data, `item` B=1 vs `bucketed` B=4 (bf16 kernel-shape
  numerics + bucket order differ). The five B8 per-epoch telemetry streams
  also carry the now-fixed per-batch generation-budget confound; use B1 final
  checkpoint evaluations for outcomes. A checkpoint↔checkpoint lens-divergence
  grid (student-vs-student — everything so far is teacher-vs-student)
  between matched arms would measure how much execution-regime drift
  becomes REPRESENTATIONAL drift at matched budget. This is the
  cheapest-ever measurement of "does bf16 batching noise matter
  scientifically"; prioritize it before code/data drift complicates the
  natural control.
- 🔬 **P0 — Intrusion depth-localization (targets open C3 #9).** Run the lens
  JS grid on the 200 intrusion prompts, teacher vs trained student, per
  layer: WHERE in depth does intrusion first appear? The 1.7B cleanliness
  question (22-40% intrusion vs 1.5-2.5% at 0.6B) currently has only an
  output-level number; the loss-safety law predicts distribution-shaped
  training writes the groove mid-stack. If 1.7B intrusion has a depth
  signature 0.6B lacks, that is the first mechanistic handle on it.
- 🔬 **P0, fold into intrusion grid — Commitment-depth profile.** Per
  prompt, the depth at which the lens pseudo-distribution's argmax first
  equals the model's final token and stays. Teacher-vs-student on recall
  vs anchor prompts: does layerwise training move commitment EARLIER
  (memorization signature) or preserve the teacher's profile? One scalar
  per prompt from machinery we already have; directly instruments the
  last-3% law's claim that storage succeeds while readout lags — students
  that commit early but recite wrong localize the failure to late layers.
- 🔬 **P0 — Write-energy spectrum.** `||W_U C Δh_i||` per layer over aligned
  spans (centered, bias-free — a contribution), teacher vs student.
  States can match while the PATH of writes differs; this is the
  measurement companion to the delta-vs-state question the grid is
  testing, computable from the cached hidden states with no new GPU work.
- 🔬 **P2 — Same-depth DTW drift probe.** The doc proposes DTW for cross-depth
  bridges; run it teacher-vs-trained-student at SAME depth too. Deviations
  from the diagonal localize where training re-timed computation (cf.
  "context integration peaked near layer 7") — a re-timing probe the
  layer-residuals instrument cannot see (it compares same-index layers by
  construction).
- 🔬 **P0 analysis, not an abort gate — Retrospective continuation validation
  (uses data we already have).**
  Completed grid arms carry per-epoch telemetry. Test on that record:
  does a depth-uniform lens-JS (or the epoch-3 recall trajectory) predict
  final word_acc well enough to justify an early-abort gate for FUTURE
  sweeps? The never-abort-before-12k-items rule exists because early
  plateaus recovered; this is the cheap way to learn whether a
  lens-observable separates "plateau, still storing" from "dead arm" —
  validation first, gate later, and only with owner sign-off since it
  touches the 12k law.
- ✅ **P1 engineering — Bias/centering hygiene as a runtime tripwire
  (trainer-adjacent).** Rule #1/#2 of the type system are prose; make
  `HiddenLoss` construction assert them: delta-kind losses must project
  bias-free, state-kind readouts must include the head bias when the
  family has one. Identity for Qwen3 (no head bias) — load-bearing the
  day a bias-ful family enters the bridge grids, and exactly the
  silent-knob bug class this branch keeps closing.
- 🔬 **P2 bridge prerequisite — Token-transport coverage report.** The
  vocab-size guard says cross-tokenizer lens comparison is undefined
  without a transport map; the constructive version: build the map once
  (exact string pairs + subword-decomposition fallback), then REPORT the
  untransportable probability mass per corpus. That single number decides
  whether cross-family bridge-grid lens comparisons (Gemma-4, ALIA) mean
  anything before anyone plots them.
