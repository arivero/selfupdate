# Related work — zeroth-order fine-tuning (MeZO/ZO-Act) and DistillLens

Companion to [`lens_diagnostics_ideas.md`](lens_diagnostics_ideas.md). Two
external threads the owner asked to place relative to this branch's framework.
Web-verified 2026-07-07 (papers are past the January-2026 knowledge cutoff).

---

## 1. Zeroth-order fine-tuning — "mezo" (ZO-Act, arXiv:2607.01125)

**Lineage.** The original **MeZO** (Malladi et al., *Fine-Tuning Language Models
with Just Forward Passes*, arXiv:2305.17333, 2023) estimates gradients from
**forward passes only** — no backprop — via SPSA-style ±ε perturbations, giving
inference-level memory. **ZO-Act** (Dong, Xu, Wang, Li, Yin, Yang, arXiv:2607.01125,
Jul 2026) is a variance-reduction descendant: it restricts perturbations to a
**fixed low-rank subspace derived from input activations**, computed once per
linear layer, then optimizes lightweight coefficient matrices with a momentum
optimizer (Adam). It cuts the variance and finite-difference error of
random-subspace ZO, supports **INT4-quantized** frozen weights, and reports gains
on Llama-3-8B / OPT-13B / INT4 models.

### Does it fit our framework?

**Short answer: it is an *optimizer*, orthogonal to *what* we optimize — so it
does not replace the layerwise objective, but it is a real option on the
extreme-scale / black-box-teacher axis.**

- **Orthogonality.** ZO changes how a gradient is *estimated*; it says nothing
  about the loss/observable. It could in principle estimate any of our signals
  (`vocab_mse`, `lens_kl`, the new `delta_vocab_cos`). It neither competes with
  nor validates our loss family.
- **Conflict with the layerwise *core*.** Our publication claim rests on
  **per-block gradient credit via sliding k-connected gradient-isolation
  windows** (`docs/windows.md`). ZO estimates the gradient of a **scalar**
  objective w.r.t. parameters *as a whole*; it produces no per-layer gradient
  isolation and cannot express the k-deep uniform-credit structure that *is* the
  forward/layerwise claim. **Adopting ZO as the trainer would dissolve the very
  mechanism the paper is about.** → Not a layerwise-core method.
- **Wall-clock reality on our bottleneck.** We are compute/sync-bound, not
  memory-bound, at 0.6–14B (see the pipeline-bubble + `.item()`-sync analysis in
  `CLAUDE.md` and the `clean_q_ch1_..._alia40b` run). ZO buys inference-level
  *memory* at the cost of *more forward passes per step*. Since our forward is
  already the expensive part (40B PP4, mostly-idle cards), ZO would likely trade
  memory we are not short of for wall-clock we are short of — a poor trade here.
  Its value only turns positive where **memory is the hard wall**: the MoE /
  120B-class rung of the Hardware Ladder, and sharded 32B.
- **Compatibility we do get for free.** ZO-Act freezes low-bit weights and trains
  only coefficient matrices — philosophically aligned with our Frozen-Vocabulary
  locks (embedding/head never trained). A ZO variant here would still need the
  vocab-signature tripwire.

### The one genuinely novel cross-over: black-box / non-differentiable teachers

The strongest fit is **not** speed but *reach*. Where the teacher is a **black
box with no gradients** — an API teacher, or `gpt-oss`-class weights we can only
run forward — the online-teacher path cannot backprop *through* the teacher.
Zeroth-order estimation gives a forward-only way to still push the student toward
such a teacher's readout. This is the natural home for ZO in our program:
**black-box-teacher distillation on the extreme-scale rung**, documented as a
scaling tactic (route toward `docs/scaling.md` / the sibling `selfupdate_kd`
checkout), **not** as a layerwise-core objective. It is *not* imported into
`train/layerwise.py`.

---

## 2. DistillLens (arXiv:2602.13567) — prior art + citation study

**Verified real** (my earlier "malformed arXiv id" note was wrong; `2602` = Feb
2026): Manish Dhakal, Uthman Jinadu, Anjila Budathoki, Rajshekhar Sunderraman,
Yi Ding (Georgia State University / Auburn University), submitted **2026-02-14**,
code on GitHub. Evaluated on GPT-2 and Llama, instruction-following benchmarks.

**Method.** Projects intermediate hidden states into vocabulary space via the
**logit lens**, then aligns student↔teacher with a **symmetric divergence**
(two-way KL) objective. The paper argues the symmetric penalty imposes a
dual-sided constraint that prevents both over- and under-confidence and preserves
"high-entropy information conduits." Claims it beats standard knowledge
distillation (Kullback–Leibler on final logits) and feature-transfer baselines.

### Citation study (as of 2026-07-07)

- **Semantic Scholar: `citationCount` = 0, `influentialCitationCount` = 0.**
- Web / Google Scholar surfaced **no follow-up work** citing it. The paper is ~5
  months old — plausibly just too recent to have accrued citations. **Recheck
  before camera-ready.** (Adjacent-but-not-citing KD-temperature work exists,
  e.g. arXiv:2605.20357, 2606.00306; not follow-ups to DistillLens.)

### Why this matters to us — it is prior art we MUST cite and differentiate

DistillLens is a published instance of the *same family* as our `lens_kl`:
logit-lens projection to vocabulary space as a distillation signal. `paper/paper1`
must cite it and stake the difference sharply. Three concrete contrasts, each a
testable claim:

1. **Loss-safety disagreement (our strongest card).** DistillLens makes
   **symmetric lens-KL the core objective**. Our loss-safety law (see the
   loss-safety memory / `docs`): distribution-shaped hidden losses — `lens_kl`,
   Fisher — **amplify intrusion**, whereas score-vector losses (`vocab_mse`,
   `nmse`) stay clean. So we *empirically disagree with their central choice*.
   Actionable: reproduce symmetric lens-KL under our intrusion battery
   (n=200 prompts) and report the intrusion delta vs `vocab_mse` at matched item
   budget. If our law holds, that is a headline differentiation.
2. **Depth-uniformity (naming contract).** Verify whether DistillLens weights
   layers uniformly or emphasizes deep/tail layers. Uniform ⇒ a legitimate
   comparison baseline; tail-weighted ⇒ exactly the depth-biased disguise our
   naming contract forbids, and a point to call out. (Not yet determined from the
   abstract — read the method section.)
3. **Training target.** Confirm DistillLens is purely teacher↔student (both
   directions between the two models) and does **not** mix reference-text
   cross-entropy. If teacher-sourced only, it is compatible with our
   training-target law; if it blends ground-truth supervision, that is the
   role-conflation we purged ("gold").

**Our novelty must be sharper than "we also use the logit lens."** The
differentiators to enumerate against DistillLens: (a) the **sliding k-connected
gradient-isolation window** with uniform k-deep credit (not a global lens loss);
(b) **frozen-vocabulary measurement discipline** + four locks + runtime tripwire;
(c) the **intrusion / loss-safety** finding; (d) **`vocab_mse`** — a *score-vector*
(non-distributional) carrier — as the safe alternative to their symmetric KL;
(e) teacher-sourced-only training (no reference-text term).

### Action items

- [ ] Add DistillLens to `paper/paper1` related work with contrasts 1–3.
- [ ] Read the DistillLens method section: resolve the depth-weighting and
      training-target questions above.
- [ ] Run the symmetric-lens-KL intrusion reproduction vs `vocab_mse` (matched
      budget, n=200 intrusion prompts); report gradient-share attribution.
- [ ] Re-query citations before submission (was 0 on 2026-07-07).

---

## Sources

- MeZO (original): <https://arxiv.org/abs/2305.17333>
- ZO-Act: <https://arxiv.org/abs/2607.01125>
- DistillLens: <https://arxiv.org/abs/2602.13567> · PDF <https://arxiv.org/pdf/2602.13567>
- DistillLens citations (0): Semantic Scholar Graph API, paper `arXiv:2602.13567`, queried 2026-07-07.
