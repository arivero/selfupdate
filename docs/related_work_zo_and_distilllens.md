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

## 3. 4-bit quantization: the actual situation for eval vs training losses

Short version: **4-bit is EVAL-only on this branch and validated as such; it is
NOT part of the training recipe, and folding it into training changes the loss
regime in four concrete ways that are not handled today.**

### What we actually do (eval)

- `evaluate.py --load-4bit` loads the base in NF4 (bitsandbytes) so a 40B eval
  fits beside a resident 40B training job. Training itself (`train/layerwise.py`)
  runs a **bf16** base with a **bf16** LoRA adapter — no quantization touches any
  loss.
- **Validated 2026-07-07:** the 4-bit machado eval (character error rate,
  CER 0.877) matched the bf16 in-training recall (CER ~0.84). So NF4 preserves the
  recall-relevant signal *at inference* for these checkpoints. That is an
  inference statement, not a training one.

### Could you train the losses on a 4-bit base? Mechanically yes (QLoRA), but…

bitsandbytes backprops through a dequantized NF4 linear to update bf16 LoRA
adapters while the base stays 4-bit. That works in general. For **our** loss
framework it introduces four specific problems:

1. **A quantized base is a quantized *teacher*.** In `online_teacher` mode the
   teacher is the base with adapters off. Quantize the base and every
   teacher-sourced target — the `teacher_kl` readout and the per-layer teacher
   hidden states — is now computed from a 4-bit model. Because our entire signal
   is teacher-sourced (training-target law), quantization degrades the **source**,
   not merely the student. This is the sharpest issue and no optimizer choice
   fixes it.
2. **The frozen measurement head must stay high-precision.** The Frozen-Vocabulary
   Principle keeps the embedding and language-model head untrained; QLoRA
   *conventionally* also keeps `lm_head`/`embed_tokens` in bf16, but that is a
   config choice, not a guarantee. If a quantization config quantizes the head,
   the per-layer lens/`vocab_mse` measurement device itself becomes lossy and
   silently shifts every hidden loss. The runtime tripwire checks for **change**,
   not **precision**, so it would not catch a frozen-but-quantized head. 4-bit
   training would need the tripwire extended to assert head/embed dtype.
3. **Loss shape interacts with quantization noise.** Quantization injects noise
   into hidden states. Distribution-shaped losses (Kullback–Leibler through the
   lens, `lens_kl`) softmax the lens scores, so small perturbations can move a
   peaky pseudo-distribution a lot — more quant-noise-sensitive. Score-vector
   losses (`vocab_mse`, centered cosine / L2) never softmax and are comparatively
   robust. This compounds the loss-safety law (distribution-shaped hidden losses
   amplify intrusion): under 4-bit they would likely be even less stable, so a
   4-bit attempt should prefer the score-vector family.
4. **Gradient error accumulates over the block walk.** QLoRA's backward pass sees
   the dequantization error; our summed schedule walks all 48 blocks with a
   per-block hidden-matching term, so that error accumulates per block. Untested
   here.

### Where ZO-Act fits this specifically

ZO-Act's INT4 support (§1) cleanly sidesteps **only** problem 4: zeroth-order needs
no gradient *through* the quantized weights, so quantization error never corrupts a
backward pass (there is none). It does **not** fix problem 1 — a 4-bit teacher is
still a degraded source under any optimizer — and it remains forward-only,
high-variance, and non-layerwise. So "train on a 4-bit base without noisy
gradients" is real via ZO, but it is a scaling/black-box-teacher tactic, not a way
to make our layerwise losses quantization-safe.

### Bottom line

4-bit is a **memory** tactic: used and validated for **eval**, reserved for the
extreme-scale rung. Training losses on this branch are computed in bf16 against a
bf16 (dequantized) teacher. Moving 4-bit into training (i) degrades the
teacher source, (ii) risks quantizing the frozen measurement head, (iii) favors
score-vector over distribution-shaped losses, and (iv) accumulates dequant
gradient error over the block walk — none of which is in place today. Treat any
4-bit-*trained* number as unproven until these are handled and tested explicitly.

---

## Sources

- MeZO (original): <https://arxiv.org/abs/2305.17333>
- ZO-Act: <https://arxiv.org/abs/2607.01125>
- DistillLens: <https://arxiv.org/abs/2602.13567> · PDF <https://arxiv.org/pdf/2602.13567>
- DistillLens citations (0): Semantic Scholar Graph API, paper `arXiv:2602.13567`, queried 2026-07-07.
