# What to Memorize: Retrieval Heads, Surprise, and Routing in Daydreaming Memorisation

**The Pierre Menard Program, Stage 2 — working paper, 2026-07-07**

*selfupdate / layerwise branch · companion to, and separate from, the
cross-checkout Report and the Stage-1 campaign report (`paper1`) ·
repository: `supercomplex/selfupdate_lw`*

---

## Abstract

Self-distillation of context asks a model to fold a privileged input (a
retrieved passage, a `<think>` trace) into its own weights so it can later
reproduce the behaviour *without* that input. This paper isolates a question
that Stage 1 left implicit: **what, exactly, should be memorized?** We give
three complementary answers, each with a fast instrument: (1) a *content-head
taxonomy* that localizes retrieval to long-range, privileged-heavy attention
heads in mid-network; (2) a *surprise decomposition* that splits each
high-loss answer token into a **knowledge gap** (the teacher resolves it from
the privileged block — a genuine memorization target) versus an **attention
misdirection** (the answer is already in shared context but the student
mis-routes — a routing fix, not a memory target); and (3) a *routing view*
that casts both as the blackbox-versus-explicit-router choice familiar from
mixture-of-experts. Along the way we retire a noise-dominated deterioration
metric in favour of a bounded, single-pass standard battery, and we surface a
methodological trap — the attention sink — that silently inverts the surprise
decomposition if uncorrected.

## 1. The question

Stage 1 established *where* context is stored (per-layer forward trajectory
matching; storage concentrates in a mid/late band) and *that* a small
teacher-sourced readout is needed to recite it. It did not pin down *what*
content the storage objective should prioritise. In the "daydreaming
memorisation" framing — the model rehearses privileged context offline and
consolidates it — the store is finite, so the selection problem is primary:
absorbing everything is neither possible nor desirable, and much of a
passage is predictable boilerplate the base model already emits.

We decompose "what to memorize" into three measurable factors.

## 2. A bounded measurement, and why the old one was retired

The Stage-1 deterioration signal was a held-out general cross-entropy
("destruction") canary. Measured across training it is **noise-dominated and
near-flat** — it neither tracks damage nor discriminates methods. Because we
now have *many small models* (161 final checkpoints across three checkouts,
100 of them Qwen3-0.6B), we need a measurement that is (a) standard, (b)
bounded and low-variance, and (c) cheap enough that the only real cost is
loading the model.

We replace it with two families, all scored in a **single teacher-forced
batched pass** (no autoregressive generation):

- **Deterioration / retention** (capability preserved vs the epoch-0 teacher
  of the same base): ARC-Easy accuracy on a fixed cached subset (option
  log-likelihood), and WikiText-2 perplexity. *Accuracy is the stable axis;
  perplexity is `exp(CE)` and shares the noise family we are leaving.*
- **Recall** (memorization of the original text; evaluation only, never a
  training target): exact continuation and interior-word cloze for the
  Machado verse; next-sentence continuation and **multi-word paragraph
  censor** for the Cervantes prose.

Loading dominates: on Qwen3-0.6B we measure a load/eval ratio of ~5.6×, so a
plain-PyTorch forward is the right tool — vLLM cannot reduce disk load and
adds engine-startup overhead. The load win is architectural instead: LoRA
adapters are hot-swapped on a resident base (47 adapters collapse onto ~8
base loads). This is also why "evaluate the currently-running trainings" is
nearly free: the eval is a few seconds on top of a load one would pay anyway.

The two retained/recall axes are directly comparable — both accuracies in
[0,1] — so the natural artifact is a **bidimensional Pareto** of recall
against retention (`runs/recall_retention.png`). Because the trainer saves a
single final checkpoint per run (no per-epoch snapshots), this is an
*endpoint* cloud, not a per-epoch path; a true trajectory would require an
in-training retention pass, which we flag as future work.

## 3. Factor 1 — the content-head taxonomy

The base model, run on the *teacher* view (privileged passage present) with
eager attention, exposes per-head structure at the answer positions. Each of
the 448 heads (Qwen3-0.6B) is scored on attention **distance**, **entropy**,
and **mass onto the privileged block**. Retrieval is the defining property:
long-range, privileged-heavy heads are **content/insight** heads — they mark
what the context considered worth absorbing — while local, privileged-blind
heads are **grammar**. Mean answer→privileged mass by layer peaks in
mid-network. This is the mechanistic target the storage objective should
weight, and it names what the recall probes ought to test.

*Caveat, load-bearing for Factor 2:* the raw distance axis is confounded by
**attention sinks** — a head that stares at token 0 (or a chat delimiter)
from the answer scores a huge "distance" with near-zero content. Distance
therefore only *defines* grammar (local + privileged-blind); it never on its
own defines content.

## 4. Factor 2 — surprise, and its two causes

Surprise is the gap between what the student predicts and what actually
happens. For each aligned answer token we measure student-view NLL, teacher-
view NLL (privileged block present), and their difference — the **excess
surprise** the privileged context resolves. A large excess is ambiguous:

- **Knowledge gap** — the teacher's attention at that token lands on the
  privileged block. The student cannot produce the token without the passage.
  *This is the thing to memorize.* (In our base 0.6B/poem run these are the
  proper nouns: *Alvargonzález*, *Jacob*, place names.)
- **Attention misdirection** — the token is reachable from shared, uncensored
  context, but the student attends to the wrong tokens. *This is a routing
  error, not a memory target*, and would be masked, not fixed, by storing it.

The disambiguator is *where the teacher attends*: privileged-block mass →
knowledge gap; in-context mass → misdirection.

**The sink nearly ruined this.** A first implementation aggregated attention
over all heads and found ~34% "misdirection", every case pointing at the
`<|im_start|>` delimiter — the attention sink of §3 — while genuine
knowledge-gap tokens (the proper nouns) were *mislabelled* misdirection
because the sink out-massed the passage. Restricting the footprint to the
taxonomy's **content heads** and masking structural/special tokens flips the
picture to the sensible one: on base Qwen3-0.6B / poem, **~33% knowledge
gap, <1% misdirection**, the rest low surprise (`runs/surprise_probe_0.6B/`).
High surprise here is almost entirely *content to absorb*, with negligible
routing error — a concrete, if single-setting, data point *for* the optimism
of §5.

## 5. Factor 3 — routing: blackbox versus explicit router

"What to memorize / where to attend" is a routing decision, and it has the
same two forms as mixture-of-experts expert selection, which the codebase
already exposes as `train.moe_mode`:

- `dense_or_black_box` — natural routing, learned implicitly from the
  distillation loss (surprise is its gradient);
- `router_aligned` — a per-layer regularizer pulling the student router
  toward the teacher's;
- `teacher_forced` — the teacher's selection imposed directly.

The same duality lifts from expert routing to *attention* routing: the
misdirection of §4 is exactly a case an explicit attention-alignment term
would fix and a blackbox objective might route around.

**Conjecture (owner, optimistic).** The blackbox objective should always
converge. We sharpen this: it converges in **behaviour** — minimising
surprise matches the output distribution — but not necessarily in
**mechanism**. A student can reach the right token by attending to the wrong
place, driving surprise to zero while the route stays wrong; and the
attention sink is a degenerate attractor blackbox routing demonstrably does
*not* escape. So "always converges" is safe for the loss, optimistic for the
route.

**The test, and a first result (gpt-oss-20B).** The multigpu campaign trained
the controlled triple — `slide2` (blackbox) vs `slide2_ra` (router_aligned) vs
`slide2_tf` (teacher_forced) — on gpt-oss-20B, one shared base (loaded once,
adapters hot-swapped). The §2 battery gives:

| routing | ARC-Easy retained (acc / teacher) | recall (exact continuation) |
|---|---|---|
| blackbox (`slide2`)          | **0.38** | 0.00 |
| router_aligned (`slide2_ra`) | **0.49** | 0.00 |
| teacher_forced (`slide2_tf`) | **0.57** | 0.00 |

The three do **not** coincide, so this is evidence *against* strict
"blackbox always converges": capability retention rises monotonically with
routing supervision, and the blackbox arm is the **most collaterally
damaging** (it keeps only 38% of the base model's ARC-Easy accuracy, versus
57% under teacher-forced routing). Two caveats bound the claim: (1) recall
(exact continuation) is **floored at 0** for all three — a 20B MoE at a
2-block sliding window does not reproduce the poem verbatim — so the contrast
rests entirely on the retention axis and the recall half of the conjecture is
untested here; (2) `teacher_forced` imposing the teacher's expert selection
means *fewer router parameters move*, so its retention edge may be "changes
less" rather than "converges better". The honest reading: on this base,
routing supervision strongly protects capability and blackbox is the worst
for collateral damage — it complicates the optimistic conjecture rather than
settling it. A recall-discriminating setting (a smaller base, or a wider
window that actually memorizes) is the needed follow-up.

## 6. Instruments

| Script | Measures |
|---|---|
| `scripts/retention_eval.py` | ARC-Easy + WikiText retention, exact recall probes; load-once, LoRA hot-swap, resumable |
| `scripts/retention_index.py` | aggregates the three checkouts → `runs/retention_index.csv` |
| `scripts/retention_plots.py` | bidimensional recall-vs-retention Pareto |
| `scripts/attention_probe.py` | content-head taxonomy (Factor 1) |
| `scripts/surprise_probe.py` | surprise decomposition (Factor 2), content-head footprint, sink-masked |
| `scripts/cross_report.py` | assembles `runs/cross_report.pdf` |

## 7. Limitations

- Endpoint-only: no per-epoch checkpoints, so retention trajectories are
  clouds, not paths. An in-training ARC pass would fix this.
- Single-model mechanism: Factors 1–2 are shown on base Qwen3-0.6B; they must
  be repeated across sizes and on trained checkpoints (not only the base).
- Probe scale: the surprise decomposition uses 16 privileged examples; the
  fractions are indicative, not tight.
- The knowledge-gap / misdirection split is thresholded on excess-surprise
  quantiles and on a priv-vs-context mass comparison; both are defensible but
  tunable, and the content-head restriction is essential to either.

## 8. Conclusion

"What to memorize" resolves into three measurable factors — retrieval-head
localization, surprise that is genuinely a knowledge gap, and the routing
choice between blackbox and explicit alignment — over a metric that is
bounded, standard, and cheap. On our first setting, high surprise is
overwhelmingly knowledge to absorb rather than misrouted attention, which is
weak but real support for the blackbox-converges conjecture; the controlled
router triple, evaluated with the same battery, will make that support strong
or refute it. The recurring lesson is methodological: the attention sink
masquerades as both long-range content and misrouted attention, and every
statement here depends on excluding it first.
