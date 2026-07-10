# Experiment Plan & Status Board

Updated: 2026-07-10. Working-file convention (as issues.md and
recommendations.md, 2026-07-10): closed/absorbed content is REMOVED from
this file — git history keeps it. The full chronological narrative (C1
waves, findings C2-1..C2-35, wave tables, chimeras, 0.8-band anatomy,
intrusion-200 detail) lives in this file's history at commit `4638ddc`
and in docs/casebook.md + paper/paper1.pdf. Machine-readable claims:
runs/conclusions.yaml (validate with scripts/conclusion_check.py).

CURRENT TRUTH: Campaign 2 closed 2026-07-05. Crown = **slide8pure**
(CER 0.007 / line-exact 99.3% / intrusion 2.5% ± 2.2 at n=200),
seed-replicated (lw_r_s43_pinned: 0.0076 / 0.991 / 1.5%). Classified
"hybrid by lab necessity, deployment-pure by transcript equivalence",
with slide8kl (0.801) standing as the measured pure-distribution bound
(Law 9).

Metrics: `runs/results.md` (auto) | report: `runs/report.pdf` | raw logs:
`runs/*/metrics.jsonl` and `runs/pipeline_*.log`.

## Standing Goal

Find a layerwise loss that trains with bounded backward depth and still
produces behavior. "Good" means:

- recites under full-corpus eval, not just the training subset
- preserves block locality except for explicitly bounded sliding
  connected windows
- has measurable forgetting/general-CE cost
- scales to online-teacher LoRA and one-block-at-a-time training

## THE LAWS OF CAMPAIGN 2
1. **Anchor-Goodhart** (C2-2/3): regularizers protect where anchored; diversify or be fooled.
2. **Dilution** (C2-8/9): co-residence is free; more content and more params-per-item lower intrusion; batching beats drilling (overtraining law C2-6).
3. **Mimicry** (C2-22): matching the teacher's with-context trajectory near the readout installs the intrusion groove; mimicry-free windows are clean.
4. **Connectivity** (C2-26): uniform sliding k-windows — recall by k=4, cleanliness and hidden-primacy by k=8.
5. **Synergy** (C2-28): trajectories store but can't speak; per-layer CE speaks but destroys; only the composition works.
6. **Reader/writer** (C2-29/grid): the teacher is distilled as a reader; students exceed its writing 30–70×; ceilings flat in scale.
7. **Loss safety** (C2-17/23/31): distribution-shaped hidden losses amplify grooves (fisher 57.5%, lens_kl 90%); vocab_mse/nmse safe; loss choice flattens under uniform windows.
8. **Premise condition** (C2-32): teacher_kl transmits only what the teacher knows in context; premise-check per data mode.
9. **The last 3%** (C2-34): pure distribution-matching converges to teacher fidelity exactly; verbatim recall lives in the residual the teacher lacks — bounded reference-CE is irreducible in the lab, transcript-CE-equivalent in deployment.
10. **Parallelism**: PP2 evals validated; the PP2 TRAINING repro failure (0.837 vs 0.015) was the config-default confound — CLOSED 2026-07-10: lw_q_pp2fix landed CER 0.011 / line-exact 0.988 (crown-class recall under pipeline_split=14), and the trainer is certified under PP2 against single-device references (certs/pp2: summed, sliding readout, anchor, offload, LoRA variants; losses + final weights match). offload_adam works (full-FT 4B at 2.3 s/item; streamed pinned paging 2026-07-10 cut step overhead 2.65x at 0.6B); TP timing benched — probe-only by policy (PP partitions at block boundaries; TP puts a collective inside every linear).

## CAMPAIGN 2 CLOSING TABLE (2026-07-05; Qwen3-0.6B / v4 Machado unless noted)
Readout sources: T = task-CE (reference text; "transcript-CE equivalent" in deployment), K = teacher_kl, — = none.
Classification: METHOD (doctrine-clean sliding), HYB (pre-law hybrid baseline), ABL (labeled ablation), CTL (instrument control).

| arm | recall CER / line% | worst cat | bench_min | intrusion | hidden share | readout | class | verdict |
|---|---|---|---|---|---|---|---|---|
| **slide8pure (THE RECIPE)** | **0.007 / 99.3** | **+0.18** | ok | **5.0%** | **84.4%** | T | METHOD | **CLEAN** |
| slide8 | 0.017 / 97.7 | +0.35 | −1.0 | 7.5% | 74.9% | T | METHOD | CLEAN |
| slide8+nmse | 0.013 / 99.0 | +0.39 | ok | 7.5% | — | T | METHOD | CLEAN |
| slide4 | 0.009 / 98.8 | +0.47 | −3.0 | 12.5% | 55.4% | T | METHOD | intr only |
| slide2 | 0.052 / 94.4 | +0.84 | −4.0 | 20.0% | 59.9% | T | METHOD | DESTR |
| slide8kl (pure bound) | 0.801 / 3.4 | +0.15 | −1.5 | 12.5% | ~100% | K | METHOD | C2-34 law |
| final_k8 (C1 "final recipe") | 0.015 / 98.6 | +0.47 | −5.5 | 12.5% | 49.5% | T | HYB | DESTR |
| combined (v4+ch1) | 0.007 / 99.2 | +0.46 | −3.5 | 10.0% | — | T | HYB | CLEAN (2 seeds) |
| thinksel (thinking channel) | 0.037 / 95.3 | +0.18 | +0.5 | 5–17.5% | — | T | HYB | CLEAN@s17 |
| tailpure | 0.017 / 97.8 | +0.25 | −5.0 | 2.5% | 74.3% | T | ABL | CLEAN |
| tailonly-4B | 0.013 / 98.6 | +0.70 | −1.0 | 25.0% | ~0% body | T | ABL | DESTR |
| fisher | 0.058 / 96.9 | +3.22 | ok | 57.5% | 56.9% | T | ABL | DESTR |
| lensdeep2 / lenskl-uni | 0.046 / 0.892 | +2.51 / +2.09 | −5.0 | 50% / 90% | 34% / — | T / — | ABL | DESTR |
| lensonly | 0.795 / 0.4 | +8.70 | — | 0% | 0% | T | CTL | fails both |
| strict | 0.849 / 0 | — | — | — | 100% | — | CTL | stores only |
| tc-modern (teacher-stream) | 0.877 / 0 | +0.82 | −6.5 | 0% | 100% | — | METHOD | no readout |
| [expunged schedule] | — | — | — | — | — | T | EXPUNGED | damnatio |

**Ladder** (prose, 0.6B): ch1 0.000@200ep (27.5% intr — overtraining law) · ch4_av2 0.084 CLEAN · ch8_av2 0.032 (15% intr) · ch16_ext 0.078 (capability-DESTR). Recall never saturated; the destruction envelope is the constraint.
**Scale** (v4): 1.7B FT 0.023 (−8.0 hs) · 4B LoRA r16/r64 −16/−18 hs (LoRA closed) · 14B LoRA 0.066 (intr 2.5%) · full-FT-4B via offload_adam in flight.
**Teacher ceilings** (with-passage generation): 0.650 / 0.666 / 0.524 / 0.651 / 0.449 (0.6→14B) — flat; consolidation beats prompting 30–70× at every size.

(Intrusion re-measured at n=200 2026-07-05: ordering stable, every
CLEAN arm under threshold, crown at 2.5% ± 2.2; detail in history.)

## Finding index (narrative in git history @ `4638ddc` + docs/casebook.md)

| id | finding | status |
|---|---|---|
| C1-loss | Loss-search winner vocab_mse (0.024 / 0.978 at 0.6B); its storage format is portable (chimera: foreign readout reads it at 0.109) | absorbed (Law 7; superiority was tail-hybrid-specific) |
| C1-dissoc | Storage is distributed and redundant in the upper-middle stack; the readout is a fragile co-adapted top-k circuit (single-layer body ablations harmless; ablating any one top block destroys recitation) | absorbed (Laws 3/5) |
| C1-poem | Whole-poem milestone: 715-verse romance, first error at verse 708, self-chained CER 0.007 (no drift) | resolved (milestone stands) |
| C1-anchor | anchor-KL halves intrusion at zero recall cost (Bécquer +2.29→+1.06); anchor-CE recorded negative (memorizes the anchors) | absorbed (superseded by Law 1 + v2 anchors) |
| C2-1 | v2 destruction battery relabels every C1 arm destructive (final_k8 intrusion 22.5%, C1 champion 40.0%); the legacy 4-text mean hid it | resolved (battery adopted) |
| C2-2 | anchor-KL taught to the test: poetry-only anchors protect poetry (+0.37) while prose_es/procedural/facts absorb displaced damage | absorbed (Law 1) |
| C2-3 | multi-genre v2 anchors generalize protection at zero recall cost (intrusion 22.5→12.5%) | absorbed (Law 1) |
| C2-4 | q_ch8 recall 0.074 at 22k items; prose_es is the most-damaged category regardless of memorized genre | resolved |
| C2-5 | q_ch4_av2 first arm to clear every destruction threshold (weakened by C2-18 to clean-at-one-seed) | resolved |
| C2-6 | overtraining a tiny corpus is destructive (ch1 200ep: recall 0.000 but intrusion 27.5%); stop on recall closure, not schedule | absorbed (Law 2) |
| C2-7 | ch16 sect-family "index saturation" was an item-budget artifact, not an addressing wall | resolved (by C2-14) |
| C2-8 | co-residence is free at 0.6B: combined arm's poem side 0.007 / 0.992, unharmed by its housemate | absorbed (Law 2) |
| C2-9 | intrusion residual is a corpus-size problem, not anchor weight (w=1.0 stuck at 12.5%; combined 10.0% CLEAN) | absorbed (Law 2) |
| C2-10 | LoRA at 4B is MORE destructive than full-FT at 0.6B (hellaswag −16.0) | absorbed (C2-24 closes the axis) |
| C2-11 | tuned lens confirms mid-depth storage is genuinely non-vocabulary-shaped (trained-vs-base separation opens only at L22+) | resolved |
| C2-12 | selective think-censoring wins (0.037, CLEAN); the thinking channel is the gentle channel | resolved (continues as C3 item 2) |
| C2-13 | LoRA destructiveness persists at 8B (0.187, hs −8.5); LoRA recall degrades with scale at matched epochs | absorbed (C2-24) |
| C2-14 | recall does NOT saturate by ch16 (0.078 passes); the destruction envelope is the binding constraint | resolved (ladder line, closing table) |
| C2-15 | gpt-oss MoE routing is a language/register instrument (ES-vs-EN JS 0.090-0.109), not a novelty detector | negative-result (router work → future) |
| C2-16 | LoRA damage is anchor- and lr-independent; intrusion falls with model scale (0.6B 12.5% → 4B 7.5% → 14B 2.5%) | absorbed (Law 2 scale-dilution; C2-24) |
| C2-17 | Fisher metric (top-k teacher-token weighting) amplifies catastrophic remembering: intrusion 57.5%, worst of the project | ablation-class; absorbed (Law 7) |
| C2-18 | intrusion is seed-noisy at n=40; capability damage replicates tightly | resolved (bait set grown to n=200) |
| C2-19 | 1.7B full-FT is MORE destructive than 0.6B at matched recipe (intrusion 27.5%, hs −8.0) | resolved; feeds C3 item 9 |
| C2-20 | ch16_av2 passes recall (0.065) and intrusion (10.0%) but fails capability — the envelope is capability-shaped | resolved (rung × anchors grid complete) |
| C2-21 | at 4B a trained window alone recites (0.013) but intrusion 25.0%: concentration near the output buys recall at the price of intrusion | ablation-class (boundary stone; tail work → ../selfupdate_kd) |
| C2-22 | in-window trajectory mimicry installs the intrusion groove; mimicry-free window 2.5% (tailpure) at identical body training | absorbed (Law 3) |
| C2-23 | depth-biased lens-CE grooves even harder with better data (lensdeep2 intrusion 50.0%, procedural +2.51) | ablation-class; absorbed (Law 7) |
| C2-24 | the LoRA axis closes: anchors, lr, and rank (r64 bench_min −18.0) all fail; scale vehicle = full-FT + offload_adam | negative-result (axis closed) |
| C2-25 | dilution does not rescue 1.7B full-FT (combined_1p7b intr 27.5%, bench −9.5) | resolved; feeds C3 item 9 |
| C2-26 | THE CONNECTIVITY LAW: k-sweep — recall by k=4 (0.009), cleanliness + hidden-primacy by k=8 (74.9% hidden share) | absorbed (Law 4) |
| C2-27 | teacher-stream inputs never teach self-driving (tcmodern 0.877); storage pre-pass role remains | negative-result; C3 item 1 |
| C2-28 | aux-100% control fails BOTH jobs (recall 0.795, worst cat +8.70): the synergy is real | absorbed (Law 5) |
| C2-29 | teacher ceiling 0.650 at 0.6B, flat 0.449-0.666 across 0.6-14B; students exceed teacher behavior 30-70× — information travels via states + reference text | absorbed (Law 6) |
| C2-30 | slide8pure is the crown: 0.007 / 99.3% / worst cat +0.18 / CLEAN | resolved (THE RECIPE) |
| C2-31 | lens_kl HIDDEN-loss family is an intrusion catastrophe (uniform variant 90.0%); vocab_mse/nmse safe | absorbed (Law 7) |
| C2-32 | teacher_kl starves without a premise-checked sharp-reader teacher (thinkselkl 0.838) | absorbed (Law 8) |
| C2-33 | ragchannel heldout: WEAK POSITIVE state-channel transfer (0.847 vs base 0.914; first exact lines) | resolved; C3 item 4 |
| C2-34 | the last-3% law: pure KL converges to the teacher's own label agreement (97.3%); verbatim recall lives in the residual (~7 replications) | absorbed (Law 9) |
| C2-35 | disjoint k=8 windows collapse (0.810) | RETRACTED-then-resolved: the collapse was the teacher_kl confound; pinned disjoint recalls 0.023 / intr 7% clean — overlap is an optimization, not a requirement |
| ft4b | vehicle verdict: offload_adam full-FT 4B recalls 0.011 (7× the LoRA r16 0.073) BUT the C1-hybrid recipe it carried is DESTR (worst cat +0.86, intrusion 50%) — vehicle solved, crowned-recipe-at-scale pending (closes the closing-table "in flight" footer) | resolved; feeds C3 |
| attrib | gradient-share attribution: slide8 74.9% hidden share vs classical hybrid 49.5% (parity) — uniform windows shift the gradient toward trajectories | absorbed (Laws 4/5; column in closing table) |

## Governance records (owner directives / integrity ledger)

### RECLASSIFICATION (owner standard, 2026-07-05 ~04:30) — readout-source column
Every arm trained before this timestamp used a task-label CE readout
(reference-text supervision) — under the refined standard these are
BASELINES, not the method, however clean their batteries (slide8,
slide4, slide2, tailpure, final_k8, all wave-K/M/N arms). The method's
readout is teacher_kl (teacher's context-conditioned logits; no
reference text in any gradient). Unaffected: within-family contrasts
(mimicry C2-22, connectivity trend), all storage results (strict/tc arms
carry no CE), lensonly, the ceiling instruments. The closing table
carries the readout-source column; the conditional headline resolved as
C2-34 / Law 9 (crown = hybrid by lab necessity, deployment-pure by
transcript equivalence).

### ERRATUM (2026-07-05 ~05:15; applies to all C1 + pre-05:00 C2 arms)
Training-side readout terms (tail_ce, lens_ce) targeted the ORIGINAL
TEXT (task labels), not teacher outputs — disclosed in mechanism but
misframed as "the method"; correctly classed as task supervision inside
a hybrid. Eval-side use of the original text is CORRECT and unchanged
(recall is measured against the reference; owner-confirmed). Scope: ~25%
of gradient in recipe arms (attribution-measured). Bound on effect:
teacher-forced-with-context agrees with labels at 96.8% top-1 / 0.226
nats (n=16 v4 items); the purification arms slide8kl + thinkselkl
quantified the residual (→ Law 9). Unaffected: hidden-loss/storage
results, strict/tc arms, anchor-KL, contrast-based laws, metrology,
ceilings. paper1 carries the erratum + readout-source column;
terminology purged repo-wide (reference / task_label).

### DAMNATIO MEMORIAE (owner directive, 2026-07-05 ~06:00)
The schedule formerly at the expunged markers is removed from code,
configs, docs and papers: its readout CE targeted the ORIGINAL TEXT
unconditionally (the ce_kind knob never reached it), so none of its
results were valid even as baselines — task-supervised runs wearing a
distillation label. teacher_censored is RESTORED to its original
definition (stationary teacher-stream, fully parallel, no window, no CE
— the drift entered at fe3201d for "like-for-like" comparability).
Refusal guards make both states unrepresentable: the expunged schedule
name raises; teacher_censored + tail knobs raise.

### THE CONFOUND STRUCK TWICE (final-hours ledger correction, 2026-07-05 ~15:30)
The tail_ce_kind default flipped mid-campaign (04:45). FIRST STRIKE: the
05:07 lw_q_pp2 repro silently inherited teacher_kl — its 0.837 / 96.5%
teacher-forced accuracy was the last-3% fingerprint, not a parallelism
defect (the hook-crossing gradient hypothesis was unit-tested and
REJECTED; honest repro → Law 10, closed 2026-07-10). SECOND STRIKE:
every final-slate arm cloned from the crown's config (which never pinned
the knob) also ran teacher_kl: s43, slide6pure, slide8disj,
q_ch8_slide8pure, thinkslide, slide8pure_1p7b, and the xs spectrum.
Consequences, honestly:
- C2-35 RETRACTED as stated (→ RESOLVED 2026-07-10, verdicts below).
- Relabels: slide6pure → k6-KL point (0.838, CLEAN 3.5% — last-3% at
  k=6); s43 → slide8kl seed-43 replication; ch8crown → ch8-KL (0.758
  flat — last-3% ON PROSE, mildly softer collapse); thinkslide →
  think-KL (0.858 — premise condition); slide8pure_1p7b → slide8kl-1.7B.
- The crown's seed claim reopened (→ CLOSED 2026-07-10: replicated,
  verdicts below).
- Tooling fix (this cannot recur): tail_ce_kind code default is now an
  UNSET sentinel; the validator refuses any windowed run without an
  explicit choice; base.yaml carries the doctrinal default EXPLICITLY;
  all crown-family + spectrum configs pin task_label. 89/89 tests.
Lesson upgraded: "defaults are experiment variables" is now ENFORCED,
not remembered. Both confounds produced valid science (the last-3% law
gained ~7 independent replications) only because the config dumps made
them detectable — the dump IS the lab notebook; the sentinel makes it
sufficient.

### Inheritance verdicts read 2026-07-10 (C3 queue item 0)
- **Seed claim REPLICATED**: lw_r_s43_pinned CER 0.0076 / line-exact
  0.991 / intrusion 1.5% (n=200) — matches the seed-17 crown
  (0.007/0.993/2.5%). Two seeds now carry the crown recipe.
- **C2-35 RESOLVED**: lw_r_disj_pinned CER 0.023 / intrusion 7% /
  non-destructive verdict — pinned disjoint windows recall AND stay
  clean; the 0.810 collapse belonged to the teacher_kl confound.
  Sliding overlap still wins on both axes; overlap is an optimization,
  not a requirement.
- **PP2 blocker CLOSED** (Law 10 updated in place): lw_q_pp2fix CER
  0.011 / line-exact 0.988; trainer certified under pipeline_split
  against single-device references (certs/pp2).
- **xs 1.7B spectrum**: recall trend holds (slide2 0.043, slide4
  0.014; fisher fails at 0.861 with 26% intrusion — loss-safety at
  1.7B) BUT intrusion stays high across the spectrum (22.5-39.5% vs
  0.6B crown 1.5-2.5%): cleanliness at 1.7B is NOT yet reproduced —
  candidate C3 question (anchor breadth? window/params-per-item
  scaling?). Verdict numbers in runs/xs_*/eval/destruction.json.
- **lw_r_crown17_pinned NEVER RAN** (empty dir): its config YAML was
  deleted by the 2026-07-05 purge before the scheduler reached it, and
  the crown recipe's task_label readout no longer exists in this
  branch's trainer — reconstructing it would reintroduce a
  reference-text training term the branch law forbids. OWNER DECISION:
  port under an explicit ablation flag, run it in ../selfupdate_kd, or
  drop the 1.7B mimicry arm.

## C3 queue

0. DONE 2026-07-10 — inheritance verdicts read (section above).
1. Teacher-stream k-windows (storage-side; docs/windows.md taxonomy).
2. Premise-gated thinking teacher_kl (carry the passage in the thinking
   TEACHER sequence, or premise-gate per item — the C2-32 fix).
3. Qwen3.6-27B bridge grid + Gemma4-E4B as C2modern-arm-0
   (embed-scaling adapter prerequisite — see ladder notes below).
4. Wide-channel ragchannel (hold labels, boost passage exposure — the
   discriminating version of C2-33).
5. DONE 2026-07-10 — trainer refactor (docs/runtime.md; certify gate).
6. Reincarnation.
7. DONE — intrusion prompt set grown to n=200 (crown 2.5% ± 2.2).
8. DONE 2026-07-10 — evaluate.py --layer-residuals.
9. NEW — 1.7B cleanliness: the xs spectrum recalls but runs 22.5-39.5%
   intrusion vs 0.6B's 1.5-2.5% (anchor breadth? window/params-per-item
   scaling?).

Pending OWNER DECISION: lw_r_crown17_pinned (see inheritance verdicts).

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

### Parallelism status (0.6B, seq 600, training hot loop, quiet pair)

single 57.9 ms/item · PP2 61.4 (+6%) · TP2 99.7 (+72%). PP2 training is
certified (Law 10; certs/pp2). TP2 runs mechanically through the block
walk (tp_plan="auto", no DTensor surgery) but is probe-only by policy —
correctness unvalidated. At 0.6B the collectives dominate (+72%); PP2's
+6% confirms activations-across-PCIe is cheap at batch 1. The 27B bridge
is where TP's fat matmuls would amortize — harness:
scripts/parallel_bench.py. offload_adam: full-FT 4B at 2.3 s/item;
streamed pinned paging (2026-07-10) cut step overhead 2.65x at 0.6B.
