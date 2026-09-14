# Local teacher-state learning and passage recall: project close-out

14 September 2026 · Codex Sol · branch `lwteacher`

## Finding

This project demonstrated that transformer blocks can optimize teacher-sourced
local objectives with gradients confined to each block, while retaining a frozen
embedding and vocabulary stack. It did not demonstrate a reliable method for
converting retrieved passages into independently usable model memory at the
campaign's specified effect size. Local loss reduction repeatedly exceeded the
improvement in passage-absent generation, and sometimes accompanied substantial
behavioral damage.

The result is not an absolute null. The final v6 campaign contains two modest
whole-set improvements: Gemma causal-residual reached a content gain of 0.0181
at its last measured epoch; Qwen partial-teacher seed 43 completed 40 epochs with
a gain of 0.0206. Their paired item intervals exclude zero. Neither satisfies
the specified minimum gain of 0.03, and no method has a successful matched-budget
replication. The strongest Qwen gains concern cloze completion rather than
general continuation of the missing text.

The final outstanding job completed successfully on 8 September at 16:11:44
CEST. The user's Slurm queue was empty at the 14 September close-out. This
document supersedes the provisional 7 September report, in which that job had
only reached epoch 17.

## Question and method genealogy

The scientific question was whether a model could internalize information
available to a passage-bearing teacher through block-local learning, without
training its vocabulary head or propagating gradients through the whole stack.
The decisive deployment condition removes the passage and asks the student to
generate the corresponding content using its own predicted prefix.

Three stages must remain distinct:

| Stage | Input to block L during training | Teacher target | Main lesson |
|---|---|---|---|
| Supported v4.6 runtime | Detached teacher state at L−1, with teacher attention context | Teacher state at L | Structural locality enables independent block ownership; local numerical success does not establish passage-absent recall. |
| Experimental v5 monolith | Detached current censored-student state | State from the passage-bearing teacher trajectory | Optimization can correct a local surrogate while damaging behavior or leaving content recall flat. |
| Experimental v6 monolith | Detached current censored-student state | Same-input passage-visible block output, or a partial-passage teacher trajectory | Better causal targets and measurement reveal small, task-dependent gains, still below the promotion criterion. |

The August campaign explored ranks, objectives, layer-selection schedules and
long horizons. Its detailed evidence and known defects are recorded in the
[August review](august_2026_gemma_memorisation_failure.md). That record supports
the history above; its heterogeneous metrics are not pooled numerically with
v6. Historical readout experiments are also excluded from the v6 conclusion.
The project does not claim those earlier controls form a single uniform study.

V6 tested two explicit laws. In `causal_residual`, the frozen block receives the
same detached current input with passage access visible and blocked; the
adapter-on blocked block learns the visible output. This isolates a local
passage-access contrast at the supplied state. In `partial_teacher`, the frozen
teacher sees the passage only at a previously selected retrieval-layer set,
and the fully blocked student learns its intermediate outputs. Neither law
changes the supported v4.6 runtime; both live in the separate
[trainv6.py monolith](../scripts/trainv6.py).

Training retains passage slots and positions. Softmax queries after the passage
cannot attend to its keys. Qwen additionally zeros passage content at the
GatedDeltaNet token mixer while retaining recurrent decay and the sequence clock.
Deployment evaluation removes the passage and uses natural positions. Thus
there remains a training/deployment distribution difference, even though v6
also records aligned-position generation as a diagnostic.

## Campaign and measurement

All five scientific runs used source commit
`48da673c5a5b4209a5865f2500fd33e5f3f779c8`, four H100s per job, 2,071 items,
LoRA rank 32 and alpha 64, learning rate 1e-5, microbatch 1, accumulation 4,
gradient clipping 0.01 and weight decay 0.01. Every decoder layer receives the
same scalar objective weights: normalized hidden Huber 1.0, frozen-final-norm
and frozen-head Jensen–Shannon loss 0.05, and prompt anchor 1.0. Embeddings,
final norm and head remain frozen. Gemma has 60 decoder layers and 244,858,880
trainable adapter parameters; Qwen has 64 layers and 233,455,616.

The primary metric is canonical content longest-common-subsequence recall in
greedy, passage-removed generation, using the trainer's sufficient-answer-length
budget. A 96-token budget is a separate secondary score. The content reference
comes from the corpus; teacher wording is separately scored. Epoch zero is the
untrained student under the same evaluated condition. The intact-RAG teacher is
a separate control: its whole-set content recall is approximately 0.989 for
Gemma and 0.978 for Qwen, through the trainer's generation path.

The fixed panel contains 24 items per corpus/chapter stratum, 120 in total.
The whole set contains 1,490 Machado items and 581 across four Quijote chapters.
Consequently, an unweighted panel mean and a whole-set item mean have different
corpus weights. The seed also selects the panel; seed 17 and seed 43 panels are
not identical. Panel differences cannot be attributed solely to training seed.

Whole-set generation runs at epoch zero, at the planned terminal epoch, and
when the panel triggers a candidate review or a scientific-stop confirmation.
Paired bootstraps use 4,000 item resamples against the same run's epoch zero.
These intervals describe item variability on the training corpus, not unseen
generalization or between-training-seed uncertainty. Repeated interim looks and
overlapping text windows further limit confirmatory interpretation.

Cross-entropy (log loss) against stored teacher answer tokens, `CE-eval-loss`,
and Kullback–Leibler divergence from the frozen teacher output distribution,
`KL-eval-loss`, cover every teacher-realized answer token after each completed
epoch. The counts are 78,810 for Gemma and 70,807 for Qwen, always 2,071 items.
These are teacher-forced output-distance evaluations; they have optimizer
weight zero and never enter backward. They are distinct from the block-local
Jensen–Shannon training objective. ARC-Easy, ARC-Challenge and HellaSwag each
use 100 vendored items, scored by normalized option log likelihood.

Promotion requires whole-set primary content gain at least 0.03 with a positive
interval, new recitation or prefix gain at least 0.03, argmax acceptance at
least 80% of epoch zero, standard macro delta above −0.03 and worst-task delta
above −0.05, plus a second training seed. The trainer logs replication as
pending; it does not establish cross-run replication automatically.

## Final results

The table uses the last available whole-set primary generation measurement,
except Qwen causal, for which only the panel has a post-training measurement.
Epochs are explicit because three jobs did not reach their planned endpoint.
Numbers are absolute score changes, not relative percentages.

| Model / law / seed | Training endpoint | Generation evidence | Primary content gain [95% item interval] | Decision |
|---|---|---|---|---|
| Gemma / causal / 17 | e24, 49,704 items; timeout | Whole set, e24 | +0.018104 [+0.012251, +0.023815] | Small improvement; below required effect; unreplicated. |
| Qwen / causal / 17 | e29, 60,059 items; timeout | Panel, e28 | −0.018390 [−0.033049, −0.004124] | No demonstrated recall gain; terminal whole-set result unavailable. |
| Gemma / partial / 17 | e8, 16,568 items; scientific stop | Whole set, e8 | +0.004830 [−0.002493, +0.011837] | Flat content with capability damage. |
| Qwen / partial / 17 | e34, 70,414 items; timeout | Whole set, e28 | +0.003183 [−0.006003, +0.012376] | Aggregate gain not established at this measured checkpoint. |
| Qwen / partial / 43 | e40, 82,840 items; completed | Whole set, e40 | +0.020601 [+0.011062, +0.030073] | Measurable gain; point estimate below 0.03. |

Qwen partial seed 43 materially improves the provisional September 7 reading.
Whole-set content rises from 0.226687 to 0.247287; recitation rises from 0.009174
to 0.031869, or 19 to 66 of 2,071 items. Correct-prefix fraction rises from
0.058851 to 0.085951. At 96 tokens, content gain is 0.029161 with interval
[0.020093, 0.038044]. That secondary result cannot replace the primary metric.
The primary interval's upper endpoint exceeds 0.03, so the data do not prove
the underlying effect is smaller than 0.03; they fail the registered observed
effect requirement.

The task decomposition is more revealing than the average. At Qwen seed 43 e40,
cloze content improves by 0.206325 [0.160141, 0.256526] over 249 items, while
next continuation changes by −0.009109 [−0.017872, −0.000165] over 1,490 items;
previous-context tasks change by 0.014640 [−0.003918, 0.033578] over 332 items.
Seed 17's e28 whole-set result has the same qualitative pattern: cloze
+0.144578 and next continuation −0.017862. This is evidence of task-specific
learning, not uniform recovery of absent passages. The unequal epoch budgets
prevent a completed matched-seed replication claim.

Gemma causal's last whole-set gain is concentrated in Quijote: chapter gains
range from 0.0358 to 0.0606, versus 0.0060 on Machado. Its best observed
whole-set gain was 0.022279 at e14, then 0.018104 at e24. Recitation is nearly
unchanged, from 0.009174 to 0.009657, and prefix fraction declines from 0.049269
to 0.046484. Content overlap improvement alone therefore overstates its value
as a memorization method.

| Model / law / seed | Local loss e1 → last epoch | CE-eval e0 → last | KL-eval e0 → last | Argmax e0 → last | Standard macro delta |
|---|---|---|---|---|---|
| Gemma / causal / 17 | 0.004050 → 0.003435 | 5.2327 → 6.7360 | 5.2142 → 6.7190 | 0.4674 → 0.4110 | +0.0100 (e24) |
| Qwen / causal / 17 | 0.000412 → 0.000205 | 2.1795 → 2.1851 | 2.1436 → 2.1493 | 0.5584 → 0.5576 | 0.0000 (e28) |
| Gemma / partial / 17 | 0.053576 → 0.033594 | 5.2327 → 6.8975 | 5.2142 → 6.8805 | 0.4674 → 0.4097 | −0.0367 (e8) |
| Qwen / partial / 17 | 0.082748 → 0.041398 | 2.1795 → 2.6297 | 2.1436 → 2.5984 | 0.5584 → 0.5771 | 0.0000 (e34, rounded) |
| Qwen / partial / 43 | 0.082715 → 0.040025 | 2.1795 → 2.4154 | 2.1436 → 2.3842 | 0.5584 → 0.6018 | −0.0067 (e40) |

Qwen partial can improve argmax accuracy while worsening cross-entropy and
divergence. These metrics measure different properties: correct top-token
ranking does not imply improved probability calibration or distributional
agreement. Small standard-benchmark changes on 100-item tasks do not prove
general capability preservation beyond these measured conditions.

## What the gradients establish

The table reports the arithmetic mean of the logged per-layer gradient norm
shares at the last measured attribution epoch. Each layer's shares normalize
the separate weighted objective norms. These are neither a decomposition of
the final vector update nor causal attribution of behavior; gradients may
align or cancel. Measurements use five fixed probe items, one per stratum.

| Model / law / seed | Epoch | Hidden | Lens JS | Anchor |
|---|---:|---:|---:|---:|
| Gemma / causal / 17 | 24 | 38.38% | 39.27% | 22.35% |
| Qwen / causal / 17 | 28 | 54.02% | 4.47% | 41.51% |
| Gemma / partial / 17 | 8 | 41.61% | 50.22% | 8.17% |
| Qwen / partial / 17 | 34 | 79.91% | 10.31% | 9.79% |
| Qwen / partial / 43 | 40 | 81.24% | 10.43% | 8.32% |

Equal scalar weights give very different gradient mixtures across models.
Gemma partial's large lens share is a hypothesis for further diagnosis, not
proof that this term caused its damage. Zero-gradient probes must remain
undefined in share reporting; they are not evidence that an entire model or
campaign has no causal passage signal. The causal-effect profiles contain
nonzero measurements.

## Reliability, limits and operational closure

The two causal seed-17 jobs (448703, 448704) and Qwen partial seed 17 (448705)
exhausted 48-hour allocations. This was an inadequate wall-time budget for the
full training-plus-evaluation schedule. Gemma partial (448706) completed with
the explicit scientific-stop marker after the 12,000-item minimum: loss had
improved, content was flat and capability had deteriorated. Its seed-43 repeat
and the blocked causal repeats were cancelled without running. They provide
no replication evidence.

Changing job 448709's dependency to `afterany:448705` allowed the independent
Qwen seed to start after the timeout. It completed in 51 hours 10 minutes with
a four-day allocation. No successor remains queued. The saved checkpoints are
Gemma causal through e22, Qwen causal through e28, Gemma partial through e8
plus final, Qwen partial seed 17 through e32, and seed 43 through e40 plus final.
Metrics emitted after the last saved checkpoint remain observations, but do
not imply that those weights were saved. In particular, seed 17 e34's panel
gain of 0.037623 had no completed whole-set confirmation or e34 checkpoint.

Launch certifications reported passage-replacement invariance, expected hook
coverage and zero gradient leaks on their probes. They support execution
integrity within the probes' scope; they are not an exhaustive mathematical
proof. The final five metric files contain 638 JSON records. The close-out audit
verified all 203 referenced generation-artifact hashes and item counts, along
with the CE/KL evaluation-only flags and whole-set token counts. No new model
evaluation was run to construct this report.

The conclusion remains bounded by two model families, one corpus collection,
limited seeds, unmatched terminal budgets, training-set evaluation and the
specific objectives and hyperparameters used. The observed failure to meet
promotion does not establish impossibility of local memory learning, absence
of latent stored information, or a uniquely identified cause. Teacher-forced
prefixes, target reachability, position changes, and the mismatch between
local distance and autonomous recall remain plausible explanations requiring
separate experiments.

The durable outcome is a tested distinction: block-local optimization and
mechanical locality are achievable, but neither guarantees retrieval of the
missing content. V6 repairs several August measurement gaps and demonstrates
limited cloze learning; it does not supply a broadly successful consolidation
recipe. This round closes without promoting a method or queuing further runs.

## Evidence and provenance

The tables summarize existing in-pipeline measurements. Run artifacts are
disposable under repository policy and are not added to Git by this close-out.
The following exact metric-file digests identify the local evidence inspected;
they identify bytes but do not replace an external archival copy if the raw
results are needed for publication.

| Run under `runs/` (file: `metrics.jsonl`) | SHA-256 |
|---|---|
| `v6_g31_cr_s17_r8` | `e3240754dcd5d11749a1a7c00fff140aa3364ad4797a8da35ab73bb2ab22a6de` |
| `v6_q27_cr_s17_r8` | `65fc372af77aa8eb1844b42fdd8144fb5e63416813087ae11813e6cf8edd9961` |
| `v6_g31_pt_s17_r8` | `6818800fc713eb9b551b442bb38dde813a393d039e29c10887d2e34efcfc5322` |
| `v6_q27_pt_s17_r8` | `8cefca976573ad00ee16927c44b98433ab382aa41cebbbaafa019768ede51ca9` |
| `v6_q27_pt_s43_r8` | `82efa964f43a883aa92e6e7736ac8278613ce5452b46ca638aa083163384a9f8` |

The model revisions are Gemma
`b9ea41a2887d8607f594846523f94c6cc75ac8a4` and Qwen
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Each run's provenance record also
pins dataset, response, retrieval-evidence and monolith hashes. See the
[v6 protocol](trainv6_monolith.md), [v4 runtime](runtime.md), and
[August review](august_2026_gemma_memorisation_failure.md) for the method and
historical evidence boundaries.
