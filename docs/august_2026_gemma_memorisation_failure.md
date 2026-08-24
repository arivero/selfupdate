# August 2026 review: why the Gemma training method did not memorise

**Cut-off:** 24 August 2026

**Code reviewed:** `145f45c` (`lwteacher`)

**Scope:** the August Gemma campaign, including the supported pipeline-v4.6 method, the experimental `trainv5.py` method, raw run metrics, the 16 August and 23 August reports, and the attention/censorship probes.

## Executive verdict

Gemma did not fail because the adapters were too small, the run was too short, or one learning-rate setting was unlucky. The campaign repeatedly demonstrated ample optimization capacity: local losses fell sharply, ranks 8 through 128 behaved similarly, and a fixed layer could reduce its assigned loss by roughly two thirds without damaging the base model. What did not improve was the thing that matters: new, content-specific recall from a prompt in which the passage was absent.

My conclusion is that the project optimized the wrong local surrogate in two opposite ways:

- Pipeline v4.6 makes the local problem nearly trivial. Every trainable block receives the detached, uncensored teacher state and frozen teacher attention context, then learns to reproduce the next teacher state. This is a valid locality experiment, but it supplies little pressure to make the deployed censored student reconstruct missing information.
- `trainv5.py` repairs that triviality by feeding each block the numerical censored-student trajectory, while still targeting the uncensored teacher state. That creates a different failure: every selected block is asked to repair a cumulative cross-trajectory gap at teacher-forced answer positions. The easiest solutions improve hidden geometry, answer style, or local continuation behavior; they need not store the missing passage in a form that survives natural autoregressive generation. Updating many blocks destroys the model, while selecting one high-loss block merely concentrates this mismatched job.

The decisive empirical pattern is therefore not “the optimizer could not learn.” It is “the optimizer learned what the loss requested, and that request was not behavioural memorisation.” Falling hidden loss alongside flat content recall is the central result of August, not an optimization anomaly to be fixed with another rank, horizon, or gate sweep.

This is a failure of demonstrated **behavioural memorisation**. The evidence does not yet prove that no passage information exists anywhere in the trained activations. A causal storage/readout diagnostic is still needed before making that stronger claim.

## What is current, and what was experimental

The branch contract matters. The only supported runtime is [pipeline v4.6](runtime.md):

```text
i_L = stopgrad(teacher h[L-1])
y_L = block_L_student(i_L; frozen teacher context)
loss_L = HiddenLoss(y_L, teacher h[L])
```

The student’s end-to-end trajectory is explicitly validation-only. By contrast, [`scripts/trainv5.py`](../scripts/trainv5.py) is a standalone experimental monolith. Its pre-hooks detach gradients at every block boundary, but the **values** passed forward are the evolving censored-student values:

```text
x_L = stopgrad(student h[L-1])
y_L = block_L_student(x_L; censored context)
loss_L = local_distance(y_L, teacher h[L])
```

That distinction is not cosmetic. `trainv5.py` violates the present v4.6 dataflow law even though its backward graph is block-local. Its results are useful research evidence, but they must not be described as results of the supported v4.6 method or used to silently redefine that method on this branch.

## What `trainv5.py` actually trains

I read the complete 1,530-line file because the failure is in the interaction of its data construction, hooks, loss, gating, and evaluation—not in any one flag.

For every item, the code joins a stored vLLM response to the source example, removes the privileged passage by text surgery, verifies an exact tokenizer round trip, and appends the stored teacher answer tokens to both the uncensored teacher prompt and censored student prompt. Training is therefore teacher-forced over every answer token. At the predictive row for answer token `a_j`, both teacher and student have already been given the true tokens `a_<j`.

During the teacher pass, adapters are disabled and the full passage is present. Raw output states from every decoder block are cached at the answer’s predictive rows. During the student pass, adapters are enabled, the passage is absent, and a pre-hook detaches every block input. The numerical output from block `L` still becomes the numerical input to block `L+1`; only the gradient is cut. Each local backward can consequently update only its own block, while its target remains a fixed state from a different, passage-bearing trajectory.

The default self-anchor has the same weight as the answer loss. At deterministic prompt rows, it asks the adapter-on censored model to reproduce adapter-off censored base states. This constrains collateral drift, but it is not a memory signal.

The strongest August configuration used `delta_cosine` and `topk_abs:1`. In the implementation,

```text
student_delta = y_L - x_L
teacher_delta = teacher_h[L] - x_L
loss = 1 - cosine(student_delta, teacher_delta)
```

Despite the name, `teacher_delta` is not the teacher’s own block increment. It is the entire remaining gap from the current censored-student input to the uncensored teacher output. `topk_abs:1` selects the block with the largest absolute answer loss on each step, and in the long runs selection collapsed mainly onto layer 54.

The later `tinc_*` losses are the genuine teacher-increment variants:

```text
student increment = y_L - student_h[L-1]
teacher increment = teacher_h[L] - teacher_h[L-1]
```

They have a compositionally sensible fixed point, but both dense and gated runs were dynamically destructive. A correct fixed point is not enough when the training trajectory starts far off that manifold.

Finally, the important train/deploy mismatch is explicit in the command-line contract. Most strong runs use `--positions aligned`, which assigns post-removal tokens the absolute/RoPE positions they would have had if the passage were still present. The recall evaluator always calls ordinary Hugging Face generation, which derives natural positions after passage deletion. Thus the best loss was optimized under a position system that deployment never uses, while no matched aligned-generation diagnostic was recorded.

## August chronology and what each stage established

| Date | Work and decision | What survives scrutiny |
|---|---|---|
| 1 Aug | Absolute top-k layer selection examined. | Absolute surprise is strongly depth-biased and can collapse into a tail-like selector. This was an early warning, not a solution. |
| 6 Aug | `delta_cosine + topk_abs` became the best apparent v5 arm. | It preserved more base behaviour and raised full-answer word-LCS slightly, but did not establish content recall. |
| 12 Aug | Gemma-4-26B v4 student-K/V refresh versus frozen K/V closed. | The two trajectories were essentially alike. Stale K/V was not the binding explanation for v4’s flat recall. |
| 16 Aug | The first joint v5 PDF reported 16 runs; v5w2 added alternative losses, DoRA, norms, ranks, teacher-increment targets, chapter telemetry, and recitation. | Capacity, rank, and simple loss-family explanations were tested broadly. The PDF’s checkpoint-selection statement was arithmetically wrong; details are below. |
| 17–19 Aug | Twelve screens and long horizons completed; rank, `delta_vmse`, dense/gated `tinc`, and DoRA were closed. A fixed-layer and gate-freeze line tested churn. | Local loss continued to improve while content did not. The campaign correctly changed its conclusion from “small recall gain” to a null-first result and closed v5w2 at the objective level. |
| 19 Aug | `--train-norms` teacher invariance bug fixed; raw generations began to be logged; teacher-ceiling probe queued. | Norm results before the fix are contaminated. Content-level inspection became possible only for late runs. The exact-path teacher ceiling was still pending at this review’s cut-off. |
| 20–22 Aug | Teacher attention and layer-censorship probes developed. The v2 conclusion was retracted after a masking confound; v3 identified eight Gemma global-attention layers as sufficient and necessary under teacher forcing. | The v3 correction was good scientific practice. It localizes direct retrieval dependence, but does not yet license the proposed training target as strongly as the final report claims. |
| 23 Aug | Final four-page campaign report published; GPU addenda left queued. | The final report correctly centers the behavioural null. Teacher-ceiling, Qwen controls, and ALIA work remained pending and are not evidence in this review. |

## The empirical case

### Pipeline v4.6 was behaviourally flat

The Gemma-4-31B r16 stage-0 run moved teacher-forced argmax agreement only from `0.46785` to `0.46806` by epoch 50 while cross-entropy worsened from about `10.16` to `10.21`. The r64 run ended at `0.46777` with cross-entropy around `10.24`. Multiplying trainable capacity did not create a behavioural signal.

On Gemma-4-26B, K/V-refresh and frozen-K/V runs followed the same deterioration over roughly 200–233 epochs: argmax fell from about `0.45` to `0.36` and cross-entropy rose from roughly `6.3` to `7.7`; recall remained zero. This closes the attractive but unsupported theory that a stale attention cache was the main bottleneck.

### Dense v5 could reduce loss by destroying the model

The initial Huber run at `1e-4` is the cleanest warning. In one epoch, teacher-forced argmax agreement fell from `0.429` to `0`, cross-entropy rose from about `11` to `123`, and recall became zero, even as the local loss fell from `0.0444` to `0.0280`.

Reducing the learning rate to `1e-5` and adding the prompt anchor delayed but did not change the mechanism. By epoch 2, cross-entropy had actually improved (`11.16` to `9.90`), yet argmax agreement fell to `0.133` and recall roughly halved. Later epochs destroyed the model while the assigned local loss continued downward. Dense vocabulary-metric arms sometimes crossed their own full-LCS baseline briefly before the same collapse.

This anti-correlation is fatal to the interpretation that local loss is a noisy but fundamentally correct proxy for recall. A useful proxy can be imperfect; it cannot repeatedly reward catastrophic loss of the target behaviour.

### The best gated arm hit a content wall

In [`v5p_dcos_topkabs1_long`](../runs/v5p_dcos_topkabs1_long/metrics.jsonl), epoch zero had Machado recall `0.1731`, pooled Quijote recall `0.1775`, teacher-forced argmax `0.4213`, and ARC-Easy `0.32`. Machado peaked at `0.1921` around epochs 160–170, but pooled Quijote was only `0.1556–0.1593`, argmax had fallen to about `0.23`, and ARC-Easy to `0.21–0.22`. By epoch 300, Machado was `0.1738`, Quijote `0.1672`, argmax `0.2369`, and ARC-Easy `0.20`. Meanwhile local loss fell from `0.4433` to `0.3369`, and the answer-only layer profile mean fell from roughly `0.442` to `0.273`.

The longer r64 run, [`v5w2_dcos_tka_r64_h400`](../runs/v5w2_dcos_tka_r64_h400/metrics.jsonl), repeated the pattern. Local loss fell from about `0.443` to `0.320`; 50 of 60 layers improved and layer 54 fell from about `0.706` to `0.370`. Yet epoch-zero Machado/Quijote recall of `0.1715/0.1885` ended around `0.1661/0.1762` at epoch 320, argmax fell from `0.4213` to `0.255`, and ARC-Easy from `0.32` to `0.20`.

Ranks 8, 32, 64, and 128 all reached the same approximate `0.19–0.20` full-LCS wall and then faded or damaged the base model. This is strong evidence against a LoRA-rank bottleneck.

### Fixing layer 54 isolated stability, not memory

The fixed-layer run [`v5w2_dcos_f54`](../runs/v5w2_dcos_f54/metrics.jsonl) is the campaign’s most informative ablation. By epoch 18, layer 54’s assigned loss fell from approximately `0.703` to `0.224`, while teacher-forced argmax and ARC-Easy remained close to epoch zero. Pooled Quijote full-LCS rose from `0.1869` to `0.2006`.

That is real evidence that dynamic gate churn contributed to damage and that one Gemma block can optimize the local task stably. It is not evidence of memorisation. The paired full-LCS change was small and statistically fragile (an item bootstrap gives an approximate interval of `-0.003` to `+0.031`), recitation did not gain a new item, and the logged generations changed formulaic Spanish scaffolding rather than reproducing passage-specific text. Layer 54 is best interpreted as a **loss/drift attractor immediately after a global-attention layer**, not a demonstrated memory address.

### The content-sensitive result is effectively null

The final report’s best synthesis is the content-only movement from roughly `0.12` to `0.13`, compared with about `0.99` word-LCS in the stored vLLM teacher artifact. Recitation stayed at zero for newly learned items; occasional `0.0417` values trace to items already above the threshold at epoch zero.

There are two reporting qualifications. First, the current trainer computes only full-answer word-LCS and recitation; the content-only parser was session analysis, is not committed, and could only inspect runs after raw generation text began to be logged on 19 August. “About 40 arms” therefore combines heterogeneous evidence rather than one reproducible content metric over every run. Second, the vLLM teacher’s `0.99` is not an exact ceiling for the trainer’s decode path. The recall evaluator caps generation at 96 new tokens; 25 of the fixed 120 evaluation references exceed 96 model tokens, while the teacher artifact allowed a much larger budget. The pending same-path teacher ceiling is necessary.

Neither qualification rescues the method. Machado answers are short and also show no content memorisation. They do mean the final numerical ceiling and cross-arm content aggregate should be re-measured before publication.

## Why the method failed

### 1. The local targets do not express the causal operation we need

The desired operation is approximately: given a current state, add whatever change access to the missing passage would have caused. V4 instead asks a block to map a teacher state with teacher context to the next teacher state. V5 asks it to map a censored-student state to a fixed state from a different trajectory. Neither isolates the block-local causal contribution of passage access.

In v5, `teacher_h[L] - student_h[L-1]` contains every upstream discrepancy: missing passage information, position differences, accumulated adapter drift, ordinary trajectory mismatch, and the teacher’s current block transform. Selecting the largest such vector selects the largest accumulated mismatch, not necessarily the block where a useful fact should be written. Once the selected block changes, all later numerical student states change, so the gate chases a moving target of its own making.

This explains the observed trilemma:

- update every layer and many layers redundantly attempt a cumulative correction, causing destructive over-writing;
- update the largest-loss layer dynamically and selection collapses/churns in the deep stack;
- fix layer 54 and the optimization becomes stable, but only the surrogate improves.

### 2. Teacher forcing gives the model an escape route around memory

Every answer-row loss is evaluated after feeding the true preceding answer tokens. The student can become better at matching the teacher’s register, syntax, and continuation geometry conditional on those tokens without learning to recover the absent passage from the censored prompt. In free generation, the first content error changes the prefix and the model immediately leaves the states on which it was trained.

This is the most plausible explanation for the fixed-layer generations: full-LCS earns credit for shared question wording, boilerplate, or stylistic continuations, while content-only recall and recitation stay flat. The training objective rewards being a good follower of a revealed answer, not necessarily being able to initiate that answer from memory.

### 3. Aligned training positions do not match natural generation

Aligned positions are a defensible diagnostic for comparing states after passage removal, but they can also let an adapter associate the learned correction with absolute positions that never occur at deployment. Because all strong aligned arms were evaluated only under natural generation, the campaign cannot distinguish “nothing stored” from “stored behind the wrong positional key.” This is not my primary explanation—the dense natural-position arms also failed—but it is an unresolved confound in the best gated result.

### 4. The hidden metric is not the claimed frozen-head metric

`trainv5.py` says its formulas were copied verbatim from [`HiddenLoss`](../src/selfupdate/train/losses.py), but its `vocab_mse` computes the raw hidden residual through `W^T W` directly. The current supported implementation first applies the frozen final norm and only then applies the frozen LM head geometry. For Gemma, that difference is material: the deployed token geometry is a function of `final_norm(h)`, not raw `h`.

Similarly, v5’s `delta_vmse` denominator uses `teacher_h[L] - student_h[L-1]`, despite being described as teacher-increment-normalized. It rescales ordinary raw-state `vocab_mse`; it does not measure the teacher’s own increment. The `tinc_*` code is the true own-increment implementation.

Consequently, the August v5 campaign did **not** close the most relevant frozen-vocabulary objective in the current loss stack: final-norm-aware, token-distribution-sensitive local KL/JS or Fisher-weighted matching. It closed the raw Gram surrogate that it actually ran.

### 5. Capacity was abundant but addressability was absent

The script’s compressed-corpus capacity check reports hundreds of times more nominal adapter bits than corpus bits, and increasing rank did not move the recall wall. This rules out parameter count, not usable memory architecture. Parameters become memory only when the objective gives them an address and retrieval path. Here the “address” was a teacher-forced answer state and often an aligned absolute position; the deployed query was a natural-position censored prompt followed by the model’s own prefix. Those are different keys.

### 6. The attention probe identifies retrieval sites, not a complete learning rule

The corrected v3 censorship result is valuable. On a short, teacher-forced Gemma probe, baseline token prediction was approximately `0.999` accurate; allowing passage keys only at the eight global-attention layers retained about `0.990`, whereas blocking those eight fell to about `0.567`, and blocking all passage access to about `0.469`. This supports the claim that direct passage retrieval is concentrated in the global layers.

It does not prove that every non-global teacher target is reachable by a passage-blind student. After a global layer reads the passage, its output carries that information into all subsequent layers; later non-global teacher states inherit it even if those layers do not directly attend to passage tokens. Nor does a teacher-forced token-accuracy result prove free-generation sufficiency. Design E is therefore **promising enough to test**, not “licensed” as an established solution.

## Corrections needed in the August record

These issues should be fixed in prose or metadata before the campaign is cited externally:

1. **The 16 August checkpoint-selection claim is false.** The report says epochs 160/170 maximize recall subject to argmax agreement remaining at least `0.8 × epoch zero`. Epoch-zero argmax is `0.4213`, so the threshold is `0.3370`; epochs 160/170 are around `0.23` and fail it. Only the earliest checkpoints satisfy that gate. The shorter final PDF wisely drops the claim, but the historical report should be marked superseded.
2. **“Copied verbatim” is inaccurate.** V5 `vocab_mse` omits the frozen final norm used by the current `HiddenLoss`, and v5 `delta_vmse` is not teacher-own-increment normalization.
3. **The teacher ceiling is not matched.** The stored vLLM artifact and the trainer’s fixed 96-token Hugging Face evaluation use different decode budgets and possibly different runtime details. The queued ceiling result must be reported before treating `0.99` as the exact denominator.
4. **Content-only recall is not yet reproducible from the repository.** Commit the versioned parser/stopword policy, item identifiers, eligible counts, and outputs. Recompute it for every run with stored texts; label earlier arms as unavailable instead of imputing them.
5. **The v3 probe conclusion is narrower than the report language.** It establishes direct retrieval geography under teacher forcing, not reachability of all layer targets and not generative sufficiency.
6. **V5 evaluation is outside the branch publication contract.** Its teacher-forced CE/KL probe uses a fixed subset rather than every teacher-realized token in the whole training set and does not emit the required evaluated counts/coverage flags.
7. **The pending Qwen launch has a dry-data hazard.** [`trainv5.sbatch`](../scripts/trainv5.sbatch) stages `$TRAIN_MODEL`, but its `--dry-data` gate does not forward the requested model/examples/responses arguments. A cross-model job can therefore validate the default Gemma artifacts rather than the intended Qwen artifacts. Fix this before interpreting a Qwen result.

## Ranked next avenues of search

The order matters. More training is not the first move; first make the behavioural claim measurable, then test a target with the right causal semantics.

### Priority 0 — repair the measurement gate

Before launching another Gemma campaign:

- Complete the teacher-ceiling evaluation through the exact same Hugging Face generation path, tokenizer, prompt, item sample, and scoring code as each student checkpoint. Use a per-item generation allowance sufficient for the stored answer, while retaining the 96-token condition as a separate deployment-budget score.
- Make content-only recall a first-class, versioned trainer metric. Emit item IDs, corpus/chapter, reference and output token lengths, finish reason, full LCS, content-only LCS, recitation, longest correct prefix, and the first divergence. Report eligible-item counts and uncertainty across items and at least two seeds.
- Add an **aligned-position generation diagnostic** without changing natural-position generation as the primary deployment metric. Also run one matched `delta_cosine + topk_abs:1` arm trained with natural positions. This cheaply tests whether the strongest run wrote a position-keyed state.
- Bring CE/KL evaluation into the current whole-training-set contract, with token/item counts and explicit `evaluation_only`, `used_for_backward=false`, and `optimizer_weight=0` fields.
- Correct the August-16 checkpoint claim and mark all pre-19-August content-only values as unavailable unless their generations can be reconstructed exactly.

No method should be promoted because full-answer LCS rises by a few points while content-only recall and longest-prefix recall are flat.

### Priority 1 — train a matched-input causal passage residual

This is the most important scientific successor to v5. For the **same detached current input** `x_L`, evaluate the frozen block twice: once with passage access and once with passage access blocked, then train the local adapter to reproduce their difference:

```text
delta_passage_L(x_L) = F_L_frozen(x_L; passage-visible)
                     - F_L_frozen(x_L; passage-blocked)

target: F_L_student(x_L; passage-blocked)
        ~= F_L_frozen(x_L; passage-visible)
```

This subtracts the ordinary block transform and upstream trajectory mismatch, leaving the causal contribution of passage access at that block and state. Apply the same rule at every layer; the signal will be naturally near zero where passage access has no direct effect, preserving depth-uniform treatment without hand-weighting the tail.

Before training, run two falsification probes:

1. free-generate the frozen teacher when passage access is allowed only at the v3 global-layer set;
2. after each allowed injection, compare partial-teacher and censored trajectories to measure how much passage information persists in later nominally “blocked” states.

The all-layer partial-censorship Design E arm is a legitimate falsification cell. A single-layer global injection is a localization diagnostic, not a branch-compliant training method. Because a same-student-input causal target changes the v4 dataflow contract, it belongs in an explicitly experimental branch or requires an explicit owner-approved protocol change—not a quiet edit to v4.6.

### Priority 2 — remove the true-prefix escape route

Train on the state distribution used at recall time. Generate a short prefix with the current censored student, then condition both sides on that **same prefix**: the passage-visible frozen teacher supplies the target and the passage-blocked student supplies the trainable output. Alternatives are progressive prefix corruption, answer-prefix dropout, or scheduled rollout length.

Start with the short-answer subset so that first-token and longest-prefix effects are easy to interpret, but preserve the project rule that a campaign verdict requires at least 12,000 training items. The decisive readout is whether the first content-bearing token and subsequent self-generated continuation improve, not whether late teacher-forced tokens become easy.

This also requires an experimental protocol change, because it uses student-generated states as local inputs. It should not be smuggled into v4.6 under a new name.

### Priority 3 — test the token-identity-aware local objective that v5 did not test

Use the current, correct frozen measurement stack: final norm followed by the frozen LM head, with a local teacher-distribution objective such as `lens_kl`, `lens_js`, or a bounded top-k/Fisher variant. Apply it uniformly at every depth, keep the embedding/head/final norm frozen, and report gradient-share attribution by layer and by objective.

This avenue is compatible with the present v4.6 law because the measurement remains block-local. It should begin at a small bounded weight alongside the hidden objective and an explicitly legal preservation term, with epoch-zero standard-damage and intrusion tests. Earlier vocabulary losses destroyed behaviour, so this is not a prediction that “more logits will work”; it is a test of token-selective geometry that the raw `W^T W` v5 implementation never actually performed.

### Priority 4 — distinguish storage failure from readout failure without training a readout

Do not revive the forbidden behavioural readout. Use evaluation-only diagnostics:

- item identification or nearest-teacher-state retrieval from trained hidden states;
- correct-answer token margins through the frozen final norm and head at every depth;
- causal activation patching between censored student, passage-visible teacher, and trained student;
- layerwise swaps that ask whether a trained state makes the frozen downstream model recover the right first content token;
- linear probes used only as diagnostic instruments, never as model-training targets or claimed deployment performance.

If passage identity is decodable but frozen downstream generation cannot use it, the next problem is addressable readout. If it is not detectable even at the trained block, the storage objective itself is falsified.

### Priority 5 — test anchoring only after an objective shows content gain

The equal-weight self-anchor may be suppressing useful changes, but current destructive runs show that removing restraint is not a credible first move. Once a causal or token-aware objective yields content gain, compare anchor weights `0`, `0.1`, and `1.0` under a fixed or quota-balanced layer schedule. Add generic-text anchors to measure whether preservation depends on reusing the training prompts. Promote only if the memory–damage Pareto frontier improves.

### Priority 6 — use model controls to test the substrate, not rescue the objective

After repairing the launch gate and metrics, run the matched Qwen control. Qwen’s recurrent/hybrid pathway may provide a more reusable state substrate than Gemma’s sparse global retrieval, but a Qwen success with the current mismatched v5 target would not explain Gemma, and a Qwen null would not validate that target. ALIA/Spanish models should follow only after artifacts and exact-path teacher ceilings are verified.

### Priority 7 — spacing and retrieval practice

The 40–400 epoch massed-repetition runs established a robust peak-and-fade pattern for churny gates. Spaced revisits, interleaved generic text, or explicit retrieval practice are worth testing only after an objective produces a reproducible content-specific gain. Scheduling cannot manufacture a signal that the loss does not reward.

### Priority 8 — expand the optimization vehicle last

Full-block fine-tuning, higher rank, or alternative adapter parameterizations should be last, not next. Rank 8–128 and DoRA already show that the current wall is not simple adapter capacity. A full-rank experiment becomes informative only after the same local target works at small scale; otherwise it merely gives a bad surrogate more power to damage the model.

## Pre-registered success and stop rules for the next campaign

A successor should be promoted only if all of the following hold on the same fixed items:

- content-only LCS improves by at least `+0.03` over that run’s own epoch zero, with an item-level interval excluding zero and replication across seeds;
- at least one genuinely new item crosses the recitation threshold, or the first-content-token/longest-prefix metric improves materially on the eligible short-answer subset;
- natural-position generation is the primary score; aligned-position generation is labelled diagnostic;
- the exact-path teacher ceiling is reported beside the student;
- teacher-forced argmax remains at least `0.8 ×` epoch zero and the full standard-damage battery shows no material regression;
- at least 12,000 training items have been seen before declaring a negative campaign result;
- evaluated item/token counts, whole-set coverage, objective weights, and gradient-share/depth attribution are in the run artifact.

Stop an arm if its local loss improves substantially across two evaluation intervals while content-only recall and first-prefix metrics remain statistically flat and damage worsens. Do not reinterpret a full-LCS scaffold gain as memory.

## What not to do next

The August evidence is already sufficient to reject another sweep of:

- LoRA ranks between 8 and 128;
- longer massed horizons under `delta_cosine + topk_abs:1`;
- learning-rate escalation;
- DoRA as a standalone remedy;
- dense raw `vocab_mse`, `delta_vmse`, or naive teacher-increment matching;
- more dynamic-gate variants whose only evidence is lower hidden loss;
- fixed layer 54 presented as a storage method;
- checkpoint selection on full-answer LCS without content and damage gates.

Most importantly, do not launch a student-trajectory v5w3 method on the main v4.6 branch until its protocol status is reconciled explicitly.

## Final assessment

August was scientifically productive because it converted an apparent small success into a well-supported null. V4 showed that exact local teacher-state imitation can be too easy to affect deployment. V5 showed that replacing teacher inputs with censored-student values makes the loss nontrivial but not well posed: it rewards cross-trajectory repair under teacher forcing, so it can fall through destruction or stylistic mimicry without creating recall.

The next experiment should not ask, “which layer has the biggest error?” It should ask, “for this exact current state, what change was caused by passage access, and can the model reproduce that change when it must generate its own prefix?” That is the shortest path from local matching to a falsifiable theory of memory.

## Evidence reviewed

- [`scripts/trainv5.py`](../scripts/trainv5.py), read end to end at `145f45c`.
- [`docs/runtime.md`](runtime.md) and [`src/selfupdate/train/losses.py`](../src/selfupdate/train/losses.py), defining the supported v4.6 law and current frozen-vocabulary metrics.
- [`WEEKEND_HANDOFF.md`](../WEEKEND_HANDOFF.md), including the complete August decision log.
- [Final 23 August campaign report](../runs/v5_campaign_report/report.pdf); the 16 August ten-page version was reviewed from Git commit `9207792`.
- Raw `metrics.jsonl` files under `runs/trainv5_*`, `runs/v5p_*`, and `runs/v5w2_*`, particularly the three linked above.
- [`layer_censor_v3.json`](../runs/v5_teacher_attn/layer_censor_v3.json) and the corresponding probe implementations [`v5_teacher_attn_probe.py`](../scripts/v5_teacher_attn_probe.py), [`v5_teacher_layer_censor_probe.py`](../scripts/v5_teacher_layer_censor_probe.py), and [`v5_teacher_ceiling_probe.py`](../scripts/v5_teacher_ceiling_probe.py).
- [`runs/vllm_h100/gemma4_31b_it/summary.json`](../runs/vllm_h100/gemma4_31b_it/summary.json) and the stored response artifact used to compare answer lengths and teacher-generation settings.
