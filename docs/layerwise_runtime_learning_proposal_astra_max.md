# Layerwise learning while the model is running

Research proposal · 14 September 2026 · **Astra Max**

Status: proposed experiments, not an implemented or validated successor.

The next question should be: **can information observed by a running model
become useful to a later prediction through local updates completed while
inference continues?** I propose studying the learning rule and its execution
schedule together. Success requires a useful update, a causal source for that
update, and a measured time by which subsequent inference can use it.

The proposed contribution is a testable combination of local teacher-sourced
learning, bounded memory, and concurrent serving. Fast weights and test-time
training already exist; novelty cannot rest on renaming them. The unresolved
question for this project is whether they can work with a frozen vocabulary,
no gradient crossing block boundaries, uniform treatment of depth, and no
future answer supplied to the learner.

## 1. What the previous project changes

The [final project write-up](project_final_writeup_2026_09.md) establishes
limited but nonzero learning. Qwen partial-teacher seed 43 improved whole-set
content recall by 0.020601 after 40 epochs. Its cloze gain was 0.206325, while
next-continuation recall changed by −0.009109. Its mean per-layer gradient
shares were 81.24% hidden, 10.43% lens Jensen–Shannon, and 8.32% anchor.
Those are local norm shares on five probes, not attribution of behavioral
improvement. Gemma causal produced a smaller positive whole-set effect;
neither law met the full promotion requirements.

My inference is that local writing is possible, but useful addressing and
autonomous continuation are unresolved. Fifty-percent reductions in a local
loss do not establish either. The new program should first test whether the
desired local correction helps from an actual deployment prefix, then whether
it can be learned from a small number of observations. Forty offline epochs
cannot serve as evidence for immediate runtime learning.

Three claims require separate experiments:

| Claim | What must be demonstrated |
|---|---|
| Local learning | An update uses a local loss; only that block's parameters change. |
| Learning during runtime | Serving remains active while useful updates are computed and published. |
| Persistent learning | The benefit survives removal of the passage, ordinary caches and transient episode state. |

A model can satisfy any one without satisfying the others. A recurrent memory
that resets after a request is useful, but establishes adaptation within that
request, not persistent consolidation.

## 2. Research context and the specific gap

Hinton's Forward-Forward algorithm uses separate objectives at each layer
with positive and negative passes. Decoupled Greedy Learning studies local
objectives and asynchronous replay in convolutional networks. They motivate
removing the wait for a full backward pass; they do not demonstrate language
memory acquired during serving. [Forward-Forward](https://arxiv.org/abs/2212.13345),
[Decoupled Greedy Learning](https://arxiv.org/abs/2106.06401).

Test-Time Training layers make recurrent state an adaptable model; TTT-Linear
and TTT-MLP update that state through a learning rule. Titans combines attention
with a neural memory updated at test time. These are precedents for treating
runtime memory as weights, rather than evidence that an arbitrary pretrained
decoder will immediately use a newly attached memory.
[TTT layers](https://arxiv.org/abs/2407.04620v4),
[Titans](https://arxiv.org/abs/2501.00663v1).

End-to-End TTT learns an initialization for subsequent test-time updates and
uses next-token prediction. Its reported implementation adapts MLPs in the
last quarter of the stack and uses an outer training objective. That is a
useful external comparison, but does not satisfy this project's all-depth
locality contract. Its preparation cost must also be included when comparing
against an unchanged released checkpoint.
[End-to-End TTT](https://arxiv.org/html/2512.23675v2).

In-Place TTT adapts existing MLP down-projections. TTT-NTP instead uses a
projection of the next position's contextual hidden state as the write target.
The latter is particularly relevant to our target-selection problem. However,
its projection and backbone undergo continual pretraining with an outer
next-token objective, and its inference write is applied before decoding.
Those results do not establish local-only learning with overlapping decode
and updates. I would compare the ideas while recording these differences,
without importing the published scores as expected performance here.
[In-Place TTT](https://arxiv.org/abs/2604.06169v1),
[TTT-NTP, method and training protocol](https://arxiv.org/html/2606.21803v2).

## 3. A causal definition of simultaneous learning

Give every external observation, teacher computation, update and prediction
an event identifier and timestamp. For each prediction, record which versions
of the layer memories it read and the latest source event used by each version.
Learning about an event may improve a later prediction; it may not retroactively
improve the scored prediction that preceded the event.

Start with a streaming reference implementation at K=1: consume only the
currently observed token or event. B denotes independent simultaneous streams.
Larger chunks may contain already received prompt tokens; they must not contain
unseen teacher completions. Future generated tokens are not available merely
because a training artifact contains them. A predictor using the hidden state
at position t+1 as a target cannot apply that write until t+1 has been observed.

Measure three schedules against the same causal event stream:

| Schedule | When an update becomes visible | Interpretation |
|---|---|---|
| Sequential reference | Serving pauses for the eligible update, then resumes | Learning-quality and timing reference. |
| Concurrent, next request | A learner runs during serving; each request pins one immutable version | First feasible concurrency experiment. |
| Concurrent, same stream | Complete updates become visible at later token boundaries within the active stream | Primary simultaneous-learning objective. |

For the last case, a trace must show learner work overlapping actual serving
work, and a later prediction in that stream must demonstrably use the published
update. Two processes being alive is insufficient. Single-GPU interleaving
and actual overlapping kernels on one or several GPUs are reported separately.

During emission of an answer with no new external evidence, local work can
still process previously observed events or a teacher's judgment of an already
emitted prefix. The student's own output is a query to the teacher, not an
independently verified fact to memorize. Reference answers remain evaluation
data and never enter the learner. A teacher may use only evidence already
available at that point; teacher compute is part of the runtime bill.

## 4. Learn a useful local correction before optimizing its speed

I would begin with deployment prefixes rather than further true-answer-prefix
training. Let p_t be an actual student prefix, and let E_t be evidence already
observed. Construct an immutable local training record for every block:

```text
x_L = detached student input to block L at prefix p_t
c_L = detached attention/recurrent context used for that local evaluation
b_L = frozen block L evaluated on x_L and c_L
y_L = detached teacher target at the corresponding predictive position
```

The teacher runs on the same emitted prefix p_t, with access to E_t. It never
replaces the prefix with the reference continuation. Whole-teacher and
same-input causal targets are separate arms: the first retains v6's trajectory
gap; the second needs the architecture-specific passage-access intervention
to remain valid on the actual runtime state. Map predictive positions by token
identity and log position conventions; extra teacher context does not guarantee
representation alignment.

Before fitting any weights, apply the proposed corrections as evaluation-only
activation patches. Compare the resulting free continuation with the unpatched
student. Scan all depths and a uniform all-layer patch; include a wrong-event
patch. This distinguishes an insufficient target from failure to fit or
retrieve a sufficient one. The teacher-assisted patch is an oracle diagnostic,
not a memory result and not a method eligible for deployment promotion.

Then compare two implementations of the same local task. The first uses an
ordinary block-local adapter and the v6 loss family on these new prefixes. The
second uses a small associative memory at every block's residual output:

```text
k_L = unit_normalize(R_L x_L)        # fixed projection; dimension r
u_L = b_L + V_L k_L                  # V_L has shape hidden_size × r
d_L = stop_gradient(y_L - b_L)
```

Begin with frozen seeded projections R_L and zero V_L. A local write minimizes
one half of ||V_L k_L − d_L||², with an optional uniform shrinkage term:

```text
V_L_next = (1 - eta*lambda) V_L
           + eta (d_L - V_L k_L) k_L^T
```

Use bounded targets and a fixed update-norm bound selected on development
streams. The same policy applies at every depth. At a zero key, skip the write
and count it. This is an explicit local gradient update, even if implemented
with matrix operations rather than automatic differentiation. A single-key
unregularized update with eta=1 and a unit key exactly fits that target at that
key; it says nothing about interference, new queries or generated behavior.

The fixed projection is a deliberately falsifiable first choice. If literal
queries work but paraphrases do not, addressing is the next hypothesis. A later
arm may prepare the local key map with a teacher-derived local objective on
disjoint development episodes. Count that preparation and compare with the
same prepared model with runtime writes disabled. No end-to-end meta-gradient
or output loss is silently added to make this work.

The memory arm deliberately changes the write mechanism. Compare it with an
iterative solver of the same squared local objective to separate solver effects
from the adapter/objective change. An optional frozen-head Jensen–Shannon term
belongs to a separately named gradient-based arm; the rank-one formula above
is not its exact optimizer. Prompt-preservation replay, if used, is also an
explicit local objective with reported gradient shares and replay cost.

All trainable memory belongs to a decoder block. Embeddings, final norm and
vocabulary head stay frozen. All layers have equal objective coefficients and
update opportunity. Gradients never traverse another block, a stored context
or a previous update in time. Using detached live student values is an
experimental successor to v6, distinct from v4.6's teacher-input law; it would
be implemented in an isolated monolith, not dispatched through v4.6.

This mechanism still needs a deployable key. During the observation phase,
the teacher can make rehearsal cues from already seen evidence, under a fixed
compute allowance. Final evaluation queries and their answers are withheld
from that rehearsal. The query-blind condition is the memory test. A separate
query-conditioned adaptation condition may see the evaluation question after
it arrives, but must be labelled and charged for its extra work.

## 5. Runtime state is part of the algorithm

Use one owner for each layer's mutable write state, immutable versions for
readers, and a bounded queue of detached training records. At a token boundary,
pin a vector of completed per-layer versions; do not let a token's matrix
multiply race a write. Records retain their source version, event time and
target provenance. Recompute the local loss on the owner's current write
state before updating; never apply an unlabelled old gradient to a newer state.
The input and teacher target can still be stale, so their age remains a
measured experimental variable.

The conceptual algorithm is:

```text
SERVE each stream:
    Pin complete published layer versions at the token boundary.
    Run the token using those versions and the declared cache policy.
    Emit and record the prediction before exposing any future observation.
    Enqueue bounded detached records for already available learning events.
    Continue serving; record overflow and delay rather than waiting unnoticed.

LEARN at each layer owner, while serving continues:
    Take the oldest eligible record within the fixed age limit.
    Obtain its teacher target using only that record's allowed evidence.
    Fit this layer's current write state to the detached local target.
    Publish a complete new immutable version after the write finishes.
    Retire old storage only after all readers release it.

EVALUATE at fixed event cutoffs:
    Snapshot the published version vector without draining extra updates.
    Use an isolated frozen snapshot; live serving and learning may continue.
    Remove the evidence and the specified caches from the probe context.
    Measure future queries, capability, retention and runtime cost.
```

Two memory banks alone do not handle arbitrarily long-lived readers. Use
reference-counted bounded versions; if none can be retired, coalesce or defer
publication under a declared limit. Report unpublished work. Never overwrite
a bank still in use. On shutdown, checkpoint the published version vector;
post-stream queue draining is a separately timed condition, not part of the
online score.

Changing an early block changes the inputs from which later layers previously
formed attention keys/values and recurrent states. Freezing attention weights
does not eliminate that issue. Evaluate these semantics separately:

1. Pin the model for a request and rebuild state for the next request. This is
   the concurrency control with straightforward snapshot semantics.
2. Publish within a stream and rebuild the retained prefix under the new
   version. This is an expensive correctness reference for a fixed-model
   interpretation; count rebuilding in latency and throughput.
3. Publish within a stream while retaining historical state. Define this as
   a time-varying recurrent system. Its numerical reference is a sequential
   replay of the same updates and historical states, not a fresh forward
   under the latest weights. Memory produced under different versions is
   intentional only in this explicitly labelled arm.

For a persistent-memory claim, start a fresh request with only the saved
learned weights. Remove raw evidence, rehearsal buffers, attention K/V and
GatedDeltaNet recurrent state. Keeping those channels would make success
ambiguous. Keep each stream's learned state separate in the first experiment;
shared persistent learning is a later interference study.

As a size calculation, 64 layers × hidden size 5,120 × r=32 gives 10,485,760
entries: 40 MiB for one float32 V bank and 80 MiB for two, per independent
memory stream. This excludes fixed projections, extra versions, the base
model, teacher, contexts and queues. At B=256, just those two banks require
20 GiB. The memory is small relative to the large-model weights for B=1, but
it is not free at serving scale. Teacher passes and memory bandwidth may
dominate; local backward does not imply negligible overhead.

## 6. Experiments that can discriminate the explanations

Use the smallest already supported model to establish mechanics, a 4B-class
model for the scientific screen, then the pinned Gemma-31B and Qwen-27B
snapshots only for successful comparisons. This is a staged budget proposal,
not a queue submission or a prediction of epoch duration.

| Order | Controlled comparison | What would be learned |
|---|---|---|
| A | Deployment-prefix target patches versus no patch and wrong-event patch | Whether the proposed local information can improve autonomous continuation at all. |
| B | Same update, teacher-answer prefix versus actual student prefix | Whether the old prefix distribution was a binding limitation. |
| C | Fixed-key memory versus ordinary adapter; direct versus iterative fit of the same memory loss | Whether rapid fitting or addressing helps independently of objective changes. |
| D | Sequential versus concurrent updates, with the same event and publication schedule | Numerical correctness first; then the quality cost of real queue delay. |
| E | One exposure, then 2 and 4 exposures; recall after 1, 10 and 100 intervening events | Acquisition speed, interference and useful retention. |
| F | Saved weights alone versus retained episodic state; reset/shuffled memory controls | Persistent memory versus context retention or an incidental computation change. |

Keep a frozen no-update baseline, an evidence-in-context teacher reference and
a bounded external retrieval baseline with access to the same observations.
Report total storage, teacher preparation and runtime compute for retrieval
too. It is a serious alternative: if it offers better recall at the same cost,
weight writing needs another demonstrated benefit.

Use newly generated arbitrary factual associations to reduce pretrained
knowledge contamination, prose passages, and the historical literary corpus
as separate strata. Include first-token factual queries, cloze, autonomous
multi-token continuation, paraphrased questions and delayed updates to a fact.
A contradictory new fact is an explicit revision event; measure whether the
model follows the latest observed revision without corrupting unrelated facts.

Separate development documents and query templates from the locked evaluation
stream. At evaluation time, the learner is allowed to observe a document as
the adaptation input; the later probe questions and reference answers remain
unseen during writing. Report performance before writing as well as after it.
Use identical evaluation item IDs across seeds and arms. Whole-set item means
and equal-weight corpus means are both emitted, never substituted silently.
Vary initialization and stream order across seeds; pair the event order and
baseline within each seed so that scheduling comparisons see the same history.

The first scientific comparison uses three seeds, equal unique observations,
equal numbers of exposures and matched compute ceilings. A run receives at
least 12,000 training items before a flatness verdict, spread across fresh
episodes so that one-shot learning within an episode stays a one-shot test.
Short mechanics jobs do not support scientific claims. Runs that hit time
limits retain their partial observations but do not masquerade as matched
terminal comparisons.

## 7. Metrics and decisions before spending at scale

Define a primary online score before the confirmatory runs: content recall on
unseen next-continuation probes after evidence removal, at 10 intervening
events and one exposure. Questions execute at fixed arrival times, whether
updates have caught up or not. Use an answer-length allowance established from
the reference for scoring, and retain the 96-token result separately. Report
cloze, first content token, prefix length and exact recitation as parallel
outcomes; a cloze improvement cannot carry a failed continuation endpoint.

Candidate proposal criteria are an absolute primary gain of at least 0.03 over
the frozen baseline, a 95% uncertainty interval entirely above zero, a positive
effect in all three seeds, and standard-benchmark accuracy changes above −0.03
for the macro mean and −0.05 for the worst task. The capability tolerances
are inherited from v6; the runtime endpoint is new. These are prospective
engineering choices, not a theorem or revised interpretation of v6.
Bootstrap independent documents or episodes, with seed-level results alongside;
do not treat overlapping windows
as independent samples. Fix the confirmatory endpoint in advance. Interim
panels guide operations and exploratory diagnosis only.

Runtime has its own success criteria. Initially target at least 80% of frozen
serving throughput and at most 20% increase in p95 per-token latency on the
same hardware and offered load. Publish a quality/cost curve even if those
targets fail. Report:

- Time to first token and p50/p95/p99 inter-token latency, including queueing.
- Successful responses and generated tokens per wall-clock second, separately
  from update count; useful recall gain per observation and per GPU-second.
- Observation-to-publication latency and publication-to-first-use latency,
  in milliseconds and stream events; distributions of target and input age.
- Kernel overlap, memory peak, queue occupancy, rejected/coalesced writes,
  uncompleted work at cutoffs, and teacher/rehearsal/rebuild compute.
- Retention curves, interference on unrelated facts and behavior when learner
  throughput falls below arrival rate. A finite queue cannot hide growing debt.

Measure serving curves both with and without scheduled evaluation. Disclose
all allocated GPU-seconds, including separate teacher/evaluator devices and
idle reservations; a second GPU is additional cost, not free concurrency.
Keep evaluation within the trainer-owned pipeline. Frozen probe snapshots do
not authorize an unreported pause or extra learner work before the cutoff.

Cross-entropy (log loss) against teacher-realized answer tokens and
Kullback–Leibler divergence from teacher output distributions remain
evaluation-only. At every scheduled full battery, score every realized answer
token of every observed training item up to that declared cutoff, and report
expected and evaluated item/token counts. For repeated-corpus bridge runs,
also retain the mandatory whole-training-set pass per completed epoch.
Emit `evaluation_only=true`, `used_for_backward=false`, and
`optimizer_weight=0`. Delayed full batteries use frozen published snapshots
and are not fed back into the running learner or learning-rate choice.

For each objective report layerwise loss, parameter delta, weighted gradient
norms, shares and pairwise cosines. For the closed-form write, record the
equivalent gradient and actual bounded parameter update separately. Undefined
zero-signal shares remain undefined. Freeze the head and embeddings and check
gradient isolation in the actual queued execution. Persistent logging must
not add a synchronization call for every layer of every token.

The first decision is whether a local target helps from a deployment prefix.
If it does not, faster scheduling will not rescue that target. If the target
helps but stored corrections fail on paraphrases, investigate the key. If
weights-only recall works sequentially but degrades under concurrent updates,
investigate publication delay and state semantics. These outcomes each justify
a different next experiment and avoid another indiscriminate loss/rank sweep.

I would proceed to large Gemma and Qwen only after the 4B screen identifies a
useful correction and a bounded-delay schedule. Before each launch, reserve
training, evaluation and checkpoint time together from measurements; run all
model tests through Slurm, keep runtime caches node-local and serialize any
new Hugging Face download to one worker. A proposed implementation would retain
a readable monolithic entry point and trainer-owned evaluation. This proposal
does not reopen the finished queue.

## 8. Applied goal: a specialist reader of Machado and Cervantes

The practical objective is a model that accumulates reliable knowledge of the
works while discussing them. Our present coverage is much narrower than either
author: Machado's *La tierra de Alvargonzález* and four chapters of *Don Quijote*
in the v6 campaign. Start there to establish acquisition; expand to more works
only after measuring transfer and interference. Existing general language and
reasoning abilities are the starting point, not an outcome that this small
corpus can be assumed to teach.

A concrete reading session would work as follows:

1. A stanza or episode becomes available through reading or retrieval. Record
   its edition, source span and time of availability. An immediate answer that
   uses retrieval is labelled retrieval-assisted; it is not a memory result.
2. While the student continues serving, the frozen teacher consults that
   passage on prefixes the student actually emits. Detached passage-hidden
   student records are matched to local teacher states. The teacher may also
   form a small, budgeted set of rehearsal cues about the observed passage.
   These cues cannot include the locked later test questions.
3. Every layer fits its own adapter or fast-memory correction. Publish complete
   versions during serving. The rank-one memory arm asks whether a useful
   correction can become available quickly; the deployment-prefix arm asks
   whether it helps the model's own continuation rather than only a supplied
   correct answer. Neither trains the head or a reference-answer log loss.
4. After unrelated conversation, ask a new question about that passage without
   retrieving it. Test a differently worded question, not just the rehearsal
   wording. Later, reopen a fresh session with saved learned weights alone.
5. Interleave the two authors and keep probing earlier material. In the
   cumulative-specialist arm, preserve the subject's learned state across
   sessions; use separate copies for independent experimental streams. A new
   Cervantes observation must not silently erase previously learned Machado.

The three desired abilities need different evidence:

| Ability | Machado/Cervantes task | Measurement |
|---|---|---|
| Accurate quotation | Continue a passage or supply an attributed quotation from a specified edition | Exactness, content recall, correct prefix, source identification and invented-quotation rate; cloze is separate. |
| Knowledge of the work | Explain who did what, reconstruct event order, connect distant passages, or answer paraphrased factual questions | Source-grounded factual correctness on unseen questions, delayed retention and interference across works. |
| Literary interpretation | Explain an image, distinguish narrator from character or author, compare passages, and support a reading | Blinded assessment of textual support, coherent reasoning, valid alternatives and calibrated uncertainty; lexical overlap is not a sufficient score. |

The teacher's interpretation is not a factual authority simply because it
supplied a hidden target. Use the source text and independent review of
trainer-emitted responses for these judgments. Historical or biographical
expertise needs additional documented evidence; it cannot be inferred from
learning one poem and several novel chapters. Include unanswerable questions
and measure fabricated citations and confident unsupported claims.

The essential addressing test is whether writing under one cue helps a
different cue for the same passage. This is where the proposed key-map and
rehearsal experiments matter most. A reduction in local error at the original
key does not establish accessible literary knowledge. Likewise, exact quoting
does not by itself establish interpretation, and fluent commentary does not
establish exact knowledge of the text.

For a useful product, keep retrieval available when exact sourcing is needed.
Compare retrieval alone, learned weights alone, and their combination. Report
accuracy and response cost as well as retrieval frequency; avoiding retrieval
is valuable only if accuracy survives. The weights-only arm isolates the
scientific learning claim. Broader specialist criteria must be registered
separately from §7's narrow continuation endpoint, with Machado and Cervantes
reported separately so that the larger Machado item count cannot decide both.

The result worth pursuing is a concrete event sequence: the model encounters
new information, local weights change while it continues serving, and a later
unseen query is answered better after the original information and ordinary
caches have disappeared. The experiments above make every link observable.

Signed: **Astra Max**
