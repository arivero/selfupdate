# The evolving person: conversations that become weights

*Motivation note, recorded 2026-07-04 from the project owner's framing.
This is the destination the layerwise program serves; the memorization
campaigns (C1: Machado, C2: saturation/Quijote) are its instrument
calibration.*

## The target scenario

A local chatbot on commodity-serious hardware — four H100s running a
120B-class MoE thinking model with tool use (gpt-oss-120B, Gemma4,
DeepSeek-Flash class). It answers with RAG: a `find` tool over the
user's documents, results injected into the thinking channel.

As people talk to it, **its weights evolve**:

1. The *usual details* it keeps retrieving from RAG migrate into the
   weights, so routine lookups stop being tool calls.
2. It learns the *patterns of questions* its users actually ask.
3. Consequently its retrieval behavior matures: fewer and more
   sophisticated RAG requests — from lookup ("what is the account
   number") toward verification and delta queries ("has anything about
   X changed since March") — because the base facts now live in the
   model and only the *frontier* needs the tool.

The model slowly becomes *someone*: a persistent local person whose
knowledge of its users' world is in its parameters, not only in its
prompt.

## Why the Pierre Menard setup is this scenario in miniature

Our training frame — teacher and student are the SAME model, the
teacher sees a privileged RAG block, the student does not, and per-layer
hidden matching moves the difference into weights — is not an analogy
for the chatbot; it is literally its inner loop:

- **The teacher forward pass already happened during serving.** When
  the deployed model answered with RAG context, its hidden trajectory
  at the answer tokens IS the teacher target. Nothing extra to compute
  at serve time; consolidation can replay the conversation through the
  same weights (adapters off — our `OnlineTeacherSource`) to regenerate
  targets on demand. No teacher cache at any scale
  (`frozen_teacher_copy` / adapters-off; the decision that made the
  Quijote ladder feasible).
- **Privileged span = the tool result** (and, in the continual setting,
  the oldest turns of a long conversation). The masking contract
  already expresses this: shared prefix = system + recent turns,
  privileged = what the student must learn to live without, aligned
  span = the answer whose trajectory we match.
- **Student view = the next conversation**, where the user asks again
  and the tool is not called.

## What the campaigns established (design rules, not hopes)

Each of these is a measured result in `EXPERIMENTS.md` / `paper/paper1.md`:

1. **Storage and readout are different problems.** Strict per-layer
   matching stores content reliably (block-local, provably bounded
   memory) but recites weakly; a small connected tail window with
   answer-CE supplies behavioral credit. → Consolidation is two-phase:
   stream-the-body storage passes, then a short `tail_only` readout
   phase. Both phases keep every gradient away from other blocks.
2. **The vocabulary is frozen, always.** Embedding, final norm, head
   are never trained (four independent locks). Over thousands of
   micro-updates this is what keeps the person speaking the same
   language as its base model — drift in the vocabulary basis would
   compound and silently re-index every cached lens and every teacher
   target.
3. **vocab_mse is the metric that transfers.** Matching hidden states
   in the geometry induced by the frozen head (Gram matrix W^T W) beat
   all geometric losses and makes the stored content *portable*
   (chimera transplants worked only for vocab-metric bodies).
4. **Catastrophic remembering is real and measurable.** Damage
   concentrates on the memorized genre's neighbors (Bécquer suffers
   when Machado is absorbed). For a chatbot this failure mode is the
   model leaking user-document phrasing into unrelated conversations —
   a quality AND privacy problem. Countermeasure: anchor-KL (stay
   close to base on neutral text, KL through the tail window), which
   halved neighbor damage at equal recall. Monitor: the destruction
   battery (5-category probes, benchmark CE-ranking, intrusion rate on
   bait prompts, degeneration counters) with pre-committed thresholds
   — run it as a *gate* after every consolidation cycle; reject the
   update if it trips.
5. **Elicitation diversity is not optional.** Content absorbed under
   one question template is template-locked; maieutic (Socratic
   dialogue) frames cured this (0.921 → 0.000 CER under novel
   elicitation). The chatbot gets this for free: real users ARE the
   maieutic corpus — their varied phrasings of the same need are
   exactly the diverse frames that keep absorbed knowledge readable.
6. **Capacity is a budget.** k=4 tail window holds any two of {trigger
   diversity, anchor discipline, chain depth}; k=8 holds all three.
   The C2 saturation ladder is measuring content-size × k ×
   destruction directly. The chatbot must throttle absorption to the
   capacity slope — consolidate the most-retrieved spans first and let
   the tail wait, rather than absorb everything and trip destruction.
7. **Thinking channels need selective censoring.** Reasoning-family
   models integrate RAG inside the think stream; censoring the whole
   think destroys the free deduction we want to keep. The
   `thinking_selective` design (C2 Part 3): censor only the tokens that
   are *verbatim from the tool result*; keep the model's own reasoning
   visible to the student. This is the exact mechanism the 120B
   scenario needs, piloted on Qwen3-think as the controlled member of
   the family.

## The consolidation cycle (concrete architecture)

Nightly (or idle-time), on the serving hardware:

1. **Collect** the day's traces: turns, think streams, tool calls,
   tool results, final answers.
2. **Score spans for "worth of attention"** — the owner's phrase, and
   the core open instrument. Candidate signals, in order of current
   readiness:
   - *retrieval frequency*: a RAG result fetched many times is the
     definition of a "usual detail" (free from tool logs);
   - *content-head attention mass*: the head-taxonomy probe (C2 Part 6,
     `scripts/attention_probe.py` → `runs/attention_probe_0.6B/`)
     classifies heads into grammar-like (local, low-entropy,
     privileged-blind) vs content (privileged-heavy); attention from
     content heads onto a span is a direct "the model considered this
     worth attention" measurement. **First results (Qwen3-0.6B, 16 v4
     examples): 112/448 content heads, concentrated at L7–L20 (peak 11
     heads at L12) and absent from the readout tail L26–28; the single
     strongest is L7-h6 with 95% of its answer-position attention on
     the passage at entropy 0.32 — the same L7 where teacher_censored
     localized context integration. Retrieval attention lives mid-net;
     the tail assembles but does not retrieve. Caveat: raw "attention
     distance" is sink-confounded (heads staring at token 0 score ~700
     tokens); distance only defines grammar heads, never content.**
   - *MoE router statistics*: if novel-content tokens route to
     identifiable experts (C2 router probe on gpt-oss-20b), routing is
     both a scoring signal AND a parameter-selection mechanism —
     consolidate into the experts that fired, leave grammar experts
     untouched. MoE models may make the grammar/content split
     architectural rather than inferred.
3. **Build items**: student view = conversation with scored spans
   censored (removed or, in thinking models, selectively masked);
   teacher = same weights with full context (adapters off).
4. **Two-phase layerwise consolidation**: strict body matching
   (streamed block-by-block — bounded memory even at 120B, can run
   concurrently with serving on spare capacity), then `tail_only`
   readout with answer-CE + anchor-KL.
5. **Gate**: destruction battery vs the running base reference. Trip →
   reject or downweight and retry with stronger anchors. Pass →
   promote (merge adapters / swap block weights).
6. **Track the two curves that define success**:
   - *RAG-independence*: fraction of previously retrieval-dependent
     answers now correct with the tool disabled;
   - *query sophistication*: distribution shift of issued tool calls
     (count, specificity, lookup-vs-delta character) at fixed answer
     quality.

## What question-pattern learning is (and is not)

Learning the users' question patterns needs no censoring — it is
ordinary adaptation on observed dialogue. Its role here is specific:
(a) it supplies the elicitation-diversity that keeps absorbed content
readable (finding 5), and (b) it shifts the tool policy — the model
that already knows the base facts asks the tool better questions. The
distillation machinery and the pattern learning are complementary, not
competing: one moves *content* into weights, the other moves the
*shape of demand* into behavior.

## Privacy note

Weights that contain user details are extractable by anyone with
access to the instance. For a single-tenant local person this is the
feature, not the bug — the machine is *theirs*. Multi-tenant or shared
deployments must treat the intrusion-rate instrument as a privacy
gate, not just a quality gate.

## Staged roadmap

- **Stage A — memorization (done, C1/C2):** fixed corpus, RAG teacher,
  full metrology. The physics of the method.
- **Stage B — single-conversation absorption (prototype, C2 stretch):**
  synthetic multi-turn conversations where the privileged content is
  the oldest turns carrying unique facts; eval = QA about censored
  turns + destruction gate. First instance of
  conversation-becomes-weights.
- **Stage C — continual (the target above):** the consolidation cycle
  on live traces, capacity-throttled, destruction-gated, on MoE
  hardware with router-guided parameter selection.

The through-line: every instrument Stage C needs — storage mechanism,
readout window, forgetting metrology, capacity law, span scoring,
selective censoring — is either measured or under measurement in the
campaigns. Nothing in the scenario requires machinery we have not
already run at small scale.
