# Issues / Follow-Ups

Post-campaign state (2026-07-04). The 24-40h campaign is recorded in
EXPERIMENTS.md (closing table) and runs/report.pdf. Closed items are
removed from this file (git history keeps them); 2026-07-10 pass removed
the campaign done-list and the completed hot-loop ladder.

## Future Work

1. **Window capacity as a budget**: study k as a budgetable capacity
   (triggers vs anchors vs depth). (The final_k8/708-chain conditional
   resolved in C1: k=8 restored the chain; thinking_selective landed in
   C2-12 and continues as C3 #2.)
4. **Tuned-lens program** (partially landed): translators + C2-11
   re-profiles are in tree (train_tuned_lens.py, tl_i_tunedlenskl). The
   Wave-I "tuned-lens-CE auxiliary" half is FORBIDDEN as label-CE; only
   the teacher-sourced tuned-lens-KL variant is a legal continuation.
5. **Scale**: final recipe at 4B/8B full-FT (sequential/offload_adam for
   VRAM — tail_only is expunged on this branch), 14B+ LoRA; Don Quijote
   data engineering.
6. **Anchor corpus breadth**: anchors_es.txt is 6 fragments; a rotating
   larger corpus may improve anchor-KL further.

## Observed RAG failure mode — full-document copying is not guaranteed (2026-07-12)

The untrained Qwen3-1.7B base model was asked the ordinary recall prompt
`¿Qué línea sigue inmediatamente a esta? «Bajo la nevada, un hombre»` with
the full Machado source appended as `Documento recuperado`, using the same
chat template and greedy RAG-ceiling path as `tasks_eval(with_context=True)`.
It answered a paraphrastic/hallucinated continuation rather than the literal
in-context line `por el camino cabalga;`.

This is **not** a context-placement or truncation bug: the expected text was
at input tokens 3092--3098, the sole `<|im_end|>` was at 6466 *after* the
document, and the 6,475-token prompt was far inside Qwen's 131,072-token
window. It is a retrieval/use failure: the base does not reliably select and
copy a relevant line from a long retrieved document. Do not call the
full-document RAG ceiling an oracle or infer that RAG exposure itself supplies
literal recitation. Required control before any such claim: repeat the exact
question with a short, provenance-recorded local passage (the retrieval form
used by the training examples), and report both full-document and local-passage
results separately.

## OPEN — untested same-width teacher/student losses (owner question, 2026-07-10)

Scope: losses below have not been campaign-tested as trainer objectives in this
branch (some already exist as diagnostics). Every target is produced by the
teacher or frozen base model; none uses reference-text labels. Every per-layer
term must use the same coefficient at every depth (or a depth-uniform sampled
alternancy), embeddings/head remain frozen, and connected credit remains a
sanctioned sliding window with `conn_stride: 1`. This is an idea ledger, not an
implementation commitment.

### Implementation status (active work only)

The state+delta, base-anchor trajectory, recombined attention/MLP contribution,
relational+absolute, offline Mahalanobis calibration/objective, and raw
multi-scale delta implementations are now in tree and queued (2026-07-11).
The only catalogue item intentionally not promoted to a trainer objective is
attention-probability/route distillation: it remains the explicit deferred
control below because fused and hybrid architectures do not expose one common,
honest target.

Diagnostics are prioritized separately in `docs/lens_diagnostics_ideas.md`:
intrusion/commitment depth, write spectrum, the expiring batching-regime
control, and retrospective epoch prediction precede another trainer loss.

General mechanism/telemetry proposals kept here rather than in the lens
document:

- **Causal residual-write patching.** Patch a teacher block write into the
  student, and vice versa, at selected layers/positions; measure final recall
  and standard-benchmark damage. Observational write energy can nominate
  layers, but intervention is required before claiming causal responsibility.
- **Loss-gradient agreement.** At sparse calibration steps, measure cosine and
  norm ratio among the candidate hidden-loss, teacher-readout, and anchor-
  preservation gradients. Persistent opposition directly measures the
  destruction tax and can justify fixed global loss weights without learning a
  depth-biased weighting scheme.

### Candidate catalogue and scientific rationale

1. **Successive block-increment matching (`delta_*`) — IMPLEMENTED 2026-07-11,
   awaiting the controlled loss-grid campaign.** For block `L`, match
   `d_s,L = h_s,L - stopgrad(h_s,L-1)` to
   `d_t,L = h_t,L - h_t,L-1`, rather than matching the absolute `h_L` alone.
   Implemented metrics are normalized MSE (`delta_nmse`), cosine
   (`delta_cosine`), and the most promising form,
   centered vocabulary-score cosine
   `1-cos(C W d_s,L, C W d_t,L)` (`C` removes the vocabulary mean; head bias is
   omitted because a vector contribution has no bias). This directly assigns a
   block responsibility for what it adds and does not repeatedly charge layer
   `L` for inherited student error. It therefore fits strict local training and
   the observed compounding of residual mismatch. The implementation uses raw
   interior updates only (`2 <= L < n`); layer 1 and the cache's post-final-norm
   endpoint use their paired state metric, so no normalization operation is
   misclassified as a transformer update. Test this current delta+boundary
   objective against a future explicitly weighted state+delta objective,
   because delta-only admits accumulated drift: all increments can be slightly
   wrong while their local losses remain small.

2. **Frozen Jacobian-pullback matching (`jacobian_*`) — IMPLEMENTED AND
   QUEUED (2026-07-11).** The sibling `../jacobian-lens` checkout holds
   Qwen3-1.7B (466 generic WikiText prompts) and Qwen3-14B Jacobian lenses.
   For an interior layer, its frozen base/teacher matrix `J_L` transports a
   residual into the pre-final residual basis: `z = J_L h`. Match either
   `||J_L(h_s-h_t)||^2 / ||J_L h_t||^2` (`jacobian_nmse`). The decisive
   comparison should use the same Jacobian-lens representation in two
   frozen-head forms:

   ```text
   z_s = final_norm(J_L h_s),  z_t = final_norm(J_L h_t)
   jacobian_vocab_mse = ||W(z_s-z_t)||^2 / ||W z_t||^2
   jacobian_lens_kl  = KL(softmax(W z_t) || softmax(W z_s))
   ```

   This is the exact MSE-versus-KL question for the *same* downstream-aware
   lens, rather than a comparison of two unrelated losses. The MSE version is
   its flat frozen-vocabulary geometry; the KL version weights errors by the
   teacher lens distribution and can favour sharp, behaviorally decisive
   token distinctions. It may also repeat the known full-vocabulary KL
   brittleness, so run both with identical prompt/layer coverage, loss weight,
   update norm, epoch telemetry, and standard-damage budget. This weights an
   error direction by its estimated downstream effect instead of treating all
   coordinates equally; it is the teacher-state analogue of the Jacobian
   lens's "what this residual is disposed to make the model say" readout.
   `J_L` is frozen, applied only as
   a matrix inside the current local loss, and therefore neither creates a
   long backward path nor introduces reference-text supervision. Map our
   1-based `L` to the lens's 0-based block output `L-1`; retain the ordinary
   state fallback for the final normalized endpoint because published lenses
   stop before that nonlinearity. Load one matrix at a time from CPU/pinned
   memory (1.7B: 16 MB/layer fp32; 14B: ~100 MB/layer), not all 14B matrices
   to GPU. Require exact model identity, width, source-layer coverage, and
   lens metadata in the run config/report. Main scientific risk: `J_L` is an
   average generic-corpus/future-position Jacobian, not the current
   prompt-conditioned derivative; compare it first against `vocab_mse` at
   matched update norm, and report whether it improves full recall without
   moving the standard-damage frontier. Do NOT apply a teacher lens to a
   student of a changed width; same-width fine-tunes share a basis but should
   still receive a post-hoc separate-lens comparison if we claim transport
   preservation.

   **MSE post-mortem / discarded objective (2026-07-11):** the first MSE implementation incorrectly
   treated `J h` as an absolute transported state, applied nonlinear final
   normalization to it, and normalized by `J h_teacher`. Epoch-0 layer losses
   exposed a ~300x L1-to-L28 scale imbalance, with L28 the sole ordinary
   fallback because fitted source coverage ends at 26. The corrected
   `jacobian_nmse` / `jacobian_vocab_mse` objectives operate on the transported
   perturbation `J(h_s-h_t)` and divide by the pullback operator's trace energy
   times teacher coordinate energy. This preserves the directional `JᵀJ`
   metric while making its expected scale depth-comparable. A direct numerical
   check confirms `WᵀW` matches explicit vocabulary-logit squared error in
   both value and gradient, so matrix orientation is not the defect. Even so,
   the owner discarded Jacobian MSE as a campaign objective: the KL arm remains
   unchanged and four Jacobian-cosine arms (two fits x slide 1/2) are queued as
   the verification. Old MSE checkpoints are invalidated and must not be used
   as corrected-loss evidence.

   Implementation: `HiddenLoss` now provides `jacobian_nmse`,
   `jacobian_vocab_mse`, and `jacobian_lens_kl`, all configured by the explicit
   `train.jacobian_lens_path`. Loading validates the artifact schema, hidden
   width, finite square matrices, and exact declared source-layer coverage.
   Matrices remain frozen on CPU and are copied lazily in the active hidden
   dtype to the device that owns each source layer; the per-device copy is
   retained to avoid a PCIe transfer per training item. Layer `L` uses source
   matrix `L-1`; the final normalized endpoint has no fitted matrix and falls
   back to `vocab_mse` or `lens_kl` respectively. The paired slide-1/slide-2
   arms use the same Qwen3-1.7B, 466-prompt artifact and are prioritized ahead
   of the remaining delta-loss queue. Their normal epoch-zero/every-epoch
   three-corpus recall and standard-benchmark damage telemetry applies. The
   pure `jacobian_nmse` ablation is queued afterward: it measures exactly the
   normalized `JᵀJ` geometry, with no final norm or vocabulary unembedding,
   isolating whether the frozen-head metric was responsible for the KL result.

3. **Multi-scale/cumulative trajectory matching — IMPLEMENTED AND QUEUED 2026-07-11.** Match short finite
   differences `h_L-h_{L-k}` for uniform `k in {1,2,4,8}` (only within the
   current sanctioned window), or cumulative change `h_L-h_0`, using normalized
   MSE or centered vocabulary-score cosine. This interpolates between local
   increment matching and absolute-state matching: `k=1` identifies the writer,
   while larger `k` constrains accumulated drift and cross-block cooperation.
   Use the same set of scales at every eligible depth and normalize each scale
   by the teacher delta energy, otherwise long spans dominate. Main risk is
   duplicated supervision and a larger effective gradient; report gradient
   share per scale and compare at matched update norm, not merely equal nominal
   weights.

4. **Relational token-geometry distillation — IMPLEMENTED AND QUEUED 2026-07-11, only beside
   an absolute state term.** Instead of requiring every
   hidden coordinate to coincide, match teacher/student relations among aligned
   token rows: pairwise cosine matrix, normalized squared-distance matrix, or
   centered Gram matrix. Example:
   `||norm(H_s) norm(H_s)^T - norm(H_t) norm(H_t)^T||_F^2 / A^2` for aligned
   length `A`. This is the same-width analogue of relational knowledge
   distillation and may preserve verse/token organization while tolerating
   harmless channel-scale error. It is teacher-sourced, depth-uniform, and
   cheap because `A << hidden_width`. However, a rotation-invariant relational
   loss alone is incompatible with the frozen next block/head: it can achieve
   zero without returning to the teacher's coordinate system. Use it only as an
   auxiliary beside `nmse` or `vocab_mse`, never alone. Include distance and
   angle variants separately; they encode different invariances.

5. **Attention-route distillation — DEFERRED CONTROL.** Match causal attention distributions for
   each head and query at the aligned answer positions:
   `mean KL(A_t || A_s)` over valid keys, optionally with a Jensen-Shannon or
   squared-logit alternative. This targets the hypothesized mechanism directly:
   attention from answer positions into privileged context writes the missing
   information. Also test a lower-memory aggregate that matches attention mass
   assigned to semantic regions (privileged span, stub, prefix, answer history)
   rather than every key. Position distributions are not vocabulary
   distributions, so the known completion-groove failure of Fisher/lens-KL need
   not transfer, but head entropy and near-zero probabilities can make KL
   brittle; temperature and masking must be pinned. Architectural caveat:
   fused/flash kernels may not expose attention probabilities, and hybrid
   attention/GatedDeltaNet models need a separate state-transition target.

6. **Value/output contribution matching — IMPLEMENTED AND QUEUED 2026-07-11.** Attention weights alone do not say
   what is written. Match each block's attention contribution after value and
   output projection (`O_L`, before residual addition), or separately match
   per-head value-weighted context vectors. Use normalized MSE/cosine and, where
   feasible, centered vocabulary-score cosine after the heads are recombined.
   This is more causally proximal than attention-KL and less underdetermined
   (different attention maps can yield the same useful update). It requires
   explicit hooks and careful definitions across GQA/MQA/fused kernels; saving
   full per-head targets is expensive, so begin with the recombined attention
   output. Keep the MLP contribution as a parallel target/control to determine
   whether retrieval or transformation is the limiting writer.

### Useful controls or higher-risk candidates

7. **Offline-whitened/Mahalanobis hidden matching — IMPLEMENTED AND QUEUED 2026-07-11.** Estimate a regularized
   activation covariance `Sigma_L` from a broad, frozen base-model calibration
   corpus, then minimize
   `(h_s-h_t)^T (Sigma_L + lambda I)^(-alpha) (h_s-h_t)` with
   `alpha in {1/2,1}`. This asks whether `vocab_mse` wins because vocabulary
   geometry is special or because it conditions anisotropic activations.
   Per-item whitening/CKA is invalid here (`A` is often smaller than hidden
   width and the covariance is rank-deficient); covariance must be accumulated
   offline, shrinkage-regularized, frozen, and estimated independently per
   layer. Clip inverse eigenvalues and report condition numbers. A low-rank
   eigensystem plus isotropic remainder avoids an `H x H` device buffer.

8. **Base-anchored trajectory preservation at every layer — IMPLEMENTED AND QUEUED 2026-07-11.** On general anchor
   text, match the trained student's states to the frozen base model using
   `nmse` or `vocab_mse` at every layer, while recall items retain the teacher
   trajectory objective. Output anchor-KL only observes final behavior; this
   version can prevent hidden damage before it becomes visible at the head and
   directly tests whether 1.7B intrusion is written mid-stack. It is a
   preservation loss rather than a new recall metric, and must be reported as
   such. Balance by alternating recall/anchor batches rather than increasing
   anchor weight with depth; compare destruction, recall, and parameter-update
   norm at matched item budgets.

9. **Cross-layer relational/flow loss — IMPLEMENTED AND QUEUED 2026-07-11.** Match teacher and student similarities
   between successive layer representations for the same tokens, e.g. the
   cosine matrix between `h_{L-1}` and `h_L`, or the normalized change in token
   Gram matrices. This supervises how geometry evolves without insisting that
   every coordinate or scale match exactly. It is related to delta matching but
   measures transformation of relations rather than vector displacement.
   Rotation invariance again makes it insufficient alone; pair it with a small
   coordinate-anchoring term. Only adjacent or uniformly sampled fixed offsets
   are legal—an output-biased layer pairing would violate the naming contract.

10. **Contrastive trajectory loss (InfoNCE / soft nearest-neighbor) — IMPLEMENTED AND QUEUED 2026-07-11.** Treat the
   same token and layer in teacher/student as the positive and other positions
   (preferably other examples) as negatives. This may retain token identity and
   prevent collapsed direction-only solutions. In-sequence negatives are often
   false negatives in repeated verse, and batch size one gives a weak/biased
   denominator; use a detached teacher queue or semantically filtered negatives.
   Temperature strongly changes gradient scale. Run only after a collision-rate
   audit, and always combine with an absolute metric because contrastive
   alignment does not guarantee frozen-head compatibility.

11. **Untied input-embedding metric — IMPLEMENTED AND QUEUED 2026-07-11.** On models whose input embedding and
    unembedding are genuinely untied, define the quadratic metric induced by
    the frozen input embedding, analogous to `vocab_mse`, or compare centered
    scores under that matrix. It supplies an independent semantic geometry and
    is identical/redundant on tied models. This is primarily a mechanistic
    control: if it matches unembedding performance with less intrusion, output
    vocabulary geometry is not uniquely necessary. Normalize by teacher energy
    and verify orientation/scaling because model families implement tied and
    untied heads differently.

12. **Robust/adaptive combinations — IMPLEMENTED AND QUEUED 2026-07-11.** Existing Huber is fixed-scale. Untested
    robust choices include per-token pseudo-Huber/Charbonnier, clipped NMSE,
    and an uncertainty-balanced sum of state, delta, and relational losses.
    These may stop a few high-error tokens/layers from setting the update.
    Avoid learned unconstrained weights: they can silently create depth bias.
    Prefer fixed global weights, GradNorm-style balancing with a single shared
    coefficient per loss family, or normalize each component by its frozen
    epoch-0 gradient norm. Log both raw and weighted loss per layer and epoch.

### Low-priority bound checks (not expected winners)

13. **Reverse or symmetric teacher-distribution divergence — `lens_js`
    IMPLEMENTED 2026-07-11; do not implement again.** Reverse KL
    `KL(student || teacher)` is mode-seeking; Jensen-Shannon and temperature-
    softened symmetric KL are bounded/more balanced. These remain shaped by the
    teacher vocabulary distribution and therefore inherit the measured groove
    risk (`vocab_fisher` intrusion 57.5%, `lens_kl` 90%). They also cannot supply
    information absent from the teacher's distribution, so C2-34 predicts they
    cannot solve the last-3% readout problem. At most run one small loss-safety
    arm as a tightness/control experiment, not a broad sweep.

### Required screen before a full loss sweep

Implement one loss at a time and certify the trainer before launching. First
run a short mechanics/locality job, but never terminate a real training arm
before 12,000 items. Then compare on the same two promising model/checkpoint
families, identical data order and item budget, evaluating recall for the
checkpoint's actual corpus/corpora plus the standard benchmark damage subset at
epoch 0 and every epoch. Persist raw and weighted loss, gradient norm/share,
update norm, and per-layer values for every epoch. Continue past 12,000 items
only while recall is improving without crossing the predeclared destruction
budget; a falling proxy loss alone is not evidence of useful learning. Priority
order after the running matrix: state+delta, base-anchored trajectory,
attention-output matching, relational+state, then offline-whitened NMSE. Do not sweep Fisher-like
or reverse-KL variants until those geometry-based candidates have been tested.

## Campaign roadmap beyond C2 (sketched 2026-07-04, owner question)

- **C3 — conversations (Stage B):** conversation-to-weights (privileged =
  oldest turns; QA-about-censored-turns eval); attention-scored span
  selection via the head taxonomy ("worth of attention" operational);
  cycle mechanics: early-stop-on-readout (C2-6), heterogeneous batching
  (C2-8/9), destruction gate as automated accept/reject; gpt-oss
  thinking_selective with harmony harvests; 8B-14B full-FT via
  offload_adam + sliding-window prefetch; before/after MoE routing-shift
  probe (C2-15 follow-up).
- **C4 — the person (Stage C):** 120B MoE on H100s, streamed-block
  consolidation during serving idle; primary metrics = RAG-independence
  curve + query-sophistication drift (docs/evolving_person.md); weeks-long
  continual run with nightly destruction gate (slow-drift watch); fleet
  evolution: experience-log replay vs gated diff-merging; intrusion
  metric as privacy audit; live-Socratic demo as the closing exhibit.

C2 built the instruments; C3 masters the unit of experience (one
conversation, one cycle); C4 composes cycles into a life.

## C3 model candidates beyond 14B (scouted 2026-07-04, web sources in chat)

Constraint: 4x L40S 46GB (184 GB node; 92 GB per 2-card PP job); need
thinking mode + tool use + HF layout compatible with BlockStack.

- **Qwen3.6-35B-A3B** (Apr 2026, Apache 2.0): 35B-total/3B-active sparse
  MoE. Primary C3 candidate — family continuity with the whole ladder,
  2-card PP (~70 GB bf16), MoE-router instrument applies, thinking mode.
- **Gemma 4 26B-A4B** (Apr 2026, Apache 2.0): 25.2B MoE, 3.8B active,
  256K ctx, native tool use / MCP. Second family for generality; 2-card.
  Also 31B dense (workstation tier) as a dense scale point; 12B unified
  multimodal as single-card option.
- **DeepSeek V4 Flash**: 284B/13B-active, 1M ctx ("Engram conditional
  memory" — relevant to our memory program conceptually). FP8 ~284 GB »
  our node; INT4 ~142 GB would fit 4 cards but quantized hidden states
  are a research risk for trajectory matching. C4-class target on H100s
  (the owner's 4xH100 scenario), alongside gpt-oss-120b.
- GLM-4.7-Flash (~30B MoE) / GLM-5.2, Kimi K2.7 (1T/32B active): noted;
  K2-class is beyond any near-term node.
- Caveat: blog-grade specs — verify model cards + licenses + BlockStack
  layout (fails loudly by design) before committing arms.

**Owner addition (2026-07-04): Qwen3.6-27B as the parallelism bridge
model.** Dense 27B (Apr 2026, Apache 2.0, thinking mode, 262K ctx,
SWE-bench 77.2): two-card on L40S (54 GB bf16 → PP2/TP2 mandatory) AND
one-card on H100 80GB (traditional reference possible). Plan: once TP+PP
are understood on current models (PP2 repro + 32B arm in flight), run
the same 27B recipe as {single-H100 reference, PP2, TP2} and compare —
parallelism correctness against a no-parallelism ground truth, and
layerwise-vs-traditional at a size both can run. Its "Thinking
Preservation" mechanism is adjacent to thinking_selective — investigate
at harvest time.

Single-L40S 27B addendum: bf16 impossible (54 GB weights alone), but the
official Qwen3.6-27B-FP8 checkpoint (~27 GB) + bf16 LoRA + adapters-off
teacher ≈ 31-33 GB fits one card. Risks: FP8 forward through our block
walk + kernels==0.12.0 pin; SCIENCE: FP8-quantized teacher trajectories
(what does trajectory distillation lose under a quantized teacher? —
same question that gates INT4-base training of V4-Flash-class at C4).
Full 27B grid: {1xH100 bf16 ref, PP2 bf16, TP2 bf16, 1xL40S FP8-LoRA}.

Qwen3.6 compatibility check (2026-07-04, transformers 5.12.1 — no
upgrade needed, kernels pin safe): configs load; 3.6 reuses qwen3_5
classes. 27B = MULTIMODAL composite (text_config: qwen3_5_text, 64
layers, hidden 5120, UNTIED head — PP-friendly) + vision_config; the
text tower is not at model.model.* → BlockStack and _pp_device_map need
a small layout adapter (the designed fail-loudly path, docs/scaling.md).
35B-A3B = qwen3_5_moe_text, 40 layers, 256 experts top-8 (finer routing
than gpt-oss's 32 — better router-probe resolution). Adoption cost ≈
half a day: layout adapter + template-pieces verification + thinking
harvest ("Thinking Preservation" mode). Why the 3.6 series was absent
from C1/C2: released 2026-04, post-dating the program design and the
assistant's knowledge cutoff — an inertia blind spot caught by the owner
2026-07-04; matched-ablation continuity justified staying on Qwen3
within-campaign, but C3 arms should default to 3.6-generation bases.

## Do-not-rebuild knowledge (measured negatives — referenced by AGENTS.md)

Standing guidance, not open work. Timing regimes are NOT comparable
across the 2026-07-10 refactor boundary — never mix pre/post ms-per-item
numbers in one table.

- NEGATIVE (2026-07-10): async pinned-memory target prefetch (side CUDA
  stream, per-tensor pin_memory, event-synced staging of layer L+1) was
  implemented and MEASURED SLOWER on L40S at 0.6B: item mode -9%,
  slide8-dedup padded B4 -25%, no memory win. Do not rebuild without
  first measuring a pinned-POOL variant at 4B+ scale where targets are
  >20 MB/layer (1.7B targets are ~2.4 MB/layer — out of scope).
- PP2 is SLOWER than single-GPU for this depth-sequential workload in
  every measured variant (hooked, hook-free, pre-moved inputs) — PP is a
  memory technology here. ~8% of the isolated PP2 walk is accelerate
  hook dispatch (pytree traversal, not transfers). Within a grad-accum
  window weights are frozen, so cross-item device overlap (item i+1 on
  partition 0 while item i runs partition 1) would be EXACT, not stale —
  the honest PP throughput move if ever needed. TP2 remains probe-only:
  collectives inside every linear lose badly at trainable sizes; use PP
  at block boundaries, TP only if a single block cannot fit.
- Speed regimes measured 2026-07-11 (1.7B, L40S): bucketed B4 = 2.7x
  item B1 at 30.6 GB; batched eval (generation_batch 8) = 2.8-3.1x. REVIEW
  CORRECTION 2026-07-11: the original B1/B8 accuracy comparison was
  confounded because short rows inherited the batch maximum generation
  budget. Per-row decode truncation now restores the B1 budget contract;
  only a fresh comparison may quantify bf16 tie-flips. Final science evals
  stayed B1 and are unaffected; historical per-epoch telemetry from the five
  flipped arms remains regime-confounded and must be labeled as such;
  card-packing via offload_adam does not raise throughput while arms are
  compute-saturated (91-98% util at B=1). The B1/B4 grid fork is labeled
  via the layer_loss_manifest `regime` column.
- Pipeline-v3 B1/K1 dispatch screen (2026-07-15, Qwen3-0.6B, longest
  answer, 256 aligned tokens, L40S): minimum-memory per-block ran at
  8.74 token-events/s; one disconnected backward per token reached 9.87.
  A serial answer anti-diagonal was 9.76, and a bounded threaded/CUDA-stream
  student pipeline was 9.84 while raising incremental peak allocation from
  370 to 800 MiB. Post-accumulation optimizer-in-backward hooks were slower
  on the student path (9.09 token-events/s) than the fused post-backward
  write. These schedules were parameter-delta equivalent and passed cache/
  vocabulary tripwires, so the negative is execution speed, not numerics.
  Teacher-hidden independent layer lanes improved 8.71 -> 10.71; grad-ready
  writes reached 11.89 with identical layer deltas; partitioning the same 28
  lanes over three L40S GPUs regressed to 11.12. All remain far below a useful
  campaign rate. Do not claim that rearranging one-GPU dispatch solves v3:
  fixed tiny forward/backward launch overhead remains dominant. The next
  distinct probes are multi-GPU teacher-layer partitioning and fixed-shape
  capture/fusion, not more one-GPU lane variants.
- Pipeline-v3 stale-window screen (2026-07-15, Qwen3-0.6B, same 256-token
  longest answer and seed): K=1 measured 10.87 token-events/s and K=8 measured
  82.69 (7.61x) with essentially flat incremental memory (82.95 vs 83.32 MiB).
  The exact trainable-delta comparison gives global relative L2 divergence
  0.154 and cosine 0.9889. Median per-layer divergence is 0.160; layer 1 is
  the clear outlier at 0.744/cosine 0.766. This is a one-answer calibration,
  not a quality verdict; the matched 12k-item K sweep must report recall,
  damage, and full per-layer dynamics. Longer-window throughput on the same
  probe reached 165.54 at K=16, 604.78 at K=64, and 1435.17 answer-wide.
  An intact K=64 control initially exposed future-token leakage when the
  review's mask-free q=1 optimization was over-generalized to chunks. With a
  shared K×prefix causal mask, intact loss returned to 2.0e-6--9.2e-6, total
  parameter delta to 7.6e-7, and throughput to 605.09 token-events/s. Keep the
  mask-free path q=1-only.
- Pipeline-v3 fixed-shape capture probe (2026-07-15, Qwen3-0.6B, 256-token
  answer): static-cache eager was numerically equivalent to dynamic-cache K=1
  (relative trainable-delta divergence 1.66e-5, cosine 0.9999999999) and ran at
  9.96 token-events/s. CUDA-graph replay ran at 51.8--53.2/s with 278--294 MiB
  incremental peak memory, but reproducibly diverged 0.0116 in relative delta
  L2 (cosine 0.999939). Repeated graphs matched each other; replacing Python
  `param.grad=None` with fixed gradient buffers explicitly zeroed inside the
  capture did not change the divergence. Do not promote capture until that
  graph-vs-eager residual is explained or accepted by an explicit numerics
  policy. Fixed cache shape itself is exonerated.
- Pipeline-v3.1 simultaneous-user probe (2026-07-15, Qwen3-0.6B, one median
  bucket): B256K1 reached 290.0 tile token-events/s and 69.4/s including the
  uncensored teacher batch plus prompt prefill, using 10.45 GiB peak. B256K16
  reached 4,055.5/s tile and 1,083.5/s end-to-end at 11.70 GiB peak. Both made
  28 physical block writes; gradients were unaveraged sums over 256 and 4,096
  valid cells respectively. K16 is not ordinary next-token online execution:
  it requires teacher-prefetched or confirmed speculative tokens. The probe
  supports full-attention Qwen3-0.6B only; Qwen3.5's alternating
  linear-attention/full-attention state needs a v3.1 adapter before the 0.8B
  campaign.
- Qwen3.5-0.8B mixed-budget target regression (2026-07-15): the fixed
  `--generation-max-tokens 4096` H100 run produced 975,919 tokens in 47.23 s,
  stopped naturally on 93.19% of examples, and scored 0.6087. Replacing that
  ceiling with the exact per-record formula (often 104--116 tokens for short
  Machado answers) reduced generation to 245,263 tokens/19.93 s but raised
  hard cuts to 58.52% and lowered score to 0.5260. Preserved cut samples end
  mid-sentence/mid-word after verbose hallucinated framing, so this is not a
  stop-sentinel or cache-reader bug. Do not use the resulting 2,071-example
  hidden cache for scientific claims. The older good response JSON lacks
  exact token IDs; regenerate at 4096 with the current writer, rerun the RAG
  gate, and mint a fresh cache identity. A 27-second generation saving is not
  a valid trade for changed teacher behavior.

## Open items (2026-07-11; closed work lives in git history)

STILL OPEN — deferred with explicit reasons:
- examples_v5: BUILT 2026-07-12 under the new question-only contract
  (owner: dataset = conversational questions + master-RAG tool turn, NO
  answers; the teacher GENERATES the answer at the cache stage and the
  student trains on its forward hidden states; answers are per-model
  cache content, never dataset content — anchoring a dataset to a model
  makes a bad dataset). Two RAG scopes built (window/chapter,
  data/combined/examples_v5_*.jsonl, full-corpus coverage asserted);
  censoring axis remove vs pad_random (length-matched random
  non-repeating ordinary-token fill — fixed pads/repeated fillers are
  attendable attractors). Ladder campaign queued:
  scripts/queue_v5_ladder_20260712.tsv (3 best grid losses x 4 teaching
  styles x 0.6B/1.7B/4B; per-rung recitation ceilings gate the arms
  against the RAG-authority failure recorded above). v4 stays
  byte-guarded. OPEN remainder: 4B jacobian lens artifact
  (../jacobian-lens) before the 4B x jacobian arms; online-teacher
  generation for open-answer datasets (disk cache required until wired).
- 0.6B v5 smoke telemetry (2026-07-12): premise contrast healthy with
  window RAG (CE 0.55 with vs 3.70 without) but the teacher is chatty —
  66.7% of generations still hit the 2x cut after the +32 framing
  margin (it recites MORE than asked); word-LCS vs target 0.34. Teacher
  behavior for the ladder ceilings to measure, not a pipeline bug; cut
  spans end with a forced <|im_end|>.
- Bench-gated speed items: pinned-POOL prefetch only at 4B+ (>20
  MB/layer targets, per the measured negative at 0.6B); cross-item PP
  overlap only if PP lanes become throughput-relevant.
- IDEA, NOT IMPLEMENTED (owner, 2026-07-12): **student-step CUDA graphs**.
  This is distinct from vLLM's successful CUDA-graph capture of a static
  autoregressive *inference* decode loop; nothing in the current trainer uses
  `torch.compile` or CUDA graphs. A PyTorch graph could capture a fixed-shape,
  resident full-FT micro-step (forward → depth-uniform local/window loss →
  backward → optimizer step), but cannot be assumed to cover the current
  variable-length examples, variable spans/window endpoints, CPU/offloaded
  optimizer states, lazy target copies, or host-side telemetry. A valid first
  experiment must bucket padded sequence length and pin a single
  `(microbatch, length bucket, window schedule)`; preallocate every input,
  loss, gradient, and optimizer buffer; move logging/CPU copies outside the
  capture; then compare to eager at identical numerics, update count, and
  peak VRAM. Start with `torch.compile` of the fixed-shape primitive, then
  capture only the resident path if it is stable. CUDA graphs reserve memory
  and may worsen the V5 OOM problem, so no production adoption without that
  memory-and-throughput measurement. Do not mistake vLLM generation gains for
  evidence that the layerwise training walk will gain similarly.
- Lens-diagnostics idea set (docs/lens_diagnostics_ideas.md, 2026-07-11
  section): all diagnostic-side; the regime-fork lens comparator is
  time-limited (needs the B=1 arms while fresh); the early-abort gate
  idea requires owner sign-off against the 12k-items law.
- IDEA, NOT IMPLEMENTED (owner, 2026-07-12): training-target grounding cutoff
  in build_teacher_cache.py's v5 generation step. The eval-battery budget fix
  this session (tasks.py: size generation budget on the RAG passage length,
  not the answer length — a teacher given enough room always stops naturally,
  no genuinely unbounded generation) is a DIFFERENT question from whether the
  TAIL of a long teacher generation is still actually grounded in the
  retrieved passage. Proposal: monitor, per generated span, whether the
  teacher's answer is still attending to / using the RAG passage, and if a
  late span has drifted into free-running continuation no longer grounded in
  it, cut the TRAINING TARGET there — training on an ungrounded tail defeats
  RAG-conditioned distillation even if the text itself is accurate. Two
  candidate signals discussed: (a) literal attention-weight mass on the
  passage token span at each generated position — the principled measure,
  but expensive during generation and noisy to threshold under GQA/sliding-
  window attention; (b) a cheap verbatim-overlap proxy reusing
  masking.find_poem_spans (already used for thinking_selective censoring) —
  conflates "grounded" with "quoting literally," would miss a still-faithful
  paraphrase. Recommended first step if picked up: a MEASUREMENT pass only
  (where does verbatim/attention grounding drop off across the v5 corpus),
  next to the existing premise-check contrast in build_teacher_cache.py,
  before deciding whether to act on it by truncating targets. Do not
  implement without a further owner decision.
- IDEA, NOT IMPLEMENTED (owner, 2026-07-13): **system-memory prompt
  rotation**. The v5 `rag_system` passage is intentionally embedded in a
  fixed system-message scaffold. That makes some scaffold tokens potentially
  load-bearing: a teacher or student could use their fixed positions/wording
  as a switch into the privileged-memory subspace, rather than representing
  the remembered text itself. Test this as a controlled prompt-family
  intervention: preserve role, meaning, privileged passage, question, answer
  budget, and tokenized passage span, while rotating semantically equivalent
  system wording, sentence order, and harmless padding/position across
  examples. Compare real-memory, no-memory, and random-memory controls under
  matched rotations. Do not call a result robust unless recall and premise
  separation survive unseen scaffold variants. This is not permission to move
  the passage into a user/tool/document turn: it must remain system-scoped
  memory. A stronger companion ablation is **full-system censoring**: give the
  teacher the complete system-memory turn but censor that entire turn in the
  student view, rather than deleting only the poem span. It removes every
  fixed scaffold token and its position as a possible subspace trigger.
  Compare it with passage-only censoring at matched question/answer geometry;
  it is a different student context and must be reported as such, not folded
  into the ordinary `rag_system` result.
- IDEA, NOT IMPLEMENTED (owner, 2026-07-13): **teacher-correct-only cache
  subset**.  Train on only those V5 examples for which the teacher's generated
  answer passes a predeclared reference-recall / task-appropriate correctness
  gate.  This may prevent a weak or misframed teacher from distilling obvious
  retrieval failures into the student.  It is also a selection intervention:
  it changes corpus coverage, answer-length mix, poem/location difficulty,
  and potentially the teacher's damage profile.  Any experiment must retain
  the rejected-example ledger and report acceptance rate plus recall/damage
  separately on accepted and rejected strata; compare against a same-item-
  budget random subset and the full teacher cache.  It must not use the
  original reference text as a training target: the reference is only the
  offline gate used to select teacher-sourced targets.

STILL OPEN — research (owner's C3 program, EXPERIMENTS.md): loss-grid
analysis (the original grid closed; the expanded Jacobian/lens queue and its
current completion count live in `runs/lossgrid_report.md`); teacher-stream
k-windows (C3 #1); H100 throughput/memory/PP-TP evidence (L40S evidence
complete); 1.7B cleanliness (intrusion 22-40% at 1.7B vs 1.5-2.5% at
0.6B — see the intrusion depth-localization probe idea).
# vLLM benchmark cloze aggregation scored missing `word_acc` as zero (2026-07-13)

`scripts/benchmark_vllm_generation.py` originally summarized every response
through `x.get("word_acc", 0.0)`.  That is invalid for V5 cloze records:
`_recitation_stats` intentionally emits `containment` because the deleted-word
reference is not stored.  Consequently all 249 cloze examples in the 2,071-row
V5RS corpus counted as zero in `mean_word_acc`, including correct outputs.

Measured on the Qwen3-14B H100 graph run, the published raw aggregate was
74.06%; combining each task's intended metric gives 85.32% (+11.26 percentage
points).  Case-folding the reference-word LCS adds only another 0.87 points on
non-cloze items and does not explain the remaining gap.  Inspection confirms
the main residual error is real: the model often copies the last line of the
RAG passage rather than the requested adjacent line (next: 89.43%, prev:
60.62%, cloze containment: 93.63%).

The next-line failures are not a target-index inversion: examples such as
`mach-v5-nx1-0003` ask "¿Me escribes el verso que sigue?", and `target_lines`
points to the immediately following verse (`en la feria de Berlanga`).  The
observed wrong answer (`Muy ricas las bodas fueron,`) is the tail of the
privileged passage.  Treat this as a prompt/RAG placement failure or teacher
behavior issue, not as evidence that the scorer expects the last line.

The benchmark now reports `mean_task_score`, `mean_word_acc` over applicable
next/prev examples only, and `mean_containment` over cloze examples only, with
their denominators.  Historical summary JSON files retain the old field
semantics; recompute their task-aware aggregates from the response JSONL before
using them in comparisons.

# Concurrent pipeline-v3 prompt prefill is Triton-autotuner unsafe (2026-07-17)

The sole Qwen3.6-27B PP4 timing optimization tried preparing four independent
activation shards in four host threads.  It failed before the first optimizer
write because the threads entered the same FLA/Triton autotuner concurrently;
Triton's shared launch state was cleared while another thread used it
(`TypeError: 'NoneType' object is not a mapping` at `self.nargs`).  This is not
an OOM and produced no throughput measurement.  `prefill_parallel_shards > 1`
is rejected at dispatch until prefill has process/stream-safe compiled kernels
or an explicitly serialized autotuning phase.  Do not retry by adding a Python
lock around the entire forward: that would serialize the intended overlap and
cannot establish a speed gain.

# Teacher-hidden PPn CPU use is strongly oscillatory (2026-07-17, open)

During the Layerwise 3.4 full-v5 overnight teacher-hidden launches, host CPU
load visibly alternated between high bursts and low intervals even though the
middle pipeline stages have no cross-stage activation work.  This is an
observation, not yet a diagnosed cache or scheduler defect.  The mode combines
mmap-backed full-input safetensor reads, pinned staging, cohort construction,
Python PPn dispatch, and occasional TorchInductor workers; instantaneous total
CPU alone cannot distinguish them.

The follow-up must sample per-PID CPU, major/minor faults, read bytes, native
thread count, and compiler-worker presence alongside cohort/tile telemetry and
the one-second GPU trace.  In particular, test whether bursts align with cache
page faults or cohort admission and whether low-CPU intervals align with one
GPU computing while peer stages wait.  Do not infer that teacher inputs are
served from RAM merely from low aggregate CPU, and do not increase thread caps
until this correlation is measured: prior shape-varying compile pools already
caused severe CPU oversubscription on these nodes.

# GPU teacher-input cache defeated activation-shard memory bounding

Status: resolved by target reuse on 2026-07-17; full-v5 admission remains the
production gate.

The full-v5 Qwen3.5-0.8B PP2 Huber and cosine arms OOMed twice, first with
`activation_shard_users: 64` and then 32, at the same 43.44-GiB allocation.
Inspection of `_bk_prepare_cohort_shards` explains the invariant footprint:
for `teacher_hidden_source: gpu_cache`, every layer tensor of every shard is
copied to its block-owner GPU and retained in the returned `shards` list for
the whole cohort.  Halving shard width merely doubles the number of resident
shards; it bounds backward-graph size but not total teacher-input residency.

The fix removes the redundant full teacher-input residency rather than falling
back to CPU-cache semantics.  During prompt prefill, one full teacher layer is
packed transiently and only `i1=h0` persists for answer tiles.  During a tile,
the owner copies only its active BxK target range to its GPU; `h[L]`, already
needed for block L's loss, is detached and reused exactly as `i[L+1]`.  A
startup tripwire checks that equality over every cached boundary before any
write.  The first corrected 0.8B PP4 gate matched the prior per-layer losses
and gradient norms exactly, had checkpoint relative-L2 drift 1.57e-13 with
cosine 1.0, and reduced aggregate reserved VRAM from 42.66 to 24.25 GiB.

Production still requires a full-v5 first-cohort admission because the
100-question gate cannot certify the largest B256 cohort.  Do not silently
fall back between `gpu_cache` and `cpu_cache`: source mode remains an
experimental variable and belongs in telemetry.
