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

## OPEN — untested same-width teacher/student losses (owner question, 2026-07-10)

Scope: losses below have not been campaign-tested as trainer objectives in this
branch (some already exist as diagnostics). Every target is produced by the
teacher or frozen base model; none uses reference-text labels. Every per-layer
term must use the same coefficient at every depth (or a depth-uniform sampled
alternancy), embeddings/head remain frozen, and connected credit remains a
sanctioned sliding window with `conn_stride: 1`. This is an idea ledger, not an
implementation commitment.

### Implementation order (active work only)

1. **State + `delta_vocab_cos`**, with raw/weighted per-layer telemetry and
   matched update norm. This is the next clean objective after the running
   delta-only grid.
2. **Base-anchored trajectory preservation**, alternating recall and general
   anchor batches. Highest-priority destruction-tax intervention for 1.7B.
3. **Attention-output + MLP-output contribution matching**, starting from
   recombined sublayer outputs rather than attention probabilities.
4. **Relational token geometry + absolute state**, never relational alone.
5. **Offline-whitened NMSE**, after a frozen covariance artifact and condition-
   number report exist.
6. **Multi-scale delta**, initially `k={1,2,4}` only. It differs from sliding
   connectivity but is partially redundant, hence below state+delta.

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

3. **Multi-scale/cumulative trajectory matching — ACTIVE PRIORITY 6.** Match short finite
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

4. **Relational token-geometry distillation — ACTIVE PRIORITY 4, only beside
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

6. **Value/output contribution matching — ACTIVE PRIORITY 3.** Attention weights alone do not say
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

7. **Offline-whitened/Mahalanobis hidden matching — ACTIVE PRIORITY 5.** Estimate a regularized
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

8. **Base-anchored trajectory preservation at every layer — ACTIVE PRIORITY 2.** On general anchor
   text, match the trained student's states to the frozen base model using
   `nmse` or `vocab_mse` at every layer, while recall items retain the teacher
   trajectory objective. Output anchor-KL only observes final behavior; this
   version can prevent hidden damage before it becomes visible at the head and
   directly tests whether 1.7B intrusion is written mid-stack. It is a
   preservation loss rather than a new recall metric, and must be reported as
   such. Balance by alternating recall/anchor batches rather than increasing
   anchor weight with depth; compare destruction, recall, and parameter-update
   norm at matched item budgets.

9. **Cross-layer relational/flow loss**. Match teacher and student similarities
   between successive layer representations for the same tokens, e.g. the
   cosine matrix between `h_{L-1}` and `h_L`, or the normalized change in token
   Gram matrices. This supervises how geometry evolves without insisting that
   every coordinate or scale match exactly. It is related to delta matching but
   measures transformation of relations rather than vector displacement.
   Rotation invariance again makes it insufficient alone; pair it with a small
   coordinate-anchoring term. Only adjacent or uniformly sampled fixed offsets
   are legal—an output-biased layer pairing would violate the naming contract.

10. **Contrastive trajectory loss (InfoNCE / soft nearest-neighbor)**. Treat the
   same token and layer in teacher/student as the positive and other positions
   (preferably other examples) as negatives. This may retain token identity and
   prevent collapsed direction-only solutions. In-sequence negatives are often
   false negatives in repeated verse, and batch size one gives a weak/biased
   denominator; use a detached teacher queue or semantically filtered negatives.
   Temperature strongly changes gradient scale. Run only after a collision-rate
   audit, and always combine with an absolute metric because contrastive
   alignment does not guarantee frozen-head compatibility.

11. **Untied input-embedding metric**. On models whose input embedding and
    unembedding are genuinely untied, define the quadratic metric induced by
    the frozen input embedding, analogous to `vocab_mse`, or compare centered
    scores under that matrix. It supplies an independent semantic geometry and
    is identical/redundant on tied models. This is primarily a mechanistic
    control: if it matches unembedding performance with less intrusion, output
    vocabulary geometry is not uniquely necessary. Normalize by teacher energy
    and verify orientation/scaling because model families implement tied and
    untied heads differently.

12. **Robust/adaptive combinations**. Existing Huber is fixed-scale. Untested
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

## Open items (2026-07-11; closed work lives in git history)

STILL OPEN — deferred with explicit reasons:
- examples_v5 + teacher-cache regeneration: queued GPU work for the
  post-grid campaign boundary (v4 stays byte-guarded for comparability
  with the completed loss grid).
- Bench-gated speed items: pinned-POOL prefetch only at 4B+ (>20
  MB/layer targets, per the measured negative at 0.6B); cross-item PP
  overlap only if PP lanes become throughput-relevant.
- Lens-diagnostics idea set (docs/lens_diagnostics_ideas.md, 2026-07-11
  section): all diagnostic-side; the regime-fork lens comparator is
  time-limited (needs the B=1 arms while fresh); the early-abort gate
  idea requires owner sign-off against the 12k-items law.

STILL OPEN — research (owner's C3 program, EXPERIMENTS.md): loss-grid
analysis (the original grid closed; the expanded Jacobian/lens queue and its
current completion count live in `runs/lossgrid_report.md`); teacher-stream
k-windows (C3 #1); H100 throughput/memory/PP-TP evidence (L40S evidence
complete); 1.7B cleanliness (intrusion 22-40% at 1.7B vs 1.5-2.5% at
0.6B — see the intrusion depth-localization probe idea).
