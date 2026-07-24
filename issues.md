# Issues / Follow-Ups

## PRIORITY — retain loss telemetry by cohort without slowing the v4 hot loop (2026-07-24)

The read-only v4 report can plot block-local loss by epoch and layer, but the
trainer does not currently retain a loss series at cohort/update granularity.
Add enough telemetry to diagnose within-epoch convergence, outlier cohorts,
length-bucket effects, and the difference between B16/B32 without reintroducing
the sync-bound failure mode.

**Non-negotiable performance constraint:** this must not add `.item()`,
`.cpu()`, host logging, printing, file I/O, or a CUDA synchronization inside
the `(layer, cohort)` walk.  The payload is small, but byte count is not the
risk: an otherwise tiny device-to-host observation can serialize the GPU.
Accumulate detached loss summaries in preallocated device tensors and transfer
them only at an existing accumulation/epoch boundary, preferably in one
batched asynchronous copy.  Bound memory explicitly and retain cohort identity
or its deterministic bucket/index mapping in the flushed row.

Acceptance requires:

1. reports can plot loss versus cohort/update for every owned block, with
   cohort width/length metadata and an unambiguous epoch/launch identity;
2. no new synchronization primitive or per-cohort Python formatting occurs in
   the hot loop;
3. an H100 A/B gate over the real sequence-length distribution shows no
   statistically meaningful regression in `epoch_seconds`,
   `token_events_per_second`, or GPU utilization; any measurable regression
   blocks the feature rather than being accepted as an observability tax; and
4. telemetry can be disabled or downsampled without changing optimizer
   semantics, update order, cohort construction, or numerical results.

## OPEN — teacher-cache answer generation wastes the GPU (H100, 2026-07-17)

**Owner call: must be patched before the ~2,000-prompt run. It was good enough
for the bring-up training test and was not blocking it.**

Measured on `agpuh01` (H100 80GB, Qwen3-0.6B, `build_teacher_cache.py` with an
empty `cache.generation_responses_path`, so the builder generates the answers
itself with a greedy Transformers decode loop):

- 100 open-answer v5 records, 8,260 generated tokens in **365.06 s**.
- **22.6 tok/s aggregate at `effective_batch: 64`** — an H100 idling.
- The first flushed batch was worse: 13.2 tok/s at effective_batch 36.
- No error, no OOM: it is simply slow.

For scale, `demos/` measured **vLLM at 285.9 tok/s on CPU** for the same v5
workload and model, and the torch CPU decode loop at 34.5 tok/s. An H100 doing
22.6 tok/s is therefore not a GPU problem at all — it is the same Python/decode
bottleneck `demos/README.md` already dissected: `DynamicCache` does a
`torch.cat` per layer per step (the whole KV cache is reallocated and copied
every token), and left-padding forces SDPA off the fused kernel.

**Why it matters now.** This smoke used the 100-item deciepoch subset. The full
v5 set is ~2,071 items: at the measured rate that is roughly **2 hours of a
mostly-idle H100 per model/cache identity**, before a single training step.

**The intended escape already exists and is unused here.** The campaign configs
set `cache.generation_responses_path` to pre-generated vLLM output
(`scripts/benchmark_vllm_generation.py`, `scripts/l40s_vllm_teacher_campaign.sh`),
which skips the builder's decode loop entirely and feeds it finished answer ids.
Those response files live under `runs/` and are not present in this checkout.

Candidate fixes, cheapest first:

1. Generate the answers with vLLM once per model and point
   `cache.generation_responses_path` at the result (the sanctioned path; the
   builder already hashes the response file into the cache identity).
2. Failing that, give the builder's decode loop a `StaticCache` and
   length-sorted batches — `demos/` showed the per-step math is not the
   bottleneck (a 0.6B decode microbench sustained 317 tok/s on 32 CPU cores).

Do not "fix" this by shrinking the dataset.

## OPEN (known bug) — PP3 allocates on a 4th card it does not own (2026-07-17)

Owner-confirmed known bug; recorded here with fresh H100 evidence. Not fixed,
not blocking the bring-up test.

A PP3 run pinning `model.pipeline_devices: [0, 1, 2]` also takes memory on
card **3**, which it never declared:

```
# nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
GPU-717a...d86b, 1122757, 1508 MiB   # PP3, card 0   (declared)
GPU-415e...cb52, 1122757, 1182 MiB   # PP3, card 1   (declared)
GPU-91f9...9adb, 1122757, 1606 MiB   # PP3, card 2   (declared)
GPU-c681...2d4e, 1122757,  518 MiB   # PP3, card 3   <-- NOT declared
GPU-c681...2d4e, 1122930, 2230 MiB   # PP1, card 3   (its own run)
```

The 518 MiB is context-sized rather than weight-sized, so the layer placement
itself is correct (blocks really do live on 0/1/2 — the split is genuine, not
collapsed). It looks like a CUDA context created on every *visible* device
rather than only on the mapped ones.

**Why it is not cosmetic.** `pipeline_devices` is the contract that says which
physical cards a run owns. Allocating outside it means:

- a co-scheduled job on card 3 silently shares with a run that claims not to be
  there, and AGENTS.md already records that "scheduler VRAM reservations are
  launch-time checks, not leases" — a 3 GB stray is exactly the kind of margin
  intruder that OOMs a big neighbour later;
- per-card VRAM attribution in reports is wrong for any PP run;
- it scales with card count, not with the split.

Reproduce: the two `configs/experiments/h100_smoke/` overlays, run together on
one 4-card node (PP3 on 0-2, PP1 on 3). Suspect the device-context creation in
`train/runtime.py` placement rather than `pp_device_map`, whose map was
verified correct here.

## OPEN — v3.2 training leaves the H100s nearly idle (agpuh01, 2026-07-17)

Sampled during the bring-up smoke with PP1 (card 3) and PP3 wavefront (cards
0-2) training concurrently, Qwen3-0.6B, B256/K16, LoRA, `micro_batch: 256`:

```
card:     0    1    2    3
        0 %  3 %  0 %  0 %
        1 %  0 %  5 %  0 %
        0 %  0 %  0 % 15 %
        0 %  0 %  0 %  8 %
        ... (15 samples, 2 s apart)
```

PP3's three cards sat at 0-5%; PP1's single card peaked at 8-15%. This is
consistent with the already-recorded dispatch-bound finding for pipeline-v3
(see "Do-not-rebuild knowledge": ~9-12 token-events/s on 0.6B L40S, one-GPU
lane/wavefront rearrangements did not help). Recorded here as an H100
confirmation on a fresh branch, NOT as a new claim: a 100-item epoch is short
and includes setup, and utilisation is a measurement, not an acceptance
condition. Read it alongside the teacher-cache issue above before optimising
anything — and read the measured NEGATIVE results first.

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
7. **Generative evaluator equivalence gate**: before a campaign, run the
   trainer's in-training generative evaluator and vLLM on the identical base
   state, prompts, tokenizer/chat template, stop policy, and deterministic
   decoding parameters.  Require identical generated token ids; record
   latency and throughput separately.  The trainer remains the scientific
   evaluator and vLLM is an implementation/speed control.  A mismatch must
   abort the campaign and be diagnosed as context, tokenizer, or decoding
   drift — never worked around by exporting or merging a model.

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

The corrected 27B PP4 target-reuse run supplied the first correlated trace.
Across a 120-second window, mean utilization was 19--22% per card; GPUs 1--3
were exactly idle in roughly half the samples, and only 5/107 samples had even
two cards above 50% (none had three or four).  The parent trainer varied from
0.5 to 5.43 CPU cores and averaged 2.38.  It incurred many minor faults but
zero major faults and only 116 MB of actual reads, while the single two-worker
Inductor child averaged under 1% CPU.  This rules out Lustre stalls and an
external compiler storm for that window.

The resource-locality evidence is stronger.  agpul06 was 95--98% CPU-idle,
had 848 GiB available RAM and zero I/O wait, and no other process owned the
GPUs.  All four cards are on distinct NUMA nodes and every inter-card route is
SYS.  The trainer's resident/private placement was heavily skewed: about
71.9 GiB on NUMA 2, 39.7 GiB on NUMA 0, 13.9 GiB on NUMA 3, and 0.7 GiB on
NUMA 1.  The unpinned main tile producer was observed on CPU 34 / NUMA 2; it
packs the shared full-depth pinned target tile before the four stage workers
copy their owned ranges.  The next optimization should pin each stage worker
to its GPU-local CPU set and pack/first-touch only that stage's owned target
range locally.  Preserve tonight's run unchanged as the before trace; do not
retrofit this into a live scientific stream.

The hardware is not the limiting capability.  On agpul06 every GPU pair
reports P2P read/write support, `nvidia_fs` is loaded, and cuFile is configured.
The current trainer simply does not use those paths: ordinary safetensors mmap
reads are gathered with CPU `torch.stack`/copy into pinned host tiles before
H2D DMA.  `/dev/shm` tmpfs does not become a GPUDirect Storage source merely
because `nvidia_fs` is present.  A controlled follow-up must compare (1) the
current pinned-host baseline, (2) NUMA-local stage-owned packing, (3) direct
cuFile/GDS reads from a verified GDS-capable cache filesystem into stage-owned
GPU buffers, and (4) bounded GPU-resident active-window caching.  P2P belongs
primarily to ordinary student-hidden boundary transport; teacher-hidden has no
cross-card activation edge.  Keep cache source and residency explicit in
telemetry rather than treating hardware capability as automatic use.

# GPU teacher-input cache defeated activation-shard memory bounding

Status: resolved by target reuse on 2026-07-17; full-v5 admission passed.

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
write.  The first corrected 0.8B PP4 comparison matched the prior per-layer losses
and gradient norms exactly, had checkpoint relative-L2 drift 1.57e-13 with
cosine 1.0, and reduced aggregate reserved VRAM from 42.66 to 24.25 GiB.

The clean-source full-v5 PP2 admission then completed all nine B256 cohorts,
2,071 items, whole-set output evaluation, locality certification, save, and
`done`.  Peak reserved VRAM was 13.65/25.39 GiB instead of the prior
43.44-GiB OOM.  Do not silently fall back between `gpu_cache` and `cpu_cache`:
source mode remains an experimental variable and belongs in telemetry.

## OPEN — DeepSeek-V4-Flash PPP8 cross-node NCCL hang (spec_verify Phase 2, 2026-07-19)

**Owner decision 2026-07-19: ship DeepSeek's vLLM TP8 leg (independent of
training), defer this fix. Do not relaunch DeepSeek PPP8 training until this
is addressed.**

**UPDATE 2026-07-20: fixed and validated on a live 2-rank cross-node repro —
see "RESOLVED 2026-07-20" below (this diagnosis is left exactly as written
for the record; the fix is additive).**

Three attempts of `configs/experiments/h100_smoke/deepseek_v4_flash_ppp8_xnode.yaml`
(2x4 H100 cross-node, agpuh01+agpuh02) all failed before ever completing an
epoch. Diagnosed via a delegated static-analysis pass (no GPU touched) reading
BOTH sides of every stage boundary, not just the failing rank's own log.

**Root cause, high confidence:** store-fill traffic (`send_fill_one`/
`recv_fill_one`) and epoch-boundary relay traffic (`send_forward`/
`recv_forward`) share ONE NCCL process group by explicit design
(`online_v4.py:1499-1503`, "there is exactly one process group"). Stage3
(agpuh01, owns layers 17-22, skews toward the heaviest forced-eager
CSA/HCA attention types) was still genuinely mid store-fill after 70+ minutes
— not deadlocked, just slow (`stage3/metrics.jsonl` for the failing launch has
only 4 lines, no `v4_store_capture`; its own NCCL counter reads a clean
`last enqueued: 708, last completed: 708`, i.e. no backlog, just behind).
Stage4 (agpuh02, layers 23-27) meanwhile finished store-fill AND all 3
configured epochs (`v4_store_capture: seconds=3481.9`, three `v4_epoch_boundary`
rows) and its final relay `submit()` — designed to be non-blocking
(`_RelayServicer` docstring, `online_v4.py:760-775`, "SUBMITS the relay and
immediately returns to training") — hung inside `post_recv`'s `irecv` loop
(`relay_nccl.py:146-154`) waiting on a peer (stage3) that had never reached its
own epoch loop and so never had a matching `isend` to offer. Under NCCL's
"eager initialization" P2P mode (see the recurring PyTorch warning in every
stage's log about unbatched P2P ops serializing), posting an `irecv` can block
waiting for the peer's `isend` — defeating the `block=False` contract. This is
the exact mechanism behind the op-count asymmetry observed directly in the
crash logs: rank3's flight-recorder dump reports `last enqueued: 708, last
completed: 708` (clean, just behind) while rank4 reports
`last enqueued: 2072, last completed: 1405`, stuck on op #1406 — the two
sides were never going to reach matching counts because stage3 hadn't started
issuing its half of the epoch-relay traffic at all.

An existing fix already on HEAD (`cf5e461`, "v4 relay #24: finalize barrier +
longer NCCL timeout", raises `v4_nccl_timeout_s` 600->1800s) targets a
DIFFERENT race per its own commit message (a fast stage tearing down NCCL
while a slow sibling's eval tail still relays) and was verified only on a
Qwen 0.6B smoke bench, never against DeepSeek. It does not fix the mechanism
above, and would make a repeat failure take 3x longer (1800s, not 600s)
before the cascade becomes visible.

Two further, INDEPENDENT live issues surfaced in the same logs, not caused by
the above: stage6 self-aborted via the `v4_min_train_gpu_util` gate
(`online_v4.py:1443-1463`, `RuntimeError: UTILIZATION GATE`) partway through
its final epoch; stage7 (last stage, holds the LM head) hit a genuine
`OutOfMemoryError` in `summed.backward()` (`online_v4.py:1404`) during epoch 0
— distinct from the three store-fill OOMs below, which are already fixed.

Separately, `scripts/launch_v4_stages.sh:236-239` documents by design that the
local reaper does not watch remote-node pids — a remote-side stall is
invisible until NCCL's own timeout fires, which is exactly what happened here.

**Proposed fixes, not yet attempted:** (a) stop sharing one NCCL process group
between store-fill and epoch-relay traffic — a dedicated sub-group per
adjacent stage pair, or route epoch-relay telemetry through the existing
file/shm control plane instead of raw NCCL; (b) gate `post_recv()` behind a
cheap out-of-band "upstream has entered its epoch loop" signal so a
not-yet-ready peer can't turn a `block=False` call into a real block;
(c) deploy `v4_stage_reaper.sh` on remote hosts too; (d) separately size
stage7's backward memory and re-examine stage6's utilization-gate trip.

### RESOLVED (already on HEAD) — DeepSeek store-fill OOM, three separate causes

Three genuinely independent OOM causes in DeepSeek's PPP8 store-fill pass,
each fixed in its own commit, confirmed via `runs/h100_dsv4f_v4_ppp8x/stage4/
metrics.jsonl` (the 05:52 launch: `v4_store_capture {seconds: 3481.894,
cohorts: 1036}` then three full `v4_epoch_boundary` rows) to actually get past
store-fill once all three were in place:

1. Rotation buffer-pool hoarding: `BlockRotator`'s device pool used a
   count-based cap (`max_pooled=2`); DeepSeek's heterogeneous ~13-15GB
   per-layer-type buffers let `staged+inflight+pooled` coexist at ~60GB.
   Fixed in `a0362650` (byte-capped pool, `rotation.py:81`,
   `max_pooled_dev_bytes: 8 << 30` — pools zero buffers at this size,
   lets the allocator recycle instead of hoard).
2. Forced-eager CSA attention transient: DeepSeek's sparse lightning indexer
   can't use flash/SDPA, so `eager_attention_forward` materializes
   `combined_logits` plus a same-size softmax-stabilization copy at T~5000.
   Mitigated via `micro_batch` 32->8->2 (commit `8735ca7`).
3. Teacher-store on-device accumulation: `_resolve_residency`
   (`online_v4.py:407-450`) underestimated `FrozenDeepseekCtx`'s real size
   (sliding K/V + compressed entries + int64 top-k, ~650MB/cohort for top-k
   alone) and picked `gpu_corpus`, accumulating unboundedly across all 1036
   cohorts. Fixed in `65f6a33` (special-cases DeepSeek to `cpu_stream`
   residency, `online_v4.py:420-421`).

### Systemic finding (not DeepSeek-specific) — `certify_locality_v4` is a no-op for every stage-scoped+store PPP8 run

`certify_locality_v4` (`online_v4.py:1836`) only runs AFTER `train_online_v4`
completes (`layerwise.py:208-213`), so it has never been reached for DeepSeek
(every attempt has crashed inside `train_online_v4`). But even when reached,
`v4_stage_scoped: true` + `v4_teacher_source: store` (the combination every
PPP8 campaign on this branch uses) trips a guard at `online_v4.py:1852-1861`
that returns early: `{"passed": false, "skipped":
"stage_scoped_store_certification_pending_relay", "owner_note": "locality
certification debt, not evidence"}` — a placeholder, not real gradient-
isolation evidence. Confirmed identical in 397B's, Gemma-31B's, and
Qwen-122B's own "successful" PPP8 runs (`h100_q397b_v4_ppp8x`,
`h100_g31b_v4_ppp8x`, all log the same skip string). Real, non-skip
certification passes exist only for non-stage-scoped PPP4/PPP2 runs. The
"GATE: certify_locality_v4 must pass" language in PPP8 config comments is
presently unsatisfiable, as coded, for ANY stage-scoped+store PPP8 campaign —
including the 397B spec_verify run in progress as of this writing. The real
cross-node "cert relay" is explicitly deferred future work
(`online_v4.py:1855`), not a DeepSeek-specific gap.

### RESOLVED 2026-07-20 — cross-node NCCL hang fixed (out-of-band readiness
gate) + issues A/B fixed (config-only)

Built the minimal 2-rank cross-node repro the coordinator asked for instead
of re-launching full DeepSeek (`scripts/relay_nccl_hang_repro.py`): drives
the REAL `BoundaryTransport`/`NcclBoundaryRelay` from `relay_nccl.py`
directly with synthetic tensors across agpuh01/agpuh02, no model, no
DeepSeek. One rank plays a slow predecessor (stage3: still issuing
store-fill-style sends, has not reached its epoch-relay send), the other a
fast successor (stage4: calls `recv_forward(epoch, block=False)` in a poll
loop — exactly `_RelayServicer.submit()` -> `service(block=False)`).

**The repro REFINES the original mechanism, not just confirms it.** Every
single `recv_forward(block=False)` poll returned in 0.000-0.002s regardless
of how long the peer had left to stall — `post_recv()`'s `dist.irecv()` does
NOT block; the non-blocking contract is honored at the primitive level. The
real hazard is a silent **order-based data mismatch**, not a blocked call:
NCCL matches send/recv strictly by call order per (src,dst) pair, untagged,
and store-fill traffic shares that one ordered channel with epoch-relay
traffic by design. Posting the epoch-relay irecv burst before the
predecessor has issued every one of its store-fill sends gets those irecvs
satisfied by LEFTOVER store-fill tensors instead of the real boundary —
reproduced as `AssertionError: relay data mismatch` (`result[0]` held a
stale store-fill value, not the true relayed 99.0). A control run with no
interleaved traffic during the predecessor's stall (`--extra-fill-during-delay
0`) completes clean, isolating that the corruption specifically needs
unconsumed same-channel traffic ahead of the premature irecv burst — which
is exactly the real DeepSeek shape (stage3 genuinely still mid store-fill;
stage4 already past it). This also fully explains the op-count asymmetry in
the original diagnosis (rank4 "last enqueued 2072, completed 1405" vs
rank3's clean 708): rank4 prematurely posts a large irecv burst that only
rank3's eventual sends can satisfy; when rank3 is stuck for 70+ minutes,
most of that burst stays unmatched far past the NCCL timeout, and the
watchdog timeout/dump is what reads as "hung" from outside.

**Fix implemented (proposal (b), not the heavier (a) process-group split):**
a tiny second `TCPStore` (`NcclBoundaryRelay.__init__`, `relay_nccl.py`),
separate from the main NCCL group and from the node-local `/dev/shm` file
relay (which cannot carry this signal cross-node at all — confirmed by
reading `_RelayFiles`: it always resolves under `SELFUPDATE_V4_RELAY_ROOT`,
which `launch_v4_stages.sh` defaults to `/dev/shm`, invisible across
nodes). Hosted at `MASTER_ADDR`/`MASTER_PORT+1` (overridable via
`SELFUPDATE_V4_READY_PORT`), `is_master=(stage==0)` mirroring the main
rendezvous leader. Each stage calls `boundary_transport.mark_relay_ready()`
exactly once, right after `capture_relay_store()` returns in
`train_online_v4` (`online_v4.py`, unconditional on whether store-fill
actually ran — correct either way, since that's the earliest point at which
there is provably no more pre-relay traffic coming). `BoundaryTransport
.recv_forward()` now checks (or, for a blocking `drain()`, waits on) the
predecessor's flag via `NcclBoundaryRelay.predecessor_relay_ready()` /
`wait_predecessor_relay_ready()` before ever calling `post_recv()` for a new
epoch; non-blocking callers just see "not ready yet" (identical to an
unarrived boundary) instead of risking the mismatch.

**Validated on the same repro, post-fix:** with the predecessor stalling 12s
(sleeping only — no interleaved traffic, so nothing is left orphaned for the
gate to be defeated by) then marking ready, then stalling 6 MORE seconds
before its true send (simulating real local-epoch training after
store-fill, before the actual relay send) — every poll during BOTH stalls
correctly returned `None` with `post_recv` NOT YET posted while
`predecessor_ready=False`; the instant the ready flag appeared, `post_recv`
was posted and polling continued to correctly report "not arrived" (still
0.000-0.002s per call, so the gate adds no meaningful latency); the data
that finally arrived was correct (no mismatch, clean exit both ranks).

**Known limitation, documented honestly, not fixed further:** the gate
prevents PREMATURE posting relative to the predecessor's declared
readiness; it does not retroactively fix a channel that already has
UNMATCHED, unconsumed pre-relay traffic sitting in it (confirmed: forcing
orphaned sends in the repro — sends with no designated receiver anywhere —
still produced a mismatch even with the gate, since the gate only holds
back the RECEIVER's irecv, not clean up someone else's uncollected mail).
This is not a gap in the current production code path: `capture_relay_store`
always fully drains store-fill 1:1 (`recv_fill_one` blocks synchronously
per cohort) before returning, so by construction nothing is ever orphaned
before `mark_relay_ready()` fires. Flagging this invariant explicitly so a
future change to store-fill's pacing doesn't silently reopen the hole.

Files: `src/selfupdate/train/relay_nccl.py` (readiness store + 3 new
methods + the `recv_forward` gate), `src/selfupdate/train/online_v4.py`
(the `mark_relay_ready()` call site), `scripts/launch_v4_stages.sh`
(forwards `SELFUPDATE_V4_READY_PORT` to remote stages),
`scripts/relay_nccl_hang_repro.py` (the repro itself, kept as a permanent
diagnostic — reusable for any future cross-node relay change).

**Issues A and B (independent, both config-only fixes, no code changed):**
read the actual crash logs (`runs/h100_dsv4f_v4_ppp8x_stage6.log`,
`..._stage7.log`, 2026-07-19 05:52 launch) instead of re-deriving from the
summary in the OPEN diagnosis above.

- **A (stage6 utilization-gate self-abort):** `RuntimeError: UTILIZATION
  GATE (mid-epoch): rolling training-phase GPU utilization 48.9% < 50%
  floor after 3840 cohort steps`. Root cause: `v4_min_train_gpu_util: 50.0`
  is inherited from `base_deepseek_v4_flash_v4_full.yaml`, calibrated for
  RESIDENT weight placement — but the PPP8 experiment config separately
  sets `v4_weight_residency: rotate` (needed because DeepSeek's ~12 GB/layer
  MoE blocks don't fit resident) without also overriding the utilization
  floor. Every OTHER rotate-residency config on this branch (the
  `*_v4_ppp1_rotate.yaml` family: gemma4_26b/31b, qwen35_122b,
  qwen36_27b/35b) already sets `v4_min_train_gpu_util: 0.0` for exactly this
  reason — legitimate weight-paging stalls under rotation dip utilization
  below a floor calibrated for resident placement. This is the CLAUDE.md
  "knob copied without its enabling context" pattern, not a code bug. Fixed
  by adding `v4_min_train_gpu_util: 0.0` to
  `configs/experiments/h100_smoke/deepseek_v4_flash_ppp8_xnode.yaml`.
- **B (stage7 backward OOM):** `torch.OutOfMemoryError` inside
  `summed.backward()` in epoch 0: "Tried to allocate 10.40 GiB ... 7.96 GiB
  is free ... 70.92 GiB in use" of 79.20 GiB — a ~2.1 GiB deficit. Distinct
  from the three already-fixed store-fill OOMs (those are a no_grad forward
  pass; this is the training step's retained backward graph, a different
  and larger memory profile). Root cause: stage7 is the only stage that
  OOMed among the several 5-layer stages (0, 2, 4, 5) — it uniquely also
  carries the resident LM head/final norm, a fixed extra cost on top of the
  same per-cohort activation footprint that `micro_batch: 2` was tuned for
  (that tuning, the 32->8->2 arc, targeted store-fill's CSA-attention memory
  specifically, never validated against the training epoch's backward
  pass). Fixed by decoupling the two: added `v4_capture_micro_batch: 2`
  (config.py: "Only affects the no-grad teacher forward, never the training
  step") to freeze store-fill at its already-proven-safe batch size, and
  lowered `micro_batch` (now training-only) to `1`, in the same PPP8
  experiment config.

**Additional validation (same day, post-advisor-review):** the readiness
gate's blocking path originally used a raw `TCPStore.wait()`, which blocks
the FULL `v4_nccl_timeout_s` (up to 1800s) with no chance to notice a
sibling's cooperative stop -- the exact failure class `drain()` exists to
survive (2026-07-18 e500 finale). Replaced with a polling loop
(`BoundaryTransport._wait_predecessor_relay_ready`) that checks
`stop_requested()`/`rf.stop_seen()` every 2s and raises the existing
`_RelayStopped`, mirroring `_RelayFiles.wait()`'s own pattern so both halves
of the boundary transport honor a stop the same way. Separately, the WHOLE
fix was then validated on a REAL cross-node trainer run, not only the
synthetic repro: `h100_q0p6b_v4_ppp2_xnode`'s base config, 2 real stages
(agpuh01 + agpuh02), `v4_relay_every_cohorts` overridden to 1 to force the
relay path every epoch (the shipped config's default of 3 against 2 epochs
would never have called `relay.submit()` at all). Both stages completed
cleanly with checkpoints; stage1's `metrics.jsonl` shows
`student_trajectory_eval` firing for both epochs with
`trajectory: student_censored_flow_staged_relay` -- direct evidence the
gated `recv_forward()`/`post_recv()` path executed correctly over real
NCCL/IB, not just in isolation. `git log` on the `dsv4-relay-repro-afa4b8fb`
branch has both commits (NCCL fix + A/B config fixes, then this
STOP-awareness follow-up).

**Recommendation on a real DeepSeek PPP8 relaunch — precise framing:** all
three issues now have a fix validated either on a real 2-stage cross-node
trainer run (the NCCL hang) or by config reasoning against the actual crash
logs (issues A/B). This makes DeepSeek's failure mode CLEARER, not
DeepSeek's completion GUARANTEED: stage3's store-fill was still running
after 70+ minutes against the current 1800s `v4_nccl_timeout_s` in the
original crash. If that recurs, the fix converts what was an opaque
watchdog dump into a plain, correctly-attributed
`RuntimeError: relay-ready timeout ... waiting for stage 3 ...` -- a correct
and diagnosable outcome, not a completed run. Whether DeepSeek's PPP8 speed
test actually finishes this time depends on whether stage3's real
store-fill duration has also improved (untested here) or whether
`v4_nccl_timeout_s` needs raising for this specific model before a relaunch.
Per the standing guidance to check in before an unsupervised overnight
cross-node DeepSeek launch (lowest scientific value remaining in the whole
spec_verify campaign per project memory "phase2-models-borrowed-answers")
— reported back to the coordinator with this summary; awaiting the go/no-go
before attempting it.

## OPEN, HARDER THAN FIRST THOUGHT: DeepSeek-V4-Flash vLLM TP8 — no available nvcc both compiles DeepGEMM and runs on this driver (spec_verify Phase 2, 2026-07-20)

CORRECTION (2026-07-20, later same day): the entry originally here said
"export `CUDA_HOME=/usr/local/cuda-12.6` + PATH" was a sufficient fix. It is
NOT — that was written from the surface error message alone
(`"NVCC compilation failed"`) without checking whether 12.6 could actually
compile the specific failing construct. A follow-up investigation read the
real compiler output and found the true, harder root cause below; the naive
"just export PATH" fix would have failed again in exactly the same way.

Separate from the training-side PPP8 NCCL hang above, DeepSeek's *inference*
(vLLM TP8, cross-node via torchrun external_launcher, TP4xPP2=world_size 8)
crashed on agpuh02's local ranks during `profile_run` -> the NVIDIA-optimized
`deepseek_v4` model's `mhc_pre_tilelang` -> `tf32_hc_prenorm_gemm` ->
DeepGEMM JIT path. The actual compiler output (not just the assertion) is:

```
Warning: please use at least NVCC 12.9 for the best DeepGEMM performance
NVCC compilation failed: .../deep_gemm/ptx/ld_st.cuh(152): error: asm operand type size(16)
  does not match type/size implied by constraint 'q'
      asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
```

DeepGEMM's `_find_cuda_home()` resolves (via the system `/usr/local/cuda ->
/etc/alternatives/cuda -> cuda-12.6` symlink) to **nvcc 12.6**, which is too
old for this inline-PTX construct — reproduced standalone with a minimal
`.cu` file containing the identical `st.shared.b128`/`"q"`-constraint
pattern: fails identically on nvcc 12.6, compiles cleanly on nvcc 13.2 (the
venv's pip-installed `nvidia-cuda-nvcc-cu13` dependency). But a 13.2-linked
binary then fails at **runtime** on this node: `CUDA driver version is
insufficient for CUDA runtime version` — driver 565.57.01 only supports up
to the cu128 runtime (which is why torch itself is pinned to cu128, not
cu13x). So the actual constraint is a **compatibility window**: DeepGEMM
needs an nvcc >=12.9 to compile this op, AND that nvcc's runtime must stay
within what driver 565.57.01 supports (<=~12.8-class). No toolkit currently
installed on either node satisfies both: 12.6 is too old, 13.2 is too new
for the driver. The cluster's `cudatoolkit/12.9` Lmod module — the likely
sweet spot — fails to load for an unrelated reason (`ncurses/6.4-aocc-4.1.0-
whgpitb` module dependency not found), a separate infrastructure gap. A
bounded attempt to pip-install `nvidia-cuda-nvcc-cu12==12.8.*` standalone
(no venv modification) to test 12.8 was inconclusive: that wheel ships only
`ptxas`, not the `nvcc` frontend binary, so it can't compile the test case.

**Net: unresolved, third-party toolkit/driver incompatibility inside
DeepGEMM's JIT, not a defect in our snapshot, code, or multi-node plumbing.**
Positive finding despite the block: rank1's PP stage fully loaded its share
of the 543GB dequantized bf16 snapshot and reached a live forward pass
before hitting this error — the first time this snapshot has been shown to
instantiate under vLLM at any topology (it never fit TP4 single-node
before). To actually unblock this: either get `cudatoolkit/12.9` loading
(fix the missing `ncurses` Lmod dependency — a cluster/module-system issue,
likely needs sysadmin access or a different ncurses module substitution) or
find another nvcc build in the 12.9-12.8ish window compatible with driver
565.57.01.

Separately, the SAME cross-node vLLM TP8 window also had a 397B
(Qwen3.5-397B-A17B, pure TP8, no PP split) attempt die silently: rank0
(agpuh01) logs nothing past `vLLM is using nccl==2.28.9` (00:50:20), then
agpuh02's ranks 4-7 lose their TCPStore connection to it with `Broken
pipe`/`TCPStore server has shut down too early` at 00:55:24. `dmesg -T`
ruled out the OOM-killer for this window. **This is very likely a plain
arithmetic capacity issue, not a bug worth chasing further**: bf16
Qwen3.5-397B-A17B is ~752-794GB of weights; ANY partitioning of
`tensor_parallel_size x pipeline_parallel_size = 8` across 8x80GB=640GB
total leaves no headroom for activations/KV-cache, since neither TP nor PP
introduces weight redundancy — confirmed by a follow-up investigation that
deliberately did NOT retry with the (otherwise-fixed) PP2xTP4 topology for
this reason. Unblocking 397B's vLLM TP8 leg needs quantization (fp8/int4)
or CPU/disk weight offload, not a retry of the same bf16 attempt under a
different parallelism shape.
