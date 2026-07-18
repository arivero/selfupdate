# Training pipeline v4 — blockwise teacher-forced, frozen teacher KV

Owner-specified 2026-07-17. Implementation: `src/selfupdate/train/online_v4.py`;
knobs in `config.py` (`train.v4_*`), rules in `validate.py` (pipeline_version 4
branch). Smoke configs: `configs/experiments/h100_smoke/base_qwen3_0p6b_v4_lora.yaml`
plus the `qwen3_0p6b_v4_1proc` / `qwen3_0p6b_v4_4stage` overlays.

## Objective

For every block `L`, at every loss position `p` (teacher coordinates):

```
loss_L = HiddenLoss( block_L^student( i_L[p] ),  h_L[p] )
i_L = teacher h[L-1]   (cached, full sequence:  cache.store_full_teacher_inputs)
h_L = teacher h[L]     (cached, aligned span)
```

- **No training loss touches the student's own trajectory.** Student hiddens
  are still computed — but only for the two validation losses (CE-eval /
  KL-eval via the relay, M3) and the generation probes.
- **Attention context = the teacher's frozen K/V**: adapters-off k/v
  projections (with RoPE at teacher positions) of the cached `i{L}`, recorded
  once by a prefill pass through the block's own attention module
  (`_FrozenKV`, record mode). During training the incoming query-row
  projections are DISCARDED and the stored tensors returned (frozen mode), so
  **gradients enter only through the query-side path** of block L: q_proj,
  o_proj, MLP, norms — never k/v of any position.
- **Censorship is attention censorship.** The additive mask removes every
  privileged key: the RAG passage AND the prompt text announcing it (both are
  inside `t_privileged`, see `masking.py`). Fill content is irrelevant
  because the fill is never attended. `mask.compaction: flow_mask` is the
  method; `intact` is the diagnostic control.
- **Loss positions** (`v4_loss_positions`): `answer` (teacher-realized answer
  tokens, default) or `aligned` (the whole cached span). `thinking_answer`
  is reserved until per-record thinking spans exist.

## Structural consequence: no sequential dependency at all

With teacher-fixed inputs AND teacher-fixed attention context, nothing an
earlier answer token computes is needed by a later one, and no layer needs
another layer. Consequences, all implemented:

1. **Whole-cohort processing is exact.** Each layer runs all loss positions
   of a cohort in one batched pass with ONE optimizer write per block per
   cohort (unaveraged cell sum). This is not a staleness approximation —
   v3's B×K tiling and `stale_gradient_window` do not apply.
2. **Layer-sharding for any GPU count.** `train.v4_stage_splits` (N cuts →
   N+1 stages) + `v4_stage_devices` (physical ids) + `train.py --v4-stage k`:
   independent OS processes, each loading the full model on ONE card and
   training only its owned blocks. No torch.distributed, no activation
   boundaries, no wavefront. Launcher: `scripts/launch_v4_stages.sh` (stage
   count read from the config — nothing assumes four GPUs).
   Sharding is numerics-neutral: per-layer results must be bit-identical to
   single-process at equal seed, and the smoke pair exists to verify that.
3. **Loop order is free** (`v4_loop_order`): `layer_major` (default) keeps
   one layer's teacher tensors and optimizer state hot across all cohorts;
   `item_major` walks owned layers per cohort.
4a. **Teacher source** (`v4_teacher_source`, owner contract "where IF ANY
   cached … or just keep calculating it"): `cache` reads the
   store_full_teacher_inputs cache; `online` computes teacher states with
   ONE adapters-off forward per cohort (vLLM contributes only answer token
   ids via an index-only cache; hidden states always come from OUR stack).
   Online capture feeds the SAME per-(layer,cohort) store, so residency
   decides cache-after-first-production vs recompute-per-epoch
   (`rebuild`). The placement triangle — GPU memory (`gpu_corpus`) vs
   pinned process CPU (`cpu_stream`) vs /dev/shm mmap (cache mode from the
   node cache) vs recompute — is a MEASURED choice: `capture_seconds` and
   the prep split in every v4_epoch row are the calibration data; do not
   hardcode a winner.
4. **Teacher tensors are constant** (`v4_kv_source: teacher_frozen`), so
   per-(layer, cohort) K/V, inputs, and targets are built once and reused
   every epoch. Residency (`v4_teacher_residency`): `gpu_corpus` keeps them
   on the card (~4 GB/layer/corpus at 0.6B/2071 items), `cpu_stream` stages
   from pinned host memory, `auto` sizes and picks.

## Structured teacher store (owner decomposition, 2026-07-17 — the contract)

Position classes store different things (v4_loss_positions=answer):

1. **Common system prefix** — identical text at identical positions,
   causally self-contained → teacher KV identical across the items that
   share it. The v5 set is TRI-PARTITE (machado, quijote ch.1, quijote
   ch.4) with per-corpus system prompts, so dedup keys on the prefix
   TOKEN-ID hash — one stored KV per (layer, prefix class), ~3 classes
   here, generalizing to any corpus mix.
2. **Per-item prompt remainder** (privileged + mid + question) — attention
   context only, never a loss position → KV only (~4 KB/pos at 27B), no
   hidden vectors. Censorship stays in the mask, orthogonal to storage.
3. **Answer span** — full hidden vectors: block inputs AND targets
   (~10.2 KB/pos × ~120 rows/item at 27B).
4. **Linear-attention layers** — the analogue of "KV only" is the
   RECURRENT STATE AT ANSWER START: one fixed-size state per (layer, item),
   computed adapters-off WITH flow censorship (privileged rows never enter
   state), stored instead of full-sequence inputs. The answer-span pass
   runs the trainable mixer from that frozen state — symmetric with
   frozen-KV attention: frozen censored context, gradients only in the
   answer span.

Measured trade at 27B/stage (1.2 M positions): naive full-input store
166 GB vs structured ~70 GB (answers 41 + item KV 17 + linear states 12 +
shared prefix ~MB); streaming ~2 s/epoch vs ~80 s recompute — the store
wins ~40x once structured. Capture-per-epoch (`rebuild`) remains the
zero-memory fallback and the epoch-1 producer. IMPLEMENTATION PENDING:
tonight's einf runs `rebuild`; the structured store is the next
implementation item, with `capture_seconds`/prep splits as the calibration
evidence.

## Optimizer

- Default `v4_optimizer: immediate_sgd` — the v3 state-free fused write,
  one per block per cohort.
- `v4_optimizer: adam` (owner's `--use-adam`): one AdamW per owned block.
  Under `layer_major` the active block's moments stay resident; this is the
  more-memory option the owner named. (`offload_adam` remains the v1/v2
  knob and is rejected for v4.)

## Phase 2: `v4_kv_source: student_refresh`

Every `v4_kv_refresh_epochs` epochs the per-(layer, cohort) tensors are
invalidated and the K/V prefill reruns with adapters ENABLED (still no
gradient through K/V; block inputs stay teacher states). A full
student-trajectory KV — where the *inputs* to the projections also come from
the student's own run — would additionally need a relay-style sequential
pass and is the documented further step, not this knob.

## Evaluation

- **Teacher-forced CE/KL** (`kind: teacher_output_eval`,
  `trajectory: teacher_forced_blockwise`): computed streamingly at layer n
  during training over the answer-predictor rows (`answer_offset - 1`
  convention — every teacher-realized answer token exactly once per epoch),
  with the full evaluation-only flag set (`used_for_backward=false`,
  `optimizer_weight=0.0`, whole-training-set coverage). Note this is the
  final block run on TEACHER h[n-1] — it measures blockwise fidelity, not
  deployment behavior.
- **Student-trajectory validation relay** (`kind: student_trajectory_eval`):
  the genuine censored student forward — flow attention mask, full causal
  walk on the student's own states. Single-process: one call per epoch.
  Staged: `_staged_relay_epoch` — stage 0 embeds and runs its blocks, each
  later stage waits for its predecessor's boundary file (`_RelayFiles`,
  atomic tmp+rename under `runs/<run>/relay/`), the last stage logs the
  CE/KL. Boundaries flow card→CPU→card. `v4_relay_every_cohorts > 0`
  enables it; the current implementation fires at EPOCH boundaries (under
  layer_major, sub-epoch sync levels do not exist; an item_major sub-epoch
  cadence needs a sequence protocol and is future work). This is the
  deployment-matched CE/KL.
- **Per-epoch particular evaluations are non-negotiable** (owner,
  2026-07-17): recall corpora (machado, quijote_ch1, quijote_ch4, incl.
  epoch zero), the standard-damage battery, and parameter-delta profiles run
  every epoch — in single-process mode directly (same telemetry as v3); in
  staged mode via `_staged_epoch_battery`: every stage publishes its owned
  adapter tensors per epoch, stage 0 grafts the foreign blocks onto its
  resident model (harmless — v4 training never reads foreign blocks) and
  runs the same probes. Stage 0 also runs epoch zero directly (zero-init
  LoRA = base model).
- **Locality certification** (`certify_locality_v4`): measured, not assumed —
  sampled (item, layer) backwards must put zero gradient on every foreign
  block and on embed/norm/head. The dispatch refuses to publish a checkpoint
  when it fails.

## Checkpoints and merging

Each stage saves an ordinary PEFT checkpoint plus `v4_stage_manifest.json`
(its owned block range). `scripts/merge_v4_adapters.py` assembles the full
adapter by taking each block's tensors from the one stage that owns it —
no averaging; the merge is exact because ownership is disjoint.

## Beyond the OOM wall (contribution statement, owner 2026-07-18)

Part of the claimed contribution is negative-space: **where traditional
fine-tuning OOMs, this system still trains — and where even the weights
don't fit, it rotates.** The memory arithmetic, per 80 GB H100:

- Traditional mixed-precision AdamW holds ~16 bytes/param (bf16 weights +
  bf16 grads + fp32 moments + fp32 master) plus full-graph activations.
  gemma-4-31B: ~500 GB of optimizer-side state — un-runnable even ZeRO-3
  sharded across one 4-card node once activations join. Qwen3.5-397B:
  ~6.4 TB — un-runnable on any single node, full stop.
- Pipeline-v4 per stage holds: the owned shard's FROZEN bf16 weights
  (no grads, no moments — teacher-forced blockwise training never
  backpropagates through frozen weights), LoRA adapters + their optimizer
  state (MBs), and ONE block's activations at micro-batch scale. 31B
  trains on 4 cards with ~15 GB of weights per stage.
- When even the owned frozen shard exceeds the card (397B: ~200 GB/stage),
  `v4_weight_residency: rotate` pages block weights one-way from mmap
  masters and pages the **Adam moments both ways with their block** — the
  card only ever holds one block plus its transient. The M1 leg-D
  certification (leg C vs leg D bit-identity, moments included) is the
  proof that rotation is pure transport: the numbers a resident run would
  have produced, on hardware where the resident run cannot exist.

Report this arithmetic next to every big-model result: the baseline that
OOMs is part of the claim, not a footnote.

## ENVELOPE SPEED TABLE (the "speed proven" deliverable — owner, 2026-07-18)

For EVERY envelope member, TWO measured points (owner directive): the
**minimal config** (fewest GPUs it runs on — rotary PPP1 on one card is
the floor) and the **best config** (fastest, most GPUs — PPP4 resident/
store). "Speed proven" = no open cells. Steady = capture-once epochs
(teacher forwards = 0); capture is a one-time cost. Corpus = 2,071 items.

| model | minimal config | min steady s/epoch | best config | best steady s/epoch |
|---|---|---|---|---|
| Qwen3.5-0.8B | PPP1 1-GPU | — | PPP4 (certified 2026-07-17) | see cert table |
| Qwen3.5-4B | PPP1 1-GPU | — | PPP4 (cert vehicle) | see cert table |
| Qwen3.6-27B | PPP1 rotary 1-GPU (pending) | — | PPP4 resident 4-GPU | **198 s** @ 91% |
| Qwen3.6-35B-A3B | PPP1 rotary 1-GPU (pending) | — | **PPP4 store 4-GPU** | **15.9 s** @ 97% (capture 117 s once) |
| gemma-4-26B-A4B | **PPP1 rotary 1-GPU** | **61.8 s** @ 72%, stall 0.128 s | PPP4 4-GPU cpu_stream | **12–14 s** @ 82–86% |
| gemma-4-31B | PPP1 rotary 1-GPU (running) | — | PPP4 4-GPU rebuild (mb8) | **328 s** @ ~25% (capture-bound) |
| DeepSeek-V4-Flash | (bf16 dequant streaming #16→#11) | — | — | — |
| Qwen3.5-122B-A10B | **PPP1 rotary 1-GPU** (244 GB model, un-runnable resident!) | **202 s** @ 88%, stall 0.287 s | **PPP8 store 8-GPU / 2 nodes** | **20.3 s** @ 98% (cross-node relay NOT the bottleneck) |
| — 122B scaling (same store lane) | — | 1-GPU 202 s → **4-GPU 40.0 s** @ 98% → 8-GPU 20.3 s | — | ~10× 1→8, near-linear |
| Qwen3.5-397B-FP8 | PPP1 rotate 1-GPU (M5) | — | PPP4-rotate stack (M5) | — |

The minimal column is the "beyond the OOM wall" proof: 122B (244 GB, can
NOT fit resident on 80 GB) trains on ONE card at 202 s/88% util, rotation
transport 99.86% hidden. The best column is throughput; the ratio between
them is rotation's cost (26B: 61.8 s min vs ~13.9 s best on 4 cards =
~90% of ideal 4→1 linear scaling — rotation is near-free). 31B PPP1 min,
35B-A3B both, and PPP5 cross-node numbers land next.

## The Pareto envelope (owner-ordered, 2026-07-17)

Speed-first certification order; every member trains under the utilization
gate (>50% floor, 90% goal) at PPP stage counts up to 4:

1. Qwen3.5-0.8B (24L, 18 linear + 6 full) — hybrid routing supported.
2. Qwen3.5-4B (32L, 24+8) — supported; certification vehicle.
3. Qwen3.6-27B (64L, 48+16) — supported; smoke cache ready.
4. Qwen3.6-35B-A3B (40L, 30+10, MoE) — MoE under dense_or_black_box.
5. google/gemma-4-26b-a4b (30L, 25 sliding + 5 full, MoE) — sliding mask
   implemented; composite loading + Gemma shared-KV TO DO.
6. google/gemma-4-31b (60L, 50 sliding + 10 full, MoE) — same, bigger.
7. deepseek-ai/DeepSeek-V4-Flash (43L, H4096, MoE, MLA) — MLA latent-KV
   attention needs its own _FrozenKV adaptation (BlockStack already carries
   the rotary-internal MLA path; the cache stores latents, not k/v heads).
8. Qwen/Qwen3.5-122B-A10B (48L, 36+12, MoE) — architecture already
   supported (same family as the A3B), but ~244 GB bf16 exceeds one card:
   requires STAGE-SCOPED LOADING (materialize only the owned blocks plus
   embed, and norm/head on the last stage) — the natural completion of
   disjoint ownership; also relaxes gemma-4-31b's ~62 GB fit.

## CERTIFIED utilization (agpuh01, 2026-07-17, floor 50 armed, NVML in-trainer)

Owner goal met: the envelope trains above 50% training-phase GPU utilization
at 2-, 3-, and 4-GPU PPP launches — at ~90%+ steady state:

| launch | model | per-stage train-phase util, epochs 2+ (min–max) |
|---|---|---|
| PPP1 | Qwen3.5-4B | 99.4–99.97% |
| PPP2 | Qwen3.5-4B | s0 95.3–99.9, s1 95.3–99.8 |
| PPP3 | Qwen3.5-4B | s0 93.3–100, s1 88.5–97.2, s2 97.2–100 |
| PPP4 | Qwen3.6-27B | s0 91.1–98.1, s1 90.6–97.8, s2 88.9–96.4, s3 82.8–97.6 |

Cold build epochs ran 52–84% (still above floor); steady prep_fraction
0.003–0.04. Locality certification passed wherever the run reached it
(4B PPP3 all stages with published checkpoints; 27B stage 0; 27B siblings
were reaped mid-drain by a reaper false positive, fixed in
scripts/v4_stage_reaper.sh — their training telemetry is complete).
Relay CE agreed across PPP1/3/4 to 4 decimals at 0.6B and was stable
(3.0402±0.0005) at 4B.

### 27B full-true-epoch timing (v4_teacher_source=online, PPP4, 2026-07-17)

Measured on the first full-corpus einf launch (2,071 tri-partite items,
1,132,912 token events/epoch, whole-training-set eval coverage 70,807/70,807
answer tokens; stage-1 `v4_epoch` rows from
`runs/failed_launches/h100_27b_v4_ppp4_einf_20260717_1951/`; the run trained
6 honest epochs before a relay-plumbing bug — fixed in `e6ceb0f` — killed it):

| epoch | seconds | tok/s | prep s | capture s | exec s | GPU util |
|---|---|---|---|---|---|---|
| 1 (cold) | 350.2 | 3,235 | 15.1 | 176.1 | 158.9 | 60.8% |
| 2 | 198.8 | 5,699 | 14.9 | 136.8 | 47.0 | 90.7% |
| 3 | 198.5 | 5,708 | 14.8 | 136.6 | 46.9 | 91.5% |
| 4–5 | ~198.3 | ~5,713 | 14.8 | 136.4 | 46.9 | ~91.4% |
| 6 (partial) | 144.8 | 5,895 | 11.0 | 99.4 | 34.3 | 89.6% |

The load-bearing observation: **capture dominates the steady state** —
~136 s of the ~198 s epoch (69%) is the per-epoch adapters-off teacher
re-capture, vs ~47 s of actual training exec. Teacher hiddens are
epoch-invariant, so the capture-once structured store (`v4_teacher_source:
store`, the contract above) removes that 136 s for every epoch ≥ 1: a
projected ~62 s/epoch, ~3.2× throughput, on measured evidence rather than
estimate.

## Speed & utilization — how the GPU got busy, and what regresses it to 3%

Owner criterion (2026-07-17): a run whose TRAINING-PHASE GPU utilization is
below 50% is a FAIL and must abort; the goal is 90%. The trainer enforces
this itself: `train.v4_min_train_gpu_util` (NVML-sampled at cohort
boundaries during the walk, mean logged as `train_phase_gpu_util` in every
`v4_epoch` row, RuntimeError below the floor from epoch 2 on). The
per-epoch generation evals are excluded from the gate on purpose — they are
owner-mandated and inherently low-util; their fix is batching the eval
generation, not skipping them.

Measured levers, 0.6B smoke on one H100 (each was worth what it says —
do not undo one "for cleanliness" without re-measuring):

1. **Whole-cohort batched passes.** One `[B, Q, H]` (or `[B, T, H]` for
   linear layers) pass per (layer, cohort) instead of v3's per-token tiles.
   v3 measured 9–12 token-events/s dispatch-bound; v4 epoch 1 measured
   **34,413 ev/s** — the Python dispatch simply left the hot path. This is
   only legal because inputs AND attention context are teacher-fixed
   (whole-answer processing is exact, not stale).
2. **Per-(layer, cohort) tensor caching across epochs.** Teacher-frozen
   K/V, inputs, and targets never change, so epoch 1 builds them and every
   later epoch reuses: leg A measured epoch 1 at 7.3 s and epoch 2 at
   **1.1 s (390,363 ev/s)** — 6.6×. Over 40 epochs the build cost
   amortizes to ~2%. `v4_kv_source: student_refresh` deliberately re-pays
   it per refresh (leg B: epoch 2 at 95k ev/s — the designed price).
3. **`gpu_corpus` residency** when the owned layers' corpus fits (auto):
   zero per-step host transfers. `cpu_stream` is the big-model fallback,
   pinned + async.
4. **One fused optimizer write per block per cohort** (foreach kernels) —
   never per parameter tensor, never per token.
5. **One host sync per epoch.** All loss/grad/util accumulators stay on
   GPU; the only `.item()` calls happen in the epoch-boundary telemetry
   flush. This is the v3 sync-bound lesson applied structurally.

The regression list — any of these quietly returns the run to v3-like idle:

- `.item()`, `.cpu()`, `print`, or per-cohort logging inside
  `layer_cohort_step` (per-layer × per-cohort syncs re-serialize the GPU);
- shrinking `micro_batch` (small matmuls can't feed an H100 — B=256 is the
  campaign default, 64 was smoke-only);
- `cpu_stream` residency where `gpu_corpus` fits (auto exists — trust it or
  measure);
- rebuilding teacher tensors every epoch without `student_refresh` needing
  it (the cache key is (layer, cohort) — keep cohort composition fixed
  across epochs, which is also why within-cohort shuffle was dropped: order
  is irrelevant to a summed per-cohort write);
- reading WHOLE-RUN utilization as training utilization: on smoke-sized
  corpora the mandated per-epoch generation evals dominate wall-clock (the
  observed "3% with bursts"); judge the training phase by
  `train_phase_gpu_util` and fix eval throughput separately.

Known remaining wall-clock sink: recall/standard-damage generation runs at
tiny batch (`eval.generation_batch: 8`, and `tasks_eval` is B=1 per the
memory note). Batching/vLLM-ing the eval path is part of the speed program,
NOT reducing eval coverage — the per-epoch battery is non-negotiable.

## Future scale-out (owner, 2026-07-17 — noted, deliberately deferred)

The disjoint-block-ownership abstraction is the load-bearing idea; two
extensions follow from it and are explicitly left for later:

1. **Layer rotation.** The same ownership contract admits both limits: very
   small machines, and very large models where even ONE layer needs all the
   GPUs (tensor-sharded) — the process set then owns layers *in time* rather
   than in space, rotating each layer's weights (and optimizer state) in and
   out of GPU while the teacher tensors stream. Deferred because tuning it
   honestly requires measured PCIe/NVLink bandwidth to choose the batch size
   that hides the rotation; guessing would reinvent the "optimizing without
   measuring" failure mode issues.md documents. (`OptimizerPlan.full_offload`
   already implements the moment-paging half of this.)
2. **Layer sets on other machines (InfiniBand).** Much easier than it
   sounds: training needs NO cross-stage communication at all, so joining
   another machine is only an exception to the *relay's* shared-memory
   boundary — when the next owned layer set lives on another host, the
   student-trajectory boundary tensor travels over InfiniBand instead of
   /dev/shm. Everything else (per-host node caches, per-stage checkpoints,
   the merge) already works per host.

## Bibliography

- **Primary blockwise-distillation ancestor (owner-confirmed):** Hui Wang,
  Hanbin Zhao, Xi Li, Xu Tan — *Progressive Blockwise Knowledge Distillation
  for Neural Network Acceleration*, IJCAI 2018. Teacher decomposed into
  blocks; student trained progressively block-by-block under teacher
  supervision. (Xu Tan later joined Moonshot/Kimi.)
- Block-wise teacher-forcing in the attention-conversion literature: each
  converted block trained independently on the teacher's hidden state as
  input ([Attention Editing, arXiv:2604.05688](https://arxiv.org/pdf/2604.05688));
  applied in practice for
  [Kimi Delta Attention distillation into AFM-4.5B (Arcee, 2026)](https://www.arcee.ai/blog/distilling-kimi-delta-attention-into-afm-4-5b-and-the-tool-we-used-to-do-it).
- Layer-wise KD with per-layer hidden matching:
  [LaDiMo, arXiv:2408.04278](https://arxiv.org/pdf/2408.04278);
  [Module-wise Adaptive Distillation, arXiv:2310.04550](https://arxiv.org/abs/2310.04550);
  [counterclockwise block-wise KD, Sci. Reports 2025](https://www.nature.com/articles/s41598-025-91152-3).

**Differentiators of v4** relative to all of the above: (1) teacher and
student are the SAME model — what is distilled is *context* (the censored
RAG), not capacity; (2) **attention censorship** of the privileged span,
absent from every found work; (3) frozen teacher KV as the attention context
with query-side-only gradients.

## Known scientific caveat (state, don't hide)

`docs/casebook.md` (teacher-stream evidence): storing well under
teacher-forced inputs does not imply the student can run on its own states —
"readout requires student-stream self-drive." v4 accepts this by design;
the relay CE/KL is precisely the deployment-matched metric that will show
whether pure teacher-forcing at every layer recites. This is the
experiment, not a bug.

## DeepSeek-V4-Flash frozen-context adapter (plan B8 — design, cert-gated)

The MLA + sparse-indexer stack does not satisfy the `_FrozenKV` duck-type:
`DeepseekV4Attention` calls `past_key_values.update(kv, kv, layer_idx)` but
its compressor then needs `past_key_values.layers[layer_idx]` typed as
`DeepseekV4HCACache`/`DeepseekV4CSACache` (`update_compressor_states` /
`update_overlap_state`), and rope arrives as a `{main, compress}` bundle the
model's own `rotary_emb` builds per layer type — `blocks.rope` returns
neither. Design (not yet implemented):

1. **Rope bundle branch** in `BlockStack.rope`: when the config is
   `deepseek_v4`, call the model's rotary with both layer types and pass the
   dict through, mirroring the gemma4 bundle path.
2. **`_FrozenDeepseekCtx`**: a per-layer typed record/frozen wrapper around
   the REAL `DeepseekV4HCACache`/`CSACache` objects — record during the
   teacher prefill (adapters off), then serve read-only during query-side
   passes, exactly the `_FrozenKV` record/consume contract but holding the
   compressed latents + indexer/overlap states instead of k/v heads. The
   linear-attention recurrent-state precedent (store the state at answer
   start) is the template for the compressor state.
3. **Quantized base**: `scripts/dequantize_snapshot.py` produces the bf16
   snapshot (fp8 e4m3 + fp4 experts -> plain tensors; HF fp8 quantizer is
   `is_trainable=False`, so LoRA-on-fp8 is out of contract). ~316 GB bf16 →
   the stage-scoped + rotation lane, identical to Qwen.
4. **LoRA targets**: MLA projections are `q_a_proj/q_b_proj/
   kv_a_proj_with_mqa/kv_b_proj/o_proj` — extend `lora.TARGET_MODULES`
   per-family before attach (today's list would adapt only `o_proj` + MLP).
5. **Gate**: `certify_locality_v4` must show exact-zero cross-block and
   frozen-vocab gradients through the adapter before any training run.

## Base-weight fine-tuning (plan B9 — design note, NOT coded)

`v4_update_target: adapter | base_blocks` (future knob). What survives
unchanged: stage-scoped loading, capture-once store, blockwise locality
(detached inputs), the relay, the battery, and the frozen-vocabulary locks —
"base weights" always means the transformer blocks, never embed/norm/head.

What changes:
- **Teacher-target semantics become visible.** Under LoRA the teacher
  (adapters-off) never moves; under base-FT the teacher and the trained
  tensors are the same object. Default (owner, 2026-07-17): **frozen
  epoch-0 targets** — the store stays exact and capture-once; the drifting
  -teacher variant (re-relay every N epochs) is a separate later experiment.
- **Rotation becomes two-way**: a trained block is dirty after its step —
  each stage needs a WRITABLE private host master (no mmap sharing of the
  snapshot; ~200 GB/stage at 397B) and pages the block back after its
  layer_major visit. Bandwidth doubles; still minutes-per-epoch scale.
- **Optimizer memory is the wall**: fp32 Adam moments at 403B are ~3.2 TB —
  infeasible on one node. `immediate_sgd` is state-free and feasible
  everywhere; full Adam works to ~122B; 8-bit moments are the middle. The
  moment-ROTATION machinery already ships in `rotation.py` (LoRA moments
  ride their block today), so base-FT is a writeback extension of proven
  code, not new invention.
- **Checkpoints balloon** to block-sized (an owned-shard save at 397B is
  ~200 GB/stage): last-k / every-N retention policy required before any
  einf-style run.
