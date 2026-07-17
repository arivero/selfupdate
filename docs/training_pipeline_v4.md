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
