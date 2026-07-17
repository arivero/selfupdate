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
- **Student-trajectory validation relay (M3, pending)**: the genuine
  censored student forward, staged across the v4 processes via CPU
  boundaries with filesystem markers, cadenced by `v4_relay_every_cohorts`.
  This is the deployment-matched CE/KL.
- **Per-epoch particular evaluations are non-negotiable** (owner,
  2026-07-17): recall corpora (machado, quijote_ch1, quijote_ch4, incl.
  epoch zero), the standard-damage battery, and parameter-delta profiles run
  every epoch — in single-process mode directly (same telemetry as v3); in
  staged mode via the merged adapter on stage 0 (M3).
- **Locality certification** (`certify_locality_v4`): measured, not assumed —
  sampled (item, layer) backwards must put zero gradient on every foreign
  block and on embed/norm/head. The dispatch refuses to publish a checkpoint
  when it fails.

## Checkpoints and merging

Each stage saves an ordinary PEFT checkpoint plus `v4_stage_manifest.json`
(its owned block range). `scripts/merge_v4_adapters.py` assembles the full
adapter by taking each block's tensors from the one stage that owns it —
no averaging; the merge is exact because ownership is disjoint.

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
