# Code review: pipeline-v3.1 trainer (`train_bk_v31`), speed & memory focus

Date: 2026-07-16. Reviewed at commit `e0615f2` (working tree). Scope:
`src/selfupdate/train/online_v3.py` — chiefly `train_bk_v31`
(`online_v3.py:1010`) and its helpers `_bk_prepare_cohort_shards` (`:865`),
`_bk_prepare_shard_tile` (`:960`), `_bk_prefix_layout` (`:814`),
`_bk_additive_mask` (`:833`), `_immediate_sgd` (`:187`) — plus the pieces the
hot loop calls into: `BlockStack.run_block`/`loss_view`
(`train/blocks.py:163,286`) and `collate_padded_items`
(`data/dataset.py:258`).

Grounding measurements (see "Measured throughput per arm" in
`docs/pareto_v31_training_progress.md`): 0.8B K=16 sustains ~2,700 tok/s over
six epochs; 0.8B K=1 sustains ~565 tok/s; 4B K=16 ~820 tok/s. Two distinct
OOM modes were observed on 2026-07-15/16: a content-driven tile-size failure
(all unsharded arms at the 780p×844a cohort) and a scheduler GPU collision
(4B arms on agpul04 GPU0).

Review discipline: none of the speed changes below may be applied without a
`scripts/train_certify.py` numerical regression against HEAD (fresh fingerprints, compare,
discard), and the measured NEGATIVE results in `issues.md` (async target
prefetch; threaded/CUDA-stream student lanes; grad-ready hooks on the student
path; multi-GPU lane partitioning) rule out several superficially attractive
follow-ups. Each finding says which prior negative result it is nearest to.

## S1 — HIGH (speed): bool-mask indexing syncs the GPU in the innermost loop

`online_v3.py:1132-1135`:

```python
view = stack.loss_view(layer, h_out)[state["query_valid"]]
target = state["window_targets"][layer - 1][state["query_valid"]]
```

Boolean advanced indexing on a CUDA tensor runs `nonzero()` and must read the
element count back to the host to size the output — a device-to-host
synchronization. This happens twice per (layer, shard) per tile: with 24
layers × 4 shards that is ~192 forced syncs per tile, and each sync also
stops the CPU from running ahead to enqueue the next layer's kernels. This is
exactly the failure mode of the repo's own "sync-bound" lesson (CLAUDE.md,
measured 1.46× recoverable in the item-loop era), and it directly contradicts
the invariant that `collate_padded_items` documents for this purpose
(`data/dataset.py:265-267`):

> Invariant: every mask marks a PREFIX of valid rows (`[:A]`). The trainer's
> per-example losses rely on this to slice by CPU-side lengths instead of
> bool-mask indexing (which would sync the GPU via nonzero()).

The v3.1 loop already has everything needed to avoid the sync without any
numerics change to the loss value:

- `state["cells"]` is computed on the CPU from `batch.A`
  (`online_v3.py:970`), so the masked-mean denominator is known host-side.
- Option A (masked mean): compute the loss elementwise, multiply by
  `query_valid` (as dtype weights), sum, divide by `cells`. Invalid rows are
  already zeroed by the mask; no data-dependent shapes anywhere. Requires the
  loss kind to expose an elementwise/weighted form — Huber and the geometric
  kinds do trivially; vocab kinds need the same masking inside their KL/MSE
  reduction.
- Option B (host-built gather index): because valid cells per row are a
  prefix with CPU-known lengths, a flat `[cells]` index tensor can be built
  on the CPU per tile (shape known in advance) and shipped once; `view` and
  `target` become `index_select`s with no sync.

Option A is smaller and keeps one code path. Certify: the reduction order
changes (sum-of-masked vs sum-of-selected), so expect bit-differences within
bf16/fp32 tolerance — this is the case `train_certify.py` exists for.

Expected effect is largest exactly where v3.1 is weakest: the K=1 and 4B
regimes, where tiles are small and per-tile dispatch dominates (measured
~565 tok/s at K=1 vs ~2,700 at K=16 — the tile machinery, not FLOPs, is the
gap).

## S2 — MEDIUM (speed): per-tile pageable, synchronous H2D of window targets

`online_v3.py:975-978`:

```python
window_targets = torch.stack([
    batch.hidden[layer][:, start:stop] for layer in range(1, n + 1)
]).to(device)
```

`batch.hidden` tensors are ordinary unpinned CPU zeros filled at collate
(`data/dataset.py:289,303`). Per tile this does (a) a CPU-side stack copy of
n×B×K×H, then (b) a synchronous pageable H2D transfer. For 4B shard tiles
that is ~36×32×16×2560×2 ≈ 94 MB per shard-tile through the slow pageable
path, and the copy blocks the stream.

Cheapest fix with no schedule change: keep one reusable **pinned** staging
buffer of the maximal shape (n, shard_users, K, H), `copy_` the slices into
it (still CPU work, but into pinned memory), then `.to(device,
non_blocking=True)`. The transfer then overlaps the previous layer-loop's
compute naturally because the consuming kernels are ordered on the same
stream.

Deliberately NOT recommended: a look-ahead prefetch of tile t+1 on a side
stream while tile t trains — that is shape-wise the async target prefetch
that `issues.md` measured as a NEGATIVE result in the item-loop era. The
pinned-buffer change is not a prefetch (no second stream, no lifetime
extension); it only upgrades the transfer path. If someone later proposes the
side-stream version, the burden is a fresh measurement against that negative
result, not an analogy.

An alternative worth one probe on 0.8B only: stage the whole shard's answer
targets to GPU once per cohort (n×B×Amax×H — ~2.7 GB for the worst 0.8B
cohort at shard64; clearly unaffordable for 4B/36-layer at large Amax). This
deletes the per-tile transfer entirely at the cost of answer-length-scaled
memory; the shard docstring (`online_v3.py:948-951`) explicitly warns against
letting the shard mechanism become "an answer-length activation cache", so
this is config-gated at best.

## S3 — LOW (speed): autocast context churn and small per-tile rebuilds

- `torch.autocast(...)` is entered per (layer, shard) (`online_v3.py:1124`);
  entering/exiting ~96× per tile is measurable only in the dispatch-bound
  K=1 regime. It can be hoisted to wrap the whole `for layer` loop — the
  only non-autocast work inside is `_immediate_sgd`, which is `@no_grad` and
  dtype-explicit, so hoisting is safe but should still be certified.
- `key_keep = torch.cat(...)` (`:982-983`) and `_bk_additive_mask` (`:984`)
  rebuild per tile from scratch. `key_keep`'s prefix half never changes; the
  additive mask could be built once per shard at max_answer and sliced per
  tile. Both are small (≤ ~13 MB at K=16); only worth doing if a profile
  after S1 still shows tile-prep time.
- `_bk_prefix_layout` runs a Python double loop over rows × privileged spans
  with one GPU slice-write per span (`:823-828`, flow_mask only). Once per
  shard per cohort, so amortized — but for span-heavy datasets this is many
  tiny kernels inside cohort setup. Vectorizable with a [B, S] mask built on
  CPU and shipped once. Low priority; measure first.

## S4 — INFORMATIONAL (speed): the layer-major shard loop is load-bearing

The nesting `for layer: for shard:` with one `_immediate_sgd` per layer
(`online_v3.py:1107-1146`) is not an optimization target: every shard must
see the same pre-write weights for the B×K update law to hold
(`_bk_prepare_cohort_shards` docstring, `:867-873`). Reordering to
shard-major (which would shrink resident tile state) changes when writes land
relative to later shards' forwards and is a *semantic* change, not a speed
patch. Anyone tempted should re-read the same lesson in
`docs/pareto_v31_training_progress.md` about unaveraged-sum-at-shared-
snapshot being part of config identity.

Likewise, the ~52 events/s CUDA-graph replay prototype (`issues.md`,
`docs/training_pipeline_v3.md:295`) remains the known big lever for the
dispatch-bound floor, and remains blocked on its reproducible 1.16%
trainable-delta divergence. Nothing in this review supersedes that gate.

## M1 — HIGH (memory): K=1 configs ran unsharded into a known-fatal tile

`activation_shard_users` defaults to B when unset (`online_v3.py:1040`).
Every K=1 arm ran at 256u/256shu×1sh and died at the same 780p×844a cohort,
after which the scheduler retried the identical failure every ~40 minutes for
~13 hours (see the throughput audit table). The K=16 arms survive the same
cohort at 64shu×4sh (0.8B) / 32shu×8sh (4B).

This is a config defect, not a code defect, but it earns a code-level guard:
`scripts/memory_plan.py` exists precisely to recommend shard/window sizes
before loading weights. Two concrete actions:

1. Pin `activation_shard_users` explicitly in every v3.1 experiment YAML
   (the "config DEFAULTS are experiment variables" law; the K=1 arms forked
   from the K=16 arms by exactly this unpinned knob).
2. Add a cheap dispatch-time check in `train_bk_v31`: estimate the worst
   cohort's tile footprint (prompt+answer lengths are known from the dataset
   before training starts, and `_bk_bucketed_cohorts` is deterministic given
   the seed) and refuse to start — not OOM five minutes in — when the
   unsharded estimate exceeds free VRAM. A refusal at dispatch also breaks
   the scheduler's blind retry loop, which burned ~13 GPU-hours overnight.

## M2 — MEDIUM (memory): finished rows' cache state is never reclaimed

A cohort's `DynamicCache` history holds K/V (full-attention layers) and
recurrent/conv state (linear-attention layers) for all B rows until the
longest answer finishes (`train_bk_v31` docstring, `:1013-1016`). Rows are
batch-independent: a finished row's cached state serves only its own dead
queries (`_bk_additive_mask` gives them one harmless key, `:841-844`), so
evicting or compacting finished rows is numerically invisible to every live
row. Length-bucketed cohorts (`_bk_bucketed_cohorts`, `:852`) keep the
within-cohort length spread small, which already bounds the waste — but the
fatal cohorts are precisely the long-tail buckets where the spread and the
absolute lengths are both largest.

Concrete shape: when ≥ half a shard's rows are finished, `index_select` the
live rows into a fresh cache and shrink the shard's tensors. This is real
implementation work on `DynamicCache` internals (dense [B, heads, S, D]
layout) and only pays in the tail cohorts; rank it below M1 (config) and S1
(sync), and certify with the PP-style semantic-hash argument that placement/
compaction knobs must not change numerics.

## M3 — LOW (memory): prefill mask transient at full shard width

`_bk_prepare_cohort_shards` materializes `prefill_allowed` (bool) and
`prefill_mask` (bf16) at (b_now, 1, S, S) (`online_v3.py:897-908`): at
256 users × 780 prompt this is ~311 MB bf16 plus ~156 MB bool, transient but
coincident with the cache-growth peak. Chunking the prefill along the query
axis (mask rows for q-chunk only) removes the S² term. Only matters for
unsharded/large-shard configs; M1's sharding makes it mostly moot.

## M4 — DIAGNOSTIC (memory): verify the allocator config actually lands

The fatal OOM traces show 10.98 GiB and 4.27 GiB "reserved by PyTorch but
unallocated" — an unusually large fragmentation share if
`expandable_segments:True` were active. `scripts/l40s_exec.sh:42` exports
`PYTORCH_ALLOC_CONF=expandable_segments:True` (the new-style variable), and
the L40S lane runs torch 2.7.1. Worth one 2-minute check inside the actual
runtime (`torch._C._cuda_getAllocatorBackend()` or
`torch.cuda.memory_stats()['num_alloc_retries']` behavior) that the
new-style name is honored by this torch build and survives the
`ld-linux --library-path` launch into the jacobian-lens venv; if not, the
old `PYTORCH_CUDA_ALLOC_CONF` name should be exported alongside. If
fragmentation is real, M1's OOMs had less headroom than the arithmetic
suggests, and fixing the export is free margin.

## What is already right (do not "fix")

- No `.item()`, `.cpu()`, or print in the tile loop: losses and grad norms
  accumulate as GPU scalars via `_foreach_accumulate` (`:325`) and hit the
  CPU once per epoch (`:1215`). This is the sync-bound lesson applied
  correctly — S1 is the one leak.
- `_immediate_sgd` uses grouped `torch._foreach_*` kernels (`:210-215`): one
  logical write per block without per-parameter launches, and the grad-norm
  telemetry it computes is required by the gradient-share attribution law,
  not overhead to strip.
- The explicit `del` housekeeping at tile and cohort boundaries
  (`:1161-1165`, `:1182-1187`) keeps the allocator's working set tied to the
  current tile; verbose but correct.
- Prefill runs under `no_grad` with per-layer cache detach
  (`_detach_cache_layer`, `:926`), so no graph leaks into the persistent
  history — consistent with the graph-leak tripwire's expectations.
- `cells`/`max_answer` come from CPU-side `batch.A` (`:970,931`): tile
  bookkeeping is sync-free.

## Recommended order of work

1. **M1.2 dispatch guard + pin `activation_shard_users` in K=1 YAMLs** —
   unblocks the wedged arms and stops retry-loop burn; no numerics risk.
2. **S1 masked-mean loss (kill the bool-index syncs)** — certify with
   `train_certify.py --all`, then compare throughput at K=1 and 4B-K=16 where
   dispatch dominates.
3. **S2 pinned staging buffer for window targets** — certify + throughput
   old-versus-new comparison; do NOT extend to side-stream prefetch without confronting the
   existing negative result.
4. **M4 allocator verification** — 2 minutes, possibly free margin.
5. **S3/M2/M3** — only with a profile in hand after (2)-(3), on the tail
   cohorts specifically.
