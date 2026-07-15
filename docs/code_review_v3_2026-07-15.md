# Code review: pipeline v3 online trainer (2026-07-15)

Max-effort review of the uncommitted pipeline-v3 work on branch `layerwise`,
centered on `src/selfupdate/train/online_v3.py`, with GPU-speed emphasis as
requested. Method: six finder agents over the full working-tree diff produced
29 candidate findings; the cloud verification stage was lost to a session
limit, so every candidate was re-verified by hand against the tree as of this
date. The tree changed substantially while the finders ran (`online_v3.py`
grew 440 → 1241 lines), so each candidate was judged against the CURRENT
code, not the snapshot the finders saw.

## Verified findings (surviving)

### 1. `SELFUPDATE_NODE_EPOCH0_CACHE_ROOT` bypasses the /dev/shm guard
`src/selfupdate/teacher/cache.py:106` resolves the node-epoch0 cache root
from the environment variable first, falling back to `cfg.cache.node_root`.
The validator (`src/selfupdate/train/validate.py:35`) checks only
`cfg.cache.node_root.startswith("/dev/shm/")`. An operator who exports the
environment variable to a Lustre path (mirroring the documented
`SELFUPDATE_DEV_PYTHON_HOST` habit) passes validation, and every v3 arm on
the host then memory-maps multi-GB safetensor shards from Lustre — the exact
near-zero-utilization page-fault stall the /dev/shm requirement exists to
prevent, reintroduced silently. Fix: validate the resolved root (after the
environment override and expansion), not the config field.

### 2. `stub`/`stub_gap` views now fail the `censored_rows` invariant
`src/selfupdate/masking.py:435` now always records an explicit privileged
span for single-block pairs. `censored_rows`
(`src/selfupdate/train/layerwise.py:342`) was given an `s0 == t0` branch that
protects the length-preserving views (`intact`, `pad_random`), but a
non-empty `student_stub` (validator-legal `stub`/`stub_gap` compaction,
`validate.py:237`) yields `s0 != t0` WITH a populated span list, entering the
interleaved branch: it emits `len(prefix) + A` rows against the
`len(rows) == s0 + A = len(prefix) + len(stub) + A` assertion. Any v1/v2
`teacher_censored`/`mixed` run (or MoE row-map path) on stub data now dies
with `AssertionError` on its first item; before this change the empty-span
branch handled it. Latent — no current campaign config uses stub — and loud
rather than silent, but it narrows the sanctioned control set. Fix: treat a
single span exactly covering `[len(prefix), t0)` as the classic block, or
subtract only the span rows that the stub does not re-occupy.

### 3. Intact arms still pay for a mask SDPA does not need (efficiency)
`_prepared_cached_masks` (`src/selfupdate/train/online_v3.py:46`) builds a
full `[1, 1, T, T]` additive causal mask per (device, dtype, layer type) even
when `full_keep is None` (the `intact`/`pad_random` controls), and every
block then runs scaled-dot-product attention with an explicit mask whose
content is exactly causal — numerically identical to passing no mask. Cost:
one T×T bf16 tensor held per answer (~134 MB at T=8192 on the 4B arms) and
the arbitrary-mask attention kernel instead of the mask-free causal fast
path in every (layer × token) cell. For q_len=1 decode rows the mask is
all-zero. Returning `None` for full-attention layers when `full_keep` is
None keeps flow arms unchanged and makes the intact control both faster and
a fairer throughput baseline. Unmeasured; worth a one-epoch A/B before the
0.6B probes are compared on token-events-per-second.

## Candidates already fixed in the current tree (verified)

The finder wave flagged these against its snapshot; all are resolved in the
code as of this review:

- Per-call `inspect.signature` in the block walk → probed once in
  `BlockStack.__init__` (`blocks.py:58`).
- Per-call `list(block.parameters())` → `_block_params` cached at
  construction (`blocks.py:57`).
- Additive mask rebuilt per (layer × token) → answer-wide
  `_prepared_cached_masks` with per-row views (`online_v3.py:46,83`).
- Rotary embeddings recomputed per layer in `_token_cached` → computed once
  per token before the layer loop.
- `torch.tensor(layer_indices)` per token in `_immediate_sgd_token` →
  `_layer_index_tensor` cache (`online_v3.py:166`).
- Cross-stream tensor lifetime in the pipeline lanes →
  `h_in.record_stream(stream)` (`online_v3.py:679`).
- Sliding/chunked attention approximated by a rolling window → hard
  `NotImplementedError` guards (`online_v3.py:921`, `:58`, `:453`).
- bf16 full-weight immediate SGD underflow → guarded; LoRA required in
  reduced precision (`online_v3.py:915`).
- Epoch-long `pending_losses` tensor retention → per-answer running sums via
  `_foreach_accumulate`.
- teacher_hidden per-cell pageable host-to-device copies →
  `full_inputs_resident` keeps `h[L-1]` on each block's device
  (`teacher_source.py:82`).
- Run-log provenance dependent on launch directory → `cwd=repo_root` pinned
  (`utils/runlog.py:49`).
- Cross-view node-cache identity mismatch → readers waive student-view span
  metadata on cross-view caches and reconstruct from the dataset masker
  (`data/dataset.py:203`).

## GPU-speed assessment

The structural cost of v3 (one write per token per block, batch size one) is
the experiment, not an inefficiency. Around it, the implementation now does
the right things: multi-tensor `torch._foreach` writes and gradient norms,
answer-local staging of ids/positions/targets, prepared masks, GPU-side
accumulation flushed at answer/epoch boundaries, and no `.item()`/`.cpu()`
inside the walk. Remaining one-GPU headroom is small and known: CLAUDE.md
(2026-07-15) records lane/wavefront rearrangements as dispatch-bound at
roughly 9–12 token-events/s on Qwen3-0.6B L40S, bounded lanes adding memory
without speed, and grad-ready hooks slowing the student path — per the
owner's note, the next lever is multi-GPU partitioning or fixed-shape
capture/fusion (CUDA graphs), not more one-GPU scheduling variants. Finding
3 above is the one residual per-kernel win visible in this diff. Two further
micro-items, noted but below the reporting bar: `_flow_keep` reallocation
once per token in `_token_cached` (an answer-wide keep mask sliced per token
would remove it), and the prefill path not receiving prepared masks (one
O(prefix²) mask build per layer per answer).

## Coverage

Reviewed: `online_v3.py` (all dispatches: per-block, per-token-disconnected,
wavefront, teacher lanes, pipeline lanes, stale windows, grad-ready),
`blocks.py`, `teacher_source.py`, `masking.py`, `data/dataset.py`,
`teacher/cache.py`, `train/validate.py`, `utils/runlog.py`,
`scripts/build_teacher_cache.py` (guard region), pareto_v3 configs (spot).
Not exercised: no run was launched; all verification is static reading plus
the repo's recorded measurements. The locality certification laws
(docs/training_pipeline_v3.md §certify) were checked as code paths
(`_detach_cache_layer`, frozen-vocab guard, disconnected-root backward), not
re-proven numerically — `scripts/train_certify.py` remains the on-demand
instrument for that.

## Resolution pass (2026-07-15)

All three surviving findings were addressed in the working tree:

1. `resolved_node_epoch0_root` now applies the environment override and path
   expansion first, resolves symlinks, and then requires the resulting path
   to remain beneath `/dev/shm`. Both dispatch validation and cache-directory
   resolution call this single guard.
2. `censored_rows` again recognizes the classic one-block/non-empty-stub map;
   intact and length-preserving random controls retain their identity map,
   while genuinely interleaved censorship still uses the kept-run map.
3. Prepared cached masks now carry an explicit mask-free sentinel for intact
   q=1 full-attention cells. K>1 cached chunks require causality within the
   chunk, so they lazily share a K×prefix additive mask; omitting it was caught
   by the intact GPU smoke as future-token leakage. This still avoids the
   unconditional answer-wide T² allocation and preserves the fair K=1 SDPA
   baseline. Flow masks are unchanged.

The speed calibration also extended beyond the review snapshot. On an L40S,
Qwen3-0.6B with a 256-token longest-answer smoke measured 10.69 token-events/s
at exact K=1, 165.54 at K=16, 604.78 at K=64, and 1435.17 for one answer-wide
window. All four runs passed the cache-graph and frozen-vocabulary guards;
incremental peak memory remained 83--95 MiB. K=8 is now a named experiment
arm so the scientific comparison can report both throughput and per-layer
weight-delta divergence from K=1 at a shared seed and item budget.

The post-fix intact K=64 smoke measured loss 2.03e-6--9.24e-6 and total
parameter delta 7.62e-7 at 605.09 token-events/s; cache history remained
graph-free and the vocabulary stayed frozen. The matched one-answer K=1/K=8
flow calibration measured 10.87/82.69 token-events/s (7.61x), with exact
trainable-delta relative L2 divergence 0.154 and cosine 0.9889. Layer 1 is the
staleness outlier (0.744 relative divergence); this motivates the full matched
12k-item quality sweep rather than selecting K from throughput alone.
