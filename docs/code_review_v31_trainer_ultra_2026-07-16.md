# Ultra code review: pipeline-v3.1 trainer, speed & memory (verified)

Date: 2026-07-16. Multi-agent workflow review at HIGH effort: independent
finder agents per angle over the v3.1 hot path, then one adversarial
verifier per distinct (file, line) finding. **All 10 retained findings are
CONFIRMED** with code-level evidence; lower-severity cleanups (mask
duplication, redundant clones, autocast churn) were folded out under the
findings cap. Reviewed at HEAD `01a7823`, branch `layerwise`. No source
files were modified.

Scope: `train_bk_v31` (`src/selfupdate/train/online_v3.py:1010`) and helpers;
`BlockStack.run_block`/`loss_view` (`train/blocks.py`); `collate_padded_items`
(`data/dataset.py`); `HiddenLoss` (`train/losses.py`). Companion to the
inline review `docs/code_review_v31_trainer_2026-07-16.md` — findings 1, 4,
6, 7, 10 verify and refine that review's M1/M2/S1/S2/M3; findings 2, 3, 5,
8, 9 are new.

Standing constraints honored: the layer-major-over-shards loop and
one-write-per-block-per-tile are load-bearing semantics, not speed targets;
issues.md negative results (async target prefetch, threaded student lanes,
grad-ready hooks, multi-GPU lanes) were checked before proposing anything
similarly shaped; numerics-adjacent proposals are marked as requiring a
`scripts/train_certify.py` numerical regression against HEAD.

Measured grounding (see `docs/pareto_v31_training_progress.md`, "Measured
throughput per arm"): 0.8B K=16 ~2,700 tok/s; 0.8B K=1 ~565 tok/s; 4B K=16
~820 tok/s on L40S; all unsharded arms OOM'd at the 780p×844a cohort.

---

## Findings, ranked most-severe first

### 1. CONFIRMED — sharding silently disabled by default
`src/selfupdate/train/online_v3.py:1040`, `src/selfupdate/config.py:224`,
`src/selfupdate/train/validate.py:75-79`

The memory bound introduced by the sharding commit (`83782cd`) is off by
default: `activation_shard_users` defaults to **0** in the config schema,
`validate.py` explicitly permits 0, and `activation_shard_users or B` at
dispatch re-establishes fully unsharded B=256 execution. Any causal_bk
config that does not pin the knob (the K=1 YAMLs shipped in `5bb63dd` did
not) runs unsharded and dies with CUDA OOM hours in when the largest cohort
arrives — observed 2026-07-15/16: every unsharded arm crashed at the
780p×844a cohort, and the scheduler then retried the identical failure for
~13 hours. Fix: pin the knob in every v3.1 YAML, make the default an error
(or a computed safe value) for causal_bk, and add a dispatch-time footprint
check that refuses to start instead of OOMing mid-epoch (cohorts are
deterministic given the seed, so the worst tile shape is knowable up front).
Config/validation change only; no numerics risk.

### 2. CONFIRMED (new) — teacher_hidden residency is unbounded by the shard knob
`src/selfupdate/train/online_v3.py:882`

For `trajectory_source: teacher_hidden` arms, `_bk_prepare_cohort_shards`
keeps every shard's full `teacher_inputs` — n tensors of [B_shard, T, H]
bf16 on device — alive simultaneously in the `shards` list for the whole
cohort. Total resident teacher state is O(n·B_total·T·H) and is NOT reduced
by `activation_shard_users`, which only bounds backward activations. A 4B
teacher_hidden cohort at the 780p×844a shape needs ~36·256·1624·2560·2 B ≈
77 GB of resident teacher inputs alone — over the L40S ceiling no matter how
small the shard knob is set; the job OOMs at cohort-prepare time. The
current 0.8B/4B student_hidden arms are unaffected (teacher_inputs is None),
but any teacher_hidden promotion at 4B-class scale hits this immediately.
Fix direction: host-pinned teacher inputs with per-tile gather+transfer, or
per-shard regeneration. Memory-layout change; certify before campaign use.

### 3. CONFIRMED (new) — unbounded host-side item cache pins the whole teacher cache in RAM
`src/selfupdate/data/dataset.py:249`

`DistillDataset.__getitem__` memoizes every fetched Item — including its
full per-layer hidden target dict materialized by safetensors `get_tensor`
— in the unbounded `self._item_cache`. One epoch pins the entire teacher
hidden cache in host RAM (~35 GB for Qwen3.5-4B). With the standard
scheduler pattern of multiple concurrent arms per host, aggregate host RAM
demand reaches hundreds of GB: kernel OOM-killer or swap-thrash risk that
stalls all lanes — on top of `collate_padded_items` copying the same rows
again into padded batches per cohort (finding 8). Fix: bounded/LRU item
cache, or rely on the safetensors mmap and drop hidden tensors from the
memo. Host-side only; no numerics risk, but measure epoch time (the memo
exists to avoid re-reading Lustre — evict to the mmap page cache, not to
cold storage).

### 4. CONFIRMED — finished rows' KV is retained until the cohort's longest answer ends
`src/selfupdate/train/online_v3.py:1129` (root cause visible at `:1096`)

The per-item trainers released each item's DynamicCache at its answer
boundary; cohort rows have no equivalent: finished rows are only masked
("masked_without_replacement"), so their full per-layer KV keeps growing by
K per tile and stays resident until the single longest answer completes. In
a length-skewed cohort most rows finish early but the cohort holds
B×(prefix+844) KV across all n layers to the end — peak KV up to ~2× what
live rows need, which is exactly the headroom lost in the observed OOMs.
Rows are batch-independent (dead rows' KV serves only their own dead
queries, see `_bk_additive_mask:841-844`), so compaction via `index_select`
of live rows is numerically invisible to every live row. Real work on
DynamicCache internals; certify. Length-bucketed cohorts already bound the
spread, so rank below findings 1-3.

### 5. CONFIRMED (new) — DynamicCache torch.cat growth: O(T²/K) copy traffic + transient doubling + fragmentation
`src/selfupdate/train/online_v3.py:887`

Each per-shard DynamicCache append (`torch.cat` per layer per K-tile)
reallocates and copies the entire prefix+answer KV so far: O(T²/K) copy
traffic per layer per answer, and during each cat the old and new KV tensors
coexist, transiently doubling that layer's KV footprint and producing
stair-step allocations that fragment the allocator. At B=256, T=1624,
4B-class KV (~1.7 GB/layer), every tile's append momentarily needs an extra
~1.7 GB and leaves non-reusable freed blocks — consistent with the fatal
OOM traces showing 4-11 GB "reserved but unallocated". A cohort-preallocated
fixed-shape KV buffer (static-cache style: allocate prefix+max_answer once,
write in place) removes both the copy traffic and the transient. This is the
same direction as the existing fixed-shape/CUDA-graph prototype and is
numerics-adjacent: requires a `scripts/train_certify.py` numerical regression.

### 6. CONFIRMED — hot-loop boolean-mask indexing syncs the GPU per layer per shard per tile
`src/selfupdate/train/online_v3.py:1132-1133`

`stack.loss_view(layer, h_out)[state["query_valid"]]` and
`state["window_targets"][layer-1][state["query_valid"]]` each force a
GPU→host `nonzero()` synchronization: 2·n_layers·n_shards syncs per tile
(72+ at 4B/36 layers). The host blocks on the stream each time, so the
dispatch queue drains and the trainer runs latency-bound instead of
dispatch-ahead — a direct throughput loss in the measured 565 tok/s K=1 and
820 tok/s 4B regimes. This violates the repo's sync-bound law AND the
explicit invariant documented in `collate_padded_items`
(`dataset.py:265-267`: "slice by CPU-side lengths instead of bool-mask
indexing (which would sync the GPU via nonzero())"). Fix: mask-multiply and
divide by `cells` (already a CPU int at `:970`) — no data-dependent shapes.
Numerics-adjacent (reduction order changes): certify.

### 7. CONFIRMED — per-tile pageable, blocking H2D restaging of window targets
`src/selfupdate/train/online_v3.py:975-978`

`_bk_prepare_shard_tile` rebuilds `window_targets` every tile as a host-side
`torch.stack` over all n layers of unpinned `batch.hidden` slices, followed
by a synchronous pageable `.to(device)`. Each K-window pays a fresh
multi-hundred-MB host allocation plus a blocking H2D copy on the compute
stream — at 4B roughly 1.9 GB staged per full-width tile, repeated for every
tile of every cohort of every epoch, even though the bytes are the same rows
of `batch.hidden` each epoch. Fix: pinned reusable staging buffer +
`non_blocking=True` (transfer path upgrade only — NOT the side-stream
prefetch that issues.md measured negative), or a once-per-shard
device-resident target buffer sliced per tile where VRAM allows. Transfer
path change is numerics-free; the device-resident variant trades the memory
the shard docstring (`:948-951`) warns about — config-gate it.

### 8. CONFIRMED (new) — per-cohort re-collation re-pads the full n-layer targets on host
`src/selfupdate/train/online_v3.py:878`

`_bk_prepare_cohort_shards` calls `collate_padded_items` per shard per
cohort per epoch, which re-pads the FULL n-layer aligned hidden targets into
fresh [B, Amax, H] host tensors (`dataset.py:289-303`) — duplicating the
already-memoized item cache — even though only K-token windows are ever
shipped to the device. For the 780p×844a cohort at 4B this is ~40 GB of
host allocation+memcpy per cohort visit, every epoch, with the GPU idle
during cohort prep; host RSS transiently holds item cache + padded copy
simultaneously (compounding finding 3). Fix directions: cache the collated
batch per cohort identity across epochs (cohort membership is deterministic
per epoch seed — memoize keyed on the index tuple), or collate lazily per
K-window. Host-side; measure epoch time either way.

### 9. CONFIRMED (new) — divide-then-remultiply kernel churn in the per-layer write loop
`src/selfupdate/train/online_v3.py:1147-1150`

Per layer per tile the loop computes `tile_losses.append(loss_sum / cells)`
and `tile_grads.append(grad / cells)`, then immediately re-multiplies both
by the same `cells` to form `weighted_losses`/`weighted_grads`: 4·n_layers
pointless scalar kernel launches per tile in a dispatch-bound trainer (~144
extra launches per tile at 4B), plus a needless fp rounding round-trip in
the telemetry sums. Fix: accumulate `loss_sum` and `grad` directly into
`epoch_loss_sums`/`epoch_grad_sums` and defer the single `/epoch_cells` to
epoch end (which already happens at `:1190`/`:1215`). Telemetry-only effect;
trivially certifiable.

### 10. CONFIRMED — O(B·S²) prefill mask transient at full shard width
`src/selfupdate/train/online_v3.py:904-908`

Prefill materializes `prefill_allowed` (bool) and `prefill_mask` (bf16) at
(b_now, 1, S, S) for the full shard width: ~311 MB bf16 + ~156 MB bool at
256 users × 780-token prompts, allocated exactly when per-layer KV is also
growing. For unsharded configs this transient contributed to the headroom
loss at the fatal cohort. Chunking the prefill along the query axis removes
the S² term without changing any attention result; finding 1's sharding
mitigates but does not remove it for large `shard_users`.

---

## Cross-check against the inline review

- Inline M1 (unsharded K=1 configs) → refined by finding 1: the root cause
  is the schema default `0` + `or B`, not merely unpinned YAMLs; validation
  actively permits the dangerous value.
- Inline S1 (bool-mask sync) → finding 6, CONFIRMED as stated.
- Inline S2 (pageable target staging) → finding 7, CONFIRMED; the workflow
  adds the same-bytes-every-epoch observation.
- Inline M2 (finished-row KV) → finding 4, CONFIRMED with the per-item
  precedent (answer-boundary release existed in v3.0 item trainers).
- Inline M3 (prefill mask transient) → finding 10, CONFIRMED.
- Inline M4 (allocator/fragmentation diagnostic) → subsumed by finding 5,
  which identifies a concrete fragmentation *source* (torch.cat stair-step)
  rather than only a symptom; the env-var verification remains worth the
  two minutes.
- New in this pass: findings 2 (teacher_hidden residency), 3 (unbounded host
  item cache), 5 (cache-append copy traffic/fragmentation), 8 (per-cohort
  re-collation), 9 (divide/re-multiply churn).

## Recommended order of work

1. Finding 1 — pin/validate/guard `activation_shard_users` (also breaks the
   scheduler's blind retry loop). Config-only.
2. Finding 6 — masked-mean loss (kill the syncs); certify.
3. Finding 7 — pinned staging for window targets; certify the resident
   variant only if a profile still shows H2D stalls.
4. Finding 9 — accumulate unnormalized sums; trivial, certify.
5. Findings 5+4 — preallocated cohort KV buffer, then live-row compaction
   on top of it; one certified change, biggest memory payoff.
6. Finding 3 and 8 — host-RAM hygiene before any multi-arm-per-host 4B
   campaign.
7. Finding 2 — before any teacher_hidden arm is promoted past 0.8B.
8. Finding 10 — only if large-shard configs persist after (1).
