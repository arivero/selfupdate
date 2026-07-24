# Training Memory Vs Parameter Count

## Current v4 law

Pipeline v4 holds a differentiable graph for one owned student block applied
to detached teacher inputs.  It never retains an end-to-end student training
trajectory.  Memory is therefore the owned frozen weights (or one rotated
block), LoRA/optimizer state, one block's activations, and the selected teacher
store residency. Independent PPP processes split block ownership; no training
activation is transferred between them.

Traditional mixed-precision AdamW full fine-tuning is roughly 16 bytes per
parameter before full-depth activations. V4 instead freezes base/vocabulary
weights, trains block-local adapters, and can rotate an owned block when even
the shard is larger than a card. Report per-stage reserved VRAM, host-master
bytes, teacher-store bytes, and rotation stall separately.

The remainder of this document is a dated v1–v3 measurement ledger. It is
retained as historical evidence, not current executable guidance; references
to removed schedules and scripts resolve through Git history.

## Historical v1–v3 ledger

The claim under test: layerwise forward distillation can train a model with
memory governed by one block or a small tail window, instead of by full-depth
activation storage.

## Resident Terms

| term | summed | sequential | tail-CE |
|---|---|---|---|
| weights | resident model | resident or streamed base | resident or streamed base |
| optimizer | per-block optimizers live | active block only | active tail window |
| activations | one block per local backward | one block | `k` top blocks |
| backward extent | one block | one block | `k` blocks |

`sequential` is the clean scaling story: load one block, train it, write/cache
its outputs, freeze it, then advance. `tail_ce_blocks=k` spends extra memory
only in the final `k` blocks to buy behavioral credit assignment.

## Measured / Expected

Qwen3-0.6B layerwise runs on the origin 12 GB GPU showed the intended memory
shape: strict sequential was roughly one third of the old full-depth training
reference, and LoRA/online-teacher runs were lower still. The exact current
numbers should be read from `runs/results.md` because L40S reruns and schema
changes can shift allocation details.

## Projection

For 120B-class models, the streamed sequential plan keeps one block and its
optimizer state on GPU. Tail-CE keeps `k` blocks in the top window. Everything
else can be loaded/offloaded or held in CPU/disk activation caches. That is
the property this branch is built to preserve.

## Measured (Campaign 2, L40S 46 GB, one card per arm)

Peak `torch.cuda.max_memory_allocated` from each run's metrics
(`vram_gb`; from 2026-07-04 runs also log `vram_reserved_gb` — the
allocator's true device hold, the honest "does it fit" number; nvidia-smi
adds ~0.5–1 GB of CUDA context on top). `runs/results.md` carries both
columns per run.

| arm | trainable | schedule | measured peak | anatomy (approx) |
|---|---|---|---|---|
| 0.6B full-FT (summed, frozen copy) | all 0.6B | summed | 10.3–11.3 GB | fp32 base 2.4 + Adam 4.8 + grads ≤2.4 + bf16 teacher 1.2 |
| 1.7B full-FT (summed, frozen copy) | all 1.7B | summed | 27.9 GB | fp32 6.8 + Adam 13.6 + bf16 teacher 3.4 + grads |
| 4B LoRA r16 (adapters-off teacher) | ~35M | summed | 10.2 GB | bf16 base 8.0 + adapters/opt ~0.5 |
| 4B LoRA r64 | ~140M | summed | 11.7 GB | +4× adapter params ≈ +1.5 GB |
| 8B LoRA r16 | ~50M | summed | 18.5 GB | bf16 base 16 |
| 14B LoRA r16 | ~70M | summed | 31.6 GB | bf16 base 28 |
| 4B full-FT tail window k=8 (frozen copy) | ~0.9B | [expunged] | ~30 GB (nvidia-smi, run in flight) | bf16 student 8 + bf16 teacher 8 + window fp32 3.6 + Adam 7.2 + grads 3.6 |
| gpt-oss-20B LoRA (MXFP4→bf16) | ~60M | summed | 40.7 GB | dequantized base ~40 |

Current complete-adapter accounting for the 2026-07-24 H100 arms (adapter
parameters only, before gradients/optimizer/transient effective deltas):

| model | coverage | rank | adapter params | bf16 bytes |
|---|---|---:|---:|---:|
| Gemma-4-26B-A4B | all decoder Linear + router + 128 packed experts/layer | 16 | 495,790,080 | 0.992 GB |
| Gemma-4-26B-A4B | same | 64 | 1,983,160,320 | 3.966 GB |
| Qwen3.6-27B | dense MLP + softmax and hybrid/linear attention | 16 | 116,727,808 | 0.233 GB |
| Qwen3.6-35B-A3B | hybrid/softmax attention + shared MLP + router + 256 packed experts/layer | 16 | 946,698,880 | 1.893 GB |

These totals are partitioned by owned layers in PPP4. PEFT's packed-parameter
path may also materialize an effective expert delta during a block forward;
the first real H100 gate must measure that transient rather than extrapolate
only from checkpoint size.

## Speed/Memory Ledger — 2026-07-06 Hot-Loop Fixes

The low-memory claim is the research goal; throughput work must never buy
speed by silently re-inflating the activation or optimizer footprint. Every
acceleration landed on 2026-07-06 is itemized here with its exact memory
price, split by WHICH memory it spends — device VRAM (the scarce, claim-
bearing resource) vs host RAM (abundant, invisible to the memory claim).

| fix (commit) | speed effect | VRAM cost | host RAM cost |
|---|---|---|---|
| prefix-slice per-example losses (`abfe6d6`) | removes ~n_layers x B host syncs per micro-batch | 0 | 0 |
| stacked log-flush transfer (`27a4943`) | n_layers -> 1 syncs per accum boundary | 0 | 0 |
| dataset item memoization (`1874e8b`) | teacher-cache Lustre reads once instead of every epoch | 0 | = needed cache, fp16 (see below) |
| streamed batched teacher targets (`83cb0ef`) | none (same math) | **saving**: n+1 -> 1 full-sequence teacher states | 0 |
| `train.window_dedup` (`b466a3e`) | 1.15-1.22x measured on slide8 W=8 | 0 (peak graph stays W blocks; measured parity within 10 MB) | 0 |
| padded/bucketed batching (pre-existing, benched together) | ~2.3x tokens/s at micro_batch 4 | +3.9 GB measured at 0.6B/W=8/B=4 (scales ~B x T x window graph) | negligible |

Measured on the slide8 reference arm (Qwen3-0.6B, W=8, vocab_mse, GPU
shared with a campaign job, `scripts/train_batch_bench.py`):

| batching | variant | tokens/s | peak alloc |
|---|---|---|---|
| item | replay | 136.5 | 8.66 GB |
| item | window_dedup | 165.9 | 8.65 GB |
| bucketed B=4 | replay | 465.2 | 12.57 GB |
| bucketed B=4 | window_dedup | 533.9 | 12.56 GB |

What each entry means for the memory claim:

- **window_dedup does not touch the claim.** The replay path holds one
  W-block connected graph during each `window_step`; dedup holds W
  single-block graphs (freed after each block's last covering window,
  ascending-endpoint order). Same peak by construction, confirmed
  empirically above. Backward extent per loss is W blocks in both — the
  "activations = k blocks" row of the Resident Terms table is unchanged.
- **Batching is the only speed knob that spends VRAM**, and it is a
  dial, not a default: `train.batching: item` remains the low-memory
  setting, and the +3.9 GB at B=4 is the padded working set
  (B x T activations inside the window graph plus padded targets). At the
  memory-critical scale rungs (streamed sequential, 120B plan), keep
  batching at `item` or budget B against the table above.
- **Memoization spends HOST RAM only**: after epoch 1 the needed teacher
  targets stay resident — the whole needed cache, fp16. Concretely
  3.85 GiB for the 0.6B rag-remove poem cache (measured); it scales as
  n_examples x n_layers x A x H x 2 bytes, so ~18 GB at 8B scale. This
  is never GPU memory, and the schedules that carry the scaling story
  are exempt by construction: `sequential` memoizes one layer at a time
  (the `need_layers` setter drops the previous stage), and online-teacher
  runs (LoRA, frozen copy, 120B streaming plan) have no disk cache at
  all. If host RAM ever becomes the binding constraint, the memo — not
  the VRAM plan — is what to revisit.
- **The rejected alternative is part of the record**: vectorizing the
  per-example readout KL as one [B, Rmax, V] softmax would have cost
  ~0.5 GB fp32 per side at V=151k — speed bought with exactly the memory
  the claim protects. The prefix-slice fix gets the sync win at zero
  memory; keep it that way when touching these loops.

## The Fits-Where-Traditional-Cannot Argument

Traditional full-backprop AdamW fine-tuning (mixed precision) needs
≈16 bytes/param (bf16 weights + fp32 master + two Adam moments) plus
full-depth activation storage:

| model | traditional full-FT | one L40S (46 GB)? | layerwise measured |
|---|---|---|---|
| 0.6B | ~10 GB | yes | 10–11 GB (summed keeps all masters — no saving BY DESIGN at this size) |
| 4B | ~64 GB | **no** | **~30 GB** ([expunged] full-FT window) / 10 GB (LoRA) |
| 8B | ~128 GB | no | 18.5 GB (LoRA) |
| 14B | ~224 GB | no | 31.6 GB (LoRA) |
| 20B MoE | ~320 GB | no | 40.7 GB (LoRA, exclusive lane) |

The structural point: `summed` full-FT saves nothing on weights/optimizer
(it trains every block every step) — its saving is ACTIVATIONS ONLY
(one block's graph instead of full depth). The memory story at scale is
carried by `[expunged]` (fp32 state for k blocks only), `sequential`
(one block at a time — the 120B streaming contract), and LoRA (base in
bf16, adapters-off teacher for free). The 4B [expunged] run is full-FT
QUALITY training of a 0.9B-param window on a card where traditional 4B
full-FT cannot even load its optimizer.
# Node-local model-cache staging

Before a GPU campaign, stage the precise Hugging Face model snapshots needed
to node-local `/tmp`; this reduces Lustre metadata/read pressure without
duplicating the durable account cache. The workflow and capacity rule are in
[cache staging](cache_staging.md).
