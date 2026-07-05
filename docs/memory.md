# Training Memory Vs Parameter Count

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
