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
