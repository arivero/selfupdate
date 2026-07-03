# Training memory vs parameter count — the headline metric

The claim under test: block-local training cuts the memory needed to train a
model of N parameters to a fraction of full backpropagation's — because full
backprop (even with LoRA) must hold activations across the whole depth and,
for full fine-tuning, gradients + optimizer states for every parameter.

## Decomposition (per method, resident on GPU)

| term | KD full-FT | KD LoRA | LW summed (full-FT) | LW sequential (full-FT) |
|---|---|---|---|---|
| weights | 4N (fp32) | 2N (bf16) | 4N (fp32) | 2N (bf16) + 2·(N/L) fp32 block |
| gradients | 4N_t | 4·adapters | 4N_t | 4·(N/L) |
| optimizer (AdamW) | 8N_t | 8·adapters | 8N_t | 8·(N/L) |
| activations | full depth (ckpt: ~√-depth) | full depth | **one block** | **one block** |
| backward extent | whole net | whole net | one block | one block |

N_t = trainable params (blocks only in this repo, ≈ 0.59·N for Qwen3-0.6B).
L = number of blocks. The sequential column is the scaling story: everything
that grows with N sits in the frozen bf16 term (2N), and *that* is the part
`load()/offload()` streaming removes at 120B — leaving O(N/L), a few GB, per
GPU regardless of model size.

## Measured (RTX 3060, peak torch allocation, Qwen3-0.6B = 0.75B params)

| method | VRAM | fraction of KD full-FT |
|---|---|---|
| KD full-FT (fp32, grad ckpt) | 9.45 GB | 100% |
| LW summed full-FT | 7.81 GB | 83% (activations saved; all-block Adam kept) |
| KD LoRA, fp32 base (initial) | 3.75 GB | 40% |
| LW sequential full-FT, fp32 base (initial) | 3.22 GB | **34% — the "one third"** |
| LW teacher_censored LoRA | 3.13 GB | 33% |
| KD LoRA, bf16 base | 1.87 GB | **20%** |
| LW sequential, bf16 base + fp32 active block | ~1.9 GB (projected; re-measure queued) | ~20% |

Note what LoRA alone does NOT fix: it shrinks gradients/optimizer to the
adapter size but still runs the full-depth backward with full-depth
activations, and the propagated error signal crosses every layer. The
sequential schedule is the only row whose activation term and backward extent
are one block — that is the property that survives to 120B.

## Projections (bf16 resident base, one fp32 active block, batch 1, seq ~1k)

| model | params | KD full-FT | KD LoRA | LW sequential (resident) | LW sequential (streamed) |
|---|---|---|---|---|---|
| Qwen3-1.7B | 2.0B | ~27 GB (doesn't fit 3060/4090-24) | ~5 GB | ~5 GB | ~1.5 GB |
| Qwen3-4B | 4B | ~60 GB | ~10 GB | ~10 GB | ~2.5 GB |
| Qwen3-32B | 32B | ~500 GB | ~70 GB | ~68 GB | ~6 GB |
| 120B-class | 120B | ~1.9 TB | ~250 GB | ~245 GB | ~8–12 GB |

"Streamed" = the `BlockRunner.load()/offload()` path (one block's weights +
optimizer on GPU at a time, activation cache on disk/CPU): the only column
that keeps a 120B trainable on a single H100 — and the sequential schedule is
the only method here that is *exactly equivalent* to its resident version
(bit-identical losses, verified by the mock-streaming test planned in M6).

Caveat to keep honest: sequential layerwise currently buys this memory profile
at the cost of recitation quality (no variant recites yet at 0.6B). The
program's practical target is therefore the cheapest *hybrid* that recites —
e.g., sequential pre-conditioning + a short KD/LoRA polish — with the memory
account of each phase reported separately.
