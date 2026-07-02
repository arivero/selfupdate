# Scaling plan: from Qwen3-0.6B to DeepSeek/GLM-class MoEs on 4×H100

The small-scale code paths were chosen so that each one maps onto a
big-model strategy without redesign. This file records the mapping.

## Where inference engines (vLLM / sglang) fit — and where they cannot

The student must be trained, so it can never run under an inference engine.
The **teacher**, however, is frozen and inference-only, and its work splits
into three products with different ideal backends:

| teacher product | needed by | big-model backend |
|---|---|---|
| `<think>` traces (thinking mode) | dataset build | **vLLM/sglang** — batched generation is exactly what they optimize; swap `teacher/generate.py`'s `model.generate` for an engine client |
| top-k logits + logsumexp | KD | vLLM `prompt_logprobs` gives per-position top-k on a forced continuation; sufficient for the whole KD cache without ever materializing hidden states |
| per-layer hidden states at the aligned span | layerwise regimes | **layer-streamed HF forward** (below) — no inference engine exposes internal residual streams; a custom loop beats patching one |

## Layer-streamed teacher forward (the same trick as `StudentActCache.advance`)

To build the hidden-state cache for a model that does not fit in GPU memory:
keep the whole dataset's activations for one layer, load one block's weights
at a time (`safetensors` lazy per-tensor reads), advance all examples through
that block, write the aligned-span slice to the teacher cache, repeat.

Memory = one block + one activation set; the model as a whole is never
resident. This is the identical contract the sequential student trainer
already implements and tests (`test_sequential_never_runs_frozen_blocks`) —
`BlockStack.run_block`'s activations-in/activations-out signature with
`load()/offload()` filled in.

Cache sizes (fp16, aligned span only), per 1k examples × 512 aligned tokens:

| model class | layers × hidden | all layers | one layer (sequential needs only this) |
|---|---|---|---|
| Qwen3-0.6B | 28 × 1024 | ~29 GB | ~1 GB |
| Qwen3-4B | 36 × 2560 | ~94 GB | ~2.6 GB |
| 30B-A3B MoE | 48 × 2048 | ~100 GB | ~2 GB |
| 120B-class (DeepSeek/GLM) | ~61 × 7168 | ~450 GB | ~7 GB |

The per-layer safetensors key layout (`{example}/h{L:02d}`) already gives
lazy single-layer reads, so sequential training streams ~7 GB per stage at
120B instead of touching the full cache. Optional `int8 + scale` storage
halves these numbers when needed.

## LoRA's structural bonus: the teacher is already resident (`train.online_teacher`)

With LoRA, student = base weights + adapters, and LoRA's B matrices start at
zero — so the frozen teacher is literally the same resident model with
adapters disabled (`peft`'s `disable_adapter()`). `train.online_teacher: true`
computes teacher targets per step this way instead of reading a disk cache:

- **Memory**: one model + adapters; no second copy, no cache.
- **Disk**: zero — at 120B/Quijote scale this replaces a ~450 GB hidden-state
  cache; the trade is one extra no-grad forward per step (teacher input is
  ~1.5–2× the student length).
- **When cache still wins**: many runs over the same fixed dataset (the 0.6B
  grid — build once, train twelve times), and full-FT runs, where the base
  weights drift and are no longer the teacher.
- **Extra capability**: teacher inputs may change during training (e.g.,
  Deng-style curriculum removal of the privileged block) — impossible with a
  prebuilt cache.

Implemented for KD and layerwise-`summed`; `sequential` needs a lockstep
teacher activation cache (planned). Equivalence with the disk cache is tested
(`tests/test_online_teacher.py`).

## Student training by regime at 4×H100 (320 GB)

- **KD full-FT**: impossible at 120B (fp32 grads+Adam ≈ 1.4 TB). Feasible to
  ~7B with FSDP2 via accelerate; beyond that KD runs LoRA-only (adapters +
  frozen bf16 base sharded across 4 GPUs, `peft` unchanged).
- **Layerwise `sequential`**: one block bf16 (2–4 GB at 120B) + fp32 AdamW
  states for it (~4×) + activations — fits on a *single* H100 regardless of
  total model size. The 4-GPU win is pipelining stage L's training with
  stage L−1's activation advance and stage L+1's target prefetch.
- **Layerwise `summed`**: per-block backward already isolates gradients, but
  all blocks' optimizer states are live — at 120B keep AdamW states on CPU/
  NVMe and stream them with the block (per-block optimizers were chosen over
  one param-grouped optimizer precisely for this).
- **Embarrassing parallelism (future schedule)**: if block L's *input* is the
  cached teacher `h_{L-1}` instead of the student's own stream, every block
  trains independently — 4 GPUs train 4 blocks concurrently with zero
  communication. Costs a full-sequence teacher cache (~3× aligned-span size)
  and accepts composition error at inference; worth registering as a third
  schedule when the H100s arrive.

## DeepSeek-V4/GLM-5.2-class MoE specifics

- Hidden-state matching stays **post-MoE-combine** (block output) — the
  `BlockStack` contract is unchanged; never match per-expert.
- Router aux/load-balancing losses: we call `model.model(...)`/per-block
  forwards, so HF's aux losses are never added to our objective —
  distillation runs with load balancing off by construction; log routing
  agreement (teacher vs student top-k experts at aligned positions) as a
  metric instead.
- Teacher/student routing disagreement makes hidden targets non-smooth early
  in training; if it stalls, add an auxiliary router-logit match at aligned
  positions before touching anything else.
- Localization gains a which-expert axis for free: `weight_deltas.py`'s layer
  regex captures expert submodules (`mlp.experts.N.*`) as distinct modules;
  report shared vs routed experts separately.
- HF module layout (`model.model.layers[i]`, `model.model.norm`,
  `model.model.embed_tokens`, `model.lm_head`) is shared by the Qwen, Llama,
  DeepSeek and GLM ports, so `BlockStack` transfers; per-arch differences are
  confined to `run_block` kwargs and RoPE handling (MLA models compute rotary
  inside attention — `position_embeddings` becomes optional there).

## What deliberately does NOT change

The masking abstraction, aligned-span convention, cache format, loss
functions, eval suite, and the per-block trainer interface are all
model-size-independent. Scaling work = implement `load()/offload()`, add an
engine-backed trace harvester, and write the arch adapters above.
