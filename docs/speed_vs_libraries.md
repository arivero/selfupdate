# v4 layerwise speed vs mainstream LoRA fine-tuning libraries

Positioning of this repo's pipeline-v4 layerwise trainer against the mainstream
open-source LoRA/QLoRA fine-tuning stacks, for the specific task we care about:
**few-shot LoRA (small ~2000-example dataset, many short epochs) on VERY large
models (27B–400B, dense and MoE).** Compiled 2026-07-19 from a fan-out of
sourced web surveys; every external number is cited. Read the caveat first —
the numbers are NOT directly comparable, and the doc says so plainly.

## The honest-comparison caveat (read this first)

A v4 "epoch" is **not** the same unit of work as an SFT epoch in the libraries
below, so raw tokens/sec and "time per epoch" are **not apples-to-apples**:

- **Objective/graph.** v4 is block-local: each block trains its LoRA adapter
  against a **stored teacher hidden state** with a detached input and no
  cross-block gradient. Mainstream LoRA does a full-depth forward+backward on
  the next-token loss. v4 therefore does no end-to-end backprop.
- **Teacher forwards are captured ONCE.** After epoch 0, v4 does **zero**
  teacher forwards (`capture_seconds=0`); later epochs replay the stored
  targets. Standard SFT recomputes the full forward every step.
- **Regime.** Our target is many cheap epochs over a tiny few-shot set (the
  "write a large system prompt into the weights" use case). The published
  library benchmarks are large-corpus SFT (10k–100k+ samples, seq-len
  2048–3072).

**The only fair axes** are therefore (a) **model-size ceiling reachable on a
given GPU count**, and (b) **wall-clock per epoch at matched model-size ×
GPU-count × dataset-size** — and for (b) no external number exists at our exact
few-shot regime, which we state rather than paper over. Raw tokens/sec below is
provided for *context on what each library optimizes*, not as a v4 scoreboard.

## v4 reference numbers (this repo, measured full 2071-item epochs)

Steady-state seconds/epoch on H100-80GB, LoRA, store-once teacher hiddens
(dataset = 2071 examples; see `docs/training_pipeline_v4.md`):

| model | arch | best config | s/epoch | GPUs |
|---|---|---|---:|---|
| Qwen3.5-397B-A17B | MoE, bf16 | PPP8 store+rotate (2 nodes) | ~35 | 8×H100 |
| Qwen3.5-122B-A10B | MoE | PPP8 store / PPP4 store | 20.3 / 40 | 8 / 4 |
| Qwen3.6-35B-A3B | MoE | PPP4 store | 15.9 | 4 |
| Qwen3.6-27B | dense | PPP4 store | 55 | 4 |
| gemma-4-31B | dense | PPP4 store | 30.6 | 4 |
| gemma-4-26B-A4B | MoE | PPP4 | 12–14 | 4 |

Adam ≡ SGD in wall-clock at this scale (122B 20.2 vs 20.3 s; 397B 35 vs 36 s) —
the per-block AdamW moment update, including rotating moments off-card, is
hidden behind the forward/relay. All numbers are audited full-epoch wall-clock
(`token_events` verified constant across epochs; see the 2026-07-18 audit that
retracted a contaminated 26B "61.8 s" figure).

## The comparison at a glance

| Library | Training paradigm | Multi-GPU LoRA | Published LoRA model ceiling | Pipeline / layer-sharded training? | MoE-LoRA throughput published? |
|---|---|---|---|---|---|
| **v4 (this repo)** | Layer-sharded pipeline + capture-once teacher, block-local | ✅ 4–8 GPU, 2 nodes | 397B-A17B bf16 on 8×H100 (measured) | ✅ (forward-only; no cross-stage backward) | ✅ (397B/122B/35B, this repo) |
| **Unsloth** | Single-GPU custom Triton kernels + QLoRA | ⚠️ not GA in free tier (routes to DDP/FSDP) | 671B via quant (not routine); single-GPU focus | ❌ | ❌ |
| **HF PEFT + TRL + FSDP/QLoRA** | Data-parallel FSDP sharding | ✅ FSDP1/2, multi-node capable | gpt-oss-120B (8×H100, no throughput) | ❌ ("not pipeline or tensor parallel") | ❌ |
| **PyTorch torchtune** | FSDP2 (+ TP/CP/EP DTensor) | ✅ FSDP2, 2D parallel | 405B QLoRA on 8×A100 (653 tok/s) | ❌ ("genuinely absent"); **sunset 2025** | ❌ |
| **NVIDIA NeMo / Megatron-Core** | TP×PP×CP×EP + dist. optimizer | ✅ multi-node, all axes | Llama-3-70B LoRA (8×H100); DeepSeek-V3 671B LoRA *recipe* (no perf) | ✅ PP (connected backward graph, 1F1B) | ❌ (MoE only under *pre-training*) |
| **Axolotl** | FSDP/DeepSpeed + sequence-parallel | ✅ DDP/FSDP/DeepSpeed/Ray | 405B (4-bit, multi-node) | ❌ (only sequence parallel) | ⚠️ ScatterMoE LoRA path, no perf table |
| **LLaMA-Factory** | DeepSpeed/FSDP/Ray | ✅ ZeRO-2/3, FSDP2 | up to 671B supported; 70B QLoRA feasible | ❌ | ❌ |
| **DeepSpeed ZeRO + kernels** | ZeRO-1/2/3 (+ optional PP) | ✅ | substrate for the above | ⚠️ has PP, but ZeRO-3+PP → ZeRO-1 only | — |

## Per-library findings (sourced)

### Unsloth
Single-GPU speed leader via hand-written Triton kernels (RoPE/RMSNorm/CE/SwiGLU
+ a manually-derived attention+LoRA backward). **2–2.7× vs HF** (QLoRA r16, 1×A100/T4;
[HF blog](https://huggingface.co/blog/unsloth-trl)); up to **7× on MoE** (gpt-oss-20B,
B200; [MoE docs](https://unsloth.ai/docs/basics/faster-moe)). Concrete: Llama-3.1-8B
LoRA, 100k samples, 1 epoch = **4h45m on 1×A100** ([tutorial](https://huggingface.co/blog/mlabonne/sft-llama3)).
**Multi-GPU not GA in the free tier** ("announcing official multi-GPU support soon",
[docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth)); paid tiers gate it.
No layer-sharded/pipeline/cached-teacher analog. Integrity note: a third-party paper found
Unsloth's headline 46k tok/s figure had **zero gradient norms** (a mode with gradients
disabled) — headline speed numbers can hide non-training configs.

### HuggingFace PEFT + TRL + FSDP/QLoRA
The "standard" path. **Data-parallel FSDP sharding, explicitly not pipeline/tensor
parallel** ([Answer.AI](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html)).
Anchors: Llama-2-70B QLoRA FSDP = **667 s/pass on 4×H100**, 672 s on 8×A100
([fsdp_qlora bench](https://github.com/AnswerDotAI/fsdp_qlora/blob/main/benchmarks_03_2024.md));
Llama-3-70B FSDP+QLoRA = **~1.25 h on 4×H100** for 3 epochs/10k samples
([philschmid](https://www.philschmid.de/fsdp-qlora-llama3)). Published ceiling:
**gpt-oss-120B LoRA on 8×H100** ([Databricks](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-gpt-oss-120b-ddp-fsdp),
no throughput). PEFT MoE cost: it **materializes the LoRA delta for every expert
regardless of routing** ([PEFT lora.md](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md)).
No verified LoRA number above ~120B on this exact stack.

### PyTorch torchtune
FSDP2, clean throughput tables. Llama-3.1-70B **LoRA = 3497 tok/s on 8×A100**;
Llama-3.1-405B **QLoRA = 653 tok/s on 8×A100** ([README](https://github.com/meta-pytorch/torchtune)).
Direct head-to-head (paper, 1×H100, Qwen3 LoRA r16, tok/s): 8B — torchtune 2745 vs
Unsloth 1836 vs Axolotl 1609; 4B — 2616/1826/1605 ([arXiv:2605.21442](https://arxiv.org/html/2605.21442v1)).
**Pipeline parallelism "genuinely absent."** Note: **torchtune is sunset** (last
release v0.6.1, 2025-04; PyTorch consolidating into torchtitan) — cite carefully.

### NVIDIA NeMo / Megatron-Core
The serious multi-node engine and the **closest mainstream analog to v4's
layer-sharding** (pipeline parallelism). Official LoRA table is **one model,
one GPU count**: Llama-3-70B, 8×H100 FP8, TP2/PP4/VP20 = **2643 tok/s/GPU
(735 TFLOP/s/GPU)**; GB200 6206, GB300 7481 ([perf summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)).
DeepSeek-V3 671B LoRA **recipe** exists (48×H100, LoRA on MLA/attention only, none
on experts by default; [guide](https://docs.nvidia.com/nemo-framework/user-guide/25.09/llms/deepseek_v3.html))
but **no published throughput**. **Zero LoRA numbers for any MoE model** — DeepSeek-V3/
Qwen3-235B/Kimi-K2 appear only under *pre-training*. Its PP keeps a **connected
end-to-end backward graph** across stages (1F1B/interleaved) to amortize the
pipeline bubble — see positioning below. EP (expert parallel, all-to-all token
routing) is the axis v4 does not replicate.

### Axolotl
Config-driven wrapper over FSDP/DeepSpeed. Llama-3.1-8B QLoRA = 5.8 h (1×A100, 2
epochs); Llama-3.1-70B **full-FT = 18 h on 8×H100 / 50k examples**; 405B via 4-bit
multi-node ([Spheron](https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/),
[405B post](https://axolotlai.substack.com/p/fine-tuning-llama-31b-waxolotl-on)).
Only **sequence parallelism** (ring-flash-attn) for scale, not pipeline. ScatterMoE
LoRA path exists ([multi-gpu docs](https://docs.axolotl.ai/docs/multi-gpu.html)), no perf table.

### LLaMA-Factory
Broadest model coverage (0.1B–671B). Llama-2-7B LoRA = 1954 tok/s (A100);
Llama-3.1-8B QLoRA = 3.4 h (A100), within 6% of Unsloth
([perf wiki](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison)).
DeepSpeed ZeRO-2/3, FSDP2, Ray; no pipeline/layerwise. Extras: GaLore (−62.5% mem),
BAdam (−~50% backward time), Liger, Unsloth integration.

### DeepSpeed ZeRO + kernel accelerators (Liger, FlashAttention)
The substrate under Axolotl/LLaMA-Factory/HF. ZeRO-2 LoRA Llama-2-7B = 15,734 tok/s
(1×A800); **ZeRO-Offload is a 20–36× slowdown** (bandwidth-bound;
[arXiv:2311.03687](https://arxiv.org/html/2311.03687v2)). **Liger Kernel: +20% throughput /
−60% memory** end-to-end (Llama-3-8B, 8×A100; [paper](https://arxiv.org/pdf/2410.10989));
fused RMSNorm 8×, CE 3× / −5× mem. **FlashAttention-3: 1.2–2.4× vs FA2** on H100
([Spheron](https://www.spheron.network/blog/flashattention-2-vs-flashattention-3-h100-h200-guide/)).
DeepSpeed **has** pipeline parallelism, but **ZeRO-3 + PP degrades to ZeRO-1**
([tutorial](https://www.deepspeed.ai/tutorials/pipeline/)) — the two don't compose cleanly.

## Where v4 sits — the architectural positioning

1. **No consumer/mid-scale LoRA tool does pipeline / layer-sharded *training*.**
   Unsloth (single-GPU), PEFT/TRL (FSDP data-parallel), torchtune (FSDP2, "pipeline
   absent"), Axolotl (sequence-parallel), LLaMA-Factory (ZeRO/FSDP) are all
   data-parallel-sharding families. The only mainstream layer-sharded path is
   **NeMo/Megatron pipeline parallelism**.

2. **Even Megatron's PP solves a strictly *harder* problem than v4.** Megatron PP
   slices layers into contiguous stages (same shape as v4's stage map) but keeps a
   **single connected end-to-end backward graph** streamed over many micro-batches
   with 1F1B/interleaved schedules and virtual pipelining (VP=20) *purely to shrink
   the pipeline bubble*. v4 is **block-local with a captured teacher target**: no
   cross-block gradient exists, so there is **no cross-stage backward P2P and no
   pipeline bubble to amortize** — only a one-time forward layer-sharded capture.
   v4 trades the ability to learn cross-layer credit (which the objective forbids
   by design) for a dramatically simpler, bubble-free parallel schedule.

3. **Capture-once removes the per-epoch teacher cost entirely.** No surveyed library
   caches teacher/target activations across epochs; every step recomputes the full
   forward. v4's `capture_seconds=0` after epoch 0 is why 20–30 epochs over a tiny
   set on a 122B–400B model cost minutes, not hours.

4. **Model-size ceiling on 8×H100.** Published LoRA ceilings on comparable hardware:
   Megatron Llama-3-70B (2643 tok/s/GPU, 8×H100), torchtune 405B **QLoRA** (653 tok/s,
   8×A100), PEFT gpt-oss-120B (8×H100, no number). v4 trains **397B-A17B in bf16 (not
   quantized)** on 8×H100 via forward-only layer-sharding + rotation — at or beyond
   the published mainstream ceiling, and notably without 4-bit quantization.

5. **The MoE-LoRA gap is real and industry-wide.** NVIDIA, torchtune, and HF publish
   **no** LoRA throughput for any MoE model; PEFT even warns it computes every
   expert's LoRA delta regardless of routing. v4 has measured MoE-LoRA epoch times
   at 35B/122B/397B — a regime with essentially no external published baseline.

## Is this novel? Prior art on the ingredients

v4's *synthesis* was **not found elsewhere as a single system**, but each
ingredient has separate prior art — the honest framing is "unusual combination,"
not "no prior art":

- **Pipeline-parallel LoRA — but a different kind.** **mLoRA** (VLDB 2025,
  [arXiv:2312.02515](https://www.vldb.org/pvldb/vol18/p1948-tang.pdf)) pipelines
  **multiple independent adapter jobs** across GPUs (different tasks/customers
  sharing a pipeline), ~30% faster average completion than FSDP — **not** one run
  whose own layers are sharded stage-to-stage. **LoRAFusion** (arXiv:2510.00206)
  is another recent LoRA-efficiency system. Neither shards a single run's depth
  with boundary relay the way v4 does.
- **Cached teacher hidden states.** Known inside distillation systems but aimed at
  a different problem (teacher↔student bandwidth) and framed as valid only
  **off-policy**: Tilde's **Nitrobrew** transmits compressed teacher hiddens and
  notes they "can be cached" off-policy, but its headline regime is on-policy and
  recomputes the teacher every step. v4's frozen-teacher few-shot corpus is exactly
  the off-policy case where capture-once is exact.
- **Block-local / greedy layer-wise training.** A real but niche line: "Greedy
  Layerwise Learning Can Scale to ImageNet" (arXiv:1812.11446), and LLM-adjacent
  **single-device** analogs **BAdam** (block-coordinate updates, one layer at a
  time; [arXiv:2404.02827](https://arxiv.org/abs/2404.02827)) and **LISA** (random
  layer freezing, arXiv:2403.17919) — none shard layers across devices or cache
  teacher targets, and none are LoRA-specific.
- **Conclusion:** no mainstream framework combines (a) contiguous-block
  layer-sharding across GPUs/nodes with boundary-state relay for a *single* LoRA
  run, (b) teacher hiddens captured once with zero later teacher forwards, and (c)
  LoRA-only block-local backward, at 100–400B scale. v4 fuses three known-but-
  separate ideas into one system.

## Related but a different task: serving-side LoRA

The most search-visible "LoRA + scale" topic is **serving** many trained adapters,
which is **not training speed** and must not be cited as a comparator: **S-LoRA**
([arXiv:2311.03285](https://www.lmsys.org/blog/2023-11-15-slora/), up to 4× vs HF
PEFT serving), **vLLM multi-LoRA** (sub-ms adapter hot-swap), **SGLang** SGMV
mixed-adapter batches. They optimize *which trained adapter answers which request*,
not how fast an adapter was trained.

## What is and isn't fair to claim

- **Fair:** v4 reaches a **higher bf16 LoRA model ceiling on 8×H100** than any
  surveyed consumer/mid-scale tool, via a paradigm (forward layer-sharding +
  capture-once) none of them implement; and it is purpose-built for the
  **many-epochs-tiny-dataset** regime none of them benchmark.
- **Fair:** v4's per-epoch wall-clock (20–35 s at 122B–397B) is far below any
  published per-epoch time at that model scale — *because* of the regime + cached
  teacher + block-local backward, which must be stated alongside the number.
- **NOT fair:** claiming v4 is "N× faster than library X" on tokens/sec. Three
  concrete reasons the token throughput is asymmetric:
  1. **v4's backward does structurally less compute per item** — it is truncated
     to one block, never crossing block boundaries and never touching the
     embedding or the LM-head/vocabulary projection (a large FLOP share at scale).
     A standard LoRA step backprops through the full depth *and* the vocab head.
     Comparing tok/s would credit v4 for doing less math per item, not for the
     pipeline/caching engineering being evaluated.
  2. **v4's teacher-forward cost is paid once, up front, outside the reported
     epoch** — the SFT baselines have no teacher at all, so neither number
     reflects a symmetric workload.
  3. **MoE active-vs-total** — 397B-A17B / 122B-A10B / 35B-A3B have far smaller
     *active* parameter counts; compute tracks active, not total. Comparing
     397B-A17B's 35 s to a dense-70B tok/s figure conflates "large MoE, cheap
     active path" with "large dense, full path" unless active-parameter-matched.
  Only wall-clock/epoch at matched model × GPU × dataset is comparable, and no
  external number exists at our few-shot regime. Always label v4's figure
  "distillation-style block-local pass with cached teacher, not standard SFT
  tok/s" and never place it in the same table column as an Unsloth/torchtune
  tok/s number without that caveat attached.
- **Borrow-from list:** Megatron's **expert parallelism** (all-to-all token routing,
  `min_gpus = PP × max(TP×CP, EP×ETP)`) if v4 ever needs MoE-aware sharding beyond
  depth-slicing; **Liger fused kernels** and **FlashAttention-3** as orthogonal
  substrate wins for the forward-capture and block compute.

## Sources

Primary per-library sources are linked inline above. Survey compiled 2026-07-19
from parallel web research; two unverifiable third-party figures (a 405B PEFT+TRL
run and a "4-node×2-A100 gpt-oss-120B" number) were investigated and **rejected**
rather than cited. Where a library claims a capability without a benchmark
(DeepSeek-V3 LoRA recipe throughput, MoE-LoRA numbers), that gap is stated
explicitly rather than filled with a pre-training or full-fine-tune number.
