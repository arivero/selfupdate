# GPT-OSS-120B: MXFP4 teacher cache and layerwise training implications

## What was measured

On the H100 node, GPUs 0--1, the native OpenAI GPT-OSS-120B checkpoint was
loaded with Transformers 5.12.1 and `kernels==0.12.0`. The run used the
automatic two-card device map, no CUDA graphs, teacher batch 64, an 8,192
token cap, and the exact response token IDs already produced by the vLLM
campaign. It therefore measured hidden-state replay/cache construction, not
answer generation.

The full V5RS set completed for all 2,071 questions:

| quantity | value |
|---|---:|
| total wall time | 113.4 s |
| teacher forward | 86.8 s |
| device-to-host copy | 0.6 s |
| asynchronous cache write | 23.3 s (28.7 s storage accounting) |
| stored hidden bytes | 39.11 GB |
| stored dtype | bfloat16 |
| effective teacher batches | 1--64 (requested 64) |

The copied vectors are 36 hidden states per example. Device-to-host transfer
is not the bottleneck here: it is under one percent of forward time. The
dominant avoidable cost after compute is serialization/storage. The durable
timing evidence is in `runs/teacher_cache_h100/gptoss120_v5rs_mxfp4_bf16_b64_full/`.

The exact responses were imported, so generation timings are deliberately
zero in this run. The recitation diagnostics attached to those responses were
next/previous word-LCS 0.510 and hard-cut fraction 22.5%; they are response
quality diagnostics, not a claim that this cache run generated the answers.

## Why MXFP4 matters

GPT-OSS-120B is released with MXFP4 expert weights. Native loading keeps the
quantized expert representation and uses the fused kernel path. Without the
pinned `kernels==0.12.0` package, Transformers silently selected a bf16
dequantization fallback; on two H100s that consumed nearly all memory and
failed during weight materialization. The cache loader now refuses that
ambiguous fallback and requires native MXFP4 support.

The first cache attempt also demonstrated that the *stored* hidden vectors
cannot be blindly cast to float16: an outlier channel became non-finite.
bfloat16 has the same two-byte storage size while retaining the required
exponent range. This is a property of the trajectory representation, not a
request to dequantize the teacher weights.

## Consequences for student--teacher comparisons

The cache is a trajectory of the quantized GPT-OSS teacher. A student trained
against it is learning the behavior of that exact MXFP4 checkpoint, including
any quantization residual, rather than an abstract bf16 version of GPT-OSS.
This is valid, but comparisons must keep the teacher identity explicit.

For a fair layerwise comparison:

1. Use the same tokenizer, prompt/answer alignment, frozen vocabulary head,
   and layer indexing on every student.
2. Treat the bfloat16 cache as the reference target. Do not convert it to
   float16 to save disk; the overflow test has already falsified that path.
3. Establish an epoch-zero loss floor for an unmodified student using the
   same MXFP4 teacher. A dequantized-bf16 student will not necessarily have
   zero loss, even when it has the same nominal checkpoint, because the
   forward trajectories differ.
4. Report whether the student base is native MXFP4 or dequantized bf16. A
   reduction in hidden loss is otherwise confounded with changing the base
   numerical regime.

The cache itself is detached. During training, gradients flow only through
the student hidden states; the teacher vectors on disk receive no gradient.
For `vocab_mse` and lens-KL losses, the frozen vocabulary head can still
decode the bfloat16 targets in float32, preserving outlier information while
keeping the head frozen under the branch's vocabulary law.

## Consequences for layerwise backpropagation

There are two distinct GPT-OSS training paths in this repository:

- `dense_or_black_box` leaves native MXFP4 experts in place. This preserves
  the memory and fused-forward advantages, but backward/autograd support
  through the installed MXFP4 expert kernel must be tested explicitly before
  claiming a trainable 120B arm. The cache benchmark proves forward support
  only.
- `teacher_forced` and `router_aligned` currently replace the MXFP4 experts
  with dequantized bf16 experts because their Python routing intervention
  needs the ordinary three-argument expert interface. That is useful for
  routing experiments, but it roughly doubles the expert-weight footprint and
  is unlikely to fit this 120B setup on two 80-GB cards once activations and
  optimizer state are included.

The current LoRA target list covers ordinary attention and MLP projection
leaves (`q/k/v/o`, `gate/up/down`). The MXFP4 GPT-OSS expert wrapper is not an
ordinary `torch.nn.Linear`, so this does not imply that the expert weights are
adapted. Any 120B result must record which modules actually received LoRA
parameters and whether the native expert path participated in backward.

The practical first training certification is therefore a one-batch,
no-update autograd probe with `dense_or_black_box`: verify finite student
gradients, confirm the frozen vocabulary locks, and record peak VRAM. Only
then compare layerwise losses or backpropagation speed. Routing-intervention
arms need a separate dequantized memory/parallelism budget and should not be
compared as if they were the same MXFP4 execution regime.

## Reproducibility notes

The cache command used `scripts/build_teacher_cache.py` with
`--device-map-auto --teacher-batch 64 --max-sequence-tokens 8192
--hidden-dtype bfloat16 --generation-responses .../responses_bs64.jsonl` and
the V5RS teacher-reference data configuration. The code-side fixes are in
the current history (`fix: keep mxfp4 teacher loads quantized` and
`fix: expose teacher cache storage dtype`). The external vLLM installation
was not modified.
