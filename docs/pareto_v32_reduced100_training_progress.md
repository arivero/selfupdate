# Pipeline v3.2 exact-100% reduced-epoch campaign

Started: 2026-07-16. Model: Qwen3.5-4B. Dataset family: v5. Training
pipeline: v3.2. Hosts agpul04 and agpul05 carry one run per L40S; agpul06 is
reserved exclusively for pipeline-v3.4 development and is outside this
campaign.

## Scientific question

The first v3.2 screen mixed two possible limits: imperfect teacher answers and
the capacity/geometry of rank-16 LoRA. This campaign removes the first source
of noise and directly probes the second:

1. The reduced epoch contains only examples whose uncensored original-RAG
   teacher answer has task-aware score exactly 1.0. Next/previous examples use
   reference-word LCS accuracy; cloze examples use deleted-word containment.
   Extra explanatory text is allowed, so “100%” means complete scored content,
   not exact string equality.
2. Four recipes are each run once with rank-16 LoRA and once with full block
   weights. Pairing holds dataset, censorship, loss, learning rate, B=256,
   K=16, seed, and immediate-SGD schedule fixed. LoRA tests the established
   low-memory path; full-FT tests whether adapter capacity caused the weak or
   unstable recall trajectories.

The four recipes are the four highest-recall Qwen3.5-4B arms in the first
v3.2 screen: flow-mask sampled-vocabulary cosine-256, intact Huber control,
flow-mask hidden cosine, and flow-mask Huber, all at learning rate 1e-6.

## Reduced dataset

Source dataset:
`data/combined/examples_v5rs_window.jsonl` (2,071 examples, SHA-256
`575b9dea35e0179dcdf7a513416e640db899c9bf9584236088f2921cce7a7042`).
Teacher responses:
`runs/vllm_benchmark_l40s/qwen35_4b_fixed4096_exactids_agpul06/responses_bs64.jsonl`.
Materialized subset:
`data/combined/examples_v5rs_window_qwen35_4b_score100.jsonl`.

| Quantity | Value |
|---|---:|
| Selected examples | 1,772 / 2,071 (85.5625%) |
| Teacher answer tokens retained | 76,090 |
| Next examples | 1,354 |
| Previous examples | 216 |
| Cloze examples | 202 |
| Machado examples | 1,290 |
| Quijote examples | 482 |

The machine-readable provenance and exact hashes are in
`data/combined/examples_v5rs_window_qwen35_4b_score100.manifest.json`; the
deterministic builder is `scripts/build_score_filtered_dataset.py`.

## Parameterization contract

Both variants remain strict block-local forward distillation. The embedding,
final norm, and LM head/unembedding are frozen and fingerprint-checked before
checkpoint publication. Neither variant trains final logits or uses
CE-eval-loss/KL-eval-loss for backward.

- LoRA: rank 16, alpha 32, dropout 0; bf16 frozen base plus trainable adapters.
- Full-FT: every decoder-block weight is trainable; the runtime keeps those
  weights and executes their block walk in fp32 so 1e-6 immediate-SGD writes
  cannot disappear through bf16 rounding and Qwen3.5 recurrent/KV state has
  one stable dtype. Only one block's fp32 gradient is live at a time. Full-FT
  uses eight-user activation shards and 16-token prefill chunks to bound the
  larger fp32 KV/activation footprint. The vocabulary machinery remains
  frozen.

Because the LoRA path uses bf16 autocast while full-FT executes in fp32, the
parameterization comparison also includes this necessary precision and
execution-shard difference. Reports must state it rather than claiming a
precision-identical ablation.

## Placement

| Host/GPU | Recipe | Parameterization |
|---|---|---|
| agpul04/0 | flow sampled-vocabulary cosine-256 | LoRA |
| agpul04/1 | flow sampled-vocabulary cosine-256 | full-FT |
| agpul04/2 | intact Huber | LoRA |
| agpul04/3 | intact Huber | full-FT |
| agpul05/0 | flow hidden cosine | LoRA |
| agpul05/1 | flow hidden cosine | full-FT |
| agpul05/2 | flow Huber | LoRA |
| agpul05/3 | flow Huber | full-FT |

The configs use `epochs: 1000000` as an operationally open-ended ceiling,
not as a scientific claim. One epoch is one complete traversal of the 1,772
selected examples. The runs are expected to be stopped manually around
midnight.

## Stop and checkpoint contract

SIGTERM/SIGINT are cooperative. The trainer completes the current B=256
cohort, records the partial-epoch coverage without presenting it as a complete
CE/KL evaluation, runs the locality/frozen-vocabulary gate, atomically
publishes `runs/<run>/checkpoint`, records `kind=done` with
`graceful_stop=true`, and then generates the individual Markdown/PDF report.
A second force-kill can still defeat this contract and must not be used while
checkpoint publication is in progress.

Use `scripts/stop_v32r_score100_campaign.sh`; it signals only the eight Python
trainer processes, leaving their launcher shells alive to wait for checkpoint
completion and generate reports.

## Live ledger

Launch, cache-build, first-cohort, throughput, stop, checkpoint, and report
times are appended here as they occur.

| Time (CEST) | Event | Evidence |
|---|---|---|
| 19:28-19:34 | Initial full-FT smoke redirected away from agpul06 | The agpul06 cache-builder chain was stopped before GPU training after the host was reserved for v3.4. GPU0 returned to 1 MiB and no reduced-v3.2 process remained. |
| 19:34 | Reduced-cache build started on agpul04/GPU1 | Atomic node-local lease acquired for cache identity `e64dd76e918a1435`; this launch continues into the full-FT signal/checkpoint smoke. |
| 19:34 | Reduced-cache build started on agpul05/GPU0 | Atomic node-local lease acquired independently for the same identity `e64dd76e918a1435`; cache-only warm-up, no training. |
| 19:37-19:40 | Both reduced caches completed | Each node processed 1,772 examples × 32 layers in approximately 194.5 s and atomically published identity `e64dd76e918a1435`; no cache warning or error. |
| 19:40 | First full-FT smoke exposed mixed KV-cache dtype | Before the first cohort, fp32 master weights caused a full-attention `StaticCache` fragment to initialize as fp32 while bf16 autocast supplied later KV, failing loudly in `index_copy_`. No training or checkpoint was claimed. The runtime now pre-initializes full-attention cache layers in the bf16 execution dtype while retaining model-authoritative lazy state for linear-attention layers; config audit passed before retry. |
| 19:45 | Forced-bf16 cache retry rejected | The complementary mismatch appeared: another authoritative Qwen3.5 path supplied fp32 KV to the forced-bf16 cache. This proved that mixed fp32-master/bf16-autocast full-FT does not provide one stable recurrent-state dtype. Full-FT now executes the block walk and KV state in fp32 with eight-user activation shards and 16-token prefill chunks; logical B256/K16 and update sums are unchanged. |
| 19:52 | Pure-fp32 retry reached backward and exposed an inference-cache mutation | The memory and dtype guards passed, epoch-zero evaluation completed, and the first real block-local backward entered FLA. Transformers then overwrote Qwen3.5's saved predecessor recurrent state with `copy_` before backward consumed it, raising an autograd version error on the `[8,32,128,128]` state. No update or checkpoint was claimed. The training cache now publishes detached successor recurrent/conv state out of place; a loss-free masked dummy keeps final one-token tails off the inference-only fused mutation path. |
| 20:06-20:10 | Full-FT smoke retry 3 crossed the recurrent-state failure, then rejected an unsafe tail workaround | Cohort 0 completed 256 answers, 13,133 token events, and 160 physical block writes at 40.5 GiB peak VRAM; this proves the out-of-place recurrent cache crossed the former first-backward failure. Cohort 1 then encountered `index_copy_(): index out of bounds`: the masked dummy used to keep a final q=1 tile on the training chunk kernel had no allocated static-KV slot. No checkpoint was claimed. The cache now reserves one explicitly masked sentinel slot and assigns it the successor position; it contributes no target, loss, gradient cell, or future history. |
| 20:11-20:17 | Full-FT retry 4 passed the complete signal/checkpoint gate | Three coherent cohorts completed before SIGTERM was observed: 768 answers, 56,370 token events, and 704 physical block writes. Peak VRAM was approximately 40.6 GiB. Locality certification passed on 16 items with local gradient norm 7.2424, cross-block leak gradient 0, frozen-vocabulary gradient 0, and signal present in every block. The trainer recorded `graceful_stop=true`, atomically published the checkpoint, generated the individual PDF report, and exited without a training error. |
