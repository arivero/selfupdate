# demos/ — pure-torch CPU generator vs vLLM CPU

Question (owner, 2026-07-14): the v5 cache pipeline now uses vLLM for batched
answer generation plus a single torch teacher-forced forward. But if a future
stage needs *generation* inside the torch stack, can a plain PyTorch decode
loop — no CUDA graphs, in fact no GPU at all — match vLLM's speed? This demo
tests the CPU-only, small-model corner of that question on the real V5
workload.

## Contenders

| | engine | environment |
|---|---|---|
| `generate_torch_cpu.py` | plain transformers + hand decode loop | repo `.venv`, `CUDA_VISIBLE_DEVICES=` |
| `generate_vllm_cpu.py` | vLLM 0.25.0 CPU backend | official CPU container (glibc 2.28 on Rocky 8.6 rejects the `manylinux_2_34` `+cpu` wheel, so the docker image runs under Singularity; see `run_vllm_cpu.sh`) |

Both consume the identical prompt file written by `build_prompt_sample.py`:
64 evenly-spaced records from `data/combined/examples_v5rs_window.jsonl`,
built by the same `ContextMasker`/budget/stop-token path as
`scripts/benchmark_vllm_generation.py` (native format, greedy,
per-record V5 budgets). Prompt lengths 135–1052 tokens, budgets 104–746.

## What the torch loop does to be fast

1. **Length-sorted static batches** — minimal left-padding waste.
2. **KV-cached prefill + one forward per decode step** for the whole batch;
   Python cost is per step, not per token-per-sequence.
3. **Retirement compaction** — when ≥25% of rows have hit their stop token or
   budget, finished rows are dropped from the batch and the KV cache
   (`DynamicCache.batch_select_indices`). This is the offline analogue of
   vLLM's continuous batching and is what stops tail stragglers from paying
   full-batch cost.
4. **bf16 + SDPA on AMX** — the Xeon 8462Y+ (Sapphire Rapids) has `amx_tile`
   and `avx512_bf16`; oneDNN routes bf16 matmuls through the tile units.

## Reproduce

```bash
# 1. prompt sample (repo venv; needs selfupdate imports)
.venv/bin/python demos/build_prompt_sample.py --model Qwen/Qwen3-0.6B --limit 64

# 2. torch contender (no GPU touched)
CUDA_VISIBLE_DEVICES= .venv/bin/python demos/generate_torch_cpu.py \
    --prompts demos/out/prompts_qwen3-0.6b_n64.jsonl \
    --batch-size 32 --threads 32 --out demos/out/torch_cpu_b32

# 3. vLLM CPU baseline (pull once: singularity pull docker://public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:v0.25.0)
demos/run_vllm_cpu.sh demos/out/prompts_qwen3-0.6b_n64.jsonl demos/out/vllm_cpu 32

# 4. compare speed and output agreement
python3 demos/compare_results.py demos/out/torch_cpu_b32 demos/out/vllm_cpu
```

Runs are sequential on an otherwise idle-CPU node so the two engines never
contend for cores.

## Results — demo 1: CPU (2026-07-14, Xeon 8462Y+, 32 cores/engine)

| engine | tok/s | generate | prefill | decode | notes |
|---|---|---|---|---|---|
| vLLM 0.25.0 CPU | **285.9** | 16.7 s | — | — | +118 s engine init/warmup |
| torch b32, cores 0-31 | 34.5 | 137.3 s | 18.2 s | 119.1 s | clean, pinned |
| torch b32, unpinned | 29.0 | 163.5 s | — | — | threads drift across sockets |
| torch b64, one batch | 6.9 | 681.7 s | — | — | everyone pads to 1052 tokens |
| torch decode-only microbench | 317 (per step) | — | — | 101 ms/step | B=32, K≈260, pinned |

Both engines produce essentially the same greedy answers (4740 vs 4779
tokens; the small delta is numerics-induced early/late stop-token timing).

**Verdict so far: vLLM CPU wins by ~8x on wall clock.** The gap is NOT the
per-step math — the microbenchmark shows the same 32 cores sustain 317
tok/s of batched decode compute, above vLLM's 286 — it is cache and mask
management:

1. `DynamicCache` does `torch.cat` per layer per step: the entire multi-GB
   KV cache is reallocated and copied every token. vLLM's paged KV cache
   (and HF's `StaticCache`) exists precisely to avoid this.
2. Left-padding needs an explicit attention mask, which forces SDPA off the
   fused flash-CPU kernel onto the materializing math path — most visible in
   prefill (18 s for ~16k prompt tokens).
3. Static batches pay for stragglers; retirement compaction recovers only
   part of what continuous batching gets for free.

Caveats measured the hard way (kept because they are the actual lesson):
- **Pin your cores.** Unpinned threads drift across the two sockets (3.5x
  slower steps); pinning into a range shared with someone else's job pinned
  to cores 32-39 was worse still (0.83 s/step: OpenMP static scheduling runs
  at the pace of the most-contended thread). Check `taskset -pc` of your
  neighbours before choosing a range.
- **bf16 beats fp32 by 3.5x** on this AMX machine (101 vs 352 ms/step).

## Results — demo 2: GPU race (torch eager GPU0 vs standard vLLM GPU1)

First round, H100 80GB, same 64 v5 prompts:

| engine | tok/s | generate | prefill | decode |
|---|---|---|---|---|
| torch eager b64, GPU0 | 38.96 | 122.4 s | 11.0 s | 111.0 s |
| vLLM 0.25.0 standard, GPU1 | (running) | | | |

**39 tok/s on an H100 — barely above the CPU number — and the profiler
explains it.** One decode step shows 515 ms of CPU time against 12.8 ms of
CUDA time; 347 ms of it is `aten::_cudnn_attention_forward` **self-CPU**
(12.5 ms × 28 layers). torch 2.11's SDPA dispatches to the cuDNN attention
backend, whose CPU-side plan cache misses at every step because the KV
length grows by one token each time — the H100 idles while cuDNN's frontend
rebuilds execution plans. Neither host syncs nor mask construction matter
(measured: removing them changes nothing).

This is a beautiful negative result for eager HF-on-GPU generation: the
bottleneck is not kernels, batching, or graphs — it is a backend-selection
footgun. Fix attempt in the next commit: exclude the cuDNN SDPA backend.
