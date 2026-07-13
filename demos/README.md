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

## Results

(to be filled by the first complete comparison run)
