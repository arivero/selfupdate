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

**Scoring rule: initialization is discounted.** `tokens_per_second` =
`gen_tokens / generate_seconds`; engine construction, weight load, warmup,
JIT and CUDA-graph capture are reported separately as `load_seconds` and do
not enter the competition. (They matter operationally — vLLM CPU paid 118 s
of init for 16.7 s of generation — but the question here is steady-state
generation speed.)

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

H100 80GB, same 64 v5 prompts, initialization discounted:

| engine | tok/s | generate | prefill | decode | init/load |
|---|---|---|---|---|---|
| vLLM 0.25.0 standard (graphs), GPU1 | **7079** | 0.66 s | — | — | 141.9 s |
| torch eager b64 no-cuDNN-SDPA, GPU0 | 758 | 6.19 s | 0.98 s | 5.17 s | 3.6 s |
| torch eager b64, first round | 38.96 | 122.4 s | 11.0 s | 111.0 s | 4.7 s |

Output agreement torch-vs-vLLM: 43/64 token-identical, mean word overlap
0.910 (greedy divergence cascades on numerics differences; expected).

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
footgun. The fix (`torch.backends.cuda.enable_cudnn_sdp(False)`, one line)
took the loop from 39 to 758 tok/s.

## Round 2: StaticCache + torch.compile (generate_torch_v2.py)

Attempt to buy back the bookkeeping losses with HF-level machinery:
`StaticCache` (in-place KV writes), a preallocated full-width mask updated
in place, power-of-two retirement compaction, and optional `torch.compile`
with globally fixed shapes (compile/warmup in the discounted load phase).

Every variant LOST to round 1's best eager loop:

| | v1 eager (DynamicCache) | v2 StaticCache | v2 + compile | vLLM |
|---|---|---|---|---|
| GPU H100 | **758** | 606 | 392 | 7079 |
| CPU 32c | **34.5** | 28.0 | 20.6 | 285.9 |

The mechanism is symmetric and instructive: `DynamicCache` pays a full
cache copy per token but attends only over real tokens; `StaticCache`
writes in place but attends over the full preallocated width (~1800 here,
median true K ~400) every step — and the second trade is worse on both
devices. `torch.compile` on top cannot help because the graph it fuses is
the wide-attention graph (and fixed shapes force all-batch padding, which
also doubles prefill: 18.8 -> 34.3 s on CPU). A first compile attempt with
toy-shaped warmup recompiled inside the timed region (37.7 tok/s on H100)
— warm up with the exact production shapes or don't bother.

**PagedAttention is precisely the data structure that pays neither cost**
(block-sparse KV: no copy, no dead width). That is the part of vLLM you
cannot reproduce from HF-level building blocks, and after removing every
removable overhead it still accounts for essentially all of the remaining
~8-9x on both devices.

## Round 3: preallocate once, slice the live prefix (generate_torch_v3.py)

Round 2's dilemma — copy-per-token (Dynamic) vs full dead width (Static) —
is a false choice: a custom `~30`-line cache layer preallocates the KV
buffer once per batch, writes in place at each step, and returns a
**sliced view** of only the live prefix. No copy, no dead width. This is
PagedAttention degenerated to one contiguous page per sequence — the part
of the trick reachable from HF-level building blocks (`DynamicLayer`
subclass + a custom `layer_class_to_replicate`).

| | v1 eager | v2 Static(+compile) | v3 prealloc+slice | vLLM |
|---|---|---|---|---|
| GPU H100 | 758 | 606 (392) | **829.9** (+9.5% vs v1) | 7079 |
| CPU 32c | 34.5 | 28.0 (20.6) | **45.5** (+32% vs v1) | 285.9 |

Outputs are 64/64 token-identical to v1 on both devices (same math, only
cache bookkeeping changed) — the speedup is free, not a numerics trade.
CPU gains more proportionally because the mask-`cat` and cache-`cat` were
a bigger fraction of its per-step cost; GPU was already less bottlenecked
on bookkeeping after the round-1 cuDNN fix, so v3's win there is smaller.

This closes the reachable gap: what v3 buys back is the entire
"unnecessary" bookkeeping tax identified in rounds 1-2. What's left
(vLLM still ~8.5x on GPU, ~6.3x on CPU) is continuous batching, real
block-sparse PagedAttention (paging across *different* sequences' KV,
not just one contiguous region per sequence), and — GPU only — CUDA
graphs. None of those are expressible without rebuilding the scheduler.

## Verdict

Can a plain torch generator match vLLM's generation speed, discounting
initialization? **No, on both fronts, and the reasons are structural:**

- **CPU:** vLLM CPU wins ~8x (286 vs 34.5 tok/s) even though the raw
  decode-step math on the same cores is at parity (317 tok/s microbench).
  The loss is `DynamicCache`'s copy-per-step and the padded-mask math path —
  and round 2 shows the HF-level alternatives (StaticCache, compile) are
  worse, not better: fixing this properly means paged KV + mask-free
  kernels, i.e. vLLM's core.
- **GPU:** standard vLLM wins ~9.3x (7079 vs 758 tok/s). Continuous
  batching, PagedAttention, CUDA graphs and fused kernels each contribute;
  none is available to an eager HF forward loop.

The operational counterweight is initialization: vLLM paid 142 s (GPU,
graphs+warmup) / 118 s (CPU) before the first token, the torch loop ~4 s.
For a one-shot job of this sample's size the torch loop wins end-to-end
(10 s vs 143 s); the crossover is around ~25 batches of 64 prompts. For
campaign-scale generation (thousands of prompts per cache build), vLLM's
init amortizes to nothing — which is exactly why the v5 pipeline's
vLLM-generate + torch-teacher-forward split is the right architecture.
