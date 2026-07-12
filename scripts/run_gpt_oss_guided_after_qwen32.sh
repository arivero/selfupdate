#!/usr/bin/env bash
# One-time dependency hand-off: do not interrupt the active Qwen3-32B PP run.
# Once it exits, use the freed pair for the GPT-OSS memory-framing control,
# then resume the pre-existing overnight queue.
set -u -o pipefail

qwen_pid="${1:?Qwen benchmark PID required}"
queue_pid="${2:?queue PID required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log="$root/runs/vllm_benchmark_h100/gpt-oss-120b_vllm025_pp2_guided_memory_h100.log"

tail --pid="$qwen_pid" -f /dev/null
cd "$root"
TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error \
TOKENIZERS_PARALLELISM=false HF_HOME=/fs/agustina/arivero/supercomplex/.cache/huggingface \
CUDA_VISIBLE_DEVICES=0,1 "$root/../venvs/vllm025/bin/python" \
  "$root/scripts/benchmark_vllm_generation.py" \
  --model openai/gpt-oss-120b --batch-sizes 64 --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --use-cudagraphs \
  --prompt-format gpt_oss_guided_memory \
  --out runs/vllm_benchmark_h100/gpt-oss-120b_vllm025_pp2_guided_memory_h100 \
  >>"$log" 2>&1
rc=$?
printf '%s guided-memory GPT-OSS rc=%s\n' "$(date '+%F %T')" "$rc" >>"$log"
kill -CONT "$queue_pid"
exit "$rc"
