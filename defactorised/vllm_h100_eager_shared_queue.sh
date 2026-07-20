#!/usr/bin/env bash
# Shared one-card eager queue: the first free DEV0/DEV1 worker takes the next
# job. Each job appears once; failures are recorded and never retried.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../venvs/vllm025/bin/python"
OUTROOT="$ROOT/runs/vllm_benchmark_h100"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTROOT/eager_shared_${STAMP}.log"
PENDING="$OUTROOT/.eager_shared_${STAMP}.tsv"
LOCK="$OUTROOT/.eager_shared_${STAMP}.lock"
mkdir -p "$OUTROOT"

export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/fs/agustina/arivero/supercomplex/.cache/huggingface"

stamp() { date '+%F %T'; }

# The first two already have summaries and are skipped. They remain here so
# this runner is a complete, inspectable eager ledger. No retry loop exists.
printf '%s\n' \
  $'Qwen/Qwen3-14B\tQwen3-14B_vllm025_eager_full_h100\tnative' \
  $'microsoft/Phi-4\tPhi-4_vllm025_eager_full_h100\tnative' \
  $'openai/gpt-oss-20b\tgpt-oss-20b_vllm025_eager_full_h100\tgpt_oss_harmony' \
  $'google/gemma-3-12b-it\tgemma-3-12b-it_vllm025_eager_full_h100\tnative' \
  $'nvidia/NVIDIA-Nemotron-Nano-9B-v2\tNVIDIA-Nemotron-Nano-9B-v2_vllm025_eager_full_h100\tnative' \
  $'meta-llama/Meta-Llama-3.1-8B-Instruct\tLlama-3.1-8B-Instruct_vllm025_eager_full_h100\tnative' \
  $'mistralai/Mistral-7B-Instruct-v0.1\tMistral-7B-Instruct-v0.1_vllm025_eager_full_h100\tnative' \
  >"$PENDING"

next_job() {
  local item=""
  exec 9>"$LOCK"
  flock -x 9
  if [ -s "$PENDING" ]; then
    IFS= read -r item <"$PENDING"
    sed -i '1d' "$PENDING"
  fi
  flock -u 9
  printf '%s\n' "$item"
}

run_one() {
  local gpu="$1" model="$2" tag="$3" format="$4"
  local out="runs/vllm_benchmark_h100/${tag}"
  if [ -f "$ROOT/$out/summary.json" ]; then
    printf '%s SKIP GPU%s eager %s\n' "$(stamp)" "$gpu" "$model" | tee -a "$LOG"
    return 0
  fi
  printf '%s START GPU%s eager %s\n' "$(stamp)" "$gpu" "$model" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/defactorised/benchmark_vllm_generation.py" \
    --model "$model" --batch-sizes 64 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --prompt-format "$format" --out "$out" >>"$LOG" 2>&1
  local rc=$?
  printf '%s END GPU%s eager rc=%s %s\n' "$(stamp)" "$gpu" "$rc" "$model" | tee -a "$LOG"
}

worker() {
  local gpu="$1" item model tag format
  while :; do
    item="$(next_job)"
    [ -n "$item" ] || break
    IFS=$'\t' read -r model tag format <<<"$item"
    run_one "$gpu" "$model" "$tag" "$format"
  done
  printf '%s WORKER GPU%s COMPLETE\n' "$(stamp)" "$gpu" | tee -a "$LOG"
}

worker 0 &
p0=$!
worker 1 &
p1=$!
wait "$p0" "$p1"
printf '%s SHARED EAGER QUEUE COMPLETE\n' "$(stamp)" | tee -a "$LOG"
