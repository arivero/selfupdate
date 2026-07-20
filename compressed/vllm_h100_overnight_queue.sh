#!/usr/bin/env bash
# Sequential H100 vLLM ladder.  One PP=2 engine owns physical GPUs 0 and 1;
# after it drains, the smaller one-card models run in pairs on those same GPUs.
# Each benchmark writes decoded answers plus CPU-scored recitation metrics.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../venvs/vllm025/bin/python"
OUTROOT="$ROOT/runs/vllm_benchmark_h100"
LOG="$OUTROOT/overnight_queue_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTROOT"

export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/fs/agustina/arivero/supercomplex/.cache/huggingface"

stamp() { date '+%F %T'; }

run_pp() {
  local model="$1" tag="$2" format="${3:-native}"
  local out="runs/vllm_benchmark_h100/${tag}"
  if [ -f "$ROOT/$out/summary.json" ]; then
    printf '%s SKIP existing PP2 result %s\n' "$(stamp)" "$model" | tee -a "$LOG"
    return 0
  fi
  printf '%s START PP2 graphs %s\n' "$(stamp)" "$model" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0,1 "$PY" "$ROOT/compressed/benchmark_vllm_generation.py" \
    --model "$model" --batch-sizes 64 --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --use-cudagraphs \
    --prompt-format "$format" --out "$out" >>"$LOG" 2>&1
  local rc=$?
  printf '%s END PP2 rc=%s %s\n' "$(stamp)" "$rc" "$model" | tee -a "$LOG"
}

run_single() {
  local gpu="$1" model="$2" tag="$3" format="${4:-native}"
  local out="runs/vllm_benchmark_h100/${tag}"
  if [ -f "$ROOT/$out/summary.json" ]; then
    printf '%s SKIP existing GPU%s result %s\n' "$(stamp)" "$gpu" "$model" | tee -a "$LOG"
    return 0
  fi
  printf '%s START GPU%s graphs %s\n' "$(stamp)" "$gpu" "$model" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/compressed/benchmark_vllm_generation.py" \
    --model "$model" --batch-sizes 64 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --use-cudagraphs --prompt-format "$format" \
    --out "$out" >>"$LOG" 2>&1
  local rc=$?
  printf '%s END GPU%s rc=%s %s\n' "$(stamp)" "$gpu" "$rc" "$model" | tee -a "$LOG"
  return 0
}

# Eager controls are intentionally deferred until every graph-mode single-card
# run has had its turn.  They use a distinct output identity, so a completed
# graph benchmark can never be mistaken for its no-CUDA-graphs control.
run_single_eager() {
  local gpu="$1" model="$2" tag="$3" format="${4:-native}"
  local out="runs/vllm_benchmark_h100/${tag}"
  if [ -f "$ROOT/$out/summary.json" ]; then
    printf '%s SKIP existing GPU%s eager result %s\n' "$(stamp)" "$gpu" "$model" | tee -a "$LOG"
    return 0
  fi
  printf '%s START GPU%s eager %s\n' "$(stamp)" "$gpu" "$model" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/compressed/benchmark_vllm_generation.py" \
    --model "$model" --batch-sizes 64 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --prompt-format "$format" \
    --out "$out" >>"$LOG" 2>&1
  local rc=$?
  printf '%s END GPU%s eager rc=%s %s\n' "$(stamp)" "$gpu" "$rc" "$model" | tee -a "$LOG"
  return 0
}

# Start with models whose normal bf16 (or supported compressed) form calls for
# both cards.  A failed architecture/config is recorded but must not block the
# rest of the night queue.
# The two completed memory-framed rows are skipped automatically on a restart.
# The guided-memory GPT-OSS control remains a separate identity, but is part
# of this one serial queue rather than an external hand-off process.
run_pp "openai/gpt-oss-120b" "gpt-oss-120b_vllm025_pp2_graph_full_h100" gpt_oss_harmony
run_pp "Qwen/Qwen3-32B" "Qwen3-32B_vllm025_pp2_graph_full_h100"
run_pp "openai/gpt-oss-120b" "gpt-oss-120b_vllm025_pp2_guided_memory_h100" gpt_oss_guided_memory
run_pp "meta-llama/Llama-3.3-70B-Instruct" "Llama-3.3-70B_vllm025_pp2_graph_full_h100"
run_pp "google/gemma-4-31B-it" "gemma-4-31B-it_vllm025_pp2_graph_full_h100"
run_pp "Qwen/Qwen3.6-27B" "Qwen3.6-27B_vllm025_pp2_graph_full_h100"
run_pp "google/gemma-4-26B-A4B-it" "gemma-4-26B-A4B-it_vllm025_pp2_graph_full_h100"
run_pp "Qwen/Qwen3.6-35B-A3B" "Qwen3.6-35B-A3B_vllm025_pp2_graph_full_h100"
run_pp "BSC-LT/ALIA-40b-fc-2606" "ALIA-40b-fc-2606_vllm025_pp2_graph_full_h100"

# If the PP ladder completes before morning, reclaim throughput with two
# independent one-card engines.  `wait` keeps each pair bounded to DEV0/DEV1.
run_single 0 "microsoft/Phi-4" "Phi-4_vllm025_graph_full_h100" &
pid0=$!
run_single 1 "openai/gpt-oss-20b" "gpt-oss-20b_vllm025_graph_full_h100" gpt_oss_harmony &
pid1=$!
wait "$pid0" "$pid1"
run_single 0 "google/gemma-3-12b-it" "gemma-3-12b-it_vllm025_graph_full_h100" &
pid0=$!
run_single 1 "nvidia/NVIDIA-Nemotron-Nano-9B-v2" "NVIDIA-Nemotron-Nano-9B-v2_vllm025_graph_full_h100" &
pid1=$!
wait "$pid0" "$pid1"
run_single 0 "meta-llama/Meta-Llama-3.1-8B-Instruct" "Llama-3.1-8B-Instruct_vllm025_graph_full_h100" &
pid0=$!
run_single 1 "mistralai/Mistral-7B-Instruct-v0.1" "Mistral-7B-Instruct-v0.1_vllm025_graph_full_h100" &
pid1=$!
wait "$pid0" "$pid1"

# Paired eager controls for every one-card graph candidate.  The four smaller
# Qwen3 controls already exist under their own eager identities; Qwen3-14B
# and every newly queued one-card model are added here.
run_single_eager 0 "Qwen/Qwen3-14B" "Qwen3-14B_vllm025_eager_full_h100" &
pid0=$!
run_single_eager 1 "microsoft/Phi-4" "Phi-4_vllm025_eager_full_h100" &
pid1=$!
wait "$pid0" "$pid1"
run_single_eager 0 "openai/gpt-oss-20b" "gpt-oss-20b_vllm025_eager_full_h100" gpt_oss_harmony &
pid0=$!
run_single_eager 1 "google/gemma-3-12b-it" "gemma-3-12b-it_vllm025_eager_full_h100" &
pid1=$!
wait "$pid0" "$pid1"
run_single_eager 0 "nvidia/NVIDIA-Nemotron-Nano-9B-v2" "NVIDIA-Nemotron-Nano-9B-v2_vllm025_eager_full_h100" &
pid0=$!
run_single_eager 1 "meta-llama/Meta-Llama-3.1-8B-Instruct" "Llama-3.1-8B-Instruct_vllm025_eager_full_h100" &
pid1=$!
wait "$pid0" "$pid1"
run_single_eager 0 "mistralai/Mistral-7B-Instruct-v0.1" "Mistral-7B-Instruct-v0.1_vllm025_eager_full_h100"

printf '%s QUEUE COMPLETE\n' "$(stamp)" | tee -a "$LOG"
