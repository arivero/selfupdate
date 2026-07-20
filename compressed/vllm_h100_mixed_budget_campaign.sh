#!/usr/bin/env bash
# Full V5RS benchmark of the mixed-per-request-budget driver introduced at
# 1f28029. DEV0/DEV1 are the only permitted devices. One-card jobs run in
# pairs; genuine capacity models use TP2 or the historically proven PP2 path.
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../venvs/vllm025/bin/python"
OUTROOT="$ROOT/runs/vllm_benchmark_h100/mixed_budget_campaign"
MAINLOG="$OUTROOT/campaign.log"
mkdir -p "$OUTROOT"

export HF_HOME="/fs/agustina/arivero/supercomplex/.cache/huggingface"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

stamp() { date '+%F %T'; }
slug() { printf '%s' "$1" | tr '/.' '__'; }

run_isolated() {
  # vLLM multiprocess initialization failures can return from the Python
  # leader while CUDA-corrupted workers survive as PPID-1 orphans.  Give each
  # engine a private process group, then clean only that group before a
  # placement fallback or the next model is allowed to start.
  setsid "$@" &
  local leader=$!
  local rc=0
  wait "$leader" || rc=$?
  kill -TERM -- "-$leader" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 -- "-$leader" 2>/dev/null || break
    sleep 0.2
  done
  kill -KILL -- "-$leader" 2>/dev/null || true
  return "$rc"
}

run_single() {
  local gpu="$1" model="$2" tag="$3" format="${4:-native}" util="${5:-0.85}"
  local out="runs/vllm_benchmark_h100/mixed_budget_campaign/$tag"
  local log="$OUTROOT/${tag}.log"
  if [[ -f "$ROOT/$out/summary.json" ]]; then
    printf '%s SKIP single GPU%s %s\n' "$(stamp)" "$gpu" "$model" >>"$MAINLOG"
    return 0
  fi
  printf '%s START single GPU%s %s\n' "$(stamp)" "$gpu" "$model" >>"$MAINLOG"
  run_isolated env CUDA_VISIBLE_DEVICES="$gpu" \
    "$PY" "$ROOT/compressed/benchmark_vllm_generation.py" \
    --model "$model" --batch-sizes 64 --max-num-seqs 64 \
    --gpu-memory-utilization "$util" --max-model-len 4096 \
    --use-cudagraphs --progress --prompt-format "$format" --out "$out" \
    >>"$log" 2>&1
  local rc=$?
  printf '%s END single GPU%s rc=%s %s\n' "$(stamp)" "$gpu" "$rc" "$model" \
    >>"$MAINLOG"
  return "$rc"
}

run_dual() {
  local model="$1" tag="$2" placement="$3" format="${4:-native}" util="${5:-0.95}"
  local out="runs/vllm_benchmark_h100/mixed_budget_campaign/$tag"
  local log="$OUTROOT/${tag}.log"
  if [[ -f "$ROOT/$out/summary.json" ]]; then
    printf '%s SKIP dual %s %s\n' "$(stamp)" "$placement" "$model" >>"$MAINLOG"
    return 0
  fi
  local parallel_args=()
  if [[ "$placement" == tp2 ]]; then
    parallel_args=(--tensor-parallel-size 2)
  else
    parallel_args=(--pipeline-parallel-size 2)
  fi
  printf '%s START dual %s %s\n' "$(stamp)" "$placement" "$model" >>"$MAINLOG"
  run_isolated env CUDA_VISIBLE_DEVICES=0,1 \
    "$PY" "$ROOT/compressed/benchmark_vllm_generation.py" \
    --model "$model" --batch-sizes 64 --max-num-seqs 64 \
    --gpu-memory-utilization "$util" --max-model-len 4096 \
    --use-cudagraphs --progress --prompt-format "$format" \
    "${parallel_args[@]}" --out "$out" >>"$log" 2>&1
  local rc=$?
  printf '%s END dual %s rc=%s %s\n' "$(stamp)" "$placement" "$rc" "$model" \
    >>"$MAINLOG"
  return "$rc"
}

run_pair() {
  run_single 0 "$1" "$2" "${3:-native}" "${4:-0.85}" & local p0=$!
  run_single 1 "$5" "$6" "${7:-native}" "${8:-0.85}" & local p1=$!
  wait "$p0"; local r0=$?
  wait "$p1"; local r1=$?
  return $((r0 != 0 || r1 != 0))
}

printf '%s CAMPAIGN START commit=%s\n' "$(stamp)" \
  "$(git -C "$ROOT" rev-parse HEAD)" >>"$MAINLOG"

# Models already known to initialize and fit a single H100 with this stack.
# Failures do not stop later pairs; each has an isolated log and partial JSONL.
run_pair \
  Qwen/Qwen3-0.6B qwen3_0p6b_mixed_b64_single \
  native 0.85 \
  Qwen/Qwen3.5-0.8B qwen35_0p8b_mixed_b64_single native 0.85 || true
run_pair \
  Qwen/Qwen3-1.7B qwen3_1p7b_mixed_b64_single \
  native 0.85 \
  Qwen/Qwen3.5-2B qwen35_2b_mixed_b64_single native 0.85 || true
run_pair \
  Qwen/Qwen3-4B qwen3_4b_mixed_b64_single \
  native 0.85 \
  Qwen/Qwen3.5-9B qwen35_9b_mixed_b64_single native 0.85 || true
run_pair \
  Qwen/Qwen3-8B qwen3_8b_mixed_b64_single \
  native 0.85 \
  meta-llama/Meta-Llama-3.1-8B-Instruct llama31_8b_mixed_b64_single native 0.85 || true
run_pair \
  Qwen/Qwen3-14B qwen3_14b_mixed_b64_single \
  native 0.85 \
  microsoft/Phi-4 phi4_mixed_b64_single native 0.85 || true
run_pair \
  openai/gpt-oss-20b gpt_oss_20b_mixed_b64_single \
  gpt_oss_harmony 0.85 \
  nvidia/NVIDIA-Nemotron-Nano-9B-v2 nemotron_nano_9b_mixed_b64_single native 0.85 || true

# Gemma-3-12B is deliberately omitted: it historically failed engine startup
# under this exact vLLM/Torch stack. Mistral is paired with the first large
# one-card capacity test so both devices remain productive.
run_pair \
  Qwen/Qwen3-32B qwen3_32b_mixed_b64_single \
  native 0.95 \
  mistralai/Mistral-7B-Instruct-v0.1 mistral_7b_mixed_b64_single native 0.85 || true
run_pair \
  Qwen/Qwen3.6-27B qwen36_27b_mixed_b64_single \
  native 0.95 \
  google/gemma-4-31B-it gemma4_31b_mixed_b64_single native 0.95 || true

# Capacity fallbacks/new dual-card results. TP2 is preferred for dense models;
# PP2 is retained for architectures already proven compatible in this tree.
if [[ ! -f "$OUTROOT/qwen3_32b_mixed_b64_single/summary.json" ]]; then
  run_dual Qwen/Qwen3-32B qwen3_32b_mixed_b64_tp2 tp2 || \
    run_dual Qwen/Qwen3-32B qwen3_32b_mixed_b64_pp2 pp2 || true
fi
if [[ ! -f "$OUTROOT/qwen36_27b_mixed_b64_single/summary.json" ]]; then
  run_dual Qwen/Qwen3.6-27B qwen36_27b_mixed_b64_tp2 tp2 || \
    run_dual Qwen/Qwen3.6-27B qwen36_27b_mixed_b64_pp2 pp2 || true
fi
if [[ ! -f "$OUTROOT/gemma4_31b_mixed_b64_single/summary.json" ]]; then
  run_dual google/gemma-4-31B-it gemma4_31b_mixed_b64_tp2 tp2 || \
    run_dual google/gemma-4-31B-it gemma4_31b_mixed_b64_pp2 pp2 || true
fi

# Known non-fits. Llama-70B loaded under PP2 but needed more KV margin, so try
# TP2 at 0.95 first. ALIA and GPT-OSS use their historically working PP2 path.
run_dual meta-llama/Llama-3.3-70B-Instruct llama33_70b_mixed_b64_tp2 tp2 native 0.95 || \
  run_dual meta-llama/Llama-3.3-70B-Instruct llama33_70b_mixed_b64_pp2 pp2 native 0.95 || true
run_dual BSC-LT/ALIA-40b-fc-2606 alia_40b_mixed_b64_pp2 pp2 native 0.95 || true
run_dual openai/gpt-oss-120b gpt_oss_120b_mixed_b64_pp2 pp2 gpt_oss_harmony 0.95 || true

printf '%s CAMPAIGN COMPLETE\n' "$(stamp)" >>"$MAINLOG"
