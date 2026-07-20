#!/usr/bin/env bash
# Shared queue for Qwen3.5 0.8B/2B/4B/9B graph and eager baselines.
set -u -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../venvs/vllm025/bin/python"
OUTROOT="$ROOT/runs/vllm_benchmark_h100"
STAMP="$(date +%Y%m%d_%H%M%S)"; LOG="$OUTROOT/qwen35_shared_${STAMP}.log"
PENDING="$OUTROOT/.qwen35_shared_${STAMP}.tsv"; LOCK="$OUTROOT/.qwen35_shared_${STAMP}.lock"
mkdir -p "$OUTROOT"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export HF_HOME="/fs/agustina/arivero/supercomplex/.cache/huggingface"
stamp() { date '+%F %T'; }

for model in Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B; do
  printf 'Qwen/%s\t%s_vllm025_graph_full_h100\tgraphs\n' "$model" "$model"
done >"$PENDING"
for model in Qwen3.5-0.8B Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B; do
  printf 'Qwen/%s\t%s_vllm025_eager_full_h100\teager\n' "$model" "$model"
done >>"$PENDING"

next_job() {
  local item=""; exec 9>"$LOCK"; flock -x 9
  if [ -s "$PENDING" ]; then IFS= read -r item <"$PENDING"; sed -i '1d' "$PENDING"; fi
  flock -u 9; printf '%s\n' "$item"
}
run_one() {
  local gpu="$1" model="$2" tag="$3" mode="$4" out="runs/vllm_benchmark_h100/${3}"
  if [ -f "$ROOT/$out/summary.json" ]; then printf '%s SKIP GPU%s %s %s\n' "$(stamp)" "$gpu" "$mode" "$model" | tee -a "$LOG"; return 0; fi
  local graph=(); [ "$mode" = graphs ] && graph=(--use-cudagraphs)
  printf '%s START GPU%s %s %s\n' "$(stamp)" "$gpu" "$mode" "$model" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/defactorised/benchmark_vllm_generation.py" --model "$model" --batch-sizes 64 --gpu-memory-utilization 0.85 --max-model-len 4096 "${graph[@]}" --prompt-format native --out "$out" >>"$LOG" 2>&1
  local rc=$?; printf '%s END GPU%s %s rc=%s %s\n' "$(stamp)" "$gpu" "$mode" "$rc" "$model" | tee -a "$LOG"
}
worker() {
  local gpu="$1" item model tag mode
  while :; do item="$(next_job)"; [ -n "$item" ] || break; IFS=$'\t' read -r model tag mode <<<"$item"; run_one "$gpu" "$model" "$tag" "$mode"; done
}
worker 0 & p0=$!; worker 1 & p1=$!; wait "$p0" "$p1"
printf '%s QWEN35 SHARED QUEUE COMPLETE\n' "$(stamp)" | tee -a "$LOG"
