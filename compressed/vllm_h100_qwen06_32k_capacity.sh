#!/usr/bin/env bash
# Qwen3-0.6B 32k context capacity controls: one fresh engine per global batch
# so max_num_seqs equals the measured batch. Failures are recorded, not retried.
set -u -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/../venvs/vllm025/bin/python"; OUTROOT="$ROOT/runs/vllm_benchmark_h100"
LOG="$OUTROOT/qwen06_32k_capacity_$(date +%Y%m%d_%H%M%S).log"; mkdir -p "$OUTROOT"
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export HF_HOME="/fs/agustina/arivero/supercomplex/.cache/huggingface"
stamp() { date '+%F %T'; }
run_one() {
  local mode="$1" bs="$2" out="runs/vllm_benchmark_h100/Qwen3-0.6B_32k_${mode}_b${bs}_h100"
  if [ -f "$ROOT/$out/summary.json" ]; then printf '%s SKIP %s B%s\n' "$(stamp)" "$mode" "$bs" | tee -a "$LOG"; return 0; fi
  local graph=(); [ "$mode" = graphs ] && graph=(--use-cudagraphs)
  printf '%s START %s B%s\n' "$(stamp)" "$mode" "$bs" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/compressed/benchmark_vllm_generation.py" --model Qwen/Qwen3-0.6B --batch-sizes "$bs" --max-num-seqs "$bs" --gpu-memory-utilization 0.85 --max-model-len 32768 "${graph[@]}" --out "$out" >>"$LOG" 2>&1
  local rc=$?; printf '%s END %s B%s rc=%s\n' "$(stamp)" "$mode" "$bs" "$rc" | tee -a "$LOG"
}
for mode in graphs eager; do for bs in 1 2 4 8 16 32 64; do run_one "$mode" "$bs"; done; done
printf '%s QWEN06 32K CAPACITY COMPLETE\n' "$(stamp)" | tee -a "$LOG"
