#!/usr/bin/env bash
# Chain the already-running eager shared queue into the remaining H100 work.
# No retries: each child script records failures and proceeds once.
set -u -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${1:?shared eager queue PID required}"
LOG="$ROOT/runs/vllm_benchmark_h100/rest_of_day_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
printf '%s START Qwen3.5 shared queue\n' "$(date '+%F %T')" | tee -a "$LOG"
bash "$ROOT/compressed/vllm_h100_qwen35_shared_queue.sh" >>"$LOG" 2>&1
printf '%s START Qwen3-0.6B 32k capacity controls\n' "$(date '+%F %T')" | tee -a "$LOG"
bash "$ROOT/compressed/vllm_h100_qwen06_32k_capacity.sh" >>"$LOG" 2>&1
printf '%s REST-OF-DAY QUEUE COMPLETE\n' "$(date '+%F %T')" | tee -a "$LOG"
