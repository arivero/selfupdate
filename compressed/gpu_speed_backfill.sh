#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

GPUS="${GPUS:-0 1 3}"
THRESHOLD="${THRESHOLD:-80}"
MIN_FREE_MB="${MIN_FREE_MB:-6000}"
INTERVAL="${INTERVAL:-30}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BATCHES="${BATCHES:-16}"
SEQ_LEN="${SEQ_LEN:-512}"
ANSWER_LEN="${ANSWER_LEN:-160}"
ITERS="${ITERS:-600}"
WARMUP="${WARMUP:-5}"
LOCK_DIR="${LOCK_DIR:-runs/.backfill}"
OUT_DIR="${OUT_DIR:-runs/speed_checks}"

mkdir -p "$LOCK_DIR" "$OUT_DIR"

log() { echo "[$(date '+%F %T')] backfill: $*"; }

gpu_field() {
  local gpu="$1" field="$2"
  nvidia-smi -i "$gpu" --query-gpu="$field" --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

while true; do
  for gpu in $GPUS; do
    lock="$LOCK_DIR/gpu${gpu}.pid"
    if [ -s "$lock" ]; then
      pid="$(cat "$lock" 2>/dev/null || true)"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      rm -f "$lock"
    fi

    util="$(gpu_field "$gpu" utilization.gpu)"
    free="$(gpu_field "$gpu" memory.free)"
    if [ "${util:-0}" -lt "$THRESHOLD" ] && [ "${free:-0}" -ge "$MIN_FREE_MB" ]; then
      stamp="$(date '+%Y%m%d_%H%M%S')"
      name="backfill_g${gpu}_${stamp}"
      if ! (set -o noclobber; echo "$$" > "$lock") 2>/dev/null; then
        continue
      fi
      log "GPU[$gpu] util=${util}% free=${free}MB -> $name"
      (
        echo "$BASHPID" > "$lock"
        CUDA_VISIBLE_DEVICES="$gpu" compressed/run_speed_check_monitored.sh "$name" \
          --model "$MODEL" \
          --batches "$BATCHES" \
          --variants gpu \
          --seq-len "$SEQ_LEN" \
          --answer-len "$ANSWER_LEN" \
          --iters "$ITERS" \
          --warmup "$WARMUP" \
          --no-optimizer
        if read -r lock_pid < "$lock" 2>/dev/null && [ "$lock_pid" = "$BASHPID" ]; then
          rm -f "$lock"
        fi
      ) >> "$OUT_DIR/gpu${gpu}_backfill.log" 2>&1 &
    fi
  done
  sleep "$INTERVAL"
done
