#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-runs/gpu_util.csv}"
INTERVAL="${INTERVAL:-30}"
GPUS="${GPUS:-0 1 3}"

mkdir -p "$(dirname "$OUT")"
if [ ! -s "$OUT" ]; then
  printf 'timestamp,gpu_index,memory_used_mb,memory_total_mb,gpu_util_pct,mem_util_pct,power_w\n' > "$OUT"
fi

while true; do
  ts="$(date -Is)"
  for gpu in $GPUS; do
    line="$(nvidia-smi -i "$gpu" \
      --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
      --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    printf '%s,%s,%s\n' "$ts" "$gpu" "$line" >> "$OUT"
  done
  sleep "$INTERVAL"
done
