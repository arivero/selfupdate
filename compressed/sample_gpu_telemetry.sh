#!/usr/bin/env bash
# Sample physical-card saturation for one named trainer. The sampler waits for
# the exact compressed/train.py command and exits when that trainer does.
set -euo pipefail

PATTERN=""
OUT=""
INTERVAL="1"
WAIT_SECONDS="1800"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --pattern) PATTERN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="$2"; shift 2 ;;
    *) echo "unsupported argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$PATTERN" && -n "$OUT" ]] || {
  echo "usage: $0 --pattern RUN_NAME --out FILE [--interval SEC]" >&2
  exit 2
}

mkdir -p "$(dirname "$OUT")"
printf 'timestamp,index,uuid,utilization_gpu_pct,power_w,temperature_c,memory_used_mib,memory_total_mib,clocks_sm_mhz\n' > "$OUT"

deadline=$((SECONDS + WAIT_SECONDS))
trainer_running() {
  local cmdline arg has_entry has_pattern
  for cmdline in /proc/[0-9]*/cmdline; do
    [[ -r "$cmdline" ]] || continue
    has_entry=0
    has_pattern=0
    while IFS= read -r -d '' arg; do
      [[ "$arg" == */compressed/train.py ]] && has_entry=1
      [[ "$arg" == *"$PATTERN"* ]] && has_pattern=1
    done < "$cmdline"
    (( has_entry && has_pattern )) && return 0
  done
  return 1
}

while ! trainer_running; do
  (( SECONDS < deadline )) || {
    echo "trainer did not appear within ${WAIT_SECONDS}s: ${PATTERN}" >&2
    exit 1
  }
  sleep 1
done

while trainer_running; do
  stamp="$(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,power.draw,temperature.gpu,memory.used,memory.total,clocks.sm \
    --format=csv,noheader,nounits | sed "s/^/${stamp},/" >> "$OUT"
  sleep "$INTERVAL"
done
