#!/usr/bin/env bash
# Sample physical-card saturation for one named trainer. The sampler waits for
# the exact scripts/train.py command and exits when that trainer does.
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
while ! pgrep -f "scripts/train.py.*${PATTERN}" >/dev/null; do
  (( SECONDS < deadline )) || {
    echo "trainer did not appear within ${WAIT_SECONDS}s: ${PATTERN}" >&2
    exit 1
  }
  sleep 1
done

while pgrep -f "scripts/train.py.*${PATTERN}" >/dev/null; do
  stamp="$(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,power.draw,temperature.gpu,memory.used,memory.total,clocks.sm \
    --format=csv,noheader,nounits | sed "s/^/${stamp},/" >> "$OUT"
  sleep "$INTERVAL"
done
