#!/usr/bin/env bash
# Run an ad-hoc check pinned to the least-loaded GPU, self-aborting if that
# GPU's memory use crosses THRESHOLD_PCT — checks must never squeeze the
# scheduler's campaign jobs (their VRAM reservations are launch-time checks,
# not leases; an eval placed into a job's margin can OOM it hours later).
#
# Usage: [THRESHOLD_PCT=85] [POLL_S=5] scripts/vram_guard.sh <command...>
# Exit 75 (EX_TEMPFAIL) on a VRAM abort, else the command's own status.
set -u
THRESHOLD_PCT="${THRESHOLD_PCT:-85}"
POLL_S="${POLL_S:-5}"
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2 -n | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES="$gpu"
echo "vram_guard: GPU $gpu (least loaded), abort above ${THRESHOLD_PCT}%" >&2
"$@" &
pid=$!
trap 'kill "$pid" 2>/dev/null' EXIT INT TERM
while kill -0 "$pid" 2>/dev/null; do
    read -r used total < <(nvidia-smi -i "$gpu" \
        --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \
        | tr -d ',')
    if [ "$((100 * used / total))" -ge "$THRESHOLD_PCT" ]; then
        echo "vram_guard: GPU $gpu at $((100 * used / total))% — aborting the" \
             "check, not the campaign" >&2
        kill "$pid"
        wait "$pid" 2>/dev/null
        exit 75
    fi
    sleep "$POLL_S"
done
wait "$pid"
