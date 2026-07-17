#!/usr/bin/env bash
# Watch one v4 PPP launch (its lease file of stage pids). If any stage dies
# while its recent log shows OOM or a UTILIZATION GATE abort, SIGTERM the
# surviving siblings: a PPP set with a dead stage cannot publish a complete
# run, so burning the remaining cards is waste (owner policy, 2026-07-17).
# Exits silently when every stage has ended.
#
# Usage: v4_stage_reaper.sh <lease-file> <stage-log-prefix>
#   e.g. v4_stage_reaper.sh runs/.v4-launch-<run>.pids runs/<run>_stage
set -euo pipefail
LEASE="${1:?lease file}"
LOGPREFIX="${2:?stage log prefix}"
mapfile -t PIDS < "$LEASE"
while true; do
  alive=0 dead_bad=""
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    else
      recent=$(tail -c 3000 "${LOGPREFIX}${i}.log" 2>/dev/null | grep -c -E 'OutOfMemoryError|UTILIZATION GATE' || true)
      if [ -n "$recent" ] && [ "$recent" != "0" ]; then
        dead_bad="stage $i (pid $pid)"
      fi
    fi
  done
  [ "$alive" = "0" ] && exit 0
  if [ -n "$dead_bad" ]; then
    echo "$(date '+%H:%M:%S') reaper: $dead_bad failed; terminating siblings" >&2
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 30
    for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    exit 1
  fi
  sleep 20
done
