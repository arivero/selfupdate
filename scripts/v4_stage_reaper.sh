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
# Lease lines are "pid" (stage = line index, the historical all-local form)
# or "pid stage" (multi-host launches supervise a local SUBSET whose line
# index is not the stage number — the log lookup must use the real stage).
PIDS=() STAGE_OF=()
idx=0
while read -r pid stg; do
  [[ -n "$pid" ]] || continue
  PIDS+=("$pid")
  STAGE_OF+=("${stg:-$idx}")
  idx=$((idx + 1))
done < "$LEASE"
while true; do
  alive=0 dead_bad=""
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    else
      # A stage that printed "run complete" exited cleanly - never a
      # failure, whatever older tracebacks remain in the appended log
      # (the 16:36 false kill: stage 0 finished normally while an OLD OOM
      # from a prior launch attempt was still inside the tail window).
      recent=$(tail -c 3000 "${LOGPREFIX}${STAGE_OF[$i]}.log" 2>/dev/null)
      if echo "$recent" | grep -q 'run complete'; then
        continue
      fi
      # Any other death is fatal to the set. A pattern list (the old
      # OOM|UTILIZATION GATE grep) is always one novel crash behind: the
      # 2026-07-17 relay-envelope RuntimeError matched nothing and left two
      # stages burning cards for hours on an unpublishable run.
      dead_bad="stage ${STAGE_OF[$i]} (pid $pid)"
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
