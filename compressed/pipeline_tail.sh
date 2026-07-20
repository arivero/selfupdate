#!/usr/bin/env bash
# Readable scheduler evidence: progress bars are useful in worker logs but
# drown live supervision in token-heavy carriage-return updates.  Preserve
# warnings/errors and every non-progress line.
set -euo pipefail

log="${1:-runs/pipeline_sched.log}"
lines="${2:-120}"
[ -f "$log" ] || exit 0
tr '\r' '\n' < "$log" \
  | grep -v -E '^(Loading weights:|teacher forward:|[[:space:]]*[0-9]+%\|)' \
  | tail -n "$lines"
